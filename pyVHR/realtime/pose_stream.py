import asyncio
import json
import logging
import math
import queue
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Set

import cv2
import mediapipe as mp
import numpy as np
import websockets
from websockets.server import WebSocketServerProtocol

logger = logging.getLogger(__name__)


FACE_3D_MODEL = np.array(
    [
        (0.0, 0.0, 0.0),          # nose tip
        (0.0, -330.0, -65.0),     # chin
        (-225.0, 170.0, -135.0),  # left eye corner
        (225.0, 170.0, -135.0),   # right eye corner
        (-150.0, -150.0, -125.0), # left mouth corner
        (150.0, -150.0, -125.0),  # right mouth corner
    ],
    dtype=np.float64,
)

FACE_LANDMARK_INDEXES = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "left_mouth": 57,
    "right_mouth": 287,
}


def _rotation_matrix_to_euler_angles(matrix: np.ndarray) -> Dict[str, float]:
    """Convert rotation matrix to Euler angles (pitch, yaw, roll)."""
    sy = math.sqrt(matrix[0, 0] * matrix[0, 0] + matrix[1, 0] * matrix[1, 0])
    singular = sy < 1e-6

    if not singular:
        pitch = math.atan2(matrix[2, 1], matrix[2, 2])
        yaw = math.atan2(-matrix[2, 0], sy)
        roll = math.atan2(matrix[1, 0], matrix[0, 0])
    else:
        pitch = math.atan2(-matrix[1, 2], matrix[1, 1])
        yaw = math.atan2(-matrix[2, 0], sy)
        roll = 0.0

    return {
        "pitch": math.degrees(pitch),
        "yaw": math.degrees(yaw),
        "roll": math.degrees(roll),
    }


@dataclass
class PoseStreamConfig:
    host: str = "0.0.0.0"
    port: int = 8766
    publish_interval: float = 0.1


class PoseBroadcaster:
    """Consume frames from shared data, estimate pose/face orientation, and broadcast over WebSocket."""

    def __init__(self, shared_data, config: PoseStreamConfig):
        self._shared_data = shared_data
        self._config = config

        self._stop_event = threading.Event()
        self._pose_thread: Optional[threading.Thread] = None

        self._latest_payload: Optional[str] = None
        self._payload_lock = threading.Lock()

        self._ws_loop: Optional[asyncio.AbstractEventLoop] = None
        self._ws_thread: Optional[threading.Thread] = None
        self._clients: Set[WebSocketServerProtocol] = set()
        self._clients_lock = threading.Lock()

    # --------------------------------------------------------------------- #
    # Public control methods
    # --------------------------------------------------------------------- #
    def start(self):
        if self._pose_thread and self._pose_thread.is_alive():
            return

        logger.info(
            "启动姿态广播：ws://%s:%s，发布间隔 %.2fs",
            self._config.host,
            self._config.port,
            self._config.publish_interval,
        )

        self._stop_event.clear()
        self._pose_thread = threading.Thread(target=self._pose_worker, name="Pose-Worker", daemon=True)
        self._pose_thread.start()
        self._start_ws_server()

    def stop(self):
        if self._stop_event.is_set():
            return

        logger.info("停止姿态广播服务。")
        self._stop_event.set()

        if hasattr(self._shared_data, "q_pose_frames"):
            try:
                self._shared_data.q_pose_frames.put_nowait(0)
            except queue.Full:
                pass

        if self._ws_loop is not None:
            self._ws_loop.call_soon_threadsafe(lambda: None)

        if self._pose_thread is not None and self._pose_thread.is_alive():
            self._pose_thread.join(timeout=2.0)

        if self._ws_thread is not None and self._ws_thread.is_alive():
            self._ws_thread.join(timeout=2.0)

        with self._clients_lock:
            self._clients.clear()

        self._ws_loop = None
        self._ws_thread = None
        self._pose_thread = None

    # --------------------------------------------------------------------- #
    # Pose estimation worker
    # --------------------------------------------------------------------- #
    def _pose_worker(self):
        mp_holistic = mp.solutions.holistic

        with mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=True,
        ) as holistic:
            while not self._stop_event.is_set():
                try:
                    frame = self._shared_data.q_pose_frames.get(timeout=0.5)
                except queue.Empty:
                    continue

                if isinstance(frame, int):
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb_frame)

                if not results.pose_landmarks:
                    continue

                payload_dict = self._build_payload(results, frame.shape)
                payload_dict["timestamp"] = time.time()

                with self._payload_lock:
                    self._latest_payload = json.dumps(payload_dict)

    def _build_payload(self, results, frame_shape) -> Dict:
        height, width = frame_shape[0], frame_shape[1]
        pose_dict = self._extract_pose(results, width, height)
        face_orientation = self._extract_face_orientation(results, width, height)

        payload = {"pose": pose_dict}
        if face_orientation is not None:
            payload["face"] = {"orientation": face_orientation}
        return payload

    def _extract_pose(self, results, width: int, height: int) -> Dict[str, Dict]:
        pose_landmarks = results.pose_landmarks.landmark
        world_landmarks = results.pose_world_landmarks.landmark if results.pose_world_landmarks else None

        joints: Dict[str, Dict[str, float]] = {}
        for idx, landmark in enumerate(pose_landmarks):
            try:
                enum_name = mp.solutions.pose.PoseLandmark(idx).name.lower()
            except ValueError:
                enum_name = f"landmark_{idx}"

            joint_data = {
                "screen": {
                    "x": float(landmark.x),
                    "y": float(landmark.y),
                    "z": float(landmark.z),
                },
                "visibility": float(landmark.visibility),
            }

            if world_landmarks is not None:
                world_point = world_landmarks[idx]
                joint_data["world"] = {
                    "x": float(world_point.x),
                    "y": float(world_point.y),
                    "z": float(world_point.z),
                }

            joints[enum_name] = joint_data

        return {"joints": joints}

    def _extract_face_orientation(self, results, width: int, height: int) -> Optional[Dict[str, float]]:
        if not results.face_landmarks:
            return None

        landmarks = results.face_landmarks.landmark
        try:
            image_points = np.array(
                [
                    (landmarks[FACE_LANDMARK_INDEXES["nose_tip"]].x * width,
                     landmarks[FACE_LANDMARK_INDEXES["nose_tip"]].y * height),
                    (landmarks[FACE_LANDMARK_INDEXES["chin"]].x * width,
                     landmarks[FACE_LANDMARK_INDEXES["chin"]].y * height),
                    (landmarks[FACE_LANDMARK_INDEXES["left_eye_outer"]].x * width,
                     landmarks[FACE_LANDMARK_INDEXES["left_eye_outer"]].y * height),
                    (landmarks[FACE_LANDMARK_INDEXES["right_eye_outer"]].x * width,
                     landmarks[FACE_LANDMARK_INDEXES["right_eye_outer"]].y * height),
                    (landmarks[FACE_LANDMARK_INDEXES["left_mouth"]].x * width,
                     landmarks[FACE_LANDMARK_INDEXES["left_mouth"]].y * height),
                    (landmarks[FACE_LANDMARK_INDEXES["right_mouth"]].x * width,
                     landmarks[FACE_LANDMARK_INDEXES["right_mouth"]].y * height),
                ],
                dtype=np.float64,
            )
        except IndexError:
            logger.debug("面部关键点不足，无法估计朝向。")
            return None

        focal_length = width
        camera_matrix = np.array(
            [
                [focal_length, 0, width / 2.0],
                [0, focal_length, height / 2.0],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

        success, rotation_vector, _ = cv2.solvePnP(
            FACE_3D_MODEL,
            image_points,
            camera_matrix,
            np.zeros((4, 1), dtype=np.float64),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            logger.debug("solvePnP 失败，无法估计面部姿态。")
            return None

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        return _rotation_matrix_to_euler_angles(rotation_matrix)

    # --------------------------------------------------------------------- #
    # WebSocket server
    # --------------------------------------------------------------------- #
    def _start_ws_server(self):
        if self._ws_thread and self._ws_thread.is_alive():
            return

        self._ws_loop = asyncio.new_event_loop()

        def runner():
            asyncio.set_event_loop(self._ws_loop)
            self._ws_loop.run_until_complete(self._ws_main())

        self._ws_thread = threading.Thread(target=runner, name="Pose-WebSocket", daemon=True)
        self._ws_thread.start()

    async def _ws_main(self):
        async with websockets.serve(self._handle_client, self._config.host, self._config.port):
            logger.info("姿态 WebSocket 服务已启动：ws://%s:%s", self._config.host, self._config.port)
            await self._broadcast_loop()
        await self._close_clients()
        logger.info("姿态 WebSocket 服务已关闭。")

    async def _handle_client(self, websocket: WebSocketServerProtocol):
        with self._clients_lock:
            self._clients.add(websocket)
            count = len(self._clients)
        logger.info("姿态客户端连接，总数 %s", count)
        try:
            await websocket.wait_closed()
        finally:
            with self._clients_lock:
                self._clients.discard(websocket)
                count = len(self._clients)
            logger.info("姿态客户端离线，总数 %s", count)

    async def _broadcast_loop(self):
        while not self._stop_event.is_set():
            await asyncio.sleep(self._config.publish_interval)
            with self._payload_lock:
                payload = self._latest_payload
            if not payload:
                continue
            await self._broadcast(payload)

    async def _broadcast(self, message: str):
        with self._clients_lock:
            clients = list(self._clients)

        if not clients:
            return

        await asyncio.gather(*[self._safe_send(client, message) for client in clients], return_exceptions=True)

    async def _safe_send(self, client: WebSocketServerProtocol, message: str):
        try:
            await client.send(message)
        except Exception:
            logger.warning("向姿态客户端发送数据失败，关闭连接。")
            try:
                await client.close()
            finally:
                with self._clients_lock:
                    self._clients.discard(client)

    async def _close_clients(self):
        with self._clients_lock:
            clients = list(self._clients)
            self._clients.clear()

        if not clients:
            return

        await asyncio.gather(*[client.close(code=1001, reason="Pose broadcaster shutdown") for client in clients], return_exceptions=True)

