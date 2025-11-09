import asyncio
import json
import logging
import signal
import sys
import threading
import time
import queue
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set

import websockets
from websockets.server import WebSocketServerProtocol

from pyVHR.BVP.methods import cpu_CHROM
from pyVHR.realtime.VHRroutine import VHRroutine, SharedData, Params
from pyVHR.realtime.pose_stream import PoseBroadcaster, PoseStreamConfig

logger = logging.getLogger(__name__)


@dataclass
class ServiceConfig:
    camera_index: int = 0
    ws_host: str = "0.0.0.0"
    ws_port: int = 8765
    bpm_publish_interval: float = 1.0
    reconnect_interval: float = 1.0
    use_cuda: bool = False
    win_size: int = 6
    stride: int = 1
    log_level: int = logging.INFO
    method: Optional[Dict[str, Any]] = field(default=None)
    skin_threshold_low: int = 0
    skin_threshold_high: int = 255
    signal_threshold_low: int = 0
    signal_threshold_high: int = 255
    filter_threshold_low: int = 0
    filter_threshold_high: int = 255
    pose_ws_host: str = "0.0.0.0"
    pose_ws_port: int = 8766
    pose_publish_interval: float = 0.1


class RealtimeHRService:
    def __init__(self, config: Optional[ServiceConfig] = None):
        self.config = config or ServiceConfig()
        self._shared_data = SharedData()
        self._stop_event = threading.Event()
        self._params_backup: Dict[str, Any] = {}

        self._processor_thread: Optional[threading.Thread] = None
        self._consumer_thread: Optional[threading.Thread] = None
        self._cli_thread: Optional[threading.Thread] = None
        self._ws_thread: Optional[threading.Thread] = None

        self._ws_loop: Optional[asyncio.AbstractEventLoop] = None
        self._ws_clients: Set[WebSocketServerProtocol] = set()

        self._state_lock = threading.Lock()
        self._latest_bpm: Optional[float] = None
        self._latest_timestamp: Optional[float] = None
        self._clients_count: int = 0
        self._pose_broadcaster: Optional[PoseBroadcaster] = None
        self._shutdown_started = False

    # ---- Public API ----
    def start(self):
        logging.basicConfig(level=self.config.log_level, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
        logger.info("即将启动实时 rPPG 服务：摄像头 %s，WebSocket %s:%s", self.config.camera_index, self.config.ws_host, self.config.ws_port)

        self._configure_params()

        self._processor_thread = threading.Thread(target=self._run_vhr, name="rPPG-Processor", daemon=True)
        self._processor_thread.start()

        self._consumer_thread = threading.Thread(target=self._consume_bpm, name="BPM-Consumer", daemon=True)
        self._consumer_thread.start()

        self._cli_thread = threading.Thread(target=self._cli_loop, name="CLI-Status", daemon=True)
        self._cli_thread.start()

        self._start_websocket_server()
        self._start_pose_stream()

    def run(self):
        self.start()
        try:
            while not self._stop_event.is_set():
                time.sleep(0.5)
        except KeyboardInterrupt:
            logger.info("收到中断信号，准备停止服务。")
        finally:
            self.stop()

    def stop(self):
        if self._shutdown_started:
            return
        self._shutdown_started = True
        logger.info("正在停止实时 rPPG 服务……")
        self._stop_event.set()

        self._shared_data.q_stop_cap.put(0)
        self._shared_data.q_stop.put(0)

        if self._ws_loop is not None:
            self._ws_loop.call_soon_threadsafe(lambda: None)

        for thread in [self._processor_thread, self._consumer_thread, self._cli_thread]:
            if thread is not None and thread.is_alive():
                thread.join(timeout=2.0)

        if self._ws_thread is not None and self._ws_thread.is_alive():
            self._ws_thread.join(timeout=2.0)

        if self._pose_broadcaster is not None:
            self._pose_broadcaster.stop()
            self._pose_broadcaster = None

        self._restore_params()
        logger.info("实时 rPPG 服务已停止。")

    # ---- Internal helpers ----
    def _configure_params(self):
        attrs = [
            "videoFileName",
            "cuda",
            "method",
            "camera_reconnect_interval",
            "winSize",
            "stride",
            "tot_sec",
            "fps_fixed",
            "visualize_skin",
            "visualize_patches",
            "visualize_landmarks",
            "visualize_landmarks_number",
            "skin_color_low_threshold",
            "skin_color_high_threshold",
            "sig_color_low_threshold",
            "sig_color_high_threshold",
            "color_low_threshold",
            "color_high_threshold",
        ]
        self._params_backup = {name: getattr(Params, name) for name in attrs}

        Params.videoFileName = self.config.camera_index
        Params.cuda = self.config.use_cuda
        Params.camera_reconnect_interval = self.config.reconnect_interval
        Params.winSize = self.config.win_size
        Params.stride = self.config.stride
        Params.tot_sec = 0
        Params.fps_fixed = None
        Params.visualize_skin = False
        Params.visualize_patches = False
        Params.visualize_landmarks = False
        Params.visualize_landmarks_number = False

        Params.skin_color_low_threshold = self.config.skin_threshold_low
        Params.skin_color_high_threshold = self.config.skin_threshold_high
        Params.sig_color_low_threshold = self.config.signal_threshold_low
        Params.sig_color_high_threshold = self.config.signal_threshold_high
        Params.color_low_threshold = self.config.filter_threshold_low
        Params.color_high_threshold = self.config.filter_threshold_high

        if not Params.cuda:
            Params.method = self.config.method or {"method_func": cpu_CHROM, "device_type": "cpu", "params": {}}
        else:
            Params.method = self.config.method or Params.method

    def _restore_params(self):
        if not self._params_backup:
            return
        for name, value in self._params_backup.items():
            setattr(Params, name, value)

    def _run_vhr(self):
        try:
            VHRroutine(self._shared_data)
        except Exception:
            logger.exception("rPPG 处理线程发生异常，将终止服务。")
            self._stop_event.set()

    def _consume_bpm(self):
        while not self._stop_event.is_set():
            try:
                bpm = self._shared_data.q_bpm.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                bpm_value = float(bpm)
            except (TypeError, ValueError):
                bpm_value = None

            with self._state_lock:
                self._latest_bpm = bpm_value
                self._latest_timestamp = time.time()

        logger.debug("BPM 消费线程已退出。")

    def _cli_loop(self):
        while not self._stop_event.is_set():
            with self._state_lock:
                bpm = self._latest_bpm
                ts = self._latest_timestamp
                clients = self._clients_count
            if bpm is None or ts is None:
                line = f"BPM: --.- | 客户端: {clients} | 状态: 等待数据"
            else:
                latency = time.time() - ts
                line = f"BPM: {bpm:6.2f} | 客户端: {clients} | 延迟: {latency:4.1f}s"
            sys.stdout.write("\r" + line.ljust(80))
            sys.stdout.flush()
            time.sleep(self.config.bpm_publish_interval)
        sys.stdout.write("\n")
        sys.stdout.flush()

    # ---- WebSocket handling ----
    def _start_websocket_server(self):
        self._ws_loop = asyncio.new_event_loop()

        def runner():
            asyncio.set_event_loop(self._ws_loop)
            self._ws_loop.run_until_complete(self._ws_main())

        self._ws_thread = threading.Thread(target=runner, name="WebSocket-Server", daemon=True)
        self._ws_thread.start()

    def _start_pose_stream(self):
        config = PoseStreamConfig(
            host=self.config.pose_ws_host or self.config.ws_host,
            port=self.config.pose_ws_port,
            publish_interval=self.config.pose_publish_interval,
        )
        self._pose_broadcaster = PoseBroadcaster(self._shared_data, config)
        self._pose_broadcaster.start()

    async def _ws_main(self):
        async with websockets.serve(self._handle_ws_client, self.config.ws_host, self.config.ws_port):
            logger.info("WebSocket 服务已启动：ws://%s:%s", self.config.ws_host, self.config.ws_port)
            await self._broadcast_loop()
        await self._close_all_clients()
        logger.info("WebSocket 服务已关闭。")

    async def _handle_ws_client(self, websocket):
        self._ws_clients.add(websocket)
        self._update_client_count()
        logger.info("客户端已连接，总数 %s", len(self._ws_clients))
        try:
            await websocket.wait_closed()
        finally:
            self._ws_clients.discard(websocket)
            self._update_client_count()
            logger.info("客户端断开，总数 %s", len(self._ws_clients))

    async def _broadcast_loop(self):
        while not self._stop_event.is_set():
            await asyncio.sleep(self.config.bpm_publish_interval)
            payload = self._build_payload()
            if payload is None:
                continue
            await self._broadcast(payload)

    async def _broadcast(self, message: str):
        if not self._ws_clients:
            return
        coros = [self._safe_send(client, message) for client in list(self._ws_clients)]
        await asyncio.gather(*coros, return_exceptions=True)

    async def _safe_send(self, client, message: str):
        try:
            await client.send(message)
        except Exception:
            logger.warning("向客户端发送数据失败，连接将关闭。")
            await client.close()
            self._ws_clients.discard(client)
            self._update_client_count()

    async def _close_all_clients(self):
        if not self._ws_clients:
            return
        await asyncio.gather(*[client.close(code=1001, reason="服务器关闭") for client in list(self._ws_clients)], return_exceptions=True)
        self._ws_clients.clear()
        self._update_client_count()

    def _build_payload(self) -> Optional[str]:
        with self._state_lock:
            if self._latest_bpm is None or self._latest_timestamp is None:
                return None
            payload = {
                "bpm": self._latest_bpm,
                "timestamp": self._latest_timestamp,
            }
        return json.dumps(payload)

    def _update_client_count(self):
        with self._state_lock:
            self._clients_count = len(self._ws_clients)


def main():
    service = RealtimeHRService()

    def handle_signal(signum, frame):
        logger.info("收到系统信号 %s，停止服务。", signum)
        service.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    service.run()


if __name__ == "__main__":
    main()
