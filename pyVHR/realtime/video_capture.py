import cv2
import logging
import threading
import time

logger = logging.getLogger(__name__)


class VideoCapture:
    """缓冲区清空型视频采集线程，支持自动重连。"""

    def __init__(self, name, sharedData, fps=None, sleep=False, resize=True, reconnect_interval=1.0):
        self.name = name
        self.sd = sharedData
        self.sleep = sleep
        self.resize = resize
        self.reconnect_interval = reconnect_interval
        self._requested_fps = fps
        self.fps = None
        self.cap = None
        self._cap_lock = threading.Lock()

        self._ensure_capture(initial=True)

        self.t = threading.Thread(target=self._reader, daemon=False)
        self.t.start()

    def _ensure_capture(self, initial=False):
        with self._cap_lock:
            if self.cap is not None:
                self.cap.release()
            self.cap = cv2.VideoCapture(self.name)
            if not self.cap.isOpened():
                if initial:
                    logger.warning("无法打开视频源 %s，等待重试。", self.name)
                else:
                    logger.warning("视频源 %s 断开，尝试重连。", self.name)
                self.cap.release()
                self.cap = None
                return False
            if self._requested_fps is not None:
                self.cap.set(cv2.CAP_PROP_FPS, self._requested_fps)
                self.fps = self._requested_fps
            else:
                reported_fps = self.cap.get(cv2.CAP_PROP_FPS)
                self.fps = reported_fps if reported_fps and reported_fps > 0 else None
            logger.info("已连接视频源 %s。", self.name)
            return True

    def _read_frame(self):
        with self._cap_lock:
            if self.cap is None or not self.cap.isOpened():
                return False, None
            return self.cap.read()

    def _resize_if_needed(self, frame):
        if not self.resize:
            return frame
        h, w = frame.shape[:2]
        if h == 480 and w == 640:
            return frame
        return cv2.resize(frame, (640, int(640 * h / w)), interpolation=cv2.INTER_NEAREST)

    def _reader(self):
        while True:
            if not self.sd.q_stop_cap.empty():  # 主动停止
                self.sd.q_stop_cap.get()
                self.sd.q_stop.put(0)
                self.sd.q_frames.put(0)
                break

            if self.cap is None or not self.cap.isOpened():
                if not self._ensure_capture():
                    time.sleep(self.reconnect_interval)
                    continue

            ret, frame = self._read_frame()
            if not ret or frame is None:
                time.sleep(self.reconnect_interval)
                self._ensure_capture()
                continue

            frame = self._resize_if_needed(frame)
            self.sd.q_frames.put(frame)

            if self.sleep and self.fps is not None:
                time.sleep(self.fps / 1000.0)

        with self._cap_lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
