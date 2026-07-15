"""
╔══════════════════════════════════════════════════════════════════╗
║          👁️  GESTURE MIND — Body Language OS Controller          ║
║     Real-time gesture, expression, age & gender recognition     ║
╚══════════════════════════════════════════════════════════════════╝

Fitur:
  ✦ Klasifikasi gesture pakai model resmi MediaPipe GestureRecognizer
    (Google AI Edge) -- fallback otomatis ke heuristik landmark manual
    kalau model gagal dimuat/diunduh
  ✦ Scroll otomatis (telapak terbuka naik/turun)
  ✦ Tutup aplikasi (kepalan / jempol ke bawah)
  ✦ Volume up/down (jari telunjuk atas/bawah)
  ✦ Screenshot (V-sign / Victory)
  ✦ Minimize semua window (gesture ILoveYou)
  ✦ Lock screen (dua tangan terlihat kamera)
  ✦ Pause/Play media (OK sign)
  ✦ Analisis ekspresi wajah (happy, sad, angry, surprised, neutral, fear, disgust)
  ✦ Tebak usia & gender
  ✦ Live HUD overlay terminal-style, fullscreen (toggle tombol F)

Dependensi (install sekali):
    pip install -r requirements.txt
    (model GestureRecognizer ~8MB diunduh otomatis dari storage.googleapis.com
     saat run pertama, ke folder models/)
"""

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import threading
import os
import sys
import math
import queue
from datetime import datetime
from collections import deque

# Terminal Windows default-nya pakai codepage cp1252, yang crash begitu ada
# library (DeepFace, dll) nge-print emoji/unicode saat download model.
if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# ─── Optional: DeepFace (bisa dinonaktifkan jika tidak terinstall) ───
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("[WARN] DeepFace tidak terinstall. Analisis wajah dinonaktifkan.")
    print("       Install dengan: pip install deepface")

# ─── Konfigurasi PyAutoGUI ───
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.01

# ════════════════════════════════════════════════════════════════════
#  KONFIGURASI & KONSTANTA
# ════════════════════════════════════════════════════════════════════

WIN_W, WIN_H = 1280, 720
PANEL_W = 340          # lebar panel kanan (HUD)
CAM_W   = WIN_W - PANEL_W
CAM_H   = WIN_H

# Cooldown antar gesture (detik)
COOLDOWN = {
    "scroll"      : 0.05,
    "close_app"   : 2.0,
    "volume"      : 0.3,
    "screenshot"  : 2.0,
    "minimize"    : 1.5,
    "lock_screen" : 3.0,
    "face_analyze": 3.0,
    "thumbs_up"   : 2.0,
}

# Warna tema (BGR untuk OpenCV)
C = {
    "bg"          : (15, 15, 20),
    "panel"       : (20, 18, 28),
    "accent"      : (0, 220, 160),       # cyan-hijau
    "accent2"     : (180, 80, 255),      # ungu
    "warn"        : (0, 150, 255),       # oranye
    "danger"      : (60, 60, 230),       # merah
    "text"        : (220, 220, 230),
    "text_dim"    : (100, 100, 115),
    "white"       : (255, 255, 255),
    "bar_bg"      : (35, 33, 45),
    "happy"       : (50, 205, 120),
    "sad"         : (200, 130, 80),
    "angry"       : (60, 60, 220),
    "surprised"   : (0, 200, 255),
    "neutral"     : (160, 160, 160),
    "fear"        : (120, 80, 200),
    "disgust"     : (80, 170, 80),
}

EMOTION_COLORS = {
    "happy"    : C["happy"],
    "sad"      : C["sad"],
    "angry"    : C["angry"],
    "surprised": C["surprised"],
    "neutral"  : C["neutral"],
    "fear"     : C["fear"],
    "disgust"  : C["disgust"],
}

EMOTION_EMOJI = {
    "happy"    : "HAPPY",
    "sad"      : "SAD",
    "angry"    : "ANGRY",
    "surprised": "SURPRISED",
    "neutral"  : "NEUTRAL",
    "fear"     : "FEAR",
    "disgust"  : "DISGUST",
}

# ── Model resmi MediaPipe GestureRecognizer (Google AI Edge) ─────────
# Sumber: https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer
# Model AI terlatih untuk 6 gesture baku (Closed_Fist, Open_Palm, Victory,
# Thumb_Up, Thumb_Down, Pointing_Up, ILoveYou) -- jauh lebih akurat
# dibanding heuristik landmark manual, dipakai sebagai jalur utama.
# Kalau gagal diunduh/dimuat, otomatis fallback ke heuristik manual lama.
MODEL_DIR         = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
GESTURE_MODEL_PATH = os.path.join(MODEL_DIR, "gesture_recognizer.task")
GESTURE_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/latest/gesture_recognizer.task"
)

# Kategori resmi -> (nama gesture internal, label aksi)
TASK_GESTURE_MAP = {
    "Closed_Fist" : ("close_app",   "CLOSE APP"),
    "Victory"     : ("screenshot",  "SCREENSHOT"),
    "Thumb_Up"    : ("thumbs_up",   "THUMBS UP"),
    "Thumb_Down"  : ("thumbs_down", "THUMBS DOWN -> CLOSE"),
    "Pointing_Up" : ("vol_up",      "VOLUME UP"),
    "ILoveYou"    : ("minimize",    "MINIMIZE WINDOW"),
}


def _ensure_gesture_model():
    """Unduh model GestureRecognizer resmi sekali saja kalau belum ada di disk."""
    if os.path.exists(GESTURE_MODEL_PATH):
        return True
    try:
        import urllib.request
        os.makedirs(MODEL_DIR, exist_ok=True)
        print("[INFO] Mengunduh model GestureRecognizer resmi dari Google (~8MB)...")
        urllib.request.urlretrieve(GESTURE_MODEL_URL, GESTURE_MODEL_PATH)
        print("[INFO] Model GestureRecognizer berhasil diunduh.")
        return True
    except Exception as e:
        print(f"[WARN] Gagal mengunduh model GestureRecognizer: {e}")
        print("       Fallback ke deteksi gesture manual (heuristik landmark).")
        return False


# ════════════════════════════════════════════════════════════════════
#  KELAS UTAMA
# ════════════════════════════════════════════════════════════════════

class GestureMind:
    def __init__(self):
        # ── GestureRecognizer resmi (Google AI Edge) -- opsional ────
        # PENTING: harus dibuat SEBELUM mp.solutions.* (legacy API) --
        # kalau dibalik, mediapipe di Windows salah cache resource-root
        # internalnya jadi folder site-packages dan gagal buka model
        # Tasks API manapun sesudahnya (bug urutan inisialisasi mediapipe).
        self.gesture_recognizer = None
        if _ensure_gesture_model():
            try:
                BaseOptions             = mp.tasks.BaseOptions
                GestureRecognizer       = mp.tasks.vision.GestureRecognizer
                GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
                VisionRunningMode       = mp.tasks.vision.RunningMode
                options = GestureRecognizerOptions(
                    # mediapipe di Windows juga salah resolve path absolut yang
                    # pakai backslash (dianggap relatif) -- paksa forward-slash.
                    base_options=BaseOptions(model_asset_path=GESTURE_MODEL_PATH.replace("\\", "/")),
                    running_mode=VisionRunningMode.IMAGE,
                    num_hands=2,
                )
                self.gesture_recognizer = GestureRecognizer.create_from_options(options)
            except Exception as e:
                print(f"[WARN] GestureRecognizer gagal diinisialisasi: {e}")
                print("       Fallback ke deteksi gesture manual (heuristik landmark).")
                self.gesture_recognizer = None

        # ── MediaPipe (legacy Solutions API) ─────────────────────────
        self.mp_hands   = mp.solutions.hands
        self.mp_face    = mp.solutions.face_detection
        self.mp_pose    = mp.solutions.pose
        self.mp_draw    = mp.solutions.drawing_utils
        self.mp_draw_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6
        )
        self.face_det = self.mp_face.FaceDetection(
            model_selection=0, min_detection_confidence=0.7
        )
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # ── State ──────────────────────────────────────────────────
        self.last_action        = {k: 0 for k in COOLDOWN}
        self.gesture_history    = deque(maxlen=60)
        self.event_log          = deque(maxlen=10)
        self.current_gesture    = "—"
        self.current_emotion    = "—"
        self.current_age        = "—"
        self.current_gender     = "—"
        self.emotion_scores     = {}
        self.face_bbox          = None
        self.scroll_dir         = 0           # -1 up, 0 none, +1 down
        self.gesture_conf       = 0.0
        self.fps_counter        = deque(maxlen=30)
        self.active_gesture     = None
        self.gesture_active_ts  = 0
        self.face_analyze_q     = queue.Queue(maxsize=1)
        self.face_result_q      = queue.Queue(maxsize=1)
        self.analyze_thread     = None
        self.running            = True
        self.frame_count        = 0
        self.wrist_history      = deque(maxlen=10)  # untuk deteksi wave
        self.prev_wrist_x       = None
        self.wave_count         = 0
        self.wave_ts            = 0

        # ── Thread analisis wajah ──────────────────────────────────
        if DEEPFACE_AVAILABLE:
            self.analyze_thread = threading.Thread(
                target=self._face_analyze_worker, daemon=True
            )
            self.analyze_thread.start()

    # ──────────────────────────────────────────────────────────────
    #  UTILITAS TANGAN
    # ──────────────────────────────────────────────────────────────

    def _landmarks_to_arr(self, hand_landmarks, w, h):
        # API lama (mp.solutions) bungkus landmark di .landmark; Tasks API
        # (GestureRecognizer) sudah kasih list mentah -- dukung keduanya.
        landmarks = hand_landmarks.landmark if hasattr(hand_landmarks, "landmark") else hand_landmarks
        pts = []
        for lm in landmarks:
            pts.append((int(lm.x * w), int(lm.y * h), lm.z))
        return pts

    def _finger_states(self, pts):
        """Kembalikan [jempol, telunjuk, tengah, manis, kelingking] = True jika terbuka"""
        tips  = [4, 8, 12, 16, 20]
        dips  = [3, 7, 11, 15, 19]
        mcp   = [2, 5, 9, 13, 17]
        open_ = []

        # Jempol: pakai jarak ke pangkal kelingking (17), bukan cuma delta-x,
        # supaya tetap akurat walau tangan miring/berputar terhadap kamera.
        pinky_mcp = pts[17]
        dist_tip = math.dist(pts[4][:2], pinky_mcp[:2])
        dist_ip  = math.dist(pts[3][:2], pinky_mcp[:2])
        thumb_open = dist_tip > dist_ip * 1.1
        open_.append(thumb_open)

        for i in range(1, 5):
            open_.append(pts[tips[i]][1] < pts[dips[i]][1])

        return open_

    def _count_fingers(self, pts):
        states = self._finger_states(pts)
        return sum(states)

    def _palm_center(self, pts):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (int(np.mean(xs)), int(np.mean(ys)))

    def _hand_openness(self, pts):
        """0.0 = kepalan, 1.0 = terbuka penuh"""
        states = self._finger_states(pts)
        return sum(states[1:]) / 4.0   # 4 jari non-jempol

    # ──────────────────────────────────────────────────────────────
    #  DETEKSI GESTURE
    # ──────────────────────────────────────────────────────────────

    def _detect_gesture(self, all_hands_pts, frame_h, frame_w, task_result=None):
        """
        Kembalikan (gesture_name, confidence, extra_data).
        Prioritas: model resmi MediaPipe GestureRecognizer (kalau tersedia
        dan berhasil mendeteksi tangan), fallback ke heuristik landmark
        manual kalau modelnya tidak aktif atau tidak melihat tangan.
        """
        if task_result is not None and task_result.hand_landmarks:
            return self._detect_gesture_task(task_result, frame_h, frame_w)
        return self._detect_gesture_heuristic(all_hands_pts, frame_h, frame_w)

    def _detect_gesture_task(self, task_result, frame_h, frame_w):
        """Klasifikasi gesture pakai model resmi google-ai-edge/mediapipe GestureRecognizer."""
        num_hands = len(task_result.hand_landmarks)
        pts    = self._landmarks_to_arr(task_result.hand_landmarks[0], frame_w, frame_h)
        states = self._finger_states(pts)
        palm   = self._palm_center(pts)

        # ── Dua tangan → Lock Screen (prioritas tertinggi) ────────
        if num_hands >= 2:
            return ("lock_screen", 0.85, {"action": "LOCK SCREEN"})

        # ── OK Sign (jempol-telunjuk nempel) -- tidak ada di kategori
        #    baku Google, tetap pakai deteksi jarak manual ──────────
        thumb_tip, index_tip = pts[4], pts[8]
        if math.dist(thumb_tip[:2], index_tip[:2]) < 30 and not states[2] and not states[3]:
            return ("ok_sign", 0.88, {"action": "PAUSE/PLAY"})

        category, score = None, 0.0
        if task_result.gestures and task_result.gestures[0]:
            top = task_result.gestures[0][0]
            category, score = top.category_name, top.score

        if category in TASK_GESTURE_MAP:
            name, label = TASK_GESTURE_MAP[category]
            return (name, score, {"action": label})

        if category == "Open_Palm":
            rel_y = pts[0][1] / frame_h
            if rel_y < 0.35:
                return ("scroll_up", score, {"action": "SCROLL UP", "dir": -1})
            elif rel_y > 0.65:
                return ("scroll_down", score, {"action": "SCROLL DOWN", "dir": 1})

        # ── Telunjuk ke bawah -- tidak ada kategori baku Google
        #    untuk "Pointing_Down", tetap pakai deteksi manual ──────
        index_down = states[1] and not states[2] and not states[3] and not states[4]
        if index_down and pts[8][1] > palm[1] + 30:
            return ("vol_down", 0.8, {"action": "VOLUME DOWN"})

        return ("unknown", score, {})

    def _detect_gesture_heuristic(self, all_hands_pts, frame_h, frame_w):
        """
        Fallback lama: klasifikasi gesture murni dari landmark tangan,
        dipakai kalau GestureRecognizer resmi gagal dimuat.
        Kembalikan (gesture_name, confidence, extra_data)
        """
        if not all_hands_pts:
            return ("idle", 0.0, {})

        pts = all_hands_pts[0]
        states = self._finger_states(pts)
        n_open = sum(states)
        palm   = self._palm_center(pts)

        # ── Tutup Aplikasi: kepalan + gerak ke bawah cepat ────────
        is_fist = n_open == 0
        if is_fist:
            return ("close_app", 0.9, {"action": "CLOSE APP"})

        # ── Peace / V-Sign → Screenshot ───────────────────────────
        peace = (not states[0] and states[1] and states[2]
                 and not states[3] and not states[4])
        if peace:
            return ("screenshot", 0.92, {"action": "SCREENSHOT ✌"})

        # ── OK Sign → Jeda / Play-Pause ───────────────────────────
        thumb_tip  = pts[4]
        index_tip  = pts[8]
        dist_ok = math.dist(thumb_tip[:2], index_tip[:2])
        ok_sign = dist_ok < 30 and not states[2] and not states[3]
        if ok_sign:
            return ("ok_sign", 0.88, {"action": "PAUSE/PLAY 👌"})

        # ── Satu Jari Telunjuk ke Atas → Volume Up ────────────────
        index_up = (states[1] and not states[2] and not states[3]
                    and not states[4])
        if index_up:
            tip = pts[8]
            if tip[1] < palm[1]:
                return ("vol_up", 0.87, {"action": "VOLUME UP ☝"})

        # ── Satu Jari Telunjuk ke Bawah → Volume Down ─────────────
        index_down = (states[1] and not states[2] and not states[3]
                      and not states[4])
        if index_down:
            tip = pts[8]
            if tip[1] > palm[1] + 30:
                return ("vol_down", 0.83, {"action": "VOLUME DOWN 👇"})

        # ── Tangan Terbuka (5 jari) + gerak naik → Scroll Up ──────
        all_open = n_open >= 4
        if all_open:
            wrist_y = pts[0][1]
            rel_y   = wrist_y / frame_h
            if rel_y < 0.35:
                return ("scroll_up", 0.90, {"action": "SCROLL UP ↑", "dir": -1})
            elif rel_y > 0.65:
                return ("scroll_down", 0.90, {"action": "SCROLL DOWN ↓", "dir": 1})

        # ── Jempol ke Atas / ke Bawah ─────────────────────────────
        if states[0] and n_open == 1:
            if pts[4][1] < pts[2][1]:
                return ("thumbs_up", 0.91, {"action": "THUMBS UP 👍"})
            else:
                return ("thumbs_down", 0.88, {"action": "THUMBS DOWN 👎 → CLOSE"})

        # ── Dua Tangan → Lock Screen ───────────────────────────────
        if len(all_hands_pts) >= 2:
            return ("lock_screen", 0.85, {"action": "LOCK SCREEN 🔒"})

        return ("unknown", 0.5, {})

    # ──────────────────────────────────────────────────────────────
    #  EKSEKUSI PERINTAH OS
    # ──────────────────────────────────────────────────────────────

    def _execute(self, gesture, extra):
        now = time.time()

        def can(key):
            return now - self.last_action.get(key, 0) >= COOLDOWN.get(key, 1.0)

        if gesture == "scroll_up" and can("scroll"):
            pyautogui.scroll(5)
            self.last_action["scroll"] = now
            self._log("^ Scroll UP")

        elif gesture == "scroll_down" and can("scroll"):
            pyautogui.scroll(-5)
            self.last_action["scroll"] = now
            self._log("v Scroll DOWN")

        elif gesture == "close_app" and can("close_app"):
            pyautogui.hotkey("alt", "F4")
            self.last_action["close_app"] = now
            self._log("X Close App (Alt+F4)")

        elif gesture == "vol_up" and can("volume"):
            pyautogui.press("volumeup")
            self.last_action["volume"] = now
            self._log("Volume UP")

        elif gesture == "vol_down" and can("volume"):
            pyautogui.press("volumedown")
            self.last_action["volume"] = now
            self._log("Volume DOWN")

        elif gesture == "screenshot" and can("screenshot"):
            ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
            fn  = f"gesture_screenshot_{ts}.png"
            pyautogui.screenshot(fn)
            self.last_action["screenshot"] = now
            self._log(f"Screenshot: {fn}")

        elif gesture == "thumbs_down" and can("close_app"):
            pyautogui.hotkey("alt", "F4")
            self.last_action["close_app"] = now
            self._log("Close (Thumbs Down)")

        elif gesture == "lock_screen" and can("lock_screen"):
            if sys.platform == "win32":
                import ctypes; ctypes.windll.user32.LockWorkStation()
            elif sys.platform == "darwin":
                os.system(
                    "osascript -e 'tell application \"System Events\" to keystroke \"q\" "
                    "using {command down, control down}'"
                )
            else:
                os.system("loginctl lock-session")
            self.last_action["lock_screen"] = now
            self._log("Lock Screen!")

        elif gesture == "ok_sign" and can("screenshot"):
            pyautogui.press("playpause")
            self.last_action["screenshot"] = now
            self._log("Play/Pause")

        elif gesture == "thumbs_up" and can("thumbs_up"):
            self.last_action["thumbs_up"] = now
            self._log("Nice! (no action)")

        elif gesture == "minimize" and can("minimize"):
            pyautogui.hotkey("win", "m")
            self.last_action["minimize"] = now
            self._log("Minimize All Windows")

    def _log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self.event_log.appendleft(f"[{ts}] {msg}")

    # ──────────────────────────────────────────────────────────────
    #  ANALISIS WAJAH (Thread Terpisah)
    # ──────────────────────────────────────────────────────────────

    def _face_analyze_worker(self):
        while self.running:
            try:
                face_img = self.face_analyze_q.get(timeout=1.0)
                if face_img is None or face_img.size == 0:
                    continue
                try:
                    result = DeepFace.analyze(
                        face_img,
                        actions=["emotion", "age", "gender"],
                        enforce_detection=False,
                        silent=True
                    )
                    if isinstance(result, list):
                        result = result[0]
                    if not self.face_result_q.full():
                        self.face_result_q.put(result)
                except Exception:
                    pass
            except queue.Empty:
                pass

    def _try_analyze_face(self, frame, bbox):
        """Kirim crop wajah ke thread analisis jika cooldown habis"""
        if not DEEPFACE_AVAILABLE:
            return
        now = time.time()
        if now - self.last_action["face_analyze"] < COOLDOWN["face_analyze"]:
            return
        if self.face_analyze_q.full():
            return
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        pad = 20
        x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)
        crop = frame[y1:y2, x1:x2]
        if crop.size > 0:
            self.face_analyze_q.put(crop.copy())
            self.last_action["face_analyze"] = now

    def _poll_face_result(self):
        if not DEEPFACE_AVAILABLE:
            return
        try:
            result = self.face_result_q.get_nowait()
            emotions = result.get("emotion", {})
            dom_emo  = result.get("dominant_emotion", "neutral")
            age      = result.get("age", "—")
            gender_d = result.get("gender", {})
            if isinstance(gender_d, dict):
                gender = max(gender_d, key=gender_d.get) if gender_d else "—"
            else:
                gender = str(gender_d)

            self.current_emotion  = dom_emo
            self.current_age      = str(age) if isinstance(age, (int, float)) else age
            self.current_gender   = gender
            self.emotion_scores   = emotions
            self._log(f"{dom_emo.upper()} | Age~{self.current_age} | {gender}")
        except queue.Empty:
            pass

    # ──────────────────────────────────────────────────────────────
    #  RENDER HUD
    # ──────────────────────────────────────────────────────────────

    def _draw_panel(self, panel):
        """Gambar panel kanan: info gesture, emosi, log"""
        panel[:] = np.array(C["panel"], dtype=np.uint8)

        def txt(text, x, y, color=C["text"], scale=0.45, thick=1):
            cv2.putText(panel, text, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

        def line(y, col=C["text_dim"]):
            cv2.line(panel, (10, y), (PANEL_W - 10, y), col, 1)

        def bar(label, value, y, max_val=100, color=C["accent"]):
            txt(label, 10, y, C["text_dim"], 0.37)
            bw = PANEL_W - 90
            cv2.rectangle(panel, (85, y - 10), (85 + bw, y + 3), C["bar_bg"], -1)
            fill = int(bw * min(value / max_val, 1.0))
            if fill > 0:
                cv2.rectangle(panel, (85, y - 10), (85 + fill, y + 3), color, -1)
            txt(f"{value:.0f}%", 85 + bw + 5, y, C["text_dim"], 0.36)

        y = 30
        # ── Header ──────────────────────────────────────────────
        cv2.rectangle(panel, (0, 0), (PANEL_W, 50), (25, 20, 40), -1)
        txt("GESTURE MIND", 10, y, C["accent"], 0.65, 2)
        txt("v1.0", PANEL_W - 45, y, C["text_dim"], 0.4)
        y += 22
        fps = len(self.fps_counter) / max(
            self.fps_counter[-1] - self.fps_counter[0], 0.001
        ) if len(self.fps_counter) > 1 else 0
        txt(f"FPS: {fps:.1f}  |  Frame #{self.frame_count}", 10, y, C["text_dim"], 0.37)

        y += 20; line(y)

        # ── Gesture saat ini ────────────────────────────────────
        y += 18
        txt("GESTURE DETECTED", 10, y, C["accent2"], 0.42, 1)
        y += 20
        gname = self.current_gesture.upper() if self.current_gesture else "—"
        g_col = C["accent"] if self.current_gesture not in ("idle", "—", "unknown") else C["text_dim"]
        txt(gname, 10, y, g_col, 0.65, 2)
        y += 18
        bar("Conf", self.gesture_conf * 100, y, color=C["accent"])

        y += 20; line(y)

        # ── Emosi ───────────────────────────────────────────────
        y += 18
        txt("FACE ANALYSIS", 10, y, C["accent2"], 0.42, 1)
        y += 18
        emo_label = EMOTION_EMOJI.get(self.current_emotion, self.current_emotion.upper())
        e_col = EMOTION_COLORS.get(self.current_emotion, C["text"])
        txt(emo_label, 10, y, e_col, 0.55, 2)
        y += 18

        # Emotion bars
        for emo in ["happy", "sad", "angry", "surprised", "fear", "neutral"]:
            score = self.emotion_scores.get(emo, 0.0)
            bar(emo[:7], score, y, color=EMOTION_COLORS.get(emo, C["accent"]))
            y += 18

        y += 4; line(y)

        # ── Usia & Gender ────────────────────────────────────────
        y += 18
        txt("IDENTITY ESTIMATE", 10, y, C["accent2"], 0.42, 1)
        y += 20
        ag_col = C["warn"] if self.current_age != "—" else C["text_dim"]
        txt(f"Age  : {self.current_age}", 10, y, ag_col, 0.52, 1)
        y += 18
        gnd = self.current_gender.upper() if self.current_gender != "—" else "—"
        gnd_col = (200, 120, 255) if gnd == "WOMAN" else (120, 180, 255) if gnd == "MAN" else C["text_dim"]
        txt(f"Gender: {gnd}", 10, y, gnd_col, 0.52, 1)

        y += 14; line(y)

        # ── Event Log ───────────────────────────────────────────
        y += 16
        txt("EVENT LOG", 10, y, C["accent2"], 0.42, 1)
        y += 16
        for i, entry in enumerate(self.event_log):
            col = C["accent"] if i == 0 else C["text_dim"]
            txt(entry[:42], 10, y, col, 0.35)
            y += 14
            if y > WIN_H - 30:
                break

        y = WIN_H - 70; line(y)

        # ── Panduan singkat ─────────────────────────────────────
        y += 14
        txt("GESTURE GUIDE", 10, y, C["accent2"], 0.38)
        y += 13
        guides = [
            "Fist  -> Close App",
            "Peace -> Screenshot",
            "Index Up/Dn -> Vol",
            "Palm High/Low -> Scroll",
            "Thumb Up/Dn -> Like/Close",
            "ILoveYou -> Minimize",
            "OK Sign -> Pause/Play",
            "Both Hands -> Lock",
        ]
        for g in guides:
            txt(g, 10, y, C["text_dim"], 0.33)
            y += 12

        return panel

    def _overlay_hand_skeleton(self, frame, results_hands):
        """Gambar skeleton tangan yang stylized"""
        if not results_hands.multi_hand_landmarks:
            return
        for hand_lm in results_hands.multi_hand_landmarks:
            self.mp_draw.draw_landmarks(
                frame, hand_lm,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_draw_styles.get_default_hand_landmarks_style(),
                self.mp_draw_styles.get_default_hand_connections_style()
            )

    def _overlay_face_box(self, frame, results_face, h, w):
        if not results_face.detections:
            self.face_bbox = None
            return
        for det in results_face.detections:
            bb  = det.location_data.relative_bounding_box
            x1  = int(bb.xmin * w)
            y1  = int(bb.ymin * h)
            bw  = int(bb.width * w)
            bh  = int(bb.height * h)
            x2, y2 = x1 + bw, y1 + bh
            self.face_bbox = (x1, y1, x2, y2)

            # Kotak stilisasi sudut
            col = EMOTION_COLORS.get(self.current_emotion, C["accent"])
            L   = 18
            thick = 2
            for (cx, cy, dx, dy) in [
                (x1, y1, 1, 1), (x2, y1, -1, 1),
                (x1, y2, 1, -1), (x2, y2, -1, -1)
            ]:
                cv2.line(frame, (cx, cy), (cx + dx * L, cy), col, thick)
                cv2.line(frame, (cx, cy), (cx, cy + dy * L), col, thick)

            # Label emosi di bawah kotak wajah
            label = f"{self.current_emotion.upper()} | {self.current_gender} ~{self.current_age}yr"
            lbl_y = min(y2 + 20, h - 5)
            cv2.putText(frame, label, (x1, lbl_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

    def _draw_gesture_feedback(self, frame, h, w):
        """Tampilkan gesture aktif sebagai overlay besar di tengah"""
        if self.current_gesture in ("idle", "—", "unknown", ""):
            return
        now = time.time()
        age = now - self.gesture_active_ts
        if age > 1.5:
            return
        alpha = max(0, 1.0 - age / 1.5)
        overlay = frame.copy()
        text = self.current_gesture.upper()
        scale = 1.2
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 2)
        tx = (w - tw) // 2
        ty = h - 80
        cv2.rectangle(overlay, (tx - 15, ty - th - 10), (tx + tw + 15, ty + 10),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        col = C["accent"]
        cv2.putText(frame, text, (tx, ty),
                    cv2.FONT_HERSHEY_DUPLEX, scale, col, 2, cv2.LINE_AA)

    def _draw_scan_lines(self, frame):
        """Sudut brackets stylized (garis scan-line full-frame dihapus -- terlalu mengganggu visibilitas)"""
        L, T, col = 40, 3, C["accent"]
        h, w = frame.shape[:2]
        for (x, y, sx, sy) in [(0, 0, 1, 1), (w, 0, -1, 1),
                                (0, h, 1, -1), (w, h, -1, -1)]:
            cv2.line(frame, (x, y), (x + sx * L, y), col, T)
            cv2.line(frame, (x, y), (x, y + sy * L), col, T)

    # ──────────────────────────────────────────────────────────────
    #  MAIN LOOP
    # ──────────────────────────────────────────────────────────────

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Kamera tidak ditemukan!")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        cap.set(cv2.CAP_PROP_FPS, 30)

        print("╔══════════════════════════════════════╗")
        print("║   GESTURE MIND — Kamera aktif! 🎥    ║")
        print("║   Tekan  Q  untuk keluar             ║")
        print("║   Tekan  F  untuk toggle fullscreen  ║")
        print("╚══════════════════════════════════════╝")

        window_name = "GESTURE MIND — Body Language OS Controller"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        is_fullscreen = True

        canvas = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)

        while self.running:
            ret, raw = cap.read()
            if not ret:
                break

            self.frame_count += 1
            self.fps_counter.append(time.time())

            # ── Resize & flip ───────────────────────────────────
            frame = cv2.resize(raw, (CAM_W, CAM_H))
            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False

            # ── Proses MediaPipe ────────────────────────────────
            res_hands = self.hands.process(rgb)
            res_face  = self.face_det.process(rgb)

            task_result = None
            if self.gesture_recognizer is not None:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                task_result = self.gesture_recognizer.recognize(mp_image)

            rgb.flags.writeable = True

            h, w = frame.shape[:2]

            # ── Kumpulkan landmark tangan ───────────────────────
            all_hands_pts = []
            if res_hands.multi_hand_landmarks:
                for hand_lm in res_hands.multi_hand_landmarks:
                    pts = self._landmarks_to_arr(hand_lm, w, h)
                    all_hands_pts.append(pts)

            # ── Deteksi gesture ──────────────────────────────────
            gesture, conf, extra = self._detect_gesture(all_hands_pts, h, w, task_result)

            if gesture not in ("idle", "unknown"):
                if gesture != self.current_gesture:
                    self.gesture_active_ts = time.time()
                self.current_gesture = gesture
                self.gesture_conf    = conf
                self._execute(gesture, extra)
            else:
                if time.time() - self.gesture_active_ts > 2.0:
                    self.current_gesture = "—"
                    self.gesture_conf    = 0.0

            # ── Wajah: kirim ke analisis, poll hasil ────────────
            if self.face_bbox:
                self._try_analyze_face(frame, self.face_bbox)
            self._poll_face_result()

            # ── Render overlay kamera ────────────────────────────
            self._overlay_hand_skeleton(frame, res_hands)
            self._overlay_face_box(frame, res_face, h, w)
            self._draw_gesture_feedback(frame, h, w)
            self._draw_scan_lines(frame)

            # ── Timestamp di pojok kiri ─────────────────────────
            ts_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            cv2.putText(frame, ts_str, (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, C["text_dim"], 1)

            # ── Gabungkan ke canvas ──────────────────────────────
            canvas[:, :CAM_W] = frame
            panel = np.zeros((WIN_H, PANEL_W, 3), dtype=np.uint8)
            canvas[:, CAM_W:] = self._draw_panel(panel)

            # Garis pemisah
            cv2.line(canvas, (CAM_W, 0), (CAM_W, WIN_H), C["accent2"], 1)

            cv2.imshow(window_name, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('f'):
                is_fullscreen = not is_fullscreen
                cv2.setWindowProperty(
                    window_name, cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
                )

        self.running = False
        cap.release()
        cv2.destroyAllWindows()
        print("\n[INFO] Gesture Mind ditutup. Sampai jumpa! 👋")


# ════════════════════════════════════════════════════════════════════
#  ENTRYPOINT
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════╗
║          👁️  GESTURE MIND — Body Language OS Controller          ║
╠══════════════════════════════════════════════════════════════════╣
║  Pastikan dependensi sudah terinstall:                           ║
║    pip install -r requirements.txt                              ║
║  (model GestureRecognizer resmi Google diunduh otomatis run 1x)  ║
║                                                                  ║
║  Kontrol Gesture (model resmi Google, fallback heuristik):      ║
║   ✊  Kepalan (Closed_Fist) → Tutup Aplikasi (Alt+F4)            ║
║   ✌  Victory/Peace         → Screenshot                         ║
║   ☝  Telunjuk atas (Pointing_Up) → Volume UP                    ║
║   👇  Telunjuk bawah        → Volume DOWN                       ║
║   🖐  Telapak (Open_Palm) tinggi → Scroll UP                    ║
║   🖐  Telapak (Open_Palm) rendah → Scroll DOWN                  ║
║   👍  Thumb_Up              → Like (no action)                  ║
║   👎  Thumb_Down            → Tutup Aplikasi                    ║
║   🤟  ILoveYou              → Minimize Semua Window             ║
║   🤝  Dua Tangan            → Lock Screen                       ║
║   👌  OK Sign               → Play/Pause Media                  ║
║                                                                  ║
║  Analisis Wajah (tiap 3 detik):                                  ║
║   😊 Emosi · 🎂 Usia · ⚧ Gender                                 ║
║                                                                  ║
║  Tekan  Q  atau  ESC  untuk keluar                              ║
╚══════════════════════════════════════════════════════════════════╝
""")
    app = GestureMind()
    app.run()
