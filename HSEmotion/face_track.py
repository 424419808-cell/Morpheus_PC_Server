#!/usr/bin/env python3
"""
Morpheus Face Track — RetinaFace NPU + MobileFaceNet NPU + MobileNetV2 Emotion NPU + PID servo tracking.
Integrated on RK3588: no network split needed.
"""
import cv2, numpy as np, os, time, threading, json
from itertools import product as product
from math import ceil
from rknnlite.api import RKNNLite
from mor_servo_dev import PCA9685, angle_to_pulse_us_with_mid, angle_to_pulse_us
import warnings
warnings.filterwarnings("ignore", message=".*feature names.*")
warnings.filterwarnings("ignore", category=UserWarning)

BASE = os.path.dirname(os.path.abspath(__file__))
RETINAFACE_MODEL = os.path.join(BASE, "RetinaFace_mobile320.rknn")
MBF_MODEL = os.path.join(BASE, "w600k_mbf.rknn")
EMOTION_MODEL = os.path.join(BASE, "emotion_mobilenetv2.rknn")
REF_PHOTO = os.path.join(BASE, "zhengzheng_ref.jpg")

MODEL_SIZE = (320, 320)
SCORE_THRESH = 0.6
NMS_THRESH = 0.4
EMBED_SIZE = (112, 112)
SIM_THRESH = 0.6

EMO_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


def crop_face_for_emotion(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    margin = 0.3
    nw = int(bw * (1 + margin))
    nh = int(bh * (1 + margin))
    x1 = max(0, int(cx - nw / 2))
    y1 = max(0, int(cy - nh / 2))
    x2 = min(w - 1, int(cx + nw / 2))
    y2 = min(h - 1, int(cy + nh / 2))
    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        return None
    face128 = cv2.resize(face, (128, 128))
    rgb = cv2.cvtColor(face128, cv2.COLOR_BGR2RGB)
    return np.expand_dims(rgb.astype(np.uint8), 0)


def prior_box(image_size):
    anchors = []
    min_sizes = [[16, 32], [64, 128], [256, 512]]
    steps = [8, 16, 32]
    feature_maps = [[ceil(image_size[0] / step), ceil(image_size[1] / step)] for step in steps]
    for k, f in enumerate(feature_maps):
        for i, j in product(range(f[0]), range(f[1])):
            for ms in min_sizes[k]:
                s_kx = ms / image_size[1]; s_ky = ms / image_size[0]
                cx = j * steps[k] / image_size[1] + 0.5 * steps[k] / image_size[1]
                cy = i * steps[k] / image_size[0] + 0.5 * steps[k] / image_size[0]
                anchors += [cx, cy, s_kx, s_ky]
    return np.array(anchors).reshape(-1, 4).astype(np.float32)


_priors_cache = prior_box(MODEL_SIZE)


def decode_boxes(loc, priors, variances=(0.1, 0.2)):
    boxes = np.concatenate((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), axis=1)
    boxes[:, :2] -= boxes[:, 2:] / 2; boxes[:, 2:] += boxes[:, :2]
    return boxes


def decode_landmarks(pre, priors, variances=(0.1, 0.2)):
    return np.concatenate((
        priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
        priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
        priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
        priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
        priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
    ), axis=1)


def nms(dets, thresh):
    if len(dets) == 0:
        return []
    x1, y1, x2, y2 = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
    scores = dets[:, 4]; areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]; keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0., xx2 - xx1 + 1); h = np.maximum(0., yy2 - yy1 + 1)
        ovr = w * h / (areas[i] + areas[order[1:]] - w * h)
        order = order[np.where(ovr <= thresh)[0] + 1]
    return keep


def letterbox_resize(image, size, bg_color=114):
    tw, th = size; h, w = image.shape[:2]
    scale = min(tw / w, th / h); nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.ones((th, tw, 3), dtype=np.uint8) * bg_color
    ox, oy = (tw - nw) // 2, (th - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas, scale, ox, oy


# ═══════════════════════════════════════════════════════════════
#  Face alignment & MobileFaceNet helpers
# ═══════════════════════════════════════════════════════════════
CANONICAL = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.6963],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.3655],
], dtype=np.float32)


def align_face(image, landmarks_5):
    src = cv2.UMat(landmarks_5.reshape(5, 2).astype(np.float32))
    M, _ = cv2.estimateAffinePartial2D(src, CANONICAL, method=cv2.LMEDS)
    if M is None:
        return None
    if hasattr(M, 'get'):
        M = M.get()
    gpu_img = cv2.UMat(image)
    return cv2.warpAffine(gpu_img, M, EMBED_SIZE, flags=cv2.INTER_LINEAR).get()


def preprocess_for_mbf(face_bgr):
    rgb = face_bgr[..., ::-1]
    chw = np.transpose(rgb, (2, 0, 1))
    return np.expand_dims(chw.astype(np.float32), 0)


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


# ═══════════════════════════════════════════════════════════════
#  Kalman Filter
# ═══════════════════════════════════════════════════════════════
class SimpleKalman:
    def __init__(self):
        self.state = np.zeros(4); self.P = np.eye(4) * 10
        self.F = np.eye(4); self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.Q = np.eye(4) * 0.1; self.R = np.eye(2) * 5

    def predict(self, dt):
        self.F[0, 2] = dt; self.F[1, 3] = dt
        self.state = np.dot(self.F, self.state)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z):
        y = z - np.dot(self.H, self.state)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.state = self.state + np.dot(K, y)
        self.P = np.dot((np.eye(4) - np.dot(K, self.H)), self.P)


# ═══════════════════════════════════════════════════════════════
#  Servo Tracker
# ═══════════════════════════════════════════════════════════════
CONFIG = {
    6:  (30.0,  110.0,  60.0),
    23: (55.0,  95.0,  75.0),
    30: (110.0, 165.0, 155.0),
    31: (25.0,  75.0,  34.0),
    32: (0, 180.0, 90)
}
EYE_LIDS = {"TL": 4, "BL": 5, "TR": 25, "BR": 24}
OPEN_POS = {"TL": 40.0, "BL": 100.0, "TR": 120.0, "BR": 20.0}


class ServoTracker:
    def __init__(self):
        self.pcas = {
            0x40: PCA9685(bus_id=4, address=0x40, freq_hz=50),
            0x41: PCA9685(bus_id=4, address=0x41, freq_hz=50),
            0x42: PCA9685(bus_id=4, address=0x42, freq_hz=50)
        }
        self.curr_angles = {ch: cfg[2] for ch, cfg in CONFIG.items()}
        self.kp_x, self.kp_y = 0.06, 0.06
        self.ki_x, self.ki_y = 0.003, 0.003
        self.kd_x, self.kd_y = 0.015, 0.015
        self.i_max = 15.0
        self.deriv_alpha = 0.3
        self.error_sum_x, self.error_sum_y = 0.0, 0.0
        self.last_error_x, self.last_error_y = 0.0, 0.0
        self.deriv_x, self.deriv_y = 0.0, 0.0
        self.smooth = 0.35; self.max_step = 3.5; self.deadzone = 8
        self.micro_zone_x, self.micro_zone_y = 115, 50
        self.eye_x_mid = CONFIG[23][2]
        self.align_p = 0.15; self.align_deadzone = 3.0
        self.kf = SimpleKalman(); self.last_time = time.time()
        self.has_detected_first_time = False; self.prediction_count = 0
        self.MAX_PREDICTIONS = 100; self.is_sleeping = False
        self.reset_pose()

    def get_target(self, global_ch):
        if 0 <= global_ch <= 15:
            return self.pcas[0x40], global_ch
        if 16 <= global_ch <= 31:
            return self.pcas[0x41], global_ch - 16
        if global_ch == 32:
            return self.pcas[0x42], 0
        return None, None

    def update_servo(self, ch, target_angle):
        current_a = self.curr_angles[ch]; diff = target_angle - current_a
        if abs(diff) > self.max_step:
            diff = self.max_step if diff > 0 else -self.max_step
        new_angle = current_a + (diff * self.smooth)
        min_a, max_a, _ = CONFIG[ch]; new_angle = max(min_a, min(max_a, new_angle))
        pca, local_ch = self.get_target(ch)
        if pca:
            pulse = angle_to_pulse_us_with_mid(new_angle, 0, 90, 180, 600, 1500, 2400)
            pca.set_servo_pulse_us(local_ch, pulse)
            self.curr_angles[ch] = new_angle

    def update_eyelids_follow(self):
        if self.is_sleeping:
            return
        eye_y = self.curr_angles[6]; y_min, y_max, y_mid = CONFIG[6]
        offset_ratio = (eye_y - y_mid) / (y_max - y_mid if eye_y > y_mid else y_mid - y_min)
        follow_amplitude = 15.0
        l_up = OPEN_POS["TL"] + (offset_ratio * follow_amplitude)
        r_up = OPEN_POS["TR"] - (offset_ratio * follow_amplitude)
        pca, ch = self.get_target(EYE_LIDS["TL"]); pca.set_servo_pulse_us(ch, angle_to_pulse_us(l_up, 0, 180, 600, 2400))
        pca, ch = self.get_target(EYE_LIDS["TR"]); pca.set_servo_pulse_us(ch, angle_to_pulse_us(r_up, 0, 180, 600, 2400))

    def track(self, error_x, error_y, is_real_data=True):
        self.is_sleeping = False; now = time.time(); dt = now - self.last_time; self.last_time = now
        self.kf.predict(dt)
        if is_real_data:
            self.kf.update(np.array([error_x, error_y]))
            self.has_detected_first_time = True; self.prediction_count = 0
        else:
            self.prediction_count += 1
            error_x, error_y = self.kf.state[0], self.kf.state[1]
        if self.prediction_count > self.MAX_PREDICTIONS:
            self.sleep(); return
        if is_real_data:
            self.error_sum_x += error_x * dt
            self.error_sum_y += error_y * dt
            self.error_sum_x = max(-self.i_max, min(self.i_max, self.error_sum_x))
            self.error_sum_y = max(-self.i_max, min(self.i_max, self.error_sum_y))
        if self.has_detected_first_time and dt > 0.001:
            raw_dx = (error_x - self.last_error_x) / dt
            raw_dy = (error_y - self.last_error_y) / dt
            self.deriv_x = self.deriv_x * (1 - self.deriv_alpha) + raw_dx * self.deriv_alpha
            self.deriv_y = self.deriv_y * (1 - self.deriv_alpha) + raw_dy * self.deriv_alpha
        self.last_error_x = error_x; self.last_error_y = error_y
        move_x = error_x * self.kp_x + self.error_sum_x * self.ki_x + self.deriv_x * self.kd_x
        move_y = error_y * self.kp_y + self.error_sum_y * self.ki_y + self.deriv_y * self.kd_y
        if abs(error_x) < self.micro_zone_x:
            self.update_servo(23, self.curr_angles[23] - move_x * 0.9)
        else:
            self.update_servo(23, self.curr_angles[23] - move_x * 0.3)
            self.update_servo(32, self.curr_angles[32] - move_x * 0.7)
        if is_real_data and abs(error_x) < (self.micro_zone_x * 0.5):
            eye_x_offset = self.curr_angles[23] - self.eye_x_mid
            if abs(eye_x_offset) > self.align_deadzone:
                align_speed = max(-1.2, min(1.2, eye_x_offset * self.align_p))
                self.update_servo(32, self.curr_angles[32] + align_speed)
                self.update_servo(23, self.curr_angles[23] - align_speed)
        if abs(error_y) < self.micro_zone_y:
            self.update_servo(6, self.curr_angles[6] - move_y * 1.0)
        else:
            self.update_servo(6, self.curr_angles[6] - move_y * 0.4)
            self.update_servo(30, self.curr_angles[30] - move_y * 0.8)
            self.update_servo(31, self.curr_angles[31] + move_y * 0.8)
        self.update_eyelids_follow()

    def reset_pose(self):
        for ch in CONFIG.keys():
            self.update_servo(ch, CONFIG[ch][2])
        for name, ch_id in EYE_LIDS.items():
            pca, l_ch = self.get_target(ch_id)
            if pca:
                pca.set_servo_pulse_us(l_ch, angle_to_pulse_us(OPEN_POS[name], 0, 180, 600, 2400))

    def sleep(self):
        if not self.is_sleeping:
            print(">>> 目标丢失，进入待机模式")
            for ch in CONFIG.keys():
                pca, local_ch = self.get_target(ch)
                if pca:
                    pca.set_channel_full_off(local_ch, True)
            self.is_sleeping = True; self.has_detected_first_time = False
            self.error_sum_x = self.error_sum_y = 0.0
            self.deriv_x = self.deriv_y = 0.0

    def cleanup(self):
        for ch in list(CONFIG.keys()) + list(EYE_LIDS.values()):
            pca, local_ch = self.get_target(ch)
            if pca:
                pca.set_channel_full_off(local_ch, True)


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════
cv2.ocl.setUseOpenCL(True)


def main():
    print("=" * 50)
    print("  Morpheus Face Track — RK3588 集成版")
    print("  RetinaFace(NPU) + MobileFaceNet(NPU) + Emotion(NPU) + 舵机追踪")
    print("=" * 50)

    # ── Load RetinaFace NPU ──
    print("\n[1/5] Loading RetinaFace NPU...")
    rkn_face = RKNNLite()
    rkn_face.load_rknn(RETINAFACE_MODEL)
    rkn_face.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    print("  RetinaFace ready.")

    # ── Load MobileFaceNet NPU ──
    print("[2/5] Loading MobileFaceNet NPU...")
    rkn_mbf = RKNNLite()
    rkn_mbf.load_rknn(MBF_MODEL)
    rkn_mbf.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    print("  MobileFaceNet ready.")

    # ── Load Emotion MobileNetV2 NPU ──
    print("[3/5] Loading Emotion MobileNetV2 NPU...")
    rkn_emo = RKNNLite()
    rkn_emo.load_rknn(EMOTION_MODEL)
    rkn_emo.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    print("  Emotion MobileNetV2 ready.")

    # ── Register zhengzheng ──
    print("[4/5] Registering zhengzheng reference...")
    ref_img = cv2.imread(REF_PHOTO)
    if ref_img is None:
        print(f"ERROR: Cannot read {REF_PHOTO}")
        return
    lb, sc, ox, oy = letterbox_resize(ref_img, MODEL_SIZE)
    out = rkn_face.inference(inputs=[np.expand_dims(lb[..., ::-1], 0)])
    loc, conf, lms_raw = out[0].squeeze(0), out[1].squeeze(0), out[2].squeeze(0)
    boxes = decode_boxes(loc, _priors_cache) * np.array([320, 320, 320, 320])
    boxes[:, 0::2] = np.clip((boxes[:, 0::2] - ox) / sc, 0, ref_img.shape[1])
    boxes[:, 1::2] = np.clip((boxes[:, 1::2] - oy) / sc, 0, ref_img.shape[0])
    lms = decode_landmarks(lms_raw, _priors_cache) * np.array([320, 320] * 5)
    lms[:, 0::2] = np.clip((lms[:, 0::2] - ox) / sc, 0, ref_img.shape[1])
    lms[:, 1::2] = np.clip((lms[:, 1::2] - oy) / sc, 0, ref_img.shape[0])
    inds = np.where(conf[:, 1] > SCORE_THRESH)[0]; boxes = boxes[inds]; lms = lms[inds]; scores = conf[inds, 1]
    if len(scores) > 0:
        dets = np.hstack((boxes, scores[:, None])).astype(np.float32)
        keep = nms(dets, NMS_THRESH); boxes = boxes[keep]; lms = lms[keep]; scores = scores[keep]
        best = np.argmax(scores)
        aligned = align_face(ref_img, lms[best])
        if aligned is not None:
            ref_embedding = rkn_mbf.inference(inputs=[preprocess_for_mbf(aligned)])[0].flatten()
            print(f"  Registered! embedding norm={np.linalg.norm(ref_embedding):.2f}")
        else:
            print("ERROR: Face alignment failed")
            return
    else:
        print("ERROR: No face found in reference photo")
        return

    # ── Init tracker ──
    print("[5/5] Initializing servo tracker...")
    tracker = ServoTracker()
    print("  Tracker ready.")

    # ── Open camera ──
    print("Opening camera (shared memory)...")
    while not os.path.exists("/dev/shm/frame.npy"):
        time.sleep(0.1)
    print("  Camera ready (shared memory).\n")

    fps_timer = time.time(); frame_count = 0; fps = 0.0; display = True
    print("Press 'q' to quit, 'd' to toggle display.\n")

    while True:
        try:
            frame = np.load("/dev/shm/frame.npy")
        except:
            time.sleep(0.001)
            continue
        h, w = frame.shape[:2]; cx_center, cy_center = w // 2, h // 2

        # RetinaFace NPU
        lb, sc, ox, oy = letterbox_resize(frame, MODEL_SIZE)
        out = rkn_face.inference(inputs=[np.expand_dims(lb[..., ::-1], 0)])
        loc, conf, lms_raw = out[0].squeeze(0), out[1].squeeze(0), out[2].squeeze(0)
        boxes = decode_boxes(loc, _priors_cache) * np.array([320, 320, 320, 320])
        boxes[:, 0::2] = np.clip((boxes[:, 0::2] - ox) / sc, 0, w)
        boxes[:, 1::2] = np.clip((boxes[:, 1::2] - oy) / sc, 0, h)
        scores = conf[:, 1]
        lms = decode_landmarks(lms_raw, _priors_cache) * np.array([320, 320] * 5)
        lms[:, 0::2] = np.clip((lms[:, 0::2] - ox) / sc, 0, w)
        lms[:, 1::2] = np.clip((lms[:, 1::2] - oy) / sc, 0, h)

        inds = np.where(scores > SCORE_THRESH)[0]; boxes = boxes[inds]; lms = lms[inds]; scores = scores[inds]

        zhengzheng_found = False; right_eye_xy = None
        emo_lbl = ""

        if len(scores) > 0:
            dets = np.hstack((boxes, scores[:, None])).astype(np.float32)
            keep = nms(dets, NMS_THRESH); boxes = boxes[keep]; lms = lms[keep]; scores = scores[keep]
            for i in range(len(scores)):
                if scores[i] < SCORE_THRESH:
                    continue
                x1, y1, x2, y2 = map(int, boxes[i]); lm = lms[i]
                right_eye = (int(lm[2]), int(lm[3]))

                # MobileFaceNet recognition
                aligned = align_face(frame, lm)
                is_zz = False; sim = 0.0
                if aligned is not None:
                    emb = rkn_mbf.inference(inputs=[preprocess_for_mbf(aligned)])[0].flatten()
                    sim = cosine_sim(ref_embedding, emb)
                    is_zz = sim > SIM_THRESH

                if is_zz:
                    zhengzheng_found = True; right_eye_xy = right_eye
                    # Emotion inference via MobileNetV2 NPU (inline, ~5ms)
                    emo_in = crop_face_for_emotion(frame, boxes[i])
                    emo_conf = 0.0
                    if emo_in is not None:
                        emo_out = rkn_emo.inference(inputs=[emo_in])[0][0]
                        emo_pred = np.argmax(emo_out)
                        emo_conf = float(emo_out[emo_pred])
                        emo_lbl = EMO_LABELS[emo_pred]
                    if display:
                        cv2.drawMarker(frame, right_eye, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
                        label = f"{emo_lbl} {emo_conf:.2f}" if emo_lbl else ""
                        if label:
                            cv2.putText(frame, label, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.line(frame, (cx_center, cy_center), right_eye, (255, 255, 0), 1)

        # Servo tracking
        if zhengzheng_found and right_eye_xy:
            error_x = right_eye_xy[0] - cx_center
            error_y = right_eye_xy[1] - cy_center
            if abs(error_x) > tracker.deadzone or abs(error_y) > tracker.deadzone:
                tracker.track(error_x, error_y, is_real_data=True)
            else:
                tracker.track(0, 0, is_real_data=False)
        else:
            if tracker.has_detected_first_time:
                tracker.track(0, 0, is_real_data=False)
            else:
                tracker.sleep()

        # Display
        if display:
            frame_count += 1
            if frame_count % 15 == 0:
                elapsed = time.time() - fps_timer
                fps = 15 / elapsed
                fps_timer = time.time()
            cv2.circle(frame, (cx_center, cy_center), 5, (255, 0, 0), 1)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if zhengzheng_found:
                cv2.putText(frame, f"err({error_x:.0f},{error_y:.0f})", (w - 200, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            if frame_count % 30 == 0:
                print(f"  running: {fps:.1f}fps, emo={emo_lbl or 'wait'}", flush=True)
            cv2.imshow("Morpheus Face Track", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            display = not display
            if not display:
                cv2.destroyAllWindows()
                print("Display off")

    cv2.destroyAllWindows()
    rkn_face.release()
    rkn_mbf.release()
    rkn_emo.release()
    tracker.cleanup()
    print("Done.")


if __name__ == "__main__":
    main()
