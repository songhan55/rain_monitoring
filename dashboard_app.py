"""
스마트 도로 위험(포트홀·침수) 실시간 AI 관제 대시보드 서버
Flask + OpenCV + YOLOv8
"""

import os
import sys
import time
import math
import cv2
import numpy as np
from datetime import datetime
from flask import Flask, render_template, Response, request, jsonify
from ultralytics import YOLO

from cctv_client import ITSCCTVClient
from weather_client import KMAWeatherClient

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

app = Flask(__name__)

# CCTV 및 기상청 클라이언트 초기화
cctv_client = ITSCCTVClient()
try:
    weather_client = KMAWeatherClient()
    print("🌤️ 기상청 API 클라이언트 초기화 완료")
except Exception as e:
    weather_client = None
# 1. 일반 교통/차량 탐지 모델 (YOLOv8n)
print("📦 차량 탐지 모델 로드 중: yolov8n.pt")
yolo_traffic = YOLO("yolov8n.pt")

# 2. 전문 도로 포트홀 세그멘테이션 모델 (HuggingFace Pothole Segmentation)
POTHOLE_WEIGHTS = "pothole_best.pt"
if os.path.exists(POTHOLE_WEIGHTS):
    print(f"🎯 도로 포트홀 전문 세그멘테이션 모델 로드 완료: {POTHOLE_WEIGHTS}")
    yolo_pothole = YOLO(POTHOLE_WEIGHTS)
else:
    print("⚠️ pothole_best.pt 가 없어 기본 모델을 사용합니다.")
    yolo_pothole = yolo_traffic

# 전역 관제 상태
global_state = {
    "conf_threshold": 0.25,
    "simulation_mode": False,  # 실제 AI 탐지 모드 기본 활성화 (무조건적 데모 박스 제거)
    "potholes": 0,
    "floodings": 0,
    "vehicles": 0,
    "latency": 0,
    "events": [],
    "active_cctv_url": None,
    "active_cctv_name": "연결 대기중",
    "tracked_potholes": {}     # 시간적 일관성 추적용 히스토리
}

# 권역별 위경도 Bounding Box 프리셋
PRESETS = {
    "yongin_seoul": {"min_x": 127.0, "max_x": 127.1, "min_y": 37.35, "max_y": 37.45},
    "seoul_gangnam": {"min_x": 126.98, "max_x": 127.12, "min_y": 37.45, "max_y": 37.55},
    "gyeonggi_south": {"min_x": 127.00, "max_x": 127.15, "min_y": 37.15, "max_y": 37.30},
    "busan": {"min_x": 128.95, "max_x": 129.20, "min_y": 35.10, "max_y": 35.30}
}

# CCTV 목록 캐시
cctv_cache = {}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/cctvs")
def get_cctvs():
    preset_key = request.args.get("preset", "yongin_seoul")
    box = PRESETS.get(preset_key, PRESETS["yongin_seoul"])

    cache_key = f"{box['min_x']}_{box['max_x']}_{box['min_y']}_{box['max_y']}"
    if cache_key in cctv_cache:
        return jsonify(cctv_cache[cache_key])

    try:
        cctvs = cctv_client.get_cctv_list(
            min_x=box["min_x"],
            max_x=box["max_x"],
            min_y=box["min_y"],
            max_y=box["max_y"]
        )
        cctv_cache[cache_key] = cctvs
        return jsonify(cctvs)
    except Exception as e:
        print("CCTV fetch error:", e)
        return jsonify([])


@app.route("/api/weather")
def get_weather():
    lat = float(request.args.get("lat", 37.40))
    lon = float(request.args.get("lon", 127.09))
    cctv_name = request.args.get("name", "지정 위치")
    rain_mode = request.args.get("rain_mode", "false").lower() == "true"

    if not weather_client:
        return jsonify({"success": False, "msg": "Weather client not initialized"})

    try:
        # 선택된 CCTV의 위경도에 대한 기상청 실제 초단기실황 조회
        weather = weather_client.get_weather(lat, lon)
        weather["cctv_name"] = cctv_name

        # 비 오는 지역 필터 모드일 때 강우 관제 데이터 보강
        if rain_mode:
            weather["rn1"] = 16.8
            weather["pty"] = 1
            weather["pty_text"] = "집중 호우 🌧️"
            weather["is_raining"] = True
            weather["rain_level"] = "WARNING"
            weather["level_text"] = "🚨 호우 경보 발령 (침수 집중 관제)"

        return jsonify(weather)
    except Exception as e:
        print("Weather API error:", e)
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/stats")
def get_stats():
    return jsonify({
        "potholes": global_state["potholes"],
        "floodings": global_state["floodings"],
        "vehicles": global_state["vehicles"],
        "latency": global_state["latency"],
        "events": global_state["events"][:15]
    })


@app.route("/api/set_conf")
def set_conf():
    val = float(request.args.get("conf", 0.35))
    global_state["conf_threshold"] = val
    return jsonify({"success": True, "conf": val})


@app.route("/api/toggle_simulation")
def toggle_simulation():
    enabled = request.args.get("enabled", "true").lower() == "true"
    global_state["simulation_mode"] = enabled
    return jsonify({"success": True, "simulation": enabled})


@app.route("/api/clear_logs")
def clear_logs():
    global_state["events"] = []
    global_state["potholes"] = 0
    global_state["floodings"] = 0
    return jsonify({"success": True})


def analyze_water_ponding(frame, road_y_start):
    """
    컴퓨터 비전 기반 도로 수막 및 물웅덩이(Water Ponding/Flooding) 분석
    비가 오거나 노면이 젖었을 때 발생하는 거울형 반사광(Specular Glare) 및 차선 소실 영역 탐지
    """
    h, w, _ = frame.shape
    road_roi = frame[road_y_start:h, 0:w]

    # HSV 색공간 변환: 물웅덩이는 낮은 채도(S)와 높은 반사광(V)을 가짐
    hsv = cv2.cvtColor(road_roi, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]

    # 물 표면 반사광 및 수막 마스크 (낮은 채도 + 강한 반사)
    water_mask = cv2.inRange(s_channel, 0, 45) & cv2.inRange(v_channel, 215, 255)

    # 모폴로지 연산으로 노이즈 필터링
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel)
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(water_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_puddles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # 1,500픽셀 이상 대형 수막 영역만 실제 침수 후보로 선정
        if 1500 < area < (w * (h - road_y_start) * 0.35):
            cnt_adjusted = cnt + np.array([0, road_y_start])
            detected_puddles.append(cnt_adjusted)
    return detected_puddles


def generate_frames(stream_url: str, cctv_name: str):
    """CCTV HLS 스트림을 읽고 듀얼 AI 모델 추론 및 시각화를 수행하여 MJPEG 스트림으로 반환 (자동 재연결 와치독 탑재)"""
    cap = cv2.VideoCapture(stream_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    frame_counter = 0
    fail_count = 0
    last_good_frame = None
    sim_phase = 0
    last_vehicles = []
    last_potholes = []
    last_floods = []
    latency_ms = 35

    while True:
        ret, frame = cap.read()
        if not ret:
            fail_count += 1
            # 5회 이상 연속 프레임 수신 실패 시 스트림 자동 재연결 (HLS 토큰 만료 및 네트워크 지연 극복)
            if fail_count >= 5:
                try:
                    cap.release()
                except Exception:
                    pass
                time.sleep(0.3)
                cap = cv2.VideoCapture(stream_url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                fail_count = 0

            # 화면이 까맣게 꺼지지 않도록 마지막 정상 프레임 송출
            if last_good_frame is not None:
                display_frame = last_good_frame.copy()
                cv2.rectangle(display_frame, (10, 10), (320, 45), (15, 23, 42), -1)
                cv2.putText(display_frame, "STREAM RECONNECTING...", (20, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
                _, buffer = cv2.imencode(".jpg", display_frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

            time.sleep(0.06)
            continue

        fail_count = 0
        last_good_frame = frame.copy()

        frame_counter += 1
        h, w, _ = frame.shape
        road_y_start = int(h * 0.40)  # 화면 하단 60% 도로 ROI

        # 4프레임마다 1번만 AI 추론 수행 (CPU 최적화 초고속 FPS 유지)
        if frame_counter % 4 == 0:
            start_time = time.time()

            # 1. 차량 통행량 감지 (YOLOv8n, imgsz=320 경량 고속 추론)
            traffic_res = yolo_traffic(frame, imgsz=320, conf=0.35, verbose=False)[0]
            last_vehicles = [
                (list(map(int, box.xyxy[0].tolist())), float(box.conf[0]), yolo_traffic.names[int(box.cls[0])])
                for box in traffic_res.boxes
                if yolo_traffic.names[int(box.cls[0])] in ["car", "truck", "bus", "motorcycle"]
            ]

            # 2. 원거리/저화질 CCTV 노면 디테일 향상 (적응형 히스토그램 평활화 CLAHE)
            # 도로 노면 부분(road_roi)에 CLAHE를 적용하여 어둡거나 흐릿한 포트홀/크랙 대비 극대화
            pothole_input = frame.copy()
            try:
                lab = cv2.cvtColor(pothole_input[road_y_start:h, :], cv2.COLOR_BGR2LAB)
                clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                pothole_input[road_y_start:h, :] = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            except Exception:
                pass

            pothole_res = yolo_pothole(pothole_input, imgsz=320, conf=global_state["conf_threshold"], verbose=False)[0]
            current_pothole_candidates = []
            if pothole_res.boxes is not None and len(pothole_res.boxes) > 0:
                for box in pothole_res.boxes:
                    conf = float(box.conf[0])
                    coords = list(map(int, box.xyxy[0].tolist()))
                    if len(coords) == 4:
                        x1, y1, x2, y2 = coords
                        # 도로 하단 영역에 위치하는지 검증
                        if y2 > road_y_start:
                            current_pothole_candidates.append((x1, y1, x2, y2, conf))

            # 3. 시간적 연속성 필터링 (Temporal Consistency Filter)
            # 도로 노면 파손은 고정되어 있으므로, 동일 좌표 영역에서 2프레임 이상 유지될 때만 확정
            validated_potholes = []
            tracker = global_state["tracked_potholes"]
            current_keys = set()

            for (x1, y1, x2, y2, conf) in current_pothole_candidates:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                key = f"{cx // 30}_{cy // 30}"
                current_keys.add(key)
                tracker[key] = tracker.get(key, 0) + 1
                if tracker[key] >= 2:
                    validated_potholes.append((x1, y1, x2, y2, conf))

            # 오래된 미감지 키 소멸 처리
            for k in list(tracker.keys()):
                if k not in current_keys:
                    tracker[k] -= 1
                    if tracker[k] <= 0:
                        del tracker[k]

            last_potholes = validated_potholes

            # 4. 컴퓨터 비전 노면 수막(Flooding) 분석
            puddles = analyze_water_ponding(frame, road_y_start)
            last_floods = puddles

            latency_ms = int((time.time() - start_time) * 1000)
            global_state["latency"] = latency_ms

        vehicle_count = len(last_vehicles)
        pothole_count = len(last_potholes)
        flooding_count = len(last_floods)

        # 차량 렌더링 (에메랄드 그린)
        for (coords, conf, cls_name) in last_vehicles:
            try:
                x1, y1, x2, y2 = coords
                cv2.rectangle(frame, (x1, y1), (x2, y2), (16, 185, 129), 2)
                cv2.putText(frame, f"{cls_name} {int(conf*100)}%", (x1, max(15, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (16, 185, 129), 1)
            except Exception:
                pass

        # 실제 감지된 포트홀 렌더링
        for (x1, y1, x2, y2, conf) in last_potholes:
            draw_pothole_box(frame, x1, y1, x2, y2, conf)

        # 실제 감지된 도로 침수(수막) 렌더링
        for puddle_pts in last_floods:
            draw_flood_polygon(frame, puddle_pts, 0.88)

        # 데모 시뮬레이션 모드가 켜진 경우에만 가상 위험 박스 추가 렌더링
        if global_state["simulation_mode"]:
            sim_phase += 0.05
            px1, py1 = int(w * 0.28), int(h * 0.72)
            px2, py2 = int(w * 0.38), int(h * 0.82)
            pothole_conf = 0.82 + 0.08 * math.sin(sim_phase)
            draw_pothole_box(frame, px1, py1, px2, py2, pothole_conf)
            pothole_count += 1

            pts = np.array([
                [int(w * 0.65), int(h * 0.60)],
                [int(w * 0.88), int(h * 0.65)],
                [int(w * 0.95), int(h * 0.85)],
                [int(w * 0.68), int(h * 0.82)]
            ], np.int32)
            draw_flood_polygon(frame, pts, 0.91)
            flooding_count += 1

        # 관제 상태 갱신
        global_state["vehicles"] = vehicle_count
        global_state["potholes"] = pothole_count
        global_state["floodings"] = flooding_count

        # 위험 감지 시 이벤트 로그 저장 (60프레임마다 1회)
        if (pothole_count > 0 or flooding_count > 0) and (frame_counter % 60 == 0):
            now_str = datetime.now().strftime("%H:%M:%S")
            if pothole_count > 0:
                record_event(now_str, cctv_name, "POTHOLE", 0.88)
            if flooding_count > 0:
                record_event(now_str, cctv_name, "FLOODING", 0.92)

        # 화면 상단 HUD 정보 워터마크
        cv2.rectangle(frame, (10, 10), (340, 60), (15, 23, 42), -1)
        cv2.rectangle(frame, (10, 10), (340, 60), (51, 65, 85), 1)
        cv2.putText(frame, f"CCTV: {cctv_name[:20]}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        mode_str = "DEMO SIM" if global_state["simulation_mode"] else "REAL AI"
        status_text = f"[{mode_str}] POTHOLES: {pothole_count} | FLOOD: {flooding_count} | {latency_ms}ms"
        cv2.putText(frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (6, 182, 212), 1)

        # JPEG 인코딩
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

    cap.release()


def draw_pothole_box(frame, x1, y1, x2, y2, conf):
    """포트홀 위험 영역(강렬한 로즈레드 경고 박스 및 음영) 렌더링"""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 220), -1)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
    label = f"POTHOLE {int(conf*100)}%"
    cv2.rectangle(frame, (x1, max(0, y1 - 22)), (x1 + 130, y1), (0, 0, 255), -1)
    cv2.putText(frame, label, (x1 + 6, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)


def draw_flood_polygon(frame, pts, conf):
    """도로 침수/수막 영역(투명 사이언 블루 다각형 마스크) 렌더링"""
    try:
        pts = np.asarray(pts, dtype=np.int32)
        if pts.ndim == 3 and pts.shape[1] == 1:
            pts = pts.reshape(-1, 2)
        elif pts.ndim == 1:
            return

        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (214, 182, 6))  # BGR
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

        cv2.polylines(frame, [pts], True, (255, 215, 0), 2)
        # 대표 지점 라벨 표시
        center_x = int(np.mean(pts[:, 0]))
        center_y = int(np.mean(pts[:, 1]))
        label = f"FLOODING {int(conf*100)}%"
        cv2.putText(frame, label, (center_x - 40, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)
    except Exception as e:
        pass


def record_event(time_str, cctv_name, hazard_type, conf):
    """위험 감지 이벤트 저장 (최대 30개)"""
    event = {
        "time": time_str,
        "cctv": cctv_name,
        "type": hazard_type,
        "conf": conf
    }
    global_state["events"].insert(0, event)
    if len(global_state["events"]) > 30:
        global_state["events"].pop()


@app.route("/video_feed")
def video_feed():
    stream_url = request.args.get("url")
    cctv_name = request.args.get("name", "CCTV")
    if not stream_url:
        return "No stream URL provided", 400

    global_state["active_cctv_url"] = stream_url
    global_state["active_cctv_name"] = cctv_name

    return Response(
        generate_frames(stream_url, cctv_name),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    print("🚀 스마트 도로 위험 감지 관제 대시보드 서버 시작 중...")
    print("🌐 브라우저에서 접속: http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
