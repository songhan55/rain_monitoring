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
    print("⚠️ 기상청 클라이언트 초기화 오류:", e)

# YOLO 모델 로드 (커스텀 가중치 우선 검색)
CUSTOM_WEIGHTS = "best.pt"
MODEL_PATH = CUSTOM_WEIGHTS if os.path.exists(CUSTOM_WEIGHTS) else "yolov8n.pt"
print(f"📦 YOLO 모델 로드 중: {MODEL_PATH}")
yolo_model = YOLO(MODEL_PATH)

# 전역 관제 상태
global_state = {
    "conf_threshold": 0.35,
    "simulation_mode": True,  # 데모 위험(포트홀/침수) 주입 모드
    "potholes": 0,
    "floodings": 0,
    "vehicles": 0,
    "latency": 0,
    "events": [],
    "active_cctv_url": None,
    "active_cctv_name": "연결 대기중"
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

    if not weather_client:
        return jsonify({"success": False, "msg": "Weather client not initialized"})

    try:
        # 데모 시뮬레이션 모드일 때 가상 호우 상황 반환 옵션
        weather = weather_client.get_weather(lat, lon)
        if global_state.get("simulation_mode"):
            # 시뮬레이션 모드 시 비가 오는 상황(호우 주의)으로 데모 데이터 보강
            weather["rn1"] = 18.5
            weather["pty"] = 1
            weather["pty_text"] = "집중 호우 🌧️ (데모 시뮬레이션)"
            weather["is_raining"] = True
            weather["rain_level"] = "WARNING"
            weather["level_text"] = "🚨 호우 경보 발령 (침수 주의 관제 가동)"
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


def generate_frames(stream_url: str, cctv_name: str):
    """CCTV HLS 스트림을 읽고 YOLO 추론 및 시각화를 수행하여 MJPEG 스트림으로 반환"""
    cap = cv2.VideoCapture(stream_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    frame_counter = 0
    sim_phase = 0
    last_boxes = []
    latency_ms = 35

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame_counter += 1
        h, w, _ = frame.shape

        # 3프레임마다 1번만 YOLO 추론 수행 (CPU 환경 15~20 FPS 부드러운 재생 유지)
        if frame_counter % 3 == 0:
            start_time = time.time()
            results = yolo_model(frame, conf=global_state["conf_threshold"], verbose=False)[0]
            latency_ms = int((time.time() - start_time) * 1000)
            global_state["latency"] = latency_ms
            last_boxes = results.boxes

        vehicle_count = 0
        pothole_count = 0
        flooding_count = 0

        # 2. 검출된 객체 렌더링 (차량, 트럭, 버스 등)
        for box in last_boxes:
            cls_id = int(box.cls[0])
            cls_name = yolo_model.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            if "pothole" in cls_name.lower():
                pothole_count += 1
                draw_pothole_box(frame, x1, y1, x2, y2, conf)
            elif "flood" in cls_name.lower() or "water" in cls_name.lower():
                flooding_count += 1
                draw_flood_polygon(frame, [(x1, y1), (x2, y1), (x2, y2), (x1, y2)], conf)
            elif cls_name in ["car", "truck", "bus", "motorcycle"]:
                vehicle_count += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (16, 185, 129), 2)
                label = f"{cls_name} {int(conf*100)}%"
                cv2.putText(frame, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (16, 185, 129), 1)

        # 3. 데모 위험 시뮬레이션 모드 활성화 시 (실제 CCTV에 포트홀/침수가 없을 때 UI 및 알림 테스트용)
        if global_state["simulation_mode"]:
            sim_phase += 0.05
            # 도로 1차선 하단에 포트홀 시뮬레이션 박스 렌더링
            px1, py1 = int(w * 0.28), int(h * 0.72)
            px2, py2 = int(w * 0.38), int(h * 0.82)
            pothole_conf = 0.82 + 0.08 * math.sin(sim_phase)
            draw_pothole_box(frame, px1, py1, px2, py2, pothole_conf)
            pothole_count += 1

            # 도로 갓길/우측 차선에 도로 침수(수막) 시뮬레이션 영역 렌더링
            pts = np.array([
                [int(w * 0.65), int(h * 0.60)],
                [int(w * 0.88), int(h * 0.65)],
                [int(w * 0.95), int(h * 0.85)],
                [int(w * 0.68), int(h * 0.82)]
            ], np.int32)
            flood_conf = 0.89 + 0.06 * math.cos(sim_phase)
            draw_flood_polygon(frame, pts, flood_conf)
            flooding_count += 1

        # 통계 갱신
        global_state["vehicles"] = vehicle_count
        global_state["potholes"] = pothole_count
        global_state["floodings"] = flooding_count

        # 위험 이벤트 발생 시 로그 기록 (중복 방지: 50프레임마다 1회)
        if (pothole_count > 0 or flooding_count > 0) and (frame_counter % 60 == 0):
            now_str = datetime.now().strftime("%H:%M:%S")
            if pothole_count > 0:
                record_event(now_str, cctv_name, "POTHOLE", 0.86)
            if flooding_count > 0:
                record_event(now_str, cctv_name, "FLOODING", 0.91)

        # 4. 화면 상단 정보 오버레이 (HUD 워터마크)
        cv2.rectangle(frame, (10, 10), (320, 60), (15, 23, 42), -1)
        cv2.rectangle(frame, (10, 10), (320, 60), (51, 65, 85), 1)
        cv2.putText(frame, f"CCTV: {cctv_name[:20]}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        status_text = f"POTHOLES: {pothole_count} | FLOOD: {flooding_count} | {latency_ms}ms"
        cv2.putText(frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (6, 182, 212), 1)

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
    overlay = frame.copy()
    if isinstance(pts, list):
        pts = np.array(pts, np.int32)
    cv2.fillPoly(overlay, [pts], (214, 182, 6))  # BGR
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

    cv2.polylines(frame, [pts], True, (255, 215, 0), 2)
    # 대표 지점 라벨 표시
    center_x = int(np.mean(pts[:, 0]))
    center_y = int(np.mean(pts[:, 1]))
    label = f"FLOODING {int(conf*100)}%"
    cv2.putText(frame, label, (center_x - 40, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)


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
