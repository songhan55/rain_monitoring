"""
기상청 초단기실황 API 연동 및 GPS 위경도 ↔ 기상청 격자(nx, ny) 변환 모듈
"""

import math
import datetime
import urllib.request
import json
from typing import Dict, Tuple, Optional
from cctv_client import load_env


class KMAWeatherClient:
    BASE_URL = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtNcst"

    def __init__(self, api_key: Optional[str] = None):
        if not api_key:
            env = load_env()
            api_key = env.get("KMA_WEATHER_API_KEY")
        if not api_key:
            raise ValueError("KMA_WEATHER_API_KEY가 없습니다. .env 파일을 확인하세요.")
        self.api_key = api_key

    @staticmethod
    def latlon_to_grid(lat: float, lon: float) -> Tuple[int, int]:
        """
        기상청 LCC 투영법 공식에 따른 위도/경도 -> 기상청 격자(nx, ny) 변환 함수
        """
        RE = 6371.00877     # 지구 반경(km)
        GRID = 5.0          # 격자 간격(km)
        SLAT1 = 30.0        # 투영 위도1(degree)
        SLAT2 = 60.0        # 투영 위도2(degree)
        OLON = 126.0        # 기준점 경도(degree)
        OLAT = 38.0         # 기준점 위도(degree)
        XO = 43             # 기준점 X좌표(GRID)
        YO = 136            # 기준점 Y좌표(GRID)

        DEG2RAD = math.pi / 180.0

        re = RE / GRID
        slat1 = SLAT1 * DEG2RAD
        slat2 = SLAT2 * DEG2RAD
        olon = OLON * DEG2RAD
        olat = OLAT * DEG2RAD

        sn = math.tan(math.pi * 0.25 + slat2 * 0.5) / math.tan(math.pi * 0.25 + slat1 * 0.5)
        sn = math.log(math.cos(slat1) / math.cos(slat2)) / math.log(sn)
        sf = math.tan(math.pi * 0.25 + slat1 * 0.5)
        sf = math.pow(sf, sn) * math.cos(slat1) / sn
        ro = math.tan(math.pi * 0.25 + olat * 0.5)
        ro = re * sf / math.pow(ro, sn)

        ra = math.tan(math.pi * 0.25 + (lat) * DEG2RAD * 0.5)
        ra = re * sf / math.pow(ra, sn)
        theta = lon * DEG2RAD - olon
        if theta > math.pi:
            theta -= 2.0 * math.pi
        if theta < -math.pi:
            theta += 2.0 * math.pi
        theta *= sn

        nx = int(math.floor(ra * math.sin(theta) + XO + 0.5))
        ny = int(math.floor(ro - ra * math.cos(theta) + YO + 0.5))
        return nx, ny

    def get_weather(self, lat: float, lon: float) -> Dict:
        """
        특정 위경도 지점의 기상청 초단기실황(강수량, 기온, 강수형태 등)을 조회합니다.
        """
        nx, ny = self.latlon_to_grid(lat, lon)

        now = datetime.datetime.now()
        # 기상청 초단기실황은 매시간 40분에 생성되므로, 45분 이전이면 1시간 전 데이터 기준
        if now.minute < 45:
            now -= datetime.timedelta(hours=1)
        base_date = now.strftime("%Y%m%d")
        base_time = now.strftime("%H00")

        url = (
            f"{self.BASE_URL}?"
            f"serviceKey={self.api_key}&pageNo=1&numOfRows=100&dataType=JSON&"
            f"base_date={base_date}&base_time={base_time}&nx={nx}&ny={ny}"
        )

        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        items = data["response"]["body"]["items"]["item"]
        parsed = {}
        for it in items:
            parsed[it["category"]] = float(it["obsrValue"])

        rn1 = parsed.get("RN1", 0.0)      # 1시간 강수량 (mm)
        pty = int(parsed.get("PTY", 0))    # 강수 형태
        temp = parsed.get("T1H", 0.0)     # 기온
        humidity = parsed.get("REH", 0.0) # 습도
        wind_speed = parsed.get("WSD", 0.0) # 풍속

        pty_map = {
            0: "강수 없음 (맑음/흐림)",
            1: "비 🌧️",
            2: "비/눈 🌨️",
            3: "눈 ❄️",
            5: "빗방울 💧",
            6: "빗방울/눈날림",
            7: "눈날림"
        }

        is_raining = (pty in [1, 2, 5]) or (rn1 > 0.0)

        # 강우 위험 등급 판정
        if rn1 >= 15.0:
            rain_level = "WARNING"  # 호우 경보 수준
            level_text = "🚨 호우 경보 (침수 극도 위험)"
        elif rn1 >= 5.0 or pty == 1:
            rain_level = "WATCH"    # 주의보 수준
            level_text = "⚠️ 강우 주의 (수막현상 주의)"
        else:
            rain_level = "NORMAL"
            level_text = "🟢 정상 (강수 위험 없음)"

        return {
            "success": True,
            "lat": lat,
            "lon": lon,
            "nx": nx,
            "ny": ny,
            "base_time": f"{base_date} {base_time}",
            "rn1": rn1,
            "pty": pty,
            "pty_text": pty_map.get(pty, "기타"),
            "temp": temp,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "is_raining": is_raining,
            "rain_level": rain_level,
            "level_text": level_text
        }


if __name__ == "__main__":
    client = KMAWeatherClient()
    # 판교/서초 부근 (위도 37.40, 경도 127.09)
    result = client.get_weather(37.40, 127.09)
    print("🌤️ 기상청 초단기실황 조회 결과:")
    print(f"  - 관측 시각: {result['base_time']}")
    print(f"  - 1시간 강수량 (RN1): {result['rn1']} mm")
    print(f"  - 강수 형태 (PTY): {result['pty_text']}")
    print(f"  - 현재 기온: {result['temp']} ℃")
    print(f"  - 위험 판정: {result['level_text']}")
