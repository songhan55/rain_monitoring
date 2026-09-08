"""
국가교통정보센터(ITS) CCTV 영상 정보 조회 및 스트림 수집 모듈
"""

import os
import sys
import json
import ssl
import urllib.request
from typing import List, Dict, Optional

# Windows 콘솔 출력 인코딩 설정
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def load_env(env_path: str = ".env") -> Dict[str, str]:
    """간단한 .env 파일 로더"""
    env_vars = {}
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    env_vars[k.strip()] = v.strip()
    return env_vars


class ITSCCTVClient:
    BASE_URL = "https://openapi.its.go.kr:9443/cctvInfo"

    def __init__(self, api_key: Optional[str] = None):
        if not api_key:
            env = load_env()
            api_key = env.get("ITS_CCTV_API_KEY")
        if not api_key:
            raise ValueError("ITS_CCTV_API_KEY가 제공되지 않았습니다. .env 파일을 확인하세요.")
        self.api_key = api_key

    def get_cctv_list(
        self,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        road_type: str = "ex",      # ex: 고속도로, its: 국도
        cctv_type: int = 1,         # 1: HLS 실시간 스트리밍, 2: 동영상 파일, 3: 정지영상
    ) -> List[Dict]:
        """
        주어진 위경도 Bounding Box 범위 내의 CCTV 목록을 조회합니다.
        
        :param min_x: 최소 경도 (예: 126.9)
        :param max_x: 최대 경도 (예: 127.1)
        :param min_y: 최소 위도 (예: 37.3)
        :param max_y: 최대 위도 (예: 37.5)
        :param road_type: 'ex' (고속도로) 또는 'its' (국도)
        :param cctv_type: 1 (HLS 실시간 스트리밍)
        :return: CCTV 정보 딕셔너리 리스트
        """
        params = f"apiKey={self.api_key}&type={road_type}&cctvType={cctv_type}&minX={min_x}&maxX={max_x}&minY={min_y}&maxY={max_y}&getType=json"
        url = f"{self.BASE_URL}?{params}"

        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
            }
        )

        with urllib.request.urlopen(req, context=ctx, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        response_data = data.get("response", {})
        cctv_items = response_data.get("data", [])
        if isinstance(cctv_items, dict):
            cctv_items = [cctv_items]
        return cctv_items


if __name__ == "__main__":
    client = ITSCCTVClient()
    print("📡 국가교통정보센터 CCTV API 호출 테스트 중...")
    
    # 서울 강남/판교 고속도로 권역 조회 (경도: 127.0~127.1, 위도: 37.35~37.45)
    cctvs = client.get_cctv_list(min_x=127.0, max_x=127.1, min_y=37.35, max_y=37.45)
    print(f"✅ 조회된 CCTV 개수: {len(cctvs)}개\n")

    for i, cctv in enumerate(cctvs[:3], 1):
        print(f"[{i}] CCTV 명칭: {cctv.get('cctvname')}")
        print(f"    - 좌표: 위도 {cctv.get('coordy')}, 경도 {cctv.get('coordx')}")
        print(f"    - 포맷: {cctv.get('cctvformat')}")
        print(f"    - 스트림 URL: {cctv.get('cctvurl')}\n")
