# Day 1 종합 실습 - 데이터 수집 미니 파이프라인 (단일 파일)
# 광주캠퍼스 4반 권경현
#
# 공개 API 3개를 asyncio + httpx로 동시에 수집하고, Pydantic v2로 타입과 범위를
# 검증한 뒤 CSV와 Parquet로 저장하며 읽기/쓰기 시간을 비교한다.
#
# 실행: python main.py   (스크립트와 같은 위치에 data 폴더를 만들어 결과를 저장)
# 필요 패키지: requirements.txt 참고 (pip install -r requirements.txt)

import asyncio
import json
import logging
import sys
import timeit
from pathlib import Path

import httpx
import pandas as pd
from pydantic import BaseModel, Field, ValidationError

# 로그를 stdout으로 보내 print 출력과 순서가 섞이지 않게 한다
logging.basicConfig(
    level=logging.INFO, format="%(levelname)s | %(message)s", stream=sys.stdout
)
# httpx가 요청마다 남기는 INFO 로그는 꺼서 파이프라인 로그만 보이게 한다
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("day1")

# 스크립트 위치를 기준으로 경로를 잡아 폴더를 다른 곳으로 옮겨도 그대로 동작하게 한다
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# Day 1 명세에 주어진 사용 API 3종
URLS = {
    "weather": (
        "https://api.open-meteo.com/v1/forecast"
        "?latitude=37.5665&longitude=126.9780"
        "&hourly=temperature_2m,precipitation_probability"
        "&forecast_days=3&timezone=Asia/Seoul"
    ),
    "country": "https://countries.dev/alpha/KOR",
    "ip": "http://ip-api.com/json/8.8.8.8",
}


# 스키마 정의: 수집한 JSON에서 필요한 필드의 타입과 범위를 검증한다 (강의 파트 4)
class WeatherRecord(BaseModel):
    time: str = Field(min_length=1)  # ISO 시각 문자열
    temperature_2m: float = Field(ge=-60, le=60)  # 섭씨, 상식적인 범위
    precipitation_probability: int = Field(ge=0, le=100)  # 강수확률 퍼센트


class CountryRecord(BaseModel):
    name: str = Field(min_length=1)
    region: str = Field(min_length=1)
    population: int = Field(gt=0)
    area: float = Field(gt=0)
    code: str = Field(min_length=3, max_length=3)  # alpha3 국가 코드


class IpRecord(BaseModel):
    ip: str = Field(min_length=1)
    country: str = Field(min_length=1)
    country_code: str = Field(min_length=2, max_length=2)  # alpha2 국가 코드
    lat: float = Field(ge=-90, le=90)  # 위도 범위
    lon: float = Field(ge=-180, le=180)  # 경도 범위
    timezone: str = Field(min_length=1)


# 수집: asyncio + httpx로 3개 API를 동시에 호출한다 (강의 파트 6)
async def fetch(client, name, url):
    # 단일 API 호출. 한 곳이 실패해도 파이프라인이 멈추지 않도록 예외를 감싼다.
    try:
        r = await client.get(url, timeout=10)
        r.raise_for_status()
        return name, r.json()
    except Exception as e:
        return name, {"error": str(e)}


async def fetch_all(urls=URLS):
    # asyncio.gather로 3개 요청을 동시에 실행해 {name: json} 형태로 반환
    # 일시적 네트워크 오류에 대비해 전송 계층에서 2회 재시도한다 (강의 파트 3 재시도 개념)
    transport = httpx.AsyncHTTPTransport(retries=2)
    async with httpx.AsyncClient(transport=transport) as client:
        tasks = [fetch(client, name, url) for name, url in urls.items()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    return dict(results)


def collect(urls=URLS):
    # 이벤트 루프를 실행하는 동기 진입점
    return asyncio.run(fetch_all(urls))


# 추출: 원본 JSON에서 필요한 필드만 골라 모델 입력 형태로 바꾼다
def extract_weather(raw):
    # Open-Meteo는 시각/기온/강수확률을 각각 다른 배열로 주므로 zip으로 행 단위로 묶는다
    hourly = raw["hourly"]
    return [
        {"time": t, "temperature_2m": temp, "precipitation_probability": pop}
        for t, temp, pop in zip(
            hourly["time"],
            hourly["temperature_2m"],
            hourly["precipitation_probability"],
        )
    ]


def extract_country(raw):
    # 없는 키는 None으로 두어 검증 단계에서 걸러지게 한다
    return {
        "name": raw.get("name"),
        "region": raw.get("region"),
        "population": raw.get("population"),
        "area": raw.get("area"),
        "code": raw.get("alpha3Code"),
    }


def extract_ip(raw):
    return {
        "ip": raw.get("query"),
        "country": raw.get("country"),
        "country_code": raw.get("countryCode"),
        "lat": raw.get("lat"),
        "lon": raw.get("lon"),
        "timezone": raw.get("timezone"),
    }


# 검증: rows를 model로 하나씩 검증해 통과분(valid)과 실패분(errors)으로 나눈다 (파트 4)
def validate(model, rows):
    valid, errors = [], []
    for i, row in enumerate(rows):
        try:
            valid.append(model(**row))
        except ValidationError as e:
            errors.append({"row": i, "error": e.errors()})
    return valid, errors


# 저장: CSV와 Parquet로 저장하고 timeit으로 읽기/쓰기 시간을 비교한다 (파트 3, 6)
def save_and_compare(records, out_dir, repeat=50):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)

    csv_path = out_dir / "weather.csv"
    parquet_path = out_dir / "weather.parquet"

    # 첫 호출에는 pyarrow 초기화 같은 일회성 비용이 섞여 Parquet이 실제보다 느리게 측정된다.
    # 측정 전에 한 번씩 warm-up)을 해서 그 비용을 제외한다.
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)
    pd.read_csv(csv_path)
    pd.read_parquet(parquet_path)

    perf = {
        # timeit은 N번 실행 총 시간을 주므로 N으로 나눠 1회 평균을 낸다
        "csv_write": timeit.timeit(
            lambda: df.to_csv(csv_path, index=False), number=repeat
        ) / repeat,
        "parquet_write": timeit.timeit(
            lambda: df.to_parquet(parquet_path, index=False), number=repeat
        ) / repeat,
        "csv_read": timeit.timeit(
            lambda: pd.read_csv(csv_path), number=repeat
        ) / repeat,
        "parquet_read": timeit.timeit(
            lambda: pd.read_parquet(parquet_path), number=repeat
        ) / repeat,
        "csv_bytes": csv_path.stat().st_size,
        "parquet_bytes": parquet_path.stat().st_size,
    }
    return df, perf


def main():
    # 1) 비동기 수집
    logger.info("수집 시작: %d개 API 동시 호출", len(URLS))
    raw = collect()
    for name in URLS:
        failed = "error" in raw.get(name, {"error": "no-response"})
        logger.info("  %s 응답: %s", name, "실패" if failed else "정상")
    if "error" in raw.get("weather", {}):
        logger.error("weather API 실패로 중단: %s", raw["weather"]["error"])
        sys.exit(1)

    # 2) 필드 추출 + Pydantic 검증 (타입/범위)
    weather_rows = extract_weather(raw["weather"])
    weather_valid, weather_errors = validate(WeatherRecord, weather_rows)
    country_valid, country_errors = validate(
        CountryRecord, [extract_country(raw["country"])]
    )
    ip_valid, ip_errors = validate(IpRecord, [extract_ip(raw["ip"])])
    logger.info(
        "검증 결과(valid/total): weather %d/%d, country %d/1, ip %d/1",
        len(weather_valid), len(weather_rows), len(country_valid), len(ip_valid),
    )

    # 3) 저장 + 성능 비교 (검증 통과한 weather 데이터를 CSV/Parquet로)
    records = [w.model_dump() for w in weather_valid]
    df, perf = save_and_compare(records, DATA_DIR)
    logger.info("저장 완료: weather %d행 -> CSV, Parquet", len(df))
    print("\n[저장/성능 비교] (평균, " + str(len(df)) + "행 기준)")
    print(
        f"  쓰기  CSV {perf['csv_write'] * 1000:.3f} ms | "
        f"Parquet {perf['parquet_write'] * 1000:.3f} ms"
    )
    print(
        f"  읽기  CSV {perf['csv_read'] * 1000:.3f} ms | "
        f"Parquet {perf['parquet_read'] * 1000:.3f} ms"
    )
    print(f"  크기  CSV {perf['csv_bytes']} B | Parquet {perf['parquet_bytes']} B")

    # 4) 국가/IP 요약을 JSON으로 저장 (한글이 깨지지 않도록 ensure_ascii=False)
    summary = {
        "country": country_valid[0].model_dump() if country_valid else None,
        "ip": ip_valid[0].model_dump() if ip_valid else None,
    }
    (DATA_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n[국가/IP 요약]")
    print(f"  country: {summary['country']}")
    print(f"  ip: {summary['ip']}")

    total_errors = len(weather_errors) + len(country_errors) + len(ip_errors)
    print(f"\n검증 실패 총 {total_errors}건")


if __name__ == "__main__":
    main()
