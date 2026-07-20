# benchmark.py - CSV vs Parquet 성능 비교 (데이터 규모별)
# main.py의 저장/성능 측정을 여러 규모로 확장해, Parquet이 CSV를 역전하는 지점을 확인한다.
# (강의 파트 6: timeit 반복 측정 / 파트 3: pandas Parquet)

import timeit
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
BENCH_DIR = BASE_DIR / "data" / "bench"
SIZES = [1_000, 10_000, 100_000, 1_000_000]  # 측정할 행 수
REPEAT = 3  # 각 측정 반복 횟수 (평균)


def make_df(n):
    # weather 스키마를 닮은 합성 데이터 n행 생성
    # 랜덤값을 써서 Parquet 압축이 비현실적으로 잘 되지 않게 한다
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=n, freq="min").astype(str),
            "temperature_2m": rng.uniform(-20, 40, n).round(1),
            "precipitation_probability": rng.integers(0, 101, n),
        }
    )


def measure(df):
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = BENCH_DIR / "bench.csv"
    pq_path = BENCH_DIR / "bench.parquet"

    def avg_ms(fn):
        # timeit으로 REPEAT번 실행한 평균 시간을 밀리초로 반환
        return timeit.timeit(fn, number=REPEAT) / REPEAT * 1000

    return {
        "csv_w": avg_ms(lambda: df.to_csv(csv_path, index=False)),
        "pq_w": avg_ms(lambda: df.to_parquet(pq_path, index=False)),
        "csv_r": avg_ms(lambda: pd.read_csv(csv_path)),
        "pq_r": avg_ms(lambda: pd.read_parquet(pq_path)),
        "csv_mb": csv_path.stat().st_size / 1e6,
        "pq_mb": pq_path.stat().st_size / 1e6,
    }


def main():
    print("규모별 CSV vs Parquet 성능 (쓰기/읽기 ms, 파일 MB)\n")
    for n in SIZES:
        m = measure(make_df(n))
        print(f"[{n:,}행]")
        print(f"  쓰기: CSV {m['csv_w']:8.1f} ms | Parquet {m['pq_w']:8.1f} ms")
        print(f"  읽기: CSV {m['csv_r']:8.1f} ms | Parquet {m['pq_r']:8.1f} ms")
        print(f"  크기: CSV {m['csv_mb']:8.2f} MB | Parquet {m['pq_mb']:8.2f} MB\n")


if __name__ == "__main__":
    main()