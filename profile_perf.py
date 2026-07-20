# profile_perf.py - 성능 프로파일링 (sys, timeit, cProfile, memory_profiler)
# CSV/Parquet 저장·읽기를 여러 도구로 분석한다. (강의 파트 6: 성능 측정)
#
# 일반 실행:      python profile_perf.py
# 라인별 메모리:  python -m memory_profiler profile_perf.py

import cProfile
import io
import pstats
import sys
import timeit
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "data" / "profile"
N = 100_000  # 프로파일링용 데이터 행 수

# memory_profiler로 실행할 때만 profile이 주입된다. 일반 실행에서는 no-op로 둔다.
try:
    profile  # noqa: F821
except NameError:
    def profile(func):
        return func


def make_df(n):
    # weather 스키마를 닮은 합성 데이터 n행 생성
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=n, freq="min").astype(str),
            "temperature_2m": rng.uniform(-20, 40, n).round(1),
            "precipitation_probability": rng.integers(0, 101, n),
        }
    )


@profile
def save_and_read(df):
    # memory_profiler로 실행하면 이 함수의 라인별 메모리 증감을 보여준다
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "p.csv"
    pq_path = OUT_DIR / "p.parquet"
    df.to_csv(csv_path, index=False)
    df.to_parquet(pq_path, index=False)
    csv_back = pd.read_csv(csv_path)
    pq_back = pd.read_parquet(pq_path)
    return csv_back, pq_back


def section_sys(df):
    # sys.getsizeof는 얕은 크기만 잰다. DataFrame 실제 메모리는 memory_usage로 본다.
    print("[sys / 메모리 크기]")
    # pandas는 __sizeof__를 재정의해 sys.getsizeof가 실제 메모리를 보고한다
    print(f"  sys.getsizeof(df)     : {sys.getsizeof(df):>12,} B")
    real = int(df.memory_usage(deep=True).sum())
    print(f"  df.memory_usage(deep) : {real:>12,} B")
    # 일반 객체는 sys.getsizeof가 얕은 크기만 잰다: 리스트 vs 제너레이터 (강의 파트 6)
    lst = list(range(N))
    gen = (x for x in range(N))
    print(f"  list(range(N))        : {sys.getsizeof(lst):>12,} B (원소 전부 보유)")
    print(f"  generator             : {sys.getsizeof(gen):>12,} B (규칙만 보유)")


def section_timeit(df):
    # timeit으로 쓰기/읽기 평균 시간(ms) 비교
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "t.csv"
    pq_path = OUT_DIR / "t.parquet"

    def ms(fn):
        return timeit.timeit(fn, number=3) / 3 * 1000

    print("\n[timeit / 시간 (ms)]")
    csv_w = ms(lambda: df.to_csv(csv_path, index=False))
    pq_w = ms(lambda: df.to_parquet(pq_path, index=False))
    csv_r = ms(lambda: pd.read_csv(csv_path))
    pq_r = ms(lambda: pd.read_parquet(pq_path))
    print(f"  쓰기 CSV {csv_w:7.1f} | Parquet {pq_w:7.1f}")
    print(f"  읽기 CSV {csv_r:7.1f} | Parquet {pq_r:7.1f}")


def section_cprofile(df):
    # cProfile로 save_and_read에서 어떤 함수가 시간을 많이 쓰는지 분석
    print("\n[cProfile / 함수별 누적시간 상위 8]")
    pr = cProfile.Profile()
    pr.enable()
    save_and_read(df)
    pr.disable()
    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats("cumtime").print_stats(8)
    for line in buf.getvalue().splitlines():
        if line.strip():
            print("  " + line.rstrip())


def main():
    df = make_df(N)
    print(f"데이터 {N:,}행 생성\n")
    section_sys(df)
    section_timeit(df)
    section_cprofile(df)


if __name__ == "__main__":
    main()