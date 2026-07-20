# test_main.py - 스키마 검증 단위 테스트 (강의 파트 5: pytest)
# main.py의 Pydantic 모델과 검증 함수가 잘못된 값을 제대로 걸러내는지 확인한다.

import pytest
from pydantic import ValidationError

from main import IpRecord, WeatherRecord, validate


@pytest.fixture
def valid_weather():
    # 정상적인 날씨 레코드 한 건 (여러 테스트에서 공용으로 사용)
    return {
        "time": "2026-07-20T00:00",
        "temperature_2m": 23.2,
        "precipitation_probability": 6,
    }


def test_valid_weather_passes(valid_weather):
    # 정상 데이터는 검증을 통과하고 값이 그대로 담긴다
    rec = WeatherRecord(**valid_weather)
    assert rec.temperature_2m == 23.2
    assert rec.precipitation_probability == 6


def test_precipitation_out_of_range_fails(valid_weather):
    # 강수확률이 0~100 범위를 벗어나면 ValidationError
    bad = {**valid_weather, "precipitation_probability": 150}
    with pytest.raises(ValidationError):
        WeatherRecord(**bad)


def test_temperature_type_error(valid_weather):
    # 기온에 숫자가 아닌 값이 오면 ValidationError
    bad = {**valid_weather, "temperature_2m": "덥다"}
    with pytest.raises(ValidationError):
        WeatherRecord(**bad)


@pytest.mark.parametrize("lat,lon", [(-91, 0), (0, 200)])
def test_ip_coordinate_range(lat, lon):
    # 위도(-90~90), 경도(-180~180) 범위를 벗어나면 ValidationError
    row = {
        "ip": "8.8.8.8",
        "country": "United States",
        "country_code": "US",
        "lat": lat,
        "lon": lon,
        "timezone": "America/New_York",
    }
    with pytest.raises(ValidationError):
        IpRecord(**row)


def test_validate_splits_valid_and_errors(valid_weather):
    # validate()가 통과분과 실패분을 나누는지 확인 (정상 1건 + 범위 위반 1건)
    bad = {**valid_weather, "precipitation_probability": 999}
    valid, errors = validate(WeatherRecord, [valid_weather, bad])
    assert len(valid) == 1
    assert len(errors) == 1
    assert errors[0]["row"] == 1