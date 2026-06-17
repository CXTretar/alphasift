from pathlib import Path

import pandas as pd

from alphasift.config import Config
from alphasift.pipeline import _sort_screened_candidates, screen
from alphasift.strategy import ScreeningConfig


def test_pipeline_enriches_daily_features_for_daily_strategy(monkeypatch):
    df = pd.DataFrame(
        [
            {
                "code": "000001",
                "name": "平安银行",
                "price": 10.0,
                "change_pct": -0.5,
                "amount": 200_000_000,
                "turnover_rate": 2.0,
                "volume_ratio": 1.2,
                "pe_ratio": 8.0,
                "pb_ratio": 0.8,
                "total_mv": 100_000_000_000,
            },
            {
                "code": "600000",
                "name": "浦发银行",
                "price": 11.0,
                "change_pct": -0.8,
                "amount": 190_000_000,
                "turnover_rate": 2.0,
                "volume_ratio": 1.1,
                "pe_ratio": 9.0,
                "pb_ratio": 0.9,
                "total_mv": 90_000_000_000,
            },
        ]
    )
    df.attrs["snapshot_source"] = "test"

    def fake_enrich(frame, **kwargs):
        enriched = frame.copy()
        for idx, row in enriched.iterrows():
            is_target = row["code"] == "000001"
            enriched.at[idx, "ma_bullish"] = is_target
            enriched.at[idx, "price_above_ma20"] = True
            enriched.at[idx, "signal_score"] = 72 if is_target else 80
            enriched.at[idx, "change_60d"] = 12 if is_target else 10
            enriched.at[idx, "macd_status"] = "bullish"
            enriched.at[idx, "rsi_status"] = "neutral"
            enriched.at[idx, "volume_ratio_20d"] = 1.0 if is_target else 1.8
            enriched.at[idx, "pullback_to_ma20_pct"] = 4 if is_target else 12
        return enriched

    monkeypatch.setattr("alphasift.pipeline.fetch_snapshot_with_fallback", lambda sources, **kwargs: df)
    monkeypatch.setattr("alphasift.pipeline.enrich_daily_features", fake_enrich)

    result = screen(
        "shrink_pullback",
        use_llm=False,
        config=Config(
            llm_api_key="",
            snapshot_source_priority=["test"],
            strategies_dir=Path("strategies"),
            risk_enabled=False,
        ),
    )

    assert result.daily_enriched is True
    assert result.after_filter_count == 1
    assert result.picks[0].code == "000001"
    assert result.picks[0].ma_bullish is True
    assert any("Daily K-line enrichment attempted 2 candidates" in item for item in result.degradation)


def test_volume_breakout_accepts_sina_snapshot_without_live_volume_ratio(monkeypatch):
    df = pd.DataFrame(
        [
            {
                "code": "000001",
                "name": "平安银行",
                "price": 10.0,
                "change_pct": 3.2,
                "amount": 220_000_000,
                "turnover_rate": 4.5,
                "pe_ratio": 8.0,
                "pb_ratio": 0.8,
            },
            {
                "code": "600000",
                "name": "浦发银行",
                "price": 11.0,
                "change_pct": 3.0,
                "amount": 190_000_000,
                "turnover_rate": 4.0,
                "pe_ratio": 9.0,
                "pb_ratio": 0.9,
            },
        ]
    )
    df.attrs["snapshot_source"] = "sina"

    captured_required_columns: list[str] = []

    def fake_fetch(_sources, **kwargs):
        captured_required_columns.extend(kwargs["required_columns"])
        assert "volume_ratio" not in kwargs["required_columns"]
        return df

    def fake_enrich(frame, **kwargs):
        enriched = frame.copy()
        for idx, row in enriched.iterrows():
            is_target = row["code"] == "000001"
            enriched.at[idx, "price_above_ma20"] = True
            enriched.at[idx, "signal_score"] = 72 if is_target else 58
            enriched.at[idx, "macd_status"] = "bullish"
            enriched.at[idx, "breakout_20d_pct"] = 0.6 if is_target else -2.5
            enriched.at[idx, "range_20d_pct"] = 24 if is_target else 28
            enriched.at[idx, "volume_ratio_20d"] = 1.8 if is_target else 1.4
            enriched.at[idx, "body_pct"] = 1.2 if is_target else 0.8
            enriched.at[idx, "consolidation_days_20d"] = 10 if is_target else 9
        return enriched

    monkeypatch.setattr("alphasift.pipeline.fetch_snapshot_with_fallback", fake_fetch)
    monkeypatch.setattr("alphasift.pipeline.enrich_daily_features", fake_enrich)

    result = screen(
        "volume_breakout",
        max_output=1,
        use_llm=False,
        post_analyzers=[],
        config=Config(
            llm_api_key="",
            snapshot_source_priority=["sina"],
            strategies_dir=Path("strategies"),
            risk_enabled=False,
        ),
    )

    assert "volume_ratio" not in captured_required_columns
    assert result.snapshot_source == "sina"
    assert result.daily_enriched is True
    assert result.after_filter_count == 1
    assert result.picks[0].code == "000001"
    assert result.picks[0].volume_ratio is None
    assert result.picks[0].volume_ratio_20d == 1.8


def test_volume_breakout_relaxes_daily_filters_to_keep_minimum_candidates(monkeypatch):
    df = pd.DataFrame(
        [
            {
                "code": f"00000{idx}",
                "name": f"候选{idx}",
                "price": 10.0 + idx,
                "change_pct": 2.5 + idx * 0.1,
                "amount": 220_000_000 + idx,
                "turnover_rate": 4.0,
                "pe_ratio": 8.0,
                "pb_ratio": 0.8,
            }
            for idx in range(1, 6)
        ]
    )
    df.attrs["snapshot_source"] = "sina"

    def fake_fetch(_sources, **kwargs):
        assert "volume_ratio" not in kwargs["required_columns"]
        return df

    def fake_enrich(frame, **kwargs):
        enriched = frame.copy()
        for idx in enriched.index:
            enriched.at[idx, "price_above_ma20"] = False
            enriched.at[idx, "signal_score"] = 30
            enriched.at[idx, "macd_status"] = "bearish"
            enriched.at[idx, "breakout_20d_pct"] = -12.0
            enriched.at[idx, "range_20d_pct"] = 95.0
            enriched.at[idx, "volume_ratio_20d"] = 0.2
            enriched.at[idx, "body_pct"] = -0.5
            enriched.at[idx, "consolidation_days_20d"] = 0
        return enriched

    monkeypatch.setattr("alphasift.pipeline.fetch_snapshot_with_fallback", fake_fetch)
    monkeypatch.setattr("alphasift.pipeline.enrich_daily_features", fake_enrich)

    result = screen(
        "volume_breakout",
        max_output=3,
        use_llm=False,
        post_analyzers=[],
        config=Config(
            llm_api_key="",
            snapshot_source_priority=["sina"],
            strategies_dir=Path("strategies"),
            risk_enabled=False,
        ),
    )

    assert result.snapshot_source == "sina"
    assert result.daily_enriched is True
    assert result.after_filter_count >= 3
    assert len(result.picks) == 3
    assert any("Daily hard filter relaxation stage" in item for item in result.degradation)
    assert "No candidates after daily hard filter" not in result.degradation


def test_pipeline_preserves_degradation_when_hard_filter_empty(monkeypatch):
    df = pd.DataFrame([
        {
            "code": "000001",
            "name": "平安银行",
            "price": 10.0,
            "change_pct": 0.0,
            "amount": 1,
            "total_mv": 1,
            "pe_ratio": 1000.0,
            "pb_ratio": 100.0,
        }
    ])
    df.attrs["snapshot_source"] = "test"
    df.attrs["source_errors"] = ["efinance failed"]
    monkeypatch.setattr("alphasift.pipeline.fetch_snapshot_with_fallback", lambda sources, **kwargs: df)

    result = screen(
        "dual_low",
        use_llm=False,
        post_analyzers=[],
        config=Config(
            llm_api_key="",
            snapshot_source_priority=["test"],
            strategies_dir=Path("strategies"),
            risk_enabled=False,
        ),
    )

    assert result.picks == []
    assert any("Snapshot source fallback: efinance failed" in item for item in result.degradation)
    assert "No candidates after hard filter" in result.degradation


def test_pipeline_passes_industry_provider_cache_config(monkeypatch, tmp_path):
    df = pd.DataFrame([
        {
            "code": "000001",
            "name": "骞冲畨閾惰",
            "price": 10.0,
            "change_pct": 0.0,
            "amount": 100_000_000,
            "turnover_rate": 2.0,
            "volume_ratio": 1.2,
            "pe_ratio": 8.0,
            "pb_ratio": 0.8,
            "total_mv": 100_000_000_000,
        }
    ])
    df.attrs["snapshot_source"] = "test"
    calls = []

    def fake_enrich(frame, **kwargs):
        calls.append(kwargs)
        return frame, []

    monkeypatch.setattr("alphasift.pipeline.fetch_snapshot_with_fallback", lambda sources, **kwargs: df)
    monkeypatch.setattr("alphasift.pipeline.enrich_industry_concepts", fake_enrich)

    cache_dir = tmp_path / "industry-cache"
    screen(
        "dual_low",
        use_llm=False,
        post_analyzers=[],
        config=Config(
            llm_api_key="",
            snapshot_source_priority=["test"],
            strategies_dir=Path("strategies"),
            industry_provider="akshare",
            industry_provider_cache_dir=cache_dir,
            industry_provider_cache_ttl_hours=7,
            risk_enabled=False,
        ),
    )

    assert calls == [{
        "map_files": [],
        "provider": "akshare",
        "max_boards": 80,
        "provider_cache_dir": cache_dir,
        "provider_cache_ttl_hours": 7,
    }]


def test_sort_screened_candidates_uses_strategy_factor_tie_breakers_then_code():
    df = pd.DataFrame([
        {"code": "600000", "screen_score": 80, "factor_momentum_score": 70, "factor_stability_score": 90},
        {"code": "000001", "screen_score": 80, "factor_momentum_score": 70, "factor_stability_score": 90},
        {"code": "300001", "screen_score": 80, "factor_momentum_score": 75, "factor_stability_score": 10},
        {"code": "002001", "screen_score": 81, "factor_momentum_score": 20, "factor_stability_score": 20},
    ])
    screening = ScreeningConfig(factor_weights={"momentum": 0.7, "stability": 0.3})

    sorted_df = _sort_screened_candidates(df, screening)

    assert list(sorted_df["code"]) == ["002001", "300001", "000001", "600000"]


def test_sort_screened_candidates_keeps_default_tie_breakers_without_weights():
    df = pd.DataFrame([
        {"code": "600000", "screen_score": 80, "factor_stability_score": 70, "factor_activity_score": 50},
        {"code": "000001", "screen_score": 80, "factor_stability_score": 70, "factor_activity_score": 50},
        {"code": "300001", "screen_score": 80, "factor_stability_score": 75, "factor_activity_score": 10},
    ])

    sorted_df = _sort_screened_candidates(df, ScreeningConfig())

    assert list(sorted_df["code"]) == ["300001", "000001", "600000"]
