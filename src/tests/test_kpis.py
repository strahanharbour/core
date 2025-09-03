import math

from main.engine.metrics.kpis import sharpe_ratio, sortino_ratio, equity_curve, max_drawdown, cvar


def test_kpis_run_on_simple_returns():
    r = [0.01, -0.02, 0.03, 0.0, 0.01]

    s = sharpe_ratio(r)
    so = sortino_ratio(r)
    eq = equity_curve(r)
    mdd = max_drawdown(r)
    cv = cvar(r)

    assert isinstance(s, float)
    assert isinstance(so, float)
    assert isinstance(mdd, float)
    assert isinstance(cv, float)
    assert len(eq) == len(r)
