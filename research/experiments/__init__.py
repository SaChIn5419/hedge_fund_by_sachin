"""Experiment and reporting modules for Chimera research."""

__all__ = ["ReportOutputs", "StaticBacktestReporter"]


def __getattr__(name):
    if name in __all__:
        from .backtest_report import ReportOutputs, StaticBacktestReporter

        return {
            "ReportOutputs": ReportOutputs,
            "StaticBacktestReporter": StaticBacktestReporter,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
