import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class BacktestResult:
    ticker: str
    model: str
    start_date: str
    end_date: str
    total_return: float
    benchmark_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    equity_curve: list[dict]  # [{date, portfolio_value, benchmark_value}]


class Backtester:
    def __init__(self, initial_capital: float = 10_000.0, transaction_cost: float = 0.001):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost

    def run(
        self,
        df: pd.DataFrame,
        predictions: pd.Series,
        ticker: str = "UNKNOWN",
        model: str = "unknown"
    ) -> BacktestResult:
        """
        Simple long/short strategy: buy if predicted > current, sell otherwise.
        df must have 'close' column. predictions indexed same as df.
        """
        df = df.copy()
        df['signal'] = np.where(predictions.values > df['close'].values, 1, -1)
        df['daily_return'] = df['close'].pct_change()
        df['strategy_return'] = df['signal'].shift(1) * df['daily_return']
        df['strategy_return'] -= abs(df['signal'].diff()) * self.transaction_cost / 2

        df.dropna(inplace=True)

        portfolio = (1 + df['strategy_return']).cumprod() * self.initial_capital
        benchmark = (1 + df['daily_return']).cumprod() * self.initial_capital

        total_return = float((portfolio.iloc[-1] / self.initial_capital) - 1)
        bench_return = float((benchmark.iloc[-1] / self.initial_capital) - 1)
        excess_returns = df['strategy_return'] - 0.02 / 252  # risk-free rate daily
        sharpe = float(np.sqrt(252) * excess_returns.mean() / excess_returns.std()) if excess_returns.std() != 0 else 0
        downside = excess_returns[excess_returns < 0].std()
        sortino = float(np.sqrt(252) * excess_returns.mean() / downside) if downside != 0 else 0
        rolling_max = portfolio.cummax()
        drawdown = (portfolio - rolling_max) / rolling_max
        max_drawdown = float(drawdown.min())
        wins = df['strategy_return'][df['signal'].shift(1) != 0] > 0
        win_rate = float(wins.mean()) if len(wins) > 0 else 0
        total_trades = int(abs(df['signal'].diff()).sum() / 2)

        equity_curve = [
            {
                "date": str(idx.date()),
                "portfolio": round(float(p), 2),
                "benchmark": round(float(b), 2),
            }
            for idx, p, b in zip(df.index, portfolio, benchmark)
        ]

        return BacktestResult(
            ticker=ticker, model=model,
            start_date=str(df.index[0].date()), end_date=str(df.index[-1].date()),
            total_return=round(total_return, 4), benchmark_return=round(bench_return, 4),
            sharpe_ratio=round(sharpe, 3), sortino_ratio=round(sortino, 3),
            max_drawdown=round(max_drawdown, 4), win_rate=round(win_rate, 4),
            total_trades=total_trades, equity_curve=equity_curve
        )
