import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

stocks = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "INFY.NS"
]

data = yf.download(
    stocks,
    start="2021-01-01",
    end="2026-01-01"
)["Close"]

print(data.head())
returns = data.pct_change().dropna()

print(returns.head())

weights = np.array([0.25, 0.25, 0.25, 0.25])

portfolio_returns = returns.dot(weights)

print(portfolio_returns.head())

investment = 100000

historical_var = portfolio_returns.quantile(0.05)

var_amount = abs(historical_var * investment)

print(f"95% Historical VaR: ₹{var_amount:.2f}")

mean = portfolio_returns.mean()
std = portfolio_returns.std()

parametric_var = abs(
    (mean + norm.ppf(0.05) * std)
    * investment
)

print(f"95% Parametric VaR: ₹{parametric_var:.2f}")

plt.figure(figsize=(10,5))
plt.hist(portfolio_returns, bins=50)

plt.axvline(
    historical_var,
    linestyle='--',
    label='Historical VaR'
)

plt.legend()
plt.title("Portfolio Return Distribution")
plt.xlabel("Daily Return")
plt.ylabel("Frequency")

plt.show()

annual_return = portfolio_returns.mean() * 252
annual_volatility = portfolio_returns.std() * np.sqrt(252)

print(f"Annual Return: {annual_return:.2%}")
print(f"Annual Volatility: {annual_volatility:.2%}")

risk_free_rate = 0.06

sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility

print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

violations = portfolio_returns < historical_var

violation_rate = violations.mean()

print(f"VaR Violation Rate: {violation_rate:.2%}")

num_simulations = 10000

simulated_returns = np.random.normal(
    mean,
    std,
    num_simulations
)
simulated_pnl = simulated_returns * investment
monte_carlo_var = abs(
    np.percentile(simulated_pnl, 5)
)

print(f"95% Monte Carlo VaR: ₹{monte_carlo_var:.2f}")
plt.figure(figsize=(10,5))

plt.hist(simulated_pnl, bins=50)

plt.axvline(
    -monte_carlo_var,
    linestyle="--",
    label="Monte Carlo VaR"
)

plt.legend()

plt.title("Monte Carlo Simulation")
plt.xlabel("Profit / Loss (₹)")
plt.ylabel("Frequency")

plt.show()
print("\n--- VaR Comparison ---")
print(f"Historical VaR : ₹{var_amount:.2f}")
print(f"Parametric VaR : ₹{parametric_var:.2f}")
print(f"Monte Carlo VaR: ₹{monte_carlo_var:.2f}")

tail_losses = portfolio_returns[
    portfolio_returns <= historical_var
]

expected_shortfall = abs(
    tail_losses.mean() * investment
)

print(f"95% Expected Shortfall: ₹{expected_shortfall:.2f}")
print("\n===== Risk Summary =====")
print(f"Historical VaR      : ₹{var_amount:.2f}")
print(f"Parametric VaR      : ₹{parametric_var:.2f}")
print(f"Monte Carlo VaR     : ₹{monte_carlo_var:.2f}")
print(f"Expected Shortfall  : ₹{expected_shortfall:.2f}")
