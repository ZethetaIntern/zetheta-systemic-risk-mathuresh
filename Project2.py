import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize

# -----------------------------
# STEP 1: Download Market Data (SAFE VERSION)
# -----------------------------

stocks = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]
start_date = "2020-01-01"
end_date = "2024-01-01"

# Download with auto_adjust
data = yf.download(
    stocks,
    start=start_date,
    end=end_date,
    auto_adjust=True,
    group_by='ticker'
)

# Handle MultiIndex columns safely
if isinstance(data.columns, pd.MultiIndex):
    close_prices = pd.DataFrame()
    for stock in stocks:
        close_prices[stock] = data[stock]["Close"]
else:
    close_prices = data["Close"]

returns = close_prices.pct_change().dropna()

mean_returns = returns.mean()
cov_matrix = returns.cov()

risk_free_rate = 0.05


# -----------------------------
# STEP 2: Portfolio Optimization
# -----------------------------

def portfolio_performance(weights):
    annual_return = np.sum(mean_returns * weights) * 252
    annual_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return annual_return, annual_std


def negative_sharpe(weights):
    p_return, p_std = portfolio_performance(weights)
    return -(p_return - risk_free_rate) / p_std


def optimize_portfolio():
    num_assets = len(mean_returns)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    init_guess = num_assets * [1. / num_assets]

    result = minimize(
        negative_sharpe,
        init_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    return result.x


optimal_weights = optimize_portfolio()

print("\n🔹 Original Optimal Portfolio Weights:")
for stock, weight in zip(stocks, optimal_weights):
    print(f"{stock}: {weight:.2f}")


# -----------------------------
# STEP 3: Behavioral Bias Detection
# -----------------------------

def detect_bias(trading_frequency, panic_selling, trend_following):
    bias_scores = {}

    bias_scores["Overconfidence"] = 80 if trading_frequency > 20 else 30
    bias_scores["Loss Aversion"] = 85 if panic_selling else 40
    bias_scores["Herd Bias"] = 75 if trend_following else 35

    return bias_scores


bias_scores = detect_bias(
    trading_frequency=25,
    panic_selling=True,
    trend_following=True
)

print("\n🧠 Detected Behavioral Bias Scores:")
for bias, score in bias_scores.items():
    print(f"{bias}: {score}")


# -----------------------------
# STEP 4: Adjust Portfolio Based on Bias
# -----------------------------

def adjust_weights(weights, bias_scores):
    adjusted = weights.copy()

    if bias_scores["Overconfidence"] > 70:
        adjusted = np.ones(len(weights)) / len(weights)

    if bias_scores["Loss Aversion"] > 70:
        adjusted[0] += 0.10

    adjusted = adjusted / np.sum(adjusted)
    return adjusted


adjusted_weights = adjust_weights(optimal_weights, bias_scores)

print("\n🔹 Behavior-Adjusted Portfolio Weights:")
for stock, weight in zip(stocks, adjusted_weights):
    print(f"{stock}: {weight:.2f}")


# -----------------------------
# STEP 5: Generate Nudges
# -----------------------------

def generate_nudges(bias_scores):
    nudges = []

    if bias_scores["Overconfidence"] > 70:
        nudges.append("You trade frequently. Consider long-term investing strategy.")

    if bias_scores["Loss Aversion"] > 70:
        nudges.append("Avoid panic selling during market downturns.")

    if bias_scores["Herd Bias"] > 70:
        nudges.append("Avoid blindly following market trends.")

    return nudges


nudges = generate_nudges(bias_scores)

print("\n💬 Personalized Investment Nudges:")
for nudge in nudges:
    print("-", nudge)


# -----------------------------
# STEP 6: Visualization
# -----------------------------

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.bar(stocks, optimal_weights)
plt.title("Original Portfolio")
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
plt.bar(stocks, adjusted_weights)
plt.title("Behavior Adjusted Portfolio")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()