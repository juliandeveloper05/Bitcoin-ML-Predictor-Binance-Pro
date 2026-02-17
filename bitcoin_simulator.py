import numpy as np
import pandas as pd

# Configuration Parameters
DAYS = 60
STARTING_PRICE = 60000  # Bitcoin starting price in USD
INITIAL_CASH = 10000    # Starting cash in USD
ANNUAL_DRIFT = 0.10     # Expected annual return (10%)
ANNUAL_VOLATILITY = 0.60  # Annual volatility (60% - typical for crypto)
SHORT_MA = 7            # Short moving average window
LONG_MA = 30            # Long moving average window

# Set random seed for reproducibility
np.random.seed(42)

# Simulate Bitcoin price data using Geometric Brownian Motion (GBM)
def simulate_bitcoin_prices(days, start_price, drift, volatility):
    """
    Simulate Bitcoin prices using Geometric Brownian Motion.
    
    Formula: S(t+1) = S(t) * exp((drift - 0.5*volatility^2)*dt + volatility*sqrt(dt)*Z)
    where Z ~ N(0,1)
    """
    dt = 1/365  # Daily time step (fraction of year)
    prices = [start_price]
    
    for _ in range(days - 1):
        # Generate random shock from standard normal distribution
        Z = np.random.standard_normal()
        
        # Calculate next price using GBM formula
        drift_component = (drift - 0.5 * volatility**2) * dt
        random_component = volatility * np.sqrt(dt) * Z
        next_price = prices[-1] * np.exp(drift_component + random_component)
        
        prices.append(next_price)
    
    return prices

# Generate simulated Bitcoin prices
prices = simulate_bitcoin_prices(DAYS, STARTING_PRICE, ANNUAL_DRIFT, ANNUAL_VOLATILITY)

# Create DataFrame
df = pd.DataFrame({
    'Day': range(1, DAYS + 1),
    'Price': prices
})

# Calculate Moving Averages
df['MA_7'] = df['Price'].rolling(window=SHORT_MA).mean()
df['MA_30'] = df['Price'].rolling(window=LONG_MA).mean()

# Initialize portfolio tracking columns
df['Position'] = 0.0  # Amount of Bitcoin held
df['Cash'] = INITIAL_CASH  # Cash balance
df['Signal'] = ''  # Buy/Sell/Hold signal
df['Action'] = ''  # Description of action taken

# Portfolio state variables
cash = INITIAL_CASH
btc_holdings = 0.0
prev_signal = None  # Track previous signal to detect crossovers

# Golden Cross Trading Algorithm
for i in range(len(df)):
    current_price = df.loc[i, 'Price']
    ma_7 = df.loc[i, 'MA_7']
    ma_30 = df.loc[i, 'MA_30']
    
    # Determine current signal
    if pd.notna(ma_7) and pd.notna(ma_30):
        if ma_7 > ma_30:
            current_signal = 'BULLISH'
        else:
            current_signal = 'BEARISH'
    else:
        current_signal = None
    
    # Detect crossover and execute trades
    action = 'HOLD'
    description = ''
    
    if current_signal and prev_signal and current_signal != prev_signal:
        # Golden Cross: 7-day MA crosses above 30-day MA
        if current_signal == 'BULLISH' and cash > 0:
            # BUY: Invest all cash
            btc_to_buy = cash / current_price
            btc_holdings += btc_to_buy
            cash = 0
            action = 'BUY'
            description = f'Bought {btc_to_buy:.6f} BTC @ ${current_price:.2f}'
        
        # Death Cross: 7-day MA crosses below 30-day MA
        elif current_signal == 'BEARISH' and btc_holdings > 0:
            # SELL: Sell all Bitcoin
            cash_received = btc_holdings * current_price
            cash += cash_received
            description = f'Sold {btc_holdings:.6f} BTC @ ${current_price:.2f}'
            btc_holdings = 0
            action = 'SELL'
    
    # Update DataFrame
    df.loc[i, 'Position'] = btc_holdings
    df.loc[i, 'Cash'] = cash
    df.loc[i, 'Signal'] = action
    df.loc[i, 'Action'] = description
    
    # Update previous signal
    prev_signal = current_signal

# Calculate portfolio value for each day
df['Portfolio_Value'] = df['Cash'] + (df['Position'] * df['Price'])

# Print Daily Ledger
print("=" * 120)
print("BITCOIN GOLDEN CROSS TRADING SIMULATOR - DAILY LEDGER")
print("=" * 120)
print(f"\nSimulation Parameters:")
print(f"  • Duration: {DAYS} days")
print(f"  • Starting Price: ${STARTING_PRICE:,.2f}")
print(f"  • Initial Cash: ${INITIAL_CASH:,.2f}")
print(f"  • Annual Drift: {ANNUAL_DRIFT*100:.1f}%")
print(f"  • Annual Volatility: {ANNUAL_VOLATILITY*100:.1f}%")
print(f"  • Moving Averages: {SHORT_MA}-day and {LONG_MA}-day")
print("\n" + "=" * 120)
print(f"{'Day':>4} {'Price':>10} {'MA-7':>10} {'MA-30':>10} {'Signal':>8} {'BTC Held':>12} {'Cash':>12} {'Portfolio':>12} {'Action':>35}")
print("=" * 120)

for i in range(len(df)):
    row = df.iloc[i]
    ma7_str = f"${row['MA_7']:.2f}" if pd.notna(row['MA_7']) else "N/A"
    ma30_str = f"${row['MA_30']:.2f}" if pd.notna(row['MA_30']) else "N/A"
    signal_display = row['Signal'] if row['Signal'] else 'HOLD'
    
    # Highlight trade days
    if row['Signal'] in ['BUY', 'SELL']:
        print(f"{int(row['Day']):>4} ${row['Price']:>9,.2f} {ma7_str:>10} {ma30_str:>10} {signal_display:>8} {row['Position']:>12.6f} ${row['Cash']:>11,.2f} ${row['Portfolio_Value']:>11,.2f} *** {row['Action']}")
    else:
        print(f"{int(row['Day']):>4} ${row['Price']:>9,.2f} {ma7_str:>10} {ma30_str:>10} {signal_display:>8} {row['Position']:>12.6f} ${row['Cash']:>11,.2f} ${row['Portfolio_Value']:>11,.2f}")

# Calculate Final Performance
final_value = df.iloc[-1]['Portfolio_Value']
profit_loss = final_value - INITIAL_CASH
return_pct = (profit_loss / INITIAL_CASH) * 100

# Count trades
buy_count = len(df[df['Signal'] == 'BUY'])
sell_count = len(df[df['Signal'] == 'SELL'])

# Calculate buy-and-hold benchmark
buy_hold_btc = INITIAL_CASH / df.iloc[0]['Price']
buy_hold_value = buy_hold_btc * df.iloc[-1]['Price']
buy_hold_return = ((buy_hold_value - INITIAL_CASH) / INITIAL_CASH) * 100

print("=" * 120)
print("\nFINAL PORTFOLIO PERFORMANCE")
print("=" * 120)
print(f"Initial Investment:        ${INITIAL_CASH:,.2f}")
print(f"Final Portfolio Value:     ${final_value:,.2f}")
print(f"Profit/Loss:               ${profit_loss:,.2f}")
print(f"Return:                    {return_pct:.2f}%")
print(f"\nTotal Trades:              {buy_count + sell_count}")
print(f"  • Buy Orders:            {buy_count}")
print(f"  • Sell Orders:           {sell_count}")
print(f"\nFinal Position:")
print(f"  • Bitcoin Holdings:      {df.iloc[-1]['Position']:.6f} BTC")
print(f"  • Cash Balance:          ${df.iloc[-1]['Cash']:,.2f}")
print(f"\nBenchmark (Buy & Hold):")
print(f"  • Final Value:           ${buy_hold_value:,.2f}")
print(f"  • Return:                {buy_hold_return:.2f}%")
print(f"\nStrategy vs Benchmark:     {return_pct - buy_hold_return:+.2f}% {'(Outperformed)' if return_pct > buy_hold_return else '(Underperformed)'}")
print("=" * 120)