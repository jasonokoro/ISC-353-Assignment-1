import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from scipy import stats

# === Load Data ===
df = pd.read_csv("Player-Per-Game.csv")  # Replace with your actual file path

# === Feature Selection ===
features = [
    'mp_per_game', 'fga_per_game', 'fta_per_game', 'x3pa_per_game',
    'fg_percent', 'x3p_percent', 'ft_percent',
    'ast_per_game', 'orb_per_game', 'tov_per_game'
]
target = 'pts_per_game'

# === Preprocessing ===
df = df[['season'] + features + [target]].dropna()
for col in features + [target]:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna()

# === Containers for results ===
seasons = sorted(df['season'].unique())
x3p_importance, r2_scores, rmse_scores, mae_scores = [], [], [], []
all_coefficients = []  # Store all coefficients for each season

# === Loop through each season ===
for season in seasons:
    season_df = df[df['season'] == season]
    X = season_df[features]
    y = season_df[target]

    # Standardize inputs
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Fit model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    # Store results
    coef_dict = dict(zip(features, model.coef_))
    x3p_importance.append(coef_dict['x3pa_per_game'])
    r2_scores.append(r2)
    rmse_scores.append(rmse)
    mae_scores.append(mae)
    
    # Store all coefficients with season info
    coef_dict['season'] = season
    all_coefficients.append(coef_dict)

# === Create a summary dataframe ===
trend_df = pd.DataFrame({
    'season': seasons,
    'x3p_importance': x3p_importance,
    'r2': r2_scores,
    'rmse': rmse_scores,
    'mae': mae_scores
})

# === Print some diagnostic info ===
print("\nModel Performance by Season:")
print(trend_df[['season', 'r2', 'rmse', 'mae']].round(3))

# === Calculate and print average metrics ===
print("\n" + "="*50)
print("AVERAGE METRICS ACROSS ALL SEASONS:")
print("="*50)
print(f"Average R²:   {np.mean(r2_scores):.4f}")
print(f"Average RMSE: {np.mean(rmse_scores):.4f}")
print(f"Average MAE:  {np.mean(mae_scores):.4f}")
print("="*50)

# === Find min and max 3PA coefficient ===
min_idx = trend_df['x3p_importance'].idxmin()
max_idx = trend_df['x3p_importance'].idxmax()

min_season = trend_df.loc[min_idx, 'season']
min_value = trend_df.loc[min_idx, 'x3p_importance']

max_season = trend_df.loc[max_idx, 'season']
max_value = trend_df.loc[max_idx, 'x3p_importance']

print("\n" + "="*50)
print("3PA COEFFICIENT EXTREMES:")
print("="*50)
print(f"Minimum: {min_value:.4f} in season {min_season}")
print(f"Maximum: {max_value:.4f} in season {max_season}")
print(f"Range:   {max_value - min_value:.4f}")
print("="*50)

# === Trend significance test ===
slope, intercept, r_value, p_value, std_err = stats.linregress(trend_df['season'], trend_df['x3p_importance'])
print(f"\nTrend in 3PA importance:")
print(f"Slope = {slope:.4f}, p-value = {p_value:.4f}, R² = {r_value**2:.3f}")

# === Visualization ===
plt.figure(figsize=(10, 6))
plt.scatter(trend_df['season'], trend_df['x3p_importance'], color='blue', label='3PA Importance (coeff)')

# Highlight min and max points
plt.scatter(min_season, min_value, color='red', s=200, marker='v', 
            label=f'Min: {min_value:.4f} ({min_season})', zorder=5)
plt.scatter(max_season, max_value, color='green', s=200, marker='^', 
            label=f'Max: {max_value:.4f} ({max_season})', zorder=5)

z = np.polyfit(trend_df['season'], trend_df['x3p_importance'], 1)
p = np.poly1d(z)
plt.plot(trend_df['season'], p(trend_df['season']), "r--", label=f"Trend line (slope={z[0]:.4f})")

plt.title("Importance of 3-Point Attempts in Predicting PPG Over Time")
plt.xlabel("Season")
plt.ylabel("Standardized Coefficient for 3PA per Game")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# === Optional: Performance plot ===
plt.figure(figsize=(10, 5))
plt.plot(trend_df['season'], trend_df['r2'], marker='o', label='R²')
plt.plot(trend_df['season'], trend_df['rmse'], marker='s', label='RMSE')
plt.title("Model Performance per Season")
plt.xlabel("Season")
plt.ylabel("Score")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# === NEW: Feature Importance Comparison for Selected Years ===
# Select 10 years including min and max
all_seasons_available = sorted(seasons)

# Ensure min and max seasons are included
selected_seasons = [min_season, max_season]

# Add 8 more evenly spaced seasons
remaining_seasons = [s for s in all_seasons_available if s not in selected_seasons]
step = len(remaining_seasons) // 8
for i in range(8):
    idx = min(i * step, len(remaining_seasons) - 1)
    selected_seasons.append(remaining_seasons[idx])

selected_seasons = sorted(list(set(selected_seasons)))[:10]  # Ensure exactly 10 and sorted

# Create coefficient dataframe for selected seasons
coef_df = pd.DataFrame(all_coefficients)
selected_coef_df = coef_df[coef_df['season'].isin(selected_seasons)]

# Prepare data for grouped bar chart
feature_names = features
n_features = len(feature_names)
n_seasons = len(selected_seasons)

# Create figure
fig, ax = plt.subplots(figsize=(14, 8))

# Set up bar positions
x = np.arange(n_features)
width = 0.08  # Width of each bar
colors = plt.cm.viridis(np.linspace(0, 1, n_seasons))

# Plot bars for each season
for i, season in enumerate(selected_seasons):
    season_data = selected_coef_df[selected_coef_df['season'] == season]
    coefficients = [season_data[feat].values[0] for feat in feature_names]
    
    # Highlight min and max seasons
    if season == min_season:
        ax.bar(x + i * width, coefficients, width, label=f'{season} (Min 3PA)', 
               color='red', alpha=0.8, edgecolor='black', linewidth=1.5)
    elif season == max_season:
        ax.bar(x + i * width, coefficients, width, label=f'{season} (Max 3PA)', 
               color='green', alpha=0.8, edgecolor='black', linewidth=1.5)
    else:
        ax.bar(x + i * width, coefficients, width, label=f'{season}', 
               color=colors[i], alpha=0.8)

# Customize plot
ax.set_xlabel('Features', fontsize=12, fontweight='bold')
ax.set_ylabel('Standardized Coefficient', fontsize=12, fontweight='bold')
ax.set_title('Feature Importance Comparison Across Selected Seasons', fontsize=14, fontweight='bold')
ax.set_xticks(x + width * (n_seasons - 1) / 2)
ax.set_xticklabels(feature_names, rotation=45, ha='right')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1)
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

plt.tight_layout()
plt.show()

# Print summary table
print("\n" + "="*70)
print("COEFFICIENT VALUES FOR SELECTED SEASONS:")
print("="*70)
print(selected_coef_df[['season'] + feature_names].round(4).to_string(index=False))
print("="*70)