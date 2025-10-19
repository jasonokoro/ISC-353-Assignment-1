# The Evolution of 3-Point Shooting in NBA Scoring Predictions

## Overview

This project analyzes how the importance of 3-point attempts (3PA) in predicting points per game (PPG) has evolved across NBA seasons from 1968 to 2025. Using linear regression models trained on season-specific data, we demonstrate a statistically significant increasing trend in the predictive power of 3-point shooting over time.

## Research Question

**How has the importance of 3-point attempts in predicting a player's scoring output changed throughout NBA history?**

## Methodology

### Data

- **Dataset**: NBA Player Per-Game Statistics (1968-2025)
- **Target Variable**: Points per game (`pts_per_game`)
- **Features**:
  - `mp_per_game` - Minutes played per game
  - `fga_per_game` - Field goal attempts per game
  - `fta_per_game` - Free throw attempts per game
  - `x3pa_per_game` - 3-point attempts per game
  - `fg_percent` - Field goal percentage
  - `x3p_percent` - 3-point percentage
  - `ft_percent` - Free throw percentage
  - `ast_per_game` - Assists per game
  - `orb_per_game` - Offensive rebounds per game
  - `tov_per_game` - Turnovers per game

### Analysis Approach

1. **Season-by-Season Regression**: For each NBA season, we fitted a separate linear regression model with standardized features
2. **Coefficient Extraction**: Extracted the standardized coefficient for `x3pa_per_game` from each model
3. **Temporal Trend Analysis**: Performed a secondary linear regression on the extracted coefficients to test for temporal trends
4. **Statistical Significance Testing**: Used ordinary least squares to test the null hypothesis H₀: γ = 0 (no temporal trend)

### Model Performance

Our models demonstrated consistently excellent performance across all seasons:

- **Average R²**: 0.9907 (99.07% variance explained)
- **Average RMSE**: 0.5436 points per game
- **Average MAE**: 0.3765 points per game

## Key Findings

### 1. Significant Temporal Trend

The importance of 3-point attempts in predicting scoring has increased dramatically over time:

![3PA Importance Over Time](photos/Importance%20Tracker.png)

- **Slope**: 0.0074 (coefficient increases by 0.0074 per season)
- **p-value**: < 0.0001 (highly statistically significant)
- **R²**: 0.597 (59.7% of variance in coefficients explained by time)

### 2. Coefficient Extremes

- **Minimum Coefficient**: -0.0263 in **1984**
  - In the early years following the 3-point line's introduction (1979), 3PA had minimal or even negative predictive value
- **Maximum Coefficient**: 0.6030 in **2017**
  - Represents the peak of the "3-point revolution" era
- **Range**: 0.6292 (a 2,393% increase from minimum to maximum)

### 3. Model Consistency

Model performance remained remarkably stable across all seasons, suggesting our features consistently capture the fundamental drivers of scoring:

![Model Performance](photos/Model%20Performance.png)

The consistently high R² values (>0.97 for most seasons) indicate that our feature set effectively predicts PPG regardless of era-specific playing styles.

### 4. Feature Importance Evolution

Comparison of all feature coefficients across selected seasons reveals the dramatic shift in what drives scoring:

![Feature Importance Comparison](photos/All%20Feature%20importance.png)

**Key Observations**:

- **Field Goal Attempts** (`fga_per_game`) and **Free Throw Attempts** (`fta_per_game`) remain the strongest predictors across all eras
- **3-Point Attempts** (`x3pa_per_game`) shows the most dramatic growth, starting near zero in 1984 and becoming a major factor by 2017
- Traditional metrics like **Offensive Rebounds** and **Assists** maintain relatively stable importance
- **Shooting Percentages** show consistent but moderate predictive power across decades

## Basketball Context

### The 3-Point Revolution

Our findings quantitatively confirm the well-documented "3-point revolution" in modern basketball:

1. **1979-1990s**: The 3-point line is introduced but remains underutilized. Players and coaches view it as a specialist weapon rather than a core offensive strategy.

2. **2000s**: Gradual adoption increases as analytics demonstrate the efficiency advantage of 3-point shooting (3 points vs. 2 points for similar shot difficulty).

3. **2010s-Present**: The "Steph Curry Era" fully embraces 3-point shooting as a primary offensive weapon. Teams like the Golden State Warriors revolutionize basketball strategy, making 3-point volume central to offensive schemes.

### Statistical Interpretation

A standardized coefficient of 0.6030 (2017) means that a one-standard-deviation increase in 3PA per game corresponds to approximately 0.60 additional points per game in predicted scoring, holding all other factors constant. This is a substantial effect size, comparable to traditional high-volume scoring methods.

## Implications

### For Basketball Analytics

- **Team Building**: Modern NBA teams must prioritize 3-point shooting ability when evaluating players
- **Strategy**: Offensive systems should maximize quality 3-point attempts
- **Player Development**: Youth development programs should emphasize 3-point shooting skills

### For Statistical Modeling

- **Era Adjustment**: Historical player comparisons must account for the changing role of 3-point shooting
- **Predictive Models**: Contemporary scoring models should weight 3PA more heavily than historical data would suggest

## Technical Details

### Model Specifications

- **Algorithm**: Ordinary Least Squares Linear Regression
- **Preprocessing**: StandardScaler for feature normalization
- **Train/Test Split**: 80/20 with fixed random state (42)
- **Software**: Python 3.x with scikit-learn, pandas, numpy, matplotlib, scipy

### Reproducibility

```bash
# Install dependencies
pip install pandas numpy matplotlib scikit-learn scipy

# Run analysis
python assignment1.py
```

## Future Work

- Incorporate player position as a categorical variable
- Analyze defensive statistics' role in the 3-point era
- Compare team-level vs. individual-level trends
- Extend analysis to playoff performance metrics

## Conclusion

This analysis provides quantitative evidence for basketball's evolution toward 3-point-centric offense. The 2,393% increase in 3PA's predictive importance from 1984 to 2017, with a highly significant temporal trend (p < 0.0001), demonstrates that the 3-point revolution is not merely a stylistic preference but a fundamental transformation in how points are scored in modern basketball.

## Repository Structure

```
.
├── assignment1.py          # Main analysis script
├── Player-Per-Game.csv     # Dataset (not included)
├── README.md              # This file
└── images/
    ├── image1.png         # 3PA importance trend
    ├── image2.png         # Model performance
    └── image3.png         # Feature importance comparison
```

## Author

Data Science Project - NBA Statistical Analysis

## License

MIT License - Feel free to use and modify for educational purposes
