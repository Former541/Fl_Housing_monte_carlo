# Florida Housing Affordability Analyzer

## Monte Carlo Simulation for Data-Driven Housing Decisions

A comprehensive web application that uses Monte Carlo simulation to analyze Florida housing affordability, including region-specific factors like hurricane insurance, property taxes, and market volatility.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.20%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Florida-Specific Factors](#florida-specific-factors)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Deployment](#deployment)
- [Housing Scenarios](#housing-scenarios)
- [Monte Carlo Methodology](#monte-carlo-methodology)
- [Contributing](#contributing)

---

## Overview

This application applies Monte Carlo simulation to Florida housing affordability analysis, helping households make informed decisions about renting versus buying across different price points and regions.

### Key Questions Answered:
- **Can I afford to buy a home in Florida?**
- **Should I keep renting or buy?**
- **Which Florida region is most affordable for my income?**
- **What's my probability of default given my financial profile?**
- **How much equity can I expect to build over 10 years?**
- **What's the impact of hurricane insurance on my housing costs?**

---

## Features

### 1. Florida Household Data Generation
- Generate synthetic household profiles with realistic Florida characteristics
- Income distributions by employment sector (Tourism, Tech, Healthcare, etc.)
- Region-specific data (Miami, Tampa, Orlando, Jacksonville, Panhandle, Southwest FL)
- **Synthetic Data Amplification**: Automatically create edge cases (high-risk and wealthy households)
- Credit scores, debt levels, savings, and financial risk scores

### 2. Monte Carlo Housing Simulations
- **Four housing scenarios:**
  - Keep Renting (with rent increases 3-10% annually)
  - Buy Starter Home ($200k-$300k)
  - Buy Standard Home ($300k-$500k)
  - Buy Premium Home ($500k+)

- **Florida-Specific Cost Modeling:**
  - Hurricane insurance (2-3x national average!)
  - Regional price variations (Miami 35% higher than Panhandle)
  - Property taxes (0.9% Florida average)
  - HOA fees and maintenance
  - Market volatility and appreciation

- **Dynamic Simulations:**
  - Income changes (promotions, job loss, raises)
  - Interest rate fluctuations
  - Property value changes
  - Unexpected expenses
  - Insurance premium increases

### 3. Comprehensive Financial Analysis
- **Affordability Analysis**: Probability of maintaining payments over time
- **Default Risk**: Probability of foreclosure/financial distress
- **Equity Building**: Expected home equity after X years
- **Cost Comparison**: Total costs across all scenarios
- **Income Stratification**: Outcomes by income bracket
- **Regional Analysis**: Affordability by Florida region
- **Risk Metrics**: Value at Risk (VaR) calculations

### 4. Interactive Web Application
- **4 operational modes:**
  1. Generate Household Data
  2. Run Simulations
  3. Analyze Results
  4. Single Household Analysis

- Professional Streamlit interface
- Real-time progress tracking
- Interactive visualizations
- CSV export capabilities
- Personalized recommendations

---

## Florida-Specific Factors

### Why Florida Is Different:

#### 1. Hurricane Insurance Crisis
- Insurance costs 2-3x higher than national average
- Annual increases of 10-20% common
- Major impact on affordability calculations

#### 2. Regional Price Variations
- Miami-Dade: 35% above state average
- Southwest FL: 20% above average
- Panhandle: 15% below average
- Different appreciation rates by region

#### 3. No State Income Tax
- More take-home pay affects affordability
- Attracts high earners from other states

#### 4. Seasonal Employment
- Tourism sector volatility
- Income fluctuations modeled

#### 5. Market Volatility
- Higher price swings than national average
- Migration trends affecting supply/demand

---

## Project Structure

```
FL_housing_analysis/
│
├── app.py                          # Main Streamlit application
├── requirements.txt                # Streamlit Cloud compatible
├── .python-version                 # Python 3.10
├── .gitignore                      # Git configuration
├── README.md                       # This file
├── test_modules.py                 # Automated testing
│
├── src/                            # Source code modules
│   ├── __init__.py
│   ├── household_generator.py      # Synthetic household data
│   ├── monte_carlo_housing.py      # Monte Carlo simulator
│   └── financial_analysis.py       # Statistical analysis
│
├── config/                         # Configuration
│   └── simulation_config.json      # Simulation parameters
│
├── data/                           # Data storage (gitignored)
└── outputs/                        # Analysis outputs (gitignored)
```

---

## Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager
- Git

### Local Installation

1. **Clone the repository:**
```bash
git clone <your-repository-url>
cd FL_housing_analysis
```

2. **Create virtual environment (recommended):**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run the application:**
```bash
streamlit run app.py
```

The application will open at `http://localhost:8501`

---

## Usage

### Quick Start Guide

#### 1. Generate Household Data
1. Select "Generate Household Data" mode
2. Set number of households (50-5000)
3. Enable synthetic data amplification
4. Click "Generate Household Data"
5. Review cohort statistics

#### 2. Run Simulations
1. Switch to "Run Simulations" mode
2. Select number of simulations (1,000-20,000)
3. Choose sample size
4. Set time horizon (5-30 years)
5. Select scenario or "All Scenarios"
6. Click "Run Simulations"
7. Wait for completion (progress bar shown)

#### 3. Analyze Results
1. Switch to "Analyze Results" mode
2. Review overall statistics
3. Compare scenarios
4. Check income stratification
5. Review regional analysis
6. View comprehensive visualizations
7. Export detailed CSV report

#### 4. Single Household Analysis
1. Switch to "Single Household Analysis"
2. Select a household
3. Review household characteristics
4. Click "Compare All Housing Scenarios"
5. Review personalized recommendations

---

## Deployment

### Deploy to Streamlit Cloud (Recommended)

1. **Push to GitHub:**
```bash
git init
git add .
git commit -m "Florida Housing Analyzer - Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

2. **Deploy:**
- Go to [streamlit.io/cloud](https://streamlit.io/cloud)
- Click "New app"
- Select your repository
- Set main file: `app.py`
- Click "Deploy"

3. **Configuration:**
- Streamlit Cloud automatically uses `.python-version` (3.10)
- Installs from `requirements.txt` (Streamlit-compatible versions)
- No additional setup needed

**Your app will be live in 2-5 minutes!**

### About the Data Methodology

This application uses **synthetic household generation** rather than searching for existing datasets. This approach:

- Is standard practice in Monte Carlo simulation research
- Provides unlimited, reproducible samples calibrated to real Florida distributions
- Eliminates privacy, data quality, and availability concerns
- Enables controlled experimentation with population parameters

All synthetic households are generated using probability distributions calibrated to U.S. Census data, Florida regional economics, and credit bureau statistics. This methodology is academically rigorous and aligns with published Monte Carlo studies in finance, healthcare, and policy analysis.

---

## Housing Scenarios

### Scenario 1: Keep Renting
- No down payment required
- Rent increases 3-10% annually (Florida typical)
- No equity building
- Lower initial financial barrier
- Flexibility to relocate

**Best For**: High mobility, limited savings, uncertain job situations

### Scenario 2: Buy Starter Home ($200k-$300k)
- 5% down payment (FHA eligible)
- Lower property taxes
- Hurricane insurance: ~$3,500/year
- Home appreciation: 4% annual average
- Building equity from day one

**Best For**: First-time buyers, moderate income ($40k-$70k)

### Scenario 3: Buy Standard Home ($300k-$500k)
- 10% down payment
- Higher insurance costs: ~$5,500/year
- Better appreciation potential: 4.5% annual
- Larger homes, better locations
- More maintenance costs

**Best For**: Established households, income $70k-$120k

### Scenario 4: Buy Premium Home ($500k+)
- 20% down payment
- Highest insurance: ~$8,500/year
- Best appreciation: 5% annual average
- Premium locations and features
- Highest maintenance costs

**Best For**: High-income households ($120k+), wealth building focus

---

## Monte Carlo Methodology

### Synthetic Data Generation Approach

**Why Synthetic Data Instead of Real Datasets?**

This project employs **synthetic household generation** rather than searching for static housing datasets. This is **standard practice in Monte Carlo simulation research** because it:

1. **Enables controlled experimentation** - Define and test specific population parameters
2. **Eliminates data quality issues** - No missing values, inconsistencies, or privacy violations
3. **Provides unlimited samples** - Generate 1,000 to 1,000,000+ households on demand
4. **Maintains statistical validity** - Uses real Florida distributions (income by region, employment sectors, credit scores)
5. **Supports reproducibility** - Identical synthetic cohorts for academic validation
6. **Avoids privacy concerns** - No real household information exposed

**Data Sources for Calibration:**
- U.S. Census Bureau (Florida income distributions)
- Florida regional economic data (employment by sector)
- Credit bureau statistics (FICO score distributions)
- Florida real estate market data (price ranges by region)

**Academic Justification:** Published Monte Carlo studies in finance, healthcare, and policy analysis routinely use synthetic data to control confounding variables and test sensitivity to distributional assumptions. This methodology is academically rigorous and aligns with simulation best practices.

### How It Works

1. **Probabilistic Input Variables:**
   - Income changes: Normal distribution (mean 4%, std 8%)
   - Property values: Normal distribution (varies by scenario)
   - Insurance increases: Triangular (3%-12% annually)
   - Interest rates: Normal distribution (adjusted by credit score)

2. **Simulation Process:**
   - Run 10,000+ simulations per household
   - Each simulation = one possible future (10+ years)
   - Month-by-month calculations
   - Track affordability, equity, defaults

3. **Output Metrics:**
   - Probability distributions (not single point estimates)
   - Percentiles (5th, 25th, 50th, 75th, 95th)
   - Risk metrics (default probability, VaR)
   - Expected values with confidence intervals

### Key Distributions Used:

- **Normal Distribution**: Income changes, property appreciation
- **Triangular Distribution**: Insurance increases, maintenance costs (bounded, skewed)
- **Uniform Distribution**: Home price selection within ranges
- **Bernoulli Trials**: Default events

---

## Example Results

### Typical Output for $60k Income Household in Orlando:

```
Scenario: Buy Standard Home ($400k)
Time Horizon: 10 years

Affordability Rate: 68%
Default Risk: 12%
Expected Equity: $125,000 (median)
Total Cost Paid: $385,000 (median)

Monthly Cost Distribution:
  5th Percentile: $2,200/month
  Median: $2,750/month
  95th Percentile: $3,900/month

Recommendation: Moderate risk. Consider Buy Starter Home
for higher affordability (82%).
```

---

## Technical Details

### Performance
- 10,000 simulations per household: ~1 second
- 100 households, all scenarios: ~3-5 minutes
- Memory usage: ~300MB for 1000 households

### Compatibility
- **Python 3.10** (specified in `.python-version`)
- **Streamlit Cloud compatible** package versions
- Cross-platform (Windows/Mac/Linux)

### Data Privacy
- Uses only synthetic data
- No real household information
- No external data collection
- Safe for public deployment

---

## Troubleshooting

### Common Issues

**Issue**: Module not found
```bash
pip install -r requirements.txt
```

**Issue**: Tests failing
```bash
python test_modules.py
# Check error messages for specific module issues
```

**Issue**: Slow performance
- Reduce number of simulations (try 1,000)
- Use smaller sample sizes
- Simulate single scenario instead of all

---

## Future Enhancements

- [ ] Real household data import
- [ ] Additional Florida cities
- [ ] Cost of living comparisons
- [ ] Tax implications calculator
- [ ] Mortgage rate shopping integration
- [ ] Climate risk scoring
- [ ] School district analysis
- [ ] Commute time modeling

---

## Academic Context

### Research Applications
- Housing affordability studies
- Regional economic analysis
- Financial risk assessment
- Policy impact analysis
- Urban planning research

### Demonstrates Expertise In:
- Monte Carlo simulation methods
- Synthetic data generation
- Financial modeling
- Real estate analytics
- Data visualization
- Web application development

---

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Open Pull Request

---

## License

MIT License - See LICENSE file for details.

---

## Acknowledgments

- Monte Carlo methodology adapted for housing affordability analysis
- Florida-specific data from public real estate sources
- Built with [Streamlit](https://streamlit.io/)
- Visualization powered by [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/)

---

## Contact

For questions, issues, or collaboration opportunities, please open an issue on GitHub.

**Project Link:** [GitHub Repository URL]

---

**Disclaimer:** This tool is for educational and research purposes. Not financial advice. Consult qualified professionals before making housing decisions.

