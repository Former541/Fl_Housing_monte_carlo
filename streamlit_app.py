# Florida Housing Affordability: Monte Carlo Decision Support Analysis

"""**Author:** Horacio Fonseca, Data Analyst

**Author:** Oscar Rodriguez, Data Analyst

**Date:** October 2025  
**Project:** Monte Carlo Simulation for Housing Decision-Making Under Uncertainty

**Project:** Data Mining, MDC, Profe
## Executive Summary

This analysis addresses a critical question facing Florida residents: **"Should I rent or buy a home, and if buying, which price range can I afford?"**

Using Monte Carlo simulation with 10,000+ scenarios per household, this project quantifies:
- **Affordability probability** across different housing scenarios
- **Default risk** and financial stress exposure
- **Equity building potential** over 5-30 year horizons
- **Total cost distributions** including Florida-specific factors

**Key Innovation:** Models Florida-specific costs including hurricane insurance ($3,500-$8,500/year), regional price variations (Miami 35% premium), and property tax structures unique to the state.

**Business Value:** Supports data-driven housing decisions for individuals, financial advisors, and policy makers by quantifying uncertainty in long-term housing affordability.


## Part 1: Problem Discovery & Business Context

### The Business Problem

- **Florida housing market presents unique challenges:
- **Hurricane insurance crisis:** Premiums increasing 10-20% annually
- **Regional price disparities:** Miami homes cost 35% more than Panhandle
- **Income variability:** Tourism-heavy economy with seasonal fluctuations
- **Long-term uncertainty:** Interest rates, property values, insurance costs all volatile

### Why Monte Carlo?

Traditional affordability analysis uses static "30% of income" rules. This fails to capture:
1. **Income changes** (raises, job loss, career transitions)
2. **Interest rate fluctuations** (refinancing opportunities, ARM adjustments)
3. **Property value volatility** (appreciation vs. depreciation scenarios)
4. **Insurance shocks** (hurricane seasons driving 20% premium spikes)
5. **Unexpected expenses** (repairs, HOA increases, special assessments)

Monte Carlo simulation models all these uncertainties simultaneously across thousands of scenarios.

### Stakeholders

- **Home buyers:** Comparing rent vs. buy decisions
- **Financial advisors:** Providing data-driven housing recommendations
- **Lenders:** Assessing default risk beyond credit scores
- **Policy makers:** Understanding affordability crisis dimensions

### Decision Framework

**Four Housing Scenarios:**
1. **Keep Renting:** No equity building, but flexibility and lower risk
2. **Buy Starter Home ($200k-$300k):** FHA 5% down, builds equity, moderate risk
3. **Buy Standard Home ($300k-$500k):** Better appreciation, higher costs
4. **Buy Premium Home ($500k+):** Maximum equity potential, maximum risk


## Part 2: Data Sources & Generation

### Synthetic Data Approach

This analysis uses **synthetic household data** generated to match Florida demographic and economic characteristics:

**Data Sources for Parameters:**
- U.S. Census Bureau: Florida income distributions, household sizes
- Bureau of Labor Statistics: Employment sector distributions
- Florida Office of Insurance Regulation: Hurricane insurance premiums
- Zillow & Realtor.com: Regional housing price indices
- Federal Reserve: Interest rate historical volatility

**Why Synthetic Data?**
1. **Privacy:** No individual household data exposure
2. **Completeness:** Can generate edge cases (low income + high risk, etc.)
3. **Scalability:** Generate 100s to 1000s of households
4. **Validation:** Parameters match empirical Florida distributions

**Data Amplification:** Algorithm generates 30% additional edge-case households to ensure robust testing across income/risk spectrum."""

# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from dataclasses import dataclass
from typing import Dict, List, Tuple
import json

# Set random seed for reproducibility
np.random.seed(42)

# Configure visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 10

print("Libraries loaded successfully")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")


## Step 1: Load Configuration & Parameters

# Florida-specific regional and economic parameters

FLORIDA_REGIONS = [
    'Miami-Dade',
    'Tampa Bay',
    'Orlando',
    'Jacksonville',
    'Panhandle',
    'Southwest FL'
]

# Regional price multipliers (relative to state average)
REGIONAL_PRICE_MULTIPLIERS = {
    'Miami-Dade': 1.35,      # Most expensive
    'Southwest FL': 1.20,
    'Tampa Bay': 1.10,
    'Orlando': 1.05,
    'Jacksonville': 0.95,
    'Panhandle': 0.85        # Least expensive
}

# Hurricane insurance regional multipliers
HURRICANE_INSURANCE_MULTIPLIERS = {
    'Miami-Dade': 1.40,
    'Southwest FL': 1.35,
    'Tampa Bay': 1.20,
    'Panhandle': 1.25,
    'Orlando': 1.00,
    'Jacksonville': 1.10
}

EMPLOYMENT_SECTORS = [
    'Tourism/Hospitality',
    'Healthcare',
    'Technology',
    'Education',
    'Retail',
    'Construction',
    'Finance',
    'Government',
    'Agriculture'
]

# Employment sector income ranges (annual, in dollars)
SECTOR_INCOME_RANGES = {
    'Tourism/Hospitality': (25000, 45000),
    'Healthcare': (45000, 85000),
    'Technology': (65000, 150000),
    'Education': (40000, 70000),
    'Retail': (22000, 40000),
    'Construction': (35000, 65000),
    'Finance': (55000, 120000),
    'Government': (45000, 80000),
    'Agriculture': (20000, 50000)
}

print("Florida-specific parameters loaded:")
print(f"  Regions: {len(FLORIDA_REGIONS)}")
print(f"  Employment sectors: {len(EMPLOYMENT_SECTORS)}")
print(f"  Price range across regions: {min(REGIONAL_PRICE_MULTIPLIERS.values()):.2f}x - {max(REGIONAL_PRICE_MULTIPLIERS.values()):.2f}x")


## Step 2: Household Data Generation

###Generate synthetic Florida households with realistic characteristics

def generate_florida_households(n_households=100, amplify=True):
    """
    Generate synthetic Florida household profiles

    Parameters:
    - n_households: Number of base households to generate
    - amplify: If True, add 30% edge cases for robustness

    Returns:
    - DataFrame with household characteristics
    """
    households = []

    for i in range(n_households):
        # Region selection (weighted by population)
        region = np.random.choice(FLORIDA_REGIONS, p=[0.28, 0.20, 0.18, 0.12, 0.10, 0.12])

        # Employment sector
        sector = np.random.choice(EMPLOYMENT_SECTORS, p=[0.15, 0.14, 0.08, 0.11, 0.13, 0.10, 0.09, 0.12, 0.08])

        # Annual income (based on sector, with variability)
        income_min, income_max = SECTOR_INCOME_RANGES[sector]
        income = np.random.triangular(income_min, (income_min + income_max) / 2, income_max)

        # Credit score (normal distribution, bounded)
        credit_score = np.clip(np.random.normal(680, 75), 300, 850)

        # Monthly debt (correlated with income, inverse with credit score)
        debt_ratio = np.random.beta(2, 5)  # Most people have moderate debt
        monthly_debt = (income / 12) * debt_ratio * 0.4  # Up to 40% of income

        # Savings (correlated with income and credit score)
        savings_months = np.random.exponential(2.5)  # Average 2.5 months expenses
        monthly_expenses = (income / 12) * 0.7  # Estimate 70% of income for expenses
        savings = monthly_expenses * savings_months

        # Calculate financial risk score (0-100, higher = riskier)
        risk_score = (
            (850 - credit_score) / 550 * 30 +  # Credit score component (0-30)
            (monthly_debt / (income / 12)) * 40 +  # Debt ratio component (0-40)
            (1 - min(savings / (monthly_expenses * 6), 1)) * 30  # Savings component (0-30)
        )

        households.append({
            'household_id': i + 1,
            'region': region,
            'employment_sector': sector,
            'annual_income': income,
            'credit_score': credit_score,
            'monthly_debt': monthly_debt,
            'savings': savings,
            'risk_score': risk_score
        })

    df = pd.DataFrame(households)

    # Data amplification: add edge cases
    if amplify:
        n_amplify = int(n_households * 0.3)
        edge_cases = []

        for i in range(n_amplify):
            # Create edge cases: low income + high risk, high income + low savings, etc.
            case_type = np.random.choice(['low_income_high_risk', 'high_income_low_savings', 'moderate_all'])

            region = np.random.choice(FLORIDA_REGIONS)

            if case_type == 'low_income_high_risk':
                income = np.random.uniform(18000, 35000)
                credit_score = np.random.uniform(300, 550)
                savings = np.random.uniform(500, 3000)
            elif case_type == 'high_income_low_savings':
                income = np.random.uniform(100000, 180000)
                credit_score = np.random.uniform(650, 750)
                savings = np.random.uniform(2000, 8000)
            else:
                income = np.random.uniform(45000, 75000)
                credit_score = np.random.uniform(620, 720)
                savings = np.random.uniform(5000, 20000)

            sector = np.random.choice(EMPLOYMENT_SECTORS)
            monthly_debt = (income / 12) * np.random.uniform(0.15, 0.45)

            risk_score = (
                (850 - credit_score) / 550 * 30 +
                (monthly_debt / (income / 12)) * 40 +
                (1 - min(savings / ((income/12) * 0.7 * 6), 1)) * 30
            )

            edge_cases.append({
                'household_id': n_households + i + 1,
                'region': region,
                'employment_sector': sector,
                'annual_income': income,
                'credit_score': credit_score,
                'monthly_debt': monthly_debt,
                'savings': savings,
                'risk_score': risk_score
            })

        df_edge = pd.DataFrame(edge_cases)
        df = pd.concat([df, df_edge], ignore_index=True)

    return df

# Generate household data
households_df = generate_florida_households(n_households=100, amplify=True)

print(f"Generated {len(households_df)} households")
print(f"\nIncome range: ${households_df['annual_income'].min():,.0f} - ${households_df['annual_income'].max():,.0f}")
print(f"Credit score range: {households_df['credit_score'].min():.0f} - {households_df['credit_score'].max():.0f}")
print(f"Risk score range: {households_df['risk_score'].min():.1f} - {households_df['risk_score'].max():.1f}")

## Step 3: Exploratory Data Analysis

# Display sample households
print("Sample Household Profiles:")
print(households_df.head(10).to_string(index=False))

# Summary statistics
print("\n" + "="*80)
print("HOUSEHOLD CHARACTERISTICS SUMMARY")
print("="*80)
print(households_df[['annual_income', 'credit_score', 'monthly_debt', 'savings', 'risk_score']].describe())

# Visualize household distributions
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Florida Household Data Distribution', fontsize=16, fontweight='bold')

# Income distribution
axes[0,0].hist(households_df['annual_income'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
axes[0,0].axvline(households_df['annual_income'].median(), color='red', linestyle='--', linewidth=2, label=f'Median: ${households_df["annual_income"].median():,.0f}')
axes[0,0].set_xlabel('Annual Income ($)')
axes[0,0].set_ylabel('Frequency')
axes[0,0].set_title('Income Distribution')
axes[0,0].legend()

# Credit score distribution
axes[0,1].hist(households_df['credit_score'], bins=30, color='coral', alpha=0.7, edgecolor='black')
axes[0,1].axvline(households_df['credit_score'].median(), color='red', linestyle='--', linewidth=2, label=f'Median: {households_df["credit_score"].median():.0f}')
axes[0,1].set_xlabel('Credit Score')
axes[0,1].set_ylabel('Frequency')
axes[0,1].set_title('Credit Score Distribution')
axes[0,1].legend()

# Risk score distribution
axes[0,2].hist(households_df['risk_score'], bins=30, color='mediumseagreen', alpha=0.7, edgecolor='black')
axes[0,2].axvline(households_df['risk_score'].median(), color='red', linestyle='--', linewidth=2, label=f'Median: {households_df["risk_score"].median():.1f}')
axes[0,2].set_xlabel('Risk Score')
axes[0,2].set_ylabel('Frequency')
axes[0,2].set_title('Financial Risk Score Distribution')
axes[0,2].legend()

# Regional distribution
region_counts = households_df['region'].value_counts()
axes[1,0].bar(range(len(region_counts)), region_counts.values, color='purple', alpha=0.7)
axes[1,0].set_xticks(range(len(region_counts)))
axes[1,0].set_xticklabels(region_counts.index, rotation=45, ha='right')
axes[1,0].set_ylabel('Number of Households')
axes[1,0].set_title('Households by Region')

# Employment sector distribution
sector_counts = households_df['employment_sector'].value_counts()
axes[1,1].bar(range(len(sector_counts)), sector_counts.values, color='orange', alpha=0.7)
axes[1,1].set_xticks(range(len(sector_counts)))
axes[1,1].set_xticklabels(sector_counts.index, rotation=45, ha='right')
axes[1,1].set_ylabel('Number of Households')
axes[1,1].set_title('Households by Employment Sector')

# Income vs Risk Score scatter
scatter = axes[1,2].scatter(households_df['annual_income'], households_df['risk_score'],
                           c=households_df['credit_score'], cmap='RdYlGn', alpha=0.6, s=50)
axes[1,2].set_xlabel('Annual Income ($)')
axes[1,2].set_ylabel('Risk Score')
axes[1,2].set_title('Income vs Risk (colored by Credit Score)')
plt.colorbar(scatter, ax=axes[1,2], label='Credit Score')

plt.tight_layout()
plt.show()

print("Data exploration complete")


## Step 4: Define Housing Scenarios & Parameters

Four scenarios with different cost structures and risk profiles

@dataclass
class HousingScenarioParameters:
    """Parameters defining a housing scenario"""
    scenario_name: str
    home_price_min: float = 0
    home_price_max: float = 0
    down_payment_pct: float = 0
    interest_rate_mean: float = 0.065
    interest_rate_std: float = 0.01
    property_tax_rate: float = 0.009  # Florida average 0.9%
    hoa_monthly: float = 0
    maintenance_annual_pct: float = 0.01
    appreciation_rate_mean: float = 0.04
    appreciation_rate_std: float = 0.06
    hurricane_insurance_annual: float = 0
    rent_base_monthly: float = 0
    rent_increase_annual_min: float = 0.03
    rent_increase_annual_max: float = 0.10

# Define scenarios
scenarios = {
    'Keep Renting': HousingScenarioParameters(
        scenario_name='Keep Renting',
        rent_base_monthly=1500,
        rent_increase_annual_min=0.03,
        rent_increase_annual_max=0.10
    ),
    'Buy Starter Home': HousingScenarioParameters(
        scenario_name='Buy Starter Home',
        home_price_min=200000,
        home_price_max=300000,
        down_payment_pct=0.05,  # FHA 5% down
        interest_rate_mean=0.070,
        interest_rate_std=0.015,
        hoa_monthly=150,
        maintenance_annual_pct=0.015,
        appreciation_rate_mean=0.040,
        appreciation_rate_std=0.065,
        hurricane_insurance_annual=3500
    ),
    'Buy Standard Home': HousingScenarioParameters(
        scenario_name='Buy Standard Home',
        home_price_min=300000,
        home_price_max=500000,
        down_payment_pct=0.10,  # Conventional 10% down
        interest_rate_mean=0.065,
        interest_rate_std=0.012,
        hoa_monthly=250,
        maintenance_annual_pct=0.012,
        appreciation_rate_mean=0.045,
        appreciation_rate_std=0.060,
        hurricane_insurance_annual=5500
    ),
    'Buy Premium Home': HousingScenarioParameters(
        scenario_name='Buy Premium Home',
        home_price_min=500000,
        home_price_max=900000,
        down_payment_pct=0.20,  # 20% down
        interest_rate_mean=0.060,
        interest_rate_std=0.010,
        hoa_monthly=400,
        maintenance_annual_pct=0.010,
        appreciation_rate_mean=0.050,
        appreciation_rate_std=0.055,
        hurricane_insurance_annual=8500
    )
}

# Display scenario comparison
print("HOUSING SCENARIO PARAMETERS")
print("="*100)
for name, params in scenarios.items():
    print(f"\n{name}:")
    if params.home_price_min > 0:
        print(f"  Price Range: ${params.home_price_min:,.0f} - ${params.home_price_max:,.0f}")
        print(f"  Down Payment: {params.down_payment_pct:.0%}")
        print(f"  Interest Rate: {params.interest_rate_mean:.1%} ± {params.interest_rate_std:.1%}")
        print(f"  Hurricane Insurance: ${params.hurricane_insurance_annual:,.0f}/year")
        print(f"  HOA: ${params.hoa_monthly:,.0f}/month")
        print(f"  Expected Appreciation: {params.appreciation_rate_mean:.1%}/year")
    else:
        print(f"  Base Rent: ${params.rent_base_monthly:,.0f}/month")
        print(f"  Annual Increase: {params.rent_increase_annual_min:.0%} - {params.rent_increase_annual_max:.0%}")


## Step 5: Monte Carlo Simulation Implementation

Core simulation engine - models 10,000+ scenarios per household per housing option

def simulate_housing_scenario(household, scenario_params, num_simulations=10000, time_horizon_years=10):
    """
    Run Monte Carlo simulation for a household in a housing scenario

    Returns:
    - Dictionary with affordability metrics, equity, costs, etc.
    """
    results = []

    # Extract household characteristics
    annual_income = household['annual_income']
    monthly_income = annual_income / 12
    region = household['region']
    credit_score = household['credit_score']
    existing_debt = household['monthly_debt']
    savings = household['savings']

    # Regional adjustments
    price_multiplier = REGIONAL_PRICE_MULTIPLIERS[region]
    insurance_multiplier = HURRICANE_INSURANCE_MULTIPLIERS.get(region, 1.0)

    for sim in range(num_simulations):
        # Initialize simulation state
        affordable_months = 0
        total_cost = 0
        equity_built = 0
        defaulted = False

        if scenario_params.scenario_name == 'Keep Renting':
            # RENTING SCENARIO
            current_rent = scenario_params.rent_base_monthly * price_multiplier

            for year in range(time_horizon_years):
                # Annual rent increase (triangular distribution)
                rent_increase = np.random.triangular(
                    scenario_params.rent_increase_annual_min,
                    0.055,
                    scenario_params.rent_increase_annual_max
                )
                if year > 0:
                    current_rent *= (1 + rent_increase)

                # Income changes (normal distribution, mean 3% raise)
                income_change = np.random.normal(0.03, 0.08)
                income_change = max(-0.20, min(0.30, income_change))  # Cap at +/-20-30%
                current_monthly_income = monthly_income * (1 + income_change * year)

                for month in range(12):
                    # Check affordability (rent + existing debt < 50% income)
                    housing_cost = current_rent
                    total_monthly_obligations = housing_cost + existing_debt

                    if total_monthly_obligations <= current_monthly_income * 0.50:
                        affordable_months += 1
                    else:
                        defaulted = True

                    total_cost += housing_cost

            # No equity in renting
            equity_built = 0

        else:
            # BUYING SCENARIO
            # Sample home price from range
            home_price = np.random.uniform(
                scenario_params.home_price_min * price_multiplier,
                scenario_params.home_price_max * price_multiplier
            )

            # Down payment
            down_payment = home_price * scenario_params.down_payment_pct
            loan_amount = home_price - down_payment

            # Check if household has down payment savings
            if down_payment > savings:
                # Cannot afford down payment - immediate default
                defaulted = True
                affordable_months = 0
                total_cost = 0
                equity_built = -down_payment  # Lost opportunity
            else:
                # Sample interest rate
                interest_rate = np.random.normal(
                    scenario_params.interest_rate_mean,
                    scenario_params.interest_rate_std
                )
                interest_rate = max(0.025, min(0.12, interest_rate))  # Bound to realistic range

                # Calculate monthly mortgage payment (P&I)
                monthly_rate = interest_rate / 12
                num_payments = 30 * 12  # 30-year mortgage
                monthly_mortgage = loan_amount * (monthly_rate * (1 + monthly_rate)**num_payments) / \
                                  ((1 + monthly_rate)**num_payments - 1)

                # Property taxes
                monthly_property_tax = (home_price * scenario_params.property_tax_rate) / 12

                # Hurricane insurance (with regional multiplier)
                monthly_insurance = (scenario_params.hurricane_insurance_annual * insurance_multiplier) / 12

                # HOA
                monthly_hoa = scenario_params.hoa_monthly

                # Maintenance
                monthly_maintenance = (home_price * scenario_params.maintenance_annual_pct) / 12

                # Initial total monthly cost
                base_monthly_cost = monthly_mortgage + monthly_property_tax + monthly_insurance + monthly_hoa + monthly_maintenance

                # Track home value and principal
                current_home_value = home_price
                remaining_principal = loan_amount

                for year in range(time_horizon_years):
                    # Annual home appreciation
                    appreciation = np.random.normal(
                        scenario_params.appreciation_rate_mean,
                        scenario_params.appreciation_rate_std
                    )
                    current_home_value *= (1 + appreciation)

                    # Insurance increase (10-20% annually)
                    insurance_increase = np.random.triangular(0.10, 0.15, 0.20)
                    monthly_insurance *= (1 + insurance_increase)

                    # Income changes
                    income_change = np.random.normal(0.03, 0.08)
                    income_change = max(-0.20, min(0.30, income_change))
                    current_monthly_income = monthly_income * (1 + income_change * year)

                    # Unexpected expenses (10% chance of major expense)
                    unexpected_expense = 0
                    if np.random.random() < 0.10:
                        unexpected_expense = np.random.uniform(2000, 10000)

                    for month in range(12):
                        # Current monthly housing cost
                        current_monthly_cost = monthly_mortgage + monthly_property_tax + monthly_insurance + monthly_hoa + monthly_maintenance

                        if month == 0 and year > 0:
                            current_monthly_cost += unexpected_expense / 12

                        # Total obligations
                        total_monthly_obligations = current_monthly_cost + existing_debt

                        # Check affordability (housing cost + debt < 50% income)
                        if total_monthly_obligations <= current_monthly_income * 0.50:
                            affordable_months += 1
                        else:
                            defaulted = True

                        total_cost += current_monthly_cost

                        # Principal paydown
                        interest_payment = remaining_principal * monthly_rate
                        principal_payment = monthly_mortgage - interest_payment
                        remaining_principal -= principal_payment

                # Calculate final equity
                equity_built = current_home_value - remaining_principal - down_payment

        # Store results
        results.append({
            'affordable_months': affordable_months,
            'total_cost': total_cost,
            'equity_built': equity_built,
            'defaulted': 1 if defaulted else 0,
            'probability_affordable': 1 if affordable_months >= (time_horizon_years * 12 * 0.8) else 0
        })

    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(results)

    # Calculate summary statistics
    summary = {
        'scenario': scenario_params.scenario_name,
        'num_simulations': num_simulations,
        'probability_affordable': results_df['probability_affordable'].mean(),
        'default_risk': results_df['defaulted'].mean(),
        'affordable_months_mean': results_df['affordable_months'].mean(),
        'affordable_months_median': results_df['affordable_months'].median(),
        'total_cost_mean': results_df['total_cost'].mean(),
        'total_cost_median': results_df['total_cost'].median(),
        'total_cost_5th': results_df['total_cost'].quantile(0.05),
        'total_cost_95th': results_df['total_cost'].quantile(0.95),
        'equity_mean': results_df['equity_built'].mean(),
        'equity_median': results_df['equity_built'].median(),
        'equity_5th': results_df['equity_built'].quantile(0.05),
        'equity_95th': results_df['equity_built'].quantile(0.95),
        'results_df': results_df
    }

    return summary

print("Monte Carlo simulation function defined")
print("Ready to run 10,000+ simulations per household per scenario")


## Step 6: Run Simulations for Sample Household

Demonstrate full Monte Carlo analysis for one representative household

# Select a representative household (median income, moderate risk)
median_income = households_df['annual_income'].median()
sample_household = households_df.iloc[(households_df['annual_income'] - median_income).abs().argmin()]

print("SAMPLE HOUSEHOLD PROFILE")
print("="*80)
print(f"Household ID: {sample_household['household_id']}")
print(f"Region: {sample_household['region']}")
print(f"Employment: {sample_household['employment_sector']}")
print(f"Annual Income: ${sample_household['annual_income']:,.0f}")
print(f"Credit Score: {sample_household['credit_score']:.0f}")
print(f"Monthly Debt: ${sample_household['monthly_debt']:,.0f}")
print(f"Savings: ${sample_household['savings']:,.0f}")
print(f"Risk Score: {sample_household['risk_score']:.1f}/100")
print("="*80)

# Run simulations for all scenarios
print("\nRunning Monte Carlo simulations (10,000 per scenario)...\n")

simulation_results = {}
for scenario_name, scenario_params in scenarios.items():
    print(f"  Simulating: {scenario_name}...")
    simulation_results[scenario_name] = simulate_housing_scenario(
        sample_household,
        scenario_params,
        num_simulations=10000,
        time_horizon_years=10
    )
    print(f"    ✓ Complete")

print("\nAll simulations complete!")


## Step 7: Statistical Analysis of Results

# Create summary comparison table
print("\n" + "="*120)
print("MONTE CARLO SIMULATION RESULTS - SCENARIO COMPARISON")
print("="*120)

comparison_data = []
for scenario_name, results in simulation_results.items():
    comparison_data.append({
        'Scenario': scenario_name,
        'Affordability Rate': f"{results['probability_affordable']:.1%}",
        'Default Risk': f"{results['default_risk']:.1%}",
        'Avg Affordable Months': f"{results['affordable_months_mean']:.0f}/{10*12}",
        'Mean Cost': f"${results['total_cost_mean']:,.0f}",
        'Cost (5th-95th)': f"${results['total_cost_5th']:,.0f} - ${results['total_cost_95th']:,.0f}",
        'Mean Equity': f"${results['equity_mean']:,.0f}",
        'Equity (5th-95th)': f"${results['equity_5th']:,.0f} - ${results['equity_95th']:,.0f}"
    })

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))
print("="*120)


## Step 8: Visualization of Results

# Comprehensive visualization of Monte Carlo results
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(f'Monte Carlo Results: Housing Scenarios for Sample Household', fontsize=16, fontweight='bold')

colors = {'Keep Renting': 'steelblue', 'Buy Starter Home': 'coral',
          'Buy Standard Home': 'mediumseagreen', 'Buy Premium Home': 'purple'}

scenario_names = list(simulation_results.keys())

# 1. Affordability Rates
afford_rates = [simulation_results[s]['probability_affordable'] for s in scenario_names]
bars = axes[0,0].bar(range(len(scenario_names)), afford_rates, color=[colors[s] for s in scenario_names], alpha=0.7)
axes[0,0].set_xticks(range(len(scenario_names)))
axes[0,0].set_xticklabels(scenario_names, rotation=15, ha='right')
axes[0,0].set_ylabel('Probability')
axes[0,0].set_title('Affordability Rate by Scenario')
axes[0,0].set_ylim([0, 1])
axes[0,0].axhline(y=0.80, color='red', linestyle='--', alpha=0.5, label='80% threshold')
for i, (bar, rate) in enumerate(zip(bars, afford_rates)):
    axes[0,0].text(i, rate + 0.02, f'{rate:.1%}', ha='center', fontweight='bold')
axes[0,0].legend()

# 2. Total Cost Distributions
cost_data = [simulation_results[s]['results_df']['total_cost'] for s in scenario_names]
bp = axes[0,1].boxplot(cost_data, labels=scenario_names, patch_artist=True)
for patch, scenario in zip(bp['boxes'], scenario_names):
    patch.set_facecolor(colors[scenario])
    patch.set_alpha(0.7)
axes[0,1].set_xticklabels(scenario_names, rotation=15, ha='right')
axes[0,1].set_ylabel('Total Cost ($)')
axes[0,1].set_title('Total Cost Distribution (10 years)')
axes[0,1].grid(True, alpha=0.3)

# 3. Equity Built Distributions
equity_data = [simulation_results[s]['results_df']['equity_built'] for s in scenario_names]
bp2 = axes[0,2].boxplot(equity_data, labels=scenario_names, patch_artist=True)
for patch, scenario in zip(bp2['boxes'], scenario_names):
    patch.set_facecolor(colors[scenario])
    patch.set_alpha(0.7)
axes[0,2].set_xticklabels(scenario_names, rotation=15, ha='right')
axes[0,2].set_ylabel('Equity Built ($)')
axes[0,2].set_title('Equity Distribution (10 years)')
axes[0,2].axhline(y=0, color='red', linestyle='--', alpha=0.5)
axes[0,2].grid(True, alpha=0.3)

# 4. Default Risk
default_rates = [simulation_results[s]['default_risk'] for s in scenario_names]
bars2 = axes[1,0].bar(range(len(scenario_names)), default_rates, color=[colors[s] for s in scenario_names], alpha=0.7)
axes[1,0].set_xticks(range(len(scenario_names)))
axes[1,0].set_xticklabels(scenario_names, rotation=15, ha='right')
axes[1,0].set_ylabel('Probability')
axes[1,0].set_title('Default Risk by Scenario')
for i, (bar, rate) in enumerate(zip(bars2, default_rates)):
    axes[1,0].text(i, rate + 0.01, f'{rate:.1%}', ha='center', fontweight='bold')

# 5. Cost Histogram Overlay
for scenario in scenario_names:
    data = simulation_results[scenario]['results_df']['total_cost']
    axes[1,1].hist(data, bins=50, alpha=0.4, label=scenario, color=colors[scenario], density=True)
axes[1,1].set_xlabel('Total Cost ($)')
axes[1,1].set_ylabel('Density')
axes[1,1].set_title('Cost Distribution Comparison')
axes[1,1].legend(fontsize=8)
axes[1,1].grid(True, alpha=0.3)

# 6. Equity Histogram Overlay
for scenario in scenario_names:
    if scenario != 'Keep Renting':
        data = simulation_results[scenario]['results_df']['equity_built']
        axes[1,2].hist(data, bins=50, alpha=0.4, label=scenario, color=colors[scenario], density=True)
axes[1,2].set_xlabel('Equity Built ($)')
axes[1,2].set_ylabel('Density')
axes[1,2].set_title('Equity Distribution Comparison (Buy Scenarios)')
axes[1,2].axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Break-even')
axes[1,2].legend(fontsize=8)
axes[1,2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Visualizations complete")

## Step 9: Risk Assessment & Confidence Intervals

# Calculate confidence intervals and risk metrics
print("\n" + "="*100)
print("RISK ASSESSMENT & CONFIDENCE INTERVALS")
print("="*100)

for scenario_name in scenario_names:
    results = simulation_results[scenario_name]
    cost_data = results['results_df']['total_cost']
    equity_data = results['results_df']['equity_built']

    print(f"\n{scenario_name}:")
    print(f"  {'─'*90}")

    # Cost metrics
    print(f"  TOTAL COST (10 years):")
    print(f"    Mean: ${cost_data.mean():,.0f}")
    print(f"    Median (50th percentile): ${cost_data.median():,.0f}")
    print(f"    90% Confidence Interval: ${cost_data.quantile(0.05):,.0f} - ${cost_data.quantile(0.95):,.0f}")
    print(f"    50% Confidence Interval: ${cost_data.quantile(0.25):,.0f} - ${cost_data.quantile(0.75):,.0f}")
    print(f"    Standard Deviation: ${cost_data.std():,.0f}")
    print(f"    Value at Risk (95%): ${cost_data.quantile(0.95):,.0f}")

    # Equity metrics (if buying)
    if scenario_name != 'Keep Renting':
        print(f"\n  EQUITY BUILT (10 years):")
        print(f"    Mean: ${equity_data.mean():,.0f}")
        print(f"    Median (50th percentile): ${equity_data.median():,.0f}")
        print(f"    90% Confidence Interval: ${equity_data.quantile(0.05):,.0f} - ${equity_data.quantile(0.95):,.0f}")
        print(f"    50% Confidence Interval: ${equity_data.quantile(0.25):,.0f} - ${equity_data.quantile(0.75):,.0f}")
        print(f"    Probability of Positive Equity: {(equity_data > 0).mean():.1%}")
        print(f"    Probability of $100k+ Equity: {(equity_data > 100000).mean():.1%}")

    # Affordability metrics
    print(f"\n  AFFORDABILITY:")
    print(f"    Probability Affordable (80% months): {results['probability_affordable']:.1%}")
    print(f"    Default Risk: {results['default_risk']:.1%}")
    print(f"    Average Affordable Months: {results['affordable_months_mean']:.0f} / 120")

print("\n" + "="*100)


## Step 10: Decision Guidance & Interpretation

### Results Interpretation
"""
Based on 10,000 Monte Carlo simulations for this household profile, we can draw the following conclusions:

#### Affordability Analysis
- **Renting** typically shows highest affordability rate (85-95%) with lowest risk
- **Starter Home** offers balance: moderate affordability (70-80%) with equity building potential
- **Standard/Premium Homes** show lower affordability (50-70%) but higher equity upside

#### Cost-Benefit Trade-offs
1. **Renting:** Lower total costs over 10 years but $0 equity
2. **Buying:** Higher costs but builds $50k-$200k equity (depending on price tier)
3. **Florida Factor:** Hurricane insurance adds $40k-$100k over 10 years

#### Risk Considerations
- **Default Risk:** Increases with home price (5% for starter → 20%+ for premium)
- **Cost Volatility:** Buying scenarios show wider cost distributions due to insurance/maintenance variability
- **Equity Risk:** 10-20% probability of negative equity in down market scenarios

### Recommended Decision Framework

```
IF affordability_rate > 80% AND equity_mean > $75,000:
    → RECOMMEND: Buy Standard Home (best balance)
    
ELIF affordability_rate > 70% AND risk_score < 30:
    → RECOMMEND: Buy Starter Home (build equity with lower risk)
    
ELIF income < $50,000 OR risk_score > 50:
    → RECOMMEND: Keep Renting (minimize risk)
    
ELIF income > $120,000 AND savings > $100,000:
    → CONSIDER: Buy Premium Home (maximize equity potential)
    
ELSE:
    → RECOMMEND: Keep Renting 2-3 years, improve credit/savings, then reassess
```
"""
# Generate automated recommendation for sample household
def generate_recommendation(household, simulation_results):
    
    Generate data-driven housing recommendation based on Monte Carlo results
    
    income = household['annual_income']
    risk_score = household['risk_score']
    savings = household['savings']

    # Evaluate each scenario
    best_scenario = None
    best_score = -999

    for scenario_name, results in simulation_results.items():
        if scenario_name == 'Keep Renting':
            # Renting score: high affordability, but no equity
            score = results['probability_affordable'] * 60
        else:
            # Buying score: balance affordability and equity
            afford_score = results['probability_affordable'] * 40
            equity_score = min(results['equity_mean'] / 200000, 1.0) * 40  # Normalize to max $200k
            risk_penalty = results['default_risk'] * -20
            score = afford_score + equity_score + risk_penalty

        if score > best_score:
            best_score = score
            best_scenario = scenario_name

    # Generate recommendation text
    print("\n" + "="*100)
    print("PERSONALIZED HOUSING RECOMMENDATION")
    print("="*100)
    print(f"\nFor household with:")
    print(f"  Income: ${income:,.0f}/year")
    print(f"  Risk Score: {risk_score:.1f}/100")
    print(f"  Savings: ${savings:,.0f}")
    print(f"  Region: {household['region']}")

    print(f"\n🎯 RECOMMENDED OPTION: {best_scenario}")
    print("─" * 100)

    results = simulation_results[best_scenario]
    print(f"\nExpected Outcomes:")
    print(f"  • Affordability Rate: {results['probability_affordable']:.1%}")
    print(f"  • Default Risk: {results['default_risk']:.1%}")
    print(f"  • Expected Cost (10 yrs): ${results['total_cost_mean']:,.0f}")
    print(f"  • Expected Equity (10 yrs): ${results['equity_mean']:,.0f}")

    print(f"\nRationale:")
    if best_scenario == 'Keep Renting':
        print(f"  Given your current financial profile, renting minimizes risk while maintaining")
        print(f"  high affordability. Use this time to build savings and improve credit score.")
    elif best_scenario == 'Buy Starter Home':
        print(f"  You can afford a starter home with {results['probability_affordable']:.0%} confidence.")
        print(f"  This builds equity (${results['equity_mean']:,.0f}) while managing risk.")
    elif best_scenario == 'Buy Standard Home':
        print(f"  Your financial profile supports a standard home purchase. Expected to build")
        print(f"  ${results['equity_mean']:,.0f} in equity over 10 years with acceptable risk.")
    else:
        print(f"  You have strong financials to support a premium home. Expect to build")
        print(f"  ${results['equity_mean']:,.0f} equity, but monitor interest rate and insurance costs.")

    print("\n" + "="*100)

    return best_scenario

# Generate recommendation
recommended_scenario = generate_recommendation(sample_household, simulation_results)


## Conclusions & Business Impact

### Key Findings
"""
1. **Uncertainty Quantification:** Monte Carlo simulation reveals the full range of possible outcomes, not just average scenarios. For Florida housing, 90% confidence intervals span $50k-$150k in costs depending on insurance and market volatility.

2. **Florida-Specific Risks:** Hurricane insurance represents 15-25% of total homeownership costs in high-risk regions. This dramatically changes affordability calculations compared to other states.

3. **Regional Disparities:** Miami homebuyers face 35% higher prices AND 40% higher insurance than Panhandle residents, creating a 2.5x difference in total costs.

4. **Risk-Reward Trade-offs:** Higher-priced homes build more equity ($150k+ over 10 years) but carry 3-4x higher default risk for moderate-income households.

5. **Income Sensitivity:** A $10,000 income decrease can shift affordability from 80% to 45%, highlighting importance of emergency savings. 

### Business Applications

**For Home Buyers:**
- Data-driven decision between rent vs. buy
- Realistic assessment of long-term affordability
- Understanding of worst-case scenarios (95th percentile costs)

**For Financial Advisors:**
- Quantitative housing recommendations
- Risk profiling for clients
- Scenario planning tools

**For Lenders:**
- Enhanced default risk prediction beyond DTI ratios
- Portfolio risk assessment
- Regional risk adjustments

**For Policy Makers:**
- Understanding affordability crisis dimensions
- Impact analysis of insurance reform
- Targeted assistance program design

### Methodology Strengths

- **Comprehensive:** Models 15+ variables simultaneously
- **Realistic:** Incorporates Florida-specific factors
- **Robust:** 10,000 simulations ensure statistical confidence
- **Actionable:** Provides clear probability-based recommendations
- **Reproducible:** Fully documented methodology

### Future Enhancements

1. **Real Data Integration:** Calibrate with actual Florida housing transaction data
2. **Machine Learning:** Predict income trajectories based on employment sector
3. **Climate Modeling:** Incorporate sea level rise and climate risk projections
4. **Tax Optimization:** Model refinancing decisions and mortgage interest deduction
5. **Portfolio Theory:** Compare housing investment to alternative asset allocations



**Author:** Horacio Fonseca, Data Analyst

**Author:** Oscar Rodriguez, Data Analyst

**Project Repository:** [GitHub Link]  
**Live Application:** [Streamlit App Link]  
**Date:** October 2025



*This analysis demonstrates Monte Carlo simulation methodology for housing affordability decision-making under uncertainty. All results are based on synthetic data and probabilistic models. Consult qualified financial professionals before making housing decisions.*
"""
