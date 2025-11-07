# Florida Housing Affordability: Monte Carlo Decision Support Analysis (Streamlit-ready)
# Adapted from original script to run inside Streamlit with safeguards and UI controls.
# Author: Horacio Fonseca / Oscar Rodriguez (original)
# Modifications: Streamlit UI, caching, progress/controls, safe defaults, plotting via st.pyplot

"""
Florida Housing Affordability: Monte Carlo Decision Support Analysis
Author: Horacio Fonseca, Data Analyst

Author: Oscar Rodriguez, Data Analyst

Date: October 2025
Project: Monte Carlo Simulation for Housing Decision-Making Under Uncertainty

Project: Data Mining, MDC, Professor Ernesto Lee.
Executive Summary
This analysis addresses a critical question facing Florida residents: "Should I rent or buy a home, and if buying, which price range can I afford?"

Using Monte Carlo simulation with 10,000+ scenarios per household, this project quantifies:

Affordability probability across different housing scenarios
Default risk and financial stress exposure
Equity building potential over 5-30 year horizons
Total cost distributions including Florida-specific factors
Key Innovation: Models Florida-specific costs including hurricane insurance ( 3,500− 8,500/year), regional price variations (Miami 35% premium), and property tax structures unique to the state.

Business Value: Supports data-driven housing decisions for individuals, financial advisors, and policy makers by quantifying uncertainty in long-term housing affordability.

Part 1: Problem Discovery & Business Context
The Business Problem
Florida's housing market presents unique challenges:

Hurricane insurance crisis: Premiums increasing 10-20% annually
Regional price disparities: Miami homes cost 35% more than Panhandle
Income variability: Tourism-heavy economy with seasonal fluctuations
Long-term uncertainty: Interest rates, property values, insurance costs all volatile
Why Monte Carlo?
Traditional affordability analysis uses static "30% of income" rules. This fails to capture:

Income changes (raises, job loss, career transitions)
Interest rate fluctuations (refinancing opportunities, ARM adjustments)
Property value volatility (appreciation vs. depreciation scenarios)
Insurance shocks (hurricane seasons driving 20% premium spikes)
Unexpected expenses (repairs, HOA increases, special assessments)
Monte Carlo simulation models all these uncertainties simultaneously across thousands of scenarios.

Stakeholders
Home buyers: Comparing rent vs. buy decisions
Financial advisors: Providing data-driven housing recommendations
Lenders: Assessing default risk beyond credit scores
Policy makers: Understanding affordability crisis dimensions
Decision Framework
Four Housing Scenarios:

Keep Renting: No equity building, but flexibility and lower risk
Buy Starter Home (200k−300k): FHA 5% down, builds equity, moderate risk
Buy Standard Home (300k−500k): Better appreciation, higher costs
Buy Premium Home ($500k+): Maximum equity potential, maximum risk
Part 2: Data Sources & Generation
Synthetic Data Approach
This analysis uses synthetic household data generated to match Florida demographic and economic characteristics:

Data Sources for Parameters:

U.S. Census Bureau: Florida income distributions, household sizes
Bureau of Labor Statistics: Employment sector distributions
Florida Office of Insurance Regulation: Hurricane insurance premiums
Zillow & Realtor.com: Regional housing price indices
Federal Reserve: Interest rate historical volatility
Why Synthetic Data?

Privacy: No individual household data exposure
Completeness: Can generate edge cases (low income + high risk, etc.)
Scalability: Generate 100s to 1000s of households
Validation: Parameters match empirical Florida distributions
Data Amplification: Algorithm generates 30% additional edge-case households to ensure robust testing across income/risk spectrum."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from dataclasses import dataclass
from typing import Dict, List, Tuple
import streamlit as st

# Set random seed for reproducibility (module-level)
RNG_SEED = 42
np.random.seed(RNG_SEED)

# Configure visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 10

# Florida-specific regional and economic parameters
FLORIDA_REGIONS = [
    'Miami-Dade',
    'Tampa Bay',
    'Orlando',
    'Jacksonville',
    'Panhandle',
    'Southwest FL'
]

REGIONAL_PRICE_MULTIPLIERS = {
    'Miami-Dade': 1.35,
    'Southwest FL': 1.20,
    'Tampa Bay': 1.10,
    'Orlando': 1.05,
    'Jacksonville': 0.95,
    'Panhandle': 0.85
}

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

@dataclass
class HousingScenarioParameters:
    scenario_name: str
    home_price_min: float = 0
    home_price_max: float = 0
    down_payment_pct: float = 0
    interest_rate_mean: float = 0.065
    interest_rate_std: float = 0.01
    property_tax_rate: float = 0.009
    hoa_monthly: float = 0
    maintenance_annual_pct: float = 0.01
    appreciation_rate_mean: float = 0.04
    appreciation_rate_std: float = 0.06
    hurricane_insurance_annual: float = 0
    rent_base_monthly: float = 0
    rent_increase_annual_min: float = 0.03
    rent_increase_annual_max: float = 0.10

# Define scenarios (kept original parameters)
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
        down_payment_pct=0.05,
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
        down_payment_pct=0.10,
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
        down_payment_pct=0.20,
        interest_rate_mean=0.060,
        interest_rate_std=0.010,
        hoa_monthly=400,
        maintenance_annual_pct=0.010,
        appreciation_rate_mean=0.050,
        appreciation_rate_std=0.055,
        hurricane_insurance_annual=8500
    )
}

# --- Utility functions ---

@st.cache_data(show_spinner=False)
def generate_florida_households(n_households=100, amplify=True, seed=RNG_SEED):
    """
    Generate synthetic Florida household profiles.
    Cached to avoid re-generation for the same input parameters.
    """
    rng = np.random.RandomState(seed)
    households = []

    for i in range(n_households):
        region = rng.choice(FLORIDA_REGIONS, p=[0.28, 0.20, 0.18, 0.12, 0.10, 0.12])
        sector = rng.choice(EMPLOYMENT_SECTORS, p=[0.15, 0.14, 0.08, 0.11, 0.13, 0.10, 0.09, 0.12, 0.08])
        income_min, income_max = SECTOR_INCOME_RANGES[sector]
        income = rng.triangular(income_min, (income_min + income_max) / 2, income_max)
        credit_score = np.clip(rng.normal(680, 75), 300, 850)
        debt_ratio = rng.beta(2, 5)
        monthly_debt = (income / 12) * debt_ratio * 0.4
        savings_months = rng.exponential(2.5)
        monthly_expenses = (income / 12) * 0.7
        savings = monthly_expenses * savings_months

        risk_score = (
            (850 - credit_score) / 550 * 30 +
            (monthly_debt / (income / 12)) * 40 +
            (1 - min(savings / (monthly_expenses * 6), 1)) * 30
        )

        households.append({
            'household_id': i + 1,
            'region': region,
            'employment_sector': sector,
            'annual_income': float(income),
            'credit_score': float(credit_score),
            'monthly_debt': float(monthly_debt),
            'savings': float(savings),
            'risk_score': float(risk_score)
        })

    df = pd.DataFrame(households)

    if amplify:
        n_amplify = int(n_households * 0.3)
        edge_cases = []
        for i in range(n_amplify):
            case_type = rng.choice(['low_income_high_risk', 'high_income_low_savings', 'moderate_all'])
            region = rng.choice(FLORIDA_REGIONS)

            if case_type == 'low_income_high_risk':
                income = rng.uniform(18000, 35000)
                credit_score = rng.uniform(300, 550)
                savings = rng.uniform(500, 3000)
            elif case_type == 'high_income_low_savings':
                income = rng.uniform(100000, 180000)
                credit_score = rng.uniform(650, 750)
                savings = rng.uniform(2000, 8000)
            else:
                income = rng.uniform(45000, 75000)
                credit_score = rng.uniform(620, 720)
                savings = rng.uniform(5000, 20000)

            sector = rng.choice(EMPLOYMENT_SECTORS)
            monthly_debt = (income / 12) * rng.uniform(0.15, 0.45)

            risk_score = (
                (850 - credit_score) / 550 * 30 +
                (monthly_debt / (income / 12)) * 40 +
                (1 - min(savings / ((income/12) * 0.7 * 6), 1)) * 30
            )

            edge_cases.append({
                'household_id': n_households + i + 1,
                'region': region,
                'employment_sector': sector,
                'annual_income': float(income),
                'credit_score': float(credit_score),
                'monthly_debt': float(monthly_debt),
                'savings': float(savings),
                'risk_score': float(risk_score)
            })

        df_edge = pd.DataFrame(edge_cases)
        df = pd.concat([df, df_edge], ignore_index=True)

    return df

def simulate_housing_scenario(household, scenario_params, num_simulations=1000, time_horizon_years=10, seed=None, show_progress=False):
    """
    Run Monte Carlo simulation for a household in a housing scenario.

    Returns summary dict with statistics and a DataFrame 'results_df' containing all simulations.
    This function intentionally keeps the original modeling logic but is defensive against edge cases.
    """
    if seed is None:
        seed = RNG_SEED
    rng = np.random.RandomState(seed)

    results = []
    annual_income = float(household['annual_income'])
    monthly_income = annual_income / 12
    region = household['region']
    credit_score = float(household['credit_score'])
    existing_debt = float(household['monthly_debt'])
    savings = float(household['savings'])

    price_multiplier = REGIONAL_PRICE_MULTIPLIERS.get(region, 1.0)
    insurance_multiplier = HURRICANE_INSURANCE_MULTIPLIERS.get(region, 1.0)

    # Progress helper
    progress_bar = None
    if show_progress:
        progress_bar = st.progress(0)

    for sim in range(int(num_simulations)):
        affordable_months = 0
        total_cost = 0.0
        equity_built = 0.0
        defaulted = False

        try:
            if scenario_params.scenario_name == 'Keep Renting':
                current_rent = scenario_params.rent_base_monthly * price_multiplier

                for year in range(int(time_horizon_years)):
                    rent_increase = rng.triangular(
                        scenario_params.rent_increase_annual_min,
                        0.055,
                        scenario_params.rent_increase_annual_max
                    )
                    if year > 0:
                        current_rent *= (1 + rent_increase)

                    income_change = rng.normal(0.03, 0.08)
                    income_change = max(-0.20, min(0.30, income_change))
                    current_monthly_income = monthly_income * (1 + income_change * year)

                    for month in range(12):
                        housing_cost = current_rent
                        total_monthly_obligations = housing_cost + existing_debt

                        if total_monthly_obligations <= current_monthly_income * 0.50:
                            affordable_months += 1
                        else:
                            defaulted = True

                        total_cost += housing_cost

                equity_built = 0.0

            else:
                home_price = rng.uniform(
                    scenario_params.home_price_min * price_multiplier,
                    scenario_params.home_price_max * price_multiplier
                )
                down_payment = home_price * scenario_params.down_payment_pct
                loan_amount = home_price - down_payment

                if down_payment > savings:
                    defaulted = True
                    affordable_months = 0
                    total_cost = 0.0
                    equity_built = -down_payment
                else:
                    interest_rate = rng.normal(
                        scenario_params.interest_rate_mean,
                        scenario_params.interest_rate_std
                    )
                    interest_rate = max(0.025, min(0.12, interest_rate))

                    monthly_rate = interest_rate / 12.0
                    num_payments = 30 * 12
                    if monthly_rate == 0:
                        monthly_mortgage = loan_amount / num_payments
                    else:
                        monthly_mortgage = loan_amount * (monthly_rate * (1 + monthly_rate)**num_payments) / \
                                           ((1 + monthly_rate)**num_payments - 1)

                    monthly_property_tax = (home_price * scenario_params.property_tax_rate) / 12.0
                    monthly_insurance = (scenario_params.hurricane_insurance_annual * insurance_multiplier) / 12.0
                    monthly_hoa = scenario_params.hoa_monthly
                    monthly_maintenance = (home_price * scenario_params.maintenance_annual_pct) / 12.0

                    current_home_value = home_price
                    remaining_principal = loan_amount

                    for year in range(int(time_horizon_years)):
                        appreciation = rng.normal(
                            scenario_params.appreciation_rate_mean,
                            scenario_params.appreciation_rate_std
                        )
                        current_home_value *= (1 + appreciation)

                        insurance_increase = rng.triangular(0.10, 0.15, 0.20)
                        monthly_insurance *= (1 + insurance_increase)

                        income_change = rng.normal(0.03, 0.08)
                        income_change = max(-0.20, min(0.30, income_change))
                        current_monthly_income = monthly_income * (1 + income_change * year)

                        unexpected_expense = 0.0
                        if rng.random() < 0.10:
                            unexpected_expense = rng.uniform(2000, 10000)

                        for month in range(12):
                            current_monthly_cost = monthly_mortgage + monthly_property_tax + monthly_insurance + monthly_hoa + monthly_maintenance

                            # Add unexpected expense (spread across months of the year)
                            if unexpected_expense > 0:
                                current_monthly_cost += unexpected_expense / 12.0

                            total_monthly_obligations = current_monthly_cost + existing_debt

                            if total_monthly_obligations <= current_monthly_income * 0.50:
                                affordable_months += 1
                            else:
                                defaulted = True

                            total_cost += current_monthly_cost

                            interest_payment = remaining_principal * monthly_rate
                            principal_payment = monthly_mortgage - interest_payment
                            remaining_principal = max(0.0, remaining_principal - principal_payment)

                    equity_built = current_home_value - remaining_principal - down_payment

        except Exception as e:
            # If an unexpected error occurs in a simulation, record as failure for that sim and continue.
            defaulted = True
            affordable_months = 0
            total_cost = 0.0
            equity_built = -1.0

        results.append({
            'affordable_months': int(affordable_months),
            'total_cost': float(total_cost),
            'equity_built': float(equity_built),
            'defaulted': 1 if defaulted else 0,
            'probability_affordable': 1 if affordable_months >= (time_horizon_years * 12 * 0.8) else 0
        })

        if show_progress and progress_bar and (sim % max(1, int(num_simulations / 100))) == 0:
            progress_bar.progress(int((sim + 1) / num_simulations * 100))

    results_df = pd.DataFrame(results)

    summary = {
        'scenario': scenario_params.scenario_name,
        'num_simulations': int(num_simulations),
        'probability_affordable': float(results_df['probability_affordable'].mean()),
        'default_risk': float(results_df['defaulted'].mean()),
        'affordable_months_mean': float(results_df['affordable_months'].mean()),
        'affordable_months_median': float(results_df['affordable_months'].median()),
        'total_cost_mean': float(results_df['total_cost'].mean()),
        'total_cost_median': float(results_df['total_cost'].median()),
        'total_cost_5th': float(results_df['total_cost'].quantile(0.05)),
        'total_cost_95th': float(results_df['total_cost'].quantile(0.95)),
        'equity_mean': float(results_df['equity_built'].mean()),
        'equity_median': float(results_df['equity_built'].median()),
        'equity_5th': float(results_df['equity_built'].quantile(0.05)),
        'equity_95th': float(results_df['equity_built'].quantile(0.95)),
        'results_df': results_df
    }

    return summary

def generate_recommendation(household, simulation_results):
    """
    Generate data-driven housing recommendation based on Monte Carlo results.
    This function only formats and returns results (no Streamlit calls inside).
    """
    income = float(household['annual_income'])
    risk_score = float(household['risk_score'])
    savings = float(household['savings'])

    best_scenario = None
    best_score = -999.0

    for scenario_name, results in simulation_results.items():
        if scenario_name == 'Keep Renting':
            score = results['probability_affordable'] * 60
        else:
            afford_score = results['probability_affordable'] * 40
            equity_score = min(results['equity_mean'] / 200000.0, 1.0) * 40
            risk_penalty = results['default_risk'] * -20.0
            score = afford_score + equity_score + risk_penalty

        if score > best_score:
            best_score = score
            best_scenario = scenario_name

    rationale = ""
    results = simulation_results[best_scenario]

    if best_scenario == 'Keep Renting':
        rationale = (
            "Given your current financial profile, renting minimizes risk while maintaining "
            "high affordability. Use this time to build savings and improve credit score."
        )
    elif best_scenario == 'Buy Starter Home':
        rationale = (
            f"You can afford a starter home with {results['probability_affordable']:.0%} confidence. "
            f"This builds equity (${results['equity_mean']:,.0f}) while managing risk."
        )
    elif best_scenario == 'Buy Standard Home':
        rationale = (
            f"Your financial profile supports a standard home purchase. Expected to build "
            f"${results['equity_mean']:,.0f} in equity over the specified horizon with acceptable risk."
        )
    else:
        rationale = (
            f"You have strong financials to support a premium home. Expect to build "
            f"${results['equity_mean']:,.0f} equity, but monitor interest rate and insurance costs."
        )

    recommendation = {
        'recommended_option': best_scenario,
        'expected_affordability': results['probability_affordable'],
        'expected_default_risk': results['default_risk'],
        'expected_cost': results['total_cost_mean'],
        'expected_equity': results['equity_mean'],
        'rationale': rationale
    }

    return recommendation

# --- Streamlit App Layout ---

st.set_page_config(page_title="Florida Housing Monte Carlo", layout="wide")

st.title("Florida Housing Affordability — Monte Carlo Decision Support")
st.markdown(
    """
    This app runs Monte Carlo simulations to compare renting vs. buying (starter / standard / premium)
    for synthetic Florida households. Results are probabilistic and use synthetic data — not financial advice.
    """
)

with st.sidebar:
    st.header("Simulation Controls")
    num_households = st.number_input("Number of synthetic households (base)", min_value=1, max_value=2000, value=100, step=1)
    amplify = st.checkbox("Add 30% edge-case households (amplify)", value=True)
    num_simulations = st.slider("Monte Carlo simulations per scenario (each household)", 100, 20000, 1000, step=100)
    time_horizon_years = st.slider("Time horizon (years)", 1, 30, 10)
    sample_household_id = st.number_input("Select sample household ID (after generation)", min_value=1, value=1, step=1)
    run_button = st.button("Generate households and run simulations")

st.info("Defaults are conservative to keep run-time reasonable. Increase simulations/time horizon for higher accuracy but expect longer runtimes.")

if run_button:
    try:
        with st.spinner("Generating households..."):
            households_df = generate_florida_households(n_households=int(num_households), amplify=bool(amplify), seed=RNG_SEED)
        st.success(f"Generated {len(households_df)} households")

        # Show sample of households
        st.subheader("Sample of generated households")
        st.dataframe(households_df.head(50))

        # Choose a representative household
        if sample_household_id <= len(households_df):
            sample_household = households_df.loc[households_df['household_id'] == sample_household_id].iloc[0]
        else:
            # If requested id out of range, pick median income household
            median_income = households_df['annual_income'].median()
            idx = (households_df['annual_income'] - median_income).abs().idxmin()
            sample_household = households_df.loc[idx]

        st.subheader("Selected household profile")
        st.json({
            'household_id': int(sample_household['household_id']),
            'region': sample_household['region'],
            'employment_sector': sample_household['employment_sector'],
            'annual_income': float(sample_household['annual_income']),
            'credit_score': float(sample_household['credit_score']),
            'monthly_debt': float(sample_household['monthly_debt']),
            'savings': float(sample_household['savings']),
            'risk_score': float(sample_household['risk_score'])
        })

        # Run simulations for the selected household across all scenarios
        simulation_results = {}
        st.subheader("Running Monte Carlo simulations")
        overall_progress = st.progress(0)
        total_tasks = len(scenarios)
        task_index = 0

        for scenario_name, scenario_params in scenarios.items():
            task_index += 1
            status_text = st.empty()
            status_text.info(f"Simulating: {scenario_name} ({task_index}/{total_tasks}) — {num_simulations} sims, {time_horizon_years} yrs")

            # Run simulation (show per-scenario progress inside)
            summary = simulate_housing_scenario(
                sample_household,
                scenario_params,
                num_simulations=int(num_simulations),
                time_horizon_years=int(time_horizon_years),
                seed=RNG_SEED + task_index,
                show_progress=False  # internal progress can be expensive in st; keep it off
            )
            simulation_results[scenario_name] = summary
            overall_progress.progress(int(task_index / total_tasks * 100))
            status_text.success(f"Complete: {scenario_name}")

        st.success("All scenario simulations complete")

        # Present comparison table
        comparison_data = []
        for scenario_name, results in simulation_results.items():
            comparison_data.append({
                'Scenario': scenario_name,
                'Affordability Rate': f"{results['probability_affordable']:.1%}",
                'Default Risk': f"{results['default_risk']:.1%}",
                'Avg Affordable Months': f"{results['affordable_months_mean']:.0f}/{time_horizon_years*12}",
                'Mean Cost': f"${results['total_cost_mean']:,.0f}",
                'Cost (5th-95th)': f"${results['total_cost_5th']:,.0f} - ${results['total_cost_95th']:,.0f}",
                'Mean Equity': f"${results['equity_mean']:,.0f}",
                'Equity (5th-95th)': f"${results['equity_5th']:,.0f} - ${results['equity_95th']:,.0f}"
            })

        comparison_df = pd.DataFrame(comparison_data)
        st.subheader("Scenario comparison (summary)")
        st.table(comparison_df)

        # Visualizations: affordability, cost distribution, equity distribution, default risk
        st.subheader("Visualizations")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        colors = {'Keep Renting': 'steelblue', 'Buy Starter Home': 'coral',
                  'Buy Standard Home': 'mediumseagreen', 'Buy Premium Home': 'purple'}
        scenario_names = list(simulation_results.keys())

        # Affordability rates
        afford_rates = [simulation_results[s]['probability_affordable'] for s in scenario_names]
        axes[0, 0].bar(range(len(scenario_names)), afford_rates, color=[colors[s] for s in scenario_names], alpha=0.8)
        axes[0, 0].set_xticks(range(len(scenario_names)))
        axes[0, 0].set_xticklabels(scenario_names, rotation=15, ha='right')
        axes[0, 0].set_ylabel('Probability')
        axes[0, 0].set_title('Affordability Rate by Scenario')
        axes[0, 0].set_ylim([0, 1])

        # Default risk
        default_rates = [simulation_results[s]['default_risk'] for s in scenario_names]
        axes[0, 1].bar(range(len(scenario_names)), default_rates, color=[colors[s] for s in scenario_names], alpha=0.8)
        axes[0, 1].set_xticks(range(len(scenario_names)))
        axes[0, 1].set_xticklabels(scenario_names, rotation=15, ha='right')
        axes[0, 1].set_ylabel('Probability')
        axes[0, 1].set_title('Default Risk by Scenario')
        axes[0, 1].set_ylim([0, 1])

        # Cost distributions (overlay histograms)
        for scenario in scenario_names:
            data = simulation_results[scenario]['results_df']['total_cost']
            axes[1, 0].hist(data, bins=50, alpha=0.4, label=scenario, color=colors[scenario], density=True)
        axes[1, 0].set_xlabel('Total Cost ($)')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Cost Distribution Comparison')
        axes[1, 0].legend(fontsize=8)

        # Equity distribution (buy scenarios only)
        for scenario in scenario_names:
            if scenario != 'Keep Renting':
                data = simulation_results[scenario]['results_df']['equity_built']
                axes[1, 1].hist(data, bins=50, alpha=0.4, label=scenario, color=colors[scenario], density=True)
        axes[1, 1].set_xlabel('Equity Built ($)')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Equity Distribution (Buy Scenarios)')
        axes[1, 1].axvline(x=0, color='red', linestyle='--', alpha=0.6, label='Break-even')
        axes[1, 1].legend(fontsize=8)

        plt.tight_layout()
        st.pyplot(fig)

        # Risk assessment text output
        st.subheader("Risk Assessment & Confidence Intervals")
        for scenario_name in scenario_names:
            results = simulation_results[scenario_name]
            cost_data = results['results_df']['total_cost']
            equity_data = results['results_df']['equity_built']

            st.markdown(f"### {scenario_name}")
            st.write(f"Total cost (mean): ${cost_data.mean():,.0f}")
            st.write(f"Total cost (median): ${cost_data.median():,.0f}")
            st.write(f"90% CI (cost): ${cost_data.quantile(0.05):,.0f} - ${cost_data.quantile(0.95):,.0f}")
            st.write(f"Std Dev (cost): ${cost_data.std():,.0f}")

            if scenario_name != 'Keep Renting':
                st.write(f"Equity mean: ${equity_data.mean():,.0f}")
                st.write(f"Probability positive equity: {(equity_data > 0).mean():.1%}")

            st.write(f"Affordability (>=80% months): {results['probability_affordable']:.1%}")
            st.write(f"Default risk: {results['default_risk']:.1%}")
            st.write("---")

        # Generate and display recommendation
        rec = generate_recommendation(sample_household, simulation_results)
        st.subheader("Automated Recommendation")
        st.write(f"Recommended option: {rec['recommended_option']}")
        st.write(f"Expected affordability: {rec['expected_affordability']:.1%}")
        st.write(f"Expected default risk: {rec['expected_default_risk']:.1%}")
        st.write(f"Expected 10-year cost (mean): ${rec['expected_cost']:,.0f}")
        st.write(f"Expected equity (mean): ${rec['expected_equity']:,.0f}")
        st.write("Rationale:")
        st.write(rec['rationale'])

    except Exception as e:
        st.error(f"An error occurred during generation or simulation: {e}")
        raise
else:
    st.info("Adjust parameters on the left and press 'Generate households and run simulations' to start.")
