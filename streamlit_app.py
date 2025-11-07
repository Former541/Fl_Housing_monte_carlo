"""
Florida Housing Affordability Analyzer
Monte Carlo Simulation Web Application

A comprehensive Streamlit application for analyzing Florida housing
affordability using Monte Carlo simulation methods.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from household_generator import FloridaHouseholdGenerator
from monte_carlo_housing import MonteCarloHousingSimulator
from financial_analysis import FloridaHousingAnalyzer
from gemini_helper import (
    create_gemini_button_with_report,
    generate_household_report,
    generate_simulation_report,
    add_sensitivity_sliders
)


# Page configuration
st.set_page_config(
    page_title="Florida Housing Affordability Analyzer",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
    }
    h1 {
        color: #1f77b4;
    }
    h2 {
        color: #2ca02c;
    }
    </style>
""", unsafe_allow_html=True)


# Initialize session state
if 'households_df' not in st.session_state:
    st.session_state.households_df = None
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None


def main():
    """Main application function"""

    # Header
    st.title("🏠 Florida Housing Affordability Analyzer")
    st.markdown("""
    ### Monte Carlo Simulation for Data-Driven Housing Decisions

    Analyze Florida housing affordability with probabilistic modeling, including region-specific
    factors like hurricane insurance, property taxes, and market volatility.
    """)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        app_mode = st.radio(
            "Select Mode",
            ["📊 Generate Household Data", "🎲 Run Simulations", "📈 Analyze Results", "🏡 Single Household Analysis"],
            index=0
        )

        st.divider()

        st.markdown("""
        ### About

        **Monte Carlo Methodology & Data Approach**

        This application employs **synthetic household generation** rather than
        static datasets. This is standard practice in Monte Carlo simulation
        research because it:

        - Enables controlled experimentation with population parameters
        - Eliminates data privacy, quality, and availability concerns
        - Provides unlimited sample generation (1,000 to 1,000,000+ households)
        - Maintains statistical validity using real Florida distributions
        - Supports academic reproducibility

        Each synthetic household is generated using probability distributions
        calibrated to U.S. Census data, Florida regional economics, and credit
        bureau statistics.

        **Monte Carlo Simulations:**
        - 10,000+ scenarios per household
        - Probabilistic modeling of income, costs, appreciation
        - Risk quantification through distribution analysis

        **Florida-Specific Factors:**
        - Hurricane insurance (15-25% of ownership costs)
        - Regional price variations (35%+ differences)
        - Property tax (0.9% avg)
        - Market volatility and climate risks

        **Academic Justification:** Published Monte Carlo studies in finance,
        healthcare, and policy routinely use synthetic data to control confounding
        variables and test sensitivity to distributional assumptions.
        """)

    # Add sensitivity parameter sliders to sidebar
    sensitivity_params = add_sensitivity_sliders()
    # Store in session state for access by simulation functions
    st.session_state['sensitivity_params'] = sensitivity_params

    # Main content based on selected mode
    if app_mode == "📊 Generate Household Data":
        generate_household_data_page()

    elif app_mode == "🎲 Run Simulations":
        run_simulations_page()

    elif app_mode == "📈 Analyze Results":
        analyze_results_page()

    elif app_mode == "🏡 Single Household Analysis":
        single_household_analysis_page()


def generate_household_data_page():
    """Page for generating household cohort data"""

    st.header("📊 Generate Florida Household Cohort")

    col1, col2 = st.columns(2)

    with col1:
        n_households = st.slider(
            "Number of Base Households",
            min_value=50,
            max_value=5000,
            value=1000,
            step=50,
            help="Number of households to generate in the base population"
        )

        amplify = st.checkbox(
            "Enable Synthetic Data Amplification",
            value=True,
            help="Add synthetic edge cases (high-risk and wealthy households)"
        )

    with col2:
        if amplify:
            amplification_factor = st.slider(
                "Amplification Factor",
                min_value=0.1,
                max_value=0.5,
                value=0.3,
                step=0.05,
                help="Proportion of synthetic households to add"
            )
        else:
            amplification_factor = 0.0

        random_seed = st.number_input(
            "Random Seed",
            min_value=0,
            max_value=9999,
            value=42,
            help="Set seed for reproducible results"
        )

    if st.button("🔄 Generate Household Data", type="primary"):
        with st.spinner("Generating Florida household cohort..."):
            generator = FloridaHouseholdGenerator(random_seed=random_seed)
            households_df = generator.generate_household_cohort(
                n_households=n_households,
                amplify=amplify,
                amplification_factor=amplification_factor
            )

            st.session_state.households_df = households_df
            st.success(f"✅ Successfully generated {len(households_df)} households!")

    # Display household data if available
    if st.session_state.households_df is not None:
        df = st.session_state.households_df

        st.subheader("Household Cohort Summary")

        # Summary metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Households", len(df))
        with col2:
            st.metric("Mean Income", f"${df['annual_income'].mean():,.0f}")
        with col3:
            st.metric("Mean Credit Score", f"{df['credit_score'].mean():.0f}")
        with col4:
            st.metric("Mean Savings", f"${df['savings'].mean():,.0f}")
        with col5:
            st.metric("Mean Risk Score", f"{df['financial_risk_score'].mean():.1f}")

        # Distribution charts
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Florida Region Distribution")
            region_counts = df['region'].value_counts()
            fig1, ax1 = plt.subplots(figsize=(8, 6))
            region_counts.plot(kind='bar', ax=ax1, color='steelblue')
            ax1.set_xlabel('Region')
            ax1.set_ylabel('Number of Households')
            ax1.set_title('Households by Florida Region')
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig1)

        with col2:
            st.subheader("Income Distribution")
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            ax2.hist(df['annual_income'], bins=30, color='green', edgecolor='black')
            ax2.axvline(df['annual_income'].mean(), color='red', linestyle='--', label='Mean')
            ax2.set_xlabel('Annual Income ($)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Distribution of Annual Income')
            ax2.legend()
            st.pyplot(fig2)

        # Data table
        st.subheader("Household Data Preview")
        st.dataframe(
            df.head(20),
            use_container_width=True,
            height=400
        )

        # Gemini AI Integration
        st.markdown("---")
        st.subheader("🤖 Ask Gemini About This Data")
        report = generate_household_report(df)
        create_gemini_button_with_report(report, "household_gen")

        # Download option
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Household Data (CSV)",
            data=csv,
            file_name="florida_households.csv",
            mime="text/csv"
        )


def run_simulations_page():
    """Page for running Monte Carlo simulations"""

    st.header("🎲 Run Monte Carlo Housing Simulations")

    if st.session_state.households_df is None:
        st.warning("⚠️ Please generate household data first!")
        return

    df = st.session_state.households_df

    col1, col2 = st.columns(2)

    with col1:
        num_simulations = st.select_slider(
            "Number of Simulations per Household",
            options=[1000, 5000, 10000, 20000],
            value=10000,
            help="More simulations = more accurate results but slower"
        )

        sample_size = st.slider(
            "Sample Size (for faster testing)",
            min_value=10,
            max_value=min(500, len(df)),
            value=min(100, len(df)),
            help="Simulate a subset of households for faster results"
        )

    with col2:
        time_horizon = st.slider(
            "Time Horizon (Years)",
            min_value=5,
            max_value=30,
            value=10,
            help="Projection period for simulations"
        )

        scenario = st.selectbox(
            "Housing Scenario to Simulate",
            options=['All Scenarios', 'Keep Renting', 'Buy Starter Home', 'Buy Standard Home', 'Buy Premium Home'],
            help="Select specific scenario or simulate all"
        )

        st.info(f"""
        **Simulation Parameters:**
        - Total households: {len(df)}
        - Sample to simulate: {sample_size}
        - Simulations per household: {num_simulations:,}
        - Total simulations: {sample_size * num_simulations:,}
        - Time horizon: {time_horizon} years

        Estimated time: ~{sample_size * num_simulations / 10000:.1f} minutes
        """)

    if st.button("▶️ Run Simulations", type="primary"):
        with st.spinner("Running Monte Carlo simulations... This may take a few minutes."):
            # Get sensitivity parameters from session state
            params = st.session_state.get('sensitivity_params', {})
            simulator = MonteCarloHousingSimulator(
                random_seed=42,
                income_growth=params.get('income_growth', 0.04),
                insurance_increase=params.get('insurance_increase', 0.08),
                affordability_threshold=params.get('affordability_threshold', 0.50)
            )

            # Sample households
            sample_df = df.sample(n=sample_size, random_state=42)

            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            results_list = []

            for idx, (_, household) in enumerate(sample_df.iterrows()):
                if scenario == 'All Scenarios':
                    # Simulate all scenarios
                    for scen_name in ['Keep Renting', 'Buy Starter Home', 'Buy Standard Home', 'Buy Premium Home']:
                        result = simulator.simulate_household(
                            household,
                            scen_name,
                            num_simulations=num_simulations,
                            time_horizon_years=time_horizon
                        )
                        results_list.append(result)
                else:
                    # Simulate single scenario
                    result = simulator.simulate_household(
                        household,
                        scenario,
                        num_simulations=num_simulations,
                        time_horizon_years=time_horizon
                    )
                    results_list.append(result)

                # Update progress
                progress = (idx + 1) / len(sample_df)
                progress_bar.progress(progress)
                status_text.text(f"Simulating household {idx + 1} of {len(sample_df)}")

            progress_bar.progress(100)
            status_text.text("Simulations complete!")

            results_df = pd.DataFrame(results_list)
            st.session_state.simulation_results = results_df

            st.success(f"✅ Successfully simulated {len(results_df)} scenarios!")

    # Display simulation results if available
    if st.session_state.simulation_results is not None:
        results = st.session_state.simulation_results

        st.subheader("Simulation Results Summary")

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Mean Affordability Rate",
                f"{results['probability_affordable'].mean()*100:.1f}%"
            )
        with col2:
            default_mean = results.get('probability_default', pd.Series([0])).mean()
            st.metric(
                "Mean Default Risk",
                f"{default_mean*100:.1f}%"
            )
        with col3:
            mean_equity = results['equity_built'].apply(lambda x: x['mean'] if isinstance(x, dict) else 0).mean()
            st.metric(
                "Mean Equity Built",
                f"${mean_equity:,.0f}"
            )
        with col4:
            st.metric(
                "Scenarios Simulated",
                len(results)
            )

        st.subheader("Results Preview")
        display_results = results[['household_id', 'scenario', 'probability_affordable',
                                   'probability_default', 'time_horizon_years']].head(20)
        st.dataframe(display_results, use_container_width=True)

        # Gemini AI Integration
        st.markdown("---")
        st.subheader("🤖 Ask Gemini About These Results")
        sim_params = {
            'num_simulations': num_simulations,
            'time_horizon': time_horizon,
            'sample_size': sample_size
        }
        report = generate_simulation_report(results, sim_params)
        create_gemini_button_with_report(report, "simulation")


def analyze_results_page():
    """Page for analyzing simulation results"""

    st.header("📈 Comprehensive Results Analysis")

    if st.session_state.simulation_results is None:
        st.warning("⚠️ Please run simulations first!")
        return

    if st.session_state.households_df is None:
        st.warning("⚠️ Household data not available!")
        return

    results = st.session_state.simulation_results
    households = st.session_state.households_df
    analyzer = FloridaHousingAnalyzer()

    # Generate summary statistics
    summary = analyzer.generate_summary_statistics(results)

    st.subheader("📊 Overall Statistics")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Scenarios Analyzed", summary['total_households'])
        st.metric("High-Risk Households", summary['overall_metrics']['high_risk_households'])
    with col2:
        st.metric("Overall Affordability Rate", f"{summary['overall_metrics']['mean_affordability_rate']*100:.1f}%")
        st.metric("Low-Risk Households", summary['overall_metrics']['low_risk_households'])
    with col3:
        st.metric("Overall Default Rate", f"{summary['overall_metrics']['mean_default_rate']*100:.1f}%")

    # Scenario comparison
    st.subheader("🏆 Scenario Comparison")
    comparison_df = analyzer.compare_scenarios_analysis(results)
    st.dataframe(comparison_df.style.format({
        'Mean_Affordability_Rate': '{:.2%}',
        'Mean_Default_Rate': '{:.2%}',
        'Mean_Equity_Built': '${:,.0f}',
        'Mean_Total_Cost': '${:,.0f}',
        'Affordability_Rate_Std': '{:.3f}'
    }), use_container_width=True)

    # Income stratification
    st.subheader("💰 Income Stratification Analysis")
    income_analysis = analyzer.income_stratification_analysis(households, results)

    income_df = pd.DataFrame(income_analysis).T
    st.dataframe(income_df.style.format({
        'n_households': '{:.0f}',
        'mean_affordability_rate': '{:.2%}',
        'mean_default_rate': '{:.2%}',
        'mean_equity': '${:,.0f}',
        'mean_income': '${:,.0f}'
    }), use_container_width=True)

    # Region analysis
    st.subheader("🗺️ Florida Region Analysis")
    region_analysis = analyzer.region_analysis(households, results)

    region_df = pd.DataFrame(region_analysis).T
    st.dataframe(region_df.style.format({
        'n_households': '{:.0f}',
        'mean_affordability_rate': '{:.2%}',
        'mean_default_rate': '{:.2%}',
        'mean_equity': '${:,.0f}'
    }), use_container_width=True)

    # Visualizations
    st.subheader("📊 Comprehensive Visualization Report")
    with st.spinner("Generating visualizations..."):
        fig = analyzer.generate_visualization_report(households, results)
        st.pyplot(fig)

    # Gemini AI Integration
    st.markdown("---")
    st.subheader("🤖 Ask Gemini About This Analysis")
    overall_afford = results['probability_affordable'].mean() * 100
    best_scenario = results.groupby('scenario')['probability_affordable'].mean().idxmax()
    analysis_report = f"""COMPREHENSIVE FLORIDA HOUSING ANALYSIS
Total Scenarios: {len(results)}
Overall Affordability: {overall_afford:.1f}%

Best Scenario: {best_scenario}

Question: What insights about Florida housing affordability?"""
    create_gemini_button_with_report(analysis_report[:500], "analysis")

        # Export options
    st.subheader("💾 Export Results")
    if st.button("📥 Generate Detailed Report (CSV)"):
        import io
        buffer = io.StringIO()
        analyzer.export_detailed_report(households, results, 'temp_report.csv')

        # Read back and offer download
        with open('temp_report.csv', 'r') as f:
            csv_data = f.read()

        st.download_button(
            label="Download Detailed CSV Report",
            data=csv_data,
            file_name="florida_housing_analysis_report.csv",
            mime="text/csv"
        )


def single_household_analysis_page():
    """Page for analyzing a single household with scenario comparison"""

    st.header("🏡 Single Household Housing Analysis")

    if st.session_state.households_df is None:
        st.warning("⚠️ Please generate household data first!")
        return

    df = st.session_state.households_df

    st.markdown("""
    Analyze a specific household and compare all housing scenarios using Monte Carlo simulation.
    Get personalized recommendations based on household characteristics.
    """)

    # Household selection
    household_id = st.selectbox(
        "Select Household ID",
        options=df['household_id'].tolist(),
        index=0
    )

    household = df[df['household_id'] == household_id].iloc[0]

    # Display household information
    st.subheader("👥 Household Information")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Annual Income", f"${household['annual_income']:,.0f}")
        st.metric("Household Size", f"{household['household_size']}")
    with col2:
        st.metric("Credit Score", f"{household['credit_score']}")
        st.metric("Employment Sector", household['employment_sector'])
    with col3:
        st.metric("Current Debt", f"${household['current_debt']:,.0f}")
        st.metric("DTI Ratio", f"{household['debt_to_income_ratio']:.2%}")
    with col4:
        st.metric("Savings", f"${household['savings']:,.0f}")
        st.metric("Region", household['region'])

    st.metric("Financial Risk Score", f"{household['financial_risk_score']:.1f}/100")
    st.metric("Current Monthly Rent", f"${household['current_monthly_rent']:,.0f}")

    # Simulation parameters
    col1, col2 = st.columns(2)
    with col1:
        num_simulations = st.select_slider(
            "Number of Simulations",
            options=[1000, 5000, 10000, 20000],
            value=10000
        )
    with col2:
        time_horizon = st.slider(
            "Time Horizon (Years)",
            min_value=5,
            max_value=30,
            value=10
        )

    if st.button("🎲 Compare All Housing Scenarios", type="primary"):
        with st.spinner("Running comparative simulations..."):
            # Get sensitivity parameters from session state
            params = st.session_state.get('sensitivity_params', {})
            simulator = MonteCarloHousingSimulator(
                random_seed=42,
                income_growth=params.get('income_growth', 0.04),
                insurance_increase=params.get('insurance_increase', 0.08),
                affordability_threshold=params.get('affordability_threshold', 0.50)
            )
            comparison = simulator.compare_scenarios(household, num_simulations, time_horizon)

            # Display comparison results
            st.subheader("📊 Scenario Comparison Results")

            # Create comparison table
            comparison_data = []
            for scenario_name, result in comparison.items():
                comparison_data.append({
                    'Scenario': scenario_name,
                    'Affordability Rate': f"{result['probability_affordable']*100:.1f}%",
                    'Default Risk': f"{result.get('probability_default', 0)*100:.1f}%",
                    'Mean Equity Built': f"${result['equity_built']['mean']:,.0f}",
                    'Mean Total Cost': f"${result['total_cost_paid']['mean']:,.0f}",
                    'Mean Affordable Months': f"{result['mean_affordable_months']:.0f}"
                })

            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)

            # Visualizations
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Affordability Rates")
                afford_rates = {k: v['probability_affordable'] for k, v in comparison.items()}
                fig1, ax1 = plt.subplots(figsize=(8, 6))
                plt.bar(afford_rates.keys(), [v*100 for v in afford_rates.values()], color='steelblue')
                plt.ylabel('Affordability Rate (%)')
                plt.title('Scenario Affordability Comparison')
                plt.xticks(rotation=45, ha='right')
                plt.ylim(0, 100)
                st.pyplot(fig1)

            with col2:
                st.subheader("Equity Built")
                equity_values = {k: v['equity_built']['mean'] for k, v in comparison.items()}
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                plt.bar(equity_values.keys(), equity_values.values(), color='green')
                plt.ylabel('Mean Equity Built ($)')
                plt.title('Expected Equity by Scenario')
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig2)

            # Recommendations
            st.subheader("🎯 Personalized Recommendations")

            best_afford = max(comparison.items(), key=lambda x: x[1]['probability_affordable'])
            safest = min(comparison.items(), key=lambda x: x[1].get('probability_default', 1))
            best_equity = max(comparison.items(), key=lambda x: x[1]['equity_built']['mean'])

            col1, col2, col3 = st.columns(3)

            with col1:
                st.success(f"""
                **Most Affordable**

                {best_afford[0]}

                Affordability: {best_afford[1]['probability_affordable']*100:.1f}%
                """)

            with col2:
                st.info(f"""
                **Lowest Risk**

                {safest[0]}

                Default Risk: {safest[1].get('probability_default', 0)*100:.1f}%
                """)

            with col3:
                st.warning(f"""
                **Best Equity Building**

                {best_equity[0]}

                Mean Equity: ${best_equity[1]['equity_built']['mean']:,.0f}
                """)

            # Gemini AI Integration
            st.markdown("---")
            st.subheader("🤖 Ask Gemini About This Household")
            hh_income = household['annual_income']
            hh_credit = household['credit_score']
            hh_region = household['region']
            hh_risk = household['financial_risk_score']
            best_scenario_name = best_afford[0]
            best_afford_pct = best_afford[1]['probability_affordable'] * 100

            single_report = f"""PERSONAL HOUSING ANALYSIS
Household Profile:
- Income: ${hh_income:,.0f}/year
- Credit: {hh_credit:.0f}
- Region: {hh_region}
- Risk Score: {hh_risk:.1f}/100

Recommended: {best_scenario_name}
Affordability: {best_afford_pct:.1f}%

Question: What should this household do?"""
            create_gemini_button_with_report(single_report[:500], "single_household")


if __name__ == "__main__":
    main()
