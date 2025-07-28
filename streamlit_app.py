import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

# --- Pokémon Themed Colors ---
POKEMON_COLORS = {
    "background": "#F8F8F8",
    "primary": "#FF1C1C",   # Pokéball red
    "secondary": "#FFD700", # Pikachu yellow
    "accent": "#3B4CCA",    # Pokémon blue
}

# --- Default Values ---
DEFAULTS = {
    "avg_price": 50.0,
    "num_cards": 10,
    "growth_rate": 2.0,  # percent
    "volatility": 15.0,  # percent
    "months": 12,
    "num_simulations": 10000,
    "asking_price": 1000.0
}

# --- Monte Carlo Simulation ---
def monte_carlo_simulation(avg_price, num_cards, growth_rate, volatility, months, num_simulations):
    start_value = avg_price * num_cards
    results = np.zeros((months, num_simulations))
    for sim in range(num_simulations):
        value = start_value
        for month in range(months):
            monthly_return = np.random.normal(growth_rate, volatility)
            value *= (1 + monthly_return)
            results[month, sim] = value
    return results

# --- Convert Results to CSV ---
def create_csv(results):
    df = pd.DataFrame(results)
    df.index.name = "Month"
    df.columns = [f"Simulation_{i+1}" for i in range(df.shape[1])]
    buffer = BytesIO()
    df.to_csv(buffer)
    buffer.seek(0)
    return buffer

# --- App Config ---
st.set_page_config(page_title="Pokémon TCG Monte Carlo Simulator", layout="wide")

# --- Header Image (from your website) ---
header_image_url = "https://pancakebreakfaststats.com/wp-content/uploads/2025/07/Sales-Simulation.png"
st.markdown(
    f"""
    <div style='text-align:center;'>
        <img src='{header_image_url}' width='600'>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Intro Text ---
st.markdown(
    """
    ### Welcome Pokémon Collectors!

    This simulation is being **pressure-tested live during the 2025 Pancake Analytics Pokémon TCG Panel at Tampa Bay Comic Con**.  
    Use it to explore **portfolio risk, price targets, and potential outcomes** for your cards.  

    Enter your details, click **Run Simulation**, and see **your odds of hitting your target selling price** — backed by Monte Carlo analytics.
    """
)

# --- Initialize Session State ---
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

# --- Sidebar Inputs with Tooltips ---
st.sidebar.header("Simulation Settings")

avg_price = st.sidebar.number_input(
    "Average Card Price ($)",
    min_value=1.0,
    value=st.session_state["avg_price"],
    step=1.0,
    key="avg_price",
    help="The current average market value of each card in your collection."
)

num_cards = st.sidebar.number_input(
    "Number of Cards",
    min_value=1,
    value=st.session_state["num_cards"],
    step=1,
    key="num_cards",
    help="The total number of cards in your collection. Used to calculate your starting portfolio value."
)

growth_rate = st.sidebar.slider(
    "Expected Monthly Growth Rate (%)",
    -10.0, 20.0, st.session_state["growth_rate"],
    help="Your expected average monthly change in card prices (positive or negative). This is the center point of the return distribution."
) / 100

volatility = st.sidebar.slider(
    "Volatility (Std Dev, %)",
    1.0, 50.0, st.session_state["volatility"],
    help="The expected variation in monthly returns. Higher volatility means wider outcome ranges (more risk and reward)."
) / 100

months = st.sidebar.slider(
    "Time Horizon (Months)",
    1, 60, st.session_state["months"],
    key="months",
    help="How far into the future to project your collection’s value, in months (1 to 60)."
)

num_simulations = st.sidebar.slider(
    "Number of Simulations",
    1000, 20000, st.session_state["num_simulations"], step=1000,
    key="num_simulations",
    help="How many randomized portfolio paths to simulate. More simulations give smoother results but may take longer."
)

asking_price = st.sidebar.number_input(
    "Target Selling Price ($)",
    min_value=1.0,
    value=st.session_state["asking_price"],
    step=50.0,
    key="asking_price",
    help="The total price you want to sell your collection for. We'll calculate your probability of reaching it."
)

# --- Reset Button ---
if st.sidebar.button("🔄 Reset to Defaults"):
    for key, val in DEFAULTS.items():
        st.session_state[key] = val
    st.experimental_rerun()

# --- Run Simulation ---
if st.button("Run Simulation"):

    # Run Monte Carlo Simulation
    results = monte_carlo_simulation(avg_price, num_cards, growth_rate, volatility, months, num_simulations)
    final_values = results[-1, :]

    # Calculate Percentiles & Probabilities
    p5, p50, p95 = np.percentile(final_values, [5, 50, 95])
    prob_profit = np.mean(final_values > (avg_price * num_cards))
    prob_asking = np.mean(final_values >= asking_price)

    # --- Simulated Paths Section ---
    with st.expander("📈 Simulated Portfolio Paths (Click to Expand)", expanded=True):
        fig1, ax1 = plt.subplots()
        ax1.set_facecolor(POKEMON_COLORS["background"])
        for i in range(100):  # Show 100 random paths
            ax1.plot(results[:, np.random.randint(num_simulations)], color=POKEMON_COLORS["accent"], alpha=0.1)
        ax1.axhline(asking_price, color=POKEMON_COLORS["primary"], linestyle="--", linewidth=2, label=f"Asking Price: ${asking_price:,.0f}")
        ax1.set_title("Simulated Portfolio Paths", fontsize=16, color=POKEMON_COLORS["primary"])
        ax1.set_xlabel("Months")
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.legend()
        st.pyplot(fig1)

        st.markdown("### How to Read This")
        st.write(f"""
        Each line shows one possible outcome for your collection over {months} months.  
        The **red dashed line** is your asking price (${asking_price:,.0f}).  
        - If **most paths end above the line**, your target is realistic.  
        - If **most are below**, you may need to adjust your price or wait longer.  
        A **wider fan of lines** means more volatility and uncertainty.
        """)

    # --- Distribution Section ---
    with st.expander("📊 Distribution of Final Portfolio Values (Click to Expand)", expanded=True):
        fig2, ax2 = plt.subplots()
        ax2.hist(final_values, bins=50, color=POKEMON_COLORS["secondary"], edgecolor="black", alpha=0.8)
        ax2.axvline(p50, color=POKEMON_COLORS["primary"], linestyle="dashed", linewidth=2, label=f"Median: ${p50:,.0f}")
        ax2.axvline(p5, color="gray", linestyle="dashed", linewidth=2, label=f"5th %ile: ${p5:,.0f}")
        ax2.axvline(p95, color=POKEMON_COLORS["accent"], linestyle="dashed", linewidth=2, label=f"95th %ile: ${p95:,.0f}")
        ax2.axvline(asking_price, color="black", linestyle="solid", linewidth=2, label=f"Asking Price: ${asking_price:,.0f}")
        ax2.set_title("Distribution of Final Portfolio Values", fontsize=16, color=POKEMON_COLORS["primary"])
        ax2.set_xlabel("Portfolio Value ($)")
        ax2.set_ylabel("Frequency")
        ax2.legend()
        st.pyplot(fig2)

        st.markdown("### How to Read This")
        st.write(f"""
        This shows the **distribution of your portfolio’s value** after {months} months.  
        - **Gold bars:** Simulated possible outcomes.  
        - **Dashed lines:** 5th (worst), 50th (median), and 95th (best) percentiles.  
        - **Black line:** Your asking price (${asking_price:,.0f}).  

        If your price is **far right of most bars**, it's likely ambitious.  
        If it’s near the **median (red dashed line)**, it’s a fair target.
        """)

    # --- Summary Section ---
    with st.expander("📜 Simulation Summary & Collector Insights (Click to Expand)", expanded=True):
        st.write(f"**Starting Value:** ${avg_price * num_cards:,.0f}")
        st.write(f"**Median Ending Value (50th %ile):** ${p50:,.0f}")
        st.write(f"**5th Percentile (Worst-Case):** ${p5:,.0f}")
        st.write(f"**95th Percentile (Best-Case):** ${p95:,.0f}")
        st.write(f"**Probability of Ending Above Starting Value:** {prob_profit*100:.1f}%")
        st.write(f"**Probability of Hitting Your Asking Price (${asking_price:,.0f}):** {prob_asking*100:.1f}%")

        if prob_asking > 0.7:
            st.markdown(f":sparkles: Your target price of **${asking_price:,.0f}** is highly realistic, with a **{prob_asking*100:.0f}% chance** of success. Holding or listing higher may be safe.")
        elif prob_asking > 0.4:
            st.markdown(f":warning: Your target has only a **{prob_asking*100:.0f}% chance** of success. Consider lowering it or waiting for the market.")
        else:
            st.markdown(f":x: Only a **{prob_asking*100:.0f}% chance** to hit **${asking_price:,.0f}**. Selling now or adjusting expectations may be wise.")

    # --- CSV Download Section ---
    with st.expander("⬇️ Download Full Simulation Results (CSV)", expanded=False):
        csv_buffer = create_csv(results)
        st.download_button(
            label="Download Monte Carlo Results as CSV",
            data=csv_buffer,
            file_name="pokemon_tcg_monte_carlo_results.csv",
            mime="text/csv"
        )

else:
    st.info("Adjust your inputs in the sidebar and click **Run Simulation** to see results.")

# --- Footer Image (from your website) ---
footer_image_url = "https://pancakebreakfaststats.com/wp-content/uploads/2025/07/Sales-Simulation.png"
st.markdown(
    f"""
    <div style='text-align:center; margin-top:20px;'>
        <img src='{footer_image_url}' width='300'>
    </div>
    """,
    unsafe_allow_html=True
)
