import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, norm
import time


# --- User Inputs ---
#lam = st.number_input("Enter average daily demand (λ)", min_value=1.0, value=22.0, step=1.0)
#lt_min = st.number_input("Minimum lead time (days)", min_value=1, value=4, step=1)
#lt_max = st.number_input("Maximum lead time (days)", min_value=lt_min+1, value=8, step=1)

# Sidebar inputs
st.sidebar.header("Simulation Parameters")
lam = st.sidebar.number_input("Daily demand is assumed to follow a Poisson distribution with Average Daily Demand (λ)", min_value=1, max_value=100, value=22, step=1)

#min_lead = st.sidebar.number_input("Minimum Lead Time (days)", min_value=1, max_value=30, value=4, step=1)
#max_lead = st.sidebar.number_input("Maximum Lead Time (days)", min_value=1, max_value=30, value=8, step=1)
st.sidebar.markdown("**Lead Time (LT)**: is modeled as a uniform distribution between the minimum and maximum days")
col1, col2 = st.sidebar.columns(2)
with col1:
    min_lead = st.number_input("Min Lead Time (days)", min_value=1, value=4, step=1)
with col2:
    max_lead = st.number_input("Max Lead Time (days)", min_value=min_lead+1, value=8, step=1)

# Service level input
st.sidebar.markdown("**Service Level**: is the probability of not running out of stock during the lead time.")
# Service Level slider (percentage view)
service_level_percent = st.sidebar.slider(
    "Service Level (%)", 
    min_value=80, 
    max_value=99, 
    value=95, 
    step=1
)

# Convert to decimal for calculations
service_level = service_level_percent / 100

# Streamlit config
st.set_page_config(layout="wide")
st.title("📦 Bayesian Safety Stock Calculator (Real-Time Simulation)")
st.markdown("(Simulated demand and lead time data without trend or seasonality.)")
import streamlit as st



link='Detailed Document [link](https://prasannamummigatti.github.io/SS_BayesianInference/SS_CalcBayesApprDoc.html)'
st.markdown(link,unsafe_allow_html=True)

st.markdown("""<b>Demand & Lead Time Assumptions :</b> <br> In this simulation, daily demand follows a Poisson distribution,
             representing random, independent customer orders with a stable average rate (λ).
             This means that while the long-term average demand stays constant, actual daily values naturally fluctuate.
            Lead time—the time between placing an order and receiving it—is modeled as a uniform distribution 
            between the minimum and maximum days you specify. 
            This captures the real-world uncertainty in supplier delivery times, 
            such as delays due to transportation or production variability.
            These assumptions allow us to generate realistic, 
            continuously updating scenarios to test and visualize safety stock calculations in dynamic conditions.""",unsafe_allow_html=True)


st.markdown("""
            <b>Posterior Daily Demand Chart</b> <br>
The Posterior Daily Demand Chart illustrates the updated probability distribution of average daily demand after incorporating observed data. 
The peak (mode) represents the most likely daily demand value based on historical observations, 
while the spread (variance) indicates the uncertainty in demand estimation—a narrower spread implies more predictable demand, 
whereas a wider spread reflects higher variability. 
By continuously updating these demand estimates, inventory planning becomes more accurate, 
enabling better safety stock and reorder point calculations.""",unsafe_allow_html=True)


st.markdown("""
<b>Posterior Lead Time Chart</b> <br>
The Posterior Lead Time Chart shows the refined probability distribution of supplier lead times after factoring in observed delivery data. 
The peak (mode) identifies the most probable delivery time in days, 
and the spread (variance) measures the consistency of supplier performance—a narrow spread suggests reliable deliveries, 
while a wide spread indicates unpredictability. 
Understanding this variability is crucial for setting appropriate buffer stock levels and preventing stockouts.""",unsafe_allow_html=True)



# -----------------------------
# Parameters
# -----------------------------
np.random.seed(42)
#service_level = 0.95
z = norm.ppf(service_level)

# Priors
alpha_d_prior = 2
beta_d_prior = 1
alpha_l_prior = 3
beta_l_prior = 1

lambda_range = np.linspace(10, 35, 300)
lt_range = np.linspace(2, 10, 300)

# Simulation parameters
frames = 100
daily_demand = np.random.poisson(lam=lam, size=frames)
lead_times = np.random.randint(min_lead, max_lead, size=frames)

safety_stock_history = []

# -----------------------------
# Streamlit chart placeholders
# -----------------------------
col1, col2 = st.columns(2)
demand_time_chart = col1.empty()
leadtime_time_chart = col2.empty()

col3, col4 = st.columns(2)
demand_hist_chart = col3.empty()
leadtime_hist_chart = col4.empty()

col5, col6 = st.columns(2)
posterior_demand_chart = col5.empty()
posterior_leadtime_chart = col6.empty()

col7, col8 = st.columns(2)
posterior_predictive_chart = col7.empty()
safety_stock_chart = col8.empty()

# -----------------------------
# Real-time simulation loop
# -----------------------------
for n in range(1, frames + 1):
    # Bayesian updates
    alpha_d_post = alpha_d_prior + daily_demand[:n].sum()
    beta_d_post = beta_d_prior + n
    alpha_l_post = alpha_l_prior + lead_times[:n].sum()
    beta_l_post = beta_l_prior + n

    lambda_hat = alpha_d_post / beta_d_post
    lt_hat = alpha_l_post / beta_l_post

    N = 5000
    lambda_samples = np.random.gamma(alpha_d_post, 1.0 / beta_d_post, size=N)
    lt_samples = np.random.gamma(alpha_l_post, 1.0 / beta_l_post, size=N)
    dlt_samples = np.random.poisson(lambda_samples * lt_samples)

    mean_dlt = np.mean(dlt_samples)
    std_dlt = np.std(dlt_samples)
    ss = z * std_dlt
    safety_stock_history.append(ss)

    # -----------------------------
    # Demand over time
    # -----------------------------
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.plot(np.arange(n), daily_demand[:n], label="Observed Demand")
    ax.axhline(np.mean(daily_demand[:n]), color='red', linestyle='--')
    ax.set_title("Daily Demand Over Time")
    demand_time_chart.pyplot(fig)
    plt.close(fig)

    # Lead time over time
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.plot(np.arange(n), lead_times[:n], color='purple', label="Observed Lead Time")
    ax.axhline(np.mean(lead_times[:n]), color='red', linestyle='--')
    ax.set_title("Lead Time Over Time")
    leadtime_time_chart.pyplot(fig)
    plt.close(fig)

    # Histogram of demand
    #fig, ax = plt.subplots(figsize=(6, 2))
    #ax.hist(daily_demand[:n], bins=30, color='skyblue', edgecolor='black')
    #ax.axvline(np.mean(daily_demand[:n]), color='red', linestyle='--')
    #ax.set_title("Histogram of Demand")
    #demand_hist_chart.pyplot(fig)
    #plt.close(fig)

    # Histogram of lead time
    #fig, ax = plt.subplots(figsize=(6, 2))
    #ax.hist(lead_times[:n], bins=30, color='orange', edgecolor='black')
    #ax.axvline(np.mean(lead_times[:n]), color='red', linestyle='--')
    #ax.set_title("Histogram of Lead Time")
    #leadtime_hist_chart.pyplot(fig)
    #plt.close(fig)

    #if n==1:
    #    st.markdown("**Posterior Predictive Demand over Lead Time**: This chart shows the updated probability distribution of total demand expected during the replenishment period, combining posterior demand and lead time uncertainties. It captures the full range of possible demand scenarios while waiting for replenishment.")

    # Posterior of demand rate
    fig, ax = plt.subplots(figsize=(6, 2))
    pdf_d = gamma.pdf(lambda_range, a=alpha_d_post, scale=1.0 / beta_d_post)
    ax.plot(lambda_range, pdf_d, label='Posterior λ')
    if alpha_d_post > 1:
        mode_lambda = (alpha_d_post - 1) / beta_d_post
    else:
        mode_lambda = lambda_hat
    ax.axvline(mode_lambda, color='green', linestyle='--', label=f'MAP λ = {mode_lambda:.2f}')
    ax.set_title("Posterior of Daily Demand (λ)")
    ax.legend()
    posterior_demand_chart.pyplot(fig)
    plt.close(fig)

    # Posterior of lead time
    fig, ax = plt.subplots(figsize=(6, 2))
    pdf_l = gamma.pdf(lt_range, a=alpha_l_post, scale=1.0 / beta_l_post)
    ax.plot(lt_range, pdf_l, color='green', label='Posterior LT')
    if alpha_l_post > 1:
        mode_lt_posterior = (alpha_l_post - 1) / beta_l_post
    else:
        mode_lt_posterior = lt_hat
    ax.axvline(mode_lt_posterior, color='green', linestyle='--', label=f'MAP LT = {mode_lt_posterior:.2f}')
    ax.set_title("Posterior of Lead Time")
    ax.legend()
    posterior_leadtime_chart.pyplot(fig)
    plt.close(fig)

    # Posterior predictive
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.hist(dlt_samples, bins=40, density=True, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(mean_dlt + ss, color='red', linestyle='--', label=f'Reorder Point = {mean_dlt + ss:.1f}')
    ax.set_title("Posterior Predictive: Demand During Lead Time")
    posterior_predictive_chart.pyplot(fig)
    plt.close(fig)

    # Safety stock over time
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.plot(safety_stock_history, color='black', label='Safety Stock')
    ax.axhline(safety_stock_history[-1], color='blue', linestyle='--',label=f'Calculated SS = {safety_stock_history[-1]:.2f}')
    ax.set_title(f"Safety Stock Over Time (Service Level: {service_level_percent}%)")
    ax.legend()
    safety_stock_chart.pyplot(fig)
    plt.close(fig)
    if n==1:
        st.markdown("**Posterior Predictive Demand over Lead Time**: This chart shows the updated probability distribution of total demand expected during the replenishment period, combining posterior demand and lead time uncertainties. It captures the full range of possible demand scenarios while waiting for replenishment.")
        st.markdown("**Safety Stock Calculation** → Determined from the predictive distribution using the chosen service level. Higher service levels shift the stock threshold upward, reducing stockouts but increasing inventory holding. This approach ensures the safety stock directly reflects both demand and lead time variability.")
    time.sleep(0.5)  # Control animation speed





