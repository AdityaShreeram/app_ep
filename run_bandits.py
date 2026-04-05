import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Multi arm bandits Bandits", layout="wide")

# -----------------------------
# Title + Description
# -----------------------------
st.title("🎰 Multi-Armed Bandit: ε-Greedy Exploration")

st.markdown("""
A **multi-armed bandit** models decision-making under uncertainty.

- Each arm has an unknown reward distribution
- Goal: maximize cumulative reward
- Tradeoff:
  - **Exploit** → pick best-known arm
  - **Explore** → try random arms

### ε-Greedy Strategy
- With probability **ε** → explore  
- With probability **1 − ε** → exploit  
""")

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Controls")

k = st.sidebar.slider("Number of arms (k)", 2, 20, 4)
epsilon = st.sidebar.slider("Epsilon (ε)", 0.0, 1.0, 0.1, step=0.01)
steps = st.sidebar.slider("Steps", 100, 5000, 1000)
runs = st.sidebar.slider("Runs (averaging)", 1, 200, 50)

run_button = st.sidebar.button("Run Simulation")

# -----------------------------
# Bandit simulation
# -----------------------------
def run_bandit(k, epsilon, steps):
    true_values = np.random.normal(0, 1, k)
    Q = np.zeros(k)
    N = np.zeros(k)

    rewards = []
    actions = []

    for t in range(steps):
        if np.random.rand() < epsilon:
            action = np.random.randint(k)
        else:
            action = np.argmax(Q)

        reward = np.random.normal(true_values[action], 1)

        N[action] += 1
        Q[action] += (reward - Q[action]) / N[action]

        rewards.append(reward)
        actions.append(action)

    return np.array(rewards), np.array(actions), true_values, Q


# -----------------------------
# Run simulation
# -----------------------------
if run_button:

    avg_rewards = np.zeros(steps)

    # Store last run details for display
    last_true_values = None
    last_Q = None
    last_actions = None

    for _ in range(runs):
        rewards, actions, true_values, Q = run_bandit(k, epsilon, steps)
        avg_rewards += rewards

        # keep last run for inspection
        last_true_values = true_values
        last_Q = Q
        last_actions = actions

    avg_rewards /= runs

    # cumulative average reward
    cumulative_avg = np.cumsum(avg_rewards) / (np.arange(steps) + 1)

    # -----------------------------
    # Plot
    # -----------------------------
    fig, ax = plt.subplots()
    ax.plot(cumulative_avg)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Average Reward")
    ax.set_title(f"ε = {epsilon}")

    st.pyplot(fig)

    # -----------------------------
    # Results
    # -----------------------------
    st.subheader("Results")

    best_arm = np.argmax(last_true_values)
    chosen_counts = np.bincount(last_actions, minlength=k)
    most_chosen_arm = np.argmax(chosen_counts)

    col1, col2, col3 = st.columns(3)

    col1.metric("Final Avg Reward", f"{cumulative_avg[-1]:.3f}")
    col2.metric("True Best Arm", int(best_arm))
    col3.metric("Most Chosen Arm", int(most_chosen_arm))

    # -----------------------------
    # Table of arms
    # -----------------------------
    st.markdown("### Arm Statistics")

    df = pd.DataFrame({
        "Arm": np.arange(k),
        "True Value (μ)": last_true_values,
        "Estimated Value (Q)": last_Q,
        "Times Chosen": chosen_counts
    })

    # Highlight best arm
    def highlight_best(row):
        return ['background-color: #90EE90' if row.Arm == best_arm else '' for _ in row]
    def highlight_row(row):
        if row.Arm == best_arm and most_chosen_arm == best_arm:
            # correct learning → highlight best arm in green
            return ['background-color: #90EE90'] * len(row)
        elif row.Arm == most_chosen_arm and most_chosen_arm != best_arm:
            # wrong learning → highlight chosen arm in red
            return ['background-color: #FFCCCB'] * len(row)
        else:
            return [''] * len(row)

    st.dataframe(
        df.style.format({
            "True Value (μ)": "{:.3f}",
            "Estimated Value (Q)": "{:.3f}"
        }).apply(highlight_row, axis=1),
        use_container_width=True
    )

    # -----------------------------
    # Extra insights
    # -----------------------------
    with st.expander("Interpretation"):
        st.markdown(f"""
- **Best arm (ground truth)**: {best_arm}  
- **Most chosen arm**: {most_chosen_arm}  

If these match → agent successfully learned optimal action.  
If not → either insufficient exploration or noise effects.
""")
        st.subheader("📊 Comparison: Different ε Values")

compare_button = st.button("Run ε Comparison")

if compare_button:

    epsilons = [0.0, 0.01, 0.1, 0.3, 0.7]

    fig, ax = plt.subplots()

    for eps in epsilons:
        avg_rewards = np.zeros(steps)

        for _ in range(runs):
            rewards, _, _, _ = run_bandit(k, eps, steps)
            avg_rewards += rewards

        avg_rewards /= runs
        cumulative_avg = np.cumsum(avg_rewards) / (np.arange(steps) + 1)

        ax.plot(cumulative_avg, label=f"ε = {eps}")

    ax.set_xlabel("Steps")
    ax.set_ylabel("Average Reward")
    ax.set_title("ε-Greedy Comparison")
    ax.legend()

    st.pyplot(fig)