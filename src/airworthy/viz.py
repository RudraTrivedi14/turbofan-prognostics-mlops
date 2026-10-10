import pandas as pd
import matplotlib.pyplot as plt

# Categorical palette (colorblind-safe, fixed order) — reused for any
# chart that needs a few distinct per-engine colors.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


def top_correlated_sensors(df: pd.DataFrame, sensor_cols: list, n: int = 5) -> pd.Series:
    """Rank sensors by how strongly they correlate with RUL (strongest first)."""
    correlations = df[sensor_cols].corrwith(df["RUL"]).abs()
    return correlations.sort_values(ascending=False).head(n)


def plot_sensor_degradation(
    df: pd.DataFrame, sensor_col: str, n_engines: int = 5, save_path: str = None
):
    """Plot one sensor's reading over engine lifetime, for a handful of engines.

    Each engine's x-axis is "fraction of life lived" (0 = start, 1 = failure),
    so engines with different lifespans overlay on the same comparable scale.
    """
    engine_ids = sorted(df["unit_number"].unique())[:n_engines]

    fig, ax = plt.subplots(figsize=(8, 5))
    for color, engine_id in zip(PALETTE, engine_ids):
        engine_df = df[df["unit_number"] == engine_id]
        life_fraction = engine_df["time_cycles"] / engine_df["time_cycles"].max()
        ax.plot(life_fraction, engine_df[sensor_col], color=color, linewidth=1.5,
                 label=f"engine {engine_id}")

    ax.set_xlabel("fraction of life lived (0 = start, 1 = failure)")
    ax.set_ylabel(sensor_col)
    ax.set_title(f"{sensor_col} over engine lifetime")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
