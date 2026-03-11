"""
CIRPA-2026-S1 — Executive BI Dashboard (Vizro)
McKinsey & Company Visual Standard

Business-focused: every graph answers a £ question or a decision question.
No model metrics (R², MAE, F1) — those belong in the technical appendix.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

import vizro.models as vm
from vizro import Vizro
from vizro.managers import data_manager
from vizro.models.types import capture

Cirpa_final = Vizro()
Cirpa_final.build()
server = Cirpa_final.dash.server

# Reset on re-run
from vizro.managers import model_manager
try:
    model_manager._clear()
except Exception:
    pass

# =============================================================================
# LOAD DATA
# =============================================================================

DATA_DIR = "/Users/nxmai0309/Documents/My project/Marketing_analytics"

master = pd.read_csv(f"{DATA_DIR}/master_scoring_table.csv")
roi = pd.read_csv(f"{DATA_DIR}/roi_scenario_grid.csv")
shap_global = pd.read_csv(f"{DATA_DIR}/shap_global_interpretation.csv")

DATA_DIR = "/Users/nxmai0309/Documents/My project/Marketing_analytics/outputs"
# Also load the churn + CLV binary scores for the priority matrix scatter
churn_scores = pd.read_csv(f"{DATA_DIR}/customer_churn_scores.csv")
clv_binary = pd.read_csv(f"{DATA_DIR}/customer_clv_binary_scores.csv")

# Back-transform log Monetary to £
master["Monetary_GBP"] = np.exp(master["Monetary"])

# Segment labels
SEG_MAP = {0: "Bronze", 1: "Platinum", 2: "Silver", 3: "Gold"}
master["Segment_Name"] = master["Segment"].map(SEG_MAP)

# Churn bands
CHURN_BINS = [0, 0.04, 0.25, 0.50, 0.75, 1.01]
CHURN_LABELS = ["<4%", "4–25%", "25–50%", "50–75%", ">75%"]
master["ChurnBand"] = pd.cut(master["Churn_Probability"], bins=CHURN_BINS, labels=CHURN_LABELS, right=False)

# Merged df for priority matrix (Churn_Prob × CLV_High_Prob)
df_matrix = churn_scores.merge(clv_binary, on="CustomerID")

# Register
data_manager["master"] = master
data_manager["roi"] = roi
data_manager["shap_global"] = shap_global
data_manager["df_matrix"] = df_matrix

# =============================================================================
# McKINSEY COLOUR PALETTE & LAYOUT
# =============================================================================

MC = {
    "blue":      "#027BFF",
    "dark":      "#051C2C",
    "light_blue":"#9FD5FF",
    "alert":     "#D94F00",
    "mid_grey":  "#8C8C8C",
    "light_grey":"#D9D9D9",
    "bg":        "#FFFFFF",
    "muted":     "#6B7280",
}

MCK_LAYOUT = dict(
    font=dict(family="Helvetica Neue, Helvetica, Arial, sans-serif", color=MC["dark"], size=12),
    title=dict(font=dict(family="Georgia, Times New Roman, serif", size=18, color=MC["dark"]), x=0, xanchor="left"),
    plot_bgcolor=MC["bg"], paper_bgcolor=MC["bg"],
    xaxis=dict(showgrid=False, zeroline=False, linecolor=MC["light_grey"], linewidth=1,
               ticks="outside", ticklen=4, tickfont=dict(size=11, color=MC["mid_grey"]),
               title_font=dict(size=12, color=MC["mid_grey"])),
    yaxis=dict(showgrid=True, gridcolor=MC["light_grey"], gridwidth=0.5, griddash="dot",
               zeroline=False, showline=False, tickfont=dict(size=11, color=MC["mid_grey"]),
               title_font=dict(size=12, color=MC["mid_grey"])),
    legend=dict(orientation="h", y=-0.15, x=0, xanchor="left",
                font=dict(size=11, color=MC["mid_grey"]), bgcolor="rgba(0,0,0,0)", borderwidth=0),
    margin=dict(t=65, b=55, l=60, r=20),
    hoverlabel=dict(bgcolor=MC["dark"], font_size=11, font_family="Helvetica Neue"),
)

def mck(fig, **overrides):
    layout = {**MCK_LAYOUT, **overrides}
    fig.update_layout(**layout)
    return fig

GRAPH_CARD_LAYOUT = vm.Grid(grid=[[0],[0],[0],[0],[1]])


# =============================================================================
# CATEGORY 1 — REVENUE CONCENTRATION
# =============================================================================

@capture("graph")
def pareto_clv(data_frame):
    """Pareto curve on actual £ CLV_Predicted with tier annotations."""
    df_s = data_frame.sort_values("CLV_Predicted", ascending=False).reset_index(drop=True)
    total = df_s["CLV_Predicted"].sum()
    df_s["cum_pct"] = df_s["CLV_Predicted"].cumsum() / total * 100
    df_s["cust_pct"] = (np.arange(1, len(df_s) + 1)) / len(df_s) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="Perfect equality",
                             line=dict(color=MC["light_grey"], width=1.5, dash="dash")))
    fig.add_trace(go.Scatter(x=df_s["cust_pct"], y=df_s["cum_pct"], mode="lines",
                             name="Actual distribution", line=dict(color=MC["blue"], width=2.5),
                             fill="tozeroy", fillcolor="rgba(2,123,255,0.06)"))

    breakpoints = {1: None, 5: None, 10: None, 20: None}
    for bp in breakpoints:
        idx = int(len(df_s) * bp / 100)
        if idx < len(df_s):
            breakpoints[bp] = df_s.iloc[idx]["cum_pct"]

    # Stagger annotations to avoid overlap — each breakpoint gets a unique ay offset
    offsets = {1: (-60, 50), 5: (-40, 60), 10: (-25, 55), 20: (-15, 45)}
    for bp, cum in breakpoints.items():
        if cum is not None:
            fig.add_trace(go.Scatter(x=[bp], y=[cum], mode="markers",
                                    marker=dict(size=9, color=MC["alert"], symbol="diamond"),
                                    showlegend=False))
            ay, ax = offsets[bp]
            fig.add_annotation(x=bp, y=cum, text=f"<b>Top {bp}%</b> → {cum:.0f}%",
                               showarrow=True, arrowhead=0, arrowcolor=MC["alert"],
                               font=dict(size=9, color=MC["alert"]),
                               ax=ax, ay=ay)

    fig.add_annotation(x=0.98, y=0.05, xref="paper", yref="paper",
                       text=f"<b>Total predicted CLV: £{total:,.0f}</b>",
                       showarrow=False, font=dict(size=12, color=MC["dark"]), xanchor="right")

    mck(fig, title=dict(text="Revenue concentration — who holds the value?"),
        xaxis=dict(title="% of customers (highest value first)", range=[0, 102]),
        yaxis=dict(title="% of cumulative CLV (£)", range=[0, 102]))
    return fig


@capture("graph")
def clv_by_quadrant(data_frame):
    """Stacked bar: £ CLV by priority quadrant."""
    q_order = ["Protect", "Nurture", "Monitor", "Deprioritise"]
    colors = [MC["blue"], MC["light_blue"], MC["light_grey"], MC["mid_grey"]]
    agg = data_frame.groupby("Priority").agg(
        n=("CustomerID", "count"), total_clv=("CLV_Predicted", "sum"),
        mean_clv=("CLV_Predicted", "mean")).reset_index()
    agg["Priority"] = pd.Categorical(agg["Priority"], categories=q_order, ordered=True)
    agg = agg.sort_values("Priority")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=agg["Priority"].astype(str), y=agg["total_clv"],
        marker_color=colors[:len(agg)],
        text=[f"<b>£{v:,.0f}</b><br>{n} customers" for v, n in zip(agg["total_clv"], agg["n"])],
        textposition="outside", textfont=dict(size=11, color=MC["dark"]),
    ))

    total = agg["total_clv"].sum()
    protect_nurture = agg[agg["Priority"].isin(["Protect", "Nurture"])]["total_clv"].sum()
    protect_nurture_n = agg[agg["Priority"].isin(["Protect", "Nurture"])]["n"].sum()
    fig.add_annotation(x=0.5, y=-0.22, xref="paper", yref="paper",
                       text=f"<b>{protect_nurture_n} customers</b> ({protect_nurture_n/len(data_frame)*100:.1f}%) "
                            f"hold <b>£{protect_nurture:,.0f}</b> ({protect_nurture/total*100:.0f}% of forward revenue)",
                       showarrow=False, font=dict(size=11, color=MC["alert"]), xanchor="center")

    mck(fig, title=dict(text="Predicted CLV by retention priority"),
        yaxis=dict(title="Total predicted CLV (£)", tickprefix="£", tickformat=",.0f"),
        margin=dict(t=65, b=75, l=60, r=20))
    return fig


@capture("graph")
def topn_loss_curve(data_frame):
    """Step chart: cumulative £ lost by losing top N customers."""
    df_s = data_frame.sort_values("CLV_Predicted", ascending=False).reset_index(drop=True)
    ns = [5, 10, 25, 50, 100, 200, 500]
    total = df_s["CLV_Predicted"].sum()
    losses = []
    for n in ns:
        lost = df_s.head(n)["CLV_Predicted"].sum()
        losses.append({"N": n, "CLV_Lost": lost, "Pct": lost / total * 100})
    df_loss = pd.DataFrame(losses)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_loss["N"], y=df_loss["CLV_Lost"], mode="lines+markers",
        line=dict(color=MC["blue"], width=2.5), marker=dict(size=8, color=MC["blue"]),
    ))

    # Add annotations with alternating positions to avoid overlap
    positions = ["top center", "bottom center", "top center", "bottom center",
                 "top center", "bottom center", "top center"]
    for i, (_, row) in enumerate(df_loss.iterrows()):
        pos = positions[i % len(positions)]
        fig.add_annotation(x=row["N"], y=row["CLV_Lost"],
                           text=f"<b>£{row['CLV_Lost']:,.0f}</b> ({row['Pct']:.0f}%)",
                           showarrow=False, font=dict(size=9, color=MC["dark"]),
                           yshift=18 if "top" in pos else -18)

    mck(fig, title=dict(text="Cost of losing your top customers"),
        xaxis=dict(title="Number of top customers lost", type="log",
                   tickvals=ns, ticktext=[str(n) for n in ns]),
        yaxis=dict(title="Cumulative CLV at risk (£)", tickprefix="£", tickformat=",.0f"),
        showlegend=False)
    return fig


@capture("graph")
def churn_risk_by_clv_tier(data_frame):
    """Stacked bar: churn band × binary CLV tier (High/Not-High from Priority)."""
    data_frame = data_frame.copy()
    data_frame["CLV_Tier"] = data_frame["Priority"].apply(
        lambda x: "High CLV" if x in ["Protect", "Nurture"] else "Not-High CLV")

    ct = pd.crosstab(data_frame["ChurnBand"], data_frame["CLV_Tier"])
    for col in ["High CLV", "Not-High CLV"]:
        if col not in ct.columns:
            ct[col] = 0

    fig = go.Figure()
    fig.add_trace(go.Bar(x=ct.index.astype(str), y=ct["High CLV"], name="High CLV",
                         marker_color=MC["blue"], text=ct["High CLV"],
                         textposition="inside", textfont=dict(size=12, color="white")))
    fig.add_trace(go.Bar(x=ct.index.astype(str), y=ct["Not-High CLV"], name="Not-High CLV",
                         marker_color=MC["light_blue"], text=ct["Not-High CLV"],
                         textposition="inside", textfont=dict(size=12, color=MC["dark"])))

    mck(fig, barmode="stack", title=dict(text="Who are you actually losing? Churn risk by value tier"),
        xaxis_title="Churn probability band", yaxis_title="Number of customers")
    return fig


# =============================================================================
# CATEGORY 2 — SEGMENTATION
# =============================================================================

@capture("graph")
def priority_matrix_scatter(data_frame):
    """Original priority matrix: Churn_Prob × CLV_High_Prob with quadrant labels."""
    churn_thresh, clv_thresh = 0.04, 0.5

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=data_frame["Churn_Prob"], y=data_frame["CLV_High_Prob"], mode="markers",
        marker=dict(size=4, opacity=0.4, color=data_frame["CLV_High_Prob"],
                    colorscale=[[0, MC["light_grey"]], [0.5, MC["light_blue"]], [1, MC["blue"]]],
                    showscale=False),
        hovertemplate="Churn: %{x:.2f}<br>CLV prob: %{y:.2f}<extra></extra>", showlegend=False))

    fig.add_hline(y=clv_thresh, line_dash="dash", line_color=MC["mid_grey"], line_width=1)
    fig.add_vline(x=churn_thresh, line_dash="dash", line_color=MC["alert"], line_width=1.5)

    protect = ((data_frame["Churn_Prob"] < churn_thresh) & (data_frame["CLV_High_Prob"] >= clv_thresh)).sum()
    nurture = ((data_frame["Churn_Prob"] >= churn_thresh) & (data_frame["CLV_High_Prob"] >= clv_thresh)).sum()
    monitor = ((data_frame["Churn_Prob"] < churn_thresh) & (data_frame["CLV_High_Prob"] < clv_thresh)).sum()
    depri = ((data_frame["Churn_Prob"] >= churn_thresh) & (data_frame["CLV_High_Prob"] < clv_thresh)).sum()

    # Place labels in corners — no overlap with data or each other
    fig.add_annotation(x=0.01, y=0.98, text=f"<b>PROTECT  {protect}</b>", showarrow=False,
                       font=dict(size=13, color=MC["blue"]), xref="paper", yref="paper",
                       xanchor="left", yanchor="top")
    fig.add_annotation(x=0.99, y=0.98, text=f"<b>NURTURE  {nurture}</b>", showarrow=False,
                       font=dict(size=13, color=MC["alert"]), xref="paper", yref="paper",
                       xanchor="right", yanchor="top")
    fig.add_annotation(x=0.01, y=0.02, text=f"<b>MONITOR  {monitor}</b>", showarrow=False,
                       font=dict(size=13, color=MC["mid_grey"]), xref="paper", yref="paper",
                       xanchor="left", yanchor="bottom")
    fig.add_annotation(x=0.99, y=0.02, text=f"<b>DEPRIORITISE  {depri}</b>", showarrow=False,
                       font=dict(size=13, color=MC["muted"]), xref="paper", yref="paper",
                       xanchor="right", yanchor="bottom")

    mck(fig, title=dict(text="Retention priority matrix — every customer positioned"),
        xaxis=dict(title="Churn probability", range=[-0.02, 1.02], showgrid=False),
        yaxis=dict(title="CLV high probability", range=[-0.02, 1.02], showgrid=False))
    return fig


@capture("graph")
def segment_profile(data_frame):
    """Grouped bar: segment profiles — mean CLV, customer count, mean churn."""
    seg_order = ["Platinum", "Gold", "Silver", "Bronze"]
    agg = data_frame.groupby("Segment_Name").agg(
        n=("CustomerID", "count"), mean_clv=("CLV_Predicted", "mean"),
        mean_churn=("Churn_Probability", "mean"),
        mean_revenue=("Monetary_GBP", "mean")).reset_index()
    agg["Segment_Name"] = pd.Categorical(agg["Segment_Name"], categories=seg_order, ordered=True)
    agg = agg.sort_values("Segment_Name")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=agg["Segment_Name"].astype(str), y=agg["mean_revenue"],
        name="Mean historical revenue (£)", marker_color=MC["blue"],
        text=[f"£{v:,.0f}" for v in agg["mean_revenue"]],
        textposition="outside", textfont=dict(size=11, color=MC["dark"]),
    ))
    fig.add_trace(go.Bar(
        x=agg["Segment_Name"].astype(str), y=agg["mean_clv"],
        name="Mean predicted CLV (£)", marker_color=MC["light_blue"],
        text=[f"£{v:,.0f}" for v in agg["mean_clv"]],
        textposition="outside", textfont=dict(size=11, color=MC["dark"]),
    ))

    mck(fig, barmode="group", title=dict(text="Segment profiles — revenue, predicted value, and risk"),
        xaxis_title="Customer segment", yaxis=dict(title="£ per customer", tickprefix="£", tickformat=",.0f"))
    return fig


# =============================================================================
# CATEGORY 3 — CHURN PREDICTION
# =============================================================================

@capture("graph")
def churn_threshold_histogram(data_frame):
    """Histogram with two threshold lines showing £ gap."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=data_frame["Churn_Probability"], nbinsx=60,
                               marker_color=MC["blue"], opacity=0.85, name="Customers"))

    n_opt = (data_frame["Churn_Probability"] >= 0.04).sum()
    n_def = (data_frame["Churn_Probability"] >= 0.50).sum()
    fig.add_vline(x=0.04, line_dash="solid", line_color=MC["alert"], line_width=2,
                  annotation_text=f"<b>Optimal 0.04</b><br>{n_opt:,} flagged",
                  annotation_position="top right", annotation_font=dict(color=MC["alert"], size=10))
    fig.add_vline(x=0.50, line_dash="dash", line_color=MC["mid_grey"], line_width=1.5,
                  annotation_text=f"Default 0.50<br>{n_def:,} flagged",
                  annotation_position="top left", annotation_font=dict(color=MC["mid_grey"], size=10))

    mck(fig, title=dict(text="£168K revenue gap between two threshold choices"),
        xaxis_title="Churn probability", yaxis_title="Number of customers", showlegend=False)
    return fig


@capture("graph")
def churn_cdf(data_frame):
    """CDF showing 96.9% of customers are at risk."""
    sorted_probs = np.sort(data_frame["Churn_Probability"].values)
    cdf = np.arange(1, len(sorted_probs) + 1) / len(sorted_probs) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sorted_probs, y=cdf, mode="lines",
                             line=dict(color=MC["blue"], width=2.5),
                             fill="tozeroy", fillcolor="rgba(2,123,255,0.06)"))

    pct_004 = (data_frame["Churn_Probability"] < 0.04).mean() * 100
    fig.add_vline(x=0.04, line_dash="solid", line_color=MC["alert"], line_width=2,
                  annotation_text=f"<b>0.04</b> → only {pct_004:.1f}% below",
                  annotation_position="top right", annotation_font=dict(color=MC["alert"], size=10))
    fig.add_vline(x=0.50, line_dash="dash", line_color=MC["mid_grey"], line_width=1.5,
                  annotation_text=f"0.50 → {(data_frame['Churn_Probability'] < 0.50).mean()*100:.1f}% below",
                  annotation_position="top left", annotation_font=dict(color=MC["mid_grey"], size=10))

    mck(fig, title=dict(text="Almost everyone is at risk — 96.9% above optimal threshold"),
        xaxis=dict(title="Churn probability", range=[-0.02, 1.02]),
        yaxis=dict(title="% of customers (cumulative)", range=[0, 105]), showlegend=False)
    return fig


# =============================================================================
# CATEGORY 4 — WHY CUSTOMERS CHURN (business-framed)
# =============================================================================

@capture("graph")
def churn_by_clv_tier_box(data_frame):
    """Box plots: churn probability by CLV tier."""
    tier_order = ["High", "Mid", "Low"]
    colors = [MC["blue"], MC["light_blue"], MC["light_grey"]]

    data_frame = data_frame.copy()
    data_frame["CLV_Tier_Reg"] = pd.Categorical(
        data_frame["CLV_Predicted"].apply(
            lambda x: "High" if x >= data_frame["CLV_Predicted"].quantile(0.75) else
                      "Mid" if x >= data_frame["CLV_Predicted"].quantile(0.40) else "Low"),
        categories=tier_order, ordered=True)

    fig = go.Figure()
    for tier, color in zip(tier_order, colors):
        subset = data_frame[data_frame["CLV_Tier_Reg"] == tier]["Churn_Probability"]
        fig.add_trace(go.Box(y=subset, name=f"{tier} CLV", marker_color=color,
                             line=dict(color=MC["dark"], width=1.5), fillcolor=color))
        med = subset.median()
        fig.add_annotation(x=tier_order.index(tier), y=med - 0.06,
                           text=f"<b>{med:.3f}</b>", showarrow=False,
                           font=dict(size=11, color=MC["dark"]))

    mck(fig, title=dict(text="High-value customers are not your churn problem"),
        yaxis=dict(title="Churn probability", range=[-0.05, 1.05]),
        xaxis_title="Customer value tier", showlegend=False)
    return fig


@capture("graph")
def shap_business_drivers(data_frame):
    """SHAP feature importance reframed as business language."""
    BUSINESS_LABELS = {
        "Monetary": "Total spend history",
        "Frequency": "Purchase frequency",
        "AverageRevenuePerOrder": "Average order value",
        "UniqueItemsPerOrder": "Product range breadth",
        "Recency": "Days since last purchase",
        "MonetaryVolatility": "Spending consistency",
        "Lastdaytofirstday": "Customer tenure signal",
        "InterPurchaseInterval": "Purchase timing regularity",
    }

    df_s = data_frame.copy()
    df_s["Business_Label"] = df_s["Feature"].map(BUSINESS_LABELS).fillna(df_s["Feature"])
    df_s = df_s.sort_values("Mean_Abs_SHAP", ascending=True)

    # Top 5 get blue, bottom 3 get light grey
    colors = [MC["blue"] if v >= df_s["Mean_Abs_SHAP"].nlargest(5).min() else MC["light_grey"]
              for v in df_s["Mean_Abs_SHAP"]]

    fig = go.Figure(go.Bar(
        y=df_s["Business_Label"], x=df_s["Mean_Abs_SHAP"], orientation="h",
        marker_color=colors,
        text=[f"<b>{v:.3f}</b>" for v in df_s["Mean_Abs_SHAP"]],
        textposition="outside", textfont=dict(size=11, color=MC["dark"]),
        hovertemplate="%{y}<br>Impact: %{x:.3f}<br>%{customdata}<extra></extra>",
        customdata=df_s["Direction"],
    ))

    # % of signal annotation for top 5
    total_shap = df_s["Mean_Abs_SHAP"].sum()
    top5_pct = df_s.nlargest(5, "Mean_Abs_SHAP")["Mean_Abs_SHAP"].sum() / total_shap * 100
    fig.add_annotation(x=0.98, y=0.08, xref="paper", yref="paper",
                       text=f"Top 5 drivers account for <b>{top5_pct:.0f}%</b> of signal",
                       showarrow=False, font=dict(size=10, color=MC["muted"]), xanchor="right")

    mck(fig, title=dict(text="What drives customers to leave?"),
        xaxis=dict(title="Predictive importance (mean |SHAP|)"),
        margin=dict(t=65, b=55, l=180, r=20))
    return fig


# =============================================================================
# CATEGORY 5 — CLV INTELLIGENCE (business-focused)
# =============================================================================

@capture("graph")
def clv_by_segment(data_frame):
    """Bar: mean CLV per segment with Protect/Nurture customer counts."""
    seg_order = ["Platinum", "Gold", "Silver", "Bronze"]
    agg = data_frame.groupby("Segment_Name").agg(
        n=("CustomerID", "count"),
        mean_clv=("CLV_Predicted", "mean"),
        total_clv=("CLV_Predicted", "sum"),
        n_protect=("Priority", lambda x: (x == "Protect").sum()),
        n_nurture=("Priority", lambda x: (x == "Nurture").sum()),
    ).reset_index()
    agg["Segment_Name"] = pd.Categorical(agg["Segment_Name"], categories=seg_order, ordered=True)
    agg = agg.sort_values("Segment_Name")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=agg["Segment_Name"].astype(str), y=agg["total_clv"], marker_color=MC["blue"],
        text=[f"<b>£{v:,.0f}</b>" for v in agg["total_clv"]],
        textposition="outside", textfont=dict(size=11, color=MC["dark"]),
    ))

    mck(fig, title=dict(text="Where is the predicted lifetime value?"),
        yaxis=dict(title="Total predicted CLV (£)", tickprefix="£", tickformat=",.0f"),
        xaxis_title="Customer segment", showlegend=False)
    return fig


@capture("graph")
def clv_feature_importance(data_frame):
    """CLV feature importance — reframed as 'what determines customer value?'"""
    # Use the same SHAP data but frame differently
    BUSINESS_LABELS = {
        "Monetary": "Total spend history",
        "Frequency": "Purchase frequency",
        "AverageRevenuePerOrder": "Average order value",
        "UniqueItemsPerOrder": "Product range breadth",
        "Recency": "Days since last purchase",
        "MonetaryVolatility": "Spending consistency",
        "Lastdaytofirstday": "Customer tenure signal",
        "InterPurchaseInterval": "Purchase timing regularity",
    }
    df_s = data_frame.copy()
    df_s["Business_Label"] = df_s["Feature"].map(BUSINESS_LABELS).fillna(df_s["Feature"])
    df_s = df_s.sort_values("Mean_Abs_SHAP", ascending=True)

    colors = [MC["blue"] if v >= df_s["Mean_Abs_SHAP"].nlargest(3).min() else MC["light_blue"]
              for v in df_s["Mean_Abs_SHAP"]]

    fig = go.Figure(go.Bar(
        y=df_s["Business_Label"], x=df_s["Mean_Abs_SHAP"], orientation="h",
        marker_color=colors,
        text=[f"{v:.3f}" for v in df_s["Mean_Abs_SHAP"]],
        textposition="outside", textfont=dict(size=11, color=MC["dark"]),
    ))

    mck(fig, title=dict(text="What determines whether a customer is high-value?"),
        xaxis=dict(title="Predictive importance"),
        margin=dict(t=65, b=55, l=180, r=20))
    return fig


# =============================================================================
# CATEGORY 6 — ROI
# =============================================================================

@capture("graph")
def revenue_at_risk(data_frame):
    """Revenue-at-risk by churn band × CLV tier — where is the money most exposed?"""
    data_frame = data_frame.copy()
    data_frame["CLV_Tier"] = data_frame["Priority"].apply(
        lambda x: "High CLV" if x in ["Protect", "Nurture"] else "Not-High CLV")

    ct = data_frame.groupby(["ChurnBand", "CLV_Tier"]).agg(
        Count=("CustomerID", "count"),
        MeanCLV=("CLV_Predicted", "mean"),
        MeanChurn=("Churn_Probability", "mean"),
    ).reset_index()
    ct["RiskValue"] = ct["Count"] * ct["MeanCLV"] * ct["MeanChurn"]

    fig = go.Figure()
    for tier, color in [("High CLV", MC["blue"]), ("Not-High CLV", MC["light_blue"])]:
        subset = ct[ct["CLV_Tier"] == tier]
        fig.add_trace(go.Bar(
            x=subset["ChurnBand"].astype(str), y=subset["RiskValue"],
            name=tier, marker_color=color,
            text=[f"£{v:,.0f}" for v in subset["RiskValue"]],
            textposition="outside", textfont=dict(size=10, color=MC["dark"]),
        ))

    # Highlight the core opportunity band
    key = ct[(ct["ChurnBand"] == "4–25%") & (ct["CLV_Tier"] == "High CLV")]
    if len(key) > 0:
        fig.add_annotation(
            x="4–25%", y=key.iloc[0]["RiskValue"] + key.iloc[0]["RiskValue"] * 0.12,
            text=f"<b>Core opportunity</b><br>{int(key.iloc[0]['Count'])} High-CLV customers",
            showarrow=True, arrowhead=0, arrowwidth=1.5, arrowcolor=MC["alert"],
            font=dict(size=10, color=MC["alert"]), ax=60, ay=-25)

    mck(fig, barmode="group", title=dict(text="Where is the money most at risk?"),
        xaxis_title="Churn probability band",
        yaxis=dict(title="Revenue at risk (£)", tickprefix="£", tickformat=",.0f"))
    return fig

@capture("graph")
def roi_scenario_heatmap(data_frame):
    """3×3 ROI scenario grid — the CFO slide."""
    costs = ["£10", "£25", "£50"]
    rates = ["10%", "20%", "30%"]

    z_values = []
    text_values = []
    for cost in costs:
        row_z = []
        row_t = []
        for rate in rates:
            match = data_frame[(data_frame["Campaign_Cost_Per_Customer"] == cost) &
                               (data_frame["Retention_Rate"] == rate)]
            if len(match) > 0:
                roi_val = match.iloc[0]["ROI_Ratio"]
                net = match.iloc[0]["Net_Benefit"]
                row_z.append(roi_val)
                row_t.append(f"<b>{roi_val:.0f}:1</b><br>Net £{net:,.0f}")
            else:
                row_z.append(0)
                row_t.append("")
        z_values.append(row_z)
        text_values.append(row_t)

    fig = go.Figure(go.Heatmap(
        z=z_values, x=rates, y=costs,
        colorscale=[[0, MC["light_blue"]], [0.5, MC["blue"]], [1.0, MC["dark"]]],
        text=text_values, texttemplate="%{text}", textfont=dict(size=13),
        hovertemplate="Cost: %{y}/customer<br>Retention: %{x}<br>ROI: %{z:.0f}:1<extra></extra>",
        showscale=False,
    ))

    mck(fig, title=dict(text="Every scenario is ROI-positive"),
        xaxis=dict(title="Retention success rate"), yaxis=dict(title="Campaign cost per customer"),
        margin=dict(t=65, b=55, l=80, r=20))
    return fig


@capture("graph")
def roi_cost_vs_return(data_frame):
    """Bar chart: programme cost vs revenue preserved at midpoint."""
    mid = data_frame[(data_frame["Campaign_Cost_Per_Customer"] == "£25") &
                     (data_frame["Retention_Rate"] == "20%")]
    if len(mid) == 0:
        return go.Figure()
    row = mid.iloc[0]

    categories = ["Programme cost", "Revenue preserved", "Net benefit"]
    values = [row["Total_Campaign_Cost"], row["Revenue_Preserved"], row["Net_Benefit"]]
    colors = [MC["alert"], MC["blue"], MC["dark"]]

    fig = go.Figure(go.Bar(
        x=categories, y=values, marker_color=colors,
        text=[f"<b>£{v:,.0f}</b>" for v in values],
        textposition="outside", textfont=dict(size=13, color=MC["dark"]),
        width=0.5,
    ))

    fig.add_annotation(x="Net benefit", y=row["Net_Benefit"] + 5000,
                       text=f"<b>ROI {row['ROI_Ratio']:.0f}:1</b>",
                       showarrow=False, font=dict(size=14, color=MC["alert"]))

    mck(fig, title=dict(text="£3,500 in → £109,307 preserved — midpoint scenario"),
        yaxis=dict(title="£", tickprefix="£", tickformat=",.0f"), showlegend=False)
    return fig


# =============================================================================
# PAGES
# =============================================================================

page_1 = vm.Page(
    title="Revenue Concentration",
    layout=vm.Flex(),
    components=[
        vm.Container(
            title="The top 1% of customers hold 32% of all predicted value",
            layout=GRAPH_CARD_LAYOUT,
            components=[
                vm.Graph(figure=pareto_clv(data_frame="master")),
                vm.Card(text="""
Top 1% (33 customers) = **32.4%** of £1.63M total CLV. Top 5% = **65.9%**. Top 10% = **81.0%**. \
The bottom 50% of customers contribute just **1.3%** of predicted forward revenue. This is not a \
diversified customer base — it is a business where a handful of accounts are existentially important.
"""),
            ],
        ),
        vm.Container(
            title="5.8% of customers hold 69% of forward revenue",
            layout=GRAPH_CARD_LAYOUT,
            components=[
                vm.Graph(figure=clv_by_quadrant(data_frame="master")),
                vm.Card(text="""
**Protect** (140 customers): £546,537 in predicted CLV. **Nurture** (57): £582,517. Together these \
197 customers — 5.8% of the base — represent **69.4%** of all forward revenue. The 3,168 Deprioritise \
customers (94%) hold just 30.4%. Every pound spent equally across the base is a pound wasted on the 94%.
"""),
            ],
        ),
        vm.Container(
            title="The cost of losing your top customers",
            layout=GRAPH_CARD_LAYOUT,
            components=[
                vm.Graph(figure=topn_loss_curve(data_frame="master")),
                vm.Card(text="""
Losing the top 10 customers costs **£349K** — 21% of total predicted CLV. Losing the top 50 costs \
**£746K** (46%). A programme costing £50,000 annually that prevents the loss of two top-10 accounts \
pays for itself many times over. The cost of inaction is the risk, not the cost of the programme.
"""),
            ],
        ),
        vm.Container(
            title="High-churn customers are overwhelmingly low-value",
            layout=GRAPH_CARD_LAYOUT,
            components=[
                vm.Graph(figure=churn_risk_by_clv_tier(data_frame="master")),
                vm.Card(text="""
The >75% churn band is dominated by Not-High CLV customers. Revenue concentration becomes risk \
concentration — you are not losing your valuable customers. The question is not "how do we stop churn" \
but **"which churn is worth stopping?"**
"""),
            ],
        ),
    ],
)

page_2 = vm.Page(
    title="Segmentation",
    layout=vm.Flex(),
    components=[
        vm.Container(
            title="Every customer positioned by risk and value",
            layout=GRAPH_CARD_LAYOUT,
            components=[
                vm.Graph(figure=priority_matrix_scatter(data_frame="df_matrix")),
                vm.Card(text="""
**Protect** (140): high value, elevated churn risk — dedicated account ownership. \
**Nurture** (57): highest value, lowest risk — loyalty investment, not urgent retention. \
**Deprioritise** (3,168): low value, high churn — zero high-touch spend justified. \
Budget allocation follows value at stake, not customer count.
"""),
            ],
        ),
        vm.Container(
            title="Four segments, four different businesses",
            layout=GRAPH_CARD_LAYOUT,
            components=[
                vm.Graph(figure=segment_profile(data_frame="master")),
                vm.Card(text="""
**Platinum** (1,246): £3,672 mean revenue, £1,261 predicted CLV, 29.5% churn — the revenue engine. \
**Bronze** (1,129): £375 mean revenue, £11 predicted CLV, 70.4% churn — one-time buyers who never converted. \
**Silver** (992): £1,187 mean revenue, £45 predicted CLV, 56.7% churn — underactivated but predictable. \
**Gold** (3): £3,540 revenue, 42.4% churn — three whale accounts gone silent. They need a phone call this week.
"""),
            ],
        ),
    ],
)

page_3 = vm.Page(
    title="Churn Prediction",
    layout=vm.Flex(),
    components=[
        vm.Container(
            title="£168K revenue gap between two threshold choices",
            layout=GRAPH_CARD_LAYOUT,
            components=[
                vm.Graph(figure=churn_threshold_histogram(data_frame="master")),
                vm.Card(text="""
Default 50% threshold flags only {n_def:,} customers. Profit-optimised **4% threshold** flags {n_opt:,}. \
At £30/customer and 20% retention: expected revenue recovered **£187,877**, net value **£168,257**, \
ROI **8.6:1**. Set the CRM trigger at **0.04**, not 0.50.
""".format(n_def=(master["Churn_Probability"] >= 0.50).sum(),
           n_opt=(master["Churn_Probability"] >= 0.04).sum())),
            ],
        ),
        vm.Container(
            title="96.9% of customers are technically at risk",
            layout=GRAPH_CARD_LAYOUT,
            components=[
                vm.Graph(figure=churn_cdf(data_frame="master")),
                vm.Card(text="""
Only **{pct:.1f}%** of customers have churn probability below 0.04. In non-contractual retail, this is \
behaviourally accurate — without intervention, almost everyone eventually lapses. \
The model (AUC-ROC 0.755) is a ranked prioritisation tool, not a certainty engine. Imperfect early \
warning deployed consistently outperforms perfect information that arrives too late.
""".format(pct=(master["Churn_Probability"] < 0.04).mean() * 100)),
            ],
        ),
    ],
)

page_4 = vm.Page(
    title="Why Customers Churn",
    layout=vm.Flex(),
    components=[
        vm.Container(
            title="What drives customers to leave?",
            layout=GRAPH_CARD_LAYOUT,
            components=[
                vm.Graph(figure=shap_business_drivers(data_frame="shap_global")),
                vm.Card(text="""
**Total spend history** is the #1 churn predictor — low-spend customers are the primary risk group. \
**Purchase frequency** is #2: repeat buyers get extreme churn protection. **Average order value** #3: \
larger orders signal commitment. **Days since last purchase** ranks only #5 — recency alone is a weak \
signal without the context of how valuable the customer was before going quiet.

The CRM health metric should be a composite: Monetary trajectory + Frequency decline + AOV movement.
"""),
            ],
        ),
        vm.Container(
            title="High-value customers are not your churn problem",
            layout=GRAPH_CARD_LAYOUT,
            components=[
                vm.Graph(figure=churn_by_clv_tier_box(data_frame="master")),
                vm.Card(text="""
The boxes barely overlap. CLV tier is the strongest natural separator of churn risk. This reframes \
the retention strategy: **Platinum churn** is a relationship management problem (owner: Sales). \
**Silver churn** is an engagement problem (owner: CRM). **Bronze churn** is a conversion failure \
(owner: low-cost digital or deprioritise entirely).
"""),
            ],
        ),
        vm.Container(
            title="Intervention deadlines by segment",
            components=[
                vm.Card(text="""
**Platinum:** CRM trigger at **75 days** post-last-purchase (10–15 day window before 90-day threshold). \
**Silver:** Trigger at **60 days** (52-day InterPurchaseInterval means 60 days is already significant). \
**Beyond 120 days (any segment):** Reclassify as win-back prospect, route to lower-cost workflow. \
The trigger must **precede** the threshold, not coincide with it.
"""),
            ],
        ),
    ],
)

page_5 = vm.Page(
    title="CLV Intelligence",
    layout=vm.Flex(),
    components=[
        vm.Container(
            title="Where is the predicted lifetime value?",
            layout=GRAPH_CARD_LAYOUT,
            components=[
                vm.Graph(figure=clv_by_segment(data_frame="master")),
                vm.Card(text="""
**Platinum** holds the vast majority of predicted CLV. The Protect and Nurture customers within Platinum \
represent the core retention investment. Bronze and Silver together represent high customer counts \
but minimal forward value — the budget should follow the value, not the headcount.
"""),
            ],
        ),
        vm.Container(
            title="What determines whether a customer is high-value?",
            layout=GRAPH_CARD_LAYOUT,
            components=[
                vm.Graph(figure=clv_feature_importance(data_frame="shap_global")),
                vm.Card(text="""
The same features that drive churn also determine value — **total spend, frequency, and order value** \
dominate both models. This confirms the feature engineering is sound and means a single CRM monitoring \
system can serve both the churn and CLV use cases simultaneously.
"""),
            ],
        ),
        vm.Container(
            title="Per-customer spend ceilings — from the model, not from judgment",
            components=[
                vm.Card(text="""
| Priority | Customers | Mean CLV | Max spend at 20% retention | Action |
|---|---|---|---|---|
| **Protect** | 140 | £3,904 | **£780** | Named account ownership |
| **Nurture** | 57 | £10,220 | **£2,044** | Loyalty investment |
| **Monitor** | 5 | £902 | **£180** | Digital-only |
| **Deprioritise** | 3,168 | £156 | **£31** | No personalised intervention |

The CLV model accuracy is **83.8%** — 62.4% of truly high-value customers are correctly identified. \
Any Protect/Nurture customer with return rate >25% needs CLV adjusted downward before budgeting.
"""),
            ],
        ),
    ],
)

page_6 = vm.Page(
    title="ROI",
    layout=vm.Flex(),
    components=[
        vm.Container(
            title="Insight 6.1 — Where is the money most at risk?",
            layout=GRAPH_CARD_LAYOUT,
            components=[
                vm.Graph(figure=revenue_at_risk(data_frame="master")),
                vm.Card(text="""
The **4–25% churn band × High CLV** cell concentrates the largest pool of recoverable revenue. \
These are customers with moderate churn risk and high lifetime value — the core retention investment \
opportunity. The >75% band is dominated by Not-High CLV — spending retention budget there has \
negligible expected return. Revenue at risk = customer count × mean CLV × mean churn probability.
"""),
            ],
        ),
        vm.Container(
            title="Every scenario is ROI-positive",
            layout=GRAPH_CARD_LAYOUT,
            components=[
                vm.Graph(figure=roi_scenario_heatmap(data_frame="roi")),
                vm.Card(text="""
The full 3×3 scenario grid: **7.8:1** at the most conservative assumption (£50/customer, 10% retention) \
to **117:1** at the most optimistic (£10/customer, 30% retention). There is **no scenario** within the \
charter assumption range where the Protect programme fails to return more than it costs.
"""),
            ],
        ),
        vm.Container(
            title="£3,500 in → £109,307 preserved",
            layout=GRAPH_CARD_LAYOUT,
            components=[
                vm.Graph(figure=roi_cost_vs_return(data_frame="roi")),
                vm.Card(text="""
At midpoint assumptions (£25/customer, 20% retention): programme cost **£3,500**, revenue preserved \
**£109,307**, net benefit **£105,807**. The break-even retention rate is **0.6%** — the programme \
covers its cost if it retains fewer than **1 customer in every 150 targeted**.
"""),
            ],
        ),
        vm.Container(
            title="Stop spending on the Deprioritise quadrant",
            components=[
                vm.Card(text="""
Deprioritise: 3,168 customers, mean CLV £156, 53.9% churn. At £25/customer and 10% retention: \
**loss of £9.38 per customer**. Current undifferentiated spend: **£79,200**.

Reallocate to 140 Protect customers → per-customer spend rises to **£591**, funding dedicated \
account management. Revenue preserved unchanged at £109,307. \
**This is a reallocation instruction, not a budget increase.**
"""),
            ],
        ),
        vm.Container(
            title="The cost of doing nothing",
            components=[
                vm.Card(text="""
Expected unmitigated loss from Protect quadrant: **£63,573** (£546,537 × 11.6% mean churn). \
Total all-tier programme investment: **£4,975**. Combined CLV at stake: **£1,129,054**. \
If actual Protect retention exceeds **1.3%**, the programme has paid for itself at highest cost assumption.
"""),
            ],
        ),
    ],
)


# =============================================================================
# DASHBOARD
# =============================================================================

dashboard = vm.Dashboard(
    title="CIRPA — Customer Intelligence & Retention Analytics",
    pages=[page_1, page_2, page_3, page_4, page_5, page_6],
    theme="vizro_light",
    navigation=vm.Navigation(
        nav_selector=vm.NavBar(
            items=[
                vm.NavLink(icon="bar_chart", label="Revenue", pages=["Revenue Concentration"]),
                vm.NavLink(icon="groups", label="Segments", pages=["Segmentation"]),
                vm.NavLink(icon="warning", label="Churn", pages=["Churn Prediction"]),
                vm.NavLink(icon="psychology", label="Drivers", pages=["Why Customers Churn"]),
                vm.NavLink(icon="payments", label="CLV", pages=["CLV Intelligence"]),
                vm.NavLink(icon="trending_up", label="ROI", pages=["ROI"]),
            ]
        )
    ),
)

#Vizro().build(dashboard).run(port=8050, jupyter_mode="external")
if __name__ == "__main__":
    Cirpa_final.run()