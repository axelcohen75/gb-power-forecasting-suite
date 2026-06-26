import os
import pandas as pd
from datetime import timedelta

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, value, PULP_CBC_CMD

# ── DATA ──────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
for _p in [
    os.path.join(BASE, "forecast_ID_2024.csv"),
    os.path.join(BASE, "..", "dataset", "forecast_ID_2024.csv"),
]:
    if os.path.exists(_p):
        _df = pd.read_csv(_p, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
        break

AVAILABLE_DATES = sorted(_df.index.normalize().unique().strftime("%Y-%m-%d").tolist())
MIN_DATE, MAX_DATE = AVAILABLE_DATES[0], AVAILABLE_DATES[-1]

# ── LP OPTIMISER ──────────────────────────────────────────────────────────────
def optimise(date_str, cap, max_pw, eff, soc_pct):
    day = pd.Timestamp(date_str)
    prices = _df.loc[day : day + timedelta(hours=23), "price_forecast"].values[:24]
    if len(prices) < 24:
        return None

    init_soc = cap * soc_pct / 100
    T = range(24)

    prob = LpProblem("BESS", LpMaximize)
    c = {t: LpVariable(f"c{t}", 0, max_pw) for t in T}
    d = {t: LpVariable(f"d{t}", 0, max_pw) for t in T}
    s = {t: LpVariable(f"s{t}", 0, cap)    for t in T}

    prob += lpSum(prices[t] * d[t] - prices[t] * c[t] for t in T)
    prob += s[0] == init_soc + c[0] * eff - d[0]
    for t in range(1, 24):
        prob += s[t] == s[t-1] + c[t] * eff - d[t]
    prob += s[23] == init_soc

    prob.solve(PULP_CBC_CMD(msg=0))

    cv = [max(0, value(c[t]) or 0) for t in T]
    dv = [max(0, value(d[t]) or 0) for t in T]
    sv = [max(0, value(s[t]) or 0) for t in T]

    revenue      = sum(prices[t] * (dv[t] - cv[t]) for t in T)
    energy_sold  = sum(dv)
    energy_bought= sum(cv)
    avg_buy  = sum(prices[t]*cv[t] for t in T) / energy_bought if energy_bought else 0
    avg_sell = sum(prices[t]*dv[t] for t in T) / energy_sold   if energy_sold   else 0

    return dict(
        prices=prices.tolist(), charge=cv, discharge=dv, soc=sv,
        revenue=round(revenue, 2),
        avg_buy=round(avg_buy, 1), avg_sell=round(avg_sell, 1),
        spread=round(avg_sell - avg_buy, 1),
        energy_sold=round(energy_sold, 1),
        cycles=round(energy_sold / cap, 2) if cap else 0,
    )

# ── THEME ─────────────────────────────────────────────────────────────────────
COLORS = dict(
    bg="#0a0c10", surface="#111418", surface2="#181c22",
    border="#252a34", border2="#2e3542",
    text="#e2e8f0", text2="#8892a4", text3="#5a6478",
    accent="#3b82f6", green="#22c55e", red="#ef4444",
    orange="#f97316", purple="#a855f7", yellow="#eab308",
)

PLOT_LAYOUT = dict(
    plot_bgcolor=COLORS["surface"], paper_bgcolor=COLORS["surface"],
    font=dict(family="Inter, sans-serif", color=COLORS["text2"]),
    margin=dict(t=12, b=40, l=58, r=16),
    hovermode="x unified",
    hoverlabel=dict(bgcolor=COLORS["surface2"], bordercolor=COLORS["border2"],
                    font=dict(family="JetBrains Mono, monospace", size=12)),
    legend=dict(orientation="h", x=0, y=-0.18, font=dict(size=11),
                bgcolor="rgba(0,0,0,0)"),
    xaxis=dict(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"],
               tickfont=dict(family="JetBrains Mono", size=10)),
    yaxis=dict(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"],
               tickfont=dict(family="JetBrains Mono", size=10)),
)

# ── LAYOUT ────────────────────────────────────────────────────────────────────
def slider_row(label, id_, min_, max_, step, val, unit):
    return html.Div([
        html.Div([
            html.Span(label, style={"fontSize": "12px", "color": COLORS["text2"]}),
            html.Span(f"{val} {unit}", id=f"lbl-{id_}", style={
                "fontFamily": "JetBrains Mono, monospace", "fontSize": "12px",
                "fontWeight": "600", "color": COLORS["text"],
            }),
        ], style={"display": "flex", "justifyContent": "space-between"}),
        dcc.Slider(id=id_, min=min_, max=max_, step=step, value=val,
                   marks=None, tooltip={"always_visible": False},
                   className="dash-slider"),
    ], style={"display": "flex", "flexDirection": "column", "gap": "6px"})


def kpi_card(label, id_, color, unit):
    return html.Div([
        html.Div(label, style={"fontSize": "10px", "fontWeight": "700",
                               "letterSpacing": "0.08em", "color": COLORS["text3"],
                               "textTransform": "uppercase"}),
        html.Div("—", id=id_, style={
            "fontFamily": "JetBrains Mono, monospace", "fontSize": "22px",
            "fontWeight": "700", "color": color, "margin": "4px 0 2px",
        }),
        html.Div(unit, style={"fontSize": "10px", "color": COLORS["text3"]}),
    ], style={
        "padding": "14px 16px", "borderRight": f"1px solid {COLORS['border']}",
        "flex": "1",
    })


SIDEBAR = html.Div([

    # Date
    html.Div([
        html.Div("Trading Date", style={"fontSize": "10px", "fontWeight": "700",
                 "letterSpacing": "0.1em", "color": COLORS["text3"],
                 "textTransform": "uppercase", "paddingBottom": "6px",
                 "borderBottom": f"1px solid {COLORS['border']}"}),
        html.Div([
            html.Button("‹", id="prev-day", n_clicks=0, style={
                "width": "28px", "height": "32px", "background": COLORS["surface2"],
                "border": f"1px solid {COLORS['border2']}", "color": COLORS["text2"],
                "borderRadius": "4px", "cursor": "pointer", "fontSize": "16px",
                "flexShrink": "0",
            }),
            dcc.DatePickerSingle(
                id="date-picker", date="2024-10-14",
                min_date_allowed=MIN_DATE, max_date_allowed=MAX_DATE,
                display_format="YYYY-MM-DD",
                style={"flex": "1"},
            ),
            html.Button("›", id="next-day", n_clicks=0, style={
                "width": "28px", "height": "32px", "background": COLORS["surface2"],
                "border": f"1px solid {COLORS['border2']}", "color": COLORS["text2"],
                "borderRadius": "4px", "cursor": "pointer", "fontSize": "16px",
                "flexShrink": "0",
            }),
        ], style={"display": "flex", "alignItems": "center", "gap": "6px"}),
        html.Div("Forecasts available for 2024 only",
                 style={"fontSize": "11px", "color": COLORS["text3"]}),
    ], style={"display": "flex", "flexDirection": "column", "gap": "10px"}),

    # Parameters
    html.Div([
        html.Div("Battery Parameters", style={"fontSize": "10px", "fontWeight": "700",
                 "letterSpacing": "0.1em", "color": COLORS["text3"],
                 "textTransform": "uppercase", "paddingBottom": "6px",
                 "borderBottom": f"1px solid {COLORS['border']}"}),
        slider_row("Capacity",            "sl-cap",  10,  500, 10,  100, "MWh"),
        slider_row("Max Power",           "sl-pw",    5,  250,  5,   50, "MW"),
        slider_row("Round-trip Efficiency","sl-eff",  60,   99,  1,   90, "%"),
        slider_row("Initial SOC",         "sl-soc",   0,  100,  5,   50, "%"),
    ], style={"display": "flex", "flexDirection": "column", "gap": "12px"}),

    # Presets
    html.Div([
        html.Div("Presets", style={"fontSize": "10px", "fontWeight": "700",
                 "letterSpacing": "0.1em", "color": COLORS["text3"],
                 "textTransform": "uppercase", "paddingBottom": "6px",
                 "borderBottom": f"1px solid {COLORS['border']}"}),
        html.Div([
            html.Button([html.Strong("Cathkin"), html.Br(), html.Small("100 MWh · 50 MW")],
                        id="preset-cathkin", n_clicks=0, className="preset-btn"),
            html.Button([html.Strong("Small"),   html.Br(), html.Small("50 MWh · 25 MW")],
                        id="preset-small",   n_clicks=0, className="preset-btn"),
            html.Button([html.Strong("Large"),   html.Br(), html.Small("200 MWh · 100 MW")],
                        id="preset-large",   n_clicks=0, className="preset-btn"),
            html.Button([html.Strong("LDES"),    html.Br(), html.Small("400 MWh · 100 MW")],
                        id="preset-ldes",    n_clicks=0, className="preset-btn"),
        ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "6px"}),
    ], style={"display": "flex", "flexDirection": "column", "gap": "10px"}),

    # Run
    html.Button("⚡  Optimise Dispatch", id="run-btn", n_clicks=0, style={
        "width": "100%", "padding": "11px", "background": COLORS["accent"],
        "color": "#fff", "border": "none", "borderRadius": "6px",
        "fontFamily": "Inter, sans-serif", "fontSize": "13px", "fontWeight": "700",
        "cursor": "pointer", "letterSpacing": "0.02em",
    }),
    html.Div("", id="solver-status", style={
        "fontSize": "11px", "textAlign": "center", "color": COLORS["text3"],
        "minHeight": "16px",
    }),

], style={
    "width": "260px", "flexShrink": "0",
    "background": COLORS["surface"], "borderRight": f"1px solid {COLORS['border']}",
    "padding": "20px 16px", "display": "flex", "flexDirection": "column",
    "gap": "22px", "overflowY": "auto",
})

MAIN = html.Div([

    # KPIs
    html.Div([
        kpi_card("Daily Revenue",  "kpi-revenue", COLORS["green"],  "GBP / day"),
        kpi_card("Avg Buy Price",  "kpi-buy",     COLORS["accent"], "£/MWh"),
        kpi_card("Avg Sell Price", "kpi-sell",    COLORS["orange"], "£/MWh"),
        kpi_card("Price Spread",   "kpi-spread",  COLORS["yellow"], "£/MWh captured"),
        kpi_card("Energy Cycled",  "kpi-cycled",  COLORS["purple"], "MWh sold"),
        html.Div([
            html.Div("Full Cycles", style={"fontSize": "10px", "fontWeight": "700",
                     "letterSpacing": "0.08em", "color": COLORS["text3"],
                     "textTransform": "uppercase"}),
            html.Div("—", id="kpi-cycles", style={
                "fontFamily": "JetBrains Mono, monospace", "fontSize": "22px",
                "fontWeight": "700", "color": COLORS["text"], "margin": "4px 0 2px",
            }),
            html.Div("equiv. full cycles", style={"fontSize": "10px", "color": COLORS["text3"]}),
        ], style={"padding": "14px 16px", "flex": "1"}),
    ], style={
        "display": "flex", "borderBottom": f"1px solid {COLORS['border']}",
    }),

    # Charts
    html.Div([
        html.Div([
            html.Div([
                html.Span("Price Forecast & Dispatch Schedule", style={
                    "fontSize": "11px", "fontWeight": "700",
                    "letterSpacing": "0.06em", "color": COLORS["text2"],
                    "textTransform": "uppercase",
                }),
            ], style={
                "padding": "10px 16px", "borderBottom": f"1px solid {COLORS['border']}",
            }),
            dcc.Graph(id="chart-dispatch", config={"displayModeBar": False},
                      style={"height": "100%"}),
        ], style={
            "flex": "1", "background": COLORS["surface"],
            "border": f"1px solid {COLORS['border']}", "borderRadius": "8px",
            "display": "flex", "flexDirection": "column", "overflow": "hidden",
        }),

        html.Div([
            html.Div([
                html.Span("State of Charge Evolution", style={
                    "fontSize": "11px", "fontWeight": "700",
                    "letterSpacing": "0.06em", "color": COLORS["text2"],
                    "textTransform": "uppercase",
                }),
            ], style={
                "padding": "10px 16px", "borderBottom": f"1px solid {COLORS['border']}",
            }),
            dcc.Graph(id="chart-soc", config={"displayModeBar": False},
                      style={"height": "100%"}),
        ], style={
            "flex": "1", "background": COLORS["surface"],
            "border": f"1px solid {COLORS['border']}", "borderRadius": "8px",
            "display": "flex", "flexDirection": "column", "overflow": "hidden",
        }),

    ], style={
        "flex": "1", "display": "flex", "flexDirection": "column",
        "gap": "12px", "padding": "16px", "overflow": "auto",
    }),

], style={"flex": "1", "display": "flex", "flexDirection": "column", "overflow": "hidden"})

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG,
                           "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap"],
    title="BESS Dispatch Optimiser",
    suppress_callback_exceptions=True,
)
server = app.server

app.layout = html.Div([

    # Top bar
    html.Div([
        html.Div([
            html.Span("BESS Dispatch Optimiser — GB Intra-Day",
                      style={"fontSize": "13px", "fontWeight": "600", "color": COLORS["text"]}),
        ], style={"display": "flex", "alignItems": "center", "gap": "16px"}),
        html.Div([
            html.Div(style={
                "width": "7px", "height": "7px", "borderRadius": "50%",
                "background": COLORS["green"], "boxShadow": f"0 0 6px {COLORS['green']}",
            }),
            html.Span("CBC Solver", style={"fontSize": "11px", "color": COLORS["text3"]}),
        ], style={"display": "flex", "alignItems": "center", "gap": "8px"}),
    ], style={
        "display": "flex", "alignItems": "center", "justifyContent": "space-between",
        "padding": "0 20px", "height": "48px",
        "background": COLORS["surface"], "borderBottom": f"1px solid {COLORS['border']}",
        "position": "sticky", "top": "0", "zIndex": "100",
    }),

    # Body
    html.Div([SIDEBAR, MAIN], style={
        "display": "flex", "height": "calc(100vh - 48px)", "overflow": "hidden",
    }),

], style={"background": COLORS["bg"], "minHeight": "100vh",
          "fontFamily": "Inter, sans-serif", "color": COLORS["text"]})

# ── CALLBACKS ─────────────────────────────────────────────────────────────────

# Slider labels
for _id, _unit in [("sl-cap","MWh"),("sl-pw","MW"),("sl-eff","%"),("sl-soc","%")]:
    @app.callback(Output(f"lbl-{_id}", "children"), Input(_id, "value"),
                  prevent_initial_call=True)
    def _lbl(v, unit=_unit): return f"{v} {unit}"


# Presets
@app.callback(
    Output("sl-cap", "value"), Output("sl-pw", "value"),
    Output("sl-eff", "value"), Output("sl-soc", "value"),
    Input("preset-cathkin", "n_clicks"), Input("preset-small", "n_clicks"),
    Input("preset-large", "n_clicks"),   Input("preset-ldes",  "n_clicks"),
    prevent_initial_call=True,
)
def apply_preset(c1, c2, c3, c4):
    presets = {
        "preset-cathkin": (100, 50,  90, 50),
        "preset-small":   (50,  25,  88, 50),
        "preset-large":   (200, 100, 92, 50),
        "preset-ldes":    (400, 100, 85, 20),
    }
    triggered = dash.ctx.triggered_id
    cap, pw, eff, soc = presets.get(triggered, (100, 50, 90, 50))
    return cap, pw, eff, soc


# Date navigation
@app.callback(
    Output("date-picker", "date"),
    Input("prev-day", "n_clicks"), Input("next-day", "n_clicks"),
    State("date-picker", "date"),
    prevent_initial_call=True,
)
def shift_date(prev, nxt, current):
    from datetime import date as dt_date
    d = pd.Timestamp(current)
    delta = -1 if dash.ctx.triggered_id == "prev-day" else 1
    new_d = (d + timedelta(days=delta)).strftime("%Y-%m-%d")
    return new_d if new_d in set(AVAILABLE_DATES) else current


# Main optimisation
@app.callback(
    Output("chart-dispatch", "figure"),
    Output("chart-soc",      "figure"),
    Output("kpi-revenue",    "children"),
    Output("kpi-buy",        "children"),
    Output("kpi-sell",       "children"),
    Output("kpi-spread",     "children"),
    Output("kpi-cycled",     "children"),
    Output("kpi-cycles",     "children"),
    Output("solver-status",  "children"),
    Output("solver-status",  "style"),
    Input("run-btn",    "n_clicks"),
    State("date-picker","date"),
    State("sl-cap",     "value"),
    State("sl-pw",      "value"),
    State("sl-eff",     "value"),
    State("sl-soc",     "value"),
)
def run_optimise(n, date_str, cap, max_pw, eff_pct, soc_pct):
    ts = [f"{h:02d}:00" for h in range(24)]
    eff = eff_pct / 100

    res = optimise(date_str, cap, max_pw, eff, soc_pct)
    if res is None:
        empty = go.Figure(layout=PLOT_LAYOUT)
        err_style = {"fontSize":"11px","textAlign":"center","color":COLORS["red"]}
        return empty, empty, "—","—","—","—","—","—", "No data for this date", err_style

    prices, charge, discharge, soc = res["prices"], res["charge"], res["discharge"], res["soc"]

    # Chart 1 — Price + Dispatch
    max_bar = max(max(charge), max(discharge), 1) * 1.3
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=ts, y=prices, name="ID Forecast (£/MWh)",
        line=dict(color=COLORS["accent"], width=2), yaxis="y1",
    ))
    fig1.add_trace(go.Bar(
        x=ts, y=charge, name="Charge (MWh)",
        marker_color="rgba(34,197,94,0.75)", yaxis="y2",
    ))
    fig1.add_trace(go.Bar(
        x=ts, y=[-v for v in discharge], name="Discharge (MWh)",
        marker_color="rgba(239,68,68,0.75)", yaxis="y2",
    ))
    layout1 = {**PLOT_LAYOUT,
        "barmode": "relative",
        "yaxis":  {**PLOT_LAYOUT["yaxis"], "title": {"text": "£/MWh", "font": {"size": 11}}},
        "yaxis2": {**PLOT_LAYOUT["yaxis"], "title": {"text": "MWh", "font": {"size": 11}},
                   "overlaying": "y", "side": "right", "range": [-max_bar, max_bar]},
    }
    fig1.update_layout(**layout1)

    # Chart 2 — SOC
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=ts, y=soc, name="SOC (MWh)",
        fill="tozeroy", fillcolor="rgba(168,85,247,0.12)",
        line=dict(color=COLORS["purple"], width=2),
    ))
    fig2.add_trace(go.Scatter(
        x=[ts[0], ts[-1]], y=[cap, cap], name="Max capacity",
        line=dict(color="#374151", width=1, dash="dot"), hoverinfo="skip",
    ))
    fig2.add_trace(go.Scatter(
        x=[ts[0], ts[-1]], y=[0, 0], name="Min (0 MWh)",
        line=dict(color="#374151", width=1, dash="dot"), hoverinfo="skip",
    ))
    layout2 = {**PLOT_LAYOUT,
        "yaxis": {**PLOT_LAYOUT["yaxis"], "title": {"text": "MWh", "font": {"size": 11}},
                  "range": [-5, cap * 1.1]},
    }
    fig2.update_layout(**layout2)

    # KPIs
    ok_style = {"fontSize":"11px","textAlign":"center","color":COLORS["green"]}
    fmt = lambda v, d=0: f"£{v:,.{d}f}" if d==0 else f"{v:,.{d}f}"

    return (
        fig1, fig2,
        f"£{res['revenue']:,.0f}",
        f"{res['avg_buy']:.1f}",
        f"{res['avg_sell']:.1f}",
        f"{res['spread']:.1f}",
        f"{res['energy_sold']:.1f}",
        f"{res['cycles']:.2f}",
        "Optimal solution found",
        ok_style,
    )


# Auto-run on load
app.clientside_callback(
    "function(n) { return n + 1; }",
    Output("run-btn", "n_clicks"),
    Input("run-btn", "id"),
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
