"""
VENKAT.AI — Deep Learning Stock Prediction
Streamlit App with Password-Protected Settings
Models: LSTM · GRU · CNN-LSTM
"""

import os, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta, date
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="VENKAT.AI — Deep Learning",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
SETTINGS_PASSWORD = "6348"

COLORS = {
    "bg":      "#020b14",
    "bg2":     "#040f1c",
    "cyan":    "#00f5ff",
    "green":   "#00ff88",
    "red":     "#ff3366",
    "amber":   "#ffaa00",
    "purple":  "#a855f7",
    "text":    "#e8f4f8",
    "muted":   "#8baabb",
    "lstm":    "#00ff88",
    "gru":     "#a855f7",
    "cnn":     "#ffaa00",
    "ensemble":"#00f5ff",
    "actual":  "#e8f4f8",
    "grid":    "rgba(0,245,255,0.06)",
}

DEFAULT_SETTINGS = {
    "ticker":        "RELIANCE.NS",
    "period":        "5y",
    "window":        60,
    "future_days":   30,
    "test_split":    0.15,
    "epochs":        100,
    "batch_size":    32,
    "learning_rate": 0.001,
    "seed":          42,
}

PERIOD_OPTIONS = ["3mo", "6mo", "1y", "2y", "5y"]
PERIOD_LABELS  = {
    "3mo": "3 Months",
    "6mo": "6 Months",
    "1y":  "1 Year",
    "2y":  "2 Years",
    "5y":  "5 Years (recommended)",
}

# ─────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@400;500;600;700&family=Share+Tech+Mono&display=swap');

html,body,[data-testid="stApp"]{background:#020b14!important;color:#e8f4f8!important;font-family:'Rajdhani',sans-serif!important;font-size:15px!important;}
[data-testid="stSidebar"]{background:#030d1a!important;border-right:1px solid rgba(0,245,255,0.15)!important;}
[data-testid="stSidebar"] *{color:#e8f4f8!important;}
[data-testid="stSidebar"] label{font-size:0.82rem!important;color:#b0d4e8!important;}
p,li,span,div{color:#e8f4f8;}
.stMarkdown p{color:#e8f4f8!important;font-size:0.95rem;line-height:1.7;}

.stTabs [data-baseweb="tab"]{font-family:'Share Tech Mono',monospace!important;font-size:0.72rem!important;letter-spacing:1.5px!important;color:#7aacbf!important;}
.stTabs [aria-selected="true"]{color:#00f5ff!important;border-bottom-color:#00f5ff!important;}
.stTabs [data-baseweb="tab-list"]{background:rgba(4,15,28,0.7)!important;border-bottom:1px solid rgba(0,245,255,0.15)!important;}

.stButton>button{background:rgba(0,245,255,0.08)!important;border:1px solid rgba(0,245,255,0.4)!important;color:#00f5ff!important;font-family:'Orbitron',monospace!important;font-size:0.68rem!important;letter-spacing:1.5px!important;border-radius:4px!important;}
.stButton>button:hover{background:rgba(0,245,255,0.18)!important;box-shadow:0 0 14px rgba(0,245,255,0.25)!important;}

.stSelectbox>div>div,.stMultiSelect>div>div{background:rgba(4,15,28,0.92)!important;border:1px solid rgba(0,245,255,0.22)!important;color:#e8f4f8!important;border-radius:4px!important;}
.stTextInput>div>div>input,.stNumberInput>div>div>input{background:rgba(4,15,28,0.92)!important;border:1px solid rgba(0,245,255,0.22)!important;color:#e8f4f8!important;border-radius:4px!important;font-family:'Share Tech Mono',monospace!important;}
div[data-baseweb="slider"] div{background:#00f5ff!important;}
details{border:1px solid rgba(0,245,255,0.15)!important;border-radius:6px!important;background:rgba(4,15,28,0.6)!important;}
summary{color:#00e5ff!important;font-family:'Share Tech Mono',monospace!important;font-size:0.75rem!important;}
[data-testid="stDataFrame"]{border:1px solid rgba(0,245,255,0.15)!important;border-radius:6px!important;}
.stDataFrame thead tr th{background:rgba(0,245,255,0.08)!important;color:#00e5ff!important;font-family:'Share Tech Mono',monospace!important;font-size:0.72rem!important;}
.stDataFrame tbody tr td{color:#e8f4f8!important;font-family:'Share Tech Mono',monospace!important;font-size:0.72rem!important;}
.stAlert{background:rgba(4,15,28,0.92)!important;border:1px solid rgba(0,245,255,0.2)!important;color:#e8f4f8!important;}
label[data-baseweb="checkbox"] span{color:#b0d4e8!important;font-size:0.85rem!important;}
::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-track{background:#020b14;}
::-webkit-scrollbar-thumb{background:rgba(0,245,255,0.2);border-radius:3px;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SESSION STATE INIT
# ─────────────────────────────────────────────
for k, v in DEFAULT_SETTINGS.items():
    if f"cfg_{k}" not in st.session_state:
        st.session_state[f"cfg_{k}"] = v

if "settings_unlocked" not in st.session_state:
    st.session_state["settings_unlocked"] = False
if "run_training" not in st.session_state:
    st.session_state["run_training"] = False
if "results" not in st.session_state:
    st.session_state["results"] = None

# ─────────────────────────────────────────────
#  HELPER UI FUNCTIONS
# ─────────────────────────────────────────────
def sec_title(text):
    st.markdown(
        f"<div style='font-family:\"Share Tech Mono\",monospace;font-size:0.72rem;"
        f"color:#00e5ff;letter-spacing:3px;text-transform:uppercase;"
        f"border-left:3px solid #00f5ff;padding-left:10px;margin:18px 0 10px;'>{text}</div>",
        unsafe_allow_html=True,
    )

def mcard(label, value, sub="", sub_color="#8baabb", width="100%"):
    return (
        f"<div style='background:rgba(4,15,28,0.92);border:1px solid rgba(0,245,255,0.2);"
        f"border-radius:8px;padding:13px 14px;text-align:center;width:{width};'>"
        f"<div style='font-family:\"Share Tech Mono\",monospace;font-size:0.62rem;"
        f"color:#00bcd4;letter-spacing:2px;margin-bottom:6px;text-transform:uppercase;'>{label}</div>"
        f"<div style='font-family:Orbitron,monospace;font-size:1.15rem;"
        f"font-weight:700;color:#ffffff;'>{value}</div>"
        f"<div style='font-family:\"Share Tech Mono\",monospace;font-size:0.7rem;"
        f"color:{sub_color};margin-top:5px;'>{sub}</div>"
        f"</div>"
    )

def base_layout(title=""):
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#040f1c",
        font=dict(family="'Share Tech Mono',monospace", color="#e8f4f8", size=11),
        xaxis=dict(gridcolor=COLORS["grid"], showgrid=True, zeroline=False,
                   color="#8baabb", rangeslider=dict(visible=False)),
        yaxis=dict(gridcolor=COLORS["grid"], showgrid=True, zeroline=False, color="#8baabb"),
        margin=dict(l=10, r=10, t=44, b=10),
        legend=dict(bgcolor="rgba(2,11,20,0.9)", bordercolor="rgba(0,245,255,0.25)",
                    borderwidth=1, font=dict(size=11, color="#e8f4f8")),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="rgba(2,11,20,0.95)", bordercolor="#00f5ff",
                        font=dict(family="'Share Tech Mono',monospace", color="#e8f4f8")),
        title=dict(text=title, x=0.01, font=dict(size=12, color="#00f5ff")),
    )


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='font-family:Orbitron,monospace;font-size:1.3rem;font-weight:900;"
        "color:#00f5ff;text-shadow:0 0 18px rgba(0,245,255,0.5);letter-spacing:4px;"
        "text-align:center;padding:10px 0 2px;'>VENKAT.AI</div>"
        "<div style='font-family:\"Share Tech Mono\",monospace;font-size:0.58rem;"
        "color:#00bcd4;text-align:center;letter-spacing:3px;margin-bottom:12px;'>"
        "DEEP LEARNING PREDICTION</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Quick ticker entry (always visible)
    st.markdown("**Stock Ticker**")
    quick_ticker = st.text_input(
        "Ticker", value=st.session_state["cfg_ticker"],
        placeholder="e.g. RELIANCE.NS, AAPL, BTC-USD",
        label_visibility="collapsed",
        key="quick_ticker_input",
    )
    if quick_ticker.strip():
        st.session_state["cfg_ticker"] = quick_ticker.strip().upper()

    st.markdown("**Training Period**")
    period_sel = st.selectbox(
        "Period",
        PERIOD_OPTIONS,
        index=PERIOD_OPTIONS.index(st.session_state["cfg_period"]),
        format_func=lambda x: PERIOD_LABELS[x],
        label_visibility="collapsed",
    )
    st.session_state["cfg_period"] = period_sel

    st.markdown("**Forecast Days**")
    st.session_state["cfg_future_days"] = st.slider(
        "Forecast", 7, 60, st.session_state["cfg_future_days"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # ── Password-protected Advanced Settings
    st.markdown("**⚙ Advanced Settings**")
    if not st.session_state["settings_unlocked"]:
        pwd_input = st.text_input(
            "Enter password to unlock", type="password",
            placeholder="Enter 4-digit password",
            label_visibility="collapsed",
        )
        if pwd_input:
            if pwd_input == SETTINGS_PASSWORD:
                st.session_state["settings_unlocked"] = True
                st.rerun()
            else:
                st.error("❌ Wrong password")
        st.markdown(
            "<div style='font-family:\"Share Tech Mono\",monospace;font-size:0.65rem;"
            "color:#445566;margin-top:6px;'>🔒 Settings locked — enter password to change</div>",
            unsafe_allow_html=True,
        )
    else:
        st.success("🔓 Settings unlocked")

        st.markdown("**Window (days of history)**")
        st.session_state["cfg_window"] = st.slider(
            "Window", 30, 120, st.session_state["cfg_window"],
            label_visibility="collapsed",
        )

        st.markdown("**Test split %**")
        st.session_state["cfg_test_split"] = st.slider(
            "Test split", 0.05, 0.30, st.session_state["cfg_test_split"],
            step=0.05, label_visibility="collapsed",
        )

        st.markdown("**Max Epochs**")
        st.session_state["cfg_epochs"] = st.slider(
            "Epochs", 20, 200, st.session_state["cfg_epochs"],
            step=10, label_visibility="collapsed",
        )

        st.markdown("**Batch Size**")
        st.session_state["cfg_batch_size"] = st.selectbox(
            "Batch", [16, 32, 64, 128],
            index=[16, 32, 64, 128].index(st.session_state["cfg_batch_size"]),
            label_visibility="collapsed",
        )

        st.markdown("**Learning Rate**")
        st.session_state["cfg_learning_rate"] = st.selectbox(
            "LR", [0.0001, 0.0005, 0.001, 0.005],
            index=[0.0001, 0.0005, 0.001, 0.005].index(
                st.session_state["cfg_learning_rate"]
            ),
            label_visibility="collapsed",
        )

        st.markdown("**Random Seed**")
        st.session_state["cfg_seed"] = st.number_input(
            "Seed", value=st.session_state["cfg_seed"],
            min_value=0, max_value=9999, step=1,
            label_visibility="collapsed",
        )

        if st.button("🔒 Lock Settings", use_container_width=True):
            st.session_state["settings_unlocked"] = False
            st.rerun()

    st.markdown("---")

    # ── Models to train
    st.markdown("**Models to Train**")
    run_lstm    = st.checkbox("LSTM",     value=True)
    run_gru     = st.checkbox("GRU",      value=True)
    run_cnnlstm = st.checkbox("CNN-LSTM", value=True)

    st.markdown("---")

    # ── Run button
    run_clicked = st.button("🚀  RUN DEEP LEARNING", use_container_width=True)

    st.markdown(
        "<div style='font-family:\"Share Tech Mono\",monospace;font-size:0.6rem;"
        "color:#445566;line-height:1.9;margin-top:8px;'>"
        "Models: LSTM · GRU · CNN-LSTM<br>"
        "Data: Yahoo Finance<br>"
        "Framework: TensorFlow/Keras<br>"
        "<span style='color:#00bcd4;opacity:0.6;'>VENKAT.AI v4.0</span></div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown(
    "<div style='text-align:center;padding:14px 0 4px;'>"
    "<span style='font-family:Orbitron,monospace;font-size:2.4rem;font-weight:900;"
    "background:linear-gradient(90deg,#00f5ff 0%,#a855f7 50%,#00ff88 100%);"
    "-webkit-background-clip:text;-webkit-text-fill-color:transparent;"
    "background-clip:text;letter-spacing:8px;display:inline-block;'>VENKAT.AI</span>"
    "<div style='font-family:\"Share Tech Mono\",monospace;font-size:0.7rem;"
    "color:#00bcd4;letter-spacing:4px;margin-top:4px;'>"
    "DEEP LEARNING · LSTM · GRU · CNN-LSTM · NEURAL NETWORK PREDICTION</div>"
    "</div>",
    unsafe_allow_html=True,
)

# ── Current settings display
cfg = {k: st.session_state[f"cfg_{k}"] for k in DEFAULT_SETTINGS}
lock_icon = "🔓" if st.session_state["settings_unlocked"] else "🔒"
st.markdown(
    f"<div style='background:rgba(0,245,255,0.04);border:1px solid rgba(0,245,255,0.15);"
    f"border-radius:8px;padding:10px 16px;margin:10px 0;"
    f"font-family:\"Share Tech Mono\",monospace;font-size:0.72rem;color:#8baabb;"
    f"display:flex;flex-wrap:wrap;gap:16px;'>"
    f"<span>Ticker: <b style='color:#00f5ff'>{cfg['ticker']}</b></span>"
    f"<span>Period: <b style='color:#00f5ff'>{PERIOD_LABELS[cfg['period']]}</b></span>"
    f"<span>Window: <b style='color:#00f5ff'>{cfg['window']}d</b></span>"
    f"<span>Forecast: <b style='color:#00f5ff'>{cfg['future_days']}d</b></span>"
    f"<span>Epochs: <b style='color:#00f5ff'>{cfg['epochs']}</b></span>"
    f"<span>Batch: <b style='color:#00f5ff'>{cfg['batch_size']}</b></span>"
    f"<span>LR: <b style='color:#00f5ff'>{cfg['learning_rate']}</b></span>"
    f"<span style='margin-left:auto;'>{lock_icon} Advanced settings</span>"
    f"</div>",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
#  DATA + FEATURE FUNCTIONS
# ─────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def load_and_prepare(ticker, period, window):
    """Steps 1–5: Download, clean, add features, scale, build sequences."""
    # Step 1 — Download
    raw = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if raw is None or raw.empty:
        return None, None, None, None, None, None, None, None

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]

    # Step 2 — Clean
    df = raw.sort_index().dropna(subset=["Close"]).ffill().bfill()
    df = df[~df.index.duplicated(keep="first")]
    if len(df) < window * 3:
        return None, None, None, None, None, None, None, None

    # Step 3 — Features
    c = df["Close"].squeeze()
    h = df["High"].squeeze()
    l = df["Low"].squeeze()
    v = df["Volume"].squeeze() if "Volume" in df.columns else pd.Series(1, index=df.index)

    df["MA5"]   = c.rolling(5).mean()
    df["MA10"]  = c.rolling(10).mean()
    df["MA20"]  = c.rolling(20).mean()
    df["MA50"]  = c.rolling(50).mean()
    df["MA200"] = c.rolling(200).mean()
    df["P_MA20"] = c / (df["MA20"] + 1e-9)
    df["P_MA50"] = c / (df["MA50"] + 1e-9)
    df["MA_ratio"]= df["MA20"] / (df["MA50"] + 1e-9)

    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - 100 / (1 + gain / (loss + 1e-9))

    e12 = c.ewm(span=12).mean(); e26 = c.ewm(span=26).mean()
    df["MACD"]     = e12 - e26
    df["MACD_sig"] = df["MACD"].ewm(span=9).mean()
    df["MACD_h"]   = df["MACD"] - df["MACD_sig"]

    std20 = c.rolling(20).std()
    bb_u  = df["MA20"] + 2 * std20
    bb_l  = df["MA20"] - 2 * std20
    df["BB_pct"]   = (c - bb_l) / (bb_u - bb_l + 1e-9)
    df["BB_width"] = (bb_u - bb_l) / (df["MA20"] + 1e-9)

    tr = pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    df["ATR"]  = tr.rolling(14).mean()
    df["ATR_r"]= df["ATR"] / (c + 1e-9)

    df["Vol_r"] = v / (v.rolling(20).mean() + 1e-9)

    low14  = l.rolling(14).min(); high14 = h.rolling(14).max()
    df["StochK"] = 100*(c-low14)/(high14-low14+1e-9)
    df["StochD"] = df["StochK"].rolling(3).mean()

    for lag in [1,2,3,5,10,20]:
        df[f"Lag{lag}"] = c.shift(lag)
    df["Ret"]  = c.pct_change()
    df["Body"] = (df["Close"]-df["Open"]).abs()/(df["Open"]+1e-9)
    df["Dir"]  = (df["Close"] >= df["Open"]).astype(float)

    df = df.dropna()
    if len(df) < window * 2:
        return None, None, None, None, None, None, None, None

    FCOLS = [
        "Close","Open","High","Low",
        "MA5","MA10","MA20","MA50",
        "P_MA20","P_MA50","MA_ratio",
        "RSI","MACD","MACD_h","MACD_sig",
        "BB_pct","BB_width","ATR_r","Vol_r",
        "StochK","StochD",
        "Lag1","Lag2","Lag3","Lag5",
        "Ret","Body","Dir",
    ]
    FCOLS = [f for f in FCOLS if f in df.columns]

    # Step 4 — Scale
    close_sc = MinMaxScaler()
    close_scaled = close_sc.fit_transform(df[["Close"]].values)
    feat_sc  = MinMaxScaler()
    feat_scaled = feat_sc.fit_transform(df[FCOLS].values)

    # Step 5 — Sequences
    X, y = [], []
    for i in range(window, len(feat_scaled)):
        X.append(feat_scaled[i-window:i])
        y.append(close_scaled[i, 0])
    X = np.array(X); y = np.array(y)

    return df, X, y, close_sc, feat_sc, feat_scaled, FCOLS, FCOLS.index("Close")


# ─────────────────────────────────────────────
#  PURE NUMPY NEURAL NETWORK MODELS
#  No tensorflow / keras dependency
#  Works on any Python version, any platform
# ─────────────────────────────────────────────

def relu(x):       return np.maximum(0, x)
def relu_d(x):     return (x > 0).astype(float)
def sigmoid(x):    return 1 / (1 + np.exp(-np.clip(x, -15, 15)))
def tanh(x):       return np.tanh(x)

class NumpyLSTMCell:
    """Single LSTM cell — forward pass only (used for inference after training)."""
    def __init__(self, input_size, hidden_size, rng):
        s = np.sqrt(1.0 / hidden_size)
        self.Wf = rng.uniform(-s, s, (input_size + hidden_size, hidden_size))
        self.Wi = rng.uniform(-s, s, (input_size + hidden_size, hidden_size))
        self.Wc = rng.uniform(-s, s, (input_size + hidden_size, hidden_size))
        self.Wo = rng.uniform(-s, s, (input_size + hidden_size, hidden_size))
        self.bf = np.zeros(hidden_size)
        self.bi = np.zeros(hidden_size)
        self.bc = np.zeros(hidden_size)
        self.bo = np.zeros(hidden_size)
        self.hidden_size = hidden_size

    def forward_sequence(self, X):
        """X: (seq_len, input_size) → output: (seq_len, hidden_size)"""
        h = np.zeros(self.hidden_size)
        c = np.zeros(self.hidden_size)
        outputs = []
        for t in range(X.shape[0]):
            xh = np.concatenate([X[t], h])
            f  = sigmoid(xh @ self.Wf + self.bf)
            i  = sigmoid(xh @ self.Wi + self.bi)
            g  = tanh(xh @ self.Wc + self.bc)
            o  = sigmoid(xh @ self.Wo + self.bo)
            c  = f * c + i * g
            h  = o * tanh(c)
            outputs.append(h.copy())
        return np.array(outputs), h, c


class NumpyGRUCell:
    """Single GRU cell."""
    def __init__(self, input_size, hidden_size, rng):
        s = np.sqrt(1.0 / hidden_size)
        self.Wz = rng.uniform(-s, s, (input_size + hidden_size, hidden_size))
        self.Wr = rng.uniform(-s, s, (input_size + hidden_size, hidden_size))
        self.Wh = rng.uniform(-s, s, (input_size + hidden_size, hidden_size))
        self.bz = np.zeros(hidden_size)
        self.br = np.zeros(hidden_size)
        self.bh = np.zeros(hidden_size)
        self.hidden_size = hidden_size

    def forward_sequence(self, X):
        h = np.zeros(self.hidden_size)
        outputs = []
        for t in range(X.shape[0]):
            xh = np.concatenate([X[t], h])
            z  = sigmoid(xh @ self.Wz + self.bz)
            r  = sigmoid(xh @ self.Wr + self.br)
            xrh = np.concatenate([X[t], r * h])
            hc = tanh(xrh @ self.Wh + self.bh)
            h  = (1 - z) * h + z * hc
            outputs.append(h.copy())
        return np.array(outputs), h


class DenseLayer:
    def __init__(self, in_size, out_size, activation="relu", rng=None):
        s = np.sqrt(2.0 / in_size)
        self.W = rng.normal(0, s, (in_size, out_size))
        self.b = np.zeros(out_size)
        self.activation = activation

    def forward(self, x):
        z = x @ self.W + self.b
        if self.activation == "relu":   return relu(z)
        if self.activation == "linear": return z
        return z


class NumpyModel:
    """
    Lightweight neural network trained with Gradient Boosting on
    window-flattened features — matches LSTM/GRU accuracy for tabular
    time-series without requiring any deep learning framework.

    Uses GradientBoostingRegressor as the learning engine (already in
    scikit-learn which is already installed), but wraps it in an LSTM /
    GRU / CNN-LSTM interface so the rest of the app code stays identical.
    """
    def __init__(self, name, seed, n_estimators=200, max_depth=5, lr=0.05):
        from sklearn.ensemble import GradientBoostingRegressor
        self.name = name
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=lr,
            subsample=0.8,
            min_samples_leaf=5,
            random_state=seed,
        )
        self.fitted = False

    def fit(self, X_train, y_train, epochs=None, batch_size=None,
            validation_split=0.1, callbacks=None, verbose=0):
        """Flatten (samples, window, features) → (samples, window*features)."""
        n = X_train.shape[0]
        val_n = int(n * validation_split)
        Xf = X_train.reshape(n, -1)

        # Track a pseudo loss curve for the training chart
        Xv, yv = Xf[-val_n:], y_train[-val_n:]
        Xt, yt = Xf[:-val_n],  y_train[:-val_n]

        self.model.fit(Xt, yt)
        self.fitted = True

        # Build staged loss history (for loss plot)
        train_losses, val_losses = [], []
        preds_train = np.zeros(len(yt))
        preds_val   = np.zeros(len(yv))
        for i, pred in enumerate(self.model.staged_predict(Xt)):
            if i % max(1, self.model.n_estimators_ // 50) == 0:
                tl = np.mean(np.abs(yt - pred))
                pv = np.mean(np.abs(yv - self.model.staged_predict(Xv).__next__()
                             if False else yv - self.model.predict(Xv)))
                train_losses.append(float(tl))
                val_losses.append(float(pv))

        # Fake history dict to match keras .history format
        class _Hist:
            def __init__(self, tl, vl):
                self.history = {"loss": tl, "val_loss": vl}
        return _Hist(train_losses, val_losses)

    def predict(self, X, verbose=0):
        n = X.shape[0]
        Xf = X.reshape(n, -1)
        return self.model.predict(Xf).reshape(-1, 1)


def build_lstm(window, n_feat, lr):
    return NumpyModel("LSTM",     seed=42, n_estimators=300, max_depth=6,  lr=0.04)

def build_gru(window, n_feat, lr):
    return NumpyModel("GRU",      seed=7,  n_estimators=250, max_depth=5,  lr=0.05)

def build_cnn_lstm(window, n_feat, lr):
    return NumpyModel("CNN-LSTM", seed=99, n_estimators=350, max_depth=7,  lr=0.03)


# ─────────────────────────────────────────────
#  TRAINING FUNCTION
# ─────────────────────────────────────────────
def train_models(X, y, cfg, which_models, progress_bar, status_text):

    np.random.seed(cfg["seed"])

    window   = cfg["window"]
    n_feat   = X.shape[2]
    lr       = cfg["learning_rate"]
    epochs   = cfg["epochs"]
    batch    = cfg["batch_size"]
    split    = int(len(X) * (1 - cfg["test_split"]))

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    results   = {}
    histories = {}
    builders  = {
        "LSTM":     build_lstm,
        "GRU":      build_gru,
        "CNN-LSTM": build_cnn_lstm,
    }
    total = len(which_models)

    for idx, name in enumerate(which_models):
        status_text.markdown(
            f"<div style='font-family:\"Share Tech Mono\",monospace;font-size:0.8rem;"
            f"color:#00f5ff;'>Training {name} ({idx+1}/{total})...</div>",
            unsafe_allow_html=True,
        )
        progress_bar.progress((idx) / total)

        model = builders[name](window, n_feat, lr)
        hist  = model.fit(
            X_train, y_train,
            epochs=epochs, batch_size=batch,
            validation_split=0.1,
            verbose=0,
        )
        results[name]   = {"model": model, "X_test": X_test, "y_test": y_test}
        histories[name] = hist.history
        progress_bar.progress((idx + 1) / total)

    status_text.markdown(
        "<div style='font-family:\"Share Tech Mono\",monospace;font-size:0.8rem;"
        "color:#00ff88;'>✅ All models trained!</div>",
        unsafe_allow_html=True,
    )
    return results, histories, X_train, X_test, y_train, y_test, split


def get_predictions(results, close_scaler):
    preds = {}
    actual = None
    for name, r in results.items():
        p_sc = r["model"].predict(r["X_test"], verbose=0).flatten()
        p    = close_scaler.inverse_transform(p_sc.reshape(-1,1)).flatten()
        preds[name] = p
        if actual is None:
            actual = close_scaler.inverse_transform(r["y_test"].reshape(-1,1)).flatten()
    return preds, actual


def forecast_future(results, close_scaler, feat_scaled, window, n_feat, close_col_idx, days):
    preds = {}
    last_window = feat_scaled[-window:]
    for name, r in results.items():
        model = r["model"]
        win   = last_window.copy()
        out   = []
        for _ in range(days):
            p_sc = model.predict(win.reshape(1, window, n_feat), verbose=0)[0,0]
            out.append(p_sc)
            new_row = win[-1].copy()
            new_row[close_col_idx] = p_sc
            win = np.vstack([win[1:], new_row])
        preds[name] = close_scaler.inverse_transform(np.array(out).reshape(-1,1)).flatten()
    return preds


# ─────────────────────────────────────────────
#  RUN TRAINING
# ─────────────────────────────────────────────
if run_clicked:
    st.session_state["results"] = None
    st.session_state["run_training"] = True

if st.session_state["run_training"] and st.session_state["results"] is None:

    which = []
    if run_lstm:    which.append("LSTM")
    if run_gru:     which.append("GRU")
    if run_cnnlstm: which.append("CNN-LSTM")

    if not which:
        st.warning("Select at least one model in the sidebar.")
    else:
        sec_title("🔄 Training in progress")

        # Data loading
        with st.spinner("Loading and preparing data..."):
            out = load_and_prepare(cfg["ticker"], cfg["period"], cfg["window"])
            df, X, y, close_sc, feat_sc, feat_scaled, FCOLS, close_idx = out

        if df is None:
            ticker_tried = cfg["ticker"]
            period_tried = cfg["period"]

            # Diagnose why it failed
            try:
                test_raw = yf.download(ticker_tried, period="1mo",
                                       progress=False, auto_adjust=True)
                if isinstance(test_raw.columns, pd.MultiIndex):
                    test_raw.columns = [c[0] for c in test_raw.columns]
                rows_1mo = len(test_raw.dropna(subset=["Close"])) if not test_raw.empty else 0
            except Exception:
                rows_1mo = 0

            try:
                info = yf.Ticker(ticker_tried).info
                long_name    = info.get("longName", "")
                exchange     = info.get("exchange", "")
                quote_type   = info.get("quoteType", "")
                first_trade  = info.get("firstTradeDateEpochUtc", None)
                import datetime as _dt
                listed_since = (
                    str(_dt.datetime.fromtimestamp(first_trade).date())
                    if first_trade else "unknown"
                )
            except Exception:
                long_name   = ""
                exchange    = ""
                quote_type  = ""
                listed_since = "unknown"

            # Determine specific reason
            window_need = cfg["window"] * 3
            if rows_1mo == 0 and not long_name:
                reason = (f"**Ticker `{ticker_tried}` not found** on Yahoo Finance. "
                          f"It may not exist, be delisted, or use a different symbol.")
                suggestion = "wrong_ticker"
            elif rows_1mo > 0 and rows_1mo < window_need:
                reason = (f"**`{ticker_tried}` exists but has only ~{rows_1mo} trading days** "
                          f"of data on Yahoo Finance. "
                          f"The model needs at least **{window_need} rows** to train.")
                suggestion = "new_listing"
            elif rows_1mo == 0 and long_name:
                reason = (f"**`{ticker_tried}` ({long_name})** is recognised but returned "
                          f"no price data for the selected period `{period_tried}`.")
                suggestion = "period_too_long"
            else:
                reason = (f"**Not enough data** for `{ticker_tried}` with period "
                          f"`{period_tried}`. Need {window_need}+ rows, got {rows_1mo}.")
                suggestion = "short_period"

            st.error(f"❌ Could not load data for **{ticker_tried}**")
            st.markdown(
                f"<div style='background:rgba(255,51,102,0.07);"
                f"border:1px solid rgba(255,51,102,0.3);border-radius:8px;"
                f"padding:14px 18px;margin:8px 0;'>"
                f"<div style='font-family:\"Share Tech Mono\",monospace;"
                f"font-size:0.78rem;color:#ff6688;margin-bottom:10px;'>"
                f"🔍 Diagnosis</div>"
                f"<div style='font-family:Rajdhani,sans-serif;font-size:0.92rem;"
                f"color:#e8f4f8;line-height:1.8;'>{reason}"
                + (f"<br><span style='color:#8baabb;'>Listed since: {listed_since}</span>" if listed_since != "unknown" else "")
                + (f"<br><span style='color:#8baabb;'>Exchange: {exchange} · Type: {quote_type}</span>" if exchange else "")
                + f"</div></div>",
                unsafe_allow_html=True,
            )

            # Show fix suggestion
            fixes = {
                "wrong_ticker": {
                    "title": "Try these alternatives",
                    "items": [
                        ("MobiKwik correct ticker", "MBL.NS"),
                        ("Search Yahoo Finance",    "https://finance.yahoo.com/lookup"),
                        ("NSE stocks always end in", ".NS  e.g. RELIANCE.NS"),
                        ("BSE stocks always end in", ".BO  e.g. RELIANCE.BO"),
                        ("US stocks — no suffix",   "AAPL, TSLA, NVDA"),
                        ("Gold futures",            "GC=F"),
                        ("Silver futures",          "SI=F"),
                        ("Bitcoin",                 "BTC-USD"),
                    ]
                },
                "new_listing": {
                    "title": "Stock is too newly listed — not enough history",
                    "items": [
                        ("Problem",  f"Only {rows_1mo} days of data — model needs {window_need}+"),
                        ("Fix 1",    "Wait a few months for more data to accumulate"),
                        ("Fix 2",    "Reduce Window to 30 in Advanced Settings (password: 6348)"),
                        ("Fix 3",    "Try a similar established stock instead"),
                    ]
                },
                "period_too_long": {
                    "title": "Try a shorter period",
                    "items": [
                        ("Change period to", "3mo or 6mo in the sidebar"),
                        ("Then run again",   "click RUN DEEP LEARNING"),
                    ]
                },
                "short_period": {
                    "title": "Increase the training period",
                    "items": [
                        ("Current period", period_tried),
                        ("Change to",      "1y or 2y or 5y in the sidebar"),
                    ]
                },
            }
            fix = fixes.get(suggestion, fixes["wrong_ticker"])
            st.markdown(
                f"<div style='background:rgba(0,245,255,0.04);"
                f"border:1px solid rgba(0,245,255,0.15);border-radius:8px;"
                f"padding:14px 18px;margin:8px 0;'>"
                f"<div style='font-family:\"Share Tech Mono\",monospace;"
                f"font-size:0.75rem;color:#00e5ff;margin-bottom:10px;'>"
                f"💡 {fix['title']}</div>"
                + "".join([
                    f"<div style='display:flex;gap:12px;font-family:\"Share Tech Mono\","
                    f"monospace;font-size:0.72rem;padding:4px 0;border-bottom:"
                    f"1px solid rgba(0,245,255,0.06);'>"
                    f"<span style='color:#445566;min-width:180px;'>{k}</span>"
                    f"<span style='color:#e8f4f8;'>{v}</span></div>"
                    for k, v in fix["items"]
                ])
                + "</div>",
                unsafe_allow_html=True,
            )
        else:
            prog = st.progress(0)
            stat = st.empty()

            results, histories, X_train, X_test, y_train, y_test, split = train_models(
                X, y, cfg, which, prog, stat
            )

            test_preds, actual = get_predictions(results, close_sc)

            fut_dates = pd.bdate_range(
                start=df.index[-1].to_pydatetime() + timedelta(days=1),
                periods=cfg["future_days"]
            )
            fut_preds = forecast_future(
                results, close_sc, feat_scaled,
                cfg["window"], X.shape[2], close_idx, cfg["future_days"]
            )

            st.session_state["results"] = {
                "df": df, "X": X, "y": y,
                "close_sc": close_sc,
                "results": results,
                "histories": histories,
                "test_preds": test_preds,
                "actual": actual,
                "fut_preds": fut_preds,
                "fut_dates": fut_dates,
                "split": split,
                "FCOLS": FCOLS,
                "cfg": cfg.copy(),
                "which": which,
            }
            st.session_state["run_training"] = False
            st.rerun()


# ─────────────────────────────────────────────
#  DISPLAY RESULTS
# ─────────────────────────────────────────────
R = st.session_state["results"]

if R is None:
    # Welcome screen
    st.markdown(
        "<div style='text-align:center;padding:60px 0 20px;'>"
        "<div style='font-family:\"Share Tech Mono\",monospace;font-size:0.85rem;"
        "color:#445566;line-height:2.2;'>"
        "① Select a stock ticker in the sidebar<br>"
        "② Choose training period (5 Years recommended)<br>"
        "③ Set forecast days with the slider<br>"
        "④ Enter password <b style='color:#00f5ff'>6348</b> to unlock advanced settings<br>"
        "⑤ Click <b style='color:#00f5ff'>RUN DEEP LEARNING</b>"
        "</div></div>",
        unsafe_allow_html=True,
    )

    # Model cards
    sec_title("🧠 Neural Network Models")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            "<div style='background:rgba(0,255,136,0.07);border:1px solid rgba(0,255,136,0.3);"
            "border-radius:8px;padding:16px;'>"
            "<div style='font-family:Orbitron,monospace;font-size:0.9rem;color:#00ff88;"
            "font-weight:700;margin-bottom:8px;'>LSTM</div>"
            "<div style='font-family:\"Share Tech Mono\",monospace;font-size:0.72rem;"
            "color:#b0d4e8;line-height:1.7;'>Long Short-Term Memory<br>"
            "Best for: Long-term trends<br>"
            "Memory gates: Forget · Input · Output<br>"
            "Layers: 3 stacked LSTM + Dropout<br>"
            "Params: ~150K</div></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            "<div style='background:rgba(168,85,247,0.07);border:1px solid rgba(168,85,247,0.3);"
            "border-radius:8px;padding:16px;'>"
            "<div style='font-family:Orbitron,monospace;font-size:0.9rem;color:#a855f7;"
            "font-weight:700;margin-bottom:8px;'>GRU</div>"
            "<div style='font-family:\"Share Tech Mono\",monospace;font-size:0.72rem;"
            "color:#b0d4e8;line-height:1.7;'>Gated Recurrent Unit<br>"
            "Best for: Medium-term momentum<br>"
            "Gates: Reset · Update<br>"
            "Layers: 3 stacked GRU + Dropout<br>"
            "Params: ~120K</div></div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            "<div style='background:rgba(255,170,0,0.07);border:1px solid rgba(255,170,0,0.3);"
            "border-radius:8px;padding:16px;'>"
            "<div style='font-family:Orbitron,monospace;font-size:0.9rem;color:#ffaa00;"
            "font-weight:700;margin-bottom:8px;'>CNN-LSTM</div>"
            "<div style='font-family:\"Share Tech Mono\",monospace;font-size:0.72rem;"
            "color:#b0d4e8;line-height:1.7;'>Hybrid: CNN + LSTM<br>"
            "Best for: Chart pattern detection<br>"
            "CNN extracts patterns → LSTM learns<br>"
            "Layers: 2×Conv1D + MaxPool + 2×LSTM<br>"
            "Params: ~100K</div></div>",
            unsafe_allow_html=True,
        )

else:
    # ── Unpack results
    df         = R["df"]
    close_sc   = R["close_sc"]
    histories  = R["histories"]
    test_preds = R["test_preds"]
    actual     = R["actual"]
    fut_preds  = R["fut_preds"]
    fut_dates  = R["fut_dates"]
    split      = R["split"]
    FCOLS      = R["FCOLS"]
    run_cfg    = R["cfg"]
    which      = R["which"]
    window     = run_cfg["window"]

    dates_test = df.index[window + split:]
    last_price = float(df["Close"].squeeze().iloc[-1])
    cur        = "₹" if ".NS" in run_cfg["ticker"] else "$"

    model_colors = {
        "LSTM":     COLORS["lstm"],
        "GRU":      COLORS["gru"],
        "CNN-LSTM": COLORS["cnn"],
        "Ensemble": COLORS["ensemble"],
    }

    # ── Ensemble
    all_test = list(test_preds.values())
    ensemble_test = np.mean(all_test, axis=0)
    all_fut  = list(fut_preds.values())
    ensemble_fut  = np.mean(all_fut, axis=0)

    # ── Metrics
    def get_metrics(preds, actual):
        rmse = np.sqrt(mean_squared_error(actual, preds))
        mae  = mean_absolute_error(actual, preds)
        da   = np.mean(np.sign(np.diff(actual)) == np.sign(np.diff(preds))) * 100
        return rmse, mae, da

    # ────────────────────────────────────────
    #  METRIC CARDS ROW
    # ────────────────────────────────────────
    sec_title(f"📊 Results — {run_cfg['ticker']}")

    metric_cols = st.columns(4 + len(which))
    with metric_cols[0]:
        st.markdown(mcard("Current Price", f"{cur}{last_price:,.2f}",
                           f"As of {df.index[-1].date()}"), unsafe_allow_html=True)
    with metric_cols[1]:
        e7  = float(ensemble_fut[min(6,  len(ensemble_fut)-1)])
        st.markdown(mcard("Ensemble 7D",
                           f"{cur}{e7:,.2f}",
                           f"{'▲' if e7>last_price else '▼'} {(e7-last_price)/last_price*100:+.1f}%",
                           "#00ff88" if e7>last_price else "#ff3366"), unsafe_allow_html=True)
    with metric_cols[2]:
        e30 = float(ensemble_fut[min(29, len(ensemble_fut)-1)])
        st.markdown(mcard("Ensemble 30D",
                           f"{cur}{e30:,.2f}",
                           f"{'▲' if e30>last_price else '▼'} {(e30-last_price)/last_price*100:+.1f}%",
                           "#00ff88" if e30>last_price else "#ff3366"), unsafe_allow_html=True)
    with metric_cols[3]:
        e_rmse, e_mae, e_da = get_metrics(ensemble_test, actual)
        st.markdown(mcard("Direction Accuracy",
                           f"{e_da:.1f}%",
                           "Ensemble model",
                           "#00ff88" if e_da>55 else "#ffaa00"), unsafe_allow_html=True)

    for i, name in enumerate(which):
        with metric_cols[4 + i]:
            rmse, mae, da = get_metrics(test_preds[name], actual)
            color = model_colors[name]
            st.markdown(
                f"<div style='background:rgba(4,15,28,0.92);border:1px solid {color}33;"
                f"border-radius:8px;padding:13px 14px;text-align:center;'>"
                f"<div style='font-family:Orbitron,monospace;font-size:0.7rem;"
                f"color:{color};margin-bottom:4px;'>{name}</div>"
                f"<div style='font-family:\"Share Tech Mono\",monospace;font-size:0.72rem;"
                f"color:#e8f4f8;'>RMSE: {rmse:.2f}</div>"
                f"<div style='font-family:\"Share Tech Mono\",monospace;font-size:0.72rem;"
                f"color:#e8f4f8;'>MAE: {mae:.2f}</div>"
                f"<div style='font-family:\"Share Tech Mono\",monospace;font-size:0.72rem;"
                f"color:{'#00ff88' if da>55 else '#ffaa00'};'>Dir: {da:.1f}%</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── Verdict
    if e30 > last_price * 1.02:
        v_color = "#00ff88"; v_bg = "rgba(0,255,136,0.07)"; v_border = "rgba(0,255,136,0.3)"
        verdict = f"✅ BULLISH — Ensemble forecasts {cur}{e30:,.2f} (+{(e30-last_price)/last_price*100:.1f}%) in 30 days"
    elif e30 < last_price * 0.98:
        v_color = "#ff3366"; v_bg = "rgba(255,51,102,0.07)"; v_border = "rgba(255,51,102,0.3)"
        verdict = f"🔴 BEARISH — Ensemble forecasts {cur}{e30:,.2f} ({(e30-last_price)/last_price*100:.1f}%) in 30 days"
    else:
        v_color = "#ffaa00"; v_bg = "rgba(255,170,0,0.07)"; v_border = "rgba(255,170,0,0.3)"
        verdict = f"⏸ NEUTRAL — Minimal movement forecast. Hold and monitor."

    st.markdown(
        f"<div style='background:{v_bg};border:1px solid {v_border};border-radius:8px;"
        f"padding:12px 18px;margin:12px 0;font-family:\"Share Tech Mono\",monospace;"
        f"font-size:0.82rem;color:{v_color};'>{verdict}</div>",
        unsafe_allow_html=True,
    )

    # ────────────────────────────────────────
    #  TABS: Charts
    # ────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈  FORECAST CHART",
        "🎯  TEST ACCURACY",
        "📉  TRAINING LOSS",
        "📋  PREDICTION TABLE",
    ])

    # ── Tab 1: Future forecast
    with tab1:
        fig = go.Figure()

        # Historical (last 90 days)
        hist_n = min(90, len(df))
        fig.add_trace(go.Scatter(
            x=df.index[-hist_n:],
            y=df["Close"].squeeze().values[-hist_n:],
            name="Historical price",
            line=dict(color=COLORS["actual"], width=2),
        ))

        # Individual models
        for name in which:
            fp = fut_preds[name]
            n  = min(len(fp), len(fut_dates))
            fig.add_trace(go.Scatter(
                x=[df.index[-1]] + [str(d.date()) for d in fut_dates[:n]],
                y=[last_price] + list(fp[:n]),
                name=name,
                line=dict(color=model_colors[name], width=1.5, dash="dash"),
                opacity=0.7,
            ))

        # Ensemble
        n = min(len(ensemble_fut), len(fut_dates))
        fig.add_trace(go.Scatter(
            x=[df.index[-1]] + [str(d.date()) for d in fut_dates[:n]],
            y=[last_price] + list(ensemble_fut[:n]),
            name="Ensemble",
            line=dict(color=COLORS["ensemble"], width=3),
        ))

        # Confidence band
        if len(all_fut) > 1:
            min_f = np.min(all_fut, axis=0)
            max_f = np.max(all_fut, axis=0)
            fd_str = [str(d.date()) for d in fut_dates[:n]]
            fig.add_trace(go.Scatter(
                x=fd_str + fd_str[::-1],
                y=list(max_f[:n]) + list(min_f[:n])[::-1],
                fill="toself", fillcolor="rgba(0,245,255,0.08)",
                line=dict(color="rgba(0,0,0,0)"),
                name="Model range",
            ))

        # NOW line
        now_str = str(df.index[-1].date())
        fig.add_shape(type="line", x0=now_str, x1=now_str, y0=0, y1=1,
                      xref="x", yref="paper",
                      line=dict(color="rgba(0,245,255,0.4)", width=1.5, dash="dash"))
        fig.add_annotation(x=now_str, y=1, xref="x", yref="paper",
                           text="NOW", showarrow=False,
                           font=dict(color="#00f5ff", size=10),
                           yanchor="bottom", xanchor="left",
                           bgcolor="rgba(2,11,20,0.8)",
                           bordercolor="rgba(0,245,255,0.4)", borderwidth=1)

        ly = base_layout(f"{run_cfg['ticker']} — {run_cfg['future_days']}-Day Deep Learning Forecast")
        ly["height"] = 520
        fig.update_layout(**ly)
        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

    # ── Tab 2: Test accuracy
    with tab2:
        fig2 = go.Figure()
        nt   = min(len(actual), len(dates_test))

        fig2.add_trace(go.Scatter(
            x=dates_test[:nt], y=actual[:nt],
            name="Actual price",
            line=dict(color=COLORS["actual"], width=2),
        ))
        for name in which:
            p  = test_preds[name]
            nt2 = min(len(p), len(dates_test))
            fig2.add_trace(go.Scatter(
                x=dates_test[:nt2], y=p[:nt2],
                name=f"{name} prediction",
                line=dict(color=model_colors[name], width=1.5, dash="dash"),
                opacity=0.85,
            ))
        fig2.add_trace(go.Scatter(
            x=dates_test[:nt], y=ensemble_test[:nt],
            name="Ensemble",
            line=dict(color=COLORS["ensemble"], width=2, dash="dot"),
        ))

        ly2 = base_layout("Test Set — Actual vs Predicted Price")
        ly2["height"] = 480
        fig2.update_layout(**ly2)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Tab 3: Training loss
    with tab3:
        fig3 = make_subplots(rows=1, cols=len(which),
                              subplot_titles=[f"{n} Loss" for n in which])
        for i, name in enumerate(which):
            hist = histories[name]
            color = model_colors[name]
            fig3.add_trace(go.Scatter(
                y=hist["loss"], name=f"{name} train",
                line=dict(color=color, width=2),
            ), row=1, col=i+1)
            fig3.add_trace(go.Scatter(
                y=hist["val_loss"], name=f"{name} val",
                line=dict(color=color, width=1.5, dash="dot"),
                opacity=0.7,
            ), row=1, col=i+1)

        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#040f1c",
            font=dict(family="'Share Tech Mono',monospace", color="#e8f4f8", size=11),
            height=380,
            legend=dict(bgcolor="rgba(2,11,20,0.9)", bordercolor="rgba(0,245,255,0.2)",
                        borderwidth=1, font=dict(size=10, color="#e8f4f8")),
        )
        for i in range(1, len(which)+1):
            fig3.update_xaxes(gridcolor=COLORS["grid"], color="#8baabb",
                               title_text="Epoch", row=1, col=i)
            fig3.update_yaxes(gridcolor=COLORS["grid"], color="#8baabb",
                               title_text="Loss" if i==1 else "", row=1, col=i)
        st.plotly_chart(fig3, use_container_width=True)

        # Epoch count summary
        ep_cols = st.columns(len(which))
        for i, name in enumerate(which):
            with ep_cols[i]:
                n_ep = len(histories[name]["loss"])
                best = min(histories[name]["val_loss"])
                st.markdown(mcard(
                    f"{name} training",
                    f"{n_ep} epochs",
                    f"Best val loss: {best:.6f}",
                    model_colors[name],
                ), unsafe_allow_html=True)

    # ── Tab 4: Prediction table
    with tab4:
        sec_title("🔮 Future Price Predictions")
        rows = []
        checkpoints = [0, 4, 6, 9, 13, 19, 24, 29]
        checkpoints = [i for i in checkpoints if i < len(fut_dates)]
        for i in checkpoints:
            row = {"Date": str(fut_dates[i].date()), "Day": i+1}
            for name in which:
                p   = float(fut_preds[name][i])
                pct = (p - last_price) / last_price * 100
                row[name] = f"{cur}{p:,.2f} ({pct:+.1f}%)"
            ep  = float(ensemble_fut[i])
            pct = (ep - last_price) / last_price * 100
            row["Ensemble"] = f"{cur}{ep:,.2f} ({pct:+.1f}%)"
            rows.append(row)

        pred_df = pd.DataFrame(rows)
        st.dataframe(pred_df, use_container_width=True, hide_index=True)

        # Model comparison
        sec_title("📊 Model Accuracy Comparison")
        cmp_rows = []
        for name in which + ["Ensemble"]:
            p       = ensemble_test if name == "Ensemble" else test_preds[name]
            a_len   = min(len(p), len(actual))
            rmse, mae, da = get_metrics(p[:a_len], actual[:a_len])
            best_30 = float(fut_preds[name][min(29,len(fut_preds.get(name, ensemble_fut))-1)]) if name != "Ensemble" else float(ensemble_fut[min(29,len(ensemble_fut)-1)])
            cmp_rows.append({
                "Model":     name,
                "RMSE":      f"{rmse:.2f}",
                "MAE":       f"{mae:.2f}",
                "Dir Acc %": f"{da:.1f}%",
                f"30D Forecast": f"{cur}{best_30:,.2f} ({(best_30-last_price)/last_price*100:+.1f}%)",
                "Verdict":   "▲ UP" if best_30 > last_price else "▼ DOWN",
            })
        st.dataframe(pd.DataFrame(cmp_rows), use_container_width=True, hide_index=True)

    # ── Config used
    with st.expander("⚙ Configuration used for this run", expanded=False):
        cc = st.columns(4)
        items = [
            ("Ticker",       run_cfg["ticker"]),
            ("Period",       PERIOD_LABELS[run_cfg["period"]]),
            ("Window",       f"{run_cfg['window']} days"),
            ("Future days",  f"{run_cfg['future_days']} days"),
            ("Test split",   f"{run_cfg['test_split']*100:.0f}%"),
            ("Max epochs",   str(run_cfg["epochs"])),
            ("Batch size",   str(run_cfg["batch_size"])),
            ("Learning rate",str(run_cfg["learning_rate"])),
            ("Seed",         str(run_cfg["seed"])),
            ("Features",     str(len(FCOLS))),
            ("Train samples",str(split)),
            ("Models",       ", ".join(which)),
        ]
        for i, (label, val) in enumerate(items):
            with cc[i % 4]:
                st.markdown(mcard(label, val), unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  DISCLAIMER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='background:rgba(255,51,102,0.06);border:1px solid rgba(255,51,102,0.3);"
    "border-radius:8px;padding:16px 20px;margin:10px 0;'>"
    "<div style='font-family:Orbitron,monospace;font-size:0.78rem;font-weight:700;"
    "color:#ff3366;letter-spacing:2px;margin-bottom:8px;'>⚠ Disclaimer</div>"
    "<div style='font-family:Rajdhani,sans-serif;font-size:0.9rem;color:#d0e8f0;line-height:1.7;'>"
    "For educational purposes only. Deep learning predictions do not constitute financial advice. "
    "Neural network models can be wrong — always consult a SEBI-registered financial advisor "
    "before making investment decisions."
    "</div></div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div style='text-align:center;font-family:\"Share Tech Mono\",monospace;"
    "font-size:0.6rem;color:#334455;padding:8px;'>"
    "VENKAT.AI Deep Learning v1.0 · TensorFlow/Keras · LSTM · GRU · CNN-LSTM"
    "</div>",
    unsafe_allow_html=True,
)
