import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Katsanos RS ETF Strategy", layout="wide", page_icon="📈")

# --- ESTILOS CSS PERSONALIZADOS ---
st.markdown("""
    <style>
    .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 5px; }
    .stDataFrame { font-size: 14px; }
    .long-signal { color: green; font-weight: bold; }
    .no-signal { color: red; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# --- FUNCIONES DE CÁLCULO DE INDICADORES ---
def calculate_rsi(series, period=5):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_bars_since(condition_series):
    """Replica la función BarsSince de AmiBroker"""
    bars_since = np.full(len(condition_series), np.nan)
    last_true = -1
    for i in range(len(condition_series)):
        if condition_series.iloc[i]:
            last_true = i
        if last_true != -1:
            bars_since[i] = i - last_true
    return bars_since

@st.cache_data(show_spinner="Descargando y calculando datos...")
def load_and_calculate_data(ticker, period="2y"):
    # Descarga de datos usando yf.Ticker para evitar problemas de MultiIndex
    etf_data = yf.Ticker(ticker).history(period=period)
    spy_data = yf.Ticker('SPY').history(period=period)
    
    if etf_data.empty or spy_data.empty:
        raise ValueError(f"No se pudieron descargar datos para {ticker} o SPY.")

    # Crear DataFrame principal y copiar TODAS las columnas OHLCV
    df = pd.DataFrame(index=etf_data.index)
    df['Open'] = etf_data['Open']
    df['High'] = etf_data['High']
    df['Low'] = etf_data['Low']
    df['Close'] = etf_data['Close']
    df['Volume'] = etf_data['Volume']
    df['SPY_Close'] = spy_data['Close']
    
    # Alinear fechas y eliminar NaNs iniciales
    df.dropna(inplace=True)

    # 1. Relative Strength (RS) y su Media Móvil (MARS)
    rs_raw = df['Close'] / df['SPY_Close']
    df['RS'] = rs_raw.ewm(span=2, adjust=False).mean() * 100
    df['MARS'] = df['RS'].rolling(60).mean()
    
    # Cruce y BarsSince
    cross = (df['RS'] > df['MARS']) & (df['RS'].shift(1) <= df['MARS'].shift(1))
    df['BSCRRS'] = calculate_bars_since(cross)

    # 2. Correlaciones
    df['COR1'] = df['Close'].rolling(100).corr(df['SPY_Close'])
    roc_etf = df['Close'].pct_change(5)
    roc_spy = df['SPY_Close'].pct_change(5)
    df['COROC'] = roc_etf.rolling(100).corr(roc_spy)

    # 3. RSI 5 periodos
    df['RSI_5'] = calculate_rsi(df['Close'], 5)

    # 4. EMA 50 (Para el chart principal)
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()

    # 5. ROC SPY 2 días (Para el filtro de mercado)
    df['ROC_SPY_2'] = df['SPY_Close'].pct_change(2) * 100

    # --- CONDICIONES FINALES DE INSCRIPCION (LONG) ---
    cond1 = df['RS'] > 1.02 * df['MARS']
    cond2 = (df['BSCRRS'] < 20) & (df['BSCRRS'] > 0) 
    cond3 = (df['COROC'] > 0.3) & (df['COR1'] > 0.3)
    cond4 = (df['ROC_SPY_2'] > -6) | (df['COROC'] < -0.3)
    cond5 = (df['RSI_5'].rolling(5).min() < 30) & (df['RSI_5'] > df['RSI_5'].shift(1))
    
    # Filtro de Liquidez
    lqd = (df['Close'] > 2) & (df['Volume'].rolling(3).mean() > 50000)

    df['LONG_SIGNAL'] = cond1 & cond2 & cond3 & cond4 & cond5 & lqd

    return df

# --- INTERFAZ DE STREAMLIT ---
st.title("📊 Estrategia de Fuerza Relativa (RS) - Markos Katsanos")
st.markdown("Análisis de Rotación Sectorial basado en el paper *Estrategia de Fuerza Relativa* (X-Trader / Hispatrading).")

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("⚙️ Configuración")
    
    etfs = ['XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY']
    selected_etf = st.selectbox("Selecciona el ETF Sectorial:", etfs, index=5) # XLK por defecto
    
    st.markdown("---")
    st.subheader("Gráfico de Indicadores")
    num_indicators = st.selectbox(
        "Número de sub-gráficos a mostrar:", 
        [1, 2, 3], 
        help="1: RS | 2: RS + RSI | 3: RS + Correlación + RSI"
    )
    
    st.markdown("---")
    st.info("💡 **Nota:** La EMA de 50 periodos se muestra siempre en el gráfico principal de precios.")

# --- CARGA DE DATOS ---
try:
    df = load_and_calculate_data(selected_etf)
except Exception as e:
    st.error(f"Error al descargar o calcular datos: {e}")
    st.stop()

# --- GRÁFICO CENTRAL (PLOTLY) ---
st.subheader(f"Análisis Técnico: {selected_etf} vs SPY")

# Determinar qué indicadores mostrar
indicators_to_plot = ['RS']
if num_indicators >= 2:
    indicators_to_plot.append('RSI')
if num_indicators >= 3:
    indicators_to_plot.append('COR')

num_rows = 1 + len(indicators_to_plot)
row_heights = [0.5] + [0.5 / len(indicators_to_plot)] * len(indicators_to_plot)

fig = make_subplots(
    rows=num_rows, cols=1, 
    shared_xaxes=True, 
    vertical_spacing=0.03,
    row_heights=row_heights,
    subplot_titles=[f"{selected_etf} Precio & EMA 50"] + indicators_to_plot
)

# Fila 1: Velas Japonesas + EMA 50
fig.add_trace(go.Candlestick(
    x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
    name='Price', increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=df.index, y=df['EMA_50'], mode='lines', name='EMA 50', 
    line=dict(color='orange', width=1.5)
), row=1, col=1)

# Sub-gráficos de indicadores
current_row = 2
for ind in indicators_to_plot:
    if ind == 'RS':
        fig.add_trace(go.Scatter(x=df.index, y=df['RS'], name='RS', line=dict(color='blue')), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MARS'], name='MARS (60)', line=dict(color='red', dash='dash')), row=current_row, col=1)
    elif ind == 'RSI':
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI_5'], name='RSI (5)', line=dict(color='purple')), row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="gray", row=current_row, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="gray", row=current_row, col=1)
    elif ind == 'COR':
        fig.add_trace(go.Scatter(x=df.index, y=df['COR1'], name='COR1 (Price)', line=dict(color='teal')), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['COROC'], name='COROC (ROC)', line=dict(color='orange')), row=current_row, col=1)
        fig.add_hline(y=0.3, line_dash="dash", line_color="green", row=current_row, col=1)
    current_row += 1

fig.update_layout(
    height=800, 
    xaxis_rangeslider_visible=False, 
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
fig.update_xaxes(type='category', tickangle=-45, nticks=20)

st.plotly_chart(fig, use_container_width=True)

# --- TABLA INFERIOR DE VALORES Y SEÑAL ---
st.subheader("📋 Tablero de Indicadores y Señal de Entrada")

# Obtener la última fila con datos válidos
last_row = df.iloc[-1]

# Crear dataframe para la tabla
metrics_data = {
    "Indicador": [
        "Precio de Cierre", "RS (Fuerza Relativa)", "MARS (Media RS 60)", 
        "BSCRRS (Barras desde cruce)", "COR1 (Correlación 100d)", 
        "COROC (Correlación ROC)", "RSI (5 periodos)", "ROC SPY (2 días)",
        "Filtro Liquidez", "SEÑAL LONG (Katsanos)"
    ],
    "Valor Actual": [
        f"${last_row['Close']:.2f}", 
        f"{last_row['RS']:.2f}", 
        f"{last_row['MARS']:.2f}",
        f"{int(last_row['BSCRRS']) if not np.isnan(last_row['BSCRRS']) else 'N/A'}",
        f"{last_row['COR1']:.3f}" if not np.isnan(last_row['COR1']) else 'N/A',
        f"{last_row['COROC']:.3f}" if not np.isnan(last_row['COROC']) else 'N/A',
        f"{last_row['RSI_5']:.2f}",
        f"{last_row['ROC_SPY_2']:.2f}%",
        "Cumplido ✅" if (last_row['Close'] > 2 and df['Volume'].rolling(3).mean().iloc[-1] > 50000) else "No cumplido ❌",
        "🟢 COMPRA (LONG)" if last_row['LONG_SIGNAL'] else "🔴 SIN SEÑAL"
    ],
    "Condición Estratégica": [
        "-", 
        "Debe ser > 1.02 * MARS", 
        "Media de 60 días del RS",
        "Debe ser < 20 y > 0",
        "Debe ser > 0.30",
        "Debe ser > 0.30",
        "Debe haber sido < 30 en los últimos 5d y subir hoy",
        "SPY > -6% (o COROC < -0.3)",
        "Precio > $2 y Vol > 50k",
        "Confluencia de todas las condiciones"
    ]
}

table_df = pd.DataFrame(metrics_data)

# Función para resaltar la fila de la señal
def highlight_signal(val):
    if "COMPRA" in val:
        return 'background-color: #d4edda; color: #155724; font-weight: bold'
    elif "SIN SEÑAL" in val:
        return 'background-color: #f8d7da; color: #721c24; font-weight: bold'
    return ''

st.dataframe(
    table_df.style.applymap(highlight_signal, subset=['Valor Actual']),
    use_container_width=True,
    hide_index=True,
    height=450
)

st.markdown("---")
st.caption("Desarrollado con Streamlit y Python. Lógica basada en el artículo de Markos Katsanos (X-Trader / Hispatrading). Los datos se obtienen en tiempo real vía Yahoo Finance.")