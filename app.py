import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import openai

# 1. CONFIGURACIÓN DE PÁGINA
st.set_page_config(page_title="CatSense AI Dashboard", layout="wide", page_icon="🐱")

# Estilo CSS para un look profesional
st.markdown("""
    <style>
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #f0f2f6; }
    .main { background-color: #f8f9fa; }
    </style>
    """, unsafe_allow_html=True)

# 2. CARGA DE DATOS (Con Cache)
@st.cache_data
def load_and_clean():
    df = pd.read_csv('CatSenseAI.csv')
    df['date'] = pd.to_datetime(df['date'])
    df['sentiment_score'] = pd.to_numeric(df['sentiment_score'], errors='coerce')
    df['urgency_level'] = pd.to_numeric(df['urgency_level'], errors='coerce')
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df['repurchase_intent'] = df['repurchase_intent'].astype(bool)
    
    # Clasificación de sentimiento (esto nos ahorra errores después)
    df['sentimiento_cat'] = df['sentiment_score'].apply(
        lambda x: 'Positivo' if x > 0.2 else ('Negativo' if x < -0.2 else 'Neutro')
    )
    return df

try:
    df_raw = load_and_clean()
except Exception as e:
    st.error(f"Error al cargar CatSenseAI.csv: {e}")
    st.stop()

# 3. BARRA LATERAL (Filtros)
st.sidebar.image("https://em-content.zobj.net/source/apple/391/cat_1f408.png", width=80)
st.sidebar.header("Panel de Filtros")

# Variables de tiempo
min_date = df_raw['date'].min().to_pydatetime()
max_date = df_raw['date'].max().to_pydatetime()

# --- CAMBIO A BARRA DESLIZABLE (SLIDER) ---
rango_fecha = st.sidebar.slider(
    "Selecciona Periodo:",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    format="DD/MM/YYYY"
)
# --- ESTA ES LA LÍNEA QUE FALTA ---
all_brands = sorted(df_raw['brand'].unique())

# --- FILTRO DE MARCAS CON CHECKBOXES ---
st.sidebar.write("### Selecciona Marcas:")

# Creamos una lista vacía para guardar las marcas que elijas
selected_brands = []

# Recorremos todas las marcas y creamos un checkbox para cada una
for brand in all_brands:
    # value=True hace que todas empiecen seleccionadas por defecto
    if st.sidebar.checkbox(brand, value=True, key=brand):
        selected_brands.append(brand)

# Validación de seguridad: si no seleccionas ninguna, usamos todas para no romper las gráficas
if not selected_brands:
    selected_brands = all_brands


st.sidebar.markdown("---")
st.sidebar.caption("Sesión iniciada como: **Analista CatSense**")

# Aplicar Filtros (Actualizado para el Slider)
start_date, end_date = rango_fecha

df_filtered = df_raw[
    (df_raw['brand'].isin(selected_brands)) & 
    (df_raw['date'] >= pd.Timestamp(start_date)) & 
    (df_raw['date'] <= pd.Timestamp(end_date))
]

# --- FUNCIONES AUXILIARES CORREGIDAS ---

def calcular_nss_val(serie_etiquetas):
    """Calcula el NSS basado en las etiquetas ya creadas"""
    total = len(serie_etiquetas)
    if total == 0: return 0
    pos = (serie_etiquetas == 'Positivo').sum()
    neg = (serie_etiquetas == 'Negativo').sum()
    return ((pos - neg) / total) * 100

def get_nth_aspect(brand_name, df_context, sentiment_type='Positivo', rank=0):
    mask = (df_context['brand'] == brand_name)
    mask &= (df_context['sentiment_score'] > 0) if sentiment_type == 'Positivo' else (df_context['sentiment_score'] < 0)
    aspectos = df_context[mask]['specific_aspect'].replace(['Sin especificar', '', 'General / Otros', 'N/A'], pd.NA).dropna()
    conteo = aspectos.value_counts()
    return conteo.index[rank] if len(conteo) > rank else "N/A"

def get_negative_vibe(brand_name, df_context):
    mask = (df_context['brand'] == brand_name) & (df_context['sentiment_score'] < 0)
    emociones = df_context[mask]['detected_emotion'].dropna()
    return emociones.mode()[0] if not emociones.mode().empty else "N/A"

# 4. CUERPO DEL DASHBOARD
st.title("🐱 CatSense AI: Dashboard de Inteligencia")
st.toast("🐾 ¡Bienvenido de nuevo! Motor de IA CatSense inicializado...")

# Creamos un bloque que parezca un mensaje de la IA
with st.container():
    col_ai_icon, col_ai_text = st.columns([0.15, 0.85])
    with col_ai_icon:
        # Un icono que represente a tu asistente IA
        st.image("https://cdn-icons-png.flaticon.com/512/4712/4712109.png", width=80) 
    
    with col_ai_text:
        st.markdown(f"""
        ### **Hola, Analista. Soy tu asistencia de Inteligencia Artificial.**
        He procesado los patrones de comportamiento y sentimiento de tus datos. Actualmente tengo bajo mi radar **{len(df_filtered):,} registros** listos para ser analizados. Mi motor está configurado para detectar crisis operativas y oportunidades de lealtad en el periodo seleccionado.
        
        *Usa los controles laterales para ajustar mi visión y actualizar los modelos en tiempo real.*
        """)

st.info(f"🛰️ **Estado del Sistema:** Análisis activo hasta el {end_date.strftime('%d de %b, %Y')}")
st.markdown("---")

# 5. PESTAÑAS (Tabs)
tab1, tab2, tab3, tab4 = st.tabs(["📊 Diagnóstico", "🎯 Matriz", "📈 Churn", "🤖 CatSense AI Agent"])

with tab1:
    st.subheader("Diagnóstico de Sentimiento y Crisis Operativa")
    
    # 1. Generar Tabla Maestra
    resumen = df_filtered.groupby('brand').agg(
        Total_Reseñas=('review', 'count'),
        Rating_Promedio=('rating', 'mean'),
        Sentiment_Score_Avg=('sentiment_score', 'mean'),
        NSS=('sentimiento_cat', calcular_nss_val), # <--- ¡Perfecto, ya está aquí!
        Loyalty_Index=('repurchase_intent', lambda x: (x.sum() / x.count()) * 100),
        Foco_Operativo=('root_cause', lambda x: x.mode()[0] if not x.mode().empty else "N/A")
    ).reset_index()

    resumen['Emocion_Negativa'] = resumen['brand'].apply(lambda x: get_negative_vibe(x, df_filtered))
    resumen['Top_Fortaleza'] = resumen['brand'].apply(lambda x: get_nth_aspect(x, df_filtered, 'Positivo', 0))
    resumen['Top_Debilidad'] = resumen['brand'].apply(lambda x: get_nth_aspect(x, df_filtered, 'Negativo', 0))
    
    criticos = df_filtered[df_filtered['urgency_level'] >= 4].groupby('brand').size()
    resumen['%_Riesgo_Critico'] = resumen['brand'].map((criticos / df_filtered.groupby('brand').size() * 100).fillna(0))

    # 2. MOSTRAR TABLA (Corregida la indentación)
    st.dataframe(
        resumen.rename(columns={'brand':'Marca', 'Sentiment_Score_Avg':'Índice Sent.'}).style
        .background_gradient(subset=['NSS'], cmap='RdYlGn', vmin=0, vmax=100)
        .background_gradient(subset=['Índice Sent.', 'Rating_Promedio'], cmap='RdYlGn')
        .background_gradient(subset=['Loyalty_Index'], cmap='Greens')
        .background_gradient(subset=['%_Riesgo_Critico'], cmap='Reds') 
        .format({
            'NSS': '{:.1f}%', 
            'Loyalty_Index': '{:.1f}%', 
            '%_Riesgo_Critico': '{:.1f}%', 
            'Índice Sent.': '{:.2f}', 
            'Rating_Promedio': '{:.2f}'
        }),
        use_container_width=True
    )

    st.markdown("---")
    # Gráfico NSS
    nss_data = df_filtered.groupby('brand')['sentimiento_cat'].apply(calcular_nss_val).reset_index(name='NSS').sort_values('NSS')
    fig_nss = px.bar(nss_data, x='NSS', y='brand', orientation='h', color='NSS',
                     color_continuous_scale="Aggrnyl_r", title="Net Sentiment Score (NSS) por Marca", text_auto='.1f')
    st.plotly_chart(fig_nss, use_container_width=True)

with tab2:
    st.subheader("Matriz Estratégica: Posicionamiento de Mercado")
    
    # 1. GRÁFICO DE BURBUJAS (MATRIZ)
    fig_bubble = px.scatter(
        resumen, 
        x='Sentiment_Score_Avg', 
        y='Loyalty_Index', 
        size='Total_Reseñas', 
        color='brand', 
        text='brand',
        labels={'Sentiment_Score_Avg': 'Índice de Sentimiento (IA)', 'Loyalty_Index': 'Fidelidad / Recompra (%)'},
        height=500,
        template='plotly_white'
    )

    x_min, x_max = resumen['Sentiment_Score_Avg'].min(), resumen['Sentiment_Score_Avg'].max()
    y_min, y_max = resumen['Loyalty_Index'].min(), resumen['Loyalty_Index'].max()
    avg_sent, avg_loyalty = resumen['Sentiment_Score_Avg'].mean(), resumen['Loyalty_Index'].mean()

    fig_bubble.add_vline(x=avg_sent, line_dash="dash", line_color="gray", opacity=0.5)
    fig_bubble.add_hline(y=avg_loyalty, line_dash="dash", line_color="gray", opacity=0.5)

    y_top_pos = avg_loyalty + (y_max - avg_loyalty) * 0.8
    y_bottom_pos = (y_min + avg_loyalty) / 2

    fig_bubble.add_annotation(x=(avg_sent + x_max)/2, y=y_top_pos, text="<b>LÍDERES</b>", showarrow=False, font=dict(color="green", size=14))
    fig_bubble.add_annotation(x=(x_min + avg_sent)/2, y=y_top_pos, text="<b>REHENES</b>", showarrow=False, font=dict(color="orange", size=14))
    fig_bubble.add_annotation(x=(x_min + avg_sent)/2, y=y_bottom_pos, text="<b>ZONA DE CRISIS</b>", showarrow=False, font=dict(color="red", size=14))
    fig_bubble.add_annotation(x=(avg_sent + x_max)/2, y=y_bottom_pos, text="<b>VULNERABLES</b>", showarrow=False, font=dict(color="blue", size=14))

    fig_bubble.update_traces(textposition='top center')
    st.plotly_chart(fig_bubble, use_container_width=True)

    st.markdown("---")
    st.subheader("🔍 Análisis Profundo: ¿De qué hablan tus clientes?")

    # --- 2. PREPARACIÓN DE DATOS PARA TREEMAPS ---
    df_filtered['specific_aspect'] = df_filtered['specific_aspect'].fillna('General / Otros').replace("", "General / Otros")
    
    df_alto = df_filtered[df_filtered['urgency_level'] >= 3].copy()
    df_bajo = df_filtered[df_filtered['urgency_level'] < 3].copy()
    
    df_alto['cantidad'] = 1
    df_bajo['cantidad'] = 1

    # --- TREEMAP A: EL MURO DE LAS LAMENTACIONES ---
    st.markdown("#### 🚩 El Muro de las Lamentaciones")
    st.caption("Urgencia Alta (≥ 3): Problemas críticos que requieren atención inmediata.")
    if not df_alto.empty:
        fig_alto = px.treemap(
            df_alto,
            path=[px.Constant("Críticos"), 'brand', 'specific_aspect'],
            values='cantidad',
            color='sentiment_score',
            color_continuous_scale='Reds_r',
            height=600,
            hover_data={'sentiment_score': ':.2f', 'executive_summary': True},
            # --- CAMBIO AQUÍ: Nombre de la barra de color ---
            labels={'sentiment_score': 'Sentiment Score'}
        )
        fig_alto.update_traces(
            textinfo="label+value",
            textfont=dict(size=14),
            insidetextfont=dict(size=14),
            hovertemplate='<b>%{label}</b><br>Menciones: %{value}<br>Sentimiento: %{color:.2f}'
        )
        fig_alto.update_layout(
            uniformtext=dict(minsize=14), 
            margin=dict(t=30, l=10, r=10, b=10)
        )
        st.plotly_chart(fig_alto, use_container_width=True)
    else:
        st.info("✅ No hay quejas críticas registradas en este periodo.")

    st.markdown("<br>", unsafe_allow_html=True)

    # --- TREEMAP B: EL MURO DE AGRADECIMIENTOS ---
    st.markdown("#### ✨ El Muro de Agradecimientos")
    st.caption("Urgencia Baja (< 3): Fortalezas de marca y menciones positivas.")
    if not df_bajo.empty:
        fig_bajo = px.treemap(
            df_bajo,
            path=[px.Constant("Satisfechos"), 'brand', 'specific_aspect'],
            values='cantidad',
            color='sentiment_score',
            color_continuous_scale='Greens',
            color_continuous_midpoint=0.5,
            height=600,
            hover_data={'sentiment_score': ':.2f', 'executive_summary': True},
            # --- CAMBIO AQUÍ: Nombre de la barra de color ---
            labels={'sentiment_score': 'Sentiment Score'}
        )
        fig_bajo.update_traces(
            textinfo="label+value",
            textfont=dict(size=14),
            insidetextfont=dict(size=14),
            hovertemplate='<b>%{label}</b><br>Menciones: %{value}<br>Sentimiento: %{color:.2f}'
        )
        fig_bajo.update_layout(
            uniformtext=dict(minsize=14),
            margin=dict(t=30, l=10, r=10, b=10)
        )
        st.plotly_chart(fig_bajo, use_container_width=True)
    else:
        st.info("💡 No hay registros de satisfacción notable en este periodo.")
        
with tab3:
    st.subheader("Semaforización de Churn (Riesgo de Abandono)")
    
    # 1. Preparar datos de Churn
    churn_data = df_filtered.groupby('brand')['repurchase_intent'].apply(
        lambda x: (len(x[x == False]) / len(x)) * 100 if len(x) > 0 else 0
    ).reset_index(name='Churn')
    
    # 2. Medidores de Churn (Gauges)
    cols = st.columns(4) 
    for i, row in enumerate(churn_data.itertuples()):
        with cols[i % 4]:
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=row.Churn,
                title={'text': f"<b>{row.brand}</b>", 'font': {'size': 24}}, 
                number={'suffix': "%", 'font': {'size': 20}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "black"},
                     'steps':[{'range':[0, 20], 'color':"green"}, {'range':[20, 50], 'color':"orange"}, {'range':[50, 100], 'color':"red"}
                    ],
                }
            ))
            fig_g.update_layout(height=180, margin=dict(l=20, r=20, t=60, b=10))
            st.plotly_chart(fig_g, use_container_width=True)

    # --- AQUÍ ESTABA EL ERROR DE INDENTACIÓN ---
    st.markdown("---")
    st.subheader("Balance Emocional: Satisfacción vs Frustración")
    
    # Clasificación de emociones
    df_filtered['emo_c'] = df_filtered['detected_emotion'].str.lower().apply(
        lambda e: 'Satisfacción' if 'satisfac' in str(e) else ('Frustración' if any(x in str(e) for x in ['frustra','eno','desconf']) else None)
    )
    
    df_e = df_filtered.dropna(subset=['emo_c'])
    
    if not df_e.empty:
        e_plot = pd.crosstab(df_e['brand'], df_e['emo_c'], normalize='index') * 100
        
        fig_e = px.bar(
            e_plot.reset_index().melt(id_vars='brand'), 
            x='value', 
            y='brand', 
            color='emo_c', 
            orientation='h',
            color_discrete_map={'Satisfacción': '#27AE60', 'Frustración': '#E74C3C'},
            text_auto='.1f', 
            template='plotly_white',
            labels={
                'emo_c': 'Tipo de sentimiento', # NOMBRE CAMBIADO AQUÍ
                'value': 'Porcentaje (%)', 
                'brand': 'Marca'
            }
        )
        fig_e.update_layout(legend_title_text='Tipo de sentimiento')
        st.plotly_chart(fig_e, use_container_width=True)
    else:
        st.info("No hay datos emocionales suficientes en este periodo.")


  
with tab4:
    st.subheader("🤖 Consultoría Estratégica CatSense")
    st.info("Este agente tiene acceso a las métricas del periodo seleccionado y a las quejas de mayor urgencia.")

    # 1. CONFIGURACIÓN DE LA API (Para local)
    try:
        client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    except:
        st.error("🔑 No se encontró la API Key de OpenAI. Configúrala en .streamlit/secrets.toml o en el código.")
        st.stop()

    # 2. PREPARACIÓN DEL CONTEXTO DINÁMICO
    # Resumen de métricas de la tabla maestra
    contexto_metrizado = resumen.to_string()
    
    # Muestreo de las 12 reseñas más críticas (Urgencia 4-5) para que la IA las cite
    muestreo_critico = df_filtered[df_filtered['urgency_level'] >= 4][['brand', 'specific_aspect', 'review']].head(12).to_string()

    # 3. EL SYSTEM PROMPT "BLINDADO"
    system_instruction = f"""
Eres el Consultor Senior de Inteligencia de Mercado de CatSenseAI. Tu especialidad es la industria de Pet Care en Amazon MX.
Tu objetivo es auditar el desempeño de las marcas basándote ESTRICTAMENTE en los siguientes benchmarks del modelo CatSenseAI:

    PERIODOS Y DATOS:
    - Análisis del {start_date.strftime('%d/%m/%Y')} al {end_date.strftime('%d/%m/%Y')}.
    - Métricas actuales: {contexto_metrizado}
    - Ejemplos de crisis reales (Reseñas Urgencia 4-5): {muestreo_critico}

    BENCHMARKS DE ÉXITO (MÉTRICAS CATSENSE):
1. NET SENTIMENT SCORE (NSS):
   - Bueno (Hegemonía): > 70%.
   - Regular (Sustituible): 30% - 60%.
   - Malo (Deterioro): < 30%.

2. CHURN RISK INDEX (Abandono):
   - Zona Verde (Fidelidad de Hierro): < 20%.
   - Zona Amarilla (Lealtad Frágil): 20% - 35%.
   - Zona Roja (Riesgo Inminente): 35% - 45%.
   - Nivel Crítico (Peligro de Negocio): > 50%.

3. % DE RIESGO CRÍTICO (Urgencia 4-5):
   - Bueno: 5.0% - 6.3%.
   - Alerta: 10% - 15%.
   - Crisis Total: > 30%.

4. BALANCE EMOCIONAL (Satisfacción/Frustración):
   - Liderazgo (Zona de Confort): 83%/17% (Alto "crédito emocional").
   - Alarma (Paridad Emocional): 59%/41% (Marketing costoso, alta decepción).

5. MATRIZ DE POSICIONAMIENTO:
   - LÍDERES 360°: Producto y Logística ambos positivos/altos.
   - REHENES: Producto positivo, pero Logística negativa. Cliente compra por necesidad, no por gusto.
   - ZONA DE CRISIS: Ambos puntajes mínimos o negativos.

REGLAS DE ORO PARA TUS RESPUESTAS:
    - Si el usuario pide un 'Resumen Ejecutivo', usa estrictamente esta estructura:
      0. Responde de forma exhaustiva, técnica y basada en evidencia. 
         Siempre que menciones un problema, busca en los ejemplos de reseñas un caso real que lo respalde y cítalo textualmente entre comillas. 
         Cruza al menos dos métricas en cada conclusión (ej. cómo el NSS impacta directamente en el Churn Index).
      1. Visión General (Metodología CatSense).
      2. KPIs de Desempeño (Usa los nombres NSS, Churn, % Riesgo).
      3. Diagnóstico Competitivo (Identifica Líderes vs Crisis).
      4. Análisis Causa Raíz (Producto vs Logística).
      5. Conclusión Estratégica (Acciones sobre margen y reputación).

    - NO menciones marcas que no estén en la lista enviada.
    - SÉ AUDAZ: si una marca tiene Churn > 50%, advierte que el modelo de negocio es insostenible.
    - UTILIZA LOS BENCHMARKS: NSS >70% (Hegemonía), Churn >35% (Peligro), Riesgo Crítico >30% (Crisis Total).
    """

    # 4. LÓGICA DEL CHAT
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Mostrar historial (Se muestra primero para que el input quede abajo)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input del usuario (Anclado al fondo por Streamlit automáticamente)
    if prompt := st.chat_input("Ej: ¿Cuál es el problema principal de Cat's Best este bimestre?"):
        # Agregar mensaje del usuario al historial
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generar respuesta de la asistente
        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_instruction},
                    *[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                ],
                stream=True,
            )
            response = st.write_stream(stream)

        # Guardar respuesta y refrescar para mantener el input abajo
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
