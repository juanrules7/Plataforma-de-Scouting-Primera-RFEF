import streamlit as st
import scouting_rfef as sr
import pandas as pd
import io

st.set_page_config(
    page_title="Scouting | Primera RFEF",
    layout="wide",
    page_icon="⚽",
)

# ──────────────────────────────────────────────────────────────────────────────
# CARGA DE DATOS (con caché en session_state)
# ──────────────────────────────────────────────────────────────────────────────
if "df" not in st.session_state:
    with st.spinner("Cargando datos y calculando rankings..."):
        df_raw = sr.cargar_datos()
        if df_raw.empty:
            st.error("❌ No se encontraron los archivos de datos. Comprueba las rutas en scouting_rfef.py.")
            st.stop()
        st.session_state.df = sr.rankings(df_raw)

df = st.session_state.df

# ──────────────────────────────────────────────────────────────────────────────
# CABECERA
# ──────────────────────────────────────────────────────────────────────────────
st.title("⚽ Plataforma de Scouting — Primera RFEF 25/26")
st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# METODOLOGÍA
# ──────────────────────────────────────────────────────────────────────────────
with st.expander("📖 LEE PRIMERO: Metodología, calibración y lógica PAdj", expanded=True):
    st.markdown("### 📊 Cómo funcionan los Rankings")
    st.write("""
    Los rankings se calculan como **percentiles dentro de cada posición**. Cada jugador se compara
    contra sus compañeros de posición en la Primera RFEF, lo que garantiza que las puntuaciones
    reflejen el rendimiento real frente a la competencia que enfrenta.
    """)
    st.markdown("### ⚖️ Ajuste por Posesión (PAdj)")
    st.write("""
    Para garantizar que el análisis refleje calidad técnica real y no el estilo de juego del equipo,
    todos los datos han sido **ajustados por posesión (PAdj)**. Las métricas 'por 90' estándar
    pueden ser engañosas porque no tienen en cuenta las 'oportunidades' que tiene un jugador de actuar.

    Por ejemplo, un defensa en un equipo con el 70% de la posesión tiene muchas menos oportunidades
    de hacer entradas que uno en un equipo con solo el 30%. Al ajustar por posesión, normalizamos
    el entorno como si todos los jugadores operaran con un reparto equilibrado del **50%**.
    """)
    st.markdown("### 📊 Cálculo de Percentiles")
    st.write("""
    Las métricas PAdj se combinan con **tasas de éxito** para generar percentiles completos.
    Esto garantiza que una puntuación alta no sea simplemente resultado de mucho volumen, sino de
    **alta eficiencia**. Los datos provienen de Wyscout.
    """)

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# BARRA LATERAL (filtros globales)
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.header("🔍 Filtros Globales")

if st.sidebar.button("🔄 Limpiar Caché y Recargar"):
    if "df" in st.session_state:
        del st.session_state["df"]
    st.rerun()

posiciones_disponibles = sorted(df["Pos_Normalizada"].unique()) if "Pos_Normalizada" in df.columns else []
pos_filtro = st.sidebar.multiselect("Filtrar por Posición", posiciones_disponibles)

# ──────────────────────────────────────────────────────────────────────────────
# TABLA PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────
df_display = df.copy()
if pos_filtro:
    df_display = df_display[df_display["Pos_Normalizada"].isin(pos_filtro)]

st.subheader("Jugadores identificados — Primera RFEF 25/26")

if "Puntuación_Final" in df_display.columns:
    df_ranked = df_display.sort_values("Puntuación_Final", ascending=False)
    cols_mostrar = {
        "Jugador": "Jugador",
        "Equipo": "Equipo",
        "Edad": "Edad",
        "Pos_Normalizada": "Posición",
    }
    cols_ok = [c for c in cols_mostrar if c in df_ranked.columns]
    st.dataframe(
        df_ranked[cols_ok].rename(columns=cols_mostrar).head(25).reset_index(drop=True),
        use_container_width=True,
    )
else:
    st.warning("No se encontraron columnas de ranking. Verifica el procesamiento.")

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# PESTAÑAS PRINCIPALES
# ──────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "👤 Ficha del Jugador",
    "🏆 Rankings",
    "📊 Radares",
    "🔍 Búsqueda Avanzada",
    "👯 Similitud",
    "📈 Z-Score",
    "🌍 Mercado",
])

jugadores_lista = sorted(df["Jugador"].dropna().unique())

# ─── TAB 1: FICHA DEL JUGADOR ─────────────────────────────────────────────────
with tab1:
    st.header("👤 Ficha del Jugador")
    with st.expander("ℹ️ Cómo leer esta ficha"):
        st.write("Muestra información general y las 3 métricas en las que el jugador destaca más.")

    jugador_ficha = st.selectbox("Buscar jugador:", jugadores_lista, key="ficha_nombre")

    if jugador_ficha:
        ficha = sr.get_ficha_jugador(df, jugador_ficha)
        if ficha:
            c1, c2, c3, c4 = st.columns(4)
            c1, c2, c3 = st.columns(3)
            c1.metric("Equipo", ficha["Equipo"])
            c2.metric("Liga", ficha["Liga"])
            c3.metric("Edad", ficha["Edad"])

            st.subheader("⭐ Principales virtudes")
            cv1, cv2, cv3 = st.columns(3)
            for i, (metrica, valor) in enumerate(ficha["Top Virtudes"]):
                [cv1, cv2, cv3][i].info(f"**{metrica}**\n\n{valor:.1f} / 100")
        else:
            st.warning("No se pudo cargar la ficha. Comprueba que el jugador tiene datos suficientes.")

# ─── TAB 2: RANKINGS ──────────────────────────────────────────────────────────
with tab2:
    st.header("🏆 Rankings por Posición")
    with st.expander("ℹ️ Sobre los Rankings"):
        st.write("Muestra exactamente dónde se sitúa el jugador respecto a todos los jugadores de su posición en la Primera RFEF.")

    jugador_rank = st.selectbox("Seleccionar jugador:", jugadores_lista, key="rank_nombre")

    if jugador_rank:
        fig_rank = sr.plot_ranking_liga(df, jugador_rank)
        if fig_rank:
            st.pyplot(fig_rank)
            st.caption("Nota: La barra representa el rating; la etiqueta muestra la posición absoluta.")
        else:
            st.warning("No se pudo generar el gráfico. El jugador puede no tener métricas suficientes.")

# ─── TAB 3: RADARES ───────────────────────────────────────────────────────────
with tab3:
    st.header("📊 Comparativa de Radares")
    with st.expander("ℹ️ Cómo usar los Radares"):
        st.write("Selecciona hasta 3 jugadores para visualizar sus ratings en un radar comparativo.")

    c1, c2, c3 = st.columns(3)
    sel_radar = []

    with c1:
        st.markdown("### 🟢 Perfil Principal")
        p1 = st.selectbox("Jugador", jugadores_lista, key="r1_n")
        sel_radar.append((p1, df, p1))

    with c2:
        st.markdown("### 🔴 Perfil Secundario")
        activar_p2 = st.checkbox("Añadir segundo jugador", key="act_p2")
        if activar_p2:
            p2 = st.selectbox("Jugador", jugadores_lista, key="r2_n")
            sel_radar.append((p2, df, p2))

    with c3:
        st.markdown("### 🔵 Perfil de Comparación")
        activar_p3 = st.checkbox("Añadir tercer jugador", key="act_p3")
        if activar_p3:
            p3 = st.selectbox("Jugador", jugadores_lista, key="r3_n")
            sel_radar.append((p3, df, p3))

    if st.button("📊 Generar Radar"):
        fig_radar = sr.plot_radar(sel_radar)
        if fig_radar:
            _, col_r, _ = st.columns([1, 5, 1])
            with col_r:
                st.pyplot(fig_radar)

            buf = io.BytesIO()
            fig_radar.savefig(buf, format="png", dpi=300, bbox_inches="tight", facecolor="white")
            buf.seek(0)
            slug = "_vs_".join(p[0].replace(" ", "_") for p in sel_radar)
            st.download_button(
                label="⬇️ Descargar Radar (PNG)",
                data=buf,
                file_name=f"radar_{slug}.png",
                mime="image/png",
            )
        else:
            st.warning(
                "⚠️ No se pudo generar el radar. La posición del jugador puede no tener "
                "suficientes pilares calculados o los datos son insuficientes."
            )

# ─── TAB 4: BÚSQUEDA AVANZADA ─────────────────────────────────────────────────
with tab4:
    st.header("🔍 Búsqueda Avanzada")
    with st.expander("ℹ️ Sobre el Motor de Búsqueda"):
        st.write("Define requisitos mínimos. Los sliders representan **Percentiles (0-100)**. Un valor de 80 significa que el jugador debe estar en el top 20% para esa métrica.")

    metrics_busqueda = [c for c in df.columns if "_Rating" in c]
    filtros_scouting = {}

    st.subheader("📊 Requisitos por Métrica (Percentil mínimo)")
    col_f1, col_f2, col_f3 = st.columns(3)

    for i, metrica in enumerate(metrics_busqueda):
        col_target = [col_f1, col_f2, col_f3][i % 3]
        with col_target:
            filtros_scouting[metrica] = st.slider(
                metrica.replace("_Rating", ""),
                0, 100, 0,
                key=f"slider_{metrica}",
            )

    st.markdown("---")
    c_edad, c_min, c_equipo = st.columns(3)
    with c_edad:
        max_edad = st.number_input("Edad máxima", value=28, step=1)
    with c_min:
        min_minutos = st.number_input("Minutos mínimos jugados", value=500, step=100)
    with c_equipo:
        equipos_disp = ["Todos"] + sorted(df["Equipo"].dropna().unique().tolist())
        equipo_filtro = st.selectbox("Filtrar por equipo", equipos_disp)

    df_res = sr.aplicar_filtros(df, filtros_scouting)

    if not df_res.empty:
        df_res = df_res[df_res["Edad"] <= max_edad]
        if "Minutos jugados" in df_res.columns:
            df_res = df_res[df_res["Minutos jugados"] >= min_minutos]
        if equipo_filtro != "Todos":
            df_res = df_res[df_res["Equipo"] == equipo_filtro]

    st.subheader(f"✅ Jugadores encontrados: {len(df_res)}")

    if not df_res.empty:
        col_sort = st.selectbox(
            "Ordenar por:",
            options=metrics_busqueda,
            format_func=lambda x: x.replace("_Rating", ""),
            key="sort_busqueda",
        )
        cols_ver = ["Jugador", "Equipo", "Edad", "Pos_Normalizada"] + metrics_busqueda
        cols_ver = [c for c in cols_ver if c in df_res.columns]
        df_vista = df_res[cols_ver].sort_values(col_sort, ascending=False).head(50)

        st.dataframe(
            df_vista.style.background_gradient(subset=metrics_busqueda, cmap="YlGn")
            .format({c: "{:.1f}" for c in metrics_busqueda if c in df_vista.columns}),
            use_container_width=True,
            height=500,
        )

        csv_data = df_vista.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Descargar Informe de Scouting (.csv)",
            data=csv_data,
            file_name="informe_scouting_rfef.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.warning("No se encontraron jugadores con estos requisitos. Prueba a bajar los sliders.")

# ─── TAB 5: SIMILITUD ─────────────────────────────────────────────────────────
with tab5:
    st.header("👯‍♂️ Similitud")
    with st.expander("ℹ️ Cómo funciona la Similitud"):
        st.write("Calcula la similitud coseno entre jugadores. Encuentra perfiles estadísticamente parecidos independientemente del equipo.")

    col_s1, col_s2 = st.columns([1, 2])
    with col_s1:
        st.markdown("### 🎯 Jugador de Referencia")
        objetivo_sim = st.selectbox("Selecciona el jugador de referencia:", jugadores_lista, key="sim_obj")
        n_sim = st.slider("Número de resultados:", 5, 15, 10)
        run_sim = st.button("🚀 Buscar Gemelos Estadísticos", use_container_width=True)

    with col_s2:
        if run_sim:
            with st.spinner(f"Analizando el ADN estadístico de {objetivo_sim}..."):
                fig_sim = sr.plot_similitud(objetivo_sim, df, top_n=n_sim)
            if fig_sim:
                st.pyplot(fig_sim)
            else:
                st.error("No se pudo generar el gráfico de similitud. Comprueba que el jugador tiene métricas disponibles.")
        else:
            st.info("Configura el jugador de referencia a la izquierda y pulsa el botón para ver los gemelos estadísticos.")

# ─── TAB 6: Z-SCORE ───────────────────────────────────────────────────────────
with tab6:
    st.header("📈 Análisis Z-Score")
    with st.expander("ℹ️ Qué es el Z-Score"):
        st.write("Muestra desviaciones estándar respecto a la media. 0 es el promedio. +1.0 es el top 16%. +2.0 es de élite. Ayuda a identificar lo 'único' de cada rasgo del jugador.")

    cz1, cz2 = st.columns(2)
    sel_z = []
    for i, col in enumerate([cz1, cz2]):
        with col:
            p = st.selectbox(f"Jugador {i + 1}", jugadores_lista, key=f"zn{i}")
            sel_z.append((p, df))

    if st.button("Calcular Z-Score"):
        fig_z = sr.plot_zscore(sel_z)
        if fig_z:
            st.pyplot(fig_z)
        else:
            st.warning("No se pudo generar el gráfico. Comprueba que los jugadores tienen datos de rating.")

# ─── TAB 7: MERCADO ───────────────────────────────────────────────────────────
with tab7:
    st.header("🌍 Análisis de Mercado")
    with st.expander("ℹ️ Cómo usar el Análisis de Mercado"):
        st.write("Compara a tu jugador (Estrella Amarilla) con el resto de su posición en las métricas seleccionadas. Permite contextualizar si un jugador es élite en su categoría.")

    st.markdown("---")
    col_m1, col_m2 = st.columns([1, 2])
    with col_m1:
        st.markdown("### ⚙️ Configuración")
        objetivo_m = st.selectbox("Analizar jugador:", jugadores_lista, key="mkt_obj")
        all_ratings = [c for c in df.columns if "_Rating" in c]
        selected_m = st.multiselect(
            "Métricas para el eje Y:",
            options=all_ratings,
            default=all_ratings[:3],
            format_func=lambda x: x.replace("_Rating", ""),
        )
        st.info(f"Comparando **{objetivo_m}** contra el resto de jugadores de la Primera RFEF.")

    with col_m2:
        if objetivo_m and selected_m:
            fig_mkt = sr.plot_mercado(df, objetivo_m, selected_m)
            if fig_mkt:
                st.pyplot(fig_mkt)
            else:
                st.warning("No se pudo generar el gráfico. Selecciona al menos una métrica.")
        else:
            st.info("Selecciona un jugador y al menos una métrica para generar el análisis.")
