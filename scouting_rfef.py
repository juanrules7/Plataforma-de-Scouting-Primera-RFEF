import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from mplsoccer import Radar
from sklearn.metrics.pairwise import cosine_similarity


# ──────────────────────────────────────────────────────────────────────────────
# 1. CARGA DE DATOS
# ──────────────────────────────────────────────────────────────────────────────
ARCHIVOS_RFEF = [
    r"C:\Users\juanf\OneDrive\Documentos\DATA\NEW\ATT SPAIN 3 2526.xlsx",
    r"C:\Users\juanf\OneDrive\Documentos\DATA\NEW\DEF SPAIN 3 2526.xlsx",
    r"C:\Users\juanf\OneDrive\Documentos\DATA\NEW\MID SPAIN 3 2526.xlsx",
]

POSESION_RFEF_JSON = [
    {"team": {"name": "Real Madrid Castilla"}, "statistics": {"averageBallPossession": 62.9}},
    {"team": {"name": "Betis Deportivo"}, "statistics": {"averageBallPossession": 59.5}},
    {"team": {"name": "Celta Fortuna"}, "statistics": {"averageBallPossession": 57.1}},
    {"team": {"name": "Lugo"}, "statistics": {"averageBallPossession": 56.3}},
    {"team": {"name": "Atlético Madrid B"}, "statistics": {"averageBallPossession": 55.4}},
    {"team": {"name": "Tenerife"}, "statistics": {"averageBallPossession": 54.7}},
    {"team": {"name": "Eldense"}, "statistics": {"averageBallPossession": 54.7}},
    {"team": {"name": "Real Murcia"}, "statistics": {"averageBallPossession": 53.3}},
    {"team": {"name": "Barakaldo"}, "statistics": {"averageBallPossession": 53.1}},
    {"team": {"name": "Hércules"}, "statistics": {"averageBallPossession": 52.4}},
    {"team": {"name": "Ibiza"}, "statistics": {"averageBallPossession": 51.7}},
    {"team": {"name": "Athletic Bilbao U21"}, "statistics": {"averageBallPossession": 51.6}},
    {"team": {"name": "Cartagena"}, "statistics": {"averageBallPossession": 51.4}},
    {"team": {"name": "Marbella"}, "statistics": {"averageBallPossession": 51.1}},
    {"team": {"name": "Sabadell"}, "statistics": {"averageBallPossession": 50.9}},
    {"team": {"name": "Arenteiro"}, "statistics": {"averageBallPossession": 50.6}},
    {"team": {"name": "Villarreal B"}, "statistics": {"averageBallPossession": 50.1}},
    {"team": {"name": "Antequera"}, "statistics": {"averageBallPossession": 49.8}},
    {"team": {"name": "Unionistas de Salamanca"}, "statistics": {"averageBallPossession": 49.7}},
    {"team": {"name": "Alcorcón"}, "statistics": {"averageBallPossession": 49.5}},
    {"team": {"name": "Europa"}, "statistics": {"averageBallPossession": 49.4}},
    {"team": {"name": "Zamora"}, "statistics": {"averageBallPossession": 49.4}},
    {"team": {"name": "Pontevedra"}, "statistics": {"averageBallPossession": 49.3}},
    {"team": {"name": "Guadalajara"}, "statistics": {"averageBallPossession": 49.0}},
    {"team": {"name": "Racing Ferrol"}, "statistics": {"averageBallPossession": 49.0}},
    {"team": {"name": "Algeciras"}, "statistics": {"averageBallPossession": 49.0}},
    {"team": {"name": "Gimnàstic Tarragona"}, "statistics": {"averageBallPossession": 48.7}},
    {"team": {"name": "Arenas Club"}, "statistics": {"averageBallPossession": 48.4}},
    {"team": {"name": "Sevilla Atlético"}, "statistics": {"averageBallPossession": 47.5}},
    {"team": {"name": "Ourense CF"}, "statistics": {"averageBallPossession": 47.3}},
    {"team": {"name": "Real Avilés"}, "statistics": {"averageBallPossession": 46.5}},
    {"team": {"name": "Mérida AD"}, "statistics": {"averageBallPossession": 46.4}},
    {"team": {"name": "Juventud Torremolinos"}, "statistics": {"averageBallPossession": 45.9}},
    {"team": {"name": "Ponferradina"}, "statistics": {"averageBallPossession": 45.8}},
    {"team": {"name": "Atlético Sanluqueño"}, "statistics": {"averageBallPossession": 45.7}},
    {"team": {"name": "Osasuna Promesas"}, "statistics": {"averageBallPossession": 45.1}},
    {"team": {"name": "Talavera CF"}, "statistics": {"averageBallPossession": 43.4}},
    {"team": {"name": "Cacereño"}, "statistics": {"averageBallPossession": 42.5}},
    {"team": {"name": "Tarazona"}, "statistics": {"averageBallPossession": 40.9}},
    {"team": {"name": "Teruel"}, "statistics": {"averageBallPossession": 40.6}},
]

POSESION_MAP = {entry["team"]["name"]: entry["statistics"]["averageBallPossession"]
                for entry in POSESION_RFEF_JSON}


def cargar_datos():
    """Carga y concatena los tres archivos de la Primera RFEF."""
    dfs = []
    for ruta in ARCHIVOS_RFEF:
        if not os.path.exists(ruta):
            print(f"⚠️ Archivo no encontrado: {ruta}")
            continue
        df = pd.read_excel(ruta)
        df["Liga"] = "Primera RFEF"
        df["Temporada"] = "25/26"
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    df_total = pd.concat(dfs, ignore_index=True)

    # Aplicar posesión por equipo (valor de posesión media del equipo)
    df_total["team_possession"] = df_total["Equipo"].map(POSESION_MAP)
    # Para equipos sin dato de posesión, usamos la media de la liga (50%)
    df_total["team_possession"] = df_total["team_possession"].fillna(50.0)

    return df_total


# ──────────────────────────────────────────────────────────────────────────────
# 2. RANKINGS Y MÉTRICAS AJUSTADAS POR POSESIÓN
# ──────────────────────────────────────────────────────────────────────────────
def rankings(df):
    mapeo_grupos = {
        "RCMF": "CMF", "LCMF": "CMF",
        "RAMF": "RW", "LAMF": "LW",
        "LWF": "LW",
        "RWF": "RW",
        "LDMF": "DMF", "RDMF": "DMF",
        "RWB": "RB", "LWB": "LB",
        "LCB": "CB", "RCB": "CB",
    }

    _SHARED = {
        "defensivas": [
            "Interceptaciones/90", "Entradas/90", "Duelos defensivos/90",
            "Faltas/90", "Duelos aéreos en los 90",
        ],
        "ofensivas": [
            "Pases progresivos/90", "Carreras en progresión/90",
            "Pases largos/90", "Centros/90",
            "Regates/90", "xA/90", "Desmarques/90", "xG/90",
        ],
        "pilares": {
            "Agresividad":        ["Faltas/90_PAdj", "Entradas/90_PAdj"],
            "Duelos Defensivos":  ["Duelos defensivos ganados, %", "Duelos defensivos/90_PAdj"],
            "Interceptaciones":   ["Interceptaciones/90_PAdj"],
            "Pases Progresivos":  ["Pases progresivos/90_OPAdj", "Precisión pases progresivos, %"],
            "Pases Largos":       ["Pases largos/90_OPAdj", "Precisión pases largos, %"],
            "Conducción":         ["Carreras en progresión/90_OPAdj"],
            "Regates":            ["Regates/90_OPAdj", "Regates realizados, %"],
            "Centros":            ["Crossing/90_OPAdj", "Precisión centros, %"],
            "Creatividad":        ["xA/90_OPAdj"],
            "Desmarques":         ["Desmarques/90_OPAdj", "Precisión desmarques, %"],
            "Amenaza Gol":        ["xG/90_OPAdj"],
            "Duelos Aéreos":      ["Duelos aéreos en los 90_PAdj", "Duelos aéreos ganados, %"],
        },
    }

    _GK = {
        "defensivas": ["Goles evitados/90", "Salidas/90"],
        "ofensivas": ["Pases progresivos/90", "Pases largos/90"],
        "pilares": {
            "Bajo Palos":         ["Goles evitados/90"],
            "Salidas":            ["Salidas/90_PAdj"],
            "Pases Largos":       ["Pases largos/90_OPAdj", "Precisión pases largos, %"],
            "Pases Progresivos":  ["Pases progresivos/90_OPAdj", "Precisión pases progresivos, %"],
        },
    }

    config_posiciones = {
        "CB": _SHARED, "RB": _SHARED, "LB": _SHARED,
        "DMF": _SHARED, "CMF": _SHARED, "AMF": _SHARED,
        "LW": _SHARED, "RW": _SHARED, "CF": _SHARED,
        "GK": _GK,
    }

    df = df.copy()

    # Normalizar nombres de columna con encoding corrupto
    col_pos = next((c for c in df.columns if "osici" in c and "spec" in c.lower()), None)
    if col_pos is None:
        col_pos = next((c for c in df.columns if "osici" in c), None)
    if col_pos:
        df.rename(columns={col_pos: "Posición específica"}, inplace=True)

    # Renombrar columna de duelos aéreos si tiene encoding corrupto
    for c in list(df.columns):
        if "reos" in c and "90" in c and "Duelos" in c and "aéreos" not in c:
            df.rename(columns={c: "Duelos aéreos en los 90"}, inplace=True)
            break

    if "Posición específica" not in df.columns:
        return df

    df["Posición específica"] = df["Posición específica"].str.strip()
    df["Posición_Principal"] = df["Posición específica"].str.split(",").str[0].str.strip()
    df["Pos_Normalizada"] = df["Posición_Principal"].map(mapeo_grupos).fillna(df["Posición_Principal"])

    all_results = []

    for TARGET_POS, conf in config_posiciones.items():
        df_pos = df[df["Pos_Normalizada"] == TARGET_POS].copy()
        if df_pos.empty:
            continue

        if TARGET_POS == "GK" and "Goles evitados/90" in df_pos.columns:
            df_pos["Goles evitados/90"] = -df_pos["Goles evitados/90"]

        for stat in conf["defensivas"]:
            if stat in df_pos.columns:
                df_pos[f"{stat}_PAdj"] = df_pos[stat] * (50 / (100 - df_pos["team_possession"]))

        for stat in conf["ofensivas"]:
            if stat in df_pos.columns:
                df_pos[f"{stat}_OPAdj"] = df_pos[stat] * (50 / df_pos["team_possession"])

        # Centros necesitan alias (Wyscout usa "Centros/90" pero pilares usan "Crossing/90_OPAdj")
        if "Centros/90_OPAdj" in df_pos.columns and "Crossing/90_OPAdj" not in df_pos.columns:
            df_pos["Crossing/90_OPAdj"] = df_pos["Centros/90_OPAdj"]

        # Toda la competición es una sola liga → percentiles sobre todos los jugadores de esa posición
        df_pos2 = df_pos.copy()
        rating_cols = []

        for pilar, metrics in conf["pilares"].items():
            valid_m = [m for m in metrics if m in df_pos2.columns]
            if valid_m:
                temp_pcts = [df_pos2[m].rank(pct=True) * 100 for m in valid_m]
                pilar_pct_avg = pd.concat(temp_pcts, axis=1).mean(axis=1)
                col_name = f"{pilar}_Rating"
                df_pos2[col_name] = pilar_pct_avg
                rating_cols.append(col_name)

        if rating_cols:
            df_pos2["Puntuación_Final"] = df_pos2[rating_cols].mean(axis=1)
            all_results.append(df_pos2)

    return pd.concat(all_results, ignore_index=True) if all_results else df


# ──────────────────────────────────────────────────────────────────────────────
# 3. FILTROS DE SCOUTING
# ──────────────────────────────────────────────────────────────────────────────
def aplicar_filtros(df, filtros):
    if df is None or df.empty:
        return pd.DataFrame()
    df_res = df.copy()
    for col, minimo in filtros.items():
        if minimo > 0 and col in df_res.columns:
            df_res = df_res[df_res[col] >= minimo]
    if "Puntuación_Final" in df_res.columns:
        df_res = df_res.sort_values("Puntuación_Final", ascending=False)
    return df_res


# ──────────────────────────────────────────────────────────────────────────────
# 4. FICHA DEL JUGADOR
# ──────────────────────────────────────────────────────────────────────────────
def get_ficha_jugador(df, nombre):
    try:
        fila = df[df["Jugador"] == nombre].iloc[0]
        ratings = {c.replace("_Rating", ""): fila[c] for c in df.columns if "_Rating" in c}
        top_3 = sorted(ratings.items(), key=lambda x: x[1], reverse=True)[:3]
        return {
            "Equipo": fila["Equipo"],
            "Liga": fila["Liga"],
            "Edad": int(fila["Edad"]),
            "Puntuación": round(fila["Puntuación_Final"], 1),
            "Top Virtudes": top_3,
        }
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# 5. GRÁFICO DE RANKING EN LA LIGA
# ──────────────────────────────────────────────────────────────────────────────
def plot_ranking_liga(df, nombre):
    try:
        fila = df[df["Jugador"] == nombre].iloc[0]
        pos_norm = fila["Pos_Normalizada"]
        df_pos = df[df["Pos_Normalizada"] == pos_norm].copy()
        metrics = [c for c in df.columns if "_Rating" in c]

        ranks = []
        for m in metrics:
            if m not in df_pos.columns:
                continue
            df_pos[f"rk_{m}"] = df_pos[m].rank(ascending=False)
            pos_rk = int(df_pos[df_pos["Jugador"] == nombre][f"rk_{m}"].iloc[0])
            total = len(df_pos)
            valor = float(fila[m]) if pd.notna(fila[m]) else 0.0
            ranks.append({"Métrica": m.replace("_Rating", ""), "Posición": pos_rk,
                          "Total": total, "Valor": valor})

        df_rank = pd.DataFrame(ranks).sort_values("Valor", ascending=True)
        df_rank = df_rank[df_rank["Valor"] > 0]

        fig, ax = plt.subplots(figsize=(9, 6))
        colores = plt.cm.RdYlGn(np.array(df_rank["Valor"]) / 100)
        bars = ax.barh(df_rank["Métrica"], df_rank["Valor"],
                       color=colores, edgecolor="black", alpha=0.75)

        for i, bar in enumerate(bars):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f"Pos: {df_rank.iloc[i]['Posición']}º / {df_rank.iloc[i]['Total']}",
                    va="center", fontweight="bold", fontsize=9)

        ax.set_xlim(0, 120)
        ax.set_title(f"Rendimiento de {nombre} vs su posición ({pos_norm})",
                     fontsize=12, fontweight="bold")
        sns.despine(left=True, bottom=True)
        plt.tight_layout()
        return fig
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# 6. RADAR CHART (hasta 3 jugadores)
# ──────────────────────────────────────────────────────────────────────────────
def plot_radar(lista_jugadores):
    """
    lista_jugadores: [(nombre, df, etiqueta), ...]
    """
    extracted, labels = [], []
    for nombre, df, etiqueta in lista_jugadores:
        try:
            fila = df[df["Jugador"] == nombre].iloc[0]
            extracted.append(fila)
            labels.append(etiqueta)
        except Exception:
            continue

    if not extracted:
        return None

    pilares_cols = [c for c in extracted[0].index
                    if "_Rating" in c and pd.notna(extracted[0][c])]
    if len(pilares_cols) < 3:
        return None

    params = [c.replace("_Rating", "") for c in pilares_cols]
    low = [0] * len(params)
    high = [100] * len(params)

    values = []
    for p in extracted:
        row_vals = [float(p[c]) if pd.notna(p[c]) else 0.0 for c in pilares_cols]
        values.append(row_vals)

    colors = ["#77BA99", "#e63946", "#F3E03B"]

    try:
        radar = Radar(params, low, high, round_int=[False] * len(params),
                      num_rings=5, center_circle_radius=1)
    except TypeError:
        radar = Radar(params, low, high, num_rings=5, center_circle_radius=1)

    fig, ax = radar.setup_axis()
    radar.draw_circles(ax=ax, facecolor="#f5f5f5", edgecolor="#cccccc", zorder=1)

    for i, (val, color) in enumerate(zip(values, colors)):
        alpha = 0.4 if len(values) > 1 else 0.5
        result = radar.draw_radar(
            ax=ax, values=val,
            kwargs_radar={"facecolor": color, "alpha": alpha},
            kwargs_rings={"facecolor": color, "alpha": 0.1},
        )
        try:
            polygon = result[0] if isinstance(result, tuple) else result
            verts = polygon.get_path().vertices
            n = len(params)
            use_verts = verts[:n] if len(verts) > n else verts
            for j, (vx, vy) in enumerate(use_verts):
                if j < len(val):
                    ax.text(vx, vy, f"{val[j]:.0f}", fontsize=8,
                            ha="center", va="center", color="white",
                            fontweight="bold", zorder=15,
                            bbox=dict(boxstyle="round,pad=0.15",
                                      facecolor=color, alpha=0.9, edgecolor="none"))
        except Exception:
            pass

    radar.draw_range_labels(ax=ax, fontsize=9, fontproperties="monospace", zorder=12)
    radar.draw_param_labels(ax=ax, fontsize=11, fontproperties="monospace",
                            fontweight="bold", zorder=12)

    colors3 = colors[:len(labels)]
    if len(labels) == 1:
        ax.text(0.5, 1.02, labels[0], fontsize=16, color=colors3[0],
                ha="center", va="center", transform=ax.transAxes,
                fontfamily="monospace", fontweight="bold")
    elif len(labels) == 2:
        ax.text(0.3, 1.02, labels[0], fontsize=14, color=colors3[0],
                ha="center", va="center", transform=ax.transAxes,
                fontfamily="monospace", fontweight="bold")
        ax.text(0.5, 1.02, "vs", fontsize=14, color="black",
                ha="center", va="center", transform=ax.transAxes, fontfamily="monospace")
        ax.text(0.7, 1.02, labels[1], fontsize=14, color=colors3[1],
                ha="center", va="center", transform=ax.transAxes,
                fontfamily="monospace", fontweight="bold")
    else:
        positions = [0.15, 0.5, 0.85]
        for idx, (lbl, clr, xp) in enumerate(zip(labels, colors3, positions)):
            ax.text(xp, 1.02, lbl, fontsize=11, color=clr,
                    ha="center", va="center", transform=ax.transAxes,
                    fontfamily="monospace", fontweight="bold")
            if idx < len(labels) - 1:
                ax.text((positions[idx] + positions[idx + 1]) / 2, 1.02, "vs",
                        fontsize=11, color="black", ha="center", va="center",
                        transform=ax.transAxes, fontfamily="monospace")

    ax.text(0.0, -0.1,
            "Ratings: Percentiles (0-100) ajustados por posesión del equipo",
            fontsize=9, ha="left", va="center", transform=ax.transAxes,
            fontfamily="monospace", color="#555555")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 7. SIMILITUD DE JUGADORES
# ──────────────────────────────────────────────────────────────────────────────
def plot_similitud(nombre, df, top_n=10):
    features = [c for c in df.columns if "_Rating" in c]
    if nombre not in df["Jugador"].values or not features:
        return None

    target_vec = df[df["Jugador"] == nombre][features].fillna(0).values
    candidatos = df[features].fillna(0).values

    try:
        scores = cosine_similarity(target_vec, candidatos)[0]
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        result = []
        for i, score in ranked:
            if df.iloc[i]["Jugador"] == nombre and score > 0.99:
                continue
            result.append((i, score))
            if len(result) >= top_n:
                break

        names = [f"{df.iloc[i]['Jugador']} ({df.iloc[i]['Equipo']})" for i, _ in result]
        sims = [score * 100 for _, score in result]

        if not names:
            return None

        fig, ax = plt.subplots(figsize=(10, 6), facecolor="#ffffff")
        bars = ax.barh(names, sims, color="#D4AF37", edgecolor="black", alpha=0.8)
        ax.invert_yaxis()
        ax.set_xlim(max(0, min(sims) - 10), 105)
        ax.set_title(f"Gemelos estadísticos de {nombre}", fontweight="bold", pad=20)
        for bar in bars:
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{bar.get_width():.1f}%", va="center", fontweight="bold")
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"Error en similitud: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# 8. Z-SCORE
# ──────────────────────────────────────────────────────────────────────────────
def plot_zscore(lista_jugadores):
    player_data, pilares_cols = [], []
    for nombre, df in lista_jugadores:
        if df.empty:
            continue
        if not pilares_cols:
            pilares_cols = [c for c in df.columns if "_Rating" in c]
        df_z = df.copy()
        for col in pilares_cols:
            mean = df_z[col].mean()
            std = df_z[col].std() + 1e-6
            df_z[f"z_{col}"] = (df_z[col] - mean) / std
        try:
            player_data.append(df_z[df_z["Jugador"] == nombre].iloc[0])
        except Exception:
            continue

    if not player_data:
        return None

    params = [c.replace("_Rating", "") for c in pilares_cols]
    fig, ax = plt.subplots(figsize=(10, 7))
    y = np.arange(len(params))
    height = 0.8 / len(player_data)
    colors = ["#77BA99", "#E84855", "#3182CE"]

    for i, data in enumerate(player_data):
        z_vals = [data.get(f"z_{col}", 0) for col in pilares_cols]
        ax.barh(y + (i - (len(player_data) - 1) / 2) * height, z_vals,
                height, label=data["Jugador"], color=colors[i % 3])

    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(params, fontweight="bold")
    ax.legend()
    ax.set_title("Comparativa Z-Score (desviaciones respecto a la media)", fontweight="bold")
    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 9. ANÁLISIS DE MERCADO
# ──────────────────────────────────────────────────────────────────────────────
def plot_mercado(df, nombre_objetivo, metrics):
    if df is None or df.empty or not metrics:
        return None

    df_plot = df.copy()
    df_plot["Combinado"] = df_plot[metrics].mean(axis=1).fillna(0)
    df_target = df_plot[df_plot["Jugador"] == nombre_objetivo].copy()

    fig, ax = plt.subplots(figsize=(12, 8), facecolor="#FCF9F0")
    ax.set_facecolor("#FCF9F0")

    u23 = df_plot[df_plot["Edad"] <= 23]
    vets = df_plot[df_plot["Edad"] > 23]

    ax.scatter(vets["Edad"], vets["Combinado"],
               color="#d3d3d3", alpha=0.4, s=60, label="Veteranos")
    ax.scatter(u23["Edad"], u23["Combinado"],
               color="#5DA5DA", alpha=0.8, s=120, edgecolor="black", lw=1.2, label="Sub-23")

    if not df_target.empty:
        p = df_target.iloc[0]
        ax.scatter(p["Edad"], p["Combinado"], color="#FDD835", s=600, marker="*",
                   edgecolor="black", zorder=15, label=f"OBJETIVO: {nombre_objetivo}")

        df_plot["dist"] = np.sqrt(
            (df_plot["Edad"] - p["Edad"]) ** 2 + (df_plot["Combinado"] - p["Combinado"]) ** 2
        )
        cercanos = df_plot[df_plot["Jugador"] != nombre_objetivo].sort_values("dist").head(6)
        for _, row in cercanos.iterrows():
            ax.text(row["Edad"], row["Combinado"] + 1.8, row["Jugador"],
                    fontsize=8.5, ha="center", fontweight="bold", color="#333333",
                    bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", boxstyle="round,pad=0.2"))

    media = df_plot["Combinado"].mean()
    ax.axhline(media, color="gray", linestyle=(0, (1, 10)), lw=1.5, alpha=0.6, label="Media")
    ax.axvline(23.5, color="#F8BBD0", linestyle=(0, (5, 5)), lw=1.5, alpha=0.8)

    nombres_m = [m.replace("_Rating", "") for m in metrics]
    ax.set_title(f"Posicionamiento de Mercado: {nombre_objetivo}", fontsize=16, fontweight="bold", pad=30)
    ax.set_xlabel("Edad", fontsize=11, labelpad=10)
    ax.set_ylabel(f"Rating combinado ({' + '.join(nombres_m)})", fontsize=11, labelpad=10)
    ax.legend(loc="lower right", facecolor="white", frameon=True, fontsize=10)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    return fig
