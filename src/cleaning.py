"""
YouTube Channel Data Cleaning
==============================
Limpia y enriquece los datos crudos extraídos de la API.
Genera archivos procesados listos para el análisis exploratorio.

Autor: Bryan Vargas]
"""

import re
import pandas as pd

# ─── Rutas ────────────────────────────────────────────────────────────────────

RAW_ALL        = "data/raw/all_videos.csv"
RAW_SHORTS     = "data/raw/shorts.csv"
RAW_LONG       = "data/raw/long_videos.csv"
RAW_COMMENTS   = "data/raw/comments.csv"

OUT_SHORTS     = "data/processed/shorts_clean.csv"
OUT_LONG       = "data/processed/long_videos_clean.csv"
OUT_COMMENTS   = "data/processed/comments_clean.csv"


# ─── Helpers ─────────────────────────────────────────────────────────────────

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["published_at"])
    print(f"   Cargado: {path}  →  {len(df)} filas, {df.shape[1]} columnas")
    return df


def basic_report(df: pd.DataFrame, name: str):
    """Imprime un diagnóstico rápido del DataFrame."""
    print(f"\n  [{name}]")
    print(f"   Filas:      {len(df)}")
    id_col = "video_id" if "video_id" in df.columns else "comment_id"
    print(f"   Duplicados: {df.duplicated(subset=id_col).sum()}")
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    if len(nulls):
        print(f"   Nulos:\n{nulls.to_string()}")
    else:
        print("   Nulos:      ninguno ✅")


def clean_text(text: str) -> str:
    """Elimina saltos de línea y espacios extra de un texto."""
    if not isinstance(text, str):
        return ""
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega columnas de fecha derivadas:
      - year, month, day_of_week (nombre), week_of_year
    Útiles para analizar patrones de publicación.
    """
    df["year"]         = df["published_at"].dt.year
    df["month"]        = df["published_at"].dt.month
    df["month_name"]   = df["published_at"].dt.strftime("%b")   # Ene, Feb...
    df["day_of_week"]  = df["published_at"].dt.dayofweek        # 0=Lunes
    df["day_name"]     = df["published_at"].dt.strftime("%A")   # Monday...
    df["week_of_year"] = df["published_at"].dt.isocalendar().week.astype(int)
    return df


def add_engagement(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula métricas de engagement:
      - engagement_rate  = (likes + comentarios) / vistas  (× 100, en %)
      - like_rate        = likes / vistas (× 100, en %)
      - comment_rate     = comentarios / vistas (× 100, en %)

    Estas métricas normalizan el rendimiento: un video con 10k vistas
    y 1k likes es más efectivo que uno con 100k vistas y 2k likes.
    """
    df["engagement_rate"] = (
        (df["like_count"] + df["comment_count"]) / df["view_count"].replace(0, 1) * 100
    ).round(3)

    df["like_rate"] = (
        df["like_count"] / df["view_count"].replace(0, 1) * 100
    ).round(3)

    df["comment_rate"] = (
        df["comment_count"] / df["view_count"].replace(0, 1) * 100
    ).round(3)

    return df


def add_performance_tier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clasifica cada video en un tier según sus vistas:
      🔴 Bajo     → cuartil inferior (Q1)
      🟡 Medio    → entre Q1 y Q3
      🟢 Alto     → entre Q3 y 90th percentile
      ⭐ Viral    → top 10%

    Esto permite identificar fácilmente qué contenido sobresale.
    """
    q1  = df["view_count"].quantile(0.25)
    q3  = df["view_count"].quantile(0.75)
    p90 = df["view_count"].quantile(0.90)

    def tier(views):
        if views >= p90: return "Viral"
        if views >= q3:  return "Alto"
        if views >= q1:  return "Medio"
        return "Bajo"

    df["performance_tier"] = df["view_count"].apply(tier)
    return df


def add_title_length(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega la longitud del título en palabras y caracteres.
    Puede correlacionarse con el rendimiento.
    """
    df["title_word_count"] = df["title"].apply(lambda t: len(str(t).split()))
    df["title_char_count"] = df["title"].apply(lambda t: len(str(t)))
    return df


def clean_videos(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline completo de limpieza para Shorts y videos largos."""

    # 1. Eliminar duplicados por video_id
    before = len(df)
    df = df.drop_duplicates(subset="video_id").reset_index(drop=True)
    if len(df) < before:
        print(f"   Duplicados eliminados: {before - len(df)}")

    # 2. Rellenar nulos en texto
    df["title"]       = df["title"].fillna("Sin título")
    df["description"] = df["description"].fillna("")
    df["tags"]        = df["tags"].fillna("")

    # 3. Limpiar texto
    df["title"]       = df["title"].apply(clean_text)
    df["description"] = df["description"].apply(clean_text)

    # 4. Asegurar tipos numéricos
    for col in ["view_count", "like_count", "comment_count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df["duration_sec"] = pd.to_numeric(df["duration_sec"], errors="coerce").fillna(0).astype(int)

    # 5. Agregar columnas derivadas
    df = add_time_features(df)
    df = add_engagement(df)
    df = add_performance_tier(df)
    df = add_title_length(df)

    return df


def clean_comments(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline de limpieza para comentarios."""

    if df.empty:
        print("   ⚠️  DataFrame de comentarios vacío, saltando limpieza.")
        return df

    # 1. Eliminar duplicados
    df = df.drop_duplicates(subset="comment_id").reset_index(drop=True)

    # 2. Limpiar texto
    df["text"] = df["text"].fillna("").apply(clean_text)

    # 3. Eliminar comentarios vacíos después de limpiar
    df = df[df["text"].str.len() > 0].reset_index(drop=True)

    # 4. Tipos numéricos
    df["like_count"]  = pd.to_numeric(df["like_count"],  errors="coerce").fillna(0).astype(int)
    df["reply_count"] = pd.to_numeric(df["reply_count"], errors="coerce").fillna(0).astype(int)

    # 5. Métrica de relevancia del comentario
    df["relevance_score"] = df["like_count"] + (df["reply_count"] * 2)

    return df


def print_summary(df_shorts, df_long, df_comments):
    """Resumen final de los datos procesados."""
    print("\n" + "=" * 55)
    print("  Resumen de datos procesados")
    print("=" * 55)

    print(f"\n  SHORTS ({len(df_shorts)} videos)")
    print(f"   Vistas totales:   {df_shorts['view_count'].sum():>12,}")
    print(f"   Vistas promedio:  {df_shorts['view_count'].mean():>12,.0f}")
    print(f"   Vistas máximas:   {df_shorts['view_count'].max():>12,}")
    print(f"   Engagement prom:  {df_shorts['engagement_rate'].mean():>11.2f}%")
    print(f"\n   Tiers de rendimiento:")
    for tier, count in df_shorts["performance_tier"].value_counts().items():
        print(f"     {tier:<8}: {count} videos")

    print(f"\n  VIDEOS LARGOS ({len(df_long)} videos)")
    print(f"   Vistas totales:   {df_long['view_count'].sum():>12,}")
    print(f"   Vistas promedio:  {df_long['view_count'].mean():>12,.0f}")
    print(f"   Vistas máximas:   {df_long['view_count'].max():>12,}")
    print(f"   Duración prom:    {df_long['duration_sec'].mean()/60:>10.1f} min")

    if not df_comments.empty:
        print(f"\n  COMENTARIOS ({len(df_comments)} comentarios)")
        print(f"   Likes promedio:   {df_comments['like_count'].mean():>12.1f}")
        print(f"   Más relevante:")
        top = df_comments.sort_values("relevance_score", ascending=False).iloc[0]
        preview = top["text"][:80] + "..." if len(top["text"]) > 80 else top["text"]
        print(f"     \"{preview}\"")

    print("\n" + "=" * 55)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    import os
    os.makedirs("data/processed", exist_ok=True)

    print("=" * 55)
    print("  YouTube Data Cleaner")
    print("=" * 55)

    # Cargar datos
    print("\n[1/4] Cargando datos crudos...")
    # Reclasificar: Short = 61s o menos, O menciona #shorts con menos de 3 min
    df_all = load_csv(RAW_ALL)
    df_all["is_short"] = (
        (df_all["duration_sec"] <= 61) |
        (
            (df_all["duration_sec"] <= 180) &
            (
                df_all["title"].str.lower().str.contains("#shorts", na=False) |
                df_all["description"].str.lower().str.contains("#shorts", na=False)
            )
        )
    )

    df_shorts = df_all[df_all["is_short"]].copy().reset_index(drop=True)
    df_long   = df_all[~df_all["is_short"]].copy().reset_index(drop=True)
    df_comments = load_csv(RAW_COMMENTS)

    # Diagnóstico inicial
    print("\n[2/4] Diagnóstico inicial...")
    basic_report(df_shorts,   "Shorts")
    basic_report(df_long,     "Videos largos")
    basic_report(df_comments, "Comentarios")

    # Limpieza
    print("\n[3/4] Aplicando limpieza...")
    df_shorts   = clean_videos(df_shorts)
    df_long     = clean_videos(df_long)
    df_comments = clean_comments(df_comments)
    print("   ✅ Limpieza completada")

    # Guardar
    print("\n[4/4] Guardando datos procesados...")
    df_shorts.to_csv("data/raw/shorts.csv",      index=False, encoding="utf-8-sig")
    df_long.to_csv("data/raw/long_videos.csv",   index=False, encoding="utf-8-sig")
    df_comments.to_csv(OUT_COMMENTS, index=False, encoding="utf-8-sig")
    print(f"   ✅ Guardado en data/processed/")

    # Resumen
    print_summary(df_shorts, df_long, df_comments)


if __name__ == "__main__":
    main()