"""
YouTube Channel Data Extraction
================================
Extrae Shorts, videos largos y comentarios de un canal de YouTube
usando la YouTube Data API v3.

Autor: [Tu nombre]
Proyecto: Análisis de contenido para recomendación de videos
"""

import os
import time
import isodate
import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build

# ─── Configuración ────────────────────────────────────────────────────────────

load_dotenv()

API_KEY = os.getenv("YOUTUBE_API_KEY")
youtube = build("youtube", "v3", developerKey=API_KEY)

# ⬇️ CAMBIA ESTOS DOS VALORES ANTES DE EJECUTAR
CHANNEL_HANDLE  = "Epical-Q"     # Ej: "@midotcom" o "UCxxxxxxxxxxxxxx"
VIDEO_PREGUNTA  = "2s8KgYcDwG8"  # ID del video donde pregunta qué contenido hacer
MAX_COMMENTS    = 500                   # Máximo de comentarios a extraer


# ─── Funciones de extracción ──────────────────────────────────────────────────

def get_channel_id(handle_or_id: str) -> str:
    """
    Obtiene el channel_id a partir de un @handle o lo devuelve directo
    si ya empieza con 'UC' (ya es un ID).
    """
    if handle_or_id.startswith("UC"):
        return handle_or_id

    # Busca el canal por nombre/handle
    response = youtube.search().list(
        part="snippet",
        q=handle_or_id,
        type="channel",
        maxResults=1
    ).execute()

    if not response.get("items"):
        raise ValueError(f"No se encontró el canal: {handle_or_id}")

    channel_id = response["items"][0]["snippet"]["channelId"]
    print(f"   Canal encontrado: {response['items'][0]['snippet']['title']} ({channel_id})")
    return channel_id


def get_uploads_playlist_id(channel_id: str) -> tuple[str, dict]:
    """
    Devuelve el ID de la playlist 'uploads' del canal y la info general del canal.
    Todos los canales tienen una playlist interna con todos sus videos subidos.
    """
    response = youtube.channels().list(
        part="contentDetails,snippet,statistics",
        id=channel_id
    ).execute()

    channel_data = response["items"][0]
    playlist_id  = channel_data["contentDetails"]["relatedPlaylists"]["uploads"]

    channel_info = {
        "channel_id":        channel_id,
        "channel_name":      channel_data["snippet"]["title"],
        "description":       channel_data["snippet"]["description"],
        "country":           channel_data["snippet"].get("country", "N/A"),
        "total_subscribers": int(channel_data["statistics"].get("subscriberCount", 0)),
        "total_views":       int(channel_data["statistics"].get("viewCount", 0)),
        "total_videos":      int(channel_data["statistics"].get("videoCount", 0)),
    }

    print(f"   Suscriptores: {channel_info['total_subscribers']:,}")
    print(f"   Videos totales: {channel_info['total_videos']:,}")

    return playlist_id, channel_info


def get_all_video_ids(playlist_id: str) -> list[str]:
    """
    Recorre la playlist 'uploads' página por página y recolecta todos los IDs.
    La API devuelve máximo 50 resultados por página, por eso el bucle.
    """
    video_ids       = []
    next_page_token = None

    while True:
        response = youtube.playlistItems().list(
            part="contentDetails",
            playlistId=playlist_id,
            maxResults=50,
            pageToken=next_page_token
        ).execute()

        for item in response["items"]:
            video_ids.append(item["contentDetails"]["videoId"])

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

        time.sleep(0.1)  # Evita saturar la API

    return video_ids


def get_video_details(video_ids: list[str]) -> list[dict]:
    """
    Obtiene estadísticas y metadatos de cada video.
    La API acepta hasta 50 IDs por llamada, así que procesamos en lotes.
    """
    all_videos = []

    for i in range(0, len(video_ids), 50):
        batch = video_ids[i : i + 50]

        response = youtube.videos().list(
            part="snippet,statistics,contentDetails",
            id=",".join(batch)
        ).execute()

        all_videos.extend(response["items"])
        time.sleep(0.1)

    return all_videos


def parse_duration(duration_str: str) -> int:
    """
    Convierte la duración del formato ISO 8601 (PT1M30S) a segundos (90).
    YouTube usa este formato para reportar la duración.
    """
    try:
        return int(isodate.parse_duration(duration_str).total_seconds())
    except Exception:
        return 0


def process_videos(raw_videos: list[dict]) -> pd.DataFrame:
    """
    Transforma la respuesta cruda de la API en un DataFrame limpio.
    Agrega la columna 'is_short' para filtrar Shorts (≤ 60 segundos).
    """
    rows = []

    for v in raw_videos:
        stats    = v.get("statistics", {})
        snippet  = v["snippet"]
        duration = parse_duration(v["contentDetails"]["duration"])

        # Thumbnail de mayor resolución disponible
        thumbnails = snippet.get("thumbnails", {})
        thumb_url  = (
            thumbnails.get("maxres") or
            thumbnails.get("standard") or
            thumbnails.get("high") or
            {}
        ).get("url", "")

        rows.append({
            "video_id":        v["id"],
            "title":           snippet["title"],
            "description":     snippet["description"][:500],  # primeros 500 chars
            "published_at":    snippet["publishedAt"],
            "tags":            ", ".join(snippet.get("tags", [])),
            "duration_sec":    duration,
            "is_short":        duration <= 60,
            "view_count":      int(stats.get("viewCount",    0)),
            "like_count":      int(stats.get("likeCount",    0)),
            "comment_count":   int(stats.get("commentCount", 0)),
            "thumbnail_url":   thumb_url,
            "video_url":       f"https://www.youtube.com/watch?v={v['id']}",
        })

    df = pd.DataFrame(rows)
    df["published_at"] = pd.to_datetime(df["published_at"])
    df = df.sort_values("published_at", ascending=False).reset_index(drop=True)

    return df


def get_comments(video_id: str, max_comments: int = 500) -> pd.DataFrame:
    """
    Extrae los comentarios más relevantes de un video.
    Usa order='relevance' para priorizar los más votados/respondidos.
    """
    comments        = []
    next_page_token = None

    while len(comments) < max_comments:
        try:
            response = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                order="relevance",
                pageToken=next_page_token
            ).execute()

            for item in response["items"]:
                top = item["snippet"]["topLevelComment"]["snippet"]
                comments.append({
                    "comment_id":    item["id"],
                    "author":        top["authorDisplayName"],
                    "text":          top["textDisplay"],
                    "like_count":    top["likeCount"],
                    "reply_count":   item["snippet"]["totalReplyCount"],
                    "published_at":  top["publishedAt"],
                })

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break

            time.sleep(0.1)

        except Exception as e:
            print(f"   ⚠️  Error al obtener comentarios: {e}")
            break

    df = pd.DataFrame(comments)
    if not df.empty:
        df["published_at"] = pd.to_datetime(df["published_at"])
        df = df.sort_values("like_count", ascending=False).reset_index(drop=True)

    return df


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  YouTube Channel Data Extractor")
    print("=" * 55)

    # 1. Obtener channel_id
    print("\n[1/5] Buscando canal...")
    channel_id = get_channel_id(CHANNEL_HANDLE)

    # 2. Obtener playlist de uploads
    print("\n[2/5] Obteniendo info del canal...")
    playlist_id, channel_info = get_uploads_playlist_id(channel_id)

    # 3. Obtener todos los IDs de videos
    print("\n[3/5] Recolectando IDs de videos (puede tardar un momento)...")
    video_ids = get_all_video_ids(playlist_id)
    print(f"   IDs encontrados: {len(video_ids)}")

    # 4. Obtener detalles de cada video
    print("\n[4/5] Descargando métricas de cada video...")
    raw_videos = get_video_details(video_ids)
    df_all     = process_videos(raw_videos)

    # Separar Shorts y videos largos
    df_shorts = df_all[df_all["is_short"]].copy()
    df_long   = df_all[~df_all["is_short"]].copy()

    print(f"   Shorts encontrados:       {len(df_shorts)}")
    print(f"   Videos largos encontrados: {len(df_long)}")

    # Guardar CSVs
    os.makedirs("data/raw", exist_ok=True)
    df_all.to_csv("data/raw/all_videos.csv",    index=False, encoding="utf-8-sig")
    df_shorts.to_csv("data/raw/shorts.csv",     index=False, encoding="utf-8-sig")
    df_long.to_csv("data/raw/long_videos.csv",  index=False, encoding="utf-8-sig")
    print("   ✅ Guardado en data/raw/")

    # 5. Obtener comentarios del video pregunta
    if VIDEO_PREGUNTA != "ID_DEL_VIDEO_AQUI":
        print(f"\n[5/5] Extrayendo comentarios del video ({VIDEO_PREGUNTA})...")
        df_comments = get_comments(VIDEO_PREGUNTA, max_comments=MAX_COMMENTS)
        df_comments.to_csv("data/raw/comments.csv", index=False, encoding="utf-8-sig")
        print(f"   Comentarios extraídos: {len(df_comments)}")
        print("   ✅ Guardado en data/raw/comments.csv")
    else:
        print("\n[5/5] ⚠️  VIDEO_PREGUNTA no configurado, omitiendo comentarios.")

    # Resumen final
    print("\n" + "=" * 55)
    print("  ¡Extracción completada!")
    print("=" * 55)
    print(f"  Canal:          {channel_info['channel_name']}")
    print(f"  Suscriptores:   {channel_info['total_subscribers']:,}")
    print(f"  Total videos:   {len(df_all)}")
    print(f"  Shorts:         {len(df_shorts)}")
    print(f"  Videos largos:  {len(df_long)}")
    print(f"\n  Archivos en data/raw/:")
    print(f"    - all_videos.csv")
    print(f"    - shorts.csv")
    print(f"    - long_videos.csv")
    print(f"    - comments.csv")
    print("=" * 55)


if __name__ == "__main__":
    main()