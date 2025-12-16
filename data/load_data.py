import requests
import mapbox_vector_tile
import pandas as pd
import geopandas as gpd
# import json
from shapely.geometry import Point
import math
from tqdm import tqdm
from astral import LocationInfo
from astral.sun import sun
from datetime import timezone

ACCESS_TOKEN = "MLY|25484984784501735|4633da3c1fba24964c4a1ce031b9e238"


def fetch_overview_tile(z, x, y):
    url = f"https://tiles.mapillary.com/maps/vtp/mly1_public/2/{z}/{x}/{y}" \
          f"?access_token={ACCESS_TOKEN}"
    r = requests.get(url)
    r.raise_for_status()
    return mapbox_vector_tile.decode(r.content)


def mvt_to_lonlat(tx, ty, z, x, y, extent=4096):
    # convert tile-local coords to world pixel coords
    world_x = x * extent + tx
    world_y = y * extent + (extent - ty)

    # convert to lon/lat
    n = 2 ** z
    lon = world_x / (extent * n) * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi - 2 * math.pi * world_y /
                                  (extent * n)))
    lat = math.degrees(lat_rad)

    return lon, lat


def tile_to_records(tile, z, x, y):
    if "overview" not in tile:
        return []

    records = []
    for feat in tile["overview"]["features"]:
        props = feat["properties"]
        coords = feat["geometry"]["coordinates"]
        lon, lat = mvt_to_lonlat(coords[0], coords[1], z, x, y, 4096)

        records.append({
            "lon": lon,
            "lat": lat,
            "captured_at": props.get("captured_at"),
            "creator_id": props.get("creator_id"),
            "image_id": props.get("id"),
            "sequence_id": props.get("sequence_id"),
            "is_pano": props.get("is_pano"),
            "tile_z": z,
            "tile_x": x,
            "tyle_y": y,
        })

    return records


def is_daytime(lon, lat, captured_at):
    location = LocationInfo(latitude=lat, longitude=lon)
    try:
        s = sun(location.observer, date=captured_at.date())
        sunrise_utc = s['sunrise'].astimezone(timezone.utc)
        sunset_utc = s['sunset'].astimezone(timezone.utc)
        return sunrise_utc <= captured_at <= sunset_utc
    except ValueError:
        # Sun never rises or never sets:
        #  treat all hours as daytime or nighttime
        # False: skip polar night
        # True: keep polar day
        # I've set it to false for now
        return False


def main():
    # europe is approximately contained within these tiles
    europe_tiles = [
        (4, 7, 5), (4, 8, 5), (4, 9, 5),
        (4, 7, 4), (4, 8, 4), (4, 9, 4),
        (4, 8, 3), (4, 9, 3), (5, 17, 12),
        (5, 15, 12)
    ]

    all_records = []

    for z, x, y in europe_tiles:
        print(f"Fetching tile z={z}, x={x}, y={y}")
        tile = fetch_overview_tile(z, x, y)
        recs = tile_to_records(tile, z, x, y)
        print(f"{len(recs)} images")
        all_records.extend(recs)

    df = pd.DataFrame(all_records)
    df["captured_at"] = pd.to_datetime(df["captured_at"], unit='ms', utc=True)
    print("Total images:", len(df))

    europe = gpd.read_file("data/europe_polygon/europe.geojson")
    dp = df.iloc[2]
    pt = Point(dp['lon'], dp['lat'])
    inside = europe[europe.contains(pt)]

    raw_data = []

    # Obtain the first x datapoints
    x = 1000000
    for idx, dp in tqdm(df.iloc[:x].iterrows(), total=min(x, len(df))):
        pt = Point(dp['lon'], dp['lat'])

        inside = europe[europe.contains(pt)]

        if not inside.empty and not dp["is_pano"]:
            if is_daytime(dp['lon'], dp['lat'], dp['captured_at']):
                dp = dp.copy()  # create a separate copy before changing values
                dp["country"] = inside.iloc[0]["NAME"]
                raw_data.append(dp)

    df = pd.DataFrame(raw_data)

    df["country"] = df["country"].astype("string")
    df["sequence_id"] = df["sequence_id"].astype("string")
    df["captured_at"] = pd.to_datetime(df["captured_at"], unit='ms', utc=True)

    print(df.head())

    print("\nDatapoint counts per country:")
    print(df["country"].value_counts().sort_values(ascending=False))

    # save df
    df.to_parquet("raw_data.parquet", engine="fastparquet", index=False)


if __name__ == "__main__":
    main()
