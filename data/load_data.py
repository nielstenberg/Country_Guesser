import requests
import mapbox_vector_tile
import pandas as pd
import geopandas as gpd
import json
from shapely.geometry import Point
import math

ACCESS_TOKEN = "MLY|25484984784501735|4633da3c1fba24964c4a1ce031b9e238"

def fetch_overview_tile(z, x, y):
    url = f"https://tiles.mapillary.com/maps/vtp/mly1_public/2/{z}/{x}/{y}?access_token={ACCESS_TOKEN}"
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
    lat_rad = math.atan(math.sinh(math.pi - 2 * math.pi * world_y / (extent * n)))
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
            "tile": (z, x, y),
        })

    return records

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
        print(f" → {len(recs)} images")
        all_records.extend(recs)

    df = pd.DataFrame(all_records)
    print("Total images:", len(df))
    
    europe = gpd.read_file("europe_polygon/europe.geojson")
    dp = df.iloc[2]
    pt = Point(dp['lon'], dp['lat'])
    inside = europe[europe.contains(pt)]
    
    if len(inside) > 0:
        country = inside.iloc[0]["NAME"]
        print("Inside:", country)
    else:
        print("Point is outside Europe")
        
    country_counts = {}

    # Loop through the first x datapoints
    x = 100000
    for idx, dp in df.iloc[:x].iterrows():
        pt = Point(dp['lon'], dp['lat'])
        
        inside = europe[europe.contains(pt)]
        
        if len(inside) > 0:
            country = inside.iloc[0]["NAME"]
            country_counts[country] = country_counts.get(country, 0) + 1
        else:
            country_counts["OUTSIDE_EUROPE"] = country_counts.get("OUTSIDE_EUROPE", 0) + 1

    country_counts = pd.Series(country_counts).sort_values(ascending=False)

    print(country_counts)

    # save df
    # df.to_csv("mapillary_europe_overview.csv", index=False)

    print(df.head())

if __name__ == "__main__":
    main()
