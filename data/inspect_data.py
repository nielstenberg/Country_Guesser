import pandas as pd
import requests
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import math
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from astral import LocationInfo
from astral.sun import sun
from datetime import timezone
from tqdm import tqdm
import asyncio
import aiohttp
from tqdm.asyncio import tqdm_asyncio


def random_image(df, seed=None, save_path="random_image.jpg"):
    if df.empty:
        raise ValueError("Dataframe is empty.")

    row = df.sample(n=1, random_state=seed).iloc[0]

    print(f"Split    : {row['split']}")
    print(f"Country  : {row['country']}")
    print(f"Image ID : {row['image_id']}")
    print(f"URL      : {row['img_url']}")
    print(f"Location : Lon: {row['lon']:.6f}, Lat: {row['lat']:.6f}")

    resp = requests.get(row["img_url"])
    resp.raise_for_status()
    img = Image.open(BytesIO(resp.content)).convert("RGB")
    img.save(save_path)

    # Sunrise/Sunset calculation to display above the image
    # This is not removing any images anymore, we already did that in load_data
    location = LocationInfo(latitude=row['lat'], longitude=row['lon'])
    s = sun(location.observer, date=row['captured_at'].date())
    sunrise_utc = s['sunrise'].astimezone(timezone.utc)
    sunset_utc = s['sunset'].astimezone(timezone.utc)

    # Formatting for displaying the correct day/time notation
    captured_str = row['captured_at'].strftime("%m/%d/%y %H:%M:%S UTC")
    sunrise_str = sunrise_utc.strftime("%H:%M:%S UTC")
    sunset_str = sunset_utc.strftime("%H:%M:%S UTC")
    lon_lat_str = f"LON: {row['lon']:.6f}, LAT: {row['lat']:.6f}"

    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"{row['country']} — {row['split']}", fontsize=14)
    plt.suptitle(
        "Captured at: " + captured_str + "\n"
        "Sunrise at this day: " + sunrise_str + "\n"
        "Sunset at this day: " + sunset_str + "\n"
        + lon_lat_str,
        fontsize=10
    )
    plt.show()

    return row


def n_random_images(df, n, session, seed=None):
    rows = df.sample(n=n, random_state=seed)

    cols = min(n, 4)                 # up to 4 images per row
    rows_grid = math.ceil(n / cols)

    plt.figure(figsize=(4 * cols, 4 * rows_grid))

    for i, (_, row) in enumerate(rows.iterrows(), start=1):
        print(f"[{i}] Split   : {row['split']}")
        print(f"    Country : {row['country']}")
        print(f"    Image ID: {row['image_id']}\n")

        resp = session.get(row["img_url"])
        resp.raise_for_status()

        img = Image.open(BytesIO(resp.content)).convert("RGB")

        ax = plt.subplot(rows_grid, cols, i)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"{row['country']} — {i}")

    plt.tight_layout()
    plt.show()

    return rows


def download_image(img_id, url, output_dir):
    filename = f"{img_id}.jpg"
    filepath = os.path.join(output_dir, filename)
    r = requests.get(url)
    r.raise_for_status()
    with open(filepath, "wb") as file:
        file.write(r.content)


def download_country_images(df, country, output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir)

    rows = df.loc[df["country"] == country, ["image_id", "img_url"]].dropna()

    print(f"Saving images to: {os.path.abspath(output_dir)}")
    print(f"Found {len(rows)} images for {country}")

    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(
            lambda row: download_image(row.image_id, row.img_url, output_dir),
            rows.itertuples(index=False)
        )

async def check_image_fast(session, sem, image_id, url):
    async with sem:
        try:
            async with session.get(url, allow_redirects=True) as r:
                if r.status >= 400:
                    return image_id

                # ensure body is readable
                await r.content.read(1024)
                return None

        except Exception:
            return image_id


async def find_bad_image_ids(df, max_checks=None, concurrency=50):
    rows = list(df[["image_id", "img_url"]].itertuples(index=False))
    if max_checks:
        rows = rows[:max_checks]

    sem = asyncio.Semaphore(concurrency)

    timeout = aiohttp.ClientTimeout(
        total=None,
        connect=10,
        sock_connect=10,
        sock_read=10,
    )

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }

    connector = aiohttp.TCPConnector(
        limit=concurrency,
        limit_per_host=10,
        ttl_dns_cache=300,
    )

    bad_ids = []

    async with aiohttp.ClientSession(
        timeout=timeout,
        headers=headers,
        connector=connector,
    ) as session:

        tasks = [
            asyncio.create_task(
                check_image_fast(session, sem, row.image_id, row.img_url)
            )
            for row in rows
        ]

        for task in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Checking images",
        ):
            result = await task
            if result is not None:
                bad_ids.append(result)

    return bad_ids

def delete_image_id(parquet_path, image_id, output_path=None):
    if output_path is None:
        output_path = parquet_path

    df = pd.read_parquet(parquet_path, engine="fastparquet")

    before = len(df)
    df = df[df["image_id"] != image_id]
    after = len(df)

    if before == after:
        print(f"image_id {image_id} not found")
    else:
        print(f"removed image_id {image_id}")
        print(f"rows before: {before}")
        print(f"rows after : {after}")

    df.to_parquet(output_path, engine="fastparquet", index=False)
    print(f"saved to {output_path}")

    return df

def image_distribution(df):
    return df["country"].value_counts().sort_values(ascending=False)

def main():
    split_data = "datasets/split_data_1000.parquet"
    df = pd.read_parquet(split_data, engine="fastparquet")
    
    # raw_data = "datasets/raw_data.parquet"
    # df = pd.read_parquet(raw_data, engine="fastparquet")
    
    distribution = image_distribution(df[df["split"] == "test"])
    print(distribution)
    print(len(df))

    # random_image(df)

    # session = requests.Session()
    # n_random_images(df, n=8, session=session)

    # download_country_images(df, "Croatia", "country_images")
    
    # bad_image_ids = asyncio.run(find_bad_image_ids(df))
    
    # print(bad_image_ids)
    # print(len(bad_image_ids))

    # delete_image_id(
    #     parquet_path="datasets/split_data.parquet",
    #     image_id=759983608038195,
    #     output_path="datasets/split_data.parquet",
    # )
    pass

if __name__ == "__main__":
    main()