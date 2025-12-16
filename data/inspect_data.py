import pandas as pd
import requests
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import math
import os
import shutil
from concurrent.futures import ThreadPoolExecutor

def random_image(df, seed=None, save_path="random_image.jpg"):
    row = df.sample(n=1, random_state=seed).iloc[0]
    
    print(f"Split   : {row['split']}")
    print(f"Country : {row['country']}")
    print(f"Image ID: {row['image_id']}")
    print(f"URL     : {row['img_url']}")
    
    resp = requests.get(row["img_url"])
    resp.raise_for_status()

    img = Image.open(BytesIO(resp.content)).convert("RGB")
    
    img.save(save_path)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"{row['country']} — {row['split']} data")
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

def main():
    split_data = "split_data.parquet"
    df = pd.read_parquet(split_data, engine="fastparquet")
    
    # random_image(df)
    
    # session = requests.Session()
    # n_random_images(df, n=8, session=session)
    
    download_country_images(df, "United Kingdom", "country_images")

if __name__ == "__main__":
    main()