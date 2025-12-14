import pandas as pd
import requests
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt

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

def main():
    split_data = "split_data.parquet"
    df = pd.read_parquet(split_data, engine="fastparquet")
    
    random_image(df)

if __name__ == "__main__":
    main()