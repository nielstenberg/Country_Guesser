import pandas as pd
from tqdm import tqdm
import requests
# import re
import time

ACCESS_TOKEN = "MLY|25484984784501735|4633da3c1fba24964c4a1ce031b9e238"


def fetch_with_retry(url, session, max_retries=3, backoff=1.0):
    for attempt in range(max_retries):
        r = session.get(url)

        if r.status_code == 200:
            return r

        # retry for server errors
        if 500 <= r.status_code < 600:
            time.sleep(backoff * (attempt + 1))
            continue

        r.raise_for_status()

    # all retries failed
    raise RuntimeError(f"Failed after {max_retries} retries: {url}")


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def obtain_image_urls(df, country, batch_size=300):
    # add image url to datapoints
    df = df.copy()

    session = requests.Session()

    image_ids = df["image_id"].astype(str).tolist()
    id_to_url = {}

    for batch in chunks(image_ids, batch_size):

        ids = ",".join(batch)

        url = (
            "https://graph.mapillary.com/images"
            f"?image_ids={ids}"
            f"&fields=id,thumb_1024_url"
            f"&access_token={ACCESS_TOKEN}"
        )

        r = fetch_with_retry(url, session)

        data = r.json()["data"]

        for item in data:
            if "thumb_1024_url" not in item:
                message = f"""Missing URL for image {item['id']} for
                country of {country}"""
                tqdm.write(message)
            id_to_url[item["id"]] = item.get("thumb_1024_url")

    df["img_url"] = df["image_id"].astype(str).map(id_to_url)

    return df


def n_per_country(df, n, test_size, val_size, seed):
    parts = []

    for country, country_df in tqdm(df.groupby("country"),
                                    total=df["country"].nunique()):
        # shuffle per country, for random sampling
        country_df = country_df.sample(
            frac=1, random_state=seed).reset_index(drop=True)
        country_size = min(len(country_df), n)
        country_df = country_df.iloc[:country_size]
        country_df = obtain_image_urls(country_df, country)
        country_df = country_df.dropna(
            subset=["img_url"]).reset_index(drop=True)

        n_test = int(test_size * country_size)
        n_val = int(val_size * country_size)

        country_df.loc[:n_test - 1, "split"] = "test"
        country_df.loc[n_test: n_test + n_val - 1, "split"] = "val"
        country_df.loc[n_test + n_val:, "split"] = "train"

        parts.append(country_df)

    # shuffle all one more time, for non-label-correlated ordering
    df_out = pd.concat(parts).sample(
        frac=1, random_state=seed).reset_index(drop=True)

    return df_out


def main():
    raw_data = "raw_data.parquet"
    df = pd.read_parquet(raw_data, engine="fastparquet")

    seed = 42

    test_size = 0.1
    val_size = 0.1

    split_data = n_per_country(
        df,
        n=300,
        test_size=test_size,
        val_size=val_size,
        seed=seed,
    )

    # save split data
    split_data.to_parquet("split_data.parquet",
                          engine="fastparquet", index=False)


if __name__ == "__main__":
    main()
