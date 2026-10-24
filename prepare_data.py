"""
Готує компактні файли моделі BookMatch із сирих CSV датасету Book-Crossing.

Запуск:
    python prepare_data.py              # CSV лежать у папці data/
    python prepare_data.py --data шлях  # або вкажіть іншу папку

Результат (папка model/):
    ratings.npz     — розріджена матриця «користувач × книга» з явними оцінками 1–10
    books.csv.gz    — каталог книг: назва, автор, рік, обкладинка, середня оцінка
    info.json       — параметри та статистика підготовки
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz


def read_csv(path):
    try:
        return pd.read_csv(path, low_memory=False, dtype={"ISBN": str})
    except UnicodeDecodeError:
        return pd.read_csv(path, low_memory=False, dtype={"ISBN": str}, encoding="latin-1")


def norm_text(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", help="папка з Books.csv і Ratings.csv")
    ap.add_argument("--out", default="model", help="куди зберегти модель")
    ap.add_argument("--min-book", type=int, default=10, help="мін. оцінок у книги")
    ap.add_argument("--min-user", type=int, default=5, help="мін. оцінок у користувача")
    args = ap.parse_args()

    books = read_csv(os.path.join(args.data, "Books.csv"))
    ratings = read_csv(os.path.join(args.data, "Ratings.csv"))
    print(f"Завантажено: {len(books):,} книг, {len(ratings):,} оцінок")

    # --- Очищення книг ---
    books["ISBN"] = books["ISBN"].astype(str).str.strip().str.upper()
    books = books.drop_duplicates("ISBN")
    books["Year-Of-Publication"] = pd.to_numeric(books["Year-Of-Publication"], errors="coerce")
    bad = (books["Year-Of-Publication"] < 1800) | (books["Year-Of-Publication"] > 2006)
    books.loc[bad, "Year-Of-Publication"] = np.nan
    books["Book-Author"] = books["Book-Author"].fillna("Unknown")

    # Різні видання однієї книги (різні ISBN) об'єднуємо в один «твір»: назва + автор
    books["work"] = norm_text(books["Book-Title"]) + " | " + norm_text(books["Book-Author"])

    # --- Явні оцінки для книг з каталогу ---
    ratings["ISBN"] = ratings["ISBN"].astype(str).str.strip().str.upper()
    r = ratings[ratings["Book-Rating"] > 0].merge(books[["ISBN", "work"]], on="ISBN")
    isbn_popularity = r["ISBN"].value_counts()
    # якщо користувач оцінив кілька видань — беремо середнє
    r = r.groupby(["User-ID", "work"], as_index=False)["Book-Rating"].mean()

    # --- Фільтр розрідженості ---
    cnt = r["work"].value_counts()
    r = r[r["work"].isin(cnt[cnt >= args.min_book].index)]
    cnt = r["User-ID"].value_counts()
    r = r[r["User-ID"].isin(cnt[cnt >= args.min_user].index)]
    if r.empty:
        raise SystemExit("Після фільтрації нічого не лишилось — зменшіть --min-book / --min-user")

    works = np.sort(r["work"].unique())
    users = np.sort(r["User-ID"].unique())
    w_idx = pd.Series(np.arange(len(works)), index=works)
    u_idx = pd.Series(np.arange(len(users)), index=users)

    R = csr_matrix(
        (r["Book-Rating"].astype(np.float32), (u_idx[r["User-ID"]].values, w_idx[r["work"]].values)),
        shape=(len(users), len(works)),
    )

    # --- Каталог: для кожного твору беремо найпопулярніше видання (назва, рік, обкладинка) ---
    b = books[books["work"].isin(works)].copy()
    b["pop"] = b["ISBN"].map(isbn_popularity).fillna(0)
    b = b.sort_values(["work", "pop"], ascending=[True, False])
    first_year = b.groupby("work")["Year-Of-Publication"].min()
    b = b.drop_duplicates("work").set_index("work").loc[works]

    stats = r.groupby("work")["Book-Rating"].agg(["mean", "size"]).loc[works]
    catalog = pd.DataFrame({
        "idx": np.arange(len(works)),
        "isbn": b["ISBN"].values,
        "title": b["Book-Title"].astype(str).str.strip().values,
        "author": b["Book-Author"].astype(str).str.strip().values,
        "year": first_year.loc[works].values,
        "publisher": b["Publisher"].fillna("").astype(str).values,
        "image": b["Image-URL-L"].fillna("").astype(str).str.replace("http://", "https://", n=1).values,
        "avg_rating": stats["mean"].round(2).values,
        "n_ratings": stats["size"].values,
    })

    os.makedirs(args.out, exist_ok=True)
    save_npz(os.path.join(args.out, "ratings.npz"), R, compressed=True)
    catalog.to_csv(os.path.join(args.out, "books.csv.gz"), index=False, compression="gzip")
    info = {
        "n_ratings": int(R.nnz), "n_users": int(R.shape[0]), "n_books": int(R.shape[1]),
        "min_book": args.min_book, "min_user": args.min_user,
    }
    with open(os.path.join(args.out, "info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"Готово: {info['n_ratings']:,} оцінок, {info['n_users']:,} читачів, {info['n_books']:,} книг → {args.out}/")


if __name__ == "__main__":
    main()
