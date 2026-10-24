"""
Рекомендаційна модель BookMatch: item-based kNN (колаборативна фільтрація).

Дві книги вважаються схожими, якщо їх однаково оцінюють одні й ті самі читачі.
Подібність — скоригована косинусна міра між стовпцями матриці оцінок
(від оцінок віднято середню оцінку кожного читача). Щоб випадкові збіги
з 1–2 спільними читачами не давали високої подібності, використовуємо
«стиснення» (shrinkage): sim * n / (n + shrink), де n — кількість спільних читачів.
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.preprocessing import normalize


class BookRecommender:
    def __init__(self, model_dir: str = "model", shrink: float = 10.0):
        R = load_npz(os.path.join(model_dir, "ratings.npz")).tocsr().astype(np.float32)
        counts = np.diff(R.indptr)
        user_mean = np.asarray(R.sum(axis=1)).ravel() / np.maximum(counts, 1)

        centered = R.copy()
        centered.data -= np.repeat(user_mean, counts).astype(np.float32)
        self.X = normalize(centered.tocsc(), axis=0).tocsc()   # нормовані стовпці-книги

        binary = R.copy()
        binary.data[:] = 1.0
        self.B = binary.tocsc()                                 # хто що оцінював

        self.shrink = shrink
        self.books = pd.read_csv(os.path.join(model_dir, "books.csv.gz"))
        self.books["year"] = self.books["year"].astype("Int64")
        self.books["label"] = (
            self.books["title"] + " — " + self.books["author"]
            + self.books["year"].map(lambda y: f" ({y})" if pd.notna(y) else "")
        )
        with open(os.path.join(model_dir, "info.json"), encoding="utf-8") as f:
            self.info = json.load(f)

    # ------------------------------------------------------------------
    def similar(self, book_idxs, n=10, min_avg=0.0, exclude_same_author=False,
                year_range=None, min_co_readers=2):
        """Рекомендації на основі однієї або кількох прочитаних книг."""
        idxs = np.asarray(book_idxs, dtype=int)
        sims = (self.X[:, idxs].T @ self.X).toarray()          # |обрані| × |усі книги|
        co = (self.B[:, idxs].T @ self.B).toarray()            # спільні читачі
        sims = sims * co / (co + self.shrink)
        sims[co < min_co_readers] = 0.0
        sims = np.maximum(sims, 0.0)

        score = sims.sum(axis=0)
        score[idxs] = -np.inf                                   # не радимо вже прочитане

        b = self.books
        mask = b["avg_rating"].values >= min_avg
        if exclude_same_author:
            mask &= ~b["author"].isin(b.loc[idxs, "author"]).values
        if year_range is not None:
            y = b["year"].astype("float").values
            mask &= np.isnan(y) | ((y >= year_range[0]) & (y <= year_range[1]))
        score[~mask] = -np.inf

        order = np.argsort(-score)
        order = order[np.isfinite(score[order]) & (score[order] > 0)][:n]

        res = b.loc[order].copy()
        best_src = sims[:, order].argmax(axis=0)
        res["similarity"] = sims[best_src, order]
        res["because_of"] = b.loc[idxs[best_src], "title"].values
        res["co_readers"] = co[best_src, order].astype(int)
        return res.reset_index(drop=True)

    # ------------------------------------------------------------------
    def popular(self, n=10, min_avg=0.0, year_range=None):
        """Топ за зваженим рейтингом (формула IMDb): враховує і оцінку, і кількість оцінок."""
        b = self.books
        v, R = b["n_ratings"].astype(float), b["avg_rating"]
        m, C = v.quantile(0.75), (R * v).sum() / v.sum()
        wr = v / (v + m) * R + m / (v + m) * C
        mask = R >= min_avg
        if year_range is not None:
            y = b["year"].astype("float")
            mask &= y.isna() | y.between(*year_range)
        res = b[mask].assign(weighted=wr[mask]).sort_values("weighted", ascending=False).head(n)
        return res.reset_index(drop=True)
