"""
BookMatch — вебінтерфейс рекомендацій книг.
Запуск:  streamlit run app.py
"""
import html
import os

import streamlit as st

from recommender import BookRecommender

st.set_page_config(page_title="BookMatch — що почитати далі", page_icon="📚", layout="wide")

MODEL_DIR = "model"

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.bm-hero h1 {font-size: 2.3rem; margin-bottom: .2rem;}
.bm-hero p {color: #6b6b6b; font-size: 1.05rem; margin-top: 0;}
.bm-card {border: 1px solid rgba(128,128,128,.25); border-radius: 12px; overflow: hidden;
          height: 100%; display: flex; flex-direction: column; background: rgba(128,128,128,.04);}
.bm-cover {aspect-ratio: 2 / 3; background-size: contain, cover; background-repeat: no-repeat;
           background-position: center;}
.bm-body {padding: .6rem .7rem .8rem; display: flex; flex-direction: column; gap: .25rem;}
.bm-title {font-weight: 600; line-height: 1.25; font-size: .95rem;}
.bm-author {color: #7a7a7a; font-size: .82rem;}
.bm-meta {font-size: .82rem;}
.bm-match {display: inline-block; align-self: flex-start; background: #2f6f4f; color: #fff;
           border-radius: 999px; padding: .05rem .55rem; font-size: .75rem; font-weight: 600;}
.bm-why {color: #7a7a7a; font-size: .76rem; line-height: 1.3;}
</style>
""", unsafe_allow_html=True)

COVER_FALLBACK = "linear-gradient(135deg, #c9b38f 0%, #7d5f3f 100%)"


def card(row, show_match=True):
    """HTML-картка книги. Якщо обкладинка не завантажилась — видно кольоровий фон."""
    img = html.escape(str(row.get("image", "") or ""))
    title = html.escape(str(row["title"]))
    author = html.escape(str(row["author"]))
    year = f" · {int(row['year'])}" if str(row["year"]) not in ("<NA>", "nan") else ""
    parts = [
        f'<div class="bm-card"><div class="bm-cover" style="background-image:url(\'{img}\'), {COVER_FALLBACK}"></div>',
        '<div class="bm-body">',
        f'<div class="bm-title">{title}</div>',
        f'<div class="bm-author">{author}{year}</div>',
        f'<div class="bm-meta">★ {row["avg_rating"]:.1f} / 10 · {int(row["n_ratings"])} оцінок</div>',
    ]
    if show_match and "similarity" in row:
        parts.append(f'<span class="bm-match">Схожість {row["similarity"] * 100:.0f}%</span>')
        parts.append(
            f'<div class="bm-why">Схожа на «{html.escape(str(row["because_of"]))}»: '
            f'{int(row["co_readers"])} спільних читачів</div>'
        )
    parts.append("</div></div>")
    return "".join(parts)


def grid(df, per_row=5, show_match=True):
    for start in range(0, len(df), per_row):
        cols = st.columns(per_row)
        for col, (_, row) in zip(cols, df.iloc[start:start + per_row].iterrows()):
            col.markdown(card(row, show_match), unsafe_allow_html=True)
        st.write("")


@st.cache_resource(show_spinner="Завантажую модель…")
def load_model():
    return BookRecommender(MODEL_DIR)


# ---------------------------------------------------------------------------
if not os.path.exists(os.path.join(MODEL_DIR, "ratings.npz")):
    st.error("Модель не знайдено. Спершу підготуйте дані: покладіть CSV у папку `data/` "
             "і виконайте `python prepare_data.py`.")
    st.stop()

model = load_model()
books = model.books

st.markdown(
    '<div class="bm-hero"><h1>BookMatch</h1>'
    '<p>Оберіть книгу, яка вам сподобалась, — і ми підберемо, що почитати далі.</p></div>',
    unsafe_allow_html=True,
)

# --- Бічна панель: фільтри ---
with st.sidebar:
    st.header("Налаштування")
    n_recs = st.slider("Скільки книг порадити", 5, 20, 10, step=5)
    min_avg = st.slider("Мінімальна середня оцінка", 0.0, 10.0, 0.0, step=0.5)
    years = books["year"].dropna().astype(int)
    y_min, y_max = int(years.min()), int(years.max())
    year_range = st.slider("Рік видання", y_min, y_max, (y_min, y_max))
    exclude_author = st.checkbox("Без книг тих самих авторів", value=False,
                                 help="Корисно, щоб відкрити для себе нових авторів")
    st.divider()
    st.caption(f"База: {model.info['n_books']:,} книг · {model.info['n_users']:,} читачів · "
               f"{model.info['n_ratings']:,} оцінок (Book-Crossing)")

tab_rec, tab_pop, tab_about = st.tabs(["Підібрати книгу", "Популярне", "Як це працює"])

# --- Вкладка 1: рекомендації за прочитаними книгами ---
with tab_rec:
    order = books.sort_values("n_ratings", ascending=False)
    chosen_labels = st.multiselect(
        "Які книги ви прочитали і вони вам сподобались? (почніть вводити назву)",
        options=order["label"].tolist(),
        max_selections=5,
        placeholder="Наприклад: Harry Potter, The Da Vinci Code, The Hobbit…",
    )

    if not chosen_labels:
        st.info("Оберіть одну або кілька книг — рекомендації з'являться тут.")
        st.subheader("Не знаєте, з чого почати? Найпопулярніші книги в базі:")
        grid(model.popular(n=5), show_match=False)
    else:
        label_to_idx = dict(zip(books["label"], books["idx"]))
        chosen = [label_to_idx[l] for l in chosen_labels]

        with st.expander("Ваші книги", expanded=False):
            grid(books.loc[chosen], show_match=False)

        recs = model.similar(chosen, n=n_recs, min_avg=min_avg,
                             exclude_same_author=exclude_author, year_range=year_range)
        if recs.empty:
            st.warning("Для цих книг замало спільних читачів або фільтри надто суворі. "
                       "Спробуйте послабити фільтри — а поки ось популярні книги:")
            grid(model.popular(n=n_recs, min_avg=min_avg, year_range=year_range), show_match=False)
        else:
            st.subheader(f"Вам можуть сподобатись ({len(recs)})")
            grid(recs)

# --- Вкладка 2: популярне ---
with tab_pop:
    st.subheader("Найкращі книги за зваженим рейтингом")
    st.caption("Рейтинг враховує і середню оцінку, і кількість оцінок, тож книга з однією «десяткою» не обійде визнану класику.")
    grid(model.popular(n=n_recs, min_avg=min_avg, year_range=year_range), show_match=False)

# --- Вкладка 3: пояснення ---
with tab_about:
    st.markdown(f"""
### Як BookMatch підбирає книги
**Ідея:** якщо читачі, яким сподобалась книга A, так само високо оцінили книгу B — книги схожі.
Це **колаборативна фільтрація** (item-based kNN): ми не аналізуємо текст книг, а вчимося на
поведінці {model.info['n_users']:,} реальних читачів.

**Кроки алгоритму**
1. Будуємо матрицю «читач × книга» з явними оцінками 1–10.
2. Від кожної оцінки віднімаємо середню оцінку читача (хтось ставить усім 9, а хтось — 6).
3. Рахуємо подібність між книгами і зменшуємо її, якщо спільних читачів мало.
4. Для обраних вами книг беремо найсхожіші, підсумовуємо подібність і показуємо топ.

**Обмеження.** Дані зібрано у 2004 році, тому новинок немає. Книги з малою кількістю оцінок
відфільтровано, тож рідкісних видань у пошуку може не бути.

**Дані:** [Book-Crossing / Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset).
""")
