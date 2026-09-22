# ============================================================
# MODULE 1: DATA PIPELINE
# Scrape books.toscrape.com -> clean -> store in SQLite -> query
# Run this in Google Colab, one section at a time (or all at once)
# ============================================================

import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
import re

# ------------------------------------------------------------
# STEP 1: SCRAPE
# ------------------------------------------------------------
BASE_URL = "https://books.toscrape.com/"
CATEGORY_URLS = {
    "Travel": "catalogue/category/books/travel_2/index.html",
    "Mystery": "catalogue/category/books/mystery_3/index.html",
    "Fiction": "catalogue/category/books/fiction_10/index.html",
}

RATING_WORDS = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}

def scrape_category(category_name, relative_url):
    books = []
    url = BASE_URL + relative_url
    while url:
        resp = requests.get(url)
        soup = BeautifulSoup(resp.text, "html.parser")
        articles = soup.find_all("article", class_="product_pod")

        for article in articles:
            title = article.h3.a["title"]
            price_raw = article.find("p", class_="price_color").text
            rating_class = article.p["class"][1]  # e.g. "Four"
            availability_raw = article.find("p", class_="instock availability").text.strip()

            books.append({
                "title": title,
                "price_raw": price_raw,
                "rating_word": rating_class,
                "availability_raw": availability_raw,
                "category": category_name,
            })

        # handle pagination
        next_btn = soup.find("li", class_="next")
        if next_btn:
            next_href = next_btn.a["href"]
            url = url.rsplit("/", 1)[0] + "/" + next_href
        else:
            url = None
    return books


all_books = []
for cat_name, cat_url in CATEGORY_URLS.items():
    all_books.extend(scrape_category(cat_name, cat_url))

df_raw = pd.DataFrame(all_books)
print(f"Scraped {len(df_raw)} books across {df_raw['category'].nunique()} categories")

# ------------------------------------------------------------
# STEP 2: CLEAN
# ------------------------------------------------------------
FIXED_GBP_TO_INR_RATE = 105.50  # fixed project-defined rate, stated in README

def clean_price(price_raw):
    # e.g. "£25.00" -> 25.00
    return float(re.sub(r"[^\d.]", "", price_raw))

df = df_raw.copy()
df["price_gbp"] = df["price_raw"].apply(clean_price)
df["price_inr"] = (df["price_gbp"] * FIXED_GBP_TO_INR_RATE).round(2)
df["rating"] = df["rating_word"].map(RATING_WORDS).astype(int)
df["in_stock"] = df["availability_raw"].str.contains("In stock")

df = df[["title", "category", "price_gbp", "price_inr", "rating", "in_stock"]]
print(df.head())
print(f"\nFinal cleaned dataset: {len(df)} rows")

# ------------------------------------------------------------
# STEP 3: BUILD SQLITE DATABASE (two related tables, PK/FK)
# ------------------------------------------------------------
conn = sqlite3.connect("books.db")
cursor = conn.cursor()

cursor.executescript("""
DROP TABLE IF EXISTS books;
DROP TABLE IF EXISTS categories;

CREATE TABLE categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT UNIQUE NOT NULL
);

CREATE TABLE books (
    book_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    category_id INTEGER,
    price_gbp REAL,
    price_inr REAL,
    rating INTEGER,
    in_stock BOOLEAN,
    FOREIGN KEY (category_id) REFERENCES categories(category_id)
);
""")

# insert categories
for cat in df["category"].unique():
    cursor.execute("INSERT INTO categories (category_name) VALUES (?)", (cat,))
conn.commit()

# map category name -> id
cat_map = pd.read_sql("SELECT * FROM categories", conn).set_index("category_name")["category_id"].to_dict()

# insert books
for _, row in df.iterrows():
    cursor.execute("""
        INSERT INTO books (title, category_id, price_gbp, price_inr, rating, in_stock)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (row["title"], cat_map[row["category"]], row["price_gbp"], row["price_inr"], row["rating"], row["in_stock"]))
conn.commit()

print("Database built: books.db")

# ------------------------------------------------------------
# STEP 4: SQL QUERIES (5+ required, with output)
# ------------------------------------------------------------

queries = {
    "Q1 - All books, cheapest first": """
        SELECT title, price_gbp FROM books ORDER BY price_gbp ASC LIMIT 10;
    """,
    "Q2 - Books priced between 10 and 30 GBP": """
        SELECT title, price_gbp FROM books WHERE price_gbp BETWEEN 10 AND 30;
    """,
    "Q3 - Distinct categories": """
        SELECT DISTINCT category_name FROM categories;
    """,
    "Q4 - Books with rating 5, in stock only": """
        SELECT title, rating FROM books WHERE rating = 5 AND in_stock = 1;
    """,
    "Q5 - JOIN: books with their category name": """
        SELECT books.title, categories.category_name, books.price_gbp
        FROM books
        JOIN categories ON books.category_id = categories.category_id
        LIMIT 10;
    """,
}

for label, query in queries.items():
    print(f"\n--- {label} ---")
    result = pd.read_sql(query, conn)
    print(result)

# ------------------------------------------------------------
# STEP 5: pd.read_sql vs pd.merge (must produce equivalent output)
# ------------------------------------------------------------
books_df = pd.read_sql("SELECT * FROM books", conn)
categories_df = pd.read_sql("SELECT * FROM categories", conn)

# Approach A: SQL JOIN read directly
sql_join_result = pd.read_sql("""
    SELECT books.title, categories.category_name, books.price_gbp
    FROM books JOIN categories ON books.category_id = categories.category_id
""", conn)

# Approach B: in-memory pandas merge (no SQL)
pandas_merge_result = books_df.merge(categories_df, on="category_id")[["title", "category_name", "price_gbp"]]

print("\n--- SQL JOIN result (first 5) ---")
print(sql_join_result.head())
print("\n--- pandas merge result (first 5) ---")
print(pandas_merge_result.head())

conn.close()