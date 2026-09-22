#Data Pipeline Module
# Data Pipeline Module

This module scrapes book data from [books.toscrape.com](https://books.toscrape.com/), cleans it, stores it in a SQLite database, and runs SQL queries against it — demonstrating a full raw-data-to-queryable-database pipeline.

## What it does

1. **Scrape** — Collects book data (title, price, star rating, availability) from 3 categories: Travel, Mystery, and Fiction, across all paginated pages, using `requests` + `BeautifulSoup`. No manual copy-pasting.
2. **Clean** — Converts raw scraped text into proper types:
   - Price string (e.g. `£25.00`) → float `price_gbp`
   - Star rating word (e.g. `"Four"`) → integer `rating` (1–5)
   - Availability text → boolean `in_stock`
3. **Convert currency** — Adds a `price_inr` column, computed using a **fixed conversion rate of 1 GBP = 105.50 INR** (a fixed project-defined constant, not looked up live, as required).
4. **Store** — Loads the cleaned data into a SQLite database (`books.db`) with two related tables:
   - `categories` (category_id PK, category_name)
   - `books` (book_id PK, title, category_id FK, price_gbp, price_inr, rating, in_stock)
5. **Query** — Runs 5 SQL queries demonstrating `SELECT`, `WHERE`, `ORDER BY`, `LIMIT`, `DISTINCT`, `BETWEEN`, and a `JOIN` between the two tables.
6. **Verify** — Reads the JOIN query result back into pandas via `pd.read_sql`, and separately reproduces the same result using `pd.merge` on the two tables loaded independently — proving both approaches match.

## How to run it

**Requirements:** Python 3.10+, internet connection (for scraping)

1. Create and activate a virtual environment:
