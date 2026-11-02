# AI Shopping Assistant

A chatbot for an online store. You type what you want in normal words, for example
*"black running shoes under ₹3,000"*, and it shows you matching products from a catalog
of **1,000 products**. It can also compare products and add them to a cart.

It is built with **RAG** (Retrieval-Augmented Generation):

1. **Retrieve**: find the right products in the catalog.
2. **Augment**: give those products to an AI model.
3. **Generate**: the AI writes a short, friendly reply about them.

**Main rule of this project:** the search decides which products are shown. The AI only
describes them, and the app checks that the AI did not invent any product, price or other detail.

---

## What you can do

| Try typing | What happens |
|---|---|
| `show me boots` | Shows all boots in the catalog |
| `cloth under 200` | Shows only clothing that costs less than ₹200 |
| `comfortable shoes for jogging` | Understands the meaning and shows running shoes |
| `laptops between 30k and 50k` | Shows laptops priced from ₹30,000 to ₹50,000 |
| `black or blue shirts under 1000 sort by price low to high` | Several filters together, plus sorting |
| `KitchenCraft Urban Storage` | Shows only that exact product |
| `I want to buy this product` | Asks for size and colour, then adds it to the cart |
| `compare the top 3` | Shows a comparison table |
| `pen` | Says *"we don't currently carry pen"* (the store doesn't sell pens) |

---

## How to run it

You need **Python 3.11**.

```powershell
# 1. Go to the project folder
cd vish_task

# 2. Create and turn on a virtual environment
py -3.11 -m venv .venv
.venv\Scripts\activate

# 3. Install the libraries
pip install -r requirements.txt

# 4. Start the app
python -m streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

**First start takes about 1 minute.** The app downloads the embedding model and builds the
search index. After that, it starts in a few seconds.

### AI replies (optional)

The app works **without any API key**: search, filters, product cards and the cart all work,
and the replies come from a simple template.

To get AI-written replies, create a file named `.env` in the project folder with one or both keys:

```
GROQ_API_KEY=your_groq_key
OPENAI_API_KEY=your_openai_key
```

Or paste a key in the sidebar while the app is running.

- **Free mode** uses Groq.
- **Paid mode** uses OpenAI.

Both modes use the same search; only the AI model changes.

---

## How it works

```
                 WHEN THE APP STARTS (once)
  Product CSV ──► one text per product ──► turned into numbers ──► saved in a FAISS index
                                          (all-MiniLM-L6-v2)

                 FOR EVERY MESSAGE
  User message
     │
     ▼
  1. Understand   what the user wants: product type, price, brand, colour, size, rating…
     │
     ▼
  2. Search       find products with a similar meaning (FAISS),
     │            then apply every filter exactly (e.g. "under ₹200" means less than 200)
     ▼
  3. Result       one final list of matching products
     │
     ▼
  4. Reply        a summary of that list (an AI model writes it, if a key is given)
     │
     ▼
  5. Check        every fact in the AI reply is checked against the list;
     │            if anything is made up, a safe template reply is shown instead
     ▼
  6. Show         reply + product cards (with arrows to see every result) + cart
```

### Important behaviour

- **Filters are strict.**
  - "under ₹200" means less than 200; "up to ₹200" means 200 or less.
  - "above" means more than; "at least" means that much or more.
  - Products that don't match are never shown as matches.
- **No silent changes.** If nothing matches, the app says *"No matching products found"* and may
  offer one clearly marked suggestion (for example *"shirts under ₹300?"*). It only applies the
  suggestion if you click it.
- **All results are shown.** The carousel shows 3 cards at a time, and the arrows let you see every match.
- **Each message is a new search.** Filters from earlier messages are not carried over.
- **Specific products.** If you name an exact product, only that product is shown, with no similar ones.
- **Size and colour first.** Before adding a product to the cart, the app asks for its size and colour.
- **Unclear price.** For something like "worth ₹300", the app asks whether you mean
  "up to", "exactly" or "around" ₹300.
- **Clear errors.** If the search index can't load, the app shows an error instead of giving wrong results.

### Chunking

There is **no chunking**: each product is one document. A product's text is only about
80–130 tokens, which fits easily within the embedding model's limit (256 tokens).
Keeping each product whole means every search result is one complete product.

---

## Project structure

```
app.py                              Streamlit web app (chat screen, sidebar, cart)
config.py                           All settings in one place (dataset, model, limits, word lists)
data/ecommerce_products_1000.csv    The product catalog (1,000 products)

rag/                                The RAG pipeline
  ingestion/product_ingestion.py      Reads and checks the CSV; makes one text per product
  embeddings/embedding_model.py       Loads the embedding model (all-MiniLM-L6-v2)
  vectorstore/faiss_manager.py        Builds, saves and loads the FAISS search index
  retrieval/product_retriever.py      Finds relevant products and ranks them
  retrieval/product_filter.py         Applies price, brand, colour, size… filters exactly
  llm/llm_factory.py                  Connects to Groq or OpenAI
  llm/system_prompt.py                Builds the AI prompt and checks the AI reply for invented facts

chatbot/                            The chatbot logic
  intent_detector.py                  Understands the message (what the user wants + filters)
  shopping_assistant.py               Decides what to do and builds the reply
  conversation_manager.py             Keeps the chat history

components/                         Parts of the screen
  product_cards.py                    Product cards and the carousel
  shopping_cart.py                    The cart (stock limits, size and colour)
  category_buttons.py                 Category and suggestion buttons

utils/helpers.py                    Small helpers (₹ formatting, text matching)
```

---

## Tech stack

| Part | Tool |
|---|---|
| Web app | Streamlit |
| Data and filters | Pandas |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| Vector search | FAISS |
| AI models | Groq (free mode) and OpenAI (paid mode) via LangChain |
| Language | Python 3.11 |

---

## Known limitations

- **Free-form questions.** Questions written without product words, like *"something to listen
  to music"*, often return no results. The search works best when the message names a product
  type, brand or feature.
- **Parsing rules and word lists.** Message understanding uses rules and word lists, so very
  unusual wording or spelling mistakes may not be understood.
- **One size and colour per product in the cart.** To change them, remove the product and add it again.
