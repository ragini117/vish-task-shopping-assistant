import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# ---------------------------------------------------------------------------
# Data & index paths
# ---------------------------------------------------------------------------
# The one place the catalog file is chosen (override with PRODUCTS_CSV_PATH). The FAISS index stores this
# file's fingerprint and row count, so switching files always triggers a rebuild.
DATA_PATH = Path(os.getenv("PRODUCTS_CSV_PATH", BASE_DIR / "data" / "ecommerce_products_1000.csv"))
# One index folder per catalog file, so an index built from another CSV (e.g. by a still-running older app
# process) can never overwrite or be mistaken for this one.
INDEX_DIR = Path(os.getenv("FAISS_INDEX_DIR", BASE_DIR / "rag" / "vectorstore" / "faiss_index" / DATA_PATH.stem))

REQUIRED_COLUMNS = [
    "product_id", "name", "category", "subcategory", "brand", "price",
    "discount_percent", "sale_price", "description", "rating", "review_count",
    "sizes", "colors", "stock_quantity", "in_stock", "material", "tags",
]

# ---------------------------------------------------------------------------
# Embeddings & retrieval
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_CONTEXT_PRODUCTS = 8          # products sent to the LLM per turn
SEMANTIC_MIN_SCORE = 0.35         # a product must be at least this similar to the request to be relevant
SEMANTIC_TOP_MARGIN = 0.10        # ...and within this margin of the best-matching product
SEMANTIC_SUBCATEGORY_SHARE = 0.80 # if this share of hits is one subcategory, that whole subcategory is relevant
CAROUSEL_CARDS_PER_PAGE = 3       # cards per carousel page on desktop (2 on tablet, 1 on mobile via CSS)
LOW_STOCK_THRESHOLD = 10

# Price semantics (sale_price, ₹):
#   "under/below/less than/cheaper than X"                -> price <  X
#   "up to/maximum/at most/budget/for X"                  -> price <= X
#   "above/over/more than X"                              -> price >  X
#   "at least/minimum/starting from X"                    -> price >= X
#   "between X and Y"                                     -> X <= price <= Y
#   "exactly/priced at/costing/of/at X"                   -> X ± EXACT_PRICE_SLACK (same rupee)
#   "around/about X"                                      -> X ± AROUND_PRICE_TOLERANCE (20%)
#   bare amount next to a named product                   -> that product's exact price
#   "worth X" or any other bare amount ("shoes ₹2000")    -> ambiguous: the assistant asks which one is meant
EXACT_PRICE_SLACK = 0.5
AROUND_PRICE_TOLERANCE = 0.20
PRICE_SLIDER_STEP = 100
HISTORY_TURNS_FOR_LLM = 6         # recent chat messages sent to the LLM

# ---------------------------------------------------------------------------
# LLM providers
# ---------------------------------------------------------------------------
FREE_MODE = "Free (Groq)"
PAID_MODE = "Paid (OpenAI)"

GROQ_DEFAULT_MODEL = "openai/gpt-oss-120b"
# The sidebar fetches the live model list from the Groq API; this list is only used when that call fails.
GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
    "allam-2-7b",
]
OPENAI_MODELS = [
    "gpt-4o-mini",
    "gpt-4.1-mini",
    "gpt-4o",
    "gpt-4.1",
]
LLM_TEMPERATURE = 0.2
LLM_TIMEOUT_SECONDS = 45

GROQ_API_KEY_ENV = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY_ENV = os.getenv("OPENAI_API_KEY", "")

CURRENCY = "₹"

# ---------------------------------------------------------------------------
# Language hints for intent detection.
# These map everyday words onto the dataset's own category/subcategory names.
# Entries whose target does not exist in the loaded CSV are ignored, so the
# buttons and filters always come from the data itself.
# ---------------------------------------------------------------------------
CATEGORY_SYNONYMS = {
    "Footwear": ["shoe", "shoes", "footwear", "chappal", "chappals"],
    "Clothing": ["cloth", "cloths", "clothes", "clothing", "apparel", "outfit", "outfits", "garment", "garments", "wear"],
    "Electronics": ["electronics", "electronic", "gadget", "gadgets", "tech", "devices", "device"],
    "Books": ["book", "books", "reading"],
    "Home & Kitchen": ["home", "kitchen", "household", "home decor"],
    "Beauty & Personal Care": ["beauty", "personal care", "cosmetic products", "self care"],
    "Sports & Outdoors": ["sports", "sport", "outdoors", "outdoor", "sporting goods"],
    "Bags & Accessories": ["bag", "bags", "fashion accessories"],
}

SUBCATEGORY_SYNONYMS = {
    "Smartphones": ["phone", "phones", "mobile", "mobiles", "mobile phone", "cellphone", "smart phone"],
    "Laptops": ["laptop", "notebook computer", "macbook"],
    "Headphones": ["earphones", "earbuds", "headset", "headsets", "earphone", "headphone"],
    "Smartwatches": ["smart watch", "smart watches", "watch", "watches", "fitness band"],
    "Speakers": ["speaker", "soundbar", "bluetooth speaker"],
    "Accessories": ["charger", "chargers", "cable", "cables", "power bank"],
    "Running Shoes": ["jogging shoes", "running shoe", "runners"],
    "Sports Shoes": ["sports shoe", "training shoes", "gym shoes", "trainers"],
    "Casual Shoes": ["sneakers", "sneaker", "loafers"],
    "Formal Shoes": ["formal shoe", "office shoes", "dress shoes", "oxfords"],
    "Sandals": ["sandal", "slippers", "flip flops", "flip-flops", "slides"],
    "Boots": ["boot"],
    "T-Shirts": ["tshirt", "tshirts", "tee", "tees", "t shirt", "t shirts"],
    "Shirts": ["shirt"],
    "Jeans": ["denim", "denims"],
    "Jackets": ["jacket", "coat", "coats", "blazer"],
    "Hoodies": ["hoodie", "sweatshirt", "sweatshirts"],
    "Dresses": ["dress", "gown", "frock"],
    "Fiction": ["novel", "novels", "story books"],
    "Non-Fiction": ["nonfiction", "non fiction"],
    "Self-Help": ["self help", "selfhelp", "motivational"],
    "Business": ["business books", "finance books"],
    "Technology": ["programming", "coding books", "tech books"],
    "Skincare": ["skin care", "moisturizer", "face cream", "sunscreen", "serum"],
    "Makeup": ["make up", "make-up", "lipstick", "cosmetics", "foundation"],
    "Haircare": ["hair care", "shampoo", "conditioner", "hair oil"],
    "Fragrance": ["perfume", "perfumes", "deodorant", "deo", "cologne", "scent"],
    "Personal Grooming": ["grooming", "trimmer", "razor", "shaver"],
    "Furniture": ["sofa", "chair", "chairs", "table", "tables", "desk", "bed frame"],
    "Lighting": ["lamp", "lamps", "lights"],
    "Cookware": ["pan", "pans", "pot", "pots", "utensils", "kadai"],
    "Bedding": ["bedsheet", "bedsheets", "pillow", "pillows", "blanket", "comforter"],
    "Storage": ["organizer", "organiser", "container", "containers", "shelf"],
    "Appliances": ["appliance", "mixer", "blender", "kettle", "toaster"],
    "Fitness Equipment": ["dumbbell", "dumbbells", "gym equipment", "workout equipment", "kettlebell"],
    "Yoga": ["yoga mat", "yoga mats"],
    "Cycling": ["bicycle", "bicycles", "bike", "bikes", "cycle"],
    "Camping Gear": ["camping", "tent", "tents", "sleeping bag"],
    "Sports Accessories": ["sports accessory", "water bottle", "gym bag"],
    "Backpacks": ["backpack", "rucksack", "school bag", "laptop bag"],
    "Handbags": ["handbag", "purse", "purses", "tote", "clutch"],
    "Wallets": ["wallet", "card holder"],
    "Belts": ["belt"],
    "Travel Bags": ["luggage", "suitcase", "suitcases", "trolley bag", "duffel"],
}

# Tags that are adjectives/occasions rather than product types; they are never
# used to infer a subcategory from the user's words.
GENERIC_TAGS = {
    "work", "soft", "design", "utility", "personal", "performance", "functional",
    "compact", "convenience", "lightweight", "comfort", "summer", "party", "school",
    "story", "warm", "notifications", "habits", "layering", "adventure", "athletic",
    "wireless", "bluetooth", "wellness", "decor", "space-saving", "open-toe",
    "daily-use", "knowledge", "management", "finance", "camera", "bedroom", "ankle",
    "durable", "everyday", "fashion", "casual", "formal", "office", "outdoor",
    "travel", "training", "fitness", "learning", "reading", "storage", "exercise",
}

# Popular real-world brands the catalog does NOT carry. Used only to tell the
# user clearly that a requested brand is unavailable instead of silently
# showing something else.
EXTERNAL_BRANDS = [
    "nike", "adidas", "puma", "reebok", "asics", "new balance", "skechers", "bata",
    "woodland", "red tape", "apple", "samsung", "oneplus", "xiaomi", "redmi", "realme",
    "oppo", "vivo", "google pixel", "sony", "boat", "jbl", "bose", "sennheiser", "dell",
    "hp", "lenovo", "asus", "acer", "msi", "levis", "levi's", "zara", "h&m", "uniqlo",
    "allen solly", "van heusen", "peter england", "lakme", "maybelline", "loreal",
    "l'oreal", "nivea", "philips", "prestige", "ikea", "wildcraft", "american tourister",
    "samsonite", "decathlon", "titan", "fossil", "fastrack", "fire-boltt",
]
