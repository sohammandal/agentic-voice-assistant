# Exploratory Data Analysis (EDA) Module

This document outlines the exploratory data analysis performed on the Amazon Product Dataset 2020 to select the optimal product category for our voice-to-voice AI assistant. Our goal was to identify a category that has enough volume and text content to create a comprehensive metadata for tool usage.

**Data Source**: [Amazon Product Dataset 2020](https://www.kaggle.com/datasets/promptcloud/amazon-product-dataset-2020) (Kaggle)


## Data Preprocessing Pipeline

### 1. Price Extraction

**Challenge**: Initially, the prices were stored as strings with inconsistent formatting (such as `"$1,070.23"`, `"$15.99"`)

**Solution**: To make the prices consistent, we applied the following regex price extraction function:

```python
def extract_prices(val):
    """Converts string prices to numerical prices (eg. '$1,070.23' --> 1070.23) """

    # Return value if null
    if pd.isna(val):
        return np.nan

    # Remove whitespace
    s = str(val).strip()

    # Extract value after the $ sign
    match = re.search(r"\$?\s*([0-9]*,?[0-9]*\.?[0-9]+)", s)
    if not match:
        return np.nan

    # Remove commas
    num_str = match.group(1).replace(",", "")

    return float(num_str)
```

---

### 2. Category Normalization

**Challenge**: Initially, the categories were stored as paths (such as `"Electronics | Computers & Accessories"`)

**Solution**: As a result, we created a function to extract the main categories:

```python
def extract_main_category(val):
    """Extracts the main category (eg. 'Electronics | Computers & Accessories' --> "Electronics") """
    
    # Return value if null
    if pd.isna(val):
        return None

    # Remove whitespace
    s = str(val).strip()
    
    # Extract main category
    main_cat = s.split(" | ")[0].strip().lower()


    return main_cat
```

---

### 3. Weight Normalization

**Challenge**: Initially, the weights were stored in different units (such as `"16 ounces"`, `"2.5 pounds"`, or `"1 lbs"`)

**Solution**: To address this, we standardized all of the weights into pounds via a regex function:

```python
def extract_weights(val):
    """Converts string weights to numerical pounds (eg. '16 ounces --> 0.5) """

    # Return value if null
    if pd.isna(val):
        return np.nan
    
    # Remove commas and trim whitespace
    val = str(val).strip().replace(",", "")
    
    # Extract the numerical weight and weight type
    match = re.match(r"([0-9]*\.?[0-9]+)\s*(pounds|pound|lbs|ounces|ounce|oz)", val.lower())
    if not match:
        return np.nan
    num = float(match.group(1))
    unit = match.group(2)

    # Convert to pounds if in ounces
    if unit in ["ounces", "ounce", "oz"]:
        return num / 16.0
    else:
        return num
```

---

### 4. Dimension Normalization

**Challenge**: Initially, the dimensions were stored in string formats (such as `"14.7 x 11.1 x 10.2 inches"` and `"3.5x6.2x13inches"`)

**Solution**: To standardize this, we created a regex function:

```python
def extract_dimensions(text):
    """Extract dimensions from string (eg. '14.7 x 11.1 x 10.2 inches')"""
    
    # Return value if null
    if not isinstance(text, str):
        return pd.Series([np.nan, np.nan, np.nan, np.nan])

    # Regex to capture L x W x H with optional unit
    dim_pattern = re.compile(
        r'(\d+\.?\d*)\s*[xX×]\s*(\d+\.?\d*)\s*(?:[xX×]\s*(\d+\.?\d*))?\s*(inches|inch|in)?',
        re.IGNORECASE
    )

    # Try match without spaces (handles formats like "3.5x6.2x13inches")
    match = dim_pattern.search(text.replace(" ", ""))

    # If no match, try again preserving spaces (handles "14.7 x 11.1 x 10.2 inches")
    if not match:
        match = dim_pattern.search(text)

    # If still no match → no dimension found
    if not match:
        return pd.Series([np.nan, np.nan, np.nan, np.nan])

    # Extract length, width, height
    length = float(match.group(1))
    width = float(match.group(2))
    height = match.group(3)
    height = float(height) if height else np.nan

    # Clean and standardize unit (default → inches)
    unit = match.group(4)
    if unit:
        unit = unit.lower()
        unit = "inches" if unit in ["inch", "in"] else unit
    else:
        unit = "inches"

    return pd.Series([length, width, height, unit])
```

---

### 5. Brand Extraction

**Challenge**: In the original dataset, the brands were not labeled for most listings (such as `"KitchenAid 5-Speed Hand Mixer"`)

**Solution**: As such, we created a hueristic-based function to extract the brands:

```python
def extract_brand(product_name):
    """
    Extract brand from the beginning of a product name.
    
    Rules:
    - Take the leading capitalized word(s)
    - Stop before digits, lower-case transitions, or model numbers
    - If a hyphenated prefix exists, take left side
    """

    # Return value if null
    if not isinstance(product_name, str):
        return np.nan

    # Remove whitespace
    name = product_name.strip()

    # If hyphen early it is likely brand prefix
    if "-" in name.split(" ")[0]:
        return name.split(" ")[0].split("-")[0]

    # Tokenize the name
    tokens = name.split()

    brand_tokens = []
    for token in tokens:
        # Stop if token begins with a digit
        if re.match(r'^\d', token):
            break
        
        # Stop if lowercase indicates description not brand
        if token[0].islower():
            break
        
        # Stop if token looks like a model number
        if re.match(r'^[A-Za-z]*\d+', token):  
            break
        
        brand_tokens.append(token)

    if len(brand_tokens) == 0:
        return np.nan

    return " ".join(brand_tokens)
```

---

### 6. Overall Text Cleaning

**Challenge**: For general cleaning for best RAG usage, we also applied a general cleaning function to remove scraped metadata and common junk phrases:

```python
def clean_text(x):
    """Cleans text (eg. removing common verbiage, collapsing whitespace, lowercase optional). """
    if not isinstance(x, str):
        return ""
    
    # Remove Amazon boilerplate
    x = re.sub(r"make sure this fits by entering your model number\.?", "", x, flags=re.IGNORECASE)
    
    # Remove separators like " | "
    x = x.replace("|", " ")
    
    # Remove multiple spaces
    x = re.sub(r"\s+", " ", x).strip()
    
    return x

def remove_ui_noise(text):
    """
    Cleans out Amazon UI boilerplate and scraped metadata blocks.
    Removes:
    - shipping weight lines
    - product dimensions lines (scraped, not structured)
    - ASIN, item model numbers, manufacturer age ranges
    - domestic/international shipping messages
    - style attributes
    - glued-together UI artifacts
    """

    noise_patterns = [
        r"View shipping rates and policies",
        r"Go to your orders and start the return",
        r"Select the ship method",
        r"Ship it!",
        r"ASIN:\S+",
        r"Itemmodelnumber:\S+",
        r"Manufacturerrecommendedage:\S+",
        r"ProductDimensions:\S+",
        r"ItemWeight:\S+",
        r"ShippingWeight:\S+",
        r"DomesticShipping:\S+",
        r"InternationalShipping:\S+",
        r"Style:\S+",
        r"LearnMore",
        r"\(\)",                   # empty parentheses
        r"\d+\.?\d*\s*ounces",     # any weight in ounces
        r"\d+\.?\d*\s*inch(es)?",  # bare dimensions
    ]

    for pat in noise_patterns:
        text = re.sub(pat, " ", text, flags=re.IGNORECASE)

    # Collapse any leftover whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()
```

---

### 7. Metadata & Embedding Text Extraction

Lastly, to setup effective RAG usage via our MCP tools, we created a metadata for each row (product):

```python
def extract_embedding(row):
    """
    Builds a merged text field for product embeddings.

    Includes:
    - product_name
    - brand_name
    - category + category path
    - about_product
    - product_specification
    - technical_details
    - dimensions (if available)
    - price per pound (human readable)

    Excludes:
    - Amazon UI text (shipping boilerplate)
    - repeated glue text
    """

    parts = []

    # Product name
    if pd.notna(row.get("product_name")):
        parts.append(clean_text(row["product_name"]))

    # Brand
    if pd.notna(row.get("brand_name")):
        parts.append(f"Brand: {clean_text(row['brand_name'])}")

    # Categories
    if pd.notna(row.get("main_category")):
        parts.append(f"Category: {clean_text(row['main_category'])}")

    if pd.notna(row.get("category")):
        parts.append(f"Category path: {clean_text(row['category'])}")

    # About product
    if pd.notna(row.get("about_product")):
        cleaned = clean_text(row["about_product"])
        cleaned = remove_ui_noise(cleaned)
        parts.append(cleaned)

    # Product specification
    if pd.notna(row.get("product_specification")):
        cleaned = clean_text(row["product_specification"])
        cleaned = remove_ui_noise(cleaned)
        parts.append(cleaned)

    # Technical details
    if pd.notna(row.get("technical_details")):
        cleaned = clean_text(row["technical_details"])
        cleaned = remove_ui_noise(cleaned)
        parts.append(cleaned)

    # Dimensions
    dims = []
    if pd.notna(row.get("dim_length")):
        dims.append(f"length {row['dim_length']}")
    if pd.notna(row.get("dim_width")):
        dims.append(f"width {row['dim_width']}")
    if pd.notna(row.get("dim_height")):
        dims.append(f"height {row['dim_height']}")

    if dims:
        unit = row.get("dim_unit", "")
        if isinstance(unit, str) and len(unit) > 0:
            parts.append("Dimensions: " + ", ".join(dims) + f" {unit}")
        else:
            parts.append("Dimensions: " + ", ".join(dims))

    # Price per pound
    if pd.notna(row.get("normalized_weight")):
        price_val = row["normalized_weight"]
        readable = f"Price per pound: ${price_val:.2f}"
        parts.append(readable)

    # Join everything
    full_text = " ".join(parts)
    full_text = re.sub(r"\s+", " ", full_text).strip()

    return full_text

def extract_metadata(row):
    """
    Build a metadata dictionary for a product row.

    Includes:
    - Core identifiers
    - Category information
    - Numeric metadata
    - Feature text (combined)
    - Optional ingredients
    - Flags / URLs / quality indicators
    """

    # Construct a minimal "features" field using your available descriptive fields
    features_text = " ".join([
        str(row.get("about_product", "")),
        str(row.get("product_specification", "")),
        str(row.get("technical_details", "")),
    ]).strip()

    metadata_dict = {
        # Identifiers
        "product_id": row.get("product_id"),
        "product_name": row.get("product_name"),

        # Categories
        "main_category": row.get("main_category"),
        "category_path": row.get("category"),

        # Numeric metadata
        "price": row.get("selling_price"),
        "shipping_weight_lbs": row.get("shipping_weight"),
        "dim_length": row.get("dim_length"),
        "dim_width": row.get("dim_width"),
        "dim_height": row.get("dim_height"),

        # Flags
        "is_amazon_seller": row.get("is_amazon_seller", False),  # Already bool earlier
        "has_variants": pd.notna(row.get("variants")),

        # Text metadata
        "model_number": row.get("model_number"),
        "brand": row.get("brand_name"),   # Use your extracted brand_name field
        "image_url": row.get("image"),
        "product_url": row.get("product_url"),

        # Computed metadata
        "features": features_text,
        "ingredients": None,  # Dataset doesn't include structured ingredients

        # Length / quality indicators
        "about_length": len(row["about_product"]) if pd.notna(row.get("about_product")) else 0,
        "spec_length": len(row["product_specification"]) if pd.notna(row.get("product_specification")) else 0,
        "tech_details_length": len(row["technical_details"]) if pd.notna(row.get("technical_details")) else 0,
    }

    return metadata_dict
```

This resulted in the following structures for the metadata extractions:

**`metadata`** (structured JSON):
```python
{
    "product_id": "B07XYZ123",
    "product_name": "KitchenAid 5-Speed Hand Mixer",
    "main_category": "home & kitchen",
    "price": 49.99,
    "brand": "KitchenAid",
    "shipping_weight_lbs": 2.5,
    "dim_length": 10.0,
    "dim_width": 5.0,
    "dim_height": 3.0,
    "model_number": "KHM512",
    "is_amazon_seller": True,
    "image_url": "https://...",
    "product_url": "https://..."
}
```

---

## Category Selection Analysis

### Evaluation Criteria

Lastly, once we had sufficiently cleaned the data, we needed to select the best category slice to use for our agentic system. Specifically, to make this decision, we evaluated all of the potential slices against four metrics:

1. **`n_products`**: Number of products
2. **`avg_about_len`**: Average length of product descriptions
3. **`missing_dims`**: Proportion of products missing dimensions
4. **`text_coverage`**: Average length of the `embedding_text`

### Results

| Category | n_products | avg_about_len | missing_dims | text_coverage |
|----------|------------|---------------|--------------|---------------|
| sports & outdoors | 540 | 564.39 | 0.628 | 1538.84 |
| **home & kitchen** | **708** | **442.73** | **0.968** | **1157.78** |
| toys & games | 6,662 | 419.12 | 0.974 | 1154.62 |
| baby products | 214 | 417.23 | 0.991 | 1265.46 |
| clothing, shoes & jewelry | 630 | 324.35 | 0.992 | 1084.61 |

Ultimately, while the Sports & Outdoors category ranked the highest on dimension completeness, we ended up choosing the Home & Kitchen data slice for the following reasons:

#### 1. **Sufficient Volume & Diversity**

The 708 products provide a lot of coverage that enables testing of cross-category queries and comparison scenarios.

#### 2. **Large Semantic Content**

Despite having slightly lower `avg_about_len` (443 chars vs. 564 for Sports), the Home & Kitchen products had more detailed feature descriptions, specific use-cases, and detailed ingredient/component lists. Ultimately, the text coverage of 1,158 characters per product offered high potential semantic information for embedding quality.

#### 3. **Safety Filtering Requirements**

Lastly, this assignment requires safety checks (see `prompts.md`). This was another benefit of the Home & Kitchen data slice, since it has clear safety boundaries and ingredient transparency. Together, these offer a good blend of attributes that enable strong safety filtering.

