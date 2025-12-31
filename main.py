from flask import Flask, render_template, request, jsonify
import requests
import json
import re
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

# ============== SMART RELEVANCE SCORING SYSTEM ==============

# Common stop words to ignore in relevance calculations
STOP_WORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
    'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
    'she', 'we', 'they', 'my', 'your', 'his', 'her', 'our', 'their'
}

# Common word variations (stemming without external libraries)
WORD_STEMS = {
    'cups': 'cup', 'glasses': 'glass', 'bottles': 'bottle', 'plates': 'plate',
    'bowls': 'bowl', 'forks': 'fork', 'spoons': 'spoon', 'knives': 'knife',
    'napkins': 'napkin', 'towels': 'towel', 'bags': 'bag', 'boxes': 'box',
    'cans': 'can', 'jars': 'jar', 'containers': 'container', 'packs': 'pack',
    'packets': 'packet', 'pieces': 'piece', 'sets': 'set', 'rolls': 'roll',
    'sheets': 'sheet', 'wipes': 'wipe', 'tissues': 'tissue', 'papers': 'paper',
    'drinks': 'drink', 'beverages': 'beverage', 'foods': 'food', 'snacks': 'snack',
    'cookies': 'cookie', 'crackers': 'cracker', 'chips': 'chip', 'nuts': 'nut',
    'waters': 'water', 'sodas': 'soda', 'juices': 'juice', 'milks': 'milk',
    'cereals': 'cereal', 'breads': 'bread', 'pastas': 'pasta', 'rices': 'rice',
    'meats': 'meat', 'chickens': 'chicken', 'beefs': 'beef', 'porks': 'pork',
    'fruits': 'fruit', 'vegetables': 'vegetable', 'veggies': 'veggie',
    'apples': 'apple', 'oranges': 'orange', 'bananas': 'banana', 'grapes': 'grape',
    'tomatoes': 'tomato', 'potatoes': 'potato', 'onions': 'onion', 'carrots': 'carrot',
    'cleaners': 'cleaner', 'detergents': 'detergent', 'soaps': 'soap',
    'shampoos': 'shampoo', 'conditioners': 'conditioner', 'lotions': 'lotion',
    'diapers': 'diaper', 'formulas': 'formula',
    'batteries': 'battery', 'bulbs': 'bulb', 'lights': 'light',
    'tumblers': 'tumbler', 'mugs': 'mug', 'straws': 'straw',
    'disposables': 'disposable', 'plastics': 'plastic',
    'reds': 'red', 'blues': 'blue', 'greens': 'green', 'blacks': 'black', 'whites': 'white',
    'gallons': 'gallon', 'quarts': 'quart', 'pints': 'pint', 'liters': 'liter', 'ounces': 'ounce',
}

# Volume/size units and their normalized forms
SIZE_UNITS = {
    'gallon': ['gallon', 'gal'],
    'half gallon': ['half gallon', '1/2 gallon', '0.5 gallon', '64 oz', '64oz', '64 fl oz'],
    'quart': ['quart', 'qt', '32 oz', '32oz', '32 fl oz'],
    'pint': ['pint', 'pt', '16 oz', '16oz', '16 fl oz'],
    'liter': ['liter', 'litre', 'l', '1l', '1 l'],
    'half liter': ['half liter', '500ml', '500 ml', '0.5l', '0.5 l'],
}

# Count/quantity patterns
COUNT_PATTERNS = [
    r'(\d+)\s*(?:ct|count|pk|pack|pc|pcs|piece|pieces)',  # 36ct, 36 count, 36pk
    r'(\d+)\s*-?\s*(?:ct|count|pk|pack)',  # 36-ct, 36-count
    r'(?:^|\s)(\d+)\s*(?:ct|count|pk|pack)(?:\s|$)',  # standalone count
]


def extract_quantity_info(text):
    """
    Extract quantity/size information from text.
    Returns dict with 'count', 'size', 'size_oz' keys.
    """
    if not text:
        return {'count': None, 'size': None, 'size_oz': None}
    
    text_lower = text.lower()
    result = {'count': None, 'size': None, 'size_oz': None}
    
    # Extract count (e.g., "36 count", "72ct", "24 pack")
    for pattern in COUNT_PATTERNS:
        match = re.search(pattern, text_lower)
        if match:
            result['count'] = int(match.group(1))
            break
    
    # Also check for counts like "36ct" without spaces
    if not result['count']:
        match = re.search(r'(\d+)ct\b', text_lower)
        if match:
            result['count'] = int(match.group(1))
    
    # Extract size in ounces (fl oz, oz)
    oz_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:fl\s*)?oz\b', text_lower)
    if oz_match:
        result['size_oz'] = float(oz_match.group(1))
    
    # Check for gallon/half gallon
    if 'half gallon' in text_lower or '1/2 gallon' in text_lower or '0.5 gallon' in text_lower:
        result['size'] = 'half gallon'
        result['size_oz'] = 64
    elif 'gallon' in text_lower and 'half' not in text_lower:
        result['size'] = 'gallon'
        result['size_oz'] = 128
    elif 'quart' in text_lower:
        result['size'] = 'quart'
        result['size_oz'] = 32
    elif 'pint' in text_lower:
        result['size'] = 'pint'
        result['size_oz'] = 16
    elif 'liter' in text_lower or 'litre' in text_lower:
        if 'half' in text_lower or '0.5' in text_lower or '500' in text_lower:
            result['size'] = 'half liter'
        else:
            result['size'] = 'liter'
    
    # Infer size from oz if not already set
    if not result['size'] and result['size_oz']:
        if 120 <= result['size_oz'] <= 140:
            result['size'] = 'gallon'
        elif 60 <= result['size_oz'] <= 68:
            result['size'] = 'half gallon'
        elif 30 <= result['size_oz'] <= 34:
            result['size'] = 'quart'
        elif 14 <= result['size_oz'] <= 18:
            result['size'] = 'pint'
    
    return result


def normalize_word(word):
    """Normalize a word by lowercasing and applying basic stemming."""
    word = word.lower().strip()
    if word in WORD_STEMS:
        return WORD_STEMS[word]
    if word.endswith('ing') and len(word) > 5:
        return word[:-3]
    if word.endswith('ed') and len(word) > 4:
        return word[:-2]
    if word.endswith('s') and len(word) > 3 and not word.endswith('ss'):
        return word[:-1]
    return word


def tokenize_query(query):
    """
    Tokenize a search query into meaningful terms.
    Returns (all_terms, important_terms, quantity_info)
    """
    # Extract quantity info first
    quantity_info = extract_quantity_info(query)
    
    # Split on non-alphanumeric characters
    words = re.findall(r'[a-zA-Z0-9]+', query.lower())
    
    # Normalize all words
    all_terms = [normalize_word(w) for w in words]
    
    # Filter out stop words and pure numbers for important terms
    important_terms = [w for w in all_terms if w not in STOP_WORDS and len(w) > 1 and not w.isdigit()]
    
    # Remove quantity-related words from important terms (we handle them separately)
    # BUT keep product-type words that happen to include sizes (like "gallon" when it's part of the product)
    quantity_only_words = {'count', 'ct', 'pack', 'pk', 'oz', 'fl', 'quart', 'pint', 'liter', 'half'}
    important_terms = [w for w in important_terms if w not in quantity_only_words]
    # Note: 'gallon' is kept because "gallon of milk" should match products with "gallon" in the name
    
    return all_terms, important_terms, quantity_info


def calculate_relevance_score(product_name, query, all_terms, important_terms, query_quantity):
    """
    Calculate a relevance score for a product.
    Returns (base_score, quantity_match, is_exact_match)
    """
    if not product_name or not important_terms:
        return 0.0, False, False
    
    # Normalize product name
    product_lower = product_name.lower()
    product_words = re.findall(r'[a-zA-Z0-9]+', product_lower)
    product_normalized = [normalize_word(w) for w in product_words]
    product_set = set(product_normalized)
    
    # Extract product quantity info
    product_quantity = extract_quantity_info(product_name)
    
    score = 0.0
    matched_terms = 0
    
    # 1. Check for exact phrase match (highest bonus)
    query_lower = query.lower()
    # Remove quantity parts for phrase matching
    query_clean = re.sub(r'\d+\s*(ct|count|pk|pack|oz|fl oz|gallon|quart|pint)', '', query_lower).strip()
    if query_clean and query_clean in product_lower:
        score += 0.5  # Increased from 0.4
    
    # 2. Check for each important term
    for term in important_terms:
        if term in product_set:
            matched_terms += 1
            score += 0.3  # Increased from 0.25
        elif any(term in pw or pw in term for pw in product_normalized if len(term) > 2 and len(pw) > 2):
            matched_terms += 0.5
            score += 0.15  # Increased from 0.1
    
    # 3. Calculate term coverage ratio (more generous for short queries)
    if important_terms:
        coverage = matched_terms / len(important_terms)
        # Bonus is higher for shorter queries (single word searches get more boost)
        coverage_bonus = 0.4 if len(important_terms) <= 2 else 0.3
        score += coverage * coverage_bonus
    
    # 4. Bonus for matching specific/brand terms
    for term in important_terms:
        if len(term) >= 4 and term in product_set:
            score += 0.15  # Increased from 0.1
    
    # 5. Base score if at least one term matches - ensure these always show
    if matched_terms >= 1:
        score += 0.2  # Bigger boost to ensure products with matches always appear
    
    # 5. Check quantity/size match
    quantity_match = True
    
    # Check count match
    if query_quantity['count'] is not None:
        if product_quantity['count'] is not None:
            if product_quantity['count'] != query_quantity['count']:
                quantity_match = False
        else:
            # Product doesn't specify count, might be single item
            quantity_match = False
    
    # Check size match (gallon vs half gallon, etc.)
    if query_quantity['size'] is not None:
        if product_quantity['size'] is not None:
            if product_quantity['size'] != query_quantity['size']:
                quantity_match = False
        elif product_quantity['size_oz'] is not None:
            # Check if oz is close enough
            query_oz = query_quantity.get('size_oz')
            if query_oz:
                tolerance = query_oz * 0.15  # 15% tolerance
                if abs(product_quantity['size_oz'] - query_oz) > tolerance:
                    quantity_match = False
        else:
            quantity_match = False
    
    # Determine if this is an exact match (good base score + quantity matches)
    is_exact_match = score >= 0.4 and quantity_match
    
    # Apply quantity bonus/penalty to final score
    if quantity_match:
        score += 0.3  # Bonus for matching quantity
    
    return score, quantity_match, is_exact_match


def categorize_products(products, query):
    """
    Categorize products into 'exact_matches' and 'similar_items'.
    Returns dict with both categories.
    """
    if not products:
        return {'exact_matches': [], 'similar_items': [], 'filtered_out': 0}
    
    all_terms, important_terms, query_quantity = tokenize_query(query)
    
    print(f"Important terms: {important_terms}")
    print(f"Query quantity: {query_quantity}")
    
    if not important_terms:
        return {'exact_matches': products, 'similar_items': [], 'filtered_out': 0}
    
    exact_matches = []
    similar_items = []
    filtered_out = 0
    debug_count = 0
    
    # Calculate scores for all products
    for product in products:
        score, quantity_match, is_exact = calculate_relevance_score(
            product.get('name', ''),
            query,
            all_terms,
            important_terms,
            query_quantity
        )
        
        product['relevance_score'] = score
        product['quantity_match'] = quantity_match
        
        # Debug first few products
        if debug_count < 5:
            store = product.get('store', '?')
            name = product.get('name', 'N/A')[:50]
            print(f"  [{store}] Score={score:.2f}, qty_match={quantity_match}, exact={is_exact}: {name}...")
            debug_count += 1
        
        # Categorize based on score and quantity match (very lenient thresholds)
        if score >= 0.1:  # Very low threshold - show almost everything
            if is_exact or score >= 0.25:  # Most relevant items
                exact_matches.append(product)
            else:
                similar_items.append(product)
        else:
            # Only filter out items with almost no relevance
            filtered_out += 1
    
    # Sort exact matches by: quantity match first, then by price
    exact_matches.sort(key=lambda x: (not x.get('quantity_match', False), x.get('price', 0)))
    
    # Sort similar items by relevance score, then price
    similar_items.sort(key=lambda x: (-x.get('relevance_score', 0), x.get('price', 0)))
    
    return {
        'exact_matches': exact_matches,
        'similar_items': similar_items,
        'filtered_out': filtered_out
    }


def smart_filter_products(products, query):
    """
    Apply smart filtering - returns all relevant products.
    (Kept for backward compatibility, now returns just the combined list)
    """
    result = categorize_products(products, query)
    return result['exact_matches'] + result['similar_items']

# ============== END SMART RELEVANCE SCORING SYSTEM ==============

def get_walmart_cookies():
    """Load Walmart cookies from config file, or return empty string if not found."""
    import os
    cookie_file = os.path.join(os.path.dirname(__file__), 'walmart_cookies.txt')
    if os.path.exists(cookie_file):
        with open(cookie_file, 'r') as f:
            return f.read().strip()
    return ""


def search_walmart(query, zip_code='90210'):
    """Search Walmart for products using their GraphQL search API."""
    products = []
    try:
        import urllib.parse
        
        # Build the mWeb (mobile web) query variables - this is key!
        variables = {
            "id": "",
            "dealsId": "",
            "query": query,
            "nudgeContext": "",
            "page": 1,
            "prg": "mWeb",  # Mobile web - less restrictive
            "catId": "",
            "facet": "",
            "sort": "best_match",
            "rawFacet": "",
            "seoPath": "",
            "ps": 40,
            "limit": 40,
            "ptss": "",
            "trsp": "",
            "beShelfId": "",
            "recall_set": "",
            "module_search": "",
            "min_price": "",
            "max_price": "",
            "storeSlotBooked": "",
            "additionalQueryParams": {
                "hidden_facet": None,
                "translation": None,
                "isMoreOptionsTileEnabled": True,
                "isGenAiEnabled": True,
                "rootDimension": "",
                "altQuery": "",
                "selectedFilter": "",
                "neuralSearchSeeAll": False,
                "isModuleArrayReq": False,
                "isLMPBrowsePage": False
            },
            "searchArgs": {
                "query": query,
                "cat_id": "",
                "prg": "mWeb",
                "facet": ""
            },
            "ffAwareSearchOptOut": False,
            "enableDesktopHighlights": False,
            "enableVolumePricing": False,
            "enableCopyBlock": True,
            "enableVariantCount": False,
            "enableSlaBadgeV2": False,
            "fitmentFieldParams": {
                "powerSportEnabled": True,
                "dynamicFitmentEnabled": True,
                "extendedAttributesEnabled": True,
                "fuelTypeEnabled": True
            },
            "fitmentSearchParams": {
                "id": "",
                "dealsId": "",
                "query": query,
                "nudgeContext": "",
                "page": 1,
                "prg": "mWeb",
                "catId": "",
                "facet": "",
                "sort": "best_match",
                "rawFacet": "",
                "seoPath": "",
                "ps": 40,
                "limit": 40,
                "ptss": "",
                "trsp": "",
                "beShelfId": "",
                "recall_set": "",
                "module_search": "",
                "min_price": "",
                "max_price": "",
                "storeSlotBooked": "",
                "additionalQueryParams": {
                    "hidden_facet": None,
                    "translation": None,
                    "isMoreOptionsTileEnabled": True,
                    "isGenAiEnabled": True,
                    "rootDimension": "",
                    "altQuery": "",
                    "selectedFilter": "",
                    "neuralSearchSeeAll": False,
                    "isModuleArrayReq": False,
                    "isLMPBrowsePage": False
                },
                "searchArgs": {
                    "query": query,
                    "cat_id": "",
                    "prg": "mWeb",
                    "facet": ""
                },
                "ffAwareSearchOptOut": False,
                "enableDesktopHighlights": False,
                "enableVolumePricing": False,
                "enableCopyBlock": True,
                "enableVariantCount": False,
                "enableSlaBadgeV2": False,
                "cat_id": "",
                "_be_shelf_id": ""
            },
            "searchParams": {
                "id": "",
                "dealsId": "",
                "query": query,
                "nudgeContext": "",
                "page": 1,
                "prg": "mWeb",
                "catId": "",
                "facet": "",
                "sort": "best_match",
                "rawFacet": "",
                "seoPath": "",
                "ps": 40,
                "limit": 40,
                "ptss": "",
                "trsp": "",
                "beShelfId": "",
                "recall_set": "",
                "module_search": "",
                "min_price": "",
                "max_price": "",
                "storeSlotBooked": "",
                "additionalQueryParams": {
                    "hidden_facet": None,
                    "translation": None,
                    "isMoreOptionsTileEnabled": True,
                    "isGenAiEnabled": True,
                    "rootDimension": "",
                    "altQuery": "",
                    "selectedFilter": "",
                    "neuralSearchSeeAll": False,
                    "isModuleArrayReq": False,
                    "isLMPBrowsePage": False
                },
                "searchArgs": {
                    "query": query,
                    "cat_id": "",
                    "prg": "mWeb",
                    "facet": ""
                },
                "ffAwareSearchOptOut": False,
                "enableDesktopHighlights": False,
                "enableVolumePricing": False,
                "enableCopyBlock": True,
                "enableVariantCount": False,
                "enableSlaBadgeV2": False,
                "cat_id": "",
                "_be_shelf_id": ""
            },
            "enableFashionTopNav": False,
            "enableUnifiedSchema": False,
            "version": "v1",
            "enableRelatedSearches": True,
            "enablePortableFacets": True,
            "enableFacetCount": True,
            "fetchMarquee": True,
            "fetchSkyline": True,
            "fetchGallery": False,
            "fetchSbaTop": True,
            "fetchSBAV1": True,
            "fungibilityEnabled": False,
            "enableAdsPromoData": False,
            "fetchDac": True,
            "tenant": "WM_GLASS",
            "enableMultiSave": False,
            "enableInStoreShelfMessage": False,
            "enableSellerType": False,
            "enableItemRank": False,
            "enableOptimisticWeightUpdate": False,
            "enableAdditionalSearchDepartmentAnalytics": True,
            "enableFulfillmentTagsEnhacements": False,
            "enableRxDrugScheduleModal": False,
            "enablePromoData": True,
            "enableSignInToSeePrice": False,
            "enablePromotionMessages": False,
            "enableItemLimits": False,
            "enableCanAddToList": False,
            "enableIsFreeWarranty": False,
            "enableShopSimilarBottomSheet": False,
            "adsParams": {
                "fungibilityEnabled": False
            },
            "pageType": "SearchPage"
        }
        
        encoded_variables = urllib.parse.quote(json.dumps(variables))
        
        # Walmart's GraphQL search endpoint
        api_url = f"https://www.walmart.com/orchestra/snb/graphql/Search/835a5736d77c2e26b21fdde4b5329fb8392714562650896f4011fc36bad373a8/search?variables={encoded_variables}"
        
        search_page_url = f"https://www.walmart.com/search?q={urllib.parse.quote(query)}"
        
        headers = {
            'accept': 'application/json',
            'accept-language': 'en-US',
            'content-type': 'application/json',
            'referer': search_page_url,
            'sec-ch-ua': '"Google Chrome";v="143", "Chromium";v="143", "Not A(Brand";v="24"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"macOS"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-origin',
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36',
            'wm_mp': 'true',
            'wm_page_url': search_page_url,
            'x-apollo-operation-name': 'Search',
            'x-o-bu': 'WALMART-US',
            'x-o-ccm': 'server',
            'x-o-gql-query': 'query Search',
            'x-o-mart': 'B2C',
            'x-o-platform': 'rweb',
            'x-o-platform-version': 'usweb-1.233.1-8fa35ef97ec792797d7e45f6351dcc65796656b3-2171430r',
            'x-o-segment': 'oaoh',
        }
        
        # Load cookies from file
        cookies = get_walmart_cookies()
        if cookies:
            headers['Cookie'] = cookies
        
        response = requests.get(api_url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            try:
                data = response.json()
                search_result = data.get('data', {}).get('search', {}).get('searchResult', {})
                item_stacks = search_result.get('itemStacks', [])
                
                for stack in item_stacks:
                    # Try both 'itemsV2' and 'items' keys
                    items = stack.get('itemsV2', []) or stack.get('items', [])
                    for item in items[:15]:
                        try:
                            if item and item.get('__typename') == 'Product':
                                name = item.get('name', '')
                                price_info = item.get('priceInfo', {})
                                current_price = price_info.get('currentPrice', {})
                                price = current_price.get('price')
                                canonical_url = item.get('canonicalUrl', '')
                                image_info = item.get('imageInfo', {})
                                image = image_info.get('thumbnailUrl', '')
                                
                                if name and price is not None:
                                    products.append({
                                        'name': name,
                                        'price': float(price),
                                        'url': f"https://www.walmart.com{canonical_url}" if canonical_url and not canonical_url.startswith('http') else canonical_url,
                                        'image': image,
                                        'store': 'Walmart'
                                    })
                        except (KeyError, TypeError):
                            continue
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Walmart JSON parse error: {e}")
        else:
            print(f"Walmart API returned status {response.status_code}")
                    
    except Exception as e:
        print(f"Walmart search error: {e}")
    
    return products


def search_target(query, zip_code='30041'):
    """Search Target for products using their search API."""
    products = []
    try:
        import urllib.parse
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
            'Origin': 'https://www.target.com',
            'Referer': f'https://www.target.com/s?searchTerm={urllib.parse.quote(query)}',
            'sec-ch-ua': '"Google Chrome";v="143", "Chromium";v="143", "Not A(Brand";v="24"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"macOS"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-site',
        }
        
        # Target's Redsky API for search - working parameters from Dec 2024
        api_url = "https://redsky.target.com/redsky_aggregations/v1/web/plp_search_v2"
        
        params = {
            'key': '9f36aeafbe60771e321a7cc95a78140772ab3e96',  # Working API key
            'channel': 'WEB',
            'count': 24,
            'default_purchasability_filter': 'true',
            'include_dmc_dmr': 'true',
            'include_sponsored': 'true',
            'include_review_summarization': 'true',
            'keyword': query,
            'new_search': 'true',
            'offset': 0,
            'page': f'/s/{urllib.parse.quote(query)}',
            'platform': 'desktop',
            'pricing_store_id': '1394',
            'scheduled_delivery_store_id': '1394',
            'spellcheck': 'true',
            'store_ids': '1394,2056,2387,1206,2431',
            'useragent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36',
            'visitor_id': '019B62EDB91F02018B748B7A1D4AF43A',
            'zip': zip_code,
        }
        
        response = requests.get(api_url, headers=headers, params=params, timeout=15)
        
        print(f"Target API status: {response.status_code}")  # Debug
        
        if response.status_code == 200:
            try:
                data = response.json()
                search_response = data.get('data', {}).get('search', {})
                items = search_response.get('products', [])
                
                print(f"Target found {len(items)} items")  # Debug
                
                for item in items[:15]:  # Limit to 15 items
                    try:
                        name = item.get('item', {}).get('product_description', {}).get('title', '')
                        price_data = item.get('price', {})
                        
                        # Get the current price
                        price = price_data.get('current_retail', 0) or price_data.get('reg_retail', 0)
                        
                        tcin = item.get('tcin', '')
                        url = f"https://www.target.com/p/-/A-{tcin}" if tcin else ''
                        
                        # Get image
                        images = item.get('item', {}).get('enrichment', {}).get('images', {})
                        image = images.get('primary_image_url', '')
                        
                        if name and price:
                            products.append({
                                'name': name,
                                'price': float(price),
                                'url': url,
                                'image': image,
                                'store': 'Target'
                            })
                    except (KeyError, TypeError, ValueError) as e:
                        continue
            except json.JSONDecodeError as e:
                print(f"Target JSON decode error: {e}")
        else:
            print(f"Target API returned status {response.status_code}")
            print(f"Response: {response.text[:500]}")  # Debug
                    
    except Exception as e:
        print(f"Target search error: {e}")
    
    return products


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search')
def search():
    query = request.args.get('q', '').strip()
    zip_code = request.args.get('zip', '90210').strip()
    
    # Validate zip code (basic validation - 5 digits)
    if not zip_code or not zip_code.isdigit() or len(zip_code) != 5:
        zip_code = '90210'  # Default fallback
    
    if not query:
        return jsonify({'error': 'Please enter a search term', 'products': []})
    
    # Search stores in parallel with location
    with ThreadPoolExecutor(max_workers=2) as executor:
        walmart_future = executor.submit(search_walmart, query, zip_code)
        target_future = executor.submit(search_target, query, zip_code)
        
        walmart_products = walmart_future.result()
        target_products = target_future.result()
    
    # Track original counts
    original_walmart_count = len(walmart_products)
    original_target_count = len(target_products)
    
    print(f"=== Search Results for '{query}' ===")
    print(f"Walmart returned: {original_walmart_count} products")
    print(f"Target returned: {original_target_count} products")
    
    # Combine all products
    all_products = walmart_products + target_products
    
    # Categorize into exact matches and similar items
    categorized = categorize_products(all_products, query)
    
    # Debug: Check what happened to Target products
    target_in_exact = len([p for p in categorized['exact_matches'] if p.get('store') == 'Target'])
    target_in_similar = len([p for p in categorized['similar_items'] if p.get('store') == 'Target'])
    print(f"Target in exact_matches: {target_in_exact}")
    print(f"Target in similar_items: {target_in_similar}")
    print(f"Total filtered out: {categorized['filtered_out']}")
    
    exact_matches = categorized['exact_matches']
    similar_items = categorized['similar_items']
    total_filtered = categorized['filtered_out']
    
    # Sort exact matches by price
    exact_matches.sort(key=lambda x: x['price'])
    
    # Sort similar items by price
    similar_items.sort(key=lambda x: x['price'])
    
    # Clean up internal scoring fields
    for product in exact_matches + similar_items:
        product.pop('relevance_score', None)
        product.pop('quantity_match', None)
    
    # Count by store for exact matches
    store_counts = {
        'Walmart': len([p for p in exact_matches if p.get('store') == 'Walmart']),
        'Target': len([p for p in exact_matches if p.get('store') == 'Target']),
    }
    
    return jsonify({
        'exact_matches': exact_matches,
        'similar_items': similar_items,
        'store_counts': store_counts,
        'walmart_count': store_counts['Walmart'],
        'target_count': store_counts['Target'],
        'total_exact': len(exact_matches),
        'total_similar': len(similar_items),
        'filtered_out': total_filtered
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)

