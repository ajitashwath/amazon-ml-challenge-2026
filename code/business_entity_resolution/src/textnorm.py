import re
import unicodedata

DEVA_CONS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n", "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "n",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n", "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "f", "ब": "b", "भ": "bh", "म": "m", "य": "y", "र": "r", "ल": "l", "ळ": "l", "व": "v",
    "श": "sh", "ष": "sh", "स": "s", "ह": "h", "क़": "k", "ख़": "kh", "ग़": "g", "ज़": "z", "ड़": "r", "ढ़": "rh", "फ़": "f",
    "ऱ": "r", "ऴ": "l", "ऩ": "n",
}
DEVA_VOWELS = {
    "अ": "a", "आ": "a", "इ": "i", "ई": "i", "उ": "u", "ऊ": "u", "ऋ": "ri", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "ऑ": "o", "ऍ": "e", "ऎ": "e", "ऒ": "o",
}
DEVA_MATRAS = {
    "ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "u", "ृ": "ri", "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ॉ": "o", "ॅ": "e", "ॆ": "e", "ॊ": "o",
}
DEVA_WORDS = {
    "प्राइवेट": "private", "प्रायवेट": "private", "लिमिटेड": "limited", "प्रा.": "pvt", "लि.": "ltd", "प्रा": "pvt", "लि": "ltd",
    "एलएलपी": "llp", "कंपनी": "company", "एंड": "and", "इंडिया": "india", "श्री": "shri", "इंडस्ट्रीज": "industries",
}
NAME_WORDS = {"एल्एल्पी": "llp", "एल्एल्पि": "llp", "ऎल्ऎल्पि": "llp", "ऎल्ऎल्पी": "llp", "ऎलऎलपी": "llp"}
INDIC_RE = re.compile("[ऀ-ൿ]")
INDIC_WORD_RE = re.compile("[ऀ-ൿ‌‍.]+")
CHILLU = {"ൺ": "ण्", "ൻ": "न्", "ർ": "र्", "ൽ": "ल्", "ൾ": "ळ्"}


def to_devanagari(word):
    word = word.replace("റ്റ", "ട്ട")
    out = []
    for c in word:
        o = ord(c)
        if c in CHILLU:
            out.append(CHILLU[c])
        elif c == "ச":
            out.append("स")
        elif 0x0980 <= o <= 0x0D7F:
            out.append(chr(0x0900 + (o - 0x0980) % 0x80))
        elif c not in "‌‍":
            out.append(c)
    return "".join(out)


def transliterate(word, name=False):
    word = to_devanagari(word).replace("ःप", "फ")
    if word in DEVA_WORDS:
        return DEVA_WORDS[word]
    if name and word in NAME_WORDS:
        return NAME_WORDS[word]
    out = []
    chars = unicodedata.normalize("NFC", word.replace(".", ""))
    i = 0
    while i < len(chars):
        c = chars[i]
        nxt = chars[i + 1] if i + 1 < len(chars) else ""
        if c + nxt in DEVA_CONS:
            c, i = c + nxt, i + 1
            nxt = chars[i + 1] if i + 1 < len(chars) else ""
        if c in DEVA_CONS:
            out.append(DEVA_CONS[c])
            if nxt == "़":
                i += 1
                nxt = chars[i + 1] if i + 1 < len(chars) else ""
            if nxt not in DEVA_MATRAS and nxt != "्" and i + 1 < len(chars):
                out.append("a")
        elif c in DEVA_VOWELS:
            out.append(DEVA_VOWELS[c])
        elif c in DEVA_MATRAS:
            out.append(DEVA_MATRAS[c])
        elif c in "ंँ":
            if i + 1 < len(chars):
                out.append("n")
        elif c == "ः":
            out.append("h")
        elif "०" <= c <= "९":
            out.append(str(ord(c) - ord("०")))
        i += 1
    return "".join(out)


def deva_to_latin(text, name=False):
    return INDIC_WORD_RE.sub(lambda m: transliterate(m.group(0), name) if INDIC_RE.search(m.group(0)) else m.group(0), text)


def strip_accents(text):
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


LEGAL = {
    "inc", "incorporated", "llc", "llp", "lp", "ltd", "limited", "pvt", "private", "corp", "corporation", "co", "company",
    "plc", "pc", "pllc", "pa", "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "cie", "public",
}
NAME_STOP = {
    "the", "and", "of", "et", "de", "du", "des", "la", "le", "les", "d", "l", "a", "an", "sri", "shri", "sree", "shree",
    "dr", "m", "s", "ms", "mr", "mrs", "www", "com", "net", "org", "in", "fr", "bis", "smt", "shrimati", "messrs",
}
LOOKALIKE = str.maketrans("013456", "oleasg")
GLUE_SUFFIX = re.compile(r"(?:private|limited|pvt|ltd|com|llp|inc|llc|corp|company)$")
GLUE_PREFIX = re.compile(r"^(?:sree|shree|shri|sri)")
KEEP_SHORT = {"la", "le", "de", "du", "des", "l", "d", "a", "an", "of", "in", "s", "m"}
DBA_RE =re.compile(r"\b(?:d\s*/?\s*b\s*/?\s*a|dba|fka|f\s*/\s*k\s*/\s*a|aka|a\s*/\s*k\s*/\s*a|trading as|t\s*/\s*a)\b", re.I)
DOMAIN_RE = re.compile(r"^(?:www\.)?([a-z0-9\-]+)\.(?:com|net|org|co\.in|in|fr|co|biz|info|us)$")
NON_ALNUM = re.compile(r"[^a-z0-9]+")
SKEL_RULES = [("tion", "shan"), ("sion", "shan"), ("ture", "char"), ("ph", "f"), ("ck", "k"), ("x", "ks")]
SOFT_C = re.compile(r"c(?=[eiy])")
SOFT_G = re.compile(r"g(?=[eiy])")
SKEL_MAP = str.maketrans({"c": "k", "q": "k", "z": "j", "w": "p", "v": "p", "f": "p", "g": "k", "b": "p", "d": "t"})
LEGAL_SKEL = {"prvt", "prpt", "lmtt", "lmt", "kmpn", "pvt", "ltt"}
SKEL_DROP = re.compile(r"[aeiouyh]")
REPEAT = re.compile(r"(.)\1+")

ADDR_MAP = {
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue", "avn": "avenue", "blvd": "boulevard",
    "bd": "boulevard", "bvd": "boulevard", "dr": "drive", "ln": "lane", "ct": "court", "cir": "circle", "pl": "place",
    "pkwy": "parkway", "hwy": "highway", "ter": "terrace", "trl": "trail", "sq": "square", "r": "rue", "ch": "chemin",
    "imp": "impasse", "rte": "route", "ngr": "nagar", "mkt": "market", "apts": "apartment", "apt": "apartment",
    "n": "north", "s": "south", "e": "east", "w": "west", "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
    "fl": "floor", "flr": "floor", "ste": "suite", "bldg": "building", "plto": "plot", "second": "2nd", "first": "1st", "third": "3rd",
}
ADDR_STOP = {
    "null", "none", "na", "n", "a", "no", "door", "near", "opp", "opposite", "unit", "suite", "floor", "apartment", "po", "box",
    "pmb", "de", "du", "des", "la", "le", "les", "d", "l", "bis", "ter", "street", "road", "avenue", "boulevard", "drive",
    "lane", "court", "circle", "place", "way", "rue", "chemin", "impasse", "allee", "route", "north", "south", "east", "west",
    "city", "of", "the", "and", "building", "plot", "house", "office", "shop", "ground", "first", "g", "f",
    "nr", "basement", "gf", "ff", "sf", "self", "behind", "main", "cross", "stage", "phase", "sector", "block", "wing", "flat",
    "room", "rno", "vill", "village", "post", "dist", "district", "tal", "taluk", "mandal", "mr", "mrs", "sri", "shri",
    "tower", "complex", "floor", "market", "nagar", "colony", "b", "c", "e", "ste", "fax",
}
STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca", "colorado": "co", "connecticut": "ct",
    "delaware": "de", "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in",
    "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
    "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny", "north carolina": "nc", "north dakota": "nd",
    "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc",
}
STATE_RE = re.compile(r"\b(" + "|".join(sorted(STATES, key=len, reverse=True)) + r")\b")


def skeleton(token):
    s = token[1:] if token[:2] in ("yu", "ye", "yo") else token
    for a, b in SKEL_RULES:
        s = s.replace(a, b)
    s = SOFT_C.sub("s", s)
    s = SOFT_G.sub("j", s).translate(SKEL_MAP)
    s = ("a" if s[:1] in "aeiou" else s[:1]) + SKEL_DROP.sub("", s[1:])
    s = REPEAT.sub(r"\1", s)
    t = s.rstrip("sj")
    return t if len(t) >= 3 else s


def unglue(token):
    while True:
        cut = GLUE_SUFFIX.sub("", token)
        if cut == token or len(cut) < 4:
            break
        token = cut
    cut = GLUE_PREFIX.sub("", token)
    return cut if len(cut) >= 4 else token


def normalize_name(raw):
    text = raw or ""
    if INDIC_RE.search(text):
        text = deva_to_latin(text, True)
    text = strip_accents(text).lower().strip()
    parts = DBA_RE.split(text)
    if len(parts) > 1 and parts[-1].strip():
        text = parts[-1]
    text = text.replace("&", " and ").replace("+", " plus ")
    m = DOMAIN_RE.match(text.strip(" -<>[]()"))
    if m:
        text = m.group(1)
    text = re.sub(r"\b([a-z])\.(?=[a-z]\b)", r"\1", text)
    text = re.sub(r"(?<=\b[a-z])\.(?=[a-z]\.)", "", text)
    tokens = []
    for t in NON_ALNUM.sub(" ", text.replace(".", "")).split():
        if sum(c.isdigit() for c in t) == 1 and sum(c.isalpha() for c in t) >= 3:
            t = t.translate(LOOKALIKE)
        if len(t) >= 8:
            t = unglue(t)
        if t not in tokens:
            tokens.append(t)
    tokens = [t for t in tokens if not t.isdigit()] or tokens
    tokens = [t for t in tokens if t not in LEGAL and skeleton(t) not in LEGAL_SKEL] or tokens
    core = [t for t in tokens if t not in NAME_STOP]
    if len(core) < 2:
        core = [t for t in tokens if t not in NAME_STOP or t in KEEP_SHORT] or tokens
    return " ".join(core), "".join(core), " ".join(skeleton(t) for t in core)


def normalize_address(raw):
    text = raw or ""
    if INDIC_RE.search(text):
        text = INDIC_WORD_RE.sub(" ", text)
    text = strip_accents(text).lower()
    text = re.sub(r"<\s*null\s*>|\bn/a\b", " ", text)
    text = STATE_RE.sub(lambda m: STATES[m.group(1)], text)
    text = re.sub(r"(\d)(?!(?:st|nd|rd|th)\b)([a-z])", r"\1 \2", text)
    text = re.sub(r"([a-z])(\d)", r"\1 \2", text)
    tokens = []
    for t in NON_ALNUM.sub(" ", text).split():
        if t.isdigit():
            t = t.lstrip("0") or "0"
        tokens.append(ADDR_MAP.get(t, t))
    numbers = [t for t in tokens if t.isdigit()]
    words = [t for t in tokens if not t.isdigit() and t not in ADDR_STOP and len(t) > 1]
    street = ""
    seen_num = False
    for t in tokens:
        if t.isdigit():
            seen_num = True
        elif seen_num and t not in ADDR_STOP and len(t) > 1:
            street = t
            break
    if not street and words:
        street = words[0]
    return " ".join(tokens), " ".join(numbers), " ".join(words), street
