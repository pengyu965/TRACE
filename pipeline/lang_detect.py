"""NLLB-style language detection from Unicode character ranges.

Used by OCR + ASR steps to tag detected text with a coarse `src_lang`. Not a
real language identifier — it's a script range lookup. Good enough to flag the
dominant non-Latin scripts we see in the dataset (Chinese, Japanese, Korean,
Devanagari, Arabic, Thai, Burmese) and to fall back to `eng_Latn`.
"""

# (NLLB code, codepoint_lo, codepoint_hi) — ordered so e.g. Japanese kana are
# checked BEFORE the CJK unified ideographs block (which Japanese also uses).
LANG_RANGES = [
    ("jpn_Jpan", 0x3040, 0x30FF),  # Hiragana + Katakana
    ("kor_Hang", 0xAC00, 0xD7AF),  # Hangul syllables
    ("npi_Deva", 0x0900, 0x097F),  # Devanagari (Nepali / Hindi)
    ("mya_Mymr", 0x1000, 0x109F),  # Myanmar
    ("arb_Arab", 0x0600, 0x06FF),  # Arabic
    ("tha_Thai", 0x0E00, 0x0E7F),  # Thai
    ("zho_Hans", 0x4E00, 0x9FFF),  # CJK unified ideographs
]


def detect_lang(text: str) -> str:
    """Return the first matching NLLB script tag, or 'eng_Latn' as fallback."""
    for c in text:
        cp = ord(c)
        for code, lo, hi in LANG_RANGES:
            if lo <= cp <= hi:
                return code
    return "eng_Latn"
