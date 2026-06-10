"""Unicode normalization aliases shared by the VRTF renderers.

pdfminer-style PDF text extraction emits a handful of code points that are
visually equivalent but miss bitmap templates or break token matching.
Normalize them here so every renderer handles the same substitutions.
"""
from __future__ import annotations


UNICODE_ALIASES: dict[str, str] = {
    '\u00B5': '\u03BC',  # MICRO SIGN -> GREEK MU
    '\u2206': '\u0394',  # INCREMENT -> GREEK DELTA
    '\u2212': '\u002D',  # MINUS SIGN -> HYPHEN-MINUS
    '\u2010': '\u002D',  # HYPHEN -> HYPHEN-MINUS
    '\u2013': '\u002D',  # EN DASH -> HYPHEN-MINUS
    '\u2018': '\u0027',  # LEFT SINGLE QUOTE -> APOSTROPHE
    '\u2019': '\u0027',  # RIGHT SINGLE QUOTE -> APOSTROPHE
    '\u201C': '\u0022',  # LEFT DOUBLE QUOTE -> QUOTATION MARK
    '\u201D': '\u0022',  # RIGHT DOUBLE QUOTE -> QUOTATION MARK
}
