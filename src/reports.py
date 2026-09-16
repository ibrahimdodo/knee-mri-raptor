"""The free-text radiology reports: which language each is in.

The competition's reports come from several countries, and language is the best available stand-in for the
site a study came from (scanner fleet, protocol, reporting habits). Detection is deliberately simple and
transparent: script first (Greek, Cyrillic), then letters unique to a language (Turkish, Croatian), then counts of
very common function words.
"""
from __future__ import annotations

import re

LANG_NAMES = {"en": "English", "es": "Spanish", "el": "Greek", "tr": "Turkish", "de": "German", "bg": "Bulgarian",
              "hr": "Croatian", "nl": "Dutch", "fr": "French", "unknown": "unknown"}

_STOPWORDS = {
    "en": [" the ", " of ", " and ", " with ", " is ", " are ", " there "],
    "es": [" de ", " la ", " el ", " con ", " sin ", " del ", " los ", " se "],
    "de": [" der ", " die ", " und ", " mit ", " kein ", " keine ", " des ", " im "],
    "nl": [" van ", " het ", " een ", " met ", " geen ", " op ", " ter "],
    "fr": [" du ", " des ", " avec ", " sans ", " les ", " une ", " est "],
    "hr": [" je ", " su ", " bez ", " se ", " na ", " koljen"],
}


def detect_language(text) -> str:
    t = " " + str(text).lower() + " "
    latin = len(re.findall(r"[a-z]", t))
    greek = len(re.findall(r"[α-ωάέήίόύώϊϋΐΰ]", t))
    cyrillic = len(re.findall(r"[а-яё]", t))
    if greek > latin and greek >= cyrillic:
        return "el"
    if cyrillic > latin:
        return "bg"
    if latin == 0:
        return "unknown"
    if len(re.findall(r"[ıışğ]", t)) >= 3 or " diz " in t:
        return "tr"
    if len(re.findall(r"[čćđž]", t)) >= 3:
        return "hr"
    scores = {k: sum(t.count(w) for w in words) for k, words in _STOPWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "unknown"
