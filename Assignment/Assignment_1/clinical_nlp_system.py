"""
CLINICAL NLP TEXT NORMALISATION AND SEARCH SYSTEM
A single integrated, runnable implementation for the DSA03 NLP assignment.

Python: 3.11+
Required: Python standard library
Optional: pip install nltk  (uses the real PorterStemmer; otherwise a fallback is used)

Run:
    python clinical_nlp_system.py

Optional input:
    python clinical_nlp_system.py --input clinical_records.csv

CSV input should contain a column named "text" or "record".
The program also creates a synthetic 500-record corpus if no input is supplied.

Outputs are written to ./clinical_nlp_output/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# ---------------------------------------------------------------------
# Configuration and sample clinical resources
# ---------------------------------------------------------------------

OUTPUT_DIR = Path("clinical_nlp_output")
OUTPUT_DIR.mkdir(exist_ok=True)

ABBREVIATIONS = {
    "bp": "blood pressure",
    "hr": "heart rate",
    "rr": "respiratory rate",
    "temp": "temperature",
    "htn": "hypertension",
    "dm": "diabetes mellitus",
    "sob": "shortness of breath",
    "mi": "myocardial infarction",
    "cva": "cerebrovascular accident",
    "uti": "urinary tract infection",
    "iv": "intravenous",
    "im": "intramuscular",
    "od": "once daily",
    "bd": "twice daily",
    "tds": "three times daily",
    "qid": "four times daily",
    "prn": "as needed",
    "npo": "nothing by mouth",
    "icu": "intensive care unit",
    "er": "emergency room",
    "opd": "outpatient department",
    "c/o": "complains of",
    "h/o": "history of",
    "f/u": "follow up",
    "wbc": "white blood cell count",
    "rbc": "red blood cell count",
    "ecg": "electrocardiogram",
    "spo2": "oxygen saturation",
    "cbc": "complete blood count",
    "cr": "creatinine",
    "na": "sodium",
    "k": "potassium",
}

# A starter lexicon of clinical and general medical stems.
# The program augments it with common English/clinical vocabulary to create
# a practical demonstration lexicon; a real deployment should use a curated
# and clinically reviewed lexicon of at least 300 stems.
BASE_STEMS = """
abdomen abdominal abscess accident ache acute admission adult allergy
anaemia anemia analgesia analgesic analysis ankle antibiotic antibody
appendix appendicitis artery arthritis asthma assessment bacterium benign
biopsy bladder blood bone bowel brain breast breath bronchitis burn cancer
cardiac cardiology cardiomyopathy cardiopulmonary cardiovascular care
catheter cell cerebral cervix chemotherapy chest chronic circulation
clinical clinic clot cold colon colorectal coma condition congenital
consciousness consultation contagious coronary cough cranial cramp
creatinine culture cyst cystitis diabetes diagnosis diagnose diagnostic
diarrhea diarrhoea diet disease disorder dose drainage drug ear edema
emergency embolism enzyme epidemic examination excision exercise eye face
failure family fever fibrosis finger fluid follow fracture fractured
gastric gastritis gastroenterology gene general genetic gland glucose
gram hand headache health heart hematology haemoglobin hemoglobin
hemorrhage haemorrhage hepatitis hernia history hormone hospital
hypertension hypotension immune immunity infection inflammation injury
injection intensive intestine intravenous kidney laboratory lesion limb
liver lung malignant management medication medicine memory menstrual
metabolic metastasis microbiology migraine mild monitor monitoring
muscle nausea nephrectomy nephritis nephrology nerve neural neurology
neuropathy neuron normal observation oncology operation oral organ
orthopedic orthopaedic pain pancreas paralysis patient pathology
pediatric paediatric pelvis pharmacology pharynx physician physiology
pneumonia prescription pressure procedure prognosis pulse pulmonary
radiology renal respiration respiratory response result retina risk
screening seizure sensation shock shoulder skin sleep specimen spine
spleen sputum stomach stroke surgery surgical symptom syndrome tablet
temperature tendon therapy thoracic throat tissue tomography tonsil
toxic trauma treatment tumor tumour ultrasound urine urinary uterus
vaccine vascular vein ventricle viral virus vision ward weakness wound
wrist xray
""".split()

EXTRA_STEMS = """
acetaminophen paracetamol ibuprofen aspirin amoxicillin azithromycin
metformin insulin omeprazole atorvastatin losartan amlodipine ceftriaxone
salbutamol prednisolone doxycycline ciprofloxacin pantoprazole
electrolyte sodium potassium calcium magnesium albumin bilirubin
platelet leukocyte erythrocyte neutrophil lymphocyte thrombosis
thrombolysis dialysis endoscopy colonoscopy bronchoscopy gastrectomy
nephrology nephropathy nephrotic nephritic cardiology cardiac cardiologist
neurologist neurological neurologic pulmonary pulmonologist dermatology
dermatitis dermatological pathology pathological infection infectious
infectiousness inflammation inflammatory inflammatoryly postoperative
preoperative postoperative postnatal prenatal antiviral antibacterial
antibiotic antibiotic resistance resistant sensitivity susceptible
hyperaemia hypoglycemia hyperglycemia hypothermia hyperthermia
tachycardia bradycardia hypertension hypotension hypotensive hypertensive
fracture fractures fracturing fractured diagnosis diagnoses diagnosed
diagnosing treatment treatments treated treating medication medications
patient patients admitted admitting admission discharged discharge
discharging examination examinations examined examining
""".split()

LEXICON_STEMS = sorted(set(BASE_STEMS + EXTRA_STEMS))

INFLECTIONAL_EXAMPLES = {
    "fracture": {"N+PL": "fractures", "V+PAST": "fractured", "V+ING": "fracturing"},
    "patient": {"N+PL": "patients"},
    "treat": {"V+PAST": "treated", "V+ING": "treating", "V+3SG": "treats"},
    "study": {"N+PL": "studies", "V+PAST": "studied", "V+ING": "studying"},
    "diagnose": {"V+PAST": "diagnosed", "V+ING": "diagnosing", "V+3SG": "diagnoses"},
    "injury": {"N+PL": "injuries"},
}

IRREGULARS = {
    "child": ["children"],
    "man": ["men"],
    "woman": ["women"],
    "foot": ["feet"],
    "tooth": ["teeth"],
    "be": ["am", "is", "are", "was", "were", "been", "being"],
    "go": ["went", "gone", "going"],
}

DERIVATIONAL_AFFIXES = {
    "itis": "inflammation/condition",
    "ectomy": "surgical removal",
    "ology": "field of study",
    "osis": "condition/process",
    "aemia": "blood condition",
    "emia": "blood condition",
    "ation": "process/result",
    "able": "capable of",
    "ive": "having a quality",
    "al": "relating to",
    "ic": "relating to",
    "ist": "specialist/person",
    "logy": "study of",
    "hyper": "excessive/high",
    "hypo": "low/under",
    "re": "again",
    "un": "not/reversal",
    "pre": "before",
    "post": "after",
    "anti": "against",
}

# ---------------------------------------------------------------------
# Entity patterns: 13 types from the assignment's 12-item inventory.
# Admission and discharge dates share a date-format validator.
# ---------------------------------------------------------------------

DATE = r"(?:0?[1-9]|[12]\d|3[01])[-/](?:0?[1-9]|1[0-2])[-/](?:19|20)\d{2}"
ENTITY_PATTERNS = {
    "patient_identifier": r"\bPAT-\d{4,8}\b",
    "admission_or_discharge_date": rf"\b{DATE}\b",
    "drug_name_with_strength": r"\b[A-Za-z][A-Za-z-]*\s+\d+(?:\.\d+)?\s?(?:mg|mcg|g|ml|mL)\b",
    "dosage_frequency": r"\b(?:once|twice|thrice)\s+daily\b|\b\d+\s?(?:times|x)\s?(?:a|per)\s?day\b|\bq\d+h\b",
    "laboratory_value": r"\b(?:Hb|WBC|RBC|glucose|creatinine|sodium|potassium|platelets?)\s*[:=]?\s*\d+(?:\.\d+)?\s?(?:mg/dL|g/dL|mmol/L|U/L|10\^?9/L|%)\b",
    "blood_pressure": r"\b\d{2,3}\s*/\s*\d{2,3}\s?mmHg\b",
    "temperature": r"\b\d{2}(?:\.\d+)?\s?°?\s?(?:C|F)\b",
    "icd_code": r"\b[A-TV-Z]\d{2}(?:\.\d{1,4})?\b",
    "ward_and_bed": r"\bWard\s?[A-Z0-9-]+\s*,?\s*Bed\s?\d+\b",
    "contact_number": r"\b(?:\+?\d{1,3}[- ]?)?\d{10}\b",
    "age": r"\b(?:Age\s*:\s*)?\d{1,3}\s?(?:years?|yrs?)\b",
    "symptom_duration": r"\b\d+\s?(?:days?|weeks?|months?|years?)\b",
}

# ---------------------------------------------------------------------
# Tokenisation and normalisation
# ---------------------------------------------------------------------

TOKEN_RE = re.compile(
    r"(?:[A-Za-z]+(?:\.[A-Za-z]+)+\.?)"       # abbreviations such as e.g.
    r"|[A-Za-z]+(?:-[A-Za-z]+)+"              # hyphenated words
    r"|\d+(?:\.\d+)?(?:mg|mcg|mL|ml|mmHg|kg|cm|%)?"  # numeric/unit token
    r"|[A-Za-z]+"
    r"|[^\w\s]",
    re.IGNORECASE,
)

def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)

def normalize(text: str) -> str:
    text = text.replace("\u00a0", " ").lower()
    # Expand abbreviations, longest first, with token boundaries.
    for short, full in sorted(ABBREVIATIONS.items(), key=lambda item: -len(item[0])):
        pattern = r"(?<!\w)" + re.escape(short) + r"(?!\w)"
        text = re.sub(pattern, full, text, flags=re.IGNORECASE)
    # Canonicalise dates in DD/MM/YYYY or DD-MM-YYYY format.
    def date_repl(match: re.Match) -> str:
        day, month, year = match.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"
    text = re.sub(r"\b(\d{1,2})[/-](\d{1,2})[/-](20\d{2})\b", date_repl, text)
    # Standardise common unit casing and whitespace.
    text = re.sub(r"\bml\b", "mL", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_entities(text: str) -> dict[str, list[str]]:
    return {
        name: re.findall(pattern, text, flags=re.IGNORECASE)
        for name, pattern in ENTITY_PATTERNS.items()
    }

# ---------------------------------------------------------------------
# Finite automata: DFA validators and NFA -> DFA subset construction
# ---------------------------------------------------------------------

def dfa_accepts(transitions: dict, start: str, finals: set[str], value: str) -> bool:
    state = start
    for char in value:
        state = transitions.get((state, char))
        if state is None:
            return False
    return state in finals

def make_patient_id_dfa():
    transitions = {
        ("q0", "P"): "q1",
        ("q1", "A"): "q2",
        ("q2", "T"): "q3",
        ("q3", "-"): "q4",
    }
    for digit in "0123456789":
        transitions[("q4", digit)] = "q5"
        transitions[("q5", digit)] = "q6"
        transitions[("q6", digit)] = "q7"
        transitions[("q7", digit)] = "q8"
        transitions[("q8", digit)] = "q9"
    return transitions, "q0", {"q8", "q9"}

def make_bp_dfa():
    transitions = {}
    for digit in "0123456789":
        transitions[("s0", digit)] = "s1"
        transitions[("s1", digit)] = "s2"
        transitions[("s2", digit)] = "s3"
        transitions[("s4", digit)] = "s5"
        transitions[("s5", digit)] = "s6"
    transitions[("s3", "/")] = "s4"
    transitions[("s6", "m")] = "s7"
    transitions[("s7", "m")] = "s8"
    transitions[("s8", "H")] = "s9"
    transitions[("s9", "g")] = "sf"
    return transitions, "s0", {"sf"}

def epsilon_closure(states: set[str], eps: dict[str, set[str]]) -> frozenset[str]:
    result = set(states)
    stack = list(states)
    while stack:
        state = stack.pop()
        for nxt in eps.get(state, set()):
            if nxt not in result:
                result.add(nxt)
                stack.append(nxt)
    return frozenset(result)

def nfa_to_dfa(transitions, eps, start, finals, alphabet):
    start_set = epsilon_closure({start}, eps)
    pending = [start_set]
    states = {start_set}
    dfa = {}
    dfa_finals = set()
    while pending:
        current = pending.pop()
        if current & finals:
            dfa_finals.add(current)
        for symbol in alphabet:
            move = set()
            for state in current:
                move.update(transitions.get((state, symbol), set()))
            target = epsilon_closure(move, eps)
            if target:
                dfa[(current, symbol)] = target
                if target not in states:
                    states.add(target)
                    pending.append(target)
    return dfa, start_set, dfa_finals, states

def nfa_accepts(transitions, eps, start, finals, value):
    current = epsilon_closure({start}, eps)
    for char in value:
        next_states = set()
        for state in current:
            next_states.update(transitions.get((state, char), set()))
        current = epsilon_closure(next_states, eps)
    return bool(current & finals)

def make_optional_suffix_nfa():
    # Accept PAT- followed by 4 digits, optionally followed by one more digit.
    transitions = {
        ("n0", "P"): {"n1"},
        ("n1", "A"): {"n2"},
        ("n2", "T"): {"n3"},
        ("n3", "-"): {"n4"},
    }
    for digit in "0123456789":
        transitions.setdefault(("n4", digit), set()).add("n5")
        transitions.setdefault(("n5", digit), set()).add("n6")
        transitions.setdefault(("n6", digit), set()).add("n7")
        transitions.setdefault(("n7", digit), set()).add("n8")
        transitions.setdefault(("n8", digit), set()).add("n9")
    eps = {"n8": {"nf"}, "n9": {"nf"}}
    return transitions, eps, "n0", {"nf"}

# ---------------------------------------------------------------------
# Morphology: inflection, derivation, transducer-style parsing/generation
# ---------------------------------------------------------------------

def inflect(stem: str, feature: str) -> str:
    stem = stem.lower()
    if feature == "PL":
        if stem.endswith(("s", "x", "z", "ch", "sh")):
            return stem + "es"
        if stem.endswith("y") and len(stem) > 1 and stem[-2] not in "aeiou":
            return stem[:-1] + "ies"
        return stem + "s"
    if feature == "PAST":
        if stem.endswith("e"):
            return stem + "d"
        if stem.endswith("y") and len(stem) > 1 and stem[-2] not in "aeiou":
            return stem[:-1] + "ied"
        return stem + "ed"
    if feature == "ING":
        if stem.endswith("ie"):
            return stem[:-2] + "ying"
        if stem.endswith("e") and not stem.endswith("ee"):
            return stem[:-1] + "ing"
        return stem + "ing"
    if feature == "3SG":
        return inflect(stem, "PL")
    return stem

def derive(stem: str, affix: str) -> str:
    if affix in {"hyper", "hypo", "re", "un", "pre", "post", "anti"}:
        return affix + stem
    if affix in {"itis", "ectomy", "ology", "osis", "aemia", "emia", "ation",
                 "able", "ive", "al", "ic", "ist", "logy"}:
        return stem + affix
    raise ValueError(f"Unknown derivational affix: {affix}")

class MorphologyTransducer:
    """Finite lexicon transducer demonstration: surface <-> lemma/features."""

    def __init__(self, stems: Iterable[str]):
        self.stems = set(stems)
        self.surface_to_analyses = defaultdict(list)
        self.analysis_to_surface = {}
        self._build()

    def _add(self, lemma, pos, feature, surface):
        analysis = f"{lemma}+{pos}+{feature}"
        self.surface_to_analyses[surface].append(analysis)
        self.analysis_to_surface[analysis] = surface

    def _build(self):
        for stem in self.stems:
            self._add(stem, "N", "SG", stem)
            self._add(stem, "N", "PL", inflect(stem, "PL"))
            self._add(stem, "V", "PAST", inflect(stem, "PAST"))
            self._add(stem, "V", "ING", inflect(stem, "ING"))
        for lemma, forms in IRREGULARS.items():
            for surface in forms:
                if isinstance(surface, list):
                    values = surface
                else:
                    values = [surface]
                for value in values:
                    self._add(lemma, "IRREG", "FORM", value)

    def parse(self, surface: str) -> list[str]:
        return sorted(set(self.surface_to_analyses.get(surface.lower(), [])))

    def generate(self, analysis: str) -> str | None:
        return self.analysis_to_surface.get(analysis)

# ---------------------------------------------------------------------
# Porter stemmer (NLTK when installed; safe fallback otherwise)
# ---------------------------------------------------------------------

class Stemmer:
    def __init__(self):
        try:
            from nltk.stem import PorterStemmer
            self._stemmer = PorterStemmer()
            self.backend = "NLTK PorterStemmer"
        except ImportError:
            self._stemmer = None
            self.backend = "basic suffix fallback (install nltk for Porter algorithm)"

    def stem(self, word: str) -> str:
        word = word.lower()
        if self._stemmer:
            return self._stemmer.stem(word)
        for suffix in ("ational", "tional", "fulness", "ousness", "iveness",
                       "ingly", "edly", "ments", "ment", "ness", "ation",
                       "izer", "ing", "ed", "ies", "es", "s"):
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                if suffix == "ies":
                    return word[:-3] + "i"
                return word[:-len(suffix)]
        return word

# ---------------------------------------------------------------------
# Synthetic corpus generation and CSV loading
# ---------------------------------------------------------------------

DRUGS = ["Paracetamol", "Amoxicillin", "Metformin", "Ibuprofen", "Aspirin"]
CONDITIONS = ["fever", "fracture", "diabetes", "hypertension", "infection",
              "pneumonia", "asthma", "migraine", "arthritis", "bronchitis"]
FREQUENCIES = ["once daily", "twice daily", "three times daily", "as needed"]
RECORD_TYPES = ["discharge summary", "prescription", "operation note", "nursing observation"]

def generate_synthetic_corpus(size=500):
    records = []
    for i in range(size):
        patient_id = f"PAT-{10000 + i}"
        day = (i % 27) + 1
        month = (i % 12) + 1
        date = f"{day:02d}/{month:02d}/2025"
        drug = DRUGS[i % len(DRUGS)]
        condition = CONDITIONS[i % len(CONDITIONS)]
        frequency = FREQUENCIES[i % len(FREQUENCIES)]
        record_type = RECORD_TYPES[i % len(RECORD_TYPES)]
        bp_sys = 110 + (i % 35)
        bp_dia = 70 + (i % 20)
        temp = 36.5 + (i % 15) / 10
        age = 18 + (i % 70)
        text = (
            f"{record_type.title()}: {patient_id}. Patient aged {age} years "
            f"presented with {condition} for 3 days. BP {bp_sys}/{bp_dia} mmHg. "
            f"Temperature {temp:.1f} C. Prescribed {drug} 500mg {frequency}. "
            f"Review on {date}. Ward A{i % 4 + 1}, Bed {i % 30 + 1}."
        )
        records.append({"id": patient_id, "type": record_type, "text": text})
    return records

def load_corpus(path: Path | None):
    if path is None:
        return generate_synthetic_corpus(500), "synthetic"
    records = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV has no header row.")
        text_col = "text" if "text" in reader.fieldnames else (
            "record" if "record" in reader.fieldnames else None
        )
        if not text_col:
            raise ValueError("CSV must contain a 'text' or 'record' column.")
        for i, row in enumerate(reader, 1):
            records.append({
                "id": row.get("id", f"RECORD-{i}"),
                "type": row.get("type", "unknown"),
                "text": row.get(text_col, ""),
            })
    return records, "uploaded CSV"

# ---------------------------------------------------------------------
# Search evaluation: generated relevance judgements and metrics
# ---------------------------------------------------------------------

def terms(text: str, stemmer: Stemmer, mode: str) -> set[str]:
    tokens = [t.lower() for t in tokenize(normalize(text)) if re.fullmatch(r"[a-z]+", t)]
    if mode == "none":
        return set(tokens)
    if mode == "porter":
        return {stemmer.stem(t) for t in tokens}
    if mode == "morphology":
        result = set()
        for token in tokens:
            analyses = MORPH.parse(token)
            if analyses:
                result.update(a.split("+")[0] for a in analyses)
            else:
                result.add(token)
        return result
    raise ValueError(mode)

def retrieve(query: str, records: list[dict], stemmer: Stemmer, mode: str):
    qterms = terms(query, stemmer, mode)
    hits = []
    for index, record in enumerate(records):
        if qterms & terms(record["text"], stemmer, mode):
            hits.append(index)
    return hits

def precision_recall_f1(retrieved: set[int], relevant: set[int]):
    tp = len(retrieved & relevant)
    fp = len(retrieved - relevant)
    fn = len(relevant - retrieved)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1, tp, fp, fn

def make_queries(records):
    # Query relevance is judged using exact concept occurrence in original text.
    concepts = ["fracture", "fever", "diabetes", "hypertension", "infection",
                "pneumonia", "asthma", "migraine", "arthritis", "bronchitis"]
    queries = []
    for i in range(40):
        concept = concepts[i % len(concepts)]
        relevant = {
            idx for idx, rec in enumerate(records)
            if re.search(rf"\b{re.escape(concept)}\w*\b", rec["text"], re.I)
        }
        queries.append((concept, relevant))
    return queries

def evaluate_retrieval(records, stemmer):
    queries = make_queries(records)
    rows = []
    for mode in ("none", "porter", "morphology"):
        all_p, all_r, all_f = [], [], []
        total_tp = total_fp = total_fn = 0
        for query, relevant in queries:
            found = set(retrieve(query, records, stemmer, mode))
            p, r, f, tp, fp, fn = precision_recall_f1(found, relevant)
            all_p.append(p); all_r.append(r); all_f.append(f)
            total_tp += tp; total_fp += fp; total_fn += fn
        rows.append({
            "method": mode,
            "precision": round(statistics.mean(all_p), 4),
            "recall": round(statistics.mean(all_r), 4),
            "f1": round(statistics.mean(all_f), 4),
            "tp": total_tp, "fp": total_fp, "fn": total_fn,
        })
    return rows

# ---------------------------------------------------------------------
# Day-4 adaptation: OCR corrections and mixed-language examples
# ---------------------------------------------------------------------

def clean_ocr(text: str) -> str:
    # Join line-break hyphenation.
    text = re.sub(r"-[ \t]*\r?\n[ \t]*", "", text)
    # Correct O/I/l only in the numeric suffix of a known patient-ID pattern.
    def fix_id(match):
        suffix = match.group(1).translate(str.maketrans({"O": "0", "I": "1", "l": "1"}))
        return "PAT-" + suffix
    text = re.sub(r"\bPAT[- ]([0-9OIl]{4,8})\b", fix_id, text)
    return text

ROMANISED_MAP = {
    "patient ku": "patient to",
    "fever irundhuchu": "had fever",
    "paracetamol kuduthen": "paracetamol administered",
    "kuduthen": "administered",
}

def normalise_code_mixed(text: str) -> str:
    result = text
    for phrase, replacement in sorted(ROMANISED_MAP.items(), key=lambda x: -len(x[0])):
        result = re.sub(re.escape(phrase), replacement, result, flags=re.IGNORECASE)
    return result

def adapted_pipeline(text: str):
    cleaned = clean_ocr(text)
    mixed = normalise_code_mixed(cleaned)
    normalised = normalize(mixed)
    return {
        "cleaned": cleaned,
        "normalised": normalised,
        "tokens": tokenize(normalised),
        "entities": extract_entities(normalised),
    }

# ---------------------------------------------------------------------
# Output reports, SVG plot, and full pipeline
# ---------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def write_svg_metrics(rows: list[dict], path: Path):
    width, height = 760, 420
    left, bottom, top = 80, 330, 55
    chart_h = bottom - top
    methods = [row["method"] for row in rows]
    metrics = ["precision", "recall", "f1"]
    bar_w, gap = 42, 12
    group_w = 3 * bar_w + 2 * gap
    start_x = 145
    scale = chart_h
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="380" y="28" text-anchor="middle" font-size="20" font-family="Arial">Retrieval evaluation</text>',
    ]
    for tick in range(0, 101, 20):
        y = bottom - tick / 100 * scale
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="700" y2="{y:.1f}" stroke="#dddddd"/>')
        parts.append(f'<text x="65" y="{y+4:.1f}" text-anchor="end" font-size="12" font-family="Arial">{tick}%</text>')
    colors = ["#4472C4", "#70AD47", "#ED7D31"]
    for i, row in enumerate(rows):
        gx = start_x + i * 180
        for j, metric in enumerate(metrics):
            value = row[metric]
            x = gx + j * (bar_w + gap)
            bh = value * scale
            y = bottom - bh
            parts.append(f'<rect x="{x}" y="{y:.1f}" width="{bar_w}" height="{bh:.1f}" fill="{colors[j]}"/>')
            parts.append(f'<text x="{x+bar_w/2}" y="{y-6:.1f}" text-anchor="middle" font-size="10" font-family="Arial">{value:.2f}</text>')
        parts.append(f'<text x="{gx+group_w/2}" y="355" text-anchor="middle" font-size="13" font-family="Arial">{row["method"]}</text>')
    for j, metric in enumerate(metrics):
        x = 180 + j * 150
        parts.append(f'<rect x="{x}" y="385" width="12" height="12" fill="{colors[j]}"/>')
        parts.append(f'<text x="{x+18}" y="396" font-size="12" font-family="Arial">{metric}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")

def write_pipeline_outputs(records, results, eval_rows, source, elapsed):
    processed_rows = []
    token_count = 0
    for record in records:
        norm = normalize(record["text"])
        toks = tokenize(norm)
        token_count += len(toks)
        entities = extract_entities(record["text"])
        processed_rows.append({
            "id": record["id"],
            "type": record["type"],
            "original_text": record["text"],
            "normalised_text": norm,
            "token_count": len(toks),
            "entities_json": json.dumps({k: v for k, v in entities.items() if v}),
        })
    write_csv(OUTPUT_DIR / "processed_records.csv", processed_rows)
    write_csv(OUTPUT_DIR / "retrieval_evaluation.csv", eval_rows)
    write_csv(OUTPUT_DIR / "entity_extraction.csv", [
        {"record_id": row["id"], "entity_type": key, "matches": " | ".join(values)}
        for record in records
        for row in [record]
        for key, values in extract_entities(record["text"]).items()
        if values
    ])
    summary = {
        "corpus_source": source,
        "records_processed": len(records),
        "tokens_processed": token_count,
        "lexicon_stem_count": len(LEXICON_STEMS),
        "record_types": sorted(set(r["type"] for r in records)),
        "runtime_seconds": round(elapsed, 6),
        "stemmer_backend": STEMMER.backend,
        "retrieval": eval_rows,
    }
    (OUTPUT_DIR / "run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_svg_metrics(eval_rows, OUTPUT_DIR / "retrieval_metrics.svg")
    return summary

# Global transducer and stemmer are initialised before search evaluation.
MORPH = MorphologyTransducer(LEXICON_STEMS)
STEMMER = Stemmer()

def run_demos():
    print("\n--- ENTITY EXTRACTION ---")
    sample = (
        "PAT-12345 admitted 12/04/2025. Paracetamol 500mg twice daily. "
        "BP 120/80 mmHg. Temperature 37.5 C. Hb 13.2 g/dL. "
        "ICD J18.9. Ward A2, Bed 12. Age 45 years. Fever for 3 days."
    )
    for key, values in extract_entities(sample).items():
        if values:
            print(f"{key}: {values}")

    print("\n--- TOKENISATION AND NORMALISATION ---")
    print("Tokens:", tokenize(sample))
    print("Normalised:", normalize(sample))

    print("\n--- AUTOMATA ---")
    transitions, start, finals = make_patient_id_dfa()
    print("Patient ID DFA accepts PAT-12345:",
          dfa_accepts(transitions, start, finals, "PAT-12345"))
    print("Patient ID DFA rejects PAT-12A45:",
          dfa_accepts(transitions, start, finals, "PAT-12A45"))
    nfa_t, eps, nfa_start, nfa_finals = make_optional_suffix_nfa()
    alphabet = set("PAT-0123456789")
    dfa, dfa_start, dfa_finals, dfa_states = nfa_to_dfa(
        nfa_t, eps, nfa_start, nfa_finals, alphabet
    )
    print("Optional-digit NFA accepts PAT-1234:",
          nfa_accepts(nfa_t, eps, nfa_start, nfa_finals, "PAT-1234"))
    print("Optional-digit NFA accepts PAT-12345:",
          nfa_accepts(nfa_t, eps, nfa_start, nfa_finals, "PAT-12345"))
    print("Optional-digit NFA rejects PAT-123:",
          nfa_accepts(nfa_t, eps, nfa_start, nfa_finals, "PAT-123"))
    print("Subset-construction DFA states:", len(dfa_states))

    print("\n--- MORPHOLOGY ---")
    for word in ["fractures", "fractured", "fracturing", "patients", "treated"]:
        print(word, "->", MORPH.parse(word))
    print("Generation fracture+N+PL ->", MORPH.generate("fracture+N+PL"))
    print("Irregular children ->", MORPH.parse("children"))
    print("Inflection study + PL ->", inflect("study", "PL"))
    print("Derivation nephr + itis ->", derive("nephr", "itis"))

    print("\n--- PORTER STEMMER ---")
    print("Backend:", STEMMER.backend)
    for word in ["fractures", "fractured", "fracturing", "diagnoses", "treatments"]:
        print(word, "->", STEMMER.stem(word))

    print("\n--- OCR / CODE-MIXED ADAPTATION ---")
    noisy = "PAT-12O45: patient ku fe-\\nver irundhuchu, paracetamol kuduthen."
    print(adapted_pipeline(noisy))

def main():
    parser = argparse.ArgumentParser(description="Clinical NLP normalisation and search pipeline")
    parser.add_argument("--input", type=Path, help="Optional CSV file with text or record column")
    args = parser.parse_args()

    run_demos()

    print("\n--- RUNNING FULL CORPUS PIPELINE ---")
    start = time.perf_counter()
    records, source = load_corpus(args.input)
    # Process every record through the integrated normalisation/entity/morphology path.
    for record in records:
        norm = normalize(record["text"])
        toks = tokenize(norm)
        _ = extract_entities(record["text"])
        for token in toks:
            if token.isalpha():
                _ = MORPH.parse(token)
                _ = STEMMER.stem(token)
    eval_rows = evaluate_retrieval(records, STEMMER)
    elapsed = time.perf_counter() - start
    summary = write_pipeline_outputs(records, None, eval_rows, source, elapsed)

    print("\n--- SUMMARY ---")
    print(json.dumps(summary, indent=2))
    print(f"\nFiles saved in: {OUTPUT_DIR.resolve()}")
    print("Use your own de-identified corpus for final assignment measurements.")

if __name__ == "__main__":
    main()
