#!/usr/bin/env python3
"""
Analyse how LLMs handle EDCS alternate readings (editorial corrections)
in PROPER NAMES only (personal names, place names, tribal/ethnic names).

EDCS notation: <correct=stone> means the stone shows 'stone' but the
intended reading is 'correct'. E.g. S<e=A>rdica → stone has "A" but
it should be read as "e", giving "Serdica" (not "Sardica").

This script checks whether models preserve the raw stone reading,
use the corrected reading, or produce something else entirely.
Only proper names are checked because common words get translated
and string-matching is meaningless.
"""

import csv
import json
import os
import re
from collections import Counter, defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILTERED = os.path.join(BASE, "data", "testset", "filtered-testset.csv")
TRANSLATIONS = os.path.join(BASE, "data", "testset", "results", "full_translations.json")

# Pattern: <correct=stone>
ALT_PATTERN = re.compile(r'<([^=]+)=([^>]+)>')

# ---- Curated list of proper-name alternate readings ----
# Format: (edcs_id, correct, stone) → type
# Only includes cases where the corrected/stone word is a proper name
# that should be preserved as-is in the translation.
PROPER_NAME_READINGS = {
    # Place names
    ("EDCS-07800443", "e", "A"):      ("place", "Serdica", "SArdica"),
    ("EDCS-60500062", "a", "E"):      ("place", "Savar(ia)", "SavEr(ia)"),
    ("EDCS-10800409", "i", "AE"):     ("tribal", "Vallensi", "VallensAE"),
    # Personal names
    ("EDCS-10800420", "e", "I"):      ("name", "Coccei(a)", "CoccIi(a)"),
    ("EDCS-07800605", "l", "I"):      ("name", "Sallienus", "SalIienus"),
    ("EDCS-07800654", "x", "S"):      ("name", "Alexan(der)", "AleSan(der)"),
    ("EDCS-07800866", "m", "N"):      ("name", "Pom/[peia]no", "PoN/[peia]no"),
    ("EDCS-07800792", "e", "I"):      ("name", "Maced(onicae)", "MacId(onicae)"),
    ("EDCS-08501624", "c", "T"):      ("name", "Porcia", "PorTia"),
    ("EDCS-08501692", "p", "B"):      ("name", "Ni{s}copolid[i]", "Ni{s}coBolid[i]"),
    ("EDCS-07800232", "er", "RE"):    ("name", "Tertia(m)", "TREtia(m)"),
    ("EDCS-19400416", "i", "I"):      ("name", "P(l)autio", "P(l)autIo"),
    ("EDCS-19400510", "a", "E"):      ("name", "Caesaris", "CaesEris"),
    # Ethnic/military unit names
    ("EDCS-07800348", "hr", "RH"):    ("ethnic", "Thra{e}c(um)", "TRHa{e}c(um)"),
    # Tribal designation
    ("EDCS-08501624", "l", "E"):      ("tribal", "Volt(inia)", "VoEt(inia)"),
    # Divine name
    ("EDCS-07800555", "i", "E"):      ("divine", "Divo", "DEvo"),
}


def clean_name(word):
    """Remove epigraphic markup from a word to get the clean name."""
    # Remove () {} [] / markers
    s = word
    for pat in [r'\{[^}]*\}', r'\([^)]*\)', r'\[[^\]]*\]', r'/']:
        s = re.sub(pat, '', s)
    return s.strip()


def build_search_variants(corrected_raw, stone_raw, correct_chars, stone_chars):
    """Build search variants for a proper name in translations."""
    corrected_clean = clean_name(corrected_raw)
    stone_clean = clean_name(stone_raw)
    
    # For "Serdica" vs "Sardica" → search for these strings
    # For "Cocceius/Cocceia" vs "Cocciius/Cocciia" → search variants
    variants_corrected = {corrected_clean.lower()}
    variants_stone = {stone_clean.lower()}
    
    # Also try without trailing case endings
    # E.g. "Vallensi" → "Vallens", "VallensAE" → "Vallens"
    # This catches cases where the translator uses a different case form
    
    return variants_corrected, variants_stone


def check_name_in_translation(translation, corrected_variants, stone_variants):
    """Check if translation uses corrected or stone form of a proper name."""
    trans_lower = translation.lower()
    
    has_corrected = any(v in trans_lower for v in corrected_variants)
    has_stone = any(v in trans_lower for v in stone_variants)
    
    if has_corrected and not has_stone:
        return 'corrected'
    elif has_stone and not has_corrected:
        return 'stone'
    elif has_corrected and has_stone:
        return 'both'
    else:
        return 'neither'


def main():
    # Load all inscriptions
    with open(FILTERED) as f:
        all_inscriptions = {r['edcs_id']: r for r in csv.DictReader(f)}
    
    # Load translations
    with open(TRANSLATIONS) as f:
        all_trans = json.load(f)
    
    # Build lookup
    trans_lookup = {}
    for entry in all_trans:
        eid = entry['edcs_id']
        lang = entry['language']
        edcs_full = f"EDCS-{eid}" if not str(eid).startswith("EDCS") else str(eid)
        trans_lookup[(edcs_full, lang)] = entry
        trans_lookup[(str(eid), lang)] = entry
    
    # Get model names
    models = sorted([k.replace('llm_', '') for k in all_trans[0].keys() if k.startswith('llm_')])
    
    # Analyse each proper name reading
    results = defaultdict(Counter)
    details = []
    
    inscriptions_checked = set()
    readings_checked = 0
    
    for (edcs_id, correct, stone), (name_type, corrected_raw, stone_raw) in PROPER_NAME_READINGS.items():
        if edcs_id not in all_inscriptions:
            print(f"⚠ {edcs_id} not in testset, skipping")
            continue
            
        inscriptions_checked.add(edcs_id)
        corrected_clean = clean_name(corrected_raw)
        stone_clean = clean_name(stone_raw)
        
        # Skip if both clean forms are identical (the difference is only in markup)
        if corrected_clean.lower() == stone_clean.lower():
            print(f"⚠ {edcs_id}: '{corrected_clean}' == '{stone_clean}' after cleaning, skipping")
            continue
        
        corrected_variants = {corrected_clean.lower()}
        stone_variants = {stone_clean.lower()}
        
        print(f"\n{'='*70}")
        print(f"{edcs_id} [{name_type}]: {corrected_clean} (corrected) vs {stone_clean} (stone)")
        print(f"{'='*70}")
        
        for lang in ['EN', 'DE', 'FR']:
            entry = trans_lookup.get((edcs_id, lang))
            if not entry:
                continue
            
            readings_checked += 1
            
            # Also check human reference
            human_ref = entry.get('human_translation', '')
            if human_ref:
                human_verdict = check_name_in_translation(human_ref, corrected_variants, stone_variants)
                print(f"  [{lang}] Human ref: {human_verdict}")
                if human_verdict in ('corrected', 'stone'):
                    print(f"         → found in: ...{human_ref[:150]}...")
            
            for model in models:
                hyp = entry.get(f'llm_{model}', '')
                if not hyp or len(hyp) < 5:
                    continue
                
                verdict = check_name_in_translation(hyp, corrected_variants, stone_variants)
                results[model][verdict] += 1
                
                if verdict in ('stone', 'corrected'):
                    # Find the actual match for display
                    match_str = corrected_clean if verdict == 'corrected' else stone_clean
                    idx = hyp.lower().find(match_str.lower())
                    context = hyp[max(0,idx-20):idx+len(match_str)+20] if idx >= 0 else hyp[:80]
                    print(f"  [{lang}] {model:30s} → {verdict:10s}  «{context}»")
                
                details.append({
                    'edcs_id': edcs_id,
                    'name_type': name_type,
                    'language': lang,
                    'model': model,
                    'verdict': verdict,
                    'corrected_name': corrected_clean,
                    'stone_name': stone_clean,
                    'translation_excerpt': hyp[:200],
                })
    
    # Print summary
    print(f"\n\n{'='*80}")
    print(f"SUMMARY: Proper names with alternate readings")
    print(f"Inscriptions checked: {len(inscriptions_checked)}")
    print(f"Name checks per model (across 3 languages): {readings_checked}")
    print(f"{'='*80}")
    print(f"{'Model':<30s} {'Corrected':>10s} {'Stone':>10s} {'Neither':>10s} {'Both':>8s} {'Total':>8s} {'% Corr.':>8s} {'% Stone':>8s}")
    print("-" * 100)
    
    model_stats = []
    for model in models:
        c = results[model]
        total = sum(c.values())
        pct_corr = c['corrected'] / total * 100 if total > 0 else 0
        pct_stone = c['stone'] / total * 100 if total > 0 else 0
        model_stats.append((model, c['corrected'], c['stone'], c['neither'], c['both'], total, pct_corr, pct_stone))
    
    model_stats.sort(key=lambda x: x[6], reverse=True)
    for model, corr, stone, neither, both, total, pct_corr, pct_stone in model_stats:
        print(f"{model:<30s} {corr:>10d} {stone:>10d} {neither:>10d} {both:>8d} {total:>8d} {pct_corr:>7.1f}% {pct_stone:>7.1f}%")
    
    # Save results
    output = {
        'summary': {model: dict(results[model]) for model in results},
        'proper_names_checked': [
            {'edcs_id': k[0], 'correct': k[1], 'stone': k[2],
             'type': v[0], 'corrected_form': v[1], 'stone_form': v[2]}
            for k, v in PROPER_NAME_READINGS.items()
        ],
        'total_inscriptions': len(inscriptions_checked),
        'readings_per_model': readings_checked,
        'details': details,
    }
    
    outpath = f"{BASE}/mt/paper_assets/data/alternate_reading_proper_names.json"
    with open(outpath, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {outpath}")


if __name__ == "__main__":
    main()
