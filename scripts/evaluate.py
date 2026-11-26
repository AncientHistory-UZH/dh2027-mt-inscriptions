import pandas as pd
import os
import evaluate
from tqdm import tqdm
import sacrebleu

def get_edcs_id_mapping(inschrift_ids):
    # TODO: Replace with the actual mapping logic or database lookup. 
    # The user mentioned: "somehow, a matching will still be possible"
    # For now, we return a 1:1 dummy or try to load an external id_mapping file.
    mapping_file = "data/id_mapping.csv"
    if os.path.exists(mapping_file):
        mapping_df = pd.read_csv(mapping_file)
        # assuming columns: inschrift_id, EDCS-ID
        return dict(zip(mapping_df['inschrift_id'], mapping_df['EDCS-ID']))
    
    print("Warning: id_mapping.csv not found. Continuing with dummy identity mapping for testing.")
    return {i: str(i) for i in inschrift_ids}

def load_metrics():
    print("Loading metrics (this might take a while for neural metrics)...")
    return {
        "bleu": evaluate.load("sacrebleu"),
        "chrf": evaluate.load("chrf"),
        "bertscore": evaluate.load("bertscore"),
        "bleurt": evaluate.load("bleurt", "BLEURT-20"),
        "comet": evaluate.load("comet")
    }

def run_evaluation(translations_path, reference_xlsx, output_path):
    print("Loading human translations...")
    refs_df = pd.read_excel(reference_xlsx)
    
    # We assume refs_df has 'EDCS-ID', 'Übersetzung', 'Sprache'
    # Let's pivot it so we have EDCS-ID as rows and Languages as columns
    refs_pivoted = refs_df.pivot(index='EDCS-ID', columns='Sprache', values='Übersetzung').to_dict('index')
    
    metrics = load_metrics()
    
    eval_results = []
    
    for filename in os.listdir(translations_path):
        if not filename.endswith(".csv"):
            continue
            
        print(f"\nEvaluating {filename}...")
        model_name = filename.replace("translations_", "").replace(".csv", "")
        hyp_df = pd.read_csv(os.path.join(translations_path, filename))
        
        mapping = get_edcs_id_mapping(hyp_df['inschrift_id'])
        
        # Languages to assess
        for lang_code in ['DE', 'EN', 'FR', 'IT', 'ES']:
            predictions = []
            references = []
            sources = []
            
            for _, row in hyp_df.iterrows():
                inschrift_id = row['inschrift_id']
                edcs_id = mapping.get(inschrift_id)
                
                if pd.isna(row.get(lang_code)):
                    continue
                    
                if edcs_id in refs_pivoted and not pd.isna(refs_pivoted[edcs_id].get(lang_code)):
                    predictions.append(row[lang_code])
                    references.append(refs_pivoted[edcs_id][lang_code])
                    sources.append(row['latin_text'])
            
            if len(predictions) == 0:
                print(f"Skipping {model_name} - {lang_code}: No matching reference data.")
                continue
                
            print(f"Computing metrics for {model_name} - {lang_code} on {len(predictions)} pairs...")
            
            # BLEU
            bleu_res = metrics["bleu"].compute(predictions=predictions, references=[[r] for r in references])
            chrf_res = metrics["chrf"].compute(predictions=predictions, references=[[r] for r in references])
            
            # BERTScore
            bs_res = metrics["bertscore"].compute(predictions=predictions, references=references, lang=lang_code.lower())
            avg_bs_f1 = sum(bs_res['f1']) / len(bs_res['f1'])
            
            # BLEURT (language agnostic version)
            bleurt_res = metrics["bleurt"].compute(predictions=predictions, references=references)
            avg_bleurt = sum(bleurt_res['scores']) / len(bleurt_res['scores'])
            
            # COMET
            comet_res = metrics["comet"].compute(predictions=predictions, references=references, sources=sources)
            avg_comet = comet_res['mean_score']
            
            eval_results.append({
                "Model": model_name,
                "Language": lang_code,
                "Samples": len(predictions),
                "BLEU": bleu_res['score'],
                "chrF": chrf_res['score'],
                "BERTScore_F1": avg_bs_f1,
                "BLEURT": avg_bleurt,
                "COMET": avg_comet
            })

    results_df = pd.DataFrame(eval_results)
    results_df.to_csv(output_path, index=False)
    print(f"Evaluation complete. Results saved to {output_path}")
    print(results_df)

if __name__ == "__main__":
    run_evaluation("data/translations", "data/HumanTranslations.xlsx", "data/evaluation_metrics.csv")
