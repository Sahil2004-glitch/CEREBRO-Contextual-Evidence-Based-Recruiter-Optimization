"""
CEREBRO v2: Optimized for <5 min CPU constraint.
Key optimization: Pre-filter to top ~3000 candidates with fast heuristics,
then apply TF-IDF + full scoring only on the shortlist.
"""

import json, math, re, csv, argparse, time
from datetime import datetime, date
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TODAY = datetime.today().date()

CONSULTING_FIRMS = {
    "tcs","infosys","wipro","accenture","cognizant","capgemini","hcl","tech mahindra",
    "mphasis","hexaware","ltimindtree","mindtree","niit technologies","mastech","syntel"
}
PRODUCT_COMPANIES = {
    "google","meta","microsoft","amazon","apple","netflix","uber","airbnb","flipkart",
    "swiggy","zomato","ola","razorpay","cred","meesho","dream11","phonepe","paytm",
    "freshworks","zoho","cleartax","browserstack","postman","hasura","groww","zepto",
    "blinkit","sharechat","linkedin","twitter","salesforce","adobe","atlassian","stripe",
    "shopify","databricks","snowflake","openai","anthropic","cohere","nvidia","qualcomm",
    "sarvam","krutrim","niramai","unacademy","byju","vedantu","lenskart","urban company",
    "genpact ai","slice","chargebee","mixpanel","dgraph","redrob"
}
INDIA_PREF = {"pune","noida","delhi","gurugram","gurgaon","bengaluru","bangalore",
               "hyderabad","mumbai","ncr"}
NON_TECH_TITLES = ["marketing","sales","finance","accounting","hr ","recruiter",
                    "teacher","professor","content writer","graphic design","doctor","lawyer"]
AI_KEYWORDS = {"embedding","embeddings","vector","rag","retrieval","ranking","reranking",
                "re-ranking","semantic search","dense","bi-encoder","cross-encoder",
                "sentence-transformer","faiss","pinecone","weaviate","qdrant","milvus",
                "opensearch","elasticsearch","ndcg","mrr","learning to rank","ltr",
                "fine-tuning","finetuning","lora","qlora","peft","llm","transformer",
                "bert","gpt","huggingface","mlops","model serving","inference","a/b test"}

JD_TEXT = """
Senior AI Engineer Founding Team Redrob AI Series A AI-native talent intelligence platform
Pune Noida India Hybrid 5 to 9 years experience applied ML AI product companies
Production experience embeddings retrieval systems sentence-transformers openai embeddings BGE E5
embedding drift index refresh retrieval quality regression production
vector databases hybrid search infrastructure Pinecone Weaviate Qdrant Milvus OpenSearch FAISS
strong Python code quality evaluation frameworks ranking systems NDCG MRR MAP
AB test LLM fine-tuning LoRA QLoRA PEFT learning to rank recommendation system
ship ranking retrieval matching systems scrappy product engineering shipper
end-to-end ranking search recommendation real users meaningful scale
strong opinions retrieval hybrid dense evaluation offline online LLM
6 to 8 years total experience product companies not pure services not consulting
avoid title chasers keyword stuffers pure researchers no langchain wrappers
sub 30 day notice period active platform engagement
open source GitHub contributions AI ML NLP IR
"""

WEIGHTS = {
    "semantic":   0.22,
    "hard_skills":0.20,
    "career":     0.18,
    "experience": 0.10,
    "behavioral": 0.15,
    "location":   0.06,
    "anti":       0.09,
}

SKILL_GROUPS = [
    # (terms, weight, must_have)
    ({"sentence-transformer","sentence_transformer","embedding","embeddings","dense retrieval",
      "vector search","semantic search","bge","e5","openai embeddings","bi-encoder",
      "cross-encoder","faiss","annoy","hnswlib"}, 0.30, True),
    ({"pinecone","weaviate","qdrant","milvus","opensearch","elasticsearch",
      "chroma","pgvector","vector database","vector store","faiss"}, 0.20, True),
    ({"ndcg","mrr","map","learning to rank","ltr","ranknet","lambdamart","a/b test",
      "a/b testing","offline eval","online eval","ranking","reranking","re-ranking"}, 0.20, True),
    ({"rag","retrieval augmented","llm","large language model","fine-tuning","finetuning",
      "lora","qlora","peft","instruction tuning","gpt","gemini","llama","mistral","huggingface"}, 0.15, False),
    ({"mlops","model serving","model deployment","feature store","mlflow","ray","triton",
      "model drift","model monitoring"}, 0.10, False),
    ({"python","pytorch","tensorflow","jax","numpy","pandas","scikit"}, 0.05, True),
]

def parse_date(s):
    if not s: return None
    try: return datetime.strptime(s, "%Y-%m-%d").date()
    except: return None

def days_since(d):
    return (TODAY - d).days if d else 9999

def tl(t): return (t or "").lower()

def full_text(c):
    p = c.get("profile",{})
    parts = [p.get("headline",""), p.get("summary",""), p.get("current_title","")]
    for j in c.get("career_history",[]): 
        parts += [j.get("title",""), j.get("description",""), j.get("company","")]
    for s in c.get("skills",[]): parts.append(s.get("name",""))
    for cert in c.get("certifications",[]): parts.append(cert.get("name",""))
    return " ".join(parts).lower()

def fast_prefilter_score(c, ft):
    """Ultra-fast heuristic score for pre-filtering (no heavy computation)."""
    score = 0.0
    p = c.get("profile",{})
    s = c.get("redrob_signals",{})
    
    # Immediate disqualifiers
    title = tl(p.get("current_title",""))
    if any(t in title for t in NON_TECH_TITLES): return -1.0
    
    yoe = p.get("years_of_experience",0) or 0
    if yoe < 2: return -1.0
    
    # AI keyword presence (fast set intersection)
    ft_words = set(ft.split())
    ai_hit = len(AI_KEYWORDS & ft_words)
    score += min(ai_hit * 0.08, 0.50)
    
    # Experience sweet spot
    if 5 <= yoe <= 9: score += 0.20
    elif 4 <= yoe <= 11: score += 0.10
    
    # Behavioral: open to work + recent activity
    if s.get("open_to_work_flag"): score += 0.10
    last = parse_date(s.get("last_active_date"))
    if days_since(last) <= 30: score += 0.10
    elif days_since(last) > 365: score -= 0.10
    
    # Location
    loc = tl(p.get("location",""))
    if any(city in loc for city in INDIA_PREF): score += 0.05
    
    return score

def score_hard_skills(ft, skills):
    total_w = sum(g[1] for g in SKILL_GROUPS)
    score = 0.0
    skill_map = {s["name"].lower(): s.get("proficiency","beginner") for s in skills}
    for terms, weight, _ in SKILL_GROUPS:
        hit = any(t in ft for t in terms) or bool(terms & set(skill_map.keys()))
        if hit:
            prof_bonus = 0.0
            for t in terms:
                if t in skill_map:
                    p = skill_map[t]
                    if p == "expert": prof_bonus = 0.15
                    elif p == "advanced": prof_bonus = 0.10
                    elif p == "intermediate": prof_bonus = 0.05
            score += weight * min(1.0, 1.0 + prof_bonus)
    return min(1.0, score / total_w)

def score_career(c):
    history = c.get("career_history",[])
    if not history: return 0.1
    product_m = consulting_m = total_m = 0
    for j in history:
        comp = tl(j.get("company",""))
        ind = tl(j.get("industry",""))
        dur = j.get("duration_months",0) or 0
        total_m += dur
        is_prod = (any(p in comp for p in PRODUCT_COMPANIES) or
                   any(x in ind for x in ["saas","fintech","edtech","ai","startup","platform","product","internet"]) or
                   j.get("company_size","") in ["11-50","51-200","201-500"])
        is_cons = any(f in comp for f in CONSULTING_FIRMS)
        if is_cons: consulting_m += dur
        elif is_prod: product_m += dur
    pr = product_m/total_m if total_m > 0 else 0
    cr = consulting_m/total_m if total_m > 0 else 0
    sc = 0.0
    sc += 0.50 if pr >= 0.7 else (0.35 if pr >= 0.4 else (0.20 if pr >= 0.2 else 0.05))
    if cr >= 0.9: sc -= 0.25
    elif cr >= 0.6: sc -= 0.10
    titles = [tl(j.get("title","")) for j in history]
    if any(x in t for t in titles for x in ["senior","staff","principal","lead","head","founding","architect"]): sc += 0.15
    recent = [j.get("duration_months",0) for j in history[:3]]
    if recent:
        avg = np.mean(recent)
        sc += 0.10 if avg >= 24 else (0.05 if avg >= 15 else (-0.10 if avg < 10 else 0))
    if sum(1 for j in history if j.get("company_size","") in ["1-10","11-50","51-200"]) >= 1: sc += 0.10
    all_cons = all(any(f in tl(j.get("company","")) for f in CONSULTING_FIRMS) for j in history)
    if all_cons and len(history) >= 2: sc -= 0.20
    return max(0.0, min(1.0, sc))

def score_experience(yoe):
    if 6 <= yoe <= 8: return 1.0
    if 5 <= yoe < 6 or 8 < yoe <= 9: return 0.85
    if 4 <= yoe < 5 or 9 < yoe <= 11: return 0.65
    if 3 <= yoe < 4 or 11 < yoe <= 13: return 0.40
    if yoe < 3: return 0.15
    return 0.50

def score_behavioral(s):
    last = parse_date(s.get("last_active_date"))
    di = days_since(last)
    rec = 1.0 if di<=7 else (0.85 if di<=30 else (0.60 if di<=90 else (0.30 if di<=180 else 0.05)))
    otw = 1.0 if s.get("open_to_work_flag") else 0.3
    rr = float(s.get("recruiter_response_rate",0.5) or 0.5)
    notice = s.get("notice_period_days",90) or 90
    ns = 1.0 if notice<=30 else (0.70 if notice<=60 else (0.45 if notice<=90 else 0.20))
    icr = float(s.get("interview_completion_rate",0.5) or 0.5)
    gh = min(1.0, float(s.get("github_activity_score",0) or 0) / 80.0)
    return 0.25*rec + 0.20*otw + 0.20*rr + 0.15*ns + 0.10*icr + 0.10*gh

def score_location(c):
    s = c.get("redrob_signals",{})
    p = c.get("profile",{})
    loc = tl(p.get("location",""))
    country = tl(p.get("country",""))
    relocate = s.get("willing_to_relocate",False)
    if "pune" in loc or "noida" in loc: return 1.0
    if any(city in loc for city in ["delhi","gurugram","gurgaon","hyderabad","mumbai","bengaluru","bangalore"]): return 0.85
    if country == "india" and relocate: return 0.75
    if country == "india": return 0.60
    if relocate: return 0.35
    return 0.10

def score_anti(c, ft):
    p = c.get("profile",{})
    skills = c.get("skills",[])
    history = c.get("career_history",[])
    s = c.get("redrob_signals",{})
    penalty = 0.0
    
    title = tl(p.get("current_title",""))
    if any(t in title for t in NON_TECH_TITLES): penalty += 0.60
    
    impossible_skills = sum(1 for sk in skills 
        if sk.get("proficiency","") in ["expert","advanced"] and sk.get("duration_months",1) == 0)
    if impossible_skills >= 3: penalty += 0.40
    elif impossible_skills >= 1: penalty += 0.15
    
    yoe = p.get("years_of_experience",0) or 0
    if history:
        earliest = min((parse_date(j.get("start_date")) for j in history if parse_date(j.get("start_date"))), default=None)
        if earliest:
            max_exp = (TODAY - earliest).days / 365.25
            if yoe > max_exp + 5: penalty += 0.50
    
    ai_kw_count = sum(1 for sk in skills if any(kw in sk["name"].lower() for kw in ["rag","vector","embedding","llm","transformer","bert","gpt","langchain","pinecone","weaviate","faiss","retrieval","ranking"]))
    tech_sigs = sum(1 for sig in ["model","deploy","pipeline","inference","training","retrieval","embedding","api","index","vector","ranking","search"] if sig in ft)
    if ai_kw_count >= 6 and tech_sigs < 3: penalty += 0.35
    
    research_cnt = sum(1 for sig in ["phd","research scientist","research engineer","arxiv","postdoc"] if sig in ft)
    prod_cnt = sum(1 for sig in ["deployed","production","users","latency","serving","shipped","launch","scale"] if sig in ft)
    if research_cnt >= 3 and prod_cnt < 2: penalty += 0.30
    
    if len(history) >= 3:
        if all(j.get("duration_months",0) <= 12 for j in history[:3]): penalty += 0.20
    
    all_cons = all(any(f in tl(j.get("company","")) for f in CONSULTING_FIRMS) for j in history) if history else False
    if all_cons and len(history) >= 2: penalty += 0.30
    
    cv_only = (("computer vision" in ft or "object detection" in ft) and 
               not any(x in ft for x in ["nlp","retrieval","ranking","language model","embedding"]))
    if cv_only: penalty += 0.25
    
    langchain_heavy = ft.count("langchain") >= 2
    pre_llm = any(sig in ft for sig in ["bm25","elasticsearch","solr","lucene","tf-idf","tfidf",
                                          "collaborative filtering","xgboost","lightgbm","learning to rank"])
    if langchain_heavy and not pre_llm: penalty += 0.20
    
    return max(0.0, min(1.0, penalty))

def generate_reasoning(c, scores, norm_score):
    p = c.get("profile",{})
    s = c.get("redrob_signals",{})
    ft = full_text(c)
    yoe = p.get("years_of_experience",0)
    title = p.get("current_title","")
    company = p.get("current_company","")
    loc = p.get("location","")
    notice = s.get("notice_period_days",90) or 90
    rr = s.get("recruiter_response_rate",0.5)
    last = parse_date(s.get("last_active_date"))
    di = days_since(last)
    
    positives = []
    if scores["hard_skills"] > 0.7: positives.append("strong embeddings/retrieval skill set")
    if scores["career"] > 0.65: positives.append("product company background")
    if any(x in ft for x in ["faiss","pinecone","weaviate","qdrant","opensearch"]): positives.append("vector DB experience")
    if any(x in ft for x in ["ndcg","mrr","learning to rank","a/b test"]): positives.append("ranking eval experience")
    if notice <= 30: positives.append(f"sub-30d notice")
    if scores["behavioral"] > 0.75: positives.append("high platform engagement")
    
    concerns = []
    if scores["anti"] > 0.35: concerns.append("profile quality concerns")
    if di > 180: concerns.append(f"inactive {di}d")
    if rr < 0.2: concerns.append(f"low response rate ({rr:.0%})")
    if notice > 90: concerns.append(f"long notice ({notice}d)")
    if scores["career"] < 0.35: concerns.append("limited product company exp")
    
    pos = "; ".join(positives[:2]) if positives else f"{yoe:.0f}yr {title}"
    if concerns:
        return f"{yoe:.0f}yr {title} ({loc}) — {pos}. Note: {concerns[0]}."
    return f"{yoe:.0f}yr {title} at {company} ({loc}) — {pos}."

def rank_candidates(candidates_path, output_path, top_n=100):
    t0 = time.time()
    print(f"[CEREBRO v2] Loading candidates...")
    
    candidates = []
    with open(candidates_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try: candidates.append(json.loads(line))
                except: continue
    print(f"  Loaded {len(candidates)} in {time.time()-t0:.1f}s")
    
    # ── Phase 1: Fast pre-filter ───────────────────────────────────────────
    t1 = time.time()
    print("[CEREBRO v2] Phase 1: Fast pre-filter...")
    
    cand_texts = []
    prefilter_scores = []
    for c in candidates:
        ft = full_text(c)
        cand_texts.append(ft)
        prefilter_scores.append(fast_prefilter_score(c, ft))
    
    # Keep top 5000 by prefilter score
    prefilter_arr = np.array(prefilter_scores)
    top_5k_idx = np.argsort(prefilter_arr)[-5000:][::-1]
    print(f"  Pre-filtered to {len(top_5k_idx)} candidates in {time.time()-t1:.1f}s")
    
    # ── Phase 2: TF-IDF on shortlist ──────────────────────────────────────
    t2 = time.time()
    print("[CEREBRO v2] Phase 2: TF-IDF on shortlist...")
    
    shortlist_texts = [cand_texts[i] for i in top_5k_idx]
    corpus = [JD_TEXT] + shortlist_texts
    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=12000, sublinear_tf=True, min_df=2, stop_words='english')
    tfidf_mat = vectorizer.fit_transform(corpus)
    sims = cosine_similarity(tfidf_mat[0], tfidf_mat[1:])[0]
    s_min, s_max = sims.min(), sims.max()
    sims_norm = (sims - s_min) / (s_max - s_min) if s_max > s_min else sims
    print(f"  TF-IDF done in {time.time()-t2:.1f}s")
    
    # ── Phase 3: Full scoring on shortlist ────────────────────────────────
    t3 = time.time()
    print("[CEREBRO v2] Phase 3: Full scoring...")
    
    results = []
    for rank_i, orig_i in enumerate(top_5k_idx):
        c = candidates[orig_i]
        ft = cand_texts[orig_i]
        p = c.get("profile",{})
        s = c.get("redrob_signals",{})
        skills = c.get("skills",[])
        
        sc = {
            "semantic":    float(sims_norm[rank_i]),
            "hard_skills": score_hard_skills(ft, skills),
            "career":      score_career(c),
            "experience":  score_experience(p.get("years_of_experience",0) or 0),
            "behavioral":  score_behavioral(s),
            "location":    score_location(c),
            "anti":        score_anti(c, ft),
        }
        
        anti_sc = 1.0 - sc["anti"]
        raw = (WEIGHTS["semantic"]   * sc["semantic"] +
               WEIGHTS["hard_skills"]* sc["hard_skills"] +
               WEIGHTS["career"]     * sc["career"] +
               WEIGHTS["experience"] * sc["experience"] +
               WEIGHTS["behavioral"] * sc["behavioral"] +
               WEIGHTS["location"]   * sc["location"] +
               WEIGHTS["anti"]       * anti_sc)
        
        if sc["anti"] >= 0.80: raw *= 0.1
        elif sc["anti"] >= 0.60: raw *= 0.3
        
        results.append({"id": c["candidate_id"], "raw": raw, "sc": sc, "c": c})
    
    print(f"  Scoring done in {time.time()-t3:.1f}s")
    
    # ── Phase 4: Sort & normalize ─────────────────────────────────────────
    results.sort(key=lambda x: x["raw"], reverse=True)
    top = results[:top_n]
    
    raw_arr = np.array([r["raw"] for r in top])
    r_min, r_max = raw_arr.min(), raw_arr.max()
    norm = 0.40 + 0.59 * (raw_arr - r_min) / (r_max - r_min) if r_max > r_min else np.full(len(top), 0.70)
    for i in range(1, len(norm)):
        if norm[i] > norm[i-1]: norm[i] = norm[i-1]
    
    # ── Phase 5: Write CSV ────────────────────────────────────────────────
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id","rank","score","reasoning"])
        for rank_i, (r, ns) in enumerate(zip(top, norm), start=1):
            writer.writerow([r["id"], rank_i, f"{ns:.4f}", generate_reasoning(r["c"], r["sc"], ns)])
    
    total = time.time() - t0
    print(f"\n[CEREBRO v2] DONE in {total:.1f}s → {output_path}")
    print(f"\n── TOP 10 ─────────────────────────────────────────────")
    for rank_i, (r, ns) in enumerate(zip(top[:10], norm[:10]), start=1):
        p2 = r["c"]["profile"]
        sc2 = r["sc"]
        print(f"  #{rank_i:2d} | {r['id']} | {ns:.3f} | {p2.get('current_title','')} @ {p2.get('current_company','')} | {p2.get('years_of_experience',0):.0f}yr | {p2.get('location','')}")
        print(f"       skills={sc2['hard_skills']:.2f} career={sc2['career']:.2f} behav={sc2['behavioral']:.2f} anti={sc2['anti']:.2f}")
    
    return top, norm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CEREBRO v2: Contextual Evidence-Based Recruiter Optimization")
    parser.add_argument("--candidates", default="candidates.jsonl")
    parser.add_argument("--out", default="submission.csv")
    parser.add_argument("--top_n", type=int, default=100)
    args = parser.parse_args()
    rank_candidates(args.candidates, args.out, args.top_n)
