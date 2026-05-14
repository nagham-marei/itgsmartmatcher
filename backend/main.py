# ======================================
# main.py — Local FastAPI Backend
# شغّليه بـ: python main.py
# ======================================

import os
import re
import json
import io
import faiss
import numpy as np
import pandas as pd
import requests
import unicodedata
import uvicorn

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Any
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import MinMaxScaler
import google.generativeai as genai

# ======================================
# CONFIG
# ======================================
GEMINI_KEY = "AIzaSyCEgyLwLAdzL1CW6CMGCEwiz2TPm_U1BHU"
JINA_KEY   = "jina_9bc741d0ed994a1b9a3fcb6610c989adihydrnvVL0L0Ek8cpTfzq1gWtQW6"

CV_EMBEDDINGS_PATH = r"C:\Users\user\Downloads\my_project\backend\data\cv_embeddings.npy"
CV_INDEX_PATH      = r"C:\Users\user\Downloads\my_project\backend\data\cv_index.faiss"
CV_DATA_PATH       = r"C:\Users\user\Downloads\my_project\backend\data\cv_data_clean.csv"

VALID_CATEGORIES = [
    "ACCOUNTANT", "BUSINESS-DEVELOPMENT", "IT", "SALES", "TEACHER",
    "FINANCE", "HR", "ENGINEERING", "DIGITAL-MEDIA", "DESIGNER",
    "CONSULTANT", "BANKING"
]

# ======================================
# FIX UNICODE — يصلح الأسماء المكسورة
# ======================================
def fix_unicode(text: str) -> str:
    if not isinstance(text, str) or text == "nan":
        return ""
    
    try:
        #  تحاول إعادة النص لأصله ثم ترميزه صح
     
        clean_text = text.encode('cp1252').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        try:
            clean_text = text.encode('latin1').decode('utf-8')
        except:
            clean_text = text

    # تنظيف المسافات والرموز الغريبة
    clean_text = unicodedata.normalize('NFC', clean_text)
    
    # حل سريع ومباشر للرموز المشهورة إذا لم تنفع الطرق السابقة
    replacements = {
        "â€™": "'",
        "â€“": "-",
        "â€”": "—",
        "â€": '"',
        "â€¦": "...",
        "ânbsp;": " "
    }
    for bad, good in replacements.items():
        clean_text = clean_text.replace(bad, good)
        
    return clean_text.strip()

# ======================================
# LOAD MODELS & DATA
# ======================================
print("⏳ Loading sentence model...")
model = SentenceTransformer('all-mpnet-base-v2')

print("⏳ Loading data files...")
cv_embeddings = np.load(CV_EMBEDDINGS_PATH)
df            = pd.read_csv(CV_DATA_PATH, encoding='utf-8')
index         = faiss.read_index(CV_INDEX_PATH)
df.fillna("", inplace=True)

if 'name' in df.columns:
    df['name'] = df['name'].apply(fix_unicode)

print("⏳ Configuring Gemini...")
genai.configure(api_key=GEMINI_KEY)
llm = genai.GenerativeModel('gemini-3.1-flash-lite-preview')

print("✅ All resources loaded!")

# ======================================
# PYDANTIC MODELS
# ======================================
class SearchRequest(BaseModel):
    job_description: str
    top_k: int = 10

class ChatRequest(BaseModel):
    question: str
    history: list[dict[str, Any]] = []
    candidate_ids: list[str] = []
    jd_analysis: dict = {}
    
# ======================================
# CLEAN TEXT
# ======================================
def clean_text(text):
    return re.sub(r"\s+", " ", str(text)).strip()

# ======================================
# JOB ANALYSIS
# ======================================
def unified_job_analysis(jd_text: str) -> dict:
    prompt = """You are a strict information extraction system for Job Description parsing.
Extract ONLY structured data for candidate-job matching.

CRITICAL RULES:
- Output ONLY valid JSON.
- No explanations, no markdown, no extra text.
- Do NOT invent information.
- If a field is missing, return "" or [] only.
- Ignore irrelevant company fluff, marketing language, or duplicated text.

IMPORTANT:
- Extract ONLY explicitly mentioned requirements.
- Normalize terminology for consistency.
- Keep extracted content concise and structured.
- Standardize skills, tools, and domains for semantic matching.
- Preserve hiring intent accurately.

FOCUS ONLY ON:
summary, technical_skills, tools, soft_skills, domains, languages, accomplishments, experience, education, projects, certifications, category, experience_level

OUTPUT SCHEMA:
{
  "summary": "",
  "technical_skills": [],
  "tools": [],
  "soft_skills": [],
  "domains": [],
  "languages": [],
  "accomplishments": [],
  "experience": "",
  "education": [],
  "projects": [],
  "certifications": [],
  "category": "One from: ACCOUNTANT, BUSINESS-DEVELOPMENT, IT, SALES, TEACHER, FINANCE, HR, ENGINEERING, DIGITAL-MEDIA, DESIGNER, CONSULTANT, BANKING",
  "experience_level": "junior or mid or senior"
}

RULES:
- summary: Extract a concise professional summary of the role.
- technical_skills: Extract explicit technical competencies only.
- tools: Extract software, frameworks, platforms, and technologies.
- soft_skills: Extract interpersonal or behavioral skills.
- domains: Extract business or technical specialization areas.
- languages: Extract required spoken/written languages only.
- accomplishments: Extract explicit performance expectations or business goals.
- experience: Extract required experience level or years exactly as stated.
- education: Extract required degrees or academic qualifications only.
- projects: Extract explicit project types or expected implementation areas.
- certifications: Extract required or preferred certifications only.
- category: Must classify into ONE of the predefined categories above.
- experience_level: Must be exactly one of: junior | mid | senior

NORMALIZATION RULES:
- Remove duplicates
- Standardize similar terms
- Keep terminology embedding-friendly
- Avoid unnecessary verbosity

JOB DESCRIPTION TEXT:
<<<""" + str(jd_text) + """>>>"""

    default_response = {
        "summary": "",
        "technical_skills": [],
        "tools": [],
        "soft_skills": [],
        "domains": [],
        "languages": [],
        "accomplishments": [],
        "experience": "",
        "education": [],
        "projects": [],
        "certifications": [],
        "category": None,
        "experience_level": "mid"
    }

    try:
        res = llm.generate_content(
            prompt,
            generation_config={
                "temperature": 0,
                "response_mime_type": "application/json"
            }
        )

        print(f"📝 RAW RESPONSE (first 300 chars): {res.text[:300]}")

        clean_res = (
            res.text
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

        if not clean_res:
            print("⚠️ WARNING: Empty response from Gemini!")
            return default_response

        parsed = json.loads(clean_res)

        for key, default_value in default_response.items():
            if key not in parsed:
                parsed[key] = default_value

        if parsed["category"] not in VALID_CATEGORIES:
            print(f"⚠️ WARNING: Category '{parsed['category']}' not in list")
            parsed["category"] = None

        if parsed["experience_level"] not in ["junior", "mid", "senior"]:
            parsed["experience_level"] = "mid"

        return parsed

    except json.JSONDecodeError as e:
        print(f"❌ JSON Parse Error: {e}")
        print(f"📝 Failed response: {res.text[:500] if res else 'NO RESPONSE'}")
        return default_response

    except Exception as e:
        print(f"❌ Job analysis failed: {e}")
        return default_response

# ======================================
# BUILD JD TEXT
# ======================================
def build_jd_text(jd_analysis: dict) -> str:
    mapping = {
        "SUMMARY":          "summary",
        "TECHNICAL_SKILLS": "technical_skills",
        "SOFT_SKILLS":      "soft_skills",
        "TOOLS":            "tools",
        "DOMAINS":          "domains",
        "EXPERIENCE":       "experience",
        "EDUCATION":        "education",
        "PROJECTS":         "projects",
        "LANGUAGES":        "languages",
        "ACHIEVEMENTS":     "accomplishments",
        "CERTIFICATIONS":   "certifications",
    }
    parts = []
    for key, field in mapping.items():
        val = jd_analysis.get(field, "")
        if isinstance(val, list):
            val = " ".join([str(v) for v in val if str(v).strip()])
        val = str(val).strip()
        if val:
            parts.append(f"{key}: {val}")
    return " || ".join(parts)

# ======================================
# CATEGORY FILTER
# ======================================
def filter_by_category(category: str | None) -> pd.DataFrame:
    if not category or "category" not in df.columns:
        print("⚠️ No category filter applied — searching all CVs")
        return df

    filtered = df[
        df["category"].str.upper().str.strip() == str(category).upper().strip()
    ]

    if len(filtered) < 10:
        print(f"⚠️ Only {len(filtered)} CVs found for '{category}', using full dataset")
        return df

    return filtered

# ======================================
# FAISS SEARCH (المعدلة لمنع الـ Mismatch)
# ======================================
def search_cv(jd_analysis: dict, filtered_df: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    # 1. تحضير النص للبحث
    structured_jd = build_jd_text(jd_analysis)
    jd_emb = model.encode([structured_jd], normalize_embeddings=True).astype("float32")

    # 2. البحث في FAISS (نطلب 100 نتيجة لضمان إيجاد العدد المطلوب بعد الفلترة)
    scores, indices = index.search(jd_emb, 100)

    # 3. استخراج الـ IDs المسموحة فقط من الفئة (Category) المختارة
    valid_ids = set(filtered_df["ID"].astype(str).tolist())

    results_list = []
    match_scores = []

    # 4. الربط الصحيح: نستخدم idx لجلب الـ ID الحقيقي من الـ df الأصلي
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(df): continue # حماية من أي أخطاء في الـ index
        
        # جلب الهوية الحقيقية للشخص الموجود في هذا السطر
        candidate_real_id = str(df.iloc[idx]["ID"])
        
        # التأكد أن هذا الشخص ينتمي للفئة المطلوبة (IT, Sales, etc.)
        if candidate_real_id in valid_ids:
            # نجلب بياناته كاملة من الـ DataFrame الأصلي
            row_data = df.iloc[idx].to_dict()
            results_list.append(row_data)
            match_scores.append(float(score))
        
        if len(results_list) >= (top_k * 2): # نأخذ ضعف العدد للـ Reranker
            break

    if not results_list:
        return pd.DataFrame()

    out = pd.DataFrame(results_list)
    out["match_score"] = match_scores
    return out

# ======================================
# JINA RERANK
# ======================================
def jina_rerank(query: str, docs: list) -> tuple:
    url = "https://api.jina.ai/v1/rerank"
    headers = {
        "Authorization": f"Bearer {JINA_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "jina-reranker-v2-base-multilingual",
        "query": query,
        "documents": docs,
        "top_n": len(docs)
    }
    res     = requests.post(url, headers=headers, json=payload, timeout=30)
    data    = res.json()
    results = data["results"]
    return (
        [r["index"] for r in results],
        [r["relevance_score"] for r in results]
    )

# ======================================
# SKILL SCORE
# ======================================
def skill_score(cv_skills_raw, jd_skills: list) -> float:
    if not jd_skills:
        return 1.0

    if isinstance(cv_skills_raw, str):
        cv_skills = [s.strip() for s in cv_skills_raw.split(",") if s.strip()]
    elif isinstance(cv_skills_raw, list):
        cv_skills = [str(s).strip() for s in cv_skills_raw if str(s).strip()]
    else:
        cv_skills = []

    if not cv_skills:
        return 0.0

    jd_skills_clean = [str(s).strip() for s in jd_skills if str(s).strip()]

    if not jd_skills_clean:
        return 1.0

    cv_emb = model.encode(cv_skills,       normalize_embeddings=True)
    jd_emb = model.encode(jd_skills_clean, normalize_embeddings=True)

    sim         = util.cos_sim(jd_emb, cv_emb)
    best_per_jd = sim.max(dim=1).values

    return float(best_per_jd.mean())

# ======================================
# RANK (نسخة معدلة بدون MinMaxScaler)
# ======================================
def rank(results: pd.DataFrame, jd: dict) -> pd.DataFrame:
    results = results.copy()

    # 1. حساب سكور المهارات الحقيقي
    results["skill_score"] = results["technical_skills"].apply(
        lambda x: skill_score(x, jd["technical_skills"])
    )

    # 2. حذف السطور تاعت الـ MinMaxScaler (هون مربط الفرس)
    # لا تعملي scaler.fit_transform أبداً

    # 3. الحسبة النهائية مباشرة من الأرقام الخام
    results["final_score"] = (
        results["match_score"]  * 0.35 +
        results["rerank_score"] * 0.40 +
        results["skill_score"]  * 0.25
    )

    return results.sort_values("final_score", ascending=False)

# ======================================
# BUILD CONTEXT
# ======================================
def build_context(candidates_df: pd.DataFrame) -> str:
    context = ""
    for i, (_, row) in enumerate(candidates_df.iterrows()):
        context += f"""
        Candidate {i+1}
        Name: {row['name']}
        ID: {row['ID']}
        Skills: {row['technical_skills']}
        Experience: {row['cv_text']}
        ---"""
    return context

# ======================================
# CHATBOT
# ======================================
def chat_bot(context: str, question: str, history: list) -> tuple:
    if not history:
        system_msg = f"""You are an HR assistant. ONLY use this candidate data to answer questions.

{context}"""
        history = [
            {"role": "user",  "parts": [system_msg]},
            {"role": "model", "parts": ["Understood. I will only use the provided candidate data to answer your questions."]}
        ]

    history.append({"role": "user", "parts": [question]})
    chat     = llm.start_chat(history=history[:-1])
    response = chat.send_message(question)
    history.append({"role": "model", "parts": [response.text]})

    return response.text, history

# ======================================
# FASTAPI APP
# ======================================
app = FastAPI(title="ITG Recruitment API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search")
def search_candidates(req: SearchRequest):
    # 1. تحليل الوظيفة
    analysis = unified_job_analysis(req.job_description)
    
    # 2. الفلترة حسب الفئة (Category)
    filtered_df = filter_by_category(analysis["category"])
    
    # 3. البحث الأولي (Semantic Search) 
    # نطلب top_k * 5 لضمان وجود خيارات كافية للـ Reranker بعد الفلترة
    results = search_cv(analysis, filtered_df, top_k=req.top_k * 5)

    if results.empty:
        raise HTTPException(status_code=404, detail="No candidates found")

    # 4. تحضير الـ Reranking (Jina)
    # نأخذ أول 20 مرشح فقط للريرانكر (لأن Jina مكلف وبطيء على الأعداد الكبيرة)
    results_for_rerank = results.head(20).copy()
    structured_jd = build_jd_text(analysis)
    
    try:
        indices, scores = jina_rerank(structured_jd, results_for_rerank["cv_text"].tolist())
        results_for_rerank = results_for_rerank.iloc[indices].copy()
        results_for_rerank["rerank_score"] = scores
    except Exception as e:
        print(f"⚠️ Reranker failed: {e}")
        # في حال الفشل، نعطي الـ rerank_score نفس قيمة الـ match_score
        results_for_rerank["rerank_score"] = results_for_rerank["match_score"]

    # 5. حساب المهارات والترتيب النهائي (الـ Logic الموجود في rank)
    # دالة rank ستطبق الأوزان: 0.30, 0.35, 0.25
    final_results = rank(results_for_rerank, analysis)
    
    # نأخذ العدد المطلوب للفرونتند
    final_results = final_results.head(req.top_k)

    # 6. تجهيز البيانات للـ JSON
    candidates = [
        {
            "id":               str(row["ID"]),
            "name":             fix_unicode(str(row["name"])),
            "technical_skills": fix_unicode(str(row.get("technical_skills", ""))),
            "category":         str(row.get("category", "")),
            "match_score":      round(float(row["match_score"]),  4),
            "rerank_score":     round(float(row["rerank_score"]), 4),
            "skill_score":      round(float(row["skill_score"]),  4),
            "final_score":      round(float(row["final_score"]),  4),
        }
        for _, row in final_results.iterrows()
    ]

    return {
        "analysis":    analysis,
        "candidates":  candidates,
        "total_found": len(candidates)
    }


# ======================================
# EXPORT CSV
# ======================================
class ExportRequest(BaseModel):
    candidates: list[dict]
    analysis:   dict

@app.post("/export-csv")
def export_csv(req: ExportRequest):
    """يرجع ملف CSV بكل نتائج البحث"""
    rows = []
    for c in req.candidates:
        clean_name = fix_unicode(str(c.get("name", "")))
        rows.append({
            "Rank":             req.candidates.index(c) + 1,
            "Name":             clean_name,
            "ID":               c.get("id", ""),
            "Category":         c.get("category", ""),
            "Final Score %":    round(c.get("final_score", 0) * 100, 1),
            "Semantic Score %": round(c.get("match_score",  0) * 100, 1),
            "Rerank Score %":   round(c.get("rerank_score", 0) * 100, 1),
            "Skill Score %":    round(c.get("skill_score",  0) * 100, 1),
            "Technical Skills": c.get("technical_skills", ""),
            "CV Summary":       c.get("cv_text", ""),
            # بيانات الـ JD المستخرجة
            "JD Category":      req.analysis.get("category", ""),
            "JD Level":         req.analysis.get("experience_level", ""),
            "JD Skills":        ", ".join(req.analysis.get("technical_skills", [])),
            "JD Tools":         ", ".join(req.analysis.get("tools", [])),
            "JD Experience":    req.analysis.get("experience", ""),
            "JD Summary":       req.analysis.get("summary", ""),
        })

    df_export = pd.DataFrame(rows)
    output    = io.StringIO()
    df_export.to_csv(output, index=False, encoding="utf-8-sig")  # utf-8-sig يدعم العربي بـ Excel
    output.seek(0)

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=ITG_SmartMatcher_Results.csv"}
    )


@app.post("/chat")
def hr_chat(req: ChatRequest):
    if not req.candidate_ids:
        raise HTTPException(status_code=400, detail="No candidates to discuss.")

    # 1. هون السحر: بنجيب الداتا الأصلية كاملة من الـ DataFrame باستخدام الـ IDs
    # هيك جيمناي بيشوف الـ CV كامل (Full Text) حتى لو الفرونتند ما استلم ولا حرف
    filtered_cvs = df[df["ID"].astype(str).isin([str(i) for i in req.candidate_ids])]
    structured_jd = build_jd_text(req.jd_analysis)
    # 2. بناء السياق (Context) من النص الكامل
    context = f"JOB REQUIREMENTS:\n{structured_jd}\n\nDetailed Candidate Profiles:\n"
    for _, row in filtered_cvs.iterrows():
        context += f"Candidate: {row['name']} (ID: {row['ID']})\n"
        context += f"Full Experience: {row['cv_text']}\n" # النص كامل هون!
        context += f"Technical Skills: {row['technical_skills']}\n"
        context += f"Summary: {row.get('summary','')}\n"
        context += f"Experience: {row.get('experience','')}\n"
        context += f"Projects: {row.get('projects','')}\n"
        context += f"Education: {row.get('education','')}\n"
        context += "---\n"

    try:
        # 3. بنبعت النص الكامل لـ Gemini
        answer, updated_history = chat_bot(context, req.question, req.history)
        return {"answer": answer, "history": updated_history}
    except Exception as e:
        print(f"Chat Error: {e}")
        raise HTTPException(status_code=500, detail="AI is having trouble reading full CVs.")


# ======================================
# RUN
# ======================================
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

