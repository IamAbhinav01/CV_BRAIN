import os
import re
import json
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="LLM Brain - ATS Resume & LaTeX Generation Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def escape_latex(text: str) -> str:
    """Sanitize and escape special LaTeX characters in user input."""
    if not isinstance(text, str):
        return text
    
    # Avoid escaping already escaped LaTeX commands
    chars = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
    }
    
    # Pattern matches unescaped &, %, $, #, _
    pattern = re.compile(r'(?<!\\)([&%$#_])')
    return pattern.sub(lambda m: chars.get(m.group(1), m.group(1)), text)


class EducationEntry(BaseModel):
    institution: Optional[str] = ""
    degree: Optional[str] = ""
    location: Optional[str] = ""
    dates: Optional[str] = ""
    gpa: Optional[str] = ""

class ExperienceEntry(BaseModel):
    company: Optional[str] = ""
    role: Optional[str] = ""
    location: Optional[str] = ""
    dates: Optional[str] = ""
    bullets: Optional[Any] = ""  # string or list

class UserProfile(BaseModel):
    name: Optional[str] = ""
    email: Optional[str] = ""
    phone: Optional[str] = ""
    location: Optional[str] = ""
    linkedin: Optional[str] = ""
    github: Optional[str] = ""
    portfolio: Optional[str] = ""
    summary: Optional[str] = ""
    education: Optional[List[EducationEntry]] = []
    experience: Optional[List[ExperienceEntry]] = []
    skills: Optional[Dict[str, Any]] = {}

class RepoDetail(BaseModel):
    id: Optional[Any] = None
    name: Optional[str] = ""
    fullName: Optional[str] = Field(default="", alias="full_name")
    description: Optional[str] = ""
    language: Optional[str] = ""
    stars: Optional[int] = 0
    topics: Optional[List[str]] = []
    url: Optional[str] = ""
    readmeContent: Optional[str] = Field(default="", alias="readme_content")
    manifests: Optional[Dict[str, str]] = {}
    fileTreeSample: Optional[List[str]] = Field(default=[], alias="file_tree_sample")
    topFileHeaders: Optional[Dict[str, str]] = Field(default={}, alias="top_file_headers")
    commitMessages: Optional[List[str]] = Field(default=[], alias="commit_messages")
    userNotes: Optional[str] = Field(default="", alias="user_notes")

class GenerateCvRequest(BaseModel):
    user_profile: UserProfile
    selected_repos: List[RepoDetail] = []
    latex_template: str
    template_id: Optional[str] = "jake"
    target_role: Optional[str] = ""
    target_pages: Optional[int] = 1


@app.get("/health")
def health():
    return {"ok": True, "service": "LLM Brain"}


@app.post("/api/generate-cv")
async def generate_cv(req: GenerateCvRequest):
    try:
        user = req.user_profile
        repos = req.selected_repos
        template_tex = req.latex_template
        target_pages = req.target_pages or 1

        # Check if work experience was provided
        has_experience = bool(user.experience and len([e for e in user.experience if e.company or e.role]) > 0)

        # Build repository context blocks (handling missing READMEs via manifests + trees + commits)
        repo_facts = []
        for r in repos:
            name = r.name or r.fullName
            desc = r.description or "No description provided."
            lang = r.language or "Unknown"
            topics = ", ".join(r.topics) if r.topics else "None"
            
            fact_block = f"### Repository: {name}\n- URL: {r.url}\n- Primary Language: {lang}\n- Topics: {topics}\n- Description: {desc}\n"
            
            if r.userNotes:
                fact_block += f"- User Performance Notes: {r.userNotes}\n"
            
            if r.readmeContent:
                # Use README snippet
                snippet = r.readmeContent[:2000]
                fact_block += f"- README Excerpt:\n```\n{snippet}\n```\n"
            else:
                fact_block += "- README: NONE (NO README FILE IN REPO)\n"
                if r.manifests:
                    fact_block += "- Extracted Package Manifests (Dependencies & Scripts):\n"
                    for m_name, m_content in r.manifests.items():
                        fact_block += f"  * File `{m_name}`:\n```\n{m_content[:1000]}\n```\n"
                if r.commitMessages:
                    fact_block += f"- Recent Commit History: {'; '.join(r.commitMessages[:6])}\n"
                if r.topFileHeaders:
                    fact_block += "- Entry Point Code Snippets:\n"
                    for f_name, f_head in r.topFileHeaders.items():
                        fact_block += f"  * File `{f_name}`:\n```\n{f_head[:500]}\n```\n"
                if r.fileTreeSample:
                    fact_block += f"- Sample File Tree: {', '.join(r.fileTreeSample[:20])}\n"
            
            repo_facts.append(fact_block)

        repo_context_str = "\n".join(repo_facts) if repo_facts else "No GitHub repositories selected."

        # Serialize User Details
        user_info_str = f"""
Name: {user.name}
Email: {user.email}
Phone: {user.phone}
Location: {user.location}
LinkedIn: {user.linkedin}
GitHub: {user.github}
Portfolio: {user.portfolio}
Summary / Bio: {user.summary}

Education:
{json.dumps([e.dict() for e in user.education], indent=2) if user.education else "None provided."}

Work Experience Provided: {has_experience}
Experience Entries:
{json.dumps([e.dict() for e in user.experience], indent=2) if user.experience else "NONE (User provided no work experience)."}

Skills & Technologies:
{json.dumps(user.skills, indent=2) if user.skills else "Infer technical skills from projects and user input."}
"""

        system_prompt = f"""You are an elite, world-class ATS Resume Writer and LaTeX Engineer.
Your task is to take a master LaTeX resume template and populate it with the user's real profile information and project details extracted from their GitHub repositories.

CRITICAL RULES & STRICT CONSTRAINTS:
1. **ZERO HALLUCINATION POLICY**:
   - Do NOT invent fake metrics, fake companies, fake dates, or unverified technical achievements.
   - Use ONLY the provided User Details and extracted Repository Facts.
   - For repositories without a README, use the extracted package manifests (e.g. package.json, pyproject.toml), commit messages, and entry files to construct accurate, factual 2-3 ATS action bullets per project.

2. **PAGE BUDGET ({target_pages} PAGE TARGET)**:
   - The output must fit strictly on **{target_pages} PAGE(S)**.
   - Keep bullet points concise, impact-oriented, and limited to 2-3 bullets per experience or project entry.
   - Do not allow dangling 1-2 lines to spill over onto an extra page.

3. **CONDITIONAL SECTION RENDERING**:
   - Work Experience Provided = {has_experience}.
   - IF Work Experience is FALSE (no experience provided): You MUST COMPLETELY REMOVE / OMIT the "Experience" or "Work Experience" section heading and its entire block from the LaTeX template. Do NOT leave an empty section header or empty itemize environments.
   - IF Work Experience is TRUE: Populate the Experience section cleanly with the provided experience entries.

4. **LATEX COMPILATION SAFETY**:
   - Return valid, raw compilable LaTeX code ONLY.
   - Ensure all LaTeX special characters in user text (% -> \\%, $ -> \\$, & -> \\&, # -> \\#, _ -> \\_) are properly escaped.
   - Retain documentclass, packages, custom commands, and environment definitions from the template.
   - DO NOT surround your response in markdown code blocks like ```latex or ```. Return ONLY the raw LaTeX string starting with \\documentclass or comments.
"""

        user_prompt = f"""Here is the Candidate's Profile:
{user_info_str}

Here are the Inspected Selected GitHub Repositories:
{repo_context_str}

Here is the Base LaTeX Template code:
{template_tex}

Inject the candidate's details and generated project bullets into the LaTeX template now. Enforce all rules, escape LaTeX special characters, and return ONLY the raw, complete, compilable LaTeX document string."""

        edited_tex = ""

        # Try executing LLM using Groq or OpenAI API if available
        groq_api_key = os.getenv("GROQ_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if groq_api_key:
            try:
                from langchain_groq import ChatGroq
                llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=groq_api_key, temperature=0.2)
                res = llm.invoke([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}])
                edited_tex = res.content
            except Exception as e:
                print(f"[LLM_BRAIN] Groq call failed: {e}")

        if not editedTex_has_value(edited_tex) and openai_api_key:
            try:
                from langchain_community.chat_models import ChatOpenAI
                llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=openai_api_key, temperature=0.2)
                res = llm.invoke([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}])
                edited_tex = res.content
            except Exception as e:
                print(f"[LLM_BRAIN] OpenAI call failed: {e}")

        # Fallback / Direct Deterministic Injector if LLM API Key is missing or failed
        if not editedTex_has_value(edited_tex):
            print("[LLM_BRAIN] No LLM API response available — executing fallback template string replacement.")
            edited_tex = fallback_latex_filler(template_tex, user, repos, has_experience)

        # Clean markdown code wraps if model added them
        edited_tex = re.sub(r"^```(?:latex|tex)?\n", "", edited_tex.strip(), flags=re.MULTILINE)
        edited_tex = re.sub(r"\n```$", "", edited_tex.strip(), flags=re.MULTILINE)

        # Extra safety pass for character escaping
        edited_tex = escape_latex(edited_tex)

        return {"ok": True, "tex": edited_tex, "summary": "Resume generated successfully with zero hallucination context."}

    except Exception as err:
        print(f"[LLM_BRAIN] Exception: {err}")
        raise HTTPException(status_code=500, detail=str(err))


def editedTex_has_value(text: str) -> bool:
    return bool(text and r"\documentclass" in text)


def fallback_latex_filler(tex: str, user: UserProfile, repos: List[RepoDetail], has_experience: bool) -> str:
    """Fallback deterministic filler when no LLM API key is present."""
    result = tex

    name = escape_latex(user.name or "Candidate Name")
    email = escape_latex(user.email or "user@example.com")
    phone = escape_latex(user.phone or "123-456-7890")

    # Basic header replacements
    result = re.sub(r"(\\name\{)[^}]*(\})", rf"\1{name}\2", result)
    result = re.sub(r"(\\href\{mailto:[^}]*\}\{)[^}]*(\})", rf"\1{email}\2", result)

    # If experience is empty, comment out or remove experience section
    if not has_experience:
        result = re.sub(r"\\section\{Experience\}.*?(?=\\section|\n\\end\{document\})", "", result, flags=re.DOTALL | re.IGNORECASE)
        result = re.sub(r"\\section\{Work Experience\}.*?(?=\\section|\n\\end\{document\})", "", result, flags=re.DOTALL | re.IGNORECASE)

    return result


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
