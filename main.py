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
    """Sanitize and escape special LaTeX characters in plain text string."""
    if not isinstance(text, str):
        return text
    
    chars = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
    }
    
    pattern = re.compile(r'(?<!\\)([&%$#_])')
    return pattern.sub(lambda m: chars.get(m.group(1), m.group(1)), text)


def sanitize_generated_tex(tex: str) -> str:
    """Clean up markdown code blocks, duplicate package includes, empty list environments, trailing linebreaks, and unescaped underscores/ampersands."""
    if not isinstance(tex, str):
        return ""
    
    # 1. Clean markdown code wraps if present
    tex = re.sub(r"^```(?:latex|tex)?\n", "", tex.strip(), flags=re.MULTILINE)
    tex = re.sub(r"\n```$", "", tex.strip(), flags=re.MULTILINE)
    tex = tex.strip()

    # 1b. Strip any natural language text before the actual LaTeX content
    #     LLMs often prepend "Here's the completed document:" etc.
    for marker in [r'\documentclass', '%']:
        idx = tex.find(marker)
        if idx > 0:
            tex = tex[idx:]
            break
    tex = tex.strip()

    # 2. Fix conflicting FontAwesome packages
    if r"\usepackage{fontawesome}" in tex and r"\usepackage{fontawesome5}" in tex:
        tex = tex.replace(r"\usepackage{fontawesome}", "% \\usepackage{fontawesome} (removed duplicate)")

    # 3. Fix underscores in \href{URL}{LABEL} URLs using string parsing (NOT regex).
    #    hyperref's \href{} handles raw _ in URLs natively.
    #    We must NOT use %5F (% is a TeX comment char!) or \_ (undefined in URL context).
    #    Just ensure URLs contain plain _ (strip any \_ the LLM may have added).
    def fix_href_urls(text):
        result = []
        i = 0
        marker = '\\href{'
        while i < len(text):
            pos = text.find(marker, i)
            if pos == -1:
                result.append(text[i:])
                break
            result.append(text[i:pos])
            j = pos + len(marker)
            url_start = j
            while j < len(text) and text[j] != '}':
                j += 1
            url = text[url_start:j]
            # Strip \_ back to raw _ — hyperref handles _ in URLs natively
            url_fixed = url.replace('\\_', '_')
            result.append(marker + url_fixed)
            i = j
        return ''.join(result)

    tex = fix_href_urls(tex)

    # 4. Process line by line for character escaping & deduplication
    lines = tex.split('\n')
    seen_packages = set()
    cleaned_lines = []
    past_begin_doc = False
    
    for line in lines:
        stripped = line.strip()
        
        if r'\begin{document}' in stripped:
            past_begin_doc = True

        # Deduplicate packages
        if stripped.startswith(r"\usepackage{fontawesome") or stripped.startswith(r"\usepackage{fontawesome5"):
            if stripped in seen_packages:
                continue
            seen_packages.add(stripped)
        
        # Only escape underscores/ampersands in document body text lines,
        # NEVER in the preamble or lines with TeX commands/hrefs
        if past_begin_doc and stripped and not stripped.startswith("%"):
            has_tex_cmd = any(stripped.startswith(c) for c in [
                r'\begin', r'\end', r'\documentclass', r'\usepackage', r'\input',
                r'\newcommand', r'\renewcommand', r'\pagestyle', r'\fancyhf',
                r'\addtolength', r'\setlength', r'\titleformat', r'\urlstyle',
                r'\raggedbottom', r'\raggedright', r'\pdfgentounicode',
                r'\section', r'\resumeSubHeadingListStart', r'\resumeSubHeadingListEnd',
                r'\resumeItemListStart', r'\resumeItemListEnd',
                r'\resumeSubheading', r'\resumeProjectHeading', r'\resumeItem',
                r'\resumeSubItem', r'\resumeSubSubheading',
            ])
            if not has_tex_cmd and r'\href{' not in line:
                line = re.sub(r'(?<!\\)_', r'\_', line)
                line = re.sub(r'(?<!\\)&', r'\&', line)

        cleaned_lines.append(line)

    res_tex = '\n'.join(cleaned_lines)

    # 5. Remove empty list environments
    res_tex = re.sub(r"\\resumeSubHeadingListStart\s*\\resumeSubHeadingListEnd", "", res_tex)
    res_tex = re.sub(r"\\resumeItemListStart\s*\\resumeItemListEnd", "", res_tex)
    res_tex = re.sub(r"\\begin\{itemize\}[^]]*\]?\s*\\end\{itemize\}", "", res_tex)
    
    # 6. Remove empty skill lines like \textbf{Databases}{: } \\
    res_tex = re.sub(r"\\textbf\{[^}]+\}\{:\s*\}\s*(\\\\)?", "", res_tex)

    # 7. Remove trailing \\ before closing braces or \end{itemize}
    res_tex = re.sub(r"\\\\\s*(\}\s*\\end\{itemize\})", r"\1", res_tex)
    res_tex = re.sub(r"\\\\\s*(\}\s*\})", r"\1", res_tex)

    # 8. Remove empty sections with no items
    res_tex = re.sub(r"\\section\{[^}]+\}\s*(?=\\section|\n\\end\{document\})", "", res_tex)

    # 9. Fix \resumeSubheading with wrong number of arguments (must have exactly 4 {} groups)
    def fix_subheading_args(match):
        full = match.group(0)
        # Count existing {} argument groups after \resumeSubheading
        args = re.findall(r'\{[^{}]*\}', full)
        while len(args) < 4:
            args.append('{}')
        return '\\resumeSubheading\n      ' + ''.join(args)
    
    res_tex = re.sub(
        r'\\resumeSubheading\s*(?:\{[^{}]*\}\s*){1,4}',
        fix_subheading_args,
        res_tex
    )

    return res_tex


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
    bullets: Optional[Any] = ""

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
    certifications: Optional[List[Any]] = []

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

        user_name = escape_latex(user.name or "")
        user_email = escape_latex(user.email or "")
        user_phone = escape_latex(user.phone or "")
        user_location = escape_latex(user.location or "")
        user_linkedin = escape_latex(user.linkedin or "")
        user_github = escape_latex(user.github or "")
        user_summary = escape_latex(user.summary or "")

        has_experience = bool(user.experience and len([e for e in user.experience if e.company or e.role]) > 0)
        has_education = bool(user.education and len([e for e in user.education if e.institution or e.degree]) > 0)

        # Build repository context blocks
        repo_facts = []
        for r in repos:
            name = escape_latex(r.name or r.fullName)
            desc = escape_latex(r.description or "No description provided.")
            lang = escape_latex(r.language or "Unknown")
            topics = escape_latex(", ".join(r.topics)) if r.topics else "None"
            
            clean_url = r.url or ""
            
            fact_block = f"### Repository: {name}\n- URL: {clean_url}\n- Primary Language: {lang}\n- Topics: {topics}\n- Description: {desc}\n"
            
            if r.userNotes:
                fact_block += f"- User Performance Notes: {escape_latex(r.userNotes)}\n"
            
            if r.readmeContent:
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

        user_info_str = f"""
Name: {user_name}
Email: {user_email}
Phone: {user_phone}
Location: {user_location}
LinkedIn: {user_linkedin}
GitHub: {user_github}
Portfolio: {user.portfolio}
Summary / Bio: {user_summary}

Education Provided: {has_education}
Education:
{json.dumps([e.dict() for e in user.education], indent=2) if user.education else "NONE (User provided no education)."}

Certifications & Achievements:
{json.dumps(user.certifications, indent=2) if user.certifications else "None provided."}

Work Experience Provided: {has_experience}
Experience Entries:
{json.dumps([e.dict() for e in user.experience], indent=2) if user.experience else "NONE (User provided no work experience)."}

Skills & Technologies:
{json.dumps(user.skills, indent=2) if user.skills else "Infer technical skills from projects and user input."}
"""

        system_prompt = f"""You are a LaTeX template filler. You receive a LaTeX resume template and user data. Your ONLY job is to replace the sample/placeholder data in the template with the user's real data. You must preserve the EXACT structure, macros, and formatting of the template.

ABSOLUTE RULES:
1. **PRESERVE TEMPLATE STRUCTURE EXACTLY**: Keep every \\documentclass, \\usepackage, \\newcommand, \\renewcommand, \\pagestyle, margin setting, and macro definition UNCHANGED. Only modify the content inside \\begin{{document}}...\\end{{document}}.

2. **USE ONLY MACROS DEFINED IN THE TEMPLATE**: The template defines macros like \\resumeSubheading{{}}{{}}{{}}{{}} (4 args), \\resumeProjectHeading{{}}{{}} (2 args), \\resumeItem{{}}, \\resumeSubHeadingListStart, \\resumeSubHeadingListEnd, \\resumeItemListStart, \\resumeItemListEnd. Use ONLY these. NEVER invent new macros or use \\begin{{center}} for sections.

3. **HEADER**: Replace the name, phone, email, linkedin, github in the header \\begin{{center}} block. Keep the exact same \\href and \\underline structure from the template.

4. **SECTIONS**: For each section, follow the template's pattern exactly:
   - Education: Use \\resumeSubheading{{Institution}}{{Location}}{{Degree}}{{Dates}} inside \\resumeSubHeadingListStart...\\resumeSubHeadingListEnd
   - Experience: Use \\resumeSubheading{{Role}}{{Dates}}{{Company}}{{Location}} with \\resumeItemListStart...\\resumeItemListEnd containing \\resumeItem{{}} entries
   - Projects: Use \\resumeProjectHeading{{\\textbf{{Name}} $|$ \\emph{{Tech}}}}{{Dates}} with \\resumeItemListStart...\\resumeItemListEnd containing \\resumeItem{{}} entries
   - Technical Skills: Use the exact \\begin{{itemize}} pattern from the template with \\textbf{{Category}}{{: values}}

5. **CONDITIONAL SECTIONS**:
   - Work Experience Provided = {has_experience}. If FALSE, DELETE the entire Experience section (\\section and its list block).
   - Education Provided = {has_education}. If FALSE, DELETE the entire Education section.
   - If user provided a Summary/Bio, add a \\section{{Summary}} with plain text underneath (NO \\begin{{center}}, just the text).
   - If user provided Certifications, add a \\section{{Certifications}} with \\resumeSubHeadingListStart containing \\resumeSubheading entries.

6. **HYPERLINKS IN PROJECTS**: For project GitHub links, use \\href{{URL}}{{\\underline{{GitHub}}}} where URL contains raw underscores (NOT \\_ or %5F). hyperref handles underscores in URLs.

7. **NO EMPTY ENVIRONMENTS**: Never output \\resumeSubHeadingListStart...\\resumeSubHeadingListEnd without items. Never output \\resumeItemListStart...\\resumeItemListEnd without items. Delete empty sections entirely.

8. **NO TRAILING BACKSLASHES**: Never put \\\\ after the last \\textbf{{}}{{:}} line in skills. Never put \\\\ after a \\resumeSubheading call.

9. **PAGE BUDGET**: Fit on {target_pages} page(s). Use 2-3 concise bullet points per project/experience.

10. **ZERO HALLUCINATION**: Use ONLY the provided user data. Do NOT invent metrics, companies, or dates.

11. **OUTPUT FORMAT**: Return ONLY raw LaTeX starting with % or \\documentclass. NO markdown code blocks."""

        user_prompt = f"""CANDIDATE DATA:
{user_info_str}

GITHUB REPOSITORIES:
{repo_context_str}

TEMPLATE TO FILL (preserve structure exactly, only replace data):
{template_tex}

Fill in the template with the candidate's data now. Return the complete LaTeX document."""

        edited_tex = ""

        groq_api_key = os.getenv("GROQ_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")

        print(f"[LLM_BRAIN] Template size: {len(template_tex)} chars, Repos: {len(repos)}, Has exp: {has_experience}, Has edu: {has_education}")
        
        if groq_api_key:
            try:
                from langchain_groq import ChatGroq
                llm = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=groq_api_key, temperature=0.1)
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
            edited_tex = fallback_latex_filler(template_tex, user, repos, has_experience, has_education)

        # Sanitize and post-process generated LaTeX code
        edited_tex = sanitize_generated_tex(edited_tex)

        # CRITICAL: Force-replace preamble with original template preamble.
        # The LLM often drops/modifies \newcommand definitions, causing
        # "Undefined control sequence" errors. By using the original preamble,
        # all macros (\resumeItemListEnd, \resumeSubheading, etc.) are guaranteed defined.
        begin_doc_marker = r'\begin{document}'
        orig_preamble_idx = template_tex.find(begin_doc_marker)
        llm_body_idx = edited_tex.find(begin_doc_marker)
        if orig_preamble_idx > 0 and llm_body_idx > 0:
            original_preamble = template_tex[:orig_preamble_idx]  # everything before \begin{document}
            llm_body = edited_tex[llm_body_idx:]  # \begin{document} ... \end{document}
            edited_tex = original_preamble + llm_body
            print("[LLM_BRAIN] Preamble replaced with original template preamble (macros guaranteed).")

        lines = edited_tex.split('\n')
        print(f"[LLM_BRAIN] Generated TeX: {len(lines)} total lines")
        print(f"[LLM_BRAIN] First 5 lines:")
        for i, l in enumerate(lines[:5], start=1):
            print(f"  {i}: {l}")
        has_docclass = r'\documentclass' in edited_tex
        has_begindoc = r'\begin{document}' in edited_tex
        has_enddoc = r'\end{document}' in edited_tex
        print(f"[LLM_BRAIN] Has \\documentclass: {has_docclass}, Has \\begin{{document}}: {has_begindoc}, Has \\end{{document}}: {has_enddoc}")
        print("[LLM_BRAIN] Generated TeX (Lines 110-160):")
        for i, l in enumerate(lines[109:160], start=110):
            print(f"{i}: {l}")
        print("="*60)

        return {"ok": True, "tex": edited_tex, "summary": "Resume generated successfully with zero hallucination context."}

    except Exception as err:
        print(f"[LLM_BRAIN] Exception: {err}")
        raise HTTPException(status_code=500, detail=str(err))


def editedTex_has_value(text: str) -> bool:
    return bool(text and r"\documentclass" in text)


def fallback_latex_filler(tex: str, user: UserProfile, repos: List[RepoDetail], has_experience: bool, has_education: bool) -> str:
    """Fallback deterministic filler when no LLM API key is present."""
    result = tex

    name = escape_latex(user.name or "Candidate Name")
    email = escape_latex(user.email or "user@example.com")
    phone = escape_latex(user.phone or "123-456-7890")

    # Basic header replacements
    result = re.sub(r"(\\name\{)[^}]*(\})", rf"\1{name}\2", result)
    result = re.sub(r"(\\href\{mailto:[^}]*\}\{)[^}]*(\})", rf"\1{email}\2", result)

    # If experience is empty, remove experience section
    if not has_experience:
        result = re.sub(r"\\section\{Experience\}.*?(?=\\section|\n\\end\{document\})", "", result, flags=re.DOTALL | re.IGNORECASE)
        result = re.sub(r"\\section\{Work Experience\}.*?(?=\\section|\n\\end\{document\})", "", result, flags=re.DOTALL | re.IGNORECASE)

    # If education is empty, remove education section
    if not has_education:
        result = re.sub(r"\\section\{Education\}.*?(?=\\section|\n\\end\{document\})", "", result, flags=re.DOTALL | re.IGNORECASE)

    return result


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
