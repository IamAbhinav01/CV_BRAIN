"""
LaTeX sanitizer — post-processing utilities to clean up LLM-generated LaTeX.
Extracted from the original main.py with no logic changes.
"""

import re
from typing import List
from app.models import UserProfile, RepoDetail


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
    """Clean up markdown code blocks, duplicate package includes, empty list
    environments, trailing linebreaks, and unescaped underscores/ampersands."""
    if not isinstance(tex, str):
        return ""

    # 1. Clean markdown code wraps if present
    tex = re.sub(r"^```(?:latex|tex)?\n", "", tex.strip(), flags=re.MULTILINE)
    tex = re.sub(r"\n```$", "", tex.strip(), flags=re.MULTILINE)
    tex = tex.strip()

    # 1b. Strip any natural language text before the actual LaTeX content
    for marker in [r'\documentclass', '%']:
        idx = tex.find(marker)
        if idx > 0:
            tex = tex[idx:]
            break
    tex = tex.strip()

    # 2. Fix conflicting FontAwesome packages
    if r"\usepackage{fontawesome}" in tex and r"\usepackage{fontawesome5}" in tex:
        tex = tex.replace(
            r"\usepackage{fontawesome}",
            r"% \usepackage{fontawesome} (removed duplicate)",
        )

    # 3. Fix underscores in \href{URL}{LABEL} URLs using string parsing.
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
        if stripped.startswith(r"\usepackage{fontawesome") or stripped.startswith(
            r"\usepackage{fontawesome5"
        ):
            if stripped in seen_packages:
                continue
            seen_packages.add(stripped)

        # Only escape underscores/ampersands in document body text lines,
        # NEVER in the preamble or lines with TeX commands/hrefs
        if past_begin_doc and stripped and not stripped.startswith("%"):
            has_tex_cmd = any(
                stripped.startswith(c)
                for c in [
                    r'\begin', r'\end', r'\documentclass', r'\usepackage', r'\input',
                    r'\newcommand', r'\renewcommand', r'\pagestyle', r'\fancyhf',
                    r'\addtolength', r'\setlength', r'\titleformat', r'\urlstyle',
                    r'\raggedbottom', r'\raggedright', r'\pdfgentounicode',
                    r'\section', r'\resumeSubHeadingListStart', r'\resumeSubHeadingListEnd',
                    r'\resumeSubheading', r'\resumeProjectHeading', r'\resumeItem',
                    r'\resumeSubItem', r'\resumeSubSubheading',
                    r'\cvsection', r'\cvevent', r'\cventry', r'\cvitem', r'\cvtag',
                    r'\cvskill', r'\namesection', r'\runsubsection', r'\descript',
                    r'\location', r'\sectionsep', r'\name', r'\position', r'\mobile',
                    r'\email', r'\homepage', r'\github', r'\linkedin', r'\social',
                    r'\makecvheader', r'\makecvfooter', r'\headerrow', r'\contactline',
                    r'\moderncvstyle', r'\moderncvcolor', r'\photo', r'\title',
                    r'\address', r'\phone', r'\born', r'\quote',
                    r'\fontdir', r'\colorlet', r'\definecolor', r'\geometry',
                    r'\divider', r'\switchcolumn', r'\columnratio',
                ]
            )
            if not has_tex_cmd and r'\href{' not in line:
                line = re.sub(r'(?<!\\)_', r'\_', line)
                line = re.sub(r'(?<!\\)&', r'\&', line)

        cleaned_lines.append(line)

    res_tex = '\n'.join(cleaned_lines)

    # 5. Remove empty list environments
    res_tex = re.sub(
        r"\\resumeSubHeadingListStart\s*\\resumeSubHeadingListEnd", "", res_tex
    )
    res_tex = re.sub(r"\\resumeItemListStart\s*\\resumeItemListEnd", "", res_tex)
    res_tex = re.sub(
        r"\\begin\{itemize\}[^]]*\]?\s*\\end\{itemize\}", "", res_tex
    )

    # 6. Remove empty skill lines like \textbf{Databases}{: } \\
    res_tex = re.sub(r"\\textbf\{[^}]+\}\{:\s*\}\s*(\\\\)?", "", res_tex)

    # 7. Remove trailing \\ before closing braces or \end{itemize}
    res_tex = re.sub(r"\\\\\s*(\}\s*\\end\{itemize\})", r"\1", res_tex)
    res_tex = re.sub(r"\\\\\s*(\}\s*\})", r"\1", res_tex)

    # 8. Remove empty sections with no items
    res_tex = re.sub(
        r"\\section\{[^}]+\}\s*(?=\\section|\n\\end\{document\})", "", res_tex
    )

    # 9. Fix \resumeSubheading with wrong number of arguments
    def fix_subheading_args(match):
        full = match.group(0)
        args = re.findall(r'\{[^{}]*\}', full)
        while len(args) < 4:
            args.append('{}')
        return '\\resumeSubheading\n      ' + ''.join(args)

    res_tex = re.sub(
        r'\\resumeSubheading\s*(?:\{[^{}]*\}\s*){1,4}',
        fix_subheading_args,
        res_tex,
    )

    return res_tex


def replace_preamble_with_original(edited_tex: str, template_tex: str) -> str:
    """
    Force-replace the LLM's preamble with the original template preamble.
    The LLM often drops/modifies \\newcommand definitions, causing
    'Undefined control sequence' errors.
    """
    begin_doc_marker = r'\begin{document}'
    orig_preamble_idx = template_tex.find(begin_doc_marker)
    llm_body_idx = edited_tex.find(begin_doc_marker)

    if orig_preamble_idx > 0 and llm_body_idx > 0:
        original_preamble = template_tex[:orig_preamble_idx]
        llm_body = edited_tex[llm_body_idx:]
        edited_tex = original_preamble + llm_body
        print("[LLM_BRAIN] Preamble replaced with original template preamble (macros guaranteed).")

    return edited_tex


def fallback_latex_filler(
    tex: str,
    user: UserProfile,
    repos: List[RepoDetail],
    has_experience: bool,
    has_education: bool,
) -> str:
    """Fallback deterministic filler when no LLM API key is present or LLM fails."""
    result = tex

    name = escape_latex(user.name or "Candidate Name")
    email = escape_latex(user.email or "user@example.com")

    # Basic header replacements
    result = re.sub(r"(\\name\{)[^}]*(})", rf"\1{name}\2", result)
    result = re.sub(
        r"(\\href\{mailto:[^}]*\}\{)[^}]*(})", rf"\1{email}\2", result
    )

    # If experience is empty, remove experience section
    if not has_experience:
        result = re.sub(
            r"\\section\{Experience\}.*?(?=\\section|\n\\end\{document\})",
            "",
            result,
            flags=re.DOTALL | re.IGNORECASE,
        )
        result = re.sub(
            r"\\section\{Work Experience\}.*?(?=\\section|\n\\end\{document\})",
            "",
            result,
            flags=re.DOTALL | re.IGNORECASE,
        )

    # If education is empty, remove education section
    if not has_education:
        result = re.sub(
            r"\\section\{Education\}.*?(?=\\section|\n\\end\{document\})",
            "",
            result,
            flags=re.DOTALL | re.IGNORECASE,
        )

    return result
