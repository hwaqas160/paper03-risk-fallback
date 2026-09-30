"""Manuscript QA: compile status, pages, abstract words, keywords, undefined refs, British spellings."""
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEX = HERE / "main.tex"


def compile_tex():
    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"], cwd=HERE,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    if "--no-compile" not in sys.argv:
        compile_tex()
    s = TEX.read_text(encoding="utf-8")
    log = (HERE / "main.log").read_text(encoding="utf-8", errors="replace")
    ab = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", s, re.S).group(1)
    ab_words = len(re.sub(r"\\[a-zA-Z]+|[{}~]|\\,", " ", ab).split())
    kw = re.search(r"\\begin\{IEEEkeywords\}(.*?)\\end\{IEEEkeywords\}", s, re.S).group(1)
    n_kw = len([k for k in kw.split(",") if k.strip()])
    pages = re.search(r"Output written on main\.pdf \((\d+) pages", log)
    errors = re.findall(r"^! .*", log, re.M)
    undef = re.findall(r"(?:Citation|Reference) `([^']+)' .*undefined", log)
    overfull = [float(x) for x in re.findall(r"Overfull \\hbox \(([\d.]+)pt", log)]
    body = s[: s.index("\\begin{thebibliography}")]
    brit = re.findall(r"\b\w*(?:manoeuvr|behaviour|labell|organis|modell|minimis|realis|favour|isation)\w*\b",
                      body + s[s.index("\\end{thebibliography}"):], re.I)
    n_bib = len(re.findall(r"\\bibitem\{", s))
    cited = set(k.strip() for c in re.findall(r"\\cite\{([^}]+)\}", s) for k in c.split(","))
    bibkeys = set(re.findall(r"\\bibitem\{([^}]+)\}", s))
    rows = [
        ("pages", pages.group(1) if pages else "?", "<= 10"),
        ("abstract words", ab_words, "150-250"),
        ("keywords", n_kw, "3-4"),
        ("LaTeX errors", len(errors), "0"),
        ("undefined refs/cites", len(set(undef)), "0"),
        ("max overfull (pt)", max(overfull) if overfull else 0, "< 1"),
        ("British spellings", len(brit), "0"),
        ("references", n_bib, "~40"),
        ("bibitems never cited", sorted(bibkeys - cited), "[]"),
        ("cited but no bibitem", sorted(cited - bibkeys), "[]"),
    ]
    for name, val, target in rows:
        print(f"{name:<24} {str(val):<40} target {target}")
    if errors:
        print("\n".join(errors[:5]))
    if undef:
        print("undefined:", sorted(set(undef)))


if __name__ == "__main__":
    main()
