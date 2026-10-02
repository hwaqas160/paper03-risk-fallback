"""List bibliography entries by publication year (newest year mentioned in each entry)."""
import re
from pathlib import Path

s = Path(__file__).with_name("main.tex").read_text(encoding="utf-8")
bib = s[s.index("\\begin{thebibliography}"):s.index("\\end{thebibliography}")]
entries = re.split(r"\\bibitem\{", bib)[1:]
rows = []
for e in entries:
    key = e.split("}", 1)[0]
    body = e.split("}", 1)[1]
    years = [int(y) for y in re.findall(r"\b(19\d\d|20\d\d)\b", body) if not re.search(r"arXiv:" + y, body)]
    years = [y for y in years if 1990 <= y <= 2026]
    rows.append((max(years) if years else 0, key))
rows.sort()
new = [r for r in rows if r[0] >= 2024]
old = [r for r in rows if r[0] < 2024]
print(f"total {len(rows)} | 2024 or later: {len(new)} | before 2024: {len(old)}")
for y, k in old:
    print(f"  {y}  {k}")
