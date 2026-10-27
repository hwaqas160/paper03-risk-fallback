# IEEE T-IV format self-check (prepared 2026-10-16 against `IEEE-TIV-Guidelines.txt`)

Paper: "Certifying the Outcome of Autonomous Vehicle Fallback Triggers: Closed-Loop Evaluation Under Latency and Dataset Shift". Submission type: Regular Paper (10 pages suggested).
Items marked **AUTHOR** can only be confirmed by the authors. Everything else was checked against `paper/main.tex` and `paper/main.pdf`.

## Checklist
- [x] Conflict of interest disclosed: "Conflict of Interest" section in the manuscript (no competing interests). Answer the portal form the same way.
- [x] Institutional email address of both co-authors: Hassan hsaleem.msee20seecs@seecs.edu.pk (NUST), Rahat rahat.mscm21nbs@student.nust.edu.pk (NUST). The paper now has TWO authors (Hina Maryam removed on 2026-10-27 at the author's request); page-1 footnotes match the biographies (public-sector and private-sector organizations).
- [x] No non-English characters in the author information (the author block and biographies are plain ASCII).
- [ ] Detailed response letter: resubmission only, not applicable.
- [ ] Highlighted revised manuscript: resubmission only, not applicable.

## Manuscript layout
- [x] IEEE template: `\documentclass[lettersize,journal]{IEEEtran}`, IEEEtran.cls V1.8b (2015/08/26), the standard journal template.
- [x] Font type and size: IEEEtran defaults, with no size or font overrides in the body (tables use `\scriptsize`, as is usual).
- [x] Double-column layout, standard line and paragraph spacing: no `\vspace`, `\baselinestretch`, `\parskip` or page-geometry changes anywhere in `main.tex`.
- [x] Title and subtitle format: three-line title in IEEE title case; no subtitle.
- [x] Author list under the title: names only. Affiliations, emails and ORCID are in the first-page footnotes, as the IEEEtran template produces them.
- [x] Standard format of figures and tables: table captions above, figure captions below, IEEE caption style. Figures remain legible in grayscale (checked at 70 dpi).
- [x] References: IEEEtran `thebibliography` (justified, hanging indent). Journal and proceedings names use IEEE abbreviations, and two inconsistencies (a missing "Proc." and an un-italicized arXiv entry) were fixed on 2026-10-16.
- [x] Short biographies for both authors at the end of the paper, text only. T-IV asks for photographs only with the final files after acceptance, so none is in the initial submission.

## Length and abstract
- [x] Regular paper, 10 pages including references and biographies (limit: 10; overlength would be $175 per page). `paper/qa_check.py` reports 10.
- [x] Abstract 249 words (150-250), one paragraph, no abbreviations, footnotes, references, equations or tables. "minimum-risk maneuver" is spelled out; "Hz" is a unit.
- [x] Four keywords: Autonomous driving, conformal prediction, minimum-risk maneuver, distribution shift.

## Other
- [x] Route: traditional (free), chosen 2026-10-02. No open-access fee. The guidelines grant waived APCs only to corresponding authors from low-income countries; do not rely on that waiver.
- [x] PDF fonts: all 23 embedded, no Type 3 fonts.
- [x] Supplementary Material: `supplementary.pdf`, upload as a separate file.
- [x] Code and data DOI 10.5281/zenodo.23054382 is published (2026-10-19) with the refreshed archive and resolves; GitHub repository is up to date.
- [ ] **AUTHOR** Create or confirm the account on the TIV Author Portal; a corresponding author for copyright transfer after acceptance.
- Single-blind review: author names are shown to reviewers, so no anonymization is needed. All articles are screened for plagiarism before review, so make sure any text shared with the authors' other papers or preprints is cited (not checked here).
