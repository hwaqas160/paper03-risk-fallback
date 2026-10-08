# IEEE T-IV author guidelines (fetched 2026-09-29)

`IEEE-TIV-Guidelines.txt` was not found in the project folder, so this was fetched directly from
the IEEE ITSS T-IV page (https://ieee-itss.org/pub/t-iv/) and the journal's Xplore listing. Replace
this file with the actual guidelines document if you have a more current or official copy; check the
numbers below against it before final submission, since anything fetched from the web can be stale.

## Page limits and charges
- Regular paper: **10 pages** suggested length (our manuscript is exactly 10 pages, including
  references and biographies; exploratory analyses are in `supplementary.pdf`)
- Perspectives/Letters/Communications: 5 pages
- Survey papers: 18 pages
- Practitioner papers: 6 pages
- Overlength charge: $175 USD per page beyond the suggested length, billed after acceptance

## Abstract and keywords
- **150-250 words**, single paragraph, no abbreviations, no footnotes
- **3-4 keywords**
- Current draft: 227 words, 4 keywords

## Required files at submission
- PDF of the paper
- Identical source file (LaTeX source + .cls, or Word)
- High-definition figures/tables as separate files (PS, EPS, PDF, PNG, TIFF accepted)
- Short biography for every author (required at submission per this source, not only on acceptance)
- Author photos, high resolution (bio-sketch requirement)

## Required disclosures / checklist items
- Conflict of interest disclosure
- Institutional email address for every co-author
- No non-English characters in author information
- (Resubmission only) detailed response letter + highlighted revised manuscript

## Formatting
- IEEE double-column template (IEEEtran.cls, `journal` mode) -- already in use
- References: hanging indent, justified
- Single-blind peer review, at least two independent reviewers

## Open access
- Hybrid journal: traditional (free) or author-pays OA ($2,495 USD, waivers available for
  low-income-country corresponding authors)

## Submission portal
- https://ieee.atyponrex.com/journal/t-iv
- Editorial contact: tiv-eic@ieee.org

## Outstanding items for THIS manuscript, updated 2026-10-02
- [x] Trim abstract to 150-250 words (done 2026-09-29)
- [x] Trim keywords to 3-4 (done 2026-09-29)
- [x] Write short biography for each of the 3 authors (done 2026-09-30)
- [x] Confirm institutional email for all 3 authors -- confirmed by the corresponding author
      2026-09-30: rahat.mscm21nbs@student.nust.edu.pk, hina.maryam@sbp.org.pk (both match what
      was already in the manuscript)
- [x] Reserve a Zenodo DOI for the code/data release: `10.5281/zenodo.23054382`
      (draft, unpublished -- see `zenodo.md`; publish it before submitting so the DOI resolves)
- [x] Route: **traditional (free)**, decided 2026-10-02. No open-access fee. The paper is held
      at 10 pages so no overlength charge ($175/page) applies either; keep it at 10 pages through
      revisions.
- [x] Conflict-of-interest statement in the manuscript (2026-10-02); answer the portal's COI form
      the same way
- [x] Funding: no external funding, stated in the first-page footnote (2026-10-02)
- [x] Supplementary Material (`supplementary.pdf`, source `supplementary.tex`): class-conditional
      comparison, reverse shift, few-label recalibration table, simulator sensitivity. Upload it
      as "Supplementary Material" in the portal.
- [x] Block diagram (Fig. 1) exported as `figures/protocol.pdf` (source `paper/figures/protocol.tex`)
- [x] Hassan's high-resolution author photo added to the biography section 2026-09-30
      (`paper/hassan_photo.jpeg`). Rahat's and Hina's photos are still pending -- their
      biographies currently use `\IEEEbiographynophoto`; swap to `\IEEEbiography` with a photo
      once available.

## Update 2026-10-08
- [x] 2025-2026 references verified against arXiv / publisher pages (all matched).
- [x] Paper reframed (new title); 10 pages; supplement has 18 sections (S1-S18).
- [ ] Rahat's and Hina's photos; publish Zenodo (refresh archive: new scripts src/cswc.py, review_fixes.py, results/final/r11_*.json); final read.
