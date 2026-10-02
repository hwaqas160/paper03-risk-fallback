# IEEE T-IV submission folder

Snapshot of the files to upload to the T-IV submission portal
(https://ieee.atyponrex.com/journal/t-iv). This is a COPY, not the working source; keep editing
`paper/main.tex` and re-copy here right before actually submitting, since this folder can go stale.

- `main.tex`, `IEEEtran.cls` — LaTeX source (source file requirement)
- `main.pdf` — compiled PDF
- `author_guidelines_source.md` — T-IV requirements fetched from the official ITSS page
  (2026-09-29); re-check against the real portal before submitting, since anything fetched from the
  web can be stale, and the checklist inside it lists what's still outstanding (bios, COI statement,
  OA/traditional route decision, etc.)

- `supplementary.tex`, `supplementary.pdf` — Supplementary Material (exploratory analyses)
- `figures/` — high-resolution figures as separate files (Fig. 1 block diagram and three plots)

To refresh this snapshot from the working copy:
```
cp paper/main.tex paper/main.pdf paper/IEEEtran.cls paper/hassan_photo.jpeg submission/IEEE-TIV/
cp paper/supplementary.tex paper/supplementary.pdf submission/IEEE-TIV/
cp paper/figures/*.pdf submission/IEEE-TIV/figures/
```
