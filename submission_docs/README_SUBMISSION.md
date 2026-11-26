# SERRA submission package — 26 November 2026 (revised after three pre-submission reviews)

**Journal:** *Stochastic Environmental Research and Risk Assessment* (Springer Nature, Editorial Manager)
**Article type:** Research Article (Original Paper)
**Title:** Can the hidden state of a recurrent precipitation model monitor drought onset? A
planned event-based evaluation of backward-coherence regularization
**Authors:** Hui-Mean Foo (ORCID 0000-0003-4200-5849); Yuan-chin Ivan Chang* (corresponding,
ORCID 0000-0002-4977-7721), Institute of Statistical Science, Academia Sinica, Taipei 11529, Taiwan

This folder is built by `python3 make_submission.py SUBMISSION_SERRA_2026_1126` from the master files
one level up (git repository knight-ivan/rmrnn-drought-monitoring, tag `serra-submission-2026-11-26`).
Edit there, not here. It replaces the packages of 12, 13 and 14 November; the title now reads "A planned
event-based evaluation", and the paper uses US spelling throughout.
`../submission_docs/REVISION_NOTES.md` lists what changed and why; `../submission_docs/Email_to_Foo.txt`
is the note to the co-author.

## What to upload, and as which item type

The journal compiles the manuscript itself from a LaTeX zip ("LaTeX documents with figures and
tables compressed into a .zip format"); the cover letter is not part of that zip and is uploaded as
its own item.

| File in `01_Upload_to_Editorial_Manager/` | Editorial Manager item |
|---|---|
| `01_Cover_Letter_SERRA.pdf` (2 pp) | Cover Letter (separate upload) |
| `02_Manuscript_LaTeX_source.zip` | Manuscript — LaTeX source zip; the system builds the review PDF |
| `03_Online_Resource_1_Supplementary_Material.pdf` (20 pp) | Supplementary Material (ESM) |
| `04_Online_Resource_2_Code_Protocol_Results.zip` (6 MB) | Supplementary Material (ESM) |

`02_Reference_only_do_not_upload/Manuscript_RM_Precipitation_SERRA_master.pdf` (31 pp) is our own
build. Do not upload it; use it to check the PDF that Editorial Manager builds from the zip (page
count, figures, tables, references) before you approve the submission.

`02_Manuscript_LaTeX_source.zip` is flat (no subfolders) and self-contained: one main document
`RM_Precipitation_SERRA.tex` (the only file with `\documentclass`), the `.bbl` (so no BibTeX run is
needed), `refs.bib`, `sn-jnl.cls`, `sn-mathphys-ay.bst`, the 6 figure PDFs, the 10 table files and
`numbers.tex`. It compiles with pdflatex. References to Online Resource 1 are written in as plain
text. The cover letter is not in it. The build script unzips it into an empty folder, compiles it and
checks that the text equals the master PDF.

`04_Online_Resource_2_...zip` holds the code (all original and all new scripts), `PROTOCOL.md` with
its three amendments, `reproduce.sh`, `requirements.txt`, every result CSV, the station lists, the
reference events, the figures, the run logs, the LaTeX sources and a SHA-256 manifest. Running
`sh reproduce.sh` inside it rebuilds every table and number of the paper from its own result files
and compares them with the published ones; the build script runs that check on an unzipped copy.
It does not hold the score panels (200 MB), the per-seed scores or the raw downloads.

Checks made by the build script: no unresolved reference ("??") in any PDF, all fonts embedded,
abstract 150–250 words, LaTeX zip holds one main document and no cover letter and compiles alone,
Online Resource 2 verifies itself.

## Form fields (copy and paste)

**Abstract** (239 words; the limit is 150–250): see `abstract.txt` in this folder.

**Keywords (6):** sequential change detection; CUSUM; drought monitoring; standardized precipitation
index; recurrent neural networks; event-based verification

**Funding:** National Science and Technology Council, Taiwan, grant 115-2118-M-001-004-

**Competing interests:** The authors have no relevant financial or non-financial interests to disclose.

**Author contributions (CRediT):**
- Hui-Mean Foo: data curation, formal analysis, validation, writing (review and editing).
- Yuan-chin Ivan Chang: conceptualization, methodology, software, formal analysis, writing (original draft).

**Data availability:** All data are public: GHCN-Daily (doi:10.7289/V5D21VHZ), the Raingel archive of
Central Weather Administration records, and MERRA-2 through NASA POWER. Station lists, reference
events and all result tables are in Online Resource 2.

**Code availability:** Online Resource 2 (submitted); the same material is at
https://github.com/knight-ivan/rmrnn-drought-monitoring (private; made public on acceptance).

**Ethics / human or animal participants:** Not applicable.

**AI use:** documented in the Methods (subsection "Software"), in generic wording.

## Author to-do before clicking Submit
- [ ] Read `../submission_docs/REVISION_NOTES.md`, in particular "Statements made on your behalf".
- [ ] Co-author (Foo) has read and approved the final PDFs, including the CRediT roles.
- [ ] Push the repository (`git push origin main --tags`).
- [ ] In the environment that trained the models, run `pip freeze > requirements-lock.txt` and add it
      to the repository, so that the exact package versions are on record.
- [ ] Decide where the score panels (`results/panel*/*.npz`, 200 MB) will be deposited (for example
      Zenodo, embargoed until acceptance). The supplement says they will be deposited with the
      public release.
- [ ] Enter both ORCID iDs and the declarations in the Editorial Manager form.
- [ ] After upload, open the PDF that Editorial Manager builds from the LaTeX zip and compare it with
      `02_Reference_only_do_not_upload/Manuscript_RM_Precipitation_SERRA_master.pdf` (31 pp).
- [x] Cover letter addressed to the Editor-in-Chief, Dr. Andreas Langousis (University of Patras), as listed on
      the journal's editorial-board page on 26 November 2026. Check the page again if you submit much later.
- [ ] Optional: suggested and opposed reviewers (each needs an institutional email address).
- [ ] After acceptance: make the repository public.

## Rebuilding
```
sh reproduce.sh                            # check tables and numbers against results/
python3 code/serra/make_tables.py          # tables and numbers from results/
python3 code/serra/figures.py && python3 code/serra/figures.py composite_mv   # needs the score panels
python3 make_submission.py SUBMISSION_SERRA_2026_1126
```
