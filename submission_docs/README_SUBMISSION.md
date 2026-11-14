# SERRA submission package — 14 November 2026 (revised after two pre-submission reviews)

**Journal:** *Stochastic Environmental Research and Risk Assessment* (Springer Nature, Editorial Manager)
**Article type:** Research Article (Original Paper)
**Title:** Can the hidden state of a recurrent precipitation model monitor drought onset? A
pre-specified event-based evaluation of backward-coherence regularisation
**Authors:** Hui-Mean Foo (ORCID 0000-0003-4200-5849); Yuan-chin Ivan Chang* (corresponding,
ORCID 0000-0002-4977-7721), Institute of Statistical Science, Academia Sinica, Taipei 11529, Taiwan

This folder is built by `python3 make_submission.py SUBMISSION_SERRA_2026_1114` from the master files
one level up (git repository knight-ivan/rmrnn-drought-monitoring, tag `serra-submission-2026-11-14`).
Edit there, not here. It replaces the packages of 12 and 13 November.
`../submission_docs/REVISION_NOTES.md` lists what changed and why; `../submission_docs/Email_to_Foo.txt`
is the note to the co-author.

## What to upload, and as which item type

| File in `01_Upload_to_Editorial_Manager/` | Editorial Manager item |
|---|---|
| `01_Cover_Letter_SERRA.pdf` (2 pp) | Cover Letter |
| `02_Manuscript_RM_Precipitation_SERRA.pdf` (31 pp) | Manuscript (PDF) |
| `03_Online_Resource_1_Supplementary_Material.pdf` (20 pp) | Supplementary Material (ESM) |
| `04_Manuscript_LaTeX_source.zip` | LaTeX Source Files |
| `05_Online_Resource_2_Code_Protocol_Results.zip` (6 MB) | Supplementary Material (ESM) |

`04_Manuscript_LaTeX_source.zip` is flat and self-contained: main `.tex`, `.bbl`, `refs.bib`,
`sn-jnl.cls`, `sn-mathphys-ay.bst`, 6 figure PDFs and 10 table files. References to Online Resource 1
are written in as plain text. The build script unzips it into an empty folder, compiles it and checks
that the text equals the manuscript PDF.

`05_Online_Resource_2_...zip` holds the code (all original and all new scripts), `PROTOCOL.md` with
its three amendments, `reproduce.sh`, `requirements.txt`, every result CSV, the station lists, the
reference events, the figures, the run logs, the LaTeX sources and a SHA-256 manifest. Running
`sh reproduce.sh` inside it rebuilds every table and number of the paper from its own result files
and compares them with the published ones; the build script runs that check on an unzipped copy.
It does not hold the score panels (200 MB), the per-seed scores or the raw downloads.

Checks made by the build script: no unresolved reference ("??") in any PDF, all fonts embedded,
abstract 150–250 words, LaTeX source compiles alone, Online Resource 2 verifies itself.

## Form fields (copy and paste)

**Abstract** (249 words; the limit is 150–250): see `abstract.txt` in this folder.

**Keywords (6):** sequential change detection; CUSUM; drought monitoring; standardized precipitation
index; recurrent neural networks; event-based verification

**Funding:** National Science and Technology Council, Taiwan, grant 115-2118-M-001-004-

**Competing interests:** The authors have no relevant financial or non-financial interests to disclose.

**Author contributions (CRediT):**
- Hui-Mean Foo: data curation, formal analysis, validation, writing (review and editing).
- Yuan-chin Ivan Chang: conceptualisation, methodology, software, formal analysis, writing (original draft).

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
- [ ] Push the repository (`git push origin main serra-submission-2026-11-13 serra-submission-2026-11-14`).
- [ ] In the environment that trained the models, run `pip freeze > requirements-lock.txt` and add it
      to the repository, so that the exact package versions are on record.
- [ ] Decide where the score panels (`results/panel*/*.npz`, 200 MB) will be deposited (for example
      Zenodo, embargoed until acceptance). The supplement says they will be deposited with the
      public release.
- [ ] Enter both ORCID iDs and the declarations in the Editorial Manager form.
- [ ] Optional: add the Editor-in-Chief's name to the cover letter (it says "Dear Editor").
- [ ] Optional: suggested and opposed reviewers (each needs an institutional email address).
- [ ] After acceptance: make the repository public.

## Rebuilding
```
sh reproduce.sh                            # check tables and numbers against results/
python3 code/serra/make_tables.py          # tables and numbers from results/
python3 code/serra/figures.py && python3 code/serra/figures.py composite_mv   # needs the score panels
python3 make_submission.py SUBMISSION_SERRA_2026_1114
```
