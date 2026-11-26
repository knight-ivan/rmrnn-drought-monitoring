"""Assemble the journal submission package from the master files in this folder.

  python3 make_submission.py SUBMISSION_SERRA_2026_1113

It (1) compiles the manuscript, Online Resource 1 and the cover letter in the order the
cross-references need; (2) writes a flat, self-contained LaTeX source of the manuscript in which
references to Online Resource 1 are resolved to plain text, and compiles it in an empty folder;
(3) writes Online Resource 2 (code, protocol, result files, logs, manifest); (4) checks the PDFs
(no unresolved reference, fonts embedded, abstract length) and stops on any failure.
The score panels (results/panel*/*.npz), raw data and per-seed score files are not packaged.
"""
import hashlib, os, re, shutil, subprocess, sys, tempfile, zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(ROOT, "tex")
MAIN, SUPP, COVER = "RM_Precipitation_SERRA", "RM_Precipitation_SERRA_supplement", "Cover_Letter_SERRA"
OR2_ROOT = "rmrnn-drought-monitoring"
OR2_INCLUDE = ["README.md", "PROTOCOL.md", "LICENSE", "requirements.txt", "reproduce.sh", "make_submission.py",
               ".gitignore", "code", "results", "figures", "logs", "tex", "data"]
OR2_SKIP = re.compile(r"(__pycache__|\.DS_Store|/panel(_mv)?/[^/]*\.npz$|/scores(_mv)?/|\.aux$|/tex/[^/]*\.log$|\.out$|"
                      r"\.blg$|\.out\.txt$|_draft\.tex$|\.synctex\.gz$|\.pyc$|"
                      r"daily_precip_long\.csv$|temperature_long\.csv$|/power/|/era5land/|/data/ghcnd/|/data/taiwan_raingel/)")


def run(cmd, cwd):
    r = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.returncode:
        sys.exit(f"FAILED: {' '.join(cmd)}\n{r.stdout[-3000:]}")
    return r.stdout


def latex(name, cwd):
    return run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", name], cwd)


def build_master():
    latex(SUPP, TEX); latex(MAIN, TEX); run(["bibtex", MAIN], TEX)
    for _ in range(2):
        latex(MAIN, TEX); latex(SUPP, TEX)
    latex(MAIN, TEX); latex(COVER, TEX)


def pdftext(pdf):
    return run(["pdftotext", "-layout", pdf, "-"], ROOT)


def check_pdf(pdf, label):
    txt = pdftext(pdf)
    assert "??" not in txt, f"{label}: unresolved reference"
    fonts = run(["pdffonts", pdf], ROOT).splitlines()[2:]
    assert all(l.split()[-5] == "yes" for l in fonts), f"{label}: font not embedded"
    pages = int(re.search(r"Pages:\s+(\d+)", run(["pdfinfo", pdf], ROOT)).group(1))
    print(f"  {label}: {pages} pages, {len(fonts)} fonts embedded, no unresolved references")
    return pages


def supplement_labels():
    """label -> printed number, from the supplement's .aux (e.g. tab:matching -> S3)."""
    aux = open(os.path.join(TEX, SUPP + ".aux")).read()
    return dict(re.findall(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}", aux))


def flat_source(dst):
    """Flat, self-contained LaTeX source of the manuscript."""
    lab = supplement_labels()
    os.makedirs(dst)

    def resolve(text):
        def r(m):
            assert m.group(1) in lab, f"unknown supplement label {m.group(1)}"
            return lab[m.group(1)]
        return re.sub(r"\\ref\{S-([^}]+)\}", r, text)
    t = open(os.path.join(TEX, MAIN + ".tex")).read()
    t = t.replace("\\graphicspath{{../figures/}}", "\\graphicspath{{./}}")
    t = t.replace("\\usepackage{xr-hyper}\n\\externaldocument[S-]{RM_Precipitation_SERRA_supplement}\n", "")
    t = t.replace("\\input{tables/", "\\input{")
    open(os.path.join(dst, MAIN + ".tex"), "w").write(resolve(t))
    for f in sorted(set(re.findall(r"\\input\{tables/([^}]+)\}", open(os.path.join(TEX, MAIN + ".tex")).read()))):
        open(os.path.join(dst, f), "w").write(resolve(open(os.path.join(TEX, "tables", f)).read()))
    for f in sorted(set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", t))):
        shutil.copy(os.path.join(ROOT, "figures", f), dst)
    for f in [MAIN + ".bbl", "refs.bib", "sn-jnl.cls", "sn-mathphys-ay.bst"]:
        shutil.copy(os.path.join(TEX, f), dst)
    # clean-room compile
    tmp = tempfile.mkdtemp()
    for f in os.listdir(dst):
        shutil.copy(os.path.join(dst, f), tmp)
    latex(MAIN, tmp); run(["bibtex", MAIN], tmp); latex(MAIN, tmp); latex(MAIN, tmp); latex(MAIN, tmp)
    log = open(os.path.join(tmp, MAIN + ".log"), errors="replace").read()
    assert "undefined" not in log, "flat source: undefined reference or citation"
    norm = lambda s: re.sub(r"\s+", " ", s)
    same = norm(pdftext(os.path.join(tmp, MAIN + ".pdf"))) == norm(pdftext(os.path.join(TEX, MAIN + ".pdf")))
    assert same, "flat source: text differs from the master PDF"
    print(f"  flat LaTeX source: {len(os.listdir(dst))} files, compiles in an empty folder, text identical to the master PDF")
    shutil.rmtree(tmp)


def zip_dir(src, dst_zip, arc_root=""):
    with zipfile.ZipFile(dst_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for d, _, fs in sorted(os.walk(src)):
            for f in sorted(fs):
                p = os.path.join(d, f)
                z.write(p, os.path.join(arc_root, os.path.relpath(p, src)))


def online_resource_2(dst_zip):
    files = []
    for item in OR2_INCLUDE:
        p = os.path.join(ROOT, item)
        if os.path.isfile(p):
            files.append(item)
        for d, _, fs in os.walk(p):
            files += [os.path.relpath(os.path.join(d, f), ROOT) for f in fs]
    files = sorted(f for f in files if not OR2_SKIP.search("/" + f))
    manifest = "".join(f"{hashlib.sha256(open(os.path.join(ROOT, f), 'rb').read()).hexdigest()}  {f}\n" for f in files)
    with zipfile.ZipFile(dst_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in files:
            z.write(os.path.join(ROOT, f), os.path.join(OR2_ROOT, f))
        z.writestr(os.path.join(OR2_ROOT, "MANIFEST.sha256"), manifest)
    need = ["PROTOCOL.md", "reproduce.sh", "code/serra/matching_checks.py", "code/serra/incremental_forward.py",
            "code/serra/evaluate_warning.py", "code/experiments_original/run_section5.py",
            "results/warning/summary.csv", "results/warning/matching_summary.csv",
            "results/covariates/incremental_forward.csv", "results/panel/texas_events.csv",
            "logs/train_3regions.log", "data/texas/station_meta.csv", "data/texas/temperature_meta.csv",
            "results/checks/intermediates.npz", "results/checks/data_table.csv", "code/serra/verify_from_csv.py"]
    assert all(n in files for n in need), [n for n in need if n not in files]
    # the archive must verify itself: unzip into an empty folder and rebuild all tables from the result files
    tmp = tempfile.mkdtemp()
    with zipfile.ZipFile(dst_zip) as z:
        z.extractall(tmp)
    out = run([sys.executable, os.path.join(tmp, OR2_ROOT, "code", "serra", "verify_from_csv.py")], tmp)
    assert out.startswith("OK"), out
    shutil.rmtree(tmp)
    print(f"  Online Resource 2: {len(files)} files, {os.path.getsize(dst_zip) / 1e6:.1f} MB; "
          "unzipped copy rebuilds all tables and numbers from its own result files")


def abstract_words():
    t = open(os.path.join(TEX, MAIN + ".tex")).read()
    ab = t[t.index("\\abstract{") + 10: t.index("\\keywords{")].replace("\\mathversion{normal}", "")
    mac = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", open(os.path.join(TEX, "tables", "numbers.tex")).read()))
    ab = re.sub(r"\\(\w+)\\?(?=[ ,;.)%\n])", lambda m: re.sub(r"\\ensuremath\{(.*)\}", r"\1", str(mac[m.group(1)]))
                if m.group(1) in mac else m.group(0), ab)
    ab = ab.replace("\\emph{defect}", "defect").replace("$\\pm2\\%$", "±2%").replace("\\%", "%").replace("--", "–")
    ab = re.sub(r"(?<=[ (])-(?=\d)", "−", " ".join(ab.strip().rstrip("}").split()))
    return ab


def main():
    out = os.path.join(ROOT, sys.argv[1] if len(sys.argv) > 1 else "SUBMISSION")
    if os.path.exists(out):
        shutil.rmtree(out)
    up = os.path.join(out, "01_Upload_to_Editorial_Manager"); os.makedirs(up)
    ref = os.path.join(out, "02_Reference_only_do_not_upload"); os.makedirs(ref)
    print("building PDFs"); build_master()
    # The journal compiles the manuscript from the LaTeX zip; the cover letter is a separate item.
    # The master PDF of the manuscript is kept only to compare with the PDF the system builds.
    for src, dst, lab in [(COVER, os.path.join(up, "01_Cover_Letter_SERRA.pdf"), "cover letter"),
                          (SUPP, os.path.join(up, "03_Online_Resource_1_Supplementary_Material.pdf"), "Online Resource 1"),
                          (MAIN, os.path.join(ref, "Manuscript_RM_Precipitation_SERRA_master.pdf"), "manuscript")]:
        shutil.copy(os.path.join(TEX, src + ".pdf"), dst); check_pdf(dst, lab)
    flat = os.path.join(out, "_flat"); flat_source(flat)
    names = sorted(os.listdir(flat))
    assert not any(n.lower().startswith("cover") for n in names), "cover letter must not be in the LaTeX zip"
    assert sum(open(os.path.join(flat, n), errors="replace").read().count("\\documentclass")
               for n in names if n.endswith(".tex")) == 1, "LaTeX zip must hold exactly one main document"
    zip_dir(flat, os.path.join(up, "02_Manuscript_LaTeX_source.zip")); shutil.rmtree(flat)
    online_resource_2(os.path.join(up, "04_Online_Resource_2_Code_Protocol_Results.zip"))
    ab = abstract_words(); n = len(ab.split())
    assert 150 <= n <= 250, f"abstract has {n} words"
    open(os.path.join(out, "abstract.txt"), "w").write(ab + "\n")
    print(f"  abstract: {n} words")
    # only the upload guide goes into the package; the co-author email and the revision notes
    # stay in submission_docs/ (local working files, not tracked)
    shutil.copy(os.path.join(ROOT, "submission_docs", "README_SUBMISSION.md"), out)
    print("package written to", out)


if __name__ == "__main__":
    main()
