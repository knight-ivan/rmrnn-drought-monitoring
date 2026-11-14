"""Check that every table and every number quoted in the paper follows from the result files.

Rebuilds all LaTeX tables and the number macros (make_tables.py) into a temporary folder,
using only the files under results/, and compares them byte for byte with the published
ones in tex/tables/. make_tables.py also stops if a descriptive sentence of the paper no
longer matches the results. Needs neither the raw data, nor the per-seed scores, nor the
score panels, so it runs from the archive (Online Resource 2) as it stands.

Usage: python3 code/serra/verify_from_csv.py
"""
import filecmp, os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PUB = os.path.join(ROOT, "tex", "tables")


def main():
    tmp = tempfile.mkdtemp()
    r = subprocess.run([sys.executable, os.path.join(HERE, "make_tables.py")], env=dict(os.environ, TABLES_OUT=tmp),
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.returncode:
        sys.exit("make_tables.py failed:\n" + r.stdout[-3000:])
    new, pub = sorted(os.listdir(tmp)), sorted(f for f in os.listdir(PUB) if f.endswith(".tex"))
    diff = [f for f in pub if f not in new or not filecmp.cmp(os.path.join(tmp, f), os.path.join(PUB, f), shallow=False)]
    extra = [f for f in new if f not in pub]
    nmac = sum(1 for l in open(os.path.join(tmp, "numbers.tex")) if l.startswith("\\newcommand"))
    if diff or extra:
        sys.exit(f"MISMATCH: differing {diff}, unpublished {extra}")
    print(f"OK: {len(pub)} table files and {nmac} numbers quoted in the text are reproduced exactly from results/; "
          "all descriptive statements checked by make_tables.py hold.")


if __name__ == "__main__":
    main()
