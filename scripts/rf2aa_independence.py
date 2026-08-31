"""Independence judge (Exp A): re-fold the held-out design set with RF2AA (RoseTTAFold-All-Atom),
a different-architecture, ligand-capable structure predictor, to test whether the Protenix ordering
STE-plateau < RFd3 ~ real is an AlphaFold3-family artifact. RF2AA is not AF3-lineage.

Design set per ligand: best composition-passing STE design, best composition-passing RFd3 design,
the real crystal anchor, and a scrambled control. Metric: protein<->ligand interface PAE from RF2AA's
err_dict (lower = more confident interface), plus ligand pLDDT.

Modes:
  build  -> assemble scripts/data/rf2aa_designset.json from panel_run + rfd3_panel
  fold   -> run RF2AA on each design (needs the RFAA micromamba env + a free GPU)
  parse  -> read RF2AA err_dicts -> scripts/data/rf2aa_scored.json
"""
import argparse
import contextlib
import glob
import json
import os
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "scripts/data"
RF2AA_DIR = Path(os.environ.get("RF2AA_DIR", "RoseTTAFold-All-Atom"))  # set to your RF2AA checkout
MM = os.environ.get("MICROMAMBA", str(Path.home() / ".local/bin/micromamba"))
POLY_X = 0.35

def maxaa(s): return max(Counter(s).values()) / len(s) if s else 1.0
def ip(r): return r.get("protenix_iptm")

def build():
    panel = {p["ligand"]: p["ligand_name"] for p in json.loads((DATA / "ligand_panel.json").read_text())}
    rows = []
    for f in sorted(glob.glob(str(DATA / "panel_run/*.json"))):
        d = json.loads(Path(f).read_text())
        smi = next((r["ligand"] for r in d if r["method"] == "anchor_real"), None)
        name = panel.get(smi, Path(f).stem.replace("lig_", ""))
        # best composition-passing STE
        ste = [r for r in d if r["method"] == "ste" and ip(r) is not None and maxaa(r["seq"]) < POLY_X]
        if ste:
            b = max(ste, key=ip)
            rows.append({"ligand_name": name, "smiles": smi, "seq": b["seq"], "method": "ste_plateau",
                         "protenix_iptm": ip(b)})
        # real + scramble anchors
        for r in d:
            if r["method"] in ("anchor_real", "anchor_scramble"):
                rows.append({"ligand_name": name, "smiles": r["ligand"], "seq": r["seq"],
                             "method": r["method"], "protenix_iptm": ip(r)})
    # best composition-passing RFd3 per ligand
    for f in sorted(glob.glob(str(DATA / "rfd3_panel/*.json"))):
        if f.endswith("_designs.json"): continue
        d = json.loads(Path(f).read_text())
        if not d: continue
        smi = d[0]["ligand"]; name = panel.get(smi, Path(f).stem)
        rf = [r for r in d if ip(r) is not None and maxaa(r["seq"]) < POLY_X]
        if rf:
            b = max(rf, key=ip)
            rows.append({"ligand_name": name, "smiles": smi, "seq": b["seq"], "method": "rfd3",
                         "protenix_iptm": ip(b)})
    for i, r in enumerate(rows):
        r["id"] = f"d{i:03d}_{r['method']}_{''.join(c for c in r['ligand_name'] if c.isalnum())[:10]}"
    out = DATA / "rf2aa_designset.json"
    out.write_text(json.dumps(rows, indent=2))
    by = Counter(r["method"] for r in rows)
    print(f"built {len(rows)} designs -> {out}  ({dict(by)})")

def _sdf_cache(smiles, key, sdfdir):
    """SMILES -> a 3D SDF file (hydra-safe path; raw SMILES breaks hydra's override grammar)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    path = sdfdir / f"{key}.sdf"
    if path.exists():
        return path
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        raise ValueError(f"bad SMILES: {smiles}")
    m = Chem.AddHs(m)
    if AllChem.EmbedMolecule(m, randomSeed=42) != 0:
        AllChem.EmbedMolecule(m, randomSeed=42, useRandomCoords=True)
    with contextlib.suppress(Exception):
        AllChem.MMFFOptimizeMolecule(m)
    w = Chem.SDWriter(str(path)); w.write(m); w.close()
    return path

def fold(gpu, limit=None):
    designs = json.loads((DATA / "rf2aa_designset.json").read_text())
    outdir = Path("/tmp/rf2aa_out"); outdir.mkdir(exist_ok=True)
    fastadir = Path("/tmp/rf2aa_fasta"); fastadir.mkdir(exist_ok=True)
    sdfdir = Path("/tmp/rf2aa_sdf"); sdfdir.mkdir(exist_ok=True)
    # empty template DB so FFindexDB init succeeds (single-seq, no templates, no DB download)
    tdb = RF2AA_DIR / "pdb100_2021Mar03"; tdb.mkdir(exist_ok=True)
    (tdb / "pdb100_2021Mar03_pdb.ffindex").write_text("")     # 0 entries -> no templates
    (tdb / "pdb100_2021Mar03_pdb.ffdata").write_bytes(b"\0")   # >=1 byte so mmap works
    done = 0; new = 0
    for r in designs:
        tag = r["id"]
        if (outdir / f"{tag}_aux.pt").exists() or (outdir / f"{tag}.pdb").exists():
            done += 1; continue
        if limit is not None and new >= limit:
            break
        fa = fastadir / f"{tag}.fasta"; fa.write_text(f">{tag}\n{r['seq']}\n")
        # single-sequence bypass: precompute make_msa outputs so RF2AA skips the DB search
        msadir = outdir / tag / "A"; msadir.mkdir(parents=True, exist_ok=True)
        (msadir / "t000_.msa0.a3m").write_text(f">query\n{r['seq']}\n")
        (msadir / "t000_.hhr").write_text("")
        (msadir / "t000_.atab").write_text("")
        ligkey = "".join(c for c in r["ligand_name"] if c.isalnum())[:16]
        try:
            sdf = _sdf_cache(r["smiles"], ligkey, sdfdir)
        except Exception as e:
            print(f"  SDF_FAIL {tag}: {e}", flush=True); new += 1; continue
        cmd = ["python", "-m", "rf2aa.run_inference", "--config-name", "protein_sm",
               f"job_name={tag}", f"output_path={outdir}",
               f"protein_inputs.A.fasta_file={fa}",
               f"sm_inputs.B.input={sdf}", "sm_inputs.B.input_type=sdf"]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
        print(f"[rf2aa] folding {tag} ({r['method']}/{r['ligand_name']})", flush=True)
        p = subprocess.run([str(MM), "run", "-n", "RFAA"] + cmd, cwd=str(RF2AA_DIR),
                           env=env, capture_output=True, text=True)
        new += 1
        if p.returncode != 0:
            print(f"  FAIL {tag}: {p.stderr[-400:]}", flush=True)
        else:
            done += 1
    print(f"[rf2aa] folded {done}/{len(designs)} ({new} attempted this run)")

def parse():
    import torch
    designs = json.loads((DATA / "rf2aa_designset.json").read_text())
    outdir = Path("/tmp/rf2aa_out")
    for r in designs:
        tag = r["id"]; L = len(r["seq"])
        aux = sorted(outdir.glob(f"{tag}*_aux.pt"))
        if not aux:
            r["rf2aa_iface_pae"] = None; continue
        e = torch.load(aux[0], map_location="cpu")
        r["rf2aa_iface_pae"] = float(e["pae_inter"])   # RF2AA inter-chain (protein<->ligand) PAE
        r["rf2aa_mean_plddt"] = float(e["mean_plddt"])
        pl = e["plddts"].squeeze()
        r["rf2aa_lig_plddt"] = float(pl[L:].mean()) if pl.shape[0] > L else None
    (DATA / "rf2aa_scored.json").write_text(json.dumps(designs, indent=2))
    # ordering summary: does RF2AA (non-AF3) reproduce STE-plateau worse than RFd3 ~ real?
    agg = defaultdict(list)
    for r in designs:
        if r.get("rf2aa_iface_pae") is not None: agg[r["method"]].append(r["rf2aa_iface_pae"])
    print("RF2AA inter-chain PAE (lower = more confident interface):")
    for m in ("anchor_real", "rfd3", "ste_plateau", "anchor_scramble"):
        v = agg.get(m, [])
        if v: print(f"  {m:16} n={len(v)} mean={sum(v)/len(v):.2f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["build", "fold", "parse"])
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--limit", type=int, default=None, help="fold only this many new designs (smoke test)")
    a = ap.parse_args()
    {"build": build, "fold": lambda: fold(a.gpu, a.limit), "parse": parse}[a.mode]()
