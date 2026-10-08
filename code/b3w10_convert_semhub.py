"""d3-zen-semantic-hub (Zenodo 10.5281/zenodo.8110724; Kocsis et al., Nat Commun 2023, doi:10.1038/s41467-023-42088-7) -> BIDS-iEEG DERIVATIVE.
Source per patient (P1, P2) and period (pre/post disconnection): *_data_raw.mat (authors' 'raw': samples x 128 channels),
*_data_DBT_SVD.mat (demodulated-band-transform denoised + first SVD component removed), *_events.mat (event code + sample),
Information_sheet.xlsx (channel -> contact label, MNI x/y/z, Desikan-Killiany and Destrieux labels, per period; event codes).
Article: ECoG sampled at 2000 Hz (TDT, 0.7-800 Hz), then downsampled to 1000 Hz and DBT-denoised. The 'raw' and DBT_SVD
matrices have the same number of samples, so both are at 1000 Hz: the 'raw' files are resampled -> derivative.
Written: desc-downsampled (= the 'data_raw' files) and desc-dbtsvd, BrainVision float32, units n/a (not stated).
Usage: python b3w10_convert_semhub.py <zenodo dir> <bids root>"""
import json, os, re, shutil, sys
import numpy as np, h5py, scipy.io as sio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b3w10_common import write_run, wtsv, wjson, sha256_file, mat_strings, flag_strings
from b3w2_xlsx import read_xlsx

SRC, OUT = sys.argv[1], sys.argv[2]
FS = 1000.0
X = read_xlsx(os.path.join(SRC, "Information_sheet.xlsx"))
report = {"runs": [], "channel_tables": {}, "privacy_flags": [], "notes": []}


def tables(rows):
    """Split a participant sheet into consecutive location tables; returns list of lists of rows (channel, label, x, y, z, dk, destrieux)."""
    out, cur = [], None
    for r in rows:
        if not r or r[0] is None:
            continue
        if isinstance(r[0], str) and "locations" in r[0].lower():
            cur = []; out.append(cur); continue
        if isinstance(r[0], str) and r[0].strip().lower() == "channel":
            continue
        if cur is not None and len(r) >= 2 and r[1] is not None:
            cur.append((r + [None] * 7)[:7])
    return out


EV = {}
for r in X["event codes"]:
    if r and isinstance(r[0], (int, float)):
        EV[int(r[0])] = " / ".join(str(v).strip() for v in r[1:] if v not in (None, ""))
report["event_codes"] = EV
priv = set()
for p, sheet in ((1, "Participant1_channels"), (2, "Participant2_channels")):
    T = tables(X[sheet])
    assert len(T) == 2, (sheet, len(T))
    report["notes"].append(f"P{p}: sheet has {len(T)} location tables, both headed 'Pre-disconnection locations'; the second is taken as post-disconnection (frontal grid coordinates differ; article: grid displacement after disconnection was corrected)")
    for per, tab in zip(("pre", "post"), T):
        nums = [int(t[0]) for t in tab]
        if nums != list(range(nums[0], nums[0] + len(nums))):
            report["notes"].append(f"P{p} {per}: channel numbers in the sheet are not consecutive ({[n for i, n in enumerate(nums) if n != nums[0] + i][:5]}...); mapped by row order starting at {nums[0]}")
        start = nums[0]
        chmap = {start + i: t for i, t in enumerate(tab)}
        report["channel_tables"][f"P{p}_{per}"] = {k: v[1] for k, v in chmap.items()}
        sub = f"sub-{p:02d}"; ses = f"ses-{per}"
        D = os.path.join(OUT, sub, ses, "ieeg")
        # data
        rawp = os.path.join(SRC, f"P{p}_{per}disconnection_data_raw.mat")
        dbtp = os.path.join(SRC, f"P{p}_{per}disconnection_data_DBT_SVD.mat")
        evp = os.path.join(SRC, f"P{p}_{per}disconnection_events.mat")
        for mp in (evp,):
            priv |= mat_strings(mp)
        def load(path):
            try:
                mm = sio.loadmat(path)
                ks = [k for k in mm if not k.startswith("__")]
                assert len(ks) == 1, (path, ks)
                report.setdefault("variables", {})[os.path.basename(path)] = ks[0]
                return np.asarray(mm[ks[0]])
            except NotImplementedError:
                with h5py.File(path, "r") as f:
                    ks = [k for k in f.keys() if not k.startswith("#")]
                    assert len(ks) == 1, (path, ks)
                    report.setdefault("variables", {})[os.path.basename(path)] = ks[0] + " (v7.3)"
                    return np.asarray(f[ks[0]]).T  # MATLAB column-major -> original orientation
        A = load(rawp)
        B = load(dbtp) if os.path.exists(dbtp) else None
        if A.shape[0] < A.shape[1]:
            A = A.T
        if B is not None and B.shape[0] < B.shape[1]:
            B = B.T
        n, nch = A.shape
        assert B is None or B.shape == A.shape, (A.shape, B.shape)
        names, types, extra = [], [], []
        for c in range(1, nch + 1):
            t = chmap.get(c)
            if t is None:
                names.append(f"ch{c:03d}"); types.append("MISC")
                extra.append({"description": f"channel {c} of the release matrix; not listed in Information_sheet.xlsx", "group": "n/a"})
            else:
                lab = re.sub(r"\s+", "", str(t[1]).title()).replace("Heschlsgyrusdepth", "HGdepth")
                names.append(lab)
                types.append("SEEG" if "depth" in str(t[1]).lower() else "ECOG")
                extra.append({"description": f"{str(t[1]).strip()} (release channel {c})", "group": re.sub(r"\d+$", "", lab) or "n/a"})
        assert len(set(names)) == len(names), [x for x in names if names.count(x) > 1]
        # events
        e = sio.loadmat(evp, squeeze_me=True, struct_as_record=False)
        v = e[[x for x in e if not x.startswith("__")][0]]
        for _ in range(5):
            if "time" in v._fieldnames:
                break
            v = getattr(v, v._fieldnames[0])
        cf = [nm for nm in v._fieldnames if nm != "time"][0]
        codes = np.atleast_1d(getattr(v, cf)).astype(int).ravel(); times = np.atleast_1d(v.time).astype(np.int64).ravel()
        assert len(codes) == len(times) and len(codes) > 0
        rows = [[round((s - 1) / FS, 6), 0, EV.get(int(c), "n/a"), int(c), int(s - 1)] for c, s in zip(codes, times)]
        sidecar = {"TaskName": "speechprediction",
                   "TaskDescription": "Speech prediction task: a biasing sentence followed by a target word whose voice onset time (VOT) varies along a /b/-/p/ continuum; filler trials (article; event codes in Information_sheet.xlsx).",
                   "PowerLineFrequency": 60,
                   "SoftwareFilters": {"Downsampling": {"Description": "2000 Hz -> 1000 Hz (article)"}},
                   "HardwareFilters": {"Bandpass": {"Description": "0.7-800 Hz, 12 dB/octave roll-off (TDT; article)"}},
                   "iEEGReference": "Subgaleal reference electrode (article: two subgaleal electrodes were used as ground and reference)",
                   "iEEGGround": "Subgaleal electrode (article)",
                   "Manufacturer": "Tucker-Davis Technologies",
                   "ElectricalStimulation": False,
                   "iEEGPlacementScheme": "Left hemisphere: frontal grid (P1 96, P2 64 contacts per article), temporal pole, subtemporal and lateral temporal strips, 8-contact Heschl's gyrus depth electrode (article); recorded minutes " + ("before" if per == "pre" else "after") + " left anterior temporal lobe disconnection under acute electrocorticography"}
        for desc, M, comment in (("downsampled", A, "release *_data_raw.mat (authors' raw, 1000 Hz after downsampling)"),
                                 ("dbtsvd", B, "release *_data_DBT_SVD.mat (DBT denoised, first SVD component removed)")):
            if M is None:
                continue
            stem = f"{sub}_{ses}_task-speechprediction_desc-{desc}"
            sc = dict(sidecar)
            if desc == "dbtsvd":
                sc["SoftwareFilters"] = {"Downsampling": {"Description": "2000 Hz -> 1000 Hz (article)"},
                                         "DBT": {"Description": "Demodulated band transform denoising, bandwidth 0.25 (article)"},
                                         "SVD": {"Description": "First principal component removed (covariance of the 300 Hz high-passed data; article)"}}
            r = write_run(D, stem, M, FS, names, ["n/a"] * nch, types, sc, ch_extra=extra, comment=f"b3w10_convert_semhub.py: Zenodo 8110724 {comment}, float64 -> float32")
            wtsv(os.path.join(D, stem + "_events.tsv"), ["onset", "duration", "trial_type", "value", "sample"], rows)
            r.update(sub=sub, ses=ses, desc=desc, stem=stem, n_events=len(rows), unlisted=[x for x in names if x.startswith("ch")])
            report["runs"].append(r)
            print(stem, json.dumps({kk: r[kk] for kk in ("n_samples", "n_channels", "duration_s", "roundtrip_ok", "n_events", "float32_max_abs_err")}), flush=True)
        # electrodes (MNI as given; template not stated)
        er = []
        for c in range(1, nch + 1):
            t = chmap.get(c)
            if t is None:
                continue
            nm = names[c - 1]
            xyz = ["n/a" if (v in (None, "NA") or isinstance(v, str)) else round(float(v), 2) for v in t[2:5]]
            er.append([nm, *xyz, "n/a", extra[c - 1]["group"], str(t[5]).strip() if t[5] not in (None, "NA") else "n/a", str(t[6]).strip() if t[6] not in (None, "NA") else "n/a"])
        wtsv(os.path.join(D, f"{sub}_{ses}_space-Other_electrodes.tsv"), ["name", "x", "y", "z", "size", "group", "desikan_killiany", "destrieux"], er)
        wjson(os.path.join(D, f"{sub}_{ses}_space-Other_coordsystem.json"), {
            "iEEGCoordinateSystem": "Other", "iEEGCoordinateUnits": "mm",
            "iEEGCoordinateSystemDescription": "MNI coordinates (mm) as listed per contact and per period (pre/post disconnection) in the release Information_sheet.xlsx; the MNI template is not stated, so the system is given as Other. Temporal pole and subtemporal strip contacts have no coordinates in the release (NA -> n/a)."})
        del A, B
report["privacy_flags"] = flag_strings(priv)
SD = os.path.join(OUT, "sourcedata", "zenodo-8110724"); os.makedirs(SD, exist_ok=True)
for fn in sorted(os.listdir(SRC)):
    if fn.startswith("_") or fn.endswith(".part"):
        continue
    shutil.copy(os.path.join(SRC, fn), os.path.join(SD, fn))
report["sourcedata"] = sorted(os.listdir(SD))
os.makedirs(os.path.join(OUT, "code"), exist_ok=True)
for fn in (__file__, "b3w10_common.py", "b3w3_common.py", "b3w2_xlsx.py"):
    shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.basename(fn)), os.path.join(OUT, "code", os.path.basename(fn)))
json.dump(report, open(os.path.join(OUT, "code", "conversion_report.json"), "w"), indent=1, default=str)
R = report["runs"]
print(json.dumps({"runs": len(R), "hours": sum(r["duration_s"] for r in R) / 3600, "rt_all": all(r["roundtrip_ok"] for r in R), "notes": report["notes"], "privacy_flags": report["privacy_flags"]}, indent=1))
print("CONVERT_DONE")
