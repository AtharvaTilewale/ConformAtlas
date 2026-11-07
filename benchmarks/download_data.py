"""Resumable, checksum-verified download of the public benchmark datasets (set CA_BENCH_DATA to choose the folder).

Needs network access to zenodo.org, api.figshare.com and the figshare storage host it redirects to.

Order: small reference files first, then AdK, HIV-1 protease MD + MSM, then T4L replicates.
Re-running skips verified files and resumes partial ones.
"""

import hashlib
import json
import os
import subprocess
import time
import urllib.request

ROOT = os.environ.get(
    "CA_BENCH_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
)
os.makedirs(ROOT, exist_ok=True)
LOG = open(os.path.join(ROOT, "download.log"), "a", buffering=1)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, file=LOG)
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def getj(u):
    return json.loads(
        urllib.request.urlopen(
            urllib.request.Request(u, headers={"Accept": "application/json"}), timeout=60
        ).read()
    )


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


manifest = {}


def fetch(url, dest, size=None, md5sum=None, label=""):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if (
        os.path.exists(dest)
        and size
        and os.path.getsize(dest) == size
        and (not md5sum or md5(dest) == md5sum)
    ):
        log("OK (already verified)", dest)
        manifest[dest] = {"url": url, "bytes": size, "md5": md5sum}
        return True
    for attempt in range(1, 6):
        subprocess.run(
            ["curl", "-sS", "-L", "-C", "-", "--retry", "3", "--retry-delay", "5", "-o", dest, url],
            check=False,
        )
        if size is None or (os.path.exists(dest) and os.path.getsize(dest) == size):
            got = md5(dest) if md5sum else None
            if md5sum is None or got == md5sum:
                log(
                    "DONE",
                    dest,
                    os.path.getsize(dest),
                    "bytes",
                    "md5 ok" if md5sum else "(no md5 provided)",
                )
                manifest[dest] = {"url": url, "bytes": os.path.getsize(dest), "md5": got or md5sum}
                return True
            log("MD5 MISMATCH, restarting", dest, got, md5sum)
            os.remove(dest)
        else:
            log(
                "incomplete",
                dest,
                os.path.getsize(dest) if os.path.exists(dest) else 0,
                "of",
                size,
                "attempt",
                attempt,
            )
    log("FAILED", dest)
    return False


def zenodo(rid, subdir, only=None, skip=None):
    r = getj(f"https://zenodo.org/api/records/{rid}")
    for f in sorted(r["files"], key=lambda f: f["size"]):
        k = f["key"]
        if only and not only(k):
            continue
        if skip and skip(k):
            log("SKIP (not needed)", rid, k, f["size"])
            continue
        fetch(
            f"https://zenodo.org/api/records/{rid}/files/{k}/content",
            os.path.join(ROOT, subdir, k),
            f["size"],
            f["checksum"].split(":")[1],
            f"{rid}/{k}",
        )
    json.dump(
        {
            "zenodo_record": rid,
            "doi": r["doi"],
            "title": r["metadata"]["title"],
            "license": r["metadata"].get("license", {}).get("id")
            or r["metadata"].get("access_right"),
        },
        open(os.path.join(ROOT, subdir, "RECORD.json"), "w"),
        indent=1,
    )


# 1. small references and experimental data
zenodo(29617, "adk/reference_PMF_apo")
zenodo(29243, "adk/AKecoLib_structures")
zenodo(20834076, "adk/CPMG_relaxation_dispersion")
# 2. ADK equilibrium trajectory (figshare 5108170)
art = getj("https://api.figshare.com/v2/articles/5108170")
for f in art["files"]:
    fetch(
        f["download_url"],
        os.path.join(ROOT, "adk/equilibrium_1us", f["name"]),
        f["size"],
        f.get("computed_md5") or f.get("supplied_md5"),
        f["name"],
    )
json.dump(
    {
        "figshare_article": 5108170,
        "doi": art["doi"],
        "title": art["title"],
        "license": art["license"]["name"],
    },
    open(os.path.join(ROOT, "adk/equilibrium_1us/RECORD.json"), "w"),
    indent=1,
)
# 2b. crystal structures used by the angle check (RCSB)
for pid in ("4AKE", "1AKE"):
    fetch(
        f"https://files.rcsb.org/download/{pid}.pdb",
        os.path.join(ROOT, "adk/crystal", f"{pid}.pdb"),
        label=pid,
    )
# 3. HIV-1 protease: MD tars + MSM; the Brownian-dynamics tars are not needed for this benchmark
zenodo(5006613, "hiv1_protease", skip=lambda k: k.startswith("data_BD_"))
# 4. T4 lysozyme replicates (two force fields); the 3x5 us set (153 GB) is not needed
zenodo(3989057, "t4l/ff99SBstar-ILDN_TIP4P2005_5x1us")
zenodo(3991429, "t4l/ff15ipq_SPCEb_3x1us")
json.dump(manifest, open(os.path.join(ROOT, "MANIFEST.json"), "w"), indent=1)
log("ALL DONE", len(manifest), "files")
