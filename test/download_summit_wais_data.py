#!/usr/bin/env python3
"""
download_summit_wais_data.py

Download and stage public datasets for firn inversions at:
  - Summit Station (Greenland)  -> ./summit/
  - WAIS Divide (Antarctica)    -> ./WAIS/

This script is intentionally "boring but robust":
  * Creates a predictable folder structure
  * Downloads DataONE / Arctic Data Center datasets by DOI (resource maps)
  * Optionally scrapes (simple HTML) landing pages for extra files (age-depth, etc.)
  * Writes manifests so your inversion scripts can point at stable local paths

Included (auto-download) sources
--------------------------------
1) FirnCover dataset (Greenland; includes Summit site)
   DOI: doi:10.18739/A25X25D7M
   (borehole strain rates/compaction, firn temperatures, 2 m air temp, surface height)

2) SUMup Working Group datasets 1952–2019 (Greenland + Antarctica; includes WAIS & Summit points/cores)
   DOI: doi:10.18739/A24Q7QR58
   (snow/firn density, accumulation on land ice, snow depth on sea ice)

Optional (best-effort) sources (scrape landing pages)
-----------------------------------------------------
- WAIS Divide WD2014 age-depth / chronology: USAP-DC landing page
  (Some USAP pages require human verification; script will tell you if it can't download)

NOTE
----
* Some repositories (e.g., NSIDC Earthdata, some USAP datasets) may require credentials or
  human verification. For those, this script prints clear instructions and leaves "stubs"
  in the folder so you can drop files in manually without changing your code paths.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import shutil
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    import requests  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "This script requires the 'requests' package.\n"
        "Install it with:  pip install requests\n"
        f"Original import error: {e}"
    )


DATAONE_MEMBER_NODE = "https://arcticdata.io/metacat/d1/mn/v2"

# --- Dataset registry (edit here if you want to add/remove datasets) ---

DATASETS_DATAONE = [
    {
        "id": "FirnCover",
        "pid": "doi:10.18739/A25X25D7M",
        "sites": ["summit"],
        "notes": "FirnCover dataset (Greenland; includes Summit).",
    },
    {
        "id": "SUMup_1952_2019",
        "pid": "doi:10.18739/A24Q7QR58",
        "sites": ["summit", "WAIS"],
        "notes": "SUMup Working Group datasets 1952–2019.",
    },
]

# Best-effort scraper sources (HTML pages where files are linked directly)
SCRAPE_SOURCES = [
    {
        "id": "WAIS_WD2014_age_depth_USAP",
        "site": "WAIS",
        "landing": "https://www.usap-dc.org/view/dataset/601015",
        # Heuristics for picking files from the page:
        "include_regex": r"(wd2014|age|depth|chron|time.?scale)",
        "allowed_ext": [".txt", ".csv", ".tsv", ".zip", ".xlsx"],
        "notes": "USAP-DC landing page for WAIS Divide WD2014 timescale (best-effort).",
    },
    # Add Summit/Greenland age-depth here if you have a preferred landing page.
]


# ---------------------------
# Utilities
# ---------------------------

def now_utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def safe_filename(name: str, max_len: int = 180) -> str:
    # Keep it filesystem-safe and reasonably short.
    name = name.strip().replace("\n", " ")
    name = re.sub(r"[^\w.\-+() ]+", "_", name)
    name = re.sub(r"\s+", "_", name)
    return name[:max_len] if len(name) > max_len else name


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def download_stream(url: str, dest: Path, *, timeout_s: int = 120, force: bool = False) -> None:
    ensure_dir(dest.parent)
    if dest.exists() and not force:
        return
    with requests.get(url, stream=True, timeout=timeout_s) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".partial")
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)
        tmp.replace(dest)


def try_symlink(link_path: Path, target_path: Path) -> bool:
    # Returns True if a symlink was created, False otherwise.
    try:
        if link_path.exists() or link_path.is_symlink():
            if link_path.is_symlink() or link_path.is_file():
                link_path.unlink()
            else:
                shutil.rmtree(link_path)
        link_path.symlink_to(target_path, target_is_directory=target_path.is_dir())
        return True
    except Exception:
        return False


def link_or_copy_tree(dst: Path, src: Path) -> None:
    ensure_dir(dst.parent)
    if try_symlink(dst, src):
        return
    # Fallback: copy
    if dst.exists():
        if dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    shutil.copytree(src, dst)


# ---------------------------
# DataONE / Arctic Data Center downloading
# ---------------------------

DATAONE_NS = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "ore": "http://www.openarchives.org/ore/terms/",
    "d1":  "http://ns.dataone.org/service/types/v2.0",
}


def dataone_url(kind: str, pid: str, base: str = DATAONE_MEMBER_NODE) -> str:
    # kind: "object" or "meta"
    pid_enc = urllib.parse.quote(pid, safe="")
    return f"{base}/{kind}/{pid_enc}"


def parse_sysmeta_filename(sysmeta_xml: bytes) -> Tuple[Optional[str], Optional[str]]:
    """
    Return (fileName, formatId) from DataONE system metadata (if present).
    """
    try:
        root = ET.fromstring(sysmeta_xml)
    except Exception:
        return (None, None)

    # System metadata is in the DataONE schema; nodes are usually un-namespaced in practice.
    # We'll search by localname.
    def find_text(local: str) -> Optional[str]:
        for el in root.iter():
            if el.tag.endswith(local) and (el.text and el.text.strip()):
                return el.text.strip()
        return None

    return (find_text("fileName"), find_text("formatId"))


def parse_resourcemap_aggregates(rdfxml: bytes) -> List[str]:
    """
    Parse an OAI-ORE ResourceMap (RDF/XML) and return aggregated PIDs.
    """
    pids: Set[str] = set()
    try:
        root = ET.fromstring(rdfxml)
    except Exception:
        return []

    # Find elements with ore:aggregates and extract rdf:resource.
    for el in root.iter():
        if el.tag.endswith("aggregates"):
            res = el.attrib.get(f"{{{DATAONE_NS['rdf']}}}resource")
            if res:
                pids.add(res)

    # ResourceMap often includes itself; remove if present.
    return sorted(pids)


def dataone_download_dataset(pid: str, out_dir: Path, *, force: bool = False) -> Dict[str, object]:
    """
    Download a DataONE dataset given its PID (commonly 'doi:...').

    Strategy:
      1) GET object(pid) -> resource map (RDF/XML)
      2) Parse aggregates -> list of PIDs
      3) For each PID, GET meta(pid) to get a fileName (if any), then GET object(pid)

    Returns a manifest dict.
    """
    ensure_dir(out_dir)
    manifest: Dict[str, object] = {
        "downloaded_at_utc": now_utc_iso(),
        "member_node": DATAONE_MEMBER_NODE,
        "resource_map_pid": pid,
        "objects": [],
        "notes": "Objects are downloaded via DataONE object/meta endpoints. Some may be metadata.",
    }

    rm_url = dataone_url("object", pid)
    rm_path = out_dir / "resource_map.rdf.xml"
    download_stream(rm_url, rm_path, force=force)

    aggregates = parse_resourcemap_aggregates(rm_path.read_bytes())
    if not aggregates:
        # Not a ResourceMap? Still keep the object.
        aggregates = []

    # Always include the resource map itself in the manifest
    objects_to_get = [pid] + aggregates

    for obj_pid in objects_to_get:
        try:
            sysmeta_url = dataone_url("meta", obj_pid)
            sysmeta = requests.get(sysmeta_url, timeout=60).content
        except Exception:
            sysmeta = b""

        filename, format_id = parse_sysmeta_filename(sysmeta)
        if filename:
            fname = safe_filename(filename)
        else:
            # Derive a stable-ish filename from PID
            fname = safe_filename(obj_pid.replace(":", "_").replace("/", "_"))
            # Some PIDs are URLs; keep only tail
            if len(fname) > 160:
                fname = fname[-160:]

        obj_path = out_dir / "objects" / fname
        meta_path = out_dir / "systemmetadata" / (fname + ".sysmeta.xml")

        if sysmeta:
            write_text(meta_path, sysmeta.decode("utf-8", errors="replace"))

        try:
            download_stream(dataone_url("object", obj_pid), obj_path, force=force)
            obj_sha = sha256_file(obj_path) if obj_path.exists() else None
        except Exception as e:
            obj_sha = None
            write_text(
                out_dir / "FAILED_DOWNLOADS.txt",
                (out_dir / "FAILED_DOWNLOADS.txt").read_text(encoding="utf-8", errors="ignore")
                + f"\nFAILED: {obj_pid}\n  {e}\n"
                if (out_dir / "FAILED_DOWNLOADS.txt").exists()
                else f"FAILED: {obj_pid}\n  {e}\n",
            )

        manifest["objects"].append(
            {
                "pid": obj_pid,
                "filename": str(obj_path.relative_to(out_dir)),
                "formatId": format_id,
                "sha256": obj_sha,
            }
        )

    # Write manifest
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


# ---------------------------
# Simple landing-page scraper (best-effort)
# ---------------------------

class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag.lower() != "a":
            return
        href = None
        for k, v in attrs:
            if k.lower() == "href":
                href = v
                break
        if href:
            self.links.append(href)


def scrape_file_links(landing_url: str) -> List[str]:
    r = requests.get(landing_url, timeout=60)
    r.raise_for_status()
    parser = _LinkParser()
    parser.feed(r.text)
    # Normalize relative links
    base = landing_url
    out: List[str] = []
    for href in parser.links:
        href = href.strip()
        if not href or href.startswith("#") or href.lower().startswith("javascript:"):
            continue
        out.append(urllib.parse.urljoin(base, href))
    return sorted(set(out))


def best_effort_scrape_download(
    landing_url: str,
    out_dir: Path,
    include_regex: str,
    allowed_ext: Sequence[str],
    *,
    force: bool = False,
) -> Dict[str, object]:
    ensure_dir(out_dir)
    manifest: Dict[str, object] = {
        "downloaded_at_utc": now_utc_iso(),
        "landing_url": landing_url,
        "downloaded": [],
        "skipped": [],
        "failed": [],
        "notes": "Best-effort scrape; may fail for JS-heavy pages or human verification (CAPTCHA).",
    }

    try:
        links = scrape_file_links(landing_url)
    except Exception as e:
        write_text(out_dir / "SCRAPE_FAILED.txt", f"Failed to scrape landing page:\n{landing_url}\n\n{e}\n")
        return manifest

    include = re.compile(include_regex, flags=re.IGNORECASE)
    allowed_ext = tuple(e.lower() for e in allowed_ext)

    candidates: List[str] = []
    for url in links:
        url_l = url.lower()
        if not url_l.endswith(allowed_ext):
            continue
        if include.search(url):
            candidates.append(url)

    if not candidates:
        write_text(
            out_dir / "NO_CANDIDATES.txt",
            "Scrape succeeded but no downloadable file links matched the filters.\n"
            f"landing_url={landing_url}\ninclude_regex={include_regex}\nallowed_ext={list(allowed_ext)}\n"
            "\nTip: edit SCRAPE_SOURCES in the script to broaden the regex or add extensions.\n",
        )
        manifest["skipped"] = links[:200]
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    for url in candidates:
        fname = safe_filename(Path(urllib.parse.urlparse(url).path).name or "download")
        dest = out_dir / fname
        try:
            download_stream(url, dest, force=force)
            manifest["downloaded"].append({"url": url, "path": str(dest.relative_to(out_dir)), "sha256": sha256_file(dest)})
        except Exception as e:
            manifest["failed"].append({"url": url, "error": str(e)})

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


# ---------------------------
# Main staging logic
# ---------------------------

def stage_dataset_links(site_dir: Path, shared_dataset_dir: Path, dataset_id: str) -> None:
    """
    Put a link (symlink or copied tree) in site_dir/sources/<dataset_id> -> shared dataset.
    """
    dst = site_dir / "sources" / dataset_id
    ensure_dir(dst.parent)
    link_or_copy_tree(dst, shared_dataset_dir)


def write_site_readme(site_dir: Path, site_name: str) -> None:
    readme = site_dir / "README.md"
    if readme.exists():
        return
    write_text(
        readme,
        f"# {site_name} staged datasets\n\n"
        "This folder is created by `download_summit_wais_data.py`.\n\n"
        "## Structure\n"
        "- `sources/`  symlinks (or copies) to downloaded source datasets\n"
        "- `manual/`   drop-in folder for any files you had to fetch manually\n"
        "- `processed/` (optional) your own processed/cleaned tables for inversion\n\n"
        "If a download requires credentials/CAPTCHA, you can place files in `manual/` and keep\n"
        "your inversion scripts stable by pointing at this folder.\n",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Download and stage Summit + WAIS datasets for firn inversions.")
    p.add_argument("--base-dir", default=".", help="Directory where WAIS/ and summit/ folders will be created.")
    p.add_argument("--force", action="store_true", help="Re-download even if files exist.")
    p.add_argument("--no-scrape", action="store_true", help="Skip best-effort scraping sources.")
    p.add_argument("--only-site", choices=["WAIS", "summit"], default=None, help="Only stage one site.")
    args = p.parse_args(list(argv) if argv is not None else None)

    base = Path(args.base_dir).resolve()
    summit_dir = base / "summit"
    wais_dir = base / "WAIS"
    shared = base / "_shared_sources"
    ensure_dir(shared)

    if args.only_site != "WAIS":
        ensure_dir(summit_dir / "sources")
        ensure_dir(summit_dir / "manual")
        ensure_dir(summit_dir / "processed")
        write_site_readme(summit_dir, "Summit Station (Greenland)")

    if args.only_site != "summit":
        ensure_dir(wais_dir / "sources")
        ensure_dir(wais_dir / "manual")
        ensure_dir(wais_dir / "processed")
        write_site_readme(wais_dir, "WAIS Divide (Antarctica)")

    run_manifest: Dict[str, object] = {"ran_at_utc": now_utc_iso(), "base_dir": str(base), "datasets": []}

    # 1) DataONE datasets (Arctic Data Center)
    for ds in DATASETS_DATAONE:
        ds_id = ds["id"]
        pid = ds["pid"]
        ds_dir = shared / ds_id
        ensure_dir(ds_dir)

        print(f"\n== Download DataONE dataset: {ds_id} ({pid})")
        man = dataone_download_dataset(pid, ds_dir, force=args.force)

        # Stage links into each site directory requested
        for site in ds["sites"]:
            if args.only_site and site != args.only_site:
                continue
            if site == "summit":
                stage_dataset_links(summit_dir, ds_dir, ds_id)
            elif site == "WAIS":
                stage_dataset_links(wais_dir, ds_dir, ds_id)

        run_manifest["datasets"].append({"id": ds_id, "pid": pid, "notes": ds.get("notes", ""), "stored_in": str(ds_dir)})

    # 2) Best-effort scraped sources
    if not args.no_scrape:
        for src in SCRAPE_SOURCES:
            site = src["site"]
            if args.only_site and site != args.only_site:
                continue

            out_dir = shared / src["id"]
            print(f"\n== Scrape+download: {src['id']} ({site})")
            man = best_effort_scrape_download(
                src["landing"],
                out_dir,
                src["include_regex"],
                src["allowed_ext"],
                force=args.force,
            )

            # Stage it
            if site == "summit":
                stage_dataset_links(summit_dir, out_dir, src["id"])
            else:
                stage_dataset_links(wais_dir, out_dir, src["id"])

            run_manifest["datasets"].append(
                {"id": src["id"], "landing": src["landing"], "notes": src.get("notes", ""), "stored_in": str(out_dir)}
            )

    # Write overall manifest
    (base / "download_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

    print("\nDone.")
    print(f"  Staged Summit dir: {summit_dir}")
    print(f"  Staged WAIS dir:   {wais_dir}")
    print(f"  Shared sources:    {shared}")
    print(f"  Run manifest:      {base / 'download_manifest.json'}")
    print("\nNext steps (recommended):")
    print("  1) Inspect *_shared_sources/*/manifest.json to see what was pulled.")
    print("  2) If any scrape failed due to CAPTCHA/credentials, place files in <site>/manual/.")
    print("  3) Convert/standardize into <site>/processed/ tables that match your inversion obs format.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
