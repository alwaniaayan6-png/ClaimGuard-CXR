"""Modal-side PadChest-GR bulk image downloader.

Downloads the 47 GB of PadChest-GR images + prior-studies directly from the
BIMCV EUDAT B2DROP Nextcloud share (``https://b2drop.bsc.es/nextcloud/s/PadChest-GR``)
to the ``claimguard-v5-data`` Modal volume's ``/padchest_gr_raw/`` path.

The small text files (``grounded_reports_20240819.json`` and
``master_table.csv``) are already on the volume, uploaded from the local
host. This entrypoint handles only the heavy image payload so the local
machine never has to stage 47 GB.

Protocol:

1. Use WebDAV (Nextcloud public-share dav endpoint ``/public.php/webdav/``)
   with basic auth user = share token (``PadChest-GR``), empty password.
2. List the top-level directories (``Padchest_GR_files``,
   ``PadChest_GR_progression_prior_studies``) and mirror each to the volume.
3. Skip files that already exist with matching size (idempotent, resumable).
4. Stream each file in 16 MiB chunks to keep memory flat.

Usage::

    modal deploy v5/modal/padchest_gr_download.py
    python -c "import modal; f=modal.Function.from_name('claimguard-v6-padchest-gr','download_images'); print(f.spawn().object_id)"

Or blocking::

    modal run v5/modal/padchest_gr_download.py::download_images
"""

from __future__ import annotations

import modal

app = modal.App("claimguard-v6-padchest-gr")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("curl")
    .pip_install("requests==2.32.3", "tqdm==4.66.5")
)

volume = modal.Volume.from_name("claimguard-v5-data", create_if_missing=True)


_BASE_WEBDAV = "https://b2drop.bsc.es/nextcloud/public.php/webdav"
_SHARE_TOKEN = "PadChest-GR"
_CHUNK = 16 * 1024 * 1024  # 16 MiB


def _list_webdav(session, url: str) -> list[tuple[str, str, int, bool]]:
    """PROPFIND one level under ``url``; return (href, name, size, is_dir)."""
    resp = session.request("PROPFIND", url, headers={"Depth": "1"}, timeout=60)
    resp.raise_for_status()
    import xml.etree.ElementTree as ET
    ns = {"d": "DAV:"}
    tree = ET.fromstring(resp.content)
    out: list[tuple[str, str, int, bool]] = []
    for r in tree.findall("d:response", ns):
        href_el = r.find("d:href", ns)
        if href_el is None or not href_el.text:
            continue
        href = href_el.text
        is_dir = r.find(".//d:resourcetype/d:collection", ns) is not None
        size_el = r.find(".//d:getcontentlength", ns)
        size = int(size_el.text) if size_el is not None and size_el.text else 0
        name = href.rstrip("/").split("/")[-1]
        out.append((href, name, size, is_dir))
    return out


def _stream_file(session, href: str, out_path, expected_size: int) -> int:
    """Stream one WebDAV file to disk in 16 MiB chunks. Returns bytes written."""
    from pathlib import Path
    out_path = Path(out_path)
    if out_path.exists() and out_path.stat().st_size == expected_size:
        return 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://b2drop.bsc.es{href}"
    with session.get(url, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        bytes_written = 0
        with open(out_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=_CHUNK):
                if not chunk:
                    continue
                fh.write(chunk)
                bytes_written += len(chunk)
    return bytes_written


def _mirror_dir(session, root_href: str, local_root, counters: dict) -> None:
    """Recursively mirror one WebDAV directory to a local Modal-volume path."""
    from pathlib import Path
    import urllib.parse
    local_root = Path(local_root)
    url = f"https://b2drop.bsc.es{root_href}"
    entries = _list_webdav(session, url)
    # drop the self entry (first element is the dir itself)
    entries = [e for e in entries if e[0].rstrip("/") != root_href.rstrip("/")]
    for href, name, size, is_dir in entries:
        decoded = urllib.parse.unquote(name)
        child_local = local_root / decoded
        if is_dir:
            child_local.mkdir(parents=True, exist_ok=True)
            _mirror_dir(session, href, child_local, counters)
        else:
            try:
                nb = _stream_file(session, href, child_local, expected_size=size)
            except Exception as exc:
                counters["errors"] += 1
                print(f"[download] ERROR on {href}: {exc}")
                continue
            counters["files"] += 1
            counters["bytes"] += nb
            if counters["files"] % 50 == 0:
                print(f"[download] files={counters['files']} "
                      f"GB={counters['bytes'] / (1024**3):.2f} "
                      f"errors={counters['errors']}")


@app.function(
    image=image,
    cpu=4,
    memory=4096,
    timeout=60 * 60 * 10,
    volumes={"/data": volume},
)
def download_images(
    out_dir: str = "/data/padchest_gr_raw",
    include_prior_studies: bool = True,
) -> dict:
    """Mirror all Padchest_GR_files and (optional) progression-prior-studies."""
    import requests
    from pathlib import Path as P

    out = P(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.auth = (_SHARE_TOKEN, "")

    targets = [("Padchest_GR_files", True)]
    if include_prior_studies:
        targets.append(("PadChest_GR_progression_prior_studies", True))

    summary: dict[str, dict] = {}
    for dir_name, _ in targets:
        root_href = f"/nextcloud/public.php/webdav/{dir_name}/"
        local_root = out / dir_name
        local_root.mkdir(parents=True, exist_ok=True)
        counters = {"files": 0, "bytes": 0, "errors": 0}
        print(f"[download] mirroring {dir_name} -> {local_root}")
        try:
            _mirror_dir(session, root_href, local_root, counters)
        except Exception as exc:
            print(f"[download] top-level failure on {dir_name}: {exc}")
            counters["errors"] += 1
        summary[dir_name] = {
            "files": counters["files"],
            "gb": round(counters["bytes"] / (1024**3), 3),
            "errors": counters["errors"],
        }
        print(f"[download] {dir_name} complete: {summary[dir_name]}")
        volume.commit()

    return {"status": "ok", "summary": summary, "out_dir": out_dir}


@app.local_entrypoint()
def main(include_prior: bool = True) -> None:
    result = download_images.remote(include_prior_studies=include_prior)
    print(result)
