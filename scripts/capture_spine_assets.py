#!/usr/bin/env python3
"""
Capture Spine assets directly from a live website session (no HAR needed).

It launches a browser via Playwright, listens to network responses, saves
matching files, then optionally bundles related assets together.
"""

import argparse
import base64
import hashlib
import re
import shutil
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple
from urllib.parse import unquote, unquote_to_bytes, urljoin, urlsplit


ASSET_EXTENSIONS = (
    ".json",
    ".atlas",
    ".png",
    ".apng",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
    ".svg",
    ".bmp",
    ".ico",
    ".avif",
    ".tif",
    ".tiff",
    ".skel",
    ".skel.bytes",
    ".mp3",
    ".ogg",
    ".wav",
    ".m4a",
    ".aac",
    ".flac",
    ".opus",
    ".weba",
)
IMAGE_EXTENSIONS = (
    ".png",
    ".apng",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
    ".svg",
    ".bmp",
    ".ico",
    ".avif",
    ".tif",
    ".tiff",
)
LogFn = Callable[[str], None]
DATA_URL_RE = re.compile(r"^data:(?P<mime>[^;,]+)?(?P<b64>;base64)?,(?P<data>.*)$", re.IGNORECASE | re.DOTALL)
MIME_EXT = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/svg+xml": ".svg",
    "image/bmp": ".bmp",
    "image/x-icon": ".ico",
    "image/vnd.microsoft.icon": ".ico",
    "image/avif": ".avif",
    "audio/mpeg": ".mp3",
    "audio/ogg": ".ogg",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/aac": ".aac",
    "audio/flac": ".flac",
    "audio/opus": ".opus",
    "audio/webm": ".weba",
}


def safe_name(name: str) -> str:
    return re.sub(r"[<>:\"/\\|?*\x00-\x1f]", "_", name).strip() or "unnamed"


def has_asset_extension(path: str) -> bool:
    lower = path.lower()
    return any(lower.endswith(ext) for ext in ASSET_EXTENSIONS)


def has_extension_in_url(url: str, extensions: Sequence[str]) -> bool:
    lower = (url or "").lower()
    if not lower:
        return False
    for ext in extensions:
        if re.search(re.escape(ext) + r"($|[?#&])", lower):
            return True
    return False


def is_image_url(url: str) -> bool:
    if (url or "").startswith("data:image/"):
        return True
    return has_extension_in_url(url, IMAGE_EXTENSIONS)


def emit(logger: Optional[LogFn], message: str) -> None:
    if logger is None:
        print(message)
    else:
        logger(message)


def extract_pages_from_atlas(atlas_path: Path) -> List[str]:
    pages: List[str] = []
    try:
        lines = atlas_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return pages

    for line in lines:
        s = line.strip()
        if not s:
            continue
        if ":" in s:
            continue
        if s.lower().endswith(".png"):
            pages.append(s)
    return pages


def copy_if_exists(src: Path, dst_dir: Path) -> bool:
    if not src.exists():
        return False
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst_dir / src.name)
    return True


def choose_png_for_page(candidates: Sequence[Path], atlas_parent: Path) -> Optional[Path]:
    if not candidates:
        return None
    for path in candidates:
        if path.parent == atlas_parent:
            return path
    return candidates[0]


def bundle_assets(raw_dir: Path, bundle_dir: Path, logger: Optional[LogFn] = None) -> None:
    json_files = sorted(raw_dir.rglob("*.json"))
    if not json_files:
        emit(logger, "[bundle] no .json files found in raw capture")
        return

    atlas_files = list(raw_dir.rglob("*.atlas"))
    png_files = list(raw_dir.rglob("*.png"))
    png_by_name: Dict[str, List[Path]] = defaultdict(list)
    for png in png_files:
        png_by_name[png.name].append(png)

    for json_file in json_files:
        stem = json_file.stem
        rel = json_file.relative_to(raw_dir)
        out = bundle_dir / rel.parent / stem
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(json_file, out / json_file.name)

        chosen_atlas: Optional[Path] = None
        exact = json_file.with_suffix(".atlas")
        if exact.exists():
            chosen_atlas = exact
        else:
            for cand in atlas_files:
                if cand.stem == stem:
                    chosen_atlas = cand
                    break

        if not chosen_atlas:
            emit(logger, f"[bundle] {stem}: no atlas matched")
            continue

        shutil.copy2(chosen_atlas, out / chosen_atlas.name)

        page_names = extract_pages_from_atlas(chosen_atlas)
        copied = 0
        for page in page_names:
            src = choose_png_for_page(png_by_name.get(page, []), chosen_atlas.parent)
            if src and copy_if_exists(src, out):
                copied += 1

        if copied == 0:
            for p in png_files:
                if p.stem == stem or p.stem.startswith(f"{stem}_"):
                    if copy_if_exists(p, out):
                        copied += 1

        emit(logger, f"[bundle] {stem}: atlas={chosen_atlas.name}, png_copied={copied}")


def split_url_path(path: str, path_anchor: str) -> Tuple[List[str], str]:
    path = unquote(path or "")
    parts = [safe_name(x) for x in path.split("/") if x and x not in (".", "..")]
    if path_anchor:
        lowered = [p.lower() for p in parts]
        anchor = path_anchor.lower()
        if anchor in lowered:
            parts = parts[lowered.index(anchor) :]
    if not parts:
        return [], ""
    return parts[:-1], parts[-1]


def build_path_from_url(raw_root: Path, url: str, keep_host: bool, path_anchor: str) -> Path:
    parts = urlsplit(url)
    host = safe_name(parts.netloc or "unknown_host")
    folders, filename = split_url_path(parts.path, path_anchor=path_anchor)
    if not filename:
        digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
        filename = f"asset_{digest}.bin"

    out_dir = raw_root / host if keep_host else raw_root
    for folder in folders:
        out_dir = out_dir / folder
    return out_dir / filename


def write_asset_file(out_file: Path, body: bytes) -> Tuple[bool, str]:
    """
    Returns: (written, status)
      status in {"written", "duplicate", "conflict_kept_existing"}
    """
    if out_file.exists():
        try:
            old = out_file.read_bytes()
        except Exception:
            old = b""
        if old == body:
            return False, "duplicate"
        return False, "conflict_kept_existing"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_bytes(body)
    return True, "written"


def decode_data_url(data_url: str) -> Optional[Tuple[str, bytes]]:
    match = DATA_URL_RE.match(data_url)
    if not match:
        return None
    mime = (match.group("mime") or "application/octet-stream").lower()
    payload = match.group("data") or ""
    if match.group("b64"):
        try:
            body = base64.b64decode(payload, validate=False)
        except Exception:
            return None
    else:
        body = unquote_to_bytes(payload)
    return mime, body


def ext_from_mime(mime: str) -> str:
    if mime in MIME_EXT:
        return MIME_EXT[mime]
    if "/" in mime:
        subtype = mime.split("/", 1)[1].split("+", 1)[0].strip().lower()
        if subtype:
            return "." + safe_name(subtype)
    return ".bin"


def save_data_url(raw_root: Path, data_url: str) -> Optional[Path]:
    decoded = decode_data_url(data_url)
    if not decoded:
        return None
    mime, body = decoded
    if not body:
        return None
    ext = ext_from_mime(mime)
    digest = hashlib.sha1(body).hexdigest()[:16]
    target_dir = raw_root / "_data_urls" / safe_name(mime.replace("/", "_"))
    target_dir.mkdir(parents=True, exist_ok=True)
    out = target_dir / f"data_{digest}{ext}"
    if not out.exists():
        out.write_bytes(body)
    return out


def normalize_url(raw_url: str, base_url: str) -> Optional[str]:
    if not raw_url:
        return None
    u = raw_url.strip()
    if not u:
        return None
    if u.startswith("data:"):
        return u
    if u.startswith("//"):
        scheme = urlsplit(base_url).scheme or "https"
        return f"{scheme}:{u}"
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", u):
        return u
    return urljoin(base_url, u)


def discover_page_urls(page, base_url: str) -> List[str]:
    raw_urls = page.evaluate(
        """
        () => {
          const out = new Set();
          const add = (v) => {
            if (!v || typeof v !== 'string') return;
            const t = v.trim();
            if (t) out.add(t);
          };
          const addSrcset = (v) => {
            if (!v || typeof v !== 'string') return;
            for (const part of v.split(',')) {
              const token = part.trim().split(/\\s+/)[0];
              if (token) add(token);
            }
          };
          const addCssUrls = (text) => {
            if (!text || typeof text !== 'string') return;
            const re = /url\\(([^)]+)\\)/g;
            let m;
            while ((m = re.exec(text)) !== null) {
              const candidate = String(m[1] || '').trim().replace(/^["']|["']$/g, '');
              add(candidate);
            }
          };

          for (const el of document.querySelectorAll('[src]')) add(el.getAttribute('src'));
          for (const el of document.querySelectorAll('[href]')) add(el.getAttribute('href'));
          for (const el of document.querySelectorAll('[poster]')) add(el.getAttribute('poster'));
          for (const el of document.querySelectorAll('[srcset]')) addSrcset(el.getAttribute('srcset'));
          for (const el of document.querySelectorAll('[style]')) addCssUrls(el.getAttribute('style'));
          for (const styleEl of document.querySelectorAll('style')) addCssUrls(styleEl.textContent || '');

          if (window.performance && performance.getEntriesByType) {
            for (const e of performance.getEntriesByType('resource')) add(e.name);
          }

          const html = document.documentElement ? document.documentElement.outerHTML : '';
          if (html) {
            const reHttp = /(https?:\\/\\/[^\\s"'<>]+)/g;
            let m;
            while ((m = reHttp.exec(html)) !== null) add(m[1]);
            const reData = /(data:(?:image|audio)\\/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=]+)/g;
            while ((m = reData.exec(html)) !== null) add(m[1]);
          }
          return Array.from(out);
        }
        """
    )
    if not isinstance(raw_urls, list):
        return []
    out: Set[str] = set()
    for item in raw_urls:
        if not isinstance(item, str):
            continue
        normalized = normalize_url(item, base_url)
        if normalized:
            out.add(normalized)
    return sorted(out)


def fetch_discovered_assets(
    context,
    page,
    raw_root: Path,
    seen: Set[str],
    keep_host: bool,
    path_anchor: str,
    image_urls: Set[str],
    logger: Optional[LogFn] = None,
) -> int:
    discovered = discover_page_urls(page, page.url)
    if not discovered:
        emit(logger, "[discover] no candidate URLs found in page source")
        return 0

    emit(logger, f"[discover] candidate URLs: {len(discovered)}")
    saved = 0
    for asset_url in discovered:
        if asset_url in seen:
            continue
        if asset_url.startswith("blob:"):
            continue

        if is_image_url(asset_url) and not asset_url.startswith("data:"):
            image_urls.add(asset_url)

        if asset_url.startswith("data:"):
            path = save_data_url(raw_root, asset_url)
            if path:
                seen.add(asset_url)
                saved += 1
                emit(logger, f"[saved:data] {path}")
            continue

        path = urlsplit(asset_url).path or ""
        if not has_asset_extension(path) and not has_extension_in_url(asset_url, ASSET_EXTENSIONS):
            continue

        try:
            resp = context.request.get(asset_url, timeout=20000)
        except Exception:
            continue
        if resp.status >= 400:
            continue
        content_type = (resp.headers.get("content-type") or "").lower()
        if content_type.startswith("image/"):
            image_urls.add(asset_url)
        try:
            body = resp.body()
        except Exception:
            continue
        if not body:
            continue

        out_file = build_path_from_url(raw_root, asset_url, keep_host=keep_host, path_anchor=path_anchor)
        written, status = write_asset_file(out_file, body)
        seen.add(asset_url)
        if written:
            saved += 1
            emit(logger, f"[saved:direct] {out_file}")
        elif status == "conflict_kept_existing":
            emit(logger, f"[skip:conflict] {out_file} (kept first file)")

    emit(logger, f"[discover] saved from discovered URLs: {saved}")
    return saved


def refetch_images_via_tabs(
    context,
    raw_root: Path,
    image_urls: Set[str],
    seen: Set[str],
    keep_host: bool,
    path_anchor: str,
    logger: Optional[LogFn] = None,
) -> int:
    urls = sorted(image_urls)
    if not urls:
        emit(logger, "[image-refetch] no image URLs collected")
        return 0

    emit(logger, f"[image-refetch] trying {len(urls)} image URLs via tab navigation")
    saved = 0
    tab = context.new_page()
    try:
        for image_url in urls:
            if image_url in seen:
                continue
            try:
                nav_resp = tab.goto(image_url, wait_until="domcontentloaded", timeout=20000)
            except Exception:
                nav_resp = None

            if nav_resp is None or nav_resp.status >= 400:
                try:
                    req_resp = context.request.get(image_url, timeout=20000)
                except Exception:
                    continue
                if req_resp.status >= 400:
                    continue
                try:
                    body = req_resp.body()
                except Exception:
                    continue
            else:
                try:
                    body = nav_resp.body()
                except Exception:
                    continue

            if not body:
                continue

            out_file = build_path_from_url(
                raw_root,
                image_url,
                keep_host=keep_host,
                path_anchor=path_anchor,
            )
            written, status = write_asset_file(out_file, body)
            seen.add(image_url)
            if written:
                saved += 1
                emit(logger, f"[saved:image-tab] {out_file}")
            elif status == "conflict_kept_existing":
                emit(logger, f"[skip:conflict] {out_file} (kept first file)")
    finally:
        tab.close()

    emit(logger, f"[image-refetch] saved via tabs: {saved}")
    return saved


def run_capture(
    url: str,
    out_dir: str,
    seconds: int,
    idle_seconds: int,
    headless: bool,
    browser_name: str,
    no_bundle: bool,
    keep_host: bool,
    path_anchor: str,
    force_image_tab_refetch: bool,
    logger: Optional[LogFn] = None,
) -> int:
    out_root = Path(out_dir).resolve()
    raw_root = out_root / "raw"
    bundles_root = out_root / "bundles"
    raw_root.mkdir(parents=True, exist_ok=True)

    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        emit(logger, "Playwright is required:")
        emit(logger, "  python3 -m pip install playwright")
        emit(logger, "  python3 -m playwright install chromium")
        return 2

    seen: Set[str] = set()
    image_urls: Set[str] = set()
    saved = 0
    max_seconds = max(1, int(seconds))
    idle_limit = max(1, int(idle_seconds))

    with sync_playwright() as p:
        browser_launcher = getattr(p, browser_name)
        browser = browser_launcher.launch(headless=headless)
        context = browser.new_context()
        page = context.new_page()
        last_new_asset_at = {"t": time.monotonic()}

        def on_response(resp) -> None:
            nonlocal saved
            asset_url = resp.url
            if asset_url in seen:
                return

            try:
                if resp.status >= 400:
                    return
                content_type = (resp.headers.get("content-type") or "").lower()
            except Exception:
                return

            if asset_url.startswith("data:"):
                path = save_data_url(raw_root, asset_url)
                if path:
                    seen.add(asset_url)
                    saved += 1
                    last_new_asset_at["t"] = time.monotonic()
                    emit(logger, f"[saved:data] {path}")
                return

            path = urlsplit(asset_url).path or ""
            by_ext = has_asset_extension(path) or has_extension_in_url(asset_url, ASSET_EXTENSIONS)
            by_type = content_type.startswith("image/") or content_type.startswith("audio/")
            if not by_ext and not by_type:
                return

            if content_type.startswith("image/") or is_image_url(asset_url):
                image_urls.add(asset_url)

            try:
                body = resp.body()
            except Exception:
                return

            if not body:
                return

            out_file = build_path_from_url(
                raw_root,
                asset_url,
                keep_host=keep_host,
                path_anchor=path_anchor,
            )
            written, status = write_asset_file(out_file, body)
            seen.add(asset_url)
            if written:
                saved += 1
                last_new_asset_at["t"] = time.monotonic()
                emit(logger, f"[saved] {out_file}")
            elif status == "conflict_kept_existing":
                emit(logger, f"[skip:conflict] {out_file} (kept first file)")

        context.on("response", on_response)
        page.goto(url, wait_until="domcontentloaded")
        emit(logger, f"[info] Opened {url}")
        emit(
            logger,
            f"[info] Capture max={max_seconds}s, auto-stop after {idle_limit}s idle. Interact/login if needed.",
        )
        start_at = time.monotonic()
        while True:
            now = time.monotonic()
            elapsed = now - start_at
            idle_for = now - last_new_asset_at["t"]
            if elapsed >= max_seconds:
                emit(logger, f"[info] stop reason: reached max seconds ({max_seconds}s)")
                break
            if elapsed >= 3 and idle_for >= idle_limit:
                emit(logger, f"[info] stop reason: idle for {int(idle_for)}s")
                break
            page.wait_for_timeout(400)
        saved += fetch_discovered_assets(
            context=context,
            page=page,
            raw_root=raw_root,
            seen=seen,
            keep_host=keep_host,
            path_anchor=path_anchor,
            image_urls=image_urls,
            logger=logger,
        )
        if force_image_tab_refetch:
            saved += refetch_images_via_tabs(
                context=context,
                raw_root=raw_root,
                image_urls=image_urls,
                seen=seen,
                keep_host=keep_host,
                path_anchor=path_anchor,
                logger=logger,
            )
        browser.close()

    emit(logger, f"[done] saved files: {saved}")

    if not no_bundle:
        bundle_assets(raw_root, bundles_root, logger=logger)
        emit(logger, f"[done] bundles: {bundles_root}")

    emit(logger, f"[done] raw: {raw_root}")
    return 0


def launch_gui(
    default_out: str,
    default_seconds: int,
    default_idle_seconds: int,
    default_browser: str,
    default_anchor: str,
) -> int:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, scrolledtext, ttk
    except Exception:
        print("Tkinter is unavailable in this Python build. Run CLI mode instead.")
        return 3

    root = tk.Tk()
    root.title("Spine Asset Capture")
    root.geometry("900x620")

    url_var = tk.StringVar()
    out_var = tk.StringVar(value=default_out)
    seconds_var = tk.IntVar(value=default_seconds)
    idle_seconds_var = tk.IntVar(value=default_idle_seconds)
    browser_var = tk.StringVar(value=default_browser)
    anchor_var = tk.StringVar(value=default_anchor)
    headless_var = tk.BooleanVar(value=False)
    no_bundle_var = tk.BooleanVar(value=False)
    keep_host_var = tk.BooleanVar(value=False)
    image_refetch_var = tk.BooleanVar(value=True)
    is_running = {"value": False}

    frame = ttk.Frame(root, padding=10)
    frame.pack(fill="both", expand=True)

    ttk.Label(frame, text="Page URL").grid(row=0, column=0, sticky="w")
    url_entry = ttk.Entry(frame, textvariable=url_var)
    url_entry.grid(row=0, column=1, columnspan=3, sticky="ew", padx=(8, 0))
    url_entry.focus_set()

    ttk.Label(frame, text="Output Dir").grid(row=1, column=0, sticky="w", pady=(8, 0))
    out_entry = ttk.Entry(frame, textvariable=out_var)
    out_entry.grid(row=1, column=1, columnspan=2, sticky="ew", padx=(8, 0), pady=(8, 0))

    def browse_dir() -> None:
        selected = filedialog.askdirectory()
        if selected:
            out_var.set(selected)

    ttk.Button(frame, text="Browse", command=browse_dir).grid(
        row=1,
        column=3,
        sticky="e",
        padx=(8, 0),
        pady=(8, 0),
    )

    ttk.Label(frame, text="Seconds").grid(row=2, column=0, sticky="w", pady=(8, 0))
    ttk.Entry(frame, textvariable=seconds_var, width=8).grid(row=2, column=1, sticky="w", padx=(8, 0), pady=(8, 0))

    ttk.Label(frame, text="Idle Sec").grid(row=2, column=2, sticky="w", padx=(12, 0), pady=(8, 0))
    ttk.Entry(frame, textvariable=idle_seconds_var, width=8).grid(row=2, column=3, sticky="w", padx=(8, 0), pady=(8, 0))

    ttk.Label(frame, text="Browser").grid(row=3, column=0, sticky="w", pady=(8, 0))
    ttk.Combobox(
        frame,
        textvariable=browser_var,
        values=["chromium", "firefox", "webkit"],
        state="readonly",
        width=12,
    ).grid(row=3, column=1, sticky="w", padx=(8, 0), pady=(8, 0))

    ttk.Label(frame, text="Path Anchor").grid(row=3, column=2, sticky="w", pady=(8, 0))
    ttk.Entry(frame, textvariable=anchor_var, width=16).grid(row=3, column=3, sticky="w", padx=(8, 0), pady=(8, 0))

    checks = ttk.Frame(frame)
    checks.grid(row=4, column=0, columnspan=4, sticky="w", pady=(8, 0))
    ttk.Checkbutton(checks, text="Headless", variable=headless_var).pack(side="left")
    ttk.Checkbutton(checks, text="Skip Bundle", variable=no_bundle_var).pack(side="left", padx=(12, 0))
    ttk.Checkbutton(checks, text="Keep Host Folder", variable=keep_host_var).pack(side="left", padx=(12, 0))
    ttk.Checkbutton(checks, text="Image Tab Refetch", variable=image_refetch_var).pack(side="left", padx=(12, 0))

    button_bar = ttk.Frame(frame)
    button_bar.grid(row=5, column=0, columnspan=4, sticky="w", pady=(10, 0))
    start_button = ttk.Button(button_bar, text="Start Capture")
    start_button.pack(side="left")

    log_box = scrolledtext.ScrolledText(frame, wrap="word", height=26, state="disabled")
    log_box.grid(row=6, column=0, columnspan=4, sticky="nsew", pady=(12, 0))

    frame.columnconfigure(1, weight=1)
    frame.columnconfigure(2, weight=1)
    frame.rowconfigure(6, weight=1)

    def append_log(text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        log_box.configure(state="normal")
        log_box.insert("end", f"[{stamp}] {text}\n")
        log_box.see("end")
        log_box.configure(state="disabled")

    def finish(code: int) -> None:
        is_running["value"] = False
        start_button.configure(state="normal")
        append_log(f"[exit] capture finished with code {code}")

    def start_capture() -> None:
        if is_running["value"]:
            return

        url = url_var.get().strip()
        if not url:
            messagebox.showerror("Missing URL", "Please paste the page URL.")
            return

        out_dir = out_var.get().strip()
        if not out_dir:
            messagebox.showerror("Missing Output", "Please select an output directory.")
            return

        try:
            seconds = int(seconds_var.get())
        except Exception:
            messagebox.showerror("Invalid Seconds", "Seconds must be an integer.")
            return

        if seconds <= 0:
            messagebox.showerror("Invalid Seconds", "Seconds must be > 0.")
            return

        try:
            idle_seconds = int(idle_seconds_var.get())
        except Exception:
            messagebox.showerror("Invalid Idle Seconds", "Idle seconds must be an integer.")
            return

        if idle_seconds <= 0:
            messagebox.showerror("Invalid Idle Seconds", "Idle seconds must be > 0.")
            return

        is_running["value"] = True
        start_button.configure(state="disabled")
        append_log(f"[start] url={url}")

        def worker() -> None:
            code = run_capture(
                url=url,
                out_dir=out_dir,
                seconds=seconds,
                idle_seconds=idle_seconds,
                headless=bool(headless_var.get()),
                browser_name=browser_var.get(),
                no_bundle=bool(no_bundle_var.get()),
                keep_host=bool(keep_host_var.get()),
                path_anchor=anchor_var.get().strip(),
                force_image_tab_refetch=bool(image_refetch_var.get()),
                logger=lambda m: root.after(0, append_log, m),
            )
            root.after(0, finish, code)

        threading.Thread(target=worker, daemon=True).start()

    start_button.configure(command=start_capture)
    root.mainloop()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture Spine assets directly from a website.")
    parser.add_argument("--url", help="Page URL to open.")
    parser.add_argument("--out", default="captured_assets", help="Output directory.")
    parser.add_argument("--seconds", type=int, default=60, help="Maximum capture duration after page load.")
    parser.add_argument("--idle-seconds", type=int, default=6, help="Auto-stop if no new assets for N seconds.")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode.")
    parser.add_argument("--browser", choices=("chromium", "firefox", "webkit"), default="chromium")
    parser.add_argument("--no-bundle", action="store_true", help="Skip bundle step.")
    parser.add_argument("--keep-host", action="store_true", help="Keep hostname as top-level folder in raw output.")
    parser.add_argument(
        "--path-anchor",
        default="assets",
        help="Trim URL path to start from this segment (default: assets). Empty keeps full URL path.",
    )
    parser.add_argument("--gui", action="store_true", help="Launch a simple GUI.")
    parser.add_argument(
        "--no-image-tab-refetch",
        action="store_true",
        help="Disable extra image capture pass that re-opens image URLs in tabs.",
    )
    args = parser.parse_args()

    if args.gui or not args.url:
        return launch_gui(
            default_out=args.out,
            default_seconds=args.seconds,
            default_idle_seconds=args.idle_seconds,
            default_browser=args.browser,
            default_anchor=args.path_anchor,
        )

    return run_capture(
        url=args.url,
        out_dir=args.out,
        seconds=args.seconds,
        idle_seconds=args.idle_seconds,
        headless=args.headless,
        browser_name=args.browser,
        no_bundle=args.no_bundle,
        keep_host=args.keep_host,
        path_anchor=args.path_anchor,
        force_image_tab_refetch=not args.no_image_tab_refetch,
    )


if __name__ == "__main__":
    raise SystemExit(main())
