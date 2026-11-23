"""
UploadThing upload service — uploads files directly via signed URLs.

Generates file keys locally (Sqids + djb2), signs the upload URL with HMAC-SHA256,
and PUTs directly to the regional ingest endpoint. No server-side API call needed.

Requires UPLOADTHING_TOKEN env var (base64-encoded JSON with apiKey, appId, regions).
"""

import base64
import hashlib
import hmac
import json
import math
import os
import time
import urllib.error
import urllib.request
import uuid

from sqids import Sqids
from sqids.constants import DEFAULT_ALPHABET


def _djb2(s: str) -> int:
    """DJB2 hash matching UploadThing's JS implementation."""
    h = 5381
    for ch in reversed(s):
        h = (h * 33) ^ ord(ch)
        h &= 0xFFFFFFFF
    h = (h & 0xBFFFFFFF) | ((h >> 1) & 0x40000000)
    if h >= 0x80000000:
        h -= 0x100000000
    return h


def _shuffle(input_str: str, seed: str) -> str:
    """Shuffle alphabet deterministically using seed (matches UploadThing JS)."""
    chars = list(input_str)
    seed_num = _djb2(seed)
    for i in range(len(chars)):
        j = int(math.fmod(math.fmod(seed_num, i + 1) + i, len(chars)))
        chars[i], chars[j] = chars[j], chars[i]
    return "".join(chars)


def _generate_file_key(app_id: str) -> str:
    """Generate a unique UploadThing file key using Sqids (matches JS SDK)."""
    alphabet = _shuffle(DEFAULT_ALPHABET, app_id)
    sqids = Sqids(alphabet=alphabet, min_length=12)
    encoded_app_id = sqids.encode([abs(_djb2(app_id))])
    file_seed = base64.urlsafe_b64encode(uuid.uuid4().bytes).decode().rstrip("=")
    return encoded_app_id + file_seed


def _pct_encode(s: str) -> str:
    """Percent-encode like JS encodeURIComponent."""
    return urllib.request.quote(s, safe="")


def _generate_signed_url(
    base_url: str, api_key: str, app_id: str,
    file_name: str, file_size: int,
    file_type: str = "application/octet-stream",
) -> str:
    """Generate HMAC-signed upload URL."""
    expires = int(time.time() * 1000) + 3600 * 1000  # 1 hour

    params = (
        f"?expires={expires}"
        f"&x-ut-identifier={_pct_encode(app_id)}"
        f"&x-ut-file-name={_pct_encode(file_name)}"
        f"&x-ut-file-size={file_size}"
        f"&x-ut-file-type={_pct_encode(file_type)}"
        f"&x-ut-content-disposition=inline"
        f"&x-ut-acl=public-read"
    )
    url = base_url + params
    sig = hmac.new(api_key.encode(), url.encode(), hashlib.sha256).hexdigest()
    return f"{url}&signature=hmac-sha256={sig}"


def _decode_token() -> tuple[str, str, str]:
    """Decode UPLOADTHING_TOKEN to get (api_key, app_id, region)."""
    token = os.environ.get("UPLOADTHING_TOKEN")
    if not token:
        raise RuntimeError("UPLOADTHING_TOKEN env var not set")
    decoded = json.loads(base64.b64decode(token))
    return decoded["apiKey"], decoded["appId"], decoded.get("regions", ["sea1"])[0]


def _build_multipart(file_name: str, file_data: bytes, content_type: str) -> tuple[bytes, str]:
    """Build multipart/form-data body manually (no requests dependency)."""
    boundary = uuid.uuid4().hex
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{file_name}"\r\n'
        f"Content-Type: {content_type}\r\n"
        f"\r\n"
    ).encode() + file_data + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def upload_file(file_path: str, custom_id: str | None = None) -> str | None:
    """
    Upload a single file to UploadThing.

    Returns the ufsUrl of the uploaded file, or None on failure.
    """
    if not os.path.isfile(file_path):
        print(f"  [upload] File not found: {file_path}", flush=True)
        return None

    api_key, app_id, region = _decode_token()
    file_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)

    # Generate file key and signed URL
    file_key = _generate_file_key(app_id)
    ingest_base = f"https://{region}.ingest.uploadthing.com/{file_key}"
    signed_url = _generate_signed_url(
        ingest_base, api_key, app_id, file_name, file_size,
    )

    print(f"  [upload] Uploading {file_name} ({file_size / 1e6:.1f} MB)...", flush=True)

    with open(file_path, "rb") as f:
        file_data = f.read()

    # PUT with multipart form-data (matching the working Python/C# implementations)
    body, content_type = _build_multipart(file_name, file_data, "application/octet-stream")

    put_req = urllib.request.Request(
        signed_url,
        data=body,
        headers={
            "Content-Type": content_type,
            "Range": "bytes=0-",
        },
        method="PUT",
    )

    try:
        with urllib.request.urlopen(put_req, timeout=300) as resp:
            resp_body = resp.read().decode()
            resp_json = json.loads(resp_body)
            ufs_url = resp_json.get("ufsUrl") or resp_json.get("url", "")
            print(f"  [upload] Success: {ufs_url}", flush=True)
            return ufs_url
    except urllib.error.HTTPError as e:
        body_text = e.read().decode() if e.fp else ""
        print(f"  [upload] PUT failed: {e.code} {body_text[:500]}", flush=True)
        return None
    except Exception as e:
        print(f"  [upload] PUT error: {e}", flush=True)
        return None


def upload_directory(dir_path: str, prefix: str = "") -> list[str]:
    """Upload all files in a directory. Returns list of uploaded URLs."""
    urls = []
    if not os.path.isdir(dir_path):
        print(f"  [upload] Directory not found: {dir_path}", flush=True)
        return urls

    for fname in sorted(os.listdir(dir_path)):
        fpath = os.path.join(dir_path, fname)
        if os.path.isfile(fpath):
            url = upload_file(fpath, custom_id=f"{prefix}/{fname}" if prefix else fname)
            if url:
                urls.append(url)

    return urls
