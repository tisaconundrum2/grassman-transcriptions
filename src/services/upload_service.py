"""
UploadThing upload service — uploads files via REST API.

Uses the /v7/prepareUpload endpoint to get presigned URLs,
then PUTs files directly to the storage provider.

Requires UPLOADTHING_TOKEN env var (base64-encoded JSON with apiKey, appId, regions).
"""

import base64
import json
import os
import urllib.request
import urllib.error


def _decode_token() -> tuple[str, str]:
    """Decode UPLOADTHING_TOKEN to get (api_key, region)."""
    token = os.environ.get("UPLOADTHING_TOKEN")
    if not token:
        raise RuntimeError("UPLOADTHING_TOKEN env var not set")
    decoded = json.loads(base64.b64decode(token))
    api_key = decoded["apiKey"]
    regions = decoded.get("regions", ["sea1"])
    return api_key, regions[0]


def upload_file(file_path: str, custom_id: str | None = None) -> str | None:
    """
    Upload a single file to UploadThing.

    Args:
        file_path: Local path to the file to upload.
        custom_id: Optional custom identifier for the file.

    Returns:
        The URL of the uploaded file, or None on failure.
    """
    if not os.path.isfile(file_path):
        print(f"  [upload] File not found: {file_path}", flush=True)
        return None

    api_key, region = _decode_token()
    file_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)

    # Step 1: Request presigned URL via /v7/prepareUpload
    prepare_payload = json.dumps({
        "files": [{
            "name": file_name,
            "size": file_size,
            "type": "application/octet-stream",
            **({"customId": custom_id} if custom_id else {}),
        }],
        "acl": "public-read",
        "contentDisposition": "attachment",
    }).encode()

    ingest_url = f"https://{region}.ingest.uploadthing.com"
    req = urllib.request.Request(
        f"{ingest_url}/v7/prepareUpload",
        data=prepare_payload,
        headers={
            "Content-Type": "application/json",
            "x-uploadthing-api-key": api_key,
            "x-uploadthing-version": "7.0.0",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        print(f"  [upload] prepareUpload failed: {e.code} {body}", flush=True)
        return None
    except Exception as e:
        print(f"  [upload] prepareUpload error: {e}", flush=True)
        return None

    if not result or "presignedUrls" not in result:
        # Try alternative response format
        if isinstance(result, list) and len(result) > 0:
            presigned = result[0]
        elif isinstance(result, dict) and "data" in result:
            data = result["data"]
            presigned = data[0] if isinstance(data, list) else data
        else:
            print(f"  [upload] Unexpected prepareUpload response: {json.dumps(result)[:500]}", flush=True)
            return None
    else:
        presigned = result["presignedUrls"][0] if isinstance(result["presignedUrls"], list) else result

    upload_url = presigned.get("url") or presigned.get("presignedUrl")
    file_url = presigned.get("fileUrl") or presigned.get("ufsUrl")

    if not upload_url:
        print(f"  [upload] No presigned URL in response: {json.dumps(presigned)[:500]}", flush=True)
        return None

    # Step 2: PUT the file to the presigned URL
    print(f"  [upload] Uploading {file_name} ({file_size / 1e6:.1f} MB)...", flush=True)

    with open(file_path, "rb") as f:
        file_data = f.read()

    put_req = urllib.request.Request(
        upload_url,
        data=file_data,
        headers={
            "Content-Type": "application/octet-stream",
        },
        method="PUT",
    )

    try:
        with urllib.request.urlopen(put_req, timeout=300) as resp:
            # Some responses include the file URL
            resp_body = resp.read().decode()
            if resp_body:
                try:
                    resp_json = json.loads(resp_body)
                    file_url = file_url or resp_json.get("url") or resp_json.get("fileUrl")
                except json.JSONDecodeError:
                    pass
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        print(f"  [upload] PUT failed: {e.code} {body}", flush=True)
        return None
    except Exception as e:
        print(f"  [upload] PUT error: {e}", flush=True)
        return None

    print(f"  [upload] Success: {file_url or 'uploaded (URL not returned)'}", flush=True)
    return file_url


def upload_directory(dir_path: str, prefix: str = "") -> list[str]:
    """
    Upload all files in a directory to UploadThing.

    Args:
        dir_path: Local directory path.
        prefix: Optional prefix for custom IDs.

    Returns:
        List of uploaded file URLs.
    """
    urls = []
    if not os.path.isdir(dir_path):
        print(f"  [upload] Directory not found: {dir_path}", flush=True)
        return urls

    for fname in sorted(os.listdir(dir_path)):
        fpath = os.path.join(dir_path, fname)
        if os.path.isfile(fpath):
            custom_id = f"{prefix}/{fname}" if prefix else fname
            url = upload_file(fpath, custom_id=custom_id)
            if url:
                urls.append(url)

    return urls
