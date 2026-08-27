"""Aliyun OSS upload/signing, aligned with ../english-test/server/lib/ossUpload.mjs.

Objects live under {OSS_PREFIX}/tennis-ai/{job_id}/...
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv() -> None:
    path = ROOT / ".env"
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key, val = key.strip(), val.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = val


_load_dotenv()


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def configured() -> bool:
    return bool(
        _env("OSS_ACCESS_KEY_ID")
        and _env("OSS_ACCESS_KEY_SECRET")
        and _env("OSS_BUCKET")
    )


def require_configured() -> None:
    if not configured():
        raise RuntimeError("服务暂时不可用，请稍后重试")


def _prefix() -> str:
    return _env("OSS_PREFIX", "wenbo").strip("/")


def object_key(job_id: str, rel: str) -> str:
    rel = rel.replace("\\", "/").lstrip("/")
    return f"{_prefix()}/tennis-ai/{job_id}/{rel}"


def _content_type(key: str) -> str:
    lower = key.lower()
    if lower.endswith(".mp4"):
        return "video/mp4"
    if lower.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    if lower.endswith(".png"):
        return "image/png"
    if lower.endswith(".webp"):
        return "image/webp"
    if lower.endswith(".json"):
        return "application/json"
    return "application/octet-stream"


def _bucket():
    require_configured()
    import oss2

    region_raw = _env("OSS_REGION", "oss-cn-shanghai")
    region = region_raw if region_raw.startswith("oss-") else f"oss-{region_raw}"
    endpoint = _env("OSS_ENDPOINT") or f"{region}.aliyuncs.com"
    endpoint = endpoint.replace("https://", "").replace("http://", "")
    auth = oss2.Auth(_env("OSS_ACCESS_KEY_ID"), _env("OSS_ACCESS_KEY_SECRET"))
    timeout = float(_env("OSS_TIMEOUT_MS", "120000")) / 1000.0
    return oss2.Bucket(
        auth,
        f"https://{endpoint}",
        _env("OSS_BUCKET"),
        connect_timeout=timeout,
    )


def _signed_seconds() -> int:
    n = int(_env("OSS_SIGNED_URL_SECONDS", str(7 * 24 * 3600)) or 0)
    return n if n > 60 else 7 * 24 * 3600


def sign_url(key: str) -> str:
    require_configured()
    key = key.lstrip("/")
    mode = _env("OSS_URL_MODE", "signed").lower()
    if mode == "public":
        public_base = _env("OSS_PUBLIC_BASE_URL").rstrip("/")
        if public_base:
            return f"{public_base}/{key}"
        bucket = _env("OSS_BUCKET")
        ep = (
            _env("OSS_PUBLIC_ENDPOINT") or _env("OSS_ENDPOINT") or "oss-cn-shanghai.aliyuncs.com"
        )
        ep = ep.replace("https://", "").replace("http://", "")
        return f"https://{bucket}.{ep}/{key}"
    seconds = _signed_seconds()
    url = _bucket().sign_url("GET", key, seconds, slash_safe=True)
    if url.startswith("http://"):
        url = "https://" + url[len("http://") :]
    return url


def upload_file(local_path: Path, key: str) -> dict:
    require_configured()
    path = Path(local_path)
    if not path.is_file():
        raise RuntimeError("保存文件失败，请稍后重试")
    key = key.lstrip("/")
    bucket = _bucket()
    headers = {
        "Content-Type": _content_type(key),
        "Content-Disposition": "inline",
    }
    result = bucket.put_object_from_file(key, str(path), headers=headers)
    status = int(getattr(result, "status", 200) or 200)
    if status >= 300:
        raise RuntimeError("保存回放失败，请稍后重试")
    url = sign_url(key)
    expires = datetime.now(timezone.utc) + timedelta(seconds=_signed_seconds())
    return {
        "key": key,
        "url": url,
        "expires_at": expires.isoformat(timespec="seconds"),
    }


def refresh_report_urls(report: dict) -> dict:
    """Re-sign stored oss_key fields so the report stays playable."""
    if not configured():
        raise RuntimeError("服务暂时不可用，请稍后重试")
    pairs = (
        ("overlay_oss_key", "overlay_video"),
        ("cover_oss_key", "cover_image"),
        ("preview_oss_key", "preview"),
    )
    for key_field, url_field in pairs:
        oss_key = report.get(key_field)
        if oss_key:
            report[url_field] = sign_url(str(oss_key))
    for clip in report.get("clips") or []:
        for swing in clip.get("swings") or []:
            for phase in (swing.get("phases") or {}).values():
                if isinstance(phase, dict) and phase.get("oss_key"):
                    phase["image"] = sign_url(str(phase["oss_key"]))
    return report
