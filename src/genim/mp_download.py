from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import requests
from tqdm import tqdm

from .chem import ChemistryPolicy, METALLOIDS
from .structure_format import StructureRecord, parse_pymatgen_mson_structure
from .symmetry import extract_wyckoff_structure


MP_BASE = "https://api.materialsproject.org"
MP_SUMMARY = f"{MP_BASE}/materials/summary/"
GENMAT_USER_AGENT = "genmat/0.6.0"


@dataclass(frozen=True)
class MPDownloadStats:
    kept: int = 0
    skipped: int = 0
    spacegroups_covered: int | None = None
    spacegroups_target: int | None = None


def _mp_session(timeout: float) -> requests.Session:
    api_key = (
        os.environ.get("MP_API_KEY", "").strip()
        or os.environ.get("PMG_MAPI_KEY", "").strip()
        or os.environ.get("MAPI_KEY", "").strip()
    )
    if not api_key:
        for p in (Path.cwd() / ".mp_api_key", Path.home() / ".mp_api_key"):
            try:
                api_key = p.read_text(encoding="utf-8").strip()
            except FileNotFoundError:
                continue
            if api_key:
                break
    if not api_key:
        raise RuntimeError("Missing Materials Project API key (set MP_API_KEY or PMG_MAPI_KEY).")
    s = requests.Session()
    s.headers.update(
        {
            "X-API-KEY": api_key,
            "Accept": "application/json",
            "User-Agent": GENMAT_USER_AGENT,
        }
    )
    s.request = _wrap_timeout(s.request, timeout)
    return s


def _wrap_timeout(request_fn, timeout: float):
    def _wrapped(method, url, **kwargs):
        kwargs.setdefault("timeout", timeout)
        return request_fn(method, url, **kwargs)

    return _wrapped


def _iter_mp_summary(
    *,
    session: requests.Session,
    params: dict[str, Any],
    limit: int,
    per_page: int,
) -> Iterable[dict[str, Any]]:
    """
    Iterate MP summary docs with a conservative pagination strategy.

    MP API commonly supports `_limit` / `_skip`. We use that as primary,
    and stop when we hit `limit` or the API returns fewer docs than requested.
    """
    skip = 0
    yielded = 0
    while yielded < limit:
        batch_n = min(per_page, limit - yielded)
        q = dict(params)
        q["_limit"] = batch_n
        q["_skip"] = skip

        resp = session.get(MP_SUMMARY, params=q)
        if resp.status_code == 429:
            time.sleep(2.0)
            continue
        if resp.status_code == 401:
            raise RuntimeError(
                "MP API authentication failed (401). "
                "Your API key is missing/invalid. "
                "Regenerate a key in your Materials Project account and set MP_API_KEY."
            )
        if resp.status_code == 403:
            raise RuntimeError(
                "MP API request forbidden (403). "
                "This can happen if the legacy REST v2 endpoints are blocked or your key lacks access. "
                "Use the official API endpoint at api.materialsproject.org."
            )
        resp.raise_for_status()

        payload = resp.json()
        docs = payload.get("data", payload)
        if not isinstance(docs, list):
            raise RuntimeError(f"Unexpected MP response shape: {type(docs)}")

        if not docs:
            return

        for d in docs:
            yield d
            yielded += 1
            if yielded >= limit:
                return

        if len(docs) < batch_n:
            return
        skip += len(docs)


def download_mp_jsonl(
    *,
    out_path: Path,
    chemsys: str | None,
    elements: list[str] | None,
    chemistry_filter: str = "any",
    max_atoms: int,
    eah_max: float,
    nelements_min: int,
    nelements_max: int,
    limit: int,
    per_page: int,
    timeout: float,
    include_metalloids: bool,
    spacegroup_number: int | None = None,
    balance_spacegroups: bool = False,
    per_spacegroup: int = 0,
    verify_spacegroup: bool = False,
    verify_symprec: float = 1e-2,
) -> MPDownloadStats:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    chemistry_filter = str(chemistry_filter or "any").strip().lower()
    policy = ChemistryPolicy(
        mode=chemistry_filter,
        include_metalloids=include_metalloids,
        min_elements=max(int(nelements_min), 2 if chemistry_filter == "intermetallic" else 1),
        max_elements=nelements_max,
    )
    session = _mp_session(timeout)

    allowed_elements = set(elements) if elements else None

    exclude_str = None
    if chemistry_filter in {"metallic", "intermetallic"}:
        # NOTE: MP API `exclude_elements` has a short maxLength (60). Use a compact list
        # to cut out the most common non-metals; the selected metallic policy is
        # still enforced locally below.
        exclude_list = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
        if not include_metalloids:
            exclude_list += sorted(METALLOIDS)
        exclude_str = ",".join(exclude_list)
        if len(exclude_str) > 60:
            exclude_str = ",".join(exclude_list[:10])

    params_base: dict[str, Any] = {
        "_fields": ",".join(
            [
                "material_id",
                "formula_pretty",
                "nsites",
                "elements",
                "energy_above_hull",
                "structure",
            ]
        ),
        # Server-side filters (match the published OpenAPI spec as of 2026-03-03).
        "nsites_max": int(max_atoms),
        "energy_above_hull_max": float(eah_max),
        "nelements_min": int(nelements_min),
        "nelements_max": int(nelements_max),
    }
    if exclude_str:
        params_base["exclude_elements"] = exclude_str
    if chemsys:
        params_base["chemsys"] = chemsys

    if balance_spacegroups and spacegroup_number is not None:
        raise ValueError("Use either spacegroup_number or balance_spacegroups, not both.")
    if spacegroup_number is not None and not (1 <= int(spacegroup_number) <= 230):
        raise ValueError("spacegroup_number must be in 1..230")
    if per_spacegroup < 0:
        raise ValueError("per_spacegroup must be >= 0")

    kept = 0
    skipped = 0

    verify_sg = bool(verify_spacegroup) or bool(balance_spacegroups) or (spacegroup_number is not None)

    def _handle_doc(doc: dict[str, Any], f, *, expected_sg: int | None = None) -> bool:
        nonlocal kept, skipped
        try:
            nsites = int(doc.get("nsites", 0) or 0)
            if nsites <= 0 or nsites > max_atoms:
                skipped += 1
                return False

            eah = doc.get("energy_above_hull", None)
            if eah is not None and float(eah) > eah_max:
                skipped += 1
                return False

            elems = doc.get("elements", None)
            if isinstance(elems, list):
                elems_s = [str(x) for x in elems]
                if not policy.accepts_elements(elems_s):
                    skipped += 1
                    return False
                if allowed_elements is not None and not set(elems_s).issubset(allowed_elements):
                    skipped += 1
                    return False

            struct_obj = doc.get("structure", None)
            if not isinstance(struct_obj, dict):
                skipped += 1
                return False

            rec: StructureRecord = parse_pymatgen_mson_structure(struct_obj)
            if not policy.accepts_elements(set(rec.species)):
                skipped += 1
                return False
            if allowed_elements is not None and not set(rec.species).issubset(allowed_elements):
                skipped += 1
                return False

            if verify_sg and expected_sg is not None:
                try:
                    wy = extract_wyckoff_structure(rec.to_ase(), symprec=float(verify_symprec))
                    if int(wy.spacegroup_number) != int(expected_sg):
                        skipped += 1
                        return False
                except Exception:
                    skipped += 1
                    return False

            row = {
                "source": "mp",
                "id": doc.get("material_id", None),
                "formula": doc.get("formula_pretty", None),
                "nsites": nsites,
                "energy_above_hull": eah,
                "structure": {
                    "lattice": rec.lattice.tolist(),
                    "frac_coords": rec.frac_coords.tolist(),
                    "species": rec.species,
                },
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
            return True
        except Exception:
            skipped += 1
            return False

    with out_path.open("w", encoding="utf-8") as f:
        if balance_spacegroups:
            if per_spacegroup <= 0:
                raise ValueError("per_spacegroup must be > 0 when balance_spacegroups is enabled")

            sg_target = 230
            sg_covered = 0
            for sg in tqdm(range(1, 231), total=230, desc="MP download (SG-balanced)"):
                if kept >= limit:
                    break
                params = dict(params_base)
                params["spacegroup_number"] = int(sg)
                sg_kept = 0
                sg_had_any = False

                # Keep scanning docs for this SG until we *keep* per_spacegroup items (or we run out).
                skip = 0
                while sg_kept < per_spacegroup and kept < limit:
                    batch_n = min(per_page, max(1, per_spacegroup * 5))
                    q = dict(params)
                    q["_limit"] = batch_n
                    q["_skip"] = skip

                    resp = session.get(MP_SUMMARY, params=q)
                    if resp.status_code == 429:
                        time.sleep(2.0)
                        continue
                    resp.raise_for_status()

                    payload = resp.json()
                    docs = payload.get("data", payload)
                    if not isinstance(docs, list) or not docs:
                        break

                    sg_had_any = True
                    for doc in docs:
                        if kept >= limit or sg_kept >= per_spacegroup:
                            break
                        if _handle_doc(doc, f, expected_sg=int(sg)):
                            sg_kept += 1

                    if len(docs) < batch_n:
                        break
                    skip += len(docs)

                if sg_had_any and sg_kept > 0:
                    sg_covered += 1

            return MPDownloadStats(kept=kept, skipped=skipped, spacegroups_covered=sg_covered, spacegroups_target=sg_target)

        params = dict(params_base)
        if spacegroup_number is not None:
            params["spacegroup_number"] = int(spacegroup_number)

        it = _iter_mp_summary(session=session, params=params, limit=limit, per_page=per_page)
        for doc in tqdm(it, total=limit, desc="MP download"):
            _handle_doc(doc, f, expected_sg=int(spacegroup_number) if spacegroup_number is not None else None)

    return MPDownloadStats(kept=kept, skipped=skipped, spacegroups_covered=None, spacegroups_target=None)
