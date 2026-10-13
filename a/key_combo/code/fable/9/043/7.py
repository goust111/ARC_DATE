from __future__ import annotations

import random
from collections import Counter

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _component_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.array(cells, dtype=int)
            objects.append({"color": color, "coords": coords})
    return objects


def _compress_mask(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask, dtype=bool)
    assert arr.ndim == 2 and arr.any()

    keep_rows = [0]
    for r in range(1, arr.shape[0]):
        if not np.array_equal(arr[r], arr[r - 1]):
            keep_rows.append(r)
    arr = arr[keep_rows, :]

    keep_cols = [0]
    for c in range(1, arr.shape[1]):
        if not np.array_equal(arr[:, c], arr[:, c - 1]):
            keep_cols.append(c)
    return arr[:, keep_cols]


def _serialize(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    arr = np.asarray(mask, dtype=bool)
    return tuple(tuple(int(v) for v in row) for row in arr)


def _serialize_grid(grid: np.ndarray) -> tuple[tuple[int, ...], ...]:
    arr = np.asarray(grid, dtype=int)
    return tuple(tuple(int(v) for v in row) for row in arr)


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    arr = np.asarray(mask, dtype=bool)
    out: list[np.ndarray] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()
    for turns in range(4):
        rotated = np.rot90(arr, turns)
        for variant in (rotated, np.fliplr(rotated)):
            key = _serialize(variant)
            if key not in seen:
                seen.add(key)
                out.append(variant.copy())
    return out


def _canonical_signature(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return min(_serialize(variant) for variant in _dihedral_variants(mask))


def _compress_object(coords: np.ndarray) -> np.ndarray:
    return _compress_mask(_component_mask(coords))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the single topological outlier after run-length normalization and output its compressed form.

    Input structure:
    - A black-background grid contains several separated monochrome objects.
    - Objects may use different colors, sizes, aspect ratios, and orientations.
    - After bbox cropping and collapsing adjacent identical rows and columns, all but one object share the same shape up to rotation and reflection.

    Rule:
    - Split the foreground into 4-connected monochrome objects.
    - For each object, crop to its bbox and run-compress repeated adjacent rows and columns.
    - Group compressed masks under rotation/reflection equivalence, find the unique object outside the majority class, and output that object's compressed mask in its original scene orientation with its original color.

    Key insight:
    - Color, area, raw bbox, and orientation are distractions; only run-compressed topology identifies the outlier, while the returned sprite keeps the outlier's own orientation instead of a normalized one.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    objects = _components(g, bg)
    assert len(objects) >= 3

    records = []
    for obj in objects:
        compressed = _compress_object(obj["coords"])
        records.append(
            {
                "color": int(obj["color"]),
                "compressed": compressed,
                "canonical": _canonical_signature(compressed),
            }
        )

    counts = Counter(record["canonical"] for record in records)
    assert len(counts) == 2
    majority_signature, majority_count = counts.most_common(1)[0]
    assert majority_count == len(records) - 1

    outliers = [record for record in records if record["canonical"] != majority_signature]
    assert len(outliers) == 1
    chosen = outliers[0]

    out = np.zeros(chosen["compressed"].shape, dtype=int)
    out[chosen["compressed"]] = chosen["color"]
    return out


def _mask(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


_BASE_MASKS = [
    _mask(["111", "101", "111"]),
    _mask(["010", "111", "010"]),
    _mask(["111", "010", "010"]),
    _mask(["101", "111"]),
    _mask(["101", "111", "101"]),
    _mask(["110", "011"]),
    _mask(["100", "111"]),
    _mask(["1111", "1001", "1111"]),
    _mask(["111", "110", "100"]),
    _mask(["111", "001", "111"]),
    _mask(["011", "110", "010"]),
    _mask(["111", "101", "100"]),
]

_PALETTE = tuple(range(1, 10))
_GRID_H_RANGE = (18, 31)
_GRID_W_RANGE = (18, 31)
_OBJECT_COUNT_RANGE = (4, 7)
_MAX_PATCH_DIM = 9
_MAX_EXTRA_REPEAT = 4
_MAX_GENERATION_ATTEMPTS = 500
_SEEN_OUTPUT_SIGS: set[tuple[tuple[int, ...], ...]] = set()


def _check_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    pending = [tuple(int(x) for x in coords[0])]
    seen = {pending[0]}
    target = {tuple(int(x) for x in rc) for rc in coords}
    while pending:
        r, c = pending.pop()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in target and (nr, nc) not in seen:
                seen.add((nr, nc))
                pending.append((nr, nc))
    return len(seen) == len(target)


def _stretch_mask(mask: np.ndarray, row_factors: tuple[int, ...], col_factors: tuple[int, ...]) -> np.ndarray:
    return np.repeat(np.repeat(mask, row_factors, axis=0), col_factors, axis=1)


def _factor_options(length: int) -> list[tuple[int, ...]]:
    options: list[tuple[int, ...]] = []
    for factors in np.ndindex(*(3 for _ in range(length))):
        vals = tuple(v + 1 for v in factors)
        if sum(vals) - length <= _MAX_EXTRA_REPEAT:
            options.append(vals)
    return options


def _variant_records(base: np.ndarray) -> list[dict]:
    variants: list[dict] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()
    compressed_variants = _dihedral_variants(base)
    for oriented in compressed_variants:
        orientation_key = _serialize(oriented)
        row_opts = _factor_options(oriented.shape[0])
        col_opts = _factor_options(oriented.shape[1])
        for row_factors in row_opts:
            for col_factors in col_opts:
                raw = _stretch_mask(oriented, row_factors, col_factors)
                if raw.shape[0] > _MAX_PATCH_DIM or raw.shape[1] > _MAX_PATCH_DIM:
                    continue
                key = _serialize(raw)
                if key in seen:
                    continue
                seen.add(key)
                variants.append(
                    {
                        "raw": raw,
                        "compressed": oriented.copy(),
                        "orientation_key": orientation_key,
                        "area": int(raw.sum()),
                        "bbox_shape": tuple(int(x) for x in raw.shape),
                        "stretched": raw.shape != oriented.shape,
                    }
                )
    return variants


_UNIQUE_BASES: list[np.ndarray] = []
_BASE_SIGNATURES: list[tuple[tuple[int, ...], ...]] = []
for _base in _BASE_MASKS:
    assert _check_connected(_base)
    _sig = _canonical_signature(_base)
    if _sig not in _BASE_SIGNATURES:
        _BASE_SIGNATURES.append(_sig)
        _UNIQUE_BASES.append(_base)

_VARIANT_POOL = {idx: _variant_records(base) for idx, base in enumerate(_UNIQUE_BASES)}


def _dilate_chebyshev(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    coords = np.argwhere(mask)
    for r, c in coords:
        r0 = max(0, int(r) - radius)
        r1 = min(h, int(r) + radius + 1)
        c0 = max(0, int(c) - radius)
        c1 = min(w, int(c) + radius + 1)
        out[r0:r1, c0:c1] = True
    return out


def _place_objects(height: int, width: int, objects: list[dict]) -> np.ndarray | None:
    canvas = np.zeros((height, width), dtype=int)
    occupied = np.zeros((height, width), dtype=bool)
    blocked = np.zeros((height, width), dtype=bool)

    order = sorted(range(len(objects)), key=lambda i: objects[i]["raw"].shape[0] * objects[i]["raw"].shape[1], reverse=True)
    for idx in order:
        obj = objects[idx]
        raw = obj["raw"]
        raw_coords = np.argwhere(raw)
        ph, pw = raw.shape
        candidates: list[tuple[int, int]] = []
        for r0 in range(height - ph + 1):
            for c0 in range(width - pw + 1):
                rr = raw_coords[:, 0] + r0
                cc = raw_coords[:, 1] + c0
                if np.any(blocked[rr, cc]):
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        rr = raw_coords[:, 0] + r0
        cc = raw_coords[:, 1] + c0
        canvas[rr, cc] = int(obj["color"])
        occupied[rr, cc] = True
        blocked = _dilate_chebyshev(occupied, radius=1)
    return canvas


def _topology_records(grid: np.ndarray) -> list[dict]:
    bg = _get_bg(grid)
    out: list[dict] = []
    for obj in _components(grid, bg):
        raw = _component_mask(obj["coords"])
        compressed = _compress_mask(raw)
        out.append(
            {
                "color": int(obj["color"]),
                "raw": raw,
                "compressed": compressed,
                "canonical": _canonical_signature(compressed),
                "area": int(raw.sum()),
                "bbox_shape": tuple(int(x) for x in raw.shape),
                "bbox_area": int(raw.shape[0] * raw.shape[1]),
            }
        )
    return out


def _select_outlier(records: list[dict]) -> int | None:
    counts = Counter(record["canonical"] for record in records)
    if len(counts) != 2:
        return None
    majority_signature, majority_count = counts.most_common(1)[0]
    if majority_count != len(records) - 1:
        return None
    outlier_ids = [i for i, record in enumerate(records) if record["canonical"] != majority_signature]
    return outlier_ids[0] if len(outlier_ids) == 1 else None


def _shares_value(records: list[dict], outlier_idx: int, key: str) -> bool:
    target = records[outlier_idx][key]
    return any(i != outlier_idx and records[i][key] == target for i in range(len(records)))


def _is_unique_extreme(values: list[int], index: int) -> bool:
    lo = min(values)
    hi = max(values)
    return (values[index] == lo and values.count(lo) == 1) or (values[index] == hi and values.count(hi) == 1)


def _semantic_ok(grid: np.ndarray) -> bool:
    records = _topology_records(grid)
    outlier_idx = _select_outlier(records)
    if outlier_idx is None:
        return False

    outlier = records[outlier_idx]
    majority = [record for i, record in enumerate(records) if i != outlier_idx]

    if len(set(_serialize(record["compressed"]) for record in majority)) < 2:
        return False
    if not any(record["raw"].shape != record["compressed"].shape for record in majority):
        return False
    if not _shares_value(records, outlier_idx, "color"):
        return False
    if not (_shares_value(records, outlier_idx, "area") or _shares_value(records, outlier_idx, "bbox_shape")):
        return False

    areas = [record["area"] for record in records]
    bbox_areas = [record["bbox_area"] for record in records]
    if _is_unique_extreme(areas, outlier_idx) or _is_unique_extreme(bbox_areas, outlier_idx):
        return False

    expected = np.zeros(outlier["compressed"].shape, dtype=int)
    expected[outlier["compressed"]] = outlier["color"]
    if not np.array_equal(main(grid), expected):
        return False
    return True


def _sample_scene() -> np.ndarray | None:
    height = random.randrange(*_GRID_H_RANGE)
    width = random.randrange(*_GRID_W_RANGE)
    n_objects = random.randint(*_OBJECT_COUNT_RANGE)
    majority_count = n_objects - 1

    majority_base, outlier_base = random.sample(range(len(_UNIQUE_BASES)), 2)
    majority_variants = _VARIANT_POOL[majority_base]
    outlier_variants = _VARIANT_POOL[outlier_base]

    objects: list[dict] = []
    forced_color = random.choice(_PALETTE)
    outlier_variant = random.choice(outlier_variants)
    objects.append({**outlier_variant, "color": forced_color})

    same_color_slot = random.randrange(majority_count)
    for i in range(majority_count):
        variant = random.choice(majority_variants)
        color = forced_color if i == same_color_slot else random.choice(_PALETTE)
        objects.append({**variant, "color": color})

    if len({obj["orientation_key"] for obj in objects[1:]}) < 2:
        return None

    random.shuffle(objects)
    grid = _place_objects(height, width, objects)
    if grid is None:
        return None
    if not _semantic_ok(grid):
        return None

    output_sig = _serialize_grid(main(grid))
    if output_sig in _SEEN_OUTPUT_SIGS:
        return None
    _SEEN_OUTPUT_SIGS.add(output_sig)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _sample_scene()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid topology-outlier scene")
