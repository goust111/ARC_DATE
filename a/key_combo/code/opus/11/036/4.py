from __future__ import annotations

from collections import Counter, deque
import random
from typing import Iterable

import numpy as np


BG = 0
INPUT_COLORS = (1, 2, 3, 4, 7)
SHARED_RECOLOR = 5
UNIQUE_RECOLOR = 6

GEN_GRID_ROWS = (14, 18)
GEN_GRID_COLS = (14, 18)
GEN_OBJECT_COUNT = (5, 7)
GEN_SHARED_COUNT = (2, 3)
GEN_AREA_RANGE = (4, 8)
GEN_MAX_BBOX_SIDE = 5
GEN_MAX_ATTEMPTS = 400
GEN_SHAPE_ATTEMPTS = 800


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _normalize_coords(coords: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    pts = [(int(r), int(c)) for r, c in coords]
    r0 = min(r for r, _c in pts)
    c0 = min(c for _r, c in pts)
    return tuple(sorted((r - r0, c - c0) for r, c in pts))


def _mask_from_shape(shape: tuple[tuple[int, int], ...]) -> np.ndarray:
    rows = [r for r, _c in shape]
    cols = [c for _r, c in shape]
    h = max(rows) + 1
    w = max(cols) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in shape:
        mask[int(r), int(c)] = True
    return mask


def _shape_from_mask(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(r), int(c)) for r, c in np.argwhere(mask)))


def _mask_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for base in (mask, np.fliplr(mask)):
        for k in range(4):
            variant = np.rot90(base, k)
            key = (tuple(map(int, variant.shape)), variant.tobytes())
            if key not in seen:
                seen.add(key)
                variants.append(variant.copy())
    return variants


def _shape_variants(shape: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    mask = _mask_from_shape(shape)
    return [_shape_from_mask(variant) for variant in _mask_variants(mask)]


def _canonical_shape_key(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    best: tuple[tuple[int, int], tuple[int, ...]] | None = None
    for variant in _mask_variants(mask):
        key = (
            tuple(map(int, variant.shape)),
            tuple(int(v) for v in variant.reshape(-1)),
        )
        if best is None or key < best:
            best = key
    assert best is not None
    return best


def _half_masks(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = mask.shape
    lower = np.zeros_like(mask, dtype=bool)
    right = np.zeros_like(mask, dtype=bool)
    lower[(h + 1) // 2 :, :] = mask[(h + 1) // 2 :, :]
    right[:, (w + 1) // 2 :] = mask[:, (w + 1) // 2 :]
    return lower, right


def _shape_is_usable(mask: np.ndarray, *, require_multiple_variants: bool = False) -> bool:
    area = int(mask.sum())
    h, w = map(int, mask.shape)
    if not (GEN_AREA_RANGE[0] <= area <= GEN_AREA_RANGE[1]):
        return False
    if not (2 <= h <= GEN_MAX_BBOX_SIDE and 2 <= w <= GEN_MAX_BBOX_SIDE):
        return False
    if area == h * w:
        return False
    lower, right = _half_masks(mask)
    if not lower.any() or not right.any():
        return False
    if np.array_equal(lower, right):
        return False
    if not mask[: (h + 1) // 2, :].any():
        return False
    if not mask[:, : (w + 1) // 2].any():
        return False
    if require_multiple_variants and len(_mask_variants(mask)) < 2:
        return False
    return True


def _find_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == BG:
                continue
            color = int(g[r, c])
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            coords = np.array(cells, dtype=int)
            bbox = _bbox_coords(coords)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": bbox,
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _component_mask(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = comp["coords"] - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _apply_rule_from_components(grid: np.ndarray, comps: list[dict]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    keys = [_canonical_shape_key(_component_mask(comp)) for comp in comps]
    freq = Counter(keys)

    for comp, key in zip(comps, keys):
        mask = _component_mask(comp)
        r0, c0, _r1, _c1 = comp["bbox"]
        local = out[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]]
        lower, right = _half_masks(mask)
        if int(freq[key]) >= 2:
            local[lower] = SHARED_RECOLOR
        else:
            local[right] = UNIQUE_RECOLOR
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each object's lower or right half based on whether its shape is shared globally.

    Input structure:
    - The background is black.
    - There are 5-7 separated monochrome 4-connected foreground objects.
    - Each object has area 4-8, bbox height and width at least 2, and positions stay fixed.

    Rule:
    - Canonicalize every object's binary shape up to the 8 dihedral transforms.
    - If that canonical shape appears in 2 or more objects, split the object's bbox horizontally and recolor only the object's pixels in the lower floor(h/2) rows to gray.
    - If that canonical shape appears once, split the object's bbox vertically and recolor only the object's pixels in the right floor(w/2) columns to magenta.

    Key insight:
    - Shape frequency decides both the branch and the recolor target, so the solver must compare all objects under rotation/flip-invariant shape matching before applying the local half-split recolor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    comps = _find_components(g)
    assert 5 <= len(comps) <= 7, "expected 5-7 objects"
    return _apply_rule_from_components(g, comps)


def _random_polyomino(area: int) -> tuple[tuple[int, int], ...]:
    cells = {(0, 0)}
    while len(cells) < int(area):
        frontier = set()
        for r, c in cells:
            frontier.update(_neighbors4(r, c))
        frontier -= cells
        cells.add(random.choice(tuple(frontier)))
    return _normalize_coords(cells)


def _random_shape(
    *,
    preferred_area: int | None = None,
    forbidden_keys: set[tuple[tuple[int, int], tuple[int, ...]]] | None = None,
    require_multiple_variants: bool = False,
) -> dict:
    forbidden = forbidden_keys or set()
    for _ in range(GEN_SHAPE_ATTEMPTS):
        area = int(preferred_area) if preferred_area is not None else random.randint(*GEN_AREA_RANGE)
        shape = _random_polyomino(area)
        mask = _mask_from_shape(shape)
        if not _shape_is_usable(mask, require_multiple_variants=require_multiple_variants):
            continue
        key = _canonical_shape_key(mask)
        if key in forbidden:
            continue
        variants = _shape_variants(shape)
        return {
            "shape": shape,
            "mask": mask,
            "key": key,
            "area": area,
            "variants": variants,
        }
    raise RuntimeError("failed to generate a usable shape")


def _boxes_far_enough(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[2] + 1 < b[0] or b[2] + 1 < a[0] or a[3] + 1 < b[1] or b[3] + 1 < a[1]


def _place_records(records: list[dict], rows: int, cols: int) -> bool:
    order = sorted(
        range(len(records)),
        key=lambda i: records[i]["mask"].shape[0] * records[i]["mask"].shape[1],
        reverse=True,
    )
    placed: dict[int, tuple[int, int, int, int]] = {}
    boxes: list[tuple[int, int, int, int]] = []

    def backtrack(pos: int) -> bool:
        if pos == len(order):
            return True

        idx = order[pos]
        h, w = map(int, records[idx]["mask"].shape)
        candidates = [(r, c) for r in range(rows - h + 1) for c in range(cols - w + 1)]
        random.shuffle(candidates)
        for r0, c0 in candidates:
            box = (r0, c0, r0 + h - 1, c0 + w - 1)
            if all(_boxes_far_enough(box, other) for other in boxes):
                boxes.append(box)
                placed[idx] = box
                if backtrack(pos + 1):
                    return True
                boxes.pop()
                placed.pop(idx, None)
        return False

    if not backtrack(0):
        return False
    for idx, box in placed.items():
        records[idx]["bbox"] = box
    return True


def _render_input(records: list[dict], rows: int, cols: int) -> np.ndarray:
    grid = np.zeros((rows, cols), dtype=int)
    for rec in records:
        r0, c0, _r1, _c1 = rec["bbox"]
        mask = rec["mask"]
        local = grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]]
        local[mask] = int(rec["color"])
    return grid


def _render_expected(records: list[dict], grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    freq = Counter(rec["key"] for rec in records)
    for rec in records:
        r0, c0, _r1, _c1 = rec["bbox"]
        mask = rec["mask"]
        local = out[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]]
        lower, right = _half_masks(mask)
        if int(freq[rec["key"]]) >= 2:
            local[lower] = SHARED_RECOLOR
        else:
            local[right] = UNIQUE_RECOLOR
    return out


def _build_records() -> list[dict]:
    n_objects = random.randint(*GEN_OBJECT_COUNT)
    shared_count = random.randint(*GEN_SHARED_COUNT)

    shared_template = _random_shape(require_multiple_variants=True)
    forbidden = {shared_template["key"]}

    unique_templates = [
        _random_shape(preferred_area=shared_template["area"], forbidden_keys=forbidden)
    ]
    forbidden.add(unique_templates[0]["key"])
    while len(unique_templates) < n_objects - shared_count:
        candidate = _random_shape(forbidden_keys=forbidden)
        unique_templates.append(candidate)
        forbidden.add(candidate["key"])

    shared_variants = shared_template["variants"]
    if len(shared_variants) < 2:
        raise RuntimeError("shared template must have at least two visible variants")

    chosen_shared = random.sample(shared_variants, 2)
    while len(chosen_shared) < shared_count:
        chosen_shared.append(random.choice(shared_variants))
    random.shuffle(chosen_shared)

    bridge_color = random.choice(INPUT_COLORS)
    shared_colors = [bridge_color] + [random.choice(INPUT_COLORS) for _ in range(shared_count - 1)]
    unique_colors = [bridge_color] + [random.choice(INPUT_COLORS) for _ in range(len(unique_templates) - 1)]
    random.shuffle(shared_colors)
    random.shuffle(unique_colors)

    records: list[dict] = []
    for shape, color in zip(chosen_shared, shared_colors):
        mask = _mask_from_shape(shape)
        records.append(
            {
                "shape": shape,
                "mask": mask,
                "key": shared_template["key"],
                "area": int(mask.sum()),
                "color": int(color),
                "raw_shape": shape,
            }
        )

    for template, color in zip(unique_templates, unique_colors):
        shape = random.choice(template["variants"])
        mask = _mask_from_shape(shape)
        records.append(
            {
                "shape": shape,
                "mask": mask,
                "key": template["key"],
                "area": int(mask.sum()),
                "color": int(color),
                "raw_shape": shape,
            }
        )

    random.shuffle(records)
    return records


def _validate_generated_case(grid: np.ndarray, expected: np.ndarray, records: list[dict]) -> bool:
    comps = _find_components(grid)
    if len(comps) != len(records):
        return False

    derived_keys = []
    for comp in comps:
        mask = _component_mask(comp)
        if not _shape_is_usable(mask):
            return False
        derived_keys.append(_canonical_shape_key(mask))

    freq = Counter(derived_keys)
    repeated = [key for key, count in freq.items() if int(count) >= 2]
    if len(repeated) != 1 or int(freq[repeated[0]]) not in GEN_SHARED_COUNT:
        return False
    if any(int(count) != 1 for key, count in freq.items() if key != repeated[0]):
        return False

    if not np.array_equal(main(grid), expected):
        return False
    if not np.array_equal(_apply_rule_from_components(grid, comps), expected):
        return False

    branches = Counter("shared" if int(freq[rec["key"]]) >= 2 else "unique" for rec in records)
    if branches["shared"] < 2 or branches["unique"] < 2:
        return False

    shared_records = [rec for rec in records if int(freq[rec["key"]]) >= 2]
    unique_records = [rec for rec in records if int(freq[rec["key"]]) == 1]
    if len({rec["raw_shape"] for rec in shared_records}) < 2:
        return False
    if not any(s["color"] == u["color"] for s in shared_records for u in unique_records):
        return False

    shared_areas = {rec["area"] for rec in shared_records}
    if not any(rec["area"] in shared_areas for rec in unique_records):
        return False

    for rec in records:
        mask = rec["mask"]
        lower, right = _half_masks(mask)
        if lower.sum() == 0 or right.sum() == 0 or np.array_equal(lower, right):
            return False
    return True


def _attempt_generate() -> np.ndarray | None:
    records = _build_records()
    rows = random.randint(*GEN_GRID_ROWS)
    cols = random.randint(*GEN_GRID_COLS)
    if not _place_records(records, rows, cols):
        return None

    grid = _render_input(records, rows, cols)
    expected = _render_expected(records, grid)
    if not _validate_generated_case(grid, expected, records):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
