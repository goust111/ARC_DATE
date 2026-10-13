from __future__ import annotations

from collections import deque
from functools import lru_cache
from itertools import combinations
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

P_TO_FILL = {
    1: BLUE,
    2: RED,
    3: GREEN,
    4: YELLOW,
}

RIGHT_OBJECT_COLORS = [6, 7, 8, 9]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _shape_key(coords: np.ndarray) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = _bbox(coords)
    offsets = tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))
    return (r1 - r0 + 1, c1 - c0 + 1), offsets


def _normalize_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, _, _ = _bbox(coords)
    out = np.asarray(coords, dtype=int).copy()
    out[:, 0] -= r0
    out[:, 1] -= c0
    return out


def _mask_from_coords(coords: np.ndarray) -> np.ndarray:
    coords = _normalize_coords(coords)
    h = int(coords[:, 0].max()) + 1
    w = int(coords[:, 1].max()) + 1
    mask = np.zeros((h, w), dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _coords_from_mask(mask: np.ndarray) -> np.ndarray:
    return np.argwhere(np.asarray(mask, dtype=bool))


def _coords_connected(coords: np.ndarray) -> bool:
    coords = np.asarray(coords, dtype=int)
    pts = {tuple(map(int, rc)) for rc in coords}
    if not pts:
        return False
    start = next(iter(pts))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in pts and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(pts)


def _rotate_coords_90(coords: np.ndarray, center2: tuple[int, int]) -> np.ndarray | None:
    out = []
    cr2, cc2 = center2
    for r, c in np.asarray(coords, dtype=int):
        pr2 = 2 * int(r)
        pc2 = 2 * int(c)
        nr2 = cr2 + (pc2 - cc2)
        nc2 = cc2 - (pr2 - cr2)
        if nr2 % 2 or nc2 % 2:
            return None
        out.append((nr2 // 2, nc2 // 2))
    return np.array(out, dtype=int)


def _completion_candidates(coords: np.ndarray) -> list[dict]:
    coords = np.asarray(coords, dtype=int)
    pts = {tuple(map(int, rc)) for rc in coords}
    r0, c0, r1, c1 = _bbox(coords)
    seen: dict[tuple, dict] = {}
    for cr2 in range(2 * r0 - 2, 2 * r1 + 3):
        for cc2 in range(2 * c0 - 2, 2 * c1 + 3):
            rotated = _rotate_coords_90(coords, (cr2, cc2))
            if rotated is None:
                continue
            rot_pts = {tuple(map(int, rc)) for rc in rotated}
            if len(rot_pts) != len(pts):
                continue
            added = rot_pts - pts
            p = len(added)
            if p < 1 or p > 4:
                continue
            full_pts = pts | rot_pts
            if not (4 <= len(full_pts) <= 8):
                continue
            full_coords = np.array(sorted(full_pts), dtype=int)
            if not _coords_connected(full_coords):
                continue
            fr0, fc0, fr1, fc1 = _bbox(full_coords)
            if max(fr1 - fr0 + 1, fc1 - fc0 + 1) > 6:
                continue
            full_key = _shape_key(full_coords)
            dedupe_key = (full_key, p)
            if dedupe_key in seen:
                continue
            seen[dedupe_key] = {
                "center2": (int(cr2), int(cc2)),
                "full_coords": full_coords,
                "full_key": full_key,
                "p": int(p),
                "added_coords": np.array(sorted(added), dtype=int),
            }
    return list(seen.values())


def _monochromatic_components(grid: np.ndarray, ignore: set[int]) -> list[dict]:
    comps = []
    for color in sorted(set(map(int, grid.flatten())) - set(ignore)):
        for coords in _find_components(grid == color):
            comps.append({"color": int(color), "coords": coords})
    return comps


def _find_separator_col(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert len(cols) == 1, "expected exactly one grey separator column"
    return int(cols[0])


def _match_object_to_hole(coords: np.ndarray, hole_lookup: dict) -> dict:
    matches = {}
    for cand in _completion_candidates(coords):
        hole_idx = hole_lookup.get(cand["full_key"])
        if hole_idx is None:
            continue
        key = (int(hole_idx), int(cand["p"]), cand["full_key"])
        matches[key] = {"hole_idx": int(hole_idx), "p": int(cand["p"]), "full_key": cand["full_key"]}
    assert len(matches) == 1, "expected exactly one hole-matching completion"
    return next(iter(matches.values()))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each right-panel object by one clockwise quarter-turn, match the completed shape to a left-panel hole, and fill that hole by completion count.

    Input structure:
    - A single grey column splits the grid into a blue left template panel and a black right object panel.
    - The left panel is one large blue template containing 3-4 disconnected black hole-regions.
    - The right panel contains 3-4 monochromatic objects; each object's visible pixels are an incomplete version of a hole shape and must be completed by rotation.

    Rule:
    - For each right-panel object, search its quarter-turn rotation center and rotate its pixels clockwise once.
    - Cells hit by that rotation that were previously black are the completion pixels; their count P determines the fill color (1->blue, 2->red, 3->green, 4->yellow).
    - Normalize the completed object shape and match it to the unique normalized hole shape in the left panel, then fill that hole.
    - Return only the filled left panel.

    Key insight:
    - The visible right-panel object is not the final matching shape; only the union of the object and its clockwise rotational completion reveals which left-hole it belongs to, while the number of newly created pixels independently encodes the output color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep = _find_separator_col(g)
    left = g[:, :sep].copy()
    right = g[:, sep + 1 :]

    holes = _find_components(left == BLACK)
    assert 3 <= len(holes) <= 4, "expected 3-4 hole components"
    hole_lookup = {}
    for idx, coords in enumerate(holes):
        key = _shape_key(coords)
        assert key not in hole_lookup, "hole shapes must be unique"
        hole_lookup[key] = idx

    objects = _monochromatic_components(right, ignore={BLACK, GREY})
    assert len(objects) == len(holes), "expected one right-side object per hole"

    assignments = []
    used_holes = set()
    for obj in objects:
        match = _match_object_to_hole(obj["coords"], hole_lookup)
        assert match["hole_idx"] not in used_holes, "hole matched more than once"
        used_holes.add(match["hole_idx"])
        assignments.append(match)

    out = left.copy()
    for match in assignments:
        coords = holes[match["hole_idx"]]
        out[coords[:, 0], coords[:, 1]] = P_TO_FILL[match["p"]]
    return out


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]]:
    return _shape_key(_coords_from_mask(mask))


@lru_cache(maxsize=1)
def _candidate_library() -> tuple[dict, ...]:
    entries: dict[tuple, dict] = {}
    for side in (4, 5):
        cells = [(r, c) for r in range(side) for c in range(side)]
        max_area = 6 if side == 4 else 5
        for area in range(2, max_area + 1):
            for combo in combinations(cells, area):
                coords = np.array(combo, dtype=int)
                if not _coords_connected(coords):
                    continue
                existing_key = _shape_key(coords)
                for cand in _completion_candidates(coords):
                    full_mask = _mask_from_coords(cand["full_coords"])
                    existing_mask = _mask_from_coords(coords)
                    if _mask_key(existing_mask) == cand["full_key"]:
                        continue
                    entry = {
                        "existing_key": existing_key,
                        "existing_mask": existing_mask,
                        "full_key": cand["full_key"],
                        "full_mask": full_mask,
                        "p": int(cand["p"]),
                        "full_area": int(full_mask.sum()),
                    }
                    dedupe_key = (existing_key, cand["full_key"], int(cand["p"]))
                    prev = entries.get(dedupe_key)
                    if prev is None or (
                        full_mask.shape[0] * full_mask.shape[1],
                        existing_mask.shape[0] * existing_mask.shape[1],
                    ) < (
                        prev["full_mask"].shape[0] * prev["full_mask"].shape[1],
                        prev["existing_mask"].shape[0] * prev["existing_mask"].shape[1],
                    ):
                        entries[dedupe_key] = entry

    library = list(entries.values())

    by_p = {p: [item for item in library if item["p"] == p] for p in range(1, 5)}
    assert all(by_p[p] for p in range(1, 5)), "shape library must cover completion counts 1..4"
    return tuple(library)


def _derangement(n: int) -> list[int]:
    perm = list(range(n))
    while True:
        random.shuffle(perm)
        if all(i != perm[i] for i in range(n)):
            return perm


def _choose_completion_counts() -> list[int]:
    if random.random() < 0.7:
        return [1, 2, 3, 4]
    counts = [1, 2, 3, 4]
    random.shuffle(counts)
    return sorted(counts[:3])


def _rotate_pair(existing_mask: np.ndarray, full_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k = random.randrange(4)
    return np.rot90(existing_mask, k).copy(), np.rot90(full_mask, k).copy()


def _sample_shapes(ps: list[int]) -> list[dict]:
    library = _candidate_library()
    by_p = {p: [item for item in library if item["p"] == p] for p in range(1, 5)}
    for attempt in range(500):
        chosen = []
        used_full = set()
        used_existing = set()
        for p in ps:
            pool = [
                item for item in by_p[p]
                if item["full_key"] not in used_full and item["existing_key"] not in used_existing
            ]
            if not pool:
                break
            item = random.choice(pool)
            chosen.append(item)
            used_full.add(item["full_key"])
            used_existing.add(item["existing_key"])
        if len(chosen) != len(ps):
            continue
        if len({item["full_area"] for item in chosen}) == len(chosen) and attempt < 450:
            continue

        transformed = []
        transformed_full_keys = set()
        for item in chosen:
            existing_mask, full_mask = _rotate_pair(item["existing_mask"], item["full_mask"])
            full_key = _mask_key(full_mask)
            existing_key = _mask_key(existing_mask)
            if full_key in transformed_full_keys:
                break
            transformed_full_keys.add(full_key)
            transformed.append(
                {
                    "p": item["p"],
                    "existing_mask": existing_mask,
                    "existing_key": existing_key,
                    "full_mask": full_mask,
                    "full_key": full_key,
                    "full_area": int(full_mask.sum()),
                }
            )
        if len(transformed) != len(ps):
            continue
        if any(item["existing_key"] in transformed_full_keys for item in transformed):
            continue
        return transformed
    raise RuntimeError("failed to sample non-degenerate shape set")


def _render_mask(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    patch = np.full(mask.shape, bg, dtype=int)
    patch[np.asarray(mask, dtype=bool)] = int(color)
    return patch


def _place_patch(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int, transparent: int) -> None:
    h, w = patch.shape
    view = canvas[r0 : r0 + h, c0 : c0 + w]
    mask = patch != transparent
    view[mask] = patch[mask]


def _build_example(shapes: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    count = len(shapes)
    hole_order = list(range(count))
    random.shuffle(hole_order)
    object_order = _derangement(count) if count > 1 else [0]

    hole_shapes = [shapes[i] for i in hole_order]
    object_shapes = [hole_shapes[i] for i in object_order]

    band_heights = []
    max_w = 0
    for hole_shape, obj_shape in zip(hole_shapes, object_shapes):
        hh, hw = hole_shape["full_mask"].shape
        oh, ow = obj_shape["existing_mask"].shape
        band_heights.append(max(hh, oh) + random.randint(2, 3))
        max_w = max(max_w, hw, ow)

    panel_w = max(8, max_w + random.randint(4, 6))
    height = sum(band_heights)
    left = np.full((height, panel_w), BLUE, dtype=int)
    right = np.zeros((height, panel_w), dtype=int)
    out = left.copy()

    object_colors = random.sample(RIGHT_OBJECT_COLORS, count)
    cursor = 0

    for band_idx, band_h in enumerate(band_heights):
        hole_shape = hole_shapes[band_idx]
        obj_shape = object_shapes[band_idx]

        hole_mask = hole_shape["full_mask"]
        obj_mask = obj_shape["existing_mask"]
        hh, hw = hole_mask.shape
        oh, ow = obj_mask.shape

        hole_r = cursor + random.randint(1, band_h - hh - 1)
        hole_c = random.randint(1, panel_w - hw - 1)
        obj_r = cursor + random.randint(1, band_h - oh - 1)
        obj_c = random.randint(1, panel_w - ow - 1)

        hole_patch = _render_mask(hole_mask, BLACK, BLUE)
        obj_patch = _render_mask(obj_mask, object_colors[band_idx], BLACK)

        _place_patch(left, hole_patch, hole_r, hole_c, transparent=BLUE)
        _place_patch(out, _render_mask(hole_mask, P_TO_FILL[hole_shape["p"]], BLUE), hole_r, hole_c, transparent=BLUE)
        _place_patch(right, obj_patch, obj_r, obj_c, transparent=BLACK)
        cursor += band_h

    inp = np.concatenate([left, np.full((height, 1), GREY, dtype=int), right], axis=1)
    return inp, out


def _example_is_good(inp: np.ndarray, out: np.ndarray) -> bool:
    if inp.ndim != 2 or out.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return False

    sep = _find_separator_col(inp)
    left = inp[:, :sep]
    right = inp[:, sep + 1 :]
    holes = _find_components(left == BLACK)
    if not (3 <= len(holes) <= 4):
        return False
    hole_keys = [_shape_key(coords) for coords in holes]
    if len(set(hole_keys)) != len(hole_keys):
        return False

    objects = _monochromatic_components(right, ignore={BLACK, GREY})
    if len(objects) != len(holes):
        return False

    hole_lookup = {key: idx for idx, key in enumerate(hole_keys)}
    matches = []
    full_areas = [int(len(coords)) for coords in holes]
    if len(set(full_areas)) == len(full_areas):
        return False

    for obj in objects:
        obj_key = _shape_key(obj["coords"])
        if obj_key in hole_lookup:
            return False
        try:
            match = _match_object_to_hole(obj["coords"], hole_lookup)
        except AssertionError:
            return False
        matches.append((match["hole_idx"], match["p"]))

    if len(set(matches)) != len(matches):
        return False
    try:
        solved = main(inp)
    except AssertionError:
        return False
    return bool(np.array_equal(solved, out))


def _gen_attempt() -> np.ndarray | None:
    shapes = _sample_shapes(_choose_completion_counts())
    inp, out = _build_example(shapes)
    return inp if _example_is_good(inp, out) else None


def generate_input() -> np.ndarray:
    for _ in range(400):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid input")
