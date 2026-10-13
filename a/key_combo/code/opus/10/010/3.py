from __future__ import annotations

import random
from collections import deque
from functools import lru_cache

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREY = 5
TEAL = 8

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _component_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_coords(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
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
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _compress_runs(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    assert mask.ndim == 2 and mask.any()

    keep_rows = [0]
    for r in range(1, mask.shape[0]):
        if not np.array_equal(mask[r], mask[r - 1]):
            keep_rows.append(r)
    row_compressed = mask[keep_rows, :]

    keep_cols = [0]
    for c in range(1, row_compressed.shape[1]):
        if not np.array_equal(row_compressed[:, c], row_compressed[:, c - 1]):
            keep_cols.append(c)
    return row_compressed[:, keep_cols]


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in np.asarray(mask, dtype=int))


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()
    for k in range(4):
        rot = np.rot90(mask, k)
        for variant in (rot, np.fliplr(rot)):
            key = _mask_key(variant)
            if key not in seen:
                seen.add(key)
                out.append(variant.copy())
    return out


def _has_no_hole(mask: np.ndarray) -> bool:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    bg = ~mask
    seen = np.zeros_like(bg, dtype=bool)
    q = deque()
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            if bg[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    while q:
        rr, cc = q.popleft()
        for dr, dc in DIR4:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return not bool((bg & ~seen).any())


def _adjacent_runs_unique(mask: np.ndarray) -> bool:
    for r in range(mask.shape[0] - 1):
        if np.array_equal(mask[r], mask[r + 1]):
            return False
    for c in range(mask.shape[1] - 1):
        if np.array_equal(mask[:, c], mask[:, c + 1]):
            return False
    return True


def _shape_ok(mask: np.ndarray, *, require_unique_runs: bool) -> bool:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not mask.any():
        return False
    area = int(mask.sum())
    h, w = mask.shape
    if area < 4 or area > 8:
        return False
    if h < 2 or w < 2 or h > 4 or w > 4:
        return False
    if area == h * w:
        return False
    if len(_components_mask(mask)) != 1:
        return False
    if not _has_no_hole(mask):
        return False
    if require_unique_runs and not _adjacent_runs_unique(mask):
        return False
    return True


def _mask_from_cells(cells: frozenset[tuple[int, int]]) -> np.ndarray:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, c0 = min(rows), min(cols)
    h = max(rows) - r0 + 1
    w = max(cols) - c0 + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask


def _normalize_cells(cells: set[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


@lru_cache(maxsize=None)
def _polyominoes(area: int) -> tuple[frozenset[tuple[int, int]], ...]:
    if area == 1:
        return (frozenset({(0, 0)}),)
    prev = _polyominoes(area - 1)
    out: set[frozenset[tuple[int, int]]] = set()
    for shape in prev:
        base = set(shape)
        frontier: set[tuple[int, int]] = set()
        for r, c in base:
            for dr, dc in DIR4:
                cand = (r + dr, c + dc)
                if cand not in base:
                    frontier.add(cand)
        for cell in frontier:
            out.add(_normalize_cells(base | {cell}))
    return tuple(sorted(out, key=lambda s: (len(s), sorted(s))))


@lru_cache(maxsize=1)
def _shape_pool() -> tuple[np.ndarray, ...]:
    seen: set[tuple[tuple[int, ...], ...]] = set()
    pool: list[np.ndarray] = []
    for area in range(4, 9):
        for shape in _polyominoes(area):
            mask = _mask_from_cells(shape)
            if not _shape_ok(mask, require_unique_runs=True):
                continue
            key = _mask_key(mask)
            if key not in seen:
                seen.add(key)
                pool.append(mask)
    return tuple(pool)


def _proto_signature_set(mask: np.ndarray) -> set[tuple[tuple[int, ...], ...]]:
    return {_mask_key(v) for v in _dihedral_variants(mask)}


def _stretch_amount(raw_mask: np.ndarray, compressed_mask: np.ndarray) -> int:
    # Count redundant row/column runs removed by compression. This is the only
    # reading that matches the task's "stretch by duplicated rows/cols" idea
    # without collapsing into a raw-size shortcut.
    return int(raw_mask.shape[0] - compressed_mask.shape[0] + raw_mask.shape[1] - compressed_mask.shape[1])


def _classify_candidates(grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    proto_comps = _components_mask(grid == GREY)
    red_comps = _components_mask(grid == RED)
    assert len(proto_comps) == 1
    assert len(red_comps) >= 1

    proto_mask = _component_mask(proto_comps[0])
    proto_keys = _proto_signature_set(proto_mask)

    info: list[dict] = []
    for coords in red_comps:
        raw_mask = _component_mask(coords)
        compressed = _compress_runs(raw_mask)
        matched = _mask_key(compressed) in proto_keys
        info.append(
            {
                "coords": coords,
                "matched": matched,
                "stretch": _stretch_amount(raw_mask, compressed) if matched else None,
                "bbox_area": int(raw_mask.shape[0] * raw_mask.shape[1]),
            }
        )
    return proto_comps[0], info


def main(grid: np.ndarray) -> np.ndarray:
    """Compress every red object to test prototype equivalence, then recolor matches by stretch rank.

    Input structure:
    - A black background contains exactly one grey 4-connected prototype object and
      several red 4-connected candidate objects separated from one another.
    - Matching candidates are stretched versions of the prototype under rotation or
      reflection; nonmatching candidates are unrelated red shapes.

    Rule:
    - Crop each red candidate to its bbox, collapse every run of identical adjacent
      rows to one row, then collapse every run of identical adjacent columns to one
      column.
    - If the compressed candidate equals the grey prototype under any rotation or
      mirror reflection, keep it; otherwise erase it to black.
    - Among kept candidates, compute the compression loss as the number of removed
      duplicate rows plus removed duplicate columns. Recolor the largest loss blue,
      the smallest loss teal, and leave any other kept candidates red. The grey
      prototype stays unchanged.

    Key insight:
    - The visible candidates are stretched along duplicated row/column runs, so the
      true prototype match and the hidden stretch ranking only appear after run
      compression.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    _proto, candidates = _classify_candidates(g)
    matched = [item for item in candidates if item["matched"]]
    assert matched

    max_stretch = max(int(item["stretch"]) for item in matched)
    min_stretch = min(int(item["stretch"]) for item in matched)

    out = g.copy()
    for item in candidates:
        coords = item["coords"]
        rr = coords[:, 0]
        cc = coords[:, 1]
        if not item["matched"]:
            out[rr, cc] = BG
            continue
        color = RED
        if int(item["stretch"]) == max_stretch:
            color = BLUE
        if int(item["stretch"]) == min_stretch:
            color = TEAL
        out[rr, cc] = color
    return out


def _random_prototype() -> np.ndarray:
    pool = list(_shape_pool())
    random.shuffle(pool)
    for mask in pool:
        if len(_dihedral_variants(mask)) >= 2:
            return mask.copy()
    raise RuntimeError("prototype pool is empty")


def _sample_stretched(mask: np.ndarray, used_stretch: set[int]) -> tuple[np.ndarray, int]:
    h, w = mask.shape
    row_ids = list(range(h))
    col_ids = list(range(w))
    for _ in range(300):
        row_extra = [0] * h
        col_extra = [0] * w

        row_count = random.randint(0, min(2, h))
        col_count = random.randint(0, min(2, w))
        if row_count == 0 and col_count == 0:
            if random.random() < 0.5:
                row_count = 1
            else:
                col_count = 1

        for idx in random.sample(row_ids, row_count):
            row_extra[idx] = random.randint(1, 3)
        for idx in random.sample(col_ids, col_count):
            col_extra[idx] = random.randint(1, 3)

        row_factors = [1 + x for x in row_extra]
        col_factors = [1 + x for x in col_extra]
        if sum(row_extra) + sum(col_extra) == 0:
            continue
        if sum(row_factors) > 7 or sum(col_factors) > 7:
            continue

        stretched = np.repeat(np.repeat(mask, row_factors, axis=0), col_factors, axis=1)
        compressed = _compress_runs(stretched)
        stretch = _stretch_amount(stretched, compressed)
        if stretch <= 0 or stretch in used_stretch:
            continue
        if _mask_key(compressed) != _mask_key(mask):
            continue
        return stretched, stretch
    raise RuntimeError("failed to sample stretched match")


def _random_distractor(
    proto_keys: set[tuple[tuple[int, ...], ...]],
    size_window: tuple[int, int] | None = None,
) -> np.ndarray:
    pool = list(_shape_pool())
    random.shuffle(pool)
    fallback: np.ndarray | None = None
    for base in pool:
        if _proto_signature_set(base) & proto_keys:
            continue
        candidates = [base]
        if random.random() < 0.7:
            try:
                stretched, _ = _sample_stretched(base, set())
                if _mask_key(_compress_runs(stretched)) not in proto_keys:
                    candidates.insert(0, stretched)
            except RuntimeError:
                pass
        for cand in candidates:
            size = int(cand.sum())
            if size_window is not None and not (size_window[0] <= size <= size_window[1]):
                if fallback is None:
                    fallback = cand.copy()
                continue
            return cand.copy()
        if fallback is None:
            fallback = base.copy()
    if fallback is not None:
        return fallback
    raise RuntimeError("failed to create distractor")


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _place_mask(grid: np.ndarray, mask: np.ndarray, color: int, r0: int, c0: int) -> np.ndarray:
    coords = np.argwhere(mask)
    grid[r0 + coords[:, 0], c0 + coords[:, 1]] = int(color)
    return coords + np.array([r0, c0], dtype=int)


def _place_objects(shape: tuple[int, int], objects: list[dict]) -> tuple[np.ndarray, list[dict]] | None:
    h, w = shape
    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    placed: list[dict] = []

    order = list(objects)
    random.shuffle(order)
    order.sort(key=lambda obj: obj["mask"].shape[0] * obj["mask"].shape[1], reverse=True)

    for obj in order:
        mask = obj["mask"]
        oh, ow = mask.shape
        if oh > h or ow > w:
            return None
        blocked = _dilate8(occupied)
        candidates: list[tuple[int, int]] = []
        for r0 in range(h - oh + 1):
            for c0 in range(w - ow + 1):
                if not bool((blocked[r0:r0 + oh, c0:c0 + ow] & mask).any()):
                    candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        coords = _place_mask(grid, mask, obj["color"], r0, c0)
        occupied[coords[:, 0], coords[:, 1]] = True

        item = dict(obj)
        item["coords"] = coords
        placed.append(item)
    return grid, placed


def _expected_output(inp: np.ndarray, placed: list[dict]) -> np.ndarray:
    out = np.zeros_like(inp)
    proto = [obj for obj in placed if obj["kind"] == "prototype"]
    assert len(proto) == 1
    out[proto[0]["coords"][:, 0], proto[0]["coords"][:, 1]] = GREY

    matched = [obj for obj in placed if obj["kind"] == "match"]
    max_stretch = max(int(obj["stretch"]) for obj in matched)
    min_stretch = min(int(obj["stretch"]) for obj in matched)
    for obj in placed:
        if obj["kind"] == "distractor":
            continue
        if obj["kind"] == "prototype":
            continue
        color = RED
        if int(obj["stretch"]) == max_stretch:
            color = BLUE
        if int(obj["stretch"]) == min_stretch:
            color = TEAL
        out[obj["coords"][:, 0], obj["coords"][:, 1]] = color
    return out


def _records_ok(placed: list[dict]) -> bool:
    matched = [obj for obj in placed if obj["kind"] == "match"]
    distractors = [obj for obj in placed if obj["kind"] == "distractor"]
    if len(matched) < 2 or len(distractors) < 1:
        return False

    stretches = [int(obj["stretch"]) for obj in matched]
    if len(stretches) != len(set(stretches)):
        return False

    # Ensure at least one distractor has a visible size comparable to a real
    # match, so the solver cannot skip the compression/matching stage and sort
    # candidates only by raw size.
    match_sizes = [int(obj["mask"].sum()) for obj in matched]
    distractor_sizes = [int(obj["mask"].sum()) for obj in distractors]
    lo = min(match_sizes)
    hi = max(match_sizes)
    if not any(lo <= size <= hi for size in distractor_sizes):
        return False

    return True


def _oracle_checks(inp: np.ndarray, placed: list[dict]) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False

    if not _records_ok(placed):
        return False

    proto_comps = _components_mask(inp == GREY)
    red_comps = _components_mask(inp == RED)
    if len(proto_comps) != 1 or not (4 <= len(red_comps) <= 6):
        return False

    proto_mask = _component_mask(proto_comps[0])
    if not _shape_ok(proto_mask, require_unique_runs=True):
        return False

    solver_proto, solver_info = _classify_candidates(inp)
    if solver_proto.shape[0] != proto_comps[0].shape[0]:
        return False

    placed_red = sorted(
        [obj for obj in placed if obj["kind"] != "prototype"],
        key=lambda obj: tuple(_bbox_coords(obj["coords"])),
    )
    solver_red = sorted(
        solver_info,
        key=lambda item: tuple(_bbox_coords(item["coords"])),
    )
    if len(placed_red) != len(solver_red):
        return False
    for obj, item in zip(placed_red, solver_red):
        want_match = obj["kind"] == "match"
        if bool(item["matched"]) != want_match:
            return False
        if want_match and int(item["stretch"]) != int(obj["stretch"]):
            return False

    expected = _expected_output(inp, placed)
    got = main(inp)
    if not np.array_equal(got, expected):
        return False

    colors = set(map(int, np.unique(got)))
    if BLUE not in colors or TEAL not in colors:
        return False
    if any(obj["kind"] == "match" for obj in placed) and len([obj for obj in placed if obj["kind"] == "match"]) >= 3:
        if RED not in colors:
            return False

    return True


def _build_scene() -> tuple[np.ndarray, list[dict]] | None:
    grid_h = random.randint(17, 22)
    grid_w = random.randint(17, 22)

    proto = _random_prototype()
    proto_keys = _proto_signature_set(proto)

    candidate_count = random.randint(4, 6)
    max_matches = min(4, candidate_count - 1)
    match_count = random.choice([n for n in (2, 3, 4) if n <= max_matches])
    distractor_count = candidate_count - match_count

    objects: list[dict] = [{"kind": "prototype", "mask": proto, "color": GREY}]
    used_stretch: set[int] = set()

    proto_variants = _dihedral_variants(proto)
    for _ in range(match_count):
        base = random.choice(proto_variants)
        stretched, stretch = _sample_stretched(base, used_stretch)
        used_stretch.add(stretch)
        objects.append(
            {
                "kind": "match",
                "mask": stretched,
                "color": RED,
                "stretch": stretch,
            }
        )

    match_size_window = (
        min(int(obj["mask"].sum()) for obj in objects if obj["kind"] == "match"),
        max(int(obj["mask"].sum()) for obj in objects if obj["kind"] == "match"),
    )
    for _ in range(distractor_count):
        objects.append(
            {
                "kind": "distractor",
                "mask": _random_distractor(proto_keys, match_size_window),
                "color": RED,
            }
        )

    placed_result = _place_objects((grid_h, grid_w), objects)
    if placed_result is None:
        return None
    return placed_result


def generate_input() -> np.ndarray:
    for _ in range(400):
        built = _build_scene()
        if built is None:
            continue
        inp, placed = built
        if _oracle_checks(inp, placed):
            return inp
    raise RuntimeError("failed to generate valid puzzle instance")
