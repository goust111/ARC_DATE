from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
MARKER = 5
SMALL_COLORS = (1, 2, 3)
LARGE_COLORS = (4, 7, 9)
DISTRACTOR_COLOR = 6
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIRS8 = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0))


def _mask(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


AREA2_SHAPES = (
    _mask(["11"]),
    _mask(["1", "1"]),
)

AREA3_SHAPES = (
    _mask(["111"]),
    _mask(["1", "1", "1"]),
    _mask(["11", "10"]),
    _mask(["11", "01"]),
    _mask(["10", "11"]),
    _mask(["01", "11"]),
)

AREA4_SHAPES = (
    _mask(["1111"]),
    _mask(["1", "1", "1", "1"]),
    _mask(["11", "11"]),
    _mask(["111", "010"]),
    _mask(["010", "111"]),
    _mask(["10", "10", "11"]),
    _mask(["01", "01", "11"]),
    _mask(["11", "10", "10"]),
    _mask(["11", "01", "01"]),
)

AREA5_SHAPES = (
    _mask(["100", "100", "111"]),
    _mask(["11111"]),
    _mask(["1", "1", "1", "1", "1"]),
    _mask(["111", "010", "010"]),
    _mask(["010", "111", "010"]),
    _mask(["11", "11", "10"]),
)

PAIR_PATTERNS = (
    ((5, 2), (2, 3), (4, 2)),
    ((5, 2), (2, 3), (4, 3)),
)


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _scale_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    k = int(factor)
    return np.kron(mask.astype(int), np.ones((k, k), dtype=int)).astype(bool)


def _center_anchor(mask: np.ndarray) -> tuple[int, int]:
    h, w = mask.shape
    return (h - 1) // 2, (w - 1) // 2


def _component_dict(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict:
    arr = np.array(coords, dtype=int)
    r0, c0, r1, c1 = _bbox_from_coords(arr)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    mask = np.zeros((h, w), dtype=bool)
    local = arr - np.array([r0, c0])
    mask[local[:, 0], local[:, 1]] = True
    colors = {int(grid[r, c]) for r, c in coords}
    assert len(colors) == 1
    return {
        "coords": arr,
        "bbox": (r0, c0, r1, c1),
        "mask": mask,
        "color": int(next(iter(colors))),
        "area": int(len(coords)),
    }


def _find_components(grid: np.ndarray) -> list[dict]:
    seen = np.zeros(grid.shape, dtype=bool)
    comps: list[dict] = []
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == BG:
                continue
            color = int(grid[r, c])
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in DIRS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(_component_dict(grid, coords))
    return comps


def _extract_marker_and_objects(grid: np.ndarray) -> tuple[tuple[int, int], list[dict]]:
    comps = _find_components(grid)
    marker_comps = [comp for comp in comps if comp["color"] == MARKER and comp["area"] == 1]
    assert len(marker_comps) == 1
    marker = tuple(map(int, marker_comps[0]["coords"][0]))
    objects = [comp for comp in comps if comp is not marker_comps[0]]
    return marker, objects


def _candidate_matches(objects: list[dict]) -> list[tuple[int, int, int]]:
    matches: list[tuple[int, int, int]] = []
    for i, small in enumerate(objects):
        for j, big in enumerate(objects):
            if i == j or small["area"] >= big["area"]:
                continue
            for factor in (2, 3):
                scaled = _scale_mask(small["mask"], factor)
                if scaled.shape == big["mask"].shape and np.array_equal(scaled, big["mask"]):
                    matches.append((i, j, factor))
    return matches


def _paired_objects(grid: np.ndarray) -> tuple[tuple[int, int], list[dict], list[dict]]:
    marker, objects = _extract_marker_and_objects(grid)
    candidates = _candidate_matches(objects)

    by_small: dict[int, list[tuple[int, int]]] = {}
    by_big: dict[int, list[tuple[int, int]]] = {}
    for small_idx, big_idx, factor in candidates:
        by_small.setdefault(small_idx, []).append((big_idx, factor))
        by_big.setdefault(big_idx, []).append((small_idx, factor))

    pairs: list[dict] = []
    used_small: set[int] = set()
    used_big: set[int] = set()
    for small_idx, options in by_small.items():
        if len(options) != 1:
            continue
        big_idx, factor = options[0]
        if len(by_big.get(big_idx, [])) != 1:
            continue
        assert small_idx not in used_small and big_idx not in used_big
        used_small.add(small_idx)
        used_big.add(big_idx)
        pairs.append(
            {
                "small_idx": small_idx,
                "big_idx": big_idx,
                "factor": int(factor),
                "small": objects[small_idx],
                "big": objects[big_idx],
            }
        )

    leftovers = [
        objects[idx]
        for idx in range(len(objects))
        if idx not in used_small and idx not in used_big
    ]
    return marker, pairs, leftovers


def _paint_patch(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    assert 0 <= top and 0 <= left and top + h <= dst.shape[0] and left + w <= dst.shape[1]
    view = dst[top : top + h, left : left + w]
    mask = patch != BG
    view[mask] = patch[mask]


def _ordered_pairs(pairs: list[dict]) -> list[dict]:
    keys = [(pair["factor"], pair["big"]["area"]) for pair in pairs]
    assert len(keys) == len(set(keys))
    return sorted(pairs, key=lambda pair: (-pair["factor"], -pair["big"]["area"]))


def _stack_pairs(shape: tuple[int, int], marker: tuple[int, int], pairs: list[dict]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for pair in pairs:
        small_mask = pair["small"]["mask"]
        patch = np.where(small_mask, pair["big"]["color"], BG).astype(int)
        anchor_r, anchor_c = _center_anchor(small_mask)
        top = marker[0] - anchor_r
        left = marker[1] - anchor_c
        _paint_patch(out, patch, top, left)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match same-shape small and large objects across scales, recolor the small ones, and stack them on the gray marker.

    Input structure:
    - Black background contains several isolated small colored shapes, several isolated larger colored shapes, one gray marker pixel, and one unmatched distractor.
    - Each useful small shape has exactly one larger partner whose mask is the same shape scaled by 2 or 3, with no rotation or reflection.
    - Small shapes and large shapes use disjoint color sets, while the gray marker sits alone in empty background.

    Rule:
    - Split the non-black cells into 4-connected components and ignore color when testing shape similarity.
    - Pair each small component with the unique larger component whose bounding-box mask equals the small mask scaled by 2 or by 3; discard unmatched components as distractors.
    - Recolor each matched small shape with its matched large shape's color, sort pairs by scale factor from large to small and then by matched large area from large to small, and stamp the recolored small masks so their bounding-box center cell lands on the gray marker.
    - Draw earlier pairs first and later pairs on top; the output is otherwise black and the gray marker is covered.

    Key insight:
    - The useful geometry comes from the small masks, the useful colors come from the large masks, and the scale factor extracted during matching controls the layer order of the final centered stack.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    marker, pairs, _leftovers = _paired_objects(g)
    assert pairs
    ordered = _ordered_pairs(pairs)
    return _stack_pairs(g.shape, marker, ordered)


def _shape_pool(area: int, *, compact_only: bool = False) -> tuple[np.ndarray, ...]:
    pool = {
        2: AREA2_SHAPES,
        3: AREA3_SHAPES,
        4: AREA4_SHAPES,
        5: AREA5_SHAPES,
    }[area]
    if not compact_only:
        return pool
    return tuple(mask for mask in pool if max(mask.shape) <= 3)


def _pick_unique_masks(pattern: tuple[tuple[int, int], ...]) -> list[np.ndarray]:
    used: set[bytes] = set()
    picked: list[np.ndarray] = []
    for area, factor in pattern:
        pool = list(_shape_pool(area, compact_only=(factor == 3 and area >= 4)))
        random.shuffle(pool)
        chosen = None
        for mask in pool:
            key = mask.astype(np.uint8).tobytes() + bytes(mask.shape)
            if key not in used:
                chosen = mask
                used.add(key)
                break
        assert chosen is not None
        picked.append(chosen.copy())
    return picked


def _apply_color(mask: np.ndarray, color: int) -> np.ndarray:
    return np.where(mask, int(color), BG).astype(int)


def _can_place(occupied: np.ndarray, mask: np.ndarray, top: int, left: int) -> bool:
    h, w = occupied.shape
    ph, pw = mask.shape
    if top < 0 or left < 0 or top + ph > h or left + pw > w:
        return False
    coords = np.argwhere(mask)
    for dr, dc in coords:
        rr = top + int(dr)
        cc = left + int(dc)
        if occupied[rr, cc]:
            return False
        for nr in range(rr - 1, rr + 2):
            for nc in range(cc - 1, cc + 2):
                if 0 <= nr < h and 0 <= nc < w and occupied[nr, nc]:
                    return False
    return True


def _place_mask(occupied: np.ndarray, mask: np.ndarray, top: int, left: int) -> None:
    for dr, dc in np.argwhere(mask):
        occupied[top + int(dr), left + int(dc)] = True


def _random_position(occupied: np.ndarray, mask: np.ndarray) -> tuple[int, int] | None:
    ph, pw = mask.shape
    candidates = [
        (top, left)
        for top in range(occupied.shape[0] - ph + 1)
        for left in range(occupied.shape[1] - pw + 1)
        if _can_place(occupied, mask, top, left)
    ]
    return random.choice(candidates) if candidates else None


def _place_objects(
    height: int, width: int, objects: list[dict]
) -> tuple[np.ndarray, np.ndarray, list[dict]] | None:
    grid = np.full((height, width), BG, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)
    placed: list[dict] = []
    order = list(objects)
    random.shuffle(order)
    for obj in order:
        pos = _random_position(occupied, obj["mask"])
        if pos is None:
            return None
        top, left = pos
        patch = _apply_color(obj["mask"], obj["color"])
        _paint_patch(grid, patch, top, left)
        _place_mask(occupied, obj["mask"], top, left)
        placed.append({**obj, "top": top, "left": left})
    return grid, occupied, placed


def _stack_footprint_free(occupied: np.ndarray, marker: tuple[int, int], masks: list[np.ndarray]) -> bool:
    h, w = occupied.shape
    needed: set[tuple[int, int]] = set()
    for mask in masks:
        anchor_r, anchor_c = _center_anchor(mask)
        for dr, dc in np.argwhere(mask):
            rr = marker[0] - anchor_r + int(dr)
            cc = marker[1] - anchor_c + int(dc)
            if not (0 <= rr < h and 0 <= cc < w):
                return False
            needed.add((rr, cc))
    for rr, cc in needed:
        if occupied[rr, cc]:
            return False
        for nr in range(rr - 1, rr + 2):
            for nc in range(cc - 1, cc + 2):
                if 0 <= nr < h and 0 <= nc < w and occupied[nr, nc]:
                    return False
    return True


def _pick_marker(occupied: np.ndarray, masks: list[np.ndarray]) -> tuple[int, int] | None:
    candidates: list[tuple[int, int]] = []
    for r in range(occupied.shape[0]):
        for c in range(occupied.shape[1]):
            if occupied[r, c]:
                continue
            if _stack_footprint_free(occupied, (r, c), masks):
                candidates.append((r, c))
    return random.choice(candidates) if candidates else None


def _absolute_center_cell(obj: dict) -> tuple[int, int]:
    anchor_r, anchor_c = _center_anchor(obj["mask"])
    return obj["top"] + anchor_r, obj["left"] + anchor_c


def _mixed_layout(placed: list[dict]) -> bool:
    small_centers = [_absolute_center_cell(obj) for obj in placed if obj["role"] == "small"]
    large_centers = [_absolute_center_cell(obj) for obj in placed if obj["role"] == "large"]
    small_rows = [r for r, _ in small_centers]
    small_cols = [c for _, c in small_centers]
    large_rows = [r for r, _ in large_centers]
    large_cols = [c for _, c in large_centers]
    separated_rows = max(small_rows) < min(large_rows) or max(large_rows) < min(small_rows)
    separated_cols = max(small_cols) < min(large_cols) or max(large_cols) < min(small_cols)
    return not (separated_rows and separated_cols)


def _visible_colors(out: np.ndarray) -> set[int]:
    return set(map(int, np.unique(out))) - {BG}


def _validate_instance(grid: np.ndarray, expected_pairs: int) -> None:
    marker, pairs, leftovers = _paired_objects(grid)
    assert len(pairs) == expected_pairs
    assert len(leftovers) == 1
    scales = {pair["factor"] for pair in pairs}
    assert scales == {2, 3}
    assert all(pair["small"]["color"] in SMALL_COLORS for pair in pairs)
    assert all(pair["big"]["color"] in LARGE_COLORS for pair in pairs)
    assert leftovers[0]["color"] == DISTRACTOR_COLOR

    ordered = _ordered_pairs(pairs)
    out = main(grid)
    assert out.shape == grid.shape
    assert _visible_colors(out) == {pair["big"]["color"] for pair in pairs}
    assert MARKER not in _visible_colors(out)

    for idx in range(len(ordered) - 1):
        swapped = list(ordered)
        swapped[idx], swapped[idx + 1] = swapped[idx + 1], swapped[idx]
        assert not np.array_equal(out, _stack_pairs(grid.shape, marker, swapped))

    visible_counts = {
        color: int(np.count_nonzero(out == color))
        for color in {pair["big"]["color"] for pair in pairs}
    }
    assert all(count > 0 for count in visible_counts.values())


def _build_attempt() -> np.ndarray | None:
    height = random.randint(22, 28)
    width = random.randint(22, 28)
    pattern = random.choice(PAIR_PATTERNS)
    small_masks = _pick_unique_masks(pattern)

    small_colors = list(SMALL_COLORS)
    large_colors = list(LARGE_COLORS)
    random.shuffle(small_colors)
    random.shuffle(large_colors)

    objects: list[dict] = []
    small_masks_for_marker: list[np.ndarray] = []
    for idx, ((area, factor), small_mask) in enumerate(zip(pattern, small_masks)):
        big_mask = _scale_mask(small_mask, factor)
        small_color = small_colors[idx]
        large_color = large_colors[idx]
        objects.append(
            {
                "role": "small",
                "mask": small_mask,
                "color": small_color,
                "pair_id": idx,
                "area": area,
                "factor": factor,
            }
        )
        objects.append(
            {
                "role": "large",
                "mask": big_mask,
                "color": large_color,
                "pair_id": idx,
                "area": int(np.count_nonzero(big_mask)),
                "factor": factor,
            }
        )
        small_masks_for_marker.append(small_mask)

    distractor_large = random.random() < 0.5
    distractor_pool = list(AREA3_SHAPES)
    random.shuffle(distractor_pool)
    distractor_mask = distractor_pool[0].copy()
    if distractor_large:
        distractor_mask = _scale_mask(distractor_mask, random.choice((2, 3)))
    objects.append(
        {
            "role": "distractor",
            "mask": distractor_mask,
            "color": DISTRACTOR_COLOR,
            "pair_id": -1,
            "area": int(np.count_nonzero(distractor_mask)),
            "factor": 0,
        }
    )

    placed_result = _place_objects(height, width, objects)
    if placed_result is None:
        return None
    grid, occupied, placed = placed_result
    if not _mixed_layout(placed):
        return None

    marker = _pick_marker(occupied, small_masks_for_marker)
    if marker is None:
        return None
    grid[marker] = MARKER

    try:
        _validate_instance(grid, expected_pairs=3)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
