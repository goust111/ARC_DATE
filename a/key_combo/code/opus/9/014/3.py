from __future__ import annotations

import random
from collections import deque

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    MAROON = 9


PANEL_SIZE = 7
DIVIDER_COLOR = Color.BLUE
OBJECT_COLOR = Color.GREEN
BEND_TO_COLOR = {
    0: Color.GREEN,
    1: Color.BLUE,
    2: Color.YELLOW,
    3: Color.RED,
    4: Color.MAROON,
}
ZONE_TOP_LEFTS = [(0, 0), (0, 4), (4, 0), (4, 4)]


def _mask_from_rows(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


def _transform_mask(mask: np.ndarray, rot_k: int, flip_lr: bool) -> np.ndarray:
    out = np.fliplr(mask) if flip_lr else mask.copy()
    return np.rot90(out, k=int(rot_k))


TEMPLATES = {
    0: {
        "left": _mask_from_rows(["110", "100", "000"]),
        "right": _mask_from_rows(["001", "011", "000"]),
    },
    1: {
        "left": _mask_from_rows(["100", "100", "100"]),
        "right": _mask_from_rows(["111", "000", "000"]),
    },
    2: {
        "left": _mask_from_rows(["111", "000", "000"]),
        "right": _mask_from_rows(["110", "010", "000"]),
    },
    3: {
        "left": _mask_from_rows(["011", "001", "001"]),
        "right": _mask_from_rows(["000", "100", "111"]),
    },
    4: {
        "left": _mask_from_rows(["111", "100", "000"]),
        "right": _mask_from_rows(["000", "001", "111"]),
    },
}

GENERATOR_BEND_SETS = {
    3: [(0, 1, 4), (0, 2, 3), (1, 2, 4), (0, 3, 4), (1, 3, 4)],
    4: [(0, 1, 2, 4), (0, 1, 3, 4), (0, 2, 3, 4), (1, 2, 3, 4), (0, 1, 2, 3)],
}
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(coords)
    components.sort(key=lambda comp: min(comp))
    return components


def _component_mask(coords: list[tuple[int, int]]) -> np.ndarray:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        out[r - r0, c - c0] = True
    return out


def _count_bends_from_mask(mask: np.ndarray) -> int:
    bends = 0
    for r in range(mask.shape[0] - 1):
        for c in range(mask.shape[1] - 1):
            if int(mask[r : r + 2, c : c + 2].sum()) == 3:
                bends += 1
    return bends


def _count_component_bends(coords: list[tuple[int, int]]) -> int:
    return _count_bends_from_mask(_component_mask(coords))


def _recolor_overlay_mask(mask: np.ndarray) -> np.ndarray:
    out = np.zeros(mask.shape, dtype=int)
    for coords in _find_components(mask):
        color = BEND_TO_COLOR[_count_component_bends(coords)]
        for r, c in coords:
            out[r, c] = int(color)
    return out


def _find_divider_col(grid: np.ndarray) -> int:
    h, w = grid.shape
    candidates = [
        c
        for c in range(w)
        if np.all(grid[:, c] == DIVIDER_COLOR) and c > 0 and c < w - 1 and 2 * c + 1 == w
    ]
    assert len(candidates) == 1, "expected exactly one full-height centered divider column"
    return int(candidates[0])


def _split_panels(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    divider = _find_divider_col(grid)
    left = np.asarray(grid[:, :divider], dtype=int)
    right = np.asarray(grid[:, divider + 1 :], dtype=int)
    assert left.shape == right.shape
    return left, right


def _overlay_mask_from_panels(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return (left == OBJECT_COLOR) | (right == OBJECT_COLOR)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay the two separator-split panels, then recolor each merged component by its bend count.

    Input structure:
    - A black grid is split into two equal panels by one full-height blue divider column.
    - Each panel contains several small green 4-connected shapes on black.
    - Left and right panel coordinates are aligned cell-by-cell across the divider.

    Rule:
    - Remove the blue divider and overlay the two panels with a transparent OR: any aligned green pixel stays green.
    - Find every green 4-connected component in the merged 7x7 overlay.
    - For each component, count 2x2 subwindows that contain exactly three green pixels; this is its bend count.
    - Recolor bends 1, 2, 3, 4 to blue, yellow, red, maroon respectively, while bend-0 components stay green.

    Key insight:
    - The decisive bend count is an emergent property of the merged overlay, not of either panel in isolation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    left, right = _split_panels(g)
    overlay = _overlay_mask_from_panels(left, right)
    return _recolor_overlay_mask(overlay)


def _stamp_zone(dst: np.ndarray, zone_mask: np.ndarray, top: int, left: int) -> None:
    h, w = zone_mask.shape
    dst[top : top + h, left : left + w][zone_mask] = OBJECT_COLOR


def _build_candidate_panels() -> tuple[np.ndarray, np.ndarray]:
    left = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=int)
    right = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=int)

    zone_count = random.choice((3, 4))
    active_zone_ids = random.sample(range(len(ZONE_TOP_LEFTS)), zone_count)
    bend_set = list(random.choice(GENERATOR_BEND_SETS[zone_count]))
    random.shuffle(bend_set)

    for zone_id, bend_count in zip(active_zone_ids, bend_set):
        top, left_col = ZONE_TOP_LEFTS[zone_id]
        rot_k = random.randrange(4)
        flip_lr = bool(random.getrandbits(1))
        left_zone = _transform_mask(TEMPLATES[bend_count]["left"], rot_k, flip_lr)
        right_zone = _transform_mask(TEMPLATES[bend_count]["right"], rot_k, flip_lr)
        _stamp_zone(left, left_zone, top, left_col)
        _stamp_zone(right, right_zone, top, left_col)

    return left, right


def _compose_input(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    h, w = left.shape
    out = np.zeros((h, w * 2 + 1), dtype=int)
    out[:, :w] = left
    out[:, w] = DIVIDER_COLOR
    out[:, w + 1 :] = right
    return out


def _component_bends_by_id(mask: np.ndarray) -> tuple[np.ndarray, dict[int, int], list[list[tuple[int, int]]]]:
    comps = _find_components(mask)
    ids = np.zeros(mask.shape, dtype=int)
    bend_by_id: dict[int, int] = {}
    for idx, coords in enumerate(comps, start=1):
        for r, c in coords:
            ids[r, c] = idx
        bend_by_id[idx] = _count_component_bends(coords)
    return ids, bend_by_id, comps


def _validate_candidate(inp: np.ndarray) -> bool:
    try:
        left, right = _split_panels(inp)
    except AssertionError:
        return False

    if left.shape != (PANEL_SIZE, PANEL_SIZE) or right.shape != (PANEL_SIZE, PANEL_SIZE):
        return False

    for panel in (left, right):
        if set(np.unique(panel)) - {Color.BLACK, OBJECT_COLOR}:
            return False
        comps = _find_components(panel == OBJECT_COLOR)
        if not (3 <= len(comps) <= 5):
            return False
        if any(not (3 <= len(comp) <= 6) for comp in comps):
            return False

    overlay = _overlay_mask_from_panels(left, right)
    overlay_ids, overlay_bends, overlay_comps = _component_bends_by_id(overlay)
    if len(overlay_comps) < 3:
        return False
    distinct_bends = {overlay_bends[idx] for idx in overlay_bends}
    if len(distinct_bends) < 3 or any(b not in BEND_TO_COLOR for b in distinct_bends):
        return False

    left_ids, left_bends, _ = _component_bends_by_id(left == OBJECT_COLOR)
    right_ids, right_bends, _ = _component_bends_by_id(right == OBJECT_COLOR)

    emergent = False
    merged = False
    for comp_id, final_bend in overlay_bends.items():
        rows, cols = np.where(overlay_ids == comp_id)
        contributing_left = {int(left_ids[r, c]) for r, c in zip(rows, cols) if left_ids[r, c] != 0}
        contributing_right = {int(right_ids[r, c]) for r, c in zip(rows, cols) if right_ids[r, c] != 0}
        if contributing_left and contributing_right:
            merged = True
            source_bends = {left_bends[idx] for idx in contributing_left} | {right_bends[idx] for idx in contributing_right}
            if final_bend not in source_bends:
                emergent = True

    if not merged or not emergent:
        return False

    out = main(inp)
    expected = _recolor_overlay_mask(overlay)
    if not np.array_equal(out, expected):
        return False

    if np.array_equal(out, _recolor_overlay_mask(left == OBJECT_COLOR)):
        return False
    if np.array_equal(out, _recolor_overlay_mask(right == OBJECT_COLOR)):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        left, right = _build_candidate_panels()
        inp = _compose_input(left, right)
        if _validate_candidate(inp):
            return inp
    raise RuntimeError("failed to generate a valid overlay-bend ARC input")
