from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
PALETTE = (1, 2, 3, 4)
MAX_GEN_ATTEMPTS = 400
GRID_MIN = 18
GRID_MAX = 32
RECT_SIDE_RANGE = (4, 8)
GAP_RANGE = (2, 7)
PADDING = 1

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

ARROW_PATTERNS = {
    "up": {(1, 1), (2, 1), (3, 1), (0, 0), (0, 2)},
    "down": {(0, 1), (1, 1), (2, 1), (3, 0), (3, 2)},
    "left": {(1, 1), (1, 2), (1, 3), (0, 0), (2, 0)},
    "right": {(1, 0), (1, 1), (1, 2), (0, 3), (2, 3)},
}


def _components8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            q.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _detect_arrow(coords: list[tuple[int, int]]) -> dict | None:
    if len(coords) != 5:
        return None
    r0, c0, r1, c1 = _bbox(coords)
    norm = {(r - r0, c - c0) for r, c in coords}
    for direction, pattern in ARROW_PATTERNS.items():
        if norm == pattern:
            return {
                "direction": direction,
                "coords": sorted((int(r), int(c)) for r, c in coords),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
            }
    return None


def _detect_rectangle(coords: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4:
        return None
    if len(coords) != h * w:
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _final_arrow_coords(
    arrow_coords: list[tuple[int, int]],
    direction: str,
    rect_bbox: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    ar0, ac0, ar1, ac1 = _bbox(arrow_coords)
    rr0, rc0, rr1, rc1 = rect_bbox
    if direction == "right":
        steps = rc0 - ac0
    elif direction == "left":
        steps = ac1 - rc1
    elif direction == "down":
        steps = rr0 - ar0
    else:
        steps = ar1 - rr1

    assert steps > 0, "arrow must start outside the target block"
    dr, dc = DIRS[direction]
    shifted = [(r + dr * steps, c + dc * steps) for r, c in arrow_coords]
    assert all(rr0 <= r <= rr1 and rc0 <= c <= rc1 for r, c in shifted), "embedded arrow must fit fully inside the rectangle"
    return shifted


def main(grid: np.ndarray) -> np.ndarray:
    """
    Translate each 5-pixel arrow along its pointing direction until it embeds into
    the same-color solid rectangle, then carve the arrow-shaped hole out of the rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    colors = [int(c) for c in np.unique(g) if int(c) != BG]
    assert 1 <= len(colors) <= len(PALETTE)

    for color in colors:
        comps = _components8(g == color)
        assert len(comps) == 2, "each color must have one arrow and one rectangle"

        arrow = None
        rect = None
        for comp in comps:
            info = _detect_arrow(comp)
            if info is not None:
                assert arrow is None, "expected a unique arrow per color"
                arrow = info
            rect_bbox = _detect_rectangle(comp)
            if rect_bbox is not None:
                assert rect is None, "expected a unique rectangle per color"
                rect = rect_bbox

        assert arrow is not None and rect is not None, "missing arrow or rectangle"

        ar0, ac0, ar1, ac1 = arrow["bbox"]
        rr0, rc0, rr1, rc1 = rect
        direction = arrow["direction"]
        if direction == "right":
            assert ac1 < rc0
        elif direction == "left":
            assert ac0 > rc1
        elif direction == "down":
            assert ar1 < rr0
        else:
            assert ar0 > rr1

        final_coords = _final_arrow_coords(arrow["coords"], direction, rect)
        for r, c in arrow["coords"]:
            out[r, c] = BG
        for r, c in final_coords:
            out[r, c] = BG

    return out


def _pattern_array(direction: str, color: int) -> np.ndarray:
    pattern = ARROW_PATTERNS[direction]
    h = max(r for r, _ in pattern) + 1
    w = max(c for _, c in pattern) + 1
    arr = np.full((h, w), BG, dtype=int)
    for r, c in pattern:
        arr[r, c] = color
    return arr


def _expand_bbox(
    r0: int, c0: int, r1: int, c1: int, h: int, w: int, pad: int
) -> tuple[int, int, int, int]:
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _free_envelope(
    reserved: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int
) -> bool:
    h, w = reserved.shape
    er0, ec0, er1, ec1 = _expand_bbox(r0, c0, r1, c1, h, w, pad)
    return not bool(reserved[er0 : er1 + 1, ec0 : ec1 + 1].any())


def _mark_envelope(
    reserved: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int
) -> None:
    h, w = reserved.shape
    er0, ec0, er1, ec1 = _expand_bbox(r0, c0, r1, c1, h, w, pad)
    reserved[er0 : er1 + 1, ec0 : ec1 + 1] = True


def _place_case(grid: np.ndarray, reserved: np.ndarray, color: int, direction: str) -> dict | None:
    h, w = grid.shape
    rect_h = random.randint(*RECT_SIDE_RANGE)
    rect_w = random.randint(*RECT_SIDE_RANGE)
    gap = random.randint(*GAP_RANGE)
    arrow = _pattern_array(direction, color)
    ah, aw = arrow.shape

    candidates: list[dict] = []

    if direction == "right":
        env_h = rect_h
        env_w = aw + gap + rect_w
        if env_h > h or env_w > w:
            return None
        for er0 in range(h - env_h + 1):
            for ec0 in range(w - env_w + 1):
                if not _free_envelope(reserved, er0, ec0, er0 + env_h - 1, ec0 + env_w - 1, PADDING):
                    continue
                offset = random.randint(0, rect_h - ah)
                ar0, ac0 = er0 + offset, ec0
                rr0, rc0 = er0, ec0 + aw + gap
                candidates.append({"arrow_rc": (ar0, ac0), "rect_rc": (rr0, rc0), "rect_hw": (rect_h, rect_w)})
    elif direction == "left":
        env_h = rect_h
        env_w = rect_w + gap + aw
        if env_h > h or env_w > w:
            return None
        for er0 in range(h - env_h + 1):
            for ec0 in range(w - env_w + 1):
                if not _free_envelope(reserved, er0, ec0, er0 + env_h - 1, ec0 + env_w - 1, PADDING):
                    continue
                offset = random.randint(0, rect_h - ah)
                rr0, rc0 = er0, ec0
                ar0, ac0 = er0 + offset, ec0 + rect_w + gap
                candidates.append({"arrow_rc": (ar0, ac0), "rect_rc": (rr0, rc0), "rect_hw": (rect_h, rect_w)})
    elif direction == "down":
        env_h = ah + gap + rect_h
        env_w = rect_w
        if env_h > h or env_w > w:
            return None
        for er0 in range(h - env_h + 1):
            for ec0 in range(w - env_w + 1):
                if not _free_envelope(reserved, er0, ec0, er0 + env_h - 1, ec0 + env_w - 1, PADDING):
                    continue
                offset = random.randint(0, rect_w - aw)
                ar0, ac0 = er0, ec0 + offset
                rr0, rc0 = er0 + ah + gap, ec0
                candidates.append({"arrow_rc": (ar0, ac0), "rect_rc": (rr0, rc0), "rect_hw": (rect_h, rect_w)})
    else:
        env_h = rect_h + gap + ah
        env_w = rect_w
        if env_h > h or env_w > w:
            return None
        for er0 in range(h - env_h + 1):
            for ec0 in range(w - env_w + 1):
                if not _free_envelope(reserved, er0, ec0, er0 + env_h - 1, ec0 + env_w - 1, PADDING):
                    continue
                offset = random.randint(0, rect_w - aw)
                rr0, rc0 = er0, ec0
                ar0, ac0 = er0 + rect_h + gap, ec0 + offset
                candidates.append({"arrow_rc": (ar0, ac0), "rect_rc": (rr0, rc0), "rect_hw": (rect_h, rect_w)})

    if not candidates:
        return None

    choice = random.choice(candidates)
    ar0, ac0 = choice["arrow_rc"]
    rr0, rc0 = choice["rect_rc"]
    rect_h, rect_w = choice["rect_hw"]
    rr1, rc1 = rr0 + rect_h - 1, rc0 + rect_w - 1

    grid[rr0 : rr1 + 1, rc0 : rc1 + 1] = color
    grid[ar0 : ar0 + ah, ac0 : ac0 + aw] = np.maximum(grid[ar0 : ar0 + ah, ac0 : ac0 + aw], arrow)

    arrow_coords = [(ar0 + r, ac0 + c) for r, c in sorted(ARROW_PATTERNS[direction])]
    final_coords = _final_arrow_coords(arrow_coords, direction, (rr0, rc0, rr1, rc1))
    _mark_envelope(reserved, min(ar0, rr0), min(ac0, rc0), max(ar0 + ah - 1, rr1), max(ac0 + aw - 1, rc1), PADDING)

    return {
        "color": color,
        "direction": direction,
        "arrow_coords": arrow_coords,
        "rect_bbox": (rr0, rc0, rr1, rc1),
        "final_coords": final_coords,
    }


def _expected_output(inp: np.ndarray, placed: list[dict]) -> np.ndarray:
    out = inp.copy()
    for item in placed:
        for r, c in item["arrow_coords"]:
            out[r, c] = BG
        for r, c in item["final_coords"]:
            out[r, c] = BG
    return out


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(1, 4)
    min_side = max(GRID_MIN, 12 + 4 * count)
    h = random.randint(min_side, GRID_MAX)
    w = random.randint(min_side, GRID_MAX)

    grid = np.full((h, w), BG, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)
    colors = random.sample(PALETTE, count)
    placed: list[dict] = []

    for color in colors:
        success = None
        directions = list(DIRS)
        random.shuffle(directions)
        for _ in range(30):
            direction = random.choice(directions)
            success = _place_case(grid, reserved, color, direction)
            if success is not None:
                placed.append(success)
                break
        if success is None:
            return None

    expected = _expected_output(grid, placed)
    try:
        if np.array_equal(main(grid), expected):
            return grid
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid input")
