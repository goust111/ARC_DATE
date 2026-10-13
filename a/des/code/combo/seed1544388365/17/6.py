import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
MARKER_TO_STEP = {
    RED: 1,
    BLUE: -1,
}


def _get_bg(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.flat))
    return counts.most_common(1)[0][0]


def _rect_cycle(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
    for r in range(r0 + 1, r1 + 1):
        coords.append((r, c1))
    for c in range(c1 - 1, c0 - 1, -1):
        coords.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        coords.append((r, c0))
    return coords


def _scan_to_non_bg(
    grid: np.ndarray,
    bg: int,
    r: int,
    c: int,
    dr: int,
    dc: int,
) -> tuple[int, int] | None:
    h, w = grid.shape
    rr, cc = r + dr, c + dc
    while 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == bg:
        rr += dr
        cc += dc
    if 0 <= rr < h and 0 <= cc < w:
        return rr, cc
    return None


def _ring_from_marker(
    grid: np.ndarray,
    bg: int,
    mr: int,
    mc: int,
) -> dict | None:
    marker = int(grid[mr, mc])
    if marker not in MARKER_TO_STEP:
        return None

    up = _scan_to_non_bg(grid, bg, mr, mc, -1, 0)
    down = _scan_to_non_bg(grid, bg, mr, mc, 1, 0)
    left = _scan_to_non_bg(grid, bg, mr, mc, 0, -1)
    right = _scan_to_non_bg(grid, bg, mr, mc, 0, 1)
    if None in (up, down, left, right):
        return None

    r0 = int(up[0])
    r1 = int(down[0])
    c0 = int(left[1])
    c1 = int(right[1])
    if not (r0 < mr < r1 and c0 < mc < c1):
        return None
    if mr - r0 != r1 - mr or mc - c0 != c1 - mc:
        return None

    cycle = _rect_cycle(r0, c0, r1, c1)
    if any(int(grid[r, c]) == bg for r, c in cycle):
        return None

    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if (r, c) == (mr, mc):
                continue
            if int(grid[r, c]) != bg:
                return None

    return {
        "bbox": (r0, c0, r1, c1),
        "marker": marker,
        "center": (mr, mc),
        "cycle": cycle,
    }


def _find_rings(grid: np.ndarray, bg: int) -> list[dict]:
    seen: set[tuple[int, int, int, int]] = set()
    rings: list[dict] = []
    marker_mask = (grid == RED) | (grid == BLUE)
    for mr, mc in np.argwhere(marker_mask):
        ring = _ring_from_marker(grid, bg, int(mr), int(mc))
        if ring is None:
            continue
        bbox = ring["bbox"]
        if bbox in seen:
            continue
        seen.add(bbox)
        rings.append(ring)
    return rings


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move colored cargos one step around each rectangular conveyor loop.

    Each loop is a 1-pixel rectangular border made of black path cells, with some
    path cells replaced by colored cargos. The single red/blue marker at the
    center of the enclosed interior sets the direction:
    - red  -> shift cargos one step clockwise
    - blue -> shift cargos one step counterclockwise
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()

    for ring in _find_rings(g, bg):
        cycle = ring["cycle"]
        step = MARKER_TO_STEP[int(ring["marker"])]
        cargos = [
            (idx, int(g[r, c]))
            for idx, (r, c) in enumerate(cycle)
            if int(g[r, c]) != BLACK
        ]

        for r, c in cycle:
            out[r, c] = BLACK
        for idx, color in cargos:
            nr, nc = cycle[(idx + step) % len(cycle)]
            out[nr, nc] = color

    return out


# --------------- generator ---------------

GEN_BG_CHOICES = [3, 4, 5, 6, 7, 8, 9]
GEN_MIN_H = 15
GEN_MAX_H = 28
GEN_MIN_W = 15
GEN_MAX_W = 28
GEN_RING_COUNT_MIN = 1
GEN_RING_COUNT_MAX = 4
GEN_RING_PAD = 2
GEN_MARGIN = 1
GEN_MIN_SIDE = 5
GEN_MAX_SIDE = 13
GEN_RETRIES = 400


def _odd_sizes(limit: int) -> list[int]:
    return [n for n in range(GEN_MIN_SIDE, limit + 1) if n % 2 == 1]


def _boxes_separated(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    pad: int,
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _sample_non_adjacent_indices(length: int, count: int) -> list[int] | None:
    order = list(range(length))
    random.shuffle(order)
    chosen: list[int] = []
    for idx in order:
        if all((idx - prev) % length not in (0, 1, length - 1) for prev in chosen):
            chosen.append(idx)
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _place_one_ring(
    h: int,
    w: int,
    placed_boxes: list[tuple[int, int, int, int]],
    bg: int,
) -> dict | None:
    max_h = min(GEN_MAX_SIDE, h - 2 * GEN_MARGIN)
    max_w = min(GEN_MAX_SIDE, w - 2 * GEN_MARGIN)
    height_choices = _odd_sizes(max_h)
    width_choices = _odd_sizes(max_w)
    if not height_choices or not width_choices:
        return None

    cargo_colors = [color for color in range(10) if color not in {BLACK, BLUE, RED, bg}]
    for _ in range(80):
        ring_h = random.choice(height_choices)
        ring_w = random.choice(width_choices)
        r0 = random.randint(GEN_MARGIN, h - ring_h - GEN_MARGIN)
        c0 = random.randint(GEN_MARGIN, w - ring_w - GEN_MARGIN)
        bbox = (r0, c0, r0 + ring_h - 1, c0 + ring_w - 1)
        if any(not _boxes_separated(bbox, other, GEN_RING_PAD) for other in placed_boxes):
            continue

        cycle = _rect_cycle(*bbox)
        max_cargos = min(6, len(cycle) // 2)
        cargo_count = random.randint(1, max_cargos)
        cargo_indices = _sample_non_adjacent_indices(len(cycle), cargo_count)
        if cargo_indices is None:
            continue

        cargos = {idx: random.choice(cargo_colors) for idx in cargo_indices}
        mr = (bbox[0] + bbox[2]) // 2
        mc = (bbox[1] + bbox[3]) // 2
        marker = random.choice([RED, BLUE])
        return {
            "bbox": bbox,
            "marker": marker,
            "center": (mr, mc),
            "cycle": cycle,
            "cargos": cargos,
        }
    return None


def _render_input(h: int, w: int, bg: int, rings: list[dict]) -> np.ndarray:
    grid = np.full((h, w), bg, dtype=int)
    for ring in rings:
        r0, c0, r1, c1 = ring["bbox"]
        grid[r0, c0 : c1 + 1] = BLACK
        grid[r1, c0 : c1 + 1] = BLACK
        grid[r0 : r1 + 1, c0] = BLACK
        grid[r0 : r1 + 1, c1] = BLACK
        for idx, color in ring["cargos"].items():
            rr, cc = ring["cycle"][idx]
            grid[rr, cc] = color
        mr, mc = ring["center"]
        grid[mr, mc] = ring["marker"]
    return grid


def _render_expected(h: int, w: int, bg: int, rings: list[dict]) -> np.ndarray:
    out = np.full((h, w), bg, dtype=int)
    for ring in rings:
        r0, c0, r1, c1 = ring["bbox"]
        out[r0, c0 : c1 + 1] = BLACK
        out[r1, c0 : c1 + 1] = BLACK
        out[r0 : r1 + 1, c0] = BLACK
        out[r0 : r1 + 1, c1] = BLACK
        step = MARKER_TO_STEP[ring["marker"]]
        for idx, color in ring["cargos"].items():
            rr, cc = ring["cycle"][(idx + step) % len(ring["cycle"])]
            out[rr, cc] = color
        mr, mc = ring["center"]
        out[mr, mc] = ring["marker"]
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GEN_MIN_H, GEN_MAX_H)
    w = random.randint(GEN_MIN_W, GEN_MAX_W)
    bg = random.choice(GEN_BG_CHOICES)
    ring_goal = random.randint(GEN_RING_COUNT_MIN, GEN_RING_COUNT_MAX)

    placed_boxes: list[tuple[int, int, int, int]] = []
    rings: list[dict] = []
    for _ in range(ring_goal):
        ring = _place_one_ring(h, w, placed_boxes, bg)
        if ring is None:
            break
        placed_boxes.append(ring["bbox"])
        rings.append(ring)

    if not rings:
        return None

    grid = _render_input(h, w, bg, rings)
    expected = _render_expected(h, w, bg, rings)

    counts = Counter(map(int, grid.flat))
    if counts[bg] <= max(count for color, count in counts.items() if color != bg):
        return None

    found = _find_rings(grid, bg)
    if len(found) != len(rings):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
