from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS = ("up", "right", "down", "left")
INNER = {
    "up": (1, 2),
    "right": (2, 3),
    "down": (3, 2),
    "left": (2, 1),
}
OUTER = {
    "up": (0, 2),
    "right": (2, 4),
    "down": (4, 2),
    "left": (2, 0),
}
CLOCK_CELLS = {
    (0, 2),
    (1, 2),
    (2, 0),
    (2, 1),
    (2, 2),
    (2, 3),
    (2, 4),
    (3, 2),
    (4, 2),
}
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_MAX_ATTEMPTS = 300
GEN_CANVAS = 30
GEN_CLOCK_COUNT = (1, 4)
GEN_MARGIN = 3
GEN_MIN_GAP = 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return 4-connected components of a boolean mask."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _parse_clock(g: np.ndarray, coords: list[tuple[int, int]]) -> dict:
    """Validate one connected component as a single 5x5 clock and extract its state."""
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0 = min(rows)
    c0 = min(cols)
    r1 = max(rows)
    c1 = max(cols)
    assert r1 - r0 == 4 and c1 - c0 == 4, "clock bbox must be 5x5"
    rel = {(r - r0, c - c0) for r, c in coords}
    assert rel == CLOCK_CELLS, "clock component must be the 5x5 cross"

    patch = g[r0 : r1 + 1, c0 : c1 + 1]
    assert int(patch[2, 2]) == GREY, "clock center must be grey"
    for rr in range(5):
        for cc in range(5):
            value = int(patch[rr, cc])
            if (rr, cc) in CLOCK_CELLS:
                assert value != BG, "all cross cells must be non-black"
            else:
                assert value == BG, "only the cross cells may be non-black"

    color_counts = Counter(
        int(patch[rr, cc])
        for rr, cc in CLOCK_CELLS
        if (rr, cc) != (2, 2) and int(patch[rr, cc]) != GREY
    )
    assert color_counts, "clock must contain colored pixels"

    pointer_dirs: list[str] = []
    dir_colors: dict[str, int] = {}
    occupied: dict[str, bool] = {}

    for direction in DIRS:
        inner_r, inner_c = INNER[direction]
        outer_r, outer_c = OUTER[direction]
        inner_color = int(patch[inner_r, inner_c])
        outer_color = int(patch[outer_r, outer_c])

        present = [color for color in (inner_color, outer_color) if color != GREY]
        if present:
            assert len(set(present)) == 1, "each direction may expose at most one tick color"
            dir_colors[direction] = present[0]
            occupied[direction] = True
        else:
            occupied[direction] = False

        if inner_color != GREY and color_counts[inner_color] >= 2:
            pointer_dirs.append(direction)

    assert len(pointer_dirs) == 1, "pointer must be the unique duplicated inner color"
    pointer_dir = pointer_dirs[0]

    start = DIRS.index(pointer_dir)
    target_dir = None
    for step in range(1, 5):
        direction = DIRS[(start + step) % 4]
        if occupied[direction]:
            target_dir = direction
            break
    assert target_dir is not None and target_dir != pointer_dir, "clock needs another occupied direction"

    return {
        "r0": r0,
        "c0": c0,
        "pointer_dir": pointer_dir,
        "target_dir": target_dir,
        "target_color": dir_colors[target_dir],
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate each clock hand to the next occupied direction clockwise.

    Input structure:
    - The background is black.
    - Each object is a 5x5 clock: a grey center with grey cross arms reaching two cells up, right, down, and left.
    - Some arm cells are recolored into distinct occupied ticks, and one inner arm cell is a colored pointer that matches one occupied tick color.

    Rule:
    - Find every non-black 4-connected clock object and identify the pointer as the only inner-arm color that is duplicated elsewhere on the same clock.
    - Read the pointer's current direction among up, right, down, and left, then scan clockwise starting from the next direction.
    - Move the pointer to the first direction that already contains a colored tick, painting the inner cell there with that tick's color and restoring the old pointer cell to grey.

    Key insight:
    - The duplicated inner color identifies the pointer, while the next clockwise non-grey direction determines both where the pointer lands and which color it takes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    comps = _components(g != BG)
    assert comps, "expected at least one clock"

    for coords in comps:
        clock = _parse_clock(g, coords)
        pr, pc = INNER[clock["pointer_dir"]]
        tr, tc = INNER[clock["target_dir"]]
        out[clock["r0"] + pr, clock["c0"] + pc] = GREY
        out[clock["r0"] + tr, clock["c0"] + tc] = int(clock["target_color"])

    return out


def _clockwise_between(src: str, dst: str) -> list[str]:
    """Directions encountered strictly between src and dst while moving clockwise."""
    out: list[str] = []
    idx = DIRS.index(src)
    while True:
        idx = (idx + 1) % 4
        direction = DIRS[idx]
        if direction == dst:
            return out
        out.append(direction)


def _build_clock() -> np.ndarray:
    """Create one random valid clock patch."""
    src = random.choice(DIRS)
    offset = random.randint(1, 3)
    src_idx = DIRS.index(src)
    dst = DIRS[(src_idx + offset) % 4]
    target_on_inner = bool(random.getrandbits(1))

    tick_pos: dict[str, str | None] = {direction: None for direction in DIRS}
    tick_pos[dst] = "inner" if target_on_inner else "outer"

    for blocked in _clockwise_between(src, dst):
        tick_pos[blocked] = None

    remaining = [
        direction
        for direction in DIRS
        if direction not in {src, dst} and direction not in _clockwise_between(src, dst)
    ]
    for direction in remaining:
        tick_pos[direction] = random.choice((None, "inner", "outer"))

    outer_candidates = [direction for direction, pos in tick_pos.items() if pos == "outer" and direction != src]
    if outer_candidates and random.random() < 0.5:
        pointer_match_dir = random.choice(outer_candidates)
    else:
        tick_pos[src] = "outer"
        pointer_match_dir = src

    occupied_dirs = [direction for direction, pos in tick_pos.items() if pos is not None]
    colors = random.sample(PALETTE, len(occupied_dirs))
    dir_color = {direction: color for direction, color in zip(occupied_dirs, colors)}

    patch = np.full((5, 5), BG, dtype=int)
    for rr, cc in CLOCK_CELLS:
        patch[rr, cc] = GREY

    for direction, pos in tick_pos.items():
        if pos is None:
            continue
        rr, cc = INNER[direction] if pos == "inner" else OUTER[direction]
        patch[rr, cc] = dir_color[direction]

    pr, pc = INNER[src]
    patch[pr, pc] = dir_color[pointer_match_dir]
    return patch


def _separated(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> bool:
    """Require at least one black row/column between two 5x5 clocks."""
    return (
        box_a[2] + GEN_MIN_GAP < box_b[0]
        or box_b[2] + GEN_MIN_GAP < box_a[0]
        or box_a[3] + GEN_MIN_GAP < box_b[1]
        or box_b[3] + GEN_MIN_GAP < box_a[1]
    )


def _place_clocks(count: int) -> tuple[np.ndarray, list[tuple[int, int, int, int]]] | None:
    """Place several random clocks on a temporary 30x30 canvas."""
    grid = np.full((GEN_CANVAS, GEN_CANVAS), BG, dtype=int)
    boxes: list[tuple[int, int, int, int]] = []

    for _ in range(count):
        patch = _build_clock()
        candidates = [
            (r0, c0)
            for r0 in range(GEN_CANVAS - 4)
            for c0 in range(GEN_CANVAS - 4)
            if all(_separated((r0, c0, r0 + 4, c0 + 4), box) for box in boxes)
        ]
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        grid[r0 : r0 + 5, c0 : c0 + 5] = patch
        boxes.append((r0, c0, r0 + 4, c0 + 4))

    return grid, boxes


def _crop(grid: np.ndarray) -> np.ndarray:
    """Crop to the occupied bbox with small random black margins."""
    coords = np.argwhere(grid != BG)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0 = max(0, int(r0) - random.randint(0, GEN_MARGIN))
    c0 = max(0, int(c0) - random.randint(0, GEN_MARGIN))
    r1 = min(grid.shape[0] - 1, int(r1) + random.randint(0, GEN_MARGIN))
    c1 = min(grid.shape[1] - 1, int(c1) + random.randint(0, GEN_MARGIN))
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _gen_attempt() -> np.ndarray | None:
    """Generate one valid input and verify it through the solver."""
    count = random.randint(*GEN_CLOCK_COUNT)
    placed = _place_clocks(count)
    if placed is None:
        return None

    grid, _boxes = placed
    cropped = _crop(grid)
    if max(cropped.shape) > 30:
        return None

    try:
        out = main(cropped)
    except AssertionError:
        return None

    if np.array_equal(out, cropped):
        return None
    return cropped


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid clock input")
