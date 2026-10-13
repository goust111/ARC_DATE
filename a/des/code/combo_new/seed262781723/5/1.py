from __future__ import annotations

import random

import numpy as np


BG = 0
FRAME_COLOR = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
GRID_H_RANGE = (12, 30)
GRID_W_RANGE = (12, 30)
FRAME_COUNT_OPTIONS = (1, 2, 3, 4)
FRAME_SIZE_RANGE = (3, 8)
SEED_DIST_RANGE = (1, 6)
UNMATCHED_SEED_PROB = 0.6
UNMATCHED_SEED_MAX = 4
FRAME_TRIES = 200
UNMATCHED_TRIES = 80
GEN_RETRIES = 300


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_for_color(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _frame_from_component(grid: np.ndarray, coords: np.ndarray, color: int, bg: int) -> dict | None:
    rows = coords[:, 0]
    cols = coords[:, 1]
    r0 = int(rows.min())
    r1 = int(rows.max())
    c0 = int(cols.min())
    c1 = int(cols.max())
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    perimeter = 2 * h + 2 * w - 4
    if int(coords.shape[0]) != perimeter - 1:
        return None

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    comp_mask = sub == color
    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True

    if np.any(comp_mask & ~border_mask):
        return None
    if not np.all(sub[1:-1, 1:-1] == bg):
        return None

    missing = np.argwhere(border_mask & ~comp_mask)
    if missing.shape[0] != 1:
        return None

    mr = int(missing[0, 0])
    mc = int(missing[0, 1])
    abs_gap = (r0 + mr, c0 + mc)
    sides: list[str] = []
    if mr == 0:
        sides.append("up")
    if mr == h - 1:
        sides.append("down")
    if mc == 0:
        sides.append("left")
    if mc == w - 1:
        sides.append("right")
    if len(sides) != 1:
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "gap": abs_gap,
        "direction": sides[0],
        "color": int(color),
    }


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        for coords in _components_for_color(grid == color):
            frame = _frame_from_component(grid, coords, color, bg)
            if frame is not None:
                frames.append(frame)
    frames.sort(key=lambda frame: frame["bbox"])
    return frames


def _find_seed(grid: np.ndarray, gap: tuple[int, int], direction: str, bg: int) -> tuple[int, int, int]:
    dr, dc = DIRS[direction]
    h, w = grid.shape
    rr = int(gap[0] + dr)
    cc = int(gap[1] + dc)
    while 0 <= rr < h and 0 <= cc < w:
        color = int(grid[rr, cc])
        if color != bg:
            return rr, cc, color
        rr += dr
        cc += dc
    raise AssertionError("window ray must hit a seed")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each hollow windowed rectangle with the color of the seed visible through its gap.

    Input structure:
    - The background is black.
    - One or more gray one-pixel-thick rectangular frames appear on the grid.
    - Each frame has exactly one missing border pixel on one side, and its interior starts as background.
    - Outside that missing pixel, along the outward row or column, sits a colored single-pixel seed.

    Rule:
    - Detect every rectangular frame whose border is complete except for one side gap.
    - Use the gap position to determine whether the frame opens up, down, left, or right.
    - Cast a ray outward from the gap and take the first non-background pixel on that ray as the frame's seed.
    - Fill the full interior of the frame with the seed color, keep the frame border unchanged, and erase the matched seed to background.
    - Leave any unmatched colored seed pixels unchanged.

    Key insight:
    - The missing border pixel is not damage; it is a directional window that uniquely tells you where to search for the color source.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _most_common_color(g)
    frames = _find_frames(g, bg)
    assert frames, "expected at least one windowed frame"

    out = g.copy()
    matched_seeds: set[tuple[int, int]] = set()
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        seed_r, seed_c, seed_color = _find_seed(g, frame["gap"], frame["direction"], bg)
        out[r0 + 1 : r1, c0 + 1 : c1] = seed_color
        matched_seeds.add((seed_r, seed_c))

    for seed_r, seed_c in matched_seeds:
        out[seed_r, seed_c] = bg
    return out


def _overlaps(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> bool:
    return not (
        box_a[2] < box_b[0]
        or box_b[2] < box_a[0]
        or box_a[3] < box_b[1]
        or box_b[3] < box_a[1]
    )


def _expand_box(box: tuple[int, int, int, int], pad: int, h: int, w: int) -> tuple[int, int, int, int]:
    return (
        max(0, box[0] - pad),
        max(0, box[1] - pad),
        min(h - 1, box[2] + pad),
        min(w - 1, box[3] + pad),
    )


def _frame_cells(r0: int, c0: int, r1: int, c1: int, gap: tuple[int, int]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            if on_border and (r, c) != gap:
                cells.append((r, c))
    return cells


def _sample_frame_placement(
    height: int,
    width: int,
    zones: list[tuple[int, int, int, int]],
) -> dict | None:
    direction = random.choice(tuple(DIRS))
    frame_h = random.randint(*FRAME_SIZE_RANGE)
    frame_w = random.randint(*FRAME_SIZE_RANGE)
    r0 = random.randint(0, height - frame_h)
    c0 = random.randint(0, width - frame_w)
    r1 = r0 + frame_h - 1
    c1 = c0 + frame_w - 1

    if direction in ("up", "down"):
        gap = (r0 if direction == "up" else r1, random.randint(c0 + 1, c1 - 1))
        max_dist = r0 if direction == "up" else height - 1 - r1
    else:
        gap = (random.randint(r0 + 1, r1 - 1), c0 if direction == "left" else c1)
        max_dist = c0 if direction == "left" else width - 1 - c1
    if max_dist < 1:
        return None

    seed_dist = random.randint(SEED_DIST_RANGE[0], min(SEED_DIST_RANGE[1], max_dist))
    dr, dc = DIRS[direction]
    seed = (gap[0] + dr * seed_dist, gap[1] + dc * seed_dist)

    zone = _expand_box(
        (min(r0, seed[0]), min(c0, seed[1]), max(r1, seed[0]), max(c1, seed[1])),
        1,
        height,
        width,
    )
    if any(_overlaps(zone, other) for other in zones):
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "gap": gap,
        "direction": direction,
        "seed": seed,
        "color": random.choice(SEED_COLORS),
        "zone": zone,
    }


def _place_unmatched_seeds(
    grid: np.ndarray,
    occupied: np.ndarray,
    zones: list[tuple[int, int, int, int]],
) -> None:
    if random.random() >= UNMATCHED_SEED_PROB:
        return

    count = random.randint(1, UNMATCHED_SEED_MAX)
    h, w = grid.shape
    for _ in range(count):
        for _ in range(UNMATCHED_TRIES):
            r = random.randint(0, h - 1)
            c = random.randint(0, w - 1)
            if occupied[r, c]:
                continue
            zone = _expand_box((r, c, r, c), 1, h, w)
            if any(_overlaps(zone, other) for other in zones):
                continue
            grid[r, c] = random.choice(SEED_COLORS)
            occupied[r, c] = True
            zones.append(zone)
            break


def _expected_output(grid: np.ndarray, placements: list[dict]) -> np.ndarray:
    out = grid.copy()
    for placement in placements:
        r0, c0, r1, c1 = placement["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = placement["color"]
        seed_r, seed_c = placement["seed"]
        out[seed_r, seed_c] = BG
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    grid = np.full((height, width), BG, dtype=int)

    max_frames = min(max(FRAME_COUNT_OPTIONS), max(1, (height * width) // 90))
    frame_count = random.choice([count for count in FRAME_COUNT_OPTIONS if count <= max_frames])

    zones: list[tuple[int, int, int, int]] = []
    placements: list[dict] = []
    for _ in range(frame_count):
        for _ in range(FRAME_TRIES):
            placement = _sample_frame_placement(height, width, zones)
            if placement is not None:
                placements.append(placement)
                zones.append(placement["zone"])
                break
        else:
            return None

    occupied = np.zeros((height, width), dtype=bool)
    for placement in placements:
        r0, c0, r1, c1 = placement["bbox"]
        gap = placement["gap"]
        for r, c in _frame_cells(r0, c0, r1, c1, gap):
            grid[r, c] = FRAME_COLOR
            occupied[r, c] = True
        seed_r, seed_c = placement["seed"]
        grid[seed_r, seed_c] = placement["color"]
        occupied[seed_r, seed_c] = True

    _place_unmatched_seeds(grid, occupied, zones)

    expected = _expected_output(grid, placements)
    try:
        got = main(grid)
    except Exception:
        return None
    if not np.array_equal(got, expected):
        return None
    if np.array_equal(got, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid window-frame ARC input")
