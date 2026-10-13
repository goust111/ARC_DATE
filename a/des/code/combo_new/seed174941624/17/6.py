import random
from typing import Iterable

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

SEED_COLOR_POOL = (BLUE, RED, YELLOW, MAGENTA, ORANGE, AZURE, MAROON)
RULE_OPTIONS = (
    ((-1, 0), (0, -1)),
    ((-1, 0), (0, 1)),
    ((1, 0), (0, -1)),
    ((1, 0), (0, 1)),
)

TOP_HEIGHT_RANGE = (6, 8)
BOTTOM_HEIGHT_RANGE = (10, 18)
HALF_WIDTH_RANGE = (8, 13)
PALETTE_SIZE = 4
SEED_COUNT_OPTIONS = (3, 4)
GEN_RETRIES = 400
PLACEMENT_RETRIES = 200


def _find_dividers(grid: np.ndarray) -> tuple[int, int]:
    split_rows = np.flatnonzero(np.all(grid == GREY, axis=1))
    split_cols = np.flatnonzero(np.all(grid == GREY, axis=0))
    assert len(split_rows) == 1, "expected exactly one full grey horizontal divider"
    assert len(split_cols) == 1, "expected exactly one full grey vertical divider"
    return int(split_rows[0]), int(split_cols[0])


def _read_palette(top: np.ndarray) -> dict[int, int]:
    dots: list[tuple[int, int]] = []
    for r, c in np.argwhere((top != BG) & (top != GREY) & (top != GREEN)):
        dots.append((int(c), int(top[r, c])))
    assert len(dots) >= 2, "expected a visible top-row color strip"

    dots.sort()
    colors = [color for _c, color in dots]
    assert len(colors) == len(set(colors)), "palette colors must be unique"
    return {color: idx + 1 for idx, color in enumerate(colors)}


def _read_rule(quadrant: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    pts = np.argwhere(quadrant == GREEN)
    assert pts.shape[0] == 3, "each legend half must contain one 3-pixel green rule"

    distances = np.abs(pts[:, None, :] - pts[None, :, :]).sum(axis=2).sum(axis=1)
    center = pts[int(np.argmin(distances))]

    directions: list[tuple[int, int]] = []
    for pt in pts:
        dr = int(pt[0] - center[0])
        dc = int(pt[1] - center[1])
        if dr == 0 and dc == 0:
            continue
        assert abs(dr) + abs(dc) == 1, "rule offsets must be cardinal neighbors of the center"
        directions.append((dr, dc))

    assert len(directions) == 2 and len(set(directions)) == 2
    return directions[0], directions[1]


def _relative_growth(
    directions: Iterable[tuple[int, int]], steps: int
) -> set[tuple[int, int]]:
    cells = {(0, 0)}
    for _ in range(steps):
        grown = set(cells)
        for r, c in cells:
            for dr, dc in directions:
                grown.add((r + dr, c + dc))
        cells = grown
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each bottom-half seed by the side-specific green rule and by the step count encoded in the legend's color strip.

    Input structure:
    - The background is black.
    - A full grey horizontal line splits the grid into a top legend and a bottom canvas, and a full grey vertical line splits both areas into left and right halves.
    - The top legend contains one 3-pixel green rule example in each half plus a strip of distinct non-green colors whose left-to-right order defines the seed-step ranking.
    - The bottom canvas contains isolated one-pixel seeds whose colors all appear in that strip.

    Rule:
    - Read the top color strip from left to right: the first color grows for 1 step, the second for 2 steps, and so on.
    - In each top-half rule example, the green pixel closest to the other two is the center; the two center-to-neighbor offsets are the directions copied on every growth step.
    - Each bottom seed uses the rule from its own side of the grey vertical divider and repeatedly copies all of its current pixels one cell along both rule directions for its assigned number of steps.
    - Growth stays below the horizontal divider, stays inside the seed's own half, and never overwrites existing non-black pixels.

    Key insight:
    - The legend has two independent controls that must be combined: the green geometry says how each half grows, while the ordered color strip says how long each seed keeps growing.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = map(int, g.shape)

    split_r, split_c = _find_dividers(g)
    step_map = _read_palette(g[:split_r])
    left_rule = _read_rule(g[:split_r, :split_c])
    right_rule = _read_rule(g[:split_r, split_c + 1 :])

    seeds: list[tuple[int, int, int]] = []
    for r, c in np.argwhere(g[split_r + 1 :, :] != BG):
        rr = int(r + split_r + 1)
        cc = int(c)
        color = int(g[rr, cc])
        if color in (GREY, GREEN):
            continue
        seeds.append((rr, cc, color))

    out = g.copy()
    seeds.sort()
    for r0, c0, color in seeds:
        assert color in step_map, "seed color must appear in the top color strip"
        steps = step_map[color]
        directions = left_rule if c0 < split_c else right_rule
        in_left = c0 < split_c

        cells = {(r0, c0)}
        for _ in range(steps):
            grown = set(cells)
            for r, c in cells:
                for dr, dc in directions:
                    nr = r + dr
                    nc = c + dc
                    if not (split_r < nr < h):
                        continue
                    if in_left:
                        if not (0 <= nc < split_c):
                            continue
                    else:
                        if not (split_c < nc < w):
                            continue
                    if out[nr, nc] != BG and (nr, nc) not in cells:
                        continue
                    grown.add((nr, nc))
            cells = grown

        for r, c in cells:
            out[r, c] = color

    return out


def _sample_dimensions() -> tuple[int, int, int, int]:
    for _ in range(100):
        top_h = random.randint(*TOP_HEIGHT_RANGE)
        bottom_h = random.randint(*BOTTOM_HEIGHT_RANGE)
        left_w = random.randint(*HALF_WIDTH_RANGE)
        right_w = random.randint(*HALF_WIDTH_RANGE)
        if top_h + 1 + bottom_h <= 30 and left_w + 1 + right_w <= 30:
            return top_h, bottom_h, left_w, right_w
    raise RuntimeError("failed to sample ARC-sized dimensions")


def _sample_spaced_columns(start: int, stop: int, count: int) -> list[int] | None:
    candidates = list(range(start, stop + 1))
    for _ in range(100):
        cols = sorted(random.sample(candidates, count))
        if all(b - a >= 2 for a, b in zip(cols, cols[1:])):
            return cols
    return None


def _cells_are_separate(
    candidate: set[tuple[int, int]], occupied: set[tuple[int, int]]
) -> bool:
    for r, c in candidate:
        for rr, cc in occupied:
            if abs(r - rr) <= 1 and abs(c - cc) <= 1:
                return False
    return True


def _translate(
    shape: set[tuple[int, int]], base_r: int, base_c: int
) -> set[tuple[int, int]]:
    return {(base_r + dr, base_c + dc) for dr, dc in shape}


def _place_rule(
    grid: np.ndarray,
    top_h: int,
    left_w: int,
    palette_r: int,
    split_c: int,
    directions: tuple[tuple[int, int], tuple[int, int]],
    right: bool,
) -> None:
    row_min = max(palette_r + 2, 1)
    row_max = top_h - 2
    assert row_min <= row_max
    center_r = random.randint(row_min, row_max)

    if right:
        center_c = random.randint(split_c + 2, grid.shape[1] - 2)
    else:
        center_c = random.randint(1, left_w - 2)

    grid[center_r, center_c] = GREEN
    for dr, dc in directions:
        grid[center_r + dr, center_c + dc] = GREEN


def _gen_attempt() -> np.ndarray | None:
    top_h, bottom_h, left_w, right_w = _sample_dimensions()
    h = top_h + 1 + bottom_h
    w = left_w + 1 + right_w
    split_r = top_h
    split_c = left_w

    grid = np.full((h, w), BG, dtype=int)
    grid[split_r, :] = GREY
    grid[:, split_c] = GREY

    palette_colors = random.sample(SEED_COLOR_POOL, PALETTE_SIZE)
    palette_order = random.sample(palette_colors, len(palette_colors))
    palette_r = random.randint(0, 1)
    left_cols = _sample_spaced_columns(1, split_c - 1, PALETTE_SIZE // 2)
    right_cols = _sample_spaced_columns(split_c + 1, w - 2, PALETTE_SIZE // 2)
    if left_cols is None or right_cols is None:
        return None
    palette_cols = left_cols + right_cols
    for c, color in zip(sorted(palette_cols), palette_order):
        grid[palette_r, c] = color

    left_rule, right_rule = random.sample(RULE_OPTIONS, 2)
    _place_rule(grid, top_h, left_w, palette_r, split_c, left_rule, right=False)
    _place_rule(grid, top_h, left_w, palette_r, split_c, right_rule, right=True)

    seed_count = random.choice(SEED_COUNT_OPTIONS)
    seed_colors = random.sample(palette_colors, seed_count)
    random.shuffle(seed_colors)
    left_count = random.randint(1, seed_count - 1)
    left_colors = seed_colors[:left_count]
    right_colors = seed_colors[left_count:]

    step_map = {color: idx + 1 for idx, color in enumerate(palette_order)}
    specs = []
    for color in left_colors:
        specs.append(
            {
                "color": color,
                "side": "left",
                "steps": step_map[color],
                "shape": _relative_growth(left_rule, step_map[color]),
            }
        )
    for color in right_colors:
        specs.append(
            {
                "color": color,
                "side": "right",
                "steps": step_map[color],
                "shape": _relative_growth(right_rule, step_map[color]),
            }
        )
    specs.sort(key=lambda spec: (-len(spec["shape"]), -spec["steps"], spec["side"]))

    occupied: set[tuple[int, int]] = set()
    placements = []
    for spec in specs:
        if spec["side"] == "left":
            r_min, r_max = split_r + 2, h - 2
            c_min, c_max = 1, split_c - 2
        else:
            r_min, r_max = split_r + 2, h - 2
            c_min, c_max = split_c + 2, w - 2

        placed = None
        for _ in range(PLACEMENT_RETRIES):
            seed_r = random.randint(r_min, r_max)
            seed_c = random.randint(c_min, c_max)
            cells = _translate(spec["shape"], seed_r, seed_c)
            if any(
                r < r_min or r > r_max or c < c_min or c > c_max for r, c in cells
            ):
                continue
            if not _cells_are_separate(cells, occupied):
                continue
            placed = (seed_r, seed_c, cells)
            break
        if placed is None:
            return None

        placements.append((spec["color"], spec["side"], placed[0], placed[1], placed[2]))
        occupied.update(placed[2])

    for color, _side, seed_r, seed_c, _cells in placements:
        grid[seed_r, seed_c] = color

    # The solver should accept every generated input without clipping or ambiguity.
    _ = main(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
