import random
from collections import deque

import numpy as np


BG = 0
YELLOW = 4
GREY = 5
FRAME_COLORS = (1, 2, 3, 6, 7, 8, 9)

# Legend layout used by the generator.
LEGEND_TOP = 1
LEGEND_LEFT = 1
LEGEND_ROW_HEIGHT = 2
LEGEND_ROW_GAP = 1
SWATCH_H = 2
SWATCH_W = 2
DOT_SLOTS = (5, 6, 7)
LEGEND_RIGHT = max(DOT_SLOTS)

# Generator bounds.
MIN_INTERIOR = 3
MAX_INTERIOR = 6
GEN_RETRIES = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _same_color_components(grid: np.ndarray, color: int):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_frame_component(grid: np.ndarray, color: int, coords) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if len(coords) != 2 * (h + w) - 4:
        return False
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.all(patch[border] == color) and np.all(patch[1:-1, 1:-1] != color))


def _extract_legend(grid: np.ndarray) -> dict[int, int]:
    colors = [int(c) for c in np.unique(grid) if int(c) not in (BG, YELLOW, GREY)]
    swatches = []
    for color in colors:
        for coords in _same_color_components(grid, color):
            if _is_frame_component(grid, color, coords):
                continue
            r0, c0, r1, c1 = _bbox(coords)
            if len(coords) == SWATCH_H * SWATCH_W and (r1 - r0 + 1, c1 - c0 + 1) == (SWATCH_H, SWATCH_W):
                if r0 <= 12 and c0 <= 3:
                    swatches.append((r0, color, c1))
    swatches.sort()
    assert 3 <= len(swatches) <= 4, "expected 3-4 legend rows"

    mapping = {}
    h, w = grid.shape
    for r0, color, c1 in swatches:
        band = grid[r0 : r0 + SWATCH_H, c1 + 1 : min(w, c1 + 6)]
        mapping[color] = int(np.count_nonzero(band == YELLOW))
    return mapping


def _find_frames(grid: np.ndarray):
    frames = []
    colors = [int(c) for c in np.unique(grid) if int(c) not in (BG, YELLOW, GREY)]
    for color in colors:
        for coords in _same_color_components(grid, color):
            if _is_frame_component(grid, color, coords):
                frames.append({"color": color, "bbox": _bbox(coords)})
    frames.sort(key=lambda item: item["bbox"])
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each framed grey pattern by the angle encoded for that frame color in the legend.

    Input structure:
    - The background is Black.
    - The top-left corner contains a small color-to-rotation legend with 3-4 rows; each row has a colored swatch and 0-3 Yellow dots.
    - Elsewhere, one-pixel-thick colored rectangular frames enclose square interior canvases containing Grey pixels on Black.

    Rule:
    - Read the legend as a mapping from frame color to a clockwise rotation count: the number of Yellow dots is 0, 1, 2, or 3 quarter-turns.
    - For every colored rectangular frame, extract its interior canvas and rotate that entire interior clockwise by the count assigned to the frame color.
    - Leave the legend, the frame borders, and all non-interior pixels unchanged.

    Key insight:
    - The frame color is only a lookup key into the top-left legend; the Yellow-dot count determines how the Grey pattern inside the matching frame must be rotated.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    legend = _extract_legend(g)

    for frame in _find_frames(g):
        color = frame["color"]
        if color not in legend:
            continue
        r0, c0, r1, c1 = frame["bbox"]
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        turns = legend[color] % 4
        rotated = np.rot90(interior, k=-turns) if turns else interior
        assert rotated.shape == interior.shape, "frame interior must stay rectangular after rotation"
        out[r0 + 1 : r1, c0 + 1 : c1] = rotated

    return out


def _legend_height(row_count: int) -> int:
    return LEGEND_TOP + (row_count - 1) * (LEGEND_ROW_HEIGHT + LEGEND_ROW_GAP) + LEGEND_ROW_HEIGHT


def _draw_legend(grid: np.ndarray, legend_rows, rng: random.Random) -> None:
    for idx, (color, turns) in enumerate(legend_rows):
        r0 = LEGEND_TOP + idx * (LEGEND_ROW_HEIGHT + LEGEND_ROW_GAP)
        c0 = LEGEND_LEFT
        grid[r0 : r0 + SWATCH_H, c0 : c0 + SWATCH_W] = color
        if turns:
            chosen = sorted(rng.sample(DOT_SLOTS, turns))
            for c in chosen:
                grid[r0, c] = YELLOW


def _informative(mask: np.ndarray) -> bool:
    if not mask.any() or np.all(mask):
        return False
    for k in (1, 2, 3):
        if np.array_equal(mask, np.rot90(mask, k=-k)):
            return False
    return True


def _random_pattern(size: int, rng: random.Random) -> np.ndarray:
    for _ in range(200):
        target = rng.randint(max(2, size - 1), size * size - 2)
        start = (rng.randrange(size), rng.randrange(size))
        cells = {start}
        frontier = set()
        for nr, nc in _neighbors4(*start):
            if 0 <= nr < size and 0 <= nc < size:
                frontier.add((nr, nc))
        while frontier and len(cells) < target:
            cell = rng.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            for nr, nc in _neighbors4(*cell):
                if 0 <= nr < size and 0 <= nc < size and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        mask = np.zeros((size, size), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if _informative(mask):
            return mask
    raise RuntimeError("failed to generate a non-symmetric pattern")


def _can_place(occupied: np.ndarray, r0: int, c0: int, size: int) -> bool:
    h, w = occupied.shape
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h, r0 + size + 1)
    cc1 = min(w, c0 + size + 1)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _mark_occupied(occupied: np.ndarray, r0: int, c0: int, size: int) -> None:
    h, w = occupied.shape
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h, r0 + size + 1)
    cc1 = min(w, c0 + size + 1)
    occupied[rr0:rr1, cc0:cc1] = True


def _draw_frame(grid: np.ndarray, r0: int, c0: int, color: int, pattern: np.ndarray) -> None:
    size = pattern.shape[0]
    outer = size + 2
    grid[r0, c0 : c0 + outer] = color
    grid[r0 + outer - 1, c0 : c0 + outer] = color
    grid[r0 : r0 + outer, c0] = color
    grid[r0 : r0 + outer, c0 + outer - 1] = color
    grid[r0 + 1 : r0 + 1 + size, c0 + 1 : c0 + 1 + size][pattern] = GREY


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    row_count = rng.choice((3, 4))
    legend_colors = rng.sample(FRAME_COLORS, row_count)
    legend_turns = rng.sample((0, 1, 2, 3), row_count)
    legend_rows = list(zip(legend_colors, legend_turns))

    frame_colors = legend_colors[:]
    rng.shuffle(frame_colors)
    extra_count = rng.randint(0, 2)
    for _ in range(extra_count):
        frame_colors.append(rng.choice(legend_colors))
    n_frames = len(frame_colors)

    min_side = 18
    if n_frames >= 5:
        min_side = 22
    elif n_frames == 4:
        min_side = 20
    legend_h = _legend_height(row_count)
    min_h = max(min_side, legend_h + 2)
    min_w = max(min_side, LEGEND_RIGHT + 3)
    h = rng.randint(min_h, 30)
    w = rng.randint(min_w, 30)

    grid = np.full((h, w), BG, dtype=int)
    _draw_legend(grid, legend_rows, rng)

    occupied = np.zeros((h, w), dtype=bool)
    occupied[: legend_h + 1, : LEGEND_RIGHT + 2] = True

    frame_specs = []
    for color in frame_colors:
        interior = rng.randint(MIN_INTERIOR, MAX_INTERIOR)
        frame_specs.append({"color": color, "pattern": _random_pattern(interior, rng)})
    rng.shuffle(frame_specs)

    for spec in frame_specs:
        outer = spec["pattern"].shape[0] + 2
        placed = False
        for _ in range(250):
            r0 = rng.randint(0, h - outer)
            c0 = rng.randint(0, w - outer)
            if _can_place(occupied, r0, c0, outer):
                _draw_frame(grid, r0, c0, spec["color"], spec["pattern"])
                _mark_occupied(occupied, r0, c0, outer)
                placed = True
                break
        if not placed:
            return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
