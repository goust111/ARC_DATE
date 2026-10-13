import random
from collections import deque

import numpy as np


BG = 0
TOWER = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
CORNERS = ("TL", "TR", "BL", "BR")

MIN_SIZE = 15
MAX_SIZE = 30
MIN_TOWERS = 1
MAX_TOWERS = 4
MAX_LONG_ARM = 5
GEN_RETRIES = 500
PLACE_RETRIES = 250


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        pts: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            pts.append((r, c))
            for dr, dc in DIRECTIONS:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(pts)
    return comps


def _legend_row_cols(width: int, side: str, length: int) -> range:
    if side == "L":
        return range(length)
    return range(width - length, width)


def _legend_rows(height: int, side: str, count: int) -> list[int]:
    if side == "T":
        return list(range(count))
    return list(range(height - count, height))


def _find_legend(g: np.ndarray) -> dict[int, int]:
    h, w = g.shape
    colored = {
        (int(r), int(c))
        for r, c in np.argwhere((g != BG) & (g != TOWER))
    }
    assert colored, "expected a colored length legend"

    matches: list[dict[int, int]] = []
    for vertical in ("T", "B"):
        for horizontal in ("L", "R"):
            for count in range(1, 5):
                rows = _legend_rows(h, vertical, count)
                expected: set[tuple[int, int]] = set()
                mapping: dict[int, int] = {}
                ok = True
                for i, r in enumerate(rows):
                    length = i + 1
                    cols = list(_legend_row_cols(w, horizontal, length))
                    vals = [int(g[r, c]) for c in cols]
                    if len(set(vals)) != 1 or vals[0] in (BG, TOWER):
                        ok = False
                        break
                    mapping[length] = vals[0]
                    expected.update((r, c) for c in cols)

                if ok and len(set(mapping.values())) == count and expected == colored:
                    matches.append(mapping)

    assert len(matches) == 1, "expected one unambiguous corner legend"
    return matches[0]


def _arm_length(cells: set[tuple[int, int]], center: tuple[int, int], direction: tuple[int, int]) -> int:
    r, c = center
    dr, dc = direction
    length = 0
    r += dr
    c += dc
    while (r, c) in cells:
        length += 1
        r += dr
        c += dc
    return length


def _analyze_tower(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], list[int], int]:
    cells = set(coords)
    centers = [
        (r, c)
        for r, c in cells
        if all((r + dr, c + dc) in cells for dr, dc in DIRECTIONS)
    ]
    assert len(centers) == 1, "each tower must have one cross center"
    center = centers[0]

    lengths = [_arm_length(cells, center, direction) for direction in DIRECTIONS]
    assert min(lengths) >= 1, "each tower arm must be present"

    expected = {center}
    for length, (dr, dc) in zip(lengths, DIRECTIONS):
        cr, cc = center
        for step in range(1, length + 1):
            expected.add((cr + dr * step, cc + dc * step))
    assert expected == cells, "tower must be a one-pixel-wide cardinal cross"

    long_arm = max(lengths)
    short_arm = min(lengths)
    assert lengths.count(long_arm) == 3 and lengths.count(short_arm) == 1
    diff = long_arm - short_arm
    assert 1 <= diff <= 4
    return center, lengths, diff


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw color-coded rays from uneven gray signal-tower crosses.

    Input structure:
    - The background is black.
    - A corner contains a vertical legend of colored horizontal bars: the first
      bar has length 1, the second length 2, and so on up to length 4.
    - The remaining foreground objects are gray cardinal crosses.  Each cross
      has one center pixel, four one-pixel-wide arms, three equal long arms, and
      one shorter arm.

    Rule:
    - For each gray cross, measure the four arm lengths and compute
      D = longest arm length - shortest arm length.
    - Look up D in the corner legend by using the horizontal bar whose visible
      width is D, and take that bar's color.
    - From the cross center, draw rays in all four cardinal directions using
      that color.  Each ray starts just beyond that direction's gray arm and
      continues through black cells until it reaches the grid edge or the first
      original non-background cell.
    - The gray crosses and the corner legend remain unchanged.

    Key insight:
    - The short arm gives a visible missing-length count, and that count indexes
      the colored width legend before the four geometric rays are painted.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    legend = _find_legend(g)
    towers = _components(g == TOWER)
    assert towers, "expected at least one gray signal tower"

    out = g.copy()
    for coords in towers:
        center, lengths, diff = _analyze_tower(coords)
        assert diff in legend, "tower difference must be present in the legend"
        color = int(legend[diff])
        cr, cc = center
        for arm_len, (dr, dc) in zip(lengths, DIRECTIONS):
            r = cr + dr * (arm_len + 1)
            c = cc + dc * (arm_len + 1)
            while 0 <= r < h and 0 <= c < w:
                if int(g[r, c]) != BG:
                    break
                assert int(out[r, c]) in (BG, color), "ray colors must not conflict"
                out[r, c] = color
                r += dr
                c += dc
    return out


def _legend_cells(
    shape: tuple[int, int], corner: str, colors: list[int]
) -> list[tuple[int, int, int]]:
    h, w = shape
    vertical = "T" if corner[0] == "T" else "B"
    horizontal = "L" if corner[1] == "L" else "R"
    rows = _legend_rows(h, vertical, len(colors))
    cells: list[tuple[int, int, int]] = []
    for i, (r, color) in enumerate(zip(rows, colors)):
        length = i + 1
        for c in _legend_row_cols(w, horizontal, length):
            cells.append((r, int(c), int(color)))
    return cells


def _paint_legend(grid: np.ndarray, corner: str, colors: list[int]) -> None:
    for r, c, color in _legend_cells(grid.shape, corner, colors):
        grid[r, c] = color


def _dilate_cells(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    out = set(cells)
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _tower_lengths(long_arm: int, diff: int, short_index: int) -> list[int]:
    lengths = [long_arm] * 4
    lengths[short_index] = long_arm - diff
    return lengths


def _cross_cells(center: tuple[int, int], lengths: list[int]) -> set[tuple[int, int]]:
    cr, cc = center
    cells = {(cr, cc)}
    for length, (dr, dc) in zip(lengths, DIRECTIONS):
        for step in range(1, length + 1):
            cells.add((cr + dr * step, cc + dc * step))
    return cells


def _center_bounds(shape: tuple[int, int], lengths: list[int]) -> tuple[int, int, int, int] | None:
    h, w = shape
    up, down, left, right = lengths
    r_min = up + 1
    r_max = h - down - 2
    c_min = left + 1
    c_max = w - right - 2
    if r_min > r_max or c_min > c_max:
        return None
    return r_min, r_max, c_min, c_max


def _try_place_tower(
    rng: random.Random,
    grid: np.ndarray,
    blocked: set[tuple[int, int]],
    diff: int,
) -> set[tuple[int, int]] | None:
    for _ in range(PLACE_RETRIES):
        long_arm = rng.randint(diff + 1, MAX_LONG_ARM)
        short_index = rng.randrange(4)
        lengths = _tower_lengths(long_arm, diff, short_index)
        bounds = _center_bounds(grid.shape, lengths)
        if bounds is None:
            continue
        r_min, r_max, c_min, c_max = bounds
        center = (rng.randint(r_min, r_max), rng.randint(c_min, c_max))
        cells = _cross_cells(center, lengths)
        if any(grid[r, c] != BG for r, c in cells):
            continue
        if cells & blocked:
            continue
        return cells
    return None


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    h = rng.randint(MIN_SIZE, MAX_SIZE)
    w = rng.randint(MIN_SIZE, MAX_SIZE)
    grid = np.full((h, w), BG, dtype=int)

    diff = rng.randint(1, 4)
    legend_count = rng.randint(diff, 4)
    legend_colors = rng.sample(PALETTE, legend_count)
    corner = rng.choice(CORNERS)
    _paint_legend(grid, corner, legend_colors)

    legend_occupied = {
        (r, c)
        for r, c, _color in _legend_cells(grid.shape, corner, legend_colors)
    }
    blocked = _dilate_cells(legend_occupied, grid.shape)

    tower_count = rng.randint(MIN_TOWERS, MAX_TOWERS)
    for _ in range(tower_count):
        cells = _try_place_tower(rng, grid, blocked, diff)
        if cells is None:
            return None
        for r, c in cells:
            grid[r, c] = TOWER
        blocked.update(_dilate_cells(cells, grid.shape))

    try:
        out = main(grid)
    except Exception:
        return None
    if not np.any(out != grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid signal-tower input")
