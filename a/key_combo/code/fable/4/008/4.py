from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
YELLOW = 4
CARDINAL_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAGONAL_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ALL_DIRS = CARDINAL_DIRS + DIAGONAL_DIRS


def _bg_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _touches_edge(box: tuple[int, int, int, int], shape: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = box
    h, w = shape
    return r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1


def _legend_info(
    grid: np.ndarray, color: int, box: tuple[int, int, int, int]
) -> tuple[tuple[int, int], int] | None:
    r0, c0, r1, c1 = box
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 5 or width < 5 or height % 2 == 0 or width % 2 == 0:
        return None

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    if not bool(np.all((sub == int(color)) | (sub == BLACK))):
        return None

    center = (r0 + height // 2, c0 + width // 2)
    if int(grid[center]) != BLACK:
        return None

    dots = [(r0 + int(rr), c0 + int(cc)) for rr, cc in np.argwhere(sub == BLACK)]
    if len(dots) != 2:
        return None
    other = dots[0] if dots[1] == center else dots[1]
    if other == center:
        return None

    dr_raw = other[0] - center[0]
    dc_raw = other[1] - center[1]
    if not (dr_raw == 0 or dc_raw == 0 or abs(dr_raw) == abs(dc_raw)):
        return None
    k = max(abs(dr_raw), abs(dc_raw))
    if k < 1:
        return None
    direction = (
        0 if dr_raw == 0 else (1 if dr_raw > 0 else -1),
        0 if dc_raw == 0 else (1 if dc_raw > 0 else -1),
    )
    if direction not in ALL_DIRS:
        return None
    return direction, int(k)


def _enclosed_mask_from_wall(
    shape: tuple[int, int], wall_coords: np.ndarray
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    box = _bbox(wall_coords)
    r0, c0, r1, c1 = box
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    wall = np.zeros((height, width), dtype=bool)
    for r, c in wall_coords:
        wall[int(r) - r0, int(c) - c0] = True

    passable = ~wall
    outside = np.zeros_like(passable, dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for rr in range(height):
        for cc in (0, width - 1):
            if bool(passable[rr, cc]) and not bool(outside[rr, cc]):
                outside[rr, cc] = True
                q.append((rr, cc))
    for cc in range(width):
        for rr in (0, height - 1):
            if bool(passable[rr, cc]) and not bool(outside[rr, cc]):
                outside[rr, cc] = True
                q.append((rr, cc))

    while q:
        rr, cc = q.popleft()
        for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
            if 0 <= nr < height and 0 <= nc < width and bool(passable[nr, nc]) and not bool(outside[nr, nc]):
                outside[nr, nc] = True
                q.append((nr, nc))

    inner_local = passable & ~outside
    inner = np.zeros(shape, dtype=bool)
    inner[r0 : r1 + 1, c0 : c1 + 1] = inner_local
    return inner, box


def _lanes(mask: np.ndarray, direction: tuple[int, int]) -> list[list[tuple[int, int]]]:
    dr, dc = direction
    h, w = mask.shape
    starts = []
    for r, c in np.argwhere(mask):
        pr, pc = int(r) - dr, int(c) - dc
        if not (0 <= pr < h and 0 <= pc < w and bool(mask[pr, pc])):
            starts.append((int(r), int(c)))

    out: list[list[tuple[int, int]]] = []
    for r, c in starts:
        cells = []
        rr, cc = r, c
        while 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]):
            cells.append((rr, cc))
            rr += dr
            cc += dc
        out.append(cells)
    return out


def _slide_yellow_on_mask(
    source: np.ndarray,
    out: np.ndarray,
    inner_mask: np.ndarray,
    direction: tuple[int, int],
    gap: int,
) -> None:
    yellow_mask = inner_mask & (source == YELLOW)
    assert bool(yellow_mask.any()), "each enclosed region must contain yellow pixels"
    out[yellow_mask] = _bg_color(source)

    for cells in _lanes(inner_mask, direction):
        count = sum(1 for r, c in cells if int(source[r, c]) == YELLOW)
        if count == 0:
            continue
        stop_idx = len(cells) - 1 - int(gap)
        assert stop_idx >= count - 1, "not enough lane space for the requested cushion"
        for i in range(count):
            r, c = cells[stop_idx - i]
            out[r, c] = YELLOW


def _parse_legends_and_walls(
    grid: np.ndarray,
) -> tuple[dict[int, tuple[tuple[int, int], int]], list[tuple[int, np.ndarray]]]:
    bg = _bg_color(grid)
    colors = [int(c) for c in np.unique(grid) if int(c) not in {bg, BLACK, YELLOW}]
    legends: dict[int, tuple[tuple[int, int], int]] = {}
    walls: list[tuple[int, np.ndarray]] = []

    for color in colors:
        for comp in _components(grid == color):
            box = _bbox(comp)
            info = _legend_info(grid, color, box) if _touches_edge(box, grid.shape) else None
            if info is not None:
                assert color not in legends, "each wall color must have one border legend"
                legends[color] = info
            else:
                walls.append((color, comp))

    wall_colors = {color for color, _coords in walls}
    assert wall_colors, "expected enclosed colored wall regions"
    assert wall_colors <= set(legends), "every wall color must have a matching legend"
    return legends, walls


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each enclosed region's yellow pixels according to its same-colored border legend.

    Input structure:
    - The background is the most common color; yellow pixels are movable.
    - Several one-color closed walls enclose separate regions, and each wall color is unique.
    - On the grid border, each wall color also appears as a filled legend rectangle with a
      black center dot and one black indicator dot aligned horizontally, vertically, or diagonally.

    Rule:
    - Match each closed region to the legend with the same wall color.
    - The vector from the legend's geometric center dot to its indicator dot gives the slide direction.
    - The Chebyshev distance between those dots is k; yellow pixels pile toward the blocking wall
      while leaving k-1 empty cells of cushion before the first yellow pixel.
    - Pixels in the same movement lane stack behind one another; walls, legends, and background stay fixed.

    Key insight:
    - The legend has two channels: point bearing gives gravity direction, while point gap gives
      the wall cushion; solving only the direction and always sticking to the wall is insufficient.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    legends, walls = _parse_legends_and_walls(g)

    seen_region_colors: set[int] = set()
    for color, wall_coords in walls:
        direction, k = legends[int(color)]
        inner_mask, _box = _enclosed_mask_from_wall(g.shape, wall_coords)
        assert bool(inner_mask.any()), "wall must enclose a non-empty region"
        _slide_yellow_on_mask(g, out, inner_mask, direction, int(k) - 1)
        seen_region_colors.add(int(color))

    assert seen_region_colors == {color for color, _coords in walls}
    return out


def _is_free(occ: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> bool:
    rows, cols = occ.shape
    if r0 < 0 or c0 < 0 or r0 + h > rows or c0 + w > cols:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(rows, r0 + h + pad)
    cc1 = min(cols, c0 + w + pad)
    return not bool(occ[rr0:rr1, cc0:cc1].any())


def _mark_occ(occ: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    occ[r0 : r0 + h, c0 : c0 + w] = True


def _place_legend(
    grid: np.ndarray,
    occ: np.ndarray,
    color: int,
    direction: tuple[int, int],
    k: int,
) -> bool:
    rows, cols = grid.shape
    size = 2 * int(k) + 3
    sides = ["top", "bottom", "left", "right"]
    random.shuffle(sides)
    for side in sides:
        for _ in range(60):
            if side == "top":
                r0 = 0
                c0 = random.randint(0, cols - size)
            elif side == "bottom":
                r0 = rows - size
                c0 = random.randint(0, cols - size)
            elif side == "left":
                r0 = random.randint(0, rows - size)
                c0 = 0
            else:
                r0 = random.randint(0, rows - size)
                c0 = cols - size
            if not _is_free(occ, r0, c0, size, size, pad=1):
                continue
            grid[r0 : r0 + size, c0 : c0 + size] = int(color)
            center = (r0 + size // 2, c0 + size // 2)
            indicator = (center[0] + direction[0] * int(k), center[1] + direction[1] * int(k))
            grid[center] = BLACK
            grid[indicator] = BLACK
            _mark_occ(occ, r0, c0, size, size)
            return True
    return False


def _draw_region_wall(
    grid: np.ndarray,
    occ: np.ndarray,
    color: int,
    inner_h: int,
    inner_w: int,
) -> tuple[int, int, int, int] | None:
    rows, cols = grid.shape
    total_h = int(inner_h) + 2
    total_w = int(inner_w) + 2
    for _ in range(240):
        r0 = random.randint(2, rows - total_h - 3)
        c0 = random.randint(2, cols - total_w - 3)
        if not _is_free(occ, r0, c0, total_h, total_w, pad=1):
            continue
        r1 = r0 + total_h - 1
        c1 = c0 + total_w - 1
        grid[r0, c0 : c1 + 1] = int(color)
        grid[r1, c0 : c1 + 1] = int(color)
        grid[r0 : r1 + 1, c0] = int(color)
        grid[r0 : r1 + 1, c1] = int(color)
        _mark_occ(occ, r0, c0, total_h, total_w)
        return r0, c0, r1, c1
    return None


def _rect_inner_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    mask = np.zeros(shape, dtype=bool)
    mask[r0 + 1 : r1, c0 + 1 : c1] = True
    return mask


def _choose_yellows_for_region(
    shape: tuple[int, int],
    box: tuple[int, int, int, int],
    direction: tuple[int, int],
    k: int,
    force_stack: bool,
) -> tuple[list[tuple[int, int]], bool] | None:
    mask = _rect_inner_mask(shape, box)
    gap = int(k) - 1
    lane_data = [cells for cells in _lanes(mask, direction) if len(cells) > gap]
    random.shuffle(lane_data)
    if not lane_data:
        return None

    stack_lanes = [cells for cells in lane_data if len(cells) - gap >= 2 and len(cells) >= 4]
    chosen: list[list[tuple[int, int]]] = []
    if force_stack:
        if not stack_lanes:
            return None
        chosen.append(random.choice(stack_lanes))

    remaining = [cells for cells in lane_data if cells not in chosen]
    lane_count = random.randint(2, min(4, len(lane_data)))
    random.shuffle(remaining)
    chosen.extend(remaining[: max(0, lane_count - len(chosen))])

    yellows: list[tuple[int, int]] = []
    made_stack = False
    for idx, cells in enumerate(chosen):
        capacity = len(cells) - gap
        if capacity <= 0:
            continue
        if force_stack and idx == 0:
            count = 2
        else:
            count = random.randint(1, min(3, capacity))
        made_stack = made_stack or count >= 2
        picks = random.sample(cells, count)
        yellows.extend(picks)

    if not yellows:
        return None
    return yellows, made_stack


def _known_specs_output(
    grid: np.ndarray,
    specs: list[dict],
    *,
    force_no_cushion: bool = False,
) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        gap = 0 if force_no_cushion else int(spec["k"]) - 1
        _slide_yellow_on_mask(grid, out, spec["inner_mask"], spec["direction"], gap)
    return out


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(24, 30)
    cols = random.randint(24, 30)
    bg = random.choice([0, 8])
    palette = [1, 2, 3, 5, 6, 7, 9]
    if bg in palette:
        palette.remove(bg)

    region_count = random.choice([2, 2, 3])
    colors = random.sample(palette, region_count)
    dirs = [random.choice(CARDINAL_DIRS), random.choice(DIAGONAL_DIRS)]
    while len(dirs) < region_count:
        dirs.append(random.choice(ALL_DIRS))
    ks = [1, random.choice([2, 3])]
    while len(ks) < region_count:
        ks.append(random.choice([1, 2, 3]))
    random.shuffle(dirs)
    random.shuffle(ks)

    grid = np.full((rows, cols), bg, dtype=int)
    occ = np.zeros((rows, cols), dtype=bool)
    specs: list[dict] = []

    for color, direction, k in zip(colors, dirs, ks):
        if not _place_legend(grid, occ, color, direction, k):
            return None

    stack_forced = False
    stack_seen = False
    for color, direction, k in zip(colors, dirs, ks):
        inner_h = random.randint(5, 8 if region_count == 3 else 9)
        inner_w = random.randint(5, 8 if region_count == 3 else 9)
        box = _draw_region_wall(grid, occ, color, inner_h, inner_w)
        if box is None:
            return None

        force_stack = not stack_forced
        choice = _choose_yellows_for_region(grid.shape, box, direction, k, force_stack)
        if choice is None:
            return None
        yellows, made_stack = choice
        stack_forced = stack_forced or force_stack
        stack_seen = stack_seen or made_stack
        for r, c in yellows:
            grid[r, c] = YELLOW
        specs.append(
            {
                "color": int(color),
                "direction": direction,
                "k": int(k),
                "inner_mask": _rect_inner_mask(grid.shape, box),
            }
        )

    if not stack_seen:
        return None
    if not any(int(spec["k"]) == 1 for spec in specs):
        return None
    if not any(int(spec["k"]) > 1 for spec in specs):
        return None
    if not any(spec["direction"] in CARDINAL_DIRS for spec in specs):
        return None
    if not any(spec["direction"] in DIAGONAL_DIRS for spec in specs):
        return None

    try:
        expected = _known_specs_output(grid, specs)
        no_cushion = _known_specs_output(grid, specs, force_no_cushion=True)
        solved = main(grid)
    except Exception:
        return None

    if np.array_equal(expected, grid):
        return None
    if np.array_equal(expected, no_cushion):
        return None
    if not np.array_equal(solved, expected):
        return None
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid dual-channel legend gravity input")
