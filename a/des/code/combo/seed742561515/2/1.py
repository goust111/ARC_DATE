from __future__ import annotations

import numpy as np


BG = 0
DIRECTIONS = ("top", "bottom", "left", "right")
COLOR_POOL = tuple(range(1, 10))

# Generator configuration.
GRID_H_RANGE = (16, 30)  # inclusive
GRID_W_RANGE = (16, 30)  # inclusive
FRAME_COUNT_RANGE = (1, 4)  # inclusive
FRAME_CLEARANCE = 2
FRAME_PLACEMENT_TRIES = 160
SEED_PLACEMENT_TRIES = 160
GEN_RETRIES = 500

# Interior sizes. The side-parallel dimension is kept >= 4 so the 2-cell trap
# can touch only its intended wall rather than a corner as well.
INTERIOR_SHORT_RANGE = (2, 6)  # inclusive
INTERIOR_LONG_RANGE = (4, 7)  # inclusive


def _randint(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(lo, hi + 1))


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _component_cells(mask: np.ndarray, start: tuple[int, int], seen: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    stack = [start]
    seen[start] = True
    cells: list[tuple[int, int]] = []
    while stack:
        r, c = stack.pop()
        cells.append((r, c))
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return cells


def _color_components(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    mask = grid == color
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        comps.append(_component_cells(mask, (rr, cc), seen))
    return comps


def _trap_side(
    frame_bbox: tuple[int, int, int, int], trap_cells: list[tuple[int, int]]
) -> str | None:
    r0, c0, r1, c1 = frame_bbox
    sides = set()
    for r, c in trap_cells:
        if r == r0 + 1:
            sides.add("top")
        if r == r1 - 1:
            sides.add("bottom")
        if c == c0 + 1:
            sides.add("left")
        if c == c1 - 1:
            sides.add("right")
    if len(sides) != 1:
        return None
    return next(iter(sides))


def _frame_from_component(
    grid: np.ndarray, color: int, cells: list[tuple[int, int]]
) -> dict[str, int | str] | None:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4:
        return None
    if len(cells) != 2 * h + 2 * w - 4:
        return None

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if not np.array_equal(sub == color, border):
        return None

    interior = sub[1:-1, 1:-1]
    trap_coords = np.argwhere(interior != BG)
    if trap_coords.shape[0] != 2:
        return None

    trap_colors = np.unique(interior[interior != BG])
    if trap_colors.size != 1:
        return None
    trap_color = int(trap_colors[0])
    if trap_color == color:
        return None

    min_r, min_c = trap_coords.min(axis=0)
    max_r, max_c = trap_coords.max(axis=0)
    min_r = int(min_r)
    min_c = int(min_c)
    max_r = int(max_r)
    max_c = int(max_c)
    trap_patch = interior[min_r : max_r + 1, min_c : max_c + 1]
    if not np.all(trap_patch == trap_color):
        return None
    if trap_patch.shape not in ((1, 2), (2, 1)):
        return None

    trap_cells = [(r0 + 1 + int(r), c0 + 1 + int(c)) for r, c in trap_coords]
    side = _trap_side((r0, c0, r1, c1), trap_cells)
    if side is None:
        return None

    return {
        "color": color,
        "trap_color": trap_color,
        "r0": r0,
        "c0": c0,
        "r1": r1,
        "c1": c1,
        "direction": side,
    }


def _find_frames(grid: np.ndarray) -> list[dict[str, int | str]]:
    frames: list[dict[str, int | str]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == BG:
            continue
        for cells in _color_components(grid, color):
            frame = _frame_from_component(grid, color, cells)
            if frame is not None:
                frames.append(frame)
    frames.sort(key=lambda fr: (int(fr["r0"]), int(fr["c0"])))
    return frames


def _is_singleton_seed(grid: np.ndarray, r: int, c: int, color: int) -> bool:
    h, w = grid.shape
    if int(grid[r, c]) != color:
        return False
    for nr, nc in _neighbors4(r, c):
        if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == color:
            return False
    return True


def _visible_seed_candidates(grid: np.ndarray, frame: dict[str, int | str]) -> list[tuple[int, int]]:
    h, w = grid.shape
    color = int(frame["color"])
    r0 = int(frame["r0"])
    c0 = int(frame["c0"])
    r1 = int(frame["r1"])
    c1 = int(frame["c1"])
    direction = str(frame["direction"])
    candidates: set[tuple[int, int]] = set()

    if direction == "top":
        for c in range(c0, c1 + 1):
            for r in range(r0 - 1, -1, -1):
                value = int(grid[r, c])
                if value == BG:
                    continue
                if value == color and _is_singleton_seed(grid, r, c, color):
                    candidates.add((r, c))
                break
    elif direction == "bottom":
        for c in range(c0, c1 + 1):
            for r in range(r1 + 1, h):
                value = int(grid[r, c])
                if value == BG:
                    continue
                if value == color and _is_singleton_seed(grid, r, c, color):
                    candidates.add((r, c))
                break
    elif direction == "left":
        for r in range(r0, r1 + 1):
            for c in range(c0 - 1, -1, -1):
                value = int(grid[r, c])
                if value == BG:
                    continue
                if value == color and _is_singleton_seed(grid, r, c, color):
                    candidates.add((r, c))
                break
    else:
        for r in range(r0, r1 + 1):
            for c in range(c1 + 1, w):
                value = int(grid[r, c])
                if value == BG:
                    continue
                if value == color and _is_singleton_seed(grid, r, c, color):
                    candidates.add((r, c))
                break

    return sorted(candidates)


def _find_directional_seed(grid: np.ndarray, frame: dict[str, int | str]) -> tuple[int, int]:
    candidates = _visible_seed_candidates(grid, frame)
    assert len(candidates) == 1, "each fence must have exactly one visible seed in the trap direction"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each rectangular fence interior using the matching outside seed selected
    by the trap strip's wall direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    frames = _find_frames(g)
    assert frames, "expected at least one valid fence"

    for frame in frames:
        seed_r, seed_c = _find_directional_seed(g, frame)
        fill_color = int(g[seed_r, seed_c])
        r0 = int(frame["r0"])
        c0 = int(frame["c0"])
        r1 = int(frame["r1"])
        c1 = int(frame["c1"])
        out[r0 + 1 : r1, c0 + 1 : c1] = fill_color

    return out


def _empty_with_margin(
    blocked: np.ndarray, r0: int, c0: int, r1: int, c1: int, margin: int
) -> bool:
    h, w = blocked.shape
    rr0 = max(0, r0 - margin)
    cc0 = max(0, c0 - margin)
    rr1 = min(h - 1, r1 + margin)
    cc1 = min(w - 1, c1 + margin)
    return not bool(blocked[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _mark_with_margin(
    blocked: np.ndarray, r0: int, c0: int, r1: int, c1: int, margin: int
) -> None:
    h, w = blocked.shape
    rr0 = max(0, r0 - margin)
    cc0 = max(0, c0 - margin)
    rr1 = min(h - 1, r1 + margin)
    cc1 = min(w - 1, c1 + margin)
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _draw_frame(grid: np.ndarray, frame: dict[str, int | str]) -> None:
    color = int(frame["color"])
    trap_color = int(frame["trap_color"])
    r0 = int(frame["r0"])
    c0 = int(frame["c0"])
    r1 = int(frame["r1"])
    c1 = int(frame["c1"])
    direction = str(frame["direction"])
    ih = int(frame["ih"])
    iw = int(frame["iw"])
    trap_offset = int(frame["trap_offset"])

    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color

    if direction == "top":
        row = r0 + 1
        col = c0 + 1 + trap_offset
        grid[row, col : col + 2] = trap_color
    elif direction == "bottom":
        row = r0 + ih
        col = c0 + 1 + trap_offset
        grid[row, col : col + 2] = trap_color
    elif direction == "left":
        row = r0 + 1 + trap_offset
        col = c0 + 1
        grid[row : row + 2, col] = trap_color
    else:
        row = r0 + 1 + trap_offset
        col = c0 + iw
        grid[row : row + 2, col] = trap_color


def _seed_is_isolated(grid: np.ndarray, r: int, c: int) -> bool:
    h, w = grid.shape
    if int(grid[r, c]) != BG:
        return False
    for nr in range(max(0, r - 1), min(h, r + 2)):
        for nc in range(max(0, c - 1), min(w, c + 2)):
            if (nr != r or nc != c) and int(grid[nr, nc]) != BG:
                return False
    return True


def _valid_seed_positions(
    grid: np.ndarray, frame: dict[str, int | str]
) -> list[tuple[int, int]]:
    h, w = grid.shape
    r0 = int(frame["r0"])
    c0 = int(frame["c0"])
    r1 = int(frame["r1"])
    c1 = int(frame["c1"])
    direction = str(frame["direction"])
    positions: list[tuple[int, int]] = []

    if direction == "top":
        for c in range(c0 + 1, c1):
            for r in range(0, r0 - 1):
                if not _seed_is_isolated(grid, r, c):
                    continue
                if np.all(grid[r + 1 : r0, c] == BG):
                    positions.append((r, c))
    elif direction == "bottom":
        for c in range(c0 + 1, c1):
            for r in range(r1 + 2, h):
                if not _seed_is_isolated(grid, r, c):
                    continue
                if np.all(grid[r1 + 1 : r, c] == BG):
                    positions.append((r, c))
    elif direction == "left":
        for r in range(r0 + 1, r1):
            for c in range(0, c0 - 1):
                if not _seed_is_isolated(grid, r, c):
                    continue
                if np.all(grid[r, c + 1 : c0] == BG):
                    positions.append((r, c))
    else:
        for r in range(r0 + 1, r1):
            for c in range(c1 + 2, w):
                if not _seed_is_isolated(grid, r, c):
                    continue
                if np.all(grid[r, c1 + 1 : c] == BG):
                    positions.append((r, c))

    return positions


def _sample_frame_spec(
    rng: np.random.Generator, grid_h: int, grid_w: int, color: int, trap_color: int
) -> dict[str, int | str]:
    direction = DIRECTIONS[int(rng.integers(0, len(DIRECTIONS)))]
    if direction in ("top", "bottom"):
        ih = _randint(rng, *INTERIOR_SHORT_RANGE)
        iw = _randint(rng, *INTERIOR_LONG_RANGE)
        trap_offset = _randint(rng, 1, iw - 3)
    else:
        ih = _randint(rng, *INTERIOR_LONG_RANGE)
        iw = _randint(rng, *INTERIOR_SHORT_RANGE)
        trap_offset = _randint(rng, 1, ih - 3)

    fh = ih + 2
    fw = iw + 2
    r0 = _randint(rng, FRAME_CLEARANCE, grid_h - fh - FRAME_CLEARANCE)
    c0 = _randint(rng, FRAME_CLEARANCE, grid_w - fw - FRAME_CLEARANCE)
    return {
        "color": color,
        "trap_color": trap_color,
        "direction": direction,
        "ih": ih,
        "iw": iw,
        "trap_offset": trap_offset,
        "r0": r0,
        "c0": c0,
        "r1": r0 + fh - 1,
        "c1": c0 + fw - 1,
    }


def _expected_output(grid: np.ndarray, frames: list[dict[str, int | str]]) -> np.ndarray:
    out = grid.copy()
    for frame in frames:
        r0 = int(frame["r0"])
        c0 = int(frame["c0"])
        r1 = int(frame["r1"])
        c1 = int(frame["c1"])
        out[r0 + 1 : r1, c0 + 1 : c1] = int(frame["color"])
    return out


def _generate_candidate() -> np.ndarray | None:
    rng = np.random.default_rng()
    grid_h = _randint(rng, *GRID_H_RANGE)
    grid_w = _randint(rng, *GRID_W_RANGE)
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    max_frames = min(FRAME_COUNT_RANGE[1], (grid_h * grid_w) // 90)
    if max_frames < 1:
        return None
    frame_count = _randint(rng, FRAME_COUNT_RANGE[0], max_frames)
    if 2 * frame_count > len(COLOR_POOL):
        return None

    colors = list(COLOR_POOL)
    rng.shuffle(colors)
    border_colors = [int(c) for c in colors[:frame_count]]
    trap_colors = [int(c) for c in colors[frame_count : 2 * frame_count]]

    blocked = np.zeros_like(grid, dtype=bool)
    frames: list[dict[str, int | str]] = []
    for color, trap_color in zip(border_colors, trap_colors, strict=False):
        placed = False
        for _ in range(FRAME_PLACEMENT_TRIES):
            frame = _sample_frame_spec(rng, grid_h, grid_w, color, trap_color)
            if not _empty_with_margin(
                blocked,
                int(frame["r0"]),
                int(frame["c0"]),
                int(frame["r1"]),
                int(frame["c1"]),
                FRAME_CLEARANCE,
            ):
                continue
            _draw_frame(grid, frame)
            _mark_with_margin(
                blocked,
                int(frame["r0"]),
                int(frame["c0"]),
                int(frame["r1"]),
                int(frame["c1"]),
                FRAME_CLEARANCE,
            )
            frames.append(frame)
            placed = True
            break
        if not placed:
            return None

    for frame in frames:
        positions = _valid_seed_positions(grid, frame)
        if not positions:
            return None
        seed_r, seed_c = positions[int(rng.integers(0, len(positions)))]
        grid[seed_r, seed_c] = int(frame["color"])

    found_frames = _find_frames(grid)
    if len(found_frames) != frame_count:
        return None

    expected = _expected_output(grid, frames)
    try:
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 1")
