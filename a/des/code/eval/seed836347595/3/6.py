from __future__ import annotations

import random

import numpy as np


BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
CYAN = 8
MAROON = 9

PIPE_COLORS = (RED, BLUE, YELLOW, GREEN)
MIX_TABLE = {
    frozenset((RED, BLUE)): MAGENTA,
    frozenset((RED, YELLOW)): ORANGE,
    frozenset((BLUE, YELLOW)): GREEN,
    frozenset((RED, GREEN)): MAROON,
    frozenset((BLUE, GREEN)): CYAN,
    frozenset((YELLOW, GREEN)): GREEN,
}

LEGEND_HEIGHT = 4
ENTRY_WIDTH = 2
ENTRY_GAP = 1
END_PADDING = 3
ORTH_PADDING = 1
MAX_GENERATION_ATTEMPTS = 400

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPPOSITE = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}


def _bg_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[np.argmax(cnts)])


def _parse_legend_entry(block: np.ndarray, bg: int) -> tuple[int, int, int] | None:
    if block.shape != (LEGEND_HEIGHT, ENTRY_WIDTH):
        return None
    if np.any(block == bg):
        return None
    left = set(map(int, block[:2, 0]))
    right = set(map(int, block[:2, 1]))
    result = set(map(int, block[2:, :].ravel()))
    if len(left) != 1 or len(right) != 1 or len(result) != 1:
        return None
    a = next(iter(left))
    b = next(iter(right))
    c = next(iter(result))
    if a == bg or b == bg or c == bg or a == b:
        return None
    return a, b, c


def _parse_legend_strip(strip: np.ndarray, bg: int) -> tuple[list[tuple[int, int, int]], int] | None:
    nonbg_cols = np.flatnonzero(np.any(strip != bg, axis=0))
    if len(nonbg_cols) == 0 or int(nonbg_cols[0]) != 0:
        return None
    width = int(nonbg_cols[-1]) + 1
    if not np.all(strip[:, width:] == bg):
        return None

    entries: list[tuple[int, int, int]] = []
    c = 0
    while c < width:
        if c + ENTRY_WIDTH > width:
            return None
        entry = _parse_legend_entry(strip[:, c : c + ENTRY_WIDTH], bg)
        if entry is None:
            return None
        entries.append(entry)
        c += ENTRY_WIDTH
        if c == width:
            break
        if c >= width or not np.all(strip[:, c] == bg):
            return None
        c += ENTRY_GAP

    if len(entries) < 3:
        return None
    return entries, width


def _find_legend(grid: np.ndarray, bg: int) -> tuple[dict[frozenset[int], int], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    strips = [
        ("top", "left", g[:LEGEND_HEIGHT, :]),
        ("top", "right", g[:LEGEND_HEIGHT, ::-1]),
        ("bottom", "left", g[h - LEGEND_HEIGHT :, :]),
        ("bottom", "right", g[h - LEGEND_HEIGHT :, ::-1]),
    ]

    candidates: list[tuple[dict[frozenset[int], int], np.ndarray]] = []
    for vertical, horizontal, strip in strips:
        parsed = _parse_legend_strip(strip, bg)
        if parsed is None:
            continue
        entries, width = parsed
        mapping: dict[frozenset[int], int] = {}
        for a, b, result in entries:
            key = frozenset((int(a), int(b)))
            if key in mapping and mapping[key] != int(result):
                raise AssertionError("legend contains conflicting mix rules")
            mapping[key] = int(result)

        mask = np.zeros_like(g, dtype=bool)
        row_slice = slice(0, LEGEND_HEIGHT) if vertical == "top" else slice(h - LEGEND_HEIGHT, h)
        if horizontal == "left":
            col_slice = slice(0, width)
        else:
            col_slice = slice(w - width, w)
        mask[row_slice, col_slice] = True

        # Ensure there are no extra non-background pixels inside the corner strip beyond the parsed legend.
        view = g[row_slice, col_slice]
        if np.any(view == bg):
            gap_cols = [c for c in range(view.shape[1]) if np.all(view[:, c] == bg)]
            if not gap_cols and len(entries) > 1:
                continue

        candidates.append((mapping, mask))

    assert len(candidates) == 1, "expected exactly one corner legend"
    return candidates[0]


def _endpoint_open_side(
    grid: np.ndarray,
    pipe_mask: np.ndarray,
    r: int,
    c: int,
) -> str | None:
    color = int(grid[r, c])
    neighbors: list[str] = []
    for name, (dr, dc) in DIRS.items():
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and pipe_mask[nr, nc]:
            if int(grid[nr, nc]) != color:
                return None
            neighbors.append(name)
    if len(neighbors) != 1:
        return None
    return OPPOSITE[neighbors[0]]


def _collect_gap_repairs(
    grid: np.ndarray,
    pipe_mask: np.ndarray,
    bg: int,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    h, w = grid.shape
    endpoint_side: dict[tuple[int, int], str | None] = {}
    for r, c in np.argwhere(pipe_mask):
        endpoint_side[(int(r), int(c))] = _endpoint_open_side(grid, pipe_mask, int(r), int(c))

    repairs: list[tuple[tuple[int, int], tuple[int, int]]] = []

    for (r, c), side in endpoint_side.items():
        if side != "right":
            continue
        x = c + 1
        while x < w and x - c - 1 <= 3 and int(grid[r, x]) == bg:
            x += 1
        gap = x - c - 1
        if 1 <= gap <= 3 and x < w and pipe_mask[r, x] and endpoint_side.get((r, x)) == "left":
            repairs.append(((r, c), (r, x)))

    for (r, c), side in endpoint_side.items():
        if side != "down":
            continue
        x = r + 1
        while x < h and x - r - 1 <= 3 and int(grid[x, c]) == bg:
            x += 1
        gap = x - r - 1
        if 1 <= gap <= 3 and x < h and pipe_mask[x, c] and endpoint_side.get((x, c)) == "up":
            repairs.append(((r, c), (x, c)))

    return repairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair aligned pipe gaps using the color-mixing legend in the corner.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _bg_color(g)
    legend_map, legend_mask = _find_legend(g, bg)
    pipe_mask = (g != bg) & (~legend_mask)
    repairs = _collect_gap_repairs(g, pipe_mask, bg)

    out = g.copy()
    for (r1, c1), (r2, c2) in repairs:
        color1 = int(g[r1, c1])
        color2 = int(g[r2, c2])
        fill = color1 if color1 == color2 else legend_map[frozenset((color1, color2))]
        if r1 == r2:
            out[r1, c1 + 1 : c2] = fill
        else:
            out[r1 + 1 : r2, c1] = fill
    return out


def _legend_width() -> int:
    return len(MIX_TABLE) * ENTRY_WIDTH + (len(MIX_TABLE) - 1) * ENTRY_GAP


def _place_legend(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    width = _legend_width()
    top = random.choice([True, False])
    left = random.choice([True, False])
    r0 = 0 if top else h - LEGEND_HEIGHT
    c0 = 0 if left else w - width

    entries = list(MIX_TABLE.items())
    random.shuffle(entries)
    for idx, (pair, result) in enumerate(entries):
        a, b = tuple(pair)
        if random.random() < 0.5:
            a, b = b, a
        cc = c0 + idx * (ENTRY_WIDTH + ENTRY_GAP)
        grid[r0 : r0 + 2, cc] = a
        grid[r0 : r0 + 2, cc + 1] = b
        grid[r0 + 2 : r0 + 4, cc : cc + 2] = result

    blocked = np.zeros_like(grid, dtype=bool)
    rr0 = max(0, r0 - 1)
    rr1 = min(h, r0 + LEGEND_HEIGHT + 1)
    cc0 = max(0, c0 - 1)
    cc1 = min(w, c0 + width + 1)
    blocked[rr0:rr1, cc0:cc1] = True
    return blocked


def _reserve_segment(
    reserved: np.ndarray,
    orientation: str,
    row: int,
    col: int,
    span: int,
) -> None:
    h, w = reserved.shape
    if orientation == "h":
        r0 = max(0, row - ORTH_PADDING)
        r1 = min(h, row + ORTH_PADDING + 1)
        c0 = max(0, col - END_PADDING)
        c1 = min(w, col + span + END_PADDING)
    else:
        r0 = max(0, row - END_PADDING)
        r1 = min(h, row + span + END_PADDING)
        c0 = max(0, col - ORTH_PADDING)
        c1 = min(w, col + ORTH_PADDING + 1)
    reserved[r0:r1, c0:c1] = True


def _place_pair(
    grid: np.ndarray,
    expected: np.ndarray,
    reserved: np.ndarray,
    orientation: str,
    color1: int,
    color2: int,
    fill: int,
) -> bool:
    h, w = grid.shape
    len1 = random.randint(2, 6)
    len2 = random.randint(2, 6)
    gap = random.randint(1, 3)
    span = len1 + gap + len2
    if orientation == "h" and span > w:
        return False
    if orientation == "v" and span > h:
        return False

    for _ in range(250):
        if orientation == "h":
            row = random.randrange(h)
            col = random.randrange(0, w - span + 1)
            if reserved[max(0, row - ORTH_PADDING) : min(h, row + ORTH_PADDING + 1), max(0, col - END_PADDING) : min(w, col + span + END_PADDING)].any():
                continue
            left = [(row, col + i) for i in range(len1)]
            right = [(row, col + len1 + gap + i) for i in range(len2)]
            gap_cells = [(row, col + len1 + i) for i in range(gap)]
        else:
            row = random.randrange(0, h - span + 1)
            col = random.randrange(w)
            if reserved[max(0, row - END_PADDING) : min(h, row + span + END_PADDING), max(0, col - ORTH_PADDING) : min(w, col + ORTH_PADDING + 1)].any():
                continue
            left = [(row + i, col) for i in range(len1)]
            right = [(row + len1 + gap + i, col) for i in range(len2)]
            gap_cells = [(row + len1 + i, col) for i in range(gap)]

        if any(grid[r, c] != GREY for r, c in left + right + gap_cells):
            continue

        for r, c in left:
            grid[r, c] = color1
            expected[r, c] = color1
        for r, c in right:
            grid[r, c] = color2
            expected[r, c] = color2
        for r, c in gap_cells:
            expected[r, c] = fill

        _reserve_segment(reserved, orientation, row, col, span)
        return True
    return False


def _place_dead_end(grid: np.ndarray, expected: np.ndarray, reserved: np.ndarray) -> bool:
    h, w = grid.shape
    orientation = random.choice(["h", "v"])
    length = random.randint(2, 6)
    color = random.choice(PIPE_COLORS)

    for _ in range(250):
        if orientation == "h":
            row = random.randrange(h)
            col = random.randrange(0, w - length + 1)
            if reserved[max(0, row - ORTH_PADDING) : min(h, row + ORTH_PADDING + 1), max(0, col - END_PADDING) : min(w, col + length + END_PADDING)].any():
                continue
            cells = [(row, col + i) for i in range(length)]
        else:
            row = random.randrange(0, h - length + 1)
            col = random.randrange(w)
            if reserved[max(0, row - END_PADDING) : min(h, row + length + END_PADDING), max(0, col - ORTH_PADDING) : min(w, col + ORTH_PADDING + 1)].any():
                continue
            cells = [(row + i, col) for i in range(length)]

        if any(grid[r, c] != GREY for r, c in cells):
            continue

        for r, c in cells:
            grid[r, c] = color
            expected[r, c] = color
        _reserve_segment(reserved, orientation, row, col, length)
        return True
    return False


def _pair_modes(count: int) -> list[str]:
    if count == 1:
        return [random.choice(["same", "mixed"])]
    mode = random.choice(["same", "mixed", "both"])
    if mode == "same":
        return ["same"] * count
    if mode == "mixed":
        return ["mixed"] * count
    modes = ["same", "mixed"]
    while len(modes) < count:
        modes.append(random.choice(["same", "mixed"]))
    random.shuffle(modes)
    return modes


def _pair_orientations(count: int) -> list[str]:
    if count == 1:
        return [random.choice(["h", "v"])]
    if random.random() < 0.5:
        dirs = ["h", "v"]
        while len(dirs) < count:
            dirs.append(random.choice(["h", "v"]))
        random.shuffle(dirs)
        return dirs
    return [random.choice(["h", "v"]) for _ in range(count)]


def _choose_pair_colors(mode: str) -> tuple[int, int, int]:
    if mode == "same":
        color = random.choice(PIPE_COLORS)
        return color, color, color
    pair = random.choice(list(MIX_TABLE))
    a, b = tuple(pair)
    if random.random() < 0.5:
        a, b = b, a
    return a, b, MIX_TABLE[pair]


def _build_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    height = random.randint(12, 18)
    width = random.randint(max(22, _legend_width() + 4), 30)
    grid = np.full((height, width), GREY, dtype=int)
    expected = grid.copy()
    reserved = _place_legend(grid)
    expected[:] = grid

    num_pairs = random.randint(1, 4)
    num_dead = random.randint(0, 3)
    for mode, orientation in zip(_pair_modes(num_pairs), _pair_orientations(num_pairs), strict=False):
        color1, color2, fill = _choose_pair_colors(mode)
        if not _place_pair(grid, expected, reserved, orientation, color1, color2, fill):
            return None

    for _ in range(num_dead):
        if not _place_dead_end(grid, expected, reserved):
            return None

    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_attempt()
        if built is None:
            continue
        grid, expected = built
        try:
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid pipe-repair puzzle")
