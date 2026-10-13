from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
PATTERN_COLORS = [1, 2, 3, 4, 6, 7, 8]
DIRECTIONS = ("top", "right", "bottom", "left")


def _spans_from_separator_lines(lines: list[int], length: int) -> list[tuple[int, int]]:
    bounds: list[int] = []
    if not lines or int(lines[0]) != 0:
        bounds.append(-1)
    bounds.extend(int(x) for x in lines)
    if int(bounds[-1]) != int(length - 1):
        bounds.append(int(length))

    spans = []
    for a, b in zip(bounds, bounds[1:]):
        if int(b - a) > 1:
            spans.append((int(a + 1), int(b)))
    return spans


def _panel_spans(g: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    H, W = g.shape
    sep_rows = [int(r) for r in range(H) if bool(np.all(g[int(r), :] == GREY))]
    sep_cols = [int(c) for c in range(W) if bool(np.all(g[:, int(c)] == GREY))]

    row_spans = _spans_from_separator_lines(sep_rows, H)
    col_spans = _spans_from_separator_lines(sep_cols, W)
    assert row_spans and col_spans, "expected at least one panel"

    sizes = [int(b - a) for a, b in row_spans + col_spans]
    assert len(set(sizes)) == 1, "panels must be equally-sized squares"
    return row_spans, col_spans


def _gap_direction(panel: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> str:
    misses = {
        "top": int(np.sum(panel[int(r0), int(c0) : int(c1) + 1] != MAROON)),
        "bottom": int(np.sum(panel[int(r1), int(c0) : int(c1) + 1] != MAROON)),
        "left": int(np.sum(panel[int(r0) : int(r1) + 1, int(c0)] != MAROON)),
        "right": int(np.sum(panel[int(r0) : int(r1) + 1, int(c1)] != MAROON)),
    }
    gap_sides = [side for side, count in misses.items() if int(count) == 2]
    assert len(gap_sides) == 1, "expected exactly one two-cell border gap"
    assert all(int(count) in (0, 2) for count in misses.values()), "unexpected border damage"
    return gap_sides[0]


def _extract_rotated_interior(panel: np.ndarray) -> np.ndarray:
    pts = np.argwhere(panel == MAROON)
    assert pts.size > 0, "each panel needs a maroon frame"
    r0, c0 = map(int, pts.min(axis=0))
    r1, c1 = map(int, pts.max(axis=0))
    assert int(r1 - r0) >= 2 and int(c1 - c0) >= 2, "frame must enclose an interior"

    direction = _gap_direction(panel, r0, c0, r1, c1)
    interior = panel[int(r0) + 1 : int(r1), int(c0) + 1 : int(c1)].copy()
    rotations = {"top": 0, "right": 1, "bottom": 2, "left": -1}
    return np.rot90(interior, int(rotations[direction]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each framed panel's interior pattern according to the direction of its border gap.

    Input structure:
    - Black is the background, and Grey full separator lines divide the grid into equal square panels.
    - Each panel contains one one-pixel-thick Maroon rectangular frame with exactly one two-cell gap on its top, right, bottom, or left side.
    - The frame interior contains a small multicolor pattern on Black background.

    Rule:
    - For every panel, find the missing two-cell gap in the Maroon frame.
    - Extract only the frame interior and rotate it so the encoded gap direction becomes upward: top = unchanged, right = 90 degrees counterclockwise, bottom = 180 degrees, left = 90 degrees clockwise.
    - Arrange the rotated interiors in the same panel order, separated by one Black column between output columns and one Black row between output rows.

    Key insight: The border gap is an orientation marker; it is not part of the picture to copy, but the instruction for how to rotate the hidden interior pattern before extracting it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    row_spans, col_spans = _panel_spans(g)
    rotated: list[list[np.ndarray]] = []
    for rr0, rr1 in row_spans:
        row = []
        for cc0, cc1 in col_spans:
            panel = g[int(rr0) : int(rr1), int(cc0) : int(cc1)]
            row.append(_extract_rotated_interior(panel))
        rotated.append(row)

    ph, pw = rotated[0][0].shape
    assert ph > 0 and pw > 0
    for row in rotated:
        for patch in row:
            assert patch.shape == (ph, pw), "all extracted patterns must share one output panel size"

    out_h = int(len(rotated) * ph + max(0, len(rotated) - 1))
    out_w = int(len(rotated[0]) * pw + max(0, len(rotated[0]) - 1))
    out = np.full((out_h, out_w), BLACK, dtype=int)

    for ri, row in enumerate(rotated):
        for ci, patch in enumerate(row):
            r = int(ri * (ph + 1))
            c = int(ci * (pw + 1))
            out[r : r + ph, c : c + pw] = patch
    return out


def _make_pattern(size: int) -> np.ndarray:
    for _ in range(100):
        pattern = np.full((int(size), int(size)), BLACK, dtype=int)
        total = int(size * size)
        target = random.randint(max(3, total // 3), max(4, total - 1))

        start = (random.randrange(size), random.randrange(size))
        cells = {start}
        frontier = {start}
        while len(cells) < target:
            base = random.choice(tuple(frontier or cells))
            br, bc = base
            neigh = []
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = int(br + dr), int(bc + dc)
                    if 0 <= nr < size and 0 <= nc < size and (nr, nc) not in cells:
                        neigh.append((nr, nc))
            if not neigh:
                frontier.discard(base)
                if not frontier:
                    break
                continue
            cell = random.choice(neigh)
            cells.add(cell)
            frontier.add(cell)

        if len(cells) < 3:
            continue

        ncols = random.randint(2, min(4, len(PATTERN_COLORS), len(cells)))
        colors = random.sample(PATTERN_COLORS, ncols)
        ordered = list(cells)
        random.shuffle(ordered)
        for i, (r, c) in enumerate(ordered):
            pattern[int(r), int(c)] = int(colors[i % len(colors)])

        if len(set(map(int, pattern[pattern != BLACK]))) < 2:
            continue
        if any(bool(np.array_equal(pattern, np.rot90(pattern, k))) for k in (1, 2, 3)):
            continue
        return pattern

    pattern = np.full((int(size), int(size)), BLACK, dtype=int)
    pattern[0, 0] = 1
    pattern[0, 1] = 2
    pattern[1, 0] = 3
    pattern[int(size) - 1, int(size) - 1] = 4
    return pattern


def _draw_frame(panel: np.ndarray, top: int, left: int, interior: np.ndarray, direction: str) -> None:
    q = int(interior.shape[0])
    side = int(q + 2)
    r0, c0 = int(top), int(left)
    r1, c1 = int(r0 + side - 1), int(c0 + side - 1)

    panel[r0, c0 : c1 + 1] = MAROON
    panel[r1, c0 : c1 + 1] = MAROON
    panel[r0 : r1 + 1, c0] = MAROON
    panel[r0 : r1 + 1, c1] = MAROON

    gap_start = random.randint(1, side - 3)
    if direction == "top":
        panel[r0, c0 + gap_start : c0 + gap_start + 2] = BLACK
    elif direction == "bottom":
        panel[r1, c0 + gap_start : c0 + gap_start + 2] = BLACK
    elif direction == "left":
        panel[r0 + gap_start : r0 + gap_start + 2, c0] = BLACK
    elif direction == "right":
        panel[r0 + gap_start : r0 + gap_start + 2, c1] = BLACK
    else:
        raise AssertionError("unknown direction")

    panel[r0 + 1 : r1, c0 + 1 : c1] = interior


def _expected_from_patterns(patterns: list[np.ndarray], directions: list[str]) -> np.ndarray:
    rotations = {"top": 0, "right": 1, "bottom": 2, "left": -1}
    patches = [np.rot90(pat, int(rotations[direction])) for pat, direction in zip(patterns, directions)]
    q = int(patches[0].shape[0])
    out = np.full((q, len(patches) * q + len(patches) - 1), BLACK, dtype=int)
    for i, patch in enumerate(patches):
        c = int(i * (q + 1))
        out[:, c : c + q] = patch
    return out


def _gen_attempt() -> np.ndarray | None:
    n_panels = random.choice([2, 3, 4])
    max_panel = int((30 - (n_panels + 1)) // n_panels)
    if max_panel < 5:
        return None

    panel_size = random.randint(5, max_panel)
    q_max = min(6, int(panel_size - 2))
    if q_max < 3:
        return None
    q = random.randint(3, q_max)
    frame_side = int(q + 2)

    H = int(panel_size + 2)
    W = int(n_panels * panel_size + n_panels + 1)
    if max(H, W) > 30:
        return None

    grid = np.full((H, W), BLACK, dtype=int)
    grid[0, :] = GREY
    grid[-1, :] = GREY
    for c in range(0, W, panel_size + 1):
        grid[:, int(c)] = GREY

    dirs = list(DIRECTIONS)
    random.shuffle(dirs)
    directions = dirs[:n_panels]
    patterns = [_make_pattern(q) for _ in range(n_panels)]

    for i, (pattern, direction) in enumerate(zip(patterns, directions)):
        panel_c0 = int(1 + i * (panel_size + 1))
        panel = grid[1 : 1 + panel_size, panel_c0 : panel_c0 + panel_size]
        slack = int(panel_size - frame_side)
        top = random.randint(0, slack)
        left = random.randint(0, slack)
        _draw_frame(panel, top, left, pattern, direction)

    expected = _expected_from_patterns(patterns, directions)
    try:
        produced = main(grid)
    except Exception:
        return None
    if not np.array_equal(produced, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid framed-rotation input")
