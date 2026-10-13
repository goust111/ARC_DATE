from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
OBJ = 1
RECOLOR = 2
SEP = 4
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
PANEL_HEIGHT_RANGE = (5, 6)
PANEL_WIDTH_RANGE = (7, 12)
MAX_GENERATION_ATTEMPTS = 500


def _translate_coords(coords: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in coords}


def _split_panels(g: np.ndarray) -> tuple[list[np.ndarray], int, int]:
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(g[0, 0])
    separator_rows = [
        r
        for r in range(g.shape[0])
        if np.all(g[r, :] == g[r, 0]) and int(g[r, 0]) != bg
    ]
    assert len(separator_rows) == 3, "expected exactly three separator rows"

    panel_h = int(separator_rows[0])
    assert panel_h > 0, "panel height must be positive"
    expected = [panel_h + i * (panel_h + 1) for i in range(3)]
    assert separator_rows == expected, "separator rows must be evenly spaced"
    assert g.shape[0] == 4 * panel_h + 3, "expected four equal-height panels"

    panels = []
    start = 0
    for sep_row in separator_rows:
        panels.append(g[start:sep_row, :].copy())
        start = sep_row + 1
    panels.append(g[start:, :].copy())
    assert len({panel.shape for panel in panels}) == 1, "all panels must have the same shape"
    return panels, bg, int(g[separator_rows[0], 0])


def _coords_from_panel(panel: np.ndarray, bg: int) -> tuple[set[tuple[int, int]], int]:
    colors = sorted({int(v) for v in np.unique(panel) if int(v) != bg})
    assert len(colors) == 1, "each panel must contain exactly one foreground color"
    color = colors[0]
    coords = {(int(r), int(c)) for r, c in np.argwhere(panel == color)}
    assert coords, "panel must contain a foreground object"
    return coords, color


def _is_connected(coords: set[tuple[int, int]]) -> bool:
    if not coords:
        return False
    start = next(iter(coords))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for dr, dc in NEIGHBORS4:
            nxt = (r + dr, c + dc)
            if nxt in coords and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(coords)


def _has_hole(coords: set[tuple[int, int]]) -> bool:
    assert coords, "shape must be nonempty"
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)

    h = r1 - r0 + 3
    w = c1 - c0 + 3
    blocked = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        blocked[r - r0 + 1, c - c0 + 1] = True

    seen = np.zeros((h, w), dtype=bool)
    q = deque([(0, 0)])
    seen[0, 0] = True
    while q:
        r, c = q.popleft()
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and not blocked[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))

    bg_mask = ~blocked
    return bool(np.any(bg_mask & ~seen))


def _infer_shift(frames: list[set[tuple[int, int]]]) -> tuple[int, int]:
    candidates = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            ok = True
            for i in range(3):
                shifted = _translate_coords(frames[i], dr, dc)
                if len(frames[i + 1]) != len(shifted) + 1:
                    ok = False
                    break
                if not shifted <= frames[i + 1]:
                    ok = False
                    break
            if ok:
                candidates.append((dr, dc))
    assert len(candidates) == 1, "expected a unique frame-to-frame translation"
    return candidates[0]


def _degrees(coords: set[tuple[int, int]]) -> dict[tuple[int, int], int]:
    deg = {}
    for r, c in coords:
        deg[(r, c)] = sum((r + dr, c + dc) in coords for dr, dc in NEIGHBORS4)
    return deg


def _walk_path(coords: set[tuple[int, int]], start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    ordered = [start]
    visited = {start}
    prev = None
    curr = start
    while curr != end:
        next_cells = []
        for dr, dc in NEIGHBORS4:
            nxt = (curr[0] + dr, curr[1] + dc)
            if nxt in coords and nxt != prev and nxt not in visited:
                next_cells.append(nxt)
        assert len(next_cells) == 1, "shape must be a simple path before closure"
        prev, curr = curr, next_cells[0]
        ordered.append(curr)
        visited.add(curr)
    assert len(ordered) == len(coords), "path traversal must cover the full object"
    return ordered


def _rectangle_cycle(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    assert r1 - r0 + 1 >= 3 and c1 - c0 + 1 >= 3, "rectangle must enclose an interior"
    seq = []
    for c in range(c0, c1 + 1):
        seq.append((r0, c))
    for r in range(r0 + 1, r1 + 1):
        seq.append((r, c1))
    for c in range(c1 - 1, c0 - 1, -1):
        seq.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        seq.append((r, c0))
    return seq


def _candidate_next_cells(
    ordered: list[tuple[int, int]],
    panel_shape: tuple[int, int],
    dr: int,
    dc: int,
) -> set[tuple[int, int]]:
    height, width = map(int, panel_shape)
    n = len(ordered)
    candidates: set[tuple[int, int]] = set()
    for r0 in range(height - 2):
        for r1 in range(r0 + 2, height):
            for c0 in range(width - 2):
                for c1 in range(c0 + 2, width):
                    base = _rectangle_cycle(r0, c0, r1, c1)
                    for cycle in (base, list(reversed(base))):
                        m = len(cycle)
                        if n >= m:
                            continue
                        for start in range(m):
                            seq = cycle[start:] + cycle[:start]
                            if seq[:n] != ordered:
                                continue
                            candidate = seq[n]
                            frame5 = ordered + [candidate]
                            if all(
                                0 <= r + 4 * dr < height and 0 <= c + 4 * dc < width
                                for r, c in frame5
                            ):
                                candidates.add(candidate)
    return candidates


def _build_output(
    aligned_ordered: list[tuple[int, int]],
    next_cell: tuple[int, int],
    panel_shape: tuple[int, int],
    dr: int,
    dc: int,
    base_color: int,
) -> np.ndarray:
    coords = set(aligned_ordered)
    coords.add(next_cell)
    moved = _translate_coords(coords, 4 * dr, 4 * dc)
    out = np.full(panel_shape, BG, dtype=int)
    color = RECOLOR if _has_hole(coords) else base_color
    for r, c in moved:
        out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Extrapolate the fifth panel of a translating growing path, then recolor it only if the extrapolated shape closes a hole.

    Input structure:
    - The grid is a vertical strip of four equal panels separated by full-width solid rows.
    - Each panel contains one 4-connected monochrome foreground object over a black background.
    - Across the four panels, the object shifts by a constant offset and gains exactly one new 4-neighbor pixel per frame.

    Rule:
    - Split the strip into the four observed panels and extract the foreground object in each one.
    - Infer the unique constant translation whose shifted object is contained in the next frame with exactly one extra pixel.
    - Undo that translation across time so the four shapes become nested prefixes of one simple path, then identify the ordered path and the three observed growth pixels.
    - Find the unique next pixel that lets the ordered path continue as a contiguous segment of an axis-aligned rectangle perimeter, translate the resulting fifth shape forward one more step, and output that panel.
    - If the extrapolated fifth shape encloses a 4-connected background hole, recolor the whole object red; otherwise keep its original color.

    Key insight:
    - The color decision cannot be read from the observed panels alone: you must first align the moving sequence, continue the one-pixel perimeter growth, and only then test whether the new frame has become a closed ring.
    """
    g = np.asarray(grid, dtype=int)
    panels, bg, sep_color = _split_panels(g)
    assert bg == BG, "expected black background"
    assert sep_color == SEP, "expected yellow separator rows"

    frames: list[set[tuple[int, int]]] = []
    colors: list[int] = []
    for panel in panels:
        coords, color = _coords_from_panel(panel, bg)
        assert _is_connected(coords), "each panel object must be 4-connected"
        assert not _has_hole(coords), "observed panels must not already contain a hole"
        frames.append(coords)
        colors.append(color)
    assert len(set(colors)) == 1, "the observed object color must stay constant"

    dr, dc = _infer_shift(frames)
    aligned = [_translate_coords(frame, -i * dr, -i * dc) for i, frame in enumerate(frames)]
    additions = []
    for i in range(3):
        assert aligned[i] < aligned[i + 1], "aligned shapes must grow monotonically"
        delta = aligned[i + 1] - aligned[i]
        assert len(delta) == 1, "each frame must add exactly one pixel"
        additions.append(next(iter(delta)))

    deg = _degrees(aligned[-1])
    endpoints = [cell for cell, degree in deg.items() if degree == 1]
    assert len(endpoints) == 2, "the fourth aligned frame must be an open path"
    assert additions[-1] in endpoints, "the latest growth pixel must be the growing endpoint"
    start = endpoints[0] if endpoints[1] == additions[-1] else endpoints[1]
    ordered = _walk_path(aligned[-1], start, additions[-1])
    assert ordered[-3:] == additions, "growth must continue at the same path endpoint"
    for i, shape in enumerate(aligned):
        assert shape == set(ordered[: len(shape)]), f"aligned frame {i + 1} must be a path prefix"

    next_cells = _candidate_next_cells(ordered, panels[0].shape, dr, dc)
    assert len(next_cells) == 1, "expected a unique extrapolated growth cell"
    next_cell = next(iter(next_cells))
    return _build_output(ordered, next_cell, panels[0].shape, dr, dc, colors[0])


def _stack_frames(frames: list[np.ndarray]) -> np.ndarray:
    sep = np.full((1, frames[0].shape[1]), SEP, dtype=int)
    rows = []
    for frame in frames[:-1]:
        rows.append(frame)
        rows.append(sep.copy())
    rows.append(frames[-1])
    return np.vstack(rows)


def _render_panel(coords: set[tuple[int, int]], shape: tuple[int, int], color: int) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for r, c in coords:
        out[r, c] = int(color)
    return out


def _rotated_cycle(seq: list[tuple[int, int]], start: int, reverse: bool) -> list[tuple[int, int]]:
    ordered = list(reversed(seq)) if reverse else list(seq)
    return ordered[start:] + ordered[:start]


def _sample_case() -> tuple[np.ndarray, np.ndarray] | None:
    panel_h = random.randint(*PANEL_HEIGHT_RANGE)
    panel_w = random.randint(*PANEL_WIDTH_RANGE)
    dc = random.choice([-1, 0, 1])
    dr = 0

    max_rect_w = panel_w - 4 * abs(dc)
    if max_rect_w < 3:
        return None

    rect_h = random.randint(3, panel_h)
    rect_w = random.randint(3, max_rect_w)
    perimeter = 2 * (rect_h + rect_w) - 4
    close_hole = random.random() < 0.5

    if close_hole:
        final_len = perimeter
    else:
        low = max(7, perimeter - 3)
        if low > perimeter - 1:
            return None
        final_len = random.randint(low, perimeter - 1)
    first_len = final_len - 4
    if first_len < 3:
        return None

    r0 = random.randint(0, panel_h - rect_h)
    min_shift = min(0, dc, 2 * dc, 3 * dc, 4 * dc)
    max_shift = max(0, dc, 2 * dc, 3 * dc, 4 * dc)
    c_lo = -min_shift
    c_hi = panel_w - rect_w - max_shift
    if c_lo > c_hi:
        return None
    c0 = random.randint(c_lo, c_hi)

    base = _rectangle_cycle(0, 0, rect_h - 1, rect_w - 1)
    start = random.randrange(len(base))
    reverse = random.random() < 0.5
    cycle = _rotated_cycle(base, start, reverse)

    prefixes = [set(cycle[: first_len + i]) for i in range(5)]
    if any(_has_hole(prefix) for prefix in prefixes[:4]):
        return None
    if not all(_is_connected(prefix) for prefix in prefixes):
        return None

    actual_frames = []
    for t in range(4):
        coords = {(r + r0, c + c0 + t * dc) for r, c in prefixes[t]}
        if not all(0 <= r < panel_h and 0 <= c < panel_w for r, c in coords):
            return None
        actual_frames.append(coords)

    output_coords = {(r + r0, c + c0 + 4 * dc) for r, c in prefixes[4]}
    if not all(0 <= r < panel_h and 0 <= c < panel_w for r, c in output_coords):
        return None

    frames_np = [_render_panel(coords, (panel_h, panel_w), OBJ) for coords in actual_frames]
    inp = _stack_frames(frames_np)
    out_color = RECOLOR if _has_hole(prefixes[4]) else OBJ
    expected = _render_panel(output_coords, (panel_h, panel_w), out_color)
    if not _validate_case(inp, expected, dr, dc):
        return None
    return inp, expected


def _validate_case(inp: np.ndarray, expected: np.ndarray, dr: int, dc: int) -> bool:
    try:
        solved = main(inp)
    except Exception:
        return False
    if not np.array_equal(solved, expected):
        return False

    panels, bg, sep_color = _split_panels(inp)
    if bg != BG or sep_color != SEP:
        return False
    frames = []
    for panel in panels:
        coords, color = _coords_from_panel(panel, bg)
        if color != OBJ or not _is_connected(coords) or _has_hole(coords):
            return False
        frames.append(coords)

    if _infer_shift(frames) != (dr, dc):
        return False

    aligned = [_translate_coords(frame, -i * dr, -i * dc) for i, frame in enumerate(frames)]
    deltas = [aligned[i + 1] - aligned[i] for i in range(3)]
    if any(len(delta) != 1 for delta in deltas):
        return False

    deg = _degrees(aligned[-1])
    endpoints = [cell for cell, degree in deg.items() if degree == 1]
    if len(endpoints) != 2:
        return False
    tip = next(iter(deltas[-1]))
    if tip not in endpoints:
        return False
    start = endpoints[0] if endpoints[1] == tip else endpoints[1]
    ordered = _walk_path(aligned[-1], start, tip)
    next_cells = _candidate_next_cells(ordered, panels[0].shape, dr, dc)
    if len(next_cells) != 1:
        return False

    out_coords = {(int(r), int(c)) for r, c in np.argwhere(expected != BG)}
    expected_aligned = _translate_coords(out_coords, -4 * dr, -4 * dc)
    if len(expected_aligned - set(ordered)) != 1:
        return False
    if next(iter(expected_aligned - set(ordered))) not in next_cells:
        return False

    expected_has_hole = _has_hole(expected_aligned)
    colors = sorted({int(v) for v in np.unique(expected) if int(v) != BG})
    if len(colors) != 1:
        return False
    if expected_has_hole != (colors[0] == RECOLOR):
        return False
    if (not expected_has_hole) and colors[0] != OBJ:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        sampled = _sample_case()
        if sampled is not None:
            return sampled[0]
    raise RuntimeError("failed to generate a valid sequence-extrapolation ARC input")
