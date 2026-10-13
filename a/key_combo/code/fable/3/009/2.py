from __future__ import annotations

from collections import Counter
import random

import numpy as np


CYAN = 8
ARC_COLORS = tuple(range(10))
INPUT_COLORS = tuple(c for c in ARC_COLORS if c != CYAN)


def _find_separator_row(g: np.ndarray) -> tuple[int, int]:
    rows, cols = map(int, g.shape)
    assert rows % 2 == 1, "two equal panels require one middle separator row"

    candidates: list[tuple[int, int]] = []
    for r in range(1, rows - 1):
        color = int(g[r, 0])
        if (
            r == rows - r - 1
            and bool(np.all(g[r, :] == color))
            and int(np.count_nonzero(g == color)) == cols
        ):
            candidates.append((r, color))

    assert len(candidates) == 1, "expected exactly one full-width separator row"
    return candidates[0]


def _panel_background(top: np.ndarray, bottom: np.ndarray, separator: int) -> int:
    counts = Counter(map(int, np.concatenate((top.ravel(), bottom.ravel()))))
    assert counts, "panels must be non-empty"
    bg, _count = counts.most_common(1)[0]
    assert int(bg) != int(separator), "separator color cannot be the panel background"
    return int(bg)


def _rare_anchor(panel: np.ndarray, bg: int) -> tuple[int, tuple[int, int]]:
    counts = Counter(map(int, panel.ravel()))
    rare_colors = [color for color, count in counts.items() if color != int(bg) and count == 1]
    assert len(rare_colors) == 1, "each panel must have exactly one non-background color appearing once"

    color = int(rare_colors[0])
    coords = np.argwhere(panel == color)
    assert coords.shape == (1, 2)
    r, c = map(int, coords[0])
    return color, (r, c)


def _shift_foreground(panel: np.ndarray, bg: int, dr: int, dc: int) -> np.ndarray:
    h, w = map(int, panel.shape)
    shifted = np.full_like(panel, int(bg))
    for r, c in np.argwhere(panel != int(bg)):
        rr = int(r) + int(dr)
        cc = int(c) + int(dc)
        if 0 <= rr < h and 0 <= cc < w:
            shifted[rr, cc] = int(panel[r, c])
    return shifted


def _overlay_panels(top: np.ndarray, shifted_bottom: np.ndarray, bg: int) -> np.ndarray:
    top_fg = top != int(bg)
    bottom_fg = shifted_bottom != int(bg)

    out = np.full_like(top, int(bg))
    only_top = top_fg & ~bottom_fg
    only_bottom = ~top_fg & bottom_fg
    both = top_fg & bottom_fg
    same = both & (top == shifted_bottom)
    conflict = both & (top != shifted_bottom)

    out[only_top] = top[only_top]
    out[only_bottom] = shifted_bottom[only_bottom]
    out[same] = top[same]
    out[conflict] = CYAN
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Register two separated patterns by their panel-rare anchor pixels, then transparently overlay them.

    Input structure:
    - One solid full-width separator row splits the grid into two equal-height panels, and the separator color occurs only in that row.
    - The two panels share one background color and contain different multicolor foreground patterns.
    - In each panel exactly one non-background color appears in a single pixel; cyan is absent from the input.

    Rule:
    - Remove the separator row and find each panel's unique-frequency foreground color pixel.
    - Translate the lower panel so its rare pixel lands on the upper panel's rare pixel, discarding translated cells outside the panel.
    - Overlay cell by cell: keep a single foreground color, keep matching double foreground colors, recolor conflicting double foreground cells cyan, and leave double background cells as background.

    Key insight:
    - The anchors are identified only by within-panel color frequency, so the alignment comes from rare-color registration rather than borders, centroids, or a fixed marker color.
    """
    g = np.asarray(grid, dtype=int)
    sep_r, sep_color = _find_separator_row(g)
    top = g[:sep_r, :].copy()
    bottom = g[sep_r + 1 :, :].copy()
    assert top.shape == bottom.shape and top.size > 0
    assert not np.any(top == CYAN) and not np.any(bottom == CYAN)

    bg = _panel_background(top, bottom, sep_color)
    _top_color, top_anchor = _rare_anchor(top, bg)
    _bottom_color, bottom_anchor = _rare_anchor(bottom, bg)

    dr = int(top_anchor[0] - bottom_anchor[0])
    dc = int(top_anchor[1] - bottom_anchor[1])
    shifted_bottom = _shift_foreground(bottom, bg, dr, dc)
    return _overlay_panels(top, shifted_bottom, bg)


def _target_to_source(cell: tuple[int, int], dr: int, dc: int) -> tuple[int, int]:
    return int(cell[0] - dr), int(cell[1] - dc)


def _source_to_target(cell: tuple[int, int], dr: int, dc: int) -> tuple[int, int]:
    return int(cell[0] + dr), int(cell[1] + dc)


def _in_bounds(cell: tuple[int, int], h: int, w: int) -> bool:
    r, c = map(int, cell)
    return 0 <= r < h and 0 <= c < w


def _sample_delta(h: int, w: int) -> tuple[int, int]:
    choices = []
    for dr in range(-4, 5):
        for dc in range(-4, 5):
            if dr == 0 and dc == 0:
                continue
            if h - abs(dr) >= 6 and w - abs(dc) >= 6:
                choices.append((dr, dc))
    return random.choice(choices)


def _intersection_cells(h: int, w: int, dr: int, dc: int) -> list[tuple[int, int]]:
    cells = []
    for r in range(h):
        for c in range(w):
            if _in_bounds(_target_to_source((r, c), dr, dc), h, w):
                cells.append((r, c))
    return cells


def _nearby_cells(cells: list[tuple[int, int]], anchor: tuple[int, int], needed: int) -> list[tuple[int, int]]:
    ar, ac = anchor
    near = [
        cell
        for cell in cells
        if cell != anchor and abs(cell[0] - ar) + abs(cell[1] - ac) <= 5
    ]
    if len(near) < needed:
        near = [cell for cell in cells if cell != anchor]
    random.shuffle(near)
    return near


def _sample_cells(
    candidates: list[tuple[int, int]],
    count: int,
    forbidden: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    pool = [cell for cell in candidates if cell not in forbidden]
    if len(pool) < count:
        return None
    return random.sample(pool, count)


def _choose_clipped_sources(
    h: int,
    w: int,
    dr: int,
    dc: int,
    forbidden: set[tuple[int, int]],
    count: int,
) -> list[tuple[int, int]] | None:
    candidates = []
    for r in range(h):
        for c in range(w):
            src = (r, c)
            if src in forbidden:
                continue
            if not _in_bounds(_source_to_target(src, dr, dc), h, w):
                candidates.append(src)
    if len(candidates) < count:
        return None
    random.shuffle(candidates)
    return candidates[:count]


def _assign(panel: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        panel[int(r), int(c)] = int(color)


def _color_plan() -> tuple[int, int, int, int, int, int, int, int, int]:
    colors = list(INPUT_COLORS)
    random.shuffle(colors)
    bg = colors.pop()
    sep = colors.pop()
    if random.random() < 0.25:
        rare_top = colors.pop()
        rare_bottom = rare_top
    else:
        rare_top = colors.pop()
        rare_bottom = colors.pop()

    shared = colors.pop()
    top_conflict = colors.pop()
    bottom_conflict = colors.pop()
    top_only = colors.pop()
    bottom_only = colors.pop()
    return bg, sep, rare_top, rare_bottom, shared, top_conflict, bottom_conflict, top_only, bottom_only


def _build_attempt() -> np.ndarray | None:
    h = random.randint(8, 14)
    w = random.randint(9, 21)
    dr, dc = _sample_delta(h, w)

    r_min = max(0, dr)
    r_max = min(h - 1, h - 1 + dr)
    c_min = max(0, dc)
    c_max = min(w - 1, w - 1 + dc)
    if r_min > r_max or c_min > c_max:
        return None
    top_anchor = (random.randint(r_min, r_max), random.randint(c_min, c_max))
    bottom_anchor = _target_to_source(top_anchor, dr, dc)

    intersection = _intersection_cells(h, w, dr, dc)
    near_intersection = _nearby_cells(intersection, top_anchor, needed=8)
    forbidden_targets = {top_anchor}

    same_cells = _sample_cells(near_intersection, 2, forbidden_targets)
    if same_cells is None:
        return None
    forbidden_targets.update(same_cells)

    conflict_cells = _sample_cells(near_intersection, 2, forbidden_targets)
    if conflict_cells is None:
        return None
    forbidden_targets.update(conflict_cells)

    bottom_only_targets = _sample_cells(near_intersection, 2, forbidden_targets)
    if bottom_only_targets is None:
        return None
    forbidden_targets.update(bottom_only_targets)

    all_panel_cells = [(r, c) for r in range(h) for c in range(w)]
    near_panel = _nearby_cells(all_panel_cells, top_anchor, needed=8)
    top_only_targets = _sample_cells(near_panel, 2, forbidden_targets)
    if top_only_targets is None:
        return None
    forbidden_targets.update(top_only_targets)

    used_bottom_sources = {bottom_anchor}
    for cell in same_cells + conflict_cells + bottom_only_targets:
        used_bottom_sources.add(_target_to_source(cell, dr, dc))

    clipped_sources = _choose_clipped_sources(h, w, dr, dc, used_bottom_sources, 2)
    if clipped_sources is None:
        return None
    used_bottom_sources.update(clipped_sources)

    extra_top_count = random.randint(0, 4)
    extra_top = _sample_cells(all_panel_cells, extra_top_count, forbidden_targets)
    if extra_top is None:
        return None
    forbidden_targets.update(extra_top)

    extra_bottom_count = random.randint(0, 4)
    extra_bottom_targets = _sample_cells(intersection, extra_bottom_count, forbidden_targets)
    if extra_bottom_targets is None:
        return None
    for cell in extra_bottom_targets:
        used_bottom_sources.add(_target_to_source(cell, dr, dc))

    extra_clipped_count = random.randint(0, 3)
    extra_clipped = _choose_clipped_sources(h, w, dr, dc, used_bottom_sources, extra_clipped_count)
    if extra_clipped is None:
        return None

    (
        bg,
        sep,
        rare_top,
        rare_bottom,
        shared,
        top_conflict,
        bottom_conflict,
        top_only,
        bottom_only,
    ) = _color_plan()

    top = np.full((h, w), bg, dtype=int)
    bottom = np.full((h, w), bg, dtype=int)

    top[top_anchor] = rare_top
    bottom[bottom_anchor] = rare_bottom
    _assign(top, same_cells, shared)
    _assign(bottom, [_target_to_source(cell, dr, dc) for cell in same_cells], shared)
    _assign(top, conflict_cells, top_conflict)
    _assign(bottom, [_target_to_source(cell, dr, dc) for cell in conflict_cells], bottom_conflict)
    _assign(top, top_only_targets + extra_top, top_only)
    _assign(bottom, [_target_to_source(cell, dr, dc) for cell in bottom_only_targets], bottom_only)
    _assign(bottom, [_target_to_source(cell, dr, dc) for cell in extra_bottom_targets], bottom_only)
    _assign(bottom, clipped_sources + extra_clipped, bottom_only)

    separator = np.full((1, w), sep, dtype=int)
    grid = np.vstack([top, separator, bottom])
    expected = _expected_from_parts(top, bottom, bg, dr, dc)
    if not _semantic_accepts(grid, expected, bg, sep, dr, dc):
        return None
    return grid


def _expected_from_parts(top: np.ndarray, bottom: np.ndarray, bg: int, dr: int, dc: int) -> np.ndarray:
    return _overlay_panels(top, _shift_foreground(bottom, bg, dr, dc), bg)


def _semantic_accepts(
    grid: np.ndarray,
    expected: np.ndarray,
    bg: int,
    sep: int,
    dr: int,
    dc: int,
) -> bool:
    h = (grid.shape[0] - 1) // 2
    top = grid[:h]
    bottom = grid[h + 1 :]
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False
    if np.any(grid == CYAN):
        return False
    if int(np.count_nonzero(grid == sep)) != grid.shape[1]:
        return False
    if (dr, dc) == (0, 0):
        return False
    if np.array_equal(top != bg, bottom != bg):
        return False

    try:
        top_rare, top_anchor = _rare_anchor(top, bg)
        bottom_rare, bottom_anchor = _rare_anchor(bottom, bg)
        solved = main(grid)
    except Exception:
        return False
    if top_rare == bg or bottom_rare == bg:
        return False
    if (top_anchor[0] - bottom_anchor[0], top_anchor[1] - bottom_anchor[1]) != (dr, dc):
        return False
    if not np.array_equal(solved, expected):
        return False

    shifted_bottom = _shift_foreground(bottom, bg, dr, dc)
    top_fg = top != bg
    bottom_fg = shifted_bottom != bg
    conflict = top_fg & bottom_fg & (top != shifted_bottom)
    same = top_fg & bottom_fg & (top == shifted_bottom)
    only_top = top_fg & ~bottom_fg
    only_bottom = ~top_fg & bottom_fg
    clipped = 0
    for r, c in np.argwhere(bottom != bg):
        if not _in_bounds((int(r) + dr, int(c) + dc), *bottom.shape):
            clipped += 1

    return (
        int(np.count_nonzero(conflict)) >= 2
        and int(np.count_nonzero(same)) >= 2
        and int(np.count_nonzero(only_top)) >= 2
        and int(np.count_nonzero(only_bottom)) >= 2
        and clipped >= 2
        and int(np.count_nonzero(expected == CYAN)) >= 2
    )


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid rare-anchor overlay input")
