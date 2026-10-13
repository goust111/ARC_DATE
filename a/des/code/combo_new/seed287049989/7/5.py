from __future__ import annotations

from collections import Counter
import random

import numpy as np


BACKGROUND = 0
SHAPE_SIZE_MIN = 3
SHAPE_SIZE_MAX = 5
PANEL_SIZE_MIN = 8
PANEL_SIZE_MAX = 14
GEN_RETRIES = 400
SHAPE_RETRIES = 200
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
PAIR_MARKER_CHOICES = (("V", "H"), ("V", "D"), ("H", "D"))


def _mode(values: np.ndarray) -> int:
    counts = Counter(map(int, np.asarray(values).ravel()))
    assert counts, "expected at least one value"
    return int(counts.most_common(1)[0][0])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size, "expected a non-empty mask"
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _find_separator(g: np.ndarray) -> tuple[int, int, int]:
    h, w = g.shape
    candidates: list[tuple[int, int, int]] = []
    for color in np.unique(g):
        full_rows = np.where(np.all(g == int(color), axis=1))[0]
        full_cols = np.where(np.all(g == int(color), axis=0))[0]
        if full_rows.size != 1 or full_cols.size != 1:
            continue
        sr = int(full_rows[0])
        sc = int(full_cols[0])
        if sr in (0, h - 1) or sc in (0, w - 1):
            continue
        if sr != h - sr - 1 or sc != w - sc - 1:
            continue
        candidates.append((int(color), sr, sc))
    assert len(candidates) == 1, "expected exactly one full-row/full-column separator"
    return candidates[0]


def _panel_background(panel: np.ndarray) -> int:
    border = np.concatenate(
        [panel[0, :], panel[-1, :], panel[1:-1, 0], panel[1:-1, -1]]
    )
    return _mode(border)


def _transform_panel(panel: np.ndarray) -> np.ndarray:
    bg = _panel_background(panel)
    colors = [int(c) for c in np.unique(panel) if int(c) != int(bg)]
    assert len(colors) == 2, "each panel must contain one pattern color and one marker color"

    counts = {color: int(np.count_nonzero(panel == color)) for color in colors}
    marker_candidates = [color for color, count in counts.items() if count in (1, 2)]
    pattern_candidates = [color for color, count in counts.items() if count > 2]
    assert len(marker_candidates) == 1 and len(pattern_candidates) == 1, "panel colors must separate into pattern and markers"

    marker_color = int(marker_candidates[0])
    pattern_color = int(pattern_candidates[0])
    pattern_mask = panel == pattern_color
    marker_mask = panel == marker_color
    marker_coords = np.argwhere(marker_mask)
    assert 1 <= marker_coords.shape[0] <= 2, "expected 1 or 2 marker pixels"

    r0, c0, r1, c1 = _bbox(pattern_mask)
    out = np.array(panel, copy=True)

    for mr, mc in marker_coords:
        row_inside = r0 <= int(mr) <= r1
        col_inside = c0 <= int(mc) <= c1

        if col_inside and not row_inside:
            col_rows = np.where(pattern_mask[:, int(mc)])[0]
            out[col_rows, int(mc)] = marker_color
        elif row_inside and not col_inside:
            row_cols = np.where(pattern_mask[int(mr), :])[0]
            out[int(mr), row_cols] = marker_color

    out[marker_mask] = bg
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor panel-local pattern slices according to how minority marker pixels sit around the majority shape.

    Input structure:
    - One full separator row and one full separator column split the grid into four equal rectangular panels.
    - Each panel has a black border-adjacent background, one connected small pattern in a majority foreground color, and 1-2 single-pixel markers in a second foreground color.
    - The four panels reuse the same shape but with different pattern colors, while the marker color is shared across panels.

    Rule:
    - Find the separator row and column, then process each panel independently.
    - In a panel, treat the larger foreground color as the pattern, the color with 1 or 2 pixels as the marker, and compute the pattern's bounding box.
    - For each marker: if its column lies inside the bbox but its row is above or below it, recolor the pattern pixels in that column to the marker color; if its row lies inside the bbox but its column is left or right of it, recolor the pattern pixels in that row; diagonal markers do nothing.
    - After all marker effects are applied, erase the marker pixels to background and keep the separator, background, and untouched pattern pixels unchanged.

    Key insight:
    - The marker does not name a specific cell; its orientation relative to the pattern bbox selects an axis, projecting either a colored row, a colored column, or no change at all.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _separator_color, sr, sc = _find_separator(g)
    out = np.array(g, copy=True)

    panel_slices = [
        (slice(0, sr), slice(0, sc)),
        (slice(0, sr), slice(sc + 1, g.shape[1])),
        (slice(sr + 1, g.shape[0]), slice(0, sc)),
        (slice(sr + 1, g.shape[0]), slice(sc + 1, g.shape[1])),
    ]

    for rs, cs in panel_slices:
        panel = g[rs, cs]
        out[rs, cs] = _transform_panel(panel)

    return out


def _shape_with_full_bbox() -> np.ndarray:
    for _ in range(SHAPE_RETRIES):
        h = random.randint(SHAPE_SIZE_MIN, SHAPE_SIZE_MAX)
        w = random.randint(SHAPE_SIZE_MIN, SHAPE_SIZE_MAX)
        min_area = max(5, h + w - 1)
        area = random.randint(min_area, h * w)

        cells = {(0, 0)}
        r = c = 0
        moves = ["D"] * (h - 1) + ["R"] * (w - 1)
        random.shuffle(moves)
        for move in moves:
            if move == "D":
                r += 1
            else:
                c += 1
            cells.add((r, c))

        while len(cells) < area:
            frontier = set()
            for rr, cc in cells:
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))

        if len(cells) != area:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for rr, cc in cells:
            mask[rr, cc] = True
        if not np.all(mask.any(axis=0)) or not np.all(mask.any(axis=1)):
            continue

        if random.random() < 0.5:
            mask = np.flipud(mask)
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        if random.random() < 0.35:
            mask = mask.T
        return mask

    raise RuntimeError("failed to build a connected panel shape")


def _single_marker(panel_h: int, panel_w: int, bbox_rc: tuple[int, int, int, int], category: str) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox_rc
    if category == "V":
        col = random.randint(c0, c1)
        if random.random() < 0.5:
            return random.randint(0, r0 - 1), col
        return random.randint(r1 + 1, panel_h - 1), col
    if category == "H":
        row = random.randint(r0, r1)
        if random.random() < 0.5:
            return row, random.randint(0, c0 - 1)
        return row, random.randint(c1 + 1, panel_w - 1)

    vertical_side = random.choice(("above", "below"))
    horizontal_side = random.choice(("left", "right"))
    row = random.randint(0, r0 - 1) if vertical_side == "above" else random.randint(r1 + 1, panel_h - 1)
    col = random.randint(0, c0 - 1) if horizontal_side == "left" else random.randint(c1 + 1, panel_w - 1)
    return row, col


def _marker_positions(panel_h: int, panel_w: int, bbox_rc: tuple[int, int, int, int], categories: list[str]) -> list[tuple[int, int]]:
    markers: list[tuple[int, int]] = []
    for category in categories:
        marker = _single_marker(panel_h, panel_w, bbox_rc, category)
        assert marker not in markers, "markers must stay distinct"
        markers.append(marker)
    return markers


def _panel_from_spec(
    panel_h: int,
    panel_w: int,
    shape_mask: np.ndarray,
    pattern_color: int,
    marker_color: int,
    categories: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    sh, sw = shape_mask.shape
    top = random.randint(1, panel_h - sh - 1)
    left = random.randint(1, panel_w - sw - 1)
    bbox_rc = (top, left, top + sh - 1, left + sw - 1)
    markers = _marker_positions(panel_h, panel_w, bbox_rc, categories)

    panel = np.full((panel_h, panel_w), BACKGROUND, dtype=int)
    shape_view = panel[top : top + sh, left : left + sw]
    shape_view[shape_mask] = pattern_color
    for mr, mc in markers:
        panel[int(mr), int(mc)] = marker_color

    expected = panel.copy()
    pattern_mask = np.zeros((panel_h, panel_w), dtype=bool)
    pattern_mask[top : top + sh, left : left + sw] = shape_mask
    r0, c0, r1, c1 = bbox_rc

    for mr, mc in markers:
        row_inside = r0 <= mr <= r1
        col_inside = c0 <= mc <= c1
        if col_inside and not row_inside:
            rows = np.where(pattern_mask[:, mc])[0]
            expected[rows, mc] = marker_color
        elif row_inside and not col_inside:
            cols = np.where(pattern_mask[mr, :])[0]
            expected[mr, cols] = marker_color
    for mr, mc in markers:
        expected[mr, mc] = BACKGROUND

    return panel, expected


def _gen_attempt() -> np.ndarray | None:
    separator_color, marker_color, *pattern_colors = random.sample(range(1, 10), 6)
    shape_mask = _shape_with_full_bbox()
    sh, sw = shape_mask.shape

    panel_h = random.randint(max(PANEL_SIZE_MIN, sh + 2), PANEL_SIZE_MAX)
    panel_w = random.randint(max(PANEL_SIZE_MIN, sw + 2), PANEL_SIZE_MAX)
    grid_h = panel_h * 2 + 1
    grid_w = panel_w * 2 + 1
    assert grid_h <= 30 and grid_w <= 30

    categories_per_panel: list[list[str]] = [
        ["V"],
        ["H"],
        ["D"],
        list(random.choice(PAIR_MARKER_CHOICES)),
    ]
    random.shuffle(categories_per_panel)

    inp = np.full((grid_h, grid_w), BACKGROUND, dtype=int)
    expected = np.full((grid_h, grid_w), BACKGROUND, dtype=int)
    sep_r = panel_h
    sep_c = panel_w
    inp[sep_r, :] = separator_color
    inp[:, sep_c] = separator_color
    expected[sep_r, :] = separator_color
    expected[:, sep_c] = separator_color

    panel_slices = [
        (slice(0, panel_h), slice(0, panel_w)),
        (slice(0, panel_h), slice(panel_w + 1, grid_w)),
        (slice(panel_h + 1, grid_h), slice(0, panel_w)),
        (slice(panel_h + 1, grid_h), slice(panel_w + 1, grid_w)),
    ]

    for (rs, cs), pattern_color, categories in zip(panel_slices, pattern_colors, categories_per_panel):
        panel_inp, panel_expected = _panel_from_spec(
            panel_h=panel_h,
            panel_w=panel_w,
            shape_mask=shape_mask,
            pattern_color=pattern_color,
            marker_color=marker_color,
            categories=categories,
        )
        inp[rs, cs] = panel_inp
        expected[rs, cs] = panel_expected

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
