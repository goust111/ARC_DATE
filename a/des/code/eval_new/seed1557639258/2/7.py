from __future__ import annotations

from collections import Counter
import random

import numpy as np


ANCHOR_ORDER = (1, 2, 3, 4)
ARRANGEMENTS = ((1, 2), (2, 1), (1, 3), (3, 1), (2, 2))
CORNER_KINDS = ("tl", "tr", "bl", "br")
ARC_LIMIT = 30
MIN_PANEL_SIZE = 7
MAX_PANEL_SIZE = 11
MAX_GEN_ATTEMPTS = 400


def _group_runs(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    runs: list[tuple[int, int]] = []
    start = prev = int(indices[0])
    for idx in indices[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
            continue
        runs.append((start, prev))
        start = prev = idx
    runs.append((start, prev))
    return runs


def _segments(size: int, bands: list[tuple[int, int]]) -> list[tuple[int, int]]:
    pieces: list[tuple[int, int]] = []
    start = 0
    for b0, b1 in bands:
        if start <= b0 - 1:
            pieces.append((start, b0 - 1))
        start = b1 + 1
    if start <= size - 1:
        pieces.append((start, size - 1))
    return pieces


def _full_color_bands(grid: np.ndarray, color: int, axis: int) -> list[tuple[int, int]]:
    if axis == 0:
        indices = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == color)]
    else:
        indices = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == color)]
    return _group_runs(indices)


def _is_corner_anchor(r: int, c: int, h: int, w: int) -> bool:
    return min(r, c, h - 1 - r, w - 1 - c) <= 2


def _parse_panels(grid: np.ndarray) -> tuple[int, list[dict[str, object]], int, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[tuple[int, list[dict[str, object]], int, int]] = []

    for sep_color in sorted(int(v) for v in np.unique(g)):
        row_bands = _full_color_bands(g, sep_color, axis=0)
        col_bands = _full_color_bands(g, sep_color, axis=1)
        if not row_bands and not col_bands:
            continue

        row_parts = _segments(h, row_bands)
        col_parts = _segments(w, col_bands)
        panel_count = len(row_parts) * len(col_parts)
        if panel_count < 2 or panel_count > len(ANCHOR_ORDER):
            continue

        row_heights = {r1 - r0 + 1 for r0, r1 in row_parts}
        col_widths = {c1 - c0 + 1 for c0, c1 in col_parts}
        if len(row_heights) != 1 or len(col_widths) != 1:
            continue
        if min(row_heights) < 4 or min(col_widths) < 4:
            continue

        row_band_widths = {r1 - r0 + 1 for r0, r1 in row_bands}
        col_band_widths = {c1 - c0 + 1 for c0, c1 in col_bands}
        if row_band_widths and len(row_band_widths) != 1:
            continue
        if col_band_widths and len(col_band_widths) != 1:
            continue

        non_sep = np.ones_like(g, dtype=bool)
        for r0, r1 in row_bands:
            non_sep[r0 : r1 + 1, :] = False
        for c0, c1 in col_bands:
            non_sep[:, c0 : c1 + 1] = False
        vals = g[non_sep]
        if vals.size == 0:
            continue
        bg = Counter(int(v) for v in vals.tolist()).most_common(1)[0][0]

        panel_h = next(iter(row_heights))
        panel_w = next(iter(col_widths))
        panels: list[dict[str, object]] = []
        seen_anchor_colors: set[int] = set()
        ok = True

        for row_range in row_parts:
            for col_range in col_parts:
                r0, r1 = row_range
                c0, c1 = col_range
                panel = g[r0 : r1 + 1, c0 : c1 + 1]
                anchor_cells: list[tuple[int, int, int]] = []
                for pr in range(panel_h):
                    for pc in range(panel_w):
                        val = int(panel[pr, pc])
                        if val in ANCHOR_ORDER and _is_corner_anchor(pr, pc, panel_h, panel_w):
                            anchor_cells.append((pr, pc, val))
                if len(anchor_cells) != 1:
                    ok = False
                    break
                ar, ac, anchor_color = anchor_cells[0]
                if anchor_color in seen_anchor_colors or anchor_color == bg:
                    ok = False
                    break
                seen_anchor_colors.add(anchor_color)
                panels.append(
                    {
                        "panel": panel,
                        "anchor": (ar, ac),
                        "anchor_color": anchor_color,
                    }
                )
            if not ok:
                break
        if not ok:
            continue

        candidates.append((bg, panels, panel_h, panel_w))

    assert len(candidates) == 1, "input must contain one unambiguous panel decomposition"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay equal-sized panel patterns by translating every panel so its corner anchor lands at the output's top-left cell.

    Input structure:
    - One separator color forms full-width and/or full-height bands that partition the grid into equal-sized rectangular panels.
    - Inside each panel, most cells are a shared background color, plus a sparse colored pattern.
    - Each panel also contains exactly one special single-pixel anchor from a fixed ordered anchor palette, placed near one corner and unique across panels.

    Rule:
    - Treat each panel as a transparent layer and extract its unique anchor pixel.
    - Remove the anchor pixel, then translate the remaining non-background cells by the negative anchor offset so the anchor would move to the output's top-left cell.
    - Overlay the translated layers onto a background canvas the same size as one panel; later anchor-palette colors cover earlier ones.

    Key insight:
    - The separator bands reveal the repeated panel grid, the anchor position encodes the per-layer translation, and the fixed anchor palette order encodes the z-order independently of geometry.
    """
    g = np.asarray(grid, dtype=int)
    bg, panels, panel_h, panel_w = _parse_panels(g)

    out = np.full((panel_h, panel_w), bg, dtype=int)
    ordered = sorted(panels, key=lambda item: ANCHOR_ORDER.index(int(item["anchor_color"])))

    for item in ordered:
        panel = np.asarray(item["panel"], dtype=int)
        ar, ac = item["anchor"]
        for r in range(panel_h):
            for c in range(panel_w):
                if r == ar and c == ac:
                    continue
                val = int(panel[r, c])
                if val == bg:
                    continue
                rr = r - int(ar)
                cc = c - int(ac)
                if 0 <= rr < panel_h and 0 <= cc < panel_w:
                    out[rr, cc] = val
    return out


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _sample_panel_size(rows: int, cols: int, sep: int) -> tuple[int, int]:
    max_h = min(MAX_PANEL_SIZE, (ARC_LIMIT - (rows - 1) * sep) // rows)
    max_w = min(MAX_PANEL_SIZE, (ARC_LIMIT - (cols - 1) * sep) // cols)
    assert max_h >= MIN_PANEL_SIZE and max_w >= MIN_PANEL_SIZE
    return random.randint(MIN_PANEL_SIZE, max_h), random.randint(MIN_PANEL_SIZE, max_w)


def _anchor_position(kind: str, panel_h: int, panel_w: int) -> tuple[int, int]:
    top_rows = [0, 1, 2]
    left_cols = [0, 1, 2]
    bottom_rows = [panel_h - 4, panel_h - 3, panel_h - 2]
    right_cols = [panel_w - 4, panel_w - 3, panel_w - 2]
    if kind == "tl":
        return random.choice(top_rows), random.choice(left_cols)
    if kind == "tr":
        return random.choice(top_rows), random.choice(right_cols)
    if kind == "bl":
        return random.choice(bottom_rows), random.choice(left_cols)
    if kind == "br":
        return random.choice(bottom_rows), random.choice(right_cols)
    raise ValueError(kind)


def _grow_shape(height: int, width: int, start: tuple[int, int], area: int) -> set[tuple[int, int]]:
    forbidden = {(0, 0)}
    cells = {start}
    frontier = {
        nxt
        for nxt in _neighbors4(*start)
        if 0 <= nxt[0] < height and 0 <= nxt[1] < width and nxt not in forbidden
    }
    while len(cells) < area and frontier:
        nxt = random.choice(sorted(frontier))
        frontier.remove(nxt)
        cells.add(nxt)
        for nb in _neighbors4(*nxt):
            if 0 <= nb[0] < height and 0 <= nb[1] < width and nb not in cells and nb not in forbidden:
                frontier.add(nb)
    return cells


def _make_layer_shape(height: int, width: int, shared: tuple[int, int]) -> set[tuple[int, int]]:
    free = height * width - 1
    max_area = min(8, free)
    min_area = min(max_area, 3)
    area = random.randint(min_area, max_area)
    cells = _grow_shape(height, width, shared, area)
    if len(cells) < area:
        extras = [(r, c) for r in range(height) for c in range(width) if (r, c) != (0, 0) and (r, c) not in cells]
        random.shuffle(extras)
        for cell in extras:
            cells.add(cell)
            if len(cells) == area:
                break
    return cells


def _build_grid(
    rows: int,
    cols: int,
    panel_h: int,
    panel_w: int,
    sep: int,
    bg: int,
    sep_color: int,
    anchor_colors: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    kinds = [random.choice(CORNER_KINDS) for _ in range(rows * cols)]
    if len(set(kinds)) == 1:
        kinds[-1] = random.choice([kind for kind in CORNER_KINDS if kind != kinds[0]])

    anchors = [_anchor_position(kind, panel_h, panel_w) for kind in kinds]
    avail_h = [panel_h - ar for ar, _ac in anchors]
    avail_w = [panel_w - ac for _ar, ac in anchors]
    common_h = random.randint(2, min(6, min(avail_h)))
    common_w = random.randint(2, min(6, min(avail_w)))

    shared_choices = [(r, c) for r in range(common_h) for c in range(common_w) if (r, c) != (0, 0)]
    shared_cell = random.choice(shared_choices)

    blocked = {bg, sep_color, *ANCHOR_ORDER}
    shape_palette = [c for c in range(10) if c not in blocked]
    main_colors = random.sample(shape_palette, rows * cols)

    total_h = rows * panel_h + (rows - 1) * sep
    total_w = cols * panel_w + (cols - 1) * sep
    grid = np.full((total_h, total_w), sep_color, dtype=int)
    out = np.full((panel_h, panel_w), bg, dtype=int)

    order_index = {color: idx for idx, color in enumerate(ANCHOR_ORDER)}
    placement_data: list[tuple[int, np.ndarray]] = []

    for idx in range(rows * cols):
        ar, ac = anchors[idx]
        shape = _make_layer_shape(common_h, common_w, shared_cell)
        panel = np.full((panel_h, panel_w), bg, dtype=int)
        panel[ar, ac] = anchor_colors[idx]

        layer_color = main_colors[idx]
        accent_candidates = [c for c in shape_palette if c != layer_color]
        accent_color = random.choice(accent_candidates) if accent_candidates and random.random() < 0.4 else None
        accent_cells = set()
        if accent_color is not None:
            non_shared = [cell for cell in shape if cell != shared_cell]
            if non_shared:
                accent_cells = set(random.sample(non_shared, k=random.randint(1, min(2, len(non_shared)))))

        layer = np.full((panel_h, panel_w), bg, dtype=int)
        for rr, cc in shape:
            color = accent_color if (rr, cc) in accent_cells else layer_color
            panel[ar + rr, ac + cc] = color
            layer[rr, cc] = color
        placement_data.append((order_index[anchor_colors[idx]], layer))

        row_id = idx // cols
        col_id = idx % cols
        r0 = row_id * (panel_h + sep)
        c0 = col_id * (panel_w + sep)
        grid[r0 : r0 + panel_h, c0 : c0 + panel_w] = panel

    placement_data.sort(key=lambda item: item[0])
    for _rank, layer in placement_data:
        mask = layer != bg
        out[mask] = layer[mask]

    return grid, out


def _valid_counts(grid: np.ndarray, bg: int, sep_color: int) -> bool:
    counts = Counter(int(v) for v in grid.ravel().tolist())
    return counts[bg] > counts[sep_color]


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        rows, cols = random.choice(ARRANGEMENTS)
        sep = random.randint(1, 2)
        panel_h, panel_w = _sample_panel_size(rows, cols, sep)
        n_panels = rows * cols

        anchor_colors = random.sample(list(ANCHOR_ORDER), n_panels)
        color_pool = [c for c in range(10) if c not in ANCHOR_ORDER]
        bg, sep_color = random.sample(color_pool, 2)

        grid, expected = _build_grid(rows, cols, panel_h, panel_w, sep, bg, sep_color, anchor_colors)
        if max(grid.shape) > ARC_LIMIT or max(expected.shape) > ARC_LIMIT:
            continue
        if not _valid_counts(grid, bg, sep_color):
            continue
        try:
            actual = main(grid)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid input")
