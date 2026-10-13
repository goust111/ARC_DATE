import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_RETRIES = 300


def _spans_from_separators(mask: np.ndarray) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = None
    for i, is_sep in enumerate(mask.tolist()):
        if is_sep:
            if start is not None:
                spans.append((start, i))
                start = None
        elif start is None:
            start = i
    if start is not None:
        spans.append((start, len(mask)))
    return spans


def _panel_spans(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    row_seps = np.all(grid == GREY, axis=1)
    col_seps = np.all(grid == GREY, axis=0)
    row_spans = _spans_from_separators(row_seps)
    col_spans = _spans_from_separators(col_seps)
    assert len(row_spans) >= 2 and len(col_spans) >= 2

    panel_heights = {r1 - r0 for r0, r1 in row_spans}
    panel_widths = {c1 - c0 for c0, c1 in col_spans}
    assert len(panel_heights) == 1 and len(panel_widths) == 1
    assert next(iter(panel_heights)) == next(iter(panel_widths))
    return row_spans, col_spans


def _seed_info(panel: np.ndarray) -> tuple[list[tuple[int, int, int]], dict[int, tuple[int, int]]]:
    coords = np.argwhere(panel != BLACK)
    assert len(coords) > 1
    pattern: list[tuple[int, int, int]] = []
    anchors: dict[int, tuple[int, int]] = {}
    for rr, cc in coords:
        r, c = int(rr), int(cc)
        color = int(panel[r, c])
        assert color not in anchors
        anchors[color] = (r, c)
        pattern.append((r, c, color))
    return pattern, anchors


def main(grid: np.ndarray) -> np.ndarray:
    """
    In a separator-divided panel array, replicate the seed pattern into marker panels
    by aligning the seed pixel with matching color to the marker.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    row_spans, col_spans = _panel_spans(g)
    panel_size = row_spans[0][1] - row_spans[0][0]

    seed_panel = None
    seed_pattern = None
    seed_anchors = None
    markers: list[tuple[int, int, int, int, int]] = []

    for pr, (r0, r1) in enumerate(row_spans):
        for pc, (c0, c1) in enumerate(col_spans):
            panel = g[r0:r1, c0:c1]
            coords = np.argwhere(panel != BLACK)
            if len(coords) == 0:
                continue
            if len(coords) == 1:
                rr, cc = coords[0]
                markers.append((pr, pc, int(rr), int(cc), int(panel[int(rr), int(cc)])))
                continue
            assert seed_panel is None, "expected exactly one seed panel"
            seed_panel = (pr, pc)
            seed_pattern, seed_anchors = _seed_info(panel)

    assert seed_panel is not None and seed_pattern is not None and seed_anchors is not None
    out = g.copy()

    for pr, pc, mr, mc, color in markers:
        assert color in seed_anchors
        if (pr, pc) == seed_panel:
            continue
        r0, r1 = row_spans[pr]
        c0, c1 = col_spans[pc]
        panel = out[r0:r1, c0:c1]
        anchor_r, anchor_c = seed_anchors[color]
        dr = mr - anchor_r
        dc = mc - anchor_c
        for sr, sc, scolor in seed_pattern:
            tr = sr + dr
            tc = sc + dc
            if 0 <= tr < panel_size and 0 <= tc < panel_size:
                panel[tr, tc] = scolor

    return out


def _random_connected_shape(h: int, w: int, cell_count: int) -> list[tuple[int, int]]:
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    while len(cells) < cell_count:
        frontier = []
        for r, c in cells:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                    frontier.append((rr, cc))
        assert frontier
        cells.add(random.choice(frontier))
    return list(cells)


def _build_attempt() -> np.ndarray | None:
    panel_size = random.randint(4, 7)
    panel_rows = random.randint(2, 4)
    panel_cols = random.randint(2, 4)
    if panel_rows * panel_cols < 4:
        return None

    grid_h = panel_rows * panel_size + (panel_rows - 1)
    grid_w = panel_cols * panel_size + (panel_cols - 1)
    grid = np.full((grid_h, grid_w), BLACK, dtype=int)

    for r in range(1, panel_rows):
        grid[r * panel_size + (r - 1), :] = GREY
    for c in range(1, panel_cols):
        grid[:, c * panel_size + (c - 1)] = GREY

    panel_positions = [(r, c) for r in range(panel_rows) for c in range(panel_cols)]
    seed_pr, seed_pc = random.choice(panel_positions)

    min_cells = 3
    max_cells = min(len(PALETTE), max(4, panel_size * panel_size // 2))
    cell_count = random.randint(min_cells, max_cells)
    shape = _random_connected_shape(panel_size, panel_size, cell_count)
    colors = random.sample(PALETTE, cell_count)
    random.shuffle(shape)
    anchor_by_color = dict(zip(colors, shape))

    seed_r0 = seed_pr * (panel_size + 1)
    seed_c0 = seed_pc * (panel_size + 1)
    for color, (r, c) in anchor_by_color.items():
        grid[seed_r0 + r, seed_c0 + c] = color

    other_panels = [(r, c) for r, c in panel_positions if (r, c) != (seed_pr, seed_pc)]
    random.shuffle(other_panels)
    marker_count = random.randint(1, min(len(other_panels), len(colors)))
    marker_panels = other_panels[:marker_count]

    used_colors = random.sample(colors, marker_count)
    make_clipped = marker_count > 0 and random.random() < 0.5
    clipped_index = random.randrange(marker_count) if make_clipped else -1

    for idx, ((pr, pc), color) in enumerate(zip(marker_panels, used_colors)):
        anchor_r, anchor_c = anchor_by_color[color]
        if idx == clipped_index:
            candidates = [
                (r, c)
                for r in range(panel_size)
                for c in range(panel_size)
                if any(
                    not (0 <= sr + (r - anchor_r) < panel_size and 0 <= sc + (c - anchor_c) < panel_size)
                    for sr, sc in shape
                )
            ]
            if not candidates:
                return None
            mr, mc = random.choice(candidates)
        else:
            mr = random.randrange(panel_size)
            mc = random.randrange(panel_size)

        r0 = pr * (panel_size + 1)
        c0 = pc * (panel_size + 1)
        grid[r0 + mr, c0 + mc] = color

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _build_attempt()
        if grid is None:
            continue
        try:
            out = main(grid)
        except Exception:
            continue
        if out.shape != grid.shape:
            continue
        if np.array_equal(out, grid):
            continue
        return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_RETRIES} attempts")
