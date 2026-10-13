import random

import numpy as np


BG = 0
PALETTE = (1, 2, 3, 4)
SEP = 5
PANEL_COUNT = 3
PANEL_SIDE_MIN = 4
PANEL_SIDE_MAX = 6
MARK_COUNT_MIN = 2
MARK_COUNT_MAX = 4
MAX_GENERATION_ATTEMPTS = 400


def _separator_spans(indices: list[int], limit: int) -> list[tuple[int, int]]:
    cuts = [-1, *indices, limit]
    spans = []
    for i in range(len(cuts) - 1):
        start = cuts[i] + 1
        stop = cuts[i + 1]
        if start >= stop:
            raise AssertionError("empty panel span")
        spans.append((start, stop))
    return spans


def _extract_panels(grid: np.ndarray) -> tuple[str, list[np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    h, w = g.shape

    sep_rows = [r for r in range(h) if np.all(g[r, :] == SEP)]
    sep_cols = [c for c in range(w) if np.all(g[:, c] == SEP)]

    horizontal_ok = len(sep_cols) == PANEL_COUNT - 1 and len(sep_rows) == 0
    vertical_ok = len(sep_rows) == PANEL_COUNT - 1 and len(sep_cols) == 0
    assert horizontal_ok ^ vertical_ok, "expected exactly one separator orientation"

    if horizontal_ok:
        spans = _separator_spans(sep_cols, w)
        assert len(spans) == PANEL_COUNT
        widths = {c1 - c0 for c0, c1 in spans}
        assert len(widths) == 1
        panels = [g[:, c0:c1].copy() for c0, c1 in spans]
        for panel in panels:
            assert not np.any(panel == SEP)
        return "horizontal", panels

    spans = _separator_spans(sep_rows, h)
    assert len(spans) == PANEL_COUNT
    heights = {r1 - r0 for r0, r1 in spans}
    assert len(heights) == 1
    panels = [g[r0:r1, :].copy() for r0, r1 in spans]
    for panel in panels:
        assert not np.any(panel == SEP)
    return "vertical", panels


def _render_output_from_panels(panels: list[np.ndarray]) -> np.ndarray:
    assert len(panels) == PANEL_COUNT
    h, w = panels[0].shape
    assert all(panel.shape == (h, w) for panel in panels)

    out = np.zeros((2 * h, 2 * w), dtype=int)
    for r in range(h):
        for c in range(w):
            visible_stack = [int(panel[r, c]) for panel in panels if int(panel[r, c]) != BG]
            rr = 2 * r
            cc = 2 * c
            if not visible_stack:
                continue
            if len(visible_stack) == 1:
                out[rr : rr + 2, cc : cc + 2] = visible_stack[-1]
                continue
            top = visible_stack[-1]
            second = visible_stack[-2]
            out[rr, cc] = top
            out[rr, cc + 1] = second
            out[rr + 1, cc] = second
            out[rr + 1, cc + 1] = top
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Overlay three separator-delimited panels, then magnify each merged pixel by a block chosen from overlap depth.

    Input structure:
    - The grid contains exactly three same-sized panels separated by two full grey lines, either vertical or horizontal.
    - Panel cells use black as transparent background and sparse colored marks from the visible palette; the separator color itself is not part of any panel.

    Rule:
    - Read the three panels in left-to-right or top-to-bottom order; later panels are visually above earlier panels.
    - For each panel-relative coordinate, count how many panels contain a non-black mark there and record the topmost and second-topmost colors among those marks.
    - Produce a 2x enlarged output: empty positions become a black 2x2 block, single-layer positions become a solid 2x2 block of the visible color, and positions with depth at least two become a 2x2 checkerboard whose main diagonal uses the top color and whose other diagonal uses the second-top color.

    Key insight:
    - The overlay result alone is insufficient: the hidden overlap depth, together with layer order, determines which 2x2 replacement pattern each merged pixel expands into.
    """
    _, panels = _extract_panels(np.asarray(grid, dtype=int))
    return _render_output_from_panels(panels)


def _compose_input(panels: list[np.ndarray], orientation: str) -> np.ndarray:
    h, w = panels[0].shape
    if orientation == "horizontal":
        grid = np.zeros((h, PANEL_COUNT * w + (PANEL_COUNT - 1)), dtype=int)
        for idx, panel in enumerate(panels):
            c0 = idx * (w + 1)
            grid[:, c0 : c0 + w] = panel
            if idx < PANEL_COUNT - 1:
                grid[:, c0 + w] = SEP
        return grid

    grid = np.zeros((PANEL_COUNT * h + (PANEL_COUNT - 1), w), dtype=int)
    for idx, panel in enumerate(panels):
        r0 = idx * (h + 1)
        grid[r0 : r0 + h, :] = panel
        if idx < PANEL_COUNT - 1:
            grid[r0 + h, :] = SEP
    return grid


def _choose_overlap_pairs(overlap_count: int) -> list[tuple[int, int]]:
    all_pairs = [(0, 1), (0, 2), (1, 2)]
    if overlap_count <= len(all_pairs) and random.random() < 0.7:
        return random.sample(all_pairs, overlap_count)
    return [random.choice(all_pairs) for _ in range(overlap_count)]


def _assign_colors(
    placements: list[list[tuple[int, int]]], overlap_cells: dict[tuple[int, int], tuple[int, int]]
) -> list[dict[tuple[int, int], int]] | None:
    panel_maps = [dict() for _ in range(PANEL_COUNT)]
    for _ in range(100):
        color_maps = []
        for cells in placements:
            colors = random.sample(PALETTE, len(cells))
            color_maps.append(dict(zip(cells, colors)))
        valid = True
        for cell, pair in overlap_cells.items():
            c0 = color_maps[pair[0]][cell]
            c1 = color_maps[pair[1]][cell]
            if c0 == c1:
                valid = False
                break
        if valid:
            return color_maps
    return None


def _build_panels(
    panel_h: int, panel_w: int, mark_counts: list[int], overlap_count: int
) -> tuple[list[np.ndarray], dict[tuple[int, int], int]] | None:
    area = panel_h * panel_w
    if area <= max(mark_counts):
        return None

    cells = [(r, c) for r in range(panel_h) for c in range(panel_w)]
    overlap_targets = random.sample(cells, overlap_count)
    overlap_pairs = _choose_overlap_pairs(overlap_count)

    placements = [[] for _ in range(PANEL_COUNT)]
    remaining = mark_counts[:]
    overlap_cells: dict[tuple[int, int], tuple[int, int]] = {}
    occupancy: dict[tuple[int, int], int] = {}

    for cell, pair in zip(overlap_targets, overlap_pairs):
        a, b = pair
        if remaining[a] <= 0 or remaining[b] <= 0:
            return None
        placements[a].append(cell)
        placements[b].append(cell)
        remaining[a] -= 1
        remaining[b] -= 1
        overlap_cells[cell] = pair
        occupancy[cell] = 2

    free_single_cells = [cell for cell in cells if cell not in occupancy]
    random.shuffle(free_single_cells)
    cursor = 0
    for panel_idx in range(PANEL_COUNT):
        need = remaining[panel_idx]
        if cursor + need > len(free_single_cells):
            return None
        for _ in range(need):
            cell = free_single_cells[cursor]
            cursor += 1
            placements[panel_idx].append(cell)
            occupancy[cell] = 1

    if 1 not in occupancy.values() or 2 not in occupancy.values():
        return None
    if len(occupancy) >= area:
        return None

    color_maps = _assign_colors(placements, overlap_cells)
    if color_maps is None:
        return None

    panels = [np.zeros((panel_h, panel_w), dtype=int) for _ in range(PANEL_COUNT)]
    for panel_idx, color_map in enumerate(color_maps):
        for (r, c), color in color_map.items():
            panels[panel_idx][r, c] = color

    depth_map = {cell: 0 for cell in cells}
    for cell, depth in occupancy.items():
        depth_map[cell] = depth
    if max(depth_map.values()) > 2:
        return None
    return panels, depth_map


def _sample_counts() -> list[int]:
    return [random.randint(MARK_COUNT_MIN, MARK_COUNT_MAX) for _ in range(PANEL_COUNT)]


def _validate_sample(inp: np.ndarray) -> bool:
    try:
        orientation, panels = _extract_panels(inp)
        assert orientation in {"horizontal", "vertical"}
        h, w = panels[0].shape
        assert all(panel.shape == (h, w) for panel in panels)
        assert max(inp.shape) <= 30

        depths = []
        for panel in panels:
            coords = list(zip(*np.nonzero(panel)))
            colors = [int(panel[r, c]) for r, c in coords]
            assert MARK_COUNT_MIN <= len(coords) <= MARK_COUNT_MAX
            assert len(set(coords)) == len(coords)
            assert len(set(colors)) == len(colors)
            assert set(colors).issubset(PALETTE)
        for r in range(h):
            for c in range(w):
                stack = [int(panel[r, c]) for panel in panels if int(panel[r, c]) != BG]
                depths.append(len(stack))
                if len(stack) >= 2:
                    assert len(set(stack)) == len(stack)
        assert 0 in depths and 1 in depths and 2 in depths
        assert max(depths) == 2

        out = main(inp)
        assert out.shape == (2 * h, 2 * w)
        assert max(out.shape) <= 30
        assert np.array_equal(out, _render_output_from_panels(panels))
        return True
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    orientation = random.choice(["horizontal", "vertical"])
    panel_h = random.randint(PANEL_SIDE_MIN, PANEL_SIDE_MAX)
    panel_w = random.randint(PANEL_SIDE_MIN, PANEL_SIDE_MAX)
    if orientation == "vertical" and 3 * panel_h + 2 > 30:
        return None
    if orientation == "horizontal" and 3 * panel_w + 2 > 30:
        return None
    if 2 * max(panel_h, panel_w) > 30:
        return None

    mark_counts = _sample_counts()
    total_marks = sum(mark_counts)
    max_overlaps = min(3, total_marks // 2, panel_h * panel_w - 2)
    if max_overlaps < 1:
        return None
    overlap_count = random.randint(1, max_overlaps)

    built = _build_panels(panel_h, panel_w, mark_counts, overlap_count)
    if built is None:
        return None
    panels, depth_map = built

    depths = list(depth_map.values())
    if not (0 in depths and 1 in depths and 2 in depths):
        return None

    inp = _compose_input(panels, orientation)
    if not _validate_sample(inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid panel-overlay input")
