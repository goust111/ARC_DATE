from __future__ import annotations

import random

import numpy as np


BG = 0
BORDER = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

CELL_H = 5
CELL_W = 6
CELL_GAP = 1

GRID_MIN = 18
GRID_MAX = 30
REGION_COUNT_MIN = 2
REGION_COUNT_MAX = 4
REGION_OUTER_MIN = 5
REGION_OUTER_MAX = 8
GEN_RETRIES = 400


def _mode_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _is_rect_frame(cells: list[tuple[int, int]], box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False

    expected = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4
    if len(cells) != expected:
        return False

    cell_set = set(cells)
    for r in range(r0, r1 + 1):
        if (r, c0) not in cell_set or (r, c1) not in cell_set:
            return False
    for c in range(c0, c1 + 1):
        if (r0, c) not in cell_set or (r1, c) not in cell_set:
            return False
    return True


def _touches_edge(box: tuple[int, int, int, int], shape: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = box
    h, w = shape
    return r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1


def _find_border_frames(grid: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int, int, int]]]:
    frames_by_color: dict[int, list[tuple[int, int, int, int]]] = {}
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for comp in _components(grid == color):
            box = _bbox(comp)
            if _is_rect_frame(comp, box):
                frames_by_color.setdefault(color, []).append(box)

    assert frames_by_color, "expected rectangular frames in the scene"
    edge_colors = [
        color
        for color, boxes in frames_by_color.items()
        if any(_touches_edge(box, grid.shape) for box in boxes)
    ]
    assert len(edge_colors) == 1, "expected one shared border color for frames and legend"
    border_color = edge_colors[0]
    return border_color, frames_by_color[border_color]


def _parse_legend_cell(grid: np.ndarray, bg: int, box: tuple[int, int, int, int]) -> tuple[int, str]:
    r0, c0, r1, c1 = box
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    ih, iw = inner.shape
    assert ih >= 2 and iw >= 4, "legend cells need a visible swatch/pattern split"

    mid = iw // 2
    left = inner[:, :mid]
    right = inner[:, mid:]
    colors = [int(v) for v in np.unique(inner) if int(v) != bg]
    assert len(colors) == 1, "legend cell should encode exactly one color"
    color = colors[0]

    assert np.all(left == color), "legend swatch must occupy the left half"
    assert np.all((right == bg) | (right == color)), "legend pattern must use only background and its color"

    mask = right == color
    row_uniform = np.all(mask == mask[:, :1])
    col_uniform = np.all(mask == mask[:1, :])
    assert row_uniform != col_uniform, "legend pattern must be either horizontal or vertical"
    return color, ("H" if row_uniform else "V")


def _region_seed_color(
    grid: np.ndarray,
    bg: int,
    border_color: int,
    box: tuple[int, int, int, int],
) -> int:
    r0, c0, r1, c1 = box
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    seed_mask = (inner != bg) & (inner != border_color)
    colors = [int(v) for v in np.unique(inner[seed_mask])]
    assert len(colors) == 1, "each region must contain exactly one seed color"
    assert int(np.count_nonzero(seed_mask)) == 1, "each region must contain a single isolated seed pixel"
    return colors[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey-framed region with seed-colored stripes according to the matching edge legend.

    Input structure:
    - The background is the dominant color, typically black.
    - Several one-pixel-thick grey rectangular frames enclose empty rectangular regions, each containing exactly one colored seed pixel.
    - Additional grey-framed legend cells sit on one grid edge; in each cell the left half is a solid color swatch and the right half shows that color's stripe orientation.

    Rule:
    - Identify the shared grey border color and separate edge-touching legend cells from non-edge interior regions.
    - Read each legend cell as a color-to-direction mapping: row-wise sample stripes mean horizontal filling, column-wise sample stripes mean vertical filling.
    - For every interior framed region, use its seed color to look up the direction, then overwrite the entire interior with alternating seed-colored stripes starting from the top row or left column; the seed pixel is treated like any other interior cell.

    Key insight:
    - The seed color alone is insufficient; the edge legend is a lookup table that tells whether that color should generate horizontal or vertical stripes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _mode_color(g)
    border_color, frames = _find_border_frames(g, bg)
    legend_boxes = [box for box in frames if _touches_edge(box, g.shape)]
    region_boxes = [box for box in frames if not _touches_edge(box, g.shape)]

    assert legend_boxes, "expected at least one edge legend cell"
    assert region_boxes, "expected at least one enclosed region"

    mapping: dict[int, str] = {}
    for box in legend_boxes:
        color, orientation = _parse_legend_cell(g, bg, box)
        assert color not in mapping, "legend colors must be unique"
        mapping[color] = orientation

    out = g.copy()
    for box in region_boxes:
        seed_color = _region_seed_color(g, bg, border_color, box)
        assert seed_color in mapping, "every seed color must appear in the legend"
        r0, c0, r1, c1 = box
        fill = np.full((r1 - r0 - 1, c1 - c0 - 1), bg, dtype=int)
        if mapping[seed_color] == "H":
            fill[::2, :] = seed_color
        else:
            fill[:, ::2] = seed_color
        out[r0 + 1 : r1, c0 + 1 : c1] = fill

    return out


def _make_legend_cell(color: int, orientation: str) -> np.ndarray:
    cell = np.full((CELL_H, CELL_W), BG, dtype=int)
    cell[0, :] = BORDER
    cell[-1, :] = BORDER
    cell[:, 0] = BORDER
    cell[:, -1] = BORDER

    inner = cell[1:-1, 1:-1]
    inner[:, :2] = color
    inner[:, 2:] = BG
    if orientation == "H":
        inner[::2, 2:] = color
    else:
        inner[:, 2] = color
    return cell


def _balanced_orientations(count: int, rng: random.Random) -> list[str]:
    if count % 2 == 0:
        orientations = ["H"] * (count // 2) + ["V"] * (count // 2)
    else:
        extra = rng.choice(["H", "V"])
        other = "V" if extra == "H" else "H"
        orientations = [extra] * ((count + 1) // 2) + [other] * (count // 2)
    rng.shuffle(orientations)
    return orientations


def _place_legend(
    grid: np.ndarray,
    edge: str,
    entries: list[tuple[int, str]],
    rng: random.Random,
) -> None:
    h, w = grid.shape
    count = len(entries)
    if edge in {"top", "bottom"}:
        span = count * CELL_W + (count - 1) * CELL_GAP
        start = rng.randint(0, w - span)
        row = 0 if edge == "top" else h - CELL_H
        for idx, (color, orientation) in enumerate(entries):
            col = start + idx * (CELL_W + CELL_GAP)
            grid[row : row + CELL_H, col : col + CELL_W] = _make_legend_cell(color, orientation)
    else:
        span = count * CELL_H + (count - 1) * CELL_GAP
        start = rng.randint(0, h - span)
        col = 0 if edge == "left" else w - CELL_W
        for idx, (color, orientation) in enumerate(entries):
            row = start + idx * (CELL_H + CELL_GAP)
            grid[row : row + CELL_H, col : col + CELL_W] = _make_legend_cell(color, orientation)


def _workspace_bounds(edge: str, h: int, w: int) -> tuple[int, int, int, int]:
    if edge == "top":
        return CELL_H + 1, h - 2, 1, w - 2
    if edge == "bottom":
        return 1, h - CELL_H - 2, 1, w - 2
    if edge == "left":
        return 1, h - 2, CELL_W + 1, w - 2
    return 1, h - 2, 1, w - CELL_W - 2


def _split_axis(start: int, end: int) -> list[tuple[int, int]] | None:
    gap = 1
    total = end - start + 1
    first = (total - gap) // 2
    second = total - gap - first
    if first < REGION_OUTER_MIN or second < REGION_OUTER_MIN:
        return None
    first_box = (start, start + first - 1)
    second_box = (first_box[1] + 1 + gap, end)
    return [first_box, second_box]


def _slots_for_regions(edge: str, h: int, w: int) -> list[tuple[int, int, int, int]] | None:
    r0, r1, c0, c1 = _workspace_bounds(edge, h, w)
    row_spans = _split_axis(r0, r1)
    col_spans = _split_axis(c0, c1)
    if row_spans is None or col_spans is None:
        return None

    slots = []
    for sr0, sr1 in row_spans:
        for sc0, sc1 in col_spans:
            slots.append((sr0, sr1, sc0, sc1))
    return slots


def _place_regions(
    grid: np.ndarray,
    entries: list[tuple[int, str]],
    edge: str,
    rng: random.Random,
) -> bool:
    slots = _slots_for_regions(edge, *grid.shape)
    if slots is None or len(entries) > len(slots):
        return False

    slot_order = list(range(len(slots)))
    rng.shuffle(slot_order)
    for (color, _orientation), slot_idx in zip(entries, slot_order):
        sr0, sr1, sc0, sc1 = slots[slot_idx]
        slot_h = sr1 - sr0 + 1
        slot_w = sc1 - sc0 + 1
        max_h = min(REGION_OUTER_MAX, slot_h)
        max_w = min(REGION_OUTER_MAX, slot_w)
        if max_h < REGION_OUTER_MIN or max_w < REGION_OUTER_MIN:
            return False

        outer_h = rng.randint(REGION_OUTER_MIN, max_h)
        outer_w = rng.randint(REGION_OUTER_MIN, max_w)
        top = rng.randint(sr0, sr1 - outer_h + 1)
        left = rng.randint(sc0, sc1 - outer_w + 1)
        bottom = top + outer_h - 1
        right = left + outer_w - 1

        grid[top, left : right + 1] = BORDER
        grid[bottom, left : right + 1] = BORDER
        grid[top : bottom + 1, left] = BORDER
        grid[top : bottom + 1, right] = BORDER

        seed_r = rng.randint(top + 1, bottom - 1)
        seed_c = rng.randint(left + 1, right - 1)
        grid[seed_r, seed_c] = color

    return True


def _grid_bounds_for_edge(edge: str, count: int) -> tuple[int, int]:
    min_h = GRID_MIN
    min_w = GRID_MIN
    if edge in {"top", "bottom"}:
        min_w = max(min_w, count * CELL_W + (count - 1) * CELL_GAP)
    else:
        min_h = max(min_h, count * CELL_H + (count - 1) * CELL_GAP)
    return min_h, min_w


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    count = rng.randint(REGION_COUNT_MIN, REGION_COUNT_MAX)
    edge = rng.choice(["top", "bottom", "left", "right"])
    min_h, min_w = _grid_bounds_for_edge(edge, count)
    h = rng.randint(min_h, GRID_MAX)
    w = rng.randint(min_w, GRID_MAX)

    grid = np.full((h, w), BG, dtype=int)
    colors = rng.sample(COLOR_POOL, count)
    orientations = _balanced_orientations(count, rng)
    entries = list(zip(colors, orientations))
    rng.shuffle(entries)

    _place_legend(grid, edge, entries, rng)
    if not _place_regions(grid, entries, edge, rng):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exceeded retry budget")
