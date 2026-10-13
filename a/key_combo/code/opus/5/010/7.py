from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
GREY = 5
ORANGE = 7
INDEX_SIZE = 3
BLOCK_SIZE = 3

COLOR_POOL = [1, 2, 3, 4, 6, 8, 9]

_TEMPLATE_BITS = [
    np.array([[1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=int),
    np.array([[1, 1, 1], [1, 0, 1], [0, 0, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[1, 0, 0], [1, 1, 1], [0, 1, 1]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[1, 0, 1], [1, 1, 1], [0, 0, 1]], dtype=int),
]
TEMPLATE_LIBRARY = [np.where(bits == 1, GREY, BG).astype(int) for bits in _TEMPLATE_BITS]

MAX_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _find_components(
    grid: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    color: int | None = None,
    connectivity: int = 4,
) -> list[dict]:
    h, w = grid.shape
    if mask is None:
        allowed = np.ones((h, w), dtype=bool)
    else:
        allowed = mask.astype(bool)

    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not allowed[r, c]:
                continue
            cell_color = int(grid[r, c])
            if color is not None and cell_color != int(color):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in steps:
                    nr = rr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or not allowed[nr, nc]:
                        continue
                    if int(grid[nr, nc]) != cell_color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            comps.append(
                {
                    "color": cell_color,
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                    "area": len(coords),
                }
            )
    return comps


def _occupied_row_bands(grid: np.ndarray, bg: int) -> list[tuple[int, int]]:
    rows = [r for r in range(grid.shape[0]) if np.any(grid[r] != bg)]
    if not rows:
        return []
    bands: list[tuple[int, int]] = []
    start = rows[0]
    prev = rows[0]
    for r in rows[1:]:
        if r == prev + 1:
            prev = r
            continue
        bands.append((start, prev))
        start = prev = r
    bands.append((start, prev))
    return bands


def _rotate_cw(grid: np.ndarray, quarter_turns: int) -> np.ndarray:
    return np.rot90(grid, -int(quarter_turns) % 4).copy()


def _render_output(index_grid: np.ndarray, row_turns: list[int], color_to_template: dict[int, np.ndarray]) -> np.ndarray:
    out = np.zeros((INDEX_SIZE * BLOCK_SIZE, INDEX_SIZE * BLOCK_SIZE), dtype=int)
    for r in range(INDEX_SIZE):
        turn = int(row_turns[r])
        for c in range(INDEX_SIZE):
            color = int(index_grid[r, c])
            template = _rotate_cw(color_to_template[color], turn)
            block = np.where(template == GREY, color, BG)
            r0 = r * BLOCK_SIZE
            c0 = c * BLOCK_SIZE
            out[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE] = block
    return out


def _parse_legend(legend: np.ndarray) -> dict[int, np.ndarray]:
    non_bg = legend != BG
    components = _find_components(legend, mask=non_bg, connectivity=4)

    swatches = []
    templates = []
    for comp in components:
        color = int(comp["color"])
        r0, c0, r1, c1 = comp["bbox"]
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if color == GREY:
            patch = _crop_bbox(legend, comp["bbox"])
            assert patch.shape == (3, 3), "legend template must occupy a 3x3 bbox"
            assert not np.array_equal(patch, _rotate_cw(patch, 1)), "template must change under 90-degree rotation"
            templates.append((c0, patch))
            continue
        assert color not in {BG, ORANGE}, "legend swatches cannot use background or marker color"
        assert h == 2 and w == 2 and comp["area"] == 4, "legend swatch must be a solid 2x2 block"
        swatches.append((c0, color))

    swatches.sort(key=lambda item: item[0])
    templates.sort(key=lambda item: item[0])
    assert len(swatches) == len(templates) >= 2, "legend must contain equally many swatches and templates"
    assert len({color for _, color in swatches}) == len(swatches), "swatch colors must be distinct"

    return {
        int(color): patch.copy()
        for (_, color), (_, patch) in zip(swatches, templates, strict=True)
    }


def _parse_index_and_markers(lower: np.ndarray, valid_colors: set[int]) -> tuple[np.ndarray, list[int]]:
    index_mask = np.isin(lower, list(valid_colors))
    coords = np.argwhere(index_mask)
    assert coords.size > 0, "index grid not found"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    assert (r1 - r0 + 1, c1 - c0 + 1) == (3, 3), "index grid must be 3x3"
    index_grid = lower[r0 : r1 + 1, c0 : c1 + 1].copy()
    assert np.all(np.isin(index_grid, list(valid_colors))), "index grid must use only swatch colors"

    markers = _find_components(lower, color=ORANGE, connectivity=4)
    assert len(markers) == 3, "expected exactly three orange domino markers"
    markers.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))

    row_turns: list[int] = []
    for comp in markers:
        r0, c0, r1, c1 = comp["bbox"]
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert comp["area"] == 2, "each marker must be a two-pixel domino"
        if h == 1 and w == 2:
            row_turns.append(0)
        elif h == 2 and w == 1:
            row_turns.append(1)
        else:
            raise AssertionError("marker must be horizontal or vertical")

    assert len(row_turns) == 3
    return index_grid, row_turns


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each colored index cell into a rotated 3x3 stencil block chosen by the legend.

    Input structure:
    - Black (0) is the background. The occupied rows split into an upper legend band and a lower index band.
    - In the legend band, distinct 2x2 solid color swatches appear on the left, and the same number of grey (5) 3x3 stencil masks appear on the right.
    - In the lower band, a 3x3 index grid uses only legend swatch colors, and three orange (7) two-pixel domino markers appear to its right.

    Rule:
    - Pair legend swatches and legend stencil masks by left-to-right order to build a color -> 3x3 stencil mapping.
    - Read the three orange dominoes from top to bottom: a horizontal domino means 0-degree rotation, and a vertical domino means 90-degree clockwise rotation for that output row.
    - For each index-grid cell, look up its stencil, rotate it according to that cell's row marker, and write the rotated 3x3 block into the output.
    - Grey stencil pixels become the index cell's color; stencil holes remain black.

    Key insight:
    - The color chooses which stencil shape to use, while the orange domino orientation independently chooses how every stencil in that row is rotated.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    assert bg == BG, "this task uses black background"

    bands = _occupied_row_bands(g, bg)
    assert len(bands) == 2, "input must contain separate legend and index bands"
    (top0, top1), (bot0, bot1) = bands
    legend = g[top0 : top1 + 1]
    lower = g[bot0 : bot1 + 1]

    color_to_template = _parse_legend(legend)
    index_grid, row_turns = _parse_index_and_markers(lower, set(color_to_template))
    return _render_output(index_grid, row_turns, color_to_template)


def _make_index_grid(rng: random.Random, colors: list[int], row_turns: list[int]) -> np.ndarray:
    while True:
        cells = [rng.choice(colors) for _ in range(9)]
        for color, pos in zip(colors, rng.sample(range(9), len(colors)), strict=True):
            cells[pos] = color
        index_grid = np.array(cells, dtype=int).reshape(3, 3)

        if not any(len(set(map(int, index_grid[r]))) >= 2 for r in range(3)):
            continue

        seen_turns_by_color: dict[int, set[int]] = {color: set() for color in colors}
        for r in range(3):
            for c in range(3):
                seen_turns_by_color[int(index_grid[r, c])].add(int(row_turns[r]))
        if not any(len(turns) >= 2 for turns in seen_turns_by_color.values()):
            continue
        return index_grid


def _build_input_from_spec(
    rng: random.Random,
    colors: list[int],
    templates: list[np.ndarray],
    index_grid: np.ndarray,
    row_turns: list[int],
) -> np.ndarray:
    n = len(colors)
    swatch_gap = rng.randint(1, 2)
    bridge_gap = rng.randint(1, 3)
    template_gap = rng.randint(1, 2)

    legend_w = n * 2 + (n - 1) * swatch_gap + bridge_gap + n * 3 + (n - 1) * template_gap
    marker_gap = rng.randint(1, 2)
    marker_w = 6
    lower_w = 3 + marker_gap + marker_w

    left_pad = rng.randint(0, 2)
    right_pad = rng.randint(0, 2)
    width = max(legend_w, lower_w) + left_pad + right_pad

    top_pad = rng.randint(0, 2)
    bottom_pad = rng.randint(0, 2)
    legend_h = 4
    lower_h = 4
    height = top_pad + legend_h + 1 + lower_h + bottom_pad

    grid = np.zeros((height, width), dtype=int)

    legend_left = left_pad + rng.randint(0, width - left_pad - right_pad - legend_w)
    lower_left = left_pad + rng.randint(0, width - left_pad - right_pad - lower_w)
    legend_top = top_pad
    lower_top = top_pad + legend_h + 1

    swatch_row = legend_top + rng.randint(1, 2)
    template_row = legend_top + rng.randint(0, 1)

    c = legend_left
    for idx, color in enumerate(colors):
        grid[swatch_row : swatch_row + 2, c : c + 2] = int(color)
        c += 2
        if idx < n - 1:
            c += swatch_gap

    c = legend_left + (n * 2 + (n - 1) * swatch_gap + bridge_gap)
    for idx, template in enumerate(templates):
        mask = template == GREY
        patch = np.where(mask, GREY, BG)
        grid[template_row : template_row + 3, c : c + 3][mask] = GREY
        c += 3
        if idx < n - 1:
            c += template_gap

    grid[lower_top : lower_top + 3, lower_left : lower_left + 3] = index_grid

    marker_left = lower_left + 3 + marker_gap
    slot_cols = [marker_left, marker_left + 2, marker_left + 4]
    for row_idx, turn in enumerate(row_turns):
        rr = lower_top + row_idx
        cc = slot_cols[row_idx]
        if turn == 0:
            grid[rr, cc : cc + 2] = ORANGE
        else:
            grid[rr : rr + 2, cc] = ORANGE

    return grid


def _validate_semantics(colors: list[int], templates: list[np.ndarray], index_grid: np.ndarray, row_turns: list[int]) -> bool:
    mapping = {int(color): template.copy() for color, template in zip(colors, templates, strict=True)}
    correct = _render_output(index_grid, row_turns, mapping)

    if len(set(row_turns)) < 2:
        return False
    if set(map(int, index_grid.ravel())) != set(map(int, colors)):
        return False
    if np.array_equal(correct, _render_output(index_grid, [0, 0, 0], mapping)):
        return False

    if len(colors) >= 2:
        swapped = mapping.copy()
        c0, c1 = int(colors[0]), int(colors[1])
        swapped[c0], swapped[c1] = swapped[c1], swapped[c0]
        if np.array_equal(correct, _render_output(index_grid, row_turns, swapped)):
            return False

    for template in templates:
        if np.array_equal(template, _rotate_cw(template, 1)):
            return False
    return True


def _generate_candidate(rng: random.Random) -> np.ndarray | None:
    n = rng.randint(2, 3)
    colors = rng.sample(COLOR_POOL, n)
    template_indices = rng.sample(range(len(TEMPLATE_LIBRARY)), n)
    templates = [TEMPLATE_LIBRARY[idx].copy() for idx in template_indices]

    row_turns = rng.choice(
        [
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
        ]
    )
    index_grid = _make_index_grid(rng, colors, row_turns)
    if not _validate_semantics(colors, templates, index_grid, row_turns):
        return None

    inp = _build_input_from_spec(rng, colors, templates, index_grid, row_turns)
    expected = _render_output(index_grid, row_turns, {color: template for color, template in zip(colors, templates, strict=True)})

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None

    non_bg_rows = _occupied_row_bands(inp, BG)
    if len(non_bg_rows) != 2:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        rng = random.Random()
        grid = _generate_candidate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
