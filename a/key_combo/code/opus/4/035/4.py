import hashlib
import itertools
import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, colors_present, find_components, get_bg, paste


PANEL_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW)
PANEL_SIDE_RANGE = (5, 7)
PANEL_AREA_RANGE = (4, 10)
SELECTED_COUNT_OPTIONS = (2, 3)
GEN_RETRY_LIMIT = 600
_SEEN_INPUT_SIGNATURES: set[str] = set()
_SEEN_OUTPUT_SIGNATURES: set[str] = set()


def _signature(grid: np.ndarray) -> str:
    return hashlib.md5(grid.tobytes() + str(grid.shape).encode()).hexdigest()


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _random_connected_mask(
    height: int,
    width: int,
    area: int,
    required: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
) -> np.ndarray | None:
    if area < len(required):
        return None
    if any(not (0 <= r < height and 0 <= c < width) for r, c in required | forbidden):
        return None
    if required & forbidden:
        return None

    allowed = {(r, c) for r in range(height) for c in range(width)} - forbidden
    if not required:
        required = {random.choice(tuple(allowed))}
    if not required <= allowed:
        return None

    mask = np.zeros((height, width), dtype=bool)
    for r, c in required:
        mask[r, c] = True

    while int(mask.sum()) < area:
        frontier = set()
        for r, c in np.argwhere(mask):
            for nr, nc in _neighbors4(int(r), int(c)):
                if (nr, nc) in allowed and not mask[nr, nc]:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        weights = []
        choices = sorted(frontier)
        for r, c in choices:
            neighbor_count = sum((nr, nc) in allowed and mask[nr, nc] for nr, nc in _neighbors4(r, c))
            center_bias = 1 if 1 <= r < height - 1 and 1 <= c < width - 1 else 0
            weights.append(1 + 2 * neighbor_count + center_bias)
        pick = random.choices(choices, weights=weights, k=1)[0]
        mask[pick] = True
    return mask


def _render_panel(mask: np.ndarray, color: int) -> np.ndarray:
    panel = np.zeros(mask.shape, dtype=int)
    panel[mask] = int(color)
    return panel


def _compose_panels(
    panel_masks: dict[int, np.ndarray],
    order: list[int],
    panel_shape: tuple[int, int],
) -> np.ndarray:
    out = np.zeros(panel_shape, dtype=int)
    for color in order:
        paste(out, _render_panel(panel_masks[color], color), 0, 0, transparent=Color.BLACK)
    return out


def _visible_colors(output: np.ndarray) -> set[int]:
    return set(colors_present(output, exclude={Color.BLACK}))


def _panel_map_from_grid(grid: np.ndarray) -> tuple[dict[int, np.ndarray], list[int], tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    assert bg == Color.BLACK, "background must be black"

    nonempty_rows = np.flatnonzero(np.any(g != bg, axis=1))
    assert nonempty_rows.size > 0, "grid must contain foreground"
    indicator_row = int(nonempty_rows[-1])
    assert indicator_row >= 3, "need room for panels, gap row, and indicators"
    assert indicator_row == g.shape[0] - 1, "indicator sequence must be on the bottom row"
    assert np.all(g[indicator_row - 1] == bg), "expected a blank separator row above the indicators"

    panel_block = g[: indicator_row - 1, :]
    panel_block_h, panel_block_w = panel_block.shape
    assert panel_block_h % 2 == 0 and panel_block_w % 2 == 0, "panel block must split evenly into 2x2 panels"
    panel_h = panel_block_h // 2
    panel_w = panel_block_w // 2

    panels: dict[int, np.ndarray] = {}
    panel_colors = []
    origins = ((0, 0), (0, panel_w), (panel_h, 0), (panel_h, panel_w))
    for r0, c0 in origins:
        panel = panel_block[r0 : r0 + panel_h, c0 : c0 + panel_w]
        fg_colors = colors_present(panel, exclude={bg})
        assert len(fg_colors) == 1, "each panel must use exactly one foreground color"
        color = int(fg_colors[0])
        assert color not in panels, "panel colors must be unique"
        comps = find_components(panel, bg=bg, connectivity=4, monochromatic=True)
        assert len(comps) == 1, "each panel must contain one 4-connected foreground object"
        assert comps[0]["color"] == color
        panels[color] = panel.copy()
        panel_colors.append(color)

    assert set(panel_colors) == set(PANEL_COLORS), "expected the four canonical panel colors"

    indicator_cols = np.flatnonzero(g[indicator_row] != bg)
    assert indicator_cols.size in SELECTED_COUNT_OPTIONS, "indicator row must select 2 or 3 panels"
    assert np.all(np.diff(indicator_cols) >= 2), "indicator pixels must be separated by at least one black cell"
    order = [int(g[indicator_row, c]) for c in indicator_cols]
    assert len(order) == len(set(order)), "indicator colors must be distinct"
    assert set(order) <= set(panel_colors), "each indicator color must match one panel"

    return panels, order, (panel_h, panel_w)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay only the selected colored panels, using the bottom color sequence as the layer order.

    Input structure:
    - The upper block is a 2x2 tiling of equal-size panels with no separator lines between them.
    - Each panel contains exactly one 4-connected non-black shape on a black background, and the four panel colors are blue, red, green, and yellow.
    - The bottom row contains 2 or 3 isolated non-black indicator pixels, separated by black cells, with one blank row between the panels and the indicator row.

    Rule:
    - Read the non-black indicator pixels from left to right to obtain a color sequence.
    - Match each indicator color to the unique panel with the same foreground color; only those panels participate.
    - Create a black canvas the size of one panel and overlay the selected panels in left-to-right indicator order, drawing only their non-black pixels so later panels overwrite earlier ones.

    Key insight:
    - The indicator colors choose which panels to keep, while the indicators' horizontal positions independently encode the transparent layer priority.
    """
    panels, order, panel_shape = _panel_map_from_grid(grid)
    panel_masks = {color: panel == color for color, panel in panels.items()}
    return _compose_panels(panel_masks, order, panel_shape)


def _choose_indicator_columns(width: int, count: int) -> list[int] | None:
    columns = list(range(width))
    valid = []
    for combo in itertools.combinations(columns, count):
        if all(combo[i + 1] - combo[i] >= 2 for i in range(len(combo) - 1)):
            valid.append(combo)
    if not valid:
        return None
    return list(random.choice(valid))


def _visible_contribution(panel_masks: dict[int, np.ndarray], order: list[int], color: int) -> bool:
    output = _compose_panels(panel_masks, order, next(iter(panel_masks.values())).shape)
    return int(color) in _visible_colors(output)


def _build_panel_masks(
    panel_shape: tuple[int, int],
    order: list[int],
    selected: set[int],
) -> dict[int, np.ndarray] | None:
    height, width = panel_shape
    all_cells = [(r, c) for r in range(height) for c in range(width)]
    overlap_cell = random.choice(all_cells)
    overlap_pair = order[-2:]

    selected_union = np.zeros(panel_shape, dtype=bool)
    masks: dict[int, np.ndarray] = {}
    selected_list = order[:]
    first_selected = selected_list[0] if len(selected_list) == 3 else None

    unselected = [color for color in PANEL_COLORS if color not in selected]
    unique_unselected_cell = None
    if unselected:
        candidates = [cell for cell in all_cells if cell != overlap_cell]
        unique_unselected_cell = random.choice(candidates)

    for color in PANEL_COLORS:
        required = set()
        forbidden = set()
        if color in overlap_pair:
            required.add(overlap_cell)
        if color == first_selected:
            forbidden.add(overlap_cell)
        if unique_unselected_cell is not None and color == unselected[0]:
            required.add(unique_unselected_cell)
        area = random.randint(*PANEL_AREA_RANGE)
        mask = _random_connected_mask(height, width, area, required, forbidden)
        if mask is None:
            return None
        masks[color] = mask
        if color in selected:
            selected_union |= mask

    if len(selected_list) == 3 and masks[first_selected][overlap_cell]:
        return None
    if not all(masks[color][overlap_cell] for color in overlap_pair):
        return None
    if any(masks[color][overlap_cell] for color in selected - set(overlap_pair)):
        return None
    if unique_unselected_cell is not None and selected_union[unique_unselected_cell]:
        return None
    return masks


def _validate_attempt(grid: np.ndarray, expected_order: list[int], panel_masks: dict[int, np.ndarray]) -> bool:
    output = main(grid)
    panel_shape = output.shape
    selected = set(expected_order)
    unselected = [color for color in PANEL_COLORS if color not in selected]

    if max(grid.shape) > 30 or max(output.shape) > 30:
        return False
    if not np.array_equal(output, _compose_panels(panel_masks, expected_order, panel_shape)):
        return False
    if len(_visible_colors(output)) < 2:
        return False
    if not all(_visible_contribution(panel_masks, expected_order, color) for color in expected_order):
        return False

    order_sensitive = False
    for i in range(len(expected_order)):
        for j in range(i + 1, len(expected_order)):
            swapped = expected_order[:]
            swapped[i], swapped[j] = swapped[j], swapped[i]
            if not np.array_equal(
                _compose_panels(panel_masks, expected_order, panel_shape),
                _compose_panels(panel_masks, swapped, panel_shape),
            ):
                order_sensitive = True
                break
        if order_sensitive:
            break
    if not order_sensitive:
        return False

    if unselected:
        selection_sensitive = any(
            not np.array_equal(
                _compose_panels(panel_masks, expected_order, panel_shape),
                _compose_panels(panel_masks, [color] + expected_order, panel_shape),
            )
            for color in unselected
        )
        if not selection_sensitive:
            return False

    for color, mask in panel_masks.items():
        panel = _render_panel(mask, color)
        comps = find_components(panel, bg=Color.BLACK, connectivity=4, monochromatic=True)
        if len(comps) != 1 or comps[0]["area"] != int(mask.sum()):
            return False
        if not (PANEL_AREA_RANGE[0] <= int(mask.sum()) <= PANEL_AREA_RANGE[1]):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    panel_h = random.randint(*PANEL_SIDE_RANGE)
    panel_w = random.randint(*PANEL_SIDE_RANGE)
    panel_shape = (panel_h, panel_w)

    selected_count = random.choice(SELECTED_COUNT_OPTIONS)
    selected = set(random.sample(PANEL_COLORS, selected_count))
    order = random.sample(list(selected), len(selected))
    panel_masks = _build_panel_masks(panel_shape, order, selected)
    if panel_masks is None:
        return None

    indicator_cols = _choose_indicator_columns(2 * panel_w, selected_count)
    if indicator_cols is None:
        return None

    grid = np.zeros((2 * panel_h + 2, 2 * panel_w), dtype=int)
    placements = (
        (0, 0, PANEL_COLORS[0]),
        (0, panel_w, PANEL_COLORS[1]),
        (panel_h, 0, PANEL_COLORS[2]),
        (panel_h, panel_w, PANEL_COLORS[3]),
    )
    for r0, c0, color in placements:
        panel = _render_panel(panel_masks[color], color)
        grid[r0 : r0 + panel_h, c0 : c0 + panel_w] = panel

    for col, color in zip(indicator_cols, order):
        grid[-1, col] = color

    if not _validate_attempt(grid, order, panel_masks):
        return None

    input_sig = _signature(grid)
    output_sig = _signature(main(grid))
    if input_sig in _SEEN_INPUT_SIGNATURES or output_sig in _SEEN_OUTPUT_SIGNATURES:
        return None
    _SEEN_INPUT_SIGNATURES.add(input_sig)
    _SEEN_OUTPUT_SIGNATURES.add(output_sig)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid selective-panel-overlay input")
