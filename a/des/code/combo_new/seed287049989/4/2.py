import math
import random

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
GEN_RETRIES = 300
SLOT_GAP = 1
OUTPUT_SIDE_MAX = 18


def _layer_areas_for_shape(height: int, width: int, n_layers: int) -> list[int]:
    areas: list[int] = []
    for depth in range(n_layers - 1):
        outer_h = height - 2 * depth
        outer_w = width - 2 * depth
        inner_h = outer_h - 2
        inner_w = outer_w - 2
        assert outer_h >= 3 and outer_w >= 3
        areas.append(outer_h * outer_w - inner_h * inner_w)
    core_h = height - 2 * (n_layers - 1)
    core_w = width - 2 * (n_layers - 1)
    assert core_h >= 1 and core_w >= 1
    areas.append(core_h * core_w)
    return areas


def _render_rings(colors: list[int], height: int, width: int) -> np.ndarray:
    out = np.full((height, width), BG, dtype=int)
    depth = len(colors) - 1
    for layer, color in enumerate(colors[:-1]):
        r0 = c0 = layer
        r1 = height - 1 - layer
        c1 = width - 1 - layer
        out[r0, c0 : c1 + 1] = color
        out[r1, c0 : c1 + 1] = color
        out[r0 : r1 + 1, c0] = color
        out[r0 : r1 + 1, c1] = color
    out[depth : height - depth, depth : width - depth] = colors[-1]
    return out


def _infer_output_shape(layer_areas: list[int]) -> tuple[int, int]:
    n_layers = len(layer_areas)
    assert n_layers >= 2
    for i in range(n_layers - 2):
        assert layer_areas[i] - layer_areas[i + 1] == 8

    outer_sum = layer_areas[0] + 4
    assert outer_sum % 2 == 0
    height_plus_width = outer_sum // 2

    depth = n_layers - 1
    core_sum = height_plus_width - 4 * depth
    core_area = layer_areas[-1]
    assert core_sum >= 2 and core_area >= 1

    disc = core_sum * core_sum - 4 * core_area
    assert disc >= 0
    root = math.isqrt(disc)
    assert root * root == disc
    assert (core_sum + root) % 2 == 0 and (core_sum - root) % 2 == 0

    core_h = (core_sum + root) // 2
    core_w = (core_sum - root) // 2
    assert core_h >= 1 and core_w >= 1

    height = core_h + 2 * depth
    width = core_w + 2 * depth
    if height < width:
        height, width = width, height

    assert _layer_areas_for_shape(height, width, n_layers) == layer_areas
    return height, width


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered color fragments into concentric rectangular rings using the bottom legend order.

    Input structure:
    - Background is black.
    - Rows above the bottom edge contain scattered fragments in several non-black colors.
    - The last row contains one single-pixel marker of each fragment color, with black gaps allowed.

    Rule:
    - Read the non-black pixels on the bottom row from left to right; this gives the colors from the outer ring to the inner region.
    - For each listed color, count how many pixels of that color appear above the legend row.
    - Build the width-1 nested rectangle pattern whose layer areas match those counts, and fill the final remaining center with the last color.

    Key insight:
    - The fragment geometry is a distractor: the pixel counts determine the ring sizes, while the bottom-row color order determines the nesting order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2 and g.shape[1] >= 1

    legend = [int(value) for value in g[-1] if int(value) != BG]
    assert len(legend) >= 2
    assert len(set(legend)) == len(legend)

    upper = g[:-1]
    present = {int(value) for value in upper.ravel() if int(value) != BG}
    assert present == set(legend)

    layer_areas = [int(np.count_nonzero(upper == color)) for color in legend]
    assert all(area > 0 for area in layer_areas)

    height, width = _infer_output_shape(layer_areas)
    return _render_rings(legend, height, width)


def _pick_spaced_positions(length: int, count: int, min_gap: int) -> list[int] | None:
    candidates = list(range(length))
    random.shuffle(candidates)
    picked: list[int] = []
    for value in candidates:
        if all(abs(value - other) >= min_gap for other in picked):
            picked.append(value)
            if len(picked) == count:
                return sorted(picked)
    return None


def _choose_ring_spec() -> tuple[list[int], list[int], int]:
    n_layers = random.randint(2, 5)
    max_core = OUTPUT_SIDE_MAX - 2 * (n_layers - 1)
    core_side = random.randint(1, max_core)
    side = core_side + 2 * (n_layers - 1)
    colors = random.sample(COLORS, n_layers)
    areas = _layer_areas_for_shape(side, side, n_layers)
    return colors, areas, side


def _layout_options(layer_areas: list[int], n_colors: int) -> list[tuple[int, int, int, int]]:
    total = sum(layer_areas)
    options: list[tuple[int, int, int, int]] = []
    for height in range(18, 31):
        for width in range(max(2 * n_colors - 1, 18), 31):
            if height * width - (total + n_colors) <= total:
                continue
            top_height = height - 1
            for cell_h in range(3, 6):
                for cell_w in range(3, 6):
                    rows = (top_height + SLOT_GAP) // (cell_h + SLOT_GAP)
                    cols = (width + SLOT_GAP) // (cell_w + SLOT_GAP)
                    if rows < 2 or cols < 2:
                        continue
                    cap = cell_h * cell_w
                    min_slots = sum((area + cap - 1) // cap for area in layer_areas)
                    if rows * cols >= min_slots:
                        options.append((height, width, cell_h, cell_w))
    return options


def _choose_piece_counts(layer_areas: list[int], cap: int, slot_budget: int) -> list[int]:
    piece_counts = [(area + cap - 1) // cap for area in layer_areas]
    slack = slot_budget - sum(piece_counts)
    extra = random.randint(0, min(slack, len(layer_areas) * 3))
    for _ in range(extra):
        choices = [i for i, area in enumerate(layer_areas) if piece_counts[i] < area]
        if not choices:
            break
        piece_counts[random.choice(choices)] += 1
    return piece_counts


def _partition_area(total: int, parts: int, max_part: int) -> list[int]:
    result: list[int] = []
    remaining = total
    for i in range(parts):
        left = parts - i - 1
        low = max(1, remaining - max_part * left)
        high = min(max_part, remaining - left)
        value = random.randint(low, high)
        result.append(value)
        remaining -= value
    random.shuffle(result)
    return result


def _slot_origins(height: int, width: int, cell_h: int, cell_w: int) -> list[tuple[int, int]]:
    rows = (height + SLOT_GAP) // (cell_h + SLOT_GAP)
    cols = (width + SLOT_GAP) // (cell_w + SLOT_GAP)
    used_h = rows * cell_h + (rows - 1) * SLOT_GAP
    used_w = cols * cell_w + (cols - 1) * SLOT_GAP
    start_r = random.randint(0, height - used_h)
    start_c = random.randint(0, width - used_w)

    origins: list[tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            rr = start_r + r * (cell_h + SLOT_GAP)
            cc = start_c + c * (cell_w + SLOT_GAP)
            origins.append((rr, cc))
    return origins


def _random_connected_patch(max_h: int, max_w: int, area: int) -> np.ndarray:
    assert 1 <= area <= max_h * max_w
    cells = {(random.randrange(max_h), random.randrange(max_w))}
    while len(cells) < area:
        frontier = set()
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < max_h and 0 <= cc < max_w and (rr, cc) not in cells:
                    frontier.add((rr, cc))
        cells.add(random.choice(tuple(frontier)))

    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in cells:
        patch[r - r0, c - c0] = 1
    if random.random() < 0.5:
        patch = np.flipud(patch)
    if random.random() < 0.5:
        patch = np.fliplr(patch)
    return patch


def _paint_fragments(
    grid: np.ndarray,
    colors: list[int],
    layer_areas: list[int],
    cell_h: int,
    cell_w: int,
) -> None:
    top_height = grid.shape[0] - 1
    slot_list = _slot_origins(top_height, grid.shape[1], cell_h, cell_w)
    cap = cell_h * cell_w
    piece_counts = _choose_piece_counts(layer_areas, cap, len(slot_list))

    pieces: list[tuple[int, int]] = []
    for color, area, count in zip(colors, layer_areas, piece_counts):
        for part in _partition_area(area, count, cap):
            pieces.append((color, part))

    random.shuffle(slot_list)
    random.shuffle(pieces)
    assert len(pieces) <= len(slot_list)

    for (color, area), (r0, c0) in zip(pieces, slot_list):
        patch = _random_connected_patch(cell_h, cell_w, area)
        off_r = random.randint(0, cell_h - patch.shape[0])
        off_c = random.randint(0, cell_w - patch.shape[1])
        rr = r0 + off_r
        cc = c0 + off_c
        mask = patch.astype(bool)
        grid[rr : rr + patch.shape[0], cc : cc + patch.shape[1]][mask] = color


def _add_legend(grid: np.ndarray, colors: list[int]) -> bool:
    positions = _pick_spaced_positions(grid.shape[1], len(colors), 2)
    if positions is None:
        return False
    for color, col in zip(colors, positions):
        grid[-1, col] = color
    return True


def _gen_attempt() -> np.ndarray | None:
    colors, layer_areas, side = _choose_ring_spec()
    layout_choices = _layout_options(layer_areas, len(colors))
    if not layout_choices:
        return None

    height, width, cell_h, cell_w = random.choice(layout_choices)
    grid = np.zeros((height, width), dtype=int)
    _paint_fragments(grid, colors, layer_areas, cell_h, cell_w)
    if not _add_legend(grid, colors):
        return None

    upper_colors = {int(value) for value in grid[:-1].ravel() if int(value) != BG}
    if upper_colors != set(colors):
        return None

    expected = _render_rings(colors, side, side)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
