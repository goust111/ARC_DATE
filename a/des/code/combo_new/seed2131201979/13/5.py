import random

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
FRAME_COLOR_POOL = (1, 2, 3, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 300


def _depth_grid(height: int, width: int) -> np.ndarray:
    rows = np.arange(height)[:, None]
    cols = np.arange(width)[None, :]
    return np.minimum(np.minimum(rows, height - 1 - rows), np.minimum(cols, width - 1 - cols))


def _find_separator(grid: np.ndarray) -> tuple[int, int]:
    height, width = grid.shape
    candidates: list[tuple[int, int]] = []
    for col in range(width):
        value = int(grid[0, col])
        if value == BLACK:
            continue
        if not np.all(grid[:, col] == value):
            continue
        if not (grid[:, :col] != BLACK).any():
            continue
        if not (grid[:, col + 1 :] != BLACK).any():
            continue
        candidates.append((col, value))
    assert candidates, "expected a full-height separator column"
    return candidates[0]


def _parse_legend(grid: np.ndarray, separator_col: int) -> tuple[set[int], set[int]]:
    legend_colors: list[int] = []
    kept_colors: set[int] = set()
    for row in range(grid.shape[0]):
        non_black = [col for col in range(separator_col) if int(grid[row, col]) != BLACK]
        if not non_black:
            continue
        values = [int(grid[row, col]) for col in non_black]
        color_cells = [(col, value) for col, value in zip(non_black, values) if value != YELLOW]
        assert len(color_cells) == 1, "each legend row must contain exactly one frame color"
        color_col, color = color_cells[0]
        legend_colors.append(color)
        if color_col + 1 < separator_col and int(grid[row, color_col + 1]) == YELLOW:
            kept_colors.add(color)
    assert 3 <= len(legend_colors) <= 5, "legend must list the frame colors"
    assert len(set(legend_colors)) == len(legend_colors), "legend colors must be unique"
    return set(legend_colors), kept_colors


def _find_canvas_bbox(grid: np.ndarray, separator_col: int, separator_color: int) -> tuple[int, int, int, int]:
    right = grid[:, separator_col + 1 :]
    mask = (right != BLACK) & (right != separator_color)
    coords = np.argwhere(mask)
    assert len(coords) > 0, "expected a colored canvas to the right of the separator"
    top, left = coords.min(axis=0)
    bottom, right_col = coords.max(axis=0)
    return int(top), int(left + separator_col + 1), int(bottom), int(right_col + separator_col + 1)


def _extract_layers(canvas: np.ndarray, legend_colors: set[int]) -> tuple[list[int], int]:
    depth = _depth_grid(*canvas.shape)
    layers: list[int] = []
    fill_color: int | None = None
    for level in range(int(depth.max()) + 1):
        ring_colors = np.unique(canvas[depth == level])
        assert len(ring_colors) == 1, "each depth ring must have a single color"
        color = int(ring_colors[0])
        if color in legend_colors:
            layers.append(color)
        else:
            fill_color = color
            break
    assert fill_color is not None, "expected a distinct innermost fill color"
    assert len(layers) == len(legend_colors), "legend must describe every frame layer"
    return layers, fill_color


def _visible_bands(layers: list[int], kept_colors: set[int], fill_color: int) -> list[int]:
    visible = [fill_color] * len(layers)
    next_visible = fill_color
    for idx in range(len(layers) - 1, -1, -1):
        color = layers[idx]
        if color in kept_colors:
            next_visible = color
            visible[idx] = color
        else:
            visible[idx] = next_visible
    return visible


def _render_stack(band_colors: list[int], fill_color: int, fill_height: int, fill_width: int) -> np.ndarray:
    height = 2 * len(band_colors) + fill_height
    width = 2 * len(band_colors) + fill_width
    depth = _depth_grid(height, width)
    canvas = np.full((height, width), int(fill_color), dtype=int)
    for level, color in enumerate(band_colors):
        canvas[depth == level] = int(color)
    return canvas


def _solve_canvas(canvas: np.ndarray, legend_colors: set[int], kept_colors: set[int]) -> np.ndarray:
    layers, fill_color = _extract_layers(canvas, legend_colors)
    visible = _visible_bands(layers, kept_colors, fill_color)
    fill_height = canvas.shape[0] - 2 * len(layers)
    fill_width = canvas.shape[1] - 2 * len(layers)
    return _render_stack(visible, fill_color, fill_height, fill_width)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reveal only the legend-marked layers of a concentric frame stack by making unmarked frames transparent.

    Input structure:
    - The background is black, and a full-height grey separator column splits the grid into a left legend area and a right canvas area.
    - The left area contains one vertical list of single-pixel frame colors; some rows also have a yellow pixel immediately to the right of the listed color.
    - The right area contains one solid rectangle made of 3-5 concentric one-pixel frame layers with distinct colors, plus a distinct solid fill color inside the innermost frame.

    Rule:
    - Read the left legend and treat every frame color with an adjacent yellow pixel as a kept layer; listed colors without yellow are removed layers.
    - On the right canvas, preserve kept frame layers exactly where they are.
    - Replace every removed frame layer with the nearest visible color from deeper inside the stack, skipping through chains of removed layers until reaching either a kept frame or the innermost fill color.

    Key insight:
    - The legend is a color-to-keep/remove lookup table, and removing a frame does not create black space; it reveals the first surviving inner layer, so consecutive removals must be resolved recursively inward.
    """
    grid = np.asarray(grid, dtype=int)
    separator_col, separator_color = _find_separator(grid)
    legend_colors, kept_colors = _parse_legend(grid, separator_col)

    top, left, bottom, right = _find_canvas_bbox(grid, separator_col, separator_color)
    canvas = grid[top : bottom + 1, left : right + 1]

    out = grid.copy()
    out[top : bottom + 1, left : right + 1] = _solve_canvas(canvas, legend_colors, kept_colors)
    return out


def _keep_patterns(layer_count: int) -> tuple[list[list[bool]], list[list[bool]]]:
    scatter: list[list[bool]] = []
    run: list[list[bool]] = []
    for mask in range(1 << layer_count):
        flags = [bool(mask & (1 << idx)) for idx in range(layer_count)]
        if all(flags) or not any(flags):
            continue
        has_removed_run = any((not flags[idx]) and (not flags[idx + 1]) for idx in range(layer_count - 1))
        if has_removed_run:
            run.append(flags)
        else:
            scatter.append(flags)
    return scatter, run


def _sample_keep_flags(layer_count: int, rng: random.Random) -> list[bool]:
    scatter, run = _keep_patterns(layer_count)
    use_run = rng.random() < 0.5
    options = run if use_run and run else scatter if scatter else run
    return list(rng.choice(options))


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    layer_count = rng.randint(3, 5)
    palette = list(FRAME_COLOR_POOL)
    rng.shuffle(palette)
    frame_colors = palette[:layer_count]
    fill_color = palette[layer_count]
    keep_flags = _sample_keep_flags(layer_count, rng)
    kept_by_color = {color: keep for color, keep in zip(frame_colors, keep_flags)}

    fill_height = rng.randint(1, 6)
    fill_width = rng.randint(1, 8)
    canvas = _render_stack(frame_colors, fill_color, fill_height, fill_width)
    out_canvas = _render_stack(_visible_bands(frame_colors, set(color for color, keep in kept_by_color.items() if keep), fill_color), fill_color, fill_height, fill_width)

    legend_col = rng.randint(0, 2)
    left_gap = rng.randint(1, 3)
    separator_col = legend_col + 2 + left_gap
    right_gap = rng.randint(1, 3)
    canvas_left = separator_col + 1 + right_gap
    right_margin = rng.randint(1, 3)
    width = canvas_left + canvas.shape[1] + right_margin
    if width > 30:
        return None

    height = canvas.shape[0] + rng.randint(2, 6)
    if height > 30 or height < layer_count + 2:
        return None

    canvas_top = rng.randint(1, height - canvas.shape[0] - 1)
    legend_top = rng.randint(1, height - layer_count - 1)
    legend_order = frame_colors[:]
    rng.shuffle(legend_order)

    grid = np.full((height, width), BLACK, dtype=int)
    grid[:, separator_col] = GREY
    for idx, color in enumerate(legend_order):
        row = legend_top + idx
        grid[row, legend_col] = int(color)
        if kept_by_color[color]:
            grid[row, legend_col + 1] = YELLOW

    grid[canvas_top : canvas_top + canvas.shape[0], canvas_left : canvas_left + canvas.shape[1]] = canvas
    expected = grid.copy()
    expected[canvas_top : canvas_top + out_canvas.shape[0], canvas_left : canvas_left + out_canvas.shape[1]] = out_canvas

    if np.array_equal(grid, expected):
        return None
    try:
        solved = main(grid)
    except Exception:
        return None
    return grid if np.array_equal(solved, expected) else None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid onion-frame puzzle")
