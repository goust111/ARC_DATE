import random

import numpy as np


BLUE = 1
BLACK = 0
SEPARATOR = 5
MUTATION_COLORS = (2, 3, 4, 6, 7, 8, 9)


def _runs(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    runs = []
    start = prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        runs.append((start, prev))
        start = prev = idx
    runs.append((start, prev))
    return runs


def _spans_around_separator(length: int, separator_indices: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for start, end in _runs(separator_indices):
        if cursor < start:
            spans.append((cursor, start))
        cursor = end + 1
    if cursor < length:
        spans.append((cursor, length))
    return spans


def _panel_layout(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    row_sep = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == SEPARATOR)]
    col_sep = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == SEPARATOR)]

    row_spans = _spans_around_separator(grid.shape[0], row_sep)
    col_spans = _spans_around_separator(grid.shape[1], col_sep)

    assert len(row_spans) == 2, "expected exactly two panel rows"
    assert len(col_spans) >= 1, "expected at least one panel column"
    assert len({r1 - r0 for r0, r1 in row_spans}) == 1, "panel rows must have equal height"
    assert len({c1 - c0 for c0, c1 in col_spans}) == 1, "panel columns must have equal width"

    return row_spans, col_spans


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the mutation coordinates from each lower panel and plot them on black output panels.

    Input structure
    - A grey separator lattice partitions the grid into equal panels arranged in two rows.
    - Each top panel is a blue template on black.
    - Each bottom panel matches the template except that some blue pixels are recolored.

    Rule
    - Pair each top panel with the panel directly below it.
    - Compare the two panels cell by cell.
    - Build a black output panel of the same size and place each mutation color at its
      differing coordinate.
    - Concatenate the output panels left-to-right with the same grey separators.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    row_spans, col_spans = _panel_layout(g)
    (top_r0, top_r1), (bot_r0, bot_r1) = row_spans
    panel_h = top_r1 - top_r0
    panel_w = col_spans[0][1] - col_spans[0][0]

    sep_widths = [col_spans[i + 1][0] - col_spans[i][1] for i in range(len(col_spans) - 1)]

    out_w = len(col_spans) * panel_w + sum(sep_widths)
    out = np.full((panel_h, out_w), BLACK, dtype=int)

    cursor = 0
    for index, (c0, c1) in enumerate(col_spans):
        template = g[top_r0:top_r1, c0:c1]
        variant = g[bot_r0:bot_r1, c0:c1]

        diff_mask = variant != template
        panel_out = np.full_like(template, BLACK)
        panel_out[diff_mask] = variant[diff_mask]
        out[:, cursor : cursor + panel_w] = panel_out
        cursor += panel_w

        if index < len(sep_widths):
            sw = sep_widths[index]
            out[:, cursor : cursor + sw] = SEPARATOR
            cursor += sw

    return out


# --------------- generator ---------------

PAIR_COUNT_RANGE = (2, 5)
PANEL_HEIGHT_RANGE = (4, 8)
PANEL_WIDTH_RANGE = (4, 8)
SEPARATOR_WIDTH_CHOICES = (1, 2)
MAX_GENERATION_ATTEMPTS = 300
PANEL_BUILD_ATTEMPTS = 40
SHAPE_MODES = ("blob", "bars", "frame", "mirror")


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _blob_mask(rng: random.Random, h: int, w: int, target: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    start = (rng.randrange(h), rng.randrange(w))
    frontier = [start]
    mask[start] = True
    while frontier and int(mask.sum()) < target:
        r, c = frontier[rng.randrange(len(frontier))]
        nbrs = [(nr, nc) for nr, nc in _neighbors4(r, c, h, w) if not mask[nr, nc]]
        if not nbrs:
            frontier.remove((r, c))
            continue
        nr, nc = rng.choice(nbrs)
        mask[nr, nc] = True
        frontier.append((nr, nc))
    return mask


def _bars_mask(rng: random.Random, h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    count = rng.randint(2, 4)
    for _ in range(count):
        if rng.random() < 0.5:
            row = rng.randrange(h)
            c0 = rng.randrange(w)
            c1 = rng.randint(c0, w - 1)
            mask[row, c0 : c1 + 1] = True
        else:
            col = rng.randrange(w)
            r0 = rng.randrange(h)
            r1 = rng.randint(r0, h - 1)
            mask[r0 : r1 + 1, col] = True
    return mask


def _frame_mask(rng: random.Random, h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    if h < 3 or w < 3:
        return _bars_mask(rng, h, w)

    r0 = rng.randrange(0, h - 2)
    r1 = rng.randrange(r0 + 2, h)
    c0 = rng.randrange(0, w - 2)
    c1 = rng.randrange(c0 + 2, w)
    if rng.random() < 0.5:
        mask[r0 : r1 + 1, c0 : c1 + 1] = True
    else:
        mask[r0, c0 : c1 + 1] = True
        mask[r1, c0 : c1 + 1] = True
        mask[r0 : r1 + 1, c0] = True
        mask[r0 : r1 + 1, c1] = True
    return mask


def _mirror_mask(rng: random.Random, h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    axis_vertical = rng.random() < 0.5
    if axis_vertical:
        half_w = (w + 1) // 2
        seed = rng.random() < 0.35
        for r in range(h):
            for c in range(half_w):
                if rng.random() < 0.35:
                    mask[r, c] = True
                    mask[r, w - 1 - c] = True
                elif seed:
                    seed = False
                    mask[r, c] = True
                    mask[r, w - 1 - c] = True
    else:
        half_h = (h + 1) // 2
        seed = rng.random() < 0.35
        for r in range(half_h):
            for c in range(w):
                if rng.random() < 0.35:
                    mask[r, c] = True
                    mask[h - 1 - r, c] = True
                elif seed:
                    seed = False
                    mask[r, c] = True
                    mask[h - 1 - r, c] = True
    return mask


def _template_mask(rng: random.Random, h: int, w: int) -> np.ndarray | None:
    target = rng.randint(max(4, (h * w) // 4), max(5, (h * w * 3) // 5))
    mode = rng.choice(SHAPE_MODES)
    if mode == "blob":
        mask = _blob_mask(rng, h, w, target)
    elif mode == "bars":
        mask = _bars_mask(rng, h, w)
    elif mode == "frame":
        mask = _frame_mask(rng, h, w)
    else:
        mask = _mirror_mask(rng, h, w)

    count = int(mask.sum())
    if count < 4 or count >= h * w:
        return None

    rows = np.any(mask, axis=1).sum()
    cols = np.any(mask, axis=0).sum()
    if rows < 2 or cols < 2:
        return None

    return mask


def _mutate_panel(
    rng: random.Random, template: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    blue_positions = np.argwhere(mask)
    blue_count = len(blue_positions)
    if blue_count < 4:
        return None

    variant = template.copy()
    max_diff = min(max(2, blue_count // 2), blue_count - 1)
    diff_count = rng.randint(1, max_diff)
    chosen = rng.sample(range(blue_count), diff_count)

    same_color_mode = rng.random() < 0.5
    base_color = rng.choice(MUTATION_COLORS)
    used = np.zeros_like(template, dtype=bool)
    for idx in chosen:
        r, c = map(int, blue_positions[idx])
        color = base_color if same_color_mode else rng.choice(MUTATION_COLORS)
        variant[r, c] = color
        used[r, c] = True

    return variant, used


def _build_panel(rng: random.Random, h: int, w: int) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(PANEL_BUILD_ATTEMPTS):
        mask = _template_mask(rng, h, w)
        if mask is None:
            continue

        template = np.full((h, w), BLACK, dtype=int)
        template[mask] = BLUE

        mutated = _mutate_panel(rng, template, mask)
        if mutated is None:
            continue
        variant, diff_mask = mutated

        assert np.any(diff_mask)
        assert np.all(
            (variant == template)
            | ((template == BLUE) & (variant != BLUE) & (variant != BLACK) & (variant != SEPARATOR))
        )
        return template, variant
    return None


def _expected_output(panels: list[tuple[np.ndarray, np.ndarray]], sep_w: int) -> np.ndarray:
    panel_h, panel_w = panels[0][0].shape
    out_w = len(panels) * panel_w + (len(panels) - 1) * sep_w
    out = np.full((panel_h, out_w), BLACK, dtype=int)
    cursor = 0
    for idx, (template, variant) in enumerate(panels):
        diff_mask = variant != template
        out[:, cursor : cursor + panel_w][diff_mask] = variant[diff_mask]
        cursor += panel_w
        if idx + 1 < len(panels):
            out[:, cursor : cursor + sep_w] = SEPARATOR
            cursor += sep_w
    return out


def _assemble_grid(panels: list[tuple[np.ndarray, np.ndarray]], sep_w: int) -> np.ndarray:
    panel_h, panel_w = panels[0][0].shape
    pair_count = len(panels)
    grid_h = panel_h * 2 + sep_w
    grid_w = pair_count * panel_w + (pair_count - 1) * sep_w
    grid = np.full((grid_h, grid_w), SEPARATOR, dtype=int)

    top_r0 = 0
    bot_r0 = panel_h + sep_w
    cursor = 0
    for template, variant in panels:
        grid[top_r0 : top_r0 + panel_h, cursor : cursor + panel_w] = template
        grid[bot_r0 : bot_r0 + panel_h, cursor : cursor + panel_w] = variant
        cursor += panel_w + sep_w

    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pair_count = rng.randint(*PAIR_COUNT_RANGE)
    panel_h = rng.randint(*PANEL_HEIGHT_RANGE)
    panel_w = rng.randint(*PANEL_WIDTH_RANGE)
    sep_w = rng.choice(SEPARATOR_WIDTH_CHOICES)

    panels: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(pair_count):
        panel = _build_panel(rng, panel_h, panel_w)
        if panel is None:
            return None
        panels.append(panel)

    grid = _assemble_grid(panels, sep_w)

    row_spans, col_spans = _panel_layout(grid)
    if len(col_spans) != pair_count:
        return None

    out = main(grid)
    expected = _expected_output(panels, sep_w)
    if out.shape != expected.shape:
        return None
    if not np.array_equal(out, expected):
        return None
    if not np.any((out != BLACK) & (out != SEPARATOR)):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
