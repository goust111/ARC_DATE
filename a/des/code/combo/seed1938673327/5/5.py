import random

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GENERATION_ATTEMPTS = 400

WIDTH_RANGE = (7, 18)
TOP_HEIGHT_RANGE = (4, 9)
LOWER_HEIGHT_RANGE = (4, 12)
COLOR_COUNT_RANGE = (2, 5)


def _separator_row(grid: np.ndarray) -> int:
    non_bg_rows = np.flatnonzero(np.any(grid != BG, axis=1))
    assert non_bg_rows.size > 0, "input must contain a non-black top panel"
    sep = int(non_bg_rows[-1]) + 1
    assert sep < int(grid.shape[0]), "input must contain a separator row and a lower area"
    return sep


def _panel_color_specs(panel: np.ndarray) -> list[tuple[int, int, int, int]]:
    specs: list[tuple[int, int, int, int]] = []
    for color in (int(v) for v in np.unique(panel) if int(v) != BG):
        coords = np.argwhere(panel == color)
        assert coords.size > 0
        count = int(coords.shape[0])
        first_col = int(coords[:, 1].min())
        first_row = int(coords[coords[:, 1] == first_col][:, 0].min())
        specs.append((first_col, first_row, color, count))
    specs.sort()
    return specs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert the colored top panel into a frequency bar chart in the lower area.

    The separator is the first full-black row after the last non-black row of the
    panel. Every non-black color in the panel contributes one bottom-aligned bar
    whose height equals its pixel count. Bars are ordered by the left-to-right
    position of each color's first appearing column and are packed from the left
    edge with one black column between adjacent bars.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    height, width = map(int, g.shape)

    sep = _separator_row(g)
    panel = g[:sep, :]
    lower_top = sep + 1
    lower_height = height - lower_top
    assert lower_height > 0

    specs = _panel_color_specs(panel)
    assert specs, "panel must contain at least one non-black color"
    assert 2 * len(specs) - 1 <= width

    out = g.copy()
    out[lower_top:, :] = BG
    for idx, (_first_col, _first_row, color, count) in enumerate(specs):
        assert 1 <= count <= lower_height
        col = 2 * idx
        out[height - count : height, col] = color
    return out


def _sample_counts(top_height: int, width: int, lower_height: int, color_count: int) -> list[int] | None:
    area = top_height * width
    density_cap = max(color_count + 1, int(area * 0.62))
    for _ in range(80):
        counts = random.sample(range(1, lower_height + 1), color_count)
        total = sum(counts)
        if total <= density_cap and total < area and max(counts) >= min(3, lower_height):
            return counts
    return None


def _sample_lead_columns(top_height: int, width: int, counts: list[int]) -> list[int] | None:
    color_count = len(counts)
    for _ in range(120):
        cols = sorted(random.sample(range(width), color_count))
        if all(sum(counts[i:]) <= top_height * (width - cols[i]) for i in range(color_count)):
            return cols
    return None


def _place_panel(top_height: int, width: int, specs: list[tuple[int, int, int]]) -> np.ndarray | None:
    panel = np.full((top_height, width), BG, dtype=int)
    anchor_rows = [random.randrange(top_height) for _ in specs]
    anchor_rows[random.randrange(len(specs))] = top_height - 1

    for idx in range(len(specs) - 1, -1, -1):
        lead_col, color, count = specs[idx]
        anchor_row = anchor_rows[idx]
        if panel[anchor_row, lead_col] != BG:
            return None

        panel[anchor_row, lead_col] = color
        cells = {(anchor_row, lead_col)}

        while len(cells) < count:
            frontier: list[tuple[int, int]] = []
            seen = set()
            for row, col in cells:
                for dr, dc in NEIGHBORS4:
                    rr = row + dr
                    cc = col + dc
                    if not (0 <= rr < top_height and lead_col <= cc < width):
                        continue
                    if panel[rr, cc] != BG or (rr, cc) in seen:
                        continue
                    frontier.append((rr, cc))
                    seen.add((rr, cc))

            if frontier and random.random() < 0.82:
                rr, cc = random.choice(frontier)
            else:
                free_cells = [
                    (row, col)
                    for row in range(top_height)
                    for col in range(lead_col, width)
                    if panel[row, col] == BG
                ]
                if not free_cells:
                    return None
                rr, cc = random.choice(free_cells)

            panel[rr, cc] = color
            cells.add((rr, cc))

    return panel


def _build_expected(panel: np.ndarray, lower_height: int) -> np.ndarray:
    top_height, width = map(int, panel.shape)
    grid = np.zeros((top_height + 1 + lower_height, width), dtype=int)
    grid[:top_height, :] = panel
    out = grid.copy()
    ordered = _panel_color_specs(panel)
    for idx, (_first_col, _first_row, color, count) in enumerate(ordered):
        out[-count:, 2 * idx] = color
    return out


def _validate_instance(panel: np.ndarray, ordered_specs: list[tuple[int, int, int]], lower_height: int) -> bool:
    observed = _panel_color_specs(panel)
    observed_triples = [(first_col, color, count) for first_col, _first_row, color, count in observed]
    if observed_triples != ordered_specs:
        return False

    inp = np.zeros((panel.shape[0] + 1 + lower_height, panel.shape[1]), dtype=int)
    inp[: panel.shape[0], :] = panel
    expected = _build_expected(panel, lower_height)

    try:
        pred = main(inp)
    except Exception:
        return False
    return bool(np.array_equal(pred, expected) and not np.array_equal(inp, expected))


def _gen_attempt() -> np.ndarray | None:
    color_count = random.randint(*COLOR_COUNT_RANGE)
    top_height = random.randint(*TOP_HEIGHT_RANGE)
    lower_height = random.randint(max(LOWER_HEIGHT_RANGE[0], color_count), LOWER_HEIGHT_RANGE[1])
    width = random.randint(max(WIDTH_RANGE[0], 2 * color_count - 1), WIDTH_RANGE[1])
    if top_height + 1 + lower_height > 30:
        return None

    counts = _sample_counts(top_height, width, lower_height, color_count)
    if counts is None:
        return None

    lead_cols = _sample_lead_columns(top_height, width, counts)
    if lead_cols is None:
        return None

    colors = random.sample(COLORS, color_count)
    ordered_specs = sorted((lead_col, color, count) for lead_col, color, count in zip(lead_cols, colors, counts))
    panel = _place_panel(top_height, width, ordered_specs)
    if panel is None:
        return None

    if not _validate_instance(panel, ordered_specs, lower_height):
        return None

    inp = np.zeros((top_height + 1 + lower_height, width), dtype=int)
    inp[:top_height, :] = panel
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
