from __future__ import annotations

import random

import numpy as np


OUTPUT_COLORS = [2, 1, 3, 4, 6]  # red, blue, green, yellow, magenta
INPUT_PALETTE = [0, 5, 7, 8, 9]

MIN_PANELS = 3
MAX_PANELS = 5
MIN_PANEL_HEIGHT = 4
MAX_PANEL_HEIGHT = 8
MIN_PANEL_WIDTH = 4
MAX_PANEL_WIDTH = 8
GENERATION_ATTEMPTS = 400


def _majority_panel(panels: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(panels, axis=0)
    num_panels, height, width = stack.shape
    base = np.empty((height, width), dtype=int)

    for r in range(height):
        for c in range(width):
            values, counts = np.unique(stack[:, r, c], return_counts=True)
            best = int(counts.max())
            winners = values[counts == best]
            assert winners.size == 1, "majority vote must be unique at every local coordinate"
            base[r, c] = int(winners[0])

    return base


def _candidate_layouts(grid: np.ndarray) -> list[dict]:
    height, width = grid.shape
    candidates: list[dict] = []

    for num_panels in range(MIN_PANELS, MAX_PANELS + 1):
        numer = width - (num_panels - 1)
        if numer <= 0 or numer % num_panels != 0:
            continue

        panel_width = numer // num_panels
        if panel_width <= 0:
            continue

        starts = [idx * (panel_width + 1) for idx in range(num_panels)]
        separator_cols = [start + panel_width for start in starts[:-1]]

        separator_values = []
        ok = True
        for col in separator_cols:
            values = np.unique(grid[:, col])
            if values.size != 1:
                ok = False
                break
            separator_values.append(int(values[0]))
        if not ok or len(set(separator_values)) != 1:
            continue

        panels = [grid[:, start : start + panel_width] for start in starts]
        panel_colors = sorted({int(v) for panel in panels for v in np.unique(panel)})
        if len(panel_colors) != 2:
            continue

        try:
            base = _majority_panel(panels)
        except AssertionError:
            continue

        base_colors, base_counts = np.unique(base, return_counts=True)
        if base_colors.size != 2:
            continue
        if np.count_nonzero(base_counts == base_counts.max()) != 1:
            continue

        anomalies: list[tuple[int, int]] = []
        for panel in panels:
            coords = np.argwhere(panel != base)
            if len(coords) != 1:
                ok = False
                break
            anomalies.append(tuple(int(x) for x in coords[0]))
        if not ok:
            continue

        candidates.append(
            {
                "num_panels": num_panels,
                "panel_width": panel_width,
                "panels": panels,
                "panel_colors": panel_colors,
                "separator_color": separator_values[0],
                "base": base,
                "anomalies": anomalies,
            }
        )

    return candidates


def _select_layout(grid: np.ndarray) -> dict:
    candidates = _candidate_layouts(grid)
    distinct_separator = [
        cand for cand in candidates if cand["separator_color"] not in set(cand["panel_colors"])
    ]
    if len(distinct_separator) == 1:
        return distinct_separator[0]
    assert len(candidates) == 1, "panel layout must be uniquely inferable"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find each panel's unique anomalous pixel by majority vote and paint those
    positions onto a fresh single-panel canvas in panel order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    layout = _select_layout(g)
    base = layout["base"]
    panel_width = int(layout["panel_width"])
    num_panels = int(layout["num_panels"])

    colors, counts = np.unique(base, return_counts=True)
    bg = int(colors[np.argmax(counts)])

    out = np.full((g.shape[0], panel_width), bg, dtype=int)
    for idx, (r, c) in enumerate(layout["anomalies"]):
        out[int(r), int(c)] = OUTPUT_COLORS[idx]

    assert num_panels <= len(OUTPUT_COLORS), "unsupported panel count"
    return out


def _neighbors4(r: int, c: int, height: int, width: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < height and 0 <= nc < width:
            yield nr, nc


def _random_polyomino(height: int, width: int, size: int) -> np.ndarray:
    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    frontier = set(_neighbors4(start[0], start[1], height, width))

    while len(cells) < size:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        for nb in _neighbors4(cell[0], cell[1], height, width):
            if nb not in cells:
                frontier.add(nb)

    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _generate_base_mask(height: int, width: int, num_panels: int) -> np.ndarray | None:
    area = height * width
    min_fg = max(num_panels, area // 5)
    max_fg = min(area - num_panels, (area - 1) // 2)
    if min_fg > max_fg:
        return None

    fg_size = random.randint(min_fg, max_fg)
    mask = _random_polyomino(height, width, fg_size)

    fg_count = int(mask.sum())
    bg_count = int(area - fg_count)
    if fg_count < num_panels or bg_count < num_panels:
        return None
    return mask


def _assign_anomalies(mask: np.ndarray, num_panels: int) -> list[tuple[int, int]] | None:
    fg_coords = [tuple(int(x) for x in rc) for rc in np.argwhere(mask)]
    bg_coords = [tuple(int(x) for x in rc) for rc in np.argwhere(~mask)]

    num_add = random.randint(1, num_panels - 1)
    num_del = num_panels - num_add
    if len(bg_coords) < num_add or len(fg_coords) < num_del:
        return None

    types = ["add"] * num_add + ["del"] * num_del
    assignments: list[tuple[tuple[int, int], str]] = []

    duplicate_branch = num_panels == 5 and random.random() < 0.5
    if duplicate_branch:
        eligible_types = []
        if num_add >= 2:
            eligible_types.append("add")
        if num_del >= 2:
            eligible_types.append("del")
        if not eligible_types:
            return None

        dup_type = random.choice(eligible_types)
        pool = bg_coords if dup_type == "add" else fg_coords
        dup_coord = random.choice(pool)
        assignments.extend([(dup_coord, dup_type), (dup_coord, dup_type)])
        types.remove(dup_type)
        types.remove(dup_type)

    used_add = {coord for coord, kind in assignments if kind == "add"}
    used_del = {coord for coord, kind in assignments if kind == "del"}

    random.shuffle(types)
    for kind in types:
        if kind == "add":
            pool = [coord for coord in bg_coords if coord not in used_add]
            if not pool:
                return None
            coord = random.choice(pool)
            used_add.add(coord)
        else:
            pool = [coord for coord in fg_coords if coord not in used_del]
            if not pool:
                return None
            coord = random.choice(pool)
            used_del.add(coord)
        assignments.append((coord, kind))

    random.shuffle(assignments)
    return [coord for coord, _kind in assignments]


def _build_input(
    mask: np.ndarray,
    anomalies: list[tuple[int, int]],
    bg: int,
    fg: int,
    separator: int,
) -> np.ndarray:
    height, panel_width = mask.shape
    num_panels = len(anomalies)
    width = num_panels * panel_width + (num_panels - 1)

    grid = np.full((height, width), separator, dtype=int)
    base_panel = np.where(mask, fg, bg).astype(int)

    for idx, coord in enumerate(anomalies):
        start = idx * (panel_width + 1)
        panel = base_panel.copy()
        r, c = coord
        panel[r, c] = fg if panel[r, c] == bg else bg
        grid[:, start : start + panel_width] = panel

    return grid


def _expected_output(
    shape: tuple[int, int], anomalies: list[tuple[int, int]], bg: int
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for idx, (r, c) in enumerate(anomalies):
        out[r, c] = OUTPUT_COLORS[idx]
    return out


def _gen_attempt() -> np.ndarray | None:
    num_panels = random.choice([3, 4, 5, 5])
    height = random.randint(MIN_PANEL_HEIGHT, MAX_PANEL_HEIGHT)
    panel_width = random.randint(MIN_PANEL_WIDTH, MAX_PANEL_WIDTH)

    colors = random.sample(INPUT_PALETTE, 3)
    bg, fg, separator = colors

    mask = _generate_base_mask(height, panel_width, num_panels)
    if mask is None:
        return None

    anomalies = _assign_anomalies(mask, num_panels)
    if anomalies is None:
        return None

    inp = _build_input(mask, anomalies, bg, fg, separator)
    expected = _expected_output((height, panel_width), anomalies, bg)

    chosen = _select_layout(inp)
    filtered = [
        cand for cand in _candidate_layouts(inp) if cand["separator_color"] not in set(cand["panel_colors"])
    ]
    if len(filtered) != 1 or chosen["num_panels"] != num_panels:
        return None

    try:
        actual = main(inp)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid panel-anomaly sample")
