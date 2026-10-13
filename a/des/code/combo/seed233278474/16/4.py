import random

import numpy as np


BG = 0
RED = 2
GREEN = 3
SEPARATOR = 5
OBJECT_COLORS = (1, 4, 6, 7, 8, 9)
LAYOUT_OPTIONS = ((2, 2), (2, 3), (3, 2))
MAX_GENERATION_ATTEMPTS = 400


def _find_separator_lines(grid: np.ndarray) -> tuple[int, list[int], list[int]]:
    height, width = map(int, grid.shape)
    candidates: list[tuple[int, int, list[int], list[int]]] = []

    for color in map(int, np.unique(grid)):
        rows = np.where(np.all(grid == color, axis=1))[0].astype(int).tolist()
        cols = np.where(np.all(grid == color, axis=0))[0].astype(int).tolist()
        if len(rows) < 2 or len(cols) < 2:
            continue
        if rows[0] != 0 or rows[-1] != height - 1 or cols[0] != 0 or cols[-1] != width - 1:
            continue

        row_steps = [b - a for a, b in zip(rows, rows[1:])]
        col_steps = [b - a for a, b in zip(cols, cols[1:])]
        if not row_steps or not col_steps:
            continue
        if len(set(row_steps)) != 1 or len(set(col_steps)) != 1:
            continue
        if row_steps[0] <= 1 or col_steps[0] <= 1:
            continue

        candidates.append((len(rows) * len(cols), color, rows, cols))

    assert candidates, "failed to identify the separator framework"
    _score, separator, rows, cols = max(candidates, key=lambda item: item[0])
    return int(separator), rows, cols


def _region_bounds(rows: list[int], cols: list[int]) -> list[tuple[int, int, int, int]]:
    bounds = []
    for row_idx in range(len(rows) - 1):
        for col_idx in range(len(cols) - 1):
            bounds.append((rows[row_idx] + 1, rows[row_idx + 1], cols[col_idx] + 1, cols[col_idx + 1]))
    return bounds


def _extract_main_block(region: np.ndarray, separator: int) -> dict[str, int]:
    candidates: list[tuple[int, int, int, int]] = []
    for color in map(int, np.unique(region)):
        if color in (BG, RED, GREEN, separator):
            continue
        coords = np.argwhere(region == color)
        assert len(coords) > 0
        (r0, c0), (r1, c1) = coords.min(axis=0), coords.max(axis=0)
        patch = region[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
        if np.all(patch == color):
            height = int(r1 - r0 + 1)
            width = int(c1 - c0 + 1)
            candidates.append((height * width, height, width, color))

    assert candidates, "each region must contain a solid main rectangle"
    best_area = max(candidate[0] for candidate in candidates)
    best = [candidate for candidate in candidates if candidate[0] == best_area]
    assert len(best) == 1, "main rectangle must be uniquely identifiable"
    area, height, width, color = best[0]
    return {"area": int(area), "height": int(height), "width": int(width), "color": int(color)}


def _render_output(
    shape: tuple[int, int],
    separator: int,
    rows: list[int],
    cols: list[int],
    blocks: list[dict[str, int]],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    out[rows, :] = separator
    out[:, cols] = separator

    bounds = _region_bounds(rows, cols)
    cell_height = bounds[0][1] - bounds[0][0]
    cell_width = bounds[0][3] - bounds[0][2]
    for (r0, _r1, c0, _c1), block in zip(bounds, blocks):
        top = int(r0 + (cell_height - block["height"]) // 2)
        left = int(c0 + (cell_width - block["width"]) // 2)
        out[top : top + block["height"], left : left + block["width"]] = block["color"]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    separator, rows, cols = _find_separator_lines(g)
    bounds = _region_bounds(rows, cols)
    assert bounds, "expected at least one region"

    row_steps = {rows[i + 1] - rows[i] for i in range(len(rows) - 1)}
    col_steps = {cols[i + 1] - cols[i] for i in range(len(cols) - 1)}
    assert len(row_steps) == 1 and len(col_steps) == 1, "regions must be equally sized"

    survivors: list[dict[str, int]] = []
    for source_index, (r0, r1, c0, c1) in enumerate(bounds):
        region = g[r0:r1, c0:c1]
        block = _extract_main_block(region, separator)
        block["source_index"] = int(source_index)

        green_votes = int(np.count_nonzero(region == GREEN))
        red_votes = int(np.count_nonzero(region == RED))
        if green_votes >= red_votes:
            survivors.append(block)

    survivors.sort(key=lambda block: (-block["area"], block["source_index"]))
    return _render_output(g.shape, separator, rows, cols, survivors)


def _blank_grid(n_rows: int, n_cols: int, cell_height: int, cell_width: int) -> tuple[np.ndarray, list[int], list[int]]:
    height = n_rows * cell_height + n_rows + 1
    width = n_cols * cell_width + n_cols + 1
    grid = np.full((height, width), BG, dtype=int)
    rows = [idx * (cell_height + 1) for idx in range(n_rows + 1)]
    cols = [idx * (cell_width + 1) for idx in range(n_cols + 1)]
    grid[rows, :] = SEPARATOR
    grid[:, cols] = SEPARATOR
    return grid, rows, cols


def _dimension_catalog(cell_height: int, cell_width: int) -> dict[int, list[tuple[int, int]]]:
    by_area: dict[int, list[tuple[int, int]]] = {}
    for height in range(2, cell_height - 1):
        if (cell_height - height) % 2 != 0:
            continue
        for width in range(2, cell_width - 1):
            if (cell_width - width) % 2 != 0:
                continue
            area = height * width
            by_area.setdefault(int(area), []).append((int(height), int(width)))
    return by_area


def _vote_slots(cell_height: int, cell_width: int, block_mask: np.ndarray) -> list[list[tuple[int, int]]]:
    slots = []
    for row_parity in range(2):
        for col_parity in range(2):
            coords = []
            for row in range(row_parity, cell_height, 2):
                for col in range(col_parity, cell_width, 2):
                    if not block_mask[row, col]:
                        coords.append((int(row), int(col)))
            if coords:
                slots.append(coords)
    return slots


def _choose_vote_counts(max_votes: int, survives: bool, use_tie: bool) -> tuple[int, int] | None:
    if use_tie:
        if max_votes < 2:
            return None
        pairs = random.randint(1, max_votes // 2)
        return pairs, pairs

    min_votes = 3 if max_votes >= 3 else 2
    if max_votes < min_votes:
        return None
    total_votes = random.randint(min_votes, max_votes)
    minority = random.randint(0, total_votes // 2)

    if survives:
        green_votes = total_votes - minority
        red_votes = minority
        if green_votes == red_votes:
            green_votes += 1
            red_votes -= 1
    else:
        green_votes = minority
        red_votes = total_votes - minority
        if green_votes == red_votes:
            green_votes -= 1
            red_votes += 1

    assert green_votes >= 0 and red_votes >= 0
    assert (green_votes >= red_votes) == survives
    return int(green_votes), int(red_votes)


def _expected_output(
    shape: tuple[int, int],
    rows: list[int],
    cols: list[int],
    specs: list[dict[str, int | bool]],
) -> np.ndarray:
    survivors = [spec for spec in specs if bool(spec["survives"])]
    survivors = sorted(survivors, key=lambda spec: (-int(spec["area"]), int(spec["source_index"])))
    compact = [
        {
            "area": int(spec["area"]),
            "height": int(spec["height"]),
            "width": int(spec["width"]),
            "color": int(spec["color"]),
        }
        for spec in survivors
    ]
    return _render_output(shape, SEPARATOR, rows, cols, compact)


def _gen_attempt() -> np.ndarray | None:
    n_rows, n_cols = random.choice(LAYOUT_OPTIONS)
    n_regions = n_rows * n_cols
    cell_height = random.randint(8, 11)
    cell_width = random.randint(8, 12)

    catalog = _dimension_catalog(cell_height, cell_width)
    if len(catalog) < n_regions:
        return None

    grid, rows, cols = _blank_grid(n_rows, n_cols, cell_height, cell_width)
    areas = random.sample(list(catalog), n_regions)
    colors = random.sample(list(OBJECT_COLORS), n_regions)

    n_survivors = random.randint(2, n_regions - 1)
    survivor_indices = set(random.sample(range(n_regions), n_survivors))
    tie_index = random.choice(list(survivor_indices)) if random.random() < 0.4 else None

    specs: list[dict[str, int | bool]] = []
    for source_index, (r0, r1, c0, c1) in enumerate(_region_bounds(rows, cols)):
        area = int(areas[source_index])
        height, width = random.choice(catalog[area])
        color = int(colors[source_index])

        local_top = random.randint(0, cell_height - height)
        local_left = random.randint(0, cell_width - width)
        grid[r0 + local_top : r0 + local_top + height, c0 + local_left : c0 + local_left + width] = color

        block_mask = np.zeros((cell_height, cell_width), dtype=bool)
        block_mask[local_top : local_top + height, local_left : local_left + width] = True
        slot_options = _vote_slots(cell_height, cell_width, block_mask)
        if not slot_options:
            return None

        survives = source_index in survivor_indices
        use_tie = survives and source_index == tie_index
        max_votes = min(7, max(len(slots) for slots in slot_options))
        vote_counts = _choose_vote_counts(max_votes, survives, use_tie)
        if vote_counts is None:
            return None
        green_votes, red_votes = vote_counts
        total_votes = green_votes + red_votes

        eligible_slots = [slots for slots in slot_options if len(slots) >= total_votes]
        if not eligible_slots:
            return None
        vote_positions = random.sample(random.choice(eligible_slots), total_votes)
        vote_colors = [GREEN] * green_votes + [RED] * red_votes
        random.shuffle(vote_colors)
        for (local_row, local_col), vote_color in zip(vote_positions, vote_colors):
            grid[r0 + local_row, c0 + local_col] = int(vote_color)

        specs.append(
            {
                "source_index": int(source_index),
                "area": int(area),
                "height": int(height),
                "width": int(width),
                "color": int(color),
                "survives": bool(survives),
            }
        )

    survivor_areas = [int(spec["area"]) for spec in specs if bool(spec["survives"])]
    if survivor_areas == sorted(survivor_areas, reverse=True):
        return None

    expected = _expected_output(grid.shape, rows, cols, specs)
    try:
        actual = main(grid)
    except Exception:
        return None
    if np.array_equal(actual, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
