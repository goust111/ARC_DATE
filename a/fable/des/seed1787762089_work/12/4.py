import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
AZURE = 8
MASTER_COLORS = (BLUE, RED, GREEN, YELLOW, MAGENTA, AZURE)

GRID_RETRIES = 400
SLOT_DIM_OPTIONS = [(r, c) for r in range(2, 5) for c in range(2, 5) if r * c >= 6]


def _line_spans(lines: np.ndarray, limit: int) -> tuple[list[tuple[int, int]], int]:
    lines = np.asarray(lines, dtype=int)
    assert len(lines) >= 3
    assert int(lines[0]) == 0 and int(lines[-1]) == limit - 1
    assert np.all(np.diff(lines) > 1)

    spans: list[tuple[int, int]] = []
    sizes: list[int] = []
    for start, stop in zip(lines[:-1], lines[1:]):
        inner_start = int(start) + 1
        inner_stop = int(stop)
        spans.append((inner_start, inner_stop))
        sizes.append(inner_stop - inner_start)

    assert min(sizes) > 0
    assert len(set(sizes)) == 1
    return spans, sizes[0]


def _slot_structure(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int]:
    grey_rows = np.flatnonzero(np.all(grid == GREY, axis=1))
    grey_cols = np.flatnonzero(np.all(grid == GREY, axis=0))
    row_spans, row_size = _line_spans(grey_rows, grid.shape[0])
    col_spans, col_size = _line_spans(grey_cols, grid.shape[1])
    assert row_size == col_size
    return row_spans, col_spans, row_size


def _slot_slices(
    row_spans: list[tuple[int, int]], col_spans: list[tuple[int, int]]
) -> list[tuple[int, int, slice, slice]]:
    slots: list[tuple[int, int, slice, slice]] = []
    for sr, (r0, r1) in enumerate(row_spans):
        for sc, (c0, c1) in enumerate(col_spans):
            slots.append((sr, sc, slice(r0, r1), slice(c0, c1)))
    return slots


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each selector pixel with the master pattern filtered to that selector color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    row_spans, col_spans, _ = _slot_structure(g)
    slots = _slot_slices(row_spans, col_spans)

    slot_data: list[dict[str, object]] = []
    for sr, sc, rs, cs in slots:
        patch = g[rs, cs]
        assert patch.shape[0] == patch.shape[1]
        assert not np.any(patch == GREY)
        non_black = patch[patch != BLACK]
        slot_data.append(
            {
                "row": sr,
                "col": sc,
                "rs": rs,
                "cs": cs,
                "patch": patch,
                "count": int(non_black.size),
                "colors": tuple(int(v) for v in np.unique(non_black)),
            }
        )

    master_slots = [entry for entry in slot_data if int(entry["count"]) > 1]
    assert len(master_slots) == 1
    master = np.asarray(master_slots[0]["patch"], dtype=int)
    master_colors = tuple(int(v) for v in np.unique(master) if int(v) != BLACK)
    assert 2 <= len(master_colors) <= 4

    out = g.copy()
    for entry in slot_data:
        count = int(entry["count"])
        if count == 0 or entry is master_slots[0]:
            continue
        assert count == 1
        colors = entry["colors"]
        assert len(colors) == 1
        color = int(colors[0])
        assert color in master_colors
        out[entry["rs"], entry["cs"]] = np.where(master == color, color, BLACK)

    return out


def _make_master(rng: random.Random, size: int, colors: list[int]) -> np.ndarray:
    total = size * size
    min_fill = max(len(colors) + 1, total // 2)
    if rng.random() < 0.5:
        fill_count = total
    else:
        fill_count = rng.randint(min_fill, total - 1)

    cells = [(r, c) for r in range(size) for c in range(size)]
    rng.shuffle(cells)
    chosen = cells[:fill_count]

    assignments = colors.copy()
    assignments.extend(rng.choice(colors) for _ in range(fill_count - len(colors)))
    rng.shuffle(assignments)

    master = np.zeros((size, size), dtype=int)
    for (r, c), color in zip(chosen, assignments):
        master[r, c] = color

    assert np.count_nonzero(master) == fill_count
    assert len([v for v in np.unique(master) if v != BLACK]) == len(colors)
    return master


def _selector_position(
    rng: random.Random, size: int, mask_cells: list[tuple[int, int]]
) -> tuple[int, int]:
    all_cells = [(r, c) for r in range(size) for c in range(size)]
    if len(mask_cells) == 1:
        all_cells = [cell for cell in all_cells if cell != mask_cells[0]]
    return rng.choice(all_cells)


def _build_grid(
    rows: int,
    cols: int,
    size: int,
    master_slot: tuple[int, int],
    master: np.ndarray,
    selectors: list[tuple[tuple[int, int], int, tuple[int, int]]],
) -> np.ndarray:
    height = rows * size + rows + 1
    width = cols * size + cols + 1
    grid = np.full((height, width), BLACK, dtype=int)

    for r in range(0, height, size + 1):
        grid[r, :] = GREY
    for c in range(0, width, size + 1):
        grid[:, c] = GREY

    mr, mc = master_slot
    r0 = 1 + mr * (size + 1)
    c0 = 1 + mc * (size + 1)
    grid[r0 : r0 + size, c0 : c0 + size] = master

    for (sr, sc), color, (pr, pc) in selectors:
        rr = 1 + sr * (size + 1)
        cc = 1 + sc * (size + 1)
        grid[rr + pr, cc + pc] = color

    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    rows, cols = rng.choice(SLOT_DIM_OPTIONS)
    size = rng.choice([3, 4])
    slot_count = rows * cols

    max_colors = min(4, slot_count - 2, len(MASTER_COLORS))
    color_count = rng.randint(2, max_colors)
    colors = rng.sample(list(MASTER_COLORS), color_count)
    master = _make_master(rng, size, colors)

    all_slots = [(r, c) for r in range(rows) for c in range(cols)]
    master_slot = rng.choice(all_slots)
    free_slots = [slot for slot in all_slots if slot != master_slot]

    selector_count = rng.randint(color_count, slot_count - 2)
    selector_colors = colors.copy()
    selector_colors.extend(rng.choice(colors) for _ in range(selector_count - color_count))
    rng.shuffle(selector_colors)

    selector_slots = rng.sample(free_slots, selector_count)
    selectors: list[tuple[tuple[int, int], int, tuple[int, int]]] = []
    for slot, color in zip(selector_slots, selector_colors):
        mask_cells = [(int(r), int(c)) for r, c in np.argwhere(master == color)]
        pos = _selector_position(rng, size, mask_cells)
        selectors.append((slot, color, pos))

    grid = _build_grid(rows, cols, size, master_slot, master, selectors)
    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(grid, out):
        return None

    for slot, _color, _pos in selectors:
        sr, sc = slot
        rr = 1 + sr * (size + 1)
        cc = 1 + sc * (size + 1)
        if np.array_equal(grid[rr : rr + size, cc : cc + size], out[rr : rr + size, cc : cc + size]):
            return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GRID_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
