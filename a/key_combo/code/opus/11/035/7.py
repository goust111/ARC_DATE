from __future__ import annotations

import random

import numpy as np


BG = 0
SOURCE_COLORS = (1, 2, 3, 4)
MASK_COLOR = 5
DIVIDER_COLOR = 7
GEN_RETRIES = 500
_GEN_CALL_INDEX = 0


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _non_divider_runs(start: int, end: int, dividers: list[int]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    for a, b in zip(dividers, dividers[1:]):
        if a + 1 <= b - 1:
            runs.append((a + 1, b - 1))
    assert runs
    assert runs[0][0] >= start and runs[-1][1] <= end
    return runs


def _find_sources(grid: np.ndarray, bg: int) -> list[tuple[int, int]]:
    top_cols = [int(c) for c in np.flatnonzero(grid[0] != bg)]
    assert 3 <= len(top_cols) <= 4, "expected 3-4 top-edge source pixels"
    assert all(b - a == 1 for a, b in zip(top_cols, top_cols[1:])), "sources must form one contiguous top run"
    colors = [int(grid[0, c]) for c in top_cols]
    assert len(set(colors)) == len(colors), "source colors must be distinct"
    return list(zip(top_cols, colors))


def _find_table_cells(grid: np.ndarray) -> list[dict]:
    divider = np.argwhere(grid == DIVIDER_COLOR)
    assert divider.size > 0, "missing divider table"
    r0, c0 = divider.min(axis=0)
    r1, c1 = divider.max(axis=0)
    r0 = int(r0)
    c0 = int(c0)
    r1 = int(r1)
    c1 = int(c1)

    row_divs = [r for r in range(r0, r1 + 1) if np.all(grid[r, c0 : c1 + 1] == DIVIDER_COLOR)]
    col_divs = [c for c in range(c0, c1 + 1) if np.all(grid[r0 : r1 + 1, c] == DIVIDER_COLOR)]
    assert len(row_divs) >= 4 and len(col_divs) >= 4, "divider must form a full lattice"

    row_runs = _non_divider_runs(r0, r1, row_divs)
    col_runs = _non_divider_runs(c0, c1, col_divs)
    cells: list[dict] = []
    for ri, (rr0, rr1) in enumerate(row_runs):
        for ci, (cc0, cc1) in enumerate(col_runs):
            cells.append(
                {
                    "id": len(cells),
                    "ri": ri,
                    "ci": ci,
                    "r0": rr0,
                    "r1": rr1,
                    "c0": cc0,
                    "c1": cc1,
                }
            )
    return cells


def _find_endpoints(grid: np.ndarray, sources: list[tuple[int, int]]) -> list[tuple[int, int, int, int]]:
    endpoints: list[tuple[int, int, int, int]] = []
    for src_col, color in sources:
        coords = [
            (int(r), int(c))
            for r, c in np.argwhere(grid == color)
            if not (int(r) == 0 and int(c) == src_col)
        ]
        assert len(coords) == 1, "each source color must have exactly one off-top endpoint"
        end_r, end_c = coords[0]
        assert end_r > 0 and end_c != src_col
        endpoints.append((src_col, color, end_r, end_c))
    return endpoints


def _path_coords(src_col: int, end_r: int, end_c: int) -> set[tuple[int, int]]:
    coords = {(r, src_col) for r in range(end_r + 1)}
    for c in range(min(src_col, end_c), max(src_col, end_c) + 1):
        coords.add((end_r, c))
    return coords


def _cell_crossings(paths: list[dict], cells: list[dict]) -> dict[int, list[int]]:
    crossed: dict[int, list[int]] = {cell["id"]: [] for cell in cells}
    for cell in cells:
        for path in paths:
            hit = any(
                cell["r0"] <= r <= cell["r1"] and cell["c0"] <= c <= cell["c1"]
                for r, c in path["coords"]
            )
            if hit:
                crossed[cell["id"]].append(int(path["index"]))
    return crossed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw each top-edge path to its same-colored endpoint, then recolor every masked table cell traversed by that path.

    Input structure:
    - The background is black, with a contiguous 3-4 pixel multicolor source run on the top edge.
    - An orange separator lattice in the lower region partitions the grid into rectangular cells.
    - Inside the lattice, each source color appears exactly once more as a singleton endpoint, and some cells contain gray mask pixels.

    Rule:
    - For each top-edge source pixel of color C, find the unique off-top endpoint of color C.
    - Trace an L-shaped path: straight down from the source column to the endpoint row, then straight horizontally to the endpoint column.
    - Paint only black pixels on that path with color C; do not overwrite orange dividers.
    - After all paths are determined, recolor every gray mask cell whose interior is traversed by a path to that path's color, breaking rare multi-path ties by the smaller source column.

    Key insight:
    - The same-color endpoint determines where each vertical stroke turns, and only those completed path trajectories reveal which masked table cells must be recolored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = _get_bg(g)
    sources = _find_sources(g, bg)
    cells = _find_table_cells(g)
    endpoints = _find_endpoints(g, sources)

    paths: list[dict] = []
    for idx, (src_col, color, end_r, end_c) in enumerate(endpoints):
        paths.append(
            {
                "index": idx,
                "src_col": src_col,
                "color": color,
                "end_r": end_r,
                "end_c": end_c,
                "coords": _path_coords(src_col, end_r, end_c),
            }
        )

    out = g.copy()
    for path in paths:
        color = int(path["color"])
        for r, c in path["coords"]:
            if out[r, c] == DIVIDER_COLOR:
                continue
            if g[r, c] in (bg, color):
                out[r, c] = color

    crossed = _cell_crossings(paths, cells)
    by_index = {int(path["index"]): path for path in paths}
    for cell in cells:
        region = out[cell["r0"] : cell["r1"] + 1, cell["c0"] : cell["c1"] + 1]
        if not np.any(region == MASK_COLOR):
            continue
        occupants = crossed[cell["id"]]
        if not occupants:
            continue
        winner = min((by_index[idx] for idx in occupants), key=lambda item: int(item["src_col"]))
        region[region == MASK_COLOR] = int(winner["color"])

    return out


def _table_layout(branch: int | None = None) -> tuple[np.ndarray, list[dict], int, int, int]:
    if branch is None:
        bands = random.choice([3, 4])
        cell_size = random.choice([3, 4])
    else:
        combos = ((3, 3), (4, 4), (4, 3), (3, 4))
        bands, cell_size = combos[branch % len(combos)]
    table_h = bands * cell_size + (bands + 1)
    table_w = bands * cell_size + (bands + 1)
    top = random.randint(3, 5)
    left = random.randint(1, 4)
    bottom = random.randint(1, 3)
    right = random.randint(1, 4)
    h = top + table_h + bottom
    w = left + table_w + right
    if h > 30 or w > 30:
        raise AssertionError("layout overflow")

    grid = np.zeros((h, w), dtype=int)
    row_divs = [top + i * (cell_size + 1) for i in range(bands + 1)]
    col_divs = [left + i * (cell_size + 1) for i in range(bands + 1)]
    for r in row_divs:
        grid[r, col_divs[0] : col_divs[-1] + 1] = DIVIDER_COLOR
    for c in col_divs:
        grid[row_divs[0] : row_divs[-1] + 1, c] = DIVIDER_COLOR
    return grid, _find_table_cells(grid), bands, cell_size, top


def _sample_source_columns(width: int, cells: list[dict], source_count: int) -> list[int]:
    table_c0 = min(cell["c0"] for cell in cells) - 1
    table_c1 = max(cell["c1"] for cell in cells) + 1
    lo = max(1, table_c0 + 1)
    hi = min(width - source_count - 1, table_c1 - source_count)
    assert lo <= hi
    center = (lo + hi) // 2
    start_lo = max(lo, center - 3)
    start_hi = min(hi, center + 3)
    start = random.randint(start_lo, start_hi)
    return list(range(start, start + source_count))


def _path_draw_overlap(coords: set[tuple[int, int]], divider_mask: np.ndarray) -> set[tuple[int, int]]:
    return {pt for pt in coords if not divider_mask[pt]}


def _endpoint_cells_ok(
    cell_ids: set[int],
    paths: list[dict],
    cells: list[dict],
) -> bool:
    crossed = _cell_crossings(paths, cells)
    endpoint_cell_by_path = {int(path["index"]): int(path["cell_id"]) for path in paths}
    untouched_exists = False
    for cell in cells:
        if cell["id"] in cell_ids:
            if len(crossed[cell["id"]]) > 1:
                return False
            if not crossed[cell["id"]]:
                untouched_exists = True
        elif not crossed[cell["id"]]:
            untouched_exists = True

    for path in paths:
        unique_cells = [
            cell_id
            for cell_id, owners in crossed.items()
            if owners == [int(path["index"])] and cell_id != endpoint_cell_by_path[int(path["index"])]
        ]
        if not unique_cells:
            return False
    return untouched_exists


def _choose_endpoints(grid: np.ndarray, cells: list[dict], sources: list[tuple[int, int]]) -> list[dict] | None:
    divider_mask = grid == DIVIDER_COLOR
    cell_points = {
        cell["id"]: [
            (r, c)
            for r in range(cell["r0"], cell["r1"] + 1)
            for c in range(cell["c0"], cell["c1"] + 1)
        ]
        for cell in cells
    }
    for _ in range(300):
        if len(sources) == 4:
            right_count = 2
        else:
            right_count = random.choice([1, 2])
        sides = ["L"] * len(sources)
        for i in random.sample(range(len(sources)), right_count):
            sides[i] = "R"
        if "L" not in sides or "R" not in sides:
            continue

        used_cells: set[int] = set()
        occupied: set[tuple[int, int]] = set()
        paths: list[dict] = []
        ok = True
        for idx, ((src_col, color), side) in enumerate(zip(sources, sides)):
            candidates = cells[:]
            random.shuffle(candidates)
            placed = False
            for cell in candidates:
                if cell["id"] in used_cells:
                    continue
                points = [
                    (r, c)
                    for r, c in cell_points[cell["id"]]
                    if (c < src_col if side == "L" else c > src_col) and abs(c - src_col) >= 2
                ]
                if not points:
                    continue
                random.shuffle(points)
                for end_r, end_c in points[: min(12, len(points))]:
                    coords = _path_coords(src_col, end_r, end_c)
                    draw_overlap = _path_draw_overlap(coords, divider_mask)
                    if draw_overlap & occupied:
                        continue
                    used_cells.add(cell["id"])
                    occupied.update(draw_overlap)
                    paths.append(
                        {
                            "index": idx,
                            "src_col": src_col,
                            "color": color,
                            "end_r": end_r,
                            "end_c": end_c,
                            "cell_id": cell["id"],
                            "coords": coords,
                        }
                    )
                    placed = True
                    break
                if placed:
                    break
            if not placed:
                ok = False
                break
        if not ok:
            continue
        if _endpoint_cells_ok(used_cells, paths, cells):
            return paths
    return None


def _mask_pattern(cell: dict, force_mode: str | None = None) -> list[tuple[int, int]]:
    coords = [
        (r, c)
        for r in range(cell["r0"], cell["r1"] + 1)
        for c in range(cell["c0"], cell["c1"] + 1)
    ]
    if force_mode == "full":
        return coords
    if force_mode == "partial":
        count = random.randint(3, max(3, len(coords) - 1))
        return random.sample(coords, count)
    if random.random() < 0.35:
        return coords
    count = random.randint(3, max(3, len(coords) - 1))
    return random.sample(coords, count)


def _place_masks(grid: np.ndarray, cells: list[dict], paths: list[dict]) -> bool:
    crossed = _cell_crossings(paths, cells)
    endpoint_cells = {int(path["cell_id"]) for path in paths}
    unique_by_path: dict[int, list[int]] = {}
    for path in paths:
        unique_by_path[int(path["index"])] = [
            cell_id
            for cell_id, owners in crossed.items()
            if owners == [int(path["index"])] and cell_id not in endpoint_cells
        ]
    if any(not ids for ids in unique_by_path.values()):
        return False

    untouched = [
        cell["id"]
        for cell in cells
        if not crossed[cell["id"]] and cell["id"] not in endpoint_cells
    ]
    if not untouched:
        return False

    selected: list[int] = []
    for idx in range(len(paths)):
        selected.append(random.choice(unique_by_path[idx]))
    selected.append(random.choice(untouched))

    extra_pool = [
        cell["id"]
        for cell in cells
        if cell["id"] not in selected
        and cell["id"] not in endpoint_cells
        and len(crossed[cell["id"]]) <= 1
    ]
    random.shuffle(extra_pool)
    extra_count = random.randint(0, min(2, len(extra_pool)))
    selected.extend(extra_pool[:extra_count])

    force_modes = {}
    if selected:
        force_modes[selected[0]] = "full"
    if len(selected) > 1:
        force_modes[selected[1]] = "partial"

    cells_by_id = {cell["id"]: cell for cell in cells}
    for cell_id in selected:
        for r, c in _mask_pattern(cells_by_id[cell_id], force_modes.get(cell_id)):
            grid[r, c] = MASK_COLOR
    return True


def _validate_generated_input(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert max(grid.shape) <= 30
    bg = _get_bg(grid)
    sources = _find_sources(grid, bg)
    cells = _find_table_cells(grid)
    paths = [
        {
            "index": idx,
            "src_col": src_col,
            "color": color,
            "end_r": end_r,
            "end_c": end_c,
            "cell_id": next(
                cell["id"]
                for cell in cells
                if cell["r0"] <= end_r <= cell["r1"] and cell["c0"] <= end_c <= cell["c1"]
            ),
            "coords": _path_coords(src_col, end_r, end_c),
        }
        for idx, (src_col, color, end_r, end_c) in enumerate(_find_endpoints(grid, sources))
    ]
    dirs = {int(np.sign(path["end_c"] - path["src_col"])) for path in paths}
    assert dirs == {-1, 1}, "generator must include both left and right turns"

    crossed = _cell_crossings(paths, cells)
    endpoint_cells = {int(path["cell_id"]) for path in paths}
    touched_mask = 0
    untouched_mask = 0
    for cell in cells:
        region = grid[cell["r0"] : cell["r1"] + 1, cell["c0"] : cell["c1"] + 1]
        if not np.any(region == MASK_COLOR):
            continue
        owners = crossed[cell["id"]]
        assert len(owners) <= 1, "masked cells must not be traversed by multiple paths"
        if owners:
            touched_mask += 1
        else:
            untouched_mask += 1
        assert cell["id"] not in endpoint_cells, "endpoint cells must stay unmasked"
    assert touched_mask >= len(paths), "each path must activate at least one mask cell"
    assert untouched_mask >= 1, "at least one mask cell must stay gray"

    for path in paths:
        unique_mask_hits = [
            cell["id"]
            for cell in cells
            if crossed[cell["id"]] == [int(path["index"])]
            and cell["id"] not in endpoint_cells
            and np.any(grid[cell["r0"] : cell["r1"] + 1, cell["c0"] : cell["c1"] + 1] == MASK_COLOR)
        ]
        assert unique_mask_hits, "each path needs a uniquely determined masked cell"

    out = main(grid)
    assert out.shape == grid.shape
    assert np.any(out != grid), "output must differ from input"


def _gen_attempt(branch: int | None = None) -> np.ndarray | None:
    try:
        grid, cells, bands, _cell_size, _top = _table_layout(branch)
    except AssertionError:
        return None

    if branch == 0:
        source_count = 3
    elif branch == 1:
        source_count = 4
    else:
        source_count = 4 if bands == 4 and random.random() < 0.6 else 3
    if grid.shape[1] < source_count + 2:
        return None
    try:
        src_cols = _sample_source_columns(grid.shape[1], cells, source_count)
    except AssertionError:
        return None
    colors = random.sample(list(SOURCE_COLORS), source_count)
    for c, color in zip(src_cols, colors):
        grid[0, c] = color

    sources = list(zip(src_cols, colors))
    paths = _choose_endpoints(grid, cells, sources)
    if paths is None:
        return None
    for path in paths:
        grid[int(path["end_r"]), int(path["end_c"])] = int(path["color"])

    if not _place_masks(grid, cells, paths):
        return None

    try:
        _validate_generated_input(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_INDEX
    branch = _GEN_CALL_INDEX
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(branch)
        if grid is not None:
            _GEN_CALL_INDEX += 1
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
