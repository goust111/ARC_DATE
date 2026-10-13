from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

DOMINO_COLORS = (BLUE, GREEN, YELLOW, ORANGE)
COUNT_TO_DIRECTION = {1: "down", 2: "right", 3: "left"}
DIRECTION_STEP = {"down": (1, 0), "right": (0, 1), "left": (0, -1)}
CELL_SIZE_RANGE = (6, 8)
GRID_RETRIES = 400
CELL_RETRIES = 200


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _partition_spans(length: int, separators: np.ndarray) -> list[tuple[int, int]]:
    idx = [int(v) for v in separators.tolist()]
    assert len(idx) == 2, "expected exactly two full separator lines"
    assert idx[0] + 1 < idx[1], "separator lines must leave three non-empty regions"
    spans = [(0, idx[0] - 1), (idx[0] + 1, idx[1] - 1), (idx[1] + 1, length - 1)]
    assert all(a <= b for a, b in spans), "each macro-cell band must be non-empty"
    return spans


def _macro_spans(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    row_seps = np.flatnonzero(np.all(grid == GREY, axis=1))
    col_seps = np.flatnonzero(np.all(grid == GREY, axis=0))
    return _partition_spans(grid.shape[0], row_seps), _partition_spans(grid.shape[1], col_seps)


def _domino_coords(anchor_r: int, anchor_c: int, orient: str) -> np.ndarray:
    if orient == "h":
        return np.array([[anchor_r, anchor_c], [anchor_r, anchor_c + 1]], dtype=int)
    assert orient == "v"
    return np.array([[anchor_r, anchor_c], [anchor_r + 1, anchor_c]], dtype=int)


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _find_components(cell: np.ndarray, colors: set[int]) -> list[dict]:
    h, w = cell.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(cell[r, c])
            if color not in colors or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(cell[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": _bbox(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _domino_components(clean_cell: np.ndarray) -> list[dict]:
    comps = _find_components(clean_cell, set(DOMINO_COLORS))
    assert 1 <= len(comps) <= 2, "each macro-cell must contain one or two dominoes"
    for comp in comps:
        assert comp["area"] == 2, "each colored object must be a domino"
    return comps


def _settle_key(comp: dict, direction: str) -> tuple[int, int, int, int]:
    if "bbox" in comp:
        r0, c0, r1, c1 = comp["bbox"]
    else:
        r0, c0, r1, c1 = _bbox(comp["coords"])
    if direction == "down":
        return (-r1, c0, c1, r0)
    if direction == "right":
        return (-c1, r0, r1, c0)
    assert direction == "left"
    return (c0, r0, r1, c1)


def _settle_dominoes(
    shape: tuple[int, int],
    dominoes: list[dict],
    direction: str,
    *,
    blocked: np.ndarray | None = None,
    order: str = "nearest",
    return_tracks: bool = False,
) -> tuple[np.ndarray, list[dict]] | np.ndarray:
    h, w = shape
    dr, dc = DIRECTION_STEP[direction]
    occupied = np.zeros((h, w), dtype=bool) if blocked is None else blocked.copy()
    out = np.zeros((h, w), dtype=int)

    ordered = sorted(dominoes, key=lambda comp: _settle_key(comp, direction), reverse=(order == "farthest"))
    tracks: list[dict] = []
    for comp in ordered:
        coords = comp["coords"].astype(int).copy()
        snapshots: list[np.ndarray] = []
        while True:
            nxt = coords + np.array([dr, dc], dtype=int)
            if np.any(nxt[:, 0] < 0) or np.any(nxt[:, 0] >= h) or np.any(nxt[:, 1] < 0) or np.any(nxt[:, 1] >= w):
                break
            if np.any(occupied[nxt[:, 0], nxt[:, 1]]):
                break
            coords = nxt
            snapshots.append(coords.copy())
        out[coords[:, 0], coords[:, 1]] = int(comp["color"])
        occupied[coords[:, 0], coords[:, 1]] = True
        if return_tracks:
            tracks.append(
                {
                    "color": int(comp["color"]),
                    "initial": comp["coords"].astype(int).copy(),
                    "final": coords.copy(),
                    "steps": len(snapshots),
                    "positions": snapshots,
                }
            )

    if return_tracks:
        return out, tracks
    return out


def _solve_cell_with_direction(
    cell: np.ndarray,
    direction: str,
    *,
    erase_markers: bool = True,
    order: str = "nearest",
    return_tracks: bool = False,
) -> tuple[np.ndarray, list[dict]] | np.ndarray:
    local = np.asarray(cell, dtype=int)
    blocked = np.zeros(local.shape, dtype=bool) if erase_markers else (local == RED)
    clean = local.copy()
    clean[clean == RED] = BG
    dominoes = _domino_components(clean)
    return _settle_dominoes(clean.shape, dominoes, direction, blocked=blocked, order=order, return_tracks=return_tracks)


def _solve_cell(cell: np.ndarray, **kwargs) -> tuple[np.ndarray, list[dict]] | np.ndarray:
    count = int(np.count_nonzero(cell == RED))
    assert count in COUNT_TO_DIRECTION, "red markers must count to 1, 2, or 3"
    return _solve_cell_with_direction(cell, COUNT_TO_DIRECTION[count], **kwargs)


def main(grid: np.ndarray) -> np.ndarray:
    """Use each macro-cell's red marker count to choose gravity, erase the markers, and pack its dominoes against that side.

    Input structure:
    - A black grid is split into a 3x3 array of macro-cells by two full grey rows and two full grey columns.
    - Each macro-cell contains 1-3 isolated red marker pixels and 1-2 colored dominoes.
    - Dominoes are monochrome 2-cell 4-connected objects, while the grey separator lines remain fixed.

    Rule:
    - Count the red markers inside each macro-cell.
    - Interpret counts as gravity directions: 1 -> down, 2 -> right, 3 -> left.
    - Erase the red markers, then settle that macro-cell's dominoes rigidly in the encoded direction.
    - Resolve dominoes nearest the target wall first; each slides until the next step would hit the macro-cell boundary or a previously settled domino.

    Key insight:
    - The red pixels are not output objects; they are disposable direction codes whose count determines how the dominoes must be packed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    row_spans, col_spans = _macro_spans(g)
    out = np.zeros_like(g)
    out[g == GREY] = GREY

    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            cell = g[r0 : r1 + 1, c0 : c1 + 1]
            assert not np.any(cell == GREY), "separator color must stay on separator lines only"
            out[r0 : r1 + 1, c0 : c1 + 1] = _solve_cell(cell)

    return out


def _paint_dominoes(shape: tuple[int, int], dominoes: list[dict]) -> np.ndarray:
    canvas = np.zeros(shape, dtype=int)
    for dom in dominoes:
        canvas[dom["coords"][:, 0], dom["coords"][:, 1]] = int(dom["color"])
    return canvas


def _initial_forbidden(dominoes: list[dict], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    forbidden: set[tuple[int, int]] = set()
    for dom in dominoes:
        for r, c in dom["coords"]:
            rr, cc = int(r), int(c)
            forbidden.add((rr, cc))
            for nr, nc in _neighbors4(rr, cc):
                if 0 <= nr < h and 0 <= nc < w:
                    forbidden.add((nr, nc))
    return forbidden


def _blocking_marker_candidates(
    dominoes: list[dict], direction: str, shape: tuple[int, int]
) -> list[tuple[int, int]]:
    _, tracks = _settle_dominoes(shape, dominoes, direction, return_tracks=True)
    forbidden = _initial_forbidden(dominoes, shape)
    candidates: set[tuple[int, int]] = set()
    for track in tracks:
        for step_index, coords in enumerate(track["positions"], start=1):
            if step_index < 2:
                continue
            for r, c in coords:
                cell = (int(r), int(c))
                if cell not in forbidden:
                    candidates.add(cell)
    return sorted(candidates)


def _random_free_markers(
    shape: tuple[int, int],
    dominoes: list[dict],
    count: int,
    blocking_cell: tuple[int, int],
) -> list[tuple[int, int]] | None:
    h, w = shape
    forbidden = _initial_forbidden(dominoes, shape)
    available = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if (r, c) not in forbidden and (r, c) != blocking_cell
    ]
    if len(available) < count - 1:
        return None
    markers = [blocking_cell]
    if count > 1:
        markers.extend(random.sample(available, count - 1))
    return markers


def _make_single_dominoes(h: int, w: int, direction: str) -> list[dict] | None:
    options: list[np.ndarray] = []
    if direction == "down":
        for r in range(0, h - 2):
            for c in range(0, w - 1):
                options.append(_domino_coords(r, c, "h"))
        for r in range(0, h - 3):
            for c in range(0, w):
                options.append(_domino_coords(r, c, "v"))
    elif direction == "right":
        for r in range(0, h):
            for c in range(0, w - 3):
                options.append(_domino_coords(r, c, "h"))
        for r in range(0, h - 1):
            for c in range(0, w - 2):
                options.append(_domino_coords(r, c, "v"))
    else:
        for r in range(0, h):
            for c in range(2, w - 1):
                options.append(_domino_coords(r, c, "h"))
        for r in range(0, h - 1):
            for c in range(2, w):
                options.append(_domino_coords(r, c, "v"))
    if not options:
        return None
    coords = random.choice(options)
    return [{"color": random.choice(DOMINO_COLORS), "coords": coords}]


def _make_order_sensitive_dominoes(h: int, w: int, direction: str) -> list[dict] | None:
    colors = random.sample(DOMINO_COLORS, 2)
    options: list[tuple[np.ndarray, np.ndarray]] = []
    if direction == "down":
        for c in range(0, w - 1):
            for near_r in range(2, h - 2):
                for far_r in range(0, near_r - 1):
                    near = _domino_coords(near_r, c, "h")
                    far = _domino_coords(far_r, c, "h")
                    options.append((near, far))
    elif direction == "right":
        for r in range(0, h - 1):
            for near_c in range(2, w - 2):
                for far_c in range(0, near_c - 1):
                    near = _domino_coords(r, near_c, "v")
                    far = _domino_coords(r, far_c, "v")
                    options.append((near, far))
    else:
        for r in range(0, h - 1):
            for near_c in range(2, w - 2):
                for far_c in range(near_c + 2, w):
                    near = _domino_coords(r, near_c, "v")
                    far = _domino_coords(r, far_c, "v")
                    options.append((near, far))
    if not options:
        return None
    near_coords, far_coords = random.choice(options)
    return [
        {"color": colors[0], "coords": near_coords},
        {"color": colors[1], "coords": far_coords},
    ]


def _build_cell(
    h: int,
    w: int,
    marker_count: int,
    *,
    force_two_dominoes: bool = False,
) -> np.ndarray | None:
    direction = COUNT_TO_DIRECTION[marker_count]
    for _ in range(CELL_RETRIES):
        use_two = force_two_dominoes or random.random() < 0.45
        dominoes = _make_order_sensitive_dominoes(h, w, direction) if use_two else _make_single_dominoes(h, w, direction)
        if dominoes is None:
            continue
        candidates = _blocking_marker_candidates(dominoes, direction, (h, w))
        if not candidates:
            continue
        blocking_cell = random.choice(candidates)
        markers = _random_free_markers((h, w), dominoes, marker_count, blocking_cell)
        if markers is None:
            continue

        cell = _paint_dominoes((h, w), dominoes)
        for r, c in markers:
            cell[r, c] = RED

        require_order = len(dominoes) == 2
        if _cell_semantics_ok(cell, require_order_sensitive=require_order):
            return cell
    return None


def _cell_semantics_ok(cell: np.ndarray, *, require_order_sensitive: bool) -> bool:
    try:
        clean = cell.copy()
        count = int(np.count_nonzero(clean == RED))
        if count not in COUNT_TO_DIRECTION:
            return False
        if np.any(clean == GREY):
            return False
        direction = COUNT_TO_DIRECTION[count]
        comps = _domino_components(np.where(clean == RED, BG, clean))
        if len({int(comp["color"]) for comp in comps}) != len(comps):
            return False

        domino_cells = {(int(r), int(c)) for comp in comps for r, c in comp["coords"]}
        for r, c in np.argwhere(clean == RED):
            rr, cc = int(r), int(c)
            for nr, nc in _neighbors4(rr, cc):
                if (nr, nc) in domino_cells:
                    return False

        solved, tracks = _solve_cell(clean, return_tracks=True)
        if np.array_equal(solved, np.where(clean == RED, BG, clean)):
            return False
        if not tracks or max(track["steps"] for track in tracks) < 2:
            return False

        blocked_result = _solve_cell_with_direction(clean, direction, erase_markers=False)
        if np.array_equal(blocked_result, solved):
            return False

        for other in ("down", "right", "left"):
            if other == direction:
                continue
            alt = _solve_cell_with_direction(clean, other)
            if np.array_equal(alt, solved):
                return False

        if require_order_sensitive:
            far = _solve_cell_with_direction(clean, direction, order="farthest")
            if np.array_equal(far, solved):
                return False
    except AssertionError:
        return False
    return True


def _assemble_grid(cells: list[list[np.ndarray]], row_heights: list[int], col_widths: list[int]) -> np.ndarray:
    total_h = sum(row_heights) + 2
    total_w = sum(col_widths) + 2
    grid = np.zeros((total_h, total_w), dtype=int)

    row_sep0 = row_heights[0]
    row_sep1 = row_heights[0] + 1 + row_heights[1]
    col_sep0 = col_widths[0]
    col_sep1 = col_widths[0] + 1 + col_widths[1]
    grid[row_sep0, :] = GREY
    grid[row_sep1, :] = GREY
    grid[:, col_sep0] = GREY
    grid[:, col_sep1] = GREY

    r_cursor = 0
    for i, height in enumerate(row_heights):
        c_cursor = 0
        for j, width in enumerate(col_widths):
            grid[r_cursor : r_cursor + height, c_cursor : c_cursor + width] = cells[i][j]
            c_cursor += width + (1 if j < 2 else 0)
        r_cursor += height + (1 if i < 2 else 0)
    return grid


def _global_semantics_ok(grid: np.ndarray, special_counts: dict[int, tuple[int, int]]) -> bool:
    try:
        row_spans, col_spans = _macro_spans(grid)
        seen_counts: list[int] = []
        one_domino_cells = 0
        two_domino_cells = 0
        for i, (r0, r1) in enumerate(row_spans):
            for j, (c0, c1) in enumerate(col_spans):
                cell = grid[r0 : r1 + 1, c0 : c1 + 1]
                require_order = special_counts.get(int(np.count_nonzero(cell == RED))) == (i, j)
                if not _cell_semantics_ok(cell, require_order_sensitive=require_order):
                    return False
                comp_count = len(_domino_components(np.where(cell == RED, BG, cell)))
                one_domino_cells += int(comp_count == 1)
                two_domino_cells += int(comp_count == 2)
                seen_counts.append(int(np.count_nonzero(cell == RED)))

        if sorted(seen_counts) != [1, 1, 1, 2, 2, 2, 3, 3, 3]:
            return False
        if one_domino_cells < 3 or two_domino_cells < 3:
            return False

        solved = main(grid)
        if np.array_equal(solved, grid):
            return False

        swapped = {
            1: "right",
            2: "down",
            3: "left",
        }
        wrong = np.zeros_like(grid)
        wrong[grid == GREY] = GREY
        for r0, r1 in row_spans:
            for c0, c1 in col_spans:
                cell = grid[r0 : r1 + 1, c0 : c1 + 1]
                wrong_dir = swapped[int(np.count_nonzero(cell == RED))]
                wrong[r0 : r1 + 1, c0 : c1 + 1] = _solve_cell_with_direction(cell, wrong_dir)
        if np.array_equal(wrong, solved):
            return False
    except AssertionError:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GRID_RETRIES):
        row_heights = [random.randint(*CELL_SIZE_RANGE) for _ in range(3)]
        col_widths = [random.randint(*CELL_SIZE_RANGE) for _ in range(3)]

        counts = [1, 1, 1, 2, 2, 2, 3, 3, 3]
        random.shuffle(counts)
        count_positions: dict[int, list[tuple[int, int]]] = {1: [], 2: [], 3: []}
        for idx, count in enumerate(counts):
            count_positions[count].append((idx // 3, idx % 3))
        special_counts = {count: random.choice(pos_list) for count, pos_list in count_positions.items()}

        cells: list[list[np.ndarray]] = [[np.zeros((row_heights[i], col_widths[j]), dtype=int) for j in range(3)] for i in range(3)]
        failed = False
        for i in range(3):
            for j in range(3):
                count = counts[i * 3 + j]
                force_two = special_counts[count] == (i, j)
                cell = _build_cell(row_heights[i], col_widths[j], count, force_two_dominoes=force_two)
                if cell is None:
                    failed = True
                    break
                cells[i][j] = cell
            if failed:
                break
        if failed:
            continue

        grid = _assemble_grid(cells, row_heights, col_widths)
        if _global_semantics_ok(grid, special_counts):
            return grid
    raise RuntimeError("failed to generate a valid input grid")
