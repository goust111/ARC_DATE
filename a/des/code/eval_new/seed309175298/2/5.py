import random
import numpy as np


BLACK = 0
GREY = 5
AZURE = 8
PALETTE = (1, 2, 3, 4, 6, 7, 9)
INNER_SIZES = ((5, 5), (5, 6), (6, 5), (6, 6))
BRANCHES = ("row", "col", "both")
MAX_GEN_ATTEMPTS = 300


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _find_windows(grid: np.ndarray) -> list[dict]:
    border_mask = (grid == GREY) | (grid == AZURE)
    windows: list[dict] = []
    for coords in _components(border_mask):
        r0, c0, r1, c1 = _bbox(coords)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue

        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        frame = np.zeros(sub.shape, dtype=bool)
        frame[0, :] = True
        frame[-1, :] = True
        frame[:, 0] = True
        frame[:, -1] = True
        if not np.array_equal(((sub == GREY) | (sub == AZURE)), frame):
            continue

        top_azure = np.flatnonzero(sub[0, 1:-1] == AZURE)
        left_azure = np.flatnonzero(sub[1:-1, 0] == AZURE)
        windows.append(
            {
                "bbox": (r0, c0, r1, c1),
                "shape": (int(sub.shape[0] - 2), int(sub.shape[1] - 2)),
                "row_offset": int(left_azure.min()) if left_azure.size else 0,
                "col_offset": int(top_azure.min()) if top_azure.size else 0,
            }
        )

    windows.sort(key=lambda item: item["bbox"])
    assert windows, "expected at least one grey/azure window"
    inner_shape = windows[0]["shape"]
    assert all(w["shape"] == inner_shape for w in windows), "all windows must share one interior size"
    return windows


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each framed window by reconstructing the shared hidden master pattern.

    Input structure:
    - The background is black.
    - Several same-sized rectangular windows appear on the grid, each surrounded by a grey border.
    - Each window border contains azure indicator pixels on the top edge and left edge, while the interior holds a partial colored crop on black.

    Rule:
    - Read the column offset from the first azure pixel on the top border and the row offset from the first azure pixel on the left border.
    - Treat every window interior as a crop of one larger master pattern at that row/column offset, and merge all visible non-black pixels into that shared master coordinate system.
    - Replace every window interior by the full crop from the reconstructed master while leaving the grey border and azure indicators unchanged.

    Key insight:
    - The different windows are not separate patterns: they are overlapping offset views of one hidden template, and the azure border marks tell how to align them.
    """
    g = np.asarray(grid, dtype=int)
    windows = _find_windows(g)
    inner_h, inner_w = windows[0]["shape"]

    master_h = max(w["row_offset"] + inner_h for w in windows)
    master_w = max(w["col_offset"] + inner_w for w in windows)
    known = np.full((master_h, master_w), -1, dtype=int)

    for window in windows:
        r0, c0, r1, c1 = window["bbox"]
        row_offset = window["row_offset"]
        col_offset = window["col_offset"]
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        for rr, cc in np.argwhere(interior != BLACK):
            mr = row_offset + int(rr)
            mc = col_offset + int(cc)
            color = int(interior[rr, cc])
            if known[mr, mc] != -1:
                assert known[mr, mc] == color, "overlapping windows must agree"
            known[mr, mc] = color

    master = np.where(known == -1, BLACK, known)
    out = g.copy()
    for window in windows:
        r0, c0, r1, c1 = window["bbox"]
        row_offset = window["row_offset"]
        col_offset = window["col_offset"]
        out[r0 + 1 : r1, c0 + 1 : c1] = master[
            row_offset : row_offset + inner_h, col_offset : col_offset + inner_w
        ]
    return out


def _coverage_map(offsets: list[tuple[int, int]], inner_h: int, inner_w: int, master_h: int, master_w: int) -> dict:
    coverage: dict[tuple[int, int], list[int]] = {}
    for r in range(master_h):
        for c in range(master_w):
            covers = [
                idx
                for idx, (row_offset, col_offset) in enumerate(offsets)
                if row_offset <= r < row_offset + inner_h and col_offset <= c < col_offset + inner_w
            ]
            assert covers, "window offsets must cover the whole master canvas"
            coverage[(r, c)] = covers
    return coverage


def _assign_hidden_positions(
    coverage: dict[tuple[int, int], list[int]],
    offsets: list[tuple[int, int]],
    inner_h: int,
    inner_w: int,
    rng: random.Random,
) -> dict[int, tuple[int, int]] | None:
    candidates: dict[int, list[tuple[int, int]]] = {}
    for idx, (row_offset, col_offset) in enumerate(offsets):
        pts = []
        for r in range(row_offset, row_offset + inner_h):
            for c in range(col_offset, col_offset + inner_w):
                if len(coverage[(r, c)]) > 1:
                    pts.append((r, c))
        rng.shuffle(pts)
        candidates[idx] = pts

    order = sorted(candidates, key=lambda idx: len(candidates[idx]))
    assignment: dict[int, tuple[int, int]] = {}
    used: set[tuple[int, int]] = set()

    def backtrack(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        for cell in candidates[idx]:
            if cell in used:
                continue
            used.add(cell)
            assignment[idx] = cell
            if backtrack(pos + 1):
                return True
            used.remove(cell)
            del assignment[idx]
        return False

    return assignment if backtrack(0) else None


def _pick_cell(
    used: set[tuple[int, int]],
    r0: int,
    r1: int,
    c0: int,
    c1: int,
    rng: random.Random,
) -> tuple[int, int]:
    cells = [(r, c) for r in range(r0, r1) for c in range(c0, c1)]
    unused = [cell for cell in cells if cell not in used]
    pool = unused or cells
    cell = rng.choice(pool)
    used.add(cell)
    return cell


def _paint_random_shape(master: np.ndarray, colors: list[int], rng: random.Random) -> None:
    h, w = master.shape
    color = rng.choice(colors)
    kind = rng.choice(("rect", "hline", "vline", "ell"))

    if kind == "rect":
        rh = rng.randint(1, min(3, h))
        rw = rng.randint(1, min(3, w))
        r0 = rng.randint(0, h - rh)
        c0 = rng.randint(0, w - rw)
        master[r0 : r0 + rh, c0 : c0 + rw] = color
    elif kind == "hline":
        rw = rng.randint(2, min(4, w))
        r = rng.randint(0, h - 1)
        c0 = rng.randint(0, w - rw)
        master[r, c0 : c0 + rw] = color
    elif kind == "vline":
        rh = rng.randint(2, min(4, h))
        r0 = rng.randint(0, h - rh)
        c = rng.randint(0, w - 1)
        master[r0 : r0 + rh, c] = color
    else:
        rh = rng.randint(2, min(3, h))
        rw = rng.randint(2, min(3, w))
        r0 = rng.randint(0, h - rh)
        c0 = rng.randint(0, w - rw)
        master[r0 : r0 + rh, c0] = color
        master[r0 + rh - 1, c0 : c0 + rw] = color


def _build_master(
    branch: str,
    offsets: list[tuple[int, int]],
    coverage: dict[tuple[int, int], list[int]],
    hidden_positions: dict[int, tuple[int, int]],
    inner_h: int,
    inner_w: int,
    master_h: int,
    master_w: int,
    rng: random.Random,
) -> tuple[np.ndarray, set[tuple[int, int]]]:
    active_colors = rng.sample(PALETTE, rng.randint(2, 4))
    extra_h = master_h - inner_h
    extra_w = master_w - inner_w
    master = np.full((master_h, master_w), BLACK, dtype=int)

    mandatory: list[tuple[int, int]] = list(hidden_positions.values())
    used = set(mandatory)
    shared_count = 2

    if branch == "row":
        mandatory.append(_pick_cell(used, 0, extra_h, 0, master_w, rng))
        mandatory.append(_pick_cell(used, inner_h, master_h, 0, master_w, rng))
        for _ in range(shared_count):
            mandatory.append(_pick_cell(used, extra_h, inner_h, 0, master_w, rng))
    elif branch == "col":
        mandatory.append(_pick_cell(used, 0, master_h, 0, extra_w, rng))
        mandatory.append(_pick_cell(used, 0, master_h, inner_w, master_w, rng))
        for _ in range(shared_count):
            mandatory.append(_pick_cell(used, 0, master_h, extra_w, inner_w, rng))
    else:
        mandatory.append(_pick_cell(used, 0, extra_h, 0, extra_w, rng))
        mandatory.append(_pick_cell(used, 0, extra_h, inner_w, master_w, rng))
        mandatory.append(_pick_cell(used, inner_h, master_h, 0, extra_w, rng))
        mandatory.append(_pick_cell(used, inner_h, master_h, inner_w, master_w, rng))
        for _ in range(shared_count):
            mandatory.append(_pick_cell(used, extra_h, inner_h, extra_w, inner_w, rng))

    for idx, cell in enumerate(mandatory):
        master[cell] = active_colors[idx % len(active_colors)]

    for _ in range(rng.randint(2, 5)):
        _paint_random_shape(master, active_colors, rng)

    colored = int(np.count_nonzero(master))
    if colored < len(offsets) + 4 or colored > int(0.6 * master.size):
        raise ValueError("master density out of bounds")
    if len(set(map(int, master[master != BLACK].flat))) < 2:
        raise ValueError("master needs at least two colors")

    always_show = set(mandatory) - set(hidden_positions.values())
    for cell in always_show:
        assert len(coverage[cell]) >= 1
    return master, always_show


def _choose_offsets(rng: random.Random, branch: str, inner_h: int, inner_w: int) -> tuple[list[tuple[int, int]], int, int]:
    if branch == "row":
        count = rng.randint(3, 5)
        extra_h = rng.randint(count - 1, inner_h - 1)
        rows = list(range(extra_h + 1))
        rng.shuffle(rows)
        chosen_rows = sorted(rows[:count])
        if 0 not in chosen_rows:
            chosen_rows[0] = 0
        if extra_h not in chosen_rows:
            chosen_rows[-1] = extra_h
        chosen_rows = sorted(set(chosen_rows))
        while len(chosen_rows) < count:
            for row in range(extra_h + 1):
                if row not in chosen_rows:
                    chosen_rows.append(row)
                    break
        offsets = [(row, 0) for row in sorted(chosen_rows)]
        return offsets, inner_h + extra_h, inner_w

    if branch == "col":
        count = rng.randint(3, 5)
        extra_w = rng.randint(count - 1, inner_w - 1)
        cols = list(range(extra_w + 1))
        rng.shuffle(cols)
        chosen_cols = sorted(cols[:count])
        if 0 not in chosen_cols:
            chosen_cols[0] = 0
        if extra_w not in chosen_cols:
            chosen_cols[-1] = extra_w
        chosen_cols = sorted(set(chosen_cols))
        while len(chosen_cols) < count:
            for col in range(extra_w + 1):
                if col not in chosen_cols:
                    chosen_cols.append(col)
                    break
        offsets = [(0, col) for col in sorted(chosen_cols)]
        return offsets, inner_h, inner_w + extra_w

    extra_h = rng.randint(1, inner_h - 1)
    extra_w = rng.randint(1, inner_w - 1)
    offsets = [(0, 0), (extra_h, 0), (0, extra_w), (extra_h, extra_w)]
    pool = [(r, c) for r in range(extra_h + 1) for c in range(extra_w + 1) if (r, c) not in offsets]
    if pool and rng.random() < 0.5:
        offsets.append(rng.choice(pool))
    return offsets, inner_h + extra_h, inner_w + extra_w


def _sample_window_crops(
    master: np.ndarray,
    offsets: list[tuple[int, int]],
    coverage: dict[tuple[int, int], list[int]],
    hidden_positions: dict[int, tuple[int, int]],
    always_show: set[tuple[int, int]],
    inner_h: int,
    inner_w: int,
    rng: random.Random,
) -> list[np.ndarray]:
    crops = [np.full((inner_h, inner_w), BLACK, dtype=int) for _ in offsets]
    hidden_by_cell = {cell: idx for idx, cell in hidden_positions.items()}

    for r, c in np.argwhere(master != BLACK):
        cell = (int(r), int(c))
        covers = coverage[cell]
        if cell in hidden_by_cell:
            reveal = [idx for idx in covers if idx != hidden_by_cell[cell]]
        elif cell in always_show or len(covers) == 1:
            reveal = covers
        else:
            reveal = [idx for idx in covers if rng.random() < 0.6]
            if not reveal:
                reveal = [rng.choice(covers)]

        for idx in reveal:
            row_offset, col_offset = offsets[idx]
            crops[idx][cell[0] - row_offset, cell[1] - col_offset] = master[cell]

    return crops


def _choose_layout(rng: random.Random, count: int, outer_h: int, outer_w: int) -> list[list[int]]:
    if count <= 3 and rng.random() < 0.5:
        return [list(range(count))]
    first = (count + 1) // 2
    return [list(range(first)), list(range(first, count))]


def _draw_window(
    grid: np.ndarray,
    top: int,
    left: int,
    interior: np.ndarray,
    top_len: int,
    left_len: int,
    row_offset: int,
    col_offset: int,
) -> None:
    inner_h, inner_w = interior.shape
    bottom = top + inner_h + 1
    right = left + inner_w + 1

    grid[top, left : right + 1] = GREY
    grid[bottom, left : right + 1] = GREY
    grid[top : bottom + 1, left] = GREY
    grid[top : bottom + 1, right] = GREY

    grid[top, left + 1 + col_offset : left + 1 + col_offset + top_len] = AZURE
    grid[top + 1 + row_offset : top + 1 + row_offset + left_len, left] = AZURE
    grid[top + 1 : bottom, left + 1 : right] = interior


def _plan_layout(
    offsets: list[tuple[int, int]],
    inner_h: int,
    inner_w: int,
    rng: random.Random,
) -> tuple[tuple[int, int], list[dict]] | None:
    outer_h, outer_w = inner_h + 2, inner_w + 2
    count = len(offsets)
    order = list(range(count))
    rng.shuffle(order)
    rows = _choose_layout(rng, count, outer_h, outer_w)
    rows = [[order[idx] for idx in row] for row in rows]

    row_widths = [len(row) * outer_w + max(0, len(row) - 1) for row in rows]
    content_w = max(row_widths)
    content_h = len(rows) * outer_h + max(0, len(rows) - 1)
    if content_w + 2 > 30 or content_h + 2 > 30:
        return None

    max_left = min(3, 30 - content_w - 1)
    max_right = min(3, 30 - content_w - 1)
    left_margin = rng.randint(1, max_left)
    right_margin = rng.randint(1, min(max_right, 30 - content_w - left_margin))
    max_top = min(3, 30 - content_h - 1)
    max_bottom = min(3, 30 - content_h - 1)
    top_margin = rng.randint(1, max_top)
    bottom_margin = rng.randint(1, min(max_bottom, 30 - content_h - top_margin))

    grid_h = content_h + top_margin + bottom_margin
    grid_w = content_w + left_margin + right_margin
    if grid_h > 30 or grid_w > 30:
        return None

    placements = [None] * count
    for row_idx, row in enumerate(rows):
        row_width = row_widths[row_idx]
        slack = grid_w - row_width
        start_c = 1 if slack <= 2 else rng.randint(1, slack - 1)
        start_r = top_margin + row_idx * (outer_h + 1)
        for col_idx, idx in enumerate(row):
            top = start_r
            left = start_c + col_idx * (outer_w + 1)
            row_offset, col_offset = offsets[idx]
            placements[idx] = {
                "top": top,
                "left": left,
                "top_len": 1 if col_offset == inner_w - 1 or rng.random() < 0.5 else 2,
                "left_len": 1 if row_offset == inner_h - 1 or rng.random() < 0.5 else 2,
            }
    return (grid_h, grid_w), placements


def _render_grid(
    offsets: list[tuple[int, int]],
    crops: list[np.ndarray],
    layout_shape: tuple[int, int],
    placements: list[dict],
) -> np.ndarray:
    grid = np.full(layout_shape, BLACK, dtype=int)
    for idx, interior in enumerate(crops):
        row_offset, col_offset = offsets[idx]
        placement = placements[idx]
        _draw_window(
            grid,
            placement["top"],
            placement["left"],
            interior,
            placement["top_len"],
            placement["left_len"],
            row_offset,
            col_offset,
        )
    return grid


def _full_crops(master: np.ndarray, offsets: list[tuple[int, int]], inner_h: int, inner_w: int) -> list[np.ndarray]:
    return [
        master[row_offset : row_offset + inner_h, col_offset : col_offset + inner_w].copy()
        for row_offset, col_offset in offsets
    ]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    branch = rng.choice(BRANCHES)
    inner_h, inner_w = rng.choice(INNER_SIZES)
    offsets, master_h, master_w = _choose_offsets(rng, branch, inner_h, inner_w)
    coverage = _coverage_map(offsets, inner_h, inner_w, master_h, master_w)
    hidden_positions = _assign_hidden_positions(coverage, offsets, inner_h, inner_w, rng)
    if hidden_positions is None:
        return None

    try:
        master, always_show = _build_master(
            branch, offsets, coverage, hidden_positions, inner_h, inner_w, master_h, master_w, rng
        )
    except ValueError:
        return None

    partial_crops = _sample_window_crops(
        master, offsets, coverage, hidden_positions, always_show, inner_h, inner_w, rng
    )
    solved_crops = _full_crops(master, offsets, inner_h, inner_w)

    for partial, solved in zip(partial_crops, solved_crops):
        if np.count_nonzero(solved) < 3:
            return None
        if np.count_nonzero(partial) < 2:
            return None
        if np.array_equal(partial, solved):
            return None

    layout = _plan_layout(offsets, inner_h, inner_w, rng)
    if layout is None:
        return None
    layout_shape, placements = layout
    grid = _render_grid(offsets, partial_crops, layout_shape, placements)
    solved_grid = _render_grid(offsets, solved_crops, layout_shape, placements)
    if not np.array_equal(main(grid), solved_grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
