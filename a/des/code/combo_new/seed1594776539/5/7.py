from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

AXIS_KINDS = ("h", "v")
GEN_GRID_ROWS = (18, 30)
GEN_GRID_COLS = (18, 30)
GEN_GROUP_RANGE = (2, 4)
GEN_BOX_HEIGHTS = range(5, 11)
GEN_BOX_WIDTHS = range(5, 11)
GEN_DIVIDER_MARGIN = 5
GEN_LOCAL_ATTEMPTS = 80
GEN_PLACEMENT_ATTEMPTS = 80
GEN_ATTEMPTS = 500


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _reflect(axis: tuple[str, int], r: int, c: int) -> tuple[int, int]:
    kind, split = axis
    if kind == "h":
        return 2 * split + 1 - r, c
    return r, 2 * split + 1 - c


def _score_axis(mask: np.ndarray, coords: np.ndarray, axis: tuple[str, int]) -> tuple[int, int]:
    h, w = mask.shape
    overlap = 0
    in_bounds = 0
    for r_raw, c_raw in coords:
        r = int(r_raw)
        c = int(c_raw)
        rr, cc = _reflect(axis, r, c)
        if 0 <= rr < h and 0 <= cc < w:
            in_bounds += 1
            if mask[rr, cc]:
                overlap += 1
    return overlap, in_bounds


def _best_axis(mask: np.ndarray) -> tuple[tuple[str, int], tuple[int, int], list[tuple[str, int]]]:
    coords = np.argwhere(mask)
    h, w = mask.shape
    assert coords.size > 0, "expected at least one colored pixel"

    best_axis: tuple[str, int] | None = None
    best_rank: tuple[int, int] | None = None
    tied: list[tuple[str, int]] = []

    for split in range(h - 1):
        axis = ("h", split)
        rank = _score_axis(mask, coords, axis)
        if best_rank is None or rank > best_rank:
            best_axis = axis
            best_rank = rank
            tied = [axis]
        elif rank == best_rank:
            tied.append(axis)

    for split in range(w - 1):
        axis = ("v", split)
        rank = _score_axis(mask, coords, axis)
        if best_rank is None or rank > best_rank:
            best_axis = axis
            best_rank = rank
            tied = [axis]
        elif rank == best_rank:
            tied.append(axis)

    assert best_axis is not None and best_rank is not None
    return best_axis, best_rank, tied


def _axis_side(axis: tuple[str, int], r: int, c: int) -> int:
    kind, split = axis
    if kind == "h":
        return -1 if r <= split else 1
    return -1 if c <= split else 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each color group by reflecting it across its own hidden half-cell symmetry axis.

    Input structure:
    - The background is black.
    - One grey straight line splits the grid into two halves but is not itself a mirror axis.
    - Every other foreground color forms a set of pixels that is nearly symmetric about one horizontal or vertical axis lying between two rows or two columns.
    - A color group may contain multiple disconnected pieces, and different colors may use different hidden axes.

    Rule:
    - Ignore the grey divider and process each non-black, non-grey color independently.
    - For one color, test every horizontal and vertical half-cell axis and score it by how many reflected pixels land on existing pixels of the same color.
    - Choose the axis with the highest overlap score.
    - Reflect every pixel of that color across the chosen axis, and fill any reflected position that is black in the original grid with the same color.
    - Keep all original colored pixels and the grey divider unchanged.

    Key insight:
    - The relevant structure is not connectivity but per-color near-symmetry: each color pool reveals its own best hidden axis, and the missing pixels are exactly the black cells reached by reflecting the surviving pixels across that axis.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    for color in np.unique(g):
        color = int(color)
        if color in {BLACK, GREY}:
            continue
        mask = g == color
        axis, _rank, _tied = _best_axis(mask)
        for r_raw, c_raw in np.argwhere(mask):
            r = int(r_raw)
            c = int(c_raw)
            rr, cc = _reflect(axis, r, c)
            if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1] and g[rr, cc] == BLACK and out[rr, cc] == BLACK:
                out[rr, cc] = color
    return out


def _mark_with_neighbors(mask: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = mask.shape
    for r, c in cells:
        mask[r, c] = True
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                mask[nr, nc] = True


def _grow_component(available: np.ndarray, size: int) -> list[tuple[int, int]] | None:
    choices = _components(available)
    choices = [part for part in choices if len(part) >= size]
    if not choices:
        return None

    for _ in range(24):
        region = random.choice(choices)
        seed_r, seed_c = map(int, region[random.randrange(len(region))])
        selected = {(seed_r, seed_c)}
        frontier: set[tuple[int, int]] = set()

        def add_frontier(r: int, c: int) -> None:
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < available.shape[0] and 0 <= nc < available.shape[1]:
                    if available[nr, nc] and (nr, nc) not in selected:
                        frontier.add((nr, nc))

        add_frontier(seed_r, seed_c)
        while len(selected) < size:
            if not frontier:
                break
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            selected.add(cell)
            add_frontier(*cell)
        if len(selected) == size:
            return sorted(selected)
    return None


def _feasible_counts(total: int, lo: int, hi: int) -> list[int]:
    return [count for count in range(1, 4) if count * lo <= total <= count * hi]


def _partition(total: int, count: int, lo: int, hi: int) -> list[int]:
    sizes: list[int] = []
    remaining = total
    for idx in range(count):
        left = count - idx - 1
        cur_lo = max(lo, remaining - left * hi)
        cur_hi = min(hi, remaining - left * lo)
        size = random.randint(cur_lo, cur_hi)
        sizes.append(size)
        remaining -= size
    random.shuffle(sizes)
    return sizes


def _available_base_mask(
    shape: tuple[int, int],
    axis: tuple[str, int],
    side: int,
    blocked: np.ndarray,
) -> np.ndarray:
    h, w = shape
    out = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if _axis_side(axis, r, c) != side:
                continue
            rr, cc = _reflect(axis, r, c)
            if 0 <= rr < h and 0 <= cc < w and not blocked[r, c] and not blocked[rr, cc]:
                out[r, c] = True
    return out


def _build_local_group(height: int, width: int) -> dict | None:
    for _ in range(GEN_LOCAL_ATTEMPTS):
        axis_kind = random.choice(AXIS_KINDS)
        axis_limit = height if axis_kind == "h" else width
        if axis_limit < 4:
            return None
        axis = (axis_kind, random.randint(1, axis_limit - 3))
        base_side = random.choice((-1, 1))

        available = _available_base_mask((height, width), axis, base_side, np.zeros((height, width), dtype=bool))
        capacity = int(available.sum())
        if capacity < 4:
            continue

        total_base = random.randint(4, min(8, capacity))
        counts = _feasible_counts(total_base, 2, 4)
        if not counts:
            continue
        sizes = _partition(total_base, random.choice(counts), 2, 4)

        full_mask = np.zeros((height, width), dtype=bool)
        blocked = np.zeros((height, width), dtype=bool)
        for size in sizes:
            available = _available_base_mask((height, width), axis, base_side, blocked)
            coords = _grow_component(available, size)
            if coords is None:
                break

            full_cells = coords[:]
            full_cells.extend(_reflect(axis, r, c) for r, c in coords)
            for r, c in full_cells:
                full_mask[r, c] = True
            _mark_with_neighbors(blocked, full_cells)
        else:
            best_axis, rank, tied = _best_axis(full_mask)
            if best_axis != axis or len(tied) != 1 or rank[0] != int(full_mask.sum()):
                continue

            delete_side = random.choice((-1, 1))
            candidates = [
                (r, c)
                for r, c in zip(*np.where(full_mask), strict=False)
                if _axis_side(axis, int(r), int(c)) == delete_side
            ]
            if len(candidates) < 3:
                continue

            max_delete = min(4, len(candidates) // 3)
            if max_delete < 1:
                continue
            delete_count = random.randint(1, max_delete)
            removed = random.sample(candidates, delete_count)

            inp_mask = full_mask.copy()
            for r, c in removed:
                inp_mask[r, c] = False

            best_axis, _rank, tied = _best_axis(inp_mask)
            if best_axis != axis or len(tied) != 1:
                continue

            return {
                "axis": axis,
                "input_mask": inp_mask,
                "output_mask": full_mask,
            }
    return None


def _sample_half_assignments(count: int) -> list[int]:
    halves = [0, 1]
    while len(halves) < count:
        halves.append(random.randrange(2))
    random.shuffle(halves)
    return halves[:count]


def _feasible_box_sizes(
    rows: int,
    cols: int,
    divider_kind: str,
    divider_index: int,
    half: int,
) -> list[tuple[int, int]]:
    if divider_kind == "h":
        usable_rows = divider_index if half == 0 else rows - divider_index - 1
        usable_cols = cols
    else:
        usable_rows = rows
        usable_cols = divider_index if half == 0 else cols - divider_index - 1

    return [
        (h, w)
        for h in GEN_BOX_HEIGHTS
        for w in GEN_BOX_WIDTHS
        if h <= usable_rows and w <= usable_cols
    ]


def _placeable_positions(
    reserved: np.ndarray,
    divider_kind: str,
    divider_index: int,
    half: int,
    height: int,
    width: int,
) -> list[tuple[int, int]]:
    rows, cols = reserved.shape
    if divider_kind == "h":
        row_lo, row_hi = (0, divider_index - height) if half == 0 else (divider_index + 1, rows - height)
        col_lo, col_hi = 0, cols - width
    else:
        row_lo, row_hi = 0, rows - height
        col_lo, col_hi = (0, divider_index - width) if half == 0 else (divider_index + 1, cols - width)

    positions: list[tuple[int, int]] = []
    for r0 in range(row_lo, row_hi + 1):
        for c0 in range(col_lo, col_hi + 1):
            rr0 = max(0, r0 - 1)
            rr1 = min(rows, r0 + height + 1)
            cc0 = max(0, c0 - 1)
            cc1 = min(cols, c0 + width + 1)
            if not reserved[rr0:rr1, cc0:cc1].any():
                positions.append((r0, c0))
    return positions


def _reserve_box(reserved: np.ndarray, r0: int, c0: int, height: int, width: int) -> None:
    rr0 = max(0, r0 - 1)
    rr1 = min(reserved.shape[0], r0 + height + 1)
    cc0 = max(0, c0 - 1)
    cc1 = min(reserved.shape[1], c0 + width + 1)
    reserved[rr0:rr1, cc0:cc1] = True


def _paste_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    coords = np.argwhere(mask)
    grid[r0 + coords[:, 0], c0 + coords[:, 1]] = color


def _axis_inside_half(
    axis: tuple[str, int],
    divider_kind: str,
    divider_index: int,
    half: int,
    box_origin: tuple[int, int],
) -> tuple[str, int]:
    kind, split = axis
    r0, c0 = box_origin
    global_axis = (kind, r0 + split) if kind == "h" else (kind, c0 + split)
    if divider_kind == "h" and half == 0:
        assert r0 + (split + 1 if kind == "h" else 0) <= divider_index
    if divider_kind == "h" and half == 1:
        assert r0 > divider_index
    if divider_kind == "v" and half == 0:
        assert c0 + (split + 1 if kind == "v" else 0) <= divider_index
    if divider_kind == "v" and half == 1:
        assert c0 > divider_index
    return global_axis


def _validate_group_axis(grid: np.ndarray, color: int, axis: tuple[str, int]) -> bool:
    mask = grid == color
    best_axis, _rank, tied = _best_axis(mask)
    return best_axis == axis and len(tied) == 1


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(*GEN_GRID_ROWS)
    cols = random.randint(*GEN_GRID_COLS)
    divider_kind = random.choice(AXIS_KINDS)
    if divider_kind == "h":
        divider_index = random.randint(GEN_DIVIDER_MARGIN, rows - GEN_DIVIDER_MARGIN - 1)
    else:
        divider_index = random.randint(GEN_DIVIDER_MARGIN, cols - GEN_DIVIDER_MARGIN - 1)

    group_count = random.randint(*GEN_GROUP_RANGE)
    colors = random.sample(PALETTE, group_count)
    halves = _sample_half_assignments(group_count)

    inp = np.full((rows, cols), BLACK, dtype=int)
    out = np.full((rows, cols), BLACK, dtype=int)
    reserved = np.zeros((rows, cols), dtype=bool)

    if divider_kind == "h":
        inp[divider_index, :] = GREY
        out[divider_index, :] = GREY
        reserved[divider_index, :] = True
    else:
        inp[:, divider_index] = GREY
        out[:, divider_index] = GREY
        reserved[:, divider_index] = True

    specs: list[dict] = []
    for color, half in zip(colors, halves, strict=False):
        placed = False
        sizes = _feasible_box_sizes(rows, cols, divider_kind, divider_index, half)
        random.shuffle(sizes)
        for height, width in sizes[:GEN_PLACEMENT_ATTEMPTS]:
            local = _build_local_group(height, width)
            if local is None:
                continue

            positions = _placeable_positions(reserved, divider_kind, divider_index, half, height, width)
            if not positions:
                continue
            r0, c0 = random.choice(positions)
            axis = _axis_inside_half(local["axis"], divider_kind, divider_index, half, (r0, c0))

            _paste_mask(inp, local["input_mask"], r0, c0, color)
            _paste_mask(out, local["output_mask"], r0, c0, color)
            _reserve_box(reserved, r0, c0, height, width)
            specs.append({"color": color, "axis": axis})
            placed = True
            break
        if not placed:
            return None

    if divider_kind == "h":
        if np.any(inp[:divider_index, :] != BLACK) and np.any(inp[divider_index + 1 :, :] != BLACK):
            pass
        elif group_count >= 2:
            return None
    else:
        if np.any(inp[:, :divider_index] != BLACK) and np.any(inp[:, divider_index + 1 :] != BLACK):
            pass
        elif group_count >= 2:
            return None

    for spec in specs:
        if not _validate_group_axis(out, spec["color"], spec["axis"]):
            return None
        if not _validate_group_axis(inp, spec["color"], spec["axis"]):
            return None

    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hidden-axis symmetry input")
