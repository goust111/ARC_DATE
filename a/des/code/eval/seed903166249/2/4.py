import random
from collections import defaultdict, deque

import numpy as np


BG = 0
GROUND = 5
BASE = 9
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIRS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 0),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
FRAGMENT_COLORS = (1, 2, 3, 4, 6, 7, 8)

MAX_GENERATION_ATTEMPTS = 400
SHAPE_ATTEMPTS = 160


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _sort_coords(coords: np.ndarray) -> np.ndarray:
    order = np.lexsort((coords[:, 1], coords[:, 0]))
    return coords[order]


def _select_anchor_col(coords: np.ndarray) -> int:
    cols, counts = np.unique(coords[:, 1], return_counts=True)
    center = (int(cols.min()) + int(cols.max())) / 2.0
    ranked = sorted(
        zip(cols.tolist(), counts.tolist()),
        key=lambda item: (abs(item[0] - center), -item[1], item[0]),
    )
    return int(ranked[0][0])


def _describe_piece(coords: np.ndarray, color: int) -> dict[str, object]:
    arr = _sort_coords(np.asarray(coords, dtype=int))
    rows = arr[:, 0]
    cols = arr[:, 1]
    anchor_col = _select_anchor_col(arr)
    anchor_rows = rows[cols == anchor_col]
    assert anchor_rows.size > 0

    return {
        "coords": arr,
        "color": int(color),
        "min_row": int(rows.min()),
        "max_row": int(rows.max()),
        "min_col": int(cols.min()),
        "max_col": int(cols.max()),
        "anchor_col": int(anchor_col),
        "anchor_top": int(anchor_rows.min()),
        "anchor_bottom": int(anchor_rows.max()),
    }


def _extract_pieces(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    pieces: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            piece = _describe_piece(np.asarray(coords, dtype=int), color)
            assert 2 <= len(coords) <= 5
            pieces.append(piece)

    assert pieces
    return pieces


def _assign_base(piece: dict[str, object], base_cols: list[int]) -> int:
    min_col = int(piece["min_col"])
    max_col = int(piece["max_col"])
    center = (min_col + max_col) / 2.0
    coords = np.asarray(piece["coords"], dtype=int)
    col_counts = {int(col): int(np.sum(coords[:, 1] == col)) for col in np.unique(coords[:, 1])}

    overlapping = [col for col in base_cols if min_col <= col <= max_col]
    candidates = overlapping if overlapping else base_cols
    ranked = sorted(
        candidates,
        key=lambda col: (-col_counts.get(col, 0), abs(col - center), abs(col - int(piece["anchor_col"])), col),
    )
    return int(ranked[0])


def _piece_sort_key(piece: dict[str, object]) -> tuple[int, int, int, int]:
    return (
        int(piece["max_row"]),
        int(piece["min_row"]),
        int(piece["min_col"]),
        int(piece["max_col"]),
    )


def _project_piece(
    piece: dict[str, object],
    base_col: int,
    support_row: int,
) -> tuple[np.ndarray, int]:
    coords = np.asarray(piece["coords"], dtype=int)
    dr = int(support_row) - int(piece["anchor_bottom"])
    dc = int(base_col) - int(piece["anchor_col"])
    placed = coords + np.array([dr, dc], dtype=int)
    next_support = int(dr + int(piece["anchor_top"]) - 1)
    return placed, next_support


def _stack_layout(
    grouped: dict[int, list[dict[str, object]]],
    base_cols: list[int],
    ground_row: int,
) -> tuple[dict[int, list[np.ndarray]], int]:
    layouts: dict[int, list[np.ndarray]] = {}
    top_row = ground_row
    for base_col in base_cols:
        support = ground_row - 1
        placed_group: list[np.ndarray] = []
        for piece in sorted(grouped.get(base_col, []), key=_piece_sort_key):
            placed, support = _project_piece(piece, base_col, support)
            top_row = min(top_row, int(placed[:, 0].min()))
            placed_group.append(placed)
        layouts[base_col] = placed_group
    return layouts, int(top_row)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stack scattered monochrome fragments onto maroon pillar bases by vertical
    projection order. Fragments are grouped by the base beneath their horizontal
    span, sorted from highest to lowest in the input, then re-stacked above that
    base using the fragment's anchor column as the contact shadow.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 3 and w >= 5

    ground = g[-1, :]
    assert np.all(np.isin(ground, (GROUND, BASE)))
    base_cols = [int(c) for c in np.flatnonzero(ground == BASE)]
    assert base_cols

    pieces = _extract_pieces(g[:-1, :])
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for piece in pieces:
        grouped[_assign_base(piece, base_cols)].append(piece)

    out = np.zeros_like(g)
    out[-1, :] = ground
    layouts, top_row = _stack_layout(grouped, base_cols, h - 1)
    assert top_row >= 0

    for base_col in base_cols:
        group = grouped.get(base_col, [])
        for piece, placed in zip(sorted(group, key=_piece_sort_key), layouts[base_col]):
            assert int(placed[:, 0].max()) < h - 1
            assert int(placed[:, 1].min()) >= 0
            assert int(placed[:, 1].max()) < w
            out[placed[:, 0], placed[:, 1]] = int(piece["color"])

    return out


def _normalize(coords: set[tuple[int, int]]) -> np.ndarray:
    arr = np.asarray(sorted(coords), dtype=int)
    arr[:, 0] -= int(arr[:, 0].min())
    arr[:, 1] -= int(arr[:, 1].min())
    return _sort_coords(arr)


def _random_shape(size: int) -> np.ndarray:
    cells = {(random.randrange(4), random.randrange(4))}
    while len(cells) < size:
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for dr, dc in DIRS4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < 4 and 0 <= nc < 4 and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            break
        cells.add(random.choice(tuple(frontier)))
    return _normalize(cells)


def _sample_shape(require_overlap_friendly: bool | None) -> dict[str, object] | None:
    for _ in range(SHAPE_ATTEMPTS):
        shape = _random_shape(random.randint(2, 5))
        piece = _describe_piece(shape, 1)
        if int(piece["max_col"]) > 2 or int(piece["max_row"]) > 3:
            continue
        if int(piece["anchor_bottom"]) != int(piece["max_row"]):
            continue
        overlap_friendly = int(piece["anchor_top"]) > int(piece["min_row"])
        if require_overlap_friendly is True and not overlap_friendly:
            continue
        if require_overlap_friendly is False and overlap_friendly:
            continue
        return piece
    return None


def _count_stack_overlaps(pieces: list[dict[str, object]]) -> int:
    support = 19
    occupied: set[tuple[int, int]] = set()
    overlap_count = 0
    for piece in pieces:
        placed, support = _project_piece(piece, 10, support)
        for r, c in map(tuple, placed.tolist()):
            if (r, c) in occupied:
                overlap_count += 1
            occupied.add((r, c))
    return overlap_count


def _sample_counts(n_bases: int, total_pieces: int) -> list[int]:
    focus = random.randrange(n_bases)
    counts = [0] * n_bases
    counts[focus] = 2
    remaining = total_pieces - 2

    other_bases = [i for i in range(n_bases) if i != focus]
    if remaining > 0 and other_bases:
        other = random.choice(other_bases)
        counts[other] += 1
        remaining -= 1

    while remaining > 0:
        choices = [i for i in range(n_bases) if counts[i] < 3]
        idx = random.choice(choices)
        counts[idx] += 1
        remaining -= 1

    return counts


def _sample_group_shapes(count: int, require_overlap: bool) -> list[dict[str, object]] | None:
    if count == 0:
        return []

    if not require_overlap:
        group: list[dict[str, object]] = []
        for _ in range(count):
            piece = _sample_shape(False)
            if piece is None:
                return None
            group.append(piece)
        return group

    if count == 1:
        piece = _sample_shape(False)
        return [piece] if piece is not None else None

    for _ in range(SHAPE_ATTEMPTS):
        lower = _sample_shape(True)
        upper = _sample_shape(None)
        if lower is None or upper is None:
            continue
        seed_group = [lower, upper]
        if _count_stack_overlaps(seed_group) == 0:
            continue
        while len(seed_group) < count:
            extra = _sample_shape(None)
            if extra is None:
                break
            seed_group.append(extra)
        if len(seed_group) == count:
            return seed_group
    return None


def _required_output_height(groups: list[list[dict[str, object]]]) -> int:
    support = -1
    top = 0
    for pieces in groups:
        support = -1
        for piece in pieces:
            placed, support = _project_piece(piece, 0, support)
            top = min(top, int(placed[:, 0].min()))
    return int(-top + 1)


def _mark_blocked(blocked: np.ndarray, coords: np.ndarray) -> None:
    h, w = blocked.shape
    for r, c in map(tuple, coords.tolist()):
        for dr, dc in DIRS8:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w:
                blocked[nr, nc] = True


def _place_input_group(
    grid: np.ndarray,
    blocked: np.ndarray,
    base_col: int,
    pieces: list[dict[str, object]],
) -> bool:
    if not pieces:
        return True

    h, w = grid.shape
    ground_row = h - 1
    prev_bottom = -1
    for index, piece in enumerate(pieces):
        coords = np.asarray(piece["coords"], dtype=int)
        remaining = len(pieces) - index - 1
        min_shift = -int(piece["min_col"])
        max_shift = (w - 1) - int(piece["max_col"])
        overlap_lo = base_col - int(piece["max_col"])
        overlap_hi = base_col - int(piece["min_col"])
        lo = max(min_shift, overlap_lo)
        hi = min(max_shift, overlap_hi)
        if lo > hi:
            return False

        min_bottom = max(int(piece["max_row"]), prev_bottom + 2)
        max_bottom = (ground_row - 1) - remaining * 2
        if min_bottom > max_bottom:
            return False

        placed_ok = False
        shifts = list(range(lo, hi + 1))
        random.shuffle(shifts)
        bottoms = list(range(min_bottom, max_bottom + 1))
        random.shuffle(bottoms)
        for bottom in bottoms:
            dr = bottom - int(piece["max_row"])
            for dc in shifts:
                placed = coords + np.array([dr, dc], dtype=int)
                if int(placed[:, 0].min()) < 0 or int(placed[:, 0].max()) >= ground_row:
                    continue
                if blocked[placed[:, 0], placed[:, 1]].any():
                    continue
                grid[placed[:, 0], placed[:, 1]] = int(piece["color"])
                _mark_blocked(blocked, placed)
                prev_bottom = bottom
                placed_ok = True
                break
            if placed_ok:
                break
        if not placed_ok:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n_bases = random.randint(2, 4)
    total_pieces = random.randint(3, min(6, len(FRAGMENT_COLORS)))
    require_overlap = random.random() < 0.5

    counts = _sample_counts(n_bases, total_pieces)
    overlap_base = max(range(n_bases), key=lambda idx: counts[idx])

    groups: list[list[dict[str, object]]] = []
    for idx, count in enumerate(counts):
        group = _sample_group_shapes(count, require_overlap=require_overlap and idx == overlap_base)
        if group is None:
            return None
        groups.append(group)

    if require_overlap:
        if _count_stack_overlaps(groups[overlap_base]) == 0:
            return None
    else:
        if any(_count_stack_overlaps(group) > 0 for group in groups):
            return None

    required_output = _required_output_height(groups)
    min_height = max(10, required_output + 2, max(counts) * 4 + 4)
    height = random.randint(min_height, min_height + 4)

    spacing = random.randint(4, 6)
    left_margin = random.randint(2, 4)
    right_margin = random.randint(2, 4)
    base_cols = [left_margin + spacing * i for i in range(n_bases)]
    width = base_cols[-1] + right_margin + 1

    grid = np.zeros((height, width), dtype=int)
    grid[-1, :] = GROUND
    for c in base_cols:
        grid[-1, c] = BASE

    colors = random.sample(FRAGMENT_COLORS, total_pieces)
    color_iter = iter(colors)
    colored_groups: list[list[dict[str, object]]] = []
    for group in groups:
        colored_group: list[dict[str, object]] = []
        for piece in group:
            colored = dict(piece)
            colored["color"] = int(next(color_iter))
            colored_group.append(colored)
        colored_groups.append(colored_group)

    blocked = np.zeros_like(grid, dtype=bool)
    blocked[-1, :] = True
    order = list(range(n_bases))
    random.shuffle(order)
    for idx in order:
        if not _place_input_group(grid, blocked, base_cols[idx], colored_groups[idx]):
            return None

    try:
        piece_count = len(_extract_pieces(grid[:-1, :]))
    except AssertionError:
        return None
    if piece_count != total_pieces:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
