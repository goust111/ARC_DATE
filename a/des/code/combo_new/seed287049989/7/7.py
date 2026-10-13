import random
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np


AZURE = 8
BLACK = 0
GREY = 5
BOARD_COLORS = (1, 2, 3, 4, 6, 7, 9)
SEPARATOR = 1
ARC_MAX = 30
MAX_ATTEMPTS = 400


def _runs(mask: Sequence[bool]) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    start = None
    for idx, value in enumerate(mask):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _compose_layers(boards: Dict[int, np.ndarray], order: Sequence[int]) -> np.ndarray:
    h, w = next(iter(boards.values())).shape
    out = np.full((h, w), AZURE, dtype=int)
    for color in order:
        patch = boards[int(color)]
        out[patch == int(color)] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stack colored window-boards in the order encoded by the bottom strip, letting holes show lower layers.

    Input structure:
    - Azure background with several equal-sized monochrome rectangular boards arranged left-to-right.
    - Each board has a distinct color and contains black hole pixels inside the rectangle.
    - A bottom grey strip contains one colored pixel at each board-center column; the left-to-right color sequence encodes bottom-to-top stacking order.

    Rule:
    - Read the colored pixels on the grey strip from left to right to obtain the stacking order.
    - Extract each board by its horizontal slot above the strip; the board's own color is the unique strip color present in that slot.
    - Compose a single board-sized output from bottom to top: a board's colored pixels overwrite lower layers, while black holes remain transparent.
    - Any position that is a hole in every board stays azure.

    Key insight:
    - The bottom strip is not a label for spatial position; it is a permutation of board colors that specifies layer order, and black pixels act as transparency rather than paint.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    non_bg_rows = np.where(np.any(g != AZURE, axis=1))[0]
    assert non_bg_rows.size > 0
    order_row = int(non_bg_rows[-1])

    strip = g[order_row]
    marker_cols = [c for c, value in enumerate(strip) if value not in (AZURE, GREY)]
    assert marker_cols
    order = [int(strip[c]) for c in marker_cols]
    order_colors = set(order)
    assert len(order_colors) == len(order)

    upper = g[:order_row]
    occupied_cols = np.any(upper != AZURE, axis=0)
    board_runs = _runs(occupied_cols.tolist())
    assert board_runs and len(board_runs) == len(order)

    occupied_rows = np.any(upper[:, occupied_cols] != AZURE, axis=1)
    board_rows = np.where(occupied_rows)[0]
    assert board_rows.size > 0
    r0, r1 = int(board_rows[0]), int(board_rows[-1])

    widths = {c1 - c0 + 1 for c0, c1 in board_runs}
    assert len(widths) == 1
    board_h = r1 - r0 + 1
    assert board_h > 0

    boards: Dict[int, np.ndarray] = {}
    for c0, c1 in board_runs:
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        candidates = set(map(int, np.unique(patch))) & order_colors
        assert len(candidates) == 1
        color = candidates.pop()
        boards[color] = patch.copy()

    assert set(boards) == order_colors
    return _compose_layers(boards, order)


def _choose_geometry(n_boards: int) -> Tuple[int, int, int, int, int, int, int] | None:
    left = random.randint(0, 3)
    right = random.randint(0, 3)
    top = random.randint(0, 3)
    gap = random.randint(1, 3)
    bottom = random.randint(0, 3)

    width_budget = ARC_MAX - left - right - (n_boards - 1) * SEPARATOR
    widths = [w for w in (5, 7, 9) if n_boards * w <= width_budget]
    if not widths:
        return None

    board_w = random.choice(widths)
    heights = [
        h
        for h in range(5, 10)
        if (h - 2) * (board_w - 2) >= n_boards + 1
        and top + h + gap + 1 + bottom <= ARC_MAX
    ]
    if not heights:
        return None

    board_h = random.choice(heights)
    return board_h, board_w, top, gap, bottom, left, right


def _rect_cells(r0: int, c0: int, r1: int, c1: int) -> Iterable[Tuple[int, int]]:
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            yield r, c


def _make_board_mask(
    h: int,
    w: int,
    must_solid: Set[Tuple[int, int]],
    must_hole: Set[Tuple[int, int]],
) -> np.ndarray:
    mask = np.ones((h, w), dtype=bool)
    inner = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    free = [cell for cell in inner if cell not in must_solid and cell not in must_hole]

    mode = random.choice(("random", "rect", "double_rect", "stripe", "hybrid"))
    holes = set(must_hole)

    if free:
        if mode in ("rect", "double_rect", "hybrid"):
            rect_count = 1 if mode == "rect" else 2 if mode == "double_rect" else 1
            for _ in range(rect_count):
                rh = random.randint(1, max(1, h - 3))
                rw = random.randint(1, max(1, w - 3))
                rr = random.randint(1, h - 1 - rh)
                cc = random.randint(1, w - 1 - rw)
                for cell in _rect_cells(rr, cc, rr + rh - 1, cc + rw - 1):
                    if cell not in must_solid:
                        holes.add(cell)

        if mode == "stripe":
            if random.random() < 0.5:
                rr = random.randint(1, h - 2)
                for c in range(1, w - 1):
                    cell = (rr, c)
                    if cell not in must_solid:
                        holes.add(cell)
            else:
                cc = random.randint(1, w - 2)
                for r in range(1, h - 1):
                    cell = (r, cc)
                    if cell not in must_solid:
                        holes.add(cell)

        if mode in ("random", "hybrid"):
            p = random.uniform(0.18, 0.48)
            for cell in free:
                if random.random() < p:
                    holes.add(cell)

    for r, c in holes:
        mask[r, c] = False
    for r, c in must_solid:
        mask[r, c] = True
    return mask


def _visible_counts(masks: Dict[int, np.ndarray], order: Sequence[int]) -> Tuple[Dict[int, int], int]:
    covered = np.zeros_like(next(iter(masks.values())), dtype=bool)
    visible: Dict[int, int] = {}
    for color in reversed(order):
        vis = masks[int(color)] & ~covered
        visible[int(color)] = int(vis.sum())
        covered |= masks[int(color)]
    return visible, int((~covered).sum())


def _stack_order(spatial_colors: Sequence[int]) -> List[int]:
    if len(spatial_colors) == 1:
        return list(spatial_colors)
    if random.random() < 0.5:
        return list(spatial_colors)
    perm = list(spatial_colors)
    while perm == list(spatial_colors):
        random.shuffle(perm)
    return perm


def _build_masks(colors: Sequence[int], order: Sequence[int], h: int, w: int) -> Dict[int, np.ndarray]:
    inner = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    chosen = random.sample(inner, len(colors) + 1)
    shared_hole = chosen[0]
    witnesses = chosen[1:]

    must_solid: Dict[int, Set[Tuple[int, int]]] = {int(color): set() for color in colors}
    must_hole: Dict[int, Set[Tuple[int, int]]] = {int(color): {shared_hole} for color in colors}

    for idx, color in enumerate(order):
        witness = witnesses[idx]
        must_solid[int(color)].add(witness)
        for higher in order[idx + 1 :]:
            must_hole[int(higher)].add(witness)

    masks = {
        int(color): _make_board_mask(h, w, must_solid[int(color)], must_hole[int(color)])
        for color in colors
    }
    visible, bg_visible = _visible_counts(masks, order)
    assert bg_visible > 0
    assert all(visible[int(color)] > 0 for color in order)
    return masks


def _render_input(
    spatial_colors: Sequence[int],
    order: Sequence[int],
    masks: Dict[int, np.ndarray],
    board_h: int,
    board_w: int,
    top: int,
    gap: int,
    bottom: int,
    left: int,
    right: int,
    full_width_bar: bool,
) -> np.ndarray:
    n_boards = len(spatial_colors)
    height = top + board_h + gap + 1 + bottom
    width = left + n_boards * board_w + (n_boards - 1) * SEPARATOR + right
    grid = np.full((height, width), AZURE, dtype=int)

    for idx, color in enumerate(spatial_colors):
        c0 = left + idx * (board_w + SEPARATOR)
        patch = np.full((board_h, board_w), BLACK, dtype=int)
        patch[masks[int(color)]] = int(color)
        grid[top : top + board_h, c0 : c0 + board_w] = patch

    order_row = top + board_h + gap
    if full_width_bar:
        grid[order_row, :] = GREY
    else:
        bar_start = left
        bar_end = left + n_boards * board_w + (n_boards - 1) * SEPARATOR
        grid[order_row, bar_start:bar_end] = GREY

    centers = [left + idx * (board_w + SEPARATOR) + board_w // 2 for idx in range(n_boards)]
    for col, color in zip(centers, order):
        grid[order_row, col] = int(color)

    return grid


def _gen_attempt() -> np.ndarray | None:
    n_boards = random.randint(2, 4)
    geometry = _choose_geometry(n_boards)
    if geometry is None:
        return None
    board_h, board_w, top, gap, bottom, left, right = geometry

    spatial_colors = random.sample(BOARD_COLORS, n_boards)
    order = _stack_order(spatial_colors)
    masks = _build_masks(spatial_colors, order, board_h, board_w)
    full_width_bar = random.random() < 0.5
    grid = _render_input(
        spatial_colors,
        order,
        masks,
        board_h,
        board_w,
        top,
        gap,
        bottom,
        left,
        right,
        full_width_bar,
    )
    expected = _compose_layers(
        {int(color): np.where(masks[int(color)], int(color), BLACK) for color in spatial_colors},
        order,
    )

    try:
        if np.array_equal(main(grid), expected):
            return grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
