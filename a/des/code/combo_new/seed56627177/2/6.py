import random

import numpy as np


BG = 0
SEP = 5
COLOR_POOL = tuple(color for color in range(10) if color not in (BG, SEP))
GRID_SIZES = (3, 4, 5)
CELL_SIZES = {
    3: (4, 5, 6, 7, 8),
    4: (4, 5, 6),
    5: (4,),
}
GEN_ATTEMPTS = 400


def _runs(mask: np.ndarray, value: bool) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for idx, flag in enumerate(mask.tolist()):
        if bool(flag) == bool(value):
            if start is None:
                start = idx
        elif start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _grid_structure(g: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    candidates: list[tuple[int, list[tuple[int, int]], list[tuple[int, int]]]] = []
    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue

        row_sep = np.all(g == color, axis=1)
        col_sep = np.all(g == color, axis=0)
        if int(row_sep.sum()) < 2 or int(col_sep.sum()) < 2:
            continue

        row_cells = _runs(row_sep, False)
        col_cells = _runs(col_sep, False)
        if len(row_cells) != len(col_cells) or len(row_cells) < 3:
            continue

        row_sizes = {r1 - r0 + 1 for r0, r1 in row_cells}
        col_sizes = {c1 - c0 + 1 for c0, c1 in col_cells}
        if len(row_sizes) != 1 or len(col_sizes) != 1 or next(iter(row_sizes)) != next(iter(col_sizes)):
            continue

        row_sep_sizes = {r1 - r0 + 1 for r0, r1 in _runs(row_sep, True)}
        col_sep_sizes = {c1 - c0 + 1 for c0, c1 in _runs(col_sep, True)}
        if len(row_sep_sizes) != 1 or len(col_sep_sizes) != 1:
            continue

        candidates.append((color, row_cells, col_cells))

    assert len(candidates) == 1, "failed to identify a unique grey separator grid"
    return candidates[0]


def _transform_cell(cell: np.ndarray, rotate_cw: bool, flip_mode: str | None) -> np.ndarray:
    out = np.rot90(cell, 3 if rotate_cw else 1)
    if flip_mode == "lr":
        out = np.fliplr(out)
    elif flip_mode == "ud":
        out = np.flipud(out)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate every cell sprite by checkerboard parity, then optionally flip it when its top or left neighbor shares the same color.

    Input structure:
    - The grid is split by complete grey rows and complete grey columns into a regular N x N board of square cells, with N at least 3.
    - Inside each cell, black is the local background and one non-grey, non-black color paints a small asymmetric sprite.
    - The sprite shape is shared across cells, while the sprite color may repeat in adjacent cells.

    Rule:
    - Leave the grey separators unchanged and process each cell independently on its full interior canvas.
    - If row + col is even, rotate that cell interior 90 degrees clockwise; if row + col is odd, rotate it 270 degrees clockwise.
    - Then compare the current cell's sprite color with the cell above and the cell to the left: matching the top neighbor triggers a left-right flip, otherwise matching the left neighbor triggers an up-down flip, and if both match only the top-triggered left-right flip is applied.

    Key insight:
    - The transformation is two-stage: checkerboard parity decides the base rotation, while equal-colored top/left neighbors add a priority-ordered mirror flip that explains cells that do not follow a pure rotation pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = BG if np.any(g == BG) else int(np.bincount(g.ravel()).argmax())
    sep, row_cells, col_cells = _grid_structure(g, bg)
    n = len(row_cells)

    colors = np.full((n, n), -1, dtype=int)
    cells: list[list[np.ndarray]] = []
    for r, (r0, r1) in enumerate(row_cells):
        row_patches: list[np.ndarray] = []
        for c, (c0, c1) in enumerate(col_cells):
            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            fg = [int(color) for color in np.unique(patch) if int(color) not in (bg, sep)]
            assert len(fg) == 1, "each cell must contain exactly one monochrome sprite"
            colors[r, c] = fg[0]
            row_patches.append(patch.copy())
        cells.append(row_patches)

    out = g.copy()
    for r, (r0, r1) in enumerate(row_cells):
        for c, (c0, c1) in enumerate(col_cells):
            top_same = r > 0 and colors[r - 1, c] == colors[r, c]
            left_same = c > 0 and colors[r, c - 1] == colors[r, c]
            flip_mode = "lr" if top_same else "ud" if left_same else None
            out[r0 : r1 + 1, c0 : c1 + 1] = _transform_cell(
                cells[r][c],
                rotate_cw=((r + c) % 2 == 0),
                flip_mode=flip_mode,
            )
    return out


def _neighbors4(r: int, c: int, size: int) -> list[tuple[int, int]]:
    pts: list[tuple[int, int]] = []
    if r > 0:
        pts.append((r - 1, c))
    if r + 1 < size:
        pts.append((r + 1, c))
    if c > 0:
        pts.append((r, c - 1))
    if c + 1 < size:
        pts.append((r, c + 1))
    return pts


def _all_d4_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    for k in range(4):
        rot = np.rot90(mask, k)
        variants.append(rot)
        variants.append(np.fliplr(rot))
    return variants


def _good_mask(mask: np.ndarray) -> bool:
    if int(mask.sum()) < 4 or int(mask.sum()) >= mask.size - 1:
        return False
    if int(np.count_nonzero(mask.any(axis=0))) < 2 or int(np.count_nonzero(mask.any(axis=1))) < 2:
        return False
    keys = {(variant.tobytes(), variant.shape) for variant in _all_d4_variants(mask)}
    return len(keys) == 8


def _random_mask(size: int) -> np.ndarray | None:
    area_lo = max(4, size - 1)
    area_hi = min(size * size - 2, size + 4)
    for _ in range(200):
        target = random.randint(area_lo, area_hi)
        mask = np.zeros((size, size), dtype=bool)
        start = (random.randrange(size), random.randrange(size))
        mask[start] = True
        frontier = set(_neighbors4(start[0], start[1], size))

        while int(mask.sum()) < target and frontier:
            r, c = random.choice(tuple(frontier))
            frontier.remove((r, c))
            mask[r, c] = True
            for nr, nc in _neighbors4(r, c, size):
                if not mask[nr, nc]:
                    frontier.add((nr, nc))

        if int(mask.sum()) == target and _good_mask(mask):
            return mask
    return None


def _seed_layout(palette: list[int]) -> np.ndarray:
    a, b, c = palette[:3]
    return np.array(
        [
            [a, b, c],
            [a, b, b],
            [b, b, a],
        ],
        dtype=int,
    )


def _pick_different(palette: list[int], blocked: set[int]) -> int:
    choices = [color for color in palette if color not in blocked]
    assert choices
    return random.choice(choices)


def _color_layout(n: int, palette: list[int]) -> np.ndarray:
    layout = np.full((n, n), -1, dtype=int)
    layout[:3, :3] = _seed_layout(palette)

    for c in range(3, n):
        left = int(layout[0, c - 1])
        layout[0, c] = left if random.random() < 0.5 else _pick_different(palette, {left})
    for r in range(3, n):
        top = int(layout[r - 1, 0])
        layout[r, 0] = top if random.random() < 0.5 else _pick_different(palette, {top})

    for r in range(1, n):
        for c in range(1, n):
            if r < 3 and c < 3:
                continue
            top = int(layout[r - 1, c])
            left = int(layout[r, c - 1])
            choices: list[int] = []
            if top == left:
                choices.append(top)
                choices.append(_pick_different(palette, {top}))
            else:
                choices.append(top)
                choices.append(left)
                choices.append(_pick_different(palette, {top, left}))
            layout[r, c] = random.choice(choices)

    return layout


def _board_shape(n: int, cell_size: int) -> tuple[int, int]:
    side = n * cell_size + (n + 1)
    return side, side


def _render_board(mask: np.ndarray, layout: np.ndarray, transformed: bool) -> np.ndarray:
    n = int(layout.shape[0])
    cell_size = int(mask.shape[0])
    h, w = _board_shape(n, cell_size)
    grid = np.full((h, w), SEP, dtype=int)

    for r in range(n):
        for c in range(n):
            color = int(layout[r, c])
            patch = np.full((cell_size, cell_size), BG, dtype=int)
            patch[mask] = color
            if transformed:
                top_same = r > 0 and int(layout[r - 1, c]) == color
                left_same = c > 0 and int(layout[r, c - 1]) == color
                flip_mode = "lr" if top_same else "ud" if left_same else None
                patch = _transform_cell(patch, rotate_cw=((r + c) % 2 == 0), flip_mode=flip_mode)

            r0 = 1 + r * (cell_size + 1)
            c0 = 1 + c * (cell_size + 1)
            grid[r0 : r0 + cell_size, c0 : c0 + cell_size] = patch

    return grid


def _branch_coverage(layout: np.ndarray) -> set[str]:
    covered: set[str] = set()
    n = int(layout.shape[0])
    for r in range(1, n):
        for c in range(1, n):
            top_same = int(layout[r - 1, c]) == int(layout[r, c])
            left_same = int(layout[r, c - 1]) == int(layout[r, c])
            if top_same and left_same:
                covered.add("both")
            elif top_same:
                covered.add("top")
            elif left_same:
                covered.add("left")
            else:
                covered.add("none")
    return covered


def _generate_case() -> np.ndarray | None:
    n = random.choice(GRID_SIZES)
    cell_size = random.choice(CELL_SIZES[n])
    mask = _random_mask(cell_size)
    if mask is None:
        return None

    palette_size = random.randint(3, min(4, len(COLOR_POOL)))
    palette = random.sample(COLOR_POOL, palette_size)
    layout = _color_layout(n, palette)
    if _branch_coverage(layout) != {"top", "left", "both", "none"}:
        return None

    inp = _render_board(mask, layout, transformed=False)
    expected = _render_board(mask, layout, transformed=True)

    if inp.shape[0] > 30 or inp.shape[1] > 30:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid checkerboard-rotation ARC input")
