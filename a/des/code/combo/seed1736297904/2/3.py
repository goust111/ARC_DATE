from __future__ import annotations

import numpy as np


def _group_consecutive(indices: list[int]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    for idx in indices:
        if not runs or idx != runs[-1][1] + 1:
            runs.append((idx, idx))
        else:
            runs[-1] = (runs[-1][0], idx)
    return runs


def _bands(size: int, runs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start = 0
    for lo, hi in runs:
        if start < lo:
            bands.append((start, lo - 1))
        start = hi + 1
    if start < size:
        bands.append((start, size - 1))
    return bands


def _lattice_structure(grid: np.ndarray) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    full_rows = [r for r in range(h) if np.all(g[r, :] == g[r, 0])]
    full_cols = [c for c in range(w) if np.all(g[:, c] == g[0, c])]
    assert full_rows and full_cols

    row_colors = {int(g[r, 0]) for r in full_rows}
    col_colors = {int(g[0, c]) for c in full_cols}
    sep_colors = row_colors & col_colors
    assert len(sep_colors) == 1
    sep = next(iter(sep_colors))

    sep_rows = [r for r in full_rows if int(g[r, 0]) == sep]
    sep_cols = [c for c in full_cols if int(g[0, c]) == sep]
    assert sep_rows and sep_cols

    row_bands = _bands(h, _group_consecutive(sep_rows))
    col_bands = _bands(w, _group_consecutive(sep_cols))
    assert len(row_bands) >= 2 and len(col_bands) >= 1

    row_heights = {r1 - r0 + 1 for r0, r1 in row_bands}
    col_widths = {c1 - c0 + 1 for c0, c1 in col_bands}
    assert len(row_heights) == 1 and len(col_widths) == 1

    cell_h = next(iter(row_heights))
    cell_w = next(iter(col_widths))
    assert cell_h == cell_w

    non_sep = g[g != sep]
    vals, counts = np.unique(non_sep, return_counts=True)
    bg = int(vals[int(np.argmax(counts))])
    return sep, bg, row_bands, col_bands


def _extract_cell(cell: np.ndarray, bg: int) -> tuple[int, np.ndarray]:
    colors = np.unique(cell[cell != bg])
    assert len(colors) == 1
    color = int(colors[0])
    mask = cell == color
    assert mask.any()
    return color, mask


def _rotation_index(base: np.ndarray, target: np.ndarray) -> int:
    matches = [k for k in range(4) if np.array_equal(np.rot90(base, k), target)]
    assert len(matches) == 1
    return int(matches[0])


def _mask_key(mask: np.ndarray) -> bytes:
    return mask.tobytes()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the bottom lattice row as a per-column rotation reference.

    Input structure
    - Black background with a full-row/full-column separator lattice.
    - Every cell contains one monochrome sprite on the black background.
    - All sprites share one underlying shape; the bottom row encodes rotations.

    Rule
    - Read each bottom-row sprite as the rotation command for its column.
    - Rotate every sprite above it in the same column by that amount.
    - Keep the bottom reference row unchanged.
    """
    g = np.asarray(grid, dtype=int)
    _sep, bg, row_bands, col_bands = _lattice_structure(g)
    n_rows, n_cols = len(row_bands), len(col_bands)

    colors: list[list[int]] = []
    masks: list[list[np.ndarray]] = []
    for r0, r1 in row_bands:
        row_colors: list[int] = []
        row_masks: list[np.ndarray] = []
        for c0, c1 in col_bands:
            color, mask = _extract_cell(g[r0 : r1 + 1, c0 : c1 + 1], bg)
            row_colors.append(color)
            row_masks.append(mask)
        colors.append(row_colors)
        masks.append(row_masks)

    upper_masks = [masks[r][c] for r in range(n_rows - 1) for c in range(n_cols)]
    assert upper_masks
    counts: dict[bytes, int] = {}
    lookup: dict[bytes, np.ndarray] = {}
    for mask in upper_masks:
        key = _mask_key(mask)
        counts[key] = counts.get(key, 0) + 1
        lookup[key] = mask
    base_mask = lookup[max(counts, key=counts.get)]

    for row in masks:
        for mask in row:
            assert any(np.array_equal(np.rot90(base_mask, k), mask) for k in range(4))

    commands = [_rotation_index(base_mask, masks[-1][c]) for c in range(n_cols)]

    out = g.copy()
    for c, rot in enumerate(commands):
        for r in range(n_rows - 1):
            r0, r1 = row_bands[r]
            c0, c1 = col_bands[c]
            patch = out[r0 : r1 + 1, c0 : c1 + 1]
            patch[:] = bg
            patch[np.rot90(masks[r][c], rot)] = colors[r][c]
    return out


# --------------- generator ---------------

_RNG = np.random.default_rng()

BG = 0
SEP = 9
COLOR_POOL = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=int)
ROW_RANGE = (2, 6)
COL_RANGE = (2, 6)
CELL_SIZE_RANGE = (5, 9)
SPRITE_AREA_RANGE = (4, 8)
MAX_GENERATION_ATTEMPTS = 400


def _layout(count: int, cell_size: int, border: bool) -> tuple[list[tuple[int, int]], list[int], int]:
    bands: list[tuple[int, int]] = []
    seps: list[int] = []
    pos = 0
    if border:
        seps.append(0)
        pos = 1
    for i in range(count):
        bands.append((pos, pos + cell_size - 1))
        pos += cell_size
        if i != count - 1 or border:
            seps.append(pos)
            pos += 1
    return bands, seps, pos


def _neighbors(r: int, c: int, size: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr = r + dr
        nc = c + dc
        if 1 <= nr < size - 1 and 1 <= nc < size - 1:
            out.append((nr, nc))
    return out


def _random_sprite(cell_size: int) -> np.ndarray | None:
    max_area = min(SPRITE_AREA_RANGE[1], (cell_size - 2) * (cell_size - 2) - 1)
    if max_area < SPRITE_AREA_RANGE[0]:
        return None

    for _ in range(120):
        area = int(_RNG.integers(SPRITE_AREA_RANGE[0], max_area + 1))
        start = (
            int(_RNG.integers(1, cell_size - 1)),
            int(_RNG.integers(1, cell_size - 1)),
        )
        cells = {start}
        frontier = set(_neighbors(start[0], start[1], cell_size))
        while len(cells) < area and frontier:
            choice = list(frontier)[int(_RNG.integers(len(frontier)))]
            frontier.remove(choice)
            if choice in cells:
                continue
            cells.add(choice)
            for nbr in _neighbors(choice[0], choice[1], cell_size):
                if nbr not in cells:
                    frontier.add(nbr)
        if len(cells) != area:
            continue

        mask = np.zeros((cell_size, cell_size), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        coords = np.argwhere(mask)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        bbox_h = int(r1 - r0 + 1)
        bbox_w = int(c1 - c0 + 1)
        if bbox_h < 2 or bbox_w < 2:
            continue
        if int(mask.sum()) == bbox_h * bbox_w:
            continue

        rots = [np.rot90(mask, k) for k in range(4)]
        if len({_mask_key(rot) for rot in rots}) != 4:
            continue
        return mask
    return None


def _cell_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = int(color)
    return patch


def _expected_output(
    base_mask: np.ndarray,
    colors: np.ndarray,
    commands: np.ndarray,
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
    sep_rows: list[int],
    sep_cols: list[int],
    shape: tuple[int, int],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    out[sep_rows, :] = SEP
    out[:, sep_cols] = SEP
    n_rows = len(row_bands)
    for r, (r0, r1) in enumerate(row_bands):
        for c, (c0, c1) in enumerate(col_bands):
            rot = int(commands[c]) if r < n_rows - 1 else int(commands[c])
            mask = np.rot90(base_mask, rot)
            out[r0 : r1 + 1, c0 : c1 + 1] = _cell_patch(mask, int(colors[r, c]))
    return out


def _gen_attempt() -> np.ndarray | None:
    n_rows = int(_RNG.integers(*ROW_RANGE))
    n_cols = int(_RNG.integers(*COL_RANGE))
    cell_size = int(_RNG.integers(*CELL_SIZE_RANGE))
    border = bool(int(_RNG.integers(2)))

    row_bands, sep_rows, height = _layout(n_rows, cell_size, border)
    col_bands, sep_cols, width = _layout(n_cols, cell_size, border)

    base_mask = _random_sprite(cell_size)
    if base_mask is None:
        return None

    colors = _RNG.choice(COLOR_POOL, size=(n_rows, n_cols), replace=True).astype(int)
    if len(set(map(int, colors.reshape(-1)))) == 1:
        colors[0, 0] = int(_RNG.choice(COLOR_POOL[COLOR_POOL != colors[0, 0]]))

    commands = _RNG.integers(1, 4, size=n_cols, dtype=int)
    if n_cols > 1 and len(set(map(int, commands))) == 1:
        idx = int(_RNG.integers(n_cols))
        choices = [k for k in (1, 2, 3) if k != int(commands[idx])]
        commands[idx] = int(_RNG.choice(choices))

    grid = np.full((height, width), BG, dtype=int)
    grid[sep_rows, :] = SEP
    grid[:, sep_cols] = SEP

    for r, (r0, r1) in enumerate(row_bands):
        for c, (c0, c1) in enumerate(col_bands):
            rot = 0 if r < n_rows - 1 else int(commands[c])
            grid[r0 : r1 + 1, c0 : c1 + 1] = _cell_patch(np.rot90(base_mask, rot), int(colors[r, c]))

    expected = _expected_output(
        base_mask=base_mask,
        colors=colors,
        commands=commands,
        row_bands=row_bands,
        col_bands=col_bands,
        sep_rows=sep_rows,
        sep_cols=sep_cols,
        shape=grid.shape,
    )
    try:
        if not np.array_equal(main(grid), expected):
            return None
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
