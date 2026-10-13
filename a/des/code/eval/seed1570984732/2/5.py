import numpy as np


BLACK = 0
AZURE = 8
STAMP_COLORS = (2, 3, 4, 6)
MAX_GENERATION_ATTEMPTS = 400
RNG = np.random.default_rng()


def _bands(length: int, separators: list[int]) -> list[tuple[int, int]]:
    seps = sorted(int(s) for s in separators)
    assert len(seps) == 2
    bands: list[tuple[int, int]] = []
    start = 0
    for sep in seps:
        assert start < sep < length
        bands.append((start, sep))
        start = sep + 1
    assert start < length
    bands.append((start, length))
    return bands


def _rotate_cw(arr: np.ndarray, quarter_turns: int) -> np.ndarray:
    return np.rot90(arr, k=(-int(quarter_turns)) % 4)


def _detect_layout(grid: np.ndarray) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]], list[int], list[int]]:
    h, w = grid.shape
    bg = int(np.bincount(grid.ravel(), minlength=10).argmax())

    full_rows = [r for r in range(h) if np.all(grid[r, :] == grid[r, 0])]
    full_cols = [c for c in range(w) if np.all(grid[:, c] == grid[0, c])]
    row_colors = {int(grid[r, 0]) for r in full_rows if int(grid[r, 0]) != bg}
    col_colors = {int(grid[0, c]) for c in full_cols if int(grid[0, c]) != bg}
    sep_colors = row_colors & col_colors
    assert len(sep_colors) == 1, "expected one separator color shared by full rows and full columns"
    sep = next(iter(sep_colors))

    sep_rows = [r for r in full_rows if int(grid[r, 0]) == sep]
    sep_cols = [c for c in full_cols if int(grid[0, c]) == sep]
    assert len(sep_rows) == 2 and len(sep_cols) == 2, "expected a 3x3 partition lattice"

    row_bands = _bands(h, sep_rows)
    col_bands = _bands(w, sep_cols)
    assert len(row_bands) == 3 and len(col_bands) == 3

    row_sizes = {r1 - r0 for r0, r1 in row_bands}
    col_sizes = {c1 - c0 for c0, c1 in col_bands}
    assert len(row_sizes) == 1 and len(col_sizes) == 1, "all partitions must be the same size"
    return bg, sep, row_bands, col_bands, sep_rows, sep_cols


def _cell(grid: np.ndarray, row_bands: list[tuple[int, int]], col_bands: list[tuple[int, int]], ri: int, ci: int) -> np.ndarray:
    r0, r1 = row_bands[ri]
    c0, c1 = col_bands[ci]
    return grid[r0:r1, c0:c1]


def _find_reference_rotation(center_cell: np.ndarray, bg: int, sep: int) -> dict[int, int]:
    ch, cw = center_cell.shape
    candidates: list[list[int]] = []
    for r0 in range(ch - 2):
        for c0 in range(cw - 2):
            block = center_cell[r0 : r0 + 3, c0 : c0 + 3]
            if block[0, 1] != bg or block[1, 0] != bg or block[1, 1] != bg or block[1, 2] != bg or block[2, 1] != bg:
                continue
            corners = [int(block[0, 0]), int(block[0, 2]), int(block[2, 2]), int(block[2, 0])]
            if any(color in {bg, sep} for color in corners):
                continue
            if len(set(corners)) != 4:
                continue
            if np.count_nonzero(block != bg) != 4:
                continue
            candidates.append(corners)
    assert len(candidates) == 1, "expected a unique 3x3 reference square"
    corners = candidates[0]
    return {
        corners[0]: 0,
        corners[1]: 1,
        corners[2]: 2,
        corners[3]: 3,
    }


def _extract_stamp(cell: np.ndarray, bg: int) -> tuple[int, tuple[int, int], np.ndarray] | None:
    coords = np.argwhere(cell != bg)
    if coords.size == 0:
        return None
    colors = np.unique(cell[cell != bg])
    assert len(colors) == 1, "each occupied partition must contain exactly one monochrome stamp"
    color = int(colors[0])
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0) + 1
    patch = cell[r0:r1, c0:c1].copy()
    return color, (int(r0), int(c0)), patch


def _paste(dst: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    h, w = patch.shape
    target = dst[top : top + h, left : left + w]
    mask = patch != bg
    target[mask] = patch[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each outer stamp to the point-symmetric partition and rotate it
    according to its color's corner position in the center reference square.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg, sep, row_bands, col_bands, sep_rows, sep_cols = _detect_layout(g)
    rotation_by_color = _find_reference_rotation(_cell(g, row_bands, col_bands, 1, 1), bg, sep)

    out = np.full_like(g, bg)
    out[sep_rows, :] = sep
    out[:, sep_cols] = sep

    for ri in range(3):
        for ci in range(3):
            if (ri, ci) == (1, 1):
                continue
            cell = _cell(g, row_bands, col_bands, ri, ci)
            stamp = _extract_stamp(cell, bg)
            if stamp is None:
                continue

            color, (local_r, local_c), patch = stamp
            assert color in rotation_by_color, "stamp color must appear in the reference square"
            rotated = _rotate_cw(patch, rotation_by_color[color])

            tri, tci = 2 - ri, 2 - ci
            tr0, tr1 = row_bands[tri]
            tc0, tc1 = col_bands[tci]
            cell_h = tr1 - tr0
            cell_w = tc1 - tc0
            assert local_r + rotated.shape[0] <= cell_h
            assert local_c + rotated.shape[1] <= cell_w
            _paste(out, rotated, tr0 + local_r, tc0 + local_c, bg)

    return out


def _normalize(coords: set[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return sorted((r - min_r, c - min_c) for r, c in coords)


def _random_shape(area: int, max_h: int, max_w: int) -> np.ndarray | None:
    for _ in range(80):
        coords = {(0, 0)}
        while len(coords) < area:
            frontier = []
            for r, c in coords:
                frontier.extend([(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)])
            frontier = [pos for pos in frontier if pos not in coords]
            if not frontier:
                break
            choice = frontier[int(RNG.integers(0, len(frontier)))]
            coords.add(choice)

        norm = _normalize(coords)
        h = max(r for r, _ in norm) + 1
        w = max(c for _, c in norm) + 1
        if h > max_h or w > max_w:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in norm:
            mask[r, c] = True

        if bool(RNG.integers(0, 2)):
            mask = np.fliplr(mask)
        mask = _rotate_cw(mask, int(RNG.integers(0, 4)))
        return mask
    return None


def _paint_shape(cell_h: int, cell_w: int, color: int, turns: int) -> np.ndarray | None:
    for _ in range(100):
        area = int(RNG.integers(3, 9))
        mask = _random_shape(area, cell_h, cell_w)
        if mask is None:
            continue

        h, w = mask.shape
        rotated = _rotate_cw(mask, turns)
        rh, rw = rotated.shape
        max_top = cell_h - max(h, rh)
        max_left = cell_w - max(w, rw)
        if max_top < 0 or max_left < 0:
            continue

        top = int(RNG.integers(0, max_top + 1))
        left = int(RNG.integers(0, max_left + 1))
        cell = np.full((cell_h, cell_w), BLACK, dtype=int)
        cell[top : top + h, left : left + w][mask] = color
        return cell
    return None


def _build_reference(center_cell: np.ndarray) -> dict[int, int]:
    perm = list(RNG.permutation(STAMP_COLORS))
    ref = np.full((3, 3), BLACK, dtype=int)
    ref[0, 0] = perm[0]
    ref[0, 2] = perm[1]
    ref[2, 2] = perm[2]
    ref[2, 0] = perm[3]

    top = int(RNG.integers(0, center_cell.shape[0] - 2))
    left = int(RNG.integers(0, center_cell.shape[1] - 2))
    center_cell[top : top + 3, left : left + 3] = ref
    return {
        perm[0]: 0,
        perm[1]: 1,
        perm[2]: 2,
        perm[3]: 3,
    }


def _gen_attempt() -> np.ndarray | None:
    cell_h = int(RNG.integers(5, 10))
    cell_w = int(RNG.integers(5, 10))
    height = 3 * cell_h + 2
    width = 3 * cell_w + 2

    grid = np.full((height, width), BLACK, dtype=int)
    sep_rows = [cell_h, 2 * cell_h + 1]
    sep_cols = [cell_w, 2 * cell_w + 1]
    grid[sep_rows, :] = AZURE
    grid[:, sep_cols] = AZURE

    row_bands = _bands(height, sep_rows)
    col_bands = _bands(width, sep_cols)
    center = _cell(grid, row_bands, col_bands, 1, 1)
    rotation_by_color = _build_reference(center)

    pair_states = [
        ((0, 0), (2, 2)),
        ((0, 1), (2, 1)),
        ((0, 2), (2, 0)),
        ((1, 0), (1, 2)),
    ]
    occupancy_options = (
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    )

    occupied = 0
    for pos_a, pos_b in pair_states:
        use_a, use_b = occupancy_options[int(RNG.integers(0, len(occupancy_options)))]
        for use_cell, (ri, ci) in ((use_a, pos_a), (use_b, pos_b)):
            if not use_cell:
                continue
            color = int(RNG.choice(STAMP_COLORS))
            turns = rotation_by_color[color]
            stamp = _paint_shape(cell_h, cell_w, color, turns)
            if stamp is None:
                return None
            r0, r1 = row_bands[ri]
            c0, c1 = col_bands[ci]
            grid[r0:r1, c0:c1] = stamp
            occupied += 1

    if occupied == 0:
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
    raise RuntimeError("failed to generate a valid sample")
