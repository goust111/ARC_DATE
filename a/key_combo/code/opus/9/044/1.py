from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2

_RNG = np.random.default_rng()
_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_N8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    neighbors = _N4 if int(connectivity) == 4 else _N8
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in neighbors:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _tile_patch(n: int, tile_r: int, tile_c: int) -> np.ndarray:
    patch = np.full((n, n), BLUE, dtype=int)
    patch[int(tile_r), int(tile_c)] = BLACK
    return patch


def _array_side(n: int) -> int:
    return int(n * n + n - 1)


def _separator_index(origin: int, n: int, idx: int) -> int:
    return int(origin + idx * (n + 1) - 1)


def _downsample_or(mask: np.ndarray, n: int) -> np.ndarray:
    h, w = mask.shape
    assert h == w
    assert h % int(n) == 0
    k = h // int(n)
    assert k >= 1
    return mask.reshape(n, k, n, k).any(axis=(1, 3))


def _infer_tile_array(grid: np.ndarray) -> tuple[int, int, int, np.ndarray]:
    blue_comps = _components(grid == BLUE, connectivity=8)
    assert len(blue_comps) >= 3, "expected at least three visible blue tiles"

    tile_info = []
    sizes = set()
    for comp in blue_comps:
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h == w and h in (3, 4), "tile size must be 3 or 4"
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert set(map(int, np.unique(patch))) <= {BLACK, BLUE}
        holes = np.argwhere(patch == BLACK)
        assert holes.shape[0] == 1, "each visible tile must have one hole"
        hole_r, hole_c = map(int, holes[0])
        assert int(np.count_nonzero(patch == BLUE)) == h * h - 1
        sizes.add(int(h))
        tile_info.append((int(r0), int(c0), hole_r, hole_c))

    assert len(sizes) == 1
    n = int(next(iter(sizes)))
    step = n + 1

    origin_rows = {r0 - hole_r * step for r0, _c0, hole_r, _hole_c in tile_info}
    origin_cols = {c0 - hole_c * step for _r0, c0, _hole_r, hole_c in tile_info}
    assert len(origin_rows) == 1 and len(origin_cols) == 1, "tile holes must encode one shared array origin"
    array_r0 = int(next(iter(origin_rows)))
    array_c0 = int(next(iter(origin_cols)))
    side = _array_side(n)
    h, w = grid.shape
    assert 0 <= array_r0 <= h - side
    assert 0 <= array_c0 <= w - side

    present = np.zeros((n, n), dtype=bool)
    for tile_r in range(n):
        for tile_c in range(n):
            r0 = array_r0 + tile_r * step
            c0 = array_c0 + tile_c * step
            patch = grid[r0 : r0 + n, c0 : c0 + n]
            assert patch.shape == (n, n)
            if np.any(patch == BLUE):
                assert np.array_equal(patch, _tile_patch(n, tile_r, tile_c))
                present[tile_r, tile_c] = True
            else:
                assert np.all(patch == BLACK)

    for idx in range(1, n):
        sep_r = _separator_index(array_r0, n, idx)
        sep_c = _separator_index(array_c0, n, idx)
        assert np.all(grid[sep_r, array_c0 : array_c0 + side] == BLACK)
        assert np.all(grid[array_r0 : array_r0 + side, sep_c] == BLACK)

    return n, array_r0, array_c0, present


def _extract_red_object(grid: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    red_comps = _components(grid == RED, connectivity=8)
    assert len(red_comps) == 1, "expected exactly one red template object"
    box = _bbox(red_comps[0])
    r0, c0, r1, c1 = box
    return box, (grid[r0 : r1 + 1, c0 : c1 + 1] == RED)


def _render_completed(shape: tuple[int, int], n: int, array_r0: int, array_c0: int) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    step = n + 1
    for tile_r in range(n):
        for tile_c in range(n):
            r0 = array_r0 + tile_r * step
            c0 = array_c0 + tile_c * step
            out[r0 : r0 + n, c0 : c0 + n] = _tile_patch(n, tile_r, tile_c)
    return out


def _render_masked_output(
    shape: tuple[int, int],
    n: int,
    array_r0: int,
    array_c0: int,
    mask: np.ndarray,
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    step = n + 1
    for tile_r in range(n):
        for tile_c in range(n):
            if not bool(mask[tile_r, tile_c]):
                continue
            r0 = array_r0 + tile_r * step
            c0 = array_c0 + tile_c * step
            out[r0 : r0 + n, c0 : c0 + n] = _tile_patch(n, tile_r, tile_c)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a coordinate-hole tile array, then keep only the tiles selected by a downscaled monochrome template.

    Input structure:
    - The background is black.
    - A blue separated n-by-n array of n-by-n tiles (n = 3 or 4) appears somewhere in the grid.
      Visible tiles are solid blue except for one black hole whose local coordinate equals the tile's array coordinate;
      missing tiles are entirely black, and separator rows/columns are one pixel wide and black.
    - A single 8-connected red object sits either to the right of the array or below it.
      Its cropped bounding box is a square of side k*n for an integer k, so it can be downsampled to n-by-n.

    Rule:
    - Infer n and the full array origin from the visible blue tiles' hole coordinates.
    - Complete every missing tile so that tile (r, c) becomes fully blue except for a black hole at local coordinate (r, c).
    - Crop the red object to its bounding box and downsample it to n-by-n using blockwise OR:
      a downsampled cell is foreground iff its corresponding k-by-k block contains at least one red pixel.
    - Use that binary n-by-n result as a tile mask: keep completed tiles where the mask is foreground and clear masked-out tiles to black.
      Separator rows/columns remain black, and the red object itself disappears in the output.

    Key insight:
    - The blue holes label tile coordinates, which uniquely reconstructs the full tiled array, while the red shape only aligns with that array after integer downsampling.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    n, array_r0, array_c0, _present = _infer_tile_array(g)
    _red_box, red_crop = _extract_red_object(g)
    mask = _downsample_or(red_crop, n)

    return _render_masked_output(g.shape, n, array_r0, array_c0, mask)


def _random_connected_mask(n: int) -> np.ndarray:
    choices = [4, 5] if n == 3 else [7, 8, 9]
    for _ in range(200):
        target = int(_RNG.choice(choices))
        start = (int(_RNG.integers(0, n)), int(_RNG.integers(0, n)))
        cells = {start}
        while len(cells) < target:
            frontier = set()
            for r, c in cells:
                for dr, dc in _N8:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < n and 0 <= nc < n and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            frontier_list = list(frontier)
            cells.add(frontier_list[int(_RNG.integers(0, len(frontier_list)))])

        if len(cells) != target:
            continue
        mask = np.zeros((n, n), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if not (mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if int(mask.sum()) in (0, n * n):
            continue
        return mask
    raise RuntimeError("failed to generate a connected side-touching mask")


def _spanning_tree_edges(mask: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    cells = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    root = cells[0]
    visited = {root}
    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []

    while len(visited) < len(cells):
        candidates = []
        for cell in cells:
            if cell in visited:
                continue
            r, c = cell
            parents = []
            for dr, dc in _N8:
                parent = (r + dr, c + dc)
                if 0 <= parent[0] < mask.shape[0] and 0 <= parent[1] < mask.shape[1] and mask[parent] and parent in visited:
                    parents.append(parent)
            if parents:
                candidates.append((cell, parents))
        assert candidates, "mask must stay connected"
        cell, parents = candidates[int(_RNG.integers(0, len(candidates)))]
        parent = parents[int(_RNG.integers(0, len(parents)))]
        visited.add(cell)
        edges.append((parent, cell))
    return edges


def _add_connector(
    required: dict[tuple[int, int], set[tuple[int, int]]],
    parent: tuple[int, int],
    child: tuple[int, int],
    k: int,
) -> None:
    pr, pc = parent
    cr, cc = child
    dr = cr - pr
    dc = cc - pc

    if dr == 0 and dc == 1:
        row = int(_RNG.integers(0, k))
        required[parent].add((row, k - 1))
        required[child].add((row, 0))
    elif dr == 0 and dc == -1:
        row = int(_RNG.integers(0, k))
        required[parent].add((row, 0))
        required[child].add((row, k - 1))
    elif dr == 1 and dc == 0:
        col = int(_RNG.integers(0, k))
        required[parent].add((k - 1, col))
        required[child].add((0, col))
    elif dr == -1 and dc == 0:
        col = int(_RNG.integers(0, k))
        required[parent].add((0, col))
        required[child].add((k - 1, col))
    elif dr == 1 and dc == 1:
        required[parent].add((k - 1, k - 1))
        required[child].add((0, 0))
    elif dr == 1 and dc == -1:
        required[parent].add((k - 1, 0))
        required[child].add((0, k - 1))
    elif dr == -1 and dc == 1:
        required[parent].add((0, k - 1))
        required[child].add((k - 1, 0))
    elif dr == -1 and dc == -1:
        required[parent].add((0, 0))
        required[child].add((k - 1, k - 1))
    else:
        raise AssertionError("unexpected tree edge")


def _build_connected_block(k: int, required_points: set[tuple[int, int]]) -> np.ndarray:
    points = set(required_points)
    if not points:
        points.add((int(_RNG.integers(0, k)), int(_RNG.integers(0, k))))
    targets = list(points)
    anchor = targets[int(_RNG.integers(0, len(targets)))]

    connected = {anchor}
    for target in targets:
        rr, cc = anchor
        while (rr, cc) != target:
            rr += _sign(target[0] - rr)
            cc += _sign(target[1] - cc)
            connected.add((rr, cc))

    block = np.zeros((k, k), dtype=bool)
    for r, c in connected:
        block[r, c] = True

    extra_steps = int(_RNG.integers(0, k + 1))
    for _ in range(extra_steps):
        frontier = set()
        for r, c in np.argwhere(block):
            rr = int(r)
            cc = int(c)
            for dr, dc in _N8:
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < k and 0 <= nc < k and not block[nr, nc]:
                    frontier.add((nr, nc))
        if not frontier:
            break
        pick = list(frontier)[int(_RNG.integers(0, len(frontier)))]
        block[pick] = True

    return block


def _shortcut_sample_count(big: np.ndarray, mask: np.ndarray, k: int) -> int:
    n = mask.shape[0]
    view = big.reshape(n, k, n, k)
    samples = [view[:, 0, :, 0], view[:, k - 1, :, k - 1]]
    if k % 2 == 1:
        samples.append(view[:, k // 2, :, k // 2])
    return sum(1 for sample in samples if not np.array_equal(sample, mask))


def _build_scaled_pattern(mask: np.ndarray, k: int) -> np.ndarray:
    n = mask.shape[0]
    fg_cells = [tuple(map(int, rc)) for rc in np.argwhere(mask)]

    for _ in range(300):
        required = {cell: set() for cell in fg_cells}
        for parent, child in _spanning_tree_edges(mask):
            _add_connector(required, parent, child, k)

        top_cell = [cell for cell in fg_cells if cell[0] == 0][int(_RNG.integers(0, int(mask[0, :].sum())))]
        bottom_cell = [cell for cell in fg_cells if cell[0] == n - 1][int(_RNG.integers(0, int(mask[-1, :].sum())))]
        left_cell = [cell for cell in fg_cells if cell[1] == 0][int(_RNG.integers(0, int(mask[:, 0].sum())))]
        right_cell = [cell for cell in fg_cells if cell[1] == n - 1][int(_RNG.integers(0, int(mask[:, -1].sum())))]

        required[top_cell].add((0, int(_RNG.integers(0, k))))
        required[bottom_cell].add((k - 1, int(_RNG.integers(0, k))))
        required[left_cell].add((int(_RNG.integers(0, k)), 0))
        required[right_cell].add((int(_RNG.integers(0, k)), k - 1))

        blocks = {}
        for cell in fg_cells:
            blocks[cell] = _build_connected_block(k, required[cell])

        big = np.zeros((n * k, n * k), dtype=bool)
        for (cell_r, cell_c), block in blocks.items():
            r0 = cell_r * k
            c0 = cell_c * k
            big[r0 : r0 + k, c0 : c0 + k] = block

        comps = _components(big, connectivity=8)
        if len(comps) != 1:
            continue
        if _bbox(comps[0]) != (0, 0, n * k - 1, n * k - 1):
            continue
        if not np.array_equal(_downsample_or(big, n), mask):
            continue
        if _shortcut_sample_count(big, mask, k) < min(2, 2 + (k % 2)):
            continue
        return big

    raise RuntimeError("failed to build an irregular connected scaled pattern")


def _choose_present_tiles(mask: np.ndarray) -> np.ndarray:
    n = mask.shape[0]
    total = n * n
    fg = {int(x) for x in np.flatnonzero(mask.reshape(-1))}
    bg = {int(x) for x in np.flatnonzero((~mask).reshape(-1))}
    lo = max(3, int(np.ceil(0.35 * total)))
    hi = min(total - 1, int(np.floor(0.65 * total)))

    for _ in range(200):
        count = int(_RNG.integers(lo, hi + 1))
        chosen = {int(x) for x in _RNG.choice(total, size=count, replace=False)}
        absent = set(range(total)) - chosen
        if not (absent & fg):
            continue
        if not (absent & bg):
            continue
        if not (chosen & fg):
            continue
        if not (chosen & bg):
            continue
        present = np.zeros((n, n), dtype=bool)
        for idx in chosen:
            present[idx // n, idx % n] = True
        return present

    raise RuntimeError("failed to choose non-degenerate visible tiles")


def _layout_case(array_side: int, pattern_side: int) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    options = []
    for gap in range(2, 5):
        min_h = max(array_side, pattern_side)
        min_w = array_side + gap + pattern_side
        if min_h <= 30 and min_w <= 30:
            options.append(("right", gap))

        min_h = array_side + gap + pattern_side
        min_w = max(array_side, pattern_side)
        if min_h <= 30 and min_w <= 30:
            options.append(("below", gap))

    if not options:
        raise RuntimeError("no layout fits within 30x30")

    orientation, gap = options[int(_RNG.integers(0, len(options)))]

    if orientation == "right":
        min_h = max(array_side, pattern_side)
        min_w = array_side + gap + pattern_side
        h = int(min_h + _RNG.integers(0, min(4, 31 - min_h)))
        w = int(min_w + _RNG.integers(0, min(4, 31 - min_w)))
        array_c0 = int(_RNG.integers(0, w - min_w + 1))
        pattern_c0 = array_c0 + array_side + gap
        array_r0 = int(_RNG.integers(0, h - array_side + 1))
        pattern_r0 = int(_RNG.integers(0, h - pattern_side + 1))
        return (h, w), (array_r0, array_c0), (pattern_r0, pattern_c0)

    min_h = array_side + gap + pattern_side
    min_w = max(array_side, pattern_side)
    h = int(min_h + _RNG.integers(0, min(4, 31 - min_h)))
    w = int(min_w + _RNG.integers(0, min(4, 31 - min_w)))
    array_r0 = int(_RNG.integers(0, h - min_h + 1))
    pattern_r0 = array_r0 + array_side + gap
    array_c0 = int(_RNG.integers(0, w - array_side + 1))
    pattern_c0 = int(_RNG.integers(0, w - pattern_side + 1))
    return (h, w), (array_r0, array_c0), (pattern_r0, pattern_c0)


def _build_case() -> tuple[np.ndarray, np.ndarray]:
    n = 3 if bool(_RNG.integers(0, 2)) else 4
    k = int(_RNG.choice([2, 3])) if n == 3 else 2
    mask = _random_connected_mask(n)
    pattern = _build_scaled_pattern(mask, k)
    present = _choose_present_tiles(mask)

    side = _array_side(n)
    shape, (array_r0, array_c0), (pattern_r0, pattern_c0) = _layout_case(side, n * k)

    inp = np.zeros(shape, dtype=int)
    out = _render_masked_output(shape, n, array_r0, array_c0, mask)
    step = n + 1
    for tile_r in range(n):
        for tile_c in range(n):
            if not bool(present[tile_r, tile_c]):
                continue
            r0 = array_r0 + tile_r * step
            c0 = array_c0 + tile_c * step
            inp[r0 : r0 + n, c0 : c0 + n] = _tile_patch(n, tile_r, tile_c)
    inp[pattern_r0 : pattern_r0 + n * k, pattern_c0 : pattern_c0 + n * k][pattern] = RED

    return inp, out


def _validate_generated_case(inp: np.ndarray, expected: np.ndarray) -> None:
    n, array_r0, array_c0, present = _infer_tile_array(inp)
    red_box, red_crop = _extract_red_object(inp)
    mask = _downsample_or(red_crop, n)
    actual = main(inp)

    assert np.array_equal(actual, expected)
    assert np.array_equal(actual, _render_masked_output(inp.shape, n, array_r0, array_c0, mask))

    completed = _render_completed(inp.shape, n, array_r0, array_c0)
    assert not np.array_equal(completed, actual), "masking step must matter"

    observed_only = np.zeros_like(inp)
    step = n + 1
    for tile_r in range(n):
        for tile_c in range(n):
            if not (present[tile_r, tile_c] and mask[tile_r, tile_c]):
                continue
            r0 = array_r0 + tile_r * step
            c0 = array_c0 + tile_c * step
            observed_only[r0 : r0 + n, c0 : c0 + n] = _tile_patch(n, tile_r, tile_c)
    assert not np.array_equal(observed_only, actual), "completion step must matter"

    absent = ~present
    assert np.any(mask)
    assert np.any(~mask)
    assert np.any(absent & mask), "need a kept tile that was missing in the input"
    assert np.any(absent & ~mask), "need a removed tile that was missing in the input"

    side = _array_side(n)
    arr_r1 = array_r0 + side - 1
    arr_c1 = array_c0 + side - 1
    pat_r0, pat_c0, _pat_r1, _pat_c1 = red_box
    assert pat_c0 - arr_c1 - 1 >= 2 or pat_r0 - arr_r1 - 1 >= 2


def generate_input() -> np.ndarray:
    for _ in range(400):
        inp, out = _build_case()
        try:
            _validate_generated_case(inp, out)
            return inp
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid template-mask tile-array input")
