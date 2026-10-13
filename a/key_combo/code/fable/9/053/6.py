import itertools
import math
import random

import numpy as np


LEGEND_ORIGIN = (1, 1)
LEGEND_STEP = 2
GRID_MIN = 22
GRID_MAX = 30
LEGEND_DIM_RANGE = (3, 5)
FILLED_COUNT_RANGE = (6, 10)
LEGEND_COLOR_COUNT_RANGE = (2, 4)
BLOCK_SIZE_RANGE = (2, 4)
BLOCK_GAP_RANGE = (1, 3)
ANCHOR_SIZE_OPTIONS = (3, 4, 5)
MAX_GENERATION_ATTEMPTS = 400
PALETTE = tuple(range(10))
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
ORIENTATION_NAMES = (
    "identity",
    "rot90",
    "rot180",
    "rot270",
    "flip_lr",
    "flip_ud",
    "transpose",
    "anti_transpose",
)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the large block grid from a rotated-or-mirrored corner minimap.

    Input structure:
    - The dominant background surrounds two object families.
    - In the top-left corner, isolated single pixels form a sparse minimap on a regular lattice.
    - Elsewhere, uniformly colored NxN solid square blocks occupy some sites of a larger regular block lattice, with other sites left empty.

    Rule:
    - Treat the corner minimap as needing one pose from the eight dihedral orientations: four rotations and four mirrored poses.
    - Find the unique pose and block-lattice offset for which every already placed square block matches the color at its mapped minimap coordinate.
    - Using that same pose, offset, and lattice step, project every non-background minimap pixel to the large lattice and fill any missing NxN block there.
    - Keep the minimap itself unchanged.

    Key insight:
    - The visible minimap is not in the main area's coordinate frame; the existing large blocks are anchors that force one unique orientation before completion is possible.
    """
    g = np.asarray(grid, dtype=int)
    solution = _solve(g)
    out = g.copy()
    oriented = solution["oriented"]
    block_size = solution["block_size"]
    step = solution["step"]
    base_r, base_c = solution["base"]
    bg = solution["bg"]
    for rr, cc in np.argwhere(oriented != bg):
        color = int(oriented[rr, cc])
        r0 = base_r + int(rr) * step
        c0 = base_c + int(cc) * step
        out[r0 : r0 + block_size, c0 : c0 + block_size] = color
    return out


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(coords)
    return parts


def _all_orientations(arr: np.ndarray) -> list[np.ndarray]:
    return [
        arr.copy(),
        np.rot90(arr, 1),
        np.rot90(arr, 2),
        np.rot90(arr, 3),
        np.fliplr(arr),
        np.flipud(arr),
        arr.T.copy(),
        np.fliplr(np.rot90(arr, 1)),
    ]


def _extract_legend_and_blocks(grid: np.ndarray, bg: int) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    singles: list[tuple[int, int, int]] = []
    blocks: list[tuple[int, int, int, int]] = []
    for coords in _components(grid != bg):
        if len(coords) == 1:
            r, c = coords[0]
            singles.append((r, c, int(grid[r, c])))
            continue
        r0, c0, r1, c1 = _bbox_coords(coords)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height != width or height < 2 or len(coords) != height * width:
            continue
        color = int(grid[r0, c0])
        if color == bg or not np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color):
            continue
        blocks.append((r0, c0, height, color))

    assert singles, "expected singleton minimap pixels"
    assert blocks, "expected at least one placed square block"

    rows = sorted({r for r, _, _ in singles})
    cols = sorted({c for _, c, _ in singles})
    row_diffs = [rows[i + 1] - rows[i] for i in range(len(rows) - 1) if rows[i + 1] > rows[i]]
    col_diffs = [cols[i + 1] - cols[i] for i in range(len(cols) - 1) if cols[i + 1] > cols[i]]
    diffs = row_diffs + col_diffs
    assert diffs, "legend must use multiple lattice rows or columns"
    step = diffs[0]
    for d in diffs[1:]:
        step = math.gcd(step, d)
    assert step > 0, "legend lattice step must be positive"

    rmin = min(r for r, _, _ in singles)
    cmin = min(c for _, c, _ in singles)
    logical = [((r - rmin) // step, (c - cmin) // step, color) for r, c, color in singles]
    h = max(r for r, _, _ in logical) + 1
    w = max(c for _, c, _ in logical) + 1
    legend = np.full((h, w), bg, dtype=int)
    for r, c, color in logical:
        legend[r, c] = color
    return legend, blocks


def _infer_block_size(blocks: list[tuple[int, int, int, int]]) -> int:
    sizes = [size for _r, _c, size, _color in blocks]
    values, counts = np.unique(np.array(sizes, dtype=int), return_counts=True)
    size = int(values[np.argmax(counts)])
    filtered = [b for b in blocks if b[2] == size]
    assert len(filtered) == len(blocks), "all blocks must share one size"
    return size


def _infer_step(blocks: list[tuple[int, int, int, int]]) -> int:
    rows = sorted({r for r, _c, _s, _color in blocks})
    cols = sorted({c for _r, c, _s, _color in blocks})
    diffs = [rows[i + 1] - rows[i] for i in range(len(rows) - 1) if rows[i + 1] > rows[i]]
    diffs += [cols[i + 1] - cols[i] for i in range(len(cols) - 1) if cols[i + 1] > cols[i]]
    assert diffs, "placed blocks must occupy multiple lattice rows or columns"
    step = diffs[0]
    for d in diffs[1:]:
        step = math.gcd(step, d)
    assert step > 0, "block step must be positive"
    return int(step)


def _candidate_solutions(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    legend, blocks = _extract_legend_and_blocks(g, bg)
    block_size = _infer_block_size(blocks)
    step = _infer_step(blocks)
    h, w = g.shape

    oriented_candidates = _all_orientations(legend)
    solutions: list[dict] = []
    for orient_idx, oriented in enumerate(oriented_candidates):
        points = [(int(r), int(c), int(oriented[r, c])) for r, c in np.argwhere(oriented != bg)]
        by_color: dict[int, list[tuple[int, int]]] = {}
        for rr, cc, color in points:
            by_color.setdefault(color, []).append((rr, cc))

        bases = set()
        for br, bc, _size, color in blocks:
            for rr, cc in by_color.get(color, []):
                bases.add((br - rr * step, bc - cc * step))

        valid_bases = []
        for base_r, base_c in sorted(bases):
            ok = True
            for rr, cc, color in points:
                tr0 = base_r + rr * step
                tc0 = base_c + cc * step
                if not (0 <= tr0 <= h - block_size and 0 <= tc0 <= w - block_size):
                    ok = False
                    break
            if not ok:
                continue

            for br, bc, _size, color in blocks:
                dr = br - base_r
                dc = bc - base_c
                if dr % step != 0 or dc % step != 0:
                    ok = False
                    break
                rr = dr // step
                cc = dc // step
                if not (0 <= rr < oriented.shape[0] and 0 <= cc < oriented.shape[1]):
                    ok = False
                    break
                if int(oriented[rr, cc]) != color:
                    ok = False
                    break
            if ok:
                valid_bases.append((base_r, base_c))

        if len(valid_bases) == 1:
            solutions.append(
                {
                    "bg": bg,
                    "legend": legend,
                    "oriented": oriented,
                    "orientation": orient_idx,
                    "orientation_name": ORIENTATION_NAMES[orient_idx],
                    "base": valid_bases[0],
                    "block_size": block_size,
                    "step": step,
                    "blocks": blocks,
                }
            )
    return solutions


def _solve(grid: np.ndarray) -> dict:
    solutions = _candidate_solutions(grid)
    assert len(solutions) == 1, "expected exactly one minimap pose and offset"
    return solutions[0]


def _grow_connected_mask(h: int, w: int, n: int) -> np.ndarray | None:
    cells = {(random.randrange(h), random.randrange(w))}
    while len(cells) < n:
        frontier = sorted(
            {
                (r + dr, c + dc)
                for r, c in cells
                for dr, dc in DIR4
                if 0 <= r + dr < h and 0 <= c + dc < w and (r + dr, c + dc) not in cells
            }
        )
        if not frontier:
            return None
        cells.add(random.choice(frontier))
    rows = {r for r, _ in cells}
    cols = {c for _, c in cells}
    if len(rows) < 2 or len(cols) < 2:
        return None
    if len(rows) != h or len(cols) != w:
        return None
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _random_legend(bg: int) -> np.ndarray | None:
    for _ in range(120):
        h = random.randint(*LEGEND_DIM_RANGE)
        w = random.randint(*LEGEND_DIM_RANGE)
        low = max(FILLED_COUNT_RANGE[0], h + w - 1)
        high = min(FILLED_COUNT_RANGE[1], h * w)
        if low > high:
            continue
        filled = random.randint(low, high)
        mask = _grow_connected_mask(h, w, filled)
        if mask is None:
            continue

        n_colors = random.randint(LEGEND_COLOR_COUNT_RANGE[0], LEGEND_COLOR_COUNT_RANGE[1])
        palette = random.sample([c for c in PALETTE if c != bg], n_colors)
        coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
        random.shuffle(coords)

        legend = np.full((h, w), bg, dtype=int)
        for idx, (r, c) in enumerate(coords):
            if idx < n_colors:
                color = palette[idx]
            else:
                color = random.choice(palette)
            legend[r, c] = color

        signatures = {arr.tobytes() + str(arr.shape).encode() for arr in _all_orientations(legend)}
        if len(signatures) < 8:
            continue
        return legend
    return None


def _place_legend_pixels(grid: np.ndarray, legend: np.ndarray, bg: int) -> None:
    base_r, base_c = LEGEND_ORIGIN
    for r, c in np.argwhere(legend != bg):
        grid[base_r + int(r) * LEGEND_STEP, base_c + int(c) * LEGEND_STEP] = int(legend[r, c])


def _paint_block(grid: np.ndarray, r0: int, c0: int, size: int, color: int) -> None:
    grid[r0 : r0 + size, c0 : c0 + size] = int(color)


def _render_from_anchor_mask(
    shape: tuple[int, int],
    bg: int,
    legend: np.ndarray,
    oriented: np.ndarray,
    block_size: int,
    step: int,
    base_r: int,
    base_c: int,
    anchor_mask: np.ndarray | None,
) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    _place_legend_pixels(grid, legend, bg)
    for rr, cc in np.argwhere(oriented != bg):
        if anchor_mask is not None and not anchor_mask[rr, cc]:
            continue
        _paint_block(
            grid,
            base_r + int(rr) * step,
            base_c + int(cc) * step,
            block_size,
            int(oriented[rr, cc]),
        )
    return grid


def _choose_base(shape: tuple[int, int], oriented_shape: tuple[int, int], block_size: int, step: int) -> tuple[int, int] | None:
    h, w = shape
    oh, ow = oriented_shape
    need_h = (oh - 1) * step + block_size
    need_w = (ow - 1) * step + block_size
    row_min = LEGEND_ORIGIN[0] + (LEGEND_DIM_RANGE[1] - 1) * LEGEND_STEP + 3
    col_min = LEGEND_ORIGIN[1] + (LEGEND_DIM_RANGE[1] - 1) * LEGEND_STEP + 4
    row_max = h - need_h
    col_max = w - need_w
    if row_min > row_max or col_min > col_max:
        return None
    return random.randint(row_min, row_max), random.randint(col_min, col_max)


def _candidate_anchor_masks(oriented: np.ndarray, bg: int) -> list[np.ndarray]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(oriented != bg)]
    masks: list[np.ndarray] = []
    for size in ANCHOR_SIZE_OPTIONS:
        if size >= len(coords):
            continue
        combos = list(itertools.combinations(range(len(coords)), size))
        random.shuffle(combos)
        for combo in combos[:80]:
            rows = {coords[i][0] for i in combo}
            cols = {coords[i][1] for i in combo}
            colors = {int(oriented[coords[i]]) for i in combo}
            if len(rows) < 2 or len(cols) < 2 or len(colors) < 2:
                continue
            mask = np.zeros_like(oriented, dtype=bool)
            for i in combo:
                r, c = coords[i]
                mask[r, c] = True
            masks.append(mask)
    random.shuffle(masks)
    return masks


def _expected_completed(shape: tuple[int, int], bg: int, legend: np.ndarray, oriented: np.ndarray, block_size: int, step: int, base_r: int, base_c: int) -> np.ndarray:
    return _render_from_anchor_mask(shape, bg, legend, oriented, block_size, step, base_r, base_c, None)


def _generation_attempt() -> np.ndarray | None:
    bg = random.choice(PALETTE)
    legend = _random_legend(bg)
    if legend is None:
        return None

    orient_idx = random.randint(1, 7)
    oriented = _all_orientations(legend)[orient_idx]
    block_size = random.randint(*BLOCK_SIZE_RANGE)
    gap = random.randint(*BLOCK_GAP_RANGE)
    step = block_size + gap

    need_h = max(GRID_MIN, (oriented.shape[0] - 1) * step + block_size + 2)
    need_w = max(GRID_MIN, (oriented.shape[1] - 1) * step + block_size + 2)
    if need_h > GRID_MAX or need_w > GRID_MAX:
        return None
    shape = (random.randint(need_h, GRID_MAX), random.randint(need_w, GRID_MAX))

    base = _choose_base(shape, oriented.shape, block_size, step)
    if base is None:
        return None
    base_r, base_c = base

    full = _expected_completed(shape, bg, legend, oriented, block_size, step, base_r, base_c)
    total_blocks = int(np.count_nonzero(oriented != bg))
    for anchor_mask in _candidate_anchor_masks(oriented, bg):
        kept = int(anchor_mask.sum())
        if total_blocks - kept < 2:
            continue
        inp = _render_from_anchor_mask(shape, bg, legend, oriented, block_size, step, base_r, base_c, anchor_mask)
        try:
            solution = _solve(inp)
        except AssertionError:
            continue
        if solution["orientation"] != orient_idx:
            continue
        if solution["base"] != (base_r, base_c):
            continue
        if solution["step"] != step or solution["block_size"] != block_size:
            continue
        if any(sol["orientation"] == 0 for sol in _candidate_solutions(inp)):
            continue
        out = main(inp)
        if not np.array_equal(out, full):
            continue
        if np.array_equal(inp, full):
            continue
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generation_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rotated-minimap completion input")
