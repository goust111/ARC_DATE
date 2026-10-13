import numpy as np


_RNG = np.random.default_rng()
_MAX_RETRIES = 400
_PALETTE = tuple(range(10))


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _in_bounds(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    r, c = cell
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _branch_cells(
    seed: tuple[int, int],
    length: int,
    side: tuple[int, int],
    fwd: tuple[int, int],
    twig_offsets: tuple[int, ...],
) -> list[tuple[int, int]]:
    sr, sc = seed
    out: list[tuple[int, int]] = []
    for dist in range(1, length + 1):
        for sign in (-1, 1):
            br = sr + sign * dist * side[0]
            bc = sc + sign * dist * side[1]
            base = (br, bc)
            out.append(base)
            for twig in twig_offsets:
                out.append((br + twig * fwd[0], bc + twig * fwd[1]))
    return out


def _detect_trunk(grid: np.ndarray, bg: int):
    h, w = grid.shape
    full_rows = [r for r in range(h) if np.all(grid[r, :] != bg)]
    full_cols = [c for c in range(w) if np.all(grid[:, c] != bg)]
    assert bool(full_rows) != bool(full_cols), "expected exactly one full non-background trunk"

    if full_cols:
        orient = "vertical"
        axis = int(full_cols[0])
        line = grid[:, axis]
        line_cells = {(r, axis) for r in range(h)}
        seed_positions = [(r, axis) for r in range(h)]
        side = (0, 1)
    else:
        orient = "horizontal"
        axis = int(full_rows[0])
        line = grid[axis, :]
        line_cells = {(axis, c) for c in range(w)}
        seed_positions = [(axis, c) for c in range(w)]
        side = (1, 0)

    vals, counts = np.unique(line, return_counts=True)
    trunk_color = int(vals[np.argmax(counts)])
    seeds = [(pos[0], pos[1], int(grid[pos])) for pos in seed_positions if int(grid[pos]) != trunk_color]
    assert seeds, "expected at least one colored seed on the trunk"
    return orient, axis, trunk_color, seeds, line_cells, side


def _extract_example(
    grid: np.ndarray,
    bg: int,
    orient: str,
    axis: int,
    side: tuple[int, int],
):
    h, w = grid.shape
    mask = grid != bg
    if orient == "vertical":
        mask[:, axis] = False
    else:
        mask[axis, :] = False

    comps = [comp for comp in _components(mask) if len(comp) > 1]
    assert comps, "expected one corner example component"
    example = max(comps, key=len)
    example_set = set(example)

    rows = [r for r, _ in example]
    cols = [c for _, c in example]
    center_r = sum(rows) / len(rows)
    center_c = sum(cols) / len(cols)

    if orient == "vertical":
        fwd = (1, 0) if center_r < h / 2 else (-1, 0)
    else:
        fwd = (0, 1) if center_c < w / 2 else (0, -1)

    anchor_candidates = []
    for r, c in example:
        left = (r - side[0], c - side[1]) in example_set
        right = (r + side[0], c + side[1]) in example_set
        if left and right:
            anchor_candidates.append((r, c))
    assert len(anchor_candidates) == 1, "expected one example anchor"
    anchor = anchor_candidates[0]

    twig_offsets: set[int] = set()
    for sign in (-1, 1):
        side_cell = (anchor[0] + sign * side[0], anchor[1] + sign * side[1])
        assert side_cell in example_set, "example must branch symmetrically"
        for twig in (-1, 1):
            test = (side_cell[0] + twig * fwd[0], side_cell[1] + twig * fwd[1])
            if test in example_set:
                twig_offsets.add(twig)

    twig_tuple = tuple(sorted(twig_offsets))
    assert twig_tuple in {(-1, 1), (-1,), (1,)}, "example must be L- or T-shaped"
    return example_set, fwd, twig_tuple


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow mirrored tree branches from trunk seeds using the corner example as the local branching template.

    Input structure:
    - The background is dominant.
    - A single full-height or full-width central line forms the trunk; most cells on it share one trunk color.
    - Several one-cell seeds of distinct colors sit on that trunk, replacing trunk-color cells.
    - A small disconnected corner component shows one complete branch motif: mirrored side arms with either one forward twig per side (L) or two opposite twigs per side (T).

    Rule:
    - Detect the trunk orientation and the seed order along it.
    - Use the corner example to infer whether each outward branch step carries one twig (L) or two opposite twigs (T), and use the example corner to determine which trunk end is the root.
    - For each seed, let its branch length be the distance to the nearest grid edge measured along the trunk direction.
    - Repeat the inferred motif for every side-distance from 1 up to that length on both sides of the trunk, recoloring the generated branch pixels to the seed color.
    - Process seeds from root-near to root-far so later seeds overwrite earlier branch pixels; keep the trunk and the corner example unchanged.

    Key insight:
    - The corner component is not another object to copy literally: it is a local branch-growth rule, while each seed position encodes both how far that rule repeats and when its pixels should win in overlaps.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    orient, axis, _trunk_color, seeds, trunk_cells, side = _detect_trunk(g, bg)
    example_cells, fwd, twig_offsets = _extract_example(g, bg, orient, axis, side)

    if orient == "vertical":
        seeds = sorted(seeds, key=lambda item: item[0] if fwd == (1, 0) else -item[0])
    else:
        seeds = sorted(seeds, key=lambda item: item[1] if fwd == (0, 1) else -item[1])

    protected = trunk_cells | example_cells
    out = g.copy()
    h, w = g.shape
    for r, c, color in seeds:
        length = min(r, h - 1 - r) if orient == "vertical" else min(c, w - 1 - c)
        for cell in _branch_cells((r, c), int(length), side, fwd, twig_offsets):
            if _in_bounds(g.shape, cell) and cell not in protected:
                out[cell] = int(color)
    return out


def _rand_odd(low: int, high: int) -> int:
    choices = [n for n in range(low, high + 1) if n % 2 == 1]
    return int(_RNG.choice(choices))


def _choose_colors(count: int) -> list[int]:
    return [int(v) for v in _RNG.choice(_PALETTE, size=count, replace=False)]


def _example_cells(anchor: tuple[int, int], side: tuple[int, int], fwd: tuple[int, int], mode: str):
    ar, ac = anchor
    cells = {(ar, ac)}
    for sign in (-1, 1):
        sr = ar + sign * side[0]
        sc = ac + sign * side[1]
        cells.add((sr, sc))
        cells.add((sr + fwd[0], sc + fwd[1]))
        if mode == "T":
            cells.add((sr - fwd[0], sc - fwd[1]))
    return cells


def _seed_line_positions(size: int, cap_length: int, root_at_start: bool) -> list[int]:
    blocked = {0, 1, 2} if root_at_start else {size - 3, size - 2, size - 1}
    valid = [i for i in range(size) if 1 <= min(i, size - 1 - i) <= cap_length and i not in blocked]
    return valid


def _pick_seed_positions(valid: list[int], count: int) -> list[int]:
    chosen: set[int] = set()
    pairs = [(a, a + 1) for a in valid if a + 1 in valid]
    if pairs and _RNG.random() < 0.55:
        pair = pairs[int(_RNG.integers(len(pairs)))]
        chosen.update(pair)
    remaining = [v for v in valid if v not in chosen]
    need = count - len(chosen)
    if need > 0:
        extra = _RNG.choice(remaining, size=need, replace=False)
        chosen.update(int(v) for v in np.atleast_1d(extra))
    return sorted(chosen)


def _gen_attempt() -> np.ndarray | None:
    height = _rand_odd(13, 29)
    width = _rand_odd(13, 29)
    orient = str(_RNG.choice(["vertical", "horizontal"]))
    mode = str(_RNG.choice(["L", "T"]))

    bg, trunk_color, example_color = _choose_colors(3)
    grid = np.full((height, width), bg, dtype=int)

    if orient == "vertical":
        axis = width // 2
        side = (0, 1)
        root_at_start = bool(_RNG.integers(2) == 0)
        fwd = (1, 0) if root_at_start else (-1, 0)
        lateral_left = bool(_RNG.integers(2) == 0)
        anchor = (1 if root_at_start else height - 2, 1 if lateral_left else width - 2)
        line_positions = _seed_line_positions(height, axis - 2, root_at_start)
        line_cells = {(r, axis) for r in range(height)}
    else:
        axis = height // 2
        side = (1, 0)
        root_at_start = bool(_RNG.integers(2) == 0)
        fwd = (0, 1) if root_at_start else (0, -1)
        lateral_top = bool(_RNG.integers(2) == 0)
        anchor = (1 if lateral_top else height - 2, 1 if root_at_start else width - 2)
        line_positions = _seed_line_positions(width, axis - 2, root_at_start)
        line_cells = {(axis, c) for c in range(width)}

    if axis - 2 < 2 or len(line_positions) < 2:
        return None

    line_rows, line_cols = zip(*sorted(line_cells))
    grid[line_rows, line_cols] = trunk_color

    example_cells = _example_cells(anchor, side, fwd, mode)
    if any(not _in_bounds(grid.shape, cell) or cell in line_cells for cell in example_cells):
        return None
    for r, c in example_cells:
        grid[r, c] = example_color

    seed_count = int(_RNG.integers(2, min(5, len(line_positions)) + 1))
    positions = _pick_seed_positions(line_positions, seed_count)
    seed_colors = _choose_colors(seed_count)
    while any(color in {bg, trunk_color, example_color} for color in seed_colors):
        seed_colors = _choose_colors(seed_count)

    predicted_branches: set[tuple[int, int]] = set()
    twig_offsets = (1,) if mode == "L" else (-1, 1)
    for pos, color in zip(positions, seed_colors):
        if orient == "vertical":
            seed = (int(pos), axis)
            length = min(int(pos), height - 1 - int(pos))
        else:
            seed = (axis, int(pos))
            length = min(int(pos), width - 1 - int(pos))
        for cell in _branch_cells(seed, int(length), side, fwd, twig_offsets):
            if not _in_bounds(grid.shape, cell):
                return None
            if cell in example_cells or cell in line_cells:
                return None
            predicted_branches.add(cell)
        grid[seed] = int(color)

    if not predicted_branches:
        return None

    branch_rows = {r for r, _ in predicted_branches}
    branch_cols = {c for _, c in predicted_branches}
    if len(branch_rows) < 2 or len(branch_cols) < 2:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid branching-tree ARC sample")
