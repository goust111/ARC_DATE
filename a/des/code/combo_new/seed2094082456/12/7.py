import numpy as np


BLACK = 0
RED = 2
GREY = 5
COLOR_POOL = (1, 3, 4, 6, 7, 8, 9)
TRANSFORMS = ("hflip", "vflip", "cw", "ccw")
_RNG = np.random.default_rng()
MAX_GENERATION_ATTEMPTS = 400


def _bands(size: int, separators: list[int]) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start = 0
    for cut in separators:
        assert start < cut < size
        bands.append((start, cut))
        start = cut + 1
    assert start < size
    bands.append((start, size))
    return bands


def _full_line_indices(g: np.ndarray, axis: int) -> tuple[list[int], set[int]]:
    size = g.shape[axis]
    lines: list[int] = []
    colors: set[int] = set()
    for idx in range(size):
        line = g[idx, :] if axis == 0 else g[:, idx]
        if np.all(line == line[0]) and int(line[0]) != BLACK:
            lines.append(idx)
            colors.add(int(line[0]))
    return lines, colors


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    seen = {tuple(coords[0])}
    stack = [tuple(coords[0])]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _apply_transform(mask: np.ndarray, transform: str) -> np.ndarray:
    if transform == "hflip":
        return np.fliplr(mask)
    if transform == "vflip":
        return np.flipud(mask)
    if transform == "cw":
        return np.rot90(mask, -1)
    if transform == "ccw":
        return np.rot90(mask, 1)
    raise AssertionError(f"unknown transform: {transform}")


def _all_dihedral(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    cur = mask.copy()
    for _ in range(4):
        variants.append(cur.copy())
        variants.append(np.fliplr(cur))
        cur = np.rot90(cur)
    return variants


def _classify_region(
    g: np.ndarray,
    region_bounds: tuple[int, int, int, int],
    cell: tuple[int, int],
) -> dict[str, object]:
    rr0, rr1, cc0, cc1 = region_bounds
    patch = g[rr0:rr1, cc0:cc1]
    assert patch.size > 0
    assert not np.any(patch == GREY)

    active = patch != BLACK
    if not active.any():
        return {"kind": "empty", "cell": cell, "region": region_bounds}

    colors = set(map(int, np.unique(patch[active])))
    if RED in colors:
        assert colors == {RED, next(iter(colors - {RED}))}
        coords = np.argwhere(active)
        assert len(coords) == 2
        red_pos = np.argwhere(patch == RED)
        assert len(red_pos) == 1
        red_r, red_c = map(int, red_pos[0])
        other_color = next(iter(colors - {RED}))
        other_pos = np.argwhere(patch == other_color)
        assert len(other_pos) == 1
        oth_r, oth_c = map(int, other_pos[0])
        assert abs(red_r - oth_r) + abs(red_c - oth_c) == 1

        if red_r == oth_r:
            transform = "hflip" if red_c < oth_c else "vflip"
        else:
            transform = "cw" if red_r < oth_r else "ccw"
        return {
            "kind": "instruction",
            "cell": cell,
            "region": region_bounds,
            "color": int(other_color),
            "transform": transform,
        }

    assert len(colors) == 1
    color = int(next(iter(colors)))
    mask = patch == color
    assert int(mask.sum()) >= 3
    assert _connected(mask)
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0) + 1
    tight = mask[r0:r1, c0:c1]
    return {
        "kind": "pattern",
        "cell": cell,
        "region": region_bounds,
        "color": color,
        "bbox_top_left": (rr0 + int(r0), cc0 + int(c0)),
        "mask": tight,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Apply instruction dominoes to same-colored neighbor patterns inside a separator lattice.

    Input structure:
    - Black background with complete Grey separator rows and columns splitting the canvas into rectangular subregions.
    - Some subregions contain one monochrome 4-connected asymmetric pattern.
    - Other subregions contain exactly one 2-pixel instruction domino made of one Red pixel and one non-Red pixel whose color matches a nearby pattern.

    Rule:
    - Split the grid by the Grey full-row/full-column separators and classify each subregion as empty, pattern, or instruction.
    - Pair each instruction region with the adjacent pattern region of the same non-Red color.
    - Horizontal dominoes encode flips: Red on the left means horizontal flip, Red on the right means vertical flip.
    - Vertical dominoes encode rotations: Red on top means rotate 90 degrees clockwise, Red on bottom means rotate 90 degrees counterclockwise.
    - Replace each pattern by the transformed version centered on its original tight bounding box, and clear each instruction region to Black.

    Key insight:
    - The non-Red instruction color chooses the target pattern, while the domino orientation and the Red pixel position jointly specify which geometric transform to apply.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    sep_rows, row_colors = _full_line_indices(g, axis=0)
    sep_cols, col_colors = _full_line_indices(g, axis=1)
    sep_colors = row_colors | col_colors
    assert sep_colors == {GREY}

    row_bands = _bands(h, sep_rows) if sep_rows else [(0, h)]
    col_bands = _bands(w, sep_cols) if sep_cols else [(0, w)]

    regions: list[dict[str, object]] = []
    for ri, (r0, r1) in enumerate(row_bands):
        for ci, (c0, c1) in enumerate(col_bands):
            regions.append(_classify_region(g, (r0, r1, c0, c1), (ri, ci)))

    patterns: dict[int, dict[str, object]] = {}
    instructions: dict[int, dict[str, object]] = {}
    for region in regions:
        kind = region["kind"]
        if kind == "pattern":
            color = int(region["color"])
            assert color not in patterns
            patterns[color] = region
        elif kind == "instruction":
            color = int(region["color"])
            assert color not in instructions
            instructions[color] = region

    assert patterns and patterns.keys() == instructions.keys()

    out = g.copy()
    for color in patterns:
        pat = patterns[color]
        ins = instructions[color]

        pr, pc = pat["cell"]
        ir, ic = ins["cell"]
        assert abs(pr - ir) + abs(pc - ic) == 1

        ir0, ir1, ic0, ic1 = ins["region"]
        out[ir0:ir1, ic0:ic1] = BLACK

        rr0, rr1, cc0, cc1 = pat["region"]
        out[rr0:rr1, cc0:cc1] = BLACK

        mask = np.asarray(pat["mask"], dtype=bool)
        transformed = _apply_transform(mask, str(ins["transform"]))
        base_r, base_c = pat["bbox_top_left"]
        old_h, old_w = mask.shape
        new_h, new_w = transformed.shape
        dest_r = int(base_r) + (old_h - new_h) // 2
        dest_c = int(base_c) + (old_w - new_w) // 2
        assert rr0 <= dest_r and cc0 <= dest_c
        assert dest_r + new_h <= rr1 and dest_c + new_w <= cc1
        out[dest_r : dest_r + new_h, dest_c : dest_c + new_w][transformed] = color

    return out


def _random_lengths(count: int, base: int, max_extra: int, extra_budget: int) -> list[int]:
    extras = [0] * count
    remaining = int(_RNG.integers(0, extra_budget + 1))
    while remaining > 0 and any(extra < max_extra for extra in extras):
        idxs = [i for i, extra in enumerate(extras) if extra < max_extra]
        idx = idxs[int(_RNG.integers(0, len(idxs)))]
        extras[idx] += 1
        remaining -= 1
    return [base + extra for extra in extras]


def _region_bands(lengths: list[int]) -> tuple[list[tuple[int, int]], list[int], int]:
    bands: list[tuple[int, int]] = []
    separators: list[int] = []
    start = 0
    for i, length in enumerate(lengths):
        bands.append((start, start + int(length)))
        start += int(length)
        if i < len(lengths) - 1:
            separators.append(start)
            start += 1
    return bands, separators, start


def _grid_edges(rows: int, cols: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r in range(rows):
        for c in range(cols):
            if r + 1 < rows:
                edges.append(((r, c), (r + 1, c)))
            if c + 1 < cols:
                edges.append(((r, c), (r, c + 1)))
    return edges


def _choose_pairs(rows: int, cols: int, pair_count: int) -> list[tuple[tuple[int, int], tuple[int, int]]] | None:
    edges = _grid_edges(rows, cols)

    def backtrack(
        chosen: list[tuple[tuple[int, int], tuple[int, int]]],
        used: set[tuple[int, int]],
    ) -> list[tuple[tuple[int, int], tuple[int, int]]] | None:
        if len(chosen) == pair_count:
            return chosen.copy()
        candidates = [edge for edge in edges if edge[0] not in used and edge[1] not in used]
        if len(candidates) < pair_count - len(chosen):
            return None
        order = list(_RNG.permutation(len(candidates)))
        for idx in order:
            edge = candidates[int(idx)]
            used.update(edge)
            chosen.append(edge)
            result = backtrack(chosen, used)
            if result is not None:
                return result
            chosen.pop()
            used.remove(edge[0])
            used.remove(edge[1])
        return None

    return backtrack([], set())


def _paint_mask(region: np.ndarray, mask: np.ndarray, color: int, top: int, left: int) -> None:
    h, w = mask.shape
    region[top : top + h, left : left + w][mask] = int(color)


def _connect_points(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> None:
    r, c = start
    er, ec = end
    if bool(_RNG.integers(0, 2)):
        while r != er:
            mask[r, c] = True
            r += 1 if er > r else -1
        while c != ec:
            mask[r, c] = True
            c += 1 if ec > c else -1
    else:
        while c != ec:
            mask[r, c] = True
            c += 1 if ec > c else -1
        while r != er:
            mask[r, c] = True
            r += 1 if er > r else -1
    mask[er, ec] = True


def _random_pattern(side: int) -> np.ndarray | None:
    for _ in range(250):
        mask = np.zeros((side, side), dtype=bool)
        anchors = [
            (0, int(_RNG.integers(0, side))),
            (side - 1, int(_RNG.integers(0, side))),
            (int(_RNG.integers(0, side)), 0),
            (int(_RNG.integers(0, side)), side - 1),
        ]
        uniq_anchors: list[tuple[int, int]] = []
        for anchor in anchors:
            if anchor not in uniq_anchors:
                uniq_anchors.append(anchor)

        mask[uniq_anchors[0]] = True
        frontier = [uniq_anchors[0]]
        for anchor in uniq_anchors[1:]:
            start = frontier[int(_RNG.integers(0, len(frontier)))]
            _connect_points(mask, start, anchor)
            frontier = [tuple(x) for x in np.argwhere(mask)]

        base_area = int(mask.sum())
        if base_area >= side * side:
            continue
        max_area = min(side * side - 1, base_area + side + 3)
        target_area = int(_RNG.integers(base_area, max_area + 1))
        while int(mask.sum()) < target_area:
            boundary: set[tuple[int, int]] = set()
            for r, c in np.argwhere(mask):
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = int(r) + dr, int(c) + dc
                    if 0 <= nr < side and 0 <= nc < side and not mask[nr, nc]:
                        boundary.add((nr, nc))
            if not boundary:
                break
            pick = list(boundary)[int(_RNG.integers(0, len(boundary)))]
            mask[pick] = True

        if not _connected(mask):
            continue
        coords = np.argwhere(mask)
        if (
            int(coords[:, 0].min()) != 0
            or int(coords[:, 0].max()) != side - 1
            or int(coords[:, 1].min()) != 0
            or int(coords[:, 1].max()) != side - 1
        ):
            continue
        variants = _all_dihedral(mask)
        keys = {(v.shape, v.tobytes()) for v in variants}
        if len(keys) != 8:
            continue
        return mask
    return None


def _place_instruction(region: np.ndarray, color: int, transform: str) -> None:
    h, w = region.shape
    if transform in ("hflip", "vflip"):
        row = int(_RNG.integers(0, h))
        col = int(_RNG.integers(0, w - 1))
        if transform == "hflip":
            region[row, col] = RED
            region[row, col + 1] = int(color)
        else:
            region[row, col] = int(color)
            region[row, col + 1] = RED
    else:
        row = int(_RNG.integers(0, h - 1))
        col = int(_RNG.integers(0, w))
        if transform == "cw":
            region[row, col] = RED
            region[row + 1, col] = int(color)
        else:
            region[row, col] = int(color)
            region[row + 1, col] = RED


def _gen_attempt() -> np.ndarray | None:
    rows = cols = 4
    pair_count = int(_RNG.integers(4, 7))
    pairs = _choose_pairs(rows, cols, pair_count)
    if pairs is None:
        return None

    row_lengths = _random_lengths(rows, base=5, max_extra=2, extra_budget=7)
    col_lengths = _random_lengths(cols, base=5, max_extra=2, extra_budget=7)
    row_bands, sep_rows, height = _region_bands(row_lengths)
    col_bands, sep_cols, width = _region_bands(col_lengths)
    if max(height, width) > 30:
        return None

    grid = np.full((height, width), BLACK, dtype=int)
    if sep_rows:
        grid[sep_rows, :] = GREY
    if sep_cols:
        grid[:, sep_cols] = GREY

    extra_ops = pair_count - len(TRANSFORMS)
    ops = list(TRANSFORMS)
    if extra_ops > 0:
        ops.extend([str(x) for x in _RNG.choice(np.array(TRANSFORMS), size=extra_ops, replace=True)])
    ops = [ops[int(i)] for i in _RNG.permutation(len(ops))]

    colors = [int(x) for x in _RNG.choice(np.array(COLOR_POOL), size=pair_count, replace=False)]

    for idx, ((a, b), color, transform) in enumerate(zip(pairs, colors, ops)):
        pattern_cell, instruction_cell = (a, b) if bool(_RNG.integers(0, 2)) else (b, a)

        pr0, pr1 = row_bands[pattern_cell[0]]
        pc0, pc1 = col_bands[pattern_cell[1]]
        ir0, ir1 = row_bands[instruction_cell[0]]
        ic0, ic1 = col_bands[instruction_cell[1]]

        pattern_region = grid[pr0:pr1, pc0:pc1]
        instruction_region = grid[ir0:ir1, ic0:ic1]
        if np.any(pattern_region != BLACK) or np.any(instruction_region != BLACK):
            return None

        side = int(_RNG.choice(np.array([3, 4, 5])))
        side = min(side, pattern_region.shape[0], pattern_region.shape[1])
        if side < 3:
            return None
        mask = _random_pattern(side)
        if mask is None:
            return None
        transformed = _apply_transform(mask, transform)
        if np.array_equal(mask, transformed):
            return None

        top = int(_RNG.integers(0, pattern_region.shape[0] - side + 1))
        left = int(_RNG.integers(0, pattern_region.shape[1] - side + 1))
        _paint_mask(pattern_region, mask, color, top, left)
        _place_instruction(instruction_region, color, transform)

    try:
        out = main(grid)
    except AssertionError:
        return None

    changed = int(np.sum(grid != out))
    if changed < 8:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
