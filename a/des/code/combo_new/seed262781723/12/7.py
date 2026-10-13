import numpy as np


BLACK = 0
GREY = 5
AZURE = 8
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
OBJECT_COLORS = [1, 2, 3, 4, 6, 7, 9]

_RNG = np.random.default_rng()
_GENERATION_RETRIES = 300
_SHAPE_RETRIES = 200


def _connected_components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        neighbors = DIRS4
    else:
        neighbors = (
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),            (0, 1),
            (1, -1),  (1, 0),   (1, 1),
        )

    components: list[list[tuple[int, int]]] = []
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
                for dr, dc in neighbors:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)
    return components


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_divider_run(grid: np.ndarray) -> tuple[int, int]:
    h, w = grid.shape
    full_grey_cols = [c for c in range(w) if np.all(grid[:, c] == GREY)]
    assert full_grey_cols, "expected at least one full grey divider column"

    runs: list[tuple[int, int]] = []
    start = full_grey_cols[0]
    prev = start
    for col in full_grey_cols[1:]:
        if col == prev + 1:
            prev = col
            continue
        runs.append((start, prev))
        start = prev = col
    runs.append((start, prev))

    candidates = []
    for left, right in runs:
        has_left_instruction = left > 0 and np.any(grid[:, :left] == AZURE)
        right_region = grid[:, right + 1 :]
        has_right_object = right < w - 1 and np.any((right_region != BLACK) & (right_region != GREY))
        if has_left_instruction and has_right_object:
            candidates.append((left, right))

    assert len(candidates) == 1, "expected a unique grey divider separating instructions and objects"
    return candidates[0]


def _extract_instruction_components(grid: np.ndarray, divider_left: int) -> list[dict]:
    components = _connected_components(grid[:, :divider_left] == AZURE, connectivity=4)
    extracted = []
    for coords in components:
        abs_coords = [(r, c) for r, c in coords]
        r0, c0, r1, c1 = _bbox(abs_coords)
        extracted.append(
            {
                "coords": abs_coords,
                "count": len(abs_coords),
                "r0": r0,
                "r1": r1,
                "c0": c0,
                "c1": c1,
            }
        )
    extracted.sort(key=lambda item: (item["r0"], item["c0"]))
    return extracted


def _extract_object_components(grid: np.ndarray, divider_right: int) -> list[dict]:
    right = grid[:, divider_right + 1 :]
    mask = (right != BLACK) & (right != GREY)
    components = _connected_components(mask, connectivity=4)
    extracted = []
    for coords in components:
        abs_coords = [(r, c + divider_right + 1) for r, c in coords]
        r0, c0, r1, c1 = _bbox(abs_coords)
        colors = {int(grid[r, c]) for r, c in abs_coords}
        assert len(colors) == 1, "each right-side object must be monochrome"
        extracted.append(
            {
                "coords": abs_coords,
                "color": colors.pop(),
                "r0": r0,
                "r1": r1,
                "c0": c0,
                "c1": c1,
            }
        )
    extracted.sort(key=lambda item: (item["r0"], item["c0"]))
    return extracted


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 < b0 or b1 < a0)


def _rotate_doubled(dr2: int, dc2: int, turns: int) -> tuple[int, int]:
    turns %= 4
    for _ in range(turns):
        dr2, dc2 = dc2, -dr2
    return dr2, dc2


def _rotate_object_coords(coords: list[tuple[int, int]], turns: int) -> list[tuple[int, int]]:
    turns %= 4
    if turns == 0:
        return list(coords)

    r0, c0, r1, c1 = _bbox(coords)
    sr = r0 + r1
    sc = c0 + c1
    rotated = []
    for r, c in coords:
        dr2 = 2 * r - sr
        dc2 = 2 * c - sc
        nr2, nc2 = _rotate_doubled(dr2, dc2, turns)
        nr_num = sr + nr2
        nc_num = sc + nc2
        assert nr_num % 2 == 0 and nc_num % 2 == 0, "rotation must land on cell centers"
        rotated.append((nr_num // 2, nc_num // 2))

    assert len(set(rotated)) == len(coords), "rotation must preserve cell count"
    return rotated


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each right-side object by the row-aligned azure instruction count, then erase the instructions.

    Input structure:
    - The background is black.
    - A full grey vertical divider splits the grid into a left instruction area and a right object area.
    - The left area contains several disconnected azure 4-connected clusters, with 1, 2, or 3 cells each.
    - The right area contains several disconnected monochrome objects, and each object's row band matches exactly one azure cluster.

    Rule:
    - Pair every right-side object with the unique azure cluster whose row span overlaps the object's row span.
    - Read the azure cluster size N, where N in {1, 2, 3} encodes a clockwise rotation of N x 90 degrees.
    - Rotate the paired object around the center of its own bounding box and redraw it in place.
    - Clear all azure instruction pixels to black and keep the grey divider unchanged.

    Key insight:
    - The scattered azure clusters are not output objects; their cell counts are rotation commands for the aligned objects across the divider.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    divider_left, divider_right = _find_divider_run(g)
    instructions = _extract_instruction_components(g, divider_left)
    objects = _extract_object_components(g, divider_right)

    assert instructions, "expected at least one instruction cluster"
    assert len(instructions) == len(objects), "instruction/object counts must match"

    instruction_for_object: list[dict] = []
    used_instruction_indices: set[int] = set()
    for obj in objects:
        matches = [
            idx
            for idx, ins in enumerate(instructions)
            if _intervals_overlap(obj["r0"], obj["r1"], ins["r0"], ins["r1"])
        ]
        assert len(matches) == 1, "each object must match exactly one instruction cluster by rows"
        match_idx = matches[0]
        assert match_idx not in used_instruction_indices, "row pairing must be one-to-one"
        used_instruction_indices.add(match_idx)
        instruction_for_object.append(instructions[match_idx])

    out = g.copy()
    for ins in instructions:
        assert ins["count"] in {1, 2, 3}
        for r, c in ins["coords"]:
            out[r, c] = BLACK

    for obj in objects:
        for r, c in obj["coords"]:
            out[r, c] = BLACK

    h, w = g.shape
    for obj, ins in zip(objects, instruction_for_object):
        rotated = _rotate_object_coords(obj["coords"], ins["count"])
        for r, c in rotated:
            assert 0 <= r < h and 0 <= c < w, "rotated object must stay inside the grid"
            assert divider_left > c or c > divider_right, "rotated object cannot land on the divider"
            assert out[r, c] == BLACK, "rotated objects must remain non-overlapping"
            out[r, c] = obj["color"]

    return out


def _touches_all_edges(mask: np.ndarray) -> bool:
    return (
        bool(mask[0, :].any())
        and bool(mask[-1, :].any())
        and bool(mask[:, 0].any())
        and bool(mask[:, -1].any())
    )


def _random_connected_shape(size: int) -> np.ndarray:
    area_low = size + 1
    area_high = size * size - 1
    for _ in range(_SHAPE_RETRIES):
        target_area = int(_RNG.integers(area_low, area_high + 1))
        start = (int(_RNG.integers(0, size)), int(_RNG.integers(0, size)))
        cells = {start}

        while len(cells) < target_area:
            frontier = set()
            for r, c in cells:
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < size and 0 <= nc < size and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break

            frontier_list = list(frontier)
            weights = np.array(
                [
                    1.0
                    + 0.7 * (r in (0, size - 1))
                    + 0.7 * (c in (0, size - 1))
                    for r, c in frontier_list
                ],
                dtype=float,
            )
            weights /= weights.sum()
            pick_idx = int(_RNG.choice(len(frontier_list), p=weights))
            cells.add(frontier_list[pick_idx])

        if len(cells) != target_area:
            continue

        mask = np.zeros((size, size), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        if not _touches_all_edges(mask):
            continue
        if any(np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3)):
            continue
        return mask

    raise RuntimeError("failed to create an asymmetric connected shape")


def _instruction_shape(count: int) -> np.ndarray:
    if count == 1:
        return np.array([[1]], dtype=bool)
    if count == 2:
        options = [
            np.array([[1, 1]], dtype=bool),
            np.array([[1], [1]], dtype=bool),
        ]
        return options[int(_RNG.integers(0, len(options)))]

    options = [
        np.array([[1, 1, 1]], dtype=bool),
        np.array([[1], [1], [1]], dtype=bool),
        np.array([[1, 0], [1, 1]], dtype=bool),
        np.array([[0, 1], [1, 1]], dtype=bool),
        np.array([[1, 1], [1, 0]], dtype=bool),
        np.array([[1, 1], [0, 1]], dtype=bool),
    ]
    return options[int(_RNG.integers(0, len(options)))]


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    sub = grid[top : top + h, left : left + w]
    sub[mask] = color


def _sample_instruction_counts(count: int) -> list[int]:
    result: list[int] = []
    while len(result) < count:
        result.extend(int(x) for x in _RNG.permutation([1, 2, 3]))
    return result[:count]


def _validate_generated(grid: np.ndarray) -> None:
    divider_left, divider_right = _find_divider_run(grid)
    instructions = _extract_instruction_components(grid, divider_left)
    objects = _extract_object_components(grid, divider_right)

    assert len(instructions) == len(objects) >= 2
    used: set[int] = set()
    for obj in objects:
        matches = [
            idx
            for idx, ins in enumerate(instructions)
            if _intervals_overlap(obj["r0"], obj["r1"], ins["r0"], ins["r1"])
        ]
        assert len(matches) == 1
        used.add(matches[0])
        rotated = _rotate_object_coords(obj["coords"], instructions[matches[0]]["count"])
        assert set(rotated) != set(obj["coords"]), "instruction should visibly rotate the object"
    assert len(used) == len(instructions)

    out = main(grid)
    assert out.shape == grid.shape
    for ins in instructions:
        for r, c in ins["coords"]:
            assert out[r, c] == BLACK
    assert np.any(out != grid)


def _gen_attempt() -> np.ndarray | None:
    num_objects = int(_RNG.integers(2, 5))
    sizes = _RNG.integers(3, 6, size=num_objects)

    top_margin = int(_RNG.integers(1, 4))
    bottom_margin = int(_RNG.integers(1, 4))
    gaps = _RNG.integers(1, 4, size=max(0, num_objects - 1))
    height = top_margin + int(sizes.sum()) + int(gaps.sum()) + bottom_margin
    if height > 30:
        return None

    max_size = int(sizes.max())
    left_width = int(_RNG.integers(6, 10))
    min_right_width = max_size + 4
    max_right_width = min(30 - left_width - 1, 16)
    if min_right_width > max_right_width:
        return None
    right_width = int(_RNG.integers(min_right_width, max_right_width + 1))
    width = left_width + 1 + right_width
    if width > 30:
        return None

    grid = np.full((height, width), BLACK, dtype=int)
    divider_col = left_width
    grid[:, divider_col] = GREY

    colors = list(_RNG.choice(OBJECT_COLORS, size=num_objects, replace=False))
    counts = _sample_instruction_counts(num_objects)

    row = top_margin
    for idx in range(num_objects):
        size = int(sizes[idx])
        color = int(colors[idx])
        count = int(counts[idx])

        shape = _random_connected_shape(size)
        shape = np.rot90(shape, int(_RNG.integers(0, 4)))
        instr = _instruction_shape(count)

        obj_col = int(_RNG.integers(1, right_width - size))
        instr_h, instr_w = instr.shape
        max_instr_col = left_width - instr_w - 1
        if max_instr_col < 1:
            return None
        instr_col = int(_RNG.integers(1, max_instr_col + 1))
        instr_row_offset = int(_RNG.integers(0, size - instr_h + 1))

        _paint_mask(grid, shape, row, divider_col + 1 + obj_col, color)
        _paint_mask(grid, instr, row + instr_row_offset, instr_col, AZURE)

        row += size
        if idx < num_objects - 1:
            row += int(gaps[idx])

    try:
        _validate_generated(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
