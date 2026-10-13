import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Generator palette and layout bounds.
BG_COLOR = 0
SEPARATOR_COLOR = 5
OBJECT_COLORS = tuple(color for color in range(10) if color not in {BG_COLOR, SEPARATOR_COLOR})

GRID_MIN = 20
GRID_MAX = 26
PART_COUNT_CHOICES = (3, 4)
REGION_MIN_3, REGION_MAX_3 = 5, 8
REGION_MIN_4, REGION_MAX_4 = 5, 6
SHAPE_RADIUS = 2
REGION_RETRIES = 200
GEN_RETRIES = 400


def _mode_color(values: np.ndarray) -> int:
    flat = np.asarray(values, dtype=int).ravel()
    counts = np.bincount(flat, minlength=int(flat.max()) + 1 if flat.size else 1)
    return int(np.argmax(counts))


def _find_separator_lines(grid: np.ndarray, bg: int) -> tuple[int, list[int], list[int]]:
    h, w = grid.shape
    candidates = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        rows = [r for r in range(h) if np.all(grid[r, :] == color)]
        cols = [c for c in range(w) if np.all(grid[:, c] == color)]
        if rows or cols:
            candidates.append((len(rows) + len(cols), color, rows, cols))

    assert candidates, "expected a separator color with full rows or columns"
    candidates.sort(reverse=True)
    score, color, rows, cols = candidates[0]
    assert rows and cols, "separator must include both horizontal and vertical lines"
    if len(candidates) > 1:
        assert score > candidates[1][0], "separator color must be uniquely identifiable"
    return color, rows, cols


def _spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    spans = []
    start = 0
    for idx in sorted(separators):
        if start <= idx - 1:
            spans.append((start, idx - 1))
        start = idx + 1
    if start <= length - 1:
        spans.append((start, length - 1))
    assert spans, "expected at least one non-separator span"
    return spans


def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _region_components(region: np.ndarray, bg: int) -> list[dict[str, object]]:
    comps: list[dict[str, object]] = []
    for color in map(int, np.unique(region)):
        if color == bg:
            continue
        for coords in _mask_components(region == color):
            comps.append({"color": color, "coords": coords})
    return comps


def _touches_pattern(seed: tuple[int, int], pattern: set[tuple[int, int]]) -> bool:
    r, c = seed
    return any((r + dr, c + dc) in pattern for dr, dc in DIR4)


def _rotation_turns(anchor: tuple[int, int], orient: tuple[int, int]) -> int:
    ar, ac = anchor
    br, bc = orient
    if bc == ac and br < ar:
        return 0
    if br == ar and bc > ac:
        return 1
    if bc == ac and br > ar:
        return 2
    if br == ar and bc < ac:
        return 3
    raise AssertionError("orientation seed must share a row or column with the anchor")


def _rotate_relative(coord: tuple[int, int], turns: int) -> tuple[int, int]:
    dr, dc = coord
    if turns == 0:
        return dr, dc
    if turns == 1:
        return dc, -dr
    if turns == 2:
        return -dr, -dc
    if turns == 3:
        return -dc, dr
    raise AssertionError("invalid turn count")


def _transform_region(region: np.ndarray, bg: int) -> np.ndarray:
    comps = _region_components(region, bg)
    assert len(comps) == 3, "each region must contain two seeds and one pattern"

    singletons = [comp for comp in comps if len(comp["coords"]) == 1]
    patterns = [comp for comp in comps if 3 <= len(comp["coords"]) <= 5]
    assert len(singletons) == 2 and len(patterns) == 1, "unexpected region contents"

    pattern = patterns[0]
    pattern_cells = {tuple(map(int, coord)) for coord in pattern["coords"]}
    anchor_candidates = [
        comp for comp in singletons if _touches_pattern(tuple(map(int, comp["coords"][0])), pattern_cells)
    ]
    assert len(anchor_candidates) == 1, "expected exactly one adjacent anchor seed"

    anchor_comp = anchor_candidates[0]
    orient_comp = singletons[0] if singletons[1] is anchor_comp else singletons[1]
    anchor = tuple(map(int, anchor_comp["coords"][0]))
    orient = tuple(map(int, orient_comp["coords"][0]))
    turns = _rotation_turns(anchor, orient)

    out = region.copy()
    for r, c in pattern_cells:
        out[r, c] = bg

    rotated_cells = []
    for r, c in pattern_cells:
        rel = (r - anchor[0], c - anchor[1])
        rr, rc = _rotate_relative(rel, turns)
        nr, nc = anchor[0] + rr, anchor[1] + rc
        assert 0 <= nr < region.shape[0] and 0 <= nc < region.shape[1], "rotation left the region bounds"
        assert (nr, nc) != anchor and (nr, nc) != orient, "rotated pattern cannot overwrite seeds"
        rotated_cells.append((nr, nc))

    assert len(set(rotated_cells)) == len(pattern_cells), "rotation must preserve distinct pattern cells"
    for nr, nc in rotated_cells:
        assert out[nr, nc] == bg, "rotated pattern cannot collide with non-pattern cells"
        out[nr, nc] = int(pattern["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each region's small pattern around its adjacent seed using the other seed as a direction marker.

    Input structure:
    - The background is a dominant blank color.
    - Full single-cell separator rows and columns of one color partition the grid into rectangular subregions.
    - Each subregion contains exactly two singleton seeds of distinct colors and one 4-connected 3-5 cell pattern in a third color.
    - Exactly one seed is 4-neighbor adjacent to the pattern; the other seed lies somewhere on the same row or column as that adjacent seed.

    Rule:
    - In every subregion, identify the adjacent seed as the anchor.
    - Read the other seed's direction relative to the anchor: above means 0 degrees, right means 90 degrees clockwise, below means 180 degrees, and left means 270 degrees clockwise.
    - Rotate the pattern around the anchor by that amount, keep both seeds unchanged, erase any old pattern cells not covered after rotation, and leave the separator lines untouched.

    Key insight:
    - The task has two coupled steps inside each partition: adjacency picks which seed is the pivot, and the other seed's row-or-column direction encodes the rotation angle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _mode_color(g)
    _, sep_rows, sep_cols = _find_separator_lines(g, bg)
    row_spans = _spans(g.shape[0], sep_rows)
    col_spans = _spans(g.shape[1], sep_cols)

    out = g.copy()
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            region = g[r0 : r1 + 1, c0 : c1 + 1]
            out[r0 : r1 + 1, c0 : c1 + 1] = _transform_region(region, bg)
    return out


def _partition_bounds(part_count: int) -> tuple[int, int, int]:
    if part_count == 3:
        return GRID_MIN, REGION_MIN_3, REGION_MAX_3
    return 23, REGION_MIN_4, REGION_MAX_4


def _random_partition(total: int, parts: int, lo: int, hi: int) -> list[int] | None:
    if total < parts * lo or total > parts * hi:
        return None
    sizes = []
    remaining = total
    for index in range(parts):
        left = parts - index - 1
        low = max(lo, remaining - left * hi)
        high = min(hi, remaining - left * lo)
        if low > high:
            return None
        sizes.append(random.randint(low, high))
        remaining -= sizes[-1]
    random.shuffle(sizes)
    return sizes


def _layout_from_lengths(lengths: list[int]) -> tuple[list[tuple[int, int]], list[int], int]:
    spans = []
    separators = []
    cursor = 0
    for index, size in enumerate(lengths):
        spans.append((cursor, cursor + size - 1))
        cursor += size
        if index != len(lengths) - 1:
            separators.append(cursor)
            cursor += 1
    return spans, separators, cursor


def _shape_frontier(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    neighbors = []
    for dr, dc in DIR4:
        nr, nc = r + dr, c + dc
        if max(abs(nr), abs(nc)) <= SHAPE_RADIUS and (nr, nc) != (0, 0):
            neighbors.append((nr, nc))
    return neighbors


def _random_shape(size: int) -> set[tuple[int, int]] | None:
    for _ in range(REGION_RETRIES):
        cells = {random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])}
        while len(cells) < size:
            frontier = set()
            for cell in cells:
                frontier.update(_shape_frontier(cell))
            frontier.difference_update(cells)
            if not frontier:
                break
            cells.add(random.choice(sorted(frontier)))
        if len(cells) == size:
            return cells
    return None


def _orientation_offset(turns: int, distance: int) -> tuple[int, int]:
    if turns == 0:
        return -distance, 0
    if turns == 1:
        return 0, distance
    if turns == 2:
        return distance, 0
    return 0, -distance


def _adjacent_to_shape(seed_rel: tuple[int, int], shape: set[tuple[int, int]]) -> bool:
    r, c = seed_rel
    return any(abs(r - sr) + abs(c - sc) == 1 for sr, sc in shape)


def _balanced_turns(count: int) -> list[int]:
    turns = list(range(4)) * (count // 4)
    turns.extend(random.sample(range(4), count % 4))
    random.shuffle(turns)
    return turns


def _make_region(height: int, width: int, turns: int) -> np.ndarray | None:
    for _ in range(REGION_RETRIES):
        shape = _random_shape(random.randint(3, 5))
        if shape is None:
            continue
        rotated = {_rotate_relative(cell, turns) for cell in shape}
        if turns != 0 and rotated == shape:
            continue

        distances = [2, 3]
        random.shuffle(distances)
        for distance in distances:
            orient_rel = _orientation_offset(turns, distance)
            if orient_rel in shape or orient_rel in rotated:
                continue
            if _adjacent_to_shape(orient_rel, shape):
                continue

            rels = set(shape) | set(rotated) | {orient_rel, (0, 0)}
            row_min = -min(r for r, _ in rels)
            row_max = height - 1 - max(r for r, _ in rels)
            col_min = -min(c for _, c in rels)
            col_max = width - 1 - max(c for _, c in rels)
            if row_min > row_max or col_min > col_max:
                continue

            row_choices = list(range(row_min, row_max + 1))
            col_choices = list(range(col_min, col_max + 1))
            inner_rows = [r for r in row_choices if 0 < r < height - 1]
            inner_cols = [c for c in col_choices if 0 < c < width - 1]
            if inner_rows:
                row_choices = inner_rows
            if inner_cols:
                col_choices = inner_cols

            anchor = (random.choice(row_choices), random.choice(col_choices))
            anchor_r, anchor_c = anchor
            region = np.full((height, width), BG_COLOR, dtype=int)
            pattern_color, anchor_color, orient_color = random.sample(OBJECT_COLORS, 3)

            for dr, dc in shape:
                region[anchor_r + dr, anchor_c + dc] = pattern_color
            region[anchor_r, anchor_c] = anchor_color
            region[anchor_r + orient_rel[0], anchor_c + orient_rel[1]] = orient_color
            return region
    return None


def _gen_attempt() -> np.ndarray | None:
    row_parts = random.choice(PART_COUNT_CHOICES)
    col_parts = random.choice(PART_COUNT_CHOICES)

    row_min_total, row_lo, row_hi = _partition_bounds(row_parts)
    col_min_total, col_lo, col_hi = _partition_bounds(col_parts)
    height = random.randint(row_min_total, GRID_MAX)
    width = random.randint(col_min_total, GRID_MAX)

    row_lengths = _random_partition(height - (row_parts - 1), row_parts, row_lo, row_hi)
    col_lengths = _random_partition(width - (col_parts - 1), col_parts, col_lo, col_hi)
    if row_lengths is None or col_lengths is None:
        return None

    row_spans, sep_rows, built_h = _layout_from_lengths(row_lengths)
    col_spans, sep_cols, built_w = _layout_from_lengths(col_lengths)
    assert built_h == height and built_w == width

    grid = np.full((height, width), BG_COLOR, dtype=int)
    if sep_rows:
        grid[sep_rows, :] = SEPARATOR_COLOR
    if sep_cols:
        grid[:, sep_cols] = SEPARATOR_COLOR

    turns = _balanced_turns(len(row_spans) * len(col_spans))
    turn_index = 0
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            patch = _make_region(r1 - r0 + 1, c1 - c0 + 1, turns[turn_index])
            turn_index += 1
            if patch is None:
                return None
            grid[r0 : r1 + 1, c0 : c1 + 1] = patch

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
