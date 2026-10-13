import random

import numpy as np


BG = 0
SHAPE_COLOR = 5
STRIPE_PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

MIN_SEG_LEN = 2
MAX_SEG_LEN = 5
MIN_SEGMENTS = 3
MAX_SEGMENTS = 6

MIN_LINE_LEN = 10
MAX_LINE_LEN = 18
MIN_SIDE_DEPTH = 2
MAX_SIDE_DEPTH = 7
MAX_SHAPES = 4
RETRY_LIMIT = 600


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _non_bg_colors(arr: np.ndarray, bg: int) -> set[int]:
    values = arr[arr != bg]
    return {int(v) for v in values.flat}


def _run_lengths(line: np.ndarray) -> tuple[list[int], list[int]]:
    colors: list[int] = []
    lengths: list[int] = []
    start = 0
    last = int(line[0])
    for i in range(1, len(line) + 1):
        if i == len(line) or int(line[i]) != last:
            colors.append(last)
            lengths.append(i - start)
            if i < len(line):
                start = i
                last = int(line[i])
    return colors, lengths


def _is_valid_stripe_line(line: np.ndarray, bg: int, shape_color: int | None = None) -> bool:
    if line.size < MIN_SEGMENTS * MIN_SEG_LEN:
        return False
    if np.any(line == bg):
        return False
    colors, lengths = _run_lengths(line)
    if len(colors) < 2:
        return False
    if len(colors) < MIN_SEGMENTS or len(colors) > MAX_SEGMENTS:
        return False
    if len(set(colors)) != len(colors):
        return False
    if any(length < MIN_SEG_LEN or length > MAX_SEG_LEN for length in lengths):
        return False
    if shape_color is not None and shape_color in colors:
        return False
    return True


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)
    return components


def _mirror_coord(r: int, c: int, orientation: str, line_index: int) -> tuple[int, int]:
    if orientation == "h":
        return 2 * line_index - r, c
    return r, 2 * line_index - c


def _validate_candidate(
    grid: np.ndarray,
    bg: int,
    orientation: str,
    line_index: int,
    side: str,
    shape_color: int,
) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    line = grid[line_index, :] if orientation == "h" else grid[:, line_index]
    assert _is_valid_stripe_line(line, bg, shape_color)

    source_mask = np.zeros_like(grid, dtype=bool)
    if orientation == "h":
        if side == "top":
            source_slice = np.s_[:line_index, :]
            target_slice = np.s_[line_index + 1 :, :]
            touch_row = line_index - 1
            source_mask[:line_index, :] = grid[:line_index, :] == shape_color
        else:
            source_slice = np.s_[line_index + 1 :, :]
            target_slice = np.s_[:line_index, :]
            touch_row = line_index + 1
            source_mask[line_index + 1 :, :] = grid[line_index + 1 :, :] == shape_color
    else:
        if side == "left":
            source_slice = np.s_[:, :line_index]
            target_slice = np.s_[:, line_index + 1 :]
            touch_col = line_index - 1
            source_mask[:, :line_index] = grid[:, :line_index] == shape_color
        else:
            source_slice = np.s_[:, line_index + 1 :]
            target_slice = np.s_[:, :line_index]
            touch_col = line_index + 1
            source_mask[:, line_index + 1 :] = grid[:, line_index + 1 :] == shape_color

    assert _non_bg_colors(grid[source_slice], bg) == {shape_color}
    assert not _non_bg_colors(grid[target_slice], bg)

    components = _connected_components(source_mask)
    assert components
    for comp in components:
        if orientation == "h":
            assert any(r == touch_row for r, _c in comp)
        else:
            assert any(c == touch_col for _r, c in comp)
        for r, c in comp:
            rr, cc = _mirror_coord(r, c, orientation, line_index)
            assert 0 <= rr < h and 0 <= cc < w
            assert grid[rr, cc] == bg
    return components


def _find_structure(
    grid: np.ndarray,
) -> tuple[str, int, str, int, list[list[tuple[int, int]]]]:
    bg = _dominant_color(grid)
    h, w = grid.shape
    candidates: list[tuple[str, int, str, int, list[list[tuple[int, int]]]]] = []

    for r in range(1, h - 1):
        line = grid[r, :]
        if np.any(line == bg):
            continue
        top_colors = _non_bg_colors(grid[:r, :], bg)
        bottom_colors = _non_bg_colors(grid[r + 1 :, :], bg)
        if len(top_colors) == 1 and not bottom_colors:
            shape_color = int(next(iter(top_colors)))
            try:
                comps = _validate_candidate(grid, bg, "h", r, "top", shape_color)
                candidates.append(("h", r, "top", shape_color, comps))
            except AssertionError:
                pass
        if len(bottom_colors) == 1 and not top_colors:
            shape_color = int(next(iter(bottom_colors)))
            try:
                comps = _validate_candidate(grid, bg, "h", r, "bottom", shape_color)
                candidates.append(("h", r, "bottom", shape_color, comps))
            except AssertionError:
                pass

    for c in range(1, w - 1):
        line = grid[:, c]
        if np.any(line == bg):
            continue
        left_colors = _non_bg_colors(grid[:, :c], bg)
        right_colors = _non_bg_colors(grid[:, c + 1 :], bg)
        if len(left_colors) == 1 and not right_colors:
            shape_color = int(next(iter(left_colors)))
            try:
                comps = _validate_candidate(grid, bg, "v", c, "left", shape_color)
                candidates.append(("v", c, "left", shape_color, comps))
            except AssertionError:
                pass
        if len(right_colors) == 1 and not left_colors:
            shape_color = int(next(iter(right_colors)))
            try:
                comps = _validate_candidate(grid, bg, "v", c, "right", shape_color)
                candidates.append(("v", c, "right", shape_color, comps))
            except AssertionError:
                pass

    assert len(candidates) == 1, "expected exactly one mirror-line interpretation"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror grey shapes across a striped line and color each copied pixel by its line projection.

    The input contains one full-width or full-height striped mirror line. Grey
    connected shapes sit on exactly one side of that line, touching it. Each
    shape is mirrored onto the empty opposite side, and every mirrored pixel
    takes the color of the stripe cell it projects onto.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    orientation, line_index, _side, _shape_color, components = _find_structure(g)
    out = g.copy()

    for comp in components:
        for r, c in comp:
            rr, cc = _mirror_coord(r, c, orientation, line_index)
            color = int(g[line_index, cc]) if orientation == "h" else int(g[rr, line_index])
            out[rr, cc] = color

    return out


def _bounded_sum(total: int, parts: int, low: int, high: int) -> list[int] | None:
    out: list[int] = []
    remaining = total
    for i in range(parts):
        min_here = max(low, remaining - high * (parts - i - 1))
        max_here = min(high, remaining - low * (parts - i - 1))
        if min_here > max_here:
            return None
        value = random.randint(min_here, max_here)
        out.append(value)
        remaining -= value
    return out if remaining == 0 else None


def _random_stripe_line(length: int) -> np.ndarray | None:
    segment_options = [
        count
        for count in range(MIN_SEGMENTS, min(MAX_SEGMENTS, len(STRIPE_PALETTE), length // MIN_SEG_LEN) + 1)
        if MIN_SEG_LEN * count <= length <= MAX_SEG_LEN * count
    ]
    if not segment_options:
        return None
    count = random.choice(segment_options)
    lengths = _bounded_sum(length, count, MIN_SEG_LEN, MAX_SEG_LEN)
    if lengths is None:
        return None

    colors = random.sample(list(STRIPE_PALETTE), count)
    line = np.empty(length, dtype=int)
    pos = 0
    for seg_len, color in zip(lengths, colors):
        line[pos : pos + seg_len] = color
        pos += seg_len
    return line


def _random_nonnegative_parts(total: int, parts: int) -> list[int]:
    values = [0] * parts
    for _ in range(total):
        values[random.randrange(parts)] += 1
    return values


def _random_intervals(length: int, count: int) -> list[tuple[int, int]] | None:
    for _ in range(80):
        widths = [random.randint(1, min(6, length - (count - 1))) for _ in range(count)]
        if sum(widths) + (count - 1) > length:
            continue
        extra = length - (sum(widths) + (count - 1))
        gap_extras = _random_nonnegative_parts(extra, count + 1)

        starts: list[tuple[int, int]] = []
        pos = gap_extras[0]
        for i, width in enumerate(widths):
            starts.append((pos, width))
            pos += width
            if i < count - 1:
                pos += 1 + gap_extras[i + 1]
        return starts
    return None


def _random_shape_cells(max_depth: int, width: int) -> set[tuple[int, int]]:
    depth = random.randint(1, max_depth)
    anchor_len = random.randint(1, min(width, 3))
    anchor_start = random.randint(0, width - anchor_len)
    cells = {(0, c) for c in range(anchor_start, anchor_start + anchor_len)}

    max_area = min(depth * width, max(anchor_len, width + depth - 1))
    target_area = random.randint(anchor_len, max_area)
    while len(cells) < target_area:
        frontier = set()
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < depth and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            break
        cells.add(random.choice(tuple(frontier)))
    return cells


def _source_coord(
    orientation: str,
    side: str,
    line_index: int,
    parallel: int,
    distance: int,
) -> tuple[int, int]:
    if orientation == "h":
        row = line_index - distance if side == "top" else line_index + distance
        return row, parallel
    col = line_index - distance if side == "left" else line_index + distance
    return parallel, col


def _manual_output(
    inp: np.ndarray,
    orientation: str,
    line_index: int,
    shape_cells: list[list[tuple[int, int]]],
) -> np.ndarray:
    out = inp.copy()
    for comp in shape_cells:
        for r, c in comp:
            rr, cc = _mirror_coord(r, c, orientation, line_index)
            color = int(inp[line_index, cc]) if orientation == "h" else int(inp[rr, line_index])
            out[rr, cc] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    orientation, side = random.choice(
        (("h", "top"), ("h", "bottom"), ("v", "left"), ("v", "right"))
    )
    source_depth = random.randint(MIN_SIDE_DEPTH, MAX_SIDE_DEPTH)
    target_depth = random.randint(MIN_SIDE_DEPTH, MAX_SIDE_DEPTH)
    usable_depth = min(source_depth, target_depth)
    line_len = random.randint(MIN_LINE_LEN, MAX_LINE_LEN)

    stripe_line = _random_stripe_line(line_len)
    if stripe_line is None:
        return None

    shape_count = random.randint(1, min(MAX_SHAPES, max(1, line_len // 3)))
    intervals = _random_intervals(line_len, shape_count)
    if intervals is None:
        return None

    if orientation == "h":
        height = source_depth + 1 + target_depth
        width = line_len
        line_index = source_depth if side == "top" else target_depth
    else:
        height = line_len
        width = source_depth + 1 + target_depth
        line_index = source_depth if side == "left" else target_depth

    inp = np.full((height, width), BG, dtype=int)
    if orientation == "h":
        inp[line_index, :] = stripe_line
    else:
        inp[:, line_index] = stripe_line

    placed_components: list[list[tuple[int, int]]] = []
    touched_color_sets: list[set[int]] = []
    for start, span in intervals:
        local_cells = _random_shape_cells(usable_depth, span)
        global_cells: list[tuple[int, int]] = []
        projected_colors: set[int] = set()
        for depth0, parallel0 in local_cells:
            parallel = start + parallel0
            distance = depth0 + 1
            r, c = _source_coord(orientation, side, line_index, parallel, distance)
            global_cells.append((r, c))
            projected_colors.add(int(stripe_line[parallel]))
        for r, c in global_cells:
            if inp[r, c] != BG:
                return None
            inp[r, c] = SHAPE_COLOR
        placed_components.append(global_cells)
        touched_color_sets.append(projected_colors)

    if _dominant_color(inp) != BG:
        return None

    if len({color for colors in touched_color_sets for color in colors}) < 2:
        return None
    if max(len(colors) for colors in touched_color_sets) < 2:
        return None

    expected = _manual_output(inp, orientation, line_index, placed_components)
    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-stamp input")
