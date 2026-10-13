import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = tuple(range(10))

GRID_H_RANGE = (14, 26)
GRID_W_RANGE = (14, 26)
COLOR_COUNT_RANGE = (2, 5)
RULER_SEGMENT_MIN = 2
RULER_SEGMENT_MAX = 7
SHAPE_BOX_MAX = 4
LAYOUT_ATTEMPTS = 150
ANCHOR_ATTEMPTS = 80
MAX_GENERATION_ATTEMPTS = 400
OVERLAP_PROBABILITY = 0.5


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = int(cr + dr), int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.asarray(coords, dtype=int)
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _parse_ruler(line: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    arr = np.asarray(line, dtype=int).reshape(-1)
    non_bg = np.flatnonzero(arr != bg)
    assert non_bg.size > 0, "ruler must contain non-background segments"
    expected = np.arange(int(non_bg[0]), int(non_bg[-1]) + 1)
    assert np.array_equal(non_bg, expected), "ruler segments must be tightly packed"

    segments: list[tuple[int, int, int]] = []
    start = int(non_bg[0])
    color = int(arr[start])
    for idx in range(start + 1, int(non_bg[-1]) + 1):
        value = int(arr[idx])
        if value != color:
            segments.append((color, start, idx - start))
            start = idx
            color = value
    segments.append((color, start, int(non_bg[-1]) - start + 1))

    colors = [segment[0] for segment in segments]
    assert len(colors) == len(set(colors)), "each ruler color must appear exactly once"
    return segments


def _extract_object_bboxes(grid: np.ndarray, colors: list[int], bg: int) -> dict[int, tuple[int, int, int, int]]:
    interior = np.asarray(grid, dtype=int).copy()
    interior[0, :] = bg
    interior[:, 0] = bg

    present = {int(color) for color in np.unique(interior) if int(color) != bg}
    assert present == set(colors), "interior colors must match ruler colors"

    bboxes: dict[int, tuple[int, int, int, int]] = {}
    for color in colors:
        comps = _components(interior == int(color))
        assert len(comps) == 1, "each color must form exactly one 4-connected blob"
        bboxes[int(color)] = _bbox(comps[0])
    return bboxes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode widths and heights from the top and left rulers, then replace each
    interior blob with a solid rectangle aligned to the blob bbox's top-left.
    Later colors in top-ruler order overwrite earlier rectangles on overlap.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    top_segments = _parse_ruler(g[0, :], bg)
    left_segments = _parse_ruler(g[:, 0], bg)

    top_order = [color for color, _start, _length in top_segments]
    width_by_color = {int(color): int(length) for color, _start, length in top_segments}
    height_by_color = {int(color): int(length) for color, _start, length in left_segments}

    assert set(width_by_color) == set(height_by_color), "top and left rulers must use the same colors"
    bboxes = _extract_object_bboxes(g, top_order, bg)

    out = g.copy()
    interior_mask = np.ones(g.shape, dtype=bool)
    interior_mask[0, :] = False
    interior_mask[:, 0] = False
    out[interior_mask & (out != bg)] = bg

    h, w = g.shape
    for color in top_order:
        r0, c0, _r1, _c1 = bboxes[color]
        rect_h = height_by_color[color]
        rect_w = width_by_color[color]
        assert r0 >= 1 and c0 >= 1, "objects must lie strictly inside the grid"
        assert r0 + rect_h <= h and c0 + rect_w <= w, "decoded rectangle must fit inside the grid"
        out[r0 : r0 + rect_h, c0 : c0 + rect_w] = int(color)

    return out


def _sample_segment_lengths(count: int, total_limit: int, rng: random.Random) -> list[int] | None:
    total_min = RULER_SEGMENT_MIN * count
    total_max = min(RULER_SEGMENT_MAX * count, total_limit)
    if total_min > total_max:
        return None

    total = rng.randint(total_min, total_max)
    lengths = [RULER_SEGMENT_MIN] * count
    capacities = [RULER_SEGMENT_MAX - RULER_SEGMENT_MIN] * count
    remaining = total - total_min
    while remaining > 0:
        choices = [idx for idx, cap in enumerate(capacities) if cap > 0]
        if not choices:
            return None
        idx = rng.choice(choices)
        extra = rng.randint(1, min(capacities[idx], remaining))
        lengths[idx] += extra
        capacities[idx] -= extra
        remaining -= extra

    rng.shuffle(lengths)
    return lengths


def _choose_shape_box(target_h: int, target_w: int, rng: random.Random) -> tuple[int, int]:
    candidates = []
    for box_h in range(1, min(SHAPE_BOX_MAX, target_h) + 1):
        for box_w in range(1, min(SHAPE_BOX_MAX, target_w) + 1):
            if box_h * box_w < 2:
                continue
            if box_h < target_h or box_w < target_w:
                candidates.append((box_h, box_w))
    assert candidates, "target rectangle must allow a smaller shape box"
    return rng.choice(candidates)


def _random_connected_shape(box_h: int, box_w: int, rng: random.Random) -> tuple[tuple[int, int], ...]:
    area = box_h * box_w
    target_area = area if area == 2 else rng.randint(2, area - 1)
    cells = {(0, 0)}
    while len(cells) < target_area:
        frontier = []
        for r, c in cells:
            for dr, dc in DIR4:
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        assert frontier, "connected box should always have an expansion frontier"
        cells.add(rng.choice(frontier))
    return tuple(sorted(cells))


def _rect_overlap(
    rect_a: tuple[int, int, int, int], rect_b: tuple[int, int, int, int]
) -> bool:
    ar0, ac0, ar1, ac1 = rect_a
    br0, bc0, br1, bc1 = rect_b
    return not (ar1 <= br0 or br1 <= ar0 or ac1 <= bc0 or bc1 <= ac0)


def _cells_conflict(
    occupied: set[tuple[int, int]], new_cells: set[tuple[int, int]]
) -> bool:
    for r, c in new_cells:
        if (r, c) in occupied:
            return True
        for dr, dc in DIR4:
            if (int(r + dr), int(c + dc)) in occupied:
                return True
    return False


def _place_shapes(
    height: int,
    width: int,
    colors: list[int],
    dims_by_color: dict[int, tuple[int, int]],
    rel_shapes: dict[int, tuple[tuple[int, int], ...]],
    require_overlap: bool,
    rng: random.Random,
) -> dict[int, tuple[int, int]] | None:
    placement_order = colors[:]
    for _ in range(LAYOUT_ATTEMPTS):
        rng.shuffle(placement_order)
        anchors: dict[int, tuple[int, int]] = {}
        occupied: set[tuple[int, int]] = set()

        ok = True
        for color in placement_order:
            rect_h, rect_w = dims_by_color[color]
            shape = rel_shapes[color]
            chosen: tuple[int, int] | None = None
            for _ in range(ANCHOR_ATTEMPTS):
                r0 = rng.randint(2, height - rect_h)
                c0 = rng.randint(2, width - rect_w)
                cells = {(r0 + dr, c0 + dc) for dr, dc in shape}
                if _cells_conflict(occupied, cells):
                    continue
                chosen = (r0, c0)
                anchors[color] = chosen
                occupied.update(cells)
                break
            if chosen is None:
                ok = False
                break

        if not ok:
            continue

        rects = {
            color: (
                anchors[color][0],
                anchors[color][1],
                anchors[color][0] + dims_by_color[color][0],
                anchors[color][1] + dims_by_color[color][1],
            )
            for color in colors
        }
        has_overlap = any(
            _rect_overlap(rects[colors[i]], rects[colors[j]])
            for i in range(len(colors))
            for j in range(i + 1, len(colors))
        )
        if has_overlap == require_overlap:
            return anchors
    return None


def _bg_is_unique_dominant(grid: np.ndarray, bg: int) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    counts = counts.astype(int)
    bg_count = int(counts[values == bg][0])
    return bg_count == int(counts.max()) and int(np.count_nonzero(counts == bg_count)) == 1


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(*GRID_H_RANGE)
    width = rng.randint(*GRID_W_RANGE)

    max_colors = min(
        COLOR_COUNT_RANGE[1],
        (height - 2) // RULER_SEGMENT_MIN,
        (width - 2) // RULER_SEGMENT_MIN,
        len(PALETTE) - 1,
    )
    if max_colors < COLOR_COUNT_RANGE[0]:
        return None

    color_count = rng.randint(COLOR_COUNT_RANGE[0], max_colors)
    bg = rng.choice(PALETTE)
    colors = rng.sample([color for color in PALETTE if color != bg], color_count)

    top_lengths = _sample_segment_lengths(color_count, width - 2, rng)
    left_lengths = _sample_segment_lengths(color_count, height - 2, rng)
    if top_lengths is None or left_lengths is None:
        return None

    top_order = colors[:]
    left_order = colors[:]
    rng.shuffle(top_order)
    rng.shuffle(left_order)

    width_by_color = {color: length for color, length in zip(top_order, top_lengths)}
    height_by_color = {color: length for color, length in zip(left_order, left_lengths)}
    dims_by_color = {
        color: (height_by_color[color], width_by_color[color])
        for color in colors
    }

    rel_shapes: dict[int, tuple[tuple[int, int], ...]] = {}
    for color in colors:
        target_h, target_w = dims_by_color[color]
        box_h, box_w = _choose_shape_box(target_h, target_w, rng)
        rel_shapes[color] = _random_connected_shape(box_h, box_w, rng)

    require_overlap = rng.random() < OVERLAP_PROBABILITY
    anchors = _place_shapes(height, width, colors, dims_by_color, rel_shapes, require_overlap, rng)
    if anchors is None:
        return None

    grid = np.full((height, width), bg, dtype=int)
    c = 1
    for color in top_order:
        segment_w = width_by_color[color]
        grid[0, c : c + segment_w] = int(color)
        c += segment_w

    r = 1
    for color in left_order:
        segment_h = height_by_color[color]
        grid[r : r + segment_h, 0] = int(color)
        r += segment_h

    for color in colors:
        r0, c0 = anchors[color]
        for dr, dc in rel_shapes[color]:
            grid[r0 + dr, c0 + dc] = int(color)

    if not _bg_is_unique_dominant(grid, bg):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
