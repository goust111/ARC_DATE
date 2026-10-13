from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
SWAP_LINE_COLOR = 5
TURN_LINE_COLOR = 1
SEPARATOR_COLORS = (SWAP_LINE_COLOR, TURN_LINE_COLOR)
SOURCE_COLOR_POOL = (2, 3, 4, 6, 7, 8, 9)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_GENERATION_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _separator_spec(grid: np.ndarray, bg: int) -> tuple[str, list[int], list[int]]:
    g = np.asarray(grid, dtype=int)
    full_rows = [
        (r, int(g[r, 0]))
        for r in range(g.shape[0])
        if int(g[r, 0]) != bg and np.all(g[r] == g[r, 0])
    ]
    full_cols = [
        (c, int(g[0, c]))
        for c in range(g.shape[1])
        if int(g[0, c]) != bg and np.all(g[:, c] == g[0, c])
    ]

    if len(full_rows) in {2, 3} and not full_cols:
        return "h", [r for r, _color in full_rows], [color for _r, color in full_rows]
    if len(full_cols) in {2, 3} and not full_rows:
        return "v", [c for c, _color in full_cols], [color for _c, color in full_cols]
    raise AssertionError("expected 2 or 3 full separator lines with a single orientation")


def _strip_bounds(length: int, separators: list[int]) -> list[tuple[int, int]]:
    bounds: list[tuple[int, int]] = []
    start = 0
    for axis in separators:
        bounds.append((start, axis - 1))
        start = axis + 1
    bounds.append((start, length - 1))
    assert all(lo <= hi for lo, hi in bounds), "every strip must have positive width"
    return bounds


def _swap_values(values: np.ndarray, color_a: int, color_b: int) -> np.ndarray:
    out = values.copy()
    mask_a = out == color_a
    mask_b = out == color_b
    out[mask_a] = color_b
    out[mask_b] = color_a
    return out


def _mirror_coords(coords: np.ndarray, axis: int, orient: str) -> np.ndarray:
    out = coords.copy()
    if orient == "v":
        out[:, 1] = 2 * int(axis) - out[:, 1]
    else:
        out[:, 0] = 2 * int(axis) - out[:, 0]
    return out


def _rotate_coords_180(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(coords)
    out = coords.copy()
    out[:, 0] = r0 + r1 - out[:, 0]
    out[:, 1] = c0 + c1 - out[:, 1]
    return out


def _transform_state(
    coords: np.ndarray,
    values: np.ndarray,
    axis: int,
    orient: str,
    line_color: int,
    color_a: int,
    color_b: int,
) -> tuple[np.ndarray, np.ndarray]:
    new_coords = _mirror_coords(coords, axis, orient)
    new_values = values.copy()

    if line_color == SWAP_LINE_COLOR:
        new_values = _swap_values(new_values, color_a, color_b)
    elif line_color == TURN_LINE_COLOR:
        new_coords = _rotate_coords_180(new_coords)
    else:
        raise AssertionError("unexpected separator color")

    return new_coords, new_values


def _paint(out: np.ndarray, coords: np.ndarray, values: np.ndarray) -> None:
    out[coords[:, 0], coords[:, 1]] = values


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect the unique two-color source pattern across each separator strip-by-strip, with the separator color choosing which operator accumulates along the chain.

    Input structure
    - The background grid is split into 3 or 4 strips by 2 or 3 full separator rows or columns.
    - Exactly one strip contains a complete source pattern made of exactly two non-background colors.
    - All other non-background pixels belong to the separator lines.

    Rule
    - Starting from the source strip, propagate one image outward across each adjacent separator until both outermost strips are reached.
    - Every crossing first mirrors the current image across that full separator line.
    - A grey separator swaps the pattern's two colors after mirroring, while a blue separator rotates the mirrored image 180 degrees inside its mirrored bounding box without changing its colors.
    - Preserve the original source pattern, the separators, and the background.

    Key insight
    - The separator color is an operator, not decoration: each reflected image becomes the input to the next separator, so geometry and color effects compose along the strip chain.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = get_bg(g)
    orient, separators, line_colors = _separator_spec(g, bg)
    assert set(line_colors).issubset(set(SEPARATOR_COLORS)), "unexpected separator colors"

    strip_axis = _strip_bounds(g.shape[1] if orient == "v" else g.shape[0], separators)
    source_mask = g != bg
    if orient == "v":
        source_mask[:, separators] = False
    else:
        source_mask[separators, :] = False

    source_coords = np.argwhere(source_mask)
    assert source_coords.size > 0, "expected a source pattern outside the separators"
    source_values = g[source_mask].astype(int)
    source_colors = sorted(set(map(int, source_values.tolist())))
    assert len(source_colors) == 2, "source pattern must use exactly two colors"
    color_a, color_b = source_colors

    r0, c0, r1, c1 = bbox_coords(source_coords)
    span0, span1 = (c0, c1) if orient == "v" else (r0, r1)
    source_strip = None
    for idx, (lo, hi) in enumerate(strip_axis):
        if lo <= span0 and span1 <= hi:
            source_strip = idx
            break
    assert source_strip is not None, "source pattern must lie fully inside one strip"

    out = g.copy()

    coords = source_coords.copy()
    values = source_values.copy()
    strip_idx = int(source_strip)
    while strip_idx > 0:
        line_idx = strip_idx - 1
        coords, values = _transform_state(
            coords, values, separators[line_idx], orient, line_colors[line_idx], color_a, color_b
        )
        lo, hi = strip_axis[strip_idx - 1]
        perp = coords[:, 1] if orient == "v" else coords[:, 0]
        assert int(perp.min()) >= lo and int(perp.max()) <= hi
        _paint(out, coords, values)
        strip_idx -= 1

    coords = source_coords.copy()
    values = source_values.copy()
    strip_idx = int(source_strip)
    while strip_idx < len(strip_axis) - 1:
        line_idx = strip_idx
        coords, values = _transform_state(
            coords, values, separators[line_idx], orient, line_colors[line_idx], color_a, color_b
        )
        lo, hi = strip_axis[strip_idx + 1]
        perp = coords[:, 1] if orient == "v" else coords[:, 0]
        assert int(perp.min()) >= lo and int(perp.max()) <= hi
        _paint(out, coords, values)
        strip_idx += 1

    return out


def _mask_frontier(mask: np.ndarray) -> list[tuple[int, int]]:
    frontier: set[tuple[int, int]] = set()
    rows, cols = mask.shape
    for r, c in np.argwhere(mask):
        for dr, dc in DIRS4:
            rr, cc = int(r + dr), int(c + dc)
            if 0 <= rr < rows and 0 <= cc < cols and not bool(mask[rr, cc]):
                frontier.add((rr, cc))
    return list(frontier)


def _random_path(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = start
    end_r, end_c = end
    path = [(r, c)]
    while (r, c) != (end_r, end_c):
        options = []
        if r != end_r:
            options.append((r + (1 if end_r > r else -1), c))
        if c != end_c:
            options.append((r, c + (1 if end_c > c else -1)))
        r, c = random.choice(options)
        path.append((r, c))
    return path


def _connected_subset(mask: np.ndarray, size: int) -> set[tuple[int, int]] | None:
    cells = [tuple(map(int, cell)) for cell in np.argwhere(mask)]
    if size <= 0 or size >= len(cells):
        return None
    chosen = {random.choice(cells)}
    while len(chosen) < size:
        frontier: set[tuple[int, int]] = set()
        for r, c in chosen:
            for dr, dc in DIRS4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and bool(mask[rr, cc]):
                    if (rr, cc) not in chosen:
                        frontier.add((rr, cc))
        if not frontier:
            return None
        chosen.add(random.choice(list(frontier)))
    return chosen


def _swap_patch_colors(patch: np.ndarray, color_a: int, color_b: int) -> np.ndarray:
    out = patch.copy()
    mask_a = out == color_a
    mask_b = out == color_b
    out[mask_a] = color_b
    out[mask_b] = color_a
    return out


def _patch_is_degenerate(patch: np.ndarray, color_a: int, color_b: int) -> bool:
    fg = patch[patch != BG]
    if fg.size < 6 or len(set(map(int, fg.tolist()))) != 2:
        return True
    counts = Counter(map(int, fg.tolist()))
    if min(counts.values()) < 2:
        return True

    hflip = np.fliplr(patch)
    vflip = np.flipud(patch)
    rot180 = np.rot90(patch, 2)
    if any(np.array_equal(patch, other) for other in (hflip, vflip, rot180)):
        return True
    if np.array_equal(patch, _swap_patch_colors(hflip, color_a, color_b)):
        return True
    if np.array_equal(patch, _swap_patch_colors(vflip, color_a, color_b)):
        return True
    if np.array_equal(_swap_patch_colors(hflip, color_a, color_b), vflip):
        return True
    if np.array_equal(_swap_patch_colors(vflip, color_a, color_b), hflip):
        return True
    return False


def _make_source_patch(height: int, width: int, colors: tuple[int, int]) -> np.ndarray | None:
    color_a, color_b = colors
    area_min = max(6, int(np.ceil(height * width * 0.45)))
    area_max = min(height * width - 1, int(np.floor(height * width * 0.75)))
    if area_min > area_max:
        return None

    for _ in range(200):
        mask = np.zeros((height, width), dtype=bool)
        center = (random.randrange(height), random.randrange(width))
        anchors = [
            (0, random.randrange(width)),
            (height - 1, random.randrange(width)),
            (random.randrange(height), 0),
            (random.randrange(height), width - 1),
        ]
        for anchor in anchors:
            for r, c in _random_path(anchor, center):
                mask[r, c] = True

        base_area = int(mask.sum())
        lower = max(area_min, base_area)
        if lower > area_max:
            continue
        target_area = random.randint(lower, area_max)
        while int(mask.sum()) < target_area:
            frontier = _mask_frontier(mask)
            if not frontier:
                break
            r, c = random.choice(frontier)
            mask[r, c] = True

        color_a_count = random.randint(2, int(mask.sum()) - 2)
        color_a_cells = _connected_subset(mask, color_a_count)
        if color_a_cells is None:
            continue

        patch = np.zeros((height, width), dtype=int)
        patch[mask] = color_b
        for r, c in color_a_cells:
            patch[r, c] = color_a

        if _patch_is_degenerate(patch, color_a, color_b):
            continue
        return patch

    return None


def _layout_from_widths(widths: list[int]) -> tuple[list[tuple[int, int]], list[int]]:
    bounds: list[tuple[int, int]] = []
    separators: list[int] = []
    start = 0
    for idx, width in enumerate(widths):
        end = start + width - 1
        bounds.append((start, end))
        if idx < len(widths) - 1:
            separators.append(end + 1)
            start = end + 2
    return bounds, separators


def _reflect_bbox(
    bbox: tuple[int, int, int, int], axis: int, orient: str
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    if orient == "v":
        return r0, 2 * axis - c1, r1, 2 * axis - c0
    return 2 * axis - r1, c0, 2 * axis - r0, c1


def _bbox_in_strip(
    bbox: tuple[int, int, int, int], strip: tuple[int, int], orient: str
) -> bool:
    lo, hi = strip
    if orient == "v":
        return lo <= bbox[1] and bbox[3] <= hi
    return lo <= bbox[0] and bbox[2] <= hi


def _valid_perp_starts(
    bounds: list[tuple[int, int]],
    separators: list[int],
    source_strip: int,
    obj_perp: int,
    obj_parallel: int,
    orient: str,
) -> list[int]:
    lo, hi = bounds[source_strip]
    starts: list[int] = []
    for start in range(lo, hi - obj_perp + 2):
        bbox = (
            (0, start, obj_parallel - 1, start + obj_perp - 1)
            if orient == "v"
            else (start, 0, start + obj_perp - 1, obj_parallel - 1)
        )
        ok = True

        cur = bbox
        for strip_idx in range(source_strip, 0, -1):
            cur = _reflect_bbox(cur, separators[strip_idx - 1], orient)
            if not _bbox_in_strip(cur, bounds[strip_idx - 1], orient):
                ok = False
                break
        if not ok:
            continue

        cur = bbox
        for strip_idx in range(source_strip, len(bounds) - 1):
            cur = _reflect_bbox(cur, separators[strip_idx], orient)
            if not _bbox_in_strip(cur, bounds[strip_idx + 1], orient):
                ok = False
                break
        if ok:
            starts.append(start)
    return starts


def _choose_line_colors(count: int) -> list[int]:
    if random.random() < 0.85:
        colors = [random.choice(SEPARATOR_COLORS) for _ in range(count)]
        if len(set(colors)) == 1:
            other = TURN_LINE_COLOR if colors[0] == SWAP_LINE_COLOR else SWAP_LINE_COLOR
            for idx in random.sample(range(count), random.randint(1, count - 1)):
                colors[idx] = other
        return colors
    return [random.choice(SEPARATOR_COLORS) for _ in range(count)]


def _weighted_placements(
    bounds: list[tuple[int, int]],
    separators: list[int],
    obj_perp: int,
    obj_parallel: int,
    orient: str,
) -> list[tuple[int, int]]:
    placements: list[tuple[int, int]] = []
    last_strip = len(bounds) - 1
    for strip_idx in range(len(bounds)):
        starts = _valid_perp_starts(bounds, separators, strip_idx, obj_perp, obj_parallel, orient)
        weight = 5 if 0 < strip_idx < last_strip else 1
        for start in starts:
            placements.extend([(strip_idx, start)] * weight)
    return placements


def _generate_attempt() -> np.ndarray | None:
    orient = random.choice(("v", "h"))
    line_count = 3 if random.random() < 0.65 else 2
    strip_count = line_count + 1
    widths = random.sample(range(4, 9), strip_count)
    random.shuffle(widths)
    perp_size = sum(widths) + line_count
    parallel_size = random.randint(10, 17)

    obj_perp_max = min(widths) - 1
    obj_parallel_max = min(6, parallel_size - 1)
    if obj_perp_max < 3 or obj_parallel_max < 3:
        return None

    obj_perp = random.randint(3, min(5, obj_perp_max))
    obj_parallel = random.randint(3, obj_parallel_max)
    colors = tuple(random.sample(SOURCE_COLOR_POOL, 2))
    patch_shape = (obj_parallel, obj_perp) if orient == "v" else (obj_perp, obj_parallel)
    patch = _make_source_patch(patch_shape[0], patch_shape[1], colors)
    if patch is None:
        return None

    bounds, separators = _layout_from_widths(widths)
    line_colors = _choose_line_colors(line_count)
    placements = _weighted_placements(bounds, separators, obj_perp, obj_parallel, orient)
    if not placements:
        return None

    source_strip, perp_start = random.choice(placements)
    parallel_start = random.randint(0, parallel_size - obj_parallel)
    shape = (parallel_size, perp_size) if orient == "v" else (perp_size, parallel_size)
    grid = np.full(shape, BG, dtype=int)

    for axis, color in zip(separators, line_colors):
        if orient == "v":
            grid[:, axis] = color
        else:
            grid[axis, :] = color

    if orient == "v":
        r0, c0 = parallel_start, perp_start
    else:
        r0, c0 = perp_start, parallel_start
    mask = patch != BG
    grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]][mask] = patch[mask]

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-house instance")
