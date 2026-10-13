from __future__ import annotations

import numpy as np

from arc_utils import colors_present, find_components, get_bg, paste
import random


def _find_separator(g: np.ndarray, bg: int) -> tuple[str, int, int]:
    axes: list[tuple[str, int, int]] = []

    for r in range(g.shape[0]):
        vals = np.unique(g[r, :])
        if vals.size == 1 and int(vals[0]) != int(bg):
            axes.append(("horizontal", int(r), int(vals[0])))

    for c in range(g.shape[1]):
        vals = np.unique(g[:, c])
        if vals.size == 1 and int(vals[0]) != int(bg):
            axes.append(("vertical", int(c), int(vals[0])))

    assert len(axes) == 1, "expected exactly one full separator row or column"
    return axes[0]


def _component_side(component: dict, orient: str, axis: int) -> int:
    r0, c0, r1, c1 = component["bbox"]
    if orient == "vertical":
        if c1 < axis:
            return -1
        if c0 > axis:
            return 1
    elif orient == "horizontal":
        if r1 < axis:
            return -1
        if r0 > axis:
            return 1
    else:
        raise AssertionError("unknown separator orientation")
    return 0


def _is_solid_rect(component: dict) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    rect_area = (int(r1) - int(r0) + 1) * (int(c1) - int(c0) + 1)
    return int(component["area"]) == rect_area


def _find_source(g: np.ndarray, bg: int, orient: str, axis: int, axis_color: int) -> tuple[int, list[dict]]:
    candidates: list[tuple[int, list[dict]]] = []

    for color in colors_present(g, exclude={bg, axis_color}):
        components = find_components(
            g,
            bg=bg,
            connectivity=4,
            monochromatic=True,
            colors=[color],
        )
        assert components, "present colors must have at least one component"
        sides = {_component_side(component, orient, axis) for component in components}
        if (
            len(sides) == 1
            and 0 not in sides
            and all(_is_solid_rect(component) and int(component["area"]) > 1 for component in components)
        ):
            candidates.append((int(color), components))

    assert len(candidates) == 1, "expected one source color made of solid rectangles on one side"
    return candidates[0]


def _mirror_and_connect(
    out: np.ndarray,
    component: dict,
    orient: str,
    axis: int,
    axis_color: int,
    source_color: int,
) -> None:
    coords = component["coords"]
    r0, c0, r1, c1 = map(int, component["bbox"])
    mirrored = coords.copy()

    if orient == "vertical":
        mirrored[:, 1] = 2 * int(axis) - mirrored[:, 1]
        assert np.all((0 <= mirrored[:, 1]) & (mirrored[:, 1] < out.shape[1])), "reflection leaves grid"
        mr0, mc0 = map(int, mirrored.min(axis=0))
        mr1, mc1 = map(int, mirrored.max(axis=0))
        assert mr0 == r0 and mr1 == r1, "vertical reflection must preserve rows"

        if c1 < mc0:
            fill_start, fill_end = c1 + 1, mc0 - 1
        elif mc1 < c0:
            fill_start, fill_end = mc1 + 1, c0 - 1
        else:
            raise AssertionError("reflected object overlaps its source")
        assert fill_start <= int(axis) <= fill_end, "separator must lie between source and reflection"
        out[r0 : r1 + 1, fill_start : fill_end + 1] = int(axis_color)
    elif orient == "horizontal":
        mirrored[:, 0] = 2 * int(axis) - mirrored[:, 0]
        assert np.all((0 <= mirrored[:, 0]) & (mirrored[:, 0] < out.shape[0])), "reflection leaves grid"
        mr0, mc0 = map(int, mirrored.min(axis=0))
        mr1, mc1 = map(int, mirrored.max(axis=0))
        assert mc0 == c0 and mc1 == c1, "horizontal reflection must preserve columns"

        if r1 < mr0:
            fill_start, fill_end = r1 + 1, mr0 - 1
        elif mr1 < r0:
            fill_start, fill_end = mr1 + 1, r0 - 1
        else:
            raise AssertionError("reflected object overlaps its source")
        assert fill_start <= int(axis) <= fill_end, "separator must lie between source and reflection"
        out[fill_start : fill_end + 1, c0 : c1 + 1] = int(axis_color)
    else:
        raise AssertionError("unknown separator orientation")

    out[mirrored[:, 0], mirrored[:, 1]] = int(source_color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect the solid rectangular source objects across the full separator line and color the gap with the separator color.

    Input structure:
    - A dominant background contains one full-width or full-height one-pixel
      separator line in a non-background color.
    - One other foreground color forms solid rectangular objects, all strictly on the same side of that separator.
    - Remaining non-background pixels are scattered distractors.

    Rule:
    - Keep the input grid shape and unchanged pixels outside each affected corridor.
    - Mirror each source object across the separator line.
    - For every source object, recolor the rectangular strip between the
      original object and its reflection to the separator color, then paint the
      reflected object in the source color.

    Key insight:
    - The full straight line is a mirror axis; the real objects are identified
      structurally as the only solid rectangles confined to one side, while
      scattered pixels are noise that may be overwritten inside the corridor.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    orient, axis, axis_color = _find_separator(g, bg)
    source_color, components = _find_source(g, bg, orient, axis, axis_color)

    out = g.copy()
    for component in components:
        _mirror_and_connect(out, component, orient, axis, axis_color, source_color)
    return out

# --------------- generator (att2, o5x) ---------------

# Canvas and palette.
GRID_SIZE_RANGE = (18, 23)
COLOR_COUNT = 10
PALETTE_SIZE = 4
ORIENTATIONS = ("vertical", "horizontal")
AXIS_RATIO_RANGE = (0.35, 0.50)
SIDES = (-1, 1)

# Source rectangles and reflected corridors.
SOURCE_RECT_COUNT_RANGE = (2, 3)
RECT_HEIGHT_RANGE = (2, 5)
RECT_WIDTH_RANGE = (2, 4)
SOURCE_GAP_TO_AXIS_RANGE = (2, 5)
MIN_RECT_INTERVAL_GAP = 1

# Distractor scatter.
DISTRACTOR_DENSITY_RANGE = (0.03, 0.10)
MAX_DISTRACTOR_CLUSTERS = 4
DISTRACTOR_CLUSTER_TARGET_DIVISOR = 12
FORCE_CLUSTER_PROB = 0.70
DISTRACTOR_CLUSTER_SHAPES = (
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (1, 0), (0, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1)),
    ((1, 0), (0, 1), (1, 1)),
)

# Retry budget.
GENERATION_ATTEMPTS = 300


def _composition(total: int, parts: int) -> list[int]:
    if total <= 0:
        return [0] * parts
    return list(np.random.multinomial(int(total), [1.0 / parts] * parts))


def _pack_intervals(length: int, sizes: list[int], *, min_gap: int) -> list[tuple[int, int]] | None:
    extra = int(length) - sum(sizes) - int(min_gap) * (len(sizes) - 1)
    if extra < 0:
        return None
    gaps = _composition(extra, len(sizes) + 1)
    intervals: list[tuple[int, int]] = []
    pos = gaps[0]
    for i, size in enumerate(sizes):
        intervals.append((pos, pos + int(size) - 1))
        pos += int(size)
        if i + 1 < len(sizes):
            pos += int(min_gap) + gaps[i + 1]
    return intervals


def _axis_candidates(length: int) -> list[int]:
    min_ratio, max_ratio = AXIS_RATIO_RANGE
    return [
        i
        for i in range(1, int(length) - 1)
        if min_ratio <= min(i, int(length) - 1 - i) / float(length) <= max_ratio
    ]


def _size_gap_pairs(size_range: tuple[int, int], min_space: int) -> list[tuple[int, int]]:
    return [
        (size, gap)
        for size in range(size_range[0], size_range[1] + 1)
        for gap in range(SOURCE_GAP_TO_AXIS_RANGE[0], SOURCE_GAP_TO_AXIS_RANGE[1] + 1)
        if size + gap <= min_space
    ]


def _side_span(axis: int, length: int, size: int, gap: int, side: int) -> tuple[int, int] | None:
    if side < 0:
        end = int(axis) - int(gap) - 1
        start = end - int(size) + 1
    else:
        start = int(axis) + int(gap) + 1
        end = start + int(size) - 1
    if 0 <= start <= end < int(length):
        return start, end
    return None


def _mirror_span(axis: int, start: int, end: int) -> tuple[int, int]:
    a = int(axis)
    return tuple(sorted((2 * a - int(start), 2 * a - int(end))))


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _paint_reflected_pair(
    out: np.ndarray,
    affected: np.ndarray,
    source_mask: np.ndarray,
    *,
    src: int,
    sep: int,
    rect: tuple[int, int, int, int],
    mirror: tuple[int, int, int, int],
    corridor: tuple[int, int, int, int],
) -> None:
    """Paint one source rectangle, its mirror, and the separator-colored corridor."""
    r0, r1, c0, c1 = rect
    mr0, mr1, mc0, mc1 = mirror
    fr0, fr1, fc0, fc1 = corridor
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), src, dtype=int)
    out[fr0 : fr1 + 1, fc0 : fc1 + 1] = sep
    paste(out, patch, r0, c0)
    paste(out, patch, mr0, mc0)
    affected[fr0 : fr1 + 1, fc0 : fc1 + 1] = True
    affected[mr0 : mr1 + 1, mc0 : mc1 + 1] = True
    source_mask[r0 : r1 + 1, c0 : c1 + 1] = True


def _place_rectangles(
    out: np.ndarray, affected: np.ndarray, source_mask: np.ndarray, *, orient: str, axis: int, side: int, count: int, src: int, sep: int
) -> bool:
    """Choose source rectangles and paint their output-side reflections."""
    H, W = out.shape
    if orient == "vertical":
        cross_len, axis_len = H, W
        cross_sizes = [random.randint(*RECT_HEIGHT_RANGE) for _ in range(count)]
        axis_size_range = RECT_WIDTH_RANGE
    else:
        cross_len, axis_len = W, H
        cross_sizes = [random.randint(*RECT_WIDTH_RANGE) for _ in range(count)]
        axis_size_range = RECT_HEIGHT_RANGE

    intervals = _pack_intervals(cross_len, cross_sizes, min_gap=MIN_RECT_INTERVAL_GAP)
    pairs = _size_gap_pairs(axis_size_range, min(axis, axis_len - 1 - axis))
    if intervals is None or not pairs:
        return False

    for p0, p1 in intervals:
        size, gap = random.choice(pairs)
        span = _side_span(axis, axis_len, size, gap, side)
        if span is None:
            return False
        s0, s1 = span
        ms0, ms1 = _mirror_span(axis, s0, s1)
        if not (0 <= ms0 <= ms1 < axis_len):
            return False
        fill0, fill1 = (s1 + 1, ms0 - 1) if s1 < ms0 else (ms1 + 1, s0 - 1)
        if orient == "vertical":
            rect = (p0, p1, s0, s1)
            mirror = (p0, p1, ms0, ms1)
            corridor = (p0, p1, fill0, fill1)
        else:
            rect = (s0, s1, p0, p1)
            mirror = (ms0, ms1, p0, p1)
            corridor = (fill0, fill1, p0, p1)
        _paint_reflected_pair(
            out, affected, source_mask, src=src, sep=sep, rect=rect, mirror=mirror, corridor=corridor
        )
    return True

def _sample_scene_config() -> tuple[int, int, int, int, int, int, str, int, int, int] | None:
    """Sample high-level scene choices before any geometry-specific drawing."""
    H = random.randint(*GRID_SIZE_RANGE)
    W = random.randint(*GRID_SIZE_RANGE)
    bg, sep, src, dist = random.sample(range(COLOR_COUNT), PALETTE_SIZE)
    orient = random.choice(ORIENTATIONS)
    axis_len = W if orient == "vertical" else H
    axes = _axis_candidates(axis_len)
    if not axes:
        return None
    axis = random.choice(axes)
    side = random.choice(SIDES)
    count = random.randint(*SOURCE_RECT_COUNT_RANGE)
    return H, W, bg, sep, src, dist, orient, axis, side, count


def _can_add_cells(cells: tuple[tuple[int, int], ...], allowed: np.ndarray, occupied: set[tuple[int, int]]) -> bool:
    H, W = allowed.shape
    cell_set = set(cells)
    for r, c in cells:
        if not (0 <= r < H and 0 <= c < W and allowed[r, c]) or (r, c) in occupied:
            return False
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in occupied and (nr, nc) not in cell_set:
                return False
    return True


def _cluster_candidates(allowed: np.ndarray, occupied: set[tuple[int, int]], side_mask: np.ndarray) -> list[tuple[tuple[int, int], ...]]:
    H, W = allowed.shape
    choices: list[tuple[tuple[int, int], ...]] = []
    for r in range(H):
        for c in range(W):
            for shape in DISTRACTOR_CLUSTER_SHAPES:
                cells = tuple((r + dr, c + dc) for dr, dc in shape)
                if all(0 <= rr < H and 0 <= cc < W and side_mask[rr, cc] for rr, cc in cells):
                    if _can_add_cells(cells, allowed, occupied):
                        choices.append(cells)
    return choices


def _isolated_candidates(allowed: np.ndarray, occupied: set[tuple[int, int]], side_mask: np.ndarray | None = None) -> list[tuple[int, int]]:
    H, W = allowed.shape
    choices: list[tuple[int, int]] = []
    for r in range(H):
        for c in range(W):
            if side_mask is not None and not side_mask[r, c]:
                continue
            if not allowed[r, c] or (r, c) in occupied:
                continue
            if all((nr, nc) not in occupied for nr, nc in _neighbors4(r, c)):
                choices.append((r, c))
    return choices


def _scatter_distractors(
    inp: np.ndarray,
    out: np.ndarray,
    *,
    dist_color: int,
    allowed: np.ndarray,
    affected: np.ndarray,
    side_masks: tuple[np.ndarray, np.ndarray],
) -> bool:
    H, W = inp.shape
    min_density, max_density = DISTRACTOR_DENSITY_RANGE
    target = random.randint(int(np.ceil(min_density * H * W)), int(np.floor(max_density * H * W)))
    occupied: set[tuple[int, int]] = set()

    max_clusters = min(MAX_DISTRACTOR_CLUSTERS, max(1, target // DISTRACTOR_CLUSTER_TARGET_DIVISOR))
    cluster_count = random.randint(0, max_clusters)
    if random.random() < FORCE_CLUSTER_PROB:
        cluster_count = max(1, cluster_count)

    for _ in range(cluster_count):
        side_mask = random.choice(side_masks)
        choices = _cluster_candidates(allowed, occupied, side_mask)
        if not choices:
            continue
        for cell in random.choice(choices):
            occupied.add(cell)

    for side_mask in side_masks:
        if not any(side_mask[r, c] for r, c in occupied):
            choices = _isolated_candidates(allowed, occupied, side_mask)
            if not choices:
                return False
            occupied.add(random.choice(choices))

    while len(occupied) < target:
        choices = _isolated_candidates(allowed, occupied)
        if not choices:
            break
        occupied.add(random.choice(choices))

    if len(occupied) < int(np.ceil(min_density * H * W)):
        return False

    for r, c in occupied:
        inp[r, c] = int(dist_color)
        if not affected[r, c]:
            out[r, c] = int(dist_color)
    return True


def _gen_attempt() -> np.ndarray | None:
    config = _sample_scene_config()
    if config is None:
        return None
    H, W, bg, sep, src, dist, orient, axis, side, count = config

    out = np.full((H, W), bg, dtype=int)
    if orient == "vertical":
        out[:, axis] = sep
    else:
        out[axis, :] = sep
    affected = np.zeros((H, W), dtype=bool)
    source_mask = np.zeros((H, W), dtype=bool)

    if not _place_rectangles(
        out, affected, source_mask, orient=orient, axis=axis, side=side, count=count, src=src, sep=sep
    ):
        return None

    inp = out.copy()
    inp[affected] = bg
    axis_mask = np.zeros((H, W), dtype=bool)
    if orient == "vertical":
        inp[:, axis] = sep
        axis_mask[:, axis] = True
        side_masks = (np.tile(np.arange(W) < axis, (H, 1)), np.tile(np.arange(W) > axis, (H, 1)))
    else:
        inp[axis, :] = sep
        axis_mask[axis, :] = True
        side_masks = (
            np.tile((np.arange(H) < axis)[:, None], (1, W)),
            np.tile((np.arange(H) > axis)[:, None], (1, W)),
        )
    allowed = ~(axis_mask | source_mask)
    if not _scatter_distractors(
        inp,
        out,
        dist_color=dist,
        allowed=allowed,
        affected=affected,
        side_masks=side_masks,
    ):
        return None

    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for 538b439f")
