import numpy as np

from arc_utils import bbox, colors_present, get_bg, paste
import random


def main(grid: np.ndarray) -> np.ndarray:
    """Rebuild the visible frame fragments as centered concentric square rings.

    Input structure
    The input has a dominant background and several non-background colors. Each
    foreground color marks one square frame or center block, possibly with gaps
    caused by overlap with other frames. The intended side length of a color is
    the larger side of the bbox of all pixels of that color.

    Rule
    Sort the colors by intended side length from largest to smallest. Create a
    square output whose side is the largest length, then draw each color as the
    border of the centered square with that side length.

    Key insight
    Positions in the input are only a scrambled/overlapped display of the
    nested frames; the bbox side lengths provide the layer order and output
    ring sizes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    layers: list[tuple[int, int]] = []
    for color in colors_present(g, exclude={bg}):
        box = bbox(g == color)
        assert box is not None
        r0, c0, r1, c1 = box
        size = max(r1 - r0 + 1, c1 - c0 + 1)
        layers.append((size, color))

    assert layers, "expected at least one foreground layer"
    sizes = [size for size, _ in layers]
    assert len(set(sizes)) == len(sizes), "layer sizes must be unique"

    n = max(sizes)
    expected_sizes = list(range(n, 0, -2))
    assert sorted(sizes, reverse=True) == expected_sizes, "layers must nest by two-cell steps"

    out = np.full((n, n), bg, dtype=int)
    for size, color in sorted(layers, reverse=True):
        inset = (n - size) // 2
        lo = inset
        hi = n - inset - 1
        out[lo, lo : hi + 1] = color
        out[hi, lo : hi + 1] = color
        out[lo : hi + 1, lo] = color
        out[lo : hi + 1, hi] = color

    return out

# --------------- generator (o5x) ---------------
# Canvas and object size bounds.
BACKGROUND_COLOR = 0
OUTPUT_SIZE_MIN = 8
OUTPUT_SIZE_MAX = 10
FRAME_SIZE_STEP = 2
INPUT_SIDE_MIN = 13
INPUT_SIDE_EXTRA_MIN = 4
INPUT_SIDE_EXTRA_MAX = 9

# Palette, overlap targets, and retries.
FOREGROUND_COLOR_MIN = 1
FOREGROUND_COLOR_MAX_EXCLUSIVE = 10
OVERLAP_TARGET_MIN = 1
OVERLAP_TARGET_MAX = 3
OVERLAP_CLUSTER_POOL = 4
GENERATION_ATTEMPTS = 300

Placement = tuple[int, int, int]
VisibleBox = tuple[int, int, int, int]


def _gen_frame_patch(size: int, color: int) -> np.ndarray:
    patch = np.full((size, size), BACKGROUND_COLOR, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    return patch


def _gen_draw_ring(out: np.ndarray, size: int, color: int) -> None:
    n = out.shape[0]
    inset = (n - size) // 2
    hi = n - inset - 1
    out[inset, inset : hi + 1] = color
    out[hi, inset : hi + 1] = color
    out[inset : hi + 1, inset] = color
    out[inset : hi + 1, hi] = color


def _gen_inclusive_boxes_overlap(a: VisibleBox, b: VisibleBox) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar0 <= br1 and br0 <= ar1 and ac0 <= bc1 and bc0 <= ac1


def _gen_placement_bounds(box: Placement) -> VisibleBox:
    r, c, size = box
    return r, c, r + size - 1, c + size - 1


def _gen_boxes_overlap(a: Placement, b: Placement) -> bool:
    return _gen_inclusive_boxes_overlap(_gen_placement_bounds(a), _gen_placement_bounds(b))


def _gen_overlap_count(placements: list[Placement | None]) -> int:
    return _gen_visible_overlap_count([_gen_placement_bounds(p) for p in placements if p is not None])


def _gen_visible_overlap_count(boxes: list[VisibleBox]) -> int:
    total = 0
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            total += int(_gen_inclusive_boxes_overlap(boxes[i], boxes[j]))
    return total


def _gen_valid_positions(
    side: int,
    size: int,
    placements: list[Placement | None],
    *,
    must_overlap: set[int] | None = None,
    must_not_overlap: set[int] | None = None,
) -> list[tuple[int, int]]:
    must_overlap = must_overlap or set()
    must_not_overlap = must_not_overlap or set()
    positions = []
    for r in range(0, side - size + 1):
        for c in range(0, side - size + 1):
            box = (r, c, size)
            ok = True
            for idx, other in enumerate(placements):
                if other is None:
                    continue
                overlaps = _gen_boxes_overlap(box, other)
                if idx in must_overlap and not overlaps:
                    ok = False
                    break
                if idx in must_not_overlap and overlaps:
                    ok = False
                    break
            if ok:
                positions.append((r, c))
    random.shuffle(positions)
    return positions


def _gen_build_output(n: int, sizes: list[int], colors: list[int]) -> np.ndarray:
    out = np.full((n, n), BACKGROUND_COLOR, dtype=int)
    for size, color in zip(sizes, colors):
        _gen_draw_ring(out, size, color)
    return out


def _gen_choose_cluster(count: int, target_overlaps: int) -> tuple[list[int], set[tuple[int, int]]]:
    pool = range(min(count, OVERLAP_CLUSTER_POOL))
    if target_overlaps == 1:
        cluster = random.sample(pool, 2)
        edges = {(cluster[0], cluster[1])}
    elif target_overlaps == 2:
        cluster = random.sample(pool, 3)
        edges = {(cluster[0], cluster[1]), (cluster[1], cluster[2])}
    else:
        cluster = random.sample(pool, 3)
        edges = {(cluster[0], cluster[1]), (cluster[0], cluster[2]), (cluster[1], cluster[2])}
    return cluster, {tuple(sorted(edge)) for edge in edges}


def _gen_place_frames(side: int, sizes: list[int], target_overlaps: int) -> list[Placement] | None:
    count = len(sizes)
    cluster, target_edges = _gen_choose_cluster(count, target_overlaps)
    placements: list[Placement | None] = [None] * count

    first = cluster[0]
    first_positions = _gen_valid_positions(side, sizes[first], placements)
    if not first_positions:
        return None
    r, c = random.choice(first_positions)
    placements[first] = (r, c, sizes[first])

    for idx in cluster[1:]:
        must_overlap = {j for j in range(count) if placements[j] is not None and tuple(sorted((idx, j))) in target_edges}
        must_not_overlap = {j for j in range(count) if placements[j] is not None and tuple(sorted((idx, j))) not in target_edges}
        candidates = _gen_valid_positions(
            side,
            sizes[idx],
            placements,
            must_overlap=must_overlap,
            must_not_overlap=must_not_overlap,
        )
        if not candidates:
            return None
        r, c = random.choice(candidates)
        placements[idx] = (r, c, sizes[idx])

    for idx in random.sample([i for i in range(count) if placements[i] is None], count - len(cluster)):
        candidates = _gen_valid_positions(
            side,
            sizes[idx],
            placements,
            must_not_overlap={j for j in range(count) if placements[j] is not None},
        )
        if not candidates:
            return None
        r, c = random.choice(candidates)
        placements[idx] = (r, c, sizes[idx])

    final = [p for p in placements if p is not None]
    if _gen_overlap_count(final) != target_overlaps:
        return None
    return final


def _gen_sample_layers() -> tuple[int, list[int], list[int], np.ndarray]:
    n = random.randint(OUTPUT_SIZE_MIN, OUTPUT_SIZE_MAX)
    sizes = list(range(n, 0, -FRAME_SIZE_STEP))
    colors = random.sample(range(FOREGROUND_COLOR_MIN, FOREGROUND_COLOR_MAX_EXCLUSIVE), len(sizes))
    expected = _gen_build_output(n, sizes, colors)
    return n, sizes, colors, expected


def _gen_render_input(side: int, sizes: list[int], colors: list[int], placements: list[Placement]) -> np.ndarray:
    inp = np.full((side, side), BACKGROUND_COLOR, dtype=int)
    order = random.sample(range(len(sizes)), len(sizes))
    for idx in order:
        r, c, _ = placements[idx]
        paste(inp, _gen_frame_patch(sizes[idx], colors[idx]), r, c, transparent=BACKGROUND_COLOR, clip=True)
    return inp


def _gen_visible_boxes_preserve_sizes(inp: np.ndarray, sizes: list[int], colors: list[int]) -> list[VisibleBox] | None:
    visible_boxes = []
    for size, color in zip(sizes, colors):
        box = bbox(inp == color)
        if box is None:
            return None
        r0, c0, r1, c1 = box
        if max(r1 - r0 + 1, c1 - c0 + 1) != size:
            return None
        visible_boxes.append(box)
    return visible_boxes


def _gen_oracle_matches(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        actual = main(inp)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    n, sizes, colors, expected = _gen_sample_layers()

    side = random.randint(max(INPUT_SIDE_MIN, n + INPUT_SIDE_EXTRA_MIN), n + INPUT_SIDE_EXTRA_MAX)
    possible_pairs = len(sizes) * (len(sizes) - 1) // 2
    target_overlaps = random.randint(OVERLAP_TARGET_MIN, min(OVERLAP_TARGET_MAX, possible_pairs))
    placements = _gen_place_frames(side, sizes, target_overlaps)
    if placements is None:
        return None

    inp = _gen_render_input(side, sizes, colors, placements)
    visible_boxes = _gen_visible_boxes_preserve_sizes(inp, sizes, colors)
    if visible_boxes is None:
        return None
    if not (1 <= _gen_visible_overlap_count(visible_boxes) <= 3):
        return None
    if not _gen_oracle_matches(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid c658a4bd input")
