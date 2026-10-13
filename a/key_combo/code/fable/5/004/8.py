from __future__ import annotations

from collections import Counter, deque

import numpy as np


# ARC grey is the visible occluder specified by the puzzle.  Pattern and
# background colors are deliberately chosen to exclude it.
GREY = 5


def _background(grid: np.ndarray) -> int:
    """Most frequent color, following arc_utils.get_bg."""
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as (N, 2) arrays of (row, column)."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    result: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        queue = deque([(sr, sc)])
        points: list[tuple[int, int]] = []
        while queue:
            r, c = queue.popleft()
            points.append((r, c))
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    queue.append((rr, cc))
        result.append(np.asarray(points, dtype=int))
    return result


def _bbox(points: np.ndarray) -> tuple[int, int, int, int]:
    """Inclusive (r0, c0, r1, c1) bbox, following arc_utils.bbox_coords."""
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    return int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])


def _fragment_patches(grid: np.ndarray, bg: int) -> tuple[list[np.ndarray], tuple[int, int], np.ndarray]:
    """Extract the equal rectangular grey-bearing fragments and isolated marker."""
    components = _components4(grid != bg)
    fragments: list[tuple[int, np.ndarray]] = []

    for index, points in enumerate(components):
        r0, c0, r1, c1 = _bbox(points)
        height, width = r1 - r0 + 1, c1 - c0 + 1
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if (
            len(points) == height * width
            and np.any(patch == GREY)
            and np.any(patch != GREY)
        ):
            fragments.append((index, patch.copy()))

    assert len(fragments) >= 2, "expected several grey-occluded fragments"
    shapes = {patch.shape for _index, patch in fragments}
    assert len(shapes) == 1, "all fragments must use one local stamp frame"
    shape = next(iter(shapes))
    assert shape[0] >= 2 and shape[1] >= 2

    fragment_indices = {index for index, _patch in fragments}
    marker_components = [
        points
        for index, points in enumerate(components)
        if index not in fragment_indices and len(points) == 1
    ]
    assert len(marker_components) == 1, "expected one isolated colored marker"
    return [patch for _index, patch in fragments], shape, marker_components[0][0]


def _reconstruct_stamp(patches: list[np.ndarray]) -> np.ndarray:
    """Overlay visible cells in the common fragment-local coordinate system."""
    stamp = np.full(patches[0].shape, -1, dtype=int)
    known = np.zeros(patches[0].shape, dtype=bool)
    for patch in patches:
        visible = patch != GREY
        assert np.all(patch[visible] >= 0)
        assert np.all((~known) | (~visible) | (stamp == patch)), "fragment colors conflict"
        stamp[visible] = patch[visible]
        known |= visible
    assert np.all(known), "the complementary fragments must reveal every stamp cell"
    assert not np.any(stamp == GREY), "grey belongs only to occluders"
    return stamp


def main(grid: np.ndarray) -> np.ndarray:
    """Reconstruct an unseen multicolor stamp from occluded fragments and copy it onto its color anchor.

    Input structure:
    - The dominant color is background. Several separated, equally sized rectangular
      footprints contain aligned remnants of one multicolor stamp under connected grey
      occluders, and one further foreground component is an isolated colored marker.

    Rule:
    - Overlay the fragment footprints in their own row/column coordinates, ignoring grey,
      to recover every cell of the stamp. The marker color occurs exactly once in that
      recovered stamp. Copy the complete stamp so that this uniquely colored stamp cell
      covers the marker, while leaving every original fragment and occluder unchanged.

    Key insight:
    - No fragment is complete: their complementary visible cells determine the stamp,
      while the marker's color—not either bounding-box corner—determines its placement.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D ARC grid"
    bg = _background(g)
    patches, (stamp_h, stamp_w), marker = _fragment_patches(g, bg)
    stamp = _reconstruct_stamp(patches)

    marker_r, marker_c = map(int, marker)
    marker_color = int(g[marker_r, marker_c])
    anchors = np.argwhere(stamp == marker_color)
    assert len(anchors) == 1, "marker color must identify one stamp-local anchor"
    anchor_r, anchor_c = map(int, anchors[0])

    dst_r, dst_c = marker_r - anchor_r, marker_c - anchor_c
    height, width = g.shape
    assert 0 <= dst_r and dst_r + stamp_h <= height
    assert 0 <= dst_c and dst_c + stamp_w <= width

    destination = g[dst_r : dst_r + stamp_h, dst_c : dst_c + stamp_w]
    allowed = destination == bg
    allowed[anchor_r, anchor_c] |= destination[anchor_r, anchor_c] == marker_color
    assert np.all(allowed), "target stamp footprint must be empty except for its marker"

    out = g.copy()
    out[dst_r : dst_r + stamp_h, dst_c : dst_c + stamp_w] = stamp
    return out


# ------------------------------ generator ---------------------------------

_MAX_GENERATION_ATTEMPTS = 400
_MASK_ATTEMPTS = 3000
_PACK_ATTEMPTS = 250
_CANVAS_MIN, _CANVAS_MAX = 22, 30
_STAMP_H_RANGE = (3, 5)
_STAMP_W_RANGE = (4, 6)
_FRAGMENT_COUNTS = (3, 4)
_GAP = 2


def _grow_connected_mask(
    height: int, width: int, size: int, rng: np.random.Generator
) -> np.ndarray:
    """Grow one irregular 4-connected grey occluder."""
    mask = np.zeros((height, width), dtype=bool)
    start = int(rng.integers(height * width))
    mask[start // width, start % width] = True
    while int(mask.sum()) < size:
        frontier: set[tuple[int, int]] = set()
        for r, c in np.argwhere(mask):
            for rr, cc in ((int(r) - 1, int(c)), (int(r) + 1, int(c)), (int(r), int(c) - 1), (int(r), int(c) + 1)):
                if 0 <= rr < height and 0 <= cc < width and not mask[rr, cc]:
                    frontier.add((rr, cc))
        assert frontier
        choices = tuple(frontier)
        r, c = choices[int(rng.integers(len(choices)))]
        mask[r, c] = True
    return mask


def _make_stamp(
    height: int, width: int, bg: int, rng: np.random.Generator
) -> tuple[np.ndarray, tuple[int, int], int]:
    """Make a dense multicolor pattern with exactly one uniquely colored anchor."""
    available = [color for color in range(10) if color not in (bg, GREY)]
    color_count = int(rng.integers(3, 5))
    palette = list(map(int, rng.choice(available, size=color_count, replace=False)))
    anchor_color = palette[0]
    fill_colors = palette[1:]

    anchor_options = [(r, c) for r in range(height) for c in range(width) if (r, c) != (0, 0)]
    anchor = anchor_options[int(rng.integers(len(anchor_options)))]
    cell_count = height * width - 1
    values = [color for color in fill_colors for _ in range(2)]
    values.extend(
        int(fill_colors[int(rng.integers(len(fill_colors)))])
        for _ in range(cell_count - len(values))
    )
    rng.shuffle(values)

    stamp = np.empty((height, width), dtype=int)
    cursor = 0
    for r in range(height):
        for c in range(width):
            if (r, c) == anchor:
                stamp[r, c] = anchor_color
            else:
                stamp[r, c] = values[cursor]
                cursor += 1
    return stamp, anchor, anchor_color


def _make_occluders(
    stamp: np.ndarray,
    anchor: tuple[int, int],
    count: int,
    rng: np.random.Generator,
) -> list[np.ndarray] | None:
    """Sample complementary masks for which every individual fragment is necessary."""
    height, width = stamp.shape
    area = height * width
    lower = max(2, int(np.ceil(0.40 * area)))
    upper = min(area - 4, int(np.floor(0.58 * area)))
    if lower > upper:
        return None

    for _ in range(_MASK_ATTEMPTS):
        masks = [
            _grow_connected_mask(height, width, int(rng.integers(lower, upper + 1)), rng)
            for _ in range(count)
        ]
        if len({mask.tobytes() for mask in masks}) != count:
            continue
        # No cell may remain hidden in every fragment.
        if np.logical_and.reduce(masks).any():
            continue
        # The anchor appears in multiple remnants, so its local position is visible.
        if sum(not bool(mask[anchor]) for mask in masks) < 2:
            continue

        necessary = True
        for index, mask in enumerate(masks):
            other_hidden = np.logical_and.reduce(
                [other for j, other in enumerate(masks) if j != index]
            )
            exclusive_visible = (~mask) & other_hidden
            visible_colors = set(map(int, stamp[~mask]))
            if not exclusive_visible.any() or len(visible_colors) < 2:
                necessary = False
                break
        if necessary:
            return masks
    return None


def _separated(
    candidate: tuple[int, int],
    placed: list[tuple[int, int]],
    height: int,
    width: int,
) -> bool:
    r, c = candidate
    for pr, pc in placed:
        if not (
            r + height + _GAP <= pr
            or pr + height + _GAP <= r
            or c + width + _GAP <= pc
            or pc + width + _GAP <= c
        ):
            return False
    return True


def _place_regions(
    canvas_shape: tuple[int, int],
    stamp_shape: tuple[int, int],
    anchor: tuple[int, int],
    fragment_count: int,
    rng: np.random.Generator,
) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    """Place fragments and reserve one empty anchor target with two-cell gaps."""
    canvas_h, canvas_w = canvas_shape
    height, width = stamp_shape
    anchor_r, anchor_c = anchor
    target_candidates = [
        (r, c)
        for r in range(1, canvas_h - height - anchor_r)
        for c in range(1, canvas_w - width - anchor_c)
    ]
    all_candidates = [
        (r, c)
        for r in range(1, canvas_h - height)
        for c in range(1, canvas_w - width)
    ]
    if not target_candidates:
        return None

    for _ in range(_PACK_ATTEMPTS):
        target = target_candidates[int(rng.integers(len(target_candidates)))]
        placed = [target]
        fragments: list[tuple[int, int]] = []
        for _index in range(fragment_count):
            valid = [position for position in all_candidates if _separated(position, placed, height, width)]
            if not valid:
                break
            position = valid[int(rng.integers(len(valid)))]
            placed.append(position)
            fragments.append(position)
        if len(fragments) == fragment_count:
            rng.shuffle(fragments)
            return fragments, target
    return None


def _semantic_valid(
    inp: np.ndarray,
    expected: np.ndarray,
    stamp: np.ndarray,
    masks: list[np.ndarray],
    fragment_positions: list[tuple[int, int]],
    target: tuple[int, int],
    anchor: tuple[int, int],
    anchor_color: int,
    bg: int,
) -> bool:
    """Check the stated semantics and anti-shortcut counterexamples independently."""
    height, width = stamp.shape
    if inp.shape != expected.shape or max(inp.shape) > 30:
        return False
    if _background(inp) != bg or len(masks) not in _FRAGMENT_COUNTS:
        return False
    if anchor == (0, 0) or int(stamp[anchor]) != anchor_color:
        return False
    if int(np.sum(stamp == anchor_color)) != 1:
        return False
    if any(int(np.sum(stamp == color)) < 2 for color in np.unique(stamp) if int(color) != anchor_color):
        return False
    if np.logical_and.reduce(masks).any():
        return False

    for index, (mask, (r0, c0)) in enumerate(zip(masks, fragment_positions)):
        if len(_components4(mask)) != 1 or not mask.any() or mask.all():
            return False
        others = np.logical_and.reduce([other for j, other in enumerate(masks) if j != index])
        if not ((~mask) & others).any():
            return False
        expected_patch = stamp.copy()
        expected_patch[mask] = GREY
        if not np.array_equal(inp[r0 : r0 + height, c0 : c0 + width], expected_patch):
            return False
        if not np.array_equal(expected[r0 : r0 + height, c0 : c0 + width], expected_patch):
            return False

    tr, tc = target
    ar, ac = anchor
    marker_r, marker_c = tr + ar, tc + ac
    target_input = inp[tr : tr + height, tc : tc + width]
    if int(np.sum(target_input != bg)) != 1 or int(target_input[ar, ac]) != anchor_color:
        return False
    if not np.array_equal(expected[tr : tr + height, tc : tc + width], stamp):
        return False
    if int(np.sum(inp != expected)) != height * width - 1:
        return False

    # Explicit counterexample: treating the marker as the stamp's upper-left corner
    # is in-bounds but produces a different result.
    if marker_r + height > inp.shape[0] or marker_c + width > inp.shape[1]:
        return False
    wrong_anchor = inp.copy()
    wrong_anchor[marker_r : marker_r + height, marker_c : marker_c + width] = stamp
    if np.array_equal(wrong_anchor, expected):
        return False

    try:
        solved = main(inp)
    except (AssertionError, ValueError, IndexError):
        return False
    return bool(np.array_equal(solved, expected))


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    canvas_h = int(rng.integers(_CANVAS_MIN, _CANVAS_MAX + 1))
    canvas_w = int(rng.integers(_CANVAS_MIN, _CANVAS_MAX + 1))
    stamp_h = int(rng.integers(_STAMP_H_RANGE[0], _STAMP_H_RANGE[1] + 1))
    stamp_w = int(rng.integers(_STAMP_W_RANGE[0], _STAMP_W_RANGE[1] + 1))
    fragment_count = int(_FRAGMENT_COUNTS[int(rng.integers(len(_FRAGMENT_COUNTS)))])

    bg = int(rng.choice([color for color in range(10) if color != GREY]))
    stamp, anchor, anchor_color = _make_stamp(stamp_h, stamp_w, bg, rng)
    masks = _make_occluders(stamp, anchor, fragment_count, rng)
    if masks is None:
        return None
    placement = _place_regions(
        (canvas_h, canvas_w), stamp.shape, anchor, fragment_count, rng
    )
    if placement is None:
        return None
    fragment_positions, target = placement

    inp = np.full((canvas_h, canvas_w), bg, dtype=int)
    for mask, (r0, c0) in zip(masks, fragment_positions):
        patch = stamp.copy()
        patch[mask] = GREY
        inp[r0 : r0 + stamp_h, c0 : c0 + stamp_w] = patch

    tr, tc = target
    ar, ac = anchor
    inp[tr + ar, tc + ac] = anchor_color
    expected = inp.copy()
    expected[tr : tr + stamp_h, tc : tc + stamp_w] = stamp

    if not _semantic_valid(
        inp,
        expected,
        stamp,
        masks,
        fragment_positions,
        target,
        anchor,
        anchor_color,
        bg,
    ):
        return None
    return inp


def generate_input() -> np.ndarray:
    """Generate a varied, semantically checked invisible-stamp instance."""
    rng = np.random.default_rng()
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        result = _generate_attempt(rng)
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid invisible-stamp input")
