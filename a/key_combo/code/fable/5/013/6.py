from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


# D4 matrices in (row, column) coordinates.  The first four preserve
# orientation (rotations); the last four reverse it (reflected rotations).
D4: tuple[tuple[int, int, int, int], ...] = (
    (1, 0, 0, 1),
    (0, 1, -1, 0),
    (-1, 0, 0, -1),
    (0, -1, 1, 0),
    (1, 0, 0, -1),
    (0, -1, -1, 0),
    (-1, 0, 0, 1),
    (0, 1, 1, 0),
)

BACKGROUND_CHOICES = (0, 1, 3, 4, 6, 7, 8, 9)
HANDLE_COLOR = 2
SEPARATOR_COLOR = 5
MAX_GENERATION_ATTEMPTS = 500

Coord = tuple[int, int]
Transform = tuple[int, int, int, int]


def _background(grid: np.ndarray) -> int:
    """Return the most common color, following arc_utils.get_bg."""
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_coords(points: set[Coord]) -> tuple[int, int, int, int]:
    """Inclusive (r0, c0, r1, c1) bbox in the arc_utils convention."""
    assert points
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _components4(points: set[Coord]) -> list[set[Coord]]:
    remaining = set(points)
    result: list[set[Coord]] = []
    while remaining:
        seed = remaining.pop()
        component = {seed}
        queue: deque[Coord] = deque([seed])
        while queue:
            r, c = queue.popleft()
            for nxt in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if nxt in remaining:
                    remaining.remove(nxt)
                    component.add(nxt)
                    queue.append(nxt)
        result.append(component)
    return result


def _apply(point: Coord, transform: Transform) -> Coord:
    r, c = point
    a, b, d, e = transform
    return a * r + b * c, d * r + e * c


def _transform_points(points: set[Coord], transform: Transform) -> set[Coord]:
    return {_apply(point, transform) for point in points}


def _ring_neighbors(points: set[Coord], anchor: Coord) -> set[Coord]:
    ar, ac = anchor
    return {
        (r, c)
        for r, c in points
        if max(abs(r - ar), abs(c - ac)) == 1
    }


def _relative(points: set[Coord], anchor: Coord) -> set[Coord]:
    ar, ac = anchor
    return {(r - ar, c - ac) for r, c in points}


def _matching_transforms(source: set[Coord], target: set[Coord]) -> list[Transform]:
    return [transform for transform in D4 if _transform_points(source, transform) == target]


def _separator_column(grid: np.ndarray) -> int:
    candidates = [
        c
        for c in range(grid.shape[1])
        if bool(np.all(grid[:, c] == SEPARATOR_COLOR))
    ]
    assert len(candidates) == 1, "expected one full grey separator column"
    sep = int(candidates[0])
    assert 0 < sep < grid.shape[1] - 1, "separator must have two nonempty panes"
    return sep


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct transformed copies of a left-hand glyph from three-point fiducials on the right.

    Input structure:
    - The most common color is the background, and one full grey column separates two panes.
    - The left pane contains one complete asymmetric monochrome glyph and one red handle pixel touching exactly two glyph pixels in its 8-neighborhood.
    - The right pane contains several groups, each consisting of a red handle and exactly the corresponding two glyph-color neighbors, with groups spatially separate.

    Rule:
    - Measure the complete glyph and its two local fiducial offsets relative to the left red handle.
    - For each right handle, choose the unique rotation or reflected rotation that maps both template offsets onto its two visible glyph pixels.
    - Draw the entire glyph under that transform about the right handle; preserve the left pane, handles, separator, and all existing pixels.

    Key insight:
    - A handle plus either one neighboring glyph pixel leaves a reflection ambiguity, while the handle plus both differently directed neighbors uniquely identifies one of all eight D4 orientations.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert 3 <= height <= 30 and 5 <= width <= 30, "grid must satisfy ARC bounds"

    bg = _background(g)
    assert bg not in (HANDLE_COLOR, SEPARATOR_COLOR), "marker colors cannot be background"
    sep = _separator_column(g)

    ordinary_colors = set(map(int, np.unique(g))) - {
        bg,
        HANDLE_COLOR,
        SEPARATOR_COLOR,
    }
    assert len(ordinary_colors) == 1, "expected one monochrome glyph color"
    glyph_color = int(next(iter(ordinary_colors)))

    left_handles = {
        (int(r), int(c))
        for r, c in np.argwhere(g[:, :sep] == HANDLE_COLOR)
    }
    assert len(left_handles) == 1, "left pane must contain exactly one handle"
    template_handle = next(iter(left_handles))

    template_points = {
        (int(r), int(c))
        for r, c in np.argwhere(g[:, :sep] == glyph_color)
    }
    assert len(template_points) >= 6, "template must be a complete, nontrivial glyph"
    assert len(_components4(template_points)) == 1, "template glyph must be 4-connected"

    local_absolute = _ring_neighbors(template_points, template_handle)
    assert len(local_absolute) == 2, "template handle must touch exactly two glyph pixels"
    local_offsets = _relative(local_absolute, template_handle)
    local_images = {frozenset(_transform_points(local_offsets, t)) for t in D4}
    assert len(local_images) == 8, "the two-offset fiducial must distinguish all D4 transforms"
    template_offsets = _relative(template_points, template_handle)

    right_handles = {
        (int(r), int(c + sep + 1))
        for r, c in np.argwhere(g[:, sep + 1 :] == HANDLE_COLOR)
    }
    assert len(right_handles) >= 2, "expected several right-hand fiducial groups"
    right_fragments = {
        (int(r), int(c + sep + 1))
        for r, c in np.argwhere(g[:, sep + 1 :] == glyph_color)
    }

    assigned: set[Coord] = set()
    solved: list[tuple[Coord, Transform]] = []
    for handle in sorted(right_handles):
        visible = _ring_neighbors(right_fragments, handle)
        assert len(visible) == 2, "each right handle must touch exactly two fragment pixels"
        assert not (assigned & visible), "a fragment pixel cannot belong to two handles"
        assigned |= visible
        observed_offsets = _relative(visible, handle)
        matches = _matching_transforms(local_offsets, observed_offsets)
        assert len(matches) == 1, "right fiducial must select one unique D4 transform"
        solved.append((handle, matches[0]))
    assert assigned == right_fragments, "every right glyph pixel must belong to one fiducial"

    out = g.copy()
    for (anchor_r, anchor_c), transform in solved:
        for dr, dc in _transform_points(template_offsets, transform):
            r, c = anchor_r + dr, anchor_c + dc
            assert 0 <= r < height and sep < c < width, "reconstructed glyph left its pane"
            assert int(out[r, c]) in (bg, glyph_color), "reconstructed glyph hit a marker"
            out[r, c] = glyph_color
    return out


def _random_template() -> set[Coord] | None:
    """Grow a varied 4-connected glyph above a handle at relative (0, 0)."""
    h = random.randint(4, 6)
    w = random.randint(4, 6)
    anchor_c = random.randint(1, w - 2)
    forced = {(h - 1, anchor_c), (h - 1, anchor_c + 1)}
    cells = set(forced)
    target_area = random.randint(8, min(14, h * w - (w - 2)))

    while len(cells) < target_area:
        frontier: set[Coord] = set()
        for r, c in cells:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if not (0 <= nr < h and 0 <= nc < w) or (nr, nc) in cells:
                    continue
                # The bottom row contains only the two fiducial pixels.
                if nr == h - 1 and (nr, nc) not in forced:
                    continue
                frontier.add((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(tuple(frontier)))

    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    if max(rows) - min(rows) + 1 < 3 or max(cols) - min(cols) + 1 < 3:
        return None

    relative = {(r - h, c - anchor_c) for r, c in cells}
    expected_local = {(-1, 0), (-1, 1)}
    if _ring_neighbors(relative, (0, 0)) != expected_local:
        return None
    if len(_components4(relative)) != 1:
        return None
    if len({frozenset(_transform_points(relative, t)) for t in D4}) != 8:
        return None
    return relative


def _bbox_at(offsets: set[Coord], anchor: Coord) -> tuple[int, int, int, int]:
    ar, ac = anchor
    return _bbox_coords({(ar + r, ac + c) for r, c in offsets} | {anchor})


def _rectangles_clear(
    box: tuple[int, int, int, int],
    existing: list[tuple[int, int, int, int]],
) -> bool:
    r0, c0, r1, c1 = box
    for er0, ec0, er1, ec1 in existing:
        # Require at least one full background row or column between groups.
        if not (r1 + 1 < er0 or er1 + 1 < r0 or c1 + 1 < ec0 or ec1 + 1 < c0):
            return False
    return True


def _anchor_candidates(
    offsets: set[Coord],
    height: int,
    c_min: int,
    c_max: int,
) -> list[Coord]:
    r0, c0, r1, c1 = _bbox_coords(offsets | {(0, 0)})
    candidates = [
        (r, c)
        for r in range(1 - r0, height - 1 - r1)
        for c in range(c_min + 1 - c0, c_max - c1)
    ]
    random.shuffle(candidates)
    return candidates


def _choose_transforms(count: int) -> list[Transform]:
    rotations_n = random.randint(1, count - 1)
    rotations = random.sample(list(D4[:4]), rotations_n)
    reflections = random.sample(list(D4[4:]), count - rotations_n)
    chosen = rotations + reflections
    random.shuffle(chosen)
    return chosen


def _independent_expected(
    inp: np.ndarray,
    glyph_color: int,
    template_offsets: set[Coord],
    placements: list[tuple[Coord, Transform]],
) -> np.ndarray:
    expected = inp.copy()
    for (ar, ac), transform in placements:
        for dr, dc in _transform_points(template_offsets, transform):
            expected[ar + dr, ac + dc] = glyph_color
    return expected


def _validate_generated(
    inp: np.ndarray,
    expected: np.ndarray,
    bg: int,
    glyph_color: int,
    sep: int,
    template_offsets: set[Coord],
    placements: list[tuple[Coord, Transform]],
) -> bool:
    """Check puzzle semantics, not merely generate/solve self-consistency."""
    if inp.ndim != 2 or max(inp.shape) > 30 or expected.shape != inp.shape:
        return False
    if _background(inp) != bg or _separator_column(inp) != sep:
        return False
    if not np.array_equal(inp[:, : sep + 1], expected[:, : sep + 1]):
        return False

    local = _ring_neighbors(template_offsets, (0, 0))
    if local != {(-1, 0), (-1, 1)}:
        return False
    if len({frozenset(_transform_points(local, t)) for t in D4}) != 8:
        return False

    # With the handle and just one glyph point, every observed point has two
    # candidate transforms.  Both glyph points together must have exactly one.
    for anchor, chosen in placements:
        observed_pair = _transform_points(local, chosen)
        if len(_matching_transforms(local, observed_pair)) != 1:
            return False
        for observed_point in observed_pair:
            # The surviving glyph point has no identity label, so either of the
            # two template neighbors may map to it.
            singleton_matches = [
                t for t in D4 if observed_point in _transform_points(local, t)
            ]
            if len(singleton_matches) < 2:
                return False

        ar, ac = anchor
        target = {(ar + r, ac + c) for r, c in _transform_points(template_offsets, chosen)}
        visible = {(ar + r, ac + c) for r, c in observed_pair}
        if len(target - visible) != len(template_offsets) - 2:
            return False
        if any(int(expected[r, c]) != glyph_color for r, c in target):
            return False

    transforms = [t for _, t in placements]
    if len(set(transforms)) != len(transforms):
        return False
    if not any(t in D4[:4] for t in transforms) or not any(t in D4[4:] for t in transforms):
        return False

    expected_new = len(placements) * (len(template_offsets) - 2)
    actual_new = int(np.sum((inp == bg) & (expected == glyph_color)))
    if actual_new != expected_new or actual_new <= 0 or np.array_equal(inp, expected):
        return False

    try:
        solved = main(inp)
    except (AssertionError, ValueError, IndexError):
        return False
    return bool(np.array_equal(solved, expected))


def _gen_attempt() -> np.ndarray | None:
    template_offsets = _random_template()
    if template_offsets is None:
        return None

    tr0, tc0, tr1, tc1 = _bbox_coords(template_offsets | {(0, 0)})
    template_h = tr1 - tr0 + 1
    template_w = tc1 - tc0 + 1
    left_width_min = max(8, template_w + 3)
    left_width_max = min(11, 27 - 14)
    if left_width_min > left_width_max:
        return None
    left_width = random.randint(left_width_min, left_width_max)
    right_width = random.randint(14, min(18, 29 - left_width))
    height = random.randint(max(22, template_h + 5), 30)
    width = left_width + 1 + right_width
    sep = left_width

    bg = random.choice(BACKGROUND_CHOICES)
    glyph_color = random.choice([c for c in BACKGROUND_CHOICES if c != bg])
    inp = np.full((height, width), bg, dtype=int)
    inp[:, sep] = SEPARATOR_COLOR

    left_candidates = _anchor_candidates(template_offsets, height, 0, sep - 1)
    if not left_candidates:
        return None
    template_anchor = random.choice(left_candidates)
    lar, lac = template_anchor
    for dr, dc in template_offsets:
        inp[lar + dr, lac + dc] = glyph_color
    inp[lar, lac] = HANDLE_COLOR

    group_count = random.randint(3, 4)
    transforms = _choose_transforms(group_count)
    placements: list[tuple[Coord, Transform]] = []
    boxes: list[tuple[int, int, int, int]] = []
    local = _ring_neighbors(template_offsets, (0, 0))

    # Place complete-output bboxes, although only the three-point groups are
    # rendered in the input.  This prevents hidden reconstruction collisions.
    transforms.sort(
        key=lambda t: -(
            (_bbox_coords(_transform_points(template_offsets, t))[2]
             - _bbox_coords(_transform_points(template_offsets, t))[0] + 1)
            *
            (_bbox_coords(_transform_points(template_offsets, t))[3]
             - _bbox_coords(_transform_points(template_offsets, t))[1] + 1)
        )
    )
    for transform in transforms:
        transformed = _transform_points(template_offsets, transform)
        candidates = _anchor_candidates(transformed, height, sep + 1, width - 1)
        chosen_anchor = None
        chosen_box = None
        for anchor in candidates:
            box = _bbox_at(transformed, anchor)
            if _rectangles_clear(box, boxes):
                chosen_anchor = anchor
                chosen_box = box
                break
        if chosen_anchor is None or chosen_box is None:
            return None
        placements.append((chosen_anchor, transform))
        boxes.append(chosen_box)

    for (ar, ac), transform in placements:
        inp[ar, ac] = HANDLE_COLOR
        for dr, dc in _transform_points(local, transform):
            if int(inp[ar + dr, ac + dc]) != bg:
                return None
            inp[ar + dr, ac + dc] = glyph_color

    expected = _independent_expected(inp, glyph_color, template_offsets, placements)
    if not _validate_generated(
        inp,
        expected,
        bg,
        glyph_color,
        sep,
        template_offsets,
        placements,
    ):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid fiducial reconstruction puzzle")
