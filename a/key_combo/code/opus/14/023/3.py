from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
GREY = 5
TEAL = 8
MARKER_PALETTE = (1, 2, 3, 4, 6, 7)

GRID_SHORT_RANGE = (13, 18)
GRID_LONG_RANGE = (16, 24)
MARKER_COUNT_RANGE = (4, 6)
STEP_CHOICES = (3, 4, 5)
END_PAD_CHOICES = (1, 2, 3)
GEN_ATTEMPTS = 500


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(
    dst: np.ndarray,
    patch: np.ndarray,
    r0: int,
    c0: int,
    *,
    transparent: int | None = None,
) -> None:
    h, w = patch.shape
    r1 = r0 + h
    c1 = c0 + w
    assert 0 <= r0 <= r1 <= dst.shape[0]
    assert 0 <= c0 <= c1 <= dst.shape[1]
    view = dst[r0:r1, c0:c1]
    if transparent is None:
        view[:] = patch
    else:
        mask = patch != int(transparent)
        view[mask] = patch[mask]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    remaining = {tuple(map(int, rc)) for rc in coords}
    stack = [next(iter(remaining))]
    seen = set()
    while stack:
        cell = stack.pop()
        if cell in seen:
            continue
        seen.add(cell)
        for nbr in _neighbors4(*cell):
            if nbr in remaining and nbr not in seen:
                stack.append(nbr)
    return seen == remaining


def _template_library() -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for bits in range(1 << 9):
        mask = np.array([(bits >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
        count = int(mask.sum())
        if count < 3 or count > 5:
            continue
        if not mask[1, 1]:
            continue
        if not _is_connected(mask):
            continue
        box = _bbox(mask)
        if box != (0, 0, 2, 2):
            continue
        masks.append(mask)
    assert masks
    return masks


TEMPLATE_MASKS = _template_library()


def _parse_scene(grid: np.ndarray) -> dict:
    grey_mask = grid == GREY
    grey_box = _bbox(grey_mask)
    assert grey_box is not None, "missing grey template"
    grey_count = int(grey_mask.sum())
    assert 3 <= grey_count <= 5, "template must contain 3-5 grey pixels"
    template = _crop_bbox(grey_mask, grey_box)
    assert _is_connected(template), "template must be 4-connected"

    teal_coords = np.argwhere(grid == TEAL)
    assert teal_coords.shape[0] == 2, "expected exactly two teal endpoints"

    markers: list[tuple[int, int, int]] = []
    for color in MARKER_PALETTE:
        coords = np.argwhere(grid == color)
        if coords.shape[0] == 0:
            continue
        assert coords.shape[0] == 1, "every marker color must appear exactly once"
        r, c = map(int, coords[0])
        markers.append((r, c, int(color)))
    assert 4 <= len(markers) <= 6, "expected 4-6 colored markers"

    rows = {r for r, _c, _color in markers}
    cols = {c for _r, c, _color in markers}
    horizontal = len(rows) == 1
    vertical = len(cols) == 1
    assert horizontal != vertical, "markers must share one row xor one column"

    teal_rows = {int(r) for r, _c in teal_coords}
    teal_cols = {int(c) for _r, c in teal_coords}
    if horizontal:
        row = next(iter(rows))
        assert teal_rows == {row}, "teal endpoints must lie on the marker row"
        markers.sort(key=lambda item: item[1])
        teal_sorted = sorted(map(int, teal_coords[:, 1]))
        positions = [c for _r, c, _color in markers]
        span0, span1 = teal_sorted
        assert span0 < positions[0] and positions[-1] < span1
    else:
        col = next(iter(cols))
        assert teal_cols == {col}, "teal endpoints must lie on the marker column"
        markers.sort(key=lambda item: item[0])
        teal_sorted = sorted(map(int, teal_coords[:, 0]))
        positions = [r for r, _c, _color in markers]
        span0, span1 = teal_sorted
        assert span0 < positions[0] and positions[-1] < span1

    gaps = [b - a for a, b in zip(positions, positions[1:], strict=False)]
    assert gaps and min(gaps) >= 3, "markers need at least two blank cells between them"

    return {
        "template_box": grey_box,
        "template_mask": template,
        "markers": markers,
        "horizontal": horizontal,
        "span": (int(span0), int(span1)),
    }


def _weave_box(shape: tuple[int, int], parsed: dict) -> tuple[int, int, int, int]:
    th, tw = map(int, parsed["template_mask"].shape)
    (span0, span1) = parsed["span"]
    if parsed["horizontal"]:
        row = int(parsed["markers"][0][0])
        r0 = row - th // 2
        r1 = r0 + th - 1
        c0 = span0
        c1 = span1
    else:
        col = int(parsed["markers"][0][1])
        r0 = span0
        r1 = span1
        c0 = col - tw // 2
        c1 = c0 + tw - 1
    assert 0 <= r0 <= r1 < shape[0]
    assert 0 <= c0 <= c1 < shape[1]
    return int(r0), int(c0), int(r1), int(c1)


def _render_output_from_parsed(grid: np.ndarray, parsed: dict) -> np.ndarray:
    out = np.full(grid.shape, BG, dtype=int)
    weave = _weave_box(grid.shape, parsed)
    r0, c0, r1, c1 = weave
    color_a = int(parsed["markers"][0][2])
    color_b = int(parsed["markers"][1][2])
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            out[r, c] = color_a if ((r - r0) + (c - c0)) % 2 == 0 else color_b

    mask = parsed["template_mask"]
    th, tw = mask.shape
    center_r = th // 2
    center_c = tw // 2
    for r, c, color in parsed["markers"]:
        patch = np.full((th, tw), BG, dtype=int)
        patch[mask] = int(color)
        _paste(out, patch, r - center_r, c - center_c, transparent=BG)

    template_box = parsed["template_box"]
    template_patch = np.full(parsed["template_mask"].shape, BG, dtype=int)
    template_patch[parsed["template_mask"]] = GREY
    _paste(out, template_patch, template_box[0], template_box[1], transparent=BG)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the marker span with a two-color weave and stamp the grey template at every marker.

    Input structure:
    - A black background contains one small 4-connected grey template of 3-5 pixels.
    - Four to six distinct single-pixel colored markers lie on one common row or one common column.
    - Two teal pixels lie on the same row or column outside the markers and define the weave span.

    Rule:
    - Crop the grey template to its bounding box and treat the grey cells as a stamp mask.
    - Sort the colored markers along their shared axis; the first two sorted marker colors define a checkerboard weave inside the teal-delimited rectangle centered on the marker line and sized by the template bbox thickness.
    - Recolor the template mask to each marker color and stamp it back with the marker pixel as the template-bbox center, overwriting the weave where the mask is filled; keep the original grey template unchanged in place.

    Key insight:
    - The same marker pixels play three roles at once: their order chooses the weave colors, their positions choose the stamp anchors, and their own colors recolor the stamped copies.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    parsed = _parse_scene(g)
    return _render_output_from_parsed(g, parsed)


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 0) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _stamp_box(center_r: int, center_c: int, size: int = 3) -> tuple[int, int, int, int]:
    half = size // 2
    return center_r - half, center_c - half, center_r + half, center_c + half


def _line_layout(axis_len: int, marker_count: int) -> tuple[int, list[int], int, int]:
    candidates = []
    for left_pad in END_PAD_CHOICES:
        for right_pad in END_PAD_CHOICES:
            for steps in itertools.product(STEP_CHOICES, repeat=marker_count - 1):
                total = left_pad + sum(steps) + right_pad + 1
                if total > axis_len:
                    continue
                candidates.append((left_pad, list(steps), right_pad, total))
    assert candidates
    left_pad, steps, right_pad, total = random.choice(candidates)
    start = random.randint(0, axis_len - total)
    return start, steps, left_pad, right_pad


def _random_template_mask() -> np.ndarray:
    return np.array(random.choice(TEMPLATE_MASKS), dtype=bool)


def _layout_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    horizontal = bool(random.getrandbits(1))

    if horizontal:
        h = random.randint(*GRID_SHORT_RANGE)
        w = random.randint(*GRID_LONG_RANGE)
        feasible_counts = [n for n in range(MARKER_COUNT_RANGE[0], MARKER_COUNT_RANGE[1] + 1) if any(
            left + sum(steps) + right + 1 <= w
            for left in END_PAD_CHOICES
            for right in END_PAD_CHOICES
            for steps in itertools.product((min(STEP_CHOICES),), repeat=n - 1)
        )]
        marker_count = random.choice(feasible_counts)
        line_row = random.randint(4, h - 5)
        start, steps, left_pad, right_pad = _line_layout(w, marker_count)
    else:
        h = random.randint(*GRID_LONG_RANGE)
        w = random.randint(*GRID_SHORT_RANGE)
        feasible_counts = [n for n in range(MARKER_COUNT_RANGE[0], MARKER_COUNT_RANGE[1] + 1) if any(
            left + sum(steps) + right + 1 <= h
            for left in END_PAD_CHOICES
            for right in END_PAD_CHOICES
            for steps in itertools.product((min(STEP_CHOICES),), repeat=n - 1)
        )]
        marker_count = random.choice(feasible_counts)
        line_col = random.randint(4, w - 5)
        start, steps, left_pad, right_pad = _line_layout(h, marker_count)

    marker_colors = random.sample(MARKER_PALETTE, marker_count)
    template_mask = _random_template_mask()
    template_patch = np.full((3, 3), BG, dtype=int)
    template_patch[template_mask] = GREY

    grid = np.full((h, w), BG, dtype=int)
    positions = [start + left_pad]
    for step in steps:
        positions.append(positions[-1] + step)
    span0 = start
    span1 = positions[-1] + right_pad

    marker_boxes = []
    if horizontal:
        markers = [(line_row, pos, color) for pos, color in zip(positions, marker_colors, strict=False)]
        weave_box = (line_row - 1, span0, line_row + 1, span1)
        for r, c, _color in markers:
            marker_boxes.append(_stamp_box(r, c))
    else:
        markers = [(pos, line_col, color) for pos, color in zip(positions, marker_colors, strict=False)]
        weave_box = (span0, line_col - 1, span1, line_col + 1)
        for r, c, _color in markers:
            marker_boxes.append(_stamp_box(r, c))

    corners = [
        (0, 0),
        (0, w - 3),
        (h - 3, 0),
        (h - 3, w - 3),
    ]
    random.shuffle(corners)
    template_top_left = None
    for tr, tc in corners:
        template_box = (tr, tc, tr + 2, tc + 2)
        if not _boxes_separated(template_box, weave_box, gap=1):
            continue
        if any(not _boxes_separated(template_box, box, gap=1) for box in marker_boxes):
            continue
        template_top_left = (tr, tc)
        break
    if template_top_left is None:
        return None

    tr, tc = template_top_left
    _paste(grid, template_patch, tr, tc, transparent=BG)
    for r, c, color in markers:
        grid[r, c] = int(color)
    if horizontal:
        grid[line_row, span0] = TEAL
        grid[line_row, span1] = TEAL
    else:
        grid[span0, line_col] = TEAL
        grid[span1, line_col] = TEAL

    expected = _expected_from_layout(grid, template_mask, markers, horizontal, (span0, span1))
    if not _semantic_checks(grid, expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def _expected_from_layout(
    grid: np.ndarray,
    template_mask: np.ndarray,
    markers: list[tuple[int, int, int]],
    horizontal: bool,
    span: tuple[int, int],
) -> np.ndarray:
    out = np.full(grid.shape, BG, dtype=int)
    span0, span1 = span
    color_a = int(markers[0][2])
    color_b = int(markers[1][2])
    if horizontal:
        r0 = markers[0][0] - 1
        c0 = span0
        for rr in range(3):
            for cc in range(span1 - span0 + 1):
                out[r0 + rr, c0 + cc] = color_a if (rr + cc) % 2 == 0 else color_b
    else:
        r0 = span0
        c0 = markers[0][1] - 1
        for rr in range(span1 - span0 + 1):
            for cc in range(3):
                out[r0 + rr, c0 + cc] = color_a if (rr + cc) % 2 == 0 else color_b

    patch = np.full((3, 3), BG, dtype=int)
    patch[template_mask] = GREY
    grey_box = _bbox(grid == GREY)
    assert grey_box is not None
    for r, c, color in markers:
        stamp = np.full((3, 3), BG, dtype=int)
        stamp[template_mask] = int(color)
        _paste(out, stamp, r - 1, c - 1, transparent=BG)
    _paste(out, patch, grey_box[0], grey_box[1], transparent=BG)
    return out


def _semantic_checks(grid: np.ndarray, expected: np.ndarray) -> bool:
    try:
        parsed = _parse_scene(grid)
    except AssertionError:
        return False

    weave_box = _weave_box(grid.shape, parsed)
    wr0, wc0, wr1, wc1 = weave_box
    covered = np.zeros(grid.shape, dtype=bool)
    for r, c, _color in parsed["markers"]:
        covered[r - 1 : r + 2, c - 1 : c + 2] |= parsed["template_mask"]

    weave_mask = np.zeros(grid.shape, dtype=bool)
    weave_mask[wr0 : wr1 + 1, wc0 : wc1 + 1] = True
    exposed_weave = weave_mask & (~covered)
    if not np.any(exposed_weave):
        return False

    color_a = int(parsed["markers"][0][2])
    color_b = int(parsed["markers"][1][2])
    overwrite_effect = False
    for r, c, color in parsed["markers"]:
        for dr, dc in np.argwhere(parsed["template_mask"]):
            rr = r + int(dr) - 1
            cc = c + int(dc) - 1
            base = color_a if ((rr - wr0) + (cc - wc0)) % 2 == 0 else color_b
            if base != int(color):
                overwrite_effect = True
                break
        if overwrite_effect:
            break
    if not overwrite_effect:
        return False

    teal_positions = np.argwhere(grid == TEAL)
    for r, c in teal_positions:
        if expected[int(r), int(c)] == TEAL:
            return False

    grey_in_output = np.argwhere(expected == GREY)
    grey_in_input = np.argwhere(grid == GREY)
    if set(map(tuple, grey_in_output)) != set(map(tuple, grey_in_input)):
        return False

    if np.array_equal(grid, expected):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        result = _layout_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate a valid input")
