from __future__ import annotations

import random
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, crop_bbox, find_components


BG = int(Color.BLACK)
PATTERN_BLUE = int(Color.BLUE)
MARKER_RED = int(Color.RED)
STAMP_YELLOW = int(Color.YELLOW)
FRAME_GREY = int(Color.GREY)

MAX_ATTEMPTS = 400


def _extract_patch(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    yellow = np.argwhere(grid == STAMP_YELLOW)
    assert yellow.shape == (4, 2), "expected exactly four yellow corner markers"
    frame_bbox = bbox_coords(yellow)
    r0, c0, r1, c1 = frame_bbox
    assert {(int(r), int(c)) for r, c in yellow} == {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}

    crop = crop_bbox(grid, frame_bbox)
    assert crop.shape[0] >= 3 and crop.shape[1] >= 3

    border = np.zeros_like(crop, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    corners = np.zeros_like(crop, dtype=bool)
    corners[0, 0] = True
    corners[0, -1] = True
    corners[-1, 0] = True
    corners[-1, -1] = True

    edge_only = border & ~corners
    assert np.all(crop[corners] == STAMP_YELLOW)
    assert np.all(crop[edge_only] == FRAME_GREY)

    interior = crop[1:-1, 1:-1]
    assert np.all(np.isin(interior, [BG, PATTERN_BLUE]))
    patch = np.where(interior == BG, BG, STAMP_YELLOW).astype(int)
    assert np.any(patch == STAMP_YELLOW)
    return patch, frame_bbox


def _stamp_with_priority(shape: tuple[int, int], patch: np.ndarray, markers: list[tuple[int, int]]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    covered = np.zeros(shape, dtype=bool)
    h, w = patch.shape

    for r, c in sorted(markers):
        assert 0 <= r <= shape[0] - h
        assert 0 <= c <= shape[1] - w
        cov_view = covered[r : r + h, c : c + w]
        out_view = out[r : r + h, c : c + w]
        write_mask = ~cov_view
        out_view[write_mask] = patch[write_mask]
        cov_view[:] = True

    return out


def _foreground_union(shape: tuple[int, int], patch: np.ndarray, markers: list[tuple[int, int]]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    h, w = patch.shape
    fg = patch == STAMP_YELLOW
    for r, c in sorted(markers):
        view = out[r : r + h, c : c + w]
        view[fg] = STAMP_YELLOW
    return out


def _reverse_priority(shape: tuple[int, int], patch: np.ndarray, markers: list[tuple[int, int]]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    covered = np.zeros(shape, dtype=bool)
    h, w = patch.shape
    for r, c in sorted(markers, reverse=True):
        cov_view = covered[r : r + h, c : c + w]
        out_view = out[r : r + h, c : c + w]
        write_mask = ~cov_view
        out_view[write_mask] = patch[write_mask]
        cov_view[:] = True
    return out


def _overlap_slices(h: int, w: int, dr: int, dc: int) -> tuple[slice, slice, slice, slice] | None:
    a_r0 = max(0, dr)
    b_r0 = max(0, -dr)
    rows = min(h - a_r0, h - b_r0)
    a_c0 = max(0, dc)
    b_c0 = max(0, -dc)
    cols = min(w - a_c0, w - b_c0)
    if rows <= 0 or cols <= 0:
        return None
    return (
        slice(a_r0, a_r0 + rows),
        slice(a_c0, a_c0 + cols),
        slice(b_r0, b_r0 + rows),
        slice(b_c0, b_c0 + cols),
    )


def _priority_deltas(patch: np.ndarray) -> list[tuple[int, int]]:
    h, w = patch.shape
    deltas: list[tuple[int, int]] = []
    for dr in range(h):
        for dc in range(-(w - 1), w):
            if dr == 0 and dc <= 0:
                continue
            sl = _overlap_slices(h, w, dr, dc)
            if sl is None:
                continue
            ar, ac, br, bc = sl
            first = patch[ar, ac]
            second = patch[br, bc]
            if np.any((first == BG) & (second == STAMP_YELLOW)):
                deltas.append((dr, dc))
    random.shuffle(deltas)
    return deltas


def _rectangles_overlap(a: tuple[int, int], b: tuple[int, int], h: int, w: int) -> bool:
    return not (a[0] + h <= b[0] or b[0] + h <= a[0] or a[1] + w <= b[1] or b[1] + w <= a[1])


def _adjacent4(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def _grow_connected_patch(height: int, width: int, area: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    start = (random.randrange(height), random.randrange(width))
    mask[start] = True
    cells = [start]

    while len(cells) < area:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < height and 0 <= cc < width and not mask[rr, cc]:
                    frontier.append((rr, cc))
        frontier = list(dict.fromkeys(frontier))
        choice = random.choice(frontier)
        mask[choice] = True
        cells.append(choice)

    return np.where(mask, STAMP_YELLOW, BG).astype(int)


def _make_patch() -> np.ndarray | None:
    height = random.randint(3, 5)
    width = random.randint(3, 5)
    max_area = min(6, height * width - 1)
    if max_area < 3:
        return None

    for _ in range(80):
        area = random.randint(3, max_area)
        patch = _grow_connected_patch(height, width, area)
        if not np.any(patch == BG):
            continue
        if not _priority_deltas(patch):
            continue
        return patch
    return None


def _place_source(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    outer_h, outer_w = h + 2, w + 2
    block = np.zeros((outer_h, outer_w), dtype=int)

    block[0, 0] = STAMP_YELLOW
    block[0, -1] = STAMP_YELLOW
    block[-1, 0] = STAMP_YELLOW
    block[-1, -1] = STAMP_YELLOW
    block[0, 1:-1] = FRAME_GREY
    block[-1, 1:-1] = FRAME_GREY
    block[1:-1, 0] = FRAME_GREY
    block[1:-1, -1] = FRAME_GREY
    block[1:-1, 1:-1] = np.where(patch == STAMP_YELLOW, PATTERN_BLUE, BG)

    grid[top : top + outer_h, left : left + outer_w] = block


def _source_marker_forbidden(
    source_top: int,
    source_left: int,
    outer_h: int,
    outer_w: int,
) -> set[tuple[int, int]]:
    forbidden: set[tuple[int, int]] = set()
    for r in range(source_top - 1, source_top + outer_h + 1):
        for c in range(source_left - 1, source_left + outer_w + 1):
            forbidden.add((r, c))
    return forbidden


def _choose_markers(
    shape: tuple[int, int],
    patch: np.ndarray,
    forbidden: set[tuple[int, int]],
    marker_count: int,
) -> list[tuple[int, int]] | None:
    H, W = shape
    h, w = patch.shape
    valid = [
        (r, c)
        for r in range(H - h + 1)
        for c in range(W - w + 1)
        if (r, c) not in forbidden
    ]
    valid_set = set(valid)
    if len(valid) < marker_count:
        return None

    deltas = _priority_deltas(patch)
    for dr, dc in deltas:
        bases = valid[:]
        random.shuffle(bases)
        for m0 in bases:
            m1 = (m0[0] + dr, m0[1] + dc)
            if m1 not in valid_set:
                continue
            if _adjacent4(m0, m1):
                continue
            if not _rectangles_overlap(m0, m1, h, w):
                continue

            markers = [m0, m1]
            extras = [
                pos
                for pos in valid
                if pos not in markers
                and pos > m0
                and all(not _adjacent4(pos, other) for other in markers)
                and all(not _rectangles_overlap(pos, other, h, w) for other in markers)
            ]
            random.shuffle(extras)
            if len(extras) < marker_count - 2:
                continue
            return sorted(markers + extras[: marker_count - 2])
    return None


def _validate_example(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30

    patch, frame_bbox = _extract_patch(g)
    crop = crop_bbox(g, frame_bbox)
    interior = crop[1:-1, 1:-1]
    blue_components = find_components(
        interior,
        bg=BG,
        colors=[PATTERN_BLUE],
        connectivity=4,
        return_format="dict",
    )
    assert len(blue_components) == 1
    assert 3 <= blue_components[0]["area"] <= 6

    markers = sorted((int(r), int(c)) for r, c in np.argwhere(g == MARKER_RED))
    assert 2 <= len(markers) <= 5
    assert len(markers) == len(set(markers))
    for a, b in combinations(markers, 2):
        assert not _adjacent4(a, b)

    r0, c0, r1, c1 = frame_bbox
    for r, c in markers:
        assert not (r0 - 1 <= r <= r1 + 1 and c0 - 1 <= c <= c1 + 1)

    h, w = patch.shape
    overlapping_pairs = [(a, b) for a, b in combinations(markers, 2) if _rectangles_overlap(a, b, h, w)]
    assert overlapping_pairs, "need at least one overlapping stamp pair"

    priority = _stamp_with_priority(g.shape, patch, markers)
    union = _foreground_union(g.shape, patch, markers)
    reverse = _reverse_priority(g.shape, patch, markers)
    assert not np.array_equal(priority, union), "priority must matter beyond foreground union"
    assert not np.array_equal(priority, reverse), "marker order must affect overlap resolution"
    assert np.array_equal(main(g), priority)


def _gen_attempt() -> np.ndarray | None:
    patch = _make_patch()
    if patch is None:
        return None

    h, w = patch.shape
    outer_h, outer_w = h + 2, w + 2
    min_h = max(14, outer_h + h + 5)
    min_w = max(14, outer_w + w + 5)
    H = random.randint(min_h, min(20, 30))
    W = random.randint(min_w, min(20, 30))

    grid = np.zeros((H, W), dtype=int)

    for _ in range(80):
        source_top = random.randint(1, H - outer_h - 1)
        source_left = random.randint(1, W - outer_w - 1)
        forbidden = _source_marker_forbidden(source_top, source_left, outer_h, outer_w)
        marker_count = random.randint(2, 5)
        markers = _choose_markers(grid.shape, patch, forbidden, marker_count)
        if markers is None:
            continue

        trial = np.zeros_like(grid)
        _place_source(trial, patch, source_top, source_left)
        for r, c in markers:
            trial[r, c] = MARKER_RED

        try:
            _validate_example(trial)
        except AssertionError:
            continue
        return trial

    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the corner-framed template, recolor it yellow, then stamp its interior rectangle at every red marker with row-major overlap priority.

    Input structure:
    - The background is black.
    - Exactly four yellow pixels are the corners of a grey 1-pixel frame.
    - Inside that frame sits a small blue 4-connected pattern on a black interior, and separate red single-pixel markers lie elsewhere in the grid.

    Rule:
    - Use the four yellow corner markers to crop the frame rectangle.
    - Recolor every non-black pixel inside that crop to yellow, then remove the outer 1-pixel border to get the yellow/black stamp template.
    - Start from an all-black output grid of the same size.
    - Stamp the whole template rectangle at each red marker, aligning the template's top-left cell to the marker position.
    - If stamp rectangles overlap, resolve each overlapped cell by marker priority: smaller row wins, and ties break by smaller column; later stamps cannot overwrite even a higher-priority black template cell.

    Key insight:
    - The framed object is only a template factory: the output depends on extracting the framed interior rectangle and treating overlap as whole-rectangle priority, not as a simple union of yellow foreground pixels.
    """
    g = np.asarray(grid, dtype=int)
    patch, _frame_bbox = _extract_patch(g)
    markers = [(int(r), int(c)) for r, c in np.argwhere(g == MARKER_RED)]
    return _stamp_with_priority(g.shape, patch, markers)


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
