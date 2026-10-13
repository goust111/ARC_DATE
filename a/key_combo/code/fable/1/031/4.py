from __future__ import annotations

import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, colors_present, components_mask, get_bg


BRIDGE_COLOR = Color.TEAL
BG_COLOR = Color.BLACK
Rect = tuple[int, int, int, int]
Bridge = tuple[int, Rect]

GEN_ATTEMPTS = 400
GEN_CANVAS_TRIES = 60
GEN_PLACEMENT_TRIES = 160
GEN_PADDING = 1
GEN_MIN_SIDE = 18
GEN_MAX_SIDE = 30
GEN_EXTRA_PAIR_RANGE = (1, 3)
GEN_SINGLETON_RANGE = (0, 2)
GEN_CROSS_GAP_POOL = tuple(range(4, 10))
GEN_PAIR_GAP_POOL = tuple(range(1, 11))
GEN_FOREGROUND_COLORS = [c for c in Color.ALL_COLORS if c not in {BG_COLOR, BRIDGE_COLOR}]


def _rectangles_by_color(grid: np.ndarray, bg: int) -> dict[int, list[Rect]]:
    rectangles: dict[int, list[Rect]] = defaultdict(list)
    for color in colors_present(grid, exclude={bg}):
        assert color != BRIDGE_COLOR, "bridge color must be absent from the input"
        for coords in components_mask(grid == color, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(coords)
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            assert int(coords.shape[0]) == h * w, "foreground objects must be solid rectangles"
            assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color), "rectangle bbox must be solid"
            rectangles[int(color)].append((int(r0), int(c0), int(r1), int(c1)))
    assert rectangles, "expected at least one foreground rectangle"
    return dict(rectangles)


def _candidate_bridge(grid: np.ndarray, bg: int, a: Rect, b: Rect) -> Bridge | None:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    candidates: list[Bridge] = []

    shared_r0 = max(ar0, br0)
    shared_r1 = min(ar1, br1)
    shared_c0 = max(ac0, bc0)
    shared_c1 = min(ac1, bc1)
    assert shared_r1 < shared_r0 or shared_c1 < shared_c0, "solid rectangles cannot overlap"

    if shared_r1 - shared_r0 + 1 >= 3 and (ac1 < bc0 or bc1 < ac0):
        if ac1 < bc0:
            gap_c0, gap_c1 = ac1 + 1, bc0 - 1
        else:
            gap_c0, gap_c1 = bc1 + 1, ac0 - 1
        if gap_c0 <= gap_c1 and np.all(grid[shared_r0 : shared_r1 + 1, gap_c0 : gap_c1 + 1] == bg):
            paint = (shared_r0 + 1, gap_c0, shared_r1 - 1, gap_c1)
            candidates.append((gap_c1 - gap_c0 + 1, paint))

    if shared_c1 - shared_c0 + 1 >= 3 and (ar1 < br0 or br1 < ar0):
        if ar1 < br0:
            gap_r0, gap_r1 = ar1 + 1, br0 - 1
        else:
            gap_r0, gap_r1 = br1 + 1, ar0 - 1
        if gap_r0 <= gap_r1 and np.all(grid[gap_r0 : gap_r1 + 1, shared_c0 : shared_c1 + 1] == bg):
            paint = (gap_r0, shared_c0 + 1, gap_r1, shared_c1 - 1)
            candidates.append((gap_r1 - gap_r0 + 1, paint))

    assert len(candidates) <= 1, "a rectangle pair should face in at most one direction"
    return candidates[0] if candidates else None


def _all_candidates(grid: np.ndarray, bg: int) -> list[Bridge]:
    candidates: list[Bridge] = []
    for rectangles in _rectangles_by_color(grid, bg).values():
        for i, rect_a in enumerate(rectangles):
            for rect_b in rectangles[i + 1 :]:
                bridge = _candidate_bridge(grid, bg, rect_a, rect_b)
                if bridge is not None:
                    candidates.append(bridge)
    return candidates


def _accepted_bridges(shape: tuple[int, int], candidates: list[Bridge]) -> tuple[list[Bridge], list[Bridge]]:
    used = np.zeros(shape, dtype=bool)
    accepted: list[Bridge] = []
    rejected: list[Bridge] = []
    for bridge in sorted(candidates, key=lambda item: (item[0], *item[1])):
        _gap, (r0, c0, r1, c1) = bridge
        assert r0 <= r1 and c0 <= c1, "bridge body must be non-empty"
        if np.any(used[r0 : r1 + 1, c0 : c1 + 1]):
            rejected.append(bridge)
            continue
        used[r0 : r1 + 1, c0 : c1 + 1] = True
        accepted.append(bridge)
    return accepted, rejected


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint teal bridges in ascending gap order, skipping any bridge whose body is already occupied by a shorter bridge.

    Input structure:
    - The dominant background surrounds several solid axis-aligned monochromatic rectangles.
    - Same-color rectangles may face each other horizontally or vertically across an all-background gap.
    - A valid pair shares at least three rows or three columns of span, and teal is absent from the input.

    Rule:
    - Build every candidate bridge between same-color rectangles that face across a pure background gap.
    - The bridge body fills the full gap and uses the shared span eroded by one cell on both ends.
    - Sort all candidate bridges by gap length from shortest to longest; paint a bridge teal only if its body does not overlap any already painted bridge cell.
    - Leave every original rectangle unchanged.

    Key insight:
    - Pairing alone is insufficient: some geometrically valid bridges disappear because a shorter crossing bridge claims the shared cells first.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = get_bg(g)
    assert bg != BRIDGE_COLOR, "background cannot be the bridge color"
    out = g.copy()

    candidates = _all_candidates(g, bg)
    accepted, _rejected = _accepted_bridges(g.shape, candidates)
    for _gap, (r0, c0, r1, c1) in accepted:
        out[r0 : r1 + 1, c0 : c1 + 1] = BRIDGE_COLOR
    return out


def _outer_padding() -> tuple[int, int, int, int]:
    return tuple(random.randint(0, 2) for _ in range(4))


def _build_cross_patch(vertical_color: int, horizontal_color: int, v_gap: int, h_gap: int) -> np.ndarray:
    top_h = random.randint(3, 6)
    bottom_h = random.randint(3, 6)
    left_w = random.randint(3, 6)
    right_w = random.randint(3, 6)
    v_rect_w = random.randint(3, min(6, h_gap))
    h_rect_h = random.randint(3, min(6, v_gap))
    x = random.randint(0, h_gap - v_rect_w)
    y = random.randint(0, v_gap - h_rect_h)

    body_h = top_h + v_gap + bottom_h
    body_w = left_w + h_gap + right_w
    pad_t, pad_b, pad_l, pad_r = _outer_padding()
    patch = np.full((pad_t + body_h + pad_b, pad_l + body_w + pad_r), BG_COLOR, dtype=int)

    v_c0 = pad_l + left_w + x
    patch[pad_t : pad_t + top_h, v_c0 : v_c0 + v_rect_w] = vertical_color
    b_r0 = pad_t + top_h + v_gap
    patch[b_r0 : b_r0 + bottom_h, v_c0 : v_c0 + v_rect_w] = vertical_color

    h_r0 = pad_t + top_h + y
    patch[h_r0 : h_r0 + h_rect_h, pad_l : pad_l + left_w] = horizontal_color
    h_c0 = pad_l + left_w + h_gap
    patch[h_r0 : h_r0 + h_rect_h, h_c0 : h_c0 + right_w] = horizontal_color

    return patch


def _build_pair_patch(color: int, gap: int) -> np.ndarray:
    pad_t, pad_b, pad_l, pad_r = _outer_padding()
    if random.random() < 0.5:
        span_h = random.randint(3, 6)
        left_w = random.randint(3, 6)
        right_w = random.randint(3, 6)
        patch = np.full((pad_t + span_h + pad_b, pad_l + left_w + gap + right_w + pad_r), BG_COLOR, dtype=int)
        patch[pad_t : pad_t + span_h, pad_l : pad_l + left_w] = color
        c0 = pad_l + left_w + gap
        patch[pad_t : pad_t + span_h, c0 : c0 + right_w] = color
        return patch

    span_w = random.randint(3, 6)
    top_h = random.randint(3, 6)
    bottom_h = random.randint(3, 6)
    patch = np.full((pad_t + top_h + gap + bottom_h + pad_b, pad_l + span_w + pad_r), BG_COLOR, dtype=int)
    patch[pad_t : pad_t + top_h, pad_l : pad_l + span_w] = color
    r0 = pad_t + top_h + gap
    patch[r0 : r0 + bottom_h, pad_l : pad_l + span_w] = color
    return patch


def _build_singleton_patch(color: int) -> np.ndarray:
    pad_t, pad_b, pad_l, pad_r = _outer_padding()
    h = random.randint(2, 6)
    w = random.randint(2, 6)
    patch = np.full((pad_t + h + pad_b, pad_l + w + pad_r), BG_COLOR, dtype=int)
    patch[pad_t : pad_t + h, pad_l : pad_l + w] = color
    return patch


def _place_patch(grid: np.ndarray, reserved: np.ndarray, patch: np.ndarray, padding: int) -> bool:
    height, width = map(int, grid.shape)
    patch_h, patch_w = map(int, patch.shape)
    if patch_h > height or patch_w > width:
        return False

    row_slack = height - patch_h
    col_slack = width - patch_w
    row_lo = 1 if row_slack >= 2 else 0
    row_hi = row_slack - 1 if row_slack >= 2 else row_slack
    col_lo = 1 if col_slack >= 2 else 0
    col_hi = col_slack - 1 if col_slack >= 2 else col_slack

    for _ in range(GEN_PLACEMENT_TRIES):
        r0 = random.randint(row_lo, row_hi)
        c0 = random.randint(col_lo, col_hi)
        rr0 = max(0, r0 - padding)
        cc0 = max(0, c0 - padding)
        rr1 = min(height, r0 + patch_h + padding)
        cc1 = min(width, c0 + patch_w + padding)
        if np.any(reserved[rr0:rr1, cc0:cc1]):
            continue

        mask = patch != BG_COLOR
        target = grid[r0 : r0 + patch_h, c0 : c0 + patch_w]
        assert np.all(target == BG_COLOR), "patch placement must land on empty background"
        target[mask] = patch[mask]
        reserved[rr0:rr1, cc0:cc1] = True
        return True
    return False


def _validate_generated(grid: np.ndarray, expected_candidates: int, expected_accepts: int) -> bool:
    if grid.ndim != 2 or max(grid.shape) > GEN_MAX_SIDE:
        return False
    if np.any(grid == BRIDGE_COLOR):
        return False
    if get_bg(grid) != BG_COLOR:
        return False
    if int(np.count_nonzero(grid != BG_COLOR)) >= int(np.count_nonzero(grid == BG_COLOR)):
        return False

    try:
        candidates = _all_candidates(grid, BG_COLOR)
    except AssertionError:
        return False

    distances = [gap for gap, _paint in candidates]
    if len(candidates) != expected_candidates or len(distances) != len(set(distances)):
        return False

    accepted, rejected = _accepted_bridges(grid.shape, candidates)
    if len(accepted) != expected_accepts or len(rejected) != expected_candidates - expected_accepts:
        return False

    bridge_mask = np.zeros(grid.shape, dtype=bool)
    for _gap, (r0, c0, r1, c1) in accepted:
        bridge_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    if len(rejected) != 1:
        return False

    try:
        out = main(grid)
    except AssertionError:
        return False
    return np.array_equal(out == BRIDGE_COLOR, bridge_mask)


def _gen_attempt() -> np.ndarray | None:
    n_extra_pairs = random.randint(*GEN_EXTRA_PAIR_RANGE)
    n_singletons = random.randint(*GEN_SINGLETON_RANGE)
    total_colors = 2 + n_extra_pairs + n_singletons
    colors = random.sample(GEN_FOREGROUND_COLORS, total_colors)

    used_distances: set[int] = set()
    v_gap, h_gap = random.sample(GEN_CROSS_GAP_POOL, 2)
    used_distances.update([v_gap, h_gap])

    patches = [_build_cross_patch(colors[0], colors[1], v_gap, h_gap)]
    next_color = 2
    for _ in range(n_extra_pairs):
        gap_choices = [gap for gap in GEN_PAIR_GAP_POOL if gap not in used_distances]
        gap = random.choice(gap_choices)
        used_distances.add(gap)
        patches.append(_build_pair_patch(colors[next_color], gap))
        next_color += 1
    for _ in range(n_singletons):
        patches.append(_build_singleton_patch(colors[next_color]))
        next_color += 1

    patches.sort(key=lambda patch: patch.shape[0] * patch.shape[1], reverse=True)
    min_h = max(GEN_MIN_SIDE, max(int(patch.shape[0]) + 2 for patch in patches))
    min_w = max(GEN_MIN_SIDE, max(int(patch.shape[1]) + 2 for patch in patches))
    area_need = sum((int(patch.shape[0]) + 2) * (int(patch.shape[1]) + 2) for patch in patches)

    for _ in range(GEN_CANVAS_TRIES):
        height = random.randint(min_h, GEN_MAX_SIDE)
        width = random.randint(min_w, GEN_MAX_SIDE)
        if height * width < area_need:
            continue

        grid = np.full((height, width), BG_COLOR, dtype=int)
        reserved = np.zeros((height, width), dtype=bool)
        ok = True
        for patch in patches:
            if not _place_patch(grid, reserved, patch, GEN_PADDING):
                ok = False
                break
        if not ok:
            continue

        if _validate_generated(grid, expected_candidates=2 + n_extra_pairs, expected_accepts=1 + n_extra_pairs):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid short-bridge-priority input")
