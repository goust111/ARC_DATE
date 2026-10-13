from __future__ import annotations

import random

import numpy as np


BG = 0
SEP = 5
RECT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
ORIENTATIONS = ("tl", "tr", "bl", "br")
GEN_RETRIES = 300
RECT_COUNT_RANGE = (3, 4)
LOCAL_SIDE_RANGE = (11, 17)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_legend(grid: np.ndarray) -> tuple[int, list[int]]:
    cols = np.where(np.any(grid != BG, axis=0))[0]
    assert cols.size > 0, "expected non-background content"
    legend_c = int(cols[-1])
    rows = np.where(grid[:, legend_c] != BG)[0]
    assert 3 <= rows.size <= 4, "legend must list 3 or 4 colors"
    if rows.size > 1:
        gaps = np.diff(rows)
        assert np.all(gaps >= 2), "legend entries must be vertically separated"
    order = [int(grid[int(r), legend_c]) for r in rows]
    return legend_c, order


def _run_right(grid: np.ndarray, r: int, c: int, color: int) -> int:
    while c + 1 < grid.shape[1] and int(grid[r, c + 1]) == int(color):
        c += 1
    return int(c)


def _run_left(grid: np.ndarray, r: int, c: int, color: int) -> int:
    while c - 1 >= 0 and int(grid[r, c - 1]) == int(color):
        c -= 1
    return int(c)


def _run_down(grid: np.ndarray, r: int, c: int, color: int) -> int:
    while r + 1 < grid.shape[0] and int(grid[r + 1, c]) == int(color):
        r += 1
    return int(r)


def _run_up(grid: np.ndarray, r: int, c: int, color: int) -> int:
    while r - 1 >= 0 and int(grid[r - 1, c]) == int(color):
        r -= 1
    return int(r)


def _all_color_on_row(grid: np.ndarray, r: int, c0: int, c1: int, color: int) -> bool:
    return bool(np.all(grid[r, c0 : c1 + 1] == int(color)))


def _all_color_on_col(grid: np.ndarray, c: int, r0: int, r1: int, color: int) -> bool:
    return bool(np.all(grid[r0 : r1 + 1, c] == int(color)))


def _has_exposed_corner(grid: np.ndarray, color: int, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return any(
        (
            _all_color_on_row(grid, r0, c0, c1, color)
            and _all_color_on_col(grid, c0, r0, r1, color),
            _all_color_on_row(grid, r0, c0, c1, color)
            and _all_color_on_col(grid, c1, r0, r1, color),
            _all_color_on_row(grid, r1, c0, c1, color)
            and _all_color_on_col(grid, c0, r0, r1, color),
            _all_color_on_row(grid, r1, c0, c1, color)
            and _all_color_on_col(grid, c1, r0, r1, color),
        )
    )


def _validate_bbox(
    observed: np.ndarray,
    visible_higher: np.ndarray,
    color: int,
    bbox: tuple[int, int, int, int],
) -> bool:
    r0, c0, r1, c1 = bbox
    if r0 < 0 or c0 < 0 or r1 >= observed.shape[0] or c1 >= observed.shape[1]:
        return False
    if r0 > r1 or c0 > c1:
        return False

    candidate_mask = np.zeros_like(observed, dtype=bool)
    candidate_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    visible_mask = observed == int(color)
    if np.any(visible_mask & (~candidate_mask)):
        return False

    expected = visible_higher[r0 : r1 + 1, c0 : c1 + 1].copy()
    expected[expected == BG] = int(color)
    if not np.array_equal(observed[r0 : r1 + 1, c0 : c1 + 1], expected):
        return False

    return True


def _corner_candidates(
    observed: np.ndarray,
    visible_higher: np.ndarray,
    color: int,
) -> list[tuple[int, int, int, int]]:
    mask = observed == int(color)
    b = _bbox(mask)
    assert b is not None, f"color {color} must be visible"
    vr0, vc0, vr1, vc1 = b
    candidates = [
        (vr0, vc0, _run_down(observed, vr0, vc0, color), _run_right(observed, vr0, vc0, color)),
        (vr0, _run_left(observed, vr0, vc1, color), _run_down(observed, vr0, vc1, color), vc1),
        (_run_up(observed, vr1, vc0, color), vc0, vr1, _run_right(observed, vr1, vc0, color)),
        (_run_up(observed, vr1, vc1, color), _run_left(observed, vr1, vc1, color), vr1, vc1),
    ]

    valid: list[tuple[int, int, int, int]] = []
    for cand in candidates:
        if _validate_bbox(observed, visible_higher, color, cand):
            valid.append(cand)

    dedup: list[tuple[int, int, int, int]] = []
    seen = set()
    for cand in valid:
        if cand not in seen:
            seen.add(cand)
            dedup.append(cand)
    return dedup


def _exhaustive_bbox(
    observed: np.ndarray,
    visible_higher: np.ndarray,
    color: int,
) -> tuple[int, int, int, int]:
    mask = observed == int(color)
    b = _bbox(mask)
    assert b is not None, f"color {color} must be visible"
    vr0, vc0, vr1, vc1 = b

    valid: list[tuple[int, int, int, int]] = []
    for r0 in range(vr0 + 1):
        for c0 in range(vc0 + 1):
            for r1 in range(vr1, observed.shape[0]):
                for c1 in range(vc1, observed.shape[1]):
                    cand = (r0, c0, r1, c1)
                    if not _validate_bbox(observed, visible_higher, color, cand):
                        continue
                    if not _has_exposed_corner(observed, color, cand):
                        continue
                    valid.append(cand)

    dedup = sorted(set(valid))
    assert len(dedup) == 1, f"expected unique reconstruction for color {color}, got {dedup}"
    return dedup[0]


def _reconstruct_bbox(
    observed: np.ndarray,
    visible_higher: np.ndarray,
    color: int,
) -> tuple[int, int, int, int]:
    candidates = _corner_candidates(observed, visible_higher, color)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        exact = sorted(set(candidates))
        if len(exact) == 1:
            return exact[0]
    return _exhaustive_bbox(observed, visible_higher, color)


def _render_output(colors_bottom_to_top: list[int], dims_bottom_to_top: list[tuple[int, int]]) -> np.ndarray:
    out_h = max(h for h, _w in dims_bottom_to_top)
    out_w = sum(w for _h, w in dims_bottom_to_top) + len(dims_bottom_to_top) - 1
    out = np.full((out_h, out_w), BG, dtype=int)

    col = 0
    for i, (color, (h, w)) in enumerate(zip(colors_bottom_to_top, dims_bottom_to_top, strict=False)):
        if i > 0:
            out[:, col] = SEP
            col += 1
        out[out_h - h : out_h, col : col + w] = int(color)
        col += w
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover fully occluded axis-aligned rectangles using the right-side stack legend,
    then lay them out from bottom layer to top layer with grey separators.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    legend_c, colors_top_to_bottom = _find_legend(g)
    observed = g.copy()
    observed[:, legend_c] = BG

    visible_higher = np.full_like(observed, BG)
    bboxes: dict[int, tuple[int, int, int, int]] = {}

    for color in colors_top_to_bottom:
        bbox = _reconstruct_bbox(observed, visible_higher, int(color))
        bboxes[int(color)] = bbox
        r0, c0, r1, c1 = bbox
        window = visible_higher[r0 : r1 + 1, c0 : c1 + 1]
        window[window == BG] = int(color)

    colors_bottom_to_top = list(reversed(colors_top_to_bottom))
    dims_bottom_to_top = [
        (bboxes[color][2] - bboxes[color][0] + 1, bboxes[color][3] - bboxes[color][1] + 1)
        for color in colors_bottom_to_top
    ]
    return _render_output(colors_bottom_to_top, dims_bottom_to_top)


def _paint_rectangles(
    rects_bottom_to_top: list[tuple[int, int, int, int]],
    colors_bottom_to_top: list[int],
    shape: tuple[int, int],
) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for (r0, c0, r1, c1), color in zip(rects_bottom_to_top, colors_bottom_to_top, strict=False):
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return grid


def _flip_grid(grid: np.ndarray, orientation: str) -> np.ndarray:
    out = grid
    if orientation in ("tr", "br"):
        out = np.fliplr(out)
    if orientation in ("bl", "br"):
        out = np.flipud(out)
    return out.copy()


def _sample_nested_rectangles(n_rects: int) -> tuple[list[tuple[int, int, int, int]], tuple[int, int]] | None:
    min_side = max(LOCAL_SIDE_RANGE[0], 2 * n_rects + 3)
    max_side = LOCAL_SIDE_RANGE[1]
    if min_side > max_side:
        return None

    h = random.randint(min_side, max_side)
    w = random.randint(min_side, max_side)

    anchor_r_lo = n_rects
    anchor_r_hi = h - n_rects - 2
    anchor_c_lo = n_rects
    anchor_c_hi = w - n_rects - 2
    if anchor_r_lo > anchor_r_hi or anchor_c_lo > anchor_c_hi:
        return None

    anchor_r = random.randint(anchor_r_lo, anchor_r_hi)
    anchor_c = random.randint(anchor_c_lo, anchor_c_hi)

    r0s = sorted(random.sample(range(anchor_r), n_rects))
    c0s = sorted(random.sample(range(anchor_c), n_rects))
    r1s = sorted(random.sample(range(anchor_r + 2, h), n_rects), reverse=True)
    c1s = sorted(random.sample(range(anchor_c + 2, w), n_rects), reverse=True)

    rects = [(r0s[i], c0s[i], r1s[i], c1s[i]) for i in range(n_rects)]
    for r0, c0, r1, c1 in rects:
        if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
            return None
    return rects, (h, w)


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    n_rects = random.randint(*RECT_COUNT_RANGE)
    sampled = _sample_nested_rectangles(n_rects)
    if sampled is None:
        return None
    rects_bottom_to_top, local_shape = sampled

    colors_top_to_bottom = random.sample(RECT_COLORS, n_rects)
    colors_bottom_to_top = list(reversed(colors_top_to_bottom))
    local = _paint_rectangles(rects_bottom_to_top, colors_bottom_to_top, local_shape)
    local = _flip_grid(local, random.choice(ORIENTATIONS))

    dims_bottom_to_top = [(r1 - r0 + 1, c1 - c0 + 1) for r0, c0, r1, c1 in rects_bottom_to_top]
    expected = _render_output(colors_bottom_to_top, dims_bottom_to_top)

    legend_h = 2 * n_rects - 1
    cluster_top = random.randint(0, 4)
    cluster_left = random.randint(0, 4)
    gap = random.randint(2, 4)
    right_margin = random.randint(0, 2)
    bottom_margin = random.randint(0, 4)

    legend_c = cluster_left + local_shape[1] + gap
    H = max(cluster_top + local_shape[0], legend_h) + bottom_margin
    W = legend_c + 1 + right_margin
    if H < legend_h:
        return None

    legend_top = random.randint(0, H - legend_h)
    grid = np.full((H, W), BG, dtype=int)
    grid[cluster_top : cluster_top + local_shape[0], cluster_left : cluster_left + local_shape[1]] = local
    for i, color in enumerate(colors_top_to_bottom):
        grid[legend_top + 2 * i, legend_c] = int(color)

    for color in colors_top_to_bottom:
        if int((grid == int(color)).sum()) == 0:
            return None

    if np.any(grid[:, legend_c + 1 :] != BG):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _build_case()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid puzzle instance")
