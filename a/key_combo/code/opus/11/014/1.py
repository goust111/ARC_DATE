from __future__ import annotations

import random

import numpy as np


BG = 0
HALO = 5
DIR_BY_COLOR = {
    1: (-1, 0),  # blue -> up
    2: (0, 1),   # red -> right
    3: (1, 0),   # green -> down
    4: (0, -1),  # yellow -> left
}
MAX_GENERATION_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _scan_markers(grid: np.ndarray) -> list[tuple[int, int, int]]:
    markers: list[tuple[int, int, int]] = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            color = int(grid[r, c])
            if color in DIR_BY_COLOR:
                markers.append((r, c, color))
    markers.sort()
    return markers


def _draw_halos(base: np.ndarray, markers: list[tuple[int, int, int]]) -> np.ndarray:
    out = np.array(base, dtype=int, copy=True)
    for r, c, _color in markers:
        r0 = max(0, r - 1)
        r1 = min(out.shape[0] - 1, r + 1)
        c0 = max(0, c - 1)
        c1 = min(out.shape[1] - 1, c + 1)
        out[r0 : r1 + 1, c0 : c1 + 1] = HALO
    for r, c, color in markers:
        out[r, c] = color
    return out


def _ray_cells_against_static(
    shape: tuple[int, int],
    markers: list[tuple[int, int, int]],
    marker_index: int,
) -> list[tuple[int, int]]:
    static = _draw_halos(np.zeros(shape, dtype=int), markers)
    r, c, color = markers[marker_index]
    dr, dc = DIR_BY_COLOR[color]
    rr = r + 2 * dr
    cc = c + 2 * dc
    cells: list[tuple[int, int]] = []
    while _in_bounds(shape, rr, cc):
        cell = int(static[rr, cc])
        if cell != BG:
            break
        cells.append((rr, cc))
        rr += dr
        cc += dc
    return cells


def _target_visible(src: tuple[int, int, int], dst: tuple[int, int, int]) -> bool:
    sr, sc, color = src
    tr, tc, _ = dst
    if color == 2:
        return tc > sc and abs(tr - sr) <= 1
    if color == 4:
        return tc < sc and abs(tr - sr) <= 1
    if color == 3:
        return tr > sr and abs(tc - sc) <= 1
    if color == 1:
        return tr < sr and abs(tc - sc) <= 1
    return False


def _pairwise_spacing_ok(markers: list[tuple[int, int, int]], min_dist: int = 5) -> bool:
    for i, (r1, c1, _color1) in enumerate(markers):
        for r2, c2, _color2 in markers[i + 1 :]:
            if abs(r1 - r2) + abs(c1 - c2) < min_dist:
                return False
    return True


def _every_marker_has_target(markers: list[tuple[int, int, int]]) -> bool:
    for i, src in enumerate(markers):
        if not any(_target_visible(src, dst) for j, dst in enumerate(markers) if i != j):
            return False
    return True


def _has_crossing(shape: tuple[int, int], markers: list[tuple[int, int, int]]) -> bool:
    paths = [
        set(_ray_cells_against_static(shape, markers, i))
        for i in range(len(markers))
    ]
    for i, path_a in enumerate(paths):
        for path_b in paths[i + 1 :]:
            if path_a & path_b:
                return True
    return False


def _markers_to_grid(shape: tuple[int, int], markers: list[tuple[int, int, int]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for r, c, color in markers:
        grid[r, c] = color
    return grid


def _sample_feeder(
    shape: tuple[int, int],
    markers: list[tuple[int, int, int]],
    rng: random.Random,
) -> tuple[int, int, int] | None:
    H, W = shape
    candidates: list[tuple[int, int, int]] = []
    for tr, tc, _target_color in markers:
        for offset in (-1, 0, 1):
            for dist in range(5, max(H, W)):
                extras = (
                    (tr - dist, tc + offset, 3),
                    (tr + dist, tc + offset, 1),
                    (tr + offset, tc - dist, 2),
                    (tr + offset, tc + dist, 4),
                )
                for candidate in extras:
                    r, c, _color = candidate
                    if _in_bounds(shape, r, c):
                        candidates.append(candidate)
    rng.shuffle(candidates)
    for candidate in candidates:
        test_markers = markers + [candidate]
        if _pairwise_spacing_ok(test_markers) and _every_marker_has_target(test_markers):
            return candidate
    return None


def _layout_is_interesting(
    shape: tuple[int, int],
    markers: list[tuple[int, int, int]],
    *,
    require_crossing: bool,
) -> bool:
    if not _pairwise_spacing_ok(markers):
        return False
    if not _every_marker_has_target(markers):
        return False
    if _has_crossing(shape, markers) != require_crossing:
        return False

    inp = _markers_to_grid(shape, markers)
    halo_only = _draw_halos(inp, _scan_markers(inp))
    out = main(inp)
    ray_mask = (halo_only == BG) & (out != BG)
    ray_cells = int(np.count_nonzero(ray_mask))
    if ray_cells < len(markers):
        return False
    if ray_cells > (shape[0] * shape[1]) // 4:
        return False
    return True


def _build_cross_layout(rng: random.Random, with_extra: bool) -> tuple[int, int, list[tuple[int, int, int]]] | None:
    H = rng.randint(13, 19)
    W = rng.randint(13, 19)
    for _ in range(120):
        left_c = rng.randint(0, 3)
        right_c = rng.randint(max(left_c + 6, W - 4), W - 1)
        top_r = rng.randint(0, 3)
        bottom_r = rng.randint(max(top_r + 6, H - 4), H - 1)
        if right_c - left_c < 6 or bottom_r - top_r < 6:
            continue

        mid_r = rng.randint(top_r + 3, bottom_r - 3)
        mid_c = rng.randint(left_c + 3, right_c - 3)
        right_r = mid_r + rng.choice((-1, 0, 1))
        bottom_c = mid_c + rng.choice((-1, 0, 1))
        if not _in_bounds((H, W), right_r, right_c):
            continue
        if not _in_bounds((H, W), bottom_r, bottom_c):
            continue

        markers = [
            (mid_r, left_c, 2),
            (right_r, right_c, 4),
            (top_r, mid_c, 3),
            (bottom_r, bottom_c, 1),
        ]
        if with_extra:
            extra = _sample_feeder((H, W), markers, rng)
            if extra is None:
                continue
            markers.append(extra)

        if _layout_is_interesting((H, W), markers, require_crossing=True):
            return H, W, markers
    return None


def _build_three_marker_layout(rng: random.Random) -> tuple[int, int, list[tuple[int, int, int]]] | None:
    H = rng.randint(12, 18)
    W = rng.randint(12, 18)
    for _ in range(150):
        if rng.random() < 0.5:
            left_c = rng.randint(0, 3)
            right_c = rng.randint(max(left_c + 6, W - 4), W - 1)
            row = rng.randint(2, H - 3)
            right_row = row + rng.choice((-1, 0, 1))
            base = [
                (row, left_c, 2),
                (right_row, right_c, 4),
            ]
        else:
            top_r = rng.randint(0, 3)
            bottom_r = rng.randint(max(top_r + 6, H - 4), H - 1)
            col = rng.randint(2, W - 3)
            bottom_col = col + rng.choice((-1, 0, 1))
            base = [
                (top_r, col, 3),
                (bottom_r, bottom_col, 1),
            ]

        feeder = _sample_feeder((H, W), base, rng)
        if feeder is None:
            continue
        markers = base + [feeder]
        if _layout_is_interesting((H, W), markers, require_crossing=False):
            return H, W, markers
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """Draw a grey halo around each colored marker, then extend color-coded rays until another halo blocks them.

    Input structure:
    - The grid is mostly black background.
    - Foreground consists of 3 to 5 isolated single-pixel markers using blue(1), red(2), green(3), and yellow(4).
    - Markers can lie near edges, so the 3x3 halo may clip against the grid boundary.

    Rule:
    - First, replace every marker's clipped 3x3 neighborhood with grey(5), then restore the marker center to its original color.
    - Interpret marker colors as ray directions: blue up, red right, green down, yellow left.
    - For each marker, in top-left to bottom-right scan order, cast a ray starting just outside its own halo and paint only black cells with the marker color.
    - Stop a ray before the first grey halo cell, non-black colored cell, or grid boundary; later rays cannot overwrite earlier ones.

    Key insight:
    - The grey dilation is not decorative: once all halos are drawn, they become walls that cap the directed rays from the other markers.
    """
    g = np.asarray(grid, dtype=int)
    markers = _scan_markers(g)
    out = _draw_halos(g, markers)
    H, W = out.shape

    for r, c, color in markers:
        dr, dc = DIR_BY_COLOR[color]
        rr = r + 2 * dr
        cc = c + 2 * dc
        while 0 <= rr < H and 0 <= cc < W:
            cell = int(out[rr, cc])
            if cell != BG:
                break
            out[rr, cc] = color
            rr += dr
            cc += dc
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    builders = [
        lambda: _build_three_marker_layout(rng),
        lambda: _build_cross_layout(rng, False),
        lambda: _build_cross_layout(rng, True),
    ]
    for _ in range(MAX_GENERATION_ATTEMPTS):
        builder = rng.choice(builders)
        built = builder()
        if built is None:
            continue
        H, W, markers = built
        grid = _markers_to_grid((H, W), markers)
        if grid.ndim != 2 or max(grid.shape) > 30:
            continue
        if np.unique(grid[grid != BG]).size < 2:
            continue
        _ = main(grid)
        return grid
    raise RuntimeError("failed to generate a valid halo-blocked-ray input")
