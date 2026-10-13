from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import find_components, get_bg


BG = 0
AXIS = 5
MARKER = 6
PALETTE = (1, 2, 3, 4)
DIRS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
ORDER = ("tl", "tr", "bl", "br")

SMALL_DIMS = ((1, 2), (2, 1), (1, 3))
MEDIUM_DIMS = ((2, 2), (1, 3), (3, 1))
LARGE_DIMS = ((2, 4), (4, 2))
FILL_DIMS = ((1, 2), (2, 1), (1, 3), (3, 1), (2, 2), (2, 3), (3, 2))


def _find_axis(grid: np.ndarray) -> tuple[int, int]:
    rows = np.where(np.all(grid == AXIS, axis=1))[0]
    cols = np.where(np.all(grid == AXIS, axis=0))[0]
    assert len(rows) == 1 and len(cols) == 1, "expected one full gray cross"
    return int(rows[0]), int(cols[0])


def _quadrant_slices(grid: np.ndarray) -> dict[str, tuple[slice, slice]]:
    mid_r, mid_c = _find_axis(grid)
    return {
        "tl": (slice(0, mid_r), slice(0, mid_c)),
        "tr": (slice(0, mid_r), slice(mid_c + 1, grid.shape[1])),
        "bl": (slice(mid_r + 1, grid.shape[0]), slice(0, mid_c)),
        "br": (slice(mid_r + 1, grid.shape[0]), slice(mid_c + 1, grid.shape[1])),
    }


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _component_from(quadrant: np.ndarray, start: tuple[int, int]) -> list[tuple[int, int]]:
    color = int(quadrant[start])
    stack = [start]
    seen = {start}
    coords: list[tuple[int, int]] = []
    while stack:
        r, c = stack.pop()
        coords.append((r, c))
        for dr, dc in DIRS.values():
            nr, nc = r + dr, c + dc
            if not _in_bounds(quadrant.shape, nr, nc):
                continue
            if (nr, nc) in seen or int(quadrant[nr, nc]) != color:
                continue
            seen.add((nr, nc))
            stack.append((nr, nc))
    return coords


def _detect_marker(quadrant: np.ndarray) -> dict | None:
    magenta = {tuple(map(int, rc)) for rc in np.argwhere(quadrant == MARKER)}
    if not magenta:
        return None
    assert len(magenta) == 3, "each marked quadrant must contain exactly three magenta pixels"

    candidates = []
    for r in range(quadrant.shape[0]):
        for c in range(quadrant.shape[1]):
            if int(quadrant[r, c]) != BG:
                continue
            dirs = []
            for dr, dc in DIRS.values():
                nr, nc = r + dr, c + dc
                if _in_bounds(quadrant.shape, nr, nc) and (nr, nc) in magenta:
                    dirs.append((dr, dc))
            if len(dirs) != 3:
                continue
            stem = None
            for dr, dc in dirs:
                if (-dr, -dc) not in dirs:
                    stem = (dr, dc)
                    break
            if stem is None:
                continue
            bar_dirs = [d for d in dirs if d != stem]
            if len(bar_dirs) != 2 or bar_dirs[0] != (-bar_dirs[1][0], -bar_dirs[1][1]):
                continue
            marker_cells = {(r + dr, c + dc) for dr, dc in dirs}
            if marker_cells == magenta:
                candidates.append(
                    {
                        "center": (r, c),
                        "stem": stem,
                        "bar_dirs": bar_dirs,
                        "cells": marker_cells,
                    }
                )
    assert len(candidates) == 1, "marker must be uniquely decodable"
    return candidates[0]


def _first_region(quadrant: np.ndarray, center: tuple[int, int], ray_dir: tuple[int, int]) -> dict:
    r, c = center
    dr, dc = ray_dir
    step = 2
    while True:
        nr, nc = r + dr * step, c + dc * step
        assert _in_bounds(quadrant.shape, nr, nc), "each ray must hit a region inside its quadrant"
        value = int(quadrant[nr, nc])
        if value not in (BG, MARKER):
            coords = _component_from(quadrant, (nr, nc))
            return {
                "color": value,
                "coords": coords,
                "area": len(coords),
            }
        step += 1


def _color_counts(grid: np.ndarray) -> dict[int, int]:
    return {color: int(np.sum(grid == color)) for color in PALETTE}


def _solve(grid: np.ndarray, order: tuple[str, ...] = ORDER, trace: bool = False):
    g = np.asarray(grid, dtype=int).copy()
    quadrants = _quadrant_slices(g)
    steps: dict[str, dict] = {}

    for name in order:
        rs, cs = quadrants[name]
        quadrant = g[rs, cs].copy()
        marker = _detect_marker(quadrant)
        if marker is None:
            steps[name] = {"marked": False}
            continue

        target = _first_region(quadrant, marker["center"], marker["stem"])
        sources = [_first_region(quadrant, marker["center"], ray_dir) for ray_dir in marker["bar_dirs"]]
        source_colors = [int(src["color"]) for src in sources]
        counts = _color_counts(g)

        if source_colors[0] == source_colors[1]:
            new_color = source_colors[0]
        else:
            left_count = counts[source_colors[0]]
            right_count = counts[source_colors[1]]
            assert left_count != right_count, "source colors must be globally distinguishable"
            new_color = source_colors[0] if left_count < right_count else source_colors[1]

        for r, c in target["coords"]:
            quadrant[r, c] = new_color
        for r, c in marker["cells"]:
            quadrant[r, c] = BG

        g[rs, cs] = quadrant
        steps[name] = {
            "marked": True,
            "source_colors": tuple(source_colors),
            "source_areas": tuple(int(src["area"]) for src in sources),
            "target_area": int(target["area"]),
            "target_color_before": int(target["color"]),
            "new_color": int(new_color),
        }

    return (g, steps) if trace else g


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each quadrant's pointed region under a cross-axis, using current global color counts and a fixed quadrant order.

    Input structure:
    - The grid has black background, plus a full gray middle row and middle column that split it into four quadrants.
    - Each quadrant contains 2 to 3 separated monochrome rectangles in colors 1 to 4.
    - Some quadrants also contain a three-pixel magenta T marker encoded as three singleton arm pixels around one empty center cell.

    Rule:
    - Process quadrants in the fixed order top-left, top-right, bottom-left, bottom-right.
    - For a marked quadrant, the lone stem arm points to the target region: along that ray, the first non-background non-marker region is the target.
    - The opposite arm pair are the two sources: along each arm's outward ray, the first non-background non-marker region gives one source color.
    - Recolor the whole target region to the source color whose total pixel count is smaller in the current full grid; if both sources already have the same color, keep that color.
    - After recoloring, erase the quadrant's magenta marker to background; unmarked quadrants stay unchanged.

    Key insight:
    - The quadrants are not independent: earlier recolors change global color totals, so later markers must be interpreted on the already-updated grid and in the hidden fixed quadrant order.
    """
    return _solve(grid)


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            src_r0 = max(0, -dr)
            src_r1 = mask.shape[0] - max(0, dr)
            src_c0 = max(0, -dc)
            src_c1 = mask.shape[1] - max(0, dc)
            dst_r0 = max(0, dr)
            dst_r1 = dst_r0 + (src_r1 - src_r0)
            dst_c0 = max(0, dc)
            dst_c1 = dst_c0 + (src_c1 - src_c0)
            out[dst_r0:dst_r1, dst_c0:dst_c1] |= mask[src_r0:src_r1, src_c0:src_c1]
    return out


def _paint_rect(canvas: np.ndarray, top: int, left: int, h: int, w: int, color: int) -> None:
    canvas[top : top + h, left : left + w] = int(color)


def _build_marked_canonical(
    q: int,
    left_color: int,
    right_color: int,
    target_color: int,
    left_dims: tuple[int, int],
    right_dims: tuple[int, int],
    target_dims: tuple[int, int],
) -> tuple[np.ndarray, dict] | None:
    hl, wl = left_dims
    hr, wr = right_dims
    ht, wt = target_dims

    for _ in range(120):
        gap_l = random.randint(0, 1)
        gap_r = random.randint(0, 1)
        gap_t = random.randint(0, 1)

        c_min = wl + gap_l + 1
        c_max = q - wr - gap_r - 2
        r_min = 1
        r_max = q - ht - gap_t - 3
        if c_min > c_max or r_min > r_max:
            continue

        center_r = random.randint(r_min, r_max)
        center_c = random.randint(c_min, c_max)
        left_start = center_c - wl - gap_l - 1
        right_start = center_c + gap_r + 2
        target_top = center_r + gap_t + 2

        top_l_low = max(0, center_r - hl + 1)
        top_l_high = min(center_r, q - hl)
        top_r_low = max(0, center_r - hr + 1)
        top_r_high = min(center_r, q - hr)
        left_t_low = max(0, center_c - wt + 1)
        left_t_high = min(center_c, q - wt)
        if top_l_low > top_l_high or top_r_low > top_r_high or left_t_low > left_t_high:
            continue

        top_l = random.randint(top_l_low, top_l_high)
        top_r = random.randint(top_r_low, top_r_high)
        left_t = random.randint(left_t_low, left_t_high)

        quadrant = np.zeros((q, q), dtype=int)
        _paint_rect(quadrant, top_l, left_start, hl, wl, left_color)
        _paint_rect(quadrant, top_r, right_start, hr, wr, right_color)
        _paint_rect(quadrant, target_top, left_t, ht, wt, target_color)

        marker_cells = {
            (center_r, center_c - 1),
            (center_r, center_c + 1),
            (center_r + 1, center_c),
        }
        if quadrant[center_r, center_c] != BG:
            continue
        if any(int(quadrant[r, c]) != BG for r, c in marker_cells):
            continue
        for r, c in marker_cells:
            quadrant[r, c] = MARKER

        return quadrant, {"target_area": int(ht * wt)}
    return None


def _build_marked_quadrant(
    q: int,
    left_color: int,
    right_color: int,
    target_color: int,
    left_dims: tuple[int, int],
    right_dims: tuple[int, int],
    target_dims: tuple[int, int],
) -> tuple[np.ndarray, dict] | None:
    built = _build_marked_canonical(q, left_color, right_color, target_color, left_dims, right_dims, target_dims)
    if built is None:
        return None
    quadrant, meta = built
    quadrant = np.rot90(quadrant, k=random.randrange(4)).copy()
    return quadrant, meta


def _build_unmarked_quadrant(q: int, colors: tuple[int, ...]) -> np.ndarray | None:
    quadrant = np.zeros((q, q), dtype=int)
    blocked = np.zeros((q, q), dtype=bool)
    rect_count = random.randint(2, 3)
    for _ in range(rect_count):
        for _ in range(120):
            h, w = random.choice(FILL_DIMS)
            top = random.randint(0, q - h)
            left = random.randint(0, q - w)
            footprint = np.zeros((q, q), dtype=bool)
            footprint[top : top + h, left : left + w] = True
            if np.any(blocked & footprint):
                continue
            _paint_rect(quadrant, top, left, h, w, random.choice(colors))
            blocked |= _dilate8(footprint)
            break
        else:
            return None
    return quadrant


def _assemble_grid(quadrants: dict[str, np.ndarray], q: int) -> np.ndarray:
    size = 2 * q + 1
    grid = np.zeros((size, size), dtype=int)
    grid[q, :] = AXIS
    grid[:, q] = AXIS
    grid[:q, :q] = quadrants["tl"]
    grid[:q, q + 1 :] = quadrants["tr"]
    grid[q + 1 :, :q] = quadrants["bl"]
    grid[q + 1 :, q + 1 :] = quadrants["br"]
    return grid


def _quadrant_region_count(quadrant: np.ndarray) -> int:
    comps = find_components(quadrant, bg=BG, colors=PALETTE, connectivity=4, monochromatic=True)
    return int(len(comps))


def _optional_quadrant(q: int, b_color: int, d_color: int) -> np.ndarray | None:
    mode = random.choice(("unmarked", "same", "different"))
    if mode == "unmarked":
        return _build_unmarked_quadrant(q, (b_color, d_color))

    if mode == "same":
        source = random.choice((b_color, d_color))
        target = d_color if source == b_color else b_color
        built = _build_marked_quadrant(
            q=q,
            left_color=source,
            right_color=source,
            target_color=target,
            left_dims=random.choice(SMALL_DIMS),
            right_dims=random.choice(SMALL_DIMS),
            target_dims=random.choice(MEDIUM_DIMS),
        )
        return None if built is None else built[0]

    built = _build_marked_quadrant(
        q=q,
        left_color=b_color,
        right_color=d_color,
        target_color=random.choice((b_color, d_color)),
        left_dims=random.choice(SMALL_DIMS),
        right_dims=random.choice(MEDIUM_DIMS),
        target_dims=random.choice(MEDIUM_DIMS),
    )
    return None if built is None else built[0]


def _validate_case(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert get_bg(grid) == BG
    out, trace = _solve(grid, trace=True)
    assert out.shape == grid.shape

    quadrants = _quadrant_slices(grid)
    marker_count = 0
    for name, (rs, cs) in quadrants.items():
        quadrant = grid[rs, cs]
        regions = _quadrant_region_count(quadrant)
        assert 2 <= regions <= 3, "each quadrant must have 2 to 3 colored rectangles"
        if np.any(quadrant == MARKER):
            marker_count += 1
            assert trace[name]["marked"]
        else:
            assert not trace[name]["marked"]
    assert marker_count >= 2

    assert trace["tl"]["marked"] and trace["tr"]["marked"]
    assert trace["tr"]["target_color_before"] == trace["tl"]["new_color"]
    assert trace["tr"]["source_colors"][0] != trace["tr"]["source_colors"][1]
    chosen_idx = 0 if trace["tr"]["new_color"] == trace["tr"]["source_colors"][0] else 1
    other_idx = 1 - chosen_idx
    assert trace["tr"]["source_areas"][chosen_idx] > trace["tr"]["source_areas"][other_idx], (
        "later quadrant must not be solvable by picking the smaller local source region"
    )

    alt_out, alt_trace = _solve(grid, order=("tr", "tl", "bl", "br"), trace=True)
    assert trace["tr"]["new_color"] != alt_trace["tr"]["new_color"], "order dependency must be real"
    assert not np.array_equal(out, alt_out), "swapping the first two quadrants must change the result"


def generate_input() -> np.ndarray:
    for _ in range(400):
        q = random.randint(8, 10)
        a_color, b_color, c_color, d_color = random.sample(PALETTE, 4)

        tl = _build_marked_quadrant(
            q=q,
            left_color=a_color,
            right_color=b_color,
            target_color=d_color,
            left_dims=random.choice(SMALL_DIMS),
            right_dims=random.choice(LARGE_DIMS),
            target_dims=random.choice(MEDIUM_DIMS),
        )
        tr = _build_marked_quadrant(
            q=q,
            left_color=a_color,
            right_color=c_color,
            target_color=a_color,
            left_dims=random.choice(((1, 2), (2, 1))),
            right_dims=random.choice(LARGE_DIMS),
            target_dims=random.choice(((1, 2), (2, 1))),
        )
        if tl is None or tr is None:
            continue

        bl = _optional_quadrant(q, b_color, d_color)
        br = _optional_quadrant(q, b_color, d_color)
        if bl is None or br is None:
            continue

        grid = _assemble_grid({"tl": tl[0], "tr": tr[0], "bl": bl, "br": br}, q)
        try:
            _validate_case(grid)
            return grid
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid ARC instance")
