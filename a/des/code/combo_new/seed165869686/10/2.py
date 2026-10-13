from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
RED = 2
GREY = 5
PALETTE = [1, 3, 4, 6, 7, 8, 9]
DIRS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _separator_positions(grid: np.ndarray) -> tuple[list[int], list[int]]:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == GREY)]
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert rows and cols, "expected full grey separator rows and columns"
    return rows, cols


def _region_bounds(size: int, seps: list[int]) -> list[tuple[int, int]]:
    bounds: list[tuple[int, int]] = []
    start = 0
    for sep in seps:
        bounds.append((start, sep - 1))
        start = sep + 1
    bounds.append((start, size - 1))
    return bounds


def _locate_region(bounds: list[tuple[int, int]], idx: int) -> int:
    for i, (lo, hi) in enumerate(bounds):
        if lo <= idx <= hi:
            return i
    raise AssertionError("cell not inside any region")


def _parse_arrow(grid: np.ndarray, comp: list[tuple[int, int]]) -> dict:
    pts = set(comp)
    h, w = grid.shape
    rows: dict[int, list[int]] = {}
    cols: dict[int, list[int]] = {}
    for r, c in comp:
        rows.setdefault(r, []).append(c)
        cols.setdefault(c, []).append(r)

    candidates: list[dict] = []

    for r, cs in rows.items():
        cs = sorted(cs)
        if not cs or cs[-1] - cs[0] + 1 != len(cs) or len(cs) < 3:
            continue
        c0, c1 = cs[0], cs[-1]
        if len(comp) != len(cs) + 2:
            continue
        if (r - 1, c1 - 1) in pts and (r + 1, c1 - 1) in pts:
            candidates.append(
                {
                    "direction": "right",
                    "tail_end": (r, c0),
                    "near_band": (r, c0 - 1),
                    "cells": set(pts),
                }
            )
        if (r - 1, c0 + 1) in pts and (r + 1, c0 + 1) in pts:
            candidates.append(
                {
                    "direction": "left",
                    "tail_end": (r, c1),
                    "near_band": (r, c1 + 1),
                    "cells": set(pts),
                }
            )

    for c, rs in cols.items():
        rs = sorted(rs)
        if not rs or rs[-1] - rs[0] + 1 != len(rs) or len(rs) < 3:
            continue
        r0, r1 = rs[0], rs[-1]
        if len(comp) != len(rs) + 2:
            continue
        if (r1 - 1, c - 1) in pts and (r1 - 1, c + 1) in pts:
            candidates.append(
                {
                    "direction": "down",
                    "tail_end": (r0, c),
                    "near_band": (r0 - 1, c),
                    "cells": set(pts),
                }
            )
        if (r0 + 1, c - 1) in pts and (r0 + 1, c + 1) in pts:
            candidates.append(
                {
                    "direction": "up",
                    "tail_end": (r1, c),
                    "near_band": (r1 + 1, c),
                    "cells": set(pts),
                }
            )

    filtered: list[dict] = []
    for cand in candidates:
        br, bc = cand["near_band"]
        if 0 <= br < h and 0 <= bc < w and int(grid[br, bc]) not in (BLACK, RED, GREY):
            filtered.append(cand)

    assert len(filtered) == 1, "arrow component must decode to one direction"
    return filtered[0]


def _ordered_band(grid: np.ndarray, start: tuple[int, int], direction: str) -> list[tuple[int, int]]:
    r, c = start
    h, w = grid.shape
    assert 0 <= r < h and 0 <= c < w
    color = int(grid[r, c])
    assert color not in (BLACK, RED, GREY)
    dr, dc = DIRS[direction]
    br, bc = -dr, -dc

    band: list[tuple[int, int]] = []
    rr, cc = r, c
    while 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == color:
        band.append((int(rr), int(cc)))
        rr += br
        cc += bc
    assert band, "band must contain at least one pixel"
    return band


def _lane_order(length: int, entry: int, preferred_sign: int) -> list[int]:
    preferred = [entry]
    k = 1
    while 0 <= entry + preferred_sign * k < length:
        preferred.append(entry + preferred_sign * k)
        k += 1
    other: list[int] = []
    k = 1
    while 0 <= entry - preferred_sign * k < length:
        other.append(entry - preferred_sign * k)
        k += 1
    return preferred + other


def _snake_path(
    height: int,
    width: int,
    start_r: int,
    start_c: int,
    direction: str,
) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    if direction == "right":
        rows = _lane_order(height, start_r, preferred_sign=1)
        for i, r in enumerate(rows):
            if i == 0:
                cols = range(start_c, width)
            elif i % 2 == 1:
                cols = range(width - 1, -1, -1)
            else:
                cols = range(width)
            for c in cols:
                path.append((r, c))
    elif direction == "left":
        rows = _lane_order(height, start_r, preferred_sign=-1)
        for i, r in enumerate(rows):
            if i == 0:
                cols = range(start_c, -1, -1)
            elif i % 2 == 1:
                cols = range(width)
            else:
                cols = range(width - 1, -1, -1)
            for c in cols:
                path.append((r, c))
    elif direction == "down":
        cols = _lane_order(width, start_c, preferred_sign=-1)
        for i, c in enumerate(cols):
            if i == 0:
                rows = range(start_r, height)
            elif i % 2 == 1:
                rows = range(height - 1, -1, -1)
            else:
                rows = range(height)
            for r in rows:
                path.append((r, c))
    else:
        assert direction == "up"
        cols = _lane_order(width, start_c, preferred_sign=1)
        for i, c in enumerate(cols):
            if i == 0:
                rows = range(start_r, -1, -1)
            elif i % 2 == 1:
                rows = range(height)
            else:
                rows = range(height - 1, -1, -1)
            for r in rows:
                path.append((r, c))
    return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push each arrow-attached color band into the neighboring region and fold it when it hits that region's boundary.

    Input structure:
    - Black background with full grey separator rows and columns forming an equal-sized rectangular region grid.
    - Some regions contain one red arrow sprite: a straight tail plus a 3-pixel head whose tip points up, down, left, or right.
    - The tail end is attached to a one-color band made of a straight line of non-black, non-grey, non-red pixels extending in the opposite direction.

    Rule:
    - Detect each red arrow, decode its pointing direction, and collect the ordered band pixels starting from the pixel nearest the tail.
    - Remove both the arrow sprite and the original band from their source region.
    - Move the band into the adjacent region in the arrow direction, keeping the band's local entry row or column and preserving pixel order.
    - Inside the target region, lay the band along the arrow direction; when the next step would leave the target region, fold into the perpendicular lane, preferring the arrow's right-hand side and then snaking across later lanes.

    Key insight:
    - The red sprite encodes direction, but the moved object is the attached ordered band; solving requires combining arrow parsing, region-grid geometry, and right-hand-side snake packing inside the neighboring region.
    """
    g = np.asarray(grid, dtype=int)
    row_seps, col_seps = _separator_positions(g)
    row_bounds = _region_bounds(g.shape[0], row_seps)
    col_bounds = _region_bounds(g.shape[1], col_seps)

    arrows = []
    for comp in _components(g == RED):
        info = _parse_arrow(g, comp)
        band = _ordered_band(g, info["near_band"], info["direction"])
        any_r, any_c = next(iter(info["cells"]))
        src_rr = _locate_region(row_bounds, any_r)
        src_cc = _locate_region(col_bounds, any_c)
        dr, dc = DIRS[info["direction"]]
        dst_rr = src_rr + dr
        dst_cc = src_cc + dc
        assert 0 <= dst_rr < len(row_bounds) and 0 <= dst_cc < len(col_bounds)

        src_r0, src_r1 = row_bounds[src_rr]
        src_c0, src_c1 = col_bounds[src_cc]
        dst_r0, dst_r1 = row_bounds[dst_rr]
        dst_c0, dst_c1 = col_bounds[dst_cc]
        near_r, near_c = band[0]
        local_r = near_r - src_r0
        local_c = near_c - src_c0
        start_r = dst_r0 + local_r
        start_c = dst_c0 + local_c

        arrows.append(
            {
                "direction": info["direction"],
                "color": int(g[band[0]]),
                "cells": set(info["cells"]) | set(band),
                "band_len": len(band),
                "target": (dst_r0, dst_r1, dst_c0, dst_c1, start_r, start_c),
            }
        )
        assert src_r0 <= near_r <= src_r1 and src_c0 <= near_c <= src_c1
        assert dst_r0 <= start_r <= dst_r1 and dst_c0 <= start_c <= dst_c1

    out = g.copy()
    for arrow in arrows:
        for r, c in arrow["cells"]:
            out[r, c] = BLACK

    for arrow in arrows:
        dst_r0, dst_r1, dst_c0, dst_c1, start_r, start_c = arrow["target"]
        path = _snake_path(
            dst_r1 - dst_r0 + 1,
            dst_c1 - dst_c0 + 1,
            start_r - dst_r0,
            start_c - dst_c0,
            arrow["direction"],
        )
        assert arrow["band_len"] <= len(path), "target region must have enough fold capacity"
        for lr, lc in path[: arrow["band_len"]]:
            out[dst_r0 + lr, dst_c0 + lc] = arrow["color"]

    return out


def _all_edges(rows: int, cols: int) -> list[tuple[int, int, str]]:
    edges: list[tuple[int, int, str]] = []
    for rr in range(rows):
        for cc in range(cols):
            src = rr * cols + cc
            if rr > 0:
                edges.append((src, (rr - 1) * cols + cc, "up"))
            if rr + 1 < rows:
                edges.append((src, (rr + 1) * cols + cc, "down"))
            if cc > 0:
                edges.append((src, rr * cols + (cc - 1), "left"))
            if cc + 1 < cols:
                edges.append((src, rr * cols + (cc + 1), "right"))
    return edges


def _choose_edges(rng: random.Random, rows: int, cols: int) -> list[tuple[int, int, str]]:
    edges = _all_edges(rows, cols)
    rng.shuffle(edges)
    max_count = min(3, (rows * cols) // 2)
    target_count = rng.randint(1, max_count)
    chosen: list[tuple[int, int, str]] = []
    used_regions: set[int] = set()
    for src, dst, direction in edges:
        if src in used_regions or dst in used_regions:
            continue
        chosen.append((src, dst, direction))
        used_regions.add(src)
        used_regions.add(dst)
        if len(chosen) == target_count:
            break
    return chosen


def _placement_options(height: int, width: int, direction: str, tail_len: int, want_fold: bool) -> list[dict]:
    opts: list[dict] = []

    if direction == "right":
        for y in range(1, height - 1):
            for x in range(0, width - tail_len - 2):
                band_max = x + 1
                straight = width - x
                lo = straight + 1 if want_fold else 1
                hi = band_max if want_fold else min(band_max, straight)
                for length in range(lo, hi + 1):
                    opts.append({"near": (y, x), "length": length})
    elif direction == "left":
        for y in range(1, height - 1):
            for x in range(tail_len + 2, width):
                band_max = width - x
                straight = x + 1
                lo = straight + 1 if want_fold else 1
                hi = band_max if want_fold else min(band_max, straight)
                for length in range(lo, hi + 1):
                    opts.append({"near": (y, x), "length": length})
    elif direction == "down":
        for x in range(1, width - 1):
            for y in range(0, height - tail_len - 2):
                band_max = y + 1
                straight = height - y
                lo = straight + 1 if want_fold else 1
                hi = band_max if want_fold else min(band_max, straight)
                for length in range(lo, hi + 1):
                    opts.append({"near": (y, x), "length": length})
    else:
        assert direction == "up"
        for x in range(1, width - 1):
            for y in range(tail_len + 2, height):
                band_max = height - y
                straight = y + 1
                lo = straight + 1 if want_fold else 1
                hi = band_max if want_fold else min(band_max, straight)
                for length in range(lo, hi + 1):
                    opts.append({"near": (y, x), "length": length})

    return opts


def _arrow_cells(direction: str, near_r: int, near_c: int, tail_len: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    if direction == "right":
        cells.extend((near_r, near_c + i) for i in range(1, tail_len + 3))
        cells.append((near_r - 1, near_c + tail_len + 1))
        cells.append((near_r + 1, near_c + tail_len + 1))
    elif direction == "left":
        cells.extend((near_r, near_c - i) for i in range(1, tail_len + 3))
        cells.append((near_r - 1, near_c - tail_len - 1))
        cells.append((near_r + 1, near_c - tail_len - 1))
    elif direction == "down":
        cells.extend((near_r + i, near_c) for i in range(1, tail_len + 3))
        cells.append((near_r + tail_len + 1, near_c - 1))
        cells.append((near_r + tail_len + 1, near_c + 1))
    else:
        assert direction == "up"
        cells.extend((near_r - i, near_c) for i in range(1, tail_len + 3))
        cells.append((near_r - tail_len - 1, near_c - 1))
        cells.append((near_r - tail_len - 1, near_c + 1))
    return [(int(r), int(c)) for r, c in cells]


def _band_cells(direction: str, near_r: int, near_c: int, length: int) -> list[tuple[int, int]]:
    if direction == "right":
        return [(near_r, near_c - i) for i in range(length)]
    if direction == "left":
        return [(near_r, near_c + i) for i in range(length)]
    if direction == "down":
        return [(near_r - i, near_c) for i in range(length)]
    return [(near_r + i, near_c) for i in range(length)]


def _target_cells(
    region_h: int,
    region_w: int,
    direction: str,
    near_r: int,
    near_c: int,
    length: int,
) -> list[tuple[int, int]]:
    path = _snake_path(region_h, region_w, near_r, near_c, direction)
    if length > len(path):
        return []
    return path[:length]


def _draw_case(
    grid: np.ndarray,
    region_h: int,
    region_w: int,
    src_rr: int,
    src_cc: int,
    direction: str,
    near_r: int,
    near_c: int,
    tail_len: int,
    length: int,
    color: int,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    src_r0 = src_rr * (region_h + 1)
    src_c0 = src_cc * (region_w + 1)
    band = {(src_r0 + r, src_c0 + c) for r, c in _band_cells(direction, near_r, near_c, length)}
    arrow = {(src_r0 + r, src_c0 + c) for r, c in _arrow_cells(direction, near_r, near_c, tail_len)}
    for r, c in band:
        grid[r, c] = color
    for r, c in arrow:
        grid[r, c] = RED

    dr, dc = DIRS[direction]
    dst_rr = src_rr + dr
    dst_cc = src_cc + dc
    dst_r0 = dst_rr * (region_h + 1)
    dst_c0 = dst_cc * (region_w + 1)
    target = {
        (dst_r0 + r, dst_c0 + c)
        for r, c in _target_cells(region_h, region_w, direction, near_r, near_c, length)
    }
    return band | arrow, target


def generate_input() -> np.ndarray:
    rng = random.Random()

    for _ in range(400):
        rows = rng.choice([2, 3])
        cols = rng.choice([2, 3])
        region_h = rng.randint(7, 8)
        region_w = rng.randint(7, 8)
        height = rows * region_h + (rows - 1)
        width = cols * region_w + (cols - 1)
        if height > 30 or width > 30:
            continue

        chosen = _choose_edges(rng, rows, cols)
        if not chosen:
            continue

        grid = np.full((height, width), BLACK, dtype=int)
        for rr in range(1, rows):
            grid[rr * region_h + (rr - 1), :] = GREY
        for cc in range(1, cols):
            grid[:, cc * region_w + (cc - 1)] = GREY

        used_targets: set[tuple[int, int]] = set()
        colors = rng.sample(PALETTE, len(chosen))
        fold_used = False
        no_fold_used = False
        ok = True

        for color, (src, dst, direction) in zip(colors, chosen):
            src_rr, src_cc = divmod(src, cols)
            tail_len = rng.randint(1, 3)
            want_fold = rng.random() < 0.5

            fold_opts = _placement_options(region_h, region_w, direction, tail_len, True)
            flat_opts = _placement_options(region_h, region_w, direction, tail_len, False)
            if not fold_opts and not flat_opts:
                ok = False
                break
            if want_fold and fold_opts:
                opts = fold_opts
            elif (not want_fold) and flat_opts:
                opts = flat_opts
            else:
                opts = fold_opts if fold_opts else flat_opts
            rng.shuffle(opts)

            placed = False
            for opt in opts:
                near_r, near_c = opt["near"]
                length = opt["length"]
                target_cells = _target_cells(region_h, region_w, direction, near_r, near_c, length)
                if not target_cells:
                    continue
                target_global = {
                    ((dst // cols) * (region_h + 1) + r, (dst % cols) * (region_w + 1) + c)
                    for r, c in target_cells
                }
                if used_targets & target_global:
                    continue

                _draw_case(
                    grid,
                    region_h,
                    region_w,
                    src_rr,
                    src_cc,
                    direction,
                    near_r,
                    near_c,
                    tail_len,
                    length,
                    color,
                )
                used_targets |= target_global
                if length > (
                    region_w - near_c
                    if direction == "right"
                    else near_c + 1
                    if direction == "left"
                    else region_h - near_r
                    if direction == "down"
                    else near_r + 1
                ):
                    fold_used = True
                else:
                    no_fold_used = True
                placed = True
                break

            if not placed:
                ok = False
                break

        if not ok:
            continue
        if not (fold_used or no_fold_used):
            continue
        return grid

    raise RuntimeError("failed to generate a valid arrow-band puzzle")
