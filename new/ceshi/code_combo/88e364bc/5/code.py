# Combo variant of 88e364bc (index 5)
# combo_spec: D3/0+D10/0
# Name: 重力流动+无堆叠
#
# Fused design summary:
# - Keep the original scene grammar: corner legends, colored wall chambers, yellow movers.
# - Apply D3/0 by reinterpreting each legend as a gravity axis. A yellow does not move on one
#   straight ray anymore: it falls along gravity and, when blocked, searches sideways for the
#   nearest spill point that lets it drop lower.
# - Apply D10/0 by removing all yellow-yellow interaction. Every yellow is solved against the
#   empty chamber only, so several yellows may converge to the same final sink cell and collapse
#   into one visible output pixel.
# - The generator uses narrow funnel chambers so both changes are externally visible:
#   side yellows must bend inward (showing gravity flow) and multiple sources merge into one sink
#   (showing no stacking).

from __future__ import annotations

import os
import random
import sys

import numpy as np


ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")
)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from arc_utils import bbox_coords, components_mask, get_bg, neighbors4, paste


GRID_H = 20
GRID_W = 20
MOVER = 4

CARDINAL_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
CORNERS = ["TL", "TR", "BL", "BR"]
WALL_COLORS = [3, 5, 6, 7, 8, 9]

CANONICAL_FUNNEL = np.array(
    [
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ],
    dtype=bool,
)
REGION_ANCHORS = [(7, 2), (7, 12)]
GEN_RETRIES = 200


def _legend_directions(g: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    h, w = g.shape
    non_bg = g != bg
    best: dict[int, tuple[int, tuple[int, int]]] = {}
    for pts in components_mask(non_bg, connectivity=4):
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        r0, c0, r1, c1 = bbox_coords(pts)
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if area != len(pts):
            continue

        counts: dict[int, int] = {}
        for r, c in pts:
            v = int(g[r, c])
            counts[v] = counts.get(v, 0) + 1
        if len(counts) != 3:
            continue

        fill_col = max(counts, key=counts.get)
        others = [(cnt, col) for col, cnt in counts.items() if col != fill_col]
        if len(others) != 2:
            continue
        others.sort()
        if others[0][0] == others[1][0]:
            rc = np.array([(r0 + r1) / 2.0, (c0 + c1) / 2.0])
            scored = []
            for _, col in others:
                pts_col = np.array([(r, c) for r, c in pts if int(g[r, c]) == col], dtype=float)
                scored.append((float(np.sum((pts_col.mean(axis=0) - rc) ** 2)), col))
            scored.sort()
            center_col = scored[0][1]
            indicator_col = scored[1][1]
        else:
            indicator_col = others[0][1]
            center_col = others[1][1]

        center = np.array([(r, c) for r, c in pts if int(g[r, c]) == center_col], dtype=float)
        indic = np.array([(r, c) for r, c in pts if int(g[r, c]) == indicator_col], dtype=float)
        if len(center) == 0 or len(indic) == 0:
            continue
        vr, vc = (indic.mean(axis=0) - center.mean(axis=0)).tolist()
        dr = 0 if abs(vr) < 1e-6 else (1 if vr > 0 else -1)
        dc = 0 if abs(vc) < 1e-6 else (1 if vc > 0 else -1)
        if dr == 0 and dc == 0:
            continue

        prev = best.get(fill_col)
        if prev is None or len(pts) > prev[0]:
            best[fill_col] = (len(pts), (dr, dc))
    return {k: v for k, (_, v) in best.items()}


def _normalize_gravity_dir(dr: int, dc: int) -> tuple[int, int]:
    if dr != 0 and dc != 0:
        return (1 if dr > 0 else -1), 0
    if dr != 0:
        return (1 if dr > 0 else -1), 0
    if dc != 0:
        return 0, (1 if dc > 0 else -1)
    raise ValueError("zero direction")


def _depth_key(p: tuple[int, int], dr: int, dc: int) -> int:
    return dr * p[0] + dc * p[1]


def _perp_key(p: tuple[int, int], dr: int, dc: int) -> int:
    return p[1] if dr != 0 else p[0]


def _lateral_dirs(dr: int, dc: int) -> list[tuple[int, int, int]]:
    if dr != 0:
        return [(0, -1, 0), (0, 1, 1)]
    return [(-1, 0, 0), (1, 0, 1)]


def _find_spill(
    region_set: set[tuple[int, int]],
    pos: tuple[int, int],
    dr: int,
    dc: int,
) -> tuple[int, int] | None:
    r0, c0 = pos
    best: tuple[int, int, int, int] | None = None
    for sdr, sdc, side_rank in _lateral_dirs(dr, dc):
        r, c = r0, c0
        dist = 0
        while True:
            r += sdr
            c += sdc
            dist += 1
            here = (r, c)
            if here not in region_set:
                break
            if (r + dr, c + dc) in region_set:
                cand = (dist, side_rank, r, c)
                if best is None or cand < best:
                    best = cand
                break
    if best is None:
        return None
    return best[2], best[3]


def _independent_sink(
    region_set: set[tuple[int, int]],
    start: tuple[int, int],
    dr: int,
    dc: int,
) -> tuple[int, int]:
    r, c = start
    for _ in range(len(region_set) * 4 + 4):
        while (r + dr, c + dc) in region_set:
            r += dr
            c += dc
        spill = _find_spill(region_set, (r, c), dr, dc)
        if spill is None:
            return r, c
        nr, nc = spill
        if (nr, nc) == (r, c):
            return r, c
        r, c = nr, nc
    return r, c


def _directional_slide_points(
    region_set: set[tuple[int, int]],
    points: list[tuple[int, int]],
    dr: int,
    dc: int,
) -> list[tuple[int, int]]:
    order = sorted(points, key=lambda p: (_depth_key(p, dr, dc), p[0], p[1]), reverse=True)
    occupied: set[tuple[int, int]] = set()
    out = []
    for r, c in order:
        while (r + dr, c + dc) in region_set and (r + dr, c + dc) not in occupied:
            r += dr
            c += dc
        occupied.add((r, c))
        out.append((r, c))
    return out


def _region_infos(g: np.ndarray, bg: int) -> list[dict]:
    h, w = g.shape
    passable = (g == bg) | (g == MOVER)
    infos = []
    for pts in components_mask(passable, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        movers = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == MOVER]
        if not movers:
            continue
        pset = {(int(r), int(c)) for r, c in pts}
        border_counts: dict[int, int] = {}
        for r, c in pts:
            for nr, nc in neighbors4(r, c):
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if (nr, nc) in pset:
                    continue
                v = int(g[nr, nc])
                if v in (bg, MOVER):
                    continue
                border_counts[v] = border_counts.get(v, 0) + 1
        if not border_counts:
            continue
        infos.append(
            {
                "pset": pset,
                "movers": movers,
                "border_col": max(border_counts, key=border_counts.get),
            }
        )
    return infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move yellow pixels by independent gravity flow inside each colored chamber.

    The legend color still selects the chamber family, but the legend direction now means the
    gravity axis. Each yellow solves its own path in the empty chamber, so multiple yellows may
    collapse onto one sink and only one yellow remains visible there.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    dirs = {col: _normalize_gravity_dir(*vec) for col, vec in _legend_directions(g, bg).items()}
    out = g.copy()
    for info in _region_infos(g, bg):
        border_col = info["border_col"]
        if border_col not in dirs:
            continue
        dr, dc = dirs[border_col]
        for r, c in info["movers"]:
            out[r, c] = bg
        for src in info["movers"]:
            rr, cc = _independent_sink(info["pset"], src, dr, dc)
            out[rr, cc] = MOVER
    return out


def _legend_patch(fill: int, dr: int, dc: int) -> np.ndarray:
    g = np.full((5, 5), fill, dtype=int)
    g[1, 1] = 1
    g[1, 2] = 1
    g[1, 3] = 2
    g[3, 1] = 1
    g[3, 2] = 1
    g[3, 3] = 2
    rot = {(0, 1): 0, (-1, 0): 1, (0, -1): 2, (1, 0): 3}
    return np.rot90(g, rot[(dr, dc)])


def _corner_origin(corner: str, ph: int, pw: int) -> tuple[int, int]:
    if corner == "TL":
        return 0, 0
    if corner == "TR":
        return 0, GRID_W - pw
    if corner == "BL":
        return GRID_H - ph, 0
    if corner == "BR":
        return GRID_H - ph, GRID_W - pw
    raise ValueError(corner)


def _rotation_steps(dr: int, dc: int) -> int:
    return {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}[(dr, dc)]


def _build_region(anchor: tuple[int, int], wall_color: int, dr: int, dc: int) -> dict:
    mask = np.rot90(CANONICAL_FUNNEL, _rotation_steps(dr, dc))
    ar, ac = anchor
    interior = {
        (ar + int(r), ac + int(c))
        for r, c in np.argwhere(mask)
    }
    walls: set[tuple[int, int]] = set()
    for r, c in interior:
        for nr, nc in neighbors4(r, c):
            if (nr, nc) not in interior:
                walls.add((int(nr), int(nc)))

    far_depth = min(_depth_key(p, dr, dc) for p in interior)
    far_tier = sorted(
        [p for p in interior if _depth_key(p, dr, dc) == far_depth],
        key=lambda p: (_perp_key(p, dr, dc), p[0], p[1]),
    )
    chosen_sources = [far_tier[0], far_tier[len(far_tier) // 2], far_tier[-1]]

    return {
        "color": wall_color,
        "dir": (dr, dc),
        "interior": interior,
        "walls": walls,
        "sources": sorted(set(chosen_sources)),
    }


def _paint_region(grid: np.ndarray, region: dict) -> None:
    for r, c in region["walls"]:
        grid[r, c] = region["color"]
    for r, c in region["sources"]:
        grid[r, c] = MOVER


def _baseline_directional_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    dirs = {col: _normalize_gravity_dir(*vec) for col, vec in _legend_directions(g, bg).items()}
    out = g.copy()
    for info in _region_infos(g, bg):
        border_col = info["border_col"]
        if border_col not in dirs:
            continue
        dr, dc = dirs[border_col]
        for r, c in info["movers"]:
            out[r, c] = bg
        for r, c in _directional_slide_points(info["pset"], info["movers"], dr, dc):
            out[r, c] = MOVER
    return out


def _valid_region_layout(regions: list[dict]) -> bool:
    occupied: set[tuple[int, int]] = set()
    for region in regions:
        cells = region["interior"] | region["walls"]
        if any(r <= 0 or r >= GRID_H - 1 or c <= 0 or c >= GRID_W - 1 for r, c in cells):
            return False
        if occupied & cells:
            return False
        occupied |= cells
    return True


def _validate_example(grid: np.ndarray, regions: list[dict]) -> bool:
    out = main(grid)
    struct_mask = (grid != 0) & (grid != MOVER)
    if not np.array_equal(out[struct_mask], grid[struct_mask]):
        return False

    in_y = int((grid == MOVER).sum())
    out_y = int((out == MOVER).sum())
    if out_y >= in_y:
        return False
    if out_y != len(regions):
        return False
    if np.array_equal(out, _baseline_directional_output(grid)):
        return False

    lateral_seen = False
    for region in regions:
        dr, dc = region["dir"]
        sinks = {_independent_sink(region["interior"], src, dr, dc) for src in region["sources"]}
        if len(sinks) != 1:
            return False
        sink = next(iter(sinks))
        for src in region["sources"]:
            if _perp_key(src, dr, dc) != _perp_key(sink, dr, dc):
                lateral_seen = True
    return lateral_seen


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid = np.zeros((GRID_H, GRID_W), dtype=int)

    n_legends = rng.choice([1, 2])
    if n_legends == 1:
        wall_colors = [rng.choice(WALL_COLORS)]
        dirs = [rng.choice(CARDINAL_DIRS)]
        region_specs = [(wall_colors[0], dirs[0]), (wall_colors[0], dirs[0])]
    else:
        wall_colors = rng.sample(WALL_COLORS, 2)
        dirs = rng.sample(CARDINAL_DIRS, 2)
        region_specs = [(wall_colors[0], dirs[0]), (wall_colors[1], dirs[1])]

    for corner, wall_color, (dr, dc) in zip(rng.sample(CORNERS, n_legends), wall_colors, dirs):
        patch = _legend_patch(wall_color, dr, dc)
        r0, c0 = _corner_origin(corner, *patch.shape)
        paste(grid, patch, r0, c0)

    regions = [
        _build_region(anchor, wall_color, dr, dc)
        for anchor, (wall_color, (dr, dc)) in zip(REGION_ANCHORS, region_specs)
    ]
    if not _valid_region_layout(regions):
        return None

    for region in regions:
        _paint_region(grid, region)
    if not _validate_example(grid, regions):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exceeded retries")
