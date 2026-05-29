# Combo variant of 4a21e3da (index 24)
# combo_spec: D11/2+D13/0
# Name: 带洞精灵+连通分量归属

from __future__ import annotations

from collections import Counter, deque
import random
import sys

import numpy as np

SOLVER_DIR = "/home/lixinyu/snap/new/solvers"
if SOLVER_DIR not in sys.path:
    sys.path.append(SOLVER_DIR)

from arc_utils import colors_present, get_bg


# Design analysis for the fused variant
# ------------------------------------
# a) Concrete behavior and trace
#    Original task:
#    - Border control pixels define one vertical and/or one horizontal split line.
#    - Rays paint background cells on those lines all the way to the farthest fg pixel.
#    - Off-line fg is grouped only by geometric rectangle/quadrant.
#    - A whole rectangle is kept iff that rectangle touches a control edge.
#    Fused variant:
#    - The sprite may contain an enclosed background hole.
#    - A ray does NOT paint through such enclosed hole cells.
#    - Off-line fg is still grouped by geometric region for its destination corner,
#      but retention is decided per 4-connected component, not per whole region.
#    - A component is kept iff at least one of its cells is 4-neighbor-adjacent to
#      a fg pixel lying on a split line; otherwise that component disappears.
#    Small mental example:
#    - Put a 5x5 ring around the split intersection. The split row/col pass through
#      the ring's 3x3 hole, so the output ray visibly contains a background gap.
#    - The ring contributes one kept corner piece in every quadrant because each
#      quadrant corner of the ring touches a split-line fg pixel at the ring edge.
#    - Add a detached same-color island in one quadrant. It lies in the same
#      geometric side but does not touch any split-line fg, so it vanishes.
#    - If the chosen controls are top+left, the bottom-right quadrant would vanish
#      under the original rule, yet it is kept here because its component touches
#      split-line fg. This makes D13/0 visibly non-degenerate.
# b) Complexity calibration
#    - The generator uses one dominant holed central component plus 1-2 detached
#      islands. That keeps the scene readable while still forcing the solver to
#      reason about holes and component ownership.
#    - Only four region categories exist (TL/TR/BL/BR), with one keep/vanish test:
#      “touches split-line fg or not”. This is a small discrete rule that can be
#      inferred from a few examples.
# c) Implicit assumptions broken from the original solver
#    - Original ray drawing assumed every bg cell on the path should become ctrl.
#      This fails once enclosed holes exist; hole cells must remain bg.
#    - Original region handling assumed each region behaves as one lump. This fails
#      when a region contains multiple disconnected fg components.
#    - Original keep test depended on border-edge adjacency. The fused variant
#      replaces that with topological adjacency to split-line fg.
# d) Placement legality
#    - The generator keeps every fg object inside a central 13x13 window with
#      margin >= 3 from the 20x20 border and never places fg on the control pixels.
#    - Detached islands are generated inside quadrant-specific boxes and rejected if
#      they touch the kept central component or any split-line fg.
# e) Visual observability
#    - The ring hole creates an obvious missing segment inside the ctrl ray.
#    - The opposite-to-control quadrant still receives a moved piece, which directly
#      contradicts the original edge-based rule and makes D13/0 visible.
#    - Detached islands disappear while nearby connected parts in the same quadrant
#      survive, exposing the component-based ownership criterion.
# f) Visual cleanliness
#    - There are only three colors: bg=1, ctrl=2, fg=7.
#    - The central ring is separated from detached islands by at least one bg cell.
#    - Each kept region contributes at most one kept component, avoiding ambiguous
#      overlap after sliding to corners.
# g) New assumptions introduced by the new solver
#    - It assumes there is at most one vertical split and at most one horizontal
#      split, matching the original task family.
#    - It assumes disconnected islands that should vanish are not adjacent to any
#      split-line fg. The generator enforces this explicitly.
#    - It assumes “holes” are enclosed bg components. The generator always creates
#      a central 5x5 ring with a 3x3 enclosed hole, so the hole detector is stable.


Coord = tuple[int, int]
Region = tuple[str, str]

BG = 1
CTRL = 2
FG = 7


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _find_ctrl_fg_bg(g: np.ndarray) -> tuple[int, int, int]:
    bg = int(get_bg(g))
    nonbg = colors_present(g, exclude={bg})
    border_cols = set(map(int, g[0, :])) | set(map(int, g[-1, :])) | set(map(int, g[:, 0])) | set(map(int, g[:, -1]))
    border_cols.discard(bg)
    if not border_cols:
        raise ValueError("missing border control color")
    ctrl = int(sorted(border_cols)[0])
    fg_candidates = [int(c) for c in nonbg if int(c) != ctrl]
    if len(fg_candidates) != 1:
        raise ValueError("expected exactly one fg color")
    fg = fg_candidates[0]
    return bg, ctrl, fg


def _split_info(g: np.ndarray, ctrl: int) -> tuple[list[Coord], set[str], int | None, int | None]:
    H, W = g.shape
    ctrl_pts = [(int(r), int(c)) for r, c in np.argwhere(g == int(ctrl))]
    edges: set[str] = set()
    c_split: int | None = None
    r_split: int | None = None
    for r, c in ctrl_pts:
        if r == 0:
            edges.add("top")
            c_split = c
        elif r == H - 1:
            edges.add("bottom")
            c_split = c
        elif c == 0:
            edges.add("left")
            r_split = r
        elif c == W - 1:
            edges.add("right")
            r_split = r
        else:
            raise ValueError("control pixel must lie on border")
    return ctrl_pts, edges, c_split, r_split


def _hole_mask(g: np.ndarray, fg: int) -> np.ndarray:
    H, W = g.shape
    seen = np.zeros((H, W), dtype=bool)
    holes = np.zeros((H, W), dtype=bool)
    for r in range(H):
        for c in range(W):
            if seen[r, c] or int(g[r, c]) == int(fg):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            touches_border = False
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                if rr in (0, H - 1) or cc in (0, W - 1):
                    touches_border = True
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and int(g[nr, nc]) != int(fg):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches_border:
                for rr, cc in comp:
                    holes[rr, cc] = True
    return holes


def _draw_rays(g: np.ndarray, out: np.ndarray, ctrl_pts: list[Coord], ctrl: int, fg: int, holes: np.ndarray) -> None:
    H, W = g.shape
    for r, c in ctrl_pts:
        if r == 0:
            rows = [int(rr) for rr in np.argwhere(g[:, c] == int(fg)).flatten()]
            if rows:
                stop = max(rows)
                for rr in range(0, stop + 1):
                    if int(g[rr, c]) == int(get_bg(g)) and not bool(holes[rr, c]):
                        out[rr, c] = int(ctrl)
        elif r == H - 1:
            rows = [int(rr) for rr in np.argwhere(g[:, c] == int(fg)).flatten()]
            if rows:
                stop = min(rows)
                for rr in range(H - 1, stop - 1, -1):
                    if int(g[rr, c]) == int(get_bg(g)) and not bool(holes[rr, c]):
                        out[rr, c] = int(ctrl)
        elif c == 0:
            cols = [int(cc) for cc in np.argwhere(g[r, :] == int(fg)).flatten()]
            if cols:
                stop = max(cols)
                for cc in range(0, stop + 1):
                    if int(g[r, cc]) == int(get_bg(g)) and not bool(holes[r, cc]):
                        out[r, cc] = int(ctrl)
        elif c == W - 1:
            cols = [int(cc) for cc in np.argwhere(g[r, :] == int(fg)).flatten()]
            if cols:
                stop = min(cols)
                for cc in range(W - 1, stop - 1, -1):
                    if int(g[r, cc]) == int(get_bg(g)) and not bool(holes[r, cc]):
                        out[r, cc] = int(ctrl)
    for r, c in ctrl_pts:
        out[r, c] = int(ctrl)


def _is_on_split(r: int, c: int, r_split: int | None, c_split: int | None) -> bool:
    return (r_split is not None and r == int(r_split)) or (c_split is not None and c == int(c_split))


def _region_of(r: int, c: int, r_split: int | None, c_split: int | None) -> Region:
    hs = "A"
    vs = "A"
    if r_split is not None:
        hs = "T" if r < int(r_split) else "B"
    if c_split is not None:
        vs = "L" if c < int(c_split) else "R"
    return hs, vs


def _touches_edges(region: Region) -> set[str]:
    hs, vs = region
    te: set[str] = set()
    if hs in ("T", "A"):
        te.add("top")
    if hs in ("B", "A"):
        te.add("bottom")
    if vs in ("L", "A"):
        te.add("left")
    if vs in ("R", "A"):
        te.add("right")
    return te


def _pick_dest(region: Region, edges: set[str], r_split: int | None, c_split: int | None) -> Region:
    hs, vs = region
    if r_split is not None:
        dh = hs
    elif "top" in edges and "bottom" not in edges:
        dh = "T"
    elif "bottom" in edges and "top" not in edges:
        dh = "B"
    else:
        dh = "T"
    if c_split is not None:
        dv = vs
    elif "left" in edges and "right" not in edges:
        dv = "L"
    elif "right" in edges and "left" not in edges:
        dv = "R"
    else:
        dv = "L"
    return dh, dv


def _components(cells: set[Coord]) -> list[set[Coord]]:
    remaining = set(cells)
    comps: list[set[Coord]] = []
    while remaining:
        start = next(iter(remaining))
        q = deque([start])
        remaining.remove(start)
        comp = {start}
        while q:
            r, c = q.popleft()
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in remaining:
                    remaining.remove((nr, nc))
                    comp.add((nr, nc))
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _slide_component(out: np.ndarray, comp: set[Coord], region: Region, edges: set[str], r_split: int | None, c_split: int | None, fg: int) -> None:
    if not comp:
        return
    H, W = out.shape
    dh, dv = _pick_dest(region, edges, r_split, c_split)
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    dr = -min(rs) if dh == "T" else (H - 1 - max(rs))
    dc = -min(cs) if dv == "L" else (W - 1 - max(cs))
    for r, c in comp:
        out[r + dr, c + dc] = int(fg)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg, ctrl, fg = _find_ctrl_fg_bg(g)
    ctrl_pts, edges, c_split, r_split = _split_info(g, ctrl)
    holes = _hole_mask(g, fg)

    out = np.full((H, W), bg, dtype=int)
    _draw_rays(g, out, ctrl_pts, ctrl, fg, holes)

    split_line_fg: set[Coord] = set()
    for r, c in np.argwhere(g == int(fg)):
        rr = int(r)
        cc = int(c)
        if _is_on_split(rr, cc, r_split, c_split):
            split_line_fg.add((rr, cc))
            out[rr, cc] = int(fg)

    regions: dict[Region, set[Coord]] = {}
    for r, c in np.argwhere(g == int(fg)):
        rr = int(r)
        cc = int(c)
        if _is_on_split(rr, cc, r_split, c_split):
            continue
        region = _region_of(rr, cc, r_split, c_split)
        regions.setdefault(region, set()).add((rr, cc))

    for region, cells in regions.items():
        for comp in _components(cells):
            anchored = False
            for r, c in comp:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) in split_line_fg:
                        anchored = True
                        break
                if anchored:
                    break
            if anchored:
                _slide_component(out, comp, region, edges, r_split, c_split, fg)

    return out


def _region_boxes(r_split: int, c_split: int) -> dict[Region, tuple[int, int, int, int]]:
    return {
        ("T", "L"): (r_split - 6, r_split - 1, c_split - 6, c_split - 1),
        ("T", "R"): (r_split - 6, r_split - 1, c_split + 1, c_split + 6),
        ("B", "L"): (r_split + 1, r_split + 6, c_split - 6, c_split - 1),
        ("B", "R"): (r_split + 1, r_split + 6, c_split + 1, c_split + 6),
    }


def _island_boxes(r_split: int, c_split: int) -> dict[Region, tuple[int, int, int, int]]:
    return {
        ("T", "L"): (r_split - 6, r_split - 4, c_split - 6, c_split - 4),
        ("T", "R"): (r_split - 6, r_split - 4, c_split + 4, c_split + 6),
        ("B", "L"): (r_split + 4, r_split + 6, c_split - 6, c_split - 4),
        ("B", "R"): (r_split + 4, r_split + 6, c_split + 4, c_split + 6),
    }


def _grow_connected(seed: Coord, allowed: set[Coord], size: int, occupied: set[Coord]) -> set[Coord] | None:
    if seed not in allowed or seed in occupied:
        return None
    cells = {seed}
    frontier = {seed}
    while len(cells) < size:
        candidates: set[Coord] = set()
        for r, c in frontier:
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in allowed and (nr, nc) not in occupied and (nr, nc) not in cells:
                    candidates.add((nr, nc))
        if not candidates:
            return None
        pick = random.choice(sorted(candidates))
        cells.add(pick)
        frontier.add(pick)
    return cells


def _make_ring(r_split: int, c_split: int) -> tuple[set[Coord], set[Coord]]:
    ring = set()
    for r in range(r_split - 2, r_split + 3):
        for c in range(c_split - 2, c_split + 3):
            if r in (r_split - 2, r_split + 2) or c in (c_split - 2, c_split + 2):
                ring.add((r, c))
    hole = {(r, c) for r in range(r_split - 1, r_split + 2) for c in range(c_split - 1, c_split + 2)}
    return ring, hole


def _old_vanish_region(edges: set[str]) -> Region:
    hs = "B" if "top" in edges else "T"
    vs = "R" if "left" in edges else "L"
    if "bottom" in edges:
        hs = "T"
    if "right" in edges:
        vs = "L"
    return hs, vs


def _region_seed(region: Region, r_split: int, c_split: int) -> Coord:
    hs, vs = region
    sr = r_split - 2 if hs == "T" else r_split + 2
    sc = c_split - 2 if vs == "L" else c_split + 2
    return sr, sc


def _place_extensions(base: set[Coord], r_split: int, c_split: int, old_vanish: Region) -> set[Coord] | None:
    occupied = set(base)
    for region in [("T", "L"), ("T", "R"), ("B", "L"), ("B", "R")]:
        hs, vs = region
        box = _region_boxes(r_split, c_split)[region]
        r0, r1, c0, c1 = box
        allowed = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
        seed = _region_seed(region, r_split, c_split)
        allowed.add(seed)
        extra = random.randint(2, 4) if region == old_vanish else random.randint(0, 3)
        if extra == 0:
            continue
        comp = _grow_connected(seed, allowed, extra + 1, occupied - {seed})
        if comp is None:
            return None
        occupied |= comp
    return occupied


def _far_seed(region: Region, r_split: int, c_split: int) -> Coord:
    hs, vs = region
    rr = r_split - 6 if hs == "T" else r_split + 6
    cc = c_split - 6 if vs == "L" else c_split + 6
    return rr, cc


def _place_detached_islands(base: set[Coord], split_line_fg: set[Coord], r_split: int, c_split: int) -> set[Coord] | None:
    occupied = set(base)
    regions = random.sample([("T", "L"), ("T", "R"), ("B", "L"), ("B", "R")], k=random.randint(1, 2))
    islands: set[Coord] = set()
    boxes = _island_boxes(r_split, c_split)
    for region in regions:
        r0, r1, c0, c1 = boxes[region]
        allowed = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
        seed = _far_seed(region, r_split, c_split)
        if seed not in allowed or seed in occupied:
            continue
        size = random.randint(2, 3)
        comp = _grow_connected(seed, allowed, size, occupied)
        if comp is None:
            continue
        if any(any((nr, nc) in split_line_fg for nr, nc in _neighbors4(r, c)) for r, c in comp):
            continue
        if any(any((nr, nc) in base for nr, nc in _neighbors4(r, c)) for r, c in comp):
            continue
        islands |= comp
        occupied |= comp
    return islands if islands else None


def _build_input() -> np.ndarray | None:
    H = W = 20
    g = np.full((H, W), BG, dtype=int)

    vertical_edge = random.choice(["top", "bottom"])
    horizontal_edge = random.choice(["left", "right"])
    edges = {vertical_edge, horizontal_edge}
    r_split = random.choice([9, 10])
    c_split = random.choice([9, 10])

    ring, hole = _make_ring(r_split, c_split)
    old_vanish = _old_vanish_region(edges)
    kept = _place_extensions(ring, r_split, c_split, old_vanish)
    if kept is None:
        return None

    split_line_fg = {pt for pt in kept if pt[0] == r_split or pt[1] == c_split}
    islands = _place_detached_islands(kept, split_line_fg, r_split, c_split)
    if islands is None:
        return None

    fg_cells = kept | islands
    for r, c in fg_cells:
        if not (3 <= r <= H - 4 and 3 <= c <= W - 4):
            return None
        g[r, c] = FG

    ctrl_pts = []
    ctrl_pts.append((0 if vertical_edge == "top" else H - 1, c_split))
    ctrl_pts.append((r_split, 0 if horizontal_edge == "left" else W - 1))
    for r, c in ctrl_pts:
        if g[r, c] != BG:
            return None
        g[r, c] = CTRL

    # Hard guarantees for fusion behavior.
    info = analyze_input(g)
    if info["hole_ray_gap_cells"] < 3:
        return None
    if info["kept_nonedge_components"] < 1:
        return None
    if info["detached_vanish_components"] < 1:
        return None
    return g


def analyze_input(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    bg, ctrl, fg = _find_ctrl_fg_bg(g)
    ctrl_pts, edges, c_split, r_split = _split_info(g, ctrl)
    holes = _hole_mask(g, fg)

    split_line_fg: set[Coord] = set()
    regions: dict[Region, set[Coord]] = {}
    for r, c in np.argwhere(g == int(fg)):
        rr = int(r)
        cc = int(c)
        if _is_on_split(rr, cc, r_split, c_split):
            split_line_fg.add((rr, cc))
        else:
            region = _region_of(rr, cc, r_split, c_split)
            regions.setdefault(region, set()).add((rr, cc))

    kept_nonedge = 0
    detached_vanish = 0
    for region, cells in regions.items():
        region_touches = _touches_edges(region)
        for comp in _components(cells):
            anchored = any((nr, nc) in split_line_fg for r, c in comp for nr, nc in _neighbors4(r, c))
            if anchored and not (region_touches & edges):
                kept_nonedge += 1
            if not anchored:
                detached_vanish += 1

    gap_cells: set[Coord] = set()
    H, W = g.shape
    for r, c in ctrl_pts:
        if r == 0:
            rows = [int(rr) for rr in np.argwhere(g[:, c] == int(fg)).flatten()]
            if rows:
                stop = max(rows)
                for rr in range(0, stop + 1):
                    if bool(holes[rr, c]):
                        gap_cells.add((rr, c))
        elif r == H - 1:
            rows = [int(rr) for rr in np.argwhere(g[:, c] == int(fg)).flatten()]
            if rows:
                stop = min(rows)
                for rr in range(stop, H):
                    if bool(holes[rr, c]):
                        gap_cells.add((rr, c))
        elif c == 0:
            cols = [int(cc) for cc in np.argwhere(g[r, :] == int(fg)).flatten()]
            if cols:
                stop = max(cols)
                for cc in range(0, stop + 1):
                    if bool(holes[r, cc]):
                        gap_cells.add((r, cc))
        elif c == W - 1:
            cols = [int(cc) for cc in np.argwhere(g[r, :] == int(fg)).flatten()]
            if cols:
                stop = min(cols)
                for cc in range(stop, W):
                    if bool(holes[r, cc]):
                        gap_cells.add((r, cc))

    return {
        "hole_ray_gap_cells": len(gap_cells),
        "kept_nonedge_components": int(kept_nonedge),
        "detached_vanish_components": int(detached_vanish),
        "split_line_fg_pixels": int(len(split_line_fg)),
        "fg_pixels": int(np.sum(g == int(fg))),
    }


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _build_input()
        if result is not None:
            out = main(result)
            if out.shape == result.shape:
                return result
    raise RuntimeError("Failed to generate valid fused variant after 500 attempts")
