from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass

import numpy as np


BG = 0
FRAME = 1
CTRL = 2
CORRIDOR = 3


@dataclass(frozen=True)
class Rect:
    name: str
    r0: int
    c0: int
    r1: int
    c1: int

    @property
    def h(self) -> int:
        return self.r1 - self.r0 + 1

    @property
    def w(self) -> int:
        return self.c1 - self.c0 + 1

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.r0, self.c0, self.r1, self.c1)


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _interval_overlap(a0: int, a1: int, b0: int, b1: int) -> tuple[int, int] | None:
    lo = max(a0, b0)
    hi = min(a1, b1)
    return (lo, hi) if lo <= hi else None


def _interval_gap(a0: int, a1: int, b0: int, b1: int) -> int:
    if a1 < b0:
        return b0 - a1 - 1
    if b1 < a0:
        return a0 - b1 - 1
    return 0


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _extract_rectangles(grid: np.ndarray) -> list[Rect]:
    comps = _components4(grid == FRAME)
    rects: list[Rect] = []
    for comp in comps:
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 + 1 < 4 or c1 - c0 + 1 < 4:
            raise AssertionError("frame too small")
        expected = set()
        for r in range(r0, r1 + 1):
            expected.add((r, c0))
            expected.add((r, c1))
        for c in range(c0, c1 + 1):
            expected.add((r0, c))
            expected.add((r1, c))
        if set(comp) != expected:
            raise AssertionError("blue component is not a 1-cell hollow rectangle")
        rects.append(Rect("", r0, c0, r1, c1))
    rects.sort(key=lambda rect: (rect.r0, rect.c0, rect.r1, rect.c1))
    return [Rect(str(i), rect.r0, rect.c0, rect.r1, rect.c1) for i, rect in enumerate(rects)]


def _rect_at(rects: list[Rect], r: int, c: int) -> Rect:
    for rect in rects:
        if rect.r0 <= r <= rect.r1 and rect.c0 <= c <= rect.c1:
            if r in (rect.r0, rect.r1) or c in (rect.c0, rect.c1):
                return rect
    raise AssertionError("wall cell not on any rectangle")


def _door_cells(rect: Rect, hit: tuple[int, int], ray_dir: tuple[int, int]) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    hr, hc = hit
    dr, dc = ray_dir
    if dc != 0:
        assert hc in (rect.c0, rect.c1)
        start = min(max(hr - 1, rect.r0), rect.r1 - 2)
        cells = [(start + k, hc) for k in range(3)]
        return cells, (start + 1, hc)
    assert dr != 0
    assert hr in (rect.r0, rect.r1)
    start = min(max(hc - 1, rect.c0), rect.c1 - 2)
    cells = [(hr, start + k) for k in range(3)]
    return cells, (hr, start + 1)


def _edge_key(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return tuple(sorted((a, b)))  # type: ignore[return-value]


def _analyze_casts(grid: np.ndarray, rects: list[Rect]) -> list[dict]:
    controls = sorted(map(tuple, np.argwhere(grid == CTRL)))
    if len(controls) != len(rects):
        raise AssertionError("expected one control per rectangle")

    casts: list[dict] = []
    for control in controls:
        r, c = control
        wall_neighbors = [(nr, nc) for nr, nc in _neighbors4(r, c) if _in_bounds(grid.shape, nr, nc) and grid[nr, nc] == FRAME]
        if len(wall_neighbors) != 1:
            raise AssertionError("control must touch exactly one wall")
        wr, wc = wall_neighbors[0]
        source = _rect_at(rects, wr, wc)
        dr, dc = r - wr, c - wc
        if (dr, dc) not in {(-1, 0), (1, 0), (0, -1), (0, 1)}:
            raise AssertionError("invalid control direction")

        if dr == -1:
            assert wr == source.r0
        elif dr == 1:
            assert wr == source.r1
        elif dc == -1:
            assert wc == source.c0
        else:
            assert wc == source.c1

        source_door_cells, source_center = _door_cells(source, (wr, wc), (dr, dc))

        rr, cc = r + dr, c + dc
        hit = None
        target = None
        while _in_bounds(grid.shape, rr, cc):
            if grid[rr, cc] == FRAME:
                target = _rect_at(rects, rr, cc)
                if target.name == source.name:
                    raise AssertionError("ray hit source rectangle")
                hit = (rr, cc)
                break
            if grid[rr, cc] != BG:
                raise AssertionError("ray path must be clear background")
            rr += dr
            cc += dc
        if hit is None or target is None:
            raise AssertionError("control ray must hit another frame")

        door_cells, door_center = _door_cells(target, hit, (dr, dc))
        casts.append(
            {
                "control": control,
                "source": source.name,
                "target": target.name,
                "source_wall": (wr, wc),
                "source_door_cells": tuple(source_door_cells),
                "source_center": source_center,
                "hit": hit,
                "ray_dir": (dr, dc),
                "target_door_cells": tuple(door_cells),
                "target_center": door_center,
                "shifted": (source_center != (wr, wc)) or (door_center != hit),
            }
        )
    return casts


class _DSU:
    def __init__(self, items: list[str]):
        self.parent = {item: item for item in items}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        self.parent[rb] = ra
        return True


def _corridor_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r1, c1 = a
    r2, c2 = b
    out: list[tuple[int, int]] = []
    step_c = 1 if c2 >= c1 else -1
    for c in range(c1, c2 + step_c, step_c):
        out.append((r1, c))
    step_r = 1 if r2 >= r1 else -1
    for r in range(r1, r2 + step_r, step_r):
        if (r, c2) != (r1, c2):
            out.append((r, c2))
    return out


def _solve_core(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    bg = _get_bg(g)
    assert bg == BG

    rects = _extract_rectangles(g)
    assert 4 <= len(rects) <= 5

    casts = _analyze_casts(g, rects)
    source_counts = Counter(cast["source"] for cast in casts)
    assert all(source_counts[rect.name] == 1 for rect in rects)

    out = g.copy()
    for cast in casts:
        out[cast["control"]] = BG
    for cast in casts:
        for cell in cast["source_door_cells"]:
            out[cell] = BG
        for cell in cast["target_door_cells"]:
            out[cell] = BG

    edge_map: dict[tuple[str, str], dict] = {}
    for cast in casts:
        a = cast["source"]
        b = cast["target"]
        key = tuple(sorted((a, b)))
        da = cast["source_center"]
        db = cast["target_center"]
        lo, hi = _edge_key(da, db)
        candidate = {
            "u": key[0],
            "v": key[1],
            "a_center": da,
            "b_center": db,
            "weight": abs(da[0] - db[0]) + abs(da[1] - db[1]),
            "tie0": lo,
            "tie1": hi,
        }
        prev = edge_map.get(key)
        if prev is None or (
            candidate["weight"],
            candidate["tie0"],
            candidate["tie1"],
            candidate["a_center"],
            candidate["b_center"],
        ) < (
            prev["weight"],
            prev["tie0"],
            prev["tie1"],
            prev["a_center"],
            prev["b_center"],
        ):
            edge_map[key] = candidate

    edges = sorted(
        edge_map.values(),
        key=lambda edge: (
            edge["weight"],
            edge["tie0"][0],
            edge["tie0"][1],
            edge["tie1"][0],
            edge["tie1"][1],
            edge["u"],
            edge["v"],
        ),
    )
    assert len(edges) >= len(rects) - 1

    dsu = _DSU([rect.name for rect in rects])
    chosen = []
    for edge in edges:
        if dsu.union(edge["u"], edge["v"]):
            chosen.append(edge)
    assert len(chosen) == len(rects) - 1

    for edge in chosen:
        cells = _corridor_cells(edge["a_center"], edge["b_center"])
        for r, c in cells:
            if out[r, c] == FRAME:
                raise AssertionError("corridor crossed an intact wall")
            out[r, c] = CORRIDOR

    meta = {
        "rects": rects,
        "casts": casts,
        "edges": edges,
        "chosen": chosen,
        "opened": out.copy(),
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast outward control rays to carve room doors, then connect the rooms through the shortest doorway tree.

    Input structure:
    - Black background contains 4-5 disconnected blue hollow rectangles, each one cell thick.
    - Every rectangle has exactly one red control pixel sitting just outside one wall cell, orthogonally adjacent to that wall.
    - Each control points away from its own rectangle and its ray reaches another rectangle wall before the grid edge.

    Rule:
    - For each red control pixel, identify its adjacent blue wall and cast a ray outward, away from that rectangle, until the ray first hits another blue wall.
    - Use that control's own wall position as the source-side doorway center, erase a 3-cell doorway there, and also erase a 3-cell doorway on the hit wall, centered on the hit when possible and otherwise shifted inward just enough to fit; then erase the red control pixel.
    - After all doorway pairs are opened, treat rectangles as nodes and treat each control-ray relation as one direct candidate edge between its source and target rectangles.
    - Weight each direct edge by the Manhattan distance between its source doorway center and hit-wall doorway center, choose the minimum spanning tree with the stated doorway-position tie breaks, and draw each chosen corridor in green using a horizontal-then-vertical orthogonal path that never paints intact blue walls.

    Key insight:
    - The red controls do not merely mark directions: each one defines a doorway pair, because its own wall fixes one corridor endpoint while the ray hit fixes the other, and those doorway pairs determine the graph used by the corridor MST.
    """
    out, _ = _solve_core(np.asarray(grid, dtype=int))
    return out


def _draw_frame(grid: np.ndarray, rect: Rect) -> None:
    grid[rect.r0 : rect.r1 + 1, rect.c0] = FRAME
    grid[rect.r0 : rect.r1 + 1, rect.c1] = FRAME
    grid[rect.r0, rect.c0 : rect.c1 + 1] = FRAME
    grid[rect.r1, rect.c0 : rect.c1 + 1] = FRAME


def _rect_row_gap(a: Rect, b: Rect) -> int:
    return _interval_gap(a.r0, a.r1, b.r0, b.r1)


def _rect_col_gap(a: Rect, b: Rect) -> int:
    return _interval_gap(a.c0, a.c1, b.c0, b.c1)


def _pairwise_clear(rects: list[Rect]) -> bool:
    for i, a in enumerate(rects):
        for b in rects[i + 1 :]:
            row_gap = _rect_row_gap(a, b)
            col_gap = _rect_col_gap(a, b)
            if row_gap == 0 and col_gap < 4:
                return False
            if col_gap == 0 and row_gap < 4:
                return False
            if row_gap > 0 and col_gap > 0 and (row_gap < 3 or col_gap < 3):
                return False
    return True


def _required_gap(src: Rect, dst: Rect, direction: str) -> bool:
    if direction == "E":
        return dst.c0 - src.c1 - 1 >= 4
    if direction == "W":
        return src.c0 - dst.c1 - 1 >= 4
    if direction == "S":
        return dst.r0 - src.r1 - 1 >= 4
    if direction == "N":
        return src.r0 - dst.r1 - 1 >= 4
    raise ValueError(direction)


def _perp_overlap(src: Rect, dst: Rect, direction: str) -> tuple[int, int] | None:
    if direction in ("E", "W"):
        return _interval_overlap(src.r0, src.r1, dst.r0, dst.r1)
    return _interval_overlap(src.c0, src.c1, dst.c0, dst.c1)


def _candidate_hit_coords(src: Rect, dst: Rect, direction: str, prefer_shift: bool) -> list[int]:
    overlap = _perp_overlap(src, dst, direction)
    assert overlap is not None
    lo, hi = overlap
    values = list(range(lo, hi + 1))
    if direction in ("E", "W"):
        endpoints = [v for v in (dst.r0, dst.r1) if lo <= v <= hi]
    else:
        endpoints = [v for v in (dst.c0, dst.c1) if lo <= v <= hi]
    mids = [values[len(values) // 2]]
    if len(values) >= 2:
        mids.append(values[(len(values) - 1) // 2])

    ordered: list[int] = []
    pools = [endpoints, values, mids] if prefer_shift else [mids, values, endpoints]
    for pool in pools:
        for v in pool:
            if v not in ordered:
                ordered.append(v)
    return ordered


def _control_from_hit(src: Rect, dst: Rect, direction: str, coord: int) -> tuple[tuple[int, int], tuple[int, int]]:
    if direction == "E":
        return (coord, src.c1 + 1), (coord, dst.c0)
    if direction == "W":
        return (coord, src.c0 - 1), (coord, dst.c1)
    if direction == "S":
        return (src.r1 + 1, coord), (dst.r0, coord)
    if direction == "N":
        return (src.r0 - 1, coord), (dst.r1, coord)
    raise ValueError(direction)


def _clear_between(src: Rect, dst: Rect, direction: str, control: tuple[int, int], hit: tuple[int, int]) -> bool:
    cr, cc = control
    hr, hc = hit
    if direction in ("E", "W"):
        step = 1 if hc > cc else -1
        for c in range(cc + step, hc, step):
            if c < 0:
                return False
    else:
        step = 1 if hr > cr else -1
        for r in range(cr + step, hr, step):
            if r < 0:
                return False
    return True


def _topology_spec(use_five: bool) -> tuple[list[str], list[tuple[str, str, str]]]:
    if use_five:
        names = ["TL", "TM", "TR", "BR", "BL"]
        edges = [("TL", "TM", "E"), ("TM", "TR", "E"), ("TR", "BR", "S"), ("BR", "BL", "W"), ("BL", "TL", "N")]
    else:
        names = ["TL", "TR", "BR", "BL"]
        edges = [("TL", "TR", "E"), ("TR", "BR", "S"), ("BR", "BL", "W"), ("BL", "TL", "N")]
    return names, edges


def _make_rect(name: str, r0: int, c0: int, h: int, w: int) -> Rect:
    return Rect(name, r0, c0, r0 + h - 1, c0 + w - 1)


def _place_layout(use_five: bool, want_tie: bool) -> tuple[tuple[int, int], list[Rect]]:
    for _ in range(200):
        if use_five:
            h = random.randint(21, 26)
            w = random.randint(24, 29)
        else:
            h = random.randint(20, 26)
            w = random.randint(20, 26)
        names, _ = _topology_spec(use_five)

        if want_tie and not use_five:
            dims = {
                "TL": (random.randint(4, 7), random.randint(4, 7)),
                "TR": (random.randint(4, 7), random.randint(4, 7)),
            }
            dims["BR"] = dims["TL"]
            dims["BL"] = dims["TR"]
        else:
            dims = {name: (random.randint(4, 7), random.randint(4, 7)) for name in names}

        top_base = random.randint(1, 3)
        left_base = random.randint(1, 3)
        max_top_h = max(dims[name][0] for name in names if name in {"TL", "TM", "TR"})
        max_left_w = max(dims[name][1] for name in names if name in {"TL", "BL"})
        bottom_base_min = top_base + max_top_h + 5
        if use_five:
            bottom_base_max = h - max(dims["BL"][0], dims["BR"][0]) - 2
            right_base_min = left_base + max_left_w + max(dims["TM"][1], dims["TR"][1]) + 7
            right_base_max = w - max(dims["TR"][1], dims["BR"][1]) - 2
            mid_base_min = left_base + max_left_w + 5
            mid_base_max = right_base_max - dims["TM"][1] - 5
            if bottom_base_min > bottom_base_max or mid_base_min > mid_base_max:
                continue
            bottom_base = random.randint(bottom_base_min, bottom_base_max)
            mid_base = random.randint(mid_base_min, mid_base_max)
            right_base_min = max(right_base_min, mid_base + dims["TM"][1] + 5)
            if right_base_min > right_base_max:
                continue
            right_base = random.randint(right_base_min, right_base_max)
        else:
            bottom_base_max = h - max(dims["BL"][0], dims["BR"][0]) - 2
            right_base_min = left_base + max_left_w + 5
            right_base_max = w - max(dims["TR"][1], dims["BR"][1]) - 2
            if bottom_base_min > bottom_base_max or right_base_min > right_base_max:
                continue
            bottom_base = random.randint(bottom_base_min, bottom_base_max)
            right_base = random.randint(right_base_min, right_base_max)

        if want_tie and not use_five:
            rows = {"TL": top_base, "TR": top_base, "BL": bottom_base, "BR": bottom_base}
            cols = {"TL": left_base, "BL": left_base, "TR": right_base, "BR": right_base}
        else:
            top_offsets = {name: random.randint(0, 2) for name in names if name in {"TL", "TM", "TR"}}
            bottom_offsets = {name: random.randint(0, 2) for name in names if name in {"BL", "BR"}}
            left_offsets = {name: random.randint(0, 2) for name in names if name in {"TL", "BL"}}
            right_offsets = {name: random.randint(0, 2) for name in names if name in {"TR", "BR"}}
            rows = {
                "TL": top_base + top_offsets.get("TL", 0),
                "TR": top_base + top_offsets.get("TR", 0),
                "BL": bottom_base + bottom_offsets.get("BL", 0),
                "BR": bottom_base + bottom_offsets.get("BR", 0),
            }
            cols = {
                "TL": left_base + left_offsets.get("TL", 0),
                "BL": left_base + left_offsets.get("BL", 0),
                "TR": right_base + right_offsets.get("TR", 0),
                "BR": right_base + right_offsets.get("BR", 0),
            }
            if use_five:
                rows["TM"] = top_base + top_offsets.get("TM", 0)
                cols["TM"] = mid_base + random.randint(0, 2)

        rects = [_make_rect(name, rows[name], cols[name], *dims[name]) for name in names]
        if any(rect.r0 < 1 or rect.c0 < 1 or rect.r1 >= h - 1 or rect.c1 >= w - 1 for rect in rects):
            continue
        if not _pairwise_clear(rects):
            continue

        by_name = {rect.name: rect for rect in rects}
        _, edge_specs = _topology_spec(use_five)
        ok = True
        for src_name, dst_name, direction in edge_specs:
            src = by_name[src_name]
            dst = by_name[dst_name]
            if _perp_overlap(src, dst, direction) is None or not _required_gap(src, dst, direction):
                ok = False
                break
        if ok:
            return (h, w), rects
    raise RuntimeError("failed to place layout")


def _build_input(use_five: bool, want_shift: bool, want_tie: bool) -> np.ndarray | None:
    try:
        shape, rects = _place_layout(use_five, want_tie)
    except RuntimeError:
        return None
    grid = np.zeros(shape, dtype=int)
    by_name = {rect.name: rect for rect in rects}
    for rect in rects:
        _draw_frame(grid, rect)

    _, edge_specs = _topology_spec(use_five)
    shift_used = False
    for src_name, dst_name, direction in edge_specs:
        src = by_name[src_name]
        dst = by_name[dst_name]
        coord_options = _candidate_hit_coords(src, dst, direction, prefer_shift=want_shift and not shift_used)
        placed = False
        for coord in coord_options:
            control, hit = _control_from_hit(src, dst, direction, coord)
            if not _in_bounds(shape, control[0], control[1]) or grid[control] != BG:
                continue
            if not _clear_between(src, dst, direction, control, hit):
                continue
            door_cells, door_center = _door_cells(dst, hit, {"E": (0, 1), "W": (0, -1), "S": (1, 0), "N": (-1, 0)}[direction])
            shifted_here = door_center != hit
            if want_shift and not shift_used and not shifted_here:
                continue
            grid[control] = CTRL
            shift_used = shift_used or shifted_here
            placed = True
            break
        if not placed:
            return None

    try:
        out, meta = _solve_core(grid)
    except AssertionError:
        return None

    if want_shift and not any(cast["shifted"] for cast in meta["casts"]):
        return None
    if len(meta["edges"]) != len(rects):
        return None
    weights = [edge["weight"] for edge in meta["edges"]]
    if want_tie and len(weights) == len(set(weights)):
        return None
    if np.count_nonzero(out == CORRIDOR) < len(rects):
        return None
    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        use_five = random.random() < 0.5
        want_shift = random.random() < 0.6
        want_tie = (not use_five) and random.random() < 0.35
        grid = _build_input(use_five, want_shift, want_tie)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
