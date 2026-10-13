import random
from collections import Counter
from itertools import combinations

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
SOURCE_COLORS = [1, 2, 3, 4, 6, 7, 8]


def _most_common(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(comp: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    return min(rows), min(cols), max(rows), max(cols)


def _find_sources(g: np.ndarray, bg: int) -> list[dict]:
    h, w = g.shape
    sources = []
    occupied = np.zeros_like(g, dtype=bool)
    for r in range(h - 1):
        for c in range(w - 1):
            color = int(g[r, c])
            if color in (bg, GREY, MAROON):
                continue
            if np.all(g[r : r + 2, c : c + 2] == color) and not np.any(occupied[r : r + 2, c : c + 2]):
                sid = len(sources)
                sources.append({"id": sid, "bbox": (r, c, r + 1, c + 1), "color": color})
                occupied[r : r + 2, c : c + 2] = True
    assert sources, "expected at least one source"
    return sources


def _find_containers(g: np.ndarray, bg: int) -> list[dict]:
    containers = []
    for comp in _components(g == MAROON):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 4 or c1 - c0 < 4:
            continue
        border = set()
        for c in range(c0, c1 + 1):
            border.add((r0, c))
            border.add((r1, c))
        for r in range(r0, r1 + 1):
            border.add((r, c0))
            border.add((r, c1))
        if set(comp) != border:
            continue
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        if np.all(interior == bg):
            containers.append({"id": len(containers), "bbox": (r0, c0, r1, c1)})
    assert containers, "expected at least one hollow maroon container"
    return containers


def _cell_map(items: list[dict], include_border: bool = False) -> dict[tuple[int, int], int]:
    out = {}
    for item in items:
        r0, c0, r1, c1 = item["bbox"]
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if include_border or r in (r0, r1) or c in (c0, c1):
                    out[(r, c)] = item["id"]
    return out


def _trace_path(comp: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells = set(comp)
    neigh = {p: [] for p in cells}
    for r, c in cells:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            q = (r + dr, c + dc)
            if q in cells:
                neigh[(r, c)].append(q)
    endpoints = [p for p, ns in neigh.items() if len(ns) == 1]
    assert len(endpoints) == 2, "pipe must be a simple path"
    path = [endpoints[0]]
    prev = None
    cur = endpoints[0]
    while cur != endpoints[1]:
        choices = [p for p in neigh[cur] if p != prev]
        assert choices, "broken pipe path"
        nxt = choices[0]
        prev, cur = cur, nxt
        path.append(cur)
    return path


def _bend_count(path: list[tuple[int, int]]) -> int:
    bends = 0
    for a, b, c in zip(path, path[1:], path[2:]):
        d1 = (b[0] - a[0], b[1] - a[1])
        d2 = (c[0] - b[0], c[1] - b[1])
        if d1 != d2:
            bends += 1
    return bends


def _pipe_infos(g: np.ndarray, sources: list[dict], containers: list[dict]) -> list[dict]:
    h, w = g.shape
    source_at = _cell_map(sources, include_border=True)
    container_border_at = _cell_map(containers, include_border=False)
    infos = []
    for comp in _components(g == GREY):
        adj_sources = set()
        adj_containers = set()
        contacts = []
        for r, c in comp:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                q = (r + dr, c + dc)
                sid = source_at.get(q)
                if sid is not None:
                    adj_sources.add(sid)
                cid = container_border_at.get(q)
                if cid is not None:
                    adj_containers.add(cid)
                    contacts.append((cid, r, c))
        assert len(adj_sources) == 1 and len(adj_containers) == 1, "each pipe connects one source and one container"
        path = _trace_path(comp)
        cid = next(iter(adj_containers))
        contact_cells = [(r, c) for ccid, r, c in contacts if ccid == cid]
        contact = min(contact_cells, key=lambda p: (p[1], p[0]))
        infos.append(
            {
                "source": next(iter(adj_sources)),
                "container": cid,
                "bends": _bend_count(path),
                "contact": contact,
            }
        )
    return infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each maroon container with source colors; pipe bend count gives layer height.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common(g)
    sources = _find_sources(g, bg)
    containers = _find_containers(g, bg)
    pipes = _pipe_infos(g, sources, containers)
    source_color = {s["id"]: int(s["color"]) for s in sources}
    out = g.copy()

    by_container: dict[int, list[dict]] = {c["id"]: [] for c in containers}
    for pipe in pipes:
        by_container[pipe["container"]].append(pipe)

    for container in containers:
        r0, c0, r1, c1 = container["bbox"]
        current = r1 - 1
        pipes_here = sorted(by_container[container["id"]], key=lambda p: (p["contact"][1], p["contact"][0]))
        for pipe in pipes_here:
            color = source_color[pipe["source"]]
            for _ in range(pipe["bends"]):
                assert current > r0, "pipe layers exceed container capacity"
                out[current, c0 + 1 : c1] = color
                current -= 1
    return out


def _draw_container(g: np.ndarray, r0: int, c0: int, inner_h: int, inner_w: int) -> tuple[int, int, int, int]:
    r1 = r0 + inner_h + 1
    c1 = c0 + inner_w + 1
    g[r0, c0 : c1 + 1] = MAROON
    g[r1, c0 : c1 + 1] = MAROON
    g[r0 : r1 + 1, c0] = MAROON
    g[r0 : r1 + 1, c1] = MAROON
    return r0, c0, r1, c1


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        return [(r0, c) for c in range(lo, hi + 1)]
    if c0 == c1:
        lo, hi = sorted((r0, r1))
        return [(r, c0) for r in range(lo, hi + 1)]
    raise ValueError("segment must be orthogonal")


def _path_cells(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells = []
    for a, b in zip(points, points[1:]):
        seg = _segment_cells(a, b)
        if cells:
            seg = seg[1:]
        cells.extend(seg)
    return cells


def _points_for_bends(start: tuple[int, int], end: tuple[int, int], bends: int, x1: int, x2: int, mid_r: int) -> list[tuple[int, int]]:
    sr, sc = start
    er, ec = end
    if bends == 1:
        return [start, (sr, ec), end]
    if bends == 2:
        return [start, (sr, x1), (er, x1), end]
    if bends == 3:
        return [start, (sr, x1), (mid_r, x1), (mid_r, ec), end]
    return [start, (sr, x1), (mid_r, x1), (mid_r, x2), (er, x2), end]


def _can_draw(g: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    cell_set = set(cells)
    h, w = g.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w) or int(g[r, c]) != BLACK:
            return False
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            q = (r + dr, c + dc)
            if q not in cell_set and 0 <= q[0] < h and 0 <= q[1] < w and int(g[q]) == GREY:
                return False
    return True


def _draw_pipe(g: np.ndarray, start: tuple[int, int], end: tuple[int, int], bends: int) -> bool:
    for _ in range(80):
        x1 = random.randint(6, max(7, min(start[1], end[1]) + 7))
        x2 = random.randint(x1 + 2, max(x1 + 3, end[1] - 2))
        mid_r = random.choice([min(start[0], end[0]) + 1, max(start[0], end[0]) - 1])
        points = _points_for_bends(start, end, bends, x1, x2, mid_r)
        cells = _path_cells(points)
        if _can_draw(g, cells):
            for r, c in cells:
                g[r, c] = GREY
            return True
    return False


def generate_input() -> np.ndarray:
    for _ in range(300):
        height = random.randint(29, 34)
        width = random.randint(33, 39)
        g = np.full((height, width), BLACK, dtype=int)
        c0 = width - random.randint(11, 13)

        containers = [
            _draw_container(g, 6, c0, random.randint(4, 6), random.randint(5, 7)),
            _draw_container(g, 19, c0 - random.randint(0, 2), random.randint(5, 7), random.randint(5, 7)),
        ]
        colors = random.sample(SOURCE_COLORS, 4)
        source_rows = [[2, 4], [15, 17]]
        ok = True
        for ci, bbox in enumerate(containers):
            r0, c0i, r1, c1 = bbox
            inner_h = r1 - r0 - 1
            pipe_count = random.randint(1, 2)
            bends = [random.randint(1, 3) for _ in range(pipe_count)]
            while sum(bends) > inner_h:
                bends = [max(1, b - 1) for b in bends]
            contact_cols = sorted(random.sample(range(c0i + 2, c1 - 1), pipe_count))
            for j in range(pipe_count):
                sr = source_rows[ci][j]
                source_top = sr - 1
                color = colors[ci * 2 + j]
                if np.any(g[source_top : source_top + 2, 1:3] != BLACK):
                    ok = False
                    break
                g[source_top : source_top + 2, 1:3] = color
                start = (sr, 3)
                end = (r0 - 1, contact_cols[j])
                if not _draw_pipe(g, start, end, bends[j]):
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        result = main(g)
        assert isinstance(result, np.ndarray)
        return g
    raise RuntimeError("failed to generate source-pipe-container input")
