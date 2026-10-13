import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ROUTER_COLORS = (6, 7, 8, 9, BLUE)
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _neighbors(r, c, h, w):
    for dr, dc in DIRS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _left(d):
    dr, dc = d
    return -dc, dr


def _right(d):
    dr, dc = d
    return dc, -dr


def _table_bbox(g):
    if g[0, 0] != GREY:
        raise AssertionError("route table must touch the upper-left corner")
    w = g.shape[1]
    c1 = 0
    while c1 + 1 < w and g[0, c1 + 1] == GREY:
        c1 += 1
    assert c1 >= 6, "route table must have a grey top border"
    r1 = None
    for r in range(1, g.shape[0]):
        if g[r, 0] == GREY and g[r, c1] == GREY and np.all(g[r, : c1 + 1] == GREY):
            r1 = r
            break
    assert r1 is not None and r1 >= 4, "route table must have a grey bottom border"
    return 0, 0, r1, c1


def _parse_table(g):
    r0, c0, r1, c1 = _table_bbox(g)
    mapping = {}
    row = r0 + 1
    while row + 2 < r1:
        color = int(g[row + 1, c0 + 1])
        patch = g[row : row + 3, c0 + 3 : c0 + 6]
        red = {tuple(p) for p in np.argwhere(patch == RED)}
        if color != BLACK:
            assert color not in {BLACK, RED, GREEN, YELLOW, GREY}
            left_pat = {(1, 1), (0, 1), (1, 0)}
            right_pat = {(1, 1), (0, 1), (1, 2)}
            if red == left_pat:
                mapping[color] = "left"
            elif red == right_pat:
                mapping[color] = "right"
            else:
                raise AssertionError("route row must contain a left/right red L")
        row += 3
    assert mapping, "expected at least one route-table entry"
    return (r0, c0, r1, c1), mapping


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        pts = []
        while q:
            r, c = q.popleft()
            pts.append((r, c))
            for nr, nc in _neighbors(r, c, h, w):
                if mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(pts)
    return comps


def _find_routers(g, table_bbox):
    r0, c0, r1, c1 = table_bbox
    routers = []
    cell_to_router = {}
    for color in sorted(set(map(int, np.unique(g))) - {BLACK, GREY, RED, GREEN, YELLOW}):
        mask = g == color
        mask[r0 : r1 + 1, c0 : c1 + 1] = False
        for pts in _components(mask):
            rs = [p[0] for p in pts]
            cs = [p[1] for p in pts]
            if len(pts) != 4:
                continue
            if max(rs) - min(rs) != 1 or max(cs) - min(cs) != 1:
                continue
            rid = len(routers)
            router = {
                "color": color,
                "cells": set(pts),
                "r0": min(rs),
                "r1": max(rs),
                "c0": min(cs),
                "c1": max(cs),
            }
            routers.append(router)
            for p in pts:
                cell_to_router[p] = rid
    return routers, cell_to_router


def _side_dir(router, cell):
    r, c = cell
    if r < router["r0"]:
        return (-1, 0)
    if r > router["r1"]:
        return (1, 0)
    if c < router["c0"]:
        return (0, -1)
    if c > router["c1"]:
        return (0, 1)
    raise AssertionError("cell is not outside router")


def _adjacent_router_ids(cell, routers):
    r, c = cell
    out = []
    for rid, router in enumerate(routers):
        for nr, nc in _neighbors(r, c, 10**9, 10**9):
            if (nr, nc) in router["cells"]:
                out.append(rid)
                break
    return out


def _router_exits(g, router):
    h, w = g.shape
    exits = {}
    for cell in router["cells"]:
        for nr, nc in _neighbors(cell[0], cell[1], h, w):
            if g[nr, nc] == GREY:
                exits[_side_dir(router, (nr, nc))] = (nr, nc)
    return exits


def _source_cell(g):
    cells = [tuple(map(int, p)) for p in np.argwhere(g == GREEN)]
    edge_sources = []
    h, w = g.shape
    for r, c in cells:
        if r in {0, h - 1} or c in {0, w - 1}:
            edge_sources.append((r, c))
    assert len(edge_sources) == 1, "expected one green signal source on the grid edge"
    return edge_sources[0]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    table_bbox, route = _parse_table(g)
    routers, _cell_to_router = _find_routers(g, table_bbox)
    assert routers, "expected at least one 2x2 router"

    out = g.copy()
    r0, c0, r1, c1 = table_bbox
    out[r0 : r1 + 1, c0 : c1 + 1] = BLACK

    h, w = g.shape
    source = _source_cell(g)
    starts = [(nr, nc) for nr, nc in _neighbors(*source, h, w) if g[nr, nc] == GREY]
    assert len(starts) == 1, "source must touch exactly one pipe"

    curr = starts[0]
    prev = source
    last_router = None
    for _ in range(h * w):
        if g[curr] == YELLOW:
            out[curr] = GREEN
            return out

        if g[curr] == GREY:
            for nr, nc in _neighbors(*curr, h, w):
                if g[nr, nc] == YELLOW:
                    out[nr, nc] = GREEN
                    return out

            adjacent = []
            for rid, router in enumerate(routers):
                if rid == last_router:
                    continue
                if any((nr, nc) in router["cells"] for nr, nc in _neighbors(*curr, h, w)):
                    adjacent.append(rid)
            if adjacent:
                assert len(adjacent) == 1, "pipe cell must enter only one router"
                rid = adjacent[0]
                router = routers[rid]
                incoming_move = tuple(-x for x in _side_dir(router, curr))
                choice = route[int(router["color"])]
                wanted = _left(incoming_move) if choice == "left" else _right(incoming_move)
                exits = _router_exits(g, router)
                assert wanted in exits, "router must expose the selected branch"
                prev = curr
                curr = exits[wanted]
                last_router = rid
                continue

            next_cells = [
                (nr, nc)
                for nr, nc in _neighbors(*curr, h, w)
                if (nr, nc) != prev and g[nr, nc] == GREY
            ]
            assert len(next_cells) == 1, "grey pipe path must be non-branching between routers"
            prev, curr = curr, next_cells[0]
            last_router = None
            continue

        raise AssertionError("signal left the pipe network")
    raise AssertionError("routing did not reach a terminal")


def _draw_table(g, router_color, choice):
    g[0:5, 0:7] = BLACK
    g[0, 0:7] = GREY
    g[4, 0:7] = GREY
    g[0:5, 0] = GREY
    g[0:5, 6] = GREY
    g[2, 1] = router_color
    red = [(2, 4), (1, 4)]
    red.append((2, 3) if choice == "left" else (2, 5))
    for r, c in red:
        g[r, c] = RED


def _draw_line(g, a, b, color=GREY):
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        g[r0, lo : hi + 1] = color
    else:
        lo, hi = sorted((r0, r1))
        g[lo : hi + 1, c0] = color


def _gen_attempt():
    h = random.randint(18, 26)
    w = random.randint(18, 28)
    g = np.zeros((h, w), dtype=int)
    rr = random.randint(8, h - 7)
    rc = random.randint(7, w - 8)
    router_color = random.choice(ROUTER_COLORS)
    choice = random.choice(("left", "right"))

    _draw_table(g, router_color, choice)
    g[rr : rr + 2, rc : rc + 2] = router_color

    source = (rr + 1, 0)
    g[source] = GREEN
    _draw_line(g, (rr + 1, 1), (rr + 1, rc - 1))

    left_terminal = (rr - 4, rc)
    right_terminal = (rr + 5, rc)
    _draw_line(g, (rr - 1, rc), (left_terminal[0] + 1, rc))
    _draw_line(g, (rr + 2, rc), (right_terminal[0] - 1, rc))
    g[left_terminal] = YELLOW
    g[right_terminal] = YELLOW

    # Add a short harmless wiggle to one branch in some samples while preserving a tree.
    if random.random() < 0.5 and rc + 3 < w:
        tr = rr - 3 if choice == "right" else rr + 4
        _draw_line(g, (tr, rc), (tr, rc + 2))
        g[tr, rc] = GREY
        g[tr, rc + 2] = YELLOW
        if tr < rr:
            g[left_terminal] = BLACK
        else:
            g[right_terminal] = BLACK

    return g


def generate_input() -> np.ndarray:
    for _ in range(300):
        g = _gen_attempt()
        try:
            out = main(g)
        except AssertionError:
            continue
        if not np.array_equal(g, out):
            return g
    raise RuntimeError("failed to generate valid routing input")
