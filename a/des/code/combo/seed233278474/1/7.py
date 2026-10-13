from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BLACK = 0
GREEN = 3
YELLOW = 4
GREY = 5
SEGMENT_COLORS = (1, 2, 6, 7, 8, 9)
RESERVED = {BLACK, GREEN, YELLOW, GREY}
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _touches_border(cells: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in cells)


def _segment_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    segments: list[dict] = []
    next_id = 0
    for color in sorted(int(v) for v in np.unique(g) if int(v) not in RESERVED):
        comps = _components_mask(g == color)
        for comp in comps:
            rows = sorted({r for r, _ in comp})
            cols = sorted({c for _, c in comp})
            if len(rows) == 1:
                orientation = "h"
                row = rows[0]
                c0, c1 = min(cols), max(cols)
                cells = [(row, c) for c in range(c0, c1 + 1)]
                endpoints = [(row, c0), (row, c1)]
            elif len(cols) == 1:
                orientation = "v"
                col = cols[0]
                r0, r1 = min(rows), max(rows)
                cells = [(r, col) for r in range(r0, r1 + 1)]
                endpoints = [(r0, col), (r1, col)]
            else:
                raise AssertionError("segment must be a straight axis-aligned line")
            if set(cells) != set(comp):
                raise AssertionError("segment component must be gap-free")
            segments.append(
                {
                    "id": next_id,
                    "color": color,
                    "orientation": orientation,
                    "cells": cells,
                    "cell_set": set(cells),
                    "endpoints": endpoints,
                }
            )
            next_id += 1
    if not segments:
        raise AssertionError("no segments found")
    return segments


def _endpoint_rays(grid: np.ndarray, segments: list[dict]) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    rays: list[dict] = []
    next_id = 0
    for seg in segments:
        for endpoint in seg["endpoints"]:
            r, c = endpoint
            greens = []
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) == GREEN:
                    greens.append((nr, nc))
            if len(greens) != 1:
                raise AssertionError("each endpoint must have exactly one adjacent green indicator")
            gr, gc = greens[0]
            dr, dc = gr - r, gc - c
            if seg["orientation"] == "h":
                if dc != 0 or dr == 0:
                    raise AssertionError("horizontal endpoints must point vertically")
                axis = "v"
            else:
                if dr != 0 or dc == 0:
                    raise AssertionError("vertical endpoints must point horizontally")
                axis = "h"
            rays.append(
                {
                    "id": next_id,
                    "segment_id": seg["id"],
                    "color": seg["color"],
                    "start": endpoint,
                    "dir": (dr, dc),
                    "axis": axis,
                }
            )
            next_id += 1
    if not rays:
        raise AssertionError("no rays found")
    return rays


def _nearest_body_block(
    start: tuple[int, int],
    direction: tuple[int, int],
    body_cells: set[tuple[int, int]],
    h: int,
    w: int,
) -> int:
    r, c = start
    dr, dc = direction
    step = 1
    while True:
        rr = r + dr * step
        cc = c + dc * step
        if not (0 <= rr < h and 0 <= cc < w):
            return 10**9
        if (rr, cc) in body_cells:
            return step
        step += 1


def _candidate_intersections(
    rays: list[dict],
    body_limits: dict[int, int],
    body_cells: set[tuple[int, int]],
) -> tuple[dict[int, list[dict]], dict[tuple[int, int], dict]]:
    by_ray: dict[int, list[dict]] = defaultdict(list)
    pair_data: dict[tuple[int, int], dict] = {}
    by_id = {ray["id"]: ray for ray in rays}
    for i in range(len(rays)):
        for j in range(i + 1, len(rays)):
            a = rays[i]
            b = rays[j]
            if a["axis"] == b["axis"]:
                continue
            if a["axis"] == "v":
                vray, hray = a, b
            else:
                vray, hray = b, a
            iv = (hray["start"][0], vray["start"][1])
            if iv in body_cells:
                continue
            vdr = vray["dir"][0]
            hdc = hray["dir"][1]
            dv = (iv[0] - vray["start"][0]) * vdr
            dh = (iv[1] - hray["start"][1]) * hdc
            if dv < 1 or dh < 1:
                continue
            if dv >= body_limits[vray["id"]] or dh >= body_limits[hray["id"]]:
                continue
            lo, hi = sorted((vray["id"], hray["id"]))
            pair_data[(lo, hi)] = {"pos": iv, lo: dv if lo == vray["id"] else dh, hi: dh if hi == hray["id"] else dv}
            by_ray[vray["id"]].append(
                {"partner": hray["id"], "pos": iv, "self_dist": dv, "partner_dist": dh}
            )
            by_ray[hray["id"]].append(
                {"partner": vray["id"], "pos": iv, "self_dist": dh, "partner_dist": dv}
            )
    for rid in by_ray:
        by_ray[rid].sort(key=lambda item: (item["self_dist"], item["partner_dist"], item["partner"]))
    for ray in rays:
        by_ray.setdefault(ray["id"], [])
    return by_ray, pair_data


def _resolve_pairs(
    rays: list[dict],
    candidates: dict[int, list[dict]],
) -> list[dict]:
    unresolved = {ray["id"] for ray in rays}
    resolved: list[dict] = []
    while True:
        pref: dict[int, dict] = {}
        for rid in unresolved:
            for cand in candidates[rid]:
                if cand["partner"] in unresolved:
                    pref[rid] = cand
                    break
        new_pairs: list[dict] = []
        used: set[int] = set()
        for rid, cand in pref.items():
            if rid in used or cand["partner"] in used:
                continue
            other = pref.get(cand["partner"])
            if other is None:
                continue
            if other["partner"] != rid or other["pos"] != cand["pos"]:
                continue
            new_pairs.append(
                {
                    "a": rid,
                    "b": cand["partner"],
                    "pos": cand["pos"],
                    "da": cand["self_dist"],
                    "db": other["self_dist"],
                }
            )
            used.add(rid)
            used.add(cand["partner"])
        if not new_pairs:
            break
        for pair in new_pairs:
            unresolved.discard(pair["a"])
            unresolved.discard(pair["b"])
        resolved.extend(new_pairs)
    return resolved


def _paint_extensions(
    out: np.ndarray,
    rays: list[dict],
    pairs: list[dict],
    body_limits: dict[int, int],
) -> None:
    by_id = {ray["id"]: ray for ray in rays}
    paired = set()
    for pair in pairs:
        paired.add(pair["a"])
        paired.add(pair["b"])
        for rid, dist in ((pair["a"], pair["da"]), (pair["b"], pair["db"])):
            ray = by_id[rid]
            r, c = ray["start"]
            dr, dc = ray["dir"]
            for step in range(1, dist):
                out[r + dr * step, c + dc * step] = ray["color"]
        pr, pc = pair["pos"]
        out[pr, pc] = GREY
    for ray in rays:
        if ray["id"] in paired:
            continue
        limit = body_limits[ray["id"]]
        if limit >= 10**9:
            raise AssertionError("unresolved ray without body hit")
        r, c = ray["start"]
        dr, dc = ray["dir"]
        for step in range(1, limit):
            out[r + dr * step, c + dc * step] = ray["color"]


def _fill_yellow_regions(out: np.ndarray, yellow_positions: list[tuple[int, int]]) -> np.ndarray:
    h, w = out.shape
    result = out.copy()
    if not yellow_positions:
        return result
    empty_comps = _components_mask(result == BLACK)
    comp_by_cell: dict[tuple[int, int], int] = {}
    for idx, comp in enumerate(empty_comps):
        for cell in comp:
            comp_by_cell[cell] = idx
    targets = {comp_by_cell[pos] for pos in yellow_positions if pos in comp_by_cell}
    for idx in targets:
        comp = empty_comps[idx]
        if _touches_border(comp, h, w):
            continue
        boundary_by_color: dict[int, set[tuple[int, int]]] = defaultdict(set)
        for r, c in comp:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w:
                    val = int(result[nr, nc])
                    if val not in RESERVED:
                        boundary_by_color[val].add((nr, nc))
        if not boundary_by_color:
            continue
        ranked = sorted(boundary_by_color.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        fill_color = ranked[0][0]
        for r, c in comp:
            result[r, c] = fill_color
    return result


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the loop indicated by endpoint-side green markers, then fill the
    enclosed black region(s) containing yellow markers with the dominant boundary color.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    if h == 0 or w == 0:
        return g.copy()

    segments = _segment_components(g)
    rays = _endpoint_rays(g, segments)
    body_cells = {cell for seg in segments for cell in seg["cell_set"]}
    body_limits = {
        ray["id"]: _nearest_body_block(ray["start"], ray["dir"], body_cells, h, w)
        for ray in rays
    }
    candidates, _pair_data = _candidate_intersections(rays, body_limits, body_cells)
    pairs = _resolve_pairs(rays, candidates)

    out = g.copy()
    out[out == GREEN] = BLACK
    out[out == YELLOW] = BLACK
    _paint_extensions(out, rays, pairs, body_limits)

    yellow_positions = [(int(r), int(c)) for r, c in np.argwhere(g == YELLOW)]
    return _fill_yellow_regions(out, yellow_positions)


GEN_ATTEMPTS = 400
OFFSET_ATTEMPTS = 80
PAD_RANGE = (1, 5)
YELLOW_COUNT_RANGE = (1, 3)


def _point_in_polygon(point: tuple[float, float], vertices: list[tuple[int, int]]) -> bool:
    py, px = point
    inside = False
    for i in range(len(vertices)):
        y1, x1 = vertices[i]
        y2, x2 = vertices[(i + 1) % len(vertices)]
        if (y1 > py) == (y2 > py):
            continue
        xin = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
        if px < xin:
            inside = not inside
    return inside


def _edge_inward_dir(
    p0: tuple[int, int],
    p1: tuple[int, int],
    vertices: list[tuple[int, int]],
) -> tuple[int, int]:
    if p0[0] == p1[0]:
        r = p0[0]
        midc = (p0[1] + p1[1]) / 2.0
        down = _point_in_polygon((r + 0.5, midc), vertices)
        up = _point_in_polygon((r - 0.5, midc), vertices)
        if down == up:
            raise AssertionError("cannot determine inward side of horizontal edge")
        return (1, 0) if down else (-1, 0)
    c = p0[1]
    midr = (p0[0] + p1[0]) / 2.0
    right = _point_in_polygon((midr, c + 0.5), vertices)
    left = _point_in_polygon((midr, c - 0.5), vertices)
    if right == left:
        raise AssertionError("cannot determine inward side of vertical edge")
    return (0, 1) if right else (0, -1)


def _canonical_rectangle(rng: random.Random) -> list[tuple[int, int]]:
    width = rng.randint(3, 7)
    height = rng.randint(3, 7)
    return [(0, 0), (0, width), (height, width), (height, 0)]


def _canonical_notch(rng: random.Random) -> list[tuple[int, int]]:
    for _ in range(100):
        a = rng.randint(2, 5)
        b = rng.randint(2, 5)
        u = rng.randint(2, 5)
        v = rng.randint(2, 5)
        if a + b <= 7 and u + v <= 7:
            return [(0, 0), (0, a + b), (u, a + b), (u, a), (u + v, a), (u + v, 0)]
    raise RuntimeError("failed to sample notch polygon")


def _build_local_pair(
    vertices: list[tuple[int, int]],
    colors: tuple[int, ...],
    offsets: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    edges: list[dict] = []
    all_points: list[tuple[int, int]] = list(vertices)
    for i, (p0, p1) in enumerate(zip(vertices, vertices[1:] + vertices[:1])):
        if p0[0] != p1[0] and p0[1] != p1[1]:
            raise AssertionError("polygon edge must be axis-aligned")
        inward = _edge_inward_dir(p0, p1, vertices)
        outward = (-inward[0], -inward[1])
        off = offsets[i]
        color = colors[i]
        if p0[0] == p1[0]:
            body_row = p0[0] + outward[0] * off
            c0, c1 = sorted((p0[1], p1[1]))
            body = [(body_row, c) for c in range(c0, c1 + 1)]
            endpoints = [(body_row, p0[1]), (body_row, p1[1])]
        else:
            body_col = p0[1] + outward[1] * off
            r0, r1 = sorted((p0[0], p1[0]))
            body = [(r, body_col) for r in range(r0, r1 + 1)]
            endpoints = [(p0[0], body_col), (p1[0], body_col)]
        if len(body) < 3 or len(body) > 8:
            raise AssertionError("body length out of range")
        all_points.extend(body)
        edges.append(
            {
                "color": color,
                "body": body,
                "endpoints": endpoints,
                "vertices": (p0, p1),
                "inward": inward,
                "offset": off,
            }
        )
    min_r = min(r for r, _ in all_points)
    min_c = min(c for _, c in all_points)
    shift = (-min_r, -min_c)
    shifted_vertices = {(r + shift[0], c + shift[1]) for r, c in vertices}
    max_r = max(r + shift[0] for r, _ in all_points)
    max_c = max(c + shift[1] for _, c in all_points)
    inp = np.full((max_r + 1, max_c + 1), BLACK, dtype=int)
    out = np.full_like(inp, BLACK)
    for edge in edges:
        edge["body"] = [(r + shift[0], c + shift[1]) for r, c in edge["body"]]
        edge["endpoints"] = [(r + shift[0], c + shift[1]) for r, c in edge["endpoints"]]
        edge["vertices"] = tuple((r + shift[0], c + shift[1]) for r, c in edge["vertices"])
        for cell in edge["body"]:
            if int(out[cell]) != BLACK:
                raise AssertionError("bodies overlap")
            out[cell] = edge["color"]
            inp[cell] = edge["color"]
    for edge in edges:
        for endpoint, vertex in zip(edge["endpoints"], edge["vertices"]):
            dr, dc = edge["inward"]
            for step in range(1, edge["offset"]):
                rr = endpoint[0] + dr * step
                cc = endpoint[1] + dc * step
                if (rr, cc) in shifted_vertices:
                    raise AssertionError("extension entered vertex too early")
                if int(out[rr, cc]) != BLACK:
                    raise AssertionError("extensions overlap")
                out[rr, cc] = edge["color"]
            gr = endpoint[0] + dr
            gc = endpoint[1] + dc
            if not (0 <= gr <= max_r and 0 <= gc <= max_c):
                raise AssertionError("green out of bounds")
            if int(inp[gr, gc]) not in (BLACK, GREEN):
                raise AssertionError("green overlaps non-green content")
            inp[gr, gc] = GREEN
    for vr, vc in shifted_vertices:
        if int(out[vr, vc]) != BLACK:
            raise AssertionError("vertex occupied before grey placement")
        out[vr, vc] = GREY
    for arr in (inp, out):
        for r in range(arr.shape[0]):
            for c in range(arr.shape[1]):
                val = int(arr[r, c])
                if val in RESERVED:
                    continue
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < arr.shape[0] and 0 <= nc < arr.shape[1]:
                        nval = int(arr[nr, nc])
                        if nval in RESERVED or nval == val:
                            continue
                        raise AssertionError("different segment colors must not 4-touch directly")
    return inp, out


def _random_local_pair(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    vertices = _canonical_rectangle(rng)
    n_edges = len(vertices)
    colors = tuple(rng.sample(SEGMENT_COLORS, n_edges))
    for _ in range(OFFSET_ATTEMPTS):
        offsets = tuple(rng.randint(1, 3) for _ in range(n_edges))
        try:
            return _build_local_pair(vertices, colors, offsets)
        except AssertionError:
            continue
    raise RuntimeError("failed to build local pair")


def _apply_symmetry(grid: np.ndarray, code: int) -> np.ndarray:
    if code == 0:
        return grid.copy()
    if code == 1:
        return np.rot90(grid, 1).copy()
    if code == 2:
        return np.rot90(grid, 2).copy()
    if code == 3:
        return np.rot90(grid, 3).copy()
    if code == 4:
        return np.fliplr(grid).copy()
    if code == 5:
        return np.flipud(grid).copy()
    if code == 6:
        return grid.T.copy()
    if code == 7:
        return np.flipud(grid.T).copy()
    raise ValueError("bad symmetry code")


def _dominant_boundary_color(out: np.ndarray, comp: list[tuple[int, int]]) -> int | None:
    boundary: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for r, c in comp:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1]:
                val = int(out[nr, nc])
                if val not in RESERVED:
                    boundary[val].add((nr, nc))
    if not boundary:
        return None
    ranked = sorted(boundary.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    if len(ranked) > 1 and len(ranked[0][1]) == len(ranked[1][1]):
        return None
    return ranked[0][0]


def _embed_with_yellows(
    local_in: np.ndarray,
    local_out: np.ndarray,
    rng: random.Random,
) -> np.ndarray:
    pad_top = rng.randint(*PAD_RANGE)
    pad_bottom = rng.randint(*PAD_RANGE)
    pad_left = rng.randint(*PAD_RANGE)
    pad_right = rng.randint(*PAD_RANGE)
    h, w = local_in.shape
    H = h + pad_top + pad_bottom
    W = w + pad_left + pad_right
    inp = np.full((H, W), BLACK, dtype=int)
    out = np.full((H, W), BLACK, dtype=int)
    inp[pad_top : pad_top + h, pad_left : pad_left + w] = local_in
    out[pad_top : pad_top + h, pad_left : pad_left + w] = local_out

    empty_comps = _components_mask(out == BLACK)
    enclosed = [comp for comp in empty_comps if not _touches_border(comp, H, W)]
    if not enclosed:
        raise AssertionError("no enclosed region available for yellows")
    target = rng.choice(enclosed)
    fill_color = _dominant_boundary_color(out, target)
    if fill_color is None:
        raise AssertionError("boundary color majority must be unique")

    count = min(len(target), rng.randint(*YELLOW_COUNT_RANGE))
    yellow_cells = rng.sample(target, count)
    for r, c in yellow_cells:
        if int(inp[r, c]) != BLACK:
            raise AssertionError("yellow must occupy empty region cells")
        inp[r, c] = YELLOW
    expected = out.copy()
    for r, c in target:
        expected[r, c] = fill_color
    solved = main(inp)
    if not np.array_equal(solved, expected):
        raise AssertionError("generator produced an invalid puzzle")
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        try:
            local_in, local_out = _random_local_pair(rng)
            sym = rng.randrange(8)
            local_in = _apply_symmetry(local_in, sym)
            local_out = _apply_symmetry(local_out, sym)
            return _embed_with_yellows(local_in, local_out, rng)
        except (AssertionError, RuntimeError):
            continue
    raise RuntimeError("failed to generate a valid input")
