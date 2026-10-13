import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
MAROON = 9
PATH_COLORS = (1, 2, 3, 4, 6, 7, 8)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
EDGES = ("left", "right", "top", "bottom")
MAX_GENERATION_ATTEMPTS = 400
BRANCH_RETRIES = 40

# Small irregular sprites. The endpoint anchors the top-left corner of these masks.
SPRITES = (
    np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[1, 0, 1], [1, 1, 0]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 0], [1, 1], [0, 1]], dtype=int),
    np.array([[1, 1], [0, 1], [1, 1]], dtype=int),
    np.array([[1, 1], [1, 0], [1, 1]], dtype=int),
)


def _shape_offsets(shape: np.ndarray) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r, c in np.argwhere(shape != 0)]


def _translate_offsets(offsets: list[tuple[int, int]], anchor: tuple[int, int]) -> set[tuple[int, int]]:
    ar, ac = anchor
    return {(int(ar + dr), int(ac + dc)) for dr, dc in offsets}


def _dilate(cells: set[tuple[int, int]], h: int, w: int, include_diagonal: bool) -> set[tuple[int, int]]:
    out = set(cells)
    extra_dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)] if include_diagonal else []
    for r, c in cells:
        for dr, dc in list(DIR4) + extra_dirs:
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < h and 0 <= cc < w:
                out.add((rr, cc))
    return out


def _border_cells(h: int, w: int) -> set[tuple[int, int]]:
    pts = set()
    for r in range(h):
        pts.add((r, 0))
        pts.add((r, w - 1))
    for c in range(w):
        pts.add((0, c))
        pts.add((h - 1, c))
    return pts


def _shortest_path(
    h: int,
    w: int,
    start: tuple[int, int],
    end: tuple[int, int],
    blocked: set[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    blocked = set(blocked)
    blocked.discard(start)
    blocked.discard(end)

    q: deque[tuple[int, int]] = deque([start])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}

    while q:
        cur = q.popleft()
        if cur == end:
            break
        nbrs = []
        for dr, dc in DIR4:
            nr = int(cur[0] + dr)
            nc = int(cur[1] + dc)
            nxt = (nr, nc)
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if nxt in blocked or nxt in parent:
                continue
            nbrs.append(nxt)
        rng.shuffle(nbrs)
        nbrs.sort(key=lambda p: abs(int(p[0] - end[0])) + abs(int(p[1] - end[1])))
        for nxt in nbrs:
            parent[nxt] = cur
            q.append(nxt)

    if end not in parent:
        return None

    path = []
    cur: tuple[int, int] | None = end
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def _trace_color_path(grid: np.ndarray, color: int, start: tuple[int, int]) -> list[tuple[int, int]] | None:
    h, w = grid.shape
    q: deque[tuple[int, int]] = deque([start])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    target: tuple[int, int] | None = None

    while q:
        r, c = q.popleft()
        if int(grid[r, c]) == MAROON:
            target = (int(r), int(c))
            break
        for dr, dc in DIR4:
            rr = int(r + dr)
            cc = int(c + dc)
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            if (rr, cc) in parent:
                continue
            val = int(grid[rr, cc])
            if val not in (int(color), GREY, MAROON):
                continue
            parent[(rr, cc)] = (int(r), int(c))
            q.append((rr, cc))

    if target is None:
        return None

    path = []
    cur: tuple[int, int] | None = target
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def _paint_shape(out: np.ndarray, color: int, endpoint: tuple[int, int], offsets: list[tuple[int, int]]) -> None:
    h, w = out.shape
    er, ec = endpoint
    for dr, dc in offsets:
        rr = int(er + dr)
        cc = int(ec + dc)
        if 0 <= rr < h and 0 <= cc < w:
            out[rr, cc] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide colored fragments along their same-colored paths to the maroon endpoints.

    Paths are single-pixel lines on a black background. Grey pixels are path intersections.
    Each path starts on the grid border and ends at one interior maroon marker. A small same-
    colored fragment sits near the start. The output erases all tracks, moves each fragment so
    its top-left corner aligns to its path's endpoint, then keeps the maroon markers visible.
    If moved fragments overlap, the shorter path is drawn on top.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    border_starts: dict[int, list[tuple[int, int]]] = {}
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            val = int(g[r, c])
            if val in (BG, GREY, MAROON):
                continue
            border_starts.setdefault(val, []).append((int(r), int(c)))

    records = []
    for color, starts in border_starts.items():
        best_path = None
        for start in starts:
            path = _trace_color_path(g, int(color), start)
            if path is None:
                continue
            if best_path is None or len(path) < len(best_path):
                best_path = path
        if best_path is None:
            continue

        path_cells = set(best_path[:-1])
        sprite_cells = [
            (int(r), int(c))
            for r, c in np.argwhere(g == int(color))
            if (int(r), int(c)) not in path_cells
        ]
        if not sprite_cells:
            continue

        r0 = min(r for r, _ in sprite_cells)
        c0 = min(c for _, c in sprite_cells)
        offsets = [(int(r - r0), int(c - c0)) for r, c in sprite_cells]
        records.append(
            {
                "color": int(color),
                "length": int(len(best_path) - 1),
                "endpoint": (int(best_path[-1][0]), int(best_path[-1][1])),
                "offsets": offsets,
            }
        )

    out = np.full_like(g, BG)
    for rec in records:
        er, ec = rec["endpoint"]
        out[er, ec] = MAROON

    # Longer paths first, shorter paths later and therefore on top.
    for rec in sorted(
        records,
        key=lambda item: (-int(item["length"]), int(item["endpoint"][0]), int(item["endpoint"][1]), int(item["color"])),
    ):
        _paint_shape(out, int(rec["color"]), rec["endpoint"], rec["offsets"])

    for rec in records:
        er, ec = rec["endpoint"]
        out[er, ec] = MAROON
    return out


def _output_from_records(h: int, w: int, records: list[dict]) -> np.ndarray:
    out = np.full((h, w), BG, dtype=int)
    for rec in records:
        er, ec = rec["endpoint"]
        out[er, ec] = MAROON
    for rec in sorted(records, key=lambda item: (-int(item["length"]), int(item["color"]))):
        _paint_shape(out, int(rec["color"]), rec["endpoint"], rec["shape_offsets"])
    for rec in records:
        er, ec = rec["endpoint"]
        out[er, ec] = MAROON
    return out


def _choose_endpoints(
    rng: random.Random,
    shapes: list[np.ndarray],
    h: int,
    w: int,
    force_overlap: bool,
) -> list[tuple[int, int]] | None:
    n = len(shapes)
    endpoints: list[tuple[int, int] | None] = [None] * n
    occupied: list[set[tuple[int, int]]] = []
    spaced: set[tuple[int, int]] = set()

    if force_overlap:
        offsets0 = _shape_offsets(shapes[0])
        offsets1 = _shape_offsets(shapes[1])
        deltas = list({(int(r0 - r1), int(c0 - c1)) for r0, c0 in offsets0 for r1, c1 in offsets1 if (r0, c0) != (r1, c1)})
        rng.shuffle(deltas)
        for _ in range(200):
            r0 = rng.randrange(2, h - shapes[0].shape[0] - 1)
            c0 = rng.randrange(2, w - shapes[0].shape[1] - 1)
            base = (int(r0), int(c0))
            cells0 = _translate_offsets(offsets0, base)
            for dr, dc in deltas:
                other = (int(base[0] + dr), int(base[1] + dc))
                if other == base:
                    continue
                if other[0] < 2 or other[1] < 2:
                    continue
                if other[0] + shapes[1].shape[0] > h - 1 or other[1] + shapes[1].shape[1] > w - 1:
                    continue
                cells1 = _translate_offsets(offsets1, other)
                if cells0 & cells1:
                    endpoints[0] = base
                    endpoints[1] = other
                    occupied.extend([cells0, cells1])
                    spaced |= _dilate(cells0 | cells1, h, w, include_diagonal=True)
                    break
            if endpoints[0] is not None:
                break
        if endpoints[0] is None:
            return None

    start_idx = 2 if force_overlap else 0
    for idx in range(start_idx, n):
        offsets = _shape_offsets(shapes[idx])
        placed = False
        for _ in range(200):
            anchor = (
                int(rng.randrange(2, h - shapes[idx].shape[0] - 1)),
                int(rng.randrange(2, w - shapes[idx].shape[1] - 1)),
            )
            cells = _translate_offsets(offsets, anchor)
            if any(cells & prev for prev in occupied):
                continue
            if cells & spaced:
                continue
            endpoints[idx] = anchor
            occupied.append(cells)
            spaced |= _dilate(cells, h, w, include_diagonal=True)
            placed = True
            break
        if not placed:
            return None

    return [pt for pt in endpoints if pt is not None]


def _fragment_candidates(edge: str, shape: np.ndarray, h: int, w: int) -> list[tuple[tuple[int, int], set[tuple[int, int]]]]:
    fh, fw = shape.shape
    offsets = _shape_offsets(shape)
    cands: list[tuple[tuple[int, int], set[tuple[int, int]]]] = []

    if edge == "left":
        for start_r in range(fh, h - 1):
            top = start_r - fh
            anchor = (int(top), 2)
            cands.append(((int(start_r), 0), _translate_offsets(offsets, anchor)))
        for start_r in range(1, h - fh - 1):
            top = start_r + 1
            anchor = (int(top), 2)
            cands.append(((int(start_r), 0), _translate_offsets(offsets, anchor)))
    elif edge == "right":
        left = w - fw - 2
        for start_r in range(fh, h - 1):
            top = start_r - fh
            anchor = (int(top), int(left))
            cands.append(((int(start_r), w - 1), _translate_offsets(offsets, anchor)))
        for start_r in range(1, h - fh - 1):
            top = start_r + 1
            anchor = (int(top), int(left))
            cands.append(((int(start_r), w - 1), _translate_offsets(offsets, anchor)))
    elif edge == "top":
        for start_c in range(fw, w - 1):
            left = start_c - fw
            anchor = (2, int(left))
            cands.append(((0, int(start_c)), _translate_offsets(offsets, anchor)))
        for start_c in range(1, w - fw - 1):
            left = start_c + 1
            anchor = (2, int(left))
            cands.append(((0, int(start_c)), _translate_offsets(offsets, anchor)))
    else:
        top = h - fh - 2
        for start_c in range(fw, w - 1):
            left = start_c - fw
            anchor = (int(top), int(left))
            cands.append(((h - 1, int(start_c)), _translate_offsets(offsets, anchor)))
        for start_c in range(1, w - fw - 1):
            left = start_c + 1
            anchor = (int(top), int(left))
            cands.append(((h - 1, int(start_c)), _translate_offsets(offsets, anchor)))

    return cands


def _choose_starts_and_fragments(
    rng: random.Random,
    edges: list[str],
    shapes: list[np.ndarray],
    endpoints: list[tuple[int, int]],
    h: int,
    w: int,
) -> list[dict] | None:
    visual_blocked = set(endpoints)
    chosen: list[dict] = []

    for edge, shape, endpoint in zip(edges, shapes, endpoints, strict=True):
        candidates = _fragment_candidates(edge, shape, h, w)
        rng.shuffle(candidates)
        picked = None
        for start, frag_cells in candidates:
            if frag_cells & visual_blocked:
                continue
            if _dilate(frag_cells, h, w, include_diagonal=True) & visual_blocked:
                continue
            picked = {"edge": edge, "start": start, "fragment_cells": frag_cells, "endpoint": endpoint}
            break
        if picked is None:
            return None
        chosen.append(picked)
        visual_blocked |= _dilate(picked["fragment_cells"], h, w, include_diagonal=True)

    return chosen


def _choose_cross(rng: random.Random, h: int, w: int, blocked: set[tuple[int, int]]) -> tuple[int, int] | None:
    cands = [
        (int(r), int(c))
        for r in range(3, h - 3)
        for c in range(3, w - 3)
        if (int(r), int(c)) not in blocked
    ]
    rng.shuffle(cands)
    cands.sort(key=lambda p: abs(int(p[0] - h // 2)) + abs(int(p[1] - w // 2)))
    return cands[0] if cands else None


def _gen_attempt(
    rng: random.Random,
    path_count: int,
    force_cross: bool,
    force_overlap: bool,
) -> np.ndarray | None:
    h = int(rng.randint(15, 24))
    w = int(rng.randint(15, 24))

    colors = list(rng.sample(PATH_COLORS, path_count))
    shapes = [np.array(rng.choice(SPRITES), dtype=int) for _ in range(path_count)]
    endpoints = _choose_endpoints(rng, shapes, h, w, force_overlap)
    if endpoints is None or len(endpoints) != path_count:
        return None

    if force_cross:
        edges = [rng.choice(("left", "right")), rng.choice(("top", "bottom"))]
        remaining = [edge for edge in EDGES if edge not in edges]
        rng.shuffle(remaining)
        edges.extend(remaining[: path_count - 2])
    else:
        edges = list(rng.sample(EDGES, path_count))
    placements = _choose_starts_and_fragments(rng, edges, shapes, endpoints, h, w)
    if placements is None:
        return None

    fragment_moats = set()
    for info in placements:
        fragment_moats |= _dilate(info["fragment_cells"], h, w, include_diagonal=False)

    endpoint_cells = set(endpoints)
    border = _border_cells(h, w)
    cross = None
    if force_cross:
        cross = _choose_cross(rng, h, w, fragment_moats | endpoint_cells)
        if cross is None:
            return None

    route_records = []
    used_line_cells: set[tuple[int, int]] = set()
    for idx, info in enumerate(placements):
        start = info["start"]
        end = info["endpoint"]
        blocked_common = fragment_moats | endpoint_cells | used_line_cells | (border - {start})
        blocked_common.discard(end)
        if cross is not None and idx < 2:
            blocked_a = set(blocked_common)
            blocked_a.discard(cross)
            seg1 = _shortest_path(h, w, start, cross, blocked_a, rng)
            if seg1 is None:
                return None
            blocked_b = set(blocked_common) | set(seg1[:-1])
            blocked_b.discard(cross)
            blocked_b.discard(end)
            seg2 = _shortest_path(h, w, cross, end, blocked_b, rng)
            if seg2 is None:
                return None
            path = seg1 + seg2[1:]
        else:
            path = _shortest_path(h, w, start, end, blocked_common, rng)
            if path is None:
                return None

        line_cells = set(path[:-1])
        if used_line_cells & line_cells:
            if cross is None:
                return None
            shared = used_line_cells & line_cells
            if shared != {cross}:
                return None

        route_records.append(
            {
                "color": int(colors[idx]),
                "shape_offsets": _shape_offsets(shapes[idx]),
                "path": path,
                "length": int(len(path) - 1),
                "endpoint": end,
                "fragment_cells": info["fragment_cells"],
            }
        )
        used_line_cells |= line_cells

    lengths = [int(rec["length"]) for rec in route_records]
    if len(lengths) != len(set(lengths)):
        return None

    grid = np.full((h, w), BG, dtype=int)
    for rec in route_records:
        for r, c in rec["path"][:-1]:
            val = int(grid[r, c])
            if val == BG:
                grid[r, c] = int(rec["color"])
            elif val == GREY:
                return None
            else:
                grid[r, c] = GREY

    for rec in route_records:
        er, ec = rec["endpoint"]
        grid[er, ec] = MAROON

    for rec in route_records:
        for r, c in rec["fragment_cells"]:
            if grid[r, c] != BG:
                return None
            grid[r, c] = int(rec["color"])

    expected = _output_from_records(h, w, route_records)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        path_count = int(rng.choice((2, 3)))
        force_cross = path_count >= 2 and rng.random() < 0.5
        force_overlap = path_count >= 2 and rng.random() < 0.5
        for _ in range(BRANCH_RETRIES):
            grid = _gen_attempt(rng, path_count, force_cross, force_overlap)
            if grid is not None:
                return grid
    raise RuntimeError("generate_input: failed to build a valid puzzle instance")
