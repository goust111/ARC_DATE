from __future__ import annotations

import random

import numpy as np


_BG = 8
_DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_DIR8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
)
_GEN_COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 9]
_GEN_PATH_RETRIES = 400
_GEN_ATTEMPTS = 300
_GEN_COMP_CACHE: dict[int, list[list[int]]] = {}


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in _DIR4:
        rr, cc = int(r + dr), int(c + dc)
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _is_king_adjacent(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1]))) == 1


def _unique_topological_order(node_count: int, edges: set[tuple[int, int]]) -> list[int] | None:
    indeg = [0] * node_count
    out = [[] for _ in range(node_count)]
    for src, dst in edges:
        indeg[int(dst)] += 1
        out[int(src)].append(int(dst))

    queue = [i for i, deg in enumerate(indeg) if deg == 0]
    order: list[int] = []
    while queue:
        if len(queue) != 1:
            return None
        cur = int(queue.pop(0))
        order.append(cur)
        for nxt in out[cur]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)

    return order if len(order) == node_count else None


def _trace_simple_path(path_cells: set[tuple[int, int]], start: tuple[int, int]) -> list[tuple[int, int]]:
    ordered = [start]
    prev = None
    cur = start
    while True:
        nxts = [p for p in ((cur[0] + dr, cur[1] + dc) for dr, dc in _DIR4) if p in path_cells and p != prev]
        if not nxts:
            break
        assert len(nxts) == 1
        prev, cur = cur, nxts[0]
        ordered.append(cur)
    return ordered


def _component(start: tuple[int, int], pts: set[tuple[int, int]]) -> set[tuple[int, int]]:
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for rr, cc in ((r + dr, c + dc) for dr, dc in _DIR4):
            if (rr, cc) in pts and (rr, cc) not in seen:
                seen.add((rr, cc))
                stack.append((rr, cc))
    return seen


def _fragment_data(grid: np.ndarray, bg: int):
    colors = [int(color) for color in np.unique(grid) if int(color) != bg]
    data = {}
    h, w = map(int, grid.shape)
    for color in colors:
        pts = [tuple(map(int, p)) for p in np.argwhere(grid == color)]
        pt_set = set(pts)
        assert pt_set
        assert len(_component(pts[0], pt_set)) == len(pt_set)

        degrees = {
            pt: sum((pt[0] + dr, pt[1] + dc) in pt_set for dr, dc in _DIR4)
            for pt in pt_set
        }
        ends = [pt for pt, deg in degrees.items() if deg == 1]
        assert len(ends) == 2
        interior = [pt for pt in pts if pt not in set(ends)]

        data[color] = {
            "pts": pts,
            "set": pt_set,
            "ends": ends,
            "interior": interior,
            "interior_set": set(interior),
        }
    return colors, data


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the layer order from endpoint-to-interior 8-neighbor contacts,
    then repaint the whole path from bottom layer to top layer along the
    global path order.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    h, w = map(int, g.shape)

    colors, frag = _fragment_data(g, bg)
    color_index = {color: idx for idx, color in enumerate(colors)}

    edges: set[tuple[int, int]] = set()
    for lower in colors:
        for upper in colors:
            if lower == upper or not frag[upper]["interior"]:
                continue
            if any(
                any((er + dr, ec + dc) in frag[upper]["interior_set"] for dr, dc in _DIR8)
                for er, ec in frag[lower]["ends"]
            ):
                edges.add((color_index[lower], color_index[upper]))

    order_idx = _unique_topological_order(len(colors), edges)
    assert order_idx is not None
    bottom_to_top = [colors[idx] for idx in order_idx]
    bottom_color = bottom_to_top[0]

    path_cells = {tuple(map(int, p)) for p in np.argwhere(g != bg)}
    global_deg = {
        pt: sum((pt[0] + dr, pt[1] + dc) in path_cells for dr, dc in _DIR4)
        for pt in path_cells
    }
    endpoints = [pt for pt, deg in global_deg.items() if deg == 1]
    assert len(endpoints) == 2

    start = next((pt for pt in endpoints if int(g[pt]) == bottom_color), None)
    assert start is not None
    ordered_path = _trace_simple_path(path_cells, start)
    assert len(ordered_path) == len(path_cells)

    out = np.full((h, w), bg, dtype=int)
    pos = 0
    for color in bottom_to_top:
        count = len(frag[color]["pts"])
        for cell in ordered_path[pos : pos + count]:
            out[cell] = int(color)
        pos += count
    assert pos == len(ordered_path)
    return out


def _gen_compositions(total: int, parts: int, min_part: int = 2) -> list[list[int]]:
    key = (total, parts, min_part)
    if key in _GEN_COMP_CACHE:
        return _GEN_COMP_CACHE[key]

    out: list[list[int]] = []

    def rec(rem: int, kk: int, prefix: list[int]) -> None:
        if kk == 1:
            if rem >= min_part:
                out.append(prefix + [rem])
            return
        hi = rem - min_part * (kk - 1)
        for part in range(min_part, hi + 1):
            rec(rem - part, kk - 1, prefix + [part])

    rec(total, parts, [])
    _GEN_COMP_CACHE[key] = out
    return out


def _gen_random_path(height: int, width: int, length: int) -> list[tuple[int, int]] | None:
    for _ in range(_GEN_PATH_RETRIES):
        start = (random.randrange(height), random.randrange(width))
        path = [start]
        occupied = {start}
        stack: list[tuple[int, list[tuple[int, int]]]] = []

        while True:
            if len(path) >= length:
                return path

            cur = path[-1]
            prev = path[-2] if len(path) >= 2 else None
            candidates = []
            for dr, dc in _DIR4:
                nr, nc = int(cur[0] + dr), int(cur[1] + dc)
                if not (0 <= nr < height and 0 <= nc < width) or (nr, nc) in occupied:
                    continue

                bad = False
                near8 = 0
                for rr, cc in occupied:
                    if abs(rr - nr) + abs(cc - nc) == 1 and (rr, cc) != cur:
                        bad = True
                        break
                    if max(abs(rr - nr), abs(cc - nc)) == 1:
                        near8 += 1
                if bad:
                    continue

                turn = 0
                if prev is not None:
                    turn = int((dr, dc) != (cur[0] - prev[0], cur[1] - prev[1]))
                candidates.append(((near8, turn, random.random()), (nr, nc)))

            candidates.sort(reverse=True)
            if candidates:
                choices = [pos for _score, pos in candidates]
                stack.append((len(path), choices[1:]))
                nxt = choices[0]
                path.append(nxt)
                occupied.add(nxt)
                continue

            while stack and not stack[-1][1]:
                depth, _ = stack.pop()
                while len(path) > depth:
                    occupied.remove(path.pop())
            if not stack:
                break

            depth, alts = stack[-1]
            nxt = alts.pop(0)
            while len(path) > depth:
                occupied.remove(path.pop())
            path.append(nxt)
            occupied.add(nxt)

    return None


def _gen_split_path(path: list[tuple[int, int]], lengths: list[int]) -> list[list[tuple[int, int]]]:
    segments = []
    start = 0
    for length in lengths:
        segments.append(path[start : start + length])
        start += length
    return segments


def _gen_relation_edges(segments: list[list[tuple[int, int]]]) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for i, seg_a in enumerate(segments):
        ends = (seg_a[0], seg_a[-1])
        for j, seg_b in enumerate(segments):
            if i == j or len(seg_b) < 3:
                continue
            interior = set(seg_b[1:-1])
            if any(any((er + dr, ec + dc) in interior for dr, dc in _DIR8) for er, ec in ends):
                edges.add((i, j))
    return edges


def _gen_expected(
    shape: tuple[int, int],
    path: list[tuple[int, int]],
    segments: list[list[tuple[int, int]]],
    order: list[int],
    colors: list[int],
) -> np.ndarray:
    out = np.full(shape, _BG, dtype=int)
    pos = 0
    for seg_idx in order:
        length = len(segments[seg_idx])
        for cell in path[pos : pos + length]:
            out[cell] = int(colors[seg_idx])
        pos += length
    return out


def _apply_symmetry(grid: np.ndarray, flip_lr: bool, rot_k: int) -> np.ndarray:
    out = np.array(grid, copy=True)
    if flip_lr:
        out = np.fliplr(out)
    if rot_k:
        out = np.rot90(out, rot_k)
    return np.array(out, copy=True)


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        height = random.randint(7, 10)
        width = random.randint(7, 10)
        length = random.randint(15, 22)
        path = _gen_random_path(height, width, length)
        if path is None:
            continue

        candidates = _gen_compositions(length, parts=3, min_part=2)
        random.shuffle(candidates)

        for lengths in candidates:
            segments = _gen_split_path(path, lengths)
            edges = _gen_relation_edges(segments)
            order = _unique_topological_order(3, edges)
            if order is None or order[0] != 0 or order == [0, 1, 2]:
                continue

            colors = random.sample(_GEN_COLOR_POOL, 3)
            inp = np.full((height, width), _BG, dtype=int)
            for color, seg in zip(colors, segments):
                for r, c in seg:
                    inp[r, c] = int(color)

            expected = _gen_expected((height, width), path, segments, order, colors)
            flip_lr = random.random() < 0.5
            rot_k = random.randrange(4)
            inp = _apply_symmetry(inp, flip_lr, rot_k)
            expected = _apply_symmetry(expected, flip_lr, rot_k)

            if not np.array_equal(main(inp), expected):
                continue
            return inp

    raise RuntimeError("failed to generate valid input")
