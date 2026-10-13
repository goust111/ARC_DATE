from __future__ import annotations

import itertools
import random

import numpy as np

BLACK = 0
GREY = 5
MARKER_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
PATTERNS = ["H", "V", "HV", "VH", "HVH", "VHV", "HVHV", "VHVH"]


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _connect_fragments(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int).copy()
    h, w = g.shape

    while True:
        path_mask = g != BLACK
        degrees = np.zeros_like(g, dtype=int)
        for r, c in np.argwhere(path_mask):
            rr = int(r)
            cc = int(c)
            degrees[rr, cc] = sum(1 for nr, nc in _neighbors4(rr, cc, h, w) if bool(path_mask[nr, nc]))

        to_fill: list[tuple[int, int]] = []
        for r, c in np.argwhere(g == BLACK):
            rr = int(r)
            cc = int(c)
            horizontal = (
                0 < cc < w - 1
                and bool(path_mask[rr, cc - 1])
                and bool(path_mask[rr, cc + 1])
                and int(degrees[rr, cc - 1]) == 1
                and int(degrees[rr, cc + 1]) == 1
            )
            vertical = (
                0 < rr < h - 1
                and bool(path_mask[rr - 1, cc])
                and bool(path_mask[rr + 1, cc])
                and int(degrees[rr - 1, cc]) == 1
                and int(degrees[rr + 1, cc]) == 1
            )
            if horizontal or vertical:
                to_fill.append((rr, cc))

        if not to_fill:
            return g

        for r, c in to_fill:
            g[r, c] = GREY


def _ordered_component_path(component: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cell_set = set(component)
    neighbors: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in cell_set:
        cur_neighbors: list[tuple[int, int]] = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (int(r + dr), int(c + dc))
            if nxt in cell_set:
                cur_neighbors.append(nxt)
        neighbors[(r, c)] = cur_neighbors

    endpoints = [cell for cell, neigh in neighbors.items() if len(neigh) == 1]
    assert len(endpoints) == 2, f"expected a simple path, got endpoints={endpoints}"

    start = min(endpoints)
    ordered = [start]
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        nxts = [nxt for nxt in neighbors[cur] if nxt != prev]
        if not nxts:
            break
        assert len(nxts) == 1, f"branch detected at {cur}"
        prev, cur = cur, nxts[0]
        ordered.append(cur)

    assert len(ordered) == len(component), "component traversal did not cover the full path"
    return ordered


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble broken gray paths and recolor only the outer path segments defined by two markers.

    Input structure:
    - The background is black.
    - One or more gray one-cell-wide path fragments appear as 4-connected polylines; fragment endpoints may face another endpoint across a single black gap.
    - Colored single-pixel markers lie on top of path cells, replacing gray at that location.
    - Each final full path contains exactly two markers.

    Rule:
    - Repeatedly connect any two fragment endpoints whose Manhattan distance is 2 along a row or column with exactly one black cell between them; fill that middle cell gray.
    - After all such joins, each connected object is one complete simple path with two colored markers on it.
    - For each full path, keep the segment between the two markers gray, including both marker cells themselves.
    - Recolor the path cells strictly outside the first marker to that marker's color, and recolor the path cells strictly outside the second marker to that marker's color.

    Key insight:
    The markers do not color the whole path. They only color the two outer arms after the hidden gap-bridging step reconstructs which fragments belong to the same full path.
    """
    g = np.asarray(grid, dtype=int)
    linked = _connect_fragments(g)

    out = np.where(linked != BLACK, GREY, BLACK)
    for comp in _components(linked != BLACK):
        path = _ordered_component_path(comp)
        marker_indices = [idx for idx, cell in enumerate(path) if int(linked[cell]) != GREY]
        assert len(marker_indices) == 2, f"expected exactly two markers, got {marker_indices}"

        left_idx, right_idx = marker_indices
        left_color = int(linked[path[left_idx]])
        right_color = int(linked[path[right_idx]])

        for idx in range(left_idx):
            out[path[idx]] = left_color
        for idx in range(right_idx + 1, len(path)):
            out[path[idx]] = right_color

    return out


def _make_polyline(pattern: str, step_lengths: list[int], hdir: int, vdir: int) -> list[tuple[int, int]]:
    pts = [(0, 0)]
    r = 0
    c = 0
    for axis, steps in zip(pattern, step_lengths):
        dr, dc = (0, hdir) if axis == "H" else (vdir, 0)
        for _ in range(steps):
            r += dr
            c += dc
            pts.append((int(r), int(c)))
    return pts


def _shift_path(path: list[tuple[int, int]], top: int, left: int) -> list[tuple[int, int]]:
    rs = [r for r, _ in path]
    cs = [c for _, c in path]
    rmin = min(rs)
    cmin = min(cs)
    return [(int(r - rmin + top), int(c - cmin + left)) for r, c in path]


def _path_bbox(path: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in path]
    cs = [c for _, c in path]
    return min(rs), min(cs), max(rs), max(cs)


def _straight_indices(path: list[tuple[int, int]]) -> list[int]:
    straight: list[int] = []
    for idx in range(1, len(path) - 1):
        dr1 = path[idx][0] - path[idx - 1][0]
        dc1 = path[idx][1] - path[idx - 1][1]
        dr2 = path[idx + 1][0] - path[idx][0]
        dc2 = path[idx + 1][1] - path[idx][1]
        if dr1 == dr2 and dc1 == dc2:
            straight.append(idx)
    return straight


def _valid_cut_sets(path: list[tuple[int, int]]) -> list[tuple[int, list[tuple[tuple[int, ...], list[int]]]]]:
    candidates = _straight_indices(path)
    path_len = len(path)
    valid: list[tuple[int, list[tuple[tuple[int, ...], list[int]]]]] = []
    for frag_count in (2, 3, 4):
        if len(candidates) < frag_count - 1:
            continue
        options: list[tuple[tuple[int, ...], list[int]]] = []
        for cuts in itertools.combinations(candidates, frag_count - 1):
            prev_cut = -1
            lengths: list[int] = []
            ok = True
            for cut in cuts:
                frag_len = int(cut - prev_cut - 1)
                lengths.append(frag_len)
                if not 3 <= frag_len <= 8:
                    ok = False
                    break
                prev_cut = cut
            if not ok:
                continue
            final_len = int(path_len - prev_cut - 1)
            lengths.append(final_len)
            if all(3 <= frag_len <= 8 for frag_len in lengths):
                options.append((cuts, lengths))
        if options:
            valid.append((frag_count, options))
    return valid


def _choose_marker_indices(path_len: int, gaps: set[int]) -> tuple[int, int]:
    available = [idx for idx in range(path_len) if idx not in gaps]
    kinds = ["internal", "left_end", "right_end"]
    random.shuffle(kinds)

    for kind in kinds:
        if kind == "internal":
            options = [(i, j) for i in available for j in available if 1 <= i < j <= path_len - 2 and j - i >= 2]
        elif kind == "left_end":
            options = [(0, j) for j in available if j >= 2]
        else:
            options = [(i, path_len - 1) for i in available if i <= path_len - 3]
        if options:
            a, b = random.choice(options)
            return int(a), int(b)

    fallback = [(i, j) for i in available for j in available if i < j]
    a, b = random.choice(fallback)
    return int(a), int(b)


def _too_close_to_existing(candidate: list[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    for r, c in candidate:
        if (r, c) in occupied:
            return True
        for rr, cc in occupied:
            if abs(r - rr) + abs(c - cc) <= 2:
                return True
    return False


def _generate_one_path(h: int, w: int, occupied: set[tuple[int, int]]) -> dict[str, object] | None:
    patterns = PATTERNS[:]
    random.shuffle(patterns)

    for pattern in patterns:
        for _ in range(80):
            hdir = random.choice([-1, 1])
            vdir = random.choice([-1, 1])
            step_lengths = [random.randint(2, 6) for _ in pattern]
            if not (7 <= sum(step_lengths) <= 22):
                continue

            local_path = _make_polyline(pattern, step_lengths, hdir, vdir)
            valid_cuts = _valid_cut_sets(local_path)
            if not valid_cuts:
                continue

            r0, c0, r1, c1 = _path_bbox(local_path)
            box_h = r1 - r0 + 1
            box_w = c1 - c0 + 1
            if box_h > h - 2 or box_w > w - 2:
                continue

            for _ in range(40):
                top = random.randint(1, h - box_h - 1)
                left = random.randint(1, w - box_w - 1)
                shifted = _shift_path(local_path, top, left)
                if _too_close_to_existing(shifted, occupied):
                    continue

                _frag_count, options = random.choice(valid_cuts)
                cuts, frag_lengths = random.choice(options)
                gaps = {int(cut) for cut in cuts}
                marker_a, marker_b = _choose_marker_indices(len(shifted), gaps)
                colors = [int(random.choice(MARKER_COLORS)), int(random.choice(MARKER_COLORS))]

                return {
                    "path": shifted,
                    "gaps": gaps,
                    "markers": (int(marker_a), int(marker_b)),
                    "colors": colors,
                    "frag_lengths": [int(v) for v in frag_lengths],
                }

    return None


def _render_input_output(h: int, w: int, specs: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((h, w), BLACK, dtype=int)
    out = np.full((h, w), BLACK, dtype=int)

    for spec in specs:
        path = spec["path"]
        gaps = spec["gaps"]
        marker_a, marker_b = spec["markers"]
        color_a, color_b = spec["colors"]

        assert isinstance(path, list)
        assert isinstance(gaps, set)
        assert isinstance(marker_a, int) and isinstance(marker_b, int)
        assert isinstance(color_a, int) and isinstance(color_b, int)

        for idx, cell in enumerate(path):
            r, c = cell
            if idx not in gaps:
                inp[r, c] = GREY
            out[r, c] = GREY

        ar, ac = path[marker_a]
        br, bc = path[marker_b]
        inp[ar, ac] = color_a
        inp[br, bc] = color_b

        for idx, cell in enumerate(path):
            r, c = cell
            if idx < marker_a:
                out[r, c] = color_a
            elif idx > marker_b:
                out[r, c] = color_b

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(16, 30)
    w = random.randint(16, 30)
    path_count = random.randint(1, 3)

    specs: list[dict[str, object]] = []
    occupied: set[tuple[int, int]] = set()
    for _ in range(path_count):
        spec = _generate_one_path(h, w, occupied)
        if spec is None:
            return None
        specs.append(spec)
        occupied.update(spec["path"])

    inp, expected = _render_input_output(h, w, specs)

    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after 400 attempts")
