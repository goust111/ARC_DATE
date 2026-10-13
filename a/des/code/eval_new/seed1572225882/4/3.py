import random
from collections import defaultdict

import numpy as np


BLACK = 0
GREY = 5
ORANGE = 7
AZURE = 8
PALETTE = [1, 2, 3, 4, 6, 9]


def _neighbors(cell):
    r, c = cell
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _turn_name(a, b):
    dr, dc = a
    nr, nc = b
    z = dc * nr - dr * nc
    return "L" if z < 0 else "R"


def _extract_path(grid):
    pts = [tuple(p) for p in np.argwhere(grid == GREY)]
    if not pts:
        return [], [], []

    cells = set(pts)
    graph = {}
    for p in cells:
        graph[p] = [q for q in _neighbors(p) if q in cells]

    ends = sorted([p for p, ns in graph.items() if len(ns) == 1])
    if len(ends) != 2:
        return [], [], []

    start = ends[0]
    prev = None
    cur = start
    order = []
    while True:
        order.append(cur)
        nxts = [n for n in graph[cur] if n != prev]
        if not nxts:
            break
        prev, cur = cur, nxts[0]

    if len(order) != len(cells) or len(order) < 2:
        return [], [], []

    dirs = []
    for (r1, c1), (r2, c2) in zip(order, order[1:]):
        dirs.append((r2 - r1, c2 - c1))

    turns = []
    segments = []
    current = [order[0]]
    last_dir = dirs[0]
    current.append(order[1])
    for i in range(1, len(dirs)):
        d = dirs[i]
        if d == last_dir:
            current.append(order[i + 1])
        else:
            turns.append(_turn_name(last_dir, d))
            segments.append(current)
            current = [order[i + 1]]
            last_dir = d
    segments.append(current)
    return turns, segments, order


def _parse_candidates(grid, needed_len):
    candidates = []
    h, w = grid.shape
    for r in range(h):
        row = grid[r]
        for c in range(w - 1):
            marker = int(row[c])
            if marker not in (AZURE, ORANGE):
                continue
            if row[c + 1] == BLACK:
                continue
            end = c + 1
            while end < w and row[end] != BLACK:
                end += 1
            colors = [int(x) for x in row[c + 1:end]]
            if len(colors) != needed_len or len(colors) < 2:
                continue
            if len(set(colors)) != 2:
                continue
            if any(colors[i] != colors[i % 2] for i in range(len(colors))):
                continue
            occupied = set(range(c, end))
            other = {i for i, v in enumerate(row) if v != BLACK} - occupied
            if other:
                continue
            candidates.append((r, c, marker, colors))
    return candidates


def _matches(marker, colors, turns):
    first, second = colors[0], colors[1]
    if marker == AZURE:
        table = {"L": first, "R": second}
    else:
        table = {"L": second, "R": first}
    return [table[t] for t in turns] == colors[:-1]


def main(grid: np.ndarray) -> np.ndarray:
    """Color the grey path with the uniquely matching encoded candidate ribbon.

    Input structure:
    The background is black. A single-pixel-wide grey four-connected polyline is
    the path; its deterministic start is the endpoint that is nearest the
    top-left corner by row, then column. Lower rows contain candidate horizontal
    ribbons: an azure or orange marker immediately to the left of an alternating
    two-color strip, with black rows separating strips.

    Rule:
    Walk the grey path from the top-left endpoint and record each right-angle
    turn as left or right. For an azure marker, left turns encode the strip's
    first color and right turns encode its second color; for an orange marker
    the mapping is reversed. The matching strip is the unique one whose first
    len(turns) cells equal the decoded turn colors, while the final strip cell
    colors the terminal path segment. Clear all candidate rows to black and
    color each path segment with the corresponding strip cell.

    Key insight:
    The correct colors are not chosen by color identity alone: the path's turn
    topology, the alternating strip phase, and the marker-controlled left/right
    mapping must agree.
    """
    out = np.array(grid, copy=True)
    turns, segments, _ = _extract_path(grid)
    if not turns:
        return out

    candidates = _parse_candidates(grid, len(segments))
    matches = [cand for cand in candidates if _matches(cand[2], cand[3], turns)]
    if not matches:
        return out

    chosen = matches[0]
    colors = chosen[3]

    for r, _, _, _ in candidates:
        out[r, :] = BLACK

    for color, segment in zip(colors, segments):
        for r, c in segment:
            out[r, c] = color
    return out


def _rotate(direction, turn):
    dr, dc = direction
    if turn == "L":
        return -dc, dr
    return dc, -dr


def _make_polyline(turn_count):
    for _ in range(200):
        first_turn = random.choice(["L", "R"])
        turns = [first_turn if i % 2 == 0 else ("R" if first_turn == "L" else "L")
                 for i in range(turn_count)]
        direction = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
        dirs = [direction]
        for t in turns:
            dirs.append(_rotate(dirs[-1], t))
        lengths = [random.randint(2, 5) for _ in dirs]

        cells = [(0, 0)]
        r = c = 0
        ok = True
        seen = {(0, 0)}
        for d, length in zip(dirs, lengths):
            dr, dc = d
            for _ in range(length):
                r += dr
                c += dc
                if (r, c) in seen:
                    ok = False
                    break
                cells.append((r, c))
                seen.add((r, c))
            if not ok:
                break
        if not ok:
            continue

        min_r = min(r for r, _ in cells)
        min_c = min(c for _, c in cells)
        cells = [(r - min_r, c - min_c) for r, c in cells]
        height = max(r for r, _ in cells) + 1
        width = max(c for _, c in cells) + 1
        if height <= 18 and width <= 24:
            return cells
    return None


def _candidate_row(width, seg_count, marker, colors):
    row = np.zeros(width, dtype=int)
    start = random.randint(0, width - seg_count - 1)
    row[start] = marker
    row[start + 1:start + 1 + seg_count] = colors
    return row


def generate_input() -> np.ndarray:
    while True:
        turn_count = random.randint(3, 7)
        cells = _make_polyline(turn_count)
        if cells is None:
            continue

        path_h = max(r for r, _ in cells) + 1
        path_w = max(c for _, c in cells) + 1
        seg_count = turn_count + 1
        cand_count = random.randint(3, 5)
        lower_h = cand_count * 2 - 1

        min_h = path_h + lower_h + 2
        min_w = max(path_w, seg_count + 1) + 2
        if min_h > 30 or min_w > 30:
            continue

        h = random.randint(min_h, 30)
        w = random.randint(min_w, 30)
        grid = np.zeros((h, w), dtype=int)

        cand_start = h - lower_h
        top_space = cand_start - 1
        if top_space < path_h:
            continue

        off_r = random.randint(0, top_space - path_h)
        off_c = random.randint(0, w - path_w)
        for r, c in cells:
            grid[off_r + r, off_c + c] = GREY

        turns, segments, _ = _extract_path(grid)
        if len(turns) != turn_count or len(segments) != seg_count:
            continue

        c1, c2 = random.sample(PALETTE, 2)
        correct_colors = [c1 if i % 2 == 0 else c2 for i in range(seg_count)]
        correct_marker = AZURE if turns[0] == "L" else ORANGE
        wrong_marker = ORANGE if correct_marker == AZURE else AZURE

        rows = []
        correct_index = random.randrange(cand_count)
        used_pairs = set()
        for i in range(cand_count):
            if i == correct_index:
                rows.append(_candidate_row(w, seg_count, correct_marker, correct_colors))
                used_pairs.add(tuple(correct_colors[:2]))
                continue
            for _ in range(50):
                a, b = random.sample(PALETTE, 2)
                if (a, b) not in used_pairs:
                    used_pairs.add((a, b))
                    break
            decoy_colors = [a if j % 2 == 0 else b for j in range(seg_count)]
            rows.append(_candidate_row(w, seg_count, wrong_marker, decoy_colors))

        rr = cand_start
        for row in rows:
            grid[rr] = row
            rr += 2

        out = main(grid)
        if out.shape == grid.shape and max(grid.shape) <= 30 and max(out.shape) <= 30:
            return grid
