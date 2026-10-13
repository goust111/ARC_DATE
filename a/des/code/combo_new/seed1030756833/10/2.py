from __future__ import annotations

from collections import deque
import itertools
import random

import numpy as np


GREY = 5
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GENERATION_ATTEMPTS = 600


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        comp: list[tuple[int, int]] = []
        q: deque[tuple[int, int]] = deque([(sr, sc)])
        seen[sr, sc] = True
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        out.append(comp)
    return out


def _segment_from_component(comp: list[tuple[int, int]]) -> dict:
    rows = sorted({r for r, _c in comp})
    cols = sorted({c for _r, c in comp})
    assert len(rows) == 1 or len(cols) == 1, "visible fragments must be straight"

    cells = set(comp)
    if len(rows) == 1:
        r = rows[0]
        assert cols == list(range(cols[0], cols[-1] + 1)), "horizontal fragment must be solid"
        endpoints = ((r, cols[0]), (r, cols[-1]))
        direction = (0, 1)
    else:
        c = cols[0]
        assert rows == list(range(rows[0], rows[-1] + 1)), "vertical fragment must be solid"
        endpoints = ((rows[0], c), (rows[-1], c))
        direction = (1, 0)
    assert endpoints[0] != endpoints[1], "fragments must have length at least two"
    return {"cells": cells, "endpoints": endpoints, "direction": direction}


def _line_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    ar, ac = a
    br, bc = b
    assert ar == br or ac == bc
    if ar == br:
        step = 1 if bc >= ac else -1
        return [(ar, c) for c in range(ac, bc + step, step)]
    step = 1 if br >= ar else -1
    return [(r, ac) for r in range(ar, br + step, step)]


def _simple_path_order(cells: set[tuple[int, int]]) -> list[tuple[int, int]] | None:
    if not cells:
        return None

    degrees: dict[tuple[int, int], int] = {}
    for r, c in cells:
        degrees[(r, c)] = sum((nr, nc) in cells for nr, nc in _neighbors4(r, c))

    ends = [p for p, degree in degrees.items() if degree == 1]
    if len(cells) == 1:
        return [next(iter(cells))]
    if len(ends) != 2:
        return None
    if any(degree not in (1, 2) for degree in degrees.values()):
        return None

    start = min(ends)
    order = [start]
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        nxts = [p for p in _neighbors4(*cur) if p in cells and p != prev]
        if not nxts:
            break
        if len(nxts) > 1:
            return None
        prev, cur = cur, nxts[0]
        order.append(cur)
        if cur in ends:
            break

    if len(order) != len(cells):
        return None
    return order


def _candidate_connections(
    g: np.ndarray,
    bg: int,
    segments: list[dict],
) -> list[dict]:
    candidates: list[dict] = []
    endpoints = []
    for seg_i, seg in enumerate(segments):
        for end_i, point in enumerate(seg["endpoints"]):
            endpoints.append((seg_i, end_i, point))

    for a, b in itertools.combinations(endpoints, 2):
        seg_a, end_a, pa = a
        seg_b, end_b, pb = b
        if seg_a == seg_b:
            continue
        if pa[0] != pb[0] and pa[1] != pb[1]:
            continue
        cells = _line_cells(pa, pb)
        if len(cells) < 3:
            continue
        internal = cells[1:-1]
        if any(int(g[r, c]) != bg for r, c in internal):
            continue
        candidates.append(
            {
                "segs": (seg_a, seg_b),
                "ends": ((seg_a, end_a), (seg_b, end_b)),
                "cells": cells,
                "gap": len(cells) - 2,
            }
        )
    return candidates


def _connected_fragment_tree(n: int, combo: tuple[dict, ...]) -> bool:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for item in combo:
        a, b = item["segs"]
        union(int(a), int(b))
    return len({find(i) for i in range(n)}) == 1


def _complete_path_for_color(g: np.ndarray, bg: int, color: int) -> list[tuple[int, int]]:
    comps = _components(g == color)
    assert 2 <= len(comps) <= 4, "each color should appear as two to four fragments"
    segments = [_segment_from_component(comp) for comp in comps]
    base_cells = set().union(*(seg["cells"] for seg in segments))

    candidates = _candidate_connections(g, bg, segments)
    solutions: list[tuple[int, tuple[tuple[int, int], ...], list[tuple[int, int]]]] = []
    for combo in itertools.combinations(candidates, len(segments) - 1):
        used_ends = [end for item in combo for end in item["ends"]]
        if len(set(used_ends)) != len(used_ends):
            continue
        if not _connected_fragment_tree(len(segments), combo):
            continue
        cells = set(base_cells)
        for item in combo:
            cells.update(item["cells"])
        order = _simple_path_order(cells)
        if order is None:
            continue
        signature = tuple(sorted(cells))
        gap_sum = sum(int(item["gap"]) for item in combo)
        solutions.append((gap_sum, signature, order))

    assert solutions, f"no valid endpoint-aligned completion for color {color}"
    solutions.sort(key=lambda item: (item[0], item[1]))
    best_signature = solutions[0][1]
    tied = [item for item in solutions if item[0] == solutions[0][0]]
    assert all(item[1] == best_signature for item in tied), "ambiguous equal-length path completion"
    return solutions[0][2]


def _turns_with_spikes(
    order: list[tuple[int, int]],
    g: np.ndarray,
) -> list[tuple[tuple[int, int], tuple[int, int], int]]:
    step_dirs = [
        (order[i + 1][0] - order[i][0], order[i + 1][1] - order[i][1])
        for i in range(len(order) - 1)
    ]
    turns: list[tuple[tuple[int, int], tuple[int, int], int]] = []
    h, w = g.shape

    for i in range(1, len(order) - 1):
        prev_dir = step_dirs[i - 1]
        next_dir = step_dirs[i]
        if prev_dir == next_dir:
            continue
        assert prev_dir[0] * next_dir[0] + prev_dir[1] * next_dir[1] == 0, "turns must be right angles"
        turn = order[i]

        markers = []
        for dr in (-1, 1):
            for dc in (-1, 1):
                rr, cc = turn[0] + dr, turn[1] + dc
                if 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == GREY:
                    markers.append((dr, dc))
        assert len(markers) == 1, "each turn must have one adjacent grey diagonal marker"
        spike_dir = markers[0]

        left_len = 0
        j = i
        while j > 0 and step_dirs[j - 1] == prev_dir:
            left_len += 1
            j -= 1

        right_len = 0
        j = i
        while j < len(step_dirs) and step_dirs[j] == next_dir:
            right_len += 1
            j += 1

        turns.append((turn, spike_dir, min(left_len, right_len)))
    return turns


def main(grid: np.ndarray) -> np.ndarray:
    """Complete broken same-color polylines and grow grey-marked outside spikes at their turns.

    Input structure:
    - The background is the dominant color.
    - Each non-background, non-grey color appears as two to four one-pixel-wide
      horizontal or vertical line fragments.
    - Consecutive same-color fragments have endpoint pairs aligned in a common
      row or column, with background gaps between them.
    - Grey single-pixel markers sit diagonally outside the turns of the completed
      polylines.

    Rule:
    - For each line color, connect aligned fragment endpoints through background
      gaps, choosing the unique simple non-branching polyline completion.
    - After all paths are completed, find every right-angle turn.
    - The adjacent grey diagonal marker gives the outward spike direction at that
      turn. Paint a one-pixel-wide same-color spike in that direction, with length
      equal to the shorter of the two straight arms meeting at the turn.
    - Grey marker pixels are erased as markers and the output contains only the
      completed colored paths and their colored spikes on the original background.

    Key insight:
    - The grey pixels are not noise: endpoint alignment first reconstructs the
      hidden polylines, then each grey diagonal encodes where the corner's short
      outside spike must grow.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _background_color(g)

    colors = [int(c) for c in np.unique(g) if int(c) not in (bg, GREY)]
    assert colors, "expected at least one line color"

    out = g.copy()
    out[g == GREY] = bg

    path_infos: list[tuple[int, list[tuple[int, int]], list[tuple[tuple[int, int], tuple[int, int], int]]]] = []
    for color in colors:
        order = _complete_path_for_color(g, bg, color)
        turns = _turns_with_spikes(order, g)
        path_infos.append((color, order, turns))
        for r, c in order:
            assert int(out[r, c]) in (bg, color), "completed paths must not cross other colors"
            out[r, c] = color

    h, w = g.shape
    for color, _order, turns in path_infos:
        for (r, c), (dr, dc), length in turns:
            for step in range(1, length + 1):
                rr, cc = r + dr * step, c + dc * step
                assert 0 <= rr < h and 0 <= cc < w, "spike must stay inside the grid"
                assert int(out[rr, cc]) in (bg, color), "spikes must not collide with other colors"
                out[rr, cc] = color

    return out


def _add_padding(cells: set[tuple[int, int]], h: int, w: int, pad: int = 1) -> set[tuple[int, int]]:
    padded: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(max(0, r - pad), min(h, r + pad + 1)):
            for cc in range(max(0, c - pad), min(w, c + pad + 1)):
                padded.add((rr, cc))
    return padded


def _left_right_dirs(direction: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    dr, dc = direction
    return ((-dc, dr), (dc, -dr))


def _sample_piece_dirs(piece_count: int) -> list[tuple[int, int]]:
    dirs = [random.choice(DIR4)]
    changed = False
    for _ in range(1, piece_count):
        prev = dirs[-1]
        options = [prev, *_left_right_dirs(prev)]
        if random.random() < 0.35:
            nxt = prev
        else:
            nxt = random.choice(options[1:])
        changed = changed or nxt != prev
        dirs.append(nxt)
    if not changed:
        idx = random.randrange(1, piece_count)
        dirs[idx] = random.choice(_left_right_dirs(dirs[idx - 1]))
    return dirs


def _build_candidate_path(
    h: int,
    w: int,
    n_fragments: int,
) -> dict | None:
    piece_count = 2 * n_fragments - 1
    piece_dirs = _sample_piece_dirs(piece_count)
    fragment_lengths = [random.randint(2, 5) for _ in range(n_fragments)]
    gap_steps = [random.randint(2, 4) for _ in range(n_fragments - 1)]

    start = (random.randint(2, h - 3), random.randint(2, w - 3))
    path = [start]
    fragment_ranges: list[tuple[int, int]] = []
    pos = start
    piece_i = 0

    for frag_i in range(n_fragments):
        start_idx = len(path) - 1
        dr, dc = piece_dirs[piece_i]
        for _ in range(fragment_lengths[frag_i] - 1):
            pos = (pos[0] + dr, pos[1] + dc)
            path.append(pos)
        fragment_ranges.append((start_idx, len(path) - 1))
        piece_i += 1

        if frag_i < n_fragments - 1:
            dr, dc = piece_dirs[piece_i]
            for _ in range(gap_steps[frag_i]):
                pos = (pos[0] + dr, pos[1] + dc)
                path.append(pos)
            piece_i += 1

    if any(r < 1 or r >= h - 1 or c < 1 or c >= w - 1 for r, c in path):
        return None
    if len(set(path)) != len(path):
        return None
    if _simple_path_order(set(path)) is None:
        return None

    step_dirs = [(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]) for i in range(len(path) - 1)]
    turns = []
    for i in range(1, len(path) - 1):
        prev_dir = step_dirs[i - 1]
        next_dir = step_dirs[i]
        if prev_dir == next_dir:
            continue
        turn = path[i]
        spike_dir = (prev_dir[0] - next_dir[0], prev_dir[1] - next_dir[1])
        if abs(spike_dir[0]) != 1 or abs(spike_dir[1]) != 1:
            return None

        left_len = 0
        j = i
        while j > 0 and step_dirs[j - 1] == prev_dir:
            left_len += 1
            j -= 1
        right_len = 0
        j = i
        while j < len(step_dirs) and step_dirs[j] == next_dir:
            right_len += 1
            j += 1
        length = min(left_len, right_len)
        if length < 1:
            return None
        spike_cells = [(turn[0] + spike_dir[0] * k, turn[1] + spike_dir[1] * k) for k in range(1, length + 1)]
        if any(r < 0 or r >= h or c < 0 or c >= w for r, c in spike_cells):
            return None
        turns.append((turn, spike_dir, length, spike_cells))

    if not turns:
        return None

    path_set = set(path)
    spike_set = {p for _turn, _direction, _length, cells in turns for p in cells}
    marker_set = {cells[0] for _turn, _direction, _length, cells in turns}
    if path_set & spike_set:
        return None
    if len(marker_set) != len(turns):
        return None

    fragments: list[list[tuple[int, int]]] = []
    for start_i, end_i in fragment_ranges:
        frag = path[start_i : end_i + 1]
        if len(frag) < 2:
            return None
        fragments.append(frag)

    return {
        "path": path,
        "fragments": fragments,
        "turns": turns,
        "occupied": path_set | spike_set | marker_set,
    }


def _try_render_scene() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(14, 30)
    w = random.randint(14, 30)
    bg = random.choice([c for c in range(10) if c != GREY])
    palette = [c for c in range(10) if c not in (bg, GREY)]
    random.shuffle(palette)
    n_paths = random.randint(1, min(3, len(palette)))
    colors = palette[:n_paths]

    inp = np.full((h, w), bg, dtype=int)
    expected = np.full((h, w), bg, dtype=int)
    reserved: set[tuple[int, int]] = set()
    specs: list[tuple[int, dict]] = []

    for color in colors:
        placed = False
        for _ in range(180):
            n_fragments = random.choice([2, 3, 4])
            spec = _build_candidate_path(h, w, n_fragments)
            if spec is None:
                continue
            occupied = set(spec["occupied"])
            if occupied & reserved:
                continue
            reserved |= _add_padding(occupied, h, w, pad=1)
            specs.append((color, spec))
            placed = True
            break
        if not placed:
            return None

    for color, spec in specs:
        for frag in spec["fragments"]:
            for r, c in frag:
                inp[r, c] = color
        for r, c in spec["path"]:
            expected[r, c] = color
        for _turn, _direction, _length, spike_cells in spec["turns"]:
            mr, mc = spike_cells[0]
            inp[mr, mc] = GREY
            for r, c in spike_cells:
                expected[r, c] = color

    if _background_color(inp) != bg:
        return None
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        rendered = _try_render_scene()
        if rendered is not None:
            return rendered[0]
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_ATTEMPTS} attempts")
