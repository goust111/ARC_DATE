from __future__ import annotations

from collections import Counter
import itertools
import random

import numpy as np


Coord = tuple[int, int]
Step = tuple[int, int]

DIRS4: tuple[Step, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
PERP_DIRS: dict[Step, tuple[Step, Step]] = {
    (-1, 0): ((0, -1), (0, 1)),
    (1, 0): ((0, -1), (0, 1)),
    (0, -1): ((-1, 0), (1, 0)),
    (0, 1): ((-1, 0), (1, 0)),
}
ALL_COLORS = tuple(range(10))


def _add(a: Coord, b: Step) -> Coord:
    return int(a[0] + b[0]), int(a[1] + b[1])


def _sub(a: Coord, b: Coord) -> Step:
    return int(a[0] - b[0]), int(a[1] - b[1])


def _in_bounds(r: int, c: int, h: int, w: int) -> bool:
    return 0 <= int(r) < int(h) and 0 <= int(c) < int(w)


def _neighbors4(rc: Coord, h: int, w: int) -> list[Coord]:
    r, c = map(int, rc)
    out: list[Coord] = []
    for dr, dc in DIRS4:
        rr = int(r + dr)
        cc = int(c + dc)
        if _in_bounds(rr, cc, h, w):
            out.append((rr, cc))
    return out


def _most_common_color(g: np.ndarray) -> int:
    vals, counts = np.unique(g, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[Coord] = []
            while stack:
                cur = stack.pop()
                comp.append(cur)
                for nxt in _neighbors4(cur, h, w):
                    if bool(mask[nxt]) and not bool(seen[nxt]):
                        seen[nxt] = True
                        stack.append(nxt)
            comps.append(comp)
    return comps


def _path_degrees(mask: np.ndarray) -> dict[Coord, int]:
    h, w = mask.shape
    deg: dict[Coord, int] = {}
    for r, c in map(tuple, np.argwhere(mask)):
        rc = (int(r), int(c))
        deg[rc] = sum(1 for rr, cc in _neighbors4(rc, h, w) if bool(mask[rr, cc]))
    return deg


def _straight_runs(path: list[Coord]) -> list[dict]:
    if len(path) < 2:
        return []
    steps = [_sub(path[i + 1], path[i]) for i in range(len(path) - 1)]
    runs: list[dict] = []
    start = 0
    cur = steps[0]
    for i in range(1, len(steps)):
        if steps[i] != cur:
            runs.append(
                {
                    "start": int(start),
                    "end": int(i),
                    "dir": cur,
                    "step_count": int(i - start),
                    "cell_count": int(i - start + 1),
                }
            )
            start = i
            cur = steps[i]
    runs.append(
        {
            "start": int(start),
            "end": int(len(steps)),
            "dir": cur,
            "step_count": int(len(steps) - start),
            "cell_count": int(len(steps) - start + 1),
        }
    )
    return runs


def _trace_path_order(mask: np.ndarray) -> list[Coord]:
    deg = _path_degrees(mask)
    endpoints = sorted(rc for rc, d in deg.items() if int(d) == 1)
    assert len(endpoints) == 2, "final path must have exactly two endpoints"

    order: list[Coord] = []
    prev: Coord | None = None
    cur = endpoints[0]
    while True:
        order.append(cur)
        nxts = [nxt for nxt in _neighbors4(cur, mask.shape[0], mask.shape[1]) if bool(mask[nxt]) and nxt != prev]
        if not nxts:
            break
        assert len(nxts) == 1, "final path must not branch"
        prev, cur = cur, nxts[0]

    assert len(order) == int(mask.sum()), "traversal must visit the whole path"
    return order


def _detect_gates(g: np.ndarray, bg: int) -> list[dict]:
    h, w = g.shape
    raw: list[dict] = []

    for r in range(h):
        for c in range(w):
            center = int(g[r, c])
            if center == int(bg):
                continue
            opts: list[tuple[str, int, tuple[Coord, Coord]]] = []
            if 0 < c < w - 1:
                side = int(g[r, c - 1])
                if side != int(bg) and side == int(g[r, c + 1]) and side != center:
                    opts.append(("h", side, ((int(r), int(c - 1)), (int(r), int(c + 1)))))
            if 0 < r < h - 1:
                side = int(g[r - 1, c])
                if side != int(bg) and side == int(g[r + 1, c]) and side != center:
                    opts.append(("v", side, ((int(r - 1), int(c)), (int(r + 1), int(c)))))
            if len(opts) == 1:
                orient, side_color, arms = opts[0]
                raw.append(
                    {
                        "center": (int(r), int(c)),
                        "orientation": orient,
                        "side_color": int(side_color),
                        "color": int(center),
                        "arm_cells": tuple(arms),
                    }
                )

    if not raw:
        return []

    grey = Counter(int(item["side_color"]) for item in raw).most_common(1)[0][0]
    by_center: dict[Coord, list[dict]] = {}
    for item in raw:
        if int(item["side_color"]) != int(grey):
            continue
        by_center.setdefault(item["center"], []).append(item)

    gates: list[dict] = []
    used_cells: set[Coord] = set()
    for center in sorted(by_center):
        items = by_center[center]
        if len(items) != 1:
            continue
        item = items[0]
        cells = {item["center"], *item["arm_cells"]}
        if cells & used_cells:
            continue
        used_cells |= cells
        gates.append(item)

    return gates


def _extract_components(g: np.ndarray, bg: int, gates: list[dict]) -> tuple[np.ndarray, list[dict], dict[Coord, dict]]:
    h, w = g.shape
    gate_cells = {cell for gate in gates for cell in (gate["center"], *gate["arm_cells"])}
    path_mask = g != int(bg)
    for r, c in gate_cells:
        path_mask[int(r), int(c)] = False

    comps = _components(path_mask)
    assert comps, "expected at least one path segment"

    comp_info: list[dict] = []
    endpoint_map: dict[Coord, dict] = {}
    deg = _path_degrees(path_mask)

    for comp_id, coords in enumerate(comps):
        colors = {int(g[r, c]) for r, c in coords}
        assert len(colors) == 1, "each input segment must be monochrome"
        color = next(iter(colors))

        endpoints = [rc for rc in coords if int(deg[rc]) == 1]
        assert len(endpoints) == 2, "each segment must be an open simple path"

        data = {
            "id": int(comp_id),
            "coords": list(coords),
            "coord_set": set(coords),
            "color": int(color),
            "size": int(len(coords)),
            "endpoints": [],
        }

        for endpoint in endpoints:
            nbrs = [nxt for nxt in _neighbors4(endpoint, h, w) if nxt in data["coord_set"]]
            assert len(nbrs) == 1
            out_dir = _sub(endpoint, nbrs[0])
            info = {"coord": endpoint, "out_dir": out_dir, "comp_id": int(comp_id)}
            data["endpoints"].append(info)
            endpoint_map[endpoint] = info

        comp_info.append(data)

    return path_mask, comp_info, endpoint_map


def _candidate_edges(g: np.ndarray, bg: int, path_mask: np.ndarray, endpoint_map: dict[Coord, dict]) -> list[dict]:
    h, w = g.shape
    seen: dict[tuple[Coord, Coord], dict] = {}

    for endpoint, info in endpoint_map.items():
        dr, dc = map(int, info["out_dir"])
        rr = int(endpoint[0] + dr)
        cc = int(endpoint[1] + dc)
        cells: list[Coord] = []
        while _in_bounds(rr, cc, h, w):
            if int(g[rr, cc]) != int(bg):
                hit = (int(rr), int(cc))
                target = endpoint_map.get(hit)
                if target is not None and int(target["comp_id"]) != int(info["comp_id"]) and tuple(target["out_dir"]) == (-dr, -dc):
                    key = tuple(sorted((endpoint, hit)))
                    seen[key] = {
                        "a": endpoint,
                        "b": hit,
                        "cells": list(cells),
                        "comp_a": int(info["comp_id"]),
                        "comp_b": int(target["comp_id"]),
                    }
                break
            cells.append((int(rr), int(cc)))
            rr += dr
            cc += dc

    return [seen[key] for key in sorted(seen)]


def _select_edges(comp_info: list[dict], edges: list[dict]) -> list[dict]:
    n = len(comp_info)
    if n == 1:
        return []
    assert len(edges) >= n - 1, "not enough candidate connections"

    valid: list[tuple[int, ...]] = []
    edge_ids = list(range(len(edges)))

    def _find(parent: list[int], x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return int(x)

    for combo in itertools.combinations(edge_ids, n - 1):
        endpoint_use: Counter[Coord] = Counter()
        comp_deg = [0] * n
        parent = list(range(n))
        ok = True

        for edge_idx in combo:
            edge = edges[edge_idx]
            a = int(edge["comp_a"])
            b = int(edge["comp_b"])
            endpoint_use[edge["a"]] += 1
            endpoint_use[edge["b"]] += 1
            if endpoint_use[edge["a"]] > 1 or endpoint_use[edge["b"]] > 1:
                ok = False
                break
            comp_deg[a] += 1
            comp_deg[b] += 1
            if comp_deg[a] > 2 or comp_deg[b] > 2:
                ok = False
                break
            ra = _find(parent, a)
            rb = _find(parent, b)
            if ra == rb:
                ok = False
                break
            parent[ra] = rb

        if not ok:
            continue

        roots = {_find(parent, i) for i in range(n)}
        if len(roots) != 1:
            continue
        valid.append(combo)

    assert len(valid) == 1, "endpoint pairing must be unique"
    return [edges[idx] for idx in valid[0]]


def _connection_color(comp_a: dict, comp_b: dict) -> int:
    if int(comp_a["size"]) < int(comp_b["size"]):
        return int(comp_a["color"])
    if int(comp_b["size"]) < int(comp_a["size"]):
        return int(comp_b["color"])
    sig_a = min(comp_a["coords"])
    sig_b = min(comp_b["coords"])
    return int(comp_a["color"] if sig_a <= sig_b else comp_b["color"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect scattered colored path segments by endpoint collinearity, then recolor the
    reconstructed path according to nearby colored gate frames.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _most_common_color(g)

    gates = _detect_gates(g, bg)
    assert gates, "expected at least one gate"

    path_mask, comp_info, endpoint_map = _extract_components(g, bg, gates)
    edges = _candidate_edges(g, bg, path_mask, endpoint_map)
    selected = _select_edges(comp_info, edges)

    out = g.copy()
    final_mask = path_mask.copy()
    comp_by_id = {int(comp["id"]): comp for comp in comp_info}

    for edge in selected:
        color = _connection_color(comp_by_id[int(edge["comp_a"])], comp_by_id[int(edge["comp_b"])])
        for r, c in edge["cells"]:
            final_mask[int(r), int(c)] = True
            out[int(r), int(c)] = int(color)

    order = _trace_path_order(final_mask)
    order_index = {rc: i for i, rc in enumerate(order)}

    gate_triggers: dict[int, int] = {}
    for gate in gates:
        r, c = map(int, gate["center"])
        if gate["orientation"] == "h":
            candidates = [(r - 1, c), (r + 1, c)]
        else:
            candidates = [(r, c - 1), (r, c + 1)]
        hits = [rc for rc in candidates if _in_bounds(rc[0], rc[1], h, w) and bool(final_mask[rc])]
        assert len(hits) == 1, "each gate must align to exactly one path pixel"
        idx = int(order_index[hits[0]])
        assert idx not in gate_triggers, "gate triggers must be distinct along the path"
        gate_triggers[idx] = int(gate["color"])

    current: int | None = None
    for idx, (r, c) in enumerate(order):
        if idx in gate_triggers:
            current = int(gate_triggers[idx])
        if current is not None:
            out[int(r), int(c)] = int(current)

    return out


# --------------- generator ---------------

GRID_H_RANGE = (18, 27)
GRID_W_RANGE = (18, 27)

PATH_MIN_RUNS = 4
PATH_MAX_RUNS = 8
PATH_MIN_LEN = 22
PATH_MAX_LEN = 65
RUN_LEN_RANGE = (3, 7)
PATH_ATTEMPTS = 300

SEGMENT_COUNT_MAX = 4
CUTS_MAX = SEGMENT_COUNT_MAX - 1

GATE_COUNT_MAX = 4
GEN_ATTEMPTS = 400


def _valid_new_cell(cell: Coord, prev: Coord, occupied: set[Coord], h: int, w: int) -> bool:
    r, c = map(int, cell)
    if not (1 <= r < h - 1 and 1 <= c < w - 1):
        return False
    if cell in occupied:
        return False
    for nbr in _neighbors4(cell, h, w):
        if nbr in occupied and nbr != prev:
            return False
    return True


def _can_extend(last: Coord, direction: Step, length: int, occupied: set[Coord], h: int, w: int) -> bool:
    cur = last
    prev = last
    staged: set[Coord] = set()
    for _ in range(length):
        cur = _add(cur, direction)
        if not _valid_new_cell(cur, prev, occupied | staged, h, w):
            return False
        staged.add(cur)
        prev = cur
    return True


def _build_random_path(h: int, w: int) -> list[Coord] | None:
    for _ in range(PATH_ATTEMPTS):
        start = (random.randint(2, h - 3), random.randint(2, w - 3))
        path = [start]
        occupied = {start}
        direction = random.choice(DIRS4)
        success = True
        target_runs = random.randint(PATH_MIN_RUNS, PATH_MAX_RUNS)

        for run_idx in range(target_runs):
            lengths = [length for length in range(RUN_LEN_RANGE[0], RUN_LEN_RANGE[1] + 1) if _can_extend(path[-1], direction, length, occupied, h, w)]
            if not lengths:
                success = False
                break
            length = random.choice(lengths)
            for _ in range(length):
                nxt = _add(path[-1], direction)
                path.append(nxt)
                occupied.add(nxt)

            if len(path) >= PATH_MIN_LEN and run_idx + 1 >= PATH_MIN_RUNS and random.random() < 0.3:
                break

            turns = [step for step in PERP_DIRS[direction] if _can_extend(path[-1], step, 2, occupied, h, w)]
            if not turns:
                if len(path) >= PATH_MIN_LEN and run_idx + 1 >= PATH_MIN_RUNS:
                    break
                success = False
                break
            direction = random.choice(turns)

        if not success:
            continue
        if not (PATH_MIN_LEN <= len(path) <= PATH_MAX_LEN):
            continue
        if path[0] > path[-1]:
            path = list(reversed(path))
        return path
    return None


def _segment_intervals(length: int, cuts: list[tuple[int, int]]) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    start = 0
    for cut_start, cut_end in sorted(cuts):
        intervals.append((int(start), int(cut_start - 1)))
        start = int(cut_end + 1)
    intervals.append((int(start), int(length - 1)))
    return intervals


def _choose_cuts(path: list[Coord]) -> list[tuple[int, int]] | None:
    runs = [run for run in _straight_runs(path) if int(run["cell_count"]) >= 4]
    if not runs:
        return None
    max_cuts = min(CUTS_MAX, len(runs))
    cut_count = random.randint(1, max_cuts)
    chosen_runs = random.sample(runs, cut_count)
    cuts: list[tuple[int, int]] = []

    for run in chosen_runs:
        max_gap = min(3, int(run["cell_count"]) - 2)
        gap_len = random.randint(1, max_gap)
        start_min = int(run["start"] + 1)
        start_max = int(run["end"] - gap_len)
        cut_start = random.randint(start_min, start_max)
        cut_end = int(cut_start + gap_len - 1)
        cuts.append((int(cut_start), int(cut_end)))

    intervals = _segment_intervals(len(path), cuts)
    if any(int(b - a + 1) < 2 for a, b in intervals):
        return None
    lengths = [int(b - a + 1) for a, b in intervals]
    if any(lengths[i] == lengths[i + 1] for i in range(len(lengths) - 1)):
        return None
    return sorted(cuts)


def _path_membership(length: int, intervals: list[tuple[int, int]]) -> list[int]:
    membership = [-1] * length
    for seg_id, (a, b) in enumerate(intervals):
        for idx in range(a, b + 1):
            membership[idx] = int(seg_id)
    return membership


def _choose_palette(segment_count: int, gate_count: int) -> tuple[int, int, list[int], list[int]]:
    pool = list(ALL_COLORS)
    random.shuffle(pool)
    bg = int(pool.pop())
    grey = int(pool.pop())
    path_colors = [int(pool.pop()) for _ in range(segment_count)]
    gate_colors = [int(pool.pop()) for _ in range(gate_count)]
    return bg, grey, path_colors, gate_colors


def _gate_candidates(
    path: list[Coord],
    membership: list[int],
    occupied: set[Coord],
    h: int,
    w: int,
) -> list[dict]:
    candidates: list[dict] = []
    for idx in range(1, len(path) - 1):
        prev_step = _sub(path[idx], path[idx - 1])
        next_step = _sub(path[idx + 1], path[idx])
        if prev_step != next_step:
            continue
        for perp in PERP_DIRS[prev_step]:
            center = _add(path[idx], perp)
            arm_a = _sub(center, prev_step)
            arm_b = _add(center, prev_step)
            cells = {center, arm_a, arm_b}
            if not all(_in_bounds(r, c, h, w) for r, c in cells):
                continue
            if any(cell in occupied for cell in cells):
                continue
            candidates.append(
                {
                    "path_idx": int(idx),
                    "center": center,
                    "arm_cells": (arm_a, arm_b),
                    "orientation": "h" if prev_step[0] == 0 else "v",
                    "on_gap": bool(membership[idx] == -1),
                }
            )
    return candidates


def _sample_gates(candidates: list[dict]) -> list[dict] | None:
    if not candidates:
        return None
    by_kind = {
        False: [cand for cand in candidates if not bool(cand["on_gap"])],
        True: [cand for cand in candidates if bool(cand["on_gap"])],
    }

    gate_count = random.randint(1, min(GATE_COUNT_MAX, len(candidates)))
    selected: list[dict] = []
    used_cells: set[Coord] = set()

    def _try_add(cand: dict) -> bool:
        cells = {cand["center"], *cand["arm_cells"]}
        if cells & used_cells:
            return False
        if any(abs(int(cand["path_idx"]) - int(other["path_idx"])) < 3 for other in selected):
            return False
        used_cells.update(cells)
        selected.append(cand)
        return True

    if gate_count >= 2 and by_kind[False] and by_kind[True] and random.random() < 0.5:
        first = random.choice(by_kind[False])
        second = random.choice(by_kind[True])
        if not _try_add(first):
            return None
        if not _try_add(second):
            return None

    random.shuffle(candidates)
    for cand in candidates:
        if len(selected) >= gate_count:
            break
        _try_add(cand)

    if len(selected) < gate_count:
        return None
    return sorted(selected, key=lambda item: int(item["path_idx"]))


def _render_expected(
    h: int,
    w: int,
    bg: int,
    grey: int,
    path: list[Coord],
    cuts: list[tuple[int, int]],
    segment_colors: list[int],
    gates: list[dict],
    gate_colors: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((h, w), int(bg), dtype=int)
    out = np.full((h, w), int(bg), dtype=int)

    intervals = _segment_intervals(len(path), cuts)
    membership = _path_membership(len(path), intervals)
    segment_sizes = [int(b - a + 1) for a, b in intervals]

    base_colors: list[int] = [-1] * len(path)
    for seg_id, (a, b) in enumerate(intervals):
        color = int(segment_colors[seg_id])
        for idx in range(a, b + 1):
            r, c = path[idx]
            inp[int(r), int(c)] = color
            base_colors[idx] = color

    for cut_idx, (cut_start, cut_end) in enumerate(cuts):
        left_id = int(membership[cut_start - 1])
        right_id = int(membership[cut_end + 1])
        left_size = int(segment_sizes[left_id])
        right_size = int(segment_sizes[right_id])
        if left_size < right_size:
            fill = int(segment_colors[left_id])
        elif right_size < left_size:
            fill = int(segment_colors[right_id])
        else:
            fill = int(segment_colors[min(left_id, right_id)])
        for idx in range(cut_start, cut_end + 1):
            base_colors[idx] = fill

    out[:] = inp
    for idx, color in enumerate(base_colors):
        r, c = path[idx]
        out[int(r), int(c)] = int(color)

    for gate, gate_color in zip(gates, gate_colors):
        cr, cc = map(int, gate["center"])
        out[int(cr), int(cc)] = int(gate_color)
        inp[int(cr), int(cc)] = int(gate_color)
        for ar, ac in gate["arm_cells"]:
            out[int(ar), int(ac)] = int(grey)
            inp[int(ar), int(ac)] = int(grey)

    trigger_map = {int(gate["path_idx"]): int(color) for gate, color in zip(gates, gate_colors)}
    current: int | None = None
    for idx, (r, c) in enumerate(path):
        if idx in trigger_map:
            current = int(trigger_map[idx])
        if current is not None:
            out[int(r), int(c)] = int(current)

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)

    path = _build_random_path(h, w)
    if path is None:
        return None

    cuts = _choose_cuts(path)
    if cuts is None:
        return None

    intervals = _segment_intervals(len(path), cuts)
    segment_count = len(intervals)
    removed = {idx for a, b in cuts for idx in range(a, b + 1)}
    occupied = set(path[idx] for seg_id, (a, b) in enumerate(intervals) for idx in range(a, b + 1))
    membership = _path_membership(len(path), intervals)

    candidates = _gate_candidates(path, membership, occupied, h, w)
    if not candidates:
        return None
    gates = _sample_gates(candidates)
    if not gates:
        return None

    bg, grey, segment_colors, gate_colors = _choose_palette(segment_count, len(gates))
    inp, expected = _render_expected(h, w, bg, grey, path, cuts, segment_colors, gates, gate_colors)

    # Require both gate placements beside original segments and beside restored links to appear over time.
    if removed and random.random() < 0.5 and not any(bool(gate["on_gap"]) for gate in gates):
        return None

    try:
        got = main(inp)
    except Exception:
        return None
    if not np.array_equal(got, expected):
        return None
    if np.array_equal(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid input")
