from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np


NORTH = (-1, 0)
SOUTH = (1, 0)
WEST = (0, -1)
EAST = (0, 1)


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in (NORTH, SOUTH, WEST, EAST):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _uniform_color(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> int | None:
    r0, c0, r1, c1 = bbox
    if r0 < 0 or c0 < 0 or r1 >= grid.shape[0] or c1 >= grid.shape[1]:
        return None
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    if patch.shape != (2, 2):
        return None
    color = int(patch[0, 0])
    if bool((patch == color).all()):
        return color
    return None


def _extract_strips(pipe_mask: np.ndarray) -> tuple[list[dict], list[dict]]:
    h, w = pipe_mask.shape

    row_runs: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
    for r in range(h):
        c = 0
        while c < w:
            if not pipe_mask[r, c]:
                c += 1
                continue
            c0 = c
            while c + 1 < w and pipe_mask[r, c + 1]:
                c += 1
            c1 = c
            if c1 - c0 + 1 >= 3:
                row_runs[(c0, c1)].append(r)
            c += 1

    col_runs: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
    for c in range(w):
        r = 0
        while r < h:
            if not pipe_mask[r, c]:
                r += 1
                continue
            r0 = r
            while r + 1 < h and pipe_mask[r + 1, c]:
                r += 1
            r1 = r
            if r1 - r0 + 1 >= 3:
                col_runs[(r0, r1)].append(c)
            r += 1

    h_strips: list[dict] = []
    for idx, ((c0, c1), rows) in enumerate(row_runs.items()):
        rows = sorted(rows)
        start = 0
        while start < len(rows):
            end = start
            while end + 1 < len(rows) and rows[end + 1] == rows[end] + 1:
                end += 1
            group = rows[start : end + 1]
            assert 1 <= len(group) <= 2, "horizontal strip width must be 1 or 2"
            r0, r1 = int(group[0]), int(group[-1])
            patch = pipe_mask[r0 : r1 + 1, c0 : c1 + 1]
            assert bool(patch.all()), "horizontal strip must be a solid rectangle"
            h_strips.append(
                {
                    "id": f"H{idx}_{start}",
                    "orient": "H",
                    "r0": r0,
                    "c0": int(c0),
                    "r1": r1,
                    "c1": int(c1),
                    "width": int(r1 - r0 + 1),
                }
            )
            start = end + 1

    v_strips: list[dict] = []
    for idx, ((r0, r1), cols) in enumerate(col_runs.items()):
        cols = sorted(cols)
        start = 0
        while start < len(cols):
            end = start
            while end + 1 < len(cols) and cols[end + 1] == cols[end] + 1:
                end += 1
            group = cols[start : end + 1]
            assert 1 <= len(group) <= 2, "vertical strip width must be 1 or 2"
            c0, c1 = int(group[0]), int(group[-1])
            patch = pipe_mask[r0 : r1 + 1, c0 : c1 + 1]
            assert bool(patch.all()), "vertical strip must be a solid rectangle"
            v_strips.append(
                {
                    "id": f"V{idx}_{start}",
                    "orient": "V",
                    "r0": int(r0),
                    "c0": c0,
                    "r1": int(r1),
                    "c1": c1,
                    "width": int(c1 - c0 + 1),
                }
            )
            start = end + 1

    assert h_strips and v_strips, "expected both horizontal and vertical pipe strips"
    return h_strips, v_strips


def _candidate_terminal_bboxes(strip: dict, side: str) -> list[tuple[int, int, int, int]]:
    r0, c0, r1, c1 = map(int, (strip["r0"], strip["c0"], strip["r1"], strip["c1"]))
    width = int(strip["width"])
    out: list[tuple[int, int, int, int]] = []

    if strip["orient"] == "H":
        row_starts = [r0] if width == 2 else [r0 - 1, r0]
        col_start = c0 - 2 if side == "L" else c1 + 1
        for rs in row_starts:
            out.append((rs, col_start, rs + 1, col_start + 1))
    else:
        col_starts = [c0] if width == 2 else [c0 - 1, c0]
        row_start = r0 - 2 if side == "T" else r1 + 1
        for cs in col_starts:
            out.append((row_start, cs, row_start + 1, cs + 1))
    return out


def _build_graph(grid: np.ndarray) -> tuple[dict, dict]:
    bg = _most_common_color(grid)
    counts = Counter(map(int, grid.flatten()))
    pipe = None
    for color, _count in counts.most_common():
        if int(color) != int(bg):
            pipe = int(color)
            break
    assert pipe is not None, "expected a non-background pipe color"

    pipe_mask = grid == int(pipe)
    h_strips, v_strips = _extract_strips(pipe_mask)

    nodes: dict[str, dict] = {}
    strip_events: dict[str, list[tuple[int, str]]] = defaultdict(list)
    touched_ends: dict[str, dict[str, bool]] = {}

    for strip in h_strips:
        touched_ends[str(strip["id"])] = {"L": False, "R": False}
    for strip in v_strips:
        touched_ends[str(strip["id"])] = {"T": False, "B": False}

    junction_id = 0
    for h_strip in h_strips:
        for v_strip in v_strips:
            rr0 = max(int(h_strip["r0"]), int(v_strip["r0"]))
            rr1 = min(int(h_strip["r1"]), int(v_strip["r1"]))
            cc0 = max(int(h_strip["c0"]), int(v_strip["c0"]))
            cc1 = min(int(h_strip["c1"]), int(v_strip["c1"]))
            if rr0 > rr1 or cc0 > cc1:
                continue

            node_id = f"J{junction_id}"
            junction_id += 1
            nodes[node_id] = {"kind": "junction"}
            strip_events[str(h_strip["id"])].append((int(cc0 + cc1), node_id))
            strip_events[str(v_strip["id"])].append((int(rr0 + rr1), node_id))

            if int(cc0) == int(h_strip["c0"]):
                touched_ends[str(h_strip["id"])]["L"] = True
            if int(cc1) == int(h_strip["c1"]):
                touched_ends[str(h_strip["id"])]["R"] = True
            if int(rr0) == int(v_strip["r0"]):
                touched_ends[str(v_strip["id"])]["T"] = True
            if int(rr1) == int(v_strip["r1"]):
                touched_ends[str(v_strip["id"])]["B"] = True

    terminal_lookup: dict[tuple[int, int, int, int], str] = {}
    terminal_id = 0

    for strip in h_strips + v_strips:
        if strip["orient"] == "H":
            side_names = ("L", "R")
            positions = {"L": 2 * int(strip["c0"]) - 1, "R": 2 * int(strip["c1"]) + 1}
        else:
            side_names = ("T", "B")
            positions = {"T": 2 * int(strip["r0"]) - 1, "B": 2 * int(strip["r1"]) + 1}

        for side in side_names:
            if touched_ends[str(strip["id"])][side]:
                continue

            matches: list[tuple[tuple[int, int, int, int], int]] = []
            for bbox in _candidate_terminal_bboxes(strip, side):
                color = _uniform_color(grid, bbox)
                if color is None or int(color) == int(pipe):
                    continue
                matches.append((bbox, int(color)))

            if len(matches) != 1:
                continue

            bbox, color = matches[0]
            if bbox in terminal_lookup:
                node_id = terminal_lookup[bbox]
            else:
                node_id = f"T{terminal_id}"
                terminal_id += 1
                kind = "collector" if int(color) == int(bg) else "source"
                info = {"kind": kind, "bbox": bbox}
                if kind == "source":
                    info["color"] = int(color)
                nodes[node_id] = info
                terminal_lookup[bbox] = node_id
            strip_events[str(strip["id"])].append((positions[side], node_id))

    adjacency: dict[str, list[dict]] = defaultdict(list)
    edge_id = 0

    for strip in h_strips:
        events = sorted(strip_events[str(strip["id"])], key=lambda item: item[0])
        assert len(events) >= 2, "each horizontal strip must connect at least two events"
        for (_pa, node_a), (_pb, node_b) in zip(events, events[1:]):
            if node_a == node_b:
                continue
            eid = f"E{edge_id}"
            edge_id += 1
            width = int(strip["width"])
            adjacency[node_a].append({"to": node_b, "edge": eid, "dir": EAST, "width": width})
            adjacency[node_b].append({"to": node_a, "edge": eid, "dir": WEST, "width": width})

    for strip in v_strips:
        events = sorted(strip_events[str(strip["id"])], key=lambda item: item[0])
        assert len(events) >= 2, "each vertical strip must connect at least two events"
        for (_pa, node_a), (_pb, node_b) in zip(events, events[1:]):
            if node_a == node_b:
                continue
            eid = f"E{edge_id}"
            edge_id += 1
            width = int(strip["width"])
            adjacency[node_a].append({"to": node_b, "edge": eid, "dir": SOUTH, "width": width})
            adjacency[node_b].append({"to": node_a, "edge": eid, "dir": NORTH, "width": width})

    sources = [node_id for node_id, info in nodes.items() if info["kind"] == "source"]
    collectors = [node_id for node_id, info in nodes.items() if info["kind"] == "collector"]
    assert sources, "expected at least one source terminal"
    assert collectors, "expected at least one collector terminal"

    for source in sources:
        assert len(adjacency[source]) == 1, "source terminal must be attached to exactly one strip"
    for collector in collectors:
        assert len(adjacency[collector]) == 1, "collector terminal must be attached to exactly one strip"

    return nodes, adjacency


def _choose_step(options: list[dict], prev_node: str, move_dir: tuple[int, int]) -> dict:
    candidates = [item for item in options if item["to"] != prev_node]
    assert candidates, "path cannot dead-end before reaching a collector"
    if len(candidates) == 1:
        return candidates[0]

    best_width = max(int(item["width"]) for item in candidates)
    best = [item for item in candidates if int(item["width"]) == int(best_width)]
    if len(best) == 1:
        return best[0]

    straight = [item for item in best if tuple(item["dir"]) == tuple(move_dir)]
    assert len(straight) == 1, "equal-width branch choice must be resolved by going straight"
    return straight[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route each source signal through an orthogonal pipe network.

    Signals enter the pipe connected to their 2x2 source block. At each junction,
    they choose the widest available non-backtracking branch; equal widths break
    toward the straight-ahead continuation. When a signal reaches a 2x2 collector,
    that collector is filled with the source color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    nodes, adjacency = _build_graph(g)
    out = g.copy()

    for source_id, source_info in nodes.items():
        if source_info["kind"] != "source":
            continue

        color = int(source_info["color"])
        first = adjacency[source_id][0]
        prev_node = source_id
        cur_node = str(first["to"])
        move_dir = tuple(first["dir"])
        seen: set[tuple[str, str, tuple[int, int]]] = {(prev_node, cur_node, move_dir)}

        while True:
            cur_info = nodes[cur_node]
            if cur_info["kind"] == "collector":
                r0, c0, r1, c1 = map(int, cur_info["bbox"])
                out[r0 : r1 + 1, c0 : c1 + 1] = color
                break

            assert cur_info["kind"] == "junction", "signals may only traverse junctions before collectors"
            step = _choose_step(adjacency[cur_node], prev_node, move_dir)
            next_node = str(step["to"])
            next_dir = tuple(step["dir"])
            state = (cur_node, next_node, next_dir)
            assert state not in seen, "signal routing must not loop"
            seen.add(state)
            prev_node, cur_node, move_dir = cur_node, next_node, next_dir

    return out


_BG_CODE = 0
_PIPE_CODE = 1
_SOURCE_CODE = 2
_COLLECTOR_CODE = 3


def _paint_rect(arr: np.ndarray, r0: int, c0: int, h: int, w: int, value: int) -> None:
    patch = arr[r0 : r0 + h, c0 : c0 + w]
    assert patch.shape == (h, w)
    if value != _BG_CODE:
        assert bool(((patch == _BG_CODE) | (patch == value)).all())
    arr[r0 : r0 + h, c0 : c0 + w] = value


def _build_turn_wider(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    h_lead = rng.randint(4, 6)
    v_to_dec = rng.randint(3, 5)
    v_after = rng.randint(3, 5)
    chosen_h = rng.randint(4, 6)
    decoy_h = rng.randint(3, 5)

    bend_col = h_lead + 1
    decision_r = 1 + v_to_dec
    lower_r = decision_r + v_after
    height = lower_r + 2
    width = bend_col + max(chosen_h, decoy_h) + 2

    arr = np.zeros((height, width), dtype=int)
    target = np.zeros((height, width), dtype=bool)

    _paint_rect(arr, 0, 0, 2, 2, _SOURCE_CODE)
    _paint_rect(arr, 0, 2, 2, h_lead, _PIPE_CODE)
    _paint_rect(arr, 1, bend_col, lower_r + 1, 1, _PIPE_CODE)
    _paint_rect(arr, decision_r, bend_col, 2, chosen_h, _PIPE_CODE)
    _paint_rect(arr, decision_r, bend_col + chosen_h, 2, 2, _COLLECTOR_CODE)
    _paint_rect(arr, lower_r, bend_col, 2, decoy_h, _PIPE_CODE)
    _paint_rect(arr, lower_r, bend_col + decoy_h, 2, 2, _COLLECTOR_CODE)

    target[decision_r : decision_r + 2, bend_col + chosen_h : bend_col + chosen_h + 2] = True
    return arr, target


def _build_straight_tie(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    h_lead = rng.randint(4, 6)
    v_to_dec = rng.randint(3, 5)
    v_after = rng.randint(5, 6)
    decoy_h = rng.randint(3, 5)
    decoy_v = rng.randint(3, v_after - 2)
    final_h = rng.randint(4, 6)

    bend_col = h_lead + 1
    decision_r = 1 + v_to_dec
    lower_r = decision_r + v_after
    turn_col = bend_col + decoy_h - 1
    height = max(lower_r + 2, decision_r + decoy_v + 2)
    width = max(bend_col + final_h + 2, turn_col + 2)

    arr = np.zeros((height, width), dtype=int)
    target = np.zeros((height, width), dtype=bool)

    _paint_rect(arr, 0, 0, 2, 2, _SOURCE_CODE)
    _paint_rect(arr, 0, 2, 2, h_lead, _PIPE_CODE)
    _paint_rect(arr, 1, bend_col, lower_r + 1, 1, _PIPE_CODE)
    _paint_rect(arr, decision_r, bend_col, 1, decoy_h, _PIPE_CODE)
    _paint_rect(arr, decision_r, turn_col, decoy_v, 2, _PIPE_CODE)
    _paint_rect(arr, decision_r + decoy_v, turn_col, 2, 2, _COLLECTOR_CODE)
    _paint_rect(arr, lower_r, bend_col, 2, final_h, _PIPE_CODE)
    _paint_rect(arr, lower_r, bend_col + final_h, 2, 2, _COLLECTOR_CODE)

    target[lower_r : lower_r + 2, bend_col + final_h : bend_col + final_h + 2] = True
    return arr, target


def _build_straight_wider(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    left_len = rng.randint(3, 5)
    right_len = rng.randint(3, 5)
    lower_v = rng.randint(3, 5)
    lower_h = rng.randint(3, 5)

    branch_col = left_len + 1
    total_h = left_len + right_len
    lower_r = 1 + lower_v
    height = lower_r + 2
    width = max(2 + total_h + 2, branch_col + lower_h + 2)

    arr = np.zeros((height, width), dtype=int)
    target = np.zeros((height, width), dtype=bool)

    _paint_rect(arr, 0, 0, 2, 2, _SOURCE_CODE)
    _paint_rect(arr, 0, 2, 2, total_h, _PIPE_CODE)
    _paint_rect(arr, 0, 2 + total_h, 2, 2, _COLLECTOR_CODE)
    _paint_rect(arr, 1, branch_col, lower_r + 1, 1, _PIPE_CODE)
    _paint_rect(arr, lower_r, branch_col, 2, lower_h, _PIPE_CODE)
    _paint_rect(arr, lower_r, branch_col + lower_h, 2, 2, _COLLECTOR_CODE)

    target[0:2, 2 + total_h : 2 + total_h + 2] = True
    return arr, target


def _rotate_component(
    arr: np.ndarray, target: np.ndarray, turns: int
) -> tuple[np.ndarray, np.ndarray]:
    return np.rot90(arr, turns), np.rot90(target, turns)


def _choose_component_types(rng: random.Random, count: int) -> list[str]:
    kinds = ["turn_wider", "straight_tie", "straight_wider"]
    selected = rng.sample(kinds, k=min(count, len(kinds)))
    while len(selected) < count:
        selected.append(rng.choice(kinds))
    rng.shuffle(selected)
    return selected


def _make_component(kind: str, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    if kind == "turn_wider":
        return _build_turn_wider(rng)
    if kind == "straight_tie":
        return _build_straight_tie(rng)
    if kind == "straight_wider":
        return _build_straight_wider(rng)
    raise ValueError(f"unknown component kind: {kind}")


def _render_component(
    grid: np.ndarray,
    occupied: np.ndarray,
    component: np.ndarray,
    target: np.ndarray,
    source_color: int,
    pipe_color: int,
    bg_color: int,
    rng: random.Random,
) -> tuple[int, int] | None:
    h, w = grid.shape
    ch, cw = component.shape
    if ch + 2 > h or cw + 2 > w:
        return None

    for _ in range(200):
        r0 = rng.randint(1, h - ch - 1)
        c0 = rng.randint(1, w - cw - 1)
        r1 = r0 + ch
        c1 = c0 + cw
        if bool(occupied[r0 - 1 : r1 + 1, c0 - 1 : c1 + 1].any()):
            continue

        view = component
        region = grid[r0:r1, c0:c1]
        region[view == _PIPE_CODE] = pipe_color
        region[view == _SOURCE_CODE] = source_color
        region[view == _COLLECTOR_CODE] = bg_color
        occupied[r0:r1, c0:c1] |= view != _BG_CODE
        return r0, c0

    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    component_count = rng.randint(2, 4)
    bg_color = rng.randrange(10)
    pipe_color = rng.choice([c for c in range(10) if c != bg_color])
    source_colors = rng.sample([c for c in range(10) if c not in {bg_color, pipe_color}], k=component_count)

    components: list[tuple[np.ndarray, np.ndarray, int]] = []
    for kind, source_color in zip(_choose_component_types(rng, component_count), source_colors):
        base, target = _make_component(kind, rng)
        turns = rng.randrange(4)
        comp, rotated_target = _rotate_component(base, target, turns)
        components.append((comp, rotated_target, source_color))

    max_h = max(comp.shape[0] for comp, _, _ in components)
    max_w = max(comp.shape[1] for comp, _, _ in components)
    packed_area = sum((comp.shape[0] + 2) * (comp.shape[1] + 2) for comp, _, _ in components)
    side = int(np.ceil(np.sqrt(packed_area))) + rng.randint(6, 10)
    h = max(max_h + 4, side + rng.randint(0, 5))
    w = max(max_w + 4, side + rng.randint(0, 7))

    grid = np.full((h, w), bg_color, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    placements: list[tuple[int, int, np.ndarray, int]] = []

    for comp, target, source_color in components:
        placed = _render_component(grid, occupied, comp, target, source_color, pipe_color, bg_color, rng)
        if placed is None:
            return None
        placements.append((placed[0], placed[1], target, source_color))

    expected = grid.copy()
    for r0, c0, target, source_color in placements:
        mask = target
        expected[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]][mask] = source_color

    try:
        solved = main(grid)
    except AssertionError:
        return None

    if not bool(np.array_equal(solved, expected)):
        return None

    if int(np.count_nonzero(solved != grid)) != 4 * component_count:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        rng = random.Random()
        result = _gen_attempt(rng)
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid pipe-routing puzzle")
