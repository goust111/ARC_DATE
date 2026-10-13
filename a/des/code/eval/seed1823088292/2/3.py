import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


BLACK = 0
WALL = 5
SEED_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
SIDES = ("N", "E", "S", "W")
ORIENT_SIDES = {"H": ("W", "E"), "V": ("N", "S")}

GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
MODULE_COUNT_CHOICES = [1, 2, 2, 3]
MIN_SEED_ARM = 4
MIN_DEAD_ARM = 3
MAX_DEAD_ARM = 8
SPEC_TRIES = 200
PLACEMENT_TRIES = 160
GEN_RETRIES = 300


def _line_runs(mask_1d: np.ndarray) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    n = int(mask_1d.shape[0])
    i = 0
    while i < n:
        if not bool(mask_1d[i]):
            i += 1
            continue
        j = i
        while j + 1 < n and bool(mask_1d[j + 1]):
            j += 1
        runs.append((int(i), int(j)))
        i = j + 1
    return runs


def _find_horizontal_strips(free: np.ndarray) -> List[Dict[str, int]]:
    h, _w = free.shape
    strips: List[Dict[str, int]] = []
    open_rects: Dict[Tuple[int, int], List[int]] = {}

    for r in range(h):
        row_keys = set(_line_runs(free[r]))
        for key in list(open_rects):
            if key not in row_keys:
                r0, r1 = open_rects.pop(key)
                c0, c1 = key
                height = r1 - r0 + 1
                width = c1 - c0 + 1
                if width > height:
                    strips.append({"r0": r0, "r1": r1, "c0": c0, "c1": c1, "width": height})
        for key in row_keys:
            if key in open_rects:
                open_rects[key][1] = r
            else:
                open_rects[key] = [r, r]

    for key, (r0, r1) in list(open_rects.items()):
        c0, c1 = key
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if width > height:
            strips.append({"r0": r0, "r1": r1, "c0": c0, "c1": c1, "width": height})

    strips.sort(key=lambda s: (s["r0"], s["c0"], s["r1"], s["c1"]))
    return strips


def _find_vertical_strips(free: np.ndarray) -> List[Dict[str, int]]:
    h, w = free.shape
    strips: List[Dict[str, int]] = []
    open_rects: Dict[Tuple[int, int], List[int]] = {}

    for c in range(w):
        col_keys = set(_line_runs(free[:, c]))
        for key in list(open_rects):
            if key not in col_keys:
                c0, c1 = open_rects.pop(key)
                r0, r1 = key
                height = r1 - r0 + 1
                width = c1 - c0 + 1
                if height > width:
                    strips.append({"r0": r0, "r1": r1, "c0": c0, "c1": c1, "width": width})
        for key in col_keys:
            if key in open_rects:
                open_rects[key][1] = c
            else:
                open_rects[key] = [c, c]

    for key, (c0, c1) in list(open_rects.items()):
        r0, r1 = key
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height > width:
            strips.append({"r0": r0, "r1": r1, "c0": c0, "c1": c1, "width": width})

    strips.sort(key=lambda s: (s["c0"], s["r0"], s["c1"], s["r1"]))
    return strips


def _overlap_bbox(a: Dict[str, int], b: Dict[str, int]) -> Optional[Tuple[int, int, int, int]]:
    r0 = max(int(a["r0"]), int(b["r0"]))
    r1 = min(int(a["r1"]), int(b["r1"]))
    c0 = max(int(a["c0"]), int(b["c0"]))
    c1 = min(int(a["c1"]), int(b["c1"]))
    if r0 > r1 or c0 > c1:
        return None
    return (r0, r1, c0, c1)


def _build_graph(grid: np.ndarray) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    free = grid != WALL
    h, w = free.shape
    h_strips = _find_horizontal_strips(free)
    v_strips = _find_vertical_strips(free)

    nodes: List[Dict[str, object]] = []
    h_nodes: List[List[int]] = [[] for _ in h_strips]
    v_nodes: List[List[int]] = [[] for _ in v_strips]

    for hi, h_strip in enumerate(h_strips):
        for vi, v_strip in enumerate(v_strips):
            bbox = _overlap_bbox(h_strip, v_strip)
            if bbox is None:
                continue
            node_id = len(nodes)
            nodes.append({"bbox": bbox, "ports": {}})
            h_nodes[hi].append(node_id)
            v_nodes[vi].append(node_id)

    segments: List[Dict[str, object]] = []

    def add_segment(
        orient: str,
        bbox: Tuple[int, int, int, int],
        width: int,
        end_a: Dict[str, object],
        end_b: Dict[str, object],
    ) -> None:
        seg_id = len(segments)
        segment = {"orient": orient, "bbox": bbox, "width": width, "ends": [end_a, end_b]}
        segments.append(segment)
        for end_index, end in enumerate(segment["ends"]):
            if end["kind"] == "node":
                node_id = int(end["node"])
                port = str(end["port"])
                nodes[node_id]["ports"][port] = (seg_id, end_index)

    for hi, h_strip in enumerate(h_strips):
        attached = sorted(h_nodes[hi], key=lambda nid: int(nodes[nid]["bbox"][2]))
        cur_c = int(h_strip["c0"])
        prev_end: Dict[str, object] = {"kind": "dead", "touches_border": cur_c == 0, "side": "W"}
        for node_id in attached:
            nr0, nr1, nc0, nc1 = map(int, nodes[node_id]["bbox"])
            if cur_c <= nc0 - 1:
                add_segment(
                    "H",
                    (int(h_strip["r0"]), int(h_strip["r1"]), cur_c, nc0 - 1),
                    int(h_strip["width"]),
                    prev_end,
                    {"kind": "node", "node": node_id, "port": "W"},
                )
            cur_c = nc1 + 1
            prev_end = {"kind": "node", "node": node_id, "port": "E"}
        if cur_c <= int(h_strip["c1"]):
            add_segment(
                "H",
                (int(h_strip["r0"]), int(h_strip["r1"]), cur_c, int(h_strip["c1"])),
                int(h_strip["width"]),
                prev_end,
                {"kind": "dead", "touches_border": int(h_strip["c1"]) == w - 1, "side": "E"},
            )

    for vi, v_strip in enumerate(v_strips):
        attached = sorted(v_nodes[vi], key=lambda nid: int(nodes[nid]["bbox"][0]))
        cur_r = int(v_strip["r0"])
        prev_end = {"kind": "dead", "touches_border": cur_r == 0, "side": "N"}
        for node_id in attached:
            nr0, nr1, nc0, nc1 = map(int, nodes[node_id]["bbox"])
            if cur_r <= nr0 - 1:
                add_segment(
                    "V",
                    (cur_r, nr0 - 1, int(v_strip["c0"]), int(v_strip["c1"])),
                    int(v_strip["width"]),
                    prev_end,
                    {"kind": "node", "node": node_id, "port": "N"},
                )
            cur_r = nr1 + 1
            prev_end = {"kind": "node", "node": node_id, "port": "S"}
        if cur_r <= int(v_strip["r1"]):
            add_segment(
                "V",
                (cur_r, int(v_strip["r1"]), int(v_strip["c0"]), int(v_strip["c1"])),
                int(v_strip["width"]),
                prev_end,
                {"kind": "dead", "touches_border": int(v_strip["r1"]) == h - 1, "side": "S"},
            )

    return nodes, segments


def _paint_rect(out: np.ndarray, bbox: Tuple[int, int, int, int], color: int) -> None:
    r0, r1, c0, c1 = bbox
    patch = out[r0 : r1 + 1, c0 : c1 + 1]
    blocked = (patch != BLACK) & (patch != WALL) & (patch != color)
    if bool(blocked.any()):
        return
    patch[patch != WALL] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route colored signals through black channels bounded by grey walls.

    Signals start from colored border pixels placed on pipe openings. They fill
    the traversable channel until a dead-end or a junction. At a junction, the
    signal only continues into the widest outgoing branch; ties split into all
    equally widest branches.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    nodes, segments = _build_graph(g)

    seeds = sorted(
        (int(r), int(c), int(g[r, c]))
        for r, c in np.argwhere((g != BLACK) & (g != WALL))
    )

    visited: set[Tuple[int, int]] = set()

    def travel(seg_id: int, from_end: int, color: int) -> None:
        state = (int(seg_id), int(from_end))
        if state in visited:
            return
        visited.add(state)

        segment = segments[seg_id]
        bbox = tuple(int(v) for v in segment["bbox"])
        _paint_rect(out, bbox, color)

        to_end = 1 - int(from_end)
        end = segment["ends"][to_end]
        if end["kind"] != "node":
            return

        node_id = int(end["node"])
        node_bbox = tuple(int(v) for v in nodes[node_id]["bbox"])
        node_patch = out[node_bbox[0] : node_bbox[1] + 1, node_bbox[2] : node_bbox[3] + 1]
        if bool(((node_patch != BLACK) & (node_patch != WALL) & (node_patch != color)).any()):
            return
        _paint_rect(out, node_bbox, color)

        incoming_port = str(end["port"])
        candidates = []
        for port, (next_seg_id, next_end_index) in nodes[node_id]["ports"].items():
            if port == incoming_port:
                continue
            candidates.append(
                (
                    str(port),
                    int(next_seg_id),
                    int(next_end_index),
                    int(segments[int(next_seg_id)]["width"]),
                )
            )
        if not candidates:
            return

        max_width = max(width for _port, _sid, _idx, width in candidates)
        for _port, next_seg_id, next_end_index, width in candidates:
            if width == max_width:
                travel(next_seg_id, next_end_index, color)

    for r, c, color in seeds:
        matched = False
        for seg_id, segment in enumerate(segments):
            r0, r1, c0, c1 = map(int, segment["bbox"])
            if not (r0 <= r <= r1 and c0 <= c <= c1):
                continue
            orient = str(segment["orient"])
            if orient == "H":
                if c == c0:
                    travel(seg_id, 0, color)
                    matched = True
                    break
                if c == c1:
                    travel(seg_id, 1, color)
                    matched = True
                    break
            else:
                if r == r0:
                    travel(seg_id, 0, color)
                    matched = True
                    break
                if r == r1:
                    travel(seg_id, 1, color)
                    matched = True
                    break
        if not matched:
            out[r, c] = color

    return out


def _selected_ports(ports: Sequence[str], seed_side: str, wh: int, wv: int) -> List[str]:
    widths = {}
    for port in ports:
        if port == seed_side:
            continue
        widths[port] = wh if port in ("W", "E") else wv
    max_width = max(widths.values())
    return sorted(port for port, width in widths.items() if width == max_width)


def _sample_module_spec(rng: random.Random) -> Dict[str, object]:
    target_case = rng.choice(["single", "split", "all"])
    for _ in range(SPEC_TRIES):
        ports = set(SIDES)
        if rng.random() < 0.55:
            ports.remove(rng.choice(SIDES))
        seed_side = rng.choice(sorted(ports))
        wh = rng.randint(1, 3)
        wv = rng.randint(1, 3)
        selected = _selected_ports(sorted(ports), seed_side, wh, wv)
        if target_case == "single" and len(selected) != 1:
            continue
        if target_case == "split" and len(selected) != 2:
            continue
        if target_case == "all" and len(selected) != len(ports) - 1:
            continue
        return {
            "ports": tuple(sorted(ports)),
            "seed_side": seed_side,
            "wh": wh,
            "wv": wv,
        }
    raise RuntimeError("failed to sample module spec")


def _module_cells(
    spec: Dict[str, object],
    h: int,
    w: int,
    rng: random.Random,
) -> Optional[Tuple[set[Tuple[int, int]], Tuple[int, int]]]:
    ports = set(str(p) for p in spec["ports"])
    seed_side = str(spec["seed_side"])
    wh = int(spec["wh"])
    wv = int(spec["wv"])

    for _ in range(PLACEMENT_TRIES):
        r0 = rng.randint(1, h - wh - 1)
        c0 = rng.randint(1, w - wv - 1)

        max_len = {
            "W": c0 - 1,
            "E": w - (c0 + wv) - 1,
            "N": r0 - 1,
            "S": h - (r0 + wh) - 1,
        }
        seed_len = {
            "W": c0,
            "E": w - (c0 + wv),
            "N": r0,
            "S": h - (r0 + wh),
        }[seed_side]
        if seed_len < MIN_SEED_ARM:
            continue

        lengths = {side: 0 for side in SIDES}
        lengths[seed_side] = seed_len
        ok = True
        for side in ports:
            if side == seed_side:
                continue
            if max_len[side] < MIN_DEAD_ARM:
                ok = False
                break
            lengths[side] = rng.randint(MIN_DEAD_ARM, min(MAX_DEAD_ARM, max_len[side]))
        if not ok:
            continue

        cells: set[Tuple[int, int]] = set()
        if "W" in ports or "E" in ports:
            left = c0 - lengths["W"] if "W" in ports else c0
            right = c0 + wv - 1 + lengths["E"] if "E" in ports else c0 + wv - 1
            for rr in range(r0, r0 + wh):
                for cc in range(left, right + 1):
                    cells.add((rr, cc))
        if "N" in ports or "S" in ports:
            top = r0 - lengths["N"] if "N" in ports else r0
            bottom = r0 + wh - 1 + lengths["S"] if "S" in ports else r0 + wh - 1
            for rr in range(top, bottom + 1):
                for cc in range(c0, c0 + wv):
                    cells.add((rr, cc))

        border_cells = [(rr, cc) for rr, cc in cells if rr in (0, h - 1) or cc in (0, w - 1)]
        if not border_cells:
            continue
        if seed_side == "W" and any(cc != 0 for _rr, cc in border_cells):
            continue
        if seed_side == "E" and any(cc != w - 1 for _rr, cc in border_cells):
            continue
        if seed_side == "N" and any(rr != 0 for rr, _cc in border_cells):
            continue
        if seed_side == "S" and any(rr != h - 1 for rr, _cc in border_cells):
            continue

        seed_candidates = []
        if seed_side in ("W", "E"):
            border_col = 0 if seed_side == "W" else w - 1
            for rr in range(r0, r0 + wh):
                if (rr, border_col) in cells:
                    seed_candidates.append((rr, border_col))
        else:
            border_row = 0 if seed_side == "N" else h - 1
            for cc in range(c0, c0 + wv):
                if (border_row, cc) in cells:
                    seed_candidates.append((border_row, cc))
        if not seed_candidates:
            continue
        return cells, rng.choice(seed_candidates)

    return None


def _place_modules(rng: random.Random) -> Optional[np.ndarray]:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    num_modules = rng.choice(MODULE_COUNT_CHOICES)
    if min(h, w) < 22 and num_modules == 3:
        num_modules = 2

    grid = np.full((h, w), WALL, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)
    colors = rng.sample(SEED_COLORS, num_modules)

    placed = 0
    for color in colors:
        success = False
        for _ in range(PLACEMENT_TRIES):
            spec = _sample_module_spec(rng)
            built = _module_cells(spec, h, w, rng)
            if built is None:
                continue
            cells, seed_cell = built
            min_r = min(r for r, _c in cells)
            max_r = max(r for r, _c in cells)
            min_c = min(c for _r, c in cells)
            max_c = max(c for _r, c in cells)
            r0 = max(0, min_r - 1)
            r1 = min(h - 1, max_r + 1)
            c0 = max(0, min_c - 1)
            c1 = min(w - 1, max_c + 1)
            if bool(reserved[r0 : r1 + 1, c0 : c1 + 1].any()):
                continue

            for rr, cc in cells:
                grid[rr, cc] = BLACK
            sr, sc = seed_cell
            grid[sr, sc] = color
            reserved[r0 : r1 + 1, c0 : c1 + 1] = True
            placed += 1
            success = True
            break
        if not success:
            return None

    if placed == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    """
    Generate disconnected pipe modules with varied junction type, width relation,
    orientation, placement, opening width, and signal color.
    """
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _place_modules(rng)
        if grid is None:
            continue
        try:
            out = main(grid)
        except Exception:
            continue
        if np.array_equal(out, grid):
            continue
        if int(((out != grid) & (grid == BLACK)).sum()) < 8:
            continue
        return grid
    raise RuntimeError("generate_input: exhausted retries")
