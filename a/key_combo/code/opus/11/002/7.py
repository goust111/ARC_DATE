import importlib.util
import math
import random
from pathlib import Path

import numpy as np


def _load_arc_utils():
    try:
        from arc_utils import get_bg, in_bounds

        return get_bg, in_bounds
    except Exception:
        arc_utils_path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
        spec = importlib.util.spec_from_file_location("_shared_arc_utils_7", arc_utils_path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)
        return module.get_bg, module.in_bounds


get_bg, in_bounds = _load_arc_utils()

BG = 0
VERTEX = 1
SHADOW = 2
BLOCKER = 5
DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

GRID_H_RANGE = (18, 25)
GRID_W_RANGE = (18, 25)
GEN_RETRIES = 500
RECENT_OUTPUT_LIMIT = 256
_RECENT_OUTPUT_SIGS: set[tuple[tuple[int, int], bytes]] = set()


def _aligned_8(a: tuple[int, int], b: tuple[int, int]) -> bool:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    return (dr == 0 and dc != 0) or (dc == 0 and dr != 0) or (dr != 0 and abs(dr) == abs(dc))


def _step_dir(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    return (0 if dr == 0 else (1 if dr > 0 else -1), 0 if dc == 0 else (1 if dc > 0 else -1))


def _line_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    sr, sc = _step_dir(a, b)
    steps = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
    return [(a[0] + sr * k, a[1] + sc * k) for k in range(steps + 1)]


def _angle_cycle(points_rc: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cr = sum(r for r, _ in points_rc) / len(points_rc)
    cc = sum(c for _, c in points_rc) / len(points_rc)

    def key(p: tuple[int, int]) -> tuple[float, float, int, int]:
        r, c = p
        ang = math.atan2(r - cr, c - cc)
        rad = (r - cr) ** 2 + (c - cc) ** 2
        return (ang, rad, r, c)

    return sorted(points_rc, key=key)


def _interior_clear(g: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> bool:
    pts = _line_cells(a, b)
    for r, c in pts[1:-1]:
        if int(g[r, c]) != BG:
            return False
    return True


def _rotate_cw90(v: tuple[int, int]) -> tuple[int, int]:
    dr, dc = v
    return dc, -dr


def _rotate_ccw90(v: tuple[int, int]) -> tuple[int, int]:
    dr, dc = v
    return -dc, dr


def _outer_normal(
    source: tuple[int, int],
    edge_dir: tuple[int, int],
    centroid: tuple[float, float],
) -> tuple[int, int]:
    cand = _rotate_cw90(edge_dir)
    to_centroid = (centroid[0] - source[0], centroid[1] - source[1])
    if cand[0] * to_centroid[0] + cand[1] * to_centroid[1] > 0:
        cand = _rotate_ccw90(edge_dir)
    return cand


def _choose_start(cycle: list[tuple[int, int]], g: np.ndarray) -> tuple[int, int]:
    seed_indices = []
    h, w = g.shape
    for i, (r, c) in enumerate(cycle):
        for dr, dc in DIR8:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == SHADOW:
                seed_indices.append(i)
                break
    if seed_indices:
        start_idx = min(seed_indices, key=lambda i: cycle[i])
    else:
        start_idx = min(range(len(cycle)), key=lambda i: cycle[i])
    return cycle[start_idx:] + cycle[:start_idx], start_idx


def _trace(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    vertices = [tuple(map(int, rc)) for rc in np.argwhere(g == VERTEX)]
    if len(vertices) < 2:
        return g.copy(), {
            "vertices": vertices,
            "cycle": vertices[:],
            "start_vertex": vertices[0] if vertices else None,
            "edges": [],
            "shadow_events": [],
            "shadow_skip_blue": 0,
            "shadow_skip_oob": 0,
            "ray_events": [],
            "ray_cells": 0,
        }

    cycle0 = _angle_cycle(vertices)
    cycle, _ = _choose_start(cycle0, g)
    start_vertex = cycle[0]
    centroid = (
        sum(r for r, _ in vertices) / len(vertices),
        sum(c for _, c in vertices) / len(vertices),
    )

    anchor_offsets = []
    sr0, sc0 = start_vertex
    for dr, dc in DIR8:
        rr, cc = sr0 + dr, sc0 + dc
        if 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == SHADOW:
            anchor_offsets.append((dr, dc))
    shadow_offset = min(anchor_offsets) if anchor_offsets else None

    if len(cycle) == 2:
        edge_pairs = [(cycle[0], cycle[1])]
    else:
        edge_pairs = [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]

    edges = []
    out = g.copy()
    for a, b in edge_pairs:
        aligned = _aligned_8(a, b)
        clear = aligned and _interior_clear(g, a, b)
        cells = _line_cells(a, b) if aligned else []
        edge = {
            "a": a,
            "b": b,
            "aligned": aligned,
            "clear": clear,
            "drawn": aligned and clear,
            "cells": cells if aligned and clear else [],
            "dir": _step_dir(a, b) if aligned else None,
        }
        edges.append(edge)
        if edge["drawn"]:
            for r, c in edge["cells"]:
                out[r, c] = VERTEX

    shadow_events = []
    shadow_skip_blue = 0
    shadow_skip_oob = 0
    if shadow_offset is not None:
        drs, dcs = shadow_offset
        for edge_index, edge in enumerate(edges):
            if not edge["drawn"]:
                continue
            for point_index, (r, c) in enumerate(edge["cells"]):
                rr, cc = r + drs, c + dcs
                event = {
                    "edge_index": edge_index,
                    "point_index": point_index,
                    "edge_dir": edge["dir"],
                    "base_point": (r, c),
                    "source": (rr, cc),
                    "placed": False,
                    "skip": None,
                }
                if not (0 <= rr < h and 0 <= cc < w):
                    event["skip"] = "oob"
                    shadow_skip_oob += 1
                elif int(out[rr, cc]) == VERTEX:
                    event["skip"] = "blue"
                    shadow_skip_blue += 1
                else:
                    out[rr, cc] = SHADOW
                    event["placed"] = True
                shadow_events.append(event)

    blocker_map = out.copy()
    ray_events = []
    for event in shadow_events:
        if not event["placed"]:
            continue
        start = event["source"]
        normal = _outer_normal(start, event["edge_dir"], centroid)
        ray_cells = []
        r, c = start
        stop = "boundary"
        while True:
            nr, nc = r + normal[0], c + normal[1]
            if not in_bounds(g.shape, nr, nc):
                stop = "boundary"
                break
            val = int(blocker_map[nr, nc])
            if val != BG:
                if val == BLOCKER:
                    stop = "blocker"
                elif val == VERTEX:
                    stop = "blue"
                elif val == SHADOW:
                    stop = "red"
                else:
                    stop = "occupied"
                break
            ray_cells.append((nr, nc))
            r, c = nr, nc
        for rr, cc in ray_cells:
            out[rr, cc] = SHADOW
        ray_events.append(
            {
                "start": start,
                "dir": normal,
                "cells": ray_cells,
                "stop": stop,
                "edge_index": event["edge_index"],
                "point_index": event["point_index"],
            }
        )

    debug = {
        "vertices": vertices,
        "cycle": cycle,
        "start_vertex": start_vertex,
        "shadow_offset": shadow_offset,
        "edges": edges,
        "shadow_events": shadow_events,
        "shadow_skip_blue": shadow_skip_blue,
        "shadow_skip_oob": shadow_skip_oob,
        "ray_events": ray_events,
        "ray_cells": sum(len(ev["cells"]) for ev in ray_events),
    }
    return out, debug


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace a blue polygon outline, carry one adjacent shadow offset along it, and emit outward perpendicular red rays.

    Input structure:
    - A black background contains 4-6 blue single-pixel vertices that roughly form one convex polygon.
    - Exactly one blue vertex has an 8-neighbor red pixel; that red pixel defines the carried shadow offset.
    - Grey single pixels act as blockers and are not part of the polygon.

    Rule:
    - Sort the blue vertices by polar angle around their centroid and connect consecutive vertices in that cyclic order.
    - Draw only horizontal, vertical, or 45-degree diagonal edges whose interior cells are black in the original input.
    - For every pixel on each drawn edge, place a red shadow at the same offset as the start vertex's adjacent red pixel, unless that offset cell is out of bounds or already blue.
    - From every successfully placed shadow, emit a red ray along the outward normal of its edge; stop before the first boundary, grey blocker, blue outline cell, or pre-existing shadow cell.
    - Draw the outline first, then all carried shadows, then the rays; later rays overwrite earlier rays if they overlap.

    Key insight:
    - The red rays are not anchored to fixed input dots: they are generated from shadow pixels copied along the reconstructed outline, and each ray direction depends on the geometry of the edge that carried it.
    """
    out, _ = _trace(np.asarray(grid, dtype=int))
    return out


def _bbox_of_points(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _shift_points(points: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for r, c in points]


def _make_rectangle() -> list[tuple[int, int]]:
    h = random.randint(4, 8)
    w = random.randint(4, 8)
    return [(0, 0), (0, w), (h, w), (h, 0)]


def _make_diamond() -> list[tuple[int, int]]:
    arm = random.randint(3, 6)
    return [(0, arm), (arm, 2 * arm), (2 * arm, arm), (arm, 0)]


def _make_pent_right() -> list[tuple[int, int]]:
    h = random.randint(5, 8)
    w = random.randint(5, 8)
    cut = random.randint(2, min(h - 1, w - 1, 4))
    return [(0, 0), (0, w - cut), (cut, w), (h, w), (h, 0)]


def _make_pent_left() -> list[tuple[int, int]]:
    h = random.randint(5, 8)
    w = random.randint(5, 8)
    cut = random.randint(2, min(h - 1, w - 1, 4))
    return [(0, cut), (0, w), (h, w), (h, 0), (cut, 0)]


def _make_hex() -> list[tuple[int, int]]:
    h = random.randint(5, 8)
    w = random.randint(5, 8)
    cut1 = random.randint(2, min(h - 1, w - 1, 4))
    cut2 = random.randint(2, min(h - 1, w - 1, 4))
    return [(0, 0), (0, w - cut1), (cut1, w), (h, w), (h, cut2), (h - cut2, 0)]


def _polygon_template() -> list[tuple[int, int]]:
    builders = [_make_rectangle, _make_diamond, _make_pent_right, _make_pent_left, _make_hex]
    weights = [20, 18, 20, 20, 22]
    return random.choices(builders, weights=weights, k=1)[0]()


def _random_shift_for_points(
    points: list[tuple[int, int]],
    shape: tuple[int, int],
    tight_side: str | None,
) -> tuple[int, int] | None:
    h, w = shape
    r0, c0, r1, c1 = _bbox_of_points(points)
    ph = r1 - r0 + 1
    pw = c1 - c0 + 1
    if ph > h - 2 or pw > w - 2:
        return None

    base_top = random.randint(2, max(2, h - ph - 2))
    base_left = random.randint(2, max(2, w - pw - 2))
    top = base_top
    left = base_left
    if tight_side == "top":
        top = random.randint(1, 2)
    elif tight_side == "left":
        left = random.randint(1, 2)
    elif tight_side == "bottom":
        top = random.randint(max(1, h - ph - 3), h - ph - 1)
    elif tight_side == "right":
        left = random.randint(max(1, w - pw - 3), w - pw - 1)
    return top - r0, left - c0


def _input_grid(shape: tuple[int, int], vertices: list[tuple[int, int]], start: tuple[int, int], offset: tuple[int, int]) -> np.ndarray:
    g = np.zeros(shape, dtype=int)
    for r, c in vertices:
        g[r, c] = VERTEX
    g[start[0] + offset[0], start[1] + offset[1]] = SHADOW
    return g


def _candidate_anchor_options(vertices: list[tuple[int, int]], shape: tuple[int, int], mode: str) -> list[tuple[int, tuple[int, int], dict]]:
    options = []
    vertices_set = set(vertices)
    for start in random.sample(vertices, len(vertices)):
        for offset in random.sample(DIR8, len(DIR8)):
            rr, cc = start[0] + offset[0], start[1] + offset[1]
            if not in_bounds(shape, rr, cc) or (rr, cc) in vertices_set:
                continue
            if any((rr, cc) != start and max(abs(rr - vr), abs(cc - vc)) == 1 for vr, vc in vertices if (vr, vc) != start):
                continue
            g = _input_grid(shape, vertices, start, offset)
            out, dbg = _trace(g)
            if dbg["start_vertex"] != start:
                continue
            if dbg["shadow_offset"] != offset:
                continue
            if sum(1 for edge in dbg["edges"] if edge["drawn"]) != len(vertices):
                continue
            placed = sum(1 for ev in dbg["shadow_events"] if ev["placed"])
            unique_shadow_positions = len({ev["source"] for ev in dbg["shadow_events"] if ev["placed"]})
            max_ray = max((len(ev["cells"]) for ev in dbg["ray_events"]), default=0)
            ray_nonempty = sum(1 for ev in dbg["ray_events"] if ev["cells"])
            if placed < 6 or unique_shadow_positions < 4 or ray_nonempty < 3 or max_ray < 2:
                continue
            if mode == "tangent" and dbg["shadow_skip_blue"] < 1:
                continue
            if mode == "outside" and dbg["shadow_skip_oob"] > 1:
                continue
            if mode == "outside" and dbg["shadow_skip_blue"] > max(2, placed // 3):
                continue
            if np.array_equal(out, g):
                continue
            score = placed + dbg["ray_cells"] - 2 * dbg["shadow_skip_blue"] - dbg["shadow_skip_oob"]
            options.append((score, start, offset, dbg))
    options.sort(key=lambda item: item[0], reverse=True)
    return [(start, offset, dbg) for _, start, offset, dbg in options[:8]]


def _safe_for_input_blocker(
    cell: tuple[int, int],
    vertices: list[tuple[int, int]],
    anchor_red: tuple[int, int],
    forbidden: set[tuple[int, int]],
) -> bool:
    if cell in forbidden:
        return False
    if cell == anchor_red:
        return False
    for vr, vc in vertices + [anchor_red]:
        if abs(cell[0] - vr) + abs(cell[1] - vc) == 1:
            return False
    return True


def _place_targeted_blockers(
    g: np.ndarray,
    vertices: list[tuple[int, int]],
    anchor_red: tuple[int, int],
    dbg: dict,
    clip_bias: float,
) -> set[tuple[int, int]]:
    blockers: set[tuple[int, int]] = set()
    shadow_positions = {ev["source"] for ev in dbg["shadow_events"] if ev["placed"]}
    forbidden = set(vertices) | shadow_positions
    ray_events = dbg["ray_events"][:]
    random.shuffle(ray_events)
    for ev in ray_events:
        if len(ev["cells"]) < 2 or random.random() > clip_bias:
            continue
        choices = ev["cells"][1 : min(len(ev["cells"]), 4)]
        random.shuffle(choices)
        for cell in choices:
            if _safe_for_input_blocker(cell, vertices, anchor_red, forbidden | blockers):
                blockers.add(cell)
                break
    for r, c in blockers:
        g[r, c] = BLOCKER
    return blockers


def _add_extra_blockers(
    g: np.ndarray,
    vertices: list[tuple[int, int]],
    anchor_red: tuple[int, int],
    forbidden: set[tuple[int, int]],
    target_count: int,
) -> None:
    h, w = g.shape
    pool = [(r, c) for r in range(h) for c in range(w) if int(g[r, c]) == BG]
    random.shuffle(pool)
    count = 0
    for cell in pool:
        if count >= target_count:
            break
        if not _safe_for_input_blocker(cell, vertices, anchor_red, forbidden):
            continue
        g[cell] = BLOCKER
        forbidden.add(cell)
        count += 1


def _output_sig(out: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return out.shape, out.tobytes()


def _accept_semantics(inp: np.ndarray, out: np.ndarray, dbg: dict, mode: str) -> bool:
    if inp.ndim != 2 or out.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return False
    if len(dbg["vertices"]) not in {4, 5, 6}:
        return False
    if dbg["shadow_offset"] is None:
        return False
    edge_fill = sum(max(0, len(edge["cells"]) - 2) for edge in dbg["edges"] if edge["drawn"])
    if edge_fill < 6:
        return False
    placed = sum(1 for ev in dbg["shadow_events"] if ev["placed"])
    if placed < 6:
        return False
    nonempty_rays = [ev for ev in dbg["ray_events"] if ev["cells"]]
    if len(nonempty_rays) < 4:
        return False
    if dbg["ray_cells"] < 8 or dbg["ray_cells"] > 55:
        return False
    stop_reasons = {ev["stop"] for ev in dbg["ray_events"]}
    if "blocker" not in stop_reasons and "boundary" not in stop_reasons:
        return False
    if mode == "tangent" and dbg["shadow_skip_blue"] < 1:
        return False
    if np.count_nonzero(inp) >= np.count_nonzero(out):
        return False
    if np.array_equal(inp, out):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    mode = random.choice(["outside", "outside", "tangent", "outside", "tangent"])
    tight_side = random.choice([None, None, "top", "bottom", "left", "right"])
    clip_bias = random.choice([0.45, 0.6, 0.75])

    local_vertices = _polygon_template()
    shift = _random_shift_for_points(local_vertices, (h, w), tight_side)
    if shift is None:
        return None
    vertices = _shift_points(local_vertices, *shift)
    if len(vertices) not in {4, 5, 6}:
        return None

    anchor_options = _candidate_anchor_options(vertices, (h, w), mode)
    if not anchor_options:
        return None
    start, offset, pre_dbg = random.choice(anchor_options)
    inp = _input_grid((h, w), vertices, start, offset)
    anchor_red = (start[0] + offset[0], start[1] + offset[1])

    blockers = _place_targeted_blockers(inp, vertices, anchor_red, pre_dbg, clip_bias)
    shadow_positions = {ev["source"] for ev in pre_dbg["shadow_events"] if ev["placed"]}
    forbidden = set(vertices) | shadow_positions | blockers | {anchor_red}
    extra_target = random.randint(2, 6)
    _add_extra_blockers(inp, vertices, anchor_red, forbidden, extra_target)

    out, dbg = _trace(inp)
    if not _accept_semantics(inp, out, dbg, mode):
        return None

    sig = _output_sig(out)
    if sig in _RECENT_OUTPUT_SIGS:
        return None
    if len(_RECENT_OUTPUT_SIGS) >= RECENT_OUTPUT_LIMIT:
        _RECENT_OUTPUT_SIGS.clear()
    _RECENT_OUTPUT_SIGS.add(sig)
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid puzzle input")
