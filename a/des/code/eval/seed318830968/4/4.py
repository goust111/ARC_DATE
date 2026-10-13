from __future__ import annotations

import random
from collections import Counter, defaultdict, deque

import numpy as np


BG_COLOR = 0
WALL_COLOR = 9
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _reachable_from(starts: list[tuple[int, int]], allowed: np.ndarray) -> np.ndarray:
    h, w = allowed.shape
    seen = np.zeros_like(allowed, dtype=bool)
    q = deque()
    for r, c in starts:
        if 0 <= r < h and 0 <= c < w and allowed[r, c] and not seen[r, c]:
            seen[r, c] = True
            q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and allowed[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def _run_lengths(mask: np.ndarray, axis: int) -> np.ndarray:
    out = np.zeros_like(mask, dtype=int)
    h, w = mask.shape
    if axis == 1:
        for r in range(h):
            c = 0
            while c < w:
                if not mask[r, c]:
                    c += 1
                    continue
                c0 = c
                while c + 1 < w and mask[r, c + 1]:
                    c += 1
                out[r, c0 : c + 1] = c - c0 + 1
                c += 1
    else:
        for c in range(w):
            r = 0
            while r < h:
                if not mask[r, c]:
                    r += 1
                    continue
                r0 = r
                while r + 1 < h and mask[r + 1, c]:
                    r += 1
                out[r0 : r + 1, c] = r - r0 + 1
                r += 1
    return out


def _cross_sections_for_segment(seg: dict) -> list[list[tuple[int, int]]]:
    cells = []
    if seg["orient"] == "H":
        rows = list(range(seg["r0"], seg["r1"] + 1))
        for c in range(seg["c0"], seg["c1"] + 1):
            cells.append([(r, c) for r in rows])
    else:
        cols = list(range(seg["c0"], seg["c1"] + 1))
        for r in range(seg["r0"], seg["r1"] + 1):
            cells.append([(r, c) for c in cols])
    return cells


def _extract_structure(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    counts = Counter(map(int, g.flatten()))
    bg = int(counts.most_common(1)[0][0])
    non_bg = [color for color in counts if color != bg]
    assert non_bg, "expected walls and seeds"
    wall = int(max(non_bg, key=lambda color: counts[color]))

    seed_pts = [(int(r), int(c), int(g[r, c])) for r, c in np.argwhere((g != bg) & (g != wall))]
    assert seed_pts, "expected at least one seed"

    walkable = _reachable_from([(r, c) for r, c, _ in seed_pts], g != wall)
    hspan = _run_lengths(walkable, axis=1)
    vspan = _run_lengths(walkable, axis=0)

    left = np.zeros_like(walkable, dtype=bool)
    right = np.zeros_like(walkable, dtype=bool)
    up = np.zeros_like(walkable, dtype=bool)
    down = np.zeros_like(walkable, dtype=bool)
    left[:, 1:] = walkable[:, :-1]
    right[:, :-1] = walkable[:, 1:]
    up[1:, :] = walkable[:-1, :]
    down[:-1, :] = walkable[1:, :]

    hinterior = walkable & left & right & (hspan >= 3) & (vspan <= 2)
    vinterior = walkable & up & down & (vspan >= 3) & (hspan <= 2)

    segments: list[dict] = []
    seg_seen: set[tuple] = set()

    def is_h_col(rows: list[int], c: int) -> bool:
        if c < 0 or c >= w:
            return False
        if any(not walkable[r, c] for r in rows):
            return False
        top = rows[0] - 1
        bottom = rows[-1] + 1
        if top >= 0 and walkable[top, c]:
            return False
        if bottom < h and walkable[bottom, c]:
            return False
        return True

    def is_v_row(cols: list[int], r: int) -> bool:
        if r < 0 or r >= h:
            return False
        if any(not walkable[r, c] for c in cols):
            return False
        left_c = cols[0] - 1
        right_c = cols[-1] + 1
        if left_c >= 0 and walkable[r, left_c]:
            return False
        if right_c < w and walkable[r, right_c]:
            return False
        return True

    for comp in _components(hinterior):
        rows = sorted({int(r) for r, _ in comp})
        if not rows or rows[-1] - rows[0] + 1 != len(rows) or len(rows) not in (1, 2):
            continue
        cols = comp[:, 1]
        c0 = int(cols.min())
        c1 = int(cols.max())
        while is_h_col(rows, c0 - 1):
            c0 -= 1
        while is_h_col(rows, c1 + 1):
            c1 += 1
        key = ("H", rows[0], rows[-1], c0, c1)
        if key in seg_seen:
            continue
        seg_seen.add(key)
        segments.append(
            {
                "id": f"S{len(segments)}",
                "orient": "H",
                "r0": rows[0],
                "r1": rows[-1],
                "c0": c0,
                "c1": c1,
                "width": len(rows),
            }
        )

    for comp in _components(vinterior):
        cols = sorted({int(c) for _, c in comp})
        if not cols or cols[-1] - cols[0] + 1 != len(cols) or len(cols) not in (1, 2):
            continue
        rows = comp[:, 0]
        r0 = int(rows.min())
        r1 = int(rows.max())
        while is_v_row(cols, r0 - 1):
            r0 -= 1
        while is_v_row(cols, r1 + 1):
            r1 += 1
        key = ("V", r0, r1, cols[0], cols[-1])
        if key in seg_seen:
            continue
        seg_seen.add(key)
        segments.append(
            {
                "id": f"S{len(segments)}",
                "orient": "V",
                "r0": r0,
                "r1": r1,
                "c0": cols[0],
                "c1": cols[-1],
                "width": len(cols),
            }
        )

    assert segments, "expected corridor segments"

    body_mask = np.zeros_like(walkable, dtype=bool)
    body_owner: dict[tuple[int, int], str] = {}
    for seg in segments:
        steps = _cross_sections_for_segment(seg)
        seg["steps"] = steps
        seg["length"] = len(steps)
        for cell_list in steps:
            for cell in cell_list:
                body_mask[cell] = True
                body_owner[cell] = seg["id"]

    node_mask = walkable & ~body_mask
    node_components = _components(node_mask)
    cell_to_node: dict[tuple[int, int], str] = {}
    nodes: dict[str, dict] = {}
    for comp in node_components:
        nid = f"N{len(nodes)}"
        cells = {(int(r), int(c)) for r, c in comp}
        for cell in cells:
            cell_to_node[cell] = nid
        nodes[nid] = {"id": nid, "cells": cells, "incidents": {}}

    seg_map = {seg["id"]: seg for seg in segments}
    for seg in segments:
        seg["ends"] = [None, None]
        if seg["orient"] == "H":
            rows = list(range(seg["r0"], seg["r1"] + 1))
            left_nodes = {cell_to_node[(r, seg["c0"] - 1)] for r in rows if (r, seg["c0"] - 1) in cell_to_node}
            right_nodes = {cell_to_node[(r, seg["c1"] + 1)] for r in rows if (r, seg["c1"] + 1) in cell_to_node}
            assert len(left_nodes) <= 1 and len(right_nodes) <= 1
            seg["ends"][0] = {"node": next(iter(left_nodes)) if left_nodes else None, "side": "E"}
            seg["ends"][1] = {"node": next(iter(right_nodes)) if right_nodes else None, "side": "W"}
        else:
            cols = list(range(seg["c0"], seg["c1"] + 1))
            top_nodes = {cell_to_node[(seg["r0"] - 1, c)] for c in cols if (seg["r0"] - 1, c) in cell_to_node}
            bottom_nodes = {cell_to_node[(seg["r1"] + 1, c)] for c in cols if (seg["r1"] + 1, c) in cell_to_node}
            assert len(top_nodes) <= 1 and len(bottom_nodes) <= 1
            seg["ends"][0] = {"node": next(iter(top_nodes)) if top_nodes else None, "side": "S"}
            seg["ends"][1] = {"node": next(iter(bottom_nodes)) if bottom_nodes else None, "side": "N"}

        for end_idx, end_info in enumerate(seg["ends"]):
            node_id = end_info["node"]
            if node_id is None:
                continue
            side = end_info["side"]
            assert side not in nodes[node_id]["incidents"], "ambiguous node side"
            nodes[node_id]["incidents"][side] = {"segment": seg["id"], "end": end_idx}

    for node in nodes.values():
        node["wide"] = any(seg_map[inc["segment"]]["width"] == 2 for inc in node["incidents"].values())

    starts = []
    for r, c, color in seed_pts:
        assert (r, c) in body_owner, "seed must sit in a corridor body"
        seg = seg_map[body_owner[(r, c)]]
        start_state = None
        if seg["orient"] == "H":
            if c == seg["c0"]:
                start_state = (seg["id"], 0, 1)
            elif c == seg["c1"]:
                start_state = (seg["id"], seg["length"] - 1, -1)
        else:
            if r == seg["r0"]:
                start_state = (seg["id"], 0, 1)
            elif r == seg["r1"]:
                start_state = (seg["id"], seg["length"] - 1, -1)
        assert start_state is not None, "seed must be on a dead-end tip"
        starts.append({"segment": start_state[0], "index": start_state[1], "dir": start_state[2], "color": color})

    return {
        "grid": g,
        "bg": bg,
        "wall": wall,
        "walkable": walkable,
        "segments": seg_map,
        "nodes": nodes,
        "starts": starts,
        "seed_points": seed_pts,
        "body_owner": body_owner,
    }


def _simulate(structure: dict) -> np.ndarray:
    g = structure["grid"]
    out = g.copy()
    segments = structure["segments"]
    nodes = structure["nodes"]

    claimed: dict[tuple, int | None] = {}
    frontier = []

    for start in structure["starts"]:
        key = ("seg", start["segment"], start["index"])
        if key in claimed:
            claimed[key] = None
            continue
        claimed[key] = start["color"]
        frontier.append(
            {
                "type": "seg",
                "segment": start["segment"],
                "index": start["index"],
                "dir": start["dir"],
                "color": start["color"],
            }
        )

    t = 0
    while frontier:
        proposals: dict[tuple, list[dict]] = defaultdict(list)
        for state in frontier:
            color = state["color"]
            if state["type"] == "seg":
                seg = segments[state["segment"]]
                next_idx = state["index"] + state["dir"]
                if 0 <= next_idx < seg["length"]:
                    proposals[("seg", seg["id"], next_idx)].append(
                        {
                            "type": "seg",
                            "segment": seg["id"],
                            "index": next_idx,
                            "dir": state["dir"],
                            "color": color,
                        }
                    )
                else:
                    end_idx = 0 if next_idx < 0 else 1
                    end_info = seg["ends"][end_idx]
                    node_id = end_info["node"]
                    if node_id is not None:
                        proposals[("node", node_id)].append(
                            {
                                "type": "node",
                                "node": node_id,
                                "enter_side": end_info["side"],
                                "color": color,
                            }
                        )
            else:
                node = nodes[state["node"]]
                if node["wide"]:
                    out_sides = [side for side in node["incidents"] if side != state["enter_side"]]
                else:
                    straight = OPPOSITE[state["enter_side"]]
                    out_sides = [straight] if straight in node["incidents"] else []
                for side in out_sides:
                    inc = node["incidents"][side]
                    seg = segments[inc["segment"]]
                    if inc["end"] == 0:
                        idx = 0
                        direction = 1
                    else:
                        idx = seg["length"] - 1
                        direction = -1
                    proposals[("seg", seg["id"], idx)].append(
                        {
                            "type": "seg",
                            "segment": seg["id"],
                            "index": idx,
                            "dir": direction,
                            "color": color,
                        }
                    )

        new_frontier = []
        for key, props in proposals.items():
            if key in claimed:
                continue
            colors = {int(p["color"]) for p in props}
            if len(colors) != 1:
                claimed[key] = None
                continue
            prop = props[0]
            claimed[key] = int(prop["color"])
            new_frontier.append(prop)
        frontier = new_frontier
        t += 1

    for key, color in claimed.items():
        if color is None:
            continue
        if key[0] == "seg":
            seg = segments[key[1]]
            for r, c in seg["steps"][key[2]]:
                out[r, c] = color
        else:
            for r, c in nodes[key[1]]["cells"]:
                out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each seed inward through corridor mazes.

    Narrow junctions keep only the straight continuation; any junction touched by
    a width-2 corridor branches into every other connected direction. Different
    colors that arrive at the same corridor unit simultaneously leave it black.
    """
    structure = _extract_structure(np.asarray(grid, dtype=int))
    return _simulate(structure)


def _interval_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 < b0 or b1 < a0)


def _candidate_cells(orient: str, fixed: int, start: int, end: int) -> set[tuple[int, int]]:
    if orient == "H":
        return {(fixed, c) for c in range(start, end + 1)}
    return {(r, fixed) for r in range(start, end + 1)}


def _valid_new_strip(strips: list[dict], cand: dict, parent_id: int | None, attach_pos: tuple[int, int] | None) -> bool:
    start = min(cand["start"], cand["end"])
    end = max(cand["start"], cand["end"])
    if end - start < 6:
        return False

    cand_cells = _candidate_cells(cand["orient"], cand["fixed"], start, end)
    allowed = {attach_pos} if attach_pos is not None else set()

    for strip in strips:
        s0 = min(strip["start"], strip["end"])
        s1 = max(strip["start"], strip["end"])
        if strip["orient"] == cand["orient"]:
            if strip["fixed"] == cand["fixed"] and _interval_overlap(start, end, s0, s1):
                return False
            if abs(strip["fixed"] - cand["fixed"]) < 4 and _interval_overlap(start - 1, end + 1, s0 - 1, s1 + 1):
                return False
        else:
            if cand["orient"] == "H":
                cross = (cand["fixed"], strip["fixed"])
                hit = s0 <= cand["fixed"] <= s1 and start <= strip["fixed"] <= end
            else:
                cross = (strip["fixed"], cand["fixed"])
                hit = start <= strip["fixed"] <= end and s0 <= cand["fixed"] <= s1
            if hit and cross not in allowed:
                return False

        if cand_cells & strip["cells"] - allowed:
            return False
    return True


def _make_root(rng: random.Random, h: int, w: int) -> dict:
    orient = rng.choice(["H", "V"])
    if orient == "H":
        fixed = rng.randint(7, h - 9)
        start = rng.randint(4, 7)
        end = rng.randint(w - 16, w - 5)
    else:
        fixed = rng.randint(7, w - 9)
        start = rng.randint(4, 7)
        end = rng.randint(h - 16, h - 5)
    root = {
        "id": 0,
        "orient": orient,
        "fixed": fixed,
        "start": start,
        "end": end,
        "anchor": None,
        "attachments": [],
    }
    root["cells"] = _candidate_cells(orient, fixed, start, end)
    return root


def _attachment_choices(strip: dict) -> list[int]:
    lo = min(strip["start"], strip["end"]) + 6
    hi = max(strip["start"], strip["end"]) - 6
    if lo > hi:
        return []
    pts = list(range(lo, hi + 1))
    used = set(strip["attachments"])
    return [p for p in pts if all(abs(p - q) >= 6 for q in used)]


def _add_child(rng: random.Random, strips: list[dict], parent: dict, h: int, w: int) -> dict | None:
    positions = _attachment_choices(parent)
    rng.shuffle(positions)
    for pos in positions:
        if parent["orient"] == "H":
            fixed = pos
            anchor = parent["fixed"]
            mode = rng.choice(["one", "one", "both"])
            up = rng.randint(6, 10) if mode == "both" or rng.random() < 0.5 else 0
            down = rng.randint(6, 10) if mode == "both" or up == 0 else rng.randint(6, 10)
            start = anchor - up
            end = anchor + down
            orient = "V"
            if start < 3 or end > h - 4:
                continue
            attach = (anchor, fixed)
        else:
            fixed = pos
            anchor = parent["fixed"]
            mode = rng.choice(["one", "one", "both"])
            left = rng.randint(6, 10) if mode == "both" or rng.random() < 0.5 else 0
            right = rng.randint(6, 10) if mode == "both" or left == 0 else rng.randint(6, 10)
            start = anchor - left
            end = anchor + right
            orient = "H"
            if start < 3 or end > w - 4:
                continue
            attach = (fixed, anchor)

        cand = {
            "id": len(strips),
            "orient": orient,
            "fixed": fixed,
            "start": start,
            "end": end,
            "anchor": anchor,
            "attachments": [],
        }
        cand["cells"] = _candidate_cells(orient, fixed, start, end)
        if _valid_new_strip(strips, cand, parent["id"], attach):
            parent["attachments"].append(pos)
            strips.append(cand)
            return cand
    return None


def _build_topology(rng: random.Random, h: int, w: int) -> list[dict] | None:
    strips = [_make_root(rng, h, w)]
    frontier = [(0, 0)]
    wanted_nodes = rng.randint(2, 4)
    made_nodes = 0
    while frontier and made_nodes < wanted_nodes:
        idx, depth = frontier.pop(0)
        parent = strips[idx]
        child_goal = rng.randint(1, 2 if depth else 3)
        made_here = 0
        attempts = 0
        while made_here < child_goal and attempts < 20 and made_nodes < wanted_nodes:
            child = _add_child(rng, strips, parent, h, w)
            attempts += 1
            if child is None:
                continue
            frontier.append((child["id"], depth + 1))
            made_here += 1
            made_nodes += 1
        if depth >= 1:
            continue
    return strips if made_nodes >= 2 else None


def _build_pieces(strips: list[dict]) -> tuple[list[dict], list[tuple[int, int]]]:
    pieces = []
    leaf_ends: list[tuple[int, int]] = []
    for strip in strips:
        marks = {strip["start"], strip["end"], *strip["attachments"]}
        if strip["anchor"] is not None:
            marks.add(strip["anchor"])
        pts = sorted(marks)
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            if strip["orient"] == "H":
                piece = {
                    "id": len(pieces),
                    "orient": "H",
                    "fixed": strip["fixed"],
                    "start": a,
                    "end": b,
                    "strip_id": strip["id"],
                    "low_is_leaf": a == strip["start"] and strip["anchor"] != strip["start"],
                    "high_is_leaf": b == strip["end"] and strip["anchor"] != strip["end"],
                }
            else:
                piece = {
                    "id": len(pieces),
                    "orient": "V",
                    "fixed": strip["fixed"],
                    "start": a,
                    "end": b,
                    "strip_id": strip["id"],
                    "low_is_leaf": a == strip["start"] and strip["anchor"] != strip["start"],
                    "high_is_leaf": b == strip["end"] and strip["anchor"] != strip["end"],
                }
            pieces.append(piece)
            if piece["low_is_leaf"]:
                leaf_ends.append((piece["id"], 0))
            if piece["high_is_leaf"]:
                leaf_ends.append((piece["id"], 1))
    return pieces, leaf_ends


def _piece_rect(piece: dict, width: int) -> tuple[int, int, int, int]:
    if piece["orient"] == "H":
        return piece["fixed"], piece["fixed"] + width - 1, piece["start"], piece["end"]
    return piece["start"], piece["end"], piece["fixed"], piece["fixed"] + width - 1


def _assign_widths(rng: random.Random, pieces: list[dict]) -> dict[int, int] | None:
    widths = {}
    internal = []
    for piece in pieces:
        if piece["low_is_leaf"] or piece["high_is_leaf"]:
            widths[piece["id"]] = 1
        else:
            internal.append(piece["id"])
            widths[piece["id"]] = rng.choice([1, 2])
    if not internal:
        return None
    if all(widths[pid] == 1 for pid in internal):
        widths[rng.choice(internal)] = 2
    return widths


def _render_candidate(
    rng: random.Random, h: int, w: int, pieces: list[dict], widths: dict[int, int], leaf_ends: list[tuple[int, int]]
) -> np.ndarray:
    carved = np.zeros((h, w), dtype=bool)
    rmin, rmax, cmin, cmax = h, -1, w, -1
    for piece in pieces:
        r0, r1, c0, c1 = _piece_rect(piece, widths[piece["id"]])
        carved[r0 : r1 + 1, c0 : c1 + 1] = True
        rmin = min(rmin, r0)
        rmax = max(rmax, r1)
        cmin = min(cmin, c0)
        cmax = max(cmax, c1)

    grid = np.full((h, w), BG_COLOR, dtype=int)
    grid[max(1, rmin - 1) : min(h - 1, rmax + 2), max(1, cmin - 1) : min(w - 1, cmax + 2)] = WALL_COLOR
    grid[carved] = BG_COLOR

    colors = list(SEED_COLORS)
    rng.shuffle(colors)
    assert len(leaf_ends) <= len(colors)
    for color, (piece_id, end_idx) in zip(colors, leaf_ends):
        piece = pieces[piece_id]
        width = widths[piece_id]
        assert width == 1, "leaf corridors must stay narrow"
        if piece["orient"] == "H":
            r = piece["fixed"]
            c = piece["start"] if end_idx == 0 else piece["end"]
        else:
            r = piece["start"] if end_idx == 0 else piece["end"]
            c = piece["fixed"]
        grid[r, c] = color
    return grid


def _valid_generated_grid(grid: np.ndarray, expected_seed_count: int) -> bool:
    try:
        structure = _extract_structure(grid)
    except AssertionError:
        return False

    if len(structure["starts"]) != expected_seed_count or expected_seed_count < 3:
        return False
    if not structure["nodes"]:
        return False
    degrees = [len(node["incidents"]) for node in structure["nodes"].values()]
    if any(deg not in (3, 4) for deg in degrees):
        return False
    wide = sum(1 for node in structure["nodes"].values() if node["wide"])
    narrow = sum(1 for node in structure["nodes"].values() if not node["wide"])
    if wide == 0 or narrow == 0:
        return False
    out = _simulate(structure)
    if np.array_equal(out, grid):
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(24, 36)
    w = rng.randint(24, 36)
    topology = _build_topology(rng, h, w)
    if topology is None:
        return None
    pieces, leaf_ends = _build_pieces(topology)
    if len(leaf_ends) < 3 or len(leaf_ends) > len(SEED_COLORS):
        return None
    widths = _assign_widths(rng, pieces)
    if widths is None:
        return None
    grid = _render_candidate(rng, h, w, pieces, widths, leaf_ends)
    if not _valid_generated_grid(grid, len(leaf_ends)):
        return None
    return grid


def generate_input() -> np.ndarray:
    """
    Build enclosed orthogonal corridor trees with both narrow-only and wide-trigger
    junctions. Leaf corridors stay width 1 so every seed sits at an actual dead end.
    """
    rng = random.Random()
    for _ in range(5000):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid maze after 5000 attempts")
