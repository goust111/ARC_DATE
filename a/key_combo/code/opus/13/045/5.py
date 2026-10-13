from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

STRUCT4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def neighbors4(r: int, c: int):
    for dr, dc in STRUCT4:
        yield r + dr, c + dc


def label_mask(mask: np.ndarray, connectivity: int = 4) -> tuple[np.ndarray, int]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    dirs = STRUCT4
    if connectivity == 8:
        dirs = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0))
    cur = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            cur += 1
            q = deque([(r, c)])
            labels[r, c] = cur
            while q:
                rr, cc = q.popleft()
                for dr, dc in dirs:
                    nr = rr + dr
                    nc = cc + dc
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = cur
                        q.append((nr, nc))
    return labels, cur


def dilate8(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r_src0 = max(0, -dr)
            r_src1 = min(h, h - dr)
            c_src0 = max(0, -dc)
            c_src1 = min(w, w - dc)
            r_dst0 = max(0, dr)
            r_dst1 = min(h, h + dr)
            c_dst0 = max(0, dc)
            c_dst1 = min(w, w + dc)
            out[r_dst0:r_dst1, c_dst0:c_dst1] |= mask[r_src0:r_src1, c_src0:c_src1]
    return out


def component_cells(labels: np.ndarray, cid: int) -> list[tuple[int, int]]:
    pts = np.argwhere(labels == cid)
    return [(int(r), int(c)) for r, c in pts]


def monochrome_objects(grid: np.ndarray) -> list[dict]:
    out: list[dict] = []
    for color in sorted(int(c) for c in np.unique(grid) if int(c) not in (BLACK, ORANGE)):
        labels, n = label_mask(grid == color, connectivity=8)
        for cid in range(1, n + 1):
            cells = component_cells(labels, cid)
            if not cells:
                continue
            rows = [r for r, _ in cells]
            cols = [c for _, c in cells]
            out.append(
                {
                    "color": int(color),
                    "cells": set(cells),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return out


def hole_infos_for_object(obj: dict) -> list[dict]:
    r0, c0, r1, c1 = obj["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    local = np.zeros((h, w), dtype=bool)
    for r, c in obj["cells"]:
        local[r - r0, c - c0] = True
    bg = ~local
    labels, n = label_mask(bg, connectivity=4)
    holes: list[dict] = []
    for cid in range(1, n + 1):
        pts = component_cells(labels, cid)
        if not pts:
            continue
        if any(rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1 for rr, cc in pts):
            continue
        hole_global = {(r0 + rr, c0 + cc) for rr, cc in pts}
        boundary = set()
        for rr, cc in hole_global:
            for nr, nc in neighbors4(rr, cc):
                if (nr, nc) in obj["cells"]:
                    boundary.add((nr, nc))
        holes.append({"cells": hole_global, "boundary": boundary})
    return holes


def channel_segments(grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    path = g == ORANGE
    labels, n = label_mask(path, connectivity=4)
    assert n == 2, "expected exactly two orange path components"
    d0 = dilate8(labels == 1)
    d1 = dilate8(labels == 2)
    channel = d0 & d1 & (~path)
    channel_labels, m = label_mask(channel, connectivity=4)
    segs: list[dict] = []
    h, _w = g.shape
    for cid in range(1, m + 1):
        cells = component_cells(channel_labels, cid)
        top = any(r == 0 for r, _ in cells)
        bottom = any(r == h - 1 for r, _ in cells)
        if top and bottom:
            kind = "both"
        elif top:
            kind = "top"
        elif bottom:
            kind = "bottom"
        else:
            kind = "middle"
        segs.append({"id": cid, "cells": set(cells), "top": top, "bottom": bottom, "kind": kind})
    return channel_labels, segs


def hole_fill_color(adjacent: list[dict]) -> int:
    if any(seg["top"] and seg["bottom"] for seg in adjacent):
        return GREY
    if any(seg["top"] for seg in adjacent):
        return RED
    if any(seg["bottom"] for seg in adjacent):
        return BLUE
    return BLACK


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill object holes using the edge-connectivity of the dilated overlap channel between two orange paths.

    Input structure:
    - Black background with exactly two orange 4-connected one-pixel paths.
    - Other foreground components are monochrome 8-connected objects, each containing 1 or more enclosed black holes.
    - The objects sit near the middle overlap channel induced by the two paths.

    Rule:
    - Dilate each orange path by one cell in the 8-neighborhood and intersect the two dilations to get the channel; split that channel into 4-connected segments.
    - For each hole, inspect every channel segment that is 4-neighbor-adjacent to a boundary pixel of that hole.
    - Fill the hole grey if any adjacent segment reaches both top and bottom edges; otherwise fill red if any adjacent segment reaches the top edge, blue if only bottom-reaching segments are adjacent, and leave it black if all adjacent segments are edge-isolated.
    - After all hole fills, erase the orange paths and the channel itself to black and keep the full canvas size.

    Key insight:
    - The channel is only an intermediate geometric witness: its global edge connectivity, not its visible color, determines each hole fill, and then both the paths and the channel disappear.
    """
    g = np.asarray(grid, dtype=int).copy()
    channel_labels, segs = channel_segments(g)
    seg_by_id = {seg["id"]: seg for seg in segs}

    for obj in monochrome_objects(g):
        for hole in hole_infos_for_object(obj):
            adjacent_ids = set()
            for r, c in hole["boundary"]:
                for nr, nc in neighbors4(r, c):
                    if in_bounds(g.shape, nr, nc) and channel_labels[nr, nc] > 0:
                        adjacent_ids.add(int(channel_labels[nr, nc]))
            if not adjacent_ids:
                continue
            fill = hole_fill_color([seg_by_id[cid] for cid in sorted(adjacent_ids)])
            if fill != BLACK:
                for r, c in hole["cells"]:
                    g[r, c] = fill

    path = g == ORANGE
    channel = channel_labels > 0
    g[path | channel] = BLACK
    return g


def trace_polyline(start: tuple[int, int], segments: list[tuple[int, int, int]]) -> list[tuple[int, int]]:
    r, c = start
    pts = [(int(r), int(c))]
    for dr, dc, length in segments:
        for _ in range(length):
            r += dr
            c += dc
            pts.append((int(r), int(c)))
    return pts


def wall_sets_for_channel(path: list[tuple[int, int]]) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    left: set[tuple[int, int]] = set()
    right: set[tuple[int, int]] = set()
    channel = set(path)

    def left_normal(dr: int, dc: int) -> tuple[int, int]:
        return -dc, dr

    for i, (r, c) in enumerate(path):
        dirs = []
        if i > 0:
            pr, pc = path[i - 1]
            dirs.append((r - pr, c - pc))
        if i + 1 < len(path):
            nr, nc = path[i + 1]
            dirs.append((nr - r, nc - c))
        for dr, dc in dirs:
            lr, lc = left_normal(dr, dc)
            if (r + lr, c + lc) not in channel:
                left.add((r + lr, c + lc))
            if (r - lr, c - lc) not in channel:
                right.add((r - lr, c - lc))
        if len(dirs) == 2 and dirs[0] != dirs[1]:
            for side in (1, -1):
                offsets = []
                for dr, dc in dirs:
                    lr, lc = left_normal(dr, dc)
                    offsets.append((side * lr, side * lc))
                direct = [(r + dr, c + dc) for dr, dc in offsets]
                if all(cell not in channel for cell in direct):
                    elbow = (r + offsets[0][0] + offsets[1][0], c + offsets[0][1] + offsets[1][1])
                    if side == 1:
                        left.add(elbow)
                    else:
                        right.add(elbow)
    return left, right


def add_vertical(cells: set[tuple[int, int]], col: int, h: int) -> None:
    for r in range(h):
        cells.add((r, col))


def add_hline(cells: set[tuple[int, int]], row: int, c0: int, c1: int) -> None:
    lo = min(c0, c1)
    hi = max(c0, c1)
    for c in range(lo, hi + 1):
        cells.add((row, c))


def add_vline(cells: set[tuple[int, int]], col: int, r0: int, r1: int) -> None:
    lo = min(r0, r1)
    hi = max(r0, r1)
    for r in range(lo, hi + 1):
        cells.add((r, col))


def layout_polylines(h: int, w: int, profile: str) -> dict[str, list[tuple[int, int]]]:
    c_top = 4 + random.randint(0, 1)
    c_both = 11 + random.randint(-1, 1)
    c_mid = 18 + random.randint(-1, 1)
    c_bottom = w - 8 + random.randint(-1, 0)

    top_drop = random.randint(4, 6)
    top_tail = random.randint(2, 3)
    bottom_r0 = h - random.randint(8, 10)
    bottom_drop_0 = random.randint(2, 3)
    bottom_drop_1 = h - 1 - (bottom_r0 + bottom_drop_0)
    mid_r0 = h // 2 - random.randint(2, 3)
    both_r1 = random.randint(3, 5)

    out: dict[str, list[tuple[int, int]]] = {
        "top": trace_polyline((0, c_top), [(1, 0, top_drop), (0, 1, random.randint(2, 3)), (1, 0, top_tail)]),
        "middle": trace_polyline(
            (mid_r0, c_mid),
            [(1, 0, random.randint(1, 2)), (0, 1, random.randint(2, 3)), (1, 0, random.randint(2, 3))],
        ),
        "bottom": trace_polyline(
            (bottom_r0, c_bottom),
            [(1, 0, bottom_drop_0), (0, 1, random.randint(2, 3)), (1, 0, bottom_drop_1)],
        ),
        "both": trace_polyline(
            (0, c_both),
            [(1, 0, both_r1), (0, 1, random.randint(2, 3)), (1, 0, h - 1 - both_r1)],
        ),
    }
    if profile == "A":
        keys = ("top", "middle", "bottom")
    else:
        keys = ("top", "middle", "bottom")
    return {k: out[k] for k in keys}


def build_paths(h: int, w: int, profile: str) -> tuple[np.ndarray, dict[str, list[tuple[int, int]]]] | None:
    polylines = layout_polylines(h, w, profile)
    left_cells: set[tuple[int, int]] = set()
    right_cells: set[tuple[int, int]] = set()
    left_pieces: dict[str, set[tuple[int, int]]] = {}
    right_pieces: dict[str, set[tuple[int, int]]] = {}

    for name, path in polylines.items():
        if len(set(path)) != len(path):
            return None
        if not all(1 <= r < h - 1 and 2 <= c < w - 2 for r, c in path if r not in (0, h - 1)):
            return None
        piece_a, piece_b = wall_sets_for_channel(path)
        for cells in (piece_a, piece_b):
            if not cells:
                return None
            if not all(in_bounds((h, w), r, c) for r, c in cells):
                return None
        mean_a = sum(c for _r, c in piece_a) / len(piece_a)
        mean_b = sum(c for _r, c in piece_b) / len(piece_b)
        if mean_a <= mean_b:
            left_piece, right_piece = piece_a, piece_b
        else:
            left_piece, right_piece = piece_b, piece_a
        left_pieces[name] = left_piece
        right_pieces[name] = right_piece
        left_cells |= left_piece
        right_cells |= right_piece

    add_vertical(left_cells, 1, h)
    add_vertical(right_cells, w - 2, h)
    left_rows = {"top": 1, "both": 5, "middle": h // 2, "bottom": h - 5}
    right_rows = {"top": 3, "both": 8, "middle": h // 2 + 3, "bottom": h - 3}
    for name, cells in left_pieces.items():
        conn_row = max(1, min(h - 2, left_rows[name]))
        anchor = min(cells, key=lambda p: (abs(p[0] - conn_row), p[1], p[0]))
        add_vline(left_cells, anchor[1], anchor[0], conn_row)
        add_hline(left_cells, conn_row, 1, anchor[1])
    for name, cells in right_pieces.items():
        conn_row = max(1, min(h - 2, right_rows[name]))
        anchor = min(cells, key=lambda p: (abs(p[0] - conn_row), -p[1], p[0]))
        add_vline(right_cells, anchor[1], anchor[0], conn_row)
        add_hline(right_cells, conn_row, anchor[1], w - 2)

    if left_cells & right_cells:
        return None
    grid = np.zeros((h, w), dtype=int)
    for r, c in left_cells | right_cells:
        if not in_bounds((h, w), r, c):
            return None
        grid[r, c] = ORANGE

    labels, n = label_mask(grid == ORANGE, connectivity=4)
    if n != 2:
        return None
    return grid, polylines


RING_SIZES = (
    (3, 3),
    (3, 4),
    (4, 3),
    (3, 5),
    (5, 3),
    (4, 4),
    (4, 5),
    (5, 4),
    (5, 5),
    (5, 6),
    (6, 5),
    (5, 7),
    (7, 5),
)

GEN_RING_SIZES = (
    (3, 3),
    (3, 4),
    (4, 3),
    (4, 4),
)


def ring_template(height: int, width: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    obj = set()
    hole = set()
    for r in range(height):
        for c in range(width):
            if r in (0, height - 1) or c in (0, width - 1):
                obj.add((r, c))
            else:
                hole.add((r, c))
    boundary = set()
    for r, c in obj:
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in hole:
                boundary.add((r, c))
                break
    return obj, hole, boundary


def object_adjacency_types(
    channel_labels: np.ndarray,
    seg_by_id: dict[int, dict],
    r0: int,
    c0: int,
    height: int,
    width: int,
) -> tuple[set[str], set[int]]:
    _obj_cells, _hole_cells, boundary = ring_template(height, width)
    ids = set()
    for rr, cc in boundary:
        gr = r0 + rr
        gc = c0 + cc
        for nr, nc in neighbors4(gr, gc):
            if in_bounds(channel_labels.shape, nr, nc) and channel_labels[nr, nc] > 0:
                ids.add(int(channel_labels[nr, nc]))
    return {seg_by_id[cid]["kind"] for cid in ids}, ids


def can_place_object(
    grid: np.ndarray,
    used: list[tuple[int, int, int, int]],
    r0: int,
    c0: int,
    height: int,
    width: int,
) -> bool:
    h, w = grid.shape
    if r0 < 1 or c0 < 1 or r0 + height >= h - 1 or c0 + width >= w - 1:
        return False
    obj_cells, hole_cells, _boundary = ring_template(height, width)
    for rr, cc in obj_cells | hole_cells:
        if grid[r0 + rr, c0 + cc] != BLACK:
            return False
    for ur, uc, uh, uw in used:
        if not (r0 + height + 1 <= ur or ur + uh + 1 <= r0 or c0 + width + 1 <= uc or uc + uw + 1 <= c0):
            return False
    return True


def find_candidates(
    grid: np.ndarray,
    channel_labels: np.ndarray,
    segs: list[dict],
    intent: str,
    used: list[tuple[int, int, int, int]],
    ring_sizes: tuple[tuple[int, int], ...] = RING_SIZES,
) -> list[tuple[int, int, int, int, set[int]]]:
    seg_by_id = {seg["id"]: seg for seg in segs}
    out = []
    for height, width in ring_sizes:
        for r0 in range(grid.shape[0] - height + 1):
            for c0 in range(grid.shape[1] - width + 1):
                if not can_place_object(grid, used, r0, c0, height, width):
                    continue
                kinds, ids = object_adjacency_types(channel_labels, seg_by_id, r0, c0, height, width)
                if not ids:
                    continue
                ok = False
                if intent == "red_top":
                    ok = kinds == {"top"}
                elif intent == "blue_bottom":
                    ok = kinds == {"bottom"}
                elif intent == "grey_both":
                    ok = kinds == {"both"}
                elif intent == "black_middle":
                    ok = kinds == {"middle"}
                elif intent == "red_mixed":
                    ok = "both" not in kinds and "top" in kinds and "bottom" in kinds
                if ok:
                    out.append((r0, c0, height, width, ids))
    random.shuffle(out)
    out.sort(key=lambda item: (item[2] * item[3], item[0], item[1]))
    return out


def paint_object(grid: np.ndarray, r0: int, c0: int, height: int, width: int, color: int) -> set[tuple[int, int]]:
    obj_cells, hole_cells, _boundary = ring_template(height, width)
    for rr, cc in obj_cells:
        grid[r0 + rr, c0 + cc] = color
    return {(r0 + rr, c0 + cc) for rr, cc in hole_cells}


def expected_fill_for_intent(intent: str) -> int:
    if intent in ("red_top", "red_mixed"):
        return RED
    if intent == "blue_bottom":
        return BLUE
    if intent == "grey_both":
        return GREY
    return BLACK


def verify_semantics(inp: np.ndarray, placements: list[dict], profile: str) -> bool:
    out = main(inp)
    channel_labels, segs = channel_segments(inp)
    seg_kinds = Counter(seg["kind"] for seg in segs)
    required = {
        "A": {"top": 1, "bottom": 1, "middle": 1},
    }[profile]
    if any(seg_kinds.get(kind, 0) < cnt for kind, cnt in required.items()):
        return False
    if np.any(out == ORANGE):
        return False
    if np.array_equal(inp, out):
        return False

    for info in placements:
        expected = expected_fill_for_intent(info["intent"])
        actual = {int(out[r, c]) for r, c in info["hole_cells"]}
        if expected == BLACK:
            if actual != {BLACK}:
                return False
        else:
            if actual != {expected}:
                return False
        if not info["adjacent_ids"]:
            return False
        kinds = {next(seg["kind"] for seg in segs if seg["id"] == cid) for cid in info["adjacent_ids"]}
        if info["intent"] == "red_top" and kinds != {"top"}:
            return False
        if info["intent"] == "blue_bottom" and kinds != {"bottom"}:
            return False
        if info["intent"] == "grey_both" and kinds != {"both"}:
            return False
        if info["intent"] == "black_middle" and kinds != {"middle"}:
            return False
        if info["intent"] == "red_mixed" and not ("top" in kinds and "bottom" in kinds and "both" not in kinds):
            return False
    return True


PROFILE_INTENTS = {
    "A": ("red_top", "blue_bottom", "black_middle"),
}


def separated_boxes(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar, ac, ah, aw = a
    br, bc, bh, bw = b
    return ar + ah + 1 <= br or br + bh + 1 <= ar or ac + aw + 1 <= bc or bc + bw + 1 <= ac


def choose_nonoverlapping(
    candidate_lists: list[list[tuple[int, int, int, int, set[int]]]],
) -> list[tuple[int, int, int, int, set[int]]] | None:
    order = sorted(range(len(candidate_lists)), key=lambda i: len(candidate_lists[i]))
    chosen: list[tuple[int, tuple[int, int, int, int, set[int]]]] = []

    def rec(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        candidates = candidate_lists[idx][:]
        random.shuffle(candidates)
        for cand in candidates[:80]:
            box = cand[:4]
            if all(separated_boxes(box, other[:4]) for _j, other in chosen):
                chosen.append((idx, cand))
                if rec(pos + 1):
                    return True
                chosen.pop()
        return False

    if not rec(0):
        return None
    out = [None] * len(candidate_lists)
    for idx, cand in chosen:
        out[idx] = cand
    return out


def generate_input() -> np.ndarray:
    for _ in range(120):
        h = 24
        w = random.choice([28, 29])
        profile = "A"
        built = build_paths(h, w, profile)
        if built is None:
            continue
        grid, _polylines = built

        try:
            channel_labels, segs = channel_segments(grid)
        except AssertionError:
            continue

        required = PROFILE_INTENTS[profile]
        colors = random.sample([GREEN, YELLOW, PINK, TEAL, MAROON], len(required))
        candidate_lists = [
            find_candidates(grid, channel_labels, segs, intent, [], GEN_RING_SIZES)
            for intent in required
        ]
        if any(not candidates for candidates in candidate_lists):
            continue
        selected = choose_nonoverlapping(candidate_lists)
        if selected is None:
            continue

        placements = []
        work = grid.copy()
        for intent, color, cand in zip(required, colors, selected):
            r0, c0, height, width, ids = cand
            hole_cells = paint_object(work, r0, c0, height, width, color)
            placements.append(
                {
                    "intent": intent,
                    "hole_cells": hole_cells,
                    "adjacent_ids": set(ids),
                }
            )
        if verify_semantics(work, placements, profile):
            return work
    raise RuntimeError("failed to generate a valid input")
