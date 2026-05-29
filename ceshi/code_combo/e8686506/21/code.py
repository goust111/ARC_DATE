# Combo variant of e8686506 (index 21)
# combo_spec: D1/1+D2/1
# Name: 连接器标记匹配+碎片带锚点

import random
from collections import Counter

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis
# ---------------
# The original task assembled monochrome fragments into an LR-symmetric rectangle.
# This combo replaces two assumptions at once:
# 1. Pieces are no longer monochrome. Every movable piece is a multicolor 4-connected
#    sprite that contains one unique anchor pixel and several seam connector pixels.
# 2. Absolute placement is no longer determined by shape alone. Singleton markers
#    inside the frame specify where each piece anchor must land, while seam connectors
#    specify which pieces may touch across horizontal seams.
#
# Concretely, the generated output uses one fixed LR-symmetric frame plus three interior
# 3x3 bands. The top and bottom bands intentionally share the same anchor color, so
# anchor markers alone do not tell them apart. Their orientation is instead determined
# by connector placement: the top band only exposes connectors on its bottom row, the
# bottom band only exposes connectors on its top row, and the middle band exposes them
# on both. This makes the two dimensions genuinely cooperate:
# - anchor markers choose the candidate band positions,
# - connector matching resolves which candidate is legal.
#
# Visual cleanliness is preserved by keeping connector colors sparse and dedicated.
# The frame remains a single easy-to-see anchor object, while markers are isolated
# singleton dots inside the frame interior.


OUT_H = 11
OUT_W = 5

CANVAS_MIN = 14
CANVAS_MAX = 16
SCATTER_PADDING = 1
GEN_MAX_RETRIES = 300


def _factor_dims(area: int) -> tuple[int, int]:
    dims = [(area // w, w) for w in range(1, area + 1) if area % w == 0 and area // w >= w]
    best_gap = min(abs(h - w) for h, w in dims)
    best = [(h, w) for h, w in dims if abs(h - w) == best_gap]
    assert len(best) == 1, "closest-to-square portrait output shape must be unique"
    return best[0]


def _coords_key(pts: np.ndarray) -> frozenset[tuple[int, int]]:
    return frozenset((int(r), int(c)) for r, c in pts)


def _component_sprite(grid: np.ndarray, pts: np.ndarray, bg: int) -> dict:
    r0, c0, r1, c1 = bbox_coords(pts)
    cells = []
    counts: Counter[int] = Counter()
    for r, c in pts:
        rr = int(r) - int(r0)
        cc = int(c) - int(c0)
        col = int(grid[int(r), int(c)])
        cells.append((rr, cc, col))
        counts[col] += 1
    cells.sort()
    anchor_candidates = [
        (rr, cc, col)
        for rr, cc, col in cells
        if 0 < int(rr) < int(r1 - r0) and 0 < int(cc) < int(c1 - c0)
    ]
    assert len(anchor_candidates) == 1, "each movable piece must have exactly one interior anchor pixel"
    anchor_rr, anchor_cc, anchor_color = anchor_candidates[0]
    anchor = (int(anchor_rr), int(anchor_cc))
    body_neighbors = {
        int(col)
        for rr, cc, col in cells
        if int(rr) == int(anchor[0]) and abs(int(cc) - int(anchor[1])) == 1
    }
    assert len(body_neighbors) == 1, "anchor should have a unique body color on its horizontal sides"
    body_color = next(iter(body_neighbors))
    connectors = {
        (rr, cc)
        for rr, cc, col in cells
        if (rr, cc) != anchor and col != body_color
    }
    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "cells": tuple(cells),
        "h": int(r1 - r0 + 1),
        "w": int(c1 - c0 + 1),
        "anchor": (int(anchor[0]), int(anchor[1])),
        "anchor_color": anchor_color,
        "body_color": int(body_color),
        "connectors": connectors,
    }


def _frame_sprite(grid: np.ndarray, pts: np.ndarray) -> dict:
    r0, c0, r1, c1 = bbox_coords(pts)
    cells = []
    for r, c in pts:
        rr = int(r) - int(r0)
        cc = int(c) - int(c0)
        col = int(grid[int(r), int(c)])
        cells.append((rr, cc, col))
    cells.sort()
    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "cells": tuple(cells),
        "h": int(r1 - r0 + 1),
        "w": int(c1 - c0 + 1),
        "anchor": (0, 0),
        "anchor_color": -1,
        "body_color": -1,
        "connectors": set(),
    }


def _is_lr_symmetric(parts: list[np.ndarray], H: int, W: int) -> bool:
    union = np.concatenate(parts)
    r0, c0, _r1, _c1 = bbox_coords(union)
    mask = np.zeros((H, W), dtype=bool)
    for pts in parts:
        for r, c in pts:
            mask[int(r) - int(r0), int(c) - int(c0)] = True
    return np.array_equal(mask, mask[:, ::-1])


def _find_frame_group(piece_comps: list[np.ndarray], H: int, W: int) -> list[np.ndarray]:
    found = []
    for i, p1 in enumerate(piece_comps):
        r0, c0, r1, c1 = bbox_coords(p1)
        if int(r1 - r0 + 1) == H and int(c1 - c0 + 1) == W and _is_lr_symmetric([p1], H, W):
            found.append([p1])
        for j in range(i + 1, len(piece_comps)):
            p2 = piece_comps[j]
            union = np.concatenate([p1, p2])
            ur0, uc0, ur1, uc1 = bbox_coords(union)
            if int(ur1 - ur0 + 1) == H and int(uc1 - uc0 + 1) == W and _is_lr_symmetric([p1, p2], H, W):
                found.append([p1, p2])
    assert len(found) == 1, "frame pieces must be uniquely identifiable"
    return found[0]


def _assemble(
    pieces: list[dict],
    frame_ids: set[int],
    frame_offsets: dict[int, tuple[int, int]],
    anchor_targets: dict[int, set[tuple[int, int]]],
    H: int,
    W: int,
) -> np.ndarray:
    board = np.full((H, W), -1, dtype=int)
    owner = np.full((H, W), -1, dtype=int)
    is_connector = np.zeros((H, W), dtype=bool)

    def place_piece(pid: int, r0: int, c0: int) -> list[tuple[int, int]]:
        sprite = pieces[pid]
        changed = []
        for dr, dc, col in sprite["cells"]:
            r, c = int(r0 + dr), int(c0 + dc)
            board[r, c] = int(col)
            owner[r, c] = int(pid)
            is_connector[r, c] = (int(dr), int(dc)) in sprite["connectors"]
            changed.append((r, c))
        return changed

    def unplace_piece(changed: list[tuple[int, int]]) -> None:
        for r, c in changed:
            board[r, c] = -1
            owner[r, c] = -1
            is_connector[r, c] = False

    def can_place(pid: int, r0: int, c0: int) -> bool:
        sprite = pieces[pid]
        local = {}
        for dr, dc, col in sprite["cells"]:
            r, c = int(r0 + dr), int(c0 + dc)
            if not (0 <= r < H and 0 <= c < W):
                return False
            if owner[r, c] != -1:
                return False
            local[(r, c)] = int(col)

        ar = int(r0 + sprite["anchor"][0])
        ac = int(c0 + sprite["anchor"][1])
        if (ar, ac) not in anchor_targets.get(int(sprite["anchor_color"]), set()):
            return False

        touched_non_frame = False
        matched_connector = False
        for (r, c), col in local.items():
            mc = W - 1 - int(c)
            if (r, mc) in local and local[(r, mc)] != col:
                return False
            if owner[r, mc] != -1 and int(board[r, mc]) != int(col):
                return False

            self_connector = (r - r0, c - c0) in sprite["connectors"]
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if not (0 <= rr < H and 0 <= cc < W):
                    continue
                if (rr, cc) in local:
                    continue
                nid = int(owner[rr, cc])
                if nid == -1:
                    continue
                if nid in frame_ids:
                    continue
                touched_non_frame = True
                if not self_connector:
                    return False
                if not bool(is_connector[rr, cc]):
                    return False
                if int(board[rr, cc]) != int(col):
                    return False
                matched_connector = True

        if touched_non_frame and not matched_connector:
            return False
        return True

    for pid in sorted(frame_ids):
        r0, c0 = frame_offsets[pid]
        place_piece(pid, r0, c0)

    movable = [pid for pid in range(len(pieces)) if pid not in frame_ids]

    def candidate_positions(pid: int) -> list[tuple[int, int]]:
        sprite = pieces[pid]
        out = []
        for tr, tc in sorted(anchor_targets[int(sprite["anchor_color"])]):
            r0 = int(tr - sprite["anchor"][0])
            c0 = int(tc - sprite["anchor"][1])
            if can_place(pid, r0, c0):
                out.append((r0, c0))
        return out

    def dfs(remaining: list[int]) -> bool:
        if not remaining:
            return np.array_equal(board, board[:, ::-1])

        best_pid = None
        best_places = None
        for pid in remaining:
            places = candidate_positions(pid)
            if not places:
                return False
            if best_places is None or len(places) < len(best_places):
                best_pid = pid
                best_places = places

        assert best_pid is not None and best_places is not None
        sprite = pieces[best_pid]
        next_remaining = [pid for pid in remaining if pid != best_pid]
        for r0, c0 in best_places:
            changed = place_piece(best_pid, r0, c0)
            ar = int(r0 + sprite["anchor"][0])
            ac = int(c0 + sprite["anchor"][1])
            anchor_targets[int(sprite["anchor_color"])].remove((ar, ac))
            if dfs(next_remaining):
                return True
            anchor_targets[int(sprite["anchor_color"])].add((ar, ac))
            unplace_piece(changed)
        return False

    assert dfs(movable), "assembly failed"
    return board.astype(int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble anchored multicolor fragments into a left-right symmetric rectangle.

    Input structure:
    - The background is the most common color.
    - One foreground component is an in-place LR-symmetric frame whose bbox already matches
      the output rectangle.
    - The remaining movable pieces are multicolor 4-connected sprites with one unique anchor
      pixel and several connector pixels; singleton foreground dots inside the frame are the
      target anchor markers.

    Rule:
    - Ignore singleton foreground dots when computing the rectangle area.
    - Keep the frame fixed, then place each movable piece so its unique anchor pixel lands on
      a same-colored target marker.
    - Whenever two movable pieces touch, every contact must occur through equal-colored
      connector pixels; ordinary body cells may touch the frame but not another movable piece.

    Key insight:
    The singleton markers provide absolute coordinates, but they do not completely disambiguate
    the top and bottom pieces because those two share the same anchor color. Their orientation
    is resolved by the connector rule: only the piece whose exposed seam connectors match the
    already placed neighbor can occupy that marker.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    comps = components_mask(g != bg, connectivity=4)
    markers = [pts for pts in comps if len(pts) == 1]
    piece_comps = [pts for pts in comps if len(pts) > 1]
    assert piece_comps, "input must contain real pieces"

    total_area = sum(len(pts) for pts in piece_comps)
    H, W = _factor_dims(int(total_area))
    frame_parts = _find_frame_group(piece_comps, H, W)
    frame_keys = {_coords_key(pts) for pts in frame_parts}

    frame_union = np.concatenate(frame_parts)
    fr0, fc0, _fr1, _fc1 = bbox_coords(frame_union)

    pieces = []
    frame_ids = set()
    frame_offsets = {}
    for pts in piece_comps:
        pid = len(pieces)
        if _coords_key(pts) in frame_keys:
            sprite = _frame_sprite(g, pts)
            frame_ids.add(pid)
            pr0, pc0, _pr1, _pc1 = sprite["bbox"]
            frame_offsets[pid] = (int(pr0 - fr0), int(pc0 - fc0))
        else:
            sprite = _component_sprite(g, pts, bg)
        pieces.append(sprite)

    anchor_targets: dict[int, set[tuple[int, int]]] = {}
    for pts in markers:
        r = int(pts[0, 0]) - int(fr0)
        c = int(pts[0, 1]) - int(fc0)
        assert 0 <= r < H and 0 <= c < W, "anchor markers must lie inside the frame bbox"
        col = int(g[int(pts[0, 0]), int(pts[0, 1])])
        anchor_targets.setdefault(col, set()).add((r, c))

    movable_count = len(pieces) - len(frame_ids)
    assert sum(len(v) for v in anchor_targets.values()) == movable_count, "each movable piece needs one target marker"

    out = _assemble(pieces, frame_ids, frame_offsets, anchor_targets, H, W)
    return out


def _build_output() -> tuple[np.ndarray, int, dict]:
    colors = list(range(10))
    random.shuffle(colors)
    bg = colors.pop()
    frame = colors.pop()
    body_outer = colors.pop()
    body_mid = colors.pop()
    anchor_outer = colors.pop()
    anchor_mid = colors.pop()
    conn_side = colors.pop()
    conn_center = colors.pop()

    out = np.full((OUT_H, OUT_W), frame, dtype=int)

    top = np.array(
        [
            [body_outer, body_outer, body_outer],
            [body_outer, anchor_outer, body_outer],
            [conn_side, conn_center, conn_side],
        ],
        dtype=int,
    )
    mid = np.array(
        [
            [conn_side, conn_center, conn_side],
            [body_mid, anchor_mid, body_mid],
            [conn_side, conn_center, conn_side],
        ],
        dtype=int,
    )
    bot = np.array(
        [
            [conn_side, conn_center, conn_side],
            [body_outer, anchor_outer, body_outer],
            [body_outer, body_outer, body_outer],
        ],
        dtype=int,
    )

    out[1:4, 1:4] = top
    out[4:7, 1:4] = mid
    out[7:10, 1:4] = bot

    markers = {
        "top": (2, 2, int(anchor_outer)),
        "mid": (5, 2, int(anchor_mid)),
        "bot": (8, 2, int(anchor_outer)),
    }
    meta = {
        "bg": int(bg),
        "markers": markers,
    }
    return out.astype(int), int(bg), meta


def _extract_patch(out: np.ndarray, rows: slice, cols: slice) -> np.ndarray:
    return np.array(out[rows, cols], copy=True)


def _scatter(out: np.ndarray, bg: int, meta: dict) -> np.ndarray:
    rows = random.randint(CANVAS_MIN, CANVAS_MAX)
    cols = random.randint(CANVAS_MIN, CANVAS_MAX)
    canvas = np.full((rows, cols), bg, dtype=int)

    max_r = rows - OUT_H - 2
    max_c = cols - OUT_W - 2
    if max_r < 1 or max_c < 1:
        raise ValueError("canvas too small")

    off_r = random.randint(1, max_r)
    off_c = random.randint(1, max_c)
    canvas[off_r : off_r + OUT_H, off_c : off_c + OUT_W] = -1

    patches = [
        _extract_patch(out, slice(1, 4), slice(1, 4)),
        _extract_patch(out, slice(4, 7), slice(1, 4)),
        _extract_patch(out, slice(7, 10), slice(1, 4)),
    ]
    random.shuffle(patches)
    for patch in patches:
        r, c = random_free_location(
            canvas,
            patch,
            transparent=bg,
            padding=SCATTER_PADDING,
            padding_connectivity=8,
        )
        paste(canvas, patch, r, c, transparent=bg)

    bbox = canvas[off_r : off_r + OUT_H, off_c : off_c + OUT_W]
    bbox[:] = bg

    frame_mask = np.zeros((OUT_H, OUT_W), dtype=bool)
    frame_mask[0, :] = True
    frame_mask[-1, :] = True
    frame_mask[:, 0] = True
    frame_mask[:, -1] = True
    bbox[frame_mask] = out[frame_mask]

    for r, c, col in meta["markers"].values():
        bbox[int(r), int(c)] = int(col)
    return canvas.astype(int)


def _fusion_metrics(grid: np.ndarray) -> dict:
    bg = int(get_bg(grid))
    comps = components_mask(grid != bg, connectivity=4)
    piece_comps = [pts for pts in comps if len(pts) > 1]
    markers = [pts for pts in comps if len(pts) == 1]
    frame_parts = _find_frame_group(piece_comps, * _factor_dims(sum(len(pts) for pts in piece_comps)))
    frame_keys = {_coords_key(pts) for pts in frame_parts}
    multi = [_component_sprite(grid, pts, bg) for pts in piece_comps if _coords_key(pts) not in frame_keys]
    shared_anchor_targets = sum(
        1
        for pts in markers
        if int(grid[int(pts[0, 0]), int(pts[0, 1])]) == multi[0]["anchor_color"]
    )
    connector_cells = sum(len(piece["connectors"]) for piece in multi if len(piece["connectors"]) > 0)
    return {
        "piece_count": len(piece_comps),
        "marker_count": len(markers),
        "shared_anchor_targets": int(shared_anchor_targets),
        "connector_cells": int(connector_cells),
    }


def _gen_attempt(return_meta: bool = False):
    out, bg, meta = _build_output()
    try:
        inp = _scatter(out, bg, meta)
        got = main(inp)
    except Exception:
        return None
    if not np.array_equal(got, out):
        return None
    if return_meta:
        return inp.astype(int), _fusion_metrics(inp)
    return inp.astype(int)


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid connector+anchor e8686506 input")
