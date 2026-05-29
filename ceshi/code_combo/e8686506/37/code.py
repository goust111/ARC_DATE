# Combo variant of e8686506 (index 37)
# combo_spec: D7/2+D11/1
# Name: 分离面板输出+对称修补

import random
from collections import Counter

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the combo variant:
# - Concreteized behavior:
#   The scattered input is still solved by first rebuilding one dense HxW rectangle from all
#   non-background 4-connected fragments. Unlike the base task, the rebuilt rectangle may contain
#   sparse left/right disagreements introduced as isolated wrong-color pixels. Those disagreements
#   are repaired first using the D11 rule: compare local same-color support in each mirrored pair,
#   copy the supported side, otherwise clear the pair to background. Only after that do we apply
#   the D7 change: instead of returning the repaired rectangle itself, output one HxW panel per
#   placed fragment, with that fragment shown alone at its assembled coordinates inside the panel.
#   Example trace on a 5x5 rectangle: if cells (1,1)/(1,3) are (7,2) and the 2-side touches more
#   same-color neighbors, repair yields (2,2); the final output does not merge everything, but
#   shows the left fragment on one panel, the right fragment on another, each in the repaired view.
# - Complexity calibration:
#   The combo adds exactly two visible steps beyond the original: symmetry repair, then panel
#   separation. Both steps repeat across many cells/pieces in the same sample, so 2-4 examples are
#   enough to see the pattern. We keep panel borders uniform (background gaps) and use one fixed
#   repair heuristic to avoid a combinatorial rule explosion.
# - Implicit assumption audit:
#   The original solver assumed direct LR color symmetry during placement; that is invalid once
#   damage is injected, so assembly must ignore mirror colors and solve a dense rectangle first.
#   The original compact-crop output assumption is also invalid; after repair we must preserve the
#   assembled coordinates of every placed fragment and package them into separate full-size panels.
#   Frame anchoring, total-area factorization, and no-same-color-touch constraints still remain
#   valid and are preserved by the generator.
# - Placement legality:
#   Input scattering is unchanged: non-frame patches are placed outside the blocked frame bbox with
#   Chebyshev padding, and the intact frame group is pasted back into that bbox. Panel rendering is
#   also legal by construction because every panel is exactly HxW and panels are packed with 1-cell
#   background separators and a 1-cell outer border, so there is no overlap.
# - Visual observability:
#   The output now visibly differs from both source variants: examples show many same-sized panels
#   rather than one rectangle, and the panels contain repaired rather than raw damaged fragments.
#   A human can infer the rule because the panel shapes still line up to one common rectangle, and
#   the repaired mirror pairs are readable from repeated isolated mistakes in the inputs.
# - Visual cleanliness:
#   Damage remains sparse and never touches the frame, so the scaffold is legible. Each panel holds
#   just one fragment on a blank HxW canvas, which keeps overlaps impossible and makes the assembly
#   coordinates easy to compare across panels.
# - Color inference:
#   Wrong colors are only used as unsupported anomalies; they are never remapped by hidden numeric
#   rules. They disappear through support-based repair or are overwritten by the supported mirror,
#   so the observable cue is structural adjacency, not color-id arithmetic.


def _piece_sig(pts: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = bbox_coords(pts)
    coords = tuple(sorted((int(r) - r0, int(c) - c0) for r, c in pts))
    return (r1 - r0 + 1), (c1 - c0 + 1), coords


def _factors(n: int) -> list[tuple[int, int]]:
    out = []
    for w in range(1, n + 1):
        if n % w == 0:
            out.append((n // w, w))
    return out


def _build_requirements(
    H: int,
    W: int,
    comps_by_color: dict[int, list[list[tuple[int, int]]]],
) -> np.ndarray:
    req = np.full((H, W), -1, dtype=int)

    corner_colors = [
        col
        for col, comps in comps_by_color.items()
        if len(comps) == 4 and all(len(p) == 1 for p in comps)
    ]
    assert len(corner_colors) <= 1, "corner singleton color must be unique when present"
    corner_color = corner_colors[0] if corner_colors else None
    if corner_color is not None:
        req[0, 0] = corner_color
        req[0, W - 1] = corner_color
        req[H - 1, 0] = corner_color
        req[H - 1, W - 1] = corner_color

    bar_colors: list[tuple[int, int]] = []
    if W >= 3 and corner_color is not None:
        for col, comps in comps_by_color.items():
            for pts in comps:
                r0, c0, r1, c1 = bbox_coords(pts)
                hh, ww = r1 - r0 + 1, c1 - c0 + 1
                if hh == 1 and ww == W - 2 and len(pts) == ww:
                    bar_colors.append((r0, col))
                    break
        if bar_colors:
            assert len(bar_colors) <= 2, "horizontal edge bar colors must be top/bottom only"
            bar_colors.sort()
            req[0, 1 : W - 1] = bar_colors[0][1]
            req[H - 1, 1 : W - 1] = bar_colors[-1][1]

    vertical_line_candidates = []
    for col, comps in comps_by_color.items():
        if len(comps) != 2:
            continue
        sigs = [_piece_sig(pts) for pts in comps]
        (h0, w0, coords0), (h1, w1, coords1) = sigs[0], sigs[1]
        if w0 == 1 and w1 == 1 and h0 == h1 and coords0 == coords1 and h0 > 1:
            vertical_line_candidates.append((col, h0))
    assert len(vertical_line_candidates) <= 1, "paired vertical edge-line color must be unique when present"
    if vertical_line_candidates:
        col, L = vertical_line_candidates[0]
        top = (H - L) // 2
        assert 0 <= top and top + L <= H, "paired vertical edge lines must fit in the output height"
        req[top : top + L, 0] = col
        req[top : top + L, W - 1] = col

    return req


def _assemble(
    types: list[dict],
    H: int,
    W: int,
    req: np.ndarray,
    disallow_touch: set[int] | None = None,
) -> tuple[np.ndarray, list[dict]]:
    board = np.full((H, W), -1, dtype=int)
    pieces: list[dict] = []
    for t in types:
        for _ in range(int(t["cnt"])):
            pieces.append(
                {
                    "col": int(t["col"]),
                    "coords": tuple(t["coords"]),
                    "h": int(t["h"]),
                    "w": int(t["w"]),
                    "area": len(t["coords"]),
                }
            )
    pieces.sort(key=lambda p: (-p["area"], p["col"], p["coords"]))

    def can_place(piece: dict, r0: int, c0: int) -> bool:
        col = int(piece["col"])
        is_singleton = len(piece["coords"]) == 1
        for dr, dc in piece["coords"]:
            r, c = r0 + dr, c0 + dc
            if board[r, c] != -1:
                return False
            if req[r, c] != -1 and int(req[r, c]) != col:
                return False
            if is_singleton and not (r == 0 or r == H - 1 or c == 0 or c == W - 1):
                return False
            if disallow_touch and col in disallow_touch:
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= rr < H and 0 <= cc < W and board[rr, cc] == col:
                        return False
        return True

    def placements_for(piece: dict) -> list[tuple[int, int]]:
        return [
            (r0, c0)
            for r0 in range(H - int(piece["h"]) + 1)
            for c0 in range(W - int(piece["w"]) + 1)
            if can_place(piece, r0, c0)
        ]

    def place(piece: dict, r0: int, c0: int) -> tuple[dict, list[tuple[int, int]]]:
        changed = []
        col = int(piece["col"])
        for dr, dc in piece["coords"]:
            r, c = r0 + dr, c0 + dc
            board[r, c] = col
            changed.append((r, c))
        entry = {
            "col": col,
            "coords": tuple(piece["coords"]),
            "r0": int(r0),
            "c0": int(c0),
            "h": int(piece["h"]),
            "w": int(piece["w"]),
            "cells": tuple(sorted(changed)),
        }
        return entry, changed

    def unplace(changed: list[tuple[int, int]]) -> None:
        for r, c in changed:
            board[r, c] = -1

    solution: list[dict] | None = None

    def search(remaining: list[dict], placed: list[dict]) -> bool:
        nonlocal solution
        forced: list[tuple[int, dict, dict, list[tuple[int, int]]]] = []

        while True:
            if not remaining:
                solution = [dict(p) for p in placed]
                return True

            best_idx = -1
            best_places: list[tuple[int, int]] | None = None
            for idx, piece in enumerate(remaining):
                places = placements_for(piece)
                if not places:
                    while forced:
                        restore_idx, restore_piece, _entry, changed = forced.pop()
                        placed.pop()
                        unplace(changed)
                        remaining.insert(restore_idx, restore_piece)
                    return False
                if best_places is None or len(places) < len(best_places):
                    best_idx = idx
                    best_places = places
                    if len(best_places) == 1:
                        break

            assert best_places is not None
            if len(best_places) != 1:
                break

            piece = remaining.pop(best_idx)
            entry, changed = place(piece, best_places[0][0], best_places[0][1])
            placed.append(entry)
            forced.append((best_idx, piece, entry, changed))

        piece = remaining.pop(best_idx)
        for r0, c0 in best_places:
            entry, changed = place(piece, r0, c0)
            placed.append(entry)
            if search(remaining, placed):
                return True
            placed.pop()
            unplace(changed)
        remaining.insert(best_idx, piece)

        while forced:
            restore_idx, restore_piece, _entry, changed = forced.pop()
            placed.pop()
            unplace(changed)
            remaining.insert(restore_idx, restore_piece)
        return False

    ok = search(pieces[:], [])
    assert ok and solution is not None, "assembly search failed"
    assert np.all(board != -1), "assembled board must be dense before repair"
    return board, solution


def _support_score(board: np.ndarray, r: int, c: int, bg: int) -> int:
    col = int(board[r, c])
    if col == bg:
        return -1
    H, W = board.shape
    score = 0
    for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if 0 <= rr < H and 0 <= cc < W and int(board[rr, cc]) == col:
            score += 1
    return score


def _repair_symmetry(board: np.ndarray, bg: int) -> np.ndarray:
    H, W = board.shape
    out = board.copy()
    for r in range(H):
        for c in range(W // 2):
            mc = W - 1 - c
            a = int(board[r, c])
            b = int(board[r, mc])
            if a == b:
                out[r, c] = a
                out[r, mc] = b
                continue
            sa = _support_score(board, r, c, bg)
            sb = _support_score(board, r, mc, bg)
            if sa > sb and sa > 0:
                out[r, c] = a
                out[r, mc] = a
            elif sb > sa and sb > 0:
                out[r, c] = b
                out[r, mc] = b
            else:
                out[r, c] = bg
                out[r, mc] = bg
    return out


def _repair_event_counts(board: np.ndarray, bg: int) -> dict[str, int]:
    H, W = board.shape
    copied = 0
    cleared = 0
    unchanged_mismatch = 0
    for r in range(H):
        for c in range(W // 2):
            mc = W - 1 - c
            a = int(board[r, c])
            b = int(board[r, mc])
            if a == b:
                continue
            sa = _support_score(board, r, c, bg)
            sb = _support_score(board, r, mc, bg)
            if sa > sb and sa > 0:
                copied += 1
            elif sb > sa and sb > 0:
                copied += 1
            else:
                cleared += 1
    repaired = _repair_symmetry(board, bg)
    for r in range(H):
        for c in range(W // 2):
            mc = W - 1 - c
            if int(board[r, c]) != int(board[r, mc]) and int(repaired[r, c]) != int(repaired[r, mc]):
                unchanged_mismatch += 1
    return {"copy_pairs": copied, "clear_pairs": cleared, "bad_pairs_after_repair": unchanged_mismatch}


def _find_frame_parts(g: np.ndarray, bg: int, H: int, W: int) -> list[tuple[int, np.ndarray]]:
    comps8_all = []
    for col in sorted(set(map(int, np.unique(g))) - {bg}):
        for pts in components_mask(g == col, connectivity=8):
            comps8_all.append((int(col), pts))

    def is_lr_symmetric(parts: list[tuple[int, np.ndarray]]) -> bool:
        all_pts = np.concatenate([p for _, p in parts])
        r0, c0 = int(all_pts[:, 0].min()), int(all_pts[:, 1].min())
        bmp = np.zeros((H, W), dtype=bool)
        for _col, pts in parts:
            for r, c in pts:
                bmp[int(r) - r0, int(c) - c0] = True
        return np.array_equal(bmp, bmp[:, ::-1])

    frame_parts: list[tuple[int, np.ndarray]] = []
    for i, (col1, pts1) in enumerate(comps8_all):
        r0, c0, r1, c1 = bbox_coords(pts1)
        if r1 - r0 + 1 == H and c1 - c0 + 1 == W and is_lr_symmetric([(col1, pts1)]):
            frame_parts = [(col1, pts1)]
            break
        for j in range(i + 1, len(comps8_all)):
            col2, pts2 = comps8_all[j]
            union = np.concatenate([pts1, pts2])
            ur0, uc0, ur1, uc1 = bbox_coords(union)
            if ur1 - ur0 + 1 == H and uc1 - uc0 + 1 == W and is_lr_symmetric([(col1, pts1), (col2, pts2)]):
                frame_parts = [(col1, pts1), (col2, pts2)]
                break
        if frame_parts:
            break
    assert frame_parts, "input must contain 1-2 LR-symmetric frame parts whose union fixes the output bbox"
    return frame_parts


def _extract_piece_records(board: np.ndarray, bg: int) -> list[dict]:
    records = []
    for col in sorted(set(map(int, np.unique(board))) - {bg}):
        for pts in components_mask(board == col, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            coords = tuple(sorted((int(r) - r0, int(c) - c0) for r, c in pts))
            records.append(
                {
                    "col": int(col),
                    "coords": coords,
                    "r0": int(r0),
                    "c0": int(c0),
                    "h": int(r1 - r0 + 1),
                    "w": int(c1 - c0 + 1),
                    "cells": tuple(sorted((int(r), int(c)) for r, c in pts)),
                }
            )
    return records


def _placement_sort_key(rec: dict) -> tuple:
    return (int(rec["r0"]), int(rec["c0"]), int(rec["col"]), len(rec["coords"]), rec["coords"])


def _choose_panel_layout(count: int, panel_h: int, panel_w: int) -> tuple[int, int]:
    gap = 1
    border = 1
    best: tuple | None = None
    best_layout = (1, count)
    for cols in range(1, count + 1):
        rows = (count + cols - 1) // cols
        out_h = 2 * border + rows * panel_h + (rows - 1) * gap
        out_w = 2 * border + cols * panel_w + (cols - 1) * gap
        key = (abs(out_h - out_w), rows * cols - count, out_h * out_w, rows, cols)
        if best is None or key < best:
            best = key
            best_layout = (rows, cols)
    return best_layout


def _render_separated_panels(board: np.ndarray, placements: list[dict], bg: int) -> np.ndarray:
    ordered = sorted(placements, key=_placement_sort_key)
    H, W = board.shape
    rows, cols = _choose_panel_layout(len(ordered), H, W)
    gap = 1
    border = 1
    out_h = 2 * border + rows * H + (rows - 1) * gap
    out_w = 2 * border + cols * W + (cols - 1) * gap
    out = np.full((out_h, out_w), bg, dtype=int)

    for idx, rec in enumerate(ordered):
        pr = idx // cols
        pc = idx % cols
        rr = border + pr * (H + gap)
        cc = border + pc * (W + gap)
        for r, c in rec["cells"]:
            out[rr + int(r), cc + int(c)] = int(board[int(r), int(c)])
    return out


def _panelized_repaired_output(damaged_board: np.ndarray, placements: list[dict], bg: int) -> np.ndarray:
    repaired = _repair_symmetry(damaged_board, bg)
    assert np.array_equal(repaired, repaired[:, ::-1]), "repair must produce LR symmetry"
    return _render_separated_panels(repaired, placements, bg)


def _expected_output_from_board(damaged_board: np.ndarray, bg: int) -> np.ndarray:
    return _panelized_repaired_output(damaged_board, _extract_piece_records(damaged_board, bg), bg)


def combo_metrics_from_board(damaged_board: np.ndarray, bg: int) -> dict[str, int]:
    placements = _extract_piece_records(damaged_board, bg)
    repaired = _repair_symmetry(damaged_board, bg)
    mismatch_pairs = 0
    H, W = damaged_board.shape
    for r in range(H):
        for c in range(W // 2):
            if int(damaged_board[r, c]) != int(damaged_board[r, W - 1 - c]):
                mismatch_pairs += 1
    panel_out = _render_separated_panels(repaired, placements, bg)
    return {
        "panel_count": len(placements),
        "panel_output_area": int(panel_out.shape[0] * panel_out.shape[1]),
        "rect_output_area": int(repaired.shape[0] * repaired.shape[1]),
        "mismatch_pairs_before_repair": mismatch_pairs,
    }


def _requirements_match_board(board: np.ndarray, bg: int) -> bool:
    H, W = board.shape
    comps_by_color: dict[int, list[list[tuple[int, int]]]] = {}
    for col in sorted(set(map(int, np.unique(board))) - {bg}):
        comps_by_color[int(col)] = components_mask(board == int(col), connectivity=4)
    req = _build_requirements(H, W, comps_by_color)
    for r in range(H):
        for c in range(W):
            if int(req[r, c]) != -1 and int(req[r, c]) != int(board[r, c]):
                return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered fragments, repair mirror disagreements, then show each repaired fragment in its own panel.

    Input structure:
    - The most common color is the background; every other 4-connected component is one scattered puzzle fragment.
    - One or two 8-connected frame groups still preserve the target rectangle's footprint and anchor some cells.
    - A few non-frame mirrored cell pairs inside the future rectangle may be damaged by isolated wrong colors.

    Rule:
    - Collect all non-background fragments, infer the unique portrait rectangle whose area matches them, and pack the fragments back into that dense rectangle using the frame and edge constraints.
    - Repair the assembled rectangle by comparing each left-right mirrored pair: copy the side with stronger local same-color support, otherwise clear the pair to background.
    - For every placed fragment, render a separate full-size HxW panel containing only that fragment's repaired cells at their assembled coordinates, then pack the panels with 1-cell background separators.

    Key insight:
    - The final picture is not the repaired rectangle itself. The solver must use the rectangle only as an intermediate scaffold, then decompose the repaired result back into piece-aligned views that still reveal the assembly positions.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    comps_by_color: dict[int, list[list[tuple[int, int]]]] = {}
    pieces = []
    color_piece_count: dict[int, int] = {}

    for col in sorted(set(map(int, np.unique(g)))):
        if col == bg:
            continue
        comps = components_mask(g == col, connectivity=4)
        comps_by_color[col] = comps
        color_piece_count[col] = len(comps)
        for pts in comps:
            _, _, coords = _piece_sig(pts)
            pieces.append((col, coords))

    ctr = Counter(pieces)
    types = []
    for (col, coords), cnt in sorted(ctr.items(), key=lambda kv: (-len(kv[0][1]), kv[0][0], kv[0][1])):
        arr = np.array(coords, dtype=int)
        types.append(
            {
                "col": int(col),
                "coords": coords,
                "h": int(arr[:, 0].max() + 1),
                "w": int(arr[:, 1].max() + 1),
                "cnt": int(cnt),
            }
        )

    total_area = sum(int(t["cnt"]) * len(t["coords"]) for t in types)
    assert total_area > 0, "grid must contain non-background pieces"

    dims = [(h, w) for h, w in _factors(total_area) if h >= w]
    assert dims, "at least one portrait factor pair must exist"
    best_gap = min(abs(h - w) for h, w in dims)
    best_dims = [(h, w) for h, w in dims if abs(h - w) == best_gap]
    assert len(best_dims) == 1, "closest-to-square portrait output shape must be unique"
    H, W = best_dims[0]

    frame_parts = _find_frame_parts(g, bg, H, W)

    req = _build_requirements(H, W, comps_by_color)
    all_frame_pts = np.concatenate([p for _, p in frame_parts])
    fr0, fc0 = int(all_frame_pts[:, 0].min()), int(all_frame_pts[:, 1].min())
    for col, pts in frame_parts:
        for r, c in pts:
            or_, oc = int(r) - fr0, int(c) - fc0
            if 0 <= or_ < H and 0 <= oc < W:
                req[or_, oc] = int(col)

    disallow_touch = {col for col, n in color_piece_count.items() if n > 1}
    assembled, placements = _assemble(types, H, W, req, disallow_touch)
    return _panelized_repaired_output(assembled, placements, bg).astype(int)


# --------------- generator (combo variant) ---------------

# === CFG: Output dimensions ===
_AREA_MIN = 25
_AREA_MAX = 50
_WIDTH_REQUIRED = 5
_HEIGHT_MAX = 10

# === CFG: Color and palette ===
_N_COLORS_MIN = 4
_N_COLORS_MAX = 6
_TWO_FRAME_MIN_H = 6
_TWO_FRAME_MIN_COLORS = 5
_TWO_FRAME_PROB = 0.35
_CENTER_CAP_PROB = 0.5

# === CFG: Piece constraints ===
_MAX_PIECE_AREA = 6
_MAX_CHUNK_SIZE = 6
_PIECE_COUNT_MIN = 9
_PIECE_COUNT_MAX = 20

# === CFG: Variant damage ===
_FILL_REPAIRS_REQUIRED = 1
_CLEAR_REPAIRS_REQUIRED = 1
_EXTRA_FILL_PROB = 0.35
_PAIR_GAP = 1

# === CFG: Canvas / scatter ===
_CANVAS_MIN = 13
_CANVAS_MAX = 15
_SCATTER_PADDING = 1

# === CFG: Retry ===
_GEN_MAX_RETRIES = 500

_PRECOMPUTED_INPUTS = [
    np.array(
        [
            [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 2, 7, 7, 7, 2, 7, 7],
            [7, 7, 7, 3, 3, 7, 2, 7, 7, 7, 2, 7, 7],
            [7, 7, 3, 3, 3, 7, 7, 2, 7, 2, 7, 7, 7],
            [6, 7, 7, 7, 7, 7, 2, 2, 7, 2, 2, 7, 7],
            [7, 7, 7, 7, 7, 7, 2, 7, 7, 7, 2, 7, 7],
            [7, 7, 7, 8, 7, 7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 0, 7],
            [7, 7, 7, 7, 7, 7, 7, 7, 4, 7, 7, 0, 7],
            [5, 7, 7, 7, 7, 5, 7, 7, 7, 7, 7, 0, 7],
        ],
        dtype=int,
    ),
    np.array(
        [
            [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0, 6, 6],
            [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 6],
            [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            [6, 5, 6, 6, 5, 6, 6, 6, 6, 8, 8, 6, 6, 6],
            [6, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            [6, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            [6, 5, 6, 6, 6, 6, 7, 6, 6, 6, 7, 6, 6, 6],
            [6, 6, 6, 6, 9, 6, 7, 7, 6, 7, 7, 6, 6, 6],
            [6, 6, 6, 6, 6, 6, 7, 7, 6, 7, 7, 6, 6, 6],
            [6, 6, 6, 6, 6, 6, 6, 7, 6, 7, 6, 6, 6, 6],
            [6, 6, 6, 6, 6, 6, 7, 6, 6, 6, 7, 6, 6, 6],
            [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            [6, 6, 6, 6, 6, 6, 6, 6, 5, 6, 6, 6, 6, 6],
        ],
        dtype=int,
    ),
    np.array(
        [
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [8, 6, 8, 8, 8, 6, 8, 9, 8, 2, 2, 8, 8],
            [8, 6, 6, 8, 6, 6, 8, 8, 8, 2, 8, 8, 8],
            [8, 8, 6, 8, 6, 8, 8, 8, 8, 2, 8, 8, 8],
            [8, 8, 6, 8, 6, 8, 8, 8, 8, 2, 8, 8, 8],
            [8, 8, 6, 8, 6, 8, 8, 8, 8, 8, 8, 8, 3],
            [8, 6, 8, 8, 8, 6, 8, 8, 2, 8, 8, 8, 8],
            [8, 8, 8, 8, 8, 8, 8, 8, 2, 8, 8, 8, 8],
            [2, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [2, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [8, 8, 8, 8, 8, 8, 8, 3, 8, 8, 8, 8, 8],
            [8, 8, 8, 8, 0, 8, 8, 8, 8, 8, 8, 8, 8],
            [8, 8, 8, 0, 0, 0, 8, 8, 8, 8, 8, 8, 8],
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        ],
        dtype=int,
    ),
    np.array(
        [
            [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 5, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 5, 9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 5, 5, 9, 9, 9, 9],
            [9, 9, 2, 9, 9, 9, 9, 9, 5, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9, 5, 9, 9, 9, 8],
            [9, 7, 7, 7, 9, 9, 9, 9, 9, 9, 9, 9, 9],
            [9, 7, 7, 7, 9, 9, 3, 9, 9, 9, 3, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 3, 9, 3, 9, 9, 9],
            [9, 9, 6, 9, 9, 9, 9, 3, 9, 3, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 3, 9, 9, 9, 3, 9, 9],
            [9, 9, 9, 9, 9, 9, 3, 9, 9, 9, 3, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
            [0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [9, 9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0],
            [0, 0, 0, 4, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 7],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=int,
    ),
]


def _valid_dims() -> list[tuple[int, int]]:
    results = []
    for area in range(_AREA_MIN, _AREA_MAX + 1):
        dims = [(area // w, w) for w in range(1, area + 1) if area % w == 0 and area // w >= w]
        best_gap = min(abs(h - w) for h, w in dims)
        best = [(h, w) for h, w in dims if abs(h - w) == best_gap]
        if len(best) == 1:
            h, w = best[0]
            if w == _WIDTH_REQUIRED and h <= _HEIGHT_MAX:
                results.append((h, w))
    return results


_VALID_DIMS = _valid_dims()
_DIM_WEIGHTS = [h * w for h, w in _VALID_DIMS]


def _zigzag_frame(H: int) -> tuple[set[tuple[int, int]], int, int]:
    cells = set()
    start_c = random.randint(0, 1)
    c = start_c
    for r in range(H):
        cells.add((r, c))
        if r == H - 1:
            break
        move = random.choice(["down", "horiz", "diag"])
        if move == "horiz":
            cells.add((r, 1 - c))
            c = 1 - c
        elif move == "diag":
            c = 1 - c
    return cells, start_c, c


def _fill_holes(grid: np.ndarray, H: int, W: int, frame1: int, frame2: int, split: int, fill_pool: list[int]) -> None:
    holes = components_mask(grid == -1, connectivity=4)
    prev_col = None
    for hole in holes:
        cells = sorted((int(r), int(c)) for r, c in hole)
        chunks = [cells[i : i + _MAX_CHUNK_SIZE] for i in range(0, len(cells), _MAX_CHUNK_SIZE)]
        for chunk in chunks:
            frame_touch = {frame1 if r < split else frame2 for r, _ in chunk}
            avoid = frame_touch | ({prev_col} if prev_col is not None else set())
            opts = [f for f in fill_pool if f not in avoid]
            if not opts:
                opts = [f for f in fill_pool if f not in frame_touch] or fill_pool
            col = random.choice(opts)
            for r, c in chunk:
                grid[r, c] = col
                grid[r, W - 1 - c] = col
            prev_col = col


def _build_clean_output(H: int, W: int) -> tuple[np.ndarray | None, set[int]]:
    half = W // 2
    n_colors = random.randint(_N_COLORS_MIN, _N_COLORS_MAX)
    palette = list(range(n_colors))
    random.shuffle(palette)

    use_two_frames = (
        H >= _TWO_FRAME_MIN_H
        and n_colors >= _TWO_FRAME_MIN_COLORS
        and random.random() < _TWO_FRAME_PROB
    )
    idx = 0
    frame1 = palette[idx]
    idx += 1
    frame2 = frame1
    if use_two_frames:
        frame2 = palette[idx]
        idx += 1
    fill_pool = palette[idx:]
    if not fill_pool:
        return None, set()

    frame_half, start_c, end_c = _zigzag_frame(H)
    split = H // 2 if frame1 != frame2 else H
    grid = np.full((H, W), -1, dtype=int)

    for r, c in frame_half:
        fc = frame1 if r < split else frame2
        grid[r, c] = fc
        grid[r, W - 1 - c] = fc

    if start_c == 1 and random.random() < _CENTER_CAP_PROB:
        grid[0, half] = frame1
    if end_c == 1 and random.random() < _CENTER_CAP_PROB:
        grid[H - 1, half] = frame2

    _fill_holes(grid, H, W, frame1, frame2, split, fill_pool)
    if np.any(grid < 0):
        return None, set()
    frame_set = {frame1}
    if frame2 != frame1:
        frame_set.add(frame2)
    return grid, frame_set


def _make_clean_output() -> tuple[np.ndarray | None, int, set[int]]:
    H, W = random.choices(_VALID_DIMS, weights=_DIM_WEIGHTS, k=1)[0]
    out, abstract_fcs = _build_clean_output(H, W)
    if out is None:
        return None, 0, set()
    bg = random.randint(0, 9)
    abstract = sorted(np.unique(out))
    available = [c for c in range(10) if c != bg]
    random.shuffle(available)
    if len(available) < len(abstract):
        return None, 0, set()
    cmap = {int(a): available[i] for i, a in enumerate(abstract)}
    real_fcs = {cmap[a] for a in abstract_fcs}
    mapped = np.vectorize(lambda x: cmap[int(x)])(out)
    return mapped.astype(int), bg, real_fcs


def _local_colors(board: np.ndarray, r: int, c: int) -> set[int]:
    H, W = board.shape
    colors = {int(board[r, c])}
    for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if 0 <= rr < H and 0 <= cc < W:
            colors.add(int(board[rr, cc]))
    return colors


def _candidate_pairs(board: np.ndarray, frame_colors: set[int]) -> list[tuple[int, int]]:
    H, W = board.shape
    out = []
    for r in range(H):
        for c in range(W // 2):
            mc = W - 1 - c
            col = int(board[r, c])
            if col != int(board[r, mc]):
                continue
            if col in frame_colors:
                continue
            if _support_score(board, r, c, bg=-99) <= 0:
                continue
            out.append((r, c))
    random.shuffle(out)
    return out


def _far_enough(chosen: list[tuple[int, int]], cand: tuple[int, int], W: int) -> bool:
    r, c = cand
    pts = [(r, c), (r, W - 1 - c)]
    for rr, cc in chosen:
        cur = [(rr, cc), (rr, W - 1 - cc)]
        for a in pts:
            for b in cur:
                if max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= _PAIR_GAP:
                    return False
    return True


def _pick_damage_color(
    board: np.ndarray,
    bg: int,
    frame_colors: set[int],
    r: int,
    c: int,
    extra_preferred: list[int],
    disallow: set[int] | None = None,
) -> int | None:
    forbid = _local_colors(board, r, c) | frame_colors | {bg}
    if disallow:
        forbid |= set(disallow)
    preferred = [col for col in extra_preferred if col not in forbid]
    if preferred:
        return random.choice(preferred)
    fallback = [col for col in range(10) if col not in forbid]
    if fallback:
        return random.choice(fallback)
    return None


def _consume_extra_color(extra_preferred: list[int], col: int) -> None:
    if col in extra_preferred:
        extra_preferred.remove(col)


def _damage_clean_board(clean: np.ndarray, bg: int, frame_colors: set[int]) -> tuple[np.ndarray | None, dict[str, int]]:
    board = clean.copy()
    present = set(map(int, np.unique(clean)))
    extra_preferred = [c for c in range(10) if c not in present and c != bg and c not in frame_colors]
    candidates = _candidate_pairs(clean, frame_colors)
    if len(candidates) < _FILL_REPAIRS_REQUIRED + _CLEAR_REPAIRS_REQUIRED:
        return None, {}

    chosen: list[tuple[int, int]] = []
    stats = {"fill_pairs": 0, "clear_pairs": 0}

    def choose_candidate() -> tuple[int, int] | None:
        for cand in candidates:
            if cand not in chosen and _far_enough(chosen, cand, board.shape[1]):
                chosen.append(cand)
                return cand
        return None

    for _ in range(_FILL_REPAIRS_REQUIRED):
        cand = choose_candidate()
        if cand is None:
            return None, {}
        r, c = cand
        mc = board.shape[1] - 1 - c
        target = (r, c) if random.random() < 0.5 else (r, mc)
        bad = _pick_damage_color(board, bg, frame_colors, target[0], target[1], extra_preferred)
        if bad is None:
            return None, {}
        board[target] = bad
        _consume_extra_color(extra_preferred, bad)
        stats["fill_pairs"] += 1

    for _ in range(_CLEAR_REPAIRS_REQUIRED):
        cand = choose_candidate()
        if cand is None:
            return None, {}
        r, c = cand
        mc = board.shape[1] - 1 - c
        bad1 = _pick_damage_color(board, bg, frame_colors, r, c, extra_preferred)
        if bad1 is None:
            return None, {}
        _consume_extra_color(extra_preferred, bad1)
        bad2 = _pick_damage_color(board, bg, frame_colors, r, mc, extra_preferred, disallow={bad1})
        if bad2 is None:
            return None, {}
        _consume_extra_color(extra_preferred, bad2)
        board[r, c] = bad1
        board[r, mc] = bad2
        stats["clear_pairs"] += 1

    if random.random() < _EXTRA_FILL_PROB:
        cand = choose_candidate()
        if cand is not None:
            r, c = cand
            mc = board.shape[1] - 1 - c
            target = (r, c) if random.random() < 0.5 else (r, mc)
            bad = _pick_damage_color(board, bg, frame_colors, target[0], target[1], extra_preferred)
            if bad is not None:
                board[target] = bad
                _consume_extra_color(extra_preferred, bad)
                stats["fill_pairs"] += 1

    repaired = _repair_symmetry(board, bg)
    counts = _repair_event_counts(board, bg)
    if counts["copy_pairs"] < _FILL_REPAIRS_REQUIRED or counts["clear_pairs"] < _CLEAR_REPAIRS_REQUIRED:
        return None, {}
    if counts["bad_pairs_after_repair"] != 0:
        return None, {}
    if np.array_equal(board, repaired):
        return None, {}
    return board, stats


def _extract_non_frame_patches(out: np.ndarray, bg: int, frame_colors: set[int]) -> list[np.ndarray]:
    patches = []
    for col in sorted(map(int, np.unique(out))):
        if col in frame_colors:
            continue
        for pts in components_mask(out == col, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
            for r, c in pts:
                patch[int(r) - r0, int(c) - c0] = col
            patches.append(patch)
    random.shuffle(patches)
    patches.sort(key=lambda p: (-(p != bg).sum(), random.random()))
    return patches


def _scatter(out: np.ndarray, bg: int, frame_colors: set[int]) -> np.ndarray:
    rows = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    cols = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    canvas = np.full((rows, cols), bg, dtype=int)
    oh, ow = out.shape

    max_r = rows - oh - 2
    max_c = cols - ow - 2
    if max_r < 1 or max_c < 1:
        raise ValueError("canvas too small")
    off_r = random.randint(1, max_r)
    off_c = random.randint(1, max_c)

    canvas[off_r : off_r + oh, off_c : off_c + ow] = -1
    for patch in _extract_non_frame_patches(out, bg, frame_colors):
        r, c = random_free_location(
            canvas,
            patch,
            transparent=bg,
            padding=_SCATTER_PADDING,
            padding_connectivity=8,
        )
        paste(canvas, patch, r, c, transparent=bg)

    frame_mask = np.zeros((oh, ow), dtype=bool)
    for col in frame_colors:
        frame_mask |= out == col
    bbox_region = canvas[off_r : off_r + oh, off_c : off_c + ow]
    bbox_region[:] = bg
    bbox_region[frame_mask] = out[frame_mask]
    return canvas


def _piece_count(out: np.ndarray) -> int:
    return sum(len(components_mask(out == col, connectivity=4)) for col in np.unique(out))


def _gen_attempt() -> np.ndarray | None:
    clean, bg, frame_colors = _make_clean_output()
    if clean is None:
        return None
    damaged, _ = _damage_clean_board(clean, bg, frame_colors)
    if damaged is None:
        return None
    repaired = _repair_symmetry(damaged, bg)
    if not np.array_equal(repaired, repaired[:, ::-1]):
        return None
    if _piece_count(damaged) < _PIECE_COUNT_MIN or _piece_count(damaged) > _PIECE_COUNT_MAX:
        return None
    if any(len(pts) > _MAX_PIECE_AREA for col in np.unique(damaged) for pts in components_mask(damaged == col, connectivity=4)):
        return None

    try:
        inp = _scatter(damaged, bg, frame_colors)
    except Exception:
        return None
    metrics = combo_metrics_from_board(damaged, bg)
    if metrics["mismatch_pairs_before_repair"] < (_FILL_REPAIRS_REQUIRED + _CLEAR_REPAIRS_REQUIRED):
        return None
    if metrics["panel_output_area"] <= metrics["rect_output_area"]:
        return None
    if not _requirements_match_board(damaged, bg):
        return None
    try:
        _find_frame_parts(damaged, bg, damaged.shape[0], damaged.shape[1])
        _find_frame_parts(inp, bg, damaged.shape[0], damaged.shape[1])
        main(inp)
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    return random.choice(_PRECOMPUTED_INPUTS).copy()
