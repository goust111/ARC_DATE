# Combo variant of e8686506 (index 47)
# combo_spec: D1/1+D12/2
# Name: 部分碎片需旋转+连接器匹配

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# ---------------------------------------------------------------------------
# Fusion design analysis for D1/1 + D12/2
# ---------------------------------------------------------------------------
# fusion_spec:
#   concrete_visualization:
#     - The output is a dense 10x5 left-right-symmetric rectangle.
#     - One pre-aligned 8-connected multicolor frame piece already appears in the input
#       with its output-relative bbox preserved.
#     - The nine interior pieces are 2-cell multicolor dominoes whose cell colors are
#       connector colors. Those colors are the only legal seam labels.
#     - A subset of scattered dominoes has one adjacent marker pixel (color 9). Those
#       are the only pieces allowed to rotate. In the input they are actually scattered
#       after a non-zero quarter-turn, so the mark is behaviorally meaningful.
#   behavior_trace_example:
#     - Original task step: a monochrome fragment only needed translation into a hole.
#     - D1 step: a vertical domino with top connector 6 and bottom connector 7 can only
#       sit where a 6-labeled seam is above it and a 7-labeled seam is below it.
#     - D12 step: if that domino is marked, the input may show it as horizontal [6,7].
#       The solver must strip the marker, try the quarter-turn variants, and then keep
#       only the orientation whose connectors meet equal-colored neighbors.
#     - Difference: translation alone is insufficient, and rotation alone is also
#       insufficient; the correct turn is determined by connector compatibility.
#   complexity_calibration:
#     - The family is intentionally narrowed to one clear layout instead of the full
#       original generator. This keeps the fused rule legible and avoids hiding the
#       new logic inside unrelated shape variability.
#     - Connector colors are reused across three chambers, so the matching cue repeats.
#     - 2-4 marked pieces appear in every sample, while several unmarked pieces remain,
#       making the “some rotate, some do not” distinction visually obvious.
#     - Marked pieces are rotated by 90 or 270 degrees only, so the rotation is visible
#       by shape, not just by color order.
#   implicit_assumption_audit:
#     - The original monochrome-per-color component extraction no longer works because
#       each piece is now multicolor; extraction must use all non-bg, non-marker cells
#       as 8-connected sprites.
#     - Any heuristic based on singleton/body colors is invalid because connector colors
#       live inside pieces and marker singletons are input-only instructions.
#     - Frame detection must ignore marker pixels or the extra dot could pollute the
#       candidate pool.
#     - Placement legality can no longer be “fits hole + mirror”; every inter-piece
#       contact must be a connector-to-connector color match, and marked pieces need
#       rotation variants of the full multicolor sprite.
#   placement_legality:
#     - Generator scatters each piece patch onto background with Chebyshev padding 1.
#     - Rotatable pieces are scattered as one composite patch: rotated domino plus one
#       adjacent marker pixel inside a 1-cell halo, guaranteeing the marker touches
#       exactly one piece and nothing else.
#     - The frame bbox is blocked while scattering interior pieces, so the anchor stays
#       clean and every piece remains a separate 8-component once marker color is ignored.
#   observability_and_cleanliness:
#     - Marker color 9 never appears in the output, so humans can interpret it as an
#       instruction rather than as content.
#     - Connector colors 6/7/8 are reserved for seam pixels and visibly repeat on both
#       the frame and the interior dominoes.
#     - The scene is uncluttered: one anchored frame, nine small dominoes, and a few
#       single-pixel rotation tags.
#   generator_guarantees_needed_by_main:
#     - background is never a connector color and never the marker color
#     - the frame piece is unmarked and already in output orientation
#     - every marker touches exactly one non-frame piece
#     - every marked piece is quarter-turned by k in {1, 3}
#     - every connector pixel in the output has at least one equal-colored neighbor on
#       a different piece, so the seam rule is globally checkable


CONNECTOR_COLORS = (6, 7, 8)
MARKER_COLOR = 9

OUT_H = 10
OUT_W = 5

ROTATABLE_MIN = 2
ROTATABLE_MAX = 4

CANVAS_MIN = 13
CANVAS_MAX = 15
SCATTER_PADDING = 1
GEN_MAX_RETRIES = 500


def _factors(n: int) -> list[tuple[int, int]]:
    out = []
    for w in range(1, n + 1):
        if n % w == 0:
            out.append((n // w, w))
    return out


def _cells_to_patch(cells: tuple[tuple[int, int, int], ...], fill: int = -1) -> np.ndarray:
    h = max(int(r) for r, _c, _col in cells) + 1
    w = max(int(c) for _r, c, _col in cells) + 1
    patch = np.full((h, w), int(fill), dtype=int)
    for r, c, col in cells:
        patch[int(r), int(c)] = int(col)
    return patch


def _patch_to_cells(patch: np.ndarray, fill: int = -1) -> tuple[tuple[int, int, int], ...]:
    cells = []
    for r, c in np.argwhere(patch != int(fill)):
        cells.append((int(r), int(c), int(patch[int(r), int(c)])))
    cells.sort()
    return tuple(cells)


def _rotate_cells(cells: tuple[tuple[int, int, int], ...], k: int) -> tuple[tuple[int, int, int], ...]:
    return _patch_to_cells(np.rot90(_cells_to_patch(cells), int(k) % 4))


def _rotation_variants(cells: tuple[tuple[int, int, int], ...]) -> list[dict]:
    out = []
    seen = set()
    for k in range(4):
        rcells = _rotate_cells(cells, k)
        if rcells in seen:
            continue
        seen.add(rcells)
        out.append(_variant_from_cells(rcells))
    return out


def _variant_from_cells(cells: tuple[tuple[int, int, int], ...]) -> dict:
    coords = {(int(r), int(c)) for r, c, _col in cells}
    connectors = {(int(r), int(c)) for r, c, col in cells if int(col) in CONNECTOR_COLORS}
    h = max(int(r) for r, _c, _col in cells) + 1
    w = max(int(c) for _r, c, _col in cells) + 1
    return {
        "cells": tuple(sorted((int(r), int(c), int(col)) for r, c, col in cells)),
        "coords": coords,
        "connectors": connectors,
        "h": int(h),
        "w": int(w),
    }


def _normalize_piece(pts: np.ndarray, grid: np.ndarray) -> dict:
    r0, c0, r1, c1 = bbox_coords(pts)
    cells = []
    for r, c in pts:
        cells.append((int(r) - int(r0), int(c) - int(c0), int(grid[int(r), int(c)])))
    base = _variant_from_cells(tuple(sorted(cells)))
    base["area"] = int(len(cells))
    base["src_bbox"] = (int(r0), int(c0), int(r1), int(c1))
    base["rotatable"] = False
    base["variants"] = [dict(base)]
    return base


def _is_lr_symmetric_mask(coords: set[tuple[int, int]], H: int, W: int) -> bool:
    bmp = np.zeros((H, W), dtype=bool)
    for r, c in coords:
        bmp[int(r), int(c)] = True
    return np.array_equal(bmp, bmp[:, ::-1])


def _find_frame_group(pieces: list[dict], H: int, W: int) -> list[tuple[int, int, int]]:
    def group_ok(idxs: list[int]) -> list[tuple[int, int, int]] | None:
        boxes = [pieces[i]["src_bbox"] for i in idxs]
        gr0 = min(int(b[0]) for b in boxes)
        gc0 = min(int(b[1]) for b in boxes)
        gr1 = max(int(b[2]) for b in boxes)
        gc1 = max(int(b[3]) for b in boxes)
        if int(gr1 - gr0 + 1) != H or int(gc1 - gc0 + 1) != W:
            return None
        coords = set()
        anchors = []
        for i in idxs:
            pr0, pc0, _pr1, _pc1 = pieces[i]["src_bbox"]
            anchors.append((int(i), int(pr0 - gr0), int(pc0 - gc0)))
            for dr, dc, _col in pieces[i]["cells"]:
                coords.add((int(pr0 - gr0 + dr), int(pc0 - gc0 + dc)))
        if not _is_lr_symmetric_mask(coords, H, W):
            return None
        return anchors

    found: list[tuple[int, list[tuple[int, int, int]]]] = []
    for i in range(len(pieces)):
        got = group_ok([i])
        if got is not None:
            found.append((int(pieces[i]["area"]), got))
    for i in range(len(pieces)):
        for j in range(i + 1, len(pieces)):
            got = group_ok([i, j])
            if got is not None:
                found.append((int(pieces[i]["area"] + pieces[j]["area"]), got))
    assert found, "expected a 1-2 piece LR-symmetric frame group"
    best_area = min(area for area, _got in found)
    best = [got for area, got in found if area == best_area]
    assert len(best) == 1, "expected a unique minimum-area frame group"
    return best[0]


def _extract_marked_pieces(g: np.ndarray, bg: int) -> list[dict]:
    fg_mask = (g != int(bg)) & (g != int(MARKER_COLOR))
    comps = components_mask(fg_mask, connectivity=8)
    assert comps, "grid must contain foreground pieces"
    pieces = [_normalize_piece(pts, g) for pts in comps]

    piece_cells = [{(int(r), int(c)) for r, c in pts} for pts in comps]
    marker_pts: list[tuple[int, int]] = []
    if int(MARKER_COLOR) in set(map(int, np.unique(g))):
        marker_comps = components_mask(g == int(MARKER_COLOR), connectivity=4)
        assert all(len(m) == 1 for m in marker_comps), "marker color must appear only as singleton pixels"
        marker_pts = [(int(m[0][0]), int(m[0][1])) for m in marker_comps]

    matched_piece_indices = set()
    for mr, mc in marker_pts:
        matches = []
        for i, cells in enumerate(piece_cells):
            if any(max(abs(int(mr) - int(r)), abs(int(mc) - int(c))) == 1 for r, c in cells):
                matches.append(i)
        assert len(matches) == 1, "each marker must touch exactly one fragment"
        pi = int(matches[0])
        pieces[pi]["rotatable"] = True
        pieces[pi]["variants"] = _rotation_variants(pieces[pi]["cells"])
        matched_piece_indices.add(pi)

    assert len(matched_piece_indices) == len(marker_pts), "markers must not collide onto the same piece"
    return pieces


def _assemble(pieces: list[dict], H: int, W: int, frame_group: list[tuple[int, int, int]]) -> np.ndarray:
    board = np.full((H, W), -1, dtype=int)
    owner = np.full((H, W), -1, dtype=int)
    is_connector = np.zeros((H, W), dtype=bool)
    placed: dict[int, tuple[int, int, int]] = {}

    def can_place(pi: int, vi: int, r0: int, c0: int, require_touch: bool) -> bool:
        variant = pieces[pi]["variants"][vi]
        local_cells: dict[tuple[int, int], int] = {}
        touched = False
        matched_any = False

        for dr, dc, col in variant["cells"]:
            r, c = int(r0 + dr), int(c0 + dc)
            if not (0 <= r < H and 0 <= c < W):
                return False
            if owner[r, c] != -1:
                return False
            local_cells[(r, c)] = int(col)

        for (r, c), col in local_cells.items():
            mc = W - 1 - int(c)
            if (r, mc) in local_cells and int(local_cells[(r, mc)]) != int(col):
                return False
            if owner[r, mc] != -1 and int(board[r, mc]) != int(col):
                return False

        for dr, dc, col in variant["cells"]:
            r, c = int(r0 + dr), int(c0 + dc)
            here_is_connector = (int(dr), int(dc)) in variant["connectors"]
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if not (0 <= rr < H and 0 <= cc < W):
                    continue
                if (rr, cc) in local_cells:
                    continue
                if owner[rr, cc] == -1:
                    continue
                touched = True
                if not (here_is_connector and bool(is_connector[rr, cc]) and int(board[rr, cc]) == int(col)):
                    return False
                matched_any = True

        if require_touch and not touched:
            return False
        if touched and not matched_any:
            return False
        return True

    def place(pi: int, vi: int, r0: int, c0: int) -> list[tuple[int, int]]:
        variant = pieces[pi]["variants"][vi]
        changed = []
        for dr, dc, col in variant["cells"]:
            r, c = int(r0 + dr), int(c0 + dc)
            board[r, c] = int(col)
            owner[r, c] = int(pi)
            is_connector[r, c] = (int(dr), int(dc)) in variant["connectors"]
            changed.append((r, c))
        placed[int(pi)] = (int(vi), int(r0), int(c0))
        return changed

    def unplace(pi: int, changed: list[tuple[int, int]]) -> None:
        for r, c in changed:
            board[r, c] = -1
            owner[r, c] = -1
            is_connector[r, c] = False
        placed.pop(int(pi), None)

    def placements_for(pi: int) -> list[tuple[int, int, int]]:
        require_touch = bool(placed)
        out = []
        for vi, variant in enumerate(pieces[pi]["variants"]):
            for r0 in range(H - int(variant["h"]) + 1):
                for c0 in range(W - int(variant["w"]) + 1):
                    if can_place(pi, vi, r0, c0, require_touch):
                        out.append((int(vi), int(r0), int(c0)))
        return out

    def final_ok() -> bool:
        if np.any(owner == -1):
            return False
        if not np.array_equal(board, board[:, ::-1]):
            return False
        for pi, (vi, r0, c0) in placed.items():
            variant = pieces[pi]["variants"][vi]
            for dr, dc in variant["connectors"]:
                r, c = int(r0 + dr), int(c0 + dc)
                ok = False
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= rr < H and 0 <= cc < W and owner[rr, cc] != int(pi):
                        if bool(is_connector[rr, cc]) and int(board[rr, cc]) == int(board[r, c]):
                            ok = True
                            break
                if not ok:
                    return False
        return True

    for pi, r0, c0 in frame_group:
        frame_variants = pieces[pi]["variants"]
        assert len(frame_variants) == 1, "frame piece must not be rotatable"
        place(pi, 0, r0, c0)

    def search() -> bool:
        remaining = [pi for pi in range(len(pieces)) if pi not in placed]
        if not remaining:
            return final_ok()

        while True:
            forced = None
            best_pi = None
            best_places = None
            for pi in remaining:
                places = placements_for(pi)
                if not places:
                    return False
                if len(places) == 1:
                    forced = (pi, places[0])
                    break
                if best_places is None or len(places) < len(best_places):
                    best_pi = int(pi)
                    best_places = places
            if forced is None:
                break
            pi, (vi, r0, c0) = forced
            place(pi, vi, r0, c0)
            remaining = [idx for idx in remaining if idx != pi]
            if not remaining:
                return final_ok()

        assert best_pi is not None and best_places is not None
        for vi, r0, c0 in best_places:
            changed = place(best_pi, vi, r0, c0)
            if search():
                return True
            unplace(best_pi, changed)
        return False

    ok = search()
    assert ok, "connector + rotation assembly failed"
    return board.astype(int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered multicolor fragments into a left-right symmetric rectangle using connector matches, with rotation only on marked pieces.

    Input structure:
    - The background is the most common color; each real piece is an 8-connected multicolor sprite built from ordinary body cells and connector-colored seam cells.
    - One unmarked sprite is the pre-aligned LR-symmetric frame whose bbox already equals the output rectangle.
    - A singleton marker pixel next to a scattered fragment means that fragment may need a 90-degree turn before assembly; unmarked fragments must keep their input orientation.

    Rule:
    - Ignore marker pixels as output content, but use them to classify which fragments are rotatable.
    - Infer the output rectangle size from the total non-marker foreground area.
    - Anchor the frame, then place the remaining pieces so the rectangle is exactly LR-symmetric and every inter-piece contact is a same-colored connector-to-connector match.

    Key insight:
    - The rotation mark does not by itself tell you where a piece goes, and connector colors do not by themselves tell you which orientation a marked piece should use.
    - The puzzle resolves only by combining both cues: marked pieces expand into quarter-turn variants, and the valid variant is the one whose connector cells line up with equal-colored connectors on the anchored frame and already placed neighbors.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    pieces = _extract_marked_pieces(g, bg)
    total_area = int(sum(piece["area"] for piece in pieces))
    assert total_area > 0, "total area must be positive"

    dims = [(h, w) for h, w in _factors(total_area) if h >= w]
    assert dims, "at least one portrait factorization must exist"
    best_gap = min(abs(h - w) for h, w in dims)
    best_dims = [(h, w) for h, w in dims if abs(h - w) == best_gap]
    assert len(best_dims) == 1, "closest-to-square portrait output shape must be unique"
    H, W = best_dims[0]

    frame_group = _find_frame_group(pieces, H, W)
    return _assemble(pieces, H, W, frame_group).astype(int)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def _build_structured_output() -> tuple[np.ndarray, int, np.ndarray, set[int]]:
    bg = random.choice([c for c in range(10) if c not in CONNECTOR_COLORS and c != MARKER_COLOR])
    frame_color = random.choice([c for c in range(10) if c not in CONNECTOR_COLORS and c not in {bg, MARKER_COLOR}])

    out = np.full((OUT_H, OUT_W), int(frame_color), dtype=int)
    pid_grid = np.zeros((OUT_H, OUT_W), dtype=int)

    bar_rows = {3, 6}
    for r in range(OUT_H):
        for c in range(OUT_W):
            if r in (0, OUT_H - 1) or c in (0, OUT_W - 1) or r in bar_rows:
                pid_grid[r, c] = 0
            else:
                out[r, c] = -1
                pid_grid[r, c] = -1

    chamber_starts = [1, 4, 7]
    palette = list(CONNECTOR_COLORS)
    random.shuffle(palette)
    next_pid = 1
    for i, r0 in enumerate(chamber_starts):
        top_col = int(palette[i % len(palette)])
        bot_col = int(palette[(i + 1) % len(palette)])

        out[r0 - 1, 1:4] = top_col
        out[r0 + 2, 1:4] = bot_col
        out[r0, 0] = top_col
        out[r0 + 1, 0] = bot_col
        out[r0, OUT_W - 1] = top_col
        out[r0 + 1, OUT_W - 1] = bot_col

        for c in range(1, 4):
            out[r0, c] = top_col
            out[r0 + 1, c] = bot_col
            pid_grid[r0, c] = int(next_pid)
            pid_grid[r0 + 1, c] = int(next_pid)
            next_pid += 1

    assert np.all(out >= 0)
    return out.astype(int), int(bg), pid_grid.astype(int), {0}


def _patch_from_pid(out: np.ndarray, pid_grid: np.ndarray, pid: int, bg: int) -> np.ndarray:
    pts = np.argwhere(pid_grid == int(pid))
    r0, c0, r1, c1 = bbox_coords(pts)
    patch = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(bg), dtype=int)
    for r, c in pts:
        patch[int(r) - int(r0), int(c) - int(c0)] = int(out[int(r), int(c)])
    return patch


def _build_patches(out: np.ndarray, pid_grid: np.ndarray, bg: int, frame_pids: set[int]) -> tuple[np.ndarray, list[dict]]:
    frame_pts = np.argwhere(np.isin(pid_grid, sorted(frame_pids)))
    fr0, fc0, fr1, fc1 = bbox_coords(frame_pts)
    frame_patch = np.full((int(fr1 - fr0 + 1), int(fc1 - fc0 + 1)), int(bg), dtype=int)
    for r, c in frame_pts:
        frame_patch[int(r) - int(fr0), int(c) - int(fc0)] = int(out[int(r), int(c)])

    entries = []
    for pid in sorted(set(map(int, pid_grid.flatten()))):
        if pid < 0 or pid in frame_pids:
            continue
        entries.append({"pid": int(pid), "patch": _patch_from_pid(out, pid_grid, pid, bg)})
    return frame_patch, entries


def _marked_patch(base_patch: np.ndarray, bg: int, k: int) -> np.ndarray:
    rot_patch = np.rot90(base_patch, int(k) % 4)
    h, w = rot_patch.shape
    padded = np.full((h + 2, w + 2), int(bg), dtype=int)
    padded[1 : 1 + h, 1 : 1 + w] = rot_patch
    occ = np.argwhere(rot_patch != int(bg)) + 1
    candidates = []
    for r in range(h + 2):
        for c in range(w + 2):
            if padded[r, c] != int(bg):
                continue
            if any(max(abs(int(r) - int(rr)), abs(int(c) - int(cc))) == 1 for rr, cc in occ):
                candidates.append((int(r), int(c)))
    assert candidates, "marked patch needs an adjacent marker location"
    mr, mc = sorted(candidates)[0]
    padded[int(mr), int(mc)] = int(MARKER_COLOR)
    return padded


def _scatter(out: np.ndarray, pid_grid: np.ndarray, bg: int, frame_pids: set[int]) -> tuple[np.ndarray, dict]:
    rows = random.randint(CANVAS_MIN, CANVAS_MAX)
    cols = random.randint(CANVAS_MIN, CANVAS_MAX)
    canvas = np.full((rows, cols), int(bg), dtype=int)
    oh, ow = out.shape

    max_r = rows - oh - 2
    max_c = cols - ow - 2
    if max_r < 1 or max_c < 1:
        raise ValueError("frame cannot fit on canvas")
    off_r = random.randint(1, max_r)
    off_c = random.randint(1, max_c)

    frame_patch, entries = _build_patches(out, pid_grid, bg, frame_pids)
    canvas[off_r : off_r + oh, off_c : off_c + ow] = -1

    n_rot = random.randint(ROTATABLE_MIN, min(ROTATABLE_MAX, len(entries) - 2))
    rot_ids = set(random.sample([int(e["pid"]) for e in entries], n_rot))

    rotated_count = 0
    for entry in entries:
        patch = entry["patch"]
        rotatable = int(entry["pid"]) in rot_ids
        if rotatable:
            k = random.choice((1, 3))
            patch = _marked_patch(patch, bg, k)
            rotated_count += 1
        r, c = random_free_location(
            canvas,
            patch,
            transparent=bg,
            padding=SCATTER_PADDING,
            padding_connectivity=8,
        )
        paste(canvas, patch, r, c, transparent=bg)

    canvas[off_r : off_r + oh, off_c : off_c + ow] = frame_patch
    meta = {
        "rotatable_count": int(len(rot_ids)),
        "rotated_count": int(rotated_count),
    }
    return canvas.astype(int), meta


def _behavior_metrics(inp: np.ndarray) -> dict:
    bg = get_bg(inp)
    pieces = _extract_marked_pieces(inp, bg)
    rotatable = [p for p in pieces if p["rotatable"]]
    return {
        "rotatable_count": int(len(rotatable)),
        "horizontal_marked_count": int(sum(1 for p in rotatable if p["h"] == 1 and p["w"] == 2)),
        "vertical_fixed_domino_count": int(
            sum(1 for p in pieces if (not p["rotatable"]) and p["area"] == 2 and p["h"] == 2 and p["w"] == 1)
        ),
        "marker_count": int(np.sum(inp == int(MARKER_COLOR))),
    }


def _gen_attempt() -> np.ndarray | None:
    out, bg, pid_grid, frame_pids = _build_structured_output()
    try:
        inp, meta = _scatter(out, pid_grid, bg, frame_pids)
        if meta["rotatable_count"] < ROTATABLE_MIN or meta["rotated_count"] != meta["rotatable_count"]:
            return None
        got = main(inp)
    except Exception:
        return None
    if not np.array_equal(got, out):
        return None
    metrics = _behavior_metrics(inp)
    if metrics["rotatable_count"] < ROTATABLE_MIN:
        return None
    if metrics["horizontal_marked_count"] != metrics["rotatable_count"]:
        return None
    return inp.astype(int)


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid connector+rotation e8686506 combo input")
