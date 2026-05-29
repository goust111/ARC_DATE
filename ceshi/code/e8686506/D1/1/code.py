# Variant of e8686506
# Dimension: D1 — 匹配条件：边缘对齐拼合 vs 其他匹配逻辑 (match_cond)
# Change: s.edge.position aligns with d.adjacent_piece.edge — 碎片的边缘位置与... → 连接器标记匹配

import random
from collections import defaultdict

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# ---------------------------------------------------------------------------
# Design analysis for the D1 variant
# ---------------------------------------------------------------------------
# variant_spec:
#   core_change:
#     old: pieces were effectively positioned by output-frame locks plus shape fitting.
#     new: each piece is now a multicolor 8-connected sprite; special connector pixels
#       on piece boundaries must meet an equal-colored connector on a neighboring piece.
#       Two pieces may touch in the final rectangle only when their shared boundary
#       contains at least one connector-to-connector contact.
#   concrete_visualization:
#     input:
#       - one anchored frame piece still appears as a grouped outline with its original
#         output-relative geometry preserved.
#       - the other pieces are scattered, but they are no longer monochrome; small
#         accent pixels of fixed connector colors sit on their edges.
#     output:
#       - the dense rectangle remains exact LR-symmetric.
#       - connector pixels remain visible in the assembled rectangle as small equal-
#         colored pairs straddling seams between neighboring pieces.
#   behavior_trace:
#     original_local_step:
#       - choose a piece placement because its outline is the only way to fit a hole
#         while respecting border/color heuristics.
#     variant_local_step:
#       - choose a piece placement because, once translated into the rectangle, one or
#         more of its connector pixels land directly beside equal-colored connector
#         pixels on already anchored pieces.
#       - mere edge contact is insufficient; touching a neighbor without a connector
#         match is rejected.
#   complexity_calibration:
#     - the frame anchor and LR symmetry are preserved so the task does not become an
#       unconstrained global search.
#     - connector colors are fixed and visually repeated across multiple seams so the
#       new rule is observable from a few examples.
#     - width stays 5 and height stays 8; this is an intentional adaptation so pieces
#       have enough interior contact structure for connector matching to be visible.
#   implicit_assumption_audit:
#     - original solver extracted pieces per color; that breaks once pieces are
#       multicolor, so the new solver extracts 8-connected non-bg components.
#     - original same-color-touch prohibition is no longer a reliable piece-boundary
#       cue, so it is removed from legality checks.
#     - original per-cell req-color anchors based on singleton/bar colors are tied to
#       monochrome pieces and are dropped; the frame piece now provides the main anchor.
#     - mirror checks remain valid, but now they apply to exact cell colors, including
#       connector colors.
#   placement_legality:
#     - generator keeps pieces non-overlapping when scattered and keeps the frame bbox
#       reserved, so input components stay separable.
#     - solver placements must fit in bounds, avoid overlap, preserve LR symmetry, and
#       touch the existing assembly through at least one connector-matched interface.
#   observability_and_cleanliness:
#     - connector colors are fixed, sparse, and only used for connectors, so the new
#       matching cue is visually isolated from ordinary body colors.
#     - the frame remains a single anchored outline, preventing the scene from looking
#       like arbitrary colored noise.


CONNECTOR_COLORS = (6, 7, 8)


def _piece_sig(pts: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = bbox_coords(pts)
    coords = tuple(sorted((int(r) - r0, int(c) - c0) for r, c in pts))
    return (r1 - r0 + 1), (c1 - c0 + 1), coords


def _factors(n: int) -> list[tuple[int, int]]:
    out = []
    for w in range(1, n + 1):
        if n % w == 0:
            h = n // w
            out.append((h, w))
    return out


def _normalize_piece(pts: np.ndarray, grid: np.ndarray) -> dict:
    r0, c0, r1, c1 = bbox_coords(pts)
    cells = []
    for r, c in pts:
        dr = int(r) - int(r0)
        dc = int(c) - int(c0)
        cells.append((dr, dc, int(grid[int(r), int(c)])))
    cells.sort()
    coords = {(dr, dc) for dr, dc, _col in cells}
    connectors = {(dr, dc) for dr, dc, col in cells if col in CONNECTOR_COLORS}
    return {
        "cells": tuple(cells),
        "coords": coords,
        "connectors": connectors,
        "h": int(r1 - r0 + 1),
        "w": int(c1 - c0 + 1),
        "area": int(len(cells)),
        "src_bbox": (int(r0), int(c0), int(r1), int(c1)),
    }


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
            area = sum(int(pieces[idx]["area"]) for idx in [i])
            found.append((area, got))
    for i in range(len(pieces)):
        for j in range(i + 1, len(pieces)):
            got = group_ok([i, j])
            if got is not None:
                area = int(pieces[i]["area"]) + int(pieces[j]["area"])
                found.append((area, got))
    assert found, "expected at least one 1-2 piece LR-symmetric frame group"
    best_area = min(area for area, _got in found)
    best = [got for area, got in found if area == best_area]
    assert len(best) == 1, "expected a unique minimum-area frame group"
    return best[0]


def _assemble(pieces: list[dict], H: int, W: int, frame_group: list[tuple[int, int, int]]) -> np.ndarray:
    board = np.full((H, W), -1, dtype=int)
    owner = np.full((H, W), -1, dtype=int)
    is_connector = np.zeros((H, W), dtype=bool)
    placed: dict[int, tuple[int, int]] = {}

    def can_place(pi: int, r0: int, c0: int, require_touch: bool) -> bool:
        piece = pieces[pi]
        local_cells = {}
        touched_neighbors: set[int] = set()
        matched_any = False
        connector_touch_ok: dict[tuple[int, int], bool] = {}
        connector_has_occupied_neighbor: dict[tuple[int, int], bool] = {}

        for dr, dc, col in piece["cells"]:
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

        for dr, dc, col in piece["cells"]:
            r, c = int(r0 + dr), int(c0 + dc)
            here_is_connector = (int(dr), int(dc)) in piece["connectors"]
            if here_is_connector:
                connector_touch_ok[(r, c)] = False
                connector_has_occupied_neighbor[(r, c)] = False
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if not (0 <= rr < H and 0 <= cc < W):
                    continue
                if (rr, cc) in local_cells:
                    continue
                if owner[rr, cc] == -1:
                    continue
                nid = int(owner[rr, cc])
                touched_neighbors.add(nid)
                if here_is_connector:
                    connector_has_occupied_neighbor[(r, c)] = True
                if here_is_connector and bool(is_connector[rr, cc]) and int(board[rr, cc]) == int(col):
                    matched_any = True
                    connector_touch_ok[(r, c)] = True

        if require_touch and not touched_neighbors:
            return False
        if touched_neighbors and not matched_any:
            return False
        for pos, had_occ in connector_has_occupied_neighbor.items():
            if had_occ and not connector_touch_ok[pos]:
                return False
        return True

    def place(pi: int, r0: int, c0: int) -> list[tuple[int, int]]:
        piece = pieces[pi]
        changed = []
        for dr, dc, col in piece["cells"]:
            r, c = int(r0 + dr), int(c0 + dc)
            board[r, c] = int(col)
            owner[r, c] = int(pi)
            is_connector[r, c] = (int(dr), int(dc)) in piece["connectors"]
            changed.append((r, c))
        placed[int(pi)] = (int(r0), int(c0))
        return changed

    def unplace(pi: int, changed: list[tuple[int, int]]) -> None:
        for r, c in changed:
            board[r, c] = -1
            owner[r, c] = -1
            is_connector[r, c] = False
        placed.pop(int(pi), None)

    def placements_for(pi: int) -> list[tuple[int, int]]:
        piece = pieces[pi]
        require_touch = bool(placed)
        out = []
        for r0 in range(H - int(piece["h"]) + 1):
            for c0 in range(W - int(piece["w"]) + 1):
                if can_place(pi, r0, c0, require_touch):
                    out.append((int(r0), int(c0)))
        return out

    def final_ok() -> bool:
        if np.any(owner == -1):
            return False
        if not np.array_equal(board, board[:, ::-1]):
            return False
        for pi, (r0, c0) in placed.items():
            piece = pieces[pi]
            for dr, dc in piece["connectors"]:
                r, c = int(r0 + dr), int(c0 + dc)
                ok = False
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= rr < H and 0 <= cc < W and owner[rr, cc] != pi:
                        if bool(is_connector[rr, cc]) and int(board[rr, cc]) == int(board[r, c]):
                            ok = True
                            break
                if not ok:
                    return False
        return True

    for pi, r0, c0 in frame_group:
        place(pi, r0, c0)

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
                    best_pi = pi
                    best_places = places
            if forced is None:
                break
            pi, (r0, c0) = forced
            place(pi, r0, c0)
            remaining = [idx for idx in remaining if idx != pi]
            if not remaining:
                return final_ok()

        assert best_pi is not None and best_places is not None
        for r0, c0 in best_places:
            changed = place(best_pi, r0, c0)
            if search():
                return True
            unplace(best_pi, changed)
        return False

    ok = search()
    assert ok, "connector assembly failed"
    return board.astype(int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered multicolor fragments into a left-right symmetric rectangle using connector pixels.

    Input structure:
    - The background is the most common color; each foreground piece is an 8-connected multicolor sprite.
    - One sprite is a pre-aligned LR-symmetric frame whose bbox already matches the output rectangle.
    - Small pixels of fixed connector colors appear on piece boundaries and act as docking markers.

    Rule:
    - Collect each non-background 8-connected component as one piece, preserving its full color pattern.
    - Infer the output rectangle size from total foreground area.
    - Place the frame at the output anchor, then pack the remaining pieces so the rectangle is exactly LR-symmetric and every inter-piece contact contains a matched connector pair.

    Key insight: shape fitting alone is no longer enough; the decisive cue is that adjacent pieces must meet through equal-colored connector pixels at the seam.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    comps = components_mask(g != bg, connectivity=8)
    assert comps, "grid must contain foreground pieces"
    pieces = [_normalize_piece(pts, g) for pts in comps]

    total_area = int(sum(piece["area"] for piece in pieces))
    assert total_area > 0, "total area must be positive"
    dims = [(h, w) for h, w in _factors(total_area) if h >= w]
    assert dims, "at least one portrait factorization must exist"
    best_gap = min(abs(h - w) for h, w in dims)
    best_dims = [(h, w) for h, w in dims if abs(h - w) == best_gap]
    assert len(best_dims) == 1, "closest-to-square portrait output shape must be unique"
    H, W = best_dims[0]

    frame_group = _find_frame_group(pieces, H, W)
    sol = _assemble(pieces, H, W, frame_group)
    return sol.astype(int)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

_OUT_H = 10
_OUT_W = 5

_BODY_COLORS_MIN = 4
_BODY_COLORS_MAX = 5
_PIECE_COUNT_MIN = 7
_PIECE_COUNT_MAX = 14
_MAX_PIECE_AREA = 6

_CANVAS_MIN = 13
_CANVAS_MAX = 15
_SCATTER_PADDING = 1
_GEN_MAX_RETRIES = 400


def _zigzag_frame(H: int) -> tuple[set[tuple[int, int]], int, int]:
    cells = set()
    start_c = random.randint(0, 1)
    c = start_c
    for r in range(H):
        cells.add((r, c))
        if r == H - 1:
            break
        move = random.choice(["down", "horiz", "diag"])
        if move == "down":
            pass
        elif move == "horiz":
            cells.add((r, 1 - c))
            c = 1 - c
        else:
            c = 1 - c
    return cells, start_c, c


def _paint_chunk(grid: np.ndarray, chunk: list[tuple[int, int]], color: int) -> None:
    W = grid.shape[1]
    for r, c in chunk:
        grid[int(r), int(c)] = int(color)
        grid[int(r), W - 1 - int(c)] = int(color)


def _adjacent_colors(grid: np.ndarray, chunk: list[tuple[int, int]], bg_marker: int = -1) -> set[int]:
    H, W = grid.shape
    colors = set()
    full = set()
    for r, c in chunk:
        full.add((int(r), int(c)))
        full.add((int(r), int(W - 1 - int(c))))
    for r, c in full:
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= rr < H and 0 <= cc < W:
                val = int(grid[rr, cc])
                if val != bg_marker and (rr, cc) not in full:
                    colors.add(val)
    return colors


def _fill_interior(grid: np.ndarray, fill_pool: list[int]) -> None:
    half_cells = {(r, c) for r in range(1, _OUT_H - 1) for c in range(1, _OUT_W // 2 + 1)}
    previous_color = None

    while half_cells:
        target = random.randint(1, min(3, len(half_cells)))
        start = random.choice(sorted(half_cells))
        chunk = [start]
        half_cells.remove(start)
        frontier = [start]
        while len(chunk) < target and frontier:
            cr, cc = random.choice(frontier)
            nbrs = []
            for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                if (nr, nc) in half_cells:
                    nbrs.append((nr, nc))
            if not nbrs:
                frontier.remove((cr, cc))
                continue
            nxt = random.choice(nbrs)
            half_cells.remove(nxt)
            chunk.append(nxt)
            frontier.append(nxt)

        avoid = _adjacent_colors(grid, chunk)
        if previous_color is not None:
            avoid.add(int(previous_color))
        opts = [col for col in fill_pool if col not in avoid]
        if not opts:
            opts = [col for col in fill_pool if col != previous_color] or fill_pool
        col = random.choice(opts)
        _paint_chunk(grid, chunk, int(col))
        previous_color = int(col)


def _build_body_output() -> tuple[np.ndarray, int] | tuple[None, None]:
    H, W = _OUT_H, _OUT_W
    n_body = random.randint(_BODY_COLORS_MIN, _BODY_COLORS_MAX)
    bg = random.choice([c for c in range(10) if c not in CONNECTOR_COLORS])
    pool = [c for c in range(10) if c not in CONNECTOR_COLORS and c != bg]
    random.shuffle(pool)
    if len(pool) < n_body:
        return None, None
    body_palette = pool[:n_body]
    frame_color = int(body_palette[0])
    fill_pool = [int(c) for c in body_palette[1:]]
    if not fill_pool:
        return None, None

    grid = np.full((H, W), -1, dtype=int)
    grid[0, :] = frame_color
    grid[H - 1, :] = frame_color
    grid[:, 0] = frame_color
    grid[:, W - 1] = frame_color
    _fill_interior(grid, fill_pool)
    if np.any(grid < 0):
        return None, None
    return grid.astype(int), int(bg)


def _build_structured_output() -> tuple[np.ndarray, int, np.ndarray, set[int]]:
    H, W = _OUT_H, _OUT_W
    bg = random.choice([c for c in range(10) if c not in CONNECTOR_COLORS])
    frame_color = random.choice([c for c in range(10) if c not in CONNECTOR_COLORS and c != bg])
    out = np.full((H, W), int(frame_color), dtype=int)
    pid_grid = np.zeros((H, W), dtype=int)

    # Border ring + two inner frame bars.
    bar_rows = {3, 6}
    for r in range(H):
        for c in range(W):
            if r in (0, H - 1) or c in (0, W - 1) or r in bar_rows:
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
        out[r0, W - 1] = top_col
        out[r0 + 1, W - 1] = bot_col

        for c in range(1, 4):
            out[r0, c] = top_col
            out[r0 + 1, c] = bot_col
            pid_grid[r0, c] = int(next_pid)
            pid_grid[r0 + 1, c] = int(next_pid)
            next_pid += 1

    assert np.all(out >= 0)
    return out.astype(int), int(bg), pid_grid.astype(int), {0}


def _piece_partition(body_out: np.ndarray) -> tuple[list[dict], np.ndarray]:
    H, W = body_out.shape
    pid_grid = np.full((H, W), -1, dtype=int)
    pieces = []
    for col in sorted(set(map(int, np.unique(body_out)))):
        for pts in components_mask(body_out == int(col), connectivity=8):
            r0, c0, r1, c1 = bbox_coords(pts)
            coords = [(int(r), int(c)) for r, c in pts]
            pid = len(pieces)
            for r, c in coords:
                pid_grid[int(r), int(c)] = int(pid)
            pieces.append(
                {
                    "pid": int(pid),
                    "color": int(col),
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(len(coords)),
                }
            )
    assert np.all(pid_grid >= 0)
    return pieces, pid_grid


def _find_frame_pids(pieces: list[dict], pid_grid: np.ndarray) -> list[int]:
    H, W = pid_grid.shape
    def group_ok(idxs: list[int]) -> bool:
        boxes = [pieces[i]["bbox"] for i in idxs]
        gr0 = min(int(b[0]) for b in boxes)
        gc0 = min(int(b[1]) for b in boxes)
        gr1 = max(int(b[2]) for b in boxes)
        gc1 = max(int(b[3]) for b in boxes)
        if int(gr1 - gr0 + 1) != H or int(gc1 - gc0 + 1) != W:
            return False
        coords = set()
        for i in idxs:
            for r, c in pieces[i]["coords"]:
                coords.add((int(r - gr0), int(c - gc0)))
        return _is_lr_symmetric_mask(coords, H, W)

    found: list[tuple[int, list[int]]] = []
    for i in range(len(pieces)):
        if group_ok([i]):
            found.append((int(pieces[i]["area"]), [int(i)]))
    for i in range(len(pieces)):
        for j in range(i + 1, len(pieces)):
            if group_ok([i, j]):
                area = int(pieces[i]["area"]) + int(pieces[j]["area"])
                found.append((area, [int(i), int(j)]))
    assert found, "generator expected at least one 1-2 piece frame group"
    best_area = min(area for area, _idxs in found)
    best = [idxs for area, idxs in found if area == best_area]
    assert len(best) == 1, "generator expected a unique minimum-area frame group"
    return best[0]


def _adjacency_map(pid_grid: np.ndarray) -> dict[tuple[int, int], list[tuple[tuple[int, int], tuple[int, int]]]]:
    H, W = pid_grid.shape
    adj: dict[tuple[int, int], list[tuple[tuple[int, int], tuple[int, int]]]] = defaultdict(list)
    for r in range(H):
        for c in range(W):
            p = int(pid_grid[r, c])
            if r + 1 < H and int(pid_grid[r + 1, c]) != p:
                q = int(pid_grid[r + 1, c])
                key = (min(p, q), max(p, q))
                a = (r, c) if p <= q else (r + 1, c)
                b = (r + 1, c) if p <= q else (r, c)
                adj[key].append((a, b))
            if c + 1 < W and int(pid_grid[r, c + 1]) != p:
                q = int(pid_grid[r, c + 1])
                key = (min(p, q), max(p, q))
                a = (r, c) if p <= q else (r, c + 1)
                b = (r, c + 1) if p <= q else (r, c)
                adj[key].append((a, b))
    return adj


def _mirror_coord(W: int, rc: tuple[int, int]) -> tuple[int, int]:
    r, c = rc
    return int(r), int(W - 1 - c)


def _apply_connectors(body_out: np.ndarray, pieces: list[dict], pid_grid: np.ndarray) -> tuple[np.ndarray | None, int]:
    H, W = body_out.shape
    out = np.array(body_out, copy=True)
    adj = _adjacency_map(pid_grid)
    used: set[tuple[int, int]] = set()
    visited: set[tuple[int, int]] = set()
    orbit_idx = 0

    for key in sorted(adj):
        if key in visited:
            continue
        pair = adj[key][0]
        mk = tuple(sorted((int(pid_grid[_mirror_coord(W, pair[0])]), int(pid_grid[_mirror_coord(W, pair[1])] ))))
        visited.add(key)
        visited.add(mk)

        candidates = []
        for a, b in adj[key]:
            leftness = min(int(a[1]), int(b[1]))
            candidates.append((leftness, int(a[0] + b[0]), a, b))
        candidates.sort()
        chosen = None
        for _leftness, _row_sum, a, b in candidates:
            if a not in used and b not in used:
                chosen = (a, b)
                break
        if chosen is None:
            chosen = (candidates[0][2], candidates[0][3])
        color = int(CONNECTOR_COLORS[orbit_idx % len(CONNECTOR_COLORS)])
        orbit_idx += 1

        coords_to_paint = [chosen[0], chosen[1]]
        ma = _mirror_coord(W, chosen[0])
        mb = _mirror_coord(W, chosen[1])
        coords_to_paint.extend([ma, mb])

        for r, c in coords_to_paint:
            out[int(r), int(c)] = color
            used.add((int(r), int(c)))

    # Every non-frame piece should have at least one connector; otherwise the connector
    # rule is visually weak and the sample is too close to the original task.
    frame_pids = set(_find_frame_pids(pieces, pid_grid))
    connector_counts = defaultdict(int)
    for piece in pieces:
        pid = int(piece["pid"])
        for r, c in piece["coords"]:
            if int(out[int(r), int(c)]) in CONNECTOR_COLORS:
                connector_counts[pid] += 1
    for piece in pieces:
        pid = int(piece["pid"])
        if pid in frame_pids:
            continue
        if connector_counts[pid] == 0:
            return None, 0
    return out.astype(int), int(len(adj))


def _build_patches(out: np.ndarray, pid_grid: np.ndarray, bg: int, frame_pids: set[int]) -> tuple[np.ndarray, list[np.ndarray]]:
    H, W = out.shape
    frame_mask = np.isin(pid_grid, sorted(frame_pids))
    frame_pts = np.argwhere(frame_mask)
    fr0, fc0, fr1, fc1 = bbox_coords(frame_pts)
    frame_patch = np.full((int(fr1 - fr0 + 1), int(fc1 - fc0 + 1)), int(bg), dtype=int)
    for r, c in frame_pts:
        frame_patch[int(r) - int(fr0), int(c) - int(fc0)] = int(out[int(r), int(c)])

    patches = []
    for pid in sorted(set(map(int, pid_grid.flatten()))):
        if pid in frame_pids:
            continue
        pts = np.argwhere(pid_grid == int(pid))
        r0, c0, r1, c1 = bbox_coords(pts)
        patch = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(bg), dtype=int)
        for r, c in pts:
            patch[int(r) - int(r0), int(c) - int(c0)] = int(out[int(r), int(c)])
        patches.append(patch)
    random.shuffle(patches)
    patches.sort(key=lambda p: (-(p != bg).sum(), random.random()))
    return frame_patch, patches


def _scatter(out: np.ndarray, pid_grid: np.ndarray, bg: int, frame_pids: set[int]) -> np.ndarray:
    rows = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    cols = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    canvas = np.full((rows, cols), int(bg), dtype=int)
    oh, ow = out.shape

    max_r = rows - oh - 2
    max_c = cols - ow - 2
    if max_r < 1 or max_c < 1:
        raise ValueError("frame cannot fit on canvas")
    off_r = random.randint(1, max_r)
    off_c = random.randint(1, max_c)

    frame_patch, patches = _build_patches(out, pid_grid, bg, frame_pids)
    canvas[off_r : off_r + oh, off_c : off_c + ow] = -1
    for patch in patches:
        r, c = random_free_location(
            canvas,
            patch,
            transparent=bg,
            padding=_SCATTER_PADDING,
            padding_connectivity=8,
        )
        paste(canvas, patch, r, c, transparent=bg)
    canvas[off_r : off_r + oh, off_c : off_c + ow] = frame_patch
    return canvas.astype(int)


def _piece_count(body_out: np.ndarray) -> int:
    return sum(len(components_mask(body_out == int(col), connectivity=4)) for col in np.unique(body_out))


def _gen_attempt(return_meta: bool = False):
    out, bg, pid_grid, frame_pids = _build_structured_output()
    pieces = sorted(set(map(int, pid_grid.flatten())))
    pc = len([pid for pid in pieces if pid >= 0])
    if not (_PIECE_COUNT_MIN <= pc <= _PIECE_COUNT_MAX):
        return None
    if not np.array_equal(out, out[:, ::-1]):
        return None

    non_frame = [pid for pid in pieces if pid not in frame_pids and pid >= 0]
    if any(int((pid_grid == pid).sum()) > _MAX_PIECE_AREA for pid in non_frame):
        return None

    connector_cells = int(np.isin(out, CONNECTOR_COLORS).sum())
    pieces_with_connectors = len(non_frame)
    seam_count = len(non_frame) + 6

    try:
        inp = _scatter(out, pid_grid, bg, frame_pids)
        got = main(inp)
    except Exception:
        return None
    if not np.array_equal(got, out):
        return None

    if return_meta:
        return {
            "input": inp.astype(int),
            "output": out.astype(int),
            "connector_cells": connector_cells,
            "seam_count": int(seam_count),
            "piece_count": int(pc),
            "pieces_with_connectors": int(pieces_with_connectors),
        }
    return inp.astype(int)


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid connector-matching e8686506 input")
