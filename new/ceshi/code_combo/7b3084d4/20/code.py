# Combo variant of 7b3084d4 (index 20)
# combo_spec: D5/0+D11/1+D14/0
# Name: 部分重叠输入+原位轮廓+保留输入网格

# Design analysis
# a) Concrete behavior:
#    Original: four separated colored pieces are scattered on a 20x20 canvas,
#    then packed into a tight rectangle whose top-left cell is the gray marker.
#    Fused variant: the packed rectangle is still solved the same way, but the
#    output stays 20x20, the assembled rectangle is pasted at the canvas
#    top-left, and each source piece leaves a gray silhouette-outline at its
#    original location.  A small example is N=2: the original output would be a
#    4x4 crop; here that same 4x4 block sits in rows/cols 0..3 of a 20x20
#    output, while the old scattered pieces are replaced by gray boundary cells.
#    For D11/1, at least one input pair has overlapping tight bounding boxes,
#    but their colored cells remain disjoint.
# b) Complexity calibration:
#    Real pixel occlusion would hide information and make 2-4 examples
#    underdetermined.  To keep the overlap visually obvious but still inferable,
#    the generator enforces bounding-box overlap rather than hidden-color
#    overwrite.  The source-outline effect is therefore visible without creating
#    ambiguous missing pixels.
# c) Broken original assumptions:
#    - `main()` can no longer return a tight crop; it must preserve 20x20.
#    - The solver must keep absolute source coordinates because output includes
#      source traces.
#    - The generator can no longer rely on padded non-overlap; it must search
#      for at least one bbox-overlap layout while still preventing real cell
#      collisions and avoiding the reserved assembly corner.
# d) Placement legality:
#    Generator constraints are explicit: every piece stays in-bounds, no two
#    colored cells overlap, at least one bbox pair overlaps, and source bboxes
#    avoid the reserved top-left assembly rectangle.
# e) Visual observability:
#    Humans can see three separate cues across examples:
#    - output size stays 20x20;
#    - the solved rectangle is always anchored at the global top-left;
#    - gray source outlines mark the old locations.
#    The bbox overlap is also visible because two colored piece rectangles
#    visually intrude into the same region in the input.
# f) Cleanliness:
#    The assembled block is reserved at the top-left and inputs are placed away
#    from that corner, so gray ghosts do not bury the main answer.
# g) New implicit assumptions:
#    The solver still assumes one uniquely colored anchor piece plus one gray
#    singleton marker in the input.  The generator preserves exactly that.

import random

import numpy as np

from arc_utils import get_bg as _bg_color, is_contiguous, neighbors8


N_MIN, N_MAX = 2, 5
ANCHOR_COLOR = 7
MARKER_COLOR = 5
COLOR_POOL = [1, 2, 3, 4, 6, 8, 9]
CANVAS_H, CANVAS_W = 20, 20
MAX_ATTEMPTS = 500
TAB_NAMES = ["H_left", "H_right", "V_top", "V_bot"]
MIN_BBOX_OVERLAP_AREA = 2

TL, TR, BL, BR = 0, 1, 2, 3


def _normalize(cells: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    min_r = min(r for r, _, _ in cells)
    min_c = min(c for _, c, _ in cells)
    out = [(r - min_r, c - min_c, col) for r, c, col in cells]
    out.sort()
    return out


def _rotate_cells(cells: list[tuple[int, int, int]], k: int) -> list[tuple[int, int, int]]:
    pts = list(cells)
    for _ in range(k % 4):
        pts = [(c, -r, col) for r, c, col in pts]
    return _normalize(pts)


def _variants(cells: list[tuple[int, int, int]]) -> list[list[tuple[int, int, int]]]:
    seen: set[tuple[tuple[int, int, int], ...]] = set()
    outs: list[list[tuple[int, int, int]]] = []
    for k in range(4):
        rot = _rotate_cells(cells, k)
        key = tuple(rot)
        if key not in seen:
            seen.add(key)
            outs.append(rot)
    return outs


def _dims(area: int) -> list[tuple[int, int]]:
    pairs = []
    for h in range(1, int(area**0.5) + 1):
        if area % h == 0:
            w = area // h
            pairs.append((h, w))
            if h != w:
                pairs.append((w, h))
    pairs.sort(key=lambda t: (abs(t[0] - t[1]), t[0], t[1]))
    return pairs


def _find_marker_color(g: np.ndarray, bg: int) -> int:
    colors = [c for c in sorted(set(map(int, g.ravel()))) if c != bg]
    counts = {c: int(np.sum(g == c)) for c in colors}
    for c in colors:
        if counts[c] != 1:
            continue
        r, col = map(int, np.argwhere(g == c)[0])
        if any(0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and g[nr, nc] != bg for nr, nc in neighbors8(r, col)):
            return c
    raise AssertionError("input must contain one adjacent singleton marker")


def _extract_pieces(g: np.ndarray) -> tuple[list[dict], int]:
    bg = _bg_color(g)
    colors = [c for c in sorted(set(map(int, g.ravel()))) if c != bg]
    if not colors:
        return [], bg

    marker_color = _find_marker_color(g, bg)
    marker_pos = tuple(map(int, np.argwhere(g == marker_color)[0]))

    mr, mc = marker_pos
    anchor_neighbors = {
        int(g[nr, nc])
        for nr, nc in neighbors8(mr, mc)
        if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and g[nr, nc] not in (bg, marker_color)
    }
    assert len(anchor_neighbors) == 1, "marker must touch exactly one anchor color"
    anchor_color = next(iter(anchor_neighbors))

    pieces: list[dict] = []
    used = {marker_color, anchor_color}

    anchor_abs = [(int(r), int(c), int(anchor_color)) for r, c in np.argwhere(g == anchor_color)]
    anchor_abs.append((mr, mc, int(marker_color)))
    pieces.append(
        {
            "name": "anchor",
            "marker_color": int(marker_color),
            "abs_cells": anchor_abs,
            "cells": _normalize(anchor_abs),
        }
    )

    for col in colors:
        if col in used:
            continue
        abs_cells = [(int(r), int(c), int(col)) for r, c in np.argwhere(g == col)]
        pieces.append(
            {
                "name": f"c{col}",
                "abs_cells": abs_cells,
                "cells": _normalize(abs_cells),
            }
        )

    return pieces, int(bg)


def _placeable(board: np.ndarray, placement: list[tuple[int, int, int]]) -> bool:
    for r, c, _ in placement:
        if board[r, c] != -1:
            return False
    return True


def _apply(board: np.ndarray, placement: list[tuple[int, int, int]], val: int | None = None) -> None:
    if val is None:
        for r, c, col in placement:
            board[r, c] = col
    else:
        for r, c, _ in placement:
            board[r, c] = val


def _all_placements_for_variant(
    variant: list[tuple[int, int, int]], H: int, W: int
) -> list[list[tuple[int, int, int]]]:
    max_r = max(r for r, _, _ in variant)
    max_c = max(c for _, c, _ in variant)
    ph, pw = max_r + 1, max_c + 1
    outs: list[list[tuple[int, int, int]]] = []
    for tr in range(H - ph + 1):
        for tc in range(W - pw + 1):
            outs.append([(r + tr, c + tc, col) for r, c, col in variant])
    return outs


def _solve_packing(pieces: list[dict], H: int, W: int) -> np.ndarray | None:
    board = np.full((H, W), -1, dtype=int)
    piece_vars = [_variants(p["cells"]) for p in pieces]

    anchor_idx = next((i for i, p in enumerate(pieces) if p["name"] == "anchor"), None)
    remaining = list(range(len(pieces)))

    def backtrack(rem: list[int]) -> bool:
        if not rem:
            return True

        best_pi = None
        best_options = None
        best_count = None
        for pi in rem:
            opts = []
            for var in piece_vars[pi]:
                for placement in _all_placements_for_variant(var, H, W):
                    if _placeable(board, placement):
                        opts.append(placement)
            cnt = len(opts)
            if cnt == 0:
                return False
            if best_count is None or cnt < best_count:
                best_count = cnt
                best_pi = pi
                best_options = opts
                if cnt == 1:
                    break

        assert best_pi is not None and best_options is not None
        next_rem = [x for x in rem if x != best_pi]
        for placement in best_options:
            _apply(board, placement)
            if backtrack(next_rem):
                return True
            _apply(board, placement, -1)
        return False

    if anchor_idx is None:
        return board if backtrack(remaining) else None

    remaining.remove(anchor_idx)
    marker_col = pieces[anchor_idx]["marker_color"]
    for var in piece_vars[anchor_idx]:
        marker_pts = [(r, c) for r, c, col in var if col == marker_col]
        if len(marker_pts) != 1:
            continue
        mr, mc = marker_pts[0]
        placed = [(r - mr, c - mc, col) for r, c, col in var]
        if not all(0 <= r < H and 0 <= c < W for r, c, _ in placed):
            continue
        if not _placeable(board, placed):
            continue
        _apply(board, placed)
        if backtrack(remaining):
            return board
        _apply(board, placed, -1)
    return None


def _outline_cells(abs_cells: list[tuple[int, int, int]]) -> list[tuple[int, int]]:
    pts = {(int(r), int(c)) for r, c, _ in abs_cells}
    out: list[tuple[int, int]] = []
    for r, c in sorted(pts):
        if any((r + dr, c + dc) not in pts for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))):
            out.append((r, c))
    return out


def _render_full_output(
    solved: np.ndarray, pieces: list[dict], bg: int, shape: tuple[int, int], outline_color: int
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for piece in pieces:
        for r, c in _outline_cells(piece["abs_cells"]):
            out[r, c] = int(outline_color)
    H, W = solved.shape
    out[:H, :W] = solved
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble the scattered puzzle pieces, keep the full canvas, and leave gray source outlines behind.

    Input structure:
    - The background is the most common color on a 20x20 grid.
    - Four colored pieces are present; one anchor piece contains a single gray marker pixel adjacent to its main color.
    - Piece bounding boxes may overlap each other, but the colored cells themselves remain distinct.

    Rule:
    - Recover the four piece shapes, rotate them as needed, and pack them into one solid rectangle exactly as in the base task.
    - The anchor is oriented so its gray marker lands at global output coordinate (0,0).
    - Return a 20x20 grid: paste the assembled rectangle at the top-left and replace each original piece position with a gray outline of that piece's silhouette.

    Key insight:
    - The packing logic is unchanged; the new work is preserving absolute source coordinates so the answer can show both the solved assembly and the source-location ghosts on the original canvas.
    """
    g = np.asarray(grid, dtype=int)
    pieces, bg = _extract_pieces(g)
    assert pieces, "grid must contain colored pieces"
    outline_color = next(p["marker_color"] for p in pieces if p["name"] == "anchor")

    area = sum(len(p["cells"]) for p in pieces)
    for H, W in _dims(area):
        solved = _solve_packing(pieces, H, W)
        if solved is not None:
            return _render_full_output(solved.astype(int), pieces, bg, g.shape, outline_color)
    raise AssertionError("packing failed for all candidate dimensions")


def _build_ownership(N: int, p: int, tab_specs: dict) -> np.ndarray | None:
    S = 2 * N
    own = np.zeros((S, S), dtype=int)
    own[:N, :N] = TL
    own[:N, N:] = TR
    own[N:, :N] = BL
    own[N:, N:] = BR

    for kind, col, up_owner, down_owner in [
        ("H_left", p, BL, TL),
        ("H_right", S - 1 - p, BR, TR),
    ]:
        spec = tab_specs.get(kind)
        if spec is None:
            continue
        depth, direction = spec
        if direction == "up":
            own[N - depth : N, col] = up_owner
        else:
            own[N : N + depth, col] = down_owner

    for kind, row, left_owner, right_owner in [
        ("V_top", p, TR, TL),
        ("V_bot", S - 1 - p, BR, BL),
    ]:
        spec = tab_specs.get(kind)
        if spec is None:
            continue
        depth, direction = spec
        if direction == "left":
            own[row, N - depth : N] = left_owner
        else:
            own[row, N : N + depth] = right_owner

    for q in (TL, TR, BL, BR):
        mask = (own == q).astype(int)
        if mask.sum() == 0 or not is_contiguous(mask, background=0, connectivity=8):
            return None
    return own


def _extract_piece(
    own: np.ndarray, q: int, color: int, marker_pos: tuple[int, int] | None = None
) -> list[tuple[int, int, int]]:
    rows, cols = np.where(own == q)
    min_r, min_c = int(rows.min()), int(cols.min())
    cells = [(int(r) - min_r, int(c) - min_c, int(color)) for r, c in zip(rows, cols)]
    if marker_pos is None:
        return cells
    mr, mc = int(marker_pos[0] - min_r), int(marker_pos[1] - min_c)
    return [(r, c, MARKER_COLOR if (r, c) == (mr, mc) else col) for r, c, col in cells]


def _bbox(cells: list[tuple[int, int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _, _ in cells]
    cs = [c for _, c, _ in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _translate(cells: list[tuple[int, int, int]], dr: int, dc: int) -> list[tuple[int, int, int]]:
    return [(r + dr, c + dc, col) for r, c, col in cells]


def _bbox_overlap_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return 0
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _bbox_intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return _bbox_overlap_area(a, b) > 0


def _inside_reserve(bbox: tuple[int, int, int, int], reserve: tuple[int, int, int, int]) -> bool:
    return _bbox_intersects(bbox, reserve)


def _candidate_placements(
    cells: list[tuple[int, int, int]], reserve: tuple[int, int, int, int]
) -> list[dict]:
    max_r = max(r for r, _, _ in cells)
    max_c = max(c for _, c, _ in cells)
    out = []
    for dr in range(CANVAS_H - max_r):
        for dc in range(CANVAS_W - max_c):
            abs_cells = _translate(cells, dr, dc)
            bbox = _bbox(abs_cells)
            if _inside_reserve(bbox, reserve):
                continue
            out.append({"abs_cells": abs_cells, "bbox": bbox, "mask": _cell_mask(abs_cells)})
    random.shuffle(out)
    return out


def _cell_mask(abs_cells: list[tuple[int, int, int]]) -> np.ndarray:
    mask = np.zeros((CANVAS_H, CANVAS_W), dtype=bool)
    for r, c, _ in abs_cells:
        mask[r, c] = True
    return mask


def _layout_pieces(pieces: list[list[tuple[int, int, int]]], reserve: tuple[int, int, int, int]) -> list[list[tuple[int, int, int]]] | None:
    indices = list(range(len(pieces)))
    candidates = {i: _candidate_placements(pieces[i], reserve) for i in indices}
    if any(len(candidates[i]) == 0 for i in indices):
        return None

    pair_indices = [(i, j) for i in indices for j in indices if i < j]
    random.shuffle(pair_indices)
    fill_order = sorted(indices, key=lambda i: -len(pieces[i]))

    for i, j in pair_indices:
        pool_i = candidates[i][:80]
        pool_j = candidates[j][:80]
        for cand_i in pool_i:
            for cand_j in pool_j:
                overlap_area = _bbox_overlap_area(cand_i["bbox"], cand_j["bbox"])
                if overlap_area < MIN_BBOX_OVERLAP_AREA:
                    continue
                if np.any(cand_i["mask"] & cand_j["mask"]):
                    continue

                placed = {i: cand_i, j: cand_j}
                occupied = cand_i["mask"] | cand_j["mask"]
                success = True

                for k in fill_order:
                    if k in placed:
                        continue
                    found = None
                    for cand_k in candidates[k][:120]:
                        if np.any(occupied & cand_k["mask"]):
                            continue
                        if any(_bbox_intersects(cand_k["bbox"], placed[m]["bbox"]) for m in placed):
                            continue
                        found = cand_k
                        break
                    if found is None:
                        success = False
                        break
                    placed[k] = found
                    occupied = occupied | found["mask"]

                if success:
                    return [placed[k]["abs_cells"] for k in indices]

    return None


def _random_tab_specs(N: int) -> dict:
    specs = {}
    for tn in TAB_NAMES:
        depth = random.randint(1, N - 1)
        direction = random.choice(["up", "down"] if tn.startswith("H") else ["left", "right"])
        specs[tn] = (depth, direction)
    if N == 2:
        specs[random.choice(TAB_NAMES)] = None
    return specs


def _build_case() -> tuple[np.ndarray, list[list[tuple[int, int, int]]], np.ndarray] | None:
    N = random.randint(N_MIN, N_MAX)
    p = random.randint(1, N - 1)
    own = _build_ownership(N, p, _random_tab_specs(N))
    if own is None:
        return None

    pool = COLOR_POOL.copy()
    random.shuffle(pool)
    quad_colors = {TL: ANCHOR_COLOR, TR: pool[0], BL: pool[1], BR: pool[2]}

    S = 2 * N
    tight = np.zeros((S, S), dtype=int)
    for q, col in quad_colors.items():
        tight[own == q] = col
    tight[0, 0] = MARKER_COLOR

    anchor_cells = _extract_piece(own, TL, ANCHOR_COLOR, marker_pos=(0, 0))
    other_cells = [_extract_piece(own, q, quad_colors[q]) for q in (TR, BL, BR)]
    all_cells = [_rotate_cells(anchor_cells, random.randint(0, 3))] + [
        _rotate_cells(cells, random.randint(0, 3)) for cells in other_cells
    ]

    reserve = (0, 0, S, S)
    placed = _layout_pieces(all_cells, reserve)
    if placed is None:
        return None

    canvas = np.zeros((CANVAS_H, CANVAS_W), dtype=int)
    for abs_cells in placed:
        for r, c, col in abs_cells:
            canvas[r, c] = col

    return canvas, placed, tight


def _expected_output(tight: np.ndarray, placed: list[list[tuple[int, int, int]]]) -> np.ndarray:
    pieces = [{"abs_cells": abs_cells} for abs_cells in placed]
    return _render_full_output(tight, pieces, 0, (CANVAS_H, CANVAS_W), MARKER_COLOR)


def _has_bbox_overlap(placed: list[list[tuple[int, int, int]]]) -> bool:
    bboxes = [_bbox(cells) for cells in placed]
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            if _bbox_overlap_area(bboxes[i], bboxes[j]) >= MIN_BBOX_OVERLAP_AREA:
                return True
    return False


def _gen_attempt() -> np.ndarray | None:
    built = _build_case()
    if built is None:
        return None
    canvas, placed, tight = built
    if not _has_bbox_overlap(placed):
        return None
    expected = _expected_output(tight, placed)
    try:
        out = main(canvas)
    except Exception:
        return None
    if not np.array_equal(out, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: exhausted attempts")
