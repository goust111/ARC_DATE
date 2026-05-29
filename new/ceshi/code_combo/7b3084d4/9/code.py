# Combo variant of 7b3084d4 (index 9)
# combo_spec: D5/2+D14/0
# Name: 分隔排列输出+原位轮廓保留
#
# Design analysis
# a) Concrete behavior:
#    - Original task: solve the jigsaw by rotating the four scattered pieces so
#      they tile one tight 2N x 2N square; the marker pixel ends at (0,0).
#    - This combo still requires the same hidden assembly step, but the final
#      rendering changes in two ways:
#      1. The solved pieces are shown separated by 1-cell gaps in a 2x2 packed
#         display rather than touching in one tight rectangle.
#      2. The original source positions are not erased completely; only each
#         piece perimeter remains as a grey outline at the old location.
#    - Example at N=3:
#      the original solver would return a 6x6 assembled square.
#      Here we first recover that same 6x6 solution, then split it into the
#      four solved pieces ordered by solved row-major position
#      (TL, TR, BL, BR), render them at the top-left with 1-cell gaps, and
#      also draw grey outlines where the rotated input pieces originally were.
#
# b) Complexity calibration:
#    - The packed output is not a trivial "copy components in input order":
#      solver must still infer the correct solved orientation of every piece.
#    - The source outline uses exact piece perimeters instead of simple bbox
#      frames, so the old positions are visible but do not hide the moved
#      pieces with large solid blocks.
#    - N is capped at 4 in the generator so the packed 2x2 layout fits cleanly
#      inside 20x20 while still allowing varied tabbed shapes.
#
# c) Hidden assumptions broken from the original code:
#    - "One color == one piece" is no longer safe once the anchor contains the
#      grey marker, so extraction is done by 8-connected components.
#    - "Output size equals packed rectangle size" no longer holds; output is
#      always the full 20x20 canvas.
#    - "Returning the filled assembly board is enough" no longer holds; solver
#      must keep per-piece placements so it can re-render them in separated form.
#
# d) Placement legality:
#    - Generator builds the solved square first, derives the packed target
#      region, then scatters the rotated input pieces only on cells that do not
#      overlap that packed target and do not touch each other by 8-neighborhood.
#    - This guarantees no out-of-bounds writes, no input overlap, and visible
#      coexistence of moved pieces plus source outlines in the output.
#
# e) Visual observability:
#    - The packed pieces appear as four separated colored shapes at the top-left.
#    - Grey outlines preserve the exact original locations and orientations.
#    - The order of the separated display is structural and visible:
#      pieces are sorted by their solved position in the assembled square.
#
# f) Visual cleanliness:
#    - Only one outline layer is kept at old positions; interiors are left black.
#    - The moved pieces use solid original colors, so the two layers are easy
#      to distinguish even though both the marker and the outline use grey.
#
# g) New solver assumptions and how generator guarantees them:
#    - Solver assumes exactly one grey singleton marker in the input.
#      Generator always uses one marker and no other grey pixels.
#    - Solver assumes the four components tile one rectangle after rotation.
#      Generator still constructs examples from a valid solved ownership grid.
#    - Solver assumes the separated output order comes from solved row-major
#      placement; generator uses the same order when creating the oracle output.

from collections import deque
import random

import numpy as np

from arc_utils import get_bg as _bg_color, is_contiguous, neighbors8


BG = 0
OUTLINE_COLOR = 5
MARKER_COLOR = 5
ANCHOR_COLOR = 7
COLOR_POOL = [1, 2, 3, 4, 6, 8, 9]
CANVAS_H = 20
CANVAS_W = 20
MAX_ATTEMPTS = 500
N_MIN, N_MAX = 2, 4
TAB_NAMES = ["H_left", "H_right", "V_top", "V_bot"]
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


def _bbox(cells: list[tuple[int, int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _, _ in cells]
    cs = [c for _, c, _ in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _component_outline(abs_cells: list[tuple[int, int, int]]) -> list[tuple[int, int]]:
    pts = {(int(r), int(c)) for r, c, _ in abs_cells}
    out = []
    for r, c in sorted(pts):
        if any((r + dr, c + dc) not in pts for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))):
            out.append((r, c))
    return out


def _components(g: np.ndarray) -> tuple[list[dict], int]:
    bg = int(_bg_color(g))
    H, W = g.shape
    seen = np.zeros((H, W), dtype=bool)
    comps: list[dict] = []
    for sr, sc in np.argwhere(g != bg):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        abs_cells: list[tuple[int, int, int]] = []
        while q:
            r, c = q.popleft()
            abs_cells.append((int(r), int(c), int(g[r, c])))
            for nr, nc in neighbors8(r, c):
                if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and g[nr, nc] != bg:
                    seen[nr, nc] = True
                    q.append((int(nr), int(nc)))
        local = _normalize(abs_cells)
        comps.append(
            {
                "abs_cells": abs_cells,
                "cells": local,
                "bbox": _bbox(abs_cells),
            }
        )
    return comps, bg


def _extract_pieces(g: np.ndarray) -> tuple[list[dict], int]:
    comps, bg = _components(g)
    assert len(comps) == 4, "input must contain exactly four separated pieces"

    marker_pts = np.argwhere(g == MARKER_COLOR)
    assert len(marker_pts) == 1, "input must contain exactly one grey marker pixel"
    mr, mc = map(int, marker_pts[0])

    pieces: list[dict] = []
    anchor_count = 0
    for comp in comps:
        abs_pts = {(r, c) for r, c, _ in comp["abs_cells"]}
        piece = {
            "abs_cells": comp["abs_cells"],
            "cells": comp["cells"],
            "bbox": comp["bbox"],
            "name": "piece",
        }
        if (mr, mc) in abs_pts:
            anchor_count += 1
            piece["name"] = "anchor"
            piece["marker_color"] = MARKER_COLOR
        pieces.append(piece)
    assert anchor_count == 1, "marker must belong to exactly one piece"
    return pieces, bg


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


def _placeable(board: np.ndarray, placement: list[tuple[int, int, int]]) -> bool:
    H, W = board.shape
    for r, c, _ in placement:
        if not (0 <= r < H and 0 <= c < W) or board[r, c] != -1:
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
    ph = max_r + 1
    pw = max_c + 1
    outs: list[list[tuple[int, int, int]]] = []
    for tr in range(H - ph + 1):
        for tc in range(W - pw + 1):
            outs.append([(r + tr, c + tc, col) for r, c, col in variant])
    return outs


def _solve_packing(
    pieces: list[dict], H: int, W: int
) -> tuple[np.ndarray, dict[int, list[tuple[int, int, int]]]] | None:
    board = np.full((H, W), -1, dtype=int)
    piece_vars = [_variants(p["cells"]) for p in pieces]
    anchor_idx = next((i for i, p in enumerate(pieces) if p["name"] == "anchor"), None)
    assert anchor_idx is not None, "one anchor piece is required"

    chosen: dict[int, list[tuple[int, int, int]]] = {}

    def backtrack(rem: list[int]) -> bool:
        if not rem:
            return True
        best_pi = None
        best_opts = None
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
                best_opts = opts
                if cnt == 1:
                    break

        assert best_pi is not None and best_opts is not None
        next_rem = [x for x in rem if x != best_pi]
        for placement in best_opts:
            _apply(board, placement)
            chosen[best_pi] = placement
            if backtrack(next_rem):
                return True
            chosen.pop(best_pi, None)
            _apply(board, placement, -1)
        return False

    remaining = [i for i in range(len(pieces)) if i != anchor_idx]
    marker_col = int(pieces[anchor_idx]["marker_color"])
    for var in piece_vars[anchor_idx]:
        marker_pts = [(r, c) for r, c, col in var if col == marker_col]
        if len(marker_pts) != 1:
            continue
        mr, mc = marker_pts[0]
        anchor_place = [(r - mr, c - mc, col) for r, c, col in var]
        if not _placeable(board, anchor_place):
            continue
        _apply(board, anchor_place)
        chosen[anchor_idx] = anchor_place
        if backtrack(remaining):
            return board.copy(), {k: list(v) for k, v in chosen.items()}
        chosen.pop(anchor_idx, None)
        _apply(board, anchor_place, -1)
    return None


def _sorted_solved_placements(
    placements: dict[int, list[tuple[int, int, int]]]
) -> list[list[tuple[int, int, int]]]:
    solved = list(placements.values())
    solved.sort(key=lambda cellset: _bbox(cellset)[:2])
    return solved


def _packed_positions(
    solved_pieces: list[list[tuple[int, int, int]]]
) -> list[list[tuple[int, int, int]]]:
    assert len(solved_pieces) == 4, "packed rendering expects four solved pieces"
    local = [_normalize(piece) for piece in solved_pieces]
    heights = [max(r for r, _, _ in piece) + 1 for piece in local]
    widths = [max(c for _, c, _ in piece) + 1 for piece in local]

    row0_h = max(heights[0], heights[1])
    top_left_w = widths[0]
    bottom_left_w = widths[2]
    offsets = [
        (0, 0),
        (0, top_left_w + 1),
        (row0_h + 1, 0),
        (row0_h + 1, bottom_left_w + 1),
    ]

    placed: list[list[tuple[int, int, int]]] = []
    for piece, (dr, dc) in zip(local, offsets):
        placed.append([(r + dr, c + dc, col) for r, c, col in piece])

    max_r = max(r for piece in placed for r, _, _ in piece)
    max_c = max(c for piece in placed for _, c, _ in piece)
    assert max_r < CANVAS_H and max_c < CANVAS_W, "packed layout must fit in 20x20"
    return placed


def _render_output(
    shape: tuple[int, int],
    bg: int,
    source_abs_cells: list[list[tuple[int, int, int]]],
    solved_placements: dict[int, list[tuple[int, int, int]]],
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)

    for abs_cells in source_abs_cells:
        for r, c in _component_outline(abs_cells):
            out[r, c] = OUTLINE_COLOR

    packed = _packed_positions(_sorted_solved_placements(solved_placements))
    for piece in packed:
        for r, c, col in piece:
            out[r, c] = int(col)
    return out


def _solve_combo(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    pieces, bg = _extract_pieces(g)
    area = sum(len(p["cells"]) for p in pieces)
    for H, W in _dims(area):
        solved = _solve_packing(pieces, H, W)
        if solved is None:
            continue
        _board, placements = solved
        return _render_output(g.shape, bg, [p["abs_cells"] for p in pieces], placements)
    raise AssertionError("packing failed for all candidate dimensions")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble the four scattered puzzle pieces, then show the solved parts separated while keeping source outlines.

    Input structure:
    - The input is a 20x20 grid with exactly four 8-connected pieces on black background.
    - One piece is the anchor: it contains a unique grey marker pixel at one corner, adjacent only to that piece.
    - The other three pieces are monochrome tabbed regions from the same hidden 2x2 puzzle partition.

    Rule:
    - Recover the original tiling exactly as in the base task: rotate the pieces so they fill one rectangle and force the anchor marker to land at the assembled top-left corner.
    - Do not output that tight assembly directly. Instead, take the four solved pieces in solved row-major order and redraw them as a separated 2x2 packed layout with 1-cell gaps, starting at the global top-left.
    - Also keep the original locations visible by drawing only a grey perimeter outline of each input piece at its source position on the 20x20 canvas.

    Key insight:
    - The grey pixel still fixes the anchor orientation, so the hidden solved square is recoverable first; the combo only changes the final rendering into "moved colored pieces + ghost outlines of where they came from".
    """
    return _solve_combo(np.asarray(grid, dtype=int))


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
        if mask.sum() == 0:
            return None
        if not is_contiguous(mask, background=0, connectivity=8):
            return None
    return own


def _extract_piece_from_solution(
    own: np.ndarray,
    q: int,
    color: int,
    marker_pos: tuple[int, int] | None = None,
) -> list[tuple[int, int, int]]:
    rows, cols = np.where(own == q)
    min_r = int(rows.min())
    min_c = int(cols.min())
    cells = [(int(r) - min_r, int(c) - min_c, int(color)) for r, c in zip(rows, cols)]
    if marker_pos is not None:
        mr = int(marker_pos[0] - min_r)
        mc = int(marker_pos[1] - min_c)
        cells = [(r, c, MARKER_COLOR if (r, c) == (mr, mc) else col) for r, c, col in cells]
    return cells


def _solution_placements(
    own: np.ndarray, colors: dict[int, int]
) -> list[list[tuple[int, int, int]]]:
    out: list[list[tuple[int, int, int]]] = []
    for q in (TL, TR, BL, BR):
        cells: list[tuple[int, int, int]] = []
        for r, c in np.argwhere(own == q):
            rr = int(r)
            cc = int(c)
            col = MARKER_COLOR if q == TL and (rr, cc) == (0, 0) else int(colors[q])
            cells.append((rr, cc, col))
        out.append(cells)
    return out


def _random_tab_specs(N: int) -> dict:
    specs = {}
    for tn in TAB_NAMES:
        depth = random.randint(1, N - 1)
        if tn.startswith("H"):
            direction = random.choice(["up", "down"])
        else:
            direction = random.choice(["left", "right"])
        specs[tn] = (depth, direction)
    if N == 2:
        specs[random.choice(TAB_NAMES)] = None
    return specs


def _occupied_mask(cell_groups: list[list[tuple[int, int, int]]], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for cells in cell_groups:
        for r, c, _ in cells:
            mask[r, c] = True
    return mask


def _expand_mask(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Reserve a small moat so source outlines stay visually separate from the packed result."""
    H, W = mask.shape
    expanded = mask.copy()
    pts = np.argwhere(mask)
    for r0, c0 in pts:
        rr = int(r0)
        cc = int(c0)
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < H and 0 <= nc < W:
                    expanded[nr, nc] = True
    return expanded


def _can_place_source_piece(
    cells: list[tuple[int, int, int]],
    tr: int,
    tc: int,
    occupied: set[tuple[int, int]],
    blocked: np.ndarray,
) -> bool:
    H, W = blocked.shape
    shifted = [(r + tr, c + tc) for r, c, _ in cells]
    for r, c in shifted:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if blocked[r, c] or (r, c) in occupied:
            return False
        for nr, nc in neighbors8(r, c):
            if (nr, nc) in occupied:
                return False
    return True


def _scatter_source_pieces(
    pieces: list[list[tuple[int, int, int]]],
    blocked: np.ndarray,
) -> list[list[tuple[int, int, int]]] | None:
    H, W = blocked.shape
    bboxes = []
    for cells in pieces:
        max_r = max(r for r, _, _ in cells)
        max_c = max(c for _, c, _ in cells)
        bboxes.append((max_r + 1, max_c + 1))

    candidates: dict[int, list[tuple[int, int]]] = {}
    for i, (cells, (ph, pw)) in enumerate(zip(pieces, bboxes)):
        opts = []
        for tr in range(H - ph + 1):
            for tc in range(W - pw + 1):
                if _can_place_source_piece(cells, tr, tc, set(), blocked):
                    opts.append((tr, tc))
        random.shuffle(opts)
        candidates[i] = opts
        if not opts:
            return None

    order = sorted(range(len(pieces)), key=lambda i: len(candidates[i]))
    occupied: set[tuple[int, int]] = set()
    placed: dict[int, list[tuple[int, int, int]]] = {}

    def backtrack(k: int) -> bool:
        if k == len(order):
            return True
        pi = order[k]
        for tr, tc in candidates[pi]:
            if not _can_place_source_piece(pieces[pi], tr, tc, occupied, blocked):
                continue
            shifted = [(r + tr, c + tc, col) for r, c, col in pieces[pi]]
            placed[pi] = shifted
            for r, c, _ in shifted:
                occupied.add((r, c))
            if backtrack(k + 1):
                return True
            for r, c, _ in shifted:
                occupied.remove((r, c))
            placed.pop(pi, None)
        return False

    if not backtrack(0):
        return None
    return [placed[i] for i in range(len(pieces))]


def _gen_attempt() -> np.ndarray | None:
    N = random.randint(N_MIN, N_MAX)
    p = random.randint(1, N - 1)
    own = _build_ownership(N, p, _random_tab_specs(N))
    if own is None:
        return None

    pool = COLOR_POOL.copy()
    random.shuffle(pool)
    colors = {TL: ANCHOR_COLOR, TR: pool[0], BL: pool[1], BR: pool[2]}

    solved_abs = _solution_placements(own, colors)
    solved_map = {i: piece for i, piece in enumerate(solved_abs)}
    packed = _packed_positions(_sorted_solved_placements(solved_map))
    blocked = _expand_mask(_occupied_mask(packed, (CANVAS_H, CANVAS_W)))

    anchor_cells = _extract_piece_from_solution(own, TL, ANCHOR_COLOR, marker_pos=(0, 0))
    other_cells = [
        _extract_piece_from_solution(own, TR, colors[TR]),
        _extract_piece_from_solution(own, BL, colors[BL]),
        _extract_piece_from_solution(own, BR, colors[BR]),
    ]
    source_local = [_rotate_cells(anchor_cells, random.randint(0, 3))]
    source_local.extend(_rotate_cells(cells, random.randint(0, 3)) for cells in other_cells)

    source_abs = _scatter_source_pieces(source_local, blocked)
    if source_abs is None:
        return None

    canvas = np.zeros((CANVAS_H, CANVAS_W), dtype=int)
    for cells in source_abs:
        for r, c, col in cells:
            canvas[r, c] = int(col)

    expected = _render_output(canvas.shape, BG, source_abs, solved_map)
    try:
        actual = main(canvas)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: exhausted attempts")
