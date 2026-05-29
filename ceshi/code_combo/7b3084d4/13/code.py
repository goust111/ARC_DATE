# Combo variant of 7b3084d4 (index 13)
# combo_spec: D3/3+D5/1+D14/2
# Name: 带边框输出+背景边距+区域外填充

import random
import numpy as np

from arc_utils import get_bg as _bg_color, neighbors8
from arc_utils import random_free_location, is_contiguous


# ---------------------------------------------------------------------------
# Design analysis for the fused variant
# ---------------------------------------------------------------------------
# a) Concrete behavior:
#    Original task stops after packing the four pieces into a tight 2N x 2N
#    square whose top-left cell is the marker. The fused task adds three layers
#    in one coordinate system:
#      1. pack the content square;
#      2. wrap it with a 1-cell border;
#      3. wrap that border with a 1-cell background margin;
#      4. place the whole framed object into a fixed 20 x 20 canvas and fill
#         every remaining cell with a dedicated outer-fill color.
#    Example with N=2:
#      - original output size: 4 x 4 content only;
#      - fused output size: always 20 x 20;
#      - content occupies rows/cols 3..6, border 2..7, bg margin 1..8, and all
#        cells outside 1..8 are the outer-fill color.
#
# b) Complexity calibration:
#    The added structure is visually obvious but low-entropy: border width = 1,
#    margin width = 1, placement offset = 1, border color and fill color are
#    fixed across samples. This avoids a hidden combinatorial rule while keeping
#    the fused output clearly different from the original tight crop.
#
# c) Broken original assumptions:
#    - Original `main()` assumes the output is exactly the packed rectangle.
#      No longer valid: the solver must decorate after packing.
#    - Original output size depends only on packed area. No longer valid:
#      final output is fixed to 20 x 20, so content must fit inside a decorated
#      envelope of size (2N + 4) x (2N + 4).
#    - Original solver can return immediately after packing. No longer valid:
#      a rendering step is required after the rectangle is found.
#
# d) Placement legality:
#    With N in [2, 7], the packed content side is S = 2N in [4, 14]. The
#    decorated footprint is S + 4 in [8, 18], so it always fits inside 20 x 20
#    when its top-left corner is fixed at (1, 1). Generator-side scattering of
#    input pieces still enforces non-overlap and a one-cell gap.
#
# e) Visual observability:
#    A human can infer the fusion from a few examples because all outputs show
#    the same layer order: content, border ring, black margin ring, then a
#    uniform outer region. The object size still changes with N, so the rule is
#    not reducible to painting a static frame.
#
# f) Visual cleanliness:
#    The three extra regions are disjoint and nested. Border/fill colors are
#    excluded from the input piece palette, so no output layer visually merges
#    with a puzzle piece.
#
# g) New solver assumptions audited against the generator:
#    - exactly one singleton marker exists and is 8-neigh adjacent to one piece;
#    - the four pieces tile a square whose side is at most 14;
#    - the fixed decoration colors do not appear in input pieces.
#    The generator constructs all three guarantees explicitly.


MARKER_COLOR = 5
ANCHOR_COLOR = 7
BORDER_COLOR = 4
OUTER_FILL_COLOR = 8
INPUT_BG = 0
OUTPUT_H = OUTPUT_W = 20
FRAME_OFFSET = 1
BG_MARGIN = 1
BORDER_WIDTH = 1

N_MIN, N_MAX = 2, 7
COLOR_POOL = [1, 2, 3, 6, 9]
CANVAS_H, CANVAS_W = 20, 20
PLACEMENT_PADDING = 1
MAX_ATTEMPTS = 300
TAB_NAMES = ["H_left", "H_right", "V_top", "V_bot"]

TL, TR, BL, BR = 0, 1, 2, 3


def _rotate_cells(cells: list[tuple[int, int, int]], k: int) -> list[tuple[int, int, int]]:
    pts = [(r, c, col) for r, c, col in cells]
    for _ in range(k % 4):
        pts = [(c, -r, col) for r, c, col in pts]
    min_r = min(r for r, _, _ in pts)
    min_c = min(c for _, c, _ in pts)
    pts = [(r - min_r, c - min_c, col) for r, c, col in pts]
    pts.sort()
    return pts


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


def _find_marker_color(g: np.ndarray, bg: int) -> int | None:
    colors = [c for c in sorted(set(map(int, g.ravel()))) if c != bg]
    counts = {c: int(np.sum(g == c)) for c in colors}
    for c in colors:
        if counts[c] != 1:
            continue
        r, col = map(int, np.argwhere(g == c)[0])
        for nr, nc in neighbors8(r, col):
            if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and g[nr, nc] != bg:
                return c
    return None


def _extract_pieces(g: np.ndarray) -> tuple[list[dict], int]:
    bg = _bg_color(g)
    colors = [c for c in sorted(set(map(int, g.ravel()))) if c != bg]
    if not colors:
        return [], bg

    marker_color = _find_marker_color(g, bg)
    assert marker_color is not None, "input must contain one adjacent singleton marker"
    marker_pos = tuple(map(int, np.argwhere(g == marker_color)[0]))

    anchor_color = None
    mr, mc = marker_pos
    neigh = set()
    for nr, nc in neighbors8(mr, mc):
        if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]:
            v = int(g[nr, nc])
            if v != bg and v != marker_color:
                neigh.add(v)
    if len(neigh) == 1:
        anchor_color = next(iter(neigh))
    assert anchor_color is not None, "marker must touch exactly one anchor color"

    pieces: list[dict] = []
    used = {marker_color, anchor_color}

    coords = [(int(r), int(c), int(anchor_color)) for r, c in np.argwhere(g == anchor_color)]
    coords.append((mr, mc, int(marker_color)))
    min_r = min(r for r, _, _ in coords)
    min_c = min(c for _, c, _ in coords)
    coords = [(r - min_r, c - min_c, col) for r, c, col in coords]
    pieces.append({"name": "anchor", "cells": coords, "marker_color": int(marker_color)})

    for col in colors:
        if col in used:
            continue
        coords = [(int(r), int(c), int(col)) for r, c in np.argwhere(g == col)]
        min_r = min(r for r, _, _ in coords)
        min_c = min(c for _, c, _ in coords)
        coords = [(r - min_r, c - min_c, col) for r, c, col in coords]
        pieces.append({"name": f"c{col}", "cells": coords})

    assert len(pieces) == 4, "input must contain four puzzle pieces including the anchor"
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
    piece_vars: list[list[list[tuple[int, int, int]]]] = [_variants(p["cells"]) for p in pieces]

    remaining = list(range(len(pieces)))
    anchor_idx = next((i for i, p in enumerate(pieces) if p["name"] == "anchor"), None)
    if anchor_idx is None:
        return None
    remaining.remove(anchor_idx)
    marker_col = pieces[anchor_idx]["marker_color"]

    anchor_solutions = []
    for var in piece_vars[anchor_idx]:
        marker_pts = [(r, c) for r, c, col in var if col == marker_col]
        if len(marker_pts) != 1:
            continue
        mr, mc = marker_pts[0]
        placed = [(r - mr, c - mc, col) for r, c, col in var]
        if all(0 <= r < H and 0 <= c < W for r, c, _ in placed):
            anchor_solutions.append(placed)
    if not anchor_solutions:
        return None

    for anchor_place in anchor_solutions:
        if not _placeable(board, anchor_place):
            continue
        _apply(board, anchor_place)

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

        if backtrack(remaining):
            return board.copy()
        _apply(board, anchor_place, -1)
    return None


def _decorate_output(content: np.ndarray, bg: int) -> np.ndarray:
    H, W = content.shape
    assert H == W and H % 2 == 0, "packed content must be an even square"
    outer_h = H + 2 * (BG_MARGIN + BORDER_WIDTH)
    outer_w = W + 2 * (BG_MARGIN + BORDER_WIDTH)
    assert outer_h <= OUTPUT_H - 2 * FRAME_OFFSET and outer_w <= OUTPUT_W - 2 * FRAME_OFFSET

    out = np.full((OUTPUT_H, OUTPUT_W), OUTER_FILL_COLOR, dtype=int)
    r0 = c0 = FRAME_OFFSET
    out[r0 : r0 + outer_h, c0 : c0 + outer_w] = bg
    out[
        r0 + BG_MARGIN : r0 + outer_h - BG_MARGIN,
        c0 + BG_MARGIN : c0 + outer_w - BG_MARGIN,
    ] = BORDER_COLOR
    out[
        r0 + BG_MARGIN + BORDER_WIDTH : r0 + BG_MARGIN + BORDER_WIDTH + H,
        c0 + BG_MARGIN + BORDER_WIDTH : c0 + BG_MARGIN + BORDER_WIDTH + W,
    ] = content
    return out


def _solve_content(grid: np.ndarray) -> tuple[np.ndarray, int]:
    g = np.asarray(grid, dtype=int)
    pieces, bg = _extract_pieces(g)
    area = sum(len(p["cells"]) for p in pieces)
    for H, W in _dims(area):
        solved = _solve_packing(pieces, H, W)
        if solved is None:
            continue
        if H == W and H % 2 == 0 and H <= 14:
            return solved.astype(int), bg
    raise AssertionError("packing failed for all candidate dimensions")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble the four scattered puzzle pieces, then render the solution inside a framed 20x20 canvas.

    Input structure:
    - Background color is the dominant color. Four separated colored pieces are scattered on the grid.
    - One piece is the anchor: it contains a unique gray marker pixel touching exactly one other color.
    - The input contains only the puzzle pieces themselves; the border, margin, and outer fill are absent.

    Rule:
    - Reconstruct the original packed square by rotating and placing the four pieces so they tile it exactly.
    - Rotate the anchor so its marker lands at the square's top-left cell.
    - After the square is found, place it into a fixed 20x20 output: add a 1-cell border around the square,
      add a 1-cell background margin outside that border, and fill every remaining cell of the 20x20 canvas
      with the dedicated outer-fill color.

    Key insight:
    - The combinatorial part is still the original packing problem; the new behavior is a deterministic
      post-processing layer applied after the packed square size is known.
    """
    content, bg = _solve_content(grid)
    return _decorate_output(content, bg)


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
    min_r, min_c = rows.min(), cols.min()
    cells = [(int(r) - min_r, int(c) - min_c, color) for r, c in zip(rows, cols)]
    if marker_pos is not None:
        mr, mc = marker_pos[0] - min_r, marker_pos[1] - min_c
        cells = [
            (r, c, MARKER_COLOR if (r, c) == (mr, mc) else col)
            for r, c, col in cells
        ]
    return cells


def _cells_to_patch(cells: list[tuple[int, int, int]]) -> np.ndarray:
    h = max(r for r, _, _ in cells) + 1
    w = max(c for _, c, _ in cells) + 1
    patch = np.full((h, w), INPUT_BG, dtype=int)
    for r, c, col in cells:
        patch[r, c] = col
    return patch


def _random_tab_specs(N: int) -> dict:
    specs = {}
    for tn in TAB_NAMES:
        depth = random.randint(1, N - 1)
        direction = random.choice(["up", "down"]) if tn.startswith("H") else random.choice(["left", "right"])
        specs[tn] = (depth, direction)
    if N == 2:
        specs[random.choice(TAB_NAMES)] = None
    return specs


def _scatter_patches(patches: list[np.ndarray]) -> np.ndarray | None:
    canvas = np.full((CANVAS_H, CANVAS_W), INPUT_BG, dtype=int)
    order = list(range(len(patches)))
    random.shuffle(order)
    for i in order:
        patch = patches[i]
        try:
            r0, c0 = random_free_location(
                canvas,
                patch,
                transparent=INPUT_BG,
                padding=PLACEMENT_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        region = canvas[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
        mask = patch != INPUT_BG
        region[mask] = patch[mask]
    return canvas


def _build_content_and_output(N: int, p: int, tab_specs: dict, quad_colors: dict) -> tuple[np.ndarray, np.ndarray] | None:
    own = _build_ownership(N, p, tab_specs)
    if own is None:
        return None
    S = 2 * N
    content = np.full((S, S), INPUT_BG, dtype=int)
    for q, color in quad_colors.items():
        content[own == q] = color
    content[0, 0] = MARKER_COLOR
    output = _decorate_output(content, INPUT_BG)
    return own, output


def _fusion_metrics_from_output(output: np.ndarray) -> dict[str, int]:
    bg_ring = int(np.sum(output == INPUT_BG))
    border = int(np.sum(output == BORDER_COLOR))
    outer_fill = int(np.sum(output == OUTER_FILL_COLOR))
    non_fill_rows = np.where(np.any(output != OUTER_FILL_COLOR, axis=1))[0]
    non_fill_cols = np.where(np.any(output != OUTER_FILL_COLOR, axis=0))[0]
    content_like = output[3:17, 3:17]
    return {
        "bg_ring": bg_ring,
        "border": border,
        "outer_fill": outer_fill,
        "bbox_h": int(non_fill_rows[-1] - non_fill_rows[0] + 1),
        "bbox_w": int(non_fill_cols[-1] - non_fill_cols[0] + 1),
        "content_nonfill": int(np.sum(content_like != OUTER_FILL_COLOR)),
    }


def _gen_attempt() -> np.ndarray | None:
    N = random.randint(N_MIN, N_MAX)
    p = random.randint(1, N - 1)

    pool = COLOR_POOL.copy()
    random.shuffle(pool)
    quad_colors = {TL: ANCHOR_COLOR, TR: pool[0], BL: pool[1], BR: pool[2]}

    built = _build_content_and_output(N, p, _random_tab_specs(N), quad_colors)
    if built is None:
        return None
    own, expected_output = built

    anchor_cells = _extract_piece(own, TL, ANCHOR_COLOR, marker_pos=(0, 0))
    other_cells = [_extract_piece(own, q, quad_colors[q]) for q in (TR, BL, BR)]
    all_cells = [_rotate_cells(anchor_cells, random.randint(0, 3))]
    all_cells.extend(_rotate_cells(cells, random.randint(0, 3)) for cells in other_cells)

    canvas = _scatter_patches([_cells_to_patch(cells) for cells in all_cells])
    if canvas is None:
        return None

    try:
        out = main(canvas)
    except Exception:
        return None
    if not np.array_equal(out, expected_output):
        return None

    metrics = _fusion_metrics_from_output(out)
    if metrics["outer_fill"] <= 0 or metrics["border"] <= 0 or metrics["bg_ring"] <= 0:
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: exhausted attempts")
