# Combo variant of 7b3084d4 (index 17)
# combo_spec: D4/3+D10/0+D13/1
# Name: 八向变换+旋转对称输出+镜像翻转锚定
#
# Design analysis
# a) Concrete behavior trace:
#    Original: color groups are extracted as four pieces, the anchor is rotated
#    until its singleton grey marker reaches (0,0), and the other pieces are
#    only tried in the 4 rotation states. Any full rectangle is accepted.
#    Combo: pieces are extracted as separated 8-connected components because
#    diagonal partners now share colors in the rotationally symmetric output.
#    The anchor arrives as a pure horizontal or vertical mirror of its output
#    shape, while at least one non-anchor piece uses a reflected-only D4 state.
#    The solver searches the full 8-state D4 orbit for every component, places
#    the anchor marker at (0,0), and only accepts packings whose assembled
#    square becomes 180-degree rotation invariant after treating the grey tag
#    as anchor-colored. Example at N=3: the TL anchor may have a downward tab
#    and a right-side notch; the input anchor is mirrored so the grey corner
#    moves to the top-right or bottom-left of that component. The output keeps
#    the grey cell at (0,0), but the rest of the board matches its half-turn.
# b) Complexity calibration:
#    The search space is larger because every piece uses D4 variants, but the
#    instance family still has only four components and one square target. The
#    half-turn symmetry reduces ambiguity: TL/BR share the anchor color and
#    TR/BL share a second color, so the solver has a visible paired pattern.
#    Symmetric tabs are paired by construction, which keeps outputs readable
#    while still making individual pieces irregular and usually chiral.
# c) Hidden assumptions broken from the original solver:
#    - "one color = one piece" is false because diagonal partners reuse colors.
#    - "anchor needs only rotation" is false because the input anchor is forced
#      to be a pure mirror image outside its rotation orbit.
#    - "any rectangle is valid" is false because the final board must satisfy
#      half-turn symmetry once the marker is interpreted as a corner tag.
#    The rewritten solver therefore uses component extraction, D4 placement,
#    square-only packing, and an explicit symmetry check.
# d) Placement legality:
#    The generator builds the output first, extracts tight patches, applies the
#    required transforms, and scatters them with `random_free_location`, so
#    pieces do not overlap, stay in-bounds, and remain 8-separated.
# e) Visual observability:
#    The new rule is visible from a few examples because outputs now show a
#    diagonal color pairing and the input often contains same-colored pieces
#    plus a reflected anchor whose grey corner is not in the original corner.
#    These are all visible pixel-level cues; no hidden ordering is used.
# f) Visual cleanliness:
#    Only two piece colors plus the grey tag appear in the output, pieces are
#    separated on the 20x20 canvas, and the tab system is still sparse, so the
#    images stay readable rather than layering many markings in one region.
# g) Inferability:
#    The color pattern is structural, not arbitrary: opposite quadrants share
#    colors because the assembled board obeys half-turn symmetry. The grey cell
#    remains a fixed anchor tag, so humans can infer "assemble into a symmetric
#    square with the tag at the top-left" from a small set of examples.

import random
from collections import deque

import numpy as np

from arc_utils import get_bg as _bg_color, is_contiguous, neighbors8, random_free_location


BG = 0
ANCHOR_COLOR = 7
MARKER_COLOR = 5
COLOR_POOL = [1, 2, 3, 4, 6, 8, 9]
CANVAS_H = 20
CANVAS_W = 20
PLACEMENT_PADDING = 1
MAX_ATTEMPTS = 500
N_MIN = 3
N_MAX = 7

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


def _reflect_h_cells(cells: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    return _normalize([(r, -c, col) for r, c, col in cells])


def _reflect_v_cells(cells: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    return _normalize([(-r, c, col) for r, c, col in cells])


def _rotation_variants(cells: list[tuple[int, int, int]]) -> list[list[tuple[int, int, int]]]:
    seen: set[tuple[tuple[int, int, int], ...]] = set()
    out: list[list[tuple[int, int, int]]] = []
    for k in range(4):
        var = _rotate_cells(cells, k)
        key = tuple(var)
        if key not in seen:
            seen.add(key)
            out.append(var)
    return out


def _d4_variants(cells: list[tuple[int, int, int]]) -> list[list[tuple[int, int, int]]]:
    seen: set[tuple[tuple[int, int, int], ...]] = set()
    out: list[list[tuple[int, int, int]]] = []
    for base in (cells, _reflect_h_cells(cells)):
        for k in range(4):
            var = _rotate_cells(base, k)
            key = tuple(var)
            if key not in seen:
                seen.add(key)
                out.append(var)
    return out


def _rotation_keys(cells: list[tuple[int, int, int]]) -> set[tuple[tuple[int, int, int], ...]]:
    return {tuple(var) for var in _rotation_variants(cells)}


def _pure_reflections(cells: list[tuple[int, int, int]]) -> list[tuple[str, list[tuple[int, int, int]]]]:
    seen: set[tuple[tuple[int, int, int], ...]] = set()
    out: list[tuple[str, list[tuple[int, int, int]]]] = []
    for name, var in (("h", _reflect_h_cells(cells)), ("v", _reflect_v_cells(cells))):
        key = tuple(var)
        if key not in seen:
            seen.add(key)
            out.append((name, var))
    return out


def _component_cells(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for sr, sc in np.argwhere(g != bg):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        abs_cells: list[tuple[int, int, int]] = []
        while q:
            r, c = q.popleft()
            abs_cells.append((r, c, int(g[r, c])))
            for nr, nc in neighbors8(r, c):
                if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and g[nr, nc] != bg:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append({"cells": _normalize(abs_cells), "abs": abs_cells})
    return comps


def _anchor_index(comps: list[dict]) -> int | None:
    for i, comp in enumerate(comps):
        if any(col == MARKER_COLOR for _, _, col in comp["cells"]):
            return i
    return None


def _piece_base_color(cells: list[tuple[int, int, int]]) -> int:
    cols = {col for _, _, col in cells if col != MARKER_COLOR}
    assert len(cols) == 1, "each generated piece must use exactly one non-marker color"
    return next(iter(cols))


def _marker_point(cells: list[tuple[int, int, int]]) -> tuple[int, int]:
    pts = [(r, c) for r, c, col in cells if col == MARKER_COLOR]
    assert len(pts) == 1, "anchor must contain exactly one marker cell"
    return pts[0]


def _placeable(board: np.ndarray, placement: list[tuple[int, int, int]]) -> bool:
    h, w = board.shape
    for r, c, _ in placement:
        if not (0 <= r < h and 0 <= c < w) or board[r, c] != -1:
            return False
    return True


def _apply(board: np.ndarray, placement: list[tuple[int, int, int]], val: int | None = None) -> None:
    for r, c, col in placement:
        board[r, c] = col if val is None else val


def _all_placements(var: list[tuple[int, int, int]], H: int, W: int) -> list[list[tuple[int, int, int]]]:
    ph = max(r for r, _, _ in var) + 1
    pw = max(c for _, c, _ in var) + 1
    out: list[list[tuple[int, int, int]]] = []
    for tr in range(H - ph + 1):
        for tc in range(W - pw + 1):
            out.append([(r + tr, c + tc, col) for r, c, col in var])
    return out


def _norm_color(col: int, anchor_base: int) -> int:
    return anchor_base if col == MARKER_COLOR else col


def _placement_respects_symmetry(
    board: np.ndarray,
    placement: list[tuple[int, int, int]],
    anchor_base: int,
) -> bool:
    H, W = board.shape
    local = {(r, c): _norm_color(col, anchor_base) for r, c, col in placement}
    for r, c, col in placement:
        want = _norm_color(col, anchor_base)
        rr, cc = H - 1 - r, W - 1 - c
        if (rr, cc) in local and local[(rr, cc)] != want:
            return False
        other = int(board[rr, cc])
        if other != -1 and _norm_color(other, anchor_base) != want:
            return False
    return True


def _rot_sym_ok(board: np.ndarray, anchor_base: int) -> bool:
    if board.shape[0] != board.shape[1]:
        return False
    if int(board[0, 0]) != MARKER_COLOR:
        return False
    norm = board.copy()
    norm[0, 0] = anchor_base
    return np.array_equal(norm, np.rot90(norm, 2))


def _solve_square_packing(comps: list[dict], side: int) -> np.ndarray | None:
    board = np.full((side, side), -1, dtype=int)
    anchor_idx = _anchor_index(comps)
    assert anchor_idx is not None, "input must contain one anchor component"
    anchor_base = _piece_base_color(comps[anchor_idx]["cells"])

    all_piece_placements: list[list[list[tuple[int, int, int]]]] = []
    for i, comp in enumerate(comps):
        seen: set[tuple[tuple[int, int, int], ...]] = set()
        options: list[list[tuple[int, int, int]]] = []
        for var in _d4_variants(comp["cells"]):
            if i == anchor_idx:
                mr, mc = _marker_point(var)
                placed = [(r - mr, c - mc, col) for r, c, col in var]
                if all(0 <= r < side and 0 <= c < side for r, c, _ in placed):
                    key = tuple(placed)
                    if key not in seen:
                        seen.add(key)
                        options.append(placed)
            else:
                for placed in _all_placements(var, side, side):
                    key = tuple(placed)
                    if key not in seen:
                        seen.add(key)
                        options.append(placed)
        all_piece_placements.append(options)

    remaining = [i for i in range(len(comps)) if i != anchor_idx]

    def backtrack(rem: list[int]) -> bool:
        if not rem:
            return _rot_sym_ok(board, anchor_base)

        best_i = None
        best_opts = None
        for i in rem:
            opts = [
                placement
                for placement in all_piece_placements[i]
                if _placeable(board, placement) and _placement_respects_symmetry(board, placement, anchor_base)
            ]
            if not opts:
                return False
            if best_opts is None or len(opts) < len(best_opts):
                best_i = i
                best_opts = opts
                if len(best_opts) == 1:
                    break

        assert best_i is not None and best_opts is not None
        next_rem = [x for x in rem if x != best_i]
        for placement in best_opts:
            _apply(board, placement)
            if backtrack(next_rem):
                return True
            _apply(board, placement, -1)
        return False

    for anchor_placement in all_piece_placements[anchor_idx]:
        if not _placeable(board, anchor_placement):
            continue
        if not _placement_respects_symmetry(board, anchor_placement, anchor_base):
            continue
        _apply(board, anchor_placement)
        if backtrack(remaining):
            return board.copy()
        _apply(board, anchor_placement, -1)
    return None


def _solve_packing(comps: list[dict]) -> np.ndarray:
    area = sum(len(comp["cells"]) for comp in comps)
    side = int(round(area ** 0.5))
    assert side * side == area, "assembled output must be a square"
    solved = _solve_square_packing(comps, side)
    assert solved is not None, "no D4 packing satisfies the mirrored-anchor rotational constraint"
    return solved.astype(int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble four scattered puzzle pieces into a square whose tagged anchor fixes orientation and whose completed color layout has half-turn symmetry.

    Input structure:
    - The background is the most common color on a 20x20 canvas, with exactly four separated 8-connected components.
    - One component is the anchor: it has a single grey corner pixel attached to an otherwise solid-color piece.
    - Pieces may be rotated or reflected in the input; the anchor specifically may appear as a horizontal or vertical mirror of its output form, and diagonal partners may share colors.

    Rule:
    - Extract pieces by connected components rather than by color.
    - Search all eight D4 transforms for every piece, place the anchor so its grey marker lands at output coordinate (0,0), and tile a square without overlaps.
    - Keep only assemblies whose final board becomes invariant under a 180-degree rotation after interpreting the grey marker as part of the anchor color.

    Key insight:
    - The grey pixel is a positional tag, not a second symmetry partner: use it to identify and pin the anchor, then use the half-turn symmetry of the assembled square to resolve repeated colors and reflected piece orientations.
    """
    comps = _component_cells(np.asarray(grid, dtype=int))
    assert len(comps) == 4, "input must contain exactly four separated pieces"
    return _solve_packing(comps)


def _build_rot_sym_ownership(
    N: int,
    p: int,
    h_left: tuple[int, str],
    v_top: tuple[int, str],
) -> np.ndarray | None:
    S = 2 * N
    own = np.zeros((S, S), dtype=int)
    own[:N, :N] = TL
    own[:N, N:] = TR
    own[N:, :N] = BL
    own[N:, N:] = BR

    h_right = (h_left[0], "down" if h_left[1] == "up" else "up")
    v_bot = (v_top[0], "right" if v_top[1] == "left" else "left")

    def apply_h(kind_col: int, spec: tuple[int, str], up_owner: int, down_owner: int) -> None:
        depth, direction = spec
        if direction == "up":
            own[N - depth:N, kind_col] = up_owner
        else:
            own[N:N + depth, kind_col] = down_owner

    def apply_v(kind_row: int, spec: tuple[int, str], left_owner: int, right_owner: int) -> None:
        depth, direction = spec
        if direction == "left":
            own[kind_row, N - depth:N] = left_owner
        else:
            own[kind_row, N:N + depth] = right_owner

    apply_h(p, h_left, BL, TL)
    apply_h(S - 1 - p, h_right, BR, TR)
    apply_v(p, v_top, TR, TL)
    apply_v(S - 1 - p, v_bot, BR, BL)

    for q in (TL, TR, BL, BR):
        mask = (own == q).astype(int)
        if mask.sum() == 0:
            return None
        if not is_contiguous(mask, background=0, connectivity=8):
            return None
    return own


def _extract_piece(
    own: np.ndarray,
    label: int,
    color: int,
    marker_pos: tuple[int, int] | None = None,
) -> list[tuple[int, int, int]]:
    rows, cols = np.where(own == label)
    min_r, min_c = int(rows.min()), int(cols.min())
    cells: list[tuple[int, int, int]] = []
    for r, c in zip(rows, cols):
        rr, cc = int(r), int(c)
        col = MARKER_COLOR if marker_pos == (rr, cc) else color
        cells.append((rr - min_r, cc - min_c, col))
    cells.sort()
    return cells


def _cells_to_patch(cells: list[tuple[int, int, int]]) -> np.ndarray:
    h = max(r for r, _, _ in cells) + 1
    w = max(c for _, c, _ in cells) + 1
    patch = np.zeros((h, w), dtype=int)
    for r, c, col in cells:
        patch[r, c] = col
    return patch


def _canonical_output_and_pieces(
    N: int,
    pair_color: int,
    own: np.ndarray,
) -> tuple[np.ndarray, list[list[tuple[int, int, int]]]]:
    output = np.zeros((2 * N, 2 * N), dtype=int)
    output[own == TL] = ANCHOR_COLOR
    output[own == BR] = ANCHOR_COLOR
    output[own == TR] = pair_color
    output[own == BL] = pair_color
    output[0, 0] = MARKER_COLOR

    pieces = [
        _extract_piece(own, TL, ANCHOR_COLOR, marker_pos=(0, 0)),
        _extract_piece(own, TR, pair_color),
        _extract_piece(own, BL, pair_color),
        _extract_piece(own, BR, ANCHOR_COLOR),
    ]
    return output, pieces


def _anchor_reflection_choices(
    anchor_cells: list[tuple[int, int, int]]
) -> list[tuple[str, list[tuple[int, int, int]]]]:
    rot_keys = _rotation_keys(anchor_cells)
    return [(name, var) for name, var in _pure_reflections(anchor_cells) if tuple(var) not in rot_keys]


def _reflected_only_variants(
    cells: list[tuple[int, int, int]]
) -> list[list[tuple[int, int, int]]]:
    rot_keys = _rotation_keys(cells)
    return [var for var in _d4_variants(cells) if tuple(var) not in rot_keys]


def _scatter_patches(patches: list[np.ndarray]) -> np.ndarray | None:
    canvas = np.zeros((CANVAS_H, CANVAS_W), dtype=int)
    order = list(range(len(patches)))
    random.shuffle(order)
    for idx in order:
        patch = patches[idx]
        try:
            r0, c0 = random_free_location(
                canvas,
                patch,
                transparent=0,
                padding=PLACEMENT_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        region = canvas[r0:r0 + patch.shape[0], c0:c0 + patch.shape[1]]
        mask = patch != 0
        region[mask] = patch[mask]
    return canvas


def _sample_case() -> dict | None:
    N = random.randint(N_MIN, N_MAX)
    p = random.randint(1, N - 1)
    h_left = (random.randint(1, N - 1), random.choice(["up", "down"]))
    v_top = (random.randint(1, N - 1), random.choice(["left", "right"]))
    own = _build_rot_sym_ownership(N, p, h_left, v_top)
    if own is None:
        return None

    pair_color = random.choice(COLOR_POOL)
    output, pieces = _canonical_output_and_pieces(N, pair_color, own)
    norm = output.copy()
    norm[0, 0] = ANCHOR_COLOR
    if not np.array_equal(norm, np.rot90(norm, 2)):
        return None

    anchor_choices = _anchor_reflection_choices(pieces[0])
    if not anchor_choices:
        return None
    anchor_mode, anchor_variant = random.choice(anchor_choices)

    reflected_idx = [i for i in range(1, 4) if _reflected_only_variants(pieces[i])]
    if not reflected_idx:
        return None
    forced_idx = random.choice(reflected_idx)

    transformed = [anchor_variant]
    reflected_non_anchor = 0
    for i in range(1, 4):
        reflected_only = _reflected_only_variants(pieces[i])
        choices = _d4_variants(pieces[i])
        if i == forced_idx:
            chosen = random.choice(reflected_only)
        elif reflected_only and random.random() < 0.5:
            chosen = random.choice(reflected_only)
        else:
            chosen = random.choice(choices)
        if tuple(chosen) not in _rotation_keys(pieces[i]):
            reflected_non_anchor += 1
        transformed.append(chosen)

    canvas = _scatter_patches([_cells_to_patch(cells) for cells in transformed])
    if canvas is None:
        return None

    try:
        solved = main(canvas)
    except Exception:
        return None
    if not np.array_equal(solved, output):
        return None

    return {
        "input": canvas,
        "output": output,
        "meta": {
            "N": N,
            "p": p,
            "anchor_mode": anchor_mode,
            "anchor_reflected_only": True,
            "reflected_non_anchor": reflected_non_anchor,
            "pair_color": pair_color,
        },
    }


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        case = _sample_case()
        if case is not None:
            return case["input"]
    raise RuntimeError("generate_input: exhausted attempts")
