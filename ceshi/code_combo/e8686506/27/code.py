# Combo variant of e8686506 (index 27)
# combo_spec: D11/2+D14/1
# Name: 迭代拼合+颜色编码顺序
#
# Design analysis for the fused variant
# a) Concrete behavior:
#    Original e8686506 lets a solver globally pack scattered fragments into one LR-symmetric
#    rectangle, with the frame and symmetry doing most of the work. This variant keeps the
#    same outer puzzle shell, but replaces the inner packing rule with an explicit sequence:
#    the border frame is the largest fragment and is placed first; the remaining fragments are
#    full-width inner bands of size 2x3 or 1x3. Example for H=8:
#      - original style: two equal 2x3 bands could be treated as an unordered pair.
#      - fused style: after the frame, all area-6 pieces are processed before area-3 pieces,
#        and within the area-6 tie the smaller color goes into the upper slot first, because
#        the solver always chooses the topmost still-valid slot for the current piece.
#      - difference: colors now decide which equal-area band occupies which row range.
# b) Complexity calibration:
#    The generator always creates at least one repeated-size group, such as 6/6/3/3 or
#    6/6/6/3/3. That keeps the task from collapsing into pure shape matching: if color order
#    is ignored, equal-area bands can swap colors. The family is still controlled because the
#    only free slots are the stacked 3-cell-wide bands inside a fixed 5-column frame.
# c) Hidden assumptions audit:
#    - The original solver assumed frame+propagation alone makes every piece placement forced.
#      That is false here by construction: repeated equal-area pieces are intentionally
#      swappable until color order breaks the tie.
#    - The new solver assumes one frame component exactly matches the HxW border mask. The
#      generator guarantees this by placing the frame as a single hollow rectangle group.
#    - The new solver assumes every non-frame piece is LR-symmetric by itself and therefore
#      can only sit in the centered 3-column interior. The generator guarantees this by using
#      only 2x3 and 1x3 centered bands.
#    - The new greedy placement rule assumes intended slots are recovered by scanning from the
#      top after sorting pieces by (-area, color). The generator guarantees this by arranging
#      the band heights in non-increasing order from top to bottom.
# d) Placement legality:
#    The generator places the frame as one patch and scatters each inner band patch outside the
#    blocked frame bbox with Chebyshev padding, so there are no overlaps or out-of-bounds cases.
# e) Visual observability:
#    The fusion is visible: the output is still a dense mirrored rectangle, but equal-size inner
#    bands carry different colors whose top-to-bottom order follows color rank. Across examples
#    a human can notice that same-shape bands are not arbitrary: the "earlier" color always
#    lands in the earlier slot within an equal-area group.
# f) Visual cleanliness:
#    No labels or overlays are added. The frame remains a single clean border, and each ordered
#    fragment occupies one centered horizontal band, so pieces stay visually separable.
# g) Inferability of color mapping:
#    The generator uses a small fixed ordered palette subset (excluding the background) and
#    always creates repeated-size groups. Reversing color order inside a tie group changes the
#    color arrangement of the solved rectangle, which confirms that color order is a real
#    control variable rather than decoration.

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


_ORDERABLE_COLORS = [1, 2, 3, 4, 6, 8, 9]
_BG_CHOICES = [0, 5, 7]

_CANVAS_MIN = 13
_CANVAS_MAX = 15
_SCATTER_PADDING = 1
_GEN_MAX_RETRIES = 300

_PATTERNS_BY_H = {
    7: [2, 1, 1, 1],        # 6/3/3/3
    8: [2, 2, 1, 1],        # 6/6/3/3
    9: [2, 2, 1, 1, 1],     # 6/6/3/3/3
    10: [2, 2, 2, 1, 1],    # 6/6/6/3/3
}


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


def _choose_dims(total_area: int) -> tuple[int, int]:
    dims = [(h, w) for h, w in _factors(total_area) if h >= w]
    assert dims, "at least one portrait factor pair must exist"
    best_gap = min(abs(h - w) for h, w in dims)
    best_dims = [(h, w) for h, w in dims if abs(h - w) == best_gap]
    assert len(best_dims) == 1, "closest-to-square portrait output shape must be unique"
    return best_dims[0]


def _border_coords(H: int, W: int) -> tuple[tuple[int, int], ...]:
    coords = []
    for r in range(H):
        for c in range(W):
            if r in (0, H - 1) or c in (0, W - 1):
                coords.append((r, c))
    return tuple(coords)


def _coords_lr_symmetric(coords: list[tuple[int, int]], W: int) -> bool:
    pts = {(int(r), int(c)) for r, c in coords}
    return pts == {(r, W - 1 - c) for r, c in pts}


def _extract_pieces(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    pieces: list[dict] = []
    for col in sorted(set(map(int, np.unique(g))) - {bg}):
        for pts in components_mask(g == col, connectivity=4):
            h, w, coords = _piece_sig(pts)
            r0, c0, _, _ = bbox_coords(pts)
            pieces.append(
                {
                    "col": int(col),
                    "coords": coords,
                    "h": int(h),
                    "w": int(w),
                    "area": int(len(coords)),
                    "input_pos": (int(r0), int(c0)),
                }
            )
    return int(bg), pieces


def _find_frame_piece(pieces: list[dict], H: int, W: int) -> dict:
    border = _border_coords(H, W)
    matches = [p for p in pieces if int(p["h"]) == H and int(p["w"]) == W and p["coords"] == border]
    assert len(matches) == 1, "expected exactly one border frame piece"
    return matches[0]


def _place_piece(board: np.ndarray, piece: dict, r0: int, c0: int) -> None:
    for dr, dc in piece["coords"]:
        board[int(r0) + int(dr), int(c0) + int(dc)] = int(piece["col"])


def _sort_pieces(pieces: list[dict], *, reverse_color_ties: bool = False) -> list[dict]:
    if reverse_color_ties:
        return sorted(
            pieces,
            key=lambda p: (-int(p["area"]), -int(p["col"]), int(p["input_pos"][0]), int(p["input_pos"][1])),
        )
    return sorted(
        pieces,
        key=lambda p: (-int(p["area"]), int(p["col"]), int(p["input_pos"][0]), int(p["input_pos"][1])),
    )


def _candidate_placements(board: np.ndarray, piece: dict) -> list[tuple[int, int]]:
    H, W = board.shape
    h, w = int(piece["h"]), int(piece["w"])
    out: list[tuple[int, int]] = []
    occupied = board != -1
    has_seed = bool(occupied.any())
    for r0 in range(H - h + 1):
        for c0 in range(W - w + 1):
            abs_coords = [(r0 + int(dr), c0 + int(dc)) for dr, dc in piece["coords"]]
            if not _coords_lr_symmetric(abs_coords, W):
                continue
            if any(board[r, c] != -1 for r, c in abs_coords):
                continue
            if has_seed:
                touches = False
                for r, c in abs_coords:
                    for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                        if 0 <= rr < H and 0 <= cc < W and board[rr, cc] != -1:
                            touches = True
                            break
                    if touches:
                        break
                if not touches:
                    continue
            out.append((int(r0), int(c0)))
    out.sort()
    return out


def _solve_with_order(grid: np.ndarray, *, reverse_color_ties: bool = False) -> np.ndarray:
    _, pieces = _extract_pieces(grid)
    total_area = sum(int(piece["area"]) for piece in pieces)
    H, W = _choose_dims(total_area)
    board = np.full((H, W), -1, dtype=int)

    frame = _find_frame_piece(pieces, H, W)
    _place_piece(board, frame, 0, 0)

    remaining = [piece for piece in pieces if piece is not frame]
    for piece in _sort_pieces(remaining, reverse_color_ties=reverse_color_ties):
        placements = _candidate_placements(board, piece)
        assert placements, "ordered assembly ran out of legal placements"
        _place_piece(board, piece, placements[0][0], placements[0][1])

    assert np.all(board != -1), "assembly must fill the whole rectangle"
    assert np.array_equal(board, board[:, ::-1]), "assembled board must be LR-symmetric"
    return board


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered fragments into a mirrored rectangle by adding one color-coded piece at a time.

    Input structure:
    - The background is the most common color.
    - One non-background component is a hollow Hx5 border frame kept intact in the input.
    - The other components are centered 2x3 or 1x3 bands scattered elsewhere.

    Rule:
    - Infer the output rectangle dimensions from total non-background area.
    - Lock the border frame in place as the initial assembled structure.
    - Sort the remaining fragments by descending area; when areas tie, smaller color values go first.
    - For each chosen fragment, place it into the topmost legal slot that touches the current assembly.

    Key insight:
    - Equal-area inner bands share the same shape, so shape alone does not decide which color occupies
      which band. The color ranking is the missing order signal, and the solver must follow that
      iterative sequence rather than treat the equal-area fragments as an unordered set.
    """
    return _solve_with_order(np.asarray(grid, dtype=int), reverse_color_ties=False).astype(int)


def _build_output() -> tuple[np.ndarray, int]:
    H = random.choice(sorted(_PATTERNS_BY_H))
    W = 5
    bg = random.choice(_BG_CHOICES)
    pattern = _PATTERNS_BY_H[H]

    needed = len(pattern) + 1
    colors = sorted(random.sample(_ORDERABLE_COLORS, needed))
    frame_color = colors[-1]
    band_colors = colors[:-1]

    out = np.full((H, W), bg, dtype=int)
    out[0, :] = frame_color
    out[H - 1, :] = frame_color
    out[:, 0] = frame_color
    out[:, W - 1] = frame_color

    r = 1
    for h, col in zip(pattern, band_colors):
        out[r : r + h, 1 : W - 1] = int(col)
        r += h

    return out, bg


def _extract_patches(out: np.ndarray, bg: int) -> tuple[np.ndarray, list[np.ndarray]]:
    H, W = out.shape
    frame_patch = np.full((H, W), bg, dtype=int)
    frame_patch[0, :] = out[0, :]
    frame_patch[H - 1, :] = out[H - 1, :]
    frame_patch[:, 0] = out[:, 0]
    frame_patch[:, W - 1] = out[:, W - 1]

    inner_patches: list[np.ndarray] = []
    colors = sorted(set(map(int, np.unique(out))) - {int(bg), int(out[0, 0])})
    for col in colors:
        pts = components_mask(out == col, connectivity=4)
        assert len(pts) == 1, "each ordered color should correspond to one band"
        r0, c0, r1, c1 = bbox_coords(pts[0])
        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
        patch[:, :] = int(col)
        inner_patches.append(patch)
    return frame_patch, inner_patches


def _scatter(out: np.ndarray, bg: int) -> np.ndarray:
    rows = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    cols = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    canvas = np.full((rows, cols), bg, dtype=int)
    frame_patch, inner_patches = _extract_patches(out, bg)
    H, W = frame_patch.shape

    max_r = rows - H - 2
    max_c = cols - W - 2
    if max_r < 1 or max_c < 1:
        raise ValueError("frame patch does not fit on the canvas")
    off_r = random.randint(1, max_r)
    off_c = random.randint(1, max_c)

    canvas[off_r : off_r + H, off_c : off_c + W] = -1
    for patch in inner_patches:
        r, c = random_free_location(
            canvas,
            patch,
            transparent=bg,
            padding=_SCATTER_PADDING,
            padding_connectivity=8,
        )
        paste(canvas, patch, r, c, transparent=bg)

    canvas[off_r : off_r + H, off_c : off_c + W] = bg
    paste(canvas, frame_patch, off_r, off_c, transparent=bg)
    return canvas


def _gen_attempt() -> np.ndarray | None:
    out, bg = _build_output()
    try:
        inp = _scatter(out, bg)
        got = main(inp)
        rev = _solve_with_order(inp, reverse_color_ties=True)
    except Exception:
        return None
    if not np.array_equal(got, out):
        return None
    if np.array_equal(got, rev):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e8686506 combo-variant input")
