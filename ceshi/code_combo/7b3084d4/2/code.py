# Combo variant of 7b3084d4 (index 2)
# combo_spec: D4/1+D7/0
# Name: 多标记定序旋转编码拼图
#
# Design analysis
# a) Concrete behavior:
#    - Original: one grey singleton marks the anchor; solver rotates only that one
#      piece so the marker lands at output (0,0), then packs the rest freely.
#    - Fused variant: all four pieces carry one corner marker, using colors 1..4.
#      After sorting pieces by marker color, the solver places them in row-major
#      corner order: 1->top-left, 2->top-right, 3->bottom-left, 4->bottom-right.
#      The same marker color also encodes a clockwise rotation count:
#      1->0, 2->1, 3->2, 4->3 quarter-turns. Example with side 6:
#      marker-2 piece is shown in the input pre-rotated by 270 degrees; solver
#      rotates it back by 90 degrees and aligns marker 2 to (0,5).
# b) Complexity calibration:
#    - Only four marker colors are used. They repeat consistently across all
#      training instances, so humans can infer both order and rotation from a few
#      examples. The tabbed shapes from the original task are preserved, so the
#      puzzle still looks like an actual assembly problem rather than a trivial
#      quadrant split.
# c) Hidden-assumption audit:
#    - The original solver assumed exactly one singleton marker and could infer
#      the anchor by adjacency. That is no longer valid.
#    - The original solver also assumed non-anchor pieces were monochrome and
#      freely rotatable. Here every piece is bi-color and rotation is fixed by
#      its own marker.
#    - New solver assumptions: exactly four 8-connected components, exactly one
#      marker from {1,2,3,4} inside each component, and total area is an even
#      square. The generator guarantees all of these.
# d) Placement legality:
#    - Output-first generation builds a legal 2N x 2N ownership grid, then
#      extracts pieces, inverse-rotates them for the input, and scatters them on
#      a 20x20 canvas with bbox padding so components never touch or overlap.
# e) Visual observability:
#    - Marker colors are visible ARC colors, never background. They sit on the
#      four outer output corners, making the order cue readable in the output.
#      In the input, the same colors visibly move with rotation, so the rotation
#      cue is also observable.
# f) Visual cleanliness:
#    - Each piece has one dominant base color from {6,7,8,9} plus one small
#      marker pixel from {1,2,3,4}. No dense overprinting or layered symbols.
# g) Non-degeneracy target:
#    - Fusion is only meaningful if all four marker colors appear and the coded
#      rotations are actually exercised. The generator therefore always uses the
#      full marker set {1,2,3,4}, so three of the four pieces require non-zero
#      prescribed rotation in every sample.

from __future__ import annotations

from collections import deque
import random

import numpy as np

from arc_utils import get_bg as _bg_color, is_contiguous, neighbors8, random_free_location


BG = 0
MARKER_COLORS = (1, 2, 3, 4)
BASE_COLORS = (6, 7, 8, 9)
CANVAS_H = 20
CANVAS_W = 20
PLACEMENT_PADDING = 1
MAX_ATTEMPTS = 400
N_MIN, N_MAX = 2, 7
TL, TR, BL, BR = 0, 1, 2, 3
TARGET_ORDER = (
    (1, TL),
    (2, TR),
    (3, BL),
    (4, BR),
)


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


def _cells_to_patch(cells: list[tuple[int, int, int]]) -> np.ndarray:
    h = max(r for r, _, _ in cells) + 1
    w = max(c for _, c, _ in cells) + 1
    patch = np.zeros((h, w), dtype=int)
    for r, c, col in cells:
        patch[r, c] = col
    return patch


def _components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = int(_bg_color(g))
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


def _marker_info(cells: list[tuple[int, int, int]]) -> tuple[int, tuple[int, int]]:
    marks = [(col, (r, c)) for r, c, col in cells if col in MARKER_COLORS]
    assert len(marks) == 1, "each piece must contain exactly one marker from {1,2,3,4}"
    color, pos = marks[0]
    return int(color), pos


def _extract_pieces(grid: np.ndarray) -> list[dict]:
    comps = _components(np.asarray(grid, dtype=int))
    assert len(comps) == 4, "input must contain exactly four separated pieces"
    pieces = []
    seen_markers = set()
    for comp in comps:
        marker_color, marker_pos = _marker_info(comp["cells"])
        assert marker_color not in seen_markers, "marker colors must be unique"
        seen_markers.add(marker_color)
        pieces.append(
            {
                "cells": comp["cells"],
                "marker_color": marker_color,
                "marker_pos": marker_pos,
            }
        )
    assert seen_markers == set(MARKER_COLORS), "marker colors must be exactly {1,2,3,4}"
    return pieces


def _target_corner(side: int, slot: int) -> tuple[int, int]:
    if slot == TL:
        return 0, 0
    if slot == TR:
        return 0, side - 1
    if slot == BL:
        return side - 1, 0
    return side - 1, side - 1


def _place_piece(
    board: np.ndarray,
    cells: list[tuple[int, int, int]],
    marker_target: tuple[int, int],
) -> None:
    marker_color, (mr, mc) = _marker_info(cells)
    tr = int(marker_target[0] - mr)
    tc = int(marker_target[1] - mc)
    for r, c, _ in cells:
        rr = r + tr
        cc = c + tc
        assert 0 <= rr < board.shape[0] and 0 <= cc < board.shape[1], "piece placement left board bounds"
        assert board[rr, cc] == -1, f"piece overlap while placing marker {marker_color}"
    for r, c, col in cells:
        board[r + tr, c + tc] = col


def _build_tab_ownership(N: int, p: int, tab_specs: dict[str, tuple[int, str] | None]) -> np.ndarray | None:
    S = 2 * N
    own = np.zeros((S, S), dtype=int)
    own[:N, :N] = TL
    own[:N, N:] = TR
    own[N:, :N] = BL
    own[N:, N:] = BR

    for kind, col, up_owner, down_owner in (
        ("H_left", p, BL, TL),
        ("H_right", S - 1 - p, BR, TR),
    ):
        spec = tab_specs.get(kind)
        if spec is None:
            continue
        depth, direction = spec
        if direction == "up":
            own[N - depth : N, col] = up_owner
        else:
            own[N : N + depth, col] = down_owner

    for kind, row, left_owner, right_owner in (
        ("V_top", p, TR, TL),
        ("V_bot", S - 1 - p, BR, BL),
    ):
        spec = tab_specs.get(kind)
        if spec is None:
            continue
        depth, direction = spec
        if direction == "left":
            own[row, N - depth : N] = left_owner
        else:
            own[row, N : N + depth] = right_owner

    for q in (TL, TR, BL, BR):
        if not is_contiguous((own == q).astype(int), background=0, connectivity=8):
            return None
    return own


def _random_tab_specs(N: int) -> dict[str, tuple[int, str] | None]:
    specs: dict[str, tuple[int, str] | None] = {}
    for name in ("H_left", "H_right", "V_top", "V_bot"):
        depth = random.randint(1, N - 1)
        if name.startswith("H"):
            direction = random.choice(("up", "down"))
        else:
            direction = random.choice(("left", "right"))
        specs[name] = (depth, direction)
    if N == 2:
        specs[random.choice(tuple(specs))] = None
    return specs


def _render_output_and_pieces(own: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    S = own.shape[0]
    base_colors = list(BASE_COLORS)
    random.shuffle(base_colors)
    color_by_slot = {TL: base_colors[0], TR: base_colors[1], BL: base_colors[2], BR: base_colors[3]}
    output = np.zeros((S, S), dtype=int)
    for slot, col in color_by_slot.items():
        output[own == slot] = int(col)

    marker_targets = {
        TL: (0, 0, 1),
        TR: (0, S - 1, 2),
        BL: (S - 1, 0, 3),
        BR: (S - 1, S - 1, 4),
    }
    for slot, (r, c, marker_color) in marker_targets.items():
        assert own[r, c] == slot, "outer corner ownership must match its slot"
        output[r, c] = marker_color

    pieces: list[dict] = []
    for slot in (TL, TR, BL, BR):
        coords_abs = np.argwhere(own == slot)
        min_r, min_c = map(int, coords_abs.min(axis=0))
        cells = []
        for r, c in coords_abs:
            r = int(r)
            c = int(c)
            cells.append((r - min_r, c - min_c, int(output[r, c])))
        marker_color, _ = _marker_info(cells)
        pieces.append({"slot": slot, "marker_color": marker_color, "cells": _normalize(cells)})
    return output, pieces


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
        region = canvas[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
        mask = patch != 0
        region[mask] = patch[mask]
    return canvas


def _build_instance() -> tuple[np.ndarray, np.ndarray]:
    N = random.randint(N_MIN, N_MAX)
    p = random.randint(1, N - 1)
    own = _build_tab_ownership(N, p, _random_tab_specs(N))
    assert own is not None
    output, pieces = _render_output_and_pieces(own)

    transformed = []
    for piece in pieces:
        rot_k = (piece["marker_color"] - 1) % 4
        transformed.append(_rotate_cells(piece["cells"], (-rot_k) % 4))

    canvas = _scatter_patches([_cells_to_patch(cells) for cells in transformed])
    if canvas is None:
        raise ValueError("scatter failed")
    return canvas, output


def _instance_metrics(grid: np.ndarray) -> dict[str, object]:
    pieces = sorted(_extract_pieces(grid), key=lambda p: p["marker_color"])
    marker_colors = [int(p["marker_color"]) for p in pieces]
    rot_quarters = [int(p["marker_color"] - 1) for p in pieces]
    nonzero_rotations = sum(rot != 0 for rot in rot_quarters)
    return {
        "marker_colors": tuple(marker_colors),
        "rot_quarters": tuple(rot_quarters),
        "nonzero_rotations": int(nonzero_rotations),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble four scattered tabbed pieces using per-piece marker colors as both order and rotation codes.

    Input structure:
    - Background is the most common color.
    - Exactly four separated 8-connected puzzle pieces are scattered on the board.
    - Every piece is mostly one base color from {6,7,8,9} and contains exactly one
      corner marker pixel from {1,2,3,4}.

    Rule:
    - Extract the four pieces and read each piece's unique marker color.
    - Sort pieces by marker color ascending; this fixes their output corner order:
      1->top-left, 2->top-right, 3->bottom-left, 4->bottom-right.
    - Rotate each piece clockwise by marker_color-1 quarter-turns, then translate
      it so that its marker lands on the assigned outer corner of the assembled square.

    Key insight:
    - The same visible marker color carries two instructions at once: it says which
      piece is placed first in the final 2x2 corner order, and it also says how much
      that specific piece must be rotated before assembly.
    """
    g = np.asarray(grid, dtype=int)
    pieces = sorted(_extract_pieces(g), key=lambda p: p["marker_color"])
    total_area = sum(len(piece["cells"]) for piece in pieces)
    side = int(round(total_area ** 0.5))
    assert side * side == total_area and side % 2 == 0, "piece area must form an even square"

    board = np.full((side, side), -1, dtype=int)
    for marker_color, slot in TARGET_ORDER:
        piece = next(piece for piece in pieces if piece["marker_color"] == marker_color)
        oriented = _rotate_cells(piece["cells"], marker_color - 1)
        _place_piece(board, oriented, _target_corner(side, slot))

    assert np.all(board != -1), "assembled board must be fully tiled"
    return board.astype(int)


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        try:
            canvas, expected = _build_instance()
        except Exception:
            continue
        try:
            solved = main(canvas)
        except Exception:
            continue
        if np.array_equal(solved, expected):
            return canvas
    raise RuntimeError("generate_input: exhausted attempts")
