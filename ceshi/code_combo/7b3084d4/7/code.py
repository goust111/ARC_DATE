# Combo variant of 7b3084d4 (index 7)
# combo_spec: D1/0+D12/0
# Name: 相邻颜色约束边缘匹配

# Design analysis for the fused variant:
# a) Concrete behavior:
#    Original samples use four mostly single-color pieces, so packing can be solved from
#    geometry alone once the grey anchor corner is rotated to the top-left. Here each piece
#    becomes a multi-color connected component. Body cells use only {6,7,8,9}; internal
#    contact stripes use only {1,2,3,4}. Example on a 6x6 target: if the anchor's right
#    boundary cells are all color 1 and its lower boundary cells are all color 2, then the
#    correct right neighbor must expose the same color-1 stripe on its touching edge, and the
#    correct lower neighbor must expose the same color-2 stripe. After assembly, the four body
#    colors alternate warm/cool around the rectangle: 7(warm), cool, 9(warm), cool.
# b) Complexity calibration:
#    The visible state space stays small and readable: exactly four stripe colors and exactly
#    four body colors. Connector colors never overlap body colors, so humans can see that
#    stripe bands are match evidence rather than extra pieces. I keep N>=3 so each piece still
#    has a visible body region after recoloring its contact edges.
# c) Hidden assumptions replaced:
#    The original solver assumes one color ~= one piece, so `_extract_pieces` groups pixels by
#    color. That fails once edge stripes are added. The new solver instead extracts 8-connected
#    components, finds the anchor by the singleton grey pixel, identifies each piece's body
#    color from the reserved body palette, and checks contacts during packing. The original
#    backtracking accepted any gap-free rectangle; the new one additionally requires every
#    inter-piece contact to use the same connector color on both sides and requires the two
#    touching body colors to have opposite warm/cool parity.
# d) Placement legality:
#    The generator still builds the final square first, paints connector stripes only on true
#    internal boundaries, rotates the pieces, and scatters their tight patches on a 20x20
#    canvas with one-cell 8-neighborhood padding. No pieces overlap or leave the canvas.
# e) Visual observability:
#    Across a few examples, a human can see two simultaneous regularities: touching sides carry
#    identical stripe colors, and neighboring bodies alternate warm/cool. Neither rule depends
#    on hidden numeric ordering of colors.
# f) Visual cleanliness:
#    There are only three visual layers: grey anchor marker, solid body colors, and stripe
#    colors on contact edges. Some stripe-colored cells can lie on the rectangle perimeter
#    because a contact edge may reach the border, but no extra overlays or detached markers are
#    introduced, so the result stays legible.
# g) Inferability of color changes:
#    Body colors are partitioned into two repeated groups: warm {7,9} and cool {6,8}. This
#    repetition gives enough support for the "adjacent pieces contrast" rule, while connector
#    stripes provide a separate repeated cue for which edges must meet.

from __future__ import annotations

import random
import sys
from collections import Counter, deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import get_bg as _bg_color, is_contiguous, neighbors8, random_free_location


BG = 0
MARKER_COLOR = 5

TL, TR, BL, BR = 0, 1, 2, 3
LABELS = (TL, TR, BL, BR)

BODY_COLOR_BY_LABEL = {TL: 7, BR: 9}
COOL_BODY_COLORS = [6, 8]
BODY_COLORS = {6, 7, 8, 9}
WARM_BODY_COLORS = {7, 9}

CONNECTOR_BY_PAIR = {
    frozenset({TL, TR}): 1,
    frozenset({TL, BL}): 2,
    frozenset({TR, BR}): 3,
    frozenset({BL, BR}): 4,
}
CONNECTOR_COLORS = set(CONNECTOR_BY_PAIR.values())
EXPECTED_ADJACENCY_PAIRS = set(CONNECTOR_BY_PAIR)

CANVAS_H, CANVAS_W = 20, 20
PLACEMENT_PADDING = 1
MAX_ATTEMPTS = 400
N_MIN, N_MAX = 4, 7
TAB_NAMES = ["H_left", "H_right", "V_top", "V_bot"]


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


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _adjacency_pairs_from_labels(labels: np.ndarray) -> set[frozenset[int]]:
    h, w = labels.shape
    pairs: set[frozenset[int]] = set()
    for r in range(h):
        for c in range(w):
            if r + 1 < h and int(labels[r, c]) != int(labels[r + 1, c]):
                pairs.add(frozenset({int(labels[r, c]), int(labels[r + 1, c])}))
            if c + 1 < w and int(labels[r, c]) != int(labels[r, c + 1]):
                pairs.add(frozenset({int(labels[r, c]), int(labels[r, c + 1])}))
    return pairs


def _components(g: np.ndarray) -> tuple[list[dict], int]:
    bg = int(_bg_color(g))
    h, w = g.shape
    seen = np.zeros_like(g, dtype=bool)
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
    return comps, bg


def _body_color(cells: list[tuple[int, int, int]]) -> int:
    body_colors = sorted({int(col) for _, _, col in cells if int(col) in BODY_COLORS})
    assert len(body_colors) == 1, "each piece must contain exactly one body color from {6,7,8,9}"
    return body_colors[0]


def _temperature(body_color: int) -> int:
    assert body_color in BODY_COLORS
    return 0 if body_color in WARM_BODY_COLORS else 1


def _extract_pieces(g: np.ndarray) -> tuple[list[dict], int]:
    comps, bg = _components(np.asarray(g, dtype=int))
    assert len(comps) == 4, "input must contain exactly four separated pieces"

    marker_points = [tuple(map(int, rc)) for rc in np.argwhere(g == MARKER_COLOR)]
    assert len(marker_points) == 1, "input must contain exactly one grey anchor marker"
    marker_abs = marker_points[0]

    pieces: list[dict] = []
    anchor_idx = None
    for i, comp in enumerate(comps):
        local_cells = comp["cells"]
        abs_set = {(r, c) for r, c, _ in comp["abs"]}
        marker_local = None
        if marker_abs in abs_set:
            for r, c, col in local_cells:
                if int(col) == MARKER_COLOR:
                    marker_local = (int(r), int(c))
                    break
            anchor_idx = i
        pieces.append(
            {
                "name": f"p{i}",
                "cells": local_cells,
                "body_color": _body_color(local_cells),
                "marker_local": marker_local,
            }
        )
    assert anchor_idx is not None, "the grey marker must belong to one piece"
    pieces[anchor_idx]["name"] = "anchor"
    return pieces, bg


def _output_dims(area: int) -> list[tuple[int, int]]:
    side = int(round(area ** 0.5))
    if side * side == area:
        return [(side, side)]
    return []


def _placeable(board_owner: np.ndarray, placement: list[tuple[int, int, int]]) -> bool:
    h, w = board_owner.shape
    for r, c, _ in placement:
        if not (0 <= r < h and 0 <= c < w) or board_owner[r, c] != -1:
            return False
    return True


def _apply(
    board_owner: np.ndarray,
    board_color: np.ndarray,
    piece_idx: int,
    placement: list[tuple[int, int, int]],
    *,
    clear: bool = False,
) -> None:
    for r, c, col in placement:
        if clear:
            board_owner[r, c] = -1
            board_color[r, c] = -1
        else:
            board_owner[r, c] = piece_idx
            board_color[r, c] = col


def _contact_state_after_placing(
    board_owner: np.ndarray,
    board_color: np.ndarray,
    pieces: list[dict],
    piece_idx: int,
    placement: list[tuple[int, int, int]],
    pair_to_connector: dict[frozenset[int], int],
) -> dict[frozenset[int], int] | None:
    if not _placeable(board_owner, placement):
        return None

    h, w = board_owner.shape
    next_map = dict(pair_to_connector)
    new_body = int(pieces[piece_idx]["body_color"])
    new_temp = _temperature(new_body)

    for r, c, col in placement:
        for nr, nc in _neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            other_idx = int(board_owner[nr, nc])
            if other_idx == -1:
                continue
            other_col = int(board_color[nr, nc])
            other_body = int(pieces[other_idx]["body_color"])
            if new_temp == _temperature(other_body):
                return None
            if int(col) in CONNECTOR_COLORS and other_col in CONNECTOR_COLORS:
                if int(col) != other_col:
                    return None
                pair = frozenset({piece_idx, other_idx})
                prev = next_map.get(pair)
                if prev is None:
                    next_map[pair] = int(col)
                elif prev != int(col):
                    return None
    return next_map


def _final_board_is_valid(board_owner: np.ndarray, board_color: np.ndarray, pieces: list[dict]) -> bool:
    h, w = board_owner.shape
    if np.any(board_owner < 0):
        return False
    if int(board_color[0, 0]) != MARKER_COLOR or int(np.sum(board_color == MARKER_COLOR)) != 1:
        return False

    pair_to_connector: dict[frozenset[int], int] = {}
    seen_contacts = set()
    for r in range(h):
        for c in range(w):
            for nr, nc in ((r + 1, c), (r, c + 1)):
                if nr >= h or nc >= w:
                    continue
                a = int(board_owner[r, c])
                b = int(board_owner[nr, nc])
                if a == b:
                    continue
                col_a = int(board_color[r, c])
                col_b = int(board_color[nr, nc])
                if _temperature(int(pieces[a]["body_color"])) == _temperature(int(pieces[b]["body_color"])):
                    return False
                pair = frozenset({a, b})
                seen_contacts.add(pair)
                if col_a in CONNECTOR_COLORS and col_b in CONNECTOR_COLORS:
                    if col_a != col_b:
                        return False
                    prev = pair_to_connector.get(pair)
                    if prev is None:
                        pair_to_connector[pair] = col_a
                    elif prev != col_a:
                        return False
    return len(seen_contacts) == 4 and len(pair_to_connector) == 4


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
    board_owner = np.full((H, W), -1, dtype=int)
    board_color = np.full((H, W), -1, dtype=int)
    piece_vars = [_variants(p["cells"]) for p in pieces]

    anchor_idx = next((i for i, p in enumerate(pieces) if p["name"] == "anchor"), None)
    assert anchor_idx is not None, "an anchor piece is required"

    remaining = [i for i in range(len(pieces)) if i != anchor_idx]

    anchor_placements: list[tuple[list[tuple[int, int, int]], dict[frozenset[int], int]]] = []
    for var in piece_vars[anchor_idx]:
        marker_pts = [(r, c) for r, c, col in var if int(col) == MARKER_COLOR]
        if len(marker_pts) != 1:
            continue
        mr, mc = marker_pts[0]
        placed = [(r - mr, c - mc, col) for r, c, col in var]
        if all(0 <= r < H and 0 <= c < W for r, c, _ in placed):
            anchor_placements.append((placed, {}))

    for anchor_place, pair_map in anchor_placements:
        if not _placeable(board_owner, anchor_place):
            continue
        _apply(board_owner, board_color, anchor_idx, anchor_place)

        def backtrack(rem: list[int], contact_map: dict[frozenset[int], int]) -> bool:
            if not rem:
                return _final_board_is_valid(board_owner, board_color, pieces)

            best_idx = None
            best_options = None
            for pi in rem:
                opts = []
                for var in piece_vars[pi]:
                    for placement in _all_placements_for_variant(var, H, W):
                        next_map = _contact_state_after_placing(board_owner, board_color, pieces, pi, placement, contact_map)
                        if next_map is not None:
                            opts.append((placement, next_map))
                if not opts:
                    return False
                if best_options is None or len(opts) < len(best_options):
                    best_idx = pi
                    best_options = opts
                    if len(best_options) == 1:
                        break

            assert best_idx is not None and best_options is not None
            next_rem = [x for x in rem if x != best_idx]
            for placement, next_map in best_options:
                _apply(board_owner, board_color, best_idx, placement)
                if backtrack(next_rem, next_map):
                    return True
                _apply(board_owner, board_color, best_idx, placement, clear=True)
            return False

        if backtrack(remaining, pair_map):
            return board_color.copy()
        _apply(board_owner, board_color, anchor_idx, anchor_place, clear=True)
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble four scattered multi-color puzzle pieces into one square while respecting connector stripes.

    Input structure:
    - Background is the most common color. The input contains exactly four 8-connected pieces scattered apart.
    - One piece is the anchor: it contains the unique grey marker pixel 5 at one corner.
    - Each piece has one body color from {6,7,8,9} plus stripe cells from {1,2,3,4} on the sides that should touch neighbors.

    Rule:
    - Rotate the anchor so the grey marker lands at output position (0,0), then pack the other pieces into the same square tiling as the original task.
    - A placement is valid only if touching cells from two different pieces carry the same stripe color, and the two touching pieces have opposite warm/cool body classes.
    - The output is the fully assembled square with all body colors, stripe colors, and the grey marker preserved.

    Key insight:
    - Geometry alone is no longer enough. The stripe colors identify which edges are allowed to meet, and the body colors add a second check: every adjacency in the assembled rectangle alternates warm and cool pieces.
    """
    g = np.asarray(grid, dtype=int)
    pieces, _bg = _extract_pieces(g)
    area = sum(len(p["cells"]) for p in pieces)
    for H, W in _output_dims(area):
        solved = _solve_packing(pieces, H, W)
        if solved is not None:
            return solved.astype(int)
    raise AssertionError("packing failed for the fused geometry+color constraints")


def _build_ownership(N: int, p: int, tab_specs: dict[str, tuple[int, str] | None]) -> np.ndarray | None:
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

    for q in LABELS:
        mask = (own == q).astype(int)
        if mask.sum() == 0 or not is_contiguous(mask, background=0, connectivity=8):
            return None
    if _adjacency_pairs_from_labels(own) != EXPECTED_ADJACENCY_PAIRS:
        return None
    return own


def _random_tab_specs(N: int) -> dict[str, tuple[int, str] | None]:
    specs: dict[str, tuple[int, str] | None] = {}
    for name in TAB_NAMES:
        depth = random.randint(1, N - 1)
        if name.startswith("H"):
            direction = random.choice(["up", "down"])
        else:
            direction = random.choice(["left", "right"])
        specs[name] = (depth, direction)
    return specs


def _paint_connectors(own: np.ndarray, out: np.ndarray) -> None:
    h, w = own.shape
    foreign_neighbors: dict[tuple[int, int], set[int]] = {}
    for r in range(h):
        for c in range(w):
            here = int(own[r, c])
            others = set()
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w:
                    there = int(own[nr, nc])
                    if there != here:
                        others.add(there)
            foreign_neighbors[(r, c)] = others

    for r in range(h):
        for c in range(w):
            for dr, dc in ((1, 0), (0, 1)):
                nr, nc = r + dr, c + dc
                if nr >= h or nc >= w:
                    continue
                a = int(own[r, c])
                b = int(own[nr, nc])
                if a == b:
                    continue
                if foreign_neighbors[(r, c)] != {b} or foreign_neighbors[(nr, nc)] != {a}:
                    continue
                conn = CONNECTOR_BY_PAIR[frozenset({a, b})]
                out[r, c] = conn
                out[nr, nc] = conn


def _extract_piece_cells(own: np.ndarray, painted: np.ndarray, label: int) -> list[tuple[int, int, int]]:
    rows, cols = np.where(own == label)
    min_r = int(rows.min())
    min_c = int(cols.min())
    cells: list[tuple[int, int, int]] = []
    for r, c in zip(rows, cols):
        cells.append((int(r) - min_r, int(c) - min_c, int(painted[int(r), int(c)])))
    return cells


def _cells_to_patch(cells: list[tuple[int, int, int]]) -> np.ndarray:
    h = max(r for r, _, _ in cells) + 1
    w = max(c for _, c, _ in cells) + 1
    patch = np.zeros((h, w), dtype=int)
    for r, c, col in cells:
        patch[r, c] = col
    return patch


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


def _make_expected_output(own: np.ndarray) -> np.ndarray:
    colors = dict(BODY_COLOR_BY_LABEL)
    cools = COOL_BODY_COLORS.copy()
    random.shuffle(cools)
    colors[TR] = cools[0]
    colors[BL] = cools[1]

    out = np.zeros(own.shape, dtype=int)
    for label, body in colors.items():
        out[own == label] = body
    _paint_connectors(own, out)
    out[0, 0] = MARKER_COLOR
    return out


def _piece_has_visible_body(cells: list[tuple[int, int, int]]) -> bool:
    body_colors = sorted({int(col) for _, _, col in cells if int(col) in BODY_COLORS})
    if len(body_colors) != 1:
        return False
    body = body_colors[0]
    return sum(1 for _, _, col in cells if int(col) == body) >= 2


def _gen_attempt() -> np.ndarray | None:
    N = random.randint(N_MIN, N_MAX)
    p = random.randint(1, N - 1)

    own = _build_ownership(N, p, _random_tab_specs(N))
    if own is None:
        return None

    expected = _make_expected_output(own)
    pieces = [_extract_piece_cells(own, expected, label) for label in LABELS]
    if not all(_piece_has_visible_body(cells) for cells in pieces):
        return None

    transformed = [_rotate_cells(pieces[0], random.randint(0, 3))]
    transformed.extend(_rotate_cells(cells, random.randint(0, 3)) for cells in pieces[1:])

    canvas = _scatter_patches([_cells_to_patch(cells) for cells in transformed])
    if canvas is None:
        return None

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
