# Combo variant of a47bf94d (index 13)
# combo_spec: D11/1+D15/2
# Name: 四边环形布局+双面板输出

from __future__ import annotations

import random

import numpy as np

from arc_utils import get_bg, neighbors4

# Design analysis for the fused variant
# a) Concrete behavior:
#    The original task placed channels on two opposite sides and returned one grid.
#    Here the network is a single square ring with four spokes, one token anchored to
#    each side (N/E/S/W). Each source token still follows the same local shape logic:
#    solid -> in-place diamond + transmitted X, X -> in-place X + transmitted diamond.
#    The transmitted result is no longer shown in the same canvas. The output splits
#    into two side-by-side panels: left shows only the source-side result, right shows
#    only the received result at the next clockwise side of the ring.
# b) Complexity calibration:
#    I keep exactly four sides active in every sample so the ring layout is visually
#    obvious, but vary grid size, ring size, token colors, and which 1-2 sides use X.
#    The generator samples one-X and two-X cases evenly so both source-shape branches
#    are exposed across examples without making the task visually noisy.
#    That keeps the transformation learnable while still making the fusion visibly
#    different from the original vertical permutation layout.
# c) Hidden assumptions from the original solver that no longer hold:
#    - Output size is no longer equal to input size.
#    - Tokens are no longer restricted to top/bottom rows.
#    - Wire tracing does not rely on "far endpoint of the same column group"; instead,
#      the destination is determined by side order on the four-sided ring.
#    The new solver therefore detects endpoint directions directly and builds the
#    source/destination mapping from side identity rather than column permutations.
# d) Placement legality:
#    The generator picks token centers first, then computes endpoint rows/cols and a
#    ring half-size that stays strictly inside those endpoints. This guarantees that
#    3x3 tokens fit, spokes do not overlap tokens except at the intended adjacency,
#    and the ring stays inside bounds.
# e) Visual observability:
#    Humans can see four edge tokens feeding one central ring. Across examples, the
#    left panel always keeps the source-side transforms while the right panel always
#    shows the received complements shifted one side clockwise, making the panel split
#    and routing rule inferable from a few pairs.
# f) Visual cleanliness:
#    There is only one router, four spokes, and at most two thin 9-colored widened
#    ring segments. No overlapping token regions are allowed, and the panel divider is
#    a full-height column so the two output roles are easy to distinguish.
# g) New solver assumptions and generator support:
#    The solver assumes exactly one detected token is attached to each of the four
#    degree-1 spokes, with the token center two cells outward from that endpoint.
#    The generator enforces those anchors explicitly and validates the final output
#    against a manual renderer before returning the sample.

_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_SIDE_ORDER = ["N", "E", "S", "W"]
_DIR_TO_SIDE = {(-1, 0): "N", (0, 1): "E", (1, 0): "S", (0, -1): "W"}
_CLOCKWISE = {"N": "E", "E": "S", "S": "W", "W": "N"}

_DIAMOND = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_X = [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)]
_SOLID = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 0), (0, 1),
    (1, -1), (1, 0), (1, 1),
]

DIVIDER_COLOR = 5
TOKEN_COLORS_POOL = [1, 2, 3, 4, 6, 7]
GRID_H_RANGE = (19, 25)
GRID_W_RANGE = (19, 25)
TOKEN_MARGIN_RANGE = (2, 3)
RING_HALF_RANGE = (2, 4)
GEN_MAX_RETRIES = 200
X_COUNT_OPTIONS = [1, 2]


def _cells_from_offsets(center: tuple[int, int], offsets: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cr, cc = center
    return [(cr + dr, cc + dc) for dr, dc in offsets]


def _isolated_color(g: np.ndarray, bg: int, cells: list[tuple[int, int]], color: int) -> bool:
    h, w = g.shape
    own = set(cells)
    for r, c in cells:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in own and int(g[nr, nc]) == color:
                return False
    return True


def _find_solid_3x3(g: np.ndarray, bg: int) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
    h, w = g.shape
    found: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            sub = g[r0:r0 + 3, c0:c0 + 3]
            color = int(sub[0, 0])
            if color == bg or np.any(sub != color):
                continue
            cells = [(r0 + dr, c0 + dc) for dr in range(3) for dc in range(3)]
            if _isolated_color(g, bg, cells, color):
                found[(r0 + 1, c0 + 1)] = (color, (r0, c0))
    return found


def _find_x_3x3(g: np.ndarray, bg: int) -> dict[tuple[int, int], int]:
    h, w = g.shape
    mask = np.zeros((3, 3), dtype=bool)
    for rr, cc in [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)]:
        mask[rr, cc] = True

    found: dict[tuple[int, int], int] = {}
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            sub = g[r0:r0 + 3, c0:c0 + 3]
            color = int(sub[0, 0])
            if color == bg:
                continue
            if np.any(sub[mask] != color) or np.any(sub[~mask] != bg):
                continue
            center = (r0 + 1, c0 + 1)
            if _isolated_color(g, bg, _cells_from_offsets(center, _X), color):
                found[center] = color
    return found


def _find_diamond_3x3(g: np.ndarray, bg: int) -> dict[tuple[int, int], int]:
    h, w = g.shape
    mask = np.zeros((3, 3), dtype=bool)
    for rr, cc in [(0, 1), (1, 0), (1, 2), (2, 1)]:
        mask[rr, cc] = True

    found: dict[tuple[int, int], int] = {}
    for cr in range(1, h - 1):
        for cc in range(1, w - 1):
            sub = g[cr - 1:cr + 2, cc - 1:cc + 2]
            vals = sub[mask]
            if len(vals) == 0:
                continue
            color = int(vals[0])
            if color == bg or np.any(vals != color) or np.any(sub[~mask] != bg):
                continue
            center = (cr, cc)
            if _isolated_color(g, bg, _cells_from_offsets(center, _DIAMOND), color):
                found[center] = color
    return found


def _collect_tokens(g: np.ndarray, bg: int) -> dict[tuple[int, int], dict[str, object]]:
    tokens: dict[tuple[int, int], dict[str, object]] = {}

    for center, (color, (r0, c0)) in _find_solid_3x3(g, bg).items():
        tokens[center] = {
            "shape": "solid",
            "color": color,
            "cells": [(r0 + dr, c0 + dc) for dr in range(3) for dc in range(3)],
        }
    for center, color in _find_x_3x3(g, bg).items():
        tokens[center] = {
            "shape": "x",
            "color": color,
            "cells": _cells_from_offsets(center, _X),
        }
    for center, color in _find_diamond_3x3(g, bg).items():
        if center not in tokens:
            tokens[center] = {
                "shape": "diamond",
                "color": color,
                "cells": _cells_from_offsets(center, _DIAMOND),
            }
    return tokens


def _erase_tokens(g: np.ndarray, tokens: dict[tuple[int, int], dict[str, object]], bg: int) -> np.ndarray:
    base = g.copy()
    for token in tokens.values():
        for r, c in token["cells"]:
            base[r, c] = bg
    return base


def _find_endpoints(track: np.ndarray) -> list[tuple[int, int]]:
    endpoints: list[tuple[int, int]] = []
    pts = list(map(tuple, np.argwhere(track)))
    for r, c in pts:
        deg = sum(1 for nr, nc in neighbors4(int(r), int(c)) if 0 <= nr < track.shape[0] and 0 <= nc < track.shape[1] and track[nr, nc])
        if deg == 1:
            endpoints.append((int(r), int(c)))
    return endpoints


def _stamp(grid: np.ndarray, center: tuple[int, int], color: int, offsets: list[tuple[int, int]]) -> None:
    cr, cc = center
    for dr, dc in offsets:
        grid[cr + dr, cc + dc] = color


def _classify_sources(
    base: np.ndarray, tokens: dict[tuple[int, int], dict[str, object]]
) -> dict[str, dict[str, object]]:
    track = base != get_bg(base)
    endpoints = _find_endpoints(track)
    sources: dict[str, dict[str, object]] = {}
    for er, ec in endpoints:
        for dr, dc in _N4:
            center = (er + 2 * dr, ec + 2 * dc)
            token = tokens.get(center)
            if token is None:
                continue
            side = _DIR_TO_SIDE[(dr, dc)]
            sources[side] = {
                "center": center,
                "shape": token["shape"],
                "color": token["color"],
            }
            break

    if set(sources) != set(_SIDE_ORDER):
        raise ValueError(f"expected four side-attached tokens, found sides={sorted(sources)}")
    return sources


def _render_dual_panels(base: np.ndarray, sources: dict[str, dict[str, object]], bg: int) -> np.ndarray:
    h, w = base.shape
    left = base.copy()
    right = base.copy()

    for side in _SIDE_ORDER:
        center = sources[side]["center"]
        color = int(sources[side]["color"])
        shape = str(sources[side]["shape"])
        if shape == "solid":
            _stamp(left, center, color, _DIAMOND)
            _stamp(right, sources[_CLOCKWISE[side]]["center"], color, _X)
        elif shape == "x":
            _stamp(left, center, color, _X)
            _stamp(right, sources[_CLOCKWISE[side]]["center"], color, _DIAMOND)
        elif shape == "diamond":
            _stamp(left, center, color, _DIAMOND)

    out = np.full((h, 2 * w + 1), bg, dtype=int)
    out[:, :w] = left
    out[:, w] = DIVIDER_COLOR
    out[:, w + 1:] = right
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split a four-sided token router into a source panel and a destination panel.

    Input structure:
    - A single ring-shaped pipe network sits near the center, with one spoke ending on
      each side of the grid.
    - Each spoke has exactly one 3x3 token attached two cells outward from the endpoint.
      Tokens are solid blocks or X-patterns in distinct colors.

    Rule:
    - Read the four edge tokens as sources on the north, east, south, and west spokes.
    - In the left output panel, keep only the source-side result: solid becomes a
      diamond, X stays X.
    - In the right output panel, place the transmitted complement at the next clockwise
      side of the ring: solid emits X, X emits diamond.
    - The background router is copied into both panels, which are concatenated with a
      full-height divider column.

    Key insight:
    - The central ring turns the puzzle from a top-bottom pairing task into an ordered
      four-side cycle. The solver must separate "what happens at the source" from
      "where the transmitted counterpart appears", then render those two roles in
      different panels.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    tokens = _collect_tokens(g, bg)
    base = _erase_tokens(g, tokens, bg)
    sources = _classify_sources(base, tokens)
    return _render_dual_panels(base, sources, bg)


def _draw_ring_router(
    h: int,
    w: int,
    top_center: tuple[int, int],
    right_center: tuple[int, int],
    bottom_center: tuple[int, int],
    left_center: tuple[int, int],
) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    grid = np.zeros((h, w), dtype=int)

    cy, cx = h // 2, w // 2
    top_ep = (top_center[0] + 2, top_center[1])
    right_ep = (right_center[0], right_center[1] - 2)
    bottom_ep = (bottom_center[0] - 2, bottom_center[1])
    left_ep = (left_center[0], left_center[1] + 2)

    max_half_h = min(cy - top_ep[0] - 2, bottom_ep[0] - cy - 2)
    max_half_w = min(cx - left_ep[1] - 2, right_ep[1] - cx - 2)
    if max_half_h < RING_HALF_RANGE[0] or max_half_w < RING_HALF_RANGE[0]:
        raise ValueError("grid too tight for ring router")

    half_h = random.randint(RING_HALF_RANGE[0], min(RING_HALF_RANGE[1], max_half_h))
    half_w = random.randint(RING_HALF_RANGE[0], min(RING_HALF_RANGE[1], max_half_w))
    r0, r1 = cy - half_h, cy + half_h
    c0, c1 = cx - half_w, cx + half_w

    grid[r0, c0:c1 + 1] = 8
    grid[r1, c0:c1 + 1] = 8
    grid[r0:r1 + 1, c0] = 8
    grid[r0:r1 + 1, c1] = 8

    grid[top_ep[0]:r0 + 1, cx] = 8
    grid[r1:bottom_ep[0] + 1, cx] = 8
    grid[cy, left_ep[1]:c0 + 1] = 8
    grid[cy, c1:right_ep[1] + 1] = 8

    for side in random.sample(_SIDE_ORDER, random.randint(0, 2)):
        if side == "N":
            grid[r0, cx - 1:cx + 2] = 9
        elif side == "S":
            grid[r1, cx - 1:cx + 2] = 9
        elif side == "W":
            grid[cy - 1:cy + 2, c0] = 9
        else:
            grid[cy - 1:cy + 2, c1] = 9

    centers = {
        "N": top_center,
        "E": right_center,
        "S": bottom_center,
        "W": left_center,
    }
    return grid, centers


def _sample_layout() -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)

    top_row = random.randint(*TOKEN_MARGIN_RANGE)
    bottom_row = h - 1 - random.randint(*TOKEN_MARGIN_RANGE)
    left_col = random.randint(*TOKEN_MARGIN_RANGE)
    right_col = w - 1 - random.randint(*TOKEN_MARGIN_RANGE)

    cy, cx = h // 2, w // 2
    top_center = (top_row, cx)
    right_center = (cy, right_col)
    bottom_center = (bottom_row, cx)
    left_center = (cy, left_col)

    if bottom_center[0] - top_center[0] < 8 or right_center[1] - left_center[1] < 8:
        raise ValueError("token anchors too close")

    return _draw_ring_router(h, w, top_center, right_center, bottom_center, left_center)


def _build_input() -> tuple[np.ndarray, np.ndarray]:
    grid, centers = _sample_layout()
    colors = random.sample(TOKEN_COLORS_POOL, 4)
    x_count = random.choice(X_COUNT_OPTIONS)
    x_sides = set(random.sample(_SIDE_ORDER, x_count))

    token_specs: dict[str, tuple[int, list[tuple[int, int]]]] = {}
    for side, color in zip(_SIDE_ORDER, colors):
        offsets = _X if side in x_sides else _SOLID
        token_specs[side] = (color, offsets)
        _stamp(grid, centers[side], color, offsets)

    expected_base = _erase_tokens(grid, _collect_tokens(grid, 0), 0)
    expected_sources = {
        side: {
            "center": centers[side],
            "shape": "x" if side in x_sides else "solid",
            "color": token_specs[side][0],
        }
        for side in _SIDE_ORDER
    }
    expected = _render_dual_panels(expected_base, expected_sources, 0)
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_RETRIES):
        try:
            inp, expected = _build_input()
            if np.array_equal(main(inp), expected):
                return inp
        except Exception:
            continue
    raise RuntimeError(f"Failed to generate valid input after {GEN_MAX_RETRIES} attempts")
