import numpy as np

from arc_utils import get_bg as _bg_color, neighbors8
import random
from arc_utils import random_free_location, is_contiguous


def _rotate_cells(cells: list[tuple[int, int, int]], k: int) -> list[tuple[int, int, int]]:
    pts = [(r, c, col) for r, c, col in cells]
    for _ in range(k % 4):
        pts = [(c, -r, col) for r, c, col in pts]  # 90° clockwise around origin
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
    singletons = [c for c in colors if counts[c] == 1]
    if singletons:
        # Prefer a singleton that is 8-neigh adjacent to some other non-bg pixel.
        for c in singletons:
            r, col = map(int, np.argwhere(g == c)[0])
            for nr, nc in neighbors8(r, col):
                if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and g[nr, nc] != bg:
                    return c
        assert False, "no adjacent singleton found — input must have adjacent singleton marker"


def _extract_pieces(g: np.ndarray) -> tuple[list[dict], int]:
    bg = _bg_color(g)
    colors = [c for c in sorted(set(map(int, g.ravel()))) if c != bg]
    if not colors:
        return [], bg

    marker_color = _find_marker_color(g, bg)
    marker_pos = None
    if marker_color is not None and int(np.sum(g == marker_color)) == 1:
        marker_pos = tuple(map(int, np.argwhere(g == marker_color)[0]))

    # Anchor color is the color 8-neigh adjacent to the marker.
    anchor_color = None
    if marker_pos is not None:
        mr, mc = marker_pos
        neigh = set()
        for nr, nc in neighbors8(mr, mc):
            if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]:
                v = int(g[nr, nc])
                if v != bg and v != marker_color:
                    neigh.add(v)
        if len(neigh) == 1:
            anchor_color = next(iter(neigh))

    pieces: list[dict] = []
    used = set()

    if anchor_color is not None and marker_pos is not None:
        # Anchor is a multi-color piece: anchor_color pixels plus the marker pixel.
        coords = []
        for r, c in np.argwhere(g == anchor_color):
            coords.append((int(r), int(c), int(anchor_color)))
        mr, mc = marker_pos
        coords.append((mr, mc, int(marker_color)))
        # Normalize to local coordinates.
        min_r = min(r for r, _, _ in coords)
        min_c = min(c for _, c, _ in coords)
        coords = [(r - min_r, c - min_c, col) for r, c, col in coords]
        pieces.append(
            {
                "name": "anchor",
                "cells": coords,
                "marker_color": int(marker_color),
            }
        )
        used.add(anchor_color)
        used.add(marker_color)

    for col in colors:
        if col in used:
            continue
        coords = [(int(r), int(c), int(col)) for r, c in np.argwhere(g == col)]
        min_r = min(r for r, _, _ in coords)
        min_c = min(c for _, c, _ in coords)
        coords = [(r - min_r, c - min_c, col) for r, c, col in coords]
        pieces.append({"name": f"c{col}", "cells": coords})

    return pieces, bg


def _dims(area: int) -> list[tuple[int, int]]:
    pairs = []
    for h in range(1, int(area**0.5) + 1):
        if area % h == 0:
            w = area // h
            pairs.append((h, w))
            if h != w:
                pairs.append((w, h))
    # Prefer close-to-square, then smaller height.
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

    # Build variants + placements per piece.
    piece_vars: list[list[list[tuple[int, int, int]]]] = []
    for p in pieces:
        piece_vars.append(_variants(p["cells"]))

    # Place anchor first if present: marker must end up at (0,0).
    remaining = list(range(len(pieces)))
    anchor_idx = next((i for i, p in enumerate(pieces) if p["name"] == "anchor"), None)
    if anchor_idx is not None:
        remaining.remove(anchor_idx)
        marker_col = pieces[anchor_idx]["marker_color"]
        anchor_solutions = []
        for var in piece_vars[anchor_idx]:
            marker_pts = [(r, c) for r, c, col in var if col == marker_col]
            if len(marker_pts) != 1:
                continue
            mr, mc = marker_pts[0]
            tr, tc = -mr, -mc
            placed = [(r + tr, c + tc, col) for r, c, col in var]
            if all(0 <= r < H and 0 <= c < W for r, c, _ in placed):
                anchor_solutions.append(placed)
        if not anchor_solutions:
            return None
        # Try each feasible anchor orientation (usually 1).
        for anchor_place in anchor_solutions:
            if not _placeable(board, anchor_place):
                continue
            _apply(board, anchor_place)

            # Backtracking for remaining pieces.
            def backtrack(rem: list[int]) -> bool:
                if not rem:
                    return True

                # Choose piece with fewest fits under current board.
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
                        if best_count == 1:
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
                return board
            _apply(board, anchor_place, -1)
        return None

    # No anchor: just pack all pieces.
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
        assert best_pi is not None and best_options is not None
        next_rem = [x for x in rem if x != best_pi]
        for placement in best_options:
            _apply(board, placement)
            if backtrack(next_rem):
                return True
            _apply(board, placement, -1)
        return False

    if backtrack(remaining):
        return board
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble colored rectangular sprites into a single tiled rectangle, anchored by a marker pixel.

    Input structure:
    - bg = most common color. Several solid-colored rectangular sprites scattered on the grid.
    - One sprite has a unique single-pixel marker (different color) at one corner — this is the anchor.

    Rule:
    - Identify the anchor sprite (contains the marker pixel) and the remaining pieces.
    - Rotate the anchor so the marker pixel lands at position (0,0) of the output.
    - Pack all remaining pieces (allowing 90° rotations) to tile a complete rectangle with no gaps.

    Key insight: The lone marker pixel picks the anchor and fixes the output orientation — it must sit at the top-left corner.
    """
    g = np.asarray(grid, dtype=int)
    pieces, bg = _extract_pieces(g)
    assert pieces, "grid has no non-background colors — input must contain colored pieces to assemble"

    area = sum(len(p["cells"]) for p in pieces)
    for H, W in _dims(area):
        solved = _solve_packing(pieces, H, W)
        if solved is None:
            continue
        return solved.astype(int)

    assert False, "packing failed for all candidate dimensions — input pieces must tile a valid rectangle"

# --------------- generator (Claude attempt 1) ---------------

# ── Configuration ────────────────────────────────────────────
N_MIN, N_MAX = 2, 7              # half-side of output grid
ANCHOR_COLOR = 7                 # color of the anchor (TL) piece
MARKER_COLOR = 5                 # singleton marker pixel color
COLOR_POOL = [1, 2, 3, 4, 6, 8, 9]  # non-anchor piece color candidates
CANVAS_H, CANVAS_W = 20, 20     # input grid dimensions
PLACEMENT_PADDING = 1            # min gap between piece bboxes
MAX_ATTEMPTS = 300               # retries in generate_input
TAB_NAMES = ['H_left', 'H_right', 'V_top', 'V_bot']

# Quadrant identifiers
TL, TR, BL, BR = 0, 1, 2, 3


def _build_ownership(N: int, p: int, tab_specs: dict) -> np.ndarray | None:
    """Build 2Nx2N quadrant-ownership grid with tab intrusions.

    tab_specs maps tab name -> (depth, direction) or None.
    Returns (2N,2N) array with values TL/TR/BL/BR, or None if any
    quadrant region is empty or not 8-connected.
    """
    S = 2 * N
    own = np.zeros((S, S), dtype=int)
    own[:N, :N] = TL
    own[:N, N:] = TR
    own[N:, :N] = BL
    own[N:, N:] = BR

    # H-tabs: single-column strips crossing the horizontal boundary (row N)
    for kind, col, up_owner, down_owner in [
        ('H_left',  p,         BL, TL),
        ('H_right', S - 1 - p, BR, TR),
    ]:
        spec = tab_specs.get(kind)
        if spec is None:
            continue
        depth, direction = spec
        if direction == 'up':
            own[N - depth:N, col] = up_owner
        else:
            own[N:N + depth, col] = down_owner

    # V-tabs: single-row strips crossing the vertical boundary (col N)
    for kind, row, left_owner, right_owner in [
        ('V_top', p,         TR, TL),
        ('V_bot', S - 1 - p, BR, BL),
    ]:
        spec = tab_specs.get(kind)
        if spec is None:
            continue
        depth, direction = spec
        if direction == 'left':
            own[row, N - depth:N] = left_owner
        else:
            own[row, N:N + depth] = right_owner

    # Verify all four quadrant regions are non-empty and 8-connected
    for q in (TL, TR, BL, BR):
        mask = (own == q).astype(int)
        if mask.sum() == 0:
            return None
        if not is_contiguous(mask, background=0, connectivity=8):
            return None
    return own


def _extract_piece(own: np.ndarray, q: int, color: int,
                   marker_pos: tuple[int, int] | None = None) -> list[tuple[int, int, int]]:
    """Extract cells for quadrant *q* in local (0-based) coordinates.

    All cells get *color*, except *marker_pos* which gets MARKER_COLOR.
    """
    rows, cols = np.where(own == q)
    min_r, min_c = rows.min(), cols.min()
    cells = [(int(r) - min_r, int(c) - min_c, color) for r, c in zip(rows, cols)]
    if marker_pos is not None:
        mr, mc = marker_pos[0] - min_r, marker_pos[1] - min_c
        cells = [(r, c, MARKER_COLOR if (r, c) == (mr, mc) else col)
                 for r, c, col in cells]
    return cells


def _cells_to_patch(cells) -> np.ndarray:
    """Render cell list into a tight numpy patch (bg=0)."""
    h = max(r for r, _, _ in cells) + 1
    w = max(c for _, c, _ in cells) + 1
    patch = np.zeros((h, w), dtype=int)
    for r, c, col in cells:
        patch[r, c] = col
    return patch


def _random_tab_specs(N: int) -> dict:
    """Sample random tab specs for all four boundaries.

    At N=2, one tab is randomly dropped (spec says only 3 may be present).
    """
    specs = {}
    for tn in TAB_NAMES:
        depth = random.randint(1, N - 1)
        if tn.startswith('H'):
            direction = random.choice(['up', 'down'])
        else:
            direction = random.choice(['left', 'right'])
        specs[tn] = (depth, direction)

    if N == 2:
        specs[random.choice(TAB_NAMES)] = None
    return specs


def _scatter_patches(patches: list[np.ndarray]) -> np.ndarray | None:
    """Place patches on a canvas in random order with padding.

    Returns the canvas, or None if any patch doesn't fit.
    """
    canvas = np.zeros((CANVAS_H, CANVAS_W), dtype=int)
    order = list(range(len(patches)))
    random.shuffle(order)
    for i in order:
        patch = patches[i]
        try:
            r0, c0 = random_free_location(
                canvas, patch, transparent=0,
                padding=PLACEMENT_PADDING, padding_connectivity=8)
        except ValueError:
            return None
        region = canvas[r0:r0 + patch.shape[0], c0:c0 + patch.shape[1]]
        mask = patch != 0
        region[mask] = patch[mask]
    return canvas


def _gen_attempt() -> np.ndarray | None:
    N = random.randint(N_MIN, N_MAX)
    p = random.randint(1, N - 1)

    # Build quadrant ownership grid
    own = _build_ownership(N, p, _random_tab_specs(N))
    if own is None:
        return None

    # Assign colors
    pool = COLOR_POOL.copy()
    random.shuffle(pool)
    quad_colors = {TL: ANCHOR_COLOR, TR: pool[0], BL: pool[1], BR: pool[2]}

    # Build expected output (solver target)
    S = 2 * N
    output = np.zeros((S, S), dtype=int)
    for q, c in quad_colors.items():
        output[own == q] = c
    output[0, 0] = MARKER_COLOR

    # Extract pieces: anchor gets marker at (0,0)
    anchor_cells = _extract_piece(own, TL, ANCHOR_COLOR, marker_pos=(0, 0))
    other_cells = [_extract_piece(own, q, quad_colors[q]) for q in (TR, BL, BR)]

    # Rotate: anchor to random orientation, each other piece randomly
    all_cells = [_rotate_cells(anchor_cells, random.randint(0, 3))] + \
                [_rotate_cells(c, random.randint(0, 3)) for c in other_cells]

    # Scatter on canvas
    canvas = _scatter_patches([_cells_to_patch(c) for c in all_cells])
    if canvas is None:
        return None

    # Oracle: verify solver reproduces the expected output
    try:
        out = main(canvas)
    except Exception:
        return None
    if not np.array_equal(out, output):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError('generate_input: exhausted attempts')
