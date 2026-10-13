import numpy as np

from arc_utils import DIR8 as DIRS8
from arc_utils import get_bg as _bg_color


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
            for dr, dc in DIRS8:
                nr, nc = r + dr, col + dc
                if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and g[nr, nc] != bg:
                    return c
        return singletons[0]
    if 5 in counts:
        return 5
    return min(colors, key=lambda c: counts[c]) if colors else None


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
        for dr, dc in DIRS8:
            nr, nc = mr + dr, mc + dc
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


def transform(grid: np.ndarray) -> np.ndarray:
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
    if not pieces:
        return g.copy()

    area = sum(len(p["cells"]) for p in pieces)
    for H, W in _dims(area):
        solved = _solve_packing(pieces, H, W)
        if solved is None:
            continue
        return solved.astype(int)

    # Fallback: return tight crop of non-bg if packing fails.
    mask = g != bg
    if not np.any(mask):
        return g.copy()
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return g[r0 : r1 + 1, c0 : c1 + 1].copy()
