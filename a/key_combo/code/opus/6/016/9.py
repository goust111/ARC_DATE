from __future__ import annotations

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7
BAND_THICKNESS = 2
MARKER_DEPTH_MIN = 1
MARKER_DEPTH_MAX = 3
MARKER_PLACE_DEPTH_MIN = 2
MARKER_PLACE_DEPTH_MAX = 3

SIDES = ("top", "bottom", "left", "right")


def _opposite(side: str) -> str:
    return {
        "top": "bottom",
        "bottom": "top",
        "left": "right",
        "right": "left",
    }[side]


def _edge_distance(shape: tuple[int, int], side: str, r: int, c: int) -> int:
    h, w = shape
    if side == "top":
        return r
    if side == "bottom":
        return h - 1 - r
    if side == "left":
        return c
    if side == "right":
        return w - 1 - c
    raise ValueError(side)


def _detect_active_edge(grid: np.ndarray) -> str:
    orange_coords = np.argwhere(grid == ORANGE)
    assert orange_coords.size > 0, "expected orange edge markers"

    counts: dict[str, int] = {}
    for side in SIDES:
        counts[side] = sum(
            MARKER_DEPTH_MIN <= _edge_distance(grid.shape, side, int(r), int(c)) <= MARKER_DEPTH_MAX
            for r, c in orange_coords
        )

    best = max(counts.values())
    candidates = [side for side, count in counts.items() if count == best]
    assert best > 0 and len(candidates) == 1, "orange markers must identify one active edge"
    return candidates[0]


def _selected_lines(grid: np.ndarray, side: str) -> list[int]:
    if side in ("top", "bottom"):
        return [c for c in range(grid.shape[1]) if np.any(grid[:, c] == BLUE)]
    return [r for r in range(grid.shape[0]) if np.any(grid[r, :] == BLUE)]


def _transfer_only(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    moved_mask = np.zeros_like(g, dtype=bool)
    side = _detect_active_edge(g)
    lines = _selected_lines(g, side)
    h, w = g.shape

    if side == "top":
        for c in lines:
            out[h - BAND_THICKNESS : h, c] = g[:BAND_THICKNESS, c]
            moved_mask[h - BAND_THICKNESS : h, c] = True
    elif side == "bottom":
        for c in lines:
            out[:BAND_THICKNESS, c] = g[h - BAND_THICKNESS : h, c]
            moved_mask[:BAND_THICKNESS, c] = True
    elif side == "left":
        for r in lines:
            out[r, w - BAND_THICKNESS : w] = g[r, :BAND_THICKNESS]
            moved_mask[r, w - BAND_THICKNESS : w] = True
    else:
        assert side == "right"
        for r in lines:
            out[r, :BAND_THICKNESS] = g[r, w - BAND_THICKNESS : w]
            moved_mask[r, :BAND_THICKNESS] = True

    out[g == BLUE] = BG
    return out, moved_mask, _opposite(side)


def _find_cross_centers(grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    centers: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if (
                grid[r, c] == YELLOW
                and grid[r - 1, c] == YELLOW
                and grid[r + 1, c] == YELLOW
                and grid[r, c - 1] == YELLOW
                and grid[r, c + 1] == YELLOW
            ):
                centers.append((r, c))
    return centers


def _extend_periodic(out: np.ndarray, tile: np.ndarray, r0: int, c0: int, direction: tuple[int, int]) -> None:
    h, w = out.shape
    tile_h, tile_w = tile.shape
    dr, dc = direction

    if (dr, dc) == (0, 1):
        for step, tc in enumerate(range(c0 + tile_w, w)):
            src_c = step % tile_w
            for rr in range(r0, r0 + tile_h):
                val = int(tile[rr - r0, src_c])
                if val != BG and out[rr, tc] == BG:
                    out[rr, tc] = val
    elif (dr, dc) == (0, -1):
        for step, tc in enumerate(range(c0 - 1, -1, -1), start=1):
            src_c = (-step) % tile_w
            for rr in range(r0, r0 + tile_h):
                val = int(tile[rr - r0, src_c])
                if val != BG and out[rr, tc] == BG:
                    out[rr, tc] = val
    elif (dr, dc) == (1, 0):
        for step, tr in enumerate(range(r0 + tile_h, h)):
            src_r = step % tile_h
            for cc in range(c0, c0 + tile_w):
                val = int(tile[src_r, cc - c0])
                if val != BG and out[tr, cc] == BG:
                    out[tr, cc] = val
    elif (dr, dc) == (-1, 0):
        for step, tr in enumerate(range(r0 - 1, -1, -1), start=1):
            src_r = (-step) % tile_h
            for cc in range(c0, c0 + tile_w):
                val = int(tile[src_r, cc - c0])
                if val != BG and out[tr, cc] == BG:
                    out[tr, cc] = val
    else:
        raise ValueError("invalid direction")


def _trigger_specs(
    out: np.ndarray,
    moved_mask: np.ndarray,
    dest_side: str,
    center: tuple[int, int],
) -> list[tuple[np.ndarray, int, int, tuple[int, int]]]:
    h, w = out.shape
    r, c = center
    specs: list[tuple[np.ndarray, int, int, tuple[int, int]]] = []

    if dest_side == "top" and r == 1:
        for dc, direction in ((-2, (0, -1)), (2, (0, 1))):
            cc = c + dc
            if 0 <= cc < w and np.all(moved_mask[:BAND_THICKNESS, cc]):
                specs.append((out[:BAND_THICKNESS, cc : cc + 1].copy(), 0, cc, direction))
    elif dest_side == "bottom" and r == h - 2:
        for dc, direction in ((-2, (0, -1)), (2, (0, 1))):
            cc = c + dc
            if 0 <= cc < w and np.all(moved_mask[h - BAND_THICKNESS : h, cc]):
                specs.append(
                    (
                        out[h - BAND_THICKNESS : h, cc : cc + 1].copy(),
                        h - BAND_THICKNESS,
                        cc,
                        direction,
                    )
                )
    elif dest_side == "left" and c == 1:
        for dr, direction in ((-2, (-1, 0)), (2, (1, 0))):
            rr = r + dr
            if 0 <= rr < h and np.all(moved_mask[rr, :BAND_THICKNESS]):
                specs.append((out[rr : rr + 1, :BAND_THICKNESS].copy(), rr, 0, direction))
    elif dest_side == "right" and c == w - 2:
        for dr, direction in ((-2, (-1, 0)), (2, (1, 0))):
            rr = r + dr
            if 0 <= rr < h and np.all(moved_mask[rr, w - BAND_THICKNESS : w]):
                specs.append(
                    (
                        out[rr : rr + 1, w - BAND_THICKNESS : w].copy(),
                        rr,
                        w - BAND_THICKNESS,
                        direction,
                    )
                )
    return specs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move selected two-cell edge-band slices across the grid, then let nearby '+' crosses project them outward periodically.

    Input structure:
    - The background is black.
    - One grid edge carries a 2-pixel-thick red/green patterned band, and orange markers sit 1-3 cells from that same edge.
    - Sparse blue trail pixels lie on rows or columns perpendicular to the active edge, and one or two yellow '+' crosses sit inside the grid near the opposite side.

    Rule:
    - Find the edge with the most orange markers and treat it as the active source edge.
    - For every perpendicular row or column containing a blue trail pixel, copy that line's 2-cell edge-band slice to the opposite edge at the matching position, then erase all blue pixels while keeping the source band and orange markers.
    - For each yellow '+' whose outer-adjacent side now touches a transferred slice, extend that 2-cell slice periodically away from the '+' toward the boundary, writing only onto background cells and never overwriting the '+' or any other existing non-background pixel.

    Key insight:
    - The orange markers choose which edge supplies slices, but the yellow '+' only becomes meaningful after transfer: a moved slice landing beside one arm turns the cross into a directional periodic repeater.
    """
    out, moved_mask, dest_side = _transfer_only(grid)
    for center in _find_cross_centers(out):
        for tile, r0, c0, direction in _trigger_specs(out, moved_mask, dest_side, center):
            _extend_periodic(out, tile, r0, c0, direction)
    return out


def _paint_cross(grid: np.ndarray, center: tuple[int, int]) -> None:
    r, c = center
    grid[r, c] = YELLOW
    grid[r - 1, c] = YELLOW
    grid[r + 1, c] = YELLOW
    grid[r, c - 1] = YELLOW
    grid[r, c + 1] = YELLOW


def _paint_band(grid: np.ndarray, side: str, line_patterns: np.ndarray) -> None:
    h, w = grid.shape
    if side == "top":
        grid[:BAND_THICKNESS, :] = line_patterns.T
    elif side == "bottom":
        grid[h - BAND_THICKNESS : h, :] = line_patterns.T
    elif side == "left":
        grid[:, :BAND_THICKNESS] = line_patterns
    else:
        grid[:, w - BAND_THICKNESS : w] = line_patterns


def _pattern_for_lines(count: int, rng: np.random.Generator) -> np.ndarray:
    phase = int(rng.integers(0, 2))
    mode = int(rng.integers(0, 3))
    patterns = np.zeros((count, BAND_THICKNESS), dtype=int)
    for i in range(count):
        parity = (i + phase) % 2
        if mode == 0:
            pair = (RED, GREEN) if parity == 0 else (GREEN, RED)
        elif mode == 1:
            pair = (RED, GREEN) if (i // 2 + phase) % 2 == 0 else (GREEN, RED)
        else:
            pair = (RED, GREEN) if (i + (i // 3) + phase) % 2 == 0 else (GREEN, RED)
        patterns[i] = pair
    return patterns


def _edge_marker_positions(
    shape: tuple[int, int],
    side: str,
    count: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    h, w = shape
    positions: list[tuple[int, int]] = []
    tries = 0
    while len(positions) < count and tries < 500:
        tries += 1
        depth = int(rng.integers(MARKER_PLACE_DEPTH_MIN, MARKER_PLACE_DEPTH_MAX + 1))
        if side == "top":
            r, c = depth, int(rng.integers(2, w - 2))
        elif side == "bottom":
            r, c = h - 1 - depth, int(rng.integers(2, w - 2))
        elif side == "left":
            r, c = int(rng.integers(2, h - 2)), depth
        else:
            r, c = int(rng.integers(2, h - 2)), w - 1 - depth
        if (r, c) not in positions:
            positions.append((r, c))
    if len(positions) < count:
        raise RuntimeError("failed to place orange markers")
    return positions


def _separated_choices(
    candidates: list[int],
    need: int,
    min_gap: int,
    rng: np.random.Generator,
) -> list[int] | None:
    order = list(rng.permutation(candidates))
    chosen: list[int] = []
    for val in order:
        if all(abs(val - prev) >= min_gap for prev in chosen):
            chosen.append(int(val))
            if len(chosen) == need:
                return sorted(chosen)
    return None


def _empty_with_band(side: str, h: int, w: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    grid = np.zeros((h, w), dtype=int)
    line_count = w if side in ("top", "bottom") else h
    patterns = _pattern_for_lines(line_count, rng)
    _paint_band(grid, side, patterns)
    return grid, patterns


def _available_cross_centers(
    shape: tuple[int, int],
    dest_side: str,
    trigger_side: str,
) -> list[tuple[int, int, int]]:
    h, w = shape
    options: list[tuple[int, int, int]] = []
    if dest_side == "top":
        r = 1
        cols = range(3, w - 3) if trigger_side == "left" else range(2, w - 4)
        for c in cols:
            line = c - 2 if trigger_side == "left" else c + 2
            options.append((r, c, line))
    elif dest_side == "bottom":
        r = h - 2
        cols = range(3, w - 3) if trigger_side == "left" else range(2, w - 4)
        for c in cols:
            line = c - 2 if trigger_side == "left" else c + 2
            options.append((r, c, line))
    elif dest_side == "left":
        c = 1
        rows = range(3, h - 3) if trigger_side == "top" else range(2, h - 4)
        for r in rows:
            line = r - 2 if trigger_side == "top" else r + 2
            options.append((r, c, line))
    else:
        c = w - 2
        rows = range(3, h - 3) if trigger_side == "top" else range(2, h - 4)
        for r in rows:
            line = r - 2 if trigger_side == "top" else r + 2
            options.append((r, c, line))
    return options


def _cross_cells(center: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    return {(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}


def _can_place_cross(grid: np.ndarray, center: tuple[int, int]) -> bool:
    h, w = grid.shape
    r, c = center
    cells = _cross_cells(center)
    if not all(0 <= rr < h and 0 <= cc < w for rr, cc in cells):
        return False
    return all(grid[rr, cc] == BG for rr, cc in cells)


def _place_blue_points(
    grid: np.ndarray,
    side: str,
    lines: list[int],
    blocked: set[tuple[int, int]],
    rng: np.random.Generator,
) -> None:
    h, w = grid.shape
    for line in lines:
        count = int(rng.integers(1, 4))
        placed = 0
        tries = 0
        while placed < count and tries < 200:
            tries += 1
            if side in ("top", "bottom"):
                r = int(rng.integers(2, h - 2))
                cell = (r, line)
            else:
                c = int(rng.integers(2, w - 2))
                cell = (line, c)
            if cell in blocked or grid[cell] != BG:
                continue
            grid[cell] = BLUE
            blocked.add(cell)
            placed += 1
        if placed == 0:
            raise RuntimeError("failed to place blue trail pixels")


def _far_interior_cross_candidates(
    shape: tuple[int, int],
    source_side: str,
    forbidden_lines: set[int],
) -> list[tuple[int, int]]:
    h, w = shape
    candidates: list[tuple[int, int]] = []
    for r in range(2, h - 2):
        for c in range(2, w - 2):
            if source_side == "top" and r <= 3:
                continue
            if source_side == "bottom" and r >= h - 4:
                continue
            if source_side == "left" and c <= 3:
                continue
            if source_side == "right" and c >= w - 4:
                continue
            if source_side in ("top", "bottom") and (c - 2 in forbidden_lines or c + 2 in forbidden_lines):
                continue
            if source_side in ("left", "right") and (r - 2 in forbidden_lines or r + 2 in forbidden_lines):
                continue
            candidates.append((r, c))
    return candidates


def _generate_case(rng: np.random.Generator) -> np.ndarray | None:
    side = SIDES[int(rng.integers(0, len(SIDES)))]
    if side in ("top", "bottom"):
        h = int(rng.integers(14, 21))
        w = int(rng.integers(14, 21))
    else:
        h = int(rng.integers(14, 21))
        w = int(rng.integers(14, 21))

    grid, _patterns = _empty_with_band(side, h, w, rng)
    dest_side = _opposite(side)

    marker_count = int(rng.integers(3, 7))
    for cell in _edge_marker_positions((h, w), side, marker_count, rng):
        grid[cell] = ORANGE

    allowed_trigger_sides = ("left", "right") if dest_side in ("top", "bottom") else ("top", "bottom")
    trigger_side_1 = allowed_trigger_sides[int(rng.integers(0, 2))]
    options_1 = _available_cross_centers((h, w), dest_side, trigger_side_1)
    if not options_1:
        return None
    r1, c1, line1 = options_1[int(rng.integers(0, len(options_1)))]

    centers: list[tuple[int, int]] = [(r1, c1)]
    trigger_lines = {line1}
    trigger_meta = [(r1, c1, line1, trigger_side_1)]

    if rng.random() < 0.55:
        trigger_side_2 = allowed_trigger_sides[int(rng.integers(0, 2))]
        options_2 = [
            item
            for item in _available_cross_centers((h, w), dest_side, trigger_side_2)
            if abs(item[2] - line1) >= 3 and abs(item[0] - r1) + abs(item[1] - c1) >= 4
        ]
        if options_2:
            r2, c2, line2 = options_2[int(rng.integers(0, len(options_2)))]
            centers.append((r2, c2))
            trigger_lines.add(line2)
            trigger_meta.append((r2, c2, line2, trigger_side_2))

    forbidden_cross_cells: set[tuple[int, int]] = set()
    for center in centers:
        if not _can_place_cross(grid, center):
            return None
        forbidden_cross_cells |= _cross_cells(center)
        _paint_cross(grid, center)

    if len(centers) == 1 and rng.random() < 0.5:
        extra_candidates = [
            center
            for center in _far_interior_cross_candidates((h, w), side, trigger_lines)
            if _can_place_cross(grid, center)
            and all(abs(center[0] - prev[0]) + abs(center[1] - prev[1]) >= 5 for prev in centers)
        ]
        if extra_candidates:
            extra_center = extra_candidates[int(rng.integers(0, len(extra_candidates)))]
            centers.append(extra_center)
            forbidden_cross_cells |= _cross_cells(extra_center)
            _paint_cross(grid, extra_center)

    line_limit = w if side in ("top", "bottom") else h
    candidates = [
        idx
        for idx in range(2, line_limit - 2)
        if idx not in trigger_lines and all(abs(idx - tr) >= 2 for tr in trigger_lines)
    ]
    extra_count = int(rng.integers(0, min(3, len(candidates)) + 1))
    extra_lines: list[int] = []
    if extra_count > 0:
        picked = _separated_choices(candidates, extra_count, 2, rng)
        if picked is None:
            return None
        extra_lines = picked
    selected_lines = sorted(trigger_lines | set(extra_lines))

    blocked = set(forbidden_cross_cells)
    _place_blue_points(grid, side, selected_lines, blocked, rng)

    mid, moved_mask, detected_dest = _transfer_only(grid)
    if detected_dest != dest_side:
        return None
    if selected_lines != _selected_lines(grid, side):
        return None

    for center in centers:
        if not _trigger_specs(mid, moved_mask, dest_side, center) and center in {(r, c) for r, c, _, _ in trigger_meta}:
            return None

    out = main(grid)
    if np.array_equal(out, grid) or np.array_equal(out, mid):
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(500):
        grid = _generate_case(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
