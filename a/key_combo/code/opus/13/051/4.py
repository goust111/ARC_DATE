import random
import numpy as np


BG = 4
BORDER = 1
INTERIOR = 2
MARKER = 3
DRAW_ORDER = ((0, 0), (0, 1), (1, 0), (1, 1))
GEN_ATTEMPTS = 600


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise AssertionError("empty mask")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _draw_rect(grid: np.ndarray, box: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1] = INTERIOR
    grid[r0, c0 : c1 + 1] = BORDER
    grid[r1, c0 : c1 + 1] = BORDER
    grid[r0 : r1 + 1, c0] = BORDER
    grid[r0 : r1 + 1, c1] = BORDER


def _parse_input(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(map(int, np.unique(g))).issubset({BG, BORDER, INTERIOR, MARKER})

    marker_coords = np.argwhere(g == MARKER)
    assert marker_coords.shape == (1, 2)
    marker = tuple(map(int, marker_coords[0]))

    rect_mask = (g == BORDER) | (g == INTERIOR)
    r0, c0, r1, c1 = _bbox(rect_mask)
    assert r1 - r0 + 1 >= 4 and c1 - c0 + 1 >= 4

    block = g[r0 : r1 + 1, c0 : c1 + 1]
    edge = np.zeros(block.shape, dtype=bool)
    edge[0, :] = True
    edge[-1, :] = True
    edge[:, 0] = True
    edge[:, -1] = True
    assert np.all(block[edge] == BORDER)
    assert np.all(block[~edge] == INTERIOR)
    assert np.all(g[rect_mask] != MARKER)

    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = BG
    outside[marker] = BG
    assert np.all(outside == BG)

    mr, mc = marker
    candidates = []
    if mr < r0 and c0 <= mc <= c1:
        candidates.append("up")
    if mr > r1 and c0 <= mc <= c1:
        candidates.append("down")
    if mc < c0 and r0 <= mr <= r1:
        candidates.append("left")
    if mc > c1 and r0 <= mr <= r1:
        candidates.append("right")
    assert len(candidates) == 1

    return {
        "shape": tuple(map(int, g.shape)),
        "box": (r0, c0, r1, c1),
        "marker": marker,
        "direction": candidates[0],
    }


def _copy_specs(meta: dict[str, object]) -> list[dict[str, object]]:
    h, w = meta["shape"]
    r0, c0, r1, c1 = meta["box"]
    mr, mc = meta["marker"]
    direction = meta["direction"]
    specs: list[dict[str, object]] = []

    for order, (qr, qc) in enumerate(DRAW_ORDER):
        ro = qr * h
        co = qc * w
        lr0, lc0 = r0 + ro, c0 + co
        lr1, lc1 = r1 + ro, c1 + co
        lmr, lmc = mr + ro, mc + co

        if direction in ("left", "right"):
            if qc == 0:
                expanded = (lr0, lc0, lr1, lmc + w)
            else:
                expanded = (lr0, lmc - w, lr1, lc1)
        else:
            assert direction in ("up", "down")
            if qr == 0:
                expanded = (lr0, lc0, lmr + h, lc1)
            else:
                expanded = (lmr - h, lc0, lr1, lc1)

        er0, ec0, er1, ec1 = expanded
        assert 0 <= er0 <= er1 < 2 * h
        assert 0 <= ec0 <= ec1 < 2 * w
        specs.append(
            {
                "order": order,
                "expanded": expanded,
                "local_marker": (lmr, lmc),
            }
        )
    return specs


def _distance_compose(
    meta: dict[str, object], *, return_details: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = meta["shape"]
    out = np.full((2 * h, 2 * w), BG, dtype=int)
    best_dist = np.full((2 * h, 2 * w), 10**9, dtype=int)
    winner = np.full((2 * h, 2 * w), -1, dtype=int)
    cover_count = np.zeros((2 * h, 2 * w), dtype=int)

    for spec in _copy_specs(meta):
        r0, c0, r1, c1 = spec["expanded"]
        mr, mc = spec["local_marker"]
        order = int(spec["order"])

        rows = np.arange(r0, r1 + 1)[:, None]
        cols = np.arange(c0, c1 + 1)[None, :]
        dist = np.abs(rows - mr) + np.abs(cols - mc)
        colors = np.full(dist.shape, INTERIOR, dtype=int)
        colors[(rows == r0) | (rows == r1) | (cols == c0) | (cols == c1)] = BORDER

        sl = np.s_[r0 : r1 + 1, c0 : c1 + 1]
        cover_count[sl] += 1
        best_sub = best_dist[sl]
        winner_sub = winner[sl]
        out_sub = out[sl]
        better = dist < best_sub
        best_sub[better] = dist[better]
        winner_sub[better] = order
        out_sub[better] = colors[better]

    if return_details:
        return out, winner, cover_count
    return out


def _paint_simple(meta: dict[str, object], keep: str) -> np.ndarray:
    h, w = meta["shape"]
    out = np.full((2 * h, 2 * w), BG, dtype=int)
    covered = np.zeros((2 * h, 2 * w), dtype=bool)
    for spec in _copy_specs(meta):
        r0, c0, r1, c1 = spec["expanded"]
        rows = np.arange(r0, r1 + 1)[:, None]
        cols = np.arange(c0, c1 + 1)[None, :]
        colors = np.full((r1 - r0 + 1, c1 - c0 + 1), INTERIOR, dtype=int)
        colors[(rows == r0) | (rows == r1) | (cols == c0) | (cols == c1)] = BORDER
        sl = np.s_[r0 : r1 + 1, c0 : c1 + 1]
        if keep == "first":
            covered_sub = covered[sl]
            out_sub = out[sl]
            paint = ~covered_sub
            out_sub[paint] = colors[paint]
            covered_sub[...] = True
        else:
            assert keep == "last"
            out[sl] = colors
    return out


def _has_overlap_block(cover_count: np.ndarray) -> bool:
    overlap = cover_count >= 2
    if overlap.shape[0] < 2 or overlap.shape[1] < 2:
        return False
    return bool(
        np.any(
            overlap[:-1, :-1]
            & overlap[:-1, 1:]
            & overlap[1:, :-1]
            & overlap[1:, 1:]
        )
    )


def _semantic_checks(inp: np.ndarray) -> bool:
    try:
        meta = _parse_input(inp)
        out, winner, cover_count = _distance_compose(meta, return_details=True)
        if not _has_overlap_block(cover_count):
            return False
        overlap = cover_count >= 2
        active_winners = set(map(int, np.unique(winner[overlap])))
        active_winners.discard(-1)
        if len(active_winners) < 2:
            return False
        if np.array_equal(out, _paint_simple(meta, "first")):
            return False
        if np.array_equal(out, _paint_simple(meta, "last")):
            return False
        return True
    except Exception:
        return False


def main(grid: np.ndarray) -> np.ndarray:
    """Tile the marked rectangle to a 2x2 canvas, stretch the four copies, and arbitrate overlaps by marker distance.

    Input structure:
    - The input is a small yellow-background grid containing exactly one blue-bordered, red-filled rectangle.
    - A single green marker pixel lies outside that rectangle, aligned with one of its row spans or column spans.
    - The marker direction determines whether the rectangle is a left, right, upward, or downward stretcher.

    Rule:
    - Duplicate the whole input into the four quadrants of a 2x2 output canvas.
    - For each quadrant copy, stretch the edge facing the central seam until it reaches the mirrored marker line from the neighboring tiled copy; the copied local marker itself is only used as that copy's distance anchor.
    - Draw the four stretched rectangles in left-top, right-top, left-bottom, right-bottom order, but whenever multiple stretched copies cover the same cell, keep the color from the copy whose local marker has the smallest Manhattan distance to that cell; exact ties keep the earlier copy.
    - Each winning rectangle paints a blue border and red interior, and every uncovered cell stays yellow.

    Key insight:
    - Tiling creates several aligned marker anchors, so the stretch is no longer a single isolated rectangle problem: the visible boundary in the overlap zone is determined by Manhattan-distance ownership between competing copies.
    """
    meta = _parse_input(grid)
    return _distance_compose(meta)


def _axis_length_choices(axis_limit: int, gap: int) -> list[int]:
    return list(range(4, axis_limit - gap + 1))


def _pick_dims(direction: str, h: int, w: int) -> tuple[int, int, int] | None:
    gap = random.randint(2, 4)
    if direction in ("left", "right"):
        widths = _axis_length_choices(w, gap)
        heights = [size for size in range(4, min(6, h) + 1)]
        if not widths or not heights:
            return None
        rect_w = random.choice(widths)
        rect_h = random.choice(heights)
        return rect_h, rect_w, gap

    heights = _axis_length_choices(h, gap)
    widths = [size for size in range(4, min(6, w) + 1)]
    if not heights or not widths:
        return None
    rect_h = random.choice(heights)
    rect_w = random.choice(widths)
    return rect_h, rect_w, gap


def _valid_positions(direction: str, h: int, w: int, rect_h: int, rect_w: int, gap: int) -> list[tuple[int, int]]:
    spots = []
    r_lo, r_hi = 0, h - rect_h
    c_lo, c_hi = 0, w - rect_w

    if direction == "right":
        c_hi = w - rect_w - gap
    elif direction == "left":
        c_lo = gap
    elif direction == "down":
        r_hi = h - rect_h - gap
    else:
        assert direction == "up"
        r_lo = gap

    if r_lo > r_hi or c_lo > c_hi:
        return []

    grid_center = ((h - 1) / 2.0, (w - 1) / 2.0)
    for r0 in range(r_lo, r_hi + 1):
        for c0 in range(c_lo, c_hi + 1):
            r1 = r0 + rect_h - 1
            c1 = c0 + rect_w - 1
            center = ((r0 + r1) / 2.0, (c0 + c1) / 2.0)
            if abs(center[0] - grid_center[0]) + abs(center[1] - grid_center[1]) < 1.0:
                continue
            spots.append((r0, c0))
    return spots


def _marker_for(direction: str, box: tuple[int, int, int, int], gap: int) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    if direction == "right":
        return r0 + 1, c1 + gap
    if direction == "left":
        return r1 - 1, c0 - gap
    if direction == "down":
        return r1 + gap, c0 + 1
    assert direction == "up"
    return r0 - gap, c1 - 1


def _gen_attempt() -> np.ndarray | None:
    h = random.choice([8, 9, 10])
    w = random.choice([8, 9, 10])
    direction = random.choice(["up", "down", "left", "right"])
    dims = _pick_dims(direction, h, w)
    if dims is None:
        return None
    rect_h, rect_w, gap = dims

    positions = _valid_positions(direction, h, w, rect_h, rect_w, gap)
    if not positions:
        return None

    r0, c0 = random.choice(positions)
    box = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
    marker = _marker_for(direction, box, gap)

    grid = np.full((h, w), BG, dtype=int)
    _draw_rect(grid, box)
    grid[marker] = MARKER

    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid tiled-stretch ARC input")
