from __future__ import annotations

from collections import Counter

import numpy as np


BG = 0
EDGE_COLORS = (1, 2, 3, 4)
SIDES = ("top", "bottom", "left", "right")

MIN_GRID = 12
MAX_GRID = 18
MIN_GAP = 2
ORTHOGONAL_MARGIN = 1
GEN_ATTEMPTS = 500


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                if rr > 0 and mask[rr - 1, cc] and not seen[rr - 1, cc]:
                    seen[rr - 1, cc] = True
                    stack.append((rr - 1, cc))
                if rr + 1 < h and mask[rr + 1, cc] and not seen[rr + 1, cc]:
                    seen[rr + 1, cc] = True
                    stack.append((rr + 1, cc))
                if cc > 0 and mask[rr, cc - 1] and not seen[rr, cc - 1]:
                    seen[rr, cc - 1] = True
                    stack.append((rr, cc - 1))
                if cc + 1 < w and mask[rr, cc + 1] and not seen[rr, cc + 1]:
                    seen[rr, cc + 1] = True
                    stack.append((rr, cc + 1))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _edge_color_to_side(grid: np.ndarray) -> dict[int, str]:
    g = np.asarray(grid, dtype=int)
    side_values = {
        "top": g[0, 1:-1],
        "bottom": g[-1, 1:-1],
        "left": g[1:-1, 0],
        "right": g[1:-1, -1],
    }
    mapping: dict[int, str] = {}
    for side, values in side_values.items():
        colors = {int(v) for v in values.tolist() if int(v) != BG}
        assert len(colors) == 1, "each border side must have one solid color"
        color = next(iter(colors))
        mapping[color] = side
    assert len(mapping) == 4, "the four border sides must use four distinct colors"
    return mapping


def _find_inner_squares(grid: np.ndarray) -> list[dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    inner = g[1:-1, 1:-1]
    squares: list[dict[str, int]] = []
    for comp in _components4(inner != BG):
        coords = comp + np.array([1, 1], dtype=int)
        rs = coords[:, 0]
        cs = coords[:, 1]
        r0, r1 = int(rs.min()), int(rs.max())
        c0, c1 = int(cs.min()), int(cs.max())
        color = int(g[r0, c0])
        k = r1 - r0 + 1
        if k != c1 - c0 + 1:
            continue
        if int(coords.shape[0]) != k * k:
            continue
        if not bool(np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color)):
            continue
        squares.append(
            {
                "color": color,
                "size": int(k),
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
            }
        )
    return squares


def _paint_dashed_band(out: np.ndarray, square: dict[str, int], side: str) -> None:
    color = int(square["color"])
    k = int(square["size"])
    r0 = int(square["r0"])
    c0 = int(square["c0"])
    r1 = int(square["r1"])
    c1 = int(square["c1"])

    if side == "top":
        for dist in range(1, r0):
            if ((dist - 1) // k) % 2 != 0:
                continue
            row = r0 - dist
            segment = out[row, c0 : c1 + 1]
            segment[segment == BG] = color
    elif side == "bottom":
        max_dist = out.shape[0] - 2 - r1
        for dist in range(1, max_dist + 1):
            if ((dist - 1) // k) % 2 != 0:
                continue
            row = r1 + dist
            segment = out[row, c0 : c1 + 1]
            segment[segment == BG] = color
    elif side == "left":
        for dist in range(1, c0):
            if ((dist - 1) // k) % 2 != 0:
                continue
            col = c0 - dist
            segment = out[r0 : r1 + 1, col]
            segment[segment == BG] = color
    elif side == "right":
        max_dist = out.shape[1] - 2 - c1
        for dist in range(1, max_dist + 1):
            if ((dist - 1) // k) % 2 != 0:
                continue
            col = c1 + dist
            segment = out[r0 : r1 + 1, col]
            segment[segment == BG] = color
    else:
        raise ValueError(f"unknown side: {side}")


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert g.shape[0] >= 3 and g.shape[1] >= 3

    color_to_side = _edge_color_to_side(g)
    squares = _find_inner_squares(g)

    out = g.copy()
    for square in squares:
        side = color_to_side[int(square["color"])]
        _paint_dashed_band(out, square, side)
    return out


def _placement_masks(h: int, w: int, side: str, r0: int, c0: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    square = np.zeros((h, w), dtype=bool)
    channel = np.zeros((h, w), dtype=bool)
    square[r0 : r0 + k, c0 : c0 + k] = True

    if side == "top":
        channel[1:r0, c0 : c0 + k] = True
    elif side == "bottom":
        channel[r0 + k : h - 1, c0 : c0 + k] = True
    elif side == "left":
        channel[r0 : r0 + k, 1:c0] = True
    elif side == "right":
        channel[r0 : r0 + k, c0 + k : w - 1] = True
    else:
        raise ValueError(f"unknown side: {side}")

    return square, channel


def _sample_sides(rng: np.random.Generator, count: int) -> list[str]:
    while True:
        choices = [SIDES[int(rng.integers(0, len(SIDES)))] for _ in range(count)]
        counts = Counter(choices)
        if len(counts) >= 2 and max(counts.values()) <= 2:
            return choices


def _enumerate_candidates(
    h: int,
    w: int,
    side: str,
    k: int,
    reserved: np.ndarray,
    square_mask: np.ndarray,
) -> list[tuple[int, int]]:
    row_min = 1 + ORTHOGONAL_MARGIN
    row_max = h - k - 1 - ORTHOGONAL_MARGIN
    col_min = 1 + ORTHOGONAL_MARGIN
    col_max = w - k - 1 - ORTHOGONAL_MARGIN

    if row_min > row_max or col_min > col_max:
        return []

    if side == "top":
        rows = range(1 + MIN_GAP, row_max + 1)
        cols = range(col_min, col_max + 1)
    elif side == "bottom":
        rows = range(row_min, h - k - 1 - MIN_GAP + 1)
        cols = range(col_min, col_max + 1)
    elif side == "left":
        rows = range(row_min, row_max + 1)
        cols = range(1 + MIN_GAP, col_max + 1)
    elif side == "right":
        rows = range(row_min, row_max + 1)
        cols = range(col_min, w - k - 1 - MIN_GAP + 1)
    else:
        raise ValueError(f"unknown side: {side}")

    candidates: list[tuple[int, int]] = []
    for r0 in rows:
        for c0 in cols:
            square, channel = _placement_masks(h, w, side, int(r0), int(c0), int(k))
            if bool(np.any(reserved & (square | channel))):
                continue

            r_lo = max(0, int(r0) - 1)
            r_hi = min(h, int(r0) + int(k) + 1)
            c_lo = max(0, int(c0) - 1)
            c_hi = min(w, int(c0) + int(k) + 1)
            if bool(square_mask[r_lo:r_hi, c_lo:c_hi].any()):
                continue

            candidates.append((int(r0), int(c0)))
    return candidates


def _check_instance(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    try:
        color_to_side = _edge_color_to_side(g)
    except AssertionError:
        return False

    squares = _find_inner_squares(g)
    if not (2 <= len(squares) <= 4):
        return False

    square_mask = np.zeros((h, w), dtype=bool)
    channel_mask = np.zeros((h, w), dtype=bool)
    for square in squares:
        color = int(square["color"])
        if color not in color_to_side:
            return False
        k = int(square["size"])
        r0 = int(square["r0"])
        c0 = int(square["c0"])
        sq_mask, ch_mask = _placement_masks(h, w, color_to_side[color], r0, c0, k)
        if bool(np.any(channel_mask & ch_mask)):
            return False
        if bool(np.any(g[ch_mask] != BG)):
            return False
        r_lo = max(0, r0 - 1)
        r_hi = min(h, r0 + k + 1)
        c_lo = max(0, c0 - 1)
        c_hi = min(w, c0 + k + 1)
        if bool(square_mask[r_lo:r_hi, c_lo:c_hi].any()):
            return False
        channel_mask |= ch_mask
        square_mask |= sq_mask

    try:
        out = main(g)
    except Exception:
        return False
    return bool(isinstance(out, np.ndarray) and out.shape == g.shape and np.any(out != g))


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(MIN_GRID, MAX_GRID + 1))
    w = int(rng.integers(MIN_GRID, MAX_GRID + 1))

    grid = np.full((h, w), BG, dtype=int)
    edge_colors = rng.permutation(np.asarray(EDGE_COLORS, dtype=int)).tolist()
    side_to_color = {side: int(color) for side, color in zip(SIDES, edge_colors)}

    grid[0, 1:-1] = side_to_color["top"]
    grid[-1, 1:-1] = side_to_color["bottom"]
    grid[1:-1, 0] = side_to_color["left"]
    grid[1:-1, -1] = side_to_color["right"]

    count = int(rng.integers(2, 5))
    sides = _sample_sides(rng, count)
    sizes = [int(rng.integers(1, 4)) for _ in range(count)]
    specs = [{"side": side, "size": size, "color": side_to_color[side]} for side, size in zip(sides, sizes)]

    order = sorted(range(count), key=lambda idx: (-specs[idx]["size"], float(rng.random())))
    reserved = np.zeros((h, w), dtype=bool)
    square_mask = np.zeros((h, w), dtype=bool)

    for idx in order:
        spec = specs[idx]
        candidates = _enumerate_candidates(h, w, spec["side"], spec["size"], reserved, square_mask)
        if not candidates:
            return None

        r0, c0 = candidates[int(rng.integers(0, len(candidates)))]
        k = int(spec["size"])
        color = int(spec["color"])
        sq_mask, ch_mask = _placement_masks(h, w, spec["side"], r0, c0, k)
        grid[r0 : r0 + k, c0 : c0 + k] = color
        reserved |= sq_mask | ch_mask
        square_mask |= sq_mask

    if not _check_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
