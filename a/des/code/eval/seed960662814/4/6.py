from __future__ import annotations

import random

import numpy as np


SIDES = ("top", "bottom", "left", "right")
PATTERNS = ((1,), (1, 1), (1, 0, 1), (1, 1, 1))

RECT_COUNT_CHOICES = (1, 2, 3, 4)
GRID_MIN_BY_COUNT = {1: 18, 2: 22, 3: 24, 4: 26}
GRID_MAX_SIDE = 32
RECT_SIZE_BY_COUNT = {1: (7, 12), 2: (7, 11), 3: (7, 10), 4: (7, 9)}
PLACEMENT_TRIES = 120
GEN_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _rotate_inward(pattern: np.ndarray, side: str) -> np.ndarray:
    if side == "top":
        return np.rot90(pattern, 3)
    if side == "bottom":
        return np.rot90(pattern, 1)
    if side == "left":
        return np.rot90(pattern, 1)
    if side == "right":
        return np.rot90(pattern, 3)
    raise ValueError(f"unknown side: {side}")


def _decode_marker(patch: np.ndarray) -> tuple[int, int, str, np.ndarray] | None:
    colors, counts = np.unique(patch, return_counts=True)
    if colors.size < 2:
        return None

    body = int(colors[np.argmax(counts)])
    marker_colors = [int(color) for color in colors if int(color) != body]
    if not marker_colors:
        return None
    marker = max(marker_colors, key=lambda color: int(np.count_nonzero(patch == color)))
    marker_mask = patch == marker
    rows, cols = np.where(marker_mask)
    if rows.size == 0:
        return None

    h, w = patch.shape
    candidates = []
    if np.all(rows == 0):
        candidates.append("top")
    if np.all(rows == h - 1):
        candidates.append("bottom")
    if np.all(cols == 0):
        candidates.append("left")
    if np.all(cols == w - 1):
        candidates.append("right")
    if len(candidates) != 1:
        return None

    side = candidates[0]
    if side in ("top", "bottom"):
        c0, c1 = int(cols.min()), int(cols.max())
        edge_row = 0 if side == "top" else h - 1
        pattern = marker_mask[edge_row : edge_row + 1, c0 : c1 + 1]
    else:
        r0, r1 = int(rows.min()), int(rows.max())
        edge_col = 0 if side == "left" else w - 1
        pattern = marker_mask[r0 : r1 + 1, edge_col : edge_col + 1]
    return body, marker, side, pattern.astype(bool)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode each rectangle's border marker and stamp the rotated marker pattern at the rectangle center.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    comps = _components(g != bg)
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        if not np.all(patch != bg):
            continue

        decoded = _decode_marker(patch)
        if decoded is None:
            continue
        body, marker, side, pattern = decoded
        stamp = _rotate_inward(pattern, side)

        rect = out[r0 : r1 + 1, c0 : c1 + 1]
        rect[patch == marker] = body

        inner_h, inner_w = rect.shape[0] - 2, rect.shape[1] - 2
        stamp_h, stamp_w = stamp.shape
        if stamp_h > inner_h or stamp_w > inner_w:
            continue
        sr = 1 + (inner_h - stamp_h) // 2
        sc = 1 + (inner_w - stamp_w) // 2
        target = rect[sr : sr + stamp_h, sc : sc + stamp_w]
        target[stamp] = marker

    return out


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(occupied.shape[0], r0 + h + 1)
    cc1 = min(occupied.shape[1], c0 + w + 1)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _pattern_array(side: str, bits: tuple[int, ...]) -> np.ndarray:
    if side in ("top", "bottom"):
        return np.array([bits], dtype=bool)
    return np.array(bits, dtype=bool).reshape(-1, 1)


def _paint_marker(
    grid: np.ndarray,
    side: str,
    bits: tuple[int, ...],
    marker: int,
    r0: int,
    c0: int,
    h: int,
    w: int,
    rng: random.Random,
) -> None:
    span = len(bits)
    if side in ("top", "bottom"):
        start = rng.randint(c0 + 1, c0 + w - span - 1)
        row = r0 if side == "top" else r0 + h - 1
        for offset, bit in enumerate(bits):
            if bit:
                grid[row, start + offset] = marker
    else:
        start = rng.randint(r0 + 1, r0 + h - span - 1)
        col = c0 if side == "left" else c0 + w - 1
        for offset, bit in enumerate(bits):
            if bit:
                grid[start + offset, col] = marker


def _expected_output(
    grid: np.ndarray,
    rects: list[dict[str, object]],
) -> np.ndarray:
    expected = grid.copy()
    for rect in rects:
        r0 = int(rect["r0"])
        c0 = int(rect["c0"])
        h = int(rect["h"])
        w = int(rect["w"])
        body = int(rect["body"])
        marker = int(rect["marker"])
        side = str(rect["side"])
        bits = tuple(int(v) for v in rect["bits"])

        patch = expected[r0 : r0 + h, c0 : c0 + w]
        patch[:, :] = body
        stamp = _rotate_inward(_pattern_array(side, bits), side)
        sh, sw = stamp.shape
        sr = 1 + (h - 2 - sh) // 2
        sc = 1 + (w - 2 - sw) // 2
        target = patch[sr : sr + sh, sc : sc + sw]
        target[stamp] = marker
    return expected


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_rects = rng.choice(RECT_COUNT_CHOICES)
    lo_size, hi_size = RECT_SIZE_BY_COUNT[n_rects]
    min_side = GRID_MIN_BY_COUNT[n_rects]

    h = rng.randint(min_side, GRID_MAX_SIDE)
    w = rng.randint(min_side, GRID_MAX_SIDE)
    bg = rng.randrange(10)
    palette = [color for color in range(10) if color != bg]
    rng.shuffle(palette)
    body_colors = palette[:n_rects]
    marker_colors = palette[n_rects : 2 * n_rects]

    side_order = rng.sample(list(SIDES), n_rects)
    rect_specs = []
    for i in range(n_rects):
        rect_specs.append(
            {
                "h": rng.randint(lo_size, hi_size),
                "w": rng.randint(lo_size, hi_size),
                "body": body_colors[i],
                "marker": marker_colors[i],
                "side": side_order[i],
                "bits": rng.choice(PATTERNS),
            }
        )
    rect_specs.sort(key=lambda spec: int(spec["h"]) * int(spec["w"]), reverse=True)

    grid = np.full((h, w), bg, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    for spec in rect_specs:
        rh = int(spec["h"])
        rw = int(spec["w"])
        if h - rh - 2 < 0 or w - rw - 2 < 0:
            return None

        placed = False
        for _ in range(PLACEMENT_TRIES):
            r0 = rng.randint(1, h - rh - 1)
            c0 = rng.randint(1, w - rw - 1)
            if not _can_place(occupied, r0, c0, rh, rw):
                continue
            grid[r0 : r0 + rh, c0 : c0 + rw] = int(spec["body"])
            occupied[r0 : r0 + rh, c0 : c0 + rw] = True
            _paint_marker(
                grid,
                str(spec["side"]),
                tuple(int(v) for v in spec["bits"]),
                int(spec["marker"]),
                r0,
                c0,
                rh,
                rw,
                rng,
            )
            spec["r0"] = r0
            spec["c0"] = c0
            placed = True
            break
        if not placed:
            return None

    if get_bg(grid) != bg:
        return None

    expected = _expected_output(grid, rect_specs)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
