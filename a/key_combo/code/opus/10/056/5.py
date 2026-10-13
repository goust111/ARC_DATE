import math
from collections import deque

import numpy as np


BLACK = 0
RED = 2
YELLOW = 4
PINK = 6
ORANGE = 7

MAX_GENERATION_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _rect_sum(prefix: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    total = int(prefix[r1, c1])
    if r0 > 0:
        total -= int(prefix[r0 - 1, c1])
    if c0 > 0:
        total -= int(prefix[r1, c0 - 1])
    if r0 > 0 and c0 > 0:
        total += int(prefix[r0 - 1, c0 - 1])
    return total


def _expand_rect(rect: tuple[int, int, int, int], pad: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = rect
    return (
        max(0, r0 - pad),
        max(0, c0 - pad),
        min(shape[0] - 1, r1 + pad),
        min(shape[1] - 1, c1 + pad),
    )


def _mask_from_rects(shape: tuple[int, int], rects: list[tuple[int, int, int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r0, c0, r1, c1 in rects:
        mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _clean_background_mask(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    clean = np.zeros((h, w), dtype=bool)
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(grid[r, c]) != BLACK:
                continue
            hood = grid[r - 1 : r + 2, c - 1 : c + 2]
            if np.all((hood == BLACK) | (hood == PINK)):
                clean[r, c] = True
    return clean


def _maximal_true_rectangles(mask: np.ndarray) -> list[tuple[int, int, int, int, int, int, int]]:
    h, w = mask.shape
    prefix = mask.astype(int).cumsum(axis=0).cumsum(axis=1)
    rects: list[tuple[int, int, int, int, int, int, int]] = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0]:
                continue
            for r1 in range(r0, h):
                for c1 in range(c0, w):
                    area = (r1 - r0 + 1) * (c1 - c0 + 1)
                    if _rect_sum(prefix, r0, c0, r1, c1) != area:
                        continue
                    if r0 > 0 and _rect_sum(prefix, r0 - 1, c0, r1, c1) == area + (c1 - c0 + 1):
                        continue
                    if r1 + 1 < h and _rect_sum(prefix, r0, c0, r1 + 1, c1) == area + (c1 - c0 + 1):
                        continue
                    if c0 > 0 and _rect_sum(prefix, r0, c0 - 1, r1, c1) == area + (r1 - r0 + 1):
                        continue
                    if c1 + 1 < w and _rect_sum(prefix, r0, c0, r1, c1 + 1) == area + (r1 - r0 + 1):
                        continue
                    hh = r1 - r0 + 1
                    ww = c1 - c0 + 1
                    rects.append((area, max(hh, ww), hh, ww, r0, c0, r1, c1))
    rects.sort(key=lambda item: (-item[0], -item[1], item[4], item[5], item[6], item[7]))
    return rects


def _channel_mask_from_grid(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    clean = _clean_background_mask(grid)
    channel = np.zeros((h, w), dtype=bool)
    min_h = (h + 1) // 2
    min_w = (w + 1) // 2

    for _area, _major, hh, ww, r0, c0, r1, c1 in _maximal_true_rectangles(clean):
        if hh > ww:
            if hh < min_h:
                continue
        elif ww > hh:
            if ww < min_w:
                continue
        else:
            if hh < min_h and ww < min_w:
                continue
        rect_mask = np.zeros_like(channel)
        rect_mask[r0 : r1 + 1, c0 : c1 + 1] = True
        new_pixels = rect_mask & clean & ~channel
        if np.any(new_pixels):
            channel |= rect_mask & clean
    return channel


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    pink = np.argwhere(g == PINK)
    assert pink.shape[0] == 1, "expected exactly one control pixel"
    control = (int(pink[0, 0]), int(pink[0, 1]))

    channel = _channel_mask_from_grid(g)
    out = g.copy()
    for r, c in np.argwhere(channel):
        d = _chebyshev((int(r), int(c)), control)
        out[r, c] = RED if d % 2 == 0 else YELLOW
    out[control] = PINK
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect the long clean black corridors hidden in orange clutter, then color each corridor pixel by Chebyshev-distance parity from the magenta control pixel.

    Input structure:
    - Black background contains dense orange clutter plus one magenta control pixel.
    - Hidden among the clutter are one or more long horizontal or vertical black corridors of width 1 or 2.
    - True corridor pixels have a full in-grid 3x3 neighborhood containing only black, except that the control pixel may replace one of those black cells.

    Rule:
    - Build a clean-background mask from black pixels whose full 3x3 neighborhood is black or magenta; border pixels fail this test.
    - Enumerate all maximal axis-aligned rectangles inside that clean mask and accept them largest-first when their major axis spans at least half of the corresponding grid dimension and they add at least one new pixel.
    - Treat the union of accepted clean pixels as the corridor region, preserve clutter and ordinary background, keep the control pixel magenta, and recolor corridor pixels red for even Chebyshev distance to the control or yellow for odd distance.

    Key insight:
    - One geometric system decides where painting is allowed by recovering long clean negative-space rectangles, while a second independent system decides what color each allowed pixel gets from distance parity around the control point.
    """
    return _solve(grid)


def _place_bar(
    shape: tuple[int, int],
    orientation: str,
    width: int,
    length: int,
    blocked: np.ndarray,
    rng: np.random.Generator,
) -> tuple[int, int, int, int] | None:
    h, w = shape
    tries = 120
    for _ in range(tries):
        if orientation == "v":
            r0 = int(rng.integers(1, h - length))
            c0 = int(rng.integers(1, w - width))
            rect = (r0, c0, r0 + length - 1, c0 + width - 1)
        else:
            r0 = int(rng.integers(1, h - width))
            c0 = int(rng.integers(1, w - length))
            rect = (r0, c0, r0 + width - 1, c0 + length - 1)
        guard = _expand_rect(rect, 2, shape)
        gr0, gc0, gr1, gc1 = guard
        if np.any(blocked[gr0 : gr1 + 1, gc0 : gc1 + 1]):
            continue
        return rect
    return None


def _pick_control_on_channel(
    channels: list[tuple[int, int, int, int]],
    rng: np.random.Generator,
) -> tuple[int, int] | None:
    eligible = [rect for rect in channels if (rect[3] - rect[1] + 1 == 2) or (rect[2] - rect[0] + 1 == 2)]
    rng.shuffle(eligible)
    for r0, c0, r1, c1 in eligible:
        rows = list(range(r0 + 1, r1)) if r1 - r0 + 1 >= 3 else list(range(r0, r1 + 1))
        cols = list(range(c0 + 1, c1)) if c1 - c0 + 1 >= 3 else list(range(c0, c1 + 1))
        if not rows or not cols:
            continue
        return int(rng.choice(rows)), int(rng.choice(cols))
    return None


def _pick_free_center(
    shape: tuple[int, int],
    blocked: np.ndarray,
    rng: np.random.Generator,
) -> tuple[int, int] | None:
    h, w = shape
    for _ in range(160):
        r = int(rng.integers(2, h - 2))
        c = int(rng.integers(2, w - 2))
        if np.any(blocked[r - 1 : r + 2, c - 1 : c + 2]):
            continue
        return r, c
    return None


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _build_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(18, 27))
    w = int(rng.integers(18, 27))
    shape = (h, w)
    min_h = (h + 1) // 2
    min_w = (w + 1) // 2

    layout = str(rng.choice(np.array(["mixed", "v", "h"], dtype=object), p=[0.6, 0.2, 0.2]))
    if layout == "mixed":
        orientations = ["v"] * int(rng.integers(1, 3)) + ["h"] * int(rng.integers(1, 3))
    elif layout == "v":
        orientations = ["v"] * int(rng.integers(2, 5))
    else:
        orientations = ["h"] * int(rng.integers(2, 5))
    rng.shuffle(orientations)

    blocked = np.zeros(shape, dtype=bool)
    channels: list[tuple[int, int, int, int]] = []
    channel_supports: list[tuple[int, int, int, int]] = []
    for orientation in orientations:
        width = int(rng.choice([1, 2], p=[0.4, 0.6]))
        if orientation == "v":
            length = int(rng.integers(min_h, h - 1))
        else:
            length = int(rng.integers(min_w, w - 1))
        rect = _place_bar(shape, orientation, width, length, blocked, rng)
        if rect is None:
            return None
        channels.append(rect)
        support = _expand_rect(rect, 1, shape)
        channel_supports.append(support)
        guard = _expand_rect(rect, 2, shape)
        gr0, gc0, gr1, gc1 = guard
        blocked[gr0 : gr1 + 1, gc0 : gc1 + 1] = True

    decoys: list[tuple[int, int, int, int]] = []
    for _ in range(int(rng.integers(1, 4))):
        orientation = str(rng.choice(np.array(["v", "h"], dtype=object)))
        width = int(rng.choice([1, 2]))
        max_len = max(4, min((min_h - 1) if orientation == "v" else (min_w - 1), 7))
        min_len = 3 if max_len >= 3 else max_len
        if min_len > max_len:
            continue
        length = int(rng.integers(min_len, max_len + 1))
        rect = _place_bar(shape, orientation, width, length, blocked, rng)
        if rect is None:
            continue
        decoys.append(rect)
        support = _expand_rect(rect, 1, shape)
        gr0, gc0, gr1, gc1 = support
        blocked[gr0 : gr1 + 1, gc0 : gc1 + 1] = True

    control_on_channel = bool(rng.integers(0, 2))
    control = _pick_control_on_channel(channels, rng) if control_on_channel else None
    if control is None:
        control_on_channel = False
        control = _pick_free_center(shape, blocked, rng)
    if control is None:
        return None
    cr, cc = control

    reserved = np.zeros(shape, dtype=bool)
    for support in channel_supports:
        r0, c0, r1, c1 = support
        reserved[r0 : r1 + 1, c0 : c1 + 1] = True
    for rect in decoys:
        r0, c0, r1, c1 = _expand_rect(rect, 1, shape)
        reserved[r0 : r1 + 1, c0 : c1 + 1] = True
    reserved[cr - 1 : cr + 2, cc - 1 : cc + 2] = True

    grid = np.zeros(shape, dtype=int)
    clutter_prob = float(rng.uniform(0.31, 0.41))
    clutter_mask = (rng.random(shape) < clutter_prob) & ~reserved
    grid[clutter_mask] = ORANGE
    grid[cr, cc] = PINK

    for support in channel_supports:
        r0, c0, r1, c1 = support
        if np.any(grid[r0 : r1 + 1, c0 : c1 + 1] == ORANGE):
            return None
    for rect in decoys:
        sr0, sc0, sr1, sc1 = _expand_rect(rect, 1, shape)
        if np.any(grid[sr0 : sr1 + 1, sc0 : sc1 + 1] == ORANGE):
            return None
    if np.any(grid[cr - 1 : cr + 2, cc - 1 : cc + 2] == ORANGE):
        return None

    planned_channel = _mask_from_rects(shape, channels)
    planned_decoy = _mask_from_rects(shape, decoys)
    out = _solve(grid)
    painted = (out == RED) | (out == YELLOW)

    if not np.all(painted <= planned_channel):
        return None
    if control_on_channel:
        if not planned_channel[cr, cc]:
            return None
        if painted.sum() < max(6, int(planned_channel.sum() * 0.7)):
            return None
    else:
        if planned_channel[cr, cc]:
            return None
        if not np.array_equal(painted, planned_channel):
            return None

    if not np.any(planned_decoy & (grid == BLACK) & ~painted):
        return None
    if not np.any(out == RED) or not np.any(out == YELLOW):
        return None

    clutter_fraction = float(np.mean(grid == ORANGE))
    if clutter_fraction < 0.22 or clutter_fraction > 0.42:
        return None

    channel_components = _components(planned_channel)
    if len(channel_components) != len(channels):
        return None
    if any(comp.shape[0] < 3 for comp in channel_components):
        return None
    if np.any(np.logical_and(planned_channel, planned_decoy)):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid corridor-parity ARC input")
