import math
from collections import deque

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 300
PAIR_COUNT_RANGE = (2, 5)  # upper bound exclusive
RECT_HEIGHT_RANGE = (2, 7)  # 2..6
RECT_WIDTH_RANGE = (2, 8)   # 2..7
MIN_RECT_AREA = 6
MIN_FRAGMENT_AREA = 3
MAX_OUTPUT_WIDTH = 28
PLACEMENT_PADDING = 1

_RNG = np.random.default_rng()
_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in _N4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _normalize(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=int)
    mins = coords.min(axis=0)
    return coords - mins


def _bbox_shape(coords: np.ndarray) -> tuple[int, int]:
    norm = _normalize(coords)
    return int(norm[:, 0].max()) + 1, int(norm[:, 1].max()) + 1


def _pair_anchor(coords_a: np.ndarray, coords_b: np.ndarray) -> tuple[int, int]:
    r = min(int(coords_a[:, 0].min()), int(coords_b[:, 0].min()))
    c = min(int(coords_a[:, 1].min()), int(coords_b[:, 1].min()))
    return c, r


def _unique_rectangle_from_pair(coords_a: np.ndarray, coords_b: np.ndarray) -> tuple[int, int]:
    a = _normalize(coords_a)
    b = _normalize(coords_b)

    aset = {tuple(cell) for cell in a.tolist()}
    bset = {tuple(cell) for cell in b.tolist()}
    ah, aw = _bbox_shape(a)
    bh, bw = _bbox_shape(b)
    total = len(aset) + len(bset)

    solutions: list[tuple[int, int]] = []
    for dr in range(-bh, ah + 1):
        for dc in range(-bw, aw + 1):
            moved = {(r + dr, c + dc) for r, c in bset}
            if aset & moved:
                continue
            union = aset | moved
            rows = [r for r, _ in union]
            cols = [c for _, c in union]
            rect_h = max(rows) - min(rows) + 1
            rect_w = max(cols) - min(cols) + 1
            if rect_h * rect_w == total:
                solutions.append((rect_h, rect_w))

    assert len(solutions) == 1, "expected exactly one assembly translation"
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-colored fragments, reassemble each pair into a solid rectangle, then pack by area.

    Input structure:
    - The background is black.
    - Each non-black color appears in exactly two separate 4-connected fragments.
    - The two fragments of one color are complementary pieces of one solid rectangle after translation only.

    Rule:
    - For every color, find the unique translation that lets its two fragments combine without overlap into a filled rectangle.
    - Measure that rectangle's area and keep its original color.
    - Sort the reconstructed rectangles by increasing area and place them left-to-right with one black column between neighbors.
    - Bottom-align every rectangle in the output; the output height is the tallest rectangle.

    Key insight:
    - The task first requires mentally reassembling each same-colored fragment pair into its hidden rectangle, then ordering those rectangles by the area of the completed shape rather than by their scattered input positions.
    """
    g = np.asarray(grid, dtype=int)
    colors = [int(color) for color in np.unique(g) if int(color) != BG]
    assert colors, "expected at least one colored pair"

    rects: list[tuple[int, int, int, int, int, int]] = []
    for color in colors:
        comps = _find_components(g == color)
        assert len(comps) == 2, "each color must appear in exactly two fragments"
        rect_h, rect_w = _unique_rectangle_from_pair(comps[0], comps[1])
        anchor_c, anchor_r = _pair_anchor(comps[0], comps[1])
        rects.append((rect_h * rect_w, anchor_c, anchor_r, rect_h, rect_w, color))

    rects.sort(key=lambda item: (item[0], item[1], item[2]))
    out_h = max(item[3] for item in rects)
    out_w = sum(item[4] for item in rects) + len(rects) - 1
    out = np.full((out_h, out_w), BG, dtype=int)

    col = 0
    for _area, _anchor_c, _anchor_r, rect_h, rect_w, color in rects:
        out[out_h - rect_h : out_h, col : col + rect_w] = int(color)
        col += rect_w + 1
    return out


def _random_profile(length: int, low: int, high: int) -> list[int]:
    if low == high:
        return [low] * length

    vals = [int(_RNG.integers(low, high + 1))]
    for _ in range(length - 1):
        if _RNG.random() < 0.7:
            proposal = vals[-1] + int(_RNG.integers(-1, 2))
        else:
            proposal = int(_RNG.integers(low, high + 1))
        vals.append(int(np.clip(proposal, low, high)))

    if len(set(vals)) == 1 and length > 1:
        idx = int(_RNG.integers(length))
        choices = [v for v in range(low, high + 1) if v != vals[idx]]
        vals[idx] = int(choices[int(_RNG.integers(len(choices)))])
    return vals


def _make_partition(rect_h: int, rect_w: int) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(100):
        vertical = bool(_RNG.integers(2))
        if vertical:
            cuts = _random_profile(rect_h, 1, rect_w - 1)
            area_a = sum(cuts)
            area_b = rect_h * rect_w - area_a
            if min(area_a, area_b) < MIN_FRAGMENT_AREA:
                continue
            a = [(r, c) for r in range(rect_h) for c in range(cuts[r])]
            b = [(r, c) for r in range(rect_h) for c in range(cuts[r], rect_w)]
        else:
            cuts = _random_profile(rect_w, 1, rect_h - 1)
            area_a = sum(cuts)
            area_b = rect_h * rect_w - area_a
            if min(area_a, area_b) < MIN_FRAGMENT_AREA:
                continue
            a = [(r, c) for c in range(rect_w) for r in range(cuts[c])]
            b = [(r, c) for c in range(rect_w) for r in range(cuts[c], rect_h)]

        coords_a = np.asarray(a, dtype=int)
        coords_b = np.asarray(b, dtype=int)
        try:
            dims = _unique_rectangle_from_pair(coords_a, coords_b)
        except AssertionError:
            continue
        if dims == (rect_h, rect_w):
            return coords_a, coords_b
    raise RuntimeError("failed to create a valid fragment pair")


def _choose_rectangles() -> list[tuple[int, int]]:
    candidates = [
        (h, w)
        for h in range(*RECT_HEIGHT_RANGE)
        for w in range(*RECT_WIDTH_RANGE)
        if h * w >= MIN_RECT_AREA and (h, w) not in {(2, 3), (3, 2)}
    ]

    for _ in range(200):
        count = int(_RNG.integers(*PAIR_COUNT_RANGE))
        picks = _RNG.choice(len(candidates), size=count, replace=False)
        rects = [candidates[int(i)] for i in picks]
        areas = [h * w for h, w in rects]
        if len(set(areas)) != len(areas):
            continue
        if len(set(rects)) != len(rects):
            continue
        if sum(w for _, w in rects) + count - 1 > MAX_OUTPUT_WIDTH:
            continue
        return rects
    raise RuntimeError("failed to choose rectangle dimensions")


def _scatter_pieces(pieces: list[dict[str, np.ndarray | int]]) -> np.ndarray | None:
    max_h = max(int(piece["bbox_h"]) for piece in pieces)
    max_w = max(int(piece["bbox_w"]) for piece in pieces)
    padded_area = sum((int(piece["bbox_h"]) + 2) * (int(piece["bbox_w"]) + 2) for piece in pieces)
    side_hint = int(math.ceil(math.sqrt(padded_area)))
    min_h = max(max_h + 2, max(18, side_hint))
    min_w = max(max_w + 2, max(18, side_hint))
    if min_h > 30 or min_w > 30:
        return None

    order = sorted(
        pieces,
        key=lambda piece: int(piece["bbox_h"]) * int(piece["bbox_w"]),
        reverse=True,
    )

    for _ in range(80):
        grid_h = int(_RNG.integers(min_h, 31))
        grid_w = int(_RNG.integers(min_w, 31))
        grid = np.full((grid_h, grid_w), BG, dtype=int)
        blocked = np.zeros((grid_h, grid_w), dtype=bool)
        ok = True

        for piece in order:
            bbox_h = int(piece["bbox_h"])
            bbox_w = int(piece["bbox_w"])
            options: list[tuple[int, int]] = []
            for r0 in range(grid_h - bbox_h + 1):
                for c0 in range(grid_w - bbox_w + 1):
                    rr0 = max(0, r0 - PLACEMENT_PADDING)
                    rr1 = min(grid_h, r0 + bbox_h + PLACEMENT_PADDING)
                    cc0 = max(0, c0 - PLACEMENT_PADDING)
                    cc1 = min(grid_w, c0 + bbox_w + PLACEMENT_PADDING)
                    if not blocked[rr0:rr1, cc0:cc1].any():
                        options.append((r0, c0))

            if not options:
                ok = False
                break

            r0, c0 = options[int(_RNG.integers(len(options)))]
            rr0 = max(0, r0 - PLACEMENT_PADDING)
            rr1 = min(grid_h, r0 + bbox_h + PLACEMENT_PADDING)
            cc0 = max(0, c0 - PLACEMENT_PADDING)
            cc1 = min(grid_w, c0 + bbox_w + PLACEMENT_PADDING)
            blocked[rr0:rr1, cc0:cc1] = True

            coords = np.asarray(piece["coords"], dtype=int)
            rows = coords[:, 0] + r0
            cols = coords[:, 1] + c0
            grid[rows, cols] = int(piece["color"])

        if ok:
            return grid
    return None


def _is_valid_input(grid: np.ndarray) -> bool:
    try:
        if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
            return False
        colors = [int(color) for color in np.unique(grid) if int(color) != BG]
        if len(colors) < 2:
            return False

        areas = []
        widths = []
        heights = []
        for color in colors:
            comps = _find_components(grid == color)
            if len(comps) != 2:
                return False
            rect_h, rect_w = _unique_rectangle_from_pair(comps[0], comps[1])
            areas.append(rect_h * rect_w)
            widths.append(rect_w)
            heights.append(rect_h)

        if len(set(areas)) != len(areas):
            return False
        if sum(widths) + len(widths) - 1 > 30 or max(heights) > 30:
            return False

        out = main(grid)
        return out.ndim == 2 and max(out.shape) <= 30
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    rects = _choose_rectangles()
    colors = _RNG.choice(PALETTE, size=len(rects), replace=False)
    pieces: list[dict[str, np.ndarray | int]] = []

    for color, (rect_h, rect_w) in zip(colors.tolist(), rects):
        coords_a, coords_b = _make_partition(rect_h, rect_w)
        for coords in (coords_a, coords_b):
            bbox_h, bbox_w = _bbox_shape(coords)
            pieces.append(
                {
                    "color": int(color),
                    "coords": _normalize(coords),
                    "bbox_h": int(bbox_h),
                    "bbox_w": int(bbox_w),
                }
            )

    _RNG.shuffle(pieces)
    grid = _scatter_pieces(pieces)
    if grid is None or not _is_valid_input(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
