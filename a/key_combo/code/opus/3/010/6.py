from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path
import random

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "arc_utils_for_opus_3_010_6",
    str(Path("/home/lixinyu/snap/new/solvers/arc_utils.py")),
).load_module()

Color = _ARC_UTILS.Color
bbox_coords = _ARC_UTILS.bbox_coords
bbox_h = _ARC_UTILS.bbox_h
bbox_w = _ARC_UTILS.bbox_w
components_mask = _ARC_UTILS.components_mask
tile_fill = _ARC_UTILS.tile_fill

PATTERN_COLORS = {Color.BLUE, Color.YELLOW}
OCCLUDER_COLORS = (Color.RED, Color.GREEN, Color.GREY)

HALF_HEIGHT_CHOICES = (8, 9, 10, 11, 12)
WIDTH_CHOICES = (9, 10, 11, 12, 13, 14)
RECT_HEIGHTS = (2, 3, 4)
RECT_WIDTHS = (3, 4, 5)
TARGET_RECT_COUNTS = (2, 3)
GENERATION_RETRIES = 400
PAIR_RETRIES = 200
PLACEMENT_RETRIES = 200

Rect = tuple[int, int, int, int]  # r0, c0, h, w


def _rect_area(rect: Rect) -> int:
    return int(rect[2] * rect[3])


def _rect_to_bbox(rect: Rect) -> tuple[int, int, int, int]:
    r0, c0, h, w = rect
    return int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)


def _rect_mask(shape: tuple[int, int], rects: list[Rect]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    for r0, c0, h, w in rects:
        out[r0 : r0 + h, c0 : c0 + w] = True
    return out


def _paint_rect(grid: np.ndarray, rect: Rect, color: int) -> None:
    r0, c0, h, w = rect
    grid[r0 : r0 + h, c0 : c0 + w] = int(color)


def _rects_touch_or_overlap(a: Rect, b: Rect, pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = _rect_to_bbox(a)
    br0, bc0, br1, bc1 = _rect_to_bbox(b)
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _rect_intersection(a: Rect, b: Rect) -> Rect | None:
    ar0, ac0, ah, aw = a
    br0, bc0, bh, bw = b
    r0 = max(ar0, br0)
    c0 = max(ac0, bc0)
    r1 = min(ar0 + ah, br0 + bh)
    c1 = min(ac0 + aw, bc0 + bw)
    if r0 >= r1 or c0 >= c1:
        return None
    return int(r0), int(c0), int(r1 - r0), int(c1 - c0)


def _solid_rectangles(half: np.ndarray) -> tuple[list[tuple[int, Rect]], np.ndarray]:
    rects: list[tuple[int, Rect]] = []
    mask = np.zeros(half.shape, dtype=bool)
    for color in sorted(int(v) for v in np.unique(half)):
        if color in PATTERN_COLORS:
            continue
        for coords in components_mask(half == color, connectivity=4):
            box = bbox_coords(coords)
            h = bbox_h(box)
            w = bbox_w(box)
            if h <= 1 or w <= 1 or int(coords.shape[0]) != int(h * w):
                continue
            rect = (int(box[0]), int(box[1]), int(h), int(w))
            rects.append((int(color), rect))
            _paint_rect(mask, rect, True)
    return rects, mask


def _visible_observations(
    top: np.ndarray,
    bottom: np.ndarray,
    top_mask: np.ndarray,
    bottom_mask: np.ndarray,
) -> list[tuple[int, int, int]]:
    obs: list[tuple[int, int, int]] = []
    for half, mask in ((top, top_mask), (bottom, bottom_mask)):
        for r, c in np.argwhere(~mask):
            value = int(half[int(r), int(c)])
            assert value in PATTERN_COLORS
            obs.append((int(r), int(c), value))
    return obs


def _candidate_patterns(
    observations: list[tuple[int, int, int]],
    shape: tuple[int, int],
) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    h, w = shape
    candidates: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for row_period in range(1, min(3, h) + 1):
        for col_period in range(1, min(3, w) + 1):
            tile = np.full((row_period, col_period), -1, dtype=int)
            seen = np.zeros((row_period, col_period), dtype=bool)
            ok = True
            for r, c, value in observations:
                rr = r % row_period
                cc = c % col_period
                if seen[rr, cc] and int(tile[rr, cc]) != int(value):
                    ok = False
                    break
                tile[rr, cc] = int(value)
                seen[rr, cc] = True
            if not ok or not bool(seen.all()):
                continue
            full = np.empty(shape, dtype=int)
            for r in range(h):
                for c in range(w):
                    full[r, c] = int(tile[r % row_period, c % col_period])
            candidates.append((int(row_period), int(col_period), tile, full))
    return candidates


def _infer_pattern(
    top: np.ndarray,
    bottom: np.ndarray,
    top_mask: np.ndarray,
    bottom_mask: np.ndarray,
) -> np.ndarray:
    observations = _visible_observations(top, bottom, top_mask, bottom_mask)
    candidates = _candidate_patterns(observations, top.shape)
    assert candidates, "visible pixels must define at least one periodic pattern"
    min_area = min(int(rp * cp) for rp, cp, _tile, _full in candidates)
    best = [full for rp, cp, _tile, full in candidates if int(rp * cp) == int(min_area)]
    first = best[0]
    assert all(np.array_equal(first, full) for full in best), "minimal periodic reconstructions must agree"
    return first


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the shared periodic background only at coordinates occluded in both stacked halves.

    Input structure:
    - The grid has even height and is split into equal top and bottom halves.
    - Each half shows the same blue/yellow periodic background plus 2-3 solid rectangular occluders in red, green, or grey.
    - Occluders differ between halves, so some local coordinates are blocked in both halves while others are blocked in only one half.

    Rule:
    - Detect solid rectangular monochrome components in each half as occluders.
    - Mark local coordinates where the top half and bottom half are both occluded.
    - Infer the smallest row/column periodic tile consistent with every visible non-occluded pixel, using half-local coordinates in both halves.
    - Output one half-sized grid: copy the top half everywhere except replace doubly occluded coordinates by the recovered periodic background color.

    Key insight:
    - The lower half is only a mask source: it tells which top-half occlusions are truly double-covered, while both halves' visible pixels together reconstruct the hidden tile.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] % 2 == 0
    half_h = g.shape[0] // 2
    top = g[:half_h].copy()
    bottom = g[half_h:].copy()

    _top_rects, top_mask = _solid_rectangles(top)
    _bottom_rects, bottom_mask = _solid_rectangles(bottom)
    dual_mask = top_mask & bottom_mask
    assert bool(dual_mask.any()), "at least one local coordinate must be doubly occluded"

    pattern = _infer_pattern(top, bottom, top_mask, bottom_mask)

    out = top.copy()
    out[dual_mask] = pattern[dual_mask]
    return out


def _random_tile(rng: random.Random) -> np.ndarray:
    size = 2 if rng.random() < 0.5 else 3
    if size == 2:
        base = [Color.BLUE, Color.YELLOW]
    else:
        while True:
            base = [rng.choice(PATTERN_COLORS_LIST) for _ in range(size)]
            if len(set(base)) == 2:
                break
    tile = np.empty((size, size), dtype=int)
    for r in range(size):
        for c in range(size):
            tile[r, c] = int(base[(c - r) % size])
    tile = np.roll(tile, shift=(rng.randrange(size), rng.randrange(size)), axis=(0, 1))
    return tile


PATTERN_COLORS_LIST = [Color.BLUE, Color.YELLOW]


def _choose_half_height(rng: random.Random, tile_size: int) -> int:
    off_period = [h for h in HALF_HEIGHT_CHOICES if h % tile_size != 0]
    if off_period and rng.random() < 0.75:
        return int(rng.choice(off_period))
    return int(rng.choice(HALF_HEIGHT_CHOICES))


def _random_rect(rng: random.Random, half_h: int, width: int) -> Rect:
    rect_h = int(rng.choice([h for h in RECT_HEIGHTS if h <= half_h]))
    rect_w = int(rng.choice([w for w in RECT_WIDTHS if w <= width]))
    r0 = int(rng.randint(0, half_h - rect_h))
    c0 = int(rng.randint(0, width - rect_w))
    return r0, c0, rect_h, rect_w


def _overlap_pair(rng: random.Random, shape: tuple[int, int], pattern: np.ndarray) -> tuple[Rect, Rect, Rect] | None:
    half_h, width = shape
    for _ in range(PAIR_RETRIES):
        top_rect = _random_rect(rng, half_h, width)
        bottom_rect = _random_rect(rng, half_h, width)
        inter = _rect_intersection(top_rect, bottom_rect)
        if inter is None:
            continue
        if inter[2] < 2 or inter[3] < 2:
            continue
        if _rect_area(inter) >= _rect_area(top_rect) or _rect_area(inter) >= _rect_area(bottom_rect):
            continue
        r0, c0, h, w = inter
        if len(set(map(int, pattern[r0 : r0 + h, c0 : c0 + w].ravel()))) < 2:
            continue
        return top_rect, bottom_rect, inter
    return None


def _add_rectangles(
    rng: random.Random,
    shape: tuple[int, int],
    rects: list[Rect],
    target_count: int,
) -> bool:
    half_h, width = shape
    for _ in range(PLACEMENT_RETRIES):
        if len(rects) >= int(target_count):
            return True
        candidate = _random_rect(rng, half_h, width)
        if any(_rects_touch_or_overlap(candidate, rect, pad=1) for rect in rects):
            continue
        rects.append(candidate)
    return len(rects) >= int(target_count)


def _distinct_colors(rng: random.Random, count: int) -> list[int]:
    colors = list(OCCLUDER_COLORS)
    rng.shuffle(colors)
    return [int(colors[i]) for i in range(count)]


def _build_half(pattern: np.ndarray, rects: list[Rect], colors: list[int]) -> np.ndarray:
    half = pattern.copy()
    for rect, color in zip(rects, colors):
        _paint_rect(half, rect, color)
    return half


def _expected_output(pattern: np.ndarray, top: np.ndarray, top_mask: np.ndarray, bottom_mask: np.ndarray) -> np.ndarray:
    out = top.copy()
    out[top_mask & bottom_mask] = pattern[top_mask & bottom_mask]
    return out


def _validate_case(case: dict[str, object]) -> bool:
    grid = np.asarray(case["input"], dtype=int)
    pattern = np.asarray(case["pattern"], dtype=int)
    top = np.asarray(case["top"], dtype=int)
    expected = np.asarray(case["expected"], dtype=int)
    top_mask = np.asarray(case["top_mask"], dtype=bool)
    bottom_mask = np.asarray(case["bottom_mask"], dtype=bool)
    dual_mask = top_mask & bottom_mask

    if not bool(dual_mask.any()):
        return False
    if not bool(np.any(top_mask & ~dual_mask)):
        return False
    if not bool(np.any(bottom_mask & ~dual_mask)):
        return False
    if len(set(map(int, pattern[dual_mask].ravel()))) < 2:
        return False
    if np.array_equal(expected, top):
        return False

    inferred = _infer_pattern(top, grid[grid.shape[0] // 2 :], top_mask, bottom_mask)
    if not np.array_equal(inferred, pattern):
        return False
    if not np.array_equal(main(grid), expected):
        return False

    naive_fill = top.copy()
    naive_fill[top_mask] = pattern[top_mask]
    if np.array_equal(naive_fill, expected):
        return False

    if not (2 <= len(case["top_rects"]) <= 3 and 2 <= len(case["bottom_rects"]) <= 3):
        return False

    return True


def _gen_attempt(rng: random.Random) -> dict[str, object] | None:
    tile = _random_tile(rng)
    tile_size = int(tile.shape[0])
    half_h = _choose_half_height(rng, tile_size)
    width = int(rng.choice(WIDTH_CHOICES))
    pattern = tile_fill((half_h, width), tile)

    pair = _overlap_pair(rng, (half_h, width), pattern)
    if pair is None:
        return None
    top_first, bottom_first, _inter = pair

    top_count = int(rng.choice(TARGET_RECT_COUNTS))
    bottom_count = int(rng.choice(TARGET_RECT_COUNTS))
    top_rects = [top_first]
    bottom_rects = [bottom_first]
    if not _add_rectangles(rng, (half_h, width), top_rects, top_count):
        return None
    if not _add_rectangles(rng, (half_h, width), bottom_rects, bottom_count):
        return None

    top_colors = _distinct_colors(rng, len(top_rects))
    bottom_colors = _distinct_colors(rng, len(bottom_rects))
    top = _build_half(pattern, top_rects, top_colors)
    bottom = _build_half(pattern, bottom_rects, bottom_colors)

    detected_top, top_mask = _solid_rectangles(top)
    detected_bottom, bottom_mask = _solid_rectangles(bottom)
    if len(detected_top) != len(top_rects) or len(detected_bottom) != len(bottom_rects):
        return None

    expected = _expected_output(pattern, top, top_mask, bottom_mask)
    case = {
        "input": np.vstack([top, bottom]).astype(int),
        "expected": expected.astype(int),
        "pattern": pattern.astype(int),
        "top": top.astype(int),
        "top_mask": top_mask,
        "bottom_mask": bottom_mask,
        "top_rects": list(top_rects),
        "bottom_rects": list(bottom_rects),
    }
    if not _validate_case(case):
        return None
    return case


def _generate_case() -> dict[str, object]:
    rng = random.Random()
    for _ in range(GENERATION_RETRIES):
        case = _gen_attempt(rng)
        if case is not None:
            return case
    raise RuntimeError("failed to generate a valid stacked-mask periodic-restore case")


def generate_input() -> np.ndarray:
    return np.asarray(_generate_case()["input"], dtype=int)
