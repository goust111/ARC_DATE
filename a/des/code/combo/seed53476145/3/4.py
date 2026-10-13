from __future__ import annotations

from itertools import combinations
import numpy as np


def _background_color(grid: np.ndarray) -> int:
    """Prefer a border-derived background; fall back to the global mode."""
    g = np.asarray(grid, dtype=int)
    border = np.concatenate((g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]))
    values, counts = np.unique(border, return_counts=True)
    candidate = int(values[int(np.argmax(counts))])
    if int(counts.max()) * 2 >= int(border.size):
        return candidate

    values, counts = np.unique(g, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _rectangles_from_visible(grid: np.ndarray, bg: int) -> list[dict[str, int]]:
    rects: list[dict[str, int]] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        pts = np.argwhere(grid == color)
        assert int(len(pts)) > 0
        r0, c0 = map(int, pts.min(axis=0))
        r1, c1 = map(int, pts.max(axis=0))
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        rects.append(
            {
                "color": int(color),
                "r0": int(r0),
                "c0": int(c0),
                "h": int(h),
                "w": int(w),
                "area": int(h * w),
            }
        )
    assert rects
    return rects


def _packed_output(rects: list[dict[str, int]], bg: int) -> np.ndarray:
    ordered = sorted(rects, key=lambda rect: (rect["area"], rect["h"], rect["w"], rect["r0"], rect["c0"]))
    out_h = max(rect["h"] for rect in ordered)
    out_w = sum(rect["w"] for rect in ordered) + len(ordered) - 1
    out = np.full((out_h, out_w), bg, dtype=int)

    col = 0
    for rect in ordered:
        out[out_h - rect["h"] :, col : col + rect["w"]] = rect["color"]
        col += rect["w"] + 1

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild occluded rectangles from visible bboxes, then sort them by area.

    Input structure
    - Grey background.
    - Several solid colored rectangles, with later rectangles painted on top of
      earlier rectangles.
    - For every color, the bbox of its visible pixels is still the bbox of the
      full underlying rectangle.

    Rule
    - Recover each rectangle's size from the bbox of its visible pixels.
    - Compute each rectangle's area.
    - Create a new canvas and place the rectangles from left to right in
      increasing area order, keeping their colors, widths, heights, and
      bottom alignment. Leave one background column between neighbors.

    Key insight
    - The occlusion only matters while recovering rectangle sizes; once every
      rectangle's bbox is known, the output ignores the original layout.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _background_color(g)
    rects = _rectangles_from_visible(g, bg)
    return _packed_output(rects, bg)


# --------------- generator (Codex) ---------------

_RNG = np.random.default_rng()

BACKGROUND_COLOR = 5
COLOR_POOL = [0, 1, 2, 3, 4, 6, 7, 8, 9]

GRID_SIDE_MIN = 15
GRID_SIDE_MAX_EXCLUSIVE = 23
RECT_COUNT_MIN = 3
RECT_COUNT_MAX_EXCLUSIVE = 7

CHAIN_MIN = 2
CHAIN_MAX = 4
MIN_SIDE = 2
CHAIN_BASE_MAX = 10
FREE_RECT_MAX = 7
OUTER_MARGIN = 1
FREE_GAP = 1

PLACEMENT_ATTEMPTS = 160
MAX_GENERATION_ATTEMPTS = 500


def _bbox(rect: dict[str, int]) -> tuple[int, int, int, int]:
    return (
        int(rect["r0"]),
        int(rect["c0"]),
        int(rect["r0"] + rect["h"] - 1),
        int(rect["c0"] + rect["w"] - 1),
    )


def _overlap(a: dict[str, int], b: dict[str, int]) -> bool:
    ar0, ac0, ar1, ac1 = _bbox(a)
    br0, bc0, br1, bc1 = _bbox(b)
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _separated(a: dict[str, int], b: dict[str, int], gap: int) -> bool:
    ar0, ac0, ar1, ac1 = _bbox(a)
    br0, bc0, br1, bc1 = _bbox(b)
    return bool(
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _render_input(rects: list[dict[str, int]], height: int, width: int, bg: int) -> np.ndarray:
    grid = np.full((height, width), bg, dtype=int)
    for rect in rects:
        r0 = int(rect["r0"])
        c0 = int(rect["c0"])
        h = int(rect["h"])
        w = int(rect["w"])
        grid[r0 : r0 + h, c0 : c0 + w] = int(rect["color"])
    return grid


def _random_start(total: int, size: int) -> int | None:
    low = OUTER_MARGIN
    high = total - size - OUTER_MARGIN
    if low > high:
        return None
    return int(_RNG.integers(low, high + 1))


def _sample_chain_dims(chain_len: int, height: int, width: int) -> list[tuple[int, int]] | None:
    min_base = MIN_SIDE + 2 * (chain_len - 1)
    max_h = min(CHAIN_BASE_MAX, height - 2 * OUTER_MARGIN)
    max_w = min(CHAIN_BASE_MAX, width - 2 * OUTER_MARGIN)
    if min_base > max_h or min_base > max_w:
        return None

    dims: list[tuple[int, int]] = []
    prev_h = int(_RNG.integers(min_base, max_h + 1))
    prev_w = int(_RNG.integers(min_base, max_w + 1))
    dims.append((prev_h, prev_w))

    for depth in range(1, chain_len):
        remain = chain_len - depth - 1
        lo = MIN_SIDE + 2 * remain
        hi_h = prev_h - 2
        hi_w = prev_w - 2
        if lo > hi_h or lo > hi_w:
            return None
        cur_h = int(_RNG.integers(lo, hi_h + 1))
        cur_w = int(_RNG.integers(lo, hi_w + 1))
        dims.append((cur_h, cur_w))
        prev_h, prev_w = cur_h, cur_w

    return dims


def _place_chain(height: int, width: int, colors: list[int]) -> list[dict[str, int]] | None:
    dims = _sample_chain_dims(len(colors), height, width)
    if dims is None:
        return None

    base_h, base_w = dims[0]
    base_r0 = _random_start(height, base_h)
    base_c0 = _random_start(width, base_w)
    if base_r0 is None or base_c0 is None:
        return None

    rects: list[dict[str, int]] = [
        {
            "color": int(colors[0]),
            "r0": int(base_r0),
            "c0": int(base_c0),
            "h": int(base_h),
            "w": int(base_w),
            "area": int(base_h * base_w),
        }
    ]

    parent = rects[0]
    for (h, w), color in zip(dims[1:], colors[1:]):
        min_r0 = int(parent["r0"] + 1)
        max_r0 = int(parent["r0"] + parent["h"] - h - 1)
        min_c0 = int(parent["c0"] + 1)
        max_c0 = int(parent["c0"] + parent["w"] - w - 1)
        if min_r0 > max_r0 or min_c0 > max_c0:
            return None

        rect = {
            "color": int(color),
            "r0": int(_RNG.integers(min_r0, max_r0 + 1)),
            "c0": int(_RNG.integers(min_c0, max_c0 + 1)),
            "h": int(h),
            "w": int(w),
            "area": int(h * w),
        }
        rects.append(rect)
        parent = rect

    return rects


def _place_free_rect(
    height: int,
    width: int,
    color: int,
    used_rects: list[dict[str, int]],
    used_areas: set[int],
) -> dict[str, int] | None:
    max_h = min(FREE_RECT_MAX, height - 2 * OUTER_MARGIN)
    max_w = min(FREE_RECT_MAX, width - 2 * OUTER_MARGIN)
    if max_h < MIN_SIDE or max_w < MIN_SIDE:
        return None

    for _ in range(PLACEMENT_ATTEMPTS):
        h = int(_RNG.integers(MIN_SIDE, max_h + 1))
        w = int(_RNG.integers(MIN_SIDE, max_w + 1))
        area = int(h * w)
        if area in used_areas:
            continue
        r0 = _random_start(height, h)
        c0 = _random_start(width, w)
        if r0 is None or c0 is None:
            continue

        rect = {"color": int(color), "r0": int(r0), "c0": int(c0), "h": int(h), "w": int(w), "area": int(area)}
        if all(_separated(rect, other, FREE_GAP) for other in used_rects):
            return rect

    return None


def _visible_bbox_matches(grid: np.ndarray, rect: dict[str, int]) -> bool:
    pts = np.argwhere(grid == int(rect["color"]))
    if int(len(pts)) == 0:
        return False
    r0, c0 = map(int, pts.min(axis=0))
    r1, c1 = map(int, pts.max(axis=0))
    return bool(
        r0 == int(rect["r0"])
        and c0 == int(rect["c0"])
        and r1 == int(rect["r0"] + rect["h"] - 1)
        and c1 == int(rect["c0"] + rect["w"] - 1)
    )


def _background_is_unique_mode(grid: np.ndarray, bg: int) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    bg_count = 0
    for value, count in zip(values, counts):
        if int(value) == int(bg):
            bg_count = int(count)
            break
    return bool(bg_count > 0 and all(int(value) == int(bg) or int(count) < bg_count for value, count in zip(values, counts)))


def _border_is_background(grid: np.ndarray, bg: int) -> bool:
    return bool(
        np.all(grid[0, :] == int(bg))
        and np.all(grid[-1, :] == int(bg))
        and np.all(grid[:, 0] == int(bg))
        and np.all(grid[:, -1] == int(bg))
    )


def _has_occlusion(grid: np.ndarray, rects: list[dict[str, int]]) -> bool:
    for rect in rects:
        visible = int(np.count_nonzero(grid == int(rect["color"])))
        if visible < int(rect["area"]):
            return True
    return False


def _gen_attempt() -> np.ndarray | None:
    height = int(_RNG.integers(GRID_SIDE_MIN, GRID_SIDE_MAX_EXCLUSIVE))
    width = int(_RNG.integers(GRID_SIDE_MIN, GRID_SIDE_MAX_EXCLUSIVE))
    rect_count = int(_RNG.integers(RECT_COUNT_MIN, RECT_COUNT_MAX_EXCLUSIVE))
    chain_len = int(_RNG.integers(CHAIN_MIN, min(CHAIN_MAX, rect_count) + 1))

    colors = _RNG.choice(np.array(COLOR_POOL, dtype=int), size=rect_count, replace=False).tolist()

    chain = _place_chain(height, width, colors[:chain_len])
    if chain is None:
        return None

    rects = chain[:]
    used_areas = {int(rect["area"]) for rect in rects}
    for color in colors[chain_len:]:
        rect = _place_free_rect(height, width, int(color), rects, used_areas)
        if rect is None:
            return None
        rects.append(rect)
        used_areas.add(int(rect["area"]))

    grid = _render_input(rects, height, width, BACKGROUND_COLOR)
    if not _border_is_background(grid, BACKGROUND_COLOR):
        return None
    if not _background_is_unique_mode(grid, BACKGROUND_COLOR):
        return None
    if not any(_overlap(a, b) for a, b in combinations(rects, 2)):
        return None
    if not _has_occlusion(grid, rects):
        return None
    if any(not _visible_bbox_matches(grid, rect) for rect in rects):
        return None

    expected = _packed_output(rects, BACKGROUND_COLOR)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
