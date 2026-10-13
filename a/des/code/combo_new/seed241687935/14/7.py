import random

import numpy as np


OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}
COLORS = tuple(range(10))
GEN_RETRIES = 500


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _project_region(
    obj: dict[str, int], height: int, width: int
) -> tuple[int, int, int, int] | None:
    direction = obj["direction"]
    length = obj["proj_len"]
    if direction == "up":
        r0, r1 = obj["outer_r0"] - length, obj["outer_r0"] - 1
        c0, c1 = obj["win_c0"], obj["win_c1"]
    elif direction == "down":
        r0, r1 = obj["outer_r1"] + 1, obj["outer_r1"] + length
        c0, c1 = obj["win_c0"], obj["win_c1"]
    elif direction == "left":
        r0, r1 = obj["win_r0"], obj["win_r1"]
        c0, c1 = obj["outer_c0"] - length, obj["outer_c0"] - 1
    else:
        r0, r1 = obj["win_r0"], obj["win_r1"]
        c0, c1 = obj["outer_c1"] + 1, obj["outer_c1"] + length

    r0 = max(0, r0)
    c0 = max(0, c0)
    r1 = min(height - 1, r1)
    c1 = min(width - 1, c1)
    if r0 > r1 or c0 > c1:
        return None
    return r0, c0, r1, c1


def _parse_objects(grid: np.ndarray) -> tuple[int, list[dict[str, int]]]:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    objs: list[dict[str, int]] = []
    for comp in _components4(g != bg):
        outer_r0, outer_c0, outer_r1, outer_c1 = _bbox(comp)
        crop = g[outer_r0 : outer_r1 + 1, outer_c0 : outer_c1 + 1]
        assert np.all(crop != bg), "each foreground component must fill its bbox"

        colors = [int(v) for v in np.unique(crop)]
        assert len(colors) == 2, "each object must contain exactly two non-background colors"
        counts = {color: int(np.count_nonzero(crop == color)) for color in colors}
        outer_color = max(colors, key=counts.get)
        window_color = min(colors, key=counts.get)
        assert counts[outer_color] > counts[window_color], "window must be smaller than the outer rectangle"

        window_points = np.argwhere(crop == window_color)
        wr0, wc0 = map(int, window_points.min(axis=0))
        wr1, wc1 = map(int, window_points.max(axis=0))
        assert int(window_points.shape[0]) == (wr1 - wr0 + 1) * (wc1 - wc0 + 1), "window must be rectangular"

        top_gap = wr0
        bottom_gap = crop.shape[0] - 1 - wr1
        left_gap = wc0
        right_gap = crop.shape[1] - 1 - wc1
        gaps = {"up": int(top_gap), "down": int(bottom_gap), "left": int(left_gap), "right": int(right_gap)}

        direction = min(gaps, key=gaps.get)
        nearest_gap = gaps[direction]
        assert nearest_gap >= 1, "window must stay strictly inside the outer rectangle"
        assert list(gaps.values()).count(nearest_gap) == 1, "window offset direction must be unique"
        proj_len = gaps[OPPOSITE[direction]]
        assert proj_len > nearest_gap, "projection length is encoded by the larger opposite-side gap"

        objs.append(
            {
                "outer_r0": outer_r0,
                "outer_c0": outer_c0,
                "outer_r1": outer_r1,
                "outer_c1": outer_c1,
                "outer_color": int(outer_color),
                "window_color": int(window_color),
                "win_r0": outer_r0 + wr0,
                "win_c0": outer_c0 + wc0,
                "win_r1": outer_r0 + wr1,
                "win_c1": outer_c0 + wc1,
                "direction": direction,
                "proj_len": int(proj_len),
            }
        )
    objs.sort(key=lambda obj: (obj["outer_r0"], obj["outer_c0"]))
    return bg, objs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill background strips outside each offset window, then restore the outer rectangles to solid blocks.

    Input structure:
    - The background is the dominant color.
    - Several disconnected foreground objects are axis-aligned solid rectangles.
    - Each rectangle contains one smaller solid rectangular window of a different color, with the window closer to exactly one side than to the opposite side.

    Rule:
    - For each foreground rectangle, identify the inner window and the side it leans toward.
    - Recolor the window itself back to the outer rectangle color, making the rectangle solid again.
    - Outside the leaned-toward side, paint a same-cross-section strip aligned with the window: it uses the window color and extends by the window's distance to the opposite side of the outer rectangle.
    - Only original background pixels are painted, and any part of the strip outside the grid is clipped away.

    Key insight:
    - The near-side offset determines the projection direction, while the far-side gap encodes the external strip length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape
    bg, objects = _parse_objects(g)

    out = g.copy()
    bg_mask = g == bg

    for obj in objects:
        out[obj["win_r0"] : obj["win_r1"] + 1, obj["win_c0"] : obj["win_c1"] + 1] = obj["outer_color"]

    for obj in objects:
        region = _project_region(obj, height, width)
        if region is None:
            continue
        r0, c0, r1, c1 = region
        strip = out[r0 : r1 + 1, c0 : c1 + 1]
        mask = bg_mask[r0 : r1 + 1, c0 : c1 + 1]
        strip[mask] = obj["window_color"]

    return out


def _expand_once(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            src_r0 = max(0, -dr)
            src_r1 = h - max(0, dr)
            src_c0 = max(0, -dc)
            src_c1 = w - max(0, dc)
            dst_r0 = max(0, dr)
            dst_r1 = h - max(0, -dr)
            dst_c0 = max(0, dc)
            dst_c1 = w - max(0, -dc)
            out[dst_r0:dst_r1, dst_c0:dst_c1] |= mask[src_r0:src_r1, src_c0:src_c1]
    return out


def _sample_spec(rng: random.Random, direction: str) -> dict[str, int | str]:
    if direction in ("up", "down"):
        window_h = rng.randint(2, 4)
        proj_len = rng.randint(2, 6)
        window_w = rng.randint(2, 5)
        left_gap = rng.randint(2, 4)
        right_gap = rng.randint(2, 4)
        top_gap, bottom_gap = (1, proj_len) if direction == "up" else (proj_len, 1)
    else:
        window_h = rng.randint(2, 5)
        proj_len = rng.randint(2, 6)
        window_w = rng.randint(2, 4)
        top_gap = rng.randint(2, 4)
        bottom_gap = rng.randint(2, 4)
        left_gap, right_gap = (1, proj_len) if direction == "left" else (proj_len, 1)

    outer_h = top_gap + window_h + bottom_gap
    outer_w = left_gap + window_w + right_gap
    return {
        "direction": direction,
        "proj_len": int(proj_len),
        "window_h": int(window_h),
        "window_w": int(window_w),
        "top_gap": int(top_gap),
        "bottom_gap": int(bottom_gap),
        "left_gap": int(left_gap),
        "right_gap": int(right_gap),
        "outer_h": int(outer_h),
        "outer_w": int(outer_w),
    }


def _position_ranges(
    spec: dict[str, int | str], height: int, width: int, clipped: bool
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    direction = str(spec["direction"])
    outer_h = int(spec["outer_h"])
    outer_w = int(spec["outer_w"])
    proj_len = int(spec["proj_len"])

    if outer_h > height or outer_w > width:
        return None

    if direction == "up":
        r_lo, r_hi = (1, min(proj_len - 1, height - outer_h)) if clipped else (proj_len, height - outer_h)
        c_lo, c_hi = 0, width - outer_w
    elif direction == "down":
        if clipped:
            r_lo = max(0, height - outer_h - (proj_len - 1))
            r_hi = height - outer_h - 1
        else:
            r_lo, r_hi = 0, height - outer_h - proj_len
        c_lo, c_hi = 0, width - outer_w
    elif direction == "left":
        r_lo, r_hi = 0, height - outer_h
        c_lo, c_hi = (1, min(proj_len - 1, width - outer_w)) if clipped else (proj_len, width - outer_w)
    else:
        r_lo, r_hi = 0, height - outer_h
        if clipped:
            c_lo = max(0, width - outer_w - (proj_len - 1))
            c_hi = width - outer_w - 1
        else:
            c_lo, c_hi = 0, width - outer_w - proj_len

    if r_lo > r_hi or c_lo > c_hi:
        return None
    return (int(r_lo), int(r_hi)), (int(c_lo), int(c_hi))


def _placed_object(
    spec: dict[str, int | str], outer_color: int, window_color: int, r0: int, c0: int
) -> dict[str, int | str]:
    top_gap = int(spec["top_gap"])
    left_gap = int(spec["left_gap"])
    window_h = int(spec["window_h"])
    window_w = int(spec["window_w"])
    outer_h = int(spec["outer_h"])
    outer_w = int(spec["outer_w"])
    return {
        "direction": str(spec["direction"]),
        "proj_len": int(spec["proj_len"]),
        "outer_color": int(outer_color),
        "window_color": int(window_color),
        "outer_r0": int(r0),
        "outer_c0": int(c0),
        "outer_r1": int(r0 + outer_h - 1),
        "outer_c1": int(c0 + outer_w - 1),
        "win_r0": int(r0 + top_gap),
        "win_c0": int(c0 + left_gap),
        "win_r1": int(r0 + top_gap + window_h - 1),
        "win_c1": int(c0 + left_gap + window_w - 1),
    }


def _footprint(obj: dict[str, int | str], height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[obj["outer_r0"] : obj["outer_r1"] + 1, obj["outer_c0"] : obj["outer_c1"] + 1] = True
    region = _project_region(obj, height, width)
    if region is not None:
        r0, c0, r1, c1 = region
        mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _draw_object(grid: np.ndarray, obj: dict[str, int | str]) -> None:
    grid[obj["outer_r0"] : obj["outer_r1"] + 1, obj["outer_c0"] : obj["outer_c1"] + 1] = obj["outer_color"]
    grid[obj["win_r0"] : obj["win_r1"] + 1, obj["win_c0"] : obj["win_c1"] + 1] = obj["window_color"]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(16, 30)
    width = rng.randint(16, 30)
    n_objects = rng.choice((2, 3, 4)) if min(height, width) >= 20 else rng.choice((2, 2, 3))

    palette = list(COLORS)
    rng.shuffle(palette)
    bg = palette[0]
    colors = palette[1 : 1 + 2 * n_objects]

    directions = ["up", "down", "left", "right"]
    rng.shuffle(directions)
    directions = directions[:n_objects]

    grid = np.full((height, width), bg, dtype=int)
    reserved = np.zeros((height, width), dtype=bool)
    placed: list[dict[str, int | str]] = []

    clip_budget = 1 if n_objects == 2 else rng.choice((1, 2))
    clip_flags = [True] * clip_budget + [False] * (n_objects - clip_budget)
    rng.shuffle(clip_flags)

    for idx, direction in enumerate(directions):
        spec = _sample_spec(rng, direction)
        ranges = _position_ranges(spec, height, width, clip_flags[idx])
        if ranges is None:
            return None
        (r_lo, r_hi), (c_lo, c_hi) = ranges

        candidates = [(r, c) for r in range(r_lo, r_hi + 1) for c in range(c_lo, c_hi + 1)]
        rng.shuffle(candidates)
        outer_color = colors[2 * idx]
        window_color = colors[2 * idx + 1]

        found = False
        for r0, c0 in candidates[:120]:
            obj = _placed_object(spec, outer_color, window_color, r0, c0)
            footprint = _footprint(obj, height, width)
            expanded = _expand_once(footprint)
            if np.any(expanded & reserved):
                continue
            _draw_object(grid, obj)
            reserved |= expanded
            placed.append(obj)
            found = True
            break
        if not found:
            return None

    try:
        _, parsed = _parse_objects(grid)
    except AssertionError:
        return None
    if len(parsed) != len(placed):
        return None

    out = main(grid)
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
