from __future__ import annotations

import hashlib
import random
from collections import Counter

import numpy as np


BG = 0
GREY = 5
BAR_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 500
_SEEN_OUTPUTS: set[bytes] = set()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _draw_ring(out: np.ndarray, size: int, color: int) -> None:
    inset = (out.shape[0] - size) // 2
    lo = inset
    hi = out.shape[0] - inset - 1
    out[lo, lo : hi + 1] = color
    out[hi, lo : hi + 1] = color
    out[lo : hi + 1, lo] = color
    out[lo : hi + 1, hi] = color


def _render_output(height_by_color: dict[int, int]) -> np.ndarray:
    sizes = sorted(height_by_color.values())
    n = max(sizes)
    out = np.full((n, n), BG, dtype=int)
    for color, size in sorted(height_by_color.items(), key=lambda kv: kv[1], reverse=True):
        _draw_ring(out, size, color)
    return out


def _parse_grid(grid: np.ndarray) -> tuple[list[dict[str, int]], tuple[int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    grey_components = _components(g == GREY)
    assert len(grey_components) == 1, "expected one grey occluder"
    occ = _bbox(grey_components[0])
    r0, c0, r1, c1 = occ
    assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == GREY), "grey occluder must be solid"

    bars: list[dict[str, int]] = []
    for color in sorted(set(map(int, g.ravel())) - {BG, GREY}):
        comps = _components(g == color)
        assert len(comps) == 1, "each bar color must form one component"
        comp = comps[0]
        br0, bc0, br1, bc1 = _bbox(comp)
        assert bc0 == bc1, "bars must be one cell wide"
        assert int(comp.shape[0]) == br1 - br0 + 1, "bars must be solid vertical segments"
        bars.append(
            {
                "color": int(color),
                "top": int(br0),
                "col": int(bc0),
                "bottom": int(br1),
                "visible_height": int(br1 - br0 + 1),
            }
        )

    assert len(bars) >= 4, "expected at least four bars"
    colors = [bar["color"] for bar in bars]
    assert len(colors) == len(set(colors)), "bar colors must be distinct"
    return bars, occ


def _infer_heights(bars: list[dict[str, int]], occ: tuple[int, int, int, int]) -> dict[int, int]:
    or0, oc0, or1, oc1 = occ
    hidden = [
        bar
        for bar in bars
        if oc0 <= bar["col"] <= oc1 and bar["top"] == or1 + 1
    ]
    assert len(hidden) == 1, "occluder must cover exactly one bar"
    hidden_bar = hidden[0]

    visible_heights = [bar["visible_height"] for bar in bars if bar["color"] != hidden_bar["color"]]
    assert len(visible_heights) == len(bars) - 1
    assert all(height % 2 == 1 for height in visible_heights), "unoccluded heights must stay odd"

    lo = min(visible_heights)
    hi = max(visible_heights)
    expected = list(range(lo, hi + 1, 2))
    missing = [height for height in expected if height not in visible_heights]
    assert len(expected) == len(bars), "visible heights must span the full odd sequence"
    assert len(missing) == 1, "there must be one missing sequence term"
    hidden_height = missing[0]
    assert hidden_height % 2 == 1
    assert hidden_height > hidden_bar["visible_height"], "occlusion must shorten the hidden bar"

    result = {bar["color"]: bar["visible_height"] for bar in bars if bar["color"] != hidden_bar["color"]}
    result[hidden_bar["color"]] = hidden_height
    return result


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct concentric square rings from bar heights, inferring the occluded missing height.

    Input structure:
    - The grid has a black background with several separated one-cell-wide vertical bars of distinct colors.
    - Exactly one solid grey rectangle covers the upper segment of one bar, so that bar appears shorter than it truly is.
    - The bars' true heights are odd numbers forming a common-difference-2 arithmetic progression.

    Rule:
    - Read the visible heights of the unoccluded bars and sort them in ascending order.
    - Recover the single missing term of that odd step-2 sequence; that missing term is the hidden bar's true height.
    - Create a square output whose side equals the maximum true height, then draw centered one-cell-thick square rings using each bar's color and true height.

    Key insight:
    - The input bars are only height carriers for rings that never appear in the input, and the grey rectangle is a clue for sequence completion rather than an output object.
    """
    bars, occ = _parse_grid(np.asarray(grid, dtype=int))
    return _render_output(_infer_heights(bars, occ))


def _sample_spaced_columns(width: int, count: int, gap: int) -> list[int] | None:
    slots = list(range(1, width - 1, gap))
    if len(slots) < count:
        return None
    chosen = random.sample(slots, count)
    random.shuffle(chosen)
    return chosen


def _pick_dimensions(max_height: int, count: int) -> tuple[int, int]:
    min_h = max_height + 2
    max_h = min(30, max_height + 8)
    for _ in range(100):
        h = random.randint(min_h, max_h)
        w_min = max(18, 4 * count + 3)
        w = random.randint(w_min, 30)
        if _sample_spaced_columns(w, count, gap=4) is not None:
            return h, w
    raise RuntimeError("failed to pick canvas dimensions")


def _build_candidate() -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    count = random.randint(4, 6)
    max_start = min(11, 21 - 2 * (count - 1))
    smallest = random.choice(list(range(3, max_start + 1, 2)))
    heights = [smallest + 2 * i for i in range(count)]
    max_height = heights[-1]
    h, w = _pick_dimensions(max_height, count)

    colors = random.sample(BAR_COLORS, count)
    x_positions = _sample_spaced_columns(w, count, gap=4)
    assert x_positions is not None

    align_bottoms = random.random() < 0.5
    shared_bottom = random.randint(max_height, h - 1) if align_bottoms else None
    bottoms = []
    for height in heights:
        low = max(height - 1, max_height // 2)
        if shared_bottom is not None:
            bottoms.append(shared_bottom)
        else:
            bottoms.append(random.randint(low, h - 1))

    hidden_idx = random.randint(1, count - 2)
    hidden_height = heights[hidden_idx]
    visible_hidden_height = random.choice(list(range(2, hidden_height, 2)))
    occlusion_height = hidden_height - visible_hidden_height
    occluder_width = random.choice((1, 3))
    if occluder_width == 3 and (x_positions[hidden_idx] <= 0 or x_positions[hidden_idx] >= w - 1):
        occluder_width = 1

    order = list(range(count))
    random.shuffle(order)
    grid = np.full((h, w), BG, dtype=int)
    spec_bars = []
    height_by_color: dict[int, int] = {}
    for idx in order:
        color = colors[idx]
        height = heights[idx]
        bottom = bottoms[idx]
        top = bottom - height + 1
        col = x_positions[idx]
        height_by_color[color] = height
        if idx == hidden_idx:
            visible_top = bottom - visible_hidden_height + 1
            grid[visible_top : bottom + 1, col] = color
            occ_top = top
            occ_left = col - (occluder_width // 2)
            grid[occ_top : visible_top, occ_left : occ_left + occluder_width] = GREY
            occ = (occ_top, occ_left, visible_top - 1, occ_left + occluder_width - 1)
            spec_bars.append((color, col, top, bottom, visible_top))
        else:
            grid[top : bottom + 1, col] = color
            spec_bars.append((color, col, top, bottom, top))

    expected = _render_output(height_by_color)
    spec = {
        "bars": spec_bars,
        "heights": dict(height_by_color),
        "hidden_color": colors[hidden_idx],
        "hidden_visible_height": visible_hidden_height,
        "heights_in_x_order": [height_by_color[colors[i]] for i in np.argsort(x_positions)],
        "occluder": occ,
    }
    return grid, expected, spec


def _visible_height_output(spec: dict[str, object]) -> np.ndarray:
    heights = dict(spec["heights"])
    heights[int(spec["hidden_color"])] = int(spec["hidden_visible_height"])
    return _render_output(heights)


def _validate_candidate(grid: np.ndarray, expected: np.ndarray, spec: dict[str, object]) -> bool:
    try:
        actual = main(grid)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    bars, occ = _parse_grid(grid)
    inferred = _infer_heights(bars, occ)
    if inferred != spec["heights"]:
        return False

    hidden_color = int(spec["hidden_color"])
    hidden_true = int(spec["heights"][hidden_color])
    hidden_visible = int(spec["hidden_visible_height"])
    if hidden_visible % 2 != 0 or hidden_visible >= hidden_true:
        return False

    x_heights = list(spec["heights_in_x_order"])
    if x_heights == sorted(x_heights) or x_heights == sorted(x_heights, reverse=True):
        return False

    if np.array_equal(_visible_height_output(spec), expected):
        return False

    occ_r0, occ_c0, occ_r1, occ_c1 = spec["occluder"]
    if not np.all(grid[occ_r0 : occ_r1 + 1, occ_c0 : occ_c1 + 1] == GREY):
        return False

    colors = Counter(map(int, expected.ravel()))
    if max(colors.values()) == expected.size:
        return False
    if np.count_nonzero(expected == BG) == 0:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid, expected, spec = _build_candidate()
        sig = hashlib.md5(expected.tobytes() + str(expected.shape).encode()).digest()
        if sig in _SEEN_OUTPUTS:
            continue
        if not _validate_candidate(grid, expected, spec):
            continue
        _SEEN_OUTPUTS.add(sig)
        return grid
    raise RuntimeError("failed to generate valid input")
