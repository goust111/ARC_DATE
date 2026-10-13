from __future__ import annotations

from collections import Counter
from importlib.machinery import SourceFileLoader
from pathlib import Path
import random

import numpy as np

_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")


def _fallback_get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _fallback_bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _fallback_find_components(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    connectivity: int = 4,
    monochromatic: bool = True,
    colors: list[int] | None = None,
    min_area: int = 1,
    return_format: str = "dict",
):
    if bg is None:
        bg = _fallback_get_bg(grid)
    if not monochromatic or return_format != "dict":
        raise NotImplementedError("fallback find_components only supports monochromatic dict mode")

    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    elif connectivity == 8:
        steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
    else:
        raise ValueError("connectivity must be 4 or 8")

    h, w = grid.shape
    if colors is None:
        colors_iter = sorted(set(map(int, grid.ravel())) - {int(bg)})
    else:
        colors_iter = [int(color) for color in colors if int(color) != int(bg)]

    out = []
    for color in colors_iter:
        mask = grid == color
        seen = np.zeros((h, w), dtype=bool)
        for r in range(h):
            for c in range(w):
                if not mask[r, c] or seen[r, c]:
                    continue
                stack = [(r, c)]
                seen[r, c] = True
                cells = []
                while stack:
                    rr, cc = stack.pop()
                    cells.append((rr, cc))
                    for dr, dc in steps:
                        nr = rr + dr
                        nc = cc + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))

                if len(cells) < int(min_area):
                    continue
                coords = np.array(cells, dtype=int)
                out.append(
                    {
                        "color": int(color),
                        "coords": coords,
                        "bbox": _fallback_bbox_coords(coords),
                        "area": int(coords.shape[0]),
                    }
                )
    return out


try:
    if not _ARC_UTILS_PATH.exists():
        raise FileNotFoundError(_ARC_UTILS_PATH)
    _arc_utils = SourceFileLoader("_arc_utils_027_3", str(_ARC_UTILS_PATH)).load_module()
    get_bg = _arc_utils.get_bg
    bbox_coords = _arc_utils.bbox_coords
    find_components = _arc_utils.find_components
except Exception:
    get_bg = _fallback_get_bg
    bbox_coords = _fallback_bbox_coords
    find_components = _fallback_find_components


BG = 0
PALETTE = tuple(range(1, 10))
GEN_ATTEMPTS = 400
SCENARIO_CYCLE = ("size_shift", "color_shift", "three_odd")
_GENERATION_CALLS = 0


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _paint_ring(grid: np.ndarray, inset: int, color: int) -> None:
    r0 = int(inset)
    c0 = int(inset)
    r1 = int(grid.shape[0] - 1 - inset)
    c1 = int(grid.shape[1] - 1 - inset)
    if r0 > r1 or c0 > c1:
        return
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _render_rings(size: int, ordered_segments: list[dict]) -> np.ndarray:
    out = np.zeros((int(size), int(size)), dtype=int)
    max_layers = (int(size) + 1) // 2
    for inset, segment in enumerate(ordered_segments[:max_layers]):
        _paint_ring(out, inset, int(segment["color"]))
    return out


def _extract_segments(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))
    assert bg == BG, "background must be black"

    components = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    assert 4 <= len(components) <= 6, "expected 4 to 6 line segments"

    owner = np.full(g.shape, -1, dtype=int)
    segments = []
    seen_colors = set()
    seen_lengths = set()
    for idx, comp in enumerate(components):
        color = int(comp["color"])
        coords = np.asarray(comp["coords"], dtype=int)
        r0, c0, r1, c1 = map(int, comp["bbox"])
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        area = int(comp["area"])

        assert color != bg
        assert color not in seen_colors, "segment colors must be unique"
        seen_colors.add(color)

        assert h == 1 or w == 1, "every object must be a straight line segment"
        length = max(h, w)
        assert 2 <= length <= 7, "segment length out of allowed range"
        assert area == length, "segments must be solid with no gaps"
        assert length not in seen_lengths, "segment lengths must be unique"
        seen_lengths.add(length)

        if h == 1:
            cols = sorted(int(c) for _r, c in coords)
            assert cols == list(range(c0, c1 + 1))
        else:
            rows = sorted(int(r) for r, _c in coords)
            assert rows == list(range(r0, r1 + 1))

        for r, c in coords:
            owner[int(r), int(c)] = idx

        segments.append(
            {
                "color": color,
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "area": area,
                "length": length,
            }
        )

    for idx, segment in enumerate(segments):
        for r, c in segment["coords"]:
            for nr, nc in _neighbors8(int(r), int(c)):
                if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]:
                    other = int(owner[nr, nc])
                    assert other in (-1, idx), "segments must not touch in the 8-neighborhood"

    odd_count = sum(segment["length"] % 2 == 1 for segment in segments)
    even_count = len(segments) - odd_count
    assert odd_count >= 2, "need at least two odd-length segments"
    assert even_count >= 1, "need at least one even-length distractor"
    return segments


def _solve(grid: np.ndarray) -> np.ndarray:
    segments = _extract_segments(grid)
    odd_segments = sorted(
        [segment for segment in segments if segment["length"] % 2 == 1],
        key=lambda segment: int(segment["length"]),
        reverse=True,
    )
    assert odd_segments
    return _render_rings(int(odd_segments[0]["length"]), odd_segments)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Filter out even-length line segments, then turn the remaining odd ones into concentric square rings.

    Input structure:
    - Black is the background.
    - Foreground objects are isolated monochromatic horizontal or vertical line segments.
    - Each segment has a unique color and a unique length, and different segments do not touch even diagonally.

    Rule:
    - Identify every foreground line segment and measure its length.
    - Discard all even-length segments.
    - Sort the remaining odd-length segments by decreasing length.
    - Let N be the longest remaining odd length and create an N-by-N black output grid.
    - Fill successive concentric square layers from outside to inside with the sorted segment colors; any untouched inner layers stay black.

    Key insight:
    - Even-length segments are distractors: parity filtering changes both which colors survive and which size the concentric output must use.
    """
    return _solve(np.asarray(grid, dtype=int))


def _segment_cells(top: int, left: int, length: int, orientation: str) -> list[tuple[int, int]]:
    if orientation == "h":
        return [(int(top), int(left + offset)) for offset in range(int(length))]
    return [(int(top + offset), int(left)) for offset in range(int(length))]


def _can_place(cells: list[tuple[int, int]], forbidden: np.ndarray) -> bool:
    h, w = forbidden.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        for nr, nc in _neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w and forbidden[nr, nc]:
                return False
        if forbidden[r, c]:
            return False
    return True


def _mark_forbidden(cells: list[tuple[int, int]], forbidden: np.ndarray) -> None:
    h, w = forbidden.shape
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    forbidden[nr, nc] = True


def _place_segment(
    length: int,
    forbidden: np.ndarray,
    *,
    fixed_top: int | None = None,
    attempts: int = 120,
) -> dict | None:
    h, w = forbidden.shape
    orientations = ["h", "v"]
    for _ in range(attempts):
        random.shuffle(orientations)
        for orientation in orientations:
            if orientation == "h":
                row_min = 0 if fixed_top is None else int(fixed_top)
                row_max = h - 1 if fixed_top is None else int(fixed_top)
                col_max = w - int(length)
            else:
                row_min = 0 if fixed_top is None else int(fixed_top)
                row_max = h - int(length) if fixed_top is None else int(fixed_top)
                col_max = w - 1
            if row_min > row_max or col_max < 0:
                continue
            top = random.randint(int(row_min), int(row_max))
            left = random.randint(0, int(col_max))
            cells = _segment_cells(top, left, int(length), orientation)
            if not _can_place(cells, forbidden):
                continue
            _mark_forbidden(cells, forbidden)
            coords = np.array(cells, dtype=int)
            return {
                "coords": coords,
                "bbox": bbox_coords(coords),
                "length": int(length),
                "orientation": orientation,
            }
    return None


def _sample_lengths(scenario: str) -> list[int]:
    if scenario == "size_shift":
        extras = random.sample([2, 4], k=random.randint(1, 2))
        lengths = [6, 5, 3, *extras]
    elif scenario == "color_shift":
        second_odd = random.choice([5, 3])
        extras = [6, *random.sample([2, 4], k=random.randint(1, 2))]
        lengths = [7, second_odd, *extras]
    elif scenario == "three_odd":
        mandatory_even = random.choice([4, 6])
        remaining_evens = [value for value in (2, 4, 6) if value != mandatory_even]
        extra_count = random.randint(0, len(remaining_evens))
        lengths = [7, 5, 3, mandatory_even, *random.sample(remaining_evens, k=extra_count)]
    else:
        raise ValueError(f"unknown scenario: {scenario}")
    return sorted(lengths, reverse=True)


def _row_targets(height: int, odd_lengths: list[int]) -> list[int]:
    max_odd = max(odd_lengths)
    usable_max = int(height - max_odd)
    assert usable_max >= 0
    if len(odd_lengths) == 1:
        return [usable_max // 2]
    return [int(value) for value in np.linspace(0, usable_max, len(odd_lengths), dtype=int)]


def _build_segments(lengths: list[int]) -> list[dict] | None:
    odd_lengths = sorted([length for length in lengths if length % 2 == 1], reverse=True)
    even_lengths = [length for length in lengths if length % 2 == 0]
    max_length = max(lengths)

    min_height = max(12, max_length + 2 * len(odd_lengths) + 1)
    height = random.randint(min_height, min(18, min_height + 3))
    min_width = max(11, max_length + 3)
    width = random.randint(min_width, min(18, min_width + 4))
    forbidden = np.zeros((height, width), dtype=bool)

    odd_order = odd_lengths[:]
    while True:
        random.shuffle(odd_order)
        if odd_order != odd_lengths:
            break

    placed = []
    for fixed_top, length in zip(_row_targets(height, odd_lengths), odd_order):
        segment = _place_segment(length, forbidden, fixed_top=fixed_top)
        if segment is None:
            return None
        placed.append(segment)

    even_order = even_lengths[:]
    random.shuffle(even_order)
    for length in even_order:
        segment = _place_segment(length, forbidden)
        if segment is None:
            return None
        placed.append(segment)

    return placed


def _assign_colors(segments: list[dict]) -> list[dict]:
    colors = random.sample(PALETTE, k=len(segments))
    while True:
        random.shuffle(colors)
        candidate = [dict(segment, color=int(color)) for segment, color in zip(segments, colors)]
        odd = sorted(
            [segment for segment in candidate if segment["length"] % 2 == 1],
            key=lambda segment: int(segment["length"]),
            reverse=True,
        )
        length_order_colors = [int(segment["color"]) for segment in odd]
        row_order_colors = [
            int(segment["color"])
            for segment in sorted(odd, key=lambda segment: (segment["bbox"][0], segment["bbox"][1]))
        ]
        if length_order_colors != row_order_colors:
            return candidate


def _segments_to_grid(segments: list[dict]) -> np.ndarray:
    r1 = max(int(segment["bbox"][2]) for segment in segments)
    c1 = max(int(segment["bbox"][3]) for segment in segments)
    grid = np.zeros((r1 + 1, c1 + 1), dtype=int)
    for segment in segments:
        for r, c in segment["coords"]:
            grid[int(r), int(c)] = int(segment["color"])
    return grid


def _validate_case(grid: np.ndarray) -> bool:
    segments = _extract_segments(grid)
    odd = sorted(
        [segment for segment in segments if segment["length"] % 2 == 1],
        key=lambda segment: int(segment["length"]),
        reverse=True,
    )
    correct = _render_rings(int(odd[0]["length"]), odd)

    naive_all = _render_rings(
        int(max(segment["length"] for segment in segments)),
        sorted(segments, key=lambda segment: int(segment["length"]), reverse=True),
    )
    if np.array_equal(correct, naive_all):
        return False

    spatial_order = sorted(odd, key=lambda segment: (segment["bbox"][0], segment["bbox"][1]))
    naive_spatial = _render_rings(int(odd[0]["length"]), spatial_order)
    if np.array_equal(correct, naive_spatial):
        return False

    return True


def _next_scenario() -> str:
    global _GENERATION_CALLS
    scenario = SCENARIO_CYCLE[_GENERATION_CALLS % len(SCENARIO_CYCLE)]
    _GENERATION_CALLS += 1
    return scenario


def generate_input() -> np.ndarray:
    scenario = _next_scenario()
    for _ in range(GEN_ATTEMPTS):
        lengths = _sample_lengths(scenario)
        segments = _build_segments(lengths)
        if segments is None:
            continue
        colored_segments = _assign_colors(segments)
        grid = _segments_to_grid(colored_segments)
        if _validate_case(grid):
            return grid
    raise RuntimeError(f"failed to generate valid input for scenario {scenario}")
