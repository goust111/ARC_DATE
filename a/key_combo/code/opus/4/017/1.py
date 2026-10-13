from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np


def _load_arc_utils():
    path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    spec = importlib.util.spec_from_file_location("_arc_utils_017_1", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load arc_utils from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


try:
    _ARC_UTILS = _load_arc_utils()
    Color = _ARC_UTILS.Color
    find_components = _ARC_UTILS.find_components
except Exception:
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        YELLOW = 4
        GREY = 5

    def find_components(
        grid: np.ndarray,
        *,
        bg: int | None = None,
        connectivity: int = 4,
        monochromatic: bool = True,
        colors: list[int] | tuple[int, ...] | None = None,
        min_area: int = 1,
        return_format: str = "dict",
    ) -> list[dict]:
        if connectivity != 4 or not monochromatic or return_format != "dict":
            raise ValueError("fallback find_components only supports 4-connected monochromatic dict mode")
        g = np.asarray(grid, dtype=int)
        if bg is None:
            values, counts = np.unique(g, return_counts=True)
            bg = int(values[np.argmax(counts)])
        if colors is None:
            targets = [int(c) for c in np.unique(g) if int(c) != int(bg)]
        else:
            targets = [int(c) for c in colors if int(c) != int(bg)]
        out: list[dict] = []
        for color in targets:
            mask = g == color
            seen = np.zeros(mask.shape, dtype=bool)
            for r, c in np.argwhere(mask):
                r = int(r)
                c = int(c)
                if seen[r, c]:
                    continue
                stack = [(r, c)]
                seen[r, c] = True
                coords: list[tuple[int, int]] = []
                while stack:
                    rr, cc = stack.pop()
                    coords.append((rr, cc))
                    for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                        if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
                if len(coords) < int(min_area):
                    continue
                arr = np.array(coords, dtype=int)
                r0, c0 = arr.min(axis=0)
                r1, c1 = arr.max(axis=0)
                out.append(
                    {
                        "color": color,
                        "coords": arr,
                        "bbox": (int(r0), int(c0), int(r1), int(c1)),
                        "area": int(arr.shape[0]),
                    }
                )
        return out


GRID_MIN = 11
GRID_MAX = 18
MARKER_COUNTS = (2, 3)
SEGMENT_MIN = 2
RANK_COLORS = (Color.BLUE, Color.GREEN, Color.YELLOW)
GENERATION_ATTEMPTS = 400
TIE_CASE_PROBABILITY = 0.5


def _validate_input(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    colors = set(map(int, np.unique(g)))
    assert colors <= {Color.BLACK, Color.RED}
    markers = np.argwhere(g == Color.RED)
    assert markers.shape[0] in MARKER_COUNTS
    rows = markers[:, 0]
    cols = markers[:, 1]
    assert np.unique(rows).size == rows.size
    assert np.unique(cols).size == cols.size
    assert np.all((markers[:, 0] > 0) & (markers[:, 0] < g.shape[0] - 1))
    assert np.all((markers[:, 1] > 0) & (markers[:, 1] < g.shape[1] - 1))
    return g, markers


def _ray_grid(shape: tuple[int, int], markers: np.ndarray) -> np.ndarray:
    out = np.full(shape, Color.BLACK, dtype=int)
    for r, c in markers:
        out[int(r), :] = Color.GREY
        out[:, int(c)] = Color.GREY
    return out


def _region_records(ray_grid: np.ndarray) -> list[dict]:
    regions = find_components(
        ray_grid,
        bg=Color.GREY,
        connectivity=4,
        monochromatic=True,
        colors=[Color.BLACK],
        return_format="dict",
    )
    for region in regions:
        r0, c0, r1, c1 = region["bbox"]
        expected_area = (r1 - r0 + 1) * (c1 - c0 + 1)
        assert int(region["area"]) == int(expected_area)
    return regions


def _rank_regions(regions: list[dict]) -> list[dict]:
    return sorted(regions, key=lambda region: (-int(region["area"]), int(region["bbox"][1]), int(region["bbox"][0])))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend orthogonal rays from red markers, then rank the black rectangles they create and recolor the largest ones.

    Input structure:
    - The grid is a black background with 2 or 3 red marker pixels.
    - Every marker lies strictly inside the border, and no two markers share a row or a column.
    - There are no other foreground objects.

    Rule:
    - Replace every marker and every cell in its row and column with grey, extending the rays to the grid edges.
    - Treat the remaining black 4-connected regions as the partition cells; each one must be a rectangle.
    - Rank those rectangles by black-cell area from large to small, breaking ties by the rectangle's top-left position from left to right, then top to bottom.
    - Recolor the largest region blue, the second largest green, and the third largest yellow; keep all other regions black and keep every grey ray cell grey.

    Key insight:
    - The markers matter only because their full horizontal and vertical rays define a rectangular partition, and the final colors depend on the global area ranking of those partition rectangles.
    """
    g, markers = _validate_input(grid)
    out = _ray_grid(g.shape, markers)
    ranked = _rank_regions(_region_records(out))
    for color, region in zip(RANK_COLORS, ranked[: len(RANK_COLORS)]):
        coords = region["coords"]
        out[coords[:, 0], coords[:, 1]] = int(color)
    return out


def _random_segments(total: int, parts: int, min_size: int) -> list[int]:
    extras = [0] * parts
    for _ in range(int(total - parts * min_size)):
        extras[random.randrange(parts)] += 1
    segments = [int(min_size + extra) for extra in extras]
    random.shuffle(segments)
    return segments


def _ordered_segments(parts: int, tie_case: bool, axis: str) -> list[int]:
    low = max(parts * SEGMENT_MIN, GRID_MIN - (parts - 1))
    high = GRID_MAX - (parts - 1)
    for _ in range(GENERATION_ATTEMPTS):
        total_black = random.randint(low, high)
        segments = _random_segments(total_black, parts, SEGMENT_MIN)
        if tie_case and axis == "rows":
            largest = max(segments)
            if segments.count(largest) != 1:
                continue
            rest = [value for value in segments if value != largest]
            return [largest] + sorted(rest, reverse=True)
        if tie_case and axis == "cols":
            largest = max(segments)
            if segments.count(largest) < 2:
                continue
            leading = [largest, largest]
            rest = sorted([value for value in segments if value != largest], reverse=True)
            while len(leading) + len(rest) < parts:
                rest.insert(0, largest)
            return leading + rest[: parts - len(leading)]
        if not tie_case:
            return segments
    raise RuntimeError(f"failed to sample {axis} segments")


def _split_positions(segments: list[int]) -> list[int]:
    positions = [int(segments[0])]
    cursor = int(segments[0])
    for size in segments[1:-1]:
        cursor += 1 + int(size)
        positions.append(int(cursor))
    return positions


def _expected_regions(row_segments: list[int], col_segments: list[int]) -> list[dict]:
    regions: list[dict] = []
    row_start = 0
    for h in row_segments:
        col_start = 0
        for w in col_segments:
            regions.append(
                {
                    "bbox": (int(row_start), int(col_start), int(row_start + h - 1), int(col_start + w - 1)),
                    "area": int(h * w),
                }
            )
            col_start += int(w) + 1
        row_start += int(h) + 1
    return _rank_regions(regions)


def _candidate_grid(marker_count: int, tie_case: bool) -> tuple[np.ndarray, list[int], list[int]]:
    parts = marker_count + 1
    row_segments = _ordered_segments(parts, tie_case=tie_case, axis="rows")
    col_segments = _ordered_segments(parts, tie_case=tie_case, axis="cols")
    row_positions = _split_positions(row_segments)
    col_positions = _split_positions(col_segments)
    grid = np.full((sum(row_segments) + marker_count, sum(col_segments) + marker_count), Color.BLACK, dtype=int)
    for r, c in zip(row_positions, random.sample(col_positions, k=marker_count)):
        grid[int(r), int(c)] = Color.RED
    return grid, row_segments, col_segments


def _check_candidate(grid: np.ndarray, row_segments: list[int], col_segments: list[int], tie_case: bool) -> bool:
    expected = _expected_regions(row_segments, col_segments)
    expected_top3 = expected[:3]
    if tie_case:
        if len(expected) < 2 or int(expected[0]["area"]) != int(expected[1]["area"]):
            return False
        if int(expected[0]["bbox"][0]) != int(expected[1]["bbox"][0]):
            return False
    else:
        top3_areas = [int(region["area"]) for region in expected_top3]
        if len(set(top3_areas)) != 3:
            return False

    out = main(grid)
    markers = np.argwhere(grid == Color.RED)
    grey_mask = _ray_grid(grid.shape, markers) == Color.GREY
    if not np.array_equal(out[grey_mask], np.full(np.count_nonzero(grey_mask), Color.GREY, dtype=int)):
        return False

    expected_out = _ray_grid(grid.shape, markers)
    for color, region in zip(RANK_COLORS, expected_top3):
        r0, c0, r1, c1 = region["bbox"]
        expected_out[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    if not np.array_equal(out, expected_out):
        return False

    if set(map(int, np.unique(out))) != {Color.BLACK, Color.BLUE, Color.GREEN, Color.YELLOW, Color.GREY}:
        return False
    return True


def generate_input() -> np.ndarray:
    marker_count = random.choice(MARKER_COUNTS)
    tie_case = random.random() < TIE_CASE_PROBABILITY
    for _ in range(GENERATION_ATTEMPTS):
        grid, row_segments, col_segments = _candidate_grid(marker_count, tie_case)
        if _check_candidate(grid, row_segments, col_segments, tie_case):
            return grid
    raise RuntimeError("failed to generate a valid ray-partition ranking input")
