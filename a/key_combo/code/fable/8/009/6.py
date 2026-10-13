from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path
import random

import numpy as np


try:
    from arc_utils import Color, bbox_h, bbox_w, find_components
except Exception:
    _ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    if _ARC_UTILS_PATH.exists():
        _arc_utils = SourceFileLoader("_arc_utils_task_fable_8_009_6", str(_ARC_UTILS_PATH)).load_module()
        Color = _arc_utils.Color
        bbox_h = _arc_utils.bbox_h
        bbox_w = _arc_utils.bbox_w
        find_components = _arc_utils.find_components
    else:
        class Color:
            BLACK = 0
            BLUE = 1
            RED = 2
            GREEN = 3
            YELLOW = 4
            GREY = 5
            GRAY = 5
            PINK = 6
            ORANGE = 7
            TEAL = 8
            MAROON = 9
            NOT_BLACK = [1, 2, 3, 4, 5, 6, 7, 8, 9]

        def bbox_h(bbox: tuple[int, int, int, int]) -> int:
            return int(bbox[2] - bbox[0] + 1)

        def bbox_w(bbox: tuple[int, int, int, int]) -> int:
            return int(bbox[3] - bbox[1] + 1)

        def find_components(
            grid: np.ndarray,
            *,
            bg: int | None = None,
            connectivity: int = 4,
            monochromatic: bool = True,
            colors: list[int] | None = None,
            min_area: int = 1,
            return_format: str = "dict",
        ) -> list[dict]:
            if return_format != "dict":
                raise ValueError("fallback find_components only supports return_format='dict'")
            if bg is None:
                vals, counts = np.unique(grid, return_counts=True)
                bg = int(vals[np.argmax(counts)])
            h, w = grid.shape
            if connectivity == 4:
                deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            else:
                deltas = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
            allowed = None if colors is None else {int(c) for c in colors}
            seen = np.zeros((h, w), dtype=bool)
            comps: list[dict] = []
            for r in range(h):
                for c in range(w):
                    col = int(grid[r, c])
                    if seen[r, c] or col == int(bg):
                        continue
                    if allowed is not None and col not in allowed:
                        continue
                    queue = [(r, c)]
                    seen[r, c] = True
                    coords = []
                    while queue:
                        cr, cc = queue.pop()
                        coords.append((cr, cc))
                        for dr, dc in deltas:
                            nr, nc = cr + dr, cc + dc
                            if not (0 <= nr < h and 0 <= nc < w) or seen[nr, nc]:
                                continue
                            same = int(grid[nr, nc]) == col if monochromatic else int(grid[nr, nc]) != int(bg)
                            if not same:
                                continue
                            if allowed is not None and int(grid[nr, nc]) not in allowed and monochromatic:
                                continue
                            seen[nr, nc] = True
                            queue.append((nr, nc))
                    arr = np.array(coords, dtype=int)
                    if arr.shape[0] < int(min_area):
                        continue
                    r0, c0 = arr.min(axis=0)
                    r1, c1 = arr.max(axis=0)
                    comps.append(
                        {
                            "color": col if monochromatic else None,
                            "coords": arr,
                            "bbox": (int(r0), int(c0), int(r1), int(c1)),
                            "area": int(arr.shape[0]),
                        }
                    )
            return comps


DIAGONALS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
GEN_RETRIES = 500


def _trace_bounce_path(
    shape: tuple[int, int],
    start: tuple[int, int],
    direction: tuple[int, int],
    segments: int,
) -> list[tuple[int, int]] | None:
    h, w = map(int, shape)
    r, c = map(int, start)
    dr, dc = map(int, direction)
    hits = 0
    cells: list[tuple[int, int]] = []
    for _ in range(4 * h * w + 10):
        cells.append((r, c))
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            hit_r = not (0 <= nr < h)
            hit_c = not (0 <= nc < w)
            if hit_r and hit_c:
                return None
            hits += 1
            if hits == int(segments):
                return cells
            if hit_r:
                dr = -dr
            if hit_c:
                dc = -dc
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                return None
        r, c = nr, nc
    return None


def _l_cells_from_missing(start: tuple[int, int], direction: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = map(int, start)
    dr, dc = map(int, direction)
    r0 = r if dr == -1 else r - 1
    c0 = c if dc == -1 else c - 1
    cells = {(r0 + rr, c0 + cc) for rr in range(2) for cc in range(2)}
    cells.remove((r, c))
    return cells


def _extract_emitters(grid: np.ndarray) -> list[dict]:
    comps = find_components(grid, bg=Color.BLACK, connectivity=4, monochromatic=True, return_format="dict")
    by_color: dict[int, list[dict]] = {}
    for comp in comps:
        color = int(comp["color"])
        by_color.setdefault(color, []).append(comp)

    emitters: list[dict] = []
    for color, color_comps in by_color.items():
        l_shapes = []
        dots = []
        extras = []
        for comp in color_comps:
            if int(comp["area"]) == 3 and bbox_h(comp["bbox"]) == 2 and bbox_w(comp["bbox"]) == 2:
                l_shapes.append(comp)
            elif int(comp["area"]) == 1:
                dots.append(comp)
            else:
                extras.append(comp)
        assert not extras, f"unexpected component shape for color {color}"
        assert len(l_shapes) <= 1, f"expected at most one L-shape for color {color}"
        if not l_shapes:
            assert not dots, f"dots without emitter for color {color}"
            continue
        l_shape = l_shapes[0]
        r0, c0, r1, c1 = map(int, l_shape["bbox"])
        occupied = {tuple(map(int, coord)) for coord in l_shape["coords"]}
        missing = [cell for cell in ((r0, c0), (r0, c1), (r1, c0), (r1, c1)) if cell not in occupied]
        assert len(missing) == 1, "2x2 L-shape must have exactly one missing corner"
        mr, mc = missing[0]
        direction = (-1 if mr == r0 else 1, -1 if mc == c0 else 1)
        emitters.append(
            {
                "color": color,
                "start": (mr, mc),
                "direction": direction,
                "segments": len(dots),
                "l_cells": occupied,
                "dot_cells": {tuple(map(int, comp["coords"][0])) for comp in dots},
            }
        )

    assert 1 <= len(emitters) <= 2, "expected one or two emitters"
    for emitter in emitters:
        assert emitter["segments"] >= 1, "each emitter needs at least one same-color dot"
    return sorted(emitters, key=lambda item: item["color"])


def _render_paths(grid: np.ndarray, emitters: list[dict], *, force_segments: dict[int, int] | None = None) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for emitter in emitters:
        color = int(emitter["color"])
        count = int(emitter["segments"] if force_segments is None else force_segments[color])
        path = _trace_bounce_path(out.shape, emitter["start"], emitter["direction"], count)
        assert path is not None, "path must avoid corner impacts"
        for r, c in path:
            assert int(out[r, c]) in {Color.BLACK, color}, "path collides with another color"
            out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire a diagonal billiard path from each missing-corner L-block for as many segments as its same-color dots count.

    Input structure:
    - A black background contains one or two colors, each with exactly one 2x2 L-shape made of three cells.
    - The missing corner of each 2x2 box is empty, and the same color also appears as isolated single-pixel dots elsewhere.
    - Dots do not touch any L-shape or one another, and valid inputs keep the future billiard paths clear of all existing colored cells.

    Rule:
    - For each color, identify the empty corner inside its 2x2 L-shape and use the diagonal from the box center toward that empty corner as the launch direction.
    - Start from that empty-corner cell, paint every visited cell in that color, and reflect diagonally whenever the next step would leave the grid.
    - Let n be the number of isolated dots of the same color: the path continues for exactly n straight segments, so the first n-1 wall hits bounce and the nth wall hit stops.

    Key insight:
    - The result needs three coupled cues at once: the defect corner sets the diagonal, the border acts like a mirror instead of an endpoint, and the global same-color dot count determines how many reflected segments are drawn.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    return _render_paths(g, _extract_emitters(g))


def _chebyshev_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _cells_are_clear(cells: set[tuple[int, int]], protected: set[tuple[int, int]], margin: int) -> bool:
    return all(_chebyshev_distance(cell, other) > margin for cell in cells for other in protected)


def _path_has_unique_direction(shape: tuple[int, int], start: tuple[int, int], direction: tuple[int, int], segments: int) -> bool:
    actual = _trace_bounce_path(shape, start, direction, segments)
    if actual is None:
        return False
    for alt in DIAGONALS:
        if alt == direction:
            continue
        other = _trace_bounce_path(shape, start, alt, segments)
        if other == actual:
            return False
    return True


def _sample_counts(num_emitters: int) -> list[int]:
    if num_emitters == 1:
        return [random.choice([2, 3, 4])]
    counts = random.sample([1, 2, 3, 4], 2)
    if counts[0] == counts[1]:
        counts[1] = 5 - counts[0]
    if max(counts) == 1:
        counts[random.randrange(2)] = random.choice([2, 3, 4])
    return counts


def _try_place_emitter(
    shape: tuple[int, int],
    segments: int,
    blocked: set[tuple[int, int]],
    protected_objects: set[tuple[int, int]],
    protected_paths: set[tuple[int, int]],
) -> dict | None:
    h, w = map(int, shape)
    candidates = [
        ((r, c), direction)
        for r in range(1, h - 1)
        for c in range(1, w - 1)
        for direction in DIAGONALS
    ]
    random.shuffle(candidates)
    for start, direction in candidates:
        l_cells = _l_cells_from_missing(start, direction)
        path = _trace_bounce_path(shape, start, direction, segments)
        if path is None or len(path) < 2:
            continue
        path_set = set(path)
        object_cells = set(l_cells) | {start}
        if object_cells & blocked:
            continue
        if path_set & blocked:
            continue
        if not _cells_are_clear(set(l_cells), protected_objects, margin=1):
            continue
        if not _cells_are_clear(path_set, protected_objects, margin=1):
            continue
        if not _cells_are_clear(path_set, protected_paths, margin=1):
            continue
        if not _path_has_unique_direction(shape, start, direction, segments):
            continue
        if segments > 1 and path == _trace_bounce_path(shape, start, direction, 1):
            continue
        return {
            "start": start,
            "direction": direction,
            "segments": segments,
            "l_cells": l_cells,
            "path_cells": path,
        }
    return None


def _dot_candidates(
    shape: tuple[int, int],
    forbidden: set[tuple[int, int]],
    protected: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    candidates = []
    for r in range(h):
        for c in range(w):
            cell = (r, c)
            if cell in forbidden:
                continue
            if any(_chebyshev_distance(cell, other) <= 1 for other in protected):
                continue
            candidates.append(cell)
    random.shuffle(candidates)
    return candidates


def _place_dots(
    shape: tuple[int, int],
    emitters: list[dict],
) -> bool:
    all_paths = {cell for emitter in emitters for cell in emitter["path_cells"]}
    all_l_cells = {cell for emitter in emitters for cell in emitter["l_cells"]}
    used = set(all_paths) | all_l_cells
    protected = set(all_paths) | all_l_cells
    for emitter in emitters:
        dots: set[tuple[int, int]] = set()
        for cell in _dot_candidates(shape, used, protected):
            if any(_chebyshev_distance(cell, dot) <= 1 for dot in dots):
                continue
            dots.add(cell)
            used.add(cell)
            protected.add(cell)
            if len(dots) == int(emitter["segments"]):
                break
        if len(dots) != int(emitter["segments"]):
            return False
        emitter["dot_cells"] = dots
    return True


def _build_input(shape: tuple[int, int], emitters: list[dict], colors: list[int]) -> np.ndarray:
    grid = np.full(shape, Color.BLACK, dtype=int)
    for emitter, color in zip(emitters, colors):
        emitter["color"] = int(color)
        for r, c in emitter["l_cells"]:
            grid[r, c] = int(color)
        for r, c in emitter["dot_cells"]:
            grid[r, c] = int(color)
    return grid


def _wrong_first_hit_output(grid: np.ndarray, emitters: list[dict]) -> np.ndarray:
    return _render_paths(grid, emitters, force_segments={int(em["color"]): 1 for em in emitters})


def _validate_semantics(grid: np.ndarray, emitters: list[dict]) -> bool:
    expected = _render_paths(grid, emitters)
    try:
        actual = main(grid)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(actual, grid):
        return False
    if np.array_equal(actual, _wrong_first_hit_output(grid, emitters)):
        return False
    for emitter in emitters:
        count1 = _trace_bounce_path(grid.shape, emitter["start"], emitter["direction"], 1)
        countn = _trace_bounce_path(grid.shape, emitter["start"], emitter["direction"], emitter["segments"])
        if emitter["segments"] > 1 and count1 == countn:
            return False
        if not _path_has_unique_direction(grid.shape, emitter["start"], emitter["direction"], emitter["segments"]):
            return False
        if not emitter["dot_cells"] or len(emitter["dot_cells"]) != int(emitter["segments"]):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(14, 22)
    w = random.randint(14, 22)
    num_emitters = random.choice([1, 2, 2])
    counts = _sample_counts(num_emitters)
    emitters: list[dict] = []
    blocked: set[tuple[int, int]] = set()
    protected_objects: set[tuple[int, int]] = set()
    protected_paths: set[tuple[int, int]] = set()
    for segments in counts:
        placed = _try_place_emitter((h, w), segments, blocked, protected_objects, protected_paths)
        if placed is None:
            return None
        emitters.append(placed)
        blocked |= set(placed["l_cells"]) | set(placed["path_cells"]) | {placed["start"]}
        protected_objects |= set(placed["l_cells"]) | {placed["start"]}
        protected_paths |= set(placed["path_cells"])
    if not any(emitter["segments"] > 1 for emitter in emitters):
        return None
    if not _place_dots((h, w), emitters):
        return None
    colors = random.sample([c for c in Color.NOT_BLACK if c != Color.GRAY], len(emitters))
    grid = _build_input((h, w), emitters, colors)
    if len({tuple(sorted(emitter["dot_cells"])) for emitter in emitters}) != len(emitters):
        return None
    if not _validate_semantics(grid, emitters):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid residual-corner billiard input")
