from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5

MIN_RECT_DIM = 3
MIN_GAP = 3
GRID_MIN = 18
GRID_MAX = 22
RECT_COUNT_OPTIONS = (2, 3, 4)
RECT_SIZE_OPTIONS = (3, 4, 5, 6)
MAX_GENERATION_ATTEMPTS = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _extract_rectangles(grid: np.ndarray) -> list[dict[str, object]]:
    components = _find_components(grid == GREY)
    assert components, "expected at least one grey rectangle"
    rectangles: list[dict[str, object]] = []
    for coords in components:
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h >= MIN_RECT_DIM and w >= MIN_RECT_DIM, "rectangles must be at least 3x3"
        assert int(coords.shape[0]) == h * w, "grey component must fill its bounding box"
        assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == GREY), "grey component must be solid"
        rectangles.append(
            {
                "bbox": (r0, c0, r1, c1),
                "h": h,
                "w": w,
            }
        )
    return rectangles


def _classify_rectangles(shape: tuple[int, int], rectangles: Iterable[dict[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    out = np.full(shape, BLACK, dtype=int)
    rect_mask = np.zeros(shape, dtype=bool)
    for rect in rectangles:
        r0, c0, r1, c1 = rect["bbox"]  # type: ignore[index]
        rect_mask[r0 : r1 + 1, c0 : c1 + 1] = True
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                on_h = r in (r0, r1)
                on_v = c in (c0, c1)
                if on_h and on_v:
                    out[r, c] = BLUE
                elif on_h or on_v:
                    out[r, c] = YELLOW
                else:
                    out[r, c] = RED
    return out, rect_mask


def _ray_sources(rect: dict[str, object]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    r0, c0, r1, c1 = rect["bbox"]  # type: ignore[index]
    return [
        ((r0, c0), (-1, -1)),
        ((r0, c1), (-1, 1)),
        ((r1, c0), (1, -1)),
        ((r1, c1), (1, 1)),
    ]


def _trace_rays(rectangles: list[dict[str, object]], shape: tuple[int, int]) -> list[dict[str, object]]:
    classified, rect_mask = _classify_rectangles(shape, rectangles)
    traces: list[dict[str, object]] = []
    for rect_index, rect in enumerate(rectangles):
        for source, direction in _ray_sources(rect):
            path: list[tuple[int, int]] = []
            r, c = source
            dr, dc = direction
            hit_rect = None
            hit_cell = None
            while True:
                nr = r + dr
                nc = c + dc
                if not _in_bounds(shape, nr, nc):
                    stop_type = "boundary"
                    break
                if rect_mask[nr, nc]:
                    stop_type = "hit"
                    hit_cell = (nr, nc)
                    for other_index, other in enumerate(rectangles):
                        rr0, cc0, rr1, cc1 = other["bbox"]  # type: ignore[index]
                        if rr0 <= nr <= rr1 and cc0 <= nc <= cc1:
                            hit_rect = other_index
                            break
                    assert hit_rect is not None
                    break
                assert classified[nr, nc] == BLACK
                path.append((nr, nc))
                r, c = nr, nc
            traces.append(
                {
                    "source_rect": rect_index,
                    "source": source,
                    "direction": direction,
                    "path": path,
                    "stop_type": stop_type,
                    "hit_rect": hit_rect,
                    "hit_cell": hit_cell,
                }
            )
    return traces


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each gray rectangle by local role, then emit yellow diagonal rays from its corners.

    Input structure:
    - The grid has black background cells and 2 to 4 solid gray rectangles.
    - Every gray component is a 4-connected filled rectangle with height and width at least 3.
    - Rectangles are separated from one another, so the diagonal rays travel only through background before a stop.

    Rule:
    - Recolor each rectangle independently: four corners become blue, the remaining border becomes yellow, and the strict interior becomes red.
    - From each blue corner, launch one ray in the corner's outward diagonal direction: top-left to up-left, top-right to up-right, bottom-left to down-left, bottom-right to down-right.
    - As a ray advances one step at a time, recolor every visited black background cell yellow.
    - A ray stops before its next step if that next cell is any classified rectangle cell, or if the next step would leave the grid.
    - Rays are evaluated against the rectangle classification only; yellow cells added by one ray do not block another ray.

    Key insight:
    - The corner classification is not decorative: it determines both where rays start and which diagonal direction each ray must follow, while the full classified rectangles define the collision mask that stops those rays.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "input must be a non-empty 2D grid"
    assert set(map(int, np.unique(g))) <= {BLACK, GREY}, "input may contain only black and grey"

    rectangles = _extract_rectangles(g)
    out, rect_mask = _classify_rectangles(g.shape, rectangles)

    for rect in rectangles:
        for source, direction in _ray_sources(rect):
            r, c = source
            dr, dc = direction
            while True:
                nr = r + dr
                nc = c + dc
                if not _in_bounds(g.shape, nr, nc):
                    break
                if rect_mask[nr, nc]:
                    break
                assert g[nr, nc] == BLACK, "rays may only traverse background in the input"
                out[nr, nc] = YELLOW
                r, c = nr, nc

    return out


def _random_composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [int(total)]
    return list(map(int, np.random.multinomial(int(total), np.ones(parts) / float(parts))))


def _sample_dims(rect_count: int, limit: int) -> list[int] | None:
    for _ in range(80):
        dims = [int(random.choice(RECT_SIZE_OPTIONS)) for _ in range(rect_count)]
        if sum(dims) + MIN_GAP * (rect_count - 1) <= limit:
            return dims
    return None


def _build_axis_positions(length: int, dims: list[int]) -> tuple[list[int], list[int]] | None:
    required = sum(dims) + MIN_GAP * (len(dims) - 1)
    if required > length:
        return None
    extras = _random_composition(length - required, len(dims) + 1)
    starts: list[int] = []
    pos = extras[0]
    for idx, dim in enumerate(dims):
        starts.append(int(pos))
        pos += int(dim)
        if idx + 1 < len(dims):
            pos += MIN_GAP + int(extras[idx + 1])
    gaps = [MIN_GAP + int(extras[idx + 1]) for idx in range(len(dims) - 1)]
    return starts, gaps


def _build_col_positions(
    width: int,
    dims: list[int],
    direction: int,
    target_gap: int,
    target_index: int,
) -> tuple[list[int], list[int]] | None:
    required = sum(dims) + MIN_GAP * (len(dims) - 1)
    if required > width:
        return None
    free = width - required
    fixed_extra = target_gap - MIN_GAP
    if fixed_extra < 0 or fixed_extra > free:
        return None

    if len(dims) == 1:
        extras = [free, 0]
    else:
        extras = [0] * (len(dims) + 1)
        extras[target_index + 1] = fixed_extra
        remainder = free - fixed_extra
        rem_parts = _random_composition(remainder, len(dims))
        write = 0
        for idx in range(len(extras)):
            if idx == target_index + 1:
                continue
            extras[idx] = int(rem_parts[write])
            write += 1

    starts_inc: list[int] = []
    pos = extras[0]
    for idx, dim in enumerate(dims):
        starts_inc.append(int(pos))
        pos += int(dim)
        if idx + 1 < len(dims):
            pos += MIN_GAP + int(extras[idx + 1])
    gaps = [MIN_GAP + int(extras[idx + 1]) for idx in range(len(dims) - 1)]

    if direction == 1:
        return starts_inc, gaps

    starts_dec = [width - starts_inc[idx] - dims[idx] for idx in range(len(dims))]
    return starts_dec, gaps


def _render_input(shape: tuple[int, int], rects: list[tuple[int, int, int, int]]) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for r0, c0, h, w in rects:
        grid[r0 : r0 + h, c0 : c0 + w] = GREY
    return grid


def _pairwise_layout_checks(rects: list[tuple[int, int, int, int]]) -> bool:
    for i, (r0a, c0a, ha, wa) in enumerate(rects):
        r1a = r0a + ha - 1
        c1a = c0a + wa - 1
        for r0b, c0b, hb, wb in rects[i + 1 :]:
            r1b = r0b + hb - 1
            c1b = c0b + wb - 1
            row_gap = min(abs(r0b - r1a) - 1, abs(r0a - r1b) - 1)
            col_gap = min(abs(c0b - c1a) - 1, abs(c0a - c1b) - 1)
            if row_gap < MIN_GAP or col_gap < MIN_GAP:
                return False
            rows_disjoint = r1a < r0b or r1b < r0a
            cols_disjoint = c1a < c0b or c1b < c0a
            if not (rows_disjoint and cols_disjoint):
                return False
    return True


def _semantic_output_checks(inp: np.ndarray, out: np.ndarray) -> bool:
    rectangles = _extract_rectangles(inp)
    classified, rect_mask = _classify_rectangles(inp.shape, rectangles)
    traces = _trace_rays(rectangles, inp.shape)

    expected_yellow = np.zeros(inp.shape, dtype=bool)
    for trace in traces:
        for r, c in trace["path"]:  # type: ignore[assignment]
            expected_yellow[r, c] = True
        if trace["stop_type"] == "hit":
            hr, hc = trace["hit_cell"]  # type: ignore[misc]
            if expected_yellow[hr, hc]:
                return False
            if out[hr, hc] == YELLOW and not rect_mask[hr, hc]:
                return False

    for r in range(inp.shape[0]):
        for c in range(inp.shape[1]):
            if rect_mask[r, c]:
                if out[r, c] != classified[r, c]:
                    return False
            elif expected_yellow[r, c]:
                if inp[r, c] != BLACK or out[r, c] != YELLOW:
                    return False
            else:
                if inp[r, c] != BLACK or out[r, c] != BLACK:
                    return False

    hit_count = sum(trace["stop_type"] == "hit" for trace in traces)
    boundary_count = sum(trace["stop_type"] == "boundary" for trace in traces)
    visible_lengths = [len(trace["path"]) for trace in traces]
    if hit_count < 1 or boundary_count < 1:
        return False
    if sum(visible_lengths) < 8:
        return False
    if sum(length >= 2 for length in visible_lengths) < 3:
        return False
    return True


def _sample_layout() -> np.ndarray | None:
    height = int(np.random.randint(GRID_MIN, GRID_MAX + 1))
    width = int(np.random.randint(GRID_MIN, GRID_MAX + 1))
    rect_count = int(random.choice(RECT_COUNT_OPTIONS))
    direction = random.choice((1, -1))

    heights = _sample_dims(rect_count, height)
    widths = _sample_dims(rect_count, width)
    if heights is None or widths is None:
        return None

    row_build = _build_axis_positions(height, heights)
    if row_build is None:
        return None
    row_starts, row_gaps = row_build
    if not row_gaps:
        return None

    link_index = int(np.random.randint(0, len(row_gaps)))
    col_build = _build_col_positions(width, widths, direction, row_gaps[link_index], link_index)
    if col_build is None:
        return None
    col_starts, _col_gaps = col_build

    rects = [(row_starts[i], col_starts[i], heights[i], widths[i]) for i in range(rect_count)]
    if not _pairwise_layout_checks(rects):
        return None

    grid = _render_input((height, width), rects)
    rectangles = _extract_rectangles(grid)
    traces = _trace_rays(rectangles, grid.shape)

    hit_rays = [trace for trace in traces if trace["stop_type"] == "hit"]
    boundary_rays = [trace for trace in traces if trace["stop_type"] == "boundary"]
    if not hit_rays or not boundary_rays:
        return None

    unique_hit_targets = {
        (trace["source_rect"], trace["hit_rect"], trace["direction"], trace["hit_cell"])
        for trace in hit_rays
    }
    if len(unique_hit_targets) < 1:
        return None

    path_fingerprints = [tuple(trace["path"]) for trace in traces]
    non_empty = [fp for fp in path_fingerprints if fp]
    if len(non_empty) != len(set(non_empty)):
        return None

    out = main(grid)
    if not _semantic_output_checks(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _sample_layout()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 031/1")
