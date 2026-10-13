from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]

GRID_MIN = 18
GRID_MAX = 24
MAX_GEN_ATTEMPTS = 400
MAX_GROUP_LAYOUT_ATTEMPTS = 160
MAX_SEGMENT_LAYOUT_ATTEMPTS = 220


def _other(orientation: str) -> str:
    return "v" if orientation == "h" else "h"


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _segment_from_cells(cells: list[tuple[int, int]]) -> dict[str, object]:
    rows = sorted({r for r, _ in cells})
    cols = sorted({c for _, c in cells})
    if len(rows) == 1 and cols == list(range(cols[0], cols[-1] + 1)):
        row = int(rows[0])
        c0 = int(cols[0])
        c1 = int(cols[-1])
        return {
            "orientation": "h",
            "length": int(len(cols)),
            "cells": [(row, int(c)) for c in cols],
            "row": row,
            "c0": c0,
            "c1": c1,
        }
    if len(cols) == 1 and rows == list(range(rows[0], rows[-1] + 1)):
        col = int(cols[0])
        r0 = int(rows[0])
        r1 = int(rows[-1])
        return {
            "orientation": "v",
            "length": int(len(rows)),
            "cells": [(int(r), col) for r in rows],
            "col": col,
            "r0": r0,
            "r1": r1,
        }
    raise AssertionError("every object must be a contiguous horizontal or vertical segment")


def _segment_cells(start: tuple[int, int], length: int, orientation: str) -> list[tuple[int, int]]:
    r, c = map(int, start)
    if orientation == "h":
        return [(r, int(c + offset)) for offset in range(length)]
    return [(int(r + offset), c) for offset in range(length)]


def _next_cursor(start: tuple[int, int], length: int, orientation: str) -> tuple[int, int]:
    r, c = map(int, start)
    if orientation == "h":
        return (r, int(c + length))
    return (int(r + length), c)


def _in_bounds(cells: list[tuple[int, int]], h: int, w: int) -> bool:
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _build_output_segments(
    start: tuple[int, int], lengths: list[int], base_orientation: str, h: int, w: int
) -> list[dict[str, object]] | None:
    segments: list[dict[str, object]] = []
    cursor = tuple(map(int, start))
    for i, length in enumerate(lengths):
        orientation = base_orientation if i % 2 == 0 else _other(base_orientation)
        cells = _segment_cells(cursor, int(length), orientation)
        if not _in_bounds(cells, h, w):
            return None
        segments.append(
            {
                "orientation": orientation,
                "length": int(length),
                "start": cursor,
                "cells": cells,
            }
        )
        cursor = _next_cursor(cursor, int(length), orientation)
    return segments


def _path_extent(lengths: list[int], base_orientation: str) -> tuple[int, int]:
    height = 0
    width = 0
    for i, length in enumerate(lengths):
        orientation = base_orientation if i % 2 == 0 else _other(base_orientation)
        if orientation == "h":
            width += int(length)
        else:
            height += int(length)
    return max(1, height), max(1, width)


def _mark_buffer(mask: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = mask.shape
    for r, c in cells:
        for dr, dc in DIR8:
            nr = int(r + dr)
            nc = int(c + dc)
            if 0 <= nr < h and 0 <= nc < w:
                mask[nr, nc] = True


def _can_place(cells: list[tuple[int, int]], *blocked_masks: np.ndarray) -> bool:
    for r, c in cells:
        for mask in blocked_masks:
            if bool(mask[r, c]):
                return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble each color's scattered line segments into a single alternating polyline.

    Segments are grouped by color, sorted from shortest to longest, then laid out from the
    shortest segment's original position. The first segment keeps its original orientation and
    later segments alternate between horizontal and vertical, rotating when needed.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)

    by_color: dict[int, list[dict[str, object]]] = defaultdict(list)
    for color in map(int, np.unique(g)):
        if color == bg:
            continue
        for comp in _components(g == color):
            by_color[color].append(_segment_from_cells(comp))

    out = np.full_like(g, bg)
    for color, segments in by_color.items():
        ordered = sorted(segments, key=lambda seg: int(seg["length"]))
        lengths = [int(seg["length"]) for seg in ordered]
        assert len(lengths) == len(set(lengths)), "same-color segment lengths must be distinct"

        first = ordered[0]
        for r, c in first["cells"]:
            out[r, c] = int(color)

        base_orientation = str(first["orientation"])
        if base_orientation == "h":
            cursor = (int(first["row"]), int(first["c1"]) + 1)
        else:
            cursor = (int(first["r1"]) + 1, int(first["col"]))

        for i, seg in enumerate(ordered[1:], start=1):
            needed = base_orientation if i % 2 == 0 else _other(base_orientation)
            cells = _segment_cells(cursor, int(seg["length"]), needed)
            assert _in_bounds(cells, h, w), "assembled path must stay inside the grid"
            for r, c in cells:
                assert int(out[r, c]) == int(bg), "assembled paths must not overlap"
                out[r, c] = int(color)
            cursor = _next_cursor(cursor, int(seg["length"]), needed)

    return out


def _sample_group_specs(h: int, w: int) -> list[dict[str, object]] | None:
    bg = int(random.randrange(10))
    group_count = int(random.choice([2, 3]))
    palette = [color for color in range(10) if color != bg]
    colors = random.sample(palette, group_count)

    base_orientations = [random.choice(["h", "v"]) for _ in range(group_count)]
    if group_count > 1 and len(set(base_orientations)) == 1:
        idx = random.randrange(group_count)
        base_orientations[idx] = _other(base_orientations[idx])

    specs: list[dict[str, object]] = []
    for color, base_orientation in zip(colors, base_orientations):
        count = int(random.choice([2, 3, 4]))
        lengths = None
        for _ in range(40):
            pool_hi = int(random.randint(6, min(8, min(h, w) - 6)))
            pool = list(range(2, pool_hi + 1))
            if len(pool) < count:
                continue
            trial = sorted(random.sample(pool, count))
            span_h, span_w = _path_extent(trial, base_orientation)
            if span_h <= h and span_w <= w:
                lengths = trial
                break
        if lengths is None:
            return None
        span_h, span_w = _path_extent(lengths, base_orientation)
        specs.append(
            {
                "color": int(color),
                "base_orientation": str(base_orientation),
                "lengths": [int(length) for length in lengths],
                "bbox_h": int(span_h),
                "bbox_w": int(span_w),
            }
        )
    return [{"bg": bg}] + specs


def _layout_output_paths(h: int, w: int, specs: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray] | None:
    bg = int(specs[0]["bg"])
    expected = np.full((h, w), bg, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    groups = sorted(specs[1:], key=lambda spec: (int(spec["bbox_h"]) * int(spec["bbox_w"]), sum(spec["lengths"])), reverse=True)
    for spec in groups:
        placed = False
        starts: list[tuple[int, int]] = []
        max_r = h - int(spec["bbox_h"])
        max_c = w - int(spec["bbox_w"])
        for _ in range(MAX_GROUP_LAYOUT_ATTEMPTS):
            starts.append((int(random.randint(0, max_r)), int(random.randint(0, max_c))))
        starts.extend((r, c) for r in range(max_r + 1) for c in range(max_c + 1))
        random.shuffle(starts)

        for start in starts:
            segments = _build_output_segments(start, spec["lengths"], str(spec["base_orientation"]), h, w)
            if segments is None:
                continue
            cells = [cell for segment in segments for cell in segment["cells"]]
            if not _can_place(cells, blocked):
                continue
            for r, c in cells:
                expected[r, c] = int(spec["color"])
            _mark_buffer(blocked, cells)
            spec["output_segments"] = segments
            placed = True
            break
        if not placed:
            return None

    output_buffer = blocked.copy()
    return expected, output_buffer


def _assign_rotation_cases(specs: list[dict[str, object]]) -> list[dict[str, object]]:
    tasks: list[dict[str, object]] = []
    for spec in specs[1:]:
        for index in range(1, len(spec["lengths"])):
            required = str(spec["output_segments"][index]["orientation"])
            tasks.append(
                {
                    "color": int(spec["color"]),
                    "length": int(spec["lengths"][index]),
                    "required": required,
                }
            )

    if len(tasks) >= 2:
        rotate_flags = [bool(random.getrandbits(1)) for _ in tasks]
        if len(set(rotate_flags)) == 1:
            rotate_flags[random.randrange(len(rotate_flags))] = not rotate_flags[0]
    else:
        rotate_flags = [bool(random.getrandbits(1)) for _ in tasks]

    for task, rotate in zip(tasks, rotate_flags):
        task["input_orientation"] = _other(str(task["required"])) if rotate else str(task["required"])
    random.shuffle(tasks)
    tasks.sort(key=lambda task: int(task["length"]), reverse=True)
    return tasks


def _place_scattered_segment(
    color: int,
    length: int,
    orientation: str,
    grid: np.ndarray,
    input_blocked: np.ndarray,
    output_blocked: np.ndarray,
) -> bool:
    h, w = grid.shape
    candidates: list[tuple[int, int]] = []
    if orientation == "h":
        for _ in range(MAX_SEGMENT_LAYOUT_ATTEMPTS):
            candidates.append((int(random.randrange(h)), int(random.randrange(w - length + 1))))
        candidates.extend((r, c) for r in range(h) for c in range(w - length + 1))
    else:
        for _ in range(MAX_SEGMENT_LAYOUT_ATTEMPTS):
            candidates.append((int(random.randrange(h - length + 1)), int(random.randrange(w))))
        candidates.extend((r, c) for r in range(h - length + 1) for c in range(w))
    random.shuffle(candidates)

    for start in candidates:
        cells = _segment_cells(start, length, orientation)
        if not _can_place(cells, input_blocked, output_blocked):
            continue
        for r, c in cells:
            grid[r, c] = int(color)
        _mark_buffer(input_blocked, cells)
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    h = int(random.randint(GRID_MIN, GRID_MAX))
    w = int(random.randint(GRID_MIN, GRID_MAX))

    specs = _sample_group_specs(h, w)
    if specs is None:
        return None
    bg = int(specs[0]["bg"])

    laid_out = _layout_output_paths(h, w, specs)
    if laid_out is None:
        return None
    expected, output_blocked = laid_out

    inp = np.full((h, w), bg, dtype=int)
    input_blocked = np.zeros((h, w), dtype=bool)

    for spec in specs[1:]:
        first = spec["output_segments"][0]
        for r, c in first["cells"]:
            inp[r, c] = int(spec["color"])
        _mark_buffer(input_blocked, first["cells"])

    tasks = _assign_rotation_cases(specs)
    for task in tasks:
        ok = _place_scattered_segment(
            int(task["color"]),
            int(task["length"]),
            str(task["input_orientation"]),
            inp,
            input_blocked,
            output_blocked,
        )
        if not ok:
            return None

    if int(np.count_nonzero(inp != bg)) >= (h * w) // 2:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
