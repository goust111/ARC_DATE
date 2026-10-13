from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np


BG = 0
MAROON = 9
PALETTE = (1, 2, 3, 4, 5, 6, 7, 8)
DIR_TO_DELTA = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
LEFT_TURN = {"U": "L", "L": "D", "D": "R", "R": "U"}
RIGHT_TURN = {"U": "R", "R": "D", "D": "L", "L": "U"}
GEN_RETRIES = 400


def _neighbors4(r: int, c: int, h: int, w: int) -> Iterable[tuple[int, int]]:
    if r > 0:
        yield (r - 1, c)
    if r + 1 < h:
        yield (r + 1, c)
    if c > 0:
        yield (r, c - 1)
    if c + 1 < w:
        yield (r, c + 1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _inside_bbox(pos: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r, c = pos
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def _line_cells_and_next_cursor(
    cursor: tuple[int, int],
    bbox: tuple[int, int, int, int],
    direction: str,
    length: int,
) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    r0, c0, r1, c1 = bbox
    r, c = cursor
    dr, dc = DIR_TO_DELTA[direction]
    end_r = r + dr * (length - 1)
    end_c = c + dc * (length - 1)
    if not (r0 <= end_r <= r1 and c0 <= end_c <= c1):
        return None

    cells = [(r + dr * k, c + dc * k) for k in range(length)]
    next_cursor = (end_r + dr, end_c + dc)
    if _inside_bbox(next_cursor, bbox):
        return cells, next_cursor

    if direction == "R":
        next_cursor = (end_r + 1, c0)
    elif direction == "L":
        next_cursor = (end_r + 1, c1)
    elif direction == "D":
        next_cursor = (r0, end_c + 1)
    else:
        next_cursor = (r1, end_c + 1)

    if not _inside_bbox(next_cursor, bbox):
        return None
    return cells, next_cursor


def _canvas_rectangles(grid: np.ndarray) -> dict[int, tuple[int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    canvases: dict[int, tuple[int, int, int, int]] = {}
    for color in sorted(int(v) for v in np.unique(g) if int(v) != BG):
        rects: list[tuple[int, int, int, int]] = []
        for comp in _components(g == color):
            r0, c0, r1, c1 = _bbox(comp)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h >= 2 and w >= 2 and comp.shape[0] == h * w:
                rects.append((r0, c0, r1, c1))
        assert len(rects) == 1, f"expected one solid canvas rectangle for color {color}, found {len(rects)}"
        canvases[color] = rects[0]
    return canvases


def _path_sequence(grid: np.ndarray, canvases: dict[int, tuple[int, int, int, int]]) -> list[tuple[int, str, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    path_mask = g != BG
    for bbox in canvases.values():
        r0, c0, r1, c1 = bbox
        path_mask[r0 : r1 + 1, c0 : c1 + 1] = False

    coords = np.argwhere(path_mask)
    assert coords.size > 0, "missing instruction path"

    endpoint_info: list[tuple[int, int, bool]] = []
    for r, c in coords:
        degree = sum(bool(path_mask[nr, nc]) for nr, nc in _neighbors4(int(r), int(c), h, w))
        if degree == 1:
            on_edge = int(r) in (0, h - 1) or int(c) in (0, w - 1)
            endpoint_info.append((int(r), int(c), on_edge))
        else:
            assert degree == 2, "instruction path must be a simple non-branching chain"

    assert len(endpoint_info) == 2, "instruction path must have exactly two endpoints"
    edge_endpoints = [(r, c) for r, c, on_edge in endpoint_info if on_edge]
    assert len(edge_endpoints) == 1, "expected exactly one edge endpoint to determine instruction order"
    start = edge_endpoints[0]

    ordered: list[tuple[int, int]] = []
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        ordered.append(cur)
        nbrs = [p for p in _neighbors4(cur[0], cur[1], h, w) if path_mask[p]]
        nexts = [p for p in nbrs if p != prev]
        if not nexts:
            break
        assert len(nexts) == 1, "instruction path became ambiguous during traversal"
        prev, cur = cur, nexts[0]

    segments: list[tuple[int, str, int]] = []
    i = 0
    while i < len(ordered):
        j = i + 1
        color = int(g[ordered[i]])
        while j < len(ordered) and int(g[ordered[j]]) == color:
            j += 1

        if j - i >= 2:
            dr = ordered[i + 1][0] - ordered[i][0]
            dc = ordered[i + 1][1] - ordered[i][1]
        else:
            prev_delta = None
            next_delta = None
            if i > 0:
                prev_delta = (ordered[i][0] - ordered[i - 1][0], ordered[i][1] - ordered[i - 1][1])
            if j < len(ordered):
                next_delta = (ordered[j][0] - ordered[i][0], ordered[j][1] - ordered[i][1])
            if prev_delta is None:
                assert next_delta is not None
                dr, dc = next_delta
            elif next_delta is None:
                dr, dc = prev_delta
            else:
                assert prev_delta == next_delta, "singleton middle segment must remain collinear"
                dr, dc = prev_delta

        direction_lookup = {v: k for k, v in DIR_TO_DELTA.items()}
        direction = direction_lookup[(int(dr), int(dc))]
        segments.append((color, direction, j - i))
        i = j

    return segments


def main(grid: np.ndarray) -> np.ndarray:
    """Decode the colored edge-start path into drawing commands for the matching canvas rectangles.

    Input structure:
    - Background is black.
    - Several solid monochrome rectangular canvas blocks sit apart from one another; each canvas color is unique.
    - A single 4-connected instruction path starts at a border endpoint, stays outside the canvases, and is split into straight monochrome segments whose colors match canvas colors.

    Rule:
    - Remove the path and keep only the canvas rectangles on a black background.
    - Traverse the path from its unique border endpoint, split it at each color change, and record each segment's color, length, and travel direction.
    - Treat each segment as a command for the same-colored canvas: from that canvas's private cursor (initially its top-left cell), draw a maroon line of the recorded length in the recorded direction, then advance the cursor one more step in that direction, wrapping to the next row or column start when the step exits the current edge.

    Key insight:
    - The path is not output geometry but a program: color selects the target canvas, segment length selects how many cells to paint, and segment direction selects the drawing orientation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    canvases = _canvas_rectangles(g)
    instructions = _path_sequence(g, canvases)

    out = np.zeros_like(g)
    cursors: dict[int, tuple[int, int]] = {}
    for color, bbox in canvases.items():
        r0, c0, r1, c1 = bbox
        out[r0 : r1 + 1, c0 : c1 + 1] = color
        cursors[color] = (r0, c0)

    for color, direction, length in instructions:
        assert color in canvases, f"missing canvas for instruction color {color}"
        result = _line_cells_and_next_cursor(cursors[color], canvases[color], direction, length)
        assert result is not None, f"instruction ({color}, {direction}, {length}) overruns its canvas"
        cells, next_cursor = result
        for r, c in cells:
            out[r, c] = MAROON
        cursors[color] = next_cursor

    return out


def _segment_cells_valid(
    cells: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
    predecessor: tuple[int, int] | None,
    h: int,
    w: int,
) -> bool:
    for idx, (r, c) in enumerate(cells):
        if not (0 <= r < h and 0 <= c < w):
            return False
        if (r, c) in occupied:
            return False
        allowed_prev = predecessor if idx == 0 else cells[idx - 1]
        for nr, nc in _neighbors4(r, c, h, w):
            if (nr, nc) in occupied and (nr, nc) != allowed_prev:
                return False
    return True


def _path_geometry(
    rng: np.random.Generator,
    h: int,
    w: int,
    n_segments: int,
    required_easy: int,
) -> list[dict[str, object]] | None:
    if rng.random() < 0.5:
        start = (0, int(rng.integers(1, w - 1)))
        first_dir = "D"
    else:
        start = (int(rng.integers(1, h - 1)), 0)
        first_dir = "R"

    occupied: set[tuple[int, int]] = set()
    segments: list[dict[str, object]] = []

    def easy_count(prefix: list[dict[str, object]]) -> int:
        return sum(1 for seg in prefix if seg["dir"] in ("R", "D"))

    def dfs(end_cell: tuple[int, int] | None, prev_dir: str | None, idx: int) -> bool:
        if idx == n_segments:
            if end_cell is None:
                return False
            r, c = end_cell
            if r in (0, h - 1) or c in (0, w - 1):
                return False
            if easy_count(segments) < required_easy:
                return False
            dirs = {seg["dir"] for seg in segments}
            if len(dirs) < 3:
                return False
            turns = sum(segments[i]["dir"] != segments[i - 1]["dir"] for i in range(1, len(segments)))
            return turns >= 2

        if prev_dir is None:
            dir_choices = [first_dir]
        else:
            dir_choices = [prev_dir, LEFT_TURN[prev_dir], RIGHT_TURN[prev_dir]]
            rng.shuffle(dir_choices)

        remaining = n_segments - idx - 1
        need_easy = max(0, required_easy - easy_count(segments))

        for direction in dir_choices:
            if remaining < need_easy and direction not in ("R", "D"):
                continue

            length_choices = list(range(2, 6))
            rng.shuffle(length_choices)
            for length in length_choices:
                dr, dc = DIR_TO_DELTA[direction]
                start_cell = start if end_cell is None else (end_cell[0] + dr, end_cell[1] + dc)
                cells = [(start_cell[0] + dr * k, start_cell[1] + dc * k) for k in range(length)]
                if not _segment_cells_valid(cells, occupied, end_cell, h, w):
                    continue

                for cell in cells:
                    occupied.add(cell)
                segments.append({"dir": direction, "len": length, "cells": cells})
                if dfs(cells[-1], direction, idx + 1):
                    return True
                segments.pop()
                for cell in cells:
                    occupied.remove(cell)

        return False

    if dfs(None, None, 0):
        return segments
    return None


def _place_canvases(
    rng: np.random.Generator,
    h: int,
    w: int,
    path_cells: Iterable[tuple[int, int]],
    n_canvases: int,
) -> list[dict[str, object]] | None:
    colors = [int(c) for c in rng.choice(np.array(PALETTE, dtype=int), size=n_canvases, replace=False)]
    blocked = np.zeros((h, w), dtype=bool)
    blocked[0, :] = True
    blocked[-1, :] = True
    blocked[:, 0] = True
    blocked[:, -1] = True

    for r, c in path_cells:
        rr0 = max(0, r - 1)
        rr1 = min(h, r + 2)
        cc0 = max(0, c - 1)
        cc1 = min(w, c + 2)
        blocked[rr0:rr1, cc0:cc1] = True

    canvases: list[dict[str, object]] = []
    for color in colors:
        placed = False
        for _ in range(250):
            if rng.random() < 0.45:
                side = int(rng.integers(4, 8))
                rect_h = side
                rect_w = side
            else:
                rect_h = int(rng.integers(4, 8))
                rect_w = int(rng.integers(4, 8))
            if rect_h + 2 >= h or rect_w + 2 >= w:
                continue
            r0 = int(rng.integers(1, h - rect_h))
            c0 = int(rng.integers(1, w - rect_w))
            r1 = r0 + rect_h - 1
            c1 = c0 + rect_w - 1
            if blocked[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2].any():
                continue
            blocked[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2] = True
            canvases.append({"color": color, "bbox": (r0, c0, r1, c1)})
            placed = True
            break
        if not placed:
            return None
    return canvases


def _assign_colors_to_segments(
    rng: np.random.Generator,
    segments: list[dict[str, object]],
    canvases: list[dict[str, object]],
) -> list[int] | None:
    colors = [int(canvas["color"]) for canvas in canvases]
    cursors = {
        int(canvas["color"]): (int(canvas["bbox"][0]), int(canvas["bbox"][1]))
        for canvas in canvases
    }
    bboxes = {int(canvas["color"]): canvas["bbox"] for canvas in canvases}
    assignment = [-1] * len(segments)

    def rec(idx: int, prev_color: int | None, used: set[int], cur: dict[int, tuple[int, int]]) -> bool:
        if len(colors) - len(used) > len(segments) - idx:
            return False
        if idx == len(segments):
            return len(used) == len(colors)

        direction = str(segments[idx]["dir"])
        length = int(segments[idx]["len"])

        candidates: list[tuple[int, float]] = []
        for color in colors:
            if color == prev_color:
                continue
            if _line_cells_and_next_cursor(cur[color], bboxes[color], direction, length) is None:
                continue
            priority = 0.0 if color not in used else 1.0
            candidates.append((color, priority + float(rng.random()) * 0.01))
        candidates.sort(key=lambda item: item[1])

        for color, _ in candidates:
            result = _line_cells_and_next_cursor(cur[color], bboxes[color], direction, length)
            assert result is not None
            _, next_cursor = result
            next_cur = dict(cur)
            next_cur[color] = next_cursor
            assignment[idx] = color
            if rec(idx + 1, color, used | {color}, next_cur):
                return True

        assignment[idx] = -1
        return False

    if rec(0, None, set(), cursors):
        return assignment
    return None


def _build_grid(
    h: int,
    w: int,
    canvases: list[dict[str, object]],
    segments: list[dict[str, object]],
    assignment: list[int],
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for canvas in canvases:
        color = int(canvas["color"])
        r0, c0, r1, c1 = canvas["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = color
    for seg, color in zip(segments, assignment):
        for r, c in seg["cells"]:
            grid[r, c] = int(color)
    return grid


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(22, 31))
    w = int(rng.integers(22, 31))
    n_canvases = int(rng.integers(3, 6))
    n_segments = int(rng.integers(n_canvases + 3, n_canvases + 7))

    segments = _path_geometry(rng, h, w, n_segments, required_easy=n_canvases)
    if segments is None:
        return None

    path_cells = [tuple(cell) for seg in segments for cell in seg["cells"]]
    canvases = _place_canvases(rng, h, w, path_cells, n_canvases)
    if canvases is None:
        return None

    assignment = _assign_colors_to_segments(rng, segments, canvases)
    if assignment is None:
        return None

    grid = _build_grid(h, w, canvases, segments, assignment)
    out = main(grid)
    if np.array_equal(grid, out):
        return None
    if np.count_nonzero(out == MAROON) < n_segments:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
