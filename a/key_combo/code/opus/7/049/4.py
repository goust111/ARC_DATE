from __future__ import annotations

import random

import numpy as np


BG = 0
FRAME = 5
INNER_H = 4
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _segment(a: int, b: int) -> range:
    step = 1 if b >= a else -1
    return range(a, b + step, step)


def _frame_data(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    frames: list[dict] = []
    for comp in _components(grid == FRAME):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        frame_h = r1 - r0 + 1
        frame_w = c1 - c0 + 1
        if frame_h < 3 or frame_w < 3:
            continue
        perim = 2 * (frame_h + frame_w) - 4
        if len(comp) != perim:
            continue
        if not np.all(grid[r0, c0 : c1 + 1] == FRAME):
            continue
        if not np.all(grid[r1, c0 : c1 + 1] == FRAME):
            continue
        if not np.all(grid[r0 : r1 + 1, c0] == FRAME):
            continue
        if not np.all(grid[r0 : r1 + 1, c1] == FRAME):
            continue

        inner_h = frame_h - 2
        inner_w = frame_w - 2
        if inner_h <= 0 or inner_w <= 0:
            continue

        colors: set[int] = set()
        left_rows: list[int] = []
        right_rows: list[int] = []
        top_cols: list[int] = []
        bottom_cols: list[int] = []

        if c0 - 1 >= 0:
            for rr in range(r0 + 1, r1):
                val = int(grid[rr, c0 - 1])
                if val != BG:
                    colors.add(val)
                    left_rows.append(rr - (r0 + 1))
        if c1 + 1 < w:
            for rr in range(r0 + 1, r1):
                val = int(grid[rr, c1 + 1])
                if val != BG:
                    colors.add(val)
                    right_rows.append(rr - (r0 + 1))
        if r0 - 1 >= 0:
            for cc in range(c0 + 1, c1):
                val = int(grid[r0 - 1, cc])
                if val != BG:
                    colors.add(val)
                    top_cols.append(cc - (c0 + 1))
        if r1 + 1 < h:
            for cc in range(c0 + 1, c1):
                val = int(grid[r1 + 1, cc])
                if val != BG:
                    colors.add(val)
                    bottom_cols.append(cc - (c0 + 1))

        assert len(colors) == 1, "expected one trace color per frame"
        assert len(left_rows) == 1 and len(right_rows) == 1, "expected one left and one right trace"

        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "inner_top": r0 + 1,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "color": int(next(iter(colors))),
                "left_row": int(left_rows[0]),
                "right_row": int(right_rows[0]),
                "top_cols": sorted(set(map(int, top_cols))),
                "bottom_cols": sorted(set(map(int, bottom_cols))),
            }
        )

    frames.sort(key=lambda item: item["bbox"][1])
    return frames


def _local_path(frame: dict) -> np.ndarray:
    h = int(frame["inner_h"])
    w = int(frame["inner_w"])
    color = int(frame["color"])
    left_row = int(frame["left_row"])
    right_row = int(frame["right_row"])

    out = np.zeros((h, w), dtype=int)

    if left_row == right_row:
        out[left_row, :] = color
        horizontal_row = left_row
    else:
        for rr in _segment(left_row, right_row):
            out[rr, 0] = color
        out[right_row, :] = color
        horizontal_row = right_row

    for col in frame["top_cols"]:
        for rr in _segment(horizontal_row, 0):
            out[rr, int(col)] = color
    for col in frame["bottom_cols"]:
        for rr in _segment(horizontal_row, h - 1):
            out[rr, int(col)] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each framed trace pattern into an internal path strip and align adjacent endpoints before concatenation.

    Input structure:
    - The grid has a black background with several small grey rectangular frames arranged left to right.
    - Each frame has a black interior and one monochrome set of external trace pixels touching its left, right, and optionally top or bottom side.
    - Pure black separator columns sit between neighboring framed regions.

    Rule:
    - Detect every grey frame and map each touching external trace pixel to the corresponding cell on the frame's inner boundary.
    - For each frame, draw the trace color inside the frame as a Manhattan path from the left endpoint to the right endpoint, using the fewest segments and breaking ties by going vertical first, then attach vertical branches to any top or bottom endpoints.
    - Remove the inter-frame separator columns conceptually by taking only the inner areas, keep the first inner area at its original vertical position, then shift each later inner area so its left endpoint row matches the previous placed area's right endpoint row before concatenating the inner areas horizontally.

    Key insight:
    - The external trace pixels determine the internal path endpoints locally, but the final strip geometry is fixed globally by chaining each frame's left endpoint row to the previous frame's right endpoint row.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frames = _frame_data(g)
    assert frames, "expected at least one framed object"

    out = np.zeros((g.shape[0], sum(int(frame["inner_w"]) for frame in frames)), dtype=int)

    local0 = _local_path(frames[0])
    top0 = int(frames[0]["inner_top"])
    assert 0 <= top0 and top0 + local0.shape[0] <= out.shape[0]
    out[top0 : top0 + local0.shape[0], : local0.shape[1]] = np.maximum(
        out[top0 : top0 + local0.shape[0], : local0.shape[1]], local0
    )

    prev_right_abs = top0 + int(frames[0]["right_row"])
    col0 = int(local0.shape[1])

    for frame in frames[1:]:
        local = _local_path(frame)
        placed_top = prev_right_abs - int(frame["left_row"])
        assert 0 <= placed_top and placed_top + local.shape[0] <= out.shape[0]
        out[placed_top : placed_top + local.shape[0], col0 : col0 + local.shape[1]] = np.maximum(
            out[placed_top : placed_top + local.shape[0], col0 : col0 + local.shape[1]],
            local,
        )
        prev_right_abs = placed_top + int(frame["right_row"])
        col0 += int(local.shape[1])

    return out


def _sample_widths_and_separators() -> tuple[list[int], list[int]] | None:
    for _ in range(100):
        widths = [random.choice((4, 5)) for _ in range(3)]
        separators = [random.choice((2, 3)) for _ in range(2)]
        total_width = sum(w + 2 for w in widths) + sum(sep + 2 for sep in separators) + 2
        if total_width <= 30:
            return widths, separators
    return None


def _sample_frame_spec(width: int) -> dict:
    return {
        "width": width,
        "left_row": random.randrange(INNER_H),
        "right_row": random.randrange(INNER_H),
        "top_cols": sorted(random.sample(range(width), random.choices((0, 1, 2), weights=(2, 5, 3), k=1)[0])),
        "bottom_cols": sorted(random.sample(range(width), random.choices((0, 1, 2), weights=(2, 5, 3), k=1)[0])),
    }


def _relative_tops(specs: list[dict]) -> list[int]:
    tops = [0]
    prev_right = int(specs[0]["right_row"])
    for spec in specs[1:]:
        top = prev_right - int(spec["left_row"])
        tops.append(top)
        prev_right = top + int(spec["right_row"])
    return tops


def _build_input(specs: list[dict], colors: list[int], separators: list[int]) -> np.ndarray:
    relative_tops = _relative_tops(specs)
    min_top = min(relative_tops)
    max_bottom = max(top + INNER_H - 1 for top in relative_tops)
    extra_top = random.randint(0, 1)
    extra_bottom = random.randint(0, 1)

    inner_top = 2 - min_top + extra_top
    frame_top = inner_top - 1
    height = (max_bottom - min_top + 1) + 4 + extra_top + extra_bottom

    outer_widths = [int(spec["width"]) + 2 for spec in specs]
    width = sum(outer_widths) + sum(sep + 2 for sep in separators) + 2
    grid = np.zeros((height, width), dtype=int)

    c0 = 1
    for idx, (spec, color) in enumerate(zip(specs, colors)):
        inner_w = int(spec["width"])
        r0, r1 = frame_top, frame_top + INNER_H + 1
        c1 = c0 + inner_w + 1

        grid[r0, c0 : c1 + 1] = FRAME
        grid[r1, c0 : c1 + 1] = FRAME
        grid[r0 : r1 + 1, c0] = FRAME
        grid[r0 : r1 + 1, c1] = FRAME

        grid[frame_top + 1 + int(spec["left_row"]), c0 - 1] = color
        grid[frame_top + 1 + int(spec["right_row"]), c1 + 1] = color
        for col in spec["top_cols"]:
            grid[r0 - 1, c0 + 1 + int(col)] = color
        for col in spec["bottom_cols"]:
            grid[r1 + 1, c0 + 1 + int(col)] = color

        if idx < len(specs) - 1:
            c0 = c0 + outer_widths[idx] + separators[idx] + 2

    return grid


def _valid_spec_mix(specs: list[dict]) -> bool:
    has_branch = any(spec["top_cols"] or spec["bottom_cols"] for spec in specs)
    has_bend = any(int(spec["left_row"]) != int(spec["right_row"]) for spec in specs)
    has_straight = any(int(spec["left_row"]) == int(spec["right_row"]) for spec in specs)
    return has_branch and has_bend and has_straight


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        layout = _sample_widths_and_separators()
        if layout is None:
            continue
        widths, separators = layout
        specs = [_sample_frame_spec(width) for width in widths]
        if not _valid_spec_mix(specs):
            continue

        colors = random.sample(PALETTE, len(specs))
        grid = _build_input(specs, colors, separators)

        try:
            out = main(grid)
        except Exception:
            continue

        if out.ndim != 2 or max(grid.shape) > 30 or max(out.shape) > 30:
            continue
        if out.shape[1] != sum(widths):
            continue
        if len(_frame_data(grid)) != len(specs):
            continue
        return grid

    raise RuntimeError("failed to generate a valid input")
