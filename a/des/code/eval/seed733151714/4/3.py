from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
ORANGE = 7
FILL_COLORS = (1, 2, 3, 6, 8, 9)

GEN_RETRIES = 300
GRID_HEIGHT_RANGE = (14, 24)
GRID_WIDTH_RANGE = (14, 24)
OBJECT_COUNT_RANGE = (2, 5)
RECT_HEIGHT_RANGE = (4, 8)
RECT_WIDTH_RANGE = (4, 8)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _locate_rectangle_from_marker(
    grid: np.ndarray, marker_r: int, marker_c: int
) -> dict[str, object] | None:
    h, w = grid.shape
    rect_top = marker_r + 1
    if rect_top >= h:
        return None

    fill = int(grid[rect_top, marker_c])
    if fill in (BLACK, GREY, YELLOW, ORANGE):
        return None

    left = marker_c
    while left - 1 >= 0 and int(grid[rect_top, left - 1]) == fill:
        left -= 1
    right = marker_c
    while right + 1 < w and int(grid[rect_top, right + 1]) == fill:
        right += 1

    bottom = rect_top
    while bottom + 1 < h:
        row = grid[bottom + 1, left : right + 1]
        if int(row[0]) != fill or int(row[-1]) != fill:
            break
        if not np.all((row == fill) | (row == BLACK)):
            break
        bottom += 1

    region = grid[rect_top : bottom + 1, left : right + 1]
    if region.shape[0] < 3 or region.shape[1] < 3:
        return None
    if not np.all((region == fill) | (region == BLACK)):
        return None
    if not np.all(region[0, :] == fill):
        return None
    if not np.all(region[-1, :] == fill):
        return None
    if not np.all(region[:, 0] == fill):
        return None
    if not np.all(region[:, -1] == fill):
        return None
    if not np.any(region == BLACK):
        return None

    marker_band = grid[marker_r, left : right + 1]
    marker_offsets = np.flatnonzero((marker_band == YELLOW) | (marker_band == ORANGE))
    if marker_offsets.size != 1:
        return None
    if int(marker_offsets[0]) != marker_c - left:
        return None

    return {
        "fill": fill,
        "marker": int(grid[marker_r, marker_c]),
        "top": rect_top,
        "left": left,
        "bottom": bottom,
        "right": right,
        "region": region.copy(),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each rectangle's hole-mask or solid-mask to a blank grey canvas.

    Yellow markers stamp the black-hole pixels.
    Orange markers stamp the non-black pixels.
    """
    g = np.asarray(grid, dtype=int)
    out = np.full(g.shape, GREY, dtype=int)

    rectangles: list[dict[str, object]] = []
    for r, c in np.argwhere((g == YELLOW) | (g == ORANGE)):
        rect = _locate_rectangle_from_marker(g, int(r), int(c))
        if rect is not None:
            rectangles.append(rect)

    seen: set[tuple[int, int, int, int]] = set()
    unique_rectangles: list[dict[str, object]] = []
    for rect in rectangles:
        key = (
            int(rect["top"]),
            int(rect["left"]),
            int(rect["bottom"]),
            int(rect["right"]),
        )
        if key not in seen:
            seen.add(key)
            unique_rectangles.append(rect)

    unique_rectangles.sort(
        key=lambda rect: (
            int(rect["top"]) + int(rect["bottom"]),
            int(rect["left"]) + int(rect["right"]),
        )
    )

    # Later rectangles overwrite earlier colored pixels if their stamp areas overlap.
    for rect in unique_rectangles:
        top = int(rect["top"])
        left = int(rect["left"])
        bottom = int(rect["bottom"])
        right = int(rect["right"])
        fill = int(rect["fill"])
        region = np.asarray(rect["region"], dtype=int)
        if int(rect["marker"]) == YELLOW:
            stamp = region == BLACK
        else:
            stamp = region != BLACK
        window = out[top : bottom + 1, left : right + 1]
        window[stamp] = fill

    return out


def _fill_component_count(mask: np.ndarray) -> int:
    return len(_components(mask))


def _make_hole_mask(height: int, width: int, rng: random.Random) -> np.ndarray | None:
    inner_h = height - 2
    inner_w = width - 2
    if inner_h <= 0 or inner_w <= 0:
        return None

    for _ in range(60):
        holes = np.zeros((height, width), dtype=bool)
        mode = rng.choice(("dots", "blocks", "bars"))

        if mode == "dots":
            coords = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1)]
            rng.shuffle(coords)
            count = rng.randint(1, min(len(coords) - 1, max(1, len(coords) // 3)))
            for r, c in coords[:count]:
                holes[r, c] = True

        elif mode == "blocks":
            block_count = rng.randint(1, 2)
            for _ in range(block_count):
                block_h = rng.randint(1, max(1, inner_h // 2))
                block_w = rng.randint(1, max(1, inner_w // 2))
                top = rng.randint(1, height - block_h - 1)
                left = rng.randint(1, width - block_w - 1)
                holes[top : top + block_h, left : left + block_w] = True

        else:
            if inner_w >= 2:
                row = rng.randint(1, height - 2)
                span = rng.randint(1, max(1, inner_w - 1))
                left = rng.randint(1, width - span - 1)
                holes[row, left : left + span] = True
            if inner_h >= 2:
                col = rng.randint(1, width - 2)
                span = rng.randint(1, max(1, inner_h - 1))
                top = rng.randint(1, height - span - 1)
                holes[top : top + span, col] = True
            if not holes.any():
                holes[rng.randint(1, height - 2), rng.randint(1, width - 2)] = True

        inner_area = inner_h * inner_w
        hole_count = int(holes.sum())
        if hole_count == 0 or hole_count >= inner_area:
            continue

        fill_mask = ~holes
        if _fill_component_count(fill_mask) != 1:
            continue

        return holes

    return None


def _balanced_markers(count: int, rng: random.Random) -> list[int]:
    yellow_count = count // 2
    orange_count = count // 2
    if count % 2 == 1:
        if rng.random() < 0.5:
            yellow_count += 1
        else:
            orange_count += 1
    markers = [YELLOW] * yellow_count + [ORANGE] * orange_count
    rng.shuffle(markers)
    return markers


def _paint_rectangle(
    grid: np.ndarray,
    top: int,
    left: int,
    fill: int,
    holes: np.ndarray,
    marker_color: int,
    marker_col: int,
) -> None:
    height, width = holes.shape
    region = np.full((height, width), fill, dtype=int)
    region[holes] = BLACK
    grid[top : top + height, left : left + width] = region
    grid[top - 1, marker_col] = marker_color


def _render_expected(
    shape: tuple[int, int], specs: list[dict[str, object]]
) -> np.ndarray:
    out = np.full(shape, GREY, dtype=int)
    ordered = sorted(
        specs,
        key=lambda spec: (
            2 * int(spec["top"]) + int(spec["holes"].shape[0]) - 1,
            2 * int(spec["left"]) + int(spec["holes"].shape[1]) - 1,
        ),
    )
    for spec in ordered:
        top = int(spec["top"])
        left = int(spec["left"])
        fill = int(spec["fill"])
        holes = np.asarray(spec["holes"], dtype=bool)
        stamp = holes if int(spec["marker"]) == YELLOW else ~holes
        window = out[top : top + holes.shape[0], left : left + holes.shape[1]]
        window[stamp] = fill
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(*GRID_HEIGHT_RANGE)
    width = rng.randint(*GRID_WIDTH_RANGE)
    grid = np.full((height, width), GREY, dtype=int)

    max_objects = min(OBJECT_COUNT_RANGE[1], len(FILL_COLORS))
    object_count = rng.randint(OBJECT_COUNT_RANGE[0], max_objects)
    fills = rng.sample(list(FILL_COLORS), object_count)
    markers = _balanced_markers(object_count, rng)

    specs: list[dict[str, object]] = []
    for fill, marker in zip(fills, markers):
        rect_h = rng.randint(
            RECT_HEIGHT_RANGE[0], min(RECT_HEIGHT_RANGE[1], height - 2)
        )
        rect_w = rng.randint(
            RECT_WIDTH_RANGE[0], min(RECT_WIDTH_RANGE[1], width - 2)
        )
        holes = _make_hole_mask(rect_h, rect_w, rng)
        if holes is None:
            return None
        specs.append({"fill": fill, "marker": marker, "holes": holes})

    occupancy = np.zeros((height, width), dtype=bool)
    placement_order = sorted(
        range(object_count),
        key=lambda idx: int(np.prod(np.asarray(specs[idx]["holes"]).shape)),
        reverse=True,
    )

    for idx in placement_order:
        holes = np.asarray(specs[idx]["holes"], dtype=bool)
        rect_h, rect_w = holes.shape
        candidates: list[tuple[int, int]] = []
        for top in range(1, height - rect_h):
            for left in range(1, width - rect_w):
                r0 = max(0, top - 2)
                r1 = min(height, top + rect_h + 1)
                c0 = max(0, left - 1)
                c1 = min(width, left + rect_w + 1)
                if occupancy[r0:r1, c0:c1].any():
                    continue
                candidates.append((top, left))
        if not candidates:
            return None
        top, left = rng.choice(candidates)
        marker_col = rng.randint(left, left + rect_w - 1)
        specs[idx]["top"] = top
        specs[idx]["left"] = left
        specs[idx]["marker_col"] = marker_col

        occupancy[max(0, top - 2) : min(height, top + rect_h + 1),
                  max(0, left - 1) : min(width, left + rect_w + 1)] = True

    for spec in specs:
        _paint_rectangle(
            grid,
            int(spec["top"]),
            int(spec["left"]),
            int(spec["fill"]),
            np.asarray(spec["holes"], dtype=bool),
            int(spec["marker"]),
            int(spec["marker_col"]),
        )

    expected = _render_expected(grid.shape, specs)
    if np.array_equal(grid, expected):
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
