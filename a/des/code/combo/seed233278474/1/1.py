import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

FRAME_COLOR = GREY
MARKER_COLOR = YELLOW
INPUT_COLORS = (BLUE, RED, GREEN, MAGENTA, ORANGE)
RANK_RECOLORS = (AZURE, MAROON, YELLOW)

GEN_RETRIES = 300
TIE_PROBABILITY = 0.4


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(int(cr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _find_frame_and_side(grid: np.ndarray) -> tuple[tuple[int, int, int, int], tuple[int, int], str]:
    grey = np.argwhere(grid == FRAME_COLOR)
    if grey.size == 0:
        raise AssertionError("expected a grey frame")

    fr0 = int(grey[:, 0].min())
    fc0 = int(grey[:, 1].min())
    fr1 = int(grey[:, 0].max())
    fc1 = int(grey[:, 1].max())
    frame = (fr0, fc0, fr1, fc1)

    marker_candidates: list[tuple[tuple[int, int], str]] = []
    for r, c in np.argwhere(grid == MARKER_COLOR):
        rr, cc = int(r), int(c)
        if rr == fr0 - 1 and fc0 <= cc <= fc1 and int(grid[fr0, cc]) != FRAME_COLOR:
            marker_candidates.append(((rr, cc), "top"))
        if rr == fr1 + 1 and fc0 <= cc <= fc1 and int(grid[fr1, cc]) != FRAME_COLOR:
            marker_candidates.append(((rr, cc), "bottom"))
        if cc == fc0 - 1 and fr0 <= rr <= fr1 and int(grid[rr, fc0]) != FRAME_COLOR:
            marker_candidates.append(((rr, cc), "left"))
        if cc == fc1 + 1 and fr0 <= rr <= fr1 and int(grid[rr, fc1]) != FRAME_COLOR:
            marker_candidates.append(((rr, cc), "right"))

    if marker_candidates:
        marker, side = marker_candidates[0]
        return frame, marker, side

    missing = {
        "top": int(np.sum(grid[fr0, fc0 : fc1 + 1] != FRAME_COLOR)),
        "bottom": int(np.sum(grid[fr1, fc0 : fc1 + 1] != FRAME_COLOR)),
        "left": int(np.sum(grid[fr0 : fr1 + 1, fc0] != FRAME_COLOR)),
        "right": int(np.sum(grid[fr0 : fr1 + 1, fc1] != FRAME_COLOR)),
    }
    side = max(missing, key=missing.get)
    if missing[side] == 0:
        raise AssertionError("could not infer gap side")

    fallback_marker = {
        "top": (fr0 - 1, fc0),
        "bottom": (fr1 + 1, fc0),
        "left": (fr0, fc0 - 1),
        "right": (fr0, fc1 + 1),
    }[side]
    return frame, fallback_marker, side


def _extract_objects(grid: np.ndarray, frame: tuple[int, int, int, int]) -> list[dict]:
    fr0, fc0, fr1, fc1 = frame
    mask = np.zeros_like(grid, dtype=bool)
    mask[fr0 + 1 : fr1, fc0 + 1 : fc1] = (grid[fr0 + 1 : fr1, fc0 + 1 : fc1] != BLACK) & (
        grid[fr0 + 1 : fr1, fc0 + 1 : fc1] != FRAME_COLOR
    )
    objects: list[dict] = []
    for comp in _components(mask):
        r0, c0, r1, c1 = _bbox(comp)
        color = int(grid[int(comp[0, 0]), int(comp[0, 1])])
        area = int(comp.shape[0])
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        if area != height * width:
            raise AssertionError("foreground object must be a solid rectangle")
        if not np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color):
            raise AssertionError("rectangle must be monochrome")
        objects.append(
            {
                "bbox": (r0, c0, r1, c1),
                "color": color,
            }
        )
    if not (3 <= len(objects) <= 5):
        raise AssertionError("expected 3-5 objects")
    return objects


def _project_object(
    obj: dict, side: str, frame: tuple[int, int, int, int], mode: str
) -> dict:
    fr0, fc0, fr1, fc1 = frame
    r0, c0, r1, c1 = obj["bbox"]
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    center_r = int((r0 + r1) // 2)
    center_c = int((c0 + c1) // 2)

    if side in ("top", "bottom"):
        if mode == "axis":
            cells = [(rr, center_c) for rr in range(r0, r1 + 1)]
            length = height
            distance = int(r0 - fr0) if side == "top" else int(fr1 - r1)
            odd_center_dim = (width % 2 == 1)
        else:
            cells = [(center_r, cc) for cc in range(c0, c1 + 1)]
            length = width
            distance = int(center_r - fr0) if side == "top" else int(fr1 - center_r)
            odd_center_dim = (height % 2 == 1)
    else:
        if mode == "axis":
            cells = [(center_r, cc) for cc in range(c0, c1 + 1)]
            length = width
            distance = int(c0 - fc0) if side == "left" else int(fc1 - c1)
            odd_center_dim = (height % 2 == 1)
        else:
            cells = [(rr, center_c) for rr in range(r0, r1 + 1)]
            length = height
            distance = int(center_c - fc0) if side == "left" else int(fc1 - center_c)
            odd_center_dim = (width % 2 == 1)

    return {
        "cells": cells,
        "length": int(length),
        "distance": int(distance),
        "odd_center_dim": bool(odd_center_dim),
        "bbox": obj["bbox"],
        "color": int(obj["color"]),
    }


def _mode_score(projected: list[dict], prefer_axis: bool) -> tuple[int, int, int]:
    seen: set[tuple[int, int]] = set()
    overlap = 0
    for item in projected:
        for cell in item["cells"]:
            if cell in seen:
                overlap += 1
            seen.add(cell)
    odd_count = sum(1 for item in projected if item["odd_center_dim"])
    return int(overlap), int(-odd_count), 0 if prefer_axis else 1


def _choose_projection_mode(objects: list[dict], side: str, frame: tuple[int, int, int, int]) -> str:
    axis_items = [_project_object(obj, side, frame, "axis") for obj in objects]
    compress_items = [_project_object(obj, side, frame, "compress") for obj in objects]
    if _mode_score(axis_items, True) <= _mode_score(compress_items, False):
        return "axis"
    return "compress"


def _rank_color(rank: int, original: int) -> int:
    if rank == 0:
        return int(original)
    return int(RANK_RECOLORS[(rank - 1) % len(RANK_RECOLORS)])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each interior rectangle onto an axis chosen by the frame gap, then
    recolor projected segments by descending length.

    The natural-language description has one ambiguity about whether the result
    should align with the chosen axis or with its perpendicular. The solver
    defaults to the more self-consistent interpretation and only switches when
    the alternate interpretation is structurally cleaner.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    frame, marker, side = _find_frame_and_side(g)
    objects = _extract_objects(g, frame)
    mode = _choose_projection_mode(objects, side, frame)
    projected = [_project_object(obj, side, frame, mode) for obj in objects]

    out = g.copy()
    if 0 <= marker[0] < g.shape[0] and 0 <= marker[1] < g.shape[1]:
        out[marker] = BLACK

    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = BLACK

    order = sorted(
        range(len(projected)),
        key=lambda i: (
            -int(projected[i]["length"]),
            int(projected[i]["distance"]),
            tuple(map(int, projected[i]["bbox"])),
        ),
    )

    for rank, idx in enumerate(order):
        color = _rank_color(int(rank), int(projected[idx]["color"]))
        for r, c in projected[idx]["cells"]:
            out[int(r), int(c)] = int(color)

    return out


def _random_parts(total: int, count: int, min_part: int) -> list[int] | None:
    rest = int(total - count * min_part)
    if rest < 0:
        return None
    parts = [int(min_part)] * count
    for _ in range(rest):
        parts[random.randrange(count)] += 1
    random.shuffle(parts)
    return parts


def _lane_ranges(start: int, total: int, count: int, min_lane: int, gap: int = 1) -> list[tuple[int, int]] | None:
    lane_space = int(total - gap * (count - 1))
    widths = _random_parts(int(lane_space), int(count), int(min_lane))
    if widths is None:
        return None
    lanes: list[tuple[int, int]] = []
    cursor = int(start)
    for width in widths:
        lanes.append((int(cursor), int(cursor + width - 1)))
        cursor += int(width + gap)
    return lanes


def _sample_lengths(count: int, lo: int, hi: int) -> list[int] | None:
    values = list(range(int(lo), int(hi) + 1))
    if len(values) < count - 1:
        return None

    want_tie = count >= 4 and random.random() < TIE_PROBABILITY
    if want_tie:
        tie_value = random.choice(values)
        others = [v for v in values if v != tie_value]
        if len(others) < count - 2:
            return None
        picked = random.sample(others, count - 2) + [tie_value, tie_value]
    else:
        if len(values) < count:
            return None
        picked = random.sample(values, count)
    random.shuffle(picked)
    return [int(v) for v in picked]


def _draw_frame_with_gap(grid: np.ndarray, side: str) -> tuple[int, int]:
    h, w = map(int, grid.shape)
    fr0, fc0, fr1, fc1 = 1, 1, h - 2, w - 2
    grid[fr0, fc0 : fc1 + 1] = FRAME_COLOR
    grid[fr1, fc0 : fc1 + 1] = FRAME_COLOR
    grid[fr0 : fr1 + 1, fc0] = FRAME_COLOR
    grid[fr0 : fr1 + 1, fc1] = FRAME_COLOR

    gap_len = random.randint(2, 4)
    if side in ("top", "bottom"):
        gap_start = random.randint(fc0 + 1, fc1 - gap_len)
        row = fr0 if side == "top" else fr1
        grid[row, gap_start : gap_start + gap_len] = BLACK
        marker_col = random.randint(gap_start, gap_start + gap_len - 1)
        marker_row = fr0 - 1 if side == "top" else fr1 + 1
        grid[marker_row, marker_col] = MARKER_COLOR
    else:
        gap_start = random.randint(fr0 + 1, fr1 - gap_len)
        col = fc0 if side == "left" else fc1
        grid[gap_start : gap_start + gap_len, col] = BLACK
        marker_row = random.randint(gap_start, gap_start + gap_len - 1)
        marker_col = fc0 - 1 if side == "left" else fc1 + 1
        grid[marker_row, marker_col] = MARKER_COLOR

    return gap_len, gap_start


def _place_vertical_axis_objects(grid: np.ndarray, side: str, count: int) -> bool:
    h, w = map(int, grid.shape)
    inner_h = h - 4
    inner_w = w - 4
    lanes = _lane_ranges(2, inner_w, count, min_lane=3, gap=1)
    if lanes is None:
        return False

    heights = _sample_lengths(count, 2, min(7, inner_h - 1))
    if heights is None:
        return False

    colors = random.sample(INPUT_COLORS, count)
    rects = []
    for lane, height, color in zip(lanes, heights, colors):
        c_lo, c_hi = lane
        width_choices = [v for v in (3, 5, 1) if v <= (c_hi - c_lo + 1) and v % 2 == 1]
        if not width_choices:
            return False
        width = random.choice(width_choices)
        r0_min = 2
        r0_max = h - 2 - height
        if r0_min > r0_max:
            return False
        c0_min = c_lo
        c0_max = c_hi - width + 1
        if c0_min > c0_max:
            return False
        r0 = random.randint(r0_min, r0_max)
        c0 = random.randint(c0_min, c0_max)
        rects.append((r0, c0, height, width, color))

    distances = {}
    for r0, _c0, height, _width, _color in rects:
        dist = int(r0 - 1) if side == "top" else int((h - 2) - (r0 + height - 1))
        key = int(height)
        distances.setdefault(key, []).append(dist)
    if any(len(vals) != len(set(vals)) for vals in distances.values()):
        return False

    for r0, c0, height, width, color in rects:
        grid[r0 : r0 + height, c0 : c0 + width] = int(color)
    return True


def _place_horizontal_axis_objects(grid: np.ndarray, side: str, count: int) -> bool:
    h, w = map(int, grid.shape)
    inner_h = h - 4
    inner_w = w - 4
    lanes = _lane_ranges(2, inner_h, count, min_lane=3, gap=1)
    if lanes is None:
        return False

    widths = _sample_lengths(count, 2, min(7, inner_w - 1))
    if widths is None:
        return False

    colors = random.sample(INPUT_COLORS, count)
    rects = []
    for lane, width, color in zip(lanes, widths, colors):
        r_lo, r_hi = lane
        height_choices = [v for v in (3, 5, 1) if v <= (r_hi - r_lo + 1) and v % 2 == 1]
        if not height_choices:
            return False
        height = random.choice(height_choices)
        c0_min = 2
        c0_max = w - 2 - width
        if c0_min > c0_max:
            return False
        r0_min = r_lo
        r0_max = r_hi - height + 1
        if r0_min > r0_max:
            return False
        r0 = random.randint(r0_min, r0_max)
        c0 = random.randint(c0_min, c0_max)
        rects.append((r0, c0, height, width, color))

    distances = {}
    for _r0, c0, _height, width, _color in rects:
        dist = int(c0 - 1) if side == "left" else int((w - 2) - (c0 + width - 1))
        key = int(width)
        distances.setdefault(key, []).append(dist)
    if any(len(vals) != len(set(vals)) for vals in distances.values()):
        return False

    for r0, c0, height, width, color in rects:
        grid[r0 : r0 + height, c0 : c0 + width] = int(color)
    return True


def _gen_attempt() -> np.ndarray | None:
    side = random.choice(("top", "bottom", "left", "right"))
    count = random.randint(3, 5)

    if side in ("top", "bottom"):
        h = random.randint(13, 19)
        w = random.randint(max(15, 4 * count + 3), max(19, 4 * count + 7))
    else:
        h = random.randint(max(15, 4 * count + 3), max(19, 4 * count + 7))
        w = random.randint(13, 19)

    grid = np.zeros((h, w), dtype=int)
    _draw_frame_with_gap(grid, side)

    ok = (
        _place_vertical_axis_objects(grid, side, count)
        if side in ("top", "bottom")
        else _place_horizontal_axis_objects(grid, side, count)
    )
    if not ok:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if not isinstance(out, np.ndarray) or out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
