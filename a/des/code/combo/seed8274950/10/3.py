from __future__ import annotations

from collections import Counter

import numpy as np


DIRS = ("up", "down", "left", "right")
GEN_ATTEMPTS = 400
GRID_MIN_SIZE = 22
GRID_MAX_SIZE = 34
FRAME_COUNT_CHOICES = (2, 3, 4)
FRAME_SIDE_MIN = 5
FRAME_SIDE_MAX = 9
PATTERN_AREA_MAX = 6
ROW_GAP_RANGE = (2, 5)
COL_GAP_RANGE = (2, 5)
MARGIN_RANGE = (1, 4)
EXTRA_SLACK_RANGE = (0, 5)
SHAPE_ATTEMPTS = 80


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            arr = np.asarray(coords, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (
                        int(arr[:, 0].min()),
                        int(arr[:, 1].min()),
                        int(arr[:, 0].max()),
                        int(arr[:, 1].max()),
                    ),
                }
            )
    return comps


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((int(r0), int(c)))
        cells.add((int(r1), int(c)))
    for r in range(r0, r1 + 1):
        cells.add((int(r), int(c0)))
        cells.add((int(r), int(c1)))
    return cells


def _gap_direction(gap: tuple[int, int], bbox: tuple[int, int, int, int]) -> str | None:
    r0, c0, r1, c1 = bbox
    r, c = gap
    matches: list[str] = []
    if r == r0:
        matches.append("up")
    if r == r1:
        matches.append("down")
    if c == c0:
        matches.append("left")
    if c == c1:
        matches.append("right")
    if len(matches) != 1:
        return None
    return matches[0]


def _frame_info(comp: dict) -> dict | None:
    r0, c0, r1, c1 = comp["bbox"]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return None

    coords = {(int(r), int(c)) for r, c in comp["coords"]}
    border = _border_cells(r0, c0, r1, c1)
    if not coords.issubset(border):
        return None

    missing = list(border - coords)
    if len(missing) != 1:
        return None
    gap = missing[0]
    if gap in {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}:
        return None
    if len(coords) != len(border) - 1:
        return None

    direction = _gap_direction(gap, (r0, c0, r1, c1))
    if direction is None:
        return None

    return {
        "color": int(comp["color"]),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "gap": (int(gap[0]), int(gap[1])),
        "direction": direction,
        "height": h,
        "width": w,
    }


def _direction_delta(direction: str, distance: int) -> tuple[int, int]:
    if direction == "up":
        return -int(distance), 0
    if direction == "down":
        return int(distance), 0
    if direction == "left":
        return 0, -int(distance)
    if direction == "right":
        return 0, int(distance)
    raise ValueError(f"unknown direction: {direction}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push each frame's interior pattern out through its 1-pixel gap.

    The gap side gives the movement direction. The pattern is translated by:
    pixel_count(pattern) + frame_span_along_that_direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    bg = _background_color(g)

    frames = []
    for comp in _find_components(g, bg):
        info = _frame_info(comp)
        if info is not None:
            frames.append(info)
    frames.sort(key=lambda item: item["bbox"])

    out = g.copy()
    clears: list[tuple[int, int]] = []
    moves: list[tuple[int, int, int]] = []

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        interior: list[tuple[int, int, int]] = []
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                color = int(g[r, c])
                if color != bg:
                    interior.append((int(r), int(c), color))
        if not interior:
            continue

        count = len(interior)
        span = frame["height"] if frame["direction"] in ("up", "down") else frame["width"]
        dy, dx = _direction_delta(frame["direction"], count + span)
        clears.extend((r, c) for r, c, _color in interior)
        moves.extend((r + dy, c + dx, color) for r, c, color in interior)

    for r, c in clears:
        out[r, c] = bg
    for r, c, color in moves:
        if 0 <= r < h and 0 <= c < w:
            out[r, c] = color
    return out


def _pick_colors(rng: np.random.Generator, count: int, exclude: set[int] | None = None) -> list[int]:
    banned = set() if exclude is None else {int(x) for x in exclude}
    pool = [color for color in range(10) if color not in banned]
    picked = rng.choice(pool, size=int(count), replace=False)
    return [int(x) for x in picked]


def _normalize_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0 = min(r for r, _c in cells)
    c0 = min(c for _r, c in cells)
    return {(int(r - r0), int(c - c0)) for r, c in cells}


def _looks_like_gap_frame(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    norm = _normalize_cells(cells)
    rows = [r for r, _c in norm]
    cols = [c for _r, c in norm]
    r1 = max(rows)
    c1 = max(cols)
    h = r1 + 1
    w = c1 + 1
    if h < 3 or w < 3:
        return False
    border = _border_cells(0, 0, r1, c1)
    missing = list(border - norm)
    if len(missing) != 1:
        return False
    if missing[0] in {(0, 0), (0, c1), (r1, 0), (r1, c1)}:
        return False
    return norm.issubset(border) and len(norm) == len(border) - 1


def _grow_pattern(
    rng: np.random.Generator,
    height: int,
    width: int,
    area: int,
) -> set[tuple[int, int]]:
    for _ in range(SHAPE_ATTEMPTS):
        start = (int(rng.integers(0, height)), int(rng.integers(0, width)))
        cells = {start}
        while len(cells) < area:
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c, height, width):
                    if (nr, nc) not in cells:
                        frontier.add((int(nr), int(nc)))
            if not frontier:
                break
            choices = list(frontier)
            pick = choices[int(rng.integers(0, len(choices)))]
            cells.add(pick)
        if len(cells) == area and not _looks_like_gap_frame(cells):
            return cells
    raise RuntimeError("failed to grow a valid interior pattern")


def _coords_bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _c in coords]
    cs = [c for _r, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _sample_object(
    rng: np.random.Generator,
    direction: str,
    frame_color: int,
    pattern_color: int,
) -> dict:
    h = int(rng.integers(FRAME_SIDE_MIN, FRAME_SIDE_MAX + 1))
    w = int(rng.integers(FRAME_SIDE_MIN, FRAME_SIDE_MAX + 1))
    inner_h = h - 2
    inner_w = w - 2
    max_area = min(PATTERN_AREA_MAX, inner_h * inner_w)
    area = int(rng.integers(1, max_area + 1))
    shape = _grow_pattern(rng, inner_h, inner_w, area)
    pattern = [(int(r + 1), int(c + 1)) for r, c in sorted(shape)]

    if direction == "up":
        gap = (0, int(rng.integers(1, w - 1)))
    elif direction == "down":
        gap = (h - 1, int(rng.integers(1, w - 1)))
    elif direction == "left":
        gap = (int(rng.integers(1, h - 1)), 0)
    else:
        gap = (int(rng.integers(1, h - 1)), w - 1)

    frame_cells = sorted(_border_cells(0, 0, h - 1, w - 1) - {gap})
    span = h if direction in ("up", "down") else w
    dy, dx = _direction_delta(direction, area + span)
    moved = [(int(r + dy), int(c + dx)) for r, c in pattern]
    mr0, mc0, mr1, mc1 = _coords_bbox(moved)
    keep_r0 = min(0, mr0)
    keep_c0 = min(0, mc0)
    keep_r1 = max(h - 1, mr1)
    keep_c1 = max(w - 1, mc1)

    return {
        "direction": direction,
        "frame_color": int(frame_color),
        "pattern_color": int(pattern_color),
        "frame_cells": frame_cells,
        "pattern_cells": pattern,
        "moved_cells": moved,
        "keep_r0": int(keep_r0),
        "keep_c0": int(keep_c0),
        "keep_h": int(keep_r1 - keep_r0 + 1),
        "keep_w": int(keep_c1 - keep_c0 + 1),
    }


def _split_extra(rng: np.random.Generator, total: int, parts: int) -> list[int]:
    if parts <= 1:
        return [int(total)]
    if total <= 0:
        return [0] * int(parts)
    cuts = sorted(int(x) for x in rng.integers(0, total + 1, size=int(parts - 1)))
    pieces: list[int] = []
    prev = 0
    for cut in cuts + [int(total)]:
        pieces.append(int(cut - prev))
        prev = cut
    return pieces


def _arrange_objects(rng: np.random.Generator, objects: list[dict]) -> tuple[int, int, list[tuple[int, int]]] | None:
    placed = [dict(obj) for obj in objects]
    order = list(range(len(placed)))
    rng.shuffle(order)
    placed = [placed[i] for i in order]

    n = len(placed)
    if n == 2:
        row_count = 1 if bool(rng.integers(0, 2)) else 2
    elif n == 3:
        row_count = 1 if int(rng.integers(0, 3)) == 0 else 2
    else:
        row_count = 2

    if row_count == 1:
        rows = [placed]
    else:
        if n == 4:
            split = 2
        else:
            split = int(rng.integers(1, n))
        rows = [placed[:split], placed[split:]]

    col_gap = int(rng.integers(*COL_GAP_RANGE))
    row_gap = int(rng.integers(*ROW_GAP_RANGE))
    top = int(rng.integers(*MARGIN_RANGE))
    bottom = int(rng.integers(*MARGIN_RANGE))
    left = int(rng.integers(*MARGIN_RANGE))
    right = int(rng.integers(*MARGIN_RANGE))

    row_heights = [max(obj["keep_h"] for obj in row) for row in rows]
    row_widths = [sum(obj["keep_w"] for obj in row) + col_gap * (len(row) - 1) for row in rows]

    need_h = top + bottom + sum(row_heights) + row_gap * (len(rows) - 1)
    need_w = left + right + max(row_widths)
    if need_h > GRID_MAX_SIZE or need_w > GRID_MAX_SIZE:
        return None

    h_min = max(GRID_MIN_SIZE, need_h)
    w_min = max(GRID_MIN_SIZE, need_w)
    h_max = min(GRID_MAX_SIZE, h_min + EXTRA_SLACK_RANGE[1])
    w_max = min(GRID_MAX_SIZE, w_min + EXTRA_SLACK_RANGE[1])
    h = int(rng.integers(h_min, h_max + 1))
    w = int(rng.integers(w_min, w_max + 1))

    extra_v = h - need_h
    extra_h = w - need_w
    v_slots = _split_extra(rng, extra_v, len(rows) + 1)

    anchors_by_ref: dict[int, tuple[int, int]] = {}
    y = top + v_slots[0]
    for row_index, row in enumerate(rows):
        row_h = row_heights[row_index]
        row_w = row_widths[row_index]
        h_slots = _split_extra(rng, extra_h, len(row) + 1)
        x = left + h_slots[0]
        for obj_index, obj in enumerate(row):
            dy = int(rng.integers(0, row_h - obj["keep_h"] + 1))
            anchors_by_ref[id(obj)] = (int(y + dy), int(x))
            x += obj["keep_w"] + col_gap + h_slots[obj_index + 1]
        y += row_h
        if row_index < len(rows) - 1:
            y += row_gap + v_slots[row_index + 1]

    anchors = [anchors_by_ref[id(obj)] for obj in placed]
    inverse = {id(obj): idx for idx, obj in enumerate(placed)}
    restored = [None] * len(objects)
    restored_anchors: list[tuple[int, int] | None] = [None] * len(objects)
    for obj in placed:
        original_index = order[inverse[id(obj)]]
        restored[original_index] = obj
        restored_anchors[original_index] = anchors_by_ref[id(obj)]
    assert all(item is not None for item in restored_anchors)
    return h, w, [anchor for anchor in restored_anchors if anchor is not None]


def _paint_object(grid: np.ndarray, obj: dict, anchor: tuple[int, int]) -> None:
    shift_r = int(anchor[0] - obj["keep_r0"])
    shift_c = int(anchor[1] - obj["keep_c0"])
    for r, c in obj["frame_cells"]:
        grid[r + shift_r, c + shift_c] = int(obj["frame_color"])
    for r, c in obj["pattern_cells"]:
        grid[r + shift_r, c + shift_c] = int(obj["pattern_color"])


def _valid_generated_grid(grid: np.ndarray, expected_frames: int) -> bool:
    bg = _background_color(grid)
    if Counter(map(int, grid.flat))[bg] <= (grid.size // 2):
        return False
    frames = []
    for comp in _find_components(grid, bg):
        info = _frame_info(comp)
        if info is not None:
            frames.append(info)
    if len(frames) != expected_frames:
        return False
    out = main(grid)
    if out.shape != grid.shape:
        return False
    if np.count_nonzero(out != bg) != np.count_nonzero(grid != bg):
        return False
    return True


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    bg = int(rng.integers(0, 10))
    n_frames = int(rng.choice(FRAME_COUNT_CHOICES))
    palette = _pick_colors(rng, 2 * n_frames, {bg})
    directions = [DIRS[int(i)] for i in rng.permutation(len(DIRS))[:n_frames]]

    objects = []
    for idx in range(n_frames):
        objects.append(
            _sample_object(
                rng,
                directions[idx],
                palette[2 * idx],
                palette[2 * idx + 1],
            )
        )

    arranged = _arrange_objects(rng, objects)
    if arranged is None:
        return None
    h, w, anchors = arranged
    grid = np.full((h, w), bg, dtype=int)
    for obj, anchor in zip(objects, anchors):
        _paint_object(grid, obj, anchor)

    if not _valid_generated_grid(grid, n_frames):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid frame-push puzzle")
