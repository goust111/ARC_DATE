import random
from typing import Iterable

import numpy as np


BG = 0
FRAME_COLOR = 5
ARROW_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 400

INNER_H_RANGE = (3, 8)
INNER_W_RANGE = (3, 8)
MARGIN_RANGE = (2, 9)


def _find_frame(grid: np.ndarray) -> tuple[int, tuple[int, int, int, int]]:
    candidates: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != BG):
        pts = np.argwhere(grid == color)
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        if int(r1 - r0) < 2 or int(c1 - c0) < 2:
            continue
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        if int(pts.shape[0]) != 2 * (height + width) - 4:
            continue
        if np.any(grid[r0, c0 : c1 + 1] != color):
            continue
        if np.any(grid[r1, c0 : c1 + 1] != color):
            continue
        if np.any(grid[r0 : r1 + 1, c0] != color):
            continue
        if np.any(grid[r0 : r1 + 1, c1] != color):
            continue
        if r1 - r0 > 1 and c1 - c0 > 1 and np.any(grid[r0 + 1 : r1, c0 + 1 : c1] == color):
            continue
        area = height * width
        candidates.append((area, color, (int(r0), int(c0), int(r1), int(c1))))

    assert candidates, "main: no single-pixel-thick rectangular frame found"
    _, color, bbox = max(candidates, key=lambda item: item[0])
    return color, bbox


def _bar_length(grid: np.ndarray, start_r: int, start_c: int, dr: int, dc: int, color: int) -> int:
    h, w = grid.shape
    length = 0
    r = start_r
    c = start_c
    while 0 <= r < h and 0 <= c < w and int(grid[r, c]) == color:
        length += 1
        r += dr
        c += dc
    return length


def _collect_specs(grid: np.ndarray, frame_color: int, bbox: tuple[int, int, int, int]) -> list[dict[str, int]]:
    r0, c0, r1, c1 = bbox
    specs: list[dict[str, int]] = []

    if r0 > 0:
        for c in range(c0 + 1, c1):
            color = int(grid[r0 - 1, c])
            if color not in (BG, frame_color):
                specs.append(
                    {
                        "side": "top",
                        "coord": c,
                        "color": color,
                        "length": _bar_length(grid, r0 - 2, c, -1, 0, color),
                    }
                )
    if c0 > 0:
        for r in range(r0 + 1, r1):
            color = int(grid[r, c0 - 1])
            if color not in (BG, frame_color):
                specs.append(
                    {
                        "side": "left",
                        "coord": r,
                        "color": color,
                        "length": _bar_length(grid, r, c0 - 2, 0, -1, color),
                    }
                )
    if r1 + 1 < grid.shape[0]:
        for c in range(c0 + 1, c1):
            color = int(grid[r1 + 1, c])
            if color not in (BG, frame_color):
                specs.append(
                    {
                        "side": "bottom",
                        "coord": c,
                        "color": color,
                        "length": _bar_length(grid, r1 + 2, c, 1, 0, color),
                    }
                )
    if c1 + 1 < grid.shape[1]:
        for r in range(r0 + 1, r1):
            color = int(grid[r, c1 + 1])
            if color not in (BG, frame_color):
                specs.append(
                    {
                        "side": "right",
                        "coord": r,
                        "color": color,
                        "length": _bar_length(grid, r, c1 + 2, 0, 1, color),
                    }
                )

    return _sort_specs(specs)


def _sort_specs(specs: Iterable[dict[str, int]]) -> list[dict[str, int]]:
    side_order = {"top": 0, "left": 1, "bottom": 2, "right": 3}
    return sorted(specs, key=lambda spec: (side_order[spec["side"]], spec["coord"]))


def _paint_from_specs(
    shape: tuple[int, int],
    frame_color: int,
    bbox: tuple[int, int, int, int],
    specs: Iterable[dict[str, int]],
) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    out = np.full(shape, BG, dtype=int)
    out[r0, c0 : c1 + 1] = frame_color
    out[r1, c0 : c1 + 1] = frame_color
    out[r0 : r1 + 1, c0] = frame_color
    out[r0 : r1 + 1, c1] = frame_color

    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1

    for spec in _sort_specs(specs):
        side = spec["side"]
        color = int(spec["color"])
        length = int(spec["length"])
        if length <= 0:
            continue
        if side == "top":
            path = [(r, spec["coord"]) for r in range(r0 + 1, r1)]
            modulo = inner_h
        elif side == "bottom":
            path = [(r, spec["coord"]) for r in range(r1 - 1, r0, -1)]
            modulo = inner_h
        elif side == "left":
            path = [(spec["coord"], c) for c in range(c0 + 1, c1)]
            modulo = inner_w
        else:
            path = [(spec["coord"], c) for c in range(c1 - 1, c0, -1)]
            modulo = inner_w

        if modulo <= 0:
            continue
        for i in range(length):
            rr, cc = path[i % modulo]
            out[rr, cc] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push each colored data bar into a grey rectangular chamber along its matching arrow direction.

    Input structure:
    - The background is black.
    - A single grey, one-pixel-thick rectangular frame encloses an empty black interior.
    - Just outside the four sides sit single-pixel colored arrows; behind each arrow, a same-colored row or column bar extends farther away from the frame.

    Rule:
    - Detect the grey frame, then read each adjacent arrow to determine whether its bar pushes right, left, down, or up into the aligned interior row or column.
    - Count the contiguous same-colored bar pixels behind the arrow and start painting from the nearest interior cell on that row or column.
    - When the paint reaches the opposite inner wall, wrap to the other end of that same interior row or column and continue until the whole bar length has been consumed.
    - Remove every exterior arrow and data bar, keep the grey frame, and let later arrows overwrite earlier ones in this order: top side left-to-right, left side top-to-bottom, bottom side left-to-right, right side top-to-bottom.

    Key insight:
    - Each arrow controls one cyclic interior row or column inside the frame; solving the task is mostly about locating the frame and replaying the prescribed overwrite order.
    """
    g = np.asarray(grid, dtype=int)
    frame_color, bbox = _find_frame(g)
    specs = _collect_specs(g, frame_color, bbox)
    return _paint_from_specs(g.shape, frame_color, bbox, specs)


def _row_slice_for_side(side: str, bbox: tuple[int, int, int, int], index: int) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if side == "top":
        return r0 - 1, index
    if side == "bottom":
        return r1 + 1, index
    if side == "left":
        return index, c0 - 1
    return index, c1 + 1


def _capacity_for_side(side: str, bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> int:
    h, w = shape
    r0, c0, r1, c1 = bbox
    if side == "top":
        return r0 - 1
    if side == "bottom":
        return h - r1 - 2
    if side == "left":
        return c0 - 1
    return w - c1 - 2


def _line_length_for_side(side: str, bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    if side in ("top", "bottom"):
        return r1 - r0 - 1
    return c1 - c0 - 1


def _place_bar(grid: np.ndarray, bbox: tuple[int, int, int, int], spec: dict[str, int]) -> None:
    side = spec["side"]
    coord = spec["coord"]
    color = int(spec["color"])
    length = int(spec["length"])
    r0, c0, r1, c1 = bbox

    ar, ac = _row_slice_for_side(side, bbox, coord)
    grid[ar, ac] = color

    if side == "top":
        for k in range(1, length + 1):
            grid[r0 - 1 - k, coord] = color
    elif side == "bottom":
        for k in range(1, length + 1):
            grid[r1 + 1 + k, coord] = color
    elif side == "left":
        for k in range(1, length + 1):
            grid[coord, c0 - 1 - k] = color
    else:
        for k in range(1, length + 1):
            grid[coord, c1 + 1 + k] = color


def _choose_lengths(
    rng: random.Random,
    specs: list[dict[str, int]],
    bbox: tuple[int, int, int, int],
    shape: tuple[int, int],
) -> bool:
    long_candidates = []
    for i, spec in enumerate(specs):
        capacity = _capacity_for_side(spec["side"], bbox, shape)
        line_len = _line_length_for_side(spec["side"], bbox)
        spec["capacity"] = capacity
        spec["line_len"] = line_len
        if capacity > line_len:
            long_candidates.append(i)

    if not long_candidates:
        return False

    force_long = rng.choice(long_candidates)
    force_short = rng.choice([i for i in range(len(specs)) if i != force_long])

    for i, spec in enumerate(specs):
        capacity = spec["capacity"]
        line_len = spec["line_len"]
        modes = ["short"]
        if capacity > line_len:
            modes.append("long")
        if i == force_long:
            mode = "long"
        elif i == force_short:
            mode = "short"
        else:
            mode = rng.choice(modes)

        if mode == "long":
            spec["length"] = rng.randint(line_len + 1, capacity)
        else:
            spec["length"] = rng.randint(1, min(line_len, capacity))

    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    inner_h = rng.randint(*INNER_H_RANGE)
    inner_w = rng.randint(*INNER_W_RANGE)
    top_margin = rng.randint(*MARGIN_RANGE)
    bottom_margin = rng.randint(*MARGIN_RANGE)
    left_margin = rng.randint(*MARGIN_RANGE)
    right_margin = rng.randint(*MARGIN_RANGE)

    h = top_margin + inner_h + 2 + bottom_margin
    w = left_margin + inner_w + 2 + right_margin
    if max(h, w) > 30:
        return None

    bbox = (
        top_margin,
        left_margin,
        top_margin + inner_h + 1,
        left_margin + inner_w + 1,
    )
    r0, c0, r1, c1 = bbox
    inner_rows = list(range(r0 + 1, r1))
    inner_cols = list(range(c0 + 1, c1))

    counts = {
        "top": rng.randint(1, min(2, len(inner_cols))),
        "bottom": rng.randint(1, min(2, len(inner_cols))),
        "left": rng.randint(1, min(2, len(inner_rows))),
        "right": rng.randint(1, min(2, len(inner_rows))),
    }
    total_arrows = sum(counts.values())
    if total_arrows > len(ARROW_COLORS):
        return None

    colors = rng.sample(ARROW_COLORS, total_arrows)
    specs: list[dict[str, int]] = []
    color_idx = 0

    side_positions = {
        "top": sorted(rng.sample(inner_cols, counts["top"])),
        "bottom": sorted(rng.sample(inner_cols, counts["bottom"])),
        "left": sorted(rng.sample(inner_rows, counts["left"])),
        "right": sorted(rng.sample(inner_rows, counts["right"])),
    }

    for side in ("top", "bottom", "left", "right"):
        for coord in side_positions[side]:
            specs.append({"side": side, "coord": coord, "color": colors[color_idx], "length": 0})
            color_idx += 1

    if not _choose_lengths(rng, specs, bbox, (h, w)):
        return None

    grid = np.full((h, w), BG, dtype=int)
    grid[r0, c0 : c1 + 1] = FRAME_COLOR
    grid[r1, c0 : c1 + 1] = FRAME_COLOR
    grid[r0 : r1 + 1, c0] = FRAME_COLOR
    grid[r0 : r1 + 1, c1] = FRAME_COLOR

    for spec in specs:
        _place_bar(grid, bbox, spec)

    expected = _paint_from_specs(grid.shape, FRAME_COLOR, bbox, specs)
    try:
        produced = main(grid)
    except Exception:
        return None

    if not np.array_equal(produced, expected):
        return None
    if np.array_equal(grid, produced):
        return None
    if len(_collect_specs(grid, FRAME_COLOR, bbox)) != len(specs):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to create a valid sample")
