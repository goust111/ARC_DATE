import random
from typing import Any

import numpy as np


FRAME_COLOR = 5
ARROW_COLOR = 2
PALETTE = tuple(range(10))
MAX_GENERATION_ATTEMPTS = 400


def _dominant_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _find_frames(grid: np.ndarray, bg: int) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for comp in _components(grid == FRAME_COLOR):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 5 or w < 5:
            continue
        if len(comp) != 2 * (h + w) - 4:
            continue
        ok = True
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                on_border = r in (r0, r1) or c in (c0, c1)
                if on_border:
                    if int(grid[r, c]) != FRAME_COLOR:
                        ok = False
                        break
                elif int(grid[r, c]) != bg:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            frames.append({"bbox": (r0, c0, r1, c1)})
    frames.sort(key=lambda item: item["bbox"])
    return frames


def _detect_arrow(grid: np.ndarray, frame: dict[str, Any]) -> tuple[str, list[tuple[int, int]]]:
    h, w = grid.shape
    r0, c0, r1, c1 = frame["bbox"]
    matches: list[tuple[str, list[tuple[int, int]]]] = []

    if r0 >= 2:
        for c in range(c0 + 2, c1 - 1):
            cells = [(r0 - 1, c), (r0 - 2, c - 1), (r0 - 2, c + 1)]
            if all(0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == ARROW_COLOR for rr, cc in cells):
                matches.append(("down", cells))

    if r1 + 2 < h:
        for c in range(c0 + 2, c1 - 1):
            cells = [(r1 + 1, c), (r1 + 2, c - 1), (r1 + 2, c + 1)]
            if all(0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == ARROW_COLOR for rr, cc in cells):
                matches.append(("up", cells))

    if c0 >= 2:
        for r in range(r0 + 2, r1 - 1):
            cells = [(r, c0 - 1), (r - 1, c0 - 2), (r + 1, c0 - 2)]
            if all(0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == ARROW_COLOR for rr, cc in cells):
                matches.append(("right", cells))

    if c1 + 2 < w:
        for r in range(r0 + 2, r1 - 1):
            cells = [(r, c1 + 1), (r - 1, c1 + 2), (r + 1, c1 + 2)]
            if all(0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == ARROW_COLOR for rr, cc in cells):
                matches.append(("left", cells))

    assert len(matches) == 1, "each frame must have exactly one arrow"
    return matches[0]


def _collect_stripes(grid: np.ndarray, frame: dict[str, Any], direction: str, bg: int) -> list[dict[str, Any]]:
    h, w = grid.shape
    r0, c0, r1, c1 = frame["bbox"]
    stripes: list[dict[str, Any]] = []

    if direction in ("down", "up"):
        start_c = c0 + 1
        row = r0 - 3 if direction == "down" else r1 + 3
        step = -1 if direction == "down" else 1
        while 0 <= row < h and int(grid[row, start_c]) != bg:
            color = int(grid[row, start_c])
            assert color not in (bg, FRAME_COLOR, ARROW_COLOR)
            cells: list[tuple[int, int]] = []
            c = start_c
            while c < w and int(grid[row, c]) == color:
                cells.append((row, c))
                c += 1
            stripes.append({"color": color, "length": len(cells), "cells": cells})
            row += step
    else:
        start_r = r0 + 1
        col = c0 - 3 if direction == "right" else c1 + 3
        step = -1 if direction == "right" else 1
        while 0 <= col < w and int(grid[start_r, col]) != bg:
            color = int(grid[start_r, col])
            assert color not in (bg, FRAME_COLOR, ARROW_COLOR)
            cells: list[tuple[int, int]] = []
            r = start_r
            while r < h and int(grid[r, col]) == color:
                cells.append((r, col))
                r += 1
            stripes.append({"color": color, "length": len(cells), "cells": cells})
            col += step

    assert stripes, "each frame must have at least one stripe"
    return stripes


def _apply_rule(grid: np.ndarray, frames: list[dict[str, Any]], bg: int) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        direction = frame["direction"]
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1

        for rr, cc in frame["arrow_cells"]:
            out[rr, cc] = bg
        for stripe in frame["stripes"]:
            for rr, cc in stripe["cells"]:
                out[rr, cc] = bg

        if direction == "down":
            for i, stripe in enumerate(frame["stripes"][:inner_h]):
                fill = min(int(stripe["length"]), inner_w)
                out[r0 + 1 + i, c0 + 1 : c0 + 1 + fill] = int(stripe["color"])
        elif direction == "up":
            for i, stripe in enumerate(frame["stripes"][:inner_h]):
                fill = min(int(stripe["length"]), inner_w)
                out[r1 - 1 - i, c0 + 1 : c0 + 1 + fill] = int(stripe["color"])
        elif direction == "right":
            for i, stripe in enumerate(frame["stripes"][:inner_w]):
                fill = min(int(stripe["length"]), inner_h)
                out[r0 + 1 : r0 + 1 + fill, c0 + 1 + i] = int(stripe["color"])
        else:
            for i, stripe in enumerate(frame["stripes"][:inner_w]):
                fill = min(int(stripe["length"]), inner_h)
                out[r0 + 1 : r0 + 1 + fill, c1 - 1 - i] = int(stripe["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push colored bands into adjacent hollow frames following the nearby arrow direction.

    Input structure:
    - The grid has one dominant background color.
    - One or more grey one-pixel-thick hollow rectangles sit on that background.
    - Outside one side of each rectangle there is a three-pixel red arrow pointing into the frame, and beyond it a stack of parallel colored horizontal or vertical one-pixel-wide bands.

    Rule:
    - Detect each grey frame, its adjacent red arrow, and the band stack on the arrow's outer side.
    - The arrow direction tells which way the bands move into the frame, and the bands are ordered from nearest to farthest from the frame wall.
    - Move the nearest band onto the first interior row or column, the next band onto the second, and so on; each moved band keeps its color and length, is truncated if longer than the interior span perpendicular to the push, and is aligned to the top or left start of that span if shorter.
    - Erase the original exterior bands and the arrow, leaving the grey frame in place.

    Key insight:
    - The arrow gives both the push direction and the distance axis used to sort the stacked bands before folding them into the frame interior.
    """
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    frames = _find_frames(g, bg)
    assert frames, "expected at least one grey hollow frame"
    for frame in frames:
        direction, arrow_cells = _detect_arrow(g, frame)
        stripes = _collect_stripes(g, frame, direction, bg)
        frame["direction"] = direction
        frame["arrow_cells"] = arrow_cells
        frame["stripes"] = stripes
    return _apply_rule(g, frames, bg)


def _paint_frame(arr: np.ndarray, top: int, left: int) -> None:
    h, w = arr.shape
    arr[top : top + h, left] = FRAME_COLOR
    arr[top : top + h, left + w - 1] = FRAME_COLOR
    arr[top, left : left + w] = FRAME_COLOR
    arr[top + h - 1, left : left + w] = FRAME_COLOR


def _make_object(bg: int) -> tuple[np.ndarray, dict[str, Any]]:
    direction = random.choice(("down", "up", "right", "left"))
    inner_h = random.randint(3, 7)
    inner_w = random.randint(3, 7)
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    depth = inner_h if direction in ("down", "up") else inner_w
    stripe_count = random.randint(1, min(depth, 6))

    span = inner_w if direction in ("down", "up") else inner_h
    lengths: list[int] = []
    for _ in range(stripe_count):
        branch = random.randrange(3)
        if branch == 0:
            lengths.append(random.randint(1, span - 1))
        elif branch == 1:
            lengths.append(span)
        else:
            lengths.append(span + random.randint(1, 3))

    stripe_colors = random.sample(
        [c for c in PALETTE if c not in (bg, FRAME_COLOR, ARROW_COLOR)],
        stripe_count,
    )
    max_len = max(lengths)

    if direction == "down":
        frame_top = stripe_count + 2
        frame_left = 0
        obj_h = frame_top + outer_h
        obj_w = max(outer_w, 1 + max_len)
        tip_r = frame_top - 1
        tip_c = frame_left + random.randint(2, inner_w - 1)
        stripe_anchor = [(frame_top - 3 - i, frame_left + 1) for i in range(stripe_count)]
        arrow_cells = [(tip_r, tip_c), (tip_r - 1, tip_c - 1), (tip_r - 1, tip_c + 1)]
    elif direction == "up":
        frame_top = 0
        frame_left = 0
        obj_h = outer_h + stripe_count + 2
        obj_w = max(outer_w, 1 + max_len)
        tip_r = frame_top + outer_h
        tip_c = frame_left + random.randint(2, inner_w - 1)
        stripe_anchor = [(frame_top + outer_h + 2 + i, frame_left + 1) for i in range(stripe_count)]
        arrow_cells = [(tip_r, tip_c), (tip_r + 1, tip_c - 1), (tip_r + 1, tip_c + 1)]
    elif direction == "right":
        frame_top = 0
        frame_left = stripe_count + 2
        obj_h = max(outer_h, 1 + max_len)
        obj_w = frame_left + outer_w
        tip_r = frame_top + random.randint(2, inner_h - 1)
        tip_c = frame_left - 1
        stripe_anchor = [(frame_top + 1, frame_left - 3 - i) for i in range(stripe_count)]
        arrow_cells = [(tip_r, tip_c), (tip_r - 1, tip_c - 1), (tip_r + 1, tip_c - 1)]
    else:
        frame_top = 0
        frame_left = 0
        obj_h = max(outer_h, 1 + max_len)
        obj_w = outer_w + stripe_count + 2
        tip_r = frame_top + random.randint(2, inner_h - 1)
        tip_c = frame_left + outer_w
        stripe_anchor = [(frame_top + 1, frame_left + outer_w + 2 + i) for i in range(stripe_count)]
        arrow_cells = [(tip_r, tip_c), (tip_r - 1, tip_c + 1), (tip_r + 1, tip_c + 1)]

    obj = np.full((obj_h, obj_w), bg, dtype=int)
    _paint_frame(obj[frame_top : frame_top + outer_h, frame_left : frame_left + outer_w], 0, 0)
    for rr, cc in arrow_cells:
        obj[rr, cc] = ARROW_COLOR

    stripes: list[dict[str, Any]] = []
    for i, (length, color) in enumerate(zip(lengths, stripe_colors)):
        sr, sc = stripe_anchor[i]
        cells: list[tuple[int, int]] = []
        if direction in ("down", "up"):
            for dc in range(length):
                rr, cc = sr, sc + dc
                obj[rr, cc] = color
                cells.append((rr, cc))
        else:
            for dr in range(length):
                rr, cc = sr + dr, sc
                obj[rr, cc] = color
                cells.append((rr, cc))
        stripes.append({"color": color, "length": length, "cells": cells})

    frame_bbox = (frame_top, frame_left, frame_top + outer_h - 1, frame_left + outer_w - 1)
    meta = {
        "bbox": frame_bbox,
        "direction": direction,
        "arrow_cells": arrow_cells,
        "stripes": stripes,
    }
    return obj, meta


def _shift_meta(meta: dict[str, Any], dr: int, dc: int) -> dict[str, Any]:
    r0, c0, r1, c1 = meta["bbox"]
    shifted = {
        "bbox": (r0 + dr, c0 + dc, r1 + dr, c1 + dc),
        "direction": meta["direction"],
        "arrow_cells": [(r + dr, c + dc) for r, c in meta["arrow_cells"]],
        "stripes": [],
    }
    for stripe in meta["stripes"]:
        shifted["stripes"].append(
            {
                "color": stripe["color"],
                "length": stripe["length"],
                "cells": [(r + dr, c + dc) for r, c in stripe["cells"]],
            }
        )
    return shifted


def _place_objects(bg: int, objects: list[tuple[np.ndarray, dict[str, Any]]]) -> tuple[np.ndarray, list[dict[str, Any]]] | None:
    max_h = max(obj.shape[0] for obj, _ in objects)
    max_w = max(obj.shape[1] for obj, _ in objects)
    padded_area = sum((obj.shape[0] + 2) * (obj.shape[1] + 2) for obj, _ in objects)
    side_floor = min(30, max(18, int(np.ceil(np.sqrt(padded_area))) + 2))
    grid_h = random.randint(max(side_floor, max_h + 2), 30)
    grid_w = random.randint(max(side_floor, max_w + 2), 30)

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    placed: list[dict[str, Any]] = []

    for obj, meta in sorted(objects, key=lambda item: item[0].shape[0] * item[0].shape[1], reverse=True):
        oh, ow = obj.shape
        candidates: list[tuple[int, int]] = []
        for _ in range(250):
            top = random.randint(0, grid_h - oh)
            left = random.randint(0, grid_w - ow)
            r0 = max(0, top - 1)
            r1 = min(grid_h, top + oh + 1)
            c0 = max(0, left - 1)
            c1 = min(grid_w, left + ow + 1)
            if occupied[r0:r1, c0:c1].any():
                continue
            candidates.append((top, left))
        if not candidates:
            return None

        top, left = random.choice(candidates)
        mask = obj != bg
        sub = grid[top : top + oh, left : left + ow]
        sub[mask] = obj[mask]
        occupied[top : top + oh, left : left + ow] = True
        placed.append(_shift_meta(meta, top, left))

    return grid, placed


def _build_expected(inp: np.ndarray, frames: list[dict[str, Any]], bg: int) -> np.ndarray:
    return _apply_rule(inp, frames, bg)


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice([c for c in PALETTE if c not in (FRAME_COLOR, ARROW_COLOR)])
    object_count = random.randint(1, 3)
    objects = [_make_object(bg) for _ in range(object_count)]
    placed = _place_objects(bg, objects)
    if placed is None:
        return None

    inp, frames = placed
    expected = _build_expected(inp, frames, bg)
    try:
        actual = main(inp)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
