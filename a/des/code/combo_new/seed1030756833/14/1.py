from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
SIDES = ("top", "bottom", "left", "right")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
GEN_RETRIES = 400


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _c in coords]
    cs = [c for _r, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _perimeter_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _side_cells(bbox: tuple[int, int, int, int], side: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if side == "top":
        return [(r0, c) for c in range(c0 + 1, c1)]
    if side == "bottom":
        return [(r1, c) for c in range(c0 + 1, c1)]
    if side == "left":
        return [(r, c0) for r in range(r0 + 1, r1)]
    if side == "right":
        return [(r, c1) for r in range(r0 + 1, r1)]
    raise AssertionError(f"unknown side {side}")


def _find_frame(g: np.ndarray) -> dict:
    candidates = []
    for raw_color in np.unique(g):
        color = int(raw_color)
        if color == BG:
            continue
        coords_arr = np.argwhere(g == color)
        coords = [(int(r), int(c)) for r, c in coords_arr]
        bbox = _bbox_from_coords(coords)
        r0, c0, r1, c1 = bbox
        if r1 - r0 + 1 < 6 or c1 - c0 + 1 < 6:
            continue

        pts = set(coords)
        perimeter = _perimeter_cells(bbox)
        if not pts <= perimeter:
            continue

        missing = perimeter - pts
        if not (2 <= len(missing) <= 4):
            continue

        for side in SIDES:
            side_list = _side_cells(bbox, side)
            side_set = set(side_list)
            if not missing <= side_set:
                continue
            idxs = [side_list.index(p) for p in missing]
            if max(idxs) - min(idxs) + 1 != len(idxs):
                continue
            candidates.append(
                {
                    "color": color,
                    "bbox": bbox,
                    "cells": pts,
                    "side": side,
                    "gap_cells": [side_list[i] for i in range(min(idxs), max(idxs) + 1)],
                    "area": len(pts),
                }
            )

    assert candidates, "expected one rectangular frame with a single side gap"
    candidates.sort(key=lambda x: x["area"], reverse=True)
    return candidates[0]


def _opposite_line(
    shape: tuple[int, int], bbox: tuple[int, int, int, int], gap_side: str
) -> list[tuple[int, int]]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    side = OPPOSITE[gap_side]
    if side == "top":
        assert r0 > 0
        return [(r0 - 1, c) for c in range(c0 + 1, c1)]
    if side == "bottom":
        assert r1 + 1 < h
        return [(r1 + 1, c) for c in range(c0 + 1, c1)]
    if side == "left":
        assert c0 > 0
        return [(r, c0 - 1) for r in range(r0 + 1, r1)]
    if side == "right":
        assert c1 + 1 < w
        return [(r, c1 + 1) for r in range(r0 + 1, r1)]
    raise AssertionError(f"unknown side {side}")


def _find_stripe(g: np.ndarray, frame: dict) -> tuple[int, set[tuple[int, int]]]:
    line = _opposite_line(g.shape, frame["bbox"], frame["side"])
    cells = [
        (r, c, int(g[r, c]))
        for r, c in line
        if int(g[r, c]) not in (BG, int(frame["color"]))
    ]
    assert cells, "expected a short color strip opposite the gap"
    colors = {v for _r, _c, v in cells}
    assert len(colors) == 1, "opposite strip must have one color"

    positions = [line.index((r, c)) for r, c, _v in cells]
    assert max(positions) - min(positions) + 1 == len(positions), "strip must be contiguous"
    assert 2 <= len(cells) <= 4, "strip must be short"
    return colors.pop(), {(r, c) for r, c, _v in cells}


def _components_by_color(g: np.ndarray, ignored: set[tuple[int, int]]) -> list[dict]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or (r, c) in ignored or int(g[r, c]) == BG:
                continue
            color = int(g[r, c])
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or (nr, nc) in ignored or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            r0, c0, r1, c1 = _bbox_from_coords(cells)
            comps.append(
                {
                    "color": color,
                    "cells": cells,
                    "bbox": (r0, c0, r1, c1),
                    "h": r1 - r0 + 1,
                    "w": c1 - c0 + 1,
                }
            )
    return comps


def _outside_frame(comp: dict, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return all(not (r0 <= r <= r1 and c0 <= c <= c1) for r, c in comp["cells"])


def _destination_origin(frame_bbox: tuple[int, int, int, int], side: str, gap_cells: list[tuple[int, int]], sprite: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = frame_bbox
    if side in ("top", "bottom"):
        gap_start = min(c for _r, c in gap_cells)
        assert int(sprite["w"]) == len(gap_cells)
        if side == "top":
            return r1 - int(sprite["h"]), gap_start
        return r0 + 1, gap_start

    gap_start = min(r for r, _c in gap_cells)
    assert int(sprite["h"]) == len(gap_cells)
    if side == "left":
        return gap_start, c1 - int(sprite["w"])
    return gap_start, c0 + 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push the size-matched outside stamp through the frame gap and recolor it with the opposite strip color.

    Input structure:
    - The background is black.
    - A one-pixel-thick monochrome rectangular frame has exactly one contiguous gap on one side.
    - Several small monochrome sprites sit outside the frame; exactly one sprite has the gap's required dimension.
    - A short one-pixel-wide color strip sits just outside the frame on the side opposite the gap.

    Rule:
    - Find the frame gap, its side, and the gap width.
    - Match a vertical gap to the outside sprite with the same height, or a horizontal gap to the sprite with the same width.
    - Remove that outside sprite, translate its shape through the gap until its leading bbox edge reaches the opposite inner wall, and recolor its pixels with the strip color.
    - Leave the frame, the unmatched outside sprites, and the color strip unchanged.

    Key insight:
    The gap chooses the sprite by orientation-specific size, while the opposite strip supplies the pasted sprite's color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frame = _find_frame(g)
    stripe_color, stripe_cells = _find_stripe(g, frame)
    ignored = set(frame["cells"]) | set(stripe_cells)
    sprites = [
        comp
        for comp in _components_by_color(g, ignored)
        if _outside_frame(comp, frame["bbox"])
    ]

    gap_len = len(frame["gap_cells"])
    if frame["side"] in ("left", "right"):
        matches = [s for s in sprites if int(s["h"]) == gap_len]
    else:
        matches = [s for s in sprites if int(s["w"]) == gap_len]
    assert len(matches) == 1, "expected exactly one sprite matching the gap dimension"
    sprite = matches[0]

    out = g.copy()
    for r, c in sprite["cells"]:
        out[r, c] = BG

    dst_r, dst_c = _destination_origin(frame["bbox"], frame["side"], frame["gap_cells"], sprite)
    sr0, sc0, _sr1, _sc1 = sprite["bbox"]
    fr0, fc0, fr1, fc1 = frame["bbox"]
    for r, c in sprite["cells"]:
        rr = dst_r + (r - sr0)
        cc = dst_c + (c - sc0)
        assert fr0 < rr < fr1 and fc0 < cc < fc1, "pasted sprite must land inside the frame"
        assert int(out[rr, cc]) == BG, "pasted sprite must not overwrite an existing object"
        out[rr, cc] = int(stripe_color)

    return out


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _connected_mask(h: int, w: int, span_axis: str) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    if span_axis == "height":
        c = random.randrange(w)
        for r in range(h):
            c = max(0, min(w - 1, c + random.choice((-1, 0, 1))))
            mask[r, c] = True
    else:
        r = random.randrange(h)
        for c in range(w):
            r = max(0, min(h - 1, r + random.choice((-1, 0, 1))))
            mask[r, c] = True

    target_area = random.randint(max(mask.sum() + 1, 3), max(mask.sum() + 1, min(h * w, mask.sum() + 5)))
    while int(mask.sum()) < target_area:
        filled = [(int(r), int(c)) for r, c in np.argwhere(mask)]
        r, c = random.choice(filled)
        neigh = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        random.shuffle(neigh)
        for nr, nc in neigh:
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                mask[nr, nc] = True
                break
    return _crop_mask(mask)


def _sprite_mask(required_axis: str, gap_len: int, max_other: int, avoid_match_axis: str | None = None) -> np.ndarray:
    for _ in range(100):
        other = random.randint(2, max(2, max_other))
        if required_axis == "height":
            h, w = gap_len, other
            span_axis = "height"
        else:
            h, w = other, gap_len
            span_axis = "width"
        mask = _connected_mask(h, w, span_axis)
        if required_axis == "height" and mask.shape[0] != gap_len:
            continue
        if required_axis == "width" and mask.shape[1] != gap_len:
            continue
        if avoid_match_axis == "height" and mask.shape[0] == gap_len:
            continue
        if avoid_match_axis == "width" and mask.shape[1] == gap_len:
            continue
        return mask
    raise RuntimeError("failed to build sprite mask")


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int, side: str, gap_len: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color

    side_list = _side_cells(bbox, side)
    start = random.randint(0, len(side_list) - gap_len)
    gap_cells = side_list[start : start + gap_len]
    for r, c in gap_cells:
        grid[r, c] = BG
    return gap_cells


def _draw_stripe(grid: np.ndarray, bbox: tuple[int, int, int, int], side: str, gap_cells: list[tuple[int, int]], color: int) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells: list[tuple[int, int]]
    if side == "top":
        cells = [(r1 + 1, c) for _r, c in gap_cells]
    elif side == "bottom":
        cells = [(r0 - 1, c) for _r, c in gap_cells]
    elif side == "left":
        cells = [(r, c1 + 1) for r, _c in gap_cells]
    else:
        cells = [(r, c0 - 1) for r, _c in gap_cells]
    for r, c in cells:
        grid[r, c] = color
    return set(cells)


def _expanded(bbox: tuple[int, int, int, int], pad: int, h: int, w: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _can_place_sprite(
    grid: np.ndarray,
    mask: np.ndarray,
    top: int,
    left: int,
    frame_block: tuple[int, int, int, int],
) -> bool:
    h, w = grid.shape
    mh, mw = mask.shape
    if top < 0 or left < 0 or top + mh > h or left + mw > w:
        return False
    if _rects_overlap((top, left, top + mh - 1, left + mw - 1), frame_block):
        return False
    for rr, cc in np.argwhere(mask):
        r = top + int(rr)
        c = left + int(cc)
        for nr in range(max(0, r - 1), min(h, r + 2)):
            for nc in range(max(0, c - 1), min(w, c + 2)):
                if int(grid[nr, nc]) != BG:
                    return False
    return True


def _place_sprite(
    grid: np.ndarray,
    mask: np.ndarray,
    color: int,
    frame_block: tuple[int, int, int, int],
) -> tuple[int, int] | None:
    h, w = grid.shape
    mh, mw = mask.shape
    positions = [(r, c) for r in range(h - mh + 1) for c in range(w - mw + 1)]
    random.shuffle(positions)
    for r, c in positions:
        if not _can_place_sprite(grid, mask, r, c, frame_block):
            continue
        for rr, cc in np.argwhere(mask):
            grid[r + int(rr), c + int(cc)] = color
        return r, c
    return None


def _expected_output(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    side: str,
    gap_cells: list[tuple[int, int]],
    selected_mask: np.ndarray,
    selected_loc: tuple[int, int],
    stripe_color: int,
) -> np.ndarray:
    out = grid.copy()
    sr, sc = selected_loc
    for rr, cc in np.argwhere(selected_mask):
        out[sr + int(rr), sc + int(cc)] = BG

    sprite = {
        "bbox": (sr, sc, sr + selected_mask.shape[0] - 1, sc + selected_mask.shape[1] - 1),
        "h": selected_mask.shape[0],
        "w": selected_mask.shape[1],
    }
    dst_r, dst_c = _destination_origin(frame_bbox, side, gap_cells, sprite)
    for rr, cc in np.argwhere(selected_mask):
        out[dst_r + int(rr), dst_c + int(cc)] = stripe_color
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(20, 30)
    w = random.randint(20, 30)
    side = random.choice(SIDES)
    gap_len = random.randint(2, 4)

    frame_h = random.randint(max(7, gap_len + 4), min(13, h - 7))
    frame_w = random.randint(max(7, gap_len + 4), min(13, w - 7))
    r0 = random.randint(3, h - frame_h - 3)
    c0 = random.randint(3, w - frame_w - 3)
    frame_bbox = (r0, c0, r0 + frame_h - 1, c0 + frame_w - 1)
    inner_h = frame_h - 2
    inner_w = frame_w - 2

    colors = list(range(1, 10))
    random.shuffle(colors)
    frame_color = colors[0]
    stripe_color = colors[1]
    num_sprites = random.randint(3, 6)
    sprite_colors = colors[2 : 2 + num_sprites]
    if len(sprite_colors) < num_sprites:
        return None

    grid = np.full((h, w), BG, dtype=int)
    gap_cells = _draw_frame(grid, frame_bbox, frame_color, side, gap_len)
    _draw_stripe(grid, frame_bbox, side, gap_cells, stripe_color)

    if side in ("left", "right"):
        selected_mask = _sprite_mask("height", gap_len, min(5, inner_w))
        distractor_required = "width"
        avoid_axis = "height"
        distractor_max_other = 5
    else:
        selected_mask = _sprite_mask("width", gap_len, min(5, inner_h))
        distractor_required = "height"
        avoid_axis = "width"
        distractor_max_other = 5

    frame_block = _expanded(frame_bbox, 1, h, w)
    selected_loc = _place_sprite(grid, selected_mask, sprite_colors[0], frame_block)
    if selected_loc is None:
        return None

    for color in sprite_colors[1:]:
        try:
            mask = _sprite_mask(distractor_required, gap_len, distractor_max_other, avoid_axis)
        except RuntimeError:
            return None
        if _place_sprite(grid, mask, color, frame_block) is None:
            return None

    expected = _expected_output(
        grid, frame_bbox, side, gap_cells, selected_mask, selected_loc, stripe_color
    )
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
