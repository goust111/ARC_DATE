from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
TEAL = 8
MAROON = 9

FRAME_COLOR = MAROON
MASK_TO_STAMP = {
    GREY: YELLOW,
    GREEN: BLUE,
    TEAL: RED,
}
PAIR_COLORS = (BLUE, RED, YELLOW)
GEN_RETRIES = 800


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _frame_mask() -> np.ndarray:
    mask = np.ones((5, 5), dtype=bool)
    mask[1:4, 1:4] = False
    return mask


def _extract_stamp(grid: np.ndarray) -> np.ndarray:
    assert grid.ndim == 2, "expected a 2D grid"
    assert grid.shape[0] >= 5 and grid.shape[1] >= 5, "grid must contain the 5x5 mask frame"
    frame = grid[:5, :5]
    fm = _frame_mask()
    assert np.all(frame[fm] == FRAME_COLOR), "top-left 5x5 border must be maroon"

    mask = frame[1:4, 1:4]
    mask_colors = colors_present(mask, exclude={BLACK})
    assert len(mask_colors) == 1, "3x3 mask must use exactly one foreground color"
    mask_color = mask_colors[0]
    assert mask_color in MASK_TO_STAMP, "mask color must be grey, green, or teal"

    foreground_count = int(np.count_nonzero(mask == mask_color))
    assert 2 <= foreground_count <= 4, "mask must contain 2-4 foreground cells"

    stamp = np.zeros((3, 3), dtype=int)
    stamp[mask == BLACK] = MASK_TO_STAMP[mask_color]
    return stamp


def _pair_rectangles(grid: np.ndarray) -> list[dict]:
    work = np.asarray(grid, dtype=int).copy()
    work[:5, :5] = BLACK
    bg = get_bg(work)
    assert bg == BLACK, "background must remain black after erasing the frame"

    rects: list[dict] = []
    occupied = np.zeros(work.shape, dtype=bool)
    for color in colors_present(work, exclude={BLACK}):
        assert color in PAIR_COLORS, "only blue, red, and yellow pair markers may appear outside the frame"
        coords = np.argwhere(work == color)
        assert coords.shape == (2, 2), "each pair color must appear exactly twice"
        r0, c0, r1, c1 = bbox_coords(coords)
        assert r0 != r1 and c0 != c1, "pair markers must be strictly diagonal"
        marker_set = {tuple(map(int, pt)) for pt in coords.tolist()}
        corners1 = {(r0, c0), (r1, c1)}
        corners2 = {(r0, c1), (r1, c0)}
        assert marker_set in (corners1, corners2), "markers must occupy opposite bbox corners"
        assert c0 >= 5 or r0 >= 5, "rectangles may not overlap the top-left mask frame"

        mask = np.zeros(work.shape, dtype=bool)
        mask[r0 : r1 + 1, c0 : c1 + 1] = True
        assert not np.any(occupied & mask), "pair rectangles must not overlap"
        occupied |= mask

        rects.append(
            {
                "color": int(color),
                "bbox": (r0, c0, r1, c1),
                "center": ((r0 + r1) // 2, (c0 + c1) // 2),
                "area": (r1 - r0 + 1) * (c1 - c0 + 1),
            }
        )

    assert 2 <= len(rects) <= 3, "expected 2-3 color pairs"
    return rects


def _stamp_order(rects: list[dict]) -> list[dict]:
    return sorted(rects, key=lambda rect: (-rect["area"], rect["bbox"][0], rect["bbox"][1]))


def _overlay_stamp(out: np.ndarray, stamp: np.ndarray, center: tuple[int, int]) -> None:
    cr, cc = center
    for sr in range(3):
        for sc in range(3):
            color = int(stamp[sr, sc])
            if color == BLACK:
                continue
            rr = cr + sr - 1
            cc2 = cc + sc - 1
            if in_bounds(out.shape, rr, cc2):
                out[rr, cc2] = color


def _solve(grid: np.ndarray, rect_order: list[dict] | None = None) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    stamp = _extract_stamp(g)
    rects = _pair_rectangles(g)

    out = g.copy()
    out[:5, :5] = BLACK
    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = rect["color"]

    for rect in (_stamp_order(rects) if rect_order is None else rect_order):
        _overlay_stamp(out, stamp, rect["center"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn the inverted 3x3 mask into a stamp, then overlay it on every same-color marker rectangle.

    Input structure:
    - A black background contains a maroon 5x5 frame fixed at the top-left corner.
    - Inside that frame is a 3x3 single-color mask using grey, green, or teal on black.
    - Outside the frame, 2-3 colors from blue, red, and yellow each appear exactly twice as isolated diagonal corner markers.

    Rule:
    - Invert the 3x3 mask: original foreground cells become black, original black cells become the paired stamp color
      (grey->yellow, green->blue, teal->red).
    - Erase the entire maroon frame region to black.
    - For each outside color, fill the inclusive axis-aligned rectangle defined by its two markers with that same color.
    - Compute each rectangle center with floor division, align the stamp center onto it, and overlay only the stamp's non-black cells.
    - Process stamp overlays from larger rectangle area to smaller; ties break by top-left row then top-left column, and later overlays win.

    Key insight:
    - The framed mask does not directly produce the final output image; it only defines a transparent stamp template, while each color pair
      simultaneously determines both a filled base rectangle and the anchor point where that stamp must be centered.
    """
    return _solve(grid)


def _rect_center(rect: dict) -> tuple[int, int]:
    return tuple(map(int, rect["center"]))


def _rect_area(rect: dict) -> int:
    r0, c0, r1, c1 = rect["bbox"]
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _render_base(shape: tuple[int, int], rects: list[dict]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = rect["color"]
    return out


def _render_with_order(shape: tuple[int, int], stamp: np.ndarray, rects: list[dict], order: list[dict]) -> np.ndarray:
    out = _render_base(shape, rects)
    for rect in order:
        _overlay_stamp(out, stamp, rect["center"])
    return out


def _stamp_changes_base(shape: tuple[int, int], stamp: np.ndarray, rects: list[dict]) -> bool:
    return not np.array_equal(_render_base(shape, rects), _render_with_order(shape, stamp, rects, _stamp_order(rects)))


def _has_clip(shape: tuple[int, int], stamp: np.ndarray, rects: list[dict]) -> bool:
    nonzero = np.argwhere(stamp != BLACK)
    for rect in rects:
        cr, cc = _rect_center(rect)
        for sr, sc in nonzero:
            rr = cr + int(sr) - 1
            cc2 = cc + int(sc) - 1
            if not in_bounds(shape, rr, cc2):
                return True
    return False


def _has_equal_area(rects: list[dict]) -> bool:
    areas = [_rect_area(rect) for rect in rects]
    return len(set(areas)) < len(areas)


def _footprints_overlap(stamp: np.ndarray, rects: list[dict]) -> bool:
    offsets = {(int(r) - 1, int(c) - 1) for r, c in np.argwhere(stamp != BLACK)}
    seen: set[tuple[int, int]] = set()
    for rect in rects:
        cr, cc = _rect_center(rect)
        cells = {(cr + dr, cc + dc) for dr, dc in offsets}
        if seen & cells:
            return True
        seen |= cells
    return False


def _opposite_corners(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    if random.random() < 0.5:
        return [(r0, c0), (r1, c1)]
    return [(r0, c1), (r1, c0)]


def _bbox_overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _random_mask() -> tuple[int, np.ndarray]:
    mask_color = random.choice(list(MASK_TO_STAMP))
    while True:
        k = random.randint(2, 4)
        cells = random.sample(range(9), k)
        mask = np.zeros((3, 3), dtype=int)
        for idx in cells:
            mask[idx // 3, idx % 3] = mask_color
        stamp = np.zeros((3, 3), dtype=int)
        stamp[mask == BLACK] = MASK_TO_STAMP[mask_color]
        if 2 <= np.count_nonzero(mask) <= 4 and 5 <= np.count_nonzero(stamp) <= 7:
            return mask_color, mask


def _candidate_rect(mode: str, h: int, w: int, existing: list[dict], idx: int) -> tuple[int, int, int, int]:
    if mode == "clip" and idx == 0:
        side_h = random.choice([2, 3])
        side_w = random.choice([2, 3])
        edge = random.choice(["top", "bottom", "left", "right"])
        if edge == "top":
            r0 = 0
            c0 = random.randint(6, w - side_w)
        elif edge == "bottom":
            r0 = h - side_h
            c0 = random.randint(5, w - side_w)
        elif edge == "left":
            r0 = random.randint(6, h - side_h)
            c0 = 0
        else:
            r0 = random.randint(5, h - side_h)
            c0 = w - side_w
        return r0, c0, r0 + side_h - 1, c0 + side_w - 1

    if mode == "crowded" and existing:
        anchor = random.choice(existing)
        ac_r, ac_c = _rect_center(anchor)
        side_h = random.choice([2, 2, 3])
        side_w = random.choice([2, 2, 3])
        target_r = ac_r + random.choice([-2, -1, 1, 2])
        target_c = ac_c + random.choice([-2, -1, 1, 2])
        r0 = max(0, min(h - side_h, target_r))
        c0 = max(0, min(w - side_w, target_c))
        return r0, c0, r0 + side_h - 1, c0 + side_w - 1

    side_h = random.randint(2, min(6, h - 1))
    side_w = random.randint(2, min(6, w - 1))
    r0 = random.randint(0, h - side_h)
    c0 = random.randint(0, w - side_w)
    return r0, c0, r0 + side_h - 1, c0 + side_w - 1


def _choose_rectangles(shape: tuple[int, int], mode: str, colors: list[int]) -> list[dict] | None:
    h, w = shape
    rects: list[dict] = []
    frame_bbox = (0, 0, 4, 4)

    for idx, color in enumerate(colors):
        placed = False
        for _ in range(250):
            if mode == "equal_area" and idx == 1:
                first = rects[0]["bbox"]
                side_h = first[2] - first[0] + 1
                side_w = first[3] - first[1] + 1
                if side_h > h or side_w > w:
                    return None
                if random.random() < 0.7:
                    c0 = max(0, min(w - side_w, first[1] + random.choice([-2, -1, 1, 2])))
                    r0 = max(0, min(h - side_h, first[0] + random.choice([-2, -1, 1, 2])))
                    cand = (r0, c0, r0 + side_h - 1, c0 + side_w - 1)
                else:
                    cand = _candidate_rect(mode, h, w, rects, idx)
            else:
                cand = _candidate_rect(mode, h, w, rects, idx)

            if _bbox_overlaps(cand, frame_bbox):
                continue
            if any(_bbox_overlaps(cand, rect["bbox"]) for rect in rects):
                continue
            r0, c0, r1, c1 = cand
            rects.append(
                {
                    "color": int(color),
                    "bbox": cand,
                    "center": ((r0 + r1) // 2, (c0 + c1) // 2),
                    "area": (r1 - r0 + 1) * (c1 - c0 + 1),
                }
            )
            placed = True
            break
        if not placed:
            return None

    return rects


def _build_input(shape: tuple[int, int], mask_color: int, mask: np.ndarray, rects: list[dict]) -> np.ndarray:
    h, w = shape
    grid = np.zeros((h, w), dtype=int)
    grid[:5, :5][_frame_mask()] = FRAME_COLOR
    grid[1:4, 1:4] = mask

    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]
        for r, c in _opposite_corners(r0, c0, r1, c1):
            grid[r, c] = rect["color"]

    assert colors_present(grid[1:4, 1:4], exclude={BLACK}) == [mask_color]
    return grid


def _semantic_checks(inp: np.ndarray, mode: str) -> bool:
    try:
        stamp = _extract_stamp(inp)
        rects = _pair_rectangles(inp)
        out = main(inp)
    except Exception:
        return False

    base = _render_base(inp.shape, rects)
    base[:5, :5] = BLACK
    if np.array_equal(out, base):
        return False
    if not _stamp_changes_base(inp.shape, stamp, rects):
        return False

    stamp_color = int(stamp[stamp != BLACK][0])
    if not any(rect["color"] != stamp_color for rect in rects):
        return False

    if mode == "clip" and not _has_clip(inp.shape, stamp, rects):
        return False
    if mode == "equal_area" and not _has_equal_area(rects):
        return False
    if mode == "crowded" and not _footprints_overlap(stamp, rects):
        return False
    if mode == "plain" and (_has_clip(inp.shape, stamp, rects) or _has_equal_area(rects) or _footprints_overlap(stamp, rects)):
        return False

    if out.shape != inp.shape or max(out.shape) > 30:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    mode = random.choice(["plain", "clip", "equal_area", "crowded"])
    h = random.randint(13, 18)
    w = random.randint(13, 18)
    shape = (h, w)

    mask_color, mask = _random_mask()
    stamp_color = MASK_TO_STAMP[mask_color]

    if mode == "crowded":
        n_rects = random.choice([2, 3])
    elif mode == "equal_area":
        n_rects = random.choice([2, 3])
    else:
        n_rects = random.choice([2, 3])
    colors = random.sample(PAIR_COLORS, n_rects)
    if all(color == stamp_color for color in colors):
        return None

    rects = _choose_rectangles(shape, mode, colors)
    if rects is None:
        return None

    inp = _build_input(shape, mask_color, mask, rects)
    if not _semantic_checks(inp, mode):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
