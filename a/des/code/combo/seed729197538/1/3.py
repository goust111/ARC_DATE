from __future__ import annotations

import numpy as np


BG = 0
PATTERN_COLOR = 5
FRAME_PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("up", "down", "left", "right")
GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 120
GRID_SIZE_RANGE = (18, 31)
FRAME_H_RANGE = (5, 10)
FRAME_W_RANGE = (5, 10)
OBJECT_COUNT = (1, 2, 3, 4)
OBJECT_WEIGHTS = np.array([1, 3, 3, 2], dtype=float)
PAD = 1


def _border_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _frame_info_for_color(grid: np.ndarray, color: int) -> dict[str, object] | None:
    coords = np.argwhere(grid == color)
    if coords.size == 0:
        return None

    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0 = int(r0)
    c0 = int(c0)
    r1 = int(r1)
    c1 = int(c1)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    sub = grid[r0 : r1 + 1, c0 : c1 + 1] == color
    border = _border_mask(h, w)
    if np.any(sub & ~border):
        return None

    missing = border & ~sub
    missing_count = int(missing.sum())
    if not (1 <= missing_count <= 3):
        return None

    sides: list[str] = []
    missing_coords = np.argwhere(missing)

    if np.all(missing_coords[:, 0] == 0):
        cols = np.flatnonzero(missing[0])
        if cols[0] > 0 and cols[-1] < w - 1 and np.all(np.diff(cols) == 1):
            sides.append("up")
    if np.all(missing_coords[:, 0] == h - 1):
        cols = np.flatnonzero(missing[-1])
        if cols[0] > 0 and cols[-1] < w - 1 and np.all(np.diff(cols) == 1):
            sides.append("down")
    if np.all(missing_coords[:, 1] == 0):
        rows = np.flatnonzero(missing[:, 0])
        if rows[0] > 0 and rows[-1] < h - 1 and np.all(np.diff(rows) == 1):
            sides.append("left")
    if np.all(missing_coords[:, 1] == w - 1):
        rows = np.flatnonzero(missing[:, -1])
        if rows[0] > 0 and rows[-1] < h - 1 and np.all(np.diff(rows) == 1):
            sides.append("right")

    if len(sides) != 1:
        return None

    return {
        "color": int(color),
        "bbox": (r0, c0, r1, c1),
        "side": sides[0],
    }


def _find_frames(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        info = _frame_info_for_color(grid, color)
        if info is not None:
            frames.append(info)
    frames.sort(key=lambda item: item["bbox"])
    return frames


def _interior_mask(grid: np.ndarray, frames: list[dict[str, object]]) -> np.ndarray:
    mask = np.zeros_like(grid, dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        mask[r0 + 1 : r1, c0 + 1 : c1] = True
    return mask


def _translated_interior_bbox(
    bbox: tuple[int, int, int, int], side: str
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    if side == "up":
        return (r0 - inner_h, c0 + 1, r0 - 1, c1 - 1)
    if side == "down":
        return (r1 + 1, c0 + 1, r1 + inner_h, c1 - 1)
    if side == "left":
        return (r0 + 1, c0 - inner_w, r1 - 1, c0 - 1)
    return (r0 + 1, c1 + 1, r1 - 1, c1 + inner_w)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = BG if np.any(g == BG) else int(np.bincount(g.ravel()).argmax())

    frames = _find_frames(g, bg)
    assert frames, "expected at least one framed region"

    inside = _interior_mask(g, frames)
    pattern_colors = set(map(int, np.unique(g[inside]))) - {bg}
    frame_colors = {int(frame["color"]) for frame in frames}
    pattern_colors -= frame_colors
    assert len(pattern_colors) == 1, "expected one shared pattern color"
    pattern_color = pattern_colors.pop()

    out = g.copy()
    h, w = g.shape

    for frame in frames:
        color = int(frame["color"])
        side = str(frame["side"])
        r0, c0, r1, c1 = frame["bbox"]
        inner = g[r0 + 1 : r1, c0 + 1 : c1] == pattern_color
        assert inner.any(), "each frame must contain a grey pattern"
        tr0, tc0, tr1, tc1 = _translated_interior_bbox((r0, c0, r1, c1), side)
        assert 0 <= tr0 <= tr1 < h and 0 <= tc0 <= tc1 < w
        target = out[tr0 : tr1 + 1, tc0 : tc1 + 1]
        target[inner] = color

    return out


def _box_in_bounds(box: tuple[int, int, int, int], h: int, w: int) -> bool:
    r0, c0, r1, c1 = box
    return 0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w


def _expand_box(
    box: tuple[int, int, int, int], pad: int, h: int, w: int
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return (
        max(0, r0 - pad),
        max(0, c0 - pad),
        min(h - 1, r1 + pad),
        min(w - 1, c1 + pad),
    )


def _reservation_mask(
    shape: tuple[int, int], boxes: list[tuple[int, int, int, int]], pad: int
) -> np.ndarray | None:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    for box in boxes:
        if not _box_in_bounds(box, h, w):
            return None
        er0, ec0, er1, ec1 = _expand_box(box, pad, h, w)
        mask[er0 : er1 + 1, ec0 : ec1 + 1] = True
    return mask


def _choose_gap(rng: np.random.Generator, side: str, outer_h: int, outer_w: int) -> tuple[int, int]:
    side_len = outer_w if side in {"up", "down"} else outer_h
    gap_len = int(rng.integers(1, min(3, side_len - 2) + 1))
    gap_start = int(rng.integers(1, side_len - gap_len))
    return gap_start, gap_len


def _grow_mask(rng: np.random.Generator, h: int, w: int, target: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    start = (int(rng.integers(0, h)), int(rng.integers(0, w)))
    mask[start] = True
    while int(mask.sum()) < target:
        frontier: list[tuple[int, int]] = []
        for r, c in np.argwhere(mask):
            r = int(r)
            c = int(c)
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.append((nr, nc))
        if not frontier:
            break
        nr, nc = frontier[int(rng.integers(0, len(frontier)))]
        mask[nr, nc] = True
    return mask


def _pattern_block(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    bh = int(rng.integers(1, h + 1))
    bw = int(rng.integers(1, w + 1))
    if bh == h and bw == w and h * w > 1:
        if h >= w and h > 1:
            bh -= 1
        elif w > 1:
            bw -= 1
    r0 = int(rng.integers(0, h - bh + 1))
    c0 = int(rng.integers(0, w - bw + 1))
    mask[r0 : r0 + bh, c0 : c0 + bw] = True
    return mask


def _pattern_line(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    horizontal = w > 1 and (h == 1 or rng.random() < 0.5)
    if horizontal:
        row = int(rng.integers(0, h))
        length = int(rng.integers(1, w + 1))
        start = int(rng.integers(0, w - length + 1))
        mask[row, start : start + length] = True
    else:
        col = int(rng.integers(0, w))
        length = int(rng.integers(1, h + 1))
        start = int(rng.integers(0, h - length + 1))
        mask[start : start + length, col] = True
    return mask


def _pattern_l(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    if h == 1 or w == 1:
        return _pattern_line(rng, h, w)

    mask = np.zeros((h, w), dtype=bool)
    arm_h = int(rng.integers(2, h + 1))
    arm_w = int(rng.integers(2, w + 1))
    r0 = int(rng.integers(0, h - arm_h + 1))
    c0 = int(rng.integers(0, w - arm_w + 1))
    orientation = int(rng.integers(0, 4))

    if orientation == 0:
        mask[r0 : r0 + arm_h, c0] = True
        mask[r0 + arm_h - 1, c0 : c0 + arm_w] = True
    elif orientation == 1:
        mask[r0 : r0 + arm_h, c0 + arm_w - 1] = True
        mask[r0 + arm_h - 1, c0 : c0 + arm_w] = True
    elif orientation == 2:
        mask[r0 : r0 + arm_h, c0] = True
        mask[r0, c0 : c0 + arm_w] = True
    else:
        mask[r0 : r0 + arm_h, c0 + arm_w - 1] = True
        mask[r0, c0 : c0 + arm_w] = True
    return mask


def _random_pattern(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    modes = ("block", "line", "l", "walk")
    mode = str(rng.choice(modes))
    if mode == "block":
        mask = _pattern_block(rng, h, w)
    elif mode == "line":
        mask = _pattern_line(rng, h, w)
    elif mode == "l":
        mask = _pattern_l(rng, h, w)
    else:
        area_limit = max(1, min(h * w - (1 if h * w > 1 else 0), (3 * h * w) // 5))
        target = int(rng.integers(1, area_limit + 1))
        mask = _grow_mask(rng, h, w, target)

    if not mask.any():
        mask[0, 0] = True
    if int(mask.sum()) == h * w and h * w > 1:
        mask[-1, -1] = False
    return mask


def _draw_frame(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: int,
    side: str,
    gap_start: int,
    gap_len: int,
) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color

    if side == "up":
        grid[r0, c0 + gap_start : c0 + gap_start + gap_len] = BG
    elif side == "down":
        grid[r1, c0 + gap_start : c0 + gap_start + gap_len] = BG
    elif side == "left":
        grid[r0 + gap_start : r0 + gap_start + gap_len, c0] = BG
    else:
        grid[r0 + gap_start : r0 + gap_start + gap_len, c1] = BG


def _place_object(
    rng: np.random.Generator,
    grid: np.ndarray,
    occupied: np.ndarray,
    color: int,
) -> bool:
    h, w = grid.shape

    for _ in range(PLACE_ATTEMPTS):
        outer_h = int(rng.integers(*FRAME_H_RANGE))
        outer_w = int(rng.integers(*FRAME_W_RANGE))
        side = str(rng.choice(SIDES))
        inner_h = outer_h - 2
        inner_w = outer_w - 2

        if side == "up":
            r_lo = inner_h + PAD
            r_hi = h - outer_h - PAD
            c_lo = PAD
            c_hi = w - outer_w - PAD
        elif side == "down":
            r_lo = PAD
            r_hi = h - outer_h - inner_h - PAD
            c_lo = PAD
            c_hi = w - outer_w - PAD
        elif side == "left":
            r_lo = PAD
            r_hi = h - outer_h - PAD
            c_lo = inner_w + PAD
            c_hi = w - outer_w - PAD
        else:
            r_lo = PAD
            r_hi = h - outer_h - PAD
            c_lo = PAD
            c_hi = w - outer_w - inner_w - PAD

        if r_lo > r_hi or c_lo > c_hi:
            continue

        r0 = int(rng.integers(r_lo, r_hi + 1))
        c0 = int(rng.integers(c_lo, c_hi + 1))
        bbox = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
        copy_bbox = _translated_interior_bbox(bbox, side)
        reserve = _reservation_mask((h, w), [bbox, copy_bbox], PAD)
        if reserve is None or np.any(occupied & reserve):
            continue

        gap_start, gap_len = _choose_gap(rng, side, outer_h, outer_w)
        pattern = _random_pattern(rng, inner_h, inner_w)

        _draw_frame(grid, bbox, color, side, gap_start, gap_len)
        grid[r0 + 1 : r0 + 1 + inner_h, c0 + 1 : c0 + 1 + inner_w][pattern] = PATTERN_COLOR
        occupied |= reserve
        return True

    return False


def _validate_instance(grid: np.ndarray, expected_frames: int) -> bool:
    frames = _find_frames(grid, BG)
    if len(frames) != expected_frames:
        return False

    inside = _interior_mask(grid, frames)
    pattern_colors = set(map(int, np.unique(grid[inside]))) - {BG}
    frame_colors = {int(frame["color"]) for frame in frames}
    pattern_colors -= frame_colors
    if pattern_colors != {PATTERN_COLOR}:
        return False

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        inner = grid[r0 + 1 : r1, c0 + 1 : c1] == PATTERN_COLOR
        if not inner.any():
            return False
        tr0, tc0, tr1, tc1 = _translated_interior_bbox(frame["bbox"], str(frame["side"]))
        if not _box_in_bounds((tr0, tc0, tr1, tc1), *grid.shape):
            return False
        if np.any(grid[tr0 : tr1 + 1, tc0 : tc1 + 1] != BG):
            return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    weights = OBJECT_WEIGHTS / OBJECT_WEIGHTS.sum()

    for _ in range(GEN_ATTEMPTS):
        h = int(rng.integers(*GRID_SIZE_RANGE))
        w = int(rng.integers(*GRID_SIZE_RANGE))
        frame_count = int(rng.choice(OBJECT_COUNT, p=weights))
        frame_colors = [int(c) for c in rng.choice(FRAME_PALETTE, size=frame_count, replace=False)]

        grid = np.full((h, w), BG, dtype=int)
        occupied = np.zeros((h, w), dtype=bool)

        ok = True
        for color in frame_colors:
            if not _place_object(rng, grid, occupied, color):
                ok = False
                break

        if ok and _validate_instance(grid, frame_count):
            return grid

    raise RuntimeError("failed to generate a valid input")
