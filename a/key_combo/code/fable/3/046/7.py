from __future__ import annotations

from collections import deque
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
SIDES = ("top", "bottom", "left", "right")


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _foreground_colors(grid: np.ndarray, bg: int) -> list[int]:
    return sorted(int(v) for v in np.unique(grid) if int(v) != int(bg))


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros(grid.shape, dtype=bool)
    comps: list[np.ndarray] = []

    for sr, sc in np.argwhere(grid == int(color)):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q: deque[Coord] = deque([(sr, sc)])
        seen[sr, sc] = True
        pts: list[Coord] = []
        while q:
            r, c = q.popleft()
            pts.append((r, c))
            for dr, dc in NEIGHBORS4:
                rr, cc = r + dr, c + dc
                if (
                    0 <= rr < h
                    and 0 <= cc < w
                    and not seen[rr, cc]
                    and int(grid[rr, cc]) == int(color)
                ):
                    seen[rr, cc] = True
                    q.append((rr, cc))
        comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> set[Coord]:
    return {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }


def _gap_side(box: BBox, gap: Coord) -> str | None:
    r0, c0, r1, c1 = box
    r, c = gap
    if r == r0 and c0 < c < c1:
        return "top"
    if r == r1 and c0 < c < c1:
        return "bottom"
    if c == c0 and r0 < r < r1:
        return "left"
    if c == c1 and r0 < r < r1:
        return "right"
    return None


def _frame_from_component(coords: np.ndarray, color: int) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 + 1 < 4 or c1 - c0 + 1 < 4:
        return None

    cells = {(int(r), int(c)) for r, c in coords}
    border = _border_cells(r0, c0, r1, c1)
    if not cells <= border:
        return None

    gaps = list(border - cells)
    if len(gaps) != 1:
        return None
    gap = gaps[0]
    side = _gap_side((r0, c0, r1, c1), gap)
    if side is None:
        return None
    if cells != border - {gap}:
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "cells": cells,
        "color": int(color),
        "gap": gap,
        "side": side,
    }


def _inside_frame(frame: dict, r: int, c: int) -> bool:
    r0, c0, r1, c1 = frame["bbox"]
    return r0 < int(r) < r1 and c0 < int(c) < c1


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for color in _foreground_colors(grid, bg):
        for comp in _components_of_color(grid, color):
            frame = _frame_from_component(comp, color)
            if frame is not None:
                frames.append(frame)

    assert frames, "expected at least one one-gap rectangular frame"
    frames.sort(key=lambda f: (f["bbox"], f["color"]))

    frame_cells: dict[Coord, int] = {}
    for idx, frame in enumerate(frames):
        r0, c0, r1, c1 = frame["bbox"]
        assert r1 - r0 + 1 >= 6 and c1 - c0 + 1 >= 6, "frames must leave cells on both sides of the gap axis"
        for cell in frame["cells"]:
            assert cell not in frame_cells, "frame borders must not overlap"
            frame_cells[cell] = idx

    for r0, c0 in np.argwhere(grid != int(bg)):
        r, c = int(r0), int(c0)
        if (r, c) in frame_cells:
            continue
        owners = [idx for idx, frame in enumerate(frames) if _inside_frame(frame, r, c)]
        assert len(owners) == 1, "every non-frame colored pixel must lie inside exactly one frame"

    return frames


def _axis_info(frame: dict) -> tuple[str, int, int]:
    r0, c0, r1, c1 = frame["bbox"]
    gap_r, gap_c = frame["gap"]
    side = frame["side"]

    if side in ("top", "bottom"):
        assert 2 * gap_c != c0 + c1, "gap must be offset from the frame's horizontal midpoint"
        source_sign = -1 if 2 * gap_c < c0 + c1 else 1
        return "vertical", int(gap_c), source_sign

    assert 2 * gap_r != r0 + r1, "gap must be offset from the frame's vertical midpoint"
    source_sign = -1 if 2 * gap_r < r0 + r1 else 1
    return "horizontal", int(gap_r), source_sign


def _apply_one_frame(g: np.ndarray, out: np.ndarray, frame: dict, bg: int) -> None:
    r0, c0, r1, c1 = frame["bbox"]
    axis_kind, axis, source_sign = _axis_info(frame)

    source_pixels: list[tuple[int, int, int]] = []
    target_cells: list[Coord] = []
    target_had_pixels = False

    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            delta = c - axis if axis_kind == "vertical" else r - axis
            if delta == 0:
                continue
            if delta * source_sign > 0:
                if int(g[r, c]) != int(bg):
                    source_pixels.append((r, c, int(g[r, c])))
            else:
                target_cells.append((r, c))
                if int(g[r, c]) != int(bg):
                    target_had_pixels = True

    assert source_pixels, "source side must contain a colored interior pattern"
    assert target_had_pixels, "target side must contain fake colored pixels to overwrite"

    for r, c in target_cells:
        out[r, c] = int(bg)

    for r, c, color in source_pixels:
        if axis_kind == "vertical":
            rr, cc = r, 2 * axis - c
            assert r0 < rr < r1 and c0 < cc < c1 and (cc - axis) * source_sign < 0
        else:
            rr, cc = 2 * axis - r, c
            assert r0 < rr < r1 and c0 < cc < c1 and (rr - axis) * source_sign < 0
        out[rr, cc] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reprint each one-gap frame's true interior pattern by mirroring the gap-biased side across the gap axis.

    Input structure:
    - The background is the most common color. The grid contains one or more separated
      hollow rectangular frames, each with exactly one missing non-corner border cell.
    - The gap is offset from the midpoint of its side, and colored interior pixels appear
      on both sides of the row or column passing through that gap.
    - Any non-frame colored pixel lies strictly inside exactly one frame.

    Rule:
    - A top or bottom gap defines the vertical mirror axis through the gap column; a left
      or right gap defines the horizontal mirror axis through the gap row.
    - The side toward which the gap is offset from the frame center is the source side.
    - Clear every interior pixel on the opposite target side to background, then mirror
      each colored source-side pixel across the axis onto that target side. Preserve the
      source side, the axis cells, every frame border pixel, and the missing gap.

    Key insight:
    - The gap has two roles: its row or column gives the mirror axis, while its off-center
      side chooses which half is genuine; the other half is erased before the mirrored
      reprint is stamped.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must be at most 30 by 30"

    bg = _get_bg(g)
    frames = _find_frames(g, bg)
    out = g.copy()
    for frame in frames:
        _apply_one_frame(g, out, frame, bg)
    return out


# ----------------------------- generator -----------------------------

GEN_MAX_ATTEMPTS = 800
GEN_PLACE_ATTEMPTS = 200
GEN_SIZE_RANGE = (20, 30)
GEN_FRAME_COUNT_RANGE = (2, 4)
GEN_FRAME_SIDE_RANGE = (6, 10)
GEN_COLORS = tuple(range(10))


def _rects_overlap(a: BBox, b: BBox) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar0 <= br1 and br0 <= ar1 and ac0 <= bc1 and bc0 <= ac1


def _pad_box(box: BBox, pad: int) -> BBox:
    r0, c0, r1, c1 = box
    return r0 - pad, c0 - pad, r1 + pad, c1 + pad


def _gap_offsets(length: int) -> list[int]:
    return [x for x in range(2, int(length) - 2) if 2 * x != int(length) - 1]


def _draw_frame(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int, side: str, offset: int) -> Coord:
    r1, c1 = r0 + h - 1, c0 + w - 1
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)

    if side == "top":
        gap = (r0, c0 + offset)
    elif side == "bottom":
        gap = (r1, c0 + offset)
    elif side == "left":
        gap = (r0 + offset, c0)
    else:
        gap = (r0 + offset, c1)
    grid[gap] = int(_GEN_BG_FOR_DRAW)
    return gap


def _side_cells(box: BBox, axis_kind: str, axis: int, source_sign: int, want_source: bool) -> list[Coord]:
    r0, c0, r1, c1 = box
    cells: list[Coord] = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            delta = c - axis if axis_kind == "vertical" else r - axis
            if delta == 0:
                continue
            is_source = delta * source_sign > 0
            if is_source == want_source:
                cells.append((r, c))
    return cells


def _mirror_cell(cell: Coord, axis_kind: str, axis: int) -> Coord:
    r, c = cell
    if axis_kind == "vertical":
        return r, 2 * axis - c
    return 2 * axis - r, c


def _paint_pattern(
    grid: np.ndarray,
    expected_base: np.ndarray,
    box: BBox,
    side: str,
    gap: Coord,
    bg: int,
    palette: list[int],
) -> tuple[dict, np.ndarray]:
    frame = {"bbox": box, "gap": gap, "side": side}
    axis_kind, axis, source_sign = _axis_info(frame)
    source_cells = _side_cells(box, axis_kind, axis, source_sign, True)
    target_cells = _side_cells(box, axis_kind, axis, source_sign, False)
    mirrored_source = {_mirror_cell(cell, axis_kind, axis) for cell in source_cells}
    target_extra = [cell for cell in target_cells if cell not in mirrored_source]

    if not source_cells or not target_cells or not target_extra:
        raise ValueError("degenerate frame geometry")

    source_count = random.randint(2, min(7, len(source_cells)))
    source_chosen = random.sample(source_cells, source_count)
    source_records: list[tuple[int, int, int]] = []
    for r, c in source_chosen:
        color = random.choice(palette)
        grid[r, c] = int(color)
        source_records.append((r, c, int(color)))

    fake_count = random.randint(2, min(7, len(target_cells)))
    fake_cells = set(random.sample(target_cells, fake_count))
    fake_cells.add(random.choice(target_extra))
    for r, c in fake_cells:
        grid[r, c] = int(random.choice(palette))

    expected = expected_base.copy()
    r0, c0, r1, c1 = box
    expected[r0 : r1 + 1, c0 : c1 + 1] = grid[r0 : r1 + 1, c0 : c1 + 1]
    for r, c in target_cells:
        expected[r, c] = int(bg)
    for r, c, color in source_records:
        rr, cc = _mirror_cell((r, c), axis_kind, axis)
        expected[rr, cc] = int(color)

    return frame, expected


def _choose_sides(count: int) -> list[str]:
    if count == 2:
        sides = [random.choice(("top", "bottom")), random.choice(("left", "right"))]
    elif count == 3:
        sides = [random.choice(("top", "bottom")), random.choice(("left", "right"))]
        sides.append(random.choice(SIDES))
    else:
        sides = list(SIDES)
    random.shuffle(sides)
    return sides


def _gen_semantic_checks(inp: np.ndarray, expected: np.ndarray, frames: list[dict], bg: int) -> bool:
    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    saw_vertical = False
    saw_horizontal = False
    saw_cleared_fake = False
    saw_reprint = False

    for frame in frames:
        axis_kind, axis, source_sign = _axis_info(frame)
        if axis_kind == "vertical":
            saw_vertical = True
        else:
            saw_horizontal = True

        box = frame["bbox"]
        source_cells = _side_cells(box, axis_kind, axis, source_sign, True)
        target_cells = _side_cells(box, axis_kind, axis, source_sign, False)
        source_nonbg = [cell for cell in source_cells if int(inp[cell]) != int(bg)]
        target_nonbg = [cell for cell in target_cells if int(inp[cell]) != int(bg)]
        if not source_nonbg or not target_nonbg:
            return False

        mirror_targets = {_mirror_cell(cell, axis_kind, axis) for cell in source_nonbg}
        for cell in source_cells:
            if int(inp[cell]) != int(expected[cell]):
                return False
        for cell in target_nonbg:
            if cell not in mirror_targets and int(expected[cell]) == int(bg):
                saw_cleared_fake = True
        for cell in source_nonbg:
            if int(expected[_mirror_cell(cell, axis_kind, axis)]) == int(inp[cell]):
                saw_reprint = True

    return saw_vertical and saw_horizontal and saw_cleared_fake and saw_reprint


_GEN_BG_FOR_DRAW = 0


def _gen_attempt() -> np.ndarray | None:
    global _GEN_BG_FOR_DRAW

    height = random.randint(*GEN_SIZE_RANGE)
    width = random.randint(*GEN_SIZE_RANGE)
    colors = list(GEN_COLORS)
    bg = random.choice(colors)
    _GEN_BG_FOR_DRAW = int(bg)
    colors.remove(bg)
    frame_color = random.choice(colors)
    pattern_palette = [c for c in colors if c != frame_color]
    if len(pattern_palette) < 3:
        return None

    grid = np.full((height, width), int(bg), dtype=int)
    expected = grid.copy()
    placed_boxes: list[BBox] = []
    frames: list[dict] = []
    count = random.randint(*GEN_FRAME_COUNT_RANGE)

    for side in _choose_sides(count):
        placed = False
        for _ in range(GEN_PLACE_ATTEMPTS):
            h = random.randint(*GEN_FRAME_SIDE_RANGE)
            w = random.randint(*GEN_FRAME_SIDE_RANGE)
            offsets = _gap_offsets(w if side in ("top", "bottom") else h)
            if not offsets:
                continue
            offset = random.choice(offsets)
            if h >= height - 2 or w >= width - 2:
                continue
            r0 = random.randint(1, height - h - 1)
            c0 = random.randint(1, width - w - 1)
            box = (r0, c0, r0 + h - 1, c0 + w - 1)
            padded = _pad_box(box, 1)
            if any(_rects_overlap(padded, _pad_box(other, 1)) for other in placed_boxes):
                continue

            trial = grid.copy()
            gap = _draw_frame(trial, r0, c0, h, w, frame_color, side, offset)
            try:
                frame, trial_expected = _paint_pattern(
                    trial,
                    expected,
                    box,
                    side,
                    gap,
                    bg,
                    pattern_palette,
                )
            except ValueError:
                continue

            grid = trial
            expected = trial_expected
            placed_boxes.append(box)
            frames.append(frame)
            placed = True
            break
        if not placed:
            return None

    if _get_bg(grid) != int(bg):
        return None
    if max(grid.shape) > 30:
        return None
    if not _gen_semantic_checks(grid, expected, frames, bg):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid one-gap frame reprint input")
