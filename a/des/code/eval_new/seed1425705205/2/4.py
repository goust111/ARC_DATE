from __future__ import annotations

import random
from collections import deque

import numpy as np


MARKER_COLOR = 4
PALETTE = tuple(range(10))
CANVAS_SIZE = 30
GEN_RETRIES = 500


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0, c0, _r1, _c1 = _bbox(cells)
    return {(r - r0, c - c0) for r, c in cells}


def _mono_components(grid: np.ndarray, blocked: set[int]) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color in blocked:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((int(nr), int(nc)))
            comps.append({"color": color, "cells": cells, "bbox": _bbox(cells)})
    return comps


def _border_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _is_frame_component(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 5 or w < 5:
        return False
    return set(comp["cells"]) == _border_cells(comp["bbox"])


def _infer_marker_side(frame_bbox: tuple[int, int, int, int], marker: tuple[int, int]) -> str:
    r0, c0, r1, c1 = frame_bbox
    ir0, ic0, ir1, ic1 = r0 + 1, c0 + 1, r1 - 1, c1 - 1
    mr, mc = marker
    distances = {
        "top": mr - ir0,
        "bottom": ir1 - mr,
        "left": mc - ic0,
        "right": ic1 - mc,
    }
    best = min(distances.values())
    sides = [side for side, dist in distances.items() if dist == best]
    assert len(sides) == 1, "marker must be uniquely closest to one inner wall"
    return sides[0]


def _place_stamp_cells(
    rel_cells: set[tuple[int, int]],
    stamp_shape: tuple[int, int],
    frame_bbox: tuple[int, int, int, int],
    marker: tuple[int, int],
) -> tuple[str, set[tuple[int, int]], set[tuple[int, int]]]:
    r0, c0, r1, c1 = frame_bbox
    ir0, ic0, ir1, ic1 = r0 + 1, c0 + 1, r1 - 1, c1 - 1
    sh, sw = stamp_shape
    mr, mc = marker
    side = _infer_marker_side(frame_bbox, marker)

    if side in {"top", "bottom"}:
        anchor_left = mc <= (ic0 + ic1) // 2
        row0 = mr + 1 if side == "top" else mr - sh
        col0 = mc if anchor_left else mc - sw + 1
    else:
        anchor_top = mr <= (ir0 + ir1) // 2
        col0 = mc + 1 if side == "left" else mc - sw
        row0 = mr if anchor_top else mr - sh + 1

    original = {(row0 + dr, col0 + dc) for dr, dc in rel_cells}
    for rr, cc in original:
        assert ir0 <= rr <= ir1 and ic0 <= cc <= ic1, "stamp placement must stay inside the frame interior"
        assert (rr, cc) != marker, "stamp must stay adjacent to the marker, not overwrite it"

    if side in {"left", "right"}:
        mirrored = {(rr, ic0 + ic1 - cc) for rr, cc in original}
    else:
        mirrored = {(ir0 + ir1 - rr, cc) for rr, cc in original}

    for rr, cc in mirrored:
        assert ir0 <= rr <= ir1 and ic0 <= cc <= ic1, "mirrored stamp must stay inside the frame interior"
        assert (rr, cc) != marker, "mirrored stamp must not overwrite the marker"

    return side, original, mirrored


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each colored stamp into its matching frame and add its mirror copy.

    Input structure:
    - The background is the most common color.
    - Several hollow rectangular frames have distinct border colors, and each frame contains one yellow marker pixel on one inner side.
    - Outside the frames, there is one monochrome connected stamp per frame color.

    Rule:
    - Match every outside stamp to the frame with the same color, erase the outside stamp, and place it inside that frame near the marker's wall.
    - The marker's row/column on that wall fixes which corner of the stamp sits next to it, while left/right markers use the vertical interior midline as the mirror axis and top/bottom markers use the horizontal interior midline.
    - Keep the frame borders and yellow markers unchanged, then paint both the placed stamp and its mirrored copy in the frame color.

    Key insight:
    - The yellow marker encodes two things at once: which side receives the original stamp and which mirror direction to apply after placement.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    out = g.copy()

    comps = _mono_components(g, blocked={bg, MARKER_COLOR})
    frames: dict[int, dict] = {}
    stamps: dict[int, dict] = {}
    for comp in comps:
        color = int(comp["color"])
        if _is_frame_component(comp):
            assert color not in frames, "frame colors must be unique"
            frames[color] = comp
        else:
            assert color not in stamps, "each frame color must have exactly one outside stamp"
            stamps[color] = comp

    assert frames and set(frames) == set(stamps), "frames and stamps must form a one-to-one color pairing"

    markers = [tuple(map(int, pos)) for pos in np.argwhere(g == MARKER_COLOR)]
    for color, frame in frames.items():
        r0, c0, r1, c1 = frame["bbox"]
        inside_markers = [pos for pos in markers if r0 < pos[0] < r1 and c0 < pos[1] < c1]
        assert len(inside_markers) == 1, "each frame must contain exactly one marker"
        marker = inside_markers[0]

        stamp = stamps[color]
        sr0, sc0, sr1, sc1 = stamp["bbox"]
        rel_cells = {(r - sr0, c - sc0) for r, c in stamp["cells"]}
        stamp_shape = (sr1 - sr0 + 1, sc1 - sc0 + 1)

        for rr, cc in stamp["cells"]:
            out[rr, cc] = bg

        _side, original, mirrored = _place_stamp_cells(rel_cells, stamp_shape, frame["bbox"], marker)
        for rr, cc in original | mirrored:
            out[rr, cc] = color

    return out


def _reflect_rel(cells: set[tuple[int, int]], shape: tuple[int, int], axis: str) -> set[tuple[int, int]]:
    h, w = shape
    if axis == "vertical":
        return {(r, w - 1 - c) for r, c in cells}
    return {(h - 1 - r, c) for r, c in cells}


def _is_frame_like_rel(cells: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    if h < 3 or w < 3:
        return False
    border = _border_cells((0, 0, h - 1, w - 1))
    return cells == border


def _grow_connected_cells(height: int, width: int, area: int) -> set[tuple[int, int]]:
    seed = (random.randrange(height), random.randrange(width))
    cells = {seed}
    while len(cells) < area:
        frontier = set()
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        cells.add(random.choice(tuple(frontier)))
    return cells


def _random_stamp_shape(axis: str) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    for _ in range(200):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        max_area = min(6, h * w - 1)
        if max_area < 3:
            continue
        cells = _normalize(_grow_connected_cells(h, w, random.randint(3, max_area)))
        sh = max(r for r, _ in cells) + 1
        sw = max(c for _, c in cells) + 1
        shape = (sh, sw)
        if sh < 2 or sw < 2:
            continue
        if len(cells) == sh * sw:
            continue
        if _is_frame_like_rel(cells, shape):
            continue
        if cells == _reflect_rel(cells, shape, axis):
            continue
        return cells, shape
    raise RuntimeError("failed to create a usable stamp shape")


def _sample_marker_rel(
    side: str,
    corner_pref: str,
    inner_h: int,
    inner_w: int,
    stamp_shape: tuple[int, int],
) -> tuple[int, int] | None:
    sh, sw = stamp_shape
    center_r = (inner_h - 1) // 2
    center_c = (inner_w - 1) // 2

    if side in {"top", "bottom"}:
        row = 0 if side == "top" else inner_h - 1
        if corner_pref == "left":
            lo, hi = 1, min(center_c, inner_w - sw)
        else:
            lo, hi = max(sw - 1, center_c + 1), inner_w - 2
        if lo > hi:
            return None
        return row, random.randint(lo, hi)

    col = 0 if side == "left" else inner_w - 1
    if corner_pref == "top":
        lo, hi = 1, min(center_r, inner_h - sh)
    else:
        lo, hi = max(sh - 1, center_r + 1), inner_h - 2
    if lo > hi:
        return None
    return random.randint(lo, hi), col


def _sample_frame_spec(color: int, side: str) -> dict:
    axis = "vertical" if side in {"left", "right"} else "horizontal"
    for _ in range(200):
        rel_cells, stamp_shape = _random_stamp_shape(axis)
        sh, sw = stamp_shape

        if side in {"left", "right"}:
            inner_w_lo = max(2 * sw + 2, 6)
            inner_w_hi = min(9, 2 * sw + 4)
            inner_h_lo = max(sh + 2, 6)
            inner_h_hi = min(10, sh + 5)
            prefs = ("top", "bottom")
        else:
            inner_h_lo = max(2 * sh + 2, 6)
            inner_h_hi = min(9, 2 * sh + 4)
            inner_w_lo = max(sw + 2, 6)
            inner_w_hi = min(10, sw + 5)
            prefs = ("left", "right")

        if inner_h_lo > inner_h_hi or inner_w_lo > inner_w_hi:
            continue

        inner_h = random.randint(inner_h_lo, inner_h_hi)
        inner_w = random.randint(inner_w_lo, inner_w_hi)
        marker_rel = _sample_marker_rel(side, random.choice(prefs), inner_h, inner_w, stamp_shape)
        if marker_rel is None:
            continue

        return {
            "color": int(color),
            "side": side,
            "stamp_rel": rel_cells,
            "stamp_shape": stamp_shape,
            "inner_h": inner_h,
            "inner_w": inner_w,
            "outer_h": inner_h + 2,
            "outer_w": inner_w + 2,
            "marker_rel": marker_rel,
        }
    raise RuntimeError("failed to build a valid frame specification")


def _can_place_bbox(occupied: np.ndarray, r0: int, c0: int, height: int, width: int, padding: int, edge_margin: int) -> bool:
    h, w = occupied.shape
    if r0 < edge_margin or c0 < edge_margin:
        return False
    if r0 + height > h - edge_margin or c0 + width > w - edge_margin:
        return False
    rr0 = max(0, r0 - padding)
    cc0 = max(0, c0 - padding)
    rr1 = min(h, r0 + height + padding)
    cc1 = min(w, c0 + width + padding)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _mark_bbox(occupied: np.ndarray, r0: int, c0: int, height: int, width: int, padding: int) -> None:
    h, w = occupied.shape
    rr0 = max(0, r0 - padding)
    cc0 = max(0, c0 - padding)
    rr1 = min(h, r0 + height + padding)
    cc1 = min(w, c0 + width + padding)
    occupied[rr0:rr1, cc0:cc1] = True


def _place_random_bbox(occupied: np.ndarray, height: int, width: int, padding: int = 1, edge_margin: int = 1) -> tuple[int, int] | None:
    h, w = occupied.shape
    candidates = [
        (r, c)
        for r in range(edge_margin, h - edge_margin - height + 1)
        for c in range(edge_margin, w - edge_margin - width + 1)
    ]
    random.shuffle(candidates)
    for r0, c0 in candidates:
        if _can_place_bbox(occupied, r0, c0, height, width, padding, edge_margin):
            return r0, c0
    return None


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _build_expected(grid: np.ndarray, specs: list[dict], bg: int) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        color = spec["color"]
        sr0, sc0 = spec["outside_top_left"]
        for dr, dc in spec["stamp_rel"]:
            out[sr0 + dr, sc0 + dc] = bg
        _side, original, mirrored = _place_stamp_cells(
            spec["stamp_rel"],
            spec["stamp_shape"],
            spec["frame_bbox"],
            spec["marker"],
        )
        for rr, cc in original | mirrored:
            out[rr, cc] = color
    return out


def _adjust_specs(specs: list[dict], row_shift: int, col_shift: int) -> list[dict]:
    adjusted: list[dict] = []
    for spec in specs:
        r0, c0, r1, c1 = spec["frame_bbox"]
        mr, mc = spec["marker"]
        sr0, sc0 = spec["outside_top_left"]
        adjusted.append(
            {
                **spec,
                "frame_bbox": (r0 - row_shift, c0 - col_shift, r1 - row_shift, c1 - col_shift),
                "marker": (mr - row_shift, mc - col_shift),
                "outside_top_left": (sr0 - row_shift, sc0 - col_shift),
            }
        )
    return adjusted


def _crop_with_margin(grid: np.ndarray, specs: list[dict], bg: int) -> tuple[np.ndarray, list[dict]]:
    rows, cols = np.where(grid != bg)
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())

    top_extra = random.randint(1, min(4, r0))
    left_extra = random.randint(1, min(4, c0))
    bottom_extra = random.randint(1, min(4, grid.shape[0] - 1 - r1))
    right_extra = random.randint(1, min(4, grid.shape[1] - 1 - c1))

    rs = r0 - top_extra
    re = r1 + bottom_extra + 1
    cs = c0 - left_extra
    ce = c1 + right_extra + 1

    cropped = grid[rs:re, cs:ce].copy()
    return cropped, _adjust_specs(specs, rs, cs)


def _sample_sides(n_frames: int) -> list[str]:
    if n_frames == 2:
        return [random.choice(["top", "bottom"]), random.choice(["left", "right"])]
    if n_frames == 3:
        return random.sample(["top", "bottom", "left", "right"], 3)
    sides = ["top", "bottom", "left", "right"]
    random.shuffle(sides)
    return sides


def _generate_attempt() -> np.ndarray | None:
    n_frames = random.randint(2, 4)
    bg_choices = [color for color in PALETTE if color != MARKER_COLOR]
    bg = random.choice(bg_choices)
    frame_colors = random.sample([color for color in PALETTE if color not in {bg, MARKER_COLOR}], n_frames)
    specs = [_sample_frame_spec(color, side) for color, side in zip(frame_colors, _sample_sides(n_frames), strict=False)]

    grid = np.full((CANVAS_SIZE, CANVAS_SIZE), bg, dtype=int)
    occupied = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=bool)

    for spec in specs:
        pos = _place_random_bbox(occupied, spec["outer_h"], spec["outer_w"], padding=1, edge_margin=1)
        if pos is None:
            return None
        r0, c0 = pos
        bbox = (r0, c0, r0 + spec["outer_h"] - 1, c0 + spec["outer_w"] - 1)
        spec["frame_bbox"] = bbox
        _draw_frame(grid, bbox, spec["color"])
        mr_rel, mc_rel = spec["marker_rel"]
        marker = (r0 + 1 + mr_rel, c0 + 1 + mc_rel)
        spec["marker"] = marker
        grid[marker] = MARKER_COLOR
        _mark_bbox(occupied, r0, c0, spec["outer_h"], spec["outer_w"], padding=1)

    for spec in specs:
        sh, sw = spec["stamp_shape"]
        pos = _place_random_bbox(occupied, sh, sw, padding=1, edge_margin=1)
        if pos is None:
            return None
        sr0, sc0 = pos
        spec["outside_top_left"] = pos
        for dr, dc in spec["stamp_rel"]:
            grid[sr0 + dr, sc0 + dc] = spec["color"]
        _mark_bbox(occupied, sr0, sc0, sh, sw, padding=1)

    cropped, cropped_specs = _crop_with_margin(grid, specs, bg)
    if max(cropped.shape) > 30:
        return None
    if _most_common_color(cropped) != bg:
        return None

    expected = _build_expected(cropped, cropped_specs, bg)
    if not np.array_equal(main(cropped), expected):
        return None
    return cropped


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
