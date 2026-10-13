from __future__ import annotations

from collections import deque
import numpy as np


BG = 0
BLUE = 1
RED = 2
OBJECT_COLORS = (3, 4, 5, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")
TEMP_SIDE = 30
GENERATION_ATTEMPTS = 500
PLACEMENT_ATTEMPTS = 250
PADDING = 1
RNG = np.random.default_rng()


def _same_color_components(grid: np.ndarray, allowed: set[int]) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color not in allowed or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((int(rr), int(cc)))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rows = [p[0] for p in cells]
            cols = [p[1] for p in cells]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            out.append(
                {
                    "color": color,
                    "cells": cells,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "size": len(cells),
                }
            )
    return out


def _rectangle_components(grid: np.ndarray) -> list[dict]:
    rectangles: list[dict] = []
    for comp in _same_color_components(grid, set(OBJECT_COLORS)):
        r0, c0, r1, c1 = comp["bbox"]
        color = int(comp["color"])
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if comp["size"] != patch.size:
            continue
        if not bool((patch == color).all()):
            continue
        rectangles.append(
            {
                "color": color,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "h": int(r1 - r0 + 1),
                "w": int(c1 - c0 + 1),
            }
        )
    return rectangles


def _indicator_components(grid: np.ndarray) -> tuple[np.ndarray, dict[int, dict]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    labels = np.zeros((h, w), dtype=int)
    info: dict[int, dict] = {}
    next_id = 1
    for comp in _same_color_components(g, {BLUE, RED}):
        r0, c0, r1, c1 = comp["bbox"]
        color = int(comp["color"])
        horizontal = r0 == r1
        vertical = c0 == c1
        if not (horizontal or vertical):
            continue
        length = int(c1 - c0 + 1) if horizontal else int(r1 - r0 + 1)
        if length != int(comp["size"]):
            continue
        for rr, cc in comp["cells"]:
            labels[rr, cc] = next_id
        info[next_id] = {
            "color": color,
            "bbox": (int(r0), int(c0), int(r1), int(c1)),
            "length": length,
        }
        next_id += 1
    return labels, info


def _adjacent_indicator(rect: dict, labels: np.ndarray) -> tuple[str, int]:
    r0, c0, r1, c1 = rect["bbox"]
    candidates: list[tuple[str, int]] = []
    if r0 > 0:
        for idx in sorted(set(map(int, labels[r0 - 1, c0 : c1 + 1])) - {0}):
            candidates.append(("top", idx))
    if r1 + 1 < labels.shape[0]:
        for idx in sorted(set(map(int, labels[r1 + 1, c0 : c1 + 1])) - {0}):
            candidates.append(("bottom", idx))
    if c0 > 0:
        for idx in sorted(set(map(int, labels[r0 : r1 + 1, c0 - 1])) - {0}):
            candidates.append(("left", idx))
    if c1 + 1 < labels.shape[1]:
        for idx in sorted(set(map(int, labels[r0 : r1 + 1, c1 + 1])) - {0}):
            candidates.append(("right", idx))
    assert len(candidates) == 1, "each rectangle must have exactly one adjacent indicator"
    return candidates[0]


def _transformed_bbox(rect: dict, side: str, bar: dict, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = rect["bbox"]
    h, w = int(rect["h"]), int(rect["w"])
    length = int(bar["length"])
    is_crop = int(bar["color"]) == RED

    if side in ("top", "bottom"):
        assert (length <= h) if is_crop else (length >= h)
        if is_crop:
            if side == "top":
                nr0, nr1 = r0, r0 + length - 1
            else:
                nr0, nr1 = r1 - length + 1, r1
        else:
            if side == "top":
                nr0, nr1 = r1 - length + 1, r1
            else:
                nr0, nr1 = r0, r0 + length - 1
        nc0, nc1 = c0, c1
    else:
        assert (length <= w) if is_crop else (length >= w)
        if is_crop:
            if side == "left":
                nc0, nc1 = c0, c0 + length - 1
            else:
                nc0, nc1 = c1 - length + 1, c1
        else:
            if side == "left":
                nc0, nc1 = c1 - length + 1, c1
            else:
                nc0, nc1 = c0, c0 + length - 1
        nr0, nr1 = r0, r1

    gh, gw = shape
    assert 0 <= nr0 <= nr1 < gh
    assert 0 <= nc0 <= nc1 < gw
    return int(nr0), int(nc0), int(nr1), int(nc1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop or expand each solid rectangle according to the adjacent indicator bar's color and length.

    Input structure:
    - The background is black.
    - Several solid rectangles use colors other than black, blue, and red.
    - Each rectangle has exactly one adjacent 1-pixel-wide blue or red line segment on one side.

    Rule:
    - Detect which side of the rectangle the indicator touches and measure the indicator length L.
    - If the indicator is red, keep only L rows or columns from that side and remove the rest.
    - If the indicator is blue, expand outward from that side until that dimension's total size becomes L.
    - Remove the indicators and keep only the transformed rectangles on the black canvas.

    Key insight:
    The indicator's side selects the axis and anchored edge, while its color switches the meaning of its length between retained thickness and target thickness.
    """
    g = np.asarray(grid, dtype=int)
    out = np.full(g.shape, BG, dtype=int)
    rectangles = _rectangle_components(g)
    labels, bars = _indicator_components(g)
    used_bars: set[int] = set()

    for rect in rectangles:
        side, bar_id = _adjacent_indicator(rect, labels)
        used_bars.add(int(bar_id))
        bbox = _transformed_bbox(rect, side, bars[int(bar_id)], g.shape)
        r0, c0, r1, c1 = bbox
        out[r0 : r1 + 1, c0 : c1 + 1] = int(rect["color"])

    assert used_bars == set(bars), "every indicator must pair with one rectangle"
    return out


def _line_start(span: int, length: int) -> int:
    if length <= span:
        return int(RNG.integers(0, span - length + 1))
    extra = length - span
    return -int(RNG.integers(0, extra + 1))


def _spec_geometry(side: str, mode: str, final_h: int, final_w: int, original_extent: int) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    if side == "top":
        if mode == "crop":
            return (0, 0, original_extent - 1, final_w - 1), (-1, 0, -1, final_h - 1)
        return (final_h - original_extent, 0, final_h - 1, final_w - 1), (final_h - original_extent - 1, 0, final_h - original_extent - 1, final_h - 1)
    if side == "bottom":
        if mode == "crop":
            return (final_h - original_extent, 0, final_h - 1, final_w - 1), (final_h, 0, final_h, final_h - 1)
        return (0, 0, original_extent - 1, final_w - 1), (original_extent, 0, original_extent, final_h - 1)
    if side == "left":
        if mode == "crop":
            return (0, 0, final_h - 1, original_extent - 1), (0, -1, final_w - 1, -1)
        return (0, final_w - original_extent, final_h - 1, final_w - 1), (0, final_w - original_extent - 1, final_w - 1, final_w - original_extent - 1)
    if mode == "crop":
        return (0, final_w - original_extent, final_h - 1, final_w - 1), (0, final_w, final_w - 1, final_w)
    return (0, 0, final_h - 1, original_extent - 1), (0, original_extent, final_w - 1, original_extent)


def _offset_line(line_bbox: tuple[int, int, int, int], offset: int, horizontal: bool) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = line_bbox
    if horizontal:
        return (int(r0), int(c0 + offset), int(r1), int(c1 + offset))
    return (int(r0 + offset), int(c0), int(r1 + offset), int(c1))


def _sample_spec(color: int, mode: str) -> dict:
    side = str(RNG.choice(SIDES))
    row_axis = side in ("top", "bottom")
    other_dim = int(RNG.integers(2, 9))

    if row_axis:
        final_h = int(RNG.integers(2, 8)) if mode == "crop" else int(RNG.integers(3, 9))
        final_w = other_dim
        if mode == "crop":
            original_h = int(RNG.integers(final_h + 1, min(final_h + 4, 10) + 1))
        else:
            original_h = int(RNG.integers(max(2, final_h - 4), final_h))
        original_w = final_w
        bar_offset = _line_start(final_w, final_h)
        original_bbox, bar_bbox = _spec_geometry(side, mode, final_h, final_w, original_h)
        bar_bbox = _offset_line(bar_bbox, bar_offset, horizontal=True)
    else:
        final_h = other_dim
        final_w = int(RNG.integers(2, 8)) if mode == "crop" else int(RNG.integers(3, 9))
        original_h = final_h
        if mode == "crop":
            original_w = int(RNG.integers(final_w + 1, min(final_w + 4, 10) + 1))
        else:
            original_w = int(RNG.integers(max(2, final_w - 4), final_w))
        bar_offset = _line_start(final_h, final_w)
        original_bbox, bar_bbox = _spec_geometry(side, mode, final_h, final_w, original_w)
        bar_bbox = _offset_line(bar_bbox, bar_offset, horizontal=False)

    coords = [
        (0, 0),
        (final_h - 1, final_w - 1),
        (original_bbox[0], original_bbox[1]),
        (original_bbox[2], original_bbox[3]),
        (bar_bbox[0], bar_bbox[1]),
        (bar_bbox[2], bar_bbox[3]),
    ]
    rows = [p[0] for p in coords]
    cols = [p[1] for p in coords]
    footprint = (int(min(rows)), int(min(cols)), int(max(rows)), int(max(cols)))

    return {
        "color": int(color),
        "mode": mode,
        "side": side,
        "bar_color": RED if mode == "crop" else BLUE,
        "final_h": int(final_h),
        "final_w": int(final_w),
        "original_bbox": tuple(int(x) for x in original_bbox),
        "bar_bbox": tuple(int(x) for x in bar_bbox),
        "footprint": footprint,
        "area": (footprint[2] - footprint[0] + 1) * (footprint[3] - footprint[1] + 1),
    }


def _paint_rect(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _crop_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _place_specs(specs: list[dict]) -> list[dict] | None:
    occupied = np.zeros((TEMP_SIDE, TEMP_SIDE), dtype=bool)
    placed: list[dict] = []
    for spec in sorted(specs, key=lambda x: int(x["area"]), reverse=True):
        fr0, fc0, fr1, fc1 = spec["footprint"]
        min_r = PADDING - int(fr0)
        max_r = TEMP_SIDE - PADDING - 1 - int(fr1)
        min_c = PADDING - int(fc0)
        max_c = TEMP_SIDE - PADDING - 1 - int(fc1)
        if min_r > max_r or min_c > max_c:
            return None

        success = False
        for _ in range(PLACEMENT_ATTEMPTS):
            final_r = int(RNG.integers(min_r, max_r + 1))
            final_c = int(RNG.integers(min_c, max_c + 1))
            rr0 = final_r + int(fr0) - PADDING
            rr1 = final_r + int(fr1) + PADDING
            cc0 = final_c + int(fc0) - PADDING
            cc1 = final_c + int(fc1) + PADDING
            if bool(occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any()):
                continue
            occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
            placed.append({**spec, "final_pos": (int(final_r), int(final_c))})
            success = True
            break
        if not success:
            return None
    return placed


def _absolute_bbox(anchor: tuple[int, int], rel_bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    ar, ac = anchor
    r0, c0, r1, c1 = rel_bbox
    return int(ar + r0), int(ac + c0), int(ar + r1), int(ac + c1)


def _line_cells(bbox: tuple[int, int, int, int]) -> tuple[slice, slice]:
    r0, c0, r1, c1 = bbox
    return slice(r0, r1 + 1), slice(c0, c1 + 1)


def _build_candidate() -> tuple[np.ndarray, np.ndarray] | None:
    n_obj = int(RNG.integers(2, 5))
    colors = RNG.choice(np.array(OBJECT_COLORS, dtype=int), size=n_obj, replace=False).tolist()

    modes = ["crop"] * (n_obj // 2) + ["expand"] * (n_obj // 2)
    if n_obj % 2 == 1:
        modes.append(str(RNG.choice(["crop", "expand"])))
    RNG.shuffle(modes)

    specs = [_sample_spec(int(color), str(mode)) for color, mode in zip(colors, modes)]
    placed = _place_specs(specs)
    if placed is None:
        return None

    inp = np.full((TEMP_SIDE, TEMP_SIDE), BG, dtype=int)
    out = np.full((TEMP_SIDE, TEMP_SIDE), BG, dtype=int)

    for spec in placed:
        anchor = tuple(int(x) for x in spec["final_pos"])
        final_bbox = _absolute_bbox(anchor, (0, 0, spec["final_h"] - 1, spec["final_w"] - 1))
        original_bbox = _absolute_bbox(anchor, spec["original_bbox"])
        bar_bbox = _absolute_bbox(anchor, spec["bar_bbox"])

        _paint_rect(out, final_bbox, int(spec["color"]))
        _paint_rect(inp, original_bbox, int(spec["color"]))
        rs, cs = _line_cells(bar_bbox)
        inp[rs, cs] = int(spec["bar_color"])

    union_mask = (inp != BG) | (out != BG)
    if not bool(union_mask.any()):
        return None
    r0, c0, r1, c1 = _crop_bbox(union_mask)

    top = int(RNG.integers(0, r0 + 1))
    left = int(RNG.integers(0, c0 + 1))
    bottom = int(RNG.integers(0, TEMP_SIDE - r1))
    right = int(RNG.integers(0, TEMP_SIDE - c1))
    rr0 = r0 - top
    cc0 = c0 - left
    rr1 = r1 + bottom
    cc1 = c1 + right

    inp = inp[rr0 : rr1 + 1, cc0 : cc1 + 1]
    out = out[rr0 : rr1 + 1, cc0 : cc1 + 1]

    if max(inp.shape) > 30 or max(out.shape) > 30:
        return None
    if np.array_equal(inp, out):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        candidate = _build_candidate()
        if candidate is not None:
            return candidate[0]
    raise RuntimeError("failed to generate a valid input")
