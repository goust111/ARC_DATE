import random
from typing import Iterable

import numpy as np


BG = 0
GREY = 5
SIDES = ("top", "bottom", "left", "right")
HORIZONTAL_SIDES = {"top", "bottom"}
VERTICAL_SIDES = {"left", "right"}
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

# Generator configuration.
GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
RECT_COUNT_SMALL = [1, 2, 3]
RECT_COUNT_LARGE = [1, 2, 3, 4]
RECT_COUNT_WEIGHTS_SMALL = [2, 4, 2]
RECT_COUNT_WEIGHTS_LARGE = [1, 3, 3, 2]
RECT_H_RANGE = (4, 8)
RECT_W_RANGE = (4, 8)
OUTER_MARGIN = 1
PLACEMENT_PAD = 1
SPEC_RETRIES = 80
FRAME_PLACE_RETRIES = 100
SEGMENT_PLACE_RETRIES = 120
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = list(points)
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    return min(rs), min(cs), max(rs), max(cs)


def _border_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _side_cells(bbox: tuple[int, int, int, int], side: str) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if side == "top":
        return {(r0, c) for c in range(c0, c1 + 1)}
    if side == "bottom":
        return {(r1, c) for c in range(c0, c1 + 1)}
    if side == "left":
        return {(r, c0) for r in range(r0, r1 + 1)}
    if side == "right":
        return {(r, c1) for r in range(r0, r1 + 1)}
    raise ValueError(f"unknown side: {side}")


def _frame_cells(bbox: tuple[int, int, int, int], missing_side: str) -> set[tuple[int, int]]:
    return _border_cells(bbox) - _side_cells(bbox, missing_side)


def _candidate_full_bboxes(comp_bbox: tuple[int, int, int, int]) -> list[tuple[str, tuple[int, int, int, int]]]:
    r0, c0, r1, c1 = comp_bbox
    return [
        ("top", (r0 - 1, c0, r1, c1)),
        ("bottom", (r0, c0, r1 + 1, c1)),
        ("left", (r0, c0 - 1, r1, c1)),
        ("right", (r0, c0, r1, c1 + 1)),
    ]


def _segment_spec(side: str, bbox: tuple[int, int, int, int]) -> tuple[str, int]:
    r0, c0, r1, c1 = bbox
    if side in HORIZONTAL_SIDES:
        return "h", c1 - c0 + 1
    return "v", r1 - r0 + 1


def _detect_frame(points: list[tuple[int, int]], shape: tuple[int, int]) -> dict:
    h, w = shape
    comp_set = set(points)
    comp_bbox = _bbox(points)
    matches = []
    for side, full_bbox in _candidate_full_bboxes(comp_bbox):
        r0, c0, r1, c1 = full_bbox
        if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
            continue
        if r1 - r0 < 3 or c1 - c0 < 3:
            continue
        gap_cells = _side_cells(full_bbox, side)
        if comp_set == _frame_cells(full_bbox, side):
            orient, length = _segment_spec(side, full_bbox)
            matches.append(
                {
                    "full_bbox": full_bbox,
                    "missing_side": side,
                    "gap_cells": gap_cells,
                    "orientation": orient,
                    "length": length,
                }
            )
    assert len(matches) == 1, f"ambiguous or invalid grey frame at {comp_bbox}"
    return matches[0]


def _detect_segment(points: list[tuple[int, int]], grid: np.ndarray) -> dict:
    color_set = {int(grid[r, c]) for r, c in points}
    assert len(color_set) == 1, f"segment must be monochrome: {color_set}"
    color = next(iter(color_set))
    assert color not in (BG, GREY), f"unexpected segment color {color}"

    bbox = _bbox(points)
    r0, c0, r1, c1 = bbox
    if r0 == r1:
        orient = "h"
        length = c1 - c0 + 1
        expected = {(r0, c) for c in range(c0, c1 + 1)}
    elif c0 == c1:
        orient = "v"
        length = r1 - r0 + 1
        expected = {(r, c0) for r in range(r0, r1 + 1)}
    else:
        raise AssertionError(f"segment must be straight: {bbox}")
    assert set(points) == expected, f"segment must be contiguous: {bbox}"
    return {"cells": expected, "orientation": orient, "length": length, "color": color}


def _analyze_scene(grid: np.ndarray) -> tuple[list[dict], list[dict], dict[tuple[str, int], dict]]:
    grey_comps = _components(grid == GREY)
    assert grey_comps, "expected at least one grey frame"
    frames = [_detect_frame(comp, grid.shape) for comp in grey_comps]

    seg_mask = (grid != BG) & (grid != GREY)
    seg_comps = _components(seg_mask)
    assert seg_comps, "expected at least one colored line segment"
    segments = [_detect_segment(comp, grid) for comp in seg_comps]

    frame_specs = [(frame["orientation"], frame["length"]) for frame in frames]
    assert len(frame_specs) == len(set(frame_specs)), "frame specs must be unique"

    seg_map: dict[tuple[str, int], dict] = {}
    for seg in segments:
        spec = (seg["orientation"], seg["length"])
        assert spec not in seg_map, f"duplicate segment spec: {spec}"
        seg_map[spec] = seg

    assert set(frame_specs) == set(seg_map), "frame specs and segment specs must match exactly"
    assert len(frames) == len(segments), "expected one segment per frame"
    return frames, segments, seg_map


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reattach each scattered colored segment to the uniquely matching missing rectangle side.

    The moved segment completes the grey frame and its color fills the rectangle interior.
    The segment's original position is erased back to black.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames, segments, seg_map = _analyze_scene(g)
    out = g.copy()

    for seg in segments:
        for r, c in seg["cells"]:
            out[r, c] = BG

    for frame in frames:
        seg = seg_map[(frame["orientation"], frame["length"])]
        color = int(seg["color"])
        for r, c in frame["gap_cells"]:
            out[r, c] = color

        r0, c0, r1, c1 = frame["full_bbox"]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[interior == BG] = color

    return out


def _expanded_bbox(
    bbox: tuple[int, int, int, int],
    pad: int,
    shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _bbox_is_free(blocked: np.ndarray, bbox: tuple[int, int, int, int], pad: int) -> bool:
    er0, ec0, er1, ec1 = _expanded_bbox(bbox, pad, blocked.shape)
    return not blocked[er0 : er1 + 1, ec0 : ec1 + 1].any()


def _occupy(blocked: np.ndarray, bbox: tuple[int, int, int, int], pad: int) -> None:
    er0, ec0, er1, ec1 = _expanded_bbox(bbox, pad, blocked.shape)
    blocked[er0 : er1 + 1, ec0 : ec1 + 1] = True


def _draw_cells(grid: np.ndarray, cells: Iterable[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _random_rect_count(rng: random.Random, h: int, w: int) -> int:
    if min(h, w) < 22:
        return rng.choices(RECT_COUNT_SMALL, weights=RECT_COUNT_WEIGHTS_SMALL, k=1)[0]
    return rng.choices(RECT_COUNT_LARGE, weights=RECT_COUNT_WEIGHTS_LARGE, k=1)[0]


def _sample_frame_specs(rng: random.Random, count: int) -> list[dict] | None:
    used_specs: set[tuple[str, int]] = set()
    frames: list[dict] = []
    for _ in range(count):
        chosen = None
        for _ in range(SPEC_RETRIES):
            height = rng.randint(*RECT_H_RANGE)
            width = rng.randint(*RECT_W_RANGE)
            missing_side = rng.choice(SIDES)
            spec = ("h", width) if missing_side in HORIZONTAL_SIDES else ("v", height)
            if spec in used_specs:
                continue
            chosen = {
                "height": height,
                "width": width,
                "missing_side": missing_side,
                "orientation": spec[0],
                "length": spec[1],
            }
            break
        if chosen is None:
            return None
        used_specs.add((chosen["orientation"], chosen["length"]))
        frames.append(chosen)
    return frames


def _place_frame(
    grid: np.ndarray,
    blocked: np.ndarray,
    height: int,
    width: int,
    missing_side: str,
    rng: random.Random,
) -> tuple[int, int, int, int] | None:
    h, w = grid.shape
    max_r0 = h - height - OUTER_MARGIN
    max_c0 = w - width - OUTER_MARGIN
    if max_r0 < OUTER_MARGIN or max_c0 < OUTER_MARGIN:
        return None

    for _ in range(FRAME_PLACE_RETRIES):
        r0 = rng.randint(OUTER_MARGIN, max_r0)
        c0 = rng.randint(OUTER_MARGIN, max_c0)
        bbox = (r0, c0, r0 + height - 1, c0 + width - 1)
        if not _bbox_is_free(blocked, bbox, PLACEMENT_PAD):
            continue
        _draw_cells(grid, _frame_cells(bbox, missing_side), GREY)
        _occupy(blocked, bbox, PLACEMENT_PAD)
        return bbox
    return None


def _place_segment(
    grid: np.ndarray,
    blocked: np.ndarray,
    orientation: str,
    length: int,
    color: int,
    rng: random.Random,
) -> tuple[int, int, int, int] | None:
    h, w = grid.shape
    seg_h = 1 if orientation == "h" else length
    seg_w = length if orientation == "h" else 1
    max_r0 = h - seg_h - OUTER_MARGIN
    max_c0 = w - seg_w - OUTER_MARGIN
    if max_r0 < OUTER_MARGIN or max_c0 < OUTER_MARGIN:
        return None

    for _ in range(SEGMENT_PLACE_RETRIES):
        r0 = rng.randint(OUTER_MARGIN, max_r0)
        c0 = rng.randint(OUTER_MARGIN, max_c0)
        bbox = (r0, c0, r0 + seg_h - 1, c0 + seg_w - 1)
        if not _bbox_is_free(blocked, bbox, PLACEMENT_PAD):
            continue
        if orientation == "h":
            cells = {(r0, c) for c in range(c0, c0 + length)}
        else:
            cells = {(r, c0) for r in range(r0, r0 + length)}
        _draw_cells(grid, cells, color)
        _occupy(blocked, bbox, PLACEMENT_PAD)
        return bbox
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    rect_count = _random_rect_count(rng, h, w)
    frame_specs = _sample_frame_specs(rng, rect_count)
    if frame_specs is None:
        return None

    colors = rng.sample(COLOR_POOL, rect_count)
    for frame, color in zip(frame_specs, colors):
        frame["color"] = color

    # Place larger rectangles first to reduce rejection rate.
    frame_specs.sort(key=lambda item: item["height"] * item["width"], reverse=True)
    for frame in frame_specs:
        bbox = _place_frame(
            grid,
            blocked,
            frame["height"],
            frame["width"],
            frame["missing_side"],
            rng,
        )
        if bbox is None:
            return None
        frame["bbox"] = bbox

    # Place longer segments first to reduce rejection rate.
    frame_specs.sort(key=lambda item: item["length"], reverse=True)
    for frame in frame_specs:
        placed = _place_segment(
            grid,
            blocked,
            frame["orientation"],
            frame["length"],
            frame["color"],
            rng,
        )
        if placed is None:
            return None

    try:
        frames, segments, seg_map = _analyze_scene(grid)
        if len(frames) != rect_count or len(segments) != rect_count:
            return None
        if set((frame["orientation"], frame["length"]) for frame in frames) != set(seg_map):
            return None
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
