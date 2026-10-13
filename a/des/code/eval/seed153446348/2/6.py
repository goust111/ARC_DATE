import random
from collections import deque

import numpy as np


BG = 0
MAGENTA = 6

# --------------- solver ---------------


def _color_components(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))

    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _is_rectangular_perimeter(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False

    mask = np.zeros((h, w), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True

    expected = np.zeros((h, w), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return bool(np.array_equal(mask, expected))


def _pattern_boundary_clear(interior: np.ndarray) -> bool:
    if interior.shape[0] < 3 or interior.shape[1] < 3:
        return False
    return bool(
        np.all(interior[0, :] == BG)
        and np.all(interior[-1, :] == BG)
        and np.all(interior[:, 0] == BG)
        and np.all(interior[:, -1] == BG)
    )


def _frame_spec_from_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> dict:
    r0, c0, r1, c1 = bbox
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    assert inner.size > 0
    assert _pattern_boundary_clear(inner), "pattern must not touch the frame"

    coords = np.argwhere(inner != BG)
    assert len(coords) > 0, "each frame must contain a pattern"

    pr0, pc0 = coords.min(axis=0)
    pr1, pc1 = coords.max(axis=0)
    pixels = [
        (int(rr - pr0), int(cc - pc0), int(inner[rr, cc]))
        for rr, cc in coords
    ]

    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "inner_r0": int(r0 + 1),
        "inner_c0": int(c0 + 1),
        "inner_h": int(inner.shape[0]),
        "inner_w": int(inner.shape[1]),
        "area": int(inner.shape[0] * inner.shape[1]),
        "pattern_top": int(pr0),
        "pattern_left": int(pc0),
        "pattern_h": int(pr1 - pr0 + 1),
        "pattern_w": int(pc1 - pc0 + 1),
        "pixels": pixels,
    }


def _find_frames(grid: np.ndarray) -> list[dict]:
    candidates: list[tuple[int, list[tuple[int, int, int, int]]]] = []

    for color in sorted(int(c) for c in np.unique(grid) if int(c) != BG):
        ring_bboxes: list[tuple[int, int, int, int]] = []
        for comp in _color_components(grid, color):
            if not _is_rectangular_perimeter(comp):
                continue
            bbox = _bbox(comp)
            r0, c0, r1, c1 = bbox
            interior = grid[r0 + 1 : r1, c0 + 1 : c1]
            if interior.size == 0 or not _pattern_boundary_clear(interior):
                continue
            if not np.any(interior != BG):
                continue
            ring_bboxes.append(bbox)
        if 3 <= len(ring_bboxes) <= 5:
            candidates.append((color, ring_bboxes))

    assert len(candidates) == 1, "expected one shared frame color"
    _frame_color, bboxes = candidates[0]

    frame_specs = [_frame_spec_from_bbox(grid, bbox) for bbox in bboxes]
    areas = [frame["area"] for frame in frame_specs]
    assert len(set(areas)) == len(areas), "frame inner areas must have a unique order"

    coverage = np.zeros(grid.shape, dtype=bool)
    for frame in frame_specs:
        r0, c0, r1, c1 = frame["bbox"]
        assert not coverage[r0 : r1 + 1, c0 : c1 + 1].any(), "frames must not overlap"
        coverage[r0 : r1 + 1, c0 : c1 + 1] = True
    assert not np.any((grid != BG) & ~coverage), "all foreground must belong to some frame"

    frame_specs.sort(key=lambda frame: frame["area"])
    return frame_specs


def _round_half_up(value: float) -> int:
    return int(np.floor(value + 0.5))


def _mapped_offset(offset: int, src_inner: int, dst_inner: int, pattern_size: int) -> int:
    src_free = int(src_inner - pattern_size)
    dst_free = int(dst_inner - pattern_size)
    assert 0 <= offset <= src_free
    assert dst_free >= src_free >= 0
    if src_free == 0:
        return 0
    return _round_half_up(offset * dst_free / src_free)


def _projected_cells(source: dict, target: dict) -> tuple[int, int, list[tuple[int, int, int]]]:
    top = _mapped_offset(
        source["pattern_top"], source["inner_h"], target["inner_h"], source["pattern_h"]
    )
    left = _mapped_offset(
        source["pattern_left"], source["inner_w"], target["inner_w"], source["pattern_w"]
    )
    return int(top), int(left), list(source["pixels"])


def _stamp_projection(out: np.ndarray, source: dict, target: dict) -> None:
    top, left, pixels = _projected_cells(source, target)
    base_r = int(target["inner_r0"] + top)
    base_c = int(target["inner_c0"] + left)

    for dr, dc, color in pixels:
        rr = base_r + int(dr)
        cc = base_c + int(dc)
        assert target["inner_r0"] <= rr < target["inner_r0"] + target["inner_h"]
        assert target["inner_c0"] <= cc < target["inner_c0"] + target["inner_w"]
        out[rr, cc] = int(color) if int(out[rr, cc]) == BG else MAGENTA


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames = _find_frames(g)
    out = g.copy()
    for i in range(len(frames) - 1):
        _stamp_projection(out, frames[i], frames[i + 1])
    return out


# --------------- generator ---------------

INPUT_COLORS = [1, 2, 3, 4, 5, 7, 8, 9]
FRAME_COUNT_RANGE = (3, 5)
GRID_SIDE_RANGE = (24, 32)
INTERIOR_DIM_VALUES = list(range(4, 9))
PATTERN_BBOX_MAX = 3
PATTERN_AREA_CAP = 5
GEN_RETRIES = 400
FRAME_PLACE_RETRIES = 120
PATTERN_RETRIES = 200


def _random_connected_mask(h: int, w: int) -> np.ndarray:
    max_area = min(h * w, PATTERN_AREA_CAP, max(1, (2 * h * w + 2) // 3))
    target_area = random.randint(1, max_area)

    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    while len(cells) < target_area:
        frontier = set()
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        cells.add(random.choice(tuple(frontier)))

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _pattern_from_params(
    inner_h: int,
    inner_w: int,
    color: int,
    top: int,
    left: int,
    mask: np.ndarray,
) -> dict:
    coords = np.argwhere(mask)
    pixels = [(int(r), int(c), int(color)) for r, c in coords]
    full_cells = {(int(top + r), int(left + c)) for r, c in coords}
    return {
        "inner_h": int(inner_h),
        "inner_w": int(inner_w),
        "pattern_top": int(top),
        "pattern_left": int(left),
        "pattern_h": int(mask.shape[0]),
        "pattern_w": int(mask.shape[1]),
        "pixels": pixels,
        "full_cells": full_cells,
    }


def _sample_pattern(
    inner_h: int,
    inner_w: int,
    color: int,
    incoming_cells: set[tuple[int, int]] | None = None,
    want_overlap: bool | None = None,
) -> dict | None:
    max_h = min(PATTERN_BBOX_MAX, inner_h - 2)
    max_w = min(PATTERN_BBOX_MAX, inner_w - 2)
    if max_h < 1 or max_w < 1:
        return None

    for _ in range(PATTERN_RETRIES):
        ph = random.randint(1, max_h)
        pw = random.randint(1, max_w)
        max_top = inner_h - ph - 1
        max_left = inner_w - pw - 1
        if max_top < 1 or max_left < 1:
            continue

        top = random.randint(1, max_top)
        left = random.randint(1, max_left)
        mask = _random_connected_mask(ph, pw)
        pattern = _pattern_from_params(inner_h, inner_w, color, top, left, mask)

        if incoming_cells is not None:
            overlaps = bool(pattern["full_cells"] & incoming_cells)
            if overlaps != want_overlap:
                continue
        return pattern

    return None


def _make_frame_spec(bbox: tuple[int, int, int, int], pattern: dict) -> dict:
    r0, c0, r1, c1 = bbox
    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "inner_r0": int(r0 + 1),
        "inner_c0": int(c0 + 1),
        "inner_h": int(pattern["inner_h"]),
        "inner_w": int(pattern["inner_w"]),
        "area": int(pattern["inner_h"] * pattern["inner_w"]),
        "pattern_top": int(pattern["pattern_top"]),
        "pattern_left": int(pattern["pattern_left"]),
        "pattern_h": int(pattern["pattern_h"]),
        "pattern_w": int(pattern["pattern_w"]),
        "pixels": list(pattern["pixels"]),
    }


def _projected_cells_relative(source_frame: dict, target_inner_h: int, target_inner_w: int) -> set[tuple[int, int]]:
    target_stub = {
        "inner_h": int(target_inner_h),
        "inner_w": int(target_inner_w),
    }
    top, left, pixels = _projected_cells(source_frame, target_stub)
    return {(int(top + dr), int(left + dc)) for dr, dc, _color in pixels}


def _choose_overlap_flags(frame_count: int) -> list[bool]:
    flags = [bool(random.getrandbits(1)) for _ in range(frame_count - 1)]
    if all(flags):
        flags[random.randrange(len(flags))] = False
    if not any(flags):
        flags[random.randrange(len(flags))] = True
    return flags


def _sample_dimensions(frame_count: int) -> list[tuple[int, int]]:
    heights = sorted(random.sample(INTERIOR_DIM_VALUES, frame_count))
    widths = sorted(random.sample(INTERIOR_DIM_VALUES, frame_count))
    return [(int(h), int(w)) for h, w in zip(heights, widths)]


def _choose_canvas(rect_dims: list[tuple[int, int]]) -> tuple[int, int] | None:
    max_h = max(h for h, _w in rect_dims)
    max_w = max(w for _h, w in rect_dims)
    total = sum(h * w for h, w in rect_dims)

    min_rows = max(GRID_SIDE_RANGE[0], max_h + 4, int(np.ceil(np.sqrt(total * 1.5))))
    if min_rows > GRID_SIDE_RANGE[1]:
        return None
    rows = random.randint(min_rows, GRID_SIDE_RANGE[1])

    min_cols = max(GRID_SIDE_RANGE[0], max_w + 4, int(np.ceil(total * 1.2 / rows)))
    if min_cols > GRID_SIDE_RANGE[1]:
        return None
    cols = random.randint(min_cols, GRID_SIDE_RANGE[1])
    return rows, cols


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    rows, cols = occupied.shape
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    rr0 = max(0, r0 - 1)
    rr1 = min(rows - 1, r1 + 1)
    cc0 = max(0, c0 - 1)
    cc1 = min(cols - 1, c1 + 1)
    return not bool(occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _place_frames(rect_dims: list[tuple[int, int]], rows: int, cols: int) -> list[tuple[int, int, int, int]] | None:
    occupied = np.zeros((rows, cols), dtype=bool)
    bboxes: list[tuple[int, int, int, int] | None] = [None] * len(rect_dims)
    order = sorted(range(len(rect_dims)), key=lambda idx: rect_dims[idx][0] * rect_dims[idx][1], reverse=True)

    for idx in order:
        h, w = rect_dims[idx]
        candidates = []
        for r0 in range(rows - h + 1):
            for c0 in range(cols - w + 1):
                if _can_place(occupied, r0, c0, h, w):
                    candidates.append((r0, c0))
        if not candidates:
            return None

        r0, c0 = random.choice(candidates)
        r1 = r0 + h - 1
        c1 = c0 + w - 1
        occupied[r0 : r1 + 1, c0 : c1 + 1] = True
        bboxes[idx] = (int(r0), int(c0), int(r1), int(c1))

    return [bbox for bbox in bboxes if bbox is not None]


def _render_input(
    rows: int,
    cols: int,
    frame_color: int,
    bboxes: list[tuple[int, int, int, int]],
    frame_specs: list[dict],
) -> np.ndarray:
    grid = np.zeros((rows, cols), dtype=int)

    for bbox, frame in zip(bboxes, frame_specs):
        r0, c0, r1, c1 = bbox
        grid[r0, c0 : c1 + 1] = frame_color
        grid[r1, c0 : c1 + 1] = frame_color
        grid[r0 : r1 + 1, c0] = frame_color
        grid[r0 : r1 + 1, c1] = frame_color

        base_r = frame["inner_r0"] + frame["pattern_top"]
        base_c = frame["inner_c0"] + frame["pattern_left"]
        for dr, dc, color in frame["pixels"]:
            grid[base_r + dr, base_c + dc] = int(color)

    return grid


def _render_output_from_spec(inp: np.ndarray, frame_specs: list[dict]) -> np.ndarray:
    out = inp.copy()
    for i in range(len(frame_specs) - 1):
        _stamp_projection(out, frame_specs[i], frame_specs[i + 1])
    return out


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.randint(*FRAME_COUNT_RANGE)
    dims = _sample_dimensions(frame_count)
    rect_dims = [(h + 2, w + 2) for h, w in dims]

    colors = random.sample(INPUT_COLORS, frame_count + 1)
    frame_color = colors[0]
    pattern_colors = colors[1:]
    overlap_flags = _choose_overlap_flags(frame_count)

    patterns: list[dict] = []
    frame_specs_no_bbox: list[dict] = []

    first_h, first_w = dims[0]
    first_pattern = _sample_pattern(first_h, first_w, pattern_colors[0])
    if first_pattern is None:
        return None
    patterns.append(first_pattern)
    frame_specs_no_bbox.append(
        {
            "inner_h": first_pattern["inner_h"],
            "inner_w": first_pattern["inner_w"],
            "pattern_top": first_pattern["pattern_top"],
            "pattern_left": first_pattern["pattern_left"],
            "pattern_h": first_pattern["pattern_h"],
            "pattern_w": first_pattern["pattern_w"],
            "pixels": list(first_pattern["pixels"]),
        }
    )

    for i in range(1, frame_count):
        inner_h, inner_w = dims[i]
        incoming = _projected_cells_relative(frame_specs_no_bbox[i - 1], inner_h, inner_w)
        pattern = _sample_pattern(
            inner_h,
            inner_w,
            pattern_colors[i],
            incoming_cells=incoming,
            want_overlap=overlap_flags[i - 1],
        )
        if pattern is None:
            return None
        patterns.append(pattern)
        frame_specs_no_bbox.append(
            {
                "inner_h": pattern["inner_h"],
                "inner_w": pattern["inner_w"],
                "pattern_top": pattern["pattern_top"],
                "pattern_left": pattern["pattern_left"],
                "pattern_h": pattern["pattern_h"],
                "pattern_w": pattern["pattern_w"],
                "pixels": list(pattern["pixels"]),
            }
        )

    for _ in range(FRAME_PLACE_RETRIES):
        canvas = _choose_canvas(rect_dims)
        if canvas is None:
            return None
        rows, cols = canvas
        bboxes = _place_frames(rect_dims, rows, cols)
        if bboxes is None:
            continue

        frame_specs = [_make_frame_spec(bbox, pattern) for bbox, pattern in zip(bboxes, patterns)]
        inp = _render_input(rows, cols, frame_color, bboxes, frame_specs)
        expected = _render_output_from_spec(inp, frame_specs)
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp

    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
