from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
YELLOW = 4
GREY = 5
FRAME_COLORS = (1, 2, 3, 6, 7, 8, 9)
RECT_COUNT = 3
COUNT_CHOICES = (1, 2, 3, 4)
SIDE_NAMES = ("top", "right", "bottom", "left")
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _perimeter_path(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    path = [(r0, c) for c in range(c0, c1 + 1)]
    path.extend((r, c1) for r in range(r0 + 1, r1 + 1))
    path.extend((r1, c) for c in range(c1 - 1, c0 - 1, -1))
    path.extend((r, c0) for r in range(r1 - 1, r0, -1))
    return path


def _side_path(h: int, w: int, side: str) -> list[tuple[int, int]]:
    if side == "top":
        return [(0, c) for c in range(w)]
    if side == "right":
        return [(r, w - 1) for r in range(h)]
    if side == "bottom":
        return [(h - 1, c) for c in range(w - 1, -1, -1)]
    if side == "left":
        return [(r, 0) for r in range(h - 1, -1, -1)]
    raise ValueError(f"unknown side: {side}")


def _yellow_run_indices(values: list[int]) -> list[int]:
    n = len(values)
    is_yellow = [v == YELLOW for v in values]
    starts = [i for i in range(n) if is_yellow[i] and not is_yellow[(i - 1) % n]]
    assert len(starts) == 1, "expected exactly one yellow segment"
    start = starts[0]
    run = [start]
    idx = (start + 1) % n
    while idx != start and is_yellow[idx]:
        run.append(idx)
        idx = (idx + 1) % n
    assert not is_yellow[idx], "yellow segment must not wrap the entire perimeter"
    assert sum(is_yellow) == len(run), "yellow border must be one contiguous run"
    return run


def _segment_starts(path_len: int, seg_len: int, gap: int) -> list[int]:
    starts = []
    step = seg_len + int(gap)
    pos = 0
    while pos + seg_len <= path_len:
        starts.append(pos)
        pos += step
        if pos >= path_len:
            break
    return starts


def _signature(path_len: int, seg_len: int, gap: int) -> tuple[int, ...]:
    painted = set()
    for start in _segment_starts(path_len, seg_len, gap):
        for off in range(seg_len):
            painted.add(start + off)
    return tuple(sorted(painted))


def _extract_rectangles(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    comps = _components((g != BG) & (g != GREY))
    rects: list[dict] = []

    for comp in comps:
        box = _bbox(comp)
        path = _perimeter_path(box)
        coord_set = {tuple(map(int, rc)) for rc in comp.tolist()}
        assert len(coord_set) == len(path), "rectangle border area must equal its perimeter"
        assert coord_set == set(path), "foreground component must be a complete hollow rectangle border"

        values = [int(g[r, c]) for r, c in path]
        colors = set(values)
        assert YELLOW in colors and len(colors) == 2, "rectangle border must use one frame color plus yellow"
        frame_color = next(color for color in colors if color != YELLOW)

        run = _yellow_run_indices(values)
        seg_len = len(run)
        assert seg_len in (2, 3), "yellow segment length must be 2 or 3"

        r0, c0, r1, c1 = box
        assert r1 - r0 + 1 >= 5 and c1 - c0 + 1 >= 5, "rectangles must be at least 5x5"
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        interior_non_bg = set(map(int, interior[interior != BG].tolist()))
        assert interior_non_bg <= {GREY}, "rectangle interior may only contain grey markers"
        marker_count = int(np.sum(interior == GREY))
        assert marker_count >= 1, "each rectangle must contain at least one grey marker"

        rects.append(
            {
                "bbox": box,
                "path": path,
                "start_idx": int(run[0]),
                "seg_len": seg_len,
                "frame_color": int(frame_color),
                "marker_count": marker_count,
            }
        )

    assert len(rects) == RECT_COUNT, f"expected exactly {RECT_COUNT} hollow rectangles"
    assert int(np.sum(g == GREY)) == sum(rect["marker_count"] for rect in rects), "all grey markers must lie inside rectangles"
    rects.sort(key=lambda rect: (rect["bbox"][1], rect["bbox"][0]))
    counts = [rect["marker_count"] for rect in rects]
    assert len(set(counts)) == RECT_COUNT, "rectangle marker counts must be pairwise distinct"
    return rects


def _apply_rule(grid: np.ndarray, rects: list[dict], gaps: list[int]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[interior == GREY] = BG

    for rect, gap in zip(rects, gaps):
        path = rect["path"]
        start = rect["start_idx"]
        seg_len = rect["seg_len"]
        rotated = path[start:] + path[:start]
        for pos in _segment_starts(len(rotated), seg_len, int(gap)):
            for off in range(seg_len):
                r, c = rotated[pos + off]
                out[r, c] = YELLOW
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repeat each frame's yellow border segment using the next frame's marker count as the spacing.

    Input structure:
    - Black background contains exactly three separated hollow rectangular frames arranged left-to-right.
    - Each frame border is one non-yellow color with one contiguous yellow segment replacing 2 or 3 border cells.
    - Each rectangle interior contains several isolated grey markers, and the three marker counts are distinct.

    Rule:
    - Count the grey markers inside the left, middle, and right rectangles as c1, c2, and c3.
    - For each rectangle, traverse its border clockwise starting from the first yellow-segment cell on that traversal.
    - Repaint the same yellow segment repeatedly on that border path with gap g1=c2, g2=c3, g3=c1, then erase all grey markers to black while keeping all non-yellow border pixels unchanged.

    Key insight:
    - The spacing that controls one rectangle is not local: every frame reads its tiling gap from its clockwise successor in the left-to-right cycle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert _get_bg(g) == BG, "expected black background"

    rects = _extract_rectangles(g)
    counts = [rect["marker_count"] for rect in rects]
    gaps = [counts[(idx + 1) % RECT_COUNT] for idx in range(RECT_COUNT)]
    return _apply_rule(g, rects, gaps)


def _build_rectangle(
    h: int,
    w: int,
    frame_color: int,
    side: str,
    seg_len: int,
    seg_offset: int,
    marker_count: int,
) -> tuple[np.ndarray, dict]:
    patch = np.full((h, w), BG, dtype=int)
    patch[0, :] = frame_color
    patch[-1, :] = frame_color
    patch[:, 0] = frame_color
    patch[:, -1] = frame_color

    side_cells = _side_path(h, w, side)
    seg_cells = side_cells[seg_offset : seg_offset + seg_len]
    for r, c in seg_cells:
        patch[r, c] = YELLOW

    interior_cells = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    markers = random.sample(interior_cells, marker_count)
    for r, c in markers:
        patch[r, c] = GREY

    return patch, {
        "h": int(h),
        "w": int(w),
        "frame_color": int(frame_color),
        "side": side,
        "seg_len": int(seg_len),
        "seg_offset": int(seg_offset),
        "marker_count": int(marker_count),
        "markers": tuple(markers),
    }


def _render_input(specs: list[dict]) -> np.ndarray:
    max_h = max(spec["h"] for spec in specs)
    top_margin = random.randint(1, 2)
    bottom_margin = random.randint(1, 2)
    left_margin = random.randint(1, 2)
    right_margin = random.randint(1, 2)
    gaps = [random.randint(2, 3) for _ in range(RECT_COUNT - 1)]
    height = top_margin + max_h + bottom_margin + random.randint(0, 2)
    width = left_margin + right_margin + sum(spec["w"] for spec in specs) + sum(gaps)
    if height > 30 or width > 30:
        raise ValueError("layout exceeds ARC limits")

    grid = np.full((height, width), BG, dtype=int)
    c = left_margin
    for idx, spec in enumerate(specs):
        top_max = height - bottom_margin - spec["h"]
        top = random.randint(top_margin, top_max)
        patch, _ = _build_rectangle(
            spec["h"],
            spec["w"],
            spec["frame_color"],
            spec["side"],
            spec["seg_len"],
            spec["seg_offset"],
            spec["marker_count"],
        )
        grid[top : top + spec["h"], c : c + spec["w"]] = patch
        spec["top"] = int(top)
        spec["left"] = int(c)
        c += spec["w"]
        if idx < len(gaps):
            c += gaps[idx]
    return grid


def _spec_rects(specs: list[dict]) -> list[dict]:
    rects = []
    for spec in specs:
        box = (
            int(spec["top"]),
            int(spec["left"]),
            int(spec["top"] + spec["h"] - 1),
            int(spec["left"] + spec["w"] - 1),
        )
        path = _perimeter_path(box)
        local_side = _side_path(spec["h"], spec["w"], spec["side"])
        local_start = local_side[spec["seg_offset"]]
        start_coord = (spec["top"] + local_start[0], spec["left"] + local_start[1])
        rects.append(
            {
                "bbox": box,
                "path": path,
                "start_idx": int(path.index(start_coord)),
                "seg_len": int(spec["seg_len"]),
                "frame_color": int(spec["frame_color"]),
                "marker_count": int(spec["marker_count"]),
            }
        )
    rects.sort(key=lambda rect: (rect["bbox"][1], rect["bbox"][0]))
    return rects


def _expected_from_specs(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    rects = _spec_rects(specs)
    counts = [spec["marker_count"] for spec in specs]
    gaps = [counts[(idx + 1) % RECT_COUNT] for idx in range(RECT_COUNT)]
    return _apply_rule(grid, rects, gaps)


def _semantic_ok(grid: np.ndarray, specs: list[dict]) -> bool:
    try:
        rects = _extract_rectangles(grid)
    except AssertionError:
        return False

    counts = [rect["marker_count"] for rect in rects]
    if sorted(counts) != sorted(spec["marker_count"] for spec in specs):
        return False

    for rect in rects:
        path_len = len(rect["path"])
        seg_len = rect["seg_len"]
        signatures = {_signature(path_len, seg_len, gap) for gap in COUNT_CHOICES}
        if len(signatures) != len(COUNT_CHOICES):
            return False
        if len(_segment_starts(path_len, seg_len, counts[(rects.index(rect) + 1) % RECT_COUNT])) < 2:
            return False

    expected = _expected_from_specs(grid, specs)
    try:
        actual = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False

    self_gaps = counts[:]
    prev_gaps = [counts[(idx - 1) % RECT_COUNT] for idx in range(RECT_COUNT)]
    if np.array_equal(expected, _apply_rule(grid, rects, self_gaps)):
        return False
    if np.array_equal(expected, _apply_rule(grid, rects, prev_gaps)):
        return False

    if Counter(map(int, actual.ravel()))[YELLOW] <= Counter(map(int, grid.ravel()))[YELLOW]:
        return False
    return True


def _sample_specs() -> list[dict]:
    counts = random.sample(COUNT_CHOICES, RECT_COUNT)
    colors = random.sample(FRAME_COLORS, RECT_COUNT)
    specs = []
    for idx in range(RECT_COUNT):
        h = random.randint(5, 7)
        w = random.randint(5, 7)
        seg_len = random.randint(2, 3)
        side = random.choice(SIDE_NAMES)
        side_len = len(_side_path(h, w, side))
        seg_offset = random.randint(0, side_len - seg_len)
        specs.append(
            {
                "h": int(h),
                "w": int(w),
                "frame_color": int(colors[idx]),
                "side": side,
                "seg_len": int(seg_len),
                "seg_offset": int(seg_offset),
                "marker_count": int(counts[idx]),
            }
        )
    return specs


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        specs = _sample_specs()
        total_width = sum(spec["w"] for spec in specs) + random.randint(6, 10)
        total_height = max(spec["h"] for spec in specs) + random.randint(3, 6)
        if total_width > 30 or total_height > 30:
            continue
        try:
            grid = _render_input(specs)
        except ValueError:
            continue
        if _semantic_ok(grid, specs):
            return grid
    raise RuntimeError("failed to generate a valid cross-count border-tiling input")
