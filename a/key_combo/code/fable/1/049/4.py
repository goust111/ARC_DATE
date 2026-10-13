from __future__ import annotations

import random

import numpy as np


MARKER_COLOR = 5
MIN_RECTS = 3
MAX_RECTS = 4
MIN_SIDE = 5
MAX_SIDE = 8
MIN_GRID = 22
MAX_GRID = 30
MOAT = 2
MIN_FREE_RUN = 5
MIN_STOP_RUN = 4
MAX_MARKERS = 6
SIDES = ("up", "down", "left", "right")
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
PALETTE = list(range(10))


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _gap_positions(rect: dict, side: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = rect["bbox"]
    if side == "up":
        return [(r0, c) for c in range(c0 + 1, c1)]
    if side == "down":
        return [(r1, c) for c in range(c0 + 1, c1)]
    if side == "left":
        return [(r, c0) for r in range(r0 + 1, r1)]
    return [(r, c1) for r in range(r0 + 1, r1)]


def _gap_dir(gap: tuple[int, int], bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    r, c = gap
    if r == r0 and c0 < c < c1:
        return DIRS["up"]
    if r == r1 and c0 < c < c1:
        return DIRS["down"]
    if c == c0 and r0 < r < r1:
        return DIRS["left"]
    if c == c1 and r0 < r < r1:
        return DIRS["right"]
    raise AssertionError("gap must lie on exactly one non-corner side")


def _bbox_contains(bbox: tuple[int, int, int, int], cell: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    r, c = cell
    return r0 <= r <= r1 and c0 <= c <= c1


def _expanded_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _trace_bbox_line(
    shape: tuple[int, int],
    gap: tuple[int, int],
    direction: tuple[int, int],
    bboxes: list[tuple[int, int, int, int]],
    source_idx: int,
) -> tuple[int, int | None]:
    dr, dc = direction
    r, c = gap
    steps = 0
    while True:
        r += dr
        c += dc
        if not _in_bounds(shape, r, c):
            return steps, None
        for idx, bbox in enumerate(bboxes):
            if idx == source_idx:
                continue
            if _bbox_contains(bbox, (r, c)):
                return steps, idx
        steps += 1


def _choose_unique_counts(rects: list[dict]) -> dict[int, int] | None:
    options_by_idx: dict[int, list[int]] = {}
    for idx, rect in enumerate(rects):
        cap = int(rect["capacity"])
        run = int(rect["run"])
        min_dots = 1 if rect["stop_kind"] == "wall" else 2
        options = [
            m
            for m in range(2, MAX_MARKERS + 1)
            if m <= cap and run // m >= min_dots
        ]
        if not options and cap >= 1 and run >= 1:
            options = [1]
        if not options:
            return None
        options_by_idx[idx] = options

    order = sorted(options_by_idx, key=lambda idx: (len(options_by_idx[idx]), rects[idx]["run"]))
    used: set[int] = set()
    assign: dict[int, int] = {}

    def dfs(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        choices = options_by_idx[idx][:]
        random.shuffle(choices)
        for marker_count in choices:
            if marker_count in used:
                continue
            used.add(marker_count)
            assign[idx] = marker_count
            if dfs(pos + 1):
                return True
            used.remove(marker_count)
            del assign[idx]
        return False

    return assign if dfs(0) else None


def _paint_frame(grid: np.ndarray, rect: dict) -> None:
    r0, c0, r1, c1 = rect["bbox"]
    color = int(rect["color"])
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    gr, gc = rect["gap"]
    grid[gr, gc] = int(rect["bg"])


def _analyze_rectangles(grid: np.ndarray) -> tuple[int, list[dict], set[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    colors = sorted(int(c) for c in np.unique(g) if int(c) not in {bg, MARKER_COLOR})
    rectangles: list[dict] = []
    wall_cells: set[tuple[int, int]] = set()

    for color in colors:
        coords = np.argwhere(g == color)
        assert coords.size > 0, "frame color must appear"
        r0, c0, r1, c1 = _bbox_coords(coords)
        assert r1 - r0 + 1 >= 3 and c1 - c0 + 1 >= 3, "frame must bound a nonempty interior"

        border = set(_border_cells(r0, c0, r1, c1))
        coords_set = {tuple(map(int, rc)) for rc in coords.tolist()}
        missing = border - coords_set
        extras = coords_set - border
        assert not extras, "frame color must stay on its rectangle border"
        assert len(missing) == 1, "each frame must have exactly one missing border cell"

        gap = next(iter(missing))
        direction = _gap_dir(gap, (r0, c0, r1, c1))
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        marker_count = int(np.sum(interior == MARKER_COLOR))
        assert marker_count > 0, "each rectangle must contain at least one grey marker"

        rectangles.append(
            {
                "color": int(color),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "gap": (int(gap[0]), int(gap[1])),
                "direction": direction,
                "marker_count": marker_count,
            }
        )
        wall_cells.update(coords_set)

    assert rectangles, "expected at least one aperture rectangle"
    return bg, rectangles, wall_cells


def _trace_paints(
    shape: tuple[int, int],
    rect: dict,
    wall_cells: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], str]:
    r, c = rect["gap"]
    dr, dc = rect["direction"]
    step = 0
    dots: list[tuple[int, int]] = []
    while True:
        r += dr
        c += dc
        if not _in_bounds(shape, r, c):
            return dots, "edge"
        if (r, c) in wall_cells:
            return dots, "wall"
        step += 1
        if step % int(rect["marker_count"]) == 0:
            dots.append((int(r), int(c)))


def _expected_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg, rectangles, wall_cells = _analyze_rectangles(g)
    out = g.copy()
    for rect in rectangles:
        dots, _ = _trace_paints(g.shape, rect, wall_cells)
        for r, c in dots:
            if int(g[r, c]) == bg:
                out[r, c] = int(rect["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast dotted rays outward from each aperture rectangle using its internal grey-count as the period.

    Input structure:
    - The grid has a dominant background plus several thin hollow rectangles whose frame colors are all different.
    - Each rectangle border is complete except for one non-corner missing cell that acts as an aperture.
    - Inside every rectangle, a unique number of grey marker pixels is scattered in the interior.

    Rule:
    - The aperture side fixes the ray direction: a top gap shoots upward, a bottom gap downward, a left gap leftward, and a right gap rightward.
    - Count the grey markers inside that rectangle to get m, then starting from the gap as distance 0, color the cells at distances m, 2m, 3m, ... along the outward ray with the frame color.
    - Advance the ray cell by cell; if it reaches any other rectangle wall, stop before painting that wall cell, otherwise continue until the grid boundary.
    - Keep all frames, grey markers, and every untouched cell unchanged.

    Key insight:
    - The gap is both an aperture and a direction cue, while the interior grey count is the hidden clock that determines the spacing of the emitted dot sequence.
    """
    return _expected_output(np.asarray(grid, dtype=int))


def _free_gap_candidates(
    rect_idx: int,
    rects: list[dict],
    shape: tuple[int, int],
    *,
    exclude_sides: set[str] | None = None,
) -> list[dict]:
    exclude_sides = exclude_sides or set()
    bboxes = [rect["bbox"] for rect in rects]
    candidates: list[dict] = []
    for side in SIDES:
        if side in exclude_sides:
            continue
        direction = DIRS[side]
        for gap in _gap_positions(rects[rect_idx], side):
            run, hit_idx = _trace_bbox_line(shape, gap, direction, bboxes, rect_idx)
            if hit_idx is None and run >= MIN_FREE_RUN:
                candidates.append({"gap": gap, "direction": direction, "run": run, "stop_kind": "edge"})
    return candidates


def _random_bbox(H: int, W: int) -> tuple[int, int, int, int]:
    h = random.randint(MIN_SIDE, MAX_SIDE)
    w = random.randint(MIN_SIDE, MAX_SIDE)
    r0 = random.randint(1, H - h - 2)
    c0 = random.randint(1, W - w - 2)
    return r0, c0, r0 + h - 1, c0 + w - 1


def _place_stop_pair(H: int, W: int) -> list[dict] | None:
    for _ in range(200):
        hs = random.randint(MIN_SIDE, MAX_SIDE)
        ws = random.randint(MIN_SIDE, MAX_SIDE)
        ht = random.randint(MIN_SIDE, MAX_SIDE)
        wt = random.randint(MIN_SIDE, MAX_SIDE)
        axis = random.choice(("horizontal", "vertical"))
        positive = random.choice((True, False))

        if axis == "horizontal":
            source_side = "right" if positive else "left"
            source_gap_offset = random.randint(1, hs - 2)
            target_hit_offset = random.randint(1, ht - 2)
            source_r0 = random.randint(1, H - hs - 2)
            hit_row = source_r0 + source_gap_offset
            target_r0 = hit_row - target_hit_offset
            if not (1 <= target_r0 <= H - ht - 2):
                continue
            free_run = random.randint(MIN_STOP_RUN, 8)
            if positive:
                max_source_c0 = W - (ws + wt + free_run) - 2
                if max_source_c0 < 1:
                    continue
                source_c0 = random.randint(1, max_source_c0)
                target_c0 = source_c0 + ws + free_run
            else:
                max_target_c0 = W - (ws + wt + free_run) - 2
                if max_target_c0 < 1:
                    continue
                target_c0 = random.randint(1, max_target_c0)
                source_c0 = target_c0 + wt + free_run
        else:
            source_side = "down" if positive else "up"
            source_gap_offset = random.randint(1, ws - 2)
            target_hit_offset = random.randint(1, wt - 2)
            source_c0 = random.randint(1, W - ws - 2)
            hit_col = source_c0 + source_gap_offset
            target_c0 = hit_col - target_hit_offset
            if not (1 <= target_c0 <= W - wt - 2):
                continue
            free_run = random.randint(MIN_STOP_RUN, 8)
            if positive:
                max_source_r0 = H - (hs + ht + free_run) - 2
                if max_source_r0 < 1:
                    continue
                source_r0 = random.randint(1, max_source_r0)
                target_r0 = source_r0 + hs + free_run
            else:
                max_target_r0 = H - (hs + ht + free_run) - 2
                if max_target_r0 < 1:
                    continue
                target_r0 = random.randint(1, max_target_r0)
                source_r0 = target_r0 + ht + free_run

        source_bbox = (source_r0, source_c0, source_r0 + hs - 1, source_c0 + ws - 1)
        target_bbox = (target_r0, target_c0, target_r0 + ht - 1, target_c0 + wt - 1)
        if _expanded_overlap(source_bbox, target_bbox, MOAT):
            continue
        return [
            {"bbox": source_bbox, "role": "stop_source", "fixed_side": source_side},
            {"bbox": target_bbox, "role": "free"},
        ]
    return None


def _place_extra_rectangles(H: int, W: int, rects: list[dict], total_rects: int) -> bool:
    placed = [rect["bbox"] for rect in rects]
    while len(rects) < total_rects:
        for _ in range(200):
            bbox = _random_bbox(H, W)
            if any(_expanded_overlap(bbox, other, MOAT) for other in placed):
                continue
            rects.append({"bbox": bbox, "role": "free"})
            placed.append(bbox)
            break
        else:
            return False
    return True


def _assign_gaps_and_runs(rects: list[dict], shape: tuple[int, int]) -> bool:
    bboxes = [rect["bbox"] for rect in rects]
    source_idx = next(i for i, rect in enumerate(rects) if rect["role"] == "stop_source")
    target_idx = 1 if source_idx == 0 else 0

    side = rects[source_idx]["fixed_side"]
    direction = DIRS[side]
    overlap_positions = []
    if side in {"left", "right"}:
        src_rows = range(rects[source_idx]["bbox"][0] + 1, rects[source_idx]["bbox"][2])
        tgt_rows = range(rects[target_idx]["bbox"][0] + 1, rects[target_idx]["bbox"][2])
        overlap_positions = sorted(set(src_rows).intersection(tgt_rows))
        if side == "right":
            overlap_positions = [r for r in overlap_positions if rects[source_idx]["bbox"][1] < rects[target_idx]["bbox"][1]]
        else:
            overlap_positions = [r for r in overlap_positions if rects[source_idx]["bbox"][1] > rects[target_idx]["bbox"][1]]
    else:
        src_cols = range(rects[source_idx]["bbox"][1] + 1, rects[source_idx]["bbox"][3])
        tgt_cols = range(rects[target_idx]["bbox"][1] + 1, rects[target_idx]["bbox"][3])
        overlap_positions = sorted(set(src_cols).intersection(tgt_cols))
        if side == "down":
            overlap_positions = [c for c in overlap_positions if rects[source_idx]["bbox"][0] < rects[target_idx]["bbox"][0]]
        else:
            overlap_positions = [c for c in overlap_positions if rects[source_idx]["bbox"][0] > rects[target_idx]["bbox"][0]]
    random.shuffle(overlap_positions)

    fixed_gap = None
    fixed_run = None
    for pos in overlap_positions:
        if side == "right":
            gap = (rects[source_idx]["bbox"][0] + (pos - rects[source_idx]["bbox"][0]), rects[source_idx]["bbox"][3])
        elif side == "left":
            gap = (rects[source_idx]["bbox"][0] + (pos - rects[source_idx]["bbox"][0]), rects[source_idx]["bbox"][1])
        elif side == "down":
            gap = (rects[source_idx]["bbox"][2], rects[source_idx]["bbox"][1] + (pos - rects[source_idx]["bbox"][1]))
        else:
            gap = (rects[source_idx]["bbox"][0], rects[source_idx]["bbox"][1] + (pos - rects[source_idx]["bbox"][1]))
        run, hit_idx = _trace_bbox_line(shape, gap, direction, bboxes, source_idx)
        if hit_idx == target_idx and run >= MIN_STOP_RUN:
            fixed_gap = gap
            fixed_run = run
            break
    if fixed_gap is None:
        return False

    rects[source_idx]["gap"] = fixed_gap
    rects[source_idx]["direction"] = direction
    rects[source_idx]["run"] = int(fixed_run)
    rects[source_idx]["stop_kind"] = "wall"

    source_facing = {"left": "right", "right": "left", "up": "down", "down": "up"}[side]
    for idx, rect in enumerate(rects):
        if idx == source_idx:
            continue
        exclude = {source_facing} if idx == target_idx else set()
        candidates = _free_gap_candidates(idx, rects, shape, exclude_sides=exclude)
        if not candidates:
            return False
        choice = random.choice(candidates)
        rect["gap"] = choice["gap"]
        rect["direction"] = choice["direction"]
        rect["run"] = int(choice["run"])
        rect["stop_kind"] = str(choice["stop_kind"])
    return True


def _draw_input_from_rects(H: int, W: int, bg: int, rects: list[dict]) -> np.ndarray | None:
    counts = _choose_unique_counts(rects)
    if counts is None:
        return None

    grid = np.full((H, W), bg, dtype=int)
    for idx, rect in enumerate(rects):
        rect["marker_count"] = int(counts[idx])
        rect["capacity"] = (rect["bbox"][2] - rect["bbox"][0] - 1) * (rect["bbox"][3] - rect["bbox"][1] - 1)
        rect["bg"] = int(bg)
        _paint_frame(grid, rect)

    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]
        interior = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
        cells = random.sample(interior, rect["marker_count"])
        for r, c in cells:
            grid[r, c] = MARKER_COLOR
    return grid


def _semantic_checks(inp: np.ndarray) -> bool:
    bg, rectangles, wall_cells = _analyze_rectangles(inp)
    assert MIN_RECTS <= len(rectangles) <= MAX_RECTS, "unexpected rectangle count"
    assert len({rect["color"] for rect in rectangles}) == len(rectangles), "frame colors must be unique"
    assert len({rect["marker_count"] for rect in rectangles}) == len(rectangles), "marker counts must be unique"

    marker_cells = {tuple(map(int, rc)) for rc in np.argwhere(inp == MARKER_COLOR).tolist()}
    allowed_markers: set[tuple[int, int]] = set()
    for rect in rectangles:
        r0, c0, r1, c1 = rect["bbox"]
        allowed_markers.update((r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1))
    assert marker_cells <= allowed_markers, "grey pixels must lie inside rectangle interiors"

    bboxes = [rect["bbox"] for rect in rectangles]
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            assert not _expanded_overlap(bboxes[i], bboxes[j], 1), "rectangles must stay separated"

    branch_kinds = []
    directions = set()
    painted_by_color: dict[int, set[tuple[int, int]]] = {}
    for rect in rectangles:
        dots, stop_kind = _trace_paints(inp.shape, rect, wall_cells)
        assert dots, "every generated ray should paint at least one dot"
        branch_kinds.append(stop_kind)
        directions.add(rect["direction"])
        painted_by_color[int(rect["color"])] = set(dots)
    assert "wall" in branch_kinds and "edge" in branch_kinds, "need both stopping and free rays"
    assert len(directions) >= 2, "need directional variety"

    all_painted: dict[tuple[int, int], int] = {}
    for color, dots in painted_by_color.items():
        for cell in dots:
            prev = all_painted.get(cell)
            assert prev is None or prev == color, "different rays must not fight over one cell"
            all_painted[cell] = color
            assert int(inp[cell]) == bg, "new dots must start on background cells"

    expected = _expected_output(inp)
    actual = main(inp)
    assert np.array_equal(actual, expected), "main must match the semantic reference"
    changed = actual != inp
    assert np.all(inp[changed] == bg), "output may only add color onto background"
    return True


def _build_attempt() -> np.ndarray | None:
    H = random.randint(MIN_GRID, MAX_GRID)
    W = random.randint(MIN_GRID, MAX_GRID)
    if random.random() < 0.35:
        H = random.randint(18, 24)
    if random.random() < 0.35:
        W = random.randint(18, 24)

    bg_choices = [c for c in PALETTE if c != MARKER_COLOR]
    bg = random.choice(bg_choices)
    frame_palette = [c for c in PALETTE if c not in {bg, MARKER_COLOR}]

    total_rects = random.randint(MIN_RECTS, MAX_RECTS)
    rects = _place_stop_pair(H, W)
    if rects is None:
        return None
    if not _place_extra_rectangles(H, W, rects, total_rects):
        return None

    if not _assign_gaps_and_runs(rects, (H, W)):
        return None

    if len(frame_palette) < len(rects):
        return None
    colors = random.sample(frame_palette, len(rects))
    random.shuffle(colors)
    for rect, color in zip(rects, colors):
        rect["color"] = int(color)
        r0, c0, r1, c1 = rect["bbox"]
        rect["capacity"] = int((r1 - r0 - 1) * (c1 - c0 - 1))

    grid = _draw_input_from_rects(H, W, bg, rects)
    if grid is None:
        return None

    try:
        if _semantic_checks(grid):
            return grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid aperture-ray grid")
