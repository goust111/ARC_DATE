from __future__ import annotations

import numpy as np


def _most_common(values: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(values, dtype=int), return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _gap_distance(side: str, row: int, col: int, bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    if side in ("top", "bottom"):
        return int(min(col - c0, c1 - col))
    return int(min(row - r0, r1 - row))


def _extract_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    bg = _most_common(g)
    non_bg = g[g != bg]
    assert non_bg.size > 0
    frame_color = _most_common(non_bg)

    frame_pts = np.argwhere(g == frame_color)
    assert frame_pts.size > 0
    r0, c0 = map(int, frame_pts.min(axis=0))
    r1, c1 = map(int, frame_pts.max(axis=0))
    assert r1 - r0 >= 2 and c1 - c0 >= 2

    for r, c in frame_pts:
        rr, cc = int(r), int(c)
        assert rr in (r0, r1) or cc in (c0, c1), "frame color must stay on the border ring"

    bbox = (r0, c0, r1, c1)
    gaps: list[dict] = []
    for col in range(c0 + 1, c1):
        if int(g[r0, col]) == bg:
            gaps.append(
                {
                    "side": "top",
                    "pos": (r0, col),
                    "corner_distance": _gap_distance("top", r0, col, bbox),
                }
            )
        if int(g[r1, col]) == bg:
            gaps.append(
                {
                    "side": "bottom",
                    "pos": (r1, col),
                    "corner_distance": _gap_distance("bottom", r1, col, bbox),
                }
            )
    for row in range(r0 + 1, r1):
        if int(g[row, c0]) == bg:
            gaps.append(
                {
                    "side": "left",
                    "pos": (row, c0),
                    "corner_distance": _gap_distance("left", row, c0, bbox),
                }
            )
        if int(g[row, c1]) == bg:
            gaps.append(
                {
                    "side": "right",
                    "pos": (row, c1),
                    "corner_distance": _gap_distance("right", row, c1, bbox),
                }
            )

    seed_mask = (g != bg) & (g != frame_color)
    seeds = []
    for row, col in np.argwhere(seed_mask):
        r, c = int(row), int(col)
        assert r0 < r < r1 and c0 < c < c1, "seeds must stay inside the frame"
        seeds.append({"pos": (r, c), "color": int(g[r, c])})
    assert seeds
    assert len({seed["color"] for seed in seeds}) == len(seeds), "seed colors must be unique"
    assert len(seeds) == len(gaps), "each gap must correspond to exactly one seed"

    pipes = []
    used_gaps: set[tuple[int, int]] = set()
    for seed in seeds:
        sr, sc = seed["pos"]
        aligned = [gap for gap in gaps if gap["pos"][0] == sr or gap["pos"][1] == sc]
        assert len(aligned) == 1, "every seed must align to exactly one gap"
        gap = aligned[0]
        assert gap["pos"] not in used_gaps, "each gap must align to exactly one seed"
        used_gaps.add(gap["pos"])

        gr, gc = gap["pos"]
        dr = int(np.sign(gr - sr))
        dc = int(np.sign(gc - sc))
        assert abs(dr) + abs(dc) == 1, "pipe direction must be cardinal"
        pipes.append(
            {
                "seed": (sr, sc),
                "gap": (gr, gc),
                "dir": (dr, dc),
                "color": seed["color"],
                "side": gap["side"],
                "priority_key": (gap["corner_distance"], gr, gc),
            }
        )

    assert len(used_gaps) == len(gaps)
    return {
        "grid": g,
        "shape": (h, w),
        "bg": bg,
        "frame_color": frame_color,
        "bbox": bbox,
        "gaps": gaps,
        "seeds": seeds,
        "pipes": pipes,
    }


def _interior_pipe_cells(pipe: dict, bbox: tuple[int, int, int, int], frame_color: int, grid: np.ndarray) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    h, w = grid.shape
    dr, dc = pipe["dir"]
    rr, cc = pipe["seed"][0] + dr, pipe["seed"][1] + dc
    cells: set[tuple[int, int]] = set()
    while 0 <= rr < h and 0 <= cc < w:
        if int(grid[rr, cc]) == frame_color:
            break
        if r0 < rr < r1 and c0 < cc < c1:
            cells.add((rr, cc))
        rr += dr
        cc += dc
    return cells


def _count_interior_crossings(scene: dict) -> int:
    bbox = scene["bbox"]
    frame_color = scene["frame_color"]
    grid = scene["grid"]
    paths = [
        _interior_pipe_cells(pipe, bbox, frame_color, grid)
        for pipe in scene["pipes"]
    ]
    crossings = 0
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            if paths[i] & paths[j]:
                crossings += len(paths[i] & paths[j])
    return crossings


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each interior seed toward its aligned frame gap, then continue to the canvas edge.

    When two pipes cross on interior background cells, the pipe whose gap is closer to
    a frame corner takes priority.
    """
    scene = _extract_scene(grid)
    g = scene["grid"]
    h, w = scene["shape"]
    bg = scene["bg"]
    frame_color = scene["frame_color"]

    out = g.copy()
    pipes = sorted(scene["pipes"], key=lambda pipe: pipe["priority_key"], reverse=True)
    for pipe in pipes:
        dr, dc = pipe["dir"]
        rr, cc = pipe["seed"][0] + dr, pipe["seed"][1] + dc
        while 0 <= rr < h and 0 <= cc < w:
            if int(g[rr, cc]) == frame_color:
                break
            if int(g[rr, cc]) == bg:
                out[rr, cc] = pipe["color"]
            rr += dr
            cc += dc
    return out


PALETTE = tuple(range(10))
MAX_GENERATION_ATTEMPTS = 400
GRID_MIN = 16
GRID_MAX = 25
FRAME_MIN = 13
FRAME_MAX = 19


def _sample_two_side_counts(rng: np.random.Generator) -> tuple[int, int]:
    if int(rng.integers(2)) == 0:
        return 1, 1
    if int(rng.integers(2)) == 0:
        return 2, 1
    return 1, 2


def _pick_without_replacement(rng: np.random.Generator, items: list[int], count: int) -> list[int]:
    assert 0 <= count <= len(items)
    if count == 0:
        return []
    chosen = rng.choice(np.array(items, dtype=int), size=count, replace=False)
    return [int(v) for v in chosen.tolist()]


def _build_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], frame_color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = frame_color
    grid[r1, c0 : c1 + 1] = frame_color
    grid[r0 : r1 + 1, c0] = frame_color
    grid[r0 : r1 + 1, c1] = frame_color


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GRID_MIN, GRID_MAX))
    w = int(rng.integers(GRID_MIN, GRID_MAX))

    max_fh = min(FRAME_MAX, h - 2)
    max_fw = min(FRAME_MAX, w - 2)
    if max_fh < FRAME_MIN or max_fw < FRAME_MIN:
        return None

    fh = int(rng.integers(FRAME_MIN, max_fh + 1))
    fw = int(rng.integers(FRAME_MIN, max_fw + 1))
    r0 = int(rng.integers(1, h - fh))
    c0 = int(rng.integers(1, w - fw))
    r1 = r0 + fh - 1
    c1 = c0 + fw - 1
    bbox = (r0, c0, r1, c1)

    top_count, bottom_count = _sample_two_side_counts(rng)
    left_count, right_count = _sample_two_side_counts(rng)
    vertical_total = top_count + bottom_count
    horizontal_total = left_count + right_count

    middle_rows = list(range(r0 + 3, r1 - 2))
    middle_cols = list(range(c0 + 3, c1 - 2))
    low_rows = [r0 + 1, r0 + 2]
    high_rows = [r1 - 2, r1 - 1]
    low_cols = [c0 + 1, c0 + 2]
    high_cols = [c1 - 2, c1 - 1]
    if len(middle_rows) < horizontal_total or len(middle_cols) < vertical_total:
        return None

    cross_mode = bool(int(rng.integers(2)))
    if cross_mode:
        row_groups = {
            0: [r for r in middle_rows if _gap_distance("left", r, c0, bbox) % 2 == 0],
            1: [r for r in middle_rows if _gap_distance("left", r, c0, bbox) % 2 == 1],
        }
        col_groups = {
            0: [c for c in middle_cols if _gap_distance("top", r0, c, bbox) % 2 == 0],
            1: [c for c in middle_cols if _gap_distance("top", r0, c, bbox) % 2 == 1],
        }
        viable = [
            parity
            for parity in (0, 1)
            if len(col_groups[parity]) >= vertical_total and len(row_groups[1 - parity]) >= horizontal_total
        ]
        if not viable:
            return None
        vertical_parity = int(rng.choice(np.array(viable, dtype=int)))
        vertical_candidates = col_groups[vertical_parity]
        horizontal_candidates = row_groups[1 - vertical_parity]
    else:
        vertical_candidates = middle_cols
        horizontal_candidates = middle_rows

    chosen_cols = _pick_without_replacement(rng, vertical_candidates, vertical_total)
    chosen_rows = _pick_without_replacement(rng, horizontal_candidates, horizontal_total)
    rng.shuffle(chosen_cols)
    rng.shuffle(chosen_rows)

    top_cols = sorted(chosen_cols[:top_count])
    bottom_cols = sorted(chosen_cols[top_count:])
    left_rows = sorted(chosen_rows[:left_count])
    right_rows = sorted(chosen_rows[left_count:])

    top_seed_rows = _pick_without_replacement(rng, high_rows if cross_mode else low_rows, top_count)
    bottom_seed_rows = _pick_without_replacement(rng, low_rows if cross_mode else high_rows, bottom_count)
    left_seed_cols = _pick_without_replacement(rng, high_cols if cross_mode else low_cols, left_count)
    right_seed_cols = _pick_without_replacement(rng, low_cols if cross_mode else high_cols, right_count)

    colors = list(rng.permutation(np.array(PALETTE, dtype=int)).tolist())
    bg = int(colors.pop())
    frame_color = int(colors.pop())
    seed_count = vertical_total + horizontal_total
    seed_colors = [int(color) for color in colors[:seed_count]]

    grid = np.full((h, w), bg, dtype=int)
    _build_frame(grid, bbox, frame_color)
    for col in top_cols:
        grid[r0, col] = bg
    for col in bottom_cols:
        grid[r1, col] = bg
    for row in left_rows:
        grid[row, c0] = bg
    for row in right_rows:
        grid[row, c1] = bg

    placements = []
    for col, row in zip(top_cols, top_seed_rows):
        placements.append((row, col))
    for col, row in zip(bottom_cols, bottom_seed_rows):
        placements.append((row, col))
    for row, col in zip(left_rows, left_seed_cols):
        placements.append((row, col))
    for row, col in zip(right_rows, right_seed_cols):
        placements.append((row, col))

    assert len(placements) == seed_count
    assert len(set(placements)) == seed_count
    for (row, col), color in zip(placements, seed_colors):
        grid[row, col] = color

    try:
        scene = _extract_scene(grid)
    except AssertionError:
        return None

    if _count_interior_crossings(scene) == 0 and cross_mode:
        return None
    if _count_interior_crossings(scene) != 0 and not cross_mode:
        return None

    try:
        _ = main(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
