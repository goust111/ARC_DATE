from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
RAY_COLORS = (1, 3, 4)
RECT = 5
MOVING = 7
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _right_edge_run(row: np.ndarray) -> tuple[int, int] | None:
    cols = np.flatnonzero(row == MOVING)
    if len(cols) == 0:
        return None
    start = int(cols[0])
    end = int(cols[-1])
    if end != row.shape[0] - 1 or len(cols) != end - start + 1:
        raise AssertionError("orange pixels must form one contiguous right-edge run")
    return start, end


def _find_rectangle_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    gray_components = _components_of_color(grid, RECT)
    if not gray_components:
        raise AssertionError("expected a gray rectangle")
    rect_coords = max(gray_components, key=lambda pts: int(pts.shape[0]))
    bbox = _bbox_coords(rect_coords)
    r0, c0, r1, c1 = bbox
    if not np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == RECT):
        raise AssertionError("largest gray component must be a solid rectangle")
    return bbox


def _find_source_pixels(grid: np.ndarray) -> list[tuple[int, int, int]]:
    out: list[tuple[int, int, int]] = []
    for color in sorted(set(map(int, np.unique(grid))) - {BG, RECT, MOVING}):
        for comp in _components_of_color(grid, color):
            if comp.shape[0] != 1:
                raise AssertionError("non-rectangle, non-orange foreground must be isolated pixels")
            r, c = map(int, comp[0])
            out.append((r, c, int(color)))
    return out


def _draw_rays_from_sources(
    grid: np.ndarray,
    rect_bbox: tuple[int, int, int, int],
    sources: list[tuple[int, int, int]],
) -> np.ndarray:
    out = grid.copy()
    r0, c0, r1, c1 = rect_bbox
    for r, c, color in sources:
        if r0 <= r <= r1 and c > c1:
            for cc in range(c1 + 1, c + 1):
                out[r, cc] = color
        elif r0 <= r <= r1 and c < c0:
            for cc in range(c, c0):
                out[r, cc] = color
        elif c0 <= c <= c1 and r < r0:
            for rr in range(r, r0):
                out[rr, c] = color
        elif c0 <= c <= c1 and r > r1:
            for rr in range(r1 + 1, r + 1):
                out[rr, c] = color
    return out


def _slide_orange_runs(original: np.ndarray, ray_grid: np.ndarray) -> np.ndarray:
    out = ray_grid.copy()
    h, _w = original.shape
    for r in range(h):
        run = _right_edge_run(original[r])
        if run is None:
            continue
        start, end = run
        run_len = end - start + 1
        blockers = np.flatnonzero((ray_grid[r, :start] != BG) & (ray_grid[r, :start] != MOVING))
        target_start = int(blockers[-1]) + 1 if len(blockers) else 0
        out[r, start : end + 1] = BG
        out[r, target_start : target_start + run_len] = MOVING
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw rays from aligned singleton pixels to the gray rectangle, then slide each right-edge orange run left until it packs against the row's newest blocker.

    Input structure:
    - The background is black.
    - A single solid gray rectangle sits near the middle-left of the grid.
    - Blue, green, and yellow singleton pixels lie outside the rectangle and may share one rectangle row or one rectangle column.
    - Some rows contain one contiguous orange run flush with the right edge.

    Rule:
    - Find the gray rectangle as the largest solid gray component.
    - For each non-orange, non-gray singleton pixel, if it shares a rectangle row, draw a horizontal ray from the pixel toward the rectangle and stop just before entering the rectangle; if it shares a rectangle column, draw the analogous vertical ray.
    - After all rays are drawn, process each orange row independently: clear its original right-edge run and repaint that whole run as far left as possible while staying immediately to the right of the row's rightmost non-black, non-orange blocker; if no blocker exists, start the run at column 0.

    Key insight:
    - The blockers are not fixed in the input: they are created by the first stage, so the sliding destination can only be determined after the rays have been added.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    rect_bbox = _find_rectangle_bbox(g)
    sources = _find_source_pixels(g)
    ray_grid = _draw_rays_from_sources(g, rect_bbox, sources)
    return _slide_orange_runs(g, ray_grid)


def _ray_row_has_new_pixels(original: np.ndarray, ray_grid: np.ndarray, row: int) -> bool:
    new_mask = (ray_grid[row] != original[row]) & (ray_grid[row] != MOVING)
    return bool(np.any(new_mask))


def _build_spec() -> dict | None:
    h = random.randint(12, 16)
    w = random.randint(16, 21)

    rect_h = random.randint(3, min(5, h - 7))
    rect_w = random.randint(3, min(5, w - 11))
    r0 = random.randint(3, h - rect_h - 4)
    c0 = random.randint(3, min(7, w - rect_w - 8))
    r1 = r0 + rect_h - 1
    c1 = c0 + rect_w - 1

    rect_rows = list(range(r0, r1 + 1))
    if len(rect_rows) < 3:
        return None

    interaction_rows = sorted(random.sample(rect_rows, 2))
    remaining_rect_rows = [r for r in rect_rows if r not in interaction_rows]
    if not remaining_rect_rows:
        return None
    rect_only_row = random.choice(remaining_rect_rows)

    vert_col = random.randint(c0, c1)
    vertical_options: list[tuple[int, list[int]]] = []
    for src_r in range(0, r0 - 1):
        crossed = list(range(src_r + 1, r0))
        if len(crossed) >= 1:
            vertical_options.append((src_r, crossed))
    for src_r in range(r1 + 2, h):
        crossed = list(range(r1 + 1, src_r))
        if len(crossed) >= 1:
            vertical_options.append((src_r, crossed))
    vertical_options = [(src_r, crossed) for src_r, crossed in vertical_options if src_r not in interaction_rows]
    if not vertical_options:
        return None
    vert_src_r, crossed_rows = random.choice(vertical_options)
    vertical_barrier_row = random.choice(crossed_rows)

    vertical_ray_rows = set(crossed_rows)
    vertical_ray_rows.add(vert_src_r)
    left_edge_candidates = [
        r
        for r in range(h)
        if r not in rect_rows
        and r not in vertical_ray_rows
        and r not in interaction_rows
        and r != rect_only_row
    ]
    if not left_edge_candidates:
        return None
    left_edge_row = random.choice(left_edge_candidates)

    orange_rows = [interaction_rows[0], interaction_rows[1], rect_only_row, vertical_barrier_row, left_edge_row]
    run_lengths = {r: random.randint(2, 4) for r in orange_rows}

    colors = list(RAY_COLORS)
    random.shuffle(colors)
    horiz_specs = []
    used_cols: set[int] = set()
    for row, color in zip(interaction_rows, colors[:2], strict=False):
        lo = c1 + 2
        hi = w - run_lengths[row] - 2
        if lo > hi:
            return None
        choices = [c for c in range(lo, hi + 1) if c not in used_cols]
        if not choices:
            return None
        src_c = random.choice(choices)
        used_cols.add(src_c)
        horiz_specs.append((row, src_c, color))

    vertical_spec = (vert_src_r, vert_col, colors[2])

    rect_target = c1 + 1
    if rect_target + run_lengths[rect_only_row] > w:
        return None
    if vert_col + 1 + run_lengths[vertical_barrier_row] > w:
        return None

    return {
        "shape": (h, w),
        "rect": (r0, c0, r1, c1),
        "horizontal_sources": horiz_specs,
        "vertical_source": vertical_spec,
        "orange_rows": orange_rows,
        "run_lengths": run_lengths,
        "interaction_rows": interaction_rows,
        "rect_only_row": rect_only_row,
        "vertical_barrier_row": vertical_barrier_row,
        "left_edge_row": left_edge_row,
    }


def _render_input(spec: dict) -> np.ndarray:
    h, w = spec["shape"]
    r0, c0, r1, c1 = spec["rect"]
    grid = np.full((h, w), BG, dtype=int)
    grid[r0 : r1 + 1, c0 : c1 + 1] = RECT

    for r, c, color in spec["horizontal_sources"]:
        grid[r, c] = color

    vr, vc, vcolor = spec["vertical_source"]
    grid[vr, vc] = vcolor

    for row in spec["orange_rows"]:
        run_len = spec["run_lengths"][row]
        grid[row, w - run_len : w] = MOVING
    return grid


def _oracle_output_from_spec(spec: dict) -> np.ndarray:
    grid = _render_input(spec)
    r0, c0, r1, c1 = spec["rect"]
    rayed = grid.copy()

    for row, src_c, color in spec["horizontal_sources"]:
        for c in range(c1 + 1, src_c + 1):
            rayed[row, c] = color

    vr, vc, vcolor = spec["vertical_source"]
    if vr < r0:
        for r in range(vr, r0):
            rayed[r, vc] = vcolor
    else:
        for r in range(r1 + 1, vr + 1):
            rayed[r, vc] = vcolor

    out = rayed.copy()
    w = grid.shape[1]
    for row in spec["orange_rows"]:
        run_len = spec["run_lengths"][row]
        start = w - run_len
        blockers = np.flatnonzero((rayed[row, :start] != BG) & (rayed[row, :start] != MOVING))
        target_start = int(blockers[-1]) + 1 if len(blockers) else 0
        out[row, start:w] = BG
        out[row, target_start : target_start + run_len] = MOVING
    return out


def _validate_spec_and_grid(spec: dict, grid: np.ndarray) -> bool:
    h, w = grid.shape
    if h > 30 or w > 30:
        return False

    rect_bbox = _find_rectangle_bbox(grid)
    if rect_bbox != spec["rect"]:
        return False

    sources = sorted(_find_source_pixels(grid))
    expected_sources = sorted(spec["horizontal_sources"] + [spec["vertical_source"]])
    if sources != expected_sources:
        return False

    ray_grid = _draw_rays_from_sources(grid, rect_bbox, sources)
    out = main(grid)
    oracle = _oracle_output_from_spec(spec)
    if not np.array_equal(out, oracle):
        return False
    if np.array_equal(out, grid):
        return False

    rect_r0, rect_c0, rect_r1, rect_c1 = rect_bbox
    interaction_rows = spec["interaction_rows"]
    if len(interaction_rows) != 2:
        return False
    for row, src_c, _color in spec["horizontal_sources"]:
        if row not in interaction_rows:
            return False
        if not _ray_row_has_new_pixels(grid, ray_grid, row):
            return False
        run_len = spec["run_lengths"][row]
        new_cols = np.flatnonzero(out[row] == MOVING)
        if len(new_cols) != run_len or int(new_cols[0]) != src_c + 1:
            return False

    rect_only_row = spec["rect_only_row"]
    if _ray_row_has_new_pixels(grid, ray_grid, rect_only_row):
        return False
    rect_only_cols = np.flatnonzero(out[rect_only_row] == MOVING)
    if int(rect_only_cols[0]) != rect_c1 + 1:
        return False

    vertical_row = spec["vertical_barrier_row"]
    if not _ray_row_has_new_pixels(grid, ray_grid, vertical_row):
        return False
    vertical_cols = np.flatnonzero(out[vertical_row] == MOVING)
    if int(vertical_cols[0]) != spec["vertical_source"][1] + 1:
        return False

    left_edge_row = spec["left_edge_row"]
    if _ray_row_has_new_pixels(grid, ray_grid, left_edge_row):
        return False
    left_cols = np.flatnonzero(out[left_edge_row] == MOVING)
    if int(left_cols[0]) != 0:
        return False

    orange_row_set = set(spec["orange_rows"])
    if len(orange_row_set) != 5:
        return False
    if Counter(spec["orange_rows"]).most_common(1)[0][1] != 1:
        return False
    if any(_right_edge_run(grid[row]) is None for row in orange_row_set):
        return False
    if len({int(np.flatnonzero(out[row] == MOVING)[0]) for row in orange_row_set}) < 3:
        return False

    if not (rect_r0 <= interaction_rows[0] <= rect_r1 and rect_r0 <= interaction_rows[1] <= rect_r1):
        return False
    if rect_only_row < rect_r0 or rect_only_row > rect_r1:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(400):
        spec = _build_spec()
        if spec is None:
            continue
        grid = _render_input(spec)
        if _validate_spec_and_grid(spec, grid):
            return grid
    raise RuntimeError("failed to generate a valid ray-and-slide ARC input")
