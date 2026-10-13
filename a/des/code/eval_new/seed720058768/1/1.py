from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
DIRS8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)


def _on_edge(r: int, c: int, h: int, w: int) -> bool:
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _find_axes(grid: np.ndarray) -> tuple[list[dict], set[tuple[int, int]]]:
    h, w = map(int, grid.shape)
    axes: list[dict] = []
    marker_cells: set[tuple[int, int]] = set()

    for color in sorted(int(v) for v in np.unique(grid) if int(v) != BG):
        coords = np.argwhere(grid == color)
        if len(coords) != 2:
            continue
        pts = [tuple(map(int, rc)) for rc in coords]
        if not all(_on_edge(r, c, h, w) for r, c in pts):
            continue

        (r1, c1), (r2, c2) = pts
        if r1 == r2 and {c1, c2} == {0, w - 1}:
            axes.append({"kind": "h", "pos": int(r1), "color": int(color)})
            marker_cells.update(pts)
        elif c1 == c2 and {r1, r2} == {0, h - 1}:
            axes.append({"kind": "v", "pos": int(c1), "color": int(color)})
            marker_cells.update(pts)

    assert axes, "expected at least one edge marker pair"
    return axes, marker_cells


def _components_8(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    visited = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(visited[r, c]):
                continue

            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIRS8:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if not bool(mask[nr, nc]) or bool(visited[nr, nc]):
                        continue
                    visited[nr, nc] = True
                    stack.append((nr, nc))

            comps.append(np.array(coords, dtype=int))

    return comps


def _shape_components(grid: np.ndarray, marker_cells: set[tuple[int, int]]) -> list[np.ndarray]:
    mask = grid != BG
    for r, c in marker_cells:
        mask[r, c] = False

    comps = _components_8(mask)
    assert comps, "expected at least one interior shape"
    for comp in comps:
        colors = {int(grid[r, c]) for r, c in comp}
        assert len(colors) == 1, "each shape must be monochromatic"
        assert all(not _on_edge(int(r), int(c), *grid.shape) for r, c in comp), "shapes must be interior"
    return comps


def _bbox_center2(coords: np.ndarray) -> tuple[int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min() + rows.max()), int(cols.min() + cols.max())


def _nearest_axis_index_or_none(coords: np.ndarray, axes: list[dict]) -> int | None:
    center_r2, center_c2 = _bbox_center2(coords)
    distances = [
        abs(center_r2 - 2 * axis["pos"]) if axis["kind"] == "h" else abs(center_c2 - 2 * axis["pos"])
        for axis in axes
    ]
    best = min(distances)
    if distances.count(best) != 1:
        return None
    return int(distances.index(best))


def _reflect(coords: np.ndarray, axis: dict) -> np.ndarray:
    out = coords.copy()
    if axis["kind"] == "h":
        out[:, 0] = 2 * int(axis["pos"]) - out[:, 0]
    else:
        out[:, 1] = 2 * int(axis["pos"]) - out[:, 1]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each interior shape across the nearest marker-defined axis and recolor the mirrored copy by that axis.

    Input structure:
    - The background is black.
    - The interior contains several monochromatic 8-connected shapes that do not touch the border.
    - Border markers appear as same-color pixel pairs on opposite edges, either left-right on one row or top-bottom on one column.

    Rule:
    - Treat each opposite-edge marker pair as an implicit mirror axis: a same-row left-right pair defines a horizontal axis, and a same-column top-bottom pair defines a vertical axis.
    - For every interior shape, compute the center of its bounding box and assign the shape to the axis with the smallest perpendicular distance to that center.
    - Keep the original shape, reflect a copy across its assigned axis, and repaint the reflected copy with the marker color of that axis; reflected pixels overwrite anything already present at those cells.

    Key insight:
    - The colored edge pairs are not foreground objects to transform themselves; they encode hidden axes, and each interior shape must first be matched to its nearest axis before the reflection can be drawn.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    axes, marker_cells = _find_axes(g)
    shapes = _shape_components(g, marker_cells)

    claims: dict[tuple[int, int], int] = {}
    for shape in shapes:
        nearest = _nearest_axis_index_or_none(shape, axes)
        assert nearest is not None, "shape-to-axis assignment must be unique"
        axis = axes[nearest]
        reflected = _reflect(shape, axis)
        assert np.all(reflected[:, 0] >= 0) and np.all(reflected[:, 0] < g.shape[0]), "reflection row out of bounds"
        assert np.all(reflected[:, 1] >= 0) and np.all(reflected[:, 1] < g.shape[1]), "reflection col out of bounds"
        for r, c in reflected:
            pt = (int(r), int(c))
            if pt in claims:
                assert claims[pt] == int(axis["color"]), "overlapping mirrored copies must agree in color"
            claims[pt] = int(axis["color"])

    out = g.copy()
    for (r, c), color in claims.items():
        out[r, c] = int(color)
    return out


def _trim(coords: set[tuple[int, int]]) -> np.ndarray:
    arr = np.array(sorted(coords), dtype=int)
    arr[:, 0] -= int(arr[:, 0].min())
    arr[:, 1] -= int(arr[:, 1].min())
    return arr


def _random_shape(rng: random.Random) -> np.ndarray:
    for _ in range(200):
        h = rng.randint(2, 5)
        w = rng.randint(2, 5)
        area = rng.randint(3, min(8, h * w - 1))
        cells = {(rng.randrange(h), rng.randrange(w))}

        while len(cells) < area:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for dr, dc in DIRS8:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))

        if len(cells) != area:
            continue

        shape = _trim(cells)
        hh = int(shape[:, 0].max()) + 1
        ww = int(shape[:, 1].max()) + 1
        if hh < 2 or ww < 2:
            continue
        if len(shape) == hh * ww:
            continue
        return shape

    raise RuntimeError("failed to sample a non-rectangular shape")


def _translate(coords: np.ndarray, r0: int, c0: int) -> np.ndarray:
    out = coords.copy()
    out[:, 0] += int(r0)
    out[:, 1] += int(c0)
    return out


def _inside_margin(coords: np.ndarray, h: int, w: int, margin: int = 1) -> bool:
    return bool(
        np.all(coords[:, 0] >= margin)
        and np.all(coords[:, 0] < h - margin)
        and np.all(coords[:, 1] >= margin)
        and np.all(coords[:, 1] < w - margin)
    )


def _separated(coords: np.ndarray, occupied: set[tuple[int, int]]) -> bool:
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (int(r) + dr, int(c) + dc) in occupied:
                    return False
    return True


def _sample_spaced_positions(rng: random.Random, values: list[int], count: int, gap: int) -> list[int] | None:
    pool = values[:]
    rng.shuffle(pool)
    chosen: list[int] = []
    for value in pool:
        if all(abs(value - old) >= gap for old in chosen):
            chosen.append(int(value))
            if len(chosen) == count:
                break
    return sorted(chosen) if len(chosen) == count else None


def _object_modes(kind: str) -> tuple[str, str, str]:
    return ("above", "below", "touch") if kind == "h" else ("left", "right", "touch")


def _candidate_origin(
    shape: np.ndarray,
    axis: dict,
    mode: str,
    h: int,
    w: int,
    rng: random.Random,
) -> tuple[int, int]:
    hh = int(shape[:, 0].max()) + 1
    ww = int(shape[:, 1].max()) + 1

    if axis["kind"] == "h":
        c0 = rng.randint(1, w - ww - 2)
        if mode == "above":
            r0 = int(axis["pos"]) - hh - rng.randint(0, 2)
        elif mode == "below":
            r0 = int(axis["pos"]) + 1 + rng.randint(0, 2)
        else:
            hit_row = rng.choice(sorted({int(r) for r in shape[:, 0]}))
            r0 = int(axis["pos"]) - hit_row
    else:
        r0 = rng.randint(1, h - hh - 2)
        if mode == "left":
            c0 = int(axis["pos"]) - ww - rng.randint(0, 2)
        elif mode == "right":
            c0 = int(axis["pos"]) + 1 + rng.randint(0, 2)
        else:
            hit_col = rng.choice(sorted({int(c) for c in shape[:, 1]}))
            c0 = int(axis["pos"]) - hit_col

    return int(r0), int(c0)


def _touches_axis(coords: np.ndarray, axis: dict) -> bool:
    if axis["kind"] == "h":
        return any(int(r) == int(axis["pos"]) for r in coords[:, 0])
    return any(int(c) == int(axis["pos"]) for c in coords[:, 1])


def _attempt_generate(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(14, 24)
    w = rng.randint(14, 24)

    layout_mode = rng.choice(["horizontal_only", "vertical_only", "mixed"])
    n_h = 2 if layout_mode == "horizontal_only" else 0
    n_v = 2 if layout_mode == "vertical_only" else 0
    if layout_mode == "mixed":
        n_h = 1
        n_v = 1

    h_rows = _sample_spaced_positions(rng, list(range(2, h - 2)), n_h, 4) if n_h else []
    v_cols = _sample_spaced_positions(rng, list(range(2, w - 2)), n_v, 4) if n_v else []
    if h_rows is None or v_cols is None:
        return None

    axes: list[dict] = [{"kind": "h", "pos": row} for row in h_rows] + [{"kind": "v", "pos": col} for col in v_cols]
    rng.shuffle(axes)

    n_objects = rng.randint(len(axes), 4)
    all_colors = rng.sample(PALETTE, len(axes) + n_objects)
    axis_colors = all_colors[: len(axes)]
    object_colors = all_colors[len(axes) :]
    for axis, color in zip(axes, axis_colors):
        axis["color"] = int(color)

    assignments = list(range(len(axes)))
    while len(assignments) < n_objects:
        assignments.append(rng.randrange(len(axes)))
    rng.shuffle(assignments)

    grid = np.zeros((h, w), dtype=int)
    marker_cells: set[tuple[int, int]] = set()
    for axis in axes:
        if axis["kind"] == "h":
            pts = [(int(axis["pos"]), 0), (int(axis["pos"]), w - 1)]
        else:
            pts = [(0, int(axis["pos"])), (h - 1, int(axis["pos"]))]
        for r, c in pts:
            grid[r, c] = int(axis["color"])
            marker_cells.add((r, c))

    occupied: set[tuple[int, int]] = set()
    reflected_taken: set[tuple[int, int]] = set()

    for obj_color, axis_index in zip(object_colors, assignments):
        axis = axes[axis_index]
        modes = _object_modes(axis["kind"])
        wanted_mode = rng.choice(modes)

        placed = False
        for _ in range(300):
            shape = _random_shape(rng)
            r0, c0 = _candidate_origin(shape, axis, wanted_mode, h, w, rng)
            placed_coords = _translate(shape, r0, c0)

            if not _inside_margin(placed_coords, h, w, margin=1):
                continue
            if not _separated(placed_coords, occupied):
                continue
            if any((int(r), int(c)) in reflected_taken for r, c in placed_coords):
                continue

            nearest = _nearest_axis_index_or_none(placed_coords, axes)
            if nearest != axis_index:
                continue

            if wanted_mode == "touch":
                if not _touches_axis(placed_coords, axis):
                    continue
            else:
                if _touches_axis(placed_coords, axis):
                    continue

            reflected = _reflect(placed_coords, axis)
            if not _inside_margin(reflected, h, w, margin=1):
                continue
            if any((int(r), int(c)) in occupied for r, c in reflected):
                continue
            if any((int(r), int(c)) in reflected_taken for r, c in reflected):
                continue

            overlap = {(int(r), int(c)) for r, c in placed_coords} & {(int(r), int(c)) for r, c in reflected}
            if wanted_mode == "touch" and not overlap:
                continue
            if wanted_mode != "touch" and overlap:
                continue

            for r, c in placed_coords:
                grid[r, c] = int(obj_color)
                occupied.add((int(r), int(c)))
            for r, c in reflected:
                reflected_taken.add((int(r), int(c)))
            placed = True
            break

        if not placed:
            return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    if len(_shape_components(grid, marker_cells)) != n_objects:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(500):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
