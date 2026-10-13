from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
MAX_GEN_ATTEMPTS = 400


def _leading_blue_run(row: np.ndarray) -> int:
    length = 0
    for value in row.tolist():
        if int(value) != BLUE:
            break
        length += 1
    return int(length)


def _marker_columns(row: np.ndarray) -> list[int]:
    return [int(c) for c in np.flatnonzero(row == RED)]


def _build_stack(row: np.ndarray) -> np.ndarray:
    row = np.asarray(row, dtype=int).reshape(-1)
    width = int(row.shape[0])
    height = width // 2
    run_length = _leading_blue_run(row)
    markers = _marker_columns(row)

    stacked = np.zeros((height, width), dtype=int)
    for r in range(height):
        end = min(run_length + r, width)
        if end > 0:
            stacked[r, :end] = BLUE
        if markers:
            stacked[r, markers] = BLACK
    return stacked


def _components4(mask: np.ndarray) -> list[dict[str, object]]:
    mask = np.asarray(mask, dtype=bool)
    height, width = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    components: list[dict[str, object]] = []

    for r0 in range(height):
        for c0 in range(width):
            if not mask[r0, c0] or seen[r0, c0]:
                continue

            stack = [(r0, c0)]
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            min_r = max_r = int(r0)
            min_c = max_c = int(c0)

            while stack:
                r, c = stack.pop()
                coords.append((int(r), int(c)))
                if r < min_r:
                    min_r = int(r)
                if r > max_r:
                    max_r = int(r)
                if c < min_c:
                    min_c = int(c)
                if c > max_c:
                    max_c = int(c)

                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))

            coords_arr = np.asarray(coords, dtype=int)
            components.append(
                {
                    "coords": coords_arr,
                    "area": int(coords_arr.shape[0]),
                    "bbox": (int(min_r), int(min_c), int(max_r), int(max_c)),
                }
            )

    return components


def _choose_largest_component(components: list[dict[str, object]]) -> dict[str, object]:
    max_area = max(int(comp["area"]) for comp in components)
    candidates = [comp for comp in components if int(comp["area"]) == max_area]
    return min(candidates, key=lambda comp: (int(comp["bbox"][0]), int(comp["bbox"][1])))


def _extreme_points(coords: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    pts = [(int(r), int(c)) for r, c in np.asarray(coords, dtype=int)]
    top = min(pts, key=lambda rc: (rc[0], rc[1]))
    bottom = max(pts, key=lambda rc: (rc[0], rc[1]))
    return top, bottom


def _manhattan_path(top: tuple[int, int], bottom: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = map(int, top)
    r1, c1 = map(int, bottom)

    dr = 0 if r0 == r1 else (1 if r1 > r0 else -1)
    dc = 0 if c0 == c1 else (1 if c1 > c0 else -1)

    path = [(int(r0), int(c0))]
    r = int(r0)
    while r != r1:
        r += dr
        path.append((int(r), int(c0)))

    c = int(c0)
    while c != c1:
        c += dc
        path.append((int(r1), int(c)))

    return path


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")
    if g.shape[0] != 1:
        raise ValueError("expected a single-row input grid")

    row = g[0].copy()
    stacked = _build_stack(row)
    components = _components4(stacked == BLUE)
    if not components:
        return stacked

    largest = _choose_largest_component(components)
    out = np.zeros_like(stacked)
    coords = np.asarray(largest["coords"], dtype=int)
    out[coords[:, 0], coords[:, 1]] = BLUE

    top, bottom = _extreme_points(coords)
    for r, c in _manhattan_path(top, bottom):
        if out[r, c] == BLACK:
            out[r, c] = RED

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stack a growing left-anchored run, let marker columns punch vertical gaps, keep the largest blue component, then route between its vertical extremes.

    Input structure:
    - The input is a single-row grid whose left edge contains one blue run and whose remaining cells are black except for 2-3 red marker pixels.
    - The row width is even, and all marker pixels lie outside the initial blue run in distinct non-adjacent columns.
    - Some marker columns are reachable by the growth process while at least one later marker column is never reached.

    Rule:
    - Expand the blue run into a stack of W/2 rows, adding one blue cell to the right on each lower row, capped by the grid width.
    - In every stacked row, any column that was a red marker in the input is forced to black, so reachable marker columns become vertical gaps.
    - Split the resulting blue foreground into 4-connected components, keep only the largest one, and break area ties by the smallest top bbox row.
    - Inside that kept component, find the topmost pixel and the bottommost pixel; draw a Manhattan path that goes vertically to the lower row and then horizontally to the lower column, painting only background cells red.

    Key insight:
    - The red input markers are not copied forward as objects; they act as fixed barrier columns whose reachability during the row-by-row growth decides the component split, which in turn decides what survives.
    """
    return _transform(grid)


def _make_input(width: int, run_length: int, markers: list[int]) -> np.ndarray:
    grid = np.zeros((1, int(width)), dtype=int)
    grid[0, : int(run_length)] = BLUE
    for c in markers:
        grid[0, int(c)] = RED
    return grid


def _analyze_input(grid: np.ndarray) -> dict[str, object]:
    row = np.asarray(grid, dtype=int)[0]
    width = int(row.shape[0])
    height = width // 2
    run_length = _leading_blue_run(row)
    markers = _marker_columns(row)
    final_length = min(run_length + height - 1, width)
    touched = [c for c in markers if c < final_length]
    effective = [c for c in markers if c <= final_length - 2]
    untouched = [c for c in markers if c >= final_length]

    stacked = _build_stack(row)
    components = _components4(stacked == BLUE)
    kept = _choose_largest_component(components)
    kept_index = next(i for i, comp in enumerate(components) if comp is kept)
    max_area = max(int(comp["area"]) for comp in components)
    ties = [comp for comp in components if int(comp["area"]) == max_area]
    tie_indices = [i for i, comp in enumerate(components) if int(comp["area"]) == max_area]
    min_top_row = min(int(comp["bbox"][0]) for comp in ties)

    return {
        "width": width,
        "height": height,
        "run_length": run_length,
        "markers": markers,
        "final_length": final_length,
        "touched": touched,
        "effective": effective,
        "untouched": untouched,
        "components": components,
        "kept": kept,
        "kept_index": kept_index,
        "ties": ties,
        "tie_indices": tie_indices,
        "min_top_row": min_top_row,
        "output": _transform(grid),
    }


def _non_adjacent(values: list[int]) -> bool:
    ordered = sorted(int(v) for v in values)
    return all(b - a > 1 for a, b in zip(ordered, ordered[1:]))


def _pick_nonadjacent(candidates: list[int], count: int, chosen: list[int] | None = None) -> list[int] | None:
    chosen = [] if chosen is None else [int(v) for v in chosen]
    pool = [int(v) for v in candidates]
    for _ in range(120):
        random.shuffle(pool)
        trial = chosen.copy()
        for value in pool:
            if all(abs(int(value) - existing) > 1 for existing in trial):
                trial.append(int(value))
            if len(trial) == len(chosen) + count:
                return trial[len(chosen) :]
    return None


def _is_valid_example(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.shape[0] != 1:
        return False

    analysis = _analyze_input(g)
    width = int(analysis["width"])
    height = int(analysis["height"])
    run_length = int(analysis["run_length"])
    markers = list(analysis["markers"])
    final_length = int(analysis["final_length"])
    components = list(analysis["components"])
    kept = analysis["kept"]
    kept_index = int(analysis["kept_index"])
    tie_indices = list(analysis["tie_indices"])

    if width < 10 or width > 16 or width % 2 != 0:
        return False
    if run_length < 3 or run_length > 6 or run_length > height:
        return False
    if len(markers) not in (2, 3):
        return False
    if not _non_adjacent(markers):
        return False
    if any(c < run_length for c in markers):
        return False
    if not any(c < height for c in analysis["effective"]):
        return False
    if not analysis["untouched"]:
        return False
    if not any(c >= height for c in analysis["untouched"]):
        return False
    if final_length >= width:
        return False
    if len(components) < 2:
        return False
    if int(kept["area"]) >= sum(int(comp["area"]) for comp in components):
        return False
    if not any(int(comp["area"]) < int(kept["area"]) for i, comp in enumerate(components) if i != kept_index):
        return False
    if any(
        int(components[i]["bbox"][0]) == int(analysis["min_top_row"])
        for i in tie_indices
        if i != kept_index
    ):
        return False

    out = np.asarray(analysis["output"], dtype=int)
    if out.ndim != 2 or max(out.shape) > 30:
        return False
    return np.array_equal(main(g), out)


def _generate_attempt() -> np.ndarray | None:
    width = random.choice([10, 12, 14, 16])
    height = width // 2
    run_length = random.randint(3, min(6, height - 1))
    final_length = run_length + height - 1
    if final_length >= width:
        return None

    active_left_pool = list(range(run_length, min(height, final_length - 1)))
    effective_pool = list(range(run_length, final_length - 1))
    untouched_pool = list(range(final_length, width))
    if not active_left_pool or not untouched_pool:
        return None

    modes: list[str] = ["one_active_one_untouched"]
    if len(untouched_pool) >= 2:
        modes.append("one_active_two_untouched")
    if len(effective_pool) >= 2:
        modes.append("two_active_one_untouched")
    mode = random.choice(modes)

    first_active = random.choice(active_left_pool)
    markers = [int(first_active)]

    if mode == "one_active_one_untouched":
        extra = _pick_nonadjacent(untouched_pool, 1, markers)
    elif mode == "one_active_two_untouched":
        extra = _pick_nonadjacent(untouched_pool, 2, markers)
    else:
        second_active = _pick_nonadjacent([c for c in effective_pool if c != first_active], 1, markers)
        if second_active is None:
            return None
        markers.extend(second_active)
        extra = _pick_nonadjacent(untouched_pool, 1, markers)

    if extra is None:
        return None
    markers.extend(extra)
    markers = sorted(set(markers))
    if len(markers) not in (2, 3):
        return None

    grid = _make_input(width, run_length, markers)
    return grid if _is_valid_example(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
