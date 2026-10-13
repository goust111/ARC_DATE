from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
PALETTE = tuple(range(1, 10))

GRID_HEIGHT_RANGE = (12, 22)
GRID_WIDTH_RANGE = (12, 22)
OBJECT_COUNT_OPTIONS = (3, 4, 5)
AREA_VALUES = tuple(range(3, 9))
MAX_GENERATION_ATTEMPTS = 400

_GEN_CALL_COUNT = 0


def _border_cycle(shape: tuple[int, int]) -> list[tuple[int, int]]:
    height, width = map(int, shape)
    coords: list[tuple[int, int]] = []
    for c in range(width):
        coords.append((0, c))
    for r in range(1, height):
        coords.append((r, width - 1))
    for c in range(width - 2, -1, -1):
        coords.append((height - 1, c))
    for r in range(height - 2, 0, -1):
        coords.append((r, 0))
    return coords


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _extract_border_segments(grid: np.ndarray) -> list[dict]:
    border = _border_cycle(grid.shape)
    values = [int(grid[r, c]) for r, c in border]
    assert any(v == BLACK for v in values), "legend segments must be separated by black border cells"
    start = next(i for i, value in enumerate(values) if value == BLACK)
    rotated_coords = [border[(start + i) % len(border)] for i in range(len(border))]
    rotated_values = [values[(start + i) % len(border)] for i in range(len(border))]

    segments: list[dict] = []
    i = 0
    while i < len(rotated_values):
        if rotated_values[i] == BLACK:
            i += 1
            continue
        color = int(rotated_values[i])
        j = i
        while j < len(rotated_values) and rotated_values[j] == color:
            j += 1
        assert j == len(rotated_values) or rotated_values[j] == BLACK, "legend segments cannot touch"
        coords = np.array(rotated_coords[i:j], dtype=int)
        segments.append(
            {
                "color": color,
                "length": int(j - i),
                "coords": coords,
            }
        )
        i = j

    lengths = [seg["length"] for seg in segments]
    colors = [seg["color"] for seg in segments]
    assert segments, "expected at least one border legend segment"
    assert len(lengths) == len(set(lengths)), "legend segment lengths must be unique"
    assert len(colors) == len(set(colors)), "legend segment colors must be unique"
    return segments


def _find_interior_objects(grid: np.ndarray) -> list[dict]:
    height, width = map(int, grid.shape)
    interior = grid[1 : height - 1, 1 : width - 1]
    visited = np.zeros(interior.shape, dtype=bool)
    objects: list[dict] = []

    for r in range(interior.shape[0]):
        for c in range(interior.shape[1]):
            if visited[r, c] or interior[r, c] == BLACK:
                continue
            color = int(interior[r, c])
            queue: deque[tuple[int, int]] = deque([(r, c)])
            visited[r, c] = True
            coords: list[tuple[int, int]] = []

            while queue:
                rr, cc = queue.popleft()
                coords.append((rr + 1, cc + 1))
                for nr, nc in _neighbors4(rr, cc):
                    if not (0 <= nr < interior.shape[0] and 0 <= nc < interior.shape[1]):
                        continue
                    if visited[nr, nc] or int(interior[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    queue.append((nr, nc))

            coords_array = np.array(coords, dtype=int)
            objects.append(
                {
                    "color": color,
                    "area": int(len(coords)),
                    "coords": coords_array,
                }
            )

    assert objects, "expected at least one interior object"
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """Correct each interior object's color using the border legend whose segment length matches the object's size.

    Input structure:
    - The outermost border contains several isolated monochrome legend segments on a black perimeter.
    - Each legend segment has a unique color and a unique length measured along the border cycle.
    - The interior contains black background plus several monochrome 4-connected objects whose areas are all distinct.

    Rule:
    - Read the border legend as a dictionary from segment length to segment color.
    - For each interior object, count its pixels and recolor the whole object to the color of the border segment with the same length.
    - After recoloring all objects, erase every border legend segment so the full border becomes black.

    Key insight:
    - The apparent color swaps between objects are a decoy; the true mapping is the visible equality between an object's area and a border segment's length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert min(g.shape) >= 3, "grid must have an interior"

    segments = _extract_border_segments(g)
    color_by_area = {int(seg["length"]): int(seg["color"]) for seg in segments}
    objects = _find_interior_objects(g)
    assert len({obj["area"] for obj in objects}) == len(objects), "object areas must be unique"

    out = g.copy()
    border = _border_cycle(g.shape)
    for r, c in border:
        out[r, c] = BLACK

    for obj in objects:
        area = int(obj["area"])
        assert area in color_by_area, "every object area must match a legend segment length"
        coords = obj["coords"]
        out[coords[:, 0], coords[:, 1]] = int(color_by_area[area])

    return out


def _normalize_cells(cells: set[tuple[int, int]]) -> np.ndarray:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    shifted = {(r - min_r, c - min_c) for r, c in cells}
    max_r = max(r for r, _ in shifted)
    max_c = max(c for _, c in shifted)
    patch = np.zeros((max_r + 1, max_c + 1), dtype=int)
    for r, c in shifted:
        patch[r, c] = 1
    return patch


def _random_shape(area: int) -> np.ndarray:
    assert area >= 1
    for _ in range(200):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = sorted(
                {
                    nbr
                    for r, c in cells
                    for nbr in _neighbors4(r, c)
                    if nbr not in cells
                }
            )
            valid: list[tuple[int, int]] = []
            limit = max(3, min(5, area))
            for candidate in frontier:
                trial = cells | {candidate}
                rows = [r for r, _ in trial]
                cols = [c for _, c in trial]
                if max(rows) - min(rows) + 1 <= limit and max(cols) - min(cols) + 1 <= limit:
                    valid.append(candidate)
            cells.add(random.choice(valid or frontier))

        patch = _normalize_cells(cells)
        if area >= 4 and (patch.shape[0] == 1 or patch.shape[1] == 1) and random.random() < 0.7:
            continue
        if area >= 5 and int(patch.sum()) == patch.size and random.random() < 0.5:
            continue
        return patch
    raise RuntimeError(f"failed to generate connected shape of area {area}")


def _positive_composition(total: int, parts: int) -> list[int]:
    assert parts >= 1
    if parts == 1:
        return [int(total)]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    values: list[int] = []
    prev = 0
    for cut in cuts + [total]:
        values.append(int(cut - prev))
        prev = cut
    return values


def _legend_layout(height: int, width: int, items: list[dict]) -> list[dict]:
    perimeter = 2 * height + 2 * width - 4
    total_length = sum(int(item["length"]) for item in items)
    assert perimeter - total_length >= len(items), "need at least one black gap between segments"
    ordered = items[:]
    random.shuffle(ordered)
    gaps = _positive_composition(perimeter - total_length, len(ordered))
    start_offset = random.randrange(perimeter)

    placements: list[dict] = []
    cursor = start_offset
    for gap, item in zip(gaps, ordered):
        cursor = (cursor + gap) % perimeter
        placements.append({"start": int(cursor), "length": int(item["length"]), "color": int(item["color"])})
        cursor = (cursor + int(item["length"])) % perimeter
    return placements


def _paint_legend(grid: np.ndarray, placements: list[dict]) -> None:
    border = _border_cycle(grid.shape)
    perimeter = len(border)
    for placement in placements:
        for offset in range(int(placement["length"])):
            r, c = border[(int(placement["start"]) + offset) % perimeter]
            grid[r, c] = int(placement["color"])


def _sample_dimensions(areas: list[int]) -> tuple[int, int]:
    required_perimeter = sum(areas) + len(areas)
    choices = [
        (height, width)
        for height in range(GRID_HEIGHT_RANGE[0], GRID_HEIGHT_RANGE[1] + 1)
        for width in range(GRID_WIDTH_RANGE[0], GRID_WIDTH_RANGE[1] + 1)
        if 2 * height + 2 * width - 4 >= required_perimeter
        and (height - 4) * (width - 4) >= max(30, 3 * sum(areas))
    ]
    if not choices:
        raise RuntimeError("no feasible grid size for sampled areas")
    return random.choice(choices)


def _choose_current_colors(target_colors: list[int], branch: str) -> tuple[list[int], list[int]]:
    n = len(target_colors)
    perm = list(range(n))
    if branch == "swap":
        i, j = random.sample(range(n), 2)
        perm[i], perm[j] = perm[j], perm[i]
    else:
        cycle_len = n if n == 3 else random.choice([3, n])
        chosen = random.sample(range(n), cycle_len)
        shifted = chosen[1:] + chosen[:1]
        for src, dst in zip(chosen, shifted):
            perm[src] = dst

    colors = [int(target_colors[perm[i]]) for i in range(n)]
    wrong = [i for i, color in enumerate(colors) if color != int(target_colors[i])]
    assert wrong, "there must be at least one miscolored object"
    return colors, wrong


def _can_place(forbidden: np.ndarray, shape: np.ndarray, top: int, left: int) -> bool:
    coords = np.argwhere(shape != 0)
    for dr, dc in coords:
        rr = int(top + dr)
        cc = int(left + dc)
        if forbidden[rr, cc]:
            return False
    return True


def _mark_forbidden(forbidden: np.ndarray, cells: np.ndarray) -> None:
    height, width = forbidden.shape
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = int(r + dr)
                nc = int(c + dc)
                if 0 <= nr < height and 0 <= nc < width:
                    forbidden[nr, nc] = True


def _place_objects(height: int, width: int, specs: list[dict]) -> list[dict] | None:
    forbidden = np.zeros((height, width), dtype=bool)
    forbidden[:2, :] = True
    forbidden[-2:, :] = True
    forbidden[:, :2] = True
    forbidden[:, -2:] = True

    placements: list[dict | None] = [None] * len(specs)
    order = sorted(range(len(specs)), key=lambda i: specs[i]["shape"].size, reverse=True)

    for idx in order:
        shape = specs[idx]["shape"]
        sh, sw = map(int, shape.shape)
        candidates: list[tuple[int, int]] = []
        for top in range(0, height - sh + 1):
            for left in range(0, width - sw + 1):
                if _can_place(forbidden, shape, top, left):
                    candidates.append((top, left))
        if not candidates:
            return None

        top, left = random.choice(candidates)
        coords_local = np.argwhere(shape != 0)
        coords = coords_local + np.array([top, left])
        placements[idx] = {
            "coords": coords.astype(int),
            "area": int(specs[idx]["area"]),
            "target_color": int(specs[idx]["target_color"]),
            "current_color": int(specs[idx]["current_color"]),
        }
        _mark_forbidden(forbidden, coords)

    return [placement for placement in placements if placement is not None]


def _render_input(height: int, width: int, placements: list[dict], legend: list[dict]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for placement in placements:
        coords = placement["coords"]
        grid[coords[:, 0], coords[:, 1]] = int(placement["current_color"])
    _paint_legend(grid, _legend_layout(height, width, legend))
    return grid


def _expected_output(inp: np.ndarray, placements: list[dict]) -> np.ndarray:
    out = inp.copy()
    for r, c in _border_cycle(inp.shape):
        out[r, c] = BLACK
    for placement in placements:
        coords = placement["coords"]
        out[coords[:, 0], coords[:, 1]] = int(placement["target_color"])
    return out


def _validate_generated_sample(inp: np.ndarray, expected: np.ndarray, branch: str) -> None:
    segments = _extract_border_segments(inp)
    objects = _find_interior_objects(inp)
    mapping = {int(seg["length"]): int(seg["color"]) for seg in segments}

    assert len(segments) == len(objects), "generator should use every legend entry exactly once"
    assert len({obj["area"] for obj in objects}) == len(objects), "object areas must be unique"
    assert set(mapping) == {int(obj["area"]) for obj in objects}, "legend lengths must match object areas exactly"

    target_colors = [int(mapping[obj["area"]]) for obj in objects]
    current_colors = [int(obj["color"]) for obj in objects]
    assert sorted(current_colors) == sorted(target_colors), "object colors must be a permutation of legend colors"

    wrong = [obj for obj in objects if int(obj["color"]) != int(mapping[obj["area"]])]
    assert wrong, "at least one object must actually be wrong-colored"
    if branch == "swap":
        assert len(wrong) == 2, "swap branch must present exactly one visible two-object swap"
        a, b = wrong
        assert int(a["color"]) == int(mapping[b["area"]])
        assert int(b["color"]) == int(mapping[a["area"]])
    else:
        assert len(wrong) >= 3, "cycle branch must defeat a simple pair-swap shortcut"

    actual = main(inp)
    assert np.array_equal(actual, expected), "solver output mismatch"
    assert np.all(actual[0, :] == BLACK) and np.all(actual[-1, :] == BLACK)
    assert np.all(actual[:, 0] == BLACK) and np.all(actual[:, -1] == BLACK)


def _generate_attempt(branch: str) -> np.ndarray | None:
    count = random.choice(OBJECT_COUNT_OPTIONS)
    areas = random.sample(AREA_VALUES, count)
    height, width = _sample_dimensions(areas)
    target_colors = random.sample(PALETTE, count)
    current_colors, _wrong = _choose_current_colors(target_colors, branch)

    specs = []
    for area, target_color, current_color in zip(areas, target_colors, current_colors):
        specs.append(
            {
                "area": int(area),
                "shape": _random_shape(int(area)),
                "target_color": int(target_color),
                "current_color": int(current_color),
            }
        )

    placements = _place_objects(height, width, specs)
    if placements is None:
        return None

    legend = [{"length": int(area), "color": int(color)} for area, color in zip(areas, target_colors)]
    inp = _render_input(height, width, placements, legend)
    expected = _expected_output(inp, placements)

    try:
        _validate_generated_sample(inp, expected, branch)
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    _GEN_CALL_COUNT += 1
    branch = "swap" if _GEN_CALL_COUNT % 2 == 1 else "cycle"
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt(branch)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample for branch {branch}")
