from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
RAY_DIRS = {
    1: [(-1, 0)],
    2: [(-1, 0), (1, 0)],
    3: [(-1, 0), (1, -1), (1, 1)],
    4: [(-1, 0), (1, 0), (0, -1), (0, 1)],
}

DARK_BG = 0
LIGHT_BG = 5
FG_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

GRID_H_RANGE = (21, 30)
GRID_W_RANGE = (21, 30)
OBJECT_COUNT_RANGE = (1, 4)
CORE_SIZE_CHOICES = (3, 5, 7)
MIN_VISIBLE_RAY_CELLS = 4
PLACEMENT_TRIES = 200
SPEC_TRIES = 80
GENERATE_RETRIES = 500


def _checkerboard(shape: tuple[int, int], even_color: int, odd_color: int) -> np.ndarray:
    rows, cols = np.indices(shape)
    return np.where((rows + cols) % 2 == 0, int(even_color), int(odd_color)).astype(int)


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _infer_checkerboard_colors(grid: np.ndarray) -> tuple[int, int]:
    counts = Counter(map(int, grid.ravel()))
    assert len(counts) >= 3, "expected checkerboard plus foreground colors"
    candidates = [color for color, _count in counts.most_common(2)]
    a, b = map(int, candidates)

    rows, cols = np.indices(grid.shape)
    even = ((rows + cols) % 2) == 0
    score_ab = int(np.count_nonzero(grid[even] == a) + np.count_nonzero(grid[~even] == b))
    score_ba = int(np.count_nonzero(grid[even] == b) + np.count_nonzero(grid[~even] == a))
    if score_ab >= score_ba:
        return a, b
    return b, a


def _find_components(
    grid: np.ndarray,
    bg_even: int,
    bg_odd: int,
) -> list[dict[str, object]]:
    expected = _checkerboard(grid.shape, bg_even, bg_odd)
    foreground = grid != expected
    visited = np.zeros(grid.shape, dtype=bool)
    height, width = grid.shape
    components: list[dict[str, object]] = []

    for r in range(height):
        for c in range(width):
            if not foreground[r, c] or visited[r, c]:
                continue
            color = int(grid[r, c])
            queue = deque([(r, c)])
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((int(cr), int(cc)))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < height and 0 <= nc < width):
                        continue
                    if visited[nr, nc] or not foreground[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    queue.append((nr, nc))

            components.append({"color": color, "coords": coords, "bbox": _bbox(coords)})

    components.sort(key=lambda comp: tuple(comp["bbox"]))  # stable ray-claim ordering
    return components


def _same_color_degree(coord_set: set[tuple[int, int]], r: int, c: int) -> int:
    return sum((r + dr, c + dc) in coord_set for dr, dc in DIRS4)


def _component_info(comp: dict[str, object]) -> dict[str, object]:
    coords = list(comp["coords"])
    coord_set = set(coords)
    spikes = sorted((r, c) for r, c in coords if _same_color_degree(coord_set, r, c) == 1)
    assert 1 <= len(spikes) <= 4, "each object must encode 1..4 spikes"

    spike_set = set(spikes)
    core = sorted(coord_set - spike_set)
    assert core, "spikes cannot consume the whole object"

    core_queue = deque([core[0]])
    seen = {core[0]}
    core_set = set(core)
    while core_queue:
        r, c = core_queue.popleft()
        for dr, dc in DIRS4:
            nxt = (r + dr, c + dc)
            if nxt in core_set and nxt not in seen:
                seen.add(nxt)
                core_queue.append(nxt)
    assert seen == core_set, "object core must stay 4-connected after spike removal"

    r0, c0, r1, c1 = _bbox(core)
    center = ((r0 + r1) // 2, (c0 + c1) // 2)
    return {
        "color": int(comp["color"]),
        "coords": coords,
        "core": core,
        "spikes": spikes,
        "center": center,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count 1-pixel spikes on each colored region, remove them, then cast that
    many rays from the cleaned region center across the checkerboard background.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg_even, bg_odd = _infer_checkerboard_colors(g)
    checker = _checkerboard(g.shape, bg_even, bg_odd)
    components = [_component_info(comp) for comp in _find_components(g, bg_even, bg_odd)]
    assert components, "expected at least one foreground region"

    out = g.copy()
    removed_spikes: set[tuple[int, int]] = set()
    for comp in components:
        for r, c in comp["spikes"]:
            out[r, c] = checker[r, c]
            removed_spikes.add((r, c))

    claims: dict[tuple[int, int], int] = {}
    height, width = g.shape
    for comp in components:
        color = int(comp["color"])
        center_r, center_c = map(int, comp["center"])
        for dr, dc in RAY_DIRS[len(comp["spikes"])]:
            r, c = center_r, center_c
            while True:
                r += dr
                c += dc
                if not (0 <= r < height and 0 <= c < width):
                    break
                if (r, c) in removed_spikes:
                    continue
                if int(out[r, c]) != int(checker[r, c]):
                    continue
                prev = claims.get((r, c))
                assert prev in (None, color), "ray claims must not conflict"
                claims[(r, c)] = color

    for (r, c), color in claims.items():
        out[r, c] = int(color)
    return out


def _on_ray(
    center: tuple[int, int],
    point: tuple[int, int],
    direction: tuple[int, int],
) -> bool:
    dr = point[0] - center[0]
    dc = point[1] - center[1]
    step_r, step_c = direction
    if step_r == 0:
        return dr == 0 and step_c != 0 and dc * step_c > 0
    if step_c == 0:
        return dc == 0 and step_r != 0 and dr * step_r > 0
    return dr == dc * (step_r // step_c) and dr * step_r > 0


def _choose_spikes(height: int, width: int, count: int) -> list[tuple[int, int]] | None:
    center = (height // 2, width // 2)
    candidates: list[tuple[int, int]] = []
    for c in range(width):
        candidates.append((-1, c))
        candidates.append((height, c))
    for r in range(height):
        candidates.append((r, -1))
        candidates.append((r, width))

    ray_dirs = RAY_DIRS[count]
    candidates = [
        cell
        for cell in candidates
        if not any(_on_ray(center, cell, ray_dir) for ray_dir in ray_dirs)
    ]
    if len(candidates) < count:
        return None

    for _ in range(SPEC_TRIES):
        shuffled = candidates[:]
        random.shuffle(shuffled)
        chosen: list[tuple[int, int]] = []
        for cell in shuffled:
            if all(abs(cell[0] - other[0]) + abs(cell[1] - other[1]) > 1 for other in chosen):
                chosen.append(cell)
                if len(chosen) == count:
                    return chosen
    return None


def _dilate8(points: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    height, width = shape
    out: set[tuple[int, int]] = set()
    for r, c in points:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width:
                    out.add((nr, nc))
    return out


def _make_spec(color: int, spike_count: int) -> dict[str, object] | None:
    core_h = random.choice(CORE_SIZE_CHOICES)
    core_w = random.choice(CORE_SIZE_CHOICES)
    spikes = _choose_spikes(core_h, core_w, spike_count)
    if spikes is None:
        return None

    core = {(r, c) for r in range(core_h) for c in range(core_w)}
    return {
        "color": int(color),
        "spike_count": int(spike_count),
        "core_h": core_h,
        "core_w": core_w,
        "center_local": (core_h // 2, core_w // 2),
        "core_local": core,
        "spikes_local": set(spikes),
    }


def _place_spec(
    spec: dict[str, object],
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    occupied_buffer: set[tuple[int, int]],
    reserved_rays: set[tuple[int, int]],
) -> dict[str, object] | None:
    height, width = shape
    core_h = int(spec["core_h"])
    core_w = int(spec["core_w"])
    center_local = tuple(spec["center_local"])
    core_local = set(spec["core_local"])
    spikes_local = set(spec["spikes_local"])

    row_options = list(range(1, height - core_h - 1))
    col_options = list(range(1, width - core_w - 1))
    random.shuffle(row_options)
    random.shuffle(col_options)

    for _ in range(PLACEMENT_TRIES):
        r0 = random.choice(row_options)
        c0 = random.choice(col_options)

        core_global = {(r0 + r, c0 + c) for r, c in core_local}
        spikes_global = {(r0 + r, c0 + c) for r, c in spikes_local}
        shape_global = core_global | spikes_global
        if shape_global & occupied:
            continue
        if shape_global & occupied_buffer:
            continue

        ray_cells: set[tuple[int, int]] = set()
        center = (r0 + int(center_local[0]), c0 + int(center_local[1]))
        valid = True
        for dr, dc in RAY_DIRS[int(spec["spike_count"])]:
            r, c = center
            visible = 0
            while True:
                r += dr
                c += dc
                if not (0 <= r < height and 0 <= c < width):
                    break
                point = (r, c)
                if point in spikes_global:
                    valid = False
                    break
                if point in core_global:
                    continue
                if point in occupied or point in reserved_rays:
                    valid = False
                    break
                ray_cells.add(point)
                visible += 1
            if not valid or visible < MIN_VISIBLE_RAY_CELLS:
                valid = False
                break
        if not valid:
            continue

        return {
            "color": int(spec["color"]),
            "core": core_global,
            "spikes": spikes_global,
            "shape": shape_global,
            "shape_buffer": _dilate8(shape_global, shape),
            "rays": ray_cells,
        }
    return None


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    grid = _checkerboard((height, width), DARK_BG, LIGHT_BG)

    object_count = random.randint(*OBJECT_COUNT_RANGE)
    colors = random.sample(FG_COLORS, object_count)
    spike_counts = [random.randint(1, 4) for _ in range(object_count)]
    specs: list[dict[str, object]] = []
    for color, spike_count in zip(colors, spike_counts):
        spec = _make_spec(color, spike_count)
        if spec is None:
            return None
        specs.append(spec)
    specs.sort(key=lambda spec: (-int(spec["spike_count"]), -(int(spec["core_h"]) * int(spec["core_w"]))))

    placed: list[dict[str, object]] = []
    occupied: set[tuple[int, int]] = set()
    occupied_buffer: set[tuple[int, int]] = set()
    reserved_rays: set[tuple[int, int]] = set()
    for spec in specs:
        placed_spec = _place_spec(spec, grid.shape, occupied, occupied_buffer, reserved_rays)
        if placed_spec is None:
            return None
        placed.append(placed_spec)
        occupied |= set(placed_spec["shape"])
        occupied_buffer |= set(placed_spec["shape_buffer"])
        reserved_rays |= set(placed_spec["rays"])

    for obj in placed:
        for r, c in obj["shape"]:
            grid[r, c] = int(obj["color"])

    counts = Counter(map(int, grid.ravel()))
    if {color for color, _count in counts.most_common(2)} != {DARK_BG, LIGHT_BG}:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is None:
            continue
        try:
            out = main(grid)
        except Exception:
            continue
        if np.array_equal(out, grid):
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
