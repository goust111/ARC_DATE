from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_TOOLBOX = Path("/home/lixinyu/snap/new/solvers")
if _TOOLBOX.is_dir() and str(_TOOLBOX) not in sys.path:
    sys.path.append(str(_TOOLBOX))

from arc_utils import Color, find_components, neighbors4, paste


BG = Color.BLACK
SEP = Color.ORANGE
GEN_MAX_ATTEMPTS = 600
PANEL_SIZE_RANGE = (6, 8)
AREA_CHOICES = (1, 2, 3, 4, 5, 6)
MARK_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.TEAL, Color.MAROON)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _extract_panels(grid: np.ndarray) -> list[np.ndarray]:
    panels = find_components(grid, bg=SEP, connectivity=4, monochromatic=False)
    assert len(panels) == 3, "expected exactly three panels"

    ordered = sorted(panels, key=lambda panel: tuple(map(int, panel["bbox"][:2])))
    boxes = [tuple(map(int, panel["bbox"])) for panel in ordered]
    shapes = {(r1 - r0 + 1, c1 - c0 + 1) for r0, c0, r1, c1 in boxes}
    assert len(shapes) == 1, "all panels must share one shape"

    patches: list[np.ndarray] = []
    for r0, c0, r1, c1 in boxes:
        patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        assert np.all(patch != SEP), "panel interiors cannot contain separator cells"
        patches.append(patch)
    return patches


def _component_area_map(panel: np.ndarray) -> np.ndarray:
    area_map = np.zeros(panel.shape, dtype=int)
    for component in find_components(panel, bg=BG, connectivity=4, monochromatic=True):
        coords = component["coords"]
        area = int(component["area"])
        area_map[coords[:, 0], coords[:, 1]] = area
    return area_map


def _overlay_panels(panels: list[np.ndarray]) -> np.ndarray:
    height, width = map(int, panels[0].shape)
    out = np.full((height, width), BG, dtype=int)
    area_maps = [_component_area_map(panel) for panel in panels]

    for r in range(height):
        for c in range(width):
            candidates = [
                (int(panel[r, c]), int(area_map[r, c]))
                for panel, area_map in zip(panels, area_maps)
                if int(panel[r, c]) != BG
            ]
            if not candidates:
                continue
            if len(candidates) == 1:
                out[r, c] = candidates[0][0]
                continue

            best_area = max(area for _color, area in candidates)
            winners = [color for color, area in candidates if area == best_area]
            assert len(winners) == 1, "conflict areas must have a unique maximum"
            out[r, c] = winners[0]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay three same-sized panels and let the largest connected mark win at overlapping positions.

    Input structure:
    - Orange separator cells partition the grid into exactly three rectangular
      non-orange panels of identical shape.
    - Inside each panel, black is blank space and colored marks form one or more
      4-connected components.
    - Multiple panels may place non-black marks at the same relative position.

    Rule:
    - Remove the separator and align the three panels by their shared bounding-box
      shape.
    - For each relative cell, copy the lone non-black mark directly if exactly one
      panel occupies that position, otherwise leave black if all are blank.
    - When two or more panels overlap on that cell, compare the areas of the
      corresponding 4-connected components inside their own panels and keep the
      color from the largest component.

    Key insight:
    - Overlay alone is insufficient: each contested pixel inherits the priority of
      its whole connected component, so local conflicts are resolved by panel-level
      topology.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    panels = _extract_panels(g)
    return _overlay_panels(panels)


def _paint_component(panel: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        panel[int(r), int(c)] = int(color)


def _reserve_neighbors(blocked: np.ndarray, cells: set[tuple[int, int]]) -> None:
    for r, c in cells:
        blocked[int(r), int(c)] = True
        for rr, cc in neighbors4(int(r), int(c)):
            if _in_bounds(blocked.shape, rr, cc):
                blocked[int(rr), int(cc)] = True


def _grow_component(
    shape: tuple[int, int],
    size: int,
    seed: tuple[int, int],
    blocked: np.ndarray,
) -> set[tuple[int, int]] | None:
    if blocked[seed]:
        return None

    component = {seed}
    frontier = {
        (int(rr), int(cc))
        for rr, cc in neighbors4(*seed)
        if _in_bounds(shape, rr, cc) and not blocked[int(rr), int(cc)]
    }
    while len(component) < int(size):
        if not frontier:
            return None

        candidates = list(frontier)
        weights = []
        for cell in candidates:
            free_neighbors = sum(
                1
                for rr, cc in neighbors4(*cell)
                if _in_bounds(shape, rr, cc) and not blocked[int(rr), int(cc)] and (int(rr), int(cc)) not in component
            )
            weights.append(free_neighbors + 1)
        chosen = random.choices(candidates, weights=weights, k=1)[0]
        frontier.remove(chosen)
        component.add(chosen)

        for rr, cc in neighbors4(*chosen):
            rr = int(rr)
            cc = int(cc)
            if _in_bounds(shape, rr, cc) and not blocked[rr, cc] and (rr, cc) not in component:
                frontier.add((rr, cc))

    return component


def _try_place_component(
    shape: tuple[int, int],
    blocked: np.ndarray,
    size: int,
    required: tuple[int, int] | None,
) -> set[tuple[int, int]] | None:
    if required is not None:
        if blocked[required]:
            return None
        for _ in range(60):
            component = _grow_component(shape, size, required, blocked)
            if component is not None:
                return component
        return None

    seeds = [(int(r), int(c)) for r, c in np.argwhere(~blocked)]
    random.shuffle(seeds)
    for seed in seeds[:40]:
        for _ in range(5):
            component = _grow_component(shape, size, seed, blocked)
            if component is not None:
                return component
    return None


def _build_panel(
    height: int,
    width: int,
    color: int,
    specs: list[dict[str, object]],
) -> np.ndarray | None:
    shape = (int(height), int(width))
    order = sorted(specs, key=lambda spec: (spec["required"] is None, -int(spec["size"])))

    for _ in range(120):
        panel = np.full(shape, BG, dtype=int)
        blocked = np.zeros(shape, dtype=bool)
        failed = False

        for spec in order:
            size = int(spec["size"])
            required = spec["required"]
            component = _try_place_component(shape, blocked, size, required)
            if component is None:
                failed = True
                break
            _paint_component(panel, component, color)
            _reserve_neighbors(blocked, component)

        if failed:
            continue

        components = find_components(panel, bg=BG, connectivity=4, monochromatic=True)
        areas = sorted(int(component["area"]) for component in components)
        expected = sorted(int(spec["size"]) for spec in specs)
        if areas == expected:
            return panel

    return None


def _sample_conflict_positions(height: int, width: int, count: int) -> list[tuple[int, int]] | None:
    candidates = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1)]
    for _ in range(100):
        random.shuffle(candidates)
        picked: list[tuple[int, int]] = []
        for cell in candidates:
            if all(max(abs(cell[0] - rr), abs(cell[1] - cc)) >= 2 for rr, cc in picked):
                picked.append(cell)
                if len(picked) == count:
                    return picked
    return None


def _conflict_groups(count: int) -> list[set[int]] | None:
    for _ in range(100):
        groups: list[set[int]] = []
        coverage = set()
        for _i in range(count):
            size = 3 if random.random() < 0.4 else 2
            group = set(random.sample(range(3), size))
            groups.append(group)
            coverage |= group
        if len(coverage) == 3:
            return groups
    return None


def _assign_specs(
    positions: list[tuple[int, int]],
    groups: list[set[int]],
) -> dict[int, list[dict[str, object]]] | None:
    specs_by_panel: dict[int, list[dict[str, object]]] = {0: [], 1: [], 2: []}
    pools = {idx: list(AREA_CHOICES) for idx in range(3)}
    for pool in pools.values():
        random.shuffle(pool)

    for position, group in zip(positions, groups):
        used_sizes: set[int] = set()
        panel_order = list(group)
        random.shuffle(panel_order)
        for panel_idx in panel_order:
            choices = [size for size in pools[panel_idx] if size not in used_sizes]
            if not choices:
                return None
            size = random.choice(choices)
            pools[panel_idx].remove(size)
            used_sizes.add(size)
            specs_by_panel[panel_idx].append({"size": size, "required": position})

    for panel_idx in range(3):
        min_extra = max(0, 2 - len(specs_by_panel[panel_idx]))
        max_extra = min(2, len(pools[panel_idx]))
        if min_extra > max_extra:
            return None
        extra_count = random.randint(min_extra, max_extra)
        extra_sizes = random.sample(pools[panel_idx], extra_count)
        for size in extra_sizes:
            pools[panel_idx].remove(size)
            specs_by_panel[panel_idx].append({"size": size, "required": None})

    return specs_by_panel


def _assemble_grid(panels: list[np.ndarray], orientation: str) -> np.ndarray:
    height, width = map(int, panels[0].shape)
    if orientation == "horizontal":
        grid = np.full((height + 2, 3 * width + 4), SEP, dtype=int)
        for idx, panel in enumerate(panels):
            paste(grid, panel, 1, 1 + idx * (width + 1))
        return grid

    grid = np.full((3 * height + 4, width + 2), SEP, dtype=int)
    for idx, panel in enumerate(panels):
        paste(grid, panel, 1 + idx * (height + 1), 1)
    return grid


def _validate_instance(grid: np.ndarray, panel_colors: list[int]) -> bool:
    try:
        panels = _extract_panels(grid)
        out = _overlay_panels(panels)
    except AssertionError:
        return False

    if len({int(color) for color in np.unique(out) if int(color) != BG}) != 3:
        return False

    if any(np.array_equal(out, panel) for panel in panels):
        return False

    area_maps = [_component_area_map(panel) for panel in panels]
    conflict_count = 0
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            competing_areas = [
                int(area_map[r, c])
                for panel, area_map in zip(panels, area_maps)
                if int(panel[r, c]) != BG
            ]
            if len(competing_areas) >= 2:
                conflict_count += 1
                if len(set(competing_areas)) != len(competing_areas):
                    return False
    if conflict_count < 2 or conflict_count > max(8, out.size // 5):
        return False

    if {int(color) for color in np.unique(out) if int(color) != BG} != set(map(int, panel_colors)):
        return False

    try:
        return np.array_equal(main(grid), out)
    except AssertionError:
        return False


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*PANEL_SIZE_RANGE)
    width = random.randint(*PANEL_SIZE_RANGE)
    conflict_count = random.choice((2, 2, 3))
    positions = _sample_conflict_positions(height, width, conflict_count)
    if positions is None:
        return None

    groups = _conflict_groups(conflict_count)
    if groups is None:
        return None

    specs_by_panel = _assign_specs(positions, groups)
    if specs_by_panel is None:
        return None

    panel_colors = random.sample(list(MARK_COLORS), 3)
    panels: list[np.ndarray] = []
    for idx in range(3):
        panel = _build_panel(height, width, panel_colors[idx], specs_by_panel[idx])
        if panel is None:
            return None
        panels.append(panel)

    orientation = random.choice(("horizontal", "vertical"))
    grid = _assemble_grid(panels, orientation)
    if max(grid.shape) > 30:
        return None

    return grid if _validate_instance(grid, panel_colors) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid panel-overlay arbitration puzzle")
