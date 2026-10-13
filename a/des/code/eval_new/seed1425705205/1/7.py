import random
from collections import deque
from dataclasses import dataclass

import numpy as np


BG = 0
WALL = 5
SOURCE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRECTIONS = ("up", "right", "down", "left")
SOURCE_SHAPES = tuple((h, w, h * w) for h in range(1, 4) for w in range(1, 4))
GEN_RETRIES = 500


@dataclass
class Component:
    color: int
    coords: np.ndarray
    bbox: tuple[int, int, int, int]


@dataclass
class Container:
    color: int
    coords: np.ndarray
    bbox: tuple[int, int, int, int]
    opening: str


@dataclass
class Instance:
    bbox: tuple[int, int, int, int]
    opening: str
    wall_color: int
    source_color: int
    source_coords: np.ndarray
    source_area: int


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(grid: np.ndarray) -> list[Component]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[Component] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(Component(color=color, coords=arr, bbox=(int(r0), int(c0), int(r1), int(c1))))
    return comps


def _parse_container(grid: np.ndarray, comp: Component) -> Container | None:
    r0, c0, r1, c1 = comp.bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    mask = np.zeros((h, w), dtype=bool)
    local = comp.coords - np.array([[r0, c0]])
    mask[local[:, 0], local[:, 1]] = True

    rows, cols = np.nonzero(mask)
    inside = (rows > 0) & (rows < h - 1) & (cols > 0) & (cols < w - 1)
    if bool(np.any(inside)):
        return None
    if not bool(mask[0, 0] and mask[0, -1] and mask[-1, 0] and mask[-1, -1]):
        return None

    top = int(mask[0].sum())
    bottom = int(mask[-1].sum())
    left = int(mask[:, 0].sum())
    right = int(mask[:, -1].sum())

    opening = None
    if top == 2 and bottom == w and left == h and right == h:
        opening = "up"
    elif bottom == 2 and top == w and left == h and right == h:
        opening = "down"
    elif left == 2 and top == w and bottom == w and right == h:
        opening = "left"
    elif right == 2 and top == w and bottom == w and left == h:
        opening = "right"
    if opening is None:
        return None

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    if interior.size == 0 or bool(np.any(interior != BG)):
        return None

    return Container(color=comp.color, coords=comp.coords, bbox=comp.bbox, opening=opening)


def _is_source_block(grid: np.ndarray, comp: Component) -> bool:
    r0, c0, r1, c1 = comp.bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h > 3 or w > 3:
        return False
    if int(comp.coords.shape[0]) != h * w:
        return False
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    return bool((patch == comp.color).all())


def _matches_source(container: Container, source: Component) -> bool:
    r0, c0, r1, c1 = container.bbox
    sr0, sc0, sr1, sc1 = source.bbox
    if container.opening == "up":
        return sr1 == r0 - 1 and sc0 >= c0 + 1 and sc1 <= c1 - 1
    if container.opening == "down":
        return sr0 == r1 + 1 and sc0 >= c0 + 1 and sc1 <= c1 - 1
    if container.opening == "left":
        return sc1 == c0 - 1 and sr0 >= r0 + 1 and sr1 <= r1 - 1
    return sc0 == c1 + 1 and sr0 >= r0 + 1 and sr1 <= r1 - 1


def _extract_instances(grid: np.ndarray) -> list[Instance]:
    containers: list[Container] = []
    sources: list[Component] = []
    for comp in _find_components(grid):
        container = _parse_container(grid, comp)
        if container is not None:
            containers.append(container)
        elif _is_source_block(grid, comp):
            sources.append(comp)
        else:
            return []

    if not containers or len(containers) != len(sources):
        return []

    instances: list[Instance] = []
    used_sources: set[int] = set()
    for container in containers:
        matches = [idx for idx, source in enumerate(sources) if idx not in used_sources and _matches_source(container, source)]
        if len(matches) != 1:
            return []
        idx = matches[0]
        used_sources.add(idx)
        source = sources[idx]
        instances.append(
            Instance(
                bbox=container.bbox,
                opening=container.opening,
                wall_color=container.color,
                source_color=source.color,
                source_coords=source.coords,
                source_area=int(source.coords.shape[0]),
            )
        )

    if len(used_sources) != len(sources):
        return []

    instances.sort(key=lambda inst: (inst.bbox[0], inst.bbox[1]))
    return instances


def _fill_order(instance: Instance) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = instance.bbox
    cells: list[tuple[int, int]] = []
    if instance.opening == "up":
        for r in range(r1 - 1, r0, -1):
            for c in range(c0 + 1, c1):
                cells.append((r, c))
    elif instance.opening == "down":
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                cells.append((r, c))
    elif instance.opening == "left":
        for c in range(c1 - 1, c0, -1):
            for r in range(r0 + 1, r1):
                cells.append((r, c))
    else:
        for c in range(c0 + 1, c1):
            for r in range(r0 + 1, r1):
                cells.append((r, c))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each U-shaped container with a source-colored deposit whose size equals the external source block area.

    Input structure:
    - Background is black.
    - Several grey U-shaped containers appear as 1-pixel-thick three-sided frames with one open side.
    - Each container has one adjacent monochrome source block outside its opening; source blocks may be 1x1 to 3x3.

    Rule:
    - Detect every U-shaped container and determine which side is open.
    - Match the unique source block touching the outside of that opening and count its area.
    - Remove the source block, then fill the container interior with the source color using exactly that many cells.
    - Deposition starts from the side opposite the opening: bottom-to-top for upward openings, top-to-bottom for downward openings, left-to-right for rightward openings, and right-to-left for leftward openings; within each row or column, fill across the whole span before moving on. If the source area exceeds the interior capacity, fill the entire interior.

    Key insight:
    - The source block area determines how much liquid is available, while the missing side of the grey U determines the deposition direction inside the container.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    instances = _extract_instances(g)
    assert instances, "expected at least one valid container-source pair"

    for inst in instances:
        out[inst.source_coords[:, 0], inst.source_coords[:, 1]] = BG
        order = _fill_order(inst)
        for r, c in order[: min(inst.source_area, len(order))]:
            out[r, c] = inst.source_color

    return out


def _build_rel_cells(direction: str, inner_h: int, inner_w: int, source_h: int, source_w: int, offset: int, color: int) -> list[tuple[int, int, int]]:
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    wall_cells: set[tuple[int, int]] = set()
    for r in range(outer_h):
        wall_cells.add((r, 0))
        wall_cells.add((r, outer_w - 1))
    for c in range(outer_w):
        wall_cells.add((0, c))
        wall_cells.add((outer_h - 1, c))
    if direction == "up":
        for c in range(1, outer_w - 1):
            wall_cells.discard((0, c))
        source_r0, source_c0 = -source_h, 1 + offset
    elif direction == "down":
        for c in range(1, outer_w - 1):
            wall_cells.discard((outer_h - 1, c))
        source_r0, source_c0 = outer_h, 1 + offset
    elif direction == "left":
        for r in range(1, outer_h - 1):
            wall_cells.discard((r, 0))
        source_r0, source_c0 = 1 + offset, -source_w
    else:
        for r in range(1, outer_h - 1):
            wall_cells.discard((r, outer_w - 1))
        source_r0, source_c0 = 1 + offset, outer_w

    rel_cells = [(r, c, WALL) for r, c in sorted(wall_cells)]
    for r in range(source_r0, source_r0 + source_h):
        for c in range(source_c0, source_c0 + source_w):
            rel_cells.append((r, c, color))
    return rel_cells


def _make_spec(rng: random.Random, direction: str, color: int, overflow: bool, used_areas: set[int]) -> dict[str, int | str | list[tuple[int, int, int]]]:
    for _ in range(200):
        inner_h = rng.randint(1, 6)
        inner_w = rng.randint(1, 6)
        capacity = inner_h * inner_w
        options: list[tuple[int, int, int]] = []
        for source_h, source_w, area in SOURCE_SHAPES:
            if area in used_areas:
                continue
            if direction in ("up", "down") and source_w > inner_w:
                continue
            if direction in ("left", "right") and source_h > inner_h:
                continue
            if overflow and area <= capacity:
                continue
            if not overflow and area > capacity:
                continue
            options.append((source_h, source_w, area))
        if not options:
            continue

        source_h, source_w, area = rng.choice(options)
        offset = rng.randint(0, inner_w - source_w) if direction in ("up", "down") else rng.randint(0, inner_h - source_h)
        cells = _build_rel_cells(direction, inner_h, inner_w, source_h, source_w, offset, color)
        rows = [r for r, _, _ in cells]
        cols = [c for _, c, _ in cells]
        return {
            "direction": direction,
            "color": color,
            "area": area,
            "inner_h": inner_h,
            "inner_w": inner_w,
            "cells": cells,
            "min_r": min(rows),
            "max_r": max(rows),
            "min_c": min(cols),
            "max_c": max(cols),
            "footprint_area": (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1),
        }
    return {}


def _place_specs(rng: random.Random, specs: list[dict[str, int | str | list[tuple[int, int, int]]]], height: int, width: int) -> np.ndarray | None:
    grid = np.zeros((height, width), dtype=int)
    blocked = np.zeros((height, width), dtype=bool)
    for spec in sorted(specs, key=lambda item: int(item["footprint_area"]), reverse=True):
        min_r = int(spec["min_r"])
        max_r = int(spec["max_r"])
        min_c = int(spec["min_c"])
        max_c = int(spec["max_c"])
        row_lo = -min_r
        row_hi = height - 1 - max_r
        col_lo = -min_c
        col_hi = width - 1 - max_c
        if row_lo > row_hi or col_lo > col_hi:
            return None

        placed = False
        for _ in range(400):
            anchor_r = rng.randint(row_lo, row_hi)
            anchor_c = rng.randint(col_lo, col_hi)
            abs_r0 = anchor_r + min_r
            abs_r1 = anchor_r + max_r
            abs_c0 = anchor_c + min_c
            abs_c1 = anchor_c + max_c
            pad_r0 = max(0, abs_r0 - 1)
            pad_r1 = min(height - 1, abs_r1 + 1)
            pad_c0 = max(0, abs_c0 - 1)
            pad_c1 = min(width - 1, abs_c1 + 1)
            if bool(blocked[pad_r0 : pad_r1 + 1, pad_c0 : pad_c1 + 1].any()):
                continue

            for dr, dc, color in spec["cells"]:
                grid[anchor_r + dr, anchor_c + dc] = color
            blocked[pad_r0 : pad_r1 + 1, pad_c0 : pad_c1 + 1] = True
            placed = True
            break
        if not placed:
            return None
    return grid


def _is_valid_generated(grid: np.ndarray, specs: list[dict[str, int | str | list[tuple[int, int, int]]]]) -> bool:
    instances = _extract_instances(grid)
    if len(instances) != len(specs):
        return False

    by_color = {inst.source_color: inst for inst in instances}
    for spec in specs:
        color = int(spec["color"])
        if color not in by_color:
            return False
        inst = by_color[color]
        if inst.opening != spec["direction"]:
            return False
        if inst.source_area != int(spec["area"]):
            return False
        r0, c0, r1, c1 = inst.bbox
        capacity = (r1 - r0 - 1) * (c1 - c0 - 1)
        if capacity != int(spec["inner_h"]) * int(spec["inner_w"]):
            return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    for inst in instances:
        if bool(np.any(out[inst.source_coords[:, 0], inst.source_coords[:, 1]] != BG)):
            return False
        r0, c0, r1, c1 = inst.bbox
        capacity = (r1 - r0 - 1) * (c1 - c0 - 1)
        if int(np.count_nonzero(out == inst.source_color)) != min(inst.source_area, capacity):
            return False
    return True


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    count = rng.choice((2, 3, 4))
    directions = rng.sample(DIRECTIONS, count)
    colors = rng.sample(SOURCE_COLORS, count)

    overflow_flags = [True] * ((count + 1) // 2) + [False] * (count // 2)
    rng.shuffle(overflow_flags)

    used_areas: set[int] = set()
    specs: list[dict[str, int | str | list[tuple[int, int, int]]]] = []
    for direction, color, overflow in zip(directions, colors, overflow_flags):
        spec = _make_spec(rng, direction, color, overflow, used_areas)
        if not spec:
            return None
        specs.append(spec)
        used_areas.add(int(spec["area"]))

    max_h = max(int(spec["max_r"]) - int(spec["min_r"]) + 1 for spec in specs)
    max_w = max(int(spec["max_c"]) - int(spec["min_c"]) + 1 for spec in specs)
    min_dim = {2: 16, 3: 20, 4: 24}[count]
    height = rng.randint(max(min_dim, max_h + 2), 30)
    width = rng.randint(max(min_dim, max_w + 2), 30)

    grid = _place_specs(rng, specs, height, width)
    if grid is None:
        return None
    if not _is_valid_generated(grid, specs):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
