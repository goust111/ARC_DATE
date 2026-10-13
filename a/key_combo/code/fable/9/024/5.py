from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
COLOR_POOL = list(range(1, 10))
DIRS = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == int(color)
    seen = np.zeros_like(mask, dtype=bool)
    h, w = grid.shape
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
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _border_coords(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    out = set()
    for c in range(c0, c1 + 1):
        out.add((r0, c))
        out.add((r1, c))
    for r in range(r0, r1 + 1):
        out.add((r, c0))
        out.add((r, c1))
    return out


def _sorted_gap_cells(side: str, cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    if side in {"top", "bottom"}:
        return sorted(cells, key=lambda rc: rc[1])
    return sorted(cells, key=lambda rc: rc[0])


def _frame_candidate(grid: np.ndarray, color: int, coords: np.ndarray) -> dict | None:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0 = int(r0)
    c0 = int(c0)
    r1 = int(r1)
    c1 = int(c1)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    coord_set = {tuple(map(int, rc)) for rc in coords}
    border = _border_coords((r0, c0, r1, c1))
    if not coord_set.issubset(border):
        return None

    missing = border - coord_set
    if len(missing) not in {1, 2}:
        return None
    if len(coord_set) != len(border) - len(missing):
        return None

    side_matches = []
    if all(r == r0 for r, _ in missing):
        side_matches.append("top")
    if all(r == r1 for r, _ in missing):
        side_matches.append("bottom")
    if all(c == c0 for _, c in missing):
        side_matches.append("left")
    if all(c == c1 for _, c in missing):
        side_matches.append("right")
    if len(side_matches) != 1:
        return None
    side = side_matches[0]

    gap_cells = _sorted_gap_cells(side, missing)
    if side in {"top", "bottom"}:
        cols = [c for _, c in gap_cells]
        if cols[0] <= c0 or cols[-1] >= c1:
            return None
        if cols[-1] - cols[0] + 1 != len(cols):
            return None
    else:
        rows = [r for r, _ in gap_cells]
        if rows[0] <= r0 or rows[-1] >= r1:
            return None
        if rows[-1] - rows[0] + 1 != len(rows):
            return None

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    liquid_cells = np.argwhere(interior != BG)
    if liquid_cells.size == 0:
        return None
    liquid_colors = {int(v) for v in interior.flat if int(v) != BG}
    if len(liquid_colors) != 1:
        return None
    liquid_color = next(iter(liquid_colors))
    liquid_count = int(liquid_cells.shape[0])
    gap_width = len(gap_cells)
    if liquid_count % gap_width != 0:
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "border_color": int(color),
        "gap_side": side,
        "gap_cells": gap_cells,
        "gap_width": gap_width,
        "liquid_color": liquid_color,
        "liquid_count": liquid_count,
        "band_length": liquid_count // gap_width,
    }


def _find_containers(grid: np.ndarray) -> list[dict]:
    containers = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != BG):
        for coords in _components_of_color(grid, color):
            candidate = _frame_candidate(grid, color, coords)
            if candidate is not None:
                containers.append(candidate)
    containers.sort(key=lambda item: item["bbox"])
    return containers


def _band_cells(container: dict) -> list[tuple[int, int]]:
    dr, dc = DIRS[container["gap_side"]]
    cells = []
    for step in range(1, int(container["band_length"]) + 1):
        for gr, gc in container["gap_cells"]:
            cells.append((gr + dr * step, gc + dc * step))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """Pour each container's liquid out through its border gap into an equal-volume straight band.

    Input structure:
    - The background is black.
    - The foreground contains one or more monochrome hollow rectangular containers.
    - Each container border has exactly one 1-cell or 2-cell gap on a single side, and
      its interior contains pixels of exactly one liquid color.

    Rule:
    - For each container, count its liquid pixels and keep the rectangular border and
      border gap unchanged.
    - Clear the whole interior to background, then draw a straight band of the liquid
      color starting one cell outside the gap and pointing away from the container.
    - The band thickness equals the gap width, and its length equals liquid_count / gap_width.

    Key insight:
    - The gap identifies both the pouring direction and the band thickness, so liquid
      volume is conserved only by combining gap width with the interior pixel count.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    containers = _find_containers(g)
    assert containers, "expected at least one gapped hollow rectangle container"

    out = g.copy()
    band_occupied = np.zeros(g.shape, dtype=bool)
    used_liquid_colors = Counter(container["liquid_color"] for container in containers)
    assert max(used_liquid_colors.values()) == 1, "each container must have a distinct liquid color"

    for container in containers:
        r0, c0, r1, c1 = container["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = BG

    for container in containers:
        for rr, cc in _band_cells(container):
            assert 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1], "band must stay within the grid"
            assert int(g[rr, cc]) == BG, "band path must start on background cells"
            assert not band_occupied[rr, cc], "bands must not overlap"
            band_occupied[rr, cc] = True
            out[rr, cc] = int(container["liquid_color"])

    return out


def _grow_connected_subset(cells: list[tuple[int, int]], count: int) -> list[tuple[int, int]]:
    cell_set = set(cells)
    start = random.choice(cells)
    chosen = {start}
    frontier = {start}
    while len(chosen) < count:
        candidates = []
        for r, c in frontier:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (nr, nc) in cell_set and (nr, nc) not in chosen:
                    candidates.append((nr, nc))
        if not candidates:
            remaining = [cell for cell in cells if cell not in chosen]
            if not remaining:
                break
            next_cell = random.choice(remaining)
        else:
            next_cell = random.choice(candidates)
        chosen.add(next_cell)
        frontier.add(next_cell)
    return list(chosen)


def _choose_liquid_cells(interior_cells: list[tuple[int, int]], count: int) -> list[tuple[int, int]]:
    assert count <= len(interior_cells)
    if count == len(interior_cells):
        return list(interior_cells)
    if random.random() < 0.5:
        return random.sample(interior_cells, count)
    connected = _grow_connected_subset(interior_cells, count)
    if len(connected) == count:
        return connected
    return random.sample(interior_cells, count)


def _spec_gap_cells(spec: dict) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = spec["bbox"]
    side = spec["gap_side"]
    width = spec["gap_width"]
    start = spec["gap_start"]
    if side == "top":
        return [(r0, c) for c in range(c0 + start, c0 + start + width)]
    if side == "bottom":
        return [(r1, c) for c in range(c0 + start, c0 + start + width)]
    if side == "left":
        return [(r, c0) for r in range(r0 + start, r0 + start + width)]
    return [(r, c1) for r in range(r0 + start, r0 + start + width)]


def _render_from_specs(height: int, width: int, specs: list[dict], *, output: bool) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        border_color = spec["border_color"]
        grid[r0 : r1 + 1, c0] = border_color
        grid[r0 : r1 + 1, c1] = border_color
        grid[r0, c0 : c1 + 1] = border_color
        grid[r1, c0 : c1 + 1] = border_color
        for gr, gc in _spec_gap_cells(spec):
            grid[gr, gc] = BG

        if output:
            for rr, cc in spec["band_cells"]:
                grid[rr, cc] = spec["liquid_color"]
        else:
            for rr, cc in spec["liquid_cells"]:
                grid[rr, cc] = spec["liquid_color"]
    return grid


def _padded_bbox_cells(bbox: tuple[int, int, int, int], height: int, width: int, pad: int = 1) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    out = []
    for r in range(max(0, r0 - pad), min(height, r1 + pad + 1)):
        for c in range(max(0, c0 - pad), min(width, c1 + pad + 1)):
            out.append((r, c))
    return out


def _union_bbox(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def _place_specs(height: int, width: int, blueprints: list[dict]) -> list[dict] | None:
    occupied = np.zeros((height, width), dtype=bool)
    specs = []
    ordered = sorted(blueprints, key=lambda item: item["outer_h"] * item["outer_w"] + item["gap_width"] * item["band_length"], reverse=True)
    for blueprint in ordered:
        outer_h = blueprint["outer_h"]
        outer_w = blueprint["outer_w"]
        side = blueprint["gap_side"]
        band_length = blueprint["band_length"]
        if side == "top":
            r_candidates = range(band_length, height - outer_h + 1)
            c_candidates = range(0, width - outer_w + 1)
        elif side == "bottom":
            r_candidates = range(0, height - outer_h - band_length + 1)
            c_candidates = range(0, width - outer_w + 1)
        elif side == "left":
            r_candidates = range(0, height - outer_h + 1)
            c_candidates = range(band_length, width - outer_w + 1)
        else:
            r_candidates = range(0, height - outer_h + 1)
            c_candidates = range(0, width - outer_w - band_length + 1)
        positions = [(r0, c0) for r0 in r_candidates for c0 in c_candidates]
        random.shuffle(positions)
        placed = None
        for r0, c0 in positions:
            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1
            if side in {"top", "bottom"}:
                max_start = outer_w - 1 - blueprint["gap_width"]
            else:
                max_start = outer_h - 1 - blueprint["gap_width"]
            gap_start = random.randint(1, max_start)
            spec = dict(blueprint)
            spec["bbox"] = (r0, c0, r1, c1)
            spec["gap_start"] = gap_start
            spec["gap_cells"] = _spec_gap_cells(spec)
            spec["band_cells"] = _band_cells(
                {
                    "gap_side": side,
                    "gap_cells": spec["gap_cells"],
                    "band_length": blueprint["band_length"],
                }
            )
            band_bbox = (
                min(r for r, _ in spec["band_cells"]),
                min(c for _, c in spec["band_cells"]),
                max(r for r, _ in spec["band_cells"]),
                max(c for _, c in spec["band_cells"]),
            )
            union = _union_bbox(spec["bbox"], band_bbox)
            pad_cells = _padded_bbox_cells(union, height, width, pad=1)
            if any(occupied[r, c] for r, c in pad_cells):
                continue

            interior_cells = [
                (r, c)
                for r in range(r0 + 1, r1)
                for c in range(c0 + 1, c1)
            ]
            if spec["liquid_count"] >= len(interior_cells):
                continue
            spec["liquid_cells"] = _choose_liquid_cells(interior_cells, spec["liquid_count"])

            for rr, cc in pad_cells:
                occupied[rr, cc] = True
            placed = spec
            break
        if placed is None:
            return None
        specs.append(placed)
    specs.sort(key=lambda item: item["bbox"])
    return specs


def _make_blueprints() -> list[dict]:
    container_count = random.randint(2, 4)
    directions = random.sample(list(DIRS), container_count)
    widths = [1, 2] + [random.choice((1, 2)) for _ in range(container_count - 2)]
    random.shuffle(widths)
    if 1 not in widths or 2 not in widths:
        widths[0] = 1
        widths[1] = 2

    palette = random.sample(COLOR_POOL, container_count * 2)
    border_colors = palette[:container_count]
    liquid_colors = palette[container_count:]

    same_volume = random.choice((4, 6, 8))
    blueprints = []
    pair_slots = []
    for idx in range(container_count):
        gap_width = widths[idx]
        if len(pair_slots) < 2:
            liquid_count = same_volume
            pair_slots.append(idx)
        else:
            band_length = random.randint(2, 6)
            liquid_count = gap_width * band_length

        min_interior_area = liquid_count + 1
        interior_h_choices = [h for h in range(3, 7) if any(h * w >= min_interior_area for w in range(3, 7))]
        interior_h = random.choice(interior_h_choices)
        interior_w_choices = [w for w in range(3, 7) if interior_h * w >= min_interior_area]
        interior_w = random.choice(interior_w_choices)

        blueprint = {
            "outer_h": interior_h + 2,
            "outer_w": interior_w + 2,
            "gap_side": directions[idx],
            "gap_width": gap_width,
            "liquid_count": liquid_count,
            "band_length": liquid_count // gap_width,
            "border_color": border_colors[idx],
            "liquid_color": liquid_colors[idx],
        }
        blueprints.append(blueprint)
    return blueprints


def _validate_instance(inp: np.ndarray, specs: list[dict]) -> bool:
    widths = {spec["gap_width"] for spec in specs}
    if widths != {1, 2}:
        return False
    if len({spec["gap_side"] for spec in specs}) < 2:
        return False

    pair_match = False
    for i, spec_a in enumerate(specs):
        for spec_b in specs[i + 1 :]:
            if spec_a["gap_width"] != spec_b["gap_width"] and spec_a["liquid_count"] == spec_b["liquid_count"]:
                pair_match = True
    if not pair_match:
        return False

    detected = _find_containers(inp)
    if len(detected) != len(specs):
        return False

    by_bbox = {spec["bbox"]: spec for spec in specs}
    for container in detected:
        target = by_bbox.get(container["bbox"])
        if target is None:
            return False
        if container["gap_side"] != target["gap_side"]:
            return False
        if container["gap_width"] != target["gap_width"]:
            return False
        if container["liquid_color"] != target["liquid_color"]:
            return False
        if container["liquid_count"] != target["liquid_count"]:
            return False

    expected = _render_from_specs(inp.shape[0], inp.shape[1], specs, output=True)
    if not np.array_equal(main(inp), expected):
        return False

    areas = [(spec["outer_h"] - 2) * (spec["outer_w"] - 2) for spec in specs]
    if not any(spec["liquid_count"] < area - 1 for spec, area in zip(specs, areas)):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(20, 29)
    width = random.randint(20, 29)
    blueprints = _make_blueprints()
    specs = _place_specs(height, width, blueprints)
    if specs is None:
        return None
    inp = _render_from_specs(height, width, specs, output=False)
    if not _validate_instance(inp, specs):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid container-pour input")
