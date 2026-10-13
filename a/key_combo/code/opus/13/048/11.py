from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
FRAME = 1
ARROW = 9
REGION_COLORS = (2, 3, 4, 5, 6, 7, 8)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]
MAX_GEN_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _neighbors8(r: int, c: int):
    for dr, dc in DIRS8:
        yield r + dr, c + dc


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _find_regions(grid: np.ndarray):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    regions = []
    region_at: dict[tuple[int, int], int] = {}
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in (BG, ARROW) or seen[r, c]:
                continue
            coords = []
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                region_at[(rr, cc)] = len(regions)
                for nr, nc in _neighbors4(rr, cc):
                    if (
                        _in_bounds(grid.shape, nr, nc)
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            regions.append({"color": color, "coords": coords, "coord_set": set(coords)})
    return regions, region_at


def _arrow_wings(shaft: tuple[int, int], direction: tuple[int, int]):
    r, c = shaft
    dr, dc = direction
    pr, pc = -dc, dr
    base_r, base_c = r - dr, c - dc
    return {
        (base_r + pr, base_c + pc),
        (base_r - pr, base_c - pc),
    }


def _first_color_on_ray(grid: np.ndarray, start: tuple[int, int], step: tuple[int, int]) -> int | None:
    r, c = start
    dr, dc = step
    while _in_bounds(grid.shape, r, c):
        color = int(grid[r, c])
        if color not in (BG, ARROW):
            return color
        r += dr
        c += dc
    return None


def _find_arrow_components(grid: np.ndarray):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != ARROW:
                continue
            coords = []
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors8(rr, cc):
                    if (
                        _in_bounds(grid.shape, nr, nc)
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == ARROW
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _parse_arrow_component(
    comp: list[tuple[int, int]],
    grid: np.ndarray,
    region_at: dict[tuple[int, int], int],
    regions: list[dict],
):
    if len(comp) != 3:
        raise AssertionError("each arrow must be a 3-pixel maroon component")

    comp_set = set(comp)
    degrees = {}
    for cell in comp:
        degrees[cell] = sum(
            (nbr in comp_set)
            for nbr in _neighbors8(*cell)
        )

    shafts = [cell for cell, deg in degrees.items() if deg == 2]
    wings = [cell for cell, deg in degrees.items() if deg == 1]
    if len(shafts) != 1 or len(wings) != 2:
        raise AssertionError("arrow must be a V with one shaft and two wings")

    shaft = shafts[0]
    avg_r = wings[0][0] + wings[1][0]
    avg_c = wings[0][1] + wings[1][1]
    if avg_r % 2 or avg_c % 2:
        raise AssertionError("wings must be symmetric")

    mid = (avg_r // 2, avg_c // 2)
    direction = (shaft[0] - mid[0], shaft[1] - mid[1])
    if direction not in DIRS4:
        raise AssertionError("arrow direction must be cardinal")
    if _arrow_wings(shaft, direction) != set(wings):
        raise AssertionError("maroon component does not match the arrow geometry")

    target_cell = (shaft[0] + direction[0], shaft[1] + direction[1])
    if target_cell not in region_at:
        raise AssertionError("shaft must touch its target region")
    target_region = region_at[target_cell]

    source_color = _first_color_on_ray(
        grid,
        (shaft[0] - direction[0], shaft[1] - direction[1]),
        (-direction[0], -direction[1]),
    )
    if source_color is None:
        raise AssertionError("no source region found behind the arrow")
    if source_color == regions[target_region]["color"]:
        raise AssertionError("source and target colors must differ")

    return {
        "shaft": shaft,
        "direction": direction,
        "target_region": target_region,
        "target_color": regions[target_region]["color"],
        "source_color": source_color,
    }


def _parse_scene(grid: np.ndarray):
    regions, region_at = _find_regions(grid)
    arrow_components = _find_arrow_components(grid)
    if not (2 <= len(regions) <= 3):
        raise AssertionError("expected 2 or 3 colored regions")
    if not (1 <= len(arrow_components) <= 2):
        raise AssertionError("expected 1 or 2 arrows")

    arrows = [
        _parse_arrow_component(comp, grid, region_at, regions)
        for comp in arrow_components
    ]
    target_regions = [arrow["target_region"] for arrow in arrows]
    if len(set(target_regions)) != len(target_regions):
        raise AssertionError("a region cannot be pointed to by multiple arrows")

    return {
        "regions": regions,
        "region_at": region_at,
        "arrows": sorted(arrows, key=lambda x: (x["shaft"], x["direction"])),
    }


def _stack_with_frame(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    h, w = top.shape
    out = np.full((2 * h + 3, w + 2), FRAME, dtype=int)
    out[1 : h + 1, 1 : w + 1] = top
    out[h + 2 : 2 * h + 2, 1 : w + 1] = bottom
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the arrow-pointed regions, erase the arrows, then stack the clean original and transformed copies inside a blue frame.

    Input structure:
    - The grid has a black background with 2 or 3 separated 4-connected monochrome regions in non-black, non-maroon colors.
    - There are 1 or 2 maroon 3-pixel arrow markers; each marker is a V-shaped pair of wings behind one shaft pixel.
    - The shaft touches its target region in the pointing direction, while a source-colored region lies on the opposite side.

    Rule:
    - Decode each maroon V marker to get its cardinal pointing direction and the target region touched by the shaft.
    - For each arrow, scan straight behind the shaft; the first non-black, non-maroon region on that opposite ray provides the recolor source color.
    - Make two copies of the grid, erase all maroon arrow pixels in both, recolor every target region in one copy with its source color, then place the clean unerased-state copy above the recolored copy.
    - Insert a blue separator row between the copies and wrap the whole stacked result in a one-pixel blue frame.

    Key insight:
    - The maroon arrows are instructions rather than content: their V geometry selects a destination, the opposite ray selects a source color, and both arrows disappear before the two-state stack is shown.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    scene = _parse_scene(g)
    clean = g.copy()
    clean[clean == ARROW] = BG

    recolored = clean.copy()
    for arrow in scene["arrows"]:
        for r, c in scene["regions"][arrow["target_region"]]["coords"]:
            recolored[r, c] = arrow["source_color"]

    return _stack_with_frame(clean, recolored)


def _required_rel_cell(direction: tuple[int, int], role: str) -> tuple[int, int]:
    if direction == (0, 1):
        return (1, 0) if role == "target" else (1, 2)
    if direction == (0, -1):
        return (1, 2) if role == "target" else (1, 0)
    if direction == (1, 0):
        return (0, 1) if role == "target" else (2, 1)
    if direction == (-1, 0):
        return (2, 1) if role == "target" else (0, 1)
    raise ValueError("unsupported direction")


def _manhattan_path(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = a
    rr, cc = b
    cells = [(r, c)]
    step_r = 1 if rr > r else -1
    while r != rr:
        r += step_r
        cells.append((r, c))
    step_c = 1 if cc > c else -1
    while c != cc:
        c += step_c
        cells.append((r, c))
    return cells


def _grow_region(required: set[tuple[int, int]], min_area: int, max_area: int) -> set[tuple[int, int]] | None:
    cells = set()
    req_list = list(required) if required else [(1, 1)]
    cells.add(req_list[0])
    for cell in req_list[1:]:
        cells.update(_manhattan_path(req_list[0], cell))
    if len(cells) > max_area:
        return None

    target_area = random.randint(max(min_area, len(cells)), max_area)
    while len(cells) < target_area:
        frontier = []
        for r, c in sorted(cells):
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < 3 and 0 <= nc < 3 and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(frontier))
    return cells


def _build_layout():
    kind = random.choices(
        ["h2", "v2", "h3_chain", "v3_chain", "h3_single", "v3_single"],
        weights=[18, 18, 22, 22, 10, 10],
        k=1,
    )[0]

    if kind.startswith("h"):
        region_count = 2 if kind in {"h2"} else 3
        footprint_w = 8 if region_count == 2 else 13
        h = random.randint(7, 11)
        w = footprint_w + random.randint(0, 2)
        top = random.randint(0, h - 3)
        left = random.randint(0, w - footprint_w)
        starts = [left + 5 * i for i in range(region_count)]
        boxes = [(top, c0) for c0 in starts]
        direction = random.choice([(0, 1), (0, -1)])

        if region_count == 2:
            if direction == (0, 1):
                arrows = [{"shaft": (top + 1, left + 4), "direction": direction, "source": 0, "target": 1}]
            else:
                arrows = [{"shaft": (top + 1, left + 3), "direction": direction, "source": 1, "target": 0}]
            chain = False
        elif kind == "h3_chain":
            if direction == (0, 1):
                arrows = [
                    {"shaft": (top + 1, left + 4), "direction": direction, "source": 0, "target": 1},
                    {"shaft": (top + 1, left + 9), "direction": direction, "source": 1, "target": 2},
                ]
            else:
                arrows = [
                    {"shaft": (top + 1, left + 3), "direction": direction, "source": 1, "target": 0},
                    {"shaft": (top + 1, left + 8), "direction": direction, "source": 2, "target": 1},
                ]
            chain = True
        else:
            if direction == (0, 1):
                if random.random() < 0.5:
                    arrows = [{"shaft": (top + 1, left + 4), "direction": direction, "source": 0, "target": 1}]
                else:
                    arrows = [{"shaft": (top + 1, left + 9), "direction": direction, "source": 1, "target": 2}]
            else:
                if random.random() < 0.5:
                    arrows = [{"shaft": (top + 1, left + 3), "direction": direction, "source": 1, "target": 0}]
                else:
                    arrows = [{"shaft": (top + 1, left + 8), "direction": direction, "source": 2, "target": 1}]
            chain = False
    else:
        region_count = 2 if kind == "v2" else 3
        footprint_h = 8 if region_count == 2 else 13
        h = footprint_h
        w = random.randint(7, 11)
        top = 0
        left = random.randint(0, w - 3)
        starts = [top + 5 * i for i in range(region_count)]
        boxes = [(r0, left) for r0 in starts]
        direction = random.choice([(1, 0), (-1, 0)])

        if region_count == 2:
            if direction == (1, 0):
                arrows = [{"shaft": (4, left + 1), "direction": direction, "source": 0, "target": 1}]
            else:
                arrows = [{"shaft": (3, left + 1), "direction": direction, "source": 1, "target": 0}]
            chain = False
        elif kind == "v3_chain":
            if direction == (1, 0):
                arrows = [
                    {"shaft": (4, left + 1), "direction": direction, "source": 0, "target": 1},
                    {"shaft": (9, left + 1), "direction": direction, "source": 1, "target": 2},
                ]
            else:
                arrows = [
                    {"shaft": (3, left + 1), "direction": direction, "source": 1, "target": 0},
                    {"shaft": (8, left + 1), "direction": direction, "source": 2, "target": 1},
                ]
            chain = True
        else:
            if direction == (1, 0):
                if random.random() < 0.5:
                    arrows = [{"shaft": (4, left + 1), "direction": direction, "source": 0, "target": 1}]
                else:
                    arrows = [{"shaft": (9, left + 1), "direction": direction, "source": 1, "target": 2}]
            else:
                if random.random() < 0.5:
                    arrows = [{"shaft": (3, left + 1), "direction": direction, "source": 1, "target": 0}]
                else:
                    arrows = [{"shaft": (8, left + 1), "direction": direction, "source": 2, "target": 1}]
            chain = False

    return {
        "shape": (h, w),
        "boxes": boxes,
        "arrows": arrows,
        "region_count": region_count,
        "chain": chain,
    }


def _paint_region(grid: np.ndarray, top: int, left: int, rel_cells: set[tuple[int, int]], color: int):
    abs_cells = []
    for rr, cc in rel_cells:
        grid[top + rr, left + cc] = color
        abs_cells.append((top + rr, left + cc))
    return abs_cells


def _manual_output(inp: np.ndarray, region_cells: list[list[tuple[int, int]]], region_colors: list[int], arrows: list[dict]):
    clean = inp.copy()
    clean[clean == ARROW] = BG
    recolored = clean.copy()
    for arrow in arrows:
        source_color = region_colors[arrow["source"]]
        for r, c in region_cells[arrow["target"]]:
            recolored[r, c] = source_color
    return _stack_with_frame(clean, recolored), clean, recolored


def _sequential_output(inp: np.ndarray, region_cells: list[list[tuple[int, int]]], arrows: list[dict]) -> np.ndarray:
    out = inp.copy()
    out[out == ARROW] = BG
    for arrow in arrows:
        source_cell = region_cells[arrow["source"]][0]
        source_color = int(out[source_cell])
        for r, c in region_cells[arrow["target"]]:
            out[r, c] = source_color
    return out


def _validate_generated(inp: np.ndarray, region_cells: list[list[tuple[int, int]]], region_colors: list[int], spec: dict) -> bool:
    try:
        scene = _parse_scene(inp)
    except AssertionError:
        return False

    expected_keys = sorted(
        (
            arrow["shaft"],
            arrow["direction"],
            region_colors[arrow["target"]],
            region_colors[arrow["source"]],
        )
        for arrow in spec["arrows"]
    )
    parsed_keys = sorted(
        (
            arrow["shaft"],
            arrow["direction"],
            arrow["target_color"],
            arrow["source_color"],
        )
        for arrow in scene["arrows"]
    )
    if parsed_keys != expected_keys:
        return False

    if any(not (4 <= len(cells) <= 8) for cells in region_cells):
        return False
    if len(set(region_colors)) != len(region_colors):
        return False

    expected, clean, recolored = _manual_output(inp, region_cells, region_colors, spec["arrows"])
    if np.array_equal(clean, recolored):
        return False
    if expected.shape[0] > 30 or expected.shape[1] > 30:
        return False

    try:
        actual = main(inp)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False

    if spec["chain"]:
        if np.array_equal(_sequential_output(inp, region_cells, spec["arrows"]), recolored):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    spec = _build_layout()
    h, w = spec["shape"]
    grid = np.zeros((h, w), dtype=int)
    region_colors = random.sample(REGION_COLORS, spec["region_count"])
    required = [set() for _ in range(spec["region_count"])]

    for arrow in spec["arrows"]:
        required[arrow["target"]].add(_required_rel_cell(arrow["direction"], "target"))
        required[arrow["source"]].add(_required_rel_cell(arrow["direction"], "source"))

    for i in range(spec["region_count"]):
        if not required[i]:
            required[i].add((1, 1))

    region_cells = []
    for i, (top, left) in enumerate(spec["boxes"]):
        min_area = 4 if len(required[i]) <= 1 else 5
        rel = _grow_region(required[i], min_area, 8)
        if rel is None:
            return None
        abs_cells = _paint_region(grid, top, left, rel, region_colors[i])
        region_cells.append(abs_cells)

    for arrow in spec["arrows"]:
        shaft = arrow["shaft"]
        wings = _arrow_wings(shaft, arrow["direction"])
        arrow_cells = {shaft, *wings}
        if any(not _in_bounds(grid.shape, r, c) for r, c in arrow_cells):
            return None
        if any(int(grid[r, c]) != BG for r, c in arrow_cells):
            return None
        target_cell = (shaft[0] + arrow["direction"][0], shaft[1] + arrow["direction"][1])
        if target_cell not in set(region_cells[arrow["target"]]):
            return None
        behind = (shaft[0] - arrow["direction"][0], shaft[1] - arrow["direction"][1])
        if _in_bounds(grid.shape, behind[0], behind[1]) and int(grid[behind]) not in (BG, region_colors[arrow["source"]]):
            return None
        for r, c in arrow_cells:
            grid[r, c] = ARROW

    if not _validate_generated(grid, region_cells, region_colors, spec):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for ARC puzzle 11")
