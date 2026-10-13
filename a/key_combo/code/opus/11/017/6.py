from __future__ import annotations

import itertools
import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
MAROON = 9

RANK_COLORS = (RED, YELLOW, PINK, MAROON)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIR8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if (dr, dc) != (0, 0)
)


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _is_corner_bracket(coords: np.ndarray) -> bool:
    pts = {tuple(map(int, rc)) for rc in coords}
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3 or len(pts) != h + w - 1:
        return False
    candidates = [
        {(r0, c) for c in range(c0, c1 + 1)} | {(r, c0) for r in range(r0, r1 + 1)},
        {(r0, c) for c in range(c0, c1 + 1)} | {(r, c1) for r in range(r0, r1 + 1)},
        {(r1, c) for c in range(c0, c1 + 1)} | {(r, c0) for r in range(r0, r1 + 1)},
        {(r1, c) for c in range(c0, c1 + 1)} | {(r, c1) for r in range(r0, r1 + 1)},
    ]
    return any(pts == candidate for candidate in candidates)


def _find_bracket_and_marker(grid: np.ndarray) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    grey_comps = _components_from_mask(grid == GREY)
    candidates: list[tuple[set[tuple[int, int]], tuple[int, int]]] = []
    for comp in grey_comps:
        if not _is_corner_bracket(comp):
            continue
        bracket = {tuple(map(int, rc)) for rc in comp}
        r0, c0, r1, c1 = _bbox_from_coords(comp)
        markers = [
            (r, c)
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if (r, c) not in bracket and int(grid[r, c]) != BG
        ]
        if len(markers) == 1:
            candidates.append((bracket, markers[0]))
    assert len(candidates) == 1, "expected exactly one grey corner bracket with one marker"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase the bracket-selected color, then recolor the resulting disconnected foreground pieces by area rank.

    Input structure:
    - The grid has black background, one grey single-pixel-wide L-shaped corner bracket, and exactly one marker
      pixel inside that bracket's bounding box.
    - The marker's color names a foreground color that also appears elsewhere in a larger 4-connected structure.
    - The marked color acts as bridge/glue inside that structure, while the grey bracket and the marker are separate.

    Rule:
    - Find the grey corner bracket and read the color of its enclosed marker.
    - Recolor every other pixel of that marker color to black, while keeping the marker itself and the grey bracket.
    - On the remaining non-background pixels excluding the bracket and marker, find 4-connected components and
      recolor them by descending area rank with red, yellow, pink, then maroon.

    Key insight:
    - The selected color is not the final target; removing it changes connectivity, and the post-erasure component
      sizes determine the new colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bracket, marker = _find_bracket_and_marker(g)
    marker_color = int(g[marker])
    assert marker_color != BG and marker_color != GREY

    out = g.copy()
    out[out == marker_color] = BG
    out[marker] = marker_color

    work_mask = out != BG
    for r, c in bracket:
        work_mask[r, c] = False
    work_mask[marker] = False

    comps = _components_from_mask(work_mask)
    comps.sort(key=lambda comp: (-len(comp), int(comp[:, 0].min()), int(comp[:, 1].min())))
    for rank, comp in enumerate(comps):
        color = RANK_COLORS[rank % len(RANK_COLORS)]
        out[comp[:, 0], comp[:, 1]] = color
    return out


def _grow_patch(area: int, side: str) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    dims = [
        (h, w)
        for h in range(2, 6)
        for w in range(2, 6)
        if area <= h * w and area >= h + w - 1
    ]
    h, w = random.choice(dims)
    if side == "up":
        anchor = (h - 1, random.randrange(w))
    elif side == "down":
        anchor = (0, random.randrange(w))
    elif side == "left":
        anchor = (random.randrange(h), w - 1)
    else:
        anchor = (random.randrange(h), 0)

    coords = {anchor}
    while len(coords) < area:
        frontier = []
        for r, c in coords:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in coords:
                    frontier.append((nr, nc))
        if not frontier:
            return _grow_patch(area, side)
        coords.add(random.choice(frontier))

    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, c0 = min(rows), min(cols)
    trimmed = {(r - r0, c - c0) for r, c in coords}
    return trimmed, (anchor[0] - r0, anchor[1] - c0)


def _choose_component_areas(count: int, blue_pixels: int) -> list[int] | None:
    lo = max(12, 20 - blue_pixels)
    hi = 35 - blue_pixels
    choices = [
        combo
        for combo in itertools.combinations(range(4, 11), count)
        if lo <= sum(combo) <= hi
    ]
    if not choices:
        return None
    areas = list(random.choice(choices))
    random.shuffle(areas)
    return areas


def _build_structure_specs() -> list[dict] | None:
    count = 4 if random.random() < 0.65 else 3
    sides = ["up", "down", "left", "right"]
    random.shuffle(sides)
    sides = sides[:count]
    bridge_lengths = {side: random.choice([1, 1, 2, 2, 3]) for side in sides}
    blue_pixels = 1 + sum(bridge_lengths.values())
    areas = _choose_component_areas(count, blue_pixels)
    if areas is None:
        return None

    color_choices = [GREEN, ORANGE]
    colors = [color_choices[i % 2] for i in range(count)]
    random.shuffle(colors)
    if len(set(colors)) == 1:
        colors[-1] = GREEN if colors[-1] == ORANGE else ORANGE

    specs = []
    for side, area, color in zip(sides, areas, colors):
        coords, anchor = _grow_patch(area, side)
        specs.append(
            {
                "side": side,
                "area": area,
                "color": color,
                "bridge": bridge_lengths[side],
                "coords": coords,
                "anchor": anchor,
            }
        )
    return specs


def _structure_relative_cells(specs: list[dict]) -> dict[tuple[int, int], int] | None:
    cells: dict[tuple[int, int], int] = {(0, 0): BLUE}
    for spec in specs:
        side = spec["side"]
        bridge = spec["bridge"]
        if side == "up":
            anchor_global = (-bridge - 1, 0)
            bridge_cells = [(-d, 0) for d in range(1, bridge + 1)]
        elif side == "down":
            anchor_global = (bridge + 1, 0)
            bridge_cells = [(d, 0) for d in range(1, bridge + 1)]
        elif side == "left":
            anchor_global = (0, -bridge - 1)
            bridge_cells = [(0, -d) for d in range(1, bridge + 1)]
        else:
            anchor_global = (0, bridge + 1)
            bridge_cells = [(0, d) for d in range(1, bridge + 1)]

        for rc in bridge_cells:
            if rc in cells and cells[rc] != BLUE:
                return None
            cells[rc] = BLUE

        ar, ac = spec["anchor"]
        base_r = anchor_global[0] - ar
        base_c = anchor_global[1] - ac
        for r, c in spec["coords"]:
            pos = (base_r + r, base_c + c)
            if pos in cells:
                return None
            cells[pos] = int(spec["color"])
    return cells


def _translate_cells(
    relative_cells: dict[tuple[int, int], int],
    shape: tuple[int, int],
) -> dict[tuple[int, int], int] | None:
    rows = [r for r, _ in relative_cells]
    cols = [c for _, c in relative_cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    h, w = shape
    min_hr = 1 - r0
    max_hr = h - 2 - r1
    min_hc = 1 - c0
    max_hc = w - 2 - c1
    if min_hr > max_hr or min_hc > max_hc:
        return None
    hub_r = random.randint(min_hr, max_hr)
    hub_c = random.randint(min_hc, max_hc)
    return {(hub_r + r, hub_c + c): color for (r, c), color in relative_cells.items()}


def _forbidden_with_margin(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    forbidden = set(cells)
    for r, c in list(cells):
        for dr, dc in DIR8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                forbidden.add((nr, nc))
    return forbidden


def _bracket_cells(height: int, width: int, orientation: str) -> set[tuple[int, int]]:
    if orientation == "tl":
        return {(0, c) for c in range(width)} | {(r, 0) for r in range(height)}
    if orientation == "tr":
        return {(0, c) for c in range(width)} | {(r, width - 1) for r in range(height)}
    if orientation == "bl":
        return {(height - 1, c) for c in range(width)} | {(r, 0) for r in range(height)}
    return {(height - 1, c) for c in range(width)} | {(r, width - 1) for r in range(height)}


def _place_bracket(
    grid: np.ndarray,
    structure_cells: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    forbidden = _forbidden_with_margin(structure_cells, grid.shape)
    choices = []
    for bh in (3, 4):
        for bw in (3, 4):
            for orientation in ("tl", "tr", "bl", "br"):
                bracket_local = _bracket_cells(bh, bw, orientation)
                interior = [
                    (r, c)
                    for r in range(bh)
                    for c in range(bw)
                    if (r, c) not in bracket_local
                ]
                for r0 in range(h - bh + 1):
                    for c0 in range(w - bw + 1):
                        cells = {(r0 + r, c0 + c) for r, c in bracket_local}
                        if cells & forbidden:
                            continue
                        marker_options = [
                            (r0 + r, c0 + c)
                            for r, c in interior
                            if (r0 + r, c0 + c) not in forbidden
                        ]
                        if marker_options:
                            choices.append((cells, random.choice(marker_options)))
    if not choices:
        return False
    bracket_cells, marker = random.choice(choices)
    for r, c in bracket_cells:
        grid[r, c] = GREY
    grid[marker] = BLUE
    return True


def _build_input_grid(shape: tuple[int, int]) -> np.ndarray | None:
    specs = _build_structure_specs()
    if specs is None:
        return None
    relative_cells = _structure_relative_cells(specs)
    if relative_cells is None:
        return None
    absolute = _translate_cells(relative_cells, shape)
    if absolute is None:
        return None

    grid = np.zeros(shape, dtype=int)
    for (r, c), color in absolute.items():
        grid[r, c] = int(color)

    if not _place_bracket(grid, set(absolute)):
        return None
    return grid


def _erase_selected(grid: np.ndarray) -> tuple[np.ndarray, set[tuple[int, int]], tuple[int, int]]:
    bracket, marker = _find_bracket_and_marker(grid)
    erased = grid.copy()
    erased[erased == BLUE] = BG
    erased[marker] = BLUE
    return erased, bracket, marker


def _validate_instance(grid: np.ndarray) -> bool:
    try:
        bracket, marker = _find_bracket_and_marker(grid)
    except AssertionError:
        return False
    if int(grid[marker]) != BLUE:
        return False

    structure_mask = grid != BG
    for r, c in bracket:
        structure_mask[r, c] = False
    structure_mask[marker] = False
    before = _components_from_mask(structure_mask)
    if len(before) != 1:
        return False
    if not 20 <= len(before[0]) <= 35:
        return False

    erased, bracket, marker = _erase_selected(grid)
    after_mask = erased != BG
    for r, c in bracket:
        after_mask[r, c] = False
    after_mask[marker] = False
    comps = _components_from_mask(after_mask)
    areas = [len(comp) for comp in comps]
    if len(comps) < 3 or len(comps) > 4:
        return False
    if len(set(areas)) != len(areas):
        return False

    expected = erased.copy()
    comps.sort(key=lambda comp: (-len(comp), int(comp[:, 0].min()), int(comp[:, 1].min())))
    for rank, comp in enumerate(comps):
        expected[comp[:, 0], comp[:, 1]] = RANK_COLORS[rank]
    if not np.array_equal(main(grid), expected):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(500):
        shape = (random.randint(14, 18), random.randint(14, 18))
        grid = _build_input_grid(shape)
        if grid is not None and _validate_instance(grid):
            return grid
    raise RuntimeError("failed to generate a valid puzzle-6 input")
