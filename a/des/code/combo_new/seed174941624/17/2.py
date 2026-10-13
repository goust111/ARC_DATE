from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BG = 0
WALL = 5
MAROON = 9
INPUT_ARROW_COLORS = [1, 2, 3, 4, 6, 7, 8]

DIR_VECS = {
    "R": (0, 1),
    "L": (0, -1),
    "U": (-1, 0),
    "D": (1, 0),
}

ARROW_OFFSETS = {
    "R": ((0, 0), (-1, -1), (1, -1)),
    "L": ((0, 0), (-1, 1), (1, 1)),
    "U": ((0, 0), (1, -1), (1, 1)),
    "D": ((0, 0), (-1, -1), (-1, 1)),
}

GENERATE_ATTEMPTS = 400
MARGIN_RANGE = (1, 3)
CLUTTER_WALL_RANGE = (0, 4)


@dataclass(frozen=True)
class Arrow:
    color: int
    head: tuple[int, int]
    orientation: str
    pixels: tuple[tuple[int, int], ...]

    @property
    def direction(self) -> tuple[int, int]:
        return DIR_VECS[self.orientation]


@dataclass(frozen=True)
class Ray:
    source_index: int
    color: int
    cells: tuple[tuple[int, int], ...]
    endpoint: tuple[int, int] | None
    stop_reason: str


def _arrow_pixels(head: tuple[int, int], orientation: str) -> tuple[tuple[int, int], ...]:
    hr, hc = head
    return tuple((hr + dr, hc + dc) for dr, dc in ARROW_OFFSETS[orientation])


def _decode_arrow(coords: list[tuple[int, int]]) -> tuple[str, tuple[int, int]]:
    ptset = set(coords)
    matches: list[tuple[str, tuple[int, int]]] = []
    for orientation, offsets in ARROW_OFFSETS.items():
        for head in coords:
            hr, hc = head
            expected = {(hr + dr, hc + dc) for dr, dc in offsets}
            if expected == ptset:
                matches.append((orientation, (hr, hc)))
    assert len(matches) == 1, "each colored 3-pixel component must be one arrow"
    return matches[0]


def _extract_arrows(grid: np.ndarray) -> list[Arrow]:
    g = np.asarray(grid, dtype=int)
    colors = [int(c) for c in np.unique(g) if int(c) not in {BG, WALL}]
    arrows: list[Arrow] = []
    occupied: set[tuple[int, int]] = set()
    for color in colors:
        coords = [tuple(map(int, rc)) for rc in np.argwhere(g == int(color))]
        assert len(coords) == 3, "each arrow color must occupy exactly three cells"
        orientation, head = _decode_arrow(coords)
        pixels = tuple(sorted(coords))
        for cell in pixels:
            assert cell not in occupied, "arrows must not overlap"
            occupied.add(cell)
        arrows.append(
            Arrow(
                color=int(color),
                head=(int(head[0]), int(head[1])),
                orientation=orientation,
                pixels=pixels,
            )
        )
    arrows.sort(key=lambda arrow: (arrow.head[0], arrow.head[1], arrow.color))
    assert arrows, "expected at least one arrow"
    return arrows


def _adjacent_to_arrow(endpoint: tuple[int, int], arrow: Arrow) -> bool:
    er, ec = endpoint
    return any(abs(er - r) + abs(ec - c) == 1 for r, c in arrow.pixels)


def _analyze_scene(grid: np.ndarray) -> tuple[list[Arrow], list[Ray], dict[tuple[int, int], int], list[list[int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    arrows = _extract_arrows(g)
    arrow_owner: dict[tuple[int, int], int] = {}
    for idx, arrow in enumerate(arrows):
        for cell in arrow.pixels:
            arrow_owner[cell] = idx

    rays: list[Ray] = []
    painted: dict[tuple[int, int], int] = {}
    for idx, arrow in enumerate(arrows):
        dr, dc = arrow.direction
        rr = int(arrow.head[0] + dr)
        cc = int(arrow.head[1] + dc)
        path: list[tuple[int, int]] = []
        stop_reason = "boundary"
        while 0 <= rr < h and 0 <= cc < w:
            if int(g[rr, cc]) == WALL:
                stop_reason = "wall"
                break
            if (rr, cc) in arrow_owner:
                stop_reason = "arrow"
                break
            path.append((int(rr), int(cc)))
            rr += dr
            cc += dc
        endpoint = path[-1] if path else None
        ray = Ray(
            source_index=int(idx),
            color=int(arrow.color),
            cells=tuple(path),
            endpoint=endpoint,
            stop_reason=stop_reason,
        )
        rays.append(ray)
        for cell in ray.cells:
            previous = painted.get(cell)
            assert previous in (None, ray.color), "different-colored rays must not overlap"
            painted[cell] = int(ray.color)

    received: list[list[int]] = [[] for _ in arrows]
    for ray in rays:
        if ray.endpoint is None:
            continue
        for idx, arrow in enumerate(arrows):
            if idx == ray.source_index:
                continue
            if _adjacent_to_arrow(ray.endpoint, arrow):
                received[idx].append(int(ray.color))

    return arrows, rays, painted, received


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast colored rays from every arrow and recolor each arrow by the rays that stop beside it.

    Input structure:
    - The background is black.
    - Each non-grey foreground color forms exactly one 3-pixel arrow: one head pixel and two tail pixels arranged to point right, left, up, or down.
    - Grey singleton wall pixels act as ray blockers; arrows have distinct colors and do not overlap.

    Rule:
    - From each arrow head, extend a straight ray in the arrow's facing direction using that arrow's own color.
    - A ray paints background cells until the next cell would be a grey wall or any arrow pixel; the wall or arrow cell itself is not overwritten.
    - After all rays are drawn, recolor each arrow by how many other rays end in a 4-neighbor cell next to it: zero received rays keeps the original color, one received ray copies that ray's color, and two or more received rays recolor the full arrow to maroon.
    - Painted ray cells remain visible in the output.

    Key insight:
    - Ray drawing and arrow recoloring are separate layers: first simulate every ray against the original blockers, then decide each arrow's final color only from which ray endpoints stop adjacent to it.
    """
    g = np.asarray(grid, dtype=int)
    arrows, _rays, painted, received = _analyze_scene(g)

    out = g.copy()
    for (r, c), color in painted.items():
        out[int(r), int(c)] = int(color)

    for arrow, colors in zip(arrows, received):
        if len(colors) == 0:
            final_color = int(arrow.color)
        elif len(colors) == 1:
            final_color = int(colors[0])
        else:
            final_color = MAROON
        for r, c in arrow.pixels:
            out[int(r), int(c)] = int(final_color)

    return out


def _shift_cells(cells: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(int(r + dr), int(c + dc)) for r, c in cells]


def _base_specs() -> tuple[list[tuple[str, tuple[int, int], str]], set[tuple[int, int]]]:
    extra_multi = random.random() < 0.5

    dist_a = random.randint(4, 6)
    dist_b = random.randint(4, 6)
    dist_c = random.randint(4, 6) if extra_multi else None
    gap_r = random.randint(6, 8)
    gap_c = random.randint(7, 10)
    dist_d = random.randint(4, 6)
    wall_m = random.randint(4, 6)
    wall_s = random.randint(4, 6)
    z_up = random.randint(4, 6)
    z_right = random.randint(4, 6)
    wall_z = random.randint(4, 6)

    specs = [
        ("A", (0, -dist_a), "R"),
        ("B", (-dist_b, -1), "D"),
        ("M", (0, 0), "R"),
        ("S", (gap_r, gap_c), "L"),
        ("D", (gap_r + dist_d, gap_c + 1), "U"),
        ("Z", (-z_up, gap_c + z_right), "D"),
    ]
    if dist_c is not None:
        specs.append(("C", (dist_c, -1), "U"))

    walls = {
        (0, wall_m),
        (gap_r, gap_c - wall_s),
        (-z_up + wall_z, gap_c + z_right),
    }
    return specs, walls


def _layout_grid(specs: list[tuple[str, tuple[int, int], str]], walls: set[tuple[int, int]]) -> np.ndarray | None:
    feature_cells = set(walls)
    for _name, head, orientation in specs:
        feature_cells.update(_arrow_pixels(head, orientation))

    min_r = min(r for r, _c in feature_cells)
    max_r = max(r for r, _c in feature_cells)
    min_c = min(c for _r, c in feature_cells)
    max_c = max(c for _r, c in feature_cells)

    top = random.randint(*MARGIN_RANGE)
    bottom = random.randint(*MARGIN_RANGE)
    left = random.randint(*MARGIN_RANGE)
    right = random.randint(*MARGIN_RANGE)

    height = int(max_r - min_r + 1 + top + bottom)
    width = int(max_c - min_c + 1 + left + right)
    if height > 30 or width > 30:
        return None

    shift_r = int(top - min_r)
    shift_c = int(left - min_c)
    colors = random.sample(INPUT_ARROW_COLORS, len(specs))

    grid = np.full((height, width), BG, dtype=int)
    shifted_walls = _shift_cells(list(walls), shift_r, shift_c)
    for r, c in shifted_walls:
        assert int(grid[r, c]) == BG
        grid[int(r), int(c)] = WALL

    for color, (_name, head, orientation) in zip(colors, specs):
        for r, c in _arrow_pixels((head[0] + shift_r, head[1] + shift_c), orientation):
            assert int(grid[r, c]) == BG
            grid[int(r), int(c)] = int(color)

    return grid


def _chebyshev_far(cell: tuple[int, int], others: list[tuple[int, int]]) -> bool:
    return all(max(abs(cell[0] - r), abs(cell[1] - c)) > 1 for r, c in others)


def _add_clutter_walls(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    arrows, rays, _painted, _received = _analyze_scene(g)
    protected: list[tuple[int, int]] = []
    for arrow in arrows:
        protected.extend(arrow.pixels)
    for ray in rays:
        protected.extend(ray.cells)
    protected.extend([tuple(map(int, rc)) for rc in np.argwhere(g == WALL)])

    candidates = [
        (int(r), int(c))
        for r in range(g.shape[0])
        for c in range(g.shape[1])
        if int(g[r, c]) == BG and _chebyshev_far((int(r), int(c)), protected)
    ]
    random.shuffle(candidates)
    placed: list[tuple[int, int]] = []
    target = random.randint(*CLUTTER_WALL_RANGE)
    for cell in candidates:
        if len(placed) >= target:
            break
        if _chebyshev_far(cell, placed):
            g[cell] = WALL
            placed.append(cell)
    return g


def _apply_random_symmetry(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    g = np.rot90(g, random.randrange(4))
    if random.random() < 0.5:
        g = np.fliplr(g)
    return np.array(g, dtype=int, copy=True)


def _scene_is_good(grid: np.ndarray) -> bool:
    try:
        arrows, rays, painted, received = _analyze_scene(grid)
    except AssertionError:
        return False

    if not any(len(colors) == 0 for colors in received):
        return False
    if not any(len(colors) == 1 for colors in received):
        return False
    if not any(len(colors) >= 2 for colors in received):
        return False

    reasons = {ray.stop_reason for ray in rays}
    if "arrow" not in reasons or "wall" not in reasons:
        return False

    if len(arrows) < 6:
        return False

    for (r, c), color in painted.items():
        if int(grid[r, c]) not in {BG, color}:
            return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        specs, walls = _base_specs()
        grid = _layout_grid(specs, walls)
        if grid is None:
            continue
        if not _scene_is_good(grid):
            continue

        grid = _add_clutter_walls(grid)
        grid = _apply_random_symmetry(grid)
        if not _scene_is_good(grid):
            continue

        try:
            out = main(grid)
        except AssertionError:
            continue
        if not (isinstance(out, np.ndarray) and out.ndim == 2 and max(out.shape) <= 30):
            continue
        return grid

    raise RuntimeError(f"generate_input: failed after {GENERATE_ATTEMPTS} attempts")
