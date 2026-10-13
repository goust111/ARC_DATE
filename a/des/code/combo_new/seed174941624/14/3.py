from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


DIRS: dict[str, tuple[int, int]] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
PALETTE = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 120


@dataclass(frozen=True)
class Frame:
    color: int
    bbox: tuple[int, int, int, int]
    gap: tuple[int, int] | None
    gap_side: str | None


@dataclass(frozen=True)
class GroupSpec:
    depth: int
    inner_h: int
    inner_w: int
    color: int
    gap_side: str
    gap_offset: int

    @property
    def outer_h(self) -> int:
        return self.inner_h + 4 * (self.depth - 1)

    @property
    def outer_w(self) -> int:
        return self.inner_w + 4 * (self.depth - 1)


@dataclass(frozen=True)
class PlacedGroup:
    spec: GroupSpec
    r0: int
    c0: int

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.r0, self.c0, self.r0 + self.spec.outer_h - 1, self.c0 + self.spec.outer_w - 1)

    @property
    def inner_bbox(self) -> tuple[int, int, int, int]:
        offset = 2 * (self.spec.depth - 1)
        r0 = self.r0 + offset
        c0 = self.c0 + offset
        return (r0, c0, r0 + self.spec.inner_h - 1, c0 + self.spec.inner_w - 1)

    @property
    def gap_coord(self) -> tuple[int, int]:
        r0, c0, r1, c1 = self.inner_bbox
        if self.spec.gap_side == "up":
            return (r0, c0 + self.spec.gap_offset)
        if self.spec.gap_side == "down":
            return (r1, c0 + self.spec.gap_offset)
        if self.spec.gap_side == "left":
            return (r0 + self.spec.gap_offset, c0)
        return (r0 + self.spec.gap_offset, c1)

    def outside_ray_cells(self, shape: tuple[int, int]) -> list[tuple[int, int]]:
        h, w = shape
        r0, c0, r1, c1 = self.bbox
        gap_r, gap_c = self.gap_coord
        if self.spec.gap_side == "up":
            return [(r, gap_c) for r in range(0, r0)]
        if self.spec.gap_side == "down":
            return [(r, gap_c) for r in range(r1 + 1, h)]
        if self.spec.gap_side == "left":
            return [(gap_r, c) for c in range(0, c0)]
        return [(gap_r, c) for c in range(c1 + 1, w)]


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _split_runs(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    runs: list[tuple[int, int]] = []
    start = prev = indices[0]
    for value in indices[1:]:
        if value == prev + 1:
            prev = value
            continue
        runs.append((start, prev))
        start = prev = value
    runs.append((start, prev))
    return runs


def _background_color(g: np.ndarray) -> int:
    vals, cnts = np.unique(g, return_counts=True)
    return int(vals[int(cnts.argmax())])


def _find_legend(g: np.ndarray, bg: int) -> tuple[list[int], np.ndarray]:
    h, w = g.shape
    best: tuple[int, int, int, int, int, int] | None = None

    for c in range(w):
        rows = [int(r) for r in np.flatnonzero(g[:, c] != bg)]
        for r0, r1 in _split_runs(rows):
            length = r1 - r0 + 1
            if length < 2:
                continue
            if c != 0 and r0 != 0:
                continue
            if c > 0 and np.any(g[r0 : r1 + 1, c - 1] != bg):
                continue
            if c < w - 1 and np.any(g[r0 : r1 + 1, c + 1] != bg):
                continue
            if r0 > 0 and int(g[r0 - 1, c]) != bg:
                continue
            if r1 < h - 1 and int(g[r1 + 1, c]) != bg:
                continue
            candidate = (length, 1 if c == 0 else 0, -c, -r0, c, r0)
            if best is None or candidate > best:
                best = candidate

    assert best is not None, "expected a boundary legend column"
    _, _, _, _, c, r0 = best
    rows = [int(r) for r in np.flatnonzero(g[:, c] != bg)]
    runs = [run for run in _split_runs(rows) if run[0] == r0]
    assert len(runs) == 1
    rr0, rr1 = runs[0]
    colors = [int(g[r, c]) for r in range(rr0, rr1 + 1)]
    mask = np.zeros_like(g, dtype=bool)
    mask[rr0 : rr1 + 1, c] = True
    return colors, mask


def _component(g: np.ndarray, start: tuple[int, int], seen: np.ndarray, blocked: np.ndarray) -> list[tuple[int, int]]:
    h, w = g.shape
    sr, sc = start
    color = int(g[sr, sc])
    stack = [(sr, sc)]
    seen[sr, sc] = True
    coords: list[tuple[int, int]] = []

    while stack:
        r, c = stack.pop()
        coords.append((r, c))
        for rr, cc in _neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and not seen[rr, cc] and not blocked[rr, cc] and int(g[rr, cc]) == color:
                seen[rr, cc] = True
                stack.append((rr, cc))
    return coords


def _border_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    out: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        out.add((r0, c))
        out.add((r1, c))
    for r in range(r0, r1 + 1):
        out.add((r, c0))
        out.add((r, c1))
    return out


def _frame_from_component(g: np.ndarray, coords: list[tuple[int, int]]) -> Frame | None:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return None

    border = _border_cells((r0, c0, r1, c1))
    coord_set = set(coords)
    if not coord_set <= border:
        return None

    missing = sorted(border - coord_set)
    if len(missing) > 1:
        return None
    if len(coord_set) != len(border) - len(missing):
        return None

    gap = None
    gap_side = None
    if missing:
        mr, mc = missing[0]
        sides = []
        if mr == r0:
            sides.append("up")
        if mr == r1:
            sides.append("down")
        if mc == c0:
            sides.append("left")
        if mc == c1:
            sides.append("right")
        if len(sides) != 1:
            return None
        gap = (mr, mc)
        gap_side = sides[0]

    color = int(g[coords[0]])
    return Frame(color=color, bbox=(r0, c0, r1, c1), gap=gap, gap_side=gap_side)


def _find_groups(g: np.ndarray, bg: int, legend_mask: np.ndarray) -> list[list[Frame]]:
    h, w = g.shape
    blocked = legend_mask | (g == bg)
    seen = np.zeros((h, w), dtype=bool)
    frames: list[Frame] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or blocked[r, c]:
                continue
            coords = _component(g, (r, c), seen, blocked)
            frame = _frame_from_component(g, coords)
            if frame is not None:
                frames.append(frame)

    key_to_frame = {(frame.color, *frame.bbox): frame for frame in frames}
    child_of: dict[tuple[int, int, int, int, int], Frame] = {}
    has_parent: set[tuple[int, int, int, int, int]] = set()
    for frame in frames:
        r0, c0, r1, c1 = frame.bbox
        child_key = (frame.color, r0 + 2, c0 + 2, r1 - 2, c1 - 2)
        if child_key in key_to_frame:
            child = key_to_frame[child_key]
            child_of[(frame.color, *frame.bbox)] = child
            has_parent.add((child.color, *child.bbox))

    groups: list[list[Frame]] = []
    for frame in frames:
        key = (frame.color, *frame.bbox)
        if key in has_parent:
            continue
        chain = [frame]
        cursor = frame
        while (cursor.color, *cursor.bbox) in child_of:
            cursor = child_of[(cursor.color, *cursor.bbox)]
            chain.append(cursor)
        assert chain[-1].gap is not None and chain[-1].gap_side is not None, "innermost frame must contain the gap"
        assert all(item.gap is None for item in chain[:-1]), "only the innermost frame may have a gap"
        groups.append(chain)

    assert groups, "expected at least one nested-frame group"
    return groups


def _paint_group(out: np.ndarray, group: list[Frame], fill_color: int, bg: int) -> None:
    r0, c0, r1, c1 = group[0].bbox
    region = out[r0 : r1 + 1, c0 : c1 + 1]
    region[region == bg] = fill_color

    gap_r, gap_c = group[-1].gap
    dr, dc = DIRS[group[-1].gap_side]
    rr, cc = gap_r, gap_c
    h, w = out.shape
    while 0 <= rr < h and 0 <= cc < w:
        out[rr, cc] = fill_color
        rr += dr
        cc += dc


def main(grid: np.ndarray) -> np.ndarray:
    """Fill nested rectangular frames by legend depth, then fire a same-color ray outward through the innermost gap.

    Input structure:
    - The background is black.
    - One boundary-attached vertical legend column lists fill colors from top to bottom, where the first pixel means depth 1, the second means depth 2, and so on.
    - Away from the legend, the grid contains one or more non-overlapping groups of nested 1-pixel-thick monochrome rectangular frames with a 1-cell gap between successive layers.
    - In each group, only the innermost frame has a single missing border pixel, and that missing pixel determines one of four cardinal opening directions.

    Rule:
    - Detect each nested-frame group and count its number of layers N.
    - Look up the Nth legend color C.
    - Fill every background cell inside the outermost frame bbox of that group, including all spaces between layers and the innermost cavity, with C while leaving untouched frame pixels unchanged.
    - Starting from the innermost missing border pixel, paint a 1-pixel-wide ray of color C in the opening direction all the way to the grid edge, overwriting any outer-frame pixels on that line.

    Key insight:
    - The output color is not encoded by the frame color itself but by the nesting depth through the edge legend, and the innermost gap then converts that chosen fill color into an outward ray direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _background_color(g)

    legend_colors, legend_mask = _find_legend(g, bg)
    groups = _find_groups(g, bg, legend_mask)

    out = g.copy()
    for group in groups:
        depth = len(group)
        assert 1 <= depth <= len(legend_colors), "group depth must be covered by the legend"
        fill_color = legend_colors[depth - 1]
        _paint_group(out, group, fill_color, bg)
    return out


def _draw_rect_border(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _legend_cells(mask: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = mask.shape
    for r, c in cells:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                mask[rr, cc] = True


def _spec_for_group_count(group_count: int) -> tuple[int, tuple[int, int]]:
    if group_count == 1:
        return random.randint(2, 4), (3, 7)
    if group_count == 2:
        return random.randint(2, 4), (3, 6)
    return random.randint(2, 3), (3, 5)


def _make_group_specs(group_count: int) -> tuple[int, list[GroupSpec]] | None:
    legend_len, inner_bounds = _spec_for_group_count(group_count)
    fill_colors = random.sample(PALETTE, legend_len)
    frame_pool = [color for color in PALETTE if color not in fill_colors]
    if len(frame_pool) < group_count:
        return None
    frame_colors = random.sample(frame_pool, group_count)

    depths = [random.randint(1, legend_len) for _ in range(group_count)]
    depths[random.randrange(group_count)] = legend_len

    specs: list[GroupSpec] = []
    for depth, color in zip(depths, frame_colors, strict=False):
        inner_h = random.randint(*inner_bounds)
        inner_w = random.randint(*inner_bounds)
        gap_side = random.choice(tuple(DIRS))
        if gap_side in ("up", "down"):
            gap_offset = random.randint(1, inner_w - 2)
        else:
            gap_offset = random.randint(1, inner_h - 2)
        specs.append(
            GroupSpec(
                depth=depth,
                inner_h=inner_h,
                inner_w=inner_w,
                color=color,
                gap_side=gap_side,
                gap_offset=gap_offset,
            )
        )
    random.shuffle(specs)
    return legend_len, specs


def _grid_shape_for_specs(specs: list[GroupSpec]) -> tuple[int, int]:
    max_outer_h = max(spec.outer_h for spec in specs)
    max_outer_w = max(spec.outer_w for spec in specs)
    group_count = len(specs)
    if group_count == 1:
        h_lo = max(14, max_outer_h + 4)
        w_lo = max(14, max_outer_w + 4)
    elif group_count == 2:
        h_lo = max(18, max_outer_h + 5)
        w_lo = max(18, max_outer_w + 5)
    else:
        h_lo = max(23, max_outer_h + 6)
        w_lo = max(23, max_outer_w + 6)
    return random.randint(h_lo, 30), random.randint(w_lo, 30)


def _reserve_bbox(mask: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> None:
    h, w = mask.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _make_legend(h: int, w: int, colors: list[int]) -> tuple[list[tuple[int, int]], np.ndarray]:
    side = random.choice(("left", "top"))
    cells: list[tuple[int, int]] = []
    if side == "left":
        start_r = random.randint(0, h - len(colors))
        cells = [(start_r + idx, 0) for idx in range(len(colors))]
    else:
        col = random.randint(1, w - 2)
        cells = [(idx, col) for idx in range(len(colors))]
    reserved = np.zeros((h, w), dtype=bool)
    _legend_cells(reserved, cells)
    return cells, reserved


def _try_place_groups(h: int, w: int, specs: list[GroupSpec], reserved: np.ndarray) -> list[PlacedGroup] | None:
    occupied = reserved.copy()
    placed: list[PlacedGroup] = []

    for spec in sorted(specs, key=lambda item: item.outer_h * item.outer_w, reverse=True):
        max_r0 = h - spec.outer_h - 1
        max_c0 = w - spec.outer_w - 1
        if max_r0 < 1 or max_c0 < 1:
            return None

        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = random.randint(1, max_r0)
            c0 = random.randint(1, max_c0)
            placed_group = PlacedGroup(spec=spec, r0=r0, c0=c0)
            bbox = placed_group.bbox

            r0b, c0b, r1b, c1b = bbox
            rr0 = r0b - 1
            cc0 = c0b - 1
            rr1 = r1b + 1
            cc1 = c1b + 1
            if occupied[max(0, rr0) : min(h, rr1 + 1), max(0, cc0) : min(w, cc1 + 1)].any():
                continue

            ray_cells = placed_group.outside_ray_cells((h, w))
            if not ray_cells:
                continue
            if any(occupied[r, c] for r, c in ray_cells):
                continue

            _reserve_bbox(occupied, bbox, pad=1)
            for r, c in ray_cells:
                occupied[r, c] = True
            placed.append(placed_group)
            break
        else:
            return None

    random.shuffle(placed)
    return placed


def _render_input(h: int, w: int, legend_colors: list[int], legend_cells_pos: list[tuple[int, int]], groups: list[PlacedGroup]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)

    for (r, c), color in zip(legend_cells_pos, legend_colors, strict=False):
        grid[r, c] = color

    for placed in groups:
        for layer in range(placed.spec.depth):
            bbox = (
                placed.r0 + 2 * layer,
                placed.c0 + 2 * layer,
                placed.r0 + placed.spec.outer_h - 1 - 2 * layer,
                placed.c0 + placed.spec.outer_w - 1 - 2 * layer,
            )
            _draw_rect_border(grid, bbox, placed.spec.color)
        gap_r, gap_c = placed.gap_coord
        grid[gap_r, gap_c] = 0

    return grid


def _gen_attempt() -> np.ndarray | None:
    group_count = random.choice((1, 2, 3))
    spec_bundle = _make_group_specs(group_count)
    if spec_bundle is None:
        return None
    legend_len, specs = spec_bundle

    h, w = _grid_shape_for_specs(specs)
    legend_colors = random.sample(PALETTE, legend_len)
    frame_colors = random.sample([color for color in PALETTE if color not in legend_colors], group_count)
    recolored_specs = [
        GroupSpec(
            depth=spec.depth,
            inner_h=spec.inner_h,
            inner_w=spec.inner_w,
            color=color,
            gap_side=spec.gap_side,
            gap_offset=spec.gap_offset,
        )
        for spec, color in zip(specs, frame_colors, strict=False)
    ]

    legend_cells_pos, reserved = _make_legend(h, w, legend_colors)
    placed = _try_place_groups(h, w, recolored_specs, reserved)
    if placed is None:
        return None

    grid = _render_input(h, w, legend_colors, legend_cells_pos, placed)

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame puzzle instance")
