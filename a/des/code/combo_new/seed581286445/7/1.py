from __future__ import annotations

import random

import numpy as np


BG = 0
COLORS = tuple(range(10))
SIDES = ("top", "bottom", "left", "right")
DIR_BY_SIDE = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

GRID_SIDE_RANGE = (12, 30)
FRAME_COUNT_OPTIONS = (1, 2, 3, 4)
FRAME_OUTER_H_RANGE = (4, 9)
FRAME_OUTER_W_RANGE = (4, 9)
MAX_SEED_COUNT = 6
PLACE_RETRIES = 200
GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _parse_frame_component(coords: list[tuple[int, int]]) -> dict[str, object] | None:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return None

    expected = _perimeter_cells(r0, c0, r1, c1)
    actual = set(coords)
    if actual - expected:
        return None

    missing = expected - actual
    if len(missing) != 1:
        return None

    gap = next(iter(missing))
    if gap in {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}:
        return None

    if gap[0] == r0:
        side = "top"
    elif gap[0] == r1:
        side = "bottom"
    elif gap[1] == c0:
        side = "left"
    elif gap[1] == c1:
        side = "right"
    else:
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "gap": gap,
        "side": side,
        "direction": DIR_BY_SIDE[side],
    }


def _find_frames(g: np.ndarray) -> tuple[int, list[dict[str, object]]]:
    colors = [int(c) for c in np.unique(g) if int(c) != BG]
    candidates: list[tuple[int, list[dict[str, object]]]] = []
    for color in colors:
        comps = _components(g == color)
        if not comps:
            continue

        parsed: list[dict[str, object]] = []
        ok = True
        for comp in comps:
            desc = _parse_frame_component(comp)
            if desc is None:
                ok = False
                break
            parsed.append(desc)
        if ok and parsed:
            candidates.append((color, parsed))

    assert len(candidates) == 1, "expected exactly one frame color"
    return candidates[0]


def _frame_seed_info(g: np.ndarray, frame: dict[str, object]) -> tuple[int, int]:
    r0, c0, r1, c1 = frame["bbox"]
    inner = g[r0 + 1 : r1, c0 + 1 : c1]
    mask = inner != BG
    assert bool(mask.any()), "each frame must contain at least one seed"
    colors = np.unique(inner[mask])
    assert colors.size == 1, "all seeds inside one frame must share one color"
    return int(colors[0]), int(np.count_nonzero(mask))


def _draw_ray(
    out: np.ndarray,
    gap: tuple[int, int],
    direction: tuple[int, int],
    length: int,
    color: int,
) -> None:
    h, w = out.shape
    dr, dc = direction
    for step in range(length):
        rr = gap[0] + dr * step
        cc = gap[1] + dc * step
        assert 0 <= rr < h and 0 <= cc < w, "ray must stay inside the grid"
        out[rr, cc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend same-colored rays outward from each frame gap, with ray length equal to the number of seeds inside that frame.

    Input structure:
    - The background is black.
    - One foreground color forms one-pixel-thick rectangular frames, and each frame has exactly one one-pixel gap on one side.
    - Each frame contains one or more interior seed pixels; within a frame, all seeds share one color, while different frames may use different colors.

    Rule:
    - Identify each rectangular frame and locate its unique missing border cell.
    - Use the side containing that gap to choose the outward growth direction.
    - Count the interior seed pixels in the frame, then draw a straight ray of that many pixels starting at the gap cell and extending outward in the seed color.
    - Keep the original frames and seed pixels unchanged apart from adding the new ray cells.

    Key insight:
    - The gap encodes direction, the seed count encodes length, and the seed color encodes the ray color, so the transformation combines three independent visual cues from each frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _frame_color, frames = _find_frames(g)
    out = g.copy()

    for frame in frames:
        seed_color, seed_count = _frame_seed_info(g, frame)
        _draw_ray(
            out,
            frame["gap"],
            frame["direction"],
            seed_count,
            seed_color,
        )

    return out


def _side_gap(side: str, r0: int, c0: int, r1: int, c1: int) -> tuple[int, int]:
    if side == "top":
        return (r0, random.randint(c0 + 1, c1 - 1))
    if side == "bottom":
        return (r1, random.randint(c0 + 1, c1 - 1))
    if side == "left":
        return (random.randint(r0 + 1, r1 - 1), c0)
    return (random.randint(r0 + 1, r1 - 1), c1)


def _ray_capacity(h: int, w: int, gap: tuple[int, int], side: str) -> int:
    r, c = gap
    if side == "top":
        return r + 1
    if side == "bottom":
        return h - r
    if side == "left":
        return c + 1
    return w - c


def _expand_cells(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w:
                    expanded.add((rr, cc))
    return expanded


def _seed_capacity(inner_h: int, inner_w: int) -> int:
    return ((inner_h + 1) // 2) * ((inner_w + 1) // 2)


def _sample_seed_cells(inner_h: int, inner_w: int, count: int) -> list[tuple[int, int]] | None:
    offsets = [(0, 0), (0, 1), (1, 0), (1, 1)]
    random.shuffle(offsets)
    for ro, co in offsets:
        cells = [
            (r, c)
            for r in range(ro, inner_h, 2)
            for c in range(co, inner_w, 2)
        ]
        if len(cells) >= count:
            return random.sample(cells, count)
    return None


def _expected_from_specs(grid: np.ndarray, specs: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        _draw_ray(
            out,
            spec["gap"],
            spec["direction"],
            spec["seed_count"],
            spec["seed_color"],
        )
    return out


def _place_one_frame(
    grid: np.ndarray,
    reserved: np.ndarray,
    frame_color: int,
    seed_palette: list[int],
) -> dict[str, object] | None:
    h, w = grid.shape
    max_outer_h = min(FRAME_OUTER_H_RANGE[1], h - 1)
    max_outer_w = min(FRAME_OUTER_W_RANGE[1], w - 1)
    if max_outer_h < FRAME_OUTER_H_RANGE[0] or max_outer_w < FRAME_OUTER_W_RANGE[0]:
        return None

    for _ in range(PLACE_RETRIES):
        side = random.choice(SIDES)
        outer_h = random.randint(FRAME_OUTER_H_RANGE[0], max_outer_h)
        outer_w = random.randint(FRAME_OUTER_W_RANGE[0], max_outer_w)
        r0 = random.randint(0, h - outer_h)
        c0 = random.randint(0, w - outer_w)
        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1

        gap = _side_gap(side, r0, c0, r1, c1)
        capacity = _ray_capacity(h, w, gap, side)
        inner_h = outer_h - 2
        inner_w = outer_w - 2
        max_count = min(MAX_SEED_COUNT, capacity, _seed_capacity(inner_h, inner_w))
        if max_count < 1:
            continue

        seed_count = random.randint(1, max_count)
        local_seed_cells = _sample_seed_cells(inner_h, inner_w, seed_count)
        if local_seed_cells is None:
            continue

        bbox_cells = {(rr, cc) for rr in range(r0, r1 + 1) for cc in range(c0, c1 + 1)}
        if any(reserved[rr, cc] for rr, cc in bbox_cells):
            continue

        dr, dc = DIR_BY_SIDE[side]
        ray_cells = {(gap[0] + dr * step, gap[1] + dc * step) for step in range(seed_count)}
        if any(reserved[rr, cc] for rr, cc in ray_cells):
            continue

        border_cells = _perimeter_cells(r0, c0, r1, c1)
        border_cells.remove(gap)
        seed_cells = {(r0 + 1 + sr, c0 + 1 + sc) for sr, sc in local_seed_cells}

        seed_color = random.choice(seed_palette)
        for rr, cc in border_cells:
            grid[rr, cc] = frame_color
        for rr, cc in seed_cells:
            grid[rr, cc] = seed_color

        footprint = bbox_cells | ray_cells
        for rr, cc in _expand_cells(footprint, h, w):
            reserved[rr, cc] = True

        return {
            "gap": gap,
            "direction": DIR_BY_SIDE[side],
            "seed_count": seed_count,
            "seed_color": seed_color,
        }

    return None


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)
    grid = np.full((h, w), BG, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    frame_color = random.choice([c for c in COLORS if c != BG])
    seed_palette = [c for c in COLORS if c not in (BG, frame_color)]
    frame_count = random.choice(FRAME_COUNT_OPTIONS)

    specs: list[dict[str, object]] = []
    for _ in range(frame_count):
        spec = _place_one_frame(grid, reserved, frame_color, seed_palette)
        if spec is None:
            return None
        specs.append(spec)

    expected = _expected_from_specs(grid, specs)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
