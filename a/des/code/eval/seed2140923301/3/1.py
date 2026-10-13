import random
from typing import Iterable

import numpy as np


BG = 0
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDES = ("top", "bottom", "left", "right")
SIDE_TO_VEC = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
PERPENDICULAR = {
    "top": ("left", "right"),
    "bottom": ("left", "right"),
    "left": ("top", "bottom"),
    "right": ("top", "bottom"),
}


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
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
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _decode_legend(grid: np.ndarray) -> tuple[dict[int, int], np.ndarray]:
    h, w = grid.shape
    fg = grid != BG
    candidates = []
    for comp in _components4(fg):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if width != 2:
            continue
        if len(comp) != 2 * height:
            continue
        if not ((r0 == 0 or r1 == h - 1) and (c0 == 0 or c1 == w - 1)):
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(sub == BG):
            continue
        candidates.append((r0, c0, r1, c1))

    assert len(candidates) == 1, "expected exactly one corner legend"
    r0, c0, r1, c1 = candidates[0]
    mapping: dict[int, int] = {}
    legend_mask = np.zeros_like(grid, dtype=bool)
    for r in range(r0, r1 + 1):
        left = int(grid[r, c0])
        right = int(grid[r, c0 + 1])
        assert left != right, "legend pairs must map two different colors"
        assert left not in mapping, "legend keys must be unique"
        mapping[left] = right
        legend_mask[r, c0] = True
        legend_mask[r, c0 + 1] = True
    return mapping, legend_mask


def _gap_side(r: int, c: int, r0: int, c0: int, r1: int, c1: int) -> str:
    if r == r0:
        return "top"
    if r == r1:
        return "bottom"
    if c == c0:
        return "left"
    if c == c1:
        return "right"
    raise AssertionError("gap must lie on the rectangle perimeter")


def _rectangle_specs(grid: np.ndarray, mapping: dict[int, int], legend_mask: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    specs: list[dict[str, object]] = []
    for border_color, branch_color in mapping.items():
        coords = np.argwhere((grid == border_color) & ~legend_mask)
        assert len(coords) > 0, "every legend color must label a rectangle"

        r0 = int(coords[:, 0].min())
        r1 = int(coords[:, 0].max())
        c0 = int(coords[:, 1].min())
        c1 = int(coords[:, 1].max())
        assert r1 - r0 >= 3 and c1 - c0 >= 3, "frames need a non-empty interior"
        assert int(grid[r0, c0]) == border_color
        assert int(grid[r0, c1]) == border_color
        assert int(grid[r1, c0]) == border_color
        assert int(grid[r1, c1]) == border_color

        border_set = {tuple(map(int, rc)) for rc in coords}
        perimeter = set(_perimeter_cells(r0, c0, r1, c1))
        assert border_set <= perimeter, "border-color pixels must stay on one rectangle perimeter"

        gaps: list[tuple[tuple[int, int], str]] = []
        for pr, pc in perimeter:
            value = int(grid[pr, pc])
            if value == border_color:
                continue
            assert value == BG, "perimeter exceptions must be background gaps"
            assert (pr, pc) not in ((r0, c0), (r0, c1), (r1, c0), (r1, c1)), "corner gaps are unsupported"
            side = _gap_side(pr, pc, r0, c0, r1, c1)
            if side in ("top", "bottom"):
                assert int(grid[pr, pc - 1]) == border_color
                assert int(grid[pr, pc + 1]) == border_color
            else:
                assert int(grid[pr - 1, pc]) == border_color
                assert int(grid[pr + 1, pc]) == border_color
            gaps.append(((pr, pc), side))

        assert gaps, "each frame must have at least one gap"

        seeds: list[tuple[int, int]] = []
        seed_set: set[tuple[int, int]] = set()
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                value = int(grid[r, c])
                if value == BG:
                    continue
                assert value != border_color, "seed color must differ from border color"
                seeds.append((r, c))
                seed_set.add((r, c))

        assert seeds, "each frame must contain at least one seed"
        for r, c in seeds:
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                assert (nr, nc) not in seed_set, "seeds must be non-adjacent"

        specs.append(
            {
                "border_color": border_color,
                "branch_color": branch_color,
                "bbox": (r0, c0, r1, c1),
                "gaps": gaps,
                "seed_count": len(seeds),
            }
        )

    specs.sort(key=lambda spec: tuple(spec["bbox"]))  # deterministic drawing order
    return specs


def _solve(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid, dtype=int)
    assert grid.ndim == 2
    mapping, legend_mask = _decode_legend(grid)
    specs = _rectangle_specs(grid, mapping, legend_mask)
    h, w = grid.shape
    out = grid.copy()

    for spec in specs:
        seed_count = int(spec["seed_count"])
        branch_color = int(spec["branch_color"])
        for (gr, gc), side in spec["gaps"]:
            dr, dc = SIDE_TO_VEC[side]
            r, c = gr + dr, gc + dc
            for _ in range(seed_count):
                if not (0 <= r < h and 0 <= c < w):
                    break
                if int(out[r, c]) != BG:
                    break
                out[r, c] = branch_color
                r += dr
                c += dc

    return out


def main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid)


GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
GEN_RETRIES = 400
MAX_PLACE_TRIES = 250
MOAT = 1
PALETTE = tuple(range(1, 10))


def _corner_legend_bounds(h: int, w: int, count: int, corner: str) -> tuple[int, int, int, int]:
    if corner == "tl":
        return 0, 0, count - 1, 1
    if corner == "tr":
        return 0, w - 2, count - 1, w - 1
    if corner == "bl":
        return h - count, 0, h - 1, 1
    if corner == "br":
        return h - count, w - 2, h - 1, w - 1
    raise ValueError(corner)


def _expanded_cells(r0: int, c0: int, r1: int, c1: int, h: int, w: int, pad: int) -> set[tuple[int, int]]:
    cells = set()
    for r in range(max(0, r0 - pad), min(h - 1, r1 + pad) + 1):
        for c in range(max(0, c0 - pad), min(w - 1, c1 + pad) + 1):
            cells.add((r, c))
    return cells


def _ray_cells(start: tuple[int, int], vec: tuple[int, int], steps: int, h: int, w: int) -> list[tuple[int, int]]:
    dr, dc = vec
    r, c = start
    cells = []
    for _ in range(steps):
        if not (0 <= r < h and 0 <= c < w):
            break
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _available_to_boundary(r0: int, c0: int, r1: int, c1: int, side: str, h: int, w: int) -> int:
    if side == "top":
        return r0
    if side == "bottom":
        return h - 1 - r1
    if side == "left":
        return c0
    if side == "right":
        return w - 1 - c1
    raise ValueError(side)


def _sample_seed_positions(rng: random.Random, ih: int, iw: int, count: int) -> list[tuple[int, int]]:
    options = [
        [(r, c) for r in range(ih) for c in range(iw) if (r + c) % 2 == parity]
        for parity in (0, 1)
    ]
    options = [cells for cells in options if len(cells) >= count]
    assert options
    cells = rng.choice(options)
    return rng.sample(cells, count)


def _gap_cells_for_sides(
    rng: random.Random,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    sides: Iterable[str],
) -> dict[str, tuple[int, int]]:
    gaps: dict[str, tuple[int, int]] = {}
    for side in sides:
        if side == "top":
            gaps[side] = (r0, rng.randint(c0 + 1, c1 - 1))
        elif side == "bottom":
            gaps[side] = (r1, rng.randint(c0 + 1, c1 - 1))
        elif side == "left":
            gaps[side] = (rng.randint(r0 + 1, r1 - 1), c0)
        elif side == "right":
            gaps[side] = (rng.randint(r0 + 1, r1 - 1), c1)
        else:
            raise ValueError(side)
    return gaps


def _try_place_object(
    rng: random.Random,
    grid: np.ndarray,
    reserved: np.ndarray,
    border_color: int,
    seed_palette: list[int],
    mode: str,
) -> bool:
    h, w = grid.shape
    for _ in range(MAX_PLACE_TRIES):
        ih = rng.randint(2, 5)
        iw = rng.randint(2, 5)
        height = ih + 2
        width = iw + 2
        capacity = max(((ih * iw) + 1) // 2, 1)
        if mode == "truncated":
            max_seed = min(4, capacity)
            if max_seed < 2:
                continue
            seed_count = rng.randint(2, max_seed)
            near_side = rng.choice(SIDES)
            side_pool = [near_side]
            extra_options = list(PERPENDICULAR[near_side])
            rng.shuffle(extra_options)
            for side in extra_options:
                if rng.random() < 0.5:
                    side_pool.append(side)
            gap_sides = tuple(side_pool)
        else:
            seed_count = rng.randint(1, min(4, capacity))
            near_side = None
            gap_count = rng.choices([1, 2, 3], weights=[4, 3, 1])[0]
            gap_sides = tuple(rng.sample(SIDES, gap_count))

        min_r0 = 1
        max_r0 = h - height - 1
        min_c0 = 1
        max_c0 = w - width - 1
        if max_r0 < min_r0 or max_c0 < min_c0:
            continue

        if "top" in gap_sides:
            if near_side == "top":
                max_r0 = min(max_r0, seed_count - 1)
            elif mode == "full":
                min_r0 = max(min_r0, seed_count)
        if "bottom" in gap_sides:
            if near_side == "bottom":
                min_r0 = max(min_r0, h - height - (seed_count - 1))
            elif mode == "full":
                max_r0 = min(max_r0, h - height - seed_count)
        if "left" in gap_sides:
            if near_side == "left":
                max_c0 = min(max_c0, seed_count - 1)
            elif mode == "full":
                min_c0 = max(min_c0, seed_count)
        if "right" in gap_sides:
            if near_side == "right":
                min_c0 = max(min_c0, w - width - (seed_count - 1))
            elif mode == "full":
                max_c0 = min(max_c0, w - width - seed_count)

        if max_r0 < min_r0 or max_c0 < min_c0:
            continue

        r0 = rng.randint(min_r0, max_r0)
        c0 = rng.randint(min_c0, max_c0)
        r1 = r0 + height - 1
        c1 = c0 + width - 1

        gaps = _gap_cells_for_sides(rng, r0, c0, r1, c1, gap_sides)
        if mode == "truncated":
            avail = _available_to_boundary(r0, c0, r1, c1, near_side, h, w)
            if not (1 <= avail < seed_count):
                continue
        else:
            if any(_available_to_boundary(r0, c0, r1, c1, side, h, w) < seed_count for side in gap_sides):
                continue

        occupied = _expanded_cells(r0, c0, r1, c1, h, w, MOAT)
        rays: set[tuple[int, int]] = set()
        for side, gap in gaps.items():
            dr, dc = SIDE_TO_VEC[side]
            ray = _ray_cells((gap[0] + dr, gap[1] + dc), (dr, dc), seed_count, h, w)
            if not ray:
                occupied = set()
                break
            rays.update(ray)
        if not occupied:
            continue

        needed = occupied | rays
        if any(reserved[r, c] for r, c in needed):
            continue
        if any(int(grid[r, c]) != BG for r, c in occupied):
            continue

        seed_positions = _sample_seed_positions(rng, ih, iw, seed_count)
        perimeter = set(_perimeter_cells(r0, c0, r1, c1))
        gap_set = set(gaps.values())
        for pr, pc in perimeter - gap_set:
            grid[pr, pc] = border_color
        for sr, sc in seed_positions:
            grid[r0 + 1 + sr, c0 + 1 + sc] = rng.choice(seed_palette)

        for r, c in needed:
            reserved[r, c] = True
        return True
    return False


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    max_objects = 4 if min(h, w) >= 24 else 3
    object_count = rng.randint(2, max_objects)

    colors = list(PALETTE)
    rng.shuffle(colors)
    border_colors = colors[:object_count]
    branch_colors = colors[object_count : 2 * object_count]
    seed_palette = colors[2 * object_count :]
    if not seed_palette:
        return None

    grid = np.zeros((h, w), dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    pairs = list(zip(border_colors, branch_colors))
    rng.shuffle(pairs)
    corner = rng.choice(["tl", "tr", "bl", "br"])
    lr0, lc0, lr1, lc1 = _corner_legend_bounds(h, w, object_count, corner)
    for i, (border_color, branch_color) in enumerate(pairs):
        r = lr0 + i if corner in ("tl", "tr") else lr1 - i
        c = lc0
        grid[r, c] = border_color
        grid[r, c + 1] = branch_color
    for r, c in _expanded_cells(lr0, lc0, lr1, lc1, h, w, 2):
        reserved[r, c] = True

    modes = ["truncated", "full"] + [rng.choice(["truncated", "full"]) for _ in range(object_count - 2)]
    rng.shuffle(modes)
    for (border_color, _), mode in zip(pairs, modes):
        if not _try_place_object(rng, grid, reserved, border_color, seed_palette, mode):
            return None

    out = _solve(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        try:
            grid = _generate_attempt(rng)
        except AssertionError:
            continue
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
