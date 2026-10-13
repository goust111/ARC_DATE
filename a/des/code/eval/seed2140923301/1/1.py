from __future__ import annotations

import random

import numpy as np


SIDE_TO_VECTOR = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

SIDES = tuple(SIDE_TO_VECTOR)
COLOR_POOL = tuple(range(10))
MAX_GENERATION_ATTEMPTS = 300
MAX_OBJECT_PLACEMENT_ATTEMPTS = 200


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


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
                comp.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        if r1 != r0:
            cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        if c1 != c0:
            cells.append((r, c1))
    return cells


def _gap_side(gap: tuple[int, int], r0: int, c0: int, r1: int, c1: int) -> str | None:
    gr, gc = gap
    if gr == r0 and c0 < gc < c1:
        return "top"
    if gr == r1 and c0 < gc < c1:
        return "bottom"
    if gc == c0 and r0 < gr < r1:
        return "left"
    if gc == c1 and r0 < gr < r1:
        return "right"
    return None


def _seed_sort_key(seed: tuple[int, int, int], side: str) -> tuple[int, int]:
    r, c, _ = seed
    if side == "top":
        return int(r), int(c)
    if side == "bottom":
        return -int(r), int(c)
    if side == "left":
        return int(c), int(r)
    return -int(c), int(r)


def _find_fences(grid: np.ndarray) -> list[dict]:
    bg = _background_color(grid)
    fences: list[dict] = []
    for color in (int(v) for v in np.unique(grid) if int(v) != bg):
        for comp in _components(grid == color):
            r0, c0, r1, c1 = _bbox(comp)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue

            border = set(_perimeter_cells(r0, c0, r1, c1))
            comp_set = set(comp)
            if comp_set - border:
                continue
            if len(comp_set) != len(border) - 1:
                continue

            missing = list(border - comp_set)
            if len(missing) != 1:
                continue
            gap = missing[0]
            side = _gap_side(gap, r0, c0, r1, c1)
            if side is None:
                continue

            seeds: list[tuple[int, int, int]] = []
            valid = True
            for r in range(r0 + 1, r1):
                for c in range(c0 + 1, c1):
                    value = int(grid[r, c])
                    if value == bg:
                        continue
                    if value == color:
                        valid = False
                        break
                    seeds.append((int(r), int(c), value))
                if not valid:
                    break
            if not valid or not seeds:
                continue

            fences.append(
                {
                    "color": int(color),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "gap": (int(gap[0]), int(gap[1])),
                    "side": side,
                    "seeds": seeds,
                }
            )
    return fences


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find each hollow rectangular fence with one border gap. The gap side fixes both
    the outward growth direction and the spatial ordering used to read seed colors
    inside the fence. Draw an outward branch of matching length, leaving the fence
    and the seeds unchanged.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    out = g.copy()

    for fence in _find_fences(g):
        gap_r, gap_c = fence["gap"]
        dr, dc = SIDE_TO_VECTOR[fence["side"]]
        ordered = sorted(fence["seeds"], key=lambda seed: _seed_sort_key(seed, fence["side"]))
        for step, (_seed_r, _seed_c, color) in enumerate(ordered, start=1):
            rr = int(gap_r + dr * step)
            cc = int(gap_c + dc * step)
            if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]:
                out[rr, cc] = int(color)
    return out


def _reserve_box_for_object(
    r0: int,
    c0: int,
    inner_h: int,
    inner_w: int,
    seed_count: int,
    side: str,
) -> tuple[int, int, int, int]:
    r1 = r0 + inner_h + 1
    c1 = c0 + inner_w + 1
    if side == "top":
        return r0 - seed_count - 1, c0 - 1, r1 + 1, c1 + 1
    if side == "bottom":
        return r0 - 1, c0 - 1, r1 + seed_count + 1, c1 + 1
    if side == "left":
        return r0 - 1, c0 - seed_count - 1, r1 + 1, c1 + 1
    return r0 - 1, c0 - 1, r1 + 1, c1 + seed_count + 1


def _placement_ranges(
    height: int,
    width: int,
    inner_h: int,
    inner_w: int,
    seed_count: int,
    side: str,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    if side == "top":
        r_lo, r_hi = seed_count + 1, height - outer_h - 2
        c_lo, c_hi = 1, width - outer_w - 2
    elif side == "bottom":
        r_lo, r_hi = 1, height - outer_h - seed_count - 2
        c_lo, c_hi = 1, width - outer_w - 2
    elif side == "left":
        r_lo, r_hi = 1, height - outer_h - 2
        c_lo, c_hi = seed_count + 1, width - outer_w - 2
    else:
        r_lo, r_hi = 1, height - outer_h - 2
        c_lo, c_hi = 1, width - outer_w - seed_count - 2
    if r_lo > r_hi or c_lo > c_hi:
        return None
    return (int(r_lo), int(r_hi)), (int(c_lo), int(c_hi))


def _box_is_free(occupied: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    if r0 < 0 or c0 < 0 or r1 >= occupied.shape[0] or c1 >= occupied.shape[1]:
        return False
    return not bool(occupied[r0 : r1 + 1, c0 : c1 + 1].any())


def _mark_box(occupied: np.ndarray, box: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = box
    occupied[r0 : r1 + 1, c0 : c1 + 1] = True


def _render_expected(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for spec in specs:
        dr, dc = SIDE_TO_VECTOR[spec["side"]]
        gap_r, gap_c = spec["gap"]
        ordered = sorted(spec["seeds"], key=lambda seed: _seed_sort_key(seed, spec["side"]))
        for step, (_r, _c, color) in enumerate(ordered, start=1):
            out[gap_r + dr * step, gap_c + dc * step] = int(color)
    return out


def _random_object_count(height: int, width: int) -> int:
    if min(height, width) >= 24:
        return random.choice([2, 3, 3, 4])
    if min(height, width) >= 20:
        return random.choice([1, 2, 2, 3])
    return random.choice([1, 1, 2])


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(16, 30)
    width = random.randint(16, 30)
    bg = random.choice(COLOR_POOL)
    object_count = _random_object_count(height, width)
    frame_colors = random.sample([c for c in COLOR_POOL if c != bg], object_count)
    seed_palette = [c for c in COLOR_POOL if c != bg and c not in frame_colors]
    if len(seed_palette) < 2:
        return None

    grid = np.full((height, width), bg, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)
    specs: list[dict] = []

    for frame_color in frame_colors:
        placed = False
        for _ in range(MAX_OBJECT_PLACEMENT_ATTEMPTS):
            side = random.choice(SIDES)
            inner_h = random.randint(1, 5)
            inner_w = random.randint(1, 5)
            area = inner_h * inner_w
            seed_count = random.randint(1, min(area, len(seed_palette), 6))
            ranges = _placement_ranges(height, width, inner_h, inner_w, seed_count, side)
            if ranges is None:
                continue

            (r_lo, r_hi), (c_lo, c_hi) = ranges
            r0 = random.randint(r_lo, r_hi)
            c0 = random.randint(c_lo, c_hi)
            reserve_box = _reserve_box_for_object(r0, c0, inner_h, inner_w, seed_count, side)
            if not _box_is_free(occupied, reserve_box):
                continue

            r1 = r0 + inner_h + 1
            c1 = c0 + inner_w + 1
            if side in ("top", "bottom"):
                gap_c = c0 + random.randint(1, inner_w)
                gap_r = r0 if side == "top" else r1
            else:
                gap_r = r0 + random.randint(1, inner_h)
                gap_c = c0 if side == "left" else c1

            grid[r0, c0 : c1 + 1] = frame_color
            grid[r1, c0 : c1 + 1] = frame_color
            grid[r0 : r1 + 1, c0] = frame_color
            grid[r0 : r1 + 1, c1] = frame_color
            grid[gap_r, gap_c] = bg

            seed_positions = random.sample(
                [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)],
                seed_count,
            )
            seed_colors = random.sample(seed_palette, seed_count)
            seeds: list[tuple[int, int, int]] = []
            for (sr, sc), seed_color in zip(seed_positions, seed_colors):
                grid[sr, sc] = seed_color
                seeds.append((int(sr), int(sc), int(seed_color)))

            _mark_box(occupied, reserve_box)
            specs.append(
                {
                    "color": int(frame_color),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "gap": (int(gap_r), int(gap_c)),
                    "side": side,
                    "seeds": seeds,
                }
            )
            placed = True
            break
        if not placed:
            return None

    if len(_find_fences(grid)) != object_count:
        return None

    values, counts = np.unique(grid, return_counts=True)
    max_other = max(int(cnt) for val, cnt in zip(values, counts) if int(val) != bg)
    if int(np.sum(grid == bg)) <= max_other:
        return None

    expected = _render_expected(grid, specs)
    try:
        observed = main(grid)
    except Exception:
        return None
    if np.array_equal(grid, observed):
        return None
    if not np.array_equal(observed, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
