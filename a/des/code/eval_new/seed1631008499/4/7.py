import random
from collections import deque

import numpy as np


SIDES = ("top", "bottom", "left", "right")
PALETTE = tuple(range(1, 10))
GEN_RETRIES = 400


def _dominant_color(grid: np.ndarray) -> int:
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
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _full_border(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    border: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        border.append((r0, c))
        if r1 != r0:
            border.append((r1, c))
    for r in range(r0 + 1, r1):
        border.append((r, c0))
        if c1 != c0:
            border.append((r, c1))
    return border


def _classify_gap(missing: list[tuple[int, int]], r0: int, c0: int, r1: int, c1: int):
    if not (1 <= len(missing) <= 3):
        return None
    if all(r == r0 for r, _ in missing):
        ordered = sorted(missing, key=lambda p: p[1])
        if all(ordered[i + 1][1] == ordered[i][1] + 1 for i in range(len(ordered) - 1)):
            return "top", ordered
    if all(r == r1 for r, _ in missing):
        ordered = sorted(missing, key=lambda p: p[1])
        if all(ordered[i + 1][1] == ordered[i][1] + 1 for i in range(len(ordered) - 1)):
            return "bottom", ordered
    if all(c == c0 for _, c in missing):
        ordered = sorted(missing, key=lambda p: p[0])
        if all(ordered[i + 1][0] == ordered[i][0] + 1 for i in range(len(ordered) - 1)):
            return "left", ordered
    if all(c == c1 for _, c in missing):
        ordered = sorted(missing, key=lambda p: p[0])
        if all(ordered[i + 1][0] == ordered[i][0] + 1 for i in range(len(ordered) - 1)):
            return "right", ordered
    return None


def _extract_fences(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    fences = []
    for color in sorted(int(c) for c in np.unique(g) if int(c) != bg):
        for comp in _components(g == color):
            r0, c0, r1, c1 = _bbox(comp)
            if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
                continue
            comp_set = set(comp)
            border = _full_border(r0, c0, r1, c1)
            border_set = set(border)
            if not comp_set <= border_set:
                continue
            missing = list(border_set - comp_set)
            gap_info = _classify_gap(missing, r0, c0, r1, c1)
            if gap_info is None:
                continue
            side, gap_cells = gap_info

            if r1 - r0 <= 1 or c1 - c0 <= 1:
                continue
            interior = g[r0 + 1 : r1, c0 + 1 : c1]
            interior_colors = [int(v) for v in interior.ravel() if int(v) != bg]
            if not interior_colors:
                continue
            seed_colors = set(interior_colors)
            if color in seed_colors or len(seed_colors) != 1:
                continue
            seed_color = next(iter(seed_colors))
            seed_count = len(interior_colors)
            fences.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "fence_color": color,
                    "gap_side": side,
                    "gap_cells": gap_cells,
                    "seed_color": seed_color,
                    "seed_count": seed_count,
                }
            )
    fences.sort(key=lambda item: item["bbox"])
    return fences


def _ray_offsets(count: int, width: int) -> list[int]:
    if count <= width:
        return list(range(count))
    if width == 1:
        return [0] * count
    return [int(round(i * (width - 1) / (count - 1))) for i in range(count)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find gapped rectangular fences and turn their interior seed counts into outward rays.

    Input structure:
    - The dominant color is the background.
    - One or more single-color hollow rectangular fences appear as 1-pixel-wide
      rectangular borders, each with exactly one contiguous gap of width 1-3 on
      one side.
    - Each fence encloses one seed color distinct from the fence color; all seed
      pixels inside the same fence share that color.

    Rule:
    - Detect each fence's bounding rectangle and identify which side contains the
      missing border segment.
    - Count the number of seed pixels N inside that fence.
    - Starting from positions along the gap, draw N parallel rays outward away
      from the interior using the seed color, extending to the grid edge.
    - If the gap width is W and N <= W, use the first N positions from the
      gap's left/top end; if N > W, distribute the N rays evenly across the W
      gap positions, allowing multiple rays to share one position.

    Key insight:
    - The fence encodes two pieces of information at once: the missing side
      gives the ray direction, and the number of interior seed pixels gives the
      number of rays to emit.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    for fence in _extract_fences(g):
        side = fence["gap_side"]
        gap_cells = fence["gap_cells"]
        seed_color = int(fence["seed_color"])
        offsets = _ray_offsets(int(fence["seed_count"]), len(gap_cells))
        for idx in offsets:
            gr, gc = gap_cells[idx]
            if side == "top":
                out[: gr + 1, gc] = seed_color
            elif side == "bottom":
                out[gr:h, gc] = seed_color
            elif side == "left":
                out[gr, : gc + 1] = seed_color
            else:
                out[gr, gc:w] = seed_color

    return out


def _gap_cells(r0: int, c0: int, outer_h: int, outer_w: int, side: str, start: int, width: int):
    if side == "top":
        return [(r0, c0 + start + i) for i in range(width)]
    if side == "bottom":
        return [(r0 + outer_h - 1, c0 + start + i) for i in range(width)]
    if side == "left":
        return [(r0 + start + i, c0) for i in range(width)]
    return [(r0 + start + i, c0 + outer_w - 1) for i in range(width)]


def _fence_cells(r0: int, c0: int, outer_h: int, outer_w: int, gap: set[tuple[int, int]]):
    cells: list[tuple[int, int]] = []
    for rr in range(r0, r0 + outer_h):
        for cc in range(c0, c0 + outer_w):
            on_border = rr in (r0, r0 + outer_h - 1) or cc in (c0, c0 + outer_w - 1)
            if on_border and (rr, cc) not in gap:
                cells.append((rr, cc))
    return cells


def _corridor_cells(h: int, w: int, side: str, gap_cells: list[tuple[int, int]]):
    cells: set[tuple[int, int]] = set()
    for gr, gc in gap_cells:
        if side == "top":
            for rr in range(0, gr + 1):
                cells.add((rr, gc))
        elif side == "bottom":
            for rr in range(gr, h):
                cells.add((rr, gc))
        elif side == "left":
            for cc in range(0, gc + 1):
                cells.add((gr, cc))
        else:
            for cc in range(gc, w):
                cells.add((gr, cc))
    return cells


def _expanded_box(h: int, w: int, r0: int, c0: int, r1: int, c1: int, pad: int = 1):
    cells: set[tuple[int, int]] = set()
    for rr in range(max(0, r0 - pad), min(h - 1, r1 + pad) + 1):
        for cc in range(max(0, c0 - pad), min(w - 1, c1 + pad) + 1):
            cells.add((rr, cc))
    return cells


def _sample_seed_count(rng: random.Random, gap_w: int, interior_area: int, want_more: bool) -> int | None:
    if want_more:
        if interior_area < gap_w + 1:
            return None
        return rng.randint(gap_w + 1, min(interior_area, gap_w + 4))
    return rng.randint(1, min(gap_w, interior_area))


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    n_fences = rng.randint(2, 4)
    min_size = 10 + 3 * n_fences
    h = rng.randint(min_size, 30)
    w = rng.randint(min_size, 30)
    grid = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    colors = list(PALETTE)
    rng.shuffle(colors)
    sides = rng.sample(list(SIDES), k=n_fences)
    more_flags = [True, False]
    while len(more_flags) < n_fences:
        more_flags.append(rng.choice([True, False]))
    rng.shuffle(more_flags)

    placed = 0
    for idx in range(n_fences):
        fence_color = colors[2 * idx]
        seed_color = colors[2 * idx + 1]
        side = sides[idx]
        want_more = more_flags[idx]
        success = False
        for _ in range(250):
            outer_h = rng.randint(5, min(9, h - 2))
            outer_w = rng.randint(5, min(9, w - 2))
            side_len = outer_w if side in ("top", "bottom") else outer_h
            gap_w = rng.randint(1, min(3, side_len - 2))
            gap_start = rng.randint(1, side_len - gap_w - 1)
            seed_count = _sample_seed_count(rng, gap_w, (outer_h - 2) * (outer_w - 2), want_more)
            if seed_count is None:
                continue

            max_r0 = h - outer_h - 1
            max_c0 = w - outer_w - 1
            if max_r0 < 1 or max_c0 < 1:
                continue
            r0 = rng.randint(1, max_r0)
            c0 = rng.randint(1, max_c0)
            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1

            gap_cells = _gap_cells(r0, c0, outer_h, outer_w, side, gap_start, gap_w)
            gap_set = set(gap_cells)
            fence_cells = _fence_cells(r0, c0, outer_h, outer_w, gap_set)
            corridor = _corridor_cells(h, w, side, gap_cells)
            reserve = _expanded_box(h, w, r0, c0, r1, c1, pad=1) | corridor

            if any(blocked[rr, cc] for rr, cc in reserve):
                continue

            interior_positions = [
                (rr, cc)
                for rr in range(r0 + 1, r1)
                for cc in range(c0 + 1, c1)
            ]
            if len(interior_positions) < seed_count:
                continue
            seed_cells = rng.sample(interior_positions, seed_count)

            for rr, cc in fence_cells:
                grid[rr, cc] = fence_color
            for rr, cc in seed_cells:
                grid[rr, cc] = seed_color
            for rr, cc in reserve:
                blocked[rr, cc] = True

            placed += 1
            success = True
            break
        if not success:
            return None

    if placed != n_fences:
        return None

    if _dominant_color(grid) != 0:
        return None

    fences = _extract_fences(grid)
    if len(fences) != n_fences:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fence-ray input")
