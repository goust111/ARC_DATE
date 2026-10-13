import random
from collections import deque

import numpy as np


DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
DEPTHS = (2, 3)
TAILS = (2, 3, 4)
GEN_RETRIES = 500


def _background(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIRS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _arrow_cells(
    tip: tuple[int, int],
    direction: tuple[int, int],
    depth: int,
    tail_len: int,
) -> set[tuple[int, int]]:
    tr, tc = tip
    dr, dc = direction
    br, bc = -dr, -dc
    pr, pc = (0, 1) if dr else (1, 0)

    cells: set[tuple[int, int]] = set()
    for layer in range(depth):
        cr = tr + br * layer
        cc = tc + bc * layer
        for offset in range(-layer, layer + 1):
            cells.add((int(cr + pr * offset), int(cc + pc * offset)))

    base_r = tr + br * (depth - 1)
    base_c = tc + bc * (depth - 1)
    for step in range(1, tail_len + 1):
        cells.add((int(base_r + br * step), int(base_c + bc * step)))
    return cells


def _in_bounds(cells: set[tuple[int, int]], h: int, w: int) -> bool:
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _match_arrow(comp: list[tuple[int, int]], color: int) -> dict:
    comp_set = set(comp)
    matches: list[dict] = []
    for tip in comp:
        for direction in DIRS:
            for depth in DEPTHS:
                for tail_len in TAILS:
                    cells = _arrow_cells(tip, direction, depth, tail_len)
                    if cells == comp_set:
                        matches.append(
                            {
                                "tip": (int(tip[0]), int(tip[1])),
                                "dir": (int(direction[0]), int(direction[1])),
                                "depth": int(depth),
                                "tail": int(tail_len),
                                "color": int(color),
                                "cells": sorted(cells),
                            }
                        )
    assert len(matches) == 1, f"expected one arrow interpretation, got {len(matches)}"
    return matches[0]


def _extract_arrows(grid: np.ndarray) -> list[dict]:
    bg = _background(grid)
    arrows: list[dict] = []
    for comp in _components(grid != bg):
        colors = {int(grid[r, c]) for r, c in comp}
        assert len(colors) == 1, "each arrow must be a single visible color"
        arrows.append(_match_arrow(comp, colors.pop()))
    assert 3 <= len(arrows) <= 5, "expected three to five arrows"
    return arrows


def _owner_grid(arrows: list[dict], shape: tuple[int, int]) -> np.ndarray:
    owner = np.full(shape, -1, dtype=int)
    for i, arrow in enumerate(arrows):
        for r, c in arrow["cells"]:
            owner[int(r), int(c)] = int(i)
    return owner


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire colored rays from arrows and recolor the first arrow each ray hits.

    Input structure:
    - The most common color is the background.
    - Three to five separated foreground components are solid one-color arrows.
    - Each arrow has a triangular head of depth two or three pointing up, down,
      left, or right, and a one-pixel-wide tail of length two to four extending
      backward from the head base center.

    Rule:
    - Identify each arrow's tip, direction, head depth, and tail length.
    - Process arrows by tail length from longest to shortest; ties are broken by
      the tip pixel in row-major order.
    - From the current tip, draw a ray in the arrow's pointing direction using
      the arrow's current color.
    - A ray stops at the grid boundary, or just before the first pixel belonging
      to another arrow. When another arrow is hit, all of that arrow's pixels
      are recolored to the firing arrow's current color.

    Key insight:
    - The visible tail length is the priority order; colors only travel through
      tip rays and through whole-arrow recoloring on hits.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)

    arrows = _extract_arrows(g)
    owner = _owner_grid(arrows, (h, w))
    current_colors = [int(a["color"]) for a in arrows]
    order = sorted(range(len(arrows)), key=lambda i: (-int(arrows[i]["tail"]), arrows[i]["tip"][0], arrows[i]["tip"][1]))

    out = g.copy()
    for i in order:
        arrow = arrows[int(i)]
        color = int(current_colors[int(i)])
        dr, dc = arrow["dir"]
        r = int(arrow["tip"][0] + dr)
        c = int(arrow["tip"][1] + dc)
        while 0 <= r < h and 0 <= c < w:
            hit = int(owner[r, c])
            if hit >= 0 and hit != int(i):
                current_colors[hit] = color
                for rr, cc in arrows[hit]["cells"]:
                    out[int(rr), int(cc)] = color
                break
            out[int(r), int(c)] = color
            r += int(dr)
            c += int(dc)

    return out


def _touches_occupied(cells: set[tuple[int, int]], occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = int(r + dr), int(c + dc)
                if 0 <= rr < h and 0 <= cc < w and bool(occupied[rr, cc]):
                    return True
    return False


def _candidate_arrows(
    h: int,
    w: int,
    occupied: np.ndarray,
    directions: tuple[tuple[int, int], ...] = DIRS,
    depths: tuple[int, ...] = DEPTHS,
    tails: tuple[int, ...] = TAILS,
) -> list[dict]:
    candidates: list[dict] = []
    for direction in directions:
        for depth in depths:
            for tail_len in tails:
                for r in range(h):
                    for c in range(w):
                        cells = _arrow_cells((r, c), direction, depth, tail_len)
                        if not _in_bounds(cells, h, w):
                            continue
                        if _touches_occupied(cells, occupied):
                            continue
                        candidates.append(
                            {
                                "tip": (int(r), int(c)),
                                "dir": (int(direction[0]), int(direction[1])),
                                "depth": int(depth),
                                "tail": int(tail_len),
                                "cells": cells,
                            }
                        )
    random.shuffle(candidates)
    return candidates


def _ray_distances(tip: tuple[int, int], direction: tuple[int, int], h: int, w: int) -> dict[tuple[int, int], int]:
    r, c = int(tip[0] + direction[0]), int(tip[1] + direction[1])
    dist = 1
    out: dict[tuple[int, int], int] = {}
    while 0 <= r < h and 0 <= c < w:
        out[(int(r), int(c))] = int(dist)
        r += int(direction[0])
        c += int(direction[1])
        dist += 1
    return out


def _mark_occupied(occupied: np.ndarray, cells: set[tuple[int, int]]) -> None:
    for r, c in cells:
        occupied[int(r), int(c)] = True


def _paint_specs(specs: list[dict], colors: list[int], bg: int, h: int, w: int) -> np.ndarray:
    grid = np.full((h, w), int(bg), dtype=int)
    for spec, color in zip(specs, colors):
        for r, c in spec["cells"]:
            grid[int(r), int(c)] = int(color)
    return grid


def _ray_outcome_counts(grid: np.ndarray) -> tuple[int, int]:
    arrows = _extract_arrows(grid)
    owner = _owner_grid(arrows, grid.shape)
    h, w = map(int, grid.shape)
    hits = 0
    boundary = 0
    order = sorted(range(len(arrows)), key=lambda i: (-int(arrows[i]["tail"]), arrows[i]["tip"][0], arrows[i]["tip"][1]))
    for i in order:
        dr, dc = arrows[int(i)]["dir"]
        r = int(arrows[int(i)]["tip"][0] + dr)
        c = int(arrows[int(i)]["tip"][1] + dc)
        while 0 <= r < h and 0 <= c < w:
            hit = int(owner[r, c])
            if hit >= 0 and hit != int(i):
                hits += 1
                break
            r += int(dr)
            c += int(dc)
        else:
            boundary += 1
    return int(hits), int(boundary)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(16, 30)
    w = random.randint(16, 30)
    n_arrows = random.randint(3, 5)
    bg, *colors = random.sample(range(10), n_arrows + 1)
    occupied = np.zeros((h, w), dtype=bool)
    specs: list[dict] = []

    first_dir = random.choice(DIRS)
    first_candidates = [
        spec
        for spec in _candidate_arrows(h, w, occupied, directions=(first_dir,), tails=(4,))
        if len(_ray_distances(spec["tip"], spec["dir"], h, w)) >= 5
    ]
    if not first_candidates:
        return None
    first = random.choice(first_candidates)
    specs.append(first)
    _mark_occupied(occupied, first["cells"])

    ray = _ray_distances(first["tip"], first["dir"], h, w)
    second_tails = (2, 3)
    target_candidates = []
    for spec in _candidate_arrows(h, w, occupied, tails=second_tails):
        hit_dists = [ray[cell] for cell in spec["cells"] if cell in ray]
        if hit_dists and min(hit_dists) >= 2:
            target_candidates.append(spec)
    if not target_candidates:
        return None
    target = random.choice(target_candidates)
    specs.append(target)
    _mark_occupied(occupied, target["cells"])

    while len(specs) < n_arrows:
        tails = TAILS
        if len(specs) == 2 and all(int(spec["tail"]) != 2 for spec in specs):
            tails = (2,)
        candidates = _candidate_arrows(h, w, occupied, tails=tails)
        if not candidates:
            return None
        spec = random.choice(candidates)
        specs.append(spec)
        _mark_occupied(occupied, spec["cells"])

    grid = _paint_specs(specs, colors, bg, h, w)
    try:
        output = main(grid)
        hits, boundary = _ray_outcome_counts(grid)
    except AssertionError:
        return None

    if output.shape != grid.shape or max(output.shape) > 30:
        return None
    if hits < 1 or boundary < 1:
        return None
    if len({int(spec["tail"]) for spec in specs}) < 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
