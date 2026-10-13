from __future__ import annotations

import random
from collections import Counter

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
PALETTE = tuple(range(1, 10))
GEN_ATTEMPTS = 500


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIR4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _find_mosaic_box(grid: np.ndarray) -> tuple[int, int, int, int]:
    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for coords in _components(grid != 0):
        box = _bbox(coords)
        r0, c0, r1, c1 = box
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if area != int(coords.shape[0]):
            continue
        if not np.all(grid[r0 : r1 + 1, c0 : c1 + 1] != 0):
            continue
        candidates.append((area, box))
    assert candidates, "expected a filled mosaic rectangle"
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][1]


def _detect_instructions(
    grid: np.ndarray, mosaic_box: tuple[int, int, int, int]
) -> tuple[dict[int, str], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    r0, c0, r1, c1 = mosaic_box
    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = 0

    used = np.zeros((h, w), dtype=bool)
    by_color: dict[int, set[str]] = {}

    for color in map(int, np.unique(outside)):
        if color == 0:
            continue
        for coords in _components(outside == color):
            if int(coords.shape[0]) != 4:
                continue
            br0, bc0, br1, bc1 = _bbox(coords)
            if br1 - br0 != 1 or bc1 - bc0 != 1:
                continue
            if np.all(outside[br0 : br1 + 1, bc0 : bc1 + 1] == color):
                by_color.setdefault(color, set()).add("expand")
                used[coords[:, 0], coords[:, 1]] = True

    for r in range(h - 2):
        for c in range(w - 2):
            patch = outside[r : r + 3, c : c + 3]
            color = int(patch[0, 0])
            if color == 0:
                continue
            if (
                int(patch[0, 2]) == color
                and int(patch[2, 0]) == color
                and int(patch[2, 2]) == color
                and int(np.count_nonzero(patch)) == 4
            ):
                by_color.setdefault(color, set()).add("shrink")
                used[r, c] = True
                used[r, c + 2] = True
                used[r + 2, c] = True
                used[r + 2, c + 2] = True

    leftover = (outside != 0) & (~used)
    assert not np.any(leftover), "expected all non-mosaic cells to be valid instruction objects"

    instructions: dict[int, str] = {}
    for color, ops in by_color.items():
        assert len(ops) == 1, "expected at most one instruction type per color"
        instructions[int(color)] = next(iter(ops))
    assert instructions, "expected at least one instruction object"
    return instructions, used


def _unique_mode(values: list[int]) -> int | None:
    if not values:
        return None
    counts = Counter(values)
    best = max(counts.values())
    winners = [color for color, count in counts.items() if count == best]
    if len(winners) != 1:
        return None
    return int(winners[0])


def _transform(
    grid: np.ndarray,
    mosaic_box: tuple[int, int, int, int],
    instructions: dict[int, str],
    instruction_mask: np.ndarray,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    r0, c0, r1, c1 = mosaic_box

    proposals: dict[tuple[int, int], set[int]] = {}
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            color = int(g[r, c])
            op = instructions.get(color)
            if op is None:
                continue

            non_self = [
                int(g[nr, nc])
                for nr, nc in _neighbors4(r, c, h, w)
                if int(g[nr, nc]) != color
            ]
            if not non_self:
                continue

            if op == "expand":
                for nr, nc in _neighbors4(r, c, h, w):
                    if int(g[nr, nc]) == color:
                        continue
                    proposals.setdefault((nr, nc), set()).add(color)
            else:
                target = _unique_mode(non_self)
                if target is not None:
                    proposals.setdefault((r, c), set()).add(target)

    out = g.copy()
    for (r, c), colors in proposals.items():
        if len(colors) == 1:
            out[r, c] = next(iter(colors))
    out[instruction_mask] = 0
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode outside same-color markers as expand/shrink instructions for the matching mosaic color.

    Input structure:
    - Black background.
    - One large filled rectangular mosaic made of several non-black colors.
    - Outside the mosaic are isolated same-color instruction objects:
      a solid 2x2 block means expand, and four corners of a 3x3 box mean shrink.

    Rule:
    - Match each instruction to the same color inside the mosaic.
    - Expand colors claim every 4-neighbor of their original mosaic pixels.
    - Shrink colors replace each edge pixel by the unique most frequent non-self 4-neighbor color.
    - All proposals are computed from the original grid simultaneously.
    - If a cell receives multiple different color proposals, keep its original value.
    - Erase all outside instruction objects to black.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    mosaic_box = _find_mosaic_box(g)
    instructions, instruction_mask = _detect_instructions(g, mosaic_box)
    return _transform(g, mosaic_box, instructions, instruction_mask)


def _frontier_cells(owner: np.ndarray, region_cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    h, w = owner.shape
    frontier: set[tuple[int, int]] = set()
    for r, c in region_cells:
        for nr, nc in _neighbors4(r, c, h, w):
            if int(owner[nr, nc]) == -1:
                frontier.add((nr, nc))
    return list(frontier)


def _grow_connected_partition(
    coarse_h: int, coarse_w: int, colors: list[int], rng: random.Random
) -> np.ndarray | None:
    all_cells = [(r, c) for r in range(coarse_h) for c in range(coarse_w)]
    min_region = max(2, (coarse_h * coarse_w) // (4 * len(colors)))

    for _ in range(80):
        seeds = rng.sample(all_cells, len(colors))
        owner = np.full((coarse_h, coarse_w), -1, dtype=int)
        regions = [set() for _ in colors]
        sizes = [0 for _ in colors]
        for idx, (r, c) in enumerate(seeds):
            owner[r, c] = idx
            regions[idx].add((r, c))
            sizes[idx] = 1

        remaining = coarse_h * coarse_w - len(colors)
        while remaining > 0:
            expandable: list[tuple[int, list[tuple[int, int]]]] = []
            for idx, cells in enumerate(regions):
                frontier = _frontier_cells(owner, cells)
                if frontier:
                    expandable.append((idx, frontier))
            if not expandable:
                break

            min_size = min(sizes[idx] for idx, _ in expandable)
            pool = [(idx, frontier) for idx, frontier in expandable if sizes[idx] <= min_size + 1]
            idx, frontier = rng.choice(pool)
            nr, nc = rng.choice(frontier)
            owner[nr, nc] = idx
            regions[idx].add((nr, nc))
            sizes[idx] += 1
            remaining -= 1

        if remaining != 0 or min(sizes) < min_region:
            continue

        coarse = np.empty((coarse_h, coarse_w), dtype=int)
        for idx, color in enumerate(colors):
            coarse[owner == idx] = color
        return coarse
    return None


def _choose_instruction_map(colors: list[int], rng: random.Random) -> dict[int, str]:
    if len(colors) == 1:
        selected = colors[:]
    else:
        lo = 2 if len(colors) >= 3 and rng.random() < 0.7 else 1
        selected = rng.sample(colors, rng.randint(lo, len(colors)))

    instructions = {
        color: ("expand" if rng.random() < 0.5 else "shrink")
        for color in selected
    }
    if len(instructions) > 1 and len(set(instructions.values())) == 1 and rng.random() < 0.8:
        flip = rng.choice(selected)
        instructions[flip] = "shrink" if instructions[flip] == "expand" else "expand"
    return instructions


def _instruction_shape(op: str) -> tuple[int, tuple[tuple[int, int], ...]]:
    if op == "expand":
        return 2, ((0, 0), (0, 1), (1, 0), (1, 1))
    return 3, ((0, 0), (0, 2), (2, 0), (2, 2))


def _reserve(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    h, w = mask.shape
    rr0 = max(0, r0)
    cc0 = max(0, c0)
    rr1 = min(h - 1, r1)
    cc1 = min(w - 1, c1)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _place_instruction(
    grid: np.ndarray,
    reserved: np.ndarray,
    color: int,
    op: str,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    h, w = grid.shape
    size, shape = _instruction_shape(op)
    for _ in range(200):
        top = rng.randrange(h - size + 1)
        left = rng.randrange(w - size + 1)
        if np.any(reserved[max(0, top - 1) : min(h, top + size + 1), max(0, left - 1) : min(w, left + size + 1)]):
            continue
        coords = [(top + dr, left + dc) for dr, dc in shape]
        if any(int(grid[r, c]) != 0 for r, c in coords):
            continue
        for r, c in coords:
            grid[r, c] = color
        _reserve(reserved, top - 1, left - 1, top + size, left + size)
        return coords
    return None


def _gen_attempt(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    coarse_h = rng.randint(4, 5)
    coarse_w = rng.randint(4, 5)
    scale_r = rng.randint(2, 3)
    scale_c = rng.randint(2, 3)

    max_colors = min(5, coarse_h * coarse_w // 2)
    color_count = rng.randint(3, max_colors)
    colors = rng.sample(list(PALETTE), color_count)

    coarse = _grow_connected_partition(coarse_h, coarse_w, colors, rng)
    if coarse is None:
        return None

    mosaic = np.repeat(np.repeat(coarse, scale_r, axis=0), scale_c, axis=1)
    mosaic_h, mosaic_w = map(int, mosaic.shape)

    instructions = _choose_instruction_map(colors, rng)

    total_h = mosaic_h + rng.randint(10, 14)
    total_w = mosaic_w + rng.randint(10, 14)
    top = rng.randint(4, total_h - mosaic_h - 4)
    left = rng.randint(4, total_w - mosaic_w - 4)

    grid = np.zeros((total_h, total_w), dtype=int)
    grid[top : top + mosaic_h, left : left + mosaic_w] = mosaic

    reserved = np.zeros_like(grid, dtype=bool)
    _reserve(reserved, top - 2, left - 2, top + mosaic_h + 1, left + mosaic_w + 1)

    instruction_mask = np.zeros_like(grid, dtype=bool)
    order = list(instructions.items())
    rng.shuffle(order)
    for color, op in order:
        coords = _place_instruction(grid, reserved, color, op, rng)
        if coords is None:
            return None
        for r, c in coords:
            instruction_mask[r, c] = True

    return grid, instruction_mask


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        attempt = _gen_attempt(rng)
        if attempt is None:
            continue
        grid, instruction_mask = attempt
        try:
            out = main(grid)
        except Exception:
            continue
        if not np.any((out != grid) & (~instruction_mask)):
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
