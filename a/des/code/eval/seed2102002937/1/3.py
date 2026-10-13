from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


BG = 0
TRACK_BASE = 5
MAROON = 9
TRACK_COLORS = [1, 2, 3, 4, 6, 7, 8]
ARROW_SIDES = ("up", "right", "down", "left")
BLOCK_SHAPES = [
    (1, 1),
    (1, 2),
    (2, 1),
    (1, 3),
    (3, 1),
    (2, 2),
    (1, 4),
    (4, 1),
    (2, 3),
    (3, 2),
    (2, 4),
    (4, 2),
]
GEN_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    counts = Counter(int(v) for v in np.asarray(grid, dtype=int).ravel())
    return int(counts.most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)

    return comps


def _bbox(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _perimeter_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
    for r in range(r0 + 1, r1 + 1):
        coords.append((r, c1))
    for c in range(c1 - 1, c0 - 1, -1):
        coords.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        coords.append((r, c0))
    return coords


def _find_ring(grid: np.ndarray, bg: int) -> dict:
    mask = (grid != bg) & (grid != MAROON)
    candidates: list[dict] = []

    for comp in _components(mask):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        order = _perimeter_coords(r0, c0, r1, c1)
        if set(comp) != set(order):
            continue
        counts = Counter(int(grid[r, c]) for r, c in order)
        base_color = int(counts.most_common(1)[0][0])
        candidates.append(
            {
                "bbox": (r0, c0, r1, c1),
                "order": order,
                "coords": set(order),
                "base_color": base_color,
            }
        )

    assert len(candidates) == 1, f"expected one ring, found {len(candidates)}"
    return candidates[0]


def _full_rectangle(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    return len(coords) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _find_instruction_blocks(grid: np.ndarray, bg: int, ring_coords: set[tuple[int, int]]) -> dict[int, dict]:
    mask = (grid != bg) & (grid != MAROON)
    blocks: dict[int, dict] = {}

    for comp in _components(mask):
        comp_set = set(comp)
        if comp_set == ring_coords:
            continue
        assert _full_rectangle(comp), f"instruction block must be a solid rectangle: {comp}"
        colors = {int(grid[r, c]) for r, c in comp}
        assert len(colors) == 1, colors
        color = next(iter(colors))
        assert color not in blocks, f"duplicate block color {color}"

        r0, c0, r1, c1 = _bbox(comp)
        adj_maroon: set[tuple[int, int]] = set()
        for rr, cc in comp:
            for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and int(grid[nr, nc]) == MAROON:
                    adj_maroon.add((int(nr), int(nc)))
        assert len(adj_maroon) == 1, f"expected one maroon arrow for color {color}"
        arrow_r, arrow_c = next(iter(adj_maroon))

        if arrow_c == c0 - 1 and r0 <= arrow_r <= r1:
            side = "left"
        elif arrow_c == c1 + 1 and r0 <= arrow_r <= r1:
            side = "right"
        elif arrow_r == r0 - 1 and c0 <= arrow_c <= c1:
            side = "up"
        elif arrow_r == r1 + 1 and c0 <= arrow_c <= c1:
            side = "down"
        else:
            raise AssertionError(f"maroon arrow is not side-adjacent for color {color}")

        blocks[color] = {
            "step": len(comp),
            "clockwise": side in {"right", "down"},
        }

    return blocks


def _apply_track_shift(
    grid: np.ndarray,
    order: list[tuple[int, int]],
    base_color: int,
    track_positions: dict[int, list[int]],
    instructions: dict[int, dict],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for r, c in order:
        out[r, c] = base_color

    first_indices = {color: min(idxs) for color, idxs in track_positions.items()}
    for _, color in sorted((idx, color) for color, idx in first_indices.items()):
        shift = instructions[color]["step"]
        if not instructions[color]["clockwise"]:
            shift = -shift
        for idx in track_positions[color]:
            rr, cc = order[(idx + shift) % len(order)]
            out[rr, cc] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move colored track pixels around a rectangular loop according to external block-and-arrow instructions.

    The ring is the unique single-cell-wide rectangular perimeter component.
    Each non-base track color has one same-colored solid rectangle outside the ring.
    The block area gives the step count, and the side-adjacent maroon pixel gives the direction:
    right/down = clockwise, left/up = counterclockwise.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    ring = _find_ring(g, bg)
    order = ring["order"]
    base_color = int(ring["base_color"])

    track_positions: dict[int, list[int]] = defaultdict(list)
    for idx, (r, c) in enumerate(order):
        color = int(g[r, c])
        if color != base_color:
            track_positions[color].append(idx)

    if not track_positions:
        return g.copy()

    instructions = _find_instruction_blocks(g, bg, ring["coords"])
    assert set(track_positions) == set(instructions), "track colors and instruction colors must match"
    return _apply_track_shift(g, order, base_color, track_positions, instructions)


def _mark_cells(blocked: np.ndarray, cells: list[tuple[int, int]], pad: int = 1) -> None:
    h, w = blocked.shape
    for r, c in cells:
        rr0 = max(0, int(r) - pad)
        cc0 = max(0, int(c) - pad)
        rr1 = min(h, int(r) + pad + 1)
        cc1 = min(w, int(c) + pad + 1)
        blocked[rr0:rr1, cc0:cc1] = True


def _place_instruction_group(
    grid: np.ndarray,
    blocked: np.ndarray,
    color: int,
    block_h: int,
    block_w: int,
    side: str,
    rng: random.Random,
) -> bool:
    h, w = grid.shape

    for _ in range(250):
        r0 = rng.randint(0, h - block_h)
        c0 = rng.randint(0, w - block_w)
        block_cells = [(r0 + dr, c0 + dc) for dr in range(block_h) for dc in range(block_w)]
        if side == "up":
            arrow = (r0 - 1, c0 + rng.randrange(block_w))
        elif side == "down":
            arrow = (r0 + block_h, c0 + rng.randrange(block_w))
        elif side == "left":
            arrow = (r0 + rng.randrange(block_h), c0 - 1)
        else:
            arrow = (r0 + rng.randrange(block_h), c0 + block_w)

        cells = block_cells + [arrow]
        if not all(0 <= rr < h and 0 <= cc < w for rr, cc in cells):
            continue
        if any(bool(blocked[rr, cc]) for rr, cc in cells):
            continue

        for rr, cc in block_cells:
            grid[rr, cc] = color
        grid[arrow] = MAROON
        _mark_cells(blocked, cells, pad=1)
        return True

    return False


def _distribute_counts(n_colors: int, ring_len: int, rng: random.Random) -> list[int]:
    max_total = min(ring_len // 2 - 1, 12)
    max_total = max(max_total, n_colors)
    max_per_color = max(1, min(6, ring_len // 3))
    max_total = min(max_total, n_colors * max_per_color)
    target_total = rng.randint(n_colors, max_total)
    counts = [1] * n_colors

    for _ in range(target_total - n_colors):
        choices = [i for i, v in enumerate(counts) if v < max_per_color]
        counts[rng.choice(choices)] += 1

    rng.shuffle(counts)
    return counts


def _allocate_ring_indices(
    colors: list[int],
    counts: dict[int, int],
    shifts: dict[int, int],
    ring_len: int,
    rng: random.Random,
) -> dict[int, list[int]] | None:
    indices_by_color = {color: [] for color in colors}
    used: set[int] = set()

    collision_candidates = [(a, b) for a in colors for b in colors if a != b and shifts[a] != shifts[b]]
    force_collision = bool(collision_candidates) and rng.random() < 0.5
    if force_collision:
        color_a, color_b = rng.choice(collision_candidates)
        for _ in range(50):
            idx_a = rng.randrange(ring_len)
            target = (idx_a + shifts[color_a]) % ring_len
            idx_b = (target - shifts[color_b]) % ring_len
            if idx_a == idx_b or idx_a in used or idx_b in used:
                continue
            used.add(idx_a)
            used.add(idx_b)
            indices_by_color[color_a].append(idx_a)
            indices_by_color[color_b].append(idx_b)
            break
        else:
            force_collision = False

    remaining = [idx for idx in range(ring_len) if idx not in used]
    rng.shuffle(remaining)
    cursor = 0
    for color in colors:
        need = counts[color] - len(indices_by_color[color])
        if cursor + need > len(remaining):
            return None
        indices_by_color[color].extend(remaining[cursor : cursor + need])
        cursor += need
        indices_by_color[color].sort()

    return indices_by_color


def _expected_output(
    grid: np.ndarray,
    order: list[tuple[int, int]],
    base_color: int,
    indices_by_color: dict[int, list[int]],
    shifts: dict[int, int],
) -> np.ndarray:
    instructions = {
        color: {
            "step": abs(shift),
            "clockwise": shift >= 0,
        }
        for color, shift in shifts.items()
    }
    return _apply_track_shift(grid, order, base_color, indices_by_color, instructions)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    ring_h = rng.randint(6, 12)
    ring_w = rng.randint(6, 12)
    top_margin = rng.randint(5, 8)
    bottom_margin = rng.randint(5, 8)
    left_margin = rng.randint(5, 8)
    right_margin = rng.randint(5, 8)

    height = top_margin + ring_h + bottom_margin
    width = left_margin + ring_w + right_margin
    grid = np.full((height, width), BG, dtype=int)

    r0 = top_margin
    c0 = left_margin
    r1 = r0 + ring_h - 1
    c1 = c0 + ring_w - 1
    order = _perimeter_coords(r0, c0, r1, c1)
    for r, c in order:
        grid[r, c] = TRACK_BASE

    ring_len = len(order)
    n_colors = rng.randint(2, 4)
    colors = rng.sample(TRACK_COLORS, n_colors)
    counts_list = _distribute_counts(n_colors, ring_len, rng)
    counts = {color: count for color, count in zip(colors, counts_list, strict=False)}

    block_specs: dict[int, tuple[int, int, str]] = {}
    shifts: dict[int, int] = {}
    for color in colors:
        block_h, block_w = rng.choice(BLOCK_SHAPES)
        side = rng.choice(ARROW_SIDES)
        block_specs[color] = (block_h, block_w, side)
        shift = block_h * block_w
        if side in {"up", "left"}:
            shift = -shift
        shifts[color] = shift

    indices_by_color = _allocate_ring_indices(colors, counts, shifts, ring_len, rng)
    if indices_by_color is None:
        return None

    for color, indices in indices_by_color.items():
        for idx in indices:
            rr, cc = order[idx]
            grid[rr, cc] = color

    blocked = np.zeros((height, width), dtype=bool)
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(height, r1 + 2)
    cc1 = min(width, c1 + 2)
    blocked[rr0:rr1, cc0:cc1] = True

    for color in colors:
        block_h, block_w, side = block_specs[color]
        if not _place_instruction_group(grid, blocked, color, block_h, block_w, side, rng):
            return None

    expected = _expected_output(grid, order, TRACK_BASE, indices_by_color, shifts)
    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None

    ring = _find_ring(grid, BG)
    if int(ring["base_color"]) != TRACK_BASE:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid puzzle instance")
