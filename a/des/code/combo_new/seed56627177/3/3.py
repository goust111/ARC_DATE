from __future__ import annotations

import random

import numpy as np


BG = 0
RING_SIZE = 5
INNER_START = 1
INNER_STOP = 4
ALL_COLORS = tuple(range(1, 10))
GEN_ATTEMPTS = 300

DIR_INFO = {
    "up": {"gap": (0, 2), "step": (-1, 0)},
    "down": {"gap": (4, 2), "step": (1, 0)},
    "left": {"gap": (2, 0), "step": (0, -1)},
    "right": {"gap": (2, 4), "step": (0, 1)},
}


def _ring_mask(direction: str) -> np.ndarray:
    mask = np.zeros((RING_SIZE, RING_SIZE), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    gap_r, gap_c = DIR_INFO[direction]["gap"]
    mask[gap_r, gap_c] = False
    return mask


RING_MASKS = {direction: _ring_mask(direction) for direction in DIR_INFO}


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _ray_cells(shape: tuple[int, int], start: tuple[int, int], step: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    r, c = start
    dr, dc = step
    cells: list[tuple[int, int]] = []
    r += dr
    c += dc
    while 0 <= r < h and 0 <= c < w:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _find_rings(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    used = np.zeros_like(grid, dtype=bool)
    rings: list[dict[str, object]] = []

    for r0 in range(h - RING_SIZE + 1):
        for c0 in range(w - RING_SIZE + 1):
            sub = grid[r0 : r0 + RING_SIZE, c0 : c0 + RING_SIZE]
            for direction, mask in RING_MASKS.items():
                ring_colors = np.unique(sub[mask])
                if len(ring_colors) != 1 or int(ring_colors[0]) == bg:
                    continue
                if np.any(sub[~mask] != bg):
                    continue
                abs_mask = np.zeros_like(grid, dtype=bool)
                abs_mask[r0 : r0 + RING_SIZE, c0 : c0 + RING_SIZE] = mask
                if np.any(used & abs_mask):
                    continue
                used |= abs_mask
                gap_r, gap_c = DIR_INFO[direction]["gap"]
                rings.append(
                    {
                        "top": r0,
                        "left": c0,
                        "color": int(ring_colors[0]),
                        "direction": direction,
                        "gap": (r0 + gap_r, c0 + gap_c),
                        "step": DIR_INFO[direction]["step"],
                    }
                )
                break
    return rings


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _bg_color(g)

    rings = _find_rings(g, bg)
    assert rings, "expected at least one ring object"

    ring_pixels = np.zeros_like(g, dtype=bool)
    for ring in rings:
        r0 = int(ring["top"])
        c0 = int(ring["left"])
        direction = str(ring["direction"])
        ring_pixels[r0 : r0 + RING_SIZE, c0 : c0 + RING_SIZE] |= RING_MASKS[direction]

    targets: list[tuple[tuple[int, int], int]] = []
    seen_targets: set[tuple[int, int]] = set()
    for ring in rings:
        gap = tuple(ring["gap"])
        step = tuple(ring["step"])
        prey_coord = None
        prey_color = None
        for r, c in _ray_cells(g.shape, gap, step):
            if int(g[r, c]) == bg:
                continue
            if ring_pixels[r, c]:
                continue
            prey_coord = (r, c)
            prey_color = int(g[r, c])
            break
        assert prey_coord is not None and prey_color is not None, "each ring must see a prey on its gap ray"
        assert prey_coord not in seen_targets, "rings should not share a prey"
        seen_targets.add(prey_coord)
        targets.append((prey_coord, prey_color))

    out = g.copy()
    for target, _ in targets:
        out[target] = bg
    for ring, (target, prey_color) in zip(rings, targets):
        r0 = int(ring["top"])
        c0 = int(ring["left"])
        gap_r, gap_c = tuple(ring["gap"])
        out[gap_r, gap_c] = prey_color
        out[r0 + INNER_START : r0 + INNER_STOP, c0 + INNER_START : c0 + INNER_STOP] = prey_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Each gapped square ring swallows the nearest prey in its opening direction and fills itself with that prey color.

    Input structure:
    - The background is black.
    - Several 5x5 monochrome ring objects appear on the grid: each is a square border with a 3x3 hollow center and a single missing border pixel at the middle of one side.
    - Additional 1x1 prey pixels appear elsewhere, and every ring has at least one prey somewhere along the straight line extending from its gap.

    Rule:
    - Detect every ring and infer its facing direction from the side containing the missing border pixel.
    - From the gap, scan cell by cell along that direction until the first prey pixel is reached.
    - Remove that prey pixel from the grid.
    - Repaint the gap pixel and the full 3x3 interior of the ring with the prey color, while leaving the rest of the ring border and all unrelated prey unchanged.

    Key insight:
    - The missing border pixel encodes a search direction, and the first prey found on that ray supplies the color that must be copied back into two separate regions of the ring.
    """
    return _transform(grid)


def _bbox_is_clear(occupied: np.ndarray, reserved: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    h0, w0 = occupied.shape
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h0, r0 + h + 1)
    cc1 = min(w0, c0 + w + 1)
    if np.any(occupied[rr0:rr1, cc0:cc1]):
        return False
    if np.any(reserved[r0 : r0 + h, c0 : c0 + w]):
        return False
    return True


def _paint_ring(grid: np.ndarray, occupied: np.ndarray, r0: int, c0: int, color: int, direction: str) -> None:
    mask = RING_MASKS[direction]
    sub = grid[r0 : r0 + RING_SIZE, c0 : c0 + RING_SIZE]
    sub[mask] = color
    occupied[r0 : r0 + RING_SIZE, c0 : c0 + RING_SIZE] |= mask


def _choose_primary_index(ray_len: int) -> int:
    near_limit = min(ray_len, 6)
    return random.randrange(near_limit)


def _place_singletons(
    grid: np.ndarray,
    occupied: np.ndarray,
    reserved: np.ndarray,
    count: int,
    allowed_colors: list[int],
) -> None:
    h, w = grid.shape
    attempts = 0
    placed = 0
    while placed < count and attempts < 300:
        attempts += 1
        r = random.randrange(h)
        c = random.randrange(w)
        if occupied[r, c] or reserved[r, c]:
            continue
        rr0 = max(0, r - 1)
        cc0 = max(0, c - 1)
        rr1 = min(h, r + 2)
        cc1 = min(w, c + 2)
        if np.any(occupied[rr0:rr1, cc0:cc1]):
            continue
        color = random.choice(allowed_colors)
        grid[r, c] = color
        occupied[r, c] = True
        placed += 1


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 30)
    w = random.randint(18, 30)
    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    reserved = np.zeros((h, w), dtype=bool)

    num_rings = random.choice([2, 3, 4])
    if 2 * num_rings > len(ALL_COLORS):
        return None

    palette = random.sample(ALL_COLORS, 2 * num_rings)
    ring_colors = palette[:num_rings]
    prey_colors = palette[num_rings:]

    directions = list(DIR_INFO)
    random.shuffle(directions)
    directions = directions[:num_rings]

    ring_specs: list[dict[str, object]] = []
    for ring_idx in range(num_rings):
        direction = directions[ring_idx]
        color = ring_colors[ring_idx]
        prey_color = prey_colors[ring_idx]

        placements = [(r0, c0) for r0 in range(h - RING_SIZE + 1) for c0 in range(w - RING_SIZE + 1)]
        random.shuffle(placements)

        placed = False
        for r0, c0 in placements:
            if not _bbox_is_clear(occupied, reserved, r0, c0, RING_SIZE, RING_SIZE):
                continue

            gap_r = r0 + int(DIR_INFO[direction]["gap"][0])
            gap_c = c0 + int(DIR_INFO[direction]["gap"][1])
            ray = _ray_cells((h, w), (gap_r, gap_c), tuple(DIR_INFO[direction]["step"]))
            if len(ray) < 3:
                continue
            if any(occupied[r, c] or reserved[r, c] for r, c in ray):
                continue

            primary_idx = _choose_primary_index(len(ray))
            primary = ray[primary_idx]
            secondary = None
            if len(ray) - primary_idx > 2 and random.random() < 0.5:
                far_choices = ray[primary_idx + 1 :]
                secondary = random.choice(far_choices)

            _paint_ring(grid, occupied, r0, c0, color, direction)
            grid[primary] = prey_color
            occupied[primary] = True
            reserved[[r for r, _ in ray], [c for _, c in ray]] = True

            if secondary is not None:
                grid[secondary] = random.choice(prey_colors)
                occupied[secondary] = True

            ring_specs.append(
                {
                    "top": r0,
                    "left": c0,
                    "direction": direction,
                    "ring_color": color,
                    "target": primary,
                    "prey_color": prey_color,
                }
            )
            placed = True
            break

        if not placed:
            return None

    leftover_colors = [color for color in ALL_COLORS if color not in ring_colors]
    _place_singletons(grid, occupied, reserved, random.randint(0, 3), leftover_colors)

    rings = _find_rings(grid, BG)
    if len(rings) != num_rings:
        return None

    try:
        out = _transform(grid)
    except AssertionError:
        return None

    for spec in ring_specs:
        r0 = int(spec["top"])
        c0 = int(spec["left"])
        prey_color = int(spec["prey_color"])
        target = tuple(spec["target"])
        if int(out[target]) != BG:
            return None
        if not np.all(out[r0 + INNER_START : r0 + INNER_STOP, c0 + INNER_START : c0 + INNER_STOP] == prey_color):
            return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
