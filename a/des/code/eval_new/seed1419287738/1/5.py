import random
from collections import Counter

import numpy as np


BG = 0
POT = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
SIDES = ("top", "bottom", "left", "right")

GRID_MAX = 30
MIN_NUTRIENTS = 2
MAX_NUTRIENTS = 7
OUTER_H_RANGE = (5, 8)
OUTER_W_RANGE = (5, 8)
POT_RETRIES = 200
GEN_RETRIES = 300


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    pts = [(r0, c) for c in range(c0, c1 + 1)]
    pts.extend((r, c1) for r in range(r0 + 1, r1))
    pts.extend((r1, c) for c in range(c1, c0 - 1, -1))
    pts.extend((r, c0) for r in range(r1 - 1, r0, -1))
    return pts


def _gap_step(
    r0: int, c0: int, r1: int, c1: int, gap: tuple[int, int]
) -> tuple[int, int] | None:
    r, c = gap
    if r == r0 and c0 < c < c1:
        return (-1, 0)
    if r == r1 and c0 < c < c1:
        return (1, 0)
    if c == c0 and r0 < r < r1:
        return (0, -1)
    if c == c1 and r0 < r < r1:
        return (0, 1)
    return None


def _read_pot(window: np.ndarray) -> tuple[int, int, int] | None:
    values = [int(v) for v in window.ravel() if int(v) not in (BG, POT)]
    counts = Counter(values)
    if len(counts) != 2:
        return None
    seed_colors = [color for color, count in counts.items() if count == 1]
    if len(seed_colors) != 1:
        return None
    seed = seed_colors[0]
    nutrient = next(color for color in counts if color != seed)
    return seed, nutrient, int(counts[nutrient])


def _find_pots(grid: np.ndarray) -> list[dict]:
    pots: list[dict] = []
    for comp in _components(grid == POT):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        border = _border_cells(r0, c0, r1, c1)
        gaps = [(r, c) for r, c in border if int(grid[r, c]) != POT]
        if len(gaps) != 1:
            continue
        gap = gaps[0]
        step = _gap_step(r0, c0, r1, c1, gap)
        if step is None:
            continue
        inside = grid[r0 + 1 : r1, c0 + 1 : c1]
        info = _read_pot(inside)
        if info is None:
            continue
        seed, nutrient, length = info
        pots.append(
            {
                "bbox": (r0, c0, r1, c1),
                "gap": gap,
                "step": step,
                "seed": seed,
                "nutrient": nutrient,
                "length": length,
            }
        )
    return pots


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow an alternating plant outward from each gapped grey pot.

    Input structure:
    - Black background surrounds one or more grey rectangular pots.
    - Each pot is a one-cell-thick grey rectangle with exactly one missing border
      pixel on a non-corner side position.
    - Inside each pot, one singleton color is the seed and one repeated color is
      the nutrient, on an otherwise black interior.

    Rule:
    - The side containing the gap determines the growth direction: top means up,
      bottom means down, left means left, and right means right.
    - Remove every nutrient pixel inside the pot, keep the seed in place, and
      draw a one-cell-wide plant starting just outside the gap.
    - The plant length equals the number of nutrient pixels, and its colors
      alternate seed, nutrient, seed, nutrient, ... until the grid edge clips it.

    Key insight:
    - Each pot independently encodes direction with the gap side, plant colors
      with the two interior colors, and plant length with the nutrient count.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    for pot in _find_pots(g):
        r0, c0, r1, c1 = pot["bbox"]
        seed = int(pot["seed"])
        nutrient = int(pot["nutrient"])
        gap_r, gap_c = pot["gap"]
        dr, dc = pot["step"]
        length = int(pot["length"])

        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[interior == nutrient] = BG

        for i in range(length):
            rr = gap_r + (i + 1) * dr
            cc = gap_c + (i + 1) * dc
            if not (0 <= rr < h and 0 <= cc < w):
                break
            out[rr, cc] = seed if i % 2 == 0 else nutrient

    return out


def _gap_and_step(
    side: str, r0: int, c0: int, outer_h: int, outer_w: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    if side == "top":
        return (r0, c0 + random.randint(1, outer_w - 2)), (-1, 0)
    if side == "bottom":
        return (r0 + outer_h - 1, c0 + random.randint(1, outer_w - 2)), (1, 0)
    if side == "left":
        return (r0 + random.randint(1, outer_h - 2), c0), (0, -1)
    return (r0 + random.randint(1, outer_h - 2), c0 + outer_w - 1), (0, 1)


def _ray_cells(
    height: int, width: int, gap: tuple[int, int], step: tuple[int, int]
) -> list[tuple[int, int]]:
    dr, dc = step
    r, c = gap
    cells = []
    while True:
        r += dr
        c += dc
        if not (0 <= r < height and 0 <= c < width):
            break
        cells.append((r, c))
    return cells


def _reserve_cells(
    height: int,
    width: int,
    r0: int,
    c0: int,
    outer_h: int,
    outer_w: int,
    gap: tuple[int, int],
    step: tuple[int, int],
) -> set[tuple[int, int]]:
    reserved: set[tuple[int, int]] = set()
    for r in range(max(0, r0 - 1), min(height, r0 + outer_h + 1)):
        for c in range(max(0, c0 - 1), min(width, c0 + outer_w + 1)):
            reserved.add((r, c))
    reserved.update(_ray_cells(height, width, gap, step))
    return reserved


def _place_pot(
    rng: random.Random,
    height: int,
    width: int,
    occupied: np.ndarray,
    seed_color: int,
    nutrient_color: int,
) -> tuple[dict, set[tuple[int, int]]] | None:
    for _ in range(POT_RETRIES):
        outer_h = rng.randint(*OUTER_H_RANGE)
        outer_w = rng.randint(*OUTER_W_RANGE)
        area = (outer_h - 2) * (outer_w - 2)
        max_len = min(MAX_NUTRIENTS, area - 1)
        if max_len < MIN_NUTRIENTS:
            continue

        length = rng.randint(MIN_NUTRIENTS, max_len)
        truncated = rng.random() < 0.5
        sides = list(SIDES)
        rng.shuffle(sides)

        for side in sides:
            max_available = height - outer_h - 1 if side in ("top", "bottom") else width - outer_w - 1
            if truncated:
                hi = min(max_available, length - 1)
                if hi < 1:
                    continue
                available = rng.randint(1, hi)
            else:
                if max_available < length:
                    continue
                available = rng.randint(length, max_available)

            if side == "top":
                r0 = available
                c0 = rng.randint(1, width - outer_w - 1)
            elif side == "bottom":
                r0 = height - outer_h - available
                c0 = rng.randint(1, width - outer_w - 1)
            elif side == "left":
                r0 = rng.randint(1, height - outer_h - 1)
                c0 = available
            else:
                r0 = rng.randint(1, height - outer_h - 1)
                c0 = width - outer_w - available

            gap, step = _gap_and_step(side, r0, c0, outer_h, outer_w)
            reserved = _reserve_cells(height, width, r0, c0, outer_h, outer_w, gap, step)
            if any(occupied[r, c] for r, c in reserved):
                continue

            interior_cells = [
                (r, c)
                for r in range(r0 + 1, r0 + outer_h - 1)
                for c in range(c0 + 1, c0 + outer_w - 1)
            ]
            seed_pos = rng.choice(interior_cells)
            nutrient_pool = [cell for cell in interior_cells if cell != seed_pos]
            if len(nutrient_pool) < length:
                continue
            nutrient_positions = rng.sample(nutrient_pool, length)

            spec = {
                "r0": r0,
                "c0": c0,
                "outer_h": outer_h,
                "outer_w": outer_w,
                "side": side,
                "gap": gap,
                "step": step,
                "seed_color": seed_color,
                "nutrient_color": nutrient_color,
                "seed_pos": seed_pos,
                "nutrient_positions": nutrient_positions,
                "length": length,
            }
            return spec, reserved
    return None


def _render_grids(height: int, width: int, pots: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.zeros((height, width), dtype=int)
    out = np.zeros((height, width), dtype=int)

    for pot in pots:
        r0 = int(pot["r0"])
        c0 = int(pot["c0"])
        outer_h = int(pot["outer_h"])
        outer_w = int(pot["outer_w"])
        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1
        gap = pot["gap"]

        for r, c in _border_cells(r0, c0, r1, c1):
            if (r, c) != gap:
                inp[r, c] = POT
                out[r, c] = POT

        sr, sc = pot["seed_pos"]
        seed_color = int(pot["seed_color"])
        nutrient_color = int(pot["nutrient_color"])
        inp[sr, sc] = seed_color
        out[sr, sc] = seed_color

        for nr, nc in pot["nutrient_positions"]:
            inp[nr, nc] = nutrient_color

        dr, dc = pot["step"]
        for i in range(int(pot["length"])):
            rr = gap[0] + (i + 1) * dr
            cc = gap[1] + (i + 1) * dc
            if not (0 <= rr < height and 0 <= cc < width):
                break
            out[rr, cc] = seed_color if i % 2 == 0 else nutrient_color

    return inp, out


def _background_is_clear(grid: np.ndarray) -> bool:
    counts = Counter(map(int, grid.ravel()))
    return counts[BG] > max((count for color, count in counts.items() if color != BG), default=0)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_pots = rng.randint(1, 4)
    min_dim = 12 + 4 * (n_pots - 1)
    height = rng.randint(min_dim, GRID_MAX)
    width = rng.randint(min_dim, GRID_MAX)

    colors = PALETTE[:]
    rng.shuffle(colors)
    occupied = np.zeros((height, width), dtype=bool)
    pots: list[dict] = []

    for _ in range(n_pots):
        seed_color = colors.pop()
        nutrient_color = colors.pop()
        placed = _place_pot(rng, height, width, occupied, seed_color, nutrient_color)
        if placed is None:
            return None
        spec, reserved = placed
        pots.append(spec)
        for r, c in reserved:
            occupied[r, c] = True

    inp, out = _render_grids(height, width, pots)
    if not _background_is_clear(inp):
        return None
    if np.array_equal(inp, out):
        return None
    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(random)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC sample")
