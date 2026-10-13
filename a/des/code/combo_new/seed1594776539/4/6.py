import math
import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
PARITY_PAIRS = (("odd", "odd"), ("odd", "even"), ("even", "odd"), ("even", "even"))
GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 250
_RNG = random.Random()


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _component_bboxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    bboxes: list[tuple[int, int, int, int]] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue

            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            rows = [pt[0] for pt in coords]
            cols = [pt[1] for pt in coords]
            bboxes.append((min(rows), min(cols), max(rows), max(cols)))

    return bboxes


def _soil_rectangles(grid: np.ndarray) -> list[tuple[tuple[int, int, int, int], tuple[int, int, int] | None]]:
    g = np.asarray(grid, dtype=int)
    soils: list[tuple[tuple[int, int, int, int], tuple[int, int, int] | None]] = []

    for r0, c0, r1, c1 in _component_bboxes(g != BLACK):
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        assert bool(np.all(patch != BLACK)), "each soil must occupy a solid rectangle"

        seed_locs = np.argwhere((patch != BLACK) & (patch != GREY))
        assert len(seed_locs) <= 1, "each soil rectangle may contain at most one seed"
        seed = None
        if len(seed_locs) == 1:
            sr, sc = map(int, seed_locs[0])
            seed = (sr, sc, int(patch[sr, sc]))

        assert bool(np.any(patch == GREY)) or seed is not None, "soil rectangle must show grey or a seed"
        soils.append(((int(r0), int(c0), int(r1), int(c1)), seed))

    return soils


def _paint_growth(out: np.ndarray, bbox: tuple[int, int, int, int], seed: tuple[int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    sr, sc, color = seed
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    grown_h = max(1, height // 2)
    grown_w = max(1, width // 2)

    start_r = sr - grown_h // 2
    start_c = sc - grown_w // 2
    start_r = min(max(start_r, 0), height - grown_h)
    start_c = min(max(start_c, 0), width - grown_w)

    out[r0 + start_r : r0 + start_r + grown_h, c0 + start_c : c0 + start_c + grown_w] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a half-size colored rectangle inside each seeded grey soil rectangle.

    Input structure:
    - The background is black.
    - Foreground objects are isolated axis-aligned solid rectangles made of grey cells, optionally with one non-black non-grey seed pixel replacing a soil cell.
    - Each rectangle is surrounded by black and represents one soil region.

    Rule:
    - For every soil rectangle containing exactly one seed, measure the full soil size H x W from the rectangle's bounding box.
    - Paint a solid seed-colored rectangle of size max(1, floor(H / 2)) x max(1, floor(W / 2)), centered on the seed with top-left floor tie-breaking for even sizes, then clamp it so it stays fully inside the soil rectangle.
    - Soil rectangles without seeds, plus all background pixels, remain unchanged.

    Key insight:
    - The seed chooses the placement anchor, but the rectangle size comes from halving the containing soil box rather than from the seed itself.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for bbox, seed in _soil_rectangles(g):
        if seed is not None:
            _paint_growth(out, bbox, seed)

    return out


def _pick_with_parity(low: int, high: int, parity: str, rng: random.Random) -> int:
    want = 1 if parity == "odd" else 0
    values = [value for value in range(low, high + 1) if value % 2 == want]
    return rng.choice(values)


def _seed_interior_position(height: int, width: int, rng: random.Random) -> tuple[int, int]:
    grown_h = max(1, height // 2)
    grown_w = max(1, width // 2)

    row_lo = grown_h // 2
    row_hi = height - grown_h + grown_h // 2
    col_lo = grown_w // 2
    col_hi = width - grown_w + grown_w // 2

    rows = [row for row in range(row_lo, row_hi + 1) if 0 < row < height - 1] or list(range(row_lo, row_hi + 1))
    cols = [col for col in range(col_lo, col_hi + 1) if 0 < col < width - 1] or list(range(col_lo, col_hi + 1))
    return rng.choice(rows), rng.choice(cols)


def _seed_position(height: int, width: int, mode: str, rng: random.Random) -> tuple[int, int]:
    if mode == "interior":
        return _seed_interior_position(height, width, rng)
    if mode == "clamp":
        return rng.choice(((0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)))
    if mode == "edge":
        edge_cells = []
        for r in range(height):
            for c in range(width):
                if r in (0, height - 1) or c in (0, width - 1):
                    edge_cells.append((r, c))
        return rng.choice(edge_cells)
    return rng.randrange(height), rng.randrange(width)


def _make_special_specs(rng: random.Random) -> list[dict[str, int | str | None]]:
    interior_parity, clamp_parity = rng.sample(PARITY_PAIRS, 2)
    specs: list[dict[str, int | str | None]] = [
        {
            "h": _pick_with_parity(3, 9, interior_parity[0], rng),
            "w": _pick_with_parity(3, 9, interior_parity[1], rng),
            "seed_mode": "interior",
        },
        {
            "h": _pick_with_parity(4, 9, clamp_parity[0], rng),
            "w": _pick_with_parity(4, 9, clamp_parity[1], rng),
            "seed_mode": "clamp",
        },
        {
            "h": rng.randint(2, 9),
            "w": rng.randint(2, 9),
            "seed_mode": None,
        },
    ]
    return specs


def _make_extra_spec(rng: random.Random) -> dict[str, int | str | None]:
    roll = rng.random()
    if roll < 0.30:
        if rng.random() < 0.5:
            return {"h": 1, "w": rng.randint(2, 9), "seed_mode": rng.choice(("interior", "edge", "random"))}
        return {"h": rng.randint(2, 9), "w": 1, "seed_mode": rng.choice(("interior", "edge", "random"))}
    if roll < 0.65:
        h = rng.randint(2, 9)
        w = rng.randint(2, 9)
        if h >= 4 and w >= 4 and rng.random() < 0.35:
            mode = "clamp"
        else:
            mode = rng.choice(("interior", "edge", "random"))
        return {"h": h, "w": w, "seed_mode": mode}
    return {"h": rng.randint(1, 9), "w": rng.randint(1, 9), "seed_mode": None}


def _build_specs(rng: random.Random) -> list[dict[str, int | str | None]]:
    total = rng.randint(3, 6)
    specs = _make_special_specs(rng)
    include_thin = total >= 4 and rng.random() < 0.55
    while len(specs) < total:
        spec = _make_extra_spec(rng)
        if include_thin and spec["seed_mode"] is None:
            continue
        if include_thin and int(spec["h"]) > 1 and int(spec["w"]) > 1:
            continue
        specs.append(spec)
        include_thin = False

    seeded = [spec for spec in specs if spec["seed_mode"] is not None]
    colors = list(SEED_COLORS)
    rng.shuffle(colors)
    for spec, color in zip(seeded, colors):
        spec["color"] = color
    return specs


def _place_specs(
    shape: tuple[int, int],
    specs: list[dict[str, int | str | None]],
    rng: random.Random,
) -> list[tuple[dict[str, int | str | None], int, int]] | None:
    height, width = shape
    blocked = np.zeros((height, width), dtype=bool)
    placements: list[tuple[dict[str, int | str | None], int, int]] = []

    for spec in sorted(specs, key=lambda item: int(item["h"]) * int(item["w"]), reverse=True):
        box_h = int(spec["h"])
        box_w = int(spec["w"])
        placed = False
        for _ in range(PLACE_ATTEMPTS):
            r0 = rng.randint(0, height - box_h)
            c0 = rng.randint(0, width - box_w)
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(height, r0 + box_h + 1)
            cc1 = min(width, c0 + box_w + 1)
            if bool(blocked[rr0:rr1, cc0:cc1].any()):
                continue
            blocked[rr0:rr1, cc0:cc1] = True
            placements.append((spec, r0, c0))
            placed = True
            break
        if not placed:
            return None

    return placements


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    specs = _build_specs(rng)
    total_area = sum(int(spec["h"]) * int(spec["w"]) for spec in specs)
    min_side = max(12, min(24, math.isqrt(total_area * 3) + 3))
    height = rng.randint(min_side, 30)
    width = rng.randint(min_side, 30)

    placements = _place_specs((height, width), specs, rng)
    if placements is None:
        return None

    grid = np.full((height, width), BLACK, dtype=int)
    for spec, r0, c0 in placements:
        box_h = int(spec["h"])
        box_w = int(spec["w"])
        grid[r0 : r0 + box_h, c0 : c0 + box_w] = GREY
        if spec["seed_mode"] is not None:
            sr, sc = _seed_position(box_h, box_w, str(spec["seed_mode"]), rng)
            grid[r0 + sr, c0 + sc] = int(spec["color"])

    soils = _soil_rectangles(grid)
    if len(soils) != len(specs):
        return None
    if sum(seed is None for _bbox, seed in soils) == 0:
        return None
    if sum(seed is not None for _bbox, seed in soils) < 2:
        return None
    if np.array_equal(main(grid), grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to produce a valid sample")
