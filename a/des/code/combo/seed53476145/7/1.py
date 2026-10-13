from __future__ import annotations

import random

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
QUADRANTS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
RADIUS_OPTIONS = (3, 4, 5, 6)
MAX_GENERATION_ATTEMPTS = 500


def _components(mask: np.ndarray, connectivity: int = 8) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 8:
        neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]
    else:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in neighbors:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _diamond_frame_cells(center: tuple[int, int], radius: int) -> set[tuple[int, int]]:
    cr, cc = center
    cells: set[tuple[int, int]] = set()
    for dr in range(-radius, radius + 1):
        rem = radius - abs(dr)
        cells.add((cr + dr, cc - rem))
        cells.add((cr + dr, cc + rem))
    return cells


def _diamond_solid_cells(center: tuple[int, int], radius: int) -> set[tuple[int, int]]:
    cr, cc = center
    cells: set[tuple[int, int]] = set()
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if abs(dr) + abs(dc) <= radius:
                cells.add((cr + dr, cc + dc))
    return cells


def _diamond_from_component(points: list[tuple[int, int]]) -> dict | None:
    if not points:
        return None
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h != w or h % 2 == 0:
        return None
    radius = h // 2
    if radius < 3 or radius > 6:
        return None
    center = ((r0 + r1) // 2, (c0 + c1) // 2)
    pts = set(points)
    expected = _diamond_frame_cells(center, radius)
    if pts != expected:
        return None
    if len(pts) != 4 * radius:
        return None
    return {"center": center, "radius": radius, "frame": pts}


def _find_diamonds(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    diamonds: list[dict] = []
    for color in sorted(int(c) for c in np.unique(g) if int(c) != BG):
        for comp in _components(g == color, connectivity=8):
            info = _diamond_from_component(comp)
            if info is None:
                continue
            cr, cc = info["center"]
            radius = info["radius"]
            seeds: list[tuple[int, int, int]] = []
            for r in range(max(0, cr - radius + 1), min(h, cr + radius)):
                for c in range(max(0, cc - radius + 1), min(w, cc + radius)):
                    if abs(r - cr) + abs(c - cc) < radius and int(g[r, c]) != BG:
                        seeds.append((r, c, int(g[r, c])))
            diamonds.append(
                {
                    "color": color,
                    "center": info["center"],
                    "radius": radius,
                    "frame": info["frame"],
                    "seeds": seeds,
                }
            )
    diamonds.sort(key=lambda d: (d["center"][0], d["center"][1], d["radius"], d["color"]))
    return diamonds


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit diagonal rays from seed pixels inside hollow diamond frames.

    Each seed emits toward the outward diagonal of its quadrant relative to the
    diamond center. The ray length is the seed's Manhattan distance to the
    center. Existing frames and seed pixels remain unchanged.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    h, w = g.shape

    for diamond in _find_diamonds(g):
        cr, cc = diamond["center"]
        for r, c, color in diamond["seeds"]:
            dr = -1 if r < cr else 1 if r > cr else 0
            dc = -1 if c < cc else 1 if c > cc else 0
            assert dr != 0 and dc != 0, "seed must lie in a strict quadrant"
            length = abs(r - cr) + abs(c - cc)
            for step in range(1, length + 1):
                rr = r + dr * step
                cc2 = c + dc * step
                if 0 <= rr < h and 0 <= cc2 < w and int(out[rr, cc2]) == BG:
                    out[rr, cc2] = color
    return out


def _positive_seed_offsets(radius: int) -> list[tuple[int, int]]:
    return [
        (a, b)
        for a in range(1, radius)
        for b in range(1, radius)
        if a + b < radius
    ]


def _choose_seed_offsets(
    rng: random.Random, radius: int, seed_count: int, want_clipped: bool
) -> tuple[list[tuple[int, int]], tuple[int, int] | None]:
    quadrants = rng.sample(QUADRANTS, seed_count)
    positive = _positive_seed_offsets(radius)
    assert positive

    clip_quadrant = rng.choice(quadrants) if want_clipped else None
    special_positive = None
    if clip_quadrant is not None:
        clip_candidates = [
            (a, b)
            for a, b in positive
            if a + b >= max(2, radius - 2) and max(a, b) >= 2
        ]
        if not clip_candidates:
            clip_candidates = sorted(positive, key=lambda ab: (ab[0] + ab[1], max(ab)), reverse=True)
        special_positive = rng.choice(clip_candidates)

    offsets: list[tuple[int, int]] = []
    special_offset = None
    for quadrant in quadrants:
        if quadrant == clip_quadrant:
            a, b = special_positive
        else:
            a, b = rng.choice(positive)
        dr = quadrant[0] * a
        dc = quadrant[1] * b
        offset = (dr, dc)
        offsets.append(offset)
        if quadrant == clip_quadrant:
            special_offset = offset
    return offsets, special_offset


def _ray_trace(
    center: tuple[int, int], seed_offset: tuple[int, int]
) -> list[tuple[int, int]]:
    dr0, dc0 = seed_offset
    step_r = -1 if dr0 < 0 else 1
    step_c = -1 if dc0 < 0 else 1
    length = abs(dr0) + abs(dc0)
    cr, cc = center
    sr, sc = cr + dr0, cc + dc0
    return [(sr + step_r * k, sc + step_c * k) for k in range(1, length + 1)]


def _ray_clips(
    center: tuple[int, int],
    seed_offset: tuple[int, int],
    height: int,
    width: int,
) -> bool:
    for r, c in _ray_trace(center, seed_offset):
        if not (0 <= r < height and 0 <= c < width):
            return True
    return False


def _future_ray_cells(
    center: tuple[int, int],
    radius: int,
    seed_offsets: list[tuple[int, int]],
    height: int,
    width: int,
) -> set[tuple[int, int]]:
    solid = _diamond_solid_cells(center, radius)
    future: set[tuple[int, int]] = set()
    for offset in seed_offsets:
        for cell in _ray_trace(center, offset):
            r, c = cell
            if 0 <= r < height and 0 <= c < width and cell not in solid:
                future.add(cell)
    return future


def _pick_center(
    rng: random.Random,
    height: int,
    width: int,
    radius: int,
    seed_offsets: list[tuple[int, int]],
    special_offset: tuple[int, int] | None,
    want_clipped: bool,
    occupied_solid: set[tuple[int, int]],
    occupied_rays: set[tuple[int, int]],
) -> tuple[int, int] | None:
    rmin, rmax = radius, height - radius - 1
    cmin, cmax = radius, width - radius - 1
    if rmin > rmax or cmin > cmax:
        return None

    candidates: list[tuple[int, int]] = []
    if want_clipped and special_offset is not None:
        sdr, sdc = special_offset
        row_anchor = rmin if sdr < 0 else rmax
        col_anchor = cmin if sdc < 0 else cmax
        row_choices = list(range(max(rmin, row_anchor - 1), min(rmax, row_anchor + 1) + 1))
        col_choices = list(range(max(cmin, col_anchor - 1), min(cmax, col_anchor + 1) + 1))
        for r in row_choices:
            for c in col_choices:
                candidates.append((r, c))
        rng.shuffle(candidates)

    all_positions = [
        (r, c)
        for r in range(rmin, rmax + 1)
        for c in range(cmin, cmax + 1)
    ]
    rng.shuffle(all_positions)
    candidates.extend(all_positions)

    checked: set[tuple[int, int]] = set()
    for center in candidates:
        if center in checked:
            continue
        checked.add(center)

        clipped = any(_ray_clips(center, offset, height, width) for offset in seed_offsets)
        if clipped != want_clipped:
            continue

        solid = _diamond_solid_cells(center, radius)
        if solid & occupied_solid or solid & occupied_rays:
            continue

        future = _future_ray_cells(center, radius, seed_offsets, height, width)
        if future & occupied_solid or future & occupied_rays:
            continue
        return center
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    diamond_count = rng.choice((1, 2, 3))
    min_side = {1: 17, 2: 24, 3: 28}[diamond_count]
    height = rng.randint(min_side, min_side + 8)
    width = rng.randint(min_side, min_side + 8)

    frame_colors = rng.sample(list(COLORS), diamond_count)
    force_clipped = diamond_count > 0 and rng.random() < 0.5
    clipped_index = rng.randrange(diamond_count) if force_clipped else -1

    occupied_solid: set[tuple[int, int]] = set()
    occupied_rays: set[tuple[int, int]] = set()
    placements: list[dict] = []

    for index in range(diamond_count):
        want_clipped = index == clipped_index
        radius_pool = (4, 5, 6) if want_clipped else RADIUS_OPTIONS
        radius = rng.choice(radius_pool)
        seed_count = rng.randint(1, 3)
        seed_offsets, special_offset = _choose_seed_offsets(rng, radius, seed_count, want_clipped)
        center = _pick_center(
            rng,
            height,
            width,
            radius,
            seed_offsets,
            special_offset,
            want_clipped,
            occupied_solid,
            occupied_rays,
        )
        if center is None:
            return None

        frame_color = frame_colors[index]
        seed_colors = rng.sample([c for c in COLORS if c != frame_color], seed_count)
        solid = _diamond_solid_cells(center, radius)
        future = _future_ray_cells(center, radius, seed_offsets, height, width)
        occupied_solid |= solid
        occupied_rays |= future
        placements.append(
            {
                "center": center,
                "radius": radius,
                "frame_color": frame_color,
                "seed_offsets": seed_offsets,
                "seed_colors": seed_colors,
            }
        )

    grid = np.zeros((height, width), dtype=int)
    for spec in placements:
        center = spec["center"]
        for r, c in _diamond_frame_cells(center, spec["radius"]):
            grid[r, c] = spec["frame_color"]
        cr, cc = center
        for (dr, dc), color in zip(spec["seed_offsets"], spec["seed_colors"]):
            grid[cr + dr, cc + dc] = color

    diamonds = _find_diamonds(grid)
    if len(diamonds) != diamond_count:
        return None
    for diamond in diamonds:
        if not (1 <= len(diamond["seeds"]) <= 3):
            return None
        seed_colors = [color for _, _, color in diamond["seeds"]]
        if len(seed_colors) != len(set(seed_colors)):
            return None
        for r, c, _ in diamond["seeds"]:
            cr, cc = diamond["center"]
            if r == cr or c == cc:
                return None

    out = main(grid)
    if not np.array_equal(out[grid != BG], grid[grid != BG]):
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
