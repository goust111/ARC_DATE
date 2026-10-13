from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


BG = 0
COLLISION_COLOR = 2
ARROW_COLORS = [1, 3, 4, 5, 6, 7, 8, 9]
AXIS_DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAGONAL_DIRECTIONS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ALL_DIRECTIONS = AXIS_DIRECTIONS + DIAGONAL_DIRECTIONS
SIZES = [3, 4, 5, 6]
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= int(r) < h and 0 <= int(c) < w


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((r + dr, c + dc))
    return out


def _bbox_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    comps: list[dict] = []
    for color in sorted(set(map(int, np.unique(grid))) - {int(bg)}):
        mask = grid == color
        seen = np.zeros((h, w), dtype=bool)
        for sr, sc in np.argwhere(mask):
            sr = int(sr)
            sc = int(sc)
            if seen[sr, sc]:
                continue
            stack = [(sr, sc)]
            seen[sr, sc] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for nr, nc in _neighbors8(r, c):
                    if not _in_bounds((h, w), nr, nc):
                        continue
                    if seen[nr, nc] or not mask[nr, nc]:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            coords.sort()
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox_coords(coords),
                    "area": len(coords),
                }
            )
    return comps


def _template_offsets(direction: tuple[int, int], size: int) -> list[tuple[int, int]]:
    dr, dc = direction
    if direction in AXIS_DIRECTIONS:
        pr, pc = -dc, dr
        wing_a = (-dr + pr, -dc + pc)
        wing_b = (-dr - pr, -dc - pc)
        coords = [(0, 0), wing_a, wing_b]
        extras = [
            (2 * wing_a[0], 2 * wing_a[1]),
            (2 * wing_b[0], 2 * wing_b[1]),
            (3 * wing_a[0], 3 * wing_a[1]),
        ]
        coords.extend(extras[: size - 3])
    else:
        coords = [(0, 0), (-dr, 0), (0, -dc)]
        extras = [(-2 * dr, 0), (0, -2 * dc), (-3 * dr, 0)]
        coords.extend(extras[: size - 3])
    assert len(coords) == size
    assert len(set(coords)) == size
    return coords


def _normalize_coords(
    coords: list[tuple[int, int]],
) -> tuple[tuple[tuple[int, int], ...], tuple[int, int]]:
    r0, c0, _r1, _c1 = _bbox_coords(coords)
    norm = tuple(sorted((r - r0, c - c0) for r, c in coords))
    return norm, (r0, c0)


def _build_template_index() -> dict[tuple[tuple[int, int], ...], dict]:
    index: dict[tuple[tuple[int, int], ...], dict] = {}
    for direction in ALL_DIRECTIONS:
        for size in SIZES:
            rel = _template_offsets(direction, size)
            signature, (r0, c0) = _normalize_coords(rel)
            apex_norm = (-r0, -c0)
            info = {
                "direction": direction,
                "size": int(size),
                "apex_norm": apex_norm,
            }
            if signature in index:
                raise ValueError("template signature collision")
            index[signature] = info
    return index


TEMPLATE_INDEX = _build_template_index()


def _parse_arrow(comp: dict) -> dict | None:
    if comp["area"] not in SIZES:
        return None
    signature, (r0, c0) = _normalize_coords(comp["coords"])
    info = TEMPLATE_INDEX.get(signature)
    if info is None:
        return None
    apex_r = r0 + info["apex_norm"][0]
    apex_c = c0 + info["apex_norm"][1]
    return {
        "color": int(comp["color"]),
        "coords": list(comp["coords"]),
        "size": int(info["size"]),
        "direction": tuple(info["direction"]),
        "apex": (int(apex_r), int(apex_c)),
    }


def _occupied_cells(arrows: list[dict]) -> set[tuple[int, int]]:
    occ: set[tuple[int, int]] = set()
    for arrow in arrows:
        occ.update(tuple(cell) for cell in arrow["coords"])
    return occ


def _pulse_map(shape: tuple[int, int], arrows: list[dict]) -> dict[tuple[int, int], list[int]]:
    hits: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, arrow in enumerate(arrows):
        r, c = arrow["apex"]
        dr, dc = arrow["direction"]
        step = int(arrow["size"])
        rr = r + step * dr
        cc = c + step * dc
        while _in_bounds(shape, rr, cc):
            hits[(rr, cc)].append(idx)
            rr += step * dr
            cc += step * dc
    return hits


def _apply_pulses(base: np.ndarray, arrows: list[dict]) -> np.ndarray:
    out = base.copy()
    occ = _occupied_cells(arrows)
    hits = _pulse_map(base.shape, arrows)
    for (r, c), shooters in hits.items():
        if (r, c) in occ:
            continue
        out[r, c] = COLLISION_COLOR if len(shooters) >= 2 else arrows[shooters[0]]["color"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend size-coded pulse rays from small arrowheads and paint phase-aligned overlaps red.

    Input structure:
    - A black background contains several monochrome 8-connected arrowhead sprites with 3 to 6 cells each.
    - Axis-pointing arrows are V-shaped; diagonal-pointing arrows are L-shaped with a missing corner.
    - Every arrow is a single connected component, and its tip cell is the pulse origin.

    Rule:
    - Detect each arrow's pointing direction from its local geometry and let k be that arrow's cell count.
    - Starting from the tip, mark cells at distances k, 2k, 3k, ... along the arrow direction until leaving the grid.
    - All arrows fire simultaneously; a background cell hit by two or more pulses becomes red (2), otherwise it takes the lone shooter's color.
    - Arrow sprites themselves stay unchanged even if a pulse would land on one of their cells.

    Key insight:
    - The sprite shape determines the direction, but the pulse period is encoded by the sprite's own area rather than by color or bbox size.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    assert bg == BG, "expected black background"

    arrows: list[dict] = []
    for comp in _monochrome_components(g, bg):
        arrow = _parse_arrow(comp)
        assert arrow is not None, "foreground component is not a supported arrow sprite"
        arrows.append(arrow)

    assert arrows, "expected at least one arrow"
    return _apply_pulses(g, arrows)


def _make_arrow(color: int, direction: tuple[int, int], size: int, apex: tuple[int, int]) -> dict:
    rel = _template_offsets(direction, size)
    ar, ac = apex
    coords = [(ar + dr, ac + dc) for dr, dc in rel]
    return {
        "color": int(color),
        "direction": tuple(direction),
        "size": int(size),
        "apex": (int(ar), int(ac)),
        "coords": coords,
    }


def _occupied_and_blocked(arrows: list[dict]) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    occ = _occupied_cells(arrows)
    blocked = set(occ)
    for r, c in occ:
        blocked.update(_neighbors8(r, c))
    return occ, blocked


def _fits(spec: dict, shape: tuple[int, int], blocked: set[tuple[int, int]]) -> bool:
    cells = spec["coords"]
    if len(set(cells)) != spec["size"]:
        return False
    if any(not _in_bounds(shape, r, c) for r, c in cells):
        return False
    if any((r, c) in blocked for r, c in cells):
        return False
    ar, ac = spec["apex"]
    dr, dc = spec["direction"]
    step = spec["size"]
    return _in_bounds(shape, ar + step * dr, ac + step * dc)


def _render_input(shape: tuple[int, int], arrows: list[dict]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for arrow in arrows:
        for r, c in arrow["coords"]:
            grid[r, c] = arrow["color"]
    return grid


def _render_expected(shape: tuple[int, int], arrows: list[dict]) -> np.ndarray:
    return _apply_pulses(_render_input(shape, arrows), arrows)


def _family(direction: tuple[int, int]) -> str:
    return "axis" if direction in AXIS_DIRECTIONS else "diag"


def _sample_random_arrow(
    shape: tuple[int, int],
    color: int,
    blocked: set[tuple[int, int]],
    *,
    direction_pool: list[tuple[int, int]] | None = None,
    size_pool: list[int] | None = None,
) -> dict | None:
    directions = list(direction_pool or ALL_DIRECTIONS)
    sizes = list(size_pool or SIZES)
    random.shuffle(directions)
    random.shuffle(sizes)
    for direction in directions:
        rel = _template_offsets(direction, 3)
        rows = [r for r, _ in rel]
        cols = [c for _, c in rel]
        for size in sizes:
            rel = _template_offsets(direction, size)
            rows = [r for r, _ in rel]
            cols = [c for _, c in rel]
            min_r, max_r = min(rows), max(rows)
            min_c, max_c = min(cols), max(cols)
            apex_rows = list(range(-min_r, shape[0] - max_r))
            apex_cols = list(range(-min_c, shape[1] - max_c))
            if not apex_rows or not apex_cols:
                continue
            for _ in range(80):
                apex = (random.choice(apex_rows), random.choice(apex_cols))
                spec = _make_arrow(color, direction, size, apex)
                if _fits(spec, shape, blocked):
                    return spec
    return None


def _sample_collision_pair(
    shape: tuple[int, int],
    color_a: int,
    color_b: int,
    blocked: set[tuple[int, int]],
    *,
    first_pair: bool,
) -> list[dict] | None:
    if first_pair:
        pools = [(AXIS_DIRECTIONS, DIAGONAL_DIRECTIONS), (DIAGONAL_DIRECTIONS, AXIS_DIRECTIONS)]
    else:
        pools = [(ALL_DIRECTIONS, ALL_DIRECTIONS)]

    for _ in range(300):
        pool_a, pool_b = random.choice(pools)
        dir_a = random.choice(pool_a)
        dir_b = random.choice(pool_b)
        if dir_a == dir_b:
            continue
        size_a = random.choice(SIZES)
        size_b = random.choice(SIZES)
        mult_a = random.choice([1, 1, 2, 2, 3])
        mult_b = random.choice([1, 1, 2, 2, 3])
        target = (random.randrange(shape[0]), random.randrange(shape[1]))
        apex_a = (target[0] - mult_a * size_a * dir_a[0], target[1] - mult_a * size_a * dir_a[1])
        apex_b = (target[0] - mult_b * size_b * dir_b[0], target[1] - mult_b * size_b * dir_b[1])
        spec_a = _make_arrow(color_a, dir_a, size_a, apex_a)
        if not _fits(spec_a, shape, blocked):
            continue
        temp_occ, temp_blocked = _occupied_and_blocked([spec_a])
        temp_blocked |= blocked
        if any((r, c) in blocked for r, c in spec_a["coords"]):
            continue
        spec_b = _make_arrow(color_b, dir_b, size_b, apex_b)
        if any((r, c) in temp_blocked for r, c in spec_b["coords"]):
            continue
        if not _fits(spec_b, shape, temp_blocked):
            continue
        if target in set(spec_a["coords"]) or target in set(spec_b["coords"]):
            continue
        return [spec_a, spec_b]
    return None


def _example_is_valid(shape: tuple[int, int], arrows: list[dict], required_collisions: int) -> bool:
    if not arrows:
        return False

    families = {_family(arrow["direction"]) for arrow in arrows}
    if families != {"axis", "diag"}:
        return False
    if len({arrow["size"] for arrow in arrows}) < 2:
        return False

    hits = _pulse_map(shape, arrows)
    occ = _occupied_cells(arrows)
    if any(cell in occ for cell in hits):
        return False

    visible_hits = {cell: shooters for cell, shooters in hits.items() if cell not in occ}
    if not visible_hits:
        return False

    collision_cells = [cell for cell, shooters in visible_hits.items() if len(shooters) >= 2]
    solo_cells = [cell for cell, shooters in visible_hits.items() if len(shooters) == 1]
    if len(collision_cells) < required_collisions:
        return False
    if not solo_cells:
        return False

    inp = _render_input(shape, arrows)
    expected = _render_expected(shape, arrows)
    if np.count_nonzero(expected != inp) < 4:
        return False
    if np.count_nonzero(expected != inp) > inp.size // 4:
        return False
    return np.array_equal(main(inp), expected)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(16, 28)
    w = random.randint(16, 28)
    shape = (h, w)

    required_collisions = random.choice([1, 1, 2])
    arrow_count = random.randint(2 * required_collisions + 1, min(7, len(ARROW_COLORS)))
    colors = random.sample(ARROW_COLORS, arrow_count)

    arrows: list[dict] = []
    color_idx = 0

    for pair_idx in range(required_collisions):
        _occ, blocked = _occupied_and_blocked(arrows)
        pair = _sample_collision_pair(
            shape,
            colors[color_idx],
            colors[color_idx + 1],
            blocked,
            first_pair=(pair_idx == 0),
        )
        if pair is None:
            return None
        arrows.extend(pair)
        color_idx += 2

    while len(arrows) < arrow_count:
        _occ, blocked = _occupied_and_blocked(arrows)
        direction_pool: list[tuple[int, int]] | None = None
        current_families = {_family(arrow["direction"]) for arrow in arrows}
        if "axis" not in current_families:
            direction_pool = AXIS_DIRECTIONS
        elif "diag" not in current_families:
            direction_pool = DIAGONAL_DIRECTIONS

        size_pool: list[int] | None = None
        existing_sizes = {arrow["size"] for arrow in arrows}
        if len(existing_sizes) == 1:
            size_pool = [size for size in SIZES if size not in existing_sizes]

        spec = _sample_random_arrow(
            shape,
            colors[color_idx],
            blocked,
            direction_pool=direction_pool,
            size_pool=size_pool,
        )
        if spec is None:
            return None
        arrows.append(spec)
        color_idx += 1

    if not _example_is_valid(shape, arrows, required_collisions):
        return None
    return _render_input(shape, arrows)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pulse-ray arrow grid")
