from __future__ import annotations

from collections import Counter
import random

import numpy as np


GREY = 5
BODY_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
BG_COLORS = (0, 1, 2, 3, 4, 6, 7, 8, 9)
DIRECTIONS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
HIT_FAMILIES = (
    ((-1, 0), (1, 0)),
    ((0, -1), (0, 1)),
    ((-1, -1), (1, 1)),
    ((-1, 1), (1, -1)),
)

MIN_SIDE = 18
MAX_SIDE = 28
TARGET_MARGIN = 6
MIN_TARGET_STEP = 3
MIN_RAY_LEN = 3
MAX_GENERATION_ATTEMPTS = 500
SCENE_TYPES = ((2, 1), (3, 1), (3, 2), (4, 1), (4, 2))

_CANDIDATE_CACHE: dict[tuple[int, int], tuple[dict, ...]] = {}


def get_bg(grid: np.ndarray) -> int:
    """Most common color in the grid."""
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    """Return True iff (r, c) lies inside a grid of shape (H, W)."""
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _ray_cells(shape: tuple[int, int], start: tuple[int, int], direction: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    cells: list[tuple[int, int]] = []
    r, c = map(int, start)
    dr, dc = map(int, direction)
    while in_bounds(shape, r, c):
        cells.append((r, c))
        r += dr
        c += dc
    return tuple(cells)


def _dilate8(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if in_bounds(shape, rr, cc):
                    out.add((rr, cc))
    return out


def _make_candidate(shape: tuple[int, int], nozzle: tuple[int, int], direction: tuple[int, int]) -> dict | None:
    dr, dc = map(int, direction)
    nr, nc = map(int, nozzle)
    body = []
    for k in range(1, 4):
        rr = nr - k * dr
        cc = nc - k * dc
        if not in_bounds(shape, rr, cc):
            return None
        body.append((rr, cc))
    ray = _ray_cells(shape, (nr + dr, nc + dc), direction)
    if len(ray) < MIN_RAY_LEN:
        return None
    occ = tuple(body + [(nr, nc)])
    return {
        "dir": (dr, dc),
        "nozzle": (nr, nc),
        "body": tuple(body),
        "occ": occ,
        "occ_set": frozenset(occ),
        "ray": ray,
        "ray_set": frozenset(ray),
    }


def _all_candidates(shape: tuple[int, int]) -> tuple[dict, ...]:
    key = tuple(map(int, shape))
    cached = _CANDIDATE_CACHE.get(key)
    if cached is not None:
        return cached

    h, w = key
    candidates: list[dict] = []
    for r in range(h):
        for c in range(w):
            for direction in DIRECTIONS8:
                cand = _make_candidate(key, (r, c), direction)
                if cand is not None:
                    candidates.append(cand)
    _CANDIDATE_CACHE[key] = tuple(candidates)
    return _CANDIDATE_CACHE[key]


def _extract_sprite_for_color(grid: np.ndarray, color: int) -> dict:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == int(color))]
    assert len(coords) == 3, "each pointer body must contain exactly 3 cells"
    coord_set = set(coords)

    solutions: dict[tuple[tuple[int, int], tuple[int, int]], dict] = {}
    for start in coords:
        sr, sc = start
        for dr, dc in DIRECTIONS8:
            expected = {(sr + k * dr, sc + k * dc) for k in range(3)}
            if expected != coord_set:
                continue

            end = (sr + 2 * dr, sc + 2 * dc)
            nozzle_a = (sr - dr, sc - dc)
            nozzle_b = (end[0] + dr, end[1] + dc)

            if in_bounds(grid.shape, nozzle_a[0], nozzle_a[1]) and int(grid[nozzle_a]) == GREY:
                ray_dir = (-dr, -dc)
                key = (tuple(map(int, nozzle_a)), tuple(map(int, ray_dir)))
                body = tuple((nozzle_a[0] - k * ray_dir[0], nozzle_a[1] - k * ray_dir[1]) for k in range(1, 4))
                solutions[key] = {
                    "color": int(color),
                    "body": body,
                    "nozzle": tuple(map(int, nozzle_a)),
                    "dir": tuple(map(int, ray_dir)),
                }

            if in_bounds(grid.shape, nozzle_b[0], nozzle_b[1]) and int(grid[nozzle_b]) == GREY:
                ray_dir = (dr, dc)
                key = (tuple(map(int, nozzle_b)), tuple(map(int, ray_dir)))
                body = tuple((nozzle_b[0] - k * ray_dir[0], nozzle_b[1] - k * ray_dir[1]) for k in range(1, 4))
                solutions[key] = {
                    "color": int(color),
                    "body": body,
                    "nozzle": tuple(map(int, nozzle_b)),
                    "dir": tuple(map(int, ray_dir)),
                }

    assert len(solutions) == 1, "each color must form one unambiguous pointer sprite"
    sprite = next(iter(solutions.values()))
    sprite["ray"] = _ray_cells(grid.shape, (sprite["nozzle"][0] + sprite["dir"][0], sprite["nozzle"][1] + sprite["dir"][1]), sprite["dir"])
    return sprite


def _parse_scene(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = get_bg(g)
    assert bg != GREY, "grey is reserved for nozzles and the target cell"

    colors = sorted(set(map(int, np.unique(g))) - {bg, GREY})
    assert 3 <= len(colors) <= 6, "expected several uniquely colored pointer bodies"

    sprites = [_extract_sprite_for_color(g, color) for color in colors]
    grey_cells = {tuple(map(int, rc)) for rc in np.argwhere(g == GREY)}
    used_nozzles = {sprite["nozzle"] for sprite in sprites}
    assert grey_cells == used_nozzles, "every grey cell must be exactly one nozzle"
    return bg, sprites


def _coverage_counts(grid: np.ndarray, bg: int, sprites: list[dict]) -> np.ndarray:
    counts = np.zeros(grid.shape, dtype=int)
    for sprite in sprites:
        for r, c in sprite["ray"]:
            if int(grid[r, c]) == bg:
                counts[r, c] += 1
    return counts


def _solve_scene(grid: np.ndarray) -> tuple[int, list[dict], tuple[int, int], list[dict]]:
    bg, sprites = _parse_scene(grid)
    counts = _coverage_counts(grid, bg, sprites)
    bg_mask = np.asarray(grid == bg)
    assert bool(np.any(bg_mask)), "expected at least one background cell"
    max_cover = int(counts[bg_mask].max())
    assert max_cover > 0, "at least one background cell must be covered"
    target_cells = [tuple(map(int, rc)) for rc in np.argwhere(bg_mask & (counts == max_cover))]
    assert len(target_cells) == 1, "the most-covered background cell must be unique"
    target = target_cells[0]

    hit_sprites = [sprite for sprite in sprites if target in set(sprite["ray"])]
    assert len(hit_sprites) == max_cover, "the target cover count must match the number of hitting rays"
    assert 1 <= len(sprites) - len(hit_sprites) <= 2, "expected one or two decoys"
    assert len(hit_sprites) > len(sprites) // 2, "the target must be supported by a majority of sprites"
    return bg, sprites, target, hit_sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the uniquely most-covered sightline cell and draw only the rays that hit it.

    Input structure:
    - A dominant background contains several pointer sprites.
    - Each sprite is exactly three same-color body cells in a straight 8-direction line plus one adjacent grey nozzle on one end's outward extension.
    - Each body color is unique; grey appears only as nozzles.

    Rule:
    - For every sprite, extend a ray from the nozzle outward in the body's pointing direction and count how many rays pass through each background cell.
    - Find the unique background cell with maximum coverage.
    - Preserve the input, then for each ray that hits that cell paint the segment from the first cell after the nozzle up to the cell just before the target using the sprite's body color; paint the target cell grey; leave decoys unchanged.

    Key insight:
    - The answer is not a common intersection of all rays because one or two decoys miss; instead, the hidden target is the unique background cell crossed by the most sightlines.
    """
    g = np.asarray(grid, dtype=int)
    _bg, _sprites, target, hit_sprites = _solve_scene(g)
    out = g.copy()

    for sprite in hit_sprites:
        dr, dc = sprite["dir"]
        r = sprite["nozzle"][0] + dr
        c = sprite["nozzle"][1] + dc
        while (r, c) != target:
            out[r, c] = sprite["color"]
            r += dr
            c += dc

    out[target] = GREY
    return out


def _hit_candidates(shape: tuple[int, int], target: tuple[int, int], direction: tuple[int, int]) -> list[dict]:
    dr, dc = map(int, direction)
    tr, tc = map(int, target)
    candidates = []
    for step in range(MIN_TARGET_STEP, max(shape) + 1):
        nozzle = (tr - step * dr, tc - step * dc)
        cand = _make_candidate(shape, nozzle, direction)
        if cand is None:
            continue
        if target not in cand["ray_set"]:
            continue
        cand = dict(cand)
        cand["hits_target"] = True
        cand["target_step"] = step
        candidates.append(cand)
    random.shuffle(candidates)
    candidates.sort(key=lambda item: (-int(item["target_step"]), int(item["nozzle"][0]), int(item["nozzle"][1])))
    return candidates


def _pick_hit_sprites(shape: tuple[int, int], target: tuple[int, int], hit_dirs: list[tuple[int, int]]) -> list[dict] | None:
    selected: list[dict] = []

    def rec(i: int, occ_set: set[tuple[int, int]], blocked_occ: set[tuple[int, int]], ray_union: set[tuple[int, int]]) -> list[dict] | None:
        if i == len(hit_dirs):
            return [dict(sprite) for sprite in selected]

        for cand in _hit_candidates(shape, target, hit_dirs[i]):
            if cand["occ_set"] & blocked_occ:
                continue
            if cand["ray_set"] & occ_set:
                continue
            if cand["occ_set"] & ray_union:
                continue
            ok = True
            for other in selected:
                if (cand["ray_set"] & other["ray_set"]) != {target}:
                    ok = False
                    break
            if not ok:
                continue

            selected.append(cand)
            next_occ = occ_set | set(cand["occ_set"])
            next_blocked = blocked_occ | _dilate8(set(cand["occ_set"]), shape)
            next_rays = ray_union | set(cand["ray_set"])
            found = rec(i + 1, next_occ, next_blocked, next_rays)
            if found is not None:
                return found
            selected.pop()

        return None

    return rec(0, set(), set(), set())


def _pick_decoys(shape: tuple[int, int], target: tuple[int, int], count: int, hit_sprites: list[dict]) -> list[dict] | None:
    hit_occ = set().union(*(set(sprite["occ_set"]) for sprite in hit_sprites))
    blocked_occ = _dilate8(hit_occ, shape)
    ray_union = set().union(*(set(sprite["ray_set"]) for sprite in hit_sprites))

    candidates = []
    for cand in _all_candidates(shape):
        if target in cand["ray_set"]:
            continue
        if cand["occ_set"] & blocked_occ:
            continue
        if cand["ray_set"] & hit_occ:
            continue
        if cand["occ_set"] & ray_union:
            continue
        if cand["ray_set"] & ray_union:
            continue
        candidates.append(dict(cand))

    random.shuffle(candidates)
    selected: list[dict] = []

    def rec(start_idx: int, occ_set: set[tuple[int, int]], blocked: set[tuple[int, int]], rays: set[tuple[int, int]]) -> list[dict] | None:
        if len(selected) == count:
            return [dict(sprite) for sprite in selected]

        for idx in range(start_idx, len(candidates)):
            cand = candidates[idx]
            if cand["occ_set"] & blocked:
                continue
            if cand["ray_set"] & occ_set:
                continue
            if cand["occ_set"] & rays:
                continue
            if cand["ray_set"] & rays:
                continue

            selected.append(cand)
            next_occ = occ_set | set(cand["occ_set"])
            next_blocked = blocked | _dilate8(set(cand["occ_set"]), shape)
            next_rays = rays | set(cand["ray_set"])
            found = rec(idx + 1, next_occ, next_blocked, next_rays)
            if found is not None:
                return found
            selected.pop()

        return None

    return rec(0, set(hit_occ), set(blocked_occ), set(ray_union))


def _render_scene(shape: tuple[int, int], bg: int, sprites: list[dict], colors: list[int]) -> np.ndarray:
    grid = np.full(shape, int(bg), dtype=int)
    for sprite, color in zip(sprites, colors):
        for r, c in sprite["body"]:
            grid[r, c] = int(color)
        nr, nc = sprite["nozzle"]
        grid[nr, nc] = GREY
    return grid


def _expected_from_specs(grid: np.ndarray, sprites: list[dict], colors: list[int], target: tuple[int, int], hit_count: int) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for sprite, color in zip(sprites[:hit_count], colors[:hit_count]):
        dr, dc = sprite["dir"]
        r = sprite["nozzle"][0] + dr
        c = sprite["nozzle"][1] + dc
        while (r, c) != target:
            out[r, c] = int(color)
            r += dr
            c += dc
    out[target] = GREY
    return out


def _validate_generated_scene(grid: np.ndarray, sprites: list[dict], target: tuple[int, int], hit_count: int) -> None:
    bg = get_bg(grid)
    assert bg != GREY
    counts = np.zeros(grid.shape, dtype=int)
    for sprite in sprites:
        for r, c in sprite["ray"]:
            assert int(grid[r, c]) == bg, "all ray cells must stay background in generated inputs"
            counts[r, c] += 1

    max_cover = int(counts.max())
    assert max_cover == hit_count, "generated target coverage must equal the number of true hit rays"
    winners = [tuple(map(int, rc)) for rc in np.argwhere(counts == max_cover)]
    assert winners == [target], "generated scene must have a unique maximum-coverage cell"

    for i in range(hit_count):
        assert target in sprites[i]["ray_set"]
    for i in range(hit_count, len(sprites)):
        assert target not in sprites[i]["ray_set"]


def _generate_once() -> np.ndarray | None:
    h = random.randint(MIN_SIDE, MAX_SIDE)
    w = random.randint(MIN_SIDE, MAX_SIDE)
    shape = (h, w)

    target = (
        random.randint(TARGET_MARGIN, h - TARGET_MARGIN - 1),
        random.randint(TARGET_MARGIN, w - TARGET_MARGIN - 1),
    )
    hit_count, decoy_count = random.choice(SCENE_TYPES)
    hit_dirs = [random.choice(family) for family in random.sample(HIT_FAMILIES, hit_count)]

    hit_sprites = _pick_hit_sprites(shape, target, hit_dirs)
    if hit_sprites is None:
        return None
    decoys = _pick_decoys(shape, target, decoy_count, hit_sprites)
    if decoys is None:
        return None

    sprites = hit_sprites + decoys
    bg = random.choice(BG_COLORS)
    usable_colors = [color for color in BODY_COLORS if color != bg]
    colors = random.sample(usable_colors, len(sprites))
    for sprite, color in zip(sprites, colors):
        sprite["color"] = int(color)

    grid = _render_scene(shape, bg, sprites, colors)
    _validate_generated_scene(grid, sprites, target, hit_count)
    expected = _expected_from_specs(grid, sprites, colors, target, hit_count)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sightline-convergence input")
