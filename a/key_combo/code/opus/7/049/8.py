from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
SPRITE_COLORS = (2, 3, 4, 5, 6, 7, 8, 9)
HALF_DISTANCES = (2, 3, 4)
MAX_ATTEMPTS = 600


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int) -> Iterable[tuple[int, int]]:
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        offsets = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]

    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in offsets:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste_sprite(dst: np.ndarray, sprite: np.ndarray, center: tuple[int, int]) -> None:
    r0 = int(center[0]) - 1
    c0 = int(center[1]) - 1
    mask = sprite != BLACK
    dst[r0 : r0 + 3, c0 : c0 + 3][mask] = sprite[mask]


def _boxes_touch(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0
    )


def _point_near_box(point: tuple[int, int], box: tuple[int, int, int, int], gap: int = 0) -> bool:
    r, c = point
    r0, c0, r1, c1 = box
    return r0 - gap <= r <= r1 + gap and c0 - gap <= c <= c1 + gap


def _build_sprite_masks() -> dict[int, tuple[np.ndarray, ...]]:
    masks: dict[int, list[np.ndarray]] = {3: [], 4: [], 5: []}
    for bits in range(1 << 9):
        area = bits.bit_count()
        if area not in masks:
            continue
        mask = np.zeros((3, 3), dtype=bool)
        for idx in range(9):
            if bits & (1 << idx):
                mask[idx // 3, idx % 3] = True
        if not mask[0].any() or not mask[2].any() or not mask[:, 0].any() or not mask[:, 2].any():
            continue
        if len(_components(mask, connectivity=4)) != 1:
            continue
        masks[area].append(mask)
    return {k: tuple(v) for k, v in masks.items()}


SPRITE_MASKS = _build_sprite_masks()
VALID_SPRITE_AREAS = tuple(area for area, masks in SPRITE_MASKS.items() if masks)


def _find_sprite(grid: np.ndarray) -> np.ndarray:
    mask = (grid != BLACK) & (grid != BLUE)
    blue_mask = grid == BLUE
    candidates: list[np.ndarray] = []
    for comp in _components(mask, connectivity=4):
        if any(
            0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and blue_mask[nr, nc]
            for r, c in comp
            for nr, nc in _neighbors8(r, c)
        ):
            continue
        box = _bbox(comp)
        if (box[2] - box[0] + 1, box[3] - box[1] + 1) != (3, 3):
            continue
        sprite = _crop(grid, box)
        area = int(np.count_nonzero(sprite != BLACK))
        if 3 <= area <= 5:
            candidates.append(sprite)
    assert len(candidates) == 1, "expected one isolated 3x3 sprite"
    return candidates[0]


def _blue_pair_candidates(blue_coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    blue_set = set(blue_coords)
    edges: list[tuple[int, int]] = []
    for i, (r1, c1) in enumerate(blue_coords):
        for j in range(i + 1, len(blue_coords)):
            r2, c2 = blue_coords[j]
            if r1 == r2:
                dist = abs(c1 - c2)
                if dist == 0 or dist % 2:
                    continue
                lo, hi = sorted((c1, c2))
                if any((r1, cc) in blue_set for cc in range(lo + 1, hi)):
                    continue
                edges.append((i, j))
            elif c1 == c2:
                dist = abs(r1 - r2)
                if dist == 0 or dist % 2:
                    continue
                lo, hi = sorted((r1, r2))
                if any((rr, c1) in blue_set for rr in range(lo + 1, hi)):
                    continue
                edges.append((i, j))
    return edges


def _unique_blue_pairs(grid: np.ndarray, expected_pair_count: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    blue_coords = sorted((int(r), int(c)) for r, c in np.argwhere(grid == BLUE))
    assert len(blue_coords) == 2 * expected_pair_count, "unexpected number of blue pixels"

    edges = _blue_pair_candidates(blue_coords)
    neighbor_map: dict[int, list[int]] = {i: [] for i in range(len(blue_coords))}
    for i, j in edges:
        neighbor_map[i].append(j)
        neighbor_map[j].append(i)

    solutions: list[list[tuple[int, int]]] = []

    def search(used: set[int], chosen: list[tuple[int, int]]) -> None:
        if len(solutions) > 1:
            return
        if len(used) == len(blue_coords):
            if len(chosen) == expected_pair_count:
                solutions.append(chosen.copy())
            return
        remaining = [idx for idx in range(len(blue_coords)) if idx not in used]
        pivot = min(remaining, key=lambda idx: sum(1 for j in neighbor_map[idx] if j not in used))
        candidates = [j for j in neighbor_map[pivot] if j not in used]
        for other in candidates:
            used.add(pivot)
            used.add(other)
            chosen.append((pivot, other))
            search(used, chosen)
            chosen.pop()
            used.remove(other)
            used.remove(pivot)

    search(set(), [])
    assert len(solutions) == 1, "expected a unique blue pairing"

    pairs = []
    for i, j in solutions[0]:
        a = blue_coords[i]
        b = blue_coords[j]
        pairs.append(tuple(sorted((a, b))))
    return sorted(pairs, key=lambda pair: ((pair[0][0] + pair[1][0]) // 2, (pair[0][1] + pair[1][1]) // 2))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the isolated 3x3 sprite to the midpoints of a matching number of blue locator pairs.

    Input structure:
    - The grid has black background, one isolated non-black/non-blue 4-connected sprite,
      and several blue locator pixels.
    - The sprite occupies a 3x3 bounding box and uses one foreground color with black holes allowed.
    - The blue pixels form unique same-row or same-column pairs with even spacing and no blue
      pixels between the paired endpoints.

    Rule:
    - Find the unique isolated sprite and crop its 3x3 bounding box.
    - Count the sprite's non-black pixels; this count K equals the number of blue pairs.
    - Recover the unique K blue pairs, compute each pair's midpoint, and paste the cropped sprite
      centered on every midpoint.
    - Keep the original grid everywhere else; sprite foreground overwrites any blue pixels it covers.

    Key insight:
    - The source sprite carries two pieces of information at once: its shape is the copied pattern,
      and its foreground-pixel count determines how many midpoint placements must be found.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sprite = _find_sprite(g)
    pair_count = int(np.count_nonzero(sprite != BLACK))
    blue_pairs = _unique_blue_pairs(g, pair_count)

    out = g.copy()
    for (r1, c1), (r2, c2) in blue_pairs:
        center = ((r1 + r2) // 2, (c1 + c2) // 2)
        _paste_sprite(out, sprite, center)
    return out


def _random_sprite(rng: random.Random) -> tuple[np.ndarray, int]:
    area = rng.choice(VALID_SPRITE_AREAS)
    color = rng.choice(SPRITE_COLORS)
    mask = rng.choice(SPRITE_MASKS[area])
    sprite = np.zeros((3, 3), dtype=int)
    sprite[mask] = color
    return sprite, area


def _sprite_origin_options(length: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    top = tuple(range(0, min(3, length - 2)))
    bottom_start = max(0, length - 5)
    bottom = tuple(range(bottom_start, length - 2))
    return top, bottom


def _choose_sprite_box(h: int, w: int, rng: random.Random) -> tuple[int, int, int, int]:
    top_rows, bottom_rows = _sprite_origin_options(h)
    left_cols, right_cols = _sprite_origin_options(w)
    corner = rng.choice(("tl", "tr", "bl", "br"))
    if corner == "tl":
        r0 = rng.choice(top_rows)
        c0 = rng.choice(left_cols)
    elif corner == "tr":
        r0 = rng.choice(top_rows)
        c0 = rng.choice(right_cols)
    elif corner == "bl":
        r0 = rng.choice(bottom_rows)
        c0 = rng.choice(left_cols)
    else:
        r0 = rng.choice(bottom_rows)
        c0 = rng.choice(right_cols)
    return r0, c0, r0 + 2, c0 + 2


def _all_pair_specs(
    h: int,
    w: int,
    sprite_box: tuple[int, int, int, int],
) -> list[dict]:
    specs: list[dict] = []
    sprite_cells = {
        (r, c)
        for r in range(sprite_box[0], sprite_box[2] + 1)
        for c in range(sprite_box[1], sprite_box[3] + 1)
    }
    for orientation in ("h", "v"):
        for half in HALF_DISTANCES:
            for r in range(1, h - 1):
                for c in range(1, w - 1):
                    patch_box = (r - 1, c - 1, r + 1, c + 1)
                    if _boxes_touch(patch_box, sprite_box, gap=1):
                        continue
                    if orientation == "h":
                        a = (r, c - half)
                        b = (r, c + half)
                    else:
                        a = (r - half, c)
                        b = (r + half, c)
                    if min(a[0], b[0]) < 0 or min(a[1], b[1]) < 0 or max(a[0], b[0]) >= h or max(a[1], b[1]) >= w:
                        continue
                    if any(_point_near_box(pt, sprite_box, gap=1) for pt in (a, b)):
                        continue
                    if a in sprite_cells or b in sprite_cells:
                        continue
                    specs.append(
                        {
                            "orientation": orientation,
                            "half": half,
                            "center": (r, c),
                            "pair": tuple(sorted((a, b))),
                            "patch_box": patch_box,
                        }
                    )
    return specs


def _orientation_plan(pair_count: int, rng: random.Random) -> list[str]:
    plan = [rng.choice(("h", "v")) for _ in range(pair_count)]
    if pair_count >= 4 and rng.random() < 0.7:
        plan[0] = "h"
        plan[1] = "v"
        rng.shuffle(plan)
    return plan


def _compatible(spec: dict, chosen: list[dict], sprite_box: tuple[int, int, int, int]) -> bool:
    if _boxes_touch(spec["patch_box"], sprite_box, gap=1):
        return False
    pair_points = set(spec["pair"])
    for other in chosen:
        if _boxes_touch(spec["patch_box"], other["patch_box"], gap=1):
            return False
        if pair_points & set(other["pair"]):
            return False
    return True


def _sample_pair_specs(
    h: int,
    w: int,
    sprite_box: tuple[int, int, int, int],
    pair_count: int,
    rng: random.Random,
) -> list[dict] | None:
    specs = _all_pair_specs(h, w, sprite_box)
    if len(specs) < pair_count:
        return None

    for _ in range(200):
        chosen: list[dict] = []
        plan = _orientation_plan(pair_count, rng)
        for want in plan:
            candidates = [spec for spec in specs if spec["orientation"] == want and _compatible(spec, chosen, sprite_box)]
            if not candidates:
                candidates = [spec for spec in specs if _compatible(spec, chosen, sprite_box)]
            if not candidates:
                break
            chosen.append(rng.choice(candidates))
        if len(chosen) != pair_count:
            continue

        blue_points = [pt for spec in chosen for pt in spec["pair"]]
        probe = np.zeros((h, w), dtype=int)
        for r, c in blue_points:
            probe[r, c] = BLUE
        try:
            recovered = _unique_blue_pairs(probe, pair_count)
        except AssertionError:
            continue
        if sorted(tuple(sorted(spec["pair"])) for spec in chosen) != recovered:
            continue
        return chosen
    return None


def _attempt_generate(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(15, 23)
    w = rng.randint(15, 23)
    sprite, pair_count = _random_sprite(rng)
    sprite_box = _choose_sprite_box(h, w, rng)
    pair_specs = _sample_pair_specs(h, w, sprite_box, pair_count, rng)
    if pair_specs is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    r0, c0, r1, c1 = sprite_box
    grid[r0 : r1 + 1, c0 : c1 + 1] = sprite
    for spec in pair_specs:
        for r, c in spec["pair"]:
            grid[r, c] = BLUE

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if out.shape != grid.shape or max(out.shape) > 30:
        return None

    changed = np.argwhere(out != grid)
    if changed.shape[0] < pair_count:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt_generate(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
