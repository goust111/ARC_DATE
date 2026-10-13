from __future__ import annotations

import random
from collections import deque

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
BG = 0
SPRITE_SIZE = 3
GRID_RANGE = (18, 25)
SPRITE_COUNT_CHOICES = (2, 3)
COPY_COUNT_CHOICES = (2, 2, 3, 3, 4)
MAX_ATTEMPTS = 400


def _component_bfs(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_contains_point(bbox: tuple[int, int, int, int], point: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    r, c = point
    return r0 <= r <= r1 and c0 <= c <= c1


def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    row_gap = max(0, br0 - ar1 - 1, ar0 - br1 - 1)
    col_gap = max(0, bc0 - ac1 - 1, ac0 - bc1 - 1)
    return int(row_gap + col_gap)


def _sprite_patch(body: int, center: int) -> np.ndarray:
    patch = np.full((3, 3), int(body), dtype=int)
    patch[1, 1] = int(center)
    return patch


def _is_sprite_patch(patch: np.ndarray) -> bool:
    if patch.shape != (3, 3):
        return False
    border = patch.copy()
    center = int(border[1, 1])
    border[1, 1] = -1
    vals = {int(v) for v in border.flatten() if int(v) != -1}
    return len(vals) == 1 and BG not in vals and center != BG and center != next(iter(vals))


def _extract_objects(grid: np.ndarray) -> tuple[list[dict], dict[int, tuple[int, int]]]:
    comps = _component_bfs(grid != BG)
    sprites: list[dict] = []
    targets: dict[int, tuple[int, int]] = {}
    for coords in comps:
        bbox = _bbox_from_coords(coords)
        area = int(coords.shape[0])
        if area == 1:
            r, c = map(int, coords[0])
            color = int(grid[r, c])
            assert color not in targets, "each target color must appear exactly once outside sprites"
            targets[color] = (r, c)
            continue
        if area != 9 or bbox[2] - bbox[0] != 2 or bbox[3] - bbox[1] != 2:
            raise AssertionError("expected only 3x3 sprites and single-pixel targets")
        patch = grid[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1]
        assert np.count_nonzero(patch) == 9 and _is_sprite_patch(patch)
        sprites.append(
            {
                "bbox": bbox,
                "patch": patch.copy(),
                "body": int(patch[0, 0]),
                "center": int(patch[1, 1]),
            }
        )

    assert 2 <= len(sprites) <= 3
    sprite_colors = {sprite["center"] for sprite in sprites}
    assert sprite_colors == set(targets), "sprite center colors must match target colors one-to-one"
    return sprites, targets


def _direction_to_target(
    sprite_bbox: tuple[int, int, int, int], target: tuple[int, int]
) -> tuple[int, int]:
    r0, c0, r1, c1 = sprite_bbox
    tr, tc = target
    directions: list[tuple[int, int]] = []
    if r0 <= tr <= r1:
        if tc > c1:
            directions.append((0, 1))
        if tc < c0:
            directions.append((0, -1))
    if c0 <= tc <= c1:
        if tr > r1:
            directions.append((1, 0))
        if tr < r0:
            directions.append((-1, 0))
    assert len(directions) == 1, "target must define one cardinal stamping direction"
    return directions[0]


def _stamped_positions(
    sprite_bbox: tuple[int, int, int, int], target: tuple[int, int], shape: tuple[int, int]
) -> list[tuple[int, int]]:
    h, w = shape
    dr, dc = _direction_to_target(sprite_bbox, target)
    r, c = sprite_bbox[0], sprite_bbox[1]
    positions = [(r, c)]
    assert not _bbox_contains_point((r, c, r + 2, c + 2), target)
    while True:
        cur_bbox = (r, c, r + 2, c + 2)
        if _bbox_contains_point(cur_bbox, target):
            break
        r += 3 * dr
        c += 3 * dc
        assert 0 <= r <= h - 3 and 0 <= c <= w - 3
        positions.append((r, c))
    return positions


def main(grid: np.ndarray) -> np.ndarray:
    """Generate Voronoi parity coloring from the centers of straight-line sprite stamps.

    Input structure:
    - The grid is black background with 2-3 separate 3x3 sprites and one single-pixel target for each sprite center color.
    - Every sprite has a uniform non-black border color and a different non-black center color.
    - Each target pixel shares its color with exactly one sprite center and lies in that sprite's unique cardinal row-band or column-band.

    Rule:
    - Pair every sprite with the single target pixel whose color matches the sprite center.
    - Move the sprite toward its target in 3-cell steps, stamping full 3x3 copies until the first copy whose bbox overlaps the target pixel; collect the centers of the original sprite and all copies as colored seeds.
    - Assign every cell to its unique nearest seed by squared Euclidean distance, leave ties black, color only even Chebyshev-distance cells with the seed color, then restore all sprite pixels and keep every target pixel unchanged.

    Key insight:
    - The Voronoi seeds are not explicit in the input; they must be reconstructed from the hidden stamping trajectory before the even-ring ownership pattern can be drawn.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    assert int(np.min(g)) >= 0
    sprites, targets = _extract_objects(g)

    h, w = g.shape
    rr = np.arange(h)[:, None]
    cc = np.arange(w)[None, :]

    seeds: list[tuple[int, int, int]] = []
    stamps: list[tuple[int, int, np.ndarray]] = []
    for sprite in sprites:
        positions = _stamped_positions(sprite["bbox"], targets[sprite["center"]], g.shape)
        for r0, c0 in positions:
            seeds.append((r0 + 1, c0 + 1, sprite["center"]))
            stamps.append((r0, c0, sprite["patch"]))

    dists = np.stack(
        [((rr - sr) ** 2 + (cc - sc) ** 2) for sr, sc, _color in seeds],
        axis=0,
    )
    best = dists.min(axis=0)
    winners = dists == best
    unique = winners.sum(axis=0) == 1
    owner = winners.argmax(axis=0)

    out = np.zeros_like(g)
    for idx, (sr, sc, color) in enumerate(seeds):
        cheb = np.maximum(np.abs(rr - sr), np.abs(cc - sc))
        mask = unique & (owner == idx) & (cheb % 2 == 0)
        out[mask] = int(color)

    for r0, c0, patch in stamps:
        out[r0 : r0 + 3, c0 : c0 + 3] = patch
    for color, (tr, tc) in targets.items():
        out[tr, tc] = int(color)
    return out


def _candidate_spec(
    h: int, w: int, direction: tuple[int, int], copy_count: int
) -> tuple[int, int, int, int, int, int, int, int, int] | None:
    valid_jitters = [j for j in (-1, 0, 1) if 3 * copy_count + j - 1 >= 6]
    jitter = random.choice(valid_jitters)
    dr, dc = direction
    if (dr, dc) == (0, 1):
        r0 = random.randint(0, h - 3)
        c0 = random.randint(0, w - (3 * copy_count + 3))
        tr = r0 + 1
        tc = c0 + 1 + 3 * copy_count + jitter
        cr0, cc0, cr1, cc1 = r0, c0, r0 + 2, c0 + 3 * copy_count + 2
    elif (dr, dc) == (0, -1):
        r0 = random.randint(0, h - 3)
        c0 = random.randint(3 * copy_count, w - 3)
        tr = r0 + 1
        tc = c0 + 1 - (3 * copy_count + jitter)
        cr0, cc0, cr1, cc1 = r0, c0 - 3 * copy_count, r0 + 2, c0 + 2
    elif (dr, dc) == (1, 0):
        r0 = random.randint(0, h - (3 * copy_count + 3))
        c0 = random.randint(0, w - 3)
        tr = r0 + 1 + 3 * copy_count + jitter
        tc = c0 + 1
        cr0, cc0, cr1, cc1 = r0, c0, r0 + 3 * copy_count + 2, c0 + 2
    else:
        r0 = random.randint(3 * copy_count, h - 3)
        c0 = random.randint(0, w - 3)
        tr = r0 + 1 - (3 * copy_count + jitter)
        tc = c0 + 1
        cr0, cc0, cr1, cc1 = r0 - 3 * copy_count, c0, r0 + 2, c0 + 2
    return r0, c0, tr, tc, cr0, cc0, cr1, cc1, jitter


def _path_positions(
    top_left: tuple[int, int], direction: tuple[int, int], copy_count: int
) -> list[tuple[int, int]]:
    r0, c0 = top_left
    dr, dc = direction
    return [(r0 + 3 * dr * k, c0 + 3 * dc * k) for k in range(copy_count + 1)]


def _valid_with_existing(candidate: dict, placed: list[dict]) -> bool:
    sprite_box = candidate["bbox"]
    target_box = candidate["target_box"]
    corridor = candidate["corridor"]
    for other in placed:
        if _bbox_gap(sprite_box, other["bbox"]) < 6:
            return False
        if _bbox_gap(sprite_box, other["target_box"]) < 6:
            return False
        if _bbox_gap(other["bbox"], target_box) < 6:
            return False
        if _bbox_gap(target_box, other["target_box"]) < 6:
            return False
        if _bbox_gap(corridor, other["corridor"]) < 1:
            return False
        if _bbox_gap(corridor, other["target_box"]) < 1:
            return False
        if _bbox_gap(other["corridor"], target_box) < 1:
            return False
    return True


def _nontrivial_output(inp: np.ndarray, out: np.ndarray, specs: list[dict]) -> bool:
    if np.array_equal(inp, out):
        return False
    seed_colors = {spec["center"] for spec in specs}
    if len(seed_colors & set(map(int, np.unique(out)))) < 2:
        return False
    covered = np.zeros_like(inp, dtype=bool)
    for spec in specs:
        for r0, c0 in spec["positions"]:
            covered[r0 : r0 + 3, c0 : c0 + 3] = True
        tr, tc = spec["target"]
        covered[tr, tc] = True
    painted = (out != BG) & ~covered
    return int(painted.sum()) >= 20


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        h = random.randint(*GRID_RANGE)
        w = random.randint(*GRID_RANGE)
        sprite_count = random.choice(SPRITE_COUNT_CHOICES)
        colors = random.sample(range(1, 10), 2 * sprite_count)
        center_colors = colors[:sprite_count]
        body_colors = colors[sprite_count:]

        specs: list[dict] = []
        shuffled_dirs = random.sample([(0, 1), (0, -1), (1, 0), (-1, 0)], 4)
        for idx in range(sprite_count):
            choices = shuffled_dirs[:]
            random.shuffle(choices)
            placed = False
            for _inner in range(120):
                direction = choices[_inner % len(choices)] if _inner < len(choices) else random.choice(choices)
                copy_count = random.choice(COPY_COUNT_CHOICES)
                candidate_raw = _candidate_spec(h, w, direction, copy_count)
                if candidate_raw is None:
                    continue
                r0, c0, tr, tc, cr0, cc0, cr1, cc1, _jitter = candidate_raw
                candidate = {
                    "bbox": (r0, c0, r0 + 2, c0 + 2),
                    "target_box": (tr, tc, tr, tc),
                    "target": (tr, tc),
                    "corridor": (cr0, cc0, cr1, cc1),
                    "direction": direction,
                    "copy_count": copy_count,
                    "center": center_colors[idx],
                    "body": body_colors[idx],
                }
                if not _valid_with_existing(candidate, specs):
                    continue
                candidate["positions"] = _path_positions((r0, c0), direction, copy_count)
                specs.append(candidate)
                placed = True
                break
            if not placed:
                break
        if len(specs) != sprite_count:
            continue

        grid = np.zeros((h, w), dtype=int)
        for spec in specs:
            r0, c0, _r1, _c1 = spec["bbox"]
            grid[r0 : r0 + 3, c0 : c0 + 3] = _sprite_patch(spec["body"], spec["center"])
            tr, tc = spec["target"]
            grid[tr, tc] = int(spec["center"])

        try:
            out = main(grid)
        except AssertionError:
            continue
        if _nontrivial_output(grid, out, specs):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
