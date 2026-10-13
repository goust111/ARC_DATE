from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
GREEN = 3
ORANGE = 7
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _line_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    (r0, c0), (r1, c1) = a, b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]
    raise ValueError("line segment must be axis-aligned")


def _paint_polyline(mask: np.ndarray, points: list[tuple[int, int]]) -> None:
    for a, b in zip(points, points[1:]):
        for r, c in _line_cells(a, b):
            mask[r, c] = True


def _black_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] != BLACK:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and grid[nr, nc] == BLACK:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _ring_cells(shape: tuple[int, int], center: tuple[int, int], radius: int) -> list[tuple[int, int]]:
    cr, cc = center
    cells: list[tuple[int, int]] = []
    seen = set()
    for c in range(cc - radius, cc + radius + 1):
        for r in (cr - radius, cr + radius):
            if _in_bounds(shape, r, c) and (r, c) not in seen:
                seen.add((r, c))
                cells.append((r, c))
    for r in range(cr - radius + 1, cr + radius):
        for c in (cc - radius, cc + radius):
            if _in_bounds(shape, r, c) and (r, c) not in seen:
                seen.add((r, c))
                cells.append((r, c))
    return cells


def _find_seed_info(grid: np.ndarray) -> tuple[tuple[int, int], int, int, list[tuple[int, int]]]:
    seeds = [tuple(map(int, rc)) for rc in np.argwhere(grid == GREEN)]
    assert len(seeds) == 3, "expected exactly three green seeds"
    same_diff = len({c - r for r, c in seeds}) == 1
    same_sum = len({c + r for r, c in seeds}) == 1
    assert same_diff ^ same_sum, "seeds must lie on one 45-degree diagonal"

    ordered = sorted(seeds)
    (r0, c0), (r1, c1), (r2, c2) = ordered
    d1 = r1 - r0
    d2 = r2 - r1
    assert d1 == d2 and d1 >= 2, "seed spacing must be equal and at least 2"

    if same_diff:
        sign = 1
        assert c1 - c0 == d1 and c2 - c1 == d1
    else:
        sign = -1
        assert c1 - c0 == -d1 and c2 - c1 == -d1

    assert r0 + r2 == 2 * r1 and c0 + c2 == 2 * c1
    return (r1, c1), d1, sign, ordered


def _reachable_black(grid: np.ndarray, center: tuple[int, int]) -> np.ndarray:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    reach = np.zeros((h, w), dtype=bool)
    q = deque([center])
    seen[center] = True
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if not _in_bounds(grid.shape, nr, nc) or seen[nr, nc]:
                continue
            seen[nr, nc] = True
            if grid[nr, nc] != BLACK:
                continue
            reach[nr, nc] = True
            q.append((nr, nc))
    return reach


def _oracle_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    center, d, _sign, _seeds = _find_seed_info(g)
    reach = _reachable_black(g, center)
    out = g.copy()
    radius = d
    while True:
        ring = _ring_cells(g.shape, center, radius)
        if not ring:
            break
        for r, c in ring:
            if reach[r, c]:
                out[r, c] = ORANGE
        radius += d
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Draw wall-clipped concentric square rings from three diagonal seeds.

    Input structure:
    - The grid is mostly black background with blue wall pixels that form 4-connected barriers touching the grid boundary.
    - Exactly three green seed pixels lie in one open room on a 45-degree diagonal with equal spacing; the middle seed is the center.

    Rule:
    - Identify the middle seed and its spacing d to the outer seeds.
    - Generate square rings centered on the middle seed at Chebyshev radii d, 2d, 3d, ... until a whole ring falls outside the grid.
    - Flood-fill from the center through 4-connected black cells only, treating blue walls as blockers, then paint orange only on ring cells that are reachable by that flood.
    - Keep the three green seeds, the blue walls, and every non-ring or unreachable cell unchanged.

    Key insight:
    - The seeds specify a perfect concentric-square geometry, but the visible parts are clipped cell-by-cell by the center room's flood-reachable mask.
    """
    return _oracle_output(grid)


def _wall_cross(h: int, w: int, rng: random.Random) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    r = rng.randint(3, h - 4)
    c = rng.randint(3, w - 4)
    mask[r, :] = True
    mask[:, c] = True
    return mask


def _wall_tee(h: int, w: int, rng: random.Random) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    if rng.random() < 0.5:
        c = rng.randint(3, w - 4)
        r = rng.randint(3, h - 4)
        mask[:, c] = True
        if rng.random() < 0.5:
            mask[r, : c + 1] = True
        else:
            mask[r, c:] = True
    else:
        r = rng.randint(3, h - 4)
        c = rng.randint(3, w - 4)
        mask[r, :] = True
        if rng.random() < 0.5:
            mask[: r + 1, c] = True
        else:
            mask[r:, c] = True
    return mask


def _wall_elbow(h: int, w: int, rng: random.Random) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    variant = rng.choice(("tr", "tl", "br", "bl"))
    if variant == "tr":
        c = rng.randint(3, w - 5)
        r = rng.randint(3, h - 5)
        _paint_polyline(mask, [(0, c), (r, c), (r, w - 1)])
    elif variant == "tl":
        c = rng.randint(4, w - 4)
        r = rng.randint(3, h - 5)
        _paint_polyline(mask, [(0, c), (r, c), (r, 0)])
    elif variant == "br":
        c = rng.randint(3, w - 5)
        r = rng.randint(4, h - 4)
        _paint_polyline(mask, [(h - 1, c), (r, c), (r, w - 1)])
    else:
        c = rng.randint(4, w - 4)
        r = rng.randint(4, h - 4)
        _paint_polyline(mask, [(h - 1, c), (r, c), (r, 0)])
    return mask


def _wall_snake(h: int, w: int, rng: random.Random) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    if rng.random() < 0.5:
        c0 = rng.randint(2, w - 3)
        c1 = rng.randint(2, w - 3)
        c2 = rng.randint(2, w - 3)
        r1 = rng.randint(3, h // 2)
        r2 = rng.randint(r1 + 2, h - 4)
        _paint_polyline(mask, [(0, c0), (r1, c0), (r1, c1), (r2, c1), (r2, c2), (h - 1, c2)])
    else:
        r0 = rng.randint(2, h - 3)
        r1 = rng.randint(2, h - 3)
        r2 = rng.randint(2, h - 3)
        c1 = rng.randint(3, w // 2)
        c2 = rng.randint(c1 + 2, w - 4)
        _paint_polyline(mask, [(r0, 0), (r0, c1), (r1, c1), (r1, c2), (r2, c2), (r2, w - 1)])
    return mask


def _make_wall_grid(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(13, 20)
    w = rng.randint(13, 20)
    template = rng.choices(
        [_wall_cross, _wall_tee, _wall_elbow, _wall_snake],
        weights=[3, 3, 2, 2],
        k=1,
    )[0]
    wall_mask = template(h, w, rng)

    if rng.random() < 0.35:
        extra = np.zeros_like(wall_mask)
        if rng.random() < 0.5:
            c = rng.choice([rng.randint(1, max(1, w // 3)), rng.randint(max(1, 2 * w // 3), w - 2)])
            r0 = 0 if rng.random() < 0.5 else h - rng.randint(3, max(3, h // 2))
            r1 = rng.randint(max(r0 + 2, 2), h - 1) if r0 == 0 else h - 1
            extra[r0 : r1 + 1, c] = True
        else:
            r = rng.choice([rng.randint(1, max(1, h // 3)), rng.randint(max(1, 2 * h // 3), h - 2)])
            c0 = 0 if rng.random() < 0.5 else w - rng.randint(3, max(3, w // 2))
            c1 = rng.randint(max(c0 + 2, 2), w - 1) if c0 == 0 else w - 1
            extra[r, c0 : c1 + 1] = True
        if not np.any(wall_mask & extra):
            wall_mask |= extra

    grid = np.zeros((h, w), dtype=int)
    grid[wall_mask] = BLUE
    if len(_black_components(grid)) < 2:
        return None
    return grid


def _diagonal_clear(base_grid: np.ndarray, center: tuple[int, int], d: int, sign: int) -> bool:
    cr, cc = center
    for step in range(-d, d + 1):
        rr = cr + step
        cc2 = cc + sign * step
        if not _in_bounds(base_grid.shape, rr, cc2) or base_grid[rr, cc2] != BLACK:
            return False
    return True


def _all_ring_cells(shape: tuple[int, int], center: tuple[int, int], d: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    radius = d
    while True:
        ring = _ring_cells(shape, center, radius)
        if not ring:
            break
        cells.update(ring)
        radius += d
    return cells


def _instance_metrics(grid: np.ndarray, center: tuple[int, int], d: int) -> dict[str, int | bool]:
    out = _oracle_output(grid)
    reach = _reachable_black(grid, center)
    ring1 = _ring_cells(grid.shape, center, d)
    drawn1 = sum(1 for r, c in ring1 if out[r, c] == ORANGE)
    blocked_black1 = sum(1 for r, c in ring1 if grid[r, c] == BLACK and out[r, c] != ORANGE)
    ring2 = _ring_cells(grid.shape, center, 2 * d)
    drawn2 = sum(1 for r, c in ring2 if out[r, c] == ORANGE)
    total_orange = int(np.sum(out == ORANGE))
    all_rings = _all_ring_cells(grid.shape, center, d)
    changed = {(int(r), int(c)) for r, c in np.argwhere(grid != out)}
    expected = {(r, c) for r, c in all_rings if reach[r, c]}
    semantic_match = changed == expected
    hidden_black_any = any(grid[r, c] == BLACK and not reach[r, c] for r, c in ring1)
    return {
        "drawn1": drawn1,
        "blocked_black1": blocked_black1,
        "drawn2": drawn2,
        "total_orange": total_orange,
        "semantic_match": semantic_match,
        "hidden_black_any": hidden_black_any,
    }


def _validate_instance(grid: np.ndarray, require_multi: bool) -> bool:
    try:
        g = np.asarray(grid, dtype=int)
        center, d, sign, seeds = _find_seed_info(g)
        assert np.any(g == BLUE)
        assert any(
            g[r, c] == BLUE and (r == 0 or c == 0 or r == g.shape[0] - 1 or c == g.shape[1] - 1)
            for r, c in np.argwhere(g == BLUE)
        )

        base = g.copy()
        for r, c in seeds:
            base[r, c] = BLACK

        comps = [set(comp) for comp in _black_components(base)]
        seed_comp = next((comp for comp in comps if center in comp), None)
        assert seed_comp is not None
        assert all(seed in seed_comp for seed in seeds)
        assert _diagonal_clear(base, center, d, sign)

        metrics = _instance_metrics(g, center, d)
        assert metrics["semantic_match"]
        assert metrics["drawn1"] >= 3
        assert metrics["blocked_black1"] >= 2
        assert metrics["hidden_black_any"]
        assert metrics["total_orange"] >= 5
        if require_multi:
            assert metrics["drawn2"] >= 1
        return True
    except AssertionError:
        return False


def _try_place_seeds(base_grid: np.ndarray, rng: random.Random, require_multi: bool) -> np.ndarray | None:
    comps = _black_components(base_grid)
    rng.shuffle(comps)
    for comp in comps:
        if len(comp) < 18:
            continue
        comp_set = set(comp)
        centers = comp[:]
        rng.shuffle(centers)
        max_d = min(5, base_grid.shape[0] // 3, base_grid.shape[1] // 3)
        ds = list(range(2, max_d + 1))
        rng.shuffle(ds)
        signs = [1, -1]
        rng.shuffle(signs)

        for center in centers[: min(90, len(centers))]:
            cr, cc = center
            for d in ds:
                for sign in signs:
                    seeds = [(cr - d, cc - sign * d), (cr, cc), (cr + d, cc + sign * d)]
                    if not all(_in_bounds(base_grid.shape, r, c) for r, c in seeds):
                        continue
                    if not all(seed in comp_set for seed in seeds):
                        continue
                    if not _diagonal_clear(base_grid, center, d, sign):
                        continue

                    candidate = base_grid.copy()
                    for r, c in seeds:
                        candidate[r, c] = GREEN
                    if _validate_instance(candidate, require_multi):
                        return candidate
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(500):
        base_grid = _make_wall_grid(rng)
        if base_grid is None:
            continue

        require_multi = rng.random() < 0.55
        candidate = _try_place_seeds(base_grid, rng, require_multi=require_multi)
        if candidate is None and require_multi:
            candidate = _try_place_seeds(base_grid, rng, require_multi=False)
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid wall-clipped concentric-ring input")
