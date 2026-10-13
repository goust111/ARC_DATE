from __future__ import annotations

import random

import numpy as np


DIRS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

CARDINAL_STEPS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS8:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            pts = np.array(coords, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": pts,
                    "area": int(len(coords)),
                    "bbox": (
                        int(pts[:, 0].min()),
                        int(pts[:, 1].min()),
                        int(pts[:, 0].max()),
                        int(pts[:, 1].max()),
                    ),
                }
            )
    return comps


def _central_gap(grid: np.ndarray, bg: int, objects: list[dict]) -> tuple[int, int]:
    h, w = grid.shape
    candidates: list[tuple[int, int, int, int, float]] = []
    mid_r = (h - 1) / 2.0
    mid_c = (w - 1) / 2.0
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != bg:
                continue
            dists = []
            for obj in objects:
                coords = obj["coords"]
                d = np.abs(coords[:, 0] - r) + np.abs(coords[:, 1] - c)
                dists.append(int(d.min()))
            total = int(sum(dists))
            worst = int(max(dists))
            geom = abs(r - mid_r) + abs(c - mid_c)
            candidates.append((total, worst, r, c, geom))

    assert candidates, "expected at least one background cell"
    candidates.sort(key=lambda x: (x[0], x[1], x[4], x[2], x[3]))
    best_total = candidates[0][0]
    tied = [cand for cand in candidates if cand[0] == best_total]
    assert len(tied) == 1 or tied[0][1:] < tied[1][1:], "expected a unique center"
    return int(candidates[0][2]), int(candidates[0][3])


def _inward_tip(obj: dict, center_r: int, center_c: int) -> tuple[int, int]:
    coords = obj["coords"]
    d = np.abs(coords[:, 0] - center_r) + np.abs(coords[:, 1] - center_c)
    mind = int(d.min())
    tips = coords[d == mind]
    assert len(tips) == 1, "expected a unique inward tip"
    return int(tips[0, 0]), int(tips[0, 1])


def _winner_cap(obj: dict, center_r: int, center_c: int) -> list[tuple[int, int]]:
    tip_r, tip_c = _inward_tip(obj, center_r, center_c)
    rows = obj["coords"][:, 0]
    cols = obj["coords"][:, 1]
    if tip_r < center_r:
        cap_r = int(rows.min()) - 1
        return [(cap_r, center_c - 1), (cap_r, center_c), (cap_r, center_c + 1)]
    if tip_r > center_r:
        cap_r = int(rows.max()) + 1
        return [(cap_r, center_c - 1), (cap_r, center_c), (cap_r, center_c + 1)]
    if tip_c < center_c:
        cap_c = int(cols.min()) - 1
        return [(center_r - 1, cap_c), (center_r, cap_c), (center_r + 1, cap_c)]
    assert tip_c > center_c, "winner must sit in a cardinal direction from the center"
    cap_c = int(cols.max()) + 1
    return [(center_r - 1, cap_c), (center_r, cap_c), (center_r + 1, cap_c)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Let the largest center-pointing object grow outward while every other object loses its inward tip.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    objects = _find_components(g, bg)
    assert len(objects) == 4, "expected four foreground objects"

    center_r, center_c = _central_gap(g, bg, objects)

    areas = [int(obj["area"]) for obj in objects]
    max_area = max(areas)
    assert areas.count(max_area) == 1, "expected a unique largest object"
    winner = objects[areas.index(max_area)]

    out = g.copy()
    for obj in objects:
        if obj is winner:
            continue
        tip_r, tip_c = _inward_tip(obj, center_r, center_c)
        out[tip_r, tip_c] = bg

    cap = _winner_cap(winner, center_r, center_c)
    for r, c in cap:
        assert 0 <= r < g.shape[0] and 0 <= c < g.shape[1], "cap must stay in bounds"
        assert int(g[r, c]) == bg, "cap must be painted on background"
        out[r, c] = int(winner["color"])

    return out


GEN_RETRIES = 400
SIZE_MIN = 13
SIZE_MAX = 19
GAP_CHOICES = (1, 2)
AREA_MIN = 4
AREA_MAX = 18


def _perp_point(direction: str, base_r: int, base_c: int, offset: int) -> tuple[int, int]:
    if direction in ("up", "down"):
        return base_r, base_c + offset
    return base_r + offset, base_c


def _tip_coord(center_r: int, center_c: int, direction: str, gap: int) -> tuple[int, int]:
    step_r, step_c = CARDINAL_STEPS[direction]
    return center_r + step_r * gap, center_c + step_c * gap


def _build_cardinal_object(
    center_r: int,
    center_c: int,
    direction: str,
    gap: int,
    target_area: int,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    tip_r, tip_c = _tip_coord(center_r, center_c, direction, gap)
    step_r, step_c = CARDINAL_STEPS[direction]
    coords = [(tip_r, tip_c)]
    remaining = target_area - 1
    prev_hw = 0
    layer = 0

    while remaining > 0:
        layer += 1
        br = tip_r + step_r * layer
        bc = tip_c + step_c * layer

        min_hw = max(prev_hw, 1) if layer == 1 else max(prev_hw - 1, 0)
        max_hw = prev_hw + 2
        feasible = [hw for hw in range(min_hw, max_hw + 1) if 2 * hw + 1 <= remaining]
        if not feasible:
            needed = (remaining - 1) // 2
            if remaining == 2 * needed + 1 and needed >= min_hw:
                feasible = [needed]
            else:
                break

        hw = rng.choice(feasible)
        for off in range(-hw, hw + 1):
            coords.append(_perp_point(direction, br, bc, off))
        remaining -= 2 * hw + 1
        prev_hw = hw

        if layer > 8:
            break

    if remaining > 0 and layer >= 1:
        br = tip_r + step_r * layer
        bc = tip_c + step_c * layer
        for side in (1, -1):
            if remaining <= 0:
                break
            coords.append(_perp_point(direction, br, bc, side * (prev_hw + 1)))
            remaining -= 1

    if remaining != 0:
        return None
    return coords


def _chebyshev_separation(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    return min(max(abs(ra - rb), abs(ca - cb)) for ra, ca in a for rb, cb in b)


def _is_single_component(points: list[tuple[int, int]]) -> bool:
    pts = set(points)
    if len(pts) != len(points):
        return False
    stack = [next(iter(pts))]
    seen = set()
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        r, c = cur
        for dr, dc in DIRS8:
            nb = (r + dr, c + dc)
            if nb in pts and nb not in seen:
                stack.append(nb)
    return len(seen) == len(pts)


def _render_scene(
    h: int,
    w: int,
    bg: int,
    groups: list[list[tuple[int, int]]],
    colors: list[int],
) -> np.ndarray:
    grid = np.full((h, w), bg, dtype=int)
    for pts, color in zip(groups, colors):
        rr, cc = zip(*pts)
        grid[rr, cc] = int(color)
    return grid


def _sample_scene(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    h = rng.randint(SIZE_MIN, SIZE_MAX)
    w = rng.randint(SIZE_MIN, SIZE_MAX)
    center_r = rng.randint(4, h - 5)
    center_c = rng.randint(4, w - 5)
    bg = rng.randint(0, 9)
    colors = rng.sample([c for c in range(10) if c != bg], 4)

    dirs = ["up", "down", "left", "right"]
    rng.shuffle(dirs)
    winner_dir = dirs[0]
    dir_order = ["up", "down", "left", "right"]
    groups_by_dir: dict[str, list[tuple[int, int]]] = {}
    gaps = {d: rng.choice(GAP_CHOICES) for d in dir_order}

    winner_area = rng.randint(8, AREA_MAX)
    loser_areas = [rng.randint(AREA_MIN, winner_area - 1) for _ in range(3)]
    area_map = {}
    loser_i = 0
    for d in dir_order:
        if d == winner_dir:
            area_map[d] = winner_area
        else:
            area_map[d] = loser_areas[loser_i]
            loser_i += 1

    for d in dir_order:
        groups_by_dir[d] = _build_cardinal_object(
            center_r, center_c, d, gaps[d], area_map[d], rng
        )
        if groups_by_dir[d] is None:
            return None

    winner_cap = _winner_cap(
        {
            "coords": np.array(groups_by_dir[winner_dir], dtype=int),
            "bbox": (
                min(r for r, _ in groups_by_dir[winner_dir]),
                min(c for _, c in groups_by_dir[winner_dir]),
                max(r for r, _ in groups_by_dir[winner_dir]),
                max(c for _, c in groups_by_dir[winner_dir]),
            ),
        },
        center_r,
        center_c,
    )

    all_input_groups = [groups_by_dir[d] for d in dir_order]
    for pts in all_input_groups:
        if not _is_single_component(pts):
            return None
        if not all(0 <= r < h and 0 <= c < w for r, c in pts):
            return None
    if not all(0 <= r < h and 0 <= c < w for r, c in winner_cap):
        return None

    expanded_groups = []
    for d in dir_order:
        pts = list(groups_by_dir[d])
        if d == winner_dir:
            pts = pts + winner_cap
        expanded_groups.append(pts)
    for i in range(4):
        for j in range(i + 1, 4):
            if _chebyshev_separation(expanded_groups[i], expanded_groups[j]) <= 1:
                return None

    inp = _render_scene(h, w, bg, [groups_by_dir[d] for d in dir_order], colors)
    try:
        comps = _find_components(inp, bg)
        if len(comps) != 4:
            return None
        center2 = _central_gap(inp, bg, comps)
        if center2 != (center_r, center_c):
            return None
    except Exception:
        return None

    out = main(inp)
    if np.array_equal(inp, out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = random.Random()
        result = _sample_scene(rng)
        if result is None:
            continue
        inp, _out = result
        return inp
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
