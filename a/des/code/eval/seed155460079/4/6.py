from __future__ import annotations

from itertools import permutations

import numpy as np


ORTHO4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _cc4(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        pts: list[tuple[int, int]] = []
        while stack:
            rr, cc = stack.pop()
            pts.append((rr, cc))
            for dr, dc in ORTHO4:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _coord_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, pt)) for pt in coords.tolist()}


def _render(shape: tuple[int, int], bg: int, light: int, shadow_color: int, objects: list[dict]) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), int(bg), dtype=int)
    shadow_union: set[tuple[int, int]] = set()
    for obj in objects:
        shadow_union.update(_coord_set(obj["shadow"]))
    for r, c in shadow_union:
        out[r, c] = int(shadow_color)
    for obj in objects:
        coords = obj["coords"]
        out[coords[:, 0], coords[:, 1]] = int(obj["color"])
    out[0, :] = int(light)
    return out


def _infer_scene(grid: np.ndarray) -> tuple[int, int, int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    light = int(g[0, 0])
    assert np.all(g[0, :] == light), "top row must be the light source row"
    colors = sorted(set(map(int, g.flatten())) - {light})
    assert len(colors) >= 4, "expected bg, shadow, and at least two object colors"

    solutions: list[tuple[int, int, list[dict]]] = []
    for bg, shadow_color in permutations(colors, 2):
        object_colors = [color for color in colors if color not in {bg, shadow_color}]
        if len(object_colors) < 2:
            continue

        objects: list[dict] = []
        object_union: set[tuple[int, int]] = set()
        ok = True
        for color in object_colors:
            comps = _cc4(g == color)
            if len(comps) != 1:
                ok = False
                break
            coords = comps[0]
            r0, c0, r1, c1 = _bbox(coords)
            if not (2 <= r1 - r0 + 1 <= 4 and 2 <= c1 - c0 + 1 <= 4):
                ok = False
                break
            if coords.shape[0] < 3:
                ok = False
                break
            shift = r0 - 1
            if shift < 1:
                ok = False
                break
            shadow = coords + np.array([shift, 0], dtype=int)
            if shadow[:, 0].max() >= h or shadow[:, 1].max() >= w or shadow[:, 1].min() < 0:
                ok = False
                break
            objects.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "shadow": shadow,
                }
            )
            object_union.update(_coord_set(coords))
        if not ok:
            continue

        shadow_union: set[tuple[int, int]] = set()
        for obj in objects:
            shadow_union.update(_coord_set(obj["shadow"]))
        expected_visible_shadow = shadow_union - object_union
        actual_visible_shadow = _coord_set(np.argwhere(g == shadow_color))
        if actual_visible_shadow != expected_visible_shadow:
            continue

        mismatch = False
        for r in range(1, h):
            for c in range(w):
                if (r, c) in object_union or (r, c) in expected_visible_shadow:
                    continue
                if int(g[r, c]) != bg:
                    mismatch = True
                    break
            if mismatch:
                break
        if mismatch:
            continue

        solutions.append((int(bg), int(shadow_color), objects))

    assert solutions, "failed to infer a valid scene"
    assert len(solutions) == 1, "scene inference should be unambiguous"
    bg, shadow_color, objects = solutions[0]
    return light, bg, shadow_color, objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove every object whose body lies entirely inside the shadows cast by other objects.
    """
    g = np.asarray(grid, dtype=int)
    light, bg, shadow_color, objects = _infer_scene(g)

    shadow_sets = [_coord_set(obj["shadow"]) for obj in objects]
    survivors: list[dict] = []
    for i, obj in enumerate(objects):
        other_shadow_union: set[tuple[int, int]] = set()
        for j, shadow_set in enumerate(shadow_sets):
            if i != j:
                other_shadow_union.update(shadow_set)
        if not all(tuple(map(int, pt)) in other_shadow_union for pt in obj["coords"].tolist()):
            survivors.append(obj)

    return _render(g.shape, bg, light, shadow_color, survivors)


GEN_BG = 0
GEN_LIGHT = 4
GEN_SHADOW = 5
GEN_OBJECT_COLORS = [1, 2, 3, 6, 7, 8, 9]
GRID_H_RANGE = (15, 21)
GRID_W_RANGE = (12, 20)
CASTER_COUNT_RANGE = (2, 3)
TARGET_COUNT_RANGE = (1, 2)
GEN_ATTEMPTS = 400
SHAPE_ATTEMPTS = 60
PLACEMENT_ATTEMPTS = 120
BODY_GAP = 1
VISIBLE_SHADOW_MIN = 6


def _sample_shape(rng: np.random.Generator) -> np.ndarray | None:
    for _ in range(SHAPE_ATTEMPTS):
        h = int(rng.integers(2, 5))
        w = int(rng.integers(2, 5))
        area_min = max(3, h + w - 1)
        target_area = int(rng.integers(area_min, h * w + 1))
        start = (int(rng.integers(0, h)), int(rng.integers(0, w)))
        cells = {start}
        while len(cells) < target_area:
            frontier: list[tuple[int, int]] = []
            seen_frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for dr, dc in ORTHO4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells and (nr, nc) not in seen_frontier:
                        frontier.append((nr, nc))
                        seen_frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(frontier[int(rng.integers(0, len(frontier)))])
        coords = np.array(sorted(cells), dtype=int)
        r0, c0, r1, c1 = _bbox(coords)
        if r1 - r0 + 1 != h or c1 - c0 + 1 != w:
            continue
        return coords - np.array([r0, c0], dtype=int)
    return None


def _place_abs(shape: np.ndarray, top: int, left: int) -> np.ndarray:
    return shape + np.array([top, left], dtype=int)


def _body_clear(coords: np.ndarray, occupied: set[tuple[int, int]], gap: int = BODY_GAP) -> bool:
    for r, c in coords.tolist():
        r = int(r)
        c = int(c)
        for rr in range(r - gap, r + gap + 1):
            for cc in range(c - gap, c + gap + 1):
                if (rr, cc) in occupied:
                    return False
    return True


def _make_object(color: int, coords: np.ndarray) -> dict:
    top = int(coords[:, 0].min())
    shift = top - 1
    shadow = coords + np.array([shift, 0], dtype=int)
    return {
        "color": int(color),
        "coords": coords,
        "shadow": shadow,
    }


def _covered_flags(objects: list[dict]) -> list[bool]:
    shadow_sets = [_coord_set(obj["shadow"]) for obj in objects]
    flags: list[bool] = []
    for i, obj in enumerate(objects):
        other_shadow_union: set[tuple[int, int]] = set()
        for j, shadow_set in enumerate(shadow_sets):
            if i != j:
                other_shadow_union.update(shadow_set)
        flags.append(all(tuple(map(int, pt)) in other_shadow_union for pt in obj["coords"].tolist()))
    return flags


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    w = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))

    caster_count = int(rng.integers(CASTER_COUNT_RANGE[0], CASTER_COUNT_RANGE[1] + 1))
    target_count = int(rng.integers(TARGET_COUNT_RANGE[0], TARGET_COUNT_RANGE[1] + 1))
    total_count = caster_count + target_count
    if total_count > len(GEN_OBJECT_COLORS):
        return None
    colors = list(rng.choice(GEN_OBJECT_COLORS, size=total_count, replace=False))

    objects: list[dict] = []
    body_occupied: set[tuple[int, int]] = set()
    caster_shadow_union: set[tuple[int, int]] = set()

    for idx in range(caster_count):
        shape = _sample_shape(rng)
        if shape is None:
            return None
        sh_h = int(shape[:, 0].max()) + 1
        sh_w = int(shape[:, 1].max()) + 1
        top_min = sh_h + 1
        top_max = min((h - sh_h + 1) // 2, max(top_min, h // 2 - 1))
        if top_min > top_max:
            return None

        placed = None
        for _ in range(PLACEMENT_ATTEMPTS):
            top = int(rng.integers(top_min, top_max + 1))
            left = int(rng.integers(0, w - sh_w + 1))
            coords = _place_abs(shape, top, left)
            if not _body_clear(coords, body_occupied):
                continue
            if all(tuple(map(int, pt)) in caster_shadow_union for pt in coords.tolist()):
                continue
            obj = _make_object(colors[idx], coords)
            placed = obj
            break
        if placed is None:
            return None
        objects.append(placed)
        body_occupied.update(_coord_set(placed["coords"]))
        caster_shadow_union.update(_coord_set(placed["shadow"]))

    if not caster_shadow_union:
        return None

    for idx in range(caster_count, total_count):
        shape = _sample_shape(rng)
        if shape is None:
            return None
        sh_h = int(shape[:, 0].max()) + 1
        sh_w = int(shape[:, 1].max()) + 1
        top_min = sh_h + 1
        top_max = (h - sh_h + 1) // 2
        if top_min > top_max:
            return None

        candidates: list[dict] = []
        for top in range(top_min, top_max + 1):
            for left in range(0, w - sh_w + 1):
                coords = _place_abs(shape, top, left)
                if not _body_clear(coords, body_occupied):
                    continue
                if not all(tuple(map(int, pt)) in caster_shadow_union for pt in coords.tolist()):
                    continue
                candidates.append(_make_object(colors[idx], coords))
        if not candidates:
            return None
        choice = candidates[int(rng.integers(0, len(candidates)))]
        objects.append(choice)
        body_occupied.update(_coord_set(choice["coords"]))

    flags = _covered_flags(objects)
    if flags.count(True) < 1 or flags.count(False) < 2:
        return None
    if any(flags[:caster_count]):
        return None
    if not all(flags[caster_count:]):
        return None

    grid = _render((h, w), GEN_BG, GEN_LIGHT, GEN_SHADOW, objects)
    if int(np.count_nonzero(grid == GEN_SHADOW)) < VISIBLE_SHADOW_MIN:
        return None

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if np.array_equal(grid, solved):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
