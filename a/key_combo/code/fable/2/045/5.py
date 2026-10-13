from __future__ import annotations

from collections import deque
from functools import lru_cache
import random

import numpy as np


GREY = 5
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
BACKGROUND_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _dilate(mask: np.ndarray, connectivity: int = 4) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    dirs = DIRS4 if connectivity == 4 else DIRS8
    for r, c in np.argwhere(mask):
        for dr, dc in dirs:
            nr, nc = int(r + dr), int(c + dc)
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = True
    return out


def _mask_from_coords(shape: tuple[int, int], coords: list[tuple[int, int]] | np.ndarray) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    if len(coords) == 0:
        return out
    arr = np.asarray(coords, dtype=int)
    out[arr[:, 0], arr[:, 1]] = True
    return out


def _find_object_components(grid: np.ndarray) -> list[dict]:
    objects = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != GREY):
        for coords in _components_mask(grid == color):
            if int(coords.shape[0]) > 1:
                objects.append({"color": int(color), "coords": coords, "area": int(coords.shape[0])})
    return objects


def _trail_endpoints(coords: np.ndarray) -> list[tuple[int, int]]:
    cells = {tuple(map(int, rc)) for rc in coords}
    endpoints: list[tuple[int, int]] = []
    for r, c in cells:
        degree = sum((r + dr, c + dc) in cells for dr, dc in DIRS4)
        assert degree in (1, 2), "grey components must be simple paths"
        if degree == 1:
            endpoints.append((r, c))
    assert len(endpoints) == 2, "grey trail must have two endpoints"
    return endpoints


def _find_axis_period(grid: np.ndarray, hidden: np.ndarray, axis: int) -> int:
    h, w = grid.shape
    limit = h if axis == 0 else w
    for step in range(1, limit):
        if axis == 0:
            visible = (~hidden[:-step, :]) & (~hidden[step:, :])
            support = int(visible.sum())
            if support == 0:
                continue
            if np.all(grid[:-step, :][visible] == grid[step:, :][visible]):
                return step
        else:
            visible = (~hidden[:, :-step]) & (~hidden[:, step:])
            support = int(visible.sum())
            if support == 0:
                continue
            if np.all(grid[:, :-step][visible] == grid[:, step:][visible]):
                return step
    raise AssertionError("failed to detect translational period")


def _restore_cell_color(grid: np.ndarray, hidden: np.ndarray, pr: int, pc: int, r: int, c: int) -> int:
    colors = set()
    for rr in range(r % pr, grid.shape[0], pr):
        for cc in range(c % pc, grid.shape[1], pc):
            if not hidden[rr, cc]:
                colors.add(int(grid[rr, cc]))
    assert len(colors) == 1, "periodic orbit must reveal a unique background color"
    return next(iter(colors))


def main(grid: np.ndarray) -> np.ndarray:
    """Send each object back to the far end of its grey trail and restore the periodic background underneath.

    Input structure:
    - The grid is an axis-aligned translational tiling made from several non-grey background colors.
    - Grey 4-connected one-cell-wide trail components lie on top of that tiling.
    - Each movable object is a monochrome 4-connected component with area greater than one, and exactly one grey trail endpoint is 4-neighbor-adjacent to exactly one cell of that object.

    Rule:
    - For every grey trail, identify the endpoint touching an object and use the unique touching object cell as that object's anchor.
    - Follow the trail to its opposite endpoint and translate the whole object so that the anchor lands on the far endpoint.
    - Remove all grey trail cells and all vacated object cells, refill those cells from the visible translational tiling, then draw the translated objects at their new locations.

    Key insight:
    - The grey paths are motion traces rather than marks to keep; the hidden pattern is recovered by grouping cells into translational orbits of the visible tiling.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    objects = _find_object_components(g)
    trails = _components_mask(g == GREY)
    assert objects and trails

    object_index = -np.ones(g.shape, dtype=int)
    for idx, obj in enumerate(objects):
        coords = obj["coords"]
        object_index[coords[:, 0], coords[:, 1]] = idx

    matched = {}
    translations: dict[int, tuple[int, int]] = {}
    hidden = np.zeros(g.shape, dtype=bool)
    for obj in objects:
        coords = obj["coords"]
        hidden[coords[:, 0], coords[:, 1]] = True
    hidden[g == GREY] = True

    for trail_coords in trails:
        endpoints = _trail_endpoints(trail_coords)
        endpoint_info = []
        for ep in endpoints:
            neighbors = []
            for dr, dc in DIRS4:
                nr, nc = ep[0] + dr, ep[1] + dc
                if _in_bounds(g.shape, nr, nc) and object_index[nr, nc] >= 0:
                    neighbors.append((int(object_index[nr, nc]), (nr, nc)))
            endpoint_info.append(neighbors)

        touching = [i for i, info in enumerate(endpoint_info) if len(info) == 1]
        assert len(touching) == 1, "each trail must touch exactly one object at exactly one endpoint"
        near_idx = touching[0]
        far_idx = 1 - near_idx
        obj_idx, anchor = endpoint_info[near_idx][0]
        assert obj_idx not in matched, "an object cannot own multiple trails"
        matched[obj_idx] = True
        far = endpoints[far_idx]
        translations[obj_idx] = (far[0] - anchor[0], far[1] - anchor[1])

    assert len(translations) == len(objects), "every object must match one grey trail"

    pr = _find_axis_period(g, hidden, axis=0)
    pc = _find_axis_period(g, hidden, axis=1)

    out = g.copy()
    restore_coords = np.argwhere(hidden)
    for r, c in restore_coords:
        out[r, c] = _restore_cell_color(g, hidden, pr, pc, int(r), int(c))

    for idx, obj in enumerate(objects):
        dr, dc = translations[idx]
        rr = obj["coords"][:, 0] + int(dr)
        cc = obj["coords"][:, 1] + int(dc)
        assert rr.min() >= 0 and cc.min() >= 0
        assert rr.max() < g.shape[0] and cc.max() < g.shape[1]
        out[rr, cc] = int(obj["color"])

    return out


def _is_connected(mask: np.ndarray) -> bool:
    comps = _components_mask(mask)
    return len(comps) == 1


def _has_smaller_period(tile: np.ndarray, axis: int) -> bool:
    limit = tile.shape[axis]
    for step in range(1, limit):
        if axis == 0:
            if np.all(tile == np.roll(tile, -step, axis=0)):
                return True
        else:
            if np.all(tile == np.roll(tile, -step, axis=1)):
                return True
    return False


def _valid_tile(tile: np.ndarray) -> bool:
    h, w = tile.shape
    if len(np.unique(tile)) < 3:
        return False
    for r in range(h):
        for c in range(w):
            if tile[r, c] == tile[(r + 1) % h, c]:
                return False
            if tile[r, c] == tile[r, (c + 1) % w]:
                return False
    if _has_smaller_period(tile, axis=0) or _has_smaller_period(tile, axis=1):
        return False
    return True


def _sample_tile(rng: random.Random) -> np.ndarray:
    for _ in range(300):
        h = rng.randint(2, 4)
        w = rng.randint(2, 4)
        palette = rng.sample(BACKGROUND_POOL, rng.randint(3, min(4, len(BACKGROUND_POOL))))
        tile = np.array([[rng.choice(palette) for _ in range(w)] for _ in range(h)], dtype=int)
        if _valid_tile(tile):
            return tile
    raise RuntimeError("failed to sample primitive tile")


def _tile_fill(shape: tuple[int, int], tile: np.ndarray, row_phase: int, col_phase: int) -> np.ndarray:
    h, w = shape
    th, tw = tile.shape
    out = np.empty((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            out[r, c] = int(tile[(r + row_phase) % th, (c + col_phase) % tw])
    return out


@lru_cache(None)
def _shape_catalog() -> tuple[np.ndarray, ...]:
    masks = []
    for h in (2, 3):
        for w in (2, 3):
            n = h * w
            for bits in range(1, 1 << n):
                area = bits.bit_count()
                if area < 3 or area > 5:
                    continue
                mask = np.array([(bits >> i) & 1 for i in range(n)], dtype=bool).reshape(h, w)
                if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
                    continue
                if not _is_connected(mask):
                    continue
                masks.append(mask.copy())
    return tuple(masks)


def _anchor_options(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = mask.shape
    options = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for dr, dc in DIRS4:
            er, ec = rr + dr, cc + dc
            if 0 <= er < h and 0 <= ec < w and mask[er, ec]:
                continue
            neighbors = []
            for sdr, sdc in DIRS4:
                nr, nc = er + sdr, ec + sdc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                    neighbors.append((nr, nc))
            if neighbors == [(rr, cc)]:
                options.append((rr, cc, dr, dc))
    return options


def _coords_from_mask(mask: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    return [(top + int(r), left + int(c)) for r, c in np.argwhere(mask)]


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _bboxes_far_apart(a: list[tuple[int, int]], b: list[tuple[int, int]], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = _bbox(a)
    br0, bc0, br1, bc1 = _bbox(b)
    return ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    if a[0] == b[0]:
        step = 1 if b[1] >= a[1] else -1
        return [(a[0], c) for c in range(a[1], b[1] + step, step)]
    if a[1] == b[1]:
        step = 1 if b[0] >= a[0] else -1
        return [(r, a[1]) for r in range(a[0], b[0] + step, step)]
    raise ValueError("segment endpoints must share row or column")


def _segment_clear(a: tuple[int, int], b: tuple[int, int], blocked: np.ndarray) -> bool:
    for cell in _segment_cells(a, b)[1:-1]:
        if blocked[cell]:
            return False
    return True


def _polyline_path(points: list[tuple[int, int]], blocked: np.ndarray) -> list[tuple[int, int]] | None:
    path: list[tuple[int, int]] = []
    seen = set()
    for i in range(len(points) - 1):
        a, b = points[i], points[i + 1]
        if a == b:
            continue
        if not _segment_clear(a, b, blocked):
            return None
        seg = _segment_cells(a, b)
        if path:
            seg = seg[1:]
        for cell in seg:
            if blocked[cell] and cell not in (points[0], points[-1]):
                return None
            if cell in seen:
                return None
            seen.add(cell)
            path.append(cell)
    return path if path else None


def _random_path(
    start: tuple[int, int],
    goal: tuple[int, int],
    blocked: np.ndarray,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    h, w = blocked.shape
    candidates: list[list[tuple[int, int]]] = []
    if start[0] == goal[0] or start[1] == goal[1]:
        candidates.append([start, goal])
    candidates.append([start, (start[0], goal[1]), goal])
    candidates.append([start, (goal[0], start[1]), goal])

    rows = list(range(h))
    cols = list(range(w))
    rng.shuffle(rows)
    rng.shuffle(cols)
    for r_mid in rows[: min(12, h)]:
        candidates.append([start, (r_mid, start[1]), (r_mid, goal[1]), goal])
    for c_mid in cols[: min(12, w)]:
        candidates.append([start, (start[0], c_mid), (goal[0], c_mid), goal])

    bend_pref = []
    straight_fallback = []
    for points in candidates:
        cleaned = [points[0]]
        for point in points[1:]:
            if point != cleaned[-1]:
                cleaned.append(point)
        if any(not _in_bounds(blocked.shape, r, c) for r, c in cleaned):
            continue
        if blocked[cleaned[0]] or blocked[cleaned[-1]]:
            continue
        turns = len(cleaned) - 2
        if turns >= 1:
            bend_pref.append(cleaned)
        else:
            straight_fallback.append(cleaned)

    rng.shuffle(bend_pref)
    rng.shuffle(straight_fallback)
    for points in bend_pref + straight_fallback:
        path = _polyline_path(points, blocked)
        if path is not None:
            return path
    return None


def _partition_lengths(total: int, count: int, min_len: int, rng: random.Random) -> list[int]:
    extra = total - count * min_len
    parts = [min_len] * count
    for _ in range(extra):
        parts[rng.randrange(count)] += 1
    rng.shuffle(parts)
    return parts


def _sample_zones(n_objects: int, rng: random.Random) -> tuple[tuple[int, int], list[tuple[int, int, int, int]]]:
    if n_objects == 1:
        h = rng.randint(12, 24)
        w = rng.randint(12, 24)
        return (h, w), [(0, 0, h - 1, w - 1)]

    gutter = 1
    min_len = 8
    if rng.random() < 0.5:
        w = rng.randint(n_objects * min_len + (n_objects - 1) * gutter, 30)
        h = rng.randint(14, 24)
        widths = _partition_lengths(w - (n_objects - 1) * gutter, n_objects, min_len, rng)
        zones = []
        c0 = 0
        for width in widths:
            zones.append((0, c0, h - 1, c0 + width - 1))
            c0 += width + gutter
        return (h, w), zones

    h = rng.randint(n_objects * min_len + (n_objects - 1) * gutter, 30)
    w = rng.randint(14, 24)
    heights = _partition_lengths(h - (n_objects - 1) * gutter, n_objects, min_len, rng)
    zones = []
    r0 = 0
    for height in heights:
        zones.append((r0, 0, r0 + height - 1, w - 1))
        r0 += height + gutter
    return (h, w), zones


def _validate_local_trail(
    current_coords: list[tuple[int, int]],
    path: list[tuple[int, int]],
    near: tuple[int, int],
    far: tuple[int, int],
) -> bool:
    if len(path) < 2 or path[0] != near or path[-1] != far:
        return False
    if len(set(path)) != len(path):
        return False
    path_set = set(path)
    for i, cell in enumerate(path):
        degree = sum((cell[0] + dr, cell[1] + dc) in path_set for dr, dc in DIRS4)
        if i in (0, len(path) - 1):
            if degree != 1:
                return False
        elif degree != 2:
            return False
    current_set = set(current_coords)
    touching = [cell for cell in current_coords if abs(cell[0] - near[0]) + abs(cell[1] - near[1]) == 1]
    if len(touching) != 1:
        return False
    for cell in path[1:]:
        for dr, dc in DIRS4:
            if (cell[0] + dr, cell[1] + dc) in current_set:
                return False
    return True


def _place_object_in_zone(
    zone: tuple[int, int, int, int],
    color: int,
    rng: random.Random,
) -> dict | None:
    zr0, zc0, zr1, zc1 = zone
    zh = zr1 - zr0 + 1
    zw = zc1 - zc0 + 1
    masks = list(_shape_catalog())
    rng.shuffle(masks)

    for mask in masks:
        h, w = mask.shape
        if h + 3 > zh or w + 3 > zw:
            continue
        options = _anchor_options(mask)
        rng.shuffle(options)
        for ar, ac, dr, dc in options:
            er, ec = ar + dr, ac + dc
            r_min = max(0, -er)
            r_max = min(zh - h, zh - 1 - er)
            c_min = max(0, -ec)
            c_max = min(zw - w, zw - 1 - ec)
            if r_min > r_max or c_min > c_max:
                continue

            for _ in range(80):
                cur_top = rng.randint(r_min, r_max)
                cur_left = rng.randint(c_min, c_max)
                current_local = _coords_from_mask(mask, cur_top, cur_left)
                near_local = (cur_top + er, cur_left + ec)

                for _ in range(120):
                    out_top = rng.randint(0, zh - h)
                    out_left = rng.randint(0, zw - w)
                    output_local = _coords_from_mask(mask, out_top, out_left)
                    if not _bboxes_far_apart(current_local, output_local, pad=1):
                        continue
                    far_local = (out_top + ar, out_left + ac)
                    if abs(far_local[0] - near_local[0]) + abs(far_local[1] - near_local[1]) < 4:
                        continue

                    current_mask = _mask_from_coords((zh, zw), current_local)
                    output_mask = _mask_from_coords((zh, zw), output_local)
                    blocked = _dilate(current_mask, connectivity=4) | output_mask
                    blocked[current_mask] = True
                    blocked[output_mask] = True
                    blocked[near_local] = False
                    blocked[far_local] = False
                    path = _random_path(near_local, far_local, blocked, rng)
                    if path is None:
                        continue
                    if not _validate_local_trail(current_local, path, near_local, far_local):
                        continue

                    current_abs = [(zr0 + r, zc0 + c) for r, c in current_local]
                    output_abs = [(zr0 + r, zc0 + c) for r, c in output_local]
                    path_abs = [(zr0 + r, zc0 + c) for r, c in path]
                    return {
                        "color": int(color),
                        "current": current_abs,
                        "output": output_abs,
                        "trail": path_abs,
                        "near": (zr0 + near_local[0], zc0 + near_local[1]),
                        "far": (zr0 + far_local[0], zc0 + far_local[1]),
                    }
    return None


def _hidden_mask_from_specs(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    hidden = np.zeros(shape, dtype=bool)
    for spec in specs:
        for r, c in spec["current"]:
            hidden[r, c] = True
        for r, c in spec["trail"]:
            hidden[r, c] = True
    return hidden


def _expected_output(base: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = base.copy()
    for spec in specs:
        for r, c in spec["output"]:
            out[r, c] = int(spec["color"])
    return out


def _render_input(base: np.ndarray, specs: list[dict]) -> np.ndarray:
    inp = base.copy()
    for spec in specs:
        for r, c in spec["current"]:
            inp[r, c] = int(spec["color"])
        for r, c in spec["trail"]:
            inp[r, c] = GREY
    return inp


def _period_candidates(grid: np.ndarray, hidden: np.ndarray, axis: int) -> list[int]:
    h, w = grid.shape
    limit = h if axis == 0 else w
    candidates = []
    for step in range(1, limit):
        if axis == 0:
            visible = (~hidden[:-step, :]) & (~hidden[step:, :])
            if int(visible.sum()) == 0:
                continue
            if np.all(grid[:-step, :][visible] == grid[step:, :][visible]):
                candidates.append(step)
        else:
            visible = (~hidden[:, :-step]) & (~hidden[:, step:])
            if int(visible.sum()) == 0:
                continue
            if np.all(grid[:, :-step][visible] == grid[:, step:][visible]):
                candidates.append(step)
    return candidates


def _validate_instance(base: np.ndarray, inp: np.ndarray, expected: np.ndarray, specs: list[dict], tile: np.ndarray) -> bool:
    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    hidden = _hidden_mask_from_specs(base.shape, specs)
    row_cands = _period_candidates(inp, hidden, axis=0)
    col_cands = _period_candidates(inp, hidden, axis=1)
    if not row_cands or not col_cands:
        return False
    if row_cands[0] != tile.shape[0] or col_cands[0] != tile.shape[1]:
        return False

    restored_colors = {int(base[r, c]) for r, c in np.argwhere(hidden)}
    if len(restored_colors) < 2:
        return False

    for spec in specs:
        trail = spec["trail"]
        if trail[0] == trail[-1]:
            return False
        path_set = set(trail)
        endpoints = [cell for cell in trail if sum((cell[0] + dr, cell[1] + dc) in path_set for dr, dc in DIRS4) == 1]
        if len(endpoints) != 2:
            return False
        touching = [cell for cell in spec["current"] if abs(cell[0] - spec["near"][0]) + abs(cell[1] - spec["near"][1]) == 1]
        if len(touching) != 1:
            return False
        orbit_seen = set()
        for r, c in spec["current"] + trail:
            key = (r % tile.shape[0], c % tile.shape[1])
            orbit_seen.add(key)
            visible_copy = False
            for rr in range(r % tile.shape[0], base.shape[0], tile.shape[0]):
                for cc in range(c % tile.shape[1], base.shape[1], tile.shape[1]):
                    if not hidden[rr, cc]:
                        visible_copy = True
            if not visible_copy:
                return False
        if len(orbit_seen) < 2:
            return False
    return True


def _generate_attempt() -> np.ndarray | None:
    rng = random.Random()
    n_objects = rng.choices([1, 2, 3], weights=[2, 4, 2], k=1)[0]
    shape, zones = _sample_zones(n_objects, rng)
    tile = _sample_tile(rng)
    base = _tile_fill(shape, tile, rng.randrange(tile.shape[0]), rng.randrange(tile.shape[1]))

    object_colors = rng.sample([c for c in BACKGROUND_POOL if c not in set(np.unique(tile))], n_objects)
    specs = []
    for zone, color in zip(zones, object_colors):
        spec = _place_object_in_zone(zone, color, rng)
        if spec is None:
            return None
        specs.append(spec)

    inp = _render_input(base, specs)
    expected = _expected_output(base, specs)
    return inp if _validate_instance(base, inp, expected, specs, tile) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        inp = _generate_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid residual-trace ARC input")
