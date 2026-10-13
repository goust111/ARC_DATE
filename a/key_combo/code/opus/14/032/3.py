from __future__ import annotations

import random

import numpy as np


BG = 0
MARKER = 9
OBJECT_COLORS = (1, 2, 3, 4)
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR_RANK = {(-1, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3}


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(grid: np.ndarray, *, bg: int | None = None, colors: set[int] | None = None) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    if bg is None:
        bg = _get_bg(g)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if seen[r, c] or color == bg:
                continue
            if colors is not None and color not in colors:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "area": int(arr.shape[0]),
                    "bbox": (
                        int(arr[:, 0].min()),
                        int(arr[:, 1].min()),
                        int(arr[:, 0].max()),
                        int(arr[:, 1].max()),
                    ),
                }
            )
    return comps


def _contact_directions(coords: np.ndarray, marker_set: set[tuple[int, int]]) -> set[tuple[int, int]]:
    dirs = set()
    for r, c in coords:
        rr, cc = int(r), int(c)
        for dr, dc in DIRS:
            if (rr + dr, cc + dc) in marker_set:
                dirs.add((dr, dc))
    return dirs


def _steps_to_edge(shape: tuple[int, int], coords: np.ndarray, outward: tuple[int, int]) -> int:
    h, w = shape
    dr, dc = outward
    if dr < 0:
        return int(coords[:, 0].min())
    if dr > 0:
        return int(h - 1 - coords[:, 0].max())
    if dc < 0:
        return int(coords[:, 1].min())
    return int(w - 1 - coords[:, 1].max())


def _shifted(coords: np.ndarray, dr: int, dc: int, steps: int) -> np.ndarray:
    return coords + np.array([dr * steps, dc * steps], dtype=int)


def _analyze_objects(grid: np.ndarray) -> tuple[np.ndarray, int, dict, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    assert bg == BG

    marker_components = _components(g, bg=bg, colors={MARKER})
    assert len(marker_components) == 1
    marker = marker_components[0]
    assert 4 <= marker["area"] <= 8
    marker_set = {tuple(map(int, xy)) for xy in marker["coords"].tolist()}

    objects = []
    for comp in _components(g, bg=bg):
        if int(comp["color"]) == MARKER:
            continue
        contact_dirs = _contact_directions(comp["coords"], marker_set)
        assert len(contact_dirs) == 1
        direction_to_marker = next(iter(contact_dirs))
        outward = (-direction_to_marker[0], -direction_to_marker[1])
        obj = dict(comp)
        obj["contact"] = direction_to_marker
        obj["outward"] = outward
        objects.append(obj)

    assert 3 <= len(objects) <= 4
    assert len({int(obj["color"]) for obj in objects}) == len(objects)
    assert len({int(obj["area"]) for obj in objects}) == len(objects)
    assert len({obj["contact"] for obj in objects}) >= 3
    return g, bg, marker, objects


def _build_output_for_order(g: np.ndarray, bg: int, marker: dict, objects: list[dict], order: list[int] | None) -> np.ndarray:
    out = g.copy()
    for obj in objects:
        coords = obj["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    if order is None:
        mapping = {idx: int(obj["color"]) for idx, obj in enumerate(objects)}
    else:
        ordered = [objects[idx] for idx in order]
        mapping = {
            order[i]: int(ordered[(i + 1) % len(ordered)]["color"])
            for i in range(len(ordered))
        }

    occupied = {tuple(map(int, xy)) for xy in marker["coords"].tolist()}
    for idx, obj in enumerate(objects):
        dr, dc = obj["outward"]
        steps = _steps_to_edge(g.shape, obj["coords"], (dr, dc))
        assert steps >= 0
        moved = _shifted(obj["coords"], dr, dc, steps)
        if dr < 0:
            assert int(moved[:, 0].min()) == 0
        elif dr > 0:
            assert int(moved[:, 0].max()) == g.shape[0] - 1
        elif dc < 0:
            assert int(moved[:, 1].min()) == 0
        else:
            assert int(moved[:, 1].max()) == g.shape[1] - 1
        for r, c in moved:
            cell = (int(r), int(c))
            assert cell not in occupied
            occupied.add(cell)
        out[moved[:, 0], moved[:, 1]] = mapping[idx]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push each marker-touching object to the opposite edge, then cyclically rotate their colors by ascending area.

    Input structure:
    - The background is black.
    - A single compact maroon 4-connected marker cluster sits near the center.
    - Three or four other monochrome 4-connected objects touch the marker on exactly one cardinal side each.

    Rule:
    - Record every non-maroon object's original color and area.
    - Sort those objects by area from smallest to largest.
    - Erase them from the input, push each whole shape straight away from its marker-contact side until it reaches the grid boundary, and keep the maroon marker fixed.
    - Recolor the pushed shapes cyclically along the area order: smallest gets the second-smallest's color, ..., largest gets the smallest's color.

    Key insight:
    - Contact side determines where a shape moves, while area ordering independently determines which color it receives after moving.
    """
    g, bg, marker, objects = _analyze_objects(grid)
    order = [idx for idx, _obj in sorted(enumerate(objects), key=lambda item: int(item[1]["area"]))]
    return _build_output_for_order(g, bg, marker, objects, order)


def _normalize(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    return sorted((r - min_r, c - min_c) for r, c in cells)


def _random_polyomino(area: int, rng: random.Random, *, max_side: int, min_span: int = 2) -> list[tuple[int, int]] | None:
    for _ in range(80):
        cells = {(0, 0)}
        while len(cells) < area:
            base_r, base_c = rng.choice(tuple(cells))
            candidates = [
                (base_r - 1, base_c),
                (base_r + 1, base_c),
                (base_r, base_c - 1),
                (base_r, base_c + 1),
            ]
            rng.shuffle(candidates)
            for cand in candidates:
                if cand not in cells:
                    cells.add(cand)
                    break
        shape = _normalize(cells)
        arr = np.array(shape, dtype=int)
        height = int(arr[:, 0].max()) + 1
        width = int(arr[:, 1].max()) + 1
        if height > max_side or width > max_side:
            continue
        if min(height, width) < min_span:
            continue
        if area >= 5 and area == height * width:
            continue
        return shape
    return None


def _place_marker(rng: random.Random, h: int, w: int) -> list[tuple[int, int]] | None:
    area = rng.randint(4, 8)
    shape = _random_polyomino(area, rng, max_side=4, min_span=2)
    if shape is None:
        return None
    arr = np.array(shape, dtype=int)
    sh = int(arr[:, 0].max()) + 1
    sw = int(arr[:, 1].max()) + 1
    r_lo = max(3, h // 2 - sh // 2 - 1)
    r_hi = min(h - sh - 4, h // 2 - sh // 2 + 1)
    c_lo = max(3, w // 2 - sw // 2 - 1)
    c_hi = min(w - sw - 4, w // 2 - sw // 2 + 1)
    if r_lo > r_hi or c_lo > c_hi:
        return None
    r0 = rng.randint(r_lo, r_hi)
    c0 = rng.randint(c_lo, c_hi)
    return [(r0 + r, c0 + c) for r, c in shape]


def _touches_existing(coords: set[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    for r, c in coords:
        if (r, c) in occupied:
            return True
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in occupied:
                return True
    return False


def _contact_candidates(marker_set: set[tuple[int, int]], shape: tuple[int, int], direction_to_marker: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    dr, dc = direction_to_marker
    out = []
    for mr, mc in marker_set:
        rr = mr - dr
        cc = mc - dc
        if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in marker_set:
            out.append((rr, cc))
    return out


def _try_place_object(
    rng: random.Random,
    shape: tuple[int, int],
    marker_set: set[tuple[int, int]],
    occupied_input_nonmarker: set[tuple[int, int]],
    occupied_output_nonmarker: set[tuple[int, int]],
    direction_to_marker: tuple[int, int],
    area: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    h, w = shape
    outward = (-direction_to_marker[0], -direction_to_marker[1])
    contact_positions = _contact_candidates(marker_set, shape, direction_to_marker)
    if not contact_positions:
        return None

    for _ in range(90):
        poly = _random_polyomino(area, rng, max_side=5, min_span=2)
        if poly is None:
            continue
        anchors = poly[:]
        rng.shuffle(anchors)
        rng.shuffle(contact_positions)
        for ar, ac in anchors:
            for cr, cc in contact_positions:
                coords = np.array([(cr - ar + r, cc - ac + c) for r, c in poly], dtype=int)
                if not (
                    np.all((0 <= coords[:, 0]) & (coords[:, 0] < h))
                    and np.all((0 <= coords[:, 1]) & (coords[:, 1] < w))
                ):
                    continue
                coord_set = {tuple(map(int, xy)) for xy in coords.tolist()}
                if coord_set & marker_set:
                    continue
                if _touches_existing(coord_set, occupied_input_nonmarker):
                    continue
                if _contact_directions(coords, marker_set) != {direction_to_marker}:
                    continue
                steps = _steps_to_edge(shape, coords, outward)
                if steps < 2:
                    continue
                moved = _shifted(coords, outward[0], outward[1], steps)
                moved_set = {tuple(map(int, xy)) for xy in moved.tolist()}
                if _touches_existing(moved_set, occupied_output_nonmarker):
                    continue
                if moved_set & marker_set:
                    continue
                if _contact_directions(moved, marker_set):
                    continue
                return coords, moved
    return None


def _order_ids_by(objects: list[dict], key) -> tuple[int, ...]:
    return tuple(idx for idx, _obj in sorted(enumerate(objects), key=lambda item: key(item[1])))


def _semantic_checks(inp: np.ndarray) -> bool:
    try:
        g, bg, marker, objects = _analyze_objects(inp)
    except AssertionError:
        return False

    areas = [int(obj["area"]) for obj in objects]
    if not all(3 <= area <= 8 for area in areas):
        return False
    if not all(_steps_to_edge(g.shape, obj["coords"], obj["outward"]) >= 2 for obj in objects):
        return False

    area_order = _order_ids_by(objects, lambda obj: int(obj["area"]))
    row_major = _order_ids_by(
        objects,
        lambda obj: (
            float(obj["coords"][:, 0].mean()),
            float(obj["coords"][:, 1].mean()),
        ),
    )
    col_major = _order_ids_by(
        objects,
        lambda obj: (
            float(obj["coords"][:, 1].mean()),
            float(obj["coords"][:, 0].mean()),
        ),
    )
    dir_order = _order_ids_by(objects, lambda obj: DIR_RANK[obj["contact"]])
    if area_order in {row_major, col_major, dir_order}:
        return False

    true_out = _build_output_for_order(g, bg, marker, objects, list(area_order))
    if np.array_equal(true_out, inp):
        return False
    if np.array_equal(true_out, _build_output_for_order(g, bg, marker, objects, None)):
        return False
    if np.array_equal(true_out, _build_output_for_order(g, bg, marker, objects, list(dir_order))):
        return False
    if np.array_equal(true_out, _build_output_for_order(g, bg, marker, objects, list(row_major))):
        return False
    if np.array_equal(true_out, _build_output_for_order(g, bg, marker, objects, list(col_major))):
        return False
    return np.array_equal(main(inp), true_out)


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(13, 19)
    w = rng.randint(13, 19)
    grid = np.full((h, w), BG, dtype=int)

    marker_cells = _place_marker(rng, h, w)
    if marker_cells is None:
        return None
    marker_set = set(marker_cells)
    for r, c in marker_cells:
        grid[r, c] = MARKER

    n_objects = 4 if rng.random() < 0.45 else 3
    directions = rng.sample(DIRS, n_objects)
    areas = rng.sample(range(3, 9), n_objects)
    colors = rng.sample(OBJECT_COLORS, n_objects)
    specs = list(zip(directions, areas, colors))
    rng.shuffle(specs)
    specs.sort(key=lambda spec: -spec[1])

    occupied_input_nonmarker: set[tuple[int, int]] = set()
    occupied_output_nonmarker: set[tuple[int, int]] = set()

    for direction_to_marker, area, color in specs:
        placed = _try_place_object(
            rng,
            (h, w),
            marker_set,
            occupied_input_nonmarker,
            occupied_output_nonmarker,
            direction_to_marker,
            area,
        )
        if placed is None:
            return None
        coords, moved = placed
        coord_set = {tuple(map(int, xy)) for xy in coords.tolist()}
        moved_set = {tuple(map(int, xy)) for xy in moved.tolist()}
        occupied_input_nonmarker.update(coord_set)
        occupied_output_nonmarker.update(moved_set)
        grid[coords[:, 0], coords[:, 1]] = int(color)

    return grid if _semantic_checks(grid) else None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(500):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
