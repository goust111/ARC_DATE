from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BACKGROUND = 0
CHAIN_COLOR_POOL = tuple(range(1, 10))
PASSENGER_COLOR_POOL = tuple(range(1, 10))
GRID_SIDE_RANGE = (12, 26)
CHAIN_HEIGHT_RANGE = (2, 5)
CHAIN_WIDTH_RANGE = (2, 5)
PASSENGER_AREA_RANGE = (3, 7)
MAX_RETRIES = 600


def get_bg(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _colors_present(grid: np.ndarray, bg: int) -> list[int]:
    return sorted(int(color) for color in np.unique(grid) if int(color) != bg)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _component_dicts(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.array(cells, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _normalize(coords: np.ndarray) -> frozenset[tuple[int, int]]:
    r0, c0, _r1, _c1 = bbox_coords(coords)
    return frozenset((int(r - r0), int(c - c0)) for r, c in coords)


def _anchor(coords: np.ndarray) -> tuple[int, int]:
    r0, c0, _r1, _c1 = bbox_coords(coords)
    return int(r0), int(c0)


def _to_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in coords}


def _translate_set(coords: np.ndarray, dr: int, dc: int) -> set[tuple[int, int]]:
    return {(int(r + dr), int(c + dc)) for r, c in coords}


def _clip_translate(coords: np.ndarray, dr: int, dc: int, shape: tuple[int, int]) -> tuple[set[tuple[int, int]], bool]:
    h, w = map(int, shape)
    visible: set[tuple[int, int]] = set()
    clipped = False
    for r, c in coords:
        rr = int(r + dr)
        cc = int(c + dc)
        if 0 <= rr < h and 0 <= cc < w:
            visible.add((rr, cc))
        else:
            clipped = True
    return visible, clipped


def _touches_edge(coords: np.ndarray | set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = map(int, shape)
    iterable = coords if isinstance(coords, set) else _to_set(coords)
    return any(r == 0 or r == h - 1 or c == 0 or c == w - 1 for r, c in iterable)


def _touches_edge_bbox(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = map(int, shape)
    r0, c0, r1, c1 = bbox_coords(coords)
    return r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1


def _is_connected(cells: frozenset[tuple[int, int]]) -> bool:
    start = next(iter(cells))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(cells)


def _touching4(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    if a & b:
        return True
    for r, c in a:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in b:
                return True
    return False


def _max_steps_in_bounds(coords: np.ndarray, delta: tuple[int, int], shape: tuple[int, int]) -> int:
    dr, dc = map(int, delta)
    h, w = map(int, shape)
    limit = float("inf")
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if dr > 0:
            limit = min(limit, (h - 1 - rr) // dr)
        elif dr < 0:
            limit = min(limit, rr // (-dr))
        if dc > 0:
            limit = min(limit, (w - 1 - cc) // dc)
        elif dc < 0:
            limit = min(limit, cc // (-dc))
    return int(limit if limit != float("inf") else 0)


def _paint_coords(shape: tuple[int, int], coords: np.ndarray, color: int) -> np.ndarray:
    out = np.full(shape, BACKGROUND, dtype=int)
    for r, c in coords:
        out[int(r), int(c)] = int(color)
    return out


def _scene_from_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))
    comps = _component_dicts(g, bg)
    assert comps, "expected foreground objects"

    by_color: dict[int, list[dict]] = defaultdict(list)
    for comp in comps:
        by_color[int(comp["color"])].append(comp)

    chain_matches: list[dict] = []
    for color, color_comps in by_color.items():
        if len(color_comps) != 3:
            continue
        ordered = sorted(color_comps, key=lambda comp: (-int(comp["area"]), _anchor(comp["coords"])))
        if int(ordered[0]["area"]) != int(ordered[1]["area"]) or int(ordered[2]["area"]) >= int(ordered[1]["area"]):
            continue
        if _normalize(ordered[0]["coords"]) != _normalize(ordered[1]["coords"]):
            continue
        if not _touches_edge(ordered[2]["coords"], g.shape):
            continue

        a = ordered[0]
        b = ordered[1]
        residue = ordered[2]
        ar, ac = _anchor(a["coords"])
        br, bc = _anchor(b["coords"])
        step = (br - ar, bc - ac)
        if step == (0, 0):
            continue

        residue_set = _to_set(residue["coords"])
        candidates: list[tuple[int, int]] = []
        for dr, dc in (step, (-step[0], -step[1])):
            front = max((a, b), key=lambda comp: _anchor(comp["coords"])[0] * dr + _anchor(comp["coords"])[1] * dc)
            visible, clipped = _clip_translate(front["coords"], dr, dc, g.shape)
            if clipped and visible == residue_set:
                candidates.append((int(dr), int(dc)))
        if len(candidates) != 1:
            continue

        chain_matches.append(
            {
                "chain_color": int(color),
                "vector": candidates[0],
                "chain_components": ordered,
            }
        )

    assert len(chain_matches) == 1, "expected exactly one translation chain"
    chain = chain_matches[0]

    passenger_comps = [comp for comp in comps if int(comp["color"]) != int(chain["chain_color"])]
    assert len(passenger_comps) == 1, "expected exactly one passenger object"
    passenger = passenger_comps[0]
    assert _normalize(passenger["coords"]) != _normalize(chain["chain_components"][0]["coords"])

    return {
        "bg": bg,
        "chain_color": int(chain["chain_color"]),
        "vector": tuple(map(int, chain["vector"])),
        "chain_components": chain["chain_components"],
        "passenger": passenger,
    }


def _expected_output(grid: np.ndarray, vector: tuple[int, int] | None = None) -> np.ndarray:
    scene = _scene_from_grid(grid)
    passenger = scene["passenger"]
    dr, dc = scene["vector"] if vector is None else tuple(map(int, vector))
    k = _max_steps_in_bounds(passenger["coords"], (dr, dc), grid.shape)
    moved = passenger["coords"] + np.array([[k * dr, k * dc]], dtype=int)
    return _paint_coords(grid.shape, moved, int(passenger["color"]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the passenger object by the vector encoded by a three-copy chain, stopping at the furthest in-bounds multiple.

    Input structure:
    - A black background contains one same-colored object copied three times along an arithmetic translation chain.
    - Two copies are complete, while the third lies against the grid boundary and is visibly clipped.
    - A second object of a different color and different shape is the passenger to be moved.

    Rule:
    - Infer the chain step vector from the offset between the two complete copies.
    - Use the clipped residue to choose the direction: the vector points from the complete-copy side toward the clipped-copy side.
    - Erase the chain copies and translate the passenger by the largest non-negative integer multiple of that vector that keeps every passenger cell inside the grid.

    Key insight:
    - The repeated object is not the thing to transform; it is a drawn displacement arrow whose spacing gives the step size and whose clipped endpoint fixes the sign.
    """
    g = np.asarray(grid, dtype=int)
    scene = _scene_from_grid(g)
    return _expected_output(g, scene["vector"])


def _random_polyomino(area: int) -> frozenset[tuple[int, int]]:
    cells = {(0, 0)}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) not in cells:
                    frontier.append((nr, nc))
        cells.add(random.choice(frontier))
    r0 = min(r for r, _c in cells)
    c0 = min(c for _r, c in cells)
    normalized = frozenset((r - r0, c - c0) for r, c in cells)
    assert _is_connected(normalized)
    return normalized


def _rectangle_shape(height: int, width: int) -> frozenset[tuple[int, int]]:
    return frozenset((r, c) for r in range(height) for c in range(width))


def _shape_bbox(shape: frozenset[tuple[int, int]]) -> tuple[int, int]:
    height = 1 + max(r for r, _c in shape)
    width = 1 + max(c for _r, c in shape)
    return int(height), int(width)


def _place_shape(shape: frozenset[tuple[int, int]], anchor: tuple[int, int]) -> set[tuple[int, int]]:
    ar, ac = map(int, anchor)
    return {(ar + r, ac + c) for r, c in shape}


def _shape_to_coords(shape: frozenset[tuple[int, int]], anchor: tuple[int, int]) -> np.ndarray:
    return np.array(sorted(_place_shape(shape, anchor)), dtype=int)


def _full_in_bounds(shape: frozenset[tuple[int, int]], anchor: tuple[int, int], grid_shape: tuple[int, int]) -> bool:
    h, w = map(int, grid_shape)
    for r, c in _place_shape(shape, anchor):
        if not (0 <= r < h and 0 <= c < w):
            return False
    return True


def _visible_after_clip(shape: frozenset[tuple[int, int]], anchor: tuple[int, int], grid_shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = map(int, grid_shape)
    visible = set()
    for r, c in _place_shape(shape, anchor):
        if 0 <= r < h and 0 <= c < w:
            visible.add((r, c))
    return visible


def _candidate_vectors() -> list[tuple[int, int]]:
    vectors = []
    for dr in range(-6, 7):
        for dc in range(-6, 7):
            if dr == 0 and dc == 0:
                continue
            if max(abs(dr), abs(dc)) < 2:
                continue
            vectors.append((dr, dc))
    random.shuffle(vectors)
    return vectors


def _nonflush(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    return not _touches_edge_bbox(coords, shape)


def _generate_case() -> tuple[np.ndarray, dict]:
    for _ in range(MAX_RETRIES):
        h = random.randint(*GRID_SIDE_RANGE)
        w = random.randint(*GRID_SIDE_RANGE)
        grid_shape = (h, w)

        chain_h = random.randint(*CHAIN_HEIGHT_RANGE)
        chain_w = random.randint(*CHAIN_WIDTH_RANGE)
        chain_shape = _rectangle_shape(chain_h, chain_w)

        passenger_area = random.randint(*PASSENGER_AREA_RANGE)
        passenger_shape = _random_polyomino(passenger_area)
        if passenger_shape == chain_shape:
            continue

        chain_color, passenger_color = random.sample(CHAIN_COLOR_POOL, 2)
        want_zero = random.random() < 0.15
        min_steps = 0 if want_zero else (2 if random.random() < 0.55 else 1)
        want_nonflush = random.random() < 0.7

        chain_cells = set(chain_shape)
        vector_found = None
        full0_anchor = full1_anchor = None
        residue_cells: set[tuple[int, int]] | None = None

        for dr, dc in _candidate_vectors():
            for r1 in range(1, h - chain_h):
                for c1 in range(1, w - chain_w):
                    p1 = (r1, c1)
                    p0 = (r1 - dr, c1 - dc)
                    if not _full_in_bounds(chain_shape, p0, grid_shape):
                        continue
                    if _touches_edge(_place_shape(chain_shape, p0), grid_shape) or _touches_edge(_place_shape(chain_shape, p1), grid_shape):
                        continue
                    p2 = (r1 + dr, c1 + dc)
                    full0 = _place_shape(chain_shape, p0)
                    full1 = _place_shape(chain_shape, p1)
                    if _touching4(full0, full1):
                        continue
                    residue = _visible_after_clip(chain_shape, p2, grid_shape)
                    if not residue or len(residue) >= len(chain_cells):
                        continue
                    if not _touches_edge(residue, grid_shape):
                        continue
                    if _touching4(residue, full0) or _touching4(residue, full1):
                        continue
                    vector_found = (dr, dc)
                    full0_anchor = p0
                    full1_anchor = p1
                    residue_cells = residue
                    break
                if vector_found is not None:
                    break
            if vector_found is not None:
                break

        if vector_found is None or full0_anchor is None or full1_anchor is None or residue_cells is None:
            continue

        full0 = _place_shape(chain_shape, full0_anchor)
        full1 = _place_shape(chain_shape, full1_anchor)
        chain_occupied = full0 | full1 | residue_cells

        passenger_h, passenger_w = _shape_bbox(passenger_shape)
        passenger_choices: list[tuple[tuple[int, int], int, np.ndarray]] = []
        for pr in range(h - passenger_h + 1):
            for pc in range(w - passenger_w + 1):
                anchor = (pr, pc)
                cells = _place_shape(passenger_shape, anchor)
                if _touching4(cells, chain_occupied):
                    continue
                coords = _shape_to_coords(passenger_shape, anchor)
                k_plus = _max_steps_in_bounds(coords, vector_found, grid_shape)
                k_minus = _max_steps_in_bounds(coords, (-vector_found[0], -vector_found[1]), grid_shape)
                if want_zero:
                    if k_plus != 0:
                        continue
                else:
                    if k_plus < min_steps:
                        continue
                plus_coords = coords + np.array([[k_plus * vector_found[0], k_plus * vector_found[1]]], dtype=int)
                minus_coords = coords + np.array([[-k_minus * vector_found[0], -k_minus * vector_found[1]]], dtype=int)
                if set(map(tuple, plus_coords)) == set(map(tuple, minus_coords)):
                    continue
                if want_nonflush and not _nonflush(plus_coords, grid_shape):
                    continue
                passenger_choices.append((anchor, k_plus, plus_coords))

        if not passenger_choices:
            continue

        passenger_anchor, k_plus, plus_coords = random.choice(passenger_choices)
        passenger_cells = _place_shape(passenger_shape, passenger_anchor)
        if _touching4(passenger_cells, chain_occupied):
            continue

        grid = np.full(grid_shape, BACKGROUND, dtype=int)
        for r, c in full0:
            grid[r, c] = chain_color
        for r, c in full1:
            grid[r, c] = chain_color
        for r, c in residue_cells:
            grid[r, c] = chain_color
        for r, c in passenger_cells:
            grid[r, c] = passenger_color

        expected = np.full(grid_shape, BACKGROUND, dtype=int)
        for r, c in plus_coords:
            expected[int(r), int(c)] = passenger_color

        if not np.array_equal(main(grid), expected):
            continue
        parsed = _scene_from_grid(grid)
        if tuple(parsed["vector"]) != tuple(vector_found):
            continue
        if not np.array_equal(_expected_output(grid, (-vector_found[0], -vector_found[1])), expected):
            pass
        else:
            continue

        return grid, {
            "vector": tuple(vector_found),
            "k": int(k_plus),
            "expected": expected,
            "chain_color": int(chain_color),
            "passenger_color": int(passenger_color),
        }

    raise RuntimeError("failed to generate a valid case")


def generate_input() -> np.ndarray:
    return _generate_case()[0]
