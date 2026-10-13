from __future__ import annotations

import math
import random
from collections import deque

import numpy as np


BG_COLORS = (0, 5)
ORANGE = 7
SQUARE_COLORS = (1, 2, 3, 4)
DIRECTIONS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
SIDE_CYCLE = {1: 9, 3: 1, 5: 3, 7: 5, 9: 7}

GRID_H_RANGE = (24, 30)
GRID_W_RANGE = (24, 30)
OBJECT_COUNT_RANGE = (3, 4)
CENTER_MARGIN = 6
MIN_CENTER_DIST2 = 100
MAX_GENERATION_ATTEMPTS = 400


def _checkerboard(shape: tuple[int, int]) -> np.ndarray:
    rows, cols = np.indices(shape)
    return np.where((rows + cols) % 2 == 0, 0, 5).astype(int)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors(r: int, c: int, connectivity: int) -> list[tuple[int, int]]:
    if connectivity == 4:
        return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
    ]


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors(rr, cc, connectivity):
                    if _in_bounds((h, w), nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _square_cells(center: tuple[int, int], side: int) -> list[tuple[int, int]]:
    cr, cc = center
    rad = side // 2
    return [(r, c) for r in range(cr - rad, cr + rad + 1) for c in range(cc - rad, cc + rad + 1)]


def _segment_cells(center: tuple[int, int], side: int, direction: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    cr, cc = center
    dr, dc = direction
    new_side = SIDE_CYCLE[side]
    outer_rad = max(side, new_side) // 2
    return (
        (cr + dr * (outer_rad + 1), cc + dc * outer_rad),
        (cr + dr * outer_rad, cc + dc * (outer_rad + 1)),
    )


def _paint_square(grid: np.ndarray, center: tuple[int, int], side: int, color: int) -> None:
    cr, cc = center
    rad = side // 2
    grid[cr - rad : cr + rad + 1, cc - rad : cc + rad + 1] = int(color)


def _spec_to_tuple(obj: dict) -> tuple[int, int, int, int, int]:
    cr, cc = obj["center"]
    dr, dc = obj["dir"]
    return int(cr), int(cc), int(obj["side"]), int(obj["color"]), 10 * dr + dc


def _extract_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    square_objects: list[dict] = []
    for color in sorted(set(int(v) for v in np.unique(g)) - set(BG_COLORS) - {ORANGE}):
        assert color in SQUARE_COLORS, "unexpected square color"
        comps = _components(g == color, connectivity=4)
        assert len(comps) == 1, "each square color must form one component"
        coords = comps[0]
        r0, c0, r1, c1 = _bbox(coords)
        side = int(r1 - r0 + 1)
        assert side == int(c1 - c0 + 1), "square bbox must be square"
        assert side in {1, 3, 5, 7}, "input squares must use odd sides from {1,3,5,7}"
        assert coords.shape[0] == side * side, "square must be solid"
        assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color), "square bbox must be fully filled"
        square_objects.append(
            {
                "center": ((r0 + r1) // 2, (c0 + c1) // 2),
                "side": side,
                "color": color,
            }
        )

    square_objects.sort(key=lambda obj: obj["center"])
    expected_segments: dict[frozenset[tuple[int, int]], list[tuple[int, tuple[int, int]]]] = {}
    for index, obj in enumerate(square_objects):
        for direction in DIRECTIONS:
            cells = frozenset(_segment_cells(obj["center"], obj["side"], direction))
            expected_segments.setdefault(cells, []).append((index, direction))

    orange_components = _components(g == ORANGE, connectivity=8)
    assert len(orange_components) == len(square_objects), "expected one 2-pixel orange segment per square"

    for comp in orange_components:
        assert comp.shape[0] == 2, "each orange marker must contain exactly two pixels"
        key = frozenset((int(r), int(c)) for r, c in comp)
        assert key in expected_segments, "orange segment must match one square corner sector"
        candidates = expected_segments[key]
        assert len(candidates) == 1, "orange segment must identify a unique square"
        index, direction = candidates[0]
        square_objects[index]["dir"] = direction
        square_objects[index]["segment"] = tuple(sorted(key))
    for obj in square_objects:
        assert "dir" in obj, "each square must own one orange segment"
    return square_objects


def _simulate_scene(
    objects: list[dict],
    shape: tuple[int, int],
    *,
    use_transformed_squares: bool,
) -> tuple[np.ndarray, list[tuple[tuple[int, int], ...]], list[str]]:
    out = _checkerboard(shape)
    for obj in objects:
        side = SIDE_CYCLE[obj["side"]] if use_transformed_squares else obj["side"]
        _paint_square(out, obj["center"], side, obj["color"])
    for obj in objects:
        for r, c in obj["segment"]:
            out[r, c] = ORANGE

    ray_paths: list[tuple[tuple[int, int], ...]] = []
    terminations: list[str] = []
    for obj in sorted(objects, key=_spec_to_tuple):
        dr, dc = obj["dir"]
        for endpoint in obj["segment"]:
            r, c = endpoint
            path: list[tuple[int, int]] = []
            while True:
                nr, nc = r + dr, c + dc
                if not _in_bounds(shape, nr, nc):
                    terminations.append("edge")
                    break
                if int(out[nr, nc]) in BG_COLORS:
                    out[nr, nc] = ORANGE
                    path.append((nr, nc))
                    r, c = nr, nc
                    continue
                terminations.append("ray" if int(out[nr, nc]) == ORANGE else "square")
                break
            ray_paths.append(tuple(path))
    return out, ray_paths, terminations


def main(grid: np.ndarray) -> np.ndarray:
    """
    Resize each odd square by the side-length cycle, then let its nearby diagonal marker emit outward corner rays.

    Input structure:
    - The background is a black/gray checkerboard with black at the top-left.
    - Foreground squares are separated solid odd-sided squares in distinct non-orange colors.
    - Each square has one orange 2-pixel diagonal marker in one corner sector.

    Rule:
    - Replace every square by a centered square of the same color using the side cycle 1->9, 3->1, 5->3, 7->5.
    - Keep the orange 2-pixel markers where they are.
    - From both endpoints of each marker, cast a diagonal orange ray in the outward corner direction, painting only checkerboard cells until the next step would leave the grid or hit a square or orange pixel.

    Key insight:
    - The squares must be resized before any ray is traced, because the transformed square footprints are the collision boundaries that decide each ray length.
    """
    objects = _extract_objects(np.asarray(grid, dtype=int))
    out, _ray_paths, _terminations = _simulate_scene(objects, grid.shape, use_transformed_squares=True)
    return out


def _sample_sides(count: int) -> list[int]:
    sides = [random.choice([1, 3, 5, 7]) for _ in range(count)]
    if 1 not in sides and random.random() < 0.75:
        sides[random.randrange(count)] = 1
    if 7 not in sides and random.random() < 0.75:
        candidates = [i for i in range(count) if sides[i] != 1] or list(range(count))
        sides[random.choice(candidates)] = 7
    if len(set(sides)) == 1:
        idx = random.randrange(count)
        sides[idx] = random.choice([side for side in [1, 3, 5, 7] if side != sides[idx]])
    return sides


def _sample_centers(shape: tuple[int, int], count: int) -> list[tuple[int, int]] | None:
    h, w = shape
    candidates = [
        (r, c)
        for r in range(CENTER_MARGIN, h - CENTER_MARGIN)
        for c in range(CENTER_MARGIN, w - CENTER_MARGIN)
    ]
    random.shuffle(candidates)
    centers: list[tuple[int, int]] = []
    for r, c in candidates:
        if all((r - rr) * (r - rr) + (c - cc) * (c - cc) >= MIN_CENTER_DIST2 for rr, cc in centers):
            centers.append((r, c))
            if len(centers) == count:
                return centers
    return None


def _build_objects(shape: tuple[int, int]) -> list[dict] | None:
    count = random.randint(*OBJECT_COUNT_RANGE)
    colors = random.sample(list(SQUARE_COLORS), count)
    sides = _sample_sides(count)
    centers = _sample_centers(shape, count)
    if centers is None:
        return None

    objects = []
    for center, side, color in zip(centers, sides, colors):
        direction = random.choice(DIRECTIONS)
        objects.append({"center": center, "side": side, "color": color, "dir": direction})
    random.shuffle(objects)
    return objects


def _all_cells(objects: list[dict], *, transformed: bool) -> list[set[tuple[int, int]]]:
    groups: list[set[tuple[int, int]]] = []
    for obj in objects:
        side = SIDE_CYCLE[obj["side"]] if transformed else obj["side"]
        groups.append(set(_square_cells(obj["center"], side)))
    return groups


def _render_input(objects: list[dict], shape: tuple[int, int]) -> np.ndarray:
    inp = _checkerboard(shape)
    for obj in objects:
        _paint_square(inp, obj["center"], obj["side"], obj["color"])
    for obj in objects:
        obj["segment"] = tuple(sorted(_segment_cells(obj["center"], obj["side"], obj["dir"])))
        for r, c in obj["segment"]:
            inp[r, c] = ORANGE
    return inp


def _validate_objects(objects: list[dict], shape: tuple[int, int]) -> bool:
    h, w = shape
    for obj in objects:
        orig_side = obj["side"]
        new_side = SIDE_CYCLE[orig_side]
        for side in (orig_side, new_side):
            for r, c in _square_cells(obj["center"], side):
                if not _in_bounds(shape, r, c):
                    return False
        seg = _segment_cells(obj["center"], orig_side, obj["dir"])
        if any(not _in_bounds(shape, r, c) for r, c in seg):
            return False

    original_squares = _all_cells(objects, transformed=False)
    transformed_squares = _all_cells(objects, transformed=True)
    for groups in (original_squares, transformed_squares):
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if groups[i] & groups[j]:
                    return False

    all_orig = set().union(*original_squares)
    all_new = set().union(*transformed_squares)
    seen_segments: set[tuple[int, int]] = set()
    for obj in objects:
        seg = set(_segment_cells(obj["center"], obj["side"], obj["dir"]))
        if seg & all_orig or seg & all_new:
            return False
        if seg & seen_segments:
            return False
        seen_segments |= seg

    inp = _render_input(objects, shape)
    orange_comps = _components(inp == ORANGE, connectivity=8)
    if len(orange_comps) != len(objects) or any(comp.shape[0] != 2 for comp in orange_comps):
        return False

    try:
        parsed = _extract_objects(inp)
    except AssertionError:
        return False
    parsed_specs = sorted(_spec_to_tuple(obj) for obj in parsed)
    wanted_specs = sorted(_spec_to_tuple(obj) for obj in objects)
    if parsed_specs != wanted_specs:
        return False

    transformed_out, new_paths, new_terms = _simulate_scene(parsed, shape, use_transformed_squares=True)
    original_out, old_paths, old_terms = _simulate_scene(parsed, shape, use_transformed_squares=False)
    if any(term == "ray" for term in new_terms):
        return False
    if any(term == "ray" for term in old_terms):
        return False

    total_ray_cells = sum(len(path) for path in new_paths)
    if total_ray_cells < 4 or total_ray_cells > (h * w) // 4:
        return False

    diffs = [abs(len(a) - len(b)) for a, b in zip(new_paths, old_paths) if a != b]
    if not diffs or max(diffs) < 2:
        return False

    if np.array_equal(transformed_out, original_out):
        return False
    if len({obj["dir"] for obj in objects}) == 1 and random.random() < 0.8:
        return False

    try:
        solved = main(inp)
    except AssertionError:
        return False
    if not np.array_equal(solved, transformed_out):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    shape = (random.randint(*GRID_H_RANGE), random.randint(*GRID_W_RANGE))
    objects = _build_objects(shape)
    if objects is None:
        return None
    if not _validate_objects(objects, shape):
        return None
    return _render_input(objects, shape)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-after-resize ARC input")
