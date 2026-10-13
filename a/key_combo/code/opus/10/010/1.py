from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
BAR = 5
CONTROL = 9
SQUARE_COLORS = (1, 2, 3, 4, 7)
SIDES = (1, 3, 5, 7, 9)
SHRINK_SIDE = {1: 9, 3: 1, 5: 3, 7: 5, 9: 7}
GROW_SIDE = {1: 3, 3: 5, 5: 7, 7: 9, 9: 1}

BASE_PATTERN_SIZE = 23
BASE_CENTERS = {
    "nw": (6, 6),
    "ne": (6, 18),
    "center": (12, 12),
    "sw": (18, 6),
    "se": (18, 18),
}
CORNER_KEYS = ("nw", "ne", "sw", "se")
DIRECTIONS = ("up", "down", "left", "right")
SIDE_WEIGHTS = (3, 2, 3, 2, 3)
MAX_GENERATION_ATTEMPTS = 600


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _square_box(center: tuple[int, int], side: int) -> tuple[int, int, int, int]:
    half = side // 2
    r, c = center
    return r - half, c - half, r + half, c + half


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _paint_square(grid: np.ndarray, center: tuple[int, int], side: int, color: int) -> None:
    r0, c0, r1, c1 = _square_box(center, side)
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _marker_layout(shape: tuple[int, int], direction: str, line: int):
    h, w = shape
    if direction == "down":
        center = (1, line)
        control = (0, line)
        bar = [(1, line - 1), (1, line), (1, line + 1)]
        step = (1, 0)
    elif direction == "up":
        center = (h - 2, line)
        control = (h - 1, line)
        bar = [(h - 2, line - 1), (h - 2, line), (h - 2, line + 1)]
        step = (-1, 0)
    elif direction == "right":
        center = (line, 1)
        control = (line, 0)
        bar = [(line - 1, 1), (line, 1), (line + 1, 1)]
        step = (0, 1)
    else:
        assert direction == "left"
        center = (line, w - 2)
        control = (line, w - 1)
        bar = [(line - 1, w - 2), (line, w - 2), (line + 1, w - 2)]
        step = (0, -1)
    return center, control, bar, step


def _ray_cells(shape: tuple[int, int], center: tuple[int, int], step: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    cells: set[tuple[int, int]] = set()
    r, c = center[0] + step[0], center[1] + step[1]
    while 0 <= r < h and 0 <= c < w:
        cells.add((r, c))
        r += step[0]
        c += step[1]
    return cells


def _find_marker(grid: np.ndarray):
    bar_coords = np.argwhere(grid == BAR)
    assert bar_coords.shape[0] == 3, "expected exactly one gray three-pixel marker bar"
    rows = sorted(set(int(v) for v in bar_coords[:, 0]))
    cols = sorted(set(int(v) for v in bar_coords[:, 1]))
    if len(rows) == 1 and len(cols) == 3 and cols == list(range(cols[0], cols[0] + 3)):
        center = (rows[0], cols[1])
        axis = "horizontal"
    elif len(cols) == 1 and len(rows) == 3 and rows == list(range(rows[0], rows[0] + 3)):
        center = (rows[1], cols[0])
        axis = "vertical"
    else:
        raise AssertionError("marker bar must be a straight contiguous segment")

    control_coords = np.argwhere(grid == CONTROL)
    assert control_coords.shape[0] == 1, "expected exactly one maroon control pixel"
    control = tuple(int(v) for v in control_coords[0])
    dr = center[0] - control[0]
    dc = center[1] - control[1]
    assert abs(dr) + abs(dc) == 1, "control pixel must be adjacent to the bar center"
    if axis == "horizontal":
        assert dc == 0, "horizontal marker bars emit vertically"
    else:
        assert dr == 0, "vertical marker bars emit horizontally"
    return center, control, (dr, dc)


def _find_square_objects(grid: np.ndarray) -> list[dict]:
    colors = [int(color) for color in np.unique(grid) if int(color) not in (BG, BAR, CONTROL)]
    objects: list[dict] = []
    for color in colors:
        comps = _components(grid == color)
        assert len(comps) == 1, "each square color must form exactly one component"
        coords = comps[0]
        r0, c0, r1, c1 = _bbox(coords)
        side = r1 - r0 + 1
        assert side == c1 - c0 + 1, "square bbox must be square"
        assert side in SIDES, "square side length must be in {1,3,5,7,9}"
        assert coords.shape[0] == side * side, "square must be solid"
        assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color), "square must fill its bbox"
        objects.append(
            {
                "color": color,
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "center": ((r0 + r1) // 2, (c0 + c1) // 2),
                "side": side,
            }
        )
    assert 4 <= len(objects) <= 5, "generator uses four or five distinct square colors"
    return objects


def _transform_objects(objects: list[dict], ray: set[tuple[int, int]]) -> list[dict]:
    transformed = []
    for obj in objects:
        hit = any((int(r), int(c)) in ray for r, c in obj["coords"])
        new_side = SHRINK_SIDE[obj["side"]] if hit else GROW_SIDE[obj["side"]]
        transformed.append(
            {
                "color": obj["color"],
                "center": obj["center"],
                "side": new_side,
                "hit": hit,
            }
        )
    return transformed


def _render_objects(
    shape: tuple[int, int],
    objects: list[dict],
    *,
    bg: int = BG,
    order: str = "canonical",
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    items = sorted(objects, key=lambda obj: (obj["center"][0], obj["center"][1]))
    if order == "reverse":
        items = list(reversed(items))
    for obj in items:
        _paint_square(out, obj["center"], obj["side"], obj["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Resize each odd solid square in the direction chosen by whether the marker ray hits it.

    Input structure:
    - The grid has a black background, one gray three-pixel horizontal or vertical marker bar, and one maroon control pixel adjacent to the bar center.
    - The remaining foreground objects are four or five separated monochrome solid squares with odd side lengths from {1, 3, 5, 7, 9}.
    - Each square keeps a fixed center and unique color; the marker determines one cardinal ray.

    Rule:
    - Start at the gray bar center and cast a one-pixel-wide ray in the direction away from the maroon control pixel until the grid edge.
    - A square is hit if any of its pixels lies on that ray.
    - Clear the canvas to black, then redraw every hit square at the same center and color with side length n -> n-2, wrapping 1 -> 9.
    - Redraw every non-hit square at the same center and color with side length n -> n+2, wrapping 9 -> 1, using center order from top-left to bottom-right so later squares overwrite earlier ones.

    Key insight:
    - The ray is a binary partitioner: it does not paint anything itself, but it decides whether each square follows the shrinking cycle or the growing cycle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"
    center, _control, step = _find_marker(g)
    ray = _ray_cells(g.shape, center, step)
    objects = _find_square_objects(g)
    transformed = _transform_objects(objects, ray)
    return _render_objects(g.shape, transformed, bg=BG, order="canonical")


def _status_for_direction(center: tuple[int, int], direction: str, line: int) -> bool:
    return center[1] == line if direction in ("up", "down") else center[0] == line


def _candidate_rays(selected: dict[str, tuple[int, int]]) -> list[tuple[str, int, dict[str, bool]]]:
    candidates = []
    for direction in DIRECTIONS:
        values = sorted({center[1] if direction in ("up", "down") else center[0] for center in selected.values()})
        for line in values:
            status_map = {name: _status_for_direction(center, direction, line) for name, center in selected.items()}
            hit_count = sum(status_map.values())
            if 0 < hit_count < len(selected):
                candidates.append((direction, line, status_map))
    return candidates


def _sample_side(rng: random.Random) -> int:
    return rng.choices(SIDES, weights=SIDE_WEIGHTS, k=1)[0]


def _input_boxes_disjoint(objects: list[dict]) -> bool:
    boxes = [_square_box(obj["center"], obj["side"]) for obj in objects]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if _boxes_overlap(boxes[i], boxes[j]):
                return False
    return True


def _overlap_footprint(objects: list[dict]) -> int:
    boxes = [_square_box(obj["center"], obj["side"]) for obj in objects]
    h = max(box[2] for box in boxes) + 1
    w = max(box[3] for box in boxes) + 1
    counts = np.zeros((h, w), dtype=int)
    for obj in objects:
        r0, c0, r1, c1 = _square_box(obj["center"], obj["side"])
        counts[r0 : r1 + 1, c0 : c1 + 1] += 1
    return int(np.count_nonzero(counts > 1))


def _all_hit_output(shape: tuple[int, int], objects: list[dict]) -> np.ndarray:
    alt = []
    for obj in objects:
        alt.append({"color": obj["color"], "center": obj["center"], "side": SHRINK_SIDE[obj["side"]]})
    return _render_objects(shape, alt, bg=BG, order="canonical")


def _all_miss_output(shape: tuple[int, int], objects: list[dict]) -> np.ndarray:
    alt = []
    for obj in objects:
        alt.append({"color": obj["color"], "center": obj["center"], "side": GROW_SIDE[obj["side"]]})
    return _render_objects(shape, alt, bg=BG, order="canonical")


def _build_input_grid(
    shape: tuple[int, int],
    objects: list[dict],
    direction: str,
    line: int,
) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for obj in objects:
        _paint_square(grid, obj["center"], obj["side"], obj["color"])
    _center, control, bar, _step = _marker_layout(shape, direction, line)
    for r, c in bar:
        grid[r, c] = BAR
    grid[control] = CONTROL
    return grid


def _validate_generated_input(grid: np.ndarray) -> None:
    out = main(grid)
    center, _control, step = _find_marker(grid)
    ray = _ray_cells(grid.shape, center, step)
    objects = _find_square_objects(grid)
    hits = [obj for obj in objects if any((int(r), int(c)) in ray for r, c in obj["coords"])]
    misses = [obj for obj in objects if obj not in hits]
    assert hits and misses, "ray partition must produce both hit and non-hit groups"
    assert {obj["side"] for obj in hits} & {obj["side"] for obj in misses}, "at least one side length must appear in both groups"

    transformed = _transform_objects(objects, ray)
    canonical = _render_objects(grid.shape, transformed, bg=BG, order="canonical")
    reverse = _render_objects(grid.shape, transformed, bg=BG, order="reverse")
    assert np.array_equal(out, canonical), "main output must match the direct transformation"
    assert not np.array_equal(out, grid), "transformation must change the scene"
    assert not np.array_equal(out, _all_hit_output(grid.shape, objects)), "ray hit/miss split must matter"
    assert not np.array_equal(out, _all_miss_output(grid.shape, objects)), "ray hit/miss split must matter"
    assert not np.array_equal(canonical, reverse), "paint order must affect the final output"
    overlap_pixels = _overlap_footprint(transformed)
    assert 1 <= overlap_pixels <= 9, "output overlaps should exist but stay visually local"


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(BASE_PATTERN_SIZE, 30)
    w = rng.randint(BASE_PATTERN_SIZE, 30)
    row_off = rng.randint(0, h - BASE_PATTERN_SIZE)
    col_off = rng.randint(0, w - BASE_PATTERN_SIZE)

    count = rng.randint(4, 5)
    kept_corners = rng.sample(CORNER_KEYS, count - 1)
    selected_names = ["center", *kept_corners]
    selected = {
        name: (BASE_CENTERS[name][0] + row_off, BASE_CENTERS[name][1] + col_off)
        for name in selected_names
    }

    ray_candidates = _candidate_rays(selected)
    if not ray_candidates:
        return None
    direction, line, status_map = rng.choice(ray_candidates)

    center_hit = status_map["center"]
    opposite_corners = [name for name in kept_corners if status_map[name] != center_hit]
    if not opposite_corners:
        return None
    overlap_corner = rng.choice(opposite_corners)

    colors = rng.sample(list(SQUARE_COLORS), count)
    side_map = {name: _sample_side(rng) for name in selected_names}

    hit_name = "center" if center_hit else overlap_corner
    miss_name = overlap_corner if center_hit else "center"
    side_map[hit_name] = 1
    side_map[miss_name] = 5 if rng.random() < 0.7 else 7

    hit_names = [name for name, hit in status_map.items() if hit]
    miss_names = [name for name, hit in status_map.items() if not hit]
    if len(miss_names) >= 2:
        extra_miss_choices = [name for name in miss_names if name != miss_name]
        if not extra_miss_choices:
            return None
        side_map[rng.choice(extra_miss_choices)] = 1
    else:
        extra_hit_choices = [name for name in hit_names if name != hit_name]
        if not extra_hit_choices:
            return None
        side_map[rng.choice(extra_hit_choices)] = 5

    objects = []
    for name, color in zip(selected_names, colors):
        objects.append(
            {
                "name": name,
                "center": selected[name],
                "color": color,
                "side": side_map[name],
                "hit": status_map[name],
            }
        )

    if not _input_boxes_disjoint(objects):
        return None

    grid = _build_input_grid((h, w), objects, direction, line)
    expected_objects = []
    for obj in objects:
        new_side = SHRINK_SIDE[obj["side"]] if obj["hit"] else GROW_SIDE[obj["side"]]
        expected_objects.append({"color": obj["color"], "center": obj["center"], "side": new_side})

    expected = _render_objects((h, w), expected_objects, bg=BG, order="canonical")
    if np.array_equal(expected, _render_objects((h, w), expected_objects, bg=BG, order="reverse")):
        return None
    if _overlap_footprint(expected_objects) == 0:
        return None
    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None

    try:
        _validate_generated_input(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-partition square-resize input")
