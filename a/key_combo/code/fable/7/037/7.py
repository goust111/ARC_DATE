from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _background(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as (n, 2) arrays of (row, column)."""
    h, w = mask.shape
    unseen = mask.copy()
    result: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if not unseen[sr, sc]:
            continue
        unseen[sr, sc] = False
        queue = deque([(sr, sc)])
        cells: list[Coord] = []
        while queue:
            r, c = queue.popleft()
            cells.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and unseen[nr, nc]:
                    unseen[nr, nc] = False
                    queue.append((nr, nc))
        result.append(np.asarray(cells, dtype=int))
    return result


def _bbox(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_rectangle_perimeter(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    if h < 3 or w < 3 or len(coords) != 2 * h + 2 * w - 4:
        return False
    return all(r in (r0, r1) or c in (c0, c1) for r, c in coords)


def _direction(center2: Coord, marker: Coord) -> Coord:
    """Use doubled center coordinates so half-cell centers remain exact."""
    dr2 = 2 * marker[0] - center2[0]
    dc2 = 2 * marker[1] - center2[1]
    assert abs(dr2) != abs(dc2), "marker bearing must have a unique major axis"
    if abs(dr2) > abs(dc2):
        return (1 if dr2 > 0 else -1, 0)
    return (0, 1 if dc2 > 0 else -1)


def _parse(grid: np.ndarray) -> tuple[int, list[dict]]:
    bg = _background(grid)
    towers: list[dict] = []
    for color in (int(x) for x in np.unique(grid) if int(x) != bg):
        rectangles: list[BBox] = []
        markers: list[Coord] = []
        for comp in _components(grid == color):
            if len(comp) == 1:
                markers.append((int(comp[0, 0]), int(comp[0, 1])))
            else:
                assert _is_rectangle_perimeter(comp), "non-marker objects must be hollow rectangles"
                rectangles.append(_bbox(comp))

        assert 2 <= len(rectangles) <= 5, "each marker color must own one tower of depth 2..5"
        assert markers, "each tower must have at least one matching marker"
        rectangles.sort(key=lambda b: (b[2] - b[0] + 1) * (b[3] - b[1] + 1), reverse=True)
        center2 = (rectangles[0][0] + rectangles[0][2], rectangles[0][1] + rectangles[0][3])
        for i, box in enumerate(rectangles):
            assert (box[0] + box[2], box[1] + box[3]) == center2, "tower layers must share a center"
            if i:
                outer, inner = rectangles[i - 1], box
                assert outer[0] < inner[0] and outer[1] < inner[1]
                assert inner[2] < outer[2] and inner[3] < outer[3]
        outer = rectangles[0]
        for r, c in markers:
            assert not (outer[0] <= r <= outer[2] and outer[1] <= c <= outer[3]), "markers lie outside their tower"
            _direction(center2, (r, c))
        towers.append({"color": color, "boxes": rectangles, "center2": center2, "markers": markers})
    assert towers, "expected at least one nested tower"
    return bg, towers


def main(grid: np.ndarray) -> np.ndarray:
    """Extend each marker away from its same-color nested tower by the tower's depth.

    Input structure:
    A dominant background contains two or more concentric hollow rectangular layers per tower. Towers have distinct
    colors, and isolated same-color marker pixels lie outside them. A marker's row and column offsets from its tower
    center have unequal magnitudes, and every required extension fits in clear background.

    Rule:
    Pair each marker with the tower of the same color and count that tower's layers k. From the tower center toward
    the marker, choose the row or column component with larger absolute magnitude. Starting just beyond the marker,
    paint exactly k cells in that cardinal direction, preserving every original cell.

    Key insight:
    Each tower is both ruler and compass: its nesting depth supplies ray length, while its center-to-marker bearing
    supplies ray direction, so neither fact can be read from the marker alone.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"
    bg, towers = _parse(g)
    out = g.copy()
    claimed: set[Coord] = set()
    h, w = g.shape
    for tower in towers:
        color = int(tower["color"])
        depth = len(tower["boxes"])
        center2 = tower["center2"]
        for marker in tower["markers"]:
            dr, dc = _direction(center2, marker)
            for step in range(1, depth + 1):
                r, c = marker[0] + step * dr, marker[1] + step * dc
                assert 0 <= r < h and 0 <= c < w, "ray must fit inside the grid"
                assert int(g[r, c]) == bg, "ray must pass only through background"
                assert (r, c) not in claimed, "rays must not overlap"
                claimed.add((r, c))
                out[r, c] = color
    return out


# Generator -----------------------------------------------------------------

_PALETTE = tuple(range(10))
_RETRIES = 400
_DIRS: tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _expanded_overlap(a: BBox, b: BBox, gap: int = 1) -> bool:
    return not (a[2] + gap < b[0] or b[2] + gap < a[0] or a[3] + gap < b[1] or b[3] + gap < a[1])


def _draw_tower(grid: np.ndarray, box: BBox, depth: int, color: int) -> None:
    r0, c0, r1, c1 = box
    for layer in range(depth):
        a, b, x, y = r0 + 2 * layer, c0 + 2 * layer, r1 - 2 * layer, c1 - 2 * layer
        grid[a, b : y + 1] = color
        grid[x, b : y + 1] = color
        grid[a : x + 1, b] = color
        grid[a : x + 1, y] = color


def _place_boxes(h: int, w: int, depths: list[int], rng: random.Random) -> list[BBox] | None:
    boxes: list[BBox] = []
    for depth in sorted(depths, reverse=True):
        minimum = 4 * depth - 1
        choices: list[BBox] = []
        for _ in range(1200):
            bh = minimum + 2 * rng.randint(0, 1)
            bw = minimum + 2 * rng.randint(0, 1)
            if bh > h or bw > w:
                continue
            r0, c0 = rng.randint(0, h - bh), rng.randint(0, w - bw)
            candidate = (r0, c0, r0 + bh - 1, c0 + bw - 1)
            if not any(_expanded_overlap(candidate, old, 1) for old in boxes):
                choices.append(candidate)
                if len(choices) >= 30:
                    break
        if not choices:
            return None
        boxes.append(rng.choice(choices))
    # Return in the caller's depth order.
    paired = sorted(zip(sorted(depths, reverse=True), boxes), key=lambda p: depths.index(p[0]))
    return [box for _, box in paired]


def _candidate_markers(
    grid: np.ndarray, center2: Coord, box: BBox, depth: int, wanted: Coord, reserved: set[Coord]
) -> list[Coord]:
    bg = _background(grid)
    h, w = grid.shape
    candidates: list[Coord] = []
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != bg or _direction_or_none(center2, (r, c)) != wanted:
                continue
            if box[0] <= r <= box[2] and box[1] <= c <= box[3]:
                continue
            local = [(rr, cc) for rr in range(max(0, r - 1), min(h, r + 2)) for cc in range(max(0, c - 1), min(w, c + 2))]
            if any(int(grid[rr, cc]) != bg or (rr, cc) in reserved for rr, cc in local):
                continue
            ray = {(r + s * wanted[0], c + s * wanted[1]) for s in range(1, depth + 1)}
            if all(0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == bg and (rr, cc) not in reserved for rr, cc in ray):
                candidates.append((r, c))
    return candidates


def _direction_or_none(center2: Coord, marker: Coord) -> Coord | None:
    dr2, dc2 = 2 * marker[0] - center2[0], 2 * marker[1] - center2[1]
    if abs(dr2) == abs(dc2):
        return None
    if abs(dr2) > abs(dc2):
        return (1 if dr2 > 0 else -1, 0)
    return (0, 1 if dc2 > 0 else -1)


def _semantic_check(inp: np.ndarray, expected: np.ndarray, plans: list[dict]) -> bool:
    try:
        actual = main(inp)
        _, parsed = _parse(inp)
    except (AssertionError, ValueError):
        return False
    if not np.array_equal(actual, expected) or np.any(actual[inp != _background(inp)] != inp[inp != _background(inp)]):
        return False
    depths = [len(t["boxes"]) for t in parsed]
    if len(set(depths)) < 2:
        return False
    directions = {_direction(t["center2"], marker) for t in parsed for marker in t["markers"]}
    if directions != set(_DIRS):
        return False
    # Exclude shortcuts based on a fixed ray length or the grid center as compass.
    if all(depth == 1 for depth in depths):
        return False
    grid_center2 = (inp.shape[0] - 1, inp.shape[1] - 1)
    if all(_direction_or_none(grid_center2, marker) == _direction(t["center2"], marker)
           for t in parsed for marker in t["markers"]):
        return False
    # At least one marker would point differently if paired with a wrong-color tower.
    centers = [t["center2"] for t in parsed]
    color_pair_matters = False
    for i, tower in enumerate(parsed):
        for marker in tower["markers"]:
            correct = _direction(tower["center2"], marker)
            for j, other_center in enumerate(centers):
                wrong = _direction_or_none(other_center, marker)
                if i != j and wrong is not None and wrong != correct:
                    color_pair_matters = True
    return color_pair_matters


def _attempt(rng: random.Random) -> np.ndarray | None:
    h, w = rng.randint(25, 30), rng.randint(25, 30)
    count = rng.choice((2, 2, 3))
    depths = rng.sample((2, 3, 4, 5), count) if count == 2 else rng.sample((2, 3, 4), count)
    boxes = _place_boxes(h, w, depths, rng)
    if boxes is None:
        return None
    bg = rng.choice(_PALETTE)
    colors = rng.sample([x for x in _PALETTE if x != bg], count)
    inp = np.full((h, w), bg, dtype=int)
    for box, depth, color in zip(boxes, depths, colors):
        _draw_tower(inp, box, depth, color)

    assignments = [(i, direction) for i in range(count) for direction in _DIRS]
    rng.shuffle(assignments)
    # Four rays cover all cardinal branches; distribute them over at least two towers.
    chosen = assignments[:4]
    if {d for _, d in chosen} != set(_DIRS) or len({i for i, _ in chosen}) < 2:
        chosen = [(j % count, direction) for j, direction in enumerate(_DIRS)]
        rng.shuffle(chosen)

    expected = inp.copy()
    reserved: set[Coord] = set()
    plans: list[dict] = []
    for i, wanted in chosen:
        box, depth, color = boxes[i], depths[i], colors[i]
        center2 = (box[0] + box[2], box[1] + box[3])
        candidates = _candidate_markers(inp, center2, box, depth, wanted, reserved)
        if not candidates:
            return None
        marker = rng.choice(candidates)
        inp[marker] = color
        expected[marker] = color
        reserved.add(marker)
        for step in range(1, depth + 1):
            cell = (marker[0] + step * wanted[0], marker[1] + step * wanted[1])
            expected[cell] = color
            reserved.add(cell)
        plans.append({"tower": i, "marker": marker, "direction": wanted, "depth": depth})
    return inp if _semantic_check(inp, expected, plans) else None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(_RETRIES):
        result = _attempt(rng)
        if result is not None:
            return result
    raise RuntimeError("failed to generate a non-degenerate nested-tower input")
