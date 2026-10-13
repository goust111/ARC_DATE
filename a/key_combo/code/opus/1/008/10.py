from __future__ import annotations

import numpy as np


BG = 0
PATH_COLOR = 5
RECTANGLE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
SIDES = ("top", "bottom", "left", "right")
MAX_GENERATION_ATTEMPTS = 400
RNG = np.random.default_rng()


def _neighbors4(r: int, c: int, height: int, width: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < height and 0 <= cc < width:
            out.append((rr, cc))
    return out


def _find_monochrome_components(grid: np.ndarray, ignore: int = BG) -> list[dict]:
    height, width = grid.shape
    visited = np.zeros((height, width), dtype=bool)
    components: list[dict] = []

    for r in range(height):
        for c in range(width):
            color = int(grid[r, c])
            if color == ignore or visited[r, c]:
                continue

            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            r0 = r1 = r
            c0 = c1 = c

            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                r0 = min(r0, rr)
                r1 = max(r1, rr)
                c0 = min(c0, cc)
                c1 = max(c1, cc)

                for nr, nc in _neighbors4(rr, cc, height, width):
                    if visited[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    stack.append((nr, nc))

            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "area": len(coords),
                    "bbox": (r0, c0, r1, c1),
                }
            )

    return components


def _bbox_dims(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return r1 - r0 + 1, c1 - c0 + 1


def _touches_border(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    height, width = shape
    return r0 == 0 or c0 == 0 or r1 == height - 1 or c1 == width - 1


def _component_is_filled_rectangle(component: dict) -> bool:
    h, w = _bbox_dims(component["bbox"])
    return int(component["area"]) == h * w


def _find_valid_border_rectangles(grid: np.ndarray) -> list[dict]:
    rectangles: list[dict] = []
    for component in _find_monochrome_components(grid, ignore=BG):
        if not _touches_border(component["bbox"], grid.shape):
            continue
        if int(component["area"]) < 4:
            continue
        if not _component_is_filled_rectangle(component):
            continue
        rectangles.append(component)
    return rectangles


def _neighbor_map(coords: list[tuple[int, int]]) -> dict[tuple[int, int], list[tuple[int, int]]]:
    points = set(coords)
    if not points:
        raise AssertionError("expected a non-empty point set")
    height = max(r for r, _ in points) + 1
    width = max(c for _, c in points) + 1
    mapping: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in points:
        mapping[(r, c)] = [p for p in _neighbors4(r, c, height, width) if p in points]
    return mapping


def _ordered_path(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    neighbor_map = _neighbor_map(coords)
    endpoints = sorted(point for point, nbrs in neighbor_map.items() if len(nbrs) == 1)
    assert len(endpoints) == 2, "path must have exactly two endpoints"
    assert all(len(nbrs) <= 2 for nbrs in neighbor_map.values()), "path must be branch-free"

    start = endpoints[0]
    path = [start]
    prev: tuple[int, int] | None = None
    cur = start

    while True:
        next_points = [p for p in neighbor_map[cur] if p != prev]
        if not next_points:
            break
        assert len(next_points) == 1, "path traversal must be unique"
        nxt = next_points[0]
        path.append(nxt)
        prev, cur = cur, nxt

    assert len(path) == len(neighbor_map), "path traversal must visit every path pixel exactly once"
    return path


def _find_gray_path(grid: np.ndarray) -> list[tuple[int, int]]:
    candidates: list[list[tuple[int, int]]] = []
    for component in _find_monochrome_components(grid, ignore=BG):
        if int(component["color"]) != PATH_COLOR:
            continue
        try:
            candidates.append(_ordered_path(component["coords"]))
        except AssertionError:
            continue

    assert len(candidates) == 1, "expected exactly one gray simple path"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the central path by the area order of border rectangles, then erase those rectangles.

    Input structure:
    - The background is black.
    - Several monochrome filled rectangles touch the outer border; each valid rectangle is 4-connected, fills its bbox, and has area at least 4.
    - A separate gray one-pixel-wide 4-connected path sits away from the border objects, and tiny clutter components may also appear.

    Rule:
    - Keep only the border-touching monochrome components that exactly fill their bounding boxes and have area at least 4.
    - Sort those rectangles by area from smallest to largest, and use their colors as a repeating sequence.
    - Traverse the gray path from the endpoint with smaller row index, breaking ties by smaller column index, repainting successive path cells by the repeating sorted color sequence.
    - Erase the valid border rectangles to black and leave every other non-rectangle clutter pixel unchanged.

    Key insight:
    - The border rectangles and the path never interact geometrically; rectangle areas determine an abstract color order, and the path topology tells where that order is written.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30

    rectangles = _find_valid_border_rectangles(g)
    assert rectangles, "expected at least one valid border rectangle"

    areas = [int(rect["area"]) for rect in rectangles]
    assert len(set(areas)) == len(areas), "rectangle areas must be unique"
    ordered_colors = [int(rect["color"]) for rect in sorted(rectangles, key=lambda rect: int(rect["area"]))]

    path = _find_gray_path(g)

    out = g.copy()
    for rect in rectangles:
        for r, c in rect["coords"]:
            out[r, c] = BG

    for idx, (r, c) in enumerate(path):
        out[r, c] = ordered_colors[idx % len(ordered_colors)]

    return out


def _box_clear(box: tuple[int, int, int, int], boxes: list[tuple[int, int, int, int]], gap: int = 1) -> bool:
    r0, c0, r1, c1 = box
    for or0, oc0, or1, oc1 in boxes:
        if not (r1 + gap < or0 or or1 + gap < r0 or c1 + gap < oc0 or oc1 + gap < c0):
            return False
    return True


def _box_coords(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _sample_border_box(height: int, width: int, side: str) -> tuple[int, int, int, int]:
    if side in ("top", "bottom"):
        h = int(RNG.integers(2, min(5, height - 1)))
        w = int(RNG.integers(2, min(7, width - 1)))
        c0 = int(RNG.integers(0, width - w + 1))
        r0 = 0 if side == "top" else height - h
        return (r0, c0, r0 + h - 1, c0 + w - 1)

    w = int(RNG.integers(2, min(5, width - 1)))
    h = int(RNG.integers(2, min(7, height - 1)))
    r0 = int(RNG.integers(0, height - h + 1))
    c0 = 0 if side == "left" else width - w
    return (r0, c0, r0 + h - 1, c0 + w - 1)


def _place_rectangles(height: int, width: int, count: int) -> list[dict] | None:
    sides = list(RNG.permutation(SIDES))
    if count > len(sides):
        while len(sides) < count:
            sides.append(SIDES[int(RNG.integers(0, len(SIDES)))])
    else:
        sides = sides[:count]
    RNG.shuffle(sides)

    boxes: list[tuple[int, int, int, int]] = []
    areas: set[int] = set()
    for side in sides:
        box = None
        for _ in range(120):
            candidate = _sample_border_box(height, width, side)
            area = (_bbox_dims(candidate)[0]) * (_bbox_dims(candidate)[1])
            if area in areas:
                continue
            if not _box_clear(candidate, boxes, gap=1):
                continue
            box = candidate
            areas.add(area)
            boxes.append(box)
            break
        if box is None:
            return None

    colors = list(RNG.choice(np.array(RECTANGLE_COLORS, dtype=int), size=count, replace=False))
    RNG.shuffle(colors)
    rectangles: list[dict] = []
    for color, box in zip(colors, boxes):
        coords = _box_coords(box)
        rectangles.append({"color": int(color), "bbox": box, "coords": coords, "area": len(coords)})
    return rectangles


def _moat_mask(coords: list[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=bool)
    for r, c in coords:
        for rr in range(max(0, r - 1), min(height, r + 2)):
            for cc in range(max(0, c - 1), min(width, c + 2)):
                mask[rr, cc] = True
    return mask


def _central_allowed_mask(grid: np.ndarray, rectangles: list[dict]) -> np.ndarray:
    height, width = grid.shape
    occupied = np.zeros((height, width), dtype=bool)
    all_rect_coords: list[tuple[int, int]] = []
    for rect in rectangles:
        for r, c in rect["coords"]:
            occupied[r, c] = True
            all_rect_coords.append((r, c))

    reserved = _moat_mask(all_rect_coords, grid.shape) if all_rect_coords else occupied
    allowed = (~reserved) & (grid == BG)

    margin = 2
    central = np.zeros_like(allowed)
    if height > 2 * margin and width > 2 * margin:
        central[margin : height - margin, margin : width - margin] = True
    else:
        central[:, :] = True
    return allowed & central


def _straight_path_from_allowed(allowed: np.ndarray, horizontal: bool) -> list[tuple[int, int]] | None:
    height, width = allowed.shape
    segments: list[tuple[int, int, int]] = []

    if horizontal:
        for r in range(height):
            c = 0
            while c < width:
                if not allowed[r, c]:
                    c += 1
                    continue
                c0 = c
                while c < width and allowed[r, c]:
                    c += 1
                if c - c0 >= 5:
                    segments.append((r, c0, c - c0))
    else:
        for c in range(width):
            r = 0
            while r < height:
                if not allowed[r, c]:
                    r += 1
                    continue
                r0 = r
                while r < height and allowed[r, c]:
                    r += 1
                if r - r0 >= 5:
                    segments.append((c, r0, r - r0))

    if not segments:
        return None

    anchor, start, span = segments[int(RNG.integers(0, len(segments)))]
    length = int(RNG.integers(5, min(15, span) + 1))
    offset = int(RNG.integers(0, span - length + 1))
    if horizontal:
        row = anchor
        col0 = start + offset
        return [(row, col0 + i) for i in range(length)]

    col = anchor
    row0 = start + offset
    return [(row0 + i, col) for i in range(length)]


def _candidate_extensions(
    current: tuple[int, int],
    used: set[tuple[int, int]],
    allowed: np.ndarray,
) -> list[tuple[int, int]]:
    height, width = allowed.shape
    candidates: list[tuple[int, int]] = []
    cr, cc = current
    for nr, nc in _neighbors4(cr, cc, height, width):
        point = (nr, nc)
        if not allowed[nr, nc] or point in used:
            continue
        touching = [nbr for nbr in _neighbors4(nr, nc, height, width) if nbr in used]
        if touching != [current] and set(touching) != {current}:
            continue
        candidates.append(point)
    return candidates


def _turn_count(path: list[tuple[int, int]]) -> int:
    if len(path) < 3:
        return 0
    turns = 0
    for i in range(2, len(path)):
        dr1 = path[i - 1][0] - path[i - 2][0]
        dc1 = path[i - 1][1] - path[i - 2][1]
        dr2 = path[i][0] - path[i - 1][0]
        dc2 = path[i][1] - path[i - 1][1]
        if (dr1, dc1) != (dr2, dc2):
            turns += 1
    return turns


def _dfs_path(
    path: list[tuple[int, int]],
    used: set[tuple[int, int]],
    allowed: np.ndarray,
    target_length: int,
) -> bool:
    if len(path) == target_length:
        return True

    current = path[-1]
    candidates = _candidate_extensions(current, used, allowed)
    if not candidates:
        return False

    order = RNG.permutation(len(candidates))
    for idx in order:
        nxt = candidates[int(idx)]
        path.append(nxt)
        used.add(nxt)
        if _dfs_path(path, used, allowed, target_length):
            return True
        used.remove(nxt)
        path.pop()
    return False


def _turning_path_from_allowed(allowed: np.ndarray) -> list[tuple[int, int]] | None:
    starts = [tuple(map(int, coord)) for coord in np.argwhere(allowed)]
    if len(starts) < 5:
        return None

    for _ in range(30):
        length = int(RNG.integers(5, min(15, len(starts)) + 1))
        order = RNG.permutation(len(starts))
        for idx in order[: min(len(starts), 40)]:
            start = starts[int(idx)]
            path = [start]
            used = {start}
            if _dfs_path(path, used, allowed, length) and _turn_count(path) >= 1:
                return path
    return None


def _place_path(grid: np.ndarray, rectangles: list[dict]) -> list[tuple[int, int]] | None:
    allowed = _central_allowed_mask(grid, rectangles)
    if int(np.count_nonzero(allowed)) < 5:
        return None

    roll = float(RNG.random())
    if roll < 0.35:
        path = _straight_path_from_allowed(allowed, horizontal=True)
        if path is not None:
            return path
    elif roll < 0.50:
        path = _straight_path_from_allowed(allowed, horizontal=False)
        if path is not None:
            return path

    path = _turning_path_from_allowed(allowed)
    if path is not None:
        return path

    for horizontal in (True, False):
        path = _straight_path_from_allowed(allowed, horizontal=horizontal)
        if path is not None:
            return path
    return None


def _place_distractors(grid: np.ndarray, path: list[tuple[int, int]], rectangles: list[dict]) -> bool:
    height, width = grid.shape
    occupied_coords: list[tuple[int, int]] = []
    for rect in rectangles:
        occupied_coords.extend(rect["coords"])
    occupied_coords.extend(path)
    reserved = _moat_mask(occupied_coords, grid.shape)

    distractor_count = int(RNG.integers(0, 4))
    for _ in range(distractor_count):
        placed = False
        size = 1 if float(RNG.random()) < 0.55 else 2
        want_border = float(RNG.random()) < 0.35

        for _ in range(100):
            color = int(RECTANGLE_COLORS[int(RNG.integers(0, len(RECTANGLE_COLORS)))])
            if size == 1:
                r = int(RNG.integers(0, height))
                c = int(RNG.integers(0, width))
                coords = [(r, c)]
            else:
                horizontal = bool(RNG.integers(0, 2))
                if horizontal:
                    r = int(RNG.integers(0, height))
                    c = int(RNG.integers(0, width - 1))
                    coords = [(r, c), (r, c + 1)]
                else:
                    r = int(RNG.integers(0, height - 1))
                    c = int(RNG.integers(0, width))
                    coords = [(r, c), (r + 1, c)]

            if want_border and not any(rr in (0, height - 1) or cc in (0, width - 1) for rr, cc in coords):
                continue
            if any(reserved[rr, cc] or grid[rr, cc] != BG for rr, cc in coords):
                continue

            for rr, cc in coords:
                grid[rr, cc] = color
            reserved |= _moat_mask(coords, grid.shape)
            placed = True
            break

        if not placed:
            return False
    return True


def _build_expected_output(
    grid: np.ndarray,
    rectangles: list[dict],
    path: list[tuple[int, int]],
) -> np.ndarray:
    out = np.array(grid, copy=True)
    for rect in rectangles:
        for r, c in rect["coords"]:
            out[r, c] = BG

    ordered_colors = [int(rect["color"]) for rect in sorted(rectangles, key=lambda rect: int(rect["area"]))]
    ordered_path = _ordered_path(path)
    for idx, (r, c) in enumerate(ordered_path):
        out[r, c] = ordered_colors[idx % len(ordered_colors)]
    return out


def _rectangle_signature(rectangles: list[dict]) -> set[tuple[int, int, tuple[int, int, int, int]]]:
    return {
        (int(rect["color"]), int(rect["area"]), tuple(map(int, rect["bbox"])))
        for rect in rectangles
    }


def _validate_generated_case(grid: np.ndarray, rectangles: list[dict], path: list[tuple[int, int]]) -> None:
    parsed_rectangles = _find_valid_border_rectangles(grid)
    assert _rectangle_signature(parsed_rectangles) == _rectangle_signature(rectangles)
    assert _find_gray_path(grid) == _ordered_path(path)

    expected = _build_expected_output(grid, rectangles, path)
    actual = main(grid)
    assert np.array_equal(actual, expected), "solver must match the planned semantics"


def _gen_attempt() -> np.ndarray | None:
    rect_count = int(RNG.integers(3, 6))
    height = int(RNG.integers(13 + (rect_count == 5), 19))
    width = int(RNG.integers(14 + (rect_count >= 4), 21))
    grid = np.full((height, width), BG, dtype=int)

    rectangles = _place_rectangles(height, width, rect_count)
    if rectangles is None:
        return None
    for rect in rectangles:
        for r, c in rect["coords"]:
            grid[r, c] = int(rect["color"])

    path = _place_path(grid, rectangles)
    if path is None:
        return None
    for r, c in path:
        grid[r, c] = PATH_COLOR

    if not _place_distractors(grid, path, rectangles):
        return None

    _validate_generated_case(grid, rectangles, path)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
