from __future__ import annotations

import random
from typing import Optional

import numpy as np


BG = 0
GRAY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
DIRECTIONS = ("up", "down", "left", "right")

GEN_GRID_MIN = 24
GEN_GRID_MAX = 30
GEN_SECTOR_GAP = 5
GEN_OUTER_MARGIN = 1
GEN_RECT_MIN = 5
GEN_RECT_MAX = 7
GEN_RETRIES = 600

Coord = tuple[int, int]
Rect = tuple[int, int, int, int]


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)


def _king_distance(a: Coord, b: Coord) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _bbox(coords: list[Coord]) -> Rect:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _rect_king_distance(rect: Rect, cell: Coord) -> int:
    r0, c0, r1, c1 = rect
    r, c = cell
    dr = 0 if r0 <= r <= r1 else min(abs(r - r0), abs(r - r1))
    dc = 0 if c0 <= c <= c1 else min(abs(c - c0), abs(c - c1))
    return max(dr, dc)


def _strict_side(rect: Rect, cell: Coord) -> Optional[str]:
    r0, c0, r1, c1 = rect
    r, c = cell
    matches: list[str] = []
    if r < r0 and c0 <= c <= c1:
        matches.append("up")
    if r > r1 and c0 <= c <= c1:
        matches.append("down")
    if c < c0 and r0 <= r <= r1:
        matches.append("left")
    if c > c1 and r0 <= r <= r1:
        matches.append("right")
    if len(matches) > 1:
        raise AssertionError("a pixel cannot strictly face two sides")
    return matches[0] if matches else None


def _door_cell(rect: Rect, far_pixel: Coord) -> Coord:
    r0, c0, r1, c1 = rect
    r, c = far_pixel
    side = _strict_side(rect, far_pixel)
    if side == "up":
        return r0, c
    if side == "down":
        return r1, c
    if side == "left":
        return r, c0
    if side == "right":
        return r, c1
    raise AssertionError("far pixel must strictly face one rectangle side")


def _ring_points(rect: Rect, dist: int) -> list[Coord]:
    r0, c0, r1, c1 = rect
    pts: list[Coord] = []
    top = r0 - dist
    bottom = r1 + dist
    left = c0 - dist
    right = c1 + dist
    for c in range(left, right + 1):
        pts.append((top, c))
        pts.append((bottom, c))
    for r in range(top + 1, bottom):
        pts.append((r, left))
        pts.append((r, right))
    return pts


def _side_points(rect: Rect, direction: str, dist: int) -> list[Coord]:
    r0, c0, r1, c1 = rect
    if direction == "up":
        return [(r0 - dist, c) for c in range(c0, c1 + 1)]
    if direction == "down":
        return [(r1 + dist, c) for c in range(c0, c1 + 1)]
    if direction == "left":
        return [(r, c0 - dist) for r in range(r0, r1 + 1)]
    if direction == "right":
        return [(r, c1 + dist) for r in range(r0, r1 + 1)]
    raise ValueError(direction)


def _in_bounds(shape: tuple[int, int], cell: Coord) -> bool:
    return 0 <= cell[0] < shape[0] and 0 <= cell[1] < shape[1]


def _extract_rectangles(grid: np.ndarray) -> list[Rect]:
    comps = _components(grid == GRAY)
    if not comps:
        raise AssertionError("expected at least one gray rectangle")
    rects: list[Rect] = []
    for coords in comps:
        rect = _bbox(coords)
        r0, c0, r1, c1 = rect
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < GEN_RECT_MIN or w < GEN_RECT_MIN:
            raise AssertionError("gray rectangles must be at least 5x5")
        if len(coords) != h * w:
            raise AssertionError("gray component must be a solid rectangle")
        if not np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == GRAY):
            raise AssertionError("gray rectangle must be filled")
        rects.append(rect)
    rects.sort()
    return rects


def _extract_pixels(grid: np.ndarray) -> list[tuple[Coord, int]]:
    coords = np.argwhere((grid != BG) & (grid != GRAY))
    return [((int(r), int(c)), int(grid[r, c])) for r, c in coords]


def _nearest_rect_index(cell: Coord, rects: list[Rect]) -> Optional[int]:
    dists = [_rect_king_distance(rect, cell) for rect in rects]
    best = min(dists)
    winners = [i for i, dist in enumerate(dists) if dist == best]
    if len(winners) != 1:
        return None
    return winners[0]


def _draw_transformed_rect(
    out: np.ndarray,
    rect: Rect,
    border_color: int,
    interior_color: int,
    door: Coord,
) -> None:
    r0, c0, r1, c1 = rect
    out[r0 : r1 + 1, c0 : c1 + 1] = interior_color
    out[r0, c0 : c1 + 1] = border_color
    out[r1, c0 : c1 + 1] = border_color
    out[r0 : r1 + 1, c0] = border_color
    out[r0 : r1 + 1, c1] = border_color
    out[door] = BG


def _parse_roles(grid: np.ndarray) -> list[dict[str, object]]:
    rects = _extract_rectangles(grid)
    pixels = _extract_pixels(grid)
    if len(pixels) != 2 * len(rects):
        raise AssertionError("each rectangle must receive exactly two singleton pixels")

    assignments: list[list[dict[str, object]]] = [[] for _ in rects]
    for cell, color in pixels:
        rect_idx = _nearest_rect_index(cell, rects)
        if rect_idx is None:
            raise AssertionError("singleton pixel must have a unique nearest rectangle")
        assignments[rect_idx].append(
            {
                "coord": cell,
                "color": color,
                "dist": _rect_king_distance(rects[rect_idx], cell),
            }
        )

    roles: list[dict[str, object]] = []
    for rect, group in zip(rects, assignments):
        if len(group) != 2:
            raise AssertionError("each rectangle must get exactly two nearest singleton pixels")
        group.sort(key=lambda item: int(item["dist"]))
        near = group[0]
        far = group[1]
        if int(near["dist"]) >= int(far["dist"]):
            raise AssertionError("distance ranks must be strict")
        if int(near["color"]) == int(far["color"]):
            raise AssertionError("near and far colors must differ so both roles stay visible")
        far_side = _strict_side(rect, far["coord"])  # type: ignore[arg-type]
        if far_side is None:
            raise AssertionError("far singleton pixel must strictly face one side")
        roles.append(
            {
                "rect": rect,
                "near": near["coord"],
                "near_color": int(near["color"]),
                "far": far["coord"],
                "far_color": int(far["color"]),
                "far_side": far_side,
            }
        )
    return roles


def _solve(grid: np.ndarray) -> np.ndarray:
    roles = _parse_roles(grid)
    out = np.full(grid.shape, BG, dtype=int)
    for role in roles:
        rect = role["rect"]  # type: ignore[assignment]
        far = role["far"]  # type: ignore[assignment]
        _draw_transformed_rect(
            out,
            rect,  # type: ignore[arg-type]
            int(role["near_color"]),
            int(role["far_color"]),
            _door_cell(rect, far),  # type: ignore[arg-type]
        )
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assign each singleton pixel to its nearest gray rectangle, then recolor the rectangle by distance rank and open a side door.

    Input structure:
    - Black background with several solid gray rectangles, each at least 5 by 5.
    - Colored singleton pixels lie around the rectangles.
    - Every singleton pixel belongs to exactly one nearest rectangle under king distance, and each rectangle receives exactly two such pixels.

    Rule:
    - For each rectangle, sort its two assigned singleton pixels by king distance to the rectangle.
    - Recolor the rectangle border with the nearer pixel's color and recolor the strict interior with the farther pixel's color.
    - The farther pixel also indicates a door: erase the border cell directly facing that pixel on the corresponding side.
    - Remove every singleton pixel from the final grid.

    Key insight:
    - The two pixels around each rectangle jointly encode ownership, role, and doorway location: nearest-rectangle assignment picks the owner, distance rank splits border versus interior colors, and the farther pixel's aligned direction selects the missing border cell.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("grid must be 2D")
    return _solve(g)


def _sector_boxes(h: int, w: int) -> list[Rect]:
    usable_h = h - 2 * GEN_OUTER_MARGIN - GEN_SECTOR_GAP
    usable_w = w - 2 * GEN_OUTER_MARGIN - GEN_SECTOR_GAP
    top_h = usable_h // 2
    bottom_h = usable_h - top_h
    left_w = usable_w // 2
    right_w = usable_w - left_w

    top_r0 = GEN_OUTER_MARGIN
    top_r1 = top_r0 + top_h - 1
    bottom_r0 = top_r1 + GEN_SECTOR_GAP + 1
    bottom_r1 = h - GEN_OUTER_MARGIN - 1

    left_c0 = GEN_OUTER_MARGIN
    left_c1 = left_c0 + left_w - 1
    right_c0 = left_c1 + GEN_SECTOR_GAP + 1
    right_c1 = w - GEN_OUTER_MARGIN - 1

    return [
        (top_r0, left_c0, top_r1, left_c1),
        (top_r0, right_c0, top_r1, right_c1),
        (bottom_r0, left_c0, bottom_r1, left_c1),
        (bottom_r0, right_c0, bottom_r1, right_c1),
    ]


def _random_rect(box: Rect) -> Optional[Rect]:
    r0, c0, r1, c1 = box
    max_h = min(GEN_RECT_MAX, r1 - r0 + 1)
    max_w = min(GEN_RECT_MAX, c1 - c0 + 1)
    if max_h < GEN_RECT_MIN or max_w < GEN_RECT_MIN:
        return None
    h = random.randint(GEN_RECT_MIN, max_h)
    w = random.randint(GEN_RECT_MIN, max_w)
    rr0 = random.randint(r0, r1 - h + 1)
    cc0 = random.randint(c0, c1 - w + 1)
    return rr0, cc0, rr0 + h - 1, cc0 + w - 1


def _feasible_directions(rect: Rect, shape: tuple[int, int]) -> list[str]:
    dirs: list[str] = []
    for direction in DIRECTIONS:
        for dist in (2, 3, 4):
            if any(_in_bounds(shape, cell) for cell in _side_points(rect, direction, dist)):
                dirs.append(direction)
                break
    return dirs


def _assign_distinct_directions(rects: list[Rect], shape: tuple[int, int]) -> Optional[list[str]]:
    feasible = [_feasible_directions(rect, shape) for rect in rects]
    if any(not dirs for dirs in feasible):
        return None

    order = sorted(range(len(rects)), key=lambda idx: len(feasible[idx]))
    out: list[Optional[str]] = [None] * len(rects)

    def rec(pos: int, used: set[str]) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        choices = list(feasible[idx])
        random.shuffle(choices)
        for direction in choices:
            if direction in used:
                continue
            out[idx] = direction
            if rec(pos + 1, used | {direction}):
                return True
            out[idx] = None
        return False

    if not rec(0, set()):
        return None
    return [direction for direction in out if direction is not None]


def _near_mode_plan(n_rects: int) -> list[str]:
    modes = ["corner", "side_other"]
    while len(modes) < n_rects:
        modes.append(random.choice(["corner", "side_other"]))
    random.shuffle(modes)
    return modes[:n_rects]


def _pixel_clear(cell: Coord, rect_idx: int, rects: list[Rect], used: set[Coord]) -> bool:
    if cell in used:
        return False
    if any(_king_distance(cell, other) <= 1 for other in used):
        return False
    winner = _nearest_rect_index(cell, rects)
    return winner == rect_idx


def _candidate_far_points(rect: Rect, direction: str, shape: tuple[int, int]) -> list[tuple[Coord, int]]:
    out: list[tuple[Coord, int]] = []
    dists = [2, 3, 4]
    random.shuffle(dists)
    for dist in dists:
        points = [cell for cell in _side_points(rect, direction, dist) if _in_bounds(shape, cell)]
        random.shuffle(points)
        out.extend((cell, dist) for cell in points)
    return out


def _candidate_near_points(
    rect: Rect,
    far_direction: str,
    far_dist: int,
    mode: str,
    shape: tuple[int, int],
) -> list[tuple[Coord, int]]:
    out: list[tuple[Coord, int]] = []
    dists = [1, 2]
    random.shuffle(dists)
    for dist in dists:
        if dist >= far_dist:
            continue
        points = [cell for cell in _ring_points(rect, dist) if _in_bounds(shape, cell)]
        random.shuffle(points)
        for cell in points:
            side = _strict_side(rect, cell)
            if mode == "corner":
                if side is not None:
                    continue
            elif mode == "side_other":
                if side is None or side == far_direction:
                    continue
            else:
                raise ValueError(mode)
            out.append((cell, dist))
    return out


def _assign_pixels_to_rects(
    rects: list[Rect],
    directions: list[str],
    near_modes: list[str],
    colors: list[int],
    shape: tuple[int, int],
) -> Optional[list[dict[str, object]]]:
    counts = [len(_candidate_far_points(rect, direction, shape)) for rect, direction in zip(rects, directions)]
    order = sorted(range(len(rects)), key=lambda idx: counts[idx])
    specs: list[Optional[dict[str, object]]] = [None] * len(rects)
    used: set[Coord] = set()

    def rec(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        rect = rects[idx]
        direction = directions[idx]
        near_mode = near_modes[idx]
        near_color = colors[2 * idx]
        far_color = colors[2 * idx + 1]

        for far_cell, far_dist in _candidate_far_points(rect, direction, shape):
            if not _pixel_clear(far_cell, idx, rects, used):
                continue
            used.add(far_cell)
            for near_cell, near_dist in _candidate_near_points(rect, direction, far_dist, near_mode, shape):
                if near_dist >= far_dist:
                    continue
                if not _pixel_clear(near_cell, idx, rects, used):
                    continue
                used.add(near_cell)
                specs[idx] = {
                    "rect": rect,
                    "near": near_cell,
                    "near_color": near_color,
                    "near_dist": near_dist,
                    "near_mode": near_mode,
                    "far": far_cell,
                    "far_color": far_color,
                    "far_dist": far_dist,
                    "far_dir": direction,
                }
                if rec(pos + 1):
                    return True
                specs[idx] = None
                used.remove(near_cell)
            used.remove(far_cell)
        return False

    if not rec(0):
        return None
    return [spec for spec in specs if spec is not None]


def _render_input(shape: tuple[int, int], specs: list[dict[str, object]]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for spec in specs:
        r0, c0, r1, c1 = spec["rect"]  # type: ignore[misc]
        grid[r0 : r1 + 1, c0 : c1 + 1] = GRAY
        nr, nc = spec["near"]  # type: ignore[misc]
        fr, fc = spec["far"]  # type: ignore[misc]
        grid[nr, nc] = int(spec["near_color"])
        grid[fr, fc] = int(spec["far_color"])
    return grid


def _render_expected(shape: tuple[int, int], specs: list[dict[str, object]]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for spec in specs:
        rect = spec["rect"]  # type: ignore[assignment]
        far = spec["far"]  # type: ignore[assignment]
        _draw_transformed_rect(
            out,
            rect,  # type: ignore[arg-type]
            int(spec["near_color"]),
            int(spec["far_color"]),
            _door_cell(rect, far),  # type: ignore[arg-type]
        )
    return out


def _validate_specs(grid: np.ndarray, specs: list[dict[str, object]]) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    rects = [spec["rect"] for spec in specs]
    pixels: list[Coord] = []
    pixel_colors: dict[Coord, int] = {}

    for idx, spec in enumerate(specs):
        rect = spec["rect"]  # type: ignore[assignment]
        near = spec["near"]  # type: ignore[assignment]
        far = spec["far"]  # type: ignore[assignment]
        near_color = int(spec["near_color"])
        far_color = int(spec["far_color"])
        near_mode = str(spec["near_mode"])
        far_dir = str(spec["far_dir"])

        if near_color in (BG, GRAY) or far_color in (BG, GRAY):
            return False
        if near_color == far_color:
            return False
        if _rect_king_distance(rect, near) != int(spec["near_dist"]):
            return False
        if _rect_king_distance(rect, far) != int(spec["far_dist"]):
            return False
        if int(spec["near_dist"]) >= int(spec["far_dist"]):
            return False
        if _strict_side(rect, far) != far_dir:
            return False

        near_side = _strict_side(rect, near)
        if near_mode == "corner":
            if near_side is not None:
                return False
        elif near_mode == "side_other":
            if near_side is None or near_side == far_dir:
                return False
        else:
            return False

        if _nearest_rect_index(near, rects) != idx or _nearest_rect_index(far, rects) != idx:
            return False

        pixels.extend([near, far])
        pixel_colors[near] = near_color
        pixel_colors[far] = far_color

        door = _door_cell(rect, far)
        if near_color == BG or door == near or door == far:
            return False
        r0, c0, r1, c1 = rect
        if r1 - r0 + 1 < 5 or c1 - c0 + 1 < 5:
            return False

    for i in range(len(pixels)):
        for j in range(i + 1, len(pixels)):
            if _king_distance(pixels[i], pixels[j]) <= 1:
                return False

    parsed_rects = _extract_rectangles(grid)
    if parsed_rects != sorted(rects):
        return False

    parsed_pixels = {cell: color for cell, color in _extract_pixels(grid)}
    if parsed_pixels != pixel_colors:
        return False

    if len({str(spec["far_dir"]) for spec in specs}) != len(specs):
        return False
    if len(specs) >= 2 and {str(spec["near_mode"]) for spec in specs} != {"corner", "side_other"}:
        return False

    expected = _render_expected(grid.shape, specs)
    if np.array_equal(expected, grid):
        return False
    for spec in specs:
        rect = spec["rect"]  # type: ignore[assignment]
        far = spec["far"]  # type: ignore[assignment]
        door = _door_cell(rect, far)
        if expected[door] != BG:
            return False

    return np.array_equal(main(grid), expected)


def _gen_attempt() -> Optional[np.ndarray]:
    h = random.randint(GEN_GRID_MIN, GEN_GRID_MAX)
    w = random.randint(GEN_GRID_MIN, GEN_GRID_MAX)
    n_rects = random.choices([2, 3, 4], weights=[1, 3, 4])[0]

    boxes = _sector_boxes(h, w)
    chosen_boxes = random.sample(boxes, n_rects)
    rects: list[Rect] = []
    for box in chosen_boxes:
        rect = _random_rect(box)
        if rect is None:
            return None
        rects.append(rect)
    rects.sort()

    directions = _assign_distinct_directions(rects, (h, w))
    if directions is None:
        return None
    near_modes = _near_mode_plan(n_rects)
    colors = random.sample(COLOR_POOL, 2 * n_rects)
    random.shuffle(colors)

    specs = _assign_pixels_to_rects(rects, directions, near_modes, colors, (h, w))
    if specs is None:
        return None

    grid = _render_input((h, w), specs)
    if not _validate_specs(grid, specs):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
