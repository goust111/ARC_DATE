from __future__ import annotations

import random
import sys
from itertools import combinations
from pathlib import Path

import numpy as np


def _ensure_arc_utils_path() -> None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "new" / "solvers"
        if candidate.is_dir():
            path = str(candidate)
            if path not in sys.path:
                sys.path.append(path)
            return

    fallback = Path("/home/lixinyu/snap/new/solvers")
    if fallback.is_dir():
        path = str(fallback)
        if path not in sys.path:
            sys.path.append(path)
        return

    raise ImportError("could not locate new/solvers/arc_utils.py")


_ensure_arc_utils_path()

from arc_utils import Color, bbox_coords, components_mask  # noqa: E402


BG = int(Color.BLACK)
SPRITE = int(Color.GREY)
DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]
MARK_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

GRID_SIDE_MIN = 15
GRID_SIDE_MAX = 24
MARK_COUNT_CHOICES = (3, 4)
STEP_DISTANCE_RANGE = (2, 6)
MAX_GENERATION_ATTEMPTS = 1200
CLIP_MODES = ("none", "left", "right")
CLIP_WEIGHTS = (0.58, 0.21, 0.21)

BASE_TEMPLATES = {
    "chevron3": [(0, 0), (1, -1), (1, 1)],
    "filled4": [(0, 0), (1, -1), (1, 0), (1, 1)],
    "stem5": [(0, 0), (1, 0), (2, -1), (2, 0), (2, 1)],
}


def _rotate_offset(offset: tuple[int, int], direction: int) -> tuple[int, int]:
    r, c = offset
    if direction == 0:
        return int(r), int(c)
    if direction == 1:
        return int(c), int(-r)
    if direction == 2:
        return int(-r), int(-c)
    return int(-c), int(r)


TEMPLATES = {
    name: {
        direction: tuple(_rotate_offset(offset, direction) for offset in offsets)
        for direction in range(4)
    }
    for name, offsets in BASE_TEMPLATES.items()
}


def _placed_cells(head: tuple[int, int], offsets: tuple[tuple[int, int], ...]) -> set[tuple[int, int]]:
    hr, hc = head
    return {(hr + dr, hc + dc) for dr, dc in offsets}


def _detect_sprite(grid: np.ndarray) -> tuple[str, int, tuple[int, int], set[tuple[int, int]]]:
    grey_components = components_mask(grid == SPRITE, connectivity=8)
    assert len(grey_components) == 1, "expected one grey sprite"

    pts = {tuple(map(int, rc)) for rc in grey_components[0]}
    matches: list[tuple[str, int, tuple[int, int]]] = []
    for name, dirs in TEMPLATES.items():
        for direction, offsets in dirs.items():
            for head in pts:
                if _placed_cells(head, offsets) == pts:
                    matches.append((name, int(direction), (int(head[0]), int(head[1]))))

    assert len(matches) == 1, "grey shape must encode one unique arrow orientation"
    name, direction, head = matches[0]
    return name, direction, head, pts


def _extract_marks(grid: np.ndarray) -> dict[tuple[int, int], int]:
    coords = np.argwhere((grid != BG) & (grid != SPRITE))
    marks = {(int(r), int(c)): int(grid[r, c]) for r, c in coords}
    assert 3 <= len(marks) <= 4, "expected three or four colored marks"
    colors = list(marks.values())
    assert len(colors) == len(set(colors)), "mark colors must be distinct"
    return marks


def _candidates_in_direction(
    pos: tuple[int, int], direction: int, remaining: dict[tuple[int, int], int]
) -> list[tuple[int, tuple[int, int], int]]:
    r, c = pos
    candidates: list[tuple[int, tuple[int, int], int]] = []
    for (rr, cc), color in remaining.items():
        if direction == 0 and cc == c and rr < r:
            candidates.append((r - rr, (rr, cc), int(color)))
        elif direction == 1 and rr == r and cc > c:
            candidates.append((cc - c, (rr, cc), int(color)))
        elif direction == 2 and cc == c and rr > r:
            candidates.append((rr - r, (rr, cc), int(color)))
        elif direction == 3 and rr == r and cc < c:
            candidates.append((c - cc, (rr, cc), int(color)))
    candidates.sort(key=lambda item: (item[0], item[1][0], item[1][1]))
    return candidates


def _trace_consumption(
    head: tuple[int, int], direction: int, marks: dict[tuple[int, int], int]
) -> tuple[tuple[int, int], int, list[int], list[tuple[int, int]]]:
    remaining = dict(marks)
    colors: list[int] = []
    visited_marks: list[tuple[int, int]] = []

    while remaining:
        current = _candidates_in_direction(head, direction, remaining)
        if current:
            options = [(direction, current)]
        else:
            options = [
                (candidate_dir, _candidates_in_direction(head, candidate_dir, remaining))
                for candidate_dir in range(4)
            ]
            options = [(candidate_dir, cand) for candidate_dir, cand in options if cand]
            assert len(options) == 1, "when the current ray is empty, exactly one direction may remain"

        direction, candidates = options[0]
        assert len(candidates) == 1, "the active half-line must contain exactly one reachable mark"

        _dist, target, color = candidates[0]
        colors.append(int(color))
        visited_marks.append(target)
        del remaining[target]
        head = target

    return head, direction, colors, visited_marks


def _render_pyramid(shape: tuple[int, int], head: tuple[int, int], colors: list[int]) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), BG, dtype=int)
    hr, hc = head

    for layer, color in enumerate(colors, start=1):
        rr = hr + layer
        if not (0 <= rr < h):
            continue
        radius = layer - 1
        c0 = max(0, hc - radius)
        c1 = min(w - 1, hc + radius)
        if c0 <= c1:
            out[rr, c0 : c1 + 1] = int(color)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Consume the colored marks in arrow-guided order and rebuild them as a centered pyramid below the final head position.

    Input structure:
    - A black grid contains one grey triangular arrow sprite and three or four colored single-pixel marks.
    - The sprite tip defines one cardinal facing direction, and the marks all have distinct colors.
    - The active mark is always searched on the sprite head's row or column half-line.

    Rule:
    - Detect the sprite's tip and facing direction from its grey arrow shape.
    - Starting at the tip, repeatedly take the unique visible mark on the current half-line; if that ray is empty, turn to the unique cardinal direction that still contains a mark.
    - Each consumed mark contributes one color to the sequence, in consumption order, and the head moves onto that mark.
    - Clear the canvas and draw a centered pyramid below the final head position: layer k uses the kth consumed color, has width 2k-1, and clips only against the left and right borders.

    Key insight:
    - The path order is encoded jointly by the arrow orientation and the constrained visibility of the remaining marks, and that hidden order becomes the top-to-bottom layer coloring of the output pyramid.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    _name, direction, head, _sprite_cells = _detect_sprite(g)
    marks = _extract_marks(g)
    final_head, _final_direction, colors, _path = _trace_consumption(head, direction, marks)
    return _render_pyramid(g.shape, final_head, colors)


def _build_direction_sequence(count: int) -> list[int]:
    first = random.randrange(4)
    dirs = [first]
    for _ in range(1, count):
        dirs.append((dirs[-1] + random.choice((-1, 1))) % 4)
    return dirs


def _build_path(directions: list[int], distances: list[int]) -> list[tuple[int, int]]:
    points = [(0, 0)]
    r = c = 0
    for direction, distance in zip(directions, distances):
        dr, dc = DIRS4[direction]
        r += dr * distance
        c += dc * distance
        points.append((int(r), int(c)))
    return points


def _path_has_unique_visibility(points: list[tuple[int, int]]) -> bool:
    for i in range(len(points) - 1):
        pr, pc = points[i]
        for j in range(i + 2, len(points)):
            qr, qc = points[j]
            if pr == qr or pc == qc:
                return False
    return len(set(points)) == len(points)


def _path_is_counterexample(points: list[tuple[int, int]]) -> bool:
    start = points[0]
    order = points[1:]
    row_major = sorted(order, key=lambda rc: (rc[0], rc[1]))
    col_major = sorted(order, key=lambda rc: (rc[1], rc[0]))
    by_start_distance = sorted(
        order,
        key=lambda rc: (abs(rc[0] - start[0]) + abs(rc[1] - start[1]), rc[0], rc[1]),
    )
    if order in (row_major, col_major, by_start_distance):
        return False

    remaining = list(order)
    pos = start
    for target in order:
        target_dist = abs(target[0] - pos[0]) + abs(target[1] - pos[1])
        other_dists = [
            abs(other[0] - pos[0]) + abs(other[1] - pos[1])
            for other in remaining
            if other != target
        ]
        if other_dists and min(other_dists) < target_dist:
            return True
        remaining.pop(0)
        pos = target
    return False


def _well_separated(
    sprite_cells: set[tuple[int, int]], marks: list[tuple[int, int]]
) -> bool:
    for a, b in combinations(marks, 2):
        if max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1:
            return False
    for mark in marks:
        for cell in sprite_cells:
            if max(abs(mark[0] - cell[0]), abs(mark[1] - cell[1])) <= 1:
                return False
    return True


def _translation_ranges(
    occupied: set[tuple[int, int]],
    final_head: tuple[int, int],
    mark_count: int,
    h: int,
    w: int,
) -> tuple[int, int, int, int] | None:
    coords = np.array(sorted(occupied), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    low_r = int(-r0)
    high_r = int(min(h - 1 - r1, h - mark_count - 1 - final_head[0]))
    low_c = int(-c0)
    high_c = int(w - 1 - c1)
    if low_r > high_r or low_c > high_c:
        return None
    return low_r, high_r, low_c, high_c


def _choose_canvas_and_translation(
    occupied: set[tuple[int, int]],
    final_head: tuple[int, int],
    mark_count: int,
    clip_mode: str,
) -> tuple[int, int, int, int] | None:
    coords = np.array(sorted(occupied), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    min_h = max(r1 - r0 + 1, final_head[0] - r0 + mark_count + 1, GRID_SIDE_MIN)
    min_w = max(c1 - c0 + 1, GRID_SIDE_MIN)
    if min_h > GRID_SIDE_MAX or min_w > GRID_SIDE_MAX:
        return None

    sizes = [(h, w) for h in range(min_h, GRID_SIDE_MAX + 1) for w in range(min_w, GRID_SIDE_MAX + 1)]
    random.shuffle(sizes)
    for h, w in sizes:
        ranges = _translation_ranges(occupied, final_head, mark_count, h, w)
        if ranges is None:
            continue
        low_r, high_r, low_c, high_c = ranges
        final_c_lo = final_head[1] + low_c
        final_c_hi = final_head[1] + high_c

        if clip_mode == "none":
            zone_lo, zone_hi = mark_count - 1, w - mark_count
        elif clip_mode == "left":
            zone_lo, zone_hi = 0, mark_count - 2
        else:
            zone_lo, zone_hi = w - mark_count + 1, w - 1

        zone_lo = max(zone_lo, final_c_lo)
        zone_hi = min(zone_hi, final_c_hi)
        if zone_lo > zone_hi:
            continue

        final_c = random.randint(zone_lo, zone_hi)
        tc = final_c - final_head[1]
        tr = random.randint(low_r, high_r)
        return int(h), int(w), int(tr), int(tc)

    return None


def _render_input(
    shape: tuple[int, int],
    template_name: str,
    start_dir: int,
    translated_points: list[tuple[int, int]],
    colors: list[int],
) -> np.ndarray:
    h, w = shape
    grid = np.full((h, w), BG, dtype=int)
    start_head = translated_points[0]
    sprite_cells = _placed_cells(start_head, TEMPLATES[template_name][start_dir])
    for r, c in sprite_cells:
        grid[r, c] = SPRITE
    for (r, c), color in zip(translated_points[1:], colors):
        grid[r, c] = int(color)
    return grid


def _clip_kind(final_head: tuple[int, int], mark_count: int, width: int) -> str:
    radius = mark_count - 1
    head_c = final_head[1]
    left = head_c - radius < 0
    right = head_c + radius >= width
    if left and not right:
        return "left"
    if right and not left:
        return "right"
    if left and right:
        return "both"
    return "none"


def _validate_candidate(grid: np.ndarray, clip_mode: str) -> bool:
    try:
        name, direction, head, sprite_cells = _detect_sprite(grid)
        marks = _extract_marks(grid)
        final_head, _final_direction, colors, path = _trace_consumption(head, direction, marks)
    except Exception:
        return False

    if name not in TEMPLATES:
        return False
    if final_head != path[-1]:
        return False

    points = [head, *path]
    if not _path_has_unique_visibility(points):
        return False
    if not _path_is_counterexample(points):
        return False
    if not _well_separated(sprite_cells, path):
        return False

    h, w = grid.shape
    if final_head[0] + len(colors) >= h:
        return False

    actual_clip = _clip_kind(final_head, len(colors), w)
    if clip_mode == "none" and actual_clip != "none":
        return False
    if clip_mode == "left" and actual_clip not in {"left", "both"}:
        return False
    if clip_mode == "right" and actual_clip not in {"right", "both"}:
        return False

    expected = _render_pyramid(grid.shape, final_head, colors)
    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    mark_count = random.choice(MARK_COUNT_CHOICES)
    directions = _build_direction_sequence(mark_count)
    distances = [random.randint(*STEP_DISTANCE_RANGE) for _ in range(mark_count)]
    points = _build_path(directions, distances)
    if not _path_has_unique_visibility(points):
        return None
    if len(set(directions)) < min(3, mark_count):
        return None
    if not _path_is_counterexample(points):
        return None

    template_name = random.choice(tuple(TEMPLATES))
    start_head = points[0]
    sprite_cells = _placed_cells(start_head, TEMPLATES[template_name][directions[0]])
    if not _well_separated(sprite_cells, points[1:]):
        return None

    clip_mode = random.choices(CLIP_MODES, weights=CLIP_WEIGHTS, k=1)[0]
    occupied = sprite_cells | set(points[1:])
    placement = _choose_canvas_and_translation(occupied, points[-1], mark_count, clip_mode)
    if placement is None:
        return None
    h, w, tr, tc = placement

    translated_points = [(r + tr, c + tc) for r, c in points]
    translated_sprite = {(r + tr, c + tc) for r, c in sprite_cells}
    if not _well_separated(translated_sprite, translated_points[1:]):
        return None

    colors = random.sample(MARK_COLORS, mark_count)
    grid = _render_input((h, w), template_name, directions[0], translated_points, colors)
    return grid if _validate_candidate(grid, clip_mode) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
