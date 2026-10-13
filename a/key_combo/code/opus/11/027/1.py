from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

VISIBLE_SIDES = ("up", "down", "left", "right")
RECT_SHAPES = ((6, 7), (7, 6), (7, 7))
MAX_GENERATION_ATTEMPTS = 500


@dataclass(frozen=True)
class MarkerInfo:
    coord: tuple[int, int]
    side: str
    distance: int


@dataclass
class RectInfo:
    bbox: tuple[int, int, int, int]
    markers: list[tuple[int, int]]


def _get_bg(grid: np.ndarray) -> int:
    values = list(map(int, np.asarray(grid).ravel()))
    return int(Counter(values).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _nearest_side(r: int, c: int, bbox: tuple[int, int, int, int]) -> tuple[str, int]:
    r0, c0, r1, c1 = bbox
    distances = {
        "up": r - r0,
        "down": r1 - r,
        "left": c - c0,
        "right": c1 - c,
    }
    best = min(distances.values())
    sides = [side for side, distance in distances.items() if distance == best]
    if len(sides) != 1:
        raise AssertionError("marker must have a unique nearest edge")
    return sides[0], int(best)


def _marker_candidates(bbox: tuple[int, int, int, int]) -> list[MarkerInfo]:
    r0, c0, r1, c1 = bbox
    out: list[MarkerInfo] = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            try:
                side, distance = _nearest_side(r, c, bbox)
            except AssertionError:
                continue
            out.append(MarkerInfo((r, c), side, distance))
    return out


def _green_target(marker: MarkerInfo) -> tuple[int, int] | None:
    if marker.distance <= 1:
        return None
    dr, dc = DIRS[marker.side]
    return marker.coord[0] + dr, marker.coord[1] + dc


def _find_rectangles(grid: np.ndarray) -> list[RectInfo]:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("grid must be 2D")
    bg = _get_bg(g)
    components = _components_mask(g != bg)
    if not components:
        raise AssertionError("expected at least one foreground rectangle")

    rects: list[RectInfo] = []
    for coords in components:
        r0, c0, r1, c1 = _bbox_coords(coords)
        block = g[r0 : r1 + 1, c0 : c1 + 1]
        if not np.all(block != bg):
            raise AssertionError("rectangle bbox must stay filled in the input")
        if not np.all(np.isin(block, (BLUE, RED))):
            raise AssertionError("unexpected colors inside rectangle bbox")
        if not (
            np.all(block[0, :] == BLUE)
            and np.all(block[-1, :] == BLUE)
            and np.all(block[:, 0] == BLUE)
            and np.all(block[:, -1] == BLUE)
        ):
            raise AssertionError("rectangle border must remain blue")

        marker_coords = [tuple(map(int, rc)) for rc in np.argwhere(block == RED) + np.array([r0, c0])]
        if not marker_coords:
            raise AssertionError("each rectangle must contain at least one marker")
        rects.append(RectInfo((r0, c0, r1, c1), sorted(marker_coords)))

    rects.sort(key=lambda rect: rect.bbox)
    return rects


def _solve(grid: np.ndarray, *, hollow: bool, rays: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    rects = _find_rectangles(g)

    if hollow:
        for rect in rects:
            r0, c0, r1, c1 = rect.bbox
            interior = out[r0 + 1 : r1, c0 + 1 : c1]
            interior[interior == BLUE] = BG

    if rays:
        ordered: list[tuple[tuple[int, int], tuple[int, int, int, int]]] = []
        for rect in rects:
            for marker in rect.markers:
                ordered.append((marker, rect.bbox))
        ordered.sort()

        for (r, c), bbox in ordered:
            side, _distance = _nearest_side(r, c, bbox)
            dr, dc = DIRS[side]
            nr, nc = r + dr, c + dc
            r0, c0, r1, c1 = bbox
            while r0 < nr < r1 and c0 < nc < c1:
                if out[nr, nc] != RED:
                    out[nr, nc] = GREEN
                nr += dr
                nc += dc

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Hollow each blue solid rectangle, then emit a green orthogonal ray from every red marker toward its unique nearest border.

    Input structure
    - The background is black.
    - Each foreground component is a blue axis-aligned solid rectangle whose interior may contain one or two red singleton markers.
    - Markers are strictly inside their rectangle, never on the blue border, and each has a unique nearest rectangle edge.

    Rule
    - Treat each mixed blue-red component's bounding box as the original solid rectangle.
    - Hollow the rectangle by turning every strictly interior blue cell black while keeping the blue border and all red markers.
    - For each red marker, find the unique nearest edge of its rectangle and draw green cells from the marker toward that edge, stopping before the blue border.
    - Process markers in reading order; red markers remain red in the output.

    Key insight
    - The red marker does not encode a fixed direction: its off-center position inside the rectangle implicitly selects the nearest side, and the ray only exists after the rectangle has been hollowed into a blocking frame.
    """
    return _solve(grid, hollow=True, rays=True)


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 2) -> bool:
    return not (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _choose_rectangles(
    rng: np.random.Generator, h: int, w: int, count: int
) -> list[tuple[int, int, int, int]] | None:
    rects: list[tuple[int, int, int, int]] = []

    if count == 2:
        shape_plan = [RECT_SHAPES[int(rng.integers(0, 2))], RECT_SHAPES[int(rng.integers(1, 3))]]
    else:
        shape_plan = [(6, 7), (7, 6), RECT_SHAPES[int(rng.integers(0, len(RECT_SHAPES)))]]
        rng.shuffle(shape_plan)

    for rh, rw in shape_plan:
        placed = False
        for _ in range(120):
            r0 = int(rng.integers(1, h - rh))
            c0 = int(rng.integers(1, w - rw))
            bbox = (r0, c0, r0 + rh - 1, c0 + rw - 1)
            if any(_boxes_too_close(bbox, other) for other in rects):
                continue
            rects.append(bbox)
            placed = True
            break
        if not placed:
            return None

    return sorted(rects)


def _pick_markers_for_rects(
    rng: np.random.Generator, rects: Iterable[tuple[int, int, int, int]]
) -> list[list[MarkerInfo]] | None:
    rects = list(rects)
    candidate_lists = [_marker_candidates(bbox) for bbox in rects]
    vertical_rects = [i for i, cands in enumerate(candidate_lists) if any(c.side in {"up", "down"} and c.distance >= 2 for c in cands)]
    horizontal_rects = [i for i, cands in enumerate(candidate_lists) if any(c.side in {"left", "right"} and c.distance >= 2 for c in cands)]
    if not vertical_rects or not horizontal_rects:
        return None

    include_zero = bool(rng.random() < 0.35)
    want_three_sides = bool(rng.random() < 0.55)

    for _ in range(300):
        chosen: list[list[MarkerInfo]] = [[] for _ in rects]

        v_rect = int(rng.choice(vertical_rects))
        v_pool = [cand for cand in candidate_lists[v_rect] if cand.side in {"up", "down"} and cand.distance >= 2]
        chosen[v_rect].append(v_pool[int(rng.integers(0, len(v_pool)))])

        h_rect = int(rng.choice(horizontal_rects))
        h_pool = [cand for cand in candidate_lists[h_rect] if cand.side in {"left", "right"} and cand.distance >= 2]
        picked_h = h_pool[int(rng.integers(0, len(h_pool)))]
        if picked_h.coord not in {m.coord for group in chosen for m in group}:
            chosen[h_rect].append(picked_h)
        else:
            continue

        for i, cands in enumerate(candidate_lists):
            if chosen[i]:
                continue
            k = 2 if rng.random() < 0.45 else 1
            k = min(k, len(cands))
            order = rng.permutation(len(cands))
            for idx in order:
                cand = cands[int(idx)]
                if cand.coord in {m.coord for group in chosen for m in group}:
                    continue
                chosen[i].append(cand)
                if len(chosen[i]) == k:
                    break
            if not chosen[i]:
                break
        if any(not group for group in chosen):
            continue

        if include_zero:
            zero_options: list[tuple[int, MarkerInfo]] = []
            for i, cands in enumerate(candidate_lists):
                if len(chosen[i]) >= 2:
                    continue
                for cand in cands:
                    if cand.distance == 1 and cand.coord not in {m.coord for group in chosen for m in group}:
                        zero_options.append((i, cand))
            if zero_options:
                idx = int(rng.integers(0, len(zero_options)))
                rect_index, cand = zero_options[idx]
                chosen[rect_index].append(cand)

        if any(len(group) > 2 for group in chosen):
            continue

        all_markers = [marker for group in chosen for marker in group]
        sides = {marker.side for marker in all_markers}
        if not (sides & {"up", "down"}) or not (sides & {"left", "right"}):
            continue
        if want_three_sides and len(sides) < 3:
            continue

        targets = [_green_target(marker) for marker in all_markers]
        concrete_targets = [target for target in targets if target is not None]
        if len(concrete_targets) != len(set(concrete_targets)):
            continue
        marker_cells = {marker.coord for marker in all_markers}
        if any(target in marker_cells for target in concrete_targets):
            continue

        return chosen

    return None


def _build_input(rects: Iterable[tuple[int, int, int, int]], marker_groups: Iterable[Iterable[MarkerInfo]], h: int, w: int) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for bbox, markers in zip(rects, marker_groups):
        r0, c0, r1, c1 = bbox
        grid[r0 : r1 + 1, c0 : c1 + 1] = BLUE
        for marker in markers:
            mr, mc = marker.coord
            grid[mr, mc] = RED
    return grid


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        rects = _find_rectangles(grid)
    except AssertionError:
        return False

    if not (2 <= len(rects) <= 3):
        return False

    all_markers: list[MarkerInfo] = []
    green_targets: set[tuple[int, int]] = set()
    untouched_interior_blue = 0
    visible_sides: set[str] = set()

    for rect in rects:
        r0, c0, r1, c1 = rect.bbox
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if (h, w) not in RECT_SHAPES:
            return False
        if not (1 <= len(rect.markers) <= 2):
            return False

        target_cells_for_rect: set[tuple[int, int]] = set()
        for marker in rect.markers:
            side, distance = _nearest_side(marker[0], marker[1], rect.bbox)
            info = MarkerInfo(marker, side, distance)
            all_markers.append(info)
            if distance >= 2:
                visible_sides.add(side)
                target = _green_target(info)
                if target is None or target in green_targets:
                    return False
                green_targets.add(target)
                target_cells_for_rect.add(target)

        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if (r, c) not in rect.markers and (r, c) not in target_cells_for_rect:
                    untouched_interior_blue += 1

    if not visible_sides & {"up", "down"}:
        return False
    if not visible_sides & {"left", "right"}:
        return False
    if len(green_targets) < 2:
        return False
    if untouched_interior_blue <= 0:
        return False

    marker_cells = {info.coord for info in all_markers}
    if green_targets & marker_cells:
        return False

    correct = main(grid)
    hollow_only = _solve(grid, hollow=True, rays=False)
    rays_only = _solve(grid, hollow=False, rays=True)

    if np.array_equal(correct, grid):
        return False
    if np.array_equal(correct, hollow_only):
        return False
    if np.array_equal(correct, rays_only):
        return False
    if set(map(tuple, np.argwhere(correct == GREEN))) != green_targets:
        return False
    if not np.all(correct[grid == RED] == RED):
        return False
    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        h = int(rng.integers(17, 23))
        w = int(rng.integers(17, 23))
        rect_count = 2 if rng.random() < 0.55 else 3
        rects = _choose_rectangles(rng, h, w, rect_count)
        if rects is None:
            continue
        marker_groups = _pick_markers_for_rects(rng, rects)
        if marker_groups is None:
            continue

        grid = _build_input(rects, marker_groups, h, w)
        if _validate_generated(grid):
            return grid

    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
