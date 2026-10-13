from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

SHAPES = ((2, 2), (2, 3), (3, 2))
MAX_GEN_ATTEMPTS = 500


@dataclass(frozen=True)
class Obj:
    color: int
    side: str
    r0: int
    c0: int
    h: int
    w: int
    radius: int
    gap: int

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.r0, self.c0, self.r0 + self.h - 1, self.c0 + self.w - 1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _intersects_expanded(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
    radius_b: int,
) -> bool:
    ar0, ac0, ar1, ac1 = box_a
    br0, bc0, br1, bc1 = box_b
    return _intervals_overlap(ar0, ar1, br0 - radius_b, br1 + radius_b) and _intervals_overlap(
        ac0, ac1, bc0 - radius_b, bc1 + radius_b
    )


def _find_divider(grid: np.ndarray) -> int:
    coords = np.argwhere(grid == GREY)
    assert coords.size > 0, "expected a grey divider"
    cols = np.unique(coords[:, 1])
    assert cols.size == 1, "divider must occupy exactly one column"
    divider = int(cols[0])
    assert coords.shape[0] == grid.shape[0], "divider must span the full height"
    assert np.all(grid[:, divider] == GREY), "divider must be contiguous"
    return divider


def _associate_marker(
    green_components: list[np.ndarray],
    used: set[int],
    obj_box: tuple[int, int, int, int],
) -> int:
    r0, c0, _r1, c1 = obj_box
    marker_row = r0 - 1
    assert marker_row >= 0, "every rectangle must have room for markers above it"
    matches: list[tuple[int, int, int]] = []
    for idx, comp in enumerate(green_components):
        if idx in used:
            continue
        rows = np.unique(comp[:, 0])
        if rows.size != 1 or int(rows[0]) != marker_row:
            continue
        cols = sorted(int(v) for v in comp[:, 1])
        if cols != list(range(cols[0], cols[-1] + 1)):
            continue
        if cols[-1] < c0 - 1 or cols[0] > c1 + 1:
            continue
        overlap = max(0, min(c1, cols[-1]) - max(c0, cols[0]) + 1)
        if overlap <= 0:
            continue
        matches.append((overlap, -abs(((c0 + c1) / 2.0) - ((cols[0] + cols[-1]) / 2.0)), idx))

    assert len(matches) == 1, "each rectangle must have exactly one adjacent marker row"
    idx = matches[0][2]
    used.add(idx)
    return int(green_components[idx].shape[0])


def _extract_objects(grid: np.ndarray) -> tuple[int, list[Obj]]:
    divider = _find_divider(grid)
    green_components = _components(grid == GREEN)
    used_markers: set[int] = set()
    objects: list[Obj] = []

    for color, side in ((RED, "left"), (BLUE, "right")):
        for comp in _components(grid == color):
            r0, c0, r1, c1 = _bbox(comp)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            assert (h, w) in SHAPES, "unexpected rectangle size"
            assert comp.shape[0] == h * w, "object must be a solid rectangle"
            if side == "left":
                assert c1 < divider, "red rectangles must be left of the divider"
                gap = divider - c1 - 1
            else:
                assert c0 > divider, "blue rectangles must be right of the divider"
                gap = c0 - divider - 1
            radius = _associate_marker(green_components, used_markers, (r0, c0, r1, c1))
            assert 1 <= radius <= 3
            objects.append(Obj(color=color, side=side, r0=r0, c0=c0, h=h, w=w, radius=radius, gap=gap))

    assert objects, "expected rectangles on both sides"
    assert len(used_markers) == len(objects), "every rectangle needs one marker group"
    return divider, objects


def _valid_position(
    box: tuple[int, int, int, int],
    radius: int,
    settled: list[tuple[tuple[int, int, int, int], int]],
) -> bool:
    for other_box, other_radius in settled:
        if _intersects_expanded(box, other_box, other_radius):
            return False
        if _intersects_expanded(other_box, box, radius):
            return False
    return True


def _solve_positions(objects: list[Obj], divider: int) -> dict[int, tuple[int, int, int, int]]:
    order = sorted(
        range(len(objects)),
        key=lambda i: (objects[i].gap, objects[i].r0, objects[i].c0, objects[i].color),
    )
    settled: list[tuple[tuple[int, int, int, int], int]] = []
    final_boxes: dict[int, tuple[int, int, int, int]] = {}

    for idx in order:
        obj = objects[idx]
        target_c0 = divider - obj.w if obj.side == "left" else divider + 1
        if obj.side == "left":
            candidates = range(target_c0, obj.c0 - 1, -1)
        else:
            candidates = range(target_c0, obj.c0 + 1)

        chosen_box: tuple[int, int, int, int] | None = None
        for c0 in candidates:
            box = (obj.r0, c0, obj.r0 + obj.h - 1, c0 + obj.w - 1)
            if _valid_position(box, obj.radius, settled):
                chosen_box = box
                break
        assert chosen_box is not None, "every rectangle must have at least its initial position available"
        final_boxes[idx] = chosen_box
        settled.append((chosen_box, obj.radius))

    return final_boxes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide colored rectangles toward the gray divider while respecting marker-encoded mutual exclusion zones.

    Input structure:
    - The grid has a black background, one full-height vertical gray divider, red solid rectangles on the left, blue solid rectangles on the right, and green horizontal marker pixels directly above each rectangle.
    - Each rectangle is a 2x2, 2x3, or 3x2 solid block, and the number of adjacent green marker pixels above it encodes that rectangle's exclusion radius.

    Rule:
    - Red rectangles slide right toward the divider and blue rectangles slide left toward the divider, always staying on their original rows and never crossing the divider.
    - Process rectangles by increasing initial distance to the divider; each one takes the closest-to-divider position that does not place its bbox inside any already-settled rectangle's exclusion zone and does not let its own exclusion zone cover an already-settled rectangle.
    - Keep the gray divider and all green marker pixels fixed, erase the old rectangle locations back to black, and redraw each rectangle at its final position with the same color and shape.

    Key insight:
    - The green markers do not mark movement length directly; they encode how strongly a rectangle repels every other rectangle, so the final layout is determined by combining slide direction, priority by divider distance, and pairwise exclusion constraints.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"

    divider, objects = _extract_objects(g)
    final_boxes = _solve_positions(objects, divider)

    out = g.copy()
    for obj in objects:
        r0, c0, r1, c1 = obj.bbox
        out[r0 : r1 + 1, c0 : c1 + 1] = BLACK

    for idx, obj in enumerate(objects):
        r0, c0, r1, c1 = final_boxes[idx]
        out[r0 : r1 + 1, c0 : c1 + 1] = obj.color

    return out


def _shape_for_radius(radius: int, rng: np.random.Generator) -> tuple[int, int]:
    choices = [shape for shape in SHAPES if shape[1] >= radius]
    return choices[int(rng.integers(0, len(choices)))]


def _place_side_rows(
    heights: list[int],
    grid_h: int,
    rng: np.random.Generator,
    fixed: dict[int, int] | None = None,
) -> list[int] | None:
    fixed = {} if fixed is None else dict(fixed)
    intervals: list[tuple[int, int]] = []
    tops = [-1] * len(heights)

    for idx, top in fixed.items():
        h = heights[idx]
        if top < 1 or top + h - 1 >= grid_h:
            return None
        start = top - 1
        end = top + h - 1
        for a0, a1 in intervals:
            if _intervals_overlap(start, end, a0, a1):
                return None
        intervals.append((start, end))
        tops[idx] = top

    order = [idx for idx in np.argsort([-h for h in heights]) if idx not in fixed]
    for idx in order:
        h = heights[idx]
        candidates = []
        for top in range(1, grid_h - h + 1):
            start = top - 1
            end = top + h - 1
            if all(not _intervals_overlap(start, end, a0, a1) for a0, a1 in intervals):
                candidates.append(top)
        if not candidates:
            return None
        top = int(candidates[int(rng.integers(0, len(candidates)))])
        intervals.append((top - 1, top + h - 1))
        tops[idx] = top

    return tops


def _build_grid(divider: int, grid_h: int, grid_w: int, objects: list[Obj], rng: np.random.Generator) -> np.ndarray:
    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    grid[:, divider] = GREY

    for obj in objects:
        r0, c0, r1, c1 = obj.bbox
        grid[r0 : r1 + 1, c0 : c1 + 1] = obj.color
        marker_start = c0 + int(rng.integers(0, obj.w - obj.radius + 1))
        grid[r0 - 1, marker_start : marker_start + obj.radius] = GREEN

    return grid


def _blocked_count(objects: list[Obj], divider: int) -> int:
    try:
        final_boxes = _solve_positions(objects, divider)
    except AssertionError:
        return -1
    blocked = 0
    for idx, obj in enumerate(objects):
        target_c0 = divider - obj.w if obj.side == "left" else divider + 1
        if final_boxes[idx][1] != target_c0:
            blocked += 1
    return blocked


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    left_count = int(rng.integers(2, 4))
    right_count = int(rng.integers(2, 4))
    total = left_count + right_count

    grid_h = int(rng.integers(13, 19))
    grid_w = int(rng.integers(21, 28))
    divider = grid_w // 2 + int(rng.integers(-1, 2))
    common_max_gap = min(divider - 3, grid_w - divider - 4)
    if common_max_gap < total + 1:
        return None

    block_from_left = bool(rng.integers(0, 2))
    left_specs: list[dict[str, int]] = []
    right_specs: list[dict[str, int]] = []

    blocker_radius = int(rng.integers(2, 4))
    blocked_radius = int(rng.integers(1, 4))
    blocker_h, blocker_w = _shape_for_radius(blocker_radius, rng)
    blocked_h, blocked_w = _shape_for_radius(blocked_radius, rng)

    if block_from_left:
        left_specs.append({"radius": blocker_radius, "h": blocker_h, "w": blocker_w, "gap": 1})
        right_specs.append(
            {
                "radius": blocked_radius,
                "h": blocked_h,
                "w": blocked_w,
                "gap": int(rng.integers(3, common_max_gap + 1)),
            }
        )
    else:
        right_specs.append({"radius": blocker_radius, "h": blocker_h, "w": blocker_w, "gap": 1})
        left_specs.append(
            {
                "radius": blocked_radius,
                "h": blocked_h,
                "w": blocked_w,
                "gap": int(rng.integers(3, common_max_gap + 1)),
            }
        )

    while len(left_specs) < left_count:
        radius = int(rng.integers(1, 4))
        h, w = _shape_for_radius(radius, rng)
        left_specs.append({"radius": radius, "h": h, "w": w, "gap": -1})
    while len(right_specs) < right_count:
        radius = int(rng.integers(1, 4))
        h, w = _shape_for_radius(radius, rng)
        right_specs.append({"radius": radius, "h": h, "w": w, "gap": -1})

    reserved_gaps = {spec["gap"] for spec in left_specs + right_specs if spec["gap"] > 0}
    remaining_pool = [gap for gap in range(2, common_max_gap + 1) if gap not in reserved_gaps]
    if len(remaining_pool) < total - len(reserved_gaps):
        return None
    rng.shuffle(remaining_pool)
    for spec in left_specs + right_specs:
        if spec["gap"] < 0:
            spec["gap"] = int(remaining_pool.pop())

    heights_left = [int(spec["h"]) for spec in left_specs]
    heights_right = [int(spec["h"]) for spec in right_specs]

    overlap_top = int(rng.integers(1, grid_h - max(blocker_h, blocked_h) + 1))
    blocked_shift_choices = [delta for delta in (-1, 0, 1) if 1 <= overlap_top + delta <= grid_h - blocked_h]
    blocked_shift = int(blocked_shift_choices[int(rng.integers(0, len(blocked_shift_choices)))])

    if block_from_left:
        left_tops = _place_side_rows(heights_left, grid_h, rng, fixed={0: overlap_top})
        right_tops = _place_side_rows(heights_right, grid_h, rng, fixed={0: overlap_top + blocked_shift})
    else:
        left_tops = _place_side_rows(heights_left, grid_h, rng, fixed={0: overlap_top + blocked_shift})
        right_tops = _place_side_rows(heights_right, grid_h, rng, fixed={0: overlap_top})
    if left_tops is None or right_tops is None:
        return None

    objects: list[Obj] = []
    for spec, top in zip(left_specs, left_tops):
        c0 = divider - int(spec["gap"]) - int(spec["w"])
        if c0 < 0:
            return None
        objects.append(
            Obj(
                color=RED,
                side="left",
                r0=int(top),
                c0=int(c0),
                h=int(spec["h"]),
                w=int(spec["w"]),
                radius=int(spec["radius"]),
                gap=int(spec["gap"]),
            )
        )
    for spec, top in zip(right_specs, right_tops):
        c0 = divider + 1 + int(spec["gap"])
        if c0 + int(spec["w"]) > grid_w:
            return None
        objects.append(
            Obj(
                color=BLUE,
                side="right",
                r0=int(top),
                c0=int(c0),
                h=int(spec["h"]),
                w=int(spec["w"]),
                radius=int(spec["radius"]),
                gap=int(spec["gap"]),
            )
        )

    if _blocked_count(objects, divider) < 1:
        return None

    grid = _build_grid(divider, grid_h, grid_w, objects, rng)
    colors = set(int(v) for v in np.unique(grid))
    if not {RED, BLUE, GREEN, GREY}.issubset(colors):
        return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
