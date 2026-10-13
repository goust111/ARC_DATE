from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import numpy as np


DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


@dataclass(frozen=True)
class Fence:
    color: int
    opening: str
    bbox: tuple[int, int, int, int]
    seed: tuple[int, int]
    area: int


@dataclass(frozen=True)
class FenceSpec:
    opening: str
    h: int
    w: int
    fence_color: int
    seed_color: int
    seed_rel: tuple[int, int]

    @property
    def area(self) -> int:
        return len(_interior_rel_cells(self.h, self.w, self.opening))


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _interior_rel_cells(h: int, w: int, opening: str) -> list[tuple[int, int]]:
    if opening == "up":
        return [(r, c) for r in range(0, h - 1) for c in range(1, w - 1)]
    if opening == "down":
        return [(r, c) for r in range(1, h) for c in range(1, w - 1)]
    if opening == "left":
        return [(r, c) for r in range(1, h - 1) for c in range(0, w - 1)]
    assert opening == "right"
    return [(r, c) for r in range(1, h - 1) for c in range(1, w)]


def _fence_rel_cells(h: int, w: int, opening: str) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    if opening == "up":
        if h < 2 or w < 3:
            return set()
        for r in range(h):
            cells.add((r, 0))
            cells.add((r, w - 1))
        for c in range(w):
            cells.add((h - 1, c))
        return cells
    if opening == "down":
        if h < 2 or w < 3:
            return set()
        for r in range(h):
            cells.add((r, 0))
            cells.add((r, w - 1))
        for c in range(w):
            cells.add((0, c))
        return cells
    if opening == "left":
        if h < 3 or w < 2:
            return set()
        for c in range(w):
            cells.add((0, c))
            cells.add((h - 1, c))
        for r in range(h):
            cells.add((r, w - 1))
        return cells
    assert opening == "right"
    if h < 3 or w < 2:
        return set()
    for c in range(w):
        cells.add((0, c))
        cells.add((h - 1, c))
    for r in range(h):
        cells.add((r, 0))
    return cells


def _fence_abs_cells(h: int, w: int, opening: str, top: int, left: int) -> set[tuple[int, int]]:
    return {(top + r, left + c) for r, c in _fence_rel_cells(h, w, opening)}


def _object_cells(spec: FenceSpec, top: int, left: int) -> set[tuple[int, int]]:
    cells = _fence_abs_cells(spec.h, spec.w, spec.opening, top, left)
    sr, sc = spec.seed_rel
    cells.add((top + sr, left + sc))
    return cells


def _expanded_bbox_cells(
    top: int, left: int, h: int, w: int, grid_h: int, grid_w: int, margin: int = 1
) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for r in range(max(0, top - margin), min(grid_h, top + h + margin)):
        for c in range(max(0, left - margin), min(grid_w, left + w + margin)):
            cells.add((r, c))
    return cells


def _full_ray_cells(spec: FenceSpec, top: int, left: int, grid_h: int, grid_w: int) -> list[tuple[int, int]] | None:
    dr, dc = DIRS[spec.opening]
    sr = top + spec.seed_rel[0]
    sc = left + spec.seed_rel[1]
    cells: list[tuple[int, int]] = []
    for step in range(1, spec.area + 1):
        rr = sr + dr * step
        cc = sc + dc * step
        if not (0 <= rr < grid_h and 0 <= cc < grid_w):
            return None
        cells.append((rr, cc))
    return cells


def _ray_until_blocked(
    origin: tuple[int, int], opening: str, area: int, occupied: set[tuple[int, int]], grid_h: int, grid_w: int
) -> tuple[list[tuple[int, int]], str]:
    dr, dc = DIRS[opening]
    sr, sc = origin
    cells: list[tuple[int, int]] = []
    for step in range(1, area + 1):
        rr = sr + dr * step
        cc = sc + dc * step
        if not (0 <= rr < grid_h and 0 <= cc < grid_w):
            return cells, "boundary"
        if (rr, cc) in occupied:
            return cells, "blocked"
        cells.append((rr, cc))
    return cells, "full"


def _detect_fences(grid: np.ndarray) -> list[Fence]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    fences: list[Fence] = []
    used_fence_cells: set[tuple[int, int]] = set()
    used_seed_cells: set[tuple[int, int]] = set()

    for color in sorted(int(c) for c in np.unique(g) if int(c) != 0):
        for comp in _components(g == color):
            if len(comp) < 5:
                continue
            r0, c0, r1, c1 = _bbox(comp)
            comp_set = set(comp)
            matches: list[Fence] = []
            for opening in ("up", "down", "left", "right"):
                expected = _fence_abs_cells(r1 - r0 + 1, c1 - c0 + 1, opening, r0, c0)
                if not expected or comp_set != expected:
                    continue
                interior = [
                    (r0 + rr, c0 + cc)
                    for rr, cc in _interior_rel_cells(r1 - r0 + 1, c1 - c0 + 1, opening)
                ]
                if not interior:
                    continue
                nonzero = [pos for pos in interior if int(g[pos]) != 0]
                if len(nonzero) != 1 or int(g[nonzero[0]]) == color:
                    continue
                if any(int(g[pos]) not in (0, int(g[nonzero[0]])) for pos in interior):
                    continue
                matches.append(
                    Fence(
                        color=color,
                        opening=opening,
                        bbox=(r0, c0, r1, c1),
                        seed=nonzero[0],
                        area=len(interior),
                    )
                )
            assert len(matches) == 1, "each fence component must match exactly one opening direction"
            fence = matches[0]
            fences.append(fence)
            used_fence_cells |= comp_set
            used_seed_cells.add(fence.seed)

    assert fences, "expected at least one fence"
    all_nonzero = {(r, c) for r in range(h) for c in range(w) if int(g[r, c]) != 0}
    assert all_nonzero == used_fence_cells | used_seed_cells, "unexpected nonzero object outside fence-seed pairs"
    assert len(used_seed_cells) == len(fences), "each fence must contain exactly one unique seed"
    return sorted(fences, key=lambda fence: (fence.seed[0], fence.seed[1], fence.color))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a fence-colored ray from each interior seed in the fence opening direction, using the fenced interior area as the ray length.

    Input structure:
    - The background is black.
    - Several disconnected monochrome fence components appear; each fence is a
      three-sided axis-aligned U/L-like outline with exactly one open side.
    - Inside the rectangular trough of each fence there is exactly one
      different-colored single-pixel seed, plus optional black empty cells.

    Rule:
    - For every fence, infer which side is open by matching the component to the
      missing-side rectangle outline.
    - Count the trough area N as the number of cells in the fence's interior
      rectangular region, including the seed cell and black empty cells.
    - Starting from the seed, draw a straight ray of up to N cells in the open
      direction using the fence color; do not recolor the seed itself.
    - The fence and seed stay unchanged, and the ray stops before the first
      pre-existing non-black cell or the grid boundary.

    Key insight:
    - The opening direction comes from the missing side of the fence, while the
      ray length comes from the trough area and the ray color comes from the
      fence rather than the seed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    fences = _detect_fences(g)
    occupied = {(r, c) for r in range(h) for c in range(w) if int(g[r, c]) != 0}

    out = g.copy()
    for fence in fences:
        cells, _state = _ray_until_blocked(fence.seed, fence.opening, fence.area, occupied, h, w)
        for r, c in cells:
            out[r, c] = fence.color
    return out


def _random_spec(fence_color: int, seed_color: int) -> FenceSpec:
    opening = random.choice(tuple(DIRS))
    if opening in {"up", "down"}:
        h = random.randint(2, 5)
        w = random.randint(3, 6)
    else:
        h = random.randint(3, 6)
        w = random.randint(2, 5)
    interior = _interior_rel_cells(h, w, opening)
    seed_rel = random.choice(interior)
    return FenceSpec(
        opening=opening,
        h=h,
        w=w,
        fence_color=fence_color,
        seed_color=seed_color,
        seed_rel=seed_rel,
    )


def _place_full_object(
    spec: FenceSpec,
    grid_h: int,
    grid_w: int,
    forbidden_object: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    reserved_rays: set[tuple[int, int]],
    extra_object_forbidden: set[tuple[int, int]] | None = None,
    extra_ray_forbidden: set[tuple[int, int]] | None = None,
) -> tuple[int, int, list[tuple[int, int]]] | None:
    extra_object_forbidden = extra_object_forbidden or set()
    extra_ray_forbidden = extra_ray_forbidden or set()
    positions = [(r, c) for r in range(grid_h - spec.h + 1) for c in range(grid_w - spec.w + 1)]
    random.shuffle(positions)

    for top, left in positions:
        obj_cells = _object_cells(spec, top, left)
        if obj_cells & forbidden_object:
            continue
        if obj_cells & reserved_rays:
            continue
        if obj_cells & extra_object_forbidden:
            continue

        ray_cells = _full_ray_cells(spec, top, left, grid_h, grid_w)
        if ray_cells is None:
            continue
        ray_set = set(ray_cells)
        if ray_set & occupied:
            continue
        if ray_set & reserved_rays:
            continue
        if ray_set & extra_ray_forbidden:
            continue
        return top, left, ray_cells
    return None


def _place_blocker(
    spec: FenceSpec,
    grid_h: int,
    grid_w: int,
    forbidden_object: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    reserved_rays: set[tuple[int, int]],
    target_origin: tuple[int, int],
    target_opening: str,
    target_area: int,
    target_path: list[tuple[int, int]],
) -> tuple[int, int, list[tuple[int, int]]] | None:
    positions = [(r, c) for r in range(grid_h - spec.h + 1) for c in range(grid_w - spec.w + 1)]
    random.shuffle(positions)
    target_first = target_path[0]
    later_target_cells = set(target_path[1:])
    target_path_set = set(target_path)

    for top, left in positions:
        obj_cells = _object_cells(spec, top, left)
        if obj_cells & forbidden_object:
            continue
        if obj_cells & reserved_rays:
            continue
        if target_first in obj_cells:
            continue
        if not (obj_cells & later_target_cells):
            continue

        ray_cells = _full_ray_cells(spec, top, left, grid_h, grid_w)
        if ray_cells is None:
            continue
        ray_set = set(ray_cells)
        if ray_set & occupied:
            continue
        if ray_set & reserved_rays:
            continue
        if ray_set & target_path_set:
            continue

        test_occupied = occupied | obj_cells
        target_ray, state = _ray_until_blocked(target_origin, target_opening, target_area, test_occupied, grid_h, grid_w)
        if state != "blocked" or not target_ray or len(target_ray) >= target_area:
            continue
        return top, left, ray_cells
    return None


def _render_grid(grid_h: int, grid_w: int, placements: list[tuple[FenceSpec, int, int]]) -> np.ndarray:
    grid = np.zeros((grid_h, grid_w), dtype=int)
    for spec, top, left in placements:
        for r, c in _fence_abs_cells(spec.h, spec.w, spec.opening, top, left):
            grid[r, c] = spec.fence_color
        sr, sc = spec.seed_rel
        grid[top + sr, left + sc] = spec.seed_color
    return grid


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(16, 26)
    grid_w = random.randint(16, 26)
    n_fences = random.randint(2, 4)
    pattern = random.choice(["all_clear", "mixed"])

    palette = list(range(1, 10))
    random.shuffle(palette)
    fence_colors = palette[:n_fences]
    seed_colors = palette[n_fences : 2 * n_fences]
    specs = [_random_spec(fence_colors[i], seed_colors[i]) for i in range(n_fences)]
    random.shuffle(specs)

    forbidden_object: set[tuple[int, int]] = set()
    occupied: set[tuple[int, int]] = set()
    reserved_rays: set[tuple[int, int]] = set()
    placements: list[tuple[FenceSpec, int, int]] = []

    if pattern == "mixed":
        target_spec = specs[0]
        blocker_spec = specs[1]
        other_specs = specs[2:]

        target_place = _place_full_object(
            target_spec, grid_h, grid_w, forbidden_object, occupied, reserved_rays
        )
        if target_place is None:
            return None
        target_top, target_left, target_path = target_place
        placements.append((target_spec, target_top, target_left))
        occupied |= _object_cells(target_spec, target_top, target_left)
        forbidden_object |= _expanded_bbox_cells(
            target_top, target_left, target_spec.h, target_spec.w, grid_h, grid_w
        )

        for spec in other_specs:
            placed = _place_full_object(
                spec,
                grid_h,
                grid_w,
                forbidden_object,
                occupied,
                reserved_rays,
                extra_object_forbidden=set(target_path),
                extra_ray_forbidden=set(target_path),
            )
            if placed is None:
                return None
            top, left, ray_cells = placed
            placements.append((spec, top, left))
            occupied |= _object_cells(spec, top, left)
            reserved_rays |= set(ray_cells)
            forbidden_object |= _expanded_bbox_cells(top, left, spec.h, spec.w, grid_h, grid_w)

        target_origin = (target_top + target_spec.seed_rel[0], target_left + target_spec.seed_rel[1])
        blocker_place = _place_blocker(
            blocker_spec,
            grid_h,
            grid_w,
            forbidden_object,
            occupied,
            reserved_rays,
            target_origin,
            target_spec.opening,
            target_spec.area,
            target_path,
        )
        if blocker_place is None:
            return None
        blocker_top, blocker_left, blocker_ray = blocker_place
        placements.append((blocker_spec, blocker_top, blocker_left))
        occupied |= _object_cells(blocker_spec, blocker_top, blocker_left)
        reserved_rays |= set(blocker_ray)
        forbidden_object |= _expanded_bbox_cells(
            blocker_top, blocker_left, blocker_spec.h, blocker_spec.w, grid_h, grid_w
        )
    else:
        for spec in specs:
            placed = _place_full_object(spec, grid_h, grid_w, forbidden_object, occupied, reserved_rays)
            if placed is None:
                return None
            top, left, ray_cells = placed
            placements.append((spec, top, left))
            occupied |= _object_cells(spec, top, left)
            reserved_rays |= set(ray_cells)
            forbidden_object |= _expanded_bbox_cells(top, left, spec.h, spec.w, grid_h, grid_w)

    grid = _render_grid(grid_h, grid_w, placements)
    expected = grid.copy()
    input_occupied = {(r, c) for r in range(grid_h) for c in range(grid_w) if int(grid[r, c]) != 0}
    states: list[str] = []
    painted_cells: set[tuple[int, int]] = set()

    for spec, top, left in placements:
        origin = (top + spec.seed_rel[0], left + spec.seed_rel[1])
        cells, state = _ray_until_blocked(origin, spec.opening, spec.area, input_occupied, grid_h, grid_w)
        if state == "boundary" or not cells:
            return None
        if painted_cells & set(cells):
            return None
        painted_cells |= set(cells)
        states.append(state)
        for r, c in cells:
            expected[r, c] = spec.fence_color

    if pattern == "all_clear":
        if any(state != "full" for state in states):
            return None
    else:
        if "full" not in states or "blocked" not in states:
            return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fence-ray input")
