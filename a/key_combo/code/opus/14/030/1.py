from __future__ import annotations

from collections import deque
import random
from typing import Iterable

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

GEN_RETRIES = 500


def _to_set(coords: Iterable[tuple[int, int]]) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in coords}


def _normalize(cells: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    pts = sorted(_to_set(cells))
    r0 = min(r for r, _ in pts)
    c0 = min(c for _, c in pts)
    return tuple(sorted((r - r0, c - c0) for r, c in pts))


def _shape_dims(cells: Iterable[tuple[int, int]]) -> tuple[int, int]:
    pts = _to_set(cells)
    return max(r for r, _ in pts) + 1, max(c for _, c in pts) + 1


def _canon_shape(cells: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    pts = list(_normalize(cells))

    def rot90(shape: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
        h, w = _shape_dims(shape)
        return [(c, h - 1 - r) for r, c in shape]

    def flip(shape: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
        h, w = _shape_dims(shape)
        return [(r, w - 1 - c) for r, c in shape]

    variants = []
    cur = pts
    for _ in range(4):
        variants.append(_normalize(cur))
        variants.append(_normalize(flip(cur)))
        cur = rot90(cur)
    return min(variants)


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr, dc in DIR4]


def _component_list(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _find_components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    return _component_list(grid == int(color))


def _is_connected(cells: Iterable[tuple[int, int]]) -> bool:
    pts = _to_set(cells)
    start = next(iter(pts))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in pts and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return seen == pts


def _has_vertical_symmetry(cells: Iterable[tuple[int, int]]) -> bool:
    pts = _to_set(cells)
    cols = [c for _, c in pts]
    axis2 = min(cols) + max(cols)
    return all((r, axis2 - c) in pts for r, c in pts)


def _has_horizontal_symmetry(cells: Iterable[tuple[int, int]]) -> bool:
    pts = _to_set(cells)
    rows = [r for r, _ in pts]
    axis2 = min(rows) + max(rows)
    return all((axis2 - r, c) in pts for r, c in pts)


def _detect_blue_axis(grid: np.ndarray) -> tuple[str, float, list[tuple[int, int]]]:
    blue_components = _find_components_of_color(grid, BLUE)
    assert len(blue_components) == 1, "expected one blue object"
    blue = blue_components[0]
    pts = _to_set(blue)
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    vertical_axis = (min(cols) + max(cols)) / 2.0
    horizontal_axis = (min(rows) + max(rows)) / 2.0
    axes = []
    if all((r, int(round(2 * vertical_axis - c))) in pts for r, c in pts):
        axes.append(("vertical", vertical_axis))
    if all((int(round(2 * horizontal_axis - r)), c) in pts for r, c in pts):
        axes.append(("horizontal", horizontal_axis))
    assert len(axes) == 1, "blue object must have a unique horizontal or vertical symmetry axis"
    return axes[0][0], axes[0][1], blue


def _mirror_point(point: tuple[int, int], axis_kind: str, axis_value: float) -> tuple[int, int]:
    r, c = point
    if axis_kind == "vertical":
        return r, int(round(2 * axis_value - c))
    return int(round(2 * axis_value - r)), c


def _ray_cells(
    shape: tuple[int, int],
    real: tuple[int, int],
    virtual: tuple[int, int],
    axis_kind: str,
) -> set[tuple[int, int]]:
    h, w = shape
    rays: set[tuple[int, int]] = set()
    if axis_kind == "vertical":
        steps = [(0, -1), (0, 1)]
    else:
        steps = [(-1, 0), (1, 0)]
    for start in (real, virtual):
        for dr, dc in steps:
            r, c = start[0] + dr, start[1] + dc
            while 0 <= r < h and 0 <= c < w:
                rays.add((r, c))
                r += dr
                c += dc
    return rays


def _erase_hit_objects(
    grid: np.ndarray,
    ray: set[tuple[int, int]],
) -> tuple[np.ndarray, list[list[tuple[int, int]]], list[list[tuple[int, int]]]]:
    out = np.array(grid, dtype=int, copy=True)
    green_components = _find_components_of_color(out, GREEN)
    hit: list[list[tuple[int, int]]] = []
    miss: list[list[tuple[int, int]]] = []
    for comp in green_components:
        if any(cell in ray for cell in comp):
            hit.append(comp)
            for r, c in comp:
                out[r, c] = BG
        else:
            miss.append(comp)
    return out, hit, miss


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase every green object touched by rays cast from a real red marker and its axis-reflected virtual copy.

    Input structure:
    - The background is black.
    - There is exactly one blue 4-connected object with a unique mirror axis, either vertical or horizontal.
    - There is exactly one red marker pixel away from that axis, plus several separate green 4-connected target objects.

    Rule:
    - Detect the blue object's unique symmetry axis.
    - Reflect the red marker across that axis to obtain a virtual marker location that stays implicit.
    - Cast rays from both the real and virtual markers in the direction perpendicular to the axis, using both opposite directions.
    - Erase an entire green object if any one of its pixels lies on any of those rays; keep the blue object, red marker, untouched green objects, and background unchanged.

    Key insight:
    - The blue object is not transformed itself; its hidden axis is borrowed as a geometric parameter that determines both the virtual marker and the ray orientation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    axis_kind, axis_value, _blue = _detect_blue_axis(g)
    red_coords = np.argwhere(g == RED)
    assert red_coords.shape == (1, 2), "expected one red marker"
    red = tuple(map(int, red_coords[0]))
    virtual = _mirror_point(red, axis_kind, axis_value)
    h, w = g.shape
    vr, vc = virtual
    assert 0 <= vr < h and 0 <= vc < w
    ray = _ray_cells(g.shape, red, virtual, axis_kind)
    out, _hit, _miss = _erase_hit_objects(g, ray)
    return out


def _random_connected_shape(
    size: int,
    *,
    require_non_rect: bool = False,
    require_multirow: bool = False,
    require_multicol: bool = False,
) -> tuple[tuple[int, int], ...]:
    for _ in range(400):
        cells = {(0, 0)}
        frontier = set(_neighbors4(0, 0))
        while len(cells) < size:
            cell = random.choice(list(frontier))
            frontier.remove(cell)
            cells.add(cell)
            for nbr in _neighbors4(*cell):
                if nbr not in cells:
                    frontier.add(nbr)
        shape = _normalize(cells)
        rows = {r for r, _ in shape}
        cols = {c for _, c in shape}
        h, w = _shape_dims(shape)
        if require_non_rect and len(shape) == h * w:
            continue
        if require_multirow and len(rows) < 2:
            continue
        if require_multicol and len(cols) < 2:
            continue
        return shape
    raise RuntimeError("failed to generate connected shape")


def _generate_vertical_blue_patch() -> tuple[tuple[int, int], ...]:
    for _ in range(500):
        h = random.randint(4, 7)
        w = random.randint(5, 7)
        center_col = w // 2 if w % 2 == 1 else None
        near_axis_col = center_col if center_col is not None else (w // 2 - 1)
        reps = {(random.randrange(h), near_axis_col)}
        total = 1 if center_col is not None else 2
        while True:
            if 8 <= total <= 14 and len(reps) >= 3 and random.random() < 0.4:
                break
            frontier = set()
            for r, c in reps:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc <= near_axis_col and (nr, nc) not in reps:
                        frontier.add((nr, nc))
            if not frontier:
                break
            choice_pool = list(frontier)
            random.shuffle(choice_pool)
            placed = False
            for nr, nc in choice_pool:
                delta = 1 if center_col is not None and nc == center_col else 2
                if total + delta > 14:
                    continue
                reps.add((nr, nc))
                total += delta
                placed = True
                break
            if not placed:
                break
        axis2 = w - 1
        full = set()
        for r, c in reps:
            full.add((r, c))
            full.add((r, axis2 - c))
        if not (8 <= len(full) <= 14):
            continue
        if not _is_connected(full):
            continue
        if not _has_vertical_symmetry(full):
            continue
        if _has_horizontal_symmetry(full):
            continue
        shape = _normalize(full)
        rows = {r for r, _ in shape}
        cols = {c for _, c in shape}
        if len(rows) < 3 or len(cols) < 3:
            continue
        return shape
    raise RuntimeError("failed to generate singly vertical-symmetric blue patch")


def _generate_blue_patch(axis_kind: str) -> tuple[tuple[int, int], ...]:
    base = _generate_vertical_blue_patch()
    if axis_kind == "vertical":
        return base
    return _normalize((c, r) for r, c in base)


def _build_blocked(
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> np.ndarray:
    h, w = shape
    blocked = np.zeros((h, w), dtype=bool)
    for r, c in occupied:
        for dr, dc in DIR8 + [(0, 0)]:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w:
                blocked[rr, cc] = True
    for r, c in reserved:
        if 0 <= r < h and 0 <= c < w:
            blocked[r, c] = True
    return blocked


def _place_shape(
    shape: tuple[tuple[int, int], ...],
    board_shape: tuple[int, int],
    blocked: np.ndarray,
    predicate,
) -> list[tuple[int, int]] | None:
    h, w = board_shape
    sh, sw = _shape_dims(shape)
    options: list[list[tuple[int, int]]] = []
    for r0 in range(h - sh + 1):
        for c0 in range(w - sw + 1):
            placed = [(r0 + r, c0 + c) for r, c in shape]
            if any(blocked[r, c] for r, c in placed):
                continue
            if predicate(placed):
                options.append(placed)
    if not options:
        return None
    return random.choice(options)


def _hits_correct(
    cells: Iterable[tuple[int, int]],
    axis_kind: str,
    line_index: int,
) -> bool:
    if axis_kind == "vertical":
        return any(r == line_index for r, _ in cells)
    return any(c == line_index for _, c in cells)


def _hits_wrong(
    cells: Iterable[tuple[int, int]],
    axis_kind: str,
    wrong_indices: set[int],
) -> bool:
    if axis_kind == "vertical":
        return any(c in wrong_indices for _, c in cells)
    return any(r in wrong_indices for r, _ in cells)


def _distance_to_correct_line(
    cells: Iterable[tuple[int, int]],
    axis_kind: str,
    line_index: int,
) -> int:
    if axis_kind == "vertical":
        return min(abs(r - line_index) for r, _ in cells)
    return min(abs(c - line_index) for _, c in cells)


def _paint_cells(grid: np.ndarray, cells: Iterable[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _generate_example() -> np.ndarray | None:
    h = random.randint(13, 18)
    w = random.randint(13, 18)
    axis_kind = random.choice(["vertical", "horizontal"])
    blue_shape = _generate_blue_patch(axis_kind)
    bh, bw = _shape_dims(blue_shape)

    blue_positions = [(r, c) for r in range(1, h - bh) for c in range(1, w - bw)]
    random.shuffle(blue_positions)
    for blue_r0, blue_c0 in blue_positions[:80]:
        blue_cells = [(blue_r0 + r, blue_c0 + c) for r, c in blue_shape]
        blue_set = set(blue_cells)
        if axis_kind == "vertical":
            cols = [c for _, c in blue_cells]
            axis_value = (min(cols) + max(cols)) / 2.0
        else:
            rows = [r for r, _ in blue_cells]
            axis_value = (min(rows) + max(rows)) / 2.0

        red_candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for rr in range(2, h - 2):
            for cc in range(2, w - 2):
                red = (rr, cc)
                if red in blue_set:
                    continue
                if any(abs(rr - br) <= 1 and abs(cc - bc) <= 1 for br, bc in blue_set):
                    continue
                if axis_kind == "vertical" and abs(cc - axis_value) < 0.5:
                    continue
                if axis_kind == "horizontal" and abs(rr - axis_value) < 0.5:
                    continue
                virtual = _mirror_point(red, axis_kind, axis_value)
                vr, vc = virtual
                if not (0 <= vr < h and 0 <= vc < w):
                    continue
                if virtual in blue_set or red == virtual:
                    continue
                red_candidates.append((red, virtual))
        random.shuffle(red_candidates)

        for red, virtual in red_candidates[:60]:
            occupied = set(blue_cells) | {red}
            reserved = {virtual}
            blocked = _build_blocked((h, w), occupied, reserved)

            correct_line = red[0] if axis_kind == "vertical" else red[1]
            wrong_indices = {red[1], virtual[1]} if axis_kind == "vertical" else {red[0], virtual[0]}
            used_sigs: set[tuple[tuple[int, int], ...]] = set()
            green_objects: list[list[tuple[int, int]]] = []

            hit_shape = _random_connected_shape(
                random.randint(3, 6),
                require_non_rect=True,
                require_multirow=(axis_kind == "vertical"),
                require_multicol=(axis_kind == "horizontal"),
            )
            used_sigs.add(_canon_shape(hit_shape))
            hit_cells = _place_shape(
                hit_shape,
                (h, w),
                blocked,
                lambda cells: _hits_correct(cells, axis_kind, correct_line)
                and not _hits_wrong(cells, axis_kind, wrong_indices)
                and _distance_to_correct_line(cells, axis_kind, correct_line) == 0,
            )
            if hit_cells is None:
                continue
            green_objects.append(hit_cells)
            occupied |= set(hit_cells)
            blocked = _build_blocked((h, w), occupied, reserved)

            wrong_shape = _random_connected_shape(random.randint(3, 6), require_non_rect=True)
            used_sigs.add(_canon_shape(wrong_shape))
            wrong_cells = _place_shape(
                wrong_shape,
                (h, w),
                blocked,
                lambda cells: _hits_wrong(cells, axis_kind, wrong_indices)
                and not _hits_correct(cells, axis_kind, correct_line),
            )
            if wrong_cells is None:
                continue
            green_objects.append(wrong_cells)
            occupied |= set(wrong_cells)
            blocked = _build_blocked((h, w), occupied, reserved)

            near_shape = None
            for _ in range(50):
                candidate = _random_connected_shape(random.randint(3, 6), require_non_rect=True)
                sig = _canon_shape(candidate)
                if sig not in used_sigs:
                    near_shape = candidate
                    used_sigs.add(sig)
                    break
            if near_shape is None:
                continue
            near_cells = _place_shape(
                near_shape,
                (h, w),
                blocked,
                lambda cells: not _hits_correct(cells, axis_kind, correct_line)
                and not _hits_wrong(cells, axis_kind, wrong_indices)
                and _distance_to_correct_line(cells, axis_kind, correct_line) == 1,
            )
            if near_cells is None:
                continue
            green_objects.append(near_cells)
            occupied |= set(near_cells)
            blocked = _build_blocked((h, w), occupied, reserved)

            extra_count = random.randint(0, 2)
            ok = True
            for _ in range(extra_count):
                placed = False
                for _ in range(80):
                    shape = _random_connected_shape(
                        random.randint(3, 6),
                        require_non_rect=True,
                        require_multirow=(axis_kind == "vertical" and random.random() < 0.5),
                        require_multicol=(axis_kind == "horizontal" and random.random() < 0.5),
                    )
                    sig = _canon_shape(shape)
                    if sig in used_sigs:
                        continue
                    if random.random() < 0.5:
                        pred = lambda cells: _hits_correct(cells, axis_kind, correct_line)
                    else:
                        pred = lambda cells: (
                            not _hits_correct(cells, axis_kind, correct_line)
                            and _distance_to_correct_line(cells, axis_kind, correct_line) >= 2
                        )
                    cells = _place_shape(shape, (h, w), blocked, pred)
                    if cells is None:
                        continue
                    used_sigs.add(sig)
                    green_objects.append(cells)
                    occupied |= set(cells)
                    blocked = _build_blocked((h, w), occupied, reserved)
                    placed = True
                    break
                if not placed:
                    ok = False
                    break
            if not ok:
                continue

            grid = np.zeros((h, w), dtype=int)
            _paint_cells(grid, blue_cells, BLUE)
            grid[red] = RED
            for obj in green_objects:
                _paint_cells(grid, obj, GREEN)

            if _validate_example(grid):
                return grid
    return None


def _wrong_direction_output(
    grid: np.ndarray,
    red: tuple[int, int],
    virtual: tuple[int, int],
    axis_kind: str,
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    green_components = _find_components_of_color(out, GREEN)
    if axis_kind == "vertical":
        wrong = {(r, red[1]) for r in range(out.shape[0])} | {(r, virtual[1]) for r in range(out.shape[0])}
    else:
        wrong = {(red[0], c) for c in range(out.shape[1])} | {(virtual[0], c) for c in range(out.shape[1])}
    for comp in green_components:
        if any(cell in wrong for cell in comp):
            for r, c in comp:
                out[r, c] = BG
    return out


def _line_only_output(grid: np.ndarray, ray: set[tuple[int, int]]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for r, c in ray:
        if out[r, c] == GREEN:
            out[r, c] = BG
    return out


def _validate_example(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    if max(grid.shape) > 30:
        return False
    if np.count_nonzero(grid == RED) != 1:
        return False
    blue_comps = _find_components_of_color(grid, BLUE)
    if len(blue_comps) != 1 or not (8 <= len(blue_comps[0]) <= 14):
        return False
    green_comps = _find_components_of_color(grid, GREEN)
    if len(green_comps) < 3:
        return False
    if any(not (3 <= len(comp) <= 6) for comp in green_comps):
        return False

    try:
        axis_kind, axis_value, _blue = _detect_blue_axis(grid)
    except AssertionError:
        return False
    red = tuple(map(int, np.argwhere(grid == RED)[0]))
    virtual = _mirror_point(red, axis_kind, axis_value)
    vr, vc = virtual
    if not (0 <= vr < grid.shape[0] and 0 <= vc < grid.shape[1]):
        return False
    if grid[virtual] != BG:
        return False

    ray = _ray_cells(grid.shape, red, virtual, axis_kind)
    out, hit, miss = _erase_hit_objects(grid, ray)
    if not hit or not miss:
        return False
    if not any(any(cell not in ray for cell in comp) for comp in hit):
        return False
    correct_line = red[0] if axis_kind == "vertical" else red[1]
    if not any(_distance_to_correct_line(comp, axis_kind, correct_line) == 1 for comp in miss):
        return False
    if np.array_equal(out, grid):
        return False
    if not np.array_equal(main(grid), out):
        return False
    if np.array_equal(_line_only_output(grid, ray), out):
        return False
    if np.array_equal(_wrong_direction_output(grid, red, virtual, axis_kind), out):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_example()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid borrowed-axis ray-erase puzzle")
