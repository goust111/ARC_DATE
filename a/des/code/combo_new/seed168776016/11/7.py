from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAROON = 9

SIDE_ORDER = ("top", "right", "bottom", "left")
SIDE_DIRS = {
    "top": (1, 0),
    "right": (0, -1),
    "bottom": (-1, 0),
    "left": (0, 1),
}
SIDE_COLORS = {
    "top": RED,
    "right": BLUE,
    "bottom": GREEN,
    "left": YELLOW,
}

BASE_SHAPES = [
    {(0, 0), (1, 0)},
    {(0, 0), (0, 1)},
    {(0, 0), (0, 1), (0, 2)},
    {(0, 0), (1, 0), (1, 1)},
    {(0, 1), (1, 0), (1, 1)},
    {(0, 0), (0, 1), (1, 0)},
    {(0, 0), (0, 1), (1, 1)},
    {(0, 0), (0, 1), (1, 0), (1, 1)},
    {(0, 0), (0, 1), (0, 2), (1, 1)},
    {(0, 1), (1, 0), (1, 1), (1, 2)},
]

GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _inner_bbox(frame_bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    fr0, fc0, fr1, fc1 = frame_bbox
    return fr0 + 1, fc0 + 1, fr1 - 1, fc1 - 1


def _find_frame(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    best: tuple[int, tuple[int, int, int, int], int] | None = None
    for color in np.unique(grid):
        color = int(color)
        if color == int(bg):
            continue
        for comp in _components_of_color(grid, color):
            r0, c0, r1, c1 = _bbox(comp)
            height = r1 - r0 + 1
            width = c1 - c0 + 1
            perimeter = 2 * (height + width) - 4
            if height < 3 or width < 3 or int(comp.shape[0]) != perimeter:
                continue
            border_ok = True
            for rr in range(r0, r1 + 1):
                for cc in range(c0, c1 + 1):
                    on_border = rr in (r0, r1) or cc in (c0, c1)
                    if on_border and int(grid[rr, cc]) != color:
                        border_ok = False
                        break
                    if (not on_border) and int(grid[rr, cc]) == color:
                        border_ok = False
                        break
                if not border_ok:
                    break
            if border_ok:
                key = (perimeter, (r1 - r0) * (c1 - c0), color)
                if best is None or key > (best[2], (best[1][2] - best[1][0]) * (best[1][3] - best[1][1]), best[0]):
                    best = (color, (r0, c0, r1, c1), perimeter)
    assert best is not None, "expected a unique rectangular frame"
    return best[0], best[1]


def _side_band_color(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    side: str,
    bg: int,
    frame_color: int,
) -> int:
    fr0, fc0, fr1, fc1 = frame_bbox
    if side == "top":
        vals = [int(grid[fr0 - 1, c]) for c in range(fc0, fc1 + 1)]
    elif side == "right":
        vals = [int(grid[r, fc1 + 1]) for r in range(fr0, fr1 + 1)]
    elif side == "bottom":
        vals = [int(grid[fr1 + 1, c]) for c in range(fc0, fc1 + 1)]
    else:
        vals = [int(grid[r, fc0 - 1]) for r in range(fr0, fr1 + 1)]
    cnt = Counter(v for v in vals if v not in {int(bg), int(frame_color)})
    assert cnt, f"missing {side} band"
    return int(max(cnt.items(), key=lambda kv: kv[1])[0])


def _is_band_component(
    coords: np.ndarray,
    side: str,
    frame_bbox: tuple[int, int, int, int],
) -> bool:
    fr0, fc0, fr1, fc1 = frame_bbox
    if side == "top":
        return any(int(r) == fr0 - 1 and fc0 <= int(c) <= fc1 for r, c in coords)
    if side == "right":
        return any(int(c) == fc1 + 1 and fr0 <= int(r) <= fr1 for r, c in coords)
    if side == "bottom":
        return any(int(r) == fr1 + 1 and fc0 <= int(c) <= fc1 for r, c in coords)
    return any(int(c) == fc0 - 1 and fr0 <= int(r) <= fr1 for r, c in coords)


def _distance_to_side(coords: np.ndarray, side: str, frame_bbox: tuple[int, int, int, int]) -> int:
    fr0, fc0, fr1, fc1 = frame_bbox
    r0, c0, r1, c1 = _bbox(coords)
    if side == "top":
        return fr0 - r1 - 1
    if side == "right":
        return c0 - fc1 - 1
    if side == "bottom":
        return r0 - fr1 - 1
    return fc0 - c1 - 1


def _shift_coords(coords: np.ndarray, dr: int, dc: int) -> np.ndarray:
    return coords + np.array([dr, dc], dtype=int)


def _flush_inside(coords: np.ndarray, side: str, frame_bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    ir0, ic0, ir1, ic1 = _inner_bbox(frame_bbox)
    if side == "top":
        return _shift_coords(coords, ir0 - r0, 0)
    if side == "right":
        return _shift_coords(coords, 0, ic1 - c1)
    if side == "bottom":
        return _shift_coords(coords, ir1 - r1, 0)
    return _shift_coords(coords, 0, ic0 - c0)


def _slide_component(
    coords: np.ndarray,
    side: str,
    frame_bbox: tuple[int, int, int, int],
    occupied: set[tuple[int, int]],
    fixed_inside: set[tuple[int, int]],
) -> tuple[np.ndarray, str]:
    ir0, ic0, ir1, ic1 = _inner_bbox(frame_bbox)
    dr, dc = SIDE_DIRS[side]
    current = coords.copy()
    while True:
        moved = _shift_coords(current, dr, dc)
        for rr, cc in moved:
            if int(rr) < ir0 or int(rr) > ir1 or int(cc) < ic0 or int(cc) > ic1:
                return current, "wall"
        hit = [(int(rr), int(cc)) for rr, cc in moved if (int(rr), int(cc)) in occupied]
        if hit:
            reason = "obstacle" if any(cell in fixed_inside for cell in hit) else "stack"
            return current, reason
        current = moved


def _simulate_movers(
    base_grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    movers: list[dict],
    fixed_inside: set[tuple[int, int]],
    bg: int,
) -> tuple[np.ndarray, list[dict]]:
    out = np.array(base_grid, dtype=int, copy=True)
    for mover in movers:
        out[mover["coords"][:, 0], mover["coords"][:, 1]] = int(bg)
    occupied = set(fixed_inside)
    results: list[dict] = []
    for mover in sorted(movers, key=lambda item: (item["distance"], SIDE_ORDER.index(item["side"]), item["order"])):
        entered = _flush_inside(mover["coords"], mover["side"], frame_bbox)
        final, reason = _slide_component(entered, mover["side"], frame_bbox, occupied, fixed_inside)
        for rr, cc in final:
            out[int(rr), int(cc)] = int(mover["color"])
            occupied.add((int(rr), int(cc)))
        results.append(
            {
                "order": mover["order"],
                "side": mover["side"],
                "color": mover["color"],
                "final": final,
                "stop": reason,
            }
        )
    return out, results


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each outside color block through the matching side of the central frame, then slide it inward until it stacks or hits a stopper.

    Input structure:
    - The background is a mostly black grid containing one large grey 1-pixel-thick rectangular frame with a black interior.
    - Short color bands sit immediately outside the four frame sides; each band identifies one entry side for blocks of the same color.
    - Several small 4-connected monochrome blocks lie outside the frame, and the frame interior may also contain fixed maroon obstacle pixels.

    Rule:
    - Identify the frame and read the four outside color bands to map each block color to its entry side.
    - For every outside block, translate it just inside the matching side while preserving its projection on the parallel axis.
    - Process blocks from smallest side-distance to largest; each block then slides inward perpendicular to its entry side until the next step would hit a fixed interior obstacle, a previously stopped block, or the opposite inner wall.

    Key insight:
    - The short side-adjacent color bands are not movable objects: they are a visible color-to-side legend that controls both where each block enters and the direction in which it must slide and stack.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frame_color, frame_bbox = _find_frame(g, bg)
    side_by_color = {
        _side_band_color(g, frame_bbox, side, bg, frame_color): side
        for side in SIDE_ORDER
    }

    movers: list[dict] = []
    order = 0
    for color, side in side_by_color.items():
        for comp in _components_of_color(g, color):
            if _is_band_component(comp, side, frame_bbox):
                continue
            distance = _distance_to_side(comp, side, frame_bbox)
            if distance < 1:
                continue
            movers.append(
                {
                    "coords": comp,
                    "color": color,
                    "side": side,
                    "distance": int(distance),
                    "order": order,
                }
            )
            order += 1

    fixed_inside = set()
    ir0, ic0, ir1, ic1 = _inner_bbox(frame_bbox)
    for r in range(ir0, ir1 + 1):
        for c in range(ic0, ic1 + 1):
            if int(g[r, c]) != int(bg):
                fixed_inside.add((r, c))

    out, _ = _simulate_movers(g, frame_bbox, movers, fixed_inside, bg)
    return out


def _normalize_shape(cells: set[tuple[int, int]]) -> np.ndarray:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    coords = sorted((r - r0, c - c0) for r, c in cells)
    return np.array(coords, dtype=int)


def _rotate_shape(coords: np.ndarray) -> np.ndarray:
    max_c = int(coords[:, 1].max())
    rotated = {(int(c), max_c - int(r)) for r, c in coords}
    return _normalize_shape(rotated)


def _shape_for_side(side: str) -> np.ndarray:
    coords = _normalize_shape(set(random.choice(BASE_SHAPES)))
    if side in {"left", "right"}:
        coords = _rotate_shape(coords)
    return coords


def _outside_coords(
    shape: np.ndarray,
    side: str,
    frame_bbox: tuple[int, int, int, int],
    distance: int,
    ortho_start: int,
    grid_shape: tuple[int, int],
) -> np.ndarray | None:
    h, w = grid_shape
    _ = (h, w)
    _r0, _c0, r1, c1 = _bbox(shape)
    fr0, fc0, fr1, fc1 = frame_bbox
    if side == "top":
        dr = fr0 - 1 - distance - r1
        dc = ortho_start
    elif side == "right":
        dr = ortho_start
        dc = fc1 + 1 + distance
    elif side == "bottom":
        dr = fr1 + 1 + distance
        dc = ortho_start
    else:
        dr = ortho_start
        dc = fc0 - 1 - distance - c1
    placed = _shift_coords(shape, dr, dc)
    if placed[:, 0].min() < 0 or placed[:, 1].min() < 0:
        return None
    if placed[:, 0].max() >= grid_shape[0] or placed[:, 1].max() >= grid_shape[1]:
        return None
    return placed


def _component_cells_clear(cells: np.ndarray, occupied: set[tuple[int, int]]) -> bool:
    cell_set = {(int(r), int(c)) for r, c in cells}
    for r, c in cell_set:
        if (r, c) in occupied:
            return False
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in occupied and (nr, nc) not in cell_set:
                return False
    return True


def _paint_cells(grid: np.ndarray, coords: np.ndarray, color: int) -> None:
    grid[coords[:, 0], coords[:, 1]] = int(color)


def _reserve_band_cells(frame_bbox: tuple[int, int, int, int], bands: dict[str, np.ndarray]) -> set[tuple[int, int]]:
    reserved: set[tuple[int, int]] = set()
    fr0, fc0, fr1, fc1 = frame_bbox
    for side, segment in bands.items():
        if side == "top":
            for c in segment:
                reserved.add((fr0 - 1, int(c)))
        elif side == "right":
            for r in segment:
                reserved.add((int(r), fc1 + 1))
        elif side == "bottom":
            for c in segment:
                reserved.add((fr1 + 1, int(c)))
        else:
            for r in segment:
                reserved.add((int(r), fc0 - 1))
    return reserved


def _sample_bands(frame_bbox: tuple[int, int, int, int]) -> dict[str, np.ndarray]:
    fr0, fc0, fr1, fc1 = frame_bbox
    bands: dict[str, np.ndarray] = {}
    top_cols = list(range(fc0 + 1, fc1))
    left_rows = list(range(fr0 + 1, fr1))
    for side, line in (("top", top_cols), ("bottom", top_cols), ("left", left_rows), ("right", left_rows)):
        length = random.randint(2, 3)
        start = random.randint(0, len(line) - length)
        bands[side] = np.array(line[start : start + length], dtype=int)
    return bands


def _sample_frame_bbox(h: int, w: int) -> tuple[int, int, int, int] | None:
    for _ in range(80):
        top = random.randint(4, 7)
        bottom = random.randint(4, 7)
        left = random.randint(4, 7)
        right = random.randint(4, 7)
        fr0 = top
        fr1 = h - bottom - 1
        fc0 = left
        fc1 = w - right - 1
        if fr1 - fr0 + 1 >= 10 and fc1 - fc0 + 1 >= 10:
            return fr0, fc0, fr1, fc1
    return None


def _plan_side_counts(mode: str) -> tuple[dict[str, int], str | None]:
    active_count = random.randint(2, 4)
    active_sides = random.sample(list(SIDE_ORDER), active_count)
    counts = {side: 0 for side in SIDE_ORDER}
    for side in active_sides:
        counts[side] = 1

    stack_side = None
    if mode in {"stack", "both"}:
        stack_side = random.choice(active_sides)
        counts[stack_side] = 2

    candidates = [side for side in active_sides if counts[side] < 2]
    extra_count = random.randint(0, min(2, len(candidates)))
    for side in random.sample(candidates, extra_count):
        counts[side] += 1

    while sum(counts.values()) < 3:
        side = random.choice([s for s in active_sides if counts[s] < 2])
        counts[side] += 1
        if stack_side is None and mode in {"stack", "both"}:
            stack_side = side

    if mode in {"stack", "both"} and stack_side is None:
        stack_side = random.choice([s for s in active_sides if counts[s] == 2])
    return counts, stack_side


def _max_distance(shape: np.ndarray, side: str, frame_bbox: tuple[int, int, int, int], grid_shape: tuple[int, int]) -> int:
    h, w = grid_shape
    _r0, _c0, r1, c1 = _bbox(shape)
    fr0, fc0, fr1, fc1 = frame_bbox
    if side == "top":
        return fr0 - 1 - r1
    if side == "right":
        return w - fc1 - 2 - c1
    if side == "bottom":
        return h - fr1 - 2 - r1
    return fc0 - 1 - c1


def _ortho_range(shape: np.ndarray, side: str, frame_bbox: tuple[int, int, int, int]) -> list[int]:
    _r0, _c0, r1, c1 = _bbox(shape)
    ir0, ic0, ir1, ic1 = _inner_bbox(frame_bbox)
    if side in {"top", "bottom"}:
        return list(range(ic0, ic1 - c1 + 1))
    return list(range(ir0, ir1 - r1 + 1))


def _build_movers(
    frame_bbox: tuple[int, int, int, int],
    bands: dict[str, np.ndarray],
    counts: dict[str, int],
    stack_side: str | None,
    grid_shape: tuple[int, int],
) -> list[dict] | None:
    reserved = _reserve_band_cells(frame_bbox, bands)
    movers: list[dict] = []
    order = 0
    stack_meta: dict[str, tuple[np.ndarray, int, int]] = {}
    sides = list(SIDE_ORDER)
    random.shuffle(sides)

    for side in sides:
        for idx in range(counts[side]):
            placed = None
            for _ in range(120):
                if side == stack_side and idx == 1 and side in stack_meta:
                    shape, ortho_start, first_distance = stack_meta[side]
                    max_d = _max_distance(shape, side, frame_bbox, grid_shape)
                    distances = list(range(1, max_d + 1))
                    random.shuffle(distances)
                    distances = [d for d in distances if d != first_distance]
                    if not distances:
                        continue
                    distance = distances[0]
                else:
                    shape = _shape_for_side(side)
                    max_d = _max_distance(shape, side, frame_bbox, grid_shape)
                    if max_d < 1:
                        continue
                    ortho_choices = _ortho_range(shape, side, frame_bbox)
                    if not ortho_choices:
                        continue
                    ortho_start = random.choice(ortho_choices)
                    distance = random.randint(1, max_d)
                coords = _outside_coords(shape, side, frame_bbox, distance, ortho_start, grid_shape)
                if coords is None or not _component_cells_clear(coords, reserved):
                    continue
                placed = {
                    "coords": coords,
                    "color": SIDE_COLORS[side],
                    "side": side,
                    "distance": distance,
                    "order": order,
                }
                for cell in coords:
                    reserved.add((int(cell[0]), int(cell[1])))
                if side == stack_side and idx == 0:
                    stack_meta[side] = (shape, ortho_start, distance)
                break
            if placed is None:
                return None
            movers.append(placed)
            order += 1
    return movers


def _front_cells(coords: np.ndarray, side: str) -> list[tuple[int, int]]:
    if side == "top":
        bound = int(coords[:, 0].max())
        return [(int(r), int(c)) for r, c in coords if int(r) == bound]
    if side == "right":
        bound = int(coords[:, 1].min())
        return [(int(r), int(c)) for r, c in coords if int(c) == bound]
    if side == "bottom":
        bound = int(coords[:, 0].min())
        return [(int(r), int(c)) for r, c in coords if int(r) == bound]
    bound = int(coords[:, 1].max())
    return [(int(r), int(c)) for r, c in coords if int(c) == bound]


def _target_obstacle(
    mover: dict,
    final_coords: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
) -> tuple[int, int] | None:
    entered = _flush_inside(mover["coords"], mover["side"], frame_bbox)
    side = mover["side"]
    if side in {"top", "bottom"}:
        start = int(max(r for r, _ in _front_cells(entered, side)) if side == "top" else min(r for r, _ in _front_cells(entered, side)))
        finish = int(max(r for r, _ in _front_cells(final_coords, side)) if side == "top" else min(r for r, _ in _front_cells(final_coords, side)))
        travel = abs(finish - start)
        if travel < 2:
            return None
        step = random.randint(2, travel)
        r, c = random.choice(_front_cells(entered, side))
        return (r + step, c) if side == "top" else (r - step, c)
    start = int(min(c for _, c in _front_cells(entered, side)) if side == "right" else max(c for _, c in _front_cells(entered, side)))
    finish = int(min(c for _, c in _front_cells(final_coords, side)) if side == "right" else max(c for _, c in _front_cells(final_coords, side)))
    travel = abs(finish - start)
    if travel < 2:
        return None
    step = random.randint(2, travel)
    r, c = random.choice(_front_cells(entered, side))
    return (r, c - step) if side == "right" else (r, c + step)


def _frame_and_bands_grid(grid_shape: tuple[int, int], frame_bbox: tuple[int, int, int, int], bands: dict[str, np.ndarray]) -> np.ndarray:
    h, w = grid_shape
    grid = np.full((h, w), BG, dtype=int)
    fr0, fc0, fr1, fc1 = frame_bbox
    grid[fr0 : fr1 + 1, fc0] = GREY
    grid[fr0 : fr1 + 1, fc1] = GREY
    grid[fr0, fc0 : fc1 + 1] = GREY
    grid[fr1, fc0 : fc1 + 1] = GREY
    for side, color in SIDE_COLORS.items():
        if side == "top":
            grid[fr0 - 1, bands[side]] = color
        elif side == "right":
            grid[bands[side], fc1 + 1] = color
        elif side == "bottom":
            grid[fr1 + 1, bands[side]] = color
        else:
            grid[bands[side], fc0 - 1] = color
    return grid


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(22, 28)
    w = random.randint(22, 28)
    frame_bbox = _sample_frame_bbox(h, w)
    if frame_bbox is None:
        return None
    bands = _sample_bands(frame_bbox)
    mode = random.choice(["plain", "obstacle", "stack", "both"])
    counts, stack_side = _plan_side_counts(mode)
    movers = _build_movers(frame_bbox, bands, counts, stack_side, (h, w))
    if movers is None:
        return None

    base = _frame_and_bands_grid((h, w), frame_bbox, bands)
    for mover in movers:
        _paint_cells(base, mover["coords"], mover["color"])

    fixed_inside: set[tuple[int, int]] = set()
    if mode in {"obstacle", "both"}:
        out0, results0 = _simulate_movers(base, frame_bbox, movers, fixed_inside, BG)
        prefer = [res for res in results0 if not (stack_side is not None and res["side"] == stack_side)]
        candidates = prefer or results0
        random.shuffle(candidates)
        obstacle = None
        for res in candidates:
            mover = next(item for item in movers if item["order"] == res["order"])
            obstacle = _target_obstacle(mover, res["final"], frame_bbox)
            if obstacle is not None:
                break
        if obstacle is None:
            return None
        fixed_inside.add(obstacle)
        base[obstacle] = MAROON
        if random.random() < 0.4:
            ir0, ic0, ir1, ic1 = _inner_bbox(frame_bbox)
            extras = [
                (r, c)
                for r in range(ir0 + 1, ir1)
                for c in range(ic0 + 1, ic1)
                if (r, c) not in fixed_inside
            ]
            if extras:
                extra = random.choice(extras)
                fixed_inside.add(extra)
                base[extra] = MAROON

    expected_out, results = _simulate_movers(base, frame_bbox, movers, fixed_inside, BG)
    if mode in {"obstacle", "both"} and not any(res["stop"] == "obstacle" for res in results):
        return None
    if mode in {"stack", "both"} and not any(res["stop"] == "stack" for res in results):
        return None

    moved_pixels = sum(int(mover["coords"].shape[0]) for mover in movers)
    ir0, ic0, ir1, ic1 = _inner_bbox(frame_bbox)
    interior_area = (ir1 - ir0 + 1) * (ic1 - ic0 + 1)
    if moved_pixels + len(fixed_inside) > int(interior_area * 0.35):
        return None
    if not np.any(base != expected_out):
        return None
    if not np.array_equal(main(base), expected_out):
        return None
    return base


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for task 7")
