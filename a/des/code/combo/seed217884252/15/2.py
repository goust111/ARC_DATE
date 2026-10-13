from __future__ import annotations

import numpy as np


DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
ALL_DIRS = ("up", "down", "left", "right")
RNG = np.random.default_rng()
MAX_ATTEMPTS = 400
GRID_H_RANGE = (18, 27)
GRID_W_RANGE = (18, 27)
SHAPE_MAX = 7


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr = int(rr) + dr
                    nc = int(cc) + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [r0 for r0, _ in coords]
            cols = [c0 for _, c0 in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    comps.sort(key=lambda comp: tuple(int(x) for x in comp["bbox"]))
    return comps


def _component_mask(comp: dict[str, object]) -> tuple[np.ndarray, int, int]:
    coords = comp["coords"]
    rows = [int(r) for r, _ in coords]
    cols = [int(c) for _, c in coords]
    r0 = min(rows)
    c0 = min(cols)
    h = max(rows) - r0 + 1
    w = max(cols) - c0 + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        mask[int(r) - r0, int(c) - c0] = True
    return mask, int(r0), int(c0)


def _is_connected_mask(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    start = (int(pts[0, 0]), int(pts[0, 1]))
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in DIRS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _slice_signatures(mask: np.ndarray, axis: str) -> list[tuple[int, int]]:
    sigs: list[tuple[int, int]] = []
    if axis == "vertical":
        for r in range(mask.shape[0]):
            cols = np.flatnonzero(mask[r])
            if cols.size == 0:
                sigs.append((0, 0))
            else:
                sigs.append((int(cols[-1] - cols[0] + 1), int(cols.size)))
    else:
        for c in range(mask.shape[1]):
            rows = np.flatnonzero(mask[:, c])
            if rows.size == 0:
                sigs.append((0, 0))
            else:
                sigs.append((int(rows[-1] - rows[0] + 1), int(rows.size)))
    return sigs


def _narrow_end(signatures: list[tuple[int, int]]) -> int:
    for i in range(len(signatures)):
        a = signatures[i]
        b = signatures[-1 - i]
        if a != b:
            return -1 if a < b else 1
    raise AssertionError("expected a unique narrow end")


def _analyze_mask(mask: np.ndarray) -> tuple[str, int]:
    if not _is_connected_mask(mask):
        raise AssertionError("object must be 4-connected")
    vertical = bool(np.array_equal(mask, np.fliplr(mask)))
    horizontal = bool(np.array_equal(mask, np.flipud(mask)))
    if vertical == horizontal:
        raise AssertionError("object must have exactly one horizontal/vertical symmetry axis")
    axis = "vertical" if vertical else "horizontal"
    sign = _narrow_end(_slice_signatures(mask, axis))
    return axis, int(sign)


def _direction_from_mask(mask: np.ndarray) -> str:
    axis, sign = _analyze_mask(mask)
    if axis == "vertical":
        return "up" if sign < 0 else "down"
    return "left" if sign < 0 else "right"


def _delta_for_direction(direction: str, h: int, w: int) -> tuple[int, int]:
    if direction == "up":
        return (-h, 0)
    if direction == "down":
        return (h, 0)
    if direction == "left":
        return (0, -w)
    if direction == "right":
        return (0, w)
    raise AssertionError(f"unknown direction: {direction}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each foreground object toward its narrower end.

    Each object is monochrome, 4-connected, and has exactly one horizontal or
    vertical symmetry axis. The motion direction is perpendicular to that axis
    and points toward the lexicographically narrower end. The translation
    distance equals the object's bounding-box size along the motion direction.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)
    out = np.full((h, w), bg, dtype=int)

    for comp in _find_components(g, bg):
        mask, r0, c0 = _component_mask(comp)
        direction = _direction_from_mask(mask)
        dh, dw = _delta_for_direction(direction, *mask.shape)
        color = int(comp["color"])
        for r, c in comp["coords"]:
            nr = int(r) + dh
            nc = int(c) + dw
            if not (0 <= nr < h and 0 <= nc < w):
                raise AssertionError("moved object left the grid")
            if int(out[nr, nc]) != bg:
                raise AssertionError("moved objects must not overlap")
            out[nr, nc] = color
    return out


def _positive_composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [int(total)]
    cuts = np.sort(RNG.choice(np.arange(1, total), size=parts - 1, replace=False))
    prev = 0
    out: list[int] = []
    for cut in cuts:
        out.append(int(cut) - prev)
        prev = int(cut)
    out.append(int(total) - prev)
    return out


def _monotone_profile(length: int, low: int, high: int) -> list[int]:
    levels = list(range(int(low), int(high) + 1, 2))
    max_parts = min(len(levels), int(length))
    parts = int(RNG.integers(2, max_parts + 1))
    if parts == 2:
        chosen = [levels[0], levels[-1]]
    else:
        mids = sorted(int(x) for x in RNG.choice(levels[1:-1], size=parts - 2, replace=False))
        chosen = [levels[0], *mids, levels[-1]]
    runs = _positive_composition(int(length), len(chosen))
    profile: list[int] = []
    for value, run in zip(chosen, runs):
        profile.extend([int(value)] * int(run))
    return profile


def _vertical_mask(direction: str, max_h: int, max_w: int) -> np.ndarray:
    h = int(RNG.integers(3, max_h + 1))
    widths = [w for w in range(3, max_w + 1) if any(range(1 if w % 2 else 2, w, 2))]
    total_w = int(widths[int(RNG.integers(0, len(widths)))])
    tip_choices = list(range(1 if total_w % 2 else 2, total_w, 2))
    tip_w = int(tip_choices[int(RNG.integers(0, len(tip_choices)))])
    profile = _monotone_profile(h, tip_w, total_w)
    if direction == "down":
        profile = profile[::-1]
    mask = np.zeros((h, total_w), dtype=bool)
    for r, width in enumerate(profile):
        start = (total_w - int(width)) // 2
        mask[r, start : start + int(width)] = True
    return mask


def _horizontal_mask(direction: str, max_h: int, max_w: int) -> np.ndarray:
    w = int(RNG.integers(3, max_w + 1))
    heights = [h for h in range(3, max_h + 1) if any(range(1 if h % 2 else 2, h, 2))]
    total_h = int(heights[int(RNG.integers(0, len(heights)))])
    tip_choices = list(range(1 if total_h % 2 else 2, total_h, 2))
    tip_h = int(tip_choices[int(RNG.integers(0, len(tip_choices)))])
    profile = _monotone_profile(w, tip_h, total_h)
    if direction == "right":
        profile = profile[::-1]
    mask = np.zeros((total_h, w), dtype=bool)
    for c, height in enumerate(profile):
        start = (total_h - int(height)) // 2
        mask[start : start + int(height), c] = True
    return mask


def _make_shape(direction: str, grid_h: int, grid_w: int) -> np.ndarray:
    if direction in {"up", "down"}:
        max_h = min(SHAPE_MAX, grid_h // 2)
        max_w = min(SHAPE_MAX, grid_w - 4)
        if max_h < 3 or max_w < 3:
            raise ValueError("grid too small for vertical-moving object")
        mask = _vertical_mask(direction, max_h, max_w)
    else:
        max_h = min(SHAPE_MAX, grid_h - 4)
        max_w = min(SHAPE_MAX, grid_w // 2)
        if max_h < 3 or max_w < 3:
            raise ValueError("grid too small for horizontal-moving object")
        mask = _horizontal_mask(direction, max_h, max_w)
    if _direction_from_mask(mask) != direction:
        raise AssertionError("generator produced the wrong direction")
    return mask


def _union_bounds(r0: int, c0: int, h: int, w: int, direction: str) -> tuple[int, int, int, int]:
    dr, dc = _delta_for_direction(direction, h, w)
    nr0 = int(r0) + dr
    nc0 = int(c0) + dc
    return (
        min(int(r0), nr0),
        min(int(c0), nc0),
        max(int(r0) + h - 1, nr0 + h - 1),
        max(int(c0) + w - 1, nc0 + w - 1),
    )


def _candidate_positions(mask: np.ndarray, direction: str, reserved: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    grid_h, grid_w = reserved.shape
    candidates: list[tuple[int, int]] = []
    if direction == "up":
        row_range = range(h, grid_h - h + 1)
        col_range = range(0, grid_w - w + 1)
    elif direction == "down":
        row_range = range(0, grid_h - 2 * h + 1)
        col_range = range(0, grid_w - w + 1)
    elif direction == "left":
        row_range = range(0, grid_h - h + 1)
        col_range = range(w, grid_w - w + 1)
    else:
        row_range = range(0, grid_h - h + 1)
        col_range = range(0, grid_w - 2 * w + 1)

    for r0 in row_range:
        for c0 in col_range:
            ur0, uc0, ur1, uc1 = _union_bounds(int(r0), int(c0), h, w, direction)
            rr0 = max(0, ur0 - 1)
            cc0 = max(0, uc0 - 1)
            rr1 = min(grid_h - 1, ur1 + 1)
            cc1 = min(grid_w - 1, uc1 + 1)
            if not reserved[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                candidates.append((int(r0), int(c0)))
    return candidates


def _reserve(reserved: np.ndarray, r0: int, c0: int, h: int, w: int, direction: str) -> None:
    ur0, uc0, ur1, uc1 = _union_bounds(r0, c0, h, w, direction)
    rr0 = max(0, ur0 - 1)
    cc0 = max(0, uc0 - 1)
    rr1 = min(reserved.shape[0] - 1, ur1 + 1)
    cc1 = min(reserved.shape[1] - 1, uc1 + 1)
    reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    for r, c in np.argwhere(mask):
        grid[int(r0) + int(r), int(c0) + int(c)] = int(color)


def _sample_directions(count: int) -> list[str]:
    dirs: list[str] = []
    if count >= 4:
        dirs.extend([ALL_DIRS[i] for i in RNG.permutation(4)])
    else:
        dirs.append(("up", "down")[int(RNG.integers(0, 2))])
        dirs.append(("left", "right")[int(RNG.integers(0, 2))])
    while len(dirs) < count:
        dirs.append(ALL_DIRS[int(RNG.integers(0, 4))])
    order = RNG.permutation(len(dirs))
    return [dirs[int(i)] for i in order[:count]]


def _gen_attempt() -> np.ndarray | None:
    grid_h = int(RNG.integers(*GRID_H_RANGE))
    grid_w = int(RNG.integers(*GRID_W_RANGE))
    max_objects = 5 if min(grid_h, grid_w) >= 22 else 4
    obj_count = int(RNG.integers(2, max_objects + 1))
    directions = _sample_directions(obj_count)
    colors = [int(x) for x in RNG.choice(np.arange(1, 10), size=obj_count, replace=False)]

    grid = np.zeros((grid_h, grid_w), dtype=int)
    reserved = np.zeros((grid_h, grid_w), dtype=bool)
    placed_specs: list[tuple[int, str]] = []

    for color, direction in zip(colors, directions):
        placed = False
        for _ in range(80):
            mask = _make_shape(direction, grid_h, grid_w)
            candidates = _candidate_positions(mask, direction, reserved)
            if not candidates:
                continue
            r0, c0 = candidates[int(RNG.integers(0, len(candidates)))]
            _paint_mask(grid, mask, r0, c0, color)
            _reserve(reserved, r0, c0, mask.shape[0], mask.shape[1], direction)
            placed_specs.append((color, direction))
            placed = True
            break
        if not placed:
            return None

    if int(np.count_nonzero(grid)) * 2 >= grid.size:
        return None

    comps = _find_components(grid, 0)
    if len(comps) != obj_count:
        return None
    for comp in comps:
        mask, _, _ = _component_mask(comp)
        if _direction_from_mask(mask) != dict(placed_specs)[int(comp["color"])]:
            return None

    out = main(grid)
    if np.array_equal(grid, out):
        return None
    if len(_find_components(out, 0)) != obj_count:
        return None
    for color, _direction in placed_specs:
        if int(np.count_nonzero(grid == color)) != int(np.count_nonzero(out == color)):
            return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
