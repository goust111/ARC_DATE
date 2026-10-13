from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    counts = Counter(int(v) for v in grid.ravel())
    return int(counts.most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(int(rr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return int(min(rows)), int(min(cols)), int(max(rows)), int(max(cols))


def _parse_shape(lines: list[str]) -> np.ndarray:
    return np.array([[ch == "#" for ch in row] for row in lines], dtype=bool)


_BASE_SHAPES = [
    _parse_shape(["#"]),
    _parse_shape(["###"]),
    _parse_shape(["#", "#", "#"]),
    _parse_shape(["###", "###", "###"]),
    _parse_shape([".#.", "###", ".#."]),
    _parse_shape(["###", ".#.", ".#."]),
    _parse_shape(["#..", "###", "..."]),
    _parse_shape(["#####", "..#..", "..#.."]),
    _parse_shape(["..#..", "..#..", "#####"]),
    _parse_shape([".#...", "#####", "...#."]),
]


def _shape_variants() -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[bytes] = set()
    for base in _BASE_SHAPES:
        for k in range(4):
            rotated = np.rot90(base, k)
            key = rotated.tobytes() + bytes(rotated.shape)
            if key in seen:
                continue
            seen.add(key)
            variants.append(rotated.copy())
    return variants


_SHAPES = _shape_variants()


def _find_gradient_bar(grid: np.ndarray, bg: int) -> tuple[list[tuple[int, int]], str, str]:
    h, w = grid.shape
    non_bg = grid != int(bg)
    candidates: list[tuple[list[tuple[int, int]], str, str]] = []
    for comp in _components(non_bg):
        if len(comp) < 2:
            continue
        rows = {r for r, _ in comp}
        cols = {c for _, c in comp}
        colors = {int(grid[r, c]) for r, c in comp}
        if len(colors) < 2:
            continue
        if len(rows) == 1:
            row = next(iter(rows))
            if row not in (0, h - 1):
                continue
            cvals = sorted(cols)
            if cvals != list(range(cvals[0], cvals[-1] + 1)):
                continue
            side = "top" if row == 0 else "bottom"
            candidates.append((comp, "horizontal", side))
        elif len(cols) == 1:
            col = next(iter(cols))
            if col not in (0, w - 1):
                continue
            rvals = sorted(rows)
            if rvals != list(range(rvals[0], rvals[-1] + 1)):
                continue
            side = "left" if col == 0 else "right"
            candidates.append((comp, "vertical", side))
    assert len(candidates) == 1, "expected exactly one multicolor edge bar"
    return candidates[0]


def _bar_sequence(grid: np.ndarray, coords: list[tuple[int, int]], orientation: str) -> list[int]:
    if orientation == "horizontal":
        ordered = sorted(coords, key=lambda rc: (int(rc[1]), int(rc[0])))
    else:
        ordered = sorted(coords, key=lambda rc: (int(rc[0]), int(rc[1])))
    return [int(grid[r, c]) for r, c in ordered]


def _object_distance(bbox: tuple[int, int, int, int], side: str, shape: tuple[int, int]) -> int:
    h, w = shape
    r0, c0, r1, c1 = bbox
    center_r = (int(r0) + int(r1)) // 2
    center_c = (int(c0) + int(c1)) // 2
    if side == "top":
        return int(center_r - 1)
    if side == "bottom":
        return int(h - 2 - center_r)
    if side == "left":
        return int(center_c - 1)
    assert side == "right"
    return int(w - 2 - center_c)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor uniform grey interior objects by looking up a border gradient strip.

    The strip is the unique multicolor 1-cell-thick segment on the outer border.
    Its colors are read in visual order: left-to-right for horizontal strips and
    top-to-bottom for vertical strips. Each grey 4-connected object is recolored
    by the strip color indexed by its perpendicular distance from the strip's
    inner-facing side, using modulo when the distance exceeds the strip length.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)

    bar_coords, orientation, side = _find_gradient_bar(g, bg)
    bar_set = set(bar_coords)
    sequence = _bar_sequence(g, bar_coords, orientation)
    assert sequence, "gradient strip must be non-empty"

    remaining = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if int(g[r, c]) != int(bg) and (int(r), int(c)) not in bar_set:
                remaining[r, c] = True

    object_colors = {int(g[r, c]) for r in range(h) for c in range(w) if remaining[r, c]}
    assert len(object_colors) == 1, "expected one interior object color"
    obj_color = next(iter(object_colors))

    object_mask = remaining & (g == int(obj_color))
    comps = _components(object_mask)
    assert comps, "expected at least one interior object"

    out = g.copy()
    for comp in comps:
        box = _bbox(comp)
        dist = _object_distance(box, side, g.shape)
        color = int(sequence[dist % len(sequence)])
        for r, c in comp:
            out[r, c] = color
    return out


def _candidate_distance_bounds(side: str, shape: np.ndarray, grid_shape: tuple[int, int]) -> tuple[int, int]:
    h, w = grid_shape
    sh, sw = shape.shape
    if side == "top":
        return sh // 2, h - 3 - sh // 2
    if side == "bottom":
        return sh // 2, h - 3 - sh // 2
    if side == "left":
        return sw // 2, w - 3 - sw // 2
    assert side == "right"
    return sw // 2, w - 3 - sw // 2


def _place_shape(
    grid: np.ndarray,
    occupied: np.ndarray,
    shape: np.ndarray,
    side: str,
    distance: int,
    color: int,
) -> bool:
    h, w = grid.shape
    sh, sw = shape.shape
    half_r = sh // 2
    half_c = sw // 2

    if side == "top":
        center_r = 1 + int(distance)
        center_c_choices = list(range(1 + half_c, w - 1 - half_c))
    elif side == "bottom":
        center_r = h - 2 - int(distance)
        center_c_choices = list(range(1 + half_c, w - 1 - half_c))
    elif side == "left":
        center_c = 1 + int(distance)
        center_r_choices = list(range(1 + half_r, h - 1 - half_r))
    else:
        center_c = w - 2 - int(distance)
        center_r_choices = list(range(1 + half_r, h - 1 - half_r))

    trials = center_c_choices if side in ("top", "bottom") else center_r_choices
    random.shuffle(trials)

    for axis_center in trials:
        if side in ("top", "bottom"):
            center_c = int(axis_center)
        else:
            center_r = int(axis_center)
        top = int(center_r - half_r)
        left = int(center_c - half_c)
        bottom = top + sh - 1
        right = left + sw - 1
        if top < 1 or left < 1 or bottom > h - 2 or right > w - 2:
            continue

        cells: list[tuple[int, int]] = []
        ok = True
        for rr in range(sh):
            for cc in range(sw):
                if not shape[rr, cc]:
                    continue
                gr = top + rr
                gc = left + cc
                if occupied[gr, gc]:
                    ok = False
                    break
                for nr in range(max(0, gr - 1), min(h, gr + 2)):
                    for nc in range(max(0, gc - 1), min(w, gc + 2)):
                        if occupied[nr, nc]:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
                cells.append((int(gr), int(gc)))
            if not ok:
                break
        if not ok:
            continue

        for gr, gc in cells:
            grid[gr, gc] = int(color)
            occupied[gr, gc] = True
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    bg_choices = [c for c in range(10) if c != 5]
    bg = random.choice(bg_choices)
    grey = 5

    orientation = random.choice(["horizontal", "vertical"])
    if orientation == "horizontal":
        h = random.randint(12, 20)
        w = random.randint(12, 20)
        side = random.choice(["top", "bottom"])
        max_bar_len = min(8, w - 2)
    else:
        h = random.randint(12, 20)
        w = random.randint(12, 20)
        side = random.choice(["left", "right"])
        max_bar_len = min(8, h - 2)
    if max_bar_len < 4:
        return None

    bar_len = random.randint(4, max_bar_len)
    palette = [c for c in range(10) if c not in (bg, grey)]
    if len(palette) < bar_len:
        return None
    bar_colors = random.sample(palette, bar_len)

    grid = np.full((h, w), int(bg), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    if side in ("top", "bottom"):
        start = random.randint(0, w - bar_len)
        row = 0 if side == "top" else h - 1
        for i, color in enumerate(bar_colors):
            grid[row, start + i] = int(color)
            occupied[row, start + i] = True
    else:
        start = random.randint(0, h - bar_len)
        col = 0 if side == "left" else w - 1
        for i, color in enumerate(bar_colors):
            grid[start + i, col] = int(color)
            occupied[start + i, col] = True

    shapes = random.sample(_SHAPES, k=min(len(_SHAPES), random.randint(5, 10)))
    wrap_mode = random.choice([False, True])
    num_objects = random.randint(2, 5)
    wrap_needed = 1 if wrap_mode else 0
    direct_needed = 1 if wrap_mode and num_objects >= 2 else 0

    for index in range(num_objects):
        placed = False
        shape_order = shapes[:]
        random.shuffle(shape_order)
        must_wrap = wrap_needed > 0 and (num_objects - index == wrap_needed or random.random() < 0.6)
        must_direct = direct_needed > 0 and not must_wrap and (
            num_objects - index == direct_needed or random.random() < 0.6
        )

        for shape in shape_order:
            lo, hi = _candidate_distance_bounds(side, shape, grid.shape)
            if lo > hi:
                continue
            direct_hi = min(hi, bar_len - 1)
            wrap_lo = max(lo, bar_len)
            if must_wrap:
                if wrap_lo > hi:
                    continue
                choices = list(range(wrap_lo, hi + 1))
            elif must_direct:
                if lo > direct_hi:
                    continue
                choices = list(range(lo, direct_hi + 1))
            elif not wrap_mode:
                if lo > direct_hi:
                    continue
                choices = list(range(lo, direct_hi + 1))
            else:
                choices = list(range(lo, hi + 1))
            random.shuffle(choices)

            for distance in choices:
                if _place_shape(grid, occupied, shape, side, int(distance), grey):
                    if must_wrap:
                        wrap_needed -= 1
                    if must_direct:
                        direct_needed -= 1
                    placed = True
                    break
            if placed:
                break
        if not placed:
            return None

    if wrap_mode and wrap_needed != 0:
        return None

    if int(np.count_nonzero(grid == grey)) == 0:
        return None
    if _get_bg(grid) != int(bg):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gradient-lookup ARC input")
