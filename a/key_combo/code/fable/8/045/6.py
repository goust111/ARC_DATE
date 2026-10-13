from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists():
    sys.path.append(str(_ARC_UTILS_DIR))

try:
    from arc_utils import Color
except Exception:
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        YELLOW = 4
        GREY = 5
        TEAL = 8


ORTHO_STEPS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAG_STEPS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ALL8_STEPS = ORTHO_STEPS + DIAG_STEPS

GRID_CHOICES = [19, 21, 23, 25, 27]
GEN_RETRIES = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    steps = ORTHO_STEPS if connectivity == 4 else ALL8_STEPS
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in steps:
                    nr, nc = cr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return int(mins[0]), int(mins[1]), int(maxs[0]), int(maxs[1])


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _symmetry_flags(mask: np.ndarray) -> tuple[bool, bool]:
    return bool(np.array_equal(mask, mask[::-1, :])), bool(np.array_equal(mask, mask[:, ::-1]))


def _gray_objects(grid: np.ndarray) -> list[dict]:
    out = []
    for coords in _components(grid == Color.GREY, connectivity=8):
        r0, c0, r1, c1 = _bbox(coords)
        patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        patch[coords[:, 0] - r0, coords[:, 1] - c0] = True
        hsym, vsym = _symmetry_flags(patch)
        out.append(
            {
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "patch": patch,
                "hsym": hsym,
                "vsym": vsym,
            }
        )
    return out


def _find_marker_axis(grid: np.ndarray) -> tuple[int, int]:
    candidates = []
    for obj in _gray_objects(grid):
        h, w = obj["patch"].shape
        if obj["hsym"] and obj["vsym"] and h % 2 == 1 and w % 2 == 1:
            r0, c0, _r1, _c1 = obj["bbox"]
            candidates.append((r0 + h // 2, c0 + w // 2))
    assert len(candidates) == 1, "expected exactly one odd-sized doubly symmetric grey marker"
    return candidates[0]


def _recolor_from_axes(grid: np.ndarray, axis_r: int, axis_c: int) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    blue = g == Color.BLUE
    out = g.copy()
    for r, c in np.argwhere(blue):
        vr, vc = r, 2 * axis_c - c
        hr, hc = 2 * axis_r - r, c
        has_v = _in_bounds((h, w), vr, vc) and blue[vr, vc]
        has_h = _in_bounds((h, w), hr, hc) and blue[hr, hc]
        if has_v and has_h:
            out[r, c] = Color.RED
        elif has_v:
            out[r, c] = Color.YELLOW
        elif has_h:
            out[r, c] = Color.TEAL
        else:
            out[r, c] = Color.GREY
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each blue pixel by the occupancy of its borrowed-axis reflection partners.

    Input structure:
    - The grid has black background, several separated grey objects, and a scattered blue pixel cloud.
    - Exactly one grey object is mirror-symmetric in both its own horizontal and vertical bbox axes.
    - All other grey objects are merely distractors with at most one such symmetry.

    Rule:
    - Detect the unique doubly symmetric grey object and borrow its center row and center column as the two reflection axes.
    - For every blue pixel, test whether the vertically reflected cell and the horizontally reflected cell across those borrowed axes are also blue in the input.
    - Recolor that blue pixel red when both partners exist, yellow when only the vertical partner exists, teal when only the horizontal partner exists, and grey when neither exists; leave every grey object unchanged.

    Key insight:
    - The required axes come from the unique off-center symmetric marker, not from the grid midpoint, so the output is an orbit-occupancy classification in the marker's coordinate system.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    axis_r, axis_c = _find_marker_axis(g)
    return _recolor_from_axes(g, axis_r, axis_c)


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rs0 = max(0, -dr)
            rs1 = min(h, h - dr) if dr >= 0 else h
            cs0 = max(0, -dc)
            cs1 = min(w, w - dc) if dc >= 0 else w
            rd0 = max(0, dr)
            rd1 = rd0 + (rs1 - rs0)
            cd0 = max(0, dc)
            cd1 = cd0 + (cs1 - cs0)
            out[rd0:rd1, cd0:cd1] |= mask[rs0:rs1, cs0:cs1]
    return out


def _orbit_double(height: int, width: int, r: int, c: int) -> set[tuple[int, int]]:
    rs = {r, height - 1 - r}
    cs = {c, width - 1 - c}
    return {(rr, cc) for rr in rs for cc in cs}


def _orbit_vertical(height: int, width: int, r: int, c: int) -> set[tuple[int, int]]:
    return {(r, c), (r, width - 1 - c)}


def _orbit_horizontal(height: int, width: int, r: int, c: int) -> set[tuple[int, int]]:
    return {(r, c), (height - 1 - r, c)}


def _orbit_touches(existing: set[tuple[int, int]], orbit_cells: set[tuple[int, int]]) -> bool:
    for r, c in orbit_cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (r + dr, c + dc) in existing:
                    return True
    return False


def _mask_from_cells(height: int, width: int, cells: set[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return _crop_mask(mask)


def _grow_symmetric_mask(
    height: int,
    width: int,
    canonical_cells: list[tuple[int, int]],
    orbit_fn,
    seed: tuple[int, int],
    min_area: int,
    max_area: int,
) -> np.ndarray | None:
    selected = {seed}
    full_cells = set(orbit_fn(height, width, *seed))
    pool = set(canonical_cells) - selected
    target = random.randint(min_area, max_area)
    while pool and len(full_cells) < target:
        frontier = [cell for cell in pool if _orbit_touches(full_cells, set(orbit_fn(height, width, *cell)))]
        if not frontier:
            break
        cell = random.choice(frontier)
        selected.add(cell)
        pool.remove(cell)
        full_cells |= set(orbit_fn(height, width, *cell))
        if len(full_cells) >= min_area and random.random() < 0.22:
            break
    mask = _mask_from_cells(height, width, full_cells)
    if len(_components(mask, connectivity=8)) != 1:
        return None
    return mask


def _make_double_marker() -> np.ndarray:
    for _ in range(300):
        h = random.choice([5, 7, 9])
        w = random.choice([5, 7, 9])
        cr, cc = h // 2, w // 2
        canonical = [(r, c) for r in range(cr + 1) for c in range(cc + 1)]
        mask = _grow_symmetric_mask(h, w, canonical, _orbit_double, (cr, cc), 9, min(h * w - 3, 23))
        if mask is None:
            continue
        hsym, vsym = _symmetry_flags(mask)
        mr, mc = mask.shape[0] // 2, mask.shape[1] // 2
        coords = np.argwhere(mask)
        has_off_axis = any(r != mr and c != mc for r, c in coords)
        has_axis_only = any((r == mr) ^ (c == mc) for r, c in coords)
        if hsym and vsym and mask.shape[0] % 2 == 1 and mask.shape[1] % 2 == 1 and has_off_axis and has_axis_only:
            return mask
    raise RuntimeError("failed to build double-symmetric marker")


def _make_vertical_only() -> np.ndarray:
    for _ in range(300):
        h = random.choice([5, 7, 9])
        w = random.choice([5, 7])
        cc = w // 2
        canonical = [(r, c) for r in range(h) for c in range(cc + 1)]
        seed = (random.randrange(h), cc)
        mask = _grow_symmetric_mask(h, w, canonical, _orbit_vertical, seed, 6, min(h * w - 2, 16))
        if mask is None:
            continue
        hsym, vsym = _symmetry_flags(mask)
        if vsym and not hsym and np.any(mask[:, mask.shape[1] // 2]):
            return mask
    raise RuntimeError("failed to build vertical-only distractor")


def _make_horizontal_only() -> np.ndarray:
    for _ in range(300):
        h = random.choice([5, 7])
        w = random.choice([5, 7, 9])
        cr = h // 2
        canonical = [(r, c) for r in range(cr + 1) for c in range(w)]
        seed = (cr, random.randrange(w))
        mask = _grow_symmetric_mask(h, w, canonical, _orbit_horizontal, seed, 6, min(h * w - 2, 16))
        if mask is None:
            continue
        hsym, vsym = _symmetry_flags(mask)
        if hsym and not vsym and np.any(mask[mask.shape[0] // 2, :]):
            return mask
    raise RuntimeError("failed to build horizontal-only distractor")


def _make_asymmetric() -> np.ndarray:
    for _ in range(300):
        h = random.randint(4, 8)
        w = random.randint(4, 8)
        area = random.randint(5, min(h * w - 1, 13))
        cells = {(random.randrange(h), random.randrange(w))}
        while len(cells) < area:
            r, c = random.choice(tuple(cells))
            dr, dc = random.choice(ALL8_STEPS)
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                cells.add((nr, nc))
        mask = _mask_from_cells(h, w, cells)
        if len(_components(mask, connectivity=8)) != 1:
            continue
        hsym, vsym = _symmetry_flags(mask)
        if not hsym and not vsym:
            return mask
    raise RuntimeError("failed to build asymmetric distractor")


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    coords = np.argwhere(mask)
    grid[r0 + coords[:, 0], c0 + coords[:, 1]] = color


def _place_mask(
    grid: np.ndarray,
    occupied: np.ndarray,
    mask: np.ndarray,
    *,
    require_off_center_axis: bool = False,
) -> tuple[int, int] | None:
    h, w = grid.shape
    mh, mw = mask.shape
    blocked = _dilate8(occupied)
    cells = np.argwhere(mask)
    options = []
    for r0 in range(h - mh + 1):
        for c0 in range(w - mw + 1):
            rr = r0 + cells[:, 0]
            cc = c0 + cells[:, 1]
            if np.any(blocked[rr, cc]):
                continue
            if require_off_center_axis:
                axis_r = r0 + mh // 2
                axis_c = c0 + mw // 2
                if min(axis_r, h - 1 - axis_r, axis_c, w - 1 - axis_c) < 4:
                    continue
                if axis_r == h // 2 and axis_c == w // 2:
                    continue
                if abs(axis_r - h // 2) + abs(axis_c - w // 2) < 3:
                    continue
            options.append((r0, c0))
    return random.choice(options) if options else None


def _mark_occupied(occupied: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> None:
    coords = np.argwhere(mask)
    occupied[r0 + coords[:, 0], c0 + coords[:, 1]] = True


def _orbit_points(axis_r: int, axis_c: int, dr: int, dc: int) -> list[tuple[int, int]]:
    return [
        (axis_r - dr, axis_c - dc),
        (axis_r - dr, axis_c + dc),
        (axis_r + dr, axis_c - dc),
        (axis_r + dr, axis_c + dc),
    ]


def _instantiate_pattern(kind: str) -> list[int]:
    if kind == "full":
        return [0, 1, 2, 3]
    if kind == "vertical":
        return random.choice([[0, 1], [2, 3]])
    if kind == "horizontal":
        return random.choice([[0, 2], [1, 3]])
    if kind == "diagonal":
        return random.choice([[0, 3], [1, 2]])
    if kind == "triad":
        miss = random.randrange(4)
        return [i for i in range(4) if i != miss]
    if kind == "single":
        return [random.randrange(4)]
    raise ValueError(kind)


def _reserve_cells(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    blocked[nr, nc] = True


def _place_blue_orbits(grid: np.ndarray, occupied: np.ndarray, axis_r: int, axis_c: int) -> bool:
    h, w = grid.shape
    blocked = _dilate8(occupied)
    required = ["full", "vertical", "horizontal", "diagonal", "triad"]
    extras = random.choices(["single", "vertical", "horizontal", "diagonal", "triad", "full"], k=random.randint(1, 4))
    all_kinds = required + extras
    max_dr = min(axis_r, h - 1 - axis_r)
    max_dc = min(axis_c, w - 1 - axis_c)
    if max_dr < 1 or max_dc < 1:
        return False
    candidate_pairs = [(dr, dc) for dr in range(1, max_dr + 1) for dc in range(1, max_dc + 1)]
    random.shuffle(candidate_pairs)
    for kind in all_kinds:
        random.shuffle(candidate_pairs)
        placed = False
        for dr, dc in candidate_pairs:
            orbit = _orbit_points(axis_r, axis_c, dr, dc)
            if any(blocked[r, c] for r, c in orbit):
                continue
            selected = _instantiate_pattern(kind)
            for idx in selected:
                r, c = orbit[idx]
                grid[r, c] = Color.BLUE
            _reserve_cells(blocked, orbit)
            placed = True
            break
        if not placed:
            return False
    return True


def _branch_counts(inp: np.ndarray, out: np.ndarray) -> dict[int, int]:
    counts = {Color.RED: 0, Color.YELLOW: 0, Color.TEAL: 0, Color.GREY: 0}
    for r, c in np.argwhere(inp == Color.BLUE):
        counts[int(out[r, c])] += 1
    return counts


def _semantic_ok(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False
    if np.array_equal(out, grid):
        return False

    objs = _gray_objects(grid)
    both = sum(1 for obj in objs if obj["hsym"] and obj["vsym"] and obj["patch"].shape[0] % 2 == 1 and obj["patch"].shape[1] % 2 == 1)
    vertical_only = sum(1 for obj in objs if obj["vsym"] and not obj["hsym"])
    horizontal_only = sum(1 for obj in objs if obj["hsym"] and not obj["vsym"])
    if both != 1 or vertical_only < 1 or horizontal_only < 1:
        return False

    counts = _branch_counts(grid, out)
    if not all(counts[color] > 0 for color in (Color.RED, Color.YELLOW, Color.TEAL, Color.GREY)):
        return False

    axis_r, axis_c = _find_marker_axis(grid)
    if axis_r == grid.shape[0] // 2 and axis_c == grid.shape[1] // 2:
        return False
    center_out = _recolor_from_axes(grid, grid.shape[0] // 2, grid.shape[1] // 2)
    if np.array_equal(center_out, out):
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    h = random.choice(GRID_CHOICES)
    w = random.choice(GRID_CHOICES)
    grid = np.full((h, w), Color.BLACK, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    marker = _make_double_marker()
    pos = _place_mask(grid, occupied, marker, require_off_center_axis=True)
    if pos is None:
        return None
    mr0, mc0 = pos
    _paint_mask(grid, marker, mr0, mc0, Color.GREY)
    _mark_occupied(occupied, marker, mr0, mc0)
    axis_r = mr0 + marker.shape[0] // 2
    axis_c = mc0 + marker.shape[1] // 2

    distractor_masks = [_make_vertical_only(), _make_horizontal_only()]
    for _ in range(random.randint(1, 2)):
        distractor_masks.append(random.choice([_make_vertical_only(), _make_horizontal_only(), _make_asymmetric()]))
    random.shuffle(distractor_masks)
    for mask in distractor_masks:
        pos = _place_mask(grid, occupied, mask)
        if pos is None:
            return None
        r0, c0 = pos
        _paint_mask(grid, mask, r0, c0, Color.GREY)
        _mark_occupied(occupied, mask, r0, c0)

    if not _place_blue_orbits(grid, occupied, axis_r, axis_c):
        return None
    if not _semantic_ok(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid borrowed-axis orbit puzzle")
