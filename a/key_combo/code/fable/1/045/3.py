from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox_coords, colors_present, crop_bbox, find_components, neighbors4


BG = Color.BLACK
BLUE = Color.BLUE
GREEN = Color.GREEN
RED = Color.RED

LOCAL_CANVAS = 17
LOCAL_CENTER = LOCAL_CANVAS // 2
GRID_H_RANGE = (18, 29)
GRID_W_RANGE = (18, 29)
GEN_RETRIES = 400
SHAPE_RETRIES = 300
PLACEMENT_RETRIES = 300

GROW_DIRECTIONS = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
]


def _hole_components(mask: np.ndarray) -> tuple[list[list[tuple[int, int]]], tuple[int, int, int, int] | None]:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return [], None

    b = bbox_coords(pts)
    sub = crop_bbox(mask.astype(int), b).astype(bool)
    h, w = sub.shape
    background = ~sub
    seen = np.zeros((h, w), dtype=bool)
    holes: list[list[tuple[int, int]]] = []

    for r0 in range(h):
        for c0 in range(w):
            if not background[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            comp: list[tuple[int, int]] = []
            touches_border = False
            while stack:
                r, c = stack.pop()
                comp.append((r, c))
                if r in (0, h - 1) or c in (0, w - 1):
                    touches_border = True
                for nr, nc in neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and background[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            if not touches_border:
                holes.append(comp)

    return holes, b


def _hole_count(mask: np.ndarray) -> int:
    return len(_hole_components(mask)[0])


def _find_unique_hole(mask: np.ndarray) -> tuple[int, int] | None:
    holes, b = _hole_components(mask)
    if b is None or len(holes) != 1 or len(holes[0]) != 1:
        return None
    r0, c0, _, _ = b
    hr, hc = holes[0][0]
    return r0 + hr, c0 + hc


def _relative_coords(mask: np.ndarray) -> set[tuple[int, int]]:
    return {(int(r - LOCAL_CENTER), int(c - LOCAL_CENTER)) for r, c in np.argwhere(mask)}


def _coords_bbox(coords: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _render_coords(coords: set[tuple[int, int]], color: int) -> np.ndarray:
    r0, c0, r1, c1 = _coords_bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in coords:
        out[r - r0, c - c0] = int(color)
    return out


def _centroid_shift(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> tuple[int, int]:
    pa = np.array(sorted(a), dtype=float)
    pb = np.array(sorted(b), dtype=float)
    da = pa.mean(axis=0)
    db = pb.mean(axis=0)
    return int(round(da[0] - db[0])), int(round(da[1] - db[1]))


def _frontier(mask: np.ndarray, banned: frozenset[tuple[int, int]] = frozenset()) -> list[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for nr, nc in neighbors4(rr, cc):
            if 0 <= nr < LOCAL_CANVAS and 0 <= nc < LOCAL_CANVAS and not mask[nr, nc] and (nr, nc) not in banned:
                cells.add((nr, nc))
    cells.discard((LOCAL_CENTER, LOCAL_CENTER))
    return list(cells)


def _choose_weighted(candidates: list[tuple[int, int]], direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    weights = []
    for r, c in candidates:
        rr = r - LOCAL_CENTER
        cc = c - LOCAL_CENTER
        score = 1.0 + max(0, rr * dr + cc * dc) + 0.35 * (abs(rr) + abs(cc))
        weights.append(score)
    return random.choices(candidates, weights=weights, k=1)[0]


def _grow_mask(
    mask: np.ndarray,
    target_area: int,
    target_holes: int,
    direction: tuple[int, int],
    *,
    banned: frozenset[tuple[int, int]] = frozenset(),
) -> np.ndarray | None:
    grown = mask.copy()
    while int(grown.sum()) < int(target_area):
        valid: list[tuple[int, int]] = []
        for cell in _frontier(grown, banned):
            trial = grown.copy()
            trial[cell] = True
            if _hole_count(trial) != int(target_holes):
                continue
            if int(target_holes) == 1 and _find_unique_hole(trial) != (LOCAL_CENTER, LOCAL_CENTER):
                continue
            valid.append(cell)
        if not valid:
            return None
        grown[_choose_weighted(valid, direction)] = True
    return grown


def _generate_local_shapes() -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]] | None:
    for _ in range(SHAPE_RETRIES):
        seed = np.zeros((LOCAL_CANVAS, LOCAL_CANVAS), dtype=bool)
        seed[LOCAL_CENTER - 1 : LOCAL_CENTER + 2, LOCAL_CENTER - 1 : LOCAL_CENTER + 2] = True

        full = _grow_mask(seed, random.randint(11, 20), 0, random.choice(GROW_DIRECTIONS))
        if full is None:
            continue

        core = full.copy()
        core[LOCAL_CENTER, LOCAL_CENTER] = False
        if _find_unique_hole(core) != (LOCAL_CENTER, LOCAL_CENTER):
            continue

        core_coords = _relative_coords(core)
        core_bbox = _coords_bbox(core_coords)
        core_h = core_bbox[2] - core_bbox[0] + 1
        core_w = core_bbox[3] - core_bbox[1] + 1
        if min(core_h, core_w) < 3 or max(core_h, core_w) > 10:
            continue

        margins = (
            LOCAL_CENTER - int(np.argwhere(core)[:, 0].min()),
            int(np.argwhere(core)[:, 0].max()) - LOCAL_CENTER,
            LOCAL_CENTER - int(np.argwhere(core)[:, 1].min()),
            int(np.argwhere(core)[:, 1].max()) - LOCAL_CENTER,
        )
        if len(set(margins)) == 1:
            continue

        blue_dir = random.choice(GROW_DIRECTIONS)
        green_dir = random.choice([d for d in GROW_DIRECTIONS if d != blue_dir])

        blue_mask = _grow_mask(core, int(core.sum()) + random.randint(3, 8), 1, blue_dir)
        if blue_mask is None:
            continue
        blue_extra = frozenset((int(r), int(c)) for r, c in np.argwhere(blue_mask & ~core))

        green_mask = _grow_mask(
            core,
            int(core.sum()) + random.randint(3, 8),
            1,
            green_dir,
            banned=blue_extra,
        )
        if green_mask is None:
            continue

        if _find_unique_hole(blue_mask) != (LOCAL_CENTER, LOCAL_CENTER):
            continue
        if _find_unique_hole(green_mask) != (LOCAL_CENTER, LOCAL_CENTER):
            continue

        blue_coords = _relative_coords(blue_mask)
        green_coords = _relative_coords(green_mask)
        overlap = blue_coords & green_coords
        if overlap != core_coords:
            continue
        if len(blue_coords - core_coords) < 3 or len(green_coords - core_coords) < 3:
            continue

        blue_bbox = _coords_bbox(blue_coords)
        green_bbox = _coords_bbox(green_coords)
        blue_dims = (blue_bbox[2] - blue_bbox[0] + 1, blue_bbox[3] - blue_bbox[1] + 1)
        green_dims = (green_bbox[2] - green_bbox[0] + 1, green_bbox[3] - green_bbox[1] + 1)
        if blue_dims == green_dims:
            continue

        correct = _render_coords(overlap, 1)
        wrong_shifts = [
            (blue_bbox[0] - green_bbox[0], blue_bbox[1] - green_bbox[1]),
            (blue_bbox[2] - green_bbox[2], blue_bbox[3] - green_bbox[3]),
            _centroid_shift(blue_coords, green_coords),
        ]
        ambiguous = False
        for dr, dc in wrong_shifts:
            if (dr, dc) == (0, 0):
                continue
            wrong_overlap = blue_coords & {(r + dr, c + dc) for r, c in green_coords}
            if wrong_overlap and np.array_equal(_render_coords(wrong_overlap, 1), correct):
                ambiguous = True
                break
        if ambiguous:
            continue

        return core_coords, blue_coords, green_coords

    return None


def _global_cells(coords: set[tuple[int, int]], hole: tuple[int, int]) -> set[tuple[int, int]]:
    return {(hole[0] + dr, hole[1] + dc) for dr, dc in coords}


def _expand8(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                out.add((r + dr, c + dc))
    return out


def _hole_limits(coords: set[tuple[int, int]], h: int, w: int) -> tuple[tuple[int, int], tuple[int, int]]:
    r0, c0, r1, c1 = _coords_bbox(coords)
    return (0 - r0, h - 1 - r1), (0 - c0, w - 1 - c1)


def _choose_positions(
    blue_coords: set[tuple[int, int]],
    green_coords: set[tuple[int, int]],
    h: int,
    w: int,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    blue_row_lim, blue_col_lim = _hole_limits(blue_coords, h, w)
    green_row_lim, green_col_lim = _hole_limits(green_coords, h, w)
    if blue_row_lim[0] > blue_row_lim[1] or blue_col_lim[0] > blue_col_lim[1]:
        return None
    if green_row_lim[0] > green_row_lim[1] or green_col_lim[0] > green_col_lim[1]:
        return None

    for _ in range(PLACEMENT_RETRIES):
        blue_hole = (random.randint(*blue_row_lim), random.randint(*blue_col_lim))
        green_hole = (random.randint(*green_row_lim), random.randint(*green_col_lim))
        if blue_hole[0] == green_hole[0] or blue_hole[1] == green_hole[1]:
            continue

        blue_cells = _global_cells(blue_coords, blue_hole)
        green_cells = _global_cells(green_coords, green_hole)
        if blue_cells & green_cells:
            continue
        if _expand8(blue_cells) & green_cells:
            continue

        blue_bbox = _coords_bbox(blue_cells)
        green_bbox = _coords_bbox(green_cells)
        if any(blue_bbox[i] == green_bbox[i] for i in range(4)):
            continue
        return blue_hole, green_hole

    return None


def _paint(grid: np.ndarray, coords: set[tuple[int, int]], hole: tuple[int, int], color: int) -> None:
    for dr, dc in coords:
        grid[hole[0] + dr, hole[1] + dc] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align the blue and green clouds by their enclosed holes, then keep only the hidden overlap.

    Input structure:
    - A black background contains one blue cloud and one green cloud, separated from each other.
    - Each cloud is a single 4-connected monochrome shape with exactly one enclosed black hole cell.
    - The two clouds have different bounding-box sizes and their visible positions are not row/column aligned.

    Rule:
    - Detect the unique enclosed black cell inside each colored cloud and treat it as that cloud's anchor.
    - Translate one cloud so the two hole anchors coincide; the hole cell itself is not part of either shape.
    - Compute the intersection of occupied cells after that alignment and crop to its tight bounding box.
      Paint intersecting cells red and keep all other cells black.

    Key insight:
    - The alignment marker is invisible in the input: only topology distinguishes the true one-cell hole from ordinary open concavities.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(BG)
    colors = colors_present(g, exclude={bg})
    assert len(colors) == 2, "expected exactly two foreground colors"

    masks: list[np.ndarray] = []
    for color in colors:
        comps = find_components(g, bg=bg, colors=[int(color)], connectivity=4)
        assert len(comps) == 1, "each color must form one cloud"
        masks.append(g == int(color))

    anchor_a = _find_unique_hole(masks[0])
    anchor_b = _find_unique_hole(masks[1])
    assert anchor_a is not None and anchor_b is not None, "each cloud must contain one enclosed 1-cell hole"

    shift_r = int(anchor_a[0] - anchor_b[0])
    shift_c = int(anchor_a[1] - anchor_b[1])
    coords_a = {tuple(map(int, rc)) for rc in np.argwhere(masks[0])}
    coords_b = {(int(r + shift_r), int(c + shift_c)) for r, c in np.argwhere(masks[1])}

    overlap = coords_a & coords_b
    assert overlap, "aligned clouds must have a non-empty overlap"
    return _render_coords(overlap, RED)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        local = _generate_local_shapes()
        if local is None:
            continue
        core_coords, blue_coords, green_coords = local

        h = random.randint(*GRID_H_RANGE)
        w = random.randint(*GRID_W_RANGE)
        placed = _choose_positions(blue_coords, green_coords, h, w)
        if placed is None:
            continue
        blue_hole, green_hole = placed

        grid = np.zeros((h, w), dtype=int)
        _paint(grid, blue_coords, blue_hole, BLUE)
        _paint(grid, green_coords, green_hole, GREEN)

        expected = _render_coords(core_coords, RED)
        try:
            solved = main(grid)
        except Exception:
            continue
        if not np.array_equal(solved, expected):
            continue
        return grid

    raise RuntimeError("failed to generate a valid input for puzzle 3")
