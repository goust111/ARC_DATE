from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 3
FRAME = 1
BLOCKER = 5
MARKER_COLORS = (2, 4, 6, 7, 8, 9)
ODD_SIZES = (15, 17, 19, 21, 23)
MAX_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _center(shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    return h // 2, w // 2


def _manhattan(r: int, c: int, cr: int, cc: int) -> int:
    return abs(int(r) - int(cr)) + abs(int(c) - int(cc))


def _frame_radius(grid: np.ndarray, center: tuple[int, int]) -> int:
    cr, cc = center
    blue = np.argwhere(grid == FRAME)
    assert blue.size > 0, "expected blue frame pixels"
    return max(_manhattan(r, c, cr, cc) for r, c in blue)


def _diamond_boundary(shape: tuple[int, int], center: tuple[int, int], radius: int) -> list[tuple[int, int]]:
    h, w = shape
    cr, cc = center
    cells = []
    for r in range(h):
        for c in range(w):
            if _manhattan(r, c, cr, cc) == radius:
                cells.append((r, c))
    return cells


def _orbit(r: int, c: int, center: tuple[int, int], shape: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    h, w = shape
    cr, cc = center
    pts = {
        (int(r), int(c)),
        (2 * cr - int(r), int(c)),
        (int(r), 2 * cc - int(c)),
        (2 * cr - int(r), 2 * cc - int(c)),
    }
    return tuple(sorted((rr, cc2) for rr, cc2 in pts if 0 <= rr < h and 0 <= cc2 < w))


def _side_and_dir(r: int, c: int, center: tuple[int, int], radius: int) -> tuple[str, tuple[int, int]]:
    cr, cc = center
    sides: list[tuple[str, tuple[int, int]]] = []
    if r == cr - radius:
        sides.append(("top", (1, 0)))
    if r == cr + radius:
        sides.append(("bottom", (-1, 0)))
    if c == cc - radius:
        sides.append(("left", (0, 1)))
    if c == cc + radius:
        sides.append(("right", (0, -1)))
    assert len(sides) == 1, "marker position must belong to exactly one bbox side"
    return sides[0]


def _entry_boundary(side: str, pos: int, center: tuple[int, int], radius: int) -> tuple[int, int]:
    cr, cc = center
    if side == "top":
        return cr - radius + abs(pos - cc), pos
    if side == "bottom":
        return cr + radius - abs(pos - cc), pos
    if side == "left":
        return pos, cc - radius + abs(pos - cr)
    if side == "right":
        return pos, cc + radius - abs(pos - cr)
    raise ValueError(side)


def _marker_items(grid: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    coords = np.argwhere((grid != bg) & (grid != FRAME) & (grid != BLOCKER))
    return [(int(r), int(c), int(grid[r, c])) for r, c in coords]


def _complete_boundary(grid: np.ndarray, bg: int, center: tuple[int, int], radius: int) -> tuple[np.ndarray, set[tuple[int, int]]]:
    out = np.array(grid, dtype=int, copy=True)
    repaired: set[tuple[int, int]] = set()
    boundary = _diamond_boundary(out.shape, center, radius)
    for r, c in boundary:
        if int(out[r, c]) != bg:
            continue
        orbit = [p for p in _orbit(r, c, center, out.shape) if _manhattan(p[0], p[1], center[0], center[1]) == radius]
        colors = {int(out[rr, cc]) for rr, cc in orbit if int(out[rr, cc]) != bg}
        if colors == {FRAME}:
            out[r, c] = FRAME
            repaired.add((r, c))
    for r, c in boundary:
        assert int(out[r, c]) == FRAME, "every frame gap must be recoverable from its mirror orbit"
    return out, repaired


def _plan_rays(
    grid: np.ndarray,
    center: tuple[int, int],
    radius: int,
    repaired: set[tuple[int, int]],
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], tuple[tuple[int, int], ...]]]:
    bg = _get_bg(grid)
    h, w = grid.shape
    planned: dict[tuple[int, int], int] = {}
    details: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {}

    for r, c, color in _marker_items(grid, bg):
        side, (dr, dc) = _side_and_dir(r, c, center, radius)
        pos = c if side in ("top", "bottom") else r
        entry = _entry_boundary(side, pos, center, radius)
        rr = r + dr
        cc = c + dc
        entered = False
        painted: list[tuple[int, int]] = []

        while 0 <= rr < h and 0 <= cc < w:
            if not entered and (rr, cc) == entry and entry in repaired:
                break

            inside = _manhattan(rr, cc, center[0], center[1]) < radius
            if inside:
                entered = True
                if int(grid[rr, cc]) != bg:
                    break
                prev = planned.get((rr, cc))
                assert prev is None or prev == color, "conflicting ray colors"
                planned[(rr, cc)] = color
                painted.append((rr, cc))
            elif entered:
                if int(grid[rr, cc]) != bg:
                    break
                break

            rr += dr
            cc += dc

        details[(r, c)] = tuple(painted)

    return planned, details


def _apply_rays(
    grid: np.ndarray,
    center: tuple[int, int],
    radius: int,
    repaired: set[tuple[int, int]],
) -> tuple[np.ndarray, dict[tuple[int, int], tuple[tuple[int, int], ...]]]:
    out = np.array(grid, dtype=int, copy=True)
    planned, details = _plan_rays(grid, center, radius, repaired)
    for (r, c), color in planned.items():
        out[r, c] = color
    return out, details


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore mirror-determined diamond-boundary gaps, then shoot side-marker rays into the repaired diamond corridor.

    Input structure:
    - Green background surrounds a centered one-pixel-thick blue diamond frame with 3-5 missing boundary cells.
    - Colored single-pixel markers sit on the four sides of the diamond's bounding box, and 1-3 gray pixels act as interior blockers.
    - The diamond boundary is symmetric across the grid-center horizontal and vertical axes, so each missing boundary cell has blue mirror evidence in its two-axis orbit.

    Rule:
    - Use the centered horizontal/vertical mirror orbit of each boundary gap to restore every missing blue frame pixel.
    - After repairing the frame, each side marker emits inward along the normal of its bounding-box side.
    - If a repaired entry-side frame pixel closes that marker's opening, the ray stops immediately; otherwise it paints green cells strictly inside the diamond until the first non-background cell after entering the interior.
    - Keep the repaired frame, the side markers, and the gray blockers unchanged.

    Key insight:
    - The same diamond boundary serves two roles: two-axis symmetry reveals where the missing blue pixels must be restored, and those restored pixels can then change which marker rays are able to penetrate and how far they can paint.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    center = _center(g.shape)
    radius = _frame_radius(g, center)
    bg = _get_bg(g)
    repaired_grid, repaired = _complete_boundary(g, bg, center, radius)
    out, _details = _apply_rays(repaired_grid, center, radius, repaired)
    return out


def _draw_full_frame(grid: np.ndarray, center: tuple[int, int], radius: int) -> None:
    for r, c in _diamond_boundary(grid.shape, center, radius):
        grid[r, c] = FRAME


def _side_candidates(center: tuple[int, int], radius: int) -> dict[str, list[tuple[int, int, tuple[int, int]]]]:
    cr, cc = center
    sides: dict[str, list[tuple[int, int, tuple[int, int]]]] = {s: [] for s in ("top", "bottom", "left", "right")}
    for c in range(cc - radius + 1, cc + radius):
        if c == cc:
            continue
        sides["top"].append((cr - radius, c, _entry_boundary("top", c, center, radius)))
        sides["bottom"].append((cr + radius, c, _entry_boundary("bottom", c, center, radius)))
    for r in range(cr - radius + 1, cr + radius):
        if r == cr:
            continue
        sides["left"].append((r, cc - radius, _entry_boundary("left", r, center, radius)))
        sides["right"].append((r, cc + radius, _entry_boundary("right", r, center, radius)))
    return sides


def _can_add_missing(
    cell: tuple[int, int],
    missing: set[tuple[int, int]],
    center: tuple[int, int],
    shape: tuple[int, int],
) -> bool:
    orbit = [p for p in _orbit(cell[0], cell[1], center, shape) if _manhattan(p[0], p[1], center[0], center[1]) == _manhattan(cell[0], cell[1], center[0], center[1])]
    return sum(p in missing for p in orbit) < len(orbit) - 1


def _sample_markers_and_gaps(
    center: tuple[int, int],
    radius: int,
    shape: tuple[int, int],
) -> tuple[list[tuple[str, int, int, tuple[int, int]]], list[tuple[str, int, int, tuple[int, int]]], set[tuple[int, int]]]:
    side_slots = _side_candidates(center, radius)
    target_missing = random.randint(3, 5)
    affected_target = random.randint(1, min(3, target_missing - 1))
    painter_target = random.randint(1, 3)

    affected: list[tuple[str, int, int, tuple[int, int]]] = []
    painters: list[tuple[str, int, int, tuple[int, int]]] = []
    missing: set[tuple[int, int]] = set()
    side_counts = {s: 0 for s in side_slots}

    affected_pool: list[tuple[str, int, int, tuple[int, int]]] = []
    for side, items in side_slots.items():
        for r, c, entry in items:
            affected_pool.append((side, r, c, entry))
    random.shuffle(affected_pool)

    for slot in affected_pool:
        side, _r, _c, entry = slot
        if side_counts[side] >= 2 or entry in missing:
            continue
        if not _can_add_missing(entry, missing, center, shape):
            continue
        affected.append(slot)
        missing.add(entry)
        side_counts[side] += 1
        if len(affected) == affected_target:
            break
    if len(affected) != affected_target:
        raise RuntimeError("failed to choose affected markers")

    extra_boundary = _diamond_boundary(shape, center, radius)
    random.shuffle(extra_boundary)
    for cell in extra_boundary:
        if len(missing) >= target_missing:
            break
        if cell in missing:
            continue
        if not _can_add_missing(cell, missing, center, shape):
            continue
        missing.add(cell)
    if len(missing) != target_missing:
        raise RuntimeError("failed to choose enough boundary gaps")

    painter_pool: list[tuple[str, int, int, tuple[int, int]]] = []
    for side, items in side_slots.items():
        for r, c, entry in items:
            if entry in missing:
                continue
            painter_pool.append((side, r, c, entry))
    random.shuffle(painter_pool)
    for slot in painter_pool:
        side, r, c, entry = slot
        if side_counts[side] >= 2:
            continue
        if any((r, c) == (rr, cc) for _s, rr, cc, _e in affected + painters):
            continue
        painters.append(slot)
        side_counts[side] += 1
        if len(painters) == painter_target:
            break
    if not painters:
        raise RuntimeError("need at least one painter marker")

    return affected, painters, missing


def _place_markers(
    grid: np.ndarray,
    affected: list[tuple[str, int, int, tuple[int, int]]],
    painters: list[tuple[str, int, int, tuple[int, int]]],
) -> None:
    slots = affected + painters
    colors = random.choices(MARKER_COLORS, k=len(slots))
    random.shuffle(colors)
    for (_side, r, c, _entry), color in zip(slots, colors):
        grid[r, c] = color


def _strict_interior(shape: tuple[int, int], center: tuple[int, int], radius: int) -> list[tuple[int, int]]:
    h, w = shape
    cr, cc = center
    return [(r, c) for r in range(h) for c in range(w) if _manhattan(r, c, cr, cc) < radius]


def _place_blockers(
    grid: np.ndarray,
    center: tuple[int, int],
    radius: int,
    repaired: set[tuple[int, int]],
) -> None:
    blocker_count = random.randint(1, 3)
    no_blocker_grid = np.array(grid, copy=True)
    try:
        _out_no_blocker, details = _apply_rays(no_blocker_grid, center, radius, repaired)
    except AssertionError as exc:
        raise RuntimeError("conflicting rays before blocker placement") from exc
    painter_cells = sorted({cell for cells in details.values() for cell in cells})
    interior = [p for p in _strict_interior(grid.shape, center, radius) if int(grid[p]) == BG]

    chosen: list[tuple[int, int]] = []
    if painter_cells:
        tail_candidates = [p for p in painter_cells if all(p != cells[0] for cells in details.values() if cells)]
        if tail_candidates:
            chosen.append(random.choice(tail_candidates))
        else:
            chosen.append(random.choice(painter_cells))

    random.shuffle(interior)
    for cell in interior:
        if len(chosen) >= blocker_count:
            break
        if cell in chosen:
            continue
        chosen.append(cell)

    for r, c in chosen[:blocker_count]:
        grid[r, c] = BLOCKER


def _validate_grid(
    grid: np.ndarray,
    missing: set[tuple[int, int]],
    affected: list[tuple[str, int, int, tuple[int, int]]],
    painters: list[tuple[str, int, int, tuple[int, int]]],
) -> bool:
    center = _center(grid.shape)
    radius = _frame_radius(grid, center)
    bg = _get_bg(grid)

    try:
        completed, repaired = _complete_boundary(grid, bg, center, radius)
        out, repaired_details = _apply_rays(completed, center, radius, repaired)
        boundary_only = completed
        incomplete_out, incomplete_details = _apply_rays(grid, center, radius, set())
    except AssertionError:
        return False

    if len(missing) < 3 or len(missing) > 5:
        return False
    if not affected or not painters:
        return False
    if np.array_equal(out, grid):
        return False
    if np.array_equal(out, boundary_only):
        return False
    if np.array_equal(out, incomplete_out):
        return False

    changed_to_marker = 0
    for r, c in np.argwhere(out != grid):
        if int(grid[r, c]) == bg and int(out[r, c]) in MARKER_COLORS:
            changed_to_marker += 1
    if changed_to_marker == 0:
        return False

    affected_changed = 0
    for _side, r, c, _entry in affected:
        before = len(incomplete_details.get((r, c), ()))
        after = len(repaired_details.get((r, c), ()))
        if before > after:
            affected_changed += 1
    if affected_changed == 0:
        return False

    painter_positive = 0
    for _side, r, c, _entry in painters:
        if len(repaired_details.get((r, c), ())) > 0:
            painter_positive += 1
    if painter_positive == 0:
        return False

    return True


def _generate_once() -> np.ndarray:
    size = random.choice(ODD_SIZES)
    grid = np.full((size, size), BG, dtype=int)
    center = _center(grid.shape)
    radius = random.randint(4, center[0] - 2)
    _draw_full_frame(grid, center, radius)

    affected, painters, missing = _sample_markers_and_gaps(center, radius, grid.shape)
    for r, c in missing:
        grid[r, c] = BG
    _place_markers(grid, affected, painters)

    completed, repaired = _complete_boundary(grid, BG, center, radius)
    _place_blockers(grid, center, radius, repaired)

    if not _validate_grid(grid, missing, affected, painters):
        raise RuntimeError("rejected degeneracy")
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        try:
            return _generate_once()
        except RuntimeError:
            continue
    raise RuntimeError("failed to generate a valid puzzle instance")
