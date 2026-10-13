from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
FRAME = 4
PATTERN_COLORS = (1, 2, 3, 8, 9)
GEN_RETRY_LIMIT = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _best_vertical_axis(g: np.ndarray, known: np.ndarray) -> int:
    h, w = g.shape
    best_a2 = 0
    best_key = None
    for a2 in range(0, 2 * (w - 1) + 1):
        mismatches = 0
        pairs = 0
        for r in range(h):
            cols = np.where(known[r])[0]
            for c in cols:
                c2 = a2 - int(c)
                if 0 <= c2 < w and known[r, c2]:
                    pairs += 1
                    if int(g[r, c]) != int(g[r, c2]):
                        mismatches += 1
        key = (mismatches, -pairs)
        if best_key is None or key < best_key:
            best_key = key
            best_a2 = a2
    return int(best_a2)


def _best_horizontal_axis(g: np.ndarray, known: np.ndarray) -> int:
    h, w = g.shape
    best_b2 = 0
    best_key = None
    rows = np.where(known.any(axis=1))[0]
    for b2 in range(0, 2 * (h - 1) + 1):
        mismatches = 0
        pairs = 0
        for r in rows:
            r2 = b2 - int(r)
            if not (0 <= r2 < h):
                continue
            cols = np.where(known[r] & known[r2])[0]
            pairs += int(len(cols))
            if len(cols):
                mismatches += int(np.sum(g[r, cols] != g[r2, cols]))
        key = (mismatches, -pairs)
        if best_key is None or key < best_key:
            best_key = key
            best_b2 = b2
    return int(best_b2)


def _mirror_candidates(cur: np.ndarray, r: int, c: int, a2: int, b2: int) -> list[int]:
    h, w = cur.shape
    vals: list[int] = []
    c2 = a2 - c
    if 0 <= c2 < w and int(cur[r, c2]) not in (BG, FRAME):
        vals.append(int(cur[r, c2]))
    r2 = b2 - r
    if 0 <= r2 < h and int(cur[r2, c]) not in (BG, FRAME):
        vals.append(int(cur[r2, c]))
    if 0 <= r2 < h and 0 <= c2 < w and int(cur[r2, c2]) not in (BG, FRAME):
        vals.append(int(cur[r2, c2]))
    return vals


def _frame_corners(box: tuple[int, int, int, int]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    r0, c0, r1, c1 = box
    return [
        ((r0, c0), (-1, -1)),
        ((r0, c1), (-1, 1)),
        ((r1, c0), (1, -1)),
        ((r1, c1), (1, 1)),
    ]


def _solve_with_metadata(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    pattern_known = (g != BG) & (g != FRAME)
    assert pattern_known.any(), "expected a visible non-frame pattern"

    pat_box = _bbox(pattern_known)
    assert pat_box is not None
    pr0, pc0, pr1, pc1 = pat_box

    a2 = _best_vertical_axis(g, pattern_known)
    b2 = _best_horizontal_axis(g, pattern_known)

    out = g.copy()
    repaired: dict[tuple[int, int], int] = {}

    changed = True
    for _ in range(6):
        if not changed:
            break
        changed = False
        for r in range(pr0, pr1 + 1):
            for c in range(pc0, pc1 + 1):
                if int(out[r, c]) != BG:
                    continue
                candidates = _mirror_candidates(out, r, c, a2, b2)
                if candidates:
                    color = int(Counter(candidates).most_common(1)[0][0])
                    out[r, c] = color
                    repaired[(r, c)] = color
                    changed = True

    assert repaired, "expected at least one repaired cell"

    frame_mask = g == FRAME
    frame_box = _bbox(frame_mask)
    assert frame_box is not None

    trigger_cells: dict[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]] = {}
    for corner, direction in _frame_corners(frame_box):
        cr, cc = corner
        dr, dc = direction
        rr, cc2 = cr + dr, cc + dc
        while _in_bounds(g.shape, rr, cc2):
            if (rr, cc2) in repaired:
                trigger_cells[(rr, cc2)] = (corner, direction)
                break
            if int(g[rr, cc2]) != BG:
                break
            rr += dr
            cc2 += dc

    for (r, c), (corner, direction) in trigger_cells.items():
        color = int(repaired[(r, c)])
        rr, cc = corner
        dr, dc = direction
        rr += dr
        cc += dc
        while _in_bounds(g.shape, rr, cc):
            if int(g[rr, cc]) != BG:
                break
            out[rr, cc] = color
            rr += dr
            cc += dc

    return out, {"repaired": repaired, "triggers": trigger_cells, "pattern_box": pat_box, "axes": (a2, b2)}


def main(grid: np.ndarray) -> np.ndarray:
    """Repair a doubly symmetric mosaic and emit corner-gated diagonal rays from an offset hollow frame.

    Input structure:
    - A black background contains one multicolor pattern with both horizontal and vertical reflection symmetry.
    - Several pattern pixels are erased to black, and a separate one-cell-thick yellow hollow rectangle sits off-center.
    - The frame does not overwrite the visible pattern, but some erased pixels lie on outward 45-degree rays from frame corners.

    Rule:
    - Infer the horizontal and vertical symmetry axes from the visible non-black, non-yellow pattern pixels.
    - Restore each black missing pattern cell from its mirrored counterparts across the inferred axes.
    - For each repaired cell that lies on an outward diagonal ray of a frame corner, use that repaired color to paint every background cell on that ray, starting just beyond the corner and stopping before any originally non-background obstacle.

    Key insight:
    - The missing-pixel colors come only from symmetry, while the offset frame decides which repaired cells additionally activate diagonal ray painting; neither subrule is sufficient on its own.
    """
    solved, _ = _solve_with_metadata(grid)
    return solved


def _build_symmetric_pattern(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    patch = np.zeros((h, w), dtype=int)
    used = set()
    for r in range((h + 1) // 2):
        for c in range((w + 1) // 2):
            orbit = {
                (r, c),
                (r, w - 1 - c),
                (h - 1 - r, c),
                (h - 1 - r, w - 1 - c),
            }
            if orbit & used:
                continue
            color = int(PATTERN_COLORS[int(rng.integers(0, len(PATTERN_COLORS)))])
            for rr, cc in orbit:
                patch[rr, cc] = color
            used |= orbit

    unique = set(map(int, np.unique(patch)))
    if len(unique) < 3:
        anchors = [
            (0, 0),
            (0, w // 2),
            (h // 2, 0),
            (h // 2, w // 2),
        ]
        for idx, (r, c) in enumerate(anchors):
            color = int(PATTERN_COLORS[idx % len(PATTERN_COLORS)])
            orbit = {
                (r, c),
                (r, w - 1 - c),
                (h - 1 - r, c),
                (h - 1 - r, w - 1 - c),
            }
            for rr, cc in orbit:
                patch[rr, cc] = color
    return patch


def _build_outline_pattern(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    patch = np.zeros((h, w), dtype=int)
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    used = set()
    for r in range((h + 1) // 2):
        for c in range((w + 1) // 2):
            orbit = {
                (r, c),
                (r, w - 1 - c),
                (h - 1 - r, c),
                (h - 1 - r, w - 1 - c),
            }
            occupied = [(rr, cc) for rr, cc in orbit if border[rr, cc]]
            if not occupied or orbit & used:
                continue
            color = int(PATTERN_COLORS[int(rng.integers(0, len(PATTERN_COLORS)))])
            for rr, cc in occupied:
                patch[rr, cc] = color
            used |= orbit

    unique = set(int(v) for v in np.unique(patch) if int(v) != BG)
    if len(unique) < 3:
        accent_cells = [(0, 0), (0, w // 2), (h // 2, 0), (h - 1, w // 2)]
        for idx, (r, c) in enumerate(accent_cells):
            orbit = {
                (r, c),
                (r, w - 1 - c),
                (h - 1 - r, c),
                (h - 1 - r, w - 1 - c),
            }
            color = int(PATTERN_COLORS[idx % len(PATTERN_COLORS)])
            for rr, cc in orbit:
                if border[rr, cc]:
                    patch[rr, cc] = color
    return patch


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME


def _trace_ray_cells(shape: tuple[int, int], corner: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    rr, cc = corner[0] + dr, corner[1] + dc
    cells: list[tuple[int, int]] = []
    while _in_bounds(shape, rr, cc):
        cells.append((rr, cc))
        rr += dr
        cc += dc
    return cells


def _candidate_frame_boxes(
    shape: tuple[int, int],
    pattern_box: tuple[int, int, int, int],
    desired_triggers: int,
) -> list[tuple[int, int, int, int]]:
    h, w = shape
    pr0, pc0, pr1, pc1 = pattern_box
    boxes: list[tuple[int, int, int, int]] = []
    for fh in range(5, 10):
        for fw in range(5, 10):
            if fh >= h or fw >= w:
                continue
            for r0 in range(1, h - fh):
                r1 = r0 + fh - 1
                for c0 in range(1, w - fw):
                    c1 = c0 + fw - 1
                    if not (r1 < pr0 or r0 > pr1 or c1 < pc0 or c0 > pc1):
                        continue
                    box = (r0, c0, r1, c1)
                    corners = _frame_corners(box)
                    hit_count = 0
                    for corner, direction in corners:
                        ray = _trace_ray_cells(shape, corner, direction)
                        if not ray:
                            continue
                        if any(pr0 <= rr <= pr1 and pc0 <= cc <= pc1 for rr, cc in ray):
                            hit_count += 1
                    if hit_count >= desired_triggers:
                        boxes.append(box)
    return boxes


def _find_corner_targets(
    output_grid: np.ndarray,
    pattern_box: tuple[int, int, int, int],
    frame_box: tuple[int, int, int, int],
) -> dict[tuple[int, int], list[tuple[int, int]]]:
    pr0, pc0, pr1, pc1 = pattern_box
    candidates: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for corner, direction in _frame_corners(frame_box):
        ray = _trace_ray_cells(output_grid.shape, corner, direction)
        found: list[tuple[int, int]] = []
        seen_pattern = False
        for idx, (r, c) in enumerate(ray):
            inside_pattern = pr0 <= r <= pr1 and pc0 <= c <= pc1
            if inside_pattern:
                if seen_pattern:
                    found = []
                    break
                seen_pattern = True
                if idx < 1 or idx == len(ray) - 1:
                    found = []
                    break
                found = [(r, c)]
            elif seen_pattern and int(output_grid[r, c]) != BG:
                found = []
                break
        if found:
            candidates[corner] = found
    return candidates


def _all_corner_ray_cells(shape: tuple[int, int], frame_box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for corner, direction in _frame_corners(frame_box):
        cells.update(_trace_ray_cells(shape, corner, direction))
    return cells


def _orbit_key(cell: tuple[int, int], pattern_box: tuple[int, int, int, int]) -> tuple[tuple[int, int], ...]:
    r, c = cell
    pr0, pc0, pr1, pc1 = pattern_box
    orbit = {
        (r, c),
        (r, pc0 + pc1 - c),
        (pr0 + pr1 - r, c),
        (pr0 + pr1 - r, pc0 + pc1 - c),
    }
    return tuple(sorted((int(rr), int(cc)) for rr, cc in orbit))


def _choose_nontrigger_holes(
    rng: np.random.Generator,
    pattern_box: tuple[int, int, int, int],
    ray_cells: set[tuple[int, int]],
    chosen: set[tuple[int, int]],
    count: int,
) -> list[tuple[int, int]]:
    pr0, pc0, pr1, pc1 = pattern_box
    used_orbits = {_orbit_key(cell, pattern_box) for cell in chosen}
    cells = [
        (r, c)
        for r in range(pr0 + 1, pr1)
        for c in range(pc0 + 1, pc1)
        if (r, c) not in ray_cells and (r, c) not in chosen
    ]
    rng.shuffle(cells)
    picked: list[tuple[int, int]] = []
    for cell in cells:
        orbit = _orbit_key(cell, pattern_box)
        if orbit in used_orbits:
            continue
        used_orbits.add(orbit)
        picked.append(cell)
        if len(picked) == count:
            break
    return picked


def _make_example() -> tuple[np.ndarray, np.ndarray] | None:
    rng = np.random.default_rng()
    h = int(rng.integers(17, 25))
    w = int(rng.integers(17, 25))
    ph = int(rng.choice([9, 11, 13]))
    pw = int(rng.choice([9, 11, 13]))
    if ph >= h - 2 or pw >= w - 2:
        return None

    grid = np.zeros((h, w), dtype=int)
    pattern = _build_outline_pattern(rng, ph, pw)

    min_r0 = max(1, h // 2 - ph // 2 - 1)
    max_r0 = min(h - ph - 1, h // 2 - ph // 2 + 1)
    min_c0 = max(1, w // 2 - pw // 2 - 1)
    max_c0 = min(w - pw - 1, w // 2 - pw // 2 + 1)
    if min_r0 > max_r0 or min_c0 > max_c0:
        return None
    pr0 = int(rng.integers(min_r0, max_r0 + 1))
    pc0 = int(rng.integers(min_c0, max_c0 + 1))
    pr1 = pr0 + ph - 1
    pc1 = pc0 + pw - 1
    grid[pr0 : pr1 + 1, pc0 : pc1 + 1] = pattern
    pattern_box = (pr0, pc0, pr1, pc1)

    desired_triggers = 3 if int(rng.integers(0, 100)) < 40 else 2
    fh = int(rng.choice([5, 7]))
    fw = int(rng.choice([5, 7]))
    fr_min = pr0 + 2
    fc_min = pc0 + 2
    fr_max = pr1 - fh - 1
    fc_max = pc1 - fw - 1
    if fr_min > fr_max or fc_min > fc_max:
        return None

    center_frame = (pr0 + pr1 - fh + 1) // 2, (pc0 + pc1 - fw + 1) // 2
    frame_boxes: list[tuple[int, int, int, int]] = []
    for fr0 in range(fr_min, fr_max + 1):
        for fc0 in range(fc_min, fc_max + 1):
            if (fr0, fc0) == center_frame:
                continue
            frame_boxes.append((fr0, fc0, fr0 + fh - 1, fc0 + fw - 1))
    random.shuffle(frame_boxes)

    border_cells = {(r, c) for r in range(pr0, pr1 + 1) for c in range(pc0, pc1 + 1) if int(grid[r, c]) != BG}

    for frame_box in frame_boxes:
        base = grid.copy()
        _draw_frame(base, frame_box)
        corner_targets: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for corner, direction in _frame_corners(frame_box):
            ray = _trace_ray_cells(base.shape, corner, direction)
            hit = None
            for idx, cell in enumerate(ray):
                if cell in border_cells:
                    if idx < 1 or idx == len(ray) - 1:
                        hit = None
                    else:
                        hit = cell
                    break
            if hit is not None:
                corner_targets[corner] = [hit]
        if len(corner_targets) < desired_triggers:
            continue

        corners = list(corner_targets.keys())
        rng.shuffle(corners)
        chosen_corners: list[tuple[int, int]] = []
        trigger_holes: list[tuple[int, int]] = []
        used_orbits: set[tuple[tuple[int, int], ...]] = set()
        for corner in corners:
            cell = corner_targets[corner][0]
            orbit = _orbit_key(cell, pattern_box)
            if orbit in used_orbits:
                continue
            used_orbits.add(orbit)
            chosen_corners.append(corner)
            trigger_holes.append(cell)
            if len(chosen_corners) == desired_triggers:
                break
        if len(chosen_corners) != desired_triggers:
            continue

        ray_cells = _all_corner_ray_cells(grid.shape, frame_box)
        total_holes = desired_triggers + int(rng.integers(1, 3))
        total_holes = min(5, total_holes)
        nontrigger_needed = total_holes - desired_triggers
        used_orbits = {_orbit_key(cell, pattern_box) for cell in trigger_holes}
        pool = [cell for cell in border_cells if cell not in ray_cells and cell not in set(trigger_holes)]
        rng.shuffle(pool)
        nontrigger_holes: list[tuple[int, int]] = []
        for cell in pool:
            orbit = _orbit_key(cell, pattern_box)
            if orbit in used_orbits:
                continue
            used_orbits.add(orbit)
            nontrigger_holes.append(cell)
            if len(nontrigger_holes) == nontrigger_needed:
                break
        if len(nontrigger_holes) != nontrigger_needed:
            continue

        inp = base.copy()
        all_holes = trigger_holes + nontrigger_holes
        for r, c in all_holes:
            inp[r, c] = BG

        solved, meta = _solve_with_metadata(inp)
        if len(meta["triggers"]) != desired_triggers:
            continue
        if set(meta["triggers"]) != set(trigger_holes):
            continue
        if len(all_holes) < 3 or len(all_holes) > 5:
            continue
        if len(set(map(int, np.unique(base[pr0 : pr1 + 1, pc0 : pc1 + 1])))) < 3:
            continue
        if np.array_equal(solved, base):
            continue
        return inp, solved
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        pair = _make_example()
        if pair is not None:
            return pair[0]
    raise RuntimeError("failed to generate a valid input for puzzle 9")
