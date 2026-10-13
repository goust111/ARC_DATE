from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np


BG = 0
GUIDE = 1
NEW = 2
OBJ = 5

GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(coords: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = list(coords)
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


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


def _is_connected(coords: set[tuple[int, int]]) -> bool:
    if not coords:
        return False
    start = next(iter(coords))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in coords and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(coords)


def _make_guide_mask(grid: np.ndarray) -> tuple[np.ndarray, list[int], list[int]]:
    h, w = grid.shape
    rows = [r for r in range(1, h - 1) if int(grid[r, 0]) == GUIDE and int(grid[r, w - 1]) == GUIDE]
    cols = [c for c in range(1, w - 1) if int(grid[0, c]) == GUIDE and int(grid[h - 1, c]) == GUIDE]
    mask = np.zeros_like(grid, dtype=bool)
    for r in rows:
        mask[r, :] = True
    for c in cols:
        mask[:, c] = True
    return mask, rows, cols


def _touch_mask(guide_mask: np.ndarray) -> np.ndarray:
    touched = guide_mask.copy()
    touched[1:, :] |= guide_mask[:-1, :]
    touched[:-1, :] |= guide_mask[1:, :]
    touched[:, 1:] |= guide_mask[:, :-1]
    touched[:, :-1] |= guide_mask[:, 1:]
    return touched


def _rotate_point(r: int, c: int, center_r2: int, center_c2: int, iters: int) -> tuple[int, int]:
    x2 = 2 * int(r) - int(center_r2)
    y2 = 2 * int(c) - int(center_c2)
    for _ in range(iters % 4):
        x2, y2 = y2, -x2
    rr2 = x2 + int(center_r2)
    cc2 = y2 + int(center_c2)
    return rr2 // 2, cc2 // 2


def _infer_rotational_completion(coords: np.ndarray, grid_shape: tuple[int, int]) -> set[tuple[int, int]]:
    observed = {tuple(map(int, rc)) for rc in coords}
    h, w = map(int, grid_shape)
    r0, c0, r1, c1 = _bbox(observed)
    best_key = None
    best_added: set[tuple[int, int]] | None = None

    for center_r2 in range(2 * (r0 - 1), 2 * (r1 + 1) + 2):
        for center_c2 in range(2 * (c0 - 1), 2 * (c1 + 1) + 2):
            if (center_r2 - center_c2) % 2 != 0:
                continue
            closure: set[tuple[int, int]] = set()
            preserved = 0
            off_grid = False
            for r, c in observed:
                for k in range(4):
                    rr, cc = _rotate_point(r, c, center_r2, center_c2, k)
                    if not (0 <= rr < h and 0 <= cc < w):
                        off_grid = True
                        break
                    closure.add((rr, cc))
                if off_grid:
                    break
                for k in (1, 2, 3):
                    if _rotate_point(r, c, center_r2, center_c2, k) in observed:
                        preserved += 1
            if off_grid:
                continue

            cr0, cc0, cr1, cc1 = _bbox(closure)
            if cr1 - cr0 != cc1 - cc0:
                continue
            if center_r2 != cr0 + cr1 or center_c2 != cc0 + cc1:
                continue
            if not _is_connected(closure):
                continue

            added = closure - observed
            if len(added) > 4:
                continue

            key = (len(added), -preserved, len(closure), center_r2, center_c2)
            if best_key is None or key < best_key:
                best_key = key
                best_added = added

    if best_added is None:
        return set()
    return best_added


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete only the guide-line-selected grey objects by 90-degree rotational symmetry, marking new cells in red.

    Input structure:
    - The grid has a black background with grey 4-connected objects scattered inside.
    - Blue marker pairs on opposite borders define full horizontal or vertical guide lines.
    - Some grey objects are incomplete versions of 90-degree rotationally symmetric shapes; some are already complete.

    Rule:
    - Extend each border marker pair into a full blue row or column across the grid, but do not overwrite object pixels.
    - A grey object is selected if any of its pixels overlaps a guide line or is 4-neighbor adjacent to one.
    - For each selected object, infer its rotational center and add the missing 90-degree rotational counterparts.
    - Newly added completion pixels become red, while unselected objects and all original grey pixels stay unchanged.

    Key insight:
    - The blue border markers are not objects to transform; they define spatial guide lines that select which damaged symmetric object should be repaired.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    guide_mask, _rows, _cols = _make_guide_mask(g)
    touched_mask = _touch_mask(guide_mask)

    out = g.copy()
    out[guide_mask & (out == BG)] = GUIDE

    for comp in _components_of_color(g, OBJ):
        if not touched_mask[comp[:, 0], comp[:, 1]].any():
            continue
        added = _infer_rotational_completion(comp, g.shape)
        for r, c in added:
            out[r, c] = NEW
    return out


def _line_positions(rng: np.random.Generator, length: int, count: int) -> list[int]:
    positions: list[int] = []
    candidates = list(range(2, length - 2))
    rng.shuffle(candidates)
    for pos in candidates:
        if all(abs(pos - prev) >= 3 for prev in positions):
            positions.append(pos)
            if len(positions) == count:
                break
    return sorted(positions)


def _canonical_orbits(side: int) -> list[tuple[tuple[int, int], ...]]:
    seen: set[frozenset[tuple[int, int]]] = set()
    out: list[tuple[tuple[int, int], ...]] = []
    center2 = side - 1
    for r in range(side):
        for c in range(side):
            orbit = frozenset(_rotate_point(r, c, center2, center2, k) for k in range(4))
            if orbit not in seen:
                seen.add(orbit)
                out.append(tuple(sorted(orbit)))
    return out


def _grow_symmetric_shape(rng: np.random.Generator) -> set[tuple[int, int]] | None:
    side = int(rng.integers(3, 7))
    orbits = _canonical_orbits(side)
    target_area = int(rng.integers(5, min(15, side * side + 1)))

    center_rank = sorted(
        range(len(orbits)),
        key=lambda idx: (
            min(abs(2 * r - (side - 1)) + abs(2 * c - (side - 1)) for r, c in orbits[idx]),
            len(orbits[idx]),
        ),
    )
    shape = set(orbits[center_rank[0]])
    used = {center_rank[0]}

    for _ in range(20):
        candidates = []
        for idx, orbit in enumerate(orbits):
            if idx in used:
                continue
            if any(any(abs(r - rr) + abs(c - cc) == 1 for rr, cc in shape) for r, c in orbit):
                candidates.append(idx)
        if not candidates:
            break

        rng.shuffle(candidates)
        chosen = None
        for idx in candidates:
            new_area = len(shape) + len(orbits[idx])
            if new_area <= target_area or not chosen:
                chosen = idx
                if new_area <= target_area:
                    break
        assert chosen is not None
        used.add(chosen)
        shape.update(orbits[chosen])
        if len(shape) >= target_area:
            break

    if not _is_connected(shape):
        return None

    r0, c0, r1, c1 = _bbox(shape)
    if r1 - r0 != c1 - c0:
        return None

    norm = {(r - r0, c - c0) for r, c in shape}
    nr0, nc0, nr1, nc1 = _bbox(norm)
    side2 = nr1 - nr0 + 1
    if side2 < 3 or side2 > 6:
        return None
    if len(norm) < 5 or len(norm) > 14:
        return None
    if len({r for r, _ in norm}) < 3 or len({c for _, c in norm}) < 3:
        return None
    return norm


def _completion_from_local(local_coords: set[tuple[int, int]], side: int) -> set[tuple[int, int]]:
    placed = np.array([(r + 2, c + 2) for r, c in sorted(local_coords)], dtype=int)
    added = _infer_rotational_completion(placed, (side + 4, side + 4))
    return {(r - 2, c - 2) for r, c in added}


def _make_incomplete_variant(
    full_coords: set[tuple[int, int]], rng: np.random.Generator
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    current = set(full_coords)
    full_box = _bbox(full_coords)
    max_remove = min(4, len(full_coords) - 4)
    target_remove = int(rng.integers(1, max_remove + 1))
    removed: set[tuple[int, int]] = set()

    for _ in range(target_remove):
        candidates = []
        for cell in sorted(current):
            proposal = current - {cell}
            if _bbox(proposal) != full_box or not _is_connected(proposal):
                continue
            candidates.append(cell)
        if not candidates:
            break
        choice = candidates[int(rng.integers(len(candidates)))]
        current.remove(choice)
        removed.add(choice)

    if not removed:
        return None

    side = _bbox(full_coords)[2] - _bbox(full_coords)[0] + 1
    inferred = _completion_from_local(current, side)
    if inferred != removed:
        return None
    return current, removed


def _generate_object(
    rng: np.random.Generator, complete: bool
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    for _ in range(80):
        full = _grow_symmetric_shape(rng)
        if full is None:
            continue
        if complete:
            return full, set()
        made = _make_incomplete_variant(full, rng)
        if made is not None:
            return made
    return None


def _touches_any_line(cell: tuple[int, int], rows: list[int], cols: list[int]) -> bool:
    r, c = cell
    return any(abs(r - row) <= 1 for row in rows) or any(abs(c - col) <= 1 for col in cols)


def _touches_spec(cell: tuple[int, int], spec: tuple[str, int]) -> bool:
    r, c = cell
    axis, pos = spec
    return abs(r - pos) == 1 if axis == "h" else abs(c - pos) == 1


def _shift(coords: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in coords}


def _can_place(
    full_global: set[tuple[int, int]],
    input_global: set[tuple[int, int]],
    placed_full: list[set[tuple[int, int]]],
    rows: list[int],
    cols: list[int],
    spec: tuple[str, int] | None,
    h: int,
    w: int,
) -> bool:
    for r, c in full_global:
        if not (1 <= r < h - 1 and 1 <= c < w - 1):
            return False
        if r in rows or c in cols:
            return False

    for other in placed_full:
        if full_global & other:
            return False
        for r, c in full_global:
            if any((nr, nc) in other for nr, nc in _neighbors4(r, c)):
                return False

    if spec is None:
        return not any(_touches_any_line(cell, rows, cols) for cell in input_global)

    axis, pos = spec
    if axis == "h":
        if not (max(r for r, _ in full_global) < pos or min(r for r, _ in full_global) > pos):
            return False
    else:
        if not (max(c for _, c in full_global) < pos or min(c for _, c in full_global) > pos):
            return False

    if not any(_touches_spec(cell, spec) for cell in input_global):
        return False

    for cell in input_global:
        if _touches_any_line(cell, rows, cols) and not _touches_spec(cell, spec):
            return False
    return True


def _find_placement(
    rng: np.random.Generator,
    full_local: set[tuple[int, int]],
    input_local: set[tuple[int, int]],
    placed_full: list[set[tuple[int, int]]],
    rows: list[int],
    cols: list[int],
    spec: tuple[str, int] | None,
    h: int,
    w: int,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    br0, bc0, br1, bc1 = _bbox(full_local)
    shape_h = br1 - br0 + 1
    shape_w = bc1 - bc0 + 1

    candidates = [(r0, c0) for r0 in range(1, h - shape_h) for c0 in range(1, w - shape_w)]
    rng.shuffle(candidates)
    for r0, c0 in candidates:
        full_global = _shift(full_local, r0, c0)
        input_global = _shift(input_local, r0, c0)
        if _can_place(full_global, input_global, placed_full, rows, cols, spec, h, w):
            return full_global, input_global
    return None


def _spec_pool(rows: list[int], cols: list[int]) -> list[tuple[str, int]]:
    return [("h", row) for row in rows] + [("v", col) for col in cols]


def _plan_objects(rng: np.random.Generator) -> list[dict]:
    total = int(rng.choice([4, 5]))
    if total == 4:
        touched_incomplete = 2
        complete_total = 1
    else:
        options = [(2, 1), (2, 2), (3, 1)]
        touched_incomplete, complete_total = options[int(rng.integers(len(options)))]
    untouched_incomplete = total - touched_incomplete - complete_total
    assert untouched_incomplete >= 1

    touched_complete = int(complete_total > 0 and rng.random() < 0.5)
    untouched_complete = complete_total - touched_complete

    plan: list[dict] = []
    for _ in range(touched_incomplete):
        plan.append({"complete": False, "needs_touch": True})
    for _ in range(untouched_incomplete):
        plan.append({"complete": False, "needs_touch": False})
    for _ in range(touched_complete):
        plan.append({"complete": True, "needs_touch": True})
    for _ in range(untouched_complete):
        plan.append({"complete": True, "needs_touch": False})
    rng.shuffle(plan)
    return plan


def _build_expected(
    grid: np.ndarray,
    rows: list[int],
    cols: list[int],
    objects: list[dict],
) -> np.ndarray:
    out = grid.copy()
    for r in rows:
        for c in range(out.shape[1]):
            if int(out[r, c]) == BG:
                out[r, c] = GUIDE
    for c in cols:
        for r in range(out.shape[0]):
            if int(out[r, c]) == BG:
                out[r, c] = GUIDE
    for obj in objects:
        if obj["needs_touch"] and not obj["complete"]:
            for r, c in obj["full_global"] - obj["input_global"]:
                out[r, c] = NEW
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(15, 20))
    w = int(rng.integers(15, 20))

    line_options = [(2, 0), (1, 1), (0, 2), (2, 1), (1, 2)]
    h_lines, v_lines = line_options[int(rng.integers(len(line_options)))]
    rows = _line_positions(rng, h, h_lines)
    cols = _line_positions(rng, w, v_lines)
    if len(rows) != h_lines or len(cols) != v_lines:
        return None

    specs = _spec_pool(rows, cols)
    if len(specs) < 2:
        return None

    plan = _plan_objects(rng)
    touched_indices = [i for i, item in enumerate(plan) if item["needs_touch"]]
    assigned_specs: list[tuple[str, int] | None] = [None] * len(plan)
    distinct = min(len(specs), len(touched_indices), 2)
    if distinct == 2:
        chosen = list(specs)
        rng.shuffle(chosen)
        for idx, spec in zip(touched_indices[:2], chosen[:2]):
            assigned_specs[idx] = spec
        for idx in touched_indices[2:]:
            assigned_specs[idx] = specs[int(rng.integers(len(specs)))]
    else:
        for idx in touched_indices:
            assigned_specs[idx] = specs[int(rng.integers(len(specs)))]

    objects: list[dict] = []
    placed_full: list[set[tuple[int, int]]] = []
    for i, item in enumerate(plan):
        generated = _generate_object(rng, complete=item["complete"])
        if generated is None:
            return None
        input_local, removed = generated
        full_local = set(input_local) | removed

        placement = _find_placement(
            rng=rng,
            full_local=full_local,
            input_local=input_local,
            placed_full=placed_full,
            rows=rows,
            cols=cols,
            spec=assigned_specs[i],
            h=h,
            w=w,
        )
        if placement is None:
            return None

        full_global, input_global = placement
        placed_full.append(full_global)
        objects.append(
            {
                "complete": item["complete"],
                "needs_touch": item["needs_touch"],
                "full_global": full_global,
                "input_global": input_global,
            }
        )

    grid = np.full((h, w), BG, dtype=int)
    for obj in objects:
        for r, c in obj["input_global"]:
            grid[r, c] = OBJ
    for r in rows:
        grid[r, 0] = GUIDE
        grid[r, w - 1] = GUIDE
    for c in cols:
        grid[0, c] = GUIDE
        grid[h - 1, c] = GUIDE

    guide_mask, _, _ = _make_guide_mask(grid)
    touched_mask = _touch_mask(guide_mask)
    comps = _components_of_color(grid, OBJ)
    touched_now = sum(int(touched_mask[comp[:, 0], comp[:, 1]].any()) for comp in comps)
    expected_touched = sum(int(obj["needs_touch"]) for obj in objects)
    if touched_now != expected_touched:
        return None

    expected = _build_expected(grid, rows, cols, objects)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    if np.count_nonzero(actual == NEW) == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
