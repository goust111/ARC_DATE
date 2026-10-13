import random

import numpy as np


BACKGROUND = 0
OCCLUDER = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 800
SYMMETRY_NAMES = ("vm", "hm", "r180", "dmain", "danti")


def _valid_symmetries(shape: tuple[int, int]) -> list[str]:
    h, w = map(int, shape)
    out = ["vm", "hm", "r180"]
    if h == w:
        out.extend(["dmain", "danti"])
    return out


def _map_coord(symmetry: str, r: int, c: int, shape: tuple[int, int]) -> tuple[int, int]:
    h, w = map(int, shape)
    if symmetry == "vm":
        return r, w - 1 - c
    if symmetry == "hm":
        return h - 1 - r, c
    if symmetry == "r180":
        return h - 1 - r, w - 1 - c
    if symmetry == "dmain":
        return c, r
    if symmetry == "danti":
        return h - 1 - c, w - 1 - r
    raise ValueError(f"unknown symmetry: {symmetry}")


def _orbits(shape: tuple[int, int], symmetry: str) -> list[tuple[tuple[int, int], ...]]:
    h, w = map(int, shape)
    seen: set[tuple[int, int]] = set()
    out: list[tuple[tuple[int, int], ...]] = []
    for r in range(h):
        for c in range(w):
            if (r, c) in seen:
                continue
            orbit = {(r, c)}
            mate = _map_coord(symmetry, r, c, shape)
            orbit.add(mate)
            canon = tuple(sorted(orbit))
            seen.update(canon)
            out.append(canon)
    return out


def _satisfies_symmetry(grid: np.ndarray, symmetry: str) -> bool:
    h, w = map(int, grid.shape)
    for r in range(h):
        for c in range(w):
            rr, cc = _map_coord(symmetry, r, c, grid.shape)
            if int(grid[r, c]) != int(grid[rr, cc]):
                return False
    return True


def _merge_panels(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    assert left.shape == right.shape
    h, w = map(int, left.shape)
    merged = np.full((h, w), -1, dtype=int)
    unknown = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            lv = int(left[r, c])
            rv = int(right[r, c])
            if lv != OCCLUDER and rv != OCCLUDER:
                assert lv == rv
                merged[r, c] = lv
            elif lv != OCCLUDER:
                merged[r, c] = lv
            elif rv != OCCLUDER:
                merged[r, c] = rv
            else:
                unknown[r, c] = True
    return merged, unknown


def _consistent_with_partial(merged: np.ndarray, unknown: np.ndarray, symmetry: str) -> bool:
    for orbit in _orbits(merged.shape, symmetry):
        visible = {int(merged[r, c]) for r, c in orbit if not unknown[r, c]}
        if len(visible) > 1:
            return False
        if any(bool(unknown[r, c]) for r, c in orbit) and not visible:
            return False
    return True


def _reconstruct_from_partial(merged: np.ndarray, unknown: np.ndarray, symmetry: str) -> np.ndarray:
    out = merged.copy()
    for orbit in _orbits(out.shape, symmetry):
        visible = {int(out[r, c]) for r, c in orbit if not unknown[r, c]}
        assert len(visible) == 1
        color = next(iter(visible))
        for r, c in orbit:
            out[r, c] = color
    return out


def _infer_unique_completion(merged: np.ndarray, unknown: np.ndarray) -> np.ndarray:
    candidates: list[np.ndarray] = []
    for symmetry in _valid_symmetries(merged.shape):
        if not _consistent_with_partial(merged, unknown, symmetry):
            continue
        full = _reconstruct_from_partial(merged, unknown, symmetry)
        satisfied = [name for name in _valid_symmetries(full.shape) if _satisfies_symmetry(full, name)]
        if satisfied == [symmetry]:
            candidates.append(full)
    assert len(candidates) == 1
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Complete the original panel by overlaying two occluded copies and using the pattern's unique dihedral symmetry to fill the last blind spots.

    Input structure:
    - The input contains two equal-size panels placed side by side with one full-height background separator column.
    - Both panels are copies of the same multicolor pattern.
    - Grey cells are solid rectangular occluders; the two panels have different occluder placements.

    Rule:
    - Align the two panels cell by cell and keep the non-grey color whenever at least one panel shows it.
    - Cells that are grey in both panels remain temporarily unknown.
    - Among the dihedral symmetries compatible with the panel shape, identify the unique nontrivial symmetry that matches the recovered visible evidence.
    - Fill every doubly occluded cell from its orbit under that symmetry and output one completed panel.

    Key insight:
    - Information arrives in three tiers: visible cells in one panel, complementary exposure in the other panel, and only then symmetry-based recovery for cells hidden in both views.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, total_w = map(int, g.shape)
    assert total_w % 2 == 1
    mid = total_w // 2
    sep_col = g[:, mid]
    assert np.all(sep_col == sep_col[0])
    bg = int(sep_col[0])
    left = g[:, :mid].copy()
    right = g[:, mid + 1 :].copy()
    assert left.shape == right.shape
    assert not np.any(left == bg)
    assert not np.any(right == bg)

    merged, unknown = _merge_panels(left, right)
    assert np.any(unknown)
    return _infer_unique_completion(merged, unknown)


def _generate_pattern(symmetry: str) -> np.ndarray:
    for _ in range(400):
        if symmetry in {"dmain", "danti"}:
            h = w = random.randint(5, 8)
        else:
            h = random.randint(5, 8)
            w = random.randint(5, 8)
            if random.random() < 0.65:
                while h == w:
                    w = random.randint(5, 8)
        colors = random.sample(PALETTE, random.randint(3, 5))
        grid = np.zeros((h, w), dtype=int)
        for orbit in _orbits(grid.shape, symmetry):
            color = random.choice(colors)
            for r, c in orbit:
                grid[r, c] = color
        satisfied = [name for name in _valid_symmetries(grid.shape) if _satisfies_symmetry(grid, name)]
        if satisfied != [symmetry]:
            continue
        if len(set(map(int, grid.flat))) < 3:
            continue
        return grid
    raise RuntimeError("failed to generate a uniquely symmetric pattern")


def _render_input(pattern: np.ndarray, left_mask: np.ndarray, right_mask: np.ndarray) -> np.ndarray:
    left = pattern.copy()
    right = pattern.copy()
    left[left_mask] = OCCLUDER
    right[right_mask] = OCCLUDER
    sep = np.full((pattern.shape[0], 1), BACKGROUND, dtype=int)
    return np.hstack([left, sep, right])


def _semantic_checks(pattern: np.ndarray, left_mask: np.ndarray, right_mask: np.ndarray, symmetry: str) -> bool:
    if np.count_nonzero(left_mask & ~right_mask) == 0:
        return False
    if np.count_nonzero(right_mask & ~left_mask) == 0:
        return False
    if np.count_nonzero(left_mask & right_mask) == 0:
        return False
    if left_mask.mean() < 0.12 or right_mask.mean() < 0.12:
        return False
    if left_mask.mean() > 0.58 or right_mask.mean() > 0.58:
        return False

    left = pattern.copy()
    right = pattern.copy()
    left[left_mask] = OCCLUDER
    right[right_mask] = OCCLUDER
    merged, unknown = _merge_panels(left, right)

    if not np.any(unknown):
        return False
    if int(np.count_nonzero(unknown)) > max(6, pattern.size // 4):
        return False

    left_essential = False
    right_essential = False
    symmetry_essential = False
    for orbit in _orbits(pattern.shape, symmetry):
        if len(orbit) < 2:
            continue
        left_visible = sum(not bool(left_mask[r, c]) for r, c in orbit)
        right_visible = sum(not bool(right_mask[r, c]) for r, c in orbit)
        both_hidden = sum(bool(left_mask[r, c] and right_mask[r, c]) for r, c in orbit)
        if left_visible == 0 and right_visible > 0:
            left_essential = True
        if right_visible == 0 and left_visible > 0:
            right_essential = True
        if both_hidden > 0 and (left_visible + right_visible - both_hidden) > 0:
            symmetry_essential = True
    if not (left_essential and right_essential and symmetry_essential):
        return False

    solutions = []
    for name in _valid_symmetries(pattern.shape):
        if not _consistent_with_partial(merged, unknown, name):
            continue
        full = _reconstruct_from_partial(merged, unknown, name)
        satisfied = [sym for sym in _valid_symmetries(full.shape) if _satisfies_symmetry(full, sym)]
        if satisfied == [name]:
            solutions.append((name, full))
    if len(solutions) != 1:
        return False
    if solutions[0][0] != symmetry:
        return False
    if not np.array_equal(solutions[0][1], pattern):
        return False

    rendered = _render_input(pattern, left_mask, right_mask)
    if not np.array_equal(main(rendered), pattern):
        return False
    return True


def _generate_masks(pattern: np.ndarray, symmetry: str) -> tuple[np.ndarray, np.ndarray]:
    shape = pattern.shape
    pair_orbits = [orbit for orbit in _orbits(shape, symmetry) if len(orbit) == 2]
    assert len(pair_orbits) >= 3

    for _ in range(600):
        left_mask = np.zeros(shape, dtype=bool)
        right_mask = np.zeros(shape, dtype=bool)

        both_orbit, left_orbit, right_orbit = random.sample(pair_orbits, 3)

        hidden_cell, mate_cell = random.choice((both_orbit, both_orbit[::-1]))
        left_mask[hidden_cell] = True
        right_mask[hidden_cell] = True
        for cell in left_orbit:
            left_mask[cell] = True
        for cell in right_orbit:
            right_mask[cell] = True

        if left_mask[mate_cell] and right_mask[mate_cell]:
            if random.random() < 0.5:
                left_mask[mate_cell] = False
            else:
                right_mask[mate_cell] = False

        for mask in (left_mask, right_mask):
            for _ in range(random.randint(2, 5)):
                rh = random.randint(1, max(1, shape[0] // 2))
                rw = random.randint(1, max(1, shape[1] // 2))
                r0 = random.randint(0, shape[0] - rh)
                c0 = random.randint(0, shape[1] - rw)
                mask[r0 : r0 + rh, c0 : c0 + rw] = True

        if _semantic_checks(pattern, left_mask, right_mask, symmetry):
            return left_mask, right_mask
    raise RuntimeError("failed to generate valid occluder masks")


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        symmetry = random.choice(SYMMETRY_NAMES)
        if symmetry in {"dmain", "danti"}:
            pattern = _generate_pattern(symmetry)
        else:
            pattern = _generate_pattern(symmetry)
        left_mask, right_mask = _generate_masks(pattern, symmetry)
        grid = _render_input(pattern, left_mask, right_mask)
        if max(grid.shape) <= 30:
            return grid
    raise RuntimeError("failed to generate a valid input")
