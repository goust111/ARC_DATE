from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

MAX_GENERATION_ATTEMPTS = 500


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _orbit_cells(row: int, col: int, axis2: int, width: int) -> tuple[tuple[int, int], ...] | None:
    mirror_col = int(axis2) - int(col)
    if not (0 <= mirror_col < width):
        return None
    if mirror_col == col:
        return ((int(row), int(col)),)
    left = min(int(col), int(mirror_col))
    right = max(int(col), int(mirror_col))
    return ((int(row), left), (int(row), right))


def _axis_score(mask: np.ndarray, axis2: int) -> int | None:
    height, width = mask.shape
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    paired = 0
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        mirror_col = int(axis2) - cc
        if not (0 <= mirror_col < width):
            return None
        if bool(mask[rr, mirror_col]):
            paired += 1
    return int(paired)


def _best_axes(mask: np.ndarray) -> list[int]:
    _, width = mask.shape
    scores: dict[int, int] = {}
    for axis2 in range(2 * width - 1):
        score = _axis_score(mask, axis2)
        if score is not None:
            scores[int(axis2)] = int(score)
    if not scores:
        return []
    best = max(scores.values())
    return [axis2 for axis2, score in scores.items() if score == best]


def _exact_axes(mask: np.ndarray) -> list[int]:
    total = int(mask.sum())
    return [axis2 for axis2 in _best_axes(mask) if _axis_score(mask, axis2) == total]


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    queue: deque[tuple[int, int]] = deque([start])
    height, width = mask.shape
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            if not bool(mask[nr, nc]) or (nr, nc) in seen:
                continue
            seen.add((nr, nc))
            queue.append((nr, nc))
    return len(seen) == int(mask.sum())


def _find_separator(grid: np.ndarray) -> int:
    full_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert len(full_cols) == 1, f"expected one full-height grey separator column, found {full_cols}"
    return int(full_cols[0])


def _split_panels(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sep = _find_separator(grid)
    left = np.asarray(grid[:, :sep], dtype=int)
    right = np.asarray(grid[:, sep + 1 :], dtype=int)
    assert left.shape == right.shape and left.size > 0, "panels must be equal-size and non-empty"
    valid_colors = {BLACK, GREEN, GREY}
    assert set(map(int, np.unique(grid))) <= valid_colors, "input should only use black/green/grey"
    return left, right


def _infer_axis(merged_mask: np.ndarray) -> int:
    axes = _best_axes(merged_mask)
    assert len(axes) == 1, f"expected a unique best vertical axis, found {axes}"
    return int(axes[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge two separator-split partial panels, infer the shared vertical symmetry axis, and recolor by source certainty.

    Input structure:
    - The grid has two equal black/green panels separated by one full-height grey column.
    - Both panels depict the same green figure on aligned coordinates, but different green pixels were erased to black.
    - A location can therefore be visible in both panels, in exactly one panel, or in neither panel.

    Rule:
    - Remove the grey separator and compare the two panels cell by cell.
    - Build the merged visible-green mask and find the unique vertical reflection axis that maximizes mirrored visible support among those green cells.
    - Any still-black location whose mirror across that axis is visible green is reconstructed as a missing symmetric counterpart.
    - Output one panel where both-source cells are green, one-source cells are yellow, reconstructed cells are red, and all other cells are black.

    Key insight:
    - Neither panel alone is sufficient; only the merged evidence reveals the symmetry axis, and that axis in turn identifies the pixels that both panels omitted.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    left, right = _split_panels(g)
    left_mask = left == GREEN
    right_mask = right == GREEN

    dual = left_mask & right_mask
    merged = left_mask | right_mask
    axis2 = _infer_axis(merged)

    out = np.zeros(left.shape, dtype=int)
    out[dual] = GREEN
    out[merged & ~dual] = YELLOW

    height, width = left.shape
    missing = ~merged
    for r, c in np.argwhere(missing):
        rr = int(r)
        cc = int(c)
        mirror_col = int(axis2) - cc
        if 0 <= mirror_col < width and bool(merged[rr, mirror_col]):
            out[rr, cc] = RED

    return out


def _canonical_orbits(mask: np.ndarray, axis2: int) -> list[tuple[tuple[int, int], ...]]:
    height, width = mask.shape
    seen: set[tuple[tuple[int, int], ...]] = set()
    out: list[tuple[tuple[int, int], ...]] = []
    for r in range(height):
        for c in range(width):
            if not bool(mask[r, c]):
                continue
            orbit = _orbit_cells(r, c, axis2, width)
            assert orbit is not None
            if not all(bool(mask[rr, cc]) for rr, cc in orbit):
                continue
            if orbit in seen:
                continue
            seen.add(orbit)
            out.append(orbit)
    return out


def _grow_full_mask(height: int, width: int, axis2: int) -> np.ndarray | None:
    target_area = random.randint(max(12, height + 2), min(24, (height * width) // 2))
    mask = np.zeros((height, width), dtype=bool)

    seed_rows = [r for r in range(max(0, height // 2 - 1), min(height, height // 2 + 2))]
    seed_row = random.choice(seed_rows)
    if axis2 % 2 == 0:
        seed_col = axis2 // 2
    else:
        seed_col = axis2 // 2
    orbit = _orbit_cells(seed_row, seed_col, axis2, width)
    assert orbit is not None
    for rr, cc in orbit:
        mask[rr, cc] = True

    max_area = min(height * width, target_area + 2)
    while int(mask.sum()) < target_area:
        frontier: list[tuple[tuple[int, int], ...]] = []
        used: set[tuple[tuple[int, int], ...]] = set()
        for r, c in np.argwhere(mask):
            for nr, nc in _neighbors4(int(r), int(c)):
                if not (0 <= nr < height and 0 <= nc < width):
                    continue
                if bool(mask[nr, nc]):
                    continue
                cand = _orbit_cells(nr, nc, axis2, width)
                if cand is None or cand in used:
                    continue
                used.add(cand)
                frontier.append(cand)

        if not frontier:
            break

        random.shuffle(frontier)
        placed = False
        for cand in frontier:
            added = sum(1 for rr, cc in cand if not bool(mask[rr, cc]))
            if int(mask.sum()) + added > max_area:
                continue
            for rr, cc in cand:
                mask[rr, cc] = True
            placed = True
            break
        if not placed:
            break

    if int(mask.sum()) < 12:
        return None
    if len(np.unique(np.argwhere(mask)[:, 0])) < 4:
        return None
    if len(np.unique(np.argwhere(mask)[:, 1])) < 4:
        return None
    if not _connected(mask):
        return None
    if _exact_axes(mask) != [int(axis2)]:
        return None
    return mask


def _select_hidden_cells(full_mask: np.ndarray, axis2: int) -> set[tuple[int, int]] | None:
    pair_orbits = [orbit for orbit in _canonical_orbits(full_mask, axis2) if len(orbit) == 2]
    if len(pair_orbits) < 4:
        return None

    upper = min(len(pair_orbits) - 1, max(2, len(pair_orbits) // 3 + 1))
    count = random.randint(2, upper)
    hidden: set[tuple[int, int]] = set()
    for orbit in random.sample(pair_orbits, count):
        hidden.add(random.choice(orbit))

    rows = {r for r, _ in hidden}
    cols = {c for _, c in hidden}
    if len(rows) < 2 and len(cols) < 2:
        return None
    return hidden


def _build_panels(merged_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    height, width = merged_mask.shape
    left = np.zeros((height, width), dtype=int)
    right = np.zeros((height, width), dtype=int)
    visible = [tuple(map(int, coord)) for coord in np.argwhere(merged_mask)]
    random.shuffle(visible)
    if len(visible) < 8:
        return None

    categories: dict[tuple[int, int], str] = {}
    dual_pick = set(random.sample(visible, k=random.randint(1, max(1, len(visible) // 4))))
    for cell in visible:
        if cell in dual_pick:
            categories[cell] = "dual"
        else:
            categories[cell] = random.choice(("left", "right"))

    if all(tag != "left" for tag in categories.values()):
        categories[random.choice(visible)] = "left"
    if all(tag != "right" for tag in categories.values()):
        categories[random.choice(visible)] = "right"
    if all(tag != "dual" for tag in categories.values()):
        categories[random.choice(visible)] = "dual"

    for (r, c), tag in categories.items():
        if tag in ("dual", "left"):
            left[r, c] = GREEN
        if tag in ("dual", "right"):
            right[r, c] = GREEN

    left_mask = left == GREEN
    right_mask = right == GREEN
    if int(left_mask.sum()) < 4 or int(right_mask.sum()) < 4:
        return None
    return left, right


def _compose_input(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    height, width = left.shape
    grid = np.zeros((height, width * 2 + 1), dtype=int)
    grid[:, :width] = left
    grid[:, width] = GREY
    grid[:, width + 1 :] = right
    return grid


def _expected_output(left: np.ndarray, right: np.ndarray, axis2: int) -> np.ndarray:
    left_mask = left == GREEN
    right_mask = right == GREEN
    merged = left_mask | right_mask
    dual = left_mask & right_mask

    out = np.zeros(left.shape, dtype=int)
    out[dual] = GREEN
    out[merged & ~dual] = YELLOW
    for r, c in np.argwhere(~merged):
        rr = int(r)
        cc = int(c)
        mirror_col = int(axis2) - cc
        if 0 <= mirror_col < left.shape[1] and bool(merged[rr, mirror_col]):
            out[rr, cc] = RED
    return out


def _semantic_check(full_mask: np.ndarray, left: np.ndarray, right: np.ndarray, axis2: int) -> bool:
    left_mask = left == GREEN
    right_mask = right == GREEN
    merged = left_mask | right_mask

    if _best_axes(merged) != [int(axis2)]:
        return False
    if len(_best_axes(left_mask)) == 1:
        return False
    if len(_best_axes(right_mask)) == 1:
        return False
    if _exact_axes(left_mask):
        return False
    if _exact_axes(right_mask):
        return False

    expected_full = merged.copy()
    for r, c in np.argwhere(~merged):
        rr = int(r)
        cc = int(c)
        mirror_col = int(axis2) - cc
        if 0 <= mirror_col < merged.shape[1] and bool(merged[rr, mirror_col]):
            expected_full[rr, cc] = True
    if not np.array_equal(expected_full, full_mask):
        return False

    out = _expected_output(left, right, axis2)
    if int(np.sum(out == RED)) < 2:
        return False
    if not np.any(out == GREEN):
        return False
    if not np.any(out == YELLOW):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(8, 12)
    width = random.randint(8, 12)
    if width * 2 + 1 > 30:
        return None

    if random.random() < 0.5:
        axis_col = random.randint(2, width - 3)
        axis2 = 2 * axis_col
    else:
        axis_left = random.randint(1, width - 3)
        axis2 = 2 * axis_left + 1

    full_mask = _grow_full_mask(height, width, axis2)
    if full_mask is None:
        return None

    hidden = _select_hidden_cells(full_mask, axis2)
    if hidden is None:
        return None

    merged = full_mask.copy()
    for r, c in hidden:
        merged[r, c] = False

    panels = _build_panels(merged)
    if panels is None:
        return None
    left, right = panels

    if not _semantic_check(full_mask, left, right, axis2):
        return None

    inp = _compose_input(left, right)
    expected = _expected_output(left, right, axis2)
    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_GENERATION_ATTEMPTS} attempts")
