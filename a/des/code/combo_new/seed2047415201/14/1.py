import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
MAGENTA = 6


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    rows, cols = mask.shape
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        stack = [(sr, sc)]
        seen[sr, sc] = True
        coords = []
        while stack:
            r, c = stack.pop()
            coords.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < rows and 0 <= nc < cols and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        yield np.array(coords, dtype=int)


def _component_mask(coords):
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0) + 1
    mask = np.zeros((int(r1 - r0), int(c1 - c0)), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _symmetry_kind(mask):
    horizontal = bool(np.array_equal(mask, mask[::-1, :]))
    vertical = bool(np.array_equal(mask, mask[:, ::-1]))
    rotation = bool(np.array_equal(mask, mask[::-1, ::-1]))

    if horizontal and not vertical:
        return "H"
    if vertical and not horizontal:
        return "V"
    if rotation and not horizontal and not vertical:
        return "R"
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor red objects by their symmetry type and erase asymmetric singleton noise.

    Input structure:
    - The grid has black background and red 4-connected foreground objects.
    - Every non-singleton red object has exactly one usable symmetry category:
      horizontal reflection only, vertical reflection only, or 180-degree rotation
      without horizontal or vertical reflection.
    - Isolated single red pixels are noise rather than classified objects.

    Rule:
    - Start from an all-black output of the same shape.
    - Repaint horizontally symmetric red objects blue.
    - Repaint vertically symmetric red objects green.
    - Repaint rotation-only red objects magenta.
    - Leave isolated red pixels black.

    Key insight:
    - All input objects have the same red color, so the output color is a label
      derived from the object's geometry rather than its position or input color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(map(int, np.unique(g))) <= {BLACK, RED}

    out = np.full_like(g, BLACK)
    for coords in _components(g == RED):
        if len(coords) == 1:
            continue
        kind = _symmetry_kind(_component_mask(coords))
        assert kind is not None, "expected exactly one supported symmetry type"
        color = {"H": BLUE, "V": GREEN, "R": MAGENTA}[kind]
        out[coords[:, 0], coords[:, 1]] = color
    return out


# --------------- generator ---------------

GRID_ROWS = (20, 30)
GRID_COLS = (20, 30)
EXTRA_OBJECTS_RANGE = (0, 3)
SINGLETON_COUNT_RANGE = (2, 7)

MASK_AREA_MIN = 5
MASK_AREA_MAX = 28
MASK_RETRIES = 120
GROW_STEPS_RANGE = (0, 8)

MIN_L1_BETWEEN_OBJECTS = 3
GENERATION_RETRIES = 500


def _middle_indices(n):
    if n % 2:
        return [n // 2]
    return [n // 2 - 1, n // 2]


def _trim_mask(mask):
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0) + 1
    return mask[r0:r1, c0:c1].copy()


def _is_connected(mask):
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    rows, cols = mask.shape
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < rows and 0 <= nc < cols and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _orbit(kind, shape, r, c):
    rows, cols = shape
    if kind == "H":
        return tuple(sorted({(r, c), (rows - 1 - r, c)}))
    if kind == "V":
        return tuple(sorted({(r, c), (r, cols - 1 - c)}))
    return tuple(sorted({(r, c), (rows - 1 - r, cols - 1 - c)}))


def _frontier_orbits(mask, kind, area_limit):
    rows, cols = mask.shape
    seen = set()
    choices = []
    for r in range(rows):
        for c in range(cols):
            if mask[r, c]:
                continue
            orbit = _orbit(kind, mask.shape, r, c)
            if orbit in seen:
                continue
            seen.add(orbit)
            new_cells = [(rr, cc) for rr, cc in orbit if not mask[rr, cc]]
            if int(mask.sum()) + len(new_cells) > area_limit:
                continue
            touches = False
            for rr, cc in new_cells:
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < rows and 0 <= nc < cols and mask[nr, nc]:
                        touches = True
                        break
                if touches:
                    break
            if touches:
                choices.append(orbit)
    return choices


def _grow_symmetric(mask, kind):
    for _ in range(random.randint(*GROW_STEPS_RANGE)):
        choices = _frontier_orbits(mask, kind, MASK_AREA_MAX)
        if not choices:
            break
        for r, c in random.choice(choices):
            mask[r, c] = True


def _valid_generated_mask(mask, kind):
    mask = _trim_mask(mask)
    area = int(mask.sum())
    return (
        MASK_AREA_MIN <= area <= MASK_AREA_MAX
        and _is_connected(mask)
        and _symmetry_kind(mask) == kind
    )


def _gen_h_mask():
    for _ in range(MASK_RETRIES):
        rows = random.randint(3, 9)
        cols = random.randint(4, 10)
        mask = np.zeros((rows, cols), dtype=bool)

        axis_rows = _middle_indices(rows)
        mask[axis_rows, :] = True

        off_center_cols = [c for c in range(cols) if c != cols - 1 - c]
        branch_col = random.choice(off_center_cols)
        branch_rows = list(range(min(axis_rows)))
        branch_len = random.randint(1, len(branch_rows))
        for r in branch_rows[-branch_len:]:
            mask[r, branch_col] = True
            mask[rows - 1 - r, branch_col] = True

        _grow_symmetric(mask, "H")
        mask = _trim_mask(mask)
        if _valid_generated_mask(mask, "H"):
            return mask
    return None


def _gen_v_mask():
    for _ in range(MASK_RETRIES):
        mask = _gen_h_mask()
        if mask is None:
            continue
        mask = mask.T.copy()
        if _valid_generated_mask(mask, "V"):
            return mask
    return None


def _paint_rot_pair(mask, r, c):
    rows, cols = mask.shape
    mask[r, c] = True
    mask[rows - 1 - r, cols - 1 - c] = True


def _gen_rotation_mask():
    for _ in range(MASK_RETRIES):
        rows = random.randint(4, 10)
        cols = random.randint(4, 10)
        mask = np.zeros((rows, cols), dtype=bool)

        center_rows = _middle_indices(rows)
        center_cols = _middle_indices(cols)
        for r in center_rows:
            for c in center_cols:
                mask[r, c] = True

        row_sign = random.choice((-1, 1))
        col_sign = random.choice((-1, 1))
        r = min(center_rows) if row_sign < 0 else max(center_rows)
        c = min(center_cols) if col_sign < 0 else max(center_cols)
        max_r = r if row_sign < 0 else rows - 1 - r
        max_c = c if col_sign < 0 else cols - 1 - c
        if max_r < 1 or max_c < 1:
            continue

        row_len = random.randint(1, max_r)
        col_len = random.randint(1, max_c)
        order = random.choice(("row_first", "col_first"))
        path = [(r, c)]
        if order == "row_first":
            for _step in range(row_len):
                r += row_sign
                path.append((r, c))
            for _step in range(col_len):
                c += col_sign
                path.append((r, c))
        else:
            for _step in range(col_len):
                c += col_sign
                path.append((r, c))
            for _step in range(row_len):
                r += row_sign
                path.append((r, c))

        for pr, pc in path:
            _paint_rot_pair(mask, pr, pc)

        _grow_symmetric(mask, "R")
        mask = _trim_mask(mask)
        if _valid_generated_mask(mask, "R"):
            return mask
    return None


def _gen_mask(kind):
    if kind == "H":
        return _gen_h_mask()
    if kind == "V":
        return _gen_v_mask()
    return _gen_rotation_mask()


def _kind_sequence():
    counts = {"H": 1, "V": 1, "R": 1}
    for _ in range(random.randint(*EXTRA_OBJECTS_RANGE)):
        counts[random.choice(("H", "V", "R"))] += 1
    kinds = []
    for kind, count in counts.items():
        kinds.extend([kind] * count)
    random.shuffle(kinds)
    return kinds


def _min_l1(a, b):
    if a is None or len(a) == 0 or b is None or len(b) == 0:
        return 999
    return int(np.abs(a[:, None, :] - b[None, :, :]).sum(axis=2).min())


def _valid_positions(shape, mask, placed):
    rows, cols = shape
    mh, mw = mask.shape
    local = np.argwhere(mask)
    positions = []
    for r0 in range(rows - mh + 1):
        for c0 in range(cols - mw + 1):
            coords = local + np.array([r0, c0])
            if _min_l1(coords, placed) < MIN_L1_BETWEEN_OBJECTS:
                continue
            positions.append((r0, c0))
    return positions


def _place_objects(grid, objects):
    placed_parts = []
    for _kind, mask in sorted(objects, key=lambda item: int(item[1].sum()), reverse=True):
        placed = np.concatenate(placed_parts, axis=0) if placed_parts else None
        positions = _valid_positions(grid.shape, mask, placed)
        if not positions:
            return None
        r0, c0 = random.choice(positions)
        coords = np.argwhere(mask) + np.array([r0, c0])
        grid[coords[:, 0], coords[:, 1]] = RED
        placed_parts.append(coords)
    return np.concatenate(placed_parts, axis=0)


def _scatter_singletons(grid, multi_coords):
    singletons = []
    rows, cols = grid.shape
    count = random.randint(*SINGLETON_COUNT_RANGE)
    for _ in range(count):
        single_arr = np.array(singletons, dtype=int) if singletons else None
        choices = []
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] != BLACK:
                    continue
                cell = np.array([[r, c]])
                if _min_l1(cell, multi_coords) < MIN_L1_BETWEEN_OBJECTS:
                    continue
                if _min_l1(cell, single_arr) < MIN_L1_BETWEEN_OBJECTS:
                    continue
                choices.append((r, c))
        if not choices:
            return False
        r, c = random.choice(choices)
        grid[r, c] = RED
        singletons.append((r, c))
    return True


def _gen_attempt():
    rows = random.randint(*GRID_ROWS)
    cols = random.randint(*GRID_COLS)
    grid = np.full((rows, cols), BLACK, dtype=int)

    objects = []
    for kind in _kind_sequence():
        mask = _gen_mask(kind)
        if mask is None:
            return None
        objects.append((kind, mask))

    multi_coords = _place_objects(grid, objects)
    if multi_coords is None:
        return None
    if not _scatter_singletons(grid, multi_coords):
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if not all(np.any(out == color) for color in (BLUE, GREEN, MAGENTA)):
        return None
    if set(map(int, np.unique(out))) - {BLACK, BLUE, GREEN, MAGENTA}:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GENERATION_RETRIES} attempts")
