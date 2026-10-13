import numpy as np


BACKGROUND = 0
INPUT_COLOR = 5
CYCLE = (2, 8, 5)
RNG = np.random.default_rng()
MAX_GENERATION_ATTEMPTS = 600


def _bbox_from_coords(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _scan_axis(coords):
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    row_span = r1 - r0
    col_span = c1 - c0
    return "row" if row_span >= col_span else "col"


def _scan_order(coords):
    axis = _scan_axis(coords)
    if axis == "row":
        return sorted(coords)
    return sorted(coords, key=lambda rc: (rc[1], rc[0]))


def _assign_colors(coords, *, consume_repeats=False, respect_diagonals=True):
    ordered = _scan_order(coords)
    diag_color = {}
    pointer = 0
    assigned = {}
    for r, c in ordered:
        diag = r + c
        if respect_diagonals and diag in diag_color:
            assigned[(r, c)] = diag_color[diag]
            if consume_repeats:
                pointer += 1
            continue
        color = CYCLE[pointer % len(CYCLE)]
        assigned[(r, c)] = color
        if respect_diagonals:
            diag_color[diag] = color
        pointer += 1
    return ordered, assigned


def _is_4_isolated(coords):
    points = set(coords)
    for r, c in points:
        if (r - 1, c) in points or (r + 1, c) in points or (r, c - 1) in points or (r, c + 1) in points:
            return False
    return True


def _has_required_conflict(coords):
    ordered = _scan_order(coords)
    diag_color = {}
    pointer = 0
    conflict = False
    for r, c in ordered:
        diag = r + c
        next_color = CYCLE[pointer % len(CYCLE)]
        if diag in diag_color:
            if diag_color[diag] != next_color:
                conflict = True
        else:
            diag_color[diag] = next_color
            pointer += 1
    return conflict


def _validate_points(coords, shape, expected_axis):
    if not (10 <= len(coords) <= 16):
        return False
    if len(set(coords)) != len(coords):
        return False
    height, width = shape
    if height > 30 or width > 30:
        return False
    if any(r < 0 or c < 0 or r >= height or c >= width for r, c in coords):
        return False
    if not _is_4_isolated(coords):
        return False
    if _scan_axis(coords) != expected_axis:
        return False

    diag_counts = {}
    for r, c in coords:
        diag = r + c
        diag_counts[diag] = diag_counts.get(diag, 0) + 1
    repeated = [count for count in diag_counts.values() if count >= 2]
    if len(repeated) < 2:
        return False
    if max(diag_counts.values()) > 3:
        return False
    if not _has_required_conflict(coords):
        return False

    _, actual = _assign_colors(coords)
    _, consume_repeats = _assign_colors(coords, consume_repeats=True)
    _, no_diagonal_lock = _assign_colors(coords, respect_diagonals=False)
    if actual == consume_repeats:
        return False
    if actual == no_diagonal_lock:
        return False
    if set(actual.values()) != set(CYCLE):
        return False
    return True


def _checkerboard_candidates(height, width, parity, axis):
    coords = [(r, c) for r in range(height) for c in range(width) if (r + c) % 2 == parity]
    if axis == "row":
        coords.sort()
    else:
        coords.sort(key=lambda rc: (rc[1], rc[0]))
    return coords


def _find_reserved_indices(candidates):
    diags = [r + c for r, c in candidates]
    diag_to_indices = {}
    for idx, diag in enumerate(diags):
        diag_to_indices.setdefault(diag, []).append(idx)

    repeated_diags = [diag for diag, idxs in diag_to_indices.items() if len(idxs) >= 2]
    RNG.shuffle(repeated_diags)

    for first_diag in repeated_diags:
        first_indices = diag_to_indices[first_diag][:]
        RNG.shuffle(first_indices)
        for i in first_indices:
            later_same = [k for k in diag_to_indices[first_diag] if k > i]
            RNG.shuffle(later_same)
            for k in later_same:
                between = [j for j in range(i + 1, k) if diags[j] != first_diag]
                RNG.shuffle(between)
                for j in between:
                    second_diag = diags[j]
                    later_fresh = [l for l in range(k + 1, len(candidates)) if diags[l] not in {first_diag, second_diag}]
                    RNG.shuffle(later_fresh)
                    for l in later_fresh:
                        banned = {first_diag, second_diag, diags[l]}
                        later_repeat_diags = [
                            diag
                            for diag, idxs in diag_to_indices.items()
                            if diag not in banned and sum(idx > l for idx in idxs) >= 2
                        ]
                        RNG.shuffle(later_repeat_diags)
                        for repeat_diag in later_repeat_diags:
                            later_indices = [idx for idx in diag_to_indices[repeat_diag] if idx > l]
                            return i, j, k, l, later_indices[0], later_indices[1]
    return None


def _pick_points(candidates, reserved, point_count):
    i, j, k, l, m, n = reserved
    reserved_set = {i, j, k, l, m, n}
    forbidden_middle = {idx for idx in range(i + 1, k) if idx != j}
    allowed = [idx for idx in range(len(candidates)) if idx not in reserved_set and idx not in forbidden_middle]
    needed = point_count - len(reserved_set)
    if needed < 0 or needed > len(allowed):
        return None
    if needed:
        extra = RNG.choice(np.array(allowed, dtype=int), size=needed, replace=False)
        selected = reserved_set | set(map(int, extra.tolist()))
    else:
        selected = reserved_set
    return [candidates[idx] for idx in sorted(selected)]


def _build_grid(coords, shape):
    grid = np.full(shape, BACKGROUND, dtype=int)
    for r, c in coords:
        grid[r, c] = INPUT_COLOR
    return grid


def _gen_attempt():
    axis = "row" if RNG.random() < 0.5 else "col"
    if axis == "row":
        box_h = int(RNG.integers(9, 14))
        box_w = int(RNG.integers(5, min(9, box_h)))
    else:
        box_w = int(RNG.integers(9, 14))
        box_h = int(RNG.integers(5, min(9, box_w)))

    parity = int(RNG.integers(0, 2))
    local_candidates = _checkerboard_candidates(box_h, box_w, parity, axis)
    reserved = _find_reserved_indices(local_candidates)
    if reserved is None:
        return None

    i, _, k, _, _, _ = reserved
    usable_count = len(local_candidates) - max(0, k - i - 1 - 1)
    max_points = min(16, usable_count)
    if max_points < 10:
        return None

    grid_h = box_h + int(RNG.integers(2, 7))
    grid_w = box_w + int(RNG.integers(2, 7))
    top = int(RNG.integers(0, grid_h - box_h + 1))
    left = int(RNG.integers(0, grid_w - box_w + 1))

    for _ in range(80):
        point_count = int(RNG.integers(10, max_points + 1))
        local_points = _pick_points(local_candidates, reserved, point_count)
        if local_points is None:
            continue
        coords = [(top + r, left + c) for r, c in local_points]
        if not _validate_points(coords, (grid_h, grid_w), axis):
            continue
        return _build_grid(coords, (grid_h, grid_w))
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor isolated gray pixels by scan order while locking each anti-diagonal to its first scanned color.

    Input structure:
    - The grid has a black background and 10-16 isolated gray pixels.
    - The gray pixels' bounding box is taller than wide or wider than tall, which selects the scan direction.
    - Some gray pixels share an anti-diagonal, meaning they have the same row+col sum.

    Rule:
    - Compute the gray-pixel bounding box. If its row span is at least its column span, scan pixels in row-major order; otherwise scan them in column-major order.
    - Use the repeating color cycle red(2), teal(8), gray(5).
    - When a scanned pixel is the first one on its anti-diagonal, assign the next cycle color and lock that anti-diagonal to the chosen color.
    - When a scanned pixel belongs to an already locked anti-diagonal, copy the locked color and do not advance the cycle.
    - Recolor only the original gray pixels; keep all background pixels unchanged.

    Key insight:
    - The result is constrained simultaneously by scan order and anti-diagonal consistency, and repeated anti-diagonals skip cycle consumption, which changes all later first-time assignments.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    coords = [tuple(map(int, rc)) for rc in np.argwhere(g == INPUT_COLOR)]
    if not coords:
        return out
    _, assigned = _assign_colors(coords)
    for (r, c), color in assigned.items():
        out[r, c] = color
    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
