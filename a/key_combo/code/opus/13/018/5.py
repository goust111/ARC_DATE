import random
from collections import Counter

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

AXES = ("row", "col")
BRANCH_LENGTHS = (1, 3, 5, 7)
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _find_spine(red_coords: list[tuple[int, int]]) -> tuple[str, int, int, int, tuple[tuple[int, int], tuple[int, int]]]:
    candidates: list[tuple[str, int, int, int, tuple[tuple[int, int], tuple[int, int]]]] = []
    for i in range(len(red_coords)):
        r1, c1 = red_coords[i]
        for j in range(i + 1, len(red_coords)):
            r2, c2 = red_coords[j]
            if r1 == r2:
                row = r1
                lo, hi = sorted((c1, c2))
                others = [pt for k, pt in enumerate(red_coords) if k not in (i, j)]
                if others and all(r != row and lo <= c <= hi for r, c in others):
                    candidates.append(("row", row, lo, hi, (red_coords[i], red_coords[j])))
            if c1 == c2:
                col = c1
                lo, hi = sorted((r1, r2))
                others = [pt for k, pt in enumerate(red_coords) if k not in (i, j)]
                if others and all(c != col and lo <= r <= hi for r, c in others):
                    candidates.append(("col", col, lo, hi, (red_coords[i], red_coords[j])))

    dedup = []
    seen = set()
    for axis, fixed, lo, hi, endpoints in candidates:
        key = (axis, fixed, lo, hi, tuple(sorted(endpoints)))
        if key not in seen:
            seen.add(key)
            dedup.append((axis, fixed, lo, hi, endpoints))

    assert len(dedup) == 1, f"expected exactly one spine, got {len(dedup)}"
    return dedup[0]


def _branch_endpoint(axis: str, fixed: int, attach: int, sign: int, length: int) -> tuple[int, int]:
    if axis == "row":
        return fixed + sign * length, attach
    return attach, fixed + sign * length


def _connector_cells(axis: str, fixed: int, attach: int, endpoint: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = endpoint
    cells = []
    if axis == "row":
        step = 1 if fixed > r else -1
        for rr in range(r + step, fixed, step):
            cells.append((rr, c))
    else:
        step = 1 if fixed > c else -1
        for cc in range(c + step, fixed, step):
            cells.append((r, cc))
    return cells


def _stripe_cells(
    axis: str,
    endpoint: tuple[int, int],
    sign: int,
    length: int,
    shape: tuple[int, int],
) -> list[list[tuple[int, int]]]:
    h, w = shape
    radius = length // 2
    stripes: list[list[tuple[int, int]]] = []
    step = 0
    while True:
        if axis == "row":
            rr = endpoint[0] + sign * 2 * step
            if not (0 <= rr < h):
                break
            stripe = [(rr, cc) for cc in range(endpoint[1] - radius, endpoint[1] + radius + 1) if 0 <= cc < w]
        else:
            cc = endpoint[1] + sign * 2 * step
            if not (0 <= cc < w):
                break
            stripe = [(rr, cc) for rr in range(endpoint[0] - radius, endpoint[0] + radius + 1) if 0 <= rr < h]
        stripes.append(stripe)
        step += 1
    return stripes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build the unique red spine-and-branch skeleton, then emit alternating parallel stripes away from each branch tip.

    Input structure:
    - A black background contains only red control pixels.
    - Exactly one red pair shares a row or a column; that aligned pair defines the green spine segment.
    - Every other red pixel projects orthogonally onto the spine span and serves as a branch endpoint.

    Rule:
    - Fill the cells between the aligned red pair with green to form the spine, keeping the red endpoints unchanged.
    - Connect each off-spine red pixel to the spine with a green orthogonal segment, and let its branch length L be its Manhattan distance to the spine row or column.
    - From each branch endpoint, march away from the spine in every other row or column, painting stripes parallel to the spine with colors Y, Y, B repeating; each stripe is centered on the emitter and has width 2*floor(L/2)+1.

    Key insight:
    - The same branch length first arises as geometry when attaching a control pixel to the spine, then reappears as the stripe-width parameter for the outward emission.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    red_coords = [tuple(map(int, rc)) for rc in np.argwhere(g == RED)]
    assert 5 <= len(red_coords) <= 7

    axis, fixed, lo, hi, endpoints = _find_spine(red_coords)
    endpoint_set = {tuple(map(int, pt)) for pt in endpoints}
    branches = [pt for pt in red_coords if pt not in endpoint_set]

    if axis == "row":
        for c in range(lo + 1, hi):
            if out[fixed, c] != RED:
                out[fixed, c] = GREEN
    else:
        for r in range(lo + 1, hi):
            if out[r, fixed] != RED:
                out[r, fixed] = GREEN

    cycle = (YELLOW, YELLOW, BLUE)
    for branch in branches:
        if axis == "row":
            length = abs(branch[0] - fixed)
            sign = -1 if branch[0] < fixed else 1
            attach = branch[1]
        else:
            length = abs(branch[1] - fixed)
            sign = -1 if branch[1] < fixed else 1
            attach = branch[0]

        for r, c in _connector_cells(axis, fixed, attach, branch):
            if out[r, c] != RED:
                out[r, c] = GREEN

        for idx, stripe in enumerate(_stripe_cells(axis, branch, sign, length, g.shape)):
            color = cycle[idx % 3]
            for r, c in stripe:
                if out[r, c] != RED:
                    out[r, c] = color

    for r, c in red_coords:
        out[r, c] = RED
    return out


def _pick_side_lengths(available: list[int], count: int) -> list[int] | None:
    if len(available) < count:
        return None
    lengths = random.sample(available, count)
    random.shuffle(lengths)
    return lengths


def _candidate_attach_positions(
    axis: str,
    lo: int,
    hi: int,
    parallel_dim: int,
    length: int,
) -> list[int]:
    radius = length // 2
    low = max(lo + 1, radius)
    high = min(hi - 1, parallel_dim - 1 - radius)
    if low > high:
        return []
    positions = list(range(low, high + 1))
    random.shuffle(positions)
    return positions


def _interval_for_attach(attach: int, length: int) -> tuple[int, int]:
    radius = length // 2
    return attach - radius, attach + radius


def _pick_attach_layout(
    axis: str,
    lo: int,
    hi: int,
    parallel_dim: int,
    lengths: list[int],
    used_positions: set[int],
) -> list[int] | None:
    order = sorted(range(len(lengths)), key=lambda idx: lengths[idx], reverse=True)
    chosen: dict[int, int] = {}
    intervals: list[tuple[int, int]] = []

    def rec(k: int) -> bool:
        if k == len(order):
            return True
        idx = order[k]
        length = lengths[idx]
        for attach in _candidate_attach_positions(axis, lo, hi, parallel_dim, length):
            if attach in used_positions or attach in chosen.values():
                continue
            left, right = _interval_for_attach(attach, length)
            if any(not (right < a or b < left) for a, b in intervals):
                continue
            chosen[idx] = attach
            intervals.append((left, right))
            if rec(k + 1):
                return True
            intervals.pop()
            del chosen[idx]
        return False

    if not rec(0):
        return None
    return [chosen[i] for i in range(len(lengths))]


def _branch_spec_cells(
    axis: str,
    fixed: int,
    attach: int,
    sign: int,
    length: int,
    shape: tuple[int, int],
) -> tuple[tuple[int, int], set[tuple[int, int]], set[tuple[int, int]]]:
    endpoint = _branch_endpoint(axis, fixed, attach, sign, length)
    connector = set(_connector_cells(axis, fixed, attach, endpoint))
    stripes = {cell for stripe in _stripe_cells(axis, endpoint, sign, length, shape) for cell in stripe}
    return endpoint, connector, stripes


def _layout_is_valid(
    axis: str,
    shape: tuple[int, int],
    fixed: int,
    endpoints: tuple[tuple[int, int], tuple[int, int]],
    branches: list[dict],
) -> bool:
    red_coords = [endpoints[0], endpoints[1], *[b["endpoint"] for b in branches]]
    if len(set(red_coords)) != len(red_coords):
        return False
    try:
        found = _find_spine(red_coords)
    except AssertionError:
        return False

    if found[0] != axis or found[1] != fixed:
        return False

    stripe_cells: set[tuple[int, int]] = set()
    for branch in branches:
        overlap = stripe_cells & branch["stripes"]
        if overlap:
            return False
        stripe_cells |= branch["stripes"]
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(16, 22)
    w = random.randint(16, 22)
    axis = random.choice(AXES)
    parallel_dim = w if axis == "row" else h
    perp_dim = h if axis == "row" else w
    branch_count = random.randint(3, 5)

    gap_min = max(branch_count + 3, int(np.ceil(0.55 * parallel_dim)))
    gap_max = min(parallel_dim - 2, int(np.floor(0.9 * parallel_dim)))
    if gap_min > gap_max:
        return None
    gap = random.randint(gap_min, gap_max)
    lo = random.randint(0, parallel_dim - 1 - gap)
    hi = lo + gap

    fixed_candidates = list(range(2, perp_dim - 2))
    random.shuffle(fixed_candidates)
    for fixed in fixed_candidates:
        neg_available = [length for length in BRANCH_LENGTHS if fixed - length >= 0]
        pos_available = [length for length in BRANCH_LENGTHS if fixed + length < perp_dim]

        splits = []
        for neg_count in range(1, branch_count):
            pos_count = branch_count - neg_count
            if len(neg_available) >= neg_count and len(pos_available) >= pos_count:
                splits.append((neg_count, pos_count))
        random.shuffle(splits)

        for neg_count, pos_count in splits:
            neg_lengths = _pick_side_lengths(neg_available, neg_count)
            pos_lengths = _pick_side_lengths(pos_available, pos_count)
            if neg_lengths is None or pos_lengths is None:
                continue
            if len(set(neg_lengths + pos_lengths)) < 2:
                continue

            used_positions: set[int] = set()
            neg_attach = _pick_attach_layout(axis, lo, hi, parallel_dim, neg_lengths, used_positions)
            if neg_attach is None:
                continue
            used_positions.update(neg_attach)
            pos_attach = _pick_attach_layout(axis, lo, hi, parallel_dim, pos_lengths, used_positions)
            if pos_attach is None:
                continue

            branches = []
            for sign, lengths, attach_positions in ((-1, neg_lengths, neg_attach), (1, pos_lengths, pos_attach)):
                for length, attach in zip(lengths, attach_positions):
                    endpoint, connector, stripes = _branch_spec_cells(axis, fixed, attach, sign, length, (h, w))
                    branches.append(
                        {
                            "sign": sign,
                            "length": length,
                            "attach": attach,
                            "endpoint": endpoint,
                            "connector": connector,
                            "stripes": stripes,
                        }
                    )

            if axis == "row":
                spine_endpoints = ((fixed, lo), (fixed, hi))
            else:
                spine_endpoints = ((lo, fixed), (hi, fixed))

            if not _layout_is_valid(axis, (h, w), fixed, spine_endpoints, branches):
                continue

            grid = np.zeros((h, w), dtype=int)
            for r, c in (spine_endpoints[0], spine_endpoints[1], *[b["endpoint"] for b in branches]):
                grid[r, c] = RED

            try:
                out = main(grid)
            except AssertionError:
                continue

            if np.array_equal(out, grid):
                continue
            return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed")
