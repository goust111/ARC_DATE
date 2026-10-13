import random
from collections import Counter, deque

import numpy as np


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, bg: int) -> list[int]:
    return sorted(int(x) for x in np.unique(grid) if int(x) != bg)


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def component_coords(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def bbox_of(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def reflect_coords(coords: list[tuple[int, int]], pivot2: tuple[int, int]) -> list[tuple[int, int]]:
    pr2, pc2 = pivot2
    return [(pr2 - r, pc2 - c) for r, c in coords]


def normalize(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(r), int(c)) for r, c in coords))


def pair_center2(comp_a: list[tuple[int, int]], comp_b: list[tuple[int, int]]) -> tuple[int, int]:
    a0, a1, a2, a3 = bbox_of(comp_a)
    b0, b1, b2, b3 = bbox_of(comp_b)
    pr2 = a0 + b2
    pc2 = a1 + b3
    assert pr2 == a2 + b0, "paired sprites must share one reflection center"
    assert pc2 == a3 + b1, "paired sprites must share one reflection center"
    reflected = normalize(reflect_coords(comp_a, (pr2, pc2)))
    assert reflected == normalize(comp_b), "same-color sprites must be 180-degree partners"
    return int(pr2), int(pc2)


def infer_ordered_centers(centers2: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    assert len(centers2) >= 2, "need at least two paired centers"
    rows = [p[0] for p in centers2]
    cols = [p[1] for p in centers2]
    if max(rows) - min(rows) >= max(cols) - min(cols):
        ordered = sorted(centers2, key=lambda p: (p[0], p[1]))
    else:
        ordered = sorted(centers2, key=lambda p: (p[1], p[0]))
    dr = ordered[1][0] - ordered[0][0]
    dc = ordered[1][1] - ordered[0][1]
    assert (dr, dc) != (0, 0), "centers must be distinct"
    for i in range(1, len(ordered)):
        assert ordered[i][0] - ordered[i - 1][0] == dr
        assert ordered[i][1] - ordered[i - 1][1] == dc
    return ordered, (int(dr), int(dc))


def analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    colors = colors_present(g, bg)
    pair_infos = []
    lone_info = None
    all_occupied = set()

    for color in colors:
        comps = component_coords(g == color)
        assert len(comps) in (1, 2), "each color must form one or two 4-connected sprites"
        for comp in comps:
            all_occupied.update(comp)
        if len(comps) == 1:
            assert lone_info is None, "expected exactly one unpaired color"
            lone_info = {
                "color": int(color),
                "coords": comps[0],
                "coord_set": set(comps[0]),
                "bbox": bbox_of(comps[0]),
            }
        else:
            center2 = pair_center2(comps[0], comps[1])
            pair_infos.append(
                {
                    "color": int(color),
                    "center2": center2,
                    "components": comps,
                }
            )

    assert lone_info is not None, "missing lone sprite"
    centers2 = [info["center2"] for info in pair_infos]
    ordered, step2 = infer_ordered_centers(centers2)
    left = (ordered[0][0] - step2[0], ordered[0][1] - step2[1])
    right = (ordered[-1][0] + step2[0], ordered[-1][1] + step2[1])

    candidate_fits = []
    for pivot2 in (left, right):
        partner = reflect_coords(lone_info["coords"], pivot2)
        fits = all(in_bounds(g.shape, r, c) for r, c in partner)
        candidate_fits.append(
            {
                "pivot2": pivot2,
                "partner": partner,
                "fits": fits,
            }
        )

    valid = [item for item in candidate_fits if item["fits"]]
    assert len(valid) == 1, "exactly one AP extension endpoint must keep the lone partner in-grid"
    chosen = valid[0]

    return {
        "bg": bg,
        "pair_infos": pair_infos,
        "ordered_centers2": ordered,
        "step2": step2,
        "hidden_pivot2": chosen["pivot2"],
        "other_endpoint2": right if chosen["pivot2"] == left else left,
        "lone": lone_info,
        "partner_coords": chosen["partner"],
        "occupied": all_occupied,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the lone sprite by reflecting it through the missing arithmetic-progression pivot.

    Input structure:
    - A black background contains several monochrome 4-connected sprites.
    - Each color appears in one or two separate sprites; exactly one color appears only once.
    - Every repeated color forms a 180-degree point-reflection pair around its own invisible center.

    Rule:
    - Recover the hidden center of each repeated-color pair from the two same-color sprites.
    - These centers form a collinear arithmetic progression; extend that progression by one point at both ends.
    - Exactly one extension point keeps the lone sprite's 180-degree partner fully inside the grid; draw that partner in the lone sprite's color and leave everything else unchanged.

    Key insight:
    - The decisive object is not a painted pixel but an invisible chain of reflection centers, and the lone sprite must use the missing endpoint of that hidden arithmetic progression as its pivot.
    """
    info = analyze_grid(np.asarray(grid, dtype=int))
    out = np.asarray(grid, dtype=int).copy()
    lone_color = info["lone"]["color"]
    occupied = info["occupied"]
    for r, c in info["partner_coords"]:
        assert in_bounds(out.shape, r, c)
        assert (r, c) not in occupied, "generated partner should occupy fresh cells"
        out[r, c] = lone_color
    return out


MAX_ATTEMPTS = 400
PAIR_COUNT_RANGE = (3, 4)
GRID_SIZE_RANGE = (18, 30)
STEP2_CHOICES = [
    (6, 0),
    (0, 6),
    (6, 6),
    (6, -6),
    (8, 4),
    (8, -4),
    (4, 8),
    (4, -8),
    (5, 3),
    (5, -3),
    (3, 5),
    (3, -5),
]
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]


def random_connected_mask(h: int, w: int) -> np.ndarray:
    target = random.randint(3, min(h * w, 7))
    mask = np.zeros((h, w), dtype=bool)
    seed = (random.randrange(h), random.randrange(w))
    mask[seed] = True
    cells = {seed}
    frontier = set()
    for cell in cells:
        frontier.update((nr, nc) for nr, nc in neighbors4(*cell) if 0 <= nr < h and 0 <= nc < w)
    while len(cells) < target and frontier:
        nr, nc = random.choice(tuple(frontier))
        frontier.remove((nr, nc))
        mask[nr, nc] = True
        cells.add((nr, nc))
        for rr, cc in neighbors4(nr, nc):
            if 0 <= rr < h and 0 <= cc < w and not mask[rr, cc]:
                frontier.add((rr, cc))
    if len(cells) < 3:
        return random_connected_mask(h, w)
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    if max(rows) == min(rows) and max(cols) == min(cols):
        return random_connected_mask(h, w)
    return mask


def mask_to_coords(mask: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    return [(top + r, left + c) for r, c in np.argwhere(mask)]


def forbidden_cells(occupied: set[tuple[int, int]]) -> set[tuple[int, int]]:
    blocked = set(occupied)
    for r, c in list(occupied):
        blocked.update(neighbors4(r, c))
    return blocked


def bbox_padded_overlap(
    bbox_a: tuple[int, int, int, int],
    bbox_b: tuple[int, int, int, int],
    pad: int = 0,
) -> bool:
    ar0, ac0, ar1, ac1 = bbox_a
    br0, bc0, br1, bc1 = bbox_b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def make_pair_candidates(
    grid_shape: tuple[int, int],
    pivot2: tuple[int, int],
    mask: np.ndarray,
    occupied: set[tuple[int, int]],
) -> list[tuple[list[tuple[int, int]], list[tuple[int, int]]]]:
    h, w = grid_shape
    mh, mw = mask.shape
    local = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    blocked = forbidden_cells(occupied)
    out = []
    for top in range(h - mh + 1):
        for left in range(w - mw + 1):
            comp_a = [(top + r, left + c) for r, c in local]
            comp_b = reflect_coords(comp_a, pivot2)
            if not all(in_bounds(grid_shape, r, c) for r, c in comp_b):
                continue
            set_a = set(comp_a)
            set_b = set(comp_b)
            if set_a & set_b:
                continue
            bbox_a = bbox_of(comp_a)
            bbox_b = bbox_of(comp_b)
            if bbox_padded_overlap(bbox_a, bbox_b, pad=0):
                continue
            if any(cell in forbidden_cells(set_a) for cell in set_b):
                continue
            if any(cell in blocked for cell in set_a | set_b):
                continue
            out.append((comp_a, comp_b))
    random.shuffle(out)
    return out


def choose_grid_and_centers() -> tuple[tuple[int, int], list[tuple[int, int]], int]:
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    dr, dc = random.choice(STEP2_CHOICES)
    if random.random() < 0.5:
        dr = -dr
    if random.random() < 0.5:
        dc = -dc

    total_steps = pair_count + 1
    min_h = max(GRID_SIZE_RANGE[0], int(np.ceil(abs(dr) * total_steps / 2.0)) + 6)
    min_w = max(GRID_SIZE_RANGE[0], int(np.ceil(abs(dc) * total_steps / 2.0)) + 6)
    if min_h > GRID_SIZE_RANGE[1] or min_w > GRID_SIZE_RANGE[1]:
        raise ValueError("chosen step cannot fit ARC limits")
    h = random.randint(min_h, GRID_SIZE_RANGE[1])
    w = random.randint(min_w, GRID_SIZE_RANGE[1])

    max_r2 = 2 * (h - 1)
    max_c2 = 2 * (w - 1)
    extent_r = dr * total_steps
    extent_c = dc * total_steps

    if dr >= 0:
        r_low = 1
        r_high = max_r2 - extent_r - 1
    else:
        r_low = -extent_r + 1
        r_high = max_r2 - 1
    if dc >= 0:
        c_low = 1
        c_high = max_c2 - extent_c - 1
    else:
        c_low = -extent_c + 1
        c_high = max_c2 - 1
    if r_low > r_high or c_low > c_high:
        raise ValueError("center line bounds failed")

    def edge_biased_choice(low: int, high: int) -> int:
        width = min(4, high - low)
        if random.random() < 0.5:
            return random.randint(low, low + width)
        return random.randint(high - width, high)

    start = (edge_biased_choice(r_low, r_high), edge_biased_choice(c_low, c_high))
    centers = [(start[0] + i * dr, start[1] + i * dc) for i in range(pair_count + 2)]
    return (h, w), centers, pair_count


def gen_attempt() -> np.ndarray | None:
    try:
        shape, full_centers, pair_count = choose_grid_and_centers()
    except ValueError:
        return None
    h, w = shape
    paired_centers = full_centers[1 : pair_count + 1]
    hidden_at_start = random.random() < 0.5
    hidden_pivot2 = full_centers[0] if hidden_at_start else full_centers[-1]
    other_endpoint2 = full_centers[-1] if hidden_at_start else full_centers[0]
    colors = random.sample(PALETTE, pair_count + 1)

    grid = np.zeros((h, w), dtype=int)
    occupied: set[tuple[int, int]] = set()

    for color, pivot2 in zip(colors[:pair_count], paired_centers):
        placed = False
        for _ in range(40):
            mask = random_connected_mask(random.randint(2, 4), random.randint(2, 4))
            candidates = make_pair_candidates(shape, pivot2, mask, occupied)
            if not candidates:
                continue
            comp_a, comp_b = random.choice(candidates[: min(8, len(candidates))])
            for r, c in comp_a + comp_b:
                grid[r, c] = color
                occupied.add((r, c))
            placed = True
            break
        if not placed:
            return None

    lone_color = colors[-1]
    visible = None
    missing_partner = None
    for _ in range(80):
        mask = random_connected_mask(random.randint(2, 4), random.randint(2, 4))
        candidates = make_pair_candidates(shape, hidden_pivot2, mask, occupied)
        if not candidates:
            continue
        random.shuffle(candidates)
        for comp_a, comp_b in candidates[:20]:
            for candidate_visible, candidate_missing in ((comp_a, comp_b), (comp_b, comp_a)):
                other_reflection = reflect_coords(candidate_visible, other_endpoint2)
                if all(in_bounds(shape, r, c) for r, c in other_reflection):
                    continue
                visible = candidate_visible
                missing_partner = candidate_missing
                break
            if visible is not None:
                break
        if visible is not None:
            break
    if visible is None or missing_partner is None:
        return None

    for r, c in visible:
        grid[r, c] = lone_color
        occupied.add((r, c))

    try:
        info = analyze_grid(grid)
    except AssertionError:
        return None

    expected_partner = set(missing_partner)
    if info["hidden_pivot2"] != hidden_pivot2:
        return None
    if set(info["partner_coords"]) != expected_partner:
        return None
    if info["other_endpoint2"] != other_endpoint2:
        return None

    out = main(grid)
    diff = {(int(r), int(c)) for r, c in np.argwhere(out != grid)}
    if diff != expected_partner:
        return None
    if any(grid[r, c] != 0 for r, c in expected_partner):
        return None
    if not np.all(out[np.array([r for r, _ in expected_partner]), np.array([c for _, c in expected_partner])] == lone_color):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hidden-pivot arithmetic-progression task")
