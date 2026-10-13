from __future__ import annotations

from itertools import combinations

import numpy as np

from arc_utils import Color, bbox_coords, label_mask
import random
from scipy import ndimage


DIAGONAL_STEPS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _border_sides(coords: np.ndarray, shape: tuple[int, int]) -> frozenset[str]:
    h, w = shape
    r0, c0, r1, c1 = bbox_coords(coords)
    sides = set()
    if r0 == 0:
        sides.add("T")
    if r1 == h - 1:
        sides.add("B")
    if c0 == 0:
        sides.add("L")
    if c1 == w - 1:
        sides.add("R")
    return frozenset(sides)


def _single_square_staircase(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = bbox_coords(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h != w or h < 4:
        return False
    if len(coords) * 2 > h * w:
        return False

    pts = {tuple(map(int, p)) for p in coords}
    corners = {
        "NW": (r0, c0) in pts,
        "NE": (r0, c1) in pts,
        "SW": (r1, c0) in pts,
        "SE": (r1, c1) in pts,
    }
    return (corners["NW"] and corners["SE"]) or (corners["NE"] and corners["SW"])


def _diagonal_contact_dirs(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> set[tuple[int, int]]:
    dirs = set()
    for r, c in a:
        for dr, dc in DIAGONAL_STEPS:
            if (r + dr, c + dc) in b:
                dirs.add((dr, dc))
    return dirs


def _max_straight_diagonal_run(coords: np.ndarray) -> int:
    pts = {tuple(map(int, p)) for p in coords}
    max_pixels = 0
    for dr, dc in ((1, 1), (1, -1)):
        for r, c in pts:
            if (r - dr, c - dc) in pts:
                continue
            length = 0
            while (r, c) in pts:
                length += 1
                r += dr
                c += dc
            max_pixels = max(max_pixels, length)
    return max_pixels


def _straight_singleton_diagonal(coords: np.ndarray, n_parts: int) -> bool:
    r0, c0, r1, c1 = bbox_coords(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    pts = {tuple(map(int, p)) for p in coords}
    return h == w == n_parts and (
        len({r - c for r, c in pts}) == 1 or len({r + c for r, c in pts}) == 1
    )


def _is_target_region(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if _max_straight_diagonal_run(coords) < 3:
        return False

    lab4, n4 = label_mask(mask, connectivity=4)

    if n4 == 1:
        return _single_square_staircase(coords)

    parts = []
    for k in range(1, n4 + 1):
        part_coords = np.argwhere(lab4 == k)
        parts.append(
            {
                "coords": part_coords,
                "pts": {tuple(map(int, p)) for p in part_coords},
                "sides": _border_sides(part_coords, mask.shape),
            }
        )

    degrees = [0] * n4
    edge_count = 0
    for i, j in combinations(range(n4), 2):
        dirs = _diagonal_contact_dirs(parts[i]["pts"], parts[j]["pts"])
        if not dirs:
            continue
        if len(dirs) != 1:
            return False
        degrees[i] += 1
        degrees[j] += 1
        edge_count += 1

    assert edge_count >= n4 - 1
    if max(degrees) > 2:
        return False

    sizes = sorted(len(part["coords"]) for part in parts)
    if sizes[-1] == 1:
        return _straight_singleton_diagonal(coords, n4)

    border_sides = [part["sides"] for part in parts if part["sides"]]
    if border_sides:
        all_border_sides = set().union(*border_sides)
        if len(all_border_sides) == 1 and len(border_sides) == n4:
            return False

    if n4 == 2 and sizes[0] == 1 and sizes[1] >= 5:
        return all(part["sides"] for part in parts)

    return True


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor the grey diagonal-chain regions in an orange field to teal.

    Input structure
    The input contains only grey pixels and orange pixels.  Grey pixels form
    several 8-connected regions embedded in the orange field.

    Rule
    Decompose each grey 8-connected region into 4-connected lobes.  Recolor the
    whole region teal only when it contains a straight diagonal run of at least
    three pixels and those lobes meet only by diagonal corner contacts in a
    non-branching diagonal chain or cycle; a single 4-connected region is
    recolored only when it is a square diagonal staircase.

    Key insight
    The teal regions are not selected by size or position; they are the grey
    regions whose internal connectivity depends on diagonal corner joins, while
    same-edge cracks, rectangles, bars, and branching/noisy regions remain grey.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(np.unique(g)) == {Color.GREY, Color.ORANGE}

    out = g.copy()
    lab8, n8 = label_mask(g == Color.GREY, connectivity=8)
    assert n8 >= 1

    for k in range(1, n8 + 1):
        mask = lab8 == k
        if _is_target_region(mask):
            out[mask] = Color.TEAL

    return out

# --------------- generator (c) ---------------

# Grid dimensions.
GRID_ROWS_RANGE, GRID_COLS_RANGE = (4, 16), (5, 19)

# Region counts.
TARGET_COUNT_RANGE, DISTRACTOR_COUNT_RANGE = (1, 4), (3, 15)

# Target shape sizes and branch probabilities.
TARGET_STAIRCASE_PROB = 0.5
STAIRCASE_NW_SE_PROB = 0.5
STAIRCASE_MIN_SIZE, STAIRCASE_MAX_SIZE = 4, 6
CHAIN_LOBES_RANGE, LOBE_SIZE_RANGE = (2, 4), (1, 4)
MIN_DIAGONAL_RUN = 3

# Distractor shape choices.
DISTRACTOR_KINDS = ("pixel", "small_blob", "bar", "rect", "L")
DISTRACTOR_WEIGHTS = (3, 4, 2, 2, 1)
BAR_VERTICAL_PROB = 0.5
SMALL_BLOB_SIZE_RANGE, BAR_LENGTH_RANGE = (2, 5), (2, 4)
RECT_SIZE_RANGE, L_SIZE_RANGE = (2, 3), (2, 3)

# Retry and acceptance thresholds.
DISTRACTOR_RETRY_LIMIT, TARGET_RETRY_LIMIT, GEN_ATTEMPT_LIMIT = 50, 80, 300
MIN_GREY_DENSITY, MAX_GREY_DENSITY = 0.15, 0.50

CARDINAL_STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))
STRUCT8 = ndimage.generate_binary_structure(2, 2)


def _gen_max_straight_diagonal_run(pts: set[tuple[int, int]]) -> int:
    best = 0
    for dr, dc in ((1, 1), (1, -1)):
        for r, c in pts:
            if (r - dr, c - dc) in pts:
                continue
            length = 0
            while (r, c) in pts:
                length += 1
                r += dr
                c += dc
            best = max(best, length)
    return best


def _gen_mask_points(mask: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, p)) for p in np.argwhere(mask)}


def _make_staircase(n: int) -> set[tuple[int, int]]:
    """Diagonal staircase path in NxN bbox spanning opposite corners."""
    if random.random() < STAIRCASE_NW_SE_PROB:
        path, r, c = [(0, 0)], 0, 0
        while r < n - 1 or c < n - 1:
            if r == n - 1: c += 1
            elif c == n - 1: r += 1
            elif random.random() < 0.5: r += 1
            else: c += 1
            path.append((r, c))
    else:
        path, r, c = [(0, n - 1)], 0, n - 1
        while r < n - 1 or c > 0:
            if r == n - 1: c -= 1
            elif c == 0: r += 1
            elif random.random() < 0.5: r += 1
            else: c -= 1
            path.append((r, c))
    return set(path)


def _grow_blob(size: int, seed: tuple[int, int], forbidden: set) -> set[tuple[int, int]]:
    blob = {seed}
    for _ in range(size - 1):
        frontier = [(r + dr, c + dc)
                     for r, c in blob for dr, dc in CARDINAL_STEPS
                     if (r + dr, c + dc) not in blob and (r + dr, c + dc) not in forbidden]
        if not frontier:
            break
        blob.add(random.choice(frontier))
    return blob


def _grow_blob_constrained(seed: tuple[int, int], size: int, all_pts: set,
                           existing_lobes: list[set]) -> set[tuple[int, int]]:
    blob = {seed}
    for _ in range(size - 1):
        frontier = []
        for r, c in blob:
            for dr, dc in CARDINAL_STEPS:
                nr, nc = r + dr, c + dc
                if (nr, nc) in blob or (nr, nc) in all_pts:
                    continue
                if any((nr + ddr, nc + ddc) in lobe
                       for lobe in existing_lobes for ddr, ddc in CARDINAL_STEPS):
                    continue
                frontier.append((nr, nc))
        if not frontier:
            break
        blob.add(random.choice(frontier))
    return blob


def _candidate_diagonal_contacts(prev_lobe: set[tuple[int, int]], dr: int, dc: int,
                                 all_pts: set[tuple[int, int]]) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for r, c in prev_lobe
            if (r + dr, c + dc) not in all_pts
            and not any((r + dr + ddr, c + dc + ddc) in prev_lobe
                        for ddr, ddc in CARDINAL_STEPS)]


def _diagonal_dirs_between(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return {(dr, dc) for r, c in a for dr, dc in DIAGONAL_STEPS
            if (r + dr, c + dc) in b}


def _try_append_chain_lobe(lobes: list[set[tuple[int, int]]],
                           all_pts: set[tuple[int, int]]) -> set[tuple[int, int]] | None:
    prev_lobe = lobes[-1]
    diag_dirs = list(DIAGONAL_STEPS)
    random.shuffle(diag_dirs)

    for dr, dc in diag_dirs:
        candidates = _candidate_diagonal_contacts(prev_lobe, dr, dc, all_pts)
        if not candidates:
            continue

        lobe = _grow_blob_constrained(random.choice(candidates),
                                      random.randint(*LOBE_SIZE_RANGE), all_pts, lobes)
        if len(_diagonal_dirs_between(lobe, prev_lobe)) != 1:
            continue
        if any(_diagonal_dirs_between(lobe, other) for other in lobes[:-1]):
            continue
        return lobe
    return None


def _make_chain_region(num_lobes: int) -> list[set[tuple[int, int]]]:
    """Multi-lobe chain connected by single diagonal contacts. Returns [] on failure."""
    lobes: list[set[tuple[int, int]]] = []
    all_pts: set[tuple[int, int]] = set()

    for i in range(num_lobes):
        if i == 0:
            lobe = _grow_blob(random.randint(*LOBE_SIZE_RANGE), (0, 0), set())
            lobes.append(lobe)
            all_pts |= lobe
            continue

        lobe = _try_append_chain_lobe(lobes, all_pts)
        if lobe is None:
            return []
        lobes.append(lobe)
        all_pts |= lobe
    return lobes


def _make_distractor_raw() -> set[tuple[int, int]]:
    """Make a small 4-connected distractor region (non-staircase, non-valid-chain)."""
    kind = random.choices(DISTRACTOR_KINDS, weights=DISTRACTOR_WEIGHTS, k=1)[0]
    if kind == "pixel":
        return {(0, 0)}
    elif kind == "small_blob":
        return _grow_blob(random.randint(*SMALL_BLOB_SIZE_RANGE), (0, 0), set())
    elif kind == "bar":
        length = random.randint(*BAR_LENGTH_RANGE)
        return ({(r, 0) for r in range(length)} if random.random() < BAR_VERTICAL_PROB
                else {(0, c) for c in range(length)})
    elif kind == "rect":
        h, w = random.randint(*RECT_SIZE_RANGE), random.randint(*RECT_SIZE_RANGE)
        return {(r, c) for r in range(h) for c in range(w)}
    else:  # L
        h, w = random.randint(*L_SIZE_RANGE), random.randint(*L_SIZE_RANGE)
        return {(r, 0) for r in range(h)} | {(h - 1, c) for c in range(w)}


def _make_distractor() -> set[tuple[int, int]]:
    """Distractors in eval never contain a straight diagonal run of length 3+."""
    for _ in range(DISTRACTOR_RETRY_LIMIT):
        pts = _make_distractor_raw()
        if _gen_max_straight_diagonal_run(pts) < MIN_DIAGONAL_RUN:
            return pts
    return {(0, 0), (0, 1), (1, 0), (1, 1)}


def _make_target_region(H: int, W: int) -> set[tuple[int, int]] | None:
    for _ in range(TARGET_RETRY_LIMIT):
        if random.random() < TARGET_STAIRCASE_PROB:
            max_n = min(STAIRCASE_MAX_SIZE, H, W)
            if max_n < STAIRCASE_MIN_SIZE:
                continue
            pts = _make_staircase(random.randint(STAIRCASE_MIN_SIZE, max_n))
        else:
            lobes = _make_chain_region(random.randint(*CHAIN_LOBES_RANGE))
            if not lobes:
                continue
            pts = set().union(*lobes)
        if _gen_max_straight_diagonal_run(pts) >= MIN_DIAGONAL_RUN:
            return pts
    return None


def _normalize_points(pts: set[tuple[int, int]]) -> tuple[list[tuple[int, int]], int, int]:
    rs = [r for r, c in pts]
    cs = [c for r, c in pts]
    min_r, max_r, min_c, max_c = min(rs), max(rs), min(cs), max(cs)
    ph, pw = max_r - min_r + 1, max_c - min_c + 1
    return [(r - min_r, c - min_c) for r, c in pts], ph, pw


def _place_region(grid: np.ndarray, pts: set[tuple[int, int]]) -> bool:
    """Place region at random valid offset with 8-separation from existing grey."""
    H, W = grid.shape
    norm_pts, ph, pw = _normalize_points(pts)
    if ph > H or pw > W:
        return False

    occ = ndimage.binary_dilation(grid != Color.ORANGE, structure=STRUCT8, iterations=1)
    valid = []
    for r0 in range(H - ph + 1):
        for c0 in range(W - pw + 1):
            if all(not occ[r0 + dr, c0 + dc] for dr, dc in norm_pts):
                valid.append((r0, c0))
    if not valid:
        return False

    r0, c0 = random.choice(valid)
    for dr, dc in norm_pts:
        grid[r0 + dr, c0 + dc] = Color.GREY
    return True


def _place_target_regions(grid: np.ndarray, H: int, W: int, num_targets: int) -> int:
    targets_placed = 0
    for _ in range(num_targets):
        pts = _make_target_region(H, W)
        if pts is None:
            continue
        if _place_region(grid, pts):
            targets_placed += 1
    return targets_placed


def _density_ok(grid: np.ndarray) -> bool:
    H, W = grid.shape
    density = np.sum(grid == Color.GREY) / (H * W)
    return MIN_GREY_DENSITY <= density <= MAX_GREY_DENSITY


def _oracle_accepts(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False

    if np.array_equal(out, grid) or Color.GREY not in out:
        return False

    lab8, n8 = ndimage.label(grid == Color.GREY, structure=STRUCT8)
    for k in range(1, n8 + 1):
        mask = lab8 == k
        diag_run = _gen_max_straight_diagonal_run(_gen_mask_points(mask))
        if np.all(out[mask] == Color.TEAL):
            if diag_run < MIN_DIAGONAL_RUN:
                return False
        elif diag_run >= MIN_DIAGONAL_RUN:
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_ROWS_RANGE)
    W = random.randint(*GRID_COLS_RANGE)
    grid = np.full((H, W), Color.ORANGE, dtype=int)

    num_targets = random.randint(*TARGET_COUNT_RANGE)
    num_distractors = random.randint(*DISTRACTOR_COUNT_RANGE)

    if _place_target_regions(grid, H, W, num_targets) == 0:
        return None
    for _ in range(num_distractors):
        _place_region(grid, _make_distractor())

    if not _density_ok(grid):
        return None
    if not _oracle_accepts(grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPT_LIMIT):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {GEN_ATTEMPT_LIMIT} attempts")
