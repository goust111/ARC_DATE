import random
from functools import lru_cache
from itertools import combinations

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
MAGENTA = 6
ORANGE = 7
CYAN = 8
MASK = 9

GRID_SIZE = 11
GENERATE_RETRIES = 400
ARM_RULES = ((-1, 0, RED), (0, -1, ORANGE), (0, 1, MAGENTA), (1, 0, CYAN))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _d4_orbit(size: int, r: int, c: int) -> tuple[tuple[int, int], ...]:
    last = size - 1
    return tuple(
        sorted(
            {
                (r, c),
                (c, last - r),
                (last - r, last - c),
                (last - c, r),
                (r, last - c),
                (last - r, c),
                (c, r),
                (last - c, last - r),
            }
        )
    )


@lru_cache(maxsize=None)
def _orbit_data(size: int) -> tuple[tuple[tuple[int, int], ...], dict[tuple[int, int], int]]:
    seen: dict[tuple[tuple[int, int], ...], int] = {}
    orbits: list[tuple[tuple[int, int], ...]] = []
    cell_to_orbit: dict[tuple[int, int], int] = {}

    for r in range(size):
        for c in range(size):
            orbit = _d4_orbit(size, r, c)
            if orbit not in seen:
                seen[orbit] = len(orbits)
                orbits.append(orbit)
            cell_to_orbit[(r, c)] = seen[orbit]

    return tuple(orbits), cell_to_orbit


def _complete_masked_grid(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    n, w = g.shape
    assert n == w and n % 2 == 1, "expected an odd square grid"

    orbits, _ = _orbit_data(n)
    out = g.copy()
    for orbit in orbits:
        if not any(int(g[r, c]) == MASK for r, c in orbit):
            continue
        visible = {int(g[r, c]) for r, c in orbit if int(g[r, c]) != MASK}
        assert len(visible) == 1, "each masked D4 orbit must expose exactly one visible color"
        fill = visible.pop()
        for r, c in orbit:
            if int(g[r, c]) == MASK:
                out[r, c] = fill
    return out


def _blue_positions(grid: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, cell)) for cell in np.argwhere(np.asarray(grid, dtype=int) == BLUE)]


def _stamp_crosses(grid: np.ndarray, anchors: list[tuple[int, int]] | None = None) -> np.ndarray:
    base = np.asarray(grid, dtype=int)
    out = base.copy()
    blue_mask = base == BLUE
    if anchors is None:
        anchors = _blue_positions(base)

    for r, c in anchors:
        for dr, dc, color in ARM_RULES:
            nr = r + dr
            nc = c + dc
            if _in_bounds(base.shape, nr, nc) and not blue_mask[nr, nc]:
                out[nr, nc] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the masked eightfold-symmetric pattern, then stamp direction-colored crosses around every blue anchor.

    Input structure:
    - An 11x11 black background square uses red, green, and blue foreground pixels plus maroon mask pixels.
    - Ignoring mask pixels, the full pattern is invariant under the eight symmetries of the square.
    - Some mask pixels hide blue anchors, while every masked D4 orbit still exposes one visible color elsewhere.

    Rule:
    - For every mask pixel, take its full D4 symmetry orbit and replace the mask with that orbit's unique visible non-mask color.
    - After restoring the full pattern, collect every blue pixel as an anchor.
    - Around each anchor, stamp a four-neighbor cross: up becomes red, left orange, right magenta, and down cyan.
    - Clip cross arms at the grid boundary, never overwrite blue pixels, and let cross arms overwrite any other color.

    Key insight:
    - The hidden blue anchors are not given directly; they only become available after completing each eightfold symmetry orbit, and those newly revealed anchors create extra cross arms in the final output.
    """
    completed = _complete_masked_grid(np.asarray(grid, dtype=int))
    return _stamp_crosses(completed)


_ORBITS_11, _CELL_TO_ORBIT_11 = _orbit_data(GRID_SIZE)
_CENTER_ORBIT_ID = _CELL_TO_ORBIT_11[(GRID_SIZE // 2, GRID_SIZE // 2)]


def _orbit_touches_boundary(orbit: tuple[tuple[int, int], ...], size: int = GRID_SIZE) -> bool:
    last = size - 1
    return any(r in (0, last) or c in (0, last) for r, c in orbit)


def _all_far_apart(cells: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> bool:
    for i, a in enumerate(cells):
        for b in cells[i + 1 :]:
            if _manhattan(a, b) <= 2:
                return False
    return True


_BLUE_CANDIDATE_IDS = tuple(
    orbit_id
    for orbit_id, orbit in enumerate(_ORBITS_11)
    if _all_far_apart(list(orbit))
)

_BLUE_COMBOS = tuple(
    combo
    for count in (1, 2)
    for combo in combinations(_BLUE_CANDIDATE_IDS, count)
    if 4 <= sum(len(_ORBITS_11[oid]) for oid in combo) <= 8
    and any(len(_ORBITS_11[oid]) > 1 for oid in combo)
    and _all_far_apart([cell for oid in combo for cell in _ORBITS_11[oid]])
)


def _arm_cells(blue_cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    arm_cells: set[tuple[int, int]] = set()
    for r, c in blue_cells:
        for dr, dc, _color in ARM_RULES:
            nr = r + dr
            nc = c + dc
            if _in_bounds((GRID_SIZE, GRID_SIZE), nr, nc):
                arm_cells.add((nr, nc))
    return arm_cells - blue_cells


def _paint_orbit(grid: np.ndarray, orbit_id: int, color: int) -> None:
    for r, c in _ORBITS_11[orbit_id]:
        grid[r, c] = color


def _choose_blue_combo() -> tuple[tuple[int, ...], bool]:
    want_clipped = random.random() < 0.65
    combos = [
        combo
        for combo in _BLUE_COMBOS
        if any(_orbit_touches_boundary(_ORBITS_11[oid]) for oid in combo) == want_clipped
    ]
    if not combos:
        combos = list(_BLUE_COMBOS)

    if random.random() < 0.25:
        centered = [combo for combo in combos if _CENTER_ORBIT_ID in combo]
        if centered:
            combos = centered
    else:
        off_center = [combo for combo in combos if _CENTER_ORBIT_ID not in combo]
        if off_center:
            combos = off_center

    return random.choice(combos), want_clipped


def _choose_nonblue_orbits(
    blue_ids: tuple[int, ...]
) -> tuple[list[int], list[int]] | None:
    blue_cells = {cell for oid in blue_ids for cell in _ORBITS_11[oid]}
    arm_cells = _arm_cells(blue_cells)
    arm_orbit_ids = list({_CELL_TO_ORBIT_11[cell] for cell in arm_cells} - set(blue_ids))
    filler_orbit_ids = [
        orbit_id
        for orbit_id in range(len(_ORBITS_11))
        if orbit_id not in blue_ids and orbit_id not in arm_orbit_ids
    ]
    if not arm_orbit_ids or not filler_orbit_ids:
        return None

    random.shuffle(arm_orbit_ids)
    random.shuffle(filler_orbit_ids)
    arm_orbit_ids.sort(key=lambda oid: len(_ORBITS_11[oid]), reverse=True)
    filler_orbit_ids.sort(key=lambda oid: len(_ORBITS_11[oid]), reverse=True)

    overwrite_ids = [arm_orbit_ids.pop(0)]
    preserve_ids = [filler_orbit_ids.pop(0)]

    def totals() -> tuple[int, int, int]:
        selected = list(blue_ids) + overwrite_ids + preserve_ids
        total_cells = sum(len(_ORBITS_11[oid]) for oid in selected)
        max_masks = sum(len(_ORBITS_11[oid]) - 1 for oid in selected)
        nonblue_count = len(overwrite_ids) + len(preserve_ids)
        return total_cells, max_masks, nonblue_count

    while True:
        total_cells, max_masks, nonblue_count = totals()
        enough = total_cells >= 16 and max_masks >= 8 and nonblue_count >= 3
        want_extra = nonblue_count < 4 and random.random() < 0.35
        if enough and not want_extra:
            break
        if not arm_orbit_ids and not filler_orbit_ids:
            break
        prefer_filler = filler_orbit_ids and (
            not arm_orbit_ids or len(preserve_ids) <= len(overwrite_ids) or random.random() < 0.60
        )
        if prefer_filler:
            preserve_ids.append(filler_orbit_ids.pop(0))
        else:
            overwrite_ids.append(arm_orbit_ids.pop(0))

    total_cells, max_masks, nonblue_count = totals()
    if total_cells < 16 or max_masks < 8 or nonblue_count < 3 or total_cells > 28:
        return None
    return overwrite_ids, preserve_ids


def _build_clean_grid(
    blue_ids: tuple[int, ...], overwrite_ids: list[int], preserve_ids: list[int]
) -> np.ndarray:
    clean = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    for orbit_id in blue_ids:
        _paint_orbit(clean, orbit_id, BLUE)

    primary, secondary = (RED, GREEN) if random.random() < 0.5 else (GREEN, RED)
    for index, orbit_id in enumerate(overwrite_ids):
        _paint_orbit(clean, orbit_id, primary if index % 2 == 0 else secondary)
    for index, orbit_id in enumerate(preserve_ids):
        _paint_orbit(clean, orbit_id, secondary if index % 2 == 0 else primary)
    return clean


def _choose_masked_cells(
    selected_ids: list[int], blue_ids: tuple[int, ...], nonblue_ids: list[int]
) -> set[tuple[int, int]] | None:
    capacities = {
        orbit_id: len(_ORBITS_11[orbit_id]) - 1
        for orbit_id in selected_ids
        if len(_ORBITS_11[orbit_id]) > 1
    }
    max_masks = sum(capacities.values())
    if max_masks < 8:
        return None

    target_masks = random.randint(8, min(15, max_masks))
    counts = {orbit_id: 0 for orbit_id in selected_ids}

    hidden_blue_choices = [orbit_id for orbit_id in blue_ids if capacities.get(orbit_id, 0) > 0]
    hidden_nonblue_choices = [
        orbit_id for orbit_id in nonblue_ids if capacities.get(orbit_id, 0) > 0
    ]
    if not hidden_blue_choices or not hidden_nonblue_choices:
        return None

    blue_orbit_id = random.choice(hidden_blue_choices)
    nonblue_orbit_id = random.choice(hidden_nonblue_choices)
    counts[blue_orbit_id] += 1
    capacities[blue_orbit_id] -= 1
    counts[nonblue_orbit_id] += 1
    capacities[nonblue_orbit_id] -= 1
    target_masks -= 2

    while target_masks > 0:
        candidates = [orbit_id for orbit_id, cap in capacities.items() if cap > 0]
        if not candidates:
            return None
        weights = []
        for orbit_id in candidates:
            weight = len(_ORBITS_11[orbit_id])
            if orbit_id in blue_ids:
                weight += 2
            weights.append(weight)
        orbit_id = random.choices(candidates, weights=weights, k=1)[0]
        counts[orbit_id] += 1
        capacities[orbit_id] -= 1
        target_masks -= 1

    masked_cells: set[tuple[int, int]] = set()
    for orbit_id, count in counts.items():
        if count == 0:
            continue
        cells = list(_ORBITS_11[orbit_id])
        random.shuffle(cells)
        masked_cells.update(cells[:count])
    return masked_cells


def _semantic_checks(
    inp: np.ndarray, clean: np.ndarray, expected: np.ndarray, want_clipped: bool
) -> bool:
    if not np.array_equal(_complete_masked_grid(inp), clean):
        return False
    if not np.array_equal(main(inp), expected):
        return False

    masked = inp == MASK
    nonblue = np.isin(clean, (RED, GREEN))
    if not 8 <= int(masked.sum()) <= 15:
        return False
    if not np.any(masked & (clean == BLUE)):
        return False
    if np.count_nonzero(inp == BLUE) < 2:
        return False
    if not np.any(masked & nonblue):
        return False
    if not np.any(nonblue & (expected != clean)):
        return False
    if not np.any(nonblue & (expected == clean)):
        return False
    if not np.any((expected == RED) & (clean != RED)):
        return False
    if not np.any(expected == ORANGE):
        return False
    if not np.any(expected == MAGENTA):
        return False
    if not np.any(expected == CYAN):
        return False

    visible_anchor_output = _stamp_crosses(clean, _blue_positions(inp))
    if np.array_equal(visible_anchor_output, expected):
        return False

    blue_cells = _blue_positions(clean)
    has_boundary_anchor = any(
        r in (0, GRID_SIZE - 1) or c in (0, GRID_SIZE - 1) for r, c in blue_cells
    )
    if has_boundary_anchor != want_clipped:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    blue_ids, want_clipped = _choose_blue_combo()
    chosen_nonblue = _choose_nonblue_orbits(blue_ids)
    if chosen_nonblue is None:
        return None
    overwrite_ids, preserve_ids = chosen_nonblue

    clean = _build_clean_grid(blue_ids, overwrite_ids, preserve_ids)
    selected_ids = list(blue_ids) + overwrite_ids + preserve_ids
    masked_cells = _choose_masked_cells(selected_ids, blue_ids, overwrite_ids + preserve_ids)
    if masked_cells is None:
        return None

    inp = clean.copy()
    for r, c in masked_cells:
        inp[r, c] = MASK

    expected = _stamp_crosses(clean)
    return inp if _semantic_checks(inp, clean, expected, want_clipped) else None


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        if (grid := _gen_attempt()) is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for the symmetry-completion stamp puzzle")
