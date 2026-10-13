from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
YELLOW = 4
GREY = 5
ORANGE = 7

PLUS_OFFSETS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
GEN_RETRIES = 400
GRID_RANGE = (18, 25)
SCENARIOS = ("disjoint", "high_low_overlap", "row_tie_overlap")
SCENARIO_PROBS = (0.25, 0.35, 0.40)


def _neighbors8(r: int, c: int) -> Iterable[tuple[int, int]]:
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        comp: list[tuple[int, int]] = []
        while q:
            rr, cc = q.popleft()
            comp.append((rr, cc))
            for nr, nc in _neighbors8(rr, cc):
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _plus_cells(shape: tuple[int, int], r: int, c: int) -> list[tuple[int, int]] | None:
    h, w = shape
    if not (1 <= r < h - 1 and 1 <= c < w - 1):
        return None
    return [(r + dr, c + dc) for dr, dc in PLUS_OFFSETS]


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _candidate_centers_for_component(grid: np.ndarray, comp: list[tuple[int, int]]) -> list[tuple[int, int]]:
    h, w = grid.shape
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    comp_set = set(comp)
    candidates: list[tuple[int, int]] = []
    for r in range(max(1, r0 - 1), min(h - 1, r1 + 2)):
        for c in range(max(1, c0 - 1), min(w - 1, c1 + 2)):
            plus = _plus_cells(grid.shape, r, c)
            if plus is None:
                continue
            plus_set = set(plus)
            if not comp_set <= plus_set:
                continue
            if all(int(grid[rr, cc]) in {YELLOW, GREY} for rr, cc in plus):
                candidates.append((r, c))
    return candidates


def _extract_crosses(grid: np.ndarray) -> list[dict]:
    crosses: list[dict] = []
    for comp in _components(grid == YELLOW):
        candidates = _candidate_centers_for_component(grid, comp)
        assert len(candidates) == 1, "expected each yellow component to imply one unique plus center"
        center = candidates[0]
        plus = _plus_cells(grid.shape, center[0], center[1])
        assert plus is not None
        k = sum(int(grid[r, c] == YELLOW) for r, c in plus)
        assert 1 <= k <= 5
        crosses.append(
            {
                "center": center,
                "k": k,
                "yellow": tuple(sorted(comp)),
                "plus": tuple(plus),
            }
        )
    assert crosses, "expected at least one yellow marker component"
    return crosses


def _diamond_ring_cells(shape: tuple[int, int], center: tuple[int, int], dist: int) -> list[tuple[int, int]]:
    h, w = shape
    r0, c0 = center
    cells: list[tuple[int, int]] = []
    for dr in range(-dist, dist + 1):
        rem = dist - abs(dr)
        rr = r0 + dr
        if not (0 <= rr < h):
            continue
        if rem == 0:
            cc = c0
            if 0 <= cc < w:
                cells.append((rr, cc))
            continue
        c_left = c0 - rem
        c_right = c0 + rem
        if 0 <= c_left < w:
            cells.append((rr, c_left))
        if 0 <= c_right < w:
            cells.append((rr, c_right))
    return cells


def _stamp_color_at(cross: dict, cell: tuple[int, int]) -> int | None:
    dist = _manhattan(cross["center"], cell)
    if dist >= cross["k"]:
        return None
    return BLUE if dist % 2 == 0 else ORANGE


def _render_output(shape: tuple[int, int], crosses: list[dict]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    order = sorted(crosses, key=lambda cross: (cross["k"], -cross["center"][0], -cross["center"][1]))
    for cross in order:
        for dist in range(cross["k"]):
            color = BLUE if dist % 2 == 0 else ORANGE
            for r, c in _diamond_ring_cells(shape, cross["center"], dist):
                out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer each partial yellow plus center and stamp alternating concentric Manhattan rings whose depth equals the marker completeness.

    Input structure:
    - The grid is a black background containing only grey and yellow cross-marker cells.
    - Each 8-connected yellow component is a subset of one interior five-cell plus, with the non-yellow cells of that plus shown in grey.
    - Different yellow components are separated and each implies a unique plus center.

    Rule:
    - For every yellow component, find the unique center whose center/up/down/left/right cells are all in-bounds and all grey or yellow.
    - Let k be the number of yellow cells in that plus.
    - Draw k concentric Manhattan-distance layers around that center: distance 0 is layer 1, distance 1 is layer 2, and so on.
    - Color odd-numbered layers blue and even-numbered layers orange.
    - If stamped layers overlap, higher k overrides lower k; if k ties, the smaller row index overrides the larger row index.

    Key insight:
    - The same partial yellow plus plays two roles at once: it locates the stamp center and its yellow-cell count directly encodes the stamp depth.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"
    assert set(map(int, np.unique(g))) <= {BLACK, YELLOW, GREY}, "expected only black, yellow, and grey input colors"
    crosses = _extract_crosses(g)
    return _render_output(g.shape, crosses)


def _is_local_connected(indices: tuple[int, ...]) -> bool:
    cells = [PLUS_OFFSETS[i] for i in indices]
    target = set(cells)
    seen = {cells[0]}
    q = deque([cells[0]])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors8(r, c):
            if (nr, nc) in target and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return seen == target


PATTERNS_BY_K: dict[int, list[tuple[int, ...]]] = {k: [] for k in range(1, 6)}
for _bits in range(1, 1 << len(PLUS_OFFSETS)):
    _pattern = tuple(i for i in range(len(PLUS_OFFSETS)) if _bits & (1 << i))
    if _is_local_connected(_pattern):
        PATTERNS_BY_K[len(_pattern)].append(_pattern)


def _choose_pattern(rng: np.random.Generator, k: int, require_center: bool | None = None) -> tuple[int, ...]:
    patterns = PATTERNS_BY_K[k]
    if require_center is not None:
        center_patterns = [pattern for pattern in patterns if ((0 in pattern) == require_center)]
        if center_patterns:
            patterns = center_patterns
    return patterns[int(rng.integers(len(patterns)))]


def _materialize_cross(center: tuple[int, int], k: int, pattern: tuple[int, ...]) -> dict:
    plus = [(center[0] + dr, center[1] + dc) for dr, dc in PLUS_OFFSETS]
    yellow = [plus[i] for i in pattern]
    return {
        "center": center,
        "k": k,
        "plus": tuple(plus),
        "yellow": tuple(sorted(yellow)),
    }


def _render_input(shape: tuple[int, int], crosses: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for cross in crosses:
        for r, c in cross["plus"]:
            grid[r, c] = GREY
        for r, c in cross["yellow"]:
            grid[r, c] = YELLOW
    return grid


def _covers_nonblack(crosses: list[dict], cell: tuple[int, int]) -> bool:
    return any(_stamp_color_at(cross, cell) is not None for cross in crosses)


def _pairwise_overlap_cells(cross_a: dict, cross_b: dict) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    min_r = min(cross_a["center"][0] - (cross_a["k"] - 1), cross_b["center"][0] - (cross_b["k"] - 1))
    max_r = max(cross_a["center"][0] + (cross_a["k"] - 1), cross_b["center"][0] + (cross_b["k"] - 1))
    min_c = min(cross_a["center"][1] - (cross_a["k"] - 1), cross_b["center"][1] - (cross_b["k"] - 1))
    max_c = max(cross_a["center"][1] + (cross_a["k"] - 1), cross_b["center"][1] + (cross_b["k"] - 1))
    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            if _stamp_color_at(cross_a, (r, c)) is not None and _stamp_color_at(cross_b, (r, c)) is not None:
                cells.append((r, c))
    return cells


def _pair_has_visible_conflict(cross_a: dict, cross_b: dict) -> bool:
    for cell in _pairwise_overlap_cells(cross_a, cross_b):
        if _stamp_color_at(cross_a, cell) != _stamp_color_at(cross_b, cell):
            return True
    return False


def _can_place_center(
    center: tuple[int, int],
    k: int,
    placed: list[dict],
    *,
    avoid_output_overlap: bool,
) -> bool:
    for other in placed:
        dist = _manhattan(center, other["center"])
        if dist < 6:
            return False
        if avoid_output_overlap and dist <= (k - 1) + (other["k"] - 1):
            return False
    return True


def _sample_extra_centers(
    rng: np.random.Generator,
    shape: tuple[int, int],
    placed: list[dict],
    ks: list[int],
) -> list[tuple[int, int]] | None:
    h, w = shape
    candidates = [(r, c) for r in range(4, h - 4) for c in range(4, w - 4)]
    rng.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    temp = list(placed)
    for k in ks:
        found = None
        for center in candidates:
            if center in chosen:
                continue
            if _can_place_center(center, k, temp, avoid_output_overlap=True):
                found = center
                break
        if found is None:
            return None
        chosen.append(found)
        temp.append({"center": found, "k": k})
    return chosen


def _build_cross_specs(rng: np.random.Generator, shape: tuple[int, int]) -> tuple[list[dict], str]:
    scenario = str(rng.choice(SCENARIOS, p=SCENARIO_PROBS))
    h, w = shape
    if scenario == "high_low_overlap":
        row = int(rng.integers(4, h - 4))
        col = int(rng.integers(4, w - 11))
        crosses = [
            _materialize_cross((row, col), 5, _choose_pattern(rng, 5, require_center=True)),
            _materialize_cross((row, col + 7), 4, _choose_pattern(rng, 4, require_center=False)),
        ]
        extra_ks = [int(x) for x in rng.choice([1, 2, 3], size=int(rng.integers(0, 3)), replace=False)]
    elif scenario == "row_tie_overlap":
        row = int(rng.integers(4, h - 11))
        col = int(rng.integers(4, w - 4))
        crosses = [
            _materialize_cross((row, col), 5, _choose_pattern(rng, 5, require_center=True)),
            _materialize_cross((row + 7, col), 5, _choose_pattern(rng, 5, require_center=True)),
        ]
        extra_pool = [1, 2, 3, 4]
        rng.shuffle(extra_pool)
        extra_ks = extra_pool[: int(rng.integers(1, 3))]
    else:
        count = int(rng.integers(2, 5))
        k_pool = [1, 2, 3, 4, 5]
        rng.shuffle(k_pool)
        ks = k_pool[:count]
        centers = _sample_extra_centers(rng, shape, [], ks)
        assert centers is not None
        crosses = []
        missing_candidates = [i for i, k in enumerate(ks) if k < 5]
        missing_idx = missing_candidates[int(rng.integers(len(missing_candidates)))]
        present_idx = int(rng.integers(count))
        if present_idx == missing_idx:
            present_idx = (present_idx + 1) % count
        for i, (center, k) in enumerate(zip(centers, ks)):
            require_center = None
            if k < 5 and i == missing_idx:
                require_center = False
            elif i == present_idx:
                require_center = True
            crosses.append(_materialize_cross(center, k, _choose_pattern(rng, k, require_center=require_center)))
        return crosses, scenario

    if len(crosses) + len(extra_ks) > 4:
        extra_ks = extra_ks[: 4 - len(crosses)]
    extra_centers = _sample_extra_centers(rng, shape, crosses, extra_ks)
    if extra_ks and extra_centers is None:
        raise AssertionError("failed to place extra non-overlapping crosses")
    for center, k in zip(extra_centers or [], extra_ks):
        has_missing_center = any(cross["center"] not in cross["yellow"] for cross in crosses)
        require_center = False if k < 5 and (scenario == "row_tie_overlap" or not has_missing_center) else None
        crosses.append(_materialize_cross(center, k, _choose_pattern(rng, k, require_center=require_center)))
    return crosses, scenario


def _validate_generated_grid(grid: np.ndarray, crosses: list[dict], scenario: str) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if not (2 <= len(crosses) <= 4):
        return False
    if len({cross["k"] for cross in crosses}) < 2:
        return False
    if not all(_can_place_center(cross["center"], cross["k"], crosses[:i], avoid_output_overlap=False) for i, cross in enumerate(crosses)):
        return False
    if not any(cross["center"] not in cross["yellow"] for cross in crosses):
        return False
    if not any(cross["center"] in cross["yellow"] for cross in crosses):
        return False

    for cross in crosses:
        if any(not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]) for r, c in cross["plus"]):
            return False

    extracted = _extract_crosses(grid)
    expected_signature = sorted((cross["center"], cross["k"], tuple(sorted(cross["yellow"]))) for cross in crosses)
    observed_signature = sorted((cross["center"], cross["k"], tuple(sorted(cross["yellow"]))) for cross in extracted)
    if expected_signature != observed_signature:
        return False

    nonblack = np.argwhere(grid != BLACK)
    if nonblack.size == 0:
        return False
    for r, c in nonblack:
        if not _covers_nonblack(crosses, (int(r), int(c))):
            return False

    overlaps = []
    for i in range(len(crosses)):
        for j in range(i + 1, len(crosses)):
            shared = _pairwise_overlap_cells(crosses[i], crosses[j])
            if shared:
                overlaps.append((i, j, shared))
            if crosses[i]["k"] == crosses[j]["k"] and crosses[i]["center"][0] == crosses[j]["center"][0] and shared:
                return False

    if scenario == "disjoint":
        if overlaps:
            return False
    elif scenario == "high_low_overlap":
        if not overlaps or crosses[0]["k"] <= crosses[1]["k"] or not _pair_has_visible_conflict(crosses[0], crosses[1]):
            return False
    elif scenario == "row_tie_overlap":
        if not overlaps or crosses[0]["k"] != crosses[1]["k"] or crosses[0]["center"][0] >= crosses[1]["center"][0]:
            return False
        if not _pair_has_visible_conflict(crosses[0], crosses[1]):
            return False

    expected = _render_output(grid.shape, crosses)
    try:
        actual = main(grid)
    except AssertionError:
        return False
    return bool(np.array_equal(actual, expected))


def _generate_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    shape = (
        int(rng.integers(*GRID_RANGE)),
        int(rng.integers(*GRID_RANGE)),
    )
    try:
        crosses, scenario = _build_cross_specs(rng, shape)
    except AssertionError:
        return None
    grid = _render_input(shape, crosses)
    if _validate_generated_grid(grid, crosses, scenario):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
