from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


GREY = 5
GEN_RETRIES = 400
GRID_H_RANGE = (14, 28)
GRID_W_RANGE = (14, 28)
FRAME_COUNT_CHOICES = (1, 2, 3, 4)
OUTER_HEIGHT_RANGE = (5, 10)
OUTER_WIDTH_RANGE = (5, 10)


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _is_rectangular_frame(grid: np.ndarray, color: int, coords: np.ndarray) -> tuple[int, int, int, int] | None:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return None
    expected_area = 2 * (h + w) - 4
    if int(coords.shape[0]) != expected_area:
        return None

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if not np.all(sub[border] == int(color)):
        return None
    if np.any(sub[1:-1, 1:-1] == int(color)):
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _find_frames(grid: np.ndarray, bg: int) -> list[tuple[int, int, int, int, int]]:
    frames: list[tuple[int, int, int, int, int]] = []
    if GREY == bg:
        return frames
    for comp in _components(grid == GREY):
        bbox = _is_rectangular_frame(grid, GREY, comp)
        if bbox is None:
            continue
        frames.append((GREY, *bbox))
    frames.sort(key=lambda item: (item[1], item[2], item[3], item[4]))
    return frames


def _majority_and_minority(values: np.ndarray) -> tuple[int, int]:
    colors, counts = np.unique(values, return_counts=True)
    assert colors.size == 2, "expected exactly two interior colors"
    order = np.argsort(counts)
    minority_idx = int(order[0])
    majority_idx = int(order[1])
    assert int(counts[majority_idx]) != int(counts[minority_idx]), "expected a strict majority"
    return int(colors[majority_idx]), int(colors[minority_idx])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each closed grey frame to the majority interior color, then point-reflect minority pixels.

    Input structure:
    - The dominant color is the background.
    - One-pixel-thick grey rectangular frames are closed and pairwise separated.
    - Each frame interior contains single pixels in exactly two colors with unequal counts.

    Rule:
    - Recolor the full frame border to the majority interior color.
    - For each minority pixel, mirror it through the interior rectangle center.
    - If the mirrored cell is occupied in the original input, keep the pixel in place.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _background_color(g)
    frames = _find_frames(g, bg)
    assert frames, "expected at least one rectangular frame"

    out = g.copy()
    for frame_color, r0, c0, r1, c1 in frames:
        assert frame_color == GREY, "expected grey frames"
        ir0, ic0 = r0 + 1, c0 + 1
        ir1, ic1 = r1 - 1, c1 - 1
        interior = g[ir0 : ir1 + 1, ic0 : ic1 + 1]
        occupied = interior != bg
        values = interior[occupied]
        assert values.size >= 2, "expected at least two interior pixels"

        majority, minority = _majority_and_minority(values)
        out[r0 : r1 + 1, c0] = majority
        out[r0 : r1 + 1, c1] = majority
        out[r0, c0 : c1 + 1] = majority
        out[r1, c0 : c1 + 1] = majority

        minority_points = np.argwhere(interior == minority)
        for lr, lc in minority_points:
            tr = int(ir0 + ir1 - (ir0 + int(lr)))
            tc = int(ic0 + ic1 - (ic0 + int(lc)))
            if g[tr, tc] != bg:
                continue
            out[ir0 + int(lr), ic0 + int(lc)] = bg
            out[tr, tc] = minority

    return out


def _orbits(height: int, width: int) -> list[tuple[tuple[int, int], ...]]:
    seen: set[tuple[int, int]] = set()
    result: list[tuple[tuple[int, int], ...]] = []
    for r in range(height):
        for c in range(width):
            if (r, c) in seen:
                continue
            mirror = (height - 1 - r, width - 1 - c)
            orbit = tuple(sorted({(r, c), mirror}))
            seen.update(orbit)
            result.append(orbit)
    return result


def _paint_state(
    interior: np.ndarray,
    orbit: tuple[tuple[int, int], ...],
    state: str,
    majority: int,
    minority: int,
) -> tuple[int, int]:
    if state == "free":
        src, _dst = random.sample(list(orbit), 2)
        interior[src] = minority
        return 0, 1
    if state == "blocked_majority":
        src, dst = random.sample(list(orbit), 2)
        interior[src] = minority
        interior[dst] = majority
        return 1, 1
    if state == "blocked_pair":
        for cell in orbit:
            interior[cell] = minority
        return 0, len(orbit)
    if state == "center_minority":
        assert len(orbit) == 1
        interior[orbit[0]] = minority
        return 0, 1
    if state == "majority_one":
        cell = random.choice(list(orbit))
        interior[cell] = majority
        return 1, 0
    if state == "majority_two":
        for cell in orbit:
            interior[cell] = majority
        return len(orbit), 0
    raise AssertionError(f"unknown state {state}")


def _sample_interior(height: int, width: int, majority: int, minority: int, scenario: str) -> np.ndarray:
    interior = np.zeros((height, width), dtype=int)
    all_orbits = _orbits(height, width)
    pair_orbits = [orbit for orbit in all_orbits if len(orbit) == 2]
    single_orbits = [orbit for orbit in all_orbits if len(orbit) == 1]
    random.shuffle(pair_orbits)
    random.shuffle(single_orbits)

    total_cells = height * width
    max_minority = max(1, (total_cells - 1) // 2)
    majority_count = 0
    minority_count = 0
    used: set[tuple[tuple[int, int], ...]] = set()

    def take_pair() -> tuple[tuple[int, int], ...] | None:
        while pair_orbits:
            orbit = pair_orbits.pop()
            if orbit not in used:
                used.add(orbit)
                return orbit
        return None

    def take_single() -> tuple[tuple[int, int], ...] | None:
        while single_orbits:
            orbit = single_orbits.pop()
            if orbit not in used:
                used.add(orbit)
                return orbit
        return None

    def minority_add(state: str) -> int:
        if state in {"free", "blocked_majority", "center_minority"}:
            return 1
        if state == "blocked_pair":
            return 2
        raise AssertionError(f"unknown minority state {state}")

    def majority_add(state: str, orbit_len: int) -> int:
        if state == "blocked_majority":
            return 1
        if state == "blocked_pair":
            return 0
        if state in {"free", "center_minority"}:
            return 0
        if state == "majority_one":
            return 1
        if state == "majority_two":
            return orbit_len
        raise AssertionError(f"unknown state {state}")

    def remaining_slots() -> int:
        return 2 * len(pair_orbits) + len(single_orbits)

    def feasible(state: str, orbit_len: int) -> bool:
        new_minority = minority_count + minority_add(state)
        if new_minority > max_minority:
            return False
        new_majority = majority_count + majority_add(state, orbit_len)
        return new_majority + (remaining_slots() - orbit_len) > new_minority

    def apply_required(state: str) -> None:
        nonlocal majority_count, minority_count
        if state == "center_minority":
            orbit = None
            if single_orbits and feasible("center_minority", 1):
                orbit = take_single()
                assert orbit is not None
                maj_add, min_add = _paint_state(interior, orbit, state, majority, minority)
            else:
                assert pair_orbits and feasible("blocked_pair", 2)
                orbit = take_pair()
                assert orbit is not None
                maj_add, min_add = _paint_state(interior, orbit, "blocked_pair", majority, minority)
        else:
            assert feasible(state, 2)
            orbit = take_pair()
            assert orbit is not None
            maj_add, min_add = _paint_state(interior, orbit, state, majority, minority)
        majority_count += maj_add
        minority_count += min_add

    required_map = {
        "free": ["free"],
        "blocked": ["blocked_majority"],
        "pair": ["blocked_pair"],
        "center": ["center_minority"],
        "mixed": ["free", random.choice(["blocked_majority", "blocked_pair", "center_minority"])],
    }
    for state in required_map[scenario]:
        apply_required(state)

    target_minority = random.randint(minority_count, min(max_minority, minority_count + 3))
    while minority_count < target_minority:
        choices = []
        if pair_orbits and minority_count + 1 <= target_minority:
            for state in ("free", "blocked_majority"):
                if feasible(state, 2):
                    choices.append(state)
        if pair_orbits and minority_count + 2 <= target_minority and feasible("blocked_pair", 2):
            choices.append("blocked_pair")
        if single_orbits and minority_count + 1 <= target_minority and feasible("center_minority", 1):
            choices.append("center_minority")
        if not choices:
            break
        state = random.choice(choices)
        if state == "center_minority":
            orbit = take_single()
        else:
            orbit = take_pair()
        if orbit is None:
            continue
        maj_add, min_add = _paint_state(interior, orbit, state, majority, minority)
        majority_count += maj_add
        minority_count += min_add

    open_orbits = [orbit for orbit in all_orbits if orbit not in used]
    random.shuffle(open_orbits)
    capacity = majority_count + sum(len(orbit) for orbit in open_orbits)
    lower = max(majority_count, minority_count + 1)
    upper = min(capacity, minority_count + 6)
    assert lower <= capacity
    target_majority = lower if upper < lower else random.randint(lower, upper)
    while majority_count < target_majority and open_orbits:
        orbit = open_orbits.pop()
        if len(orbit) == 1:
            state = "majority_one"
        else:
            state = random.choice(["majority_one", "majority_two"])
        maj_add, min_add = _paint_state(interior, orbit, state, majority, minority)
        majority_count += maj_add
        minority_count += min_add

    if majority_count <= minority_count:
        for orbit in open_orbits:
            state = "majority_one" if len(orbit) == 1 or random.random() < 0.5 else "majority_two"
            maj_add, min_add = _paint_state(interior, orbit, state, majority, minority)
            majority_count += maj_add
            minority_count += min_add
            if majority_count > minority_count:
                break

    assert majority_count > minority_count >= 1
    return interior


def _frame_intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _place_frames(height: int, width: int, count: int) -> list[tuple[int, int, int, int]] | None:
    placed: list[tuple[int, int, int, int]] = []
    for _ in range(count):
        success = False
        for _attempt in range(120):
            outer_h = random.randint(*OUTER_HEIGHT_RANGE)
            outer_w = random.randint(*OUTER_WIDTH_RANGE)
            if outer_h > height or outer_w > width:
                continue
            r0 = random.randint(0, height - outer_h)
            c0 = random.randint(0, width - outer_w)
            bbox = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
            if any(_frame_intersects(bbox, other, pad=1) for other in placed):
                continue
            placed.append(bbox)
            success = True
            break
        if not success:
            return None
    return placed


def _sample_case() -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    grid_w = random.randint(*GRID_W_RANGE)
    bg_choices = [c for c in range(10) if c != GREY]
    bg = random.choice(bg_choices)
    grid = np.full((grid_h, grid_w), bg, dtype=int)

    target_frames = random.choice(FRAME_COUNT_CHOICES)
    frames = _place_frames(grid_h, grid_w, target_frames)
    if frames is None:
        return None

    scenarios = ["free", "blocked", "pair", "center", "mixed"]

    for r0, c0, r1, c1 in frames:
        grid[r0 : r1 + 1, c0] = GREY
        grid[r0 : r1 + 1, c1] = GREY
        grid[r0, c0 : c1 + 1] = GREY
        grid[r1, c0 : c1 + 1] = GREY

        colors = [c for c in range(10) if c not in (bg, GREY)]
        majority, minority = random.sample(colors, 2)
        interior_h = r1 - r0 - 1
        interior_w = c1 - c0 - 1
        interior = None
        for scenario in random.sample(scenarios, len(scenarios)):
            try:
                interior = _sample_interior(interior_h, interior_w, majority, minority, scenario)
                break
            except AssertionError:
                continue
        if interior is None:
            return None
        grid[r0 + 1 : r1, c0 + 1 : c1] = np.where(interior == 0, bg, interior)

    if len(_find_frames(grid, bg)) != len(frames):
        return None
    try:
        _ = main(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _sample_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
