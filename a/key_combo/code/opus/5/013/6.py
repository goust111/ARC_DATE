from __future__ import annotations

import hashlib
import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components, get_bg, in_bounds


BG = Color.BLACK
HELPER = Color.RED
BAR = Color.GREEN
SEED = Color.YELLOW
TRACE = Color.PINK

ARM_OFFSETS = (
    (0, -3),
    (0, -2),
    (0, 2),
    (0, 3),
    (-3, 0),
    (-2, 0),
    (2, 0),
    (3, 0),
)
GAP_OFFSETS = ((0, -1), (0, 1), (-1, 0), (1, 0))
HELPER_MAGNITUDES = ((1, 1), (1, 2), (2, 1), (2, 2))
QUADRANTS = ((-1, -1), (-1, 1), (1, -1), (1, 1))

GRID_SIDE_RANGE = (18, 24)
GROUP_COUNT_RANGE = (2, 4)
MIN_CENTER_GAP = 8
MAX_DISTRACTOR_BARS = 4
MAX_GENERATION_ATTEMPTS = 600

_SEEN_OUTPUT_SIGS: set[str] = set()
_GEN_CALL_COUNT = 0


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _center_bar_cells(center: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = center
    return [(r + dr, c + dc) for dr, dc in ARM_OFFSETS]


def _is_implied_plus_center(grid: np.ndarray, r: int, c: int) -> bool:
    if int(grid[r, c]) != BG:
        return False
    return all(int(grid[r + dr, c + dc]) == BAR for dr, dc in ARM_OFFSETS) and all(
        int(grid[r + dr, c + dc]) == BG for dr, dc in GAP_OFFSETS
    )


def _detect_centers(grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    centers = []
    for r in range(3, h - 3):
        for c in range(3, w - 3):
            if _is_implied_plus_center(grid, r, c):
                centers.append((r, c))
    return centers


def _helper_candidates(grid: np.ndarray, center: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = center
    coords = []
    for rr in range(r0 - 2, r0 + 3):
        for cc in range(c0 - 2, c0 + 3):
            if not in_bounds(grid.shape, rr, cc):
                continue
            dr = rr - r0
            dc = cc - c0
            if dr == 0 or dc == 0:
                continue
            if 1 <= abs(dr) <= 2 and 1 <= abs(dc) <= 2 and int(grid[rr, cc]) == HELPER:
                coords.append((rr, cc))
    return coords


def _detect_centers_and_directions(
    grid: np.ndarray,
) -> tuple[list[tuple[int, int]], dict[tuple[int, int], tuple[int, int]], dict[tuple[int, int], tuple[int, int]]]:
    centers = _detect_centers(grid)
    assert centers, "expected at least one implied-plus center"

    helper_dirs: dict[tuple[int, int], tuple[int, int]] = {}
    helper_pos: dict[tuple[int, int], tuple[int, int]] = {}
    used_helpers: set[tuple[int, int]] = set()
    for center in centers:
        candidates = _helper_candidates(grid, center)
        assert len(candidates) == 1, "each center must have exactly one nearby diagonal helper"
        hr, hc = candidates[0]
        helper = (int(hr), int(hc))
        dr = _sign(helper[0] - center[0])
        dc = _sign(helper[1] - center[1])
        assert dr != 0 and dc != 0
        helper_dirs[center] = (dr, dc)
        helper_pos[center] = helper
        used_helpers.add(helper)

    red_pixels = {tuple(map(int, rc)) for rc in np.argwhere(grid == HELPER)}
    assert used_helpers == red_pixels, "helpers must correspond one-to-one with implied plus centers"
    return centers, helper_dirs, helper_pos


def _simulate_growth(
    grid: np.ndarray,
    centers: list[tuple[int, int]],
    directions: dict[tuple[int, int], tuple[int, int]],
    *,
    seed_override: dict[tuple[int, int], tuple[int, int]] | None = None,
    ignore_collisions: bool = False,
) -> tuple[np.ndarray, dict[str, int]]:
    out = np.array(grid, copy=True)
    out[out == HELPER] = BG

    start_map = seed_override or {center: center for center in centers}
    for center in centers:
        sr, sc = start_map[center]
        out[sr, sc] = SEED

    active = [
        {
            "center": center,
            "seed": start_map[center],
            "direction": directions[center],
            "step": 1,
        }
        for center in centers
    ]
    stats = {"placed": 0, "simultaneous_collisions": 0, "occupied_stops": 0, "out_of_bounds_stops": 0}

    while active:
        requests: list[tuple[int, tuple[int, int]]] = []
        next_active: list[dict] = []
        for idx, info in enumerate(active):
            sr, sc = info["seed"]
            dr, dc = info["direction"]
            step = int(info["step"])
            rr = sr + 2 * step * dr
            cc = sc + 2 * step * dc
            if not in_bounds(out.shape, rr, cc):
                stats["out_of_bounds_stops"] += 1
                continue
            if not ignore_collisions and int(out[rr, cc]) != BG:
                stats["occupied_stops"] += 1
                continue
            requests.append((idx, (int(rr), int(cc))))

        if not requests:
            break

        if ignore_collisions:
            for idx, pos in requests:
                info = dict(active[idx])
                color = TRACE if info["step"] % 2 == 1 else SEED
                if int(out[pos]) == BG:
                    out[pos] = color
                    stats["placed"] += 1
                info["step"] += 1
                next_active.append(info)
        else:
            counts: dict[tuple[int, int], int] = {}
            for _idx, pos in requests:
                counts[pos] = counts.get(pos, 0) + 1

            any_growth = False
            for idx, pos in requests:
                if counts[pos] > 1:
                    stats["simultaneous_collisions"] += 1
                    continue
                info = dict(active[idx])
                color = TRACE if info["step"] % 2 == 1 else SEED
                out[pos] = color
                stats["placed"] += 1
                info["step"] += 1
                next_active.append(info)
                any_growth = True

            if not any_growth:
                break

        active = next_active

    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect each missing implied-plus center and grow an alternating diagonal trace from it toward the helper-indicated quadrant.

    Input structure:
    - The grid has black (0) background plus 2 to 4 green (3) implied-plus motifs; each motif is four separate two-pixel bars around one empty center.
    - Every true center is a background pixel with green bar cells at offsets 2 and 3 in all four cardinal directions, while the one-cell gaps next to the center stay background.
    - Each center has exactly one nearby red (2) helper pixel in one diagonal quadrant, at row and column offsets whose magnitudes are 1 or 2.

    Rule:
    - Find every background pixel that is the missing center of an implied plus.
    - Use the sign of the helper's row and column offsets from that center to choose one of the four diagonal growth directions.
    - Recolor the center yellow (4), erase every red helper to background, then grow from each seed in two-cell diagonal jumps: distance 2 becomes pink (6), distance 4 becomes yellow (4), and the colors alternate afterward.
    - All seeds grow synchronously. A trajectory stops if its next target leaves the grid, lands on an already occupied cell, or is contested by another trajectory in the same growth round. Existing pixels keep their current color.

    Key insight:
    - The green bars do not mark the output directly; they only reveal the hidden center, while the helper's diagonal quadrant encodes the direction of a multi-source alternating growth process.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert get_bg(g) == BG, "expected black background"

    centers, directions, _helpers = _detect_centers_and_directions(g)
    out, _stats = _simulate_growth(g, centers, directions)
    return out


def _helper_offset(direction: tuple[int, int], mode: str, rng: np.random.Generator) -> tuple[int, int]:
    dr, dc = direction
    if mode == "exact":
        base = (2, 2)
    elif mode == "offpath":
        base = HELPER_MAGNITUDES[int(rng.integers(0, 3))]
    else:
        base = HELPER_MAGNITUDES[int(rng.integers(0, len(HELPER_MAGNITUDES)))]
    return dr * base[0], dc * base[1]


def _allowed_center(center: tuple[int, int], chosen: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    r, c = center
    if not (3 <= r < shape[0] - 3 and 3 <= c < shape[1] - 3):
        return False
    return all(max(abs(r - rr), abs(c - cc)) >= MIN_CENTER_GAP for rr, cc in chosen)


def _pick_collision_pair(
    rng: np.random.Generator, shape: tuple[int, int], chosen: list[tuple[int, int]]
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None:
    h, w = shape
    dirs = list(QUADRANTS)
    rng.shuffle(dirs)
    for dr, dc in dirs:
        for m in (3, 2):
            delta_r = 4 * m * dr
            delta_c = 4 * m * dc

            if dr > 0:
                r_lo, r_hi = 3, h - 4 - delta_r
            else:
                r_lo, r_hi = 3 - delta_r, h - 4
            if dc > 0:
                c_lo, c_hi = 3, w - 4 - delta_c
            else:
                c_lo, c_hi = 3 - delta_c, w - 4

            if r_lo > r_hi or c_lo > c_hi:
                continue

            row_choices = list(range(r_lo, r_hi + 1))
            col_choices = list(range(c_lo, c_hi + 1))
            rng.shuffle(row_choices)
            rng.shuffle(col_choices)
            for r in row_choices[:20]:
                for c in col_choices[:20]:
                    a = (int(r), int(c))
                    b = (int(r + delta_r), int(c + delta_c))
                    if not _allowed_center(a, chosen, shape):
                        continue
                    if not _allowed_center(b, chosen + [a], shape):
                        continue
                    return a, b, (dr, dc)
    return None


def _pick_random_center(
    rng: np.random.Generator, shape: tuple[int, int], chosen: list[tuple[int, int]]
) -> tuple[int, int] | None:
    h, w = shape
    rows = list(range(3, h - 3))
    cols = list(range(3, w - 3))
    rng.shuffle(rows)
    rng.shuffle(cols)
    for r in rows:
        for c in cols:
            center = (int(r), int(c))
            if _allowed_center(center, chosen, shape):
                return center
    return None


def _build_spec(rng: np.random.Generator, phase: int) -> list[dict] | None:
    h = int(rng.integers(GRID_SIDE_RANGE[0], GRID_SIDE_RANGE[1] + 1))
    w = int(rng.integers(GRID_SIDE_RANGE[0], GRID_SIDE_RANGE[1] + 1))
    shape = (h, w)

    require_collision = phase in (0, 3)
    force_exact_helper = phase in (2, 3)
    offpath_only = phase == 1
    target_groups = 4 if phase == 3 else int(rng.integers(GROUP_COUNT_RANGE[0], GROUP_COUNT_RANGE[1] + 1))
    if require_collision:
        target_groups = max(2, target_groups)

    groups: list[dict] = []
    chosen: list[tuple[int, int]] = []

    if require_collision:
        pair = _pick_collision_pair(rng, shape, chosen)
        if pair is None:
            return None
        a, b, direction = pair
        groups.append({"center": a, "direction": direction})
        groups.append({"center": b, "direction": (-direction[0], -direction[1])})
        chosen.extend([a, b])

    while len(groups) < target_groups:
        center = _pick_random_center(rng, shape, chosen)
        if center is None:
            return None
        chosen.append(center)
        groups.append(
            {
                "center": center,
                "direction": QUADRANTS[int(rng.integers(0, len(QUADRANTS)))],
            }
        )

    rng.shuffle(groups)
    if len({tuple(g["direction"]) for g in groups}) < 2:
        groups[-1]["direction"] = random.choice([d for d in QUADRANTS if d != groups[0]["direction"]])

    exact_idx = int(rng.integers(0, len(groups))) if force_exact_helper else None
    offpath_idx = int(rng.integers(0, len(groups)))
    for idx, group in enumerate(groups):
        if idx == exact_idx:
            mode = "exact"
        elif offpath_only or idx == offpath_idx:
            mode = "offpath"
        else:
            mode = "any"
        offset = _helper_offset(group["direction"], mode, rng)
        group["helper"] = (group["center"][0] + offset[0], group["center"][1] + offset[1])

    return groups


def _path_cells_from_seed(seed: tuple[int, int], direction: tuple[int, int], shape: tuple[int, int]) -> set[tuple[int, int]]:
    dr, dc = direction
    cells = set()
    step = 1
    while True:
        rr = seed[0] + 2 * step * dr
        cc = seed[1] + 2 * step * dc
        if not in_bounds(shape, rr, cc):
            break
        cells.add((int(rr), int(cc)))
        step += 1
    return cells


def _place_distractors(
    grid: np.ndarray, groups: list[dict], rng: np.random.Generator
) -> None:
    forbidden = set()
    for group in groups:
        center = group["center"]
        forbidden.update(_center_bar_cells(center))
        forbidden.add(center)
        forbidden.add(group["helper"])
        forbidden.update(_path_cells_from_seed(center, group["direction"], grid.shape))

    count = int(rng.integers(0, MAX_DISTRACTOR_BARS + 1))
    placed = 0
    for _ in range(200):
        if placed >= count:
            break
        orientation = int(rng.integers(0, 2))
        r = int(rng.integers(0, grid.shape[0] - (orientation == 1)))
        c = int(rng.integers(0, grid.shape[1] - (orientation == 0)))
        cells = [(r, c), (r + orientation, c + (1 - orientation))]
        if any(cell in forbidden for cell in cells):
            continue
        if any(int(grid[rr, cc]) != BG for rr, cc in cells):
            continue
        cell_set = set(cells)
        touches_fg = False
        for rr, cc in cells:
            for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                if (nr, nc) in cell_set:
                    continue
                if in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) != BG:
                    touches_fg = True
                    break
            if touches_fg:
                break
        if touches_fg:
            continue
        for rr, cc in cells:
            grid[rr, cc] = BAR
            forbidden.add((rr, cc))
        placed += 1


def _render_input(groups: list[dict], rng: np.random.Generator) -> np.ndarray:
    h = max(group["center"][0] for group in groups) + 4
    w = max(group["center"][1] for group in groups) + 4
    grid = np.full((h, w), BG, dtype=int)

    for group in groups:
        r, c = group["center"]
        for rr, cc in _center_bar_cells((r, c)):
            grid[rr, cc] = BAR
        hr, hc = group["helper"]
        grid[hr, hc] = HELPER

    _place_distractors(grid, groups, rng)
    return grid


def _spec_maps(groups: list[dict]) -> tuple[list[tuple[int, int]], dict[tuple[int, int], tuple[int, int]], dict[tuple[int, int], tuple[int, int]]]:
    centers = [tuple(group["center"]) for group in groups]
    directions = {tuple(group["center"]): tuple(group["direction"]) for group in groups}
    helpers = {tuple(group["center"]): tuple(group["helper"]) for group in groups}
    return centers, directions, helpers


def _validate_green_objects(grid: np.ndarray) -> bool:
    for comp in find_components(grid, bg=BG, connectivity=4, colors=[BAR]):
        coords = np.asarray(comp["coords"], dtype=int)
        if coords.shape != (2, 2):
            return False
        r0, c0 = map(int, coords[0])
        r1, c1 = map(int, coords[1])
        if not ((r0 == r1 and abs(c0 - c1) == 1) or (c0 == c1 and abs(r0 - r1) == 1)):
            return False
    return True


def _helper_seed_output(
    grid: np.ndarray,
    centers: list[tuple[int, int]],
    directions: dict[tuple[int, int], tuple[int, int]],
    helpers: dict[tuple[int, int], tuple[int, int]],
) -> np.ndarray:
    return _simulate_growth(grid, centers, directions, seed_override=helpers)[0]


def _constant_direction_output(
    grid: np.ndarray, centers: list[tuple[int, int]], direction: tuple[int, int]
) -> np.ndarray:
    directions = {center: direction for center in centers}
    return _simulate_growth(grid, centers, directions)[0]


def _validate_case(grid: np.ndarray, groups: list[dict], phase: int) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if not _validate_green_objects(grid):
        return False

    expected_centers, expected_dirs, helpers = _spec_maps(groups)
    expected_center_set = set(expected_centers)

    try:
        centers, directions, detected_helpers = _detect_centers_and_directions(grid)
        actual, stats = _simulate_growth(grid, centers, directions)
        solved = main(grid)
    except Exception:
        return False

    if set(centers) != expected_center_set:
        return False
    if directions != expected_dirs:
        return False
    if detected_helpers != helpers:
        return False
    if not np.array_equal(actual, solved):
        return False
    if int(np.count_nonzero(actual == SEED)) < len(centers):
        return False
    if stats["placed"] < len(centers):
        return False
    if len({directions[c] for c in centers}) < 2:
        return False

    exact_helper_count = sum(abs(helpers[c][0] - c[0]) == 2 and abs(helpers[c][1] - c[1]) == 2 for c in centers)
    offpath_helper_count = len(centers) - exact_helper_count
    if offpath_helper_count == 0:
        return False
    if phase in (2, 3) and exact_helper_count == 0:
        return False
    if phase == 1 and exact_helper_count != 0:
        return False

    if np.array_equal(_helper_seed_output(grid, centers, directions, helpers), actual):
        return False

    first_direction = directions[centers[0]]
    if np.array_equal(_constant_direction_output(grid, centers, first_direction), actual):
        return False

    if phase in (0, 3):
        if stats["simultaneous_collisions"] == 0:
            return False
        no_collision = _simulate_growth(grid, centers, directions, ignore_collisions=True)[0]
        if np.array_equal(no_collision, actual):
            return False
    else:
        if stats["simultaneous_collisions"] != 0:
            return False

    return True


def _gen_attempt(phase: int) -> np.ndarray | None:
    rng = np.random.default_rng()
    groups = _build_spec(rng, phase)
    if groups is None:
        return None

    grid = _render_input(groups, rng)
    if not _validate_case(grid, groups, phase):
        return None

    out = main(grid)
    sig = hashlib.md5(out.tobytes() + str(out.shape).encode()).hexdigest()
    if sig in _SEEN_OUTPUT_SIGS:
        return None
    _SEEN_OUTPUT_SIGS.add(sig)
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    phase = _GEN_CALL_COUNT % 4
    _GEN_CALL_COUNT += 1
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(phase)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid implied-plus diagonal-growth input")
