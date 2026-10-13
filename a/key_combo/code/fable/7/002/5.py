from __future__ import annotations

import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

_ARC_ROOT = Path("/home/lixinyu/snap/new")
if str(_ARC_ROOT) not in sys.path:
    sys.path.append(str(_ARC_ROOT))

from solvers.arc_utils import Color, get_bg, in_bounds, neighbors4


GEN_ATTEMPTS = 400
PALETTE = [
    Color.BLUE,
    Color.GREEN,
    Color.YELLOW,
    Color.PINK,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
]


def _distance_grid(passable: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    dist = np.full(passable.shape, -1, dtype=int)
    q: deque[tuple[int, int]] = deque([seed])
    dist[seed] = 0
    while q:
        r, c = q.popleft()
        for rr, cc in neighbors4(r, c):
            if not in_bounds(passable.shape, rr, cc):
                continue
            if not passable[rr, cc] or dist[rr, cc] != -1:
                continue
            dist[rr, cc] = dist[r, c] + 1
            q.append((rr, cc))
    return dist


def _extract_parts(grid: np.ndarray) -> tuple[tuple[int, int], np.ndarray, list[tuple[int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert get_bg(g) == Color.BLACK, "expected black background"

    seed_coords = np.argwhere(g == Color.RED)
    assert seed_coords.shape == (1, 2), "expected exactly one red seed"
    seed = tuple(map(int, seed_coords[0]))

    blockers = g == Color.GREY
    samples = []
    for r_raw, c_raw in np.argwhere((g != Color.BLACK) & (g != Color.GREY) & (g != Color.RED)):
        r = int(r_raw)
        c = int(c_raw)
        samples.append((r, c, int(g[r, c])))
    assert samples, "expected at least one colored sample pixel"
    return seed, blockers, samples


def _infer_cycle(samples: list[tuple[int, int]], sample_colors: list[int]) -> list[int]:
    assert len(samples) == len(sample_colors) and samples
    pairs = list(zip(samples, sample_colors))
    candidates: list[list[int]] = []
    for k in range(1, len(pairs) + 1):
        mapping: dict[int, int] = {}
        ok = True
        for d, color in pairs:
            residue = d % k
            prev = mapping.get(residue)
            if prev is not None and prev != color:
                ok = False
                break
            mapping[residue] = color
        if not ok or len(mapping) != k:
            continue
        cycle = [mapping[idx] for idx in range(k)]
        if cycle not in candidates:
            candidates.append(cycle)
    assert len(candidates) == 1, "samples must uniquely determine one full cycle"
    return candidates[0]


def _solve(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[int]]:
    g = np.asarray(grid, dtype=int)
    seed, blockers, samples = _extract_parts(g)
    passable = ~blockers
    dist = _distance_grid(passable, seed)

    sample_ds: list[int] = []
    sample_colors: list[int] = []
    for r, c, color in samples:
        d = int(dist[r, c])
        assert d >= 0, "sample pixels must be reachable from the seed"
        sample_ds.append(d)
        sample_colors.append(color)

    cycle = _infer_cycle(sample_ds, sample_colors)
    out = g.copy()
    reachable = dist >= 0
    for r_raw, c_raw in np.argwhere(reachable):
        r = int(r_raw)
        c = int(c_raw)
        out[r, c] = cycle[int(dist[r, c]) % len(cycle)]
    out[blockers] = Color.GREY
    return out, dist, cycle


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct a distance-modulo color cycle from sparse samples and flood the reachable region from the red seed.

    Input structure:
    - The background is black.
    - Exactly one red seed pixel marks the source of the wavefront.
    - Grey cells are blockers, and every other colored pixel is a sample of the final answer on that cell.

    Rule:
    - Compute each passable cell's 4-neighbor shortest-path distance from the red seed while treating grey cells as walls.
    - Use the sample pixels' distances and colors to recover the unique periodic table mapping `distance mod k` to color, including the seed's distance-0 color.
    - Recolor every reachable non-grey cell by that periodic table; keep grey blockers and unreachable cells unchanged.

    Key insight:
    - The samples encode a ripple pattern indexed by true walk distance around obstacles, so the correct phase is revealed only by the obstacle-aware wavefront rather than straight-line geometry.
    """
    out, _dist, _cycle = _solve(grid)
    return out


def _try_place_rect(mask: np.ndarray, r0: int, c0: int, h: int, w: int, *, frame: bool) -> bool:
    H, W = mask.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    view = mask[r0 : r0 + h, c0 : c0 + w]
    if np.any(view):
        return False
    if frame:
        if h < 3 or w < 3:
            return False
        view[0, :] = True
        view[-1, :] = True
        view[:, 0] = True
        view[:, -1] = True
    else:
        view[:, :] = True
    return True


def _paint_main_barrier(blockers: np.ndarray) -> dict[str, int | str]:
    H, W = blockers.shape
    orientation = random.choice(["vertical", "horizontal"])
    thickness = random.choice([1, 1, 2])
    side = random.choice(["left", "right"]) if orientation == "vertical" else random.choice(["top", "bottom"])
    band = random.choice(["upper", "lower"]) if orientation == "vertical" else random.choice(["left", "right"])

    if orientation == "vertical":
        wall_c = random.randint(3, W - thickness - 4)
        gap_len = random.choice([1, 2])
        if band == "upper":
            gap_r = random.randint(H // 2, H - gap_len - 2)
        else:
            gap_r = random.randint(1, max(1, H // 2 - gap_len))
        blockers[:, wall_c : wall_c + thickness] = True
        blockers[gap_r : gap_r + gap_len, wall_c : wall_c + thickness] = False
        return {
            "orientation": orientation,
            "wall": wall_c,
            "thickness": thickness,
            "side": side,
            "band": band,
            "gap": gap_r,
        }

    wall_r = random.randint(3, H - thickness - 4)
    gap_len = random.choice([1, 2])
    if band == "left":
        gap_c = random.randint(W // 2, W - gap_len - 2)
    else:
        gap_c = random.randint(1, max(1, W // 2 - gap_len))
    blockers[wall_r : wall_r + thickness, :] = True
    blockers[wall_r : wall_r + thickness, gap_c : gap_c + gap_len] = False
    return {
        "orientation": orientation,
        "wall": wall_r,
        "thickness": thickness,
        "side": side,
        "band": band,
        "gap": gap_c,
    }


def _add_extra_blockers(blockers: np.ndarray, config: dict[str, int | str]) -> None:
    H, W = blockers.shape
    for _ in range(random.randint(0, 2)):
        for _attempt in range(20):
            h = random.randint(2, 4)
            w = random.randint(2, 5)
            r0 = random.randint(1, H - h - 1)
            c0 = random.randint(1, W - w - 1)
            if _try_place_rect(blockers, r0, c0, h, w, frame=False):
                break

    if random.random() < 0.4:
        for _attempt in range(30):
            h = random.randint(4, 6)
            w = random.randint(4, 6)
            r0 = random.randint(1, H - h - 1)
            c0 = random.randint(1, W - w - 1)
            before = blockers.copy()
            if not _try_place_rect(blockers, r0, c0, h, w, frame=True):
                continue
            opening_side = random.choice(["top", "bottom", "left", "right", "none"])
            if opening_side == "top":
                blockers[r0, c0 + 1 : c0 + w - 1] = False
            elif opening_side == "bottom":
                blockers[r0 + h - 1, c0 + 1 : c0 + w - 1] = False
            elif opening_side == "left":
                blockers[r0 + 1 : r0 + h - 1, c0] = False
            elif opening_side == "right":
                blockers[r0 + 1 : r0 + h - 1, c0 + w - 1] = False
            if np.sum(blockers) <= (H * W) // 3:
                break
            blockers[:, :] = before


def _seed_candidates(blockers: np.ndarray, config: dict[str, int | str]) -> list[tuple[int, int]]:
    H, W = blockers.shape
    orientation = str(config["orientation"])
    side = str(config["side"])
    band = str(config["band"])
    thickness = int(config["thickness"])
    wall = int(config["wall"])

    rows = range(1, H - 1)
    cols = range(1, W - 1)
    candidates: list[tuple[int, int]] = []

    if orientation == "vertical":
        row_range = range(1, H // 3 + 1) if band == "upper" else range((2 * H) // 3, H - 1)
        if side == "left":
            col_range = range(1, wall - 1)
        else:
            col_range = range(wall + thickness + 1, W - 1)
        rows = row_range if list(row_range) else rows
        cols = col_range if list(col_range) else cols
    else:
        col_range = range(1, W // 3 + 1) if band == "left" else range((2 * W) // 3, W - 1)
        if side == "top":
            row_range = range(1, wall - 1)
        else:
            row_range = range(wall + thickness + 1, H - 1)
        rows = row_range if list(row_range) else rows
        cols = col_range if list(col_range) else cols

    for r in rows:
        for c in cols:
            if not blockers[r, c]:
                candidates.append((r, c))
    random.shuffle(candidates)
    return candidates


def _available_ks(dist: np.ndarray, seed: tuple[int, int]) -> list[int]:
    reach = dist >= 0
    dvals = [int(dist[r, c]) for r, c in np.argwhere(reach) if (int(r), int(c)) != seed]
    ks = []
    for k in range(2, 6):
        if not any(d > 0 and d % k == 0 for d in dvals):
            continue
        if all(any(d % k == residue for d in dvals) for residue in range(k)):
            ks.append(k)
    return ks


def _cell_side(config: dict[str, int | str], cell: tuple[int, int]) -> str:
    r, c = cell
    orientation = str(config["orientation"])
    wall = int(config["wall"])
    thickness = int(config["thickness"])
    if orientation == "vertical":
        if c < wall:
            return "near" if str(config["side"]) == "left" else "far"
        if c >= wall + thickness:
            return "near" if str(config["side"]) == "right" else "far"
        return "wall"
    if r < wall:
        return "near" if str(config["side"]) == "top" else "far"
    if r >= wall + thickness:
        return "near" if str(config["side"]) == "bottom" else "far"
    return "wall"


def _pick_spaced(candidates: list[tuple[int, int]], chosen: list[tuple[int, int]], min_gap: int) -> tuple[int, int] | None:
    for cell in candidates:
        if all(abs(cell[0] - rr) + abs(cell[1] - cc) >= min_gap for rr, cc in chosen):
            return cell
    return None


def _choose_samples(
    dist: np.ndarray,
    output: np.ndarray,
    seed: tuple[int, int],
    cycle_len: int,
    config: dict[str, int | str],
) -> list[tuple[int, int]]:
    seed_r, seed_c = seed
    chosen: list[tuple[int, int]] = []
    by_residue: dict[int, list[tuple[int, int]]] = {r: [] for r in range(cycle_len)}
    detour: list[tuple[int, int]] = []

    for r_raw, c_raw in np.argwhere(dist >= 0):
        r = int(r_raw)
        c = int(c_raw)
        if (r, c) == seed:
            continue
        d = int(dist[r, c])
        by_residue[d % cycle_len].append((r, c))
        if d > abs(r - seed_r) + abs(c - seed_c) and _cell_side(config, (r, c)) == "far":
            detour.append((r, c))

    assert detour, "expected at least one detour sample candidate"
    detour.sort(key=lambda rc: (dist[rc], rc[0], rc[1]), reverse=True)
    chosen.append(detour[0])

    for residue in range(cycle_len):
        candidates = by_residue[residue][:]
        random.shuffle(candidates)
        candidates.sort(key=lambda rc: (_cell_side(config, rc) != "far", dist[rc], rc[0], rc[1]))
        if residue == int(dist[chosen[0]]) % cycle_len:
            continue
        picked = None
        for min_gap in (4, 3, 2, 1):
            picked = _pick_spaced(candidates, chosen, min_gap)
            if picked is not None:
                break
        assert picked is not None, "failed to place residue-covering sample"
        chosen.append(picked)

    extras = []
    for cell in detour[1:]:
        if int(dist[cell]) % cycle_len != int(dist[chosen[0]]) % cycle_len:
            continue
        extras.append(cell)
    for residue, cells in by_residue.items():
        for cell in cells:
            if cell not in chosen and int(dist[cell]) >= cycle_len + residue:
                extras.append(cell)
                break
    random.shuffle(extras)
    extra_budget = random.randint(1, 3)
    for cell in extras:
        if extra_budget == 0:
            break
        picked = None
        for min_gap in (4, 3, 2, 1):
            picked = _pick_spaced([cell], chosen, min_gap)
            if picked is not None:
                break
        if picked is None:
            continue
        chosen.append(cell)
        extra_budget -= 1

    chosen.sort()
    assert len({tuple(cell) for cell in chosen}) == len(chosen)
    assert all(output[r, c] != Color.RED for r, c in chosen)
    return chosen


def _build_instance() -> tuple[np.ndarray, np.ndarray]:
    H = random.randint(12, 22)
    W = random.randint(12, 22)
    blockers = np.zeros((H, W), dtype=bool)

    config = _paint_main_barrier(blockers)
    _add_extra_blockers(blockers, config)

    passable = ~blockers
    seed = None
    dist = None
    for candidate in _seed_candidates(blockers, config):
        candidate_dist = _distance_grid(passable, candidate)
        detour_exists = False
        for r_raw, c_raw in np.argwhere(candidate_dist >= 0):
            r = int(r_raw)
            c = int(c_raw)
            if (r, c) == candidate:
                continue
            if candidate_dist[r, c] > abs(r - candidate[0]) + abs(c - candidate[1]) and _cell_side(config, (r, c)) == "far":
                detour_exists = True
                break
        if detour_exists and np.count_nonzero(candidate_dist >= 0) >= (H * W) // 2:
            seed = candidate
            dist = candidate_dist
            break
    assert seed is not None and dist is not None, "failed to place a seed with a useful wavefront"

    ks = _available_ks(dist, seed)
    assert ks, "expected at least one usable cycle length"
    cycle_len = random.choice(ks)
    cycle = random.sample(PALETTE, cycle_len)

    output = np.full((H, W), Color.BLACK, dtype=int)
    output[blockers] = Color.GREY
    for r_raw, c_raw in np.argwhere(dist >= 0):
        r = int(r_raw)
        c = int(c_raw)
        output[r, c] = cycle[int(dist[r, c]) % cycle_len]

    samples = _choose_samples(dist, output, seed, cycle_len, config)
    grid = np.full((H, W), Color.BLACK, dtype=int)
    grid[blockers] = Color.GREY
    grid[seed] = Color.RED
    for r, c in samples:
        grid[r, c] = output[r, c]

    assert np.count_nonzero(grid == Color.RED) == 1
    assert np.any((dist == -1) & (~blockers)) or random.random() < 1.0
    return grid, output


def _validate_generated(grid: np.ndarray, expected: np.ndarray) -> None:
    actual, dist, cycle = _solve(grid)
    assert np.array_equal(actual, expected), "solver output must match constructed target"

    seed, blockers, samples = _extract_parts(grid)
    sample_ds = [int(dist[r, c]) for r, c, _color in samples]
    sample_colors = [color for _r, _c, color in samples]
    inferred_cycle = _infer_cycle(sample_ds, sample_colors)
    assert inferred_cycle == cycle, "samples must recover the same cycle"
    assert len(set(cycle)) == len(cycle), "cycle colors must stay distinct"
    assert any(d > 0 and d % len(cycle) == 0 for d in sample_ds), "samples must reveal the residue-0 color away from the seed"

    sr, sc = seed
    assert any(
        int(dist[r, c]) > abs(r - sr) + abs(c - sc)
        for r, c, _color in samples
    ), "at least one sample must prove obstacle-aware distance"
    assert np.all(expected[blockers] == Color.GREY)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        try:
            grid, expected = _build_instance()
            _validate_generated(grid, expected)
            return grid
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid sampled ripple puzzle")
