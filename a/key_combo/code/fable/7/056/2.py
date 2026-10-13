from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import find_components, get_bg


GRID_SIDE_RANGE = (18, 28)
PERIOD_HEIGHT_RANGE = (2, 4)
PERIOD_WIDTH_RANGE = (2, 5)
MIN_CAVITY_SIDE = 4
MAX_GEN_ATTEMPTS = 80
PALETTE = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def _is_primitive_bitmap(bitmap: np.ndarray) -> bool:
    pr, pc = bitmap.shape
    for dr in _divisors(pr):
        for dc in _divisors(pc):
            if dr == pr and dc == pc:
                continue
            if np.array_equal(bitmap, np.tile(bitmap[:dr, :dc], (pr // dr, pc // dc))):
                return False
    return True


def _phase_bitmap(seed_coords: np.ndarray, pr: int, pc: int) -> np.ndarray:
    bitmap = np.zeros((pr, pc), dtype=bool)
    if seed_coords.size:
        bitmap[seed_coords[:, 0] % pr, seed_coords[:, 1] % pc] = True
    return bitmap


def _render_windows_mask(
    shape: tuple[int, int],
    windows: list[tuple[int, int, int, int]],
    bitmap: np.ndarray,
) -> np.ndarray:
    pr, pc = bitmap.shape
    out = np.zeros(shape, dtype=bool)
    for r0, c0, r1, c1 in windows:
        rr = np.arange(r0, r1 + 1)[:, None] % pr
        cc = np.arange(c0, c1 + 1)[None, :] % pc
        out[r0 : r1 + 1, c0 : c1 + 1] = bitmap[rr, cc]
    return out


def _enumerate_best_outputs(
    shape: tuple[int, int],
    windows: list[tuple[int, int, int, int]],
    seed_coords: np.ndarray,
) -> list[np.ndarray]:
    assert len(windows) >= 1
    assert seed_coords.size > 0

    max_h = max(r1 - r0 + 1 for r0, _c0, r1, _c1 in windows)
    max_w = max(c1 - c0 + 1 for _r0, c0, _r1, c1 in windows)
    candidates: list[tuple[tuple[int, int, int, int], np.ndarray]] = []

    for pr in range(1, max_h + 1):
        for pc in range(1, max_w + 1):
            bitmap = _phase_bitmap(seed_coords, pr, pc)
            on_count = int(bitmap.sum())
            if on_count == 0 or on_count == pr * pc:
                continue
            if not _is_primitive_bitmap(bitmap):
                continue
            score = (pr * pc, on_count, pr, pc)
            candidates.append((score, bitmap))

    assert candidates, "expected at least one primitive candidate period"
    best_score = min(score for score, _bitmap in candidates)
    outputs: dict[bytes, np.ndarray] = {}
    for score, bitmap in candidates:
        if score != best_score:
            continue
        rendered = _render_windows_mask(shape, windows, bitmap)
        outputs.setdefault(rendered.tobytes(), rendered)
    return list(outputs.values())


def _fill_cavities(
    grid: np.ndarray,
    cavities: list[tuple[int, int, int, int]],
    empty: int,
    seed: int,
    bitmap: np.ndarray,
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    pr, pc = bitmap.shape
    for r0, c0, r1, c1 in cavities:
        rr = np.arange(r0, r1 + 1)[:, None] % pr
        cc = np.arange(c0, c1 + 1)[None, :] % pc
        out[r0 : r1 + 1, c0 : c1 + 1] = np.where(bitmap[rr, cc], seed, empty)
    return out


def _parse_scene(grid: np.ndarray) -> tuple[int, int, int, list[tuple[int, int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    wall = int(get_bg(g))
    non_wall = sorted(set(map(int, g.ravel())) - {wall})
    assert len(non_wall) == 2, "expected exactly wall, floor, and seed colors"

    counts = {color: int(np.count_nonzero(g == color)) for color in non_wall}
    empty = max(non_wall, key=lambda color: counts[color])
    seed = min(non_wall, key=lambda color: counts[color])
    assert counts[empty] > counts[seed], "cavity floor must be the unique most common non-wall color"

    cavities: list[tuple[int, int, int, int]] = []
    for comp in find_components(g, bg=wall, connectivity=4, monochromatic=False, return_format="dict"):
        r0, c0, r1, c1 = comp["bbox"]
        region = g[r0 : r1 + 1, c0 : c1 + 1]
        assert set(map(int, region.ravel())) <= {empty, seed}
        mask = np.zeros(g.shape, dtype=bool)
        coords = comp["coords"]
        mask[coords[:, 0], coords[:, 1]] = True
        assert mask[r0 : r1 + 1, c0 : c1 + 1].all(), "every cavity must be a full rectangle"
        assert np.any(region == empty), "each cavity must contain visible floor cells"
        assert np.any(region == seed), "each cavity must contain visible seed cells"
        cavities.append((int(r0), int(c0), int(r1), int(c1)))

    cavities.sort()
    assert cavities, "expected at least one cavity"
    return wall, int(empty), int(seed), cavities


def _infer_bitmap(
    shape: tuple[int, int],
    cavities: list[tuple[int, int, int, int]],
    seed_coords: np.ndarray,
) -> np.ndarray:
    outputs = _enumerate_best_outputs(shape, cavities, seed_coords)
    assert len(outputs) == 1, "global best periodic fill must be unique"
    rendered = outputs[0]

    max_h = max(r1 - r0 + 1 for r0, _c0, r1, _c1 in cavities)
    max_w = max(c1 - c0 + 1 for _r0, c0, _r1, c1 in cavities)
    candidates: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    for pr in range(1, max_h + 1):
        for pc in range(1, max_w + 1):
            bitmap = _phase_bitmap(seed_coords, pr, pc)
            on_count = int(bitmap.sum())
            if on_count == 0 or on_count == pr * pc:
                continue
            if not _is_primitive_bitmap(bitmap):
                continue
            score = (pr * pc, on_count, pr, pc)
            if np.array_equal(_render_windows_mask(shape, cavities, bitmap), rendered):
                candidates.append((score, bitmap))

    best_score = min(score for score, _bitmap in candidates)
    best = [bitmap for score, bitmap in candidates if score == best_score]
    assert len(best) == 1
    return best[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill every rectangular cavity from one globally phased periodic seed mask shared across all windows.

    Input structure:
    - A dominant wall color forms solid walls around several rectangular cavities.
    - Each cavity contains exactly two non-wall colors: a floor color and scattered seed pixels.
    - The seed pixels are incomplete samples of one shared two-color periodic pattern indexed by absolute grid coordinates.

    Rule:
    - Treat all cavities as windows onto the same global periodic bitmap, using the full-grid coordinate system for phase.
    - Infer the unique smallest primitive period whose repeated seed phases simultaneously explain every visible seed pixel.
    - Fill every cavity completely with that bitmap, using the seed color on on-phases and the floor color elsewhere, while leaving the walls unchanged.

    Key insight:
    - A cavity cannot be tiled independently: complementary fragments across different windows must be combined, and their alignment is fixed by absolute coordinates rather than cavity-local origin.
    """
    g = np.asarray(grid, dtype=int)
    _wall, empty, seed, cavities = _parse_scene(g)
    seed_coords = np.argwhere(g == seed)
    bitmap = _infer_bitmap(g.shape, cavities, seed_coords)
    return _fill_cavities(g, cavities, empty, seed, bitmap)


def _pattern_library(pr: int, pc: int) -> list[np.ndarray]:
    area = pr * pc
    patterns: dict[bytes, np.ndarray] = {}
    rr = np.arange(pr)[:, None]
    cc = np.arange(pc)[None, :]

    formulas = [
        (1, 1, 0, 2, 1),
        (1, 2, 0, 3, 1),
        (2, 1, 1, 3, 1),
        (1, 3, 1, 4, 1),
        (2, 3, 0, 5, 2),
        (3, 1, 2, 5, 2),
        (1, 4, 0, 5, 2),
    ]
    for a, b, offset, mod, width in formulas:
        bitmap = ((a * rr + b * cc + offset) % mod) < width
        on_count = int(bitmap.sum())
        if 3 <= on_count <= area - 2 and _is_primitive_bitmap(bitmap):
            patterns.setdefault(bitmap.tobytes(), bitmap.copy())

    cornerish = np.zeros((pr, pc), dtype=bool)
    cornerish[0, 0] = True
    cornerish[pr - 1, pc - 1] = True
    cornerish[0, pc - 1] = True
    if area >= 8 and _is_primitive_bitmap(cornerish):
        patterns.setdefault(cornerish.tobytes(), cornerish.copy())

    skew = np.zeros((pr, pc), dtype=bool)
    skew[0, 0] = True
    skew[1 % pr, 1 % pc] = True
    skew[pr - 1, 1 % pc] = True
    skew[(pr - 1) // 2, pc - 1] = True
    if 3 <= int(skew.sum()) <= area - 2 and _is_primitive_bitmap(skew):
        patterns.setdefault(skew.tobytes(), skew.copy())

    return list(patterns.values())


def _slot_boxes(rows: int, cols: int) -> list[tuple[int, int, int, int]]:
    mid_r = rows // 2
    mid_c = cols // 2
    return [
        (1, 1, mid_r - 2, mid_c - 2),
        (1, mid_c + 1, mid_r - 2, cols - 2),
        (mid_r + 1, 1, rows - 2, mid_c - 2),
        (mid_r + 1, mid_c + 1, rows - 2, cols - 2),
    ]


def _sample_cavities(
    rng: np.random.Generator,
    rows: int,
    cols: int,
    count: int,
    pr: int,
    pc: int,
) -> list[tuple[int, int, int, int]]:
    slots = _slot_boxes(rows, cols)
    chosen = rng.choice(len(slots), size=count, replace=False)
    cavities = []
    for idx in chosen:
        sr0, sc0, sr1, sc1 = slots[int(idx)]
        slot_h = sr1 - sr0 + 1
        slot_w = sc1 - sc0 + 1
        h_lo = max(MIN_CAVITY_SIDE, pr + 1)
        w_lo = max(MIN_CAVITY_SIDE, pc + 1)
        h = int(rng.integers(h_lo, slot_h + 1))
        w = int(rng.integers(w_lo, slot_w + 1))
        r0 = int(rng.integers(sr0, sr1 - h + 2))
        c0 = int(rng.integers(sc0, sc1 - w + 2))
        cavities.append((r0, c0, r0 + h - 1, c0 + w - 1))
    cavities.sort()
    return cavities


def _assign_visible_sets(
    rng: np.random.Generator,
    full_phases: list[tuple[int, int]],
    cavity_count: int,
) -> list[set[tuple[int, int]]]:
    visible = [set() for _ in range(cavity_count)]
    order = full_phases.copy()
    rng.shuffle(order)
    for idx, phase in enumerate(order):
        visible[idx % cavity_count].add(phase)

    for idx in range(cavity_count):
        extras = [phase for phase in order if phase not in visible[idx]]
        rng.shuffle(extras)
        limit = max(0, len(full_phases) - 1 - len(visible[idx]))
        extra_count = int(rng.integers(0, limit + 1))
        for phase in extras[:extra_count]:
            visible[idx].add(phase)
        if len(visible[idx]) == len(full_phases):
            visible[idx].remove(order[idx % len(order)])
    return visible


def _build_input(
    shape: tuple[int, int],
    wall: int,
    empty: int,
    seed: int,
    cavities: list[tuple[int, int, int, int]],
    bitmap: np.ndarray,
    visible_sets: list[set[tuple[int, int]]],
) -> np.ndarray:
    pr, pc = bitmap.shape
    grid = np.full(shape, wall, dtype=int)
    for (r0, c0, r1, c1), visible in zip(cavities, visible_sets):
        grid[r0 : r1 + 1, c0 : c1 + 1] = empty
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                phase = (r % pr, c % pc)
                if bitmap[phase] and phase in visible:
                    grid[r, c] = seed
    return grid


def _build_output(
    shape: tuple[int, int],
    wall: int,
    empty: int,
    seed: int,
    cavities: list[tuple[int, int, int, int]],
    bitmap: np.ndarray,
) -> np.ndarray:
    return _fill_cavities(np.full(shape, wall, dtype=int), cavities, empty, seed, bitmap)


def _local_cavity_is_sufficient(
    region: np.ndarray,
    empty: int,
    seed: int,
    expected_region: np.ndarray,
) -> bool:
    seed_coords = np.argwhere(region == seed)
    local_outputs = _enumerate_best_outputs(region.shape, [(0, 0, region.shape[0] - 1, region.shape[1] - 1)], seed_coords)
    return len(local_outputs) == 1 and np.array_equal(local_outputs[0], expected_region == seed)


def _semantic_checks(
    inp: np.ndarray,
    out: np.ndarray,
    empty: int,
    seed: int,
    cavities: list[tuple[int, int, int, int]],
) -> bool:
    try:
        solved = main(inp)
    except Exception:
        return False
    if not np.array_equal(solved, out):
        return False

    for r0, c0, r1, c1 in cavities:
        region_in = inp[r0 : r1 + 1, c0 : c1 + 1]
        region_out = out[r0 : r1 + 1, c0 : c1 + 1]
        if _local_cavity_is_sufficient(region_in, empty, seed, region_out):
            return False
    return True


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    rows = int(rng.integers(GRID_SIDE_RANGE[0], GRID_SIDE_RANGE[1] + 1))
    cols = int(rng.integers(GRID_SIDE_RANGE[0], GRID_SIDE_RANGE[1] + 1))
    pr = int(rng.integers(PERIOD_HEIGHT_RANGE[0], PERIOD_HEIGHT_RANGE[1] + 1))
    pc = int(rng.integers(PERIOD_WIDTH_RANGE[0], PERIOD_WIDTH_RANGE[1] + 1))

    patterns = _pattern_library(pr, pc)
    if not patterns:
        return None
    bitmap = patterns[int(rng.integers(0, len(patterns)))]
    full_phases = [(r, c) for r in range(pr) for c in range(pc) if bitmap[r, c]]
    if len(full_phases) < 3:
        return None

    cavity_count = 3 if len(full_phases) == 3 else int(rng.integers(3, 5))
    cavities = _sample_cavities(rng, rows, cols, cavity_count, pr, pc)
    wall, empty, seed = map(int, rng.choice(PALETTE, size=3, replace=False))
    visible_sets = _assign_visible_sets(rng, full_phases, cavity_count)

    inp = _build_input((rows, cols), wall, empty, seed, cavities, bitmap, visible_sets)
    out = _build_output((rows, cols), wall, empty, seed, cavities, bitmap)

    if np.array_equal(inp, out):
        return None
    if not _semantic_checks(inp, out, empty, seed, cavities):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid global-window tiling task")
