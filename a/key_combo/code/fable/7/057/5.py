from __future__ import annotations

import importlib.util
import random
from itertools import combinations
from pathlib import Path

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_UTILS_SPEC = importlib.util.spec_from_file_location("arc_utils_shared_057_5_rework", _ARC_UTILS_PATH)
if _ARC_UTILS_SPEC is None or _ARC_UTILS_SPEC.loader is None:
    raise ImportError(f"failed to load ARC utilities from {_ARC_UTILS_PATH}")
_ARC_UTILS = importlib.util.module_from_spec(_ARC_UTILS_SPEC)
_ARC_UTILS_SPEC.loader.exec_module(_ARC_UTILS)

Color = _ARC_UTILS.Color
get_bg = _ARC_UTILS.get_bg


BG = Color.BLACK
RESIDUE = Color.TEAL
SOURCE = Color.YELLOW
DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))

GRID_SIDE_RANGE = (15, 26)
SOURCE_COUNT_CHOICES = (1, 2, 2, 3)
MIN_SOURCE_MARGIN = 2
MIN_SOURCE_CHEB_GAP = 6
MAX_GENERATION_ATTEMPTS = 160
MAX_EXTRA_NOISE = 2
NEAR_MISS_PROB = 0.75


def _ray_cells(shape: tuple[int, int], r: int, c: int, dr: int, dc: int) -> list[tuple[int, int]]:
    h, w = shape
    out: list[tuple[int, int]] = []
    rr, cc = r + dr, c + dc
    while 0 <= rr < h and 0 <= cc < w:
        out.append((rr, cc))
        rr += dr
        cc += dc
    return out


def _beam_mask(shape: tuple[int, int], sources: tuple[tuple[int, int], ...]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in sources:
        for dr, dc in DIAGONALS:
            for rr, cc in _ray_cells(shape, r, c, dr, dc):
                mask[rr, cc] = True
    return mask


def _directional_counts(teal_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w = teal_mask.shape
    teal = teal_mask.astype(np.int16, copy=False)
    nw = np.zeros((h, w), dtype=np.int16)
    ne = np.zeros((h, w), dtype=np.int16)
    sw = np.zeros((h, w), dtype=np.int16)
    se = np.zeros((h, w), dtype=np.int16)

    for r in range(1, h):
        for c in range(1, w):
            nw[r, c] = nw[r - 1, c - 1] + teal[r - 1, c - 1]
        for c in range(w - 2, -1, -1):
            ne[r, c] = ne[r - 1, c + 1] + teal[r - 1, c + 1]

    for r in range(h - 2, -1, -1):
        for c in range(1, w):
            sw[r, c] = sw[r + 1, c - 1] + teal[r + 1, c - 1]
        for c in range(w - 2, -1, -1):
            se[r, c] = se[r + 1, c + 1] + teal[r + 1, c + 1]

    return nw, ne, sw, se


def _find_sources(grid: np.ndarray) -> tuple[tuple[int, int], ...]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    teal_mask = g == RESIDUE
    nw, ne, sw, se = _directional_counts(teal_mask)
    source_mask = (g == bg) & (nw >= 2) & (ne >= 2) & (sw >= 2) & (se >= 2)
    return tuple((int(r), int(c)) for r, c in np.argwhere(source_mask))


def _render_output(shape: tuple[int, int], sources: tuple[tuple[int, int], ...], bg: int = BG) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    out[_beam_mask(shape, sources)] = RESIDUE
    for r, c in sources:
        out[r, c] = SOURCE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the hidden emitters from diagonal residue evidence and redraw their full X-beams.

    Input structure:
    - The grid has a black background and scattered cyan pixels.
    - Most cyan pixels are residue dots lying on diagonal beams from one or more hidden sources.
    - The true source cells themselves are still black, and some extra cyan dots may be unrelated noise.

    Rule:
    - Find every black cell whose four diagonal directions each contain at least two cyan pixels aligned with that ray.
    - Treat every such cell as a hidden source, paint it yellow, and extend cyan diagonals from it to the grid edges.
    - Erase every cyan input dot that does not lie on any reconstructed beam.

    Key insight:
    - The deciding object never appears in the input: only an empty cell with four-direction double witness qualifies as a source, and the fake residue dots are exposed only after those sources are inferred.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    fg_colors = set(map(int, np.unique(g))) - {bg}
    assert fg_colors <= {RESIDUE}, "input should contain only cyan residue pixels on the background"

    sources = _find_sources(g)
    assert sources, "expected at least one hidden emitter"
    return _render_output(g.shape, sources, bg=bg)


def _sample_sources(h: int, w: int, count: int) -> tuple[tuple[int, int], ...] | None:
    candidates = [
        (r, c)
        for r in range(MIN_SOURCE_MARGIN, h - MIN_SOURCE_MARGIN)
        for c in range(MIN_SOURCE_MARGIN, w - MIN_SOURCE_MARGIN)
    ]
    random.shuffle(candidates)

    chosen: list[tuple[int, int]] = []
    for r, c in candidates:
        if any(max(abs(r - rr), abs(c - cc)) < MIN_SOURCE_CHEB_GAP for rr, cc in chosen):
            continue
        if any((r - c) == (rr - cc) or (r + c) == (rr + cc) for rr, cc in chosen):
            continue
        chosen.append((r, c))
        if len(chosen) == count:
            return tuple(sorted(chosen))
    return None


def _choose_residue_cells(ray: list[tuple[int, int]]) -> list[tuple[int, int]]:
    assert len(ray) >= 2
    if len(ray) == 2:
        return list(ray)

    near_hi = min(3, len(ray) - 1)
    first_idx = random.randint(0, near_hi - 1)
    second_idx = random.randint(first_idx + 1, len(ray) - 1)
    indices = {first_idx, second_idx}

    if len(ray) >= 4 and random.random() < 0.45:
        indices.add(random.randint(0, len(ray) - 1))
    if len(ray) >= 5 and random.random() < 0.20:
        indices.add(random.randint(0, len(ray) - 1))

    return [ray[idx] for idx in sorted(indices)]


def _build_true_residues(shape: tuple[int, int], sources: tuple[tuple[int, int], ...]) -> set[tuple[int, int]]:
    residues: set[tuple[int, int]] = set()
    for r, c in sources:
        for dr, dc in DIAGONALS:
            ray = _ray_cells(shape, r, c, dr, dc)
            residues.update(_choose_residue_cells(ray))
    return residues


def _direction_satisfied(
    nw: np.ndarray,
    ne: np.ndarray,
    sw: np.ndarray,
    se: np.ndarray,
    r: int,
    c: int,
) -> tuple[bool, bool, bool, bool]:
    return (nw[r, c] >= 2, ne[r, c] >= 2, sw[r, c] >= 2, se[r, c] >= 2)


def _try_add_near_miss(
    grid: np.ndarray,
    beam_mask: np.ndarray,
    true_sources: tuple[tuple[int, int], ...],
) -> bool:
    h, w = grid.shape
    centers = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if int(grid[r, c]) == BG and not beam_mask[r, c]
    ]
    random.shuffle(centers)

    for r, c in centers[:120]:
        available_by_dir: list[list[tuple[int, int]]] = []
        for dr, dc in DIAGONALS:
            ray = [
                cell
                for cell in _ray_cells(grid.shape, r, c, dr, dc)
                if not beam_mask[cell] and int(grid[cell]) == BG
            ]
            available_by_dir.append(ray)

        usable = [idx for idx, ray in enumerate(available_by_dir) if len(ray) >= 2]
        if len(usable) < 3:
            continue

        order = list(combinations(usable, 3))
        random.shuffle(order)
        for chosen_dirs in order:
            additions: list[tuple[int, int]] = []
            for idx in chosen_dirs:
                ray = available_by_dir[idx]
                picks = sorted(random.sample(range(len(ray)), 2))
                additions.extend(ray[picks[0] : picks[0] + 1])
                additions.extend(ray[picks[1] : picks[1] + 1])

            for rr, cc in additions:
                grid[rr, cc] = RESIDUE

            inferred = _find_sources(grid)
            nw, ne, sw, se = _directional_counts(grid == RESIDUE)
            satisfied = _direction_satisfied(nw, ne, sw, se, r, c)
            if inferred == true_sources and sum(satisfied) == 3:
                return True

            for rr, cc in additions:
                grid[rr, cc] = BG

    return False


def _add_extra_noise(grid: np.ndarray, beam_mask: np.ndarray) -> None:
    candidates = [
        (r, c)
        for r in range(grid.shape[0])
        for c in range(grid.shape[1])
        if int(grid[r, c]) == BG and not beam_mask[r, c]
    ]
    if not candidates:
        return
    random.shuffle(candidates)
    extra = random.randint(0, min(MAX_EXTRA_NOISE, len(candidates)))
    for r, c in candidates[:extra]:
        grid[r, c] = RESIDUE


def _validate_instance(
    grid: np.ndarray,
    sources: tuple[tuple[int, int], ...],
    beam_mask: np.ndarray,
    require_near_miss: bool,
) -> bool:
    inferred = _find_sources(grid)
    if inferred != sources:
        return False

    expected = _render_output(grid.shape, sources)
    if not np.array_equal(main(grid), expected):
        return False

    residues = grid == RESIDUE
    if not residues.any():
        return False
    if not np.any(residues & ~beam_mask):
        return False
    if not np.any(beam_mask & ~residues):
        return False

    nw, ne, sw, se = _directional_counts(residues)
    for r, c in sources:
        if int(grid[r, c]) != BG:
            return False
        if not all(_direction_satisfied(nw, ne, sw, se, r, c)):
            return False

    if require_near_miss:
        near_miss_found = False
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if int(grid[r, c]) != BG or (r, c) in sources:
                    continue
                if sum(_direction_satisfied(nw, ne, sw, se, r, c)) == 3:
                    near_miss_found = True
                    break
            if near_miss_found:
                break
        if not near_miss_found:
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)
    source_count = random.choice(SOURCE_COUNT_CHOICES)
    sources = _sample_sources(h, w, source_count)
    if sources is None:
        return None

    beam_mask = _beam_mask((h, w), sources)
    grid = np.full((h, w), BG, dtype=int)

    for r, c in _build_true_residues((h, w), sources):
        grid[r, c] = RESIDUE

    require_near_miss = random.random() < NEAR_MISS_PROB
    if require_near_miss and not _try_add_near_miss(grid, beam_mask, sources):
        return None

    _add_extra_noise(grid, beam_mask)

    if not _validate_instance(grid, sources, beam_mask, require_near_miss):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hidden-emitter instance")
