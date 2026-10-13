from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


BG = 0
MOVING = 1
SEED_COLORS = (2, 3, 4)
BAR = 5

HEIGHT_RANGE = (10, 15)
WIDTH_CAP = 30
TOTAL_MOVERS_RANGE = (3, 5)
SEED_COUNT_WEIGHTS = (3, 4, 2)  # bias toward 1-2 seeds so unfilled cavities are common
GEN_RETRIES = 400
MIN_SECTOR_TOP_WIDTH = 3


def _bar_row(grid: np.ndarray) -> int:
    bars = [r for r in range(grid.shape[0]) if np.all(grid[r] == BAR)]
    assert len(bars) == 1, "expected exactly one full-width trigger bar"
    return int(bars[0])


def _seed_positions(grid: np.ndarray) -> list[tuple[int, int, int]]:
    seeds = []
    for c in range(grid.shape[1]):
        value = int(grid[0, c])
        if value in SEED_COLORS:
            seeds.append((0, c, value))
    return seeds


def _shift_and_splash(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    out = np.asarray(grid, dtype=int).copy()
    h, w = out.shape
    bar_r = _bar_row(out)

    movers = [tuple(map(int, rc)) for rc in np.argwhere(out == MOVING)]
    out[out == MOVING] = BG
    splash = np.zeros((h, w), dtype=bool)

    moved: list[tuple[int, int]] = []
    for r, c in movers:
        nr = r + 1
        assert nr < h, "moving pixels must have room to shift down"
        moved.append((nr, c))
        out[nr, c] = MOVING

    for r, c in moved:
        if r + 1 < h and int(out[r + 1, c]) == BAR:
            for dc in (-1, 1):
                rr, cc = r, c
                while 0 <= rr < h and 0 <= cc < w:
                    splash[rr, cc] = True
                    out[rr, cc] = MOVING
                    rr -= 1
                    cc += dc

    return out, splash


def _seed_component_masks(grid: np.ndarray) -> dict[int, np.ndarray]:
    fg_mask = grid != BG
    labels = np.zeros(grid.shape, dtype=int)
    for idx, comp in enumerate(components_mask(fg_mask, connectivity=8), start=1):
        labels[comp[:, 0], comp[:, 1]] = idx

    masks: dict[int, np.ndarray] = {}
    for r, c, color in _seed_positions(grid):
        comp_id = int(labels[r, c])
        assert comp_id > 0, "seed must belong to a foreground component"
        masks[color] = labels == comp_id
    return masks


def _cavity_components(grid: np.ndarray, bar_r: int) -> list[np.ndarray]:
    cavity_mask = grid == BG
    cavity_mask[0, :] = False
    cavity_mask[-1, :] = False
    cavity_mask[:, 0] = False
    cavity_mask[:, -1] = False
    cavity_mask[bar_r:, :] = False
    return components_mask(cavity_mask, connectivity=4)


def _touches_component(cavity: np.ndarray, comp_mask: np.ndarray) -> bool:
    h, w = comp_mask.shape
    for r_raw, c_raw in cavity:
        r = int(r_raw)
        c = int(c_raw)
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and comp_mask[nr, nc]:
                return True
    return False


def _fill_cavities(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    bar_r = _bar_row(out)
    seed_masks = _seed_component_masks(out)

    for cavity in _cavity_components(out, bar_r):
        touching_colors = [
            color for color, comp_mask in seed_masks.items() if _touches_component(cavity, comp_mask)
        ]
        if len(touching_colors) == 1:
            color = touching_colors[0]
            out[cavity[:, 0], cavity[:, 1]] = color

    return out


def _oracle(grid: np.ndarray) -> np.ndarray:
    shifted, _splash = _shift_and_splash(grid)
    return _fill_cavities(shifted)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shift blue pixels downward, let only the bar-adjacent one splash upward diagonals, then flood-fill the resulting fenced black cavities from top-edge seeds.

    Input structure:
    - Black background with one full-width grey trigger bar near the bottom.
    - Three to five isolated blue moving pixels lie above the bar; after shifting, only the ones directly above the bar can trigger a splash.
    - One to three unique-color seed pixels (red/green/yellow) sit on the top edge.

    Rule:
    - Erase every blue moving pixel from its original cell and redraw it one row lower in the same column.
    - Whenever a shifted blue pixel lands immediately above the grey bar, draw blue diagonals from that landing cell toward the upper-left and upper-right until they leave the grid.
    - Treat the splash rays, the grey bar, and the grid boundary as fences; find every 4-connected black cavity above the bar, and fill a cavity with a seed color exactly when that cavity is 4-adjacent to the 8-connected foreground component containing that seed.
    - If multiple seed components touch the same cavity, leave that cavity unchanged.

    Key insight:
    - The moved blue pixel is only the trigger: solving the task requires first constructing the diagonal fence geometry, then using seed-to-cavity adjacency to decide which enclosed black regions get colored.
    """
    shifted, _splash = _shift_and_splash(grid)
    return _fill_cavities(shifted)


def _sector_candidates(output_after_shift: np.ndarray, cavities: list[np.ndarray]) -> list[list[int]]:
    h, w = output_after_shift.shape
    comp_of = np.full((h, w), -1, dtype=int)
    for idx, cavity in enumerate(cavities):
        comp_of[cavity[:, 0], cavity[:, 1]] = idx

    candidates: list[list[int]] = []
    for idx in range(len(cavities)):
        cols = []
        for c in range(1, w - 1):
            if comp_of[1, c] != idx:
                continue
            if output_after_shift[1, c - 1] != BG or output_after_shift[1, c] != BG or output_after_shift[1, c + 1] != BG:
                continue
            cols.append(c)
        candidates.append(cols)
    return candidates


def _independent_geometry_checks(inp: np.ndarray, out: np.ndarray) -> bool:
    bar_r = _bar_row(inp)
    movers = [tuple(map(int, rc)) for rc in np.argwhere(inp == MOVING)]
    if not (3 <= len(movers) <= 5):
        return False

    trigger_sources = [(r, c) for r, c in movers if r + 2 == bar_r]
    non_trigger_sources = [(r, c) for r, c in movers if r + 2 != bar_r]
    if len(trigger_sources) != 1 or len(non_trigger_sources) < 2:
        return False

    seeds = _seed_positions(inp)
    if not (1 <= len(seeds) <= 3):
        return False
    if len({color for _r, _c, color in seeds}) != len(seeds):
        return False
    if any(abs(c1 - c2) <= 1 for i, (_r1, c1, _v1) in enumerate(seeds) for _r2, c2, _v2 in seeds[i + 1 :]):
        return False

    shifted, _splash = _shift_and_splash(inp)
    cavities = _cavity_components(shifted, bar_r)
    if len(cavities) != 3:
        return False

    seed_masks = _seed_component_masks(shifted)
    if any(int(mask.sum()) != 1 for mask in seed_masks.values()):
        return False

    touches: list[list[int]] = []
    for cavity in cavities:
        colors = [color for color, mask in seed_masks.items() if _touches_component(cavity, mask)]
        if len(colors) > 1:
            return False
        touches.append(colors)

    if len([colors for colors in touches if colors]) != len(seeds):
        return False

    expected = shifted.copy()
    for cavity, colors in zip(cavities, touches, strict=False):
        if len(colors) == 1:
            expected[cavity[:, 0], cavity[:, 1]] = colors[0]

    return np.array_equal(expected, out)


def _build_attempt() -> np.ndarray | None:
    h = random.randint(*HEIGHT_RANGE)
    apex_r = h - 4
    min_w = max(18, 2 * apex_r + 9)
    if min_w > WIDTH_CAP:
        return None
    w = random.randint(min_w, WIDTH_CAP)
    bar_r = h - 3

    left_hit_min = MIN_SECTOR_TOP_WIDTH + 1
    left_hit_max = w - MIN_SECTOR_TOP_WIDTH - 2 * apex_r - 2
    if left_hit_min > left_hit_max:
        return None
    left_hit = random.randint(left_hit_min, left_hit_max)
    trigger_col = left_hit + apex_r

    inp = np.full((h, w), BG, dtype=int)
    inp[bar_r, :] = BAR
    inp[bar_r - 2, trigger_col] = MOVING

    shifted_preview, _splash_preview = _shift_and_splash(inp)
    forbidden = shifted_preview == MOVING
    forbidden[0, :] = True
    forbidden[bar_r - 1, :] = True

    total_movers = random.randint(*TOTAL_MOVERS_RANGE)
    extra_needed = total_movers - 1
    extra_sources: list[tuple[int, int]] = []
    candidate_sources = []
    for r in range(1, bar_r - 2):
        if r == bar_r - 2:
            continue
        for c in range(1, w - 1):
            if c == trigger_col:
                continue
            shifted_r = r + 1
            if forbidden[shifted_r, c]:
                continue
            candidate_sources.append((r, c))
    random.shuffle(candidate_sources)

    for r, c in candidate_sources:
        if len(extra_sources) == extra_needed:
            break
        if any(abs(r - rr) <= 1 and abs(c - cc) <= 1 for rr, cc in extra_sources):
            continue
        extra_sources.append((r, c))
    if len(extra_sources) != extra_needed:
        return None

    for r, c in extra_sources:
        inp[r, c] = MOVING

    shifted_preview, _splash_preview = _shift_and_splash(inp)
    cavities = _cavity_components(shifted_preview, bar_r)
    if len(cavities) != 3:
        return None

    top_candidates = _sector_candidates(shifted_preview, cavities)
    usable_indices = [idx for idx, cols in enumerate(top_candidates) if cols]
    if len(usable_indices) < 1:
        return None

    requested_seed_count = random.choices((1, 2, 3), weights=SEED_COUNT_WEIGHTS, k=1)[0]
    seed_count = min(requested_seed_count, len(usable_indices))
    chosen_components = random.sample(usable_indices, seed_count)
    random.shuffle(chosen_components)
    chosen_colors = random.sample(SEED_COLORS, seed_count)

    used_seed_cols: set[int] = set()
    for comp_idx, color in zip(chosen_components, chosen_colors, strict=False):
        options = top_candidates[comp_idx][:]
        random.shuffle(options)
        placed = False
        for c in options:
            if any(abs(c - prev) <= 1 for prev in used_seed_cols):
                continue
            inp[0, c] = color
            used_seed_cols.add(c)
            placed = True
            break
        if not placed:
            return None

    out = _oracle(inp)
    if not np.array_equal(out, main(inp)):
        return None
    if not _independent_geometry_checks(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for the splash-fence puzzle")
