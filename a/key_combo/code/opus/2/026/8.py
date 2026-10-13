from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox, crop_bbox


GRID_SIZE = 12
MASK_COLOR = Color.ORANGE
PALETTE = np.array([0, 1, 2, 3], dtype=int)
PERIODS = (3, 4)
GEN_ATTEMPTS = 500
OUTER_ZERO_PROB = 0.4


def _period_candidates(grid: np.ndarray) -> list[int]:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    valid: list[int] = []
    for period in PERIODS:
        ok = True
        for r in range(height):
            rr = (r + period) % height
            for c in range(width):
                a = int(g[r, c])
                b = int(g[rr, c])
                if a == MASK_COLOR or b == MASK_COLOR:
                    continue
                if a != b:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            valid.append(period)
    return valid


def _restore_full(grid: np.ndarray, period: int) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    out = g.copy()
    masked = [tuple(map(int, rc)) for rc in np.argwhere(g == MASK_COLOR)]

    for _ in range(len(masked) + 1):
        changed = False
        for r, c in masked:
            if int(out[r, c]) != MASK_COLOR:
                continue
            colors: list[int] = []

            mirror = int(out[r, width - 1 - c])
            if mirror != MASK_COLOR:
                colors.append(mirror)

            for rr in ((r + period) % height, (r - period) % height):
                color = int(out[rr, c])
                if color != MASK_COLOR:
                    colors.append(color)

            if colors and len(set(colors)) == 1:
                out[r, c] = colors[0]
                changed = True
        if not changed:
            break
    return out


def _mask_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    mask = np.asarray(grid == MASK_COLOR, dtype=bool)
    b = bbox(mask)
    assert b is not None, "expected one orange rectangle"
    r0, c0, r1, c1 = b
    rect = crop_bbox(mask.astype(int), b).astype(bool)
    assert bool(np.all(rect)), "orange component must be a solid rectangle"
    assert int(mask.sum()) == int(rect.size), "expected exactly one orange rectangle"
    return int(r0), int(c0), int(r1), int(c1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the masked rectangle by combining horizontal mirror symmetry with vertical periodicity.

    Input structure:
    - The input is a 12x12 grid on a black background, with colors 1/2/3 forming a pattern.
    - The pattern is horizontally mirror-symmetric across the axis between columns 5 and 6.
    - The same pattern also repeats vertically with period 3 or 4.
    - One solid orange rectangle replaces part of the pattern.

    Rule:
    - Locate the orange rectangle and output a grid with exactly that rectangle's size.
    - Infer the smallest valid vertical period from the visible non-orange pixels.
    - Restore each orange cell from the working grid by using its visible mirror cell when available,
      otherwise by using a visible cell one period above or below in the same column.
    - If multiple visible correspondences exist, they must agree; return the restored rectangle crop.

    Key insight:
    - The hidden patch sits inside a pattern that is redundant in two independent ways, so the mask
      can be reconstructed from either a horizontal mirror twin or a vertically periodic twin.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_SIZE, GRID_SIZE), "expected a 12x12 grid"

    b = _mask_bbox(g)
    candidates = _period_candidates(g)
    assert candidates, "expected period 3 or 4"
    period = min(candidates)

    restored = _restore_full(g, period)
    mask = g == MASK_COLOR
    assert not np.any(restored[mask] == MASK_COLOR), "mask must be fully recoverable"
    return crop_bbox(restored, b)


def _sample_pattern(period: int) -> np.ndarray:
    while True:
        left = np.random.choice([1, 2, 3], size=(period, GRID_SIZE // 2))
        zero_mask = np.random.random(size=left[:, :2].shape) < OUTER_ZERO_PROB
        left[:, :2][zero_mask] = 0
        if not all(np.any(left == color) for color in (1, 2, 3)):
            continue

        tile = np.concatenate([left, left[:, ::-1]], axis=1)
        pattern = np.vstack([tile[r % period] for r in range(GRID_SIZE)]).astype(int)
        if _period_candidates(pattern) != [period]:
            continue
        return pattern


def _choose_rect(period: int) -> tuple[int, int, int, int]:
    if period == 4:
        height = 5
    else:
        height = int(np.random.choice([4, 5]))
    width = int(np.random.choice([3, 4]))

    if width == 3:
        c0 = int(np.random.choice([4, 5]))
    else:
        c0 = int(np.random.choice([3, 5]))
    r0 = int(np.random.choice(np.arange(0, GRID_SIZE - height + 1)))
    return r0, c0, height, width


def _coverage_stats(r0: int, c0: int, height: int, width: int, period: int) -> dict[str, int]:
    r1 = r0 + height - 1
    c1 = c0 + width - 1

    def in_rect(r: int, c: int) -> bool:
        return r0 <= r <= r1 and c0 <= c <= c1

    stats = {
        "mirror_blocked": 0,
        "mirror_visible": 0,
        "period_overlap": 0,
        "mirror_hint": 0,
        "period_hint": 0,
    }
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            mirror_blocked = in_rect(r, GRID_SIZE - 1 - c)
            overlap = in_rect(r + period, c) or in_rect(r - period, c)
            if mirror_blocked:
                stats["mirror_blocked"] += 1
                stats["period_hint"] += 1
            else:
                stats["mirror_visible"] += 1
                if overlap:
                    stats["mirror_hint"] += 1
            if overlap:
                stats["period_overlap"] += 1
    return stats


def _mirror_only_recovers(masked: np.ndarray) -> np.ndarray:
    out = masked.copy()
    for _ in range(GRID_SIZE):
        changed = False
        for r, c in np.argwhere(out == MASK_COLOR):
            r = int(r)
            c = int(c)
            mirror = int(out[r, GRID_SIZE - 1 - c])
            if mirror != MASK_COLOR:
                out[r, c] = mirror
                changed = True
        if not changed:
            break
    return out


def _period_only_recovers(masked: np.ndarray, period: int) -> np.ndarray:
    out = masked.copy()
    for _ in range(GRID_SIZE):
        changed = False
        for r, c in np.argwhere(out == MASK_COLOR):
            r = int(r)
            c = int(c)
            colors: list[int] = []
            for rr in ((r + period) % GRID_SIZE, (r - period) % GRID_SIZE):
                color = int(out[rr, c])
                if color != MASK_COLOR:
                    colors.append(color)
            if colors and len(set(colors)) == 1:
                out[r, c] = colors[0]
                changed = True
        if not changed:
            break
    return out


def _valid_instance(pattern: np.ndarray, masked: np.ndarray, period: int, rect: tuple[int, int, int, int]) -> bool:
    r0, c0, height, width = rect
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    true_patch = pattern[r0 : r1 + 1, c0 : c1 + 1]
    if len(np.unique(true_patch)) < 2:
        return False
    if not np.any(true_patch != 0):
        return False

    stats = _coverage_stats(r0, c0, height, width, period)
    if stats["mirror_blocked"] == 0 or stats["mirror_visible"] == 0:
        return False
    if stats["period_overlap"] == 0 or stats["mirror_hint"] == 0 or stats["period_hint"] == 0:
        return False

    if _period_candidates(masked) != [period]:
        return False

    restored = _restore_full(masked, period)
    if not np.array_equal(restored[r0 : r1 + 1, c0 : c1 + 1], true_patch):
        return False

    mirror_only = _mirror_only_recovers(masked)
    if not np.any(mirror_only == MASK_COLOR):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        period = int(np.random.choice(PERIODS))
        pattern = _sample_pattern(period)
        rect = _choose_rect(period)
        r0, c0, height, width = rect
        r1 = r0 + height - 1
        c1 = c0 + width - 1

        masked = pattern.copy()
        masked[r0 : r1 + 1, c0 : c1 + 1] = MASK_COLOR
        if _valid_instance(pattern, masked, period, rect):
            return masked

    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
