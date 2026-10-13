from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, components_mask, get_bg


PINK = Color.PINK
YELLOW = Color.YELLOW
INPUT_COLORS = [Color.BLUE, Color.RED, Color.GREEN, Color.GREY, Color.ORANGE, Color.TEAL, Color.MAROON]
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
PLUS_ARM_OFFSETS = [(-3, 0), (-2, 0), (2, 0), (3, 0), (0, -3), (0, -2), (0, 2), (0, 3)]
PLUS_GAP_OFFSETS = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
GEN_HALF_H_RANGE = (9, 16)
GEN_W_RANGE = (10, 21)
GEN_MAX_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _component_is_domino(coords: np.ndarray) -> bool:
    if int(coords.shape[0]) != 2:
        return False
    r0, c0, r1, c1 = bbox_coords(coords)
    return (r1 - r0, c1 - c0) in {(1, 0), (0, 1)}


def _find_implied_plus_centers(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    centers: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                continue
            ok = True
            for dr, dc in PLUS_GAP_OFFSETS:
                rr, cc = r + dr, c + dc
                if not _in_bounds(mask.shape, rr, cc) or mask[rr, cc]:
                    ok = False
                    break
            if not ok:
                continue
            for dr, dc in PLUS_ARM_OFFSETS:
                rr, cc = r + dr, c + dc
                if not _in_bounds(mask.shape, rr, cc) or not mask[rr, cc]:
                    ok = False
                    break
            if ok:
                centers.append((r, c))
    return centers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reveal the symmetric difference of the upper and lower halves as pink micro-bars, then fill each implied-plus center in yellow.

    Input structure:
    - The grid height is even and the canvas is split into equally sized upper and lower halves with no separator.
    - Both halves share one background color; the upper half has one foreground color and the lower half has another.
    - When aligned by position, the two halves differ only on isolated horizontal or vertical two-cell bars; some quartets of bars surround one empty center with one-cell gaps.

    Rule:
    - Split the grid into the upper and lower halves and compare aligned positions by foreground-vs-background occupancy.
    - Mark positions where exactly one half is foreground; those symmetric-difference cells become pink in the output half-grid.
    - Whenever the revealed pink bars occupy the eight cells at offsets up/down/left/right two and three from one empty center, paint that missing center yellow.

    Key insight:
    - The bars and the plus do not exist inside either half on their own; they appear only after taking the aligned symmetric difference of the two half-masks.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] % 2 == 0

    bg = int(get_bg(g))
    half_h = g.shape[0] // 2
    top = g[:half_h]
    bottom = g[half_h:]
    diff = (top != bg) ^ (bottom != bg)

    comps = components_mask(diff, connectivity=4)
    assert comps, "revealed shape must contain bars"
    assert all(_component_is_domino(coords) for coords in comps), "revealed foreground must be domino bars"

    out = np.full(diff.shape, bg, dtype=int)
    out[diff] = PINK
    for r, c in _find_implied_plus_centers(diff):
        out[r, c] = YELLOW
    return out


def _mark_cells(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        if _in_bounds((h, w), r, c):
            blocked[r, c] = True


def _shield_around(cells: list[tuple[int, int]], shape: tuple[int, int]) -> list[tuple[int, int]]:
    shield: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr, dc in NEIGHBORS8:
            rr, cc = r + dr, c + dc
            if _in_bounds(shape, rr, cc):
                shield.add((rr, cc))
    return sorted(shield)


def _plus_cells(center: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = center
    return sorted((r + dr, c + dc) for dr, dc in PLUS_ARM_OFFSETS)


def _plus_shield(center: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = center
    shield = []
    for rr in range(r - 4, r + 5):
        for cc in range(c - 4, c + 5):
            if _in_bounds(shape, rr, cc):
                shield.append((rr, cc))
    return shield


def _noise_bar_cells(rng: np.random.Generator, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    if rng.random() < 0.5:
        r = int(rng.integers(0, h))
        c = int(rng.integers(0, w - 1))
        return [(r, c), (r, c + 1)]
    r = int(rng.integers(0, h - 1))
    c = int(rng.integers(0, w))
    return [(r, c), (r + 1, c)]


def _place_output_mask(
    rng: np.random.Generator, shape: tuple[int, int]
) -> tuple[np.ndarray, list[tuple[int, int]], np.ndarray]:
    h, w = shape
    mask = np.zeros(shape, dtype=bool)
    blocked = np.zeros(shape, dtype=bool)

    max_pluses = 1
    if h >= 12 and w >= 14:
        max_pluses = 2
    if h >= 15 and w >= 19:
        max_pluses = 3
    plus_count = int(rng.integers(1, max_pluses + 1))

    centers: list[tuple[int, int]] = []
    for _ in range(plus_count):
        candidates = [(r, c) for r in range(3, h - 3) for c in range(3, w - 3)]
        rng.shuffle(candidates)
        placed = False
        for center in candidates:
            cells = _plus_cells(center)
            shield = _plus_shield(center, shape)
            if any(blocked[r, c] for r, c in shield):
                continue
            for r, c in cells:
                mask[r, c] = True
            _mark_cells(blocked, shield)
            centers.append(center)
            placed = True
            break
        if not placed:
            raise ValueError("failed to place implied plus")

    max_noise = max(2, min(6, (h * w) // 18))
    noise_target = int(rng.integers(1, max_noise + 1))
    noise_placed = 0
    for _ in range(200):
        if noise_placed >= noise_target:
            break
        cells = _noise_bar_cells(rng, shape)
        shield = _shield_around(cells, shape)
        if any(blocked[r, c] for r, c in shield):
            continue
        for r, c in cells:
            mask[r, c] = True
        _mark_cells(blocked, shield)
        noise_placed += 1
    if noise_placed == 0:
        raise ValueError("failed to place noise bars")

    return mask, centers, blocked


def _split_dominoes(
    rng: np.random.Generator, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    top = np.zeros_like(mask, dtype=bool)
    bottom = np.zeros_like(mask, dtype=bool)
    for coords in components_mask(mask, connectivity=4):
        assert _component_is_domino(coords)
        pts = [(int(r), int(c)) for r, c in coords]
        if rng.random() < 0.5:
            pts = pts[::-1]
        (r0, c0), (r1, c1) = pts
        top[r0, c0] = True
        bottom[r1, c1] = True
    return top, bottom


def _shared_point_ok(
    top: np.ndarray, bottom: np.ndarray, blocked: np.ndarray, r: int, c: int
) -> bool:
    if blocked[r, c] or top[r, c] or bottom[r, c]:
        return False
    for dr, dc in NEIGHBORS4:
        rr, cc = r + dr, c + dc
        if _in_bounds(top.shape, rr, cc):
            if top[rr, cc] or bottom[rr, cc]:
                return False
    return True


def _add_shared_points(
    rng: np.random.Generator,
    output_blocked: np.ndarray,
    top: np.ndarray,
    bottom: np.ndarray,
) -> int:
    shared_blocked = output_blocked.copy()
    count = 0
    candidates = [(r, c) for r in range(top.shape[0]) for c in range(top.shape[1])]
    rng.shuffle(candidates)
    target = int(rng.integers(1, min(7, len(candidates)) + 1))
    for r, c in candidates:
        if count >= target:
            break
        if not _shared_point_ok(top, bottom, shared_blocked, r, c):
            continue
        top[r, c] = True
        bottom[r, c] = True
        for dr, dc in NEIGHBORS8:
            rr, cc = r + dr, c + dc
            if _in_bounds(top.shape, rr, cc):
                shared_blocked[rr, cc] = True
        count += 1
    return count


def _make_output(bg: int, mask: np.ndarray, centers: list[tuple[int, int]]) -> np.ndarray:
    out = np.full(mask.shape, int(bg), dtype=int)
    out[mask] = PINK
    for r, c in centers:
        out[r, c] = YELLOW
    return out


def _validate_half_is_singletons(mask: np.ndarray) -> bool:
    comps = components_mask(mask, connectivity=4)
    return all(int(coords.shape[0]) == 1 for coords in comps)


def _build_input_from_masks(
    bg: int, top_fg: int, bottom_fg: int, top_mask: np.ndarray, bottom_mask: np.ndarray
) -> np.ndarray:
    top = np.full(top_mask.shape, int(bg), dtype=int)
    bottom = np.full(bottom_mask.shape, int(bg), dtype=int)
    top[top_mask] = int(top_fg)
    bottom[bottom_mask] = int(bottom_fg)
    return np.vstack([top, bottom])


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    half_h = int(rng.integers(*GEN_HALF_H_RANGE))
    w = int(rng.integers(*GEN_W_RANGE))
    shape = (half_h, w)

    colors = list(rng.choice(INPUT_COLORS, size=3, replace=False))
    bg, top_fg, bottom_fg = map(int, colors)

    try:
        target_mask, centers, output_blocked = _place_output_mask(rng, shape)
    except ValueError:
        return None

    top_mask, bottom_mask = _split_dominoes(rng, target_mask)
    shared_count = _add_shared_points(rng, output_blocked, top_mask, bottom_mask)
    if shared_count == 0:
        return None

    if not _validate_half_is_singletons(top_mask):
        return None
    if not _validate_half_is_singletons(bottom_mask):
        return None

    xor_mask = top_mask ^ bottom_mask
    if not np.array_equal(xor_mask, target_mask):
        return None

    plus_centers = _find_implied_plus_centers(target_mask)
    if sorted(plus_centers) != sorted(centers):
        return None

    comps = components_mask(target_mask, connectivity=4)
    if not comps or not all(_component_is_domino(coords) for coords in comps):
        return None
    if len(comps) - 4 * len(centers) <= 0:
        return None

    inp = _build_input_from_masks(bg, top_fg, bottom_fg, top_mask, bottom_mask)
    expected = _make_output(bg, target_mask, centers)
    if not np.array_equal(main(inp), expected):
        return None
    if np.array_equal((inp[:half_h] != bg) | (inp[half_h:] != bg), target_mask):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetric-difference plus input")
