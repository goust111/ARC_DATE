# Combo variant of 8e5c0c38 (index 29)
# combo_spec: D6/1+D13/0
# Name: 旋转增强旋转对称修复

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import colors_present, get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace:
#    The original task kept a same-row mirror mate across one vertical axis. This combo
#    changes the survivor test to a 90-degree orbit around one hidden center per color.
#    Example on a 5x5 local sprite: if a pixel at (0,3) is valid, then the same color
#    must also appear at (1,0), (4,1), and (3,4). A lone extra pixel adjacent to one arm
#    is deleted even if it visually "looks close" to the shape, because its quarter-turn
#    orbit is incomplete. The global D13 augmentation now rotates the whole 22x22 scene
#    by 0/90/180/270 degrees after construction, so the same local rule must be recovered
#    regardless of where the three sprites land after that scene rotation.
# b) Complexity calibration:
#    A plain plus sign would be too simple because it is also mirror-symmetric and would
#    collapse back toward the source task. The generator therefore builds chiral C4
#    sprites: they are invariant under quarter-turns but reject vertical and horizontal
#    mirror tests. This keeps the visual change obvious while still using only one rule.
# c) Hidden assumptions broken by the original main():
#    - "best symmetry parameter is one scalar axis k" no longer holds; we now search a
#      2D center (kr/2, kc/2) on the doubled lattice.
#    - "partner stays on the same row" no longer holds; 90-degree rotation changes both
#      row and column.
#    - "one mirror lookup is enough" no longer holds; a pixel survives only if its full
#      four-step orbit closes back on itself and every visited cell has the same color.
# d) Placement legality:
#    The original separated slot layout is preserved, and each color still owns exactly
#    one sprite. Clean sprites are generated as square local canvases that fit entirely
#    inside their chosen slot. Noise is added only to empty cells 4-adjacent to that
#    sprite and remains inside the same local square, so objects never overlap.
# e) Visual observability:
#    A human can compare input/output and see that whole pinwheel-like monochrome sprites
#    remain, while attached off-orbit pixels disappear. Because the clean shapes are not
#    left-right symmetric, the old "vertical mirror repair" explanation no longer fits.
# f) Visual tidiness:
#    Objects stay in three large separated regions, use distinct colors, and noise is
#    sparse and local. No extra markers, legends, or color remapping are introduced.
# g) Color reasoning:
#    Colors only index independent sprites. The rule depends on geometry alone, not on
#    whether a color id is small, large, warm, cool, or black.
# h) Anti-degeneration check:
#    The generator rejects any sample where the legacy vertical-axis cleaner would produce
#    the same output as the new solver. It also requires every clean sprite to lose at
#    least a few pixels under the best vertical-mirror score, making the C4 behavior real.
# i) New solver assumptions and generator guarantees:
#    - Assumption: one non-background color corresponds to one hidden C4 center.
#      Guarantee: generator still uses exactly three distinct non-background colors.
#    - Assumption: the true center wins a complete-orbit score search.
#      Guarantee: clean sprites and noisy sprites must pass center-gap thresholds.
#    - Assumption: noise never completes a valid quarter-turn orbit.
#      Guarantee: at most one cell is sampled from any empty orbit class.


def _rotate90_about(r: int, c: int, kr: int, kc: int) -> tuple[int, int] | None:
    """Rotate one cell 90 degrees clockwise about center (kr/2, kc/2)."""
    rr2 = int(kr) + int(kc) - 2 * int(c)
    cc2 = int(kc) - int(kr) + 2 * int(r)
    if rr2 % 2 != 0 or cc2 % 2 != 0:
        return None
    return rr2 // 2, cc2 // 2


def _orbit_c4(
    r: int,
    c: int,
    kr: int,
    kc: int,
    height: int,
    width: int,
) -> set[tuple[int, int]] | None:
    """Return the C4 orbit of one cell, or None if the orbit leaves the grid."""
    orbit: list[tuple[int, int]] = []
    cur = (int(r), int(c))
    for _ in range(4):
        rr, cc = cur
        if not (0 <= rr < height and 0 <= cc < width):
            return None
        orbit.append((rr, cc))
        nxt = _rotate90_about(rr, cc, kr, kc)
        if nxt is None:
            return None
        cur = nxt
    if cur != (int(r), int(c)):
        return None
    return set(orbit)


def _c4_score(
    pts: set[tuple[int, int]],
    kr: int,
    kc: int,
    height: int,
    width: int,
) -> int:
    """Count how many pixels already belong to complete same-color quarter-turn orbits."""
    score = 0
    for r, c in pts:
        orbit = _orbit_c4(r, c, kr, kc, height, width)
        if orbit is not None and orbit <= pts:
            score += 1
    return score


def _best_c4_center(
    pts: set[tuple[int, int]],
    height: int,
    width: int,
) -> tuple[int, int, int]:
    best_score = -1
    best_kr = 0
    best_kc = 0
    for kr in range(0, 2 * (height - 1) + 1):
        for kc in range(0, 2 * (width - 1) + 1):
            if (kr - kc) % 2 != 0:
                continue
            score = _c4_score(pts, kr, kc, height, width)
            if score > best_score:
                best_score = int(score)
                best_kr = int(kr)
                best_kc = int(kc)
    return best_kr, best_kc, best_score


def _second_best_c4_score(
    pts: set[tuple[int, int]],
    true_center: tuple[int, int],
    height: int,
    width: int,
) -> int:
    true_kr, true_kc = map(int, true_center)
    best = -1
    for kr in range(0, 2 * (height - 1) + 1):
        for kc in range(0, 2 * (width - 1) + 1):
            if (kr - kc) % 2 != 0:
                continue
            if kr == true_kr and kc == true_kc:
                continue
            best = max(best, _c4_score(pts, kr, kc, height, width))
    return max(best, 0)


def _best_vertical_score(coords: list[tuple[int, int]], width: int) -> int:
    pts = {(int(r), int(c)) for r, c in coords}
    best = 0
    for k in range(0, 2 * (width - 1) + 1):
        score = 0
        for r, c in pts:
            cc = int(k) - int(c)
            if 0 <= cc < width and (int(r), cc) in pts:
                score += 1
        best = max(best, score)
    return best


def _legacy_vertical_clean(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    _height, width = g.shape
    out = g.copy()
    for col in colors_present(g, exclude={bg}):
        pts = np.argwhere(g == int(col))
        if pts.size == 0:
            continue
        best_score = -1
        best_k = 0
        for k in range(0, 2 * (width - 1) + 1):
            score = 0
            for r, c in pts:
                cc = int(k) - int(c)
                if 0 <= cc < width and int(g[int(r), cc]) == int(col):
                    score += 1
            if score > best_score:
                best_score = int(score)
                best_k = int(k)
        for r, c in pts:
            cc = int(best_k) - int(c)
            if not (0 <= cc < width and int(g[int(r), cc]) == int(col)):
                out[int(r), int(c)] = int(bg)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Clean each color layer by enforcing 90-degree rotational symmetry.

    Input structure:
    - One background color fills most of the 22x22 grid.
    - Exactly several non-background colors appear as separate monochrome sprites, each close to a quarter-turn symmetric square-like shape around its own hidden center.
    - Extra same-color pixels may be attached near a sprite, but they do not complete the sprite's 0/90/180/270-degree orbit.

    Rule:
    - For each non-background color independently, search for the rotation center that maximizes complete same-color quarter-turn orbits.
    - Delete any pixel whose orbit under that chosen center is incomplete, keeping only pixels supported at all rotated positions.

    Key insight:
    - The task is still per-color denoising, but the hidden parameter is now a 2D rotation center rather than a 1D mirror axis, and a survivor must satisfy a four-position consistency check instead of a single mirror check.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    height, width = g.shape

    out = g.copy()
    colors = colors_present(g, exclude={bg})
    assert colors, "grid must contain at least one non-background color"

    for col in colors:
        pts = {(int(r), int(c)) for r, c in np.argwhere(g == int(col))}
        if not pts:
            continue

        best_kr, best_kc, _best_score = _best_c4_center(pts, height, width)
        for r, c in pts:
            orbit = _orbit_c4(r, c, best_kr, best_kc, height, width)
            if orbit is None or not (orbit <= pts):
                out[int(r), int(c)] = int(bg)

    return out


# --------------- generator ---------------

GRID_SIZE = 22
NUM_OBJECTS = 4  # bg + 3 sprite colors
LAYOUTS = [
    [(0, 0, 7, 7), (0, 14, 7, 21), (14, 5, 21, 16)],
    [(0, 0, 8, 5), (0, 16, 8, 21), (15, 5, 21, 16)],
]

SIDE_MIN = 4
SIDE_MAX = 7
PIXEL_COUNT_MIN = 8
PIXEL_COUNT_MAX = 40
DENSITY_LO = 0.32
DENSITY_HI = 0.82

NOISE_FRAC_LO = 0.12
NOISE_FRAC_HI = 0.30
NOISE_MAX = 8

CLEAN_CENTER_GAP_MIN = 4
FINAL_CENTER_GAP_MIN = 3
MIN_VERTICAL_DIFF = 4
MIN_VARIANT_DIFF = 4

SPRITE_RETRIES = 260
GEN_RETRIES = 500

FLIP_PROB = 0.5


def _local_rot90(cell: tuple[int, int], side: int) -> tuple[int, int]:
    r, c = map(int, cell)
    return int(side - 1 - c), int(r)


def _local_c4_orbits(side: int) -> list[tuple[tuple[int, int], ...]]:
    orbits: list[tuple[tuple[int, int], ...]] = []
    seen: set[tuple[int, int]] = set()
    for r in range(side):
        for c in range(side):
            if (r, c) in seen:
                continue
            cur = (r, c)
            orbit: list[tuple[int, int]] = []
            for _ in range(4):
                orbit.append(cur)
                cur = _local_rot90(cur, side)
            uniq = tuple(sorted(set(orbit)))
            for cell in uniq:
                seen.add(cell)
            orbits.append(uniq)
    return orbits


def _touch_count(orbit: tuple[tuple[int, int], ...], occupied: set[tuple[int, int]]) -> int:
    count = 0
    for r, c in orbit:
        count += int((r - 1, c) in occupied)
        count += int((r + 1, c) in occupied)
        count += int((r, c - 1) in occupied)
        count += int((r, c + 1) in occupied)
    return count


def _orbit_touches(orbit: tuple[tuple[int, int], ...], occupied: set[tuple[int, int]]) -> bool:
    return _touch_count(orbit, occupied) > 0


def _component_count(mask: np.ndarray) -> int:
    side = int(mask.shape[0])
    seen = np.zeros_like(mask, dtype=bool)
    count = 0
    for r in range(side):
        for c in range(side):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            count += 1
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < side and 0 <= nc < side and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
    return count


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _build_clean_sprite(color: int, max_side: int) -> tuple[np.ndarray, tuple[int, int]]:
    candidate_sides = [side for side in range(SIDE_MIN, min(SIDE_MAX, max_side) + 1)]
    if not candidate_sides:
        raise ValueError("no feasible square side")

    for _ in range(SPRITE_RETRIES):
        side = random.choice(candidate_sides)
        area = side * side
        orbits = _local_c4_orbits(side)

        center = (side - 1) / 2.0
        hub_idx = min(
            range(len(orbits)),
            key=lambda idx: min(abs(r - center) + abs(c - center) for r, c in orbits[idx]),
        )
        edge_indices = [idx for idx, orbit in enumerate(orbits) if any(r == 0 for r, _c in orbit)]
        if not edge_indices:
            continue

        chosen = {hub_idx, random.choice(edge_indices)}
        base_pixels = sum(len(orbits[idx]) for idx in chosen)

        feasible_pixel_counts: list[int] = []
        max_extra = len(orbits) - len(chosen)
        for extra in range(0, max_extra + 1):
            pixels = base_pixels + 4 * extra
            density = pixels / area
            if PIXEL_COUNT_MIN <= pixels <= PIXEL_COUNT_MAX and DENSITY_LO <= density <= DENSITY_HI:
                feasible_pixel_counts.append(pixels)
        if not feasible_pixel_counts:
            continue

        target_pixels = random.choice(feasible_pixel_counts)
        target_orbits = len(chosen) + (target_pixels - base_pixels) // 4

        occupied: set[tuple[int, int]] = set()
        for idx in chosen:
            occupied.update(orbits[idx])

        ok = True
        while len(chosen) < target_orbits:
            frontier = [idx for idx in range(len(orbits)) if idx not in chosen and _orbit_touches(orbits[idx], occupied)]
            if not frontier:
                ok = False
                break
            weights = [1.0 + 0.7 * _touch_count(orbits[idx], occupied) for idx in frontier]
            idx = random.choices(frontier, weights=weights, k=1)[0]
            chosen.add(idx)
            occupied.update(orbits[idx])
        if not ok:
            continue

        mask = np.zeros((side, side), dtype=bool)
        for r, c in occupied:
            mask[int(r), int(c)] = True

        if not _touches_all_sides(mask):
            continue
        if _component_count(mask) != 1:
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        if np.array_equal(mask, np.flipud(mask)):
            continue

        coords = [tuple(x) for x in np.argwhere(mask)]
        true_center = (side - 1, side - 1)
        pixel_count = len(coords)
        second_best = _second_best_c4_score(set(coords), true_center, side, side)
        if pixel_count - second_best < CLEAN_CENTER_GAP_MIN:
            continue
        if pixel_count - _best_vertical_score(coords, side) < MIN_VERTICAL_DIFF:
            continue

        sprite = np.zeros((side, side), dtype=int)
        sprite[mask] = int(color)
        return sprite, true_center

    raise ValueError("failed to build clean C4 sprite")


def _adjacent_empty(mask: np.ndarray) -> list[tuple[int, int]]:
    adj = np.zeros_like(mask, dtype=bool)
    adj[1:, :] |= mask[:-1, :]
    adj[:-1, :] |= mask[1:, :]
    adj[:, 1:] |= mask[:, :-1]
    adj[:, :-1] |= mask[:, 1:]
    adj &= ~mask
    rows, cols = np.where(adj)
    return list(zip(rows.tolist(), cols.tolist()))


def _add_noise_local(clean_sprite: np.ndarray, color: int) -> np.ndarray | None:
    side = int(clean_sprite.shape[0])
    clean_mask = clean_sprite == int(color)
    clean_coords = {(int(r), int(c)) for r, c in np.argwhere(clean_mask)}
    clean_count = len(clean_coords)
    true_center = (side - 1, side - 1)

    orbit_choices: dict[tuple[tuple[int, int], ...], list[tuple[int, int]]] = {}
    for cell in _adjacent_empty(clean_mask):
        orbit = _orbit_c4(int(cell[0]), int(cell[1]), true_center[0], true_center[1], side, side)
        if orbit is None:
            continue
        if orbit & clean_coords:
            continue
        key = tuple(sorted(orbit))
        orbit_choices.setdefault(key, []).append(cell)

    lo = max(1, math.ceil(NOISE_FRAC_LO * clean_count))
    hi = min(NOISE_MAX, math.floor(NOISE_FRAC_HI * clean_count), len(orbit_choices))
    if lo > hi:
        return None

    target = random.randint(lo, hi)
    chosen: list[tuple[int, int]] = []
    current = set(clean_coords)
    remaining = dict(orbit_choices)

    for _ in range(target):
        scored: list[tuple[int, float, tuple[tuple[int, int], ...], tuple[int, int]]] = []
        for key, cells in remaining.items():
            cell = random.choice(cells)
            alt = _second_best_c4_score(current | {cell}, true_center, side, side)
            scored.append((alt, random.random(), key, cell))
        if not scored:
            return None
        scored.sort(key=lambda item: (item[0], item[1]))
        best_alt = scored[0][0]
        pool = [item for item in scored if item[0] <= best_alt + 1]
        _alt, _rand, key, cell = random.choice(pool)
        chosen.append(cell)
        current.add(cell)
        remaining.pop(key, None)

    if clean_count - _second_best_c4_score(current, true_center, side, side) < FINAL_CENTER_GAP_MIN:
        return None

    noisy = clean_sprite.copy()
    for r, c in chosen:
        noisy[int(r), int(c)] = int(color)
    return noisy


def _place_sprites(
    bg: int,
    sprite_specs: list[dict],
    slots: list[tuple[int, int, int, int]],
) -> tuple[np.ndarray, np.ndarray] | None:
    inp = np.full((GRID_SIZE, GRID_SIZE), int(bg), dtype=int)
    out = np.full((GRID_SIZE, GRID_SIZE), int(bg), dtype=int)
    for spec, (r0s, c0s, r1s, c1s) in zip(sprite_specs, slots):
        clean = spec["clean"]
        noisy = spec["noisy"]
        side = int(clean.shape[0])
        if side > r1s - r0s + 1 or side > c1s - c0s + 1:
            return None
        r0 = random.randint(r0s, r1s - side + 1)
        c0 = random.randint(c0s, c1s - side + 1)
        paste(inp, noisy, r0, c0, transparent=0)
        paste(out, clean, r0, c0, transparent=0)
    return inp, out


def _augment(inp: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k = random.randint(0, 3)
    if k:
        inp = np.rot90(inp, k)
        out = np.rot90(out, k)
    if random.random() < FLIP_PROB:
        inp = np.flipud(inp)
        out = np.flipud(out)
    if random.random() < FLIP_PROB:
        inp = np.fliplr(inp)
        out = np.fliplr(out)
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    bg, *obj_colors = random.sample(range(10), NUM_OBJECTS)
    slots = random.choice(LAYOUTS)

    sprite_specs: list[dict] = []
    try:
        for color, slot in zip(obj_colors, slots):
            max_h = int(slot[2] - slot[0] + 1)
            max_w = int(slot[3] - slot[1] + 1)
            clean, _true_center = _build_clean_sprite(int(color), min(max_h, max_w))
            noisy = _add_noise_local(clean, int(color))
            if noisy is None:
                return None
            sprite_specs.append({"clean": clean, "noisy": noisy})
    except ValueError:
        return None

    placed = _place_sprites(int(bg), sprite_specs, slots)
    if placed is None:
        return None
    inp, out = placed
    inp, out = _augment(inp, out)

    if not np.array_equal(main(inp), out):
        return None

    legacy = _legacy_vertical_clean(inp)
    if np.array_equal(legacy, out):
        return None
    if int(np.count_nonzero(legacy != out)) < MIN_VARIANT_DIFF:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
