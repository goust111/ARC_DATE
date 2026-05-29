# Variant of 8e5c0c38
# Dimension: D1 — 对称修复的变换方式 (transform)
# Change: 删除没有镜像对应像素的噪声像素，使精灵恢复垂直镜像对称... → 删除像素使精灵变为点对称而非轴对称
#
# Design analysis:
# variant_spec:
#   core_change:
#     original_rule: each color layer is cleaned by finding a vertical axis and deleting pixels without a mirror mate
#     new_rule: each color layer is cleaned by finding a 180-degree rotation center and deleting pixels without a rotated mate
#   concrete_behavior:
#     input:
#       - 22x22 grid, one background color, exactly 3 non-background sprite colors
#       - each color appears in a separate slot region as one noisy sprite
#       - each clean sprite is point-symmetric around the center of its bbox, which may be integer or half-integer in row and/or column
#       - noise pixels are attached near the clean shape but their 180-degree counterparts are absent
#     output:
#       - keep only those same-color pixels that belong to the best point-symmetric explanation of that color layer
#     trace_example:
#       - original logic on a pixel at (r,c): test only (r, k-c) on the same row
#       - new logic on a pixel at (r,c): test (kr-r, kc-c), so both row and column change together
#       - consequence: a diagonal pair can survive now even if it has no left-right mirror on its row
#   complexity_calibration:
#     - keeping separate colors and fixed layout slots avoids cross-object ambiguity even though center search is now 2D
#     - clean sprites are rejected if they look too close to the original vertical-axis family, so the visual change is observable
#     - only one symmetry type exists in the variant: per-color point symmetry; no mixed branches are introduced
#   implicit_assumption_audit:
#     - original best_k loop assumes one scalar axis parameter; variant needs a pair (kr,kc)
#     - original deletion step uses same-row partner lookup; variant must recompute both row and col of the partner
#     - original generator hardcodes left-half drawing plus mirroring; variant must instead build 180-degree orbit pairs
#     - original noise legality checks only the mirrored column; variant must check the rotated partner cell and exclude the exact center cell
#   placement_legality:
#     - sprites are still placed output-first into non-overlapping slot boxes
#     - each sprite's true center is derived from its local bbox center plus placement offset, so solver search and generator labels agree
#     - noise is added only to background cells adjacent to the target sprite, preserving separation between colors
#   visual_observability:
#     - generator rejects samples where the old vertical-axis solver would produce the same output
#     - generator also penalizes clean sprites that are too vertically symmetric, so examples visibly favor diagonal / rotational structure
#   visual_tidiness:
# 
from __future__ import annotations
#     - large slot gaps from the source task are preserved
#     - only one object per color is used, and noise remains sparse and local to the object it corrupts
#   color_reasoning:
#     - colors are unchanged; color id never controls ordering or transformation
#   solver_assumptions_to_guarantee:
#     - one sprite per non-background color
#     - the true rotation center scores strictly above alternatives by a small gap
#     - noise pixels never appear in 180-degree pairs, so deletion is necessary

import math
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import colors_present, get_bg, paste


def _rotation_score_map(coords: list[tuple[int, int]]) -> Counter[tuple[int, int]]:
    """Score every 180-degree center by counting ordered point pairs with the same midpoint."""
    counts: Counter[tuple[int, int]] = Counter()
    pts = [(int(r), int(c)) for r, c in coords]
    for r1, c1 in pts:
        for r2, c2 in pts:
            counts[(r1 + r2, c1 + c2)] += 1
    return counts


def _axis_score_map(coords: list[tuple[int, int]]) -> Counter[int]:
    """Score every vertical axis by counting ordered same-row point pairs."""
    counts: Counter[int] = Counter()
    pts = [(int(r), int(c)) for r, c in coords]
    for r1, c1 in pts:
        for r2, c2 in pts:
            if r1 == r2:
                counts[c1 + c2] += 1
    return counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Clean each color layer by enforcing 180-degree rotational symmetry.

    Input structure:
    - bg = background. Several non-bg colors each form a nearly point-symmetric sprite, but every color may have its own center.
    - Some same-color pixels are extra attachments whose rotated counterparts are missing.

    Rule:
    - For each non-bg color, search for the rotation center that best matches that color's pixel set.
    - Remove any pixel that does not have a same-color partner after a 180-degree turn around that chosen center.

    Key insight: The denoising is still independent per color, but the hidden symmetry parameter is now a 2D center rather than a 1D vertical axis.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    out = g.copy()
    colors = colors_present(g, exclude={bg})
    assert colors, "grid must contain at least one non-background color"

    for col in colors:
        pts = np.argwhere(g == int(col))
        if pts.size == 0:
            continue

        coords = [tuple(x) for x in pts]
        best_center = max(_rotation_score_map(coords).items(), key=lambda item: item[1])[0]

        kr, kc = best_center
        for r, c in pts:
            rr = int(kr) - int(r)
            cc = int(kc) - int(c)
            if not (0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(col)):
                out[int(r), int(c)] = int(bg)

    return out


# --------------- generator ---------------

# ── CFG: canvas & layout ──
GRID_SIZE = 22
NUM_OBJECTS = 4
LAYOUTS = [
    [(0, 0, 7, 7), (0, 14, 7, 21), (14, 5, 21, 16)],
    [(0, 0, 8, 5), (0, 16, 8, 21), (15, 5, 21, 16)],
]

# ── CFG: clean sprite geometry ──
PIXEL_COUNT_MIN = 6
PIXEL_COUNT_MAX = 46
HEIGHT_MIN = 2
HEIGHT_MAX = 9
WIDTH_MIN = 2
WIDTH_MAX = 12
DENSITY_LO = 0.35
DENSITY_HI = 0.89

# ── CFG: noise ──
NOISE_FRAC_LO = 0.11
NOISE_FRAC_HI = 0.33
NOISE_MAX = 12

# ── CFG: uniqueness / anti-degeneration ──
SPRITE_CENTER_GAP_MIN = 3
FINAL_CENTER_GAP_MIN = 2
CLEAN_VERTICAL_GAP_MIN = 2
MIN_VARIANT_DIFF = 3

# ── CFG: retry limits ──
SPRITE_RETRIES = 180
GEN_RETRIES = 400

# ── CFG: augmentation ──
FLIP_PROB = 0.5


def _second_best_rotation_score(
    coords: list[tuple[int, int]],
    true_center: tuple[int, int],
    height: int,
    width: int,
) -> int:
    """Return the highest rotation score excluding the true center."""
    del height, width
    scores = _rotation_score_map(coords)
    scores.pop(true_center, None)
    return max(scores.values(), default=0)


def _best_vertical_axis_score(coords: list[tuple[int, int]], width: int) -> int:
    del width
    return max(_axis_score_map(coords).values(), default=0)


def _feasible_shape_options(max_h: int, max_w: int) -> list[tuple[int, int, int, int]]:
    """Enumerate feasible (h, w, target_count, center_count) tuples for point-symmetric sprites."""
    options: list[tuple[int, int, int, int]] = []
    h_max = min(HEIGHT_MAX, max_h)
    w_max = min(WIDTH_MAX, max_w)
    for target_count in range(PIXEL_COUNT_MIN, PIXEL_COUNT_MAX + 1):
        for h in range(HEIGHT_MIN, h_max + 1):
            for w in range(WIDTH_MIN, w_max + 1):
                area = h * w
                if not (math.ceil(DENSITY_LO * area) <= target_count <= math.floor(DENSITY_HI * area)):
                    continue

                center_options = [0]
                if h % 2 == 1 and w % 2 == 1:
                    center_options = [0, 1]

                for center_count in center_options:
                    remaining = target_count - center_count
                    if remaining < 0 or remaining % 2 != 0:
                        continue
                    pair_count = remaining // 2
                    max_pairs = (area - center_count) // 2
                    if 1 <= pair_count <= max_pairs:
                        options.append((h, w, target_count, center_count))
    return options


def _choose_shape_spec(max_h: int, max_w: int) -> tuple[int, int, int, int]:
    """Pick a random clean sprite spec, biased slightly toward denser boxes."""
    options = _feasible_shape_options(max_h, max_w)
    if not options:
        raise ValueError("no feasible clean sprite spec")
    counts = sorted({target_count for _, _, target_count, _ in options})
    target_count = random.choice(counts)
    target_options = [opt for opt in options if opt[2] == target_count]
    weights = [1.0 / (h * w) for h, w, _, _ in target_options]
    return random.choices(target_options, weights=weights, k=1)[0]


def _rotation_orbits(h: int, w: int) -> list[tuple[tuple[int, int], ...]]:
    """List all cell orbits under 180-degree rotation inside an h x w bbox."""
    orbits: list[tuple[tuple[int, int], ...]] = []
    seen: set[tuple[int, int]] = set()
    for r in range(h):
        for c in range(w):
            if (r, c) in seen:
                continue
            rr, cc = h - 1 - r, w - 1 - c
            if (r, c) == (rr, cc):
                orbit = ((r, c),)
            else:
                orbit = ((r, c), (rr, cc))
            for cell in orbit:
                seen.add(cell)
            orbits.append(orbit)
    return orbits


def _touch_count(orbit: tuple[tuple[int, int], ...], occupied: set[tuple[int, int]]) -> int:
    """How many 4-neighbor contacts this orbit has with the current shape."""
    total = 0
    for r, c in orbit:
        total += int((r - 1, c) in occupied)
        total += int((r + 1, c) in occupied)
        total += int((r, c - 1) in occupied)
        total += int((r, c + 1) in occupied)
    return total


def _build_clean_sprite(color: int, max_h: int, max_w: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Build a clean point-symmetric sprite and return (sprite, local_center)."""
    for _ in range(SPRITE_RETRIES):
        h, w, target_count, center_count = _choose_shape_spec(max_h, max_w)
        pair_count = (target_count - center_count) // 2
        orbits = _rotation_orbits(h, w)
        pair_indices = [i for i, orbit in enumerate(orbits) if len(orbit) == 2]
        if len(pair_indices) < pair_count:
            continue

        top_candidates = [i for i in pair_indices if any(r == 0 for r, _ in orbits[i])]
        left_candidates = [i for i in pair_indices if any(c == 0 for _, c in orbits[i])]
        if not top_candidates or not left_candidates:
            continue

        selected: set[int] = {random.choice(top_candidates), random.choice(left_candidates)}
        if len(selected) > pair_count:
            continue

        occupied: set[tuple[int, int]] = set()
        for idx in selected:
            occupied.update(orbits[idx])

        while len(selected) < pair_count:
            remaining = [i for i in pair_indices if i not in selected]
            if not remaining:
                break
            frontier = [i for i in remaining if _touch_count(orbits[i], occupied) > 0]
            pool = frontier if frontier else remaining
            weights = [1 + _touch_count(orbits[i], occupied) for i in pool]
            idx = random.choices(pool, weights=weights, k=1)[0]
            selected.add(idx)
            occupied.update(orbits[idx])

        if len(selected) != pair_count:
            continue

        sprite = np.zeros((h, w), dtype=int)
        for idx in selected:
            for r, c in orbits[idx]:
                sprite[r, c] = color
        if center_count == 1:
            sprite[h // 2, w // 2] = color

        coords = [tuple(x) for x in np.argwhere(sprite == color)]
        if len(coords) != target_count:
            continue
        if coords:
            rows = [r for r, _ in coords]
            cols = [c for _, c in coords]
            if min(rows) != 0 or max(rows) != h - 1 or min(cols) != 0 or max(cols) != w - 1:
                continue

        true_center = (h - 1, w - 1)
        if _second_best_rotation_score(coords, true_center, h, w) > target_count - SPRITE_CENTER_GAP_MIN:
            continue
        if _best_vertical_axis_score(coords, w) > target_count - CLEAN_VERTICAL_GAP_MIN:
            continue
        return sprite, true_center
    raise ValueError("failed to build clean sprite")


def _slot_place_sprites(
    bg: int,
    sprite_specs: list[dict],
    slots: list[tuple[int, int, int, int]],
) -> tuple[np.ndarray, list[dict]] | None:
    """Place clean sprites into fixed non-overlapping slots."""
    out = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
    placed: list[dict] = []

    for spec, (r0s, c0s, r1s, c1s) in zip(sprite_specs, slots):
        spec = dict(spec)
        sprite = spec["sprite"]
        h, w = sprite.shape
        if h > r1s - r0s + 1 or w > c1s - c0s + 1:
            return None
        r0 = random.randint(r0s, r1s - h + 1)
        c0 = random.randint(c0s, c1s - w + 1)
        paste(out, sprite, r0, c0, transparent=0)
        spec["r0"] = r0
        spec["c0"] = c0
        local_kr, local_kc = spec["center_local"]
        spec["true_center"] = (local_kr + 2 * r0, local_kc + 2 * c0)
        placed.append(spec)
    return out, placed


def _noise_candidates(
    clean_mask: np.ndarray,
    true_center: tuple[int, int],
) -> list[tuple[int, int]]:
    """Return valid local noise cells: adjacent to the shape and unmatched under 180-degree rotation."""
    H, W = clean_mask.shape
    adj = np.zeros_like(clean_mask)
    adj[1:, :] |= clean_mask[:-1, :]
    adj[:-1, :] |= clean_mask[1:, :]
    adj[:, 1:] |= clean_mask[:, :-1]
    adj[:, :-1] |= clean_mask[:, 1:]
    adj &= ~clean_mask

    kr, kc = true_center
    rows, cols = np.where(adj)
    candidates: list[tuple[int, int]] = []
    for r, c in zip(rows.tolist(), cols.tolist()):
        rr = kr - r
        cc = kc - c
        if rr == r and cc == c:
            continue
        if 0 <= rr < H and 0 <= cc < W and clean_mask[rr, cc]:
            continue
        candidates.append((r, c))
    return candidates


def _add_noise(inp: np.ndarray, out: np.ndarray, spec: dict) -> bool:
    """Add unmatched rotational noise to one placed sprite."""
    color = int(spec["color"])
    true_center = tuple(spec["true_center"])
    clean_coords = [tuple(x) for x in np.argwhere(out == color)]
    clean_mask = out == color
    clean_count = len(clean_coords)

    lo = max(1, math.ceil(NOISE_FRAC_LO * clean_count))
    hi = min(NOISE_MAX, math.floor(NOISE_FRAC_HI * clean_count))
    if lo > hi:
        return False

    noise_target = random.randint(lo, hi)
    candidates = _noise_candidates(clean_mask, true_center)
    if len(candidates) < noise_target:
        return False

    chosen: list[tuple[int, int]] = []
    forbidden: set[tuple[int, int]] = set()
    current_coords = list(clean_coords)
    for _ in range(noise_target):
        viable = [cell for cell in candidates if cell not in forbidden and cell not in chosen]
        if not viable:
            return False

        scored = []
        for cell in viable:
            alt = _second_best_rotation_score(current_coords + [cell], true_center, GRID_SIZE, GRID_SIZE)
            scored.append((alt, random.random(), cell))
        scored.sort(key=lambda item: (item[0], item[1]))
        best_alt = scored[0][0]
        pool = [cell for alt, _, cell in scored if alt <= best_alt + 1]
        cell = random.choice(pool)
        chosen.append(cell)
        current_coords.append(cell)
        forbidden.add(cell)
        rr = true_center[0] - cell[0]
        cc = true_center[1] - cell[1]
        if 0 <= rr < GRID_SIZE and 0 <= cc < GRID_SIZE:
            forbidden.add((rr, cc))

    if clean_count - _second_best_rotation_score(current_coords, true_center, GRID_SIZE, GRID_SIZE) < FINAL_CENTER_GAP_MIN:
        return False

    for r, c in chosen:
        inp[r, c] = color
    return True


def _vertical_baseline(grid: np.ndarray) -> np.ndarray:
    """Original task behavior, kept only as a rejection test to prevent degeneration."""
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    _, W = g.shape
    out = g.copy()

    for col in colors_present(g, exclude={bg}):
        pts = np.argwhere(g == int(col))
        if pts.size == 0:
            continue
        coords = [tuple(x) for x in pts]
        best_k = max(_axis_score_map(coords).items(), key=lambda item: item[1])[0]

        for r, c in pts:
            cc = best_k - int(c)
            if not (0 <= cc < W and int(g[int(r), cc]) == int(col)):
                out[int(r), int(c)] = bg
    return out


def _gen_attempt() -> np.ndarray | None:
    """Single output-first generation attempt."""
    bg, *obj_colors = random.sample(range(10), NUM_OBJECTS)
    slots = random.choice(LAYOUTS)

    sprite_specs = []
    try:
        for color, slot in zip(obj_colors, slots):
            max_h = slot[2] - slot[0] + 1
            max_w = slot[3] - slot[1] + 1
            sprite, center_local = _build_clean_sprite(color, max_h, max_w)
            sprite_specs.append({"color": color, "sprite": sprite, "center_local": center_local})
    except ValueError:
        return None

    result = _slot_place_sprites(bg, sprite_specs, slots)
    if result is None:
        return None
    out, placed_specs = result

    inp = out.copy()
    for spec in placed_specs:
        if not _add_noise(inp, out, spec):
            return None

    if random.random() < FLIP_PROB:
        inp = np.flipud(inp)
        out = np.flipud(out)
    if random.random() < FLIP_PROB:
        inp = np.fliplr(inp)
        out = np.fliplr(out)

    solved = main(inp)
    if not np.array_equal(solved, out):
        return None

    baseline = _vertical_baseline(inp)
    if np.array_equal(baseline, out):
        return None
    if int(np.count_nonzero(baseline != out)) < MIN_VARIANT_DIFF:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
