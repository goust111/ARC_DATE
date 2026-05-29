# Combo variant of 8e5c0c38 (index 19)
# combo_spec: D5/3+D14/0
# Name: 空间模式决定轴位置

from __future__ import annotations

import itertools
import math
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT / "solvers") not in sys.path:
    sys.path.insert(0, str(ROOT / "solvers"))

from arc_utils import colors_present, get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior trace:
#    Base task: each color independently searches the full grid for the axis that
#    best matches its own pixels, then removes unmatched pixels. In this combo,
#    that assumption is intentionally broken. The three sprites now sit in one of
#    a few rigid geometric layouts, and each layout defines three hidden "slots".
#    Each sprite's correct axis is the center line of its assigned slot, not the
#    axis that best fits that sprite alone. Example: in the diagonal layout, the
#    slot centers are at columns 6, 11, 16. A blue sprite in the left slot may
#    get several extra pixels chosen so that, by itself, it scores better around
#    column 5.5. The original solver would keep those misleading pixels or delete
#    genuine ones. The combo solver first recognizes the diagonal arrangement of
#    the three bboxes, assigns the blue sprite to the left diagonal slot, fixes
#    its axis at column 6, and only then removes pixels that fail that slot axis.
# b) Complexity calibration:
#    The layout vocabulary is deliberately small: three templates only (horizontal
#    row, descending diagonal, triangle). That keeps the pattern human-visible in
#    2-4 examples while still making the spatial relation matter. Sprite shapes
#    remain simple vertical-symmetry denoising targets, but they are generated to
#    have a small axis gap, so the extra layout cue is actually needed.
# c) Broken original assumptions and line-by-line audit for main():
#    - The original loop over `k` assumed each color's pixels contain enough local
#      evidence to reveal their own axis. That no longer holds; noise can make a
#      nearby false axis win.
#    - `pts = np.argwhere(g == col)` still assumes one logical sprite per color.
#      Generator preserves one connected-ish sprite per distinct color.
#    - Using the input grid as mirror evidence is still valid, but now the axis
#      comes from the inferred slot assignment rather than from a per-color search.
#    - Slot inference assumes object bboxes remain readable under noise. Generator
#      guarantees that by using fixed 7x7 slots with a centered 7x5 clean core,
#      then keeping all noise inside the owning slot,
#      owning slot, so each noisy bbox still identifies the same spatial template.
# d) Placement legality:
#    Generation is output-first. Clean sprites are drawn inside fixed, non-overlap
#    slot boxes, then misleading pixels are added only inside those same boxes and
#    only at background cells. This guarantees in-bounds placement, no collisions,
#    and a one-to-one mapping from noisy sprite bbox to latent slot.
# e) Visual observability:
#    The layout itself is a visible clue: examples repeatedly show three colored
#    sprites arranged as an even row, a diagonal, or a triangle. The output keeps
#    exactly the same placement but prunes asymmetric protrusions according to the
#    slot-centered mirror lines, so a human can infer that geometry determines the
#    axis rather than each sprite being solved in isolation.
# f) Visual cleanliness:
#    Only three sprites appear, each inside its own reserved box. No markers or
#    auxiliary colors are introduced. Noise stays attached to the clean shape, so
#    the grid remains legible rather than becoming scattered speckle.
# g) Inferability and non-degeneracy:
#    There is no color remapping or hidden ordering by color id. The only control
#    variable is spatial role within the repeated layout templates. Generator also
#    rejects samples where the old independent-axis solver would already produce
#    the same answer, so the fusion behavior is not a silent no-op.
# h) New solver assumptions checked against generator:
#    1. Exactly three non-background colors exist. Guaranteed by construction.
#    2. Each color occupies one unique slot-centered region. Guaranteed because every
#       clean sprite spans the full 7x5 core centered in a 7x7 slot, and all noise
#       stays inside that same slot.
#    3. The nearest-template assignment is unambiguous. Guaranteed by using three
#       well-separated templates whose slot centers never coincide.


GRID_SIZE = 22
NUM_SPRITES = 3
SLOT_H = 7
SLOT_W = 7
CLEAN_W = 5
CLEAN_X0 = 1
LOCAL_TRUE_K = CLEAN_W - 1  # clean core axis at column 2 -> k = 4

PIXEL_COUNT_MIN = 18
PIXEL_COUNT_MAX = 22
DENSITY_LO = 0.35
DENSITY_HI = 0.78

CLEAN_AXIS_GAP_MIN = 8
CLEAN_AXIS_GAP_MAX = 8
NOISE_FRAC_LO = 0.24
NOISE_FRAC_HI = 0.60
NOISE_MAX = 11

SPRITE_RETRIES = 400
NOISE_RETRIES = 120
GEN_RETRIES = 500

PATTERNS = [
    {
        "name": "line",
        "slots": [(1, 0, 7, 6), (1, 8, 7, 14), (1, 15, 7, 21)],
    },
    {
        "name": "diag",
        "slots": [(1, 1, 7, 7), (8, 8, 14, 14), (15, 14, 21, 20)],
    },
    {
        "name": "triangle",
        "slots": [(1, 7, 7, 13), (14, 0, 20, 6), (14, 15, 20, 21)],
    },
]


def _slot_center(slot: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = slot
    return ((r0 + r1) // 2, (c0 + c1) // 2)


for pattern in PATTERNS:
    pattern["centers"] = [_slot_center(slot) for slot in pattern["slots"]]


def _axis_score(coords: list[tuple[int, int]], k: int, width: int) -> int:
    pts = {(int(r), int(c)) for r, c in coords}
    score = 0
    for r, c in pts:
        cc = int(k) - int(c)
        if 0 <= cc < width and (int(r), cc) in pts:
            score += 1
    return score


def _axis_scores(coords: list[tuple[int, int]], width: int) -> dict[int, int]:
    return {k: _axis_score(coords, k, width) for k in range(2 * (width - 1) + 1)}


def _best_axis_k(coords: list[tuple[int, int]], width: int) -> tuple[int, int]:
    scores = _axis_scores(coords, width)
    best_k = max(scores, key=lambda k: (scores[k], -abs(k - (width - 1))))
    return int(best_k), int(scores[best_k])


def _second_best_score(coords: list[tuple[int, int]], true_k: int, width: int) -> int:
    scores = _axis_scores(coords, width)
    return max(score for k, score in scores.items() if k != int(true_k))


def _bbox_center_of_color(grid: np.ndarray, color: int) -> tuple[int, int]:
    pts = np.argwhere(grid == int(color))
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return (int((r0 + r1) // 2), int((c0 + c1) // 2))


def _infer_pattern_assignment(grid: np.ndarray, colors: list[int]) -> dict[int, tuple[int, int, int, int]]:
    objects = [(int(color), _bbox_center_of_color(grid, int(color))) for color in colors]
    best_score = None
    best_assignment = None

    for pattern in PATTERNS:
        for perm in itertools.permutations(range(NUM_SPRITES)):
            total = 0
            assignment: dict[int, tuple[int, int, int, int]] = {}
            for obj_idx, slot_idx in enumerate(perm):
                color, (rr, cc) = objects[obj_idx]
                sr, sc = pattern["centers"][slot_idx]
                total += abs(rr - sr) + abs(cc - sc)
                assignment[color] = pattern["slots"][slot_idx]
            if best_score is None or total < best_score:
                best_score = total
                best_assignment = assignment

    assert best_assignment is not None
    return best_assignment


def _independent_clean(grid: np.ndarray) -> np.ndarray:
    """Original behavior: choose the best axis per color from the sprite itself."""
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    out = g.copy()
    for color in colors_present(g, exclude={bg}):
        pts = [tuple(x) for x in np.argwhere(g == int(color))]
        if not pts:
            continue
        best_k, _ = _best_axis_k(pts, g.shape[1])
        for r, c in pts:
            cc = int(best_k) - int(c)
            if not (0 <= cc < g.shape[1] and int(g[int(r), cc]) == int(color)):
                out[int(r), int(c)] = bg
    return out


def _layout_axis_map(grid: np.ndarray) -> dict[int, int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    colors = colors_present(g, exclude={bg})
    assert len(colors) == NUM_SPRITES, "grid must contain exactly three non-background colors"
    assignment = _infer_pattern_assignment(g, colors)
    return {int(color): int(slot[1] + slot[3]) for color, slot in assignment.items()}


def fusion_metrics(grid: np.ndarray) -> dict[str, int]:
    """
    Quantify whether the sample really uses the fused rule.

    Returns counts of colors whose layout axis differs from the old best-fit axis,
    plus whether the whole old solver output disagrees with the fused solver.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    colors = colors_present(g, exclude={bg})
    layout_axes = _layout_axis_map(g)
    differing = 0
    for color in colors:
        pts = [tuple(x) for x in np.argwhere(g == int(color))]
        best_k, _ = _best_axis_k(pts, g.shape[1])
        if int(best_k) != int(layout_axes[int(color)]):
            differing += 1
    old_out = _independent_clean(g)
    new_out = main(g)
    return {
        "colors_with_different_axis": int(differing),
        "old_solver_differs": int(not np.array_equal(old_out, new_out)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove asymmetric noise, but use the global spatial template to decide each sprite's axis.

    Input structure:
    - The grid contains exactly three non-background colored sprites arranged as a visible row,
      diagonal, or triangle template.
    - Each color occupies one slot in that template and is almost vertically symmetric, except for
      a few extra pixels that can make the sprite itself prefer the wrong nearby axis.

    Rule:
    - Infer which slot each colored sprite belongs to from the overall three-object layout.
    - For each color, use that slot's center column as the vertical mirror axis and delete any
      same-color pixel whose reflected partner across that slot axis is absent.

    Key insight:
    - Axis finding is no longer an independent per-sprite search problem. The placement pattern of
      the three sprites reveals hidden slot centers, and those slot centers override whatever axis
      would look best from a single noisy sprite alone.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    out = g.copy()
    axis_map = _layout_axis_map(g)

    for color, k in axis_map.items():
        pts = np.argwhere(g == int(color))
        for r, c in pts:
            rr = int(r)
            cc = int(k) - int(c)
            if not (0 <= cc < g.shape[1] and int(g[rr, cc]) == int(color)):
                out[rr, int(c)] = bg
    return out


def _left_cells(left_count: int) -> set[tuple[int, int]]:
    """Pick left-half cells on a 7x2 grid, forcing full-height/full-width bbox."""
    row_choices = {r: random.randrange(2) for r in range(SLOT_H)}
    row_choices[0] = 0
    row_choices[random.randrange(1, SLOT_H)] = 1
    cells: set[tuple[int, int]] = {(r, c) for r, c in row_choices.items()}

    remaining = left_count - len(cells)
    if remaining < 0:
        raise ValueError("left_count too small")

    pool = [(r, c) for r in range(SLOT_H) for c in range(2) if (r, c) not in cells]
    if remaining > len(pool):
        raise ValueError("left_count too large")
    cells.update(random.sample(pool, remaining))
    return cells


def _center_rows(center_count: int) -> set[int]:
    if center_count == 0:
        return set()
    return set(random.sample(range(SLOT_H), center_count))


def _reference_noise_feasible(sprite: np.ndarray) -> bool:
    slot = PATTERNS[0]["slots"][0]
    true_k = int(slot[1] + slot[3])
    probe = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    paste(probe, sprite, slot[0], slot[1] + CLEAN_X0, transparent=0)
    clean_mask = probe != 0
    clean_coords = [tuple(x) for x in np.argwhere(sprite != 0)]
    clean_count = len(clean_coords)
    hi = min(NOISE_MAX, math.floor(NOISE_FRAC_HI * clean_count))
    if hi <= 0:
        return False

    local_scores = _axis_scores(clean_coords, CLEAN_W)
    for local_k, score in sorted(local_scores.items(), key=lambda item: item[1], reverse=True):
        if local_k == LOCAL_TRUE_K:
            continue
        false_k = int(local_k + 2 * (slot[1] + CLEAN_X0))
        required = clean_count - score + 1
        singles = _noise_candidates(clean_mask, slot, true_k, false_k)
        pairs = _noise_pairs(clean_mask, slot, true_k, false_k)
        if required <= hi and len(singles) + 2 * len(pairs) >= required:
            return True
    return False


def _reference_noise_realizable(sprite: np.ndarray) -> bool:
    slot = PATTERNS[0]["slots"][0]
    color = int(sprite.max())
    for _ in range(3):
        out = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        paste(out, sprite, slot[0], slot[1] + CLEAN_X0, transparent=0)
        inp = out.copy()
        spec = {"color": color, "slot": slot, "true_k": int(slot[1] + slot[3])}
        if _add_misleading_noise(inp, out, spec):
            return True
    return False


def _build_clean_sprite(color: int) -> np.ndarray:
    """Build a symmetric sprite whose nearby false axes score almost as well."""
    area = SLOT_H * CLEAN_W
    for _ in range(SPRITE_RETRIES):
        target_count = random.randint(PIXEL_COUNT_MIN, PIXEL_COUNT_MAX)
        if not (math.ceil(DENSITY_LO * area) <= target_count <= math.floor(DENSITY_HI * area)):
            continue

        possible_centers = [
            center_count
            for center_count in range(1, SLOT_H + 1)
            if (target_count - center_count) % 2 == 0
        ]
        if not possible_centers:
            continue
        center_count = random.choice(possible_centers)
        left_count = (target_count - center_count) // 2
        if not (SLOT_H <= left_count <= SLOT_H * 2):
            continue

        left = _left_cells(left_count)
        centers = _center_rows(center_count)
        sprite = np.zeros((SLOT_H, CLEAN_W), dtype=int)
        for r, c in left:
            sprite[r, c] = int(color)
            sprite[r, CLEAN_W - 1 - c] = int(color)
        for r in centers:
            sprite[r, CLEAN_W // 2] = int(color)

        coords = [tuple(x) for x in np.argwhere(sprite == int(color))]
        if len(coords) != target_count:
            continue
        if sprite[:, 0].sum() == 0 or sprite[:, -1].sum() == 0:
            continue
        gap = len(coords) - _second_best_score(coords, LOCAL_TRUE_K, CLEAN_W)
        if not (CLEAN_AXIS_GAP_MIN <= gap <= CLEAN_AXIS_GAP_MAX):
            continue
        if not _reference_noise_feasible(sprite):
            continue
        if not _reference_noise_realizable(sprite):
            continue
        return sprite
    raise ValueError("failed to build ambiguous clean sprite")


def _slot_neighbors(clean_mask: np.ndarray, slot: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = slot
    out: set[tuple[int, int]] = set()
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if clean_mask[r, c]:
                continue
            if (
                (r > r0 and clean_mask[r - 1, c])
                or (r < r1 and clean_mask[r + 1, c])
                or (c > c0 and clean_mask[r, c - 1])
                or (c < c1 and clean_mask[r, c + 1])
            ):
                out.add((r, c))
    return out


def _noise_candidates(
    clean_mask: np.ndarray,
    slot: tuple[int, int, int, int],
    true_k: int,
    false_k: int,
) -> list[tuple[int, int]]:
    candidates = []
    for r, c in _slot_neighbors(clean_mask, slot):
        mirror_true = true_k - c
        mirror_false = false_k - c
        if mirror_true == c or mirror_false == c:
            continue
        if not (0 <= mirror_true < GRID_SIZE and 0 <= mirror_false < GRID_SIZE):
            continue
        if clean_mask[r, mirror_true]:
            continue
        if not clean_mask[r, mirror_false]:
            continue
        candidates.append((int(r), int(c)))
    return candidates


def _noise_pairs(
    clean_mask: np.ndarray,
    slot: tuple[int, int, int, int],
    true_k: int,
    false_k: int,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    neighbors = _slot_neighbors(clean_mask, slot)
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r, c in neighbors:
        c2 = false_k - c
        mate = (r, c2)
        if not (0 <= c2 < GRID_SIZE):
            continue
        if mate not in neighbors or c >= c2:
            continue
        mirror_a = true_k - c
        mirror_b = true_k - c2
        if mirror_a == c or mirror_b == c2:
            continue
        if mirror_a == c2 or mirror_b == c:
            continue
        if not (0 <= mirror_a < GRID_SIZE and 0 <= mirror_b < GRID_SIZE):
            continue
        if clean_mask[r, mirror_a] or clean_mask[r, mirror_b]:
            continue
        pairs.append(((int(r), int(c)), (int(r), int(c2))))
    return pairs


def _can_add_cells(
    chosen: list[tuple[int, int]],
    used: set[tuple[int, int]],
    cells: list[tuple[int, int]],
    true_k: int,
) -> bool:
    for r, c in cells:
        if (r, c) in used or (r, true_k - c) in used:
            return False
    return True


def _register_cells(
    chosen: list[tuple[int, int]],
    used: set[tuple[int, int]],
    cells: list[tuple[int, int]],
    true_k: int,
) -> None:
    for r, c in cells:
        chosen.append((int(r), int(c)))
        used.add((int(r), int(c)))
        used.add((int(r), int(true_k - c)))


def _add_misleading_noise(inp: np.ndarray, out: np.ndarray, spec: dict) -> bool:
    color = int(spec["color"])
    slot = spec["slot"]
    true_k = int(spec["true_k"])
    clean_coords = [tuple(x) for x in np.argwhere(out == color)]
    clean_mask = out == color
    clean_count = len(clean_coords)

    lo = max(2, math.ceil(NOISE_FRAC_LO * clean_count))
    hi = min(NOISE_MAX, math.floor(NOISE_FRAC_HI * clean_count))
    if lo > hi:
        return False

    clean_scores = _axis_scores(clean_coords, GRID_SIZE)
    false_axes = [
        k
        for k, _score in sorted(clean_scores.items(), key=lambda item: item[1], reverse=True)
        if k != true_k
    ]

    for false_k in false_axes:
        required = clean_count - clean_scores[false_k] + 1
        singles = _noise_candidates(clean_mask, slot, true_k, false_k)
        pairs = _noise_pairs(clean_mask, slot, true_k, false_k)
        max_gain = len(singles) + 2 * len(pairs)
        if max_gain < required:
            continue

        pair_indices = list(range(len(pairs)))
        subset_masks = list(range(1 << len(pair_indices)))
        random.shuffle(subset_masks)

        for mask in subset_masks[:NOISE_RETRIES]:
            chosen: list[tuple[int, int]] = []
            used: set[tuple[int, int]] = set()
            pair_gain = 0
            valid_subset = True

            for idx in pair_indices:
                if not (mask & (1 << idx)):
                    continue
                cells = [pairs[idx][0], pairs[idx][1]]
                if len(chosen) + 2 > hi or not _can_add_cells(chosen, used, cells, true_k):
                    valid_subset = False
                    break
                _register_cells(chosen, used, cells, true_k)
                pair_gain += 2

            if not valid_subset:
                continue

            available_singles = [
                cell for cell in singles if _can_add_cells(chosen, used, [cell], true_k)
            ]
            min_single_take = max(0, required - pair_gain, lo - len(chosen))
            max_single_take = min(len(available_singles), hi - len(chosen))
            if min_single_take > max_single_take:
                continue

            take_options = list(range(min_single_take, max_single_take + 1))
            random.shuffle(take_options)
            for take in take_options:
                if take == 0:
                    trial_single_sets = [()]
                else:
                    trial_single_sets = []
                    for _ in range(min(24, NOISE_RETRIES)):
                        trial_single_sets.append(tuple(random.sample(available_singles, take)))
                for single_set in trial_single_sets:
                    noisy_coords = clean_coords + chosen + list(single_set)
                    best_k, _ = _best_axis_k(noisy_coords, GRID_SIZE)
                    if best_k != false_k:
                        continue
                    for r, c in chosen:
                        inp[r, c] = color
                    for r, c in single_set:
                        inp[r, c] = color
                    spec["misleading_k"] = int(false_k)
                    return True
    return False


def _build_scene(bg: int, colors: list[int], pattern: dict) -> tuple[np.ndarray, list[dict]]:
    out = np.full((GRID_SIZE, GRID_SIZE), int(bg), dtype=int)
    specs: list[dict] = []
    for color, slot in zip(colors, pattern["slots"]):
        sprite = _build_clean_sprite(int(color))
        r0, c0, _r1, _c1 = slot
        paste(out, sprite, r0, c0 + CLEAN_X0, transparent=0)
        specs.append(
            {
                "color": int(color),
                "slot": slot,
                "true_k": int(slot[1] + slot[3]),
            }
        )
    return out, specs


def _gen_attempt() -> np.ndarray | None:
    bg = random.randrange(10)
    obj_pool = [color for color in range(1, 10) if color != bg]
    obj_colors = random.sample(obj_pool, NUM_SPRITES)
    pattern = random.choice(PATTERNS)

    try:
        out, specs = _build_scene(bg, obj_colors, pattern)
    except ValueError:
        return None

    inp = out.copy()
    misleading = 0
    for spec in specs:
        if not _add_misleading_noise(inp, out, spec):
            return None
        if spec.get("misleading_k") != spec["true_k"]:
            misleading += 1

    if misleading == 0:
        return None
    if not np.array_equal(main(inp), out):
        return None
    if np.array_equal(_independent_clean(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
