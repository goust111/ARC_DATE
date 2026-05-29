# Variant of 269e22fb
# Dimension: D5 — 图案结构：基础图案类型 (meta)
# Change: 硬编码的20x20固定二值图案，前景密度42.2%，8个二面体变体均不同且无互补关系... → 程序化生成的对称二值图案

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present


# Design analysis for the D5 variant
# ----------------------------------
# spec:
#   core_change:
#     original: one hard-coded 20x20 binary motif; solver wins by matching the crop
#       against all offsets of 8 stored dihedral variants.
#     variant: a whole family of 20x20 motifs generated from a 5x5 binary seed.
#       The full motif has C4 rotational symmetry, so the dihedral orbit shrinks:
#       all rotations coincide, while reflections form one second distinct class.
#   concrete_shape:
#     seed: a 5x5 boolean matrix.
#     quadrant_10x10:
#       built from 4 transformed seed copies:
#         [[seed, rot90(seed)],
#          [fliplr(seed), rot90(fliplr(seed))]]
#     full_20x20:
#       built from 4 rotated copies of the 10x10 quadrant:
#         [[Q, rot270(Q)],
#          [rot90(Q), rot180(Q)]]
#       This guarantees 4-fold rotational symmetry of the whole 20x20 pattern.
#   behavior_trace:
#     original_flow:
#       crop -> try both polarities -> compare with every fixed full-pattern window.
#     new_flow:
#       crop -> try both polarities and both unique dihedral classes ->
#       for each candidate window, map visible cells back to seed labels 0..24 ->
#       if the crop assigns all 25 labels consistently, reconstruct the seed ->
#       regenerate the full 20x20 motif from that seed and repaint it.
#     difference:
#       matching is no longer against one memorized bitmap; it is against a
#       structural label map that lets the solver infer the hidden seed.
#   complexity_calibration:
#     - If the crop were smaller than 10 in either dimension, some windows would
#       miss part of the 25-label seed support, leaving multiple legal completions.
#     - Therefore generator uses H in [10,16], W in [10,12], which still looks
#       like the original rectangular crop task while guaranteeing full seed
#       coverage for every offset.
#     - To avoid the family collapsing under complement, a few visible seed cells
#       are anchored to fixed 0/1 values. This minimally changes the family while
#       preserving random-looking interiors.
#   implicit_assumptions_review:
#     - original `base = _pattern_mask()` assumed one immutable motif; invalid now.
#     - original 8-way dihedral enumeration assumed all variants can differ; now
#       C4 symmetry collapses them to 2 unique full-grid classes, so dedup is needed.
#     - original polarity loop was safe because the complement was not a legal base
#       pattern; now complement is legal unless the generator constrains the seed.
#     - original solver never had to infer latent state; now successful recovery
#       assumes the crop covers all 25 seed labels and the seed has no extra
#       dihedral symmetry. Generator enforces both.
#   placement_legality:
#     - The crop is still a simple axis-aligned rectangle cut from a 20x20 output.
#     - No object placement interactions exist, so legality reduces to staying
#       inside bounds and choosing windows whose reconstruction is unique.
#   visual_observability:
#     - Across examples, the output is visibly not the same bitmap anymore:
#       each task instance has a different globally symmetric motif.
#     - The strong quarter-turn symmetry is human-visible, and the crop/output
#       relation remains the same "complete the whole motif" concept as original.
#   cleanliness:
#     - Only two colors remain in the input/output, so the variant stays visually
#       tidy. The extra structure lives in binary geometry, not added markers.
#   new_main_assumptions_checked_against_generator:
#     - main assumes exactly 2 colors: generator always paints with 2 colors.
#     - main assumes a 10x10+ crop can determine all 25 seed cells: generator
#       restricts window sizes accordingly.
#     - main assumes inferred seed satisfies anchor bits and has no nontrivial
#       dihedral symmetry: generator samples only such seeds.


GRID_SIZE = 20
SEED_SIZE = 5
QUADRANT_SIZE = 10

# A few fixed seed bits break complement ambiguity while remaining visually minor.
SEED_ANCHORS: dict[tuple[int, int], int] = {
    (0, 0): 1,
    (0, 4): 0,
    (2, 2): 1,
    (4, 0): 1,
    (4, 4): 0,
}

SEED_SUM_MIN = 8
SEED_SUM_MAX = 17

# 2x2 seed-copy layout inside the canonical 10x10 quadrant.
QUADRANT_LAYOUT = (
    (0, 1),
    (4, 5),
)


def _apply_transform(arr: np.ndarray, variant: int) -> np.ndarray:
    if variant < 4:
        return np.rot90(arr, variant)
    return np.fliplr(np.rot90(arr, variant - 4))


def _build_quadrant(seed_like: np.ndarray) -> np.ndarray:
    rows = []
    for row in QUADRANT_LAYOUT:
        rows.append(np.hstack([_apply_transform(seed_like, variant) for variant in row]))
    return np.vstack(rows)


def _build_full(seed_like: np.ndarray) -> np.ndarray:
    quadrant = _build_quadrant(seed_like)
    return np.block(
        [
            [quadrant, np.rot90(quadrant, 3)],
            [np.rot90(quadrant, 1), np.rot90(quadrant, 2)],
        ]
    )


def _unique_dihedral_variants(arr: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    for k in range(4):
        for candidate in (np.rot90(arr, k), np.fliplr(np.rot90(arr, k))):
            if not any(np.array_equal(candidate, seen) for seen in variants):
                variants.append(candidate.copy())
    return variants


SEED_LABELS = np.arange(SEED_SIZE * SEED_SIZE, dtype=int).reshape(SEED_SIZE, SEED_SIZE)
LABEL_VARIANTS = _unique_dihedral_variants(_build_full(SEED_LABELS))


def _seed_from_assignments(assignments: np.ndarray) -> np.ndarray:
    return assignments.reshape(SEED_SIZE, SEED_SIZE).astype(bool)


def _seed_is_valid(seed: np.ndarray) -> bool:
    for (r, c), value in SEED_ANCHORS.items():
        if int(seed[r, c]) != int(value):
            return False

    total = int(seed.sum())
    if not (SEED_SUM_MIN <= total <= SEED_SUM_MAX):
        return False

    for variant in range(1, 8):
        if np.array_equal(seed, _apply_transform(seed, variant)):
            return False

    return True


def _reconstruct_seed(label_window: np.ndarray, visible: np.ndarray) -> np.ndarray | None:
    assignments = np.full(SEED_SIZE * SEED_SIZE, -1, dtype=np.int8)
    flat_labels = label_window.reshape(-1)
    flat_visible = visible.reshape(-1)

    for label, bit in zip(flat_labels, flat_visible, strict=True):
        idx = int(label)
        bit_i = int(bool(bit))
        prev = int(assignments[idx])
        if prev != -1 and prev != bit_i:
            return None
        assignments[idx] = bit_i

    if np.any(assignments < 0):
        return None

    seed = _seed_from_assignments(assignments)
    return seed if _seed_is_valid(seed) else None


def _pattern_from_seed_and_labels(seed: np.ndarray, label_variant: np.ndarray) -> np.ndarray:
    return seed.reshape(-1)[label_variant].astype(bool)


def _find_matches(grid: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    colors = colors_present(g)

    matches: list[tuple[int, int, np.ndarray]] = []
    for fg in colors:
        bg = int(colors[0] if int(colors[1]) == int(fg) else colors[1])
        visible = g == int(fg)
        for label_variant in LABEL_VARIANTS:
            for r in range(GRID_SIZE - H + 1):
                for c in range(GRID_SIZE - W + 1):
                    label_window = label_variant[r : r + H, c : c + W]
                    seed = _reconstruct_seed(label_window, visible)
                    if seed is None:
                        continue
                    pattern = _pattern_from_seed_and_labels(seed, label_variant)
                    matches.append((int(fg), int(bg), pattern))
    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild a full 20x20 symmetric motif from a two-color crop of one of its legal views.

    Input structure:
    - The grid is a two-color rectangular crop cut from a 20x20 binary motif generated from a hidden 5x5 seed.
    - The full motif has quarter-turn symmetry, so its dihedral family collapses to two distinct full-grid classes: the canonical class and its reflected class.

    Rule:
    - Try each input color as the motif foreground.
    - For each unique dihedral class and each possible crop offset, map the visible crop cells back to seed labels 0..24.
    - If the crop determines all seed labels consistently and the recovered seed satisfies the family constraints, regenerate the whole 20x20 motif and repaint it with the matched foreground/background colors.

    Key insight:
    - The crop is not matched against one stored bitmap anymore; instead, the crop encodes enough repeated symmetry evidence to recover the hidden 5x5 seed, which then determines the entire 20x20 output.
    """
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    assert H <= GRID_SIZE and W <= GRID_SIZE

    colors = colors_present(g)
    assert len(colors) == 2

    matches = _find_matches(g)
    assert len(matches) == 1
    fg, bg, pattern = matches[0]
    return np.where(pattern, int(fg), int(bg)).astype(int)


def _random_seed(rng: random.Random) -> np.ndarray:
    while True:
        seed = np.array([[rng.random() < 0.5 for _ in range(SEED_SIZE)] for _ in range(SEED_SIZE)], dtype=bool)
        for (r, c), value in SEED_ANCHORS.items():
            seed[r, c] = bool(value)
        if _seed_is_valid(seed):
            return seed


def _random_colors(rng: random.Random) -> tuple[int, int]:
    fg, bg = rng.sample(range(10), 2)
    return int(fg), int(bg)


def _random_example(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    seed = _random_seed(rng)
    label_variant = rng.choice(LABEL_VARIANTS)
    pattern = _pattern_from_seed_and_labels(seed, label_variant)
    fg, bg = _random_colors(rng)
    output = np.where(pattern, fg, bg).astype(int)

    H = rng.randint(10, 16)
    W = rng.randint(10, 12)
    r0 = rng.randint(0, GRID_SIZE - H)
    c0 = rng.randint(0, GRID_SIZE - W)
    input_grid = output[r0 : r0 + H, c0 : c0 + W].copy()
    return input_grid, output


def generate_input() -> np.ndarray:
    """
    Output-first generator for the D5 variant.

    The crop size is chosen so every legal window covers all 25 seed labels, which
    keeps the solver's latent-seed reconstruction uniquely determined.
    """
    rng = random.Random()
    for _ in range(200):
        input_grid, output = _random_example(rng)
        solved = main(input_grid)
        if np.array_equal(solved, output):
            return input_grid
    raise RuntimeError("failed to generate a valid input for the D5 variant")
