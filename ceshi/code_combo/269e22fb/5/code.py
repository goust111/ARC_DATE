# Combo variant of 269e22fb (index 5)
# combo_spec: D5/0+D6/0
# Name: 对称图案旋转群匹配

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present


# Design analysis for the combo variant
# -------------------------------------
# spec:
#   core_change:
#     original:
#       - one fixed 20x20 bitmap is stored directly in the solver;
#       - the solver tests every crop against 8 distinct D4 views
#         (4 rotations and 4 reflections);
#       - uniqueness comes from orientation and offset together.
#     combo:
#       - the 20x20 bitmap is generated from a hidden 5x5 seed;
#       - only rotations are legal transforms, but the generated family has
#         global C4 symmetry, so all 4 rotations collapse to the same 20x20 view;
#       - uniqueness therefore comes mostly from crop offset plus the repeated
#         label structure needed to reconstruct the seed.
#   concrete_shape:
#     seed:
#       - a 5x5 boolean array.
#     10x10 quarter:
#       - built from four seed transforms:
#         [[seed, rot90(seed)],
#          [fliplr(seed), rot90(fliplr(seed))]]
#     20x20 full motif:
#       - tiled from the 10x10 quarter by quarter-turns:
#         [[Q, rot270(Q)],
#          [rot90(Q), rot180(Q)]]
#       - this makes the full pattern exactly invariant under 0/90/180/270
#         degree rotation; the generator additionally rejects any full pattern
#         that would also be reflection-invariant.
#   behavior_trace:
#     micro_example:
#       - label 0 from the hidden seed appears in four rotated places in the
#         20x20 label map; a 9x9 crop already sees every label at least once.
#       - if one visible crop cell saying "foreground" lands on label 0, then
#         every occurrence of label 0 in that candidate window must also mean 1.
#     original_flow:
#       - try polarity -> compare the crop with every fixed bitmap window.
#     combo_flow:
#       - try polarity -> slide the crop over one canonical C4 label map ->
#         convert the crop into assignments for seed labels 0..24 ->
#         reject inconsistent or incomplete assignments ->
#         rebuild the whole 20x20 motif from the recovered seed.
#     difference:
#       - matching no longer selects among several orientations. Rotations are
#         legal in the abstract task family, but they produce the same full
#         picture, so the solver's external search only needs offsets.
#   complexity_calibration:
#     - If the crop is too small, some windows would miss seed labels and allow
#       multiple completions. The chosen family is stronger than the single D5
#       variant: every 9x9-or-larger legal crop already touches all 25 labels.
#     - Generator therefore uses H in [9,16] and W in [9,12], close to the
#       original task sizes while keeping the reconstruction exact.
#     - Fixed seed anchors and a moderate seed-population range prevent the
#       complement polarity from becoming another legal solution.
#     - Banning nontrivial seed dihedral symmetries and reflected full outputs
#       keeps the outputs visually varied and makes the "rotations only" change
#       visible instead of degenerating back into a reflection-safe family.
#   implicit_assumptions_review:
#     - original `_BASE_PATTERN` assumed one immutable motif; invalid now.
#     - original `_dihedral_variants` assumed orientation choice mattered;
#       invalid now because reflections are disallowed and rotations coincide.
#     - original polarity loop relied on the complement not being a legal motif;
#       still needs explicit seed anchors in the generated family.
#     - original `assert len(matches) == 1` now depends on two generator
#       guarantees: every crop covers all 25 labels, and sampled seeds satisfy
#       the family constraints used by `_seed_is_valid`.
#   placement_legality:
#     - The crop is still one axis-aligned rectangle inside a 20x20 canvas.
#     - No extra objects are introduced, so legality reduces to choosing
#       `0 <= r0 <= 20-H` and `0 <= c0 <= 20-W`.
#   visual_observability:
#     - Across examples, the full outputs visibly change because each hidden
#       seed is different, yet every output has obvious quarter-turn symmetry.
#     - The absence of reflections is inferable because all observed examples
#       complete into a C4-symmetric figure rather than into unrelated mirror
#       views of a single fixed bitmap.
#   cleanliness:
#     - Input and output still use exactly two colors.
#     - The repeated quarter-turn structure gives strong regularity instead of a
#       cluttered collage, so the new rule stays readable to humans.
#   new_main_assumptions_checked_against_generator:
#     - main assumes exactly two colors: generator always paints one foreground
#       color and one background color.
#     - main assumes the crop can determine every seed label: generator restricts
#       crop sizes to 9x9 or larger, which covers all 25 labels in this family.
#     - main assumes the recovered seed passes `_seed_is_valid`: generator only
#       samples seeds from that legal subfamily, including the no-reflection
#       full-pattern constraint.


GRID_SIZE = 20
SEED_SIZE = 5

# These fixed bits break polarity/complement ambiguity without adding markers.
SEED_ANCHORS: dict[tuple[int, int], int] = {
    (0, 0): 1,
    (0, 4): 0,
    (2, 2): 1,
    (4, 0): 1,
    (4, 4): 0,
}

SEED_SUM_MIN = 8
SEED_SUM_MAX = 17

# 2x2 seed-copy layout inside the canonical 10x10 quarter.
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
    quarter = _build_quadrant(seed_like)
    return np.block(
        [
            [quarter, np.rot90(quarter, 3)],
            [np.rot90(quarter, 1), np.rot90(quarter, 2)],
        ]
    )


SEED_LABELS = np.arange(SEED_SIZE * SEED_SIZE, dtype=int).reshape(SEED_SIZE, SEED_SIZE)
LABEL_MAP = _build_full(SEED_LABELS)
assert all(np.array_equal(LABEL_MAP, np.rot90(LABEL_MAP, k)) for k in range(4))


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

    pattern = _pattern_from_seed(seed)
    for k in range(4):
        if np.array_equal(pattern, np.fliplr(np.rot90(pattern, k))):
            return False

    return True


def _reconstruct_seed(label_window: np.ndarray, visible: np.ndarray) -> np.ndarray | None:
    assignments = np.full(SEED_SIZE * SEED_SIZE, -1, dtype=np.int8)
    for label, bit in zip(label_window.reshape(-1), visible.reshape(-1), strict=True):
        idx = int(label)
        bit_i = int(bool(bit))
        prev = int(assignments[idx])
        if prev != -1 and prev != bit_i:
            return None
        assignments[idx] = bit_i

    if np.any(assignments < 0):
        return None

    seed = assignments.reshape(SEED_SIZE, SEED_SIZE).astype(bool)
    return seed if _seed_is_valid(seed) else None


def _pattern_from_seed(seed: np.ndarray) -> np.ndarray:
    return seed.reshape(-1)[LABEL_MAP].astype(bool)


def _find_matches(grid: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    colors = colors_present(g)

    matches: list[tuple[int, int, np.ndarray]] = []
    for fg in colors:
        bg = int(colors[0] if int(colors[1]) == int(fg) else colors[1])
        visible = g == int(fg)
        for r in range(GRID_SIZE - H + 1):
            for c in range(GRID_SIZE - W + 1):
                label_window = LABEL_MAP[r : r + H, c : c + W]
                seed = _reconstruct_seed(label_window, visible)
                if seed is None:
                    continue
                matches.append((int(fg), int(bg), _pattern_from_seed(seed)))
    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the full 20x20 motif from a two-color crop of a C4-symmetric generated pattern.

    Input structure:
    - The grid is a two-color rectangular crop cut from a 20x20 binary motif produced from a hidden 5x5 seed.
    - Only rotational transforms belong to the task family, and because the full motif has quarter-turn symmetry, all legal rotations show the same 20x20 image.

    Rule:
    - Try each input color as the motif foreground.
    - Slide the crop across every offset of the canonical 20x20 label map, convert the crop into assignments for seed labels 0..24, and keep only assignments that are complete and family-valid.
    - Regenerate the whole 20x20 motif from the uniquely recovered seed and repaint it with the matched foreground/background colors.

    Key insight:
    - The crop does not identify an orientation anymore; the rotational orbit has collapsed to one visible class, so the repeated symmetry labels let the solver recover the hidden seed, and the offset is the only remaining external degree of freedom.
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
    base_pattern = _pattern_from_seed(seed)

    # The transform set only contains rotations. Because the motif is C4-symmetric,
    # this sampled rotation changes the latent view choice but not the visible output.
    rotation_k = rng.randrange(4)
    pattern = np.rot90(base_pattern, rotation_k)

    fg, bg = _random_colors(rng)
    output = np.where(pattern, fg, bg).astype(int)

    H = rng.randint(9, 16)
    W = rng.randint(9, 12)
    r0 = rng.randint(0, GRID_SIZE - H)
    c0 = rng.randint(0, GRID_SIZE - W)
    input_grid = output[r0 : r0 + H, c0 : c0 + W].copy()
    return input_grid, output


def generate_input() -> np.ndarray:
    """
    Output-first generator for the D5/0 + D6/0 combo variant.

    The crop remains a plain rectangle, but the sampled examples come from a
    procedural C4-symmetric motif family and only use the rotation transform set.
    """
    rng = random.Random()
    for _ in range(200):
        input_grid, output = _random_example(rng)
        solved = main(input_grid)
        if np.array_equal(solved, output):
            return input_grid
    raise RuntimeError("failed to generate a valid input for the combo variant")
