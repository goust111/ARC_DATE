# Combo variant of db0c5428 (index 35)
# combo_spec: D10/0+D15/1
# Name: 固定模板模式+严格两色

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, paste

# Design analysis for combo D10/0 + D15/1
# A. Concrete behavior trace:
#    Original db0c5428 allowed free multi-color 3x3 blocks, with one corner seed and
#    one edge seed expanded by symmetry. Here a sample first picks a single asymmetric
#    3x3 binary template mask. Every visible non-center block is a rotation/reflection
#    of that same mask, colored with one global majority color and one global accent
#    color only. The output rule is still "copy blocks to opposite outer slots and
#    rebuild the center from the hole-facing ring", but now the reconstructed center is
#    forced to be another instance of the same template family.
# B. Complexity calibration:
#    Restricting to two colors could collapse the task into nearly uniform tiles, so the
#    generator keeps the template asymmetric and requires at least three distinct rigid
#    transforms around the hole. This preserves orientation evidence for humans.
# C. Original assumptions that changed:
#    - Foreground is no longer 2-4 colors; it is exactly two colors.
#    - There are no separate "corner pattern" and "edge pattern" families anymore.
#      Instead all eight visible blocks must normalize to one shared binary template.
#    - The center is still reconstructed from the inner ring, but now the ring is chosen
#      so that the filled center also belongs to the same transform family.
# D. Placement legality:
#    The motif is still 9x9 and is placed with pad >= 3, so the 15x15 expanded output
#    always fits inside the 16-19 square canvas without overlap or clipping.
# E. Visual observability:
#    Across examples a human can immediately see two stable cues: only two foreground
#    colors exist, and every block repeats the same accent silhouette under rotations or
#    reflections. The center hole is therefore not arbitrary; its rim already specifies
#    the missing block's outer eight cells.
# F. Visual cleanliness:
#    No extra markers or overlapping layers are added. Each slot contains one clean 3x3
#    block on a plain background, so the template repetition remains readable.
# G. Color inferability:
#    The majority color is the dominant foreground globally and always fills the center
#    pixel of the missing block; the accent color is sparse and occupies a small fixed
#    mask. This avoids hidden numeric-color logic.
# H. Non-degeneracy target:
#    Post-generation checks should verify exact two-color usage, single-template-family
#    reuse, and nontrivial transform diversity among the eight visible blocks.
# I. New solver assumptions audited:
#    - bbox(...) assumes one foreground motif; generator builds exactly one.
#    - block == 3 remains required; generator never changes block size.
#    - the shared-template assertion assumes every visible block uses only the two
#      foreground colors and differs only by rigid transforms; generator guarantees it.
#    - center reconstruction still assumes the eight sampled rim pixels are in the same
#      places as the original task; the slot->sample mapping is preserved exactly.

# -- CFG: canvas & structure --
GRID_SIZE_MIN, GRID_SIZE_MAX = 16, 19
BLOCK = 3
MOTIF_SIZE = BLOCK * 3
EXPANDED_SIZE = BLOCK * 5
MOTIF_PAD = 3

# -- CFG: colors --
BG = 8
FOREGROUND_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

# -- CFG: generation --
MAX_RETRIES = 300
MIN_TRANSFORM_DIVERSITY = 3

# Boolean template library: True = accent color, False = majority color.
# Center stays majority so the reconstructed center pixel can stay the global main color.
TEMPLATE_LIBRARY = (
    np.array(
        [
            [1, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 0, 0],
            [0, 0, 1],
            [0, 0, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 0],
            [0, 0, 0],
            [1, 0, 0],
        ],
        dtype=bool,
    ),
)

# Center reconstruction uses the same eight hole-facing cells as the original task.
_CENTER_OFFSETS = np.array([-1, 1, 3])
_CENTER_SAMPLE_MAP = {
    (0, 0): ((0, 0), (2, 2)),
    (0, 1): ((0, 1), (2, 1)),
    (0, 2): ((0, 2), (2, 0)),
    (1, 0): ((1, 0), (1, 2)),
    (1, 2): ((1, 2), (1, 0)),
    (2, 0): ((2, 0), (0, 2)),
    (2, 1): ((2, 1), (0, 1)),
    (2, 2): ((2, 2), (0, 0)),
}


def _apply_transform(arr: np.ndarray, idx: int) -> np.ndarray:
    if idx == 0:
        return arr.copy()
    if idx == 1:
        return np.rot90(arr, 1)
    if idx == 2:
        return np.rot90(arr, 2)
    if idx == 3:
        return np.rot90(arr, 3)
    if idx == 4:
        return np.fliplr(arr)
    if idx == 5:
        return np.flipud(arr)
    if idx == 6:
        return arr.T.copy()
    if idx == 7:
        return np.rot90(arr.T, 2)
    raise ValueError(f"unknown transform index: {idx}")


def _mask_key(mask: np.ndarray) -> bytes:
    return np.ascontiguousarray(mask.astype(np.uint8)).tobytes()


def _major_foreground_color(g: np.ndarray, bg: int) -> int:
    vals, counts = np.unique(g[g != int(bg)], return_counts=True)
    assert vals.size, "input must contain foreground colors"
    return int(vals[int(np.argmax(counts))])


def _foreground_colors(g: np.ndarray, bg: int) -> list[int]:
    return sorted(int(v) for v in np.unique(g) if int(v) != int(bg))


def _visible_blocks(g: np.ndarray, r0: int, c0: int, block: int) -> list[np.ndarray]:
    blocks: list[np.ndarray] = []
    for br in range(3):
        for bc in range(3):
            if (br, bc) == (1, 1):
                continue
            blocks.append(
                crop_bbox(
                    g,
                    (
                        r0 + br * block,
                        c0 + bc * block,
                        r0 + (br + 1) * block - 1,
                        c0 + (bc + 1) * block - 1,
                    ),
                )
            )
    return blocks


def _assert_shared_template(g: np.ndarray, r0: int, c0: int, block: int, majority: int, accent: int) -> None:
    masks = []
    for blk in _visible_blocks(g, r0, c0, block):
        vals = set(int(v) for v in np.unique(blk))
        assert vals == {majority, accent}, "each visible block must use the two foreground colors only"
        masks.append(blk == accent)

    family = {_mask_key(_apply_transform(masks[0], idx)) for idx in range(8)}
    assert all(_mask_key(mask) in family for mask in masks[1:]), "blocks must share one template family"


def _center_patch(g: np.ndarray, r0: int, c0: int, block: int, bg: int) -> np.ndarray:
    assert block == 3, "this variant keeps 3x3 blocks"
    majority = _major_foreground_color(g, bg)
    patch = np.full((block, block), majority, dtype=int)

    hole_r = int(r0) + block
    hole_c = int(c0) + block
    for r in range(block):
        for c in range(block):
            if r == block // 2 and c == block // 2:
                continue
            patch[r, c] = int(g[hole_r + _CENTER_OFFSETS[r], hole_c + _CENTER_OFFSETS[c]])
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a two-color fixed-template block ring into the same 5x5 block layout as the original task.

    Input structure:
    - A single 9x9 foreground motif sits on a teal background.
    - The motif is a 3x3 array of 3x3 blocks with the middle block missing.
    - Exactly two foreground colors are used: one dominant main color and one sparse accent color.
    - Every visible block is a rotation or reflection of one shared 3x3 template.

    Rule:
    - Leave the original 9x9 motif in place.
    - Copy each of the eight visible 3x3 blocks to its point-symmetric outer slot in the implied 5x5 block layout, without re-rotating it.
    - Rebuild the missing center block from the eight hole-facing pixels around the gap, then set its center pixel to the global majority foreground color.

    Key insight:
    The two-color template family makes all surrounding blocks structurally equivalent, so the only missing information is the center block's rim. Those eight rim pixels are already exposed around the hole, and the middle cell is the sample's majority color.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    motif_bbox = bbox(g != bg)
    assert motif_bbox is not None, "expected a foreground motif"
    r0, c0, r1, c1 = motif_bbox
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    assert h == w and h % 3 == 0, "motif must be a square 3x3 block grid"
    block = h // 3

    fg_colors = _foreground_colors(g, bg)
    assert len(fg_colors) == 2, "combo variant expects exactly two foreground colors"
    majority = _major_foreground_color(g, bg)
    accent = next(color for color in fg_colors if color != majority)
    _assert_shared_template(g, int(r0), int(c0), block, majority, accent)

    out = g.copy()
    expanded_r0 = int(r0) - block
    expanded_c0 = int(c0) - block

    for br in range(3):
        for bc in range(3):
            if (br, bc) == (1, 1):
                continue
            src = crop_bbox(
                g,
                (
                    int(r0) + br * block,
                    int(c0) + bc * block,
                    int(r0) + (br + 1) * block - 1,
                    int(c0) + (bc + 1) * block - 1,
                ),
            )
            dst_r = expanded_r0 + (4 - 2 * br) * block
            dst_c = expanded_c0 + (4 - 2 * bc) * block
            paste(out, src, dst_r, dst_c)

    paste(out, _center_patch(g, int(r0), int(c0), block, bg), int(r0) + block, int(c0) + block)
    return out


def _colorize(mask: np.ndarray, majority: int, accent: int) -> np.ndarray:
    return np.where(mask, accent, majority).astype(int)


def _choose_center_mask(base_mask: np.ndarray) -> np.ndarray:
    return _apply_transform(base_mask, random.randrange(8))


def _choose_surrounding_transforms(base_mask: np.ndarray, center_mask: np.ndarray) -> dict[tuple[int, int], int]:
    options: dict[tuple[int, int], list[int]] = {}
    for slot, (center_rc, sample_rc) in _CENTER_SAMPLE_MAP.items():
        want_accent = bool(center_mask[center_rc])
        choices = [
            idx
            for idx in range(8)
            if bool(_apply_transform(base_mask, idx)[sample_rc]) == want_accent
        ]
        assert choices, "template must admit at least one matching neighbor orientation"
        options[slot] = choices

    for _ in range(200):
        picks = {slot: random.choice(choices) for slot, choices in options.items()}
        if len(set(picks.values())) >= MIN_TRANSFORM_DIVERSITY:
            return picks

    # Deterministic fallback: spread choices cyclically to preserve visible variety.
    picks = {}
    used = []
    for slot, choices in options.items():
        choices = sorted(choices)
        idx = choices[len(used) % len(choices)]
        picks[slot] = idx
        used.append(idx)
    return picks


def _assemble_motif(base_mask: np.ndarray, majority: int, accent: int) -> tuple[np.ndarray, np.ndarray, dict[tuple[int, int], int]]:
    center_mask = _choose_center_mask(base_mask)
    slot_transforms = _choose_surrounding_transforms(base_mask, center_mask)

    motif = np.full((MOTIF_SIZE, MOTIF_SIZE), BG, dtype=int)
    for br in range(3):
        for bc in range(3):
            if (br, bc) == (1, 1):
                continue
            mask = _apply_transform(base_mask, slot_transforms[(br, bc)])
            paste(motif, _colorize(mask, majority, accent), br * BLOCK, bc * BLOCK)
    center = _colorize(center_mask, majority, accent)
    return motif, center, slot_transforms


def _expanded_output(motif: np.ndarray, center: np.ndarray) -> np.ndarray:
    expanded = np.full((EXPANDED_SIZE, EXPANDED_SIZE), BG, dtype=int)
    output_motif = motif.copy()
    paste(output_motif, center, BLOCK, BLOCK)
    paste(expanded, output_motif, BLOCK, BLOCK)
    for br in range(3):
        for bc in range(3):
            if (br, bc) == (1, 1):
                continue
            src = motif[br * BLOCK:(br + 1) * BLOCK, bc * BLOCK:(bc + 1) * BLOCK]
            paste(expanded, src, (4 - 2 * br) * BLOCK, (4 - 2 * bc) * BLOCK)
    return expanded


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    majority, accent = random.sample(FOREGROUND_COLORS, 2)
    base_mask = TEMPLATE_LIBRARY[random.randrange(len(TEMPLATE_LIBRARY))].copy()

    motif, center, slot_transforms = _assemble_motif(base_mask, majority, accent)
    expanded = _expanded_output(motif, center)

    r0 = random.randint(MOTIF_PAD, n - MOTIF_SIZE - MOTIF_PAD)
    c0 = random.randint(MOTIF_PAD, n - MOTIF_SIZE - MOTIF_PAD)
    inp = np.full((n, n), BG, dtype=int)
    out = np.full((n, n), BG, dtype=int)
    paste(inp, motif, r0, c0)
    paste(out, expanded, r0 - BLOCK, c0 - BLOCK)

    # Generator-side structural audit: exact two colors + one template family + visible transform diversity.
    if len(set(slot_transforms.values())) < MIN_TRANSFORM_DIVERSITY:
        return None
    if len(_foreground_colors(inp, BG)) != 2:
        return None

    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid combo db0c5428 input")
