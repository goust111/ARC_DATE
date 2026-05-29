# Combo variant of db0c5428 (index 41)
# combo_spec: D4/3+D15/2
# Name: 朝向标记像素+条件旋转

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior:
#    The canvas still contains one 9x9 motif built from eight visible 3x3 blocks
#    around a missing 3x3 center. The output still expands that motif to the
#    sparse 5x5 block scaffold from the base task. Two changes are fused:
#    - every visible block now contains the same unique marker color exactly once,
#      and that pixel sits on one of the four block corners to show orientation
#    - when copying outward, edge-source blocks keep their orientation, but
#      corner-source blocks rotate 90 degrees clockwise before pasting
#    Small trace: if the top-left source block carries the marker on its own
#    top-left corner, the outer bottom-right copy must show that same marker on
#    the top-right corner, because the whole block was turned CW once. By
#    contrast, the top-middle edge block keeps its marker in the same local
#    corner because edge copies do not rotate. Center reconstruction is unchanged:
#    the 8 cells facing the hole fill the center border, and the center pixel is
#    the majority non-marker foreground color.
# b) Complexity calibration:
#    The branch space stays small and repeated: only two source categories exist
#    in every sample (4 corners, 4 edges), and the marker has only 4 visible
#    corner states. This makes the fusion readable from a few examples without
#    introducing many independent cases.
# c) Hidden assumptions audit:
#    - `bbox(g != bg)` still works because there is still exactly one motif.
#    - block size remains 3, so the old center-fill geometry still applies.
#    - the original solver assumed raw outer pasting was always correct; the
#      fused solver replaces that with "read the shared marker, then rotate iff
#      the source slot is a corner slot".
#    - the base generator's mirror-symmetry assumption no longer holds cleanly:
#      one unique corner marker per block conflicts with full D4 mirror symmetry.
#      The fused generator therefore uses rotationally related seed blocks rather
#      than mirror-derived symmetric blocks.
#    - the new solver assumes the marker color is uniquely identifiable, appears
#      exactly once per visible block, and always lies on a corner. The generator
#      explicitly guarantees all three properties.
# d) Placement legality:
#    The motif is still placed with at least a 3-cell margin on a 16..19 square
#    canvas, so the full 15x15 expanded output always fits in-bounds. There is
#    only one object, so there is no overlap ambiguity.
# e) Visual observability:
#    A human can compare each source block with its opposite outer copy and see
#    the marker either stay in place (edge sources) or move one corner clockwise
#    (corner sources). Because the motif contains 4 edge examples and 4 corner
#    examples, the positional condition is repeated inside every sample.
# f) Visual cleanliness:
#    No extra objects are added beyond a single corner marker inside each block.
#    The marker never touches the eight inner-boundary cells used to rebuild the
#    center, so the hole reconstruction remains visually separate from the outer
#    copy rule.
# g) Color inferability:
#    The control color has one job only: encode orientation. It does not control
#    the center color and it is shared across all visible blocks, so the rule is
#    discoverable by shape/orientation rather than by memorizing arbitrary color
#    mappings.
# h) Non-degeneracy metric:
#    The main quantitative metric is the number of output cells that differ from
#    the base task's "never rotate outer copies" behavior on the same input.
#    Because each rotated corner copy must move the unique marker to a new corner,
#    every valid sample differs from the base rule in at least 8 cells overall.
# i) New-assumption check against the generator:
#    The generator guarantees the marker is unique-in-role, keeps it off the
#    inner boundary, enforces asymmetric seed blocks so orientations are visible,
#    and verifies the fused output differs from the base output by the metric
#    above. Every branch used by `main()` is therefore backed by construction.


BLOCK = 3
MOTIF_SIZE = BLOCK * 3
EXPANDED_SIZE = BLOCK * 5

GRID_SIZE_MIN, GRID_SIZE_MAX = 16, 19
MOTIF_PAD = 3
MAX_RETRIES = 400

BG = 8
PALETTE = [1, 2, 3, 4, 5, 6, 7, 9]
BODY_COLOR_COUNT_MIN, BODY_COLOR_COUNT_MAX = 2, 4

_CENTER_OFFSETS = np.array([-1, 1, 3])
_VISIBLE_BLOCKS = (
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 2),
    (2, 0),
    (2, 1),
    (2, 2),
)
_CORNERS_CW = ((0, 0), (0, 2), (2, 2), (2, 0))
_CORNER_TO_INDEX = {corner: idx for idx, corner in enumerate(_CORNERS_CW)}


def _major_foreground_color(g: np.ndarray, bg: int, exclude: set[int] | None = None) -> int:
    mask = g != int(bg)
    if exclude:
        for color in exclude:
            mask &= g != int(color)
    vals, counts = np.unique(g[mask], return_counts=True)
    assert vals.size, "input must contain foreground colors"
    return int(vals[int(np.argmax(counts))])


def _block_slice(r0: int, c0: int, br: int, bc: int) -> tuple[int, int, int, int]:
    return (
        int(r0) + br * BLOCK,
        int(c0) + bc * BLOCK,
        int(r0) + (br + 1) * BLOCK - 1,
        int(c0) + (bc + 1) * BLOCK - 1,
    )


def _extract_motif(g: np.ndarray, bg: int) -> tuple[int, int]:
    motif_bbox = bbox(g != bg)
    assert motif_bbox is not None, "expected a foreground motif"
    r0, c0, r1, c1 = motif_bbox
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    assert h == w and h == MOTIF_SIZE, "motif must be a 9x9 3x3-block arrangement"
    return int(r0), int(c0)


def _iter_visible_blocks(g: np.ndarray, r0: int, c0: int) -> list[np.ndarray]:
    return [crop_bbox(g, _block_slice(r0, c0, br, bc)) for br, bc in _VISIBLE_BLOCKS]


def _marker_color(g: np.ndarray, r0: int, c0: int, bg: int) -> int:
    blocks = _iter_visible_blocks(g, r0, c0)
    colors = [int(c) for c in np.unique(g) if int(c) != int(bg)]
    candidates: list[int] = []
    for color in colors:
        ok = True
        for block in blocks:
            hits = np.argwhere(block == int(color))
            if hits.shape[0] != 1:
                ok = False
                break
            if tuple(int(x) for x in hits[0]) not in _CORNERS_CW:
                ok = False
                break
        if ok:
            candidates.append(int(color))
    assert len(candidates) == 1, "expected one unique marker color"
    return int(candidates[0])


def _marker_corner(block: np.ndarray, marker: int) -> tuple[int, int]:
    hits = np.argwhere(block == int(marker))
    assert hits.shape[0] == 1, "each visible block must contain the marker exactly once"
    rc = tuple(int(x) for x in hits[0])
    assert rc in _CORNERS_CW, "marker must lie on a block corner"
    return rc


def _rotate_cw(block: np.ndarray, steps: int) -> np.ndarray:
    return np.rot90(block, -int(steps))


def _is_corner_slot(br: int, bc: int) -> bool:
    return br != 1 and bc != 1


def _outer_copy_transform(block: np.ndarray, br: int, bc: int, marker: int) -> np.ndarray:
    start_corner = _marker_corner(block, marker)
    steps = 1 if _is_corner_slot(br, bc) else 0
    rotated = _rotate_cw(block, steps)
    end_corner = _marker_corner(rotated, marker)
    assert _CORNER_TO_INDEX[end_corner] == (_CORNER_TO_INDEX[start_corner] + steps) % 4
    return rotated


def _center_patch(g: np.ndarray, r0: int, c0: int, bg: int, marker: int) -> np.ndarray:
    main = _major_foreground_color(g, bg, exclude={marker})
    patch = np.full((BLOCK, BLOCK), main, dtype=int)
    hole_r = int(r0) + BLOCK
    hole_c = int(c0) + BLOCK
    for r in range(BLOCK):
        for c in range(BLOCK):
            if r == BLOCK // 2 and c == BLOCK // 2:
                continue
            patch[r, c] = int(g[hole_r + _CENTER_OFFSETS[r], hole_c + _CENTER_OFFSETS[c]])
    return patch


def behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    """Quantify how strongly the fused rule differs from the base task."""
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    r0, c0 = _extract_motif(g, bg)
    marker = _marker_color(g, r0, c0, bg)
    center = _center_patch(g, r0, c0, bg, marker)
    motif = g[r0 : r0 + MOTIF_SIZE, c0 : c0 + MOTIF_SIZE]
    fused = _expanded_output(motif, center, marker)
    base = _expanded_output_base(motif, center)
    corner_changed_blocks = 0
    for br, bc in ((0, 0), (0, 2), (2, 0), (2, 2)):
        src = motif[br * BLOCK : (br + 1) * BLOCK, bc * BLOCK : (bc + 1) * BLOCK]
        if not np.array_equal(src, _outer_copy_transform(src, br, bc, marker)):
            corner_changed_blocks += 1
    return {
        "outer_diff_vs_base": int(np.sum(fused != base)),
        "corner_changed_blocks": int(corner_changed_blocks),
        "corner_rotations": 4,
        "edge_rotations": 0,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a marked 3x3-block motif into a 5x5 frame with position-based corner rotation.

    Input structure:
    - One 9x9 motif sits on a uniform background, arranged as eight visible 3x3 blocks around an empty 3x3 center.
    - Every visible block contains the same special marker color exactly once, and that marker sits on one of the block's four corners to show orientation.
    - The eight cells bordering the center hole still encode the non-central pixels of the missing middle block.

    Rule:
    - Keep the original 9x9 motif in place.
    - Reconstruct the missing center block from the eight inner-boundary pixels, using the motif's majority non-marker color at the center.
    - Copy each visible block to the opposite outer slot of the implied 5x5 block layout; edge-source blocks are pasted unchanged, while corner-source blocks are rotated 90 degrees clockwise before pasting.

    Key insight:
    - The marker tells you how each block is locally oriented, so you can see that only the four corner-source copies are quarter-turned; the center block is still recovered independently from the ring around the hole.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    r0, c0 = _extract_motif(g, bg)
    marker = _marker_color(g, r0, c0, bg)

    out = g.copy()
    expanded_r0 = int(r0) - BLOCK
    expanded_c0 = int(c0) - BLOCK

    for br in range(3):
        for bc in range(3):
            if (br, bc) == (1, 1):
                continue
            src = crop_bbox(g, _block_slice(r0, c0, br, bc))
            dst_r = expanded_r0 + (4 - 2 * br) * BLOCK
            dst_c = expanded_c0 + (4 - 2 * bc) * BLOCK
            paste(out, _outer_copy_transform(src, br, bc, marker), dst_r, dst_c)

    paste(out, _center_patch(g, r0, c0, bg, marker), int(r0) + BLOCK, int(c0) + BLOCK)
    return out


def _random_seed_block(
    marker_corner: tuple[int, int],
    marker_color: int,
    majority: int,
    accent_colors: list[int],
    force_body_cell: tuple[int, int] | None = None,
) -> np.ndarray:
    block = np.full((BLOCK, BLOCK), int(majority), dtype=int)
    block[marker_corner] = int(marker_color)

    cells = [(r, c) for r in range(BLOCK) for c in range(BLOCK) if (r, c) != marker_corner]
    random.shuffle(cells)

    accent_count = random.randint(max(2, len(accent_colors)), min(4, len(cells)))
    chosen: list[tuple[int, int]] = []
    if force_body_cell is not None and force_body_cell != marker_corner:
        chosen.append(force_body_cell)
    for cell in cells:
        if cell not in chosen:
            chosen.append(cell)
        if len(chosen) >= accent_count:
            break

    colors = list(accent_colors)
    random.shuffle(colors)
    while len(colors) < len(chosen):
        colors.append(random.choice(accent_colors))
    random.shuffle(colors)
    for (r, c), color in zip(chosen, colors):
        block[int(r), int(c)] = int(color)
    return block


def _all_rotations_distinct(block: np.ndarray) -> bool:
    seen = {tuple(np.rot90(block, k).flatten()) for k in range(4)}
    return len(seen) == 4


def _assemble_motif(corner_seed: np.ndarray, edge_seed: np.ndarray) -> np.ndarray:
    motif = np.full((MOTIF_SIZE, MOTIF_SIZE), BG, dtype=int)
    blocks = {
        (0, 0): corner_seed,
        (0, 1): edge_seed,
        (0, 2): np.rot90(corner_seed, -1),
        (1, 0): np.rot90(edge_seed, 1),
        (1, 2): np.rot90(edge_seed, -1),
        (2, 0): np.rot90(corner_seed, 1),
        (2, 1): np.rot90(edge_seed, 2),
        (2, 2): np.rot90(corner_seed, 2),
    }
    for (br, bc), blk in blocks.items():
        paste(motif, blk, br * BLOCK, bc * BLOCK)
    return motif


def _gen_center_patch(motif: np.ndarray, majority: int) -> np.ndarray:
    rows = BLOCK + _CENTER_OFFSETS
    cols = BLOCK + _CENTER_OFFSETS
    patch = motif[np.ix_(rows, cols)].copy()
    patch[1, 1] = int(majority)
    return patch


def _expanded_output_base(motif: np.ndarray, center: np.ndarray) -> np.ndarray:
    expanded = np.full((EXPANDED_SIZE, EXPANDED_SIZE), BG, dtype=int)
    output_motif = motif.copy()
    paste(output_motif, center, BLOCK, BLOCK)
    paste(expanded, output_motif, BLOCK, BLOCK)
    for br, bc in _VISIBLE_BLOCKS:
        src = motif[br * BLOCK : (br + 1) * BLOCK, bc * BLOCK : (bc + 1) * BLOCK]
        paste(expanded, src, (4 - 2 * br) * BLOCK, (4 - 2 * bc) * BLOCK)
    return expanded


def _expanded_output(motif: np.ndarray, center: np.ndarray, marker: int) -> np.ndarray:
    expanded = np.full((EXPANDED_SIZE, EXPANDED_SIZE), BG, dtype=int)
    output_motif = motif.copy()
    paste(output_motif, center, BLOCK, BLOCK)
    paste(expanded, output_motif, BLOCK, BLOCK)
    for br, bc in _VISIBLE_BLOCKS:
        src = motif[br * BLOCK : (br + 1) * BLOCK, bc * BLOCK : (bc + 1) * BLOCK]
        rotated = _outer_copy_transform(src, br, bc, marker)
        paste(expanded, rotated, (4 - 2 * br) * BLOCK, (4 - 2 * bc) * BLOCK)
    return expanded


def _marker_histogram_ok(motif: np.ndarray, marker: int) -> bool:
    counts = {corner: 0 for corner in _CORNERS_CW}
    for br, bc in _VISIBLE_BLOCKS:
        block = motif[br * BLOCK : (br + 1) * BLOCK, bc * BLOCK : (bc + 1) * BLOCK]
        counts[_marker_corner(block, marker)] += 1
    return all(counts[corner] == 2 for corner in _CORNERS_CW)


def _marker_unique_in_motif(motif: np.ndarray, marker: int) -> bool:
    colors = [int(c) for c in np.unique(motif) if int(c) != BG]
    candidates: list[int] = []
    for color in colors:
        ok = True
        for br, bc in _VISIBLE_BLOCKS:
            block = motif[br * BLOCK : (br + 1) * BLOCK, bc * BLOCK : (bc + 1) * BLOCK]
            hits = np.argwhere(block == int(color))
            if hits.shape[0] != 1:
                ok = False
                break
            if tuple(int(x) for x in hits[0]) not in _CORNERS_CW:
                ok = False
                break
        if ok:
            candidates.append(int(color))
    return candidates == [int(marker)]


def _body_majority_ok(motif: np.ndarray, majority: int, marker: int) -> bool:
    vals, counts = np.unique(motif[(motif != BG) & (motif != int(marker))], return_counts=True)
    freq = {int(v): int(c) for v, c in zip(vals, counts)}
    if not freq or int(majority) not in freq:
        return False
    if len(freq) == 1:
        return True
    return freq[int(majority)] > max(v for k, v in freq.items() if k != int(majority))


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    body_color_count = random.randint(BODY_COLOR_COUNT_MIN, BODY_COLOR_COUNT_MAX)
    colors = random.sample(PALETTE, body_color_count + 1)
    marker = int(colors[0])
    body_colors = [int(c) for c in colors[1:]]
    majority = int(body_colors[0])
    accent_colors = body_colors[1:] if len(body_colors) > 1 else [majority]

    corner_seed = _random_seed_block((0, 0), marker, majority, accent_colors, force_body_cell=(2, 2))
    edge_seed = _random_seed_block((0, 2), marker, majority, accent_colors, force_body_cell=(2, 1))
    if not _all_rotations_distinct(corner_seed) or not _all_rotations_distinct(edge_seed):
        return None

    motif = _assemble_motif(corner_seed, edge_seed)
    center = _gen_center_patch(motif, majority)
    if np.any(center == int(marker)):
        return None
    if not _marker_histogram_ok(motif, marker):
        return None
    if not _marker_unique_in_motif(motif, marker):
        return None
    if not _body_majority_ok(motif, majority, marker):
        return None

    fused = _expanded_output(motif, center, marker)
    base = _expanded_output_base(motif, center)
    if int(np.sum(fused != base)) < 8:
        return None

    r0 = random.randint(MOTIF_PAD, n - MOTIF_SIZE - MOTIF_PAD)
    c0 = random.randint(MOTIF_PAD, n - MOTIF_SIZE - MOTIF_PAD)
    inp = np.full((n, n), BG, dtype=int)
    out = np.full((n, n), BG, dtype=int)
    paste(inp, motif, r0, c0)
    paste(out, fused, r0 - BLOCK, c0 - BLOCK)

    metrics = behavior_metrics(inp)
    if metrics["corner_changed_blocks"] != 4:
        return None
    if metrics["outer_diff_vs_base"] < 8:
        return None
    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid db0c5428 combo input")
