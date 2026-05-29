# Combo variant of b10624e5 (index 7)
# combo_spec: D6/0+D15/1
# Name: 擦除源图案 + 每个象限既是源也是目标
#
# Design analysis:
# - The original task assumes a single privileged source in the top-left quadrant and
#   empty non-source quadrants except for red anchors. This fusion breaks both
#   assumptions: every quadrant now contains one anchor-attached motif component, and
#   all four motifs participate in the transformation.
# - A literal many-to-many interpretation would make every quadrant receive several
#   overlaid motifs, which is legal but visually noisy and hard to infer from 2-4 ARC
#   examples. To keep D15/1 observable while preserving the "every quadrant is both
#   source and target" idea, this variant uses the only clean one-to-one global
#   symmetry: diagonal exchange. TL swaps with BR, and TR swaps with BL.
# - D6/0 is implemented as move semantics on the decorative part of each motif. The
#   red anchor blocks stay in place, while every non-red pixel attached to an anchor is
#   erased from its source quadrant before the incoming opposite motif is stamped.
# - The original solver line "take the TL component containing red" no longer holds.
#   The new solver extracts one such component per quadrant, pairs quadrants by
#   diagonal opposition, then rescales each source motif to the opposite anchor size.
# - The generator must satisfy new hidden assumptions introduced by the solver:
#   every quadrant has exactly one red square anchor, exactly one 4-connected
#   non-background component containing that anchor, and the incoming swapped motif
#   must fit around the same anchor used by the local motif. Anchor placement is
#   therefore chosen from the intersection of "local motif fits" and "incoming motif
#   fits" ranges.
# - To keep scaling lossless for all 1..4 anchor sizes, motifs are generated on a
#   small logical grid and each logical cell is expanded to an SxS block, where S is
#   the local anchor size. Nearest-neighbor resize between source and target anchor
#   sizes then exactly preserves the logical pattern.
# - Degeneracy is explicitly avoided: diagonal partners are rejected if one motif is
#   already the diagonal reflection of the other, because that would make the swap
#   visually invisible in that quadrant.

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_coords, components_mask, get_bg, paste


BG = 4
DIV = 1
RED = 2
GRID_SIZE = 27
MID = 13
QUADS = {
    (0, 0): (0, 12, 0, 12),
    (0, 1): (0, 12, 14, 26),
    (1, 0): (14, 26, 0, 12),
    (1, 1): (14, 26, 14, 26),
}
OPPOSITE = {
    (0, 0): (1, 1),
    (0, 1): (1, 0),
    (1, 0): (0, 1),
    (1, 1): (0, 0),
}

SIZE_RANGE = (1, 4)
LOGICAL_DIMS = [(2, 2), (2, 3), (3, 2)]
FILL_RATIO_MIN = 0.6
FILL_RATIO_MAX = 1.0
MOTIF_COLORS_MIN = 3
MOTIF_COLORS_MAX = 4
COLOR_POOL = [3, 5, 6, 7, 8, 9]
MAX_RETRIES = 500


def _find_dividers(g: np.ndarray, bg: int) -> tuple[int, int]:
    """Return the unique full-row/full-col divider intersection."""
    h, w = g.shape
    full_rows = [(r, int(g[r, 0])) for r in range(h) if np.all(g[r, :] == g[r, 0]) and int(g[r, 0]) != bg]
    full_cols = [(c, int(g[0, c])) for c in range(w) if np.all(g[:, c] == g[0, c]) and int(g[0, c]) != bg]
    matches = []
    for r, vr in full_rows:
        for c, vc in full_cols:
            if vr == vc:
                matches.append((int(r), int(c)))
    assert len(matches) == 1, "grid must have exactly one matching divider row/col"
    return matches[0]


def _quadrant_bounds(h: int, w: int, mid_r: int, mid_c: int, key: tuple[int, int]) -> tuple[int, int, int, int]:
    qb, qr = key
    r0, r1 = (0, mid_r - 1) if qb == 0 else (mid_r + 1, h - 1)
    c0, c1 = (0, mid_c - 1) if qr == 0 else (mid_c + 1, w - 1)
    return r0, r1, c0, c1


def _resize_nn(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    src_h, src_w = img.shape
    assert new_h > 0 and new_w > 0
    rr = np.clip(np.floor(np.arange(new_h) * src_h / new_h).astype(int), 0, src_h - 1)
    cc = np.clip(np.floor(np.arange(new_w) * src_w / new_w).astype(int), 0, src_w - 1)
    return img[rr[:, None], cc[None, :]]


def _flip_diagonal(img: np.ndarray) -> np.ndarray:
    return np.flipud(np.fliplr(img))


def _transform_to_opposite(patch: np.ndarray, source_size: int, target_size: int) -> np.ndarray:
    ratio = float(target_size) / float(source_size)
    flipped = _flip_diagonal(patch)
    new_h = max(1, int(round(flipped.shape[0] * ratio)))
    new_w = max(1, int(round(flipped.shape[1] * ratio)))
    scaled = _resize_nn(flipped, new_h, new_w)
    roundtrip = _resize_nn(scaled, flipped.shape[0], flipped.shape[1])
    assert np.array_equal(roundtrip, flipped), "motif must be exactly rescalable between anchor sizes"
    return scaled


def _extract_anchor_data(g: np.ndarray, mid_r: int, mid_c: int) -> dict[tuple[int, int], tuple[int, int, int]]:
    red_components = components_mask(g == RED, connectivity=4)
    assert len(red_components) == 4, "grid must contain exactly four red anchors"
    anchors: dict[tuple[int, int], tuple[int, int, int]] = {}
    for pts in red_components:
        r0, c0, r1, c1 = bbox_coords(pts)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        assert height == width, "each anchor must be a square"
        rr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        key = (0 if rr < mid_r else 1, 0 if cc < mid_c else 1)
        assert key not in anchors, "each quadrant must contain exactly one anchor"
        anchors[key] = (int(r0), int(c0), int(height))
    assert len(anchors) == 4, "all quadrants must contain a red anchor"
    return anchors


def _extract_quadrant_patch(
    g: np.ndarray,
    bg: int,
    h: int,
    w: int,
    mid_r: int,
    mid_c: int,
    key: tuple[int, int],
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    q_r0, q_r1, q_c0, q_c1 = _quadrant_bounds(h, w, mid_r, mid_c, key)
    quad = g[q_r0 : q_r1 + 1, q_c0 : q_c1 + 1]
    comps = components_mask(quad != bg, connectivity=4)
    matches: list[tuple[int, int, int, int]] = []
    for pts in comps:
        if any(quad[r, c] == RED for r, c in pts):
            matches.append(bbox_coords(pts))
    assert len(matches) == 1, "each quadrant must have exactly one anchor-attached motif"
    pr0, pc0, pr1, pc1 = matches[0]
    patch = quad[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()
    assert np.any((patch != bg) & (patch != RED)), "each quadrant motif must contain non-red decoration"
    return patch, (int(pr0), int(pc0), int(pr1), int(pc1))


def _red_offset(patch: np.ndarray) -> tuple[int, int]:
    rr, cc = np.where(patch == RED)
    assert rr.size > 0, "patch must contain red anchor pixels"
    return int(rr.min()), int(cc.min())


def _stamp_aligned(
    out: np.ndarray,
    patch: np.ndarray,
    anchor_r: int,
    anchor_c: int,
    mid_r: int,
    mid_c: int,
    bg: int,
) -> None:
    red_r0, red_c0 = _red_offset(patch)
    place_r0 = anchor_r - red_r0
    place_c0 = anchor_c - red_c0
    rr_src, cc_src = np.where(patch != bg)
    for rr, cc in zip(rr_src.tolist(), cc_src.tolist(), strict=False):
        rr_t = place_r0 + rr
        cc_t = place_c0 + cc
        if not (0 <= rr_t < out.shape[0] and 0 <= cc_t < out.shape[1]):
            continue
        if rr_t == mid_r or cc_t == mid_c:
            continue
        if out[rr_t, cc_t] == RED:
            continue
        out[rr_t, cc_t] = int(patch[rr, cc])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Exchange every quadrant's anchor-attached motif with the diagonally opposite one,
    while rescaling to the destination anchor size and erasing the source decoration.

    Input structure:
    - A full-row/full-column divider splits the grid into four quadrants.
    - Each quadrant contains exactly one red square anchor block.
    - Unlike the original task, every quadrant already has one 4-connected non-bg motif
      attached to its own red anchor.

    Rule:
    - Treat each quadrant's anchor-containing component as a local source motif.
    - Move that motif to the diagonally opposite quadrant: reflect it in both axes,
      resize it by (opposite_anchor_size / source_anchor_size), and align its red
      sub-block to the opposite anchor.
    - Before stamping incoming motifs, erase every source quadrant's non-red decoration
      but keep the red anchors in place. The result is a simultaneous diagonal swap.

    Key insight:
    - No quadrant is privileged anymore. The stable global pattern is that diagonal
      quadrants exchange decorations, and anchor size still encodes the scale factor.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)
    anchors = _extract_anchor_data(g, mid_r, mid_c)

    patches: dict[tuple[int, int], np.ndarray] = {}
    patch_boxes: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for key in anchors:
        patch, bb = _extract_quadrant_patch(g, bg, h, w, mid_r, mid_c, key)
        patches[key] = patch
        patch_boxes[key] = bb

    out = g.copy()

    # Move semantics: clear every quadrant's local decoration but preserve red anchors.
    for key, patch in patches.items():
        q_r0, _q_r1, q_c0, _q_c1 = _quadrant_bounds(h, w, mid_r, mid_c, key)
        pr0, pc0, _pr1, _pc1 = patch_boxes[key]
        rr, cc = np.where((patch != bg) & (patch != RED))
        for dr, dc in zip(rr.tolist(), cc.tolist(), strict=False):
            out[q_r0 + pr0 + dr, q_c0 + pc0 + dc] = bg

    # Each quadrant sends its motif to the diagonally opposite anchor.
    for key, patch in patches.items():
        source_size = anchors[key][2]
        target_key = OPPOSITE[key]
        target_r0, target_c0, target_size = anchors[target_key]
        moved = _transform_to_opposite(patch, source_size, target_size)
        _stamp_aligned(out, moved, target_r0, target_c0, mid_r, mid_c, bg)

    return out


def _grow_4conn(h: int, w: int, seeds: set[tuple[int, int]], n_fill: int) -> set[tuple[int, int]]:
    filled = set(seeds)
    frontier = set()
    for r, c in seeds:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in filled:
                frontier.add((nr, nc))
    while len(filled) < n_fill and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        filled.add(cell)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = cell[0] + dr, cell[1] + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in filled:
                frontier.add((nr, nc))
    return filled


def _assign_logical_colors(
    filled: set[tuple[int, int]],
    red_cell: tuple[int, int],
    h: int,
    w: int,
) -> np.ndarray | None:
    rows = {r for r, _ in filled}
    cols = {c for _, c in filled}
    if rows != set(range(h)) or cols != set(range(w)):
        return None
    non_red = [p for p in filled if p != red_cell]
    if len(non_red) < 2:
        return None
    max_colors = min(MOTIF_COLORS_MAX, len(non_red) + 1)
    if max_colors < MOTIF_COLORS_MIN:
        return None
    n_colors = random.randint(MOTIF_COLORS_MIN, max_colors)
    extra_colors = random.sample(COLOR_POOL, n_colors - 1)
    random.shuffle(non_red)
    logical = np.full((h, w), BG, dtype=int)
    logical[red_cell] = RED
    for idx, (r, c) in enumerate(non_red):
        logical[r, c] = extra_colors[idx] if idx < len(extra_colors) else random.choice(extra_colors)
    return logical


def _make_logical_motif() -> np.ndarray | None:
    h, w = random.choice(LOGICAL_DIMS)
    n_cells = h * w
    min_fill = max(3, int(np.ceil(FILL_RATIO_MIN * n_cells)))
    max_fill = max(min_fill, int(np.floor(FILL_RATIO_MAX * n_cells)))
    ar, ac = random.randrange(h), random.randrange(w)
    filled = _grow_4conn(h, w, {(ar, ac)}, random.randint(min_fill, max_fill))
    logical = _assign_logical_colors(filled, (ar, ac), h, w)
    if logical is None:
        return None
    return logical


def _expand_logical(logical: np.ndarray, size: int) -> np.ndarray:
    return np.repeat(np.repeat(logical, size, axis=0), size, axis=1)


def _choose_anchor_with_fit(
    key: tuple[int, int],
    size: int,
    patches: list[np.ndarray],
) -> tuple[int, int] | None:
    q_r0, q_r1, q_c0, q_c1 = QUADS[key]
    r_lo, r_hi = q_r0, q_r1 - size + 1
    c_lo, c_hi = q_c0, q_c1 - size + 1
    for patch in patches:
        red_r0, red_c0 = _red_offset(patch)
        fit_r0 = q_r0 + red_r0
        fit_r1 = q_r1 - patch.shape[0] + 1 + red_r0
        fit_c0 = q_c0 + red_c0
        fit_c1 = q_c1 - patch.shape[1] + 1 + red_c0
        r_lo = max(r_lo, fit_r0)
        r_hi = min(r_hi, fit_r1)
        c_lo = max(c_lo, fit_c0)
        c_hi = min(c_hi, fit_c1)
    if r_lo > r_hi or c_lo > c_hi:
        return None
    return random.randint(r_lo, r_hi), random.randint(c_lo, c_hi)


def _anchor_to_patch_origin(patch: np.ndarray, anchor_r: int, anchor_c: int) -> tuple[int, int]:
    red_r0, red_c0 = _red_offset(patch)
    return anchor_r - red_r0, anchor_c - red_c0


def _gen_attempt() -> np.ndarray | None:
    sizes = {key: random.randint(*SIZE_RANGE) for key in QUADS}
    if sizes[(0, 0)] == sizes[(1, 1)] and sizes[(0, 1)] == sizes[(1, 0)]:
        if random.random() < 0.7:
            return None

    logicals: dict[tuple[int, int], np.ndarray] = {}
    patches: dict[tuple[int, int], np.ndarray] = {}
    for key, size in sizes.items():
        logical = _make_logical_motif()
        if logical is None:
            return None
        logicals[key] = logical
        patches[key] = _expand_logical(logical, size)

    # Reject invisible swaps: a quadrant must not already equal the incoming diagonal motif.
    for key, opposite in (((0, 0), (1, 1)), ((0, 1), (1, 0))):
        if logicals[key].shape == logicals[opposite].shape:
            if np.array_equal(logicals[key], _flip_diagonal(logicals[opposite])):
                return None

    incoming = {
        key: _transform_to_opposite(patches[OPPOSITE[key]], sizes[OPPOSITE[key]], sizes[key])
        for key in QUADS
    }

    anchors: dict[tuple[int, int], tuple[int, int]] = {}
    for key in QUADS:
        anchor = _choose_anchor_with_fit(key, sizes[key], [patches[key], incoming[key]])
        if anchor is None:
            return None
        anchors[key] = anchor

    grid = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    grid[MID, :] = DIV
    grid[:, MID] = DIV

    for key in QUADS:
        patch = patches[key]
        ar, ac = anchors[key]
        pr0, pc0 = _anchor_to_patch_origin(patch, ar, ac)
        paste(grid, patch, pr0, pc0)

    # Ensure every quadrant contains exactly one anchor-attached component.
    try:
        out = main(grid)
    except Exception:
        return None

    # Additional non-degeneracy: all four quadrants should visibly change after swapping.
    for key in QUADS:
        q_r0, q_r1, q_c0, q_c1 = QUADS[key]
        before = grid[q_r0 : q_r1 + 1, q_c0 : q_c1 + 1]
        after = out[q_r0 : q_r1 + 1, q_c0 : q_c1 + 1]
        if np.array_equal(before, after):
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fused b10624e5 input")
