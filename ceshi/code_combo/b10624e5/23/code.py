# Combo variant of b10624e5 (index 23)
# combo_spec: D2/3+D14/2
# Name: 分散像素组合源图案 + 目标预填充噪声

from pathlib import Path
import random
import sys

import numpy as np

try:
    from arc_utils import bbox_coords, components_mask, get_bg, paste
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[4]))
    from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis for combo D2/3 + D14/2
# =======================================
# a) Concrete behavior trace:
# - Original task: TL has one connected motif attached to the red anchor; other quadrants contain
#   only red anchors. Solver finds the connected TL component, then reflects/scales/stamps it.
# - Combo task: TL no longer shows a connected motif. The true motif is a logical block pattern,
#   but every non-red logical cell is rendered as a single pixel on a 2-step lattice anchored by
#   the 2x2 red block. Extra off-lattice pixels are source-side noise. The solver must:
#   1. use the TL red 2x2 anchor as the lattice unit,
#   2. keep only non-red pixels whose row and col offsets from the anchor are multiples of 2,
#   3. rebuild the dense source patch by expanding each surviving logical cell back to 2x2.
# - Target quadrants also change: the input may already contain colored noise pixels around the
#   red anchors. Those pixels do not belong to the true motif. The solver must clear the entire
#   target quadrant down to background except for the red anchor, then stamp the reflected/scaled
#   motif. This creates a visible "clear noise then fill motif" effect.
# - Small example:
#   original TL: a 6x6 dense pattern touching the anchor.
#   combo TL input: the same 3x3 logical pattern shown as a 6x6 sparse lattice:
#       red cell -> full 2x2 anchor block,
#       green cell -> one green pixel at that cell's lattice corner,
#       pink cell -> one pink pixel at its lattice corner,
#       off-lattice orange speck -> ignore as noise.
#   combo target input: a 3x3 red anchor plus a few stray teal/maroon pixels nearby.
#   combo output: target noise disappears, and a clean reflected+scaled dense pattern appears.
#
# b) Complexity calibration:
# - The source remains readable because all true pixels sit on one obvious 2-step lattice tied to
#   the red anchor. Noise is limited to 1-2 off-lattice specks, so the denoising cue is visible
#   without becoming arbitrary.
# - Logical motif size is limited to 2x2, 2x3, 3x2, or 3x3. This keeps scaled targets within a
#   13x13 quadrant even at anchor size 4, while still producing visibly scattered source pixels.
# - At least two target quadrants receive noise, and each noisy quadrant gets at least one pixel
#   that should become background in the output. That prevents degeneration into "noise happens to
#   be overwritten anyway".
#
# c) Implicit-assumption audit against the original solver:
# - Original "source is one connected non-bg component touching red" is false now. The new solver
#   cannot rely on connected components in TL at all.
# - Original "copy by overlaying non-bg pixels onto existing target quadrant" is false now because
#   target quadrants can contain junk pixels that must be erased even where the true motif is bg.
# - Original "all informative pixels are already dense" is false. The new solver assumes the TL
#   anchor is 2x2 and uses it as the lattice period for sparse-to-dense reconstruction.
# - New assumption review:
#   * every quadrant still has exactly one red square anchor -> generator enforces this.
#   * TL anchor size is exactly 2 -> generator fixes it.
#   * true TL motif pixels (except the anchor) lie on the anchor-defined 2-step lattice ->
#     generator renders them that way.
#   * source-side noise is off-lattice -> generator forbids on-lattice source noise.
#   * target noise is non-red and lives only in non-TL quadrants -> generator enforces this so
#     anchor detection stays unambiguous.
#
# d) Placement legality:
# - Dense source patches are generated from logical motifs and therefore have sizes 4, 6, or 6x4.
# - Target anchor placement uses the scaled dense patch shape and chooses positions where the whole
#   transformed motif fits inside the quadrant. This avoids clipping and prevents overlap with the
#   divider.
# - Target noise is added only to non-red cells inside the quadrant, so it never merges with the
#   anchor component or crosses the divider.
#
# e) Visual observability:
# - Across 2-4 samples a human can see the same scattered lattice around TL anchors and the same
#   type of stray target specks that disappear in output.
# - Reflections remain visible because source motifs are forced to be asymmetric under LR and UD.
# - Scaling is still explained by anchor size, so the new difficulty comes from denoising and
#   assembly, not from hidden numeric rules.
#
# f) Visual cleanliness:
# - Source noise is sparse and off-lattice, so it does not create ambiguous connected blobs.
# - Target noise is limited to a few pixels per quadrant; quadrants remain easy to parse.
# - Colors are copied verbatim. No color-id arithmetic or invisible ordering is used.
#
# g) Fusion metrics worth checking:
# - source_off_lattice_noise > 0 on most/all samples,
# - target_noise_pixels > 0 in at least two target quadrants,
# - target_noise_cleared_to_bg > 0 so the new solver must erase, not just overwrite.


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

SOURCE_ANCHOR_SIZE = 2
BLOCK_DIMS = [(2, 2), (2, 3), (3, 2), (3, 3)]
FILL_RATIO_MIN = 0.55
FILL_RATIO_MAX = 0.85
MOTIF_COLORS_MIN = 3
MOTIF_COLORS_MAX = 5
COLOR_POOL = [0, 3, 5, 6, 7, 8, 9]
TARGET_SIZE_RANGE = (1, 4)
TARGET_NOISY_QUADRANTS = (2, 3)
TARGET_CLEAR_NOISE_RANGE = (1, 3)
TARGET_COVER_NOISE_RANGE = (0, 2)
SOURCE_NOISE_RANGE = (1, 2)
MAX_RETRIES = 500


def _find_dividers(g: np.ndarray, bg: int) -> tuple[int, int]:
    """Return the unique full-row/full-col divider crossing."""
    h, w = g.shape
    full_rows = [(r, int(g[r, 0])) for r in range(h) if np.all(g[r, :] == g[r, 0]) and int(g[r, 0]) != bg]
    full_cols = [(c, int(g[0, c])) for c in range(w) if np.all(g[:, c] == g[0, c]) and int(g[0, c]) != bg]
    matches = []
    for r, vr in full_rows:
        for c, vc in full_cols:
            if vr == vc:
                matches.append((int(r), int(c)))
    assert len(matches) == 1, "grid must have exactly one matching divider row/col pair"
    return matches[0]


def _resize_nn(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    src_h, src_w = img.shape
    assert new_h > 0 and new_w > 0
    rr = np.clip(np.floor(np.arange(new_h) * src_h / new_h).astype(int), 0, src_h - 1)
    cc = np.clip(np.floor(np.arange(new_w) * src_w / new_w).astype(int), 0, src_w - 1)
    return img[rr[:, None], cc[None, :]]


def _reflect_patch(patch: np.ndarray, qb: int, qr: int) -> np.ndarray:
    out = patch
    if qr:
        out = np.fliplr(out)
    if qb:
        out = np.flipud(out)
    return out


def _scaled_patch(patch: np.ndarray, qb: int, qr: int, target_size: int) -> np.ndarray:
    patch_t = _reflect_patch(patch, qb, qr)
    ratio = target_size / SOURCE_ANCHOR_SIZE
    new_h = max(1, int(round(patch_t.shape[0] * ratio)))
    new_w = max(1, int(round(patch_t.shape[1] * ratio)))
    patch_s = _resize_nn(patch_t, new_h, new_w)
    roundtrip = _resize_nn(patch_s, patch_t.shape[0], patch_t.shape[1])
    assert np.array_equal(roundtrip, patch_t), "scaled motif must remain lossless under roundtrip resize"
    return patch_s


def _grow_4conn(h: int, w: int, seeds: set[tuple[int, int]], n_fill: int) -> set[tuple[int, int]]:
    filled = set(seeds)
    frontier = set()
    for r, c in seeds:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in filled:
                frontier.add((nr, nc))
    while len(filled) < n_fill and frontier:
        cell = random.choice(list(frontier))
        frontier.discard(cell)
        filled.add(cell)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = cell[0] + dr, cell[1] + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in filled:
                frontier.add((nr, nc))
    return filled


def _assign_logical_colors(
    filled: set[tuple[int, int]],
    anchor_cell: tuple[int, int],
    h: int,
    w: int,
) -> np.ndarray | None:
    rows = {r for r, _ in filled}
    cols = {c for _, c in filled}
    if rows != set(range(h)) or cols != set(range(w)):
        return None
    non_red = [p for p in filled if p != anchor_cell]
    random.shuffle(non_red)
    max_colors = min(MOTIF_COLORS_MAX, len(non_red) + 1)
    if max_colors < MOTIF_COLORS_MIN or len(non_red) < 2:
        return None
    n_colors = random.randint(MOTIF_COLORS_MIN, max_colors)
    palette = random.sample(COLOR_POOL, n_colors - 1)
    logical = np.full((h, w), BG, dtype=int)
    logical[anchor_cell] = RED
    for idx, cell in enumerate(non_red):
        logical[cell] = palette[idx] if idx < len(palette) else random.choice(palette)
    return logical


def _sparse_source_from_logical(logical: np.ndarray, anchor_cell: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    dense = np.repeat(np.repeat(logical, SOURCE_ANCHOR_SIZE, axis=0), SOURCE_ANCHOR_SIZE, axis=1)
    sparse = np.full_like(dense, BG)
    for r in range(logical.shape[0]):
        for c in range(logical.shape[1]):
            val = int(logical[r, c])
            if val == BG:
                continue
            rr = r * SOURCE_ANCHOR_SIZE
            cc = c * SOURCE_ANCHOR_SIZE
            if (r, c) == anchor_cell:
                sparse[rr : rr + SOURCE_ANCHOR_SIZE, cc : cc + SOURCE_ANCHOR_SIZE] = RED
            else:
                sparse[rr, cc] = val
    return dense, sparse


def _add_source_noise(sparse: np.ndarray, motif_colors: list[int]) -> int:
    real_mask = sparse != BG
    blocked = real_mask.copy()
    for r, c in np.argwhere(real_mask):
        for dr, dc in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = int(r + dr), int(c + dc)
            if 0 <= nr < sparse.shape[0] and 0 <= nc < sparse.shape[1]:
                blocked[nr, nc] = True
    candidates = [(int(r), int(c)) for r, c in np.argwhere(~blocked)]
    if not candidates:
        return 0
    n_noise = min(len(candidates), random.randint(*SOURCE_NOISE_RANGE))
    for r, c in random.sample(candidates, n_noise):
        sparse[r, c] = random.choice(motif_colors)
    return n_noise


def _sample_source() -> tuple[np.ndarray, np.ndarray, int, int, int]:
    while True:
        lh, lw = random.choice(BLOCK_DIMS)
        n_cells = lh * lw
        min_fill = max(3, int(np.ceil(FILL_RATIO_MIN * n_cells)))
        max_fill = max(min_fill, int(np.floor(FILL_RATIO_MAX * n_cells)))
        anchor_cell = (random.randrange(lh), random.randrange(lw))
        filled = _grow_4conn(lh, lw, {anchor_cell}, random.randint(min_fill, max_fill))
        logical = _assign_logical_colors(filled, anchor_cell, lh, lw)
        if logical is None:
            continue
        dense, sparse = _sparse_source_from_logical(logical, anchor_cell)
        if np.array_equal(logical, np.fliplr(logical)):
            continue
        if np.array_equal(logical, np.flipud(logical)):
            continue
        motif_colors = sorted({int(v) for v in logical.flatten() if v not in (BG, RED)})
        if len(motif_colors) < 2:
            continue
        n_noise = _add_source_noise(sparse, motif_colors)
        if n_noise == 0:
            continue
        return dense, sparse, anchor_cell[0] * SOURCE_ANCHOR_SIZE, anchor_cell[1] * SOURCE_ANCHOR_SIZE, n_noise


def _choose_anchor(qb: int, qr: int, size: int, patch_s: np.ndarray) -> tuple[int, int]:
    q_r0, q_r1, q_c0, q_c1 = QUADS[(qb, qr)]
    rr, cc = np.where(patch_s == RED)
    red_r0, red_c0 = int(rr.min()), int(cc.min())
    fit_r0 = q_r0 + red_r0
    fit_r1 = q_r1 - patch_s.shape[0] + 1 + red_r0
    fit_c0 = q_c0 + red_c0
    fit_c1 = q_c1 - patch_s.shape[1] + 1 + red_c0
    return random.randint(fit_r0, fit_r1), random.randint(fit_c0, fit_c1)


def _stamp(out: np.ndarray, patch_s: np.ndarray, anchor_r: int, anchor_c: int, bg: int) -> None:
    rr_red, cc_red = np.where(patch_s == RED)
    place_r0 = anchor_r - int(rr_red.min())
    place_c0 = anchor_c - int(cc_red.min())
    rr, cc = np.where(patch_s != bg)
    for sr, sc in zip(rr.tolist(), cc.tolist(), strict=False):
        tr = place_r0 + sr
        tc = place_c0 + sc
        val = int(patch_s[sr, sc])
        if val == RED:
            continue
        out[tr, tc] = val


def _anchors_by_quadrant(g: np.ndarray, mid_r: int, mid_c: int) -> dict[tuple[int, int], tuple[int, int, int]]:
    anchors: dict[tuple[int, int], tuple[int, int, int]] = {}
    for pts in components_mask(g == RED, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts)
        side_r = r1 - r0 + 1
        side_c = c1 - c0 + 1
        assert side_r == side_c, "red anchors must be square"
        rr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        key = (0 if rr < mid_r else 1, 0 if cc < mid_c else 1)
        assert key not in anchors, "each quadrant must contain exactly one red anchor"
        anchors[key] = (int(r0), int(c0), int(side_r))
    assert set(anchors) == set(QUADS), "must detect one red anchor in every quadrant"
    return anchors


def _reconstruct_sparse_source(g: np.ndarray, anchor: tuple[int, int, int], mid_r: int, mid_c: int, bg: int) -> np.ndarray:
    sr0, sc0, sL = anchor
    assert sL == SOURCE_ANCHOR_SIZE, "combo source assumes a 2x2 top-left anchor"
    tl = g[0:mid_r, 0:mid_c]
    logical_samples: dict[tuple[int, int], int] = {(0, 0): RED}
    for r, c in np.argwhere(tl != bg):
        rr = int(r)
        cc = int(c)
        if sr0 <= rr < sr0 + sL and sc0 <= cc < sc0 + sL:
            continue
        dr = rr - sr0
        dc = cc - sc0
        if dr % sL != 0 or dc % sL != 0:
            continue
        logical_samples[(dr // sL, dc // sL)] = int(tl[rr, cc])

    assert len(logical_samples) > 1, "source must contain lattice-aligned motif pixels beyond the anchor"
    min_lr = min(k[0] for k in logical_samples)
    max_lr = max(k[0] for k in logical_samples)
    min_lc = min(k[1] for k in logical_samples)
    max_lc = max(k[1] for k in logical_samples)
    logical = np.full((max_lr - min_lr + 1, max_lc - min_lc + 1), bg, dtype=int)
    for (lr, lc), val in logical_samples.items():
        logical[lr - min_lr, lc - min_lc] = val
    return np.repeat(np.repeat(logical, sL, axis=0), sL, axis=1)


def _target_noise_positions(inp: np.ndarray, out: np.ndarray, key: tuple[int, int], anchor: tuple[int, int, int]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    q_r0, q_r1, q_c0, q_c1 = QUADS[key]
    ar, ac, size = anchor
    anchor_mask = np.zeros_like(inp, dtype=bool)
    anchor_mask[ar : ar + size, ac : ac + size] = True
    clear_slots: list[tuple[int, int]] = []
    cover_slots: list[tuple[int, int]] = []
    for r in range(q_r0, q_r1 + 1):
        for c in range(q_c0, q_c1 + 1):
            if anchor_mask[r, c]:
                continue
            if int(out[r, c]) == BG:
                clear_slots.append((r, c))
            elif int(out[r, c]) != RED:
                cover_slots.append((r, c))
    return clear_slots, cover_slots


def _add_target_noise(inp: np.ndarray, out: np.ndarray, key: tuple[int, int], anchor: tuple[int, int, int]) -> int:
    clear_slots, cover_slots = _target_noise_positions(inp, out, key, anchor)
    if not clear_slots:
        return 0
    n_clear = min(len(clear_slots), random.randint(*TARGET_CLEAR_NOISE_RANGE))
    n_cover = min(len(cover_slots), random.randint(*TARGET_COVER_NOISE_RANGE))
    placed = 0
    for r, c in random.sample(clear_slots, n_clear):
        inp[r, c] = random.choice(COLOR_POOL)
        placed += 1
    if n_cover > 0:
        for r, c in random.sample(cover_slots, n_cover):
            choices = [col for col in COLOR_POOL if col != int(out[r, c])]
            inp[r, c] = random.choice(choices)
            placed += 1
    return placed


def _sample_target_sizes() -> dict[tuple[int, int], int]:
    while True:
        sizes = {
            (0, 0): SOURCE_ANCHOR_SIZE,
            (0, 1): random.randint(*TARGET_SIZE_RANGE),
            (1, 0): random.randint(*TARGET_SIZE_RANGE),
            (1, 1): random.randint(*TARGET_SIZE_RANGE),
        }
        non_tl = [sizes[(0, 1)], sizes[(1, 0)], sizes[(1, 1)]]
        if len(set(non_tl)) == 1:
            continue
        if all(size == SOURCE_ANCHOR_SIZE for size in non_tl):
            continue
        return sizes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy a motif from a sparse top-left lattice to the other quadrants, while removing target-side noise.

    Input structure:
    - The 27x27 grid is split into four 13x13 quadrants by one full divider row and one full divider column.
    - Every quadrant contains one red square anchor; the top-left anchor is always 2x2.
    - In the top-left quadrant, the true source motif is not dense: non-red motif cells appear as isolated
      lattice pixels spaced by the anchor size, and a few off-lattice pixels may be irrelevant noise.
    - In the other quadrants, the red anchors are already present and some quadrants may also contain
      stray colored pixels that do not belong to the final motif.

    Rule:
    - Use the top-left red anchor as the lattice unit, keep only top-left non-red pixels aligned to that
      2-step lattice, and rebuild the dense source patch by expanding each logical cell back to 2x2.
    - For each non-top-left quadrant, reflect the rebuilt patch left/right for right-side quadrants and
      up/down for bottom quadrants, then scale it by the target-anchor-size / 2 ratio.
    - Before stamping the transformed motif into a target quadrant, erase that quadrant to clean away
      prefilled noise while preserving the red anchor. Then align the transformed red region to the anchor
      and draw the non-red motif pixels.

    Key insight:
    - The sparse pixels near the top-left anchor are samples of a hidden block motif, not the motif itself.
      The anchor gives the sampling stride, quadrant location gives the reflection, anchor size gives the
      scaling, and any extra target-side colored specks are disposable noise.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()
    mid_r, mid_c = _find_dividers(g, bg)
    anchors = _anchors_by_quadrant(g, mid_r, mid_c)

    src_patch = _reconstruct_sparse_source(g, anchors[(0, 0)], mid_r, mid_c, bg)

    for key, (tr0, tc0, tL) in anchors.items():
        if key == (0, 0):
            continue
        q_r0, q_r1, q_c0, q_c1 = QUADS[key]
        out[q_r0 : q_r1 + 1, q_c0 : q_c1 + 1] = bg
        paste(out, np.full((tL, tL), RED, dtype=int), tr0, tc0)

        patch_s = _scaled_patch(src_patch, key[0], key[1], tL)
        _stamp(out, patch_s, tr0, tc0, bg)

    out[mid_r, :] = DIV
    out[:, mid_c] = DIV
    return out


def _gen_attempt() -> np.ndarray | None:
    dense_src, sparse_src, src_ar, src_ac, _src_noise = _sample_source()
    target_sizes = _sample_target_sizes()
    scaled = {
        key: _scaled_patch(dense_src, key[0], key[1], size)
        for key, size in target_sizes.items()
        if key != (0, 0)
    }

    anchors: dict[tuple[int, int], tuple[int, int, int]] = {}
    tl_top = random.randint(0, MID - sparse_src.shape[0])
    tl_left = random.randint(0, MID - sparse_src.shape[1])
    anchors[(0, 0)] = (tl_top + src_ar, tl_left + src_ac, SOURCE_ANCHOR_SIZE)
    for key, patch_s in scaled.items():
        ar, ac = _choose_anchor(key[0], key[1], target_sizes[key], patch_s)
        anchors[key] = (ar, ac, target_sizes[key])

    out = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    out[MID, :] = DIV
    out[:, MID] = DIV
    paste(out, sparse_src, tl_top, tl_left)
    for key, (ar, ac, size) in anchors.items():
        if key == (0, 0):
            continue
        paste(out, np.full((size, size), RED, dtype=int), ar, ac)
    for key, patch_s in scaled.items():
        ar, ac, _size = anchors[key]
        _stamp(out, patch_s, ar, ac, BG)

    inp = out.copy()
    noisy_quadrants = random.sample([(0, 1), (1, 0), (1, 1)], random.randint(*TARGET_NOISY_QUADRANTS))
    total_target_noise = 0
    for key, (ar, ac, size) in anchors.items():
        if key == (0, 0):
            continue
        q_r0, q_r1, q_c0, q_c1 = QUADS[key]
        inp[q_r0 : q_r1 + 1, q_c0 : q_c1 + 1] = BG
        paste(inp, np.full((size, size), RED, dtype=int), ar, ac)
        if key in noisy_quadrants:
            total_target_noise += _add_target_noise(inp, out, key, anchors[key])

    if total_target_noise == 0:
        return None
    if np.array_equal(main(inp), out):
        metrics = variant_metrics(inp)
        if metrics["source_off_lattice_noise"] > 0 and metrics["noise_quadrants"] >= 2 and metrics["target_noise_cleared_to_bg"] > 0:
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid combo input")


def variant_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)
    anchors = _anchors_by_quadrant(g, mid_r, mid_c)
    sr0, sc0, sL = anchors[(0, 0)]
    tl = g[0:mid_r, 0:mid_c]

    source_signal = 0
    source_noise = 0
    for r, c in np.argwhere(tl != bg):
        rr = int(r)
        cc = int(c)
        if sr0 <= rr < sr0 + sL and sc0 <= cc < sc0 + sL:
            continue
        if (rr - sr0) % sL == 0 and (cc - sc0) % sL == 0:
            source_signal += 1
        else:
            source_noise += 1

    out = main(g)
    target_noise = 0
    noise_quadrants = 0
    cleared_to_bg = 0
    for key, (ar, ac, size) in anchors.items():
        if key == (0, 0):
            continue
        q_r0, q_r1, q_c0, q_c1 = QUADS[key]
        local_noise = 0
        for r in range(q_r0, q_r1 + 1):
            for c in range(q_c0, q_c1 + 1):
                if ar <= r < ar + size and ac <= c < ac + size:
                    continue
                if int(g[r, c]) != bg:
                    target_noise += 1
                    local_noise += 1
                    if int(out[r, c]) == bg:
                        cleared_to_bg += 1
        if local_noise > 0:
            noise_quadrants += 1

    return {
        "source_signal_pixels": source_signal,
        "source_off_lattice_noise": source_noise,
        "target_noise_pixels": target_noise,
        "noise_quadrants": noise_quadrants,
        "target_noise_cleared_to_bg": cleared_to_bg,
    }
