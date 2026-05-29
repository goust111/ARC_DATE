# Variant of 5dbc8537
# Dimension: D3 — 目标洞的结构与分布 (dst)
# Change: 模板框架内的矩形洞区域，所有洞形成单个4-连通的连续背景色区域，洞总面积精确等于所有精灵面积之和（完美拼贴）... → 多框架分布

"""
Design analysis for the D3 variant
==================================

a) Concrete behavior and flow trace
-----------------------------------
- The original task had one long base-colored strip whose entire cavity region lived inside one bbox.
- This variant keeps the same local mechanic, "rectangular sprite bbox must match a rectangular hole bbox",
  but splits the destination into 2-3 separate small frames of the same base color.
- Each frame is a small square block with a 1-cell base border and 2-3 bg rectangular holes inside.
- The frames are arranged in a single row or a single column with bg gaps between them.
- Output is the tight crop of the whole frame arrangement, so bg gaps between frames remain visible.
- Input additionally contains the filled rectangles scattered outside the arrangement. The key change is that
  sprites belonging to frame i are scattered only in the narrow band aligned with that frame:
  same columns for a horizontal frame row, or same rows for a vertical frame column.

- Small trace:
  original logic:
  1. Find one large base bbox.
  2. Treat every non-base cell inside that bbox as a candidate part of one global cavity mass.
  3. Exact-cover all sprites into that one mass.

  new logic:
  1. Find the shared base color and split it into several disconnected frame components.
  2. For each outside sprite, first determine which frame band it belongs to.
  3. Inside that frame only, match sprite size to one local rectangular hole.

  difference:
  - The old solver could ignore frame ownership because there was only one destination region.
  - The new solver must preserve the extra grouping layer; two different frames can contain holes of the same size.

b) Complexity calibration
-------------------------
- The variant is not allowed to collapse back into the original single-frame case, so every sample has 2 or 3 frames.
- To make the new behavior visible and logically necessary, at least one hole size is duplicated across different frames.
- Within one frame, hole sizes stay unique. That avoids local ambiguity and keeps the puzzle focused on the new D3 change:
  first choose the right frame, then fill the frame's holes.
- Sprites are still simple axis-aligned filled rectangles with 1 or 2 colors, so the task remains close to the source task.

c) Implicit-assumption audit of the original main()
---------------------------------------------------
- Original base-color bbox extraction:
  `base_coords = np.argwhere(g == base_col)` still works for the union bbox of all frames.
  What breaks is the next assumption that every non-base cell inside that union bbox is a fill target.
- Original `hole_mask = dst != base_col`:
  invalid now, because bg gaps between separate frames would be misread as holes.
- Original sprite placement search over the whole cropped bbox:
  invalid now, because equal-size holes can exist in different frames and frame ownership matters.
- Original `sum(sprite areas) == hole_mask.sum()`:
  invalid at the global bbox level, since the union bbox now contains intentional bg gaps between frames.
- Safe replacement:
  detect each base connected component as one frame, identify its own hole rectangles, assign sprites to frames by
  band alignment, then match sizes locally inside that frame.

d) Placement legality
---------------------
- Frames never overlap and are separated by a 2-cell bg gap.
- Holes are placed strictly inside the 1-cell frame border and remain separated by at least one base cell.
- Sprites are placed outside the entire frame arrangement bbox with 1-cell padding from frames and other sprites.
- The generator oracle-checks `main(input) == expected_output` on every accepted sample.

e) Visual observability
-----------------------
- Multiple disconnected destination frames are immediately visible in the input and in the output crop.
- The aligned sprite bands make frame affiliation human-visible without relying on hidden metadata.
- Repeated hole sizes across different frames prevent the samples from being explainable as one global cavity again.

f) Visual cleanliness
---------------------
- The arrangement is only one row or one column, so relative structure is easy to read.
- Sprites are kept out of the crop region and out of the inter-frame gaps, avoiding clutter.
- Each frame has only 2-3 holes, which keeps the local structure legible.

g) Color inferability
---------------------
- Colors are not used as a sorting key or as an invisible code.
- Sprite colors only decorate pieces; matching is driven by visible rectangle size plus visible frame-band grouping.
- Base color is shared by all frames and is made dominant by count so the solver can recover it reliably.

h) New solver assumptions and generator guarantees
--------------------------------------------------
- Assumption: every base-colored connected component is a real frame.
  Guarantee: generator never uses the base color inside sprites.
- Assumption: a sprite overlaps the alignment interval of exactly one frame.
  Guarantee: each frame's sprites are placed only inside that frame's own row/column band.
- Assumption: hole sizes are unique within one frame.
  Guarantee: generator samples a set, not a multiset, of rectangle sizes per frame.
- Assumption: some sizes repeat across frames, otherwise the extra grouping stage would be unnecessary.
  Guarantee: frames 0 and 1 always share one repeated rectangle size.
- Assumption: the shared base color is uniquely recoverable.
  Guarantee: accepted samples are rejected if any non-base color count reaches or exceeds the base count.
"""

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, colors_present, components_mask, crop_bbox, find_components, get_bg, paste


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill several separated frame templates by routing each outside sprite to the correct frame and hole.

    Input structure:
    - bg is the most common color. One shared base color forms multiple disconnected square frames, each with rectangular bg holes.
    - Filled rectangular sprites sit outside the frame arrangement, but each sprite is aligned with one frame's row/column band.

    Rule:
    - Recover the base color, split it into individual frame components, and crop the union bbox that contains the whole arrangement.
    - Assign every outside sprite to the frame whose band it aligns with; then match sprite bbox size to one local hole inside that frame.
    - Paste the sprite into that matching hole and keep bg only in the gaps between different frames.

    Key insight: Size matching is still local jigsaw fitting, but it now happens after a frame-grouping step because repeated hole sizes can appear in different frames.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # Base color = dominant non-bg color by total area across all its disconnected frame pieces.
    base_candidates: list[tuple[int, int, int]] = []
    for col in colors_present(g, exclude={bg}):
        comps = components_mask(g == int(col), connectivity=8)
        if not comps:
            continue
        total_area = sum(int(coords.shape[0]) for coords in comps)
        max_area = max(int(coords.shape[0]) for coords in comps)
        base_candidates.append((int(total_area), int(max_area), int(col)))

    assert base_candidates
    base_candidates.sort(reverse=True)
    if len(base_candidates) > 1:
        assert base_candidates[0][0] > base_candidates[1][0]
    base_col = int(base_candidates[0][2])

    frame_coords = components_mask(g == int(base_col), connectivity=8)
    assert len(frame_coords) >= 2

    all_base = np.argwhere(g == int(base_col))
    ur0, uc0, ur1, uc1 = bbox_coords(all_base)
    out = np.full((int(ur1 - ur0 + 1), int(uc1 - uc0 + 1)), int(bg), dtype=int)

    frames: list[dict] = []
    for coords in frame_coords:
        fr0, fc0, fr1, fc1 = bbox_coords(coords)
        patch = crop_bbox(g, (int(fr0), int(fc0), int(fr1), int(fc1)))
        base_mask = patch == int(base_col)
        hole_mask = patch == int(bg)
        assert bool(hole_mask.any())

        hole_rects: list[tuple[int, int, int, int]] = []
        seen_sizes: set[tuple[int, int]] = set()
        for hole in components_mask(hole_mask, connectivity=4):
            hr0, hc0, hr1, hc1 = bbox_coords(hole)
            h = int(hr1 - hr0 + 1)
            w = int(hc1 - hc0 + 1)
            assert int(hole.shape[0]) == int(h * w)
            size = (int(h), int(w))
            assert size not in seen_sizes
            seen_sizes.add(size)
            hole_rects.append((int(hr0), int(hc0), int(h), int(w)))

        base_patch = np.full(patch.shape, int(base_col), dtype=int)
        paste(out, base_patch, int(fr0 - ur0), int(fc0 - uc0), mask=base_mask, overwrite=True)
        frames.append(
            {
                "bbox": (int(fr0), int(fc0), int(fr1), int(fc1)),
                "hole_rects": hole_rects,
            }
        )

    row_spread = max(f["bbox"][0] for f in frames) - min(f["bbox"][0] for f in frames)
    col_spread = max(f["bbox"][1] for f in frames) - min(f["bbox"][1] for f in frames)
    horizontal = bool(col_spread >= row_spread)
    frames.sort(key=lambda f: int(f["bbox"][1] if horizontal else f["bbox"][0]))

    masked = g.copy()
    masked[g == int(base_col)] = int(bg)
    sprite_comps = find_components(masked, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    sprites: list[dict] = []
    for comp in sprite_comps:
        pr0, pc0, pr1, pc1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (int(pr0), int(pc0), int(pr1), int(pc1)))
        ph, pw = map(int, patch.shape)
        mask = np.zeros((ph, pw), dtype=bool)
        coords = comp["coords"]
        mask[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        sprites.append(
            {
                "bbox": (int(pr0), int(pc0), int(pr1), int(pc1)),
                "h": int(ph),
                "w": int(pw),
                "mask": mask,
                "patch": patch,
            }
        )

    assert sprites
    assigned: dict[int, list[dict]] = {i: [] for i in range(len(frames))}
    for sp in sprites:
        pr0, pc0, pr1, pc1 = sp["bbox"]
        best_score: tuple[int, int, int, int] | None = None
        best_idx: int | None = None
        for i, frame in enumerate(frames):
            fr0, fc0, fr1, fc1 = frame["bbox"]
            if horizontal:
                overlap = max(0, min(pc1, fc1) - max(pc0, fc0) + 1)
                band_dist = abs((pc0 + pc1) - (fc0 + fc1))
                orth_dist = min(abs(pr1 - fr0), abs(pr0 - fr1))
            else:
                overlap = max(0, min(pr1, fr1) - max(pr0, fr0) + 1)
                band_dist = abs((pr0 + pr1) - (fr0 + fr1))
                orth_dist = min(abs(pc1 - fc0), abs(pc0 - fc1))
            score = (int(overlap), -int(band_dist), -int(orth_dist), -int(sp["h"] * sp["w"]))
            if best_score is None or score > best_score:
                best_score = score
                best_idx = i
        assert best_idx is not None
        assert best_score is not None and best_score[0] > 0
        assigned[best_idx].append(sp)

    for i, frame in enumerate(frames):
        rects = frame["hole_rects"]
        pieces = assigned[i]
        assert len(pieces) == len(rects)
        by_size: dict[tuple[int, int], dict] = {}
        for sp in pieces:
            size = (int(sp["h"]), int(sp["w"]))
            assert size not in by_size
            by_size[size] = sp
        fr0, fc0, _fr1, _fc1 = frame["bbox"]
        for rr0, cc0, h, w in rects:
            sp = by_size[(int(h), int(w))]
            paste(out, sp["patch"], int(fr0 - ur0 + rr0), int(fc0 - uc0 + cc0), mask=sp["mask"], overwrite=True)

    return out


# --------------- generator ---------------

GRID_MIN, GRID_MAX = 18, 25
FRAME_COUNT_MIN, FRAME_COUNT_MAX = 2, 3
FRAME_SIZE_CHOICES = [6, 7]
FRAME_GAP = 2
OUTER_MARGIN = 4
HOLES_PER_FRAME = [2, 3]
BICHROMATIC_FRAC = 0.4
MAX_RETRIES = 400

RECT_LIBRARY = [
    (1, 2),
    (2, 1),
    (1, 3),
    (3, 1),
    (2, 2),
    (2, 3),
    (3, 2),
]


def _make_patch(h: int, w: int, cols: list[int]) -> np.ndarray:
    """Fill an (h, w) rectangle with 1 or 2 colors using a visible asymmetric pattern."""
    out = np.full((h, w), cols[0], dtype=int)
    if len(cols) == 1:
        return out

    b = int(cols[1])
    pattern = random.choice(["checker", "dots", "split"])
    rr, cc = np.indices((h, w))
    if pattern == "checker":
        out[(rr + cc + random.randint(0, 1)) % 2 == 0] = b
    elif pattern == "dots":
        step = random.choice([2, 3])
        mask = (rr * 2 + cc + random.randint(0, step - 1)) % step == 0
        if not mask.any() or mask.all():
            mask = np.zeros((h, w), dtype=bool)
            mask[random.randrange(h), random.randrange(w)] = True
        out[mask] = b
    elif h > 1 and (w == 1 or random.random() < 0.5):
        out[random.randint(1, h - 1) :, :] = b
    else:
        out[:, random.randint(1, w - 1) :] = b
    return out


def _pack_rectangles_in_frame(size: int, dims: list[tuple[int, int]]) -> list[tuple[int, int, int, int]] | None:
    """Pack rectangles into the inner area of one frame with a 1-cell gap between holes."""
    inner = int(size - 2)
    ordered = sorted([(int(h), int(w)) for h, w in dims], key=lambda x: (-x[0] * x[1], -x[0], -x[1]))
    for _ in range(120):
        occupied = np.zeros((inner, inner), dtype=bool)
        rects: list[tuple[int, int, int, int]] = []
        ok = True
        for h, w in ordered:
            candidates: list[tuple[int, int]] = []
            for r in range(inner - h + 1):
                for c in range(inner - w + 1):
                    rr0 = max(0, r - 1)
                    cc0 = max(0, c - 1)
                    rr1 = min(inner, r + h + 1)
                    cc1 = min(inner, c + w + 1)
                    if bool(occupied[rr0:rr1, cc0:cc1].any()):
                        continue
                    candidates.append((int(r), int(c)))
            if not candidates:
                ok = False
                break
            r, c = random.choice(candidates)
            occupied[r : r + h, c : c + w] = True
            rects.append((int(r + 1), int(c + 1), int(h), int(w)))
        if ok:
            return rects
    return None


def _build_frame_layouts(frame_count: int, size: int) -> tuple[list[list[tuple[int, int, int, int]]], tuple[int, int]] | None:
    """Sample per-frame hole layouts. Frames 0 and 1 share one repeated size."""
    library = [(h, w) for h, w in RECT_LIBRARY if h <= size - 2 and w <= size - 2]
    shared_pool = [d for d in library if d[0] * d[1] >= 3]
    for _ in range(200):
        shared_dim = random.choice(shared_pool)
        layouts: list[list[tuple[int, int, int, int]]] = []
        ok = True
        for fi in range(frame_count):
            include_shared = fi < 2
            packed = None
            for _ in range(120):
                hole_count = random.choice(HOLES_PER_FRAME)
                dims = [shared_dim] if include_shared else []
                pool = library[:]
                random.shuffle(pool)
                for d in pool:
                    if d in dims:
                        continue
                    dims.append(d)
                    if len(dims) == hole_count:
                        break
                if len(dims) != hole_count:
                    continue
                total_area = sum(h * w for h, w in dims)
                if not (6 <= total_area <= 12):
                    continue
                packed = _pack_rectangles_in_frame(size, dims)
                if packed is not None:
                    break
            if packed is None:
                ok = False
                break
            layouts.append(packed)
        if ok:
            return layouts, shared_dim
    return None


def _band_location(
    occupied: np.ndarray,
    h: int,
    w: int,
    row_lo: int,
    row_hi: int,
    col_lo: int,
    col_hi: int,
    *,
    padding: int = 1,
) -> tuple[int, int] | None:
    """Find a non-overlapping top-left within one rectangular band."""
    H, W = occupied.shape
    if row_lo > row_hi or col_lo > col_hi:
        return None
    candidates: list[tuple[int, int]] = []
    for r in range(int(row_lo), int(row_hi) + 1):
        for c in range(int(col_lo), int(col_hi) + 1):
            rr0 = max(0, r - padding)
            cc0 = max(0, c - padding)
            rr1 = min(H, r + h + padding)
            cc1 = min(W, c + w + padding)
            if not bool(occupied[rr0:rr1, cc0:cc1].any()):
                candidates.append((int(r), int(c)))
    if not candidates:
        return None
    return random.choice(candidates)


def _place_frame_sprites(
    inp: np.ndarray,
    occupied: np.ndarray,
    patches: list[np.ndarray],
    frame_bbox: tuple[int, int, int, int],
    arrangement_bbox: tuple[int, int, int, int],
    horizontal: bool,
) -> bool:
    """Place one frame's sprite subset in the aligned outer band assigned to that frame."""
    H, W = inp.shape
    fr0, fc0, fr1, fc1 = map(int, frame_bbox)
    ar0, ac0, ar1, ac1 = map(int, arrangement_bbox)

    def try_side(side: str) -> bool:
        trial_occ = occupied.copy()
        placed: list[tuple[np.ndarray, int, int]] = []
        order = list(range(len(patches)))
        random.shuffle(order)
        order.sort(key=lambda i: int(patches[i].shape[0] * patches[i].shape[1]), reverse=True)
        for idx in order:
            patch = patches[idx]
            h, w = map(int, patch.shape)
            if horizontal and side == "top":
                loc = _band_location(trial_occ, h, w, 0, ar0 - h - 1, fc0, fc1 - w + 1)
            elif horizontal and side == "bottom":
                loc = _band_location(trial_occ, h, w, ar1 + 2, H - h, fc0, fc1 - w + 1)
            elif (not horizontal) and side == "left":
                loc = _band_location(trial_occ, h, w, fr0, fr1 - h + 1, 0, ac0 - w - 1)
            else:
                loc = _band_location(trial_occ, h, w, fr0, fr1 - h + 1, ac1 + 2, W - w)
            if loc is None:
                return False
            r, c = loc
            trial_occ[r : r + h, c : c + w] = True
            placed.append((patch, int(r), int(c)))
        for patch, r, c in placed:
            paste(inp, patch, int(r), int(c), overwrite=True)
            occupied[r : r + patch.shape[0], c : c + patch.shape[1]] = True
        return True

    sides = ["top", "bottom"] if horizontal else ["left", "right"]
    random.shuffle(sides)
    for side in sides:
        if try_side(side):
            return True
    return False


def _gen_attempt() -> np.ndarray | None:
    horizontal = random.random() < 0.5
    frame_count = random.randint(FRAME_COUNT_MIN, FRAME_COUNT_MAX)
    size_choices = FRAME_SIZE_CHOICES[:]
    random.shuffle(size_choices)

    for frame_size in size_choices:
        out_h = int(frame_size if horizontal else frame_count * frame_size + (frame_count - 1) * FRAME_GAP)
        out_w = int(frame_count * frame_size + (frame_count - 1) * FRAME_GAP if horizontal else frame_size)
        min_h = max(GRID_MIN, out_h + 2 * OUTER_MARGIN if horizontal else out_h)
        min_w = max(GRID_MIN, out_w if horizontal else out_w + 2 * OUTER_MARGIN)
        if min_h > GRID_MAX or min_w > GRID_MAX:
            continue

        H = random.randint(min_h, GRID_MAX)
        W = random.randint(min_w, GRID_MAX)
        if horizontal:
            tr0 = random.randint(OUTER_MARGIN, H - out_h - OUTER_MARGIN)
            tc0 = random.randint(0, W - out_w)
        else:
            tr0 = random.randint(0, H - out_h)
            tc0 = random.randint(OUTER_MARGIN, W - out_w - OUTER_MARGIN)

        built = _build_frame_layouts(frame_count, frame_size)
        if built is None:
            continue
        frame_rects, shared_dim = built

        bg, base = random.sample(range(1, 10), 2)
        available = [c for c in range(10) if c not in (bg, base)]
        random.shuffle(available)

        expected = np.full((out_h, out_w), int(bg), dtype=int)
        inp = np.full((H, W), int(bg), dtype=int)
        occupied = np.zeros((H, W), dtype=bool)
        occupied[tr0 : tr0 + out_h, tc0 : tc0 + out_w] = True

        frame_bboxes_global: list[tuple[int, int, int, int]] = []
        frame_patches: list[list[np.ndarray]] = []
        shared_records: list[tuple[int, int]] = []
        primary_cursor = 0

        for fi, rects in enumerate(frame_rects):
            local_r0 = 0 if horizontal else fi * (frame_size + FRAME_GAP)
            local_c0 = fi * (frame_size + FRAME_GAP) if horizontal else 0
            gr0 = tr0 + local_r0
            gc0 = tc0 + local_c0
            frame_bboxes_global.append((int(gr0), int(gc0), int(gr0 + frame_size - 1), int(gc0 + frame_size - 1)))

            expected[local_r0 : local_r0 + frame_size, local_c0 : local_c0 + frame_size] = int(base)
            inp[gr0 : gr0 + frame_size, gc0 : gc0 + frame_size] = int(base)

            patches_for_frame: list[np.ndarray] = []
            for hi, (r, c, h, w) in enumerate(rects):
                primary = int(available[primary_cursor % len(available)])
                primary_cursor += 1
                cols = [primary]
                if h * w > 1 and random.random() < BICHROMATIC_FRAC:
                    sec_choices = [x for x in available if x != primary]
                    cols.append(int(random.choice(sec_choices)))
                patch = _make_patch(int(h), int(w), cols)
                patches_for_frame.append(patch)
                paste(expected, patch, int(local_r0 + r), int(local_c0 + c), overwrite=True)
                inp[gr0 + r : gr0 + r + h, gc0 + c : gc0 + c + w] = int(bg)
                if (int(h), int(w)) == shared_dim and fi < 2:
                    shared_records.append((int(fi), int(hi)))
            frame_patches.append(patches_for_frame)

        # Shared-size pieces must not be identical; otherwise the frame-grouping step would be visually weaker.
        if len(shared_records) >= 2:
            (f0, h0), (f1, h1) = shared_records[:2]
            if np.array_equal(frame_patches[f0][h0], frame_patches[f1][h1]):
                alt_primary = int(available[(primary_cursor + 1) % len(available)])
                sh, sw = map(int, frame_patches[f1][h1].shape)
                frame_patches[f1][h1] = _make_patch(sh, sw, [alt_primary])
                local_r0 = 0 if horizontal else f1 * (frame_size + FRAME_GAP)
                local_c0 = f1 * (frame_size + FRAME_GAP) if horizontal else 0
                rr, cc, _, _ = frame_rects[f1][h1]
                paste(expected, frame_patches[f1][h1], int(local_r0 + rr), int(local_c0 + cc), overwrite=True)

        arrangement_bbox = (int(tr0), int(tc0), int(tr0 + out_h - 1), int(tc0 + out_w - 1))
        for fi in range(frame_count):
            if not _place_frame_sprites(inp, occupied, frame_patches[fi], frame_bboxes_global[fi], arrangement_bbox, horizontal):
                return None

        # Solver assumptions guaranteed by construction; reject if the dominant-color criterion is violated.
        counts = {int(col): int((inp == int(col)).sum()) for col in colors_present(inp, exclude={bg})}
        if any(col != int(base) and counts[col] >= counts[int(base)] for col in counts):
            return None

        try:
            solved = main(inp)
        except Exception:
            return None
        if not np.array_equal(solved, expected):
            return None
        return inp

    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 5dbc8537 D3 input")
