# Combo variant of 5dbc8537 (index 17)
# combo_spec: D3/0+D11/1
# Name: 多基底色多框架结构解析

"""
Design analysis for the combo variant
=====================================

a) Concrete behavior and flow trace
-----------------------------------
- The source task had one large single-color strip template. All bg cells inside its bbox were one global cavity,
  and the solver only had to exact-pack rectangle sprites by bbox size.
- The D3 part splits the destination into 2-3 separate small frames. That already adds one extra step:
  a sprite must first be assigned to a frame before it can be assigned to a hole.
- The D11 part removes the old "one dominant base color" assumption. Every frame is now built from two alternating
  structural colors in a checkerboard pattern. The holes are still bg rectangles, but the color touching the hole's
  top edge becomes meaningful.

- Small trace:
  original:
  1. Find one non-bg base color.
  2. Take one bbox around that color.
  3. Treat every non-base cell in the bbox as part of one global cavity.
  4. Match each outside rectangle only by size.

  combo:
  1. Find the two-color structural pair whose union forms several disconnected hole-bearing frames.
  2. Crop the union bbox of those frames, preserving bg gaps between them.
  3. Group outside sprites by the frame band they align with.
  4. Within a frame, match a sprite by `(height, width, top-left marker color)`, where the marker color equals the
     hole's visible top-edge frame color.

  difference:
  - Single-color base detection is no longer valid.
  - Global hole extraction is no longer valid because the crop contains intentional bg gaps between frames.
  - Size alone is no longer sufficient because one frame contains two same-size holes with different edge colors.

b) Complexity calibration
-------------------------
- The generator always makes at least 2 frames, so the task cannot collapse back to the original single-frame case.
- Frame 0 always contains one duplicated hole size twice, with opposite edge-color signatures. That makes the D11 cue
  necessary inside a frame.
- Frame 1 also contains that same size once, so repeated sizes also exist across frames. That keeps the D3 grouping
  stage necessary before the D11 disambiguation stage.
- Only two structural colors are used, and the marker is always the sprite's top-left pixel. That keeps the extra
  degree of freedom visible and low-entropy instead of becoming arbitrary color coding.

c) Implicit-assumption audit of the original main()
---------------------------------------------------
- Original base-color pick:
  `largest connected component among non-bg colors` breaks because each frame is bichromatic.
- Original `hole_mask = dst != base_col` breaks because there is no single base color and because bg gaps between
  frames would be misread as holes.
- Original source extraction by removing all base-colored pixels breaks because source sprites now intentionally contain
  one base-colored marker pixel.
- Original global exact cover by size breaks because two holes in one frame can share the same size.
- Safe replacement:
  detect the structural color pair by scanning non-bg pairs for large disconnected components that contain both colors
  and enclose bg rectangles; blank only the union arrangement bbox when extracting sprites; then solve locally inside
  each frame by `(size + edge-color marker)`.

d) Placement legality
---------------------
- Frames are axis-aligned squares, separated by a 2-cell bg gap, and never overlap.
- Every hole is a strict rectangle placed inside a 1-cell frame border with at least one structural cell separating
  any two holes.
- Sprites are scattered only outside the full arrangement bbox and keep a 1-cell bg buffer from one another and from
  the arrangement.
- The generator rejects samples unless `main(input)` reproduces the constructed output exactly.

e) Visual observability
-----------------------
- The multiple destination frames are directly visible, and the output crop keeps the bg gaps between them.
- The structural pair is also directly visible because each frame is checkerboard-like rather than monochrome.
- The top-left base-color marker on a sprite is a visible cue, and the matching hole exposes the same cue on its top
  border. With a few examples, a human can infer that the marker picks which same-size slot inside the frame is correct.

f) Visual cleanliness
---------------------
- The arrangement is only one row or one column of small frames, so the crop stays legible.
- Each frame has only 2-3 holes; only one frame carries the within-frame duplicate-size event.
- The marker is exactly one pixel at the sprite's top-left corner, which is strong enough to be visible without turning
  the sprites into noisy composite objects.

g) Change inferability
----------------------
- No hidden use is made of numeric color ids. The relevant distinction is purely visual: "these two colors form the
  frame fabric", and "the marker color matches the hole edge color".
- The structural pair never appears as normal fill colors elsewhere, so the marker reads as a deliberate cue instead
  of an incidental palette collision.
- The repeated-size cases are sparse and supported: one within a frame, one across frames. That is enough to make the
  new logic necessary without flooding the sample with ambiguous duplicates.

h) New solver assumptions and generator guarantees
--------------------------------------------------
- Assumption: exactly one color pair forms multiple large hole-bearing components.
  Guarantee: only the structural pair creates multi-cell components with enclosed bg holes.
- Assumption: each sprite lies in exactly one frame's alignment band.
  Guarantee: scattering is constrained to that frame's own top/bottom band or left/right band.
- Assumption: within one frame, `(height, width, edge_color)` is unique.
  Guarantee: only one size is duplicated, and its two copies have opposite edge colors.
- Assumption: the sprite marker is the top-left pixel and belongs to the structural pair.
  Guarantee: generator overwrites exactly that pixel after creating the patch.
"""

from __future__ import annotations

import random
from itertools import combinations

import numpy as np

from arc_utils import bbox_coords, colors_present, components_mask, crop_bbox, find_components, get_bg, paste


GRID_MIN, GRID_MAX = 18, 25
FRAME_COUNT_MIN, FRAME_COUNT_MAX = 2, 3
FRAME_SIZE = 7
FRAME_GAP = 2
OUTER_MARGIN = 4
HOLES_PER_FRAME = [2, 3]
BICHROMATIC_FRAC = 0.4
MAX_RETRIES = 500
MIN_FRAME_AREA = 12

RECT_LIBRARY = [
    (1, 2),
    (2, 1),
    (1, 3),
    (3, 1),
    (2, 2),
    (2, 3),
    (3, 2),
]

DUPLICATE_DIM_POOL = [
    (1, 2),
    (2, 1),
    (1, 3),
    (3, 1),
]


def _make_pattern_patch(h: int, w: int, cols: list[int]) -> np.ndarray:
    """Fill an (h, w) rectangle with 1 or 2 colors using a visible nontrivial pattern."""
    out = np.full((h, w), int(cols[0]), dtype=int)
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


def _frame_pattern(size: int, cols: tuple[int, int], phase: int) -> np.ndarray:
    rr, cc = np.indices((size, size))
    idx = (rr + cc + int(phase)) % 2
    out = np.full((size, size), int(cols[0]), dtype=int)
    out[idx == 1] = int(cols[1])
    return out


def _edge_color_for_hole(rect: tuple[int, int, int, int], cols: tuple[int, int], phase: int) -> int:
    r, c, _h, _w = map(int, rect)
    return int(cols[((r - 1) + c + int(phase)) % 2])


def _detect_structural_pair(grid: np.ndarray, bg: int) -> tuple[tuple[int, int], list[dict]]:
    g = np.array(grid, dtype=int)
    best_score: tuple[int, int, int] | None = None
    best_pair: tuple[int, int] | None = None
    best_frames: list[dict] | None = None

    for a, b in combinations(colors_present(g, exclude={bg}), 2):
        mask = (g == int(a)) | (g == int(b))
        comps = components_mask(mask, connectivity=8)
        frames: list[dict] = []
        total_area = 0
        for coords in comps:
            area = int(coords.shape[0])
            if area < MIN_FRAME_AREA:
                continue
            vals = {int(v) for v in g[coords[:, 0], coords[:, 1]]}
            if vals != {int(a), int(b)}:
                continue
            r0, c0, r1, c1 = bbox_coords(coords)
            patch = crop_bbox(g, (int(r0), int(c0), int(r1), int(c1)))
            bg_mask = patch == int(bg)
            if not bool(bg_mask.any()):
                continue
            holes: list[tuple[int, int, int, int]] = []
            ok = True
            for hole in components_mask(bg_mask, connectivity=4):
                hr0, hc0, hr1, hc1 = bbox_coords(hole)
                if hr0 == 0 or hc0 == 0 or hr1 == patch.shape[0] - 1 or hc1 == patch.shape[1] - 1:
                    ok = False
                    break
                h = int(hr1 - hr0 + 1)
                w = int(hc1 - hc0 + 1)
                if int(hole.shape[0]) != int(h * w):
                    ok = False
                    break
                holes.append((int(hr0), int(hc0), int(h), int(w)))
            if not ok or not holes:
                continue
            total_area += area
            frames.append(
                {
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "patch": patch,
                    "hole_rects": holes,
                }
            )
        if len(frames) < 2:
            continue
        score = (len(frames), total_area, -abs(int(a) - int(b)))
        if best_score is None or score > best_score:
            best_score = score
            best_pair = (int(a), int(b))
            best_frames = frames

    assert best_pair is not None and best_frames is not None
    return best_pair, best_frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill several separated bichromatic frames by first finding the structural color pair, then routing each sprite to its frame and color-tagged hole.

    Input structure:
    - bg is the most common color. The destination is not one monochrome strip anymore: it is a row or column of small frames built from two alternating structural colors.
    - Each frame contains rectangular bg holes, and the outer sprites are rectangles scattered outside the arrangement. Every sprite carries one 1-pixel structural-color marker at its top-left corner.

    Rule:
    - Recover the two structural colors by finding the color pair whose union forms multiple disconnected hole-bearing frames, then crop the union bbox of the whole arrangement.
    - Assign each outside sprite to the frame band it aligns with. Inside that frame, match the sprite to the unique hole with the same rectangle size and the same visible top-edge frame color.
    - Paste the sprite into that hole and preserve bg only in the intentional gaps between different frames.

    Key insight: The puzzle is no longer "one base color + size fit". You must first separate the two-color frame fabric from the clutter, then use frame grouping and the hole-edge color cue together to disambiguate repeated rectangle sizes.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    base_pair, frames = _detect_structural_pair(g, bg)
    base_set = set(base_pair)

    ur0 = min(int(f["bbox"][0]) for f in frames)
    uc0 = min(int(f["bbox"][1]) for f in frames)
    ur1 = max(int(f["bbox"][2]) for f in frames)
    uc1 = max(int(f["bbox"][3]) for f in frames)
    out = np.full((int(ur1 - ur0 + 1), int(uc1 - uc0 + 1)), int(bg), dtype=int)

    for frame in frames:
        fr0, fc0, fr1, fc1 = frame["bbox"]
        patch = frame["patch"]
        frame_mask = (patch == int(base_pair[0])) | (patch == int(base_pair[1]))
        paste(out, patch, int(fr0 - ur0), int(fc0 - uc0), mask=frame_mask, overwrite=True)

        hole_map: dict[tuple[int, int, int], tuple[int, int, int, int]] = {}
        for rr0, cc0, h, w in frame["hole_rects"]:
            edge = int(patch[int(rr0 - 1), int(cc0)])
            key = (int(h), int(w), edge)
            assert key not in hole_map
            hole_map[key] = (int(rr0), int(cc0), int(h), int(w))
        frame["hole_map"] = hole_map

    row_spread = max(f["bbox"][0] for f in frames) - min(f["bbox"][0] for f in frames)
    col_spread = max(f["bbox"][1] for f in frames) - min(f["bbox"][1] for f in frames)
    horizontal = bool(col_spread >= row_spread)
    frames.sort(key=lambda f: int(f["bbox"][1] if horizontal else f["bbox"][0]))

    masked = g.copy()
    masked[int(ur0) : int(ur1) + 1, int(uc0) : int(uc1) + 1] = int(bg)
    sprite_comps = find_components(masked, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    sprites: list[dict] = []
    for comp in sprite_comps:
        pr0, pc0, pr1, pc1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (int(pr0), int(pc0), int(pr1), int(pc1)))
        ph, pw = map(int, patch.shape)
        coords = comp["coords"]
        mask = np.zeros((ph, pw), dtype=bool)
        mask[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        marker = int(patch[0, 0])
        assert marker in base_set
        sprites.append(
            {
                "bbox": (int(pr0), int(pc0), int(pr1), int(pc1)),
                "h": int(ph),
                "w": int(pw),
                "marker": marker,
                "patch": patch,
                "mask": mask,
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
        pieces = assigned[i]
        hole_map = frame["hole_map"]
        assert len(pieces) == len(hole_map)
        piece_map: dict[tuple[int, int, int], dict] = {}
        for sp in pieces:
            key = (int(sp["h"]), int(sp["w"]), int(sp["marker"]))
            assert key not in piece_map
            piece_map[key] = sp
        fr0, fc0, _fr1, _fc1 = frame["bbox"]
        for key, (rr0, cc0, _h, _w) in hole_map.items():
            sp = piece_map[key]
            paste(out, sp["patch"], int(fr0 - ur0 + rr0), int(fc0 - uc0 + cc0), mask=sp["mask"], overwrite=True)

    return out


def _pack_rectangles_in_frame(
    size: int,
    dims: list[tuple[int, int]],
    *,
    duplicate_pair: tuple[int, int] | None = None,
) -> list[tuple[int, int, int, int]] | None:
    """Pack rectangles into one frame interior with at least one structural cell between them."""
    inner = int(size - 2)
    order = sorted(
        [(idx, int(h), int(w)) for idx, (h, w) in enumerate(dims)],
        key=lambda x: (-x[1] * x[2], -x[1], -x[2], x[0]),
    )
    for _ in range(160):
        occupied = np.zeros((inner, inner), dtype=bool)
        placed: list[tuple[int, int, int, int] | None] = [None] * len(dims)
        ok = True
        for idx, h, w in order:
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
            placed[idx] = (int(r + 1), int(c + 1), int(h), int(w))
        if not ok:
            continue
        rects = [p for p in placed if p is not None]
        if duplicate_pair is not None:
            i0, i1 = map(int, duplicate_pair)
            r0, c0, _h0, _w0 = rects[i0]
            r1, c1, _h1, _w1 = rects[i1]
            if ((int(r0 - 1) + int(c0)) % 2) == ((int(r1 - 1) + int(c1)) % 2):
                continue
        return rects
    return None


def _build_frame_layouts(frame_count: int, size: int) -> tuple[list[list[tuple[int, int, int, int]]], tuple[int, int]] | None:
    """Build frame layouts with one within-frame duplicate-size event and one cross-frame repeated size."""
    for _ in range(240):
        dup_dim = random.choice(DUPLICATE_DIM_POOL)
        layouts: list[list[tuple[int, int, int, int]]] = []
        ok = True
        for fi in range(frame_count):
            packed = None
            for _ in range(160):
                hole_count = random.choice(HOLES_PER_FRAME)
                if fi == 0:
                    dims = [dup_dim, dup_dim]
                    pool = [d for d in RECT_LIBRARY if d != dup_dim]
                    random.shuffle(pool)
                    while len(dims) < hole_count and pool:
                        dims.append(pool.pop())
                    if len(dims) != hole_count:
                        continue
                    packed = _pack_rectangles_in_frame(size, dims, duplicate_pair=(0, 1))
                else:
                    dims = [dup_dim] if fi == 1 else []
                    pool = RECT_LIBRARY[:]
                    random.shuffle(pool)
                    while len(dims) < hole_count and pool:
                        d = pool.pop()
                        if d in dims:
                            continue
                        dims.append(d)
                    if len(dims) != hole_count:
                        continue
                    packed = _pack_rectangles_in_frame(size, dims)
                if packed is None:
                    continue
                total_area = sum(h * w for _r, _c, h, w in packed)
                if not (6 <= total_area <= 12):
                    packed = None
                    continue
                break
            if packed is None:
                ok = False
                break
            layouts.append(packed)
        if ok:
            return layouts, dup_dim
    return None


def _make_piece_patch(h: int, w: int, marker: int, fill_palette: list[int]) -> np.ndarray:
    primary = int(random.choice(fill_palette))
    cols = [primary]
    if h * w > 1 and random.random() < BICHROMATIC_FRAC:
        secondary = int(random.choice([c for c in fill_palette if c != primary]))
        cols.append(secondary)
    patch = _make_pattern_patch(int(h), int(w), cols)
    patch[0, 0] = int(marker)
    return patch


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
    out_h = int(FRAME_SIZE if horizontal else frame_count * FRAME_SIZE + (frame_count - 1) * FRAME_GAP)
    out_w = int(frame_count * FRAME_SIZE + (frame_count - 1) * FRAME_GAP if horizontal else FRAME_SIZE)

    min_h = max(GRID_MIN, out_h + 2 * OUTER_MARGIN if horizontal else out_h)
    min_w = max(GRID_MIN, out_w if horizontal else out_w + 2 * OUTER_MARGIN)
    if min_h > GRID_MAX or min_w > GRID_MAX:
        return None

    H = random.randint(min_h, GRID_MAX)
    W = random.randint(min_w, GRID_MAX)
    if horizontal:
        tr0 = random.randint(OUTER_MARGIN, H - out_h - OUTER_MARGIN)
        tc0 = random.randint(0, W - out_w)
    else:
        tr0 = random.randint(0, H - out_h)
        tc0 = random.randint(OUTER_MARGIN, W - out_w - OUTER_MARGIN)

    built = _build_frame_layouts(frame_count, FRAME_SIZE)
    if built is None:
        return None
    frame_rects, dup_dim = built

    bg, base_a, base_b = random.sample(range(10), 3)
    structural_pair = (int(base_a), int(base_b))
    fill_palette = [c for c in range(10) if c not in {bg, base_a, base_b}]
    if len(fill_palette) < 3:
        return None

    expected = np.full((out_h, out_w), int(bg), dtype=int)
    inp = np.full((H, W), int(bg), dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    occupied[tr0 : tr0 + out_h, tc0 : tc0 + out_w] = True

    frame_bboxes_global: list[tuple[int, int, int, int]] = []
    frame_patches: list[list[np.ndarray]] = []
    duplicate_edge_events = 0
    cross_frame_dup_events = 0

    for fi, rects in enumerate(frame_rects):
        local_r0 = 0 if horizontal else fi * (FRAME_SIZE + FRAME_GAP)
        local_c0 = fi * (FRAME_SIZE + FRAME_GAP) if horizontal else 0
        gr0 = tr0 + local_r0
        gc0 = tc0 + local_c0
        phase = random.randint(0, 1)
        frame_canvas = _frame_pattern(FRAME_SIZE, structural_pair, phase)
        frame_bboxes_global.append((int(gr0), int(gc0), int(gr0 + FRAME_SIZE - 1), int(gc0 + FRAME_SIZE - 1)))

        expected[local_r0 : local_r0 + FRAME_SIZE, local_c0 : local_c0 + FRAME_SIZE] = frame_canvas
        inp[gr0 : gr0 + FRAME_SIZE, gc0 : gc0 + FRAME_SIZE] = frame_canvas

        size_to_edges: dict[tuple[int, int], set[int]] = {}
        patches_for_frame: list[np.ndarray] = []
        for r, c, h, w in rects:
            marker = _edge_color_for_hole((int(r), int(c), int(h), int(w)), structural_pair, phase)
            patch = _make_piece_patch(int(h), int(w), marker, fill_palette)
            patches_for_frame.append(patch)
            paste(expected, patch, int(local_r0 + r), int(local_c0 + c), overwrite=True)
            inp[gr0 + r : gr0 + r + h, gc0 + c : gc0 + c + w] = int(bg)
            size_to_edges.setdefault((int(h), int(w)), set()).add(int(marker))
        if any(len(edges) >= 2 for edges in size_to_edges.values()):
            duplicate_edge_events += 1
        if any(size == dup_dim for size in size_to_edges):
            cross_frame_dup_events += 1
        frame_patches.append(patches_for_frame)

    if duplicate_edge_events < 1 or cross_frame_dup_events < 2:
        return None

    arrangement_bbox = (int(tr0), int(tc0), int(tr0 + out_h - 1), int(tc0 + out_w - 1))
    for fi in range(frame_count):
        if not _place_frame_sprites(inp, occupied, frame_patches[fi], frame_bboxes_global[fi], arrangement_bbox, horizontal):
            return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 5dbc8537 combo input")
