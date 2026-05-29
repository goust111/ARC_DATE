# Combo variant of 8698868d (index 28)
# combo_spec: D1/4+D3/3+D18/1
# Name: 边缘色匹配+渐变框架+不等厚边框

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask, paste, random_free_location


"""
design_analysis:
  concrete_form:
    - Frames still appear as one tight rows x cols block, but each tile is now a square with:
      one dominant base color, four differently colored border bands, and asymmetric thickness.
    - The top and bottom bands share one thickness `tb`; the left and right bands share another
      thickness `lr`; the generator only uses `(tb, lr) in {(1,2), (2,1)}` so the cavity is
      visibly rectangular and the asymmetry is easy to notice.
    - Sprites are detached rectangles whose outer size exactly matches the frame cavity.
      Their own border is 1-pixel thick, but its four sides carry the same ordered color
      signature `(top, right, bottom, left)` as the matching frame.
    - Sprite holes stay bg-colored in the input. In the output, each hole component is recolored
      by the frame side nearest to that component's centroid, so one sprite can receive several
      frame-edge colors.

  behavior_trace:
    - Original task:
      1. detect square frames and square sprites
      2. count bg-hole components inside both
      3. pair equal counts
      4. paste sprite at offset `+1,+1` and recolor every hole to one frame color
    - Fusion task:
      1. detect one multicolor frame block plus repeated detached rectangular sprites
      2. infer `frame_size`, `tb`, `lr` from the repeated sprite rectangle
      3. extract each object's ordered edge signature from its four colored sides
      4. pair by exact signature equality
      5. paste sprite at offset `(tb, lr)` and recolor each hole component with the nearest
         frame side color, not the frame base color
    - Small trace:
      - Suppose `frame_size=8`, `(tb,lr)=(1,2)`, so sprites are `6x4`.
      - Frame A has signature `(2,3,4,5)` and frame B has `(6,1,2,3)`.
      - Sprite X is `6x4` with signature `(2,3,4,5)` and two holes, one near the top and one
        near the right; sprite Y has signature `(6,1,2,3)` but also two holes.
      - The original solver could confuse X and Y because the hole counts match and the old
        `+1,+1` inset is wrong for a `6x4` cavity. The fusion solver must choose X for A from
        the border colors, inset it at `(1,2)`, then recolor A's top-near hole to color `2`
        and right-near hole to color `3`.

  complexity_calibration:
    - Keeping only two thickness modes avoids overfitting the generator while still making the
      geometry visibly non-square in every sample.
    - Using the full ordered side signature keeps matching bijective without requiring too many
      extra markers; four sides already exist visually, so the color code is clean.
    - Hole counts are intentionally decoupled from matching and may repeat, which prevents the
      family from silently collapsing back to the original hole-count rule.
    - Hole recoloring is component-wise, not per-pixel Voronoi, so the output stays legible.

  implicit_assumption_review_for_original_main:
    - `find_components(..., monochromatic=True)` no longer works because both frames and sprites
      are multicolor. The fused solver must use `monochromatic=False`.
    - `h == w` no longer identifies sprites because sprites are rectangular.
    - `sprite_size = frame_size - 2` is invalid because height and width shrink by different
      amounts.
    - `region[1:-1, 1:-1]` is no longer the frame cavity; it must become
      `region[tb:frame_size-tb, lr:frame_size-lr]`.
    - `content != sprite_color` cannot be used for recoloring because it would erase the sprite's
      colored border. Only bg holes should change.
    - Border-majority color is not the frame identity anymore. The frame contains four side colors
      plus one base color; the base must be read from the interior, while pairing uses the ordered
      side tuple.

  placement_legality:
    - Frames are pasted as one contiguous block with no gaps, so slicing by `frame_size` recovers
      the tiles exactly.
    - Sprites are placed with `random_free_location(..., padding=1)` and therefore never overlap
      one another or touch the frame block.
    - Candidate hole components are chosen from connected 1- or 2-cell shapes inside the sprite
      interior, with an explicit non-overlap and non-adjacency check, so each hole component stays
      separate.

  visual_observability:
    - A human can see three repeated cues across examples:
      1. frames and sprites share a four-side color order,
      2. the cavity is consistently rectangular because opposite border thicknesses differ,
      3. hole color in the output depends on the hole's location relative to the colored frame.
    - No ranking by numeric color id is used. Everything depends on visible side positions,
      component positions, and repeated geometry.

  neatness_and_color_inference:
    - Each object uses one base color plus four side colors; there is no extra overlay noise.
    - Hole recoloring borrows only the already visible frame side colors, so the new colors in the
      output are justified by nearby structure instead of appearing arbitrary.
    - The generator rejects samples whose hole components do not have a unique nearest side, which
      keeps the recolor rule crisp.

  fusion_metric_intent:
    - Non-degeneracy is checked by measuring three visible factors on generated samples:
      asymmetric geometry (`tb != lr`), signature-based matching disagreement with a naive hole-
      count pairing, and multi-side hole recoloring across the sample.
    - If any of those dropped to zero systematically, the variant would be regressing toward the
      original task or toward a weaker single-dimension variant.

  new_main_assumptions_and_generator_guarantees:
    - assumption: the largest multicolor component is the frame block
      guarantee: sprites are smaller detached rectangles and the frame block is always largest
    - assumption: all sprites share one repeated rectangular bbox
      guarantee: one sample uses one `(sprite_h, sprite_w)` everywhere
    - assumption: `(sprite_h, sprite_w)` corresponds to exactly one thickness option
      guarantee: only `(1,2)` and `(2,1)` are used
    - assumption: each frame/sprite side has a single dominant color band
      guarantee: generator paints each side as one solid band
    - assumption: every sprite hole component has a unique nearest side
      guarantee: candidate hole components are filtered by centroid-side uniqueness
    - assumption: signatures are bijective
      guarantee: generator samples unique ordered 4-tuples
"""


BG = 8
FRAME_SIZE_RANGE = [8, 9, 10]
THICKNESS_OPTIONS = [(1, 2), (2, 1)]
GRID_DIM_RANGE = (1, 4)
N_RANGE = (2, 6)
CANVAS_MIN = 15
CANVAS_MAX = 30
SPRITE_PADDING = 2
SPRITE_PLACEMENT_PAD = 1
GEN_RETRIES = 500
MIN_FILL_RATIO = 0.6
MAX_SPRITE_HOLE_COMPONENTS = 3
SIDES = ("top", "right", "bottom", "left")

_LAYOUTS_BY_N: dict[int, list[tuple[int, int]]] = {}
for _r in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
    for _c in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
        if N_RANGE[0] <= _r * _c <= N_RANGE[1]:
            _LAYOUTS_BY_N.setdefault(_r * _c, []).append((_r, _c))
_VALID_NS = sorted(_LAYOUTS_BY_N.keys())


def _non_bg_counter(region: np.ndarray) -> Counter:
    return Counter(int(v) for v in region.flatten() if int(v) != BG)


def _most_common_non_bg(region: np.ndarray) -> int:
    counter = _non_bg_counter(region)
    assert counter
    return int(counter.most_common(1)[0][0])


def _draw_frame_patch(
    frame_size: int,
    tb: int,
    lr: int,
    base_color: int,
    signature: tuple[int, int, int, int],
) -> np.ndarray:
    patch = np.full((int(frame_size), int(frame_size)), int(base_color), dtype=int)
    top_color, right_color, bottom_color, left_color = map(int, signature)
    patch[:tb, :] = int(top_color)
    patch[frame_size - tb :, :] = int(bottom_color)
    patch[tb : frame_size - tb, :lr] = int(left_color)
    patch[tb : frame_size - tb, frame_size - lr :] = int(right_color)
    return patch


def _draw_sprite_patch(
    sprite_h: int,
    sprite_w: int,
    base_color: int,
    signature: tuple[int, int, int, int],
    hole_components: list[list[tuple[int, int]]],
) -> np.ndarray:
    patch = np.full((int(sprite_h), int(sprite_w)), int(base_color), dtype=int)
    top_color, right_color, bottom_color, left_color = map(int, signature)
    patch[0, :] = int(top_color)
    patch[-1, :] = int(bottom_color)
    patch[1:-1, 0] = int(left_color)
    patch[1:-1, -1] = int(right_color)
    for cells in hole_components:
        for r, c in cells:
            patch[int(r), int(c)] = int(BG)
    return patch


def _dense_enough(comp: dict) -> bool:
    h = int(bbox_h(comp["bbox"]))
    w = int(bbox_w(comp["bbox"]))
    if h <= 0 or w <= 0:
        return False
    return float(comp["area"]) / float(h * w) >= float(MIN_FILL_RATIO)


def _centroid(cells: list[tuple[int, int]]) -> tuple[float, float]:
    rs = [float(r) for r, _c in cells]
    cs = [float(c) for _r, c in cells]
    return (sum(rs) / len(rs), sum(cs) / len(cs))


def _distance_map(h: int, w: int, center: tuple[float, float]) -> dict[str, float]:
    r, c = center
    return {
        "top": float(r),
        "right": float((w - 1) - c),
        "bottom": float((h - 1) - r),
        "left": float(c),
    }


def _nearest_side_for_center(h: int, w: int, center: tuple[float, float]) -> str | None:
    dist = _distance_map(int(h), int(w), center)
    best_value = min(dist.values())
    best_sides = [side for side, value in dist.items() if abs(value - best_value) < 1e-9]
    if len(best_sides) != 1:
        return None
    return str(best_sides[0])


def _candidate_margin(h: int, w: int, cells: list[tuple[int, int]]) -> float:
    values = sorted(_distance_map(int(h), int(w), _centroid(cells)).values())
    return float(values[1] - values[0]) if len(values) >= 2 else 0.0


def _enumerate_hole_candidates(sprite_h: int, sprite_w: int) -> dict[str, list[list[tuple[int, int]]]]:
    inner_cells = [(r, c) for r in range(1, sprite_h - 1) for c in range(1, sprite_w - 1)]
    by_side: dict[str, dict[tuple[tuple[int, int], ...], float]] = {side: {} for side in SIDES}

    for cell in inner_cells:
        shapes = [[cell]]
        r, c = cell
        for dr, dc in ((1, 0), (0, 1)):
            nr, nc = r + dr, c + dc
            if 1 <= nr <= sprite_h - 2 and 1 <= nc <= sprite_w - 2:
                shapes.append(sorted([cell, (nr, nc)]))
        for shape in shapes:
            center = _centroid(shape)
            side = _nearest_side_for_center(sprite_h, sprite_w, center)
            if side is None:
                continue
            key = tuple(sorted((int(rr), int(cc)) for rr, cc in shape))
            margin = _candidate_margin(sprite_h, sprite_w, list(key))
            prev = by_side[side].get(key)
            if prev is None or margin > prev:
                by_side[side][key] = float(margin)

    out: dict[str, list[list[tuple[int, int]]]] = {}
    for side in SIDES:
        items = sorted(by_side[side].items(), key=lambda kv: (-kv[1], -len(kv[0]), kv[0]))
        out[side] = [list(k) for k, _margin in items]
        assert out[side]
    return out


def _touches(cells_a: list[tuple[int, int]], cells_b: list[tuple[int, int]]) -> bool:
    bset = {(int(r), int(c)) for r, c in cells_b}
    for r, c in cells_a:
        if (int(r), int(c)) in bset:
            return True
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            if (int(r + dr), int(c + dc)) in bset:
                return True
    return False


def _choose_hole_components(sprite_h: int, sprite_w: int, targets: list[str]) -> list[list[tuple[int, int]]] | None:
    candidates = _enumerate_hole_candidates(int(sprite_h), int(sprite_w))
    ordered_targets = sorted(list(targets), key=lambda side: len(candidates[side]))
    chosen: list[list[tuple[int, int]]] = []

    def backtrack(idx: int) -> bool:
        if idx == len(ordered_targets):
            return True
        side = ordered_targets[idx]
        for cand in candidates[side]:
            if any(_touches(cand, prev) for prev in chosen):
                continue
            chosen.append(cand)
            if backtrack(idx + 1):
                return True
            chosen.pop()
        return False

    if not backtrack(0):
        return None
    return chosen


def _sample_sprite_hole_targets(n_sprites: int) -> list[list[str]] | None:
    for _ in range(40):
        hole_counts = [random.randint(1, MAX_SPRITE_HOLE_COMPONENTS) for _ in range(int(n_sprites))]
        if int(n_sprites) >= 2 and len(set(hole_counts)) == len(hole_counts):
            continue

        plans = [list(random.sample(SIDES, count)) for count in hole_counts]
        all_side_hits = [side for plan in plans for side in plan]
        if len(set(all_side_hits)) >= 3:
            return plans
    return None


def _extract_sprite_info(region: np.ndarray) -> dict:
    base_color = _most_common_non_bg(region[1:-1, 1:-1])
    signature = (
        _most_common_non_bg(region[0:1, :]),
        _most_common_non_bg(region[1:-1, -1:]),
        _most_common_non_bg(region[-1:, :]),
        _most_common_non_bg(region[1:-1, 0:1]),
    )
    return {
        "base_color": int(base_color),
        "signature": tuple(map(int, signature)),
        "content": np.array(region, dtype=int, copy=True),
        "h": int(region.shape[0]),
        "w": int(region.shape[1]),
    }


def _extract_frame_info(region: np.ndarray, tb: int, lr: int) -> dict:
    frame_size = int(region.shape[0])
    base_color = _most_common_non_bg(region[tb : frame_size - tb, lr : frame_size - lr])
    signature = (
        _most_common_non_bg(region[:tb, :]),
        _most_common_non_bg(region[tb : frame_size - tb, frame_size - lr :]),
        _most_common_non_bg(region[frame_size - tb :, :]),
        _most_common_non_bg(region[tb : frame_size - tb, :lr]),
    )
    return {
        "base_color": int(base_color),
        "signature": tuple(map(int, signature)),
    }


def _recolor_holes_by_frame_sides(
    sprite_patch: np.ndarray,
    frame_signature: tuple[int, int, int, int],
) -> np.ndarray:
    out = np.array(sprite_patch, dtype=int, copy=True)
    hole_mask = out == int(BG)
    labels, num = label_mask(hole_mask, connectivity=4)
    h, w = out.shape
    side_to_color = {
        "top": int(frame_signature[0]),
        "right": int(frame_signature[1]),
        "bottom": int(frame_signature[2]),
        "left": int(frame_signature[3]),
    }
    for lab in range(1, int(num) + 1):
        coords = np.argwhere(labels == lab)
        cells = [(int(r), int(c)) for r, c in coords]
        side = _nearest_side_for_center(h, w, _centroid(cells))
        assert side is not None
        out[labels == lab] = int(side_to_color[side])
    return out


def _infer_geometry(sprite_h: int, sprite_w: int) -> tuple[int, int, int]:
    matches: list[tuple[int, int, int]] = []
    for tb, lr in THICKNESS_OPTIONS:
        frame_h = int(sprite_h) + 2 * int(tb)
        frame_w = int(sprite_w) + 2 * int(lr)
        if frame_h == frame_w:
            matches.append((int(frame_h), int(tb), int(lr)))
    assert len(matches) == 1
    return matches[0]


def _locate_frame_block_and_sprites(grid: np.ndarray) -> dict:
    comps = find_components(grid, bg=BG, connectivity=4, monochromatic=False, return_format="dict")
    assert len(comps) >= 3

    frame_block = max(comps, key=lambda comp: int(comp["area"]))
    sprites_pool = [comp for comp in comps if comp is not frame_block and _dense_enough(comp)]
    assert sprites_pool

    dim_counts = Counter()
    for comp in sprites_pool:
        h = int(bbox_h(comp["bbox"]))
        w = int(bbox_w(comp["bbox"]))
        if h != w and min(h, w) >= 4:
            dim_counts[(h, w)] += 1
    assert dim_counts

    (sprite_h, sprite_w), _freq = max(dim_counts.items(), key=lambda kv: (kv[1], kv[0][0] * kv[0][1]))
    sprites = [
        comp
        for comp in sprites_pool
        if (int(bbox_h(comp["bbox"])), int(bbox_w(comp["bbox"]))) == (int(sprite_h), int(sprite_w))
    ]
    assert len(sprites) >= 2

    frame_size, tb, lr = _infer_geometry(int(sprite_h), int(sprite_w))
    fb = tuple(map(int, frame_block["bbox"]))
    block_h = int(bbox_h(fb))
    block_w = int(bbox_w(fb))
    assert block_h % frame_size == 0 and block_w % frame_size == 0
    n_rows = block_h // frame_size
    n_cols = block_w // frame_size
    assert 1 <= n_rows <= 3 and 1 <= n_cols <= 3
    assert int(n_rows * n_cols) == len(sprites)

    return {
        "frame_size": int(frame_size),
        "sprite_h": int(sprite_h),
        "sprite_w": int(sprite_w),
        "tb": int(tb),
        "lr": int(lr),
        "frame_block_bbox": fb,
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "sprites_raw": sprites,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match detached rectangular sprites to multicolor frames by ordered side colors, then nest them.

    Input structure:
    - Background is azure (`8`). One large connected foreground block is a tight grid of square frames.
    - Every frame has a dominant interior/base color, four differently colored border bands, and asymmetric
      border thickness: top/bottom use one thickness, left/right use another.
    - Elsewhere on the canvas sit the matching sprites: detached rectangles whose size equals the frame cavity,
      with their own base color, the same four-side color signature, and bg-colored holes inside.

    Rule:
    - Infer the common rectangular sprite size, then backsolve the frame size and asymmetric inset.
    - Slice the frame block into frame tiles, extract each tile's ordered side signature
      `(top, right, bottom, left)`, and pair it with the sprite carrying the same signature.
    - Output only the tight frame grid. Rebuild each frame, insert the matched sprite at the asymmetric inset,
      and recolor each sprite hole component with the color of the nearest frame side.

    Key insight:
    - The decisive cue is the ordered distribution of edge colors together with the rectangular cavity shape.
      Hole count no longer drives matching, and hole recoloring depends on spatial relation to the frame sides.
    """
    g = np.array(grid, dtype=int)
    geom = _locate_frame_block_and_sprites(g)
    frame_size = int(geom["frame_size"])
    sprite_h = int(geom["sprite_h"])
    sprite_w = int(geom["sprite_w"])
    tb = int(geom["tb"])
    lr = int(geom["lr"])
    min_r, min_c, max_r, max_c = map(int, geom["frame_block_bbox"])
    n_rows = int(geom["n_rows"])
    n_cols = int(geom["n_cols"])

    frame_grid: list[dict] = []
    for ri in range(n_rows):
        for ci in range(n_cols):
            inp_r = int(min_r + ri * frame_size)
            inp_c = int(min_c + ci * frame_size)
            region = g[inp_r : inp_r + frame_size, inp_c : inp_c + frame_size]
            assert region.shape == (frame_size, frame_size)
            frame = _extract_frame_info(region, tb, lr)
            frame["out_r"] = int(ri * frame_size)
            frame["out_c"] = int(ci * frame_size)
            frame_grid.append(frame)

    sprites: list[dict] = []
    for comp in geom["sprites_raw"]:
        sprite_region = crop_bbox(g, comp["bbox"])
        sprite = _extract_sprite_info(sprite_region)
        sprites.append(sprite)

    signature_to_sprite = {tuple(sprite["signature"]): sprite for sprite in sprites}
    assert len(signature_to_sprite) == len(sprites) == len(frame_grid)

    out = np.full((n_rows * frame_size, n_cols * frame_size), int(BG), dtype=int)
    for frame in frame_grid:
        fr = int(frame["out_r"])
        fc = int(frame["out_c"])
        signature = tuple(map(int, frame["signature"]))
        frame_patch = _draw_frame_patch(frame_size, tb, lr, int(frame["base_color"]), signature)
        out[fr : fr + frame_size, fc : fc + frame_size] = frame_patch

        sprite = signature_to_sprite[signature]
        content = _recolor_holes_by_frame_sides(np.array(sprite["content"], dtype=int, copy=True), signature)
        assert content.shape == (sprite_h, sprite_w)
        out[fr + tb : fr + tb + sprite_h, fc + lr : fc + lr + sprite_w] = content

    return out


def _sample_unique_signatures(n: int) -> list[tuple[int, int, int, int]] | None:
    available = [c for c in range(10) if c != BG]
    seen: set[tuple[int, int, int, int]] = set()
    signatures: list[tuple[int, int, int, int]] = []
    for _ in range(2000):
        if len(signatures) == int(n):
            return signatures
        sig = tuple(map(int, random.sample(available, 4)))
        if sig in seen:
            continue
        seen.add(sig)
        signatures.append(sig)
    return None


def _pick_base_color(signature: tuple[int, int, int, int], avoid: set[int] | None = None) -> int:
    blocked = set(map(int, signature))
    if avoid is not None:
        blocked |= set(map(int, avoid))
    choices = [c for c in range(10) if c != BG and c not in blocked]
    assert choices
    return int(random.choice(choices))


def _build_expected_output(
    frames: list[dict],
    sprite_by_signature: dict[tuple[int, int, int, int], np.ndarray],
    rows_grid: int,
    cols_grid: int,
    frame_size: int,
    tb: int,
    lr: int,
) -> np.ndarray:
    expected = np.full((rows_grid * frame_size, cols_grid * frame_size), BG, dtype=int)
    for idx, frame in enumerate(frames):
        ri = idx // cols_grid
        ci = idx % cols_grid
        rr = int(ri * frame_size)
        cc = int(ci * frame_size)
        signature = tuple(map(int, frame["signature"]))
        frame_patch = _draw_frame_patch(frame_size, tb, lr, int(frame["base_color"]), signature)
        expected[rr : rr + frame_size, cc : cc + frame_size] = frame_patch

        sprite = np.array(sprite_by_signature[signature], dtype=int, copy=True)
        sprite = _recolor_holes_by_frame_sides(sprite, signature)
        sh, sw = sprite.shape
        expected[rr + tb : rr + tb + sh, cc + lr : cc + lr + sw] = sprite
    return expected


def _hole_count(patch: np.ndarray) -> int:
    _labels, num = label_mask(patch == int(BG), connectivity=4)
    return int(num)


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(_VALID_NS)
    rows_grid, cols_grid = random.choice(_LAYOUTS_BY_N[n])
    frame_size = random.choice(FRAME_SIZE_RANGE)
    tb, lr = random.choice(THICKNESS_OPTIONS)
    sprite_h = int(frame_size - 2 * tb)
    sprite_w = int(frame_size - 2 * lr)
    if sprite_h <= 2 or sprite_w <= 2 or sprite_h == sprite_w:
        return None

    grid_pixel_h = rows_grid * frame_size
    grid_pixel_w = cols_grid * frame_size
    min_h = max(CANVAS_MIN, grid_pixel_h + sprite_h + SPRITE_PADDING)
    min_w = max(CANVAS_MIN, grid_pixel_w + sprite_w + SPRITE_PADDING)
    if min_h > CANVAS_MAX or min_w > CANVAS_MAX:
        return None
    canvas_h = random.randint(min_h, CANVAS_MAX)
    canvas_w = random.randint(min_w, CANVAS_MAX)

    signatures = _sample_unique_signatures(n)
    if signatures is None:
        return None
    random.shuffle(signatures)

    hole_target_plans = _sample_sprite_hole_targets(n)
    if hole_target_plans is None:
        return None

    frames: list[dict] = []
    sprites: list[dict] = []
    side_hits: list[str] = []

    for idx in range(n):
        signature = tuple(signatures[idx])
        targets = list(hole_target_plans[idx])
        hole_components = _choose_hole_components(sprite_h, sprite_w, targets)
        if hole_components is None:
            return None

        frame_base = _pick_base_color(signature)
        sprite_base = _pick_base_color(signature, avoid={frame_base})
        frame_patch = _draw_frame_patch(frame_size, tb, lr, frame_base, signature)
        sprite_patch = _draw_sprite_patch(sprite_h, sprite_w, sprite_base, signature, hole_components)

        if _hole_count(sprite_patch) != len(targets):
            return None

        inferred_sides = []
        labels, num = label_mask(sprite_patch == int(BG), connectivity=4)
        for lab in range(1, int(num) + 1):
            coords = np.argwhere(labels == lab)
            cells = [(int(r), int(c)) for r, c in coords]
            side = _nearest_side_for_center(sprite_h, sprite_w, _centroid(cells))
            if side is None:
                return None
            inferred_sides.append(side)
        if sorted(inferred_sides) != sorted(targets):
            return None

        frames.append({"signature": signature, "base_color": int(frame_base), "patch": frame_patch})
        sprites.append({"signature": signature, "base_color": int(sprite_base), "patch": sprite_patch})
        side_hits.extend(inferred_sides)

    if len(set(side_hits)) < 3:
        return None

    random.shuffle(sprites)

    canvas = np.full((canvas_h, canvas_w), BG, dtype=int)
    r0 = random.randint(0, canvas_h - grid_pixel_h)
    c0 = random.randint(0, canvas_w - grid_pixel_w)
    for idx, frame in enumerate(frames):
        ri = idx // cols_grid
        ci = idx % cols_grid
        rr = int(r0 + ri * frame_size)
        cc = int(c0 + ci * frame_size)
        canvas[rr : rr + frame_size, cc : cc + frame_size] = frame["patch"]

    for sprite in sprites:
        patch = sprite["patch"]
        try:
            rr, cc = random_free_location(canvas, patch, transparent=BG, padding=SPRITE_PLACEMENT_PAD)
        except ValueError:
            return None
        paste(canvas, patch, rr, cc, transparent=BG)

    expected = _build_expected_output(
        frames,
        {tuple(sprite["signature"]): np.array(sprite["patch"], dtype=int, copy=True) for sprite in sprites},
        rows_grid,
        cols_grid,
        frame_size,
        tb,
        lr,
    )

    try:
        out = main(canvas)
    except Exception:
        return None
    if out.shape != expected.shape or not np.array_equal(out, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
