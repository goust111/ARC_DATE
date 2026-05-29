# Combo variant of 8698868d (index 3)
# combo_spec: D1/4+D18/1
# Name: 边缘颜色匹配不等厚框架

from __future__ import annotations

from collections import Counter
import itertools
import random
import sys

import numpy as np

_ARC_UTILS_DIR = "/home/lixinyu/snap/new/solvers"
if _ARC_UTILS_DIR not in sys.path:
    sys.path.append(_ARC_UTILS_DIR)

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask, paste, random_free_location


"""
design_analysis:
  concrete_form:
    - Frames no longer encode the pairing key with interior holes. Each frame is a square
      multicolor object in a tight grid, but its cavity is rectangular because the
      top/bottom thickness `tb` and left/right thickness `lr` differ.
    - Sprites are detached multicolor rectangles whose size exactly matches that cavity:
      `(frame_size - 2*tb) x (frame_size - 2*lr)`.
    - Pairing is carried entirely by border markers. Every object has a dominant base
      color plus one centered marker segment on each side. The ordered signature
      `(top, right, bottom, left)` is shared by one frame and one sprite.
    - The frame markers occupy the full border thickness on that side. Therefore the
      same marker color has more visible weight on thick sides than on thin sides.

  behavior_trace:
    - Original task:
        * detect monochrome square frames and smaller square sprites
        * count connected bg-hole components in frame interior and sprite interior
        * pair by equal hole count
        * paste sprite at a fixed `+1,+1` inset
    - Fused variant:
        * detect one large multicolor frame block and repeated detached multicolor
          rectangular sprites
        * infer `frame_size`, `tb`, and `lr` from the repeated sprite rectangle
        * read each object's ordered edge signature instead of counting holes
        * paste the matched sprite at `(tb, lr)` so the unequal border thickness is preserved
    - Concrete example:
        * if `frame_size=9`, `tb=1`, `lr=2`, then the cavity is `7x5`
        * frame A may have signature `(red, green, yellow, blue)`
        * sprite X with the same signature belongs in A even if its hole count equals
          sprite Y, because Y might be `(yellow, green, red, blue)` and only differs by
          swapping top and bottom
        * a solver that still uses `region[1:-1,1:-1]` or hole counts will fail

  complexity_calibration:
    - Thickness options are intentionally restricted to `(1,2)` and `(2,1)`. The
      cavity asymmetry is easy to see and repeated across all frames in one sample.
    - Hole counts on sprites are sampled from a tiny range with forced duplicates, and
      frames are kept hole-free. This prevents accidental recovery of the original
      hole-count rule.
    - Signatures are chosen from permutations of one shared 4-color marker palette.
      That keeps the visual vocabulary small while making side order matter.
    - At least one pair of signatures in most samples differs only by swapping top/bottom
      or left/right. Those pairs have the same marker multiset and the same thick-vs-thin
      color weighting, so the solver really must read side order.

  implicit_assumption_review_for_original_main:
    - `find_components(..., monochromatic=True)` no longer works because both frames and
      sprites are multi-colored. The frame grid also merges into one connected block.
      Fix: use `monochromatic=False`, isolate the largest block as frames, and recover
      detached sprites by their repeated rectangular bbox.
    - `if h == w` cannot classify sprites anymore because sprites are rectangular.
      Fix: infer the repeated non-square sprite dimensions and backsolve `tb/lr`.
    - `frame_color = border majority` is unsafe if border pixels include marker colors.
      Fix: use the most common non-bg color over the whole object as the base color.
    - `inner = region[1:-1,1:-1]` and output placement at `+1,+1` both assume uniform
      1-pixel borders. Fix: slice and paste with `(tb, lr)`.
    - `content != sprite_color` would wrongly recolor border markers. Fix: recolor only
      `content == BG`, leaving the sprite's colored edge markers intact.

  placement_legality:
    - Frames are pasted as a contiguous `rows x cols` square grid with stride `frame_size`.
    - Sprites are placed only on background cells outside that grid using
      `random_free_location(..., padding=1)`, so sprites neither overlap the frame block
      nor touch each other.
    - Marker segments never touch the outer corners, so every object's bbox remains a
      clean rectangle and component detection is stable.

  visual_observability:
    - Across a few examples, humans can see two coupled cues: the frames have a consistent
      asymmetric cavity, and each detached sprite repeats one frame's side-color order.
    - Because frames have no interior hole cue, the border markers become the only stable
      explanation for pairing.
    - The same four marker colors reappear across many objects, which encourages reading
      their positions rather than memorizing object-specific palettes.

  neatness_and_color_inference:
    - Only two layers are present inside each output tile: the asymmetric frame and the
      inserted sprite. There is no extra overlay or clutter.
    - The sprite hole recolor remains local and interpretable: only bg holes inside the
      inserted sprite become the frame base color.
    - The generator caps frame count at 5 because each object needs one base color that
      is different from all four shared marker colors; this keeps extraction clean within
      the 10-color ARC palette.

  fusion_metric_to_check_after_implementation:
    - Weak matcher A: compare only sprite hole counts. It should be ambiguous because
      holes are duplicated and frames carry no hole structure.
    - Weak matcher B: compare only weighted edge-color histograms (ignoring side order).
      It should often be ambiguous because some signature pairs differ only by swapping
      top/bottom or left/right.
    - The implemented generator is acceptable only if these weak keys collide regularly
      while the full ordered signature remains bijective.

  new_main_assumptions_and_generator_guarantees:
    - assumption: all sprites in one sample share one rectangular bbox
      guarantee: generator uses one `(sprite_h, sprite_w)` per sample
    - assumption: the frame block is the largest foreground component
      guarantee: at least two frames are present and every frame is part of one tight grid
    - assumption: one border thickness is 1 and the other is 2
      guarantee: generator samples only `(tb, lr) in {(1,2), (2,1)}`
    - assumption: every side band contains exactly one marker color in addition to the base
      guarantee: marker drawing paints one centered segment on each side and keeps corners base-colored
    - assumption: frame and sprite signatures are unique as ordered 4-tuples
      guarantee: generator samples unique permutations from a shared 4-color palette
"""


BG = 8
FRAME_SIZE_RANGE = [8, 9, 10]
THICKNESS_OPTIONS = [(1, 2), (2, 1)]  # (top/bottom, left/right)
GRID_DIM_RANGE = (1, 4)
N_RANGE = (2, 5)
CANVAS_MIN = 15
CANVAS_MAX = 30
SPRITE_PADDING = 2
SPRITE_PLACEMENT_PAD = 1
SEED_RETRIES = 60
GEN_RETRIES = 400
MIN_FILL_RATIO = 0.45
MAX_SPRITE_HOLES = 2

_LAYOUTS: list[tuple[int, int]] = []
for _r in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
    for _c in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
        if N_RANGE[0] <= _r * _c <= N_RANGE[1]:
            _LAYOUTS.append((_r, _c))


def _max_isolated_cells(h: int, w: int) -> int:
    h = int(h)
    w = int(w)
    if h <= 0 or w <= 0:
        return 0
    return int((h * w + 1) // 2)


def _dense_enough(comp: dict) -> bool:
    b = comp["bbox"]
    h = int(bbox_h(b))
    w = int(bbox_w(b))
    if h <= 0 or w <= 0:
        return False
    return float(comp["area"]) / float(h * w) >= float(MIN_FILL_RATIO)


def _place_isolated_rect_seeds(r0: int, r1: int, c0: int, c1: int, count: int) -> list[tuple[int, int]] | None:
    if count == 0:
        return []
    if r1 < r0 or c1 < c0:
        return None
    positions = [(r, c) for r in range(int(r0), int(r1) + 1) for c in range(int(c0), int(c1) + 1)]
    if _max_isolated_cells(r1 - r0 + 1, c1 - c0 + 1) < int(count):
        return None
    for _ in range(SEED_RETRIES):
        random.shuffle(positions)
        seeds: list[tuple[int, int]] = []
        for r, c in positions:
            if all(abs(r - rr) + abs(c - cc) > 1 for rr, cc in seeds):
                seeds.append((int(r), int(c)))
                if len(seeds) == int(count):
                    return seeds
    return None


def _centered_segment(start: int, span: int) -> tuple[int, int]:
    span = int(span)
    assert span >= 1
    seg_len = max(1, span // 2)
    offset = (span - seg_len) // 2
    return int(start + offset), int(start + offset + seg_len)


def _marker_slices(h: int, w: int, tb: int, lr: int) -> list[tuple[slice, slice]]:
    h = int(h)
    w = int(w)
    tb = int(tb)
    lr = int(lr)
    assert h > 2 * tb and w > 2 * lr
    top_c0, top_c1 = _centered_segment(lr, w - 2 * lr)
    side_r0, side_r1 = _centered_segment(tb, h - 2 * tb)
    return [
        (slice(0, tb), slice(top_c0, top_c1)),
        (slice(side_r0, side_r1), slice(w - lr, w)),
        (slice(h - tb, h), slice(top_c0, top_c1)),
        (slice(side_r0, side_r1), slice(0, lr)),
    ]


def _side_bands(region: np.ndarray, tb: int, lr: int) -> list[np.ndarray]:
    h, w = region.shape
    tb = int(tb)
    lr = int(lr)
    return [
        region[0:tb, lr : w - lr],
        region[tb : h - tb, w - lr : w],
        region[h - tb : h, lr : w - lr],
        region[tb : h - tb, 0:lr],
    ]


def _overlay_markers(
    patch: np.ndarray,
    base_color: int,
    signature: tuple[int, int, int, int],
    tb: int,
    lr: int,
) -> np.ndarray:
    out = np.array(patch, dtype=int, copy=True)
    out[:, :] = np.where(out == BG, BG, int(base_color))
    for (rs, cs), marker_color in zip(_marker_slices(out.shape[0], out.shape[1], tb, lr), signature):
        out[rs, cs] = int(marker_color)
    return out


def _make_clean_frame(
    frame_size: int,
    tb: int,
    lr: int,
    base_color: int,
    signature: tuple[int, int, int, int],
) -> np.ndarray:
    patch = np.full((int(frame_size), int(frame_size)), int(base_color), dtype=int)
    return _overlay_markers(patch, int(base_color), tuple(map(int, signature)), int(tb), int(lr))


def _make_sprite_patch(
    sprite_h: int,
    sprite_w: int,
    hole_count: int,
    base_color: int,
    signature: tuple[int, int, int, int],
) -> np.ndarray | None:
    sprite_h = int(sprite_h)
    sprite_w = int(sprite_w)
    sprite = np.full((sprite_h, sprite_w), int(base_color), dtype=int)

    if hole_count > 0:
        if sprite_h < 3 or sprite_w < 3:
            return None
        inner_h = sprite_h - 2
        inner_w = sprite_w - 2
        if _max_isolated_cells(inner_h, inner_w) < int(hole_count):
            return None

        seeds = _place_isolated_rect_seeds(1, sprite_h - 2, 1, sprite_w - 2, hole_count)
        if seeds is None:
            return None

        comps: list[set[tuple[int, int]]] = [{seed} for seed in seeds]
        all_hole = set(seeds)
        for r, c in seeds:
            sprite[r, c] = BG

        max_grow = max(0, inner_h * inner_w // max(int(hole_count), 1) - 2)
        for ci, comp in enumerate(comps):
            grow = random.randint(0, min(MAX_SPRITE_HOLES, max_grow))
            for _ in range(grow):
                frontier: list[tuple[int, int]] = []
                for r, c in comp:
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if not (1 <= nr <= sprite_h - 2 and 1 <= nc <= sprite_w - 2):
                            continue
                        if (nr, nc) in all_hole:
                            continue
                        if any(
                            (nr + dr2, nc + dc2) in other
                            for j, other in enumerate(comps)
                            if j != ci
                            for dr2, dc2 in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
                        ):
                            continue
                        frontier.append((int(nr), int(nc)))
                if not frontier:
                    break
                cell = random.choice(frontier)
                comp.add(cell)
                all_hole.add(cell)
                sprite[cell[0], cell[1]] = BG

        _, num = label_mask(sprite == BG, connectivity=4)
        if int(num) != int(hole_count):
            return None

    return _overlay_markers(sprite, int(base_color), tuple(map(int, signature)), 1, 1)


def _non_bg_counter(region: np.ndarray) -> Counter:
    vals = [int(v) for v in region.flatten() if int(v) != BG]
    return Counter(vals)


def _extract_base_and_signature(region: np.ndarray, tb: int, lr: int) -> tuple[int, tuple[int, int, int, int]]:
    counter = _non_bg_counter(region)
    assert counter
    base_color = int(counter.most_common(1)[0][0])
    signature: list[int] = []
    for band in _side_bands(region, int(tb), int(lr)):
        marker_vals = [int(v) for v in band.flatten() if int(v) not in (BG, base_color)]
        assert marker_vals
        signature.append(int(Counter(marker_vals).most_common(1)[0][0]))
    return int(base_color), tuple(signature)


def _infer_geometry(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    comps = find_components(g, bg=BG, connectivity=4, monochromatic=False, return_format="dict")
    assert comps

    sprite_candidates: list[dict] = []
    for comp in comps:
        if not _dense_enough(comp):
            continue
        h = int(bbox_h(comp["bbox"]))
        w = int(bbox_w(comp["bbox"]))
        if h != w and 4 <= h <= 8 and 4 <= w <= 8:
            sprite_candidates.append(comp)

    assert sprite_candidates
    dim_counts = Counter((int(bbox_h(comp["bbox"])), int(bbox_w(comp["bbox"]))) for comp in sprite_candidates)
    (sprite_h, sprite_w), _ = max(dim_counts.items(), key=lambda kv: (kv[1], kv[0][0] * kv[0][1]))
    frame_size = int(max(sprite_h, sprite_w) + 2)
    tb_delta = int(frame_size - sprite_h)
    lr_delta = int(frame_size - sprite_w)
    assert tb_delta % 2 == 0 and lr_delta % 2 == 0
    tb = int(tb_delta // 2)
    lr = int(lr_delta // 2)
    assert sorted((tb, lr)) == [1, 2]

    frame_block = max(comps, key=lambda comp: int(comp["area"]))
    fb = tuple(map(int, frame_block["bbox"]))
    fb_h = int(bbox_h(fb))
    fb_w = int(bbox_w(fb))
    assert fb_h % frame_size == 0 and fb_w % frame_size == 0
    n_rows = int(fb_h // frame_size)
    n_cols = int(fb_w // frame_size)

    sprites = [
        comp
        for comp in sprite_candidates
        if (int(bbox_h(comp["bbox"])), int(bbox_w(comp["bbox"]))) == (int(sprite_h), int(sprite_w))
    ]
    assert len(sprites) == int(n_rows * n_cols)
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
    Match detached sprites to grid frames by ordered edge-color signatures, then nest each
    sprite into the asymmetric frame cavity that shares the same signature.

    Input structure:
    - bg is azure (8). One large multicolor frame block forms a tight square grid, and
      several smaller multicolor rectangular sprites are scattered elsewhere.
    - Every frame has a square outer bbox but unequal border thickness: top/bottom share
      one thickness and left/right share a different one.
    - Every frame and sprite has one dominant base color plus one centered marker segment
      on each side; sprites may also contain bg holes in their interior.

    Rule:
    - Infer the common sprite rectangle and use it to recover the frame cavity geometry.
    - Read each object's ordered side signature `(top, right, bottom, left)` from its edge
      markers, pairing each frame with the sprite that has the same signature.
    - Rebuild the tight frame grid as output, and place the matched sprite into the frame's
      cavity at the asymmetric inset. Recolor only the sprite's bg holes to the frame base color.

    Key insight:
    - The matching cue is on the border, not in the interior. The unequal frame thickness
      changes the visible weight of marker colors, so the solver must interpret edge markers
      per side and preserve the asymmetric inset when nesting the sprite.
    """
    g = np.array(grid, dtype=int)
    geom = _infer_geometry(g)
    frame_size = int(geom["frame_size"])
    sprite_h = int(geom["sprite_h"])
    sprite_w = int(geom["sprite_w"])
    tb = int(geom["tb"])
    lr = int(geom["lr"])
    min_r, min_c, _max_r, _max_c = map(int, geom["frame_block_bbox"])
    n_rows = int(geom["n_rows"])
    n_cols = int(geom["n_cols"])

    frame_grid: list[dict] = []
    for ri in range(n_rows):
        for ci in range(n_cols):
            inp_r = int(min_r + ri * frame_size)
            inp_c = int(min_c + ci * frame_size)
            region = g[inp_r : inp_r + frame_size, inp_c : inp_c + frame_size]
            assert region.shape == (frame_size, frame_size)
            base_color, signature = _extract_base_and_signature(region, tb, lr)
            frame_grid.append(
                {
                    "base_color": int(base_color),
                    "signature": tuple(map(int, signature)),
                    "out_r": int(ri * frame_size),
                    "out_c": int(ci * frame_size),
                }
            )

    sprites: list[dict] = []
    for comp in geom["sprites_raw"]:
        sprite_region = crop_bbox(g, comp["bbox"])
        base_color, signature = _extract_base_and_signature(sprite_region, 1, 1)
        sprites.append(
            {
                "base_color": int(base_color),
                "signature": tuple(map(int, signature)),
                "content": np.array(sprite_region, dtype=int, copy=True),
                "h": int(sprite_region.shape[0]),
                "w": int(sprite_region.shape[1]),
            }
        )

    matches: dict[tuple[int, int], dict] = {}
    used_sprites: set[int] = set()
    for frame in frame_grid:
        for idx, sprite in enumerate(sprites):
            if int(idx) in used_sprites:
                continue
            if tuple(frame["signature"]) == tuple(sprite["signature"]):
                matches[(int(frame["out_r"]), int(frame["out_c"]))] = sprite
                used_sprites.add(int(idx))
                break

    assert len(matches) == len(frame_grid)

    out = np.full((n_rows * frame_size, n_cols * frame_size), int(BG), dtype=int)
    for frame in frame_grid:
        fr = int(frame["out_r"])
        fc = int(frame["out_c"])
        frame_patch = _make_clean_frame(frame_size, tb, lr, int(frame["base_color"]), tuple(frame["signature"]))
        out[fr : fr + frame_size, fc : fc + frame_size] = frame_patch

        sprite = matches[(fr, fc)]
        content = np.array(sprite["content"], dtype=int, copy=True)
        content[content == int(BG)] = int(frame["base_color"])
        out[fr + tb : fr + tb + sprite_h, fc + lr : fc + lr + sprite_w] = content

    return out


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
    sprite_h = int(frame_size - 2 * tb)
    sprite_w = int(frame_size - 2 * lr)
    for idx, frame in enumerate(frames):
        ri = idx // cols_grid
        ci = idx % cols_grid
        rr = ri * frame_size
        cc = ci * frame_size
        fpatch = _make_clean_frame(frame_size, tb, lr, int(frame["base_color"]), tuple(frame["signature"]))
        expected[rr : rr + frame_size, cc : cc + frame_size] = fpatch

        sprite = np.array(sprite_by_signature[tuple(frame["signature"])], dtype=int, copy=True)
        sprite[sprite == BG] = int(frame["base_color"])
        expected[rr + tb : rr + tb + sprite_h, cc + lr : cc + lr + sprite_w] = sprite
    return expected


def _sample_signatures(n: int, marker_palette: tuple[int, int, int, int]) -> list[tuple[int, int, int, int]]:
    perms = [tuple(map(int, p)) for p in itertools.permutations(marker_palette)]
    anchor = random.choice(perms)
    signatures: list[tuple[int, int, int, int]] = [anchor]
    if n >= 2:
        signatures.append((int(anchor[2]), int(anchor[1]), int(anchor[0]), int(anchor[3])))
    if n >= 3:
        signatures.append((int(anchor[0]), int(anchor[3]), int(anchor[2]), int(anchor[1])))
    used = set(signatures)
    random.shuffle(perms)
    for sig in perms:
        if len(signatures) == int(n):
            break
        if sig in used:
            continue
        used.add(sig)
        signatures.append(sig)
    assert len(signatures) == int(n)
    random.shuffle(signatures)
    return signatures


def _sample_sprite_holes(n: int) -> list[int]:
    holes = [random.randint(0, MAX_SPRITE_HOLES) for _ in range(int(n))]
    if len(set(map(int, holes))) == int(n):
        holes[-1] = int(holes[0])
    if all(int(h) == 0 for h in holes):
        holes[0] = 1
        holes[-1] = 1
    return [int(h) for h in holes]


def _gen_attempt() -> np.ndarray | None:
    frame_size = random.choice(FRAME_SIZE_RANGE)
    tb, lr = random.choice(THICKNESS_OPTIONS)
    sprite_h = int(frame_size - 2 * tb)
    sprite_w = int(frame_size - 2 * lr)
    if sprite_h <= 0 or sprite_w <= 0 or sprite_h == sprite_w:
        return None

    layout_choices: list[tuple[int, int, int, int, int]] = []
    for rows_grid, cols_grid in _LAYOUTS:
        n_frames = int(rows_grid * cols_grid)
        grid_pixel_h = int(rows_grid * frame_size)
        grid_pixel_w = int(cols_grid * frame_size)
        min_h = max(CANVAS_MIN, grid_pixel_h + sprite_h + SPRITE_PADDING)
        min_w = max(CANVAS_MIN, grid_pixel_w + sprite_w + SPRITE_PADDING)
        if min_h <= CANVAS_MAX and min_w <= CANVAS_MAX:
            layout_choices.append((int(rows_grid), int(cols_grid), n_frames, int(min_h), int(min_w)))
    if not layout_choices:
        return None

    rows_grid, cols_grid, n_frames, min_h, min_w = random.choice(layout_choices)
    h = random.randint(min_h, CANVAS_MAX)
    w = random.randint(min_w, CANVAS_MAX)

    available = [c for c in range(10) if c != BG]
    marker_palette = tuple(map(int, random.sample(available, 4)))
    base_choices = [int(c) for c in available if int(c) not in set(marker_palette)]
    if len(base_choices) < n_frames:
        return None

    signatures = _sample_signatures(n_frames, marker_palette)
    frame_colors = random.sample(base_choices, n_frames)
    sprite_colors = random.sample(base_choices, n_frames)
    sprite_holes = _sample_sprite_holes(n_frames)

    frames: list[dict] = []
    sprites: list[dict] = []
    for idx in range(n_frames):
        sig = tuple(signatures[idx])
        frame_patch = _make_clean_frame(frame_size, tb, lr, int(frame_colors[idx]), sig)
        sprite_patch = _make_sprite_patch(sprite_h, sprite_w, int(sprite_holes[idx]), int(sprite_colors[idx]), sig)
        if sprite_patch is None:
            return None
        frames.append({"signature": sig, "base_color": int(frame_colors[idx]), "patch": frame_patch})
        sprites.append({"signature": sig, "base_color": int(sprite_colors[idx]), "patch": sprite_patch})

    canvas = np.full((h, w), BG, dtype=int)
    grid_pixel_h = int(rows_grid * frame_size)
    grid_pixel_w = int(cols_grid * frame_size)
    r0 = random.randint(0, h - grid_pixel_h)
    c0 = random.randint(0, w - grid_pixel_w)
    for idx, frame in enumerate(frames):
        ri = idx // cols_grid
        ci = idx % cols_grid
        rr = r0 + ri * frame_size
        cc = c0 + ci * frame_size
        canvas[rr : rr + frame_size, cc : cc + frame_size] = frame["patch"]

    random.shuffle(sprites)
    for sprite in sprites:
        patch = sprite["patch"]
        try:
            rr, cc = random_free_location(canvas, patch, transparent=BG, padding=SPRITE_PLACEMENT_PAD)
        except ValueError:
            return None
        paste(canvas, patch, rr, cc, transparent=BG)

    expected = _build_expected_output(
        frames,
        {tuple(sp["signature"]): np.array(sp["patch"], dtype=int, copy=True) for sp in sprites},
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
