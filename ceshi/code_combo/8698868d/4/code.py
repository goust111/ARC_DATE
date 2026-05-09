# Combo variant of 8698868d (index 4)
# combo_spec: D1/1+D3/1
# Name: 形状匹配矩形框架

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask, paste, random_free_location

# Design analysis
# ===============
# design_analysis:
#   a_concreteization_and_trace:
#     input:
#       - The original task had one grid of large square frames and the same number of
#         smaller square sprites elsewhere. Matching was by the count of bg hole-components.
#       - This combo changes both moving parts together: every frame is now a visible
#         non-square rectangle, and every free sprite is the matching inner rectangle
#         `(frame_h-2, frame_w-2)`.
#       - Counting is intentionally made useless: each frame and sprite carries exactly one
#         connected bg hole region, but the hole contour differs across objects.
#     concrete_flow_example:
#       - Original: a 7x7 frame with 3 isolated bg pixels pairs with the sprite that also
#         has 3 hole-components.
#       - Combo: a 7x9 frame may contain one L-like bg region, another 7x9 frame one T-like
#         region, and all sprites also have exactly one connected bg region. The solver must
#         recover the rectangular frame lattice, extract each hole silhouette, canonicalize
#         it over 90-degree rotations, and pair by contour rather than by any scalar count.
#       - Output still keeps only the tight frame grid, inserting the matched sprite into
#         each frame and recoloring sprite bg cells to the frame color.
#   b_complexity_calibration:
#     - Every chosen shape has hole-component count 1, so the old count-based rule cannot
#       distinguish any pair.
#     - Every sample includes at least two distinct shapes with the same area, so naive area
#       matching also fails.
#     - At least one matched frame/sprite pair is forced to use different orientations of the
#       same contour, making rotation-aware shape comparison observable.
#     - Rectangular outer frames are shared across a sample, keeping the layout learnable
#       while still making aspect ratio an explicit visual cue.
#   c_implicit_assumption_audit:
#     - Original `if h == w` candidate filtering is invalid and replaced by exact `(h,w)`
#       grouping with a required non-square frame size.
#     - Original grouping by a single side length is invalid; the solver now searches for a
#       frame dimension group whose inner dimensions also appear as sprite dimensions.
#     - Original matching on `num_holes` is invalid because all objects now have the same
#       hole count; those lines become canonical contour extraction.
#     - Output assembly can still recolor every non-sprite-color pixel to the frame color
#       because sprites still contain only one layer of bg holes and keep solid borders.
#   d_placement_legality:
#     - The generator uses one rectangle size per sample, places frames on a tight grid with
#       row step `frame_h` and column step `frame_w`, and places sprites only in remaining
#       bg space with padding.
#     - Hole masks are embedded inside a protected core: frames keep a 1-cell border plus a
#       1-cell moat around the hole-host area; sprites keep a 1-cell border.
#     - This guarantees no overlap, no clipping, and stable monochrome components for solver
#       detection.
#   e_visual_observability:
#     - Across examples a human can see that the output is a grid of rectangles, not squares.
#     - Because all hole counts are identical while contours differ, the examples visibly rule
#       out count-based matching and point to shape-based pairing.
#     - The forced rotation mismatch in at least one pair makes exact orientation matching
#       insufficient, so the intended invariance can be inferred from a few samples.
#   f_visual_cleanliness:
#     - Each object contains only one hole contour and no extra markers.
#     - The moat around frame holes prevents the contour from visually merging with the frame
#       border, and the output remains a clean two-layer rectangle grid.
#   g_color_reasoning:
#     - Colors remain identifiers only. Matching never depends on numeric color ids.
#     - The only recoloring event is structural and local: sprite hole pixels adopt the color
#       of the surrounding matched frame in the output.
#   h_variant_metric:
#     - Post-test validation should check that `frame_h != frame_w`, every selected frame and
#       sprite has exactly one hole-component, at least one chosen shape area repeats, and at
#       least one matched pair uses different orientations. If any of these fail systematically,
#       the combo has degenerated.
#   i_new_main_assumptions:
#     - `main()` assumes there is exactly one non-square frame dimension group whose inner
#       size `(h-2,w-2)` appears among the scattered sprites.
#     - `main()` assumes sprite shape signatures are unique after rotation-canonicalization.
#     - The generator guarantees both properties and ensures every frame slot is visible.


BG = 8
MIN_FILL_RATIO = 0.5
FRAME_HEIGHT_RANGE = range(7, 10)
FRAME_WIDTH_RANGE = range(7, 10)
GRID_DIM_RANGE = (1, 4)
N_RANGE = (3, 6)
CANVAS_MIN = 15
CANVAS_MAX = 30
SPRITE_PADDING = 2
SPRITE_PLACEMENT_PAD = 1
GEN_RETRIES = 300


_LAYOUTS_BY_N: dict[int, list[tuple[int, int]]] = {}
for _r in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
    for _c in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
        if N_RANGE[0] <= _r * _c <= N_RANGE[1]:
            _LAYOUTS_BY_N.setdefault(_r * _c, []).append((_r, _c))
_VALID_NS = sorted(_LAYOUTS_BY_N.keys())


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    trimmed = _trim_mask(mask)
    if trimmed.size == 0:
        return ()
    return tuple(sorted((int(r), int(c)) for r, c in np.argwhere(trimmed)))


def _canonical_shape_signature(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    trimmed = _trim_mask(mask)
    if trimmed.size == 0:
        return ()
    cur = trimmed
    sigs: list[tuple[tuple[int, int], ...]] = []
    for _ in range(4):
        sigs.append(_mask_signature(cur))
        cur = np.rot90(cur)
    return min(sigs)


def _coords_to_mask(coords: list[tuple[int, int]]) -> np.ndarray:
    h = 1 + max(r for r, _ in coords)
    w = 1 + max(c for _, c in coords)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        mask[int(r), int(c)] = True
    return mask


def _mask_from_signature(sig: tuple[tuple[int, int], ...]) -> np.ndarray:
    if not sig:
        return np.zeros((0, 0), dtype=bool)
    h = 1 + max(r for r, _ in sig)
    w = 1 + max(c for _, c in sig)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in sig:
        mask[int(r), int(c)] = True
    return mask


def _rotation_signatures(mask: np.ndarray) -> list[tuple[tuple[int, int], ...]]:
    cur = _trim_mask(mask)
    sigs: list[tuple[tuple[int, int], ...]] = []
    for _ in range(4):
        sig = _mask_signature(cur)
        if sig not in sigs:
            sigs.append(sig)
        cur = np.rot90(cur)
    return sigs


def _build_shape_library() -> list[dict]:
    raw_shapes = [
        [(0, 0), (0, 1), (0, 2)],
        [(0, 0), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (0, 2), (1, 1)],
        [(0, 0), (0, 1), (1, 1), (1, 2)],
        [(0, 0), (1, 0), (2, 0), (2, 1)],
        [(0, 0), (0, 1), (1, 0), (2, 0)],
        [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)],
        [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)],
        [(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)],
        [(0, 1), (1, 0), (1, 1), (1, 2), (2, 2)],
    ]
    library: dict[tuple[tuple[int, int], ...], dict] = {}
    for coords in raw_shapes:
        mask = _coords_to_mask(coords)
        if label_mask(mask, connectivity=4)[1] != 1:
            continue
        rot_sigs = _rotation_signatures(mask)
        if len(rot_sigs) < 2:
            continue
        sig = min(rot_sigs)
        library[sig] = {
            "sig": sig,
            "mask": _mask_from_signature(sig),
            "area": int(len(sig)),
            "rot_sigs": rot_sigs,
        }
    return sorted(library.values(), key=lambda item: (int(item["area"]), item["sig"]))


SHAPE_LIBRARY = _build_shape_library()


def _candidate_groups(grid: np.ndarray) -> dict[tuple[int, int], list[dict]]:
    comps = find_components(grid, bg=BG, connectivity=4, monochromatic=True, return_format="dict")
    groups: dict[tuple[int, int], list[dict]] = {}
    for comp in comps:
        h = int(bbox_h(comp["bbox"]))
        w = int(bbox_w(comp["bbox"]))
        area = int(comp["area"])
        if 4 <= h <= 10 and 4 <= w <= 10 and area >= int(h * w * MIN_FILL_RATIO):
            groups.setdefault((h, w), []).append(comp)
    return groups


def _select_frame_and_sprite_dims(groups: dict[tuple[int, int], list[dict]]) -> tuple[tuple[int, int], tuple[int, int]]:
    candidates: list[tuple[int, int, int, int, int]] = []
    for frame_h, frame_w in groups:
        sprite_dims = (frame_h - 2, frame_w - 2)
        if frame_h < 7 or frame_w < 7 or frame_h == frame_w:
            continue
        if sprite_dims not in groups:
            continue
        frame_count = len(groups[(frame_h, frame_w)])
        sprite_count = len(groups[sprite_dims])
        candidates.append((frame_h * frame_w, min(frame_count, sprite_count), frame_h, frame_w, frame_count))
    assert candidates
    candidates.sort(reverse=True)
    _area, _paired, frame_h, frame_w, _count = candidates[0]
    return (frame_h, frame_w), (frame_h - 2, frame_w - 2)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match scattered sprites to rectangular frames by hole-contour shape, then pack them into the frame grid.

    Input structure:
    - Background is azure (8). Larger monochrome rectangular frames of one shared non-square size form a tight grid.
    - Smaller monochrome rectangles of size `(frame_h-2, frame_w-2)` are scattered elsewhere as sprites.
    - Each frame and sprite may contain bg-colored hole pixels, but every relevant hole pattern is a single connected region with a distinctive contour.

    Rule:
    - Detect the non-square frame dimensions and the corresponding sprite dimensions.
    - For each frame, extract the bg hole silhouette inside the border; for each sprite, extract its internal bg silhouette.
    - Canonicalize each silhouette up to 90-degree rotation and pair frames with sprites by equal canonical contour.
    - Build the output as the tight frame grid only, inserting each matched sprite into its frame and recoloring the sprite's bg hole cells to the frame color.

    Key insight:
    - Matching is no longer based on the number of hole-components. All objects can share the same hole count, so only the rotated contour shape distinguishes which sprite belongs in which rectangular frame.
    """
    g = np.array(grid, dtype=int)
    groups = _candidate_groups(g)
    assert groups

    frame_dims, sprite_dims = _select_frame_and_sprite_dims(groups)
    frame_h, frame_w = frame_dims
    sprite_h, sprite_w = sprite_dims
    frames = groups[frame_dims]
    sprites_raw = groups[sprite_dims]
    assert frames and sprites_raw

    row_positions = sorted({int(comp["bbox"][0]) for comp in frames})
    col_positions = sorted({int(comp["bbox"][1]) for comp in frames})
    n_rows = len(row_positions)
    n_cols = len(col_positions)
    assert n_rows * n_cols == len(frames)
    assert all(row_positions[i] - row_positions[0] == i * frame_h for i in range(n_rows))
    assert all(col_positions[i] - col_positions[0] == i * frame_w for i in range(n_cols))

    frame_lookup = {(int(comp["bbox"][0]), int(comp["bbox"][1])): comp for comp in frames}
    frame_grid: list[dict] = []
    for ri, src_r in enumerate(row_positions):
        for ci, src_c in enumerate(col_positions):
            comp = frame_lookup[(src_r, src_c)]
            region = crop_bbox(g, comp["bbox"])
            frame_color = int(comp["color"])
            hole_sig = _canonical_shape_signature(region[1:-1, 1:-1] != frame_color)
            frame_grid.append(
                {
                    "color": frame_color,
                    "shape": hole_sig,
                    "out_r": int(ri * frame_h),
                    "out_c": int(ci * frame_w),
                }
            )

    sprite_by_shape: dict[tuple[tuple[int, int], ...], dict] = {}
    for comp in sprites_raw:
        sprite_region = crop_bbox(g, comp["bbox"])
        sprite_color = int(comp["color"])
        hole_sig = _canonical_shape_signature(sprite_region != sprite_color)
        assert hole_sig not in sprite_by_shape
        sprite_by_shape[hole_sig] = {
            "color": sprite_color,
            "content": sprite_region.copy(),
        }

    out = np.full((n_rows * frame_h, n_cols * frame_w), BG, dtype=int)
    for frame in frame_grid:
        fr = int(frame["out_r"])
        fc = int(frame["out_c"])
        frame_color = int(frame["color"])
        out[fr : fr + frame_h, fc : fc + frame_w] = frame_color

        assert frame["shape"] in sprite_by_shape
        sprite = sprite_by_shape[frame["shape"]]
        content = np.array(sprite["content"], dtype=int).copy()
        sprite_color = int(sprite["color"])
        content[content != sprite_color] = frame_color
        out[fr + 1 : fr + 1 + sprite_h, fc + 1 : fc + 1 + sprite_w] = content

    return out


def _pick_shapes(count: int) -> list[dict]:
    by_area: dict[int, list[dict]] = {}
    for item in SHAPE_LIBRARY:
        by_area.setdefault(int(item["area"]), []).append(item)

    repeated_areas = [area for area, items in by_area.items() if len(items) >= 2]
    assert repeated_areas

    ordered_areas = repeated_areas[:]
    random.shuffle(ordered_areas)

    chosen: list[dict] = []
    remaining = count
    for area in sorted(ordered_areas, key=lambda value: len(by_area[value]), reverse=True):
        if remaining == 0:
            break
        pool = by_area[area][:]
        random.shuffle(pool)
        take = min(len(pool), remaining)
        chosen.extend(pool[:take])
        remaining -= take

    if remaining:
        used = {item["sig"] for item in chosen}
        leftovers = [item for item in SHAPE_LIBRARY if item["sig"] not in used]
        random.shuffle(leftovers)
        chosen.extend(leftovers[:remaining])

    random.shuffle(chosen)
    return chosen


def _shape_signature_after_rotation(mask: np.ndarray, k: int) -> tuple[tuple[int, int], ...]:
    return _mask_signature(np.rot90(mask, int(k)))


def _different_rotation(mask: np.ndarray, ref_k: int) -> int:
    ref_sig = _shape_signature_after_rotation(mask, ref_k)
    choices = [k for k in range(4) if _shape_signature_after_rotation(mask, k) != ref_sig]
    assert choices
    return random.choice(choices)


def _render_hole_host(core_h: int, core_w: int, hole_mask: np.ndarray) -> np.ndarray:
    host = np.zeros((core_h, core_w), dtype=bool)
    h, w = hole_mask.shape
    assert h <= core_h and w <= core_w
    top = random.randint(0, core_h - h)
    left = random.randint(0, core_w - w)
    host[top : top + h, left : left + w] = hole_mask
    return host


def _render_frame(frame_h: int, frame_w: int, frame_color: int, hole_mask: np.ndarray) -> np.ndarray:
    core_h = frame_h - 4
    core_w = frame_w - 4
    frame = np.full((frame_h, frame_w), frame_color, dtype=int)
    host = _render_hole_host(core_h, core_w, _trim_mask(hole_mask))
    inner = frame[2 : 2 + core_h, 2 : 2 + core_w]
    inner[host] = BG
    assert label_mask(frame != frame_color, connectivity=4)[1] == 1
    return frame


def _render_sprite(sprite_h: int, sprite_w: int, sprite_color: int, hole_mask: np.ndarray) -> np.ndarray:
    core_h = sprite_h - 2
    core_w = sprite_w - 2
    sprite = np.full((sprite_h, sprite_w), sprite_color, dtype=int)
    host = _render_hole_host(core_h, core_w, _trim_mask(hole_mask))
    inner = sprite[1 : 1 + core_h, 1 : 1 + core_w]
    inner[host] = BG
    assert label_mask(sprite != sprite_color, connectivity=4)[1] == 1
    return sprite


def _build_expected_output(
    frames: list[dict],
    sprites: list[dict],
    rows_grid: int,
    cols_grid: int,
    frame_h: int,
    frame_w: int,
    sprite_h: int,
    sprite_w: int,
) -> np.ndarray:
    expected = np.full((rows_grid * frame_h, cols_grid * frame_w), BG, dtype=int)
    sprite_by_sig = {sprite["sig"]: sprite for sprite in sprites}
    for idx, frame in enumerate(frames):
        ri, ci = divmod(idx, cols_grid)
        rr = int(ri * frame_h)
        cc = int(ci * frame_w)
        frame_color = int(frame["color"])
        expected[rr : rr + frame_h, cc : cc + frame_w] = frame_color
        content = sprite_by_sig[frame["sig"]]["patch"].copy()
        content[content == BG] = frame_color
        expected[rr + 1 : rr + 1 + sprite_h, cc + 1 : cc + 1 + sprite_w] = content
    return expected


def _valid_frame_dims(rows_grid: int, cols_grid: int) -> list[tuple[int, int]]:
    dims: list[tuple[int, int]] = []
    for frame_h in FRAME_HEIGHT_RANGE:
        for frame_w in FRAME_WIDTH_RANGE:
            if frame_h == frame_w:
                continue
            sprite_h = frame_h - 2
            sprite_w = frame_w - 2
            min_h = max(CANVAS_MIN, rows_grid * frame_h + sprite_h + SPRITE_PADDING)
            min_w = max(CANVAS_MIN, cols_grid * frame_w + sprite_w + SPRITE_PADDING)
            if min_h <= CANVAS_MAX and min_w <= CANVAS_MAX:
                dims.append((frame_h, frame_w))
    return dims


def _gen_attempt() -> np.ndarray | None:
    n_frames = random.choice(_VALID_NS)
    rows_grid, cols_grid = random.choice(_LAYOUTS_BY_N[n_frames])

    valid_dims = _valid_frame_dims(rows_grid, cols_grid)
    if not valid_dims:
        return None
    frame_h, frame_w = random.choice(valid_dims)
    sprite_h, sprite_w = frame_h - 2, frame_w - 2

    grid_pixel_h = rows_grid * frame_h
    grid_pixel_w = cols_grid * frame_w
    min_h = max(CANVAS_MIN, grid_pixel_h + sprite_h + SPRITE_PADDING)
    min_w = max(CANVAS_MIN, grid_pixel_w + sprite_w + SPRITE_PADDING)
    if min_h > CANVAS_MAX or min_w > CANVAS_MAX:
        return None
    height = random.randint(min_h, CANVAS_MAX)
    width = random.randint(min_w, CANVAS_MAX)

    available = [color for color in range(10) if color != BG]
    frame_colors = random.sample(available, n_frames)
    sprite_colors = random.sample(available, n_frames)

    shape_items = _pick_shapes(n_frames)
    frame_items = shape_items[:]
    sprite_items = shape_items[:]
    random.shuffle(frame_items)
    random.shuffle(sprite_items)

    frame_rot: dict[tuple[tuple[int, int], ...], int] = {}
    sprite_rot: dict[tuple[tuple[int, int], ...], int] = {}
    by_sig = {item["sig"]: item for item in shape_items}
    for sig, item in by_sig.items():
        frame_rot[sig] = random.randint(0, 3)
        sprite_rot[sig] = random.randint(0, 3)
        if _shape_signature_after_rotation(item["mask"], frame_rot[sig]) == _shape_signature_after_rotation(item["mask"], sprite_rot[sig]):
            sprite_rot[sig] = _different_rotation(item["mask"], frame_rot[sig])

    frames: list[dict] = []
    for idx, item in enumerate(frame_items):
        sig = item["sig"]
        hole_mask = np.rot90(item["mask"], frame_rot[sig])
        patch = _render_frame(frame_h, frame_w, frame_colors[idx], hole_mask)
        frames.append({"sig": sig, "patch": patch, "color": frame_colors[idx], "area": int(item["area"])})

    sprites: list[dict] = []
    for idx, item in enumerate(sprite_items):
        sig = item["sig"]
        hole_mask = np.rot90(item["mask"], sprite_rot[sig])
        patch = _render_sprite(sprite_h, sprite_w, sprite_colors[idx], hole_mask)
        sprites.append({"sig": sig, "patch": patch, "color": sprite_colors[idx], "area": int(item["area"])})

    if frame_h == frame_w:
        return None
    if len({frame["area"] for frame in frames}) == n_frames:
        return None
    if not all(label_mask(frame["patch"] != frame["color"], connectivity=4)[1] == 1 for frame in frames):
        return None
    if not all(label_mask(sprite["patch"] != sprite["color"], connectivity=4)[1] == 1 for sprite in sprites):
        return None
    if not any(
        _shape_signature_after_rotation(by_sig[sig]["mask"], frame_rot[sig])
        != _shape_signature_after_rotation(by_sig[sig]["mask"], sprite_rot[sig])
        for sig in by_sig
    ):
        return None

    canvas = np.full((height, width), BG, dtype=int)
    r0 = random.randint(0, height - grid_pixel_h)
    c0 = random.randint(0, width - grid_pixel_w)
    for idx, frame in enumerate(frames):
        ri, ci = divmod(idx, cols_grid)
        rr = int(r0 + ri * frame_h)
        cc = int(c0 + ci * frame_w)
        canvas[rr : rr + frame_h, cc : cc + frame_w] = frame["patch"]

    for sprite in sprites:
        try:
            r, c = random_free_location(canvas, sprite["patch"], transparent=BG, padding=SPRITE_PLACEMENT_PAD)
        except ValueError:
            return None
        paste(canvas, sprite["patch"], r, c, transparent=BG)

    try:
        out = main(canvas)
    except Exception:
        return None

    expected = _build_expected_output(frames, sprites, rows_grid, cols_grid, frame_h, frame_w, sprite_h, sprite_w)
    if out.shape != expected.shape or not np.array_equal(out, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
