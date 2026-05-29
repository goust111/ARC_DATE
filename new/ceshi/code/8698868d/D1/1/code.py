# Variant of 8698868d
# Dimension: D1 — 匹配条件：连通洞计数 (match_cond)
# Change: hole_counting — 精灵与框架各自内部的 4-连通背景色洞分量数目相等即配对... → 形状匹配替代洞计数

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask, paste, random_free_location

# Design analysis
# ===============
# design_analysis:
#   a_concreteization_and_trace:
#     input:
#       - Frames stay as larger square monochrome components arranged in a tight grid.
#       - Sprites stay as smaller square monochrome components scattered elsewhere.
#       - The old frame interior contained several isolated bg pixels whose only useful
#         attribute was the number of connected hole components.
#       - In this variant each frame instead contains one connected bg hole region with a
#         recognizable polyomino-like contour. Each sprite also contains one connected bg
#         hole region. The pairing key is that contour, ignoring translation and allowing
#         90-degree rotation.
#     concrete_flow_example:
#       - Old logic: frame A has 2 holes, sprite X has 2 hole-components -> pair them.
#       - New logic: frame A may show an L-shaped hole, sprite X may show a rotated L,
#         sprite Y may show a T-like shape; both sprites still have exactly 1 hole-component,
#         so counting is useless and only contour comparison survives.
#       - Output assembly itself is unchanged: once paired, the sprite is nested inside the
#         frame and the sprite's bg pixels are recolored to the frame color.
#   b_complexity_calibration:
#     - To avoid degeneration back to counting, the generator makes all chosen shapes have
#       the same hole-component count (=1).
#     - To avoid degeneration into simple area matching, every sample includes at least one
#       pair of distinct shapes with the same hole area.
#     - Rotation is observable because frame and sprite orientations are sampled independently
#       and at least one matched pair is forced to differ in orientation.
#   c_implicit_assumption_audit:
#     - Original lines computing `num_holes` assumed the whole matching signal collapsed to
#       an integer. That assumption is removed; those places now extract a canonical shape key.
#     - Original sprite/frame detection by square size and fill ratio still works because the
#       border remains solid and the hole area stays well below the 50% threshold.
#     - Output painting originally replaced every non-sprite-color pixel with frame color.
#       That remains valid because sprite bg holes still represent the only non-sprite pixels.
#   d_placement_legality:
#     - Frames remain on a regular touching grid and keep distinct colors, so component-based
#       recovery of grid dimensions still works.
#     - Sprites are still placed only on free bg cells with padding, so they never overlap
#       each other or the frame grid.
#     - Hole masks are embedded strictly inside a protected inner core, preserving solid borders.
#   e_visual_observability:
#     - The new rule is visible from examples because counting gives the same answer for all
#       objects, while the hole silhouettes differ and some matched pairs are rotated versions.
#   f_visual_cleanliness:
#     - Only one hole layer exists inside each object; there are no extra markers or overlays.
#     - A one-cell moat around the frame hole and a one-cell border around the sprite keep the
#       silhouettes easy to read.
#   g_color_reasoning:
#     - Colors are still only object identifiers. Matching never depends on numeric color ids.
#   h_generator_guardrails:
#     - `frame_size=6` is intentionally removed. Its effective shape core is only 2x2, which
#       does not provide enough distinct rotationally informative contours for N up to 6.
#   i_new_main_assumptions:
#     - `main()` now assumes each relevant object exposes a stable hole silhouette after
#       trimming empty margins and canonicalizing over rotations.
#     - The generator guarantees one unique sprite per canonical silhouette, so greedy matching
#       by shape key remains safe.


# --------------- solver ---------------

BG = 8


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
    coords = np.argwhere(trimmed)
    return tuple(sorted((int(r), int(c)) for r, c in coords))


def _canonical_shape_signature(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    trimmed = _trim_mask(mask)
    if trimmed.size == 0:
        return ()
    sigs = []
    cur = trimmed
    for _ in range(4):
        sigs.append(_mask_signature(cur))
        cur = np.rot90(cur)
    return min(sigs)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match sprites to frames by hole contour shape, allowing rotation, then nest sprites inside frames.

    The detection pipeline is inherited from the original task; only the matching key changes:
    instead of counting hole-components, compare the canonicalized shape of the non-object pixels.
    """
    g = np.array(grid, dtype=int)

    comps = find_components(g, bg=BG, connectivity=4, monochromatic=True, return_format="dict")

    MIN_FILL_RATIO = 0.5
    square_regions: list[tuple[int, dict]] = []
    for comp in comps:
        b = comp["bbox"]
        h = int(bbox_h(b))
        w = int(bbox_w(b))
        area = int(comp["area"])
        if h == w and 4 <= h <= 10 and area >= int(h * h * MIN_FILL_RATIO):
            square_regions.append((h, comp))

    assert square_regions

    sizes: dict[int, list[dict]] = {}
    for size, comp in square_regions:
        sizes.setdefault(int(size), []).append(comp)

    size_list = sorted(sizes.keys(), reverse=True)
    frame_size = int(size_list[0])
    sprite_size = int(size_list[1]) if len(size_list) > 1 else int(frame_size - 2)

    frames = sizes.get(frame_size, [])
    sprites_raw = sizes.get(sprite_size, [])
    assert frames and sprites_raw

    frame_positions = [(int(f["bbox"][0]), int(f["bbox"][1])) for f in frames]
    min_r = min(r for r, _ in frame_positions)
    min_c = min(c for _, c in frame_positions)
    max_r = max(r for r, _ in frame_positions)
    max_c = max(c for _, c in frame_positions)

    n_rows = int((max_r - min_r) // frame_size + 1)
    n_cols = int((max_c - min_c) // frame_size + 1)

    frame_grid = []
    for ri in range(n_rows):
        for ci in range(n_cols):
            inp_r = int(min_r + ri * frame_size)
            inp_c = int(min_c + ci * frame_size)
            region = g[inp_r : inp_r + frame_size, inp_c : inp_c + frame_size]
            border = np.concatenate([region[0, :], region[-1, :], region[:, 0], region[:, -1]])
            frame_color = int(Counter(map(int, border.flatten())).most_common(1)[0][0])
            hole_sig = _canonical_shape_signature(region[1:-1, 1:-1] != frame_color)
            frame_grid.append(
                {
                    "color": frame_color,
                    "out_r": int(ri * frame_size),
                    "out_c": int(ci * frame_size),
                    "shape": hole_sig,
                }
            )

    sprites = []
    for comp in sprites_raw:
        sprite_region = crop_bbox(g, comp["bbox"])
        sprite_color = int(comp["color"])
        hole_sig = _canonical_shape_signature(sprite_region != sprite_color)
        sprites.append(
            {
                "color": sprite_color,
                "shape": hole_sig,
                "content": sprite_region.copy(),
                "h": int(sprite_region.shape[0]),
                "w": int(sprite_region.shape[1]),
            }
        )

    matches: dict[tuple[int, int], dict] = {}
    used_sprites: set[int] = set()
    for f in frame_grid:
        for i, s in enumerate(sprites):
            if i in used_sprites:
                continue
            if f["shape"] == s["shape"]:
                matches[(int(f["out_r"]), int(f["out_c"]))] = s
                used_sprites.add(i)
                break

    out_h = int(n_rows * frame_size)
    out_w = int(n_cols * frame_size)
    out = np.full((out_h, out_w), BG, dtype=int)

    for f in frame_grid:
        fr = int(f["out_r"])
        fc = int(f["out_c"])
        frame_color = int(f["color"])
        out[fr : fr + frame_size, fc : fc + frame_size] = frame_color

        sprite = matches.get((fr, fc))
        if sprite is None:
            continue

        content = np.array(sprite["content"], dtype=int).copy()
        sprite_color = int(sprite["color"])
        content[content != sprite_color] = frame_color

        sh = int(sprite["h"])
        sw = int(sprite["w"])
        out[fr + 1 : fr + 1 + sh, fc + 1 : fc + 1 + sw] = content

    return out


# --------------- generator ---------------

FRAME_SIZE_RANGE = [7, 8]
GRID_DIM_RANGE = (1, 4)
N_RANGE = (2, 6)
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


def _coords_to_mask(sig: tuple[tuple[int, int], ...]) -> np.ndarray:
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
    sigs = []
    for _ in range(4):
        sig = _mask_signature(cur)
        if sig not in sigs:
            sigs.append(sig)
        cur = np.rot90(cur)
    return sigs


def _enumerate_shape_pool(core_size: int) -> list[dict]:
    pool: dict[tuple[tuple[int, int], ...], dict] = {}
    total = 1 << (core_size * core_size)
    for bits in range(1, total):
        area = int(bits.bit_count())
        if area < 2 or area >= core_size * core_size:
            continue
        mask = np.zeros((core_size, core_size), dtype=bool)
        for idx in range(core_size * core_size):
            if (bits >> idx) & 1:
                mask[idx // core_size, idx % core_size] = True
        if label_mask(mask, connectivity=4)[1] != 1:
            continue
        rot_sigs = _rotation_signatures(mask)
        if len(rot_sigs) < 2:
            continue
        key = min(rot_sigs)
        if key in pool:
            continue
        pool[key] = {
            "sig": key,
            "mask": _coords_to_mask(key),
            "area": area,
            "rot_sigs": rot_sigs,
        }
    return sorted(pool.values(), key=lambda item: (int(item["area"]), item["sig"]))


SHAPE_POOLS = {frame_size - 4: _enumerate_shape_pool(frame_size - 4) for frame_size in FRAME_SIZE_RANGE}


def _pick_shapes(core_size: int, count: int) -> list[dict]:
    pool = SHAPE_POOLS[core_size]
    by_area: dict[int, list[dict]] = {}
    for item in pool:
        by_area.setdefault(int(item["area"]), []).append(item)

    repeated_areas = [area for area, items in by_area.items() if len(items) >= 2]
    assert repeated_areas

    area = random.choice(repeated_areas)
    chosen = random.sample(by_area[area], 2)
    remaining = [item for item in pool if item["sig"] not in {x["sig"] for x in chosen}]
    chosen.extend(random.sample(remaining, count - 2))
    random.shuffle(chosen)
    return chosen


def _render_hole_host(core_size: int, hole_mask: np.ndarray) -> np.ndarray:
    host = np.zeros((core_size, core_size), dtype=bool)
    h, w = hole_mask.shape
    top = random.randint(0, core_size - h)
    left = random.randint(0, core_size - w)
    host[top : top + h, left : left + w] = hole_mask
    return host


def _render_frame(frame_size: int, frame_color: int, hole_mask: np.ndarray) -> np.ndarray:
    core_size = frame_size - 4
    frame = np.full((frame_size, frame_size), frame_color, dtype=int)
    host = _render_hole_host(core_size, hole_mask)
    inner = frame[2 : 2 + core_size, 2 : 2 + core_size]
    inner[host] = BG
    assert label_mask(frame != frame_color, connectivity=4)[1] == 1
    return frame


def _render_sprite(sprite_size: int, sprite_color: int, hole_mask: np.ndarray) -> np.ndarray:
    core_size = sprite_size - 2
    sprite = np.full((sprite_size, sprite_size), sprite_color, dtype=int)
    host = _render_hole_host(core_size, hole_mask)
    inner = sprite[1:-1, 1:-1]
    inner[host] = BG
    assert label_mask(sprite != sprite_color, connectivity=4)[1] == 1
    return sprite


def _shape_sig_after_rotation(mask: np.ndarray, k: int) -> tuple[tuple[int, int], ...]:
    return _mask_signature(np.rot90(mask, int(k)))


def _different_rotation(mask: np.ndarray, ref_k: int) -> int:
    ref_sig = _shape_sig_after_rotation(mask, ref_k)
    choices = [k for k in range(4) if _shape_sig_after_rotation(mask, k) != ref_sig]
    assert choices
    return random.choice(choices)


def _build_expected_output(
    frames: list[dict],
    sprites: list[dict],
    rows_grid: int,
    cols_grid: int,
    frame_size: int,
    sprite_size: int,
) -> np.ndarray:
    expected = np.full((rows_grid * frame_size, cols_grid * frame_size), BG, dtype=int)
    sprite_by_sig = {sprite["sig"]: sprite for sprite in sprites}
    for idx, frame in enumerate(frames):
        ri = idx // cols_grid
        ci = idx % cols_grid
        rr = int(ri * frame_size)
        cc = int(ci * frame_size)
        frame_color = int(frame["color"])
        expected[rr : rr + frame_size, cc : cc + frame_size] = frame_color
        content = sprite_by_sig[frame["sig"]]["patch"].copy()
        content[content == BG] = frame_color
        expected[rr + 1 : rr + 1 + sprite_size, cc + 1 : cc + 1 + sprite_size] = content
    return expected


def _gen_attempt() -> np.ndarray | None:
    frame_size = random.choice(FRAME_SIZE_RANGE)
    sprite_size = frame_size - 2
    core_size = frame_size - 4

    N = random.choice(_VALID_NS)
    rows_grid, cols_grid = random.choice(_LAYOUTS_BY_N[N])

    grid_pixel_h = rows_grid * frame_size
    grid_pixel_w = cols_grid * frame_size
    min_H = max(CANVAS_MIN, grid_pixel_h + sprite_size + SPRITE_PADDING)
    min_W = max(CANVAS_MIN, grid_pixel_w + sprite_size + SPRITE_PADDING)
    if min_H > CANVAS_MAX or min_W > CANVAS_MAX:
        return None
    H = random.randint(min_H, CANVAS_MAX)
    W = random.randint(min_W, CANVAS_MAX)

    available = [c for c in range(10) if c != BG]
    frame_colors = random.sample(available, N)
    sprite_colors = random.sample(available, N)

    shape_items = _pick_shapes(core_size, N)
    frame_items = shape_items[:]
    sprite_items = shape_items[:]
    random.shuffle(frame_items)
    random.shuffle(sprite_items)

    frame_rot: dict[tuple[tuple[int, int], ...], int] = {}
    sprite_rot: dict[tuple[tuple[int, int], ...], int] = {}
    item_by_sig = {item["sig"]: item for item in shape_items}
    for sig in item_by_sig:
        frame_rot[sig] = random.randint(0, 3)
        sprite_rot[sig] = random.randint(0, 3)
    if all(
        _shape_sig_after_rotation(item_by_sig[sig]["mask"], frame_rot[sig])
        == _shape_sig_after_rotation(item_by_sig[sig]["mask"], sprite_rot[sig])
        for sig in item_by_sig
    ):
        sig = random.choice(list(item_by_sig))
        sprite_rot[sig] = _different_rotation(item_by_sig[sig]["mask"], frame_rot[sig])

    frames: list[dict] = []
    for idx, item in enumerate(frame_items):
        sig = item["sig"]
        patch = _render_frame(frame_size, frame_colors[idx], np.rot90(item["mask"], frame_rot[sig]))
        frames.append({"sig": sig, "patch": patch, "color": frame_colors[idx], "area": int(item["area"])})

    sprites: list[dict] = []
    for idx, item in enumerate(sprite_items):
        sig = item["sig"]
        patch = _render_sprite(sprite_size, sprite_colors[idx], np.rot90(item["mask"], sprite_rot[sig]))
        sprites.append({"sig": sig, "patch": patch, "color": sprite_colors[idx], "area": int(item["area"])})

    # Core anti-degeneration checks: everything has one connected hole-component, but shape/area
    # still disambiguate the pairs.
    if len({frame["area"] for frame in frames}) == N:
        return None
    if not all(label_mask(frame["patch"] != frame["color"], connectivity=4)[1] == 1 for frame in frames):
        return None
    if not all(label_mask(sprite["patch"] != sprite["color"], connectivity=4)[1] == 1 for sprite in sprites):
        return None

    canvas = np.full((H, W), BG, dtype=int)
    r0 = random.randint(0, H - grid_pixel_h)
    c0 = random.randint(0, W - grid_pixel_w)
    for idx, frame in enumerate(frames):
        ri = idx // cols_grid
        ci = idx % cols_grid
        rr = int(r0 + ri * frame_size)
        cc = int(c0 + ci * frame_size)
        canvas[rr : rr + frame_size, cc : cc + frame_size] = frame["patch"]

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

    expected = _build_expected_output(frames, sprites, rows_grid, cols_grid, frame_size, sprite_size)
    if out.shape != expected.shape or not np.array_equal(out, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
