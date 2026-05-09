# Variant of 58f5dbd5
# Dimension: D10 — 轨迹：源对象处理 (trajectory)
# Change: motif在输出中消失（不保留源），仅在方块上留下雕刻痕迹... → motif移动轨迹涂色

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import (
    Color,
    bbox_h,
    bbox_w,
    crop_bbox,
    find_components,
    get_bg,
    is_contiguous,
    paste,
)


# Design analysis for the D10 variant
# ===================================
# a) Concrete behavior:
#    The original task moved same-colored 3x3 motifs conceptually into 5x5 blocks and then
#    discarded the whole input scene by packing only the carved blocks. Here the carving rule
#    stays the same, but the output remains on the original canvas. Each matched motif is
#    treated as a moving source: its own colored pixels are recolored to a fixed trail color,
#    and a 1-cell axis-aligned trail is drawn from the motif center toward the center of its
#    matched block. The block itself is still carved with the centered motif mask. Unmatched
#    distractor motifs remain unchanged. Example: a motif above a block used to disappear in
#    the packed output; now it stays at its source location only as black trail pixels, the
#    gap between source and block gets a black line, and the destination block is carved in
#    place inside the full grid.
# b) Complexity calibration:
#    The main visual change must be obvious in 2-4 ARC examples, so the generator keeps the
#    original object vocabulary (3-6 colored 5x5 blocks, 3x3 connected motifs, color-matched
#    pairing) and adds repeated trail evidence. To avoid a degenerate variant where source and
#    destination are too far or paths are ambiguous, every sample uses one of a small number of
#    side-attachment schemes: motifs sit on outward-facing sides of the block grid, so multiple
#    same-style trails appear in one sample.
# c) Implicit-assumption audit of the original solver:
#    - Original step 1 (find solid 5x5 blocks) remains valid and unchanged.
#    - Original step 2 assumed the smallest same-colored non-5x5 component is the motif. This
#      still works only if each block color is unique and has exactly one matching motif; the
#      generator preserves that invariant.
#    - Original step 3 carved isolated 5x5 tiles and ignored source position entirely. That is
#      no longer sufficient: the new solver must also know each motif bbox and source coords,
#      because output depends on where the motif started.
#    - Original step 4 packed tiles into a new compact canvas. That assumption is invalid in the
#      variant; the output must instead start from the input grid and modify it in place.
# d) Placement legality:
#    Straight trails are only readable if they do not pass through unrelated blocks. The
#    generator therefore places matched motifs only on outward-facing perimeter sides of the
#    block rectangle: top/bottom schemes for 1x3, 2x2, 2x3 layouts and left/right schemes for
#    3x1, 2x2, 3x2 layouts. A 2-cell gap prevents 8-connected merging between motif and block.
#    Distractors are added only in cells that avoid all future trail corridors.
# e) Visual observability:
#    The trail color is a fixed visible color (black), never the background. The block colors
#    stay distinct, the motif shape still controls the carved hole pattern, and several motifs
#    in each example transform with the same trail behavior, making the rule inferable without
#    hidden ordering tricks.
# f) Visual cleanliness:
#    Paths are 1-cell-wide and axis-aligned, blocks stay intact except for centered carved
#    holes, and distractors are kept away from the reserved corridors. This avoids ambiguous
#    overpainting and keeps each element visually separable.
# g) Color-change inferability:
#    Only one new color behavior is introduced: matched motifs and their connecting path become
#    black. Because the black pixels appear exactly between a colored motif and its same-colored
#    block, humans can infer that black is a trajectory marker rather than an arbitrary remap.
# h) New solver assumptions introduced here:
#    - Background is never black.
#    - Each matched motif shares exactly one axis-aligned center line with its destination block.
#    - Matching motifs use full 3x3 bboxes (their masks touch all four sides), so the center of
#      the bbox is a stable path anchor. The generator enforces all three conditions.


TRAIL_COLOR = int(Color.BLACK)


def mask_from_coords(coords, shape: tuple[int, int]) -> np.ndarray:
    """Build a local boolean mask with True at in-bounds coords."""
    m = np.zeros(shape, dtype=bool)
    H, W = shape
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if 0 <= rr < H and 0 <= cc < W:
            m[rr, cc] = True
    return m


def _center_of_bbox(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, bbox)
    return int((r0 + r1) // 2), int((c0 + c1) // 2)


def _cells_on_axis_path(src: tuple[int, int], dst: tuple[int, int]) -> list[tuple[int, int]]:
    sr, sc = map(int, src)
    tr, tc = map(int, dst)
    if sr == tr:
        step = 1 if tc >= sc else -1
        return [(int(sr), int(c)) for c in range(sc, tc + step, step)]
    if sc == tc:
        step = 1 if tr >= sr else -1
        return [(int(r), int(sc)) for r in range(sr, tr + step, step)]

    cells = [(int(sr), int(sc))]
    step_r = 1 if tr > sr else -1
    for r in range(sr + step_r, tr + step_r, step_r):
        cells.append((int(r), int(sc)))
    step_c = 1 if tc > sc else -1
    for c in range(sc + step_c, tc + step_c, step_c):
        cells.append((int(tr), int(c)))
    return cells


def _paint_trail(
    out: np.ndarray,
    src_bbox: tuple[int, int, int, int],
    src_mask: np.ndarray,
    dst_bbox: tuple[int, int, int, int],
) -> None:
    """Recolor the source motif and paint the connecting axis-aligned trail."""
    sr0, sc0, _sr1, _sc1 = map(int, src_bbox)
    dr0, dc0, dr1, dc1 = map(int, dst_bbox)

    for rr, cc in np.argwhere(src_mask):
        out[sr0 + int(rr), sc0 + int(cc)] = int(TRAIL_COLOR)

    for r, c in _cells_on_axis_path(_center_of_bbox(src_bbox), _center_of_bbox(dst_bbox)):
        if dr0 <= int(r) <= dr1 and dc0 <= int(c) <= dc1:
            continue
        out[int(r), int(c)] = int(TRAIL_COLOR)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Carve color-matched motifs into their blocks and draw source-to-block trails in place.

    Input structure:
    - bg = most common color, and black is reserved as a non-background trail color.
    - Several solid 5x5 monochrome blocks, each a distinct non-bg color.
    - For each block color, one small same-colored 3x3-bbox motif sits on an outward side of the block layout; extra motifs may appear as distractors.

    Rule:
    - Match each motif to the solid 5x5 block of the same color.
    - Center that motif inside the block and carve its mask out to background, but keep the full canvas instead of repacking.
    - Recolor the matched source motif to black and paint a black straight trail from its source position toward the matched block.

    Key insight: The motif is still a carving stencil, but the output now records the motion path from source to destination rather than discarding the source scene.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    assert int(bg) != int(TRAIL_COLOR)

    blocks: list[dict[str, object]] = []
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != 5 or bbox_w(b) != 5:
            continue
        col = int(comp["color"])
        patch = crop_bbox(g, b)
        if bool((patch == int(col)).all()):
            r0, c0, _r1, _c1 = map(int, b)
            blocks.append({"col": int(col), "bbox": (int(r0), int(c0), int(r0 + 4), int(c0 + 4))})

    assert blocks

    motifs: dict[int, dict[str, object]] = {}
    for block in blocks:
        bcol = int(block["col"])
        candidates: list[tuple[int, int, int, int, tuple[int, int, int, int], np.ndarray]] = []
        comps_col = find_components(
            g,
            bg=bg,
            connectivity=8,
            monochromatic=True,
            colors=[int(bcol)],
            return_format="dict",
        )
        for comp in comps_col:
            b = comp["bbox"]
            if bbox_h(b) == 5 and bbox_w(b) == 5:
                patch = crop_bbox(g, b)
                if bool((patch == int(bcol)).all()):
                    continue

            r0, c0, r1, c1 = map(int, b)
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            rel_coords = [(int(r) - r0, int(c) - c0) for r, c in comp["coords"]]
            m = mask_from_coords(rel_coords, (h, w))
            candidates.append((int(comp["area"]), int(h * w), int(r0), int(c0), (r0, c0, r1, c1), m))

        if candidates:
            candidates.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
            _area, _bbox_area, _r0, _c0, bbox, mask = candidates[0]
            motifs[int(bcol)] = {"bbox": bbox, "mask": mask}

    out = g.copy()

    for block in blocks:
        col = int(block["col"])
        br0, bc0, _br1, _bc1 = map(int, block["bbox"])
        mm = motifs.get(int(col))
        if mm is None:
            continue

        mask = np.array(mm["mask"], dtype=bool)
        mh, mw = map(int, mask.shape)
        off_r = (5 - mh) // 2
        off_c = (5 - mw) // 2
        for rr, cc in np.argwhere(mask):
            out[br0 + off_r + int(rr), bc0 + off_c + int(cc)] = int(bg)

        _paint_trail(out, mm["bbox"], mask, block["bbox"])

    return out


# --------------- generator (D10 variant) ---------------

CANVAS_HEIGHT_RANGE = (18, 22)
CANVAS_WIDTH_RANGE = (18, 24)
COLOR_COUNT = 10

DST_BLOCK_SIZE = 5
DST_BLOCK_STEP = 6
DST_LAYOUTS = {
    3: [(1, 3), (3, 1)],
    4: [(2, 2)],
    6: [(2, 3), (3, 2)],
}

MOTIF_SIZE = 3
MOTIF_AREA_RANGE = (4, 8)
PATH_GAP = 2
OUTER_MARGIN = 1
MAX_DISTRACTORS = 2
GENERATION_RETRIES = 400


def _spanned_size(count: int, item_size: int, step: int) -> int:
    return (count - 1) * step + item_size


def _make_masks_by_area() -> dict[int, list[np.ndarray]]:
    min_area, max_area = MOTIF_AREA_RANGE
    masks: dict[int, list[np.ndarray]] = {area: [] for area in range(min_area, max_area + 1)}
    for bits in itertools.product([False, True], repeat=MOTIF_SIZE * MOTIF_SIZE):
        m = np.array(bits, dtype=bool).reshape(MOTIF_SIZE, MOTIF_SIZE)
        area = int(m.sum())
        if area not in masks:
            continue
        if not (m[0].any() and m[-1].any() and m[:, 0].any() and m[:, -1].any()):
            continue
        if is_contiguous(m.astype(int), background=0, connectivity=8):
            masks[area].append(m)
    return masks


MASKS_BY_AREA = _make_masks_by_area()


def _random_mask() -> np.ndarray:
    area = random.randint(*MOTIF_AREA_RANGE)
    return random.choice(MASKS_BY_AREA[area]).copy()


def _valid_schemes(R: int, C: int) -> list[str]:
    if R == 1:
        return ["up", "down"]
    if C == 1:
        return ["left", "right"]
    if (R, C) == (2, 2):
        return ["vertical", "horizontal"]
    if R == 2:
        return ["vertical"]
    if C == 2:
        return ["horizontal"]
    return []


def _scheme_padding(scheme: str) -> tuple[int, int, int, int]:
    extra = MOTIF_SIZE + PATH_GAP
    if scheme == "up":
        return extra, 0, 0, 0
    if scheme == "down":
        return 0, extra, 0, 0
    if scheme == "left":
        return 0, 0, extra, 0
    if scheme == "right":
        return 0, 0, 0, extra
    if scheme == "vertical":
        return extra, extra, 0, 0
    if scheme == "horizontal":
        return 0, 0, extra, extra
    raise ValueError(f"unknown scheme: {scheme}")


def _feasible_configs(H: int, W: int) -> list[tuple[int, int, str]]:
    configs: list[tuple[int, int, str]] = []
    for count, layouts in DST_LAYOUTS.items():
        for R, C in layouts:
            dst_h = _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
            dst_w = _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
            for scheme in _valid_schemes(R, C):
                top, bottom, left, right = _scheme_padding(scheme)
                need_h = top + dst_h + bottom + 2 * OUTER_MARGIN
                need_w = left + dst_w + right + 2 * OUTER_MARGIN
                if need_h <= H and need_w <= W:
                    configs.append((R, C, scheme))
    return configs


def _choose_config(H: int, W: int) -> tuple[int, int, str] | None:
    configs = _feasible_configs(H, W)
    if not configs:
        return None
    counts = sorted({int(R * C) for R, C, _scheme in configs})
    count = random.choice(counts)
    return random.choice([cfg for cfg in configs if cfg[0] * cfg[1] == count])


def _place_blocks(H: int, W: int, R: int, C: int, scheme: str) -> tuple[int, int]:
    dst_h = _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    dst_w = _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    top, bottom, left, right = _scheme_padding(scheme)
    occ_h = top + dst_h + bottom
    occ_w = left + dst_w + right
    occ_r = random.randint(OUTER_MARGIN, H - OUTER_MARGIN - occ_h)
    occ_c = random.randint(OUTER_MARGIN, W - OUTER_MARGIN - occ_w)
    return int(occ_r + top), int(occ_c + left)


def _side_for_block(i: int, j: int, R: int, C: int, scheme: str) -> str:
    if scheme in {"up", "down", "left", "right"}:
        return scheme
    if scheme == "vertical":
        return "up" if i == 0 else "down"
    if scheme == "horizontal":
        return "left" if j == 0 else "right"
    raise ValueError(f"unknown scheme: {scheme}")


def _motif_origin_for_block(block_r: int, block_c: int, side: str) -> tuple[int, int]:
    if side == "up":
        return int(block_r - MOTIF_SIZE - PATH_GAP), int(block_c + 1)
    if side == "down":
        return int(block_r + DST_BLOCK_SIZE + PATH_GAP), int(block_c + 1)
    if side == "left":
        return int(block_r + 1), int(block_c - MOTIF_SIZE - PATH_GAP)
    if side == "right":
        return int(block_r + 1), int(block_c + DST_BLOCK_SIZE + PATH_GAP)
    raise ValueError(f"unknown side: {side}")


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            src_r0 = max(0, -dr)
            src_r1 = mask.shape[0] - max(0, dr)
            src_c0 = max(0, -dc)
            src_c1 = mask.shape[1] - max(0, dc)
            dst_r0 = max(0, dr)
            dst_r1 = dst_r0 + (src_r1 - src_r0)
            dst_c0 = max(0, dc)
            dst_c1 = dst_c0 + (src_c1 - src_c0)
            out[dst_r0:dst_r1, dst_c0:dst_c1] |= mask[src_r0:src_r1, src_c0:src_c1]
    return out


def _paint_motif(grid: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> None:
    patch = np.full((MOTIF_SIZE, MOTIF_SIZE), int(color), dtype=int)
    paste(grid, patch, r0, c0, mask=mask)


def _reserved_mask(
    H: int,
    W: int,
    assignments: list[dict[str, object]],
    blocks: list[dict[str, int]],
) -> np.ndarray:
    reserved = np.zeros((H, W), dtype=bool)
    for block in blocks:
        br0 = int(block["r0"])
        bc0 = int(block["c0"])
        reserved[br0 : br0 + DST_BLOCK_SIZE, bc0 : bc0 + DST_BLOCK_SIZE] = True
    for item in assignments:
        mr0 = int(item["motif_r"])
        mc0 = int(item["motif_c"])
        reserved[mr0 : mr0 + MOTIF_SIZE, mc0 : mc0 + MOTIF_SIZE] = True
        dst_bbox = (
            int(item["block_r"]),
            int(item["block_c"]),
            int(item["block_r"]) + 4,
            int(item["block_c"]) + 4,
        )
        for r, c in _cells_on_axis_path(
            (int(item["motif_r"]) + 1, int(item["motif_c"]) + 1),
            (int(item["block_r"]) + 2, int(item["block_c"]) + 2),
        ):
            br0, bc0, br1, bc1 = dst_bbox
            if br0 <= int(r) <= br1 and bc0 <= int(c) <= bc1:
                continue
            reserved[int(r), int(c)] = True
    return reserved


def _place_distractors(
    grid: np.ndarray,
    colors: list[int],
    reserved: np.ndarray,
) -> None:
    if not colors:
        return

    H, W = grid.shape
    occupied = grid != int(get_bg(grid))
    forbidden = _dilate8(occupied | reserved)
    candidates = [
        (int(r), int(c))
        for r in range(H - MOTIF_SIZE + 1)
        for c in range(W - MOTIF_SIZE + 1)
        if not forbidden[r : r + MOTIF_SIZE, c : c + MOTIF_SIZE].any()
    ]
    random.shuffle(candidates)

    for color in colors:
        if not candidates:
            break
        r0, c0 = candidates.pop()
        _paint_motif(grid, r0, c0, int(color), _random_mask())
        occupied[r0 : r0 + MOTIF_SIZE, c0 : c0 + MOTIF_SIZE] = True
        forbidden = _dilate8(occupied | reserved)
        candidates = [
            (int(r), int(c))
            for r, c in candidates
            if not forbidden[r : r + MOTIF_SIZE, c : c + MOTIF_SIZE].any()
        ]


def _choose_palette(bg: int, count: int, distractors: int) -> tuple[list[int], list[int]]:
    palette = [c for c in range(COLOR_COUNT) if c not in {bg, TRAIL_COLOR}]
    random.shuffle(palette)
    block_colors = palette[:count]
    distractor_colors = palette[count : count + distractors]
    return block_colors, distractor_colors


def _render_expected(
    inp: np.ndarray,
    bg: int,
    assignments: list[dict[str, object]],
) -> np.ndarray:
    out = np.array(inp, dtype=int)
    for item in assignments:
        mask = np.array(item["mask"], dtype=bool)
        br0 = int(item["block_r"])
        bc0 = int(item["block_c"])
        for rr, cc in np.argwhere(mask):
            out[br0 + 1 + int(rr), bc0 + 1 + int(cc)] = int(bg)
        _paint_trail(
            out,
            (
                int(item["motif_r"]),
                int(item["motif_c"]),
                int(item["motif_r"]) + 2,
                int(item["motif_c"]) + 2,
            ),
            mask,
            (br0, bc0, br0 + 4, bc0 + 4),
        )
    return out


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*CANVAS_HEIGHT_RANGE)
    W = random.randint(*CANVAS_WIDTH_RANGE)
    bg = random.choice([c for c in range(COLOR_COUNT) if c != TRAIL_COLOR])

    config = _choose_config(H, W)
    if config is None:
        return None
    R, C, scheme = config

    dst_r, dst_c = _place_blocks(H, W, R, C, scheme)
    count = R * C

    if scheme in {"up", "down", "left", "right"}:
        distractor_count = random.randint(0, MAX_DISTRACTORS)
    else:
        distractor_count = random.randint(0, 1)
    block_colors, distractor_colors = _choose_palette(bg, count, distractor_count)
    motif_masks = {int(color): _random_mask() for color in block_colors}

    inp = np.full((H, W), int(bg), dtype=int)
    blocks: list[dict[str, int]] = []
    assignments: list[dict[str, object]] = []

    for i in range(R):
        for j in range(C):
            color = int(block_colors[i * C + j])
            br0 = int(dst_r + i * DST_BLOCK_STEP)
            bc0 = int(dst_c + j * DST_BLOCK_STEP)
            paste(inp, np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), color, dtype=int), br0, bc0)
            blocks.append({"r0": br0, "c0": bc0, "color": color})

            side = _side_for_block(i, j, R, C, scheme)
            mr0, mc0 = _motif_origin_for_block(br0, bc0, side)
            mask = motif_masks[color]
            _paint_motif(inp, mr0, mc0, color, mask)
            assignments.append(
                {
                    "color": color,
                    "block_r": br0,
                    "block_c": bc0,
                    "motif_r": mr0,
                    "motif_c": mc0,
                    "mask": mask,
                }
            )

    reserved = _reserved_mask(H, W, assignments, blocks)
    _place_distractors(inp, distractor_colors, reserved)

    expected = _render_expected(inp, bg, assignments)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 58f5dbd5 D10 input")
