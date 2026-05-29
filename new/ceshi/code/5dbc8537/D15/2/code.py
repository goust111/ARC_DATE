# Variant of 5dbc8537
# Dimension: D15 — 符号编码与图例 (symbol_instructions)
# Change: 无符号编码或图例——精灵的颜色直接作为填充颜色使用... → 方向编码标记

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import (
    bbox_coords,
    colors_present,
    components_mask,
    crop_bbox,
    find_components,
    get_bg,
    paste,
    random_free_location,
)

# Design analysis for the D15 variant
# a) Concreteization and behavior trace:
#    - The template strip and its rectangular cavities stay unchanged.
#    - Each outside sprite is no longer a solid rectangle by itself. Instead it is a full
#      rectangular content block plus one diagonally touching corner marker pixel, making a
#      sparse (h+1) x (w+1) component. The marker is an instruction, not part of the filled
#      cavity. The output still contains only the rectangular content blocks.
#    - Small trace example: suppose a 2x3 cavity is filled in the output by a pattern whose
#      accent sits in the top-left corner. The input may show the same 2x3 content rotated
#      90 degrees counter-clockwise, together with a marker on the sprite's top-right outer
#      corner. Old logic would paste that source appearance directly and would also keep the
#      marker; new logic must strip the marker, rotate the core clockwise according to the
#      marker corner, and only then use the exact-cover placement.
# b) Complexity calibration:
#    - Rotation must be visible, so the generator removes 1x1 pieces and makes every content
#      rectangle bichromatic with an oriented asymmetric motif.
#    - At least several non-square rectangles are required, and odd-quarter turns are used,
#      so the corner instruction changes dimensions for many sprites and therefore matters for
#      the packing logic rather than being a cosmetic no-op.
# c) Hidden-assumption audit against the original main():
#    - Original code assumed every source component bbox exactly matched a hole rectangle.
#      That is false now because the marker enlarges the bbox by one row and one column.
#    - Original code pasted source patches opaquely. That would wrongly paste the marker and
#      would also copy bbox background cells. The new solver extracts the rectangular core and
#      pastes only the rotated core.
#    - Original scatter logic also relied on solid patches. The new generator must place sparse
#      marker+core sprites with transparent pasting, otherwise bbox background would erase
#      previously drawn pixels.
# d) Placement legality:
#    - Source components remain single 8-connected objects because the marker touches the core
#      only diagonally.
#    - The hole region is still a perfect exact cover by axis-aligned rectangles after rotation.
#      Generator-side oracle verification rejects any sample where the new solver cannot recover
#      the intended fill.
# e) Visual observability:
#    - The marker color is shared within a grid and appears only as the symbolic corner tag.
#    - The four marker corners are used repeatedly across a sample, and the rectangular cores
#      use a consistent asymmetric motif, so humans can compare source orientations with output
#      orientations and infer the legend from a few examples.
# f) Clean visualization:
#    - Markers stay outside the fill rectangle, so they do not occlude the actual sprite motif.
#      This keeps the symbolic layer and the content layer visually separable.
# g) Color inference:
#    - No invisible color-id ordering is used. Colors are only used as directly visible pixel
#      values; the instruction is encoded by marker position, not by numeric color identity.

GRID_MIN, GRID_MAX = 15, 25
THICK_MIN, THICK_MAX = 7, 10
OUTER_GAP = 10
SPRITE_COUNT_MIN, SPRITE_COUNT_MAX = 5, 8
HOLE_FRAC_LO, HOLE_FRAC_HI = 0.39, 0.50
CORRIDOR_COVERAGE = 0.78
MAX_RECT_LEN = 8
MAX_RECT_WIDTH = 6
MIN_RECT_LEN = 2
MIN_RECT_WIDTH = 2
MIN_SHAPE_GROUPS = 3
MIN_NON_SQUARE = 3
WIDTH_TOLERANCE = 2
MAX_RETRIES = 400

CORNERS = ("tl", "tr", "br", "bl")


def _rot_cw(arr: np.ndarray, steps: int) -> np.ndarray:
    return np.rot90(arr, k=(-int(steps)) % 4)


def _corner_index(corner: str) -> int:
    return CORNERS.index(str(corner))


def _corner_to_rotation(corner: str, offset: int = 0) -> int:
    return (_corner_index(corner) + int(offset)) % 4


def _corner_bbox_spec(ph: int, pw: int, corner: str) -> tuple[tuple[int, int], tuple[int, int, int, int]]:
    if corner == "tl":
        return (0, 0), (1, 1, ph - 1, pw - 1)
    if corner == "tr":
        return (0, pw - 1), (1, 0, ph - 1, pw - 2)
    if corner == "br":
        return (ph - 1, pw - 1), (0, 0, ph - 2, pw - 2)
    if corner == "bl":
        return (ph - 1, 0), (0, 1, ph - 2, pw - 1)
    raise ValueError(f"unknown corner {corner}")


def _extract_symbol_sprite(patch: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, str]:
    ph, pw = map(int, patch.shape)
    assert ph >= 3 and pw >= 3
    for corner in CORNERS:
        (mr, mc), (r0, c0, r1, c1) = _corner_bbox_spec(ph, pw, corner)
        if r1 < r0 or c1 < c0:
            continue
        expected = np.zeros((ph, pw), dtype=bool)
        expected[r0 : r1 + 1, c0 : c1 + 1] = True
        expected[mr, mc] = True
        if np.array_equal(mask, expected):
            core = patch[r0 : r1 + 1, c0 : c1 + 1].copy()
            return core, corner
    raise AssertionError("source component is not a valid corner-marked rectangle")


def _candidate_positions(hole_mask: np.ndarray, h: int, w: int) -> list[tuple[int, int]]:
    dh, dw = map(int, hole_mask.shape)
    pos: list[tuple[int, int]] = []
    if h > dh or w > dw:
        return pos
    for rr0 in range(dh - h + 1):
        for cc0 in range(dw - w + 1):
            if bool(hole_mask[rr0 : rr0 + h, cc0 : cc0 + w].all()):
                pos.append((int(rr0), int(cc0)))
    return pos


def _solve_with_offset(
    sprites: list[dict[str, object]], hole_mask: np.ndarray, base_col: int, offset: int
) -> np.ndarray | None:
    cache: dict[tuple[int, int], list[tuple[int, int]]] = {}
    transformed: list[dict[str, object]] = []
    total_area = 0
    for sp in sprites:
        rot = _corner_to_rotation(str(sp["corner"]), offset)
        patch = _rot_cw(np.array(sp["core"], dtype=int), rot)
        h, w = map(int, patch.shape)
        total_area += h * w
        key = (h, w)
        if key not in cache:
            cache[key] = _candidate_positions(hole_mask, h, w)
        if not cache[key]:
            return None
        transformed.append({"patch": patch, "h": h, "w": w, "cands": cache[key]})

    if total_area != int(hole_mask.sum()):
        return None

    order = sorted(
        range(len(transformed)),
        key=lambda i: (-int(transformed[i]["h"]) * int(transformed[i]["w"]), len(transformed[i]["cands"])),
    )
    used = ~hole_mask.copy()
    placement: list[tuple[int, int] | None] = [None] * len(transformed)

    def dfs(k: int) -> bool:
        if k == len(order):
            return True
        i = order[k]
        h, w = int(transformed[i]["h"]), int(transformed[i]["w"])
        for rr0, cc0 in transformed[i]["cands"]:
            if bool(used[rr0 : rr0 + h, cc0 : cc0 + w].any()):
                continue
            used[rr0 : rr0 + h, cc0 : cc0 + w] = True
            placement[i] = (int(rr0), int(cc0))
            if dfs(k + 1):
                return True
            placement[i] = None
            used[rr0 : rr0 + h, cc0 : cc0 + w] = False
        return False

    if not dfs(0):
        return None

    out = np.full(hole_mask.shape, int(base_col), dtype=int)
    for i, pos in enumerate(placement):
        assert pos is not None
        rr0, cc0 = map(int, pos)
        paste(out, np.array(transformed[i]["patch"], dtype=int), rr0, cc0, overwrite=True)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct a strip template by inserting outside sprites into its rectangular cavities.

    Input structure:
    - The dominant non-background color forms a large strip-like template whose holes are still
      axis-aligned rectangles.
    - Every outside sprite is a rectangular content block plus one single corner marker pixel
      attached diagonally outside the block.

    Rule:
    - Detect the template exactly as before, but treat each source component as a marker-tagged
      rectangle rather than as the rectangle itself.
    - Remove the marker, rotate the rectangular core according to the marker corner, and then
      pack those rotated rectangles into the template holes.

    Key insight:
    - The marker's position changes the rectangle's orientation before matching sizes to holes,
      so the puzzle is now "infer corner instruction, then exact-cover the cavities".
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    base_candidates: list[tuple[int, int]] = []
    for col in colors_present(g, exclude={bg}):
        max_area = 0
        for coords in components_mask(g == int(col), connectivity=8):
            max_area = max(int(max_area), int(coords.shape[0]))
        base_candidates.append((int(max_area), int(col)))

    assert base_candidates
    base_candidates.sort(reverse=True)
    if len(base_candidates) > 1:
        assert base_candidates[0][0] > base_candidates[1][0]
    base_col = int(base_candidates[0][1])

    base_coords = np.argwhere(g == int(base_col))
    assert base_coords.size
    r0, c0, r1, c1 = bbox_coords(base_coords)
    dst = crop_bbox(g, (int(r0), int(c0), int(r1), int(c1)))
    hole_mask = dst != int(base_col)

    masked = g.copy()
    masked[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(bg)
    sprite_comps = find_components(masked, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    sprites: list[dict[str, object]] = []
    for comp in sprite_comps:
        pr0, pc0, pr1, pc1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (pr0, pc0, pr1, pc1))
        mask = np.zeros_like(patch, dtype=bool)
        coords = comp["coords"]
        mask[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        core, corner = _extract_symbol_sprite(patch, mask)
        sprites.append({"core": core, "corner": corner})

    assert sprites

    # Try the canonical legend first, then nearby cyclic alternatives. The generator uses the
    # canonical order, but searching keeps the solver tied to the visible instruction instead of
    # assuming source bboxes already equal hole bboxes.
    for offset in (0, 1, 2, 3):
        out = _solve_with_offset(sprites, hole_mask, base_col, offset)
        if out is not None:
            return out
    raise AssertionError("no valid corner-to-rotation interpretation")


def _split_lengths(total: int, count: int) -> list[int] | None:
    if count <= 0 or count * MIN_RECT_LEN > total or count * MAX_RECT_LEN < total:
        return None
    lengths = [MIN_RECT_LEN] * count
    remaining = total - count * MIN_RECT_LEN
    while remaining:
        choices = [i for i, v in enumerate(lengths) if v < MAX_RECT_LEN]
        if not choices:
            return None
        i = random.choice(choices)
        add = random.randint(1, min(remaining, MAX_RECT_LEN - lengths[i]))
        lengths[i] += add
        remaining -= add
    random.shuffle(lengths)
    return lengths


def _choose_widths(lengths: list[int], thick: int, full_area: int) -> list[int] | None:
    max_w = min(MAX_RECT_WIDTH, thick - 2)
    if max_w < MIN_RECT_WIDTH:
        return None
    lo = math.ceil(HOLE_FRAC_LO * full_area)
    hi = math.floor(HOLE_FRAC_HI * full_area)
    paths: dict[int, list[int]] = {0: []}
    for h in lengths:
        opts = list(range(MIN_RECT_WIDTH, max_w + 1))
        random.shuffle(opts)
        items = list(paths.items())
        random.shuffle(items)
        nxt: dict[int, list[int]] = {}
        for cur, path in items:
            for w in opts:
                s = cur + h * w
                if s <= hi and s not in nxt:
                    nxt[s] = path + [w]
        paths = nxt
        if not paths:
            return None
    valid = [(s, p) for s, p in paths.items() if lo <= s <= hi]
    if not valid:
        return None
    target = random.randint(lo, hi)
    best = min(abs(s - target) for s, _ in valid)
    near = [p for s, p in valid if abs(s - target) <= best + WIDTH_TOLERANCE]
    return random.choice(near)


def _corridor_rects(long: int, thick: int, count: int) -> list[tuple[int, int, int, int]] | None:
    min_len = max(count * MIN_RECT_LEN, round(CORRIDOR_COVERAGE * long))
    min_len = min(min_len, long - 1)
    if min_len > long - 1:
        return None
    hole_len = random.randint(min_len, long - 1)
    lengths = _split_lengths(hole_len, count)
    if lengths is None:
        return None
    widths = _choose_widths(lengths, thick, long * thick)
    if widths is None:
        return None

    start = random.randint(0, long - hole_len)
    r = start
    prev_c = prev_w = None
    rects: list[tuple[int, int, int, int]] = []
    for h, w in zip(lengths, widths):
        c_hi = thick - w - 1
        if c_hi < 1:
            return None
        if prev_c is None:
            c = random.randint(1, c_hi)
        else:
            c_lo = max(1, prev_c - w + 1)
            c_hi = min(c_hi, prev_c + prev_w - 1)
            if c_lo > c_hi:
                return None
            c = random.randint(c_lo, c_hi)
        rects.append((r, c, h, w))
        r += h
        prev_c, prev_w = c, w
    return rects


def _make_oriented_patch(h: int, w: int, primary: int, secondary: int) -> np.ndarray:
    out = np.full((h, w), int(primary), dtype=int)
    out[0, :] = int(secondary)
    out[:, 0] = int(secondary)
    if h > 2 and w > 2:
        out[1, -1] = int(secondary)
    return out


def _assign_group_patches(
    rects: list[tuple[int, int, int, int]], bg: int, base: int, marker: int
) -> dict[tuple[int, int], np.ndarray] | None:
    groups = sorted({(h, w) for _, _, h, w in rects})
    if len(groups) < MIN_SHAPE_GROUPS:
        return None

    available = [c for c in range(10) if c not in (bg, base, marker)]
    if len(available) < 3:
        return None

    random.shuffle(available)
    patches: dict[tuple[int, int], np.ndarray] = {}
    for i, key in enumerate(groups):
        primary = available[i % len(available)]
        secondary_choices = [c for c in available if c != primary]
        secondary = random.choice(secondary_choices)
        patches[key] = _make_oriented_patch(key[0], key[1], primary, secondary)
    return patches


def _build_symbol_sprite(core: np.ndarray, marker: int, corner: str, bg: int) -> np.ndarray:
    h, w = map(int, core.shape)
    out = np.full((h + 1, w + 1), int(bg), dtype=int)
    if corner == "tl":
        out[0, 0] = int(marker)
        out[1:, 1:] = core
    elif corner == "tr":
        out[0, w] = int(marker)
        out[1:, :w] = core
    elif corner == "br":
        out[:h, :w] = core
        out[h, w] = int(marker)
    elif corner == "bl":
        out[:h, 1:] = core
        out[h, 0] = int(marker)
    else:
        raise ValueError(f"unknown corner {corner}")
    return out


def _render_output(
    dh: int, dw: int, base: int, rects: list[tuple[int, int, int, int]], pieces: dict[int, np.ndarray]
) -> np.ndarray:
    out = np.full((dh, dw), int(base), dtype=int)
    for i, (r, c, _, _) in enumerate(rects):
        paste(out, pieces[i], int(r), int(c), overwrite=True)
    return out


def _scatter_sprites(
    H: int,
    W: int,
    bg: int,
    base: int,
    tr0: int,
    tc0: int,
    dh: int,
    dw: int,
    rects: list[tuple[int, int, int, int]],
    pieces: dict[int, np.ndarray],
) -> np.ndarray:
    inp = np.full((H, W), int(bg), dtype=int)
    inp[tr0 : tr0 + dh, tc0 : tc0 + dw] = int(base)
    for r, c, h, w in rects:
        inp[tr0 + r : tr0 + r + h, tc0 + c : tc0 + c + w] = int(bg)

    placement = np.full((H, W), int(bg), dtype=int)
    placement[tr0 : tr0 + dh, tc0 : tc0 + dw] = int(base)
    order = list(range(len(rects)))
    random.shuffle(order)
    order.sort(key=lambda i: pieces[i].shape[0] * pieces[i].shape[1], reverse=True)
    for i in order:
        patch = pieces[i]
        r, c = random_free_location(placement, patch, transparent=bg, padding=1, padding_connectivity=8)
        paste(inp, patch, r, c, transparent=bg, overwrite=True)
        paste(placement, patch, r, c, transparent=bg, overwrite=True)
    return inp


def _build_piece_sets(
    rects: list[tuple[int, int, int, int]], bg: int, base: int, marker: int
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], list[int]] | None:
    group_patches = _assign_group_patches(rects, bg, base, marker)
    if group_patches is None:
        return None

    non_square = [i for i, (_, _, h, w) in enumerate(rects) if h != w]
    if len(non_square) < MIN_NON_SQUARE:
        return None

    n = len(rects)
    rotations = [0, 1, 2, 3]
    while len(rotations) < n:
        rotations.append(random.randrange(4))
    random.shuffle(rotations)
    if not any(rotations[i] % 2 == 1 for i in non_square):
        i = random.choice(non_square)
        rotations[i] = random.choice([1, 3])

    out_pieces: dict[int, np.ndarray] = {}
    src_pieces: dict[int, np.ndarray] = {}
    for i, (_, _, h, w) in enumerate(rects):
        final_patch = group_patches[(h, w)].copy()
        rot = int(rotations[i])
        input_core = np.rot90(final_patch, k=rot)
        src_pieces[i] = _build_symbol_sprite(input_core, marker, CORNERS[rot], bg)
        out_pieces[i] = final_patch
    return out_pieces, src_pieces, rotations


def _gen_attempt() -> np.ndarray | None:
    vertical = random.random() < 0.5
    thick = random.randint(THICK_MIN, THICK_MAX)
    if vertical:
        H = random.randint(GRID_MIN, GRID_MAX)
        W = random.randint(max(GRID_MIN, thick + OUTER_GAP), GRID_MAX)
        dh, dw = H, thick
        tr0, tc0 = 0, (0 if random.random() < 0.5 else W - thick)
    else:
        H = random.randint(max(GRID_MIN, thick + OUTER_GAP), GRID_MAX)
        W = random.randint(GRID_MIN, GRID_MAX)
        dh, dw = thick, W
        tr0, tc0 = (0 if random.random() < 0.5 else H - thick), 0
    long = dh if vertical else dw

    total = random.randint(SPRITE_COUNT_MIN, SPRITE_COUNT_MAX)
    rects_long = _corridor_rects(long, thick, total)
    if rects_long is None:
        return None
    rects = rects_long if vertical else [(c, r, w, h) for r, c, h, w in rects_long]

    bg, base, marker = random.sample(range(10), 3)
    built = _build_piece_sets(rects, bg, base, marker)
    if built is None:
        return None
    out_pieces, src_pieces, _ = built

    out = _render_output(dh, dw, base, rects, out_pieces)
    try:
        inp = _scatter_sprites(H, W, bg, base, tr0, tc0, dh, dw, rects, src_pieces)
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 5dbc8537 D15 input")
