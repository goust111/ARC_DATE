# Combo variant of bf45cf4b (index 8)
# combo_spec: D2/3+D8/2
# Name: 相邻位置角色识别

from __future__ import annotations

import itertools

import numpy as np

from arc_utils import colors_present, crop_bbox, find_components, get_bg, paste

# Design analysis for the fused combo variant
# -------------------------------------------
# a) Concrete behavior and trace:
#    - Base bf45cf4b:
#      * two 8-disconnected foreground objects
#      * the multicolor square object is the tile
#      * the monochrome holey rectangle is the control mask
#      * output stamps the tile once per filled control pixel
#    - Fused D2/3 + D8/2:
#      * the two semantic objects are still "control mask" and "tile"
#      * they are now arranged as one near-adjacent pair, either ctrl-left/tile-right
#        or ctrl-top/tile-bottom, with gap 0 or 1
#      * role assignment is by relative position only: upper/left object is ctrl,
#        lower/right object is tile
#      * the transformation itself is unchanged: every non-background cell in ctrl
#        stamps one full copy of the tile
#    - Small trace:
#      * suppose a 4x4 holey control patch sits to the left of a 3x3 tile
#      * original logic would try "more colors means tile"; this combo instead first
#        reads the left object as ctrl because it is leftmost, then scales by the
#        3x3 tile on the right
#      * if the same pair is stacked vertically, the upper object is ctrl and the
#        lower object is tile
# b) Complexity calibration:
#    - D8/2 is made visible by using both gap=0 and gap=1 layouts; gap=0 merges the
#      pair into one 8-connected foreground union, while gap=1 keeps two unions.
#    - D2/3 would be degenerate if the control stayed monochrome and the tile stayed
#      strictly "more colorful". To make role inference genuinely depend on position,
#      the generator minimally relaxes control appearance: the control uses two colors,
#      but semantically it is still just a binary mask (`ctrl != bg`).
#    - Both ctrl and tile are often square and both use two colors, so the old
#      "multicolor/square means tile" heuristic is not reliable.
# c) Implicit-assumption audit of the original main():
#    - `find_components(..., monochromatic=False)` returning exactly two objects is
#      broken by gap=0 touching: the whole foreground can merge into one union.
#    - Sorting two isolated objects by color-count and squareness is intentionally
#      broken: both objects can now be two-color and square.
#    - Using one distinguished ctrl color as the mask is invalid. The fused solver
#      must interpret every non-background ctrl cell as filled.
#    - Cropping object bboxes is still safe because the generator guarantees only
#      external adjacency, never overlap.
#    - For touching cases, the solver now assumes ctrl-palette and tile-palette are
#      disjoint so the merged pair can be separated by color partition. The generator
#      enforces this exactly.
# d) Placement legality:
#    - Objects are first composed in a tight local pair canvas, then embedded into a
#      12-18 canvas.
#    - For gap=0, contact is real 4-neighbor side contact between non-background
#      boundary pixels, never a diagonal-only or overlapping contact.
#    - For both orientations, ctrl is always on the upper/left side and tile on the
#      lower/right side, matching the fused specification.
# e) Visual observability:
#    - A human sees one holey patch and one compact square patch placed right next to
#      each other, sometimes horizontally and sometimes vertically.
#    - Across examples, the persistent rule is spatial: upper/left behaves like the
#      mask source, lower/right behaves like the repeated block.
#    - The touching cases visibly differ from the base task because there is no blank
#      moat separating the two shapes.
# f) Visual cleanliness:
#    - Only one pair is present; there are no extra markers or overlapping paints.
#    - Colors are few: 2 ctrl colors and 2 tile colors, with disjoint palettes.
#    - The pair is side-adjacent rather than interleaved, so both shapes remain easy
#      to read by eye.
# g) Change inferability / anti-degeneracy:
#    - No recoloring rule is introduced; colors remain structural only.
#    - The generator is constructed so the old base-role heuristic is wrong or
#      undefined on every sample:
#      * gap=0 -> original "exactly two components" assumption fails outright
#      * gap=1 -> ctrl and tile tie on color-count and squareness, so the old sort
#        cannot correctly recover roles
# h) New solver assumptions and generator support:
#    - `_assign_by_position` assumes exactly one valid horizontal or vertical near
#      relation with gap in {0,1}; the generator only emits such layouts.
#    - `_extract_touching_pair` assumes foreground colors can be partitioned into two
#      disjoint palettes whose crops are a valid ctrl patch and a valid tile patch;
#      the generator uses disjoint 2-color palettes and rejects ambiguous samples.
#    - The stamping loop assumes ctrl occupancy is exactly `ctrl != bg`; generator and
#      solver both use that same binary-mask interpretation.


BBox = tuple[int, int, int, int]

_LAYOUT_CYCLE: list[tuple[str, int]] = [
    ("horizontal", 0),
    ("vertical", 0),
    ("horizontal", 1),
    ("vertical", 1),
]
_GEN_COUNTER = 0


def _axis_overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(int(a1), int(b1)) - max(int(a0), int(b0)) + 1)


def _horizontal_gap(left_bbox: BBox, right_bbox: BBox) -> int | None:
    if int(left_bbox[3]) >= int(right_bbox[1]):
        return None
    gap = int(right_bbox[1]) - int(left_bbox[3]) - 1
    if gap not in (0, 1):
        return None
    if _axis_overlap(left_bbox[0], left_bbox[2], right_bbox[0], right_bbox[2]) <= 0:
        return None
    return int(gap)


def _vertical_gap(top_bbox: BBox, bottom_bbox: BBox) -> int | None:
    if int(top_bbox[2]) >= int(bottom_bbox[0]):
        return None
    gap = int(bottom_bbox[0]) - int(top_bbox[2]) - 1
    if gap not in (0, 1):
        return None
    if _axis_overlap(top_bbox[1], top_bbox[3], bottom_bbox[1], bottom_bbox[3]) <= 0:
        return None
    return int(gap)


def _assign_by_position(
    first_bbox: BBox,
    first_patch: np.ndarray,
    second_bbox: BBox,
    second_patch: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str, int]:
    gap = _horizontal_gap(first_bbox, second_bbox)
    if gap is not None:
        return first_patch, second_patch, "horizontal", int(gap)

    gap = _horizontal_gap(second_bbox, first_bbox)
    if gap is not None:
        return second_patch, first_patch, "horizontal", int(gap)

    gap = _vertical_gap(first_bbox, second_bbox)
    if gap is not None:
        return first_patch, second_patch, "vertical", int(gap)

    gap = _vertical_gap(second_bbox, first_bbox)
    if gap is not None:
        return second_patch, first_patch, "vertical", int(gap)

    raise ValueError("objects do not follow the upper/left-to-lower/right rule")


def _ctrl_ok(patch: np.ndarray, bg: int) -> bool:
    h, w = map(int, patch.shape)
    if min(h, w) < 3:
        return False
    fg = patch != int(bg)
    if not fg.any():
        return False
    if not np.any(patch == int(bg)):
        return False
    cols = colors_present(patch, exclude={bg})
    if len(cols) not in (1, 2):
        return False
    area = h * w
    fill = int(fg.sum())
    if fill < max(4, int(np.ceil(0.35 * area))):
        return False
    if fill > int(np.floor(0.8 * area)):
        return False
    return True


def _tile_ok(patch: np.ndarray, bg: int) -> bool:
    if int(patch.shape[0]) != int(patch.shape[1]):
        return False
    cols = colors_present(patch, exclude={bg})
    return len(cols) >= 2


def _extract_palette_object(grid: np.ndarray, bg: int, palette: tuple[int, ...]) -> tuple[BBox, np.ndarray] | None:
    comps = find_components(
        grid,
        bg=bg,
        connectivity=8,
        monochromatic=False,
        colors=list(palette),
        return_format="dict",
    )
    if len(comps) != 1:
        return None

    bbox = tuple(int(x) for x in comps[0]["bbox"])
    patch = crop_bbox(grid, bbox)
    allowed = set(int(c) for c in palette)
    seen = set(colors_present(patch, exclude={bg}))
    if seen != allowed:
        return None
    if np.any((patch != int(bg)) & (~np.isin(patch, list(allowed)))):
        return None
    return bbox, patch


def _palette_partitions(cols: list[int]) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    items = tuple(sorted(int(c) for c in cols))
    out: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for k in range(1, len(items)):
        for subset in itertools.combinations(items, k):
            other = tuple(c for c in items if c not in subset)
            if items[0] not in subset:
                continue
            out.append((tuple(subset), other))
    return out


def _extract_touching_pair(grid: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray]:
    fg_cols = colors_present(grid, exclude={bg})
    candidates: list[tuple[tuple[int, int, int, int, int], np.ndarray, np.ndarray]] = []

    for palette_a, palette_b in _palette_partitions(fg_cols):
        obj_a = _extract_palette_object(grid, bg, palette_a)
        obj_b = _extract_palette_object(grid, bg, palette_b)
        if obj_a is None or obj_b is None:
            continue

        bbox_a, patch_a = obj_a
        bbox_b, patch_b = obj_b
        try:
            ctrl, tile, orientation, gap = _assign_by_position(bbox_a, patch_a, bbox_b, patch_b)
        except ValueError:
            continue
        if not _ctrl_ok(ctrl, bg):
            continue
        if not _tile_ok(tile, bg):
            continue

        ctrl_cols = len(colors_present(ctrl, exclude={bg}))
        tile_cols = len(colors_present(tile, exclude={bg}))
        score = (
            int(gap == 0),
            int(ctrl_cols == 2),
            int(tile_cols == 2),
            int(orientation == "horizontal"),
            int(tile.shape[0]),
        )
        candidates.append((score, ctrl, tile))

    if not candidates:
        raise ValueError("failed to split touching pair")

    candidates.sort(reverse=True, key=lambda item: item[0])
    _score, ctrl, tile = candidates[0]
    return ctrl, tile


def _extract_objects(grid: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray]:
    comps = find_components(grid, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    if len(comps) == 2:
        bbox_a = tuple(int(x) for x in comps[0]["bbox"])
        bbox_b = tuple(int(x) for x in comps[1]["bbox"])
        patch_a = crop_bbox(grid, bbox_a)
        patch_b = crop_bbox(grid, bbox_b)
        ctrl, tile, _orientation, _gap = _assign_by_position(bbox_a, patch_a, bbox_b, patch_b)
        if not _ctrl_ok(ctrl, bg):
            raise ValueError("position-selected ctrl patch is invalid")
        if not _tile_ok(tile, bg):
            raise ValueError("position-selected tile patch is invalid")
        return ctrl, tile

    if len(comps) == 1:
        return _extract_touching_pair(grid, bg)

    raise ValueError("unexpected foreground component count")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Solve the fused D2/3 + D8/2 variant of bf45cf4b.

    The extraction rule is changed:
    - identify the two near-adjacent objects
    - assign roles by position: upper/left -> ctrl, lower/right -> tile
    - treat every non-background ctrl cell as one filled mask pixel
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    ctrl, tile = _extract_objects(g, bg)

    th, tw = map(int, tile.shape)
    ch, cw = map(int, ctrl.shape)
    mask = ctrl != int(bg)

    out = np.full((ch * th, cw * tw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            if bool(mask[r, c]):
                paste(out, tile, r * th, c * tw)
    return out


def _dihedral_orbits(size: int) -> list[tuple[list[tuple[int, int]], bool]]:
    seen: set[tuple[int, int]] = set()
    out: list[tuple[list[tuple[int, int]], bool]] = []
    s = int(size)
    for r in range(s):
        for c in range(s):
            if (r, c) in seen:
                continue
            orbit = {
                (r, c),
                (c, s - 1 - r),
                (s - 1 - r, s - 1 - c),
                (s - 1 - c, r),
                (r, s - 1 - c),
                (s - 1 - r, c),
                (c, r),
                (s - 1 - c, s - 1 - r),
            }
            cells = sorted(orbit)
            seen.update(cells)
            border = any(rr in (0, s - 1) or cc in (0, s - 1) for rr, cc in cells)
            out.append((cells, border))
    return out


def _make_tile(bg: int, tile_colors: list[int], rng: np.random.Generator) -> np.ndarray:
    size = int(rng.integers(3, 6))
    palette = [int(x) for x in rng.choice(tile_colors, size=2, replace=False)]
    orbits = _dihedral_orbits(size)
    rng.shuffle(orbits)

    tile = np.full((size, size), int(bg), dtype=int)
    assignments: dict[int, int] = {
        0: int(palette[0]),
        1: int(palette[1]),
    }
    interior_idxs = [i for i, (_cells, border) in enumerate(orbits) if not border and i not in assignments]
    if interior_idxs and rng.random() < 0.25:
        assignments[int(rng.choice(interior_idxs))] = int(bg)

    for i, (_cells, border) in enumerate(orbits):
        if i in assignments:
            continue
        choices = palette if border else palette + ([bg] if interior_idxs else [])
        assignments[i] = int(rng.choice(choices))

    for i, (cells, _border) in enumerate(orbits):
        for r, c in cells:
            tile[r, c] = assignments[i]

    if len(colors_present(tile, exclude={bg})) != 2:
        tile[0, 0] = int(palette[0])
        tile[0, 1] = int(palette[1])
    return tile


def _neighbors8(h: int, w: int, r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w:
                out.append((rr, cc))
    return out


def _make_ctrl(bg: int, ctrl_colors: list[int], rng: np.random.Generator) -> np.ndarray:
    size = int(rng.integers(3, 6))
    area = size * size
    min_fill = max(5, int(np.ceil(0.42 * area)))
    max_fill = max(min_fill, int(np.floor(0.72 * area)))
    target = int(rng.integers(min_fill, max_fill + 1))

    anchor_r = int(rng.integers(0, size))
    anchor_c = int(rng.integers(0, size))
    mask = np.zeros((size, size), dtype=bool)
    mask[:, anchor_c] = True
    mask[anchor_r, :] = True

    while int(mask.sum()) < target:
        frontier = []
        for r in range(size):
            for c in range(size):
                if mask[r, c]:
                    continue
                if any(mask[rr, cc] for rr, cc in _neighbors8(size, size, r, c)):
                    frontier.append((r, c))
        if not frontier:
            break
        pick = frontier[int(rng.integers(len(frontier)))]
        mask[pick] = True

    ctrl = np.full((size, size), int(bg), dtype=int)
    fill_cells = [tuple(x) for x in np.argwhere(mask)]
    rng.shuffle(fill_cells)
    for i, (r, c) in enumerate(fill_cells):
        ctrl[r, c] = int(ctrl_colors[i % 2])

    if len(colors_present(ctrl, exclude={bg})) != 2:
        r0, c0 = fill_cells[-1]
        ctrl[r0, c0] = int(ctrl_colors[1])
    return ctrl


def _compose_output(tile: np.ndarray, ctrl: np.ndarray, bg: int) -> np.ndarray:
    th, tw = tile.shape
    ch, cw = ctrl.shape
    out = np.full((ch * th, cw * tw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            if int(ctrl[r, c]) != int(bg):
                paste(out, tile, r * th, c * tw)
    return out


def _make_near_pair(
    ctrl: np.ndarray,
    tile: np.ndarray,
    bg: int,
    orientation: str,
    gap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    ch, cw = ctrl.shape
    th, tw = tile.shape

    ctrl_r0 = 0
    ctrl_c0 = 0
    if orientation == "horizontal":
        ctrl_rows = np.where(ctrl[:, cw - 1] != int(bg))[0]
        tile_rows = np.where(tile[:, 0] != int(bg))[0]
        hit_ctrl = int(rng.choice(ctrl_rows))
        hit_tile = int(rng.choice(tile_rows))
        tile_r0 = hit_ctrl - hit_tile
        tile_c0 = cw + int(gap)
    else:
        ctrl_cols = np.where(ctrl[ch - 1, :] != int(bg))[0]
        tile_cols = np.where(tile[0, :] != int(bg))[0]
        hit_ctrl = int(rng.choice(ctrl_cols))
        hit_tile = int(rng.choice(tile_cols))
        tile_r0 = ch + int(gap)
        tile_c0 = hit_ctrl - hit_tile

    min_r = min(ctrl_r0, tile_r0)
    min_c = min(ctrl_c0, tile_c0)
    ctrl_r0 -= min_r
    ctrl_c0 -= min_c
    tile_r0 -= min_r
    tile_c0 -= min_c

    pair_h = max(ctrl_r0 + ch, tile_r0 + th)
    pair_w = max(ctrl_c0 + cw, tile_c0 + tw)
    pair = np.full((pair_h, pair_w), int(bg), dtype=int)
    paste(pair, tile, tile_r0, tile_c0)
    paste(pair, ctrl, ctrl_r0, ctrl_c0, transparent=bg)
    return pair


def _embed_pair(pair: np.ndarray, bg: int, rng: np.random.Generator) -> np.ndarray:
    ph, pw = pair.shape
    H = int(rng.integers(max(12, ph), 19))
    W = int(rng.integers(max(12, pw), 19))
    r0 = int(rng.integers(0, H - ph + 1))
    c0 = int(rng.integers(0, W - pw + 1))
    grid = np.full((H, W), int(bg), dtype=int)
    paste(grid, pair, r0, c0)
    return grid


def _next_layout() -> tuple[str, int]:
    global _GEN_COUNTER
    layout = _LAYOUT_CYCLE[_GEN_COUNTER % len(_LAYOUT_CYCLE)]
    _GEN_COUNTER += 1
    return layout


def _original_role_heuristic_picks_tile(ctrl: np.ndarray, tile: np.ndarray, bg: int) -> bool:
    infos = [
        {"label": "ctrl", "patch": ctrl, "cols": colors_present(ctrl, exclude={bg})},
        {"label": "tile", "patch": tile, "cols": colors_present(tile, exclude={bg})},
    ]
    infos.sort(
        key=lambda d: (
            len(d["cols"]) >= 2,
            len(d["cols"]),
            int(d["patch"].shape[0]) == int(d["patch"].shape[1]),
        ),
        reverse=True,
    )
    return infos[0]["label"] == "tile"


def _role_flip_changes_output(ctrl: np.ndarray, tile: np.ndarray, bg: int) -> bool:
    return not np.array_equal(_compose_output(tile, ctrl, bg), _compose_output(ctrl, tile, bg))


def _layout_signature(grid: np.ndarray, bg: int) -> tuple[str, int]:
    comps = find_components(grid, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    if len(comps) == 2:
        bbox_a = tuple(int(x) for x in comps[0]["bbox"])
        bbox_b = tuple(int(x) for x in comps[1]["bbox"])
        _ctrl, _tile, orientation, gap = _assign_by_position(
            bbox_a,
            crop_bbox(grid, bbox_a),
            bbox_b,
            crop_bbox(grid, bbox_b),
        )
        return orientation, int(gap)

    if len(comps) == 1:
        ctrl, tile = _extract_touching_pair(grid, bg)
        fg_cols = colors_present(grid, exclude={bg})
        for palette_a, palette_b in _palette_partitions(fg_cols):
            obj_a = _extract_palette_object(grid, bg, palette_a)
            obj_b = _extract_palette_object(grid, bg, palette_b)
            if obj_a is None or obj_b is None:
                continue
            bbox_a, patch_a = obj_a
            bbox_b, patch_b = obj_b
            try:
                cand_ctrl, cand_tile, orientation, gap = _assign_by_position(bbox_a, patch_a, bbox_b, patch_b)
            except ValueError:
                continue
            if np.array_equal(cand_ctrl, ctrl) and np.array_equal(cand_tile, tile):
                return orientation, int(gap)
    raise ValueError("cannot determine layout signature")


def _gen_attempt(orientation: str, gap: int) -> np.ndarray | None:
    rng = np.random.default_rng()
    palette = list(rng.choice(np.arange(10, dtype=int), size=5, replace=False))
    bg = int(palette[0])
    ctrl_colors = [int(palette[1]), int(palette[2])]
    tile_colors = [int(palette[3]), int(palette[4])]

    ctrl = _make_ctrl(bg, ctrl_colors, rng)
    tile = _make_tile(bg, tile_colors, rng)
    expected = _compose_output(tile, ctrl, bg)
    pair = _make_near_pair(ctrl, tile, bg, orientation, gap, rng)
    inp = _embed_pair(pair, bg, rng)

    try:
        solved = main(inp)
        rec_ctrl, rec_tile = _extract_objects(inp, bg)
        found_orientation, found_gap = _layout_signature(inp, bg)
    except Exception:
        return None

    if not np.array_equal(solved, expected):
        return None
    if not np.array_equal(rec_ctrl, ctrl):
        return None
    if not np.array_equal(rec_tile, tile):
        return None
    if found_orientation != orientation or int(found_gap) != int(gap):
        return None
    if not _ctrl_ok(ctrl, bg):
        return None
    if not _tile_ok(tile, bg):
        return None
    if _original_role_heuristic_picks_tile(ctrl, tile, bg):
        return None
    if not _role_flip_changes_output(ctrl, tile, bg):
        return None

    comps = find_components(inp, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    if int(gap) == 0 and len(comps) != 1:
        return None
    if int(gap) == 1 and len(comps) != 2:
        return None
    return inp


def generate_input() -> np.ndarray:
    orientation, gap = _next_layout()
    for _ in range(400):
        inp = _gen_attempt(orientation, gap)
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid input")
