# Combo variant of 5dbc8537 (index 20)
# combo_spec: D1/1+D4/1+D15/2
# Name: 方向标记解码+旋转匹配+旋转填入

from __future__ import annotations

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

# Design analysis for the combo variant
# -------------------------------------
# a) Concreteization and behavior trace:
#    - The strip template is kept, but its holes are now separate irregular polyomino cavities.
#    - Each outside sprite is a sparse component: an irregular core plus one diagonally attached
#      corner marker pixel. The marker is not part of the filled content; it only encodes a
#      quarter-turn amount.
#    - Small trace: suppose a destination hole is a 4x3 hook. In the output, that hook is filled
#      by a bichromatic patch with an accent along the hook's top/left edges. In the input, the
#      same core may appear rotated away from the hole, and a marker on the source's bottom-left
#      outer corner says "rotate clockwise 270 degrees before using me". The solver must first
#      strip the marker, then decode the corner-to-rotation legend, then rotate the core, then
#      compare the resulting irregular mask against the template holes.
#    - Compared with the original task, three old simplifications disappear at once: source bbox
#      no longer equals hole bbox because of the extra marker row/col; holes are not rectangles;
#      and pasting the raw source appearance is wrong because the source must be rotated first.
# b) Complexity calibration:
#    - Only four marker states exist (the four outer corners), and they form a cyclic legend.
#      This keeps the instruction space small enough to infer from examples.
#    - Every generated instance uses at least four pieces and forces all four marker corners to
#      appear at least once, so the symbolic layer has repeated support instead of one-off cases.
#    - Shape ambiguity is controlled by making every hole unique up to the full dihedral group.
#      That means the decoded rotation has to be correct; a wrong global legend offset leaves no
#      alternative hole with the same rotated shape.
# c) Hidden-assumption audit against the original solver:
#    - `hole_mask[r:r+h, c:c+w].all()` assumed every cavity was a solid rectangle. The combo task
#      breaks that immediately because cavities are sparse masks in their own bboxes.
#    - Treating the whole source component bbox as the fill payload is now invalid because the
#      bbox includes background padding and a symbolic marker cell.
#    - Pasting the source patch without reorientation is wrong because the marker-defined rotation
#      must be applied before both matching and filling.
#    - Sorting or matching by bbox alone is no longer safe: after stripping the marker, the core
#      may need a quarter-turn, which can swap bbox dimensions before the shape matches any hole.
# d) Placement legality:
#    - The generator places hole bboxes with a 1-cell base border and a 1-cell gap along the strip
#      direction, so separate cavities never overlap, never touch the strip boundary, and never
#      merge into a larger background component.
#    - Every core mask occupies all four bbox corners. This guarantees that a diagonally attached
#      outer-corner marker stays 8-connected to the core for any chosen rotation.
#    - Scattered source sprites are placed with `random_free_location(..., padding=1)`, preserving
#      separation from one another and from the template.
# e) Visual observability:
#    - The same four corner markers recur within each sample, and the core motifs are directional,
#      so humans can compare input sprites against filled output holes and infer the corner legend.
#    - Non-zero rotations are common rather than rare, and the irregular hole shapes make the
#      before/after orientation change visible even if a color pattern is simple.
# f) Visual cleanliness:
#    - The marker sits outside the core instead of on top of it, so the symbolic instruction does
#      not occlude the actual colored content.
#    - Holes are laid out as a spaced sequence inside the strip, which keeps each cavity legible.
# g) Color inference:
#    - No hidden color ordering is used. The rule depends only on visible geometry: marker corner,
#      source orientation, and destination hole shape. Colors merely travel with the rotated core.

GRID_MIN, GRID_MAX = 15, 25
THICK_MIN, THICK_MAX = 8, 10
OUTER_GAP = 10
SPRITE_COUNT_MIN, SPRITE_COUNT_MAX = 4, 6
PIECE_DIM_MIN, PIECE_DIM_MAX = 3, 4
PIECE_AREA_MIN = 6
HOLE_FRAC_LO, HOLE_FRAC_HI = 0.16, 0.34
MIN_PALETTE_SIZE = 5
MAX_RETRIES = 500
PIECE_RETRIES = 200

CORNERS = ("tl", "tr", "br", "bl")


def _rot_cw(arr: np.ndarray, steps: int) -> np.ndarray:
    return np.rot90(arr, k=(-int(steps)) % 4)


def _corner_index(corner: str) -> int:
    return CORNERS.index(str(corner))


def _corner_to_cw(corner: str, offset: int = 0) -> int:
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


def _extract_symbol_sprite(patch: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    ph, pw = map(int, patch.shape)
    assert ph >= 4 and pw >= 4
    for corner in CORNERS:
        (mr, mc), (r0, c0, r1, c1) = _corner_bbox_spec(ph, pw, corner)
        if not bool(mask[mr, mc]):
            continue
        core_mask = mask[r0 : r1 + 1, c0 : c1 + 1].copy()
        if not core_mask.any():
            continue
        expected = np.zeros((ph, pw), dtype=bool)
        expected[r0 : r1 + 1, c0 : c1 + 1] = core_mask
        expected[mr, mc] = True
        if np.array_equal(mask, expected):
            core_patch = patch[r0 : r1 + 1, c0 : c1 + 1].copy()
            return core_patch, core_mask, corner
    raise AssertionError("source component is not a valid corner-marked irregular sprite")


def _rotation_signature(mask: np.ndarray) -> bytes:
    return min(np.rot90(mask, k).tobytes() for k in range(4))


def _dihedral_signature(mask: np.ndarray) -> bytes:
    return min(_transform(mask, k, flip).tobytes() for k in range(4) for flip in (False, True))


def _transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(arr, int(k))
    if flip:
        out = np.fliplr(out)
    return out


def _detect_base_color(g: np.ndarray, bg: int) -> int:
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
    return int(base_candidates[0][1])


def _parse_grid(grid: np.ndarray) -> tuple[int, int, list[dict], list[dict], tuple[int, int]]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    base_col = _detect_base_color(g, bg)

    base_coords = np.argwhere(g == int(base_col))
    assert base_coords.size
    r0, c0, r1, c1 = bbox_coords(base_coords)
    dst = crop_bbox(g, (int(r0), int(c0), int(r1), int(c1)))
    dh, dw = map(int, dst.shape)
    hole_mask = dst == int(bg)

    hole_comps = components_mask(hole_mask, connectivity=4)
    holes: list[dict] = []
    for coords in hole_comps:
        hr0, hc0, hr1, hc1 = bbox_coords(coords)
        h, w = int(hr1 - hr0 + 1), int(hc1 - hc0 + 1)
        m = np.zeros((h, w), dtype=bool)
        m[coords[:, 0] - hr0, coords[:, 1] - hc0] = True
        holes.append(
            {
                "bbox": (int(hr0), int(hc0), int(hr1), int(hc1)),
                "mask": m,
                "area": int(m.sum()),
            }
        )
    holes.sort(key=lambda d: (-int(d["area"]), int(d["bbox"][0]), int(d["bbox"][1])))

    masked = g.copy()
    masked[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(bg)
    sprite_comps = find_components(masked, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    sprites: list[dict] = []
    for comp in sprite_comps:
        pr0, pc0, pr1, pc1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (pr0, pc0, pr1, pc1))
        mask = np.zeros_like(patch, dtype=bool)
        coords = comp["coords"]
        mask[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        core_patch, core_mask, corner = _extract_symbol_sprite(patch, mask)
        sprites.append(
            {
                "core_patch": core_patch,
                "core_mask": core_mask,
                "corner": corner,
                "area": int(core_mask.sum()),
            }
        )

    assert holes and sprites
    return bg, base_col, holes, sprites, (dh, dw)


def _solve_with_offset(
    holes: list[dict], sprites: list[dict], base_col: int, shape: tuple[int, int], offset: int
) -> np.ndarray | None:
    dh, dw = map(int, shape)
    transformed: list[dict] = []
    total_area = 0
    for sp in sprites:
        cw = _corner_to_cw(str(sp["corner"]), offset)
        patch = _rot_cw(np.array(sp["core_patch"], dtype=int), cw)
        mask = _rot_cw(np.array(sp["core_mask"], dtype=bool), cw)
        total_area += int(mask.sum())
        opts: list[int] = []
        for hi, hole in enumerate(holes):
            if int(hole["area"]) != int(mask.sum()):
                continue
            if mask.shape == hole["mask"].shape and np.array_equal(mask, hole["mask"]):
                opts.append(int(hi))
        if not opts:
            return None
        transformed.append({"patch": patch, "mask": mask, "area": int(mask.sum()), "opts": opts})

    if total_area != sum(int(h["area"]) for h in holes):
        return None

    order = sorted(range(len(transformed)), key=lambda i: (len(transformed[i]["opts"]), -int(transformed[i]["area"])))
    used_holes = [False] * len(holes)
    placement: list[int | None] = [None] * len(transformed)

    def dfs(t: int) -> bool:
        if t == len(order):
            return True
        si = int(order[t])
        for hi in transformed[si]["opts"]:
            if used_holes[int(hi)]:
                continue
            used_holes[int(hi)] = True
            placement[si] = int(hi)
            if dfs(t + 1):
                return True
            placement[si] = None
            used_holes[int(hi)] = False
        return False

    if not dfs(0):
        return None

    out = np.full((dh, dw), int(base_col), dtype=int)
    for si, hi in enumerate(placement):
        assert hi is not None
        hr0, hc0, _hr1, _hc1 = holes[int(hi)]["bbox"]
        paste(out, transformed[si]["patch"], int(hr0), int(hc0), mask=transformed[si]["mask"], overwrite=True)
    return out


def _all_offset_solutions(grid: np.ndarray) -> list[tuple[int, np.ndarray]]:
    bg, base_col, holes, sprites, shape = _parse_grid(grid)
    _ = bg
    sols: list[tuple[int, np.ndarray]] = []
    for offset in range(4):
        out = _solve_with_offset(holes, sprites, base_col, shape, offset)
        if out is not None:
            sols.append((int(offset), out))
    return sols


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild a strip template by decoding corner markers, rotating irregular sprites, and fitting them into matching cavities.

    Input structure:
    - The dominant background surrounds one large base-colored strip whose interior contains several separate irregular holes.
    - Outside the strip, each source object is an irregular colored core plus one diagonally attached marker pixel placed on an outer corner.

    Rule:
    - Use the largest non-background connected component color to find the strip and crop its bbox.
    - Interpret each marker corner as a quarter-turn instruction, remove the marker, rotate the irregular core accordingly, and then match that rotated shape to exactly one cavity in the strip.
    - Paste the rotated core into its matching hole and return the tight strip bbox with all cavities filled.

    Key insight:
    - The puzzle only resolves if the same global corner-to-rotation legend is applied to every sprite: after that decoding step, the rotated core masks line up one-to-one with the irregular hole masks.
    """
    solutions = _all_offset_solutions(np.array(grid, dtype=int))
    assert solutions
    first = solutions[0][1]
    for _offset, out in solutions[1:]:
        assert np.array_equal(first, out)
    return first


def _is_connected(mask: np.ndarray) -> bool:
    return len(components_mask(mask, connectivity=4)) == 1


def _random_connected_mask() -> np.ndarray | None:
    for _ in range(PIECE_RETRIES):
        h = random.randint(PIECE_DIM_MIN, PIECE_DIM_MAX)
        w = random.randint(PIECE_DIM_MIN, PIECE_DIM_MAX)
        mask = np.ones((h, w), dtype=bool)
        full = int(mask.sum())
        removable = [(r, c) for r in range(h) for c in range(w) if (r, c) not in {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}]
        random.shuffle(removable)
        max_remove = min(len(removable), full - PIECE_AREA_MIN)
        if max_remove <= 0:
            continue
        goal = random.randint(1, max_remove)
        removed = 0
        for r, c in removable:
            if removed >= goal:
                break
            if random.random() < 0.35:
                continue
            trial = mask.copy()
            trial[int(r), int(c)] = False
            if int(trial.sum()) < PIECE_AREA_MIN or not _is_connected(trial):
                continue
            mask = trial
            removed += 1
        if removed == 0 or bool(mask.all()):
            continue
        if not (bool(mask[0, 0]) and bool(mask[0, -1]) and bool(mask[-1, 0]) and bool(mask[-1, -1])):
            continue
        if len({np.rot90(mask, k).tobytes() for k in range(4)}) != 4:
            continue
        if len({_transform(mask, k, flip).tobytes() for k in range(4) for flip in (False, True)}) != 8:
            continue
        return mask
    return None


def _choose_color_pairs(bg: int, base: int, marker: int, count: int) -> list[tuple[int, int]] | None:
    available = [c for c in range(10) if c not in (bg, base, marker)]
    if len(available) < MIN_PALETTE_SIZE:
        return None
    palette = random.sample(available, random.randint(MIN_PALETTE_SIZE, len(available)))
    pairs: list[tuple[int, int]] = []
    for _ in range(count):
        primary = int(random.choice(palette))
        secondary = int(random.choice([c for c in palette if c != primary]))
        pairs.append((primary, secondary))
    return pairs


def _make_core_patch(mask: np.ndarray, cols: tuple[int, int], bg: int) -> np.ndarray:
    primary, secondary = map(int, cols)
    out = np.full(mask.shape, int(bg), dtype=int)
    out[mask] = primary
    coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    accent = {(r, c) for r, c in coords if r == 0 or c == 0}
    if not accent:
        accent.add(coords[0])
    if len(accent) == len(coords):
        accent = {coords[0]}
    anchor = min(coords, key=lambda rc: (rc[0] + rc[1], rc[0], rc[1]))
    accent.add(anchor)
    for r, c in accent:
        out[int(r), int(c)] = secondary
    return out


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


def _place_destinations(dh: int, dw: int, pieces: list[dict], *, vertical: bool) -> list[tuple[int, int]] | None:
    long = int(dh if vertical else dw)
    sizes = [int(p["dest_mask"].shape[0] if vertical else p["dest_mask"].shape[1]) for p in pieces]
    min_need = 2 + sum(sizes) + max(0, len(pieces) - 1)
    if min_need > long:
        return None
    gaps = [1] * (len(pieces) + 1)
    for _ in range(long - min_need):
        gaps[random.randrange(len(gaps))] += 1

    pos: list[tuple[int, int]] = []
    cursor = gaps[0]
    for i, piece in enumerate(pieces):
        h, w = map(int, piece["dest_mask"].shape)
        if vertical:
            if dw - w - 1 < 1:
                return None
            rr = int(cursor)
            cc = random.randint(1, dw - w - 1)
            pos.append((rr, cc))
            cursor += h + gaps[i + 1]
        else:
            if dh - h - 1 < 1:
                return None
            rr = random.randint(1, dh - h - 1)
            cc = int(cursor)
            pos.append((rr, cc))
            cursor += w + gaps[i + 1]
    return pos


def _build_instance() -> tuple[np.ndarray, np.ndarray] | None:
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

    long = int(dh if vertical else dw)
    count_hi = min(SPRITE_COUNT_MAX, max(SPRITE_COUNT_MIN, (long - 2) // 3))
    if count_hi < SPRITE_COUNT_MIN:
        return None
    count = random.randint(SPRITE_COUNT_MIN, count_hi)

    bg, base, marker = random.sample(range(10), 3)
    color_pairs = _choose_color_pairs(bg, base, marker, count)
    if color_pairs is None:
        return None

    masks: list[np.ndarray] = []
    seen_shapes: set[bytes] = set()
    for _ in range(count):
        mask = None
        for _ in range(60):
            cand = _random_connected_mask()
            if cand is None:
                continue
            sig = _dihedral_signature(cand)
            if sig in seen_shapes:
                continue
            mask = cand
            seen_shapes.add(sig)
            break
        if mask is None:
            return None
        masks.append(mask)

    cw_steps = [0, 1, 2, 3]
    while len(cw_steps) < count:
        cw_steps.append(random.randrange(4))
    random.shuffle(cw_steps)

    pieces: list[dict] = []
    for mask, cols, cw in zip(masks, color_pairs, cw_steps):
        dest_rot = random.randrange(4)
        core_patch = _make_core_patch(mask, cols, bg)
        dest_mask = _rot_cw(mask, dest_rot)
        dest_patch = _rot_cw(core_patch, dest_rot)
        source_core = np.rot90(dest_patch, k=int(cw))
        source_mask = np.rot90(dest_mask, k=int(cw))
        corner = CORNERS[int(cw)]
        pieces.append(
            {
                "dest_mask": dest_mask,
                "dest_patch": dest_patch,
                "source_mask": source_mask,
                "source_patch": _build_symbol_sprite(source_core, marker, corner, bg),
                "cw": int(cw),
                "area": int(dest_mask.sum()),
                "shape_sig": _rotation_signature(dest_mask),
            }
        )

    random.shuffle(pieces)
    placements = _place_destinations(dh, dw, pieces, vertical=vertical)
    if placements is None:
        return None

    total_hole_area = sum(int(p["area"]) for p in pieces)
    frac = total_hole_area / float(dh * dw)
    if not (HOLE_FRAC_LO <= frac <= HOLE_FRAC_HI):
        return None
    if len({int(p["cw"]) for p in pieces}) < 4:
        return None

    out = np.full((dh, dw), int(base), dtype=int)
    for piece, (rr, cc) in zip(pieces, placements):
        paste(out, piece["dest_patch"], int(rr), int(cc), mask=piece["dest_mask"], overwrite=True)

    inp = np.full((H, W), int(bg), dtype=int)
    inp[int(tr0) : int(tr0 + dh), int(tc0) : int(tc0 + dw)] = int(base)
    for piece, (rr, cc) in zip(pieces, placements):
        sub = inp[int(tr0 + rr) : int(tr0 + rr + piece["dest_mask"].shape[0]), int(tc0 + cc) : int(tc0 + cc + piece["dest_mask"].shape[1])]
        sub[np.array(piece["dest_mask"], dtype=bool)] = int(bg)

    placement = np.full((H, W), int(bg), dtype=int)
    placement[int(tr0) : int(tr0 + dh), int(tc0) : int(tc0 + dw)] = int(base)
    order = list(range(len(pieces)))
    random.shuffle(order)
    order.sort(
        key=lambda i: (
            -int(pieces[i]["area"]),
            -int(pieces[i]["source_patch"].shape[0] * pieces[i]["source_patch"].shape[1]),
        )
    )
    try:
        for i in order:
            patch = np.array(pieces[i]["source_patch"], dtype=int)
            rr, cc = random_free_location(placement, patch, transparent=bg, padding=1, padding_connectivity=8)
            paste(inp, patch, int(rr), int(cc), transparent=bg, overwrite=True)
            paste(placement, patch, int(rr), int(cc), transparent=bg, overwrite=True)
    except ValueError:
        return None

    solutions = _all_offset_solutions(inp)
    if len(solutions) != 1:
        return None
    solved = solutions[0][1]
    if not np.array_equal(solved, out):
        return None
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    built = _build_instance()
    if built is None:
        return None
    inp, out = built
    try:
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
    raise RuntimeError("failed to generate valid 5dbc8537 combo input")
