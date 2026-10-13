from __future__ import annotations

from collections import Counter

import numpy as np


_RNG = np.random.default_rng()
_NBRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_NBRS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
_CCW_STEPS = {1: 0, 2: 3, 3: 2, 4: 1}


def _get_bg(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(cnts))])


def _components(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    nbrs = _NBRS4 if int(connectivity) == 4 else _NBRS8
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in nbrs:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [int(r) for r, _ in coords]
    cs = [int(c) for _, c in coords]
    return int(min(rs)), int(min(cs)), int(max(rs)), int(max(cs))


def _mask_from_coords(coords: list[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    for r, c in coords:
        out[int(r), int(c)] = True
    return out


def _crop_mask(coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    for r, c in coords:
        mask[int(r - r0), int(c - c0)] = True
    return mask, (int(r0), int(c0), int(r1), int(c1))


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _can_place(occupied: np.ndarray, patch_mask: np.ndarray, r0: int, c0: int) -> bool:
    h, w = patch_mask.shape
    hh, ww = occupied.shape
    if r0 < 0 or c0 < 0 or r0 + h > hh or c0 + w > ww:
        return False
    blocked = _dilate8(occupied)
    return not bool((blocked[r0 : r0 + h, c0 : c0 + w] & patch_mask).any())


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    h, w = patch.shape
    sub = dst[int(r0) : int(r0) + h, int(c0) : int(c0) + w]
    mask = patch != int(bg)
    sub[mask] = patch[mask]


def _find_block_candidates(grid: np.ndarray, bg: int) -> list[dict]:
    candidates: list[dict] = []

    for color in map(int, np.unique(grid)):
        if int(color) == int(bg):
            continue
        for comp in _components(grid == int(color), connectivity=4):
            r0, c0, r1, c1 = _bbox(comp)
            sub = grid[r0 : r1 + 1, c0 : c1 + 1]
            if not (
                bool(np.all(sub[0, :] == int(color)))
                and bool(np.all(sub[-1, :] == int(color)))
                and bool(np.all(sub[:, 0] == int(color)))
                and bool(np.all(sub[:, -1] == int(color)))
            ):
                continue
            if not set(map(int, np.unique(sub))).issubset({int(bg), int(color)}):
                continue

            local_holes: list[dict] = []
            for bg_comp in _components(sub == int(bg), connectivity=4):
                if any(
                    rr == 0 or cc == 0 or rr == sub.shape[0] - 1 or cc == sub.shape[1] - 1
                    for rr, cc in bg_comp
                ):
                    continue
                hole_mask, (hr0, hc0, hr1, hc1) = _crop_mask(bg_comp)
                local_holes.append(
                    {
                        "bbox": (int(r0 + hr0), int(c0 + hc0), int(r0 + hr1), int(c0 + hc1)),
                        "mask": hole_mask,
                        "block_color": int(color),
                    }
                )

            if not local_holes:
                continue

            candidates.append(
                {
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "color": int(color),
                    "coords": [(int(rr), int(cc)) for rr, cc in comp],
                    "holes": local_holes,
                    "area": int(len(comp)),
                }
            )

    return candidates


def _extract_sprites(grid: np.ndarray, bg: int, block_pixels: np.ndarray) -> list[dict]:
    sprite_mask = (grid != int(bg)) & (~block_pixels)
    sprites: list[dict] = []
    for comp in _components(sprite_mask, connectivity=8):
        r0, c0, r1, c1 = _bbox(comp)
        patch = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(bg), dtype=int)
        colors: set[int] = set()
        for r, c in comp:
            val = int(grid[int(r), int(c)])
            patch[int(r - r0), int(c - c0)] = val
            colors.add(val)
        n_colors = len(colors)
        assert 1 <= n_colors <= 4, "each sprite must use 1 to 4 colors"
        rotated = np.rot90(patch, int(_CCW_STEPS[int(n_colors)]))
        sprites.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "patch": patch,
                "rotated": rotated,
                "n_colors": int(n_colors),
            }
        )
    return sprites


def _match_sprites_to_holes(sprites: list[dict], holes: list[dict], bg: int) -> dict[int, int]:
    assert len(sprites) == len(holes), "sprite count must equal hole count"

    candidates: dict[int, list[int]] = {}
    for si, sprite in enumerate(sprites):
        fg_mask = sprite["rotated"] != int(bg)
        exact: list[int] = []
        loose: list[int] = []
        for hi, hole in enumerate(holes):
            if sprite["rotated"].shape != hole["mask"].shape:
                continue
            if np.array_equal(fg_mask, hole["mask"]):
                exact.append(int(hi))
            elif int(fg_mask.sum()) == int(hole["mask"].sum()):
                loose.append(int(hi))
        candidates[int(si)] = exact if exact else loose
        assert candidates[int(si)], "every sprite must match at least one hole"

    order = sorted(
        range(len(sprites)),
        key=lambda idx: (len(candidates[int(idx)]), sprites[int(idx)]["rotated"].size),
    )
    assignment: dict[int, int] = {}
    used: set[int] = set()

    def dfs(pos: int) -> bool:
        if pos == len(order):
            return True
        si = int(order[pos])
        for hi in candidates[int(si)]:
            if int(hi) in used:
                continue
            assignment[int(si)] = int(hi)
            used.add(int(hi))
            if dfs(int(pos + 1)):
                return True
            used.remove(int(hi))
            del assignment[int(si)]
        return False

    assert dfs(0), "sprite-hole matching must be unambiguous"
    return assignment


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each carved rectangle-hole with its matching scattered sprite, rotating by the sprite's color count.

    Input structure:
    - bg is the most common color.
    - Several large monochrome rectangular blocks appear on the canvas; each block has one or more irregular bg holes inside it.
    - Outside the blocks, there are scattered small sprite components whose tight bounding boxes match the holes' tight bounding boxes.
    - Each sprite uses between 1 and 4 distinct non-bg colors.

    Rule:
    - Detect the monochrome rectangular blocks and extract every internal hole as a connected bg region with its own tight bbox.
    - Detect the scattered sprite components outside the blocks and count each sprite's distinct non-bg colors N.
    - Rotate each sprite by a color-count key: N=1 -> 0 degrees, N=2 -> 90 degrees clockwise, N=3 -> 180 degrees, N=4 -> 270 degrees clockwise.
    - Match sprites to holes by bbox size, then place the rotated sprite into the matched hole's bbox; hole cells not covered by sprite colors revert to the surrounding block color, and the original scattered sprites are erased to bg.

    Key insight:
    Bounding-box size chooses which hole a sprite belongs to, while the number of colors inside that sprite independently chooses the rotation needed before insertion.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)

    candidates = _find_block_candidates(g, bg)
    assert candidates, "expected at least one holed rectangular block"

    best: tuple[tuple[int, int, int], np.ndarray, list[dict], list[dict], dict[int, int]] | None = None
    hh, ww = g.shape
    for mask_bits in range(1, 1 << len(candidates)):
        block_pixels = np.zeros((hh, ww), dtype=bool)
        holes: list[dict] = []
        area_sum = 0
        block_count = 0
        for idx, cand in enumerate(candidates):
            if not ((int(mask_bits) >> int(idx)) & 1):
                continue
            block_count += 1
            area_sum += int(cand["area"])
            for rr, cc in cand["coords"]:
                block_pixels[int(rr), int(cc)] = True
            holes.extend(cand["holes"])
        if not holes:
            continue
        sprites = _extract_sprites(g, bg, block_pixels)
        if len(sprites) != len(holes) or not sprites:
            continue
        try:
            assignment = _match_sprites_to_holes(sprites, holes, bg)
        except AssertionError:
            continue
        score = (int(area_sum), int(len(holes)), -int(block_count))
        if best is None or score > best[0]:
            best = (score, block_pixels, holes, sprites, assignment)

    assert best is not None, "could not resolve block subset and sprite-hole matches"
    _, block_pixels, holes, sprites, assignment = best

    out = g.copy()
    out[(g != int(bg)) & (~block_pixels)] = int(bg)

    for hole in holes:
        r0, c0, r1, c1 = map(int, hole["bbox"])
        sub = out[r0 : r1 + 1, c0 : c1 + 1]
        sub[hole["mask"]] = int(hole["block_color"])

    for si, hi in assignment.items():
        hole = holes[int(hi)]
        rotated = sprites[int(si)]["rotated"]
        paint_mask = rotated != int(bg)
        r0, c0, r1, c1 = map(int, hole["bbox"])
        sub = out[r0 : r1 + 1, c0 : c1 + 1]
        sub[paint_mask] = rotated[paint_mask]

    return out


def _random_connected_mask(h: int, w: int) -> np.ndarray:
    min_area = max(int(np.ceil(0.55 * h * w)), int(h + w - 1))
    min_area = min(int(min_area), int(h * w - 1))
    for _ in range(200):
        target = int(_RNG.integers(min_area, h * w))
        seed = (int(_RNG.integers(0, h)), int(_RNG.integers(0, w)))
        mask = np.zeros((h, w), dtype=bool)
        mask[seed] = True
        while int(mask.sum()) < target:
            frontier: set[tuple[int, int]] = set()
            for r, c in np.argwhere(mask):
                for dr, dc in _NBRS4:
                    nr = int(r + dr)
                    nc = int(c + dc)
                    if 0 <= nr < h and 0 <= nc < w and not bool(mask[nr, nc]):
                        frontier.add((int(nr), int(nc)))
            if not frontier:
                break
            nr, nc = tuple(list(frontier)[int(_RNG.integers(0, len(frontier)))])
            mask[int(nr), int(nc)] = True
        if int(mask.sum()) == h * w:
            continue
        coords = [tuple(map(int, x)) for x in np.argwhere(mask)]
        if not coords:
            continue
        if bool(mask[0, :].all() and mask[-1, :].all() and mask[:, 0].all() and mask[:, -1].all()):
            continue
        touches = {
            "top": any(r == 0 for r, _ in coords),
            "bottom": any(r == h - 1 for r, _ in coords),
            "left": any(c == 0 for _, c in coords),
            "right": any(c == w - 1 for _, c in coords),
        }
        if all(touches.values()):
            return mask
    raise RuntimeError("failed to build a connected irregular hole mask")


def _label_mask_regions(mask: np.ndarray, n_regions: int) -> np.ndarray:
    coords = [tuple(map(int, x)) for x in np.argwhere(mask)]
    assert len(coords) >= n_regions
    seed_ids = _RNG.choice(len(coords), size=n_regions, replace=False)
    labels = np.full(mask.shape, -1, dtype=int)
    frontier: list[tuple[int, int, int]] = []
    for lab, idx in enumerate(seed_ids):
        r, c = coords[int(idx)]
        labels[int(r), int(c)] = int(lab)
        frontier.append((int(r), int(c), int(lab)))

    while frontier:
        pick = int(_RNG.integers(0, len(frontier)))
        r, c, lab = frontier.pop(pick)
        nbrs = _NBRS4.copy()
        _RNG.shuffle(nbrs)
        for dr, dc in nbrs:
            nr = int(r + dr)
            nc = int(c + dc)
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and bool(mask[nr, nc]) and int(labels[nr, nc]) == -1:
                labels[int(nr), int(nc)] = int(lab)
                frontier.append((int(nr), int(nc), int(lab)))
    assert np.all(labels[mask] >= 0)
    return labels


def _colorized_patch(mask: np.ndarray, n_colors: int, colors: list[int], bg: int) -> np.ndarray:
    labels = _label_mask_regions(mask, int(n_colors))
    patch = np.full(mask.shape, int(bg), dtype=int)
    for lab, color in enumerate(colors):
        patch[labels == int(lab)] = int(color)
    return patch


def _inverse_input_patch(placed_patch: np.ndarray, n_colors: int) -> np.ndarray:
    ccw = int(_CCW_STEPS[int(n_colors)])
    return np.rot90(placed_patch, -ccw)


def _arrange_holes(hole_specs: list[dict]) -> tuple[list[tuple[int, int]], int, int]:
    n = len(hole_specs)
    order = list(range(n))
    _RNG.shuffle(order)
    layouts = ["horizontal", "vertical"]
    if n >= 3:
        layouts.append("two_row")
    layout = str(layouts[int(_RNG.integers(0, len(layouts)))])
    gap = int(_RNG.integers(1, 3))
    top = int(_RNG.integers(1, 3))
    bottom = int(_RNG.integers(1, 3))
    left = int(_RNG.integers(1, 3))
    right = int(_RNG.integers(1, 3))

    positions = [(0, 0)] * n
    if layout == "horizontal":
        row_h = max(int(hole_specs[idx]["mask"].shape[0]) for idx in order)
        cur_c = 0
        for idx in order:
            h, w = hole_specs[idx]["mask"].shape
            r = int(_RNG.integers(0, row_h - h + 1))
            positions[idx] = (int(top + r), int(left + cur_c))
            cur_c += int(w + gap)
        inner_h = int(row_h)
        inner_w = int(cur_c - gap)
    elif layout == "vertical":
        col_w = max(int(hole_specs[idx]["mask"].shape[1]) for idx in order)
        cur_r = 0
        for idx in order:
            h, w = hole_specs[idx]["mask"].shape
            c = int(_RNG.integers(0, col_w - w + 1))
            positions[idx] = (int(top + cur_r), int(left + c))
            cur_r += int(h + gap)
        inner_h = int(cur_r - gap)
        inner_w = int(col_w)
    else:
        split = int(_RNG.integers(1, n))
        row1 = order[:split]
        row2 = order[split:]
        row1_h = max(int(hole_specs[idx]["mask"].shape[0]) for idx in row1)
        row2_h = max(int(hole_specs[idx]["mask"].shape[0]) for idx in row2)
        row1_w = sum(int(hole_specs[idx]["mask"].shape[1]) for idx in row1) + gap * (len(row1) - 1)
        row2_w = sum(int(hole_specs[idx]["mask"].shape[1]) for idx in row2) + gap * (len(row2) - 1)
        cur_c = 0
        for idx in row1:
            h, w = hole_specs[idx]["mask"].shape
            r = int(_RNG.integers(0, row1_h - h + 1))
            positions[idx] = (int(top + r), int(left + cur_c))
            cur_c += int(w + gap)
        cur_c = 0
        row2_top = int(top + row1_h + gap)
        for idx in row2:
            h, w = hole_specs[idx]["mask"].shape
            r = int(_RNG.integers(0, row2_h - h + 1))
            positions[idx] = (int(row2_top + r), int(left + cur_c))
            cur_c += int(w + gap)
        inner_h = int(row1_h + gap + row2_h)
        inner_w = int(max(row1_w, row2_w))

    block_h = int(top + inner_h + bottom)
    block_w = int(left + inner_w + right)
    return positions, block_h, block_w


def _build_block(hole_specs: list[dict], block_color: int, bg: int) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    positions, block_h, block_w = _arrange_holes(hole_specs)
    inp = np.full((block_h, block_w), int(block_color), dtype=int)
    out = inp.copy()
    hole_infos: list[dict] = []
    for spec, (r0, c0) in zip(hole_specs, positions):
        mask = spec["mask"]
        h, w = mask.shape
        sub_in = inp[int(r0) : int(r0) + h, int(c0) : int(c0) + w]
        sub_out = out[int(r0) : int(r0) + h, int(c0) : int(c0) + w]
        sub_in[mask] = int(bg)
        sub_out[mask] = spec["placed_patch"][mask]
        hole_infos.append(
            {
                "local_bbox": (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)),
                "input_sprite": spec["input_sprite"],
            }
        )
    return inp, out, hole_infos


def _place_patches(
    patches: list[np.ndarray],
    bg: int,
    height: int,
    width: int,
    occupied0: np.ndarray | None = None,
    canvas0: np.ndarray | None = None,
) -> tuple[list[tuple[int, int]], np.ndarray] | None:
    canvas = np.full((height, width), int(bg), dtype=int) if canvas0 is None else np.array(canvas0, dtype=int, copy=True)
    occupied = np.zeros((height, width), dtype=bool) if occupied0 is None else np.array(occupied0, dtype=bool, copy=True)
    order = sorted(range(len(patches)), key=lambda idx: -int(np.count_nonzero(patches[idx] != int(bg))))
    positions: list[tuple[int, int] | None] = [None] * len(patches)

    for idx in order:
        patch = patches[int(idx)]
        mask = patch != int(bg)
        h, w = patch.shape
        blocked = _dilate8(occupied)
        candidates: list[tuple[int, int]] = []
        for r0 in range(height - h + 1):
            for c0 in range(width - w + 1):
                if not bool((blocked[r0 : r0 + h, c0 : c0 + w] & mask).any()):
                    candidates.append((int(r0), int(c0)))
        if not candidates:
            return None
        r0, c0 = candidates[int(_RNG.integers(0, len(candidates)))]
        _paste(canvas, patch, int(r0), int(c0), bg)
        occupied[r0 : r0 + h, c0 : c0 + w] |= mask
        positions[int(idx)] = (int(r0), int(c0))

    return [pos for pos in positions if pos is not None], canvas


def _gen_attempt() -> np.ndarray | None:
    total_holes = int(_RNG.integers(4, 6))
    n_blocks = int(_RNG.integers(2, min(4, total_holes + 1)))

    palette = list(map(int, _RNG.permutation(10)))
    bg = int(palette[0])
    block_colors = palette[1 : 1 + n_blocks]
    piece_pool = [int(c) for c in range(10) if int(c) not in {int(bg), *map(int, block_colors)}]
    if len(piece_pool) < 4:
        return None

    branch_values = [1, 2, 3, 4]
    if total_holes == 5:
        branch_values.append(int(_RNG.integers(1, 5)))
    _RNG.shuffle(branch_values)

    square_pool = [3, 4, 5]
    rect_pool = [
        (2, 3),
        (3, 2),
        (2, 4),
        (4, 2),
        (2, 5),
        (5, 2),
        (3, 4),
        (4, 3),
        (3, 5),
        (5, 3),
    ]
    used_dims: set[tuple[int, int]] = set()
    hole_specs: list[dict] = []

    for n_colors in branch_values:
        if int(n_colors) in (2, 4):
            available_sq = [s for s in square_pool if (int(s), int(s)) not in used_dims]
            if not available_sq:
                return None
            side = int(available_sq[int(_RNG.integers(0, len(available_sq)))])
            dims = (int(side), int(side))
        else:
            choices = [d for d in rect_pool + [(3, 3), (4, 4)] if tuple(map(int, d)) not in used_dims]
            if not choices:
                return None
            dims = tuple(map(int, choices[int(_RNG.integers(0, len(choices)))]))
        used_dims.add(tuple(map(int, dims)))

        mask = _random_connected_mask(int(dims[0]), int(dims[1]))
        colors = list(map(int, _RNG.choice(piece_pool, size=int(n_colors), replace=False)))
        placed_patch = _colorized_patch(mask, int(n_colors), colors, bg)
        input_sprite = _inverse_input_patch(placed_patch, int(n_colors))
        hole_specs.append(
            {
                "mask": mask,
                "n_colors": int(n_colors),
                "placed_patch": placed_patch,
                "input_sprite": input_sprite,
            }
        )

    order = list(range(total_holes))
    _RNG.shuffle(order)
    groups: list[list[int]] = [[int(order[i])] for i in range(n_blocks)]
    for idx in order[n_blocks:]:
        groups[int(_RNG.integers(0, n_blocks))].append(int(idx))
    _RNG.shuffle(groups)

    input_blocks: list[np.ndarray] = []
    output_blocks: list[np.ndarray] = []
    block_holes: list[list[dict]] = []
    for bi, group in enumerate(groups):
        specs = [hole_specs[int(idx)] for idx in group]
        inp_block, out_block, holes = _build_block(specs, int(block_colors[bi]), bg)
        input_blocks.append(inp_block)
        output_blocks.append(out_block)
        block_holes.append(holes)

    sprite_patches = [spec["input_sprite"] for spec in hole_specs]

    max_obj_h = max(max(p.shape[0] for p in input_blocks), max(p.shape[0] for p in sprite_patches))
    max_obj_w = max(max(p.shape[1] for p in input_blocks), max(p.shape[1] for p in sprite_patches))
    height_lo = max(20, int(max_obj_h + 4))
    width_lo = max(20, int(max_obj_w + 4))
    if height_lo > 30 or width_lo > 30:
        return None
    height = int(_RNG.integers(height_lo, 31))
    width = int(_RNG.integers(width_lo, 31))

    placed_blocks = _place_patches(input_blocks, bg, height, width)
    if placed_blocks is None:
        return None
    block_positions, input_grid = placed_blocks
    output_grid = np.full((height, width), int(bg), dtype=int)
    for patch, (r0, c0) in zip(output_blocks, block_positions):
        _paste(output_grid, patch, int(r0), int(c0), bg)

    block_occupied = input_grid != int(bg)
    sprite_place = _place_patches(
        sprite_patches,
        bg,
        height,
        width,
        occupied0=block_occupied,
        canvas0=input_grid,
    )
    if sprite_place is None:
        return None
    sprite_positions, input_grid = sprite_place

    if _get_bg(input_grid) != int(bg):
        return None
    if len(set(tuple(p.shape) for p in sprite_patches)) != len(sprite_patches):
        return None
    try:
        solved = main(input_grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, output_grid):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
