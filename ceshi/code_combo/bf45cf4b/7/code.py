# Combo variant of bf45cf4b (index 7)
# combo_spec: D1/2+D8/0
# Name: 邻域条件双瓦片选择
#
# Design analysis
# a) Concrete behavior and trace
#    The fused input contains exactly three disconnected foreground objects:
#    - one monochrome rectangular control sprite with bg holes inside its bbox
#    - two multicolor square tiles of the same size
#    Original task: every filled control cell expanded to one repeated tile.
#    Fused task: first identify tile A and tile B from the two candidates using
#    a visible rule. This implementation uses foreground color count:
#    - tile A = the more colorful tile
#    - tile B = the less colorful tile
#    Then expand each filled control cell according to the control mask's local
#    8-neighborhood inside the control bbox:
#    - 8 filled neighbors  -> tile A
#    - otherwise           -> tile B
#    - bg holes            -> bg-only blocks
#    Small trace: if the control sprite has a solid 4x4 core with thin arms,
#    the center 2x2 cells are fully surrounded and receive tile A, while arm
#    cells and outline cells see at least one missing neighbor and receive tile
#    B. The output therefore mixes two motifs on one regular lattice, unlike
#    the source task's uniform tiling.
# b) Complexity calibration
#    D8/0 adds a third foreground object, but this fusion should not devolve to
#    "pick one correct tile and ignore the other". Both candidate tiles are
#    semantically active in the output. To keep the expanded image on one clean
#    square lattice, the two tiles must share one common side length; using
#    different sizes would make per-pixel expansion inconsistent. Distinguish
#    A/B by color count instead of area, and force the control mask to contain
#    both interior and non-interior filled cells so both tiles appear in every
#    valid sample.
# c) Implicit assumptions broken from the original main()
#    - `len(comps) == 2` is no longer valid; the solver now requires 3 objects.
#    - "the only multicolor square object is the tile" is false; there are two.
#    - "all filled control cells paste the same patch" is false; tile choice now
#      depends on a local 8-neighborhood predicate.
#    - Sorting by area would be unsafe here because equal-size tiles are needed
#      for a shared lattice. The solver instead assumes equal square size and
#      different visible fg color counts; the generator guarantees both.
# d) Placement legality
#    The generator reserves whole bounding boxes, not just current foreground
#    pixels, before pasting the actual shapes. This avoids the subtle failure
#    mode where one disconnected object gets dropped into another object's hole
#    or empty bbox corner: that would still satisfy 8-connectivity separation
#    but would pollute cropped object bboxes and hurt readability. Bounding-box
#    reservation plus border/padding keeps all three objects non-overlapping,
#    in-bounds, disconnected, and individually readable. The control sprite is
#    still pasted with `transparent=bg` so its holes remain holes inside its
#    own reserved rectangle.
# e) Visual observability
#    A human can compare the three separated input objects and then inspect the
#    output: the more colorful tile consistently appears at the fully enclosed
#    positions of the control mask, while the less colorful tile appears on the
#    edge and around holes. This is visible from repeated interior/edge cases
#    within one example and across examples; it does not rely on numeric color
#    ids or hidden ordering.
# f) Visual cleanliness
#    No overlays or markers are added. The input remains sparse and readable:
#    one control sprite plus two square tile candidates. The output remains
#    tightly cropped and grid-aligned because both tiles have the same size.
# g) Change inferability
#    The fusion introduces no arbitrary recoloring. Colors are copied verbatim
#    from tile A or tile B. The only decision variable is the visibly deducible
#    8-neighborhood fullness of the control mask.
# 9) New solver assumptions and generator guarantees
#    Solver assumptions:
#    - exactly 3 foreground components
#    - exactly 1 monochrome control sprite
#    - exactly 2 multicolor square tiles
#    - the two tiles have the same square size
#    - the two tiles have different foreground color counts
#    - the control mask contains at least one interior filled cell and at least
#      one non-interior filled cell
#    Generator guarantees each of these before accepting a sample.

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, find_components, get_bg, paste, random_free_location


def _neighbor_count8(mask: np.ndarray, r: int, c: int) -> int:
    h, w = mask.shape
    total = 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]):
                total += 1
    return total


def _component_infos(grid: np.ndarray, bg: int) -> list[dict]:
    comps = find_components(grid, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    infos = []
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
        for rr, cc in np.asarray(comp["coords"], dtype=int):
            patch[int(rr) - r0, int(cc) - c0] = int(grid[int(rr), int(cc)])
        infos.append(
            {
                "bbox": comp["bbox"],
                "patch": patch,
                "shape": tuple(int(x) for x in patch.shape),
                "cols": colors_present(patch, exclude={bg}),
            }
        )
    return infos


def _classify_objects(grid: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    infos = _component_infos(grid, bg)
    assert len(infos) == 3

    ctrl_candidates = [info for info in infos if len(info["cols"]) == 1]
    tile_candidates = [
        info for info in infos if len(info["cols"]) >= 2 and info["shape"][0] == info["shape"][1]
    ]
    assert len(ctrl_candidates) == 1
    assert len(tile_candidates) == 2

    tile_shapes = {info["shape"] for info in tile_candidates}
    assert len(tile_shapes) == 1

    tile_candidates.sort(key=lambda info: len(info["cols"]), reverse=True)
    color_counts = [len(info["cols"]) for info in tile_candidates]
    assert color_counts[0] > color_counts[1]

    ctrl = ctrl_candidates[0]["patch"]
    tile_a = tile_candidates[0]["patch"]
    tile_b = tile_candidates[1]["patch"]
    return ctrl, tile_a, tile_b


def _compose_output(tile_a: np.ndarray, tile_b: np.ndarray, ctrl: np.ndarray, bg: int) -> np.ndarray:
    assert tile_a.shape == tile_b.shape

    th, tw = map(int, tile_a.shape)
    ch, cw = map(int, ctrl.shape)
    out = np.full((ch * th, cw * tw), int(bg), dtype=int)
    mask = ctrl != int(bg)

    for r in range(ch):
        for c in range(cw):
            if not bool(mask[r, c]):
                continue
            patch = tile_a if _neighbor_count8(mask, r, c) == 8 else tile_b
            paste(out, patch, r * th, c * tw)
    return out


def _compose_uniform(tile: np.ndarray, ctrl: np.ndarray, bg: int) -> np.ndarray:
    th, tw = map(int, tile.shape)
    ch, cw = map(int, ctrl.shape)
    out = np.full((ch * th, cw * tw), int(bg), dtype=int)
    mask = ctrl != int(bg)
    for r in range(ch):
        for c in range(cw):
            if bool(mask[r, c]):
                paste(out, tile, r * th, c * tw)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify the monochrome control sprite and the two same-sized multicolor
    tiles. The more colorful tile is A and the less colorful tile is B.
    Expand each filled control pixel using tile A only when all 8 neighboring
    control cells are also filled; otherwise use tile B.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    ctrl, tile_a, tile_b = _classify_objects(g, bg)
    return _compose_output(tile_a, tile_b, ctrl, bg)


def _is_single_component(mask: np.ndarray) -> bool:
    comps = find_components(mask.astype(int), bg=0, connectivity=8, monochromatic=False, return_format="dict")
    return len(comps) == 1


def _count_interior_edge(mask: np.ndarray) -> tuple[int, int]:
    interior = 0
    edge = 0
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]):
                continue
            if _neighbor_count8(mask, r, c) == 8:
                interior += 1
            else:
                edge += 1
    return interior, edge


def _make_tile(bg: int, fg_colors: list[int], size: int, n_colors: int, rng: np.random.Generator) -> np.ndarray:
    rr, cc = np.indices((size, size))
    formulas = (
        lambda shift: (rr + 2 * cc + shift) % n_colors,
        lambda shift: (2 * rr + cc + shift) % n_colors,
        lambda shift: (rr * cc + rr + cc + shift) % n_colors,
        lambda shift: ((rr // 2) + cc + shift) % n_colors,
    )

    for _ in range(200):
        palette = [int(x) for x in rng.choice(fg_colors, size=n_colors, replace=False)]
        pattern = formulas[int(rng.integers(0, len(formulas)))]
        shift = int(rng.integers(0, n_colors))
        tile = np.array([[palette[int(idx)] for idx in row] for row in pattern(shift)], dtype=int)

        # Break accidental look-alikes while keeping all cells foreground.
        for _ in range(int(rng.integers(1, 3))):
            r = int(rng.integers(0, size))
            c = int(rng.integers(0, size))
            tile[r, c] = int(palette[int(rng.integers(0, n_colors))])

        if len(colors_present(tile, exclude={bg})) != n_colors:
            continue
        if not _is_single_component(tile != bg):
            continue
        return tile
    raise RuntimeError("failed to construct tile")


def _make_ctrl_mask(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    for _ in range(400):
        mask = np.zeros((h, w), dtype=bool)

        top = int(rng.integers(1, h - 3))
        left = int(rng.integers(1, w - 3))
        if top + 4 > h or left + 4 > w:
            continue
        mask[top : top + 4, left : left + 4] = True

        top_col = left + int(rng.integers(1, 3))
        bottom_col = left + int(rng.integers(1, 3))
        left_row = top + int(rng.integers(1, 3))
        right_row = top + int(rng.integers(1, 3))

        mask[:top, top_col] = True
        mask[top + 4 :, bottom_col] = True
        mask[left_row, :left] = True
        mask[right_row, left + 4 :] = True

        area = h * w
        max_fill = int(np.floor(0.72 * area))
        extra_budget = max(0, min(5, max_fill - int(mask.sum())))
        for _ in range(int(rng.integers(0, extra_budget + 1))):
            frontier = []
            for r in range(h):
                for c in range(w):
                    if mask[r, c]:
                        continue
                    found = False
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0:
                                continue
                            rr = r + dr
                            cc = c + dc
                            if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]):
                                found = True
                                break
                        if found:
                            break
                    if found:
                        frontier.append((r, c))
            if not frontier:
                break
            pick = frontier[int(rng.integers(0, len(frontier)))]
            mask[pick] = True

        fill = int(mask.sum())
        if fill == area:
            continue
        fill_ratio = fill / area
        if not (0.35 <= fill_ratio <= 0.72):
            continue
        if not _is_single_component(mask):
            continue

        interior, edge = _count_interior_edge(mask)
        if interior < 2 or edge < 6:
            continue
        return mask
    raise RuntimeError("failed to build control mask")


def _make_ctrl(bg: int, color: int, rng: np.random.Generator) -> np.ndarray:
    h = int(rng.integers(6, 8))
    w = int(rng.integers(6, 8))
    mask = _make_ctrl_mask(h, w, rng)
    ctrl = np.full((h, w), int(bg), dtype=int)
    ctrl[mask] = int(color)
    return ctrl


def _make_input(
    tile_a: np.ndarray,
    tile_b: np.ndarray,
    ctrl: np.ndarray,
    bg: int,
    rng: np.random.Generator,
) -> np.ndarray:
    objects = [tile_a, tile_b, ctrl]
    footprints = [np.ones(obj.shape, dtype=int) for obj in objects]
    max_h = max(int(obj.shape[0]) for obj in objects)
    max_w = max(int(obj.shape[1]) for obj in objects)
    h_min = max(12, max_h + 2)
    w_min = max(12, max_w + 2)

    for _ in range(120):
        H = int(rng.integers(h_min, 19))
        W = int(rng.integers(w_min, 19))
        grid = np.full((H, W), int(bg), dtype=int)
        occ_grid = np.zeros((H, W), dtype=int)

        order = [0, 1, 2]
        rng.shuffle(order)
        ok = True
        for idx in order:
            try:
                pos = random_free_location(
                    occ_grid,
                    footprints[idx],
                    transparent=0,
                    border=1,
                    padding=1,
                    padding_connectivity=8,
                )
            except ValueError:
                ok = False
                break
            patch = objects[idx]
            paste(occ_grid, footprints[idx], pos[0], pos[1])
            if idx == 2:
                paste(grid, patch, pos[0], pos[1], transparent=bg)
            else:
                paste(grid, patch, pos[0], pos[1])
        if ok and int(get_bg(grid)) == int(bg):
            return grid
    raise RuntimeError("placement failed")


def _behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    ctrl, tile_a, tile_b = _classify_objects(g, bg)
    mask = ctrl != int(bg)
    interior, edge = _count_interior_edge(mask)
    fused = _compose_output(tile_a, tile_b, ctrl, bg)
    all_a = _compose_uniform(tile_a, ctrl, bg)
    all_b = _compose_uniform(tile_b, ctrl, bg)
    return {
        "interior_cells": int(interior),
        "edge_cells": int(edge),
        "tile_side": int(tile_a.shape[0]),
        "fg_colors_a": int(len(colors_present(tile_a, exclude={bg}))),
        "fg_colors_b": int(len(colors_present(tile_b, exclude={bg}))),
        "tile_pixel_diff": int(np.count_nonzero(tile_a != tile_b)),
        "diff_vs_all_a": int(np.count_nonzero(fused != all_a)),
        "diff_vs_all_b": int(np.count_nonzero(fused != all_b)),
    }


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    palette = [int(x) for x in rng.choice(np.arange(10, dtype=int), size=7, replace=False)]
    bg = int(palette[0])
    ctrl_color = int(palette[1])
    tile_palette = [int(x) for x in palette[2:]]

    size = int(rng.choice((3, 4)))
    n_colors_a = int(rng.choice((3, 4)))
    n_colors_b = int(rng.choice((2, 3)))
    if n_colors_b >= n_colors_a:
        n_colors_b = n_colors_a - 1

    tile_a = _make_tile(bg, tile_palette, size, n_colors_a, rng)
    tile_b = _make_tile(bg, tile_palette, size, n_colors_b, rng)
    if np.array_equal(tile_a, tile_b):
        return None
    if int(np.count_nonzero(tile_a != tile_b)) < size:
        return None

    ctrl = _make_ctrl(bg, ctrl_color, rng)
    expected = _compose_output(tile_a, tile_b, ctrl, bg)
    all_a = _compose_uniform(tile_a, ctrl, bg)
    all_b = _compose_uniform(tile_b, ctrl, bg)
    if np.array_equal(expected, all_a) or np.array_equal(expected, all_b):
        return None

    inp = _make_input(tile_a, tile_b, ctrl, bg, rng)
    metrics = _behavior_metrics(inp)
    if metrics["interior_cells"] < 1 or metrics["edge_cells"] < 1:
        return None
    if metrics["fg_colors_a"] <= metrics["fg_colors_b"]:
        return None
    if metrics["tile_pixel_diff"] < metrics["tile_side"]:
        return None
    if metrics["diff_vs_all_a"] == 0 or metrics["diff_vs_all_b"] == 0:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid input")
