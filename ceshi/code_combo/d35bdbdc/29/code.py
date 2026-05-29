# Combo variant of d35bdbdc (index 29)
# combo_spec: D2/2+D11/0
# Name: 空间邻近性判定+按路径长度排序处理

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, paste


# Design analysis for the fused variant
# a) Concrete behavior:
#    Original task used a grey one-stroke path to nominate exactly two destination sprites.
#    Here the path is removed entirely. We keep the original asymmetric color logic
#    (source.border matches destination.center, then source.center is copied into the
#    destination center), but the "which relation matters first" signal now comes from
#    center-to-center Manhattan distance. We generate chains such as A->B->C(->D) where:
#      B.center = A.border, C.center = B.border, D.center = C.border.
#    If dist(A,B) < dist(B,C) < dist(C,D), processing in that order yields:
#      B := A.center, then C := B.current_center, then D := C.current_center.
#    So the early short-distance update propagates through later, longer-distance pairs.
# b) Complexity calibration:
#    A pure distance-only undirected pairing would destroy the original lookup asymmetry
#    and make the propagation direction ambiguous. Keeping border->center compatibility
#    while replacing path evidence with geometry preserves the original semantic hook and
#    keeps the visual rule learnable from a few examples. Chain length is capped at 3-4
#    sprites so the cascade is visible without producing clutter.
# c) Hidden assumptions changed from the original main():
#    - Grey cells/endpoints no longer exist, so every path-specific helper and every
#      "destination selected by endpoint port" assumption is invalid and removed.
#    - Pairs are no longer independent. A sprite can first be a destination and later
#      become a source, so center colors must be updated in sequence rather than read
#      once from the input.
#    - Output deletion is now keyed by "received at least one transfer" instead of
#      "connected to the path".
# d) Placement legality:
#    The generator keeps all sprites non-overlapping and non-4-touching. It places chain
#    sprites by backtracking so consecutive chain distances are strictly increasing, and
#    any optional distractor is pushed away from the chain.
# e) Visual observability:
#    Grey trails are gone, which is immediately visible. The remaining cue is geometric:
#    there is a near pair, then a farther pair, and the farther target inherits the color
#    produced by the nearer transfer. The distance ordering is therefore readable from
#    sprite layout rather than hidden metadata.
# f) Visual cleanliness:
#    Only 4-5 sprites are used, all in the same 3x3 box/plus family, with generous
#    spacing and at most one distractor. This avoids overlapping cues.
# g) Inferrable color change:
#    Center colors that act as lookup keys are limited to the predecessor border colors in
#    the chain. Distractor centers are chosen outside the border palette so accidental
#    extra relations do not appear.


GRID_H, GRID_W = 10, 10
BG, GREY = 0, 5
GEN_RETRIES = 400
MIN_CHAIN_DIST = 3
CHAIN_LENGTHS = [3, 4]
SHAPES = ["box", "plus"]


def _bg_boundary(g: np.ndarray) -> int:
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _detect_bordered_sprites(g: np.ndarray, bg: int, grey: int) -> list[dict]:
    mask = (g != bg) & (g != grey)
    comps = components_mask(mask, connectivity=4)
    sprites: list[dict] = []
    for pts in comps:
        cols = [int(g[r, c]) for r, c in pts]
        uniq = sorted(set(cols))
        if len(uniq) != 2:
            continue
        counts = {c: cols.count(c) for c in uniq}
        center_color = min(counts, key=lambda c: counts[c])
        if counts[center_color] != 1:
            continue
        border_color = [c for c in uniq if c != center_color][0]
        center_pos = None
        for r, c in pts:
            if int(g[r, c]) == center_color:
                center_pos = (int(r), int(c))
                break
        if center_pos is None:
            continue
        sprites.append(
            {
                "pts": [(int(r), int(c)) for r, c in pts],
                "border": int(border_color),
                "center": int(center_color),
                "center_pos": center_pos,
            }
        )
    return sprites


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _compatible_pairs(sprites: list[dict]) -> list[tuple[int, int, int]]:
    pairs: list[tuple[int, int, int]] = []
    for si, src in enumerate(sprites):
        for di, dst in enumerate(sprites):
            if si == di:
                continue
            if int(src["border"]) != int(dst["center"]):
                continue
            dist = _manhattan(src["center_pos"], dst["center_pos"])
            pairs.append((dist, si, di))
    pairs.sort(
        key=lambda item: (
            item[0],
            sprites[item[2]]["center_pos"][0],
            sprites[item[2]]["center_pos"][1],
            sprites[item[1]]["center_pos"][0],
            sprites[item[1]]["center_pos"][1],
        )
    )
    return pairs


def _apply_transfers_from_sprites(sprites: list[dict]) -> tuple[list[int], set[int], list[tuple[int, int, int]]]:
    current = [int(sp["center"]) for sp in sprites]
    kept: set[int] = set()
    applied: list[tuple[int, int, int]] = []
    for dist, si, di in _compatible_pairs(sprites):
        if int(sprites[si]["border"]) != current[di]:
            continue
        current[di] = current[si]
        kept.add(di)
        applied.append((dist, si, di))
    return current, kept, applied


def _simultaneous_transfer_values(sprites: list[dict]) -> tuple[list[int], set[int]]:
    base = [int(sp["center"]) for sp in sprites]
    out = base.copy()
    kept: set[int] = set()
    for _, si, di in _compatible_pairs(sprites):
        if int(sprites[si]["border"]) != base[di]:
            continue
        out[di] = base[si]
        kept.add(di)
    return out, kept


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate center colors through nearby bordered sprites, with short-distance relations firing first.

    Input structure:
    - The grid contains only small bordered sprites on a black background; there is no grey trail anymore.
    - Each sprite still has one border color and one single-pixel center color, and some sprites form
      border->center lookup chains when compared to nearby sprites.

    Rule:
    - Build every ordered source/destination relation where source.border equals destination.center.
    - Sort those relations by Manhattan distance between sprite centers, from shortest to longest.
    - Apply them in that order, always copying the source's current center color into the destination center.
      Because a sprite may be updated early and reused later as a source, longer-distance relations can depend
      on shorter-distance results.
    - Keep only sprites that received at least one transfer; erase all other sprites.

    Key insight:
    - Geometry no longer selects destinations via a path. Instead, distance supplies the processing order for
      a color-lookup chain, so one transferred center can cascade through progressively farther compatible sprites.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    sprites = _detect_bordered_sprites(g, bg, GREY)
    assert len(sprites) >= 3, "expected several bordered sprites"

    current_centers, kept, applied = _apply_transfers_from_sprites(sprites)
    assert applied, "expected at least one compatible transfer"

    out = g.copy()
    for idx, sp in enumerate(sprites):
        if idx in kept:
            continue
        for r, c in sp["pts"]:
            out[r, c] = bg

    keep_mask = np.zeros(out.shape, dtype=bool)
    for idx in kept:
        for r, c in sprites[idx]["pts"]:
            keep_mask[r, c] = True
        cr, cc = sprites[idx]["center_pos"]
        out[cr, cc] = current_centers[idx]

    out[(out != bg) & (~keep_mask)] = bg
    return out


def _make_sprite_patch(shape_type: str, border_color: int, center_color: int) -> np.ndarray:
    if shape_type == "box":
        patch = np.full((3, 3), border_color, dtype=int)
    else:
        patch = np.zeros((3, 3), dtype=int)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            patch[1 + dr, 1 + dc] = border_color
    patch[1, 1] = center_color
    return patch


def _place_sprite(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    paste(grid, patch, r0, c0, transparent=BG, clip=True)


def _erase_sprite(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = grid.shape
    for dr in range(3):
        for dc in range(3):
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < h and 0 <= cc < w and patch[dr, dc] != BG:
                grid[rr, cc] = BG


def _overlaps(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    h, w = grid.shape
    patch_cells: set[tuple[int, int]] = set()
    for dr in range(3):
        for dc in range(3):
            if patch[dr, dc] == BG:
                continue
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < h and 0 <= cc < w:
                if grid[rr, cc] != BG:
                    return True
                patch_cells.add((rr, cc))
    for rr, cc in patch_cells:
        for nr, nc in [(rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)]:
            if (nr, nc) in patch_cells:
                continue
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] != BG:
                return True
    return False


def _sprite_positions() -> list[tuple[int, int]]:
    return [
        (r0, c0)
        for r0 in range(-1, GRID_H)
        for c0 in range(-1, GRID_W)
        if 0 <= r0 + 1 < GRID_H and 0 <= c0 + 1 < GRID_W
    ]


def _search_chain_positions(
    grid: np.ndarray,
    patches: list[np.ndarray],
    all_pos: list[tuple[int, int]],
    idx: int,
    chosen_pos: list[tuple[int, int]],
    centers: list[tuple[int, int]],
    edge_dists: list[int],
) -> list[tuple[int, int]] | None:
    if idx == len(patches):
        return chosen_pos.copy()

    pool = list(all_pos)
    random.shuffle(pool)
    for r0, c0 in pool:
        patch = patches[idx]
        if _overlaps(grid, patch, r0, c0):
            continue
        center = (r0 + 1, c0 + 1)
        if centers:
            d = _manhattan(center, centers[-1])
            if d < MIN_CHAIN_DIST:
                continue
            if edge_dists and d <= edge_dists[-1]:
                continue
            if any(_manhattan(center, prev) <= d for prev in centers[:-1]):
                continue

        _place_sprite(grid, patch, r0, c0)
        chosen_pos.append((r0, c0))
        centers.append(center)
        if len(centers) >= 2:
            edge_dists.append(_manhattan(centers[-2], centers[-1]))
        result = _search_chain_positions(grid, patches, all_pos, idx + 1, chosen_pos, centers, edge_dists)
        if result is not None:
            return result
        if len(centers) >= 2:
            edge_dists.pop()
        centers.pop()
        chosen_pos.pop()
        _erase_sprite(grid, patch, r0, c0)
    return None


def _place_distractor(
    grid: np.ndarray,
    patch: np.ndarray,
    all_pos: list[tuple[int, int]],
    chain_centers: list[tuple[int, int]],
    min_dist: int,
) -> tuple[int, int] | None:
    pool = list(all_pos)
    random.shuffle(pool)
    for r0, c0 in pool:
        if _overlaps(grid, patch, r0, c0):
            continue
        center = (r0 + 1, c0 + 1)
        if chain_centers and min(_manhattan(center, c) for c in chain_centers) <= min_dist:
            continue
        _place_sprite(grid, patch, r0, c0)
        return r0, c0
    return None


def _build_expected_output(input_grid: np.ndarray) -> np.ndarray:
    return main(input_grid)


def _gen_attempt() -> np.ndarray | None:
    shape_type = random.choice(SHAPES)
    chain_len = random.choice(CHAIN_LENGTHS)
    total_sprites = chain_len + 1
    palette = [c for c in range(1, 10) if c != GREY]
    border_colors = random.sample(palette, total_sprites)

    all_borders = set(border_colors)
    free_centers = [c for c in range(1, 10) if c not in all_borders and c != GREY]
    if len(free_centers) < 2:
        return None

    root_center = random.choice(free_centers)
    distractor_center_choices = [c for c in free_centers if c != root_center]
    if not distractor_center_choices:
        distractor_center_choices = [root_center]
    distractor_center = random.choice(distractor_center_choices)

    centers = [root_center]
    for i in range(1, chain_len):
        centers.append(border_colors[i - 1])
    centers.append(distractor_center)

    patches = [
        _make_sprite_patch(shape_type, border_colors[i], centers[i])
        for i in range(total_sprites)
    ]

    input_grid = np.zeros((GRID_H, GRID_W), dtype=int)
    all_pos = _sprite_positions()
    chain_positions = _search_chain_positions(
        input_grid,
        patches[:chain_len],
        all_pos,
        0,
        [],
        [],
        [],
    )
    if chain_positions is None:
        return None

    chain_centers = [(r0 + 1, c0 + 1) for r0, c0 in chain_positions]
    edge_dists = [
        _manhattan(chain_centers[i], chain_centers[i + 1])
        for i in range(chain_len - 1)
    ]
    distractor_min_dist = edge_dists[-1] if edge_dists else MIN_CHAIN_DIST + 1
    if _place_distractor(input_grid, patches[-1], all_pos, chain_centers, distractor_min_dist) is None:
        return None

    sprites = _detect_bordered_sprites(input_grid, BG, GREY)
    if len(sprites) != total_sprites:
        return None

    current, kept, applied = _apply_transfers_from_sprites(sprites)
    if len(applied) != chain_len - 1:
        return None
    if len(kept) != chain_len - 1:
        return None
    if [dist for dist, _, _ in applied] != sorted(dist for dist, _, _ in applied):
        return None

    simultaneous, _ = _simultaneous_transfer_values(sprites)
    if current == simultaneous:
        return None

    out = _build_expected_output(input_grid)
    if np.array_equal(out, input_grid):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {GEN_RETRIES} attempts")
