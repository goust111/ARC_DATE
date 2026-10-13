from __future__ import annotations

import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

_TOOLBOX = Path("/home/lixinyu/snap/new/solvers")
if str(_TOOLBOX) not in sys.path:
    sys.path.append(str(_TOOLBOX))

from arc_utils import Color, components_mask, find_components, get_bg, paste, random_free_location


BG = Color.BLACK
WILDCARD = Color.BLUE
ACCENTS = (Color.RED, Color.GREEN, Color.YELLOW, Color.ORANGE)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_RETRIES = 500

L3 = ((0, 0), (1, 0), (1, 1))
L4 = ((0, 0), (1, 0), (2, 0), (2, 1))
U6 = ((0, 1), (0, 2), (1, 0), (1, 1), (2, 1), (2, 2))


def _normalize_cells(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _c in cells)
    c0 = min(c for _r, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _bbox_of_cells(cells: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int]:
    h = max(r for r, _c in cells) + 1
    w = max(c for _r, c in cells) + 1
    return h, w


def _patch_from_local_colors(local_colors: dict[tuple[int, int], int]) -> np.ndarray:
    h, w = _bbox_of_cells(list(local_colors))
    patch = np.zeros((h, w), dtype=int)
    for (r, c), color in local_colors.items():
        patch[r, c] = int(color)
    return patch


def _shift_patch_cells(
    patch: np.ndarray, anchor_r: int, anchor_c: int
) -> dict[tuple[int, int], int]:
    placed: dict[tuple[int, int], int] = {}
    for r, c in np.argwhere(patch != BG):
        placed[(int(anchor_r + r), int(anchor_c + c))] = int(patch[r, c])
    return placed


def _hole_parts_for_component(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    outside = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for rr in range(h):
        for cc in (0, w - 1):
            if not mask[rr, cc] and not outside[rr, cc]:
                outside[rr, cc] = True
                q.append((rr, cc))
    for cc in range(w):
        for rr in (0, h - 1):
            if not mask[rr, cc] and not outside[rr, cc]:
                outside[rr, cc] = True
                q.append((rr, cc))

    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    enclosed = (~mask) & (~outside)
    return components_mask(enclosed, connectivity=4)


def _component_holes(grid: np.ndarray, comp: dict) -> list[dict]:
    r0, c0, r1, c1 = comp["bbox"]
    coords = comp["coords"]
    local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True

    holes = []
    for local_coords in _hole_parts_for_component(local):
        global_coords = local_coords + np.array([r0, c0])
        holes.append(
            {
                "coords": global_coords,
                "cells": frozenset(map(tuple, global_coords.tolist())),
                "bbox": (
                    int(global_coords[:, 0].min()),
                    int(global_coords[:, 1].min()),
                    int(global_coords[:, 0].max()),
                    int(global_coords[:, 1].max()),
                ),
                "area": int(global_coords.shape[0]),
            }
        )
    return holes


def _extract_container_and_sprites(grid: np.ndarray) -> tuple[dict, list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    comps = find_components(g, bg=BG, connectivity=4, monochromatic=False, return_format="dict")
    container_candidates = []
    for comp in comps:
        colors = {int(g[r, c]) for r, c in comp["coords"]}
        if colors != {WILDCARD}:
            continue
        holes = _component_holes(g, comp)
        if holes:
            container_candidates.append((int(comp["area"]), comp, holes))

    assert len(container_candidates) == 1, "expected exactly one blue container with holes"
    _area, container, holes = container_candidates[0]

    container_coords = {tuple(map(int, rc)) for rc in container["coords"].tolist()}
    sprites = []
    for comp in comps:
        coords_list = [tuple(map(int, rc)) for rc in comp["coords"].tolist()]
        if set(coords_list) == container_coords:
            continue
        colors = [int(g[r, c]) for r, c in coords_list]
        r0 = min(r for r, _c in coords_list)
        c0 = min(c for _r, c in coords_list)
        rel = np.array([(r - r0, c - c0) for r, c in coords_list], dtype=int)
        sprites.append(
            {
                "coords": comp["coords"],
                "area": int(comp["area"]),
                "rel": rel,
                "colors": np.array(colors, dtype=int),
            }
        )

    assert 2 <= len(sprites) <= 4, "expected 2-4 source sprites"
    return container, holes, sprites


def _placement_options(sprite: dict, holes: list[dict]) -> list[dict]:
    rel = sprite["rel"]
    colors = sprite["colors"]
    max_r = int(rel[:, 0].max())
    max_c = int(rel[:, 1].max())
    placements = []
    seen: set[frozenset[tuple[int, int]]] = set()

    for hole in holes:
        r0, c0, r1, c1 = hole["bbox"]
        hole_cells = hole["cells"]
        for anchor_r in range(r0 - max_r, r1 + 1):
            for anchor_c in range(c0 - max_c, c1 + 1):
                cells = [
                    (int(anchor_r + dr), int(anchor_c + dc))
                    for dr, dc in rel.tolist()
                ]
                cell_set = frozenset(cells)
                if cell_set in seen or not cell_set <= hole_cells:
                    continue
                color_map = {
                    cell: int(colors[idx])
                    for idx, cell in enumerate(cells)
                }
                placements.append({"cells": cell_set, "colors": color_map})
                seen.add(cell_set)
    return placements


def _compatible(color_a: int, color_b: int) -> bool:
    return int(color_a) == int(color_b) or int(color_a) == WILDCARD or int(color_b) == WILDCARD


def _solution_key(assignments: list[tuple[int, dict]]) -> tuple[tuple[int, int, int], ...]:
    colored_cells = []
    for _sprite_idx, placement in assignments:
        for (r, c), color in placement["colors"].items():
            colored_cells.append((int(r), int(c), int(color)))
    return tuple(sorted(colored_cells))


def _search_solutions(
    sprites: list[dict], holes: list[dict], *, enforce_colors: bool, max_solutions: int
) -> list[tuple[tuple[int, int, int], ...]]:
    all_hole_cells = frozenset(cell for hole in holes for cell in hole["cells"])
    assert sum(sprite["area"] for sprite in sprites) == len(all_hole_cells)

    placements_by_sprite = [_placement_options(sprite, holes) for sprite in sprites]
    assert all(placements_by_sprite), "every sprite must fit at least one hole position"

    cell_to_options: dict[tuple[int, int], list[tuple[int, int]]] = {cell: [] for cell in all_hole_cells}
    for sprite_idx, placements in enumerate(placements_by_sprite):
        for place_idx, placement in enumerate(placements):
            for cell in placement["cells"]:
                cell_to_options[cell].append((sprite_idx, place_idx))

    solution_keys: set[tuple[tuple[int, int, int], ...]] = set()
    solutions: list[tuple[tuple[int, int, int], ...]] = []

    def placement_ok(placement: dict, occupied: dict[tuple[int, int], int]) -> bool:
        if not enforce_colors:
            return True
        for (r, c), color in placement["colors"].items():
            for dr, dc in DIR4:
                nb = (r + dr, c + dc)
                if nb in occupied and nb not in placement["cells"]:
                    if not _compatible(color, occupied[nb]):
                        return False
        return True

    def search(
        unused_sprites: frozenset[int],
        remaining_cells: frozenset[tuple[int, int]],
        occupied: dict[tuple[int, int], int],
        assignments: list[tuple[int, dict]],
    ) -> None:
        if len(solutions) >= max_solutions:
            return
        if not remaining_cells:
            if not unused_sprites:
                key = _solution_key(assignments)
                if key not in solution_keys:
                    solution_keys.add(key)
                    solutions.append(key)
            return

        best_options = None
        for cell in remaining_cells:
            options = []
            for sprite_idx, place_idx in cell_to_options[cell]:
                if sprite_idx not in unused_sprites:
                    continue
                placement = placements_by_sprite[sprite_idx][place_idx]
                if not placement["cells"] <= remaining_cells:
                    continue
                if not placement_ok(placement, occupied):
                    continue
                options.append((sprite_idx, place_idx))
            if best_options is None or len(options) < len(best_options):
                best_options = options
            if best_options == []:
                return

        assert best_options is not None
        for sprite_idx, place_idx in best_options:
            placement = placements_by_sprite[sprite_idx][place_idx]
            new_occupied = dict(occupied)
            new_occupied.update(placement["colors"])
            search(
                unused_sprites - {sprite_idx},
                remaining_cells - placement["cells"],
                new_occupied,
                [*assignments, (sprite_idx, placement)],
            )

    search(frozenset(range(len(sprites))), all_hole_cells, {}, [])
    return solutions


def _render_solution(grid: np.ndarray, sprites: list[dict], key: tuple[tuple[int, int, int], ...]) -> np.ndarray:
    out = np.array(grid, dtype=int)
    for sprite in sprites:
        coords = sprite["coords"]
        out[coords[:, 0], coords[:, 1]] = BG
    for r, c, color in key:
        out[r, c] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the scattered multicolor sprites into the blue object's enclosed holes so exact tiling and sprite-edge color compatibility both hold.

    Input structure:
    - Black is the background.
    - Exactly one large monochrome blue object contains 1-2 enclosed black holes.
    - Outside that object there are 2-4 scattered multicolor 4-connected source sprites, some containing blue wildcard pixels.

    Rule:
    - Detect the enclosed hole cells inside the large blue object.
    - Detect the external source sprites and translate each one without rotation.
    - Use every sprite exactly once so the translated sprites partition all hole cells exactly.
    - Whenever two placed sprites become 4-neighbor adjacent, the touching colors must match, unless at least one touching pixel is blue.
    - Erase the original sprite locations; keep the background and blue container unchanged.

    Key insight:
    - Hole coverage alone can leave several exact tilings, so the unique solution comes from combining geometry with local sprite-to-sprite boundary color compatibility, with blue acting as a wildcard.
    """
    g = np.asarray(grid, dtype=int)
    _container, holes, sprites = _extract_container_and_sprites(g)
    solutions = _search_solutions(sprites, holes, enforce_colors=True, max_solutions=2)
    assert len(solutions) == 1, "expected a unique color-compatible tiling"
    return _render_solution(g, sprites, solutions[0])


def _contact_sets(
    shape: tuple[tuple[int, int], ...], anchors: list[tuple[int, int]]
) -> list[dict[str, set[tuple[int, int]]]]:
    shape_set = set(shape)
    contacts = [{"prev": set(), "next": set()} for _ in anchors]
    for idx in range(len(anchors) - 1):
        a_r, a_c = anchors[idx]
        b_r, b_c = anchors[idx + 1]
        for cell in shape:
            gr, gc = a_r + cell[0], a_c + cell[1]
            for dr, dc in DIR4:
                nbr = (gr + dr - b_r, gc + dc - b_c)
                if nbr in shape_set:
                    contacts[idx]["next"].add(cell)
                    contacts[idx + 1]["prev"].add(nbr)
    assert all(contacts[i]["next"] for i in range(len(anchors) - 1))
    assert all(contacts[i]["prev"] for i in range(1, len(anchors)))
    return contacts


def _build_chain_group(
    shape: tuple[tuple[int, int], ...], anchors: list[tuple[int, int]]
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    count = len(anchors)
    assert count in (2, 3)
    accents = list(random.sample(list(ACCENTS), k=4))
    contacts = _contact_sets(shape, anchors)

    if count == 2:
        a, b, c = accents[:3]
        specs = [
            {"base": a, "prev": a, "next": WILDCARD},
            {"base": c, "prev": b, "next": c},
        ]
    else:
        a, b, c, d = accents
        specs = [
            {"base": a, "prev": a, "next": WILDCARD},
            {"base": b, "prev": b, "next": c},
            {"base": d, "prev": c, "next": d},
        ]

    sprites: list[np.ndarray] = []
    global_colors: dict[tuple[int, int], int] = {}
    for idx, spec in enumerate(specs):
        local_colors = {cell: int(spec["base"]) for cell in shape}
        for cell in contacts[idx]["prev"]:
            local_colors[cell] = int(spec["prev"])
        for cell in contacts[idx]["next"]:
            local_colors[cell] = int(spec["next"])
        patch = _patch_from_local_colors(local_colors)
        sprites.append(patch)
        for (r, c), color in _shift_patch_cells(patch, anchors[idx][0], anchors[idx][1]).items():
            global_colors[(r, c)] = int(color)

    max_r = max(r for r, _c in global_colors) + 1
    max_c = max(c for _r, c in global_colors) + 1
    filled = np.zeros((max_r, max_c), dtype=int)
    for (r, c), color in global_colors.items():
        filled[r, c] = int(color)
    hole_mask = filled != BG
    return hole_mask, filled, sprites


def _build_unique_group() -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    accents = list(random.sample(list(ACCENTS), k=3))
    local_colors = {}
    order = list(U6)
    split = random.randint(2, 4)
    for idx, cell in enumerate(order):
        if idx < split:
            local_colors[cell] = accents[0]
        elif idx < 5:
            local_colors[cell] = accents[1]
        else:
            local_colors[cell] = accents[2]
    sprite = _patch_from_local_colors(local_colors)
    return sprite != BG, sprite.copy(), [sprite]


def _make_group(kind: str) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    if kind == "pair_l3_h":
        return _build_chain_group(L3, [(0, 0), (0, 2)])
    if kind == "pair_l3_v":
        return _build_chain_group(L3, [(0, 0), (2, 0)])
    if kind == "pair_l4_h":
        return _build_chain_group(L4, [(0, 0), (0, 2)])
    if kind == "pair_l4_v":
        return _build_chain_group(L4, [(0, 0), (3, 0)])
    if kind == "triple_l3_h":
        return _build_chain_group(L3, [(0, 0), (0, 2), (0, 4)])
    if kind == "triple_l3_v":
        return _build_chain_group(L3, [(0, 0), (2, 0), (4, 0)])
    if kind == "unique_u6":
        return _build_unique_group()
    raise ValueError(kind)


def _place_holes(
    hole_masks: list[np.ndarray], hole_fills: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    assert len(hole_masks) == len(hole_fills)
    if len(hole_masks) == 1:
        top = random.randint(3, 5)
        left = random.randint(4, 6)
        bottom = random.randint(3, 5)
        right = random.randint(4, 6)
        h, w = hole_masks[0].shape
        H = top + h + bottom
        W = left + w + right
        container_mask = np.ones((H, W), dtype=bool)
        filled = np.zeros((H, W), dtype=int)
        positions = [(top, left)]
        container_mask[top : top + h, left : left + w][hole_masks[0]] = False
        paste(filled, hole_fills[0], top, left, transparent=BG)
        return container_mask, filled, positions

    layout_horizontal = random.random() < 0.5
    h0, w0 = hole_masks[0].shape
    h1, w1 = hole_masks[1].shape
    if layout_horizontal:
        gap = random.randint(3, 5)
        top0 = random.randint(3, 5)
        top1 = random.randint(3, 5)
        left0 = random.randint(3, 5)
        left1 = left0 + w0 + gap
        H = max(top0 + h0, top1 + h1) + random.randint(3, 5)
        W = left1 + w1 + random.randint(3, 6)
        positions = [(top0, left0), (top1, left1)]
    else:
        gap = random.randint(3, 5)
        left0 = random.randint(3, 5)
        left1 = random.randint(3, 5)
        top0 = random.randint(3, 5)
        top1 = top0 + h0 + gap
        H = top1 + h1 + random.randint(3, 5)
        W = max(left0 + w0, left1 + w1) + random.randint(3, 6)
        positions = [(top0, left0), (top1, left1)]

    container_mask = np.ones((H, W), dtype=bool)
    filled = np.zeros((H, W), dtype=int)
    for (top, left), mask, patch in zip(positions, hole_masks, hole_fills):
        hh, ww = mask.shape
        container_mask[top : top + hh, left : left + ww][mask] = False
        paste(filled, patch, top, left, transparent=BG)
    return container_mask, filled, positions


def _dilate(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    out = mask.copy()
    for _ in range(iterations):
        grown = out.copy()
        grown[1:, :] |= out[:-1, :]
        grown[:-1, :] |= out[1:, :]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        out = grown
    return out


def _add_boundary_notches(container_mask: np.ndarray, hole_mask: np.ndarray) -> np.ndarray:
    mask = container_mask.copy()
    protected = _dilate(hole_mask, iterations=1)
    target_holes = len(components_mask(hole_mask, connectivity=4))

    def valid(candidate: np.ndarray) -> bool:
        if len(components_mask(candidate, connectivity=4)) != 1:
            return False
        if len(_hole_parts_for_component(candidate)) != target_holes:
            return False
        return True

    attempts = 0
    desired = random.randint(1, 2)
    applied = 0
    H, W = mask.shape
    while applied < desired and attempts < 40:
        attempts += 1
        candidate = mask.copy()
        side = random.choice(("top", "bottom", "left", "right"))
        if side in ("top", "bottom"):
            depth = random.randint(1, min(2, H - 3))
            span = random.randint(2, min(4, W - 2))
            start = random.randint(0, W - span)
            rows = slice(0, depth) if side == "top" else slice(H - depth, H)
            cols = slice(start, start + span)
        else:
            depth = random.randint(1, min(2, W - 3))
            span = random.randint(2, min(4, H - 2))
            start = random.randint(0, H - span)
            rows = slice(start, start + span)
            cols = slice(0, depth) if side == "left" else slice(W - depth, W)
        if protected[rows, cols].any():
            continue
        candidate[rows, cols] = False
        if valid(candidate):
            mask = candidate
            applied += 1

    return mask


def _build_container_and_solution(
    groups: list[tuple[np.ndarray, np.ndarray, list[np.ndarray]]]
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    hole_masks = [group[0] for group in groups]
    hole_fills = [group[1] for group in groups]
    sprite_patches = [sprite for _mask, _fill, sprites in groups for sprite in sprites]
    container_mask, filled, _positions = _place_holes(hole_masks, hole_fills)
    hole_union = container_mask == False
    container_mask = _add_boundary_notches(container_mask, hole_union)
    container_patch = np.zeros_like(container_mask, dtype=int)
    container_patch[container_mask] = WILDCARD
    return container_patch, filled, sprite_patches


def _assemble_case() -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    cases = [
        ["pair_l3_h"],
        ["pair_l3_v"],
        ["pair_l4_h"],
        ["pair_l4_v"],
        ["triple_l3_h"],
        ["triple_l3_v"],
        ["pair_l3_h", "unique_u6"],
        ["pair_l4_v", "unique_u6"],
        ["pair_l3_h", "pair_l4_v"],
        ["pair_l3_v", "pair_l4_h"],
    ]
    kinds = random.choice(cases)
    groups = [_make_group(kind) for kind in kinds]
    return _build_container_and_solution(groups)


def _scatter_sprites(
    container_patch: np.ndarray, sprite_patches: list[np.ndarray], shape: tuple[int, int]
) -> tuple[np.ndarray, tuple[int, int]]:
    grid = np.zeros(shape, dtype=int)
    container_r, container_c = random_free_location(
        grid, container_patch, transparent=BG, border=1, padding=0, padding_connectivity=4
    )
    paste(grid, container_patch, container_r, container_c)
    for patch in sorted(sprite_patches, key=lambda arr: int((arr != BG).sum()), reverse=True):
        r0, c0 = random_free_location(
            grid, patch, transparent=BG, border=0, padding=1, padding_connectivity=8
        )
        paste(grid, patch, r0, c0, transparent=BG)
    return grid, (container_r, container_c)


def _required_canvas(container_patch: np.ndarray, sprite_patches: list[np.ndarray]) -> tuple[int, int]:
    ch, cw = container_patch.shape
    extra_h = sum(p.shape[0] for p in sprite_patches) + 4
    extra_w = max((p.shape[1] for p in sprite_patches), default=0) + 6
    min_h = max(18, ch + 2, extra_h)
    min_w = max(18, cw + 2, cw + extra_w)
    return min_h, min_w


def _validate_generated(inp: np.ndarray, expected: np.ndarray) -> None:
    container, holes, sprites = _extract_container_and_sprites(inp)
    assert 1 <= len(holes) <= 2
    assert all(6 <= int(hole["area"]) <= 12 for hole in holes)
    assert all(3 <= int(sprite["area"]) <= 6 for sprite in sprites)

    for sprite in sprites:
        colors = {int(inp[r, c]) for r, c in sprite["coords"]}
        assert 2 <= len(colors) <= 3
        assert len(colors - {WILDCARD, *ACCENTS}) == 0

    area_ratio = int(container["area"]) / float(inp.shape[0] * inp.shape[1])
    assert 0.22 <= area_ratio <= 0.55

    geometry_only = _search_solutions(sprites, holes, enforce_colors=False, max_solutions=3)
    color_aware = _search_solutions(sprites, holes, enforce_colors=True, max_solutions=2)
    assert len(geometry_only) >= 2, "geometry alone should be ambiguous"
    assert len(color_aware) == 1, "edge colors should make the tiling unique"
    assert np.array_equal(main(inp), expected)

    wildcard_contact = False
    colored = {(r, c): color for r, c, color in color_aware[0]}
    for r, c, color in color_aware[0]:
        for dr, dc in ((1, 0), (0, 1)):
            nb = (r + dr, c + dc)
            if nb not in colored:
                continue
            other = colored[nb]
            if color != other and (color == WILDCARD or other == WILDCARD):
                wildcard_contact = True
    assert wildcard_contact, "at least one matched boundary should rely on blue wildcard"


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        container_patch, fill_patch, sprite_patches = _assemble_case()
        min_h, min_w = _required_canvas(container_patch, sprite_patches)
        max_h = min(30, max(min_h, 24))
        max_w = min(30, max(min_w, 24))
        H = random.randint(min_h, max_h)
        W = random.randint(min_w, max_w)
        if H > 30 or W > 30:
            continue

        try:
            inp, (container_r, container_c) = _scatter_sprites(container_patch, sprite_patches, (H, W))
        except ValueError:
            continue

        expected = inp.copy()
        try:
            _container, _holes, scattered_sprites = _extract_container_and_sprites(inp)
        except AssertionError:
            continue
        for sprite in scattered_sprites:
            coords = sprite["coords"]
            expected[coords[:, 0], coords[:, 1]] = BG
        paste(expected, fill_patch, container_r, container_c, transparent=BG)

        try:
            _validate_generated(inp, expected)
        except AssertionError:
            continue
        return inp

    raise RuntimeError("failed to generate a non-degenerate puzzle instance")
