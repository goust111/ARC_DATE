from __future__ import annotations

from collections import Counter
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: BBox) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> None:
    r0 = int(r0)
    c0 = int(c0)
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        view[:] = patch
    else:
        mask = patch != int(transparent)
        view[mask] = patch[mask]


def _neighbors(r: int, c: int, connectivity: int) -> list[Coord]:
    if int(connectivity) == 4:
        return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if dr != 0 or dc != 0
    ]


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors(rr, cc, connectivity):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _repeat_tile(tile: np.ndarray, h: int, w: int) -> np.ndarray:
    reps_r = (int(h) + tile.shape[0] - 1) // tile.shape[0]
    reps_c = (int(w) + tile.shape[1] - 1) // tile.shape[1]
    return np.tile(tile, (reps_r, reps_c))[:h, :w]


def _component_records(grid: np.ndarray, bg: int) -> list[dict]:
    comps = components_mask(grid != bg, connectivity=4)
    records = []
    for coords in comps:
        records.append(
            {
                "coords": coords,
                "bbox": bbox_coords(coords),
                "area": int(coords.shape[0]),
            }
        )
    return records


def _texture_pattern(region: np.ndarray, bg: int, pr: int, pc: int) -> np.ndarray | None:
    pattern = np.full((pr, pc), -1, dtype=int)
    for r, c in np.argwhere(region != bg):
        rr = int(r) % pr
        cc = int(c) % pc
        color = int(region[r, c])
        if pattern[rr, cc] == -1:
            pattern[rr, cc] = color
        elif int(pattern[rr, cc]) != color:
            return None
    if np.any(pattern == -1):
        return None
    return pattern


def _sprite_record(grid: np.ndarray, coords: np.ndarray) -> dict:
    r0, c0, r1, c1 = bbox_coords(coords)
    cells = []
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        cells.append((rr - r0, cc - c0, int(grid[rr, cc])))
    cells.sort()
    return {
        "area": int(coords.shape[0]),
        "cells": tuple(cells),
        "height": int(r1 - r0 + 1),
        "width": int(c1 - c0 + 1),
    }


def _sprite_placements(
    sprite: dict,
    defect_cells: frozenset[Coord],
    color_at: dict[Coord, int],
    bbox: BBox,
) -> list[frozenset[Coord]]:
    r0, c0, r1, c1 = bbox
    max_dr = max(dr for dr, _dc, _color in sprite["cells"])
    max_dc = max(dc for _dr, dc, _color in sprite["cells"])
    placements: list[frozenset[Coord]] = []
    seen: set[tuple[Coord, ...]] = set()
    for ar in range(r0, r1 - max_dr + 1):
        for ac in range(c0, c1 - max_dc + 1):
            coords = []
            ok = True
            for dr, dc, color in sprite["cells"]:
                cell = (ar + dr, ac + dc)
                if cell not in defect_cells or color_at.get(cell) != int(color):
                    ok = False
                    break
                coords.append(cell)
            if ok:
                key = tuple(sorted(coords))
                if key not in seen:
                    seen.add(key)
                    placements.append(frozenset(coords))
    return placements


def _unique_exact_cover(
    sprites: list[dict],
    defect_cells: frozenset[Coord],
    color_at: dict[Coord, int],
) -> bool:
    if not defect_cells or not sprites:
        return False
    if sum(sprite["area"] for sprite in sprites) != len(defect_cells):
        return False

    defect_bbox = bbox_coords(np.array(list(defect_cells), dtype=int))
    placements_by_sprite: list[list[frozenset[Coord]]] = []
    for sprite in sprites:
        placements = _sprite_placements(sprite, defect_cells, color_at, defect_bbox)
        if not placements:
            return False
        placements_by_sprite.append(placements)

    solutions: set[tuple[Coord, ...]] = set()

    def search(remaining: frozenset[Coord], unused: frozenset[int], cover: list[frozenset[Coord]]) -> None:
        if len(solutions) > 1:
            return
        if not remaining:
            if not unused:
                key = tuple(sorted(cell for cells in cover for cell in cells))
                solutions.add(key)
            return
        if sum(sprites[i]["area"] for i in unused) != len(remaining):
            return

        best_options: list[tuple[int, frozenset[Coord]]] | None = None
        for cell in remaining:
            options = []
            for sprite_index in unused:
                for placement in placements_by_sprite[sprite_index]:
                    if cell in placement and placement <= remaining:
                        options.append((sprite_index, placement))
            if not options:
                return
            if best_options is None or len(options) < len(best_options):
                best_options = options
                if len(best_options) == 1:
                    break

        assert best_options is not None
        for sprite_index, placement in best_options:
            search(remaining - placement, unused - {sprite_index}, [*cover, placement])

    search(defect_cells, frozenset(range(len(sprites))), [])
    return len(solutions) == 1


def _solve_with_texture_component(grid: np.ndarray, bg: int, components: list[dict], texture_index: int) -> np.ndarray | None:
    texture = components[texture_index]
    r0, c0, r1, c1 = texture["bbox"]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return None

    for i, comp in enumerate(components):
        if i == texture_index:
            continue
        coords = comp["coords"]
        inside = (
            (coords[:, 0] >= r0)
            & (coords[:, 0] <= r1)
            & (coords[:, 1] >= c0)
            & (coords[:, 1] <= c1)
        )
        if bool(np.any(inside)):
            return None

    sprites = [_sprite_record(grid, comp["coords"]) for i, comp in enumerate(components) if i != texture_index]
    if not sprites:
        return None

    region = crop_bbox(grid, texture["bbox"])
    if not np.any(region == bg):
        return None

    candidate_outputs: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}
    for pr in range(1, h + 1):
        for pc in range(1, w + 1):
            pattern = _texture_pattern(region, bg, pr, pc)
            if pattern is None:
                continue
            ideal = _repeat_tile(pattern, h, w)
            defect_mask = region != ideal
            if not np.any(defect_mask):
                continue
            if bool(np.any((region != bg) & defect_mask)):
                continue
            if defect_mask[0].any() or defect_mask[-1].any() or defect_mask[:, 0].any() or defect_mask[:, -1].any():
                continue

            defect_coords = np.argwhere(defect_mask) + np.array([r0, c0], dtype=int)
            defect_cells = frozenset(map(tuple, defect_coords.tolist()))
            if len(defect_cells) != sum(sprite["area"] for sprite in sprites):
                continue

            color_at = {
                (r0 + rr, c0 + cc): int(ideal[rr, cc])
                for rr, cc in np.argwhere(defect_mask)
            }
            if not _unique_exact_cover(sprites, defect_cells, color_at):
                continue

            out = np.full_like(grid, bg)
            out[r0 : r1 + 1, c0 : c1 + 1] = ideal
            candidate_outputs[(out.shape, out.tobytes())] = out

    if len(candidate_outputs) != 1:
        return None
    return next(iter(candidate_outputs.values()))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the hidden periodic texture, detect the missing cells, and translate the scattered cut-out sprites back into those defects.

    Input structure:
    - A background grid contains one rectangular multi-color texture region whose colors repeat with an unknown row period and column period.
    - Some interior cells of that texture have been replaced by the background color, creating defect regions with no border.
    - Outside the texture, the removed textured pieces appear as separated source sprites on the background.

    Rule:
    - Infer which foreground component is the damaged texture and recover the periodic tile from the intact cells inside its bounding box.
    - The cells where the observed texture disagrees with the recovered ideal texture are the defect cells.
    - Translate every outside sprite without rotation, reflection, or recoloring so that the translated sprites exactly partition all defect cells and match the ideal texture colors there.
    - Erase the original sprite locations; keep the background unchanged and restore the texture region to its ideal periodic appearance.

    Key insight:
    - The holes are invisible until the periodic texture is mentally completed; once the ideal lattice is recovered, the scattered sprites become exact cut-outs that can only fit back by pure translation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2-D grid"

    bg = get_bg(g)
    components = _component_records(g, bg)
    assert len(components) >= 2, "expected one texture component and at least one sprite"

    outputs: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}
    order = sorted(
        range(len(components)),
        key=lambda i: (-components[i]["area"], -((components[i]["bbox"][2] - components[i]["bbox"][0] + 1) * (components[i]["bbox"][3] - components[i]["bbox"][1] + 1))),
    )
    for texture_index in order:
        out = _solve_with_texture_component(g, bg, components, texture_index)
        if out is not None:
            outputs[(out.shape, out.tobytes())] = out

    assert len(outputs) == 1, "expected a unique texture reconstruction"
    return next(iter(outputs.values()))


BG = 0
GEN_RETRIES = 600
GRID_H_RANGE = (18, 27)
GRID_W_RANGE = (18, 30)
PERIOD_RANGE = (2, 4)
TEXTURE_H_MAX = 14
TEXTURE_W_MAX = 15
SPRITE_COUNT_RANGE = (3, 6)
SPRITE_AREA_RANGE = (2, 5)
TOTAL_DEFECT_AREA_RANGE = (8, 18)
PALETTE = tuple(range(1, 10))


def _has_period(tile: np.ndarray, pr: int, pc: int) -> bool:
    for r in range(tile.shape[0]):
        for c in range(tile.shape[1]):
            if int(tile[r, c]) != int(tile[r % pr, c % pc]):
                return False
    return True


def _random_tile() -> tuple[np.ndarray, int, int] | None:
    for _ in range(200):
        pr = random.randint(*PERIOD_RANGE)
        pc = random.randint(*PERIOD_RANGE)
        n_colors = random.randint(3, min(6, pr * pc))
        colors = random.sample(PALETTE, n_colors)
        tile = np.array([[random.choice(colors) for _ in range(pc)] for _ in range(pr)], dtype=int)
        if len(set(map(int, tile.flatten()))) < 3:
            continue
        if any(_has_period(tile, sr, sc) for sr in range(1, pr + 1) for sc in range(1, pc + 1) if (sr, sc) != (pr, pc)):
            continue
        return tile, pr, pc
    return None


def _random_composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    vals = [b - a for a, b in zip([0] + cuts, cuts + [total])]
    random.shuffle(vals)
    return vals


def _gen_sprite_areas(n: int) -> list[int] | None:
    for _ in range(200):
        areas = [random.randint(*SPRITE_AREA_RANGE) for _ in range(n)]
        total = sum(areas)
        if TOTAL_DEFECT_AREA_RANGE[0] <= total <= TOTAL_DEFECT_AREA_RANGE[1]:
            return areas
    return None


def _normalize_shape(cells: set[Coord]) -> list[Coord]:
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    return sorted((r - min_r, c - min_c) for r, c in cells)


def _random_polyomino(area: int) -> list[Coord] | None:
    for _ in range(200):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors(r, c, 4):
                    if (nr, nc) not in cells:
                        frontier.append((nr, nc))
            cells.add(random.choice(frontier))
        shape = _normalize_shape(cells)
        h = 1 + max(r for r, _c in shape)
        w = 1 + max(c for _r, c in shape)
        if h <= 4 and w <= 4:
            return shape
    return None


def _touches4(cells: set[Coord], other: set[Coord]) -> bool:
    if not other:
        return False
    for r, c in cells:
        for nr, nc in _neighbors(r, c, 4):
            if (nr, nc) in other:
                return True
    return False


def _place_defect_sprites(h: int, w: int, shapes: list[list[Coord]], cluster_sizes: list[int]) -> list[set[Coord]] | None:
    occupied: set[Coord] = set()
    placements: list[set[Coord]] = []
    shape_index = 0
    for cluster_size in cluster_sizes:
        cluster_cells: set[Coord] = set()
        for _ in range(cluster_size):
            shape = shapes[shape_index]
            shape_index += 1
            shape_h = 1 + max(r for r, _c in shape)
            shape_w = 1 + max(c for _r, c in shape)
            candidates: list[set[Coord]] = []
            for ar in range(1, h - shape_h):
                for ac in range(1, w - shape_w):
                    cells = {(ar + dr, ac + dc) for dr, dc in shape}
                    if cells & occupied:
                        continue
                    if cluster_cells:
                        if not _touches4(cells, cluster_cells):
                            continue
                        if _touches4(cells, occupied - cluster_cells):
                            continue
                    else:
                        if _touches4(cells, occupied):
                            continue
                    candidates.append(cells)
            if not candidates:
                return None
            chosen = random.choice(candidates)
            occupied |= chosen
            cluster_cells |= chosen
            placements.append(chosen)
    return placements


def _all_residues_visible(h: int, w: int, pr: int, pc: int, defect_mask: np.ndarray) -> bool:
    for rr in range(pr):
        for cc in range(pc):
            seen = False
            for r in range(rr, h, pr):
                for c in range(cc, w, pc):
                    if not defect_mask[r, c]:
                        seen = True
                        break
                if seen:
                    break
            if not seen:
                return False
    return True


def _dilate8(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    out = mask.copy()
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = True
    return out


def _place_patch_random(grid: np.ndarray, patch: np.ndarray) -> Coord | None:
    h, w = grid.shape
    ph, pw = patch.shape
    coords = np.argwhere(patch != BG)
    if coords.size == 0:
        return None

    forbidden = _dilate8(grid != BG)
    positions = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
    random.shuffle(positions)
    for r0, c0 in positions:
        ok = True
        for dr, dc in coords:
            if forbidden[r0 + int(dr), c0 + int(dc)]:
                ok = False
                break
        if ok:
            return int(r0), int(c0)
    return None


def _build_sprite_patch(ideal: np.ndarray, cells: set[Coord]) -> np.ndarray:
    coords = np.array(sorted(cells), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    for r, c in coords:
        patch[int(r) - r0, int(c) - c0] = int(ideal[int(r), int(c)])
    return patch


def _gen_attempt() -> np.ndarray | None:
    tile_info = _random_tile()
    if tile_info is None:
        return None
    tile, pr, pc = tile_info

    h_lo = max(2 * pr + 2, 8)
    w_lo = max(2 * pc + 2, 8)
    h_hi = min(TEXTURE_H_MAX, 4 * pr + 3)
    w_hi = min(TEXTURE_W_MAX, 4 * pc + 4)
    if h_lo > h_hi or w_lo > w_hi:
        return None
    tex_h = random.randint(h_lo, h_hi)
    tex_w = random.randint(w_lo, w_hi)
    ideal_texture = _repeat_tile(tile, tex_h, tex_w)

    sprite_count = random.randint(*SPRITE_COUNT_RANGE)
    sprite_areas = _gen_sprite_areas(sprite_count)
    if sprite_areas is None:
        return None
    shapes = []
    for area in sprite_areas:
        shape = _random_polyomino(area)
        if shape is None:
            return None
        shapes.append(shape)

    if sprite_count >= 4 and random.random() < 0.55:
        cluster_count = random.randint(1, min(3, sprite_count - 1))
    else:
        cluster_count = sprite_count
    cluster_sizes = _random_composition(sprite_count, cluster_count)

    placements = _place_defect_sprites(tex_h, tex_w, shapes, cluster_sizes)
    if placements is None:
        return None

    defect_mask = np.zeros((tex_h, tex_w), dtype=bool)
    for cells in placements:
        for r, c in cells:
            defect_mask[r, c] = True

    if len(components_mask(defect_mask, connectivity=4)) != len(cluster_sizes):
        return None
    remaining = ~defect_mask
    if len(components_mask(remaining, connectivity=4)) != 1:
        return None
    if not _all_residues_visible(tex_h, tex_w, pr, pc, defect_mask):
        return None

    grid_h = random.randint(max(GRID_H_RANGE[0], tex_h + 6), min(GRID_H_RANGE[1], 30))
    grid_w = random.randint(max(w_lo + 6, tex_w + 6), min(GRID_W_RANGE[1], 30))
    if grid_h <= tex_h + 1 or grid_w <= tex_w + 1:
        return None
    top = random.randint(1, grid_h - tex_h - 1)
    left = random.randint(1, grid_w - tex_w - 1)

    out = np.full((grid_h, grid_w), BG, dtype=int)
    out[top : top + tex_h, left : left + tex_w] = ideal_texture

    inp = out.copy()
    for r, c in np.argwhere(defect_mask):
        inp[top + int(r), left + int(c)] = BG

    sprite_patches = [_build_sprite_patch(ideal_texture, cells) for cells in placements]
    sprite_patches.sort(key=lambda patch: (-np.count_nonzero(patch != BG), -(patch.shape[0] * patch.shape[1])))
    for patch in sprite_patches:
        pos = _place_patch_random(inp, patch)
        if pos is None:
            return None
        paste(inp, patch, pos[0], pos[1], transparent=BG)

    if len(_component_records(inp, BG)) != 1 + len(sprite_patches):
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 036/1")
