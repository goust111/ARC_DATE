from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]
Direction = tuple[int, int]
BBox = tuple[int, int, int, int]


DIRECTIONS4: tuple[Direction, ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))

# --------------- lightweight ARC utils (copied/adapted from arc_utils contracts) ---------------


def get_bg(grid: np.ndarray) -> int:
    """Background color = most common value in the grid."""
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    cols = set(map(int, grid.ravel()))
    if exclude is not None:
        cols -= set(map(int, exclude))
    return sorted(cols)


def neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: BBox) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    """4-connected components of a boolean mask, each as an (N,2) coords array."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q: deque[Coord] = deque([(r, c)])
            pts: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _component_cells(coords: np.ndarray) -> set[Coord]:
    return {tuple(map(int, rc)) for rc in coords}


def _color_components(grid: np.ndarray, color: int) -> list[np.ndarray]:
    return components_mask(grid == int(color))


def _multicolor_components(grid: np.ndarray, allowed_colors: set[int]) -> list[np.ndarray]:
    if not allowed_colors:
        return []
    return components_mask(np.isin(grid, sorted(allowed_colors)))


def _is_filled_straight_bar(coords: np.ndarray) -> str | None:
    r0, c0, r1, c1 = bbox_coords(coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if int(coords.shape[0]) != h * w:
        return None
    if h == 1 and w >= 2:
        return "h"
    if w == 1 and h >= 2:
        return "v"
    return None


def _is_filled_odd_square(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = bbox_coords(coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    return h == w and h >= 3 and h % 2 == 1 and int(coords.shape[0]) == h * w


def _detect_base_color(grid: np.ndarray, bg: int) -> tuple[int, list[np.ndarray]]:
    h, w = grid.shape
    candidates: list[tuple[int, list[np.ndarray]]] = []
    for color in colors_present(grid, exclude={bg}):
        comps = _color_components(grid, color)
        if not comps:
            continue
        ok = True
        for coords in comps:
            orient = _is_filled_straight_bar(coords)
            if orient is None:
                ok = False
                break
            dirs: set[Direction] = set()
            for r, c in coords:
                rr = int(r)
                cc = int(c)
                for dr, dc in DIRECTIONS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        cell = int(grid[nr, nc])
                        if cell != bg and cell != color:
                            dirs.add((dr, dc))
            if len(dirs) != 1:
                ok = False
                break
            dr, dc = next(iter(dirs))
            r0, c0, r1, c1 = bbox_coords(coords)
            if orient == "v":
                if dr != 0:
                    ok = False
                    break
                adj_c = c0 + dc
                if not (0 <= adj_c < w):
                    ok = False
                    break
                if any(int(grid[rr, adj_c]) in (bg, color) for rr in range(r0, r1 + 1)):
                    ok = False
                    break
            else:
                if dc != 0:
                    ok = False
                    break
                adj_r = r0 + dr
                if not (0 <= adj_r < h):
                    ok = False
                    break
                if any(int(grid[adj_r, cc]) in (bg, color) for cc in range(c0, c1 + 1)):
                    ok = False
                    break
        if ok:
            candidates.append((int(color), comps))
    assert len(candidates) == 1, "expected a unique base-bar color"
    return candidates[0]


def _detect_occluder_color(grid: np.ndarray, bg: int, base_color: int) -> tuple[int, list[np.ndarray]]:
    candidates: list[tuple[int, list[np.ndarray]]] = []
    for color in colors_present(grid, exclude={bg, base_color}):
        comps = _color_components(grid, color)
        if len(comps) < 2:
            continue
        if all(_is_filled_odd_square(coords) for coords in comps):
            candidates.append((int(color), comps))
    assert len(candidates) == 1, "expected a unique odd-square occluder color"
    return candidates[0]


def _extract_pattern_patches(
    grid: np.ndarray,
    bg: int,
    base_color: int,
    occluder_color: int,
    base_components: list[np.ndarray],
) -> list[dict]:
    h, w = grid.shape
    base_id = np.full((h, w), -1, dtype=int)
    for idx, coords in enumerate(base_components):
        rows, cols = np.asarray(coords, dtype=int).T
        base_id[rows, cols] = int(idx)

    patch_colors = set(colors_present(grid, exclude={bg, base_color, occluder_color}))
    comps = _multicolor_components(grid, patch_colors)
    assert comps, "expected at least one multi-color pattern patch"

    patches: list[dict] = []
    attached_base_ids: set[int] = set()
    for coords in comps:
        r0, c0, r1, c1 = bbox_coords(coords)
        tile = crop_bbox(grid, (r0, c0, r1, c1))
        h0, w0 = tile.shape
        assert int(coords.shape[0]) == int(h0 * w0), "pattern patch bbox must be fully filled"
        assert len(set(map(int, tile.ravel()))) >= 2, "pattern patch must be multi-color"

        dirs: set[Direction] = set()
        contact_base_ids: set[int] = set()
        for r, c in coords:
            rr = int(r)
            cc = int(c)
            for dr, dc in DIRECTIONS4:
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == base_color:
                    dirs.add((-dr, -dc))
                    contact_base_ids.add(int(base_id[nr, nc]))
        assert len(dirs) == 1, "patch must attach to base on exactly one side"
        assert len(contact_base_ids) == 1, "patch must touch exactly one base bar"

        direction = next(iter(dirs))
        base_idx = next(iter(contact_base_ids))
        assert base_idx not in attached_base_ids, "each base bar must anchor one patch"
        attached_base_ids.add(base_idx)

        if direction == (0, 1):
            assert c0 > 0 and all(int(grid[rr, c0 - 1]) == base_color for rr in range(r0, r1 + 1))
        elif direction == (0, -1):
            assert c1 + 1 < w and all(int(grid[rr, c1 + 1]) == base_color for rr in range(r0, r1 + 1))
        elif direction == (1, 0):
            assert r0 > 0 and all(int(grid[r0 - 1, cc]) == base_color for cc in range(c0, c1 + 1))
        else:
            assert r1 + 1 < h and all(int(grid[r1 + 1, cc]) == base_color for cc in range(c0, c1 + 1))

        patches.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "tile": tile,
                "dir": direction,
                "base_idx": int(base_idx),
            }
        )

    assert len(attached_base_ids) == len(base_components), "every base bar must anchor one patch"
    return patches


def _virtual_map_for_patch(patch: dict, shape: tuple[int, int]) -> dict[Coord, int]:
    h, w = map(int, shape)
    r0, c0, _r1, _c1 = map(int, patch["bbox"])
    tile = np.asarray(patch["tile"], dtype=int)
    th, tw = tile.shape
    dr, dc = map(int, patch["dir"])
    step_r = dr * th
    step_c = dc * tw
    out: dict[Coord, int] = {}
    k = 1
    while True:
        sr = int(r0 + k * step_r)
        sc = int(c0 + k * step_c)
        if sr >= h or sc >= w or sr + th - 1 < 0 or sc + tw - 1 < 0:
            break
        for tr in range(th):
            for tc in range(tw):
                rr = int(sr + tr)
                cc = int(sc + tc)
                if 0 <= rr < h and 0 <= cc < w:
                    out[(rr, cc)] = int(tile[tr, tc])
        k += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each attached pattern block periodically away from its base bar, letting hidden phase pass through gray blockers and recoloring every hit blocker by the invisible center sample.

    Input structure:
    - The background is the most common color.
    - One foreground color forms several short solid base bars; each bar touches exactly one small multi-color rectangular pattern block on one side.
    - Another foreground color forms odd-side filled square occluders, some on the extension strips and some off them.

    Rule:
    - For each pattern block, use the side touching its base bar to determine the extension direction, then repeat the whole block as one period all the way to the grid edge.
    - The repeated pattern is only painted onto background cells, so the original bases, original pattern blocks, and gray occluders are not directly overwritten by the strip.
    - Phase still advances underneath occluders; every occluder whose area is crossed by a strip is recolored as one solid block to the pattern color that would lie at that occluder's center cell in the hidden layer.

    Key insight:
    The gray squares are probes into an invisible periodic layer: you must compute the full hidden phase through the blocked region, not restart the motif after the occluder.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = int(get_bg(g))
    base_color, base_components = _detect_base_color(g, bg)
    occluder_color, occluder_components = _detect_occluder_color(g, bg, base_color)
    patches = _extract_pattern_patches(g, bg, base_color, occluder_color, base_components)

    global_virtual: dict[Coord, int] = {}
    for patch in patches:
        vmap = _virtual_map_for_patch(patch, g.shape)
        patch["virtual"] = vmap
        for coord, color in vmap.items():
            previous = global_virtual.get(coord)
            assert previous is None or previous == int(color), "virtual strips must not conflict"
            global_virtual[coord] = int(color)

    out = g.copy()
    for (r, c), color in global_virtual.items():
        if int(out[r, c]) == bg:
            out[r, c] = int(color)

    for coords in occluder_components:
        cells = _component_cells(coords)
        if not any(cell in global_virtual for cell in cells):
            continue
        r0, c0, r1, c1 = bbox_coords(coords)
        side = int(r1 - r0 + 1)
        center = (int(r0 + side // 2), int(c0 + side // 2))
        assert center in global_virtual, "a crossed occluder must have its center inside the hidden strip"
        sample_color = int(global_virtual[center])
        for r, c in cells:
            out[r, c] = sample_color

    return out


# --------------- generator ---------------

GEN_RETRIES = 400
BG_COLOR = 0
OCCLUDER_COLOR = 5
ALL_COLORS = tuple(range(10))
GRID_MIN = 18
GRID_MAX = 29
SYSTEM_COUNT_CHOICES = (2, 3)
TILE_CROSS_CHOICES = (2, 3)
TILE_ALONG_CHOICES = (2, 3, 4)
HIT_OCCLUDER_SIDES = (3, 5)
MISS_OCCLUDER_SIDES = (3, 5)
SYSTEM_PLACE_TRIES = 120
OCCLUDER_PLACE_TRIES = 200
MAX_BAND_RADIUS = 2  # allows 5x5 occluders to extend beyond the strip safely


def _square_cells(center: Coord, side: int, shape: tuple[int, int]) -> list[Coord] | None:
    h, w = map(int, shape)
    half = side // 2
    r, c = map(int, center)
    r0 = r - half
    c0 = c - half
    r1 = r + half
    c1 = c + half
    if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
        return None
    return [(rr, cc) for rr in range(r0, r1 + 1) for cc in range(c0, c1 + 1)]


def _cyclic_shift_invariant(tile: np.ndarray, axis: int, shift: int) -> bool:
    return bool(np.array_equal(tile, np.roll(tile, -int(shift), axis=int(axis))))


def _tile_has_full_cyclic_period(tile: np.ndarray, axis: int) -> bool:
    span = int(tile.shape[axis])
    return all(not _cyclic_shift_invariant(tile, axis, shift) for shift in range(1, span))


def _bad_tile_monochrome_component(tile: np.ndarray) -> bool:
    for color in sorted(set(map(int, tile.ravel()))):
        for coords in _color_components(tile, color):
            if _is_filled_straight_bar(coords) is not None:
                return True
            if _is_filled_odd_square(coords):
                return True
    return False


def _make_tile(height: int, width: int, direction: Direction, palette: list[int]) -> np.ndarray | None:
    axis = 1 if int(direction[1]) != 0 else 0
    n_colors = random.randint(2, min(4, len(palette)))
    for _ in range(150):
        colors = random.sample(palette, n_colors)
        tile = np.array(
            [[random.choice(colors) for _ in range(width)] for _ in range(height)],
            dtype=int,
        )
        if len(set(map(int, tile.ravel()))) < 2:
            continue
        if not _tile_has_full_cyclic_period(tile, axis):
            continue
        if _bad_tile_monochrome_component(tile):
            continue
        return tile
    return None


def _system_virtual_and_body(system: dict, shape: tuple[int, int]) -> tuple[dict[Coord, int], set[Coord]]:
    virtual = _virtual_map_for_patch(system, shape)
    r0, c0, r1, c1 = map(int, system["bbox"])
    body = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    body |= set(system["base_cells"])
    return virtual, body


def _band_interval(system: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, system["bbox"])
    dr, dc = map(int, system["dir"])
    if dc != 0:
        return int(r0 - MAX_BAND_RADIUS), int(r1 + MAX_BAND_RADIUS)
    return int(c0 - MAX_BAND_RADIUS), int(c1 + MAX_BAND_RADIUS)


def _intervals_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def _place_systems(grid: np.ndarray, base_color: int, family: str, palette: list[int]) -> list[dict] | None:
    h, w = grid.shape
    n_systems = random.choice(SYSTEM_COUNT_CHOICES)
    systems: list[dict] = []
    reserved_body: set[Coord] = set()
    reserved_virtual: set[Coord] = set()
    reserved_bands: list[tuple[int, int]] = []

    for _sys_idx in range(n_systems):
        placed = False
        for _ in range(SYSTEM_PLACE_TRIES):
            if family == "horizontal":
                direction = random.choice(((0, 1), (0, -1)))
                tile_h = random.choice(TILE_CROSS_CHOICES)
                tile_w = random.choice(TILE_ALONG_CHOICES)
            else:
                direction = random.choice(((1, 0), (-1, 0)))
                tile_h = random.choice(TILE_ALONG_CHOICES)
                tile_w = random.choice(TILE_CROSS_CHOICES)

            tile = _make_tile(tile_h, tile_w, direction, palette)
            if tile is None:
                continue

            if family == "horizontal":
                if h - tile_h - 2 * MAX_BAND_RADIUS <= 0:
                    continue
                r0 = random.randint(MAX_BAND_RADIUS, h - tile_h - MAX_BAND_RADIUS)
                interval = (r0 - MAX_BAND_RADIUS, r0 + tile_h - 1 + MAX_BAND_RADIUS)
                if any(_intervals_overlap(interval, other) for other in reserved_bands):
                    continue
                if direction == (0, 1):
                    max_c0 = w - 4 * tile_w
                    if max_c0 < 1:
                        continue
                    c0 = random.randint(1, max_c0)
                    base_cells = [(r0 + rr, c0 - 1) for rr in range(tile_h)]
                else:
                    min_c0 = 3 * tile_w
                    max_c0 = w - tile_w - 1
                    if max_c0 < min_c0:
                        continue
                    c0 = random.randint(min_c0, max_c0)
                    base_cells = [(r0 + rr, c0 + tile_w) for rr in range(tile_h)]
            else:
                if w - tile_w - 2 * MAX_BAND_RADIUS <= 0:
                    continue
                c0 = random.randint(MAX_BAND_RADIUS, w - tile_w - MAX_BAND_RADIUS)
                interval = (c0 - MAX_BAND_RADIUS, c0 + tile_w - 1 + MAX_BAND_RADIUS)
                if any(_intervals_overlap(interval, other) for other in reserved_bands):
                    continue
                if direction == (1, 0):
                    max_r0 = h - 4 * tile_h
                    if max_r0 < 1:
                        continue
                    r0 = random.randint(1, max_r0)
                    base_cells = [(r0 - 1, c0 + cc) for cc in range(tile_w)]
                else:
                    min_r0 = 3 * tile_h
                    max_r0 = h - tile_h - 1
                    if max_r0 < min_r0:
                        continue
                    r0 = random.randint(min_r0, max_r0)
                    base_cells = [(r0 + tile_h, c0 + cc) for cc in range(tile_w)]

            bbox = (int(r0), int(c0), int(r0 + tile_h - 1), int(c0 + tile_w - 1))
            system = {
                "bbox": bbox,
                "tile": tile,
                "dir": direction,
                "base_cells": tuple(base_cells),
            }
            virtual, body = _system_virtual_and_body(system, grid.shape)
            if len(virtual) < tile_h * tile_w * 2:
                continue
            if body & reserved_body:
                continue
            if body & reserved_virtual:
                continue
            if set(virtual) & reserved_body:
                continue
            if set(virtual) & reserved_virtual:
                continue

            rows = np.arange(r0, r0 + tile_h)
            cols = np.arange(c0, c0 + tile_w)
            for rr in rows:
                for cc in cols:
                    grid[int(rr), int(cc)] = int(tile[int(rr - r0), int(cc - c0)])
            for rr, cc in base_cells:
                grid[int(rr), int(cc)] = int(base_color)

            system["virtual"] = virtual
            system["body_cells"] = body
            system["period"] = int(tile_w if direction[1] != 0 else tile_h)
            systems.append(system)
            reserved_body |= body
            reserved_virtual |= set(virtual)
            reserved_bands.append(interval)
            placed = True
            break
        if not placed:
            return None

    return systems


def _global_virtual_map(systems: list[dict]) -> dict[Coord, tuple[int, int]]:
    merged: dict[Coord, tuple[int, int]] = {}
    for sys_idx, system in enumerate(systems):
        for coord, color in system["virtual"].items():
            prev = merged.get(coord)
            if prev is not None and prev[0] != int(color):
                raise AssertionError("scene generator built conflicting strips")
            merged[coord] = (int(color), int(sys_idx))
    return merged


def _beyond_square(coord: Coord, center: Coord, side: int, direction: Direction) -> bool:
    r, c = coord
    cr, cc = center
    half = side // 2
    dr, dc = direction
    if dr == 0 and dc == 1:
        return c > cc + half
    if dr == 0 and dc == -1:
        return c < cc - half
    if dr == 1 and dc == 0:
        return r > cr + half
    return r < cr - half


def _phase_sensitive_hit(system: dict, center: Coord, side: int) -> bool:
    dr, dc = map(int, system["dir"])
    if side % int(system["period"]) == 0:
        return False
    for coord, color in system["virtual"].items():
        if not _beyond_square(coord, center, side, system["dir"]):
            continue
        back = (int(coord[0] - dr * side), int(coord[1] - dc * side))
        if back in system["virtual"] and int(system["virtual"][back]) != int(color):
            return True
    return False


def _place_hit_occluder(
    grid: np.ndarray,
    system: dict,
    existing_occ: set[Coord],
    require_phase_sensitive: bool,
) -> dict | None:
    centers = list(system["virtual"])
    random.shuffle(centers)
    sizes = list(HIT_OCCLUDER_SIDES)
    random.shuffle(sizes)
    if require_phase_sensitive:
        sizes.sort(key=lambda side: int(side % int(system["period"]) == 0))

    for side in sizes:
        for center in centers:
            cells = _square_cells(center, side, grid.shape)
            if cells is None:
                continue
            if any(tuple(cell) in existing_occ for cell in cells):
                continue
            if any(int(grid[r, c]) != BG_COLOR for r, c in cells):
                continue
            if require_phase_sensitive and not _phase_sensitive_hit(system, center, side):
                continue
            for r, c in cells:
                grid[r, c] = OCCLUDER_COLOR
            sample_color = int(system["virtual"][center])
            return {
                "cells": tuple(cells),
                "center": tuple(map(int, center)),
                "side": int(side),
                "hit": True,
                "sample_color": sample_color,
            }
    return None


def _place_miss_occluder(
    grid: np.ndarray,
    global_virtual: dict[Coord, tuple[int, int]],
    existing_occ: set[Coord],
) -> dict | None:
    sizes = list(MISS_OCCLUDER_SIDES)
    for _ in range(OCCLUDER_PLACE_TRIES):
        side = random.choice(sizes)
        half = side // 2
        center = (
            random.randint(half, grid.shape[0] - 1 - half),
            random.randint(half, grid.shape[1] - 1 - half),
        )
        cells = _square_cells(center, side, grid.shape)
        assert cells is not None
        if any(cell in existing_occ for cell in cells):
            continue
        if any(cell in global_virtual for cell in cells):
            continue
        if any(int(grid[r, c]) != BG_COLOR for r, c in cells):
            continue
        for r, c in cells:
            grid[r, c] = OCCLUDER_COLOR
        return {
            "cells": tuple(cells),
            "center": tuple(map(int, center)),
            "side": int(side),
            "hit": False,
            "sample_color": None,
        }
    return None


def _expected_output(grid: np.ndarray, systems: list[dict], occluders: list[dict]) -> np.ndarray:
    out = grid.copy()
    global_virtual = _global_virtual_map(systems)
    for (r, c), (color, _sys_idx) in global_virtual.items():
        if int(out[r, c]) == BG_COLOR:
            out[r, c] = int(color)
    for occ in occluders:
        if not bool(occ["hit"]):
            continue
        color = int(occ["sample_color"])
        for r, c in occ["cells"]:
            out[int(r), int(c)] = color
    return out


def _scene_semantic_checks(grid: np.ndarray, systems: list[dict], occluders: list[dict]) -> bool:
    if len(systems) < 2:
        return False
    if sum(bool(occ["hit"]) for occ in occluders) < 1:
        return False
    if sum(not bool(occ["hit"]) for occ in occluders) < 1:
        return False
    bg_count = int(np.count_nonzero(grid == BG_COLOR))
    if bg_count <= grid.size // 2:
        return False

    global_virtual = _global_virtual_map(systems)
    visible_extension = sum(1 for coord in global_virtual if int(grid[coord]) == BG_COLOR)
    if visible_extension < 10:
        return False

    phase_hits = 0
    for occ in occluders:
        if not bool(occ["hit"]):
            continue
        center = tuple(map(int, occ["center"]))
        side = int(occ["side"])
        sample = occ["sample_color"]
        if sample is None:
            return False
        sys_idx = global_virtual[center][1]
        if _phase_sensitive_hit(systems[sys_idx], center, side):
            phase_hits += 1
    if phase_hits < 1:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    grid = np.full((h, w), BG_COLOR, dtype=int)

    base_color = random.choice([c for c in ALL_COLORS if c not in {BG_COLOR, OCCLUDER_COLOR}])
    tile_palette = [c for c in ALL_COLORS if c not in {BG_COLOR, OCCLUDER_COLOR, base_color}]
    family = random.choice(("horizontal", "vertical"))

    systems = _place_systems(grid, base_color, family, tile_palette)
    if systems is None:
        return None

    global_virtual = _global_virtual_map(systems)
    occluders: list[dict] = []
    occupied_occ: set[Coord] = set()

    hit_systems = list(range(len(systems)))
    random.shuffle(hit_systems)
    hit_count = random.randint(1, min(2, len(systems)))
    hit_systems = hit_systems[:hit_count]
    phase_done = False
    for sys_idx in hit_systems:
        occ = _place_hit_occluder(grid, systems[sys_idx], occupied_occ, require_phase_sensitive=not phase_done)
        if occ is None:
            if not phase_done:
                return None
            continue
        phase_done = phase_done or _phase_sensitive_hit(systems[sys_idx], occ["center"], occ["side"])
        occluders.append(occ)
        occupied_occ |= set(occ["cells"])

    if not phase_done:
        return None

    miss_count = random.randint(1, 2)
    for _ in range(miss_count):
        occ = _place_miss_occluder(grid, global_virtual, occupied_occ)
        if occ is None:
            return None
        occluders.append(occ)
        occupied_occ |= set(occ["cells"])

    expected = _expected_output(grid, systems, occluders)
    if np.array_equal(expected, grid):
        return None
    if not _scene_semantic_checks(grid, systems, occluders):
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
