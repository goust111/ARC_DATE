from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
BG = 0
MAX_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(np.asarray(grid, dtype=int))))
    if exclude is not None:
        colors -= {int(c) for c in exclude}
    return sorted(colors)


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.asarray(grid)[r0 : r1 + 1, c0 : c1 + 1].copy()


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = DIR4 if int(connectivity) == 4 else DIR8
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in dirs:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))

    return components


def _expanded_slice(r0: int, c0: int, h: int, w: int, shape: tuple[int, int], pad: int) -> tuple[slice, slice]:
    hh, ww = shape
    rr0 = max(0, int(r0) - int(pad))
    cc0 = max(0, int(c0) - int(pad))
    rr1 = min(hh, int(r0) + int(h) + int(pad))
    cc1 = min(ww, int(c0) + int(w) + int(pad))
    return slice(rr0, rr1), slice(cc0, cc1)


def _region_free(reserved: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> bool:
    rs, cs = _expanded_slice(r0, c0, h, w, reserved.shape, pad)
    return not bool(reserved[rs, cs].any())


def _reserve_region(reserved: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> None:
    rs, cs = _expanded_slice(r0, c0, h, w, reserved.shape, pad)
    reserved[rs, cs] = True


def _candidate_positions(grid_shape: tuple[int, int], h: int, w: int) -> list[tuple[int, int]]:
    gh, gw = grid_shape
    return [(r, c) for r in range(gh - h + 1) for c in range(gw - w + 1)]


def _stamp_patch(shape: tuple[int, int], local_coords: list[tuple[int, int]], color: int, bg: int) -> np.ndarray:
    patch = np.full(shape, int(bg), dtype=int)
    for r, c in local_coords:
        patch[int(r), int(c)] = int(color)
    return patch


def _random_hole_mask(h: int, w: int) -> np.ndarray:
    center = (h // 2, w // 2)
    for _ in range(80):
        mask = {center}
        targets = [
            (0, random.randrange(w)),
            (h - 1, random.randrange(w)),
            (random.randrange(h), 0),
            (random.randrange(h), w - 1),
        ]
        for target in targets:
            rr, cc = center
            while (rr, cc) != target:
                moves: list[tuple[int, int]] = []
                if rr != target[0]:
                    moves.append((rr + (1 if target[0] > rr else -1), cc))
                if cc != target[1]:
                    moves.append((rr, cc + (1 if target[1] > cc else -1)))
                rr, cc = random.choice(moves)
                mask.add((int(rr), int(cc)))

        frontier = {
            (rr + dr, cc + dc)
            for rr, cc in mask
            for dr, dc in DIR4
            if 0 <= rr + dr < h and 0 <= cc + dc < w and (rr + dr, cc + dc) not in mask
        }
        max_area = h * w - 1
        min_target = min(max_area, len(mask) + 1)
        max_target = min(max_area, len(mask) + random.randint(1, max(1, h + w)))
        target_area = random.randint(min_target, max_target)
        while frontier and len(mask) < target_area:
            cell = random.choice(tuple(frontier))
            mask.add(cell)
            frontier.remove(cell)
            rr, cc = cell
            for dr, dc in DIR4:
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in mask:
                    frontier.add((nr, nc))

        arr = np.zeros((h, w), dtype=bool)
        for rr, cc in mask:
            arr[int(rr), int(cc)] = True
        coords = np.argwhere(arr)
        if len(coords) == 0:
            continue
        if bbox_coords(coords) != (0, 0, h - 1, w - 1):
            continue
        if int(arr.sum()) >= h * w:
            continue
        if not bool(arr[center]):
            continue
        if len(components_mask(arr, connectivity=4)) != 1:
            continue
        return arr
    raise RuntimeError("failed to sample a connected odd-bbox hole mask")


def _build_holey_block(color: int, bg: int) -> dict:
    for _ in range(80):
        hh = random.choice([3, 5])
        ww = random.choice([3, 5])
        hole_mask = _random_hole_mask(hh, ww)
        top = random.randint(1, 2)
        bottom = random.randint(1, 2)
        left = random.randint(1, 2)
        right = random.randint(1, 2)
        bh = hh + top + bottom
        bw = ww + left + right
        patch = np.full((bh, bw), int(color), dtype=int)
        patch[top : top + hh, left : left + ww][hole_mask] = int(bg)
        if len(components_mask(patch == int(color), connectivity=4)) != 1:
            continue
        local_coords = [tuple(map(int, rc)) for rc in np.argwhere(hole_mask)]
        rel_coords = [(int(r - hh // 2), int(c - ww // 2)) for r, c in local_coords]
        return {
            "color": int(color),
            "patch": patch,
            "hole_local": local_coords,
            "hole_rel": rel_coords,
            "hole_shape": (int(hh), int(ww)),
        }
    raise RuntimeError("failed to build a connected carved block")


def _build_solid_block(color: int) -> dict:
    bh = random.randint(4, 7)
    bw = random.randint(4, 7)
    patch = np.full((bh, bw), int(color), dtype=int)
    return {"color": int(color), "patch": patch, "hole_local": None, "hole_rel": None, "hole_shape": None}


def _place_patch(grid: np.ndarray, reserved: np.ndarray, patch: np.ndarray, pad: int) -> tuple[int, int] | None:
    h, w = patch.shape
    positions = _candidate_positions(grid.shape, h, w)
    random.shuffle(positions)
    for r0, c0 in positions:
        if not _region_free(reserved, r0, c0, h, w, pad):
            continue
        mask = patch != BG
        grid[r0 : r0 + h, c0 : c0 + w][mask] = patch[mask]
        _reserve_region(reserved, r0, c0, h, w, pad)
        return int(r0), int(c0)
    return None


def _place_marker(grid: np.ndarray, reserved: np.ndarray, color: int, bbox_shape: tuple[int, int], pad: int) -> tuple[int, int] | None:
    h, w = bbox_shape
    positions = _candidate_positions(grid.shape, h, w)
    random.shuffle(positions)
    center_r = h // 2
    center_c = w // 2
    for r0, c0 in positions:
        if not _region_free(reserved, r0, c0, h, w, pad):
            continue
        mr = int(r0 + center_r)
        mc = int(c0 + center_c)
        if grid[mr, mc] != BG:
            continue
        grid[mr, mc] = int(color)
        _reserve_region(reserved, r0, c0, h, w, pad)
        return int(mr), int(mc)
    return None


def _parse_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    blocks: dict[int, dict] = {}
    markers: dict[int, list[tuple[int, int]]] = defaultdict(list)

    for color in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(color), connectivity=4):
            area = int(coords.shape[0])
            if area == 1:
                r, c = map(int, coords[0])
                for dr, dc in DIR8:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and g[nr, nc] != bg:
                        raise AssertionError("marker is not isolated")
                markers[int(color)].append((int(r), int(c)))
                continue

            bbox = bbox_coords(coords)
            patch = crop_bbox(g, bbox)
            if not np.isin(patch, [bg, color]).all():
                raise AssertionError("block bbox contains unexpected color")
            if area != int(np.count_nonzero(patch == color)):
                raise AssertionError("block bbox contains detached same-color islands")
            if not (
                np.all(patch[0, :] == color)
                and np.all(patch[-1, :] == color)
                and np.all(patch[:, 0] == color)
                and np.all(patch[:, -1] == color)
            ):
                raise AssertionError("block is not a solid-border rectangle")
            hole_components = components_mask(patch == bg, connectivity=4)
            if len(hole_components) > 1:
                raise AssertionError("block contains multiple hole components")
            hole = None
            if hole_components:
                hole_coords = hole_components[0]
                hb = bbox_coords(hole_coords)
                hh = int(hb[2] - hb[0] + 1)
                ww = int(hb[3] - hb[1] + 1)
                if hh % 2 == 0 or ww % 2 == 0:
                    raise AssertionError("hole bbox must have odd side lengths")
                local_coords = [(int(r - hb[0]), int(c - hb[1])) for r, c in hole_coords]
                rel_coords = [(int(r - hh // 2), int(c - ww // 2)) for r, c in local_coords]
                hole = {
                    "bbox": hb,
                    "shape": (int(hh), int(ww)),
                    "local_coords": sorted(local_coords),
                    "rel_coords": sorted(rel_coords),
                }
            if int(color) in blocks:
                raise AssertionError("multiple blocks share a color")
            blocks[int(color)] = {"color": int(color), "bbox": bbox, "hole": hole}

    if not blocks:
        raise AssertionError("no block detected")
    return {"bg": int(bg), "blocks": blocks, "markers": {k: sorted(v) for k, v in markers.items()}}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract each carved-out hole pattern from its rectangle and stamp it positively at same-color markers.

    Input structure:
    - The background is the dominant color.
    - Several distinct colors form rectangular blocks; some are solid, while others contain one enclosed background-colored hole pattern.
    - Isolated single-pixel markers lie on the background; some share a color with a block and some do not.

    Rule:
    - Restore every block to a full solid rectangle by filling any enclosed hole with the block color.
    - Erase every isolated marker pixel.
    - For each block that had a hole, stamp that hole pattern in positive form at every same-color marker, centered on the marker cell and using the block color.
    - Markers whose color has no block, or only matches a holeless solid block, stay erased with no stamp.

    Key insight:
    - The missing background shape inside a block is the template: convert that negative hole into a positive same-color stamp elsewhere.
    """
    g = np.asarray(grid, dtype=int)
    info = _parse_grid(g)
    bg = int(info["bg"])
    out = g.copy()

    for marker_positions in info["markers"].values():
        for r, c in marker_positions:
            out[int(r), int(c)] = int(bg)

    for block in info["blocks"].values():
        r0, c0, r1, c1 = block["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(block["color"])

    for color, block in info["blocks"].items():
        hole = block["hole"]
        if hole is None:
            continue
        for mr, mc in info["markers"].get(int(color), []):
            for dr, dc in hole["rel_coords"]:
                rr = int(mr + dr)
                cc = int(mc + dc)
                if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                    out[rr, cc] = int(color)

    return out


def _semantic_check(grid: np.ndarray) -> bool:
    info = _parse_grid(grid)
    bg = int(info["bg"])
    blocks = info["blocks"]
    markers = info["markers"]

    holey_colors = [color for color, block in blocks.items() if block["hole"] is not None]
    solid_colors = [color for color, block in blocks.items() if block["hole"] is None]
    unmatched_colors = [color for color in markers if color not in blocks]
    if not holey_colors or not solid_colors or not unmatched_colors:
        return False

    for color in blocks:
        if len(markers.get(color, [])) != 1:
            return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False

    for block in blocks.values():
        r0, c0, r1, c1 = block["bbox"]
        if not np.all(out[r0 : r1 + 1, c0 : c1 + 1] == int(block["color"])):
            return False

    for color in holey_colors:
        hole = blocks[color]["hole"]
        assert hole is not None
        sh, sw = hole["shape"]
        marker = markers[color][0]
        mr, mc = marker
        r0 = int(mr - sh // 2)
        c0 = int(mc - sw // 2)
        if r0 < 0 or c0 < 0 or r0 + sh > grid.shape[0] or c0 + sw > grid.shape[1]:
            return False
        input_patch = grid[r0 : r0 + sh, c0 : c0 + sw]
        center_mask = np.ones((sh, sw), dtype=bool)
        center_mask[sh // 2, sw // 2] = False
        if not np.all(input_patch[center_mask] == bg):
            return False
        expected = _stamp_patch((sh, sw), hole["local_coords"], int(color), bg)
        if not np.array_equal(out[r0 : r0 + sh, c0 : c0 + sw], expected):
            return False

    for color in solid_colors + unmatched_colors:
        mr, mc = markers[color][0]
        if out[mr, mc] != bg:
            return False

    return True


def _generate_once() -> np.ndarray | None:
    h = random.randint(22, 30)
    w = random.randint(22, 30)
    grid = np.full((h, w), BG, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    n_holey = random.randint(2, 4)
    n_solid = random.randint(1, 2)
    n_unmatched = random.randint(1, 2)
    total_colors = n_holey + n_solid + n_unmatched
    if total_colors > len(PALETTE):
        return None

    chosen_colors = random.sample(PALETTE, total_colors)
    block_colors = chosen_colors[: n_holey + n_solid]
    unmatched_colors = chosen_colors[n_holey + n_solid :]

    block_specs: list[dict] = []
    for color in block_colors[:n_holey]:
        block_specs.append(_build_holey_block(int(color), BG))
    for color in block_colors[n_holey:]:
        block_specs.append(_build_solid_block(int(color)))
    random.shuffle(block_specs)

    for spec in block_specs:
        placed = _place_patch(grid, reserved, spec["patch"], pad=1)
        if placed is None:
            return None
        spec["top_left"] = placed

    marker_items: list[tuple[int, tuple[int, int]]] = []
    for spec in block_specs:
        if spec["hole_shape"] is None:
            bbox_shape = (1, 1)
        else:
            bbox_shape = spec["hole_shape"]
        marker_items.append((int(spec["color"]), bbox_shape))
    marker_items.extend((int(color), (1, 1)) for color in unmatched_colors)
    random.shuffle(marker_items)

    for color, bbox_shape in marker_items:
        marker = _place_marker(grid, reserved, int(color), bbox_shape, pad=1)
        if marker is None:
            return None

    if not _semantic_check(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid negative-stamp input")
