from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
CORNER_NAMES = ("tl", "tr", "bl", "br")


def _corners(shape: tuple[int, int]) -> dict[str, tuple[int, int]]:
    h, w = shape
    return {
        "tl": (0, 0),
        "tr": (0, w - 1),
        "bl": (h - 1, 0),
        "br": (h - 1, w - 1),
    }


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _parse_grid(grid: np.ndarray) -> tuple[np.ndarray, dict[str, int], dict[int, str], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert 1 < h <= 30 and 1 < w <= 30

    corner_pos = _corners(g.shape)
    corner_colors = {name: int(g[r, c]) for name, (r, c) in corner_pos.items()}
    assert all(color != BG for color in corner_colors.values())
    assert len(set(corner_colors.values())) == 4
    color_to_corner = {color: name for name, color in corner_colors.items()}

    sprite_mask = g != BG
    for r, c in corner_pos.values():
        sprite_mask[r, c] = False

    comps = _components(sprite_mask)
    assert 3 <= len(comps) <= 4

    sprites: list[dict] = []
    body_colors_seen: set[int] = set()
    for coords in comps:
        r0, c0, r1, c1 = _bbox(coords)
        bh = r1 - r0 + 1
        bw = c1 - c0 + 1
        assert 2 <= bh <= 5
        assert 3 <= bw <= 6
        assert r0 > 0 and c0 > 0 and r1 < h - 1 and c1 < w - 1

        local = coords - np.array([r0, c0], dtype=int)
        cells = {(int(r), int(c)) for r, c in local}
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        marker_locs = [(int(r), int(c)) for r, c in local if int(patch[int(r), int(c)]) in color_to_corner]
        assert len(marker_locs) == 1
        marker_local = marker_locs[0]
        marker_color = int(patch[marker_local])

        body_colors = {int(patch[r, c]) for r, c in cells if (r, c) != marker_local}
        assert BG not in body_colors
        assert len(body_colors) == 1
        body_color = next(iter(body_colors))
        assert body_color not in color_to_corner
        assert body_color not in body_colors_seen
        body_colors_seen.add(body_color)

        assert any(c == 0 for _r, c in cells)
        assert any(c == 1 for _r, c in cells)
        assert any(c >= 2 for _r, c in cells)

        mask = np.zeros((bh, bw), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        sprites.append(
            {
                "bbox": (r0, c0, r1, c1),
                "sort_key": (r0, c0),
                "body_color": body_color,
                "marker_color": marker_color,
                "target_corner": color_to_corner[marker_color],
                "local_cells": cells,
                "mask": mask,
                "marker_local": marker_local,
                "shape": (bh, bw),
                "abs_coords": coords,
            }
        )

    sprites.sort(key=lambda sp: sp["sort_key"])
    return g, corner_colors, color_to_corner, sprites


def _cropped_patch(sprite: dict, *, local_crop: bool, erase_marker: bool) -> np.ndarray:
    bh, bw = sprite["shape"]
    patch = np.zeros((bh, bw), dtype=int)
    patch[sprite["mask"]] = int(sprite["body_color"])
    mr, mc = sprite["marker_local"]
    patch[mr, mc] = int(sprite["marker_color"])
    if local_crop:
        patch[:, :2] = BG
    if erase_marker:
        patch[mr, mc] = BG
    return patch


def _stamp_origin(shape: tuple[int, int], patch_shape: tuple[int, int], corner: str) -> tuple[int, int]:
    h, w = shape
    ph, pw = patch_shape
    return {
        "tl": (0, 0),
        "tr": (0, w - pw),
        "bl": (h - ph, 0),
        "br": (h - ph, w - pw),
    }[corner]


def _transform(
    grid: np.ndarray,
    *,
    local_crop: bool = True,
    erase_marker: bool = True,
    preserve_corner_markers: bool = True,
    reverse_order: bool = False,
) -> np.ndarray:
    g, _corner_colors, _color_to_corner, sprites = _parse_grid(grid)
    out = g.copy()
    for sprite in sprites:
        coords = sprite["abs_coords"]
        out[coords[:, 0], coords[:, 1]] = BG

    corner_cells = set(_corners(g.shape).values())
    order = list(reversed(sprites)) if reverse_order else sprites
    for sprite in order:
        patch = _cropped_patch(sprite, local_crop=local_crop, erase_marker=erase_marker)
        r0, c0 = _stamp_origin(g.shape, patch.shape, sprite["target_corner"])
        for r, c in np.argwhere(patch != BG):
            rr = int(r0 + r)
            cc = int(c0 + c)
            if preserve_corner_markers and (rr, cc) in corner_cells:
                continue
            out[rr, cc] = int(patch[int(r), int(c)])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop each interior sprite locally and stamp the cropped residue into the corner named by its embedded marker color.

    Input structure:
    - Black `0` is the background on a grid up to 30x30.
    - The four grid corners each contain one distinct non-black marker color, defining the four corner targets.
    - The interior contains 3-4 separated 4-connected multicolor sprites; each sprite has one unique body color plus exactly one embedded corner-marker-color pixel, and each sprite stays away from the border.

    Rule:
    - Read the four corner pixels to build the color-to-corner mapping.
    - For each interior sprite, use its own bounding box as the local frame, erase all sprite pixels in the two leftmost bbox columns, then erase the embedded marker pixel itself.
    - Remove the original sprite from the interior, and stamp the remaining bbox-sized residue into the matched global corner so the bbox corner aligned with that target corner coincides with the grid corner.
    - Stamp only non-black cells; if several sprites target the same corner, process them by original top-left position in row-major order so later ones overwrite earlier ones, while the original single-pixel corner markers stay unchanged.

    Key insight:
    - The crop is measured in each sprite's own local bounding box, but the destination is chosen globally by the embedded corner-marker color, so both the local crop and the marker-to-corner grouping are necessary.
    """
    return _transform(grid)


def _random_path(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = a
    dr = 1 if b[0] > r else -1
    dc = 1 if b[1] > c else -1
    steps = [(dr, 0)] * abs(b[0] - r) + [(0, dc)] * abs(b[1] - c)
    random.shuffle(steps)
    path = [(r, c)]
    for sr, sc in steps:
        r += sr
        c += sc
        path.append((r, c))
    return path


def _grow_shape(
    height: int,
    width: int,
    *,
    require_cells: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    cells = set()
    anchors = list(require_cells)
    random.shuffle(anchors)
    current = anchors[0]
    cells.add(current)
    for nxt in anchors[1:]:
        cells.update(_random_path(current, nxt))
        current = nxt
    if len(cells) > min(height * width, 12):
        return None

    target = random.randint(max(5, len(cells)), min(height * width, max(len(cells), 12)))
    while len(cells) < target:
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(frontier))

    if not any(c == 0 for _r, c in cells):
        return None
    if not any(c == 1 for _r, c in cells):
        return None
    survivors = [(r, c) for r, c in cells if c >= 2]
    if len(survivors) < 2:
        return None
    return cells


def _make_sprite_shape(
    corner: str,
    *,
    require_corner_preserve: bool,
    require_overlap_cell: bool,
    marker_outside_crop: bool,
) -> tuple[set[tuple[int, int]], tuple[int, int]] | None:
    for _ in range(200):
        height = random.randint(2, 5)
        width = random.randint(3, 6)
        require_cells = {
            (random.randrange(height), 0),
            (random.randrange(height), 1),
            (random.randrange(height), width - 1),
            (0, random.randrange(width)),
            (height - 1, random.randrange(width)),
        }

        if corner == "tr":
            if require_corner_preserve:
                require_cells.add((0, width - 1))
            if require_overlap_cell:
                require_cells.add((0, width - 2))
        elif corner == "br":
            if require_corner_preserve:
                require_cells.add((height - 1, width - 1))
            if require_overlap_cell:
                require_cells.add((height - 1, width - 2))

        cells = _grow_shape(height, width, require_cells=require_cells)
        if cells is None:
            continue

        candidates = []
        for cell in cells:
            _r, c = cell
            if marker_outside_crop and c < 2:
                continue
            if not marker_outside_crop and c >= 2 and random.random() < 0.75:
                continue
            if require_corner_preserve:
                if corner == "tr" and cell == (0, width - 1):
                    continue
                if corner == "br" and cell == (height - 1, width - 1):
                    continue
            if require_overlap_cell:
                if corner == "tr" and cell == (0, width - 2):
                    continue
                if corner == "br" and cell == (height - 1, width - 2):
                    continue
            remain = [other for other in cells if other != cell and other[1] >= 2]
            if len(remain) >= 2:
                candidates.append(cell)

        if not candidates:
            continue
        return cells, random.choice(candidates)
    return None


def _place_bboxes(
    grid_shape: tuple[int, int],
    sprite_shapes: list[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    h, w = grid_shape
    occupied = np.zeros((h, w), dtype=bool)
    placements: list[tuple[int, int]] = []
    for bh, bw in sprite_shapes:
        options = []
        for r0 in range(1, h - bh):
            for c0 in range(1, w - bw):
                r1 = r0 + bh - 1
                c1 = c0 + bw - 1
                rr0 = max(0, r0 - 1)
                cc0 = max(0, c0 - 1)
                rr1 = min(h, r1 + 2)
                cc1 = min(w, c1 + 2)
                if not occupied[rr0:rr1, cc0:cc1].any():
                    options.append((r0, c0))
        if not options:
            return None
        r0, c0 = random.choice(options)
        placements.append((r0, c0))
        occupied[r0 : r0 + bh, c0 : c0 + bw] = True
    return placements


def _sample_corner_targets(num_sprites: int) -> tuple[list[str], bool]:
    duplicate_branch = random.random() < 0.6
    if duplicate_branch:
        repeated = random.choice(["tr", "br"])
        others = [name for name in CORNER_NAMES if name != repeated]
        targets = [repeated, repeated] + random.sample(others, num_sprites - 2)
        random.shuffle(targets)
    else:
        targets = random.sample(CORNER_NAMES, num_sprites)
        if not any(name in {"tr", "br"} for name in targets):
            replace_idx = random.randrange(num_sprites)
            targets[replace_idx] = random.choice(["tr", "br"])
    return targets, duplicate_branch


def _build_input() -> np.ndarray | None:
    h = random.randint(18, 24)
    w = random.randint(18, 24)
    num_sprites = random.randint(3, 4)
    targets, duplicate_branch = _sample_corner_targets(num_sprites)

    corner_palette = random.sample(list(range(1, 10)), 4)
    corner_colors = dict(zip(CORNER_NAMES, corner_palette))
    body_palette = [color for color in range(1, 10) if color not in corner_palette]
    if len(body_palette) < num_sprites:
        return None
    body_colors = random.sample(body_palette, num_sprites)

    right_targets = [i for i, corner in enumerate(targets) if corner in {"tr", "br"}]
    preserve_idx = random.choice(right_targets)
    marker_outside_idx = random.randrange(num_sprites)

    duplicate_corner = None
    duplicate_indices: list[int] = []
    if duplicate_branch:
        for corner in ("tr", "br"):
            inds = [i for i, name in enumerate(targets) if name == corner]
            if len(inds) >= 2:
                duplicate_corner = corner
                duplicate_indices = inds[:2]
                break
        if duplicate_corner is None:
            return None

    sprite_specs = []
    for i, corner in enumerate(targets):
        require_overlap = i in duplicate_indices
        require_preserve = i == preserve_idx
        marker_outside = i == marker_outside_idx or random.random() < 0.35
        shape_data = _make_sprite_shape(
            corner,
            require_corner_preserve=require_preserve,
            require_overlap_cell=require_overlap,
            marker_outside_crop=marker_outside,
        )
        if shape_data is None:
            return None
        cells, marker_local = shape_data
        bh = max(r for r, _c in cells) + 1
        bw = max(c for _r, c in cells) + 1
        sprite_specs.append(
            {
                "corner": corner,
                "body_color": body_colors[i],
                "marker_color": corner_colors[corner],
                "cells": cells,
                "marker_local": marker_local,
                "shape": (bh, bw),
            }
        )

    placements = _place_bboxes((h, w), [spec["shape"] for spec in sprite_specs])
    if placements is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    for corner, (r, c) in _corners(grid.shape).items():
        grid[r, c] = corner_colors[corner]

    for spec, (r0, c0) in zip(sprite_specs, placements):
        for rr, cc in spec["cells"]:
            color = spec["marker_color"] if (rr, cc) == spec["marker_local"] else spec["body_color"]
            grid[r0 + rr, c0 + cc] = color

    if not _validate_instance(grid, duplicate_branch=duplicate_branch):
        return None
    return grid


def _validate_instance(grid: np.ndarray, *, duplicate_branch: bool) -> bool:
    try:
        g, corner_colors, _color_to_corner, sprites = _parse_grid(grid)
    except AssertionError:
        return False

    if len({sprite["target_corner"] for sprite in sprites}) < 2:
        return False

    true_out = _transform(g)
    if np.array_equal(true_out, g):
        return False

    # Local-vs-global crop must matter.
    if np.array_equal(true_out, _transform(g, local_crop=False)):
        return False

    # Explicit marker erasure must matter.
    if np.array_equal(true_out, _transform(g, erase_marker=False)):
        return False

    # Corner preservation must be visible in at least one right-corner stamp.
    corner_effect_visible = False
    for sprite in sprites:
        patch = _cropped_patch(sprite, local_crop=True, erase_marker=True)
        bh, bw = patch.shape
        if sprite["target_corner"] == "tr" and patch[0, bw - 1] != BG:
            corner_effect_visible = True
        if sprite["target_corner"] == "br" and patch[bh - 1, bw - 1] != BG:
            corner_effect_visible = True
    if not corner_effect_visible:
        return False
    if np.array_equal(true_out, _transform(g, preserve_corner_markers=False)):
        return False

    if duplicate_branch:
        groups: dict[str, list[dict]] = {}
        for sprite in sprites:
            groups.setdefault(sprite["target_corner"], []).append(sprite)
        overlap_found = False
        order_matters = False
        for group in groups.values():
            if len(group) < 2:
                continue
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    p1 = _cropped_patch(group[i], local_crop=True, erase_marker=True)
                    p2 = _cropped_patch(group[j], local_crop=True, erase_marker=True)
                    h = min(p1.shape[0], p2.shape[0])
                    w = min(p1.shape[1], p2.shape[1])
                    if np.any((p1[:h, :w] != BG) & (p2[:h, :w] != BG)):
                        overlap_found = True
            if not np.array_equal(true_out, _transform(g, reverse_order=True)):
                order_matters = True
        if not (overlap_found and order_matters):
            return False

    # The four original corner labels must still be present.
    for corner, color in corner_colors.items():
        r, c = _corners(g.shape)[corner]
        if int(true_out[r, c]) != int(color):
            return False

    return np.array_equal(true_out, main(g))


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _build_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input for puzzle 7")
