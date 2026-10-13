from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_UTILS_SPEC = importlib.util.spec_from_file_location("arc_utils_shared", _ARC_UTILS_PATH)
if _ARC_UTILS_SPEC is None or _ARC_UTILS_SPEC.loader is None:
    raise ImportError(f"unable to load arc_utils from {_ARC_UTILS_PATH}")
_ARC_UTILS = importlib.util.module_from_spec(_ARC_UTILS_SPEC)
_ARC_UTILS_SPEC.loader.exec_module(_ARC_UTILS)

crop_bbox = _ARC_UTILS.crop_bbox
find_components = _ARC_UTILS.find_components
get_bg = _ARC_UTILS.get_bg
in_bounds = _ARC_UTILS.in_bounds
neighbors4 = _ARC_UTILS.neighbors4
paste = _ARC_UTILS.paste


BG = 0
PATH = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
SIDES = ("top", "bottom", "left", "right")
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_ATTEMPTS = 500

SPRITE_MASKS_BY_HEIGHT = {
    2: [
        np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
        np.array([[1, 1, 1], [1, 0, 1]], dtype=int),
        np.array([[1, 1, 1, 0], [0, 1, 1, 1]], dtype=int),
        np.array([[1, 1, 0, 1], [0, 1, 1, 1]], dtype=int),
    ],
    3: [
        np.array([[1, 1], [1, 0], [1, 1]], dtype=int),
        np.array([[1, 1, 0], [0, 1, 0], [1, 1, 1]], dtype=int),
        np.array([[1, 1, 1], [1, 0, 1], [0, 0, 1]], dtype=int),
        np.array([[0, 1, 1], [1, 1, 0], [1, 1, 1]], dtype=int),
        np.array([[1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=int),
    ],
    4: [
        np.array([[1, 1], [1, 0], [1, 1], [0, 1]], dtype=int),
        np.array([[1, 0, 1], [1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
        np.array([[1, 1, 0], [0, 1, 0], [1, 1, 1], [0, 0, 1]], dtype=int),
        np.array([[0, 1, 1], [1, 1, 0], [1, 1, 1], [0, 0, 1]], dtype=int),
    ],
}


def _is_boundary_cell(cell: tuple[int, int], shape: tuple[int, int]) -> bool:
    r, c = cell
    h, w = shape
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _is_corner(cell: tuple[int, int], shape: tuple[int, int]) -> bool:
    r, c = cell
    h, w = shape
    return (r in (0, h - 1)) and (c in (0, w - 1))


def _touches_side(cell: tuple[int, int], side: str, shape: tuple[int, int]) -> bool:
    r, c = cell
    h, w = shape
    if side == "top":
        return r == 0
    if side == "bottom":
        return r == h - 1
    if side == "left":
        return c == 0
    if side == "right":
        return c == w - 1
    raise ValueError(f"unknown side {side}")


def _side_distance(cell: tuple[int, int], side: str, shape: tuple[int, int]) -> int:
    r, c = cell
    h, w = shape
    if side == "top":
        return r
    if side == "bottom":
        return h - 1 - r
    if side == "left":
        return c
    if side == "right":
        return w - 1 - c
    raise ValueError(f"unknown side {side}")


def _ordered_simple_path(cells: set[tuple[int, int]], start: tuple[int, int] | None = None) -> list[tuple[int, int]]:
    degrees = {
        cell: sum((neighbor in cells) for neighbor in neighbors4(*cell))
        for cell in cells
    }
    endpoints = sorted(cell for cell, degree in degrees.items() if degree == 1)
    assert len(endpoints) == 2, "path must have exactly two endpoints"
    assert all(degree in (1, 2) for degree in degrees.values()), "path must not branch"

    cur = endpoints[0] if start is None else start
    assert cur in endpoints, "ordered traversal must start at an endpoint"
    prev = None
    path = []
    while True:
        path.append(cur)
        next_cells = [
            neighbor
            for neighbor in neighbors4(*cur)
            if neighbor in cells and neighbor != prev
        ]
        if not next_cells:
            break
        assert len(next_cells) == 1, "path traversal must be unique"
        prev, cur = cur, next_cells[0]
    assert len(path) == len(cells), "path traversal must visit every cell"
    return path


def _compress_sprite(crop: np.ndarray, color: int) -> np.ndarray:
    out = np.full((crop.shape[0], 1), BG, dtype=int)
    for r in range(crop.shape[0]):
        if np.any(crop[r, :] == int(color)):
            out[r, 0] = int(color)
    return out


def _extract_instance(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    assert bg == BG, "background must be black"

    components = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    sprites = [comp for comp in components if int(comp["color"]) not in (bg, PATH)]
    paths = [comp for comp in components if int(comp["color"]) == PATH]

    assert 3 <= len(sprites) <= 5, "expected 3-5 sprites"
    assert len(paths) == len(sprites), "each sprite must have one grey path"

    sprite_infos = []
    cell_to_sprite: dict[tuple[int, int], int] = {}
    for idx, comp in enumerate(sprites):
        color = int(comp["color"])
        bbox = tuple(map(int, comp["bbox"]))
        crop = crop_bbox(g, bbox)
        sprite_cells = {tuple(map(int, coord)) for coord in comp["coords"]}
        sprite_infos.append(
            {
                "index": idx,
                "color": color,
                "bbox": bbox,
                "crop": crop,
                "cells": sprite_cells,
                "area": int(comp["area"]),
                "height": int(crop.shape[0]),
                "width": int(crop.shape[1]),
            }
        )
        for cell in sprite_cells:
            cell_to_sprite[cell] = idx

    heights = {info["height"] for info in sprite_infos}
    assert len(heights) == 1, "all sprite heights must match"

    path_infos = []
    for comp in paths:
        cells = {tuple(map(int, coord)) for coord in comp["coords"]}
        endpoints = [cell for cell in cells if sum((nbr in cells) for nbr in neighbors4(*cell)) == 1]
        boundary_endpoints = [cell for cell in endpoints if _is_boundary_cell(cell, g.shape)]
        assert len(boundary_endpoints) == 1, "each path must end at exactly one boundary pixel"
        ordered = _ordered_simple_path(cells, start=boundary_endpoints[0])
        sprite_end = ordered[-1]
        attached = [
            neighbor
            for neighbor in neighbors4(*sprite_end)
            if in_bounds(g.shape, *neighbor) and int(g[neighbor]) not in (bg, PATH)
        ]
        assert len(attached) == 1, "sprite-side endpoint must touch exactly one sprite pixel"
        sprite_idx = cell_to_sprite[attached[0]]
        path_infos.append(
            {
                "cells": cells,
                "ordered": ordered,
                "length": len(ordered),
                "boundary_endpoint": ordered[0],
                "sprite_endpoint": sprite_end,
                "sprite_index": sprite_idx,
            }
        )

    lengths = [info["length"] for info in path_infos]
    assert len(set(lengths)) == len(lengths), "path lengths must be unique"
    return sprite_infos, path_infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort sprites by the lengths of their attached boundary paths, then compress each sprite into one column.

    Input structure:
    - The grid has black background, 3-5 monochrome 4-connected sprites, and one grey 4-connected path for each sprite.
    - Every grey path is one pixel wide, unbranched, touches the boundary at one endpoint, and touches exactly one sprite at its other endpoint.
    - The sprites have distinct colors and equal bounding-box heights, while their widths and shapes may differ.

    Rule:
    - Find every grey path and traverse it from the boundary endpoint to the sprite endpoint to measure its pixel length.
    - Associate each path to the unique sprite it touches, then sort the sprites by path length in ascending order.
    - Crop each sprite to its bounding box, compress each crop row-wise into a single column, and concatenate those columns left-to-right in the sorted order.

    Key insight:
    - The sorting key is not a visible property of the sprite itself; it is encoded by the length of the external grey path that reaches the boundary.
    """
    sprite_infos, path_infos = _extract_instance(np.asarray(grid, dtype=int))
    path_by_sprite = {info["sprite_index"]: info for info in path_infos}
    ordered_sprites = sorted(sprite_infos, key=lambda info: path_by_sprite[info["index"]]["length"])

    out_h = ordered_sprites[0]["height"]
    out = np.full((out_h, len(ordered_sprites)), BG, dtype=int)
    for col, sprite in enumerate(ordered_sprites):
        column = _compress_sprite(sprite["crop"], sprite["color"])
        out[:, col : col + 1] = column
    return out


def _validate_mask_library() -> None:
    for height, masks in SPRITE_MASKS_BY_HEIGHT.items():
        for mask in masks:
            assert mask.shape[0] == height
            assert 4 <= int(mask.sum()) <= 8
            assert not bool(np.all(mask == 1))
            assert all(mask[r].any() for r in range(mask.shape[0]))
            comps = find_components(mask, bg=0, connectivity=4, monochromatic=True, return_format="dict")
            assert len(comps) == 1, "sprite masks must be 4-connected"


_validate_mask_library()


def _make_sprite(mask: np.ndarray, color: int) -> np.ndarray:
    return np.where(mask == 1, int(color), BG).astype(int)


def _attachment_offsets(mask: np.ndarray, side: str) -> list[tuple[int, int]]:
    h, w = mask.shape
    if side == "top":
        return [(0, c) for c in range(w) if mask[0, c] == 1]
    if side == "bottom":
        return [(h - 1, c) for c in range(w) if mask[h - 1, c] == 1]
    if side == "left":
        return [(r, 0) for r in range(h) if mask[r, 0] == 1]
    if side == "right":
        return [(r, w - 1) for r in range(h) if mask[r, w - 1] == 1]
    raise ValueError(f"unknown side {side}")


def _attachment_start(r0: int, c0: int, offset: tuple[int, int], side: str) -> tuple[int, int]:
    rr, cc = r0 + offset[0], c0 + offset[1]
    if side == "top":
        return rr - 1, cc
    if side == "bottom":
        return rr + 1, cc
    if side == "left":
        return rr, cc - 1
    if side == "right":
        return rr, cc + 1
    raise ValueError(f"unknown side {side}")


def _can_place_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, padding: int = 1) -> bool:
    h, w = patch.shape
    H, W = grid.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    for dr, dc in np.argwhere(patch != BG):
        rr = int(r0 + dr)
        cc = int(c0 + dc)
        r1 = max(0, rr - padding)
        r2 = min(H, rr + padding + 1)
        c1 = max(0, cc - padding)
        c2 = min(W, cc + padding + 1)
        if np.any(grid[r1:r2, c1:c2] != BG):
            return False
    return True


def _candidate_positions(shape: tuple[int, int], mask: np.ndarray, side: str, length: int) -> list[tuple[int, int]]:
    H, W = shape
    h, w = mask.shape
    candidates = []
    for r0 in range(1, H - h):
        for c0 in range(1, W - w):
            if side == "top" and not (2 <= r0 <= length):
                continue
            if side == "bottom" and not (2 <= H - (r0 + h) <= length):
                continue
            if side == "left" and not (2 <= c0 <= length):
                continue
            if side == "right" and not (2 <= W - (c0 + w) <= length):
                continue
            candidates.append((r0, c0))
    random.shuffle(candidates)
    return candidates


def _neighbor_non_bg_cells(grid: np.ndarray, cell: tuple[int, int]) -> list[tuple[int, int]]:
    return [
        nbr
        for nbr in neighbors4(*cell)
        if in_bounds(grid.shape, *nbr) and int(grid[nbr]) != BG
    ]


def _search_path(
    grid_with_sprite: np.ndarray,
    start: tuple[int, int],
    side: str,
    length: int,
    min_turns: int,
) -> list[tuple[int, int]] | None:
    H, W = grid_with_sprite.shape
    if _is_boundary_cell(start, grid_with_sprite.shape):
        return None
    start_neighbors = _neighbor_non_bg_cells(grid_with_sprite, start)
    if len(start_neighbors) != 1:
        return None

    visited = {start}

    def rec(
        cur: tuple[int, int],
        prev_dir: tuple[int, int] | None,
        turns: int,
        path: list[tuple[int, int]],
    ) -> list[tuple[int, int]] | None:
        if len(path) == length:
            if _touches_side(cur, side, (H, W)) and not _is_corner(cur, (H, W)) and turns >= min_turns:
                return path
            return None

        remaining_moves = length - len(path)
        if _side_distance(cur, side, (H, W)) > remaining_moves:
            return None

        neighbors = list(neighbors4(*cur))
        random.shuffle(neighbors)
        for nxt in neighbors:
            if not in_bounds((H, W), *nxt):
                continue
            if nxt in visited:
                continue
            if int(grid_with_sprite[nxt]) != BG:
                continue

            will_finish = len(path) + 1 == length
            if will_finish:
                if not _touches_side(nxt, side, (H, W)) or _is_corner(nxt, (H, W)):
                    continue
            elif _is_boundary_cell(nxt, (H, W)):
                continue

            if any(int(grid_with_sprite[nbr]) != BG for nbr in neighbors4(*nxt) if in_bounds((H, W), *nbr)):
                continue

            step_dir = (nxt[0] - cur[0], nxt[1] - cur[1])
            next_turns = turns + int(prev_dir is not None and step_dir != prev_dir)
            visited.add(nxt)
            found = rec(nxt, step_dir, next_turns, path + [nxt])
            if found is not None:
                return found
            visited.remove(nxt)
        return None

    return rec(start, None, 0, [start])


def _count_turns(path: list[tuple[int, int]]) -> int:
    turns = 0
    prev_dir = None
    for a, b in zip(path, path[1:]):
        cur_dir = (b[0] - a[0], b[1] - a[1])
        if prev_dir is not None and cur_dir != prev_dir:
            turns += 1
        prev_dir = cur_dir
    return turns


def _expected_output(items: list[dict], sprite_height: int) -> np.ndarray:
    ordered = sorted(items, key=lambda item: item["length"])
    out = np.full((sprite_height, len(ordered)), BG, dtype=int)
    for col, item in enumerate(ordered):
        out[:, col : col + 1] = _compress_sprite(item["sprite"], item["color"])
    return out


def _orders_match(items: list[dict], key_fn) -> bool:
    ids_by_length = tuple(item["id"] for item in sorted(items, key=lambda item: item["length"]))
    ids_by_key = tuple(item["id"] for item in sorted(items, key=key_fn))
    return ids_by_length == ids_by_key


def _reject_degenerate_orderings(items: list[dict]) -> bool:
    checks = [
        lambda item: (item["bbox"][1], item["bbox"][0]),
        lambda item: (item["bbox"][0], item["bbox"][1]),
        lambda item: (item["area"], item["bbox"][1], item["bbox"][0]),
        lambda item: (item["width"], item["bbox"][1], item["bbox"][0]),
        lambda item: (item["color"], item["bbox"][1], item["bbox"][0]),
    ]
    return any(_orders_match(items, key_fn) for key_fn in checks)


def _build_item(
    grid: np.ndarray,
    mask: np.ndarray,
    color: int,
    side: str,
    length: int,
    item_id: int,
) -> dict | None:
    sprite = _make_sprite(mask, color)
    positions = _candidate_positions(grid.shape, mask, side, length)
    if not positions:
        return None

    attach_offsets = _attachment_offsets(mask, side)
    random.shuffle(attach_offsets)

    for r0, c0 in positions:
        if not _can_place_patch(grid, sprite, r0, c0, padding=1):
            continue
        temp = grid.copy()
        paste(temp, sprite, r0, c0, transparent=BG)

        offsets = attach_offsets[:]
        random.shuffle(offsets)
        for offset in offsets:
            start = _attachment_start(r0, c0, offset, side)
            if not in_bounds(temp.shape, *start) or int(temp[start]) != BG:
                continue
            min_turns = 1 if length >= 5 and random.random() < 0.7 else 0
            path = _search_path(temp, start, side, length, min_turns=min_turns)
            if path is None:
                path = _search_path(temp, start, side, length, min_turns=0)
            if path is None:
                continue

            path_grid = np.full(grid.shape, BG, dtype=int)
            for rr, cc in path:
                path_grid[rr, cc] = PATH
            paste(temp, path_grid, 0, 0, transparent=BG)
            bbox = (r0, c0, r0 + sprite.shape[0] - 1, c0 + sprite.shape[1] - 1)
            return {
                "id": item_id,
                "color": color,
                "length": length,
                "bbox": bbox,
                "sprite": sprite,
                "mask": mask,
                "path": path,
                "side": side,
                "turns": _count_turns(path),
                "area": int(mask.sum()),
                "width": int(sprite.shape[1]),
                "grid": temp,
            }
    return None


def _validate_instance(grid: np.ndarray, items: list[dict], expected: np.ndarray) -> bool:
    if not any(item["turns"] >= 1 for item in items):
        return False
    if _reject_degenerate_orderings(items):
        return False

    try:
        sprites, paths = _extract_instance(grid)
    except AssertionError:
        return False
    if len(sprites) != len(items) or len(paths) != len(items):
        return False
    if sorted(item["length"] for item in items) != sorted(info["length"] for info in paths):
        return False
    actual = main(grid)
    return bool(np.array_equal(actual, expected))


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(18, 24)
    W = random.randint(18, 24)
    n = random.randint(3, 5)
    sprite_height = random.choice(sorted(SPRITE_MASKS_BY_HEIGHT))
    lengths = random.sample(range(2, 9), n)
    colors = random.sample(PALETTE, n)
    masks = [random.choice(SPRITE_MASKS_BY_HEIGHT[sprite_height]).copy() for _ in range(n)]

    order = sorted(range(n), key=lambda idx: lengths[idx], reverse=True)
    grid = np.full((H, W), BG, dtype=int)
    items: list[dict] = []

    for idx in order:
        side_choices = list(SIDES)
        random.shuffle(side_choices)
        built = None
        for side in side_choices:
            built = _build_item(grid, masks[idx], colors[idx], side, lengths[idx], idx)
            if built is not None:
                grid = built.pop("grid")
                items.append(built)
                break
        if built is None:
            return None

    expected = _expected_output(items, sprite_height)
    if max(expected.shape) > 30:
        return None
    if not _validate_instance(grid, items, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
