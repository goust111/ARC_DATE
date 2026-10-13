from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
LINE_BLUE = 1
LINE_RED = 2
LINE_GREEN = 3
MARKER = 5

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
MAX_ATTEMPTS = 600


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _tight_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_coords(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _analyze_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    comps = _components_mask(g == MARKER)
    objects = []
    for coords in comps:
        bbox = _bbox_coords(coords)
        mask = _tight_mask(coords)
        objects.append(
            {
                "coords": coords,
                "bbox": bbox,
                "mask": mask,
                "area": int(coords.shape[0]),
                "active": not np.array_equal(mask, np.fliplr(mask)),
            }
        )
    assert bg == BG
    assert len(objects) >= 2
    return objects


def _line_color(total_active_pixels: int) -> int:
    if total_active_pixels <= 8:
        return LINE_BLUE
    if total_active_pixels <= 14:
        return LINE_RED
    return LINE_GREEN


def _solve_from_objects(grid: np.ndarray, objects: list[dict]) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    active = [obj for obj in objects if obj["active"]]
    assert len(active) >= 2
    color = _line_color(sum(obj["area"] for obj in active))
    out = g.copy()
    h, w = g.shape

    for r in range(h):
        participants = [obj for obj in active if obj["bbox"][0] <= r <= obj["bbox"][2]]
        if len(participants) >= 2:
            c0 = min(obj["bbox"][1] for obj in participants)
            c1 = max(obj["bbox"][3] for obj in participants)
            row = out[r, c0 : c1 + 1]
            row[row == BG] = color

    for c in range(w):
        participants = [obj for obj in active if obj["bbox"][1] <= c <= obj["bbox"][3]]
        if len(participants) >= 2:
            r0 = min(obj["bbox"][0] for obj in participants)
            r1 = max(obj["bbox"][2] for obj in participants)
            col = out[r0 : r1 + 1, c]
            col[col == BG] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect the left-right-asymmetric grey objects with row/column span fills colored by their total size.

    Input structure:
    - A black background contains several separated grey 4-connected objects.
    - Every object is small and monochrome, and each can be inspected inside its tight bounding box.
    - Some objects are mirror-symmetric left-to-right inside that box while others are not.

    Rule:
    - Split the grey foreground into 4-connected objects and test each tight-bbox mask against its left-right reflection.
    - Treat only the non-symmetric objects as active markers, and count their total grey pixels.
    - Use blue for totals up to 8, red for totals from 9 to 14, and green for totals of 15 or more.
    - For every row crossed by at least two active-marker bounding boxes, fill black cells from the leftmost active bbox edge to the rightmost active bbox edge with that line color.
    - Do the analogous span fill for every column crossed by at least two active-marker bounding boxes, while leaving original grey object pixels unchanged.

    Key insight:
    - Symmetry is an invisible gate: it decides which objects exist for the connector logic, and the connectors use bounding-box spans rather than per-row or per-column foreground pixels.
    """
    return _solve_from_objects(np.asarray(grid, dtype=int), _analyze_objects(grid))


def _solve_all_objects(grid: np.ndarray) -> np.ndarray:
    objs = _analyze_objects(grid)
    for obj in objs:
        obj["active"] = True
    return _solve_from_objects(grid, objs)


def _solve_pixel_extents(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    active = [obj for obj in _analyze_objects(g) if obj["active"]]
    color = _line_color(sum(obj["area"] for obj in active))
    out = g.copy()
    h, w = g.shape

    for r in range(h):
        intervals = []
        for obj in active:
            hits = obj["coords"][obj["coords"][:, 0] == r]
            if hits.size:
                intervals.append((int(hits[:, 1].min()), int(hits[:, 1].max())))
        if len(intervals) >= 2:
            c0 = min(lo for lo, _ in intervals)
            c1 = max(hi for _, hi in intervals)
            row = out[r, c0 : c1 + 1]
            row[row == BG] = color

    for c in range(w):
        intervals = []
        for obj in active:
            hits = obj["coords"][obj["coords"][:, 1] == c]
            if hits.size:
                intervals.append((int(hits[:, 0].min()), int(hits[:, 0].max())))
        if len(intervals) >= 2:
            r0 = min(lo for lo, _ in intervals)
            r1 = max(hi for _, hi in intervals)
            col = out[r0 : r1 + 1, c]
            col[col == BG] = color

    return out


ASYM_BASES = [
    np.array([[1, 0], [1, 1]], dtype=int),
    np.array([[1, 0, 0], [1, 1, 1]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=int),
    np.array([[1, 1, 0], [1, 1, 1], [0, 0, 1]], dtype=int),
]

SYM_BASES = [
    np.array([[0, 1, 0], [1, 1, 1]], dtype=int),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 0, 1], [1, 1, 1]], dtype=int),
    np.array([[0, 1, 0], [1, 1, 1], [1, 0, 1]], dtype=int),
    np.array([[1, 0, 0, 0, 1], [1, 1, 1, 1, 1]], dtype=int),
]


def _is_lr_symmetric(mask: np.ndarray) -> bool:
    return bool(np.array_equal(mask, np.fliplr(mask)))


def _unique_variants(base: np.ndarray, keep_symmetric: bool | None) -> list[np.ndarray]:
    variants = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for rot in range(4):
        cur = np.rot90(base, rot)
        for variant in (cur, np.fliplr(cur), np.flipud(cur)):
            coords = np.argwhere(variant > 0)
            r0, c0, r1, c1 = _bbox_coords(coords)
            tight = variant[r0 : r1 + 1, c0 : c1 + 1].copy()
            if keep_symmetric is not None and _is_lr_symmetric(tight) != keep_symmetric:
                continue
            key = (tight.shape, tight.tobytes())
            if key not in seen:
                seen.add(key)
                variants.append(tight)
    return variants


ASYM_VARIANTS = [variant for base in ASYM_BASES for variant in _unique_variants(base, keep_symmetric=False)]
SYM_VARIANTS = [variant for base in SYM_BASES for variant in _unique_variants(base, keep_symmetric=True)]


def _sprite_area(sprite: np.ndarray) -> int:
    return int(sprite.sum())


def _sprite_cells(sprite: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    cells = []
    for r, c in np.argwhere(sprite > 0):
        cells.append((int(top + r), int(left + c)))
    return cells


def _build_blocked(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    blocked = set()
    for r, c in cells:
        for dr, dc in DIR8 + [(0, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                blocked.add((nr, nc))
    return blocked


def _placement_ok(
    sprite: np.ndarray,
    top: int,
    left: int,
    h: int,
    w: int,
    blocked: set[tuple[int, int]],
) -> bool:
    sh, sw = sprite.shape
    if top < 0 or left < 0 or top + sh > h or left + sw > w:
        return False
    for cell in _sprite_cells(sprite, top, left):
        if cell in blocked:
            return False
    return True


def _bbox_at(sprite: np.ndarray, top: int, left: int) -> tuple[int, int, int, int]:
    sh, sw = sprite.shape
    return int(top), int(left), int(top + sh - 1), int(left + sw - 1)


def _place_sprite(
    placed: list[dict],
    sprite: np.ndarray,
    h: int,
    w: int,
    *,
    include_row: int | None = None,
    include_col: int | None = None,
    exclude_row: int | None = None,
    exclude_col: int | None = None,
    left_min: int | None = None,
    left_max: int | None = None,
    top_min: int | None = None,
    top_max: int | None = None,
    side_filter=None,
) -> dict | None:
    cells = {cell for item in placed for cell in item["cells"]}
    blocked = _build_blocked(cells, h, w)
    sh, sw = sprite.shape
    tops = range(0 if top_min is None else max(0, top_min), h - sh + 1 if top_max is None else min(h - sh, top_max) + 1)
    lefts = range(0 if left_min is None else max(0, left_min), w - sw + 1 if left_max is None else min(w - sw, left_max) + 1)
    candidates = []
    for top in tops:
        for left in lefts:
            if include_row is not None and not (top <= include_row <= top + sh - 1):
                continue
            if include_col is not None and not (left <= include_col <= left + sw - 1):
                continue
            if exclude_row is not None and top <= exclude_row <= top + sh - 1:
                continue
            if exclude_col is not None and left <= exclude_col <= left + sw - 1:
                continue
            if not _placement_ok(sprite, top, left, h, w, blocked):
                continue
            bbox = _bbox_at(sprite, top, left)
            if side_filter is not None and not side_filter(bbox):
                continue
            candidates.append((top, left, bbox))
    random.shuffle(candidates)
    if not candidates:
        return None
    top, left, bbox = candidates[0]
    return {"sprite": sprite, "top": top, "left": left, "bbox": bbox, "cells": _sprite_cells(sprite, top, left)}


def _pick_active_sprites(branch: str, need_count_at_least: int) -> list[np.ndarray] | None:
    for _ in range(200):
        if branch == "blue":
            count = 2
            target_ok = lambda total: total <= 8
        elif branch == "red":
            count = random.randint(max(2, need_count_at_least), 3)
            target_ok = lambda total: 9 <= total <= 14
        else:
            count = random.randint(max(3, need_count_at_least), 5)
            target_ok = lambda total: total >= 15
        sprites = [random.choice(ASYM_VARIANTS).copy() for _ in range(count)]
        total = sum(_sprite_area(sprite) for sprite in sprites)
        if target_ok(total):
            return sprites
    return None


def _render_grid(items: list[dict], h: int, w: int) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for item in items:
        for r, c in item["cells"]:
            grid[r, c] = MARKER
    return grid


def _active_row_span(objs: list[dict], row: int) -> tuple[int, int] | None:
    participants = [obj for obj in objs if obj["active"] and obj["bbox"][0] <= row <= obj["bbox"][2]]
    if len(participants) < 2:
        return None
    return min(obj["bbox"][1] for obj in participants), max(obj["bbox"][3] for obj in participants)


def _active_col_span(objs: list[dict], col: int) -> tuple[int, int] | None:
    participants = [obj for obj in objs if obj["active"] and obj["bbox"][1] <= col <= obj["bbox"][3]]
    if len(participants) < 2:
        return None
    return min(obj["bbox"][0] for obj in participants), max(obj["bbox"][2] for obj in participants)


def _place_horizontal_layout(active_sprites: list[np.ndarray], h: int, w: int):
    target_row = random.randint(3, h - 4)
    placed: list[dict] = []
    cur_left = random.randint(0, 2)
    for idx, sprite in enumerate(active_sprites):
        item = _place_sprite(
            placed,
            sprite,
            h,
            w,
            include_row=target_row,
            left_min=cur_left,
            left_max=min(w - sprite.shape[1], cur_left + 2),
        )
        if item is None:
            return None
        placed.append(item)
        cur_left = item["bbox"][3] + random.randint(2, 4)
        if idx < len(active_sprites) - 1 and cur_left >= w - 2:
            return None
    return placed, target_row


def _place_vertical_layout(active_sprites: list[np.ndarray], h: int, w: int):
    target_col = random.randint(3, w - 4)
    placed: list[dict] = []
    cur_top = random.randint(0, 2)
    for idx, sprite in enumerate(active_sprites):
        item = _place_sprite(
            placed,
            sprite,
            h,
            w,
            include_col=target_col,
            top_min=cur_top,
            top_max=min(h - sprite.shape[0], cur_top + 2),
        )
        if item is None:
            return None
        placed.append(item)
        cur_top = item["bbox"][2] + random.randint(2, 4)
        if idx < len(active_sprites) - 1 and cur_top >= h - 2:
            return None
    return placed, target_col


def _place_both_layout(active_sprites: list[np.ndarray], h: int, w: int):
    if len(active_sprites) < 3:
        return None
    target_row = random.randint(4, h - 5)
    target_col = random.randint(4, w - 5)
    placed: list[dict] = []

    pivot = _place_sprite(
        placed,
        active_sprites[0],
        h,
        w,
        include_row=target_row,
        include_col=target_col,
        left_min=max(1, target_col - 2),
        left_max=min(w - active_sprites[0].shape[1] - 1, target_col),
        top_min=max(1, target_row - 2),
        top_max=min(h - active_sprites[0].shape[0] - 1, target_row),
    )
    if pivot is None:
        return None
    placed.append(pivot)

    left_item = _place_sprite(
        placed,
        active_sprites[1],
        h,
        w,
        include_row=target_row,
        exclude_col=target_col,
        left_max=max(0, pivot["bbox"][1] - 2),
    )
    if left_item is None:
        return None
    placed.append(left_item)

    vertical_item = _place_sprite(
        placed,
        active_sprites[2],
        h,
        w,
        include_col=target_col,
        exclude_row=target_row,
        top_min=min(h - active_sprites[2].shape[0], pivot["bbox"][2] + 2),
    )
    if vertical_item is None:
        vertical_item = _place_sprite(
            placed,
            active_sprites[2],
            h,
            w,
            include_col=target_col,
            exclude_row=target_row,
            top_max=max(0, pivot["bbox"][0] - 2),
        )
    if vertical_item is None:
        return None
    placed.append(vertical_item)

    for sprite in active_sprites[3:]:
        if random.random() < 0.5:
            extra = _place_sprite(placed, sprite, h, w, include_row=target_row)
        else:
            extra = _place_sprite(placed, sprite, h, w, include_col=target_col)
        if extra is None:
            return None
        placed.append(extra)
    return placed, target_row, target_col


def _place_special_symmetric(
    placed: list[dict],
    sprite: np.ndarray,
    h: int,
    w: int,
    *,
    mode: str,
    target_row: int | None,
    target_col: int | None,
) -> dict | None:
    active_grid = _render_grid(placed, h, w)
    objs = _analyze_objects(active_grid)
    row_span = _active_row_span(objs, target_row) if target_row is not None else None
    col_span = _active_col_span(objs, target_col) if target_col is not None else None

    if mode == "horizontal" and row_span is not None:
        lo, hi = row_span
        side_filter = lambda bbox: bbox[3] < lo - 1 or bbox[1] > hi + 1
        return _place_sprite(placed, sprite, h, w, include_row=target_row, side_filter=side_filter)
    if mode == "vertical" and col_span is not None:
        lo, hi = col_span
        side_filter = lambda bbox: bbox[2] < lo - 1 or bbox[0] > hi + 1
        return _place_sprite(placed, sprite, h, w, include_col=target_col, side_filter=side_filter)
    if mode == "both":
        if row_span is not None and random.random() < 0.5:
            lo, hi = row_span
            side_filter = lambda bbox: bbox[3] < lo - 1 or bbox[1] > hi + 1
            item = _place_sprite(placed, sprite, h, w, include_row=target_row, side_filter=side_filter)
            if item is not None:
                return item
        if col_span is not None:
            lo, hi = col_span
            side_filter = lambda bbox: bbox[2] < lo - 1 or bbox[0] > hi + 1
            return _place_sprite(placed, sprite, h, w, include_col=target_col, side_filter=side_filter)
    return None


def _validate_generated(inp: np.ndarray) -> bool:
    try:
        objs = _analyze_objects(inp)
        if not 5 <= len(objs) <= 8:
            return False
        if any(not (3 <= obj["area"] <= 7) for obj in objs):
            return False
        active = [obj for obj in objs if obj["active"]]
        symmetric = [obj for obj in objs if not obj["active"]]
        if len(active) < 2 or len(symmetric) < 1:
            return False

        out = main(inp)
        total = sum(obj["area"] for obj in active)
        expected_color = _line_color(total)
        added = (inp == BG) & (out == expected_color)
        if int(added.sum()) == 0:
            return False
        if np.any((inp != BG) & (out != inp)):
            return False
        if set(map(int, np.unique(out))) - {BG, MARKER, expected_color}:
            return False
        if np.array_equal(out, _solve_all_objects(inp)):
            return False
        if np.array_equal(out, _solve_pixel_extents(inp)):
            return False

        row_hits = 0
        col_hits = 0
        for r in range(inp.shape[0]):
            if _active_row_span(objs, r) is not None:
                row_hits += 1
        for c in range(inp.shape[1]):
            if _active_col_span(objs, c) is not None:
                col_hits += 1
        if row_hits == 0 and col_hits == 0:
            return False
        if int(added.sum()) > (inp.shape[0] * inp.shape[1]) // 2:
            return False
        return True
    except AssertionError:
        return False


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        h = random.randint(15, 18)
        w = random.randint(15, 18)
        mode = random.choice(["horizontal", "vertical", "both"])
        branch = random.choice(["blue", "red", "green"])
        need_active = 3 if mode == "both" else 2
        active_sprites = _pick_active_sprites(branch, need_active)
        if active_sprites is None:
            continue

        if mode == "horizontal":
            placed_info = _place_horizontal_layout(active_sprites, h, w)
            if placed_info is None:
                continue
            placed, target_row = placed_info
            target_col = None
        elif mode == "vertical":
            placed_info = _place_vertical_layout(active_sprites, h, w)
            if placed_info is None:
                continue
            placed, target_col = placed_info
            target_row = None
        else:
            placed_info = _place_both_layout(active_sprites, h, w)
            if placed_info is None:
                continue
            placed, target_row, target_col = placed_info

        total_count = random.randint(max(5, len(placed) + 1), 8)
        sym_needed = total_count - len(placed)
        special = _place_special_symmetric(
            placed,
            random.choice(SYM_VARIANTS).copy(),
            h,
            w,
            mode=mode,
            target_row=target_row,
            target_col=target_col,
        )
        if special is None:
            continue
        placed.append(special)

        while len(placed) < total_count:
            extra = _place_sprite(placed, random.choice(SYM_VARIANTS).copy(), h, w)
            if extra is None:
                break
            placed.append(extra)
        if len(placed) != total_count or sym_needed < 1:
            continue

        inp = _render_grid(placed, h, w)
        if _validate_generated(inp):
            return inp
    raise RuntimeError("failed to generate a valid input for asymmetric marker spans")
