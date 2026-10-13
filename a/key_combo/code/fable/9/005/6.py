from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

BACKGROUND = ORANGE
PALETTE = [BLUE, RED, GREEN, YELLOW, GREY, PINK, TEAL, MAROON]
GEN_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]

    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in steps:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rect_outline_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells = [(r0, c) for c in range(c0, c1 + 1)]
    cells += [(r1, c) for c in range(c0, c1 + 1)]
    cells += [(r, c0) for r in range(r0 + 1, r1)]
    cells += [(r, c1) for r in range(r0 + 1, r1)]
    return cells


def _diamond_outline_cells(cr: int, cc: int, radius: int) -> list[tuple[int, int]]:
    cells = []
    for dr in range(-radius, radius + 1):
        dc = radius - abs(dr)
        r = cr + dr
        if dc == 0:
            cells.append((r, cc))
        else:
            cells.append((r, cc - dc))
            cells.append((r, cc + dc))
    return cells


def _rect_interior_mask(shape: tuple[int, int], r0: int, c0: int, r1: int, c1: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if r1 - r0 >= 2 and c1 - c0 >= 2:
        mask[r0 + 1 : r1, c0 + 1 : c1] = True
    return mask


def _diamond_interior_mask(shape: tuple[int, int], cr: int, cc: int, radius: int) -> np.ndarray:
    h, w = shape
    mask = np.zeros(shape, dtype=bool)
    for r in range(max(0, cr - radius + 1), min(h, cr + radius)):
        rem = radius - 1 - abs(r - cr)
        if rem >= 0:
            c_lo = max(0, cc - rem)
            c_hi = min(w - 1, cc + rem)
            mask[r, c_lo : c_hi + 1] = True
    return mask


def _in_bounds(shape: tuple[int, int], cells: list[tuple[int, int]]) -> bool:
    h, w = shape
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _touches_occupied(cells: list[tuple[int, int]], occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    for r, c in cells:
        if occupied[r, c]:
            return True
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and occupied[nr, nc]:
                return True
    return False


def _detect_outline(coords: np.ndarray) -> dict | None:
    pts = {tuple(map(int, rc)) for rc in coords.tolist()}
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1

    rect = set(_rect_outline_cells(r0, c0, r1, c1))
    if h >= 3 and w >= 3 and pts == rect:
        return {
            "kind": "rect",
            "bbox": (r0, c0, r1, c1),
            "cells": coords.copy(),
        }

    if h == w and h >= 3 and h % 2 == 1:
        radius = h // 2
        cr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        diamond = set(_diamond_outline_cells(cr, cc, radius))
        if pts == diamond:
            return {
                "kind": "diamond",
                "center": (cr, cc),
                "radius": radius,
                "cells": coords.copy(),
            }

    return None


def _outline_cells_from_object(obj: dict, expanded: int = 0) -> list[tuple[int, int]]:
    if obj["kind"] == "rect":
        r0, c0, r1, c1 = obj["bbox"]
        return _rect_outline_cells(r0 - expanded, c0 - expanded, r1 + expanded, c1 + expanded)
    cr, cc = obj["center"]
    return _diamond_outline_cells(cr, cc, obj["radius"] + expanded)


def _interior_mask_from_object(shape: tuple[int, int], obj: dict) -> np.ndarray:
    if obj["kind"] == "rect":
        return _rect_interior_mask(shape, *obj["bbox"])
    cr, cc = obj["center"]
    return _diamond_interior_mask(shape, cr, cc, obj["radius"])


def _parse_objects(grid: np.ndarray) -> list[dict]:
    bg = _get_bg(grid)
    all_non_bg = grid != bg
    objects: list[dict] = []

    for color in sorted(set(map(int, np.unique(grid))) - {bg}):
        for coords in _components(grid == color, connectivity=8):
            info = _detect_outline(coords)
            if info is None:
                continue
            info["color"] = int(color)
            info["outline_mask"] = np.zeros(grid.shape, dtype=bool)
            info["outline_mask"][coords[:, 0], coords[:, 1]] = True
            interior = _interior_mask_from_object(grid.shape, info)
            content_mask = interior & all_non_bg & ~info["outline_mask"]
            comps = _components(content_mask, connectivity=4)
            if not comps:
                continue
            info["content_mask"] = content_mask
            info["content_components"] = comps
            objects.append(info)

    assert objects, "expected at least one outlined object"
    return objects


def _shrink_outline(obj: dict) -> list[tuple[int, int]]:
    coords = np.argwhere(obj["content_mask"])
    if obj["kind"] == "rect":
        r0, c0, r1, c1 = obj["bbox"]
        min_r = int(coords[:, 0].min())
        max_r = int(coords[:, 0].max())
        min_c = int(coords[:, 1].min())
        max_c = int(coords[:, 1].max())
        k = min(min_r - r0 - 1, r1 - max_r - 1, min_c - c0 - 1, c1 - max_c - 1)
        return _rect_outline_cells(r0 + k, c0 + k, r1 - k, c1 - k)

    cr, cc = obj["center"]
    max_dist = max(abs(int(r) - cr) + abs(int(c) - cc) for r, c in coords)
    return _diamond_outline_cells(cr, cc, max_dist + 1)


def _expand_outline(obj: dict, occupied_other: np.ndarray, shape: tuple[int, int]) -> list[tuple[int, int]]:
    if obj["kind"] == "rect":
        r0, c0, r1, c1 = obj["bbox"]
        grow = 0
        while True:
            candidate = _rect_outline_cells(r0 - (grow + 1), c0 - (grow + 1), r1 + (grow + 1), c1 + (grow + 1))
            if not _in_bounds(shape, candidate) or _touches_occupied(candidate, occupied_other):
                break
            grow += 1
        return _rect_outline_cells(r0 - grow, c0 - grow, r1 + grow, c1 + grow)

    cr, cc = obj["center"]
    radius = obj["radius"]
    grow = 0
    while True:
        candidate = _diamond_outline_cells(cr, cc, radius + grow + 1)
        if not _in_bounds(shape, candidate) or _touches_occupied(candidate, occupied_other):
            break
        grow += 1
    return _diamond_outline_cells(cr, cc, radius + grow)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shrink each single-piece enclosure around its payload and expand each fragmented enclosure to its maximal safe shell.

    Input structure:
    - The background is orange.
    - Several one-cell-thick hollow outlines appear, each being either an axis-aligned rectangle or a Manhattan diamond.
    - Every outline encloses payload pixels of another color; some payloads are one 4-connected block, others are split into multiple 4-connected pieces.

    Rule:
    - For each outline, inspect only the non-background pixels inside that outline.
    - If the interior payload forms exactly one 4-connected component, redraw the same-colored outline as the smallest concentric shell of the same shape that still strictly contains that payload.
    - If the interior payload has multiple 4-connected components, redraw the same-colored outline as the largest concentric shell of the same shape whose next outward layer would either leave the grid or touch another object.
    - Payload pixels themselves stay fixed.

    Key insight:
    - Outline color is a decoy; the selector is payload fragmentation, while the amount of motion is target-driven: shrink until just wrapping the payload, or expand until the environment blocks the next layer.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    objects = _parse_objects(g)
    out = g.copy()

    for obj in objects:
        cells = obj["cells"]
        out[cells[:, 0], cells[:, 1]] = bg

    occupied_all = g != bg
    for obj in objects:
        occupied_other = occupied_all & ~obj["outline_mask"]
        if len(obj["content_components"]) == 1:
            new_cells = _shrink_outline(obj)
        else:
            new_cells = _expand_outline(obj, occupied_other, g.shape)
        for r, c in new_cells:
            out[r, c] = obj["color"]

    return out


def _halo_mask(shape: tuple[int, int], cells: set[tuple[int, int]]) -> np.ndarray:
    h, w = shape
    mask = np.zeros(shape, dtype=bool)
    for r, c in cells:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if 0 <= nr < h and 0 <= nc < w:
                    mask[nr, nc] = True
    return mask


def _grow_connected(allowed: set[tuple[int, int]], start: tuple[int, int], size: int) -> set[tuple[int, int]] | None:
    cells = {start}
    while len(cells) < size:
        frontier = []
        for r, c in cells:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (nr, nc) in allowed and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(frontier))
    return cells


def _make_connected_payload(allowed: set[tuple[int, int]], anchor_pool: set[tuple[int, int]]) -> set[tuple[int, int]] | None:
    if not allowed or not anchor_pool:
        return None
    start = random.choice(tuple(anchor_pool))
    max_size = min(len(allowed), 8)
    size = random.randint(2, max(2, max_size))
    return _grow_connected(allowed, start, size)


def _make_fragmented_payload(allowed: set[tuple[int, int]]) -> set[tuple[int, int]] | None:
    if len(allowed) < 2:
        return None
    comp_count = random.randint(2, min(4, len(allowed)))
    used: set[tuple[int, int]] = set()
    blocked: set[tuple[int, int]] = set()

    for _ in range(comp_count):
        free = list(allowed - blocked)
        if not free:
            return None
        seed = random.choice(free)
        size = random.randint(1, 3)
        local_allowed = allowed - blocked
        comp = _grow_connected(local_allowed, seed, size)
        if comp is None:
            comp = {seed}
        used |= comp
        for r, c in comp:
            blocked.add((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                blocked.add((nr, nc))

    if len(_components(_coords_to_mask(used), connectivity=4)) < 2:
        return None
    return used


def _coords_to_mask(coords: set[tuple[int, int]]) -> np.ndarray:
    if not coords:
        return np.zeros((1, 1), dtype=bool)
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    mask = np.zeros((max(rows) + 1, max(cols) + 1), dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    return mask


def _absolute(local: set[tuple[int, int]], r0: int, c0: int) -> set[tuple[int, int]]:
    return {(r + r0, c + c0) for r, c in local}


def _make_rect_object(
    role: str,
    outline_color: int,
    content_color: int,
    grid_shape: tuple[int, int],
    occupied: np.ndarray,
) -> dict | None:
    h, w = grid_shape
    delta = random.randint(1, 3)
    small_h = random.randint(5, 9)
    small_w = random.randint(5, 9)
    big_h = small_h + 2 * delta
    big_w = small_w + 2 * delta
    if big_h > h or big_w > w:
        return None

    touch_border = role == "expand"
    for _ in range(100):
        if touch_border:
            side = random.choice(["top", "bottom", "left", "right"])
            if side == "top":
                big_r0 = 0
                big_c0 = random.randint(0, w - big_w)
            elif side == "bottom":
                big_r0 = h - big_h
                big_c0 = random.randint(0, w - big_w)
            elif side == "left":
                big_r0 = random.randint(0, h - big_h)
                big_c0 = 0
            else:
                big_r0 = random.randint(0, h - big_h)
                big_c0 = w - big_w
        else:
            if h - big_h - 2 < 0 or w - big_w - 2 < 0:
                return None
            big_r0 = random.randint(1, h - big_h - 1)
            big_c0 = random.randint(1, w - big_w - 1)

        small_r0 = big_r0 + delta
        small_c0 = big_c0 + delta

        if role == "shrink":
            input_box = (big_r0, big_c0, big_r0 + big_h - 1, big_c0 + big_w - 1)
            output_box = (small_r0, small_c0, small_r0 + small_h - 1, small_c0 + small_w - 1)
            payload_box = output_box
        else:
            input_box = (small_r0, small_c0, small_r0 + small_h - 1, small_c0 + small_w - 1)
            output_box = (big_r0, big_c0, big_r0 + big_h - 1, big_c0 + big_w - 1)
            payload_box = input_box

        pr0, pc0, pr1, pc1 = payload_box
        allowed = {(r, c) for r in range(pr0 + 1, pr1) for c in range(pc0 + 1, pc1)}
        anchor_pool = {
            (r, c)
            for r, c in allowed
            if r in (pr0 + 1, pr1 - 1) or c in (pc0 + 1, pc1 - 1)
        }
        if role == "shrink":
            payload = _make_connected_payload(allowed, anchor_pool)
        else:
            payload = _make_fragmented_payload(allowed)
        if payload is None:
            continue

        input_cells = set(_rect_outline_cells(*input_box))
        output_cells = set(_rect_outline_cells(*output_box))
        all_cells = input_cells | output_cells | payload
        halo = _halo_mask(grid_shape, all_cells)
        if np.any(occupied & halo):
            continue

        occupied |= halo
        return {
            "role": role,
            "kind": "rect",
            "outline_color": outline_color,
            "content_color": content_color,
            "input_cells": input_cells,
            "output_cells": output_cells,
            "content_cells": payload,
        }
    return None


def _make_diamond_object(
    role: str,
    outline_color: int,
    content_color: int,
    grid_shape: tuple[int, int],
    occupied: np.ndarray,
) -> dict | None:
    h, w = grid_shape
    delta = random.randint(1, 3)
    small_radius = random.randint(2, 5)
    big_radius = small_radius + delta

    for _ in range(100):
        if role == "expand":
            side = random.choice(["top", "bottom", "left", "right"])
            if side == "top":
                cr = big_radius
                cc = random.randint(big_radius, w - big_radius - 1)
            elif side == "bottom":
                cr = h - 1 - big_radius
                cc = random.randint(big_radius, w - big_radius - 1)
            elif side == "left":
                cr = random.randint(big_radius, h - big_radius - 1)
                cc = big_radius
            else:
                cr = random.randint(big_radius, h - big_radius - 1)
                cc = w - 1 - big_radius
        else:
            if h - 2 * big_radius - 3 < 0 or w - 2 * big_radius - 3 < 0:
                return None
            cr = random.randint(big_radius + 1, h - big_radius - 2)
            cc = random.randint(big_radius + 1, w - big_radius - 2)

        input_radius = big_radius if role == "shrink" else small_radius
        output_radius = small_radius if role == "shrink" else big_radius
        payload_radius = output_radius if role == "shrink" else input_radius

        allowed = {
            (r, c)
            for r in range(cr - payload_radius + 1, cr + payload_radius)
            for c in range(cc - payload_radius + 1, cc + payload_radius)
            if abs(r - cr) + abs(c - cc) < payload_radius
        }
        anchor_pool = {(r, c) for r, c in allowed if abs(r - cr) + abs(c - cc) == payload_radius - 1}
        if role == "shrink":
            payload = _make_connected_payload(allowed, anchor_pool)
        else:
            payload = _make_fragmented_payload(allowed)
        if payload is None:
            continue

        input_cells = set(_diamond_outline_cells(cr, cc, input_radius))
        output_cells = set(_diamond_outline_cells(cr, cc, output_radius))
        all_cells = input_cells | output_cells | payload
        halo = _halo_mask(grid_shape, all_cells)
        if np.any(occupied & halo):
            continue

        occupied |= halo
        return {
            "role": role,
            "kind": "diamond",
            "outline_color": outline_color,
            "content_color": content_color,
            "input_cells": input_cells,
            "output_cells": output_cells,
            "content_cells": payload,
        }
    return None


def _render(objects: list[dict], shape: tuple[int, int], use_output: bool) -> np.ndarray:
    grid = np.full(shape, BACKGROUND, dtype=int)
    for obj in objects:
        outline = obj["output_cells"] if use_output else obj["input_cells"]
        for r, c in outline:
            grid[r, c] = obj["outline_color"]
        for r, c in obj["content_cells"]:
            grid[r, c] = obj["content_color"]
    return grid


def _sample_roles(count: int) -> list[str]:
    roles = ["shrink", "expand"]
    while len(roles) < count:
        roles.append(random.choice(["shrink", "expand"]))
    random.shuffle(roles)
    return roles


def _sample_shapes(count: int) -> list[str]:
    if count == 1:
        return [random.choice(["rect", "diamond"])]
    shapes = ["rect", "diamond"]
    while len(shapes) < count:
        shapes.append(random.choice(["rect", "diamond"]))
    random.shuffle(shapes)
    return shapes


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(20, 30)
    w = random.randint(20, 30)
    count = random.randint(2, 4)
    roles = _sample_roles(count)
    shapes = _sample_shapes(count)
    colors = random.sample(PALETTE, 2 * count)

    objects: list[dict] = []
    occupied = np.zeros((h, w), dtype=bool)
    ordered = sorted(zip(roles, shapes, colors[::2], colors[1::2]), key=lambda item: item[0] != "expand")

    for role, kind, outline_color, content_color in ordered:
        if kind == "rect":
            obj = _make_rect_object(role, outline_color, content_color, (h, w), occupied)
        else:
            obj = _make_diamond_object(role, outline_color, content_color, (h, w), occupied)
        if obj is None:
            return None
        objects.append(obj)

    inp = _render(objects, (h, w), use_output=False)
    expected = _render(objects, (h, w), use_output=True)

    if np.array_equal(inp, expected):
        return None
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for fable/9/005/6")
