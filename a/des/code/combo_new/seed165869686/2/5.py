from __future__ import annotations

from collections import deque
from itertools import combinations
import random

import numpy as np


BG = 0
MAX_GRID_SIDE = 30
EDGE_MARGIN = 1
ITEM_PADDING = 1
GEN_RETRIES = 400
TEMPLATE_BBOX_MAX = 4
MAX_TEMPLATE_AREA = 8
PAIR_COUNT_RANGE = (2, 4)

_TEMPLATE_LIBRARY: dict[tuple[int, int], dict[int, list[np.ndarray]]] | None = None


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _bbox_from_cells(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _component_mask(
    cells: list[tuple[int, int]], bbox: tuple[int, int, int, int]
) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask


def _components_by_color(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            bbox = _bbox_from_cells(cells)
            comps.append(
                {
                    "color": color,
                    "cells": cells,
                    "bbox": bbox,
                    "area": len(cells),
                    "mask": _component_mask(cells, bbox),
                }
            )

    return comps


def _is_frame_component(grid: np.ndarray, comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False

    color = int(comp["color"])
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    if comp["area"] != int(border.sum()):
        return False
    if not np.all(patch[border] == color):
        return False
    if not np.all(patch[~border] == BG):
        return False
    return True


def _has_hole(mask: np.ndarray) -> bool:
    h, w = mask.shape
    empty = ~mask
    if not empty.any():
        return False

    seen = np.zeros((h, w), dtype=bool)
    q = deque()
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            if empty[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c, h, w):
            if empty[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))

    return bool(np.any(empty & ~seen))


def _is_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False

    h, w = mask.shape
    start_r, start_c = map(int, pts[0])
    q = deque([(start_r, start_c)])
    seen = np.zeros((h, w), dtype=bool)
    seen[start_r, start_c] = True
    count = 0

    while q:
        r, c = q.popleft()
        count += 1
        for nr, nc in _neighbors4(r, c, h, w):
            if mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))

    return count == int(mask.sum())


def _template_library() -> dict[tuple[int, int], dict[int, list[np.ndarray]]]:
    global _TEMPLATE_LIBRARY
    if _TEMPLATE_LIBRARY is not None:
        return _TEMPLATE_LIBRARY

    library: dict[tuple[int, int], dict[int, list[np.ndarray]]] = {}
    for h in range(1, TEMPLATE_BBOX_MAX + 1):
        for w in range(1, TEMPLATE_BBOX_MAX + 1):
            cell_count = h * w
            parity_groups = {0: [], 1: []}
            for area in range(1, min(MAX_TEMPLATE_AREA, cell_count) + 1):
                for cells in combinations(range(cell_count), area):
                    mask = np.zeros((h, w), dtype=bool)
                    mask.flat[list(cells)] = True
                    if not mask[0, :].any() or not mask[-1, :].any():
                        continue
                    if not mask[:, 0].any() or not mask[:, -1].any():
                        continue
                    if not _is_connected(mask):
                        continue
                    if _has_hole(mask):
                        continue
                    parity_groups[area % 2].append(mask.copy())

            size = (h, w)
            groups = {parity: masks for parity, masks in parity_groups.items() if masks}
            if groups:
                library[size] = groups

    _TEMPLATE_LIBRARY = library
    return library


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each small template object into the matching hollow frame and choose the fill color by parity.

    Input structure:
    - The background is black.
    - Several small solid monochrome foreground objects act as templates; each one has area at most 8.
    - Several monochrome hollow rectangular frames are present; each frame has a 1-pixel border and a black interior.

    Rule:
    - Classify components into template objects and hollow frames.
    - Pair each template with the frame whose interior size matches the template's bounding-box size.
    - Copy the template's shape into the matched frame interior at the same relative bbox offsets.
    - If the template area is odd, use the frame color for the copied shape; if it is even, use the template's own color.
    - Leave the original templates and frame borders unchanged.

    Key insight:
    The pairing is determined by geometry, not by color, and the copied color switches between frame color and template color based on the template area's parity.
    """
    g = np.asarray(grid, dtype=int)
    comps = _components_by_color(g)

    frames = []
    templates = []
    for comp in comps:
        if _is_frame_component(g, comp):
            r0, c0, r1, c1 = comp["bbox"]
            comp["inner_size"] = (r1 - r0 - 1, c1 - c0 - 1)
            frames.append(comp)
        else:
            assert comp["area"] <= MAX_TEMPLATE_AREA
            templates.append(comp)

    assert frames and templates
    assert len(frames) == len(templates)

    frames_by_size: dict[tuple[int, int], dict] = {}
    for frame in frames:
        key = tuple(map(int, frame["inner_size"]))
        assert key not in frames_by_size
        frames_by_size[key] = frame

    templates_by_size: dict[tuple[int, int], dict] = {}
    for template in templates:
        key = tuple(map(int, template["mask"].shape))
        assert key not in templates_by_size
        templates_by_size[key] = template

    assert set(frames_by_size) == set(templates_by_size)

    out = g.copy()
    for size, template in templates_by_size.items():
        frame = frames_by_size[size]
        fill_color = (
            int(frame["color"]) if template["area"] % 2 == 1 else int(template["color"])
        )
        fr0, fc0, _fr1, _fc1 = frame["bbox"]
        mask = template["mask"]
        target = out[fr0 + 1 : fr0 + 1 + size[0], fc0 + 1 : fc0 + 1 + size[1]]
        target[mask] = fill_color

    return out


def _choose_template_specs() -> list[dict]:
    library = _template_library()
    all_sizes = list(library)
    max_pairs = min(PAIR_COUNT_RANGE[1], len(all_sizes))
    n_pairs = random.randint(PAIR_COUNT_RANGE[0], max_pairs)

    odd_sizes = [size for size in all_sizes if 1 in library[size]]
    even_sizes = [size for size in all_sizes if 0 in library[size]]
    odd_size = random.choice(odd_sizes)
    even_choices = [size for size in even_sizes if size != odd_size]
    even_size = random.choice(even_choices)

    chosen: list[tuple[tuple[int, int], int]] = [(odd_size, 1), (even_size, 0)]
    remaining = [size for size in all_sizes if size not in {odd_size, even_size}]
    random.shuffle(remaining)
    for size in remaining[: n_pairs - 2]:
        parity = random.choice(sorted(library[size]))
        chosen.append((size, parity))
    random.shuffle(chosen)

    specs = []
    for size, parity in chosen:
        mask = random.choice(library[size][parity]).copy()
        specs.append({"size": size, "mask": mask, "area": int(mask.sum())})
    return specs


def _sample_grid_shape(items: list[dict]) -> tuple[int, int]:
    total_area = sum(int(item["h"]) * int(item["w"]) for item in items)
    max_h = max(int(item["h"]) for item in items)
    max_w = max(int(item["w"]) for item in items)
    base = max(max_h + 2, max_w + 2, int(np.ceil(np.sqrt(total_area * 2.6))))

    h_low = max(base - 2, max_h + 2, 10)
    h_high = min(MAX_GRID_SIDE, base + 6)
    w_low = max(base - 2, max_w + 2, 10)
    w_high = min(MAX_GRID_SIDE, base + 6)
    if h_low > h_high:
        h_low = h_high
    if w_low > w_high:
        w_low = w_high

    return random.randint(h_low, h_high), random.randint(w_low, w_high)


def _valid_positions(occupied: np.ndarray, h: int, w: int) -> list[tuple[int, int]]:
    grid_h, grid_w = occupied.shape
    positions = []
    r_min = EDGE_MARGIN
    r_max = grid_h - h - EDGE_MARGIN
    c_min = EDGE_MARGIN
    c_max = grid_w - w - EDGE_MARGIN
    if r_max < r_min or c_max < c_min:
        return positions

    for r0 in range(r_min, r_max + 1):
        for c0 in range(c_min, c_max + 1):
            rr0 = max(0, r0 - ITEM_PADDING)
            rr1 = min(grid_h, r0 + h + ITEM_PADDING)
            cc0 = max(0, c0 - ITEM_PADDING)
            cc1 = min(grid_w, c0 + w + ITEM_PADDING)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            positions.append((r0, c0))
    return positions


def _place_items(items: list[dict], shape: tuple[int, int]) -> bool:
    occupied = np.zeros(shape, dtype=bool)
    order = list(range(len(items)))
    random.shuffle(order)
    order.sort(key=lambda idx: items[idx]["h"] * items[idx]["w"], reverse=True)

    for idx in order:
        item = items[idx]
        options = _valid_positions(occupied, int(item["h"]), int(item["w"]))
        if not options:
            return False
        item["pos"] = random.choice(options)
        r0, c0 = item["pos"]
        occupied[r0 : r0 + item["h"], c0 : c0 + item["w"]] = True
    return True


def _draw_frame(grid: np.ndarray, top: int, left: int, h: int, w: int, color: int) -> None:
    grid[top, left : left + w] = color
    grid[top + h - 1, left : left + w] = color
    grid[top : top + h, left] = color
    grid[top : top + h, left + w - 1] = color


def _build_scene() -> tuple[np.ndarray, np.ndarray] | None:
    template_specs = _choose_template_specs()
    n_pairs = len(template_specs)
    palette = random.sample(range(1, 10), 2 * n_pairs)
    template_colors = palette[:n_pairs]
    frame_colors = palette[n_pairs:]

    pairs = []
    items = []
    for idx, spec in enumerate(template_specs):
        h, w = map(int, spec["size"])
        pair = {
            "mask": spec["mask"],
            "size": (h, w),
            "area": int(spec["area"]),
            "template_color": int(template_colors[idx]),
            "frame_color": int(frame_colors[idx]),
        }
        pairs.append(pair)
        items.append({"kind": "template", "pair_idx": idx, "h": h, "w": w})
        items.append({"kind": "frame", "pair_idx": idx, "h": h + 2, "w": w + 2})

    for _ in range(40):
        grid_shape = _sample_grid_shape(items)
        items_copy = [item.copy() for item in items]
        if not _place_items(items_copy, grid_shape):
            continue

        inp = np.zeros(grid_shape, dtype=int)
        out = np.zeros(grid_shape, dtype=int)

        for item in items_copy:
            pair = pairs[int(item["pair_idx"])]
            r0, c0 = item["pos"]
            h, w = int(item["h"]), int(item["w"])
            if item["kind"] == "template":
                mask = pair["mask"]
                patch = inp[r0 : r0 + h, c0 : c0 + w]
                patch[mask] = int(pair["template_color"])
                out_patch = out[r0 : r0 + h, c0 : c0 + w]
                out_patch[mask] = int(pair["template_color"])
                pair["template_pos"] = (r0, c0)
            else:
                _draw_frame(inp, r0, c0, h, w, int(pair["frame_color"]))
                _draw_frame(out, r0, c0, h, w, int(pair["frame_color"]))
                pair["frame_pos"] = (r0, c0)

        for pair in pairs:
            fr0, fc0 = pair["frame_pos"]
            h, w = pair["size"]
            fill_color = (
                int(pair["frame_color"])
                if int(pair["area"]) % 2 == 1
                else int(pair["template_color"])
            )
            fill_patch = out[fr0 + 1 : fr0 + 1 + h, fc0 + 1 : fc0 + 1 + w]
            fill_patch[pair["mask"]] = fill_color

        try:
            solved = main(inp)
        except Exception:
            continue
        if np.array_equal(solved, out):
            return inp, out

    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _build_scene()
        if built is None:
            continue
        inp, _out = built
        return inp
    raise RuntimeError("failed to generate valid input")
