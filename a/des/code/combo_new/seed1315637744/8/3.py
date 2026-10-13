import random
from collections import deque

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))

GRID_H_RANGE = (12, 24)
GRID_W_RANGE = (12, 24)
PAIR_COUNT_OPTIONS = (1, 2, 3)
ISOLATED_PRESENT_PROB = 0.5
ISOLATED_COUNT_RANGE = (1, 2)
PAIR_RECT_H_RANGE = (2, 4)
PAIR_RECT_W_RANGE = (2, 4)
ISOLATED_RECT_H_RANGE = (2, 5)
ISOLATED_RECT_W_RANGE = (2, 5)
BRIDGE_LEN_RANGE = (1, 6)
PLACEMENT_ATTEMPTS = 120
GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _find_components(grid: np.ndarray, bg: int = BG):
    h, w = grid.shape
    labels = np.full((h, w), -1, dtype=int)
    seen = np.zeros((h, w), dtype=bool)
    components = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue

            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            r0 = r1 = r
            c0 = c1 = c

            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                r0 = min(r0, rr)
                r1 = max(r1, rr)
                c0 = min(c0, cc)
                c1 = max(c1, cc)
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            comp_id = len(components)
            for rr, cc in coords:
                labels[rr, cc] = comp_id

            height = r1 - r0 + 1
            width = c1 - c0 + 1
            area = len(coords)
            components.append(
                {
                    "id": comp_id,
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "area": area,
                    "shape": (height, width),
                    "solid": area == height * width,
                }
            )

    return components, labels


def _same_shape(a, b) -> bool:
    return tuple(a["shape"]) == tuple(b["shape"])


def _perpendicular_clear(
    grid: np.ndarray, bbox: tuple[int, int, int, int], orientation: str, bg: int
) -> bool:
    h, w = grid.shape
    r0, c0, r1, c1 = bbox
    if orientation == "horizontal":
        r = r0
        for c in range(c0, c1 + 1):
            if r > 0 and int(grid[r - 1, c]) != bg:
                return False
            if r + 1 < h and int(grid[r + 1, c]) != bg:
                return False
    else:
        c = c0
        for r in range(r0, r1 + 1):
            if c > 0 and int(grid[r, c - 1]) != bg:
                return False
            if c + 1 < w and int(grid[r, c + 1]) != bg:
                return False
    return True


def _bridge_match(
    comp: dict,
    orientation: str,
    components: list[dict],
    labels: np.ndarray,
    grid: np.ndarray,
    bg: int,
):
    h, w = grid.shape
    r0, c0, r1, c1 = comp["bbox"]
    bridge_color = int(comp["color"])

    if orientation == "horizontal":
        if comp["shape"][0] != 1 or c0 == 0 or c1 == w - 1:
            return None
        if not _perpendicular_clear(grid, comp["bbox"], orientation, bg):
            return None
        left_id = int(labels[r0, c0 - 1])
        right_id = int(labels[r0, c1 + 1])
        if left_id < 0 or right_id < 0 or left_id == right_id:
            return None
        left = components[left_id]
        right = components[right_id]
        if not left["solid"] or not right["solid"]:
            return None
        if int(left["color"]) != int(right["color"]) or int(left["color"]) == bridge_color:
            return None
        if not _same_shape(left, right):
            return None
    else:
        if comp["shape"][1] != 1 or r0 == 0 or r1 == h - 1:
            return None
        if not _perpendicular_clear(grid, comp["bbox"], orientation, bg):
            return None
        top_id = int(labels[r0 - 1, c0])
        bottom_id = int(labels[r1 + 1, c0])
        if top_id < 0 or bottom_id < 0 or top_id == bottom_id:
            return None
        left = components[top_id]
        right = components[bottom_id]
        if not left["solid"] or not right["solid"]:
            return None
        if int(left["color"]) != int(right["color"]) or int(left["color"]) == bridge_color:
            return None
        if not _same_shape(left, right):
            return None

    lr0, lc0, lr1, lc1 = left["bbox"]
    rr0, rc0, rr1, rc1 = right["bbox"]
    bbox = (
        min(r0, lr0, rr0),
        min(c0, lc0, rc0),
        max(r1, lr1, rr1),
        max(c1, lc1, rc1),
    )

    return {
        "bridge_id": int(comp["id"]),
        "rect_ids": tuple(sorted((int(left["id"]), int(right["id"])))),
        "bridge_color": bridge_color,
        "bbox": bbox,
    }


def _find_bridges(grid: np.ndarray, bg: int = BG):
    components, labels = _find_components(grid, bg=bg)
    bridges = []
    used_pairs = set()

    for comp in components:
        if not comp["solid"]:
            continue
        matches = []
        if comp["shape"][0] == 1:
            match = _bridge_match(comp, "horizontal", components, labels, grid, bg)
            if match is not None:
                matches.append(match)
        if comp["shape"][1] == 1:
            match = _bridge_match(comp, "vertical", components, labels, grid, bg)
            if match is not None:
                matches.append(match)
        if len(matches) != 1:
            continue
        match = matches[0]
        if match["rect_ids"] in used_pairs:
            continue
        used_pairs.add(match["rect_ids"])
        bridges.append(match)

    bridges.sort(key=lambda item: (item["bbox"], item["bridge_color"], item["rect_ids"]))
    return bridges


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge bridge-linked same-colored rectangle pairs into solid blocks colored by the bridge.

    Input structure:
    - The background is black.
    - Non-black pixels form solid monochrome rectangles and some single-pixel-wide straight line segments.
    - A valid bridge is a horizontal or vertical line segment whose endpoints 4-touch two different same-colored solid rectangles of the same size.
    - Other rectangles may be isolated and have no bridge.

    Rule:
    - Find each bridge segment whose color differs from the two rectangles it connects.
    - For every bridged pair, take the tight bounding box covering both rectangles and the bridge.
    - Fill that entire bounding box with the bridge color.
    - Leave isolated rectangles and any non-bridge objects unchanged.

    Key insight:
    The thin connector is not just a signal of pairing: its color becomes the final filled rectangle, while the source rectangles only determine which bounding box to merge.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for bridge in _find_bridges(g, bg=BG):
        r0, c0, r1, c1 = bridge["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(bridge["bridge_color"])
    return out


def _make_pair_spec(rect_color: int, bridge_color: int) -> dict:
    orientation = random.choice(("horizontal", "vertical"))
    rect_h = random.randint(*PAIR_RECT_H_RANGE)
    rect_w = random.randint(*PAIR_RECT_W_RANGE)
    bridge_len = random.randint(*BRIDGE_LEN_RANGE)

    cells = []
    if orientation == "horizontal":
        attach = random.randrange(rect_h)
        bbox_h = rect_h
        bbox_w = 2 * rect_w + bridge_len
        for r in range(rect_h):
            for c in range(rect_w):
                cells.append((r, c, rect_color))
                cells.append((r, rect_w + bridge_len + c, rect_color))
        for c in range(rect_w, rect_w + bridge_len):
            cells.append((attach, c, bridge_color))
    else:
        attach = random.randrange(rect_w)
        bbox_h = 2 * rect_h + bridge_len
        bbox_w = rect_w
        for r in range(rect_h):
            for c in range(rect_w):
                cells.append((r, c, rect_color))
                cells.append((rect_h + bridge_len + r, c, rect_color))
        for r in range(rect_h, rect_h + bridge_len):
            cells.append((r, attach, bridge_color))

    return {
        "kind": "pair",
        "bbox_shape": (bbox_h, bbox_w),
        "cells": cells,
        "fill_color": bridge_color,
    }


def _make_isolated_spec(color: int) -> dict:
    rect_h = random.randint(*ISOLATED_RECT_H_RANGE)
    rect_w = random.randint(*ISOLATED_RECT_W_RANGE)
    cells = [(r, c, color) for r in range(rect_h) for c in range(rect_w)]
    return {
        "kind": "isolated",
        "bbox_shape": (rect_h, rect_w),
        "cells": cells,
    }


def _can_place(reserved: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(reserved.shape[0], r0 + h + 1)
    cc1 = min(reserved.shape[1], c0 + w + 1)
    return not reserved[rr0:rr1, cc0:cc1].any()


def _mark_reserved(reserved: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(reserved.shape[0], r0 + h + 1)
    cc1 = min(reserved.shape[1], c0 + w + 1)
    reserved[rr0:rr1, cc0:cc1] = True


def _place_specs(specs: list[dict], h: int, w: int):
    reserved = np.zeros((h, w), dtype=bool)
    placed = []

    for spec in sorted(specs, key=lambda item: item["bbox_shape"][0] * item["bbox_shape"][1], reverse=True):
        bh, bw = spec["bbox_shape"]
        if bh + 2 > h or bw + 2 > w:
            return None

        success = False
        row_hi = h - bh - 1
        col_hi = w - bw - 1
        if row_hi < 1 or col_hi < 1:
            return None

        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = random.randint(1, row_hi)
            c0 = random.randint(1, col_hi)
            if not _can_place(reserved, r0, c0, bh, bw):
                continue
            _mark_reserved(reserved, r0, c0, bh, bw)
            placed.append({**spec, "origin": (r0, c0)})
            success = True
            break

        if not success:
            return None

    return placed


def _render_input(h: int, w: int, placed_specs: list[dict]) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for spec in placed_specs:
        r0, c0 = spec["origin"]
        for dr, dc, color in spec["cells"]:
            grid[r0 + dr, c0 + dc] = int(color)
    return grid


def _expected_output(grid: np.ndarray, placed_specs: list[dict]) -> np.ndarray:
    out = grid.copy()
    for spec in placed_specs:
        if spec["kind"] != "pair":
            continue
        r0, c0 = spec["origin"]
        bh, bw = spec["bbox_shape"]
        out[r0 : r0 + bh, c0 : c0 + bw] = int(spec["fill_color"])
    return out


def _gen_attempt():
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)

    pair_count = random.choice(PAIR_COUNT_OPTIONS)
    isolated_count = (
        random.randint(*ISOLATED_COUNT_RANGE) if random.random() < ISOLATED_PRESENT_PROB else 0
    )
    total_colors = 2 * pair_count + isolated_count
    if total_colors > len(COLORS):
        return None

    palette = random.sample(COLORS, total_colors)
    specs = []
    idx = 0
    for _ in range(pair_count):
        rect_color = palette[idx]
        bridge_color = palette[idx + 1]
        idx += 2
        specs.append(_make_pair_spec(rect_color, bridge_color))
    for _ in range(isolated_count):
        specs.append(_make_isolated_spec(palette[idx]))
        idx += 1

    placed = _place_specs(specs, h, w)
    if placed is None:
        return None

    grid = _render_input(h, w, placed)
    expected = _expected_output(grid, placed)
    bridges = _find_bridges(grid, bg=BG)
    if len(bridges) != pair_count:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid bridged-rectangle ARC input")
