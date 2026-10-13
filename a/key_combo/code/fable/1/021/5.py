from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
FRAME = 4
GREY = 5
DIRS4 = ((-1, 0), (0, 1), (1, 0), (0, -1))
EAST = (0, 1)
PALETTE = (1, 2, 3, 6, 7, 8, 9)

CANVAS_LIMIT = 30
PANEL_GAP = 2
GEN_RETRIES = 400
SPRITE_RETRIES = 120
PLACEMENT_RETRIES = 80


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 4:
        steps = DIRS4
    else:
        steps = tuple(
            (dr, dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if (dr, dc) != (0, 0)
        )

    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in steps:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def rotate_offset(dr: int, dc: int, turns: int) -> tuple[int, int]:
    rr = int(dr)
    cc = int(dc)
    for _ in range(turns % 4):
        rr, cc = cc, -rr
    return int(rr), int(cc)


def rotate_cells(cells: list[tuple[int, int, int]], turns: int) -> list[tuple[int, int, int]]:
    out = []
    for dr, dc, color in cells:
        rr, cc = rotate_offset(dr, dc, turns)
        out.append((rr, cc, int(color)))
    return out


def direction_index(vec: tuple[int, int]) -> int:
    vec = (int(vec[0]), int(vec[1]))
    assert vec in DIRS4, f"invalid direction {vec}"
    return DIRS4.index(vec)


def _frame_coords(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    coords = set()
    for c in range(c0, c1 + 1):
        coords.add((r0, c))
        coords.add((r1, c))
    for r in range(r0, r1 + 1):
        coords.add((r, c0))
        coords.add((r, c1))
    return coords


def find_panel_frames(grid: np.ndarray, bg: int) -> list[dict]:
    panels: list[dict] = []
    for color in sorted(set(map(int, grid.ravel())) - {int(bg)}):
        for comp in components_mask(grid == color, connectivity=4):
            box = bbox_coords(comp)
            r0, c0, r1, c1 = box
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            expected = _frame_coords(box)
            actual = {tuple(map(int, rc)) for rc in comp}
            if actual == expected:
                panels.append({"color": int(color), "bbox": box})
    assert len(panels) == 2, "expected exactly two bordered panels"
    return panels


def parse_target_panel(inner: np.ndarray, bg: int) -> dict[int, dict]:
    markers: dict[int, dict] = {}
    comps = components_mask(inner != bg, connectivity=4)
    assert comps, "target panel must contain markers"
    for comp in comps:
        assert comp.shape[0] == 2, "each target marker must be a 2-cell pair"
        cells = [(int(r), int(c), int(inner[r, c])) for r, c in comp]
        grey_cells = [(r, c) for r, c, color in cells if color == GREY]
        anchor_cells = [(r, c, color) for r, c, color in cells if color != GREY]
        assert len(grey_cells) == 1 and len(anchor_cells) == 1
        anchor_r, anchor_c, anchor_color = anchor_cells[0]
        grey_r, grey_c = grey_cells[0]
        direction = (grey_r - anchor_r, grey_c - anchor_c)
        assert direction in DIRS4
        assert anchor_color not in markers, "target anchor colors must be unique"
        markers[anchor_color] = {
            "anchor": (anchor_r, anchor_c),
            "dir": direction,
        }
    return markers


def parse_source_panel(inner: np.ndarray, bg: int) -> dict[int, dict]:
    sprites: dict[int, dict] = {}
    comps = components_mask(inner != bg, connectivity=4)
    assert comps, "source panel must contain sprites"
    for comp in comps:
        assert comp.shape[0] >= 5, "source sprites must be larger than markers"
        entries = [(int(r), int(c), int(inner[r, c])) for r, c in comp]
        counts = Counter(color for _r, _c, color in entries)
        grey_cells = [(r, c) for r, c, color in entries if color == GREY]
        assert len(grey_cells) == 1, "each sprite must have exactly one grey direction cell"
        anchor_colors = [color for color, count in counts.items() if color not in {GREY, bg} and count == 1]
        assert len(anchor_colors) == 1, "anchor must be the only non-grey singleton color"
        anchor_color = int(anchor_colors[0])
        anchor_cells = [(r, c) for r, c, color in entries if color == anchor_color]
        assert len(anchor_cells) == 1
        anchor_r, anchor_c = anchor_cells[0]
        grey_r, grey_c = grey_cells[0]
        direction = (grey_r - anchor_r, grey_c - anchor_c)
        assert direction in DIRS4
        rel_cells = [(r - anchor_r, c - anchor_c, color) for r, c, color in entries]
        assert anchor_color not in sprites, "source anchor colors must be unique"
        sprites[anchor_color] = {
            "dir": direction,
            "cells": rel_cells,
        }
    return sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match source sprites to target markers by anchor color, rotate each sprite around its anchor to the marker direction, and stamp it into the target panel.

    Input structure:
    - The canvas contains exactly two hollow rectangular panels on a common background.
    - One panel holds several disconnected multicolor sprites; each sprite has one colored anchor cell and one adjacent grey direction cell.
    - The other panel holds disconnected 2-cell markers; each marker is one colored anchor pixel with one adjacent grey direction pixel.

    Rule:
    - Identify the source panel by its larger connected sprites and the target panel by its 2-cell markers.
    - For each source sprite, use its unique non-grey singleton color as the anchor color, pair it with the target marker of the same color, and compare the anchor-to-grey directions.
    - Rotate the entire sprite by 0, 90, 180, or 270 degrees around the anchor so its direction matches the target marker, then translate it so the anchor lands on the marker position.
    - Return the target panel alone, keeping its border and the stamped sprites inside it.

    Key insight:
    - Color decides which sprite goes to which target, while the grey neighbor decides how much that specific sprite must rotate before stamping.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    panels = find_panel_frames(g, bg)

    source = None
    target = None
    target_box = None
    for panel in panels:
        r0, c0, r1, c1 = panel["bbox"]
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        source_info = None
        target_info = None
        try:
            source_info = parse_source_panel(inner, bg)
        except AssertionError:
            pass
        try:
            target_info = parse_target_panel(inner, bg)
        except AssertionError:
            pass

        assert (source_info is None) != (target_info is None), "each panel must have one unambiguous role"
        if source_info is not None:
            source = source_info
        else:
            target = target_info
            target_box = panel["bbox"]

    assert source is not None and target is not None and target_box is not None
    assert set(source) == set(target), "source anchors and target markers must match by color"

    out = crop_bbox(g, target_box)
    inner_out = out[1:-1, 1:-1]
    for anchor_color, marker in target.items():
        sprite = source[anchor_color]
        turns = (direction_index(marker["dir"]) - direction_index(sprite["dir"])) % 4
        for dr, dc, color in rotate_cells(sprite["cells"], turns):
            rr = marker["anchor"][0] + dr
            cc = marker["anchor"][1] + dc
            assert 0 <= rr < inner_out.shape[0] and 0 <= cc < inner_out.shape[1], "stamped sprite must stay inside the target panel"
            inner_out[rr, cc] = color
    return out


def _rel_bbox(cells: list[tuple[int, int, int]]) -> tuple[int, int, int, int]:
    rows = [dr for dr, _dc, _color in cells]
    cols = [dc for _dr, dc, _color in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    return (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _gen_body_offsets() -> list[tuple[int, int]] | None:
    target_size = random.choice((4, 5))
    occupied = {(0, 0), EAST}
    body: list[tuple[int, int]] = []
    for _ in range(target_size):
        frontier = set()
        for r, c in occupied:
            for dr, dc in DIRS4:
                nr = r + dr
                nc = c + dc
                if not (-2 <= nr <= 2 and -2 <= nc <= 2):
                    continue
                if (nr, nc) in occupied:
                    continue
                frontier.add((nr, nc))
        if not frontier:
            return None
        choice = random.choice(sorted(frontier))
        occupied.add(choice)
        body.append(choice)

    full = [(0, 0), EAST] + body
    rows = [r for r, _c in full]
    cols = [c for _r, c in full]
    if max(rows) - min(rows) < 2 or max(cols) - min(cols) < 2:
        return None
    return body


def _gen_base_sprite(anchor_color: int, body_pool: list[int]) -> list[tuple[int, int, int]] | None:
    assert len(body_pool) >= 2
    for _ in range(SPRITE_RETRIES):
        body = _gen_body_offsets()
        if body is None:
            continue
        colors = random.sample(body_pool, 2)
        body_cells = body[:]
        random.shuffle(body_cells)
        split = 2
        cells = [(0, 0, anchor_color), (EAST[0], EAST[1], GREY)]
        for i, (dr, dc) in enumerate(body_cells):
            color = colors[0] if i < split else colors[1]
            cells.append((dr, dc, color))
        return cells
    return None


def _place_shapes(
    panel_shape: tuple[int, int], shapes: list[list[tuple[int, int, int]]]
) -> list[tuple[int, int]] | None:
    h, w = panel_shape
    order = sorted(
        range(len(shapes)),
        key=lambda idx: (_rel_bbox(shapes[idx])[2] - _rel_bbox(shapes[idx])[0] + 1)
        * (_rel_bbox(shapes[idx])[3] - _rel_bbox(shapes[idx])[1] + 1),
        reverse=True,
    )
    anchors: list[tuple[int, int] | None] = [None] * len(shapes)
    boxes: list[tuple[int, int, int, int]] = []

    for idx in order:
        r0, c0, r1, c1 = _rel_bbox(shapes[idx])
        candidates = []
        for ar in range(-r0, h - r1):
            for ac in range(-c0, w - c1):
                abs_box = (ar + r0, ac + c0, ar + r1, ac + c1)
                if all(_boxes_separated(abs_box, old_box, pad=1) for old_box in boxes):
                    candidates.append((ar, ac, abs_box))
        if not candidates:
            return None
        ar, ac, abs_box = random.choice(candidates)
        anchors[idx] = (ar, ac)
        boxes.append(abs_box)

    return [anchor for anchor in anchors if anchor is not None]


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int, int]], anchor: tuple[int, int]) -> None:
    ar, ac = anchor
    for dr, dc, color in cells:
        grid[ar + dr, ac + dc] = color


def _panel_with_frame(inner: np.ndarray) -> np.ndarray:
    h, w = inner.shape
    panel = np.full((h + 2, w + 2), BG, dtype=int)
    panel[0, :] = FRAME
    panel[-1, :] = FRAME
    panel[:, 0] = FRAME
    panel[:, -1] = FRAME
    panel[1:-1, 1:-1] = inner
    return panel


def _compose_canvas(source_panel: np.ndarray, target_panel: np.ndarray, layout: str) -> np.ndarray:
    panels = [("source", source_panel), ("target", target_panel)]
    random.shuffle(panels)
    first_name, first = panels[0]
    second_name, second = panels[1]

    if layout == "horizontal":
        base_h = max(first.shape[0], second.shape[0]) + 2
        base_w = first.shape[1] + second.shape[1] + PANEL_GAP + 2
        canvas_h = base_h + random.randint(0, min(3, CANVAS_LIMIT - base_h))
        canvas_w = base_w + random.randint(0, min(2, CANVAS_LIMIT - base_w))
        total_margin_w = canvas_w - (first.shape[1] + second.shape[1] + PANEL_GAP)
        left_margin = random.randint(1, total_margin_w - 1)
        first_r = random.randint(1, canvas_h - first.shape[0] - 1)
        second_r = random.randint(1, canvas_h - second.shape[0] - 1)
        first_c = left_margin
        second_c = first_c + first.shape[1] + PANEL_GAP
    else:
        base_h = first.shape[0] + second.shape[0] + PANEL_GAP + 2
        base_w = max(first.shape[1], second.shape[1]) + 2
        canvas_h = base_h + random.randint(0, min(2, CANVAS_LIMIT - base_h))
        canvas_w = base_w + random.randint(0, min(3, CANVAS_LIMIT - base_w))
        total_margin_h = canvas_h - (first.shape[0] + second.shape[0] + PANEL_GAP)
        top_margin = random.randint(1, total_margin_h - 1)
        first_c = random.randint(1, canvas_w - first.shape[1] - 1)
        second_c = random.randint(1, canvas_w - second.shape[1] - 1)
        first_r = top_margin
        second_r = first_r + first.shape[0] + PANEL_GAP

    canvas = np.full((canvas_h, canvas_w), BG, dtype=int)
    canvas[first_r : first_r + first.shape[0], first_c : first_c + first.shape[1]] = first
    canvas[second_r : second_r + second.shape[0], second_c : second_c + second.shape[1]] = second
    assert get_bg(canvas) == BG
    return canvas


def _build_case() -> tuple[np.ndarray, np.ndarray, dict] | None:
    num_sprites = random.choice((2, 3, 3))
    anchor_colors = random.sample(PALETTE, num_sprites)
    body_pool = [color for color in PALETTE if color not in anchor_colors]

    source_turns = [random.randrange(4) for _ in range(num_sprites)]
    target_turns = [random.randrange(4) for _ in range(num_sprites)]
    rotations = [(dst - src) % 4 for src, dst in zip(source_turns, target_turns)]
    if all(turn == 0 for turn in rotations):
        return None
    if num_sprites == 3 and len(set(rotations)) == 1:
        return None

    source_shapes: list[list[tuple[int, int, int]]] = []
    target_shapes: list[list[tuple[int, int, int]]] = []
    target_dirs: list[tuple[int, int]] = []

    for anchor_color, src_turn, dst_turn in zip(anchor_colors, source_turns, target_turns):
        base = _gen_base_sprite(anchor_color, body_pool)
        if base is None:
            return None
        source_shapes.append(rotate_cells(base, src_turn))
        target_shapes.append(rotate_cells(base, dst_turn))
        target_dirs.append(rotate_offset(EAST[0], EAST[1], dst_turn))

    size_lo = 10 if num_sprites == 3 else 9
    for _ in range(PLACEMENT_RETRIES):
        source_shape = (random.randint(size_lo, 11), random.randint(size_lo, 11))
        target_shape = (random.randint(size_lo, 11), random.randint(size_lo, 11))
        source_anchors = _place_shapes(source_shape, source_shapes)
        target_anchors = _place_shapes(target_shape, target_shapes)
        if source_anchors is None or target_anchors is None:
            continue

        source_inner = np.full(source_shape, BG, dtype=int)
        target_inner = np.full(target_shape, BG, dtype=int)
        expected_inner = np.full(target_shape, BG, dtype=int)

        for cells, anchor in zip(source_shapes, source_anchors):
            _paint_cells(source_inner, cells, anchor)
        for anchor_color, anchor, direction, cells in zip(anchor_colors, target_anchors, target_dirs, target_shapes):
            ar, ac = anchor
            dr, dc = direction
            target_inner[ar, ac] = anchor_color
            target_inner[ar + dr, ac + dc] = GREY
            _paint_cells(expected_inner, cells, anchor)

        source_panel = _panel_with_frame(source_inner)
        target_panel = _panel_with_frame(target_inner)
        expected = _panel_with_frame(expected_inner)
        layout = random.choice(("horizontal", "vertical"))
        canvas = _compose_canvas(source_panel, target_panel, layout)

        try:
            actual = main(canvas)
        except AssertionError:
            continue
        if not np.array_equal(actual, expected):
            continue

        meta = {
            "rotations": rotations,
            "layout": layout,
            "num_sprites": num_sprites,
        }
        return canvas, expected, meta
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        case = _build_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid anchor-stamping input")
