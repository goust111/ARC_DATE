import random
from collections import Counter

import numpy as np


MAX_GRID_SIZE = 30
BACKGROUND = 0
GEN_RETRIES = 400
PANEL_SHAPES = [(4, 5), (4, 6), (5, 5), (5, 6)]
GEOMETRIES = [
    ("left", "top"),
    ("left", "bottom"),
    ("right", "top"),
    ("right", "bottom"),
]
SIDE_GAP_SLACK = (0, 2)
OUTER_MARGIN_RANGE = (1, 3)
BETWEEN_PAIR_GAP_RANGE = (3, 6)


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comp = np.zeros_like(mask, dtype=bool)
            rr, cc = zip(*cells)
            comp[list(rr), list(cc)] = True
            comps.append(comp)
    return comps


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("empty mask has no bbox")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(arr: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return arr[r0 : r1 + 1, c0 : c1 + 1]


def _full_rectangle(mask: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    return bool(np.all(mask[r0 : r1 + 1, c0 : c1 + 1]))


def _base_and_motif(panel: np.ndarray) -> tuple[int, np.ndarray]:
    colors, counts = np.unique(panel, return_counts=True)
    base = int(colors[np.argmax(counts)])
    return base, panel != base


def _core_mask(mask: np.ndarray) -> np.ndarray:
    up = np.zeros_like(mask, dtype=bool)
    down = np.zeros_like(mask, dtype=bool)
    left = np.zeros_like(mask, dtype=bool)
    right = np.zeros_like(mask, dtype=bool)
    up[1:] = mask[:-1]
    down[:-1] = mask[1:]
    left[:, 1:] = mask[:, :-1]
    right[:, :-1] = mask[:, 1:]
    return mask & (up | down) & (left | right)


def _find_panels(pair_grid: np.ndarray, bg: int) -> list[dict]:
    fg = pair_grid != bg
    panels = []
    for comp in _components(_core_mask(fg)):
        if int(comp.sum()) < 9:
            continue
        box = _bbox(comp)
        if not _full_rectangle(fg, box):
            continue
        panel = _crop(pair_grid, box)
        base, motif = _base_and_motif(panel)
        if not np.any(motif):
            continue
        panels.append(
            {
                "bbox": box,
                "grid": panel,
                "base": base,
                "motif": motif,
            }
        )
    if len(panels) != 2:
        raise AssertionError(f"expected 2 panels, found {len(panels)}")
    return panels


def _wire_info(pair_grid: np.ndarray, panel: dict, bg: int) -> dict:
    h, w = pair_grid.shape
    r0, c0, r1, c1 = panel["bbox"]
    sides = []
    if r0 > 0:
        cols = [c for c in range(c0, c1 + 1) if pair_grid[r0 - 1, c] != bg]
        if cols:
            sides.append(("top", cols))
    if r1 + 1 < h:
        cols = [c for c in range(c0, c1 + 1) if pair_grid[r1 + 1, c] != bg]
        if cols:
            sides.append(("bottom", cols))
    if c0 > 0:
        rows = [r for r in range(r0, r1 + 1) if pair_grid[r, c0 - 1] != bg]
        if rows:
            sides.append(("left", rows))
    if c1 + 1 < w:
        rows = [r for r in range(r0, r1 + 1) if pair_grid[r, c1 + 1] != bg]
        if rows:
            sides.append(("right", rows))
    if len(sides) != 1:
        raise AssertionError(f"expected exactly one wire side, got {sides}")
    side, positions = sides[0]
    if side in ("top", "bottom"):
        if len(positions) != 1:
            raise AssertionError("vertical wire must attach at exactly one column")
        col = positions[0]
        step = -1 if side == "top" else 1
        start = r0 - 1 if side == "top" else r1 + 1
        cells = []
        rr = start
        while 0 <= rr < h and pair_grid[rr, col] != bg:
            cells.append((rr, col))
            rr += step
        return {
            "orientation": "V",
            "side": side,
            "fixed": col,
            "step": step,
            "start": start,
            "cells": cells,
        }
    if len(positions) != 1:
        raise AssertionError("horizontal wire must attach at exactly one row")
    row = positions[0]
    step = -1 if side == "left" else 1
    start = c0 - 1 if side == "left" else c1 + 1
    cells = []
    cc = start
    while 0 <= cc < w and pair_grid[row, cc] != bg:
        cells.append((row, cc))
        cc += step
    return {
        "orientation": "H",
        "side": side,
        "fixed": row,
        "step": step,
        "start": start,
        "cells": cells,
    }


def _validate_path(pair_grid: np.ndarray, panel: dict, wire: dict, cross: tuple[int, int], other_base: int, bg: int) -> None:
    base = panel["base"]
    if wire["orientation"] == "H":
        row = wire["fixed"]
        c = wire["start"]
        while c != cross[1]:
            if pair_grid[row, c] != base:
                raise AssertionError("horizontal wire visibility broken before crossing")
            c += wire["step"]
        if pair_grid[cross] not in (base, other_base):
            raise AssertionError("crossing cell must show one of the two bases")
    else:
        col = wire["fixed"]
        r = wire["start"]
        while r != cross[0]:
            if pair_grid[r, col] != base:
                raise AssertionError("vertical wire visibility broken before crossing")
            r += wire["step"]
        if pair_grid[cross] not in (base, other_base):
            raise AssertionError("crossing cell must show one of the two bases")
    if pair_grid[cross] == bg:
        raise AssertionError("crossing cannot be background")


def _solve_pair(pair_grid: np.ndarray, bg: int) -> np.ndarray:
    panels = _find_panels(pair_grid, bg)
    wires = [_wire_info(pair_grid, panel, bg) for panel in panels]
    if {wire["orientation"] for wire in wires} != {"H", "V"}:
        raise AssertionError("a pair must contain one horizontal and one vertical wire")

    h_idx = 0 if wires[0]["orientation"] == "H" else 1
    v_idx = 1 - h_idx
    h_panel, v_panel = panels[h_idx], panels[v_idx]
    h_wire, v_wire = wires[h_idx], wires[v_idx]
    cross = (h_wire["fixed"], v_wire["fixed"])

    for panel in panels:
        r0, c0, r1, c1 = panel["bbox"]
        if r0 <= cross[0] <= r1 and c0 <= cross[1] <= c1:
            raise AssertionError("crossing must be outside both panels")
    if h_panel["base"] == v_panel["base"]:
        raise AssertionError("pair bases must differ to make source visible")

    _validate_path(pair_grid, h_panel, h_wire, cross, v_panel["base"], bg)
    _validate_path(pair_grid, v_panel, v_wire, cross, h_panel["base"], bg)

    source_panel = h_panel if pair_grid[cross] == h_panel["base"] else v_panel
    host_panel = v_panel if source_panel is h_panel else h_panel

    out = pair_grid.copy()
    hr0, hc0, hr1, hc1 = host_panel["bbox"]
    host_fill = np.full(host_panel["grid"].shape, host_panel["base"], dtype=int)
    host_fill[source_panel["motif"]] = source_panel["base"]
    out[hr0 : hr1 + 1, hc0 : hc1 + 1] = host_fill
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer each source panel's motif mask onto its paired host panel according to which wire color sits on top at the unique crossing.

    Input structure:
    - A dominant background surrounds several disconnected pair-components.
    - Each component contains exactly two same-sized filled rectangular panels, each with one dominant base color and a sparse motif inside.
    - From each panel edge, a one-cell-thick straight wire of the same base color reaches a single perpendicular crossing outside both panels.

    Rule:
    - Process each connected pair independently.
    - The panel whose base color is visible at the crossing is the source; the other panel is the host.
    - Keep both wires, the crossing, and the source panel unchanged; clear the host panel to its own base color, then stamp the source panel's motif mask into the host panel using the source panel's base color.

    Key insight:
    - The crossing is not decoration: its visible color reveals the local occlusion order, so the transfer direction must be inferred separately for every pair.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("grid must be 2D")

    bg = _get_bg(g)
    out = g.copy()
    for comp in _components(g != bg):
        box = _bbox(comp)
        pair_in = _crop(g, box)
        pair_out = _solve_pair(pair_in, bg)
        r0, c0, r1, c1 = box
        out[r0 : r1 + 1, c0 : c1 + 1] = pair_out
    return out


def _grow_mask(h: int, w: int, rng: random.Random) -> np.ndarray:
    inner_h, inner_w = h - 2, w - 2
    inner_area = inner_h * inner_w
    target = rng.randint(max(3, inner_area // 5), max(4, inner_area // 2))
    component_count = rng.choices([1, 2, 3], weights=[0.6, 0.25, 0.15])[0]
    cells = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    seeds = rng.sample(cells, component_count)
    mask = np.zeros((inner_h, inner_w), dtype=bool)
    frontier = []
    for cell in seeds:
        mask[cell] = True
        frontier.append(cell)
    while int(mask.sum()) < target:
        cell = frontier[rng.randrange(len(frontier))]
        nbrs = []
        for nr, nc in ((cell[0] - 1, cell[1]), (cell[0] + 1, cell[1]), (cell[0], cell[1] - 1), (cell[0], cell[1] + 1)):
            if 0 <= nr < inner_h and 0 <= nc < inner_w and not mask[nr, nc]:
                nbrs.append((nr, nc))
        if nbrs:
            pick = rng.choice(nbrs)
            mask[pick] = True
            frontier.append(pick)
        elif int(mask.sum()) < target:
            leftover = [(r, c) for r, c in cells if not mask[r, c]]
            if not leftover:
                break
            pick = rng.choice(leftover)
            mask[pick] = True
            frontier.append(pick)
    full = np.zeros((h, w), dtype=bool)
    full[1:-1, 1:-1] = mask
    return full


def _distinct_masks(h: int, w: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(100):
        a = _grow_mask(h, w, rng)
        b = _grow_mask(h, w, rng)
        if np.array_equal(a, b):
            continue
        if np.any(a & ~b) and np.any(b & ~a):
            return a, b
    raise RuntimeError("failed to create non-degenerate motif masks")


def _panel_with_motif(h: int, w: int, base: int, motif_color: int, motif_mask: np.ndarray) -> np.ndarray:
    panel = np.full((h, w), base, dtype=int)
    panel[motif_mask] = motif_color
    return panel


def _draw_h_line(grid: np.ndarray, row: int, c0: int, c1: int, color: int) -> None:
    lo, hi = sorted((c0, c1))
    grid[row, lo : hi + 1] = color


def _draw_v_line(grid: np.ndarray, col: int, r0: int, r1: int, color: int) -> None:
    lo, hi = sorted((r0, r1))
    grid[lo : hi + 1, col] = color


def _build_pair(
    panel_shape: tuple[int, int],
    geometry: tuple[str, str],
    horizontal_on_top: bool,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray, dict]:
    h, w = panel_shape
    h_side, v_side = geometry
    h_attach = rng.randint(1, h - 2)
    v_attach = rng.randint(1, w - 2)
    min_d_h = v_attach + 1 if h_side == "left" else w - v_attach
    min_d_v = h_attach + 1 if v_side == "top" else h - h_attach
    d_h = min_d_h + rng.randint(*SIDE_GAP_SLACK)
    d_v = min_d_v + rng.randint(*SIDE_GAP_SLACK)

    cross_r = 0
    cross_c = 0
    h_r0 = cross_r - h_attach
    h_c0 = cross_c - d_h - (w - 1) if h_side == "left" else cross_c + d_h
    v_r0 = cross_r - d_v - (h - 1) if v_side == "top" else cross_r + d_v
    v_c0 = cross_c - v_attach

    h_box = (h_r0, h_c0, h_r0 + h - 1, h_c0 + w - 1)
    v_box = (v_r0, v_c0, v_r0 + h - 1, v_c0 + w - 1)
    min_r = min(0, h_box[0], v_box[0])
    min_c = min(0, h_box[1], v_box[1])
    shift_r = -min_r
    shift_c = -min_c
    cross = (cross_r + shift_r, cross_c + shift_c)
    h_box = tuple(v + (shift_r if i % 2 == 0 else shift_c) for i, v in enumerate(h_box))
    v_box = tuple(v + (shift_r if i % 2 == 0 else shift_c) for i, v in enumerate(v_box))
    patch_h = max(cross[0], h_box[2], v_box[2]) + 1
    patch_w = max(cross[1], h_box[3], v_box[3]) + 1

    bases = rng.sample(range(1, 10), 4)
    h_base, v_base, h_motif_color, v_motif_color = bases
    h_mask, v_mask = _distinct_masks(h, w, rng)

    patch = np.full((patch_h, patch_w), BACKGROUND, dtype=int)
    hr0, hc0, hr1, hc1 = h_box
    vr0, vc0, vr1, vc1 = v_box
    patch[hr0 : hr1 + 1, hc0 : hc1 + 1] = _panel_with_motif(h, w, h_base, h_motif_color, h_mask)
    patch[vr0 : vr1 + 1, vc0 : vc1 + 1] = _panel_with_motif(h, w, v_base, v_motif_color, v_mask)

    if h_side == "left":
        _draw_h_line(patch, cross[0], hc1 + 1, cross[1] - 1, h_base)
    else:
        _draw_h_line(patch, cross[0], cross[1] + 1, hc0 - 1, h_base)
    if v_side == "top":
        _draw_v_line(patch, cross[1], vr1 + 1, cross[0] - 1, v_base)
    else:
        _draw_v_line(patch, cross[1], cross[0] + 1, vr0 - 1, v_base)
    patch[cross] = h_base if horizontal_on_top else v_base

    expected = patch.copy()
    if horizontal_on_top:
        host_box = v_box
        host_base = v_base
        src_base = h_base
        src_mask = h_mask
    else:
        host_box = h_box
        host_base = h_base
        src_base = v_base
        src_mask = v_mask
    rr0, cc0, rr1, cc1 = host_box
    host = np.full((h, w), host_base, dtype=int)
    host[src_mask] = src_base
    expected[rr0 : rr1 + 1, cc0 : cc1 + 1] = host

    info = {
        "cross": cross,
        "bases": (h_base, v_base),
        "horizontal_on_top": horizontal_on_top,
        "host_box": host_box,
        "source_mask": src_mask,
        "host_mask_before": v_mask if horizontal_on_top else h_mask,
        "expected": expected,
    }
    return patch, expected, info


def _compose_two_pairs(pair_a: np.ndarray, pair_b: np.ndarray, rng: random.Random) -> np.ndarray:
    layouts = ["horizontal", "vertical"]
    rng.shuffle(layouts)
    top_margin = rng.randint(*OUTER_MARGIN_RANGE)
    left_margin = rng.randint(*OUTER_MARGIN_RANGE)
    bottom_margin = rng.randint(*OUTER_MARGIN_RANGE)
    right_margin = rng.randint(*OUTER_MARGIN_RANGE)
    gap = rng.randint(*BETWEEN_PAIR_GAP_RANGE)

    for layout in layouts:
        if layout == "horizontal":
            h = max(pair_a.shape[0], pair_b.shape[0]) + top_margin + bottom_margin
            w = pair_a.shape[1] + pair_b.shape[1] + gap + left_margin + right_margin
            if h > MAX_GRID_SIZE or w > MAX_GRID_SIZE:
                continue
            grid = np.full((h, w), BACKGROUND, dtype=int)
            max_ra = h - bottom_margin - pair_a.shape[0]
            max_rb = h - bottom_margin - pair_b.shape[0]
            ra = rng.randint(top_margin, max_ra)
            rb = rng.randint(top_margin, max_rb)
            ca = left_margin
            cb = left_margin + pair_a.shape[1] + gap
            grid[ra : ra + pair_a.shape[0], ca : ca + pair_a.shape[1]] = pair_a
            grid[rb : rb + pair_b.shape[0], cb : cb + pair_b.shape[1]] = pair_b
            return grid

        h = pair_a.shape[0] + pair_b.shape[0] + gap + top_margin + bottom_margin
        w = max(pair_a.shape[1], pair_b.shape[1]) + left_margin + right_margin
        if h > MAX_GRID_SIZE or w > MAX_GRID_SIZE:
            continue
        grid = np.full((h, w), BACKGROUND, dtype=int)
        max_ca = w - right_margin - pair_a.shape[1]
        max_cb = w - right_margin - pair_b.shape[1]
        ca = rng.randint(left_margin, max_ca)
        cb = rng.randint(left_margin, max_cb)
        ra = top_margin
        rb = top_margin + pair_a.shape[0] + gap
        grid[ra : ra + pair_a.shape[0], ca : ca + pair_a.shape[1]] = pair_a
        grid[rb : rb + pair_b.shape[0], cb : cb + pair_b.shape[1]] = pair_b
        return grid
    raise RuntimeError("no valid canvas layout fits within 30x30")


def _stamp_without_clearing(host_before: np.ndarray, source_mask: np.ndarray, source_base: int) -> np.ndarray:
    host_after = host_before.copy()
    host_after[source_mask] = source_base
    return host_after


def _validate_case(inp: np.ndarray, expected: np.ndarray, pair_infos: list[dict]) -> None:
    actual = main(inp)
    if not np.array_equal(actual, expected):
        raise AssertionError("generated input does not match its constructed output")
    if np.array_equal(inp, expected):
        raise AssertionError("generator produced a no-op transformation")

    saw_horizontal_source = any(info["horizontal_on_top"] for info in pair_infos)
    saw_vertical_source = any(not info["horizontal_on_top"] for info in pair_infos)
    if not (saw_horizontal_source and saw_vertical_source):
        raise AssertionError("branch coverage missing: both source directions must appear")

    for info in pair_infos:
        rr0, cc0, rr1, cc1 = info["host_box"]
        host_before = inp[rr0 : rr1 + 1, cc0 : cc1 + 1]
        host_after = expected[rr0 : rr1 + 1, cc0 : cc1 + 1]
        source_base = info["bases"][0] if info["horizontal_on_top"] else info["bases"][1]
        overlay_only = _stamp_without_clearing(host_before, info["source_mask"], source_base)
        if np.array_equal(overlay_only, host_after):
            raise AssertionError("host clearing step became redundant")

        flipped = inp.copy()
        cross = info["cross"]
        flipped[cross] = info["bases"][1] if info["horizontal_on_top"] else info["bases"][0]
        flipped_out = main(flipped)
        if np.array_equal(flipped_out, expected):
            raise AssertionError("crossing color did not uniquely determine the source")


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = random.Random()
        panel_shape = rng.choice(PANEL_SHAPES)
        geometries = rng.sample(GEOMETRIES, 2)
        if geometries[0][0] == geometries[1][0]:
            geometries[1] = ({"left": "right", "right": "left"}[geometries[1][0]], geometries[1][1])
        pair_a, out_a, info_a = _build_pair(panel_shape, geometries[0], True, rng)
        pair_b, out_b, info_b = _build_pair(panel_shape, geometries[1], False, rng)
        if max(pair_a.shape + pair_b.shape) > 20:
            continue
        try:
            inp = _compose_two_pairs(pair_a, pair_b, rng)
            expected = _compose_two_pairs(out_a, out_b, rng)
        except RuntimeError:
            continue

        # Rebuild expected on the same placement by using the solved output.
        expected = main(inp)
        pair_infos = []
        bg = _get_bg(inp)
        for comp in _components(inp != bg):
            box = _bbox(comp)
            pair_grid = _crop(inp, box)
            panels = _find_panels(pair_grid, bg)
            wires = [_wire_info(pair_grid, panel, bg) for panel in panels]
            h_idx = 0 if wires[0]["orientation"] == "H" else 1
            v_idx = 1 - h_idx
            h_panel, v_panel = panels[h_idx], panels[v_idx]
            cross = (wires[h_idx]["fixed"], wires[v_idx]["fixed"])
            horizontal_on_top = pair_grid[cross] == h_panel["base"]
            host_panel = v_panel if horizontal_on_top else h_panel
            pair_infos.append(
                {
                    "cross": (box[0] + cross[0], box[1] + cross[1]),
                    "bases": (h_panel["base"], v_panel["base"]),
                    "horizontal_on_top": horizontal_on_top,
                    "host_box": (
                        box[0] + host_panel["bbox"][0],
                        box[1] + host_panel["bbox"][1],
                        box[0] + host_panel["bbox"][2],
                        box[1] + host_panel["bbox"][3],
                    ),
                    "source_mask": h_panel["motif"] if horizontal_on_top else v_panel["motif"],
                }
            )
        try:
            _validate_case(inp, expected, pair_infos)
        except Exception:
            continue
        return inp
    raise RuntimeError("failed to generate a valid occlusion-transfer input")
