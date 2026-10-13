from __future__ import annotations

import random

import numpy as np


BG = 0
BORDER = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]

INNER_H_RANGE = (6, 11)
INNER_W_RANGE = (6, 11)
GAP_WIDTH_RANGE = (1, 3)
OBJECT_COUNT_RANGE = (1, 3)
SPRITE_MAX_H = 4
SPRITE_MAX_W = 4
SPRITE_AREA_RANGE = (1, 6)
PLACEMENT_PADDING = 1
GEN_ATTEMPTS = 400


def _components(grid: np.ndarray, forbidden: set[int]) -> list[dict]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color in forbidden:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "area": len(coords),
                }
            )
    return comps


def _panel_bounds(shape: tuple[int, int]) -> dict[str, tuple[int, int, int, int]]:
    h, w = map(int, shape)
    assert h >= 9 and w >= 9 and h % 2 == 1 and w % 2 == 1, "grid must support four equal panels"
    mid_r = (h - 1) // 2
    mid_c = (w - 1) // 2
    assert 1 < mid_r < h - 2 and 1 < mid_c < w - 2, "panel interiors must be nonempty"
    return {
        "tl": (1, mid_r, 1, mid_c),
        "tr": (1, mid_r, mid_c + 1, w - 1),
        "bl": (mid_r + 1, h - 1, 1, mid_c),
        "br": (mid_r + 1, h - 1, mid_c + 1, w - 1),
    }


def _corner_starts(height: int, width: int) -> list[tuple[int, int]]:
    return [(0, 0), (0, width - 2), (height - 2, 0), (height - 2, width - 2)]


def _legend(panel: np.ndarray) -> dict | None:
    h, w = map(int, panel.shape)
    legends = []
    for comp in _components(panel, {BG, BORDER}):
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if comp["area"] == 4 and r1 - r0 + 1 == 2 and c1 - c0 + 1 == 2 and (r0, c0) in _corner_starts(h, w):
            legends.append(
                {
                    "color": int(comp["color"]),
                    "cells": [(r, c) for r, c in comp["coords"]],
                    "bbox": (r0, c0, r1, c1),
                }
            )
    assert len(legends) <= 1, "each panel may contain at most one legend block"
    return legends[0] if legends else None


def _pair_mode(grid: np.ndarray, bounds: dict[str, tuple[int, int, int, int]]) -> str:
    mid_r = bounds["tl"][1]
    mid_c = bounds["tl"][3]

    top_r0, top_r1, _, _ = bounds["tl"]
    bot_r0, bot_r1, _, _ = bounds["bl"]
    _, _, left_c0, left_c1 = bounds["tl"]
    _, _, right_c0, right_c1 = bounds["tr"]

    vertical_gaps = [
        np.any(grid[top_r0:top_r1, mid_c] == BG),
        np.any(grid[bot_r0:bot_r1, mid_c] == BG),
    ]
    horizontal_gaps = [
        np.any(grid[mid_r, left_c0:left_c1] == BG),
        np.any(grid[mid_r, right_c0:right_c1] == BG),
    ]

    has_vertical = any(vertical_gaps)
    has_horizontal = any(horizontal_gaps)
    assert has_vertical != has_horizontal, "gaps must define exactly one pairing mode"
    if has_vertical:
        assert all(vertical_gaps) and not has_horizontal, "both row pairs must share vertical-border gaps"
        return "horizontal"
    assert all(horizontal_gaps) and not has_vertical, "both column pairs must share horizontal-border gaps"
    return "vertical"


def _project_local(coords: list[tuple[int, int]], mode: str, height: int, width: int) -> list[tuple[int, int]]:
    if mode == "horizontal":
        return [(r, width - 1 - c) for r, c in coords]
    return [(height - 1 - r, c) for r, c in coords]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each source-panel object through the aligned border gap into its neighboring legend panel, then recolor it by that panel's legend.

    Input structure:
    - The background is black, and single-pixel grey borders divide the grid into four equal panels arranged as a 2x2 layout.
    - Each panel has exactly one border gap on a shared separator with a neighboring panel, and exactly two panels contain a 2x2 monochrome legend block in one inner corner.
    - The non-legend panels contain one or more monochrome 4-connected foreground objects inside their bordered interiors.

    Rule:
    - Use the aligned opposite gaps to pair neighboring panels; each pair contains one source panel without a legend and one target panel with a legend.
    - Erase every source-panel object and erase both legend blocks.
    - Mirror each source object across the shared panel boundary into the paired target panel, keeping its offset from the gap-aligned border, then recolor the projected object to the target legend color.

    Key insight:
    - The gap identifies which adjacent panels communicate, while the corner 2x2 block is a legend that supplies only the target recolor and is not kept in the output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    border = int(g[0, 0])
    assert border != BG, "outer frame must be non-background"
    assert np.all(g[0, :] == border) and np.all(g[-1, :] == border), "top and bottom borders must be solid"
    assert np.all(g[:, 0] == border) and np.all(g[:, -1] == border), "left and right borders must be solid"

    bounds = _panel_bounds(g.shape)
    panels = {name: g[r0:r1, c0:c1].copy() for name, (r0, r1, c0, c1) in bounds.items()}
    legends = {name: _legend(panel) for name, panel in panels.items()}

    legend_panels = [name for name, info in legends.items() if info is not None]
    assert len(legend_panels) == 2, "expected exactly two legend panels"

    mode = _pair_mode(g, bounds)
    pair_names = [("tl", "tr"), ("bl", "br")] if mode == "horizontal" else [("tl", "bl"), ("tr", "br")]

    out = g.copy()

    for name in legend_panels:
        r0, _r1, c0, _c1 = bounds[name]
        for lr, lc in legends[name]["cells"]:
            out[r0 + lr, c0 + lc] = BG

    for a, b in pair_names:
        src_name, dst_name = (a, b) if legends[a] is None else (b, a)
        assert legends[src_name] is None and legends[dst_name] is not None, "each pair must have one source and one target"

        src_panel = panels[src_name].copy()
        src_comps = _components(src_panel, {BG, border})

        sr0, sr1, sc0, sc1 = bounds[src_name]
        dr0, _dr1, dc0, _dc1 = bounds[dst_name]
        panel_h = int(sr1 - sr0)
        panel_w = int(sc1 - sc0)
        dst_color = int(legends[dst_name]["color"])

        src_view = out[sr0:sr1, sc0:sc1]
        src_view[src_panel != BG] = BG

        for comp in src_comps:
            projected = _project_local(comp["coords"], mode, panel_h, panel_w)
            for rr, cc in projected:
                out[dr0 + rr, dc0 + cc] = dst_color

    return out


def _random_sprite() -> np.ndarray:
    max_h = SPRITE_MAX_H
    max_w = SPRITE_MAX_W
    h = random.randint(1, max_h)
    w = random.randint(1, max_w)
    area = random.randint(SPRITE_AREA_RANGE[0], min(SPRITE_AREA_RANGE[1], h * w))

    seed = (random.randrange(h), random.randrange(w))
    cells = {seed}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        cells.add(random.choice(frontier))

    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)

    sprite = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in cells:
        sprite[r - r0, c - c0] = 1
    return sprite


def _can_place(panel: np.ndarray, sprite: np.ndarray, top: int, left: int) -> bool:
    sh, sw = map(int, sprite.shape)
    h, w = map(int, panel.shape)
    if top + sh > h or left + sw > w:
        return False

    r0 = max(0, top - PLACEMENT_PADDING)
    r1 = min(h, top + sh + PLACEMENT_PADDING)
    c0 = max(0, left - PLACEMENT_PADDING)
    c1 = min(w, left + sw + PLACEMENT_PADDING)
    return bool(np.all(panel[r0:r1, c0:c1] == BG))


def _place_sprite(panel: np.ndarray, sprite: np.ndarray, color: int) -> bool:
    h, w = map(int, panel.shape)
    sh, sw = map(int, sprite.shape)
    positions = [(r, c) for r in range(h - sh + 1) for c in range(w - sw + 1)]
    random.shuffle(positions)
    for top, left in positions:
        if not _can_place(panel, sprite, top, left):
            continue
        patch = panel[top : top + sh, left : left + sw]
        patch[sprite.astype(bool)] = int(color)
        return True
    return False


def _random_gap_segment(length: int) -> tuple[int, int]:
    max_width = min(GAP_WIDTH_RANGE[1], length - 2)
    width = random.randint(GAP_WIDTH_RANGE[0], max_width)
    start = random.randint(0, length - width)
    return start, width


def _panel_has_fake_legend(panel: np.ndarray) -> bool:
    return _legend(panel) is not None


def _fill_source_panel(panel: np.ndarray, colors: list[int]) -> int | None:
    count = random.randint(OBJECT_COUNT_RANGE[0], min(OBJECT_COUNT_RANGE[1], len(colors)))
    for color in colors[:count]:
        for _ in range(40):
            sprite = _random_sprite()
            if _place_sprite(panel, sprite, color):
                break
        else:
            return None
    return count if not _panel_has_fake_legend(panel) else None


def _add_legend(panel: np.ndarray, color: int) -> None:
    h, w = map(int, panel.shape)
    corners = _corner_starts(h, w)
    top, left = random.choice(corners)
    panel[top : top + 2, left : left + 2] = int(color)


def _assemble_grid(
    tl: np.ndarray,
    tr: np.ndarray,
    bl: np.ndarray,
    br: np.ndarray,
    pair_mode: str,
    gap_specs: list[tuple[int, int]],
) -> np.ndarray:
    inner_h, inner_w = map(int, tl.shape)
    h = 2 * inner_h + 3
    w = 2 * inner_w + 3
    grid = np.full((h, w), BG, dtype=int)

    grid[0, :] = BORDER
    grid[-1, :] = BORDER
    grid[:, 0] = BORDER
    grid[:, -1] = BORDER
    mid_r = inner_h + 1
    mid_c = inner_w + 1
    grid[mid_r, :] = BORDER
    grid[:, mid_c] = BORDER

    grid[1:mid_r, 1:mid_c] = tl
    grid[1:mid_r, mid_c + 1 : -1] = tr
    grid[mid_r + 1 : -1, 1:mid_c] = bl
    grid[mid_r + 1 : -1, mid_c + 1 : -1] = br

    if pair_mode == "horizontal":
        for idx, (start, width) in enumerate(gap_specs):
            base = 1 if idx == 0 else mid_r + 1
            grid[base + start : base + start + width, mid_c] = BG
    else:
        for idx, (start, width) in enumerate(gap_specs):
            base = 1 if idx == 0 else mid_c + 1
            grid[mid_r, base + start : base + start + width] = BG

    return grid


def _sample_case() -> np.ndarray | None:
    inner_h = random.randint(*INNER_H_RANGE)
    inner_w = random.randint(*INNER_W_RANGE)

    legend_colors = random.sample(COLOR_POOL, 2)
    object_colors = [c for c in COLOR_POOL if c not in legend_colors]
    random.shuffle(object_colors)

    pair_mode = random.choice(["horizontal", "vertical"])
    gap_length = inner_h if pair_mode == "horizontal" else inner_w
    gap_specs = [_random_gap_segment(gap_length), _random_gap_segment(gap_length)]

    panels = {name: np.full((inner_h, inner_w), BG, dtype=int) for name in ("tl", "tr", "bl", "br")}

    pairs = [("tl", "tr"), ("bl", "br")] if pair_mode == "horizontal" else [("tl", "bl"), ("tr", "br")]
    color_cursor = 0

    for pair_index, pair in enumerate(pairs):
        target = random.choice(pair)
        source = pair[1] if target == pair[0] else pair[0]

        _add_legend(panels[target], legend_colors[pair_index])

        remaining = object_colors[color_cursor:]
        if not remaining:
            return None
        used = _fill_source_panel(panels[source], remaining)
        if used is None:
            return None
        color_cursor += used

        if np.any(panels[target] != BG) and _legend(panels[target]) is None:
            return None
        if _legend(panels[source]) is not None:
            return None

    grid = _assemble_grid(panels["tl"], panels["tr"], panels["bl"], panels["br"], pair_mode, gap_specs)
    out = main(grid)

    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _sample_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
