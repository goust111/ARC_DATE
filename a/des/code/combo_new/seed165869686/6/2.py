from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
DIVIDER = 5
PALETTE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_ATTEMPTS = 400


def _separator_rows(grid: np.ndarray) -> list[int]:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == grid[r, 0]) and int(grid[r, 0]) != BG]
    assert rows, "expected horizontal separator rows"
    colors = {int(grid[r, 0]) for r in rows}
    assert len(colors) == 1, "all separators must share one color"
    return rows


def _panel_ranges(height: int, sep_rows: list[int]) -> list[tuple[int, int]]:
    bounds = [-1, *sep_rows, height]
    ranges = [(a + 1, b) for a, b in zip(bounds, bounds[1:])]
    panel_heights = {r1 - r0 for r0, r1 in ranges}
    assert len(panel_heights) == 1, "panels must have equal height"
    assert next(iter(panel_heights)) > 0, "panel height must be positive"
    return ranges


def _components(panel: np.ndarray) -> list[dict]:
    h, w = panel.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(panel[r, c])
            if color == BG or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(panel[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            pts = np.array(coords, dtype=int)
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            mask[pts[:, 0] - r0, pts[:, 1] - c0] = True
            comps.append(
                {
                    "color": color,
                    "top": int(r0),
                    "left": int(c0),
                    "mask": mask,
                }
            )
    return comps


def _paste_mask(panel: np.ndarray, top: int, left: int, color: int, mask: np.ndarray) -> None:
    h, w = mask.shape
    region = panel[top : top + h, left : left + w]
    assert region.shape == mask.shape, "paste would exceed panel bounds"
    assert not np.any(region[mask] != BG), "objects may not overlap"
    region[mask] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder each panel's colored objects by the palette row shown above the separators.

    Input structure:
    - The background is black, and one-pixel horizontal separator lines split
      the grid into equal-height horizontal panels.
    - The top panel is a palette legend: several distinct single-pixel colors
      appear from left to right on one row and define a global color order.
    - Every lower panel contains several 4-connected monochrome foreground
      objects whose colors come from a subset of the palette colors.

    Rule:
    - Read the palette colors from left to right in the top panel.
    - For each non-palette panel, keep every object's exact shape and top row,
      but clear the panel and place the objects back from left to right in
      palette order, leaving exactly one black column between consecutive
      objects.
    - Leave the palette panel and the separator rows unchanged.

    Key insight:
    - The top panel is not another scene panel; it is a legend that defines the
      ordering key used to horizontally sort the lower-panel objects.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_rows = _separator_rows(g)
    panel_ranges = _panel_ranges(g.shape[0], sep_rows)
    assert len(panel_ranges) >= 2, "expected a palette panel and at least one object panel"

    palette_r0, palette_r1 = panel_ranges[0]
    palette_panel = g[palette_r0:palette_r1, :]
    palette_pixels = [
        (c, int(palette_panel[r, c]))
        for r in range(palette_panel.shape[0])
        for c in range(palette_panel.shape[1])
        if int(palette_panel[r, c]) != BG
    ]
    assert palette_pixels, "palette panel must contain colored pixels"
    palette_pixels.sort(key=lambda item: item[0])
    palette_order = [color for _c, color in palette_pixels]
    assert len(set(palette_order)) == len(palette_order), "palette colors must be unique"
    palette_index = {color: idx for idx, color in enumerate(palette_order)}

    out = g.copy()
    for r0, r1 in panel_ranges[1:]:
        panel = g[r0:r1, :]
        objects = _components(panel)
        assert objects, "each lower panel must contain at least one object"
        assert len({obj["color"] for obj in objects}) == len(objects), "each panel should use distinct object colors"
        assert all(obj["color"] in palette_index for obj in objects), "panel colors must come from the palette"

        objects.sort(key=lambda obj: palette_index[obj["color"]])
        repacked = np.full_like(panel, BG)
        cursor = 0
        for idx, obj in enumerate(objects):
            mask = obj["mask"]
            _paste_mask(repacked, obj["top"], cursor, obj["color"], mask)
            cursor += mask.shape[1]
            if idx != len(objects) - 1:
                cursor += 1
        assert cursor <= panel.shape[1], "repacked panel must fit inside the original width"
        out[r0:r1, :] = repacked

    return out


def _split_slack(total: int, slots: int) -> list[int]:
    parts = [0] * slots
    for _ in range(total):
        parts[random.randrange(slots)] += 1
    return parts


def _layout_positions(widths: list[int], total_width: int) -> list[int]:
    base = sum(widths) + max(0, len(widths) - 1)
    assert base <= total_width, "items do not fit"
    extras = _split_slack(total_width - base, len(widths) + 1)
    positions = []
    x = extras[0]
    for idx, width in enumerate(widths):
        positions.append(x)
        x += width
        if idx != len(widths) - 1:
            x += 1 + extras[idx + 1]
    return positions


def _random_mask(max_height: int) -> np.ndarray:
    raw_h = random.randint(1, min(4, max_height))
    raw_w = random.randint(1, 4)
    area = random.randint(1, raw_h * raw_w)
    mask = np.zeros((raw_h, raw_w), dtype=bool)
    start = (random.randrange(raw_h), random.randrange(raw_w))
    mask[start] = True
    frontier = [start]
    while int(mask.sum()) < area:
        border = set()
        for r, c in frontier:
            for dr, dc in NEIGHBORS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < raw_h and 0 <= nc < raw_w and not mask[nr, nc]:
                    border.add((nr, nc))
        if not border:
            break
        cell = random.choice(tuple(border))
        mask[cell] = True
        frontier.append(cell)

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _make_panel_objects(panel_h: int, palette: list[int]) -> list[dict]:
    count = random.randint(2, len(palette))
    colors = random.sample(palette, count)
    objects = []
    for color in colors:
        mask = _random_mask(panel_h)
        top = random.randint(0, panel_h - mask.shape[0])
        objects.append({"color": int(color), "top": int(top), "mask": mask})
    return objects


def _render_panel(panel_h: int, panel_w: int, ordered_objects: list[dict]) -> np.ndarray:
    panel = np.full((panel_h, panel_w), BG, dtype=int)
    widths = [obj["mask"].shape[1] for obj in ordered_objects]
    xs = _layout_positions(widths, panel_w)
    for obj, x in zip(ordered_objects, xs):
        _paste_mask(panel, obj["top"], x, obj["color"], obj["mask"])
    return panel


def _gen_attempt() -> np.ndarray | None:
    total_panels = random.randint(3, 5)
    max_panel_h = min(7, (30 - (total_panels - 1)) // total_panels)
    if max_panel_h < 4:
        return None
    panel_h = random.randint(4, max_panel_h)
    height = total_panels * panel_h + (total_panels - 1)
    if height > 30:
        return None

    palette_size = random.randint(3, 5)
    palette = random.sample(PALETTE_COLORS, palette_size)
    palette_index = {color: idx for idx, color in enumerate(palette)}

    panel_specs = []
    min_width = 2 * palette_size - 1
    changed = False
    for _ in range(total_panels - 1):
        objects = _make_panel_objects(panel_h, palette)
        sorted_colors = [obj["color"] for obj in sorted(objects, key=lambda obj: palette_index[obj["color"]])]
        input_objects = list(objects)
        random.shuffle(input_objects)
        input_colors = [obj["color"] for obj in input_objects]
        if input_colors != sorted_colors:
            changed = True
        min_width = max(min_width, sum(obj["mask"].shape[1] for obj in objects) + len(objects) - 1)
        panel_specs.append({"objects": objects, "input_objects": input_objects})

    if not changed:
        for spec in panel_specs:
            if len(spec["input_objects"]) >= 2:
                spec["input_objects"][0], spec["input_objects"][1] = spec["input_objects"][1], spec["input_objects"][0]
                break

    if min_width > 30:
        return None
    panel_w = random.randint(min_width, min(30, min_width + 10))

    grid = np.full((height, panel_w), BG, dtype=int)
    for idx in range(total_panels - 1):
        sep_r = (idx + 1) * panel_h + idx
        grid[sep_r, :] = DIVIDER

    palette_row = random.randrange(panel_h)
    palette_xs = _layout_positions([1] * palette_size, panel_w)
    for x, color in zip(palette_xs, palette):
        grid[palette_row, x] = int(color)

    for panel_idx, spec in enumerate(panel_specs, start=1):
        r0 = panel_idx * (panel_h + 1)
        r1 = r0 + panel_h
        grid[r0:r1, :] = _render_panel(panel_h, panel_w, spec["input_objects"])

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
