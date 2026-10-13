from __future__ import annotations

import itertools
import random
from collections import Counter
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
if _ARC_UTILS_PATH.exists():
    try:
        _arc_utils = SourceFileLoader("_shared_arc_utils_7", str(_ARC_UTILS_PATH)).load_module()
        get_bg = _arc_utils.get_bg
        components_mask = _arc_utils.components_mask
        paste = _arc_utils.paste
    except Exception:  # pragma: no cover - fallback kept for standalone robustness
        _arc_utils = None
else:  # pragma: no cover - fallback kept for standalone robustness
    _arc_utils = None


if _arc_utils is None:
    def get_bg(grid: np.ndarray) -> int:
        return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


    def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
        mask = np.asarray(mask, dtype=bool)
        h, w = mask.shape
        seen = np.zeros_like(mask, dtype=bool)
        if int(connectivity) == 4:
            steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        else:
            steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
        comps: list[np.ndarray] = []
        for r in range(h):
            for c in range(w):
                if seen[r, c] or not mask[r, c]:
                    continue
                stack = [(r, c)]
                seen[r, c] = True
                pts: list[tuple[int, int]] = []
                while stack:
                    rr, cc = stack.pop()
                    pts.append((rr, cc))
                    for dr, dc in steps:
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
                comps.append(np.array(pts, dtype=int))
        return comps


    def paste(
        dst: np.ndarray,
        patch: np.ndarray,
        r0: int,
        c0: int,
        *,
        transparent: int | None = None,
    ) -> np.ndarray:
        h, w = patch.shape
        view = dst[int(r0) : int(r0) + h, int(c0) : int(c0) + w]
        if transparent is None:
            view[:, :] = patch
        else:
            mask = patch != int(transparent)
            view[mask] = patch[mask]
        return dst


BG = 0
MAX_GENERATION_ATTEMPTS = 500
PANEL_COUNT_RANGE = (3, 5)
GRID_HEIGHT_RANGE = (8, 14)
PANEL_WIDTH_RANGE = (3, 5)
SEPARATOR_WIDTH = 1
PALETTE = tuple(range(1, 10))


def _separator_info(grid: np.ndarray) -> tuple[int, list[int], int]:
    h, w = map(int, grid.shape)
    full_cols: dict[int, list[int]] = {}
    for c in range(w):
        if bool(np.all(grid[:, c] == grid[0, c])):
            full_cols.setdefault(int(grid[0, c]), []).append(int(c))

    candidates: list[tuple[int, list[int], int]] = []
    for color, cols in full_cols.items():
        if len(cols) < 2:
            continue
        widths = [cols[0], *[b - a - 1 for a, b in zip(cols, cols[1:])], w - cols[-1] - 1]
        if min(widths) <= 0:
            continue
        if len(set(widths)) != 1:
            continue
        panel_w = widths[0]
        if len(widths) < PANEL_COUNT_RANGE[0]:
            continue
        if len(cols) != len(widths) - 1:
            continue
        expected_cols = [panel_w + i * (panel_w + SEPARATOR_WIDTH) for i in range(len(cols))]
        if cols != expected_cols:
            continue
        candidates.append((int(color), cols, int(panel_w)))

    assert len(candidates) == 1, "expected exactly one regular separator-column color"
    return candidates[0]


def _panel_bounds(width: int, separator_cols: list[int]) -> list[tuple[int, int]]:
    bounds: list[tuple[int, int]] = []
    start = 0
    for sep_c in separator_cols:
        bounds.append((int(start), int(sep_c)))
        start = int(sep_c) + SEPARATOR_WIDTH
    bounds.append((int(start), int(width)))
    return bounds


def _terrain_color(panel: np.ndarray, bg: int) -> int:
    vals = panel[panel != int(bg)]
    assert vals.size > 0, "panel must contain foreground pixels"
    return int(Counter(map(int, vals.reshape(-1))).most_common(1)[0][0])


def _column_height(panel: np.ndarray, color: int, col: int) -> int:
    rows = np.where(panel[:, int(col)] == int(color))[0]
    assert rows.size > 0, "every terrain column must have positive height"
    height = int(rows.size)
    start = panel.shape[0] - height
    assert rows[0] == start and rows[-1] == panel.shape[0] - 1
    assert np.all(panel[start:, int(col)] == int(color))
    return height


def _extract_panels(grid: np.ndarray) -> tuple[int, int, list[int], list[tuple[int, int]], list[dict], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))
    sep_color, separator_cols, panel_w = _separator_info(g)
    bounds = _panel_bounds(g.shape[1], separator_cols)
    assert len({c1 - c0 for c0, c1 in bounds}) == 1 and bounds[0][1] - bounds[0][0] == panel_w

    comps = components_mask((g != int(bg)) & (g != int(sep_color)), connectivity=4)
    marker_candidates = [pts for pts in comps if int(pts.shape[0]) == 1]
    assert len(marker_candidates) == 1, "expected exactly one isolated marker pixel"
    marker_r, marker_c = map(int, marker_candidates[0][0])

    panels: list[dict] = []
    marked_panel = None
    for idx, (c0, c1) in enumerate(bounds):
        panel = g[:, int(c0) : int(c1)].copy()
        color = _terrain_color(panel, bg)
        heights = tuple(_column_height(panel, color, local_c) for local_c in range(panel.shape[1]))
        panels.append(
            {
                "index": int(idx),
                "grid": panel,
                "color": int(color),
                "heights": heights,
                "left": int(heights[0]),
                "right": int(heights[-1]),
                "peak": int(max(heights)),
                "area": int(sum(heights)),
            }
        )
        if c0 <= marker_c < c1:
            marked_panel = int(idx)

    assert marked_panel is not None, "marker must sit above exactly one panel"
    return bg, sep_color, separator_cols, bounds, panels, marked_panel


def _chain_order(panels: list[dict], start_idx: int) -> list[int]:
    remaining = set(range(len(panels)))
    order = [int(start_idx)]
    remaining.remove(int(start_idx))
    while remaining:
        right = int(panels[order[-1]]["right"])
        nxt = [idx for idx in remaining if int(panels[idx]["left"]) == right]
        assert len(nxt) == 1, "domino chain must have exactly one continuation"
        order.append(int(nxt[0]))
        remaining.remove(int(nxt[0]))
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder the vertical terrain panels so their touching edge heights interlock into one ridge.

    Input structure:
    - The grid is split by full-height separator columns into equal-width vertical panels.
    - Each panel contains one bottom-aligned monochrome terrain silhouette with one height per column.
    - Exactly one panel has a single marker pixel in its sky region above the terrain.

    Rule:
    - Keep every panel's internal pixels unchanged and move panels only as whole blocks.
    - Put the marked panel in the leftmost slot, then order the remaining panels so each panel's right-edge column height equals the next panel's left-edge column height.
    - Keep the separator columns fixed in place, and move the marker together with its panel.

    Key insight:
    - The correct order is a domino-style chain on paired edge heights; colors, peak heights, and widths are distractions, while the marker only chooses which panel starts the chain.
    """
    g = np.asarray(grid, dtype=int)
    bg, sep_color, separator_cols, bounds, panels, marked_panel = _extract_panels(g)
    order = _chain_order(panels, marked_panel)

    out = np.full_like(g, bg)
    for sep_c in separator_cols:
        out[:, int(sep_c)] = int(sep_color)
    for slot_idx, panel_idx in enumerate(order):
        c0, c1 = bounds[slot_idx]
        out[:, int(c0) : int(c1)] = panels[int(panel_idx)]["grid"]
    return out


def _rotation_order(count: int, start: int) -> list[int]:
    return [int((start + offset) % count) for offset in range(count)]


def _render_panel(
    height: int,
    panel_w: int,
    terrain_heights: tuple[int, ...],
    color: int,
    marker: tuple[int, int, int] | None = None,
) -> np.ndarray:
    panel = np.full((int(height), int(panel_w)), BG, dtype=int)
    for c, h in enumerate(terrain_heights):
        panel[int(height) - int(h) :, int(c)] = int(color)
    if marker is not None:
        mr, mc, marker_color = marker
        panel[int(mr), int(mc)] = int(marker_color)
    return panel


def _render_grid(
    height: int,
    panel_w: int,
    separator_color: int,
    order: list[int],
    panels: list[dict],
    marker_panel: int,
    marker_row: int,
    marker_col: int,
    marker_color: int,
) -> np.ndarray:
    total_w = len(order) * panel_w + (len(order) - 1) * SEPARATOR_WIDTH
    grid = np.full((int(height), int(total_w)), BG, dtype=int)
    for slot_idx, panel_idx in enumerate(order):
        c0 = slot_idx * (panel_w + SEPARATOR_WIDTH)
        marker = None
        if int(panel_idx) == int(marker_panel):
            marker = (int(marker_row), int(marker_col), int(marker_color))
        panel = _render_panel(height, panel_w, panels[int(panel_idx)]["heights"], panels[int(panel_idx)]["color"], marker)
        paste(grid, panel, 0, c0, transparent=None)
        if slot_idx != len(order) - 1:
            grid[:, c0 + panel_w : c0 + panel_w + SEPARATOR_WIDTH] = int(separator_color)
    return grid


def _valid_orders(panels: list[dict], start_idx: int | None = None) -> list[tuple[int, ...]]:
    ids = list(range(len(panels)))
    orders: list[tuple[int, ...]] = []
    for perm in itertools.permutations(ids):
        if start_idx is not None and perm[0] != int(start_idx):
            continue
        if all(int(panels[a]["right"]) == int(panels[b]["left"]) for a, b in zip(perm, perm[1:])):
            orders.append(tuple(map(int, perm)))
    return orders


def _matches_scalar_sort(order: tuple[int, ...], panels: list[dict]) -> bool:
    ids = list(range(len(panels)))
    metrics = {
        "peak": [int(p["peak"]) for p in panels],
        "area": [int(p["area"]) for p in panels],
        "left": [int(p["left"]) for p in panels],
        "right": [int(p["right"]) for p in panels],
        "color": [int(p["color"]) for p in panels],
    }
    for values in metrics.values():
        if len(set(values)) != len(values):
            continue
        asc = tuple(sorted(ids, key=lambda idx: values[idx]))
        desc = tuple(sorted(ids, key=lambda idx: values[idx], reverse=True))
        if order in {asc, desc}:
            return True
    return False


def _make_terrain_heights(panel_w: int, left: int, right: int, height: int) -> tuple[int, ...]:
    vals = [int(left)]
    for _ in range(panel_w - 2):
        vals.append(random.randint(1, int(height) - 1))
    vals.append(int(right))
    if len(set(vals)) == 1:
        mid = 1 if panel_w > 2 else 0
        vals[mid] = int(height) - 1 if vals[mid] == 1 else 1
    return tuple(vals)


def _gen_attempt() -> np.ndarray | None:
    panel_count = random.randint(*PANEL_COUNT_RANGE)
    max_w = min(PANEL_WIDTH_RANGE[1], (30 - (panel_count - 1) * SEPARATOR_WIDTH) // panel_count)
    if max_w < PANEL_WIDTH_RANGE[0]:
        return None
    panel_w = random.randint(PANEL_WIDTH_RANGE[0], max_w)
    height = random.randint(max(GRID_HEIGHT_RANGE[0], panel_count + 2), GRID_HEIGHT_RANGE[1])

    palette = random.sample(PALETTE, panel_count + 2)
    separator_color = int(palette[0])
    marker_color = int(palette[1])
    panel_colors = [int(color) for color in palette[2:]]

    cycle_heights = random.sample(range(1, height), panel_count)
    panels: list[dict] = []
    for idx in range(panel_count):
        left = int(cycle_heights[idx])
        right = int(cycle_heights[(idx + 1) % panel_count])
        terrain = _make_terrain_heights(panel_w, left, right, height)
        panels.append(
            {
                "color": int(panel_colors[idx]),
                "heights": terrain,
                "left": int(left),
                "right": int(right),
                "peak": int(max(terrain)),
                "area": int(sum(terrain)),
            }
        )

    marker_panel = random.randrange(panel_count)
    marker_row = 0
    marker_col = random.randrange(panel_w)
    output_panel_ids = _rotation_order(panel_count, marker_panel)
    input_panel_ids = random.sample(range(panel_count), panel_count)
    if input_panel_ids == output_panel_ids:
        return None

    inp = _render_grid(
        height,
        panel_w,
        separator_color,
        input_panel_ids,
        panels,
        marker_panel,
        marker_row,
        marker_col,
        marker_color,
    )
    expected = _render_grid(
        height,
        panel_w,
        separator_color,
        output_panel_ids,
        panels,
        marker_panel,
        marker_row,
        marker_col,
        marker_color,
    )

    if int(get_bg(inp)) != BG:
        return None
    if np.array_equal(inp, expected):
        return None

    try:
        bg, sep_color, _separator_cols, _bounds, parsed_panels, marked_idx = _extract_panels(inp)
        assert bg == BG and sep_color == separator_color
        chain = tuple(_chain_order(parsed_panels, marked_idx))
        fixed_orders = _valid_orders(parsed_panels, marked_idx)
        any_orders = _valid_orders(parsed_panels, None)
        assert len(fixed_orders) == 1, "marker-fixed chain must be unique"
        assert chain == fixed_orders[0]
        assert len(any_orders) > 1, "marker must be necessary to choose the start panel"
        assert not _matches_scalar_sort(chain, parsed_panels), "chain must not collapse to a scalar sort"
        solved = main(inp)
    except Exception:
        return None

    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cliff-chain ARC input")
