import random
from collections import defaultdict

import numpy as np


BG = 8
SEP = 9
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIRS8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)
SIDE_ORDER = ("top", "left", "bottom", "right")
GEN_RETRY_LIMIT = 300


def _component_cells(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    start = None
    for i, flag in enumerate(flags.tolist() + [False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            out.append((start, i))
            start = None
    return out


def _find_separator_color(grid: np.ndarray) -> int:
    candidates: list[tuple[int, int]] = []
    for color in np.unique(grid):
        full_rows = int(np.all(grid == color, axis=1).sum())
        full_cols = int(np.all(grid == color, axis=0).sum())
        if full_rows > 0 and full_cols > 0:
            candidates.append((full_rows + full_cols, int(color)))
    assert candidates, "expected a separator color spanning full rows and columns"
    candidates.sort(reverse=True)
    return candidates[0][1]


def _extract_panels(grid: np.ndarray) -> tuple[int, int, list[dict]]:
    sep = _find_separator_color(grid)
    non_sep = grid[grid != sep]
    assert non_sep.size > 0, "grid cannot be all separator"
    bg = int(np.bincount(non_sep.ravel(), minlength=10).argmax())

    row_runs = _runs(~np.all(grid == sep, axis=1))
    col_runs = _runs(~np.all(grid == sep, axis=0))
    assert row_runs and col_runs, "expected at least one panel"

    panels: list[dict] = []
    for pr, (r0, r1) in enumerate(row_runs):
        for pc, (c0, c1) in enumerate(col_runs):
            sub = grid[r0:r1, c0:c1]
            assert sub.size > 0
            assert not np.any(sub == sep), "separator must not cut through panel interiors"

            side = None
            if pr == 0:
                side = "top"
            elif pc == 0:
                side = "left"
            elif pr == len(row_runs) - 1:
                side = "bottom"
            elif pc == len(col_runs) - 1:
                side = "right"
            assert side is not None, "every panel must touch an outer edge"

            panels.append(
                {
                    "row_idx": pr,
                    "col_idx": pc,
                    "bounds": (r0, r1, c0, c1),
                    "side": side,
                    "subgrid": sub,
                }
            )
    return bg, sep, panels


def _crop_patch(subgrid: np.ndarray, cells: list[tuple[int, int]], bg: int, color: int) -> np.ndarray:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r, c in cells:
        patch[r - r0, c - c0] = color
    return patch


def _collect_lonely_objects(grid: np.ndarray) -> tuple[int, list[dict]]:
    bg, _sep, panels = _extract_panels(grid)
    color_panels: dict[int, set[tuple[int, int]]] = defaultdict(set)
    color_counts: dict[int, int] = defaultdict(int)

    for panel in panels:
        sub = panel["subgrid"]
        objects = []
        for color in np.unique(sub):
            color = int(color)
            if color == bg:
                continue
            for comp in _component_cells(sub == color):
                color_panels[color].add((panel["row_idx"], panel["col_idx"]))
                color_counts[color] += 1
                objects.append(
                    {
                        "color": color,
                        "cells": comp,
                        "patch": _crop_patch(sub, comp, bg, color),
                    }
                )
        panel["objects"] = objects

    lonely: list[dict] = []
    for panel in panels:
        matches = [
            obj
            for obj in panel["objects"]
            if len(color_panels[obj["color"]]) == 1 and color_counts[obj["color"]] == 1
        ]
        assert len(matches) == 1, "each panel must contain exactly one lonely object"
        lonely.append(
            {
                "side": panel["side"],
                "row_idx": panel["row_idx"],
                "col_idx": panel["col_idx"],
                "patch": matches[0]["patch"],
            }
        )
    return bg, lonely


def _paint_patch(canvas: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    mask = patch != bg
    target = canvas[top : top + patch.shape[0], left : left + patch.shape[1]]
    target[mask] = patch[mask]


def _build_band(items: list[np.ndarray], side: str, bg: int) -> np.ndarray:
    assert items
    if side in ("top", "bottom"):
        band_h = max(int(p.shape[0]) for p in items)
        band_w = sum(int(p.shape[1]) for p in items) + len(items) - 1
        band = np.full((band_h, band_w), bg, dtype=int)
        c = 0
        for patch in items:
            r = 0 if side == "top" else band_h - patch.shape[0]
            _paint_patch(band, patch, r, c, bg)
            c += patch.shape[1] + 1
        return band

    band_h = sum(int(p.shape[0]) for p in items) + len(items) - 1
    band_w = max(int(p.shape[1]) for p in items)
    band = np.full((band_h, band_w), bg, dtype=int)
    r = 0
    for patch in items:
        c = 0 if side == "left" else band_w - patch.shape[1]
        _paint_patch(band, patch, r, c, bg)
        r += patch.shape[0] + 1
    return band


def _render_output(bg: int, lonely: list[dict]) -> np.ndarray:
    bands: list[tuple[str, np.ndarray]] = []
    for side in SIDE_ORDER:
        group = [x for x in lonely if x["side"] == side]
        if not group:
            continue
        group.sort(key=lambda x: (x["row_idx"], x["col_idx"]))
        bands.append((side, _build_band([x["patch"] for x in group], side, bg)))

    out_h = sum(band.shape[0] for _side, band in bands) + max(0, len(bands) - 1)
    out_w = max(band.shape[1] for _side, band in bands)
    out = np.full((out_h, out_w), bg, dtype=int)

    r = 0
    for side, band in bands:
        c = out_w - band.shape[1] if side == "right" else 0
        out[r : r + band.shape[0], c : c + band.shape[1]] = band
        r += band.shape[0] + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg, lonely = _collect_lonely_objects(g)
    return _render_output(bg, lonely)


SHAPES = [
    {(0, 0), (0, 1)},
    {(0, 0), (1, 0)},
    {(0, 0), (0, 1), (0, 2)},
    {(0, 0), (1, 0), (2, 0)},
    {(0, 0), (0, 1), (1, 0)},
    {(0, 0), (0, 1), (1, 1)},
    {(0, 1), (1, 0), (1, 1)},
    {(0, 0), (1, 0), (1, 1)},
    {(0, 0), (0, 1), (1, 0), (1, 1)},
    {(0, 0), (1, 0), (2, 0), (2, 1)},
    {(0, 1), (1, 1), (2, 1), (2, 0)},
    {(0, 0), (0, 1), (0, 2), (1, 1)},
    {(0, 1), (1, 0), (1, 1), (2, 1)},
    {(0, 0), (1, 0), (1, 1), (2, 1)},
]


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _rotate(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    max_c = max(c for _, c in cells)
    return _normalize({(c, max_c - r) for r, c in cells})


def _flip_h(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    max_c = max(c for _, c in cells)
    return _normalize({(r, max_c - c) for r, c in cells})


def _random_shape() -> set[tuple[int, int]]:
    cells = set(random.choice(SHAPES))
    for _ in range(random.randrange(4)):
        cells = _rotate(cells)
    if random.random() < 0.5:
        cells = _flip_h(cells)
    return _normalize(cells)


def _shape_dims(cells: set[tuple[int, int]]) -> tuple[int, int]:
    return max(r for r, _ in cells) + 1, max(c for _, c in cells) + 1


def _place_objects(panel_h: int, panel_w: int, colors: list[int]) -> list[dict] | None:
    occupied: set[tuple[int, int]] = set()
    blocked: set[tuple[int, int]] = set()
    placed: list[dict] = []

    for color in colors:
        shape = _random_shape()
        h, w = _shape_dims(shape)
        candidates = []
        for r0 in range(panel_h - h + 1):
            for c0 in range(panel_w - w + 1):
                translated = {(r0 + r, c0 + c) for r, c in shape}
                if translated & blocked:
                    continue
                candidates.append((r0, c0, translated))
        if not candidates:
            return None

        r0, c0, translated = random.choice(candidates)
        occupied |= translated
        for r, c in translated:
            for dr, dc in DIRS8:
                nr, nc = r + dr, c + dc
                if 0 <= nr < panel_h and 0 <= nc < panel_w:
                    blocked.add((nr, nc))
            blocked.add((r, c))
        placed.append({"color": color, "cells": translated})

    return placed


def _panel_assignment() -> list[list[int]]:
    panel_ids = list(range(6))
    chosen = set(random.sample(panel_ids, 3))
    assignment: list[list[int]] = []
    for panel_id in panel_ids:
        base = 0 if panel_id in chosen else 1
        colors = [base]
        if random.random() < 0.5:
            colors.append(1 - base)
        assignment.append(colors)
    if sum(0 in xs for xs in assignment) < 2 or sum(1 in xs for xs in assignment) < 2:
        return _panel_assignment()
    return assignment


def _build_input_and_expected() -> tuple[np.ndarray, np.ndarray] | None:
    row_heights = [random.randint(5, 8) for _ in range(3)]
    col_widths = [random.randint(5, 8) for _ in range(2)]
    h = sum(row_heights) + 2
    w = sum(col_widths) + 1
    if h > 30 or w > 30:
        return None

    grid = np.full((h, w), BG, dtype=int)
    grid[row_heights[0], :] = SEP
    grid[row_heights[0] + 1 + row_heights[1], :] = SEP
    grid[:, col_widths[0]] = SEP

    common_palette = random.sample(list(range(8)), 2)
    unique_palette = [c for c in range(8) if c not in common_palette]
    random.shuffle(unique_palette)
    common_by_panel = _panel_assignment()

    lonely_items: list[dict] = []
    panel_id = 0
    row_starts = [0, row_heights[0] + 1, row_heights[0] + row_heights[1] + 2]
    col_starts = [0, col_widths[0] + 1]
    for pr, panel_h in enumerate(row_heights):
        for pc, panel_w in enumerate(col_widths):
            unique_color = unique_palette[panel_id]
            colors = [unique_color] + [common_palette[i] for i in common_by_panel[panel_id]]
            random.shuffle(colors)
            placed = _place_objects(panel_h, panel_w, colors)
            if placed is None:
                return None

            r0 = row_starts[pr]
            c0 = col_starts[pc]
            lonely_patch = None
            for obj in placed:
                for r, c in obj["cells"]:
                    grid[r0 + r, c0 + c] = obj["color"]
                if obj["color"] == unique_color:
                    lonely_patch = _crop_patch(
                        np.full((panel_h, panel_w), BG, dtype=int),
                        [(r, c) for r, c in obj["cells"]],
                        BG,
                        unique_color,
                    )
            assert lonely_patch is not None

            if pr == 0:
                side = "top"
            elif pc == 0:
                side = "left"
            elif pr == len(row_heights) - 1:
                side = "bottom"
            else:
                side = "right"
            lonely_items.append(
                {
                    "side": side,
                    "row_idx": pr,
                    "col_idx": pc,
                    "patch": lonely_patch,
                }
            )
            panel_id += 1

    expected = _render_output(BG, lonely_items)
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        built = _build_input_and_expected()
        if built is None:
            continue
        grid, expected = built
        try:
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid input")
