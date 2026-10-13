from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np


def _add_arc_utils_to_path() -> None:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "new" / "solvers" / "arc_utils.py"
        if candidate.exists():
            utils_dir = str(candidate.parent)
            if utils_dir not in sys.path:
                sys.path.append(utils_dir)
            return
    raise ImportError("could not locate new/solvers/arc_utils.py")


_add_arc_utils_to_path()

from arc_utils import Color, crop_bbox, find_components, get_bg, paste, random_free_location


ARC_COLORS = list(Color.ALL_COLORS)
MAX_GENERATION_ATTEMPTS = 300
CORNERS = ("tl", "tr", "bl", "br")


def _corner_coord(h: int, w: int, corner: str) -> tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, w - 1
    if corner == "bl":
        return h - 1, 0
    if corner == "br":
        return h - 1, w - 1
    raise ValueError(f"unknown corner: {corner}")


def _component_shape(component: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = component["bbox"]
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _is_solid_rectangle(component: dict) -> bool:
    h, w = _component_shape(component)
    return int(component["area"]) == h * w


def _read_strip(patch: np.ndarray) -> list[int] | None:
    if patch.shape[0] == 1:
        return [int(v) for v in patch[0, :]]
    if patch.shape[1] == 1:
        return [int(v) for v in patch[:, 0]]
    return None


def _dominant_color(patch: np.ndarray) -> int:
    colors, counts = np.unique(patch, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _overlay_panels(
    panels_by_tag: dict[int, np.ndarray], order: list[int], base: int, tag_pos: tuple[int, int]
) -> np.ndarray:
    h, w = next(iter(panels_by_tag.values())).shape
    out = np.full((h, w), int(base), dtype=int)
    tag_r, tag_c = tag_pos
    for tag in order:
        layer = panels_by_tag[int(tag)].copy()
        layer[tag_r, tag_c] = int(base)
        paste(out, layer, 0, 0, transparent=int(base))
    return out


def _parse_problem(grid: np.ndarray) -> tuple[int, int, tuple[int, int], list[int], dict[int, np.ndarray]]:
    bg = int(get_bg(grid))
    components = find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=False,
        return_format="dict",
    )
    assert components, "expected foreground components"

    solid_components = [component for component in components if _is_solid_rectangle(component)]
    shape_counts = Counter(
        _component_shape(component)
        for component in solid_components
        if min(_component_shape(component)) > 1
    )
    assert shape_counts, "expected repeated rectangular panels"
    panel_shape, panel_count = max(
        shape_counts.items(),
        key=lambda item: (int(item[1]), int(item[0][0] * item[0][1])),
    )
    assert panel_count >= 2, "expected at least two same-sized panels"

    panel_components = [component for component in solid_components if _component_shape(component) == panel_shape]
    assert len(panel_components) == panel_count
    other_components = [component for component in solid_components if _component_shape(component) != panel_shape]

    panel_patches = [crop_bbox(grid, component["bbox"]) for component in panel_components]
    dominant_colors = [_dominant_color(patch) for patch in panel_patches]
    assert len(set(dominant_colors)) == 1, "panels must share one base color"
    base = int(dominant_colors[0])

    h, w = panel_shape
    matches: list[tuple[int, tuple[int, int], list[int]]] = []
    for corner in CORNERS:
        tag_pos = _corner_coord(h, w, corner)
        tag_colors = [int(patch[tag_pos]) for patch in panel_patches]
        if len(set(tag_colors)) != len(tag_colors):
            continue
        if bg in tag_colors or base in tag_colors:
            continue

        for component in other_components:
            strip = crop_bbox(grid, component["bbox"])
            order = _read_strip(strip)
            if order is None or len(order) != len(panel_patches):
                continue
            if base in order or len(set(order)) != len(order):
                continue
            if set(order) == set(tag_colors):
                matches.append((base, tag_pos, order))

    assert len(matches) == 1, "expected one legend/corner pairing"
    _base, tag_pos, order = matches[0]
    panels_by_tag = {
        int(patch[tag_pos]): patch
        for patch in panel_patches
    }
    return bg, base, tag_pos, order, panels_by_tag


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay same-sized panels opaquely using the legend strip as the bottom-to-top layer order.

    Input structure:
    - A separator-color background contains several disconnected rectangular panels of the same size.
    - Each panel is filled with one common base color, contains sparse colored marks, and has one unique tag pixel at a fixed corner.
    - Another disconnected one-cell-thick horizontal or vertical strip lists those tag colors in order.

    Rule:
    - Treat the strip order from left to right, or from top to bottom for a vertical strip, as the stack order from bottom layer to top layer.
    - Match each strip color to the panel with the same corner tag, start from a blank panel of the common base color, and copy each panel's non-base marks in that order.
    - Ignore both the corner tag pixels and the legend strip itself; later marks overwrite earlier marks at conflicts.

    Key insight:
    - The strip is not decoration: it is the only source of z-order, so overlapping marks must be resolved by opaque layering rather than by an unordered union.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _bg, base, tag_pos, order, panels_by_tag = _parse_problem(g)
    return _overlay_panels(panels_by_tag, order, base, tag_pos)


def _paint_cells(patch: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        patch[int(r), int(c)] = int(color)


def _random_hline(h: int, w: int) -> list[tuple[int, int]]:
    length = random.randint(2, w - 2)
    r = random.randint(1, h - 2)
    c0 = random.randint(1, w - length - 1)
    return [(r, c) for c in range(c0, c0 + length)]


def _random_vline(h: int, w: int) -> list[tuple[int, int]]:
    length = random.randint(2, h - 2)
    c = random.randint(1, w - 2)
    r0 = random.randint(1, h - length - 1)
    return [(r, c) for r in range(r0, r0 + length)]


def _random_block(h: int, w: int) -> list[tuple[int, int]]:
    block_h = random.randint(2, min(3, h - 2))
    block_w = random.randint(2, min(3, w - 2))
    r0 = random.randint(1, h - block_h - 1)
    c0 = random.randint(1, w - block_w - 1)
    return [(r, c) for r in range(r0, r0 + block_h) for c in range(c0, c0 + block_w)]


def _random_mark_cells(h: int, w: int, blocked: set[tuple[int, int]]) -> list[tuple[int, int]] | None:
    generators = (_random_hline, _random_vline, _random_block)
    for _ in range(40):
        cells = random.choice(generators)(h, w)
        if not any(cell in blocked for cell in cells):
            return cells
    return None


def _build_panels(
    panel_count: int,
    h: int,
    w: int,
    base: int,
    tag_corner: str,
    tag_colors: list[int],
    drawing_colors: list[int],
) -> list[np.ndarray]:
    panels = [np.full((h, w), int(base), dtype=int) for _ in range(panel_count)]
    reserved_corners = {_corner_coord(h, w, corner) for corner in CORNERS}

    conflict_layers = random.sample(range(panel_count), random.randint(2, min(3, panel_count)))
    conflict_cells = _random_mark_cells(h, w, reserved_corners)
    assert conflict_cells is not None
    conflict_colors = random.sample(drawing_colors, len(conflict_layers))
    for layer_idx, color in zip(conflict_layers, conflict_colors):
        _paint_cells(panels[int(layer_idx)], conflict_cells, int(color))

    for idx, panel in enumerate(panels):
        blocked = set(reserved_corners)
        if idx in conflict_layers:
            blocked |= set(conflict_cells)
        mark_count = random.randint(1, 2)
        for _ in range(mark_count):
            cells = _random_mark_cells(h, w, blocked)
            if cells is None:
                break
            _paint_cells(panel, cells, random.choice(drawing_colors))
            blocked |= set(cells)

    tag_pos = _corner_coord(h, w, tag_corner)
    for panel, tag_color in zip(panels, tag_colors):
        panel[tag_pos] = int(tag_color)

    return panels


def _has_order_dependent_conflict(
    panels: list[np.ndarray], base: int, tag_pos: tuple[int, int]
) -> bool:
    h, w = panels[0].shape
    for r in range(h):
        for c in range(w):
            if (r, c) == tag_pos:
                continue
            colors = {int(panel[r, c]) for panel in panels if int(panel[r, c]) != int(base)}
            if len(colors) >= 2:
                return True
    return False


def _place_objects(
    panels: list[np.ndarray],
    separator: int,
    legend: np.ndarray,
    canvas_shape: tuple[int, int],
) -> np.ndarray | None:
    grid = np.full(canvas_shape, int(separator), dtype=int)
    for panel in panels:
        try:
            r0, c0 = random_free_location(
                grid,
                panel,
                transparent=int(separator),
                border=1,
                padding=2,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(grid, panel, r0, c0)

    try:
        r0, c0 = random_free_location(
            grid,
            legend,
            transparent=int(separator),
            border=1,
            padding=2,
            padding_connectivity=8,
        )
    except ValueError:
        return None
    paste(grid, legend, r0, c0)
    return grid


def _semantic_accepts(
    grid: np.ndarray,
    panels_by_tag: dict[int, np.ndarray],
    order: list[int],
    base: int,
    tag_pos: tuple[int, int],
) -> bool:
    expected = _overlay_panels(panels_by_tag, order, base, tag_pos)
    reversed_expected = _overlay_panels(panels_by_tag, list(reversed(order)), base, tag_pos)
    if np.array_equal(expected, reversed_expected):
        return False

    try:
        solved = main(grid)
    except Exception:
        return False
    return np.array_equal(solved, expected)


def _gen_attempt() -> np.ndarray | None:
    panel_count = random.randint(2, 4)
    h = random.randint(5, 8)
    w = random.randint(5, 8)
    tag_corner = random.choice(CORNERS)

    palette = random.sample(ARC_COLORS, len(ARC_COLORS))
    separator = int(palette[0])
    base = int(palette[1])
    tag_colors = [int(v) for v in palette[2 : 2 + panel_count]]
    drawing_colors = [int(v) for v in palette[2 + panel_count :]]
    if len(drawing_colors) < 3:
        return None

    panels = _build_panels(
        panel_count,
        h,
        w,
        base,
        tag_corner,
        tag_colors,
        drawing_colors,
    )
    tag_pos = _corner_coord(h, w, tag_corner)
    if not _has_order_dependent_conflict(panels, base, tag_pos):
        return None

    order = tag_colors[:]
    random.shuffle(order)
    panels_by_tag = {int(tag): panel for tag, panel in zip(tag_colors, panels)}

    legend_is_horizontal = random.choice((True, False))
    legend = (
        np.array([order], dtype=int)
        if legend_is_horizontal
        else np.array(order, dtype=int).reshape(-1, 1)
    )

    min_side = max(
        22,
        h + 6,
        w + 6,
        int(np.ceil(np.sqrt(panel_count * (h + 4) * (w + 4)))) + 4,
    )
    canvas_h = random.randint(min_side, 30)
    canvas_w = random.randint(min_side, 30)
    grid = _place_objects(panels, separator, legend, (canvas_h, canvas_w))
    if grid is None:
        return None
    if int(get_bg(grid)) != separator:
        return None

    return grid if _semantic_accepts(grid, panels_by_tag, order, base, tag_pos) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid opaque-overlay legend input")
