from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np


def _add_arc_utils_to_path() -> None:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "new" / "solvers" / "arc_utils.py"
        if candidate.is_file():
            utils_dir = str(candidate.parent)
            if utils_dir not in sys.path:
                sys.path.append(utils_dir)
            return
    raise ImportError("could not locate new/solvers/arc_utils.py")


_add_arc_utils_to_path()

from arc_utils import find_components, paste


BACKGROUND = 0
SEPARATOR = 5
PANEL_SIZE = 9
INPUT_SHAPE = (PANEL_SIZE, PANEL_SIZE * 3 + 2)
AREA_MIN = 3
AREA_MAX = 8
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
PANEL_BUILD_RETRIES = 120
PLACEMENT_RETRIES = 160
GEN_ATTEMPTS = 500
PANEL_AREA_BUDGET = 26
OVERLAP_PROBABILITY = 0.7


def _split_panels(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert width == 3 * height + 2, "expected three equal square panels separated by two columns"
    assert bool(np.all(g[:, height] == SEPARATOR)), "first separator column must be solid gray"
    assert bool(np.all(g[:, 2 * height + 1] == SEPARATOR)), "second separator column must be solid gray"
    return [
        g[:, :height].copy(),
        g[:, height + 1 : 2 * height + 1].copy(),
        g[:, 2 * height + 2 :].copy(),
    ]


def _strict_mode(values: list[int]) -> int:
    counts = Counter(int(v) for v in values)
    max_freq = max(counts.values())
    winners = [value for value, freq in counts.items() if freq == max_freq]
    assert len(winners) == 1, "expected a unique strict mode"
    return int(winners[0])


def _panel_objects(panel: np.ndarray) -> tuple[list[dict], int]:
    objects = find_components(panel, bg=BACKGROUND, connectivity=4, monochromatic=True, return_format="dict")
    assert 4 <= len(objects) <= 5, "each panel must contain 4-5 monochrome objects"

    colors = [int(obj["color"]) for obj in objects]
    assert len(set(colors)) == len(colors), "object colors must be unique within a panel"

    areas = [int(obj["area"]) for obj in objects]
    assert all(AREA_MIN <= area <= AREA_MAX for area in areas), "object areas must stay within 3-8"

    mode_area = _strict_mode(areas)
    mode_count = areas.count(mode_area)
    assert mode_count >= 2, "each panel needs at least two mode-area objects"
    assert mode_count < len(objects), "each panel needs at least one non-mode object"
    return objects, mode_area


def _paint_component(shape: tuple[int, int], obj: dict) -> np.ndarray:
    patch = np.full(shape, BACKGROUND, dtype=int)
    coords = np.asarray(obj["coords"], dtype=int)
    patch[coords[:, 0], coords[:, 1]] = int(obj["color"])
    return patch


def _filtered_panel(panel: np.ndarray) -> np.ndarray:
    objects, mode_area = _panel_objects(panel)
    kept = np.full(panel.shape, BACKGROUND, dtype=int)
    for obj in objects:
        if int(obj["area"]) == mode_area:
            patch = _paint_component(panel.shape, obj)
            paste(kept, patch, 0, 0, transparent=BACKGROUND)
    return kept


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep the mode-area objects inside each separator-defined panel, then merge the survivors onto one canvas.

    Input structure:
    - The input is a grid made of three equal square black panels separated by two solid gray columns.
    - Each panel contains 4-5 monochrome 4-connected foreground objects, with unique colors inside that panel.
    - Every object area is between 3 and 8 pixels, and each panel has a unique strict mode among those area values.

    Rule:
    - Split the grid into left, middle, and right 10x10 panels using the gray separator columns.
    - In each panel, compute the pixel area of every 4-connected object and keep only the objects whose area equals that panel's strict mode.
    - Overlay the three filtered panels onto a new 10x10 black canvas at their original panel coordinates, with later panels overwriting earlier ones on overlaps.

    Key insight:
    - The voting happens independently inside each panel and votes over object areas rather than colors, so the final merge depends on both per-panel counting and right-priority overlay.
    """
    panels = _split_panels(grid)
    size = panels[0].shape[0]
    out = np.full((size, size), BACKGROUND, dtype=int)
    for panel in panels:
        paste(out, _filtered_panel(panel), 0, 0, transparent=BACKGROUND)
    return out


def _normalize_shape(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _c in shape]
    cols = [c for _r, c in shape]
    return max(rows) + 1, max(cols) + 1


def _random_connected_shape(area: int) -> tuple[tuple[int, int], ...]:
    dim_options = [
        (h, w)
        for h in range(1, 5)
        for w in range(1, 5)
        if h * w >= area
    ]
    h, w = random.choice(dim_options)
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    frontier = set()

    def add_frontier(r: int, c: int) -> None:
        for dr, dc in NEIGHBORS4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                frontier.add((rr, cc))

    add_frontier(*start)
    while len(cells) < area:
        if not frontier:
            return _random_connected_shape(area)
        options = list(frontier)
        weights = []
        for r, c in options:
            neighbors = sum((r + dr, c + dc) in cells for dr, dc in NEIGHBORS4)
            weights.append((neighbors + 1) ** 2)
        cell = random.choices(options, weights=weights, k=1)[0]
        frontier.remove(cell)
        cells.add(cell)
        add_frontier(*cell)

    shape = _normalize_shape(cells)
    if random.random() < 0.5:
        shape = _normalize_shape({(c, r) for r, c in shape})
    sh, sw = _shape_dims(shape)
    transformed = set(shape)
    if random.random() < 0.5:
        transformed = {(sh - 1 - r, c) for r, c in transformed}
    if random.random() < 0.5:
        transformed = {(r, sw - 1 - c) for r, c in transformed}
    return _normalize_shape(transformed)


def _can_place(cells: set[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        if not (0 <= r < PANEL_SIZE and 0 <= c < PANEL_SIZE):
            return False
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if (rr, cc) in occupied:
                    return False
    return True


def _sample_placed_cells(area: int, occupied: set[tuple[int, int]], anchor_target: tuple[int, int] | None) -> tuple[tuple[int, int], ...] | None:
    for _ in range(PLACEMENT_RETRIES):
        shape = _random_connected_shape(area)
        anchor = random.choice(shape)
        target = anchor_target if anchor_target is not None else (random.randrange(PANEL_SIZE), random.randrange(PANEL_SIZE))
        cells = {(r - anchor[0] + target[0], c - anchor[1] + target[1]) for r, c in shape}
        if _can_place(cells, occupied):
            return tuple(sorted(cells))
    return None


def _sample_area_profile(count: int, modal_count: int, relation: str) -> dict | None:
    other_count = count - modal_count
    values = list(range(AREA_MIN, AREA_MAX + 1))

    if relation == "smallest":
        candidates = [m for m in values if len([v for v in values if v > m]) >= other_count]
        if not candidates:
            return None
        modal_area = random.choice(candidates)
        others = random.sample([v for v in values if v > modal_area], other_count)
    elif relation == "largest":
        candidates = [m for m in values if len([v for v in values if v < m]) >= other_count]
        if not candidates:
            return None
        modal_area = random.choice(candidates)
        others = random.sample([v for v in values if v < modal_area], other_count)
    else:
        candidates = [m for m in values if m > AREA_MIN and m < AREA_MAX]
        random.shuffle(candidates)
        modal_area = None
        others = None
        for middle in candidates:
            splits = [
                low_count
                for low_count in range(1, other_count)
                if len([v for v in values if v < middle]) >= low_count
                and len([v for v in values if v > middle]) >= other_count - low_count
            ]
            if not splits:
                continue
            low_count = random.choice(splits)
            lower = random.sample([v for v in values if v < middle], low_count)
            upper = random.sample([v for v in values if v > middle], other_count - low_count)
            modal_area = middle
            others = lower + upper
            break
        if modal_area is None or others is None:
            return None

    areas = [modal_area] * modal_count + list(others)
    random.shuffle(areas)
    retained_indices = [idx for idx, area in enumerate(areas) if area == modal_area]
    return {
        "count": count,
        "modal_area": int(modal_area),
        "modal_count": int(modal_count),
        "areas": [int(area) for area in areas],
        "retained_indices": retained_indices,
        "anchor_index": random.choice(retained_indices),
        "relation": relation,
    }


def _sample_specs() -> dict | None:
    relation_options = ["smallest", "middle", "largest"]
    overlap_choices = [(0, 1), (1, 2), (0, 2), (0, 1, 2)]

    for _ in range(200):
        specs = []
        for _panel_index in range(3):
            count = random.choice([4, 5])
            modal_count = 2 if count == 4 else random.choice([2, 3])
            random.shuffle(relation_options)
            profile = None
            for relation in relation_options:
                candidate = _sample_area_profile(count, modal_count, relation)
                if candidate is not None and sum(candidate["areas"]) <= PANEL_AREA_BUDGET:
                    profile = candidate
                    break
            if profile is None:
                break
            specs.append(profile)

        if len(specs) != 3:
            continue
        if len({spec["modal_area"] for spec in specs}) < 2:
            continue
        if len({spec["relation"] for spec in specs}) < 2:
            continue

        require_overlap = random.random() < OVERLAP_PROBABILITY
        overlap_panels = random.choice(overlap_choices) if require_overlap else ()
        shared_anchor = (random.randint(2, 7), random.randint(2, 7)) if require_overlap else None

        for _color_attempt in range(80):
            for spec in specs:
                spec["colors"] = random.sample(PALETTE, spec["count"])
            if require_overlap:
                anchor_colors = [
                    specs[idx]["colors"][specs[idx]["anchor_index"]]
                    for idx in overlap_panels
                ]
                if len(set(anchor_colors)) != len(anchor_colors):
                    continue
            return {
                "panels": specs,
                "require_overlap": require_overlap,
                "overlap_panels": overlap_panels,
                "shared_anchor": shared_anchor,
            }
    return None


def _bbox_area(cells: tuple[tuple[int, int], ...]) -> int:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)


def _build_panel(spec: dict, fixed_anchor: tuple[int, int] | None) -> tuple[np.ndarray, list[dict]] | None:
    for _ in range(PANEL_BUILD_RETRIES):
        pending = []
        for idx, (area, color) in enumerate(zip(spec["areas"], spec["colors"])):
            pending.append(
                {
                    "index": idx,
                    "area": int(area),
                    "color": int(color),
                    "retained": idx in spec["retained_indices"],
                    "anchor_target": fixed_anchor if idx == spec["anchor_index"] else None,
                }
            )

        random.shuffle(pending)
        pending.sort(key=lambda obj: (obj["anchor_target"] is None, -obj["area"]))

        occupied: set[tuple[int, int]] = set()
        placed = []
        ok = True
        for obj in pending:
            cells = _sample_placed_cells(obj["area"], occupied, obj["anchor_target"])
            if cells is None:
                ok = False
                break
            occupied.update(cells)
            placed.append(
                {
                    "index": obj["index"],
                    "area": obj["area"],
                    "color": obj["color"],
                    "retained": obj["retained"],
                    "cells": cells,
                    "bbox_area": _bbox_area(cells),
                }
            )

        if not ok:
            continue

        panel = np.full((PANEL_SIZE, PANEL_SIZE), BACKGROUND, dtype=int)
        for obj in placed:
            rows, cols = zip(*obj["cells"])
            panel[rows, cols] = obj["color"]

        try:
            parsed, mode_area = _panel_objects(panel)
        except AssertionError:
            continue
        if mode_area != spec["modal_area"]:
            continue
        if len(parsed) != spec["count"]:
            continue
        if sorted(int(obj["area"]) for obj in parsed) != sorted(spec["areas"]):
            continue
        return panel, placed
    return None


def _render_input(panels: list[np.ndarray]) -> np.ndarray:
    grid = np.full(INPUT_SHAPE, BACKGROUND, dtype=int)
    grid[:, PANEL_SIZE] = SEPARATOR
    grid[:, 2 * PANEL_SIZE + 1] = SEPARATOR
    paste(grid, panels[0], 0, 0)
    paste(grid, panels[1], 0, PANEL_SIZE + 1)
    paste(grid, panels[2], 0, 2 * PANEL_SIZE + 2)
    return grid


def _layer_from_metadata(panel_objects: list[dict], keep_pred) -> np.ndarray:
    layer = np.full((PANEL_SIZE, PANEL_SIZE), BACKGROUND, dtype=int)
    for obj in panel_objects:
        if keep_pred(obj):
            rows, cols = zip(*obj["cells"])
            layer[rows, cols] = obj["color"]
    return layer


def _merge_layers(layers: list[np.ndarray], *, priority: str) -> np.ndarray:
    out = np.full((PANEL_SIZE, PANEL_SIZE), BACKGROUND, dtype=int)
    seq = layers if priority == "right" else list(reversed(layers))
    for layer in seq:
        paste(out, layer, 0, 0, transparent=BACKGROUND)
    return out


def _merge_with_sources(layers: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    out = np.full((PANEL_SIZE, PANEL_SIZE), BACKGROUND, dtype=int)
    sources = np.full((PANEL_SIZE, PANEL_SIZE), -1, dtype=int)
    for panel_index, layer in enumerate(layers):
        mask = layer != BACKGROUND
        out[mask] = layer[mask]
        sources[mask] = panel_index
    return out, sources


def _panel_mode_layer(panel_objects: list[dict]) -> np.ndarray:
    mode_area = _strict_mode([obj["area"] for obj in panel_objects])
    return _layer_from_metadata(panel_objects, lambda obj: obj["area"] == mode_area)


def _wrong_global_mode(panel_objects: list[list[dict]]) -> np.ndarray | None:
    areas = [obj["area"] for panel in panel_objects for obj in panel]
    counts = Counter(areas)
    max_freq = max(counts.values())
    winners = [area for area, freq in counts.items() if freq == max_freq]
    if len(winners) != 1:
        return None
    area = winners[0]
    layers = [_layer_from_metadata(panel, lambda obj, area=area: obj["area"] == area) for panel in panel_objects]
    return _merge_layers(layers, priority="right")


def _wrong_panel_extreme(panel_objects: list[list[dict]], *, which: str) -> np.ndarray:
    layers = []
    for panel in panel_objects:
        values = [obj["area"] for obj in panel]
        target = min(values) if which == "min" else max(values)
        layers.append(_layer_from_metadata(panel, lambda obj, target=target: obj["area"] == target))
    return _merge_layers(layers, priority="right")


def _wrong_bbox_mode(panel_objects: list[list[dict]]) -> np.ndarray | None:
    layers = []
    for panel in panel_objects:
        values = [obj["bbox_area"] for obj in panel]
        counts = Counter(values)
        max_freq = max(counts.values())
        winners = [value for value, freq in counts.items() if freq == max_freq]
        if len(winners) != 1:
            return None
        target = winners[0]
        layers.append(_layer_from_metadata(panel, lambda obj, target=target: obj["bbox_area"] == target))
    return _merge_layers(layers, priority="right")


def _validate_candidate(grid: np.ndarray, panel_objects: list[list[dict]], spec_bundle: dict) -> bool:
    try:
        actual = main(grid)
    except AssertionError:
        return False

    retained_layers = [_panel_mode_layer(panel) for panel in panel_objects]
    expected, sources = _merge_with_sources(retained_layers)
    if not np.array_equal(actual, expected):
        return False

    visible_panels = set(map(int, sources[sources >= 0].flatten()))
    if len(visible_panels) < 2:
        return False

    if np.count_nonzero(expected) < 8:
        return False

    if spec_bundle["require_overlap"]:
        overlap_found = False
        panels = spec_bundle["overlap_panels"]
        for i, panel_i in enumerate(panels):
            for panel_j in panels[i + 1 :]:
                if np.any((retained_layers[panel_i] != BACKGROUND) & (retained_layers[panel_j] != BACKGROUND)):
                    overlap_found = True
        if not overlap_found:
            return False

    wrong_outputs = [
        _wrong_global_mode(panel_objects),
        _wrong_panel_extreme(panel_objects, which="min"),
        _wrong_panel_extreme(panel_objects, which="max"),
        _wrong_bbox_mode(panel_objects),
    ]
    for wrong in wrong_outputs:
        if wrong is not None and np.array_equal(wrong, expected):
            return False

    if spec_bundle["require_overlap"]:
        left_priority = _merge_layers(retained_layers, priority="left")
        if np.array_equal(left_priority, expected):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    spec_bundle = _sample_specs()
    if spec_bundle is None:
        return None

    panels = []
    panel_objects = []
    for panel_index, spec in enumerate(spec_bundle["panels"]):
        fixed_anchor = spec_bundle["shared_anchor"] if panel_index in spec_bundle["overlap_panels"] else None
        built = _build_panel(spec, fixed_anchor)
        if built is None:
            return None
        panel, objects = built
        panels.append(panel)
        panel_objects.append(objects)

    grid = _render_input(panels)
    if _validate_candidate(grid, panel_objects, spec_bundle):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
