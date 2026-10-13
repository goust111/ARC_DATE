from __future__ import annotations

import itertools
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

try:
    from new.solvers.arc_utils import Color
except Exception:
    for parent in Path(__file__).resolve().parents:
        if (parent / "new" / "solvers" / "arc_utils.py").exists():
            sys.path.insert(0, str(parent))
            break
    try:
        from new.solvers.arc_utils import Color
    except Exception:
        class Color:
            BLACK = 0
            BLUE = 1
            RED = 2
            GREEN = 3
            YELLOW = 4
            GREY = 5


PANEL_COUNT = 4
PANEL_WIDTH = 3
SEPARATOR_WIDTH = 1
INPUT_WIDTH = PANEL_COUNT * PANEL_WIDTH + (PANEL_COUNT - 1) * SEPARATOR_WIDTH
OUTPUT_WIDTH = PANEL_COUNT * PANEL_WIDTH
HEIGHT_RANGE = (10, 14)
MARKER_LENGTHS = (2, 3)
GEN_MAX_ATTEMPTS = 500
PANEL_PALETTE = (int(Color.BLUE), int(Color.RED), int(Color.GREEN), int(Color.YELLOW))
IDENTITY_ORDER = tuple(range(PANEL_COUNT))
NON_IDENTITY_ORDERS = [order for order in itertools.permutations(range(PANEL_COUNT)) if order != IDENTITY_ORDER]
KEY_CATALOG = {
    panel_id: [
        key
        for length in MARKER_LENGTHS
        for key in itertools.permutations([other for other in range(PANEL_COUNT) if other != panel_id], length)
    ]
    for panel_id in range(PANEL_COUNT)
}


def _separator_cols(grid: np.ndarray) -> list[int]:
    return [int(c) for c in range(grid.shape[1]) if np.all(grid[:, c] == int(Color.GREY))]


def _panel_bounds(grid: np.ndarray) -> list[tuple[int, int]]:
    seps = _separator_cols(grid)
    assert len(seps) == PANEL_COUNT - 1, "expected exactly three grey separator columns"

    bounds: list[tuple[int, int]] = []
    start = 0
    for sep in [*seps, grid.shape[1]]:
        bounds.append((start, sep))
        start = sep + 1

    assert len(bounds) == PANEL_COUNT, "expected four panels"
    assert all(c1 - c0 == PANEL_WIDTH for c0, c1 in bounds), "each panel must be exactly three columns wide"
    return bounds


def _dominant_panel_color(panel: np.ndarray) -> int:
    counts = Counter(int(value) for value in panel.ravel() if int(value) not in (int(Color.BLACK), int(Color.GREY)))
    assert counts, "each panel must contain non-background colors"
    color, top_count = counts.most_common(1)[0]
    assert sum(count == top_count for count in counts.values()) == 1, "panel dominant color must be unique"
    return int(color)


def _strip_legend(panel: np.ndarray, panel_index: int, main_color: int) -> np.ndarray:
    cleaned = panel.copy()
    if panel_index == 0:
        cleaned[:2, :2] = int(main_color)
    return cleaned


def _legend_pairs(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    legend = np.asarray(grid[:2, :2], dtype=int)
    pairs = (
        (int(legend[0, 0]), int(legend[0, 1])),
        (int(legend[1, 0]), int(legend[1, 1])),
    )
    flat = [color for pair in pairs for color in pair]
    assert len(set(flat)) == PANEL_COUNT, "legend must use four distinct panel colors"
    return pairs


def _sorted_order(keys: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    order = tuple(sorted(range(PANEL_COUNT), key=lambda idx: keys[idx]))
    assert len(set(order)) == PANEL_COUNT
    return order


def _build_key_assignments() -> dict[tuple[int, ...], list[tuple[tuple[int, ...], ...]]]:
    by_order = {order: [] for order in itertools.permutations(range(PANEL_COUNT))}
    for key_tuple in itertools.product(*(KEY_CATALOG[panel_id] for panel_id in range(PANEL_COUNT))):
        if len(set(key_tuple)) != PANEL_COUNT:
            continue
        by_order[_sorted_order(key_tuple)].append(key_tuple)
    return by_order


KEY_ASSIGNMENTS_BY_ORDER = _build_key_assignments()


def _extract_structure(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[1] == INPUT_WIDTH, f"expected width {INPUT_WIDTH}, got {g.shape[1]}"

    bounds = _panel_bounds(g)
    raw_panels = [g[:, c0:c1].copy() for c0, c1 in bounds]
    main_colors = tuple(_dominant_panel_color(panel) for panel in raw_panels)
    assert len(set(main_colors)) == PANEL_COUNT, "panel dominant colors must be unique"
    assert set(main_colors) == set(PANEL_PALETTE), "the four panel dominant colors must be blue/red/green/yellow"

    legend_pairs = _legend_pairs(g)
    assert set(color for pair in legend_pairs for color in pair) == set(main_colors), "legend must pair the panel colors"

    color_to_panel = {color: panel_id for panel_id, color in enumerate(main_colors)}
    panels: list[np.ndarray] = []
    keys: list[tuple[int, ...]] = []
    raw_color_keys: list[tuple[int, ...]] = []

    for panel_id, raw_panel in enumerate(raw_panels):
        panel = _strip_legend(raw_panel, panel_id, main_colors[panel_id])
        panels.append(panel)

        marker_coords = np.argwhere(
            (panel != main_colors[panel_id]) & (panel != int(Color.BLACK)) & (panel != int(Color.GREY))
        )
        assert 2 <= len(marker_coords) <= 3, "each panel needs two or three marker pixels"

        marker_cols = {int(c) for _, c in marker_coords}
        assert len(marker_cols) == 1, "marker pixels inside a panel must lie in one column"
        marker_col = marker_cols.pop()

        ordered_coords = marker_coords[np.argsort(marker_coords[:, 0])]
        color_key = tuple(int(panel[r, marker_col]) for r, _ in ordered_coords)
        key = tuple(color_to_panel[color] for color in color_key)
        assert len(set(key)) == len(key), "marker ids must not repeat within one panel"
        assert all(panel_ref != panel_id for panel_ref in key), "marker colors must refer to other panels"

        raw_color_keys.append(color_key)
        keys.append(key)

    key_tuple = tuple(keys)
    assert len(set(key_tuple)) == PANEL_COUNT, "panel sort keys must be unique"
    return {
        "grid": g,
        "main_colors": main_colors,
        "legend_pairs": legend_pairs,
        "panels": tuple(panels),
        "keys": key_tuple,
        "raw_color_keys": tuple(raw_color_keys),
    }


def _swap_periodic_columns(grid: np.ndarray, legend_pairs: tuple[tuple[int, int], tuple[int, int]]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    mapping: dict[int, int] = {}
    for left, right in legend_pairs:
        mapping[int(left)] = int(right)
        mapping[int(right)] = int(left)

    for c in range(0, out.shape[1], PANEL_WIDTH):
        original = out[:, c].copy()
        swapped = original.copy()
        for src, dst in mapping.items():
            swapped[original == src] = dst
        out[:, c] = swapped
    return out


def _oracle_output(clean_panels: list[np.ndarray], keys: tuple[tuple[int, ...], ...], legend_pairs) -> tuple[np.ndarray, tuple[int, ...]]:
    order = _sorted_order(keys)
    stitched = np.concatenate([clean_panels[idx] for idx in order], axis=1)
    assert stitched.shape[1] == OUTPUT_WIDTH
    return _swap_periodic_columns(stitched, legend_pairs), order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder the four vertical panels by their marker-column keys, then swap legend-paired colors on every third output column.

    Input structure:
    - The grid has four 3-column color-major panels separated by three full-height grey columns.
    - The four panel dominant colors are blue, red, green, and yellow in some left-to-right order.
    - The upper-left 2x2 cells form a legend overlay on the first panel, pairing those four colors into two bidirectional swap pairs.
    - Each panel also contains one column of 2 or 3 foreign-color marker pixels whose top-to-bottom color order refers to the original source-panel identities.

    Rule:
    - Use the grey separators to recover the four source panels and identify each panel's dominant color.
    - Read each panel's foreign-color marker column from top to bottom; interpret each marker color as the id of the panel whose dominant color matches it, and use that id sequence as the panel's lexicographic sort key.
    - Remove the separator columns, erase the 2x2 legend overlay back to the first panel's dominant color, and concatenate the whole panels from the smallest sort key to the largest sort key.
    - In the stitched output, swap the two legend colors within each legend row on columns 0, 3, 6, 9, ... for every non-black cell.

    Key insight:
    - The legend only defines which colors swap, while the reorder itself comes from reading foreign marker colors as references to the original panel identities rather than by comparing raw color ids.
    """
    info = _extract_structure(grid)
    order = _sorted_order(info["keys"])
    stitched = np.concatenate([info["panels"][idx] for idx in order], axis=1)
    return _swap_periodic_columns(stitched, info["legend_pairs"])


def _sample_legend_pairs(main_colors: list[int]) -> tuple[tuple[int, int], tuple[int, int]]:
    shuffled = list(main_colors)
    random.shuffle(shuffled)
    return ((shuffled[0], shuffled[1]), (shuffled[2], shuffled[3]))


def _sample_keys(target_order: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    assignments = KEY_ASSIGNMENTS_BY_ORDER[target_order]
    assert assignments, "target order must have at least one valid key assignment"
    if random.random() < 0.75:
        mixed = [keys for keys in assignments if {len(key) for key in keys} == {2, 3}]
        if mixed:
            return random.choice(mixed)
    return random.choice(assignments)


def _place_markers(panel: np.ndarray, key: tuple[int, ...], main_colors: list[int], panel_index: int) -> tuple[int, tuple[int, ...]]:
    height = panel.shape[0]
    marker_col = random.randrange(PANEL_WIDTH)
    row_pool = list(range(2, height)) if panel_index == 0 else list(range(height))
    rows = tuple(sorted(random.sample(row_pool, len(key))))
    for row, target_panel in zip(rows, key):
        panel[row, marker_col] = int(main_colors[target_panel])
    return marker_col, rows


def _render_input(panels: list[np.ndarray], legend_pairs: tuple[tuple[int, int], tuple[int, int]]) -> np.ndarray:
    grid = np.full((panels[0].shape[0], INPUT_WIDTH), int(Color.BLACK), dtype=int)
    legend = np.array(legend_pairs, dtype=int)

    for panel_id, panel in enumerate(panels):
        c0 = panel_id * (PANEL_WIDTH + SEPARATOR_WIDTH)
        patch = panel.copy()
        if panel_id == 0:
            patch[:2, :2] = legend
        grid[:, c0 : c0 + PANEL_WIDTH] = patch
        if panel_id < PANEL_COUNT - 1:
            grid[:, c0 + PANEL_WIDTH] = int(Color.GREY)
    return grid


def _semantic_sample_ok(sample: dict[str, object]) -> bool:
    inp = sample["input"]
    expected = sample["expected"]
    clean_panels = sample["clean_panels"]
    main_colors = tuple(sample["main_colors"])
    keys = sample["keys"]
    legend_pairs = sample["legend_pairs"]
    order = sample["order"]
    raw_color_keys = sample["raw_color_keys"]

    try:
        parsed = _extract_structure(inp)
        actual = main(inp)
    except Exception:
        return False

    if not np.array_equal(actual, expected):
        return False
    if tuple(parsed["main_colors"]) != main_colors:
        return False
    if tuple(parsed["keys"]) != keys:
        return False
    if tuple(parsed["legend_pairs"]) != legend_pairs:
        return False
    if order == IDENTITY_ORDER:
        return False

    for parsed_panel, clean_panel in zip(parsed["panels"], clean_panels):
        if not np.array_equal(parsed_panel, clean_panel):
            return False

    sorted_panels = np.concatenate([clean_panels[idx] for idx in order], axis=1)
    if np.array_equal(expected, sorted_panels):
        return False

    unsorted_swapped = _swap_periodic_columns(np.concatenate(clean_panels, axis=1), legend_pairs)
    if np.array_equal(expected, unsorted_swapped):
        return False

    raw_color_order = tuple(sorted(range(PANEL_COUNT), key=lambda idx: raw_color_keys[idx]))
    if raw_color_order == order:
        return False
    raw_color_output = _swap_periodic_columns(
        np.concatenate([clean_panels[idx] for idx in raw_color_order], axis=1),
        legend_pairs,
    )
    if np.array_equal(expected, raw_color_output):
        return False

    dominant_color_order = tuple(sorted(range(PANEL_COUNT), key=lambda idx: main_colors[idx]))
    if dominant_color_order == order:
        return False
    dominant_color_output = _swap_periodic_columns(
        np.concatenate([clean_panels[idx] for idx in dominant_color_order], axis=1),
        legend_pairs,
    )
    if np.array_equal(expected, dominant_color_output):
        return False

    swap_col_colors = set(map(int, np.unique(sorted_panels[:, ::PANEL_WIDTH])))
    if swap_col_colors != set(PANEL_PALETTE):
        return False

    for pair in legend_pairs:
        left, right = pair
        if left not in swap_col_colors or right not in swap_col_colors:
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*HEIGHT_RANGE)
    main_colors = random.sample(list(PANEL_PALETTE), PANEL_COUNT)
    legend_pairs = _sample_legend_pairs(main_colors)
    target_order = random.choice(NON_IDENTITY_ORDERS)
    keys = _sample_keys(target_order)

    clean_panels: list[np.ndarray] = []
    raw_color_keys: list[tuple[int, ...]] = []
    for panel_id in range(PANEL_COUNT):
        panel = np.full((height, PANEL_WIDTH), main_colors[panel_id], dtype=int)
        _place_markers(panel, keys[panel_id], main_colors, panel_id)
        clean_panels.append(panel)
        raw_color_keys.append(tuple(main_colors[target_panel] for target_panel in keys[panel_id]))

    inp = _render_input(clean_panels, legend_pairs)
    expected, order = _oracle_output(clean_panels, keys, legend_pairs)
    sample = {
        "input": inp,
        "expected": expected,
        "clean_panels": clean_panels,
        "main_colors": main_colors,
        "legend_pairs": legend_pairs,
        "keys": keys,
        "raw_color_keys": tuple(raw_color_keys),
        "order": order,
    }
    if not _semantic_sample_ok(sample):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
