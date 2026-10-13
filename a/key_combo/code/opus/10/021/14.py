from __future__ import annotations

import random

import numpy as np


BG = 0
MARKER = 2
BODY_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)

GEN_GRID_MIN = 14
GEN_GRID_MAX = 18
GEN_RECT_COUNT_OPTIONS = (4, 5)
GEN_COUNTS = (1, 2, 3, 4, 5)
GEN_MAX_ATTEMPTS = 500
GEN_DIM_OPTIONS = (
    (3, 3),
    (3, 4),
    (4, 3),
    (3, 5),
    (5, 3),
    (4, 4),
    (4, 5),
    (5, 4),
    (5, 5),
)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_rectangles(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    rects: list[dict] = []
    for coords in _components(g != BG):
        r0, c0, r1, c1 = _bbox(coords)
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(patch != BG), "foreground component must fill its bbox"

        body_colors = sorted(set(map(int, patch.flatten())) - {MARKER})
        assert len(body_colors) == 1, "each rectangle must have one body color"
        color = int(body_colors[0])
        assert color != BG and color != MARKER

        marker_count = int(np.count_nonzero(patch == MARKER))
        assert marker_count >= 1, "each rectangle must contain at least one marker"

        rects.append(
            {
                "bbox": (r0, c0, r1, c1),
                "height": int(r1 - r0 + 1),
                "width": int(c1 - c0 + 1),
                "area": int(coords.shape[0]),
                "color": color,
                "marker_count": marker_count,
                "patch": patch.copy(),
            }
        )
    return rects


def _histogram_from_rectangles(rects: list[dict]) -> np.ndarray:
    counts = [int(rect["marker_count"]) for rect in rects]
    assert counts and len(counts) == len(set(counts)), "marker counts must be unique"

    ordered = sorted(rects, key=lambda rect: int(rect["marker_count"]))
    out_h = max(counts)
    out_w = len(ordered)
    out = np.zeros((out_h, out_w), dtype=int)
    for col, rect in enumerate(ordered):
        m = int(rect["marker_count"])
        out[out_h - m :, col] = int(rect["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort colored rectangles by their internal red-marker counts and emit color bars with matching heights.

    Input structure:
    - The background is black.
    - The foreground contains 4-5 non-touching solid axis-aligned rectangles in distinct body colors.
    - Inside each rectangle, a sparse set of red pixels replaces some interior body pixels but still belongs to the same rectangle.

    Rule:
    - Treat every 4-connected non-black component as one rectangle, using the combined body-color and red pixels to define its bbox.
    - Count the red marker pixels inside each rectangle and use that count as the rectangle's unique key.
    - Sort the rectangles by increasing marker count, then output one vertical bar per rectangle whose color is the rectangle body color and whose height equals that marker count.

    Key insight:
    - The red pixels are not output directly; their count simultaneously determines both the left-to-right order of the bars and each bar's height.
    """
    rects = _extract_rectangles(grid)
    assert 1 <= len(rects) <= 9, "expected at least one rectangle"
    return _histogram_from_rectangles(rects)


def _dims_for_count(count: int) -> list[tuple[int, int]]:
    options = []
    for h, w in GEN_DIM_OPTIONS:
        interior = (h - 2) * (w - 2)
        area = h * w
        if interior >= count and count * 2 < area:
            options.append((h, w))
    return options


def _sample_specs() -> list[dict]:
    n_rects = random.choice(GEN_RECT_COUNT_OPTIONS)
    counts = random.sample(GEN_COUNTS, n_rects)
    colors = random.sample(BODY_COLORS, n_rects)

    specs = []
    for count, color in zip(counts, colors):
        h, w = random.choice(_dims_for_count(int(count)))
        specs.append({"count": int(count), "color": int(color), "h": int(h), "w": int(w)})
    random.shuffle(specs)
    return specs


def _candidate_positions(occupied: np.ndarray, h: int, w: int) -> list[tuple[int, int]]:
    grid_h, grid_w = occupied.shape
    positions = []
    for r in range(grid_h - h + 1):
        for c in range(grid_w - w + 1):
            r0 = max(0, r - 1)
            c0 = max(0, c - 1)
            r1 = min(grid_h, r + h + 1)
            c1 = min(grid_w, c + w + 1)
            if not np.any(occupied[r0:r1, c0:c1]):
                positions.append((r, c))
    random.shuffle(positions)
    return positions


def _place_specs(specs: list[dict], grid_h: int, grid_w: int) -> list[dict] | None:
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    placed: list[dict] = []
    for spec in sorted(specs, key=lambda item: item["h"] * item["w"], reverse=True):
        positions = _candidate_positions(occupied, int(spec["h"]), int(spec["w"]))
        if not positions:
            return None
        r, c = random.choice(positions)
        occupied[r : r + int(spec["h"]), c : c + int(spec["w"])] = True
        placed.append({**spec, "r": int(r), "c": int(c)})
    return placed


def _marker_coords(spec: dict) -> list[tuple[int, int]]:
    r = int(spec["r"])
    c = int(spec["c"])
    h = int(spec["h"])
    w = int(spec["w"])
    cells = [(rr, cc) for rr in range(r + 1, r + h - 1) for cc in range(c + 1, c + w - 1)]
    return random.sample(cells, int(spec["count"]))


def _render_grid(grid_h: int, grid_w: int, specs: list[dict]) -> np.ndarray:
    grid = np.zeros((grid_h, grid_w), dtype=int)
    for spec in specs:
        r = int(spec["r"])
        c = int(spec["c"])
        h = int(spec["h"])
        w = int(spec["w"])
        color = int(spec["color"])
        grid[r : r + h, c : c + w] = color
        for mr, mc in _marker_coords(spec):
            grid[mr, mc] = MARKER
    return grid


def _expected_output(specs: list[dict]) -> np.ndarray:
    ordered = sorted(specs, key=lambda item: int(item["count"]))
    height = max(int(spec["count"]) for spec in ordered)
    out = np.zeros((height, len(ordered)), dtype=int)
    for col, spec in enumerate(ordered):
        count = int(spec["count"])
        out[height - count :, col] = int(spec["color"])
    return out


def _order_signature(specs: list[dict], key) -> list[int]:
    ordered = sorted(specs, key=key)
    return [int(spec["count"]) for spec in ordered]


def _semantic_checks(grid: np.ndarray, specs: list[dict], expected: np.ndarray) -> bool:
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False

    parsed = _extract_rectangles(grid)
    if len(parsed) != len(specs):
        return False

    parsed_counts = sorted(int(rect["marker_count"]) for rect in parsed)
    spec_counts = sorted(int(spec["count"]) for spec in specs)
    if parsed_counts != spec_counts:
        return False

    parsed_colors = sorted(int(rect["color"]) for rect in parsed)
    spec_colors = sorted(int(spec["color"]) for spec in specs)
    if parsed_colors != spec_colors:
        return False

    for spec in specs:
        patch = grid[
            int(spec["r"]) : int(spec["r"]) + int(spec["h"]),
            int(spec["c"]) : int(spec["c"]) + int(spec["w"]),
        ]
        if np.any(patch[0, :] == MARKER) or np.any(patch[-1, :] == MARKER):
            return False
        if np.any(patch[:, 0] == MARKER) or np.any(patch[:, -1] == MARKER):
            return False
        if int(np.count_nonzero(patch == MARKER)) != int(spec["count"]):
            return False

    count_order = _order_signature(specs, key=lambda item: int(item["count"]))
    left_order = _order_signature(specs, key=lambda item: (int(item["c"]), int(item["r"])))
    top_order = _order_signature(specs, key=lambda item: (int(item["r"]), int(item["c"])))
    area_order = _order_signature(specs, key=lambda item: (int(item["h"]) * int(item["w"]), int(item["c"])))

    if count_order == left_order:
        return False
    if count_order == top_order:
        return False
    if count_order == area_order:
        return False

    widths = [int(spec["w"]) for spec in specs]
    heights = [int(spec["h"]) for spec in specs]
    counts = [int(spec["count"]) for spec in specs]
    if widths == counts or heights == counts:
        return False

    try:
        solved = main(grid)
    except AssertionError:
        return False
    return np.array_equal(solved, expected)


def _gen_attempt() -> np.ndarray | None:
    specs = _sample_specs()
    min_side = max(max(int(spec["h"]) for spec in specs), max(int(spec["w"]) for spec in specs))
    grid_h = random.randint(max(GEN_GRID_MIN, min_side + 2), GEN_GRID_MAX)
    grid_w = random.randint(max(GEN_GRID_MIN, min_side + 2), GEN_GRID_MAX)

    placed = _place_specs(specs, grid_h, grid_w)
    if placed is None:
        return None

    grid = _render_grid(grid_h, grid_w, placed)
    expected = _expected_output(placed)
    if not _semantic_checks(grid, placed, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_MAX_ATTEMPTS} attempts")
