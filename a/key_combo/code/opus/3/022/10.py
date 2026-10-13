from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox_h, bbox_w, find_components


INPUT_COLORS = [Color.BLUE, Color.RED, Color.GREEN, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
HEIGHT_RANGE = (3, 6)
WIDTH_RANGE = (3, 6)
GAP_RANGE = (4, 11)
ROW_MARGIN_RANGE = (0, 3)
ROW_GAP_RANGE = (1, 4)
WIDTH_SLACK_RANGE = (0, 4)
MAX_GENERATION_ATTEMPTS = 400


def _rect_from_component(grid: np.ndarray, component: dict) -> dict:
    color = int(component["color"])
    r0, c0, r1, c1 = map(int, component["bbox"])
    height = bbox_h(component["bbox"])
    width = bbox_w(component["bbox"])
    assert int(component["area"]) == height * width, "foreground objects must be solid rectangles"
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color), "rectangle bbox must be uniformly filled"
    return {
        "color": color,
        "r0": r0,
        "c0": c0,
        "r1": r1,
        "c1": c1,
        "h": int(height),
        "w": int(width),
    }


def _pair_rectangles(grid: np.ndarray) -> list[dict]:
    colors = sorted(int(color) for color in np.unique(grid) if int(color) != int(Color.BLACK))
    assert len(colors) == 2, "expected exactly two foreground colors"
    assert int(Color.YELLOW) not in colors, "input rectangles must not already use the output interior color"

    components = find_components(
        grid,
        bg=Color.BLACK,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    assert len(components) == 4, "expected exactly four rectangles"

    grouped: dict[int, list[dict]] = {}
    for component in components:
        rect = _rect_from_component(grid, component)
        grouped.setdefault(rect["color"], []).append(rect)

    assert set(grouped) == set(colors), "all foreground colors must belong to rectangles"

    pairs = []
    for color in colors:
        rects = grouped[color]
        assert len(rects) == 2, "each color must appear on exactly two rectangles"
        left, right = sorted(rects, key=lambda rect: rect["c0"])
        assert left["r0"] == right["r0"] and left["r1"] == right["r1"], "paired rectangles must share a row band"
        assert left["h"] == right["h"], "paired rectangles must have the same height"
        gap = right["c0"] - left["c1"] - 1
        assert gap >= 4, "paired rectangles must be separated by at least four columns"
        pairs.append(
            {
                "color": int(color),
                "left": left,
                "right": right,
                "r0": left["r0"],
                "r1": left["r1"],
                "gap": int(gap),
            }
        )

    pairs.sort(key=lambda pair: (pair["r0"], pair["left"]["c0"]))
    for upper, lower in zip(pairs, pairs[1:]):
        assert upper["r1"] + 1 < lower["r0"], "row bands must be separated by at least one blank row"
    return pairs


def _merged_bbox(pair: dict) -> tuple[int, int, int, int]:
    left = pair["left"]
    right = pair["right"]
    black_gap = int(pair["gap"])
    edge_distance = black_gap + 1
    left_shift = (edge_distance - 1) // 2
    right_shift = edge_distance // 2 - 1

    moved_left_end = left["c1"] + left_shift
    moved_right_start = right["c0"] - right_shift
    assert moved_right_start - moved_left_end - 1 == 1, "sliding must leave exactly one bridge column"

    c0 = left["c0"] + left_shift
    c1 = right["c1"] - right_shift
    width = c1 - c0 + 1
    assert width == left["w"] + right["w"] + 1, "merged width must depend on both rectangle widths and the bridge"
    return int(left["r0"]), int(c0), int(left["r1"]), int(c1)


def _paint_merged_pair(out: np.ndarray, pair: dict) -> None:
    r0, c0, r1, c1 = _merged_bbox(pair)
    color = int(pair["color"])
    assert np.all(out[r0 : r1 + 1, c0 : c1 + 1] == Color.BLACK), "merged pairs must not overlap"
    out[r0 : r1 + 1, c0 : c1 + 1] = color
    if (r1 - r0 + 1) >= 3 and (c1 - c0 + 1) >= 3:
        out[r0 + 1 : r1, c0 + 1 : c1] = Color.YELLOW


def _expected_output_from_pairs(shape: tuple[int, int], pairs: list[dict]) -> np.ndarray:
    out = np.full(shape, Color.BLACK, dtype=int)
    for pair in pairs:
        _paint_merged_pair(out, pair)
    return out


def _strict_interior_area(height: int, width: int) -> int:
    return max(0, height - 2) * max(0, width - 2)


def _validate_emergent_interior(pair: dict) -> None:
    left = pair["left"]
    right = pair["right"]
    merged = _merged_bbox(pair)
    merged_height = merged[2] - merged[0] + 1
    merged_width = merged[3] - merged[1] + 1

    original_interior = _strict_interior_area(left["h"], left["w"]) + _strict_interior_area(right["h"], right["w"])
    merged_interior = _strict_interior_area(merged_height, merged_width)
    assert merged_interior > original_interior, "merging must create new interior pixels beyond the original interiors"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each same-color opposite-side rectangle pair together, bridge the one-column gap, and recolor the merged interior yellow.

    Input structure:
    - The grid has a black background and exactly two foreground colors.
    - Each foreground color appears on exactly two solid 4-connected rectangles.
    - Within each color pair, one rectangle is left, one is right; they share one row band and have the same height.

    Rule:
    - Pair rectangles by color.
    - Move the left rectangle rightward and the right rectangle leftward so they end with exactly one blank column between them; when the original edge-to-edge distance is odd, the left rectangle moves one extra cell.
    - Fill that one-column gap with the pair color so the pair becomes one solid rectangle.
    - Recompute border versus strict interior on the merged rectangle and recolor only the strict interior to yellow, leaving the outer ring in the original color.

    Key insight:
    - The yellow region is defined after the two rectangles are merged, so pixels next to the new bridge can become interior even though they started as border pixels on separate rectangles.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    pairs = _pair_rectangles(g)
    out = _expected_output_from_pairs(g.shape, pairs)
    return out


def _sample_gap(parity: int) -> int:
    choices = [gap for gap in range(GAP_RANGE[0], GAP_RANGE[1] + 1) if gap % 2 == parity]
    assert choices
    return int(random.choice(choices))


def _sample_pair_specs() -> list[dict]:
    colors = random.sample(INPUT_COLORS, 2)
    parity_patterns = [(0, 1), (1, 0), (0, 0), (1, 1)]
    parity_weights = [3, 3, 1, 1]
    parities = random.choices(parity_patterns, weights=parity_weights, k=1)[0]

    pairs = []
    for idx, color in enumerate(colors):
        height = random.randint(*HEIGHT_RANGE)
        left_width = random.randint(*WIDTH_RANGE)
        right_width = random.randint(*WIDTH_RANGE)
        gap = _sample_gap(parities[idx])
        left_margin = random.randint(*ROW_MARGIN_RANGE)
        right_margin = random.randint(*ROW_MARGIN_RANGE)
        required_width = left_margin + left_width + gap + right_width + right_margin
        pairs.append(
            {
                "color": int(color),
                "h": int(height),
                "left_w": int(left_width),
                "right_w": int(right_width),
                "gap": int(gap),
                "left_margin": int(left_margin),
                "right_margin": int(right_margin),
                "required_width": int(required_width),
            }
        )

    random.shuffle(pairs)
    return pairs


def _build_grid_from_specs(specs: list[dict]) -> np.ndarray | None:
    top_margin = random.randint(*ROW_MARGIN_RANGE)
    middle_gap = random.randint(*ROW_GAP_RANGE)
    bottom_margin = random.randint(*ROW_MARGIN_RANGE)
    height = top_margin + specs[0]["h"] + middle_gap + specs[1]["h"] + bottom_margin
    if height > 30:
        return None

    width = max(spec["required_width"] for spec in specs)
    width += random.randint(*WIDTH_SLACK_RANGE)
    if width > 30:
        return None

    grid = np.full((height, width), Color.BLACK, dtype=int)
    row_starts = [top_margin, top_margin + specs[0]["h"] + middle_gap]

    for row_start, spec in zip(row_starts, specs):
        slack = width - spec["required_width"]
        offset = random.randint(0, slack)
        left_c0 = offset + spec["left_margin"]
        right_c0 = left_c0 + spec["left_w"] + spec["gap"]

        grid[row_start : row_start + spec["h"], left_c0 : left_c0 + spec["left_w"]] = spec["color"]
        grid[row_start : row_start + spec["h"], right_c0 : right_c0 + spec["right_w"]] = spec["color"]

    return grid


def _oracle_output_from_grid(grid: np.ndarray) -> np.ndarray:
    parsed_pairs = _pair_rectangles(grid)
    for pair in parsed_pairs:
        _validate_emergent_interior(pair)
    return _expected_output_from_pairs(grid.shape, parsed_pairs)


def _gen_attempt() -> np.ndarray | None:
    specs = _sample_pair_specs()
    grid = _build_grid_from_specs(specs)
    if grid is None:
        return None

    try:
        expected = _oracle_output_from_grid(grid)
        actual = main(grid)
    except Exception:
        return None

    if not np.array_equal(actual, expected):
        return None

    odd_count = sum(int(pair["gap"]) % 2 for pair in _pair_rectangles(grid))
    assert odd_count in {0, 1, 2}
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid opposite-pair sliding rectangle puzzle")
