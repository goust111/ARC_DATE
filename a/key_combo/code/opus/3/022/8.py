import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_h, bbox_w, colors_present, find_components


STRIPE_COLORS = (
    Color.BLUE,
    Color.RED,
    Color.GREEN,
    Color.YELLOW,
    Color.ORANGE,
    Color.TEAL,
)
GEN_ATTEMPTS = 600


def _min_manhattan(coords_a: np.ndarray, coords_b: np.ndarray) -> int:
    diffs = np.abs(coords_a[:, None, :] - coords_b[None, :, :])
    return int(diffs.sum(axis=2).min())


def _find_u_frame(grid: np.ndarray) -> dict:
    grey_components = find_components(
        grid,
        bg=Color.BLACK,
        connectivity=4,
        monochromatic=True,
        colors=[Color.GREY],
        return_format="dict",
    )
    matches = []
    for comp in grey_components:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if bbox_h(comp["bbox"]) < 2 or bbox_w(comp["bbox"]) < 3:
            continue
        mask = np.zeros(grid.shape, dtype=bool)
        coords = comp["coords"]
        mask[coords[:, 0], coords[:, 1]] = True
        sub = mask[r0 : r1 + 1, c0 : c1 + 1]
        if not sub[-1, :].all():
            continue
        if not sub[:, 0].all() or not sub[:, -1].all():
            continue
        if sub[:-1, 1:-1].any():
            continue
        if not np.all(grid[r0:r1, c0 + 1 : c1] == Color.BLACK):
            continue
        matches.append(
            {
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "interior_rows": slice(r0, r1),
                "interior_cols": slice(c0 + 1, c1),
                "interior_h": r1 - r0,
                "interior_w": c1 - c0 - 1,
            }
        )
    assert len(matches) == 1, "expected exactly one gray U-shaped frame"
    return matches[0]


def _find_stripes(grid: np.ndarray) -> tuple[str, list[dict]]:
    stripe_colors = colors_present(grid, exclude={Color.BLACK, Color.GREY})
    assert stripe_colors, "expected colored stripes"
    stripes = find_components(
        grid,
        bg=Color.BLACK,
        connectivity=4,
        monochromatic=True,
        colors=stripe_colors,
        return_format="dict",
    )
    assert len(stripes) == len(stripe_colors), "each stripe color should form one component"

    parsed = []
    for comp in stripes:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        h = bbox_h(comp["bbox"])
        w = bbox_w(comp["bbox"])
        if h == 1 and w >= 2:
            orientation = "horizontal"
        elif w == 1 and h >= 2:
            orientation = "vertical"
        else:
            raise AssertionError("foreground components must be 1-pixel-wide stripes")
        parsed.append(
            {
                "color": int(comp["color"]),
                "coords": comp["coords"],
                "bbox": (r0, c0, r1, c1),
                "orientation": orientation,
            }
        )

    orientations = {stripe["orientation"] for stripe in parsed}
    assert len(orientations) == 1, "all stripes must share one orientation"
    return parsed[0]["orientation"], parsed


def _analyze(grid: np.ndarray) -> tuple[dict, str, list[dict]]:
    frame = _find_u_frame(grid)
    orientation, stripes = _find_stripes(grid)
    expected_count = frame["interior_h"] if orientation == "horizontal" else frame["interior_w"]
    assert len(stripes) == expected_count, "stripe count must match the fill dimension"

    distances = []
    for stripe in stripes:
        dist = _min_manhattan(stripe["coords"], frame["coords"])
        stripe["distance"] = dist
        distances.append(dist)
    assert len(set(distances)) == len(distances), "stripe distances must be unique"
    return frame, orientation, stripes


def _fill_from_sorted_stripes(out: np.ndarray, frame: dict, orientation: str, stripes: list[dict]) -> np.ndarray:
    ordered = sorted(stripes, key=lambda stripe: stripe["distance"])
    rr = frame["interior_rows"]
    cc = frame["interior_cols"]
    if orientation == "horizontal":
        assert len(ordered) == frame["interior_h"]
        for idx, stripe in enumerate(ordered):
            out[rr.start + idx, cc] = stripe["color"]
    else:
        assert len(ordered) == frame["interior_w"]
        for idx, stripe in enumerate(ordered):
            out[rr, cc.start + idx] = stripe["color"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the cavity of a gray U-shaped frame with stripe colors ordered by stripe-to-frame distance.

    Input structure:
    - The grid has a black background, one gray U-shaped frame open upward on the right, and several colored stripes on the left.
    - Every stripe is a single monochrome 4-connected line segment with width 1, and all stripes are either horizontal or vertical.
    - The U-frame encloses a black rectangular cavity whose row count or column count matches the number of stripes, depending on stripe orientation.

    Rule:
    - Detect the shared stripe orientation.
    - For each stripe, measure the minimum Manhattan distance from any stripe pixel to any gray frame pixel.
    - Sort stripes from nearest to farthest.
    - Preserve the whole input, then fill the frame cavity row by row if the stripes are horizontal, or column by column if the stripes are vertical, using the sorted stripe colors.

    Key insight:
    - The fill axis comes from stripe orientation, but the fill order comes from stripe-to-frame distance rather than from the stripes' own row or column positions.
    """
    g = np.asarray(grid, dtype=int)
    frame, orientation, stripes = _analyze(g)
    out = g.copy()
    return _fill_from_sorted_stripes(out, frame, orientation, stripes)


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0] = Color.GREY
    grid[r0 : r1 + 1, c1] = Color.GREY
    grid[r1, c0 : c1 + 1] = Color.GREY


def _wrong_order_colors(stripes: list[dict], orientation: str) -> tuple[list[int], list[int], list[int]]:
    distance_colors = [stripe["color"] for stripe in sorted(stripes, key=lambda stripe: stripe["distance"])]
    if orientation == "horizontal":
        position_colors = [stripe["color"] for stripe in sorted(stripes, key=lambda stripe: stripe["bbox"][0])]
        proximity_colors = [stripe["color"] for stripe in sorted(stripes, key=lambda stripe: -stripe["bbox"][3])]
    else:
        position_colors = [stripe["color"] for stripe in sorted(stripes, key=lambda stripe: stripe["bbox"][1])]
        proximity_colors = [stripe["color"] for stripe in sorted(stripes, key=lambda stripe: -stripe["bbox"][1])]
    return distance_colors, position_colors, proximity_colors


def _validate_generated_grid(grid: np.ndarray) -> bool:
    try:
        frame, orientation, stripes = _analyze(grid)
        if len({stripe["color"] for stripe in stripes}) != len(stripes):
            return False

        distance_colors, position_colors, proximity_colors = _wrong_order_colors(stripes, orientation)
        if distance_colors == position_colors:
            return False
        if distance_colors == proximity_colors:
            return False

        expected = grid.copy()
        _fill_from_sorted_stripes(expected, frame, orientation, stripes)
        actual = main(grid)
        if not np.array_equal(actual, expected):
            return False

        keep_mask = np.ones(grid.shape, dtype=bool)
        keep_mask[frame["interior_rows"], frame["interior_cols"]] = False
        if not np.array_equal(actual[keep_mask], grid[keep_mask]):
            return False
        if np.any(actual[frame["interior_rows"], frame["interior_cols"]] == Color.BLACK):
            return False
        return True
    except AssertionError:
        return False


def _gen_horizontal() -> np.ndarray | None:
    interior_h = random.randint(3, len(STRIPE_COLORS))
    interior_w = random.randint(3, 6)
    frame_h = interior_h + 1
    frame_w = interior_w + 2

    left_width = random.randint(8, 13)
    right_margin = random.randint(1, 3)
    width = left_width + frame_w + right_margin

    min_height = max(13, 2 * interior_h + 1, frame_h + 2)
    height = random.randint(min_height, 24)
    frame_r0 = random.randint(1, height - frame_h - 1)
    frame_c0 = left_width
    frame_bbox = (frame_r0, frame_c0, frame_r0 + frame_h - 1, frame_c0 + frame_w - 1)

    grid = np.full((height, width), Color.BLACK, dtype=int)
    _draw_frame(grid, frame_bbox)

    candidate_rows = list(range(0, height, 2))
    candidate_end_cols = list(range(1, frame_c0))
    if len(candidate_rows) < interior_h or len(candidate_end_cols) < interior_h:
        return None

    rows = random.sample(candidate_rows, interior_h)
    end_cols = random.sample(candidate_end_cols, interior_h)
    colors = random.sample(STRIPE_COLORS, interior_h)

    for row, end_col, color in zip(rows, end_cols, colors):
        length = random.randint(2, min(6, end_col + 1))
        start_col = end_col - length + 1
        grid[row, start_col : end_col + 1] = color

    return grid if _validate_generated_grid(grid) else None


def _gen_vertical() -> np.ndarray | None:
    interior_h = random.randint(3, 6)
    interior_w = random.randint(3, len(STRIPE_COLORS))
    frame_h = interior_h + 1
    frame_w = interior_w + 2

    left_width = random.randint(max(2 * interior_w + 1, 9), 14)
    right_margin = random.randint(1, 3)
    width = left_width + frame_w + right_margin

    top_margin = random.randint(max(6, interior_w + 2), 12)
    bottom_margin = random.randint(1, 4)
    height = top_margin + frame_h + bottom_margin
    frame_r0 = top_margin
    frame_c0 = left_width
    frame_bbox = (frame_r0, frame_c0, frame_r0 + frame_h - 1, frame_c0 + frame_w - 1)

    grid = np.full((height, width), Color.BLACK, dtype=int)
    _draw_frame(grid, frame_bbox)

    candidate_cols = list(range(0, frame_c0, 2))
    if len(candidate_cols) < interior_w or frame_r0 < 2:
        return None

    cols = random.sample(candidate_cols, interior_w)
    colors = random.sample(STRIPE_COLORS, interior_w)

    for col, color in zip(cols, colors):
        length = random.randint(2, min(6, frame_r0))
        bottom = random.randint(length - 1, frame_r0 - 1)
        top = bottom - length + 1
        grid[top : bottom + 1, col] = color

    return grid if _validate_generated_grid(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_horizontal() if random.random() < 0.5 else _gen_vertical()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stripe-distance U-frame task")
