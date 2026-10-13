from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

MARKER_COLORS = (BLUE, RED, YELLOW, ORANGE)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_GRID_RANGE = (18, 29)
GEN_FRAME_COUNT_CHOICES = (2, 3)
GEN_MAX_ATTEMPTS = 500

TALL_INTERIORS = [(6, 4), (8, 4), (8, 6)]
WIDE_INTERIORS = [(4, 6), (4, 8), (6, 8)]
SQUARE_INTERIORS = [(4, 4), (6, 6), (8, 8)]


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _rect_border_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = bbox
    out[r0 : r1 + 1, c0] = True
    out[r0 : r1 + 1, c1] = True
    out[r0, c0 : c1 + 1] = True
    out[r1, c0 : c1 + 1] = True
    return out


def _marker_class_for_frame(marker: tuple[int, int], bbox: tuple[int, int, int, int]) -> str | None:
    r, c = marker
    r0, c0, r1, c1 = bbox
    if r0 <= r <= r1 or c0 <= c <= c1:
        return None
    center_r2 = r0 + r1
    center_c2 = c0 + c1
    same_side = (2 * r < center_r2) == (2 * c < center_c2)
    return "main" if same_side else "anti"


def _marker_distance_to_frame(marker: tuple[int, int], bbox: tuple[int, int, int, int]) -> int | None:
    diagonal_class = _marker_class_for_frame(marker, bbox)
    if diagonal_class is None:
        return None
    r, c = marker
    r0, c0, r1, c1 = bbox
    corners = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
    return min(abs(r - rr) + abs(c - cc) for rr, cc in corners)


def _extract_frame_objects(grid: np.ndarray, *, expect_empty_interior: bool) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)

    frame_components = _components_mask(g == GREY)
    assert 2 <= len(frame_components) <= 3, "expected 2-3 grey frame components"

    frames = []
    for comp in frame_components:
        bbox = _bbox(comp)
        r0, c0, r1, c1 = bbox
        border_mask = _rect_border_mask(g.shape, bbox)
        comp_mask = np.zeros_like(g, dtype=bool)
        comp_mask[comp[:, 0], comp[:, 1]] = True
        assert np.array_equal(comp_mask, border_mask), "each frame must be a 1px hollow rectangle"

        interior_h = r1 - r0 - 1
        interior_w = c1 - c0 - 1
        assert interior_h >= 4 and interior_w >= 4
        assert interior_h % 2 == 0 and interior_w % 2 == 0
        if expect_empty_interior:
            assert np.all(g[r0 + 1 : r1, c0 + 1 : c1] == bg), "frame interior must start empty"

        frames.append(
            {
                "bbox": bbox,
                "interior_h": interior_h,
                "interior_w": interior_w,
                "markers": {},
                "marker_positions": {},
            }
        )

    marker_positions = np.argwhere((g != bg) & (g != GREY))
    assert marker_positions.shape[0] == 2 * len(frames), "expected exactly two markers per frame"

    for pos in marker_positions:
        marker = (int(pos[0]), int(pos[1]))
        color = int(g[marker])
        assert color in MARKER_COLORS, "unexpected marker color"
        candidates = []
        for idx, frame in enumerate(frames):
            dist = _marker_distance_to_frame(marker, frame["bbox"])
            if dist is not None:
                candidates.append((dist, idx))
        assert candidates, "marker must sit in a diagonal quadrant of some frame"
        candidates.sort()
        best_dist, best_idx = candidates[0]
        assert len(candidates) == 1 or best_dist < candidates[1][0], "marker-to-frame assignment must be unique"

        diagonal_class = _marker_class_for_frame(marker, frames[best_idx]["bbox"])
        assert diagonal_class not in frames[best_idx]["markers"], "need one main and one anti marker"
        frames[best_idx]["markers"][diagonal_class] = color
        frames[best_idx]["marker_positions"][diagonal_class] = marker

    for frame in frames:
        assert set(frame["markers"]) == {"main", "anti"}
        assert frame["markers"]["main"] != frame["markers"]["anti"]

    return frames


def _paint_frame_interior(out: np.ndarray, frame: dict) -> None:
    r0, c0, r1, c1 = frame["bbox"]
    interior_h = frame["interior_h"]
    interior_w = frame["interior_w"]
    main_color = frame["markers"]["main"]
    anti_color = frame["markers"]["anti"]

    for dr in range(interior_h):
        r = r0 + 1 + dr
        bottom_half = dr >= interior_h // 2
        for dc in range(interior_w):
            c = c0 + 1 + dc
            right_half = dc >= interior_w // 2
            out[r, c] = main_color if bottom_half == right_half else anti_color

    if interior_h > interior_w:
        mid_c = c0 + interior_w // 2
        out[r0 + 1 : r1, mid_c] = GREEN
    else:
        mid_r = r0 + interior_h // 2
        out[mid_r, c0 + 1 : c1] = GREEN


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each hollow frame by marker-defined diagonal quadrants, then recolor the frame's longer-axis middle line green.

    Input structure:
    - A black-dominant grid contains 2-3 grey one-pixel-thick hollow rectangular frames.
    - Every frame has an even-by-even empty interior and exactly two single-pixel markers outside it.
    - The two markers use different colors from the visible marker palette and lie in different diagonal quadrants relative to that frame's center.

    Rule:
    - For each frame, identify which marker is on the main-diagonal side (northwest or southeast) and which is on the anti-diagonal side (northeast or southwest).
    - Split the frame interior into four equal quadrants and fill top-left plus bottom-right with the main-diagonal marker color, and top-right plus bottom-left with the anti-diagonal marker color.
    - After filling, recolor the frame's interior middle line green: use the vertical middle column when the interior is taller than wide, otherwise use the horizontal middle row, taking the upper/left member of the central pair.

    Key insight:
    - Marker position determines the diagonal color assignment, but the green separator is not marker-driven: it is chosen only after the quadrant fill, from the frame's interior aspect ratio.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames = _extract_frame_objects(g, expect_empty_interior=True)
    out = g.copy()
    for frame in frames:
        _paint_frame_interior(out, frame)
    return out


def _rectangles_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _choose_layout_types(count: int) -> list[str]:
    if count == 2:
        layout = ["tall", "wide"]
    else:
        layout = ["tall", "wide", "square"]
    random.shuffle(layout)
    return layout


def _choose_interior_dims(kind: str) -> tuple[int, int]:
    if kind == "tall":
        return random.choice(TALL_INTERIORS)
    if kind == "wide":
        return random.choice(WIDE_INTERIORS)
    return random.choice(SQUARE_INTERIORS)


def _sample_frame_specs(count: int) -> list[dict]:
    for _ in range(50):
        specs = []
        for kind in _choose_layout_types(count):
            interior_h, interior_w = _choose_interior_dims(kind)
            colors = random.sample(MARKER_COLORS, 2)
            specs.append(
                {
                    "kind": kind,
                    "interior_h": interior_h,
                    "interior_w": interior_w,
                    "outer_h": interior_h + 2,
                    "outer_w": interior_w + 2,
                    "main_color": colors[0],
                    "anti_color": colors[1],
                    "main_corner": random.choice(["NW", "SE"]),
                    "anti_corner": random.choice(["NE", "SW"]),
                }
            )
        if len({color for s in specs for color in (s["main_color"], s["anti_color"])}) >= 3:
            return specs
    raise RuntimeError("failed to sample sufficiently varied marker colors")


def _marker_corner(spec: dict, bbox: tuple[int, int, int, int], diagonal_class: str) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if diagonal_class == "main":
        return (r0 - 1, c0 - 1) if spec["main_corner"] == "NW" else (r1 + 1, c1 + 1)
    return (r0 - 1, c1 + 1) if spec["anti_corner"] == "NE" else (r1 + 1, c0 - 1)


def _place_specs_on_grid(h: int, w: int, specs: list[dict]) -> list[dict] | None:
    placed: list[dict] = []
    reserved: list[tuple[int, int, int, int]] = []

    order = specs[:]
    random.shuffle(order)
    for spec in order:
        oh = spec["outer_h"]
        ow = spec["outer_w"]
        if oh + 2 > h or ow + 2 > w:
            return None

        for _ in range(250):
            r0 = random.randint(1, h - oh - 1)
            c0 = random.randint(1, w - ow - 1)
            bbox = (r0, c0, r0 + oh - 1, c0 + ow - 1)
            reserve = (max(0, r0 - 2), max(0, c0 - 2), min(h - 1, bbox[2] + 2), min(w - 1, bbox[3] + 2))
            if any(_rectangles_overlap(reserve, old) for old in reserved):
                continue

            main_marker = _marker_corner(spec, bbox, "main")
            anti_marker = _marker_corner(spec, bbox, "anti")
            if not (0 <= main_marker[0] < h and 0 <= main_marker[1] < w):
                continue
            if not (0 <= anti_marker[0] < h and 0 <= anti_marker[1] < w):
                continue

            spec_copy = dict(spec)
            spec_copy["bbox"] = bbox
            spec_copy["main_marker"] = main_marker
            spec_copy["anti_marker"] = anti_marker
            placed.append(spec_copy)
            reserved.append(reserve)
            break
        else:
            return None

    return placed


def _render_input(h: int, w: int, placed_specs: list[dict]) -> np.ndarray:
    grid = np.full((h, w), BLACK, dtype=int)
    for spec in placed_specs:
        r0, c0, r1, c1 = spec["bbox"]
        grid[r0 : r1 + 1, c0] = GREY
        grid[r0 : r1 + 1, c1] = GREY
        grid[r0, c0 : c1 + 1] = GREY
        grid[r1, c0 : c1 + 1] = GREY
        grid[spec["main_marker"]] = spec["main_color"]
        grid[spec["anti_marker"]] = spec["anti_color"]
    return grid


def _frame_wrong_axis_output(inp: np.ndarray, frame: dict) -> np.ndarray:
    out = inp.copy()
    r0, c0, r1, c1 = frame["bbox"]
    interior_h = frame["interior_h"]
    interior_w = frame["interior_w"]
    main_color = frame["markers"]["main"]
    anti_color = frame["markers"]["anti"]

    for dr in range(interior_h):
        r = r0 + 1 + dr
        bottom_half = dr >= interior_h // 2
        for dc in range(interior_w):
            c = c0 + 1 + dc
            right_half = dc >= interior_w // 2
            out[r, c] = main_color if bottom_half == right_half else anti_color

    if interior_h > interior_w:
        wrong_mid_r = r0 + interior_h // 2
        out[wrong_mid_r, c0 + 1 : c1] = GREEN
    else:
        wrong_mid_c = c0 + interior_w // 2
        out[r0 + 1 : r1, wrong_mid_c] = GREEN
    return out


def _validate_generated_case(inp: np.ndarray) -> bool:
    try:
        frames = _extract_frame_objects(inp, expect_empty_interior=True)
        out = main(inp)
    except AssertionError:
        return False

    kinds = {"tall": 0, "wide": 0, "square": 0}
    for frame in frames:
        h = frame["interior_h"]
        w = frame["interior_w"]
        if h > w:
            kinds["tall"] += 1
            mid_c = frame["bbox"][1] + w // 2
            if not np.all(out[frame["bbox"][0] + 1 : frame["bbox"][2], mid_c] == GREEN):
                return False
        elif w > h:
            kinds["wide"] += 1
            mid_r = frame["bbox"][0] + h // 2
            if not np.all(out[mid_r, frame["bbox"][1] + 1 : frame["bbox"][3]] == GREEN):
                return False
        else:
            kinds["square"] += 1
            mid_r = frame["bbox"][0] + h // 2
            if not np.all(out[mid_r, frame["bbox"][1] + 1 : frame["bbox"][3]] == GREEN):
                return False

        r0, c0, r1, c1 = frame["bbox"]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        visible_colors = set(map(int, interior.ravel()))
        if GREEN not in visible_colors or frame["markers"]["main"] not in visible_colors or frame["markers"]["anti"] not in visible_colors:
            return False

        swapped = inp.copy()
        main_pos = frame["marker_positions"]["main"]
        anti_pos = frame["marker_positions"]["anti"]
        swapped[main_pos], swapped[anti_pos] = swapped[anti_pos], swapped[main_pos]
        if np.array_equal(main(swapped), out):
            return False

        wrong = _frame_wrong_axis_output(inp, frame)
        if np.array_equal(
            wrong[r0 + 1 : r1, c0 + 1 : c1],
            out[r0 + 1 : r1, c0 + 1 : c1],
        ):
            return False

    if len(frames) == 2 and not (kinds["tall"] == 1 and kinds["wide"] == 1):
        return False
    if len(frames) == 3 and not (kinds["tall"] == 1 and kinds["wide"] == 1 and kinds["square"] == 1):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    count = random.choice(GEN_FRAME_COUNT_CHOICES)
    specs = _sample_frame_specs(count)
    h = random.randint(*GEN_GRID_RANGE)
    w = random.randint(*GEN_GRID_RANGE)
    placed = _place_specs_on_grid(h, w, specs)
    if placed is None:
        return None

    inp = _render_input(h, w, placed)
    if not _validate_generated_case(inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
