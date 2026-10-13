from __future__ import annotations

import random
from collections import Counter

import numpy as np


TRACK_COLOR = 5
BG_COLOR = 0
RECTANGLE_SOURCE_COLORS = (1, 2, 3)
DISTRACTOR_COLORS = (1, 2, 3, 4)
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _neighbors8(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if dr != 0 or dc != 0
    )


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r0 in range(h):
        for c0 in range(w):
            color = int(grid[r0, c0])
            if color == bg or seen[r0, c0]:
                continue

            stack = [(r0, c0)]
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []

            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [r for r, _ in coords]
            cols = [c for _, c in coords]
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "coord_set": set(coords),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "area": len(coords),
                }
            )

    return components


def _border_coords(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    border = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0, r1 + 1):
        border.add((r, c0))
        border.add((r, c1))
    return border


def _is_hollow_rectangle(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    expected = _border_coords(comp["bbox"])
    return comp["coord_set"] == expected


def _extract_scene(grid: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    bg = _get_bg(grid)
    rectangles: list[dict] = []
    pixels: list[dict] = []

    for comp in _find_components(grid, bg):
        if _is_hollow_rectangle(comp):
            r0, c0, r1, c1 = comp["bbox"]
            rectangles.append(
                {
                    "color": int(comp["color"]),
                    "coords": comp["coords"],
                    "bbox": comp["bbox"],
                    "center_r": (r0 + r1) // 2,
                    "center_c": (c0 + c1) // 2,
                    "height": r1 - r0 + 1,
                    "width": c1 - c0 + 1,
                }
            )
        elif comp["area"] == 1:
            r, c = comp["coords"][0]
            pixels.append({"color": int(comp["color"]), "r": r, "c": c})
        else:
            raise ValueError("input contains a non-rectangle component with area > 1")

    rectangles.sort(key=lambda rect: (rect["center_r"], rect["center_c"]))
    return bg, rectangles, pixels


def _draw_rectangle_border(grid: np.ndarray, rect: dict) -> None:
    for r, c in rect["coords"]:
        grid[r, c] = rect["color"]


def main(grid: np.ndarray) -> np.ndarray:
    """Build full-canvas cross rails from hollow rectangles, then slide same-color lone pixels toward the matching rectangle.

    Input structure:
    - The grid has a black background with 2-3 colored hollow rectangles and several isolated single pixels.
    - Each rectangle is a 1-cell-thick monochrome outline whose center row and center column are unique among rectangles.
    - Single pixels may share a rectangle color or use an unmatched distractor color.

    Rule:
    - Detect every hollow rectangle and draw a grey full-width row plus a grey full-height column through its center.
    - Redraw the original rectangle outlines over those rails so the rectangles occlude the rail cells they occupy.
    - For each isolated pixel, keep it only if its color matches a rectangle and it lies exactly on that rectangle's center row or center column; slide it along that rail toward the rectangle and stop one cell outside the border, except a pixel already at the rail intersection stays put.

    Key insight:
    - A loose pixel survives only when two cues agree at once: it must match a rectangle by color and also sit on that rectangle's center cross.
    """
    g = np.asarray(grid, dtype=int)
    bg, rectangles, pixels = _extract_scene(g)
    out = np.full_like(g, bg)

    color_to_rect = {}
    for rect in rectangles:
        if rect["color"] in color_to_rect:
            raise ValueError("rectangle colors must be unique")
        color_to_rect[rect["color"]] = rect

    for rect in rectangles:
        out[rect["center_r"], :] = TRACK_COLOR
        out[:, rect["center_c"]] = TRACK_COLOR

    for rect in rectangles:
        _draw_rectangle_border(out, rect)

    for pixel in pixels:
        rect = color_to_rect.get(pixel["color"])
        if rect is None:
            continue

        r = pixel["r"]
        c = pixel["c"]
        on_row = r == rect["center_r"]
        on_col = c == rect["center_c"]

        if on_row and on_col:
            out[r, c] = pixel["color"]
            continue
        if on_row:
            if c < rect["bbox"][1]:
                out[r, rect["bbox"][1] - 1] = pixel["color"]
            elif c > rect["bbox"][3]:
                out[r, rect["bbox"][3] + 1] = pixel["color"]
            continue
        if on_col:
            if r < rect["bbox"][0]:
                out[rect["bbox"][0] - 1, c] = pixel["color"]
            elif r > rect["bbox"][2]:
                out[rect["bbox"][2] + 1, c] = pixel["color"]

    return out


def _bbox_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    return a[2] + gap < b[0] or b[2] + gap < a[0] or a[3] + gap < b[1] or b[3] + gap < a[1]


def _draw_input_rectangle(grid: np.ndarray, rect: dict) -> None:
    r0, c0, r1, c1 = rect["bbox"]
    color = rect["color"]
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _empty_8_neighborhood(grid: np.ndarray, r: int, c: int) -> bool:
    h, w = grid.shape
    if grid[r, c] != BG_COLOR:
        return False
    for nr, nc in _neighbors8(r, c):
        if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] != BG_COLOR:
            return False
    return True


def _inside_any_bbox(r: int, c: int, rectangles: list[dict]) -> bool:
    for rect in rectangles:
        r0, c0, r1, c1 = rect["bbox"]
        if r0 <= r <= r1 and c0 <= c <= c1:
            return True
    return False


def _ontrack_candidates(rect: dict, shape: tuple[int, int], kind: str) -> list[tuple[int, int]]:
    h, w = shape
    r0, c0, r1, c1 = rect["bbox"]
    cr = rect["center_r"]
    cc = rect["center_c"]
    candidates: list[tuple[int, int]] = []

    if kind == "h":
        candidates.extend((cr, c) for c in range(0, c0 - 1))
        candidates.extend((cr, c) for c in range(c1 + 2, w))
    elif kind == "v":
        candidates.extend((r, cc) for r in range(0, r0 - 1))
        candidates.extend((r, cc) for r in range(r1 + 2, h))
    elif kind == "x":
        if rect["height"] >= 5 and rect["width"] >= 5:
            candidates.append((cr, cc))
    random.shuffle(candidates)
    return candidates


def _place_ontrack_pixel(grid: np.ndarray, rectangles: list[dict], rect: dict, kind: str) -> bool:
    for r, c in _ontrack_candidates(rect, grid.shape, kind):
        if kind != "x" and _inside_any_bbox(r, c, rectangles):
            continue
        if _empty_8_neighborhood(grid, r, c):
            grid[r, c] = rect["color"]
            return True
    return False


def _place_offtrack_pixel(grid: np.ndarray, rectangles: list[dict], color: int, forbidden_rect: dict | None) -> bool:
    h, w = grid.shape
    cells = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(cells)

    for r, c in cells:
        if forbidden_rect is not None and (r == forbidden_rect["center_r"] or c == forbidden_rect["center_c"]):
            continue
        if _inside_any_bbox(r, c, rectangles):
            continue
        if _empty_8_neighborhood(grid, r, c):
            grid[r, c] = color
            return True
    return False


def _build_rectangles(shape: tuple[int, int], count: int) -> list[dict] | None:
    h, w = shape
    colors = random.sample(list(RECTANGLE_SOURCE_COLORS), count)
    dims = [random.choice([(3, 3), (3, 5), (5, 3), (5, 5)]) for _ in range(count)]
    if all(dim != (5, 5) for dim in dims) and random.random() < 0.6:
        dims[random.randrange(count)] = (5, 5)

    rectangles: list[dict] = []
    for color, (rh, rw) in zip(colors, dims):
        placed = False
        for _ in range(200):
            r0 = random.randint(2, h - rh - 3)
            c0 = random.randint(2, w - rw - 3)
            bbox = (r0, c0, r0 + rh - 1, c0 + rw - 1)
            center_r = (bbox[0] + bbox[2]) // 2
            center_c = (bbox[1] + bbox[3]) // 2

            if any(center_r == rect["center_r"] or center_c == rect["center_c"] for rect in rectangles):
                continue
            if any(not _bbox_separated(bbox, rect["bbox"], gap=1) for rect in rectangles):
                continue

            rectangles.append(
                {
                    "color": color,
                    "bbox": bbox,
                    "center_r": center_r,
                    "center_c": center_c,
                    "height": rh,
                    "width": rw,
                }
            )
            placed = True
            break
        if not placed:
            return None

    return rectangles


def _validate_generated_input(grid: np.ndarray) -> bool:
    try:
        _ = main(grid)
        bg, rectangles, pixels = _extract_scene(grid)
    except ValueError:
        return False

    if bg != BG_COLOR or not (2 <= len(rectangles) <= 3) or not (4 <= len(pixels) <= 8):
        return False

    centers_r = {rect["center_r"] for rect in rectangles}
    centers_c = {rect["center_c"] for rect in rectangles}
    if len(centers_r) != len(rectangles) or len(centers_c) != len(rectangles):
        return False

    color_to_rect = {rect["color"]: rect for rect in rectangles}
    if len(color_to_rect) != len(rectangles):
        return False

    for rect in rectangles:
        if not any(
            pixel["color"] == rect["color"]
            and (pixel["r"] == rect["center_r"] or pixel["c"] == rect["center_c"])
            for pixel in pixels
        ):
            return False

    if not any(
        pixel["color"] not in color_to_rect
        or (
            pixel["r"] != color_to_rect[pixel["color"]]["center_r"]
            and pixel["c"] != color_to_rect[pixel["color"]]["center_c"]
        )
        for pixel in pixels
    ):
        return False

    if not any(
        pixel["color"] in color_to_rect
        and (
            (pixel["r"] == color_to_rect[pixel["color"]]["center_r"]) ^ (pixel["c"] == color_to_rect[pixel["color"]]["center_c"])
        )
        for pixel in pixels
    ):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        h = random.randint(14, 18)
        w = random.randint(14, 18)
        rect_count = random.choice([2, 3])
        rectangles = _build_rectangles((h, w), rect_count)
        if rectangles is None:
            continue

        grid = np.zeros((h, w), dtype=int)
        for rect in rectangles:
            _draw_input_rectangle(grid, rect)

        mandatory_kinds = ["h", "v"][:rect_count]
        while len(mandatory_kinds) < rect_count:
            mandatory_kinds.append(random.choice(["h", "v"]))

        square5_indices = [i for i, rect in enumerate(rectangles) if rect["height"] >= 5 and rect["width"] >= 5]
        if square5_indices and random.random() < 0.6:
            mandatory_kinds[random.choice(square5_indices)] = "x"

        success = True
        for rect, kind in zip(rectangles, mandatory_kinds):
            if not _place_ontrack_pixel(grid, rectangles, rect, kind):
                success = False
                break
        if not success:
            continue

        total_pixels = random.randint(max(4, rect_count + 1), 8)
        pixels_to_add = total_pixels - rect_count

        distractor_choices = ["unmatched", "offtrack"]
        planned_extras = []
        if pixels_to_add > 0:
            planned_extras.append(random.choice(distractor_choices))
        while len(planned_extras) < pixels_to_add:
            planned_extras.append(random.choices(["ontrack", "offtrack", "unmatched"], weights=[4, 3, 2])[0])

        random.shuffle(planned_extras)

        for extra in planned_extras:
            if extra == "ontrack":
                rect = random.choice(rectangles)
                kind = random.choice(["h", "v", "x"] if rect["height"] >= 5 and rect["width"] >= 5 else ["h", "v"])
                if not _place_ontrack_pixel(grid, rectangles, rect, kind):
                    success = False
                    break
            elif extra == "offtrack":
                rect = random.choice(rectangles)
                if not _place_offtrack_pixel(grid, rectangles, rect["color"], rect):
                    success = False
                    break
            else:
                if not _place_offtrack_pixel(grid, rectangles, 4, None):
                    success = False
                    break

        if success and _validate_generated_input(grid):
            return grid

    raise RuntimeError("failed to generate a valid input for puzzle 5")
