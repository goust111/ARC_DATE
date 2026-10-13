from __future__ import annotations

import random
from collections import Counter

import numpy as np


PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
BG = 0
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _mono_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append({"color": color, "coords": coords})
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _border_cells(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _parse_frame(grid: np.ndarray, comp: dict, bg: int) -> dict:
    coords = [(int(r), int(c)) for r, c in comp["coords"]]
    color = int(comp["color"])
    r0, c0, r1, c1 = _bbox(coords)
    assert r1 - r0 + 1 >= 3 and c1 - c0 + 1 >= 3, "frame bbox must be at least 3x3"

    coord_set = set(coords)
    border = _border_cells((r0, c0, r1, c1))
    missing = [(r, c) for r, c in border if (r, c) not in coord_set]
    assert len(missing) == 1, "frame must have exactly one border gap"
    gap = missing[0]
    assert gap not in {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}, "gap must lie on one side, not a corner"

    for r, c in coord_set:
        on_border = r in (r0, r1) or c in (c0, c1)
        assert on_border, "frame color must stay on the border"
        assert int(grid[r, c]) == color

    if gap[0] == r0:
        orientation = "top"
    elif gap[0] == r1:
        orientation = "bottom"
    elif gap[1] == c0:
        orientation = "left"
    else:
        orientation = "right"

    return {
        "color": color,
        "bbox": (r0, c0, r1, c1),
        "gap": gap,
        "orientation": orientation,
    }


def _extract_scene(grid: np.ndarray, bg: int) -> tuple[list[dict], list[tuple[int, int, int]]]:
    frames: list[dict] = []
    loose: list[tuple[int, int, int]] = []
    for comp in _mono_components(grid, bg):
        coords = comp["coords"]
        if len(coords) == 1:
            r, c = coords[0]
            loose.append((int(r), int(c), int(comp["color"])))
        else:
            frames.append(_parse_frame(grid, comp, bg))
    assert frames, "expected at least one frame"
    assert loose, "expected loose singleton pixels"
    return frames, loose


def _physics_mask(shape: tuple[int, int], frames: list[dict]) -> np.ndarray:
    occupied = np.zeros(shape, dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        for r, c in _border_cells((r0, c0, r1, c1)):
            occupied[r, c] = True
        gap_r, gap_c = frame["gap"]
        if frame["orientation"] == "top":
            occupied[gap_r, gap_c] = False
        else:
            occupied[gap_r, gap_c] = True
    return occupied


def _settle_loose_pixels(
    shape: tuple[int, int], frames: list[dict], loose: list[tuple[int, int, int]]
) -> list[dict]:
    occupied = _physics_mask(shape, frames)
    settled: list[dict] = []
    for start_r, start_c, color in sorted(loose, key=lambda item: (-item[0], item[1], item[2])):
        r = int(start_r)
        c = int(start_c)
        while r + 1 < shape[0] and not occupied[r + 1, c]:
            r += 1
        occupied[r, c] = True
        settled.append(
            {
                "color": int(color),
                "start": (int(start_r), int(start_c)),
                "end": (int(r), int(c)),
            }
        )
    return settled


def _inside_frame(pos: tuple[int, int], frame: dict) -> bool:
    r, c = pos
    r0, c0, r1, c1 = frame["bbox"]
    return r0 < r < r1 and c0 < c < c1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop loose pixels straight downward so only top-open hollow frames can swallow them into nested cavities.

    Input structure:
    - The background is black.
    - Foreground objects are monochrome hollow rectangular frames, each with exactly one one-cell gap on its border.
    - Additional foreground singletons are loose colored pixels suspended above the frames; frames may be nested.

    Rule:
    - Treat every loose singleton as a falling pixel and clear its source cell.
    - Pixels fall straight down and stop on the first obstacle among the grid floor, frame borders, or previously settled pixels.
    - A gap is traversable only when it lies on a frame's top border; left, right, and bottom gaps remain visually open but behave as solid blockers for the falling simulation.
    - Once a pixel enters a top-open frame, it keeps falling within that cavity and may pass through aligned top gaps of inner nested frames before stacking at the lowest reachable cell.

    Key insight:
    - The missing border cell does not matter by itself; its side determines whether the frame is an entrance or just another roof.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    frames, loose = _extract_scene(g, bg)

    out = g.copy()
    for r, c, _color in loose:
        out[r, c] = bg

    for item in _settle_loose_pixels(g.shape, frames, loose):
        r, c = item["end"]
        out[r, c] = int(item["color"])
    return out


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _sample_bbox(
    h: int,
    w: int,
    min_h: int,
    max_h: int,
    min_w: int,
    max_w: int,
    blocked: list[tuple[int, int, int, int]],
    top_margin: int,
    bottom_margin: int,
    side_margin: int,
    pad: int,
) -> tuple[int, int, int, int] | None:
    max_h = min(max_h, h - top_margin - bottom_margin)
    max_w = min(max_w, w - 2 * side_margin)
    if min_h > max_h or min_w > max_w:
        return None
    for _ in range(250):
        box_h = random.randint(min_h, max_h)
        box_w = random.randint(min_w, max_w)
        r0 = random.randint(top_margin, h - bottom_margin - box_h)
        c0 = random.randint(side_margin, w - side_margin - box_w)
        bbox = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
        if all(not _boxes_overlap(bbox, other, pad=pad) for other in blocked):
            return bbox
    return None


def _draw_frame(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: int,
    orientation: str,
    gap_index: int,
) -> dict:
    r0, c0, r1, c1 = bbox
    for r, c in _border_cells(bbox):
        grid[r, c] = color
    if orientation == "top":
        gap = (r0, gap_index)
    elif orientation == "bottom":
        gap = (r1, gap_index)
    elif orientation == "left":
        gap = (gap_index, c0)
    else:
        gap = (gap_index, c1)
    grid[gap] = BG
    return {"color": color, "bbox": bbox, "gap": gap, "orientation": orientation}


def _pick_color(forbidden: set[int]) -> int:
    options = [color for color in PALETTE if color not in forbidden]
    if not options:
        options = PALETTE[:]
    return random.choice(options)


def _place_loose_pixel(grid: np.ndarray, r: int, c: int) -> tuple[int, int, int] | None:
    if int(grid[r, c]) != BG:
        return None
    blocked_colors = {
        int(grid[nr, nc])
        for nr, nc in _neighbors4(r, c)
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and int(grid[nr, nc]) != BG
    }
    color = _pick_color(blocked_colors)
    grid[r, c] = color
    return (r, c, color)


def _sample_rows(limit: int, count: int) -> list[int] | None:
    if limit <= 0 or count <= 0 or limit < count:
        return None
    rows = sorted(random.sample(range(limit), count))
    return rows


def _find_free_column(width: int, blocked: list[tuple[int, int, int, int]]) -> int | None:
    candidates = []
    for c in range(width):
        if all(not (c0 <= c <= c1) for _r0, c0, _r1, c1 in blocked):
            candidates.append(c)
    if not candidates:
        return None
    return random.choice(candidates)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(20, 28)
    w = random.randint(22, 30)
    grid = np.zeros((h, w), dtype=int)

    frame_palette = PALETTE[:]
    random.shuffle(frame_palette)
    frame_color_idx = 0

    def next_frame_color() -> int:
        nonlocal frame_color_idx
        color = frame_palette[frame_color_idx % len(frame_palette)]
        frame_color_idx += 1
        return color

    blocked_boxes: list[tuple[int, int, int, int]] = []
    frames: list[dict] = []

    chain_depth = random.choices([1, 2, 3], weights=[1, 3, 2])[0]
    min_chain_side = 3 + 4 * (chain_depth - 1)
    chain_bbox = _sample_bbox(
        h,
        w,
        min_h=min_chain_side,
        max_h=min(min_chain_side + 6, 13),
        min_w=min_chain_side + 1,
        max_w=min(min_chain_side + 8, 15),
        blocked=blocked_boxes,
        top_margin=5,
        bottom_margin=2,
        side_margin=1,
        pad=2,
    )
    if chain_bbox is None:
        return None
    blocked_boxes.append(chain_bbox)

    cr0, cc0, cr1, cc1 = chain_bbox
    gap_lo = cc0 + 2 * (chain_depth - 1) + 1
    gap_hi = cc1 - 2 * (chain_depth - 1) - 1
    if gap_lo > gap_hi:
        return None
    chain_gap_c = random.randint(gap_lo, gap_hi)
    chain_levels: list[dict] = []
    for level in range(chain_depth):
        bbox = (cr0 + 2 * level, cc0 + 2 * level, cr1 - 2 * level, cc1 - 2 * level)
        chain_levels.append(_draw_frame(grid, bbox, next_frame_color(), "top", chain_gap_c))
    frames.extend(chain_levels)

    barrier_count = random.choice([1, 2])
    barrier_frames: list[dict] = []
    barrier_orientations = ["bottom"]
    if barrier_count == 2:
        barrier_orientations.append(random.choice(["bottom", "left", "right"]))

    for orientation in barrier_orientations:
        bbox = _sample_bbox(
            h,
            w,
            min_h=5,
            max_h=9,
            min_w=5,
            max_w=10,
            blocked=blocked_boxes,
            top_margin=4,
            bottom_margin=1,
            side_margin=1,
            pad=2,
        )
        if bbox is None:
            return None
        blocked_boxes.append(bbox)
        r0, c0, r1, c1 = bbox
        if orientation in {"top", "bottom"}:
            gap_index = random.randint(c0 + 1, c1 - 1)
        else:
            gap_index = random.randint(r0 + 1, r1 - 1)
        frame = _draw_frame(grid, bbox, next_frame_color(), orientation, gap_index)
        barrier_frames.append(frame)
        frames.append(frame)

    loose: list[tuple[int, int, int]] = []

    capture_rows = _sample_rows(chain_levels[0]["bbox"][0] - 1, random.randint(2, 3))
    if capture_rows is None:
        return None
    for row in capture_rows:
        pixel = _place_loose_pixel(grid, row, chain_gap_c)
        if pixel is None:
            return None
        loose.append(pixel)

    roof_cols = [c for c in range(chain_levels[0]["bbox"][1] + 1, chain_levels[0]["bbox"][3]) if c != chain_gap_c]
    if not roof_cols:
        return None
    roof_col = random.choice(roof_cols)
    roof_rows = _sample_rows(chain_levels[0]["bbox"][0] - 1, random.randint(1, 2))
    if roof_rows is None:
        return None
    for row in roof_rows:
        pixel = _place_loose_pixel(grid, row, roof_col)
        if pixel is None:
            return None
        loose.append(pixel)

    bottom_tests: list[tuple[dict, int]] = []
    for frame in barrier_frames:
        r0, c0, r1, c1 = frame["bbox"]
        if frame["orientation"] == "bottom":
            test_col = frame["gap"][1]
            rows = _sample_rows(r0 - 1, 1)
            if rows is None:
                return None
            pixel = _place_loose_pixel(grid, rows[0], test_col)
            if pixel is None:
                return None
            loose.append(pixel)
            bottom_tests.append((frame, test_col))
        elif random.random() < 0.5:
            test_col = random.randint(c0 + 1, c1 - 1)
            rows = _sample_rows(r0 - 1, 1)
            if rows is None:
                return None
            pixel = _place_loose_pixel(grid, rows[0], test_col)
            if pixel is None:
                return None
            loose.append(pixel)

    floor_col = _find_free_column(w, blocked_boxes)
    if floor_col is None:
        return None
    floor_rows = _sample_rows(h - 2, random.randint(1, 2))
    if floor_rows is None:
        return None
    for row in floor_rows:
        pixel = _place_loose_pixel(grid, row, floor_col)
        if pixel is None:
            return None
        loose.append(pixel)

    if len(loose) < 5:
        return None

    settled = _settle_loose_pixels(grid.shape, frames, loose)
    transformed = main(grid)
    if np.array_equal(transformed, grid):
        return None

    inner_chain = chain_levels[-1]
    if not any(item["start"][1] == chain_gap_c and _inside_frame(item["end"], inner_chain) for item in settled):
        return None
    if not any(item["start"][1] == roof_col and item["end"][0] < chain_levels[0]["bbox"][0] for item in settled):
        return None
    if not any(item["start"][1] == floor_col and item["end"][0] == h - 1 for item in settled):
        return None
    if not barrier_frames:
        return None
    for frame, test_col in bottom_tests:
        if not any(item["start"][1] == test_col and item["end"][0] < frame["bbox"][0] for item in settled):
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
