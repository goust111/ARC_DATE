from __future__ import annotations

import random

import numpy as np


_BG = 0
_FRAME_INSET = 2
_GEN_ATTEMPTS = 400
_DIR_VECTORS: dict[str, tuple[int, int]] = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
_DIR_TO_ARROW_CELLS: dict[str, frozenset[tuple[int, int]]] = {
    "up": frozenset({(0, 0), (1, 0), (1, 1)}),
    "right": frozenset({(0, 0), (0, 1), (1, 0)}),
    "down": frozenset({(0, 0), (0, 1), (1, 1)}),
    "left": frozenset({(0, 1), (1, 0), (1, 1)}),
}
_ARROW_CELLS_TO_DIR = {cells: name for name, cells in _DIR_TO_ARROW_CELLS.items()}


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _find_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            pts = np.array(coords, dtype=int)
            components.append({"color": color, "coords": pts, "bbox": _bbox(pts)})
    return components


def _frame_from_component(grid: np.ndarray, comp: dict[str, object]) -> dict[str, int] | None:
    color = int(comp["color"])
    coords = np.asarray(comp["coords"], dtype=int)
    r0, c0, r1, c1 = comp["bbox"]
    side_h = int(r1 - r0 + 1)
    side_w = int(c1 - c0 + 1)
    if side_h != side_w or side_h < 3:
        return None
    border_cells = 4 * side_h - 4
    if len(coords) != border_cells:
        return None
    border = np.zeros((side_h, side_w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    if not np.array_equal(sub == color, border):
        return None
    return {"color": color, "r0": int(r0), "c0": int(c0), "r1": int(r1), "c1": int(c1), "side": side_h}


def _arrow_from_component(comp: dict[str, object]) -> tuple[int, tuple[int, int]] | None:
    color = int(comp["color"])
    coords = np.asarray(comp["coords"], dtype=int)
    r0, c0, r1, c1 = comp["bbox"]
    if int(r1 - r0 + 1) != 2 or int(c1 - c0 + 1) != 2 or len(coords) != 3:
        return None
    local = frozenset((int(r) - int(r0), int(c) - int(c0)) for r, c in coords)
    direction = _ARROW_CELLS_TO_DIR.get(local)
    if direction is None:
        return None
    return color, _DIR_VECTORS[direction]


def _paint_frame(canvas: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    canvas[r0, c0 : c1 + 1] = color
    canvas[r1, c0 : c1 + 1] = color
    canvas[r0 : r1 + 1, c0] = color
    canvas[r0 : r1 + 1, c1] = color


def _parse_scene(grid: np.ndarray) -> tuple[int, list[dict[str, int]], dict[int, tuple[int, int]]]:
    bg = _background_color(grid)
    frames: list[dict[str, int]] = []
    arrows: dict[int, tuple[int, int]] = {}

    for comp in _find_components(grid, bg):
        frame = _frame_from_component(grid, comp)
        if frame is not None:
            frames.append(frame)
            continue
        arrow = _arrow_from_component(comp)
        if arrow is not None:
            color, vec = arrow
            assert color not in arrows, "legend color must be unique"
            arrows[color] = vec
            continue
        raise AssertionError("unexpected foreground object")

    assert 3 <= len(frames) <= 5, "expected 3-5 nested square frames"
    frame_colors = {frame["color"] for frame in frames}
    assert frame_colors == set(arrows), "each frame color must map to exactly one arrow"

    centers = {(frame["r0"] + frame["r1"], frame["c0"] + frame["c1"]) for frame in frames}
    assert len(centers) == 1, "all frames must share one center"

    frames.sort(key=lambda frame: int(frame["side"]), reverse=True)
    sides = [int(frame["side"]) for frame in frames]
    assert all(sides[i] > sides[i + 1] for i in range(len(sides) - 1)), "frame sizes must be distinct"
    return bg, frames, arrows


def _render_shifted_frames(
    frames: list[dict[str, int]],
    arrows: dict[int, tuple[int, int]],
    bg: int,
) -> np.ndarray:
    shifted: list[tuple[int, int, int, int, int]] = []
    cum_r = 0
    cum_c = 0
    for frame in frames:
        dr, dc = arrows[int(frame["color"])]
        cum_r += int(dr)
        cum_c += int(dc)
        shifted.append(
            (
                int(frame["r0"]) + cum_r,
                int(frame["c0"]) + cum_c,
                int(frame["r1"]) + cum_r,
                int(frame["c1"]) + cum_c,
                int(frame["color"]),
            )
        )

    min_r = min(item[0] for item in shifted)
    min_c = min(item[1] for item in shifted)
    max_r = max(item[2] for item in shifted)
    max_c = max(item[3] for item in shifted)
    out = np.full((max_r - min_r + 1, max_c - min_c + 1), int(bg), dtype=int)

    for r0, c0, r1, c1, color in shifted:
        _paint_frame(out, (r0 - min_r, c0 - min_c, r1 - min_r, c1 - min_c), color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shift each nested square frame by the direction encoded by its same-color legend arrow, accumulating outer moves into inner ones.

    Input structure:
    - Black background with one group of 3-5 nested hollow square frames on the left; every frame is monochromatic and all share one center.
    - The right side contains one disconnected 3-pixel L-shaped arrow per frame color; the four L orientations encode up, right, down, or left.
    - Non-background objects are only those frames and legend arrows, and each frame color appears exactly once in the legend.

    Rule:
    - Match every frame to the arrow of the same color.
    - Order the frames from the largest square to the smallest square.
    - Move the outermost frame by one cell in its arrow direction; for each next inner frame, add its own one-cell move to the full cumulative displacement already applied to all enclosing frames.
    - Redraw only the shifted frames on a black background in the same outer-to-inner order, then crop to the tight bounding box of the final pattern.

    Key insight:
    - The legend directions are inherited inward: each layer contributes one step to a cumulative path, so deeper frames move by the sum of all outer directions plus their own.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg, frames, arrows = _parse_scene(g)
    return _render_shifted_frames(frames, arrows, bg)


def _choose_frame_sides() -> list[int]:
    layer_count = random.randint(3, 5)
    inner_options = [side for side in (3, 5, 7, 9, 11) if side + 4 * (layer_count - 1) <= 19]
    inner_side = random.choice(inner_options)
    outer_side = inner_side + 4 * (layer_count - 1)
    return [outer_side - 4 * i for i in range(layer_count)]


def _choose_directions(count: int) -> list[str]:
    names = list(_DIR_VECTORS)
    while True:
        directions = [random.choice(names) for _ in range(count)]
        if len(set(directions)) >= 2:
            return directions


def _legend_slots(count: int, column_count: int) -> list[tuple[int, int]]:
    row_count = (count + column_count - 1) // column_count
    slots: list[tuple[int, int]] = []
    for idx in range(count):
        rr = idx // column_count
        cc = idx % column_count
        slots.append((rr * 4, cc * 4))
    assert len(slots) == count and row_count >= 1
    return slots


def _paint_arrow(grid: np.ndarray, r0: int, c0: int, color: int, direction: str) -> None:
    for dr, dc in _DIR_TO_ARROW_CELLS[direction]:
        grid[r0 + dr, c0 + dc] = int(color)


def _build_input() -> tuple[np.ndarray, list[dict[str, int]], dict[int, tuple[int, int]]] | None:
    sides = _choose_frame_sides()
    layer_count = len(sides)
    colors = random.sample(list(range(1, 10)), layer_count)
    directions = _choose_directions(layer_count)
    arrow_map = {int(color): _DIR_VECTORS[direction] for color, direction in zip(colors, directions)}

    legend_cols = random.choice([1, 2])
    legend_slots = _legend_slots(layer_count, legend_cols)
    legend_height = max(r for r, _ in legend_slots) + 2
    legend_width = max(c for _, c in legend_slots) + 2
    outer_side = sides[0]
    mid_gap = random.randint(2, 3)

    min_h = max(outer_side, legend_height) + 2
    min_w = outer_side + mid_gap + legend_width + 2
    if min_h > 30 or min_w > 30:
        return None

    height = random.randint(min_h, 30)
    width = random.randint(min_w, 30)

    frame_c_max = width - outer_side - mid_gap - legend_width - 1
    if frame_c_max < 1:
        return None
    frame_c0 = random.randint(1, frame_c_max)
    legend_c_min = frame_c0 + outer_side + mid_gap
    legend_c_max = width - legend_width - 1
    if legend_c_min > legend_c_max:
        return None
    legend_c0 = random.randint(legend_c_min, legend_c_max)

    frame_center_c = frame_c0 + (outer_side - 1) / 2
    legend_center_c = legend_c0 + (legend_width - 1) / 2
    grid_center_c = (width - 1) / 2
    if not (frame_center_c < grid_center_c < legend_center_c):
        return None

    frame_r0 = random.randint(1, height - outer_side - 1)
    legend_r0 = random.randint(1, height - legend_height - 1)

    grid = np.full((height, width), _BG, dtype=int)
    frames: list[dict[str, int]] = []
    for idx, (side, color) in enumerate(zip(sides, colors)):
        inset = _FRAME_INSET * idx
        bbox = (
            frame_r0 + inset,
            frame_c0 + inset,
            frame_r0 + inset + side - 1,
            frame_c0 + inset + side - 1,
        )
        _paint_frame(grid, bbox, int(color))
        frames.append(
            {
                "color": int(color),
                "r0": int(bbox[0]),
                "c0": int(bbox[1]),
                "r1": int(bbox[2]),
                "c1": int(bbox[3]),
                "side": int(side),
            }
        )

    slot_order = legend_slots[:]
    random.shuffle(slot_order)
    for (slot_r, slot_c), color, direction in zip(slot_order, colors, directions):
        _paint_arrow(grid, legend_r0 + slot_r, legend_c0 + slot_c, int(color), direction)

    return grid, frames, arrow_map


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        built = _build_input()
        if built is None:
            continue
        grid, frames, arrow_map = built
        expected = _render_shifted_frames(frames, arrow_map, _BG)
        try:
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected) and grid.ndim == 2 and max(grid.shape) <= 30 and max(expected.shape) <= 30:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
