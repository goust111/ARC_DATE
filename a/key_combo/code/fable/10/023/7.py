from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
CTRL = 8
FRAME_COLORS = (1, 2, 3, 4, 5, 6, 7, 9)
QUADRANTS = ("nw", "ne", "sw", "se")

_RNG = random.Random()


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _paint_frame(grid: np.ndarray, top: int, left: int, height: int, width: int, color: int) -> None:
    grid[top, left : left + width] = color
    grid[top + height - 1, left : left + width] = color
    grid[top : top + height, left] = color
    grid[top : top + height, left + width - 1] = color


def _quadrant_of_control(frame: dict) -> str:
    ctrl_r, ctrl_c = frame["control"]
    inner_r = int(ctrl_r - frame["r0"] - 1)
    inner_c = int(ctrl_c - frame["c0"] - 1)
    inner_h = int(frame["height"] - 2)
    inner_w = int(frame["width"] - 2)
    assert 0 <= inner_r < inner_h and 0 <= inner_c < inner_w

    row_cmp = 2 * inner_r + 1 - inner_h
    col_cmp = 2 * inner_c + 1 - inner_w
    assert row_cmp != 0 and col_cmp != 0, "control pixel cannot lie on a quadrant divider"

    vertical = "n" if row_cmp < 0 else "s"
    horizontal = "w" if col_cmp < 0 else "e"
    return vertical + horizontal


def _frame_from_color(grid: np.ndarray, color: int) -> dict:
    coords = np.argwhere(grid == color)
    assert coords.size > 0
    r0, c0, r1, c1 = _bbox_coords(coords)
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    assert height >= 3 and width >= 3

    patch = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
    border = np.zeros((height, width), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    assert np.array_equal(patch == color, border), "each non-control color must be a hollow rectangle"

    inside_colors = set(map(int, np.unique(patch)))
    assert inside_colors <= {BG, CTRL, int(color)}

    ctrl_coords = np.argwhere(patch == CTRL)
    ctrl_count = int(ctrl_coords.shape[0])
    assert ctrl_count <= 1, "each frame may contain at most one control pixel"
    control = None
    if ctrl_count == 1:
        rel_r, rel_c = map(int, ctrl_coords[0])
        assert 0 < rel_r < height - 1 and 0 < rel_c < width - 1
        control = (r0 + rel_r, c0 + rel_c)

    return {
        "color": int(color),
        "r0": r0,
        "c0": c0,
        "r1": r1,
        "c1": c1,
        "height": height,
        "width": width,
        "area": int(height * width),
        "control": control,
    }


def _parse_frames(grid: np.ndarray) -> list[dict]:
    colors = [int(c) for c in np.unique(grid) if int(c) not in (BG, CTRL)]
    frames = [_frame_from_color(grid, color) for color in colors]
    assert len(frames) >= 2, "expected at least two frames"

    areas = [frame["area"] for frame in frames]
    dims = [(frame["height"], frame["width"]) for frame in frames]
    assert len(areas) == len(set(areas)), "frame areas must define a unique nesting order"
    assert len(dims) == len(set(dims)), "frame sizes must be unique"

    frames.sort(key=lambda frame: int(frame["area"]), reverse=True)
    for frame in frames[:-1]:
        assert frame["control"] is not None, "every non-smallest frame needs a control pixel"
        _quadrant_of_control(frame)
    assert frames[-1]["control"] is None, "the smallest frame must have no control pixel"

    total_ctrl = int(np.count_nonzero(grid == CTRL))
    assert total_ctrl == len(frames) - 1, "all control pixels must belong to non-smallest frames"
    return frames


def _render_nested(frames: list[dict], route: Iterable[str] | None = None) -> np.ndarray:
    route_list = list(route) if route is not None else [_quadrant_of_control(frame) for frame in frames[:-1]]
    assert len(route_list) == len(frames) - 1

    outer = frames[0]
    out = np.zeros((int(outer["height"]), int(outer["width"])), dtype=int)
    _paint_frame(out, 0, 0, int(outer["height"]), int(outer["width"]), int(outer["color"]))

    region_top, region_left = 1, 1
    region_h = int(outer["height"] - 2)
    region_w = int(outer["width"] - 2)
    for idx, child in enumerate(frames[1:]):
        child_h = int(child["height"])
        child_w = int(child["width"])
        assert child_h <= region_h and child_w <= region_w
        quad = route_list[idx]
        top = region_top if quad[0] == "n" else region_top + region_h - child_h
        left = region_left if quad[1] == "w" else region_left + region_w - child_w
        _paint_frame(out, int(top), int(left), child_h, child_w, int(child["color"]))
        region_top = int(top + 1)
        region_left = int(left + 1)
        region_h = int(child_h - 2)
        region_w = int(child_w - 2)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the scattered hollow frames into an area-descending nested stack guided by control quadrants.

    Input structure:
    - The grid has black background, several pairwise non-touching hollow rectangular frames of distinct colors and distinct sizes, and teal control pixels.
    - Every frame except the smallest contains exactly one teal control pixel strictly inside its border.
    - The frames are scattered in arbitrary positions; the control pixel inside a frame lies in one of that frame's four interior quadrants.

    Rule:
    - Ignore the scattering and sort the frames by bounding-box area from largest to smallest.
    - Make the output the size of the largest frame and draw that frame first.
    - For each next smaller frame, place it inside the previous frame so it touches the inner corner selected by the previous frame's control-pixel quadrant; do not draw the control pixels themselves.

    Key insight:
    - The control pixels are navigational markers, not output content: each one tells you which inner corner receives the next frame in the descending size chain.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    frames = _parse_frames(g)
    return _render_nested(frames)


def _quadrant_cells(height: int, width: int, quad: str) -> list[tuple[int, int]]:
    inner_h = int(height - 2)
    inner_w = int(width - 2)
    cells = []
    for ir in range(inner_h):
        row_cmp = 2 * ir + 1 - inner_h
        if row_cmp == 0:
            continue
        vertical = "n" if row_cmp < 0 else "s"
        for ic in range(inner_w):
            col_cmp = 2 * ic + 1 - inner_w
            if col_cmp == 0:
                continue
            horizontal = "w" if col_cmp < 0 else "e"
            if vertical + horizontal == quad:
                cells.append((ir + 1, ic + 1))
    return cells


def _build_chain() -> list[dict] | None:
    count = _RNG.choice((3, 4, 5))
    inner_h = _RNG.randint(3, 6)
    inner_w = _RNG.randint(3, 6)
    dims = [(inner_h, inner_w)]
    used_areas = {int(inner_h * inner_w)}

    cur_h, cur_w = inner_h, inner_w
    outer_limit = 20
    for step in range(count - 1):
        remaining_outer = count - 2 - step
        h_lo = cur_h + 3
        h_hi = min(cur_h + 6, outer_limit - 3 * remaining_outer)
        w_lo = cur_w + 3
        w_hi = min(cur_w + 6, outer_limit - 3 * remaining_outer)
        candidates = [
            (h, w)
            for h in range(h_lo, h_hi + 1)
            for w in range(w_lo, w_hi + 1)
            if h * w not in used_areas
        ]
        if not candidates:
            return None
        cur_h, cur_w = _RNG.choice(candidates)
        dims.append((cur_h, cur_w))
        used_areas.add(int(cur_h * cur_w))

    dims.reverse()
    colors = _RNG.sample(FRAME_COLORS, len(dims))
    frames = []
    for (height, width), color in zip(dims, colors):
        frames.append(
            {
                "height": int(height),
                "width": int(width),
                "area": int(height * width),
                "color": int(color),
                "control_quadrant": None,
            }
        )
    return frames


def _sample_route(length: int) -> list[str] | None:
    for _ in range(200):
        route = [_RNG.choice(QUADRANTS) for _ in range(length)]
        verticals = {quad[0] for quad in route}
        horizontals = {quad[1] for quad in route}
        if len(set(route)) >= 2 and len(verticals) >= 2 and len(horizontals) >= 2:
            return route
    return None


def _attach_controls(frames: list[dict], route: list[str]) -> bool:
    for frame, quad in zip(frames[:-1], route):
        cells = _quadrant_cells(int(frame["height"]), int(frame["width"]), quad)
        if not cells:
            return False
        rel_r, rel_c = _RNG.choice(cells)
        frame["control_quadrant"] = quad
        frame["control_rel"] = (int(rel_r), int(rel_c))
    frames[-1]["control_quadrant"] = None
    frames[-1]["control_rel"] = None
    return True


def _frame_patch(frame: dict) -> np.ndarray:
    patch = np.zeros((int(frame["height"]), int(frame["width"])), dtype=int)
    _paint_frame(patch, 0, 0, int(frame["height"]), int(frame["width"]), int(frame["color"]))
    if frame.get("control_rel") is not None:
        ctrl_r, ctrl_c = frame["control_rel"]
        patch[int(ctrl_r), int(ctrl_c)] = CTRL
    return patch


def _render_expected_from_spec(frames: list[dict]) -> np.ndarray:
    shadow = []
    for frame in frames:
        shadow.append(
            {
                "height": int(frame["height"]),
                "width": int(frame["width"]),
                "area": int(frame["area"]),
                "color": int(frame["color"]),
                "control": None if frame.get("control_rel") is None else (int(frame["control_rel"][0]), int(frame["control_rel"][1])),
                "r0": 0,
                "c0": 0,
            }
        )
    route = [str(frame["control_quadrant"]) for frame in frames[:-1]]
    return _render_nested(shadow, route)


def _place_patches(patches: list[np.ndarray], canvas_h: int, canvas_w: int) -> list[tuple[int, int]] | None:
    blocked = np.zeros((canvas_h, canvas_w), dtype=bool)
    placements: list[tuple[int, int]] = []
    for patch in patches:
        height, width = map(int, patch.shape)
        candidates = []
        for top in range(canvas_h - height + 1):
            for left in range(canvas_w - width + 1):
                r0 = max(0, top - 1)
                c0 = max(0, left - 1)
                r1 = min(canvas_h, top + height + 1)
                c1 = min(canvas_w, left + width + 1)
                if not blocked[r0:r1, c0:c1].any():
                    candidates.append((top, left))
        if not candidates:
            return None
        top, left = _RNG.choice(candidates)
        placements.append((int(top), int(left)))
        r0 = max(0, top - 1)
        c0 = max(0, left - 1)
        r1 = min(canvas_h, top + height + 1)
        c1 = min(canvas_w, left + width + 1)
        blocked[r0:r1, c0:c1] = True
    return placements


def _scatter_input(frames: list[dict]) -> tuple[np.ndarray, list[dict]] | None:
    patches = [_frame_patch(frame) for frame in frames]
    order = list(range(len(frames)))
    _RNG.shuffle(order)
    shuffled_patches = [patches[idx] for idx in order]
    shuffled_frames = [frames[idx] for idx in order]

    for _ in range(40):
        canvas_h = _RNG.randint(max(max(frame["height"] for frame in frames) + 4, 22), 30)
        canvas_w = _RNG.randint(max(max(frame["width"] for frame in frames) + 4, 22), 30)
        placements = _place_patches(shuffled_patches, canvas_h, canvas_w)
        if placements is None:
            continue

        grid = np.zeros((canvas_h, canvas_w), dtype=int)
        placed = []
        for frame, patch, (top, left) in zip(shuffled_frames, shuffled_patches, placements):
            height, width = map(int, patch.shape)
            mask = patch != BG
            grid[top : top + height, left : left + width][mask] = patch[mask]
            placed.append(
                {
                    "color": int(frame["color"]),
                    "r0": int(top),
                    "c0": int(left),
                    "r1": int(top + height - 1),
                    "c1": int(left + width - 1),
                    "area": int(frame["area"]),
                }
            )

        size_order = [int(frame["color"]) for frame in frames]
        spatial_order = [rec["color"] for rec in sorted(placed, key=lambda rec: (rec["r0"], rec["c0"]))]
        if spatial_order == size_order:
            continue
        return grid, placed
    return None


def _gen_attempt() -> np.ndarray | None:
    frames = _build_chain()
    if frames is None:
        return None

    route = _sample_route(len(frames) - 1)
    if route is None:
        return None
    if not _attach_controls(frames, route):
        return None

    expected = _render_expected_from_spec(frames)
    for fixed in QUADRANTS:
        if np.array_equal(expected, _render_expected_from_spec([{**frame, "control_quadrant": fixed if idx < len(frames) - 1 else None} for idx, frame in enumerate(frames)])):
            return None

    scattered = _scatter_input(frames)
    if scattered is None:
        return None
    grid, _placed = scattered

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-control-frame input")
