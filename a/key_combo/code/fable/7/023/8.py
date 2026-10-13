from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np


def _load_arc_utils():
    arc_utils_path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    spec = importlib.util.spec_from_file_location("arc_utils_shared", arc_utils_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load arc_utils from {arc_utils_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ARC = _load_arc_utils()
Color = _ARC.Color
find_components = _ARC.find_components
get_bg = _ARC.get_bg
in_bounds = _ARC.in_bounds


BACKGROUND = Color.BLACK
CORNERS = ("TL", "TR", "BL", "BR")
STEP_CHOICES = (1, 3, 5, 7)
GRID_MAX_SIDE = 30
MAX_GENERATION_ATTEMPTS = 500
PLACEMENT_ATTEMPTS = 200

CORNER_SIGNS = {
    "TL": (-1, -1),
    "TR": (-1, 1),
    "BL": (1, -1),
    "BR": (1, 1),
}


def _frame_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _corner_distances(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> dict[str, int]:
    r0, c0, r1, c1 = bbox
    h, w = shape
    return {
        "TL": int(r0 + c0),
        "TR": int(r0 + (w - 1 - c1)),
        "BL": int((h - 1 - r1) + c0),
        "BR": int((h - 1 - r1) + (w - 1 - c1)),
    }


def _nearest_corner(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> str:
    distances = _corner_distances(bbox, shape)
    nearest = min(distances.values())
    winners = [corner for corner, dist in distances.items() if dist == nearest]
    assert len(winners) == 1, "expected a unique nearest corner"
    return winners[0]


def _start_corner(bbox: tuple[int, int, int, int], corner: str) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if corner == "TL":
        return r0, c0
    if corner == "TR":
        return r0, c1
    if corner == "BL":
        return r1, c0
    return r1, c1


def _trace_stair(
    shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
    step: int,
    corner: str,
) -> list[tuple[int, int]]:
    start_r, start_c = _start_corner(bbox, corner)
    vr, hc = CORNER_SIGNS[corner]
    deltas = [(0, hc), (vr, 0)]
    cells: list[tuple[int, int]] = []
    cur_r, cur_c = start_r, start_c
    segment_index = 0

    while True:
        dr, dc = deltas[segment_index % 2]
        moved = False
        for _ in range(step):
            nr = cur_r + dr
            nc = cur_c + dc
            if not in_bounds(shape, nr, nc):
                return cells
            cells.append((nr, nc))
            cur_r, cur_c = nr, nc
            moved = True
        if not moved:
            return cells
        segment_index += 1


def _extract_specs(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    specs = []
    covered = set()

    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        frame_color = int(comp["color"])
        r0, c0, r1, c1 = map(int, comp["bbox"])
        side = r1 - r0 + 1
        if side != c1 - c0 + 1 or side < 3 or side % 2 == 0:
            continue
        if int(comp["area"]) != 4 * side - 4:
            continue

        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        ring_mask = np.zeros((side, side), dtype=bool)
        ring_mask[0, :] = True
        ring_mask[-1, :] = True
        ring_mask[:, 0] = True
        ring_mask[:, -1] = True
        if not np.all(patch[ring_mask] == frame_color):
            continue
        if np.any(patch[~ring_mask] == frame_color):
            continue

        interior = patch[1:-1, 1:-1]
        if int(np.count_nonzero(interior != bg)) != 1:
            continue
        center = (r0 + side // 2, c0 + side // 2)
        if int(g[center]) == bg or int(g[center]) == frame_color:
            continue

        bbox = (r0, c0, r1, c1)
        spec = {
            "bbox": bbox,
            "step": side - 2,
            "frame_color": frame_color,
            "anchor_pos": center,
            "anchor_color": int(g[center]),
            "corner": _nearest_corner(bbox, g.shape),
        }
        spec["stair_cells"] = _trace_stair(g.shape, bbox, spec["step"], spec["corner"])
        specs.append(spec)
        covered.update(_frame_cells(bbox))
        covered.add(center)

    assert specs, "expected at least one hollow square frame"
    assert len({spec["step"] for spec in specs}) == len(specs), "frame interior sizes must be distinct"
    assert len({spec["corner"] for spec in specs}) == len(specs), "nearest corners must be distinct"

    non_bg_cells = {tuple(map(int, coord)) for coord in np.argwhere(g != bg)}
    assert non_bg_cells == covered, "all foreground cells must belong to frames or anchors"
    return sorted(specs, key=lambda spec: (spec["bbox"][0], spec["bbox"][1]))


def _render_from_specs(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    bg = int(get_bg(out))
    for spec in specs:
        for r, c in spec["stair_cells"]:
            if int(out[r, c]) == bg:
                out[r, c] = spec["anchor_color"]
    return out


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Grow an anchor-colored staircase from each hollow square frame toward its unique nearest grid corner.

    Input structure:
    - The grid has a black background plus one or more non-touching monochrome hollow square frames.
    - Each frame is one pixel thick, has a unique odd interior size, and contains exactly one colored anchor pixel at its center.
    - Every frame has a unique nearest grid corner under Manhattan distance.

    Rule:
    - Keep every frame and anchor pixel unchanged.
    - For each frame, let the step length be its interior side length and choose the frame corner that faces the nearest grid corner.
    - Starting just beyond that outer frame corner, draw the anchor color in a staircase: move horizontally by the step length, then vertically by the step length, alternating until the next step would leave the grid.

    Key insight:
    - The three staircase parameters are split across different structures: the frame size gives the step length, the frame location gives the direction, and the center anchor gives the color.
    """
    specs = _extract_specs(np.asarray(input_grid, dtype=int))
    return _render_from_specs(input_grid, specs)


def _corner_ranges(shape: tuple[int, int], outer: int, step: int, corner: str) -> tuple[tuple[int, int], tuple[int, int]] | None:
    h, w = shape
    near_r_max = (h - outer - 1) // 2
    near_c_max = (w - outer - 1) // 2
    far_r_min = (h - outer) // 2 + 1
    far_c_min = (w - outer) // 2 + 1

    if corner == "TL":
        row_range = (step, near_r_max)
        col_range = (step, near_c_max)
    elif corner == "TR":
        row_range = (step, near_r_max)
        col_range = (far_c_min, w - outer - step)
    elif corner == "BL":
        row_range = (far_r_min, h - outer - step)
        col_range = (step, near_c_max)
    else:
        row_range = (far_r_min, h - outer - step)
        col_range = (far_c_min, w - outer - step)

    if row_range[0] > row_range[1] or col_range[0] > col_range[1]:
        return None
    return row_range, col_range


def _expanded_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + margin < br0
        or br1 + margin < ar0
        or ac1 + margin < bc0
        or bc1 + margin < ac0
    )


def _spec_from_latent(
    shape: tuple[int, int],
    corner: str,
    step: int,
    r0: int,
    c0: int,
    frame_color: int,
    anchor_color: int,
) -> dict:
    outer = step + 2
    bbox = (r0, c0, r0 + outer - 1, c0 + outer - 1)
    spec = {
        "bbox": bbox,
        "corner": corner,
        "step": step,
        "frame_color": frame_color,
        "anchor_pos": (r0 + outer // 2, c0 + outer // 2),
        "anchor_color": anchor_color,
    }
    spec["stair_cells"] = _trace_stair(shape, bbox, step, corner)
    return spec


def _place_specs(shape: tuple[int, int], placements: list[tuple[str, int, int, int]]) -> list[dict] | None:
    specs: list[dict] = []
    occupied: set[tuple[int, int]] = set()
    bboxes: list[tuple[int, int, int, int]] = []

    for corner, step, frame_color, anchor_color in placements:
        outer = step + 2
        ranges = _corner_ranges(shape, outer, step, corner)
        if ranges is None:
            return None
        row_range, col_range = ranges

        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = random.randint(*row_range)
            c0 = random.randint(*col_range)
            spec = _spec_from_latent(shape, corner, step, r0, c0, frame_color, anchor_color)
            if len(spec["stair_cells"]) < 2 * step:
                continue
            if any(_expanded_overlap(spec["bbox"], other_bbox, margin=1) for other_bbox in bboxes):
                continue

            cells = _frame_cells(spec["bbox"])
            cells.add(spec["anchor_pos"])
            cells.update(spec["stair_cells"])
            if cells & occupied:
                continue

            specs.append(spec)
            occupied.update(cells)
            bboxes.append(spec["bbox"])
            break
        else:
            return None

    return specs


def _draw_input(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    grid = np.full(shape, BACKGROUND, dtype=int)
    for spec in specs:
        for r, c in _frame_cells(spec["bbox"]):
            grid[r, c] = spec["frame_color"]
        ar, ac = spec["anchor_pos"]
        grid[ar, ac] = spec["anchor_color"]
    return grid


def _validate_semantics(grid: np.ndarray, latent_specs: list[dict]) -> bool:
    parsed_specs = _extract_specs(grid)
    latent_by_anchor = {spec["anchor_color"]: spec for spec in latent_specs}
    if {spec["anchor_color"] for spec in parsed_specs} != set(latent_by_anchor):
        return False

    for parsed in parsed_specs:
        latent = latent_by_anchor[parsed["anchor_color"]]
        for key in ("bbox", "step", "frame_color", "anchor_pos", "corner"):
            if parsed[key] != latent[key]:
                return False
        if parsed["stair_cells"] != latent["stair_cells"]:
            return False
        if len(parsed["stair_cells"]) < 2 * parsed["step"]:
            return False

        alt_corner_cells = [
            _trace_stair(grid.shape, parsed["bbox"], parsed["step"], corner)
            for corner in CORNERS
            if corner != parsed["corner"]
        ]
        if not all(cells != parsed["stair_cells"] for cells in alt_corner_cells):
            return False

        alt_steps = [step for step in STEP_CHOICES if step != parsed["step"]]
        if not any(_trace_stair(grid.shape, parsed["bbox"], step, parsed["corner"]) != parsed["stair_cells"] for step in alt_steps):
            return False

    expected = _render_from_specs(grid, latent_specs)
    actual = main(grid)
    if not np.array_equal(expected, actual):
        return False
    if np.array_equal(grid, actual):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    num_frames = random.choices([1, 2, 3, 4], weights=[1, 3, 4, 4], k=1)[0]
    steps = random.sample(STEP_CHOICES, k=num_frames)
    corners = random.sample(CORNERS, k=num_frames)
    max_step = max(steps)
    min_side = max(3 * max_step + 3, 12)
    shape = (
        random.randint(min_side, GRID_MAX_SIDE),
        random.randint(min_side, GRID_MAX_SIDE),
    )

    palette = random.sample(list(range(1, 10)), k=2 * num_frames)
    frame_colors = palette[:num_frames]
    anchor_colors = palette[num_frames:]

    placements = list(zip(corners, steps, frame_colors, anchor_colors))
    placements.sort(key=lambda item: item[1], reverse=True)
    specs = _place_specs(shape, placements)
    if specs is None:
        return None

    grid = _draw_input(shape, specs)
    return grid if _validate_semantics(grid, specs) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
