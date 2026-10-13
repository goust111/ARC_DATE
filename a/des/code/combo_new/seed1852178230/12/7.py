from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
FENCE = 7
OBJECT_COLORS = np.array([1, 2, 3, 4, 5, 6, 8, 9], dtype=int)
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

GEN_ATTEMPTS = 400
SPEC_ATTEMPTS = 80
SHAPE_ATTEMPTS = 120
AREA_RANGE = (2, 7)  # exclusive upper bound
NUM_FENCES_RANGE = (1, 5)  # exclusive upper bound
CLIP_PROB = 0.28


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[np.argmax(counts)])


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            r0 = int(coords[:, 0].min())
            c0 = int(coords[:, 1].min())
            r1 = int(coords[:, 0].max())
            c1 = int(coords[:, 1].max())
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _identify_fence(grid: np.ndarray, bg: int, comp: dict) -> dict | None:
    color = int(comp["color"])
    coords = comp["coords"]
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 5 or w < 5:
        return None

    points = {(int(r), int(c)) for r, c in coords}
    boundary = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }
    if points - boundary:
        return None
    missing = boundary - points
    if len(missing) != 1:
        return None
    gap = next(iter(missing))
    if gap in {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}:
        return None
    if len(points) != len(boundary) - 1:
        return None

    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    inner_mask = (inner != bg) & (inner != color)
    if not np.any(inner_mask):
        return None

    gr, gc = gap
    if gr == r0:
        direction = "up"
    elif gr == r1:
        direction = "down"
    elif gc == c0:
        direction = "left"
    elif gc == c1:
        direction = "right"
    else:
        return None

    return {
        "color": color,
        "coords": coords,
        "bbox": (r0, c0, r1, c1),
        "gap": gap,
        "direction": direction,
    }


def _inner_object(grid: np.ndarray, bg: int, fence: dict) -> dict | None:
    r0, c0, r1, c1 = fence["bbox"]
    fence_color = int(fence["color"])
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    mask = (inner != bg) & (inner != fence_color)
    comps = _components_from_mask(mask)
    if len(comps) != 1:
        return None

    coords_local = comps[0]
    coords = coords_local.copy()
    coords[:, 0] += r0 + 1
    coords[:, 1] += c0 + 1
    values = np.array([int(grid[r, c]) for r, c in coords], dtype=int)
    return {"coords": coords, "values": values, "area": int(coords.shape[0])}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Launch each fenced object out through its wall gap by a distance equal to its area.

    Input structure:
    - The background is black.
    - One or more orange hollow rectangular fences appear as 1-pixel-thick frames with exactly one 1-pixel gap on one side.
    - Each fence contains exactly one connected non-background object fully inside the frame.

    Rule:
    - For every fence, detect which side contains the missing border pixel and use that side as the launch direction.
    - Count the enclosed object's pixel area N, erase that object from the fence interior, and translate the whole object by N cells in the gap direction.
    - Draw the translated object with its original shape and colors, clipping any cells that leave the grid, while keeping all fence pixels unchanged.

    Key insight:
    - The gap tells you where the object travels, but the travel distance is not geometric frame size; it is the enclosed object's own area.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    out = g.copy()

    fences = []
    fence_mask = np.zeros_like(g, dtype=bool)
    for comp in _monochrome_components(g, bg):
        fence = _identify_fence(g, bg, comp)
        if fence is None:
            continue
        obj = _inner_object(g, bg, fence)
        if obj is None:
            continue
        fence["object_coords"] = obj["coords"]
        fence["object_values"] = obj["values"]
        fence["area"] = obj["area"]
        fences.append(fence)
        fence_mask[fence["coords"][:, 0], fence["coords"][:, 1]] = True

    for fence in fences:
        coords = fence["object_coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    h, w = g.shape
    for fence in fences:
        dr, dc = DIRS[fence["direction"]]
        shift_r = dr * int(fence["area"])
        shift_c = dc * int(fence["area"])
        for (r, c), value in zip(fence["object_coords"], fence["object_values"]):
            rr = int(r) + shift_r
            cc = int(c) + shift_c
            if 0 <= rr < h and 0 <= cc < w and not fence_mask[rr, cc]:
                out[rr, cc] = int(value)

    return out


def _grow_connected_patch(rng: np.random.Generator, h: int, w: int, area: int) -> np.ndarray | None:
    seed = (int(rng.integers(0, h)), int(rng.integers(0, w)))
    cells = {seed}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(frontier[int(rng.integers(0, len(frontier)))])

    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    r0, c0 = min(rs), min(cs)
    patch = np.zeros((max(rs) - r0 + 1, max(cs) - c0 + 1), dtype=bool)
    for r, c in cells:
        patch[r - r0, c - c0] = True
    return patch


def _make_object_patch(rng: np.random.Generator, area: int, direction: str) -> np.ndarray:
    axis_vertical = direction in ("up", "down")
    max_dim = min(area, 4)
    dims = [
        (h, w)
        for h in range(1, max_dim + 1)
        for w in range(1, max_dim + 1)
        if h * w >= area
    ]
    for _ in range(SHAPE_ATTEMPTS):
        h, w = dims[int(rng.integers(0, len(dims)))]
        patch = _grow_connected_patch(rng, h, w, area)
        if patch is None:
            continue
        ph, pw = map(int, patch.shape)
        if axis_vertical and ph >= area:
            continue
        if not axis_vertical and pw >= area:
            continue
        return patch
    raise RuntimeError("failed to generate a non-degenerate object patch")


def _build_spec(rng: np.random.Generator, color: int) -> dict:
    direction = str(rng.choice(np.array(["up", "down", "left", "right"], dtype=object)))
    area = int(rng.integers(*AREA_RANGE))
    patch = _make_object_patch(rng, area, direction)
    obj_h, obj_w = map(int, patch.shape)

    if direction in ("up", "down"):
        side_left = int(rng.integers(1, 4))
        side_right = int(rng.integers(1, 4))
        tail_space = int(rng.integers(1, 4))
        inner_h = obj_h + tail_space
        inner_w = side_left + obj_w + side_right
        obj_top = 1 if direction == "up" else 1 + inner_h - obj_h
        obj_left = 1 + side_left
        gap = (0 if direction == "up" else inner_h + 1, obj_left + int(rng.integers(0, obj_w)))
    else:
        side_top = int(rng.integers(1, 4))
        side_bottom = int(rng.integers(1, 4))
        tail_space = int(rng.integers(1, 4))
        inner_h = side_top + obj_h + side_bottom
        inner_w = obj_w + tail_space
        obj_top = 1 + side_top
        obj_left = 1 if direction == "left" else 1 + inner_w - obj_w
        gap = (obj_top + int(rng.integers(0, obj_h)), 0 if direction == "left" else inner_w + 1)

    frame_h = inner_h + 2
    frame_w = inner_w + 2

    frame_cells = []
    for r in range(frame_h):
        for c in range(frame_w):
            if r in (0, frame_h - 1) or c in (0, frame_w - 1):
                if (r, c) != gap:
                    frame_cells.append((r, c))

    object_cells = [(obj_top + r, obj_left + c) for r, c in np.argwhere(patch)]
    step_r, step_c = DIRS[direction]
    dst_cells = [(r + step_r * area, c + step_c * area) for r, c in object_cells]

    return {
        "direction": direction,
        "area": area,
        "color": int(color),
        "frame_h": frame_h,
        "frame_w": frame_w,
        "frame_cells": frame_cells,
        "object_cells": object_cells,
        "dst_cells": dst_cells,
        "clip": bool(rng.random() < CLIP_PROB),
    }


def _reserve(mask: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(mask.shape[0], r0 + h + pad)
    cc1 = min(mask.shape[1], c0 + w + pad)
    mask[rr0:rr1, cc0:cc1] = True


def _place_spec(
    rng: np.random.Generator,
    spec: dict,
    grid_h: int,
    grid_w: int,
    input_reserved: np.ndarray,
    output_blocked: np.ndarray,
) -> dict | None:
    positions = [(r0, c0) for r0 in range(grid_h - spec["frame_h"] + 1) for c0 in range(grid_w - spec["frame_w"] + 1)]
    rng.shuffle(positions)
    for top, left in positions:
        if np.any(input_reserved[max(0, top - 1) : min(grid_h, top + spec["frame_h"] + 1), max(0, left - 1) : min(grid_w, left + spec["frame_w"] + 1)]):
            continue
        if np.any(output_blocked[top : top + spec["frame_h"], left : left + spec["frame_w"]]):
            continue

        dest_in_bounds = []
        fully_in_bounds = True
        for r, c in spec["dst_cells"]:
            rr = top + int(r)
            cc = left + int(c)
            if 0 <= rr < grid_h and 0 <= cc < grid_w:
                dest_in_bounds.append((rr, cc))
            else:
                fully_in_bounds = False
        if not dest_in_bounds:
            continue
        if spec["clip"] and fully_in_bounds:
            continue
        if not spec["clip"] and not fully_in_bounds:
            continue
        if any(output_blocked[r, c] for r, c in dest_in_bounds):
            continue

        _reserve(input_reserved, top, left, spec["frame_h"], spec["frame_w"], pad=1)
        output_blocked[top : top + spec["frame_h"], left : left + spec["frame_w"]] = True
        for r, c in dest_in_bounds:
            output_blocked[r, c] = True

        frame_cells = [(top + r, left + c) for r, c in spec["frame_cells"]]
        source_cells = [(top + r, left + c) for r, c in spec["object_cells"]]
        return {
            **spec,
            "top": top,
            "left": left,
            "frame_cells_global": frame_cells,
            "source_cells_global": source_cells,
            "dest_cells_global": dest_in_bounds,
        }
    return None


def _render_input(grid_h: int, grid_w: int, placed_specs: list[dict]) -> np.ndarray:
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for spec in placed_specs:
        for r, c in spec["frame_cells_global"]:
            grid[r, c] = FENCE
        for r, c in spec["source_cells_global"]:
            grid[r, c] = int(spec["color"])
    return grid


def _expected_output(inp: np.ndarray, placed_specs: list[dict]) -> np.ndarray:
    out = inp.copy()
    fence_mask = inp == FENCE
    for spec in placed_specs:
        for r, c in spec["source_cells_global"]:
            out[r, c] = BG
    for spec in placed_specs:
        for r, c in spec["dest_cells_global"]:
            if not fence_mask[r, c]:
                out[r, c] = int(spec["color"])
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_fences = int(rng.integers(*NUM_FENCES_RANGE))
    min_side = 14 + 2 * n_fences
    grid_h = int(rng.integers(min_side, 31))
    grid_w = int(rng.integers(min_side, 31))

    colors = [int(v) for v in rng.choice(OBJECT_COLORS, size=n_fences, replace=False)]
    input_reserved = np.zeros((grid_h, grid_w), dtype=bool)
    output_blocked = np.zeros((grid_h, grid_w), dtype=bool)

    placed_specs = []
    for color in colors:
        placed = None
        for _ in range(SPEC_ATTEMPTS):
            spec = _build_spec(rng, color)
            placed = _place_spec(rng, spec, grid_h, grid_w, input_reserved, output_blocked)
            if placed is not None:
                placed_specs.append(placed)
                break
        if placed is None:
            return None

    inp = _render_input(grid_h, grid_w, placed_specs)
    out = _expected_output(inp, placed_specs)
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
