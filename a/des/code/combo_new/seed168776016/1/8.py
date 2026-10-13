import random
from collections import deque

import numpy as np


BG_COLOR = 0
PIPE_COLOR = 5
# ARC's standard palette has no true white, so cyan(8) is the visible stand-in.
SOURCE_COLOR = 8
ORANGE_COLOR = 7
TARGET_COLORS = (1, 2, 3, 4, 6, 9)

RECT_SIDE_MIN = 2
RECT_SIDE_MAX = 4
PIPE_LEN_MIN = 2
PIPE_LEN_MAX = 5
PADDING_MAX = 2
MAX_GENERATE_ATTEMPTS = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _connected_components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _line_cells(a, b):
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        return [(r0, c) for c in range(lo, hi + 1)]
    if c0 == c1:
        lo, hi = sorted((r0, r1))
        return [(r, c0) for r in range(lo, hi + 1)]
    raise AssertionError("segments must be axis-aligned")


def _side_contacts(pipe_mask: np.ndarray, bbox):
    r0, c0, r1, c1 = bbox
    h, w = pipe_mask.shape
    contacts = {"top": [], "bottom": [], "left": [], "right": []}
    if r0 > 0:
        contacts["top"] = [(r0 - 1, c) for c in range(c0, c1 + 1) if pipe_mask[r0 - 1, c]]
    if r1 + 1 < h:
        contacts["bottom"] = [(r1 + 1, c) for c in range(c0, c1 + 1) if pipe_mask[r1 + 1, c]]
    if c0 > 0:
        contacts["left"] = [(r, c0 - 1) for r in range(r0, r1 + 1) if pipe_mask[r, c0 - 1]]
    if c1 + 1 < w:
        contacts["right"] = [(r, c1 + 1) for r in range(r0, r1 + 1) if pipe_mask[r, c1 + 1]]
    return contacts


def _paint_side(out: np.ndarray, bbox, side: str, color: int):
    r0, c0, r1, c1 = bbox
    if side == "top":
        out[r0, c0 : c1 + 1] = color
    elif side == "bottom":
        out[r1, c0 : c1 + 1] = color
    elif side == "left":
        out[r0 : r1 + 1, c0] = color
    elif side == "right":
        out[r0 : r1 + 1, c1] = color
    else:
        raise AssertionError(f"unknown side: {side}")


def _analyze_grid(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    rect_mask = (g != BG_COLOR) & (g != PIPE_COLOR)
    rects = []
    for coords in _connected_components(rect_mask):
        colors = {int(g[r, c]) for r, c in coords}
        assert len(colors) == 1, "each block must be monochrome"
        color = next(iter(colors))
        bbox = _bbox(coords)
        r0, c0, r1, c1 = bbox
        rh = r1 - r0 + 1
        rw = c1 - c0 + 1
        assert rh >= 2 and rw >= 2, "blocks must be at least 2x2"
        assert len(coords) == rh * rw, "blocks must be solid rectangles"
        assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color), "rectangle area must be solid"
        rects.append({"color": int(color), "bbox": bbox, "coords": coords})

    sources = [rect for rect in rects if rect["color"] == SOURCE_COLOR]
    assert len(sources) == 1, "expected exactly one white source block"
    source = sources[0]
    targets = [rect for rect in rects if rect is not source]
    assert targets, "expected at least one target block"

    target_colors = [rect["color"] for rect in targets]
    assert len(set(target_colors)) == len(target_colors), "target colors must be unique"

    pipe_mask = g == PIPE_COLOR
    pipe_cells = [tuple(map(int, rc)) for rc in np.argwhere(pipe_mask)]
    assert pipe_cells, "expected gray pipes"

    cross_mask = np.zeros_like(pipe_mask, dtype=bool)
    for r, c in pipe_cells:
        if (
            0 < r < h - 1
            and 0 < c < w - 1
            and pipe_mask[r - 1, c]
            and pipe_mask[r + 1, c]
            and pipe_mask[r, c - 1]
            and pipe_mask[r, c + 1]
        ):
            cross_mask[r, c] = True

    graph = {cell: set() for cell in pipe_cells}
    for r, c in pipe_cells:
        for nr, nc in _neighbors4(r, c, h, w):
            if pipe_mask[nr, nc]:
                graph[(r, c)].add((nr, nc))

    source_contacts = _side_contacts(pipe_mask, source["bbox"])
    source_starts = {cell for cells in source_contacts.values() for cell in cells}
    assert source_starts, "source must touch the pipe network"

    supernode = ("source",)
    graph[supernode] = set(source_starts)
    for start in source_starts:
        graph[start].add(supernode)

    parent = {supernode: None}
    crossed = {supernode: False}
    q = deque([supernode])
    while q:
        node = q.popleft()
        for nb in graph[node]:
            if nb not in parent:
                parent[nb] = node
                crossed[nb] = crossed[node] or (nb != supernode and bool(cross_mask[nb]))
                q.append(nb)
            elif parent[node] != nb:
                raise AssertionError("pipe network must give unique source-to-target paths")

    for cell in pipe_cells:
        assert cell in parent, "all pipes must be reachable from the source"

    analyzed_targets = []
    for rect in targets:
        contacts = _side_contacts(pipe_mask, rect["bbox"])
        entrance_sides = [side for side, cells in contacts.items() if cells]
        assert len(entrance_sides) == 1, "each target must have exactly one pipe entrance side"
        entrance_side = entrance_sides[0]
        entrance_cells = contacts[entrance_side]
        assert len(entrance_cells) == 1, "pipe should touch a target at one endpoint cell"
        entrance_cell = entrance_cells[0]
        assert entrance_cell in parent, "target must be reachable from the source"
        analyzed_targets.append(
            {
                "color": rect["color"],
                "bbox": rect["bbox"],
                "entrance_side": entrance_side,
                "entrance_cell": entrance_cell,
                "crossed": bool(crossed[entrance_cell]),
            }
        )

    return {
        "source": source,
        "targets": analyzed_targets,
        "pipe_cells": pipe_cells,
        "cross_mask": cross_mask,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn the source-connected pipe network white and mark each target according to whether its path crosses an intersection.

    Input structure:
    - The background is black.
    - Several solid colored rectangles are present; exactly one rectangle is white and all others are distinct target colors.
    - Gray one-cell-wide horizontal or vertical pipe segments connect rectangle edges, and some pipes may cross at plus-shaped intersections.

    Rule:
    - Treat the white rectangle as the source and follow the pipe network to every target rectangle.
    - Recolor every pipe cell that lies on a source-to-target route from gray to white.
    - For each target, keep its shape, paint a one-cell white border along the side where its pipe enters, and leave other borders unchanged.
    - If that target's unique path from the source passes through at least one pipe crossing, recolor the whole target orange before drawing the white border; otherwise keep the target's original color.

    Key insight:
    - The important hidden attribute is not only connectivity but whether a target's unique path traverses a plus-shaped pipe intersection, because that single topological event triggers the target recolor.
    """
    g = np.asarray(grid, dtype=int)
    scene = _analyze_grid(g)

    out = g.copy()
    for r, c in scene["pipe_cells"]:
        out[r, c] = SOURCE_COLOR

    for target in scene["targets"]:
        if target["crossed"]:
            r0, c0, r1, c1 = target["bbox"]
            out[r0 : r1 + 1, c0 : c1 + 1] = ORANGE_COLOR
        _paint_side(out, target["bbox"], target["entrance_side"], SOURCE_COLOR)

    return out


def _rand_dim(rng: random.Random):
    return rng.randint(RECT_SIDE_MIN, RECT_SIDE_MAX)


def _symmetry(grid: np.ndarray, rng: random.Random):
    out = np.array(grid, copy=True)
    out = np.rot90(out, k=rng.randrange(4))
    if rng.random() < 0.5:
        out = np.fliplr(out)
    return out


def _pad_grid(grid: np.ndarray, rng: random.Random):
    h, w = grid.shape
    max_top = min(PADDING_MAX, 30 - h)
    max_bottom = min(PADDING_MAX, 30 - h)
    max_left = min(PADDING_MAX, 30 - w)
    max_right = min(PADDING_MAX, 30 - w)
    top = rng.randint(0, max_top)
    bottom = rng.randint(0, min(max_bottom, 30 - h - top))
    left = rng.randint(0, max_left)
    right = rng.randint(0, min(max_right, 30 - w - left))
    out = np.full((h + top + bottom, w + left + right), BG_COLOR, dtype=int)
    out[top : top + h, left : left + w] = grid
    return out


def _draw_rect(occupancy: dict, bbox, color: int):
    r0, c0, r1, c1 = bbox
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if (r, c) in occupancy:
                raise AssertionError("rectangles may not overlap")
            occupancy[(r, c)] = color


def _draw_pipe(pipe_cells: set, occupancy: dict, cells, shared):
    for cell in cells:
        if cell in occupancy:
            raise AssertionError("pipes may not overlap blocks")
        if cell in pipe_cells and cell != shared:
            raise AssertionError("unexpected pipe overlap")
        pipe_cells.add(cell)


def _gen_attempt(rng: random.Random):
    direct_count = rng.randint(1, 3)
    direct_sides = rng.sample(["top", "bottom", "left"], k=direct_count)
    target_colors = rng.sample(TARGET_COLORS, 3 + direct_count)
    color_iter = iter(target_colors)

    source_h = _rand_dim(rng)
    source_w = _rand_dim(rng)
    source_right_row = rng.randrange(source_h)
    source_len = rng.randint(PIPE_LEN_MIN, PIPE_LEN_MAX)

    source_r0 = -source_right_row
    source_c1 = -source_len - 1
    source_c0 = source_c1 - source_w + 1
    source_bbox = (source_r0, source_c0, source_r0 + source_h - 1, source_c1)

    right_h = _rand_dim(rng)
    right_w = _rand_dim(rng)
    right_row = rng.randrange(right_h)
    right_len = rng.randint(PIPE_LEN_MIN, PIPE_LEN_MAX)
    right_bbox = (-right_row, right_len + 1, -right_row + right_h - 1, right_len + right_w)

    up_h = _rand_dim(rng)
    up_w = _rand_dim(rng)
    up_col = rng.randrange(up_w)
    up_len = rng.randint(PIPE_LEN_MIN, PIPE_LEN_MAX)
    up_r1 = -up_len - 1
    up_bbox = (up_r1 - up_h + 1, -up_col, up_r1, -up_col + up_w - 1)

    down_h = _rand_dim(rng)
    down_w = _rand_dim(rng)
    down_col = rng.randrange(down_w)
    down_len = rng.randint(PIPE_LEN_MIN, PIPE_LEN_MAX)
    down_r0 = down_len + 1
    down_bbox = (down_r0, -down_col, down_r0 + down_h - 1, -down_col + down_w - 1)

    rects = [
        {"bbox": source_bbox, "color": SOURCE_COLOR},
        {"bbox": right_bbox, "color": next(color_iter)},
        {"bbox": up_bbox, "color": next(color_iter)},
        {"bbox": down_bbox, "color": next(color_iter)},
    ]

    segments = [
        _line_cells((0, source_c1 + 1), (0, 0)),
        _line_cells((0, 0), (0, right_bbox[1] - 1)),
        _line_cells((up_bbox[2] + 1, 0), (0, 0)),
        _line_cells((0, 0), (down_bbox[0] - 1, 0)),
    ]

    source_top_col = rng.randrange(source_w)
    source_bottom_col = rng.randrange(source_w)
    source_left_row = rng.randrange(source_h)

    for side in direct_sides:
        color = next(color_iter)
        th = _rand_dim(rng)
        tw = _rand_dim(rng)
        arm_len = rng.randint(PIPE_LEN_MIN, PIPE_LEN_MAX)

        if side == "top":
            attach_col = source_c0 + source_top_col
            target_c0 = attach_col - (tw - 1)
            target_r1 = source_bbox[0] - arm_len - 1
            target_bbox = (target_r1 - th + 1, target_c0, target_r1, target_c0 + tw - 1)
            segment = _line_cells((target_bbox[2] + 1, attach_col), (source_bbox[0] - 1, attach_col))
        elif side == "bottom":
            attach_col = source_c0 + source_bottom_col
            target_c0 = attach_col - (tw - 1)
            target_r0 = source_bbox[2] + arm_len + 1
            target_bbox = (target_r0, target_c0, target_r0 + th - 1, target_c0 + tw - 1)
            segment = _line_cells((source_bbox[2] + 1, attach_col), (target_bbox[0] - 1, attach_col))
        else:
            attach_row = source_bbox[0] + source_left_row
            target_row = rng.randrange(th)
            target_c1 = source_bbox[1] - arm_len - 1
            target_bbox = (attach_row - target_row, target_c1 - tw + 1, attach_row - target_row + th - 1, target_c1)
            segment = _line_cells((attach_row, target_bbox[3] + 1), (attach_row, source_bbox[1] - 1))

        rects.append({"bbox": target_bbox, "color": color})
        segments.append(segment)

    occupancy = {}
    try:
        for rect in rects:
            _draw_rect(occupancy, rect["bbox"], rect["color"])

        pipe_cells = set()
        for segment in segments:
            _draw_pipe(pipe_cells, occupancy, segment, shared=(0, 0))
    except AssertionError:
        return None

    rows = [r for r, _ in occupancy] + [r for r, _ in pipe_cells]
    cols = [c for _, c in occupancy] + [c for _, c in pipe_cells]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    local_h = max_r - min_r + 1
    local_w = max_c - min_c + 1
    if local_h > 30 or local_w > 30:
        return None

    local = np.full((local_h, local_w), BG_COLOR, dtype=int)
    for (r, c), color in occupancy.items():
        local[r - min_r, c - min_c] = color
    for r, c in pipe_cells:
        local[r - min_r, c - min_c] = PIPE_COLOR

    local = _symmetry(local, rng)
    local = _pad_grid(local, rng)
    if max(local.shape) > 30:
        return None

    try:
        scene = _analyze_grid(local)
    except AssertionError:
        return None

    crosses = [target["crossed"] for target in scene["targets"]]
    if not any(crosses) or all(crosses):
        return None

    _ = main(local)
    return local


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _gen_attempt(random)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATE_ATTEMPTS} attempts")
