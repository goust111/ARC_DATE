from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import bbox_coords, components_mask, get_bg, in_bounds


BG = 0
BLUE = 1
RED = 2
MASK_COLORS = (3, 4, 5, 7)
FRAME_SIZES = (6, 7)
BORDER_PALETTE = {RED, *MASK_COLORS}

GEN_ATTEMPTS = 400
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _border_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros(shape, dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _border_coords(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    coords: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
        if r1 != r0:
            coords.append((r1, c))
    for r in range(r0 + 1, r1):
        coords.append((r, c0))
        if c1 != c0:
            coords.append((r, c1))
    return coords


def _cw_angle(center: tuple[float, float], centroid: tuple[float, float]) -> float:
    ang = math.atan2(center[0] - centroid[0], center[1] - centroid[1])
    return ang if ang >= 0 else ang + 2 * math.pi


def _connected_mask(size: int, color: int) -> np.ndarray:
    total = (size - 2) * (size - 2)
    min_fill = max(3, math.ceil(total * 0.30))
    max_fill = max(min_fill, math.floor(total * 0.55))
    target = random.randint(min_fill, max_fill)

    mask = np.zeros((size - 2, size - 2), dtype=int)
    start = ((size - 2) // 2, (size - 2) // 2)
    mask[start] = color
    frontier = {start}
    filled = {start}

    while len(filled) < target:
        candidates = set()
        for r, c in frontier:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] == 0:
                        candidates.add((nr, nc))
        if not candidates:
            break
        cell = random.choice(sorted(candidates))
        mask[cell] = color
        frontier.add(cell)
        filled.add(cell)
    assert np.any(mask == color)
    return mask


def _frame_candidates_for_color(g: np.ndarray, color: int) -> list[dict]:
    H, W = g.shape
    candidates: list[dict] = []
    for size in FRAME_SIZES:
        for r0 in range(H - size + 1):
            r1 = r0 + size - 1
            for c0 in range(W - size + 1):
                c1 = c0 + size - 1
                interior = g[r0 + 1 : r1, c0 + 1 : c1]
                interior_colors = set(map(int, np.unique(interior))) - {BG}
                if interior_colors != {int(color)}:
                    continue
                border_vals = [
                    int(g[r, c])
                    for r, c in _border_coords((r0, c0, r1, c1))
                ]
                if not border_vals or any(v not in BORDER_PALETTE for v in border_vals):
                    continue
                if RED not in border_vals:
                    continue
                mask = interior == color
                candidates.append(
                    {
                        "bbox": (r0, c0, r1, c1),
                        "center": ((r0 + r1) / 2.0, (c0 + c1) / 2.0),
                        "color": int(color),
                        "mask": mask,
                    }
                )
    return candidates


def _infer_frames(g: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    seen_bboxes = set()
    for color in MASK_COLORS:
        candidates = _frame_candidates_for_color(g, color)
        assert len(candidates) == 1, f"expected one frame for mask color {color}, got {len(candidates)}"
        frame = candidates[0]
        bbox = frame["bbox"]
        assert bbox not in seen_bboxes, "frame bboxes must be unique"
        seen_bboxes.add(bbox)
        frames.append(frame)
    assert len(frames) == 4
    return frames


def _blue_cycle_edges(g: np.ndarray, frames: list[dict]) -> set[tuple[int, int]]:
    owner: dict[tuple[int, int], int] = {}
    for i, frame in enumerate(frames):
        for cell in _border_coords(frame["bbox"]):
            owner[cell] = i

    edges: set[tuple[int, int]] = set()
    blue_components = components_mask(g == BLUE, connectivity=4)
    assert len(blue_components) == 4, "expected four blue connector components"
    for coords in blue_components:
        touched = set()
        for r_raw, c_raw in coords:
            r = int(r_raw)
            c = int(c_raw)
            for dr, dc in DIR4:
                frame_id = owner.get((r + dr, c + dc))
                if frame_id is not None:
                    touched.add(frame_id)
        assert len(touched) == 2, f"connector must touch exactly two frames, got {touched}"
        a, b = sorted(touched)
        edges.add((a, b))

    for i in range(len(frames)):
        assert sum(i in edge for edge in edges) == 2, "every frame must touch two connectors"
    return edges


def _clockwise_order(frames: list[dict], edges: set[tuple[int, int]]) -> list[int]:
    centroid = (
        sum(frame["center"][0] for frame in frames) / len(frames),
        sum(frame["center"][1] for frame in frames) / len(frames),
    )
    angles = [_cw_angle(frame["center"], centroid) for frame in frames]
    assert len({round(a, 12) for a in angles}) == len(angles), "frame angles must be distinct"
    order = sorted(range(len(frames)), key=lambda i: angles[i])
    for a, b in zip(order, order[1:] + order[:1]):
        assert tuple(sorted((a, b))) in edges, "blue cycle must agree with clockwise order"
    return order


def _occlusion_depths(g: np.ndarray, frames: list[dict]) -> list[int]:
    masks = [_border_mask(g.shape, frame["bbox"]) for frame in frames]
    depths = [0] * len(frames)
    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            overlap = masks[i] & masks[j]
            if not overlap.any():
                continue
            rr, cc = np.nonzero(overlap)
            visible = {int(g[r, c]) for r, c in zip(rr, cc)}
            allowed = {frames[i]["color"], frames[j]["color"]}
            assert visible <= allowed and visible, f"overlap must show one frame color, got {visible}"
            assert len(visible) == 1, "pairwise overlap must expose a single front frame"
            top = next(iter(visible))
            if top == frames[i]["color"]:
                depths[j] += 1
            else:
                depths[i] += 1
    return depths


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each framed interior mask by rotating along the blue frame ring with a step count given by border occlusion depth.

    Input structure:
    - The grid has black background, four hollow red square frames, and four blue connector components that link the frames into one cycle.
    - Each frame contains exactly one non-background interior mask color, with the four mask colors being green, yellow, grey, and orange.
    - Adjacent frames in the cycle overlap only on border cells; each overlap shows exactly one of the two frame mask colors, indicating which frame is visible in front there.

    Rule:
    - Infer the four square frames and sort their centers clockwise around the shared centroid, starting from the rightward direction.
    - Let the clockwise color ring be the list of current interior mask colors in that order.
    - For each frame, count how many different other frames cover it at border overlaps; that occlusion depth is the number of clockwise steps to move on the color ring.
    - Recolor that frame's interior mask in place with the reached color, leaving frame borders, overlap markers, blue connectors, background, and interior holes unchanged.

    Key insight:
    - The blue ring supplies only the direction of color transfer, while the visible overlap color independently supplies how many steps that frame advances along the ring.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    assert get_bg(g) == BG

    frames = _infer_frames(g)
    edges = _blue_cycle_edges(g, frames)
    order = _clockwise_order(frames, edges)
    depths = _occlusion_depths(g, frames)

    colors = [frames[i]["color"] for i in order]
    out = g.copy()
    for idx, frame_id in enumerate(order):
        frame = frames[frame_id]
        new_color = colors[(idx + depths[frame_id]) % len(order)]
        r0, c0, r1, c1 = frame["bbox"]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[frame["mask"]] = int(new_color)
    return out


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = RED
    grid[r1, c0 : c1 + 1] = RED
    grid[r0 : r1 + 1, c0] = RED
    grid[r0 : r1 + 1, c1] = RED


def _shared_segment(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    ma = _border_mask((max(a[2], b[2]) + 1, max(a[3], b[3]) + 1), a)
    mb = _border_mask((max(a[2], b[2]) + 1, max(a[3], b[3]) + 1), b)
    rr, cc = np.nonzero(ma & mb)
    return [(int(r), int(c)) for r, c in zip(rr, cc)]


def _rect_path(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    for (r0, c0), (r1, c1) in zip(points, points[1:]):
        assert r0 == r1 or c0 == c1
        if r0 == r1:
            step = 1 if c1 >= c0 else -1
            segment = [(r0, c) for c in range(c0, c1 + step, step)]
        else:
            step = 1 if r1 >= r0 else -1
            segment = [(r, c0) for r in range(r0, r1 + step, step)]
        if path:
            segment = segment[1:]
        path.extend(segment)
    return path


def _paint_connector(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    blocked: set[tuple[int, int]],
    frame_border_cells: set[tuple[int, int]],
) -> bool:
    for cell in path:
        if not in_bounds(grid.shape, *cell):
            return False
        if cell in blocked or cell in frame_border_cells:
            return False
    for cell in path:
        grid[cell] = BLUE
        blocked.add(cell)
    return True


def _connector_path(
    start: tuple[int, int],
    end: tuple[int, int],
    mode: str,
    corridor: int,
) -> list[tuple[int, int]]:
    if mode in {"top", "bottom"}:
        return _rect_path([start, (corridor, start[1]), (corridor, end[1]), end])
    return _rect_path([start, (start[0], corridor), (end[0], corridor), end])


def _attachment_choices(bbox: tuple[int, int, int, int], side: str) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    r0, c0, r1, c1 = bbox
    choices: list[tuple[tuple[int, int], tuple[int, int]]] = []
    if side == "top":
        for c in range(c0 + 1, c1):
            choices.append(((r0, c), (r0 - 1, c)))
    elif side == "bottom":
        for c in range(c0 + 1, c1):
            choices.append(((r1, c), (r1 + 1, c)))
    elif side == "left":
        for r in range(r0 + 1, r1):
            choices.append(((r, c0), (r, c0 - 1)))
    elif side == "right":
        for r in range(r0 + 1, r1):
            choices.append(((r, c1), (r, c1 + 1)))
    else:
        raise ValueError(side)
    random.shuffle(choices)
    return choices


def _build_layout(size: int, overlap_len: int) -> list[tuple[int, int, int, int]]:
    cluster_top = 0
    cluster_left = size - overlap_len
    top = (cluster_top, cluster_left, cluster_top + size - 1, cluster_left + size - 1)
    right = (
        cluster_top + size - 1,
        cluster_left + size - overlap_len,
        cluster_top + 2 * size - 2,
        cluster_left + 2 * size - overlap_len - 1,
    )
    bottom = (
        cluster_top + 2 * size - 2,
        cluster_left,
        cluster_top + 3 * size - 3,
        cluster_left + size - 1,
    )
    left = (
        cluster_top + size - 1,
        cluster_left - (size - overlap_len),
        cluster_top + 2 * size - 2,
        cluster_left + overlap_len - 1,
    )
    return [top, right, bottom, left]


def _shift_bboxes(
    bboxes: list[tuple[int, int, int, int]],
    dr: int,
    dc: int,
) -> list[tuple[int, int, int, int]]:
    return [(r0 + dr, c0 + dc, r1 + dr, c1 + dc) for r0, c0, r1, c1 in bboxes]


def _depths_from_fronts(fronts: dict[tuple[int, int], int], frame_count: int) -> list[int]:
    depths = [0] * frame_count
    for (a, b), top in fronts.items():
        if top == a:
            depths[b] += 1
        else:
            depths[a] += 1
    return depths


def _generate_connectors(
    grid: np.ndarray,
    bboxes: list[tuple[int, int, int, int]],
) -> bool:
    frame_border_cells = {cell for bbox in bboxes for cell in _border_coords(bbox)}
    blocked: set[tuple[int, int]] = set()
    connectors = [
        (0, "top", 1, "top", "top"),
        (1, "right", 2, "right", "right"),
        (2, "bottom", 3, "bottom", "bottom"),
        (3, "left", 0, "left", "left"),
    ]

    for a, side_a, b, side_b, mode in connectors:
        placed = False
        choices_a = _attachment_choices(bboxes[a], side_a)
        choices_b = _attachment_choices(bboxes[b], side_b)
        for border_a, start in choices_a[:4]:
            del border_a
            for border_b, end in choices_b[:4]:
                del border_b
                if mode == "top":
                    corridor = min(start[0], end[0]) - random.randint(1, 2)
                elif mode == "bottom":
                    corridor = max(start[0], end[0]) + random.randint(1, 2)
                elif mode == "left":
                    corridor = min(start[1], end[1]) - random.randint(1, 2)
                else:
                    corridor = max(start[1], end[1]) + random.randint(1, 2)
                path = _connector_path(start, end, mode, corridor)
                if _paint_connector(grid, path, blocked, frame_border_cells):
                    placed = True
                    break
            if placed:
                break
        if not placed:
            return False
    return True


def _expected_output_from_layout(
    grid: np.ndarray,
    bboxes: list[tuple[int, int, int, int]],
    colors: list[int],
    depths: list[int],
) -> np.ndarray:
    frames = []
    for bbox, color in zip(bboxes, colors):
        r0, c0, r1, c1 = bbox
        frames.append(
            {
                "bbox": bbox,
                "center": ((r0 + r1) / 2.0, (c0 + c1) / 2.0),
                "color": color,
                "mask": grid[r0 + 1 : r1, c0 + 1 : c1] == color,
            }
        )
    centroid = (
        sum(frame["center"][0] for frame in frames) / 4.0,
        sum(frame["center"][1] for frame in frames) / 4.0,
    )
    order = sorted(range(4), key=lambda i: _cw_angle(frames[i]["center"], centroid))
    ring = [frames[i]["color"] for i in order]

    out = grid.copy()
    for idx, frame_id in enumerate(order):
        frame = frames[frame_id]
        new_color = ring[(idx + depths[frame_id]) % 4]
        r0, c0, r1, c1 = frame["bbox"]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[frame["mask"]] = new_color
    return out


def _semantic_check(
    inp: np.ndarray,
    expected: np.ndarray,
    placed_bboxes: list[tuple[int, int, int, int]],
    depths: list[int],
) -> bool:
    try:
        inferred = _infer_frames(inp)
        inferred_bboxes = sorted(frame["bbox"] for frame in inferred)
        if inferred_bboxes != sorted(placed_bboxes):
            return False
        depth_by_bbox = {bbox: depth for bbox, depth in zip(placed_bboxes, depths)}
        edges = _blue_cycle_edges(inp, inferred)
        order = _clockwise_order(inferred, edges)
        if sorted(order) != [0, 1, 2, 3]:
            return False
        actual_depths = _occlusion_depths(inp, inferred)
        for frame, actual_depth in zip(inferred, actual_depths):
            if actual_depth != depth_by_bbox[frame["bbox"]]:
                return False
        solved = main(inp)
        if not np.array_equal(solved, expected):
            return False
        if np.array_equal(inp, expected):
            return False
        changed_frames = 0
        for frame in inferred:
            r0, c0, r1, c1 = frame["bbox"]
            before = inp[r0 + 1 : r1, c0 + 1 : c1]
            after = expected[r0 + 1 : r1, c0 + 1 : c1]
            if not np.array_equal(before, after):
                changed_frames += 1
        return changed_frames >= 2 and len(set(depths)) >= 2
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    size = random.choice(FRAME_SIZES)
    overlap_len = random.choice([2, 3])
    bboxes = _build_layout(size, overlap_len)
    u_r0 = min(b[0] for b in bboxes)
    u_c0 = min(b[1] for b in bboxes)
    u_r1 = max(b[2] for b in bboxes)
    u_c1 = max(b[3] for b in bboxes)
    cluster_h = u_r1 - u_r0 + 1
    cluster_w = u_c1 - u_c0 + 1

    H = random.randint(max(22, cluster_h + 6), 30)
    W = random.randint(max(22, cluster_w + 6), 30)
    top_margin = random.randint(3, H - cluster_h - 3)
    left_margin = random.randint(3, W - cluster_w - 3)
    bboxes = _shift_bboxes(bboxes, top_margin - u_r0, left_margin - u_c0)

    colors = list(MASK_COLORS)
    random.shuffle(colors)

    fronts: dict[tuple[int, int], int] = {}
    for pair in ((0, 1), (1, 2), (2, 3), (3, 0)):
        fronts[pair] = random.choice(pair)
    depths = _depths_from_fronts(fronts, 4)
    if len(set(depths)) < 2:
        return None

    grid = np.zeros((H, W), dtype=int)
    for bbox, color in zip(bboxes, colors):
        _draw_frame(grid, bbox)
        r0, c0, r1, c1 = bbox
        grid[r0 + 1 : r1, c0 + 1 : c1] = _connected_mask(size, color)

    for (a, b), top in fronts.items():
        shared = _shared_segment(bboxes[a], bboxes[b])
        if len(shared) < overlap_len:
            return None
        top_color = colors[top]
        for r, c in shared:
            grid[r, c] = top_color

    if not _generate_connectors(grid, bboxes):
        return None

    expected = _expected_output_from_layout(grid, bboxes, colors, depths)
    if not _semantic_check(grid, expected, bboxes, depths):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
