import random
from collections import deque

import numpy as np


BG = 0
MASTER_SIZE = 30
RETRY_LIMIT = 400
LAYER_PATTERNS = {
    1: [(2,), (3,), (4,)],
    2: [(2, 2), (2, 3), (3, 2), (3, 3), (2, 4), (4, 2)],
    3: [
        (2, 2, 2),
        (2, 2, 3),
        (2, 3, 2),
        (3, 2, 2),
        (2, 3, 3),
        (3, 2, 3),
        (3, 3, 2),
        (3, 3, 3),
    ],
    4: [(a, b, c, d) for a in (2, 3) for b in (2, 3) for c in (2, 3) for d in (2, 3)],
}
INNER_LIMITS = {
    1: {2: 10, 3: 7, 4: 5},
    2: {2: 7, 3: 5, 4: 2},
    3: {2: 5, 3: 2},
    4: {2: 5, 3: 2},
}


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _component_objects(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue

            seen[r, c] = True
            q = deque([(r, c)])
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            rows = [pt[0] for pt in pts]
            cols = [pt[1] for pt in pts]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for pr, pc in pts:
                mask[pr - r0, pc - c0] = True

            if bool(mask.all()):
                kind = "solid"
            else:
                border = np.zeros_like(mask)
                border[0, :] = True
                border[-1, :] = True
                border[:, 0] = True
                border[:, -1] = True
                if mask.shape[0] < 3 or mask.shape[1] < 3 or not np.array_equal(mask, border):
                    raise AssertionError("unexpected non-rectangular object")
                kind = "frame"

            objects.append(
                {
                    "kind": kind,
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "pts": pts,
                }
            )

    return objects


def _expand_bbox(bbox: tuple[int, int, int, int], pad: int = 2) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return (r0 - pad, c0 - pad, r1 + pad, c1 + pad)


def _paint_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shift the innermost color outward through each nested-rectangle stack, replacing every frame with the color of its inner neighbor.

    Input structure:
    - The background is black.
    - The foreground consists of several disjoint groups of concentric rectangular layers.
    - Each group has 2 to 4 one-cell-thick rectangular frames with a one-cell black gap between consecutive layers, plus a solid rectangular center whose color differs from every frame color in that group.

    Rule:
    - For each group, order the layers from the solid center outward through the nested frames.
    - Keep the solid center unchanged.
    - Recolor the first frame to the solid center color, the second frame to the original first-frame color, and continue propagating each original color one layer outward.
    - Leave the black gaps and all group positions unchanged.

    Key insight:
    - The geometry is a containment hierarchy of rectangles; the transformation is a one-step outward color shift along that inner-to-outer hierarchy rather than an independent recoloring of each frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    objects = _component_objects(g)
    frames_by_bbox = {obj["bbox"]: obj for obj in objects if obj["kind"] == "frame"}
    solids = [obj for obj in objects if obj["kind"] == "solid"]

    out = g.copy()
    used_frames = set()
    for solid in sorted(solids, key=lambda obj: (obj["bbox"][0], obj["bbox"][1])):
        current_bbox = solid["bbox"]
        chain = []
        while True:
            next_bbox = _expand_bbox(current_bbox, pad=2)
            frame = frames_by_bbox.get(next_bbox)
            if frame is None:
                break
            chain.append(frame)
            current_bbox = next_bbox

        carry = int(solid["color"])
        for frame in chain:
            bbox = frame["bbox"]
            if bbox in used_frames:
                raise AssertionError("frame cannot belong to multiple nested groups")
            used_frames.add(bbox)
            next_carry = int(frame["color"])
            for r, c in frame["pts"]:
                out[r, c] = carry
            carry = next_carry

    if frames_by_bbox:
        assert used_frames == set(frames_by_bbox), "every frame should be linked to one solid center"
    return out


def _sample_layers(n_groups: int) -> list[int]:
    pattern = random.choice(LAYER_PATTERNS[n_groups])
    return list(pattern)


def _sample_group_specs(n_groups: int) -> list[dict]:
    specs = []
    for layers in _sample_layers(n_groups):
        limit = INNER_LIMITS[n_groups][layers]
        inner_h = random.randint(1, limit)
        inner_w = random.randint(1, limit)
        colors = random.sample(range(1, 10), layers + 1)
        specs.append(
            {
                "layers": layers,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "solid_color": colors[0],
                "frame_colors_inner_to_outer": colors[1:],
                "outer_h": inner_h + 4 * layers,
                "outer_w": inner_w + 4 * layers,
            }
        )
    return specs


def _choose_position(blocked: np.ndarray, box_h: int, box_w: int, pad: int) -> tuple[int, int] | None:
    h, w = blocked.shape
    candidates = []
    for r0 in range(h - box_h + 1):
        rr0 = max(0, r0 - pad)
        rr1 = min(h, r0 + box_h + pad)
        for c0 in range(w - box_w + 1):
            cc0 = max(0, c0 - pad)
            cc1 = min(w, c0 + box_w + pad)
            if not blocked[rr0:rr1, cc0:cc1].any():
                candidates.append((r0, c0))
    if not candidates:
        return None
    return random.choice(candidates)


def _paint_group(grid: np.ndarray, top: int, left: int, spec: dict) -> dict:
    layers = spec["layers"]
    inner_h = spec["inner_h"]
    inner_w = spec["inner_w"]
    frame_colors = spec["frame_colors_inner_to_outer"]
    outer_h = spec["outer_h"]
    outer_w = spec["outer_w"]

    frame_bboxes_inner_to_outer = []
    for depth_from_outer in range(layers):
        bbox = (
            top + 2 * depth_from_outer,
            left + 2 * depth_from_outer,
            top + outer_h - 1 - 2 * depth_from_outer,
            left + outer_w - 1 - 2 * depth_from_outer,
        )
        color = frame_colors[layers - 1 - depth_from_outer]
        _paint_frame(grid, bbox, color)
        frame_bboxes_inner_to_outer.append(bbox)

    frame_bboxes_inner_to_outer.reverse()
    sr0 = top + 2 * layers
    sc0 = left + 2 * layers
    grid[sr0 : sr0 + inner_h, sc0 : sc0 + inner_w] = spec["solid_color"]

    meta = dict(spec)
    meta["outer_bbox"] = (top, left, top + outer_h - 1, left + outer_w - 1)
    meta["solid_bbox"] = (sr0, sc0, sr0 + inner_h - 1, sc0 + inner_w - 1)
    meta["frame_bboxes_inner_to_outer"] = frame_bboxes_inner_to_outer
    return meta


def _crop_used_area(grid: np.ndarray) -> np.ndarray:
    pts = np.argwhere(grid != BG)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    top_extra = random.randint(0, min(3, int(r0)))
    left_extra = random.randint(0, min(3, int(c0)))
    bottom_extra = random.randint(0, min(3, int(grid.shape[0] - 1 - r1)))
    right_extra = random.randint(0, min(3, int(grid.shape[1] - 1 - c1)))
    return grid[
        int(r0) - top_extra : int(r1) + bottom_extra + 1,
        int(c0) - left_extra : int(c1) + right_extra + 1,
    ]


def _gen_attempt(return_meta: bool = False):
    grid = np.zeros((MASTER_SIZE, MASTER_SIZE), dtype=int)
    blocked = np.zeros_like(grid, dtype=bool)
    n_groups = random.randint(1, 4)
    specs = _sample_group_specs(n_groups)
    specs.sort(key=lambda spec: spec["outer_h"] * spec["outer_w"], reverse=True)

    painted = []
    for spec in specs:
        pad = random.choice((1, 2))
        pos = _choose_position(blocked, spec["outer_h"], spec["outer_w"], pad)
        if pos is None:
            return None
        top, left = pos
        rr0 = max(0, top - pad)
        rr1 = min(MASTER_SIZE, top + spec["outer_h"] + pad)
        cc0 = max(0, left - pad)
        cc1 = min(MASTER_SIZE, left + spec["outer_w"] + pad)
        blocked[rr0:rr1, cc0:cc1] = True
        painted.append(_paint_group(grid, top, left, spec))

    cropped = _crop_used_area(grid)
    if return_meta:
        return cropped, painted
    return cropped


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frames input")
