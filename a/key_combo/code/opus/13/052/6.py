import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

FRAME_COLORS = (BLUE, RED, GREEN, YELLOW)
DIRS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}

MAX_GENERATION_ATTEMPTS = 500
NESTED_CASES = [
    (1, 1),
    (1, 2),
    (1, 3),
    (2, 1),
]
NESTED_CASE_WEIGHTS = np.array([0.24, 0.28, 0.18, 0.30], dtype=float)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


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


def _interior_cells(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]


def _bbox_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return outer[0] < inner[0] and outer[1] < inner[1] and outer[2] > inner[2] and outer[3] > inner[3]


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _bbox_expand(bbox: tuple[int, int, int, int], pad: int) -> tuple[int, int, int, int]:
    return bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    return (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)


def _in_bbox(cell: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r, c = cell
    return bbox[0] <= r <= bbox[2] and bbox[1] <= c <= bbox[3]


def _ray_cells(center: tuple[int, int], direction: str, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    dr, dc = DIRS[direction]
    r, c = center
    ray = []
    while 0 <= r < h and 0 <= c < w:
        ray.append((r, c))
        r += dr
        c += dc
    return ray


def _nearest_edge(center: tuple[int, int], shape: tuple[int, int]) -> tuple[str, dict[str, int]]:
    h, w = shape
    r, c = center
    dists = {
        "U": r,
        "D": h - 1 - r,
        "L": c,
        "R": w - 1 - c,
    }
    best = min(dists.values())
    winners = [direction for direction, dist in dists.items() if dist == best]
    return winners[0], dists


def _find_frames(grid: np.ndarray) -> list[dict]:
    frames = []
    for color in FRAME_COLORS:
        for coords in _components(grid == color):
            bbox = _bbox_coords(coords)
            r0, c0, r1, c1 = bbox
            if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
                continue
            cells = {(int(r), int(c)) for r, c in coords}
            border = _frame_cells(bbox)
            if cells != border:
                continue
            frames.append(
                {
                    "color": int(color),
                    "bbox": bbox,
                    "cells": border,
                    "center": ((r0 + r1) // 2, (c0 + c1) // 2),
                }
            )
    return frames


def _nested_pairs(frames: list[dict]) -> list[dict]:
    pairs = []
    for frame in frames:
        containers = [other for other in frames if other is not frame and _bbox_contains(other["bbox"], frame["bbox"])]
        if not containers:
            continue
        outer = min(containers, key=lambda item: _bbox_area(item["bbox"]))
        pairs.append({"inner": frame, "outer": outer})
    return pairs


def _draw_ejection(
    out: np.ndarray,
    center: tuple[int, int],
    direction: str,
    count: int,
    protected: set[tuple[int, int]],
) -> None:
    h, w = out.shape
    dr, dc = DIRS[direction]
    r, c = center
    placed = 0
    while 0 <= r < h and 0 <= c < w and placed < count:
        if (r, c) not in protected:
            out[r, c] = GREY
            placed += 1
        r += dr
        c += dc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Eject each nested frame's grey payload toward its unique nearest canvas edge while preserving all frame borders.

    Input structure:
    - The grid has black background and several single-pixel-thick hollow rectangular frame components in blue, red, green, or yellow.
    - Some frames are strictly nested inside larger rectangular frames; only those innermost nested frames contain scattered grey pixels in their interiors.
    - Independent frames and outer shell frames contain no grey payload.

    Rule:
    - Detect every rectangular frame component and classify a frame as an inner frame when its bounding box is strictly contained in another frame's bounding box.
    - For each inner frame, count and erase all grey interior pixels.
    - Take the floored center of that inner frame's bounding box, choose the uniquely nearest canvas edge, and draw a grey ray from the center in that direction using exactly the erased grey count.
    - While drawing, any occupied frame-border cell is protected: skip over it without recoloring it and without consuming ray length; keep all other cells unchanged.

    Key insight:
    - The grey count belongs to the innermost frame, but the launch direction comes from the inner frame center's distance to the outer canvas, so containment, counting, and protected border skipping all matter.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    frames = _find_frames(g)
    pairs = _nested_pairs(frames)
    protected = set().union(*(frame["cells"] for frame in frames)) if frames else set()

    for pair in pairs:
        inner = pair["inner"]
        interior = _interior_cells(inner["bbox"])
        grey_cells = [(r, c) for r, c in interior if int(g[r, c]) == GREY]
        if not grey_cells:
            continue
        for r, c in grey_cells:
            out[r, c] = BLACK
        direction, _ = _nearest_edge(inner["center"], g.shape)
        _draw_ejection(out, inner["center"], direction, len(grey_cells), protected)

    return out


def _pick_case(rng: np.random.Generator) -> tuple[int, int]:
    idx = int(rng.choice(len(NESTED_CASES), p=NESTED_CASE_WEIGHTS))
    return NESTED_CASES[idx]


def _shape_for_case(rng: np.random.Generator, nested_count: int, independent_count: int) -> tuple[int, int]:
    base = 18 if nested_count == 2 or independent_count >= 2 else 16
    h = int(rng.integers(base, base + 5))
    w = int(rng.integers(base, base + 5))
    return h, w


def _pick_frame_color(rng: np.random.Generator, used: list[int], forbidden: set[int] | None = None) -> int:
    forbidden = forbidden or set()
    options = [color for color in FRAME_COLORS if color not in forbidden]
    unused = [color for color in options if color not in used]
    pool = unused if unused else options
    color = int(rng.choice(pool))
    used.append(color)
    return color


def _make_inner_bbox(center: tuple[int, int], height: int, width: int) -> tuple[int, int, int, int]:
    r, c = center
    r0 = r - (height - 1) // 2
    c0 = c - (width - 1) // 2
    return r0, c0, r0 + height - 1, c0 + width - 1


def _bbox_in_bounds(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> bool:
    h, w = shape
    return 0 <= bbox[0] <= bbox[2] < h and 0 <= bbox[1] <= bbox[3] < w


def _place_nested_pair(
    rng: np.random.Generator,
    shape: tuple[int, int],
    target_direction: str,
    blocked_boxes: list[tuple[int, int, int, int]],
    blocked_rays: set[tuple[int, int]],
    used_colors: list[int],
) -> dict | None:
    h, w = shape
    for _ in range(180):
        inner_h = int(rng.integers(4, 8))
        inner_w = int(rng.integers(4, 8))
        if (inner_h - 2) * (inner_w - 2) < 5:
            continue
        top_gap = int(rng.integers(2, 5))
        bottom_gap = int(rng.integers(2, 5))
        left_gap = int(rng.integers(2, 5))
        right_gap = int(rng.integers(2, 5))
        outer_h = inner_h + top_gap + bottom_gap
        outer_w = inner_w + left_gap + right_gap
        if outer_h >= h - 1 or outer_w >= w - 1:
            continue

        for _ in range(140):
            center = (int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1)))
            inner_bbox = _make_inner_bbox(center, inner_h, inner_w)
            outer_bbox = (
                inner_bbox[0] - top_gap,
                inner_bbox[1] - left_gap,
                inner_bbox[2] + bottom_gap,
                inner_bbox[3] + right_gap,
            )
            if not _bbox_in_bounds(inner_bbox, shape) or not _bbox_in_bounds(outer_bbox, shape):
                continue
            direction, dists = _nearest_edge(center, shape)
            if direction != target_direction:
                continue
            if list(dists.values()).count(min(dists.values())) != 1:
                continue
            padded = _bbox_expand(outer_bbox, 1)
            if any(_bbox_overlap(padded, box) for box in blocked_boxes):
                continue

            ray = _ray_cells(center, direction, shape)
            if any(_in_bbox(cell, box) for cell in ray for box in blocked_boxes):
                continue
            if any(cell in blocked_rays for cell in ray):
                continue

            inner_frame = _frame_cells(inner_bbox)
            outer_frame = _frame_cells(outer_bbox)
            protected_hits = sum(cell in inner_frame or cell in outer_frame for cell in ray)
            if protected_hits < 2:
                continue

            drawable = [cell for cell in ray if cell not in inner_frame and cell not in outer_frame]
            k = int(rng.integers(2, min(5, len(_interior_cells(inner_bbox))) + 1))
            if len(drawable) < k:
                continue

            interior = _interior_cells(inner_bbox)
            grey_idx = rng.choice(len(interior), size=k, replace=False)
            grey_cells = [interior[int(i)] for i in grey_idx]

            outer_color = _pick_frame_color(rng, used_colors)
            inner_color = _pick_frame_color(rng, used_colors, forbidden={outer_color})
            return {
                "outer": {"bbox": outer_bbox, "color": outer_color, "cells": outer_frame, "kind": "outer"},
                "inner": {
                    "bbox": inner_bbox,
                    "color": inner_color,
                    "cells": inner_frame,
                    "kind": "inner",
                    "center": center,
                    "direction": direction,
                    "grey_cells": grey_cells,
                    "k": k,
                    "ray": ray,
                },
            }
    return None


def _place_independent_frame(
    rng: np.random.Generator,
    shape: tuple[int, int],
    blocked_boxes: list[tuple[int, int, int, int]],
    blocked_rays: set[tuple[int, int]],
    used_colors: list[int],
) -> dict | None:
    h, w = shape
    for _ in range(180):
        fh = int(rng.integers(4, 9))
        fw = int(rng.integers(4, 9))
        r0 = int(rng.integers(0, h - fh + 1))
        c0 = int(rng.integers(0, w - fw + 1))
        bbox = (r0, c0, r0 + fh - 1, c0 + fw - 1)
        padded = _bbox_expand(bbox, 1)
        if any(_bbox_overlap(padded, box) for box in blocked_boxes):
            continue
        if any(_in_bbox(cell, padded) for cell in blocked_rays):
            continue
        color = _pick_frame_color(rng, used_colors)
        return {"bbox": bbox, "color": color, "cells": _frame_cells(bbox), "kind": "independent"}
    return None


def _render_input(shape: tuple[int, int], frames: list[dict], inners: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for frame in frames:
        for r, c in frame["cells"]:
            grid[r, c] = frame["color"]
    for inner in inners:
        for r, c in inner["grey_cells"]:
            grid[r, c] = GREY
    return grid


def _expected_from_scene(shape: tuple[int, int], frames: list[dict], inners: list[dict]) -> np.ndarray:
    out = _render_input(shape, frames, inners)
    protected = set().union(*(frame["cells"] for frame in frames))
    for inner in inners:
        for r, c in inner["grey_cells"]:
            out[r, c] = BLACK
        _draw_ejection(out, inner["center"], inner["direction"], inner["k"], protected)
    return out


def _validate_scene(shape: tuple[int, int], frames: list[dict], inners: list[dict], inp: np.ndarray, out: np.ndarray) -> bool:
    if np.array_equal(inp, out):
        return False
    detected_frames = _find_frames(inp)
    if len(detected_frames) != len(frames):
        return False
    detected_pairs = _nested_pairs(detected_frames)
    if len(detected_pairs) != len(inners):
        return False

    planned_frame_cells = {cell for frame in frames for cell in frame["cells"]}
    grey_in = int((inp == GREY).sum())
    grey_out = int((out == GREY).sum())
    if grey_in != sum(inner["k"] for inner in inners):
        return False
    if grey_out != grey_in:
        return False

    planned_output_greys = set()
    for inner in inners:
        direction, dists = _nearest_edge(inner["center"], shape)
        if direction != inner["direction"]:
            return False
        if list(dists.values()).count(min(dists.values())) != 1:
            return False
        if any(_in_bbox(cell, other["bbox"]) for other in inners if other is not inner for cell in inner["ray"]):
            return False
        if sum(cell in planned_frame_cells for cell in inner["ray"]) < 2:
            return False

        ray_positions = []
        for cell in inner["ray"]:
            if cell not in planned_frame_cells:
                ray_positions.append(cell)
            if len(ray_positions) == inner["k"]:
                break
        if len(ray_positions) != inner["k"]:
            return False
        if planned_output_greys.intersection(ray_positions):
            return False
        planned_output_greys.update(ray_positions)

    if planned_output_greys != {tuple(cell) for cell in np.argwhere(out == GREY)}:
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    nested_count, independent_count = _pick_case(rng)
    shape = _shape_for_case(rng, nested_count, independent_count)

    frames: list[dict] = []
    inners: list[dict] = []
    blocked_boxes: list[tuple[int, int, int, int]] = []
    blocked_rays: set[tuple[int, int]] = set()
    used_colors: list[int] = []

    if nested_count == 1:
        directions = [str(rng.choice(list(DIRS.keys())))]
    else:
        directions = list(rng.choice(list(DIRS.keys()), size=nested_count, replace=False))

    for direction in directions:
        pair = _place_nested_pair(rng, shape, direction, blocked_boxes, blocked_rays, used_colors)
        if pair is None:
            return None
        frames.extend([pair["outer"], pair["inner"]])
        inners.append(pair["inner"])
        blocked_boxes.append(_bbox_expand(pair["outer"]["bbox"], 1))
        blocked_rays.update(pair["inner"]["ray"])

    for _ in range(independent_count):
        frame = _place_independent_frame(rng, shape, blocked_boxes, blocked_rays, used_colors)
        if frame is None:
            return None
        frames.append(frame)
        blocked_boxes.append(_bbox_expand(frame["bbox"], 1))

    inp = _render_input(shape, frames, inners)
    expected = _expected_from_scene(shape, frames, inners)
    if not _validate_scene(shape, frames, inners, inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame ejection input")
