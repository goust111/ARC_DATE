from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
OBJECT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
AXES = ("horizontal", "vertical")


def _same_color_components(grid: np.ndarray) -> list[dict]:
    H, W = grid.shape
    seen = np.zeros((H, W), dtype=bool)
    components: list[dict] = []
    for r in range(H):
        for c in range(W):
            color = int(grid[r, c])
            if color == BLACK or seen[r, c]:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(len(coords)),
                }
            )
    return components


def _assert_solid_rectangle(grid: np.ndarray, obj: dict) -> None:
    r0, c0, r1, c1 = obj["bbox"]
    color = int(obj["color"])
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert obj["area"] == patch.size, "foreground objects must be solid rectangles"
    assert np.all(patch == color), "foreground object bbox must contain only its color"


def _distance_to_centerline(lo: int, hi: int, length: int) -> float:
    center = (int(length) - 1) / 2.0
    if hi < center:
        return center - hi
    if lo > center:
        return lo - center
    return 0.0


def _nearest_center_axis(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> str:
    r0, c0, r1, c1 = bbox
    H, W = shape
    row_distance = _distance_to_centerline(r0, r1, H)
    col_distance = _distance_to_centerline(c0, c1, W)
    assert row_distance > 0 and col_distance > 0, "rectangles must stay on one side of both centerlines"
    assert row_distance != col_distance, "each rectangle must have a unique nearest centerline"
    return "horizontal" if row_distance < col_distance else "vertical"


def _mirror_bbox(
    bbox: tuple[int, int, int, int], shape: tuple[int, int], axis: str
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    H, W = shape
    if axis == "horizontal":
        return H - 1 - r1, c0, H - 1 - r0, c1
    return r0, W - 1 - c1, r1, W - 1 - c0


def _paint_gap_and_pair(
    out: np.ndarray,
    bbox: tuple[int, int, int, int],
    mirror: tuple[int, int, int, int],
    axis: str,
    color: int,
) -> None:
    r0, c0, r1, c1 = bbox
    mr0, mc0, mr1, mc1 = mirror

    if axis == "horizontal":
        if r1 < mr0:
            gr0, gr1 = r1 + 1, mr0 - 1
        else:
            assert mr1 < r0, "mirror must be on the opposite side of the horizontal centerline"
            gr0, gr1 = mr1 + 1, r0 - 1
        if gr0 <= gr1:
            out[gr0 : gr1 + 1, c0 : c1 + 1] = GREY
    else:
        if c1 < mc0:
            gc0, gc1 = c1 + 1, mc0 - 1
        else:
            assert mc1 < c0, "mirror must be on the opposite side of the vertical centerline"
            gc0, gc1 = mc1 + 1, c0 - 1
        if gc0 <= gc1:
            out[r0 : r1 + 1, gc0 : gc1 + 1] = GREY

    out[r0 : r1 + 1, c0 : c1 + 1] = color
    out[mr0 : mr1 + 1, mc0 : mc1 + 1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each solid rectangle across its nearest global centerline and fill the gap in grey.

    Input structure:
    - The grid background is black.
    - Several differently colored foreground objects are solid axis-aligned rectangles.
    - Each rectangle lies away from the horizontal and vertical centerlines so one
      of those full-grid virtual lines is uniquely nearest.

    Rule:
    - For every foreground rectangle, compare its distance to the horizontal
      centerline with its distance to the vertical centerline.
    - Reflect the rectangle across the nearer global centerline, preserving its
      color in both the original and mirrored positions.
    - Fill the axis-aligned strip strictly between the original rectangle and
      its mirrored copy with grey pixels.

    Key insight:
    - The mirror axis is not a property of the rectangle itself; it is whichever
      global grid centerline is closest to that rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be two-dimensional"
    assert not np.any(g == GREY), "grey appears only as the output gap color"

    objects = _same_color_components(g)
    assert objects, "expected at least one foreground rectangle"

    operations = []
    for obj in objects:
        _assert_solid_rectangle(g, obj)
        axis = _nearest_center_axis(obj["bbox"], g.shape)
        mirror = _mirror_bbox(obj["bbox"], g.shape, axis)
        operations.append((obj["bbox"], mirror, axis, int(obj["color"])))

    out = g.copy()
    for bbox, mirror, axis, color in operations:
        _paint_gap_and_pair(out, bbox, mirror, axis, color)
    return out


# --------------- generator ---------------

GRID_SIZE_RANGE = (14, 29)
OBJECT_COUNT_RANGE = (2, 4)
RECT_HEIGHT_RANGE = (2, 5)
RECT_WIDTH_RANGE = (2, 5)
PLACEMENT_TRIES = 500
GENERATION_ATTEMPTS = 300
RESERVE_PAD = 1


def _expanded_clear(reserved: np.ndarray, bbox: tuple[int, int, int, int], pad: int) -> bool:
    H, W = reserved.shape
    r0, c0, r1, c1 = bbox
    er0, ec0 = max(0, r0 - pad), max(0, c0 - pad)
    er1, ec1 = min(H - 1, r1 + pad), min(W - 1, c1 + pad)
    return not bool(np.any(reserved[er0 : er1 + 1, ec0 : ec1 + 1]))


def _mark_expanded(reserved: np.ndarray, bbox: tuple[int, int, int, int], pad: int) -> None:
    H, W = reserved.shape
    r0, c0, r1, c1 = bbox
    er0, ec0 = max(0, r0 - pad), max(0, c0 - pad)
    er1, ec1 = min(H - 1, r1 + pad), min(W - 1, c1 + pad)
    reserved[er0 : er1 + 1, ec0 : ec1 + 1] = True


def _affected_bbox(
    bbox: tuple[int, int, int, int], mirror: tuple[int, int, int, int], axis: str
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    mr0, mc0, mr1, mc1 = mirror
    if axis == "horizontal":
        return min(r0, mr0), c0, max(r1, mr1), c1
    return r0, min(c0, mc0), r1, max(c1, mc1)


def _axis_gap_cells(bbox: tuple[int, int, int, int], mirror: tuple[int, int, int, int], axis: str) -> int:
    r0, c0, r1, c1 = bbox
    mr0, mc0, mr1, mc1 = mirror
    if axis == "horizontal":
        return max(0, max(r0, mr0) - min(r1, mr1) - 1)
    return max(0, max(c0, mc0) - min(c1, mc1) - 1)


def _sample_near_center_span(length: int, size: int, side: int) -> tuple[int, int] | None:
    center = (length - 1) / 2.0
    if side < 0:
        ends = [hi for hi in range(size - 1, length) if hi < center and length - 1 - hi > hi]
        if not ends:
            return None
        near = sorted(ends, reverse=True)[: min(5, len(ends))]
        hi = random.choice(near)
        return hi - size + 1, hi

    starts = [lo for lo in range(0, length - size + 1) if lo > center and length - 1 - lo < lo]
    if not starts:
        return None
    near = sorted(starts)[: min(5, len(starts))]
    lo = random.choice(near)
    return lo, lo + size - 1


def _random_far_span(length: int, size: int) -> tuple[int, int] | None:
    center = (length - 1) / 2.0
    side = random.choice((-1, 1))
    if side < 0:
        starts = [lo for lo in range(0, length - size + 1) if lo + size - 1 < center]
        if not starts:
            return None
        lo = random.choice(starts[: max(1, len(starts) * 2 // 3)])
    else:
        starts = [lo for lo in range(0, length - size + 1) if lo > center]
        if not starts:
            return None
        lo = random.choice(starts[-max(1, len(starts) * 2 // 3) :])
    return lo, lo + size - 1


def _sample_bbox(shape: tuple[int, int], desired_axis: str) -> tuple[int, int, int, int] | None:
    H, W = shape
    rh = random.randint(*RECT_HEIGHT_RANGE)
    rw = random.randint(*RECT_WIDTH_RANGE)
    side = random.choice((-1, 1))

    if desired_axis == "horizontal":
        row_span = _sample_near_center_span(H, rh, side)
        col_span = _random_far_span(W, rw)
    else:
        row_span = _random_far_span(H, rh)
        col_span = _sample_near_center_span(W, rw, side)

    if row_span is None or col_span is None:
        return None
    r0, r1 = row_span
    c0, c1 = col_span
    return r0, c0, r1, c1


def _place_object(
    grid: np.ndarray, reserved: np.ndarray, desired_axis: str, color: int
) -> bool:
    for _ in range(PLACEMENT_TRIES):
        bbox = _sample_bbox(grid.shape, desired_axis)
        if bbox is None:
            continue
        try:
            axis = _nearest_center_axis(bbox, grid.shape)
        except AssertionError:
            continue
        if axis != desired_axis:
            continue

        mirror = _mirror_bbox(bbox, grid.shape, axis)
        if _axis_gap_cells(bbox, mirror, axis) < 1:
            continue
        affected = _affected_bbox(bbox, mirror, axis)
        if not _expanded_clear(reserved, affected, RESERVE_PAD):
            continue

        r0, c0, r1, c1 = bbox
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)
        _mark_expanded(reserved, affected, RESERVE_PAD)
        return True
    return False


def _axis_plan(count: int) -> list[str]:
    axes = ["horizontal", "vertical"]
    while len(axes) < count:
        axes.append(random.choice(AXES))
    random.shuffle(axes)
    return axes


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_SIZE_RANGE)
    W = random.randint(*GRID_SIZE_RANGE)
    count = random.randint(*OBJECT_COUNT_RANGE)
    colors = random.sample(OBJECT_COLORS, count)
    axes = _axis_plan(count)

    grid = np.full((H, W), BLACK, dtype=int)
    reserved = np.zeros((H, W), dtype=bool)
    for axis, color in zip(axes, colors):
        if not _place_object(grid, reserved, axis, color):
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if not np.any(out == GREY):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
