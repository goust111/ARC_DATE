import random

import numpy as np


BLACK = 0
GREY = 5
MAGENTA = 6
FRAME_COLORS = (1, 2, 3, 4, 7, 8, 9)
MAX_ATTEMPTS = 400


def _bbox_of_color(grid: np.ndarray, color: int) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == color)
    assert coords.size > 0, f"missing color {color}"
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _is_hollow_frame(grid: np.ndarray, color: int) -> bool:
    r0, c0, r1, c1 = _bbox_of_color(grid, color)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False

    color_mask = grid == color
    if not np.all(grid[r0, c0 : c1 + 1] == color):
        return False
    if not np.all(grid[r1, c0 : c1 + 1] == color):
        return False
    if not np.all(grid[r0 : r1 + 1, c0] == color):
        return False
    if not np.all(grid[r0 : r1 + 1, c1] == color):
        return False
    if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] == color):
        return False

    expected = np.zeros_like(color_mask)
    expected[r0, c0 : c1 + 1] = True
    expected[r1, c0 : c1 + 1] = True
    expected[r0 : r1 + 1, c0] = True
    expected[r0 : r1 + 1, c1] = True
    return bool(np.array_equal(color_mask, expected))


def _count_corner_marker(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> int:
    r0, c0, _r1, _c1 = bbox
    horizontal = 0
    vertical = 0

    if r0 > 0:
        c = c0 - 1
        while c >= 0 and int(grid[r0 - 1, c]) == MAGENTA:
            horizontal += 1
            c -= 1
    if c0 > 0:
        r = r0 - 1
        while r >= 0 and int(grid[r, c0 - 1]) == MAGENTA:
            vertical += 1
            r -= 1

    assert horizontal > 0 or vertical > 0, "missing corner marker"
    assert min(horizontal, vertical) <= 1, "marker must be a straight row or column"
    count = int(max(horizontal, vertical))
    assert 1 <= count <= 5, "marker count must stay in the visible 1..5 range"
    return count


def _find_anchor(grid: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(grid == GREY)
    assert len(coords) == 1, "expected a unique grey anchor"
    return int(coords[0, 0]), int(coords[0, 1])


def _draw_square_frame(grid: np.ndarray, center: tuple[int, int], radius: int, color: int) -> None:
    row, col = center
    top = row - radius
    bottom = row + radius
    left = col - radius
    right = col + radius
    assert top >= 0 and left >= 0 and bottom < grid.shape[0] and right < grid.shape[1]
    grid[top, left : right + 1] = color
    grid[bottom, left : right + 1] = color
    grid[top : bottom + 1, left] = color
    grid[top : bottom + 1, right] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rearrange scattered colored frames into concentric nested frames around the grey anchor.

    Input structure:
    - The background is black.
    - Several non-overlapping hollow rectangular frames appear in distinct non-magenta,
      non-grey colors, each with a black interior.
    - Near each frame's top-left outside corner there is a short magenta marker line,
      horizontal or vertical, and a single grey anchor pixel sits in the lower-right area.

    Rule:
    - For each frame, count the magenta marker pixels near its top-left corner.
    - Sort frames by that count in ascending order so fewer markers become outer layers;
      if multiple frames have the same count, sort those tied frames by original rectangle
      area from large to small.
    - Clear the original frames and markers, keep the grey anchor, and redraw the frame
      colors as concentric square borders centered on the anchor, assigning the sorted
      frames from the outside inward with one-cell shrink per side at each step.

    Key insight:
    - The magenta dots encode nesting order rather than color identity, and ties are
      resolved by the original frame size before the scene is rebuilt around the anchor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    anchor = _find_anchor(g)
    colors = [int(color) for color in np.unique(g) if color not in (BLACK, GREY, MAGENTA)]
    frames = []
    for color in colors:
        assert _is_hollow_frame(g, color), f"color {color} is not a hollow frame"
        bbox = _bbox_of_color(g, color)
        marker_count = _count_corner_marker(g, bbox)
        r0, c0, r1, c1 = bbox
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        frames.append(
            {
                "color": color,
                "bbox": bbox,
                "marker_count": marker_count,
                "area": int(area),
            }
        )

    assert frames, "expected at least one frame"
    ordered = sorted(frames, key=lambda frame: (frame["marker_count"], -frame["area"]))

    out = np.full_like(g, BLACK)
    out[anchor] = GREY
    total = len(ordered)
    for idx, frame in enumerate(ordered):
        radius = total - idx
        _draw_square_frame(out, anchor, radius, int(frame["color"]))

    return out


def _sample_level_counts() -> list[int]:
    has_tie = random.random() < 0.5
    if not has_tie:
        frame_count = random.randint(3, 5)
        return list(range(1, frame_count + 1))

    frame_count = random.randint(4, 7)
    group_count = random.randint(2, min(5, frame_count - 1))
    group_sizes = [1] * group_count
    remaining = frame_count - group_count
    while remaining > 0:
        idx = random.randrange(group_count)
        group_sizes[idx] += 1
        remaining -= 1

    counts: list[int] = []
    for depth, size in enumerate(group_sizes, start=1):
        counts.extend([depth] * size)
    return counts


def _sample_frame_specs(level_counts: list[int]) -> list[dict]:
    size_pool = [
        (h, w)
        for h in range(3, 8)
        for w in range(3, 8)
        if h * w not in {4, 8}
    ]
    random.shuffle(size_pool)

    used_areas: set[int] = set()
    specs = []
    for count in level_counts:
        chosen = None
        for h, w in size_pool:
            area = h * w
            if area in used_areas:
                continue
            chosen = (h, w, area)
            used_areas.add(area)
            break
        assert chosen is not None, "failed to sample enough unique frame areas"
        h, w, area = chosen
        specs.append(
            {
                "marker_count": int(count),
                "height": int(h),
                "width": int(w),
                "area": int(area),
                "orientation": random.choice(("horizontal", "vertical")),
            }
        )
        size_pool.remove((h, w))

    random.shuffle(specs)
    return specs


def _expanded_box(
    r0: int,
    c0: int,
    h: int,
    w: int,
    orientation: str,
    marker_count: int,
    grid_shape: tuple[int, int],
    padding: int = 1,
) -> tuple[int, int, int, int]:
    if orientation == "horizontal":
        mr0 = r0 - 1
        mc0 = c0 - marker_count
        mr1 = r0 - 1
        mc1 = c0 - 1
    else:
        mr0 = r0 - marker_count
        mc0 = c0 - 1
        mr1 = r0 - 1
        mc1 = c0 - 1

    union_r0 = min(r0, mr0)
    union_c0 = min(c0, mc0)
    union_r1 = max(r0 + h - 1, mr1)
    union_c1 = max(c0 + w - 1, mc1)

    top = max(0, union_r0 - padding)
    left = max(0, union_c0 - padding)
    bottom = min(grid_shape[0] - 1, union_r1 + padding)
    right = min(grid_shape[1] - 1, union_c1 + padding)
    return top, left, bottom, right


def _can_place(
    blocked: np.ndarray,
    r0: int,
    c0: int,
    h: int,
    w: int,
    orientation: str,
    marker_count: int,
) -> bool:
    if orientation == "horizontal":
        if r0 < 1 or c0 < marker_count:
            return False
        marker_cells = [(r0 - 1, c) for c in range(c0 - marker_count, c0)]
    else:
        if r0 < marker_count or c0 < 1:
            return False
        marker_cells = [(r, c0 - 1) for r in range(r0 - marker_count, r0)]

    if r0 + h > blocked.shape[0] or c0 + w > blocked.shape[1]:
        return False

    top, left, bottom, right = _expanded_box(
        r0, c0, h, w, orientation, marker_count, blocked.shape
    )
    if np.any(blocked[top : bottom + 1, left : right + 1]):
        return False

    for rr, cc in marker_cells:
        if not (0 <= rr < blocked.shape[0] and 0 <= cc < blocked.shape[1]):
            return False
    return True


def _place_frame(
    grid: np.ndarray,
    blocked: np.ndarray,
    color: int,
    marker_count: int,
    height: int,
    width: int,
    orientation: str,
) -> bool:
    row_positions = list(range(0, grid.shape[0] - height + 1))
    col_positions = list(range(0, grid.shape[1] - width + 1))
    random.shuffle(row_positions)
    random.shuffle(col_positions)

    for r0 in row_positions:
        for c0 in col_positions:
            if not _can_place(blocked, r0, c0, height, width, orientation, marker_count):
                continue

            r1 = r0 + height - 1
            c1 = c0 + width - 1
            grid[r0, c0 : c1 + 1] = color
            grid[r1, c0 : c1 + 1] = color
            grid[r0 : r1 + 1, c0] = color
            grid[r0 : r1 + 1, c1] = color

            if orientation == "horizontal":
                grid[r0 - 1, c0 - marker_count : c0] = MAGENTA
            else:
                grid[r0 - marker_count : r0, c0 - 1] = MAGENTA

            top, left, bottom, right = _expanded_box(
                r0, c0, height, width, orientation, marker_count, grid.shape
            )
            blocked[top : bottom + 1, left : right + 1] = True
            return True
    return False


def _reserve_anchor_zone(blocked: np.ndarray, anchor: tuple[int, int], radius: int) -> None:
    row, col = anchor
    top = max(0, row - radius - 1)
    left = max(0, col - radius - 1)
    bottom = min(blocked.shape[0] - 1, row + radius + 1)
    right = min(blocked.shape[1] - 1, col + radius + 1)
    blocked[top : bottom + 1, left : right + 1] = True


def _make_input() -> np.ndarray | None:
    level_counts = _sample_level_counts()
    frame_count = len(level_counts)

    height = random.randint(max(18, 2 * frame_count + 7), 30)
    width = random.randint(max(18, 2 * frame_count + 7), 30)
    grid = np.full((height, width), BLACK, dtype=int)
    blocked = np.zeros_like(grid, dtype=bool)

    row_lo = max(frame_count + 1, height // 2)
    row_hi = height - frame_count - 2
    col_lo = max(frame_count + 1, width // 2)
    col_hi = width - frame_count - 2
    if row_lo > row_hi or col_lo > col_hi:
        return None

    anchor = (random.randint(row_lo, row_hi), random.randint(col_lo, col_hi))
    grid[anchor] = GREY
    _reserve_anchor_zone(blocked, anchor, frame_count)

    specs = _sample_frame_specs(level_counts)
    colors = random.sample(FRAME_COLORS, frame_count)
    for spec, color in zip(
        sorted(specs, key=lambda spec: -(spec["height"] * spec["width"])),
        colors,
    ):
        ok = _place_frame(
            grid,
            blocked,
            color=color,
            marker_count=int(spec["marker_count"]),
            height=int(spec["height"]),
            width=int(spec["width"]),
            orientation=str(spec["orientation"]),
        )
        if not ok:
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _make_input()
        if grid is None:
            continue
        try:
            out = main(grid)
        except AssertionError:
            continue
        if out.ndim == 2 and max(grid.shape) <= 30 and max(out.shape) <= 30:
            return grid
    raise RuntimeError("failed to generate a valid input")
