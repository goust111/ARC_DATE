import random
from collections import deque

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    counts = np.bincount(grid.reshape(-1), minlength=10)
    return int(np.argmax(counts))


def _border_cells(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells = [(r0, c) for c in range(c0, c1 + 1)]
    if r1 != r0:
        cells.extend((r1, c) for c in range(c0, c1 + 1))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        if c1 != c0:
            cells.append((r, c1))
    return cells


def _border_set(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    return set(_border_cells(bbox))


def _corners(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]


def _paint_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _infer_frames(grid: np.ndarray, bg: int) -> dict[int, dict[str, int | tuple[int, int, int, int]]]:
    frames: dict[int, dict[str, int | tuple[int, int, int, int]]] = {}
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        pts = np.argwhere(grid == color)
        assert pts.size > 0, "each frame color must be visible"
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        assert h >= 3 and w >= 3, "inferred frame bbox must be at least 3x3"
        frames[color] = {
            "bbox": (r0, c0, r1, c1),
            "height": h,
            "width": w,
        }
    assert frames, "expected at least one frame"
    return frames


def _occlusion_graph(
    grid: np.ndarray,
    frames: dict[int, dict[str, int | tuple[int, int, int, int]]],
    bg: int,
) -> dict[int, set[int]]:
    graph = {color: set() for color in frames}
    for color, info in frames.items():
        bbox = info["bbox"]
        assert isinstance(bbox, tuple)
        for r, c in _border_cells(bbox):
            seen = int(grid[r, c])
            assert seen != bg, "frame border cannot reveal background"
            if seen != color:
                assert seen in frames, "all non-background colors must be frames"
                graph[color].add(seen)
    return graph


def _unique_topological_order(graph: dict[int, set[int]]) -> list[int]:
    indegree = {node: 0 for node in graph}
    for nbrs in graph.values():
        for nbr in nbrs:
            indegree[nbr] += 1

    queue = deque(node for node, deg in indegree.items() if deg == 0)
    order: list[int] = []
    while queue:
        assert len(queue) == 1, "layering order must be uniquely determined"
        node = queue.popleft()
        order.append(node)
        for nbr in sorted(graph[node]):
            indegree[nbr] -= 1
            if indegree[nbr] == 0:
                queue.append(nbr)

    assert len(order) == len(graph), "occlusion graph must be acyclic"
    return order


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    frames = _infer_frames(g, bg)
    graph = _occlusion_graph(g, frames, bg)
    order = _unique_topological_order(graph)

    out = np.full((len(order), 3), bg, dtype=int)
    for r, color in enumerate(order):
        width = int(frames[color]["width"])
        height = int(frames[color]["height"])
        if width >= height:
            out[r, :] = color
        else:
            out[r, 1] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect overlapping rectangular frames, recover their bottom-to-top order,
    and summarize each frame as one row in a 3-column output.
    """
    return _solve(grid)


GRID_H_RANGE = (20, 30)
GRID_W_RANGE = (20, 30)
FRAME_COUNT_RANGE = (3, 5)
FRAME_MIN_SIZE = 3
FRAME_MAX_HEIGHT = 10
FRAME_MAX_WIDTH = 12
FRAME_TRIES = 200
SCENE_TRIES = 400


def _sample_dims(
    rng: random.Random,
    orientation: str,
    max_h: int,
    max_w: int,
    min_h: int = FRAME_MIN_SIZE,
    min_w: int = FRAME_MIN_SIZE,
) -> tuple[int, int] | None:
    max_h = min(FRAME_MAX_HEIGHT, max_h)
    max_w = min(FRAME_MAX_WIDTH, max_w)
    if max_h < min_h or max_w < min_w:
        return None

    for _ in range(50):
        if orientation == "wide":
            h = rng.randint(min_h, max_h)
            lo_w = max(min_w, h)
            if lo_w > max_w:
                continue
            w = rng.randint(lo_w, max_w)
            return h, w

        w = rng.randint(min_w, max_w)
        lo_h = max(min_h, w + 1)
        if lo_h > max_h:
            continue
        h = rng.randint(lo_h, max_h)
        return h, w

    return None


def _sample_initial_bbox(
    rng: random.Random, grid_h: int, grid_w: int, orientation: str
) -> tuple[int, int, int, int] | None:
    dims = _sample_dims(rng, orientation, grid_h - 4, grid_w - 4)
    if dims is None:
        return None
    h, w = dims

    top_margin = 2 if grid_h - h >= 5 else 0
    left_margin = 2 if grid_w - w >= 5 else 0
    r0 = rng.randint(top_margin, grid_h - h - top_margin)
    c0 = rng.randint(left_margin, grid_w - w - left_margin)
    return (r0, c0, r0 + h - 1, c0 + w - 1)


def _anchor_candidates(
    bbox: tuple[int, int, int, int], protected: set[tuple[int, int]]
) -> list[tuple[tuple[int, int], str]]:
    r0, c0, r1, c1 = bbox
    candidates: list[tuple[tuple[int, int], str]] = []
    for c in range(c0 + 1, c1):
        if (r0, c) not in protected:
            candidates.append(((r0, c), "top"))
        if (r1, c) not in protected:
            candidates.append(((r1, c), "bottom"))
    for r in range(r0 + 1, r1):
        if (r, c0) not in protected:
            candidates.append(((r, c0), "left"))
        if (r, c1) not in protected:
            candidates.append(((r, c1), "right"))
    return candidates


def _bbox_from_template(
    anchor: tuple[int, int], template: str, dims: tuple[int, int]
) -> tuple[int, int, int, int]:
    ar, ac = anchor
    h, w = dims
    if template == "up_left":
        return (ar - h + 1, ac - w + 1, ar, ac)
    if template == "up_right":
        return (ar - h + 1, ac, ar, ac + w - 1)
    if template == "down_left":
        return (ar, ac - w + 1, ar + h - 1, ac)
    assert template == "down_right"
    return (ar, ac, ar + h - 1, ac + w - 1)


def _bbox_inside_grid(bbox: tuple[int, int, int, int], grid_h: int, grid_w: int) -> bool:
    r0, c0, r1, c1 = bbox
    return 0 <= r0 < r1 < grid_h and 0 <= c0 < c1 < grid_w


def _expected_output(
    frames: list[dict[str, int | tuple[int, int, int, int]]], bg: int
) -> np.ndarray:
    out = np.full((len(frames), 3), bg, dtype=int)
    for r, frame in enumerate(frames):
        color = int(frame["color"])
        height = int(frame["bbox"][2] - frame["bbox"][0] + 1)  # type: ignore[index]
        width = int(frame["bbox"][3] - frame["bbox"][1] + 1)   # type: ignore[index]
        if width >= height:
            out[r, :] = color
        else:
            out[r, 1] = color
    return out


def _render_scene(
    grid_h: int, grid_w: int, bg: int, frames: list[dict[str, int | tuple[int, int, int, int]]]
) -> np.ndarray:
    grid = np.full((grid_h, grid_w), bg, dtype=int)
    for frame in frames:
        _paint_frame(grid, frame["bbox"], int(frame["color"]))  # type: ignore[arg-type]
    return grid


def _visible_bbox(grid: np.ndarray, color: int) -> tuple[int, int, int, int]:
    pts = np.argwhere(grid == color)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return tuple(map(int, (r0, c0, r1, c1)))


def _scene_is_valid(
    grid: np.ndarray,
    bg: int,
    frames: list[dict[str, int | tuple[int, int, int, int]]],
) -> bool:
    if _get_bg(grid) != bg:
        return False

    for frame in frames:
        color = int(frame["color"])
        bbox = frame["bbox"]
        if _visible_bbox(grid, color) != bbox:
            return False

    expected = _expected_output(frames, bg)
    try:
        actual = _solve(grid)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def _make_frame(
    rng: random.Random,
    grid_h: int,
    grid_w: int,
    orientation: str,
    prev_frame: dict[str, int | tuple[int, int, int, int]],
    older_frames: list[dict[str, int | tuple[int, int, int, int]]],
    protected: set[tuple[int, int]],
) -> tuple[dict[str, int | tuple[int, int, int, int]], set[tuple[int, int]]] | None:
    prev_bbox = prev_frame["bbox"]
    assert isinstance(prev_bbox, tuple)
    prev_border = _border_set(prev_bbox)
    older_borders = []
    for frame in older_frames:
        bbox = frame["bbox"]
        assert isinstance(bbox, tuple)
        older_borders.append(_border_set(bbox))

    candidates = _anchor_candidates(prev_bbox, protected)
    if not candidates:
        return None

    for _ in range(FRAME_TRIES):
        anchor, side = rng.choice(candidates)
        ar, ac = anchor
        r0, c0, r1, c1 = prev_bbox

        options: list[tuple[str, int, int, int, int]] = []
        if side == "top":
            options.append(("up_left", FRAME_MIN_SIZE, ac - c0 + 2, ar + 1, ac + 1))
            options.append(("up_right", FRAME_MIN_SIZE, c1 - ac + 2, ar + 1, grid_w - ac))
        elif side == "bottom":
            options.append(("down_left", FRAME_MIN_SIZE, ac - c0 + 2, grid_h - ar, ac + 1))
            options.append(("down_right", FRAME_MIN_SIZE, c1 - ac + 2, grid_h - ar, grid_w - ac))
        elif side == "left":
            options.append(("up_left", ar - r0 + 2, FRAME_MIN_SIZE, ar + 1, ac + 1))
            options.append(("down_left", r1 - ar + 2, FRAME_MIN_SIZE, grid_h - ar, ac + 1))
        else:
            options.append(("up_right", ar - r0 + 2, FRAME_MIN_SIZE, ar + 1, grid_w - ac))
            options.append(("down_right", r1 - ar + 2, FRAME_MIN_SIZE, grid_h - ar, grid_w - ac))

        template, min_h, min_w, max_h, max_w = rng.choice(options)
        dims = _sample_dims(
            rng,
            orientation,
            max_h=max_h,
            max_w=max_w,
            min_h=min_h,
            min_w=min_w,
        )
        if dims is None:
            continue

        bbox = _bbox_from_template(anchor, template, dims)
        if not _bbox_inside_grid(bbox, grid_h, grid_w):
            continue
        if any(bbox == frame["bbox"] for frame in older_frames + [prev_frame]):
            continue

        border = _border_set(bbox)
        if protected & border:
            continue

        overlap = prev_border & border
        if not overlap:
            continue
        if any(border & old_border for old_border in older_borders):
            continue

        return ({"bbox": bbox}, overlap)

    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid_h = rng.randint(*GRID_H_RANGE)
    grid_w = rng.randint(*GRID_W_RANGE)
    frame_count = rng.randint(*FRAME_COUNT_RANGE)

    colors = list(range(10))
    bg = rng.choice(colors)
    colors.remove(bg)
    rng.shuffle(colors)
    frame_colors = colors[:frame_count]

    orientations = [rng.choice(("wide", "tall")) for _ in range(frame_count)]
    if len(set(orientations)) == 1:
        flip_idx = rng.randrange(frame_count)
        orientations[flip_idx] = "tall" if orientations[flip_idx] == "wide" else "wide"

    first_bbox = _sample_initial_bbox(rng, grid_h, grid_w, orientations[0])
    if first_bbox is None:
        return None

    frames: list[dict[str, int | tuple[int, int, int, int]]] = [
        {"color": int(frame_colors[0]), "bbox": first_bbox}
    ]
    protected: set[tuple[int, int]] = set()

    for idx in range(1, frame_count):
        built = _make_frame(
            rng=rng,
            grid_h=grid_h,
            grid_w=grid_w,
            orientation=orientations[idx],
            prev_frame=frames[-1],
            older_frames=frames[:-1],
            protected=protected,
        )
        if built is None:
            return None
        frame_stub, overlap = built
        frame_stub["color"] = int(frame_colors[idx])
        frames.append(frame_stub)
        protected.update(overlap)

    grid = _render_scene(grid_h, grid_w, bg, frames)
    if not _scene_is_valid(grid, bg, frames):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(SCENE_TRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {SCENE_TRIES} attempts")
