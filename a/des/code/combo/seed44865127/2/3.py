from __future__ import annotations

from collections import defaultdict
import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_ATTEMPTS = 400
GRID_MIN = 18
GRID_MAX = 30
FRAME_COUNT_RANGE = (2, 5)  # high is exclusive
MAX_LAYERS = 3


def _dominant_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            comps.append(coords)

    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return int(min(rows)), int(min(cols)), int(max(rows)), int(max(cols))


def _is_hollow_frame(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    if height < 3 or width < 3:
        return False
    if len(coords) != 2 * (height + width) - 4:
        return False

    mask = np.zeros((height, width), dtype=bool)
    for r, c in coords:
        mask[int(r - r0), int(c - c0)] = True

    border = np.zeros((height, width), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.array_equal(mask, border))


def _point_to_rect_distance(point: tuple[int, int], bbox: tuple[int, int, int, int]) -> int:
    r, c = point
    r0, c0, r1, c1 = bbox

    if r < r0:
        dr = int(r0 - r)
    elif r > r1:
        dr = int(r - r1)
    else:
        dr = 0

    if c < c0:
        dc = int(c0 - c)
    elif c > c1:
        dc = int(c - c1)
    else:
        dc = 0

    return int(dr + dc)


def _draw_rect_outline(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _max_ring_layers(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    inner_h = int(r1 - r0 - 1)
    inner_w = int(c1 - c0 - 1)
    return max(0, min(inner_h, inner_w) // 2)


def _render_output(
    shape: tuple[int, int], bg: int, frames: list[dict[str, int]], layer_counts: list[int]
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)

    for frame in frames:
        _draw_rect_outline(out, frame["bbox"], int(frame["color"]))

    for frame, count in zip(frames, layer_counts):
        color = int(frame["color"])
        r0, c0, r1, c1 = frame["bbox"]
        layers = min(int(count), _max_ring_layers(frame["bbox"]))
        for offset in range(layers):
            inner_box = (int(r0 + 1 + offset), int(c0 + 1 + offset), int(r1 - 1 - offset), int(c1 - 1 - offset))
            _draw_rect_outline(out, inner_box, color)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group scattered same-color marker pixels with their matching frame and draw
    that many concentric rectangular rings inside the frame.

    Input structure:
    - The dominant background color fills the canvas.
    - Several non-overlapping 1-pixel-thick hollow rectangular frames appear.
    - Outside the frames are singleton marker pixels whose color matches a frame.

    Rule:
    - Remove all outside markers.
    - Count how many markers belong to each frame.
    - Inside each frame, draw that many concentric rectangular rings in the
      frame color, starting flush with the inner wall and shrinking inward by
      one cell per layer.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    bg = _dominant_color(g)
    frame_mask = np.zeros_like(g, dtype=bool)
    frames: list[dict[str, int]] = []

    for color0 in np.unique(g):
        color = int(color0)
        if color == bg:
            continue
        for coords in _components(g == color):
            if _is_hollow_frame(coords):
                bbox = _bbox(coords)
                frames.append({"color": color, "bbox": bbox})
                for r, c in coords:
                    frame_mask[r, c] = True

    frames.sort(key=lambda item: (item["bbox"][0], item["bbox"][1], item["color"]))
    if not frames:
        return g.copy()

    frames_by_color: dict[int, list[int]] = defaultdict(list)
    for idx, frame in enumerate(frames):
        frames_by_color[int(frame["color"])].append(idx)

    layer_counts = [0] * len(frames)
    marker_points = np.argwhere((g != bg) & ~frame_mask)
    for rr, cc in marker_points:
        r = int(rr)
        c = int(cc)
        color = int(g[r, c])
        candidates = frames_by_color.get(color, [])
        if not candidates:
            continue
        best_idx = min(
            candidates,
            key=lambda idx: (_point_to_rect_distance((r, c), frames[idx]["bbox"]), idx),
        )
        layer_counts[int(best_idx)] += 1

    return _render_output(g.shape, bg, frames, layer_counts)


def _frame_spec(rng: np.random.Generator, color: int) -> dict[str, int]:
    layers = int(rng.integers(0, MAX_LAYERS + 1))
    min_inner = max(3, 2 * layers)
    inner_h = int(min_inner + rng.integers(0, 4))
    inner_w = int(min_inner + rng.integers(0, 4))
    return {
        "color": int(color),
        "layers": int(layers),
        "outer_h": int(inner_h + 2),
        "outer_w": int(inner_w + 2),
    }


def _choose_grid_shape(
    rng: np.random.Generator, specs: list[dict[str, int]]
) -> tuple[int, int] | None:
    max_h = max(int(spec["outer_h"]) for spec in specs)
    max_w = max(int(spec["outer_w"]) for spec in specs)
    padded_area = sum((int(spec["outer_h"]) + 2) * (int(spec["outer_w"]) + 2) for spec in specs)
    total_markers = sum(int(spec["layers"]) for spec in specs)
    required_area = int(padded_area + 8 * total_markers + 30)

    for _ in range(80):
        height = int(rng.integers(GRID_MIN, GRID_MAX + 1))
        width = int(rng.integers(GRID_MIN, GRID_MAX + 1))
        if height < max_h + 2 or width < max_w + 2:
            continue
        if height * width < required_area:
            continue
        return height, width
    return None


def _place_frames(
    rng: np.random.Generator, height: int, width: int, specs: list[dict[str, int]]
) -> tuple[list[dict[str, int]], np.ndarray] | None:
    blocked = np.zeros((height, width), dtype=bool)
    frames: list[dict[str, int] | None] = [None] * len(specs)

    order = sorted(
        range(len(specs)),
        key=lambda idx: (int(specs[idx]["outer_h"]) + 2) * (int(specs[idx]["outer_w"]) + 2),
        reverse=True,
    )

    for idx in order:
        outer_h = int(specs[idx]["outer_h"])
        outer_w = int(specs[idx]["outer_w"])
        placed = False
        for _ in range(400):
            r0 = int(rng.integers(1, height - outer_h))
            c0 = int(rng.integers(1, width - outer_w))
            if blocked[r0 - 1 : r0 + outer_h + 1, c0 - 1 : c0 + outer_w + 1].any():
                continue
            blocked[r0 - 1 : r0 + outer_h + 1, c0 - 1 : c0 + outer_w + 1] = True
            frames[idx] = {
                "color": int(specs[idx]["color"]),
                "bbox": (int(r0), int(c0), int(r0 + outer_h - 1), int(c0 + outer_w - 1)),
            }
            placed = True
            break
        if not placed:
            return None

    return [frame for frame in frames if frame is not None], blocked


def _place_markers(
    rng: np.random.Generator,
    height: int,
    width: int,
    frames: list[dict[str, int]],
    layer_counts: list[int],
    frame_blocked: np.ndarray,
) -> list[tuple[int, int, int]] | None:
    blocked = frame_blocked.copy()
    jobs: list[int] = []
    for frame, count in zip(frames, layer_counts):
        jobs.extend([int(frame["color"])] * int(count))
    rng.shuffle(jobs)

    markers: list[tuple[int, int, int]] = []
    for color in jobs:
        free = np.argwhere(~blocked)
        if free.size == 0:
            return None
        choice = free[int(rng.integers(len(free)))]
        r, c = map(int, choice)
        markers.append((r, c, int(color)))
        blocked[max(0, r - 1) : min(height, r + 2), max(0, c - 1) : min(width, c + 2)] = True

    return markers


def _build_input(
    height: int,
    width: int,
    bg: int,
    frames: list[dict[str, int]],
    markers: list[tuple[int, int, int]],
) -> np.ndarray:
    grid = np.full((height, width), int(bg), dtype=int)
    for frame in frames:
        _draw_rect_outline(grid, frame["bbox"], int(frame["color"]))
    for r, c, color in markers:
        grid[int(r), int(c)] = int(color)
    return grid


def _input_is_clear(grid: np.ndarray, bg: int) -> bool:
    colors, counts = np.unique(grid, return_counts=True)
    count_by_color = {int(color): int(count) for color, count in zip(colors, counts)}
    bg_count = count_by_color.get(int(bg), 0)
    return all(color == int(bg) or count < bg_count for color, count in count_by_color.items())


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    n_frames = int(rng.integers(*FRAME_COUNT_RANGE))

    palette = rng.choice(np.arange(10), size=n_frames + 1, replace=False)
    bg = int(palette[0])
    specs = [_frame_spec(rng, int(color)) for color in palette[1:]]
    layer_counts = [int(spec["layers"]) for spec in specs]

    if sum(layer_counts) == 0:
        chosen = int(rng.integers(n_frames))
        specs[chosen]["layers"] = int(rng.integers(1, MAX_LAYERS + 1))
        layer_counts[chosen] = int(specs[chosen]["layers"])

    shape = _choose_grid_shape(rng, specs)
    if shape is None:
        return None
    height, width = shape

    placed = _place_frames(rng, height, width, specs)
    if placed is None:
        return None
    frames, frame_blocked = placed

    markers = _place_markers(rng, height, width, frames, layer_counts, frame_blocked)
    if markers is None:
        return None

    inp = _build_input(height, width, bg, frames, markers)
    if not _input_is_clear(inp, bg):
        return None

    expected = _render_output(inp.shape, bg, frames, layer_counts)
    try:
        if np.array_equal(main(inp), expected):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
