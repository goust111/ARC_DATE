from __future__ import annotations

import random

import numpy as np


PALETTE = [1, 2, 3, 4, 7]
SIZE_CHOICES = [
    [9, 7, 5, 3],
    [11, 9, 7, 5],
    [11, 9, 7, 5, 3],
]
GRID_SIDES = [15, 17, 19, 21]
MAX_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _chebyshev_grid(side: int) -> np.ndarray:
    center = side // 2
    rows, cols = np.indices((side, side))
    return np.maximum(np.abs(rows - center), np.abs(cols - center))


def _square_border(size: int, color: int, bg: int = 0) -> np.ndarray:
    patch = np.full((size, size), bg, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    return patch


def _extract_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    H, W = g.shape
    center_r = (H - 1) / 2.0
    center_c = (W - 1) / 2.0

    frames: list[dict] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        coords = np.argwhere(g == color)
        if coords.size == 0:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height != width or height < 3 or height % 2 == 0:
            raise AssertionError("expected odd square frame bbox per color")
        local = coords - np.array([[r0, c0]])
        if not np.all(
            (local[:, 0] == 0)
            | (local[:, 0] == height - 1)
            | (local[:, 1] == 0)
            | (local[:, 1] == width - 1)
        ):
            raise AssertionError("visible pixels must lie on the square border")
        center = ((r0 + r1) / 2.0, (c0 + c1) / 2.0)
        dist2 = (center[0] - center_r) ** 2 + (center[1] - center_c) ** 2
        frames.append(
            {
                "color": color,
                "size": height,
                "half": height // 2,
                "bbox": (r0, c0, r1, c1),
                "center": center,
                "dist2": dist2,
            }
        )

    if not frames:
        raise AssertionError("expected at least one frame")
    if len({frame["size"] for frame in frames}) != len(frames):
        raise AssertionError("frame sizes must be unique")
    if len({frame["dist2"] for frame in frames}) != len(frames):
        raise AssertionError("frame distances to input center must be unique")
    return frames


def _build_output_from_frames(frames: list[dict], bg: int = 0) -> np.ndarray:
    layers = sorted(frames, key=lambda frame: frame["size"], reverse=True)
    side = int(layers[0]["size"])
    dist = _chebyshev_grid(side)

    out = np.full((side, side), bg, dtype=int)
    priority = np.full((side, side), np.inf, dtype=float)

    for idx, frame in enumerate(layers):
        upper = int(frame["half"])
        lower = int(layers[idx + 1]["half"]) if idx + 1 < len(layers) else 0
        mask = (dist <= upper) & (dist >= lower)
        better = mask & (frame["dist2"] < priority)
        out[better] = int(frame["color"])
        priority[better] = float(frame["dist2"])

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recenter scattered square frames into concentric filled layers with distance-priority overlaps.

    Input structure:
    - The background is black and the foreground consists of 4-5 different colors.
    - Each foreground color belongs to one odd-sized 1-pixel-thick hollow square frame.
    - Frames are scattered on the input canvas and may partially occlude one another, but each color still preserves its square bounding box.

    Rule:
    - Recover every frame's color, side length, and original center from its visible square bounding box.
    - Sort frames by side length from large to small to define the outer-to-inner concentric order.
    - Create a square output whose side is the largest recovered frame size, and convert the ordered frames into centered Chebyshev layers: each layer fills from its own boundary inward to the next smaller boundary, with adjacent layers sharing that boundary.
    - Draw layers from outer to inner; whenever a shared-boundary pixel is claimed twice, keep the color whose original frame center was closer to the input grid center.
    - Background cells remain black.

    Key insight:
    - Size order determines nesting, but overlap priority remembers the original position: shared boundaries are resolved by distance to the source-grid center, not by which layer is geometrically outer.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = _get_bg(g)
    frames = _extract_frames(g)
    return _build_output_from_frames(frames, bg=bg)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def _side_presence(grid: np.ndarray, color: int, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return (
        np.any(grid[r0, c0 : c1 + 1] == color)
        and np.any(grid[r1, c0 : c1 + 1] == color)
        and np.any(grid[r0 : r1 + 1, c0] == color)
        and np.any(grid[r0 : r1 + 1, c1] == color)
    )


def _shared_boundary_mask(side: int, half: int) -> np.ndarray:
    return _chebyshev_grid(side) == half


def _semantic_ok(input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
    try:
        frames = sorted(_extract_frames(input_grid), key=lambda frame: frame["size"], reverse=True)
    except AssertionError:
        return False
    if len(frames) < 4:
        return False

    sizes = [int(frame["size"]) for frame in frames]
    if any(a - b != 2 for a, b in zip(sizes, sizes[1:])):
        return False

    inner_wins = 0
    outer_wins = 0
    for outer, inner in zip(frames, frames[1:]):
        shared = _shared_boundary_mask(output_grid.shape[0], int(inner["half"]))
        colors = set(int(v) for v in np.unique(output_grid[shared]))
        winner = inner if inner["dist2"] < outer["dist2"] else outer
        loser = outer if winner is inner else inner
        if colors != {int(winner["color"])}:
            return False
        if winner is inner:
            inner_wins += 1
        else:
            outer_wins += 1
        if winner is inner and winner["dist2"] >= loser["dist2"]:
            return False
        if winner is outer and winner["dist2"] >= loser["dist2"]:
            return False

    return inner_wins >= 1 and outer_wins >= 1


def _candidate_centers(side: int, size: int) -> list[tuple[int, int]]:
    half = size // 2
    return [(r, c) for r in range(half, side - half) for c in range(half, side - half)]


def _sample_layout(side: int, sizes: list[int]) -> tuple[list[tuple[int, int]], list[float]] | None:
    grid_center = (side - 1) / 2.0
    candidates = [_candidate_centers(side, size)[:] for size in sizes]
    for cells in candidates:
        random.shuffle(cells)

    for _ in range(200):
        centers = [random.choice(cells) for cells in candidates]
        if len(set(centers)) != len(centers):
            continue
        dist2 = [float((r - grid_center) ** 2 + (c - grid_center) ** 2) for r, c in centers]
        if len(set(dist2)) != len(dist2):
            continue
        pair_signs = [dist2[i + 1] - dist2[i] for i in range(len(dist2) - 1)]
        if not any(delta < 0 for delta in pair_signs):
            continue
        if not any(delta > 0 for delta in pair_signs):
            continue

        boxes = []
        for size, (r, c) in zip(sizes, centers):
            half = size // 2
            boxes.append((r - half, c - half, r + half, c + half))
        overlaps = sum(
            int(_boxes_overlap(boxes[i], boxes[j]))
            for i in range(len(boxes))
            for j in range(i + 1, len(boxes))
        )
        need_overlap = random.random() < 0.6
        if need_overlap and overlaps < 1:
            continue
        if not need_overlap and overlaps != 0:
            continue
        if overlaps > 2:
            continue
        return centers, dist2
    return None


def _render_input(side: int, sizes: list[int], colors: list[int], centers: list[tuple[int, int]]) -> np.ndarray:
    grid = np.zeros((side, side), dtype=int)
    order = list(range(len(sizes)))
    random.shuffle(order)
    for idx in order:
        size = int(sizes[idx])
        color = int(colors[idx])
        center_r, center_c = centers[idx]
        half = size // 2
        patch = _square_border(size, color)
        r0 = center_r - half
        c0 = center_c - half
        grid[r0 : r0 + size, c0 : c0 + size] = np.where(patch != 0, patch, grid[r0 : r0 + size, c0 : c0 + size])
    return grid


def _attempt_generate() -> np.ndarray | None:
    side = random.choice(GRID_SIDES)
    sizes = random.choice(SIZE_CHOICES)
    colors = random.sample(PALETTE, len(sizes))
    sampled = _sample_layout(side, sizes)
    if sampled is None:
        return None
    centers, _dist2 = sampled
    inp = _render_input(side, sizes, colors, centers)

    intended = {
        color: {
            "size": size,
            "bbox": (r - size // 2, c - size // 2, r + size // 2, c + size // 2),
        }
        for color, size, (r, c) in zip(colors, sizes, centers)
    }

    try:
        extracted = _extract_frames(inp)
    except AssertionError:
        return None

    if {frame["color"] for frame in extracted} != set(colors):
        return None

    for frame in extracted:
        info = intended[int(frame["color"])]
        if int(frame["size"]) != int(info["size"]):
            return None
        if tuple(frame["bbox"]) != tuple(info["bbox"]):
            return None
        if not _side_presence(inp, int(frame["color"]), tuple(frame["bbox"])):
            return None

    try:
        out = main(inp)
    except AssertionError:
        return None

    if out.shape != (max(sizes), max(sizes)):
        return None
    if not _semantic_ok(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid scattered-frame input")
