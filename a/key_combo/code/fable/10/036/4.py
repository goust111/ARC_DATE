from collections import deque

import numpy as np


MAX_GENERATION_ATTEMPTS = 500
BG = 0
PALETTE = tuple(range(1, 10))
RNG = np.random.default_rng()


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _color_components(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == color
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _frame_info(grid: np.ndarray, color: int, coords: np.ndarray) -> dict | None:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return None
    if len(coords) != 2 * (h + w) - 4:
        return None

    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if not np.all(patch[border] == color):
        return None
    if np.any(patch[~border] == color):
        return None

    interior = patch[1:-1, 1:-1]
    count = int(np.count_nonzero(interior != BG))
    return {
        "color": int(color),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "h": h,
        "w": w,
        "count": count,
    }


def _extract_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    if bg != BG:
        raise AssertionError("expected black background")

    frames: list[dict] = []
    residual_coords: set[tuple[int, int]] = set()
    for color in sorted(set(map(int, np.unique(g))) - {bg}):
        for coords in _color_components(g, color):
            info = _frame_info(g, color, coords)
            if info is None:
                residual_coords.update(map(tuple, coords.tolist()))
            else:
                frames.append(info)

    assert frames, "expected at least one hollow rectangle"
    frames.sort(key=lambda info: (info["bbox"][1], info["bbox"][0]))

    for rr, cc in residual_coords:
        owners = 0
        for info in frames:
            r0, c0, r1, c1 = info["bbox"]
            if r0 < rr < r1 and c0 < cc < c1:
                owners += 1
        assert owners == 1, "every non-frame pixel must be a dot inside exactly one frame"

    return frames


def _render_packed(
    specs: list[dict],
    *,
    use_counts: bool,
    use_normalized: bool,
    bg: int = BG,
) -> np.ndarray:
    heights = [min(spec["h"], spec["w"]) if use_normalized else spec["h"] for spec in specs]
    widths = [max(spec["h"], spec["w"]) if use_normalized else spec["w"] for spec in specs]
    out_h = int(max(heights))
    out_w = int(sum(widths) + (sum(spec["count"] for spec in specs) if use_counts else 0))
    out = np.full((out_h, out_w), bg, dtype=int)

    cursor = 0
    for spec, rect_h, rect_w in zip(specs, heights, widths):
        if use_counts:
            cursor += int(spec["count"])
        out[:rect_h, cursor : cursor + rect_w] = int(spec["color"])
        cursor += rect_w
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Solidify each hollow frame, normalize it to horizontal, and pack them using dot-count left padding.

    Input structure:
    - The grid has a black background with several separated colored hollow rectangles.
    - Each rectangle may contain zero or more counting dots inside its hollow interior.
    - Dots disappear in the output and serve only as per-rectangle gap cues.

    Rule:
    - Sort the rectangles by their left border from left to right.
    - For each rectangle, count the non-background cells strictly inside its frame.
    - Build an output whose height is the maximum shorter side among the rectangles.
    - Starting from the left edge, leave `k` background columns where `k` is that frame's interior dot count,
      then paint a top-aligned solid rectangle of size `min(h,w) x max(h,w)` in the frame color.
    - Stop the canvas exactly at the right edge of the last painted rectangle.

    Key insight:
    - The dots do not label order; they encode each frame's own left padding, while the frame contributes only its color
      and its orientation-normalized bounding-box dimensions.
    """
    frames = _extract_frames(np.asarray(grid, dtype=int))
    return _render_packed(frames, use_counts=True, use_normalized=True)


def _is_monotone(values: list[int]) -> bool:
    return values == sorted(values) or values == sorted(values, reverse=True)


def _dot_capacity(h: int, w: int) -> int:
    inner_h = h - 2
    inner_w = w - 2
    return ((inner_h + 1) // 2) * ((inner_w + 1) // 2)


def _sample_non_touching_points(h: int, w: int, count: int) -> list[tuple[int, int]]:
    candidates = [(r, c) for r in range(1, h - 1, 2) for c in range(1, w - 1, 2)]
    if count > len(candidates):
        raise AssertionError("insufficient interior capacity for requested dots")
    chosen = RNG.choice(len(candidates), size=count, replace=False).tolist()
    return [candidates[index] for index in chosen]


def _build_grid_from_specs(specs: list[dict], dot_color: int) -> np.ndarray:
    input_ws = [spec["w"] for spec in specs]
    input_hs = [spec["h"] for spec in specs]
    gaps = RNG.integers(1, 4, size=len(specs) - 1).tolist()
    left_margin = int(RNG.integers(0, 3))
    right_margin = int(RNG.integers(0, 3))
    total_w = left_margin + sum(input_ws) + sum(gaps) + right_margin
    total_h = max(input_hs) + int(RNG.integers(2, 6))
    if total_w > 30 or total_h > 30:
        raise AssertionError("input grid exceeds ARC limits")

    grid = np.full((total_h, total_w), BG, dtype=int)
    cursor = left_margin
    for index, spec in enumerate(specs):
        h = int(spec["h"])
        w = int(spec["w"])
        top = int(RNG.integers(0, total_h - h + 1))
        color = int(spec["color"])
        grid[top, cursor : cursor + w] = color
        grid[top + h - 1, cursor : cursor + w] = color
        grid[top : top + h, cursor] = color
        grid[top : top + h, cursor + w - 1] = color
        for dr, dc in _sample_non_touching_points(h, w, int(spec["count"])):
            grid[top + dr, cursor + dc] = dot_color
        spec["bbox"] = (top, cursor, top + h - 1, cursor + w - 1)
        cursor += w
        if index < len(specs) - 1:
            cursor += int(gaps[index])
    return grid


def _make_specs() -> tuple[list[dict], int]:
    n = int(RNG.integers(3, 5))
    frame_colors = RNG.choice(PALETTE, size=n + 1, replace=False).tolist()
    dot_color = int(frame_colors.pop())

    orientations = ["landscape", "portrait"]
    while len(orientations) < n:
        orientations.append(str(RNG.choice(["landscape", "portrait"])))
    RNG.shuffle(orientations)

    shorts = RNG.integers(3, 6, size=n).tolist()
    deltas = RNG.integers(1, 3, size=n).tolist()
    longs = [short + delta for short, delta in zip(shorts, deltas)]

    capacities = [_dot_capacity(longs[i], shorts[i]) for i in range(n)]
    if any(cap == 0 for cap in capacities):
        raise AssertionError("invalid interior capacity")
    counts = []
    zero_index = int(RNG.integers(0, n))
    positive_index = int(RNG.integers(0, n - 1))
    if positive_index >= zero_index:
        positive_index += 1
    for i in range(n):
        max_count = min(3, capacities[i])
        if i == zero_index:
            counts.append(0)
        elif i == positive_index:
            counts.append(int(RNG.integers(1, max_count + 1)))
        else:
            counts.append(int(RNG.integers(0, max_count + 1)))

    if len(set(counts)) < 2:
        raise AssertionError("counts are too degenerate")

    specs_unordered: list[dict] = []
    for color, short, long, count, orient in zip(frame_colors, shorts, longs, counts, orientations):
        if orient == "portrait":
            h, w = long, short
        else:
            h, w = short, long
        specs_unordered.append({"color": int(color), "h": int(h), "w": int(w), "count": int(count)})

    if not any(spec["h"] > spec["w"] for spec in specs_unordered):
        raise AssertionError("need at least one portrait frame")
    if not any(spec["h"] < spec["w"] for spec in specs_unordered):
        raise AssertionError("need at least one landscape frame")

    specs = []
    for _ in range(20):
        order = RNG.permutation(n).tolist()
        specs = [specs_unordered[index] for index in order]
        counts_in_order = [spec["count"] for spec in specs]
        longs_in_order = [max(spec["h"], spec["w"]) for spec in specs]
        if not _is_monotone(counts_in_order) and not _is_monotone(longs_in_order):
            break
    else:
        raise AssertionError("sizes are too degenerate")

    packed = _render_packed(specs, use_counts=True, use_normalized=True)
    if max(packed.shape) > 30:
        raise AssertionError("output grid exceeds ARC limits")

    packed_without_counts = _render_packed(specs, use_counts=False, use_normalized=True)
    if np.array_equal(packed, packed_without_counts):
        raise AssertionError("dot counts must affect the output")

    packed_without_normalization = _render_packed(specs, use_counts=True, use_normalized=False)
    if np.array_equal(packed, packed_without_normalization):
        raise AssertionError("orientation normalization must affect the output")

    return specs, dot_color


def _validate_generated(grid: np.ndarray, specs: list[dict]) -> bool:
    detected = _extract_frames(grid)
    expected = _render_packed(specs, use_counts=True, use_normalized=True)
    if not np.array_equal(main(grid), expected):
        return False
    if len(detected) != len(specs):
        return False
    for got, want in zip(detected, specs):
        if got["color"] != want["color"] or got["h"] != want["h"] or got["w"] != want["w"] or got["count"] != want["count"]:
            return False
    counts = [spec["count"] for spec in specs]
    if 0 not in counts or max(counts) == 0:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    try:
        specs, dot_color = _make_specs()
        grid = _build_grid_from_specs(specs, dot_color)
        if not _validate_generated(grid, specs):
            return None
        return grid
    except AssertionError:
        return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 留白量尺 input")
