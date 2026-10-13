from __future__ import annotations

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components_by_color(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            components.append({"color": color, "coords": coords})

    return components


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _parse_frame(grid: np.ndarray, component: dict, bg: int) -> dict:
    coords = {(int(r), int(c)) for r, c in component["coords"]}
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    assert h >= 3 and w >= 3, "frame bbox must be at least 3x3"

    expected = _border_cells(r0, c0, r1, c1)
    extra = coords - expected
    missing = expected - coords
    assert not extra, "component must lie on its rectangular border"
    assert len(missing) == 1, "frame must have exactly one border gap"

    gap_r, gap_c = next(iter(missing))
    center_r = r0 + h // 2
    center_c = c0 + w // 2

    if gap_r == r0 and w % 2 == 1 and gap_c == center_c:
        side = "up"
        axis = "vertical"
    elif gap_r == r1 and w % 2 == 1 and gap_c == center_c:
        side = "down"
        axis = "vertical"
    elif gap_c == c0 and h % 2 == 1 and gap_r == center_r:
        side = "left"
        axis = "horizontal"
    elif gap_c == c1 and h % 2 == 1 and gap_r == center_r:
        side = "right"
        axis = "horizontal"
    else:
        raise AssertionError("gap must be centered on one side")

    if h > 2 and w > 2:
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        assert np.all(interior == bg), "frame interior must stay background in the input"

    return {
        "color": int(component["color"]),
        "bbox": (r0, c0, r1, c1),
        "h": h,
        "w": w,
        "gap": (gap_r, gap_c),
        "side": side,
        "axis": axis,
        "rect_area": h * w,
        "component_area": len(coords),
    }


def _segment_clear(grid: np.ndarray, a: dict, b: dict, bg: int) -> bool:
    (ar, ac) = a["gap"]
    (br, bc) = b["gap"]
    if ar == br and a["axis"] == b["axis"] == "horizontal":
        c0, c1 = sorted((ac, bc))
        return c1 - c0 >= 2 and bool(np.all(grid[ar, c0 + 1 : c1] == bg))
    if ac == bc and a["axis"] == b["axis"] == "vertical":
        r0, r1 = sorted((ar, br))
        return r1 - r0 >= 2 and bool(np.all(grid[r0 + 1 : r1, ac] == bg))
    return False


def _pair_frames(grid: np.ndarray, frames: list[dict], bg: int) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    n = len(frames)
    for i in range(n):
        for j in range(i + 1, n):
            if _segment_clear(grid, frames[i], frames[j], bg):
                candidates.append((i, j))

    assert candidates, "expected at least one bridge pair"
    degree = [0] * n
    for i, j in candidates:
        degree[i] += 1
        degree[j] += 1
    assert all(d == 1 for d in degree), "pairing must be unique for every frame"
    return candidates


def _bridge_color(a: dict, b: dict) -> int:
    area_a = int(a["rect_area"])
    area_b = int(b["rect_area"])
    assert area_a != area_b, "paired frames must have distinct rectangle areas"
    return int(a["color"] if area_a > area_b else b["color"])


def _paint_bridge(out: np.ndarray, a: dict, b: dict, color: int, bg: int) -> None:
    (ar, ac) = a["gap"]
    (br, bc) = b["gap"]
    if ar == br:
        c0, c1 = sorted((ac, bc))
        assert bool(np.all(out[ar, c0 : c1 + 1] == bg)), "horizontal bridge must not collide"
        out[ar, c0 : c1 + 1] = color
    else:
        r0, r1 = sorted((ar, br))
        assert bool(np.all(out[r0 : r1 + 1, ac] == bg)), "vertical bridge must not collide"
        out[r0 : r1 + 1, ac] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find aligned gapped rectangular frames and connect each pair with a straight bridge.

    Each object is a hollow single-color rectangular border with exactly one centered
    one-cell gap on one side. Two frames form a pair when their gap cells are aligned
    on the same row or column and the straight corridor between the gaps is background.
    The bridge uses the color of the larger rectangle area; the original frame pixels
    remain unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _most_common_color(g)

    components = _components_by_color(g, bg)
    frames = [_parse_frame(g, component, bg) for component in components]
    assert len(frames) >= 2 and len(frames) % 2 == 0, "expected an even number of frames"

    pairs = _pair_frames(g, frames, bg)
    out = g.copy()
    for i, j in pairs:
        color = _bridge_color(frames[i], frames[j])
        _paint_bridge(out, frames[i], frames[j], color, bg)
    return out


BG = 0
FRAME_COLORS = np.arange(1, 10, dtype=int)
GRID_H_RANGE = (20, 29)
GRID_W_RANGE = (40, 49)
FRAME_SIZES = (
    (5, 5),
    (5, 7),
    (7, 5),
    (7, 7),
)
MODES = ("V", "H", "VH")
MAX_GENERATION_ATTEMPTS = 400


def _draw_frame(grid: np.ndarray, spec: dict) -> None:
    top = int(spec["top"])
    left = int(spec["left"])
    h = int(spec["h"])
    w = int(spec["w"])
    color = int(spec["color"])
    side = spec["side"]

    bottom = top + h - 1
    right = left + w - 1
    grid[top, left : right + 1] = color
    grid[bottom, left : right + 1] = color
    grid[top : bottom + 1, left] = color
    grid[top : bottom + 1, right] = color

    if side == "up":
        grid[top, left + w // 2] = BG
    elif side == "down":
        grid[bottom, left + w // 2] = BG
    elif side == "left":
        grid[top + h // 2, left] = BG
    else:
        assert side == "right"
        grid[top + h // 2, right] = BG


def _gap_of(spec: dict) -> tuple[int, int]:
    top = int(spec["top"])
    left = int(spec["left"])
    h = int(spec["h"])
    w = int(spec["w"])
    side = spec["side"]
    if side == "up":
        return top, left + w // 2
    if side == "down":
        return top + h - 1, left + w // 2
    if side == "left":
        return top + h // 2, left
    return top + h // 2, left + w - 1


def _expected_bridge_color(a: dict, b: dict) -> int:
    area_a = int(a["h"]) * int(a["w"])
    area_b = int(b["h"]) * int(b["w"])
    assert area_a != area_b
    return int(a["color"] if area_a > area_b else b["color"])


def _paint_expected_bridge(grid: np.ndarray, a: dict, b: dict) -> None:
    color = _expected_bridge_color(a, b)
    (ar, ac) = _gap_of(a)
    (br, bc) = _gap_of(b)
    if ar == br:
        c0, c1 = sorted((ac, bc))
        grid[ar, c0 : c1 + 1] = color
    else:
        r0, r1 = sorted((ar, br))
        grid[r0 : r1 + 1, ac] = color


def _sample_shape_pair(rng: np.random.Generator) -> tuple[tuple[int, int], tuple[int, int]]:
    while True:
        a = FRAME_SIZES[int(rng.integers(len(FRAME_SIZES)))]
        b = FRAME_SIZES[int(rng.integers(len(FRAME_SIZES)))]
        if a[0] * a[1] != b[0] * b[1]:
            return a, b


def _place_vertical_pair(
    rng: np.random.Generator,
    h: int,
    left_zone: int,
    right_zone: int,
    colors: list[int],
) -> tuple[dict, dict] | None:
    for _ in range(200):
        (h1, w1), (h2, w2) = _sample_shape_pair(rng)
        gap = int(rng.integers(2, 5))
        total_height = h1 + h2 + gap
        if total_height > h - 2:
            continue

        max_w = max(w1, w2)
        min_center = left_zone + max_w // 2
        max_center = right_zone - max_w // 2
        if min_center > max_center:
            continue

        center = int(rng.integers(min_center, max_center + 1))
        top1_max = h - total_height - 1
        if top1_max < 1:
            continue

        top1 = int(rng.integers(1, top1_max + 1))
        top2 = top1 + h1 + gap
        a = {"top": top1, "left": center - w1 // 2, "h": h1, "w": w1, "color": colors[0], "side": "down"}
        b = {"top": top2, "left": center - w2 // 2, "h": h2, "w": w2, "color": colors[1], "side": "up"}
        return a, b
    return None


def _place_horizontal_pair(
    rng: np.random.Generator,
    h: int,
    left_zone: int,
    right_zone: int,
    colors: list[int],
) -> tuple[dict, dict] | None:
    for _ in range(200):
        (h1, w1), (h2, w2) = _sample_shape_pair(rng)
        gap = int(rng.integers(2, 5))
        total_width = w1 + w2 + gap
        zone_width = right_zone - left_zone + 1
        if total_width > zone_width:
            continue

        max_h = max(h1, h2)
        min_row = 1 + max_h // 2
        max_row = h - 2 - max_h // 2
        if min_row > max_row:
            continue

        row = int(rng.integers(min_row, max_row + 1))
        left1_max = right_zone - total_width + 1
        if left1_max < left_zone:
            continue

        left1 = int(rng.integers(left_zone, left1_max + 1))
        left2 = left1 + w1 + gap
        a = {"top": row - h1 // 2, "left": left1, "h": h1, "w": w1, "color": colors[0], "side": "right"}
        b = {"top": row - h2 // 2, "left": left2, "h": h2, "w": w2, "color": colors[1], "side": "left"}
        return a, b
    return None


def _build_from_specs(h: int, w: int, pairs: list[tuple[dict, dict]]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((h, w), BG, dtype=int)
    for a, b in pairs:
        _draw_frame(inp, a)
        _draw_frame(inp, b)

    out = inp.copy()
    for a, b in pairs:
        _paint_expected_bridge(out, a, b)
    return inp, out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(*GRID_H_RANGE))
    w = int(rng.integers(*GRID_W_RANGE))
    mode = MODES[int(rng.integers(len(MODES)))]

    split = w // 2
    left_zone = (1, split - 3)
    right_zone = (split + 2, w - 2)

    pair_count = 1 if mode in ("V", "H") else 2
    colors = [int(c) for c in rng.choice(FRAME_COLORS, size=pair_count * 2, replace=False)]
    color_idx = 0
    pairs: list[tuple[dict, dict]] = []

    if mode in ("V", "VH"):
        pair = _place_vertical_pair(rng, h, left_zone[0], left_zone[1], colors[color_idx : color_idx + 2])
        if pair is None:
            return None
        pairs.append(pair)
        color_idx += 2

    if mode in ("H", "VH"):
        pair = _place_horizontal_pair(rng, h, right_zone[0], right_zone[1], colors[color_idx : color_idx + 2])
        if pair is None:
            return None
        pairs.append(pair)

    inp, expected = _build_from_specs(h, w, pairs)
    try:
        actual = main(inp)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
