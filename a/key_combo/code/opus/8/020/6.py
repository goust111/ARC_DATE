from __future__ import annotations

from collections import Counter
import random

import numpy as np


GREY = 5
RED = 2
BLACK = 0
DEPTH_TO_COLOR = {0: 0, 1: 1, 2: 3, 3: 4, 4: 7}
ANCHOR_MODES = ("tr", "bl", "br")
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) != bg:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            components.append({"coords": arr, "bbox": _bbox_coords(arr), "area": int(arr.shape[0])})

    return components


def _extract_rectangles(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    records: list[dict] = []

    for comp in _find_components(g, bg):
        r0, c0, r1, c1 = comp["bbox"]
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        h, w = patch.shape
        if int(comp["area"]) != h * w:
            raise AssertionError("every foreground component must be a solid rectangle")
        colors = set(map(int, np.unique(patch)))
        if colors != {GREY, RED}:
            raise AssertionError("each rectangle must contain only gray pixels and one red marker")
        marker_coords = np.argwhere(patch == RED)
        if marker_coords.shape != (1, 2):
            raise AssertionError("each rectangle must contain exactly one red marker")
        mr, mc = map(int, marker_coords[0])
        records.append(
            {
                "bbox": (r0, c0, r1, c1),
                "top": int(r0),
                "left": int(c0),
                "h": int(h),
                "w": int(w),
                "area": int(h * w),
                "marker_local": (mr, mc),
                "marker_abs": (int(r0 + mr), int(c0 + mc)),
            }
        )

    if not records:
        raise AssertionError("expected marked rectangles")
    return records


def _largest_record(records: list[dict]) -> dict:
    areas = [rec["area"] for rec in records]
    largest_area = max(areas)
    if areas.count(largest_area) != 1:
        raise AssertionError("largest rectangle must be unique")
    return records[areas.index(largest_area)]


def _render_overlay(shape: tuple[int, int], records: list[dict], target_marker: tuple[int, int]) -> np.ndarray:
    h, w = map(int, shape)
    coverage = np.zeros((h, w), dtype=int)
    target_r, target_c = map(int, target_marker)

    for rec in records:
        top = target_r - rec["marker_local"][0]
        left = target_c - rec["marker_local"][1]
        bottom = top + rec["h"]
        right = left + rec["w"]
        if top < 0 or left < 0 or bottom > h or right > w:
            raise AssertionError("aligned rectangle moved out of bounds")
        coverage[top:bottom, left:right] += 1

    out = np.zeros((h, w), dtype=int)
    for depth, color in DEPTH_TO_COLOR.items():
        if depth == 0:
            continue
        out[coverage == depth] = int(color)
    out[target_r, target_c] = RED
    return out


def _render_top_left_shortcut(shape: tuple[int, int], records: list[dict]) -> np.ndarray:
    largest = _largest_record(records)
    target_top = largest["top"]
    target_left = largest["left"]
    h, w = map(int, shape)
    coverage = np.zeros((h, w), dtype=int)

    for rec in records:
        bottom = target_top + rec["h"]
        right = target_left + rec["w"]
        if bottom > h or right > w:
            raise AssertionError("shortcut overlay moved out of bounds")
        coverage[target_top:bottom, target_left:right] += 1

    out = np.zeros((h, w), dtype=int)
    for depth, color in DEPTH_TO_COLOR.items():
        if depth == 0:
            continue
        out[coverage == depth] = int(color)
    out[target_top + largest["marker_local"][0], target_left + largest["marker_local"][1]] = RED
    return out


def _marker_local(h: int, w: int, vr: int, hc: int, mode: str) -> tuple[int, int]:
    if mode == "tr":
        return int(vr), int(w - 1 - hc)
    if mode == "bl":
        return int(h - 1 - vr), int(hc)
    if mode == "br":
        return int(h - 1 - vr), int(w - 1 - hc)
    raise ValueError(f"unknown anchor mode: {mode}")


def main(grid: np.ndarray) -> np.ndarray:
    """Overlay the marked gray rectangles at their shared red anchor and recolor by coverage depth.

    Input structure:
    - The grid has a black background plus 3-4 separated solid gray rectangles.
    - Each rectangle contains exactly one red marker pixel.
    - The marker occupies the same corner-relative anchor position inside every rectangle.

    Rule:
    - Identify each non-black 4-connected component as one marked rectangle.
    - Choose the unique largest-area rectangle and use its red marker as the reference anchor.
    - Translate every rectangle so its own red marker lands on that anchor, count how many translated
      rectangles cover each cell, map depths 1/2/3/4 to blue/green/yellow/orange, keep uncovered
      cells black, and finally restore the shared anchor cell to red.

    Key insight:
    - The visible output is a heatmap of overlap depth; that depth does not exist in any single input
      rectangle and only appears after all rectangles are aligned on the common marker.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    records = _extract_rectangles(g)
    largest = _largest_record(records)
    return _render_overlay(g.shape, records, largest["marker_abs"])


def _independent_expected(records: list[dict], shape: tuple[int, int], target_marker: tuple[int, int]) -> np.ndarray:
    h, w = map(int, shape)
    target_r, target_c = map(int, target_marker)
    out = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            depth = 0
            for rec in records:
                top = target_r - rec["marker_local"][0]
                left = target_c - rec["marker_local"][1]
                if top <= r < top + rec["h"] and left <= c < left + rec["w"]:
                    depth += 1
            out[r, c] = DEPTH_TO_COLOR[depth]
    out[target_r, target_c] = RED
    return out


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        records = _extract_rectangles(grid)
        if len(records) not in (3, 4):
            return False
        largest = _largest_record(records)
        areas = [rec["area"] for rec in records]
        if len(set(areas)) != len(areas):
            return False
        if set(map(int, np.unique(grid))) != {BLACK, RED, GREY}:
            return False

        out = main(grid)
        expected = _independent_expected(records, grid.shape, largest["marker_abs"])
        if not np.array_equal(out, expected):
            return False

        required_colors = {BLACK, RED} | {DEPTH_TO_COLOR[d] for d in range(1, len(records) + 1)}
        if set(map(int, np.unique(out))) != required_colors:
            return False
        if int(out[largest["marker_abs"]]) != RED:
            return False
        if np.array_equal(out, _render_top_left_shortcut(grid.shape, records)):
            return False
        if int((out != BLACK).sum()) >= int(out.size * 0.65):
            return False
        return True
    except AssertionError:
        return False


def _sample_sizes(count: int) -> list[tuple[int, int]]:
    heights = sorted(random.sample(range(4, 10), count))
    widths = sorted(random.sample(range(4, 10), count))
    sizes = list(zip(heights, widths, strict=False))
    if len({h * w for h, w in sizes}) != count:
        raise AssertionError("areas must be unique")
    return sizes


def _distribute_slack(slots: int, slack: int) -> list[int]:
    parts = [0] * slots
    for _ in range(slack):
        parts[random.randrange(slots)] += 1
    return parts


def _build_candidate() -> np.ndarray | None:
    count = random.choice((3, 4))
    mode = random.choice(ANCHOR_MODES)
    sizes = _sample_sizes(count)

    min_h = min(h for h, _w in sizes)
    min_w = min(w for _h, w in sizes)
    vr = random.randint(1, min(2, min_h - 2))
    hc = random.randint(1, min(2, min_w - 2))

    specs = []
    for h, w in sizes:
        specs.append({"h": int(h), "w": int(w), "marker_local": _marker_local(h, w, vr, hc, mode)})

    order = list(range(count))
    random.shuffle(order)
    placed_specs = [specs[i] for i in order]

    base_gaps = [random.randint(2, 4) for _ in range(count - 1)]
    min_total_w = 2 + sum(spec["w"] for spec in placed_specs) + sum(base_gaps)
    if min_total_w > 30:
        return None

    grid_w = random.randint(min_total_w, min(30, min_total_w + 6))
    slack_parts = _distribute_slack(count + 1, grid_w - min_total_w)
    left_margin = 1 + slack_parts[0]
    gaps = [base_gaps[i] + slack_parts[i + 1] for i in range(count - 1)]

    max_h = max(spec["h"] for spec in placed_specs)
    min_total_h = max_h + 3
    if min_total_h > 30:
        return None
    grid_h = random.randint(min_total_h, min(30, min_total_h + 7))

    grid = np.zeros((grid_h, grid_w), dtype=int)
    x = left_margin
    for idx, spec in enumerate(placed_specs):
        top = random.randint(1, grid_h - spec["h"] - 1)
        bottom = top + spec["h"]
        right = x + spec["w"]
        grid[top:bottom, x:right] = GREY
        mr, mc = spec["marker_local"]
        grid[top + mr, x + mc] = RED
        x = right + (gaps[idx] if idx < len(gaps) else 0)

    return grid if _validate_generated(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        candidate = _build_candidate()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid marked-rectangle heatmap input")
