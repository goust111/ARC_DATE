from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
PALETTE = tuple(range(1, 10))
GRID_SIDE_MAX = 30
PAIR_COUNT_RANGE = (2, 4)
EDGE_MARGIN = 1
PLACEMENT_ATTEMPTS = 200
GENERATION_ATTEMPTS = 300


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            parts.append(np.array(coords, dtype=int))

    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _solid_rect_bbox(grid: np.ndarray, coords: np.ndarray, color: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert patch.shape[0] * patch.shape[1] == len(coords), "component must fill its bounding box"
    assert np.all(patch == color), "component must be a solid monochromatic rectangle"
    return r0, c0, r1, c1


def _analyze_pairs(grid: np.ndarray) -> list[dict[str, int | str | tuple[int, int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    bg = int(np.bincount(g.ravel()).argmax())
    pairs: list[dict[str, int | str | tuple[int, int, int, int]]] = []

    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        parts = _components(g == color)
        assert len(parts) == 2, "each foreground color must have exactly two components"
        rects = [_solid_rect_bbox(g, coords, color) for coords in parts]

        if rects[0][0] == rects[1][0] and rects[0][2] == rects[1][2]:
            orientation = "horizontal"
            first, second = sorted(rects, key=lambda b: b[1])
            r0, _, r1, c1_left = first
            _, c0_right, _, _ = second
            gap = c0_right - c1_left - 1
            threshold = min(first[2] - first[0] + 1, second[2] - second[0] + 1)
        elif rects[0][1] == rects[1][1] and rects[0][3] == rects[1][3]:
            orientation = "vertical"
            first, second = sorted(rects, key=lambda b: b[0])
            _, c0, _, c1 = first
            r1_top = first[2]
            r0_bottom = second[0]
            gap = r0_bottom - r1_top - 1
            threshold = min(first[3] - first[1] + 1, second[3] - second[1] + 1)
        else:
            raise AssertionError("same-color rectangles must align on rows or on columns")

        assert gap >= 1, "paired rectangles must be separated by background"
        pairs.append(
            {
                "color": color,
                "orientation": orientation,
                "rect_a": first,
                "rect_b": second,
                "gap": gap,
                "threshold": threshold,
            }
        )

    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each aligned same-color rectangle pair with either a solid bridge or a two-rail outline bridge.

    Input structure:
    - The background is black.
    - Every non-black color forms exactly two disjoint 4-connected solid rectangles.
    - The two rectangles of a color are aligned either horizontally with identical
      row span or vertically with identical column span.

    Rule:
    - For each same-color pair, measure the background gap between the two facing sides.
    - Compare that gap with the smaller rectangle side length on the aligned axis:
      height for horizontal pairs, width for vertical pairs.
    - If the gap is at most that threshold, fill the whole gap rectangle with the
      pair color so the pair becomes one larger solid rectangle.
    - Otherwise, only extend the two endpoint pixels of the facing sides across the
      gap, producing two parallel one-pixel rails while keeping the original rectangles unchanged.

    Key insight:
    - The transformation is controlled by a size comparison, not just by color matching:
      the same paired rectangles switch between a filled bridge and an outline bridge
      depending on how wide their separation is relative to their own thickness.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for pair in _analyze_pairs(g):
        color = int(pair["color"])
        gap = int(pair["gap"])
        threshold = int(pair["threshold"])

        if pair["orientation"] == "horizontal":
            left = pair["rect_a"]
            right = pair["rect_b"]
            r0, _, r1, c1_left = left
            _, c0_right, _, _ = right
            cs = c1_left + 1
            ce = c0_right - 1
            if gap <= threshold:
                out[r0 : r1 + 1, cs : ce + 1] = color
            else:
                out[r0, cs : ce + 1] = color
                out[r1, cs : ce + 1] = color
        else:
            top = pair["rect_a"]
            bottom = pair["rect_b"]
            _, c0, r1_top, c1 = top
            r0_bottom = bottom[0]
            rs = r1_top + 1
            re = r0_bottom - 1
            if gap <= threshold:
                out[rs : re + 1, c0 : c1 + 1] = color
            else:
                out[rs : re + 1, c0] = color
                out[rs : re + 1, c1] = color

    return out


def _sample_spec(rng: random.Random, orientation: str, branch: str) -> dict[str, int | str]:
    if orientation == "horizontal":
        height = rng.randint(2 if branch == "wide" else 1, 5)
        left_w = rng.randint(2, 6)
        right_w = rng.randint(2, 6)
        if branch == "narrow":
            gap = rng.randint(1, height)
        else:
            gap = rng.randint(height + 1, height + 4)
        return {
            "orientation": orientation,
            "branch": branch,
            "height": height,
            "left_w": left_w,
            "right_w": right_w,
            "gap": gap,
            "bbox_h": height,
            "bbox_w": left_w + gap + right_w,
        }

    width = rng.randint(2 if branch == "wide" else 1, 5)
    top_h = rng.randint(2, 6)
    bottom_h = rng.randint(2, 6)
    if branch == "narrow":
        gap = rng.randint(1, width)
    else:
        gap = rng.randint(width + 1, width + 4)
    return {
        "orientation": orientation,
        "branch": branch,
        "width": width,
        "top_h": top_h,
        "bottom_h": bottom_h,
        "gap": gap,
        "bbox_h": top_h + gap + bottom_h,
        "bbox_w": width,
    }


def _paint_pair(grid: np.ndarray, top: int, left: int, color: int, spec: dict[str, int | str]) -> None:
    if spec["orientation"] == "horizontal":
        height = int(spec["height"])
        left_w = int(spec["left_w"])
        right_w = int(spec["right_w"])
        gap = int(spec["gap"])
        grid[top : top + height, left : left + left_w] = color
        right_c = left + left_w + gap
        grid[top : top + height, right_c : right_c + right_w] = color
        return

    width = int(spec["width"])
    top_h = int(spec["top_h"])
    bottom_h = int(spec["bottom_h"])
    gap = int(spec["gap"])
    grid[top : top + top_h, left : left + width] = color
    bottom_r = top + top_h + gap
    grid[bottom_r : bottom_r + bottom_h, left : left + width] = color


def _place_specs(
    rng: random.Random, grid: np.ndarray, colors: list[int], specs: list[dict[str, int | str]]
) -> bool:
    h, w = grid.shape
    blocked = np.zeros((h, w), dtype=bool)

    for color, spec in zip(colors, sorted(specs, key=lambda s: int(s["bbox_h"]) * int(s["bbox_w"]), reverse=True)):
        box_h = int(spec["bbox_h"])
        box_w = int(spec["bbox_w"])
        if box_h + 2 * EDGE_MARGIN > h or box_w + 2 * EDGE_MARGIN > w:
            return False

        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            top = rng.randint(EDGE_MARGIN, h - box_h - EDGE_MARGIN)
            left = rng.randint(EDGE_MARGIN, w - box_w - EDGE_MARGIN)
            rs = max(0, top - 1)
            re = min(h, top + box_h + 1)
            cs = max(0, left - 1)
            ce = min(w, left + box_w + 1)
            if blocked[rs:re, cs:ce].any():
                continue

            _paint_pair(grid, top, left, color, spec)
            blocked[rs:re, cs:ce] = True
            placed = True
            break

        if not placed:
            return False

    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pair_count = rng.randint(*PAIR_COUNT_RANGE)
    min_side = 10 + 4 * pair_count
    height = rng.randint(min_side, GRID_SIDE_MAX)
    width = rng.randint(min_side, GRID_SIDE_MAX)
    grid = np.full((height, width), BACKGROUND, dtype=int)

    branches = ["narrow", "wide"] + [rng.choice(("narrow", "wide")) for _ in range(pair_count - 2)]
    orientations = ["horizontal", "vertical"] + [rng.choice(("horizontal", "vertical")) for _ in range(pair_count - 2)]
    rng.shuffle(branches)
    rng.shuffle(orientations)

    specs = [_sample_spec(rng, orientation, branch) for orientation, branch in zip(orientations, branches)]
    colors = rng.sample(PALETTE, pair_count)

    if not _place_specs(rng, grid, colors, specs):
        return None

    pairs = _analyze_pairs(grid)
    if len(pairs) != pair_count:
        return None

    seen_branches = {
        "narrow" if int(pair["gap"]) <= int(pair["threshold"]) else "wide"
        for pair in pairs
    }
    seen_orientations = {str(pair["orientation"]) for pair in pairs}
    if seen_branches != {"narrow", "wide"}:
        return None
    if seen_orientations != {"horizontal", "vertical"}:
        return None

    if np.array_equal(main(grid), grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
