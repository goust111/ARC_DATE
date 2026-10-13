from __future__ import annotations

import random
from itertools import combinations

import numpy as np


BLACK = 0
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 500
ORDER_ATTEMPTS = 40


def _bbox_of_color(grid: np.ndarray, color: int) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == int(color))
    assert len(coords) > 0, f"missing color {color}"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _frame_area(frame: dict) -> int:
    r0, c0, r1, c1 = frame["bbox"]
    return int((r1 - r0 - 1) * (c1 - c0 - 1))


def _on_border(frame: dict, r: int, c: int) -> bool:
    r0, c0, r1, c1 = frame["bbox"]
    return r0 <= r <= r1 and c0 <= c <= c1 and (r == r0 or r == r1 or c == c0 or c == c1)


def _in_interior(frame: dict, r: int, c: int) -> bool:
    r0, c0, r1, c1 = frame["bbox"]
    return r0 < r < r1 and c0 < c < c1


def _side_hits(mask: np.ndarray) -> tuple[bool, bool, bool, bool]:
    return bool(mask[0].any()), bool(mask[-1].any()), bool(mask[:, 0].any()), bool(mask[:, -1].any())


def _infer_frame(grid: np.ndarray, color: int) -> dict:
    r0, c0, r1, c1 = _bbox_of_color(grid, color)
    assert 0 < r0 < r1 < grid.shape[0] - 1, "frame must stay away from top/bottom edges"
    assert 0 < c0 < c1 < grid.shape[1] - 1, "frame must stay away from left/right edges"
    assert r1 - r0 + 1 >= 3 and c1 - c0 + 1 >= 3, "frame must have a nonempty interior"

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    mask = sub == int(color)
    border = np.zeros_like(mask, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    assert np.all(mask <= border), "visible frame pixels must lie on the inferred perimeter"
    top, bottom, left, right = _side_hits(mask)
    assert top and bottom and left and right, "each frame side must remain visible after overlaps"

    frame = {"color": int(color), "bbox": (r0, c0, r1, c1)}
    frame["area"] = _frame_area(frame)
    return frame


def _render(frames: list[dict], shape: tuple[int, int], *, smallest_wins: bool) -> np.ndarray:
    h, w = shape
    out = np.zeros((h, w), dtype=int)
    sign = 1 if smallest_wins else -1
    ranked = sorted(frames, key=lambda frame: (sign * int(frame["area"]), int(frame["color"])))

    for r in range(h):
        for c in range(w):
            border_hits = [frame for frame in ranked if _on_border(frame, r, c)]
            if border_hits:
                out[r, c] = int(border_hits[0]["color"])
                continue

            interior_hits = [frame for frame in ranked if _in_interior(frame, r, c)]
            if interior_hits:
                out[r, c] = int(interior_hits[0]["color"])

    return out


def _render_last_draw(frames: list[dict], shape: tuple[int, int], order: tuple[int, int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for idx in order:
        frame = frames[int(idx)]
        color = int(frame["color"])
        r0, c0, r1, c1 = frame["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = color
    return out


def _draw_input(frames: list[dict], shape: tuple[int, int], order: tuple[int, int, int]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for idx in order:
        frame = frames[int(idx)]
        color = int(frame["color"])
        r0, c0, r1, c1 = frame["bbox"]
        grid[r0, c0 : c1 + 1] = color
        grid[r1, c0 : c1 + 1] = color
        grid[r0 : r1 + 1, c0] = color
        grid[r0 : r1 + 1, c1] = color
    return grid


def _analyze_layout(frames: list[dict], shape: tuple[int, int]) -> dict:
    h, w = shape
    shared_border = 0
    shared_interior = 0
    border_interior = 0
    overlap_winners: set[int] = set()
    overlap_pairs: set[tuple[int, int]] = set()

    for r in range(h):
        for c in range(w):
            border_hits = [i for i, frame in enumerate(frames) if _on_border(frame, r, c)]
            interior_hits = [i for i, frame in enumerate(frames) if _in_interior(frame, r, c)]

            if len(border_hits) >= 2:
                shared_border += 1
            if len(border_hits) == 1 and interior_hits:
                border_interior += 1
            if len(interior_hits) >= 2 and not border_hits:
                shared_interior += 1
                winner = min(interior_hits, key=lambda idx: int(frames[idx]["area"]))
                overlap_winners.add(int(frames[winner]["color"]))
                for a, b in combinations(interior_hits, 2):
                    overlap_pairs.add(tuple(sorted((int(a), int(b)))))

    return {
        "shared_border": shared_border,
        "shared_interior": shared_interior,
        "border_interior": border_interior,
        "overlap_winners": overlap_winners,
        "overlap_pairs": overlap_pairs,
    }


def _disjoint(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _make_frame(color: int, bbox: tuple[int, int, int, int]) -> dict:
    frame = {"color": int(color), "bbox": tuple(map(int, bbox))}
    frame["area"] = _frame_area(frame)
    return frame


def _sample_colors() -> list[int]:
    return random.sample(PALETTE, 3)


def _try_make_nested_case() -> tuple[list[dict], tuple[int, int]] | None:
    h = random.randint(12, 16)
    w = random.randint(12, 16)

    for _ in range(120):
        large_h = random.randint(8, min(11, h - 2))
        large_w = random.randint(8, min(11, w - 2))
        large_r0 = random.randint(1, h - large_h - 1)
        large_c0 = random.randint(1, w - large_w - 1)
        large = (large_r0, large_c0, large_r0 + large_h - 1, large_c0 + large_w - 1)

        colors = _sample_colors()
        large_frame = _make_frame(colors[0], large)

        inner_r0_min = large[0] + 1
        inner_r1_max = large[2] - 1
        inner_c0_min = large[1] + 1
        inner_c1_max = large[3] - 1
        inner_h = inner_r1_max - inner_r0_min + 1
        inner_w = inner_c1_max - inner_c0_min + 1
        if inner_h < 8 or inner_w < 8:
            continue

        for _ in range(120):
            small_h = random.randint(4, min(6, inner_h - 2))
            small_w = random.randint(4, min(6, inner_w - 2))
            medium_h = random.randint(4, min(7, inner_h - 2))
            medium_w = random.randint(4, min(7, inner_w - 2))
            small_area = (small_h - 2) * (small_w - 2)
            medium_area = (medium_h - 2) * (medium_w - 2)
            if small_area == medium_area:
                continue

            small_r0 = random.randint(inner_r0_min, inner_r1_max - small_h + 1)
            small_c0 = random.randint(inner_c0_min, inner_c1_max - small_w + 1)
            medium_r0 = random.randint(inner_r0_min, inner_r1_max - medium_h + 1)
            medium_c0 = random.randint(inner_c0_min, inner_c1_max - medium_w + 1)
            small = (small_r0, small_c0, small_r0 + small_h - 1, small_c0 + small_w - 1)
            medium = (medium_r0, medium_c0, medium_r0 + medium_h - 1, medium_c0 + medium_w - 1)

            if not _disjoint(small, medium, gap=1):
                continue

            frames = [
                large_frame,
                _make_frame(colors[1], small),
                _make_frame(colors[2], medium),
            ]
            areas = [int(frame["area"]) for frame in frames]
            if len(set(areas)) != 3:
                continue

            analysis = _analyze_layout(frames, (h, w))
            if analysis["shared_border"] != 0:
                continue
            if analysis["shared_interior"] <= 0:
                continue
            if analysis["border_interior"] <= 0:
                continue
            if len(analysis["overlap_winners"]) < 2:
                continue
            return frames, (h, w)
    return None


def _try_make_cross_case() -> tuple[list[dict], tuple[int, int]] | None:
    h = random.randint(12, 16)
    w = random.randint(12, 16)

    for _ in range(200):
        colors = _sample_colors()

        large_h = random.randint(8, min(10, h - 4))
        large_w = random.randint(8, min(10, w - 4))
        large_r0 = random.randint(1, h - large_h - 3)
        large_c0 = random.randint(1, w - large_w - 3)
        large = (large_r0, large_c0, large_r0 + large_h - 1, large_c0 + large_w - 1)
        lr0, lc0, lr1, lc1 = large

        if lr1 - lr0 < 5 or lc1 - lc0 < 5:
            continue

        medium_r0 = random.randint(lr0 + 1, lr1 - 4)
        medium_c0 = random.randint(lc0 + 1, lc1 - 4)
        if min(h - 2, medium_r0 + 8) < lr1 + 1 or min(w - 2, medium_c0 + 8) < lc1 + 1:
            continue
        medium_r1 = random.randint(lr1 + 1, min(h - 2, medium_r0 + 8))
        medium_c1 = random.randint(lc1 + 1, min(w - 2, medium_c0 + 8))
        medium = (medium_r0, medium_c0, medium_r1, medium_c1)

        overlap_r0 = max(lr0 + 1, medium_r0 + 1)
        overlap_r1 = min(lr1 - 1, medium_r1 - 1)
        overlap_c0 = max(lc0 + 1, medium_c0 + 1)
        overlap_c1 = min(lc1 - 1, medium_c1 - 1)
        if overlap_r1 - overlap_r0 + 1 < 4 or overlap_c1 - overlap_c0 + 1 < 4:
            continue

        small_h = random.randint(4, min(6, overlap_r1 - overlap_r0 + 1))
        small_w = random.randint(4, min(6, overlap_c1 - overlap_c0 + 1))
        small_r0 = random.randint(overlap_r0, overlap_r1 - small_h + 1)
        small_c0 = random.randint(overlap_c0, overlap_c1 - small_w + 1)
        small = (small_r0, small_c0, small_r0 + small_h - 1, small_c0 + small_w - 1)

        frames = [
            _make_frame(colors[0], large),
            _make_frame(colors[1], medium),
            _make_frame(colors[2], small),
        ]
        areas = [int(frame["area"]) for frame in frames]
        if len(set(areas)) != 3:
            continue

        analysis = _analyze_layout(frames, (h, w))
        if analysis["shared_border"] <= 0:
            continue
        if analysis["shared_interior"] <= 0:
            continue
        if analysis["border_interior"] <= 0:
            continue
        if len(analysis["overlap_winners"]) < 2:
            continue
        return frames, (h, w)
    return None


def _pick_input_order(frames: list[dict], shape: tuple[int, int], mode: str) -> tuple[int, int, int] | None:
    target = _render(frames, shape, smallest_wins=True)
    target_max = _render(frames, shape, smallest_wins=False)
    areas = [int(frame["area"]) for frame in frames]

    for _ in range(ORDER_ATTEMPTS):
        order = list(range(3))
        random.shuffle(order)

        if mode == "cross":
            large_idx, small_idx = sorted(range(3), key=lambda idx: areas[idx])[-1], sorted(range(3), key=lambda idx: areas[idx])[0]
            if order.index(large_idx) < order.index(small_idx):
                order[order.index(large_idx)], order[order.index(small_idx)] = order[order.index(small_idx)], order[order.index(large_idx)]

        order_tuple = tuple(order)
        inp = _draw_input(frames, shape, order_tuple)
        if not np.array_equal(target, main(inp)):
            continue

        if any(_infer_frame(inp, frame["color"])["bbox"] != frame["bbox"] for frame in frames):
            continue

        last_draw = _render_last_draw(frames, shape, order_tuple)
        if np.array_equal(target, target_max):
            continue
        if np.array_equal(target, last_draw):
            continue

        if mode == "cross":
            diff_on_shared_border = False
            for r in range(shape[0]):
                for c in range(shape[1]):
                    if sum(_on_border(frame, r, c) for frame in frames) >= 2 and inp[r, c] != target[r, c]:
                        diff_on_shared_border = True
                        break
                if diff_on_shared_border:
                    break
            if not diff_on_shared_border:
                continue

        return order_tuple

    return None


def _generate_case() -> np.ndarray | None:
    mode = random.choice(("nested", "cross"))
    builder = _try_make_nested_case if mode == "nested" else _try_make_cross_case
    built = builder()
    if built is None:
        return None

    frames, shape = built
    order = _pick_input_order(frames, shape, mode)
    if order is None:
        return None

    grid = _draw_input(frames, shape, order)
    analysis = _analyze_layout(frames, shape)
    assert analysis["shared_interior"] > 0
    assert len(analysis["overlap_winners"]) >= 2
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill every overlapping hollow frame interior, giving each conflict to the smallest-area frame.

    Input structure:
    - The grid has a black background and exactly three single-color hollow rectangular frames.
    - Each frame is axis-aligned, one cell thick, fully inside the grid, and has a unique border color.
    - Frame borders may overlap at a cell, and one frame's border may pass through another frame's interior.

    Rule:
    - Infer each frame from the bounding box of its visible color pixels and measure its interior area as the
      number of cells strictly inside that rectangle.
    - Keep border cells as borders; if multiple frame borders claim the same cell, use the color of the
      frame with the smallest interior area.
    - For every non-border cell that lies inside one or more frame interiors, color it with the frame color
      whose interior area is smallest among the interiors covering that cell.
    - Leave all cells outside every border and every interior black.

    Key insight:
    - The visible colors identify three rectangles, but overlap resolution is not based on draw order or the
      largest container: every contested cell is assigned by the smallest enclosed area.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must stay within 30x30"

    colors = sorted(int(color) for color in np.unique(g) if int(color) != BLACK)
    assert len(colors) == 3, "expected exactly three frame colors"

    frames = [_infer_frame(g, color) for color in colors]
    areas = [int(frame["area"]) for frame in frames]
    assert len(set(areas)) == 3, "frame interior areas must be pairwise distinct"

    analysis = _analyze_layout(frames, g.shape)
    assert analysis["shared_interior"] > 0, "at least one interior overlap is required"
    return _render(frames, g.shape, smallest_wins=True)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid overlapping-frame input")
