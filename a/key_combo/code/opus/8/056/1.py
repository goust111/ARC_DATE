from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
AREA_TO_DIMS: dict[int, list[tuple[int, int]]] = {
    6: [(2, 3), (3, 2)],
    8: [(2, 4), (4, 2)],
    9: [(3, 3)],
    10: [(2, 5), (5, 2)],
    12: [(3, 4), (4, 3)],
    15: [(3, 5), (5, 3)],
    16: [(4, 4)],
    20: [(4, 5), (5, 4)],
    25: [(5, 5)],
}
AREA_OPTIONS = sorted(AREA_TO_DIMS)
GEN_ATTEMPTS = 600


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _box_from_rc_hw(r: int, c: int, h: int, w: int) -> tuple[int, int, int, int]:
    return int(r), int(c), int(r + h - 1), int(c + w - 1)


def _intersection(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> tuple[int, int, int]:
    r0 = max(box_a[0], box_b[0])
    c0 = max(box_a[1], box_b[1])
    r1 = min(box_a[2], box_b[2])
    c1 = min(box_a[3], box_b[3])
    if r0 > r1 or c0 > c1:
        return 0, 0, 0
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    return h * w, h, w


def _touch_or_overlap(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int], margin: int = 0) -> bool:
    return not (
        box_a[2] + margin < box_b[0]
        or box_b[2] + margin < box_a[0]
        or box_a[3] + margin < box_b[1]
        or box_b[3] + margin < box_a[1]
    )


def _draw_rect(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _build_output(rects: list[dict]) -> np.ndarray:
    ordered = sorted(rects, key=lambda rect: (rect["area"], rect["box"][0], rect["box"][1]))
    out_h = max(rect["h"] for rect in ordered)
    out_w = sum(rect["w"] for rect in ordered)
    out = np.full((out_h, out_w), BG, dtype=int)
    col = 0
    for rect in ordered:
        r0 = out_h - rect["h"]
        out[r0:, col : col + rect["w"]] = rect["color"]
        col += rect["w"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Restore each occluded rectangle, then stack them left-to-right by true area.

    Input structure:
    - The grid has a black background and 3-5 distinct non-black colors.
    - Each color belongs to exactly one axis-aligned filled rectangle, but later rectangles may cover parts of earlier ones.
    - Every rectangle still shows enough edge pixels that the bbox of its visible cells is the rectangle's original full extent.

    Rule:
    - For each non-background color, take the bbox of all visible cells of that color and treat that whole bbox as the restored full rectangle.
    - Compute each restored rectangle's true area from that bbox width and height.
    - Sort the restored rectangles by true area from smallest to largest.
    - On a fresh canvas whose height is the maximum rectangle height and whose width is the sum of rectangle widths, place the sorted full rectangles tightly from left to right with their bottom edges aligned.

    Key insight:
    - Visible pixel count is not the sorting key; the hidden cells must be restored first from each color's bbox, and only that restored area determines the final order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    rects: list[dict] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != BG):
        box = _bbox(g == color)
        assert box is not None
        r0, c0, r1, c1 = box
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        rects.append(
            {
                "color": int(color),
                "box": box,
                "h": h,
                "w": w,
                "area": h * w,
            }
        )

    assert 3 <= len(rects) <= 5
    return _build_output(rects)


def _sample_rect_specs(count: int) -> list[dict]:
    while True:
        window = random.randint(count, min(len(AREA_OPTIONS), count + 2))
        start = random.randint(0, len(AREA_OPTIONS) - window)
        areas = sorted(random.sample(AREA_OPTIONS[start : start + window], count))
        if min(np.diff(areas)) <= 5:
            break

    colors = random.sample(PALETTE, count)
    specs = []
    for color, area in zip(colors, areas):
        h, w = random.choice(AREA_TO_DIMS[area])
        specs.append({"color": color, "area": area, "h": h, "w": w})
    return specs


def _candidate_boxes(
    shape: tuple[int, int],
    h: int,
    w: int,
    placed: list[dict],
    target: dict | None,
) -> list[tuple[float, tuple[int, int, int, int]]]:
    H, W = shape
    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    for r in range(H - h + 1):
        for c in range(W - w + 1):
            box = _box_from_rc_hw(r, c, h, w)
            if target is None:
                if any(_touch_or_overlap(box, other["box"], margin=1) for other in placed):
                    continue
                center_bias = abs((r + h / 2) - H / 2) + abs((c + w / 2) - W / 2)
                candidates.append((-center_bias, box))
                continue

            overlap_area, overlap_h, overlap_w = _intersection(box, target["box"])
            if overlap_area == 0:
                continue
            if overlap_h >= min(h, target["h"]) or overlap_w >= min(w, target["w"]):
                continue
            blocked = False
            for other in placed:
                if other is target:
                    continue
                if _touch_or_overlap(box, other["box"], margin=1):
                    blocked = True
                    break
            if blocked:
                continue
            desired = min(target["area"] / 2 - 0.75, h * w * 0.75)
            score = -abs(overlap_area - desired)
            candidates.append((score, box))
    return candidates


def _place_scene(specs: list[dict], H: int, W: int) -> list[dict] | None:
    placed: list[dict] = []
    order = list(sorted(range(len(specs)), key=lambda i: specs[i]["area"], reverse=True))
    overlap_steps = set(random.sample(range(1, len(order)), random.randint(1, min(2, len(order) - 1))))

    for step, idx in enumerate(order):
        spec = dict(specs[idx])
        h = spec["h"]
        w = spec["w"]
        if step == 0:
            candidates = _candidate_boxes((H, W), h, w, placed, None)
        else:
            candidates = []
            if step in overlap_steps:
                targets = placed[:]
                random.shuffle(targets)
                targets.sort(key=lambda rect: rect["area"], reverse=True)
                for target in targets:
                    target_candidates = _candidate_boxes((H, W), h, w, placed, target)
                    if target_candidates:
                        candidates = target_candidates
                        break
            if not candidates:
                candidates = _candidate_boxes((H, W), h, w, placed, None)
        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0], reverse=True)
        _, box = random.choice(candidates[: min(12, len(candidates))])
        spec["box"] = box
        placed.append(spec)
    return placed


def _render_input(rects: list[dict], shape: tuple[int, int]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for rect in rects:
        _draw_rect(grid, rect["box"], rect["color"])
    return grid


def _scene_valid(inp: np.ndarray, rects: list[dict]) -> bool:
    overlap_pairs = 0
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            if _intersection(rects[i]["box"], rects[j]["box"])[0] > 0:
                overlap_pairs += 1
    if overlap_pairs == 0:
        return False

    infos = []
    for idx, rect in enumerate(rects):
        mask = inp == rect["color"]
        box = _bbox(mask)
        if box != rect["box"]:
            return False
        visible = int(mask.sum())
        if visible * 2 < rect["area"]:
            return False
        infos.append(
            {
                "idx": idx,
                "area": rect["area"],
                "visible": visible,
                "box": rect["box"],
            }
        )

    if not any(info["visible"] < info["area"] for info in infos):
        return False
    visible_counts = [info["visible"] for info in infos]
    if len(set(visible_counts)) != len(visible_counts):
        return False

    true_order = tuple(info["idx"] for info in sorted(infos, key=lambda info: info["area"]))
    visible_order = tuple(
        info["idx"] for info in sorted(infos, key=lambda info: (info["visible"], info["box"][0], info["box"][1]))
    )
    row_major_order = tuple(info["idx"] for info in sorted(infos, key=lambda info: (info["box"][0], info["box"][1])))
    left_to_right_order = tuple(info["idx"] for info in sorted(infos, key=lambda info: (info["box"][1], info["box"][0])))

    if visible_order == true_order:
        return False
    if row_major_order == true_order or left_to_right_order == true_order:
        return False

    expected = _build_output(rects)
    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False
    return actual.shape != inp.shape or not np.array_equal(actual, inp)


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(14, 18)
    W = random.randint(14, 18)
    count = random.randint(3, 5)
    specs = _sample_rect_specs(count)
    placed = _place_scene(specs, H, W)
    if placed is None:
        return None
    inp = _render_input(placed, (H, W))
    if _scene_valid(inp, placed):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
