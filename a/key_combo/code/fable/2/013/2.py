from __future__ import annotations

from collections import Counter
from itertools import combinations
import random

import numpy as np


Box = tuple[int, int, int, int]
Rect = tuple[int, Box]

_COLORS = tuple(range(10))
_GEN_ATTEMPTS = 400


def _get_bg_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _box_area(box: Box) -> int:
    r0, c0, r1, c1 = box
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _strictly_contains(outer: Box, inner: Box) -> bool:
    return (
        outer[0] < inner[0]
        and outer[1] < inner[1]
        and outer[2] > inner[2]
        and outer[3] > inner[3]
    )


def _boxes_overlap(a: Box, b: Box) -> bool:
    return not (
        a[2] < b[0]
        or b[2] < a[0]
        or a[3] < b[1]
        or b[3] < a[1]
    )


def _shift_box(box: Box, dr: int, dc: int) -> Box:
    r0, c0, r1, c1 = box
    return (r0 + dr, c0 + dc, r1 + dr, c1 + dc)


def _union_box(boxes: list[Box]) -> Box:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _render_summary(colors: list[int]) -> np.ndarray:
    size = 2 * len(colors)
    out = np.empty((size, size), dtype=int)
    for inset, color in enumerate(colors):
        out[inset : size - inset, inset : size - inset] = int(color)
    return out


def _extract_rectangles(grid: np.ndarray) -> tuple[int, list[Rect]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg_color(g)
    rectangles: list[Rect] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        coords = np.argwhere(g == color)
        assert coords.shape == (2, 2), "each foreground color must appear exactly twice"
        rows = sorted(int(v) for v in coords[:, 0])
        cols = sorted(int(v) for v in coords[:, 1])
        assert rows[0] < rows[1] and cols[0] < cols[1], "anchor pair must be diagonal"
        rectangles.append((color, (rows[0], cols[0], rows[1], cols[1])))
    return bg, rectangles


def _find_unique_nested_chain(rectangles: list[Rect]) -> list[Rect]:
    n = len(rectangles)
    for size in range(n, 1, -1):
        chains: set[tuple[int, ...]] = set()
        for subset in combinations(range(n), size):
            ordered = tuple(
                sorted(subset, key=lambda idx: _box_area(rectangles[idx][1]), reverse=True)
            )
            if all(
                _strictly_contains(rectangles[ordered[i]][1], rectangles[ordered[i + 1]][1])
                for i in range(size - 1)
            ):
                chains.add(ordered)
        if chains:
            assert len(chains) == 1, "expected one maximal strict nesting chain"
            return [rectangles[idx] for idx in next(iter(chains))]
    raise AssertionError("expected at least one strict nesting relation")


def main(grid: np.ndarray) -> np.ndarray:
    """Compress the valid implicit rectangle chain into a concentric square-ring summary.

    Input structure:
    - The background is the most common color.
    - Every other color appears exactly twice as one diagonal anchor pair.
    - Each pair defines an unseen axis-aligned rectangle by its inclusive bounding box.

    Rule:
    - Reconstruct one implicit rectangle per foreground color.
    - Keep only the unique maximal set that forms a strict containment chain, dropping the
      overlapping but non-nested distractor rectangle.
    - Order the surviving rectangles from outermost to innermost and render a 2n x 2n
      nested square summary with one-pixel-thick rings and a 2x2 solid center.

    Key insight:
    - The real objects are the invisible bounding boxes induced by color-matched corner
      pairs; selecting by area alone fails because one overlapping rectangle is intentionally
      interleaved in the size order but not in the nesting order.
    """
    _bg, rectangles = _extract_rectangles(np.asarray(grid, dtype=int))
    chain = _find_unique_nested_chain(rectangles)
    colors = [color for color, _box in chain]
    return _render_summary(colors)


def _gen_chain_boxes(length: int) -> list[Box]:
    inner_h = random.randint(2, 4)
    inner_w = random.randint(2, 4)
    boxes = [(0, 0, inner_h - 1, inner_w - 1)]
    for _ in range(length - 1):
        r0, c0, r1, c1 = boxes[0]
        top = random.randint(1, 2)
        bottom = random.randint(1, 2)
        left = random.randint(1, 2)
        right = random.randint(1, 2)
        boxes.insert(0, (r0 - top, c0 - left, r1 + bottom, c1 + right))
    outer_r0 = min(box[0] for box in boxes)
    outer_c0 = min(box[1] for box in boxes)
    return [_shift_box(box, -outer_r0, -outer_c0) for box in boxes]


def _valid_distractor(box: Box, chain_boxes: list[Box]) -> bool:
    outer = chain_boxes[0]
    inner = chain_boxes[-1]
    area = _box_area(box)
    chain_areas = [_box_area(chain_box) for chain_box in chain_boxes]
    return (
        chain_areas[0] > area > chain_areas[-1]
        and _boxes_overlap(box, chain_boxes[1])
        and not _strictly_contains(box, inner)
        and all(
            not _strictly_contains(box, chain_box)
            and not _strictly_contains(chain_box, box)
            for chain_box in chain_boxes
        )
        and box != outer
    )


def _candidate_distractors_by_side(chain_boxes: list[Box]) -> dict[str, list[Box]]:
    outer = chain_boxes[0]
    r0o, c0o, r1o, c1o = outer
    candidates = {"top": [], "bottom": [], "left": [], "right": []}

    for protrusion in (1, 2):
        top_r0 = r0o - protrusion
        for r1 in range(r0o + 1, r1o):
            for c0 in range(c0o, c1o):
                for c1 in range(c0 + 1, c1o + 1):
                    box = (top_r0, c0, r1, c1)
                    if _valid_distractor(box, chain_boxes):
                        candidates["top"].append(box)

        bottom_r1 = r1o + protrusion
        for r0 in range(r0o, r1o):
            for c0 in range(c0o, c1o):
                for c1 in range(c0 + 1, c1o + 1):
                    box = (r0, c0, bottom_r1, c1)
                    if _valid_distractor(box, chain_boxes):
                        candidates["bottom"].append(box)

        left_c0 = c0o - protrusion
        for c1 in range(c0o + 1, c1o):
            for r0 in range(r0o, r1o):
                for r1 in range(r0 + 1, r1o + 1):
                    box = (r0, left_c0, r1, c1)
                    if _valid_distractor(box, chain_boxes):
                        candidates["left"].append(box)

        right_c1 = c1o + protrusion
        for c0 in range(c0o, c1o):
            for r0 in range(r0o, r1o):
                for r1 in range(r0 + 1, r1o + 1):
                    box = (r0, c0, r1, right_c1)
                    if _valid_distractor(box, chain_boxes):
                        candidates["right"].append(box)
    return candidates


def _corner_pair(box: Box, flipped: bool) -> tuple[tuple[int, int], tuple[int, int]]:
    r0, c0, r1, c1 = box
    if flipped:
        return (r0, c1), (r1, c0)
    return (r0, c0), (r1, c1)


def _assign_anchor_pairs(boxes: list[Box]) -> list[tuple[tuple[int, int], tuple[int, int]]] | None:
    for _ in range(64):
        pairs = [_corner_pair(box, bool(random.getrandbits(1))) for box in boxes]
        used: set[tuple[int, int]] = set()
        ok = True
        for pair in pairs:
            if pair[0] == pair[1] or pair[0] in used or pair[1] in used:
                ok = False
                break
            used.add(pair[0])
            used.add(pair[1])
        if ok:
            return pairs
    return None


def _place_boxes(boxes: list[Box]) -> tuple[tuple[int, int], list[Box]] | None:
    union = _union_box(boxes)
    union_h = union[2] - union[0] + 1
    union_w = union[3] - union[1] + 1
    if union_h > 30 or union_w > 30:
        return None

    extra_h = random.randint(0, 30 - union_h)
    extra_w = random.randint(0, 30 - union_w)
    height = union_h + extra_h
    width = union_w + extra_w
    dr = random.randint(0, extra_h) - union[0]
    dc = random.randint(0, extra_w) - union[1]
    return (height, width), [_shift_box(box, dr, dc) for box in boxes]


def _render_input(shape: tuple[int, int], bg: int, rectangles: list[Rect]) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    boxes = [box for _color, box in rectangles]
    anchor_pairs = _assign_anchor_pairs(boxes)
    if anchor_pairs is None:
        raise ValueError("failed to assign non-overlapping anchor pairs")
    for (color, _box), pair in zip(rectangles, anchor_pairs):
        for r, c in pair:
            grid[r, c] = int(color)
    return grid


def _semantic_checks(
    grid: np.ndarray,
    chain_rectangles: list[Rect],
    distractor: Rect,
) -> np.ndarray:
    bg = _get_bg_color(grid)
    expected = _render_summary([color for color, _box in chain_rectangles])
    values = Counter(map(int, grid.ravel()))
    assert values[bg] == max(values.values())
    for color, count in values.items():
        if color != bg:
            assert count == 2

    chain_boxes = [box for _color, box in chain_rectangles]
    distractor_box = distractor[1]
    assert all(_strictly_contains(chain_boxes[i], chain_boxes[i + 1]) for i in range(len(chain_boxes) - 1))
    assert any(_boxes_overlap(distractor_box, box) for box in chain_boxes)
    assert all(
        not _strictly_contains(distractor_box, box) and not _strictly_contains(box, distractor_box)
        for box in chain_boxes
    )
    chain_areas = [_box_area(box) for box in chain_boxes]
    assert chain_areas[0] > _box_area(distractor_box) > chain_areas[-1]
    assert np.array_equal(main(grid), expected)
    return expected


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(_COLORS)
    chain_len = random.randint(3, 6)
    palette = random.sample([color for color in _COLORS if color != bg], chain_len + 1)
    chain_colors = palette[:chain_len]
    distractor_color = palette[-1]

    chain_boxes = _gen_chain_boxes(chain_len)
    by_side = _candidate_distractors_by_side(chain_boxes)
    available_sides = [side for side, boxes in by_side.items() if boxes]
    if not available_sides:
        return None
    side = random.choice(available_sides)
    distractor_box = random.choice(by_side[side])

    placed = _place_boxes(chain_boxes + [distractor_box])
    if placed is None:
        return None
    shape, placed_boxes = placed

    chain_rectangles: list[Rect] = [
        (int(color), box) for color, box in zip(chain_colors, placed_boxes[:-1])
    ]
    distractor: Rect = (int(distractor_color), placed_boxes[-1])
    rectangles = chain_rectangles + [distractor]

    try:
        grid = _render_input(shape, bg, rectangles)
        _semantic_checks(grid, chain_rectangles, distractor)
    except (AssertionError, ValueError):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid implicit-rectangle case")
