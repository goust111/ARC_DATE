from __future__ import annotations

import itertools
import random
from collections import Counter

import numpy as np


Rect = tuple[int, int, int, int]  # inclusive (r0, c0, r1, c1)


def _bbox(mask: np.ndarray) -> Rect | None:
    """Small self-contained counterpart of arc_utils.bbox, using (row, col)."""
    coords = np.argwhere(mask)
    if not len(coords):
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rect_mask(shape: tuple[int, int], rect: Rect) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = rect
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _render(
    shape: tuple[int, int], bg: int, order: tuple[int, ...] | list[int], rects: dict[int, Rect]
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for color in order:  # bottom to top
        r0, c0, r1, c1 = rects[int(color)]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return out


def _area_rectangles(
    shape: tuple[int, int], visible_box: Rect, area: int
) -> list[tuple[Rect, np.ndarray]]:
    """Enumerate in-canvas, area-sized rectangles containing a visible bbox."""
    H, W = shape
    vr0, vc0, vr1, vc1 = visible_box
    found: list[tuple[Rect, np.ndarray]] = []
    for h in range(1, H + 1):
        if area % h:
            continue
        w = area // h
        if w > W or h < vr1 - vr0 + 1 or w < vc1 - vc0 + 1:
            continue
        for r0 in range(max(0, vr1 - h + 1), min(vr0, H - h) + 1):
            for c0 in range(max(0, vc1 - w + 1), min(vc0, W - w) + 1):
                rect = (r0, c0, r0 + h - 1, c0 + w - 1)
                found.append((rect, _rect_mask(shape, rect)))
    return found


def _solve(grid: np.ndarray, limit: int = 2) -> list[tuple[tuple[int, ...], dict[int, Rect], int]]:
    """Return up to ``limit`` exact equal-area rectangle-stack explanations."""
    g = np.asarray(grid, dtype=int)
    bg = int(Counter(map(int, g.ravel())).most_common(1)[0][0])
    colors = tuple(int(c) for c in np.unique(g) if int(c) != bg)
    assert 3 <= len(colors) <= 9, "expected at least three colored rectangles"

    visible = {color: g == color for color in colors}
    boxes = {color: _bbox(visible[color]) for color in colors}
    assert all(box is not None for box in boxes.values())
    solutions: list[tuple[tuple[int, ...], dict[int, Rect], int]] = []

    # The top rectangle is necessarily wholly visible, so its visible cells form
    # a solid rectangle and reveal the conserved area without color-id semantics.
    for top in colors:
        top_box = boxes[top]
        assert top_box is not None
        r0, c0, r1, c1 = top_box
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if int(visible[top].sum()) != area:
            continue

        candidate_cache = {
            color: _area_rectangles(g.shape, boxes[color], area)  # type: ignore[arg-type]
            for color in colors
        }
        top_mask = _rect_mask(g.shape, top_box)

        def descend(
            remaining: tuple[int, ...],
            top_down: list[int],
            rects: dict[int, Rect],
            higher_union: np.ndarray,
        ) -> None:
            if len(solutions) >= limit:
                return
            if not remaining:
                bottom_up = tuple(reversed(top_down))
                if np.array_equal(_render(g.shape, bg, bottom_up, rects), g):
                    solutions.append((bottom_up, dict(rects), area))
                return

            # Choose the next layer below all already chosen higher layers.
            for color in remaining:
                for rect, mask in candidate_cache[color]:
                    # Exactly the part outside higher rectangles must be visible.
                    if not np.array_equal(mask & ~higher_union, visible[color]):
                        continue
                    next_remaining = tuple(c for c in remaining if c != color)
                    rects[color] = rect
                    descend(
                        next_remaining,
                        top_down + [color],
                        rects,
                        higher_union | mask,
                    )
                    rects.pop(color)

        descend(
            tuple(c for c in colors if c != top),
            [top],
            {top: top_box},
            top_mask,
        )
        if len(solutions) >= limit:
            break
    return solutions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete equal-area occluded rectangles and redraw their stack in reversed depth order.

    Input structure:
    - A dominant uniform background surrounds at least three differently colored solid
      rectangles; overlaps hide parts of lower rectangles, while the top one is complete.
    - Every rectangle has the same true area, and the visible truncations determine one
      complete bottom-to-top depth order, including a region shared by three layers.

    Rule:
    - Read the common area from a wholly visible rectangle, enumerate complete rectangles
      of that area containing each color's visible cells, and retain the unique stack whose
      normal bottom-to-top rendering exactly reproduces the input.
    - Repaint those complete rectangles on the same background from the original top layer
      down to the original bottom layer, thereby reversing the entire depth order.

    Key insight:
    - Occluded visible fragments alone permit several rectangular completions; equal true
      area removes that ambiguity, after which occlusion recovers a global order that must
      be reversed as a whole rather than altered locally at pairwise overlaps.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    solutions = _solve(g, limit=2)
    assert len(solutions) == 1, "input must have one equal-area stack explanation"
    order, rects, _area = solutions[0]
    bg = int(Counter(map(int, g.ravel())).most_common(1)[0][0])
    return _render(g.shape, bg, tuple(reversed(order)), rects)


def _all_completion_candidates(
    visible: np.ndarray, higher_union: np.ndarray
) -> set[Rect]:
    """All geometrically valid completions, deliberately ignoring equal area."""
    box = _bbox(visible)
    assert box is not None
    H, W = visible.shape
    vr0, vc0, vr1, vc1 = box
    candidates: set[Rect] = set()
    for r0 in range(vr0 + 1):
        for c0 in range(vc0 + 1):
            for r1 in range(vr1, H):
                for c1 in range(vc1, W):
                    mask = _rect_mask(visible.shape, (r0, c0, r1, c1))
                    if np.array_equal(mask & ~higher_union, visible):
                        candidates.add((r0, c0, r1, c1))
    return candidates


def _semantic_checks(
    inp: np.ndarray,
    expected: np.ndarray,
    bg: int,
    order: tuple[int, ...],
    rects: dict[int, Rect],
) -> bool:
    """Check uniqueness, conservation dependence, depth dependence, and non-degeneracy."""
    solved = _solve(inp, limit=2)
    if len(solved) != 1:
        return False
    got_order, got_rects, area = solved[0]
    if got_order != order or got_rects != rects or not np.array_equal(main(inp), expected):
        return False
    if np.array_equal(inp, expected):
        return False

    masks = {c: _rect_mask(inp.shape, rects[c]) for c in order}
    common = np.logical_and.reduce([masks[c] for c in order])
    # A genuine three-plus-layer overlap must change from original top to bottom color.
    if not common.any() or not np.all(inp[common] == order[-1]) or not np.all(expected[common] == order[0]):
        return False

    higher = np.zeros(inp.shape, dtype=bool)
    area_is_essential = False
    for color in reversed(order):
        visible = inp == color
        all_candidates = _all_completion_candidates(visible, higher)
        equal_area = {
            rect
            for rect in all_candidates
            if (rect[2] - rect[0] + 1) * (rect[3] - rect[1] + 1) == area
        }
        if equal_area != {rects[color]}:
            return False
        if len(all_candidates) >= 2 and any(
            (r1 - r0 + 1) * (c1 - c0 + 1) != area for r0, c0, r1, c1 in all_candidates
        ):
            area_is_essential = True
        higher |= masks[color]
    if not area_is_essential:
        return False

    # Reversing only visible fragments cannot reproduce newly exposed hidden colors.
    if not any(np.any((expected == c) & (inp != c)) for c in order[:-1]):
        return False
    return int(np.count_nonzero(inp != bg)) < int(inp.size * 0.55)


def _random_rect_containing(
    H: int, W: int, h: int, w: int, point: tuple[int, int], rng: random.Random
) -> Rect | None:
    pr, pc = point
    r0s = list(range(max(1, pr - h + 1), min(pr, H - h - 1) + 1))
    c0s = list(range(max(1, pc - w + 1), min(pc, W - w - 1) + 1))
    if not r0s or not c0s:
        return None
    r0, c0 = rng.choice(r0s), rng.choice(c0s)
    return r0, c0, r0 + h - 1, c0 + w - 1


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    H, W = rng.randint(15, 23), rng.randint(15, 23)
    count = rng.choice((3, 4, 5))
    area = rng.choice((18, 20, 24, 30))
    dims = [(h, area // h) for h in range(2, 11) if area % h == 0 and area // h <= 10]
    if len(dims) < 2:
        return None

    common_point = (rng.randint(5, H - 6), rng.randint(5, W - 6))
    chosen_rects: list[Rect] = []
    for _ in range(count):
        h, w = rng.choice(dims)
        rect = _random_rect_containing(H, W, h, w, common_point, rng)
        if rect is None or rect in chosen_rects:
            return None
        chosen_rects.append(rect)

    # Keep the common overlap small enough that the stack remains visually readable.
    intersection = np.logical_and.reduce([_rect_mask((H, W), r) for r in chosen_rects])
    if not (1 <= int(intersection.sum()) <= 6):
        return None

    palette = list(range(10))
    bg = rng.choice(palette)
    palette.remove(bg)
    colors = rng.sample(palette, count)
    order = tuple(colors)  # sampled color assignment makes this a random depth order
    rects = dict(zip(order, chosen_rects))
    inp = _render((H, W), bg, order, rects)
    expected = _render((H, W), bg, tuple(reversed(order)), rects)

    # Every layer must leave a visible trace, and background must be unambiguous.
    if any(not np.any(inp == color) for color in order):
        return None
    if Counter(map(int, inp.ravel())).most_common(1)[0][0] != bg:
        return None
    if not _semantic_checks(inp, expected, bg, order, rects):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(1200):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to create a unique non-degenerate stack")
