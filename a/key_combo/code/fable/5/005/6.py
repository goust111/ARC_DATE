"""ARC task: mutually length-referenced rectangle extensions."""

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BBox = tuple[int, int, int, int]


def _background(grid: np.ndarray) -> int:
    """Return the dominant color, the conventional ARC background cue."""
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    """Find monochromatic 4-connected components using (row, column) coordinates."""
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []

    for sr in range(h):
        for sc in range(w):
            color = int(grid[sr, sc])
            if color == bg or seen[sr, sc]:
                continue

            seen[sr, sc] = True
            queue = deque([(sr, sc)])
            coords: list[tuple[int, int]] = []
            while queue:
                r, c = queue.popleft()
                coords.append((r, c))
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if (
                        0 <= rr < h
                        and 0 <= cc < w
                        and not seen[rr, cc]
                        and int(grid[rr, cc]) == color
                    ):
                        seen[rr, cc] = True
                        queue.append((rr, cc))

            rows = [r for r, _ in coords]
            cols = [c for _, c in coords]
            box: BBox = (min(rows), min(cols), max(rows), max(cols))
            r0, c0, r1, c1 = box
            assert len(coords) == (r1 - r0 + 1) * (c1 - c0 + 1), (
                "every foreground object must be a solid rectangle"
            )
            assert bool(np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color))
            objects.append({"bbox": box, "color": color})

    return objects


def _pair_relation(a: dict, b: dict) -> str | None:
    """Return H/V exactly when two rectangles satisfy the stated pairing cue."""
    ar0, ac0, ar1, ac1 = a["bbox"]
    br0, bc0, br1, bc1 = b["bbox"]

    if (ar0, ar1) == (br0, br1):
        left, right = (a, b) if ac1 < bc0 else (b, a) if bc1 < ac0 else (None, None)
        if left is not None:
            _lr0, lc0, _lr1, lc1 = left["bbox"]
            _rr0, rc0, _rr1, rc1 = right["bbox"]
            gap = rc0 - lc1 - 1
            left_len = lc1 - lc0 + 1
            right_len = rc1 - rc0 + 1
            if gap == left_len + right_len:
                return "H"

    if (ac0, ac1) == (bc0, bc1):
        top, bottom = (a, b) if ar1 < br0 else (b, a) if br1 < ar0 else (None, None)
        if top is not None:
            tr0, _tc0, tr1, _tc1 = top["bbox"]
            br0, _bc0, br1, _bc1 = bottom["bbox"]
            gap = br0 - tr1 - 1
            top_len = tr1 - tr0 + 1
            bottom_len = br1 - br0 + 1
            if gap == top_len + bottom_len:
                return "V"

    return None


def _analyze(grid: np.ndarray) -> tuple[int, list[tuple[dict, dict, str]]]:
    """Extract a unique perfect matching of valid rectangle pairs."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _background(g)
    objects = _components(g, bg)
    assert len(objects) >= 2 and len(objects) % 2 == 0

    candidates: list[list[tuple[int, str]]] = [[] for _ in objects]
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            if objects[i]["color"] == objects[j]["color"]:
                continue
            relation = _pair_relation(objects[i], objects[j])
            if relation is not None:
                candidates[i].append((j, relation))
                candidates[j].append((i, relation))

    assert all(len(options) == 1 for options in candidates), (
        "each rectangle must have exactly one partner determined by alignment and gap"
    )
    pairs: list[tuple[dict, dict, str]] = []
    for i, options in enumerate(candidates):
        j, relation = options[0]
        assert candidates[j] == [(i, relation)], "pairing must be mutual and unique"
        if i < j:
            pairs.append((objects[i], objects[j], relation))
    assert len(pairs) * 2 == len(objects)
    return bg, pairs


def _apply_pairs(grid: np.ndarray, pairs: list[tuple[dict, dict, str]]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for a, b, relation in pairs:
        if relation == "H":
            left, right = sorted((a, b), key=lambda obj: obj["bbox"][1])
            r0, lc0, r1, lc1 = left["bbox"]
            _r0, rc0, _r1, rc1 = right["bbox"]
            left_len = lc1 - lc0 + 1
            right_len = rc1 - rc0 + 1

            # Each side uses the *other* rectangle's axial length.
            out[r0 : r1 + 1, lc1 + 1 : lc1 + 1 + right_len] = left["color"]
            out[r0 : r1 + 1, rc0 - left_len : rc0] = right["color"]
        else:
            top, bottom = sorted((a, b), key=lambda obj: obj["bbox"][0])
            tr0, c0, tr1, c1 = top["bbox"]
            br0, _c0, br1, _c1 = bottom["bbox"]
            top_len = tr1 - tr0 + 1
            bottom_len = br1 - br0 + 1

            out[tr1 + 1 : tr1 + 1 + bottom_len, c0 : c1 + 1] = top["color"]
            out[br0 - top_len : br0, c0 : c1 + 1] = bottom["color"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Extend every facing rectangle toward its partner by the partner's axial body length.

    Input structure:
    - A dominant background contains several pairs of differently colored solid rectangles.
    - Partners have exactly the same perpendicular span and face one another horizontally or vertically.
    - Their background gap equals the sum of their two body lengths along that axis; these visible cues give every rectangle exactly one partner.

    Rule:
    - A left (top) rectangle advances toward its partner by the right (bottom) rectangle's axial body length.
    - The right (bottom) rectangle advances oppositely by the left (top) rectangle's axial body length, across the rectangle's full perpendicular span.
    - The two colored extensions are disjoint and together fill the complete inter-rectangle gap.

    Key insight:
    - The two lengths are cross-referenced: each rectangle's advance is controlled by its partner, so an unequal pair rules out both halving the gap and copying one's own length.
    """
    g = np.asarray(grid, dtype=int)
    _bg, pairs = _analyze(g)
    return _apply_pairs(g, pairs)


def _expanded_box(box: BBox, pad: int, shape: tuple[int, int]) -> BBox:
    r0, c0, r1, c1 = box
    h, w = shape
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _box_is_free(mask: np.ndarray, box: BBox, pad: int = 1) -> bool:
    r0, c0, r1, c1 = _expanded_box(box, pad, mask.shape)
    return not bool(mask[r0 : r1 + 1, c0 : c1 + 1].any())


def _try_generate() -> tuple[np.ndarray, list[dict]] | None:
    """Lay out separated complete pair corridors, then retain only endpoint bodies."""
    h = random.randint(20, 30)
    w = random.randint(20, 30)
    bg = random.randrange(10)
    grid = np.full((h, w), bg, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)
    pair_count = random.choice((2, 2, 3, 3, 4))

    # Every generated puzzle exercises both coordinate branches.  Unequal lengths
    # are mandatory; an occasional equal pair remains as an explicit decoy case.
    orientations = ["H", "V"] + [random.choice(("H", "V")) for _ in range(pair_count - 2)]
    random.shuffle(orientations)
    specs: list[dict] = []

    for pair_index, orientation in enumerate(orientations):
        placed = False
        for _ in range(250):
            if pair_index == 0 or random.random() < 0.78:
                a_len, b_len = random.sample((1, 2, 3, 4), 2)
            else:
                a_len = b_len = random.randint(1, 4)
            perp = random.randint(1, 3)
            corridor_len = 2 * (a_len + b_len)

            if orientation == "H":
                if corridor_len + 2 > w or perp + 2 > h:
                    continue
                r0 = random.randint(1, h - perp - 1)
                c0 = random.randint(1, w - corridor_len - 1)
                footprint = (r0, c0, r0 + perp - 1, c0 + corridor_len - 1)
                if not _box_is_free(reserved, footprint, pad=1):
                    continue
                left_box = (r0, c0, r0 + perp - 1, c0 + a_len - 1)
                right_c0 = c0 + a_len + (a_len + b_len)
                right_box = (r0, right_c0, r0 + perp - 1, right_c0 + b_len - 1)
                boxes = (left_box, right_box)
            else:
                if corridor_len + 2 > h or perp + 2 > w:
                    continue
                r0 = random.randint(1, h - corridor_len - 1)
                c0 = random.randint(1, w - perp - 1)
                footprint = (r0, c0, r0 + corridor_len - 1, c0 + perp - 1)
                if not _box_is_free(reserved, footprint, pad=1):
                    continue
                top_box = (r0, c0, r0 + a_len - 1, c0 + perp - 1)
                bottom_r0 = r0 + a_len + (a_len + b_len)
                bottom_box = (bottom_r0, c0, bottom_r0 + b_len - 1, c0 + perp - 1)
                boxes = (top_box, bottom_box)

            available = [color for color in range(10) if color != bg]
            color_a, color_b = random.sample(available, 2)
            for box, color in zip(boxes, (color_a, color_b)):
                rr0, cc0, rr1, cc1 = box
                grid[rr0 : rr1 + 1, cc0 : cc1 + 1] = color
            fr0, fc0, fr1, fc1 = footprint
            reserved[fr0 : fr1 + 1, fc0 : fc1 + 1] = True
            specs.append(
                {
                    "orientation": orientation,
                    "a_len": a_len,
                    "b_len": b_len,
                    "perp": perp,
                    "footprint": footprint,
                    "boxes": boxes,
                    "colors": (color_a, color_b),
                }
            )
            placed = True
            break
        if not placed:
            return None

    return grid, specs


def _validate_generated(grid: np.ndarray, specs: list[dict]) -> bool:
    """Check task semantics, uniqueness, necessary dependencies, and non-degeneracy."""
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if {spec["orientation"] for spec in specs} != {"H", "V"}:
        return False
    unequal = [spec for spec in specs if spec["a_len"] != spec["b_len"]]
    if not unequal:
        return False

    try:
        bg, pairs = _analyze(grid)
    except AssertionError:
        return False
    if len(pairs) != len(specs) or bg != _background(grid):
        return False

    expected = grid.copy()
    for spec in specs:
        a_len, b_len = spec["a_len"], spec["b_len"]
        if spec["orientation"] == "H":
            r0, c0, r1, c1 = spec["footprint"]
            left_box, right_box = spec["boxes"]
            lc1 = left_box[3]
            rc0 = right_box[1]
            expected[r0 : r1 + 1, lc1 + 1 : lc1 + 1 + b_len] = spec["colors"][0]
            expected[r0 : r1 + 1, rc0 - a_len : rc0] = spec["colors"][1]
            if lc1 + b_len + 1 != rc0 - a_len:
                return False
        else:
            r0, c0, r1, c1 = spec["footprint"]
            top_box, bottom_box = spec["boxes"]
            tr1 = top_box[2]
            br0 = bottom_box[0]
            expected[tr1 + 1 : tr1 + 1 + b_len, c0 : c1 + 1] = spec["colors"][0]
            expected[br0 - a_len : br0, c0 : c1 + 1] = spec["colors"][1]
            if tr1 + b_len + 1 != br0 - a_len:
                return False

    actual = _apply_pairs(grid, pairs)
    if not np.array_equal(actual, expected) or np.array_equal(actual, grid):
        return False

    # On every unequal pair, cross-reference differs from both tempting shortcuts.
    for spec in unequal:
        gap = spec["a_len"] + spec["b_len"]
        if spec["b_len"] == spec["a_len"] or 2 * spec["b_len"] == gap:
            return False
    return True


def generate_input() -> np.ndarray:
    """Generate varied, unambiguous multi-pair examples with both orientations."""
    for _ in range(400):
        candidate = _try_generate()
        if candidate is None:
            continue
        grid, specs = candidate
        if _validate_generated(grid, specs):
            return grid
    raise RuntimeError("failed to generate an unambiguous mutual-length input")
