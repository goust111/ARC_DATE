from __future__ import annotations

from collections import Counter
import random

import numpy as np


BACKGROUND = 0
BASE_COLOR = 1
MARKER_COLORS = (2, 3, 4, 6)
REFERENCE_COLOR = 7
GEN_RETRY_LIMIT = 400


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _majority_color(patch: np.ndarray) -> int:
    counts = Counter(map(int, patch.ravel()))
    color, freq = counts.most_common(1)[0]
    assert list(counts.values()).count(freq) == 1
    return int(color)


def _extract_objects(grid: np.ndarray) -> tuple[list[np.ndarray], int, tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = int(Counter(map(int, g.ravel())).most_common(1)[0][0])
    comps = _components(g != bg)
    assert len(comps) >= 4

    rectangles: list[tuple[int, np.ndarray]] = []
    reference: tuple[int, int] | None = None
    for coords in comps:
        if len(coords) == 1:
            assert reference is None
            reference = tuple(map(int, coords[0]))
            continue

        r0, c0, r1, c1 = _bbox(coords)
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(patch != bg)
        assert patch.size == len(coords), "foreground objects must be filled rectangles"
        rectangles.append((c0, patch.copy()))

    assert reference is not None
    assert 3 <= len(rectangles) <= 4

    rectangles.sort(key=lambda item: item[0])
    patches = [patch for _c0, patch in rectangles]
    shapes = {patch.shape for patch in patches}
    assert len(shapes) == 1

    bases = {_majority_color(patch) for patch in patches}
    assert len(bases) == 1
    base = next(iter(bases))
    return patches, int(base), reference


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge same-sized marker rectangles into one rectangle using parity-dependent conflict arbitration.

    Input structure:
    - A black background contains 3-4 separated filled rectangles of the same size, ordered left-to-right.
    - Every rectangle shares one base color and may contain several non-base marker pixels at relative positions.
    - A single isolated reference pixel sits to the lower-right of the rectangles and is not part of any rectangle.

    Rule:
    - Sort the rectangles by their left column and align them by their common bounding-box size.
    - Build one output rectangle of that size, filled with the shared base color.
    - At each relative position, copy the lone non-base marker if exactly one rectangle has it; if several rectangles agree on the same non-base color, keep that color.
    - If several rectangles provide different non-base colors at that position, compare `(dr % 2, dc % 2)` with the reference pixel's absolute parity: matching parity takes the leftmost rectangle's marker, otherwise take the rightmost rectangle's marker.

    Key insight:
    - The rectangles are overlay layers of one template, but conflict resolution is spatially local: parity chooses whether each contested cell prefers the left edge of the row of layers or the right edge.
    """
    patches, base, reference = _extract_objects(grid)
    h, w = patches[0].shape
    ref_r, ref_c = reference

    out = np.full((h, w), base, dtype=int)
    for dr in range(h):
        for dc in range(w):
            markers = [
                (idx, int(patch[dr, dc]))
                for idx, patch in enumerate(patches)
                if int(patch[dr, dc]) != base
            ]
            if not markers:
                continue

            colors = {color for _idx, color in markers}
            if len(markers) == 1 or len(colors) == 1:
                out[dr, dc] = markers[0][1]
                continue

            same_parity = (dr % 2 == ref_r % 2) and (dc % 2 == ref_c % 2)
            out[dr, dc] = markers[0][1] if same_parity else markers[-1][1]
    return out


def _choose_layout() -> tuple[int, int, int, int, int, list[int], int, int, int]:
    for _ in range(200):
        count = random.choice((3, 4))
        rect_h = random.randint(4, 6)
        rect_w = random.randint(4, 6 if count == 3 else 5)
        left_margin = random.randint(1, 2)
        gap = random.randint(1, 3 if count == 3 else 2)
        ref_dx = random.randint(1, 3)
        right_margin = random.randint(1, 2)
        top = random.randint(1, 5)
        ref_dy = random.randint(1, 4)
        bottom_margin = random.randint(1, 3)

        lefts = [left_margin + i * (rect_w + gap) for i in range(count)]
        ref_r = top + rect_h - 1 + ref_dy
        ref_c = lefts[-1] + rect_w - 1 + ref_dx
        height = ref_r + 1 + bottom_margin
        width = ref_c + 1 + right_margin
        if height <= 30 and width <= 30:
            return rect_h, rect_w, top, height, width, lefts, ref_r, ref_c, count
    raise RuntimeError("failed to choose a valid layout")


def _pick_conflict_positions(
    rect_h: int, rect_w: int, ref_r: int, ref_c: int
) -> list[tuple[int, int]]:
    all_cells = [(r, c) for r in range(rect_h) for c in range(rect_w)]
    match = [(r, c) for r, c in all_cells if r % 2 == ref_r % 2 and c % 2 == ref_c % 2]
    other = [(r, c) for r, c in all_cells if (r, c) not in set(match)]
    assert match and other

    positions = [random.choice(match), random.choice(other)]
    remaining = [cell for cell in all_cells if cell not in positions]
    if remaining and random.random() < 0.6:
        positions.append(random.choice(remaining))
    random.shuffle(positions)
    return positions


def _assign_conflicts(
    layers: list[dict[tuple[int, int], int]],
    conflict_positions: list[tuple[int, int]],
) -> None:
    for pos in conflict_positions:
        while True:
            colors = [random.choice(MARKER_COLORS) for _ in layers]
            if len(set(colors)) >= 2 and colors[0] != colors[-1]:
                break
        for layer, color in zip(layers, colors):
            layer[pos] = color


def _fill_unique_markers(
    layers: list[dict[tuple[int, int], int]],
    rect_h: int,
    rect_w: int,
    reserved: set[tuple[int, int]],
) -> None:
    global_free = [(r, c) for r in range(rect_h) for c in range(rect_w) if (r, c) not in reserved]
    random.shuffle(global_free)
    used = set(reserved)
    for layer in layers:
        target = random.randint(max(3, len(layer)), 5)
        while len(layer) < target:
            assert global_free
            pos = global_free.pop()
            assert pos not in used
            layer[pos] = random.choice(MARKER_COLORS)
            used.add(pos)


def _build_grid(
    rect_h: int,
    rect_w: int,
    top: int,
    height: int,
    width: int,
    lefts: list[int],
    ref_r: int,
    ref_c: int,
    layers: list[dict[tuple[int, int], int]],
) -> np.ndarray:
    grid = np.full((height, width), BACKGROUND, dtype=int)
    for left, markers in zip(lefts, layers):
        grid[top : top + rect_h, left : left + rect_w] = BASE_COLOR
        for (dr, dc), color in markers.items():
            grid[top + dr, left + dc] = color
    grid[ref_r, ref_c] = REFERENCE_COLOR
    return grid


def _validate_generated(grid: np.ndarray, conflict_positions: list[tuple[int, int]], reference: tuple[int, int]) -> None:
    out = main(grid)
    ref_r, ref_c = reference
    assert out.ndim == 2
    assert len(conflict_positions) >= 2

    parity_hits = {"left": False, "right": False}
    patches, base, _reference = _extract_objects(grid)
    assert base == BASE_COLOR
    assert _reference == reference
    for dr, dc in conflict_positions:
        markers = [int(patch[dr, dc]) for patch in patches if int(patch[dr, dc]) != base]
        assert len(set(markers)) >= 2
        if (dr % 2 == ref_r % 2) and (dc % 2 == ref_c % 2):
            assert int(out[dr, dc]) == int(patches[0][dr, dc])
            parity_hits["left"] = True
        else:
            assert int(out[dr, dc]) == int(patches[-1][dr, dc])
            parity_hits["right"] = True
    assert all(parity_hits.values())


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        rect_h, rect_w, top, height, width, lefts, ref_r, ref_c, count = _choose_layout()
        layers = [dict() for _ in range(count)]
        conflict_positions = _pick_conflict_positions(rect_h, rect_w, ref_r, ref_c)
        _assign_conflicts(layers, conflict_positions)
        _fill_unique_markers(layers, rect_h, rect_w, set(conflict_positions))
        grid = _build_grid(rect_h, rect_w, top, height, width, lefts, ref_r, ref_c, layers)

        try:
            _validate_generated(grid, conflict_positions, (ref_r, ref_c))
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid parity-overlay input")
