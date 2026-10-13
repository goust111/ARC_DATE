from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = (1, 2, 3, 4, 5, 6, 7, 8, 9)
MAX_GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 80
MAX_LEN_CHOICES = (5, 7, 9, 11, 13, 15, 17, 19, 21)


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _parse_segments(grid: np.ndarray) -> list[tuple[str, int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    objects = _components(g != bg)
    assert objects, "expected at least one segment"

    segments: list[tuple[str, int, int]] = []
    seen_lengths: dict[str, set[int]] = {"h": set(), "v": set()}
    point_count = 0

    for coords in objects:
        rows = coords[:, 0]
        cols = coords[:, 1]
        color = int(g[rows[0], cols[0]])
        assert np.all(g[rows, cols] == color), "each segment must be monochrome"

        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        area = int(coords.shape[0])
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        assert area == height * width, "segment must be contiguous without holes"

        if area == 1:
            point_count += 1
            assert point_count == 1, "length-1 center marker must be unique"
            segments.append(("p", 1, color))
            continue

        if height == 1:
            length = width
            assert length % 2 == 1, "segment lengths must be odd"
            assert length not in seen_lengths["h"], "horizontal lengths must be unique"
            seen_lengths["h"].add(length)
            segments.append(("h", length, color))
            continue

        if width == 1:
            length = height
            assert length % 2 == 1, "segment lengths must be odd"
            assert length not in seen_lengths["v"], "vertical lengths must be unique"
            seen_lengths["v"].add(length)
            segments.append(("v", length, color))
            continue

        raise AssertionError("each component must be a one-cell-thick straight segment")

    return segments


def _paint_output(size: int, segments: list[tuple[str, int, int]], bg: int = BG) -> np.ndarray:
    out = np.full((size, size), bg, dtype=int)
    center = size // 2

    for orient, length, color in sorted(segments, key=lambda item: (item[1], item[0])):
        if length == 1:
            out[center, center] = color
            continue

        d = length // 2
        r0 = center - d
        r1 = center + d
        c0 = center - d
        c1 = center + d
        if orient == "h":
            out[r0, c0 : c1 + 1] = color
            out[r1, c0 : c1 + 1] = color
        else:
            out[r0 + 1 : r1, c0] = color
            out[r0 + 1 : r1, c1] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast scattered line segments into concentric square rings whose side lengths match segment lengths.

    Input structure:
    - The canvas has a dominant black background and several isolated one-cell-thick monochrome segments.
    - Every foreground component is either a horizontal segment, a vertical segment, or a single cell.
    - Segment positions are distractors; odd lengths are the meaningful property, and lengths are unique within each direction.

    Rule:
    - Create a square output whose side length equals the longest segment length.
    - A horizontal segment of length L paints the top and bottom edges of the centered square ring of side length L, including the corners.
    - A vertical segment of length L paints the left and right edges of that ring, excluding the corners; a length-1 segment paints the center cell.

    Key insight:
    - Ignore where segments appear in the input: length chooses which concentric ring is activated, and orientation chooses whether it contributes horizontal edges or vertical edges.
    """
    segments = _parse_segments(grid)
    max_length = max(length for _orient, length, _color in segments)
    assert max_length % 2 == 1, "largest segment length must be odd"
    bg = _get_bg(np.asarray(grid, dtype=int))
    return _paint_output(max_length, segments, bg=bg)


def _choose_specs() -> tuple[int, list[tuple[str, int, int]]] | None:
    max_len = random.choice(MAX_LEN_CHOICES)
    lengths = list(range(1, max_len + 1, 2))

    modes: dict[int, str] = {}
    include_point = random.random() < 0.4
    if include_point:
        modes[1] = "p"

    for length in lengths[1:]:
        mode = random.choices(
            ("none", "h", "v", "hv"),
            weights=(4, 3, 3, 4),
            k=1,
        )[0]
        modes[length] = mode

    if modes.get(max_len, "none") == "none":
        modes[max_len] = random.choice(("h", "v", "hv"))

    if 1 not in modes and random.random() < 0.25:
        modes[1] = "p"

    if "hv" not in modes.values() and random.random() < 0.5:
        candidates = [length for length in lengths[1:] if modes.get(length, "none") != "none"]
        if candidates:
            modes[random.choice(candidates)] = "hv"

    specs: list[tuple[str, int, int]] = []
    for length in lengths:
        mode = modes.get(length, "none")
        if mode == "p":
            specs.append(("p", length, -1))
        elif mode == "h":
            specs.append(("h", length, -1))
        elif mode == "v":
            specs.append(("v", length, -1))
        elif mode == "hv":
            specs.append(("h", length, -1))
            specs.append(("v", length, -1))

    segment_count = len(specs)
    if not (3 <= segment_count <= len(PALETTE)):
        return None

    if not any(orient == "h" for orient, _length, _color in specs):
        return None
    if not any(orient == "v" for orient, _length, _color in specs):
        return None

    if random.random() < 0.35 and 1 not in modes:
        return None

    colors = random.sample(PALETTE, segment_count)
    colored = [(orient, length, color) for (orient, length, _), color in zip(specs, colors)]
    return max_len, colored


def _segment_box(orient: str, length: int, anchor: tuple[int, int]) -> tuple[int, int, int, int]:
    r, c = anchor
    if orient == "h":
        return r, c, r, c + length - 1
    if orient == "v":
        return r, c, r + length - 1, c
    return r, c, r, c


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + 1 < br0 or br1 + 1 < ar0 or ac1 + 1 < bc0 or bc1 + 1 < ac0)


def _candidates(h: int, w: int, orient: str, length: int) -> list[tuple[int, int]]:
    pts: list[tuple[int, int]] = []
    if orient == "h":
        for r in range(h):
            for c in range(w - length + 1):
                pts.append((r, c))
    elif orient == "v":
        for r in range(h - length + 1):
            for c in range(w):
                pts.append((r, c))
    else:
        for r in range(h):
            for c in range(w):
                pts.append((r, c))
    random.shuffle(pts)
    return pts


def _place_segments(h: int, w: int, specs: list[tuple[str, int, int]]) -> list[tuple[int, int]] | None:
    order = sorted(range(len(specs)), key=lambda i: (specs[i][1], specs[i][0] != "p"), reverse=True)
    all_candidates = {
        i: _candidates(h, w, specs[i][0], specs[i][1])
        for i in range(len(specs))
    }
    anchors: list[tuple[int, int] | None] = [None] * len(specs)
    boxes: list[tuple[int, int, int, int]] = []

    def search(k: int) -> bool:
        if k == len(order):
            return True
        idx = order[k]
        orient, length, _color = specs[idx]
        for anchor in all_candidates[idx]:
            box = _segment_box(orient, length, anchor)
            if any(_boxes_too_close(box, other) for other in boxes):
                continue
            anchors[idx] = anchor
            boxes.append(box)
            if search(k + 1):
                return True
            boxes.pop()
            anchors[idx] = None
        return False

    if search(0):
        return [anchor for anchor in anchors if anchor is not None]
    return None


def _render_input(h: int, w: int, specs: list[tuple[str, int, int]], anchors: list[tuple[int, int]]) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for (orient, length, color), (r, c) in zip(specs, anchors):
        if orient == "h":
            grid[r, c : c + length] = color
        elif orient == "v":
            grid[r : r + length, c] = color
        else:
            grid[r, c] = color
    return grid


def _gen_attempt() -> np.ndarray | None:
    choice = _choose_specs()
    if choice is None:
        return None
    max_len, specs = choice

    longest_h = max((length for orient, length, _color in specs if orient in {"h", "p"}), default=1)
    longest_v = max((length for orient, length, _color in specs if orient in {"v", "p"}), default=1)
    count = len(specs)

    h_min = max(longest_v + 2, count * 2 + 1, 9)
    w_min = max(longest_h + 2, count * 2 + 1, 9)
    if h_min > 30 or w_min > 30:
        return None

    for _ in range(PLACEMENT_ATTEMPTS):
        h = random.randint(h_min, 30)
        w = random.randint(w_min, 30)
        anchors = _place_segments(h, w, specs)
        if anchors is None:
            continue
        grid = _render_input(h, w, specs, anchors)
        if np.count_nonzero(grid == BG) <= np.count_nonzero(grid != BG):
            continue
        expected = _paint_output(max_len, specs, bg=BG)
        try:
            actual = main(grid)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
