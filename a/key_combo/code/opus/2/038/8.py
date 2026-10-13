from __future__ import annotations

import itertools
import random
from collections import deque

import numpy as np


BG = 0
LINE_COLORS = (1, 2, 3)
LABEL_COLORS = (4, 5, 7)
LINE_LENGTHS = (4, 5, 6, 7)
MIN_LINE_GAP = 6
GEN_ATTEMPTS = 400

# Normalized 4-connected label shapes keyed by visible pixel count.
LABEL_SHAPES: dict[int, list[list[tuple[int, int]]]] = {
    2: [
        [(0, 0), (0, 1)],
    ],
    3: [
        [(0, 0), (0, 1), (1, 0)],
        [(0, 0), (0, 1), (1, 1)],
        [(0, 0), (1, 0), (1, 1)],
        [(0, 1), (1, 0), (1, 1)],
    ],
    4: [
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        [(0, 0), (1, 0), (2, 0), (2, 1)],
        [(0, 1), (1, 1), (2, 0), (2, 1)],
        [(0, 0), (1, 0), (1, 1), (2, 1)],
    ],
}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_by_color(grid: np.ndarray, bg: int = BG) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    comps: list[dict] = []

    for color in sorted(set(map(int, np.unique(g))) - {int(bg)}):
        seen = np.zeros((h, w), dtype=bool)
        for r in range(h):
            for c in range(w):
                if seen[r, c] or int(g[r, c]) != color:
                    continue
                q = deque([(r, c)])
                seen[r, c] = True
                pts: list[tuple[int, int]] = []
                while q:
                    rr, cc = q.popleft()
                    pts.append((rr, cc))
                    for nr, nc in _neighbors4(rr, cc):
                        if not _in_bounds(g.shape, nr, nc):
                            continue
                        if seen[nr, nc] or int(g[nr, nc]) != color:
                            continue
                        seen[nr, nc] = True
                        q.append((nr, nc))
                coords = np.array(pts, dtype=int)
                comps.append(
                    {
                        "color": int(color),
                        "coords": coords,
                        "bbox": _bbox_coords(coords),
                        "area": int(coords.shape[0]),
                    }
                )
    return comps


def _is_vertical_line(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    return c0 == c1 and int(comp["area"]) == (r1 - r0 + 1) and 4 <= int(comp["area"]) <= 7


def _extract_specs(grid: np.ndarray) -> list[dict]:
    comps = _components_by_color(grid, bg=BG)
    lines = [comp for comp in comps if _is_vertical_line(comp)]
    labels = [comp for comp in comps if not _is_vertical_line(comp)]

    assert 2 <= len(lines) <= 3, "expected 2 or 3 vertical line segments"
    assert len(labels) == len(lines), "expected one label per line"

    used_labels: set[int] = set()
    specs: list[dict] = []
    for line in sorted(lines, key=lambda comp: comp["bbox"][1]):
        r0, c0, r1, _ = line["bbox"]
        target_col = c0 + 2
        candidates: list[tuple[int, dict]] = []
        for idx, label in enumerate(labels):
            if idx in used_labels:
                continue
            lr0, lc0, lr1, _ = label["bbox"]
            if lc0 != target_col:
                continue
            if lr1 < r0 - 1 or lr0 > r1 + 1:
                continue
            candidates.append((idx, label))

        assert len(candidates) == 1, "each line must have one label on its right"
        idx, label = candidates[0]
        used_labels.add(idx)
        specs.append(
            {
                "row0": int(r0),
                "col": int(c0),
                "length": int(line["area"]),
                "line_color": int(line["color"]),
                "label_color": int(label["color"]),
                "label_area": int(label["area"]),
            }
        )

    return specs


def _draw_pyramid(out: np.ndarray, spec: dict) -> None:
    row0 = int(spec["row0"])
    col = int(spec["col"])
    length = int(spec["length"])
    line_color = int(spec["line_color"])
    alt_color = int(spec["label_color"])
    upward = (int(spec["label_area"]) % 2 == 0)

    for depth in range(length):
        width = 2 * depth + 1
        half = width // 2
        if upward:
            r = row0 + depth
        else:
            r = row0 + length - 1 - depth
        if not (0 <= r < out.shape[0]):
            continue
        for dc in range(-half, half + 1):
            c = col + dc
            if 0 <= c < out.shape[1]:
                out[r, c] = line_color if abs(dc) % 2 == 0 else alt_color


def _render_output(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for spec in sorted(specs, key=lambda item: item["col"]):
        _draw_pyramid(out, spec)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn each vertical segment into an alternating pyramid whose direction and secondary color come from its right-side label.

    Input structure:
    - The background is black.
    - There are 2 or 3 monochrome vertical 4-connected line segments with distinct colors and distinct lengths.
    - Each line has one small 4-connected label object one column gap to its right; the label's left edge starts two columns right of the line.

    Rule:
    - Pair every vertical line with the unique nearby label component whose left edge is at line_col + 2.
    - Count the label pixels: even count makes an upright pyramid whose apex is the line's top pixel; odd count makes an inverted pyramid whose bottom tip is the line's bottom pixel.
    - The pyramid height equals the line length, each farther row widens by 2 around the line column, even horizontal offsets use the line color, and odd horizontal offsets use the label color.
    - Erase the labels and draw the pyramids from left to right, letting later pyramids overwrite earlier ones where they overlap.

    Key insight:
    The adjacent label encodes two independent parameters at once: its area parity chooses up versus down, while its visible color chooses the alternating secondary stripe color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    specs = _extract_specs(g)
    return _render_output(g.shape, specs)


def _random_partition(total: int, parts: int) -> list[int]:
    buckets = [0] * parts
    for _ in range(total):
        buckets[random.randrange(parts)] += 1
    return buckets


def _sample_label_areas(count: int) -> list[int]:
    if count == 2:
        areas = [3, random.choice((2, 4))]
    else:
        if random.random() < 0.5:
            areas = [3, 3, random.choice((2, 4))]
        else:
            areas = [3, 2, 4]
    random.shuffle(areas)
    return areas


def _sample_columns(width: int, count: int) -> list[int]:
    slack = width - 4 - (2 + MIN_LINE_GAP * (count - 1))
    assert slack >= 0
    extra = _random_partition(slack, count + 1)

    cols = [2 + extra[0]]
    for i in range(1, count):
        cols.append(cols[-1] + MIN_LINE_GAP + extra[i])
    return cols


def _clipping_score(width: int, columns: list[int], lengths: tuple[int, ...]) -> int:
    score = 0
    for col, length in zip(columns, lengths):
        half = length - 1
        score += max(0, half - col)
        score += max(0, col + half - (width - 1))
    return score


def _assign_lengths(width: int, columns: list[int], lengths: list[int]) -> list[int]:
    perms = list(itertools.permutations(lengths))
    best_score = min(_clipping_score(width, columns, perm) for perm in perms)
    best = [perm for perm in perms if _clipping_score(width, columns, perm) == best_score]
    return list(random.choice(best))


def _assign_row_starts(height: int, lengths: list[int]) -> list[int]:
    count = len(lengths)
    minimum = sum(lengths) + max(0, count - 1) + 2
    assert minimum <= height
    slack = height - minimum
    extra = _random_partition(slack, count + 1)
    order = list(range(count))
    random.shuffle(order)

    starts = [0] * count
    cursor = 1 + extra[0]
    for pos, idx in enumerate(order):
        starts[idx] = cursor
        cursor += lengths[idx]
        if pos < count - 1:
            cursor += 1 + extra[pos + 1]
    return starts


def _sample_shape(area: int) -> list[tuple[int, int]]:
    return list(random.choice(LABEL_SHAPES[area]))


def _build_grid(specs: list[dict], shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    grid = np.full((h, w), BG, dtype=int)

    for spec in specs:
        row0 = int(spec["row0"])
        col = int(spec["col"])
        length = int(spec["length"])
        line_color = int(spec["line_color"])
        label_color = int(spec["label_color"])
        label_row0 = int(spec["label_row0"])
        label_shape = spec["label_shape"]

        assert np.all(grid[row0 : row0 + length, col] == BG), "line placement overlap"
        grid[row0 : row0 + length, col] = line_color

        for dr, dc in label_shape:
            rr = label_row0 + dr
            cc = col + 2 + dc
            assert 0 <= rr < h and 0 <= cc < w, "label out of bounds"
            assert int(grid[rr, cc]) == BG, "label placement overlap"
            grid[rr, cc] = label_color

    return grid


def _label_row(row0: int, length: int, shape_h: int, height: int) -> int:
    center = row0 + (length - 1) / 2
    anchor = int(round(center - (shape_h - 1) / 2))
    low = max(0, row0 - 1)
    high = min(height - shape_h, row0 + length - shape_h + 1)
    offset = random.choice((-1, 0, 1))
    return min(max(anchor + offset, low), high)


def _gen_attempt() -> np.ndarray | None:
    count = random.choice((2, 3))
    raw_lengths = random.sample(LINE_LENGTHS, count)

    width_min = max(15 if count == 2 else 18, 2 * max(raw_lengths) + 5 * (count - 1))
    width = random.randint(width_min, min(29, width_min + 4))
    columns = _sample_columns(width, count)
    lengths = _assign_lengths(width, columns, raw_lengths)

    height_min = max(14, sum(lengths) + max(0, count - 1) + 2)
    height = random.randint(height_min, min(29, height_min + 4))
    row_starts = _assign_row_starts(height, lengths)

    line_colors = random.sample(LINE_COLORS, count)
    label_colors = random.sample(LABEL_COLORS, count)
    label_areas = _sample_label_areas(count)

    specs: list[dict] = []
    for i in range(count):
        shape = _sample_shape(label_areas[i])
        shape_h = 1 + max(r for r, _ in shape)
        row0 = row_starts[i]
        specs.append(
            {
                "row0": int(row0),
                "col": int(columns[i]),
                "length": int(lengths[i]),
                "line_color": int(line_colors[i]),
                "label_color": int(label_colors[i]),
                "label_area": int(label_areas[i]),
                "label_shape": shape,
                "label_row0": int(_label_row(row0, lengths[i], shape_h, height)),
            }
        )

    grid = _build_grid(specs, (height, width))
    expected = _render_output((height, width), specs)

    extracted = _extract_specs(grid)
    if sorted((s["col"], s["row0"], s["length"], s["line_color"], s["label_color"], s["label_area"]) for s in extracted) != sorted(
        (s["col"], s["row0"], s["length"], s["line_color"], s["label_color"], s["label_area"]) for s in specs
    ):
        return None

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
