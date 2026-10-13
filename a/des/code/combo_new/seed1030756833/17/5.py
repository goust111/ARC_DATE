import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
COLORS = tuple(c for c in range(10) if c not in (BLACK, GREY))
MAX_GENERATION_ATTEMPTS = 800


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        q = deque([(sr, sc)])
        pts: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            pts.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]) and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
        comps.append(pts)
    return comps


def _line_cells(
    orientation: str,
    r: int,
    c: int,
    length: int,
    inclusive_distance: bool = False,
) -> list[tuple[int, int]]:
    count = length + 1 if inclusive_distance else length
    if orientation == "H":
        return [(r, c + i) for i in range(count)]
    return [(r + i, c) for i in range(count)]


def _padded_cells(
    h: int,
    w: int,
    cells: list[tuple[int, int]],
    pad: int = 1,
) -> set[tuple[int, int]]:
    padded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    padded.add((rr, cc))
    return padded


def _segment_record(pts: list[tuple[int, int]]) -> dict[str, object]:
    rows = sorted({r for r, _ in pts})
    cols = sorted({c for _, c in pts})
    if len(rows) == 1:
        row = rows[0]
        assert cols == list(range(cols[0], cols[-1] + 1)), "grey segment must be contiguous"
        return {"orientation": "H", "length": len(cols), "cells": sorted(pts)}
    if len(cols) == 1:
        col = cols[0]
        assert rows == list(range(rows[0], rows[-1] + 1)), "grey segment must be contiguous"
        return {"orientation": "V", "length": len(rows), "cells": sorted(pts)}
    raise AssertionError("grey components must be straight horizontal or vertical segments")


def _grey_segments(g: np.ndarray) -> list[dict[str, object]]:
    segments = []
    for pts in _components(g == GREY):
        assert len(pts) >= 2, "grey segments must have length at least two"
        segments.append(_segment_record(pts))
    return segments


def _anchor_pairs(g: np.ndarray) -> list[dict[str, object]]:
    pairs = []
    for color in sorted(int(c) for c in np.unique(g) if int(c) not in (BLACK, GREY)):
        coords = [tuple(map(int, p)) for p in np.argwhere(g == color)]
        assert len(coords) == 2, "each anchor color must appear exactly twice"
        (r0, c0), (r1, c1) = sorted(coords)
        if r0 == r1:
            length = abs(c1 - c0)
            orientation = "H"
        elif c0 == c1:
            length = abs(r1 - r0)
            orientation = "V"
        else:
            raise AssertionError("same-color anchors must be row- or column-aligned")
        assert length >= 2, "anchor spacing must match a grey segment of length at least two"
        pairs.append(
            {
                "color": color,
                "orientation": orientation,
                "length": length,
                "a": (r0, c0),
                "b": (r1, c1),
            }
        )
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace grey stroke fragments with color-matched anchor-to-anchor connection lines.

    Input structure:
    - The background is black.
    - Each non-black, non-grey color appears as exactly two single-pixel anchors.
    - The two anchors of a color are aligned horizontally or vertically.
    - Separate grey components are straight horizontal or vertical line segments of length at least two.
    - For every anchor pair there is a grey segment with the same orientation and length equal to the anchor coordinate distance.

    Rule:
    - For each same-color anchor pair, compute its horizontal or vertical direction and the distance between the anchors.
    - Find the grey segment with that same direction and distance-length.
    - Erase the matched grey segment from its original noisy location.
    - Draw a solid line in the anchor color between the two anchors, including both anchor pixels.

    Key insight: The grey segment positions are distractors; only their direction and length encode which colored anchor pair must be connected.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    out = g.copy()

    segments = _grey_segments(g)
    used_segments: set[int] = set()

    for pair in _anchor_pairs(g):
        matches = [
            i
            for i, seg in enumerate(segments)
            if i not in used_segments
            and seg["orientation"] == pair["orientation"]
            and int(seg["length"]) == int(pair["length"])
        ]
        assert len(matches) == 1, "each anchor pair must match exactly one grey segment"
        seg = segments[matches[0]]
        used_segments.add(matches[0])

        for r, c in seg["cells"]:
            out[int(r), int(c)] = BLACK

        (r0, c0) = pair["a"]
        (r1, c1) = pair["b"]
        color = int(pair["color"])
        if pair["orientation"] == "H":
            c_start, c_stop = sorted((c0, c1))
            out[r0, c_start : c_stop + 1] = color
        else:
            r_start, r_stop = sorted((r0, r1))
            out[r_start : r_stop + 1, c0] = color

    assert len(used_segments) == len(segments), "all grey segments must be consumed"
    assert not bool(np.any(out == GREY)), "grey pixels must not remain in the output"
    return out


def _can_use_cells(
    cells: list[tuple[int, int]],
    h: int,
    w: int,
    forbidden: set[tuple[int, int]],
) -> bool:
    return all(0 <= r < h and 0 <= c < w and (r, c) not in forbidden for r, c in cells)


def _place_anchor_pair(
    grid: np.ndarray,
    orientation: str,
    length: int,
    color: int,
    forbidden_paths: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    h, w = grid.shape
    candidates: list[tuple[int, int]] = []
    if orientation == "H":
        for r in range(h):
            for c in range(w - length):
                candidates.append((r, c))
    else:
        for r in range(h - length):
            for c in range(w):
                candidates.append((r, c))
    random.shuffle(candidates)

    for r, c in candidates:
        path = _line_cells(orientation, r, c, length, inclusive_distance=True)
        if not _can_use_cells(path, h, w, forbidden_paths):
            continue
        anchor_a = path[0]
        anchor_b = path[-1]
        if grid[anchor_a] != BLACK or grid[anchor_b] != BLACK:
            continue
        grid[anchor_a] = color
        grid[anchor_b] = color
        return path
    return None


def _place_grey_segment(
    grid: np.ndarray,
    orientation: str,
    length: int,
    forbidden: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    h, w = grid.shape
    candidates: list[tuple[int, int]] = []
    if orientation == "H":
        for r in range(h):
            for c in range(w - length + 1):
                candidates.append((r, c))
    else:
        for r in range(h - length + 1):
            for c in range(w):
                candidates.append((r, c))
    random.shuffle(candidates)

    for r, c in candidates:
        cells = _line_cells(orientation, r, c, length)
        if not _can_use_cells(cells, h, w, forbidden):
            continue
        if any(grid[rr, cc] != BLACK for rr, cc in cells):
            continue
        for rr, cc in cells:
            grid[rr, cc] = GREY
        return cells
    return None


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(12, 30)
    w = random.randint(12, 30)
    horizontal_lengths = list(range(2, min(10, w - 2) + 1))
    vertical_lengths = list(range(2, min(10, h - 2) + 1))
    if len(horizontal_lengths) < 1 or len(vertical_lengths) < 1:
        return None

    max_pairs = min(len(COLORS), len(horizontal_lengths) + len(vertical_lengths), 6)
    if max_pairs < 2:
        return None
    pair_count = random.randint(2, max_pairs)
    min_h = max(1, pair_count - len(vertical_lengths))
    max_h = min(len(horizontal_lengths), pair_count - 1)
    if min_h > max_h:
        return None
    horizontal_count = random.randint(min_h, max_h)
    vertical_count = pair_count - horizontal_count

    random.shuffle(horizontal_lengths)
    random.shuffle(vertical_lengths)
    attrs = [("H", length) for length in horizontal_lengths[:horizontal_count]]
    attrs.extend(("V", length) for length in vertical_lengths[:vertical_count])
    random.shuffle(attrs)

    colors = random.sample(COLORS, pair_count)
    grid = np.full((h, w), BLACK, dtype=int)
    forbidden_paths: set[tuple[int, int]] = set()
    no_grey: set[tuple[int, int]] = set()

    for color, (orientation, length) in zip(colors, attrs, strict=True):
        path = _place_anchor_pair(grid, orientation, length, color, forbidden_paths)
        if path is None:
            return None
        forbidden_paths.update(_padded_cells(h, w, path, pad=1))
        no_grey.update(_padded_cells(h, w, path, pad=1))

    grey_forbidden = set(no_grey)
    for orientation, length in sorted(attrs, key=lambda item: random.random()):
        cells = _place_grey_segment(grid, orientation, length, grey_forbidden)
        if cells is None:
            return None
        grey_forbidden.update(_padded_cells(h, w, cells, pad=1))

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape or max(out.shape) > 30:
        return None
    if bool(np.any(out == GREY)):
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
