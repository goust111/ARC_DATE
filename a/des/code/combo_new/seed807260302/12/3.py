from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
FRAME = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
CORNERS = ("tl", "tr", "bl", "br")
ORIENTATIONS = ("h", "v")
GEN_RETRIES = 400
PAIR_PLACE_RETRIES = 80


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _template_bbox(
    shape: tuple[int, int], corner: str, orientation: str, seq_len: int
) -> tuple[int, int, int, int]:
    h, w = shape
    if orientation == "h":
        box_h, box_w = 3, seq_len + 2
    else:
        box_h, box_w = seq_len + 2, 3

    if corner == "tl":
        return 0, 0, box_h - 1, box_w - 1
    if corner == "tr":
        return 0, w - box_w, box_h - 1, w - 1
    if corner == "bl":
        return h - box_h, 0, h - 1, box_w - 1
    assert corner == "br"
    return h - box_h, w - box_w, h - 1, w - 1


def _template_cells(
    shape: tuple[int, int], corner: str, orientation: str, seq_len: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _template_bbox(shape, corner, orientation, seq_len)
    border = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if r in (r0, r1) or c in (c0, c1):
                border.append((r, c))

    if orientation == "h":
        inner = [(r0 + 1, c) for c in range(c0 + 1, c1)]
    else:
        inner = [(r, c0 + 1) for r in range(r0 + 1, r1)]
    return border, inner, (r0, c0, r1, c1)


def _find_template(
    grid: np.ndarray, bg: int
) -> tuple[tuple[int, int, int, int], list[int], int, str]:
    h, w = grid.shape
    candidates: list[tuple[tuple[int, int, int, int], list[int], int, str]] = []
    for corner in CORNERS:
        for orientation in ORIENTATIONS:
            max_len = (w - 2) if orientation == "h" else (h - 2)
            for seq_len in range(1, max_len + 1):
                if orientation == "h" and h < 3:
                    continue
                if orientation == "v" and w < 3:
                    continue
                r0, c0, r1, c1 = _template_bbox((h, w), corner, orientation, seq_len)
                if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
                    continue
                sub = grid[r0 : r1 + 1, c0 : c1 + 1]
                border_mask = np.ones_like(sub, dtype=bool)
                border_mask[1:-1, 1:-1] = False
                border_vals = np.unique(sub[border_mask])
                if border_vals.size != 1:
                    continue
                frame_color = int(border_vals[0])
                if frame_color == bg:
                    continue

                inner = sub[1:-1, 1:-1]
                if orientation == "h":
                    if inner.shape[0] != 1:
                        continue
                    seq = [int(v) for v in inner[0]]
                else:
                    if inner.shape[1] != 1:
                        continue
                    seq = [int(v) for v in inner[:, 0]]
                if not seq:
                    continue
                if any(color in (bg, frame_color) for color in seq):
                    continue
                candidates.append(((r0, c0, r1, c1), seq, frame_color, orientation))

    assert len(candidates) == 1, "expected exactly one corner template"
    return candidates[0]


def _template_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = bbox
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _ray_and_segment_cells(
    shape: tuple[int, int],
    a: tuple[int, int],
    b: tuple[int, int],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = shape
    (r0, c0), (r1, c1) = a, b
    if r0 == r1:
        if c0 > c1:
            c0, c1 = c1, c0
        gap = c1 - c0 - 1
        between = [(r0, c) for c in range(c0 + 1, c1)]
        rays = []
        rays.extend((r0, c0 - step) for step in range(1, gap + 1) if c0 - step >= 0)
        rays.extend((r0, c1 + step) for step in range(1, gap + 1) if c1 + step < w)
        endpoints = [(r0, c0), (r0, c1)]
        return endpoints, between, rays

    assert c0 == c1, "pair must be aligned"
    if r0 > r1:
        r0, r1 = r1, r0
    gap = r1 - r0 - 1
    between = [(r, c0) for r in range(r0 + 1, r1)]
    rays = []
    rays.extend((r0 - step, c0) for step in range(1, gap + 1) if r0 - step >= 0)
    rays.extend((r1 + step, c0) for step in range(1, gap + 1) if r1 + step < h)
    endpoints = [(r0, c0), (r1, c0)]
    return endpoints, between, rays


def _extract_pairs(
    grid: np.ndarray, template_box: tuple[int, int, int, int], bg: int
) -> list[dict]:
    h, w = grid.shape
    mask = _template_mask((h, w), template_box)
    outside_colors = sorted(int(v) for v in np.unique(grid[~mask]) if int(v) != bg)
    pairs = []
    for color in outside_colors:
        coords = np.argwhere((grid == color) & (~mask))
        assert coords.shape[0] == 2, "each antenna color must appear exactly twice"
        a = tuple(map(int, coords[0]))
        b = tuple(map(int, coords[1]))
        assert a[0] == b[0] or a[1] == b[1], "antenna endpoints must share a row or column"
        endpoints, between, rays = _ray_and_segment_cells((h, w), a, b)
        (er0, ec0), (er1, ec1) = endpoints
        gap = abs(ec1 - ec0) - 1 if er0 == er1 else abs(er1 - er0) - 1
        assert gap >= 1, "antenna endpoints must have at least one background cell between them"
        assert all(int(grid[r, c]) == bg for r, c in between), "space between endpoints must be background"

        endpoint_set = set(endpoints)
        for r, c in endpoints:
            for nr in range(max(0, r - 1), min(h, r + 2)):
                for nc in range(max(0, c - 1), min(w, c + 2)):
                    if (nr, nc) in endpoint_set and (nr, nc) != (r, c):
                        continue
                    if (nr, nc) != (r, c):
                        assert int(grid[nr, nc]) == bg, "antenna endpoints must be isolated pixels"

        pairs.append(
            {
                "color": color,
                "gap": gap,
                "endpoints": endpoints,
                "between": between,
                "rays": rays,
            }
        )

    gaps = [pair["gap"] for pair in pairs]
    assert len(gaps) == len(set(gaps)), "antenna gaps must be unique for an unambiguous ordering"
    return sorted(pairs, key=lambda pair: pair["gap"])


def _draw_rays(
    grid: np.ndarray, pairs: list[dict], sequence: list[int]
) -> np.ndarray:
    out = np.array(grid, copy=True)
    assert len(sequence) >= len(pairs), "template sequence must cover all antenna pairs"
    for index, pair in enumerate(pairs):
        ray_color = int(sequence[index])
        for r, c in pair["rays"]:
            out[r, c] = ray_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend aligned same-color pixel pairs into outward rays using the corner template's color order.

    Input structure:
    - The background is black, with a single small template framed in gray at one grid corner.
    - The template interior is a 1-pixel-wide horizontal or vertical strip whose visible color order is the control sequence.
    - Outside the template, every non-background color appears as exactly one isolated aligned pixel pair in the same row or column.

    Rule:
    - Read the template strip in row order for a horizontal strip or column order for a vertical strip.
    - Ignore the template, collect the aligned same-color pairs, and sort them by the number of background cells between the two endpoints.
    - Assign the first template color to the smallest-gap pair, the second template color to the next pair, and so on.
    - For each pair, draw rays from both endpoints away from the partner along the same line, using the assigned template color and a ray length equal to that gap; clip rays at the grid edge.

    Key insight:
    - The hidden linkage is that a pair's spacing plays two roles at once: it determines both the ranking that selects a template color and the geometric ray length to draw.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _most_common_color(g)
    template_box, sequence, _frame_color, _orientation = _find_template(g, bg)
    pairs = _extract_pairs(g, template_box, bg)
    return _draw_rays(g, pairs, sequence)


def _render_template(
    grid: np.ndarray, corner: str, orientation: str, sequence: list[int]
) -> tuple[int, int, int, int]:
    border_cells, inner_cells, bbox = _template_cells(grid.shape, corner, orientation, len(sequence))
    for r, c in border_cells:
        grid[r, c] = FRAME
    for (r, c), color in zip(inner_cells, sequence):
        grid[r, c] = int(color)
    return bbox


def _mark_cells(mask: np.ndarray, cells: list[tuple[int, int]]) -> None:
    for r, c in cells:
        mask[r, c] = True


def _touches_nonbg(nonbg_mask: np.ndarray, cell: tuple[int, int]) -> bool:
    h, w = nonbg_mask.shape
    r, c = cell
    for nr in range(max(0, r - 1), min(h, r + 2)):
        for nc in range(max(0, c - 1), min(w, c + 2)):
            if nonbg_mask[nr, nc]:
                return True
    return False


def _pair_endpoints_from_start(
    orientation: str, start_r: int, start_c: int, gap: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    if orientation == "h":
        return (start_r, start_c), (start_r, start_c + gap + 1)
    return (start_r, start_c), (start_r + gap + 1, start_c)


def _pair_is_truncated(
    shape: tuple[int, int],
    endpoints: tuple[tuple[int, int], tuple[int, int]],
) -> bool:
    h, w = shape
    (a, b) = endpoints
    endpoint_list, _between, rays = _ray_and_segment_cells(shape, a, b)
    if endpoint_list[0][0] == endpoint_list[1][0]:
        gap = endpoint_list[1][1] - endpoint_list[0][1] - 1
        return len(rays) < 2 * gap
    gap = endpoint_list[1][0] - endpoint_list[0][0] - 1
    return len(rays) < 2 * gap


def _place_pair(
    grid: np.ndarray,
    nonbg_mask: np.ndarray,
    line_reserved: np.ndarray,
    color: int,
    gap: int,
    orientation: str,
    want_truncated: bool,
) -> dict | None:
    h, w = grid.shape
    if orientation == "h":
        candidates = [
            (r, c)
            for r in range(h)
            for c in range(w - gap - 1)
        ]
    else:
        candidates = [
            (r, c)
            for r in range(h - gap - 1)
            for c in range(w)
        ]
    random.shuffle(candidates)

    for start_r, start_c in candidates:
        a, b = _pair_endpoints_from_start(orientation, start_r, start_c, gap)
        endpoints, between, rays = _ray_and_segment_cells((h, w), a, b)
        truncated = _pair_is_truncated((h, w), (a, b))
        if truncated != want_truncated:
            continue
        occupied_line = endpoints + between + rays
        if any(line_reserved[r, c] for r, c in occupied_line):
            continue
        if any(_touches_nonbg(nonbg_mask, cell) for cell in endpoints):
            continue

        for r, c in endpoints:
            grid[r, c] = color
        _mark_cells(nonbg_mask, endpoints)
        _mark_cells(line_reserved, occupied_line)
        return {
            "color": color,
            "gap": gap,
            "endpoints": endpoints,
            "between": between,
            "rays": rays,
            "orientation": orientation,
        }
    return None


def _expected_output(
    grid: np.ndarray,
    pairs: list[dict],
    sequence: list[int],
) -> np.ndarray:
    return _draw_rays(grid, sorted(pairs, key=lambda pair: pair["gap"]), sequence)


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(12, 28)
    w = random.randint(12, 28)
    pair_count = random.randint(2, 4)
    sequence_colors = random.sample(COLOR_POOL, pair_count)
    remaining = [color for color in COLOR_POOL if color not in sequence_colors]
    pair_colors = random.sample(remaining, pair_count)

    corner = random.choice(CORNERS)
    orientation = random.choice(ORIENTATIONS)
    if orientation == "h" and w < pair_count + 2:
        return None
    if orientation == "v" and h < pair_count + 2:
        return None

    grid = np.full((h, w), BG, dtype=int)
    template_box = _render_template(grid, corner, orientation, sequence_colors)

    nonbg_mask = grid != BG
    line_reserved = _template_mask((h, w), template_box)

    max_gap = min(7, max(1, min(h, w) - 3))
    if max_gap < pair_count:
        return None
    gaps = sorted(random.sample(range(1, max_gap + 1), pair_count))

    pair_specs = []
    place_order = list(range(pair_count))
    place_order.sort(key=lambda idx: gaps[idx], reverse=True)

    want_truncated_flags = [bool(random.getrandbits(1)) for _ in range(pair_count)]
    if pair_count >= 2:
        want_truncated_flags[0] = False
        want_truncated_flags[1] = True

    for idx in place_order:
        gap = gaps[idx]
        color = pair_colors[idx]
        orientation_choices = list(ORIENTATIONS)
        random.shuffle(orientation_choices)
        placed = None
        for preferred in orientation_choices:
            for _ in range(PAIR_PLACE_RETRIES):
                placed = _place_pair(
                    grid,
                    nonbg_mask,
                    line_reserved,
                    color=color,
                    gap=gap,
                    orientation=preferred,
                    want_truncated=want_truncated_flags[idx],
                )
                if placed is not None:
                    break
            if placed is not None:
                break
        if placed is None:
            return None
        pair_specs.append(placed)

    expected = _expected_output(grid, pair_specs, sequence_colors)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    parsed_box, parsed_sequence, frame_color, _ = _find_template(grid, BG)
    if parsed_box != template_box or frame_color != FRAME or parsed_sequence != sequence_colors:
        return None
    gaps_after_parse = [pair["gap"] for pair in _extract_pairs(grid, template_box, BG)]
    if gaps_after_parse != gaps:
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
