import random
from collections import Counter, deque

import numpy as np


BG = 0
MARKER = 5
DRAW_COLORS = (1, 2, 3, 4)

MAX_GEN_ATTEMPTS = 500


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _collect_passable_regions(grid: np.ndarray) -> list[dict]:
    """Collect 4-connected regions of black/grey interior cells."""
    g = np.asarray(grid, dtype=int)
    passable = (g == BG) | (g == MARKER)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    regions: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not passable[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            boundary = set()
            touches_border = False
            marker_count = 0
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                if rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1:
                    touches_border = True
                if g[rr, cc] == MARKER:
                    marker_count += 1
                for nr, nc in neighbors4(rr, cc):
                    if not _in_bounds(g.shape, nr, nc):
                        continue
                    if passable[nr, nc]:
                        if not seen[nr, nc]:
                            seen[nr, nc] = True
                            q.append((nr, nc))
                    elif g[nr, nc] != BG:
                        boundary.add((nr, nc))
            regions.append(
                {
                    "coords": coords,
                    "boundary": boundary,
                    "touches_border": touches_border,
                    "marker_count": marker_count,
                }
            )
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey-marked enclosed cavity with the least frequent boundary color.

    Input structure:
    - Black background contains a 1-pixel-wide multicolor line drawing made from blue/red/green/yellow.
    - The drawing forms several enclosed cavities whose interiors are black, except that some cavities contain exactly one grey marker pixel.
    - Boundary pixels stay outside the cavities; the grey marker only indicates which enclosed cavity should be processed.

    Rule:
    - Find every 4-connected enclosed interior region made of black and possibly grey cells, ignoring the outside background that reaches the grid border.
    - For each enclosed region containing a grey marker, collect all side-adjacent non-black boundary pixels and count how many times each non-grey color appears on that boundary.
    - Choose the unique least frequent boundary color and flood the whole marked cavity with that color, covering both its black cells and its grey marker.
    - Leave unmarked cavities and all original boundary lines unchanged.

    Key insight:
    - The grey pixel selects which cavity to fill, but the fill color is determined independently by a frequency count over that cavity's surrounding boundary colors.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for region in _collect_passable_regions(g):
        if region["touches_border"] or region["marker_count"] == 0:
            continue
        color_counts = Counter(int(g[r, c]) for r, c in region["boundary"] if int(g[r, c]) != MARKER)
        assert len(color_counts) >= 2, "marked cavity boundary must contain at least two colors"
        min_count = min(color_counts.values())
        winners = [color for color, count in color_counts.items() if count == min_count]
        assert len(winners) == 1, "marked cavity boundary must have a unique minority color"
        fill_color = winners[0]
        for r, c in region["coords"]:
            out[r, c] = fill_color
    return out


def _perimeter_coords(top: int, left: int, height: int, width: int) -> list[tuple[int, int]]:
    coords = []
    for c in range(left, left + width):
        coords.append((top, c))
    for r in range(top + 1, top + height - 1):
        coords.append((r, left + width - 1))
    if height > 1:
        for c in range(left + width - 1, left - 1, -1):
            coords.append((top + height - 1, c))
    if width > 1:
        for r in range(top + height - 2, top, -1):
            coords.append((r, left))
    return coords


def _sample_distinct_counts(perimeter_len: int, color_count: int, rng: random.Random) -> list[int] | None:
    for _ in range(200):
        if color_count == 2:
            a = rng.randint(4, perimeter_len - 4)
            b = perimeter_len - a
            if a != b and b >= 4:
                return sorted([a, b])
        else:
            a = rng.randint(3, perimeter_len - 6)
            b = rng.randint(3, perimeter_len - a - 3)
            c = perimeter_len - a - b
            vals = sorted([a, b, c])
            if min(vals) >= 3 and len(set(vals)) == 3:
                return vals
    return None


def _expand_segments(segments: list[tuple[int, int]]) -> list[int]:
    seq: list[int] = []
    for color, length in segments:
        assert length > 0
        seq.extend([color] * length)
    return seq


def _cyclic_segments(seq: list[int]) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    current = seq[0]
    length = 1
    for color in seq[1:]:
        if color == current:
            length += 1
        else:
            segments.append((current, length))
            current = color
            length = 1
    segments.append((current, length))
    if len(segments) > 1 and segments[0][0] == segments[-1][0]:
        segments[0] = (segments[0][0], segments[0][1] + segments[-1][1])
        segments.pop()
    return segments


def _rotate_to_break_wrap(seq: list[int], rng: random.Random) -> list[int]:
    n = len(seq)
    starts = [k for k in range(n) if seq[k] != seq[k - 1]]
    rng.shuffle(starts)
    for k in starts:
        rotated = seq[k:] + seq[:k]
        if rotated[0] != rotated[-1]:
            return rotated
    return seq


def _make_boundary_sequence(
    perimeter_len: int,
    palette: list[int],
    rng: random.Random,
    *,
    force_fill_color: int | None = None,
    anti_shortcut: bool = False,
) -> tuple[list[int], Counter]:
    for _ in range(200):
        local_count = 2 if len(palette) == 2 else rng.choice([2, 3])
        local_colors = rng.sample(palette, local_count)
        if force_fill_color is not None and force_fill_color not in local_colors:
            local_colors[0] = force_fill_color
            rng.shuffle(local_colors)

        counts = _sample_distinct_counts(perimeter_len, local_count, rng)
        if counts is None:
            continue

        if force_fill_color is None:
            ordered_colors = rng.sample(local_colors, len(local_colors))
            counts_by_color = {ordered_colors[i]: counts[i] for i in range(local_count)}
        else:
            others = [color for color in local_colors if color != force_fill_color]
            rng.shuffle(others)
            ordered_colors = [force_fill_color] + others
            counts_by_color = {ordered_colors[i]: counts[i] for i in range(local_count)}

        sorted_items = sorted(counts_by_color.items(), key=lambda kv: kv[1])
        fill_color, fill_count = sorted_items[0]
        if force_fill_color is not None and fill_color != force_fill_color:
            continue

        if local_count == 2:
            other_color, other_count = sorted_items[1]
            if anti_shortcut and fill_count >= 6 and other_count >= 5:
                a1 = fill_count // 2
                a2 = fill_count - a1
                small = 1
                b1 = max(2, (other_count - small) // 2)
                b2 = other_count - small - b1
                if min(a1, a2, b1, b2) < 2:
                    continue
                segments = [
                    (other_color, small),
                    (fill_color, a1),
                    (other_color, b1),
                    (fill_color, a2),
                    (other_color, b2),
                ]
            else:
                if fill_count >= 4 and other_count >= 4 and rng.random() < 0.8:
                    a1 = rng.randint(2, fill_count - 2)
                    a2 = fill_count - a1
                    b1 = rng.randint(2, other_count - 2)
                    b2 = other_count - b1
                    segments = [
                        (fill_color, a1),
                        (other_color, b1),
                        (fill_color, a2),
                        (other_color, b2),
                    ]
                else:
                    segments = [(fill_color, fill_count), (other_color, other_count)]
        else:
            mid_color, mid_count = sorted_items[1]
            max_color, max_count = sorted_items[2]
            if anti_shortcut and fill_count >= 6 and mid_count >= 5 and max_count >= 4:
                a1 = fill_count // 2
                a2 = fill_count - a1
                small = 1
                c1 = max(2, max_count // 2)
                c2 = max_count - c1
                if min(a1, a2, mid_count - small, c1, c2) < 2:
                    continue
                segments = [
                    (mid_color, small),
                    (fill_color, a1),
                    (max_color, c1),
                    (mid_color, mid_count - small),
                    (fill_color, a2),
                    (max_color, c2),
                ]
            else:
                b1 = max(2, mid_count // 2) if mid_count >= 4 else mid_count
                b2 = mid_count - b1
                c1 = max(2, max_count // 2) if max_count >= 4 else max_count
                c2 = max_count - c1
                segments = [(fill_color, fill_count)]
                if b2 > 0 and c2 > 0:
                    segments.extend([(mid_color, b1), (max_color, c1), (mid_color, b2), (max_color, c2)])
                else:
                    segments.extend([(mid_color, mid_count), (max_color, max_count)])

        seq = _expand_segments(segments)
        if len(seq) != perimeter_len:
            continue
        if rng.random() < 0.5:
            seq = list(reversed(seq))
        seq = _rotate_to_break_wrap(seq, rng)
        seq_counts = Counter(seq)
        if seq_counts != Counter(counts_by_color):
            continue
        if anti_shortcut:
            cyclic = _cyclic_segments(seq)
            shortest = min(length for _, length in cyclic)
            shortest_colors = {color for color, length in cyclic if length == shortest}
            if fill_color in shortest_colors:
                continue
        return seq, seq_counts
    raise RuntimeError("failed to build a non-degenerate boundary sequence")


def _rect_gap_ok(rect: dict, others: list[dict], gap: int) -> bool:
    r0 = rect["top"] - gap
    c0 = rect["left"] - gap
    r1 = rect["top"] + rect["height"] - 1 + gap
    c1 = rect["left"] + rect["width"] - 1 + gap
    for other in others:
        or0 = other["top"] - gap
        oc0 = other["left"] - gap
        or1 = other["top"] + other["height"] - 1 + gap
        oc1 = other["left"] + other["width"] - 1 + gap
        if not (r1 < or0 or or1 < r0 or c1 < oc0 or oc1 < c0):
            return False
    return True


def _place_rectangles(h: int, w: int, count: int, rng: random.Random) -> list[dict] | None:
    rects: list[dict] = []
    size_hi = 5 if count >= 5 else (6 if count == 4 else 7)
    for _ in range(300):
        rects.clear()
        for _idx in range(count):
            placed = False
            for _ in range(200):
                height = rng.randint(4, size_hi)
                width = rng.randint(4, size_hi)
                top = rng.randint(1, h - height - 1)
                left = rng.randint(1, w - width - 1)
                rect = {"top": top, "left": left, "height": height, "width": width}
                if _rect_gap_ok(rect, rects, gap=1):
                    rects.append(rect)
                    placed = True
                    break
            if not placed:
                break
        if len(rects) == count:
            return rects
    return None


def _anchor_candidates(rect: dict, shape: tuple[int, int]) -> list[tuple[int, int]]:
    top = rect["top"]
    left = rect["left"]
    height = rect["height"]
    width = rect["width"]
    out = []
    for c in range(left, left + width):
        if top - 1 >= 0:
            out.append((top - 1, c))
        if top + height < shape[0]:
            out.append((top + height, c))
    for r in range(top, top + height):
        if left - 1 >= 0:
            out.append((r, left - 1))
        if left + width < shape[1]:
            out.append((r, left + width))
    return out


def _bfs_path(blocked: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]] | None:
    h, w = blocked.shape
    q = deque([start])
    prev = {start: None}
    while q:
        r, c = q.popleft()
        if (r, c) == end:
            break
        for nr, nc in neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if blocked[nr, nc] and (nr, nc) != end:
                continue
            if (nr, nc) in prev:
                continue
            prev[(nr, nc)] = (r, c)
            q.append((nr, nc))
    if end not in prev:
        return None
    path = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _connect_rectangles(grid: np.ndarray, rects: list[dict], palette: list[int], rng: random.Random, fill_bias: int | None) -> bool:
    centers = [
        (rect["top"] + rect["height"] // 2, rect["left"] + rect["width"] // 2, idx)
        for idx, rect in enumerate(rects)
    ]
    centers.sort()
    order = [idx for _, _, idx in centers]

    rect_mask = np.zeros_like(grid, dtype=bool)
    for rect in rects:
        rect_mask[rect["top"] : rect["top"] + rect["height"], rect["left"] : rect["left"] + rect["width"]] = True

    occupied = grid != BG
    for a_idx, b_idx in zip(order, order[1:]):
        rect_a = rects[a_idx]
        rect_b = rects[b_idx]
        candidates_a = [p for p in _anchor_candidates(rect_a, grid.shape) if not rect_mask[p] and not occupied[p]]
        candidates_b = [p for p in _anchor_candidates(rect_b, grid.shape) if not rect_mask[p] and not occupied[p]]
        if not candidates_a or not candidates_b:
            return False

        pairs = [(abs(a[0] - b[0]) + abs(a[1] - b[1]), a, b) for a in candidates_a for b in candidates_b]
        pairs.sort(key=lambda item: item[0])
        path = None
        for _dist, start, end in pairs[:25]:
            blocked = rect_mask | occupied
            blocked[start] = False
            blocked[end] = False
            path = _bfs_path(blocked, start, end)
            if path is not None:
                break
        if path is None:
            return False

        if fill_bias is not None and rng.random() < 0.7:
            color = fill_bias
        else:
            color = rng.choice(palette)
        for r, c in path:
            grid[r, c] = color
            occupied[r, c] = True
    return True


def _region_bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _validate_example(grid: np.ndarray, meta: dict) -> bool:
    regions = [region for region in _collect_passable_regions(grid) if not region["touches_border"]]
    if not (3 <= len(regions) <= 5):
        return False

    marked = [region for region in regions if region["marker_count"] == 1]
    unmarked = [region for region in regions if region["marker_count"] == 0]
    if len(marked) != meta["marked_count"] or not (2 <= len(marked) <= 3):
        return False
    if len(unmarked) < 1:
        return False
    if int(np.count_nonzero(grid == MARKER)) != len(marked):
        return False

    expected_fill_colors = []
    for region in marked:
        counts = Counter(int(grid[r, c]) for r, c in region["boundary"] if int(grid[r, c]) != MARKER)
        if len(counts) < 2:
            return False
        min_count = min(counts.values())
        winners = [color for color, count in counts.items() if count == min_count]
        if len(winners) != 1:
            return False
        expected_fill_colors.append(winners[0])

    if len(expected_fill_colors) != len(meta["expected_fill_colors"]):
        return False

    output = main(grid)
    if np.array_equal(output, grid):
        return False

    for region in marked:
        counts = Counter(int(grid[r, c]) for r, c in region["boundary"] if int(grid[r, c]) != MARKER)
        fill_color = min(counts, key=counts.get)
        if any(int(output[r, c]) != fill_color for r, c in region["coords"]):
            return False

    for region in unmarked:
        if any(int(output[r, c]) != int(grid[r, c]) for r, c in region["coords"]):
            return False

    if np.any(output[(grid != BG) & (grid != MARKER)] != grid[(grid != BG) & (grid != MARKER)]):
        return False

    if not any(len(Counter(int(grid[r, c]) for r, c in region["boundary"])) >= 2 for region in unmarked):
        return False

    draw_counts = Counter(int(value) for value in grid.ravel() if int(value) not in (BG, MARKER))
    global_min = min(draw_counts.values())
    global_min_colors = {color for color, count in draw_counts.items() if count == global_min}
    fill_set = set(expected_fill_colors)
    if len(fill_set) == 1 and next(iter(fill_set)) in global_min_colors:
        return False

    special_seq = meta["special_boundary_seq"]
    special_counts = Counter(special_seq)
    special_fill = min(special_counts, key=special_counts.get)
    segments = _cyclic_segments(special_seq)
    shortest = min(length for _, length in segments)
    shortest_colors = {color for color, length in segments if length == shortest}
    if special_fill in shortest_colors:
        return False

    expected_boxes = sorted(meta["interior_boxes"])
    observed_boxes = sorted(_region_bbox(region["coords"]) for region in regions)
    if expected_boxes != observed_boxes:
        return False

    return True


def _gen_attempt(rng: random.Random) -> tuple[np.ndarray, dict] | None:
    h = rng.randint(18, 24)
    w = rng.randint(18, 24)
    cavity_count = rng.randint(3, 5)
    marked_count = rng.randint(2, min(3, cavity_count - 1))
    palette_size = rng.choice([2, 3])
    palette = rng.sample(list(DRAW_COLORS), palette_size)

    rects = _place_rectangles(h, w, cavity_count, rng)
    if rects is None:
        return None

    grid = np.full((h, w), BG, dtype=int)
    marked_indices = set(rng.sample(range(cavity_count), marked_count))

    same_fill_branch = rng.random() < 0.45
    common_fill = rng.choice(palette) if same_fill_branch else None
    expected_fill_colors = []
    special_boundary_seq = None
    interior_boxes = []

    marked_order = [idx for idx in range(cavity_count) if idx in marked_indices]
    forced_colors: dict[int, int] = {}
    if same_fill_branch:
        for idx in marked_order:
            forced_colors[idx] = common_fill
    else:
        diverse = rng.sample(palette, min(len(palette), len(marked_order)))
        for pos, idx in enumerate(marked_order):
            forced_colors[idx] = diverse[pos % len(diverse)]

    for idx, rect in enumerate(rects):
        top = rect["top"]
        left = rect["left"]
        height = rect["height"]
        width = rect["width"]
        perimeter = _perimeter_coords(top, left, height, width)
        anti_shortcut = idx == marked_order[0]
        force_fill_color = forced_colors.get(idx)
        try:
            seq, counts = _make_boundary_sequence(
                len(perimeter),
                palette,
                rng,
                force_fill_color=force_fill_color,
                anti_shortcut=anti_shortcut,
            )
        except RuntimeError:
            return None
        for (r, c), color in zip(perimeter, seq):
            grid[r, c] = color

        fill_color = min(counts, key=counts.get)
        if idx in marked_indices:
            expected_fill_colors.append(fill_color)
            interior_h = height - 2
            interior_w = width - 2
            mr = rng.randint(top + 1, top + interior_h)
            mc = rng.randint(left + 1, left + interior_w)
            grid[mr, mc] = MARKER
        if anti_shortcut:
            special_boundary_seq = seq
        interior_boxes.append((top + 1, left + 1, top + height - 2, left + width - 2))

    fill_bias = common_fill if same_fill_branch else rng.choice(expected_fill_colors)
    if not _connect_rectangles(grid, rects, palette, rng, fill_bias):
        return None

    meta = {
        "marked_count": marked_count,
        "expected_fill_colors": expected_fill_colors,
        "special_boundary_seq": special_boundary_seq,
        "interior_boxes": interior_boxes,
    }
    if not _validate_example(grid, meta):
        return None
    return grid, meta


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        result = _gen_attempt(random.Random())
        if result is not None:
            grid, _meta = result
            return grid
    raise RuntimeError("failed to generate a valid cavity-frequency ARC input")
