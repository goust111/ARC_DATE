import random
from collections import Counter

import numpy as np


COLORS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
FREQ_PATTERNS = {
    3: [
        (1, 2, 4),
        (1, 2, 5),
        (1, 3, 4),
        (1, 3, 5),
        (1, 4, 5),
        (2, 3, 4),
        (2, 3, 5),
        (2, 4, 5),
    ],
    4: [
        (1, 2, 3, 4),
        (1, 2, 3, 5),
        (1, 2, 4, 5),
        (1, 3, 4, 5),
        (2, 3, 4, 5),
    ],
}
CHAIN_PATTERNS = [
    [(0, 0), (1, 1)],
    [(0, 0), (2, 0)],
    [(0, 0), (0, 2)],
    [(0, 0), (1, 1), (2, 0)],
    [(0, 0), (1, 1), (0, 2)],
    [(0, 0), (2, 0), (2, 2)],
]
MAX_GENERATION_ATTEMPTS = 500


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_non_black(grid: np.ndarray) -> np.ndarray:
    box = _bbox(grid != 0)
    if box is None:
        return np.zeros((1, 1), dtype=int)
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _square_box(r: int, c: int, radius: int) -> tuple[int, int, int, int]:
    return r - radius, c - radius, r + radius, c + radius


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _extract_points(grid: np.ndarray) -> list[tuple[int, int, int]]:
    return [(int(r), int(c), int(grid[r, c])) for r, c in np.argwhere(grid != 0)]


def _render_scene(
    points: list[tuple[int, int, int]],
    shape: tuple[int, int],
    *,
    with_stats: bool = False,
):
    freq = Counter(color for _r, _c, color in points)
    ordered = sorted(points, key=lambda item: (-freq[item[2]], item[0], item[1]))
    canvas = np.zeros(shape, dtype=int)
    events: list[dict] = []

    for r, c, color in ordered:
        radius = int(freq[color] - 1)
        r0, c0, r1, c1 = _square_box(r, c, radius)
        assert 0 <= r0 <= r1 < shape[0]
        assert 0 <= c0 <= c1 < shape[1]

        if with_stats:
            before = canvas[r0 : r1 + 1, c0 : c1 + 1].copy()
            overwritten = before[(before != 0) & (before != color)]
            if overwritten.size:
                events.append(
                    {
                        "color": int(color),
                        "freq": int(freq[color]),
                        "overwrote": Counter(map(int, overwritten.flat)),
                    }
                )

        canvas[r0 : r1 + 1, c0 : c1 + 1] = int(color)

    if with_stats:
        return canvas, {"freq": dict(freq), "events": events}
    return canvas


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand isolated colored pixels into frequency-sized centered squares, let smaller squares cover larger ones, then crop the result.

    Input structure:
    - The grid is a black background with several isolated non-black pixels.
    - Pixels of the same color may appear multiple times; different colors have different total frequencies.
    - Each colored pixel is used as the center of a square, and valid inputs keep those squares inside the grid.

    Rule:
    - Count how many times each non-black color appears.
    - For every non-black pixel of color c, draw a solid odd-sided square centered on that pixel with side length 2*freq(c)-1.
    - Draw larger squares first and smaller squares later so that later, smaller squares overwrite earlier, larger ones; ties are broken by row-major order of centers.
    - Crop the final painted canvas to the minimal bounding box containing all non-black pixels.

    Key insight:
    - Color frequency is not a label to select an object; it encodes square size, and the required output depends on the full chain count -> size -> overlap order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    points = _extract_points(g)
    assert points, "expected at least one non-black pixel"
    painted = _render_scene(points, g.shape)
    return _crop_non_black(painted)


def _sample_frequencies() -> dict[int, int]:
    num_colors = random.choice([3, 4])
    pattern = list(random.choice(FREQ_PATTERNS[num_colors]))
    random.shuffle(pattern)
    colors = random.sample(COLORS, num_colors)
    return {int(color): int(freq) for color, freq in zip(colors, pattern)}


def _interior_cells(height: int, width: int, margin: int = 5) -> list[tuple[int, int]]:
    return [(r, c) for r in range(margin, height - margin) for c in range(margin, width - margin)]


def _isolated(center: tuple[int, int], occupied: list[tuple[int, int, int]]) -> bool:
    r, c = center
    return all(abs(r - rr) + abs(c - cc) >= 2 for rr, cc, _color in occupied)


def _pattern_variants(length: int) -> list[list[tuple[int, int]]]:
    variants: list[list[tuple[int, int]]] = []
    for base in CHAIN_PATTERNS:
        if len(base) != length:
            continue
        for swap in (False, True):
            for sr in (-1, 1):
                for sc in (-1, 1):
                    pts = []
                    for dr, dc in base:
                        if swap:
                            dr, dc = dc, dr
                        pts.append((sr * dr, sc * dc))
                    if pts not in variants:
                        variants.append(pts)
    return variants


def _place_chain(
    freq_by_color: dict[int, int],
    height: int,
    width: int,
) -> list[tuple[int, int, int]] | None:
    ordered_colors = sorted(freq_by_color, key=lambda color: freq_by_color[color], reverse=True)
    chain_len = 3 if len(ordered_colors) >= 3 and random.random() < 0.7 else 2
    chain_colors = ordered_colors[:chain_len]
    variants = _pattern_variants(chain_len)
    random.shuffle(variants)

    row_choices = list(range(7, height - 7))
    col_choices = list(range(7, width - 7))
    random.shuffle(row_choices)
    random.shuffle(col_choices)

    for offsets in variants:
        for base_r in row_choices:
            for base_c in col_choices:
                placed: list[tuple[int, int, int]] = []
                ok = True
                for color, (dr, dc) in zip(chain_colors, offsets):
                    r = base_r + dr
                    c = base_c + dc
                    if not (5 <= r < height - 5 and 5 <= c < width - 5):
                        ok = False
                        break
                    if not _isolated((r, c), placed):
                        ok = False
                        break
                    placed.append((r, c, int(color)))
                if ok:
                    return placed
    return None


def _try_place_point(
    color: int,
    freq_by_color: dict[int, int],
    occupied: list[tuple[int, int, int]],
    height: int,
    width: int,
    *,
    avoid_all_overlap: bool,
) -> tuple[int, int, int] | None:
    radius = int(freq_by_color[color] - 1)
    candidates = _interior_cells(height, width)
    random.shuffle(candidates)

    for r, c in candidates:
        if not _isolated((r, c), occupied):
            continue
        new_box = _square_box(r, c, radius)
        ok = True
        for rr, cc, other_color in occupied:
            other_box = _square_box(rr, cc, int(freq_by_color[other_color] - 1))
            if other_color != color and _boxes_overlap(new_box, other_box):
                ok = False
                break
            if avoid_all_overlap and _boxes_overlap(new_box, other_box):
                ok = False
                break
        if ok:
            return int(r), int(c), int(color)
    return None


def _build_input_grid(freq_by_color: dict[int, int], height: int, width: int) -> np.ndarray | None:
    occupied = _place_chain(freq_by_color, height, width)
    if occupied is None:
        return None

    max_color = max(freq_by_color, key=lambda color: freq_by_color[color])
    remaining = Counter(freq_by_color)
    for _r, _c, color in occupied:
        remaining[color] -= 1

    if remaining[max_color] > 0:
        showcase = _try_place_point(
            max_color,
            freq_by_color,
            occupied,
            height,
            width,
            avoid_all_overlap=True,
        )
        if showcase is None:
            return None
        occupied.append(showcase)
        remaining[max_color] -= 1

    placement_order = sorted(freq_by_color, key=lambda color: (freq_by_color[color], random.random()), reverse=True)
    for color in placement_order:
        for _ in range(int(remaining[color])):
            point = _try_place_point(
                color,
                freq_by_color,
                occupied,
                height,
                width,
                avoid_all_overlap=False,
            )
            if point is None:
                return None
            occupied.append(point)

    grid = np.zeros((height, width), dtype=int)
    for r, c, color in occupied:
        grid[r, c] = int(color)
    return grid


def _semantic_checks(inp: np.ndarray) -> bool:
    points = _extract_points(inp)
    if not (3 <= len(set(color for _r, _c, color in points)) <= 4):
        return False

    freq = Counter(color for _r, _c, color in points)
    if any(count < 1 or count > 5 for count in freq.values()):
        return False
    if len(set(freq.values())) != len(freq):
        return False

    for r, c, _color in points:
        if not (5 <= r < inp.shape[0] - 5 and 5 <= c < inp.shape[1] - 5):
            return False
    for idx, (r0, c0, _color0) in enumerate(points):
        for r1, c1, _color1 in points[idx + 1 :]:
            if abs(r0 - r1) + abs(c0 - c1) < 2:
                return False

    painted, stats = _render_scene(points, inp.shape, with_stats=True)
    if any(np.count_nonzero(painted == color) == 0 for color in freq):
        return False
    if not stats["events"]:
        return False

    kept_cover_event = False
    for event in stats["events"]:
        if np.count_nonzero(painted == event["color"]) == 0:
            continue
        if all(np.count_nonzero(painted == other_color) > 0 for other_color in event["overwrote"]):
            kept_cover_event = True
            break
    if not kept_cover_event:
        return False

    cropped = _crop_non_black(painted)
    if cropped.shape == inp.shape:
        return False
    if np.count_nonzero(painted) <= np.count_nonzero(inp):
        return False
    if not (1 in freq.values() or 5 in freq.values()):
        return False

    try:
        solved = main(inp)
    except Exception:
        return False
    return bool(np.array_equal(solved, cropped))


def _gen_attempt() -> np.ndarray | None:
    freq_by_color = _sample_frequencies()
    total_points = sum(freq_by_color.values())
    span_bias = max(freq_by_color.values()) + len(freq_by_color)
    height = random.randint(max(20, span_bias + 14), 26)
    width = random.randint(max(20, total_points // 2 + 16), 26)

    grid = _build_input_grid(freq_by_color, height, width)
    if grid is None:
        return None
    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
