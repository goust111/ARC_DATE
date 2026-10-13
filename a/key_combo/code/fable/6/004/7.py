import random
from collections import Counter
from itertools import combinations

import numpy as np


GEN_ATTEMPTS = 600
MARKER_ATTEMPTS = 200


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    remaining = {tuple(map(int, rc)) for rc in np.argwhere(grid == int(color))}
    components = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        comp = [start]
        while stack:
            r, c = stack.pop()
            for nr, nc in _neighbors4(r, c):
                cell = (nr, nc)
                if cell in remaining:
                    remaining.remove(cell)
                    stack.append(cell)
                    comp.append(cell)
        components.append(np.array(comp, dtype=int))
    return components


def _find_barrier_color(grid: np.ndarray, bg: int) -> int:
    foreground = Counter(int(v) for v in grid.flatten() if int(v) != int(bg))
    assert foreground, "expected non-background pixels"
    top_count = max(foreground.values())
    candidates = [color for color, count in foreground.items() if count == top_count]
    assert len(candidates) == 1, "expected a unique most frequent foreground color"
    return int(candidates[0])


def _extract_segments(grid: np.ndarray, barrier: int) -> list[dict]:
    segments = []
    for coords in _components_of_color(grid, barrier):
        rows = coords[:, 0]
        cols = coords[:, 1]
        assert np.all(cols == cols[0]), "barrier component must be vertical"
        r0 = int(rows.min())
        r1 = int(rows.max())
        col = int(cols[0])
        expected = r1 - r0 + 1
        assert int(coords.shape[0]) == expected, "barrier segment must be contiguous"
        assert expected % 2 == 1, "barrier length must be odd"
        segments.append(
            {
                "r0": r0,
                "r1": r1,
                "col": col,
                "mid": (r0 + r1) // 2,
                "length": expected,
            }
        )
    assert segments, "expected at least one barrier segment"
    segments.sort(key=lambda seg: (seg["r0"], seg["col"]))
    return segments


def _marker_cells(grid: np.ndarray, bg: int, barrier: int) -> list[tuple[int, int]]:
    markers = [tuple(map(int, rc)) for rc in np.argwhere((grid != int(bg)) & (grid != int(barrier)))]
    assert markers, "expected at least one marker"
    marker_set = set(markers)
    for r, c in markers:
        for nr, nc in _neighbors4(r, c):
            assert (nr, nc) not in marker_set, "markers must be isolated single pixels"
    return markers


def _segment_for_marker(r: int, c: int, segments: list[dict]) -> tuple[dict, int]:
    candidates = [seg for seg in segments if seg["r0"] <= r <= seg["r1"]]
    assert len(candidates) == 1, "each marker row must belong to exactly one barrier segment"
    seg = candidates[0]
    assert c != seg["col"], "marker cannot sit on the barrier column"
    side = -1 if c < seg["col"] else 1
    return seg, side


def _planned_moves(grid: np.ndarray) -> tuple[int, int, list[dict], list[tuple[int, int]], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    barrier = _find_barrier_color(g, bg)
    segments = _extract_segments(g, barrier)
    markers = _marker_cells(g, bg, barrier)
    moves = []
    destinations = set()

    for r, c in markers:
        seg, side = _segment_for_marker(r, c, segments)
        rr = 2 * seg["mid"] - r
        cc = seg["col"] + 1 if side < 0 else seg["col"] - 1
        assert _in_bounds(g.shape, rr, cc), "reflected exit must stay in bounds"
        assert int(g[rr, cc]) == int(bg), "exit cell must start as background"
        move = {
            "src": (r, c),
            "dst": (rr, cc),
            "color": int(g[r, c]),
            "segment": seg,
            "side": side,
            "distance": abs(c - seg["col"]),
        }
        assert move["dst"] not in destinations, "marker destinations must be unique"
        destinations.add(move["dst"])
        moves.append(move)

    return bg, barrier, segments, markers, moves


def _apply_moves(grid: np.ndarray, moves: list[dict], bg: int, *, mode: str) -> np.ndarray:
    out = np.array(grid, copy=True)
    global_mid = None
    if mode == "global_mid":
        segments = {id(move["segment"]): move["segment"] for move in moves}.values()
        global_mid = (min(seg["r0"] for seg in segments) + max(seg["r1"] for seg in segments)) // 2
    for move in moves:
        r, c = move["src"]
        out[r, c] = int(bg)
    for move in moves:
        seg = move["segment"]
        r, c = move["src"]
        side = move["side"]
        if mode == "correct":
            rr = 2 * seg["mid"] - r
            cc = seg["col"] + 1 if side < 0 else seg["col"] - 1
        elif mode == "same_row":
            rr = r
            cc = seg["col"] + 1 if side < 0 else seg["col"] - 1
        elif mode == "keep_distance":
            rr = 2 * seg["mid"] - r
            cc = seg["col"] + move["distance"] if side < 0 else seg["col"] - move["distance"]
        elif mode == "global_mid":
            rr = 2 * global_mid - r
            cc = seg["col"] + 1 if side < 0 else seg["col"] - 1
        else:
            raise ValueError(f"unknown mode: {mode}")
        assert _in_bounds(out.shape, rr, cc)
        out[rr, cc] = move["color"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each marker through its vertical barrier segment and make it emerge on the mirrored row.

    Input structure:
    - The grid has a dominant background color.
    - The unique most frequent foreground color forms one or more odd-length vertical barrier segments.
    - Every other foreground pixel is an isolated single-cell marker whose row lies inside exactly one segment span and which sits on one side of that segment.

    Rule:
    - For each marker at row r with associated barrier-segment midpoint row m, compute the reflected row r' = 2m - r.
    - Erase the marker from its source cell and repaint it on the opposite side of that same segment, in the cell immediately adjacent to the barrier, at row r'.
    - Keep every barrier segment unchanged.

    Key insight:
    - The barrier acts both as the crossed wall and as the mirror axis: the exit row is determined by reflection around that segment's own midpoint, not by straight-through motion or by preserving the original horizontal offset.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg, _barrier, _segments, _markers, moves = _planned_moves(g)
    return _apply_moves(g, moves, bg, mode="correct")


def _random_gap_partition(extra: int, parts: int) -> list[int]:
    vals = [0] * parts
    for _ in range(extra):
        vals[random.randrange(parts)] += 1
    return vals


def _sample_spans(n_segments: int) -> tuple[int, list[tuple[int, int]]]:
    lengths = [random.choice((3, 5, 7, 9)) for _ in range(n_segments)]
    mandatory = sum(lengths) + (n_segments + 1)
    if mandatory > 30:
        raise ValueError("cannot fit spans")
    extra = random.randint(0, min(8, 30 - mandatory))
    gaps = [1 + val for val in _random_gap_partition(extra, n_segments + 1)]
    spans = []
    row = gaps[0]
    for length, gap_after in zip(lengths, gaps[1:]):
        spans.append((row, row + length - 1))
        row += length + gap_after
    return row, spans


def _sample_columns(width: int, n_segments: int) -> list[int] | None:
    candidates = list(range(4, width - 4))
    combos = [combo for combo in combinations(candidates, n_segments) if all(b - a >= 4 for a, b in zip(combo, combo[1:]))]
    if not combos:
        return None
    return list(random.choice(combos))


def _marker_spec_list(segments: list[dict]) -> list[dict]:
    specs = [
        {"segment_idx": 0, "force_mid": False, "require_non_mid": True, "require_far": True},
        {"segment_idx": 1, "force_mid": True, "require_non_mid": False, "require_far": False},
    ]
    for idx in range(2, len(segments)):
        specs.append({"segment_idx": idx, "force_mid": None, "require_non_mid": False, "require_far": False})
    extra = random.randint(1, len(segments))
    for _ in range(extra):
        specs.append(
            {
                "segment_idx": random.randrange(len(segments)),
                "force_mid": random.choice((None, True, False)),
                "require_non_mid": False,
                "require_far": random.choice((False, True)),
            }
        )
    if len(specs) > 2:
        tail = specs[2:]
        random.shuffle(tail)
        specs[2:] = tail
    return specs


def _choose_marker_position(
    shape: tuple[int, int],
    seg: dict,
    sources: set[tuple[int, int]],
    destinations: set[tuple[int, int]],
    *,
    force_mid: bool | None,
    require_non_mid: bool,
    require_far: bool,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    h, w = shape
    rows = list(range(seg["r0"], seg["r1"] + 1))
    if force_mid is True:
        rows = [seg["mid"]]
    elif require_non_mid or force_mid is False:
        rows = [r for r in rows if r != seg["mid"]]
    if not rows:
        return None

    options = []
    for row in rows:
        for side in (-1, 1):
            max_dist = seg["col"] if side < 0 else (w - 1 - seg["col"])
            min_dist = 2 if require_far else 1
            for dist in range(min_dist, min(4, max_dist) + 1):
                src_c = seg["col"] - dist if side < 0 else seg["col"] + dist
                dst_r = 2 * seg["mid"] - row
                dst_c = seg["col"] + 1 if side < 0 else seg["col"] - 1
                src = (row, src_c)
                dst = (dst_r, dst_c)
                if not _in_bounds((h, w), *src) or not _in_bounds((h, w), *dst):
                    continue
                if src in sources or dst in destinations or src in destinations or dst in sources:
                    continue
                if any((nr, nc) in sources for nr, nc in _neighbors4(*src)):
                    continue
                options.append((src, dst))
    if not options:
        return None
    return random.choice(options)


def _assign_marker_colors(count: int, palette: list[int]) -> list[int]:
    assert len(palette) >= 2
    colors = [palette[idx % len(palette)] for idx in range(count)]
    random.shuffle(colors)
    return colors


def _build_case() -> np.ndarray | None:
    n_segments = random.randint(2, 4)
    try:
        height, spans = _sample_spans(n_segments)
    except ValueError:
        return None
    width = random.randint(max(12, 11 + 4 * (n_segments - 2)), 30)
    cols = _sample_columns(width, n_segments)
    if cols is None:
        return None

    all_colors = list(range(10))
    bg = random.choice(all_colors)
    remaining = [color for color in all_colors if color != bg]
    barrier = random.choice(remaining)
    marker_palette = random.sample([color for color in remaining if color != barrier], k=random.randint(2, 4))

    grid = np.full((height, width), bg, dtype=int)
    segments = []
    for (r0, r1), col in zip(spans, cols):
        grid[r0 : r1 + 1, col] = barrier
        segments.append({"r0": r0, "r1": r1, "col": col, "mid": (r0 + r1) // 2})

    specs = _marker_spec_list(segments)
    sources: set[tuple[int, int]] = set()
    destinations: set[tuple[int, int]] = set()
    placements = []

    for spec in specs:
        seg = segments[spec["segment_idx"]]
        placed = None
        for _ in range(MARKER_ATTEMPTS):
            candidate = _choose_marker_position(
                grid.shape,
                seg,
                sources,
                destinations,
                force_mid=spec["force_mid"],
                require_non_mid=spec["require_non_mid"],
                require_far=spec["require_far"],
            )
            if candidate is not None:
                placed = candidate
                break
        if placed is None:
            return None
        src, dst = placed
        sources.add(src)
        destinations.add(dst)
        placements.append({"src": src, "dst": dst, "segment": seg})

    colors = _assign_marker_colors(len(placements), marker_palette)
    for placement, color in zip(placements, colors):
        r, c = placement["src"]
        grid[r, c] = color

    barrier_pixels = sum(seg["r1"] - seg["r0"] + 1 for seg in segments)
    marker_counts = Counter(colors)
    if max(marker_counts.values()) >= barrier_pixels:
        return None

    out = main(grid)
    if not _validate_case(grid, out):
        return None
    return grid


def _validate_case(inp: np.ndarray, out: np.ndarray) -> bool:
    try:
        bg, barrier, segments, markers, moves = _planned_moves(inp)
    except AssertionError:
        return False

    if not np.array_equal(out, main(inp)):
        return False
    if out.shape != inp.shape:
        return False
    if np.array_equal(inp, out):
        return False

    barrier_mask = inp == barrier
    if not np.array_equal(inp[barrier_mask], out[barrier_mask]):
        return False

    marker_rows = []
    used_mids = set()
    has_mid_marker = False
    has_non_mid_marker = False
    has_far_marker = False
    for move in moves:
        src_r, src_c = move["src"]
        dst_r, dst_c = move["dst"]
        seg = move["segment"]
        used_mids.add(seg["mid"])
        marker_rows.append(src_r)
        has_mid_marker |= src_r == seg["mid"]
        has_non_mid_marker |= src_r != seg["mid"]
        has_far_marker |= move["distance"] > 1
        if out[src_r, src_c] != bg:
            return False
        if out[dst_r, dst_c] != move["color"]:
            return False
        if abs(dst_c - seg["col"]) != 1:
            return False
        if dst_r != 2 * seg["mid"] - src_r:
            return False

    if len(segments) < 2 or len(used_mids) < 2:
        return False
    if not has_mid_marker or not has_non_mid_marker or not has_far_marker:
        return False

    same_row = _apply_moves(inp, moves, bg, mode="same_row")
    keep_distance = _apply_moves(inp, moves, bg, mode="keep_distance")
    global_mid = _apply_moves(inp, moves, bg, mode="global_mid")
    if np.array_equal(out, same_row):
        return False
    if np.array_equal(out, keep_distance):
        return False
    if np.array_equal(out, global_mid):
        return False

    fg_counts = Counter(int(v) for v in inp.flatten() if int(v) != bg)
    if fg_counts[barrier] != max(fg_counts.values()):
        return False
    if sum(1 for count in fg_counts.values() if count == fg_counts[barrier]) != 1:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
