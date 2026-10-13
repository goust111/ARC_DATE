from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.is_dir() and str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))
try:
    from arc_utils import get_bg as _arc_get_bg
except Exception:  # pragma: no cover - keep this file loadable without external deps.
    _arc_get_bg = None


GEN_RETRIES = 400
ARC_COLORS = tuple(range(10))


def _get_bg(grid: np.ndarray) -> int:
    if _arc_get_bg is not None:
        return int(_arc_get_bg(grid))
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _beat_columns(period: int, width: int) -> list[int]:
    return list(range(int(period), int(width), int(period)))


def _scan_left_stripes(grid: np.ndarray) -> list[dict[str, int]]:
    h, _w = grid.shape
    segments: list[dict[str, int]] = []
    r = 0
    while r < h:
        color = int(grid[r, 0])
        r1 = r
        while r1 + 1 < h and int(grid[r1 + 1, 0]) == color:
            r1 += 1
        segments.append(
            {
                "r0": int(r),
                "r1": int(r1),
                "height": int(r1 - r + 1),
                "color": color,
            }
        )
        r = r1 + 1
    return segments


def _row_marker(grid: np.ndarray, row: int, bg: int) -> tuple[int, int] | None:
    if grid.shape[1] <= 1:
        return None
    cols = np.flatnonzero(grid[int(row), 1:] != bg)
    assert cols.size <= 1, "each row must contain at most one marker pixel"
    if cols.size == 0:
        return None
    col = int(cols[0] + 1)
    return col, int(grid[int(row), col])


def _render_output_from_input(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("input must be a 2D grid")
    if g.shape[1] == 1:
        return g.copy()

    bg = _get_bg(g[:, 1:])
    segments = _scan_left_stripes(g)
    out = g.copy()

    for seg in segments:
        period = int(seg["height"])
        stripe_color = int(seg["color"])
        beats = _beat_columns(period, g.shape[1])
        for r in range(seg["r0"], seg["r1"] + 1):
            marker = _row_marker(g, r, bg)
            active_color = stripe_color
            for c in beats:
                if marker is not None and c == marker[0]:
                    active_color = int(marker[1])
                out[r, c] = active_color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each left-edge stripe across its own row band using a beat spacing equal to the stripe height.

    Input structure:
    - The leftmost column is tiled top-to-bottom by vertical stripe segments with distinct colors and heights.
    - All other cells are background except for a few isolated single-cell colored markers, with at most one marker in any row.

    Rule:
    - A stripe of height h controls every row in its band.
    - In those rows, paint the cells at columns h, 2h, 3h, ... with the stripe color.
    - If a marker in that row lies exactly on one of those beat columns, keep that marker color there and recolor every later beat in the same row to that marker color.
    - A marker off the beat stays unchanged and does not alter the periodic fill.

    Key insight:
    - The segment height is reused as the horizontal period, so only markers aligned modulo their own band height can take over the relay.
    """
    return _render_output_from_input(grid)


def _build_segments(heights: list[int], colors: list[int]) -> list[dict[str, int]]:
    segments: list[dict[str, int]] = []
    r0 = 0
    for height, color in zip(heights, colors, strict=False):
        segments.append(
            {
                "r0": int(r0),
                "r1": int(r0 + height - 1),
                "height": int(height),
                "color": int(color),
            }
        )
        r0 += height
    return segments


def _sample_heights() -> list[int] | None:
    for _ in range(200):
        count = random.randint(3, 5)
        heights = random.sample(range(2, 8), count)
        total = sum(heights)
        if 9 <= total <= 24:
            random.shuffle(heights)
            return heights
    return None


def _choose_roles(segments: list[dict[str, int]], width: int) -> tuple[int, int, int, int] | None:
    options: list[tuple[int, int, int, int]] = []
    for relay_idx, relay_seg in enumerate(segments):
        relay_beats = _beat_columns(relay_seg["height"], width)
        if len(relay_beats) < 2:
            continue
        for off_idx, off_seg in enumerate(segments):
            if off_idx == relay_idx:
                continue
            for plain_idx in range(len(segments)):
                if plain_idx in {relay_idx, off_idx}:
                    continue
                for col in relay_beats[:-1]:
                    if col % off_seg["height"] != 0:
                        options.append((relay_idx, off_idx, plain_idx, col))
    if not options:
        return None
    return random.choice(options)


def _pick_row(seg: dict[str, int], used_rows: set[int]) -> int | None:
    rows = [r for r in range(seg["r0"], seg["r1"] + 1) if r not in used_rows]
    if not rows:
        return None
    row = random.choice(rows)
    used_rows.add(row)
    return row


def _pick_nonlast_beat(period: int, width: int, avoid: set[int] | None = None) -> int | None:
    avoid = set() if avoid is None else set(avoid)
    beats = [c for c in _beat_columns(period, width)[:-1] if c not in avoid]
    if not beats:
        return None
    return random.choice(beats)


def _pick_nearby_offbeat(anchor: int, period: int, width: int, avoid: set[int] | None = None) -> int | None:
    avoid = set() if avoid is None else set(avoid)
    for delta in range(1, width + 1):
        for sign in (1, -1):
            col = anchor + sign * delta
            if 1 <= col < width and col not in avoid and col % period != 0:
                return int(col)
    return None


def _render_input(
    segments: list[dict[str, int]],
    width: int,
    bg: int,
    markers: dict[int, tuple[int, int]],
) -> np.ndarray:
    height = int(segments[-1]["r1"] + 1)
    grid = np.full((height, width), int(bg), dtype=int)
    for seg in segments:
        grid[seg["r0"] : seg["r1"] + 1, 0] = int(seg["color"])
    for row, (col, color) in markers.items():
        grid[int(row), int(col)] = int(color)
    return grid


def _render_output_from_plan(
    grid: np.ndarray,
    segments: list[dict[str, int]],
    markers: dict[int, tuple[int, int]],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    width = int(out.shape[1])
    for seg in segments:
        stripe_color = int(seg["color"])
        beats = _beat_columns(seg["height"], width)
        for row in range(seg["r0"], seg["r1"] + 1):
            marker = markers.get(row)
            active_color = stripe_color
            for col in beats:
                if marker is not None and col == marker[0]:
                    active_color = int(marker[1])
                out[row, col] = active_color
    return out


def _validate_instance(
    grid: np.ndarray,
    expected: np.ndarray,
    segments: list[dict[str, int]],
    markers: dict[int, tuple[int, int]],
    bg: int,
    plain_idx: int,
) -> bool:
    height, width = map(int, grid.shape)
    if height > 30 or width > 30:
        return False
    if len(set(seg["color"] for seg in segments)) != len(segments):
        return False
    if len(set(seg["height"] for seg in segments)) != len(segments):
        return False
    if any(len(_beat_columns(seg["height"], width)) < 2 for seg in segments):
        return False
    if np.count_nonzero(grid[:, 1:] != bg) != len(markers):
        return False
    if len(markers) < 4:
        return False
    if np.array_equal(grid, expected):
        return False
    if np.count_nonzero(grid != expected) < max(6, 2 * len(segments)):
        return False

    row_to_seg: dict[int, dict[str, int]] = {}
    for seg in segments:
        for row in range(seg["r0"], seg["r1"] + 1):
            row_to_seg[row] = seg

    plain_seg = segments[plain_idx]
    if any(plain_seg["r0"] <= row <= plain_seg["r1"] for row in markers):
        return False

    relay_ok = False
    offbeat_ok = False
    plain_ok = False
    contrast_ok = False
    seen_by_col: dict[int, set[bool]] = {}

    for row, (col, color) in markers.items():
        seg = row_to_seg[row]
        period = int(seg["height"])
        stripe_color = int(seg["color"])
        beats = _beat_columns(period, width)
        on_beat = (col % period == 0)
        seen_by_col.setdefault(int(col), set()).add(bool(on_beat))
        if on_beat and any(beat > col for beat in beats):
            if all(int(expected[row, beat]) == int(color) for beat in beats if beat >= col):
                relay_ok = True
        if not on_beat:
            if all(int(expected[row, beat]) == stripe_color for beat in beats):
                offbeat_ok = True

    for states in seen_by_col.values():
        if len(states) >= 2:
            contrast_ok = True
            break

    for row in range(plain_seg["r0"], plain_seg["r1"] + 1):
        beats = _beat_columns(plain_seg["height"], width)
        if all(int(expected[row, beat]) == int(plain_seg["color"]) for beat in beats):
            plain_ok = True
            break

    if not (relay_ok and offbeat_ok and plain_ok and contrast_ok):
        return False

    solved = main(grid)
    return np.array_equal(solved, expected)


def _gen_attempt() -> np.ndarray | None:
    heights = _sample_heights()
    if heights is None:
        return None

    width_min = max(12, 2 * max(heights) + 1)
    width_max = min(30, width_min + 10)
    width = random.randint(width_min, width_max)

    bg = random.choice(ARC_COLORS)
    non_bg = [color for color in ARC_COLORS if color != bg]
    stripe_colors = random.sample(non_bg, len(heights))
    marker_source = [color for color in non_bg if color not in stripe_colors]
    marker_palette = random.sample(marker_source, random.randint(2, min(4, len(marker_source))))

    segments = _build_segments(heights, stripe_colors)
    roles = _choose_roles(segments, width)
    if roles is None:
        return None
    relay_idx, off_idx, plain_idx, contrast_col = roles

    relay_seg = segments[relay_idx]
    off_seg = segments[off_idx]

    used_rows: set[int] = set()
    markers: dict[int, tuple[int, int]] = {}

    relay_on_row = _pick_row(relay_seg, used_rows)
    relay_off_row = _pick_row(relay_seg, used_rows)
    off_off_row = _pick_row(off_seg, used_rows)
    off_on_row = _pick_row(off_seg, used_rows)
    if None in {relay_on_row, relay_off_row, off_off_row, off_on_row}:
        return None

    relay_off_col = _pick_nearby_offbeat(contrast_col, relay_seg["height"], width)
    off_on_col = _pick_nonlast_beat(off_seg["height"], width)
    if relay_off_col is None or off_on_col is None:
        return None

    markers[int(relay_on_row)] = (int(contrast_col), int(random.choice(marker_palette)))
    markers[int(relay_off_row)] = (int(relay_off_col), int(random.choice(marker_palette)))
    markers[int(off_off_row)] = (int(contrast_col), int(random.choice(marker_palette)))
    markers[int(off_on_row)] = (int(off_on_col), int(random.choice(marker_palette)))

    extra_segments = [idx for idx in range(len(segments)) if idx != plain_idx]
    random.shuffle(extra_segments)
    for idx in extra_segments:
        seg = segments[idx]
        if random.random() < 0.55:
            row = _pick_row(seg, used_rows)
            if row is None:
                continue
            if random.random() < 0.5:
                col = _pick_nonlast_beat(seg["height"], width)
            else:
                anchor = _pick_nonlast_beat(seg["height"], width)
                col = None if anchor is None else _pick_nearby_offbeat(anchor, seg["height"], width)
            if col is not None:
                markers[int(row)] = (int(col), int(random.choice(marker_palette)))

    grid = _render_input(segments, width, bg, markers)
    expected = _render_output_from_plan(grid, segments, markers)
    if not _validate_instance(grid, expected, segments, markers, bg, plain_idx):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic-relay ARC input")
