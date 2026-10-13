from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components, in_bounds, paste, scale_sprite


Coord = tuple[int, int]


PALETTE = [c for c in Color.NOT_BLACK if c != Color.GREY]
SCALE = 3
GEN_RETRIES = 400


def _thumbnail_layout(grid: np.ndarray) -> tuple[list[np.ndarray], tuple[int, int, int, int]]:
    grey = find_components(grid, bg=Color.BLACK, colors=[Color.GREY], connectivity=4)
    assert len(grey) == 2, "expected exactly two grey separators"
    grey = sorted(grey, key=lambda comp: (int(comp["bbox"][1]), int(comp["bbox"][0])))

    (r0a, c0a, r1a, c1a) = map(int, grey[0]["bbox"])
    (r0b, c0b, r1b, c1b) = map(int, grey[1]["bbox"])
    assert r0a == r0b and r1a == r1b
    assert c0a == c1a and c0b == c1b

    panel_h = r1a - r0a + 1
    panel_w = c0b - c0a - 1
    assert panel_h >= 3 and panel_w >= 3

    left_c0 = c0a - panel_w
    right_c1 = c0b + panel_w
    assert left_c0 >= 0 and right_c1 < grid.shape[1]

    frames = [
        grid[r0a : r1a + 1, left_c0 : c0a],
        grid[r0a : r1a + 1, c0a + 1 : c0b],
        grid[r0a : r1a + 1, c0b + 1 : right_c1 + 1],
    ]
    assert all(frame.shape == (panel_h, panel_w) for frame in frames)
    return [frame.copy() for frame in frames], (r0a, left_c0, r1a, right_c1)


def _tracks_by_color(frames: list[np.ndarray]) -> dict[int, list[Coord | None]]:
    colors = set()
    for frame in frames:
        colors.update(map(int, np.unique(frame)))
    colors -= {Color.BLACK, Color.GREY}

    tracks: dict[int, list[Coord | None]] = {}
    for color in sorted(colors):
        track: list[Coord | None] = []
        for frame in frames:
            coords = np.argwhere(frame == color)
            assert len(coords) <= 1, "each thumbnail color must occupy at most one pixel per frame"
            if len(coords) == 0:
                track.append(None)
            else:
                r, c = map(int, coords[0])
                track.append((r, c))
        tracks[color] = track
    return tracks


def _split_static_dynamic(
    frames: list[np.ndarray],
) -> tuple[dict[int, Coord], dict[int, tuple[Coord | None, Coord, Coord]]]:
    tracks = _tracks_by_color(frames)
    static: dict[int, Coord] = {}
    dynamic: dict[int, tuple[Coord | None, Coord, Coord]] = {}

    for color, track in tracks.items():
        p1, p2, p3 = track
        if p1 is not None and p1 == p2 == p3:
            static[color] = p3
            continue

        assert p2 is not None and p3 is not None, "dynamic colors must appear in frames 2 and 3"
        dr = p3[0] - p2[0]
        dc = p3[1] - p2[1]
        assert (dr == 0) ^ (dc == 0), "dynamic motion must be axis-aligned"
        assert dr != 0 or dc != 0, "dynamic motion cannot be stationary"
        if p1 is not None:
            assert (p2[0] - p1[0], p2[1] - p1[1]) == (dr, dc)
        dynamic[color] = (p1, p2, p3)

    assert static and dynamic
    return static, dynamic


def _main_blocks(grid: np.ndarray, thumb_bbox: tuple[int, int, int, int]) -> tuple[int, dict[int, list[Coord]]]:
    tr0, tc0, tr1, tc1 = thumb_bbox
    blocks: list[tuple[int, int, int, int]] = []
    for comp in find_components(grid, bg=Color.BLACK, connectivity=4, monochromatic=True):
        color = int(comp["color"])
        if color in (Color.BLACK, Color.GREY):
            continue
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if tr0 <= r0 and r1 <= tr1 and tc0 <= c0 and c1 <= tc1:
            continue
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 2 or h != w or int(comp["area"]) != h * w:
            continue
        if not np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color):
            continue
        blocks.append((color, r0, c0, h))

    assert blocks, "expected large square blocks outside the thumbnail"
    sizes = [size for _color, _r0, _c0, size in blocks]
    scale, count = Counter(sizes).most_common(1)[0]
    assert count == len(blocks), "all large blocks must share one scale"

    by_color: dict[int, list[Coord]] = {}
    for color, r0, c0, size in blocks:
        assert size == scale
        by_color.setdefault(color, []).append((r0, c0))
    return int(scale), by_color


def _infer_mapping(static: dict[int, Coord], scale: int, block_map: dict[int, list[Coord]]) -> tuple[int, int]:
    votes: Counter[tuple[int, int]] = Counter()
    for color, (r, c) in static.items():
        assert color in block_map, "every static thumbnail color must already exist as a large block"
        for br, bc in block_map[color]:
            votes[(int(br - r * scale), int(bc - c * scale))] += 1

    assert votes, "mapping votes cannot be empty"
    best_votes = max(votes.values())
    best = [offset for offset, n in votes.items() if n == best_votes]
    assert len(best) == 1 and best_votes == len(static), "scale/offset mapping must be unique"
    dr, dc = best[0]

    for color, (r, c) in static.items():
        expected = (int(dr + r * scale), int(dc + c * scale))
        assert expected in block_map[color]
    return int(dr), int(dc)


def _predict_dynamic(
    frame_shape: tuple[int, int],
    dynamic: dict[int, tuple[Coord | None, Coord, Coord]],
) -> dict[int, Coord]:
    out: dict[int, Coord] = {}
    for color, (p1, p2, p3) in dynamic.items():
        dr = p3[0] - p2[0]
        dc = p3[1] - p2[1]
        if p1 is not None:
            assert (p2[0] - p1[0], p2[1] - p1[1]) == (dr, dc)
        p4 = (int(p3[0] + dr), int(p3[1] + dc))
        assert in_bounds(frame_shape, *p4), "predicted frame-4 pixel must remain inside the thumbnail panel"
        out[color] = p4
    assert len(set(out.values())) == len(out), "predicted dynamic pixels must not collide"
    return out


def _paint_block(grid: np.ndarray, top_left: Coord, color: int, scale: int) -> None:
    r0, c0 = map(int, top_left)
    block = np.full((scale, scale), int(color), dtype=int)
    paste(grid, block, r0, c0)


def _analyze(grid: np.ndarray) -> dict[str, object]:
    frames, thumb_bbox = _thumbnail_layout(grid)
    static, dynamic = _split_static_dynamic(frames)
    scale, block_map = _main_blocks(grid, thumb_bbox)
    offset = _infer_mapping(static, scale, block_map)
    predicted = _predict_dynamic(frames[0].shape, dynamic)
    return {
        "frames": frames,
        "thumb_bbox": thumb_bbox,
        "static": static,
        "dynamic": dynamic,
        "scale": scale,
        "offset": offset,
        "predicted": predicted,
        "block_map": block_map,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extrapolate the next thumbnail frame and paint its moving pixels as full-size blocks in the large grid.

    Input structure:
    - Black background surrounds a three-frame thumbnail strip in the top-left, with the frames separated by two full-height grey vertical bars.
    - Thumbnail colors other than grey are single pixels; some colors stay fixed across all three frames while others move by a constant row or column step.
    - The lower-right area contains uniform square color blocks whose size and offset match the static pixels of thumbnail frame 3.

    Rule:
    - Read the three thumbnail panels from left to right as frames 1, 2, and 3.
    - Treat colors that occupy the same thumbnail cell in all three frames as static, and treat every other thumbnail color as dynamic.
    - Infer the large-grid scale and offset from the existing large blocks that match frame-3 static colors.
    - For each dynamic color, use its frame-2 to frame-3 displacement (and confirm frame-1 to frame-2 when available) to predict its frame-4 thumbnail cell.
    - Map each predicted frame-4 cell into the large grid and paint one full square block there, leaving the existing static blocks and the thumbnail strip unchanged.

    Key insight:
    - The thumbnail is a time series rather than a layout legend: motion is inferred from the small frames, while size and placement are inferred independently from the large static blocks.
    """
    g = np.asarray(grid, dtype=int)
    analysis = _analyze(g)
    scale = int(analysis["scale"])
    dr, dc = analysis["offset"]
    predicted: dict[int, Coord] = analysis["predicted"]  # type: ignore[assignment]

    out = g.copy()
    for color, (r, c) in predicted.items():
        top_left = (int(dr + r * scale), int(dc + c * scale))
        rr, cc = top_left
        assert rr >= 0 and cc >= 0 and rr + scale <= out.shape[0] and cc + scale <= out.shape[1]
        target = out[rr : rr + scale, cc : cc + scale]
        assert np.all(target == Color.BLACK), "predicted large block must land on an empty slot"
        _paint_block(out, top_left, color, scale)
    return out


def _thumb_canvas(panel_h: int, panel_w: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    thumb = np.full((panel_h, panel_w * 3 + 2), Color.BLACK, dtype=int)
    sep_cols = [panel_w, panel_w * 2 + 1]
    thumb[:, sep_cols[0]] = Color.GREY
    thumb[:, sep_cols[1]] = Color.GREY
    starts = [(0, 0), (0, panel_w + 1), (0, panel_w * 2 + 2)]
    return thumb, starts


def _all_trajectory_options(panel_h: int, panel_w: int, late_only: bool) -> list[tuple[list[Coord | None], Coord]]:
    options: list[tuple[list[Coord | None], Coord]] = []
    for axis in (0, 1):
        for step in (-2, -1, 1, 2):
            dr = step if axis == 0 else 0
            dc = step if axis == 1 else 0
            for r3 in range(panel_h):
                for c3 in range(panel_w):
                    p3 = (r3, c3)
                    p2 = (r3 - dr, c3 - dc)
                    p4 = (r3 + dr, c3 + dc)
                    if not in_bounds((panel_h, panel_w), *p2) or not in_bounds((panel_h, panel_w), *p4):
                        continue
                    if late_only:
                        options.append(([None, p2, p3], p4))
                        continue
                    p1 = (p2[0] - dr, p2[1] - dc)
                    if in_bounds((panel_h, panel_w), *p1):
                        options.append(([p1, p2, p3], p4))
    return options


def _sample_trajectories(panel_h: int, panel_w: int, dynamic_count: int) -> list[tuple[list[Coord | None], Coord]]:
    kinds = []
    if dynamic_count == 1:
        kinds = [random.choice([False, True])]
    else:
        kinds = [False, True]
        random.shuffle(kinds)

    used = [set() for _ in range(4)]
    chosen: list[tuple[list[Coord | None], Coord]] = []
    for late_only in kinds:
        options = _all_trajectory_options(panel_h, panel_w, late_only=late_only)
        random.shuffle(options)
        for track, p4 in options:
            fits = True
            for t, pos in enumerate(track):
                if pos is not None and pos in used[t]:
                    fits = False
                    break
            if not fits or p4 in used[3]:
                continue
            cells = {pos for pos in track if pos is not None}
            if p4 in cells:
                continue
            for pos in cells:
                if pos in used[3]:
                    fits = False
                    break
            if not fits:
                continue

            chosen.append((track, p4))
            for t, pos in enumerate(track):
                if pos is not None:
                    used[t].add(pos)
            used[3].add(p4)
            break
        else:
            raise RuntimeError("failed to place a non-overlapping dynamic trajectory")

    return chosen


def _sample_static_positions(
    panel_h: int,
    panel_w: int,
    trajectories: list[tuple[list[Coord | None], Coord]],
    static_count: int,
) -> list[Coord]:
    forbidden: set[Coord] = set()
    for track, p4 in trajectories:
        forbidden.update(pos for pos in track if pos is not None)
        forbidden.add(p4)

    candidates = [(r, c) for r in range(panel_h) for c in range(panel_w) if (r, c) not in forbidden]
    random.shuffle(candidates)
    for _ in range(200):
        choice = candidates[:static_count]
        random.shuffle(candidates)
        rows = {r for r, _c in choice}
        cols = {c for _r, c in choice}
        if len(choice) == static_count and len(rows) >= 2 and len(cols) >= 2:
            return list(choice)
    raise RuntimeError("failed to place static positions with row/column diversity")


def _render_frames(
    panel_h: int,
    panel_w: int,
    static_map: dict[int, Coord],
    dynamic_map: dict[int, tuple[list[Coord | None], Coord]],
) -> list[np.ndarray]:
    frames = [np.full((panel_h, panel_w), Color.BLACK, dtype=int) for _ in range(3)]
    for color, pos in static_map.items():
        for frame in frames:
            frame[pos] = color
    for color, (track, _p4) in dynamic_map.items():
        for t, pos in enumerate(track):
            if pos is not None:
                frames[t][pos] = color
    return frames


def _expected_output_from_spec(inp: np.ndarray, spec: dict[str, object]) -> np.ndarray:
    out = inp.copy()
    base_r, base_c = spec["base"]  # type: ignore[assignment]
    dynamic_map: dict[int, tuple[list[Coord | None], Coord]] = spec["dynamic_map"]  # type: ignore[assignment]
    for color, (_track, p4) in dynamic_map.items():
        top_left = (int(base_r + p4[0] * SCALE), int(base_c + p4[1] * SCALE))
        _paint_block(out, top_left, color, SCALE)
    return out


def _build_case() -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    panel_h = random.choice([4, 5])
    panel_w = random.choice([3, 4])
    thumb_w = panel_w * 3 + 2
    dynamic_count = random.choice([1, 2])

    trajectories = _sample_trajectories(panel_h, panel_w, dynamic_count)
    forbidden = set()
    for track, p4 in trajectories:
        forbidden.update(pos for pos in track if pos is not None)
        forbidden.add(p4)
    max_static = min(5, panel_h * panel_w - len(forbidden), len(PALETTE) - dynamic_count)
    assert max_static >= 3
    static_count = random.randint(3, max_static)
    static_positions = _sample_static_positions(panel_h, panel_w, trajectories, static_count)

    colors = random.sample(PALETTE, static_count + dynamic_count)
    static_colors = colors[:static_count]
    dynamic_colors = colors[static_count:]

    static_map = {color: pos for color, pos in zip(static_colors, static_positions)}
    dynamic_map = {color: traj for color, traj in zip(dynamic_colors, trajectories)}
    frames = _render_frames(panel_h, panel_w, static_map, dynamic_map)

    thumb, starts = _thumb_canvas(panel_h, panel_w)
    for frame, (_r0, c0) in zip(frames, starts):
        paste(thumb, frame, 0, c0, transparent=Color.BLACK)

    min_base_r = panel_h + 2
    max_base_r = 30 - panel_h * SCALE
    min_base_c = thumb_w + 2
    max_base_c = 30 - panel_w * SCALE
    assert min_base_r <= max_base_r and min_base_c <= max_base_c
    base_r = random.randint(min_base_r, max_base_r)
    base_c = random.randint(min_base_c, max_base_c)

    grid_h = random.randint(base_r + panel_h * SCALE, min(30, base_r + panel_h * SCALE + 2))
    grid_w = random.randint(base_c + panel_w * SCALE, min(30, base_c + panel_w * SCALE + 2))
    inp = np.full((grid_h, grid_w), Color.BLACK, dtype=int)
    paste(inp, thumb, 0, 0, transparent=Color.BLACK)

    for color, pos in static_map.items():
        block = scale_sprite(np.array([[color]], dtype=int), SCALE)
        paste(inp, block, base_r + pos[0] * SCALE, base_c + pos[1] * SCALE)

    spec = {
        "panel_h": panel_h,
        "panel_w": panel_w,
        "static_map": static_map,
        "dynamic_map": dynamic_map,
        "base": (base_r, base_c),
    }
    out = _expected_output_from_spec(inp, spec)
    return inp, out, spec


def _validate_case(inp: np.ndarray, out: np.ndarray, spec: dict[str, object]) -> bool:
    analysis = _analyze(inp)
    if int(analysis["scale"]) != SCALE:
        return False
    if tuple(analysis["offset"]) != tuple(spec["base"]):  # type: ignore[arg-type]
        return False

    static_map: dict[int, Coord] = spec["static_map"]  # type: ignore[assignment]
    if analysis["static"] != static_map:
        return False

    dynamic_map: dict[int, tuple[list[Coord | None], Coord]] = spec["dynamic_map"]  # type: ignore[assignment]
    predicted = analysis["predicted"]
    expected_predicted = {color: p4 for color, (_track, p4) in dynamic_map.items()}
    if predicted != expected_predicted:
        return False

    if np.array_equal(inp, out):
        return False

    frame3_projection = inp.copy()
    base_r, base_c = spec["base"]  # type: ignore[assignment]
    for color, (track, _p4) in dynamic_map.items():
        p3 = track[2]
        _paint_block(frame3_projection, (base_r + p3[0] * SCALE, base_c + p3[1] * SCALE), color, SCALE)
    if np.array_equal(frame3_projection, out):
        return False

    return np.array_equal(main(inp), out)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        inp, out, spec = _build_case()
        if _validate_case(inp, out, spec):
            return inp
    raise RuntimeError("failed to generate a valid puzzle-6 input")
