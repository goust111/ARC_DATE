from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
FRAME = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 400
MIN_ZONE_H = 6
MIN_ZONE_W = 6


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _most_common_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[cnts.argmax()])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in points]
    cols = [c for _r, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _is_rect_frame(points: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(points)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    pts = set(points)
    expected = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4
    if len(pts) != expected:
        return False
    for r in range(r0, r1 + 1):
        if (r, c0) not in pts or (r, c1) not in pts:
            return False
    for c in range(c0, c1 + 1):
        if (r0, c) not in pts or (r1, c) not in pts:
            return False
    return True


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    raw_candidates: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        for pts in _components(grid == color):
            if not _is_rect_frame(pts):
                continue
            r0, c0, r1, c1 = _bbox(pts)
            stamp = grid[r0 + 1 : r1, c0 + 1 : c1].copy()
            raw_candidates.append(
                {
                    "frame_color": int(color),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "stamp": stamp,
                }
            )

    inside_any_frame = np.zeros((h, w), dtype=bool)
    for frame in raw_candidates:
        r0, c0, r1, c1 = frame["bbox"]
        inside_any_frame[r0 : r1 + 1, c0 : c1 + 1] = True

    frame_candidates: list[dict] = []
    for frame in raw_candidates:
        r0, c0, r1, c1 = frame["bbox"]
        fc = frame["frame_color"]
        marker_cells: set[tuple[int, int]] = set()
        for c in range(c0, c1 + 1):
            if r0 > 0:
                marker_cells.add((r0 - 1, c))
            if r1 + 1 < h:
                marker_cells.add((r1 + 1, c))
        for r in range(r0, r1 + 1):
            if c0 > 0:
                marker_cells.add((r, c0 - 1))
            if c1 + 1 < w:
                marker_cells.add((r, c1 + 1))
        candidates = [
            (mr, mc)
            for mr, mc in marker_cells
            if not inside_any_frame[mr, mc] and int(grid[mr, mc]) not in (bg, fc)
        ]
        if len(candidates) != 1:
            continue
        mr, mc = candidates[0]
        frame["marker_color"] = int(grid[mr, mc])
        frame["marker_pos"] = (int(mr), int(mc))
        frame_candidates.append(frame)

    return frame_candidates


def _path_component(grid: np.ndarray, frames: list[dict], bg: int) -> set[tuple[int, int]]:
    h, w = grid.shape
    mask = grid != bg
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        mask[r0 : r1 + 1, c0 : c1 + 1] = False

    best: set[tuple[int, int]] | None = None
    for pts in _components(mask):
        if len(pts) <= 1:
            continue
        pset = set(pts)
        degrees = []
        for r, c in pts:
            deg = sum((nr, nc) in pset for nr, nc in _neighbors4(r, c))
            degrees.append(deg)
        if degrees.count(1) == 2 and all(deg <= 2 for deg in degrees):
            if best is None or len(pset) > len(best):
                best = pset
    assert best is not None, "expected one simple path component"
    return best


def _follow_path(path_cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    endpoints = sorted(
        [cell for cell in path_cells if sum(n in path_cells for n in _neighbors4(*cell)) == 1]
    )
    assert len(endpoints) == 2, "path must have exactly two endpoints"
    start = endpoints[0]
    ordered = [start]
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        nxt = [n for n in _neighbors4(*cur) if n in path_cells and n != prev]
        if not nxt:
            break
        assert len(nxt) == 1, "simple path must have a unique next step"
        prev, cur = cur, nxt[0]
        ordered.append(cur)
    assert len(ordered) == len(path_cells)
    return ordered


def _stamp(out: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    ph, pw = patch.shape
    r0 = max(0, int(top))
    c0 = max(0, int(left))
    r1 = min(out.shape[0], int(top) + ph)
    c1 = min(out.shape[1], int(left) + pw)
    if r0 >= r1 or c0 >= c1:
        return
    out[r0:r1, c0:c1] = patch[r0 - top : r1 - top, c0 - left : c1 - left]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Re-stamp the framed motifs onto a blank grid using the guide path's first color occurrences as anchors.

    Input structure:
    - The background is black.
    - Several grey one-pixel rectangular frames each contain a small multicolor stamp patch.
    - Every frame has one adjacent outside marker pixel of a unique color, and elsewhere there is a single 4-connected guide path made only from those marker colors.

    Rule:
    - Detect every rectangular frame, extract its interior stamp, and match it to the color of its adjacent marker pixel.
    - Traverse the guide path from one endpoint to the other; the first position where each marker color appears is that stamp's top-left anchor, and the order of those first appearances is the stamping order.
    - Start from an empty background grid, stamp each extracted patch at its anchor, and let later stamps overwrite earlier ones.
    - Erase the guide path, the frames, their markers, and the original framed stamps from the output.

    Key insight:
    - The path encodes two things at once: each color's first-seen location gives a coordinate, and the first-seen order gives the layer order needed to reconstruct overlaps.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    frames = _find_frames(g, bg)
    assert frames, "expected at least one frame"

    stamps_by_color = {frame["marker_color"]: frame["stamp"] for frame in frames}
    path_cells = _path_component(g, frames, bg)
    ordered_path = _follow_path(path_cells)

    first_pos: dict[int, tuple[int, int]] = {}
    color_order: list[int] = []
    for r, c in ordered_path:
        color = int(g[r, c])
        if color not in first_pos:
            first_pos[color] = (int(r), int(c))
            color_order.append(color)

    for color in stamps_by_color:
        assert color in first_pos, "every marker color must appear on the path"

    out = np.full_like(g, bg)
    for color in color_order:
        if color not in stamps_by_color:
            continue
        anchor_r, anchor_c = first_pos[color]
        _stamp(out, stamps_by_color[color], anchor_r, anchor_c)
    return out


def _make_stamp(rng: random.Random, palette: list[int], max_dim: int) -> np.ndarray:
    h = rng.randint(2, max_dim)
    w = rng.randint(2, max_dim)
    colors = rng.sample(palette, rng.randint(2, min(3, len(palette))))
    base = colors[0]
    patch = np.full((h, w), base, dtype=int)
    mode = rng.choice(["stripes", "checker", "scatter"])

    if mode == "stripes":
        if h >= w:
            cut = rng.randint(1, h - 1)
            patch[cut:, :] = colors[1]
        else:
            cut = rng.randint(1, w - 1)
            patch[:, cut:] = colors[1]
        if len(colors) == 3:
            patch[rng.randrange(h), rng.randrange(w)] = colors[2]
    elif mode == "checker":
        for r in range(h):
            for c in range(w):
                patch[r, c] = colors[(r + c) % 2]
        if len(colors) == 3:
            patch[h // 2, w // 2] = colors[2]
    else:
        coords = [(r, c) for r in range(h) for c in range(w)]
        rng.shuffle(coords)
        used = {base}
        for color in colors[1:]:
            count = rng.randint(1, max(1, (h * w) // 2))
            for r, c in coords[:count]:
                patch[r, c] = color
            used.add(color)
        if len(used) < 2:
            patch[0, 0] = colors[1]

    if len(set(int(v) for v in patch.flat)) < 2:
        patch[0, 0] = colors[1]
    return patch


def _induced_path(
    rng: random.Random,
    row_lo: int,
    row_hi: int,
    col_lo: int,
    col_hi: int,
    length: int,
) -> list[tuple[int, int]] | None:
    start = (int(row_lo), int(col_lo))

    def in_zone(cell: tuple[int, int]) -> bool:
        r, c = cell
        return row_lo <= r <= row_hi and col_lo <= c <= col_hi

    def next_options(path: list[tuple[int, int]], visited: set[tuple[int, int]]) -> list[tuple[int, int]]:
        cur = path[-1]
        out: list[tuple[int, int]] = []
        for nxt in _neighbors4(*cur):
            if not in_zone(nxt) or nxt in visited:
                continue
            if sum(nei in visited for nei in _neighbors4(*nxt)) != 1:
                continue
            out.append(nxt)
        rng.shuffle(out)
        out.sort(
            key=lambda cell: sum(
                1
                for nei in _neighbors4(*cell)
                if in_zone(nei) and nei not in visited and nei != path[-1]
            ),
            reverse=True,
        )
        return out

    def dfs(path: list[tuple[int, int]], visited: set[tuple[int, int]]) -> bool:
        if len(path) == length:
            return True
        for nxt in next_options(path, visited):
            path.append(nxt)
            visited.add(nxt)
            if dfs(path, visited):
                return True
            visited.remove(nxt)
            path.pop()
        return False

    for _ in range(80):
        path = [start]
        visited = {start}
        if dfs(path, visited):
            return path
    return None


def _assign_path_colors(
    rng: random.Random,
    intro_order: list[int],
    intro_positions: list[int],
    length: int,
) -> list[int] | None:
    pos_to_color = {int(pos): int(color) for pos, color in zip(intro_positions, intro_order, strict=False)}
    sequence: list[int] = []
    introduced: list[int] = []
    prev: int | None = None

    for idx in range(length):
        if idx in pos_to_color:
            color = pos_to_color[idx]
            if prev == color:
                return None
            sequence.append(color)
            introduced.append(color)
            prev = color
            continue

        candidates = [color for color in introduced if color != prev]
        if idx + 1 in pos_to_color:
            next_color = pos_to_color[idx + 1]
            lookahead = [color for color in candidates if color != next_color]
            if lookahead:
                candidates = lookahead
        if not candidates:
            return None
        color = rng.choice(candidates)
        sequence.append(color)
        prev = color

    return sequence


def _layout_frames(
    rng: random.Random,
    stamps: list[np.ndarray],
    side: str,
) -> tuple[tuple[int, int], list[tuple[int, int]], tuple[int, int, int, int]] | None:
    outer_shapes = [(stamp.shape[0] + 2, stamp.shape[1] + 2) for stamp in stamps]
    max_sh = max(stamp.shape[0] for stamp in stamps)
    max_sw = max(stamp.shape[1] for stamp in stamps)
    gap = rng.randint(2, 3)
    n = len(stamps)

    if side in {"top", "bottom"}:
        strip = max(h for h, _w in outer_shapes) + 2
        span = sum(w for _h, w in outer_shapes) + gap * (n - 1)
        min_h = strip + max_sh + MIN_ZONE_H + 1
        min_w = max(span + 2, max_sw + MIN_ZONE_W + 1)
        if min_h > 30 or min_w > 30:
            return None
        h = rng.randint(min_h, 30)
        w = rng.randint(min_w, 30)
        start_c = rng.randint(1, w - span - 1)
        row = 1 if side == "top" else h - strip + 1
        positions: list[tuple[int, int]] = []
        cur_c = start_c
        for _idx, (_oh, ow) in enumerate(outer_shapes):
            positions.append((row, cur_c))
            cur_c += ow + gap
        if side == "top":
            zone = (strip + 1, h - max_sh - 1, 1, w - max_sw - 1)
        else:
            zone = (1, h - strip - max_sh - 1, 1, w - max_sw - 1)
        return (h, w), positions, zone

    strip = max(w for _h, w in outer_shapes) + 2
    span = sum(h for h, _w in outer_shapes) + gap * (n - 1)
    min_h = max(span + 2, max_sh + MIN_ZONE_H + 1)
    min_w = strip + max_sw + MIN_ZONE_W + 1
    if min_h > 30 or min_w > 30:
        return None
    h = rng.randint(min_h, 30)
    w = rng.randint(min_w, 30)
    start_r = rng.randint(1, h - span - 1)
    col = 1 if side == "left" else w - strip + 1
    positions = []
    cur_r = start_r
    for _idx, (oh, _ow) in enumerate(outer_shapes):
        positions.append((cur_r, col))
        cur_r += oh + gap
    if side == "left":
        zone = (1, h - max_sh - 1, strip + 1, w - max_sw - 1)
    else:
        zone = (1, h - max_sh - 1, 1, w - strip - max_sw - 1)
    return (h, w), positions, zone


def _paint_frame(
    grid: np.ndarray,
    top: int,
    left: int,
    stamp: np.ndarray,
    marker_color: int,
    rng: random.Random,
    allow_sides: tuple[str, ...],
) -> None:
    sh, sw = stamp.shape
    oh, ow = sh + 2, sw + 2
    bottom = top + oh - 1
    right = left + ow - 1
    grid[top : bottom + 1, left] = FRAME
    grid[top : bottom + 1, right] = FRAME
    grid[top, left : right + 1] = FRAME
    grid[bottom, left : right + 1] = FRAME
    grid[top + 1 : top + 1 + sh, left + 1 : left + 1 + sw] = stamp

    marker_choices: list[tuple[int, int]] = []
    if "top" in allow_sides:
        marker_choices.append((top - 1, rng.randint(left, right)))
    if "bottom" in allow_sides:
        marker_choices.append((bottom + 1, rng.randint(left, right)))
    if "left" in allow_sides:
        marker_choices.append((rng.randint(top, bottom), left - 1))
    if "right" in allow_sides:
        marker_choices.append((rng.randint(top, bottom), right + 1))
    valid = [
        (r, c)
        for r, c in marker_choices
        if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and grid[r, c] == BG
    ]
    mr, mc = rng.choice(valid)
    grid[mr, mc] = int(marker_color)


def _expected_output(shape: tuple[int, int], intro_order: list[int], anchors: dict[int, tuple[int, int]], stamps: dict[int, np.ndarray]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for color in intro_order:
        r, c = anchors[color]
        _stamp(out, stamps[color], r, c)
    return out


def _generate_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    n_stamps = rng.randint(2, 4)
    marker_colors = rng.sample(COLOR_POOL, n_stamps)
    stamp_palette = [color for color in COLOR_POOL if color not in marker_colors]
    max_dim = 3 if n_stamps == 4 else 4
    stamps = [_make_stamp(rng, stamp_palette, max_dim) for _ in range(n_stamps)]
    side = rng.choice(["top", "bottom", "left", "right"])
    layout = _layout_frames(rng, stamps, side)
    if layout is None:
        return None

    shape, positions, zone = layout
    row_lo, row_hi, col_lo, col_hi = zone
    if row_hi - row_lo + 1 < MIN_ZONE_H or col_hi - col_lo + 1 < MIN_ZONE_W:
        return None

    zone_h = row_hi - row_lo + 1
    zone_w = col_hi - col_lo + 1
    min_len = max(6, n_stamps + 2)
    max_len = min(18, (zone_h * zone_w) // 2)
    if max_len < min_len:
        return None

    path_len = rng.randint(min_len, max_len)
    path = _induced_path(rng, row_lo, row_hi, col_lo, col_hi, path_len)
    if path is None:
        return None

    extra_intro = sorted(rng.sample(range(2, path_len), n_stamps - 2))
    intro_positions = [0, 1] + extra_intro
    intro_order = marker_colors[:]
    rng.shuffle(intro_order)
    path_colors = _assign_path_colors(rng, intro_order, intro_positions, path_len)
    if path_colors is None:
        return None

    anchors = {color: path[pos] for color, pos in zip(intro_order, intro_positions, strict=False)}
    stamp_by_color = {color: stamp for color, stamp in zip(marker_colors, stamps, strict=False)}
    expected = _expected_output(shape, intro_order, anchors, stamp_by_color)

    grid = np.full(shape, BG, dtype=int)
    marker_sides = ("top", "bottom") if side in {"top", "bottom"} else ("left", "right")
    for stamp_idx, (top, left) in enumerate(positions):
        _paint_frame(grid, top, left, stamps[stamp_idx], marker_colors[stamp_idx], rng, marker_sides)

    for (r, c), color in zip(path, path_colors, strict=False):
        if grid[r, c] != BG:
            return None
        grid[r, c] = int(color)

    if _most_common_color(grid) != BG:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        case = _generate_case(rng)
        if case is not None:
            grid, _out = case
            return grid
    raise RuntimeError("generate_input: failed to build a valid puzzle")
