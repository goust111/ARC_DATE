from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
MAX_GENERATION_ATTEMPTS = 400
CANVAS_SIZE = 30
_GEN_COUNTER = 0


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _apply_op(mask: np.ndarray, op: int) -> np.ndarray:
    if op == 0:
        return mask.copy()
    if op == 1:
        return np.fliplr(mask)
    if op == 2:
        return np.flipud(mask)
    if op == 3:
        return np.rot90(mask, 2)
    raise AssertionError(f"unknown transform op {op}")


def _same_color_components(g: np.ndarray) -> list[dict]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BG or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append({"color": color, "coords": np.asarray(coords, dtype=int)})
    return comps


def _ring_bbox(coords: np.ndarray) -> tuple[int, int, int, int] | None:
    rows = coords[:, 0]
    cols = coords[:, 1]
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return None
    if len(coords) != 2 * (height + width) - 4:
        return None

    expected: set[tuple[int, int]] = set()
    for rr in range(r0, r1 + 1):
        expected.add((rr, c0))
        expected.add((rr, c1))
    for cc in range(c0, c1 + 1):
        expected.add((r0, cc))
        expected.add((r1, cc))

    actual = {tuple(map(int, xy)) for xy in coords}
    if actual != expected:
        return None
    return r0, c0, r1, c1


def _parse_frame(g: np.ndarray, comp: dict) -> dict | None:
    bbox = _ring_bbox(comp["coords"])
    if bbox is None:
        return None

    r0, c0, r1, c1 = bbox
    h, w = g.shape
    if not (0 < r0 and 0 < c0 and r1 + 1 < h and c1 + 1 < w):
        return None

    frame_color = int(comp["color"])
    inner = g[r0 + 1 : r1, c0 + 1 : c1]
    inner_colors = {int(x) for x in np.unique(inner)}
    pattern_colors = inner_colors - {BG, frame_color}
    if len(pattern_colors) != 1:
        return None
    pattern_color = next(iter(pattern_colors))
    if inner_colors - {BG, pattern_color}:
        return None

    pattern = inner == pattern_color
    if not pattern.any():
        return None

    corners = [
        (r0 - 1, c0 - 1),
        (r0 - 1, c1 + 1),
        (r1 + 1, c0 - 1),
        (r1 + 1, c1 + 1),
    ]
    hits = [i for i, (rr, cc) in enumerate(corners) if int(g[rr, cc]) == pattern_color]
    if len(hits) != 1:
        return None
    for i, (rr, cc) in enumerate(corners):
        if i != hits[0] and int(g[rr, cc]) != BG:
            return None

    return {
        "bbox": bbox,
        "frame_color": frame_color,
        "pattern_color": pattern_color,
        "op": int(hits[0]),
    }


def _find_frames(g: np.ndarray) -> list[dict]:
    frames = []
    for comp in _same_color_components(g):
        frame = _parse_frame(g, comp)
        if frame is not None:
            frames.append(frame)
    frames.sort(key=lambda item: item["bbox"])
    assert frames, "expected at least one marked frame"
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transform each framed inner pattern according to the marked outside corner.

    Input structure:
    - The background is black.
    - One or more one-pixel-thick rectangular frames appear in different colors.
    - Each frame encloses a sparse single-color pattern, and exactly one of the
      four outside diagonal corner cells contains a one-pixel marker in that
      same pattern color.

    Rule:
    - Detect every rectangular frame and read which outside corner carries the
      marker pixel.
    - Rewrite only the pattern inside that frame: top-left marker keeps it
      unchanged, top-right mirrors it left-right, bottom-left mirrors it
      up-down, and bottom-right rotates it by 180 degrees.
    - Leave the frame pixels, the marker pixels, and the rest of the grid
      unchanged.

    Key insight:
    - The single outside-corner marker encodes the geometric transform by its
      position rather than by its color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for frame in _find_frames(g):
        r0, c0, r1, c1 = frame["bbox"]
        color = int(frame["pattern_color"])
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        pattern = inner == color
        transformed = _apply_op(pattern, int(frame["op"]))
        new_inner = np.full(inner.shape, BG, dtype=int)
        new_inner[transformed] = color
        out[r0 + 1 : r1, c0 + 1 : c1] = new_inner

    return out


def _distinct_variants(mask: np.ndarray) -> bool:
    variants = {
        mask.tobytes(),
        np.fliplr(mask).tobytes(),
        np.flipud(mask).tobytes(),
        np.rot90(mask, 2).tobytes(),
    }
    return len(variants) == 4


def _random_pattern(height: int, width: int) -> np.ndarray:
    area = height * width
    min_cells = 3 if area >= 6 else 2
    max_cells = min(area - 1, max(min_cells, (2 * area) // 3))
    for _ in range(200):
        target = random.randint(min_cells, max_cells)
        cells = {(random.randrange(height), random.randrange(width))}
        while len(cells) < target:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        mask = np.zeros((height, width), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if len(cells) == target and _distinct_variants(mask):
            return mask
    raise RuntimeError("failed to generate an asymmetric pattern")


def _choose_ops(count: int) -> list[int]:
    global _GEN_COUNTER
    omit = _GEN_COUNTER % 4
    _GEN_COUNTER += 1
    if count == 4:
        ops = [0, 1, 2, 3]
    else:
        ops = [op for op in range(4) if op != omit]
    random.shuffle(ops)
    return ops


def _place_objects(objs: list[dict]) -> list[tuple[int, int]] | None:
    occupied = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(objs)
    order = sorted(
        range(len(objs)),
        key=lambda i: (objs[i]["frame_h"] + 4) * (objs[i]["frame_w"] + 4),
        reverse=True,
    )

    for idx in order:
        frame_h = int(objs[idx]["frame_h"])
        frame_w = int(objs[idx]["frame_w"])
        options: list[tuple[int, int]] = []
        for r0 in range(2, CANVAS_SIZE - frame_h - 1):
            for c0 in range(2, CANVAS_SIZE - frame_w - 1):
                reserve = occupied[r0 - 2 : r0 + frame_h + 2, c0 - 2 : c0 + frame_w + 2]
                if not reserve.any():
                    options.append((r0, c0))
        if not options:
            return None
        r0, c0 = random.choice(options)
        occupied[r0 - 2 : r0 + frame_h + 2, c0 - 2 : c0 + frame_w + 2] = True
        placements[idx] = (r0, c0)

    return [xy for xy in placements if xy is not None]


def _paint_ring(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _paint_object(inp: np.ndarray, out: np.ndarray, obj: dict, r0: int, c0: int) -> tuple[int, int, int, int]:
    frame_h = int(obj["frame_h"])
    frame_w = int(obj["frame_w"])
    r1 = r0 + frame_h - 1
    c1 = c0 + frame_w - 1
    frame_color = int(obj["frame_color"])
    pattern_color = int(obj["pattern_color"])
    op = int(obj["op"])
    pattern = obj["pattern"]

    _paint_ring(inp, r0, c0, r1, c1, frame_color)
    _paint_ring(out, r0, c0, r1, c1, frame_color)

    inner_slice = np.s_[r0 + 1 : r1, c0 + 1 : c1]
    inp[inner_slice][pattern] = pattern_color
    out[inner_slice][_apply_op(pattern, op)] = pattern_color

    corners = [
        (r0 - 1, c0 - 1),
        (r0 - 1, c1 + 1),
        (r1 + 1, c0 - 1),
        (r1 + 1, c1 + 1),
    ]
    mr, mc = corners[op]
    inp[mr, mc] = pattern_color
    out[mr, mc] = pattern_color
    return r0 - 1, c0 - 1, r1 + 1, c1 + 1


def _build_scene() -> tuple[np.ndarray, np.ndarray] | None:
    count = random.choice((3, 4))
    ops = _choose_ops(count)
    colors = random.sample(range(1, 10), 2 * count)
    objs: list[dict] = []
    for i, op in enumerate(ops):
        inner_h = random.randint(3, 6)
        inner_w = random.randint(3, 6)
        objs.append(
            {
                "frame_h": inner_h + 2,
                "frame_w": inner_w + 2,
                "frame_color": colors[2 * i],
                "pattern_color": colors[2 * i + 1],
                "pattern": _random_pattern(inner_h, inner_w),
                "op": op,
            }
        )

    placements = _place_objects(objs)
    if placements is None:
        return None

    inp = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
    out = np.zeros_like(inp)

    used_r0, used_c0 = CANVAS_SIZE - 1, CANVAS_SIZE - 1
    used_r1, used_c1 = 0, 0
    for obj, (r0, c0) in zip(objs, placements):
        er0, ec0, er1, ec1 = _paint_object(inp, out, obj, r0, c0)
        used_r0 = min(used_r0, er0)
        used_c0 = min(used_c0, ec0)
        used_r1 = max(used_r1, er1)
        used_c1 = max(used_c1, ec1)

    top = random.randint(0, used_r0)
    left = random.randint(0, used_c0)
    bottom = random.randint(used_r1, CANVAS_SIZE - 1)
    right = random.randint(used_c1, CANVAS_SIZE - 1)
    return inp[top : bottom + 1, left : right + 1], out[top : bottom + 1, left : right + 1]


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_scene()
        if built is None:
            continue
        inp, expected = built
        if np.array_equal(main(inp), expected):
            return inp
    raise RuntimeError("failed to generate a valid framed-transform scene")
