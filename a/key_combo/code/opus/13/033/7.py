from __future__ import annotations

from collections import deque
import hashlib
import random

import numpy as np


BG = 0
ORANGE = 7
GREY = 5
FRAME_SIZE = 4
MIN_GAP = 2
FRAME_PALETTE = (1, 2, 3, 4, 8)
MARKER_CHOICES = {
    "up": ((0, 1), (0, 2)),
    "down": ((3, 1), (3, 2)),
    "left": ((1, 0), (2, 0)),
    "right": ((1, 3), (2, 3)),
}
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
GEN_ATTEMPTS = 600
_GEN_CALLS = 0
_SEEN_INPUT_SIGS: set[str] = set()
_SEEN_OUTPUT_SIGS: set[str] = set()


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()


def _paste(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    dst[top : top + h, left : left + w] = patch


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(pts, dtype=int))
    return out


def _marker_side(local: tuple[int, int]) -> str:
    r, c = local
    if r == 0:
        assert c in (1, 2)
        return "up"
    if r == FRAME_SIZE - 1:
        assert c in (1, 2)
        return "down"
    if c == 0:
        assert r in (1, 2)
        return "left"
    if c == FRAME_SIZE - 1:
        assert r in (1, 2)
        return "right"
    raise AssertionError("marker must lie on a frame side midpoint")


def _classify_object(grid: np.ndarray, coords: np.ndarray) -> dict[str, object]:
    box = _bbox(coords)
    patch = _crop_bbox(grid, box)
    assert patch.shape == (FRAME_SIZE, FRAME_SIZE), f"unexpected object shape {patch.shape}"

    if np.all(patch == GREY):
        return {"kind": "solid", "bbox": box, "patch": patch}

    if not np.all(patch[1:3, 1:3] == BG):
        raise AssertionError("non-solid objects must be hollow 4x4 frames")

    border_mask = np.ones((FRAME_SIZE, FRAME_SIZE), dtype=bool)
    border_mask[1:3, 1:3] = False
    border = patch[border_mask]
    marker_coords = np.argwhere(patch == ORANGE)
    assert marker_coords.shape == (1, 2), "frame must have one orange marker"
    mr, mc = map(int, marker_coords[0])
    assert border_mask[mr, mc], "marker must lie on the border"
    assert (mr, mc) not in {(0, 0), (0, 3), (3, 0), (3, 3)}, "marker must not be on a corner"

    other_colors = sorted(
        {
            int(v)
            for v in np.unique(border)
            if int(v) not in {BG, ORANGE}
        }
    )
    assert len(other_colors) == 1, "frame border must have one non-orange color"
    frame_color = other_colors[0]

    expected = np.full((FRAME_SIZE, FRAME_SIZE), frame_color, dtype=int)
    expected[1:3, 1:3] = BG
    expected[mr, mc] = ORANGE
    assert np.array_equal(patch, expected), "object must be a 4x4 hollow frame with one marker"

    return {
        "kind": "frame",
        "bbox": box,
        "patch": patch,
        "color": frame_color,
        "marker_local": (mr, mc),
        "direction": _marker_side((mr, mc)),
    }


def _extract_objects(grid: np.ndarray) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    frames: list[dict[str, object]] = []
    solids: list[dict[str, object]] = []
    for coords in _components(grid != BG):
        obj = _classify_object(grid, coords)
        if obj["kind"] == "frame":
            frames.append(obj)
        else:
            solids.append(obj)
    assert 3 <= len(frames) <= 5, "expected 3-5 hollow frames"
    assert 1 <= len(solids) <= 2, "expected 1-2 solid distractors"
    return frames, solids


def _pack_frames(frames: list[dict[str, object]], horizontal: bool) -> list[dict[str, object]]:
    if horizontal:
        ordered = sorted(frames, key=lambda item: (int(item["bbox"][1]), int(item["bbox"][0])))
    else:
        ordered = sorted(frames, key=lambda item: (int(item["bbox"][0]), int(item["bbox"][1])))

    packed = []
    for idx, frame in enumerate(ordered):
        top = 0 if horizontal else idx * FRAME_SIZE
        left = idx * FRAME_SIZE if horizontal else 0
        packed.append(
            {
                **frame,
                "packed_top": top,
                "packed_left": left,
                "packed_marker": (
                    top + int(frame["marker_local"][0]),
                    left + int(frame["marker_local"][1]),
                ),
            }
        )
    return packed


def _transform(grid: np.ndarray, *, collect_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    in_h, in_w = map(int, g.shape)

    frames, solids = _extract_objects(g)
    frame_rows = [int(frame["bbox"][0]) for frame in frames]
    frame_cols = [int(frame["bbox"][1]) for frame in frames]
    row_span = max(frame_rows) - min(frame_rows)
    col_span = max(frame_cols) - min(frame_cols)
    horizontal = col_span > row_span

    packed = _pack_frames(frames, horizontal)
    strip_h = FRAME_SIZE if horizontal else FRAME_SIZE * len(packed)
    strip_w = FRAME_SIZE * len(packed) if horizontal else FRAME_SIZE
    out_h = min(30, max(strip_h + 10, in_h)) if horizontal else strip_h
    out_w = strip_w if horizontal else min(30, max(strip_w + 10, in_w))
    assert out_h <= 30 and out_w <= 30

    out = np.full((out_h, out_w), BG, dtype=int)
    for frame in packed:
        _paste(out, np.asarray(frame["patch"], dtype=int), int(frame["packed_top"]), int(frame["packed_left"]))
    obstacles = out != BG

    rays = []
    for frame in packed:
        sr, sc = map(int, frame["packed_marker"])
        dr, dc = DIRS[str(frame["direction"])]
        r, c = sr, sc
        drawn = 0
        fate = "boundary"
        while True:
            nr, nc = r + dr, c + dc
            if not _in_bounds(out.shape, nr, nc):
                fate = "boundary"
                break
            if obstacles[nr, nc]:
                fate = "obstacle"
                break
            out[nr, nc] = ORANGE
            drawn += 1
            r, c = nr, nc
        rays.append(
            {
                "start": (sr, sc),
                "direction": str(frame["direction"]),
                "drawn": drawn,
                "fate": fate,
            }
        )

    if collect_meta:
        return out, {
            "horizontal": horizontal,
            "row_span": row_span,
            "col_span": col_span,
            "packed": packed,
            "rays": rays,
            "solids": solids,
            "output_shape": out.shape,
        }
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only the hollow marked frames, pack them into one strip, then fire orange rays from their packed marker positions.

    Input structure:
    - The grid has a black background with separated 4x4 objects.
    - Retained objects are hollow 4x4 square frames whose 2x2 interior is black and whose border contains exactly one orange side-midpoint marker.
    - Distractors are solid 4x4 gray blocks that must be ignored.

    Rule:
    - Discard every solid gray block and keep only the hollow frames.
    - Compare the retained-frame row span and column span; if the column span is larger, sort by column then row and pack left-to-right, otherwise sort by row then column and pack top-to-bottom.
    - Place the packed strip at output position (0,0), keep every frame unchanged, then extend an orange ray from each packed marker in the direction encoded by its side: top=up, bottom=down, left=left, right=right.
    - A ray paints black cells until the output boundary, but it stops immediately before any non-black frame pixel.

    Key insight:
    - The orange marker is only a directional clue in the input scene; after filtering and packing, that same marker becomes the ray origin in a new absolute position, so the packing step is necessary before any ray can be drawn.
    """
    return _transform(grid)


def _boxes_clear(candidate: tuple[int, int], occupied: list[tuple[int, int]]) -> bool:
    r, c = candidate
    for rr, cc in occupied:
        if not (
            r + FRAME_SIZE - 1 + MIN_GAP < rr
            or rr + FRAME_SIZE - 1 + MIN_GAP < r
            or c + FRAME_SIZE - 1 + MIN_GAP < cc
            or cc + FRAME_SIZE - 1 + MIN_GAP < c
        ):
            return False
    return True


def _free_positions(height: int, width: int, occupied: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out = []
    for r in range(height - FRAME_SIZE + 1):
        for c in range(width - FRAME_SIZE + 1):
            if _boxes_clear((r, c), occupied):
                out.append((r, c))
    return out


def _random_partition(total: int, parts: int) -> list[int]:
    slots = [0] * parts
    for _ in range(total):
        slots[random.randrange(parts)] += 1
    return slots


def _spaced_positions(size: int, count: int) -> list[int] | None:
    max_anchor = size - FRAME_SIZE
    min_step = FRAME_SIZE + MIN_GAP
    slack = max_anchor - min_step * (count - 1)
    if slack < 0:
        return None
    gaps = _random_partition(slack, count + 1)
    pos = []
    cursor = gaps[0]
    for idx in range(count):
        pos.append(cursor)
        if idx + 1 < count:
            cursor += min_step + gaps[idx + 1]
    return pos


def _limited_positions(size: int, count: int, max_span: int) -> list[int] | None:
    max_anchor = size - FRAME_SIZE
    if max_anchor < 0:
        return None
    span = random.randint(0, max(0, min(max_anchor, max_span)))
    base = random.randint(0, max_anchor - span)
    return [base + random.randint(0, span) for _ in range(count)]


def _frame_patch(color: int, side: str) -> np.ndarray:
    patch = np.full((FRAME_SIZE, FRAME_SIZE), color, dtype=int)
    patch[1:3, 1:3] = BG
    mr, mc = random.choice(MARKER_CHOICES[side])
    patch[mr, mc] = ORANGE
    return patch


def _pick_sides(axis: str, count: int) -> list[str]:
    sides = [None] * count
    if axis == "horizontal":
        open_idx = random.randrange(count)
        blocked_idx = random.randrange(1, count)
        sides[open_idx] = "down"
        sides[blocked_idx] = "left" if random.random() < 0.5 else "right"
    else:
        open_idx = random.randrange(count)
        blocked_idx = random.randrange(1, count)
        sides[open_idx] = "right"
        sides[blocked_idx] = "up" if random.random() < 0.5 else "down"
    choices = ("up", "down", "left", "right")
    for idx in range(count):
        if sides[idx] is None:
            sides[idx] = random.choice(choices)
    if len(set(sides)) == 1:
        bump = (open_idx + 1) % count
        sides[bump] = "down" if axis == "horizontal" else "right"
    return [str(side) for side in sides]


def _frame_anchors(axis: str, count: int, height: int, width: int) -> list[tuple[int, int]] | None:
    if axis == "horizontal":
        cols = _spaced_positions(width, count)
        if cols is None:
            return None
        col_span = max(cols) - min(cols)
        rows = _limited_positions(height, count, max(0, col_span - 1))
        if rows is None:
            return None
        return list(zip(rows, cols))

    rows = _spaced_positions(height, count)
    if rows is None:
        return None
    row_span = max(rows) - min(rows)
    cols = _limited_positions(width, count, max(0, row_span - 1))
    if cols is None:
        return None
    return list(zip(rows, cols))


def _sample_sizes(axis: str, frame_count: int) -> tuple[int, int]:
    base_major = FRAME_SIZE + (FRAME_SIZE + MIN_GAP) * (frame_count - 1)
    if axis == "horizontal":
        height = random.randint(15, 22)
        width = random.randint(max(18, base_major), 29)
    else:
        height = random.randint(max(18, base_major), 29)
        width = random.randint(15, 22)
    return height, width


def _signature(grid: np.ndarray) -> str:
    return hashlib.md5(grid.tobytes() + str(grid.shape).encode()).hexdigest()


def _semantic_checks(inp: np.ndarray, expected: np.ndarray, axis: str, frame_count: int, solid_count: int) -> bool:
    try:
        frames, solids = _extract_objects(inp)
    except AssertionError:
        return False
    if len(frames) != frame_count or len(solids) != solid_count:
        return False

    rows = [int(frame["bbox"][0]) for frame in frames]
    cols = [int(frame["bbox"][1]) for frame in frames]
    row_span = max(rows) - min(rows)
    col_span = max(cols) - min(cols)
    if axis == "horizontal" and not (col_span > row_span):
        return False
    if axis == "vertical" and not (row_span >= col_span):
        return False

    _, meta = _transform(inp, collect_meta=True)
    if expected.shape != tuple(meta["output_shape"]):
        return False

    rays = meta["rays"]
    has_open = any(ray["fate"] == "boundary" and int(ray["drawn"]) > 0 for ray in rays)
    has_blocked = any(ray["fate"] == "obstacle" for ray in rays)
    if not (has_open and has_blocked):
        return False

    if len({int(frame["color"]) for frame in frames}) != frame_count:
        return False

    return np.array_equal(main(inp), expected)


def _gen_attempt(axis: str, frame_count: int, solid_count: int) -> np.ndarray | None:
    height, width = _sample_sizes(axis, frame_count)
    anchors = _frame_anchors(axis, frame_count, height, width)
    if anchors is None:
        return None

    colors = random.sample(FRAME_PALETTE, frame_count)
    sides = _pick_sides(axis, frame_count)
    grid = np.zeros((height, width), dtype=int)
    occupied: list[tuple[int, int]] = []

    for (r, c), color, side in zip(anchors, colors, sides):
        if not _boxes_clear((r, c), occupied):
            return None
        _paste(grid, _frame_patch(color, side), r, c)
        occupied.append((r, c))

    for _ in range(solid_count):
        free = _free_positions(height, width, occupied)
        if not free:
            return None
        r, c = random.choice(free)
        _paste(grid, np.full((FRAME_SIZE, FRAME_SIZE), GREY, dtype=int), r, c)
        occupied.append((r, c))

    expected = _transform(grid)
    if max(expected.shape) > 30:
        return None
    if not _semantic_checks(grid, expected, axis, frame_count, solid_count):
        return None

    in_sig = _signature(grid)
    out_sig = _signature(expected)
    if in_sig in _SEEN_INPUT_SIGS or out_sig in _SEEN_OUTPUT_SIGS:
        return None

    _SEEN_INPUT_SIGS.add(in_sig)
    _SEEN_OUTPUT_SIGS.add(out_sig)
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALLS

    axis = "horizontal" if _GEN_CALLS % 2 == 0 else "vertical"
    frame_count = (3, 4, 5)[_GEN_CALLS % 3]
    solid_count = 1 + (_GEN_CALLS % 2)

    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(axis, frame_count, solid_count)
        if grid is not None:
            _GEN_CALLS += 1
            return grid
    raise RuntimeError("generate_input: failed to build a valid sample")
