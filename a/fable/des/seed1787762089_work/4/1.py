from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
COUNT_COLOR = 5
FRAME_SIZE = 7
INNER_SIZE = FRAME_SIZE - 2
FRAME_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 600
MAX_CANDIDATES_PER_FRAME = 80

_BASE_SHAPES = (
    np.array([[1, 1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1], [0, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _component_mask(coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _trim(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _shape_signature(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask.astype(int))


def _center_origin(top: int, left: int, outer_h: int, outer_w: int, inner_shape: tuple[int, int]) -> tuple[int, int]:
    ih, iw = inner_shape
    return top + (outer_h - ih) // 2, left + (outer_w - iw) // 2


def _detect_frames(grid: np.ndarray, bg: int) -> tuple[list[dict], list[list[tuple[int, int]]], set[int]]:
    comps = _components(grid != bg)
    frames: list[dict] = []
    frame_ids: set[int] = set()

    for idx, comp in enumerate(comps):
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h != w or h < 5:
            continue
        expected = {
            (r, c)
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if r in (r0, r1) or c in (c0, c1)
        }
        if set(comp) != expected:
            continue
        beacon_pos = (r0, c0)
        colors = Counter(int(grid[r, c]) for r, c in comp)
        if len(colors) != 2 or beacon_pos not in comp:
            continue
        beacon_color = int(grid[beacon_pos])
        if colors[beacon_color] != 1:
            continue
        frame_color, count = colors.most_common(1)[0]
        if frame_color == beacon_color or count != len(comp) - 1:
            continue
        if any(int(grid[r, c]) != frame_color for r, c in expected if (r, c) != beacon_pos):
            continue
        frames.append(
            {
                "idx": idx,
                "coords": comp,
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "frame_color": int(frame_color),
                "beacon_color": beacon_color,
                "size": h,
            }
        )
        frame_ids.add(idx)

    return frames, comps, frame_ids


def _extract_sprites_and_k(
    grid: np.ndarray, comps: list[list[tuple[int, int]]], frames: list[dict], frame_ids: set[int]
) -> int:
    used: set[int] = set()
    for frame in frames:
        candidates: list[tuple[int, list[tuple[int, int]]]] = []
        for idx, comp in enumerate(comps):
            if idx in frame_ids or idx in used:
                continue
            r0, c0, r1, c1 = _bbox(comp)
            if r0 != frame["r1"] + 2:
                continue
            if not (frame["c0"] + 1 <= c0 <= c1 <= frame["c1"] - 1):
                continue
            colors = {int(grid[r, c]) for r, c in comp}
            if colors != {frame["frame_color"]}:
                continue
            candidates.append((idx, comp))
        assert len(candidates) == 1, "each frame must have exactly one sprite directly below it"
        comp_idx, comp = candidates[0]
        frame["sprite_shape"] = _trim(_component_mask(comp))
        used.add(comp_idx)

    leftovers = [comp for idx, comp in enumerate(comps) if idx not in frame_ids and idx not in used]
    assert len(leftovers) == 1, "expected exactly one leftover component for the count bar"
    bar = leftovers[0]
    r0, c0, r1, c1 = _bbox(bar)
    assert r0 == r1 and len(bar) == c1 - c0 + 1, "count bar must be a horizontal segment"
    return len(bar)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = int(np.bincount(g.ravel()).argmax())
    frames, comps, frame_ids = _detect_frames(g, bg)
    assert len(frames) == 4, "expected four hollow frames"
    k = _extract_sprites_and_k(g, comps, frames, frame_ids)

    color_to_idx = {frame["frame_color"]: i for i, frame in enumerate(frames)}
    assert len(color_to_idx) == len(frames), "frame colors must be distinct"

    next_idx: dict[int, int] = {}
    for i, frame in enumerate(frames):
        assert frame["beacon_color"] in color_to_idx, "beacon must point to an existing frame color"
        next_idx[i] = color_to_idx[frame["beacon_color"]]

    walk = []
    cur = 0
    for _ in range(len(frames)):
        walk.append(cur)
        cur = next_idx[cur]
    assert len(set(walk)) == len(frames) and cur == walk[0], "beacons must form one simple cycle"

    prev_idx = {dst: src for src, dst in next_idx.items()}
    out = np.full_like(g, bg)
    for frame in frames:
        for r, c in frame["coords"]:
            out[r, c] = g[r, c]

    for i, frame in enumerate(frames):
        src = i
        for _ in range(k):
            src = prev_idx[src]
        shape = frames[src]["sprite_shape"]
        top, left = _center_origin(frame["r0"] + 1, frame["c0"] + 1, frame["size"] - 2, frame["size"] - 2, shape.shape)
        out[top : top + shape.shape[0], left : left + shape.shape[1]][shape] = frame["frame_color"]

    return out


def _transform_shape(shape: np.ndarray, rng: random.Random) -> np.ndarray:
    out = np.array(shape, copy=True)
    out = np.rot90(out, rng.randrange(4))
    if rng.random() < 0.5:
        out = np.fliplr(out)
    return _trim(out)


def _choose_shapes(rng: random.Random) -> list[np.ndarray]:
    bases = list(_BASE_SHAPES)
    rng.shuffle(bases)
    shapes: list[np.ndarray] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()
    for base in bases:
        for _ in range(8):
            shape = _transform_shape(base, rng)
            sig = _shape_signature(shape)
            if sig in seen:
                continue
            seen.add(sig)
            shapes.append(shape)
            break
        if len(shapes) == 4:
            return shapes
    raise RuntimeError("failed to choose four distinct shapes")


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _place_frames(h: int, w: int, shapes: list[np.ndarray], rng: random.Random) -> list[tuple[int, int]]:
    candidates_by_idx: dict[int, list[tuple[int, int]]] = {}
    for idx, shape in enumerate(shapes):
        sh, _ = shape.shape
        row_max = h - sh - 10
        col_max = w - FRAME_SIZE - 1
        candidates = [(r, c) for r in range(1, row_max + 1) for c in range(1, col_max + 1)]
        rng.shuffle(candidates)
        if len(candidates) > MAX_CANDIDATES_PER_FRAME:
            candidates = candidates[:MAX_CANDIDATES_PER_FRAME]
        candidates_by_idx[idx] = candidates

    order = sorted(candidates_by_idx, key=lambda i: (len(candidates_by_idx[i]), -shapes[i].shape[0]))
    positions: dict[int, tuple[int, int]] = {}
    reserved: list[tuple[int, int, int, int]] = []

    def backtrack(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        sh, _ = shapes[idx].shape
        for r, c in candidates_by_idx[idx]:
            footprint_bottom = r + 7 + sh
            actual = (r, c, footprint_bottom, c + FRAME_SIZE - 1)
            expanded = (actual[0] - 1, actual[1] - 1, actual[2] + 1, actual[3] + 1)
            if any(_rects_overlap(expanded, other) for other in reserved):
                continue
            positions[idx] = (r, c)
            reserved.append(expanded)
            if backtrack(pos + 1):
                return True
            reserved.pop()
            del positions[idx]
        return False

    if not backtrack(0):
        raise RuntimeError("failed to place all frames")

    return [positions[i] for i in range(4)]


def _draw_frame(grid: np.ndarray, top: int, left: int, frame_color: int, beacon_color: int) -> None:
    bottom = top + FRAME_SIZE - 1
    right = left + FRAME_SIZE - 1
    grid[top : bottom + 1, left] = frame_color
    grid[top : bottom + 1, right] = frame_color
    grid[top, left : right + 1] = frame_color
    grid[bottom, left : right + 1] = frame_color
    grid[top, left] = beacon_color


def _draw_shape(grid: np.ndarray, top: int, left: int, shape: np.ndarray, color: int) -> None:
    grid[top : top + shape.shape[0], left : left + shape.shape[1]][shape] = color


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        try:
            h = rng.randint(23, 28)
            w = rng.randint(23, 28)
            grid = np.full((h, w), BG, dtype=int)

            frame_colors = rng.sample(FRAME_COLORS, 4)
            shapes = _choose_shapes(rng)
            positions = _place_frames(h, w, shapes, rng)

            cycle = list(range(4))
            rng.shuffle(cycle)
            next_color: dict[int, int] = {}
            for i, idx in enumerate(cycle):
                next_color[idx] = frame_colors[cycle[(i + 1) % 4]]

            for idx, (top, left) in enumerate(positions):
                frame_color = frame_colors[idx]
                _draw_frame(grid, top, left, frame_color, next_color[idx])
                shape = shapes[idx]
                sprite_top = top + FRAME_SIZE + 1
                sprite_left = left + 1 + (INNER_SIZE - shape.shape[1]) // 2
                _draw_shape(grid, sprite_top, sprite_left, shape, frame_color)

            k = rng.randint(1, 3)
            bar_row = rng.choice([h - 2, h - 1])
            right_margin = rng.randint(1, 2)
            bar_left = w - k - right_margin
            grid[bar_row, bar_left : bar_left + k] = COUNT_COLOR

            main(grid)
            return grid
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid input")
