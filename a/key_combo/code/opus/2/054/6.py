"""Puzzle 6: marker-side-guided mirror embedding."""

from __future__ import annotations

from collections import deque

import numpy as np

BG = 0
FG = 5
MARKERS = {1, 2, 3, 4}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []

    for sr, sc in zip(*np.where(mask)):
        if seen[sr, sc]:
            continue
        q = deque([(int(sr), int(sc))])
        seen[sr, sc] = True
        comp: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for nr, nc in _neighbors4(r, c):
                if not _in_bounds((h, w), nr, nc):
                    continue
                if not mask[nr, nc] or seen[nr, nc]:
                    continue
                seen[nr, nc] = True
                q.append((nr, nc))
        out.append(comp)

    return out


def _rect_frame(comp: list[tuple[int, int]], grid: np.ndarray) -> dict | None:
    rs, cs = zip(*comp)
    r0, c0 = min(rs), min(cs)
    r1, c1 = max(rs), max(cs)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    for c in range(c0, c1 + 1):
        if grid[r0, c] != FG or grid[r1, c] != FG:
            return None
    for r in range(r0, r1 + 1):
        if grid[r, c0] != FG or grid[r, c1] != FG:
            return None

    for rr in range(r0 + 1, r1):
        for cc in range(c0 + 1, c1):
            if grid[rr, cc] != BG:
                return None

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    if interior.size == 0 or np.any(interior == FG):
        return None

    hole = interior == BG
    hole_cells = int(hole.sum())
    if not (3 <= hole_cells <= 8):
        return None

    mid_r = (r0 + r1) // 2
    mid_c = (c0 + c1) // 2
    marker = None
    candidates = [
        (r0 - 1, mid_c, "V"),
        (r1 + 1, mid_c, "V"),
        (mid_r, c0 - 1, "H"),
        (mid_r, c1 + 1, "H"),
    ]

    for mr, mc, axis in candidates:
        if _in_bounds(grid.shape, mr, mc) and int(grid[mr, mc]) in MARKERS:
            marker = (mr, mc, axis, int(grid[mr, mc]))
            break

    if marker is None:
        return None

    return {
        "r0": r0,
        "c0": c0,
        "r1": r1,
        "c1": c1,
        "hole": hole,
        "axis": marker[2],
        "marker_color": marker[3],
        "bbox": {(r, c) for r, c in comp},
    }


def _sprite_shape(comp: list[tuple[int, int]], grid: np.ndarray) -> dict | None:
    rs, cs = zip(*comp)
    r0, c0 = min(rs), min(cs)
    r1, c1 = max(rs), max(cs)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    area = len(comp)
    if not (3 <= area <= 8):
        return None

    shape = np.zeros((h, w), dtype=bool)
    for r, c in comp:
        shape[r - r0, c - c0] = True

    # must not be a rectangle frame
    if sum(1 for rr, cc in comp if rr in (r0, r1) or cc in (c0, c1)) == area:
        if h >= 3 and w >= 3 and h * w - (h - 2) * (w - 2) == area:
            return None

    return {
        "r0": r0,
        "c0": c0,
        "shape": shape,
        "bbox": {(r, c) for r, c in comp},
    }


def _shape_key(shape: np.ndarray) -> tuple[int, ...]:
    return tuple(int(v) for v in shape.flatten())


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    out = grid.copy()

    components = _components(grid == FG)
    frames: list[dict] = []
    sprites: list[dict] = []

    for comp in components:
        frame = _rect_frame(comp, grid)
        if frame is not None:
            frames.append(frame)
            continue
        sp = _sprite_shape(comp, grid)
        if sp is not None:
            sprites.append(sp)

    if not frames or not sprites:
        return out

    frame_by_key: dict[tuple[int, ...], list[dict]] = {}
    for fr in frames:
        # transform depends on axis before matching
        key = _shape_key(fr["hole"])
        frame_by_key.setdefault(key, []).append(fr)

    used_frames = set()
    used_sprites = set()

    for idx, sp in enumerate(sprites):
        best = None
        shape = sp["shape"]

        # Try all unused frames and choose the one matching transform rule.
        for key, flist in frame_by_key.items():
            for fr in flist:
                axis = fr["axis"]
                trans = np.fliplr(shape) if axis == "H" else np.flipud(shape)
                k = _shape_key(fr["hole"])
                if _shape_key(trans) != k:
                    continue

                fr_id = (fr["r0"], fr["c0"], fr["r1"], fr["c1"])
                if fr_id in used_frames or idx in used_sprites:
                    continue
                best = (fr, trans, idx, fr_id)
                break
            if best is not None:
                break

        if best is None:
            continue

        fr, transformed, sid, fr_id = best
        used_frames.add(fr_id)
        used_sprites.add(sid)

        # erase sprite cells
        for r, c in sp["bbox"]:
            out[r, c] = BG

        for rr, cc in zip(*np.where(transformed)):
            out[fr["r0"] + 1 + int(rr), fr["c0"] + 1 + int(cc)] = fr["marker_color"]

    return out


def _draw_shape(grid: np.ndarray, top: int, left: int, shape: np.ndarray, color: int) -> None:
    for r, c in zip(*np.where(shape)):
        grid[top + int(r), left + int(c)] = color


def _draw_frame(grid: np.ndarray, top: int, left: int, hole: np.ndarray) -> None:
    h0, w0 = hole.shape
    r0, c0 = top, left
    r1, c1 = top + h0 + 1, left + w0 + 1

    for c in range(c0, c1 + 1):
        grid[r0, c] = FG
        grid[r1, c] = FG
    for r in range(r0, r1 + 1):
        grid[r, c0] = FG
        grid[r, c1] = FG

    for rr in range(h0):
        for cc in range(w0):
            grid[top + 1 + rr, left + 1 + cc] = 0


def _sprite_templates(rng: np.random.Generator) -> list[np.ndarray]:
    return [
        np.array([[1, 1], [1, 0]], dtype=int),
        np.array([[1, 1, 1]], dtype=int),
        np.array([[1, 1], [1, 1]], dtype=int),
        np.array([[1, 1], [0, 1]], dtype=int),
        np.array([[1, 0, 1], [1, 1, 1]], dtype=int),
        np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
    ]


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    H = 18
    W = 18

    for _ in range(250):
        grid = np.zeros((H, W), dtype=int)
        occupied = np.zeros((H, W), dtype=bool)

        n = int(rng.integers(3, 5))
        sprites: list[dict] = []
        shapes = _sprite_templates(rng)

        # place sprites in upper area
        for i in range(n):
            for _ in range(200):
                shape = shapes[int(rng.integers(0, len(shapes)))]
                h0, w0 = shape.shape
                top = int(rng.integers(1, 7 - h0))
                left = int(rng.integers(1, 8 - w0))
                left = int(rng.integers(1, W - w0 - 1))
                if occupied[top : top + h0, left : left + w0].any():
                    continue

                _draw_shape(grid, top, left, shape, FG)
                occupied[top : top + h0 + 1, left : left + w0 + 1] = True
                sprites.append(
                    {
                        "top": top,
                        "left": left,
                        "shape": shape,
                    }
                )
                break

        if len(sprites) != n:
            continue

        marker_colors = [1, 2, 3, 4]
        row = 8
        for i, sprite in enumerate(sprites):
            for _ in range(80):
                shape = sprite["shape"]
                axis = rng.choice(["H", "H", "V", "V"])
                frame_hole = np.fliplr(shape) if axis == "H" else np.flipud(shape)
                h0, w0 = frame_hole.shape
                max_top = H - h0 - 1
                if row >= max_top:
                    continue

                top = int(rng.integers(row, max_top))
                left = int(rng.integers(1, W - w0 - 1))

                if top + h0 + 1 >= H - 1:
                    continue
                if occupied[top : top + h0 + 2, left : left + w0 + 2].any():
                    continue

                # frame hole contains the transformed shape by design
                _draw_frame(grid, top, left, frame_hole)
                occupied[top : top + h0 + 2, left : left + w0 + 2] = True

                mr = (top + h0 + 1) // 2
                if axis == "V":
                    if rng.integers(0, 2) == 0:
                        marker_pos = (top - 1, (left + 1 + w0 // 2))
                    else:
                        marker_pos = (top + h0 + 2, (left + 1 + w0 // 2))
                else:
                    if rng.integers(0, 2) == 0:
                        marker_pos = ((top + 1 + h0 // 2), left - 1)
                    else:
                        marker_pos = ((top + 1 + h0 // 2), left + w0 + 2)

                if not _in_bounds((H, W), marker_pos[0], marker_pos[1]):
                    continue
                grid[marker_pos] = marker_colors[i % len(marker_colors)]
                occupied[marker_pos] = True

                row = top + h0 + 4
                break

        return grid

    return np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 5, 5, 5, 0, 0, 0, 0, 0, 5, 5, 5, 0, 0, 0, 0, 0],
            [0, 0, 5, 1, 5, 0, 0, 0, 0, 0, 5, 0, 5, 0, 0, 0, 0, 0],
            [0, 0, 5, 5, 5, 0, 5, 5, 5, 0, 5, 5, 5, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 5, 1, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 5, 0, 5, 0, 5, 0, 5, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 5, 1, 5, 5, 0, 5, 5, 5, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 5, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=int,
    )


if __name__ == "__main__":
    g = generate_input()
    print(main(g))
