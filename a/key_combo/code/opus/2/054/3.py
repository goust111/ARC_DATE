"""Puzzle 3: gravity drop then control-indexed frame matching and embed."""

from __future__ import annotations

from collections import deque

import numpy as np

BG = 0
FG = 5
SPRITE_COLORS = frozenset({1, 2, 3})
STOP = {4}
PASS = {7}


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
    out = []

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


def _collect_sprites(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    sprite_mask = np.zeros((h, w), dtype=bool)
    for r in range(min(3, h)):
        for c in range(w):
            if grid[r, c] in SPRITE_COLORS:
                sprite_mask[r, c] = True

    comps = _components(sprite_mask)
    sprites: list[dict] = []

    for comp in comps:
        if not (4 <= len(comp) <= 6):
            continue

        rs, cs = zip(*comp)
        rmin, rmax = min(rs), max(rs)
        cmin, cmax = min(cs), max(cs)
        if rmin < 0 or rmax >= 3:
            continue

        h0, w0 = rmax - rmin + 1, cmax - cmin + 1
        shape = np.zeros((h0, w0), dtype=bool)
        for r, c in comp:
            shape[r - rmin, c - cmin] = True

        control_color = BG
        control_col = None
        ctrl_row = rmax + 1
        if _in_bounds((h, w), ctrl_row, cmin) and _in_bounds((h, w), ctrl_row, cmax):
            for cc in range(cmin, cmax + 1):
                if grid[ctrl_row, cc] in STOP | PASS:
                    control_color = int(grid[ctrl_row, cc])
                    control_col = cc
                    break
        if control_color not in STOP | PASS:
            continue

        sprites.append(
            {
                "color": int(grid[rmin, cmin]),
                "rmin": rmin,
                "rmax": rmax,
                "cmin": cmin,
                "cmax": cmax,
                "shape": shape,
                "ctrl_color": control_color,
                "ctrl_col": control_col,
            }
        )

    return sprites


def _collect_frames(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    frame_mask = grid == FG
    comps = _components(frame_mask)
    frames: list[dict] = []

    for comp in comps:
        rs, cs = zip(*comp)
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue

        if not all(grid[r0, c] == FG for c in range(c0, c1 + 1)):
            continue
        if not all(grid[r1, c] == FG for c in range(c0, c1 + 1)):
            continue
        if not all(grid[r, c0] == FG for r in range(r0, r1 + 1)):
            continue
        if not all(grid[r, c1] == FG for r in range(r0, r1 + 1)):
            continue

        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        if interior.size == 0:
            continue
        if np.any(interior != BG):
            continue

        hole = interior == BG
        if hole.size < 3 or not (1 <= int(hole.sum()) <= 8):
            continue

        frames.append(
            {
                "r0": r0,
                "r1": r1,
                "c0": c0,
                "c1": c1,
                "hole": hole,
            }
        )

    return frames


def _shape_key(shape: np.ndarray) -> tuple[int, ...]:
    return tuple(int(v) for v in shape.flatten())


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    out = grid.copy()

    sprites = _collect_sprites(grid)
    frames = _collect_frames(grid)
    if not sprites or not frames:
        return out

    used_frames: set[tuple[int, int, int, int]] = set()
    frame_by_shape: dict[tuple[int, ...], list[dict]] = {}
    for f in frames:
        k = _shape_key(f["hole"])
        frame_by_shape.setdefault(k, []).append(f)

    for sprite in sprites:
        key = _shape_key(sprite["shape"])
        candidates = []
        for fr in frame_by_shape.get(key, []):
            if (fr["r0"], fr["c0"], fr["r1"], fr["c1"]) in used_frames:
                continue
            if fr["r0"] <= sprite["rmax"]:
                continue
            if fr["c1"] < sprite["cmin"] or fr["c0"] > sprite["cmax"]:
                continue
            candidates.append(fr)

        if not candidates:
            continue
        candidates.sort(key=lambda f: f["r0"])
        idx = 0 if sprite["ctrl_color"] in STOP else 1
        if idx >= len(candidates):
            continue

        frame = candidates[idx]
        used_frames.add((frame["r0"], frame["c0"], frame["r1"], frame["c1"]))

        # erase sprite body and control pixel
        for rr, cc in zip(*np.where(sprite["shape"])):
            out[sprite["rmin"] + int(rr), sprite["cmin"] + int(cc)] = BG
        if sprite["ctrl_col"] is not None:
            out[sprite["rmax"] + 1, sprite["ctrl_col"]] = BG

        for rr, cc in zip(*np.where(sprite["shape"])):
            out[frame["r0"] + 1 + int(rr), frame["c0"] + 1 + int(cc)] = sprite["color"]

    return out


def _draw_sprite_shape(
    grid: np.ndarray,
    top: int,
    left: int,
    shape: np.ndarray,
    color: int,
) -> None:
    for r, c in zip(*np.where(shape)):
        grid[top + r, left + c] = color


def _random_shape(rng: np.random.Generator) -> np.ndarray:
    templates = [
        np.array([[1, 1], [1, 1]], dtype=int),
        np.array([[1, 1, 1, 1]], dtype=int),
        np.array([[1, 1, 1], [1, 1, 1]], dtype=int),
    ]
    return templates[int(rng.integers(0, len(templates)))]


def _paint_frame(grid: np.ndarray, top: int, left: int, hole: np.ndarray) -> None:
    h0, w0 = hole.shape
    r0, c0 = top, left
    r1, c1 = top + h0 + 1, left + w0 + 1

    grid[r0, c0 : c1 + 1] = FG
    grid[r1, c0 : c1 + 1] = FG
    grid[r0 : r1 + 1, c0] = FG
    grid[r0 : r1 + 1, c1] = FG
    grid[top + 1 : top + 1 + h0, left + 1 : left + 1 + w0] = 0

    for r in range(hole.shape[0]):
        for c in range(hole.shape[1]):
            if not hole[r, c]:
                grid[top + 1 + r, left + 1 + c] = 0


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()

    for _ in range(120):
        H = int(rng.integers(22, 29))
        W = int(rng.integers(18, 25))
        grid = np.zeros((H, W), dtype=int)
        occupied = np.zeros((H, W), dtype=bool)

        sprite_num = 3
        sprite_colors = [1, 2, 3]
        rng.shuffle(sprite_colors)
        controls = [4, 7, 4]
        rng.shuffle(controls)
        left_slots = np.array([1, 6, 11], dtype=int)
        rng.shuffle(left_slots)
        sprites: list[dict] = []

        ok = True
        for i in range(sprite_num):
            shape = _random_shape(rng)
            h0, w0 = shape.shape
            top = 0
            left = int(left_slots[i])
            if left + w0 > W - 2:
                left = max(1, W - w0 - 2)

            if np.any(occupied[top : top + h0 + 1, left : left + w0]):
                ok = False
                break

            _draw_sprite_shape(grid, top, left, shape, sprite_colors[i])
            control_row = top + h0
            control_col = left + (w0 - 1) // 2
            if not _in_bounds((H, W), control_row + 1, control_col):
                ok = False
                break

            grid[control_row + 1, control_col] = controls[i]
            occupied[top : top + h0 + 2, left : left + w0 + 1] = True
            sprites.append(
                {
                    "top": top,
                    "left": left,
                    "shape": shape,
                    "control": controls[i],
                }
            )

        if not ok or len(sprites) != sprite_num:
            continue

        frame_count = 0
        row_cursor = 6

        for sp in sprites:
            shape = sp["shape"]
            h0, w0 = shape.shape
            matches = 1 if sp["control"] == 4 else 2
            for _ in range(matches):
                max_top = H - h0 - 2
                if row_cursor >= max_top:
                    ok = False
                    break

                placed = False
                left = sp["left"] + int(rng.integers(-1, 2))
                left = max(1, min(W - w0 - 2, left))

                for _ in range(80):
                    top = int(rng.integers(row_cursor, max_top))
                    r0, c0 = top, left
                    r1, c1 = top + h0 + 1, left + w0 + 1
                    if occupied[r0 : r1 + 1, c0 : c1 + 1].any():
                        continue

                    _paint_frame(grid, top, left, shape)
                    occupied[r0 : r1 + 1, c0 : c1 + 1] = True
                    frame_count += 1
                    row_cursor = top + h0 + 3
                    placed = True
                    break

                if not placed:
                    ok = False
                    break

            if not ok:
                break

        if not ok:
            continue

        if frame_count < 4:
            continue

        used_controls = {s["control"] for s in sprites}
        if used_controls != {4, 7}:
            continue

        return grid

    # randomized fallback to keep generator diverse and valid enough for self-check
    H = 22
    W = 18
    grid = np.zeros((H, W), dtype=int)
    color_cycle = [1, 2, 3]
    shape = np.array([[1, 1], [1, 1]], dtype=int)
    for i, left in enumerate((1, 6, 11)):
        _draw_sprite_shape(grid, 0, left, shape, color_cycle[i % len(color_cycle)])
        grid[2, left + 1] = [4, 7, 4][i]

    for idx in range(4):
        top = 6 + idx * 4
        _paint_frame(grid, top, 1 + (idx % 4), np.array([[1, 1], [1, 1]], dtype=int))
        frame_color = 1 + (idx % 3)
        grid[top + 1, 2 + (idx % 4)] = frame_color

    return grid


if __name__ == "__main__":
    g = generate_input()
    print(main(g))
