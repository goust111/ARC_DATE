from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
RED = 2
FRAME_COLORS = [1, 3, 4, 5, 6, 7, 8, 9]
SPRITE_COLORS = [1, 3, 4, 5, 6, 7, 8, 9]
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _is_frame(grid: np.ndarray, color: int, r0: int, c0: int, r1: int, c1: int) -> bool:
    if r1 - r0 + 1 < 4 or c1 - c0 + 1 < 4:
        return False
    if not np.all(grid[r0, c0 : c1 + 1] == color):
        return False
    if not np.all(grid[r1, c0 : c1 + 1] == color):
        return False
    if not np.all(grid[r0 : r1 + 1, c0] == color):
        return False
    if not np.all(grid[r0 : r1 + 1, c1] == color):
        return False
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    return not np.any(interior == color)


def _find_frames(grid: np.ndarray) -> list[dict[str, int | tuple[int, int, int, int]]]:
    h, w = grid.shape
    frames: list[dict[str, int | tuple[int, int, int, int]]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in (BG, RED)):
        for r0 in range(h - 3):
            for c0 in range(w - 3):
                for r1 in range(r0 + 3, min(h, r0 + 8)):
                    for c1 in range(c0 + 3, min(w, c0 + 8)):
                        if _is_frame(grid, color, r0, c0, r1, c1):
                            frames.append({"color": color, "bbox": (r0, c0, r1, c1)})
    # Exact rectangular frames do not contain other valid same-color frames in the
    # generated data, but this guard removes accidental duplicate scans.
    unique: list[dict[str, int | tuple[int, int, int, int]]] = []
    seen: set[tuple[int, int, int, int, int]] = set()
    for frame in frames:
        key = (int(frame["color"]), *map(int, frame["bbox"]))
        if key not in seen:
            seen.add(key)
            unique.append(frame)
    return unique


def _clockwise_rot(sprite: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(sprite, (-turns) % 4)


def _red_turn_count(grid: np.ndarray, color: int, frame_boxes: list[tuple[int, int, int, int]]) -> int:
    h, w = grid.shape
    inside_frame = np.zeros((h, w), dtype=bool)
    for r0, c0, r1, c1 in frame_boxes:
        inside_frame[r0 : r1 + 1, c0 : c1 + 1] = True
    count = 0
    for r, c in np.argwhere((grid == RED) & (~inside_frame)):
        nbs = 0
        for dr, dc in NEIGHBORS4:
            nr, nc = int(r) + dr, int(c) + dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == color:
                nbs += 1
        if nbs >= 2:
            count += 1
    return count


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    out = g.copy()
    by_color: dict[int, list[tuple[int, int, int, int]]] = defaultdict(list)
    for frame in _find_frames(g):
        by_color[int(frame["color"])].append(tuple(map(int, frame["bbox"])))

    for color, boxes in by_color.items():
        if len(boxes) < 2:
            continue
        sources: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
        targets: list[tuple[int, int, int, int]] = []
        for box in boxes:
            r0, c0, r1, c1 = box
            interior = g[r0 + 1 : r1, c0 + 1 : c1]
            payload = (interior != BG) & (interior != color) & (interior != RED)
            if np.any(payload):
                sources.append((box, interior.copy()))
            elif np.all(interior == BG):
                targets.append(box)
        if len(sources) != 1 or len(targets) != 1:
            continue
        (_src_box, sprite), target = sources[0], targets[0]
        turns = _red_turn_count(g, color, boxes)
        rotated = _clockwise_rot(sprite, turns)
        tr0, tc0, tr1, tc1 = target
        ih, iw = tr1 - tr0 - 1, tc1 - tc0 - 1
        assert rotated.shape[0] <= ih and rotated.shape[1] <= iw
        patch = out[tr0 + 1 : tr1, tc0 + 1 : tc1]
        mask = (rotated != BG) & (rotated != color) & (rotated != RED)
        patch[: rotated.shape[0], : rotated.shape[1]][mask] = rotated[mask]
    return out


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _draw_path(grid: np.ndarray, points: list[tuple[int, int]], color: int) -> None:
    for a, b in zip(points, points[1:]):
        r0, c0 = a
        r1, c1 = b
        assert r0 == r1 or c0 == c1
        dr = (r1 > r0) - (r1 < r0)
        dc = (c1 > c0) - (c1 < c0)
        n = max(abs(r1 - r0), abs(c1 - c0))
        for i in range(n + 1):
            r, c = r0 + dr * i, c0 + dc * i
            if grid[r, c] == BG:
                grid[r, c] = color
    for r, c in points[1:-1]:
        grid[r, c] = RED


def _make_sprite(rng: random.Random, size: int, avoid: set[int]) -> np.ndarray:
    choices = [c for c in SPRITE_COLORS if c not in avoid]
    if not choices:
        choices = [c for c in SPRITE_COLORS if c not in (BG, RED)]
    sprite = np.zeros((size, size), dtype=int)
    for r in range(size):
        for c in range(size):
            if rng.random() < 0.45:
                sprite[r, c] = rng.choice(choices)
    # Avoid accidental hollow-frame sprites being detected as extra regions.
    sprite[0, 0] = BG
    sprite[-1, -1] = BG
    if not np.any(sprite):
        sprite[size // 2, size // 2] = rng.choice(choices)
    return sprite


def _pair_layout(pair_index: int, color: int, turns: int, size: int, sprite: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    band_h = 10
    h, w = band_h, 30
    grid = np.zeros((h, w), dtype=int)
    expected = grid.copy()
    top = 3
    src = (top, 1, top + size + 1, 1 + size + 1)
    tgt = (top, 22, top + size + 1, 22 + size + 1)
    _draw_frame(grid, src, color)
    _draw_frame(grid, tgt, color)
    _draw_frame(expected, src, color)
    _draw_frame(expected, tgt, color)

    sr0, sc0, sr1, sc1 = src
    tr0, tc0, tr1, tc1 = tgt
    grid[sr0 + 1 : sr1, sc0 + 1 : sc1] = sprite
    expected[sr0 + 1 : sr1, sc0 + 1 : sc1] = sprite

    source_mid = (sr0 + size // 2 + 1, sc1 + 1)
    target_left = (tr0 + size // 2 + 1, tc0 - 1)
    target_top = (tr0 - 1, tc0 + size // 2 + 1)
    mid_c = 14
    if turns == 0:
        pts = [source_mid, target_left]
        target = target_left
    elif turns == 1:
        pts = [source_mid, (source_mid[0], target_top[1]), target_top]
        target = target_top
    elif turns == 2:
        pts = [source_mid, (source_mid[0], mid_c), (target_left[0], mid_c), target_left]
        target = target_left
    else:
        upper = 1
        pts = [source_mid, (source_mid[0], mid_c), (upper, mid_c), (upper, target_top[1]), target_top]
        target = target_top
    _draw_path(grid, pts, color)
    _draw_path(expected, pts, color)
    # Ensure the final path cell touches the chosen target side without painting inside it.
    assert target

    rotated = _clockwise_rot(sprite, turns)
    patch = expected[tr0 + 1 : tr1, tc0 + 1 : tc1]
    mask = rotated != BG
    patch[: rotated.shape[0], : rotated.shape[1]][mask] = rotated[mask]
    return grid, expected


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n = rng.randint(1, 3)
    band_h = 10
    grid = np.zeros((band_h * n + 1, 30), dtype=int)
    colors = rng.sample(FRAME_COLORS, n)
    for i, color in enumerate(colors):
        size = rng.randint(3, 5)
        turns = rng.randint(0, 3)
        sprite = _make_sprite(rng, size, set(colors) | {BG, RED})
        sub, _sub_expected = _pair_layout(i, color, turns, size, sprite)
        r0 = i * band_h
        grid[r0 : r0 + band_h, :] = sub
    try:
        if np.array_equal(main(grid), grid):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(300):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
