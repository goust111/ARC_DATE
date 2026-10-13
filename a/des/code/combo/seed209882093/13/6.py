from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
YELLOW = 4
FRAME_COLORS = (1, 2, 3, 5, 6, 7, 8, 9)
GEN_RETRIES = 500


def _dominant_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    if np.any(vals == BG):
        return BG
    return int(vals[np.argmax(counts)])


def _segments_1d(mask: np.ndarray) -> list[tuple[int, int]]:
    segs: list[tuple[int, int]] = []
    n = int(mask.size)
    i = 0
    while i < n:
        if not bool(mask[i]):
            i += 1
            continue
        j = i
        while j + 1 < n and bool(mask[j + 1]):
            j += 1
        segs.append((i, j))
        i = j + 1
    return segs


def _scan_horizontal_sides(
    grid: np.ndarray, color: int, want_top: bool
) -> list[dict[str, int]]:
    h, _w = grid.shape
    sides: list[dict[str, int]] = []
    for r in range(h):
        segs = _segments_1d(grid[r] == color)
        for (s1, e1), (s2, e2) in zip(segs, segs[1:]):
            if s2 != e1 + 2:
                continue
            c0, c1 = s1, e2
            if c1 - c0 < 2:
                continue
            if want_top:
                if r > 0 and np.any(grid[r - 1, c0 : c1 + 1] == color):
                    continue
                if r + 1 < h and np.any(grid[r + 1, c0 + 1 : c1] == color):
                    continue
                orientation = "top"
            else:
                if r + 1 < h and np.any(grid[r + 1, c0 : c1 + 1] == color):
                    continue
                if r > 0 and np.any(grid[r - 1, c0 + 1 : c1] == color):
                    continue
                orientation = "bottom"
            sides.append(
                {
                    "kind": orientation,
                    "color": int(color),
                    "r": int(r),
                    "c0": int(c0),
                    "c1": int(c1),
                    "gap": int(e1 + 1),
                }
            )
    return sides


def _scan_vertical_sides(
    grid: np.ndarray, color: int, want_left: bool
) -> list[dict[str, int]]:
    h, w = grid.shape
    sides: list[dict[str, int]] = []
    for c in range(w):
        segs = _segments_1d(grid[:, c] == color)
        for (s1, e1), (s2, e2) in zip(segs, segs[1:]):
            if s2 != e1 + 2:
                continue
            r0, r1 = s1, e2
            if r1 - r0 < 2:
                continue
            if want_left:
                if c > 0 and np.any(grid[r0 : r1 + 1, c - 1] == color):
                    continue
                if c + 1 < w and np.any(grid[r0 + 1 : r1, c + 1] == color):
                    continue
                orientation = "left"
            else:
                if c + 1 < w and np.any(grid[r0 : r1 + 1, c + 1] == color):
                    continue
                if c > 0 and np.any(grid[r0 + 1 : r1, c - 1] == color):
                    continue
                orientation = "right"
            sides.append(
                {
                    "kind": orientation,
                    "color": int(color),
                    "c": int(c),
                    "r0": int(r0),
                    "r1": int(r1),
                    "gap": int(e1 + 1),
                }
            )
    return sides


def _frame_border_mask(shape: tuple[int, int], frame: dict[str, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    r0, c0, r1, c1 = frame["r0"], frame["c0"], frame["r1"], frame["c1"]
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    mask[r0, frame["top_gap"]] = False
    mask[r1, frame["bottom_gap"]] = False
    mask[frame["left_gap"], c0] = False
    mask[frame["right_gap"], c1] = False
    return mask


def _validate_frame(grid: np.ndarray, frame: dict[str, int], bg: int) -> bool:
    h, w = grid.shape
    color = frame["color"]
    r0, c0, r1, c1 = frame["r0"], frame["c0"], frame["r1"], frame["c1"]
    top_gap = frame["top_gap"]
    bottom_gap = frame["bottom_gap"]
    left_gap = frame["left_gap"]
    right_gap = frame["right_gap"]

    if not (0 <= r0 < r1 < h and 0 <= c0 < c1 < w):
        return False
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    if not (c0 < top_gap < c1 and c0 < bottom_gap < c1):
        return False
    if not (r0 < left_gap < r1 and r0 < right_gap < r1):
        return False

    if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] != bg):
        return False

    top = grid[r0, c0 : c1 + 1]
    bottom = grid[r1, c0 : c1 + 1]
    left = grid[r0 : r1 + 1, c0]
    right = grid[r0 : r1 + 1, c1]

    exp_top = np.full(c1 - c0 + 1, color, dtype=int)
    exp_bottom = np.full(c1 - c0 + 1, color, dtype=int)
    exp_left = np.full(r1 - r0 + 1, color, dtype=int)
    exp_right = np.full(r1 - r0 + 1, color, dtype=int)
    exp_top[top_gap - c0] = bg
    exp_bottom[bottom_gap - c0] = bg
    exp_left[left_gap - r0] = bg
    exp_right[right_gap - r0] = bg

    if not np.array_equal(top, exp_top):
        return False
    if not np.array_equal(bottom, exp_bottom):
        return False
    if not np.array_equal(left, exp_left):
        return False
    if not np.array_equal(right, exp_right):
        return False

    for c in range(c0, c1 + 1):
        if c == top_gap:
            continue
        if r0 > 0 and int(grid[r0 - 1, c]) == color:
            return False
        if r1 + 1 < h and int(grid[r1 + 1, c]) == color:
            return False
    for r in range(r0, r1 + 1):
        if r == left_gap:
            continue
        if c0 > 0 and int(grid[r, c0 - 1]) == color:
            return False
        if c1 + 1 < w and int(grid[r, c1 + 1]) == color:
            return False

    border = _frame_border_mask(grid.shape, frame)
    window = grid[r0 : r1 + 1, c0 : c1 + 1]
    local_border = border[r0 : r1 + 1, c0 : c1 + 1]
    if np.any(window[~local_border] != bg):
        return False
    if np.any(window[local_border] != color):
        return False
    return True


def _find_frames(grid: np.ndarray, bg: int | None = None) -> list[dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    bg_color = _dominant_bg(g) if bg is None else int(bg)
    colors = [int(c) for c in np.unique(g) if int(c) != bg_color]

    frames: list[dict[str, int]] = []
    seen: set[tuple[int, int, int, int, int]] = set()
    for color in colors:
        tops = _scan_horizontal_sides(g, color, want_top=True)
        bottoms = _scan_horizontal_sides(g, color, want_top=False)
        lefts = _scan_vertical_sides(g, color, want_left=True)
        rights = _scan_vertical_sides(g, color, want_left=False)

        bottoms_by_span: dict[tuple[int, int, int], list[dict[str, int]]] = {}
        lefts_by_span: dict[tuple[int, int, int], list[dict[str, int]]] = {}
        rights_by_span: dict[tuple[int, int, int], list[dict[str, int]]] = {}

        for side in bottoms:
            bottoms_by_span.setdefault((side["color"], side["c0"], side["c1"]), []).append(side)
        for side in lefts:
            lefts_by_span.setdefault((side["color"], side["r0"], side["r1"]), []).append(side)
        for side in rights:
            rights_by_span.setdefault((side["color"], side["r0"], side["r1"]), []).append(side)

        for top in tops:
            span_key = (top["color"], top["c0"], top["c1"])
            for bottom in bottoms_by_span.get(span_key, []):
                if bottom["r"] <= top["r"]:
                    continue
                side_span = (color, top["r"], bottom["r"])
                for left in lefts_by_span.get(side_span, []):
                    if left["c"] != top["c0"]:
                        continue
                    for right in rights_by_span.get(side_span, []):
                        if right["c"] != top["c1"]:
                            continue
                        frame = {
                            "color": int(color),
                            "r0": int(top["r"]),
                            "c0": int(top["c0"]),
                            "r1": int(bottom["r"]),
                            "c1": int(top["c1"]),
                            "top_gap": int(top["gap"]),
                            "bottom_gap": int(bottom["gap"]),
                            "left_gap": int(left["gap"]),
                            "right_gap": int(right["gap"]),
                        }
                        bbox_key = (
                            frame["color"],
                            frame["r0"],
                            frame["c0"],
                            frame["r1"],
                            frame["c1"],
                        )
                        if bbox_key in seen:
                            continue
                        if _validate_frame(g, frame, bg_color):
                            seen.add(bbox_key)
                            frames.append(frame)
    frames.sort(key=lambda fr: (fr["r0"], fr["c0"], fr["r1"], fr["c1"], fr["color"]))
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep every hollow rectangular frame unchanged, then shoot one inward ray from
    each side gap. Horizontal/vertical crossings become yellow.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    out = g.copy()
    bg = _dominant_bg(g)
    frames = _find_frames(g, bg=bg)

    for frame in frames:
        color = frame["color"]
        r0, c0, r1, c1 = frame["r0"], frame["c0"], frame["r1"], frame["c1"]
        top_gap = frame["top_gap"]
        bottom_gap = frame["bottom_gap"]
        left_gap = frame["left_gap"]
        right_gap = frame["right_gap"]

        vertical = np.zeros_like(g, dtype=bool)
        horizontal = np.zeros_like(g, dtype=bool)
        vertical[r0 + 1 : r1, top_gap] = True
        vertical[r0 + 1 : r1, bottom_gap] = True
        horizontal[left_gap, c0 + 1 : c1] = True
        horizontal[right_gap, c0 + 1 : c1] = True

        ray_only = (vertical | horizontal) & ~(vertical & horizontal)
        out[ray_only] = color
        out[vertical & horizontal] = YELLOW

    return out


def _draw_frame(grid: np.ndarray, frame: dict[str, int], bg: int = BG) -> None:
    color = frame["color"]
    r0, c0, r1, c1 = frame["r0"], frame["c0"], frame["r1"], frame["c1"]
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, frame["top_gap"]] = bg
    grid[r1, frame["bottom_gap"]] = bg
    grid[frame["left_gap"], c0] = bg
    grid[frame["right_gap"], c1] = bg


def _sample_gap(rng: random.Random, start: int, stop: int) -> int:
    return int(rng.randint(start, stop))


def _random_frame(
    rng: random.Random, h: int, w: int, color: int
) -> dict[str, int]:
    r0 = 0
    c0 = 0
    r1 = h - 1
    c1 = w - 1
    top_gap = _sample_gap(rng, c0 + 1, c1 - 1)
    if rng.random() < 0.35:
        bottom_gap = top_gap
    else:
        bottom_gap = _sample_gap(rng, c0 + 1, c1 - 1)
    left_gap = _sample_gap(rng, r0 + 1, r1 - 1)
    if rng.random() < 0.35:
        right_gap = left_gap
    else:
        right_gap = _sample_gap(rng, r0 + 1, r1 - 1)
    return {
        "color": int(color),
        "r0": int(r0),
        "c0": int(c0),
        "r1": int(r1),
        "c1": int(c1),
        "top_gap": int(top_gap),
        "bottom_gap": int(bottom_gap),
        "left_gap": int(left_gap),
        "right_gap": int(right_gap),
    }


def _shift_frame(frame: dict[str, int], dr: int, dc: int) -> dict[str, int]:
    out = dict(frame)
    out["r0"] += int(dr)
    out["r1"] += int(dr)
    out["c0"] += int(dc)
    out["c1"] += int(dc)
    out["top_gap"] += int(dc)
    out["bottom_gap"] += int(dc)
    out["left_gap"] += int(dr)
    out["right_gap"] += int(dr)
    return out


def _occupied_with_margin(
    blocked: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1
) -> bool:
    h, w = blocked.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h, r1 + pad + 1)
    cc1 = min(w, c1 + pad + 1)
    return bool(np.any(blocked[rr0:rr1, cc0:cc1]))


def _mark_blocked(
    blocked: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1
) -> None:
    h, w = blocked.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h, r1 + pad + 1)
    cc1 = min(w, c1 + pad + 1)
    blocked[rr0:rr1, cc0:cc1] = True


def _all_border_masks(shape: tuple[int, int], frames: Iterable[dict[str, int]]) -> np.ndarray:
    acc = np.zeros(shape, dtype=bool)
    for frame in frames:
        acc |= _frame_border_mask(shape, frame)
    return acc


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = int(rng.randint(10, 24))
    w = int(rng.randint(10, 24))
    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    max_frames = 3 if h * w < 220 else 4
    if h * w > 360 and rng.random() < 0.35:
        max_frames = 5
    target_frames = int(rng.randint(1, max_frames))

    colors = list(rng.sample(FRAME_COLORS, k=target_frames))
    if target_frames >= 2 and rng.random() < 0.3:
        j = int(rng.randrange(target_frames))
        k = int(rng.randrange(target_frames))
        while k == j:
            k = int(rng.randrange(target_frames))
        colors[k] = colors[j]

    placed: list[dict[str, int]] = []
    for color in colors:
        placed_here = False
        for _ in range(250):
            frame_h = int(rng.randint(3, min(8, h)))
            frame_w = int(rng.randint(3, min(8, w)))
            if rng.random() < 0.15:
                frame_h = int(rng.randint(3, min(10, h)))
            if rng.random() < 0.15:
                frame_w = int(rng.randint(3, min(10, w)))

            r0 = int(rng.randint(0, h - frame_h))
            c0 = int(rng.randint(0, w - frame_w))
            r1 = r0 + frame_h - 1
            c1 = c0 + frame_w - 1
            if _occupied_with_margin(blocked, r0, c0, r1, c1, pad=1):
                continue

            local = _random_frame(rng, frame_h, frame_w, color)
            frame = _shift_frame(local, r0, c0)
            test_grid = np.full((h, w), BG, dtype=int)
            _draw_frame(test_grid, frame, BG)
            if not _validate_frame(test_grid, frame, BG):
                continue

            _draw_frame(grid, frame, BG)
            _mark_blocked(blocked, r0, c0, r1, c1, pad=1)
            placed.append(frame)
            placed_here = True
            break
        if not placed_here:
            break

    if not placed:
        return None
    if len(placed) < target_frames and target_frames > 1:
        return None

    detected = _find_frames(grid, bg=BG)
    if len(detected) != len(placed):
        return None
    if not np.array_equal(_all_border_masks(grid.shape, detected), grid != BG):
        return None

    out = main(grid)
    if not np.any(out == YELLOW):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
