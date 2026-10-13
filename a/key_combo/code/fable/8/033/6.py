from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, paste


RING_POS = (
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 2),
    (2, 2),
    (2, 1),
    (2, 0),
    (1, 0),
)
CORNER_IDXS = (0, 2, 4, 6)
EDGE_IDXS = (1, 3, 5, 7)
GEN_RETRIES = 300


def _border_color(grid: np.ndarray) -> int:
    border = np.concatenate((grid[0], grid[-1], grid[1:-1, 0], grid[1:-1, -1]))
    colors = set(map(int, border))
    assert len(colors) == 1, "outer border must be a single solid color"
    return next(iter(colors))


def _background_color(grid: np.ndarray, key_color: int) -> int:
    inner = np.asarray(grid[1:-1, 1:-1], dtype=int)
    candidates = [int(v) for v in inner.flat if int(v) != key_color]
    assert candidates, "interior must contain a non-border color"
    return Counter(candidates).most_common(1)[0][0]


def _rotate_clockwise(seq: list[int], steps: int) -> list[int]:
    n = len(seq)
    out = [0] * n
    for i, value in enumerate(seq):
        out[(i + steps) % n] = int(value)
    return out


def _extract_rings(grid: np.ndarray) -> tuple[int, int, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 5, "grid must be at least 5x5"

    key_color = _border_color(g)
    bg = _background_color(g, key_color)
    h, w = g.shape
    rings: list[dict[str, object]] = []
    occupied = np.zeros_like(g, dtype=bool)

    for r0 in range(1, h - 2):
        for c0 in range(1, w - 2):
            patch = g[r0 : r0 + 3, c0 : c0 + 3]
            if patch.shape != (3, 3) or int(patch[1, 1]) != bg:
                continue

            seq = [int(patch[dr, dc]) for dr, dc in RING_POS]
            if seq.count(bg) != 1 or seq.count(key_color) != 1:
                continue
            if any(value == bg for idx, value in enumerate(seq) if idx != seq.index(bg)):
                continue

            coords = [(r0 + dr, c0 + dc) for dr, dc in RING_POS]
            assert not any(occupied[r, c] for r, c in coords), "rings must not overlap"
            for r, c in coords:
                occupied[r, c] = True

            gap_idx = seq.index(bg)
            key_idx = seq.index(key_color)
            steps = (gap_idx - key_idx) % 8
            assert steps != 0, "gap and key positions must differ"
            rings.append(
                {
                    "r0": int(r0),
                    "c0": int(c0),
                    "seq": seq,
                    "gap_idx": int(gap_idx),
                    "key_idx": int(key_idx),
                    "steps": int(steps),
                }
            )

    assert rings, "expected at least one one-gap ring"

    expected_foreground = np.zeros_like(g, dtype=bool)
    for ring in rings:
        r0 = int(ring["r0"])
        c0 = int(ring["c0"])
        for dr, dc in RING_POS:
            rr = r0 + dr
            cc = c0 + dc
            if int(g[rr, cc]) != bg:
                expected_foreground[rr, cc] = True

    inner_non_bg = g != bg
    inner_non_bg[0, :] = False
    inner_non_bg[-1, :] = False
    inner_non_bg[:, 0] = False
    inner_non_bg[:, -1] = False
    assert np.array_equal(inner_non_bg, expected_foreground), "all interior non-background cells must belong to rings"

    return key_color, bg, rings


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each 3x3 one-gap color ring clockwise until its border-colored key lands in the original gap.

    Input structure:
    - The outermost canvas border is a solid single color that acts as a global key color.
    - Inside the border, the background is uniform except for several 3x3 ring objects.
    - Each ring has a background center, exactly one background gap on its perimeter,
      and exactly one perimeter pixel whose color matches the outer border.

    Rule:
    - Read each ring's eight perimeter positions as one clockwise cycle, including the gap.
    - Rotate that whole cycle clockwise by the unique number of steps that moves the
      border-colored ring pixel onto the gap's original position.
    - Move the other six colors and the gap along with the same rotation.
    - Keep the ring centers, the outer border, and every other background cell unchanged.

    Key insight:
    - The border supplies a global key color, while each ring's local gap supplies the target slot;
      their relative clockwise offset determines that ring's rotation amount.
    """
    g = np.asarray(grid, dtype=int)
    key_color, _bg, rings = _extract_rings(g)
    out = g.copy()

    for ring in rings:
        r0 = int(ring["r0"])
        c0 = int(ring["c0"])
        seq = list(ring["seq"])
        gap_idx = int(ring["gap_idx"])
        steps = int(ring["steps"])
        rotated = _rotate_clockwise(seq, steps)
        assert rotated[gap_idx] == key_color
        for idx, (dr, dc) in enumerate(RING_POS):
            out[r0 + dr, c0 + dc] = rotated[idx]

    return out


def _bbox_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _ring_patch(bg: int, key_color: int, gap_idx: int, steps: int) -> np.ndarray:
    key_idx = (gap_idx - steps) % 8
    palette = [c for c in Color.ALL_COLORS if c not in {bg, key_color}]
    extras = random.sample(palette, 6)
    remaining_idxs = [idx for idx in range(8) if idx not in {gap_idx, key_idx}]
    random.shuffle(remaining_idxs)

    seq = [bg] * 8
    seq[gap_idx] = bg
    seq[key_idx] = key_color
    for idx, color in zip(remaining_idxs, extras):
        seq[idx] = int(color)

    patch = np.full((3, 3), bg, dtype=int)
    for idx, (dr, dc) in enumerate(RING_POS):
        patch[dr, dc] = int(seq[idx])
    return patch


def _choose_specs(count: int) -> list[tuple[int, int]]:
    specs: list[tuple[int, int]] = []
    available_gaps = list(range(8))
    random.shuffle(available_gaps)
    available_steps = list(range(1, 8))
    random.shuffle(available_steps)

    if count >= 2:
        specs.append((random.choice(CORNER_IDXS), random.choice((1, 2, 3))))
        specs.append((random.choice(EDGE_IDXS), random.choice((4, 5, 6, 7))))

    while len(specs) < count:
        if available_gaps:
            gap_idx = available_gaps.pop()
        else:
            gap_idx = random.randrange(8)
        if available_steps:
            steps = available_steps.pop()
        else:
            steps = random.randint(1, 7)
        specs.append((gap_idx, steps))

    random.shuffle(specs)
    return specs[:count]


def _place_positions(h: int, w: int, count: int, border_margin: int, gap: int) -> list[tuple[int, int]] | None:
    start_r = 1 + border_margin
    end_r = h - 4 - border_margin
    start_c = 1 + border_margin
    end_c = w - 4 - border_margin
    if start_r > end_r or start_c > end_c:
        return None

    candidates = [(r, c) for r in range(start_r, end_r + 1) for c in range(start_c, end_c + 1)]
    random.shuffle(candidates)
    boxes = [(r, c, r + 2, c + 2) for r, c in candidates]
    chosen: list[tuple[int, int]] = []
    chosen_boxes: list[tuple[int, int, int, int]] = []

    def search(idx: int) -> bool:
        if len(chosen) == count:
            return True
        if idx >= len(candidates):
            return False
        remaining = count - len(chosen)
        if len(candidates) - idx < remaining:
            return False

        for j in range(idx, len(candidates)):
            box = boxes[j]
            if all(_bbox_separated(box, other, gap=gap) for other in chosen_boxes):
                chosen.append(candidates[j])
                chosen_boxes.append(box)
                if search(j + 1):
                    return True
                chosen.pop()
                chosen_boxes.pop()
        return False

    if not search(0):
        return None
    return chosen


def _semantic_check(inp: np.ndarray, out: np.ndarray) -> bool:
    try:
        key_color, bg, rings = _extract_rings(inp)
    except AssertionError:
        return False

    if len(rings) < 2:
        return False

    gap_kinds = {int(ring["gap_idx"]) % 2 for ring in rings}
    if len(gap_kinds) < min(2, len(rings)):
        return False

    steps = [int(ring["steps"]) for ring in rings]
    if len(set(steps)) < min(2, len(steps)):
        return False

    if not np.array_equal(inp[0, :], out[0, :]) or not np.array_equal(inp[-1, :], out[-1, :]):
        return False
    if not np.array_equal(inp[:, 0], out[:, 0]) or not np.array_equal(inp[:, -1], out[:, -1]):
        return False

    for ring in rings:
        r0 = int(ring["r0"])
        c0 = int(ring["c0"])
        seq_in = [int(inp[r0 + dr, c0 + dc]) for dr, dc in RING_POS]
        seq_out = [int(out[r0 + dr, c0 + dc]) for dr, dc in RING_POS]
        rotated = _rotate_clockwise(seq_in, int(ring["steps"]))
        if seq_out != rotated:
            return False
        if seq_out[int(ring["gap_idx"])] != key_color:
            return False
        if out[r0 + 1, c0 + 1] != bg:
            return False

    return not np.array_equal(inp, out)


def _gen_attempt() -> np.ndarray | None:
    key_color, bg = random.sample(Color.ALL_COLORS, 2)
    count = random.randint(2, 5)
    border_margin = random.randint(0, 2)
    ring_gap = random.randint(1, 2)

    h = random.randint(max(7, 7 + border_margin * 2), 24)
    w = random.randint(max(7, 7 + border_margin * 2), 24)
    positions = _place_positions(h, w, count, border_margin, ring_gap)
    if positions is None:
        return None

    grid = np.full((h, w), bg, dtype=int)
    grid[0, :] = key_color
    grid[-1, :] = key_color
    grid[:, 0] = key_color
    grid[:, -1] = key_color

    specs = _choose_specs(count)
    for (r0, c0), (gap_idx, steps) in zip(positions, specs):
        paste(grid, _ring_patch(bg, key_color, gap_idx, steps), r0, c0)

    out = main(grid)
    return grid if _semantic_check(grid, out) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle-6 input")
