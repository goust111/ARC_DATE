from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
SIDES = ("top", "bottom", "left", "right")
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
GEN_RETRIES = 400
GRID_SIDE_RANGE = (18, 30)
FRAME_COUNT_CHOICES = [1, 2, 3]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(int(r), int(c))])
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _false_runs(arr: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, value in enumerate(arr.tolist()):
        if not value:
            if start is None:
                start = int(i)
        elif start is not None:
            runs.append((start, int(i - start)))
            start = None
    if start is not None:
        runs.append((start, int(len(arr) - start)))
    return runs


def _frame_gap_side(mask: np.ndarray) -> str | None:
    h, w = mask.shape
    if h < 4 or w < 4:
        return None
    if np.any(mask[1:-1, 1:-1]):
        return None
    if not (mask[0, 0] and mask[0, -1] and mask[-1, 0] and mask[-1, -1]):
        return None

    side_slices = {
        "top": mask[0, 1:-1],
        "bottom": mask[-1, 1:-1],
        "left": mask[1:-1, 0],
        "right": mask[1:-1, -1],
    }

    gap_side: str | None = None
    for side, arr in side_slices.items():
        runs = _false_runs(arr)
        if not runs:
            if not bool(arr.all()):
                return None
            continue
        if len(runs) != 1:
            return None
        _start, length = runs[0]
        if length not in (2, 3):
            return None
        if gap_side is not None:
            return None
        gap_side = side
    return gap_side


def _find_blocks_and_frames(
    grid: np.ndarray,
) -> tuple[list[tuple[int, int, int, int, int]], list[tuple[int, int, int, int, str]]]:
    blocks: list[tuple[int, int, int, int, int]] = []
    frames: list[tuple[int, int, int, int, str]] = []

    for color in [int(v) for v in np.unique(grid) if int(v) != BG]:
        for comp in _components(grid == color):
            r0, c0, r1, c1 = _bbox(comp)
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            if h == 2 and w == 2 and len(comp) == 4:
                blocks.append((r0, c0, r1, c1, color))
                continue

            mask = np.zeros((h, w), dtype=bool)
            for r, c in comp:
                mask[r - r0, c - c0] = True
            gap_side = _frame_gap_side(mask)
            if gap_side is not None:
                frames.append((r0, c0, r1, c1, gap_side))

    return blocks, frames


def _side_block_color(
    frame: tuple[int, int, int, int, str],
    blocks: list[tuple[int, int, int, int, int]],
) -> int:
    r0, c0, r1, c1, gap_side = frame
    matches: list[int] = []
    for br0, bc0, br1, bc1, color in blocks:
        if gap_side == "top" and br1 == r0 - 1 and br0 == r0 - 2 and c0 <= bc0 and bc1 <= c1:
            matches.append(color)
        elif gap_side == "bottom" and br0 == r1 + 1 and br1 == r1 + 2 and c0 <= bc0 and bc1 <= c1:
            matches.append(color)
        elif gap_side == "left" and bc1 == c0 - 1 and bc0 == c0 - 2 and r0 <= br0 and br1 <= r1:
            matches.append(color)
        elif gap_side == "right" and bc0 == c1 + 1 and bc1 == c1 + 2 and r0 <= br0 and br1 <= r1:
            matches.append(color)

    assert len(matches) == 1, "gap side must have exactly one adjacent 2x2 block"
    return int(matches[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each hollow frame interior with the color of the outside block indicated by its gap direction.

    Input structure:
    - The background is black.
    - The grid contains one-pixel-thick hollow rectangular frames, each drawn in one non-background color.
    - Every frame has exactly one gap of length 2-3 on one side, and 2x2 monochrome blocks may be attached just outside its sides.
    - The frame interior is black, and the gap side always has one adjacent outside 2x2 block.

    Rule:
    - Detect each rectangular frame and identify which side contains the gap.
    - On that same outside side, find the adjacent 2x2 block and read its color.
    - Fill all black pixels inside the frame with that color.
    - Keep the border, the gap, and every outside block unchanged.

    Key insight:
    - The missing segment is not repaired; it acts as a directional pointer that selects which neighboring side block provides the fill color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    blocks, frames = _find_blocks_and_frames(g)
    assert frames, "expected at least one gapped rectangular frame"

    for frame in sorted(frames):
        r0, c0, r1, c1, _gap_side = frame
        fill_color = _side_block_color(frame, blocks)
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[interior == BG] = fill_color

    return out


def _draw_frame(patch: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    patch[r0, c0 : c0 + w] = color
    patch[r0 + h - 1, c0 : c0 + w] = color
    patch[r0 : r0 + h, c0] = color
    patch[r0 : r0 + h, c0 + w - 1] = color


def _make_gap(h: int, w: int, side: str) -> tuple[str, int, int]:
    span = w if side in ("top", "bottom") else h
    inner_span = span - 2
    gap_len = random.choice([k for k in (2, 3) if k <= inner_span])
    start = random.randint(1, span - gap_len - 1)
    return side, start, gap_len


def _sample_patch(frame_count: int) -> np.ndarray:
    size_hi = {1: 10, 2: 8, 3: 6}[frame_count]
    h = random.randint(4, size_hi)
    w = random.randint(4, size_hi)
    gap_side = random.choice(SIDES)
    _side, gap_start, gap_len = _make_gap(h, w, gap_side)

    block_count = random.randint(2, 4)
    other_sides = [side for side in SIDES if side != gap_side]
    chosen_sides = [gap_side] + random.sample(other_sides, block_count - 1)

    frame_color = random.choice(COLOR_POOL)
    block_colors = random.sample([c for c in COLOR_POOL if c != frame_color], len(chosen_sides))

    patch = np.full((h + 4, w + 4), BG, dtype=int)
    fr0 = 2
    fc0 = 2
    _draw_frame(patch, fr0, fc0, h, w, frame_color)

    if gap_side == "top":
        patch[fr0, fc0 + gap_start : fc0 + gap_start + gap_len] = BG
    elif gap_side == "bottom":
        patch[fr0 + h - 1, fc0 + gap_start : fc0 + gap_start + gap_len] = BG
    elif gap_side == "left":
        patch[fr0 + gap_start : fr0 + gap_start + gap_len, fc0] = BG
    else:
        patch[fr0 + gap_start : fr0 + gap_start + gap_len, fc0 + w - 1] = BG

    for side, color in zip(chosen_sides, block_colors, strict=False):
        if side in ("top", "bottom"):
            offset = random.randint(1, w - 3)
            rr = 0 if side == "top" else h + 2
            cc = fc0 + offset
            patch[rr : rr + 2, cc : cc + 2] = color
        else:
            offset = random.randint(1, h - 3)
            rr = fr0 + offset
            cc = 0 if side == "left" else w + 2
            patch[rr : rr + 2, cc : cc + 2] = color

    return patch


def _place_patches(h: int, w: int, patches: list[np.ndarray]) -> np.ndarray | None:
    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    for patch in sorted(patches, key=lambda x: x.shape[0] * x.shape[1], reverse=True):
        ph, pw = patch.shape
        candidates: list[tuple[int, int]] = []
        for r in range(h - ph + 1):
            for c in range(w - pw + 1):
                if not occupied[r : r + ph, c : c + pw].any():
                    candidates.append((int(r), int(c)))
        if not candidates:
            return None

        r0, c0 = random.choice(candidates)
        mask = patch != BG
        grid[r0 : r0 + ph, c0 : c0 + pw][mask] = patch[mask]
        occupied[r0 : r0 + ph, c0 : c0 + pw] = True

    return grid


def _attempt() -> np.ndarray | None:
    frame_count = random.choice(FRAME_COUNT_CHOICES)
    patches = [_sample_patch(frame_count) for _ in range(frame_count)]
    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)

    if max(p.shape[0] for p in patches) > h or max(p.shape[1] for p in patches) > w:
        return None

    grid = _place_patches(h, w, patches)
    if grid is None:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {GEN_RETRIES} attempts")
