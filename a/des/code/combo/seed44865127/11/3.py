from __future__ import annotations

from collections import deque

import numpy as np


DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
OPEN_DIRECTIONS = ("up", "down", "left", "right")
GEN_RETRIES = 300


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _classify_frame(grid: np.ndarray, comp: list[tuple[int, int]], bg: int) -> tuple[int, tuple[int, int, int, int], str]:
    colors = {int(grid[r, c]) for r, c in comp}
    assert len(colors) == 1, "each frame component must be monochrome"
    color = next(iter(colors))
    assert color != bg, "frame color must differ from background"

    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    assert h >= 3 and w >= 3, "frame bbox must have room for an opening"

    local = np.zeros((h, w), dtype=bool)
    for r, c in comp:
        local[r - r0, c - c0] = True

    top_full = bool(local[0, :].all())
    bottom_full = bool(local[-1, :].all())
    left_full = bool(local[:, 0].all())
    right_full = bool(local[:, -1].all())

    valid: list[str] = []
    if local[0, 0] and local[0, -1] and not local[0, 1:-1].any():
        if bottom_full and left_full and right_full and int(local.sum()) == 2 * h + w - 2:
            valid.append("up")
    if local[-1, 0] and local[-1, -1] and not local[-1, 1:-1].any():
        if top_full and left_full and right_full and int(local.sum()) == 2 * h + w - 2:
            valid.append("down")
    if local[0, 0] and local[-1, 0] and not local[1:-1, 0].any():
        if top_full and bottom_full and right_full and int(local.sum()) == h + 2 * w - 2:
            valid.append("left")
    if local[0, -1] and local[-1, -1] and not local[1:-1, -1].any():
        if top_full and bottom_full and left_full and int(local.sum()) == h + 2 * w - 2:
            valid.append("right")

    assert len(valid) == 1, "expected a unique opening direction"
    return color, (r0, c0, r1, c1), valid[0]


def _paint_interior(out: np.ndarray, bbox: tuple[int, int, int, int], direction: str, color: int) -> None:
    r0, c0, r1, c1 = bbox

    if direction == "up":
        for offset, r in enumerate(range(r0, r1)):
            if offset % 2 == 0:
                out[r, c0 + 1 : c1] = color
        return

    if direction == "down":
        for offset, c in enumerate(range(c0 + 1, c1)):
            if offset % 2 == 0:
                out[r0 + 1 : r1 + 1, c] = color
        return

    if direction == "left":
        out[r0 + 1 : r1, c0:c1] = color
        return

    assert direction == "right"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each three-sided rectangular frame according to its opening direction.

    Opening up    -> horizontal stripes
    Opening down  -> vertical stripes
    Opening left  -> solid fill
    Opening right -> leave the interior empty
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    comps = _components(g != bg)
    assert comps, "expected at least one frame"

    out = g.copy()
    for comp in comps:
        color, bbox, direction = _classify_frame(g, comp, bg)
        _paint_interior(out, bbox, direction, color)
    return out


def _frame_patch(height: int, width: int, direction: str, bg: int, fg: int) -> np.ndarray:
    patch = np.full((height, width), bg, dtype=int)
    if direction == "up":
        patch[:, 0] = fg
        patch[:, -1] = fg
        patch[-1, :] = fg
    elif direction == "down":
        patch[:, 0] = fg
        patch[:, -1] = fg
        patch[0, :] = fg
    elif direction == "left":
        patch[0, :] = fg
        patch[-1, :] = fg
        patch[:, -1] = fg
    else:
        assert direction == "right"
        patch[0, :] = fg
        patch[-1, :] = fg
        patch[:, 0] = fg
    return patch


def _layout_for_count(rng: np.random.Generator, n_frames: int) -> tuple[int, int]:
    candidates = [(rows, cols) for rows, cols in ((2, 2), (2, 3), (3, 2)) if rows * cols >= n_frames]
    idx = int(rng.integers(len(candidates)))
    return candidates[idx]


def _direction_list(rng: np.random.Generator, n_frames: int) -> list[str]:
    dirs = list(OPEN_DIRECTIONS)
    rng.shuffle(dirs)
    while len(dirs) < n_frames:
        dirs.append(OPEN_DIRECTIONS[int(rng.integers(len(OPEN_DIRECTIONS)))])
    rng.shuffle(dirs)
    return dirs[:n_frames]


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_frames = int(rng.integers(4, 7))
    rows, cols = _layout_for_count(rng, n_frames)

    slot_h = int(rng.integers(5, 8 if rows == 3 else 10))
    slot_w = int(rng.integers(5, 8 if cols == 3 else 10))
    margin = int(rng.integers(1, 3))
    gap = int(rng.integers(1, 4))

    height = margin * 2 + rows * slot_h + (rows - 1) * gap
    width = margin * 2 + cols * slot_w + (cols - 1) * gap
    if height > 30 or width > 30:
        return None

    palette = rng.choice(np.arange(10), size=2, replace=False)
    bg, fg = int(palette[0]), int(palette[1])
    grid = np.full((height, width), bg, dtype=int)

    slot_origins = [(rr, cc) for rr in range(rows) for cc in range(cols)]
    perm = rng.permutation(len(slot_origins))
    chosen_slots = [slot_origins[int(i)] for i in perm[:n_frames]]
    directions = _direction_list(rng, n_frames)

    for (slot_r, slot_c), direction in zip(chosen_slots, directions):
        frame_h = int(rng.integers(4, slot_h + 1))
        frame_w = int(rng.integers(4, slot_w + 1))
        base_r = margin + slot_r * (slot_h + gap)
        base_c = margin + slot_c * (slot_w + gap)
        jitter_r = int(rng.integers(0, slot_h - frame_h + 1))
        jitter_c = int(rng.integers(0, slot_w - frame_w + 1))
        r0 = base_r + jitter_r
        c0 = base_c + jitter_c
        patch = _frame_patch(frame_h, frame_w, direction, bg, fg)
        grid[r0 : r0 + frame_h, c0 : c0 + frame_w] = patch

    # Validate that the deterministic solver accepts the generated structure.
    main(grid)
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
