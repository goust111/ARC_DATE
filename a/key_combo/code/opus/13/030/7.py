from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6

FRAME_FILL = {
    0: BLACK,
    1: RED,
    2: YELLOW,
}


Frame = tuple[int, int, int, int]  # r0, c0, h, w


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


def _find_seed_bar(grid: np.ndarray) -> Frame:
    coords = [tuple(map(int, p)) for p in np.argwhere(grid == GREEN)]
    assert coords, "expected one green seed bar"
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    assert h != w, "seed bar must be non-square"
    assert len(coords) == h * w, "seed bar must be solid"
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == GREEN), "seed bar bbox must be solid green"
    return r0, c0, h, w


def _find_frames(grid: np.ndarray) -> list[Frame]:
    frames: list[Frame] = []
    for comp in _components(grid == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h >= 3 and w >= 3, "frame must have an interior"

        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True

        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(patch[border] == GREY), "frame border must be grey"
        assert np.all(patch[1:-1, 1:-1] == BLACK), "frame interior must be black in the input"
        assert len(comp) == int(border.sum()), "grey pixels must form a single-pixel rectangular frame"
        frames.append((r0, c0, h, w))

    assert 3 <= len(frames) <= 5, "expected three to five hollow frames"
    return sorted(frames)


def _stamp_rect(out: np.ndarray, mask: np.ndarray, color: int, r0: int, c0: int, h: int, w: int) -> None:
    rr0 = max(0, int(r0))
    cc0 = max(0, int(c0))
    rr1 = min(out.shape[0], int(r0) + int(h))
    cc1 = min(out.shape[1], int(c0) + int(w))
    if rr0 >= rr1 or cc0 >= cc1:
        return
    view = out[rr0:rr1, cc0:cc1]
    conflict = (view != BLACK) & (view != color)
    assert not bool(conflict.any()), "zigzag bars must not conflict"
    view[:] = color
    mask[rr0:rr1, cc0:cc1] = True


def _build_chain(shape: tuple[int, int], seed: Frame) -> tuple[np.ndarray, np.ndarray]:
    r0, c0, h, w = seed
    chain = np.zeros(shape, dtype=int)
    mask = np.zeros(shape, dtype=bool)
    period = h + w
    limit = shape[0] + shape[1] + period + 2

    for n in range(-limit, limit + 1):
        offset = n * period
        _stamp_rect(chain, mask, GREEN, r0 + offset, c0 + offset, h, w)
        _stamp_rect(chain, mask, GREEN, r0 + offset, c0 - offset, h, w)
        _stamp_rect(chain, mask, PINK, r0 + h + offset, c0 + w + offset, w, h)
        _stamp_rect(chain, mask, PINK, r0 + h + offset, c0 - h - offset, w, h)

    return chain, mask


def _frame_border(frame: Frame) -> set[tuple[int, int]]:
    r0, c0, h, w = frame
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    border = {(r0, c) for c in range(c0, c1 + 1)}
    border.update((r1, c) for c in range(c0, c1 + 1))
    border.update((r, c0) for r in range(r0, r1 + 1))
    border.update((r, c1) for r in range(r0, r1 + 1))
    return border


def _count_border_segments(chain_mask: np.ndarray, frame: Frame) -> int:
    covered = {cell for cell in _frame_border(frame) if chain_mask[cell]}
    segments = 0
    while covered:
        segments += 1
        start = covered.pop()
        stack = [start]
        while stack:
            r, c = stack.pop()
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in covered:
                    covered.remove((nr, nc))
                    stack.append((nr, nc))
    return segments


def _fill_color(segment_count: int) -> int:
    if segment_count >= 3:
        return BLUE
    return FRAME_FILL[segment_count]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each hollow grey frame according to how many zigzag-chain segments cover its border.

    Input structure:
    - A black background contains one solid green non-square rectangular bar and three to five grey one-pixel-thick hollow rectangles.
    - The green bar is the only solid seed object; each grey rectangle encloses a black interior.
    - The seed and frames are separated in the input, and the chain itself is not yet drawn.

    Rule:
    - Use the green seed bar to generate two diagonal zigzag chains: copies of the seed recur every seed-height-plus-seed-width cells on both 45-degree diagonals, and the gaps between consecutive seed copies are bridged by 90-degree-rotated pink copies.
    - For each grey frame, inspect only its border pixels and count how many 4-connected covered segments are touched by the zigzag chain, treating green and pink coverage identically.
    - Stamp the zigzag chain onto the scene, then fill each frame interior with black for 0 segments, red for 1 segment, yellow for 2 segments, and blue for 3 or more segments.

    Key insight:
    - The zigzag bars are not the answer by themselves; they act as a global measuring tool, and the local frame fill color is determined by the number of separate chain-overlap runs along that frame's border.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    seed = _find_seed_bar(g)
    frames = _find_frames(g)
    chain, chain_mask = _build_chain(g.shape, seed)

    out = g.copy()
    out[chain_mask] = chain[chain_mask]
    for frame in frames:
        r0, c0, h, w = frame
        fill = _fill_color(_count_border_segments(chain_mask, frame))
        out[r0 + 1 : r0 + h - 1, c0 + 1 : c0 + w - 1] = fill
    return out


GRID_ROWS = (18, 24)
GRID_COLS = (18, 24)
SEED_DIMS = (
    (1, 4),
    (1, 5),
    (2, 4),
    (2, 5),
    (3, 4),
    (3, 5),
)
SEED_CENTER_JITTER = 2
FRAME_SIDE_RANGE = (4, 9)
FRAME_GAP = 1
GENERATION_RETRIES = 300


def _boxes_too_close(a: Frame, b: Frame, gap: int = FRAME_GAP) -> bool:
    ar0, ac0, ah, aw = a
    br0, bc0, bh, bw = b
    ar1 = ar0 + ah - 1
    ac1 = ac0 + aw - 1
    br1 = br0 + bh - 1
    bc1 = bc0 + bw - 1
    return not (ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0)


def _sample_seed(shape: tuple[int, int]) -> Frame | None:
    h, w = random.choice(SEED_DIMS)
    if random.random() < 0.5:
        h, w = w, h
    if min(shape) / (h + w) < 3.1:
        return None

    center_r = shape[0] // 2 + random.randint(-SEED_CENTER_JITTER, SEED_CENTER_JITTER)
    center_c = shape[1] // 2 + random.randint(-SEED_CENTER_JITTER, SEED_CENTER_JITTER)
    r0 = max(0, min(shape[0] - h, center_r - h // 2))
    c0 = max(0, min(shape[1] - w, center_c - w // 2))
    return r0, c0, h, w


def _enumerate_candidates(shape: tuple[int, int], seed: Frame, chain_mask: np.ndarray) -> dict[int, list[Frame]]:
    rows, cols = shape
    candidates = {k: [] for k in range(5)}
    for h in range(FRAME_SIDE_RANGE[0], FRAME_SIDE_RANGE[1] + 1):
        for w in range(FRAME_SIDE_RANGE[0], FRAME_SIDE_RANGE[1] + 1):
            for r0 in range(rows - h + 1):
                for c0 in range(cols - w + 1):
                    frame = (r0, c0, h, w)
                    if _boxes_too_close(frame, seed):
                        continue
                    segments = _count_border_segments(chain_mask, frame)
                    if segments in candidates:
                        candidates[segments].append(frame)
    return candidates


def _target_options(candidates: dict[int, list[Frame]]) -> list[list[int]]:
    options: list[list[int]] = []
    subsets = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
    random.shuffle(subsets)

    if candidates[4] and random.random() < 0.15:
        options.append([0, 1, 2, 3, 4])

    if random.random() < 0.75:
        options.append([0, 1, 2, 3])
        options.extend(subsets)
    else:
        options.append(subsets[0])
        options.append([0, 1, 2, 3])
        options.extend(subsets[1:])

    if candidates[4] and [0, 1, 2, 3, 4] not in options:
        options.append([0, 1, 2, 3, 4])
    return options


def _select_frames(candidates: dict[int, list[Frame]]) -> tuple[list[Frame], list[int]] | None:
    for targets in _target_options(candidates):
        if any(not candidates[t] for t in targets):
            continue

        order = sorted(targets, key=lambda t: len(candidates[t]))
        shuffled = {t: random.sample(candidates[t], len(candidates[t])) for t in order}
        chosen: list[Frame] = []
        used_sizes: set[tuple[int, int]] = set()

        def backtrack(i: int) -> bool:
            if i == len(order):
                return True
            target = order[i]
            for frame in shuffled[target][:200]:
                size = (frame[2], frame[3])
                if size in used_sizes:
                    continue
                if any(_boxes_too_close(frame, other) for other in chosen):
                    continue
                chosen.append(frame)
                used_sizes.add(size)
                if backtrack(i + 1):
                    return True
                chosen.pop()
                used_sizes.remove(size)
            return False

        if backtrack(0):
            return sorted(chosen), targets
    return None


def _render_input(shape: tuple[int, int], seed: Frame, frames: list[Frame]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    r0, c0, h, w = seed
    grid[r0 : r0 + h, c0 : c0 + w] = GREEN
    for fr0, fc0, fh, fw in frames:
        grid[fr0, fc0 : fc0 + fw] = GREY
        grid[fr0 + fh - 1, fc0 : fc0 + fw] = GREY
        grid[fr0 : fr0 + fh, fc0] = GREY
        grid[fr0 : fr0 + fh, fc0 + fw - 1] = GREY
    return grid


def _gen_attempt() -> np.ndarray | None:
    shape = (
        random.randint(GRID_ROWS[0], GRID_ROWS[1]),
        random.randint(GRID_COLS[0], GRID_COLS[1]),
    )
    seed = _sample_seed(shape)
    if seed is None:
        return None

    try:
        _chain, chain_mask = _build_chain(shape, seed)
    except AssertionError:
        return None

    candidates = _enumerate_candidates(shape, seed, chain_mask)
    picked = _select_frames(candidates)
    if picked is None:
        return None
    frames, targets = picked

    grid = _render_input(shape, seed, frames)
    try:
        out = main(grid)
    except AssertionError:
        return None

    actual_segments = [_count_border_segments(chain_mask, frame) for frame in frames]
    if sorted(actual_segments) != sorted(targets):
        return None
    if len(set(actual_segments)) != len(actual_segments):
        return None
    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
