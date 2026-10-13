from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
FRAME_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_H_RANGE = (18, 26)
GRID_W_RANGE = (18, 26)
FRAME_COUNT_CHOICES = (2, 2, 3)
INTERIOR_H_RANGE = (4, 9)
INTERIOR_W_RANGE = (5, 11)
FRAME_GAP = 2
OUTSIDE_DOODLE_PROB = 0.65
GENERATION_ATTEMPTS = 500

SLOPES = (-1, 1)
PATTERN_MODES = (
    "full_single",
    "partial_single",
    "double_stacked",
    "double_parallel",
    "double_parallel",
)


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) != color:
                continue
            queue: deque[tuple[int, int]] = deque([(r, c)])
            visited[r, c] = True
            comp: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            components.append(comp)
    return components


def _frame_from_component(coords: list[tuple[int, int]], color: int) -> tuple[int, int, int, int, int] | None:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    border = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }
    if set(coords) != border:
        return None
    return color, r0, c0, r1, c1


def _detect_frames(grid: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    frames: list[tuple[int, int, int, int, int]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in (BLACK, GREY)):
        for comp in _components_of_color(grid, color):
            frame = _frame_from_component(comp, color)
            if frame is not None:
                frames.append(frame)
    frames.sort(key=lambda fr: ((fr[3] - fr[1] - 1) * (fr[4] - fr[2] - 1), fr[1], fr[2]))
    return frames


def _apply_fill_rule(grid: np.ndarray, frames: list[tuple[int, int, int, int, int]]) -> np.ndarray:
    out = grid.copy()
    for color, r0, c0, r1, c1 in frames:
        for r in range(r0 + 1, r1):
            cols = np.flatnonzero(grid[r, c0 + 1 : c1] == BLACK) + c0 + 1
            for left, right in zip(cols[0::2], cols[1::2]):
                if right - left <= 1:
                    continue
                segment = out[r, left + 1 : right]
                segment[segment == GREY] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each frame interior row-by-row between paired black diagonal contour pixels using the frame's border color.

    Input structure:
    - The grid has a grey background with two or more complete one-pixel-thick rectangular frames in distinct colors.
    - Each frame encloses a grey interior containing black diagonal contour pixels.
    - Black pixels outside frames may also exist and must remain untouched.

    Rule:
    - Detect every complete monochrome rectangular frame.
    - For each frame, work only inside its strict interior; if frames are nested, process smaller interiors first.
    - In each interior row, read black contour pixels from left to right and pair them as (1st,2nd), (3rd,4th), and so on.
    - Recolor only the grey pixels strictly between each pair to the frame's border color, while preserving the border and all black pixels.

    Key insight:
    - The black row endpoints define where scanline filling happens, but the fill color is not fixed globally: each enclosing frame supplies the color for its own local fill.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames = _detect_frames(g)
    assert frames, "expected at least one complete rectangular frame"
    return _apply_fill_rule(g, frames)


def _pair_options(length: int, width: int, lo: int = 0, hi: int | None = None) -> list[tuple[list[int], list[int]]]:
    if hi is None:
        hi = width - 1
    options: list[tuple[list[int], list[int]]] = []
    for left_slope in SLOPES:
        if left_slope == 1:
            left_starts = range(0, width - length + 1)
        else:
            left_starts = range(length - 1, width)
        for right_slope in SLOPES:
            if right_slope == 1:
                right_starts = range(0, width - length + 1)
            else:
                right_starts = range(length - 1, width)
            for left_start in left_starts:
                left_cols = [left_start + left_slope * t for t in range(length)]
                if min(left_cols) < lo or max(left_cols) > hi:
                    continue
                for right_start in right_starts:
                    right_cols = [right_start + right_slope * t for t in range(length)]
                    if min(right_cols) < lo or max(right_cols) > hi:
                        continue
                    if all(right - left >= 2 for left, right in zip(left_cols, right_cols)):
                        options.append((left_cols, right_cols))
    return options


def _pattern_fill_count(mask: np.ndarray) -> int:
    fill = 0
    for r in range(mask.shape[0]):
        cols = np.flatnonzero(mask[r])
        for left, right in zip(cols[0::2], cols[1::2]):
            fill += max(0, int(right - left - 1))
    return fill


def _pattern_is_valid(mask: np.ndarray) -> bool:
    row_counts = np.count_nonzero(mask, axis=1)
    if any(count % 2 == 1 for count in row_counts):
        return False
    return _pattern_fill_count(mask) > 0


def _draw_pair(mask: np.ndarray, row0: int, left_cols: list[int], right_cols: list[int]) -> None:
    for offset, (left, right) in enumerate(zip(left_cols, right_cols)):
        mask[row0 + offset, left] = True
        mask[row0 + offset, right] = True


def _gen_pattern(h: int, w: int, preferred_mode: str | None = None) -> np.ndarray | None:
    unique_modes = []
    for mode in PATTERN_MODES:
        if mode not in unique_modes:
            unique_modes.append(mode)
    if preferred_mode is None:
        preferred_mode = random.choice(PATTERN_MODES)
    modes = [preferred_mode] + [mode for mode in unique_modes if mode != preferred_mode]
    for mode in modes:
        mask = np.zeros((h, w), dtype=bool)
        if mode == "full_single":
            options = _pair_options(h, w)
            if not options:
                continue
            _draw_pair(mask, 0, *random.choice(options))
        elif mode == "partial_single":
            length = random.randint(3, h)
            row0 = random.randint(0, h - length)
            options = _pair_options(length, w)
            if not options:
                continue
            _draw_pair(mask, row0, *random.choice(options))
        elif mode == "double_stacked":
            if h < 6:
                continue
            split = random.randint(3, h - 3)
            top_options = _pair_options(split, w)
            bottom_options = _pair_options(h - split, w)
            if not top_options or not bottom_options:
                continue
            _draw_pair(mask, 0, *random.choice(top_options))
            _draw_pair(mask, split, *random.choice(bottom_options))
        else:
            if w < 8:
                continue
            length = random.randint(3, h)
            row0 = random.randint(0, h - length)
            split = random.randint(3, w - 4)
            left_options = _pair_options(length, w, lo=0, hi=split - 1)
            right_options = _pair_options(length, w, lo=split + 1, hi=w - 1)
            if not left_options or not right_options:
                continue
            _draw_pair(mask, row0, *random.choice(left_options))
            _draw_pair(mask, row0, *random.choice(right_options))
        if _pattern_is_valid(mask):
            return mask
    return None


def _render_frame(interior_mask: np.ndarray, color: int) -> np.ndarray:
    h, w = interior_mask.shape
    patch = np.full((h + 2, w + 2), GREY, dtype=int)
    patch[1:-1, 1:-1][interior_mask] = BLACK
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    return patch


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = FRAME_GAP) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _make_frame_spec(preferred_mode: str | None = None) -> dict[str, object] | None:
    h = random.randint(*INTERIOR_H_RANGE)
    width_lo = INTERIOR_W_RANGE[0]
    if preferred_mode == "double_parallel":
        width_lo = max(width_lo, 8)
    w = random.randint(width_lo, INTERIOR_W_RANGE[1])
    mask = _gen_pattern(h, w, preferred_mode=preferred_mode)
    if mask is None:
        return None
    return {"shape": (h, w), "mask": mask}


def _place_patches(grid: np.ndarray, patches: list[np.ndarray]) -> list[tuple[int, int, int, int]] | None:
    h, w = grid.shape
    bboxes: list[tuple[int, int, int, int]] = []
    for patch in patches:
        ph, pw = patch.shape
        candidates = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
        random.shuffle(candidates)
        placed = False
        for r0, c0 in candidates:
            bbox = (r0, c0, r0 + ph - 1, c0 + pw - 1)
            if all(_boxes_separated(bbox, other) for other in bboxes):
                grid[r0 : r0 + ph, c0 : c0 + pw] = patch
                bboxes.append(bbox)
                placed = True
                break
        if not placed:
            return None
    return bboxes


def _add_outside_doodles(grid: np.ndarray, frame_bboxes: list[tuple[int, int, int, int]]) -> None:
    if random.random() >= OUTSIDE_DOODLE_PROB:
        return
    h, w = grid.shape
    blocked = np.zeros((h, w), dtype=bool)
    for r0, c0, r1, c1 in frame_bboxes:
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(h - 1, r1 + 1)
        cc1 = min(w - 1, c1 + 1)
        blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True

    for _ in range(random.randint(1, 3)):
        for _ in range(50):
            length = random.randint(2, 4)
            slope = random.choice(SLOPES)
            row0 = random.randint(0, h - length)
            if slope == 1:
                col0 = random.randint(0, w - length)
            else:
                col0 = random.randint(length - 1, w - 1)
            cells = [(row0 + t, col0 + slope * t) for t in range(length)]
            if any(blocked[r, c] or int(grid[r, c]) != GREY for r, c in cells):
                continue
            for r, c in cells:
                grid[r, c] = BLACK
                blocked[r, c] = True
            break


def _oracle_expected(grid: np.ndarray) -> np.ndarray | None:
    frames = _detect_frames(grid)
    if len(frames) not in (2, 3):
        return None
    colors = [frame[0] for frame in frames]
    if len(set(colors)) != len(colors):
        return None

    changed_any = False
    interior_union = np.zeros(grid.shape, dtype=bool)
    for _color, r0, c0, r1, c1 in frames:
        interior_union[r0 + 1 : r1, c0 + 1 : c1] = True
        counts = [int(np.count_nonzero(grid[r, c0 + 1 : c1] == BLACK)) for r in range(r0 + 1, r1)]
        if any(count % 2 == 1 for count in counts):
            return None
        frame_fill = 0
        for r in range(r0 + 1, r1):
            cols = np.flatnonzero(grid[r, c0 + 1 : c1] == BLACK) + c0 + 1
            for left, right in zip(cols[0::2], cols[1::2]):
                frame_fill += max(0, int(right - left - 1))
        if frame_fill == 0:
            return None
        changed_any = True

    if not changed_any:
        return None

    out = _apply_fill_rule(grid, frames)
    changed = out != grid
    if not np.any(changed):
        return None
    if np.any(changed & ~interior_union):
        return None
    if np.any(grid[changed] != GREY):
        return None
    return out


def _gen_attempt() -> np.ndarray | None:
    grid = np.full(
        (random.randint(*GRID_H_RANGE), random.randint(*GRID_W_RANGE)),
        GREY,
        dtype=int,
    )
    frame_count = random.choice(FRAME_COUNT_CHOICES)
    colors = random.sample(FRAME_COLORS, frame_count)

    specs: list[dict[str, object]] = []
    force_parallel = random.random() < 0.55
    same_pattern = frame_count >= 2 and random.random() < 0.4
    if same_pattern:
        base = _make_frame_spec(preferred_mode="double_parallel" if force_parallel else None)
        if base is None:
            return None
        specs.extend([base, {"shape": base["shape"], "mask": np.array(base["mask"], copy=True)}])
        force_parallel = False
    while len(specs) < frame_count:
        preferred_mode = "double_parallel" if force_parallel else None
        spec = _make_frame_spec(preferred_mode=preferred_mode)
        if spec is None:
            return None
        specs.append(spec)
        force_parallel = False

    patches = [_render_frame(np.asarray(spec["mask"], dtype=bool), color) for spec, color in zip(specs, colors)]
    frame_bboxes = _place_patches(grid, patches)
    if frame_bboxes is None:
        return None

    _add_outside_doodles(grid, frame_bboxes)

    expected = _oracle_expected(grid)
    if expected is None:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
