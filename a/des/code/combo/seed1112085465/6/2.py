from __future__ import annotations

import math
from collections import Counter

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _rectangular_frame(
    grid: np.ndarray, color: int, pts: list[tuple[int, int]]
) -> tuple[int, int, int, int, int] | None:
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    h = r1 - r0 + 1
    w = c1 - c0 + 1

    if h < 3 or w < 3:
        return None
    if len(pts) != 2 * (h + w) - 4:
        return None

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    if not np.all(sub[border] == color):
        return None
    if np.any(sub[~border] == color):
        return None
    return (r0, c0, r1, c1, color)


def _find_frames(grid: np.ndarray, bg: int | None = None) -> list[tuple[int, int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g) if bg is None else int(bg)
    frames: list[tuple[int, int, int, int, int]] = []

    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue
        for pts in _components(g == color):
            frame = _rectangular_frame(g, color, pts)
            if frame is not None:
                frames.append(frame)

    frames.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4]))
    return frames


def _color_stats(interior: np.ndarray, bg: int, frame_color: int) -> list[tuple[int, int]]:
    counts: Counter[int] = Counter()
    first_seen: dict[int, int] = {}

    idx = 0
    for r in range(interior.shape[0]):
        for c in range(interior.shape[1]):
            color = int(interior[r, c])
            if color != bg and color != frame_color:
                counts[color] += 1
                if color not in first_seen:
                    first_seen[color] = idx
            idx += 1

    ordered = sorted(counts, key=lambda color: (-counts[color], first_seen[color]))
    return [(int(color), int(counts[color])) for color in ordered]


def _transform_frame(
    out: np.ndarray, frame: tuple[int, int, int, int, int], bg: int
) -> None:
    r0, c0, r1, c1, frame_color = frame
    interior = out[r0 + 1 : r1, c0 + 1 : c1]
    stats = _color_stats(interior, bg, frame_color)

    interior[:, :] = bg
    bottom = interior.shape[0]
    width = interior.shape[1]
    for color, count in stats:
        rows = math.ceil(count / width)
        assert rows <= bottom, "invalid frame contents for described rule"
        start = bottom - rows
        interior[start:bottom, :] = color
        bottom = start


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rearrange each rectangular container's interior into bottom-aligned color bands
    ordered by descending color frequency.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    out = g.copy()

    for frame in _find_frames(g, bg):
        _transform_frame(out, frame, bg)
    return out


# --------------- generator ---------------

_RNG = np.random.default_rng()

_BG = 8
_FRAME = 5
_INNER_COLORS = np.array([0, 1, 2, 3, 4, 6, 7, 9], dtype=int)

_FRAME_COUNT_RANGE = (2, 5)  # [lo, hi)
_INTERIOR_H_RANGE = (3, 8)   # [lo, hi)
_INTERIOR_W_RANGE = (3, 8)   # [lo, hi)
_MAX_COLORS_PER_FRAME = 4
_PLACEMENT_PAD = 1
_GEN_RETRIES = 400


def _random_partition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [int(total)]
    cuts = np.sort(_RNG.choice(np.arange(1, total), size=parts - 1, replace=False))
    values: list[int] = []
    prev = 0
    for cut in cuts:
        cut = int(cut)
        values.append(cut - prev)
        prev = cut
    values.append(total - prev)
    return values


def _sample_num_colors(max_colors: int) -> int:
    options = np.arange(1, max_colors + 1)
    weights = np.array([1.0, 3.0, 3.0, 2.0], dtype=float)[:max_colors]
    weights /= weights.sum()
    return int(_RNG.choice(options, p=weights))


def _sample_row_counts(width: int, height: int, n_colors: int) -> list[int] | None:
    for _ in range(100):
        if _RNG.random() < 0.5:
            used_rows = height
        else:
            used_rows = int(_RNG.integers(n_colors, height + 1))
        rows = sorted(_random_partition(used_rows, n_colors), reverse=True)
        multiplicities = Counter(rows)
        if max(multiplicities.values()) <= width:
            return [int(v) for v in rows]
    return None


def _counts_from_rows(width: int, row_counts: list[int]) -> list[int]:
    counts: list[int] = []
    i = 0
    while i < len(row_counts):
        row_count = int(row_counts[i])
        j = i
        while j < len(row_counts) and int(row_counts[j]) == row_count:
            j += 1
        multiplicity = j - i
        lo = (row_count - 1) * width + 1
        hi = row_count * width
        choices = _RNG.choice(np.arange(lo, hi + 1), size=multiplicity, replace=False)
        counts.extend(sorted((int(v) for v in choices), reverse=True))
        i = j

    assert all(a > b for a, b in zip(counts, counts[1:]))
    return counts


def _make_frame_patch() -> np.ndarray | None:
    inner_h = int(_RNG.integers(*_INTERIOR_H_RANGE))
    inner_w = int(_RNG.integers(*_INTERIOR_W_RANGE))
    max_colors = min(_MAX_COLORS_PER_FRAME, inner_h, len(_INNER_COLORS))

    for _ in range(100):
        n_colors = _sample_num_colors(max_colors)
        row_counts = _sample_row_counts(inner_w, inner_h, n_colors)
        if row_counts is None:
            continue
        counts = _counts_from_rows(inner_w, row_counts)
        total_pixels = sum(counts)
        if total_pixels > inner_h * inner_w:
            continue

        patch = np.full((inner_h + 2, inner_w + 2), _BG, dtype=int)
        patch[0, :] = _FRAME
        patch[-1, :] = _FRAME
        patch[:, 0] = _FRAME
        patch[:, -1] = _FRAME

        colors = [int(v) for v in _RNG.choice(_INNER_COLORS, size=n_colors, replace=False)]
        slots = _RNG.choice(inner_h * inner_w, size=total_pixels, replace=False)
        _RNG.shuffle(slots)

        cursor = 0
        for color, count in zip(colors, counts):
            for flat in slots[cursor : cursor + count]:
                flat = int(flat)
                rr = flat // inner_w
                cc = flat % inner_w
                patch[1 + rr, 1 + cc] = int(color)
            cursor += count

        if np.count_nonzero(patch[1:-1, 1:-1] != _BG) > 0:
            return patch
    return None


def _pick_canvas(patches: list[np.ndarray]) -> tuple[int, int] | None:
    max_h = max(int(p.shape[0]) for p in patches)
    max_w = max(int(p.shape[1]) for p in patches)
    area = sum(int(p.shape[0] * p.shape[1]) for p in patches)

    base = max(max_h + 2, max_w + 2, int(math.ceil(math.sqrt(area * _RNG.uniform(2.8, 3.4)))))
    if base > 30:
        return None

    min_h = max(max_h + 2, max(18, base - 2))
    min_w = max(max_w + 2, max(18, base - 2))
    max_h_choice = min(30, base + 4)
    max_w_choice = min(30, base + 4)
    if min_h > max_h_choice or min_w > max_w_choice:
        return None

    canvas_h = int(_RNG.integers(min_h, max_h_choice + 1))
    canvas_w = int(_RNG.integers(min_w, max_w_choice + 1))
    return canvas_h, canvas_w


def _placement_candidates(occupied: np.ndarray, h: int, w: int) -> list[tuple[int, int]]:
    canvas_h, canvas_w = occupied.shape
    candidates: list[tuple[int, int]] = []
    for r0 in range(1, canvas_h - h):
        for c0 in range(1, canvas_w - w):
            pr0 = max(0, r0 - _PLACEMENT_PAD)
            pc0 = max(0, c0 - _PLACEMENT_PAD)
            pr1 = min(canvas_h, r0 + h + _PLACEMENT_PAD)
            pc1 = min(canvas_w, c0 + w + _PLACEMENT_PAD)
            if not occupied[pr0:pr1, pc0:pc1].any():
                candidates.append((r0, c0))
    return candidates


def _place_patches(canvas_shape: tuple[int, int], patches: list[np.ndarray]) -> list[tuple[int, int, np.ndarray]] | None:
    occupied = np.zeros(canvas_shape, dtype=bool)
    placements: list[tuple[int, int, np.ndarray] | None] = [None] * len(patches)
    order = sorted(
        range(len(patches)),
        key=lambda idx: int(patches[idx].shape[0] * patches[idx].shape[1]),
        reverse=True,
    )

    for idx in order:
        patch = patches[idx]
        h, w = map(int, patch.shape)
        candidates = _placement_candidates(occupied, h, w)
        if not candidates:
            return None
        r0, c0 = candidates[int(_RNG.integers(0, len(candidates)))]
        occupied[r0 : r0 + h, c0 : c0 + w] = True
        placements[idx] = (r0, c0, patch)

    return [item for item in placements if item is not None]


def _expected_frames(placements: list[tuple[int, int, np.ndarray]]) -> list[tuple[int, int, int, int, int]]:
    frames = []
    for r0, c0, patch in placements:
        h, w = map(int, patch.shape)
        frames.append((r0, c0, r0 + h - 1, c0 + w - 1, _FRAME))
    frames.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4]))
    return frames


def _gen_attempt() -> np.ndarray | None:
    n_frames = int(_RNG.integers(*_FRAME_COUNT_RANGE))
    patches: list[np.ndarray] = []
    for _ in range(n_frames):
        patch = _make_frame_patch()
        if patch is None:
            return None
        patches.append(patch)

    canvas = _pick_canvas(patches)
    if canvas is None:
        return None
    placements = _place_patches(canvas, patches)
    if placements is None:
        return None

    grid = np.full(canvas, _BG, dtype=int)
    for r0, c0, patch in placements:
        h, w = map(int, patch.shape)
        grid[r0 : r0 + h, c0 : c0 + w] = patch

    expected_frames = _expected_frames(placements)
    if _find_frames(grid, _BG) != expected_frames:
        return None

    counts = np.bincount(grid.ravel(), minlength=10)
    if int(np.argmax(counts)) != _BG:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
