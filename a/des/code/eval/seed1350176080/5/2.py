from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

INSTRUCTION_COLORS = (RED, BLUE, GREEN)
COLOR_PRIORITY = {RED: 0, BLUE: 1, GREEN: 2}
SEED_PALETTE = (4, 6, 7, 8, 9)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _segment_info(grid: np.ndarray, coords: np.ndarray) -> dict | None:
    values = np.unique(grid[coords[:, 0], coords[:, 1]])
    if values.size != 1:
        return None
    color = int(values[0])
    if color not in INSTRUCTION_COLORS:
        return None

    r0, c0, r1, c1 = _bbox(coords)
    area = int(coords.shape[0])
    if r0 == r1:
        length = c1 - c0 + 1
        if area != length or length < 2:
            return None
        if not np.array_equal(np.sort(coords[:, 1]), np.arange(c0, c1 + 1)):
            return None
        orientation = "h"
    elif c0 == c1:
        length = r1 - r0 + 1
        if area != length or length < 2:
            return None
        if not np.array_equal(np.sort(coords[:, 0]), np.arange(r0, r1 + 1)):
            return None
        orientation = "v"
    else:
        return None

    return {
        "color": color,
        "orientation": orientation,
        "row": r0,
        "col": c0,
    }


def _apply_fold(pattern: np.ndarray, orientation: str) -> np.ndarray:
    if orientation == "h":
        return np.vstack([pattern, np.flipud(pattern)])
    if orientation == "v":
        return np.hstack([pattern, np.fliplr(pattern)])
    raise AssertionError(f"unknown orientation: {orientation}")


def _separator_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == GREY)]
    assert len(rows) == 1, "expected exactly one full-width grey separator row"
    sep = int(rows[0])
    assert 0 < sep < grid.shape[0] - 1, "separator must split non-empty top and bottom panels"
    return sep


def _ordered_instructions(segments: list[dict]) -> list[dict]:
    return sorted(segments, key=lambda seg: (COLOR_PRIORITY[seg["color"]], seg["row"], seg["col"]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group the colored line segments by priority color and use their directions as
    mirror-fold instructions for the seed pattern above the grey separator.

    Horizontal segments duplicate the current pattern with a vertically flipped
    copy below; vertical segments duplicate it with a horizontally flipped copy
    on the right. The final folded pattern is drawn into the blank lower panel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    sep = _separator_row(g)
    top = g[:sep, :]
    mask = (top != BLACK) & (top != GREY)
    comps = _components(mask)
    assert comps, "top panel must contain a seed and instruction segments"

    segments: list[dict] = []
    seed_components: list[np.ndarray] = []
    for coords in comps:
        info = _segment_info(top, coords)
        if info is None:
            seed_components.append(coords)
        else:
            segments.append(info)

    assert len(seed_components) == 1, "expected exactly one non-segment seed component"
    assert segments, "expected at least one instruction segment"

    seed_coords = seed_components[0]
    r0, c0, r1, c1 = _bbox(seed_coords)
    seed = top[r0 : r1 + 1, c0 : c1 + 1].copy()

    pattern = seed
    for seg in _ordered_instructions(segments):
        pattern = _apply_fold(pattern, seg["orientation"])

    out = g.copy()
    out[sep + 1 :, :] = BLACK
    bottom_h = g.shape[0] - sep - 1
    bottom_w = g.shape[1]
    ph, pw = pattern.shape
    assert ph <= bottom_h and pw <= bottom_w, "lower panel must fit the folded pattern"
    out[sep + 1 : sep + 1 + ph, :pw] = pattern
    return out


def _sample_seed() -> np.ndarray:
    for _ in range(100):
        h = random.randint(3, 4)
        w = random.randint(3, 4)
        palette = random.sample(SEED_PALETTE, k=random.randint(2, min(4, len(SEED_PALETTE))))
        seed = np.array([random.choice(palette) for _ in range(h * w)], dtype=int).reshape(h, w)
        if np.unique(seed).size < 2:
            continue
        if np.array_equal(seed, np.flipud(seed)):
            continue
        if np.array_equal(seed, np.fliplr(seed)):
            continue
        return seed
    raise RuntimeError("failed to sample a non-degenerate seed")


def _sample_counts() -> tuple[int, int, int]:
    options = [
        (r, b, g)
        for r in range(1, 4)
        for b in range(1, 4)
        for g in range(1, 4)
        if 3 <= r + b + g <= 5
    ]
    return random.choice(options)


def _sample_segments() -> list[dict]:
    red_count, blue_count, green_count = _sample_counts()
    counts = {RED: red_count, BLUE: blue_count, GREEN: green_count}
    total = red_count + blue_count + green_count

    for _ in range(200):
        orientations = [random.choice(("h", "v")) for _ in range(total)]
        h_count = orientations.count("h")
        v_count = total - h_count
        if 1 <= h_count <= 3 and 1 <= v_count <= 3:
            break
    else:
        raise RuntimeError("failed to balance segment orientations")

    random.shuffle(orientations)
    segments: list[dict] = []
    idx = 0
    for color in (RED, BLUE, GREEN):
        for _ in range(counts[color]):
            segments.append(
                {
                    "color": color,
                    "orientation": orientations[idx],
                    "length": random.randint(2, 4),
                }
            )
            idx += 1
    return segments


def _segment_patch(segment: dict) -> np.ndarray:
    if segment["orientation"] == "h":
        return np.full((1, segment["length"]), segment["color"], dtype=int)
    return np.full((segment["length"], 1), segment["color"], dtype=int)


def _mark_blocked(blocked: np.ndarray, r: int, c: int, h: int, w: int) -> None:
    r0 = max(0, r - 1)
    c0 = max(0, c - 1)
    r1 = min(blocked.shape[0], r + h + 1)
    c1 = min(blocked.shape[1], c + w + 1)
    blocked[r0:r1, c0:c1] = True


def _place_patch(canvas: np.ndarray, blocked: np.ndarray, patch: np.ndarray) -> tuple[int, int] | None:
    h, w = patch.shape
    max_r = canvas.shape[0] - h
    max_c = canvas.shape[1] - w
    if max_r < 0 or max_c < 0:
        return None

    for _ in range(300):
        r = random.randint(0, max_r)
        c = random.randint(0, max_c)
        if np.any(blocked[r : r + h, c : c + w]):
            continue
        canvas[r : r + h, c : c + w] = patch
        _mark_blocked(blocked, r, c, h, w)
        return r, c
    return None


def _render_top(seed: np.ndarray, segments: list[dict], width: int, height: int) -> np.ndarray | None:
    canvas = np.full((height, width), BLACK, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)

    if _place_patch(canvas, blocked, seed) is None:
        return None

    for segment in segments:
        patch = _segment_patch(segment)
        placed = _place_patch(canvas, blocked, patch)
        if placed is None:
            return None
        segment["row"], segment["col"] = placed

    return canvas


def _folded_pattern(seed: np.ndarray, segments: list[dict]) -> np.ndarray:
    pattern = seed.copy()
    for seg in _ordered_instructions(segments):
        pattern = _apply_fold(pattern, seg["orientation"])
    return pattern


def _gen_attempt() -> np.ndarray | None:
    seed = _sample_seed()
    segments = _sample_segments()
    h_count = sum(seg["orientation"] == "h" for seg in segments)
    v_count = sum(seg["orientation"] == "v" for seg in segments)
    fh = seed.shape[0] * (2 ** h_count)
    fw = seed.shape[1] * (2 ** v_count)

    top_h = random.randint(8, 12)
    width = max(fw + random.randint(0, 4), 12)
    top = _render_top(seed, segments, width=width, height=top_h)
    if top is None:
        return None
    folded = _folded_pattern(seed, segments)

    bottom_h = fh + random.randint(0, 3)
    grid = np.full((top_h + 1 + bottom_h, width), BLACK, dtype=int)
    grid[:top_h, :] = top
    grid[top_h, :] = GREY

    expected = grid.copy()
    expected[top_h + 1 : top_h + 1 + fh, :fw] = folded

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid folding-instruction puzzle")
