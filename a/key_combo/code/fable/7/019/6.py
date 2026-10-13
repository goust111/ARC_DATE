from __future__ import annotations

import random

import numpy as np


def _background(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _row_runs(grid: np.ndarray, color: int) -> dict[tuple[int, int], list[int]]:
    """Map each inclusive horizontal run (start, end) to its row indices."""
    runs: dict[tuple[int, int], list[int]] = {}
    for r in range(grid.shape[0]):
        c = 0
        while c < grid.shape[1]:
            if int(grid[r, c]) != color:
                c += 1
                continue
            start = c
            while c + 1 < grid.shape[1] and int(grid[r, c + 1]) == color:
                c += 1
            if c - start + 1 >= 3:
                runs.setdefault((start, c), []).append(r)
            c += 1
    return runs


def _rounded_frames(grid: np.ndarray, bg: int) -> list[tuple[int, int, int, int, int]]:
    """Return (area, r0, c0, r1, c1) for exact one-cell rounded frames."""
    h, w = grid.shape
    found: set[tuple[int, int, int, int, int]] = set()
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for (a, b), rows in _row_runs(grid, color).items():
            c0, c1 = a - 1, b + 1
            if c0 < 0 or c1 >= w:
                continue
            for i, r0 in enumerate(rows):
                for r1 in rows[i + 1 :]:
                    if r1 - r0 < 4:
                        continue
                    if not (
                        int(grid[r0, c0]) == bg
                        and int(grid[r0, c1]) == bg
                        and int(grid[r1, c0]) == bg
                        and int(grid[r1, c1]) == bg
                    ):
                        continue
                    if not (
                        np.all(grid[r0, c0 + 1 : c1] == color)
                        and np.all(grid[r1, c0 + 1 : c1] == color)
                        and np.all(grid[r0 + 1 : r1, c0] == color)
                        and np.all(grid[r0 + 1 : r1, c1] == color)
                    ):
                        continue
                    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
                    if np.any(interior != bg):
                        found.add(((r1 - r0 + 1) * (c1 - c0 + 1), r0, c0, r1, c1))

    frames = sorted(found)
    assert len(frames) >= 2, "expected at least two rounded carriers"
    # The puzzle specifies separate carriers; this also rejects accidental nested
    # frame-like motifs rather than silently assigning them a second meaning.
    for i, (_a, r0, c0, r1, c1) in enumerate(frames):
        for _b, s0, d0, s1, d1 in frames[i + 1 :]:
            assert r1 < s0 or s1 < r0 or c1 < d0 or d1 < c0, "ambiguous overlapping frames"
    assert len({item[0] for item in frames}) == len(frames), "carrier sizes must be unique"
    return frames


def _crop_pattern(grid: np.ndarray, frame: tuple[int, int, int, int, int], bg: int) -> np.ndarray:
    _area, r0, c0, r1, c1 = frame
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    coords = np.argwhere(interior != bg)
    assert len(coords), "each carrier must contain a pattern"
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    return interior[lo[0] : hi[0] + 1, lo[1] : hi[1] + 1].copy()


def _overlay(patterns: list[tuple[int, np.ndarray]], bg: int) -> np.ndarray:
    max_h = max(p.shape[0] for _area, p in patterns)
    max_w = max(p.shape[1] for _area, p in patterns)
    assert all((max_h - p.shape[0]) % 2 == 0 for _area, p in patterns)
    assert all((max_w - p.shape[1]) % 2 == 0 for _area, p in patterns)
    out = np.full((max_h, max_w), bg, dtype=int)
    # Large carriers first; successively smaller carriers overwrite them.
    for _area, patch in sorted(patterns, key=lambda item: item[0], reverse=True):
        r = (max_h - patch.shape[0]) // 2
        c = (max_w - patch.shape[1]) // 2
        view = out[r : r + patch.shape[0], c : c + patch.shape[1]]
        mask = patch != bg
        view[mask] = patch[mask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Discard rounded carriers and center-overlay their cropped inner patterns, with smaller carriers on top.

    Input structure:
    - The dominant background contains several mutually separate rounded rectangular
      frames whose four corners are background and whose carrier areas are unique.
    - Every frame encloses a nonempty pattern; the cropped pattern dimensions have
      compatible row and column parity for exact common-center alignment.

    Rule:
    - Remove every frame, crop each enclosed pattern to its own non-background
      bounding box, and make a background canvas of the maximum cropped height and width.
    - Center every crop on that canvas. At contested cells, a pattern carried by a
      smaller frame overwrites one carried by a larger frame.

    Key insight:
    - The discarded frame is the sole depth cue: carrier size, not pattern size or
      color, determines the transparent-overlay order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and 1 <= min(g.shape) and max(g.shape) <= 30
    bg = _background(g)
    frames = _rounded_frames(g, bg)
    patterns = [(frame[0], _crop_pattern(g, frame, bg)) for frame in frames]
    return _overlay(patterns, bg)


# ------------------------------ generator ------------------------------

_SEEN_OUTPUTS: set[tuple[tuple[int, int], bytes]] = set()
_MAX_ATTEMPTS = 800


def _tight_mask(h: int, w: int, rng: random.Random) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    required = {(h // 2, w // 2)}
    if h == 1:
        required.update({(0, 0), (0, w - 1)})
    elif w == 1:
        required.update({(0, 0), (h - 1, 0)})
    else:
        required.update({(0, 0), (h - 1, w - 1)})
    for cell in required:
        mask[cell] = True
    pool = [(r, c) for r in range(h) for c in range(w) if not mask[r, c]]
    rng.shuffle(pool)
    extra = rng.randint(1, max(1, min(len(pool), (h * w) // 3))) if pool else 0
    for r, c in pool[:extra]:
        mask[r, c] = True
    return mask


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 + 1 : c1] = color
    grid[r1, c0 + 1 : c1] = color
    grid[r0 + 1 : r1, c0] = color
    grid[r0 + 1 : r1, c1] = color


def _independent_overlay(layers: list[tuple[int, np.ndarray]], bg: int) -> np.ndarray:
    """Generator oracle, deliberately separate from the solver's helper."""
    oh = max(p.shape[0] for _, p in layers)
    ow = max(p.shape[1] for _, p in layers)
    out = np.full((oh, ow), bg, dtype=int)
    for area, patch in sorted(layers, reverse=True, key=lambda x: x[0]):
        top, left = (oh - patch.shape[0]) // 2, (ow - patch.shape[1]) // 2
        for r in range(patch.shape[0]):
            for c in range(patch.shape[1]):
                if int(patch[r, c]) != bg:
                    out[top + r, left + c] = int(patch[r, c])
    return out


def _wrong_overlay(layers: list[tuple[int, np.ndarray]], bg: int, mode: str) -> np.ndarray:
    oh = max(p.shape[0] for _, p in layers)
    ow = max(p.shape[1] for _, p in layers)
    out = np.full((oh, ow), bg, dtype=int)
    if mode == "reverse_depth":
        ordered = sorted(layers, key=lambda x: x[0])
    elif mode == "pattern_depth":  # Pattern area incorrectly supplies depth.
        ordered = sorted(layers, key=lambda x: x[1].size, reverse=True)
    else:  # Keep the correct depth order, changing alignment only.
        ordered = sorted(layers, key=lambda x: x[0], reverse=True)
    for _area, patch in ordered:
        top = 0 if mode == "uncentered" else (oh - patch.shape[0]) // 2
        left = 0 if mode == "uncentered" else (ow - patch.shape[1]) // 2
        view = out[top : top + patch.shape[0], left : left + patch.shape[1]]
        view[patch != bg] = patch[patch != bg]
    return out


def _attempt() -> tuple[np.ndarray, np.ndarray] | None:
    rng = random.Random()
    n = rng.choice((2, 3, 4))
    H, W = rng.randint(25, 30), rng.randint(25, 30)
    palette = rng.sample(range(10), n + 2)
    bg, frame_color = palette[:2]
    pattern_colors = palette[2:]

    dim_pool = [(h, w) for h in range(9, 12) for w in range(9, 12)]
    rng.shuffle(dim_pool)
    dims: list[tuple[int, int]] = []
    used_areas: set[int] = set()
    for dims_candidate in dim_pool:
        area = dims_candidate[0] * dims_candidate[1]
        if area not in used_areas:
            dims.append(dims_candidate)
            used_areas.add(area)
        if len(dims) == n:
            break
    dims.sort(key=lambda x: x[0] * x[1])

    row_bands = ((1, H // 2 - 1), (H // 2 + 1, H - 2))
    col_bands = ((1, W // 2 - 1), (W // 2 + 1, W - 2))
    slots = [(rr, cc) for rr in range(2) for cc in range(2)]
    rng.shuffle(slots)
    boxes: list[tuple[int, int, int, int]] = []
    for (fh, fw), (sr, sc) in zip(dims, slots):
        ra, rb = row_bands[sr]
        ca, cb = col_bands[sc]
        if rb - ra + 1 < fh or cb - ca + 1 < fw:
            return None
        r0 = rng.randint(ra, rb - fh + 1)
        c0 = rng.randint(ca, cb - fw + 1)
        boxes.append((r0, c0, r0 + fh - 1, c0 + fw - 1))

    # The smallest carrier deliberately gets the largest pattern, while the
    # largest carrier gets a smaller one: pattern size cannot proxy for depth.
    shape_options = [(3, 3), (3, 5), (5, 3), (5, 5)]
    shapes = [(5, 5)]
    for _ in range(n - 2):
        shapes.append(rng.choice(shape_options))
    shapes.append(rng.choice(((1, 3), (3, 1), (3, 3))))

    grid = np.full((H, W), bg, dtype=int)
    layers: list[tuple[int, np.ndarray]] = []
    crop_was_needed = False
    for (fh, fw), box, shape, color in zip(dims, boxes, shapes, pattern_colors):
        _draw_frame(grid, box, frame_color)
        ph, pw = shape
        patch = np.full((ph, pw), bg, dtype=int)
        patch[_tight_mask(ph, pw, rng)] = color
        r0, c0, r1, c1 = box
        choices_r = list(range(r0 + 2, r1 - ph))
        choices_c = list(range(c0 + 2, c1 - pw))
        if not choices_r or not choices_c:
            return None
        pr, pc = rng.choice(choices_r), rng.choice(choices_c)
        crop_was_needed |= pr != r0 + 1 or pc != c0 + 1
        target = grid[pr : pr + ph, pc : pc + pw]
        target[patch != bg] = patch[patch != bg]
        layers.append((fh * fw, patch))

    expected = _independent_overlay(layers, bg)
    if not crop_was_needed:
        return None
    if np.array_equal(expected, _wrong_overlay(layers, bg, "reverse_depth")):
        return None
    if np.array_equal(expected, _wrong_overlay(layers, bg, "pattern_depth")):
        return None
    if np.array_equal(expected, _wrong_overlay(layers, bg, "uncentered")):
        return None

    # Every carrier contributes at least one final visible pixel.
    for i in range(n):
        if np.array_equal(expected, _independent_overlay(layers[:i] + layers[i + 1 :], bg)):
            return None
    try:
        parsed = _rounded_frames(grid, bg)
        if len(parsed) != n or not np.array_equal(main(grid), expected):
            return None
    except (AssertionError, ValueError):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    """Generate a varied, nondegenerate instance satisfying all puzzle constraints."""
    for _ in range(_MAX_ATTEMPTS):
        pair = _attempt()
        if pair is None:
            continue
        grid, out = pair
        signature = (out.shape, out.tobytes())
        if signature in _SEEN_OUTPUTS:
            continue
        _SEEN_OUTPUTS.add(signature)
        return grid
    raise RuntimeError("failed to generate a unique valid rounded-frame overlay instance")
