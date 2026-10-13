from __future__ import annotations

import random

import numpy as np


BG = 0
RED = 2
GREEN = 3
MAROON = 9
PATTERN_PALETTE = (1, 4, 5, 6, 7, 8)

GRID_MIN = 18
GRID_MAX = 24
PATTERN_H_RANGE = (8, 12)
HALF_W_RANGE = (4, 6)
OCCLUDER_SIZES = (3, 4)
MARKER_COUNT_RANGE = (2, 4)
GEN_ATTEMPTS = 400


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_distance(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int]:
    dr = max(0, b[0] - a[2] - 1, a[0] - b[2] - 1)
    dc = max(0, b[1] - a[3] - 1, a[1] - b[3] - 1)
    return int(dr), int(dc)


def _paste_opaque(canvas: np.ndarray, sprite: np.ndarray, r0: int, c0: int) -> None:
    h, w = sprite.shape
    canvas[r0 : r0 + h, c0 : c0 + w] = sprite


def _find_occluder(grid: np.ndarray) -> tuple[int, int, int, int]:
    mask = grid == MAROON
    box = _bbox_from_mask(mask)
    assert box is not None, "expected a maroon occluder"
    r0, c0, r1, c1 = box
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    assert h == w and h in OCCLUDER_SIZES, "occluder must be a 3x3 or 4x4 square"
    assert int(mask.sum()) == h * w, "occluder must be a solid maroon square"
    return box


def _recover_template(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    r0, c0, r1, c1 = _find_occluder(g)
    mc0 = W - 1 - c1
    mc1 = W - 1 - c0
    assert 0 <= mc0 <= mc1 < W
    mirrored = g[r0 : r1 + 1, mc0 : mc1 + 1]
    assert not np.any(mirrored == MAROON), "occluder mirror must stay visible"
    return np.fliplr(mirrored).copy(), (r0, c0, r1, c1)


def _marker_positions(grid: np.ndarray) -> list[tuple[int, int, int]]:
    markers = []
    for r, c in np.argwhere((grid == RED) | (grid == GREEN)):
        markers.append((int(r), int(c), int(grid[r, c])))
    return sorted(markers)


def _render_output(shape: tuple[int, int], markers: list[tuple[int, int, int]], template: np.ndarray) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for r, c, color in markers:
        out[r, c] = color
    flipped = np.fliplr(template)
    h, w = template.shape
    for r, c, color in markers:
        sprite = template if color == GREEN else flipped
        assert r + h <= shape[0] and c + w <= shape[1], "stamp must stay on the canvas"
        _paste_opaque(out, sprite, r, c)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Recover the occluded mirrored patch, then use it as a stamp whose orientation is marker-controlled.

    Input structure:
    - A black background contains one central multicolor pattern that is mirror-symmetric about the scene's vertical axis.
    - A solid maroon 3x3 or 4x4 square occludes part of that pattern, and its mirrored counterpart is fully visible.
    - Outside the pattern sit isolated green and red single-pixel markers.

    Rule:
    - Identify the maroon square and reflect its mirrored counterpart back across the vertical axis to recover the hidden patch.
    - Treat that recovered patch as the stamp template, then erase the whole central pattern and the occluder while keeping only the markers on black.
    - Stamp the template with its top-left corner at every green marker, and stamp a left-right mirrored copy with its top-left corner at every red marker.

    Key insight:
    - The mirror-recovered patch is not the final answer; it becomes a reusable stamp, and marker color decides whether the stamp is mirrored first.
    """
    g = np.asarray(grid, dtype=int)
    template, _ = _recover_template(g)
    markers = _marker_positions(g)
    assert markers, "expected at least one marker"
    return _render_output(g.shape, markers, template)


def _build_left_half(height: int, half_w: int, palette: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    left = rng.choice(palette, size=(height, half_w))
    mode = int(rng.integers(0, 3))
    if mode == 0:
        for _ in range(int(rng.integers(6, 12))):
            rh = int(rng.integers(1, min(5, height) + 1))
            rw = int(rng.integers(1, min(4, half_w) + 1))
            r0 = int(rng.integers(0, height - rh + 1))
            c0 = int(rng.integers(0, half_w - rw + 1))
            left[r0 : r0 + rh, c0 : c0 + rw] = int(rng.choice(palette))
    elif mode == 1:
        for r in range(height):
            left[r, :] = int(rng.choice(palette))
        for _ in range(int(rng.integers(4, 8))):
            c = int(rng.integers(0, half_w))
            left[:, c] = int(rng.choice(palette))
    else:
        for c in range(half_w):
            left[:, c] = int(rng.choice(palette))
        for _ in range(int(rng.integers(4, 8))):
            r0 = int(rng.integers(0, height))
            left[r0, :] = int(rng.choice(palette))

    used = set(int(x) for x in left.flat)
    for color in palette:
        if color not in used:
            rr = int(rng.integers(0, height))
            cc = int(rng.integers(0, half_w))
            left[rr, cc] = color
            used.add(color)
    return left.astype(int)


def _build_center_column(height: int, palette: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    col = rng.choice(palette, size=(height, 1))
    for _ in range(int(rng.integers(2, 5))):
        r0 = int(rng.integers(0, height))
        r1 = int(rng.integers(r0 + 1, height + 1))
        col[r0:r1, 0] = int(rng.choice(palette))
    return col.astype(int)


def _pattern_from_halves(height: int, half_w: int, width_parity: int, rng: np.random.Generator) -> np.ndarray:
    color_count = int(rng.integers(3, 6))
    palette = tuple(int(x) for x in rng.choice(PATTERN_PALETTE, size=color_count, replace=False))
    left = _build_left_half(height, half_w, palette, rng)
    if width_parity == 0:
        return np.concatenate([left, np.fliplr(left)], axis=1)
    center = _build_center_column(height, palette, rng)
    return np.concatenate([left, center, np.fliplr(left)], axis=1)


def _valid_hidden_patch(patch: np.ndarray) -> bool:
    return len(set(int(x) for x in patch.flat)) >= 2 and not np.array_equal(patch, np.fliplr(patch))


def _choose_occluder(
    pattern: np.ndarray,
    scene_width: int,
    pattern_c0: int,
    rng: np.random.Generator,
) -> tuple[tuple[int, int, int, int], np.ndarray]:
    ph, pw = pattern.shape
    half_w = pw // 2
    side = "left" if rng.random() < 0.5 else "right"
    sizes = list(OCCLUDER_SIZES)
    rng.shuffle(sizes)

    for size in sizes:
        rows = list(range(ph - size + 1))
        cols = list(range(half_w - size + 1))
        rng.shuffle(rows)
        rng.shuffle(cols)
        for rr in rows:
            for cc in cols:
                left_patch = pattern[rr : rr + size, cc : cc + size]
                if not _valid_hidden_patch(left_patch):
                    continue
                if side == "left":
                    hidden = left_patch.copy()
                    gc0 = pattern_c0 + cc
                else:
                    hidden = np.fliplr(left_patch).copy()
                    gc0 = pattern_c0 + pw - size - cc
                gc1 = gc0 + size - 1
                mirror_c0 = scene_width - 1 - gc1
                mirror_c1 = scene_width - 1 - gc0
                visible = pattern[rr : rr + size, mirror_c0 - pattern_c0 : mirror_c1 - pattern_c0 + 1]
                if np.array_equal(np.fliplr(visible), hidden):
                    return (rr, gc0, rr + size - 1, gc1), hidden
    raise AssertionError("failed to find a non-degenerate occluder patch")


def _candidate_marker_positions(
    shape: tuple[int, int],
    stamp_size: int,
    pattern_box: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    H, W = shape
    candidates = []
    for r in range(H - stamp_size + 1):
        for c in range(W - stamp_size + 1):
            stamp_box = (r, c, r + stamp_size - 1, c + stamp_size - 1)
            dr, dc = _bbox_distance(stamp_box, pattern_box)
            if max(dr, dc) < 2:
                continue
            candidates.append((r, c))
    return candidates


def _pick_markers(
    shape: tuple[int, int],
    stamp_size: int,
    pattern_box: tuple[int, int, int, int],
    rng: np.random.Generator,
) -> list[tuple[int, int, int]]:
    count = int(rng.integers(MARKER_COUNT_RANGE[0], MARKER_COUNT_RANGE[1] + 1))
    colors = [GREEN, RED]
    while len(colors) < count:
        colors.append(GREEN if rng.random() < 0.5 else RED)
    rng.shuffle(colors)

    candidates = _candidate_marker_positions(shape, stamp_size, pattern_box)
    rng.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    for r, c in candidates:
        stamp_box = (r, c, r + stamp_size - 1, c + stamp_size - 1)
        ok = True
        for rr, cc in chosen:
            other = (rr, cc, rr + stamp_size - 1, cc + stamp_size - 1)
            dr, dc = _bbox_distance(stamp_box, other)
            if max(dr, dc) < 1:
                ok = False
                break
            if abs(r - rr) <= 1 and abs(c - cc) <= 1:
                ok = False
                break
        if ok:
            chosen.append((r, c))
        if len(chosen) == count:
            break

    assert len(chosen) == count, "not enough non-overlapping marker positions"
    return [(r, c, color) for (r, c), color in zip(chosen, colors)]


def _validate_scene(inp: np.ndarray, out: np.ndarray, template: np.ndarray) -> None:
    recovered, occ_box = _recover_template(inp)
    assert np.array_equal(recovered, template)

    markers = _marker_positions(inp)
    assert 2 <= len(markers) <= 4
    colors = {color for _, _, color in markers}
    assert colors == {RED, GREEN}
    for r, c, _ in markers:
        neighborhood = inp[max(0, r - 1) : r + 2, max(0, c - 1) : c + 2]
        assert int(np.count_nonzero(neighborhood)) == 1, "markers must be isolated"

    r0, c0, r1, c1 = occ_box
    hidden = inp.copy()
    hidden[r0 : r1 + 1, c0 : c1 + 1] = template
    pattern_mask = (hidden != BG) & (hidden != RED) & (hidden != GREEN)
    pattern_box = _bbox_from_mask(pattern_mask)
    assert pattern_box is not None

    size = template.shape[0]
    boxes = []
    for r, c, color in markers:
        sprite = template if color == GREEN else np.fliplr(template)
        assert r + size <= inp.shape[0] and c + size <= inp.shape[1]
        boxes.append((r, c, r + size - 1, c + size - 1, sprite))
        dr, dc = _bbox_distance((r, c, r + size - 1, c + size - 1), pattern_box)
        assert max(dr, dc) >= 2

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            dr, dc = _bbox_distance(boxes[i][:4], boxes[j][:4])
            assert max(dr, dc) >= 1, "stamps must not overlap"

    expected = _render_output(inp.shape, markers, template)
    assert np.array_equal(out, expected)
    assert not np.array_equal(template, np.fliplr(template)), "red branch must matter"
    assert len(set(int(x) for x in template.flat)) >= 2, "template must be multicolor"


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()

    for _ in range(GEN_ATTEMPTS):
        H = int(rng.integers(GRID_MIN, GRID_MAX + 1))
        W = int(rng.integers(GRID_MIN, GRID_MAX + 1))
        ph = int(rng.integers(PATTERN_H_RANGE[0], PATTERN_H_RANGE[1] + 1))
        half_w = int(rng.integers(HALF_W_RANGE[0], HALF_W_RANGE[1] + 1))
        pw = 2 * half_w + (W % 2)
        if pw > W - 6 or ph > H - 6:
            continue

        pr0_lo = 3
        pr0_hi = H - ph - 3
        if pr0_lo > pr0_hi:
            continue
        pr0 = int(rng.integers(pr0_lo, pr0_hi + 1))
        pc0 = (W - pw) // 2
        pattern = _pattern_from_halves(ph, half_w, W % 2, rng)
        assert pattern.shape == (ph, pw)

        try:
            occ_local_box, template = _choose_occluder(pattern, W, pc0, rng)
        except AssertionError:
            continue

        inp = np.zeros((H, W), dtype=int)
        _paste_opaque(inp, pattern, pr0, pc0)

        lr0, gc0, lr1, gc1 = occ_local_box
        gr0 = pr0 + lr0
        gr1 = pr0 + lr1
        inp[gr0 : gr1 + 1, gc0 : gc1 + 1] = MAROON

        pattern_box = (pr0, pc0, pr0 + ph - 1, pc0 + pw - 1)
        try:
            markers = _pick_markers(inp.shape, template.shape[0], pattern_box, rng)
        except AssertionError:
            continue

        for r, c, color in markers:
            inp[r, c] = color

        out = main(inp)
        try:
            _validate_scene(inp, out, template)
        except AssertionError:
            continue
        return inp

    raise RuntimeError("failed to generate a valid puzzle instance")
