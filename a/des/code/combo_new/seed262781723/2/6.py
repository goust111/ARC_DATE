from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
ARC_MAX_DIM = 30
GEN_ATTEMPTS = 500

PAIR_COUNT_CHOICES = (2, 3, 4)
TOTAL_WIDTH_RANGES = {
    2: (6, 10),
    3: (6, 10),
    4: (6, 9),
}
HEIGHT_RANGES = {
    2: (4, 8),
    3: (3, 7),
    4: (3, 6),
}
TOP_BOTTOM_MARGIN = (0, 2)
VERTICAL_GAP_RANGE = (1, 3)
LEFT_RIGHT_MARGIN = (0, 2)
REGION_SLACK = (0, 2)
CENTER_GAP_RANGE = (3, 6)
PROFILE_STEP_CHOICES = (-2, -1, 0, 1, 2)
PROFILE_SPAN_MAX = 4


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_components(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    seen = np.zeros_like(g, dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if int(g[r, c]) == BG or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(int(r), int(c))])
            coords: list[tuple[int, int]] = []
            colors: set[int] = set()
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                colors.add(int(g[rr, cc]))
                for nr, nc in _neighbors4(int(rr), int(cc), h, w):
                    if int(g[nr, nc]) != BG and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))

            assert len(colors) == 1, "each object must be monochromatic"
            color = next(iter(colors))
            r0, c0, r1, c1 = _bbox(coords)
            patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
            for rr, cc in coords:
                patch[rr - r0, cc - c0] = color
            components.append(
                {
                    "color": int(color),
                    "r0": int(r0),
                    "c0": int(c0),
                    "r1": int(r1),
                    "c1": int(c1),
                    "patch": patch,
                }
            )

    return components


def _contiguous_run(cols: np.ndarray) -> bool:
    return bool(cols.size) and np.array_equal(cols, np.arange(int(cols[0]), int(cols[-1]) + 1))


def _lock_recess_profile(patch: np.ndarray) -> np.ndarray:
    mask = patch != BG
    h, w = map(int, patch.shape)
    widths: list[int] = []
    for r in range(h):
        cols = np.flatnonzero(mask[r])
        assert _contiguous_run(cols), "lock rows must be contiguous"
        assert int(cols[0]) == 0, "lock rows must start at the left edge"
        widths.append(int(cols.size))
    return np.array([w - width for width in widths], dtype=int)


def _key_offset_profile(patch: np.ndarray) -> np.ndarray:
    mask = patch != BG
    h, w = map(int, patch.shape)
    offsets: list[int] = []
    for r in range(h):
        cols = np.flatnonzero(mask[r])
        assert _contiguous_run(cols), "key rows must be contiguous"
        assert int(cols[-1]) == w - 1, "key rows must end at the right edge"
        offsets.append(int(cols[0]))
    return np.array(offsets, dtype=int)


def _match_lock_and_key(lock: dict[str, object], key: dict[str, object]) -> np.ndarray | None:
    lock_patch = np.asarray(lock["patch"], dtype=int)
    key_patch = np.asarray(key["patch"], dtype=int)
    if lock_patch.shape[0] != key_patch.shape[0]:
        return None

    lock_recess = np.asarray(lock["recess"], dtype=int)
    key_offsets = np.asarray(key["offsets"], dtype=int)
    totals = lock_recess + key_offsets
    if len(set(map(int, totals))) != 1:
        return None

    lock_w = int(lock_patch.shape[1])
    key_w = int(key_patch.shape[1])
    const = int(totals[0])
    shift = lock_w - const
    out_w = lock_w + key_w - const
    if shift < 0 or shift + key_w != out_w or out_w <= 0:
        return None

    out = np.full((lock_patch.shape[0], out_w), BG, dtype=int)
    lock_mask = lock_patch != BG
    key_mask = key_patch != BG

    out[:, :lock_w][lock_mask] = int(lock["color"])
    overlap = (out[:, shift : shift + key_w] != BG) & key_mask
    if np.any(overlap):
        return None
    out[:, shift : shift + key_w][key_mask] = int(key["color"])

    if np.any(out == BG):
        return None
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match left-side lock shapes with right-side key shapes by complementary edge profiles, then stack the merged rectangles.

    Input structure:
    - The background is black.
    - Several monochrome lock objects appear on the left side, and the same number of monochrome key objects appear on the right side.
    - Each lock is a solid row-prefix shape whose right edge has a varying notch profile.
    - Each key is a solid row-suffix shape whose left edge has the complementary protrusion profile.

    Rule:
    - Extract every foreground connected component and split them into the left group and the right group by horizontal position.
    - For each left-side lock, read its per-row recession depths from the right edge; for each right-side key, read its per-row starting offsets from the left edge.
    - Match a lock with the unique key of the same height whose rowwise recession-plus-offset sum is constant, then dock the pair into a solid rectangle with the lock on the left and the key on the right.
    - Preserve the locks' top-to-bottom order and vertically stack the merged rectangles with no blank rows between them.

    Key insight:
    - Colors are irrelevant for pairing; the only reliable signal is that a lock's missing edge profile and a key's protruding edge profile are rowwise complements of the same filled rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    components = _find_components(g)
    assert components and len(components) % 2 == 0, "expected equal numbers of locks and keys"

    components_by_x = sorted(
        components,
        key=lambda comp: (int(comp["c0"]) + int(comp["c1"]), int(comp["r0"])),
    )
    half = len(components_by_x) // 2
    locks = components_by_x[:half]
    keys = components_by_x[half:]
    assert int(locks[-1]["c1"]) < int(keys[0]["c0"]), "left and right groups must be separated"

    for lock in locks:
        lock["recess"] = _lock_recess_profile(np.asarray(lock["patch"], dtype=int))
    for key in keys:
        key["offsets"] = _key_offset_profile(np.asarray(key["patch"], dtype=int))

    locks = sorted(locks, key=lambda comp: (int(comp["r0"]), int(comp["c0"])))

    used_keys: set[int] = set()
    merged_patches: list[np.ndarray] = []
    for lock in locks:
        matches: list[tuple[int, np.ndarray]] = []
        for idx, key in enumerate(keys):
            if idx in used_keys:
                continue
            merged = _match_lock_and_key(lock, key)
            if merged is not None:
                matches.append((idx, merged))
        assert len(matches) == 1, "each lock must match exactly one key"
        match_idx, merged_patch = matches[0]
        used_keys.add(match_idx)
        merged_patches.append(merged_patch)

    assert len(used_keys) == len(keys)
    out_h = sum(int(patch.shape[0]) for patch in merged_patches)
    out_w = max(int(patch.shape[1]) for patch in merged_patches)
    out = np.full((out_h, out_w), BG, dtype=int)

    r0 = 0
    for patch in merged_patches:
        h, w = map(int, patch.shape)
        out[r0 : r0 + h, :w] = patch
        r0 += h
    return out


def _sample_heights(pair_count: int) -> list[int] | None:
    low, high = HEIGHT_RANGES[pair_count]
    heights = [random.randint(low, high) for _ in range(pair_count)]
    if sum(heights) > ARC_MAX_DIM:
        return None
    return heights


def _sample_profile(height: int, total_width: int, used_signatures: set[tuple[int, tuple[int, ...]]]) -> list[int] | None:
    for _ in range(200):
        min_width = random.randint(1, total_width - 2)
        max_span = min(PROFILE_SPAN_MAX, total_width - 1 - min_width)
        if max_span < 1:
            continue
        span = random.randint(1, max_span)
        values = [random.randint(0, span)]
        for _ in range(height - 1):
            nxt = values[-1] + random.choice(PROFILE_STEP_CHOICES)
            values.append(min(span, max(0, nxt)))
        values[random.randrange(height)] = 0
        values[random.randrange(height)] = span
        if len(set(values)) < 2:
            continue
        signature = (height, tuple(values))
        if signature in used_signatures:
            continue
        widths = [min_width + value for value in values]
        if min(widths) < 1 or max(widths) >= total_width:
            continue
        if len(set(widths)) < 2:
            continue
        used_signatures.add(signature)
        return widths
    return None


def _render_lock(widths: list[int], color: int) -> np.ndarray:
    h = len(widths)
    w = max(widths)
    patch = np.full((h, w), BG, dtype=int)
    for r, width in enumerate(widths):
        patch[r, :width] = int(color)
    return patch


def _render_key(widths: list[int], total_width: int, color: int) -> np.ndarray:
    h = len(widths)
    min_width = min(widths)
    w = total_width - min_width
    patch = np.full((h, w), BG, dtype=int)
    for r, width in enumerate(widths):
        start = width - min_width
        patch[r, start:] = int(color)
    return patch


def _render_pair_output(widths: list[int], total_width: int, lock_color: int, key_color: int) -> np.ndarray:
    h = len(widths)
    patch = np.full((h, total_width), BG, dtype=int)
    for r, width in enumerate(widths):
        patch[r, :width] = int(lock_color)
        patch[r, width:total_width] = int(key_color)
    return patch


def _sample_vertical_layout(heights: list[int]) -> tuple[list[int], int]:
    starts: list[int] = []
    cursor = random.randint(*TOP_BOTTOM_MARGIN)
    for i, height in enumerate(heights):
        starts.append(cursor)
        cursor += int(height)
        if i + 1 < len(heights):
            cursor += random.randint(*VERTICAL_GAP_RANGE)
    cursor += random.randint(*TOP_BOTTOM_MARGIN)
    return starts, cursor


def _non_identity_permutation(n: int) -> list[int]:
    perm = list(range(n))
    if n <= 1:
        return perm
    while True:
        random.shuffle(perm)
        if any(i != perm[i] for i in range(n)):
            return perm[:]


def _paint_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = map(int, patch.shape)
    mask = patch != BG
    grid[r0 : r0 + h, c0 : c0 + w][mask] = patch[mask]


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choice(PAIR_COUNT_CHOICES)
    total_width = random.randint(*TOTAL_WIDTH_RANGES[pair_count])
    heights = _sample_heights(pair_count)
    if heights is None:
        return None

    colors = random.sample(PALETTE, pair_count * 2)
    used_signatures: set[tuple[int, tuple[int, ...]]] = set()
    pairs: list[dict[str, object]] = []

    for idx, height in enumerate(heights):
        widths = _sample_profile(height, total_width, used_signatures)
        if widths is None:
            return None
        lock_color = int(colors[2 * idx])
        key_color = int(colors[2 * idx + 1])
        lock_patch = _render_lock(widths, lock_color)
        key_patch = _render_key(widths, total_width, key_color)
        expected_patch = _render_pair_output(widths, total_width, lock_color, key_color)
        pairs.append(
            {
                "widths": widths,
                "lock_patch": lock_patch,
                "key_patch": key_patch,
                "expected": expected_patch,
            }
        )

    lock_starts, left_height = _sample_vertical_layout(heights)
    key_perm = _non_identity_permutation(pair_count)
    key_heights = [heights[i] for i in key_perm]
    key_starts, right_height = _sample_vertical_layout(key_heights)
    grid_h = max(left_height, right_height)
    if grid_h > ARC_MAX_DIM:
        return None

    max_lock_w = max(int(pair["lock_patch"].shape[1]) for pair in pairs)
    max_key_w = max(int(pair["key_patch"].shape[1]) for pair in pairs)
    left_margin = random.randint(*LEFT_RIGHT_MARGIN)
    right_margin = random.randint(*LEFT_RIGHT_MARGIN)
    left_region_w = max_lock_w + random.randint(*REGION_SLACK)
    right_region_w = max_key_w + random.randint(*REGION_SLACK)
    center_gap = random.randint(*CENTER_GAP_RANGE)
    right_region_c0 = left_margin + left_region_w + center_gap
    grid_w = right_region_c0 + right_region_w + right_margin
    if grid_w > ARC_MAX_DIM:
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for pair, r0 in zip(pairs, lock_starts):
        patch = np.asarray(pair["lock_patch"], dtype=int)
        max_c0 = left_margin + left_region_w - patch.shape[1]
        c0 = random.randint(left_margin, int(max_c0))
        _paint_patch(grid, patch, int(r0), int(c0))

    for key_slot, pair_idx in enumerate(key_perm):
        patch = np.asarray(pairs[pair_idx]["key_patch"], dtype=int)
        r0 = key_starts[key_slot]
        max_c0 = right_region_c0 + right_region_w - patch.shape[1]
        c0 = random.randint(int(right_region_c0), int(max_c0))
        _paint_patch(grid, patch, int(r0), int(c0))

    expected = np.vstack([np.asarray(pair["expected"], dtype=int) for pair in pairs])
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid lock-key ARC sample")
