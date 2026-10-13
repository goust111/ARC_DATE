from __future__ import annotations

from collections import Counter

import numpy as np


BG = 0
BLUE = 1
NON_BLUE_COLORS = tuple(range(2, 10))
ORIENTATION_TO_CORNER = {
    "id": (0, 0),
    "h": (0, -1),
    "v": (-1, 0),
    "r": (-1, -1),
}


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start = None
    for i, flag in enumerate(mask.tolist()):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            bands.append((start, i - 1))
            start = None
    if start is not None:
        bands.append((start, len(mask) - 1))
    return bands


def _select_dense_bands(counts: np.ndarray) -> list[tuple[int, int]]:
    counts = np.asarray(counts, dtype=int)
    positive = sorted(int(v) for v in set(counts.tolist()) if v > 0)
    assert positive, "expected non-background structure"

    best_bands = None
    best_score = None
    for threshold in positive:
        bands = _runs(counts >= threshold)
        if len(bands) < 2:
            continue
        lengths = [b1 - b0 + 1 for b0, b1 in bands]
        if len(set(lengths)) != 1:
            continue
        score = (sum(lengths), len(bands), threshold)
        if best_score is None or score > best_score:
            best_score = score
            best_bands = bands

    assert best_bands is not None, "failed to isolate dense slot bands"
    return best_bands


def _apply_orientation(patch: np.ndarray, orientation: str) -> np.ndarray:
    if orientation == "id":
        return patch.copy()
    if orientation == "h":
        return np.fliplr(patch)
    if orientation == "v":
        return np.flipud(patch)
    if orientation == "r":
        return np.flipud(np.fliplr(patch))
    raise ValueError(f"unknown orientation: {orientation}")


def _anchor_corner(patch: np.ndarray) -> str:
    blue_coords = np.argwhere(patch == BLUE)
    assert len(blue_coords) == 1, "each slot must contain exactly one blue anchor"
    r, c = map(int, blue_coords[0])
    h, w = map(int, patch.shape)
    if (r, c) == (0, 0):
        return "id"
    if (r, c) == (0, w - 1):
        return "h"
    if (r, c) == (h - 1, 0):
        return "v"
    if (r, c) == (h - 1, w - 1):
        return "r"
    raise AssertionError("blue anchor must sit on a slot corner")


def _majority_patch(patches: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(patches, axis=0)
    k, h, w = map(int, stack.shape)
    out = np.empty((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            vals, freqs = np.unique(stack[:, r, c], return_counts=True)
            best = int(freqs.max())
            winners = vals[freqs == best]
            assert len(winners) == 1, "majority vote must be unique"
            out[r, c] = int(winners[0])
    assert k >= 4
    return out


def _raw_vote(patches: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(patches, axis=0)
    h, w = map(int, stack.shape[1:])
    out = np.full((h, w), -1, dtype=int)
    for r in range(h):
        for c in range(w):
            vals, freqs = np.unique(stack[:, r, c], return_counts=True)
            best = int(freqs.max())
            if np.count_nonzero(freqs == best) != 1:
                continue
            out[r, c] = int(vals[np.argmax(freqs)])
    return out


def _slot_bounds_from_bands(row_bands: list[tuple[int, int]], col_bands: list[tuple[int, int]]) -> list[tuple[int, int, int, int]]:
    return [(r0, r1, c0, c1) for r0, r1 in row_bands for c0, c1 in col_bands]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the clean prototype by orientation-normalized voting, then repaint each slot in its own original orientation.

    Input structure:
    - The dominant color is the background.
    - Non-background cells form a rectangular array of equal-sized dense slot bands, plus sparse noise outside the array.
    - Every slot contains a damaged copy of one shared multicolor prototype, transformed by horizontal mirror, vertical mirror, or 180-degree rotation.
    - The prototype has exactly one blue anchor pixel, and that anchor is always preserved on one corner of the slot.

    Rule:
    - Detect the slot row bands and slot column bands from the dense non-background rows and columns.
    - Crop each slot and use the blue anchor corner to determine its orientation, then transform the slot back to the standard orientation.
    - Take a cellwise unique majority vote over the aligned slots to recover the clean prototype.
    - Clear the whole canvas to background and paste the clean prototype back into every slot using that slot's original orientation.

    Key insight:
    - Voting is only correct after all copies are normalized into the anchor-defined coordinate frame; the outer sparse noise and the mixed slot orientations must both be ignored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    fg = g != bg
    row_bands = _select_dense_bands(np.count_nonzero(fg, axis=1))
    col_bands = _select_dense_bands(np.count_nonzero(fg, axis=0))

    slot_bounds = _slot_bounds_from_bands(row_bands, col_bands)
    slots: list[np.ndarray] = []
    orientations: list[str] = []
    shapes = set()

    for r0, r1, c0, c1 in slot_bounds:
        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        shapes.add(tuple(map(int, patch.shape)))
        orientation = _anchor_corner(patch)
        slots.append(_apply_orientation(patch, orientation))
        orientations.append(orientation)

    assert len(shapes) == 1, "all slots must share one rectangle size"
    prototype = _majority_patch(slots)

    out = np.full_like(g, bg)
    for (r0, r1, c0, c1), orientation in zip(slot_bounds, orientations):
        patch = _apply_orientation(prototype, orientation)
        assert patch.shape == (r1 - r0 + 1, c1 - c0 + 1)
        out[r0 : r1 + 1, c0 : c1 + 1] = patch
    return out


def _sample_margins(rng: np.random.Generator, total_extra: int) -> tuple[int, int]:
    left = int(rng.integers(1, total_extra))
    right = int(rng.integers(1, total_extra - left + 1))
    if left + right < 2:
        return _sample_margins(rng, total_extra)
    return left, right


def _sample_layout(rng: np.random.Generator) -> dict | None:
    slot_h = int(rng.integers(4, 7))
    slot_w = int(rng.integers(4, 7))
    slot_rows = int(rng.integers(2, 4))
    slot_cols = int(rng.integers(2, 4))
    gap_r = int(rng.integers(1, 4))
    gap_c = int(rng.integers(1, 4))

    array_h = slot_rows * slot_h + (slot_rows - 1) * gap_r
    array_w = slot_cols * slot_w + (slot_cols - 1) * gap_c
    if array_h > 28 or array_w > 28:
        return None

    extra_h = int(rng.integers(2, 31 - array_h))
    extra_w = int(rng.integers(2, 31 - array_w))
    top, bottom = _sample_margins(rng, extra_h)
    left, right = _sample_margins(rng, extra_w)

    return {
        "slot_h": slot_h,
        "slot_w": slot_w,
        "slot_rows": slot_rows,
        "slot_cols": slot_cols,
        "gap_r": gap_r,
        "gap_c": gap_c,
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
        "height": array_h + top + bottom,
        "width": array_w + left + right,
    }


def _prototype(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    palette_size = int(rng.integers(3, 7))
    palette = rng.choice(np.array(NON_BLUE_COLORS, dtype=int), size=palette_size, replace=False)
    patch = rng.choice(palette, size=(h, w), replace=True).astype(int)
    patch[0, 0] = BLUE

    used = {int(v) for v in patch.ravel()}
    if len(used) < 3:
        return _prototype(rng, h, w)
    return patch


def _slot_positions(layout: dict) -> list[tuple[int, int]]:
    positions = []
    for rr in range(layout["slot_rows"]):
        r0 = layout["top"] + rr * (layout["slot_h"] + layout["gap_r"])
        for cc in range(layout["slot_cols"]):
            c0 = layout["left"] + cc * (layout["slot_w"] + layout["gap_c"])
            positions.append((r0, c0))
    return positions


def _damage_patch(
    rng: np.random.Generator,
    patch: np.ndarray,
    orientation: str,
    max_changes: int = 3,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    out = patch.copy()
    h, w = map(int, patch.shape)
    anchor_r, anchor_c = ORIENTATION_TO_CORNER[orientation]
    if anchor_r < 0:
        anchor_r += h
    if anchor_c < 0:
        anchor_c += w

    cells = [(r, c) for r in range(h) for c in range(w) if (r, c) != (anchor_r, anchor_c)]
    rng.shuffle(cells)
    change_count = int(rng.integers(1, min(max_changes, len(cells)) + 1))
    changed = []
    for r, c in cells[:change_count]:
        choices = [color for color in NON_BLUE_COLORS if color != int(out[r, c])]
        out[r, c] = int(rng.choice(np.array(choices, dtype=int)))
        changed.append((r, c))
    return out, changed


def _vote_winners_by_cell(normalized_slots: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(normalized_slots, axis=0)
    h, w = map(int, stack.shape[1:])
    winners = np.full((h, w), -1, dtype=int)
    for r in range(h):
        for c in range(w):
            vals, counts = np.unique(stack[:, r, c], return_counts=True)
            best = int(counts.max())
            if np.count_nonzero(counts == best) != 1:
                return winners
            winners[r, c] = int(vals[np.argmax(counts)])
    return winners


def _expected_from_metadata(meta: dict) -> np.ndarray:
    out = np.full((meta["height"], meta["width"]), BG, dtype=int)
    for (r0, c0), orientation in zip(meta["positions"], meta["orientations"]):
        patch = _apply_orientation(meta["prototype"], orientation)
        h, w = patch.shape
        out[r0 : r0 + h, c0 : c0 + w] = patch
    return out


def _generate_case(rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng() if rng is None else rng

    for _ in range(400):
        layout = _sample_layout(rng)
        if layout is None:
            continue

        prototype = _prototype(rng, layout["slot_h"], layout["slot_w"])
        slot_count = layout["slot_rows"] * layout["slot_cols"]
        orientations = ["id", "h", "v", "r"]
        while len(orientations) < slot_count:
            orientations.append(str(rng.choice(np.array(["id", "h", "v", "r"], dtype=object))))
        rng.shuffle(orientations)

        positions = _slot_positions(layout)
        damaged_slots = []
        normalized_slots = []
        changed_cells = []
        any_perfect = False

        for orientation in orientations:
            clean = _apply_orientation(prototype, orientation)
            damaged, changed = _damage_patch(rng, clean, orientation)
            if np.array_equal(clean, damaged):
                any_perfect = True
            damaged_slots.append(damaged)
            normalized_slots.append(_apply_orientation(damaged, orientation))
            changed_cells.append(changed)

        if any_perfect:
            continue

        voted = _vote_winners_by_cell(normalized_slots)
        if np.any(voted < 0) or not np.array_equal(voted, prototype):
            continue
        if np.array_equal(_raw_vote(damaged_slots), prototype):
            continue

        inp = np.full((layout["height"], layout["width"]), BG, dtype=int)
        for (r0, c0), patch in zip(positions, damaged_slots):
            h, w = patch.shape
            inp[r0 : r0 + h, c0 : c0 + w] = patch

        occupied = np.zeros_like(inp, dtype=bool)
        for r0, c0 in positions:
            occupied[r0 : r0 + layout["slot_h"], c0 : c0 + layout["slot_w"]] = True

        free = np.argwhere(~occupied)
        if len(free) == 0:
            continue
        noise_count = int(rng.integers(1, min(9, len(free)) + 1))
        for rr, cc in free[rng.choice(len(free), size=noise_count, replace=False)]:
            inp[int(rr), int(cc)] = int(rng.choice(np.array(NON_BLUE_COLORS, dtype=int)))

        if _get_bg(inp) != BG:
            continue

        expected = np.full_like(inp, BG)
        for (r0, c0), orientation in zip(positions, orientations):
            clean = _apply_orientation(prototype, orientation)
            expected[r0 : r0 + layout["slot_h"], c0 : c0 + layout["slot_w"]] = clean

        meta = dict(layout)
        meta.update(
            {
                "prototype": prototype,
                "positions": positions,
                "orientations": orientations,
                "changed_cells": changed_cells,
                "slot_bounds": [
                    (r0, r0 + layout["slot_h"] - 1, c0, c0 + layout["slot_w"] - 1)
                    for r0, c0 in positions
                ],
            }
        )

        row_counts = np.count_nonzero(inp != BG, axis=1)
        col_counts = np.count_nonzero(inp != BG, axis=0)
        try:
            row_bands = _select_dense_bands(row_counts)
            col_bands = _select_dense_bands(col_counts)
        except AssertionError:
            continue

        expected_row_bands = []
        for rr in range(layout["slot_rows"]):
            r0 = layout["top"] + rr * (layout["slot_h"] + layout["gap_r"])
            expected_row_bands.append((r0, r0 + layout["slot_h"] - 1))
        expected_col_bands = []
        for cc in range(layout["slot_cols"]):
            c0 = layout["left"] + cc * (layout["slot_w"] + layout["gap_c"])
            expected_col_bands.append((c0, c0 + layout["slot_w"] - 1))
        if row_bands != expected_row_bands or col_bands != expected_col_bands:
            continue

        if not np.array_equal(_expected_from_metadata(meta), expected):
            continue
        try:
            actual = main(inp)
        except Exception:
            continue
        if not np.array_equal(actual, expected):
            continue
        return inp, expected, meta

    raise RuntimeError("failed to generate a valid orientation-voting case")


def generate_input() -> np.ndarray:
    inp, _expected, _meta = _generate_case()
    return inp
