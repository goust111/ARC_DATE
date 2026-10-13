from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
BODY = 2
STEP = 4
PATCH_SIZE = 3
TOTAL_SLOT_CHOICES = (4, 5)
PERIOD_CHOICES = (2, 3)
CENTER_POOL = (1, 3, 4, 6, 8, 9)
MAX_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components8(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr = rr + dr
                        nc = cc + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _extract_tiles(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    tiles: list[dict] = []
    for coords in _components8(g != bg):
        b = _bbox(coords)
        r0, c0, r1, c1 = b
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        assert patch.shape == (PATCH_SIZE, PATCH_SIZE)
        assert coords.shape[0] == PATCH_SIZE * PATCH_SIZE
        tiles.append({"top": r0, "left": c0, "patch": patch.copy()})
    assert 3 <= len(tiles) <= 4
    return sorted(tiles, key=lambda t: t["left"])


def _slot_columns(tiles: list[dict]) -> tuple[list[int], int]:
    cols = [int(tile["left"]) for tile in tiles]
    assert cols == sorted(cols)
    first = cols[0]
    last = cols[-1]
    assert (last - first) % STEP == 0
    total_slots = (last - first) // STEP + 1
    assert total_slots in TOTAL_SLOT_CHOICES
    slot_cols = [first + STEP * i for i in range(total_slots)]
    missing = [idx for idx, col in enumerate(slot_cols) if col not in set(cols)]
    assert len(missing) == 1
    missing_index = missing[0]
    assert 0 < missing_index < total_slots - 1
    return slot_cols, missing_index


def _infer_full_tops(sequence: list[int | None]) -> tuple[int, list[int]]:
    assert sequence.count(None) == 1
    missing_index = sequence.index(None)
    candidates: list[tuple[int, list[int]]] = []
    for period in PERIOD_CHOICES:
        if period >= len(sequence):
            continue
        residue_to_value: dict[int, int] = {}
        ok = True
        for idx, value in enumerate(sequence):
            if value is None:
                continue
            residue = idx % period
            if residue in residue_to_value and residue_to_value[residue] != value:
                ok = False
                break
            residue_to_value[residue] = int(value)
        if not ok or (missing_index % period) not in residue_to_value:
            continue
        filled = [
            int(value if value is not None else residue_to_value[idx % period])
            for idx, value in enumerate(sequence)
        ]
        candidates.append((period, filled))
    assert candidates
    shortest = min(period for period, _filled in candidates)
    best = [filled for period, filled in candidates if period == shortest]
    assert len(best) == 1
    return shortest, best[0]


def _missing_patch(tiles: list[dict], missing_top: int) -> np.ndarray:
    exemplar: np.ndarray | None = None
    for tile in tiles:
        if int(tile["top"]) != int(missing_top):
            continue
        if exemplar is None:
            exemplar = np.array(tile["patch"], copy=True)
        else:
            assert np.array_equal(exemplar, tile["patch"])
    assert exemplar is not None
    return exemplar


def _solve(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    tiles = _extract_tiles(grid)
    slot_cols, missing_index = _slot_columns(tiles)
    by_col = {int(tile["left"]): tile for tile in tiles}
    top_sequence = [by_col[col]["top"] if col in by_col else None for col in slot_cols]
    period, full_tops = _infer_full_tops(top_sequence)
    predicted_top = int(full_tops[missing_index])
    patch_for_missing = _missing_patch(tiles, predicted_top)

    full_tiles: list[dict] = []
    for idx, col in enumerate(slot_cols):
        if col in by_col:
            patch = np.array(by_col[col]["patch"], copy=True)
            top = int(by_col[col]["top"])
        else:
            patch = patch_for_missing.copy()
            top = predicted_top
        full_tiles.append({"index": idx, "top": top, "patch": patch})

    order = sorted(range(len(full_tiles)), key=lambda idx: (full_tiles[idx]["top"], full_tiles[idx]["index"]))
    output = np.vstack([full_tiles[idx]["patch"] for idx in order])
    return output, {
        "period": period,
        "missing_index": missing_index,
        "full_tops": full_tops,
        "order": order,
        "slot_count": len(slot_cols),
        "predicted_top": predicted_top,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Extrapolate the missing vertical offset in an equally spaced motif row, then stack the full set by offset.

    Input structure:
    - The grid has a black background and 4 or 5 equally spaced slot columns separated by 4 cells.
    - Exactly one interior slot is empty; every other slot contains a 3x3 motif, and all motifs share the same outer shape.
    - Reading left to right, the motifs' top-row positions follow a shortest repeating period of length 2 or 3.

    Rule:
    - Detect the occupied 3x3 motifs and recover the full slot list, including the missing interior slot.
    - Infer the missing slot's top row by continuing the shortest repeating top-row pattern, then copy a motif from the same offset class into that slot.
    - Sort the complete set of slot motifs by increasing top row, breaking ties by original left-to-right slot order, and stack them vertically with no gap.

    Key insight:
    - The empty slot must be filled from the periodic offset pattern first, because the reconstructed motif becomes part of the final offset-sorted stack.
    """
    out, _meta = _solve(np.asarray(grid, dtype=int))
    return out


def _make_patch(center_color: int) -> np.ndarray:
    patch = np.full((PATCH_SIZE, PATCH_SIZE), BODY, dtype=int)
    patch[1, 1] = int(center_color)
    return patch


def _period3_missing_choices(total_slots: int) -> list[int]:
    choices = []
    for idx in range(1, total_slots - 1):
        residue = idx % 3
        peers = [j for j in range(total_slots) if j % 3 == residue and j != idx]
        if peers:
            choices.append(idx)
    return choices


def _expected_from_meta(meta: dict) -> np.ndarray:
    ordered = sorted(meta["tiles"], key=lambda tile: (tile["top"], tile["index"]))
    return np.vstack([tile["patch"] for tile in ordered])


def _chosen_period(full_tops: list[int]) -> int:
    for period in PERIOD_CHOICES:
        if period >= len(full_tops):
            continue
        if all(full_tops[idx] == full_tops[idx % period] for idx in range(len(full_tops))):
            return period
    raise AssertionError("no valid period")


def _build_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray, dict]:
    period = rng.choice(PERIOD_CHOICES)
    total_slots = 5 if period == 3 else rng.choice(TOTAL_SLOT_CHOICES)
    missing_choices = _period3_missing_choices(total_slots) if period == 3 else list(range(1, total_slots - 1))
    missing_index = rng.choice(missing_choices)

    offset_values = rng.sample(range(5), period)
    rng.shuffle(offset_values)
    top_pattern = [offset_values[idx % period] for idx in range(total_slots)]

    center_colors = rng.sample(CENTER_POOL, period)
    residue_to_patch = {
        residue: _make_patch(center_colors[residue]) for residue in range(period)
    }

    base_top = rng.randint(1, 6)
    full_tops = [base_top + top_pattern[idx] for idx in range(total_slots)]
    chosen = _chosen_period(full_tops)
    if chosen != period:
        raise AssertionError("accidentally created a shorter period")

    order = sorted(range(total_slots), key=lambda idx: (full_tops[idx], idx))
    if order == list(range(total_slots)):
        raise AssertionError("sorting must differ from left-to-right order")
    missing_rank = order.index(missing_index)
    if missing_rank in (0, total_slots - 1):
        raise AssertionError("missing motif should land inside the sorted stack")

    left_margin = rng.randint(0, 3)
    right_margin = rng.randint(0, 3)
    width = left_margin + STEP * (total_slots - 1) + PATCH_SIZE + right_margin
    height = max(full_tops) + PATCH_SIZE + rng.randint(0, 3)
    if max(height, width) > 30:
        raise AssertionError("grid too large")

    grid = np.zeros((height, width), dtype=int)
    tiles_meta: list[dict] = []
    for idx in range(total_slots):
        top = full_tops[idx]
        left = left_margin + STEP * idx
        patch = residue_to_patch[idx % period]
        tiles_meta.append({"index": idx, "top": top, "left": left, "patch": patch.copy()})
        if idx == missing_index:
            continue
        grid[top : top + PATCH_SIZE, left : left + PATCH_SIZE] = patch

    expected = np.vstack([tiles_meta[idx]["patch"] for idx in order])
    meta = {
        "period": period,
        "total_slots": total_slots,
        "missing_index": missing_index,
        "full_tops": full_tops,
        "order": order,
        "tiles": tiles_meta,
    }
    return grid, expected, meta


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        try:
            grid, expected, meta = _build_case(rng)
        except AssertionError:
            continue

        try:
            actual, solved = _solve(grid)
        except AssertionError:
            continue

        if not np.array_equal(actual, expected):
            continue

        if solved["period"] != meta["period"]:
            continue
        if solved["slot_count"] != meta["total_slots"]:
            continue
        if solved["missing_index"] != meta["missing_index"]:
            continue
        if solved["full_tops"] != meta["full_tops"]:
            continue
        if solved["order"] != meta["order"]:
            continue

        # Non-degeneracy: the reconstructed missing motif must matter, not just the observed ones.
        observed_tiles = [tile for tile in meta["tiles"] if tile["index"] != meta["missing_index"]]
        observed_only = np.vstack(
            [tile["patch"] for tile in sorted(observed_tiles, key=lambda tile: (tile["top"], tile["index"]))]
        )
        if observed_only.shape == expected.shape and np.array_equal(observed_only, expected):
            continue

        # The shortest valid period must be the designed one, and the solver must use it.
        if _chosen_period(meta["full_tops"]) != meta["period"]:
            continue

        return grid
    raise RuntimeError("failed to generate a valid periodic extrapolation stack input")
