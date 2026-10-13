from __future__ import annotations

import hashlib
import random
from typing import Optional

import numpy as np


NON_BLACK_COLORS = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 800
_SEEN_INPUT_SIGNATURES: set[str] = set()
_SEEN_OUTPUT_SIGNATURES: set[str] = set()


def _grid_signature(grid: np.ndarray) -> str:
    g = np.asarray(grid, dtype=int)
    return hashlib.md5(g.tobytes() + str(g.shape).encode()).hexdigest()


def _bbox(mask: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_full_rectangle(mask: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    return bool(mask.sum() == (r1 - r0 + 1) * (c1 - c0 + 1) and mask[r0 : r1 + 1, c0 : c1 + 1].all())


def _has_row_period(tile: np.ndarray, period: int) -> bool:
    return bool(np.array_equal(tile[:-period, :], tile[period:, :]))


def _has_col_period(tile: np.ndarray, period: int) -> bool:
    return bool(np.array_equal(tile[:, :-period], tile[:, period:]))


def _candidate_key(candidate: dict) -> tuple[int, int, int, int]:
    return (
        int(candidate["period_area"]),
        int(candidate["period_sum"]),
        -int(candidate["margin_score"]),
        int(candidate["bbox_area"]),
    )


def _evaluate_period(grid: np.ndarray, period_r: int, period_c: int) -> Optional[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    buckets = [[{} for _ in range(period_c)] for _ in range(period_r)]
    for r in range(h):
        for c in range(w):
            bucket = buckets[r % period_r][c % period_c]
            color = int(g[r, c])
            bucket[color] = bucket.get(color, 0) + 1

    template = np.zeros((period_r, period_c), dtype=int)
    margin_score = 0
    for rr in range(period_r):
        for cc in range(period_c):
            items = sorted(buckets[rr][cc].items(), key=lambda kv: (-kv[1], kv[0]))
            if not items:
                return None
            if len(items) > 1 and items[0][1] == items[1][1]:
                return None
            template[rr, cc] = int(items[0][0])
            margin_score += int(items[0][1]) - int(items[1][1] if len(items) > 1 else 0)

    expected = np.empty_like(g)
    for r in range(h):
        for c in range(w):
            expected[r, c] = template[r % period_r, c % period_c]

    mismatch = g != expected
    box = _bbox(mismatch)
    if box is None or not _is_full_rectangle(mismatch, box):
        return None

    r0, c0, r1, c1 = box
    return {
        "period": (int(period_r), int(period_c)),
        "period_area": int(period_r * period_c),
        "period_sum": int(period_r + period_c),
        "bbox": box,
        "bbox_area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
        "expected": expected,
        "output": expected[r0 : r1 + 1, c0 : c1 + 1].copy(),
        "margin_score": int(margin_score),
    }


def _all_candidates(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[dict] = []
    for period_r in range(1, h):
        for period_c in range(1, w):
            if h // period_r < 2 or w // period_c < 2:
                continue
            candidate = _evaluate_period(g, period_r, period_c)
            if candidate is not None:
                candidates.append(candidate)
    candidates.sort(key=_candidate_key)
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reveal the covered rectangle by finding where a periodic tiling breaks, then restore and crop it.

    Input structure:
    - The whole grid is a fully colored translational tiling with no explicit missing-marker color.
    - One axis-aligned rectangular intruder patch overwrites some cells of that tiling.
    - Every intruder cell differs from the hidden periodic color underneath it.

    Rule:
    - Detect the row and column translation periods that make almost all orbit copies agree.
    - For each residue class under that period pair, take the unique majority visible color as the
      true tiled color and rebuild the full expected periodic image.
    - The cells where the observed grid disagrees with that rebuilt image form one filled rectangle;
      output the restored colors inside exactly that rectangle.

    Key insight:
    - The patch is invisible to marker-based methods, but it becomes visible as the only rectangular
      set of violations against the dominant translational orbit colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    candidates = _all_candidates(g)
    assert candidates, "failed to detect a valid periodic model"
    return candidates[0]["output"]


def _make_primitive_tile(period_r: int, period_c: int) -> np.ndarray:
    for _ in range(400):
        n_colors = random.randint(3, min(6, len(NON_BLACK_COLORS)))
        palette = random.sample(list(NON_BLACK_COLORS), n_colors)
        tile = np.array(
            [[random.choice(palette) for _ in range(period_c)] for _ in range(period_r)],
            dtype=int,
        )
        if len(np.unique(tile)) < 3:
            continue
        if any(_has_row_period(tile, d) for d in range(1, period_r)):
            continue
        if any(_has_col_period(tile, d) for d in range(1, period_c)):
            continue
        if len({tuple(row.tolist()) for row in tile}) < 2:
            continue
        if len({tuple(col.tolist()) for col in tile.T}) < 2:
            continue
        return tile
    raise RuntimeError("failed to build a primitive tile")


def _count_patch_orbits(
    box: tuple[int, int, int, int], period_r: int, period_c: int
) -> np.ndarray:
    r0, c0, r1, c1 = box
    counts = np.zeros((period_r, period_c), dtype=int)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            counts[r % period_r, c % period_c] += 1
    return counts


def _sample_box(h: int, w: int, period_r: int, period_c: int) -> Optional[tuple[int, int, int, int]]:
    repeats_r = h // period_r
    repeats_c = w // period_c
    orbit_size = repeats_r * repeats_c
    align_mode = random.choice(("aligned", "row_aligned", "col_aligned", "misaligned"))

    for _ in range(500):
        if random.random() < 0.2:
            if random.random() < 0.5:
                box_h = 1
                box_w = random.randint(2, max(2, w - 1))
            else:
                box_h = random.randint(2, max(2, h - 1))
                box_w = 1
        else:
            box_h = random.randint(2, max(2, h - 1))
            box_w = random.randint(2, max(2, w - 1))

        if box_h == h and box_w == w:
            continue
        if box_h * box_w > (h * w) // 2:
            continue

        valid_r = list(range(0, h - box_h + 1))
        valid_c = list(range(0, w - box_w + 1))
        if align_mode in {"aligned", "row_aligned"}:
            aligned_r = [r for r in valid_r if r % period_r == 0]
            if aligned_r:
                valid_r = aligned_r
        else:
            misaligned_r = [r for r in valid_r if r % period_r != 0]
            if misaligned_r:
                valid_r = misaligned_r
        if align_mode in {"aligned", "col_aligned"}:
            aligned_c = [c for c in valid_c if c % period_c == 0]
            if aligned_c:
                valid_c = aligned_c
        else:
            misaligned_c = [c for c in valid_c if c % period_c != 0]
            if misaligned_c:
                valid_c = misaligned_c
        if not valid_r or not valid_c:
            continue

        r0 = random.choice(valid_r)
        c0 = random.choice(valid_c)
        box = (int(r0), int(c0), int(r0 + box_h - 1), int(c0 + box_w - 1))
        counts = _count_patch_orbits(box, period_r, period_c)
        if np.any(counts * 2 >= orbit_size):
            continue
        if counts.max() <= 1 and box_h * box_w > 3 and random.random() < 0.5:
            continue
        return box
    return None


def _make_intruder_patch(expected_patch: np.ndarray) -> np.ndarray:
    h, w = expected_patch.shape
    for _ in range(200):
        tile_h = random.randint(1, min(3, h))
        tile_w = random.randint(1, min(3, w))
        n_colors = random.randint(2 if h * w > 1 else 1, min(5, len(NON_BLACK_COLORS)))
        palette = random.sample(list(NON_BLACK_COLORS), n_colors)
        seed = np.array(
            [[random.choice(palette) for _ in range(tile_w)] for _ in range(tile_h)],
            dtype=int,
        )
        patch = np.tile(seed, (h // tile_h + 1, w // tile_w + 1))[:h, :w].copy()
        for r in range(h):
            for c in range(w):
                if patch[r, c] == expected_patch[r, c]:
                    choices = [color for color in NON_BLACK_COLORS if color != int(expected_patch[r, c])]
                    patch[r, c] = random.choice(choices)
        if np.any(patch == expected_patch):
            continue
        if h * w > 1 and len(np.unique(patch)) < 2:
            continue
        return patch
    raise RuntimeError("failed to build an intruder patch")


def _build_attempt() -> Optional[tuple[np.ndarray, np.ndarray, tuple[int, int, int, int], tuple[int, int]]]:
    for _ in range(300):
        period_r = random.randint(2, 5)
        period_c = random.randint(2, 5)
        repeats_r = random.randint(3, min(5, 30 // period_r))
        repeats_c = random.randint(3, min(5, 30 // period_c))
        h = period_r * repeats_r
        w = period_c * repeats_c
        if h > 30 or w > 30:
            continue

        tile = _make_primitive_tile(period_r, period_c)
        full = np.tile(tile, (repeats_r, repeats_c))
        box = _sample_box(h, w, period_r, period_c)
        if box is None:
            continue

        r0, c0, r1, c1 = box
        expected_patch = full[r0 : r1 + 1, c0 : c1 + 1].copy()
        if expected_patch.size > 1 and len(np.unique(expected_patch)) < 2:
            continue

        intruder = _make_intruder_patch(expected_patch)
        grid = full.copy()
        grid[r0 : r1 + 1, c0 : c1 + 1] = intruder
        return grid, expected_patch, box, (period_r, period_c)
    return None


def _semantic_ok(
    grid: np.ndarray,
    expected_patch: np.ndarray,
    box: tuple[int, int, int, int],
    period: tuple[int, int],
) -> bool:
    period_r, period_c = period
    direct = _evaluate_period(grid, period_r, period_c)
    if direct is None:
        return False
    if direct["bbox"] != box:
        return False
    if not np.array_equal(direct["output"], expected_patch):
        return False

    candidates = _all_candidates(grid)
    if not candidates:
        return False
    best = candidates[0]
    if best["period"] != (period_r, period_c):
        return False
    if best["bbox"] != box:
        return False
    if not np.array_equal(best["output"], expected_patch):
        return False
    if len(candidates) > 1 and _candidate_key(candidates[1]) == _candidate_key(best):
        return False

    mismatch = grid != direct["expected"]
    return bool(_is_full_rectangle(mismatch, box))


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_attempt()
        if built is None:
            continue
        grid, expected_patch, box, period = built
        if not _semantic_ok(grid, expected_patch, box, period):
            continue

        input_sig = _grid_signature(grid)
        output_sig = _grid_signature(expected_patch)
        if input_sig in _SEEN_INPUT_SIGNATURES or output_sig in _SEEN_OUTPUT_SIGNATURES:
            continue

        _SEEN_INPUT_SIGNATURES.add(input_sig)
        _SEEN_OUTPUT_SIGNATURES.add(output_sig)
        return grid
    raise RuntimeError("failed to generate a valid periodic intruder instance")
