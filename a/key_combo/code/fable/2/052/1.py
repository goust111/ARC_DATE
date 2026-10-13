from __future__ import annotations

import random

import numpy as np


UNKNOWN = -1
GEN_ATTEMPTS = 400


def _merge_known_rows(known: np.ndarray, column: np.ndarray, noise: int) -> np.ndarray | None:
    merged = known.copy()
    for r, value in enumerate(column):
        v = int(value)
        if v == int(noise):
            continue
        if int(merged[r]) == UNKNOWN:
            merged[r] = v
        elif int(merged[r]) != v:
            return None
    return merged


def _partition_bands(grid: np.ndarray, noise: int) -> list[dict[str, object]] | None:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    if width == 0:
        return None

    bands: list[dict[str, object]] = []
    start = 0
    known = np.full(height, UNKNOWN, dtype=int)
    known = _merge_known_rows(known, g[:, 0], noise)
    if known is None:
        return None

    for c in range(1, width):
        merged = _merge_known_rows(known, g[:, c], noise)
        if merged is None:
            bands.append({"start": start, "end": c - 1, "known": known})
            start = c
            known = np.full(height, UNKNOWN, dtype=int)
            known = _merge_known_rows(known, g[:, c], noise)
            if known is None:
                return None
        else:
            known = merged

    bands.append({"start": start, "end": width - 1, "known": known})
    return bands


def _resolve_band_rows(bands: list[dict[str, object]]) -> list[np.ndarray] | None:
    if not bands:
        return None

    height = len(bands[0]["known"])
    resolved: list[np.ndarray | None] = [None] * len(bands)
    left = 0
    right = len(bands) - 1

    while left <= right:
        left_known = np.asarray(bands[left]["known"], dtype=int)
        right_known = np.asarray(bands[right]["known"], dtype=int)
        rows = np.full(height, UNKNOWN, dtype=int)
        for r in range(height):
            a = int(left_known[r])
            b = int(right_known[r])
            if a != UNKNOWN and b != UNKNOWN and a != b:
                return None
            if a != UNKNOWN:
                rows[r] = a
            elif b != UNKNOWN:
                rows[r] = b
            else:
                return None
        resolved[left] = rows.copy()
        resolved[right] = rows.copy()
        left += 1
        right -= 1

    return [np.asarray(x, dtype=int) for x in resolved]


def _repair_grid(grid: np.ndarray, noise: int, bands: list[dict[str, object]], resolved: list[np.ndarray]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for band_index, band in enumerate(bands):
        start = int(band["start"])
        end = int(band["end"])
        row_colors = resolved[band_index]
        for c in range(start, end + 1):
            noise_rows = out[:, c] == int(noise)
            out[noise_rows, c] = row_colors[noise_rows]
    return out


def _compress_exact_columns(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    if g.shape[1] == 0:
        return []
    compressed = [g[:, 0].copy()]
    for c in range(1, g.shape[1]):
        if not np.array_equal(g[:, c], g[:, c - 1]):
            compressed.append(g[:, c].copy())
    return compressed


def _evaluate_candidate(grid: np.ndarray, noise: int) -> dict[str, object] | None:
    g = np.asarray(grid, dtype=int)
    if not np.any(g == int(noise)):
        return None

    bands = _partition_bands(g, noise)
    if bands is None or len(bands) < 3:
        return None

    resolved = _resolve_band_rows(bands)
    if resolved is None:
        return None

    out = _repair_grid(g, noise, bands, resolved)
    compressed = _compress_exact_columns(out)
    if len(compressed) != len(bands):
        return None

    for i in range(len(compressed)):
        if not np.array_equal(compressed[i], compressed[-1 - i]):
            return None

    return {
        "noise": int(noise),
        "bands": bands,
        "resolved": resolved,
        "output": out,
        "band_count": len(bands),
        "noise_count": int(np.count_nonzero(g == int(noise))),
    }


def _valid_candidates(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    candidates = []
    for color in np.unique(g):
        result = _evaluate_candidate(g, int(color))
        if result is not None:
            candidates.append(result)
    candidates.sort(key=lambda item: (-int(item["band_count"]), int(item["noise_count"]), int(item["noise"])))
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair noise-covered pixels by mirroring their band index in the run-compressed column sequence.

    Input structure:
    - The grid is made from a smaller left-right symmetric column pattern whose columns were stretched into
      variable-width vertical bands of repeated columns.
    - A small number of pixels are overwritten by one noise color that does not belong to the original pattern.
    - Mirror-matched bands can have different pixel widths, so raw pixel coordinates are not symmetric.

    Rule:
    - Treat each candidate color as possible noise, ignore it, and merge adjacent columns while their remaining
      visible row colors stay compatible; this recovers the repeated-column bands.
    - Keep the candidate whose recovered band sequence is left-right symmetric, then pair each band with the
      mirror band of the same band index.
    - For every noise pixel, copy the color from the mirror band on the same row, leaving all non-noise pixels unchanged.

    Key insight:
    - The symmetry lives in compressed band-index space, not in raw pixel coordinates: you must compress repeated
      columns first, mirror the band indices, and only then map back to the stretched columns.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    candidates = _valid_candidates(g)
    assert candidates, "failed to identify a valid noise color"
    assert len(candidates) == 1, "noise color must be uniquely determined"
    return np.asarray(candidates[0]["output"], dtype=int)


def _sample_band_columns(height: int, band_count: int, palette: list[int]) -> list[np.ndarray]:
    half = band_count // 2
    for _ in range(200):
        left: list[np.ndarray] = []
        prev = None
        ok = True
        for _idx in range(half):
            found = False
            for _inner in range(200):
                col = np.array([random.choice(palette) for _ in range(height)], dtype=int)
                if prev is not None and np.count_nonzero(col != prev) < 2:
                    continue
                left.append(col)
                prev = col
                found = True
                break
            if not found:
                ok = False
                break
        if not ok:
            continue

        center = None
        for _inner in range(200):
            col = np.array([random.choice(palette) for _ in range(height)], dtype=int)
            if prev is not None and np.count_nonzero(col != prev) < 2:
                continue
            center = col
            break
        if center is None:
            continue

        columns = left + [center] + [col.copy() for col in left[::-1]]
        if all(np.any(columns[i] != columns[i + 1]) for i in range(len(columns) - 1)):
            return columns
    raise RuntimeError("failed to sample distinct symmetric band columns")


def _sample_widths(band_count: int, require_unequal: bool) -> list[int]:
    half = band_count // 2
    for _ in range(300):
        widths = [0] * band_count
        unequal_pairs = 0
        for i in range(half):
            left_width = random.randint(1, 4)
            if require_unequal:
                right_width = random.randint(1, 4)
            else:
                right_width = left_width
            widths[i] = left_width
            widths[-1 - i] = right_width
            if left_width != right_width:
                unequal_pairs += 1
        widths[half] = random.randint(2, 5)
        total_width = sum(widths)
        if total_width > 30:
            continue
        if require_unequal and unequal_pairs == 0:
            continue
        if not require_unequal and unequal_pairs != 0:
            continue
        return widths
    raise RuntimeError("failed to sample valid band widths")


def _band_ranges(widths: list[int]) -> list[tuple[int, int]]:
    ranges = []
    start = 0
    for width in widths:
        end = start + width - 1
        ranges.append((start, end))
        start = end + 1
    return ranges


def _render_clean(columns: list[np.ndarray], widths: list[int]) -> np.ndarray:
    height = len(columns[0])
    total_width = sum(widths)
    out = np.zeros((height, total_width), dtype=int)
    c0 = 0
    for col, width in zip(columns, widths):
        out[:, c0 : c0 + width] = col[:, None]
        c0 += width
    return out


def _safe_rows(columns: list[np.ndarray]) -> list[set[int]]:
    band_count = len(columns)
    per_band = [set() for _ in range(band_count)]
    for idx in range(band_count):
        per_band[idx].add(random.randrange(len(columns[0])))

    for idx in range(band_count - 1):
        diffs = np.flatnonzero(columns[idx] != columns[idx + 1])
        assert diffs.size > 0
        row = int(random.choice(diffs.tolist()))
        per_band[idx].add(row)
        per_band[idx + 1].add(row)
    return per_band


def _noiseable_rows(height: int, blocked: set[int]) -> list[int]:
    return [r for r in range(height) if r not in blocked]


def _add_pair_noise(
    grid: np.ndarray,
    widths: list[int],
    ranges: list[tuple[int, int]],
    safe_rows: list[set[int]],
    noise_color: int,
    pair_index: int,
    must_place: bool,
) -> int:
    right_index = len(widths) - 1 - pair_index
    height = grid.shape[0]
    options = _noiseable_rows(height, safe_rows[pair_index] | safe_rows[right_index])
    if not options:
        return 0

    random.shuffle(options)
    placed = 0
    target_rows = options[: random.randint(1, min(3, len(options)))]
    for row in target_rows:
        chosen_band = pair_index if random.random() < 0.5 else right_index
        start, end = ranges[chosen_band]
        width = end - start + 1
        take = random.randint(1, width)
        cols = random.sample(range(start, end + 1), take)
        grid[row, cols] = noise_color
        placed += len(cols)

    if must_place and placed == 0:
        row = options[0]
        chosen_band = pair_index
        start, _end = ranges[chosen_band]
        grid[row, start] = noise_color
        placed += 1
    return placed


def _add_center_noise(
    grid: np.ndarray,
    widths: list[int],
    ranges: list[tuple[int, int]],
    safe_rows: list[set[int]],
    noise_color: int,
) -> int:
    center = len(widths) // 2
    start, end = ranges[center]
    width = end - start + 1
    if width < 2:
        return 0

    options = _noiseable_rows(grid.shape[0], safe_rows[center])
    if not options or random.random() < 0.5:
        return 0

    placed = 0
    rows = random.sample(options, k=random.randint(1, min(2, len(options))))
    for row in rows:
        take = random.randint(1, width - 1)
        cols = random.sample(range(start, end + 1), take)
        grid[row, cols] = noise_color
        placed += len(cols)
    return placed


def _naive_pixel_mirror_restore(grid: np.ndarray, noise: int) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    width = g.shape[1]
    for r, c in np.argwhere(g == int(noise)):
        out[int(r), int(c)] = g[int(r), width - 1 - int(c)]
    return out


def _build_case() -> tuple[np.ndarray, np.ndarray]:
    height = random.randint(6, 12)
    band_count = random.choice([5, 7, 9])
    require_unequal = random.random() < 0.8

    palette = random.sample(list(range(1, 10)), k=random.randint(3, 5))
    columns = _sample_band_columns(height, band_count, palette)
    widths = _sample_widths(band_count, require_unequal=require_unequal)
    ranges = _band_ranges(widths)
    clean = _render_clean(columns, widths)
    safe_rows = _safe_rows(columns)

    noise_color = random.choice([c for c in range(1, 10) if c not in palette])
    inp = clean.copy()

    half = band_count // 2
    unequal_pairs = [i for i in range(half) if widths[i] != widths[-1 - i]]
    forced_pair = random.choice(unequal_pairs) if unequal_pairs else None

    total_noise = 0
    for i in range(half):
        must_place = forced_pair is not None and i == forced_pair
        if must_place or random.random() < 0.7:
            total_noise += _add_pair_noise(inp, widths, ranges, safe_rows, noise_color, i, must_place)

    total_noise += _add_center_noise(inp, widths, ranges, safe_rows, noise_color)
    if total_noise == 0:
        total_noise += _add_pair_noise(inp, widths, ranges, safe_rows, noise_color, random.randrange(half), True)

    if total_noise == 0:
        raise RuntimeError("failed to place noise")

    return inp, clean


def _validate_case(inp: np.ndarray, clean: np.ndarray) -> bool:
    candidates = _valid_candidates(inp)
    if len(candidates) != 1:
        return False

    result = candidates[0]
    noise = int(result["noise"])
    if not np.array_equal(np.asarray(result["output"], dtype=int), clean):
        return False
    if not np.array_equal(main(inp), clean):
        return False

    bands = result["bands"]
    for band in bands:
        start = int(band["start"])
        end = int(band["end"])
        clean_rows = np.all(inp[:, start : end + 1] != noise, axis=1)
        if not np.any(clean_rows):
            return False

    widths = [int(band["end"]) - int(band["start"]) + 1 for band in bands]
    unequal_exists = any(widths[i] != widths[-1 - i] for i in range(len(widths) // 2))
    if unequal_exists:
        naive = _naive_pixel_mirror_restore(inp, noise)
        if np.array_equal(naive, clean):
            return False
        mismatched = False
        for r, c in np.argwhere(inp == noise):
            if int(clean[int(r), int(c)]) != int(naive[int(r), int(c)]):
                mismatched = True
                break
        if not mismatched:
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        inp, clean = _build_case()
        if _validate_case(inp, clean):
            return inp
    raise RuntimeError("failed to generate a valid band-mirror repair input")
