from __future__ import annotations

import random

import numpy as np


BG_COLOR = 0
STRIPE_COLORS = (1, 3, 4, 7)
SEED_COLOR = 6
_GEN_COUNTER = 0


def _compress_runs(values: list[int]) -> list[int]:
    out: list[int] = []
    for value in values:
        value = int(value)
        if not out or value != out[-1]:
            out.append(value)
    return out


def _segments(values: list[int]) -> list[dict[str, int]]:
    segments: list[dict[str, int]] = []
    start = 0
    while start < len(values):
        end = start
        color = int(values[start])
        while end + 1 < len(values) and int(values[end + 1]) == color:
            end += 1
        segments.append({"color": color, "start": start, "end": end})
        start = end + 1
    return segments


def _constant_non_seed(values: np.ndarray) -> int | None:
    non_seed = values[values != SEED_COLOR]
    if non_seed.size == 0:
        return None
    uniques = np.unique(non_seed)
    if uniques.size != 1:
        return None
    return int(uniques[0])


def _detect_orientation(grid: np.ndarray) -> tuple[str, list[int]]:
    h, w = grid.shape

    row_colors: list[int] = []
    rows_ok = True
    for r in range(h):
        color = _constant_non_seed(grid[r, :])
        if color is None:
            rows_ok = False
            break
        row_colors.append(color)
    if rows_ok:
        return "horizontal", row_colors

    col_colors: list[int] = []
    for c in range(w):
        color = _constant_non_seed(grid[:, c])
        assert color is not None, "expected vertical full-height stripes"
        col_colors.append(color)
    return "vertical", col_colors


def _analyze(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    orientation, axis_colors = _detect_orientation(g)
    segments = _segments(axis_colors)
    compressed = [int(seg["color"]) for seg in segments]
    assert 3 <= len(set(compressed)) <= 4
    assert 4 <= len(compressed) <= 6

    first_position: dict[int, int] = {}
    first_segment: dict[int, int] = {}
    segment_lookup = np.empty(len(axis_colors), dtype=int)
    for idx, seg in enumerate(segments):
        segment_lookup[seg["start"] : seg["end"] + 1] = idx
        color = int(seg["color"])
        if color not in first_position:
            first_position[color] = len(first_position) + 1
            first_segment[color] = idx

    seeds: list[dict[str, int | bool]] = []
    for r, c in np.argwhere(g == SEED_COLOR):
        axis_index = int(r) if orientation == "horizontal" else int(c)
        owner = int(axis_colors[axis_index])
        seg_idx = int(segment_lookup[axis_index])
        seeds.append(
            {
                "r": int(r),
                "c": int(c),
                "owner": owner,
                "length": int(first_position[owner]),
                "segment_index": seg_idx,
                "later_repeat": bool(seg_idx != first_segment[owner]),
            }
        )
    seeds.sort(key=lambda item: (int(item["r"]), int(item["c"])))

    return {
        "grid": g,
        "orientation": orientation,
        "axis_colors": axis_colors,
        "segments": segments,
        "compressed": compressed,
        "first_position": first_position,
        "seeds": seeds,
    }


def _render_output(grid: np.ndarray, info: dict[str, object] | None = None) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    info = _analyze(g) if info is None else info
    out = g.copy()
    h, w = out.shape
    dr, dc = (0, 1) if info["orientation"] == "horizontal" else (1, 0)

    for seed in info["seeds"]:
        r = int(seed["r"])
        c = int(seed["c"])
        length = int(seed["length"])
        for step in range(1, length + 1):
            rr = r + dr * step
            cc = c + dc * step
            if not (0 <= rr < h and 0 <= cc < w):
                break
            out[rr, cc] = SEED_COLOR
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each magenta seed along the stripe-perpendicular ray whose length is set by the seed color's first position in the compressed stripe sequence.

    Input structure:
    - The grid is filled by either full-width horizontal color bands or full-height vertical color bands using blue, green, yellow, and orange.
    - Band thickness varies from one to three rows or columns, so adjacent rows or columns can repeat the same color before compression.
    - Three to five magenta seed pixels replace stripe cells; a repeated stripe color can host a seed in a later occurrence of that color.

    Rule:
    - Detect stripe orientation by checking whether each row is monochromatic after ignoring magenta seeds; otherwise the stripes are vertical.
    - Recover each seed's owner color from the non-seed cells in its row or column, then read the stripe colors along the band axis and remove consecutive duplicates.
    - For each seed, find the owner's first 1-indexed position in that compressed sequence and draw a magenta ray of that many cells in the positive perpendicular direction: right for horizontal stripes, down for vertical stripes.

    Key insight:
    - Stripe thickness is irrelevant after compression, and even a seed placed in a later repeated segment still uses the owner's first compressed position, so the ray lengths can only be recovered by coupling stripe compression with seed ownership.
    """
    return _render_output(grid)


def _bounded_composition(parts: int, total: int, low: int, high: int) -> list[int]:
    remain = int(total)
    out: list[int] = []
    for i in range(parts):
        left = parts - i - 1
        lo = max(low, remain - high * left)
        hi = min(high, remain - low * left)
        out.append(random.randint(lo, hi))
        remain -= out[-1]
    assert remain == 0
    return out


def _make_compressed_sequence(counter: int) -> list[int]:
    distinct_count = 3 + ((counter // 2) % 2)
    base = random.sample(STRIPE_COLORS, distinct_count)
    random.shuffle(base)
    seq = base[:]
    extra_count = random.randint(1, 6 - distinct_count)

    for _ in range(extra_count):
        candidates: list[tuple[int, int]] = []
        for color in base:
            first_idx = seq.index(color)
            for pos in range(first_idx + 1, len(seq) + 1):
                left = int(seq[pos - 1])
                right = int(seq[pos]) if pos < len(seq) else None
                if left == color or right == color:
                    continue
                candidates.append((color, pos))
        color, pos = random.choice(candidates)
        seq.insert(pos, int(color))

    assert 4 <= len(seq) <= 6
    assert len(set(seq)) == distinct_count
    assert any(seq.count(color) > 1 for color in seq)
    return [int(color) for color in seq]


def _build_base_grid(orientation: str, seq: list[int]) -> tuple[np.ndarray, list[dict[str, int]]]:
    axis_total = random.randint(max(10, len(seq)), min(16, 3 * len(seq)))
    lengths = _bounded_composition(len(seq), axis_total, 1, 3)
    cross_total = random.randint(13, 18)

    axis_colors: list[int] = []
    segments: list[dict[str, int]] = []
    start = 0
    for color, length in zip(seq, lengths):
        axis_colors.extend([int(color)] * int(length))
        segments.append({"color": int(color), "start": start, "end": start + length - 1})
        start += length

    axis_arr = np.array(axis_colors, dtype=int)
    if orientation == "horizontal":
        grid = np.repeat(axis_arr[:, None], cross_total, axis=1)
    else:
        grid = np.repeat(axis_arr[None, :], cross_total, axis=0)
    return grid, segments


def _pick_positions(candidates: list[int], used: list[int]) -> int:
    spaced = [pos for pos in candidates if all(abs(pos - other) >= 2 for other in used)]
    pool = spaced or candidates
    return int(random.choice(pool))


def _seed_segments(seq: list[int]) -> list[int]:
    by_color: dict[int, list[int]] = {}
    for idx, color in enumerate(seq):
        by_color.setdefault(int(color), []).append(int(idx))

    repeated = [color for color, slots in by_color.items() if len(slots) > 1]
    forced_later = random.choice(repeated)

    chosen: list[int] = []
    used: set[int] = set()
    for color in list(dict.fromkeys(seq)):
        options = by_color[int(color)]
        if int(color) == int(forced_later):
            slot = int(random.choice(options[1:]))
        else:
            slot = int(random.choice(options))
        chosen.append(slot)
        used.add(slot)

    target = random.randint(len(chosen), min(5, len(seq)))
    leftovers = [idx for idx in range(len(seq)) if idx not in used]
    random.shuffle(leftovers)
    while len(chosen) < target:
        chosen.append(int(leftovers.pop()))

    random.shuffle(chosen)
    return chosen


def _assign_cross_positions(lengths: list[int], cross_total: int) -> list[int] | None:
    clipped_options = [idx for idx, length in enumerate(lengths) if length > 1]
    if not clipped_options:
        return None

    clipped_idx = max(clipped_options, key=lambda idx: lengths[idx])
    positions = [-1] * len(lengths)
    used: list[int] = []

    clip_inside = random.randint(1, lengths[clipped_idx] - 1)
    clip_pos = cross_total - 1 - clip_inside
    positions[clipped_idx] = int(clip_pos)
    used.append(int(clip_pos))

    order = [idx for idx in range(len(lengths)) if idx != clipped_idx]
    order.sort(key=lambda idx: lengths[idx], reverse=True)
    for idx in order:
        max_pos = cross_total - 1 - lengths[idx]
        candidates = [pos for pos in range(max_pos + 1) if pos not in used]
        if not candidates:
            return None
        pos = _pick_positions(candidates, used)
        positions[idx] = int(pos)
        used.append(int(pos))

    return positions


def _place_seeds(grid: np.ndarray, orientation: str, seq: list[int], segments: list[dict[str, int]]) -> np.ndarray | None:
    out = np.array(grid, dtype=int, copy=True)
    first_position: dict[int, int] = {}
    for color in seq:
        color = int(color)
        if color not in first_position:
            first_position[color] = len(first_position) + 1

    chosen_segments = _seed_segments(seq)
    lengths = [int(first_position[int(seq[idx])]) for idx in chosen_segments]
    cross_total = out.shape[1] if orientation == "horizontal" else out.shape[0]
    cross_positions = _assign_cross_positions(lengths, cross_total)
    if cross_positions is None:
        return None

    for seg_idx, cross_pos in zip(chosen_segments, cross_positions):
        seg = segments[int(seg_idx)]
        axis_pos = random.randint(int(seg["start"]), int(seg["end"]))
        if orientation == "horizontal":
            out[axis_pos, int(cross_pos)] = SEED_COLOR
        else:
            out[int(cross_pos), axis_pos] = SEED_COLOR
    return out


def _semantic_check(grid: np.ndarray, *, expected_orientation: str) -> None:
    info = _analyze(grid)
    orientation = str(info["orientation"])
    seeds = info["seeds"]
    compressed = info["compressed"]
    out = _render_output(grid, info)

    assert orientation == expected_orientation
    assert set(map(int, compressed)).issubset(set(STRIPE_COLORS))
    assert 3 <= len(set(map(int, compressed))) <= 4
    assert 4 <= len(compressed) <= 6
    assert 3 <= len(seeds) <= 5
    assert {int(seed["owner"]) for seed in seeds} == set(map(int, compressed))
    assert len({int(seed["segment_index"]) for seed in seeds}) == len(seeds)
    assert len({int(seed["length"]) for seed in seeds}) >= 3
    assert any(bool(seed["later_repeat"]) for seed in seeds)

    h, w = grid.shape
    extra_magenta = int(np.count_nonzero((out == SEED_COLOR) & (grid != SEED_COLOR)))
    assert extra_magenta >= len(seeds)

    if orientation == "horizontal":
        assert any(0 < w - 1 - int(seed["c"]) < int(seed["length"]) for seed in seeds)
        assert any(w - 1 - int(seed["c"]) >= int(seed["length"]) for seed in seeds if int(seed["length"]) > 1)
    else:
        assert any(0 < h - 1 - int(seed["r"]) < int(seed["length"]) for seed in seeds)
        assert any(h - 1 - int(seed["r"]) >= int(seed["length"]) for seed in seeds if int(seed["length"]) > 1)

    for seed in seeds:
        r = int(seed["r"])
        c = int(seed["c"])
        length = int(seed["length"])
        for step in range(1, length + 1):
            rr = r + step if orientation == "vertical" else r
            cc = c + step if orientation == "horizontal" else c
            if not (0 <= rr < h and 0 <= cc < w):
                break
            assert int(out[rr, cc]) == SEED_COLOR

    assert np.array_equal(main(grid), out)


def _generate_once(counter: int) -> np.ndarray | None:
    orientation = "horizontal" if counter % 2 == 0 else "vertical"
    seq = _make_compressed_sequence(counter)
    grid, segments = _build_base_grid(orientation, seq)
    seeded = _place_seeds(grid, orientation, seq, segments)
    if seeded is None:
        return None

    try:
        _semantic_check(seeded, expected_orientation=orientation)
    except AssertionError:
        return None
    return seeded


def generate_input() -> np.ndarray:
    global _GEN_COUNTER

    for _ in range(400):
        grid = _generate_once(_GEN_COUNTER)
        _GEN_COUNTER += 1
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle-8 input")
