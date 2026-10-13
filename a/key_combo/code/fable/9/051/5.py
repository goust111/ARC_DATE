from __future__ import annotations

import hashlib
import random
from collections import Counter

import numpy as np


BLACK = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
RETRY_LIMIT = 400

_GEN_COUNTER = 0
_SEEN_INPUT_SIGS: set[str] = set()
_SEEN_OUTPUT_SIGS: set[str] = set()


def _grid_sig(grid: np.ndarray) -> str:
    return hashlib.md5(grid.tobytes() + str(tuple(grid.shape)).encode()).hexdigest()


def _random_composition(total: int, parts: int, rng: random.Random) -> list[int] | None:
    if parts <= 0 or total < parts:
        return None
    if parts == 1:
        return [total]
    cuts = sorted(rng.sample(range(1, total), parts - 1))
    values: list[int] = []
    prev = 0
    for cut in cuts + [total]:
        values.append(cut - prev)
        prev = cut
    return values


def _separator_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == GREY)]
    assert len(rows) == 1, "expected one full-width grey separator row"
    return int(rows[0])


def _code_row(grid: np.ndarray, sep: int) -> int:
    rows = [r for r in range(sep + 1, grid.shape[0]) if np.any(grid[r] != BLACK)]
    assert len(rows) == 1, "expected one non-background code row below the separator"
    return int(rows[0])


def _active_bbox(r: int, c: int, radius: int, sep: int, width: int) -> tuple[int, int, int, int]:
    return (
        max(0, r - radius),
        max(0, c - radius),
        min(sep - 1, r + radius),
        min(width - 1, c + radius),
    )


def _rectangles_separate(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _marker_is_clipped(r: int, c: int, radius: int, sep: int, width: int) -> bool:
    return r - radius < 0 or c - radius < 0 or c + radius >= width


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow clipped solid square halos above a separator using code-row color frequencies as radii.

    Input structure:
    - A black background grid contains one full-width grey separator row and exactly one non-background code row below it.
    - Above the separator, the only foreground objects are isolated colored single-pixel markers.
    - Halo squares may clip against the outer grid border.

    Rule:
    - Count how many times each color appears in the code row.
    - For each marker above the separator, if its color appears k times in the code row, paint the clipped solid square of side 2k+1 centered on that marker.
    - Markers whose colors do not appear in the code row are erased to black.
    - Keep the separator row and the code row unchanged.

    Key insight:
    - The bottom row is a legend: color frequency becomes geometric radius, while missing colors mean delete that marker entirely.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep = _separator_row(g)
    code_r = _code_row(g, sep)

    out = g.copy()
    out[:sep, :] = BLACK

    code_counts = Counter(int(v) for v in g[code_r] if int(v) != BLACK)
    for r, c in np.argwhere(g[:sep, :] != BLACK):
        rr = int(r)
        cc = int(c)
        color = int(g[rr, cc])
        radius = int(code_counts.get(color, 0))
        if radius <= 0:
            continue
        r0, c0, r1, c1 = _active_bbox(rr, cc, radius, sep, g.shape[1])
        out[r0 : r1 + 1, c0 : c1 + 1] = color

    out[sep, :] = g[sep, :]
    out[code_r, :] = g[code_r, :]
    return out


def _mode_features(mode: int) -> dict[str, bool]:
    return {
        "all_one": mode == 0,
        "want_absent": mode in (1, 2, 5),
        "want_clip": mode in (2, 5),
        "want_repeat_color": mode in (3, 4, 5),
        "want_duplicate_radius": mode == 4,
        "want_large_radius": mode == 5,
    }


def _sample_active_counts(rng: random.Random, n_active: int, feats: dict[str, bool]) -> list[int]:
    if feats["all_one"]:
        return [1] * n_active

    for _ in range(200):
        if feats["want_large_radius"]:
            counts = [rng.randint(2, 4) for _ in range(n_active)]
            counts[0] = rng.randint(4, 5)
        elif feats["want_duplicate_radius"] and n_active >= 3:
            repeated = rng.randint(2, 4)
            counts = [repeated, repeated]
            counts.extend(rng.randint(1, 4) for _ in range(n_active - 2))
        else:
            counts = [rng.randint(1, 4) for _ in range(n_active)]

        if max(counts) == 1:
            continue
        if len(set(counts)) == 1 and not feats["want_duplicate_radius"] and n_active > 1:
            continue
        if feats["want_duplicate_radius"] and len(set(counts)) == len(counts):
            continue
        return counts

    raise RuntimeError("failed to sample active counts")


def _sample_code_plan(rng: random.Random, feats: dict[str, bool]) -> tuple[list[int], dict[int, int], list[int]]:
    if feats["all_one"]:
        n_active = rng.randint(3, 4)
    elif feats["want_large_radius"]:
        n_active = rng.randint(2, 3)
    else:
        n_active = rng.randint(2, 4)

    active_colors = rng.sample(PALETTE, n_active)
    active_counts = _sample_active_counts(rng, n_active, feats)
    active_map = {color: count for color, count in zip(active_colors, active_counts)}

    min_extra = max(1, 10 - sum(active_counts))
    max_extra = min(8, 22 - sum(active_counts))
    if min_extra > max_extra:
        raise RuntimeError("no width budget for code row")
    extra_total = rng.randint(min_extra, max_extra)
    extra_parts = rng.randint(1, min(3, extra_total))
    extra_counts = _random_composition(extra_total, extra_parts, rng)
    assert extra_counts is not None

    extra_color_pool = [c for c in PALETTE if c not in active_colors]
    extra_colors = rng.sample(extra_color_pool, extra_parts)

    code_values: list[int] = []
    for color, count in active_map.items():
        code_values.extend([color] * count)
    for color, count in zip(extra_colors, extra_counts):
        code_values.extend([color] * count)
    rng.shuffle(code_values)

    absent_colors: list[int] = []
    if feats["want_absent"]:
        unused = [c for c in PALETTE if c not in set(code_values)]
        absent_n = rng.randint(1, min(2, len(unused)))
        absent_colors = rng.sample(unused, absent_n)

    return code_values, active_map, absent_colors


def _sample_marker_multisets(
    rng: random.Random,
    active_map: dict[int, int],
    absent_colors: list[int],
    feats: dict[str, bool],
) -> tuple[list[int], list[int]]:
    active_colors = list(active_map)
    active_markers: list[int] = []
    for color in active_colors:
        active_markers.append(color)

    if feats["want_repeat_color"]:
        repeat_color = rng.choice(active_colors)
        extra_repeats = 1 if feats["want_large_radius"] else rng.randint(1, 2)
        active_markers.extend([repeat_color] * extra_repeats)

    if len(active_markers) > 5:
        rng.shuffle(active_markers)
        active_markers = active_markers[:5]

    absent_markers: list[int] = list(absent_colors)
    rng.shuffle(absent_markers)
    return active_markers, absent_markers


def _candidate_centers_for_active(
    sep: int,
    width: int,
    placed_centers: list[tuple[int, int]],
    placed_boxes: list[tuple[int, int, int, int]],
    radius: int,
    require_clipped: bool,
) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for r in range(sep):
        if r + radius >= sep:
            continue
        for c in range(width):
            if require_clipped and not _marker_is_clipped(r, c, radius, sep, width):
                continue
            if any(max(abs(r - rr), abs(c - cc)) <= 1 for rr, cc in placed_centers):
                continue
            bbox = _active_bbox(r, c, radius, sep, width)
            if any(not _rectangles_separate(bbox, old, gap=1) for old in placed_boxes):
                continue
            if any(bbox[0] <= rr <= bbox[2] and bbox[1] <= cc <= bbox[3] for rr, cc in placed_centers):
                continue
            candidates.append((r, c))
    return candidates


def _candidate_centers_for_absent(
    sep: int,
    width: int,
    placed_centers: list[tuple[int, int]],
    placed_boxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for r in range(sep):
        for c in range(width):
            if any(max(abs(r - rr), abs(c - cc)) <= 1 for rr, cc in placed_centers):
                continue
            if any(box[0] <= r <= box[2] and box[1] <= c <= box[3] for box in placed_boxes):
                continue
            candidates.append((r, c))
    return candidates


def _place_markers(
    rng: random.Random,
    sep: int,
    width: int,
    active_map: dict[int, int],
    active_markers: list[int],
    absent_markers: list[int],
    feats: dict[str, bool],
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]], bool] | None:
    placed_centers: list[tuple[int, int]] = []
    placed_boxes: list[tuple[int, int, int, int]] = []
    active_pixels: list[tuple[int, int, int]] = []
    absent_pixels: list[tuple[int, int, int]] = []

    active_specs = [(color, active_map[color]) for color in active_markers]
    active_specs.sort(key=lambda item: item[1], reverse=True)
    clip_target = None
    if feats["want_clip"]:
        clip_target = max(range(len(active_specs)), key=lambda i: active_specs[i][1])

    for idx, (color, radius) in enumerate(active_specs):
        need_clip = clip_target == idx
        candidates = _candidate_centers_for_active(
            sep,
            width,
            placed_centers,
            placed_boxes,
            radius,
            require_clipped=need_clip,
        )
        if not candidates:
            return None
        r, c = rng.choice(candidates)
        placed_centers.append((r, c))
        placed_boxes.append(_active_bbox(r, c, radius, sep, width))
        active_pixels.append((r, c, color))

    for color in absent_markers:
        candidates = _candidate_centers_for_absent(sep, width, placed_centers, placed_boxes)
        if not candidates:
            return None
        r, c = rng.choice(candidates)
        placed_centers.append((r, c))
        absent_pixels.append((r, c, color))

    clipped = any(_marker_is_clipped(r, c, active_map[color], sep, width) for r, c, color in active_pixels)
    return active_pixels, absent_pixels, clipped


def _build_grid(
    rng: random.Random,
    code_values: list[int],
    active_pixels: list[tuple[int, int, int]],
    absent_pixels: list[tuple[int, int, int]],
    active_map: dict[int, int],
) -> np.ndarray:
    width = len(code_values)
    max_radius = max(active_map.values())
    sep = rng.randint(max_radius + 3, min(18, max_radius + 10))
    code_gap = rng.randint(1, 2)
    code_r = sep + code_gap
    height = code_r + rng.randint(2, 3)

    grid = np.zeros((height, width), dtype=int)
    grid[sep, :] = GREY
    grid[code_r, :] = np.array(code_values, dtype=int)

    for r, c, color in active_pixels + absent_pixels:
        grid[r, c] = color
    return grid


def _expected_output(
    grid: np.ndarray,
    active_pixels: list[tuple[int, int, int]],
    active_map: dict[int, int],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    sep = _separator_row(out)
    out[:sep, :] = BLACK
    for r, c, color in active_pixels:
        radius = active_map[color]
        r0, c0, r1, c1 = _active_bbox(r, c, radius, sep, out.shape[1])
        out[r0 : r1 + 1, c0 : c1 + 1] = color
    return out


def _validate_case(
    grid: np.ndarray,
    expected: np.ndarray,
    active_pixels: list[tuple[int, int, int]],
    absent_pixels: list[tuple[int, int, int]],
    active_map: dict[int, int],
    feats: dict[str, bool],
) -> bool:
    if grid.ndim != 2 or expected.ndim != 2 or max(grid.shape) > 30 or max(expected.shape) > 30:
        return False

    sep = _separator_row(grid)
    code_r = _code_row(grid, sep)
    if not np.all(grid[sep] == GREY):
        return False
    if any(np.any(grid[r] != BLACK) for r in range(sep + 1, grid.shape[0]) if r != code_r):
        return False

    marker_coords = np.argwhere(grid[:sep] != BLACK)
    if marker_coords.shape[0] != len(active_pixels) + len(absent_pixels):
        return False
    centers = [(int(r), int(c)) for r, c in marker_coords]
    if any(max(abs(r0 - r1), abs(c0 - c1)) <= 1 for i, (r0, c0) in enumerate(centers) for r1, c1 in centers[i + 1 :]):
        return False

    code_counts = Counter(int(v) for v in grid[code_r] if int(v) != BLACK)
    if any(code_counts.get(color, 0) != radius for color, radius in active_map.items()):
        return False
    if any(code_counts.get(color, 0) != 0 for _r, _c, color in absent_pixels):
        return False
    if feats["want_absent"] and not absent_pixels:
        return False
    if feats["want_clip"] and not any(_marker_is_clipped(r, c, active_map[color], sep, grid.shape[1]) for r, c, color in active_pixels):
        return False

    boxes = [_active_bbox(r, c, active_map[color], sep, grid.shape[1]) for r, c, color in active_pixels]
    for i, box in enumerate(boxes):
        for other in boxes[i + 1 :]:
            if not _rectangles_separate(box, other, gap=0):
                return False

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(actual, grid):
        return False
    return True


def _gen_attempt(rng: random.Random, mode: int) -> tuple[np.ndarray, np.ndarray] | None:
    feats = _mode_features(mode)
    try:
        code_values, active_map, absent_colors = _sample_code_plan(rng, feats)
    except RuntimeError:
        return None

    active_markers, absent_markers = _sample_marker_multisets(rng, active_map, absent_colors, feats)
    max_radius = max(active_map.values())

    for _ in range(120):
        sep = rng.randint(max_radius + 3, min(18, max_radius + 10))
        placed = _place_markers(rng, sep, len(code_values), active_map, active_markers, absent_markers, feats)
        if placed is None:
            continue
        active_pixels, absent_pixels, _clipped = placed
        code_gap = rng.randint(1, 2)
        code_r = sep + code_gap
        height = code_r + rng.randint(2, 3)
        if height > 30:
            continue

        grid = np.zeros((height, len(code_values)), dtype=int)
        grid[sep, :] = GREY
        grid[code_r, :] = np.array(code_values, dtype=int)
        for r, c, color in active_pixels + absent_pixels:
            grid[r, c] = color

        expected = _expected_output(grid, active_pixels, active_map)
        if not _validate_case(grid, expected, active_pixels, absent_pixels, active_map, feats):
            continue

        in_sig = _grid_sig(grid)
        out_sig = _grid_sig(expected)
        if in_sig in _SEEN_INPUT_SIGS or out_sig in _SEEN_OUTPUT_SIGS:
            continue
        return grid, expected
    return None


def generate_input() -> np.ndarray:
    global _GEN_COUNTER

    rng = random.Random()
    for _ in range(RETRY_LIMIT):
        mode = _GEN_COUNTER % 6
        result = _gen_attempt(rng, mode)
        _GEN_COUNTER += 1
        if result is None:
            continue
        grid, expected = result
        _SEEN_INPUT_SIGS.add(_grid_sig(grid))
        _SEEN_OUTPUT_SIGS.add(_grid_sig(expected))
        return grid
    raise RuntimeError("failed to generate a valid frequency-halo input")
