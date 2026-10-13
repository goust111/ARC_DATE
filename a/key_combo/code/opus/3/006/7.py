from __future__ import annotations

from collections import Counter
import importlib.util
from pathlib import Path
import random

import numpy as np


def _load_arc_utils():
    candidates = [
        Path("/home/lixinyu/snap/new/solvers/arc_utils.py"),
    ]
    for path in candidates:
        if not path.is_file():
            continue
        spec = importlib.util.spec_from_file_location("_shared_arc_utils_7", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    raise ImportError("unable to locate arc_utils.py")


_ARC_UTILS = _load_arc_utils()
in_bounds = _ARC_UTILS.in_bounds
neighbors4 = _ARC_UTILS.neighbors4


MAIN_COLORS = tuple(range(1, 10))
HEIGHT_RANGE = (5, 7)
TILE_WIDTH_CHOICES = (4, 5, 6, 7, 8)
REPEAT_CHOICES = (3, 4)
NOISE_COUNT_RANGE = (5, 10)
MAX_GENERATION_ATTEMPTS = 400


def _rank_colors(grid: np.ndarray) -> list[int]:
    flat = [int(v) for v in np.asarray(grid, dtype=int).ravel()]
    counts = Counter(flat)
    first_seen: dict[int, int] = {}
    for idx, value in enumerate(flat):
        first_seen.setdefault(value, idx)
    ordered = sorted(counts.items(), key=lambda item: (-item[1], first_seen[item[0]]))
    return [int(color) for color, _count in ordered]


def _repair_grid(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    ranked = _rank_colors(g)
    assert len(ranked) >= 2, "expected at least two colors"
    main_colors = (int(ranked[0]), int(ranked[1]))
    main_set = set(main_colors)

    repaired = g.copy()
    noise_coords = np.argwhere(~np.isin(g, list(main_set)))
    for r_raw, c_raw in noise_coords:
        r = int(r_raw)
        c = int(c_raw)
        local_counts = {main_colors[0]: 0, main_colors[1]: 0}
        for nr, nc in neighbors4(r, c):
            if not in_bounds(g.shape, nr, nc):
                continue
            value = int(g[int(nr), int(nc)])
            if value in main_set:
                local_counts[value] += 1
        best = main_colors[0]
        if local_counts[main_colors[1]] > local_counts[best]:
            best = main_colors[1]
        repaired[r, c] = best
    return repaired, main_colors


def _smallest_mirror_period(grid: np.ndarray) -> tuple[int, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    width = int(g.shape[1])
    for candidate_w in range(1, width + 1):
        if width % candidate_w != 0:
            continue
        first_block = g[:, :candidate_w]
        flipped = np.fliplr(first_block)
        ok = True
        for c0 in range(0, width, candidate_w):
            block = g[:, c0 : c0 + candidate_w]
            if np.array_equal(block, first_block) or np.array_equal(block, flipped):
                continue
            ok = False
            break
        if ok:
            return candidate_w, first_block.copy()
    raise AssertionError("a grid must always match itself at full width")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair rare noise colors and extract the smallest horizontally reflected repeat block.

    Input structure:
    - The grid is filled by horizontally concatenating copies of one two-color tile, with some copies optionally flipped left-to-right.
    - Two colors dominate the grid and form the true tile pattern.
    - A few scattered pixels use one or two rarer colors and act as non-periodic noise.

    Rule:
    - Rank colors by global frequency and treat the two most frequent colors as the main colors.
    - For every other-color pixel, replace it by the most common main color in its 4-neighborhood; break local ties in favor of the globally more frequent main color.
    - On the repaired grid, scan divisors of the full width from smallest to largest and find the first block width whose column blocks are each either equal to the leftmost block or to its horizontal mirror.
    - Output that leftmost repaired block.

    Key insight:
    - The noise pixels are identified globally by frequency, but each one must be repaired locally before the hidden mirror-repeat period can be detected.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    repaired, _main_colors = _repair_grid(g)
    _period_w, first_block = _smallest_mirror_period(repaired)
    return first_block


def _add_vertical_bar(mask: np.ndarray) -> None:
    h, w = mask.shape
    thickness = 1 if w <= 5 else random.choice((1, 1, 2))
    c0 = random.randint(0, w - thickness)
    r0 = random.randint(0, max(0, h - 2))
    r1 = random.randint(r0 + 2, h)
    mask[r0:r1, c0 : c0 + thickness] = True


def _add_horizontal_bar(mask: np.ndarray) -> None:
    h, w = mask.shape
    thickness = 1 if h <= 5 else random.choice((1, 1, 2))
    r0 = random.randint(0, h - thickness)
    c0 = random.randint(0, max(0, w - 2))
    c1 = random.randint(c0 + 2, w)
    mask[r0 : r0 + thickness, c0:c1] = True


def _add_rectangle(mask: np.ndarray) -> None:
    h, w = mask.shape
    rect_h = random.randint(2, min(3, h))
    rect_w = random.randint(2, min(max(3, w - 1), w))
    r0 = random.randint(0, h - rect_h)
    c0 = random.randint(0, w - rect_w)
    mask[r0 : r0 + rect_h, c0 : c0 + rect_w] = True


def _add_stairs(mask: np.ndarray) -> None:
    h, w = mask.shape
    if h < 3 or w < 3:
        return
    step_count = random.randint(3, min(h, w))
    r0 = random.randint(0, h - step_count)
    dc = random.choice((-1, 1))
    if dc == 1:
        c0 = random.randint(0, w - step_count)
    else:
        c0 = random.randint(step_count - 1, w - 1)
    for step in range(step_count):
        r = r0 + step
        c = c0 + dc * step
        mask[r, c] = True
        if random.random() < 0.35:
            nc = c + dc
            if 0 <= nc < w:
                mask[r, nc] = True


def _build_base_tile(height: int, tile_w: int, dominant: int, accent: int) -> np.ndarray | None:
    for _ in range(120):
        mask = np.zeros((height, tile_w), dtype=bool)
        _add_rectangle(mask)
        _add_vertical_bar(mask)
        _add_horizontal_bar(mask)
        if random.random() < 0.8:
            random.choice((_add_rectangle, _add_vertical_bar, _add_horizontal_bar, _add_stairs))(mask)

        accent_count = int(mask.sum())
        total = height * tile_w
        dominant_count = total - accent_count
        if accent_count < max(4, height):
            continue
        if accent_count > total // 2:
            continue
        if dominant_count - accent_count < max(3, height // 2):
            continue

        tile = np.full((height, tile_w), dominant, dtype=int)
        tile[mask] = accent
        if np.array_equal(tile, np.fliplr(tile)):
            continue
        return tile
    return None


def _choose_orientations(repeats: int) -> list[int]:
    first = random.choice((0, 1))
    if repeats == 1:
        return [first]
    if random.random() < 0.7:
        rest = [random.choice((0, 1)) for _ in range(repeats - 1)]
        if all(value == first for value in rest):
            rest[random.randrange(len(rest))] = 1 - first
    else:
        rest = [first] * (repeats - 1)
    return [first] + rest


def _assemble_clean_grid(tile: np.ndarray, orientations: list[int]) -> np.ndarray:
    blocks = [np.fliplr(tile) if flag else tile for flag in orientations]
    return np.concatenate(blocks, axis=1)


def _repair_choice(clean: np.ndarray, main_colors: tuple[int, int], r: int, c: int) -> tuple[int, dict[int, int], bool]:
    local_counts = {main_colors[0]: 0, main_colors[1]: 0}
    for nr, nc in neighbors4(r, c):
        if in_bounds(clean.shape, nr, nc):
            value = int(clean[int(nr), int(nc)])
            local_counts[value] += 1
    best = main_colors[0]
    if local_counts[main_colors[1]] > local_counts[best]:
        best = main_colors[1]
    tie = local_counts[main_colors[0]] == local_counts[main_colors[1]]
    return best, local_counts, tie


def _candidate_noise_cells(clean: np.ndarray, main_colors: tuple[int, int], tile_w: int) -> list[dict[str, object]]:
    h, w = clean.shape
    candidates: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            best, local_counts, tie = _repair_choice(clean, main_colors, r, c)
            cell_color = int(clean[r, c])
            if best != cell_color:
                continue
            candidates.append(
                {
                    "rc": (r, c),
                    "color": cell_color,
                    "first": c < tile_w,
                    "later": c >= tile_w,
                    "seam": 0 < c < w - 1 and (c % tile_w in (0, tile_w - 1)),
                    "tie": tie,
                    "local_counts": local_counts,
                }
            )
    return candidates


def _non_adjacent(selected: list[tuple[int, int]], rc: tuple[int, int]) -> bool:
    return all(abs(rc[0] - sr) + abs(rc[1] - sc) > 1 for sr, sc in selected)


def _choose_noise_cells(
    candidates: list[dict[str, object]],
    main_colors: tuple[int, int],
    noise_count: int,
    require_tie: bool,
    require_seam: bool,
    require_secondary_first: bool,
) -> list[dict[str, object]] | None:
    requirements: list[tuple[str, object]] = [
        ("first", lambda cand: bool(cand["first"])),
        ("later", lambda cand: bool(cand["later"])),
    ]
    if require_seam:
        requirements.append(("seam", lambda cand: bool(cand["seam"])))
    if require_tie:
        requirements.append(("tie", lambda cand: bool(cand["tie"])))
    if require_secondary_first:
        requirements.append(
            (
                "secondary_first",
                lambda cand: bool(cand["first"]) and int(cand["color"]) == int(main_colors[1]),
            )
        )

    selected: list[dict[str, object]] = []
    selected_rc: list[tuple[int, int]] = []
    remaining = candidates[:]
    random.shuffle(remaining)
    covered: set[str] = set()

    for name, predicate in requirements:
        if name in covered:
            continue
        options = [
            cand
            for cand in remaining
            if _non_adjacent(selected_rc, cand["rc"]) and bool(predicate(cand))
        ]
        if not options:
            return None
        choice = random.choice(options)
        selected.append(choice)
        selected_rc.append(choice["rc"])
        for req_name, req_predicate in requirements:
            if bool(req_predicate(choice)):
                covered.add(req_name)

    random.shuffle(remaining)
    for cand in remaining:
        if cand["rc"] in selected_rc:
            continue
        if not _non_adjacent(selected_rc, cand["rc"]):
            continue
        selected.append(cand)
        selected_rc.append(cand["rc"])
        if len(selected) >= noise_count:
            break

    if len(selected) < noise_count:
        return None
    return selected


def _assign_noise_colors(main_colors: tuple[int, int], noise_count: int) -> list[int]:
    palette = [color for color in MAIN_COLORS if color not in main_colors]
    if noise_count <= 6 and random.random() < 0.45:
        n_noise_colors = 1
    else:
        n_noise_colors = 2
    chosen = random.sample(palette, n_noise_colors)
    if n_noise_colors == 1:
        return [chosen[0]] * noise_count

    counts = [1, 1]
    for _ in range(noise_count - 2):
        counts[random.randrange(2)] += 1
    assignment = [chosen[0]] * counts[0] + [chosen[1]] * counts[1]
    random.shuffle(assignment)
    return assignment


def _make_noisy_grid(clean: np.ndarray, chosen_cells: list[dict[str, object]], main_colors: tuple[int, int]) -> tuple[np.ndarray, list[tuple[int, int]]]:
    noisy = clean.copy()
    colors = _assign_noise_colors(main_colors, len(chosen_cells))
    noise_coords: list[tuple[int, int]] = []
    for cell, noise_color in zip(chosen_cells, colors):
        r, c = cell["rc"]
        noisy[r, c] = noise_color
        noise_coords.append((r, c))
    return noisy, noise_coords


def _global_main_counts(grid: np.ndarray, main_colors: tuple[int, int]) -> tuple[int, int]:
    counts = Counter(int(v) for v in grid.ravel())
    return counts[main_colors[0]], counts[main_colors[1]]


def _validate_noise_frequencies(grid: np.ndarray, main_colors: tuple[int, int]) -> bool:
    counts = Counter(int(v) for v in grid.ravel())
    main_counts = [counts[main_colors[0]], counts[main_colors[1]]]
    if main_counts[0] <= main_counts[1]:
        return False
    bound = min(main_counts) / 5.0
    for color, count in counts.items():
        if color in main_colors:
            continue
        if count > bound:
            return False
    return True


def _naive_fill(grid: np.ndarray, main_colors: tuple[int, int], noise_coords: list[tuple[int, int]], fill_color: int) -> np.ndarray:
    naive = np.asarray(grid, dtype=int).copy()
    for r, c in noise_coords:
        naive[r, c] = fill_color
    return naive


def _erase_noise(grid: np.ndarray, noise_coords: list[tuple[int, int]]) -> np.ndarray:
    erased = np.asarray(grid, dtype=int).copy()
    for r, c in noise_coords:
        erased[r, c] = 0
    return erased


def _validate_example(
    noisy: np.ndarray,
    clean: np.ndarray,
    tile_w: int,
    main_colors: tuple[int, int],
    noise_coords: list[tuple[int, int]],
) -> bool:
    clean_period, clean_first = _smallest_mirror_period(clean)
    if clean_period != tile_w:
        return False

    ranked = _rank_colors(noisy)
    if tuple(ranked[:2]) != tuple(main_colors):
        return False
    if not _validate_noise_frequencies(noisy, main_colors):
        return False

    noisy_period, noisy_first = _smallest_mirror_period(noisy)
    if noisy_period <= tile_w:
        return False
    if np.array_equal(noisy_first, clean_first):
        return False

    repaired, repaired_main_colors = _repair_grid(noisy)
    if repaired_main_colors != main_colors:
        return False
    if not np.array_equal(repaired, clean):
        return False

    out = main(noisy)
    if not np.array_equal(out, clean_first):
        return False
    if np.array_equal(out, noisy[:, :tile_w]):
        return False

    erased = _erase_noise(noisy, noise_coords)
    erased_period, erased_first = _smallest_mirror_period(erased)
    if erased_period == tile_w and np.array_equal(erased_first, clean_first):
        return False

    dominant_fill = _naive_fill(noisy, main_colors, noise_coords, main_colors[0])
    dominant_period, dominant_first = _smallest_mirror_period(dominant_fill)
    if dominant_period == tile_w and np.array_equal(dominant_first, clean_first):
        return False
    return True


def _generate_once() -> np.ndarray | None:
    height = random.randint(*HEIGHT_RANGE)
    tile_w = random.choice(TILE_WIDTH_CHOICES)
    repeats = random.choice(REPEAT_CHOICES)
    if tile_w * repeats > 30:
        return None

    dominant, accent = random.sample(MAIN_COLORS, 2)
    tile = _build_base_tile(height, tile_w, dominant, accent)
    if tile is None:
        return None

    orientations = _choose_orientations(repeats)
    clean = _assemble_clean_grid(tile, orientations)
    clean_counts = Counter(int(v) for v in clean.ravel())
    main_colors = (dominant, accent)
    if clean_counts[dominant] <= clean_counts[accent]:
        return None

    clean_period, _first = _smallest_mirror_period(clean)
    if clean_period != tile_w:
        return None

    candidates = _candidate_noise_cells(clean, main_colors, tile_w)
    if len(candidates) < NOISE_COUNT_RANGE[0]:
        return None

    require_tie = random.random() < 0.55
    require_seam = random.random() < 0.7
    require_secondary_first = random.random() < 0.6
    min_noise = max(NOISE_COUNT_RANGE[0], 2 + int(require_tie) + int(require_seam) + int(require_secondary_first))
    max_noise = min(NOISE_COUNT_RANGE[1], len(candidates))
    if min_noise > max_noise:
        return None
    noise_count = random.randint(min_noise, max_noise)

    chosen_cells = _choose_noise_cells(
        candidates,
        main_colors,
        noise_count=noise_count,
        require_tie=require_tie,
        require_seam=require_seam,
        require_secondary_first=require_secondary_first,
    )
    if chosen_cells is None:
        return None

    noisy, noise_coords = _make_noisy_grid(clean, chosen_cells, main_colors)
    if not _validate_example(noisy, clean, tile_w, main_colors, noise_coords):
        return None
    return noisy


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid reflected-period noise puzzle")
