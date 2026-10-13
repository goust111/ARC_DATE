from __future__ import annotations

import random
from collections import Counter

import numpy as np


BACKGROUND = 0
ARC_COLORS = tuple(range(10))
FOREGROUND_COLORS = tuple(range(1, 10))

GEN_MAX_ATTEMPTS = 600
GEN_GRID_H = (20, 28)
GEN_GRID_W = (20, 28)
GEN_WINDOW_COUNT = (4, 6)
GEN_PERIOD = (1, 5)
GEN_INNER_H = (3, 7)
GEN_INNER_W = (3, 7)
GEN_PADDING = 1


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _sample_tile(tile: np.ndarray, top: int, left: int, height: int, width: int) -> np.ndarray:
    pr, pc = tile.shape
    patch = np.zeros((height, width), dtype=int)
    for r in range(height):
        for c in range(width):
            patch[r, c] = int(tile[(top + r) % pr, (left + c) % pc])
    return patch


def _periodic_under(tile: np.ndarray, pr: int, pc: int) -> bool:
    rows, cols = tile.shape
    for r in range(rows):
        for c in range(cols):
            if int(tile[r, c]) != int(tile[r % pr, c % pc]):
                return False
    return True


def _is_primitive_tile(tile: np.ndarray) -> bool:
    pr, pc = tile.shape
    for rr in range(1, pr + 1):
        for cc in range(1, pc + 1):
            if rr == pr and cc == pc:
                continue
            if _periodic_under(tile, rr, cc):
                return False
    return True


def _random_tile(pr: int, pc: int, palette: list[int]) -> np.ndarray | None:
    area = pr * pc
    min_colors = 2 if area < 3 else 3
    color_count = random.randint(min_colors, min(len(palette), max(min_colors, min(5, area))))
    colors = random.sample(palette, color_count)
    cells = [colors[i % color_count] for i in range(area)]
    random.shuffle(cells)
    tile = np.array(cells, dtype=int).reshape(pr, pc)

    for _ in range(40):
        rr = random.randrange(pr)
        cc = random.randrange(pc)
        tile[rr, cc] = random.choice(colors)
        if len(set(map(int, tile.ravel()))) >= min_colors and _is_primitive_tile(tile):
            return tile
    return None


def _detect_windows(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    windows: list[dict] = []
    for color in sorted(set(map(int, g.ravel())) - {bg}):
        for comp in _components(g == color):
            r0, c0, r1, c1 = _bbox(comp)
            height = r1 - r0 + 1
            width = c1 - c0 + 1
            if height < 5 or width < 5:
                continue
            expected_area = 2 * height + 2 * width - 4
            if len(comp) != expected_area:
                continue
            border_ok = True
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    on_border = r in (r0, r1) or c in (c0, c1)
                    if on_border:
                        if int(g[r, c]) != color:
                            border_ok = False
                            break
                if not border_ok:
                    break
            if not border_ok:
                continue
            inner = g[r0 + 1 : r1, c0 + 1 : c1]
            if inner.size == 0 or np.any(inner == bg):
                continue
            windows.append(
                {
                    "frame_color": color,
                    "bbox": (r0, c0, r1, c1),
                    "top": (r0 + 1, c0 + 1),
                    "patch": inner.copy(),
                }
            )
    windows.sort(key=lambda item: item["bbox"])
    assert len(windows) >= 3, "expected several framed windows"
    return windows


def _window_mismatch_count(tile: np.ndarray, window: dict, row_shift: int = 0, col_shift: int = 0) -> int:
    top, left = window["top"]
    patch = window["patch"]
    pr, pc = tile.shape
    mismatches = 0
    for r in range(patch.shape[0]):
        for c in range(patch.shape[1]):
            target = int(tile[(top + r + row_shift) % pr, (left + c + col_shift) % pc])
            mismatches += int(int(patch[r, c]) != target)
    return mismatches


def _matching_shifts(tile: np.ndarray, window: dict) -> list[tuple[int, int]]:
    pr, pc = tile.shape
    shifts: list[tuple[int, int]] = []
    for row_shift in range(pr):
        for col_shift in range(pc):
            if _window_mismatch_count(tile, window, row_shift, col_shift) == 0:
                shifts.append((row_shift, col_shift))
    return shifts


def _evaluate_candidate(windows: list[dict], pr: int, pc: int) -> dict | None:
    counts = np.zeros((pr, pc, 10), dtype=int)
    seen = np.zeros((pr, pc), dtype=int)
    for window in windows:
        top, left = window["top"]
        patch = window["patch"]
        for r in range(patch.shape[0]):
            for c in range(patch.shape[1]):
                rr = (top + r) % pr
                cc = (left + c) % pc
                color = int(patch[r, c])
                counts[rr, cc, color] += 1
                seen[rr, cc] += 1

    tile = np.zeros((pr, pc), dtype=int)
    for rr in range(pr):
        for cc in range(pc):
            if seen[rr, cc] == 0:
                continue
            phase_counts = counts[rr, cc]
            best = int(phase_counts.max())
            winners = np.flatnonzero(phase_counts == best)
            if len(winners) != 1:
                return None
            tile[rr, cc] = int(winners[0])

    mismatches = [_window_mismatch_count(tile, window) for window in windows]
    bad = [idx for idx, value in enumerate(mismatches) if value > 0]
    if len(bad) != 1:
        return None

    liar_idx = bad[0]
    shifts = [shift for shift in _matching_shifts(tile, windows[liar_idx]) if shift != (0, 0)]
    if not shifts:
        return None

    top, left = windows[liar_idx]["top"]
    height, width = windows[liar_idx]["patch"].shape
    expected = _sample_tile(tile, top, left, height, width)
    return {
        "period": (pr, pc),
        "tile": tile,
        "liar_idx": liar_idx,
        "mismatches": mismatches,
        "expected": expected,
        "shifts": shifts,
    }


def _valid_candidates(windows: list[dict], shape: tuple[int, int]) -> list[dict]:
    h, w = shape
    candidates: list[dict] = []
    for pr in range(1, h + 1):
        for pc in range(1, w + 1):
            candidate = _evaluate_candidate(windows, pr, pc)
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def _candidate_rank(candidate: dict) -> tuple[int, int, int, int, int]:
    pr, pc = candidate["period"]
    liar_bad = int(candidate["mismatches"][candidate["liar_idx"]])
    return pr * pc, pr + pc, pr, pc, -liar_bad


def _pick_best_candidate(candidates: list[dict]) -> dict:
    assert candidates, "failed to reconstruct a periodic fabric"
    candidates = sorted(candidates, key=_candidate_rank)
    best = candidates[0]
    best_rank = _candidate_rank(best)
    same_rank = [cand for cand in candidates if _candidate_rank(cand) == best_rank]
    fingerprints = {
        (cand["liar_idx"], cand["expected"].shape, cand["expected"].tobytes())
        for cand in same_rank
    }
    assert len(fingerprints) == 1, "best reconstruction is ambiguous"
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """Recover the shared periodic fabric, identify the unique wrong-phase window, and output its correct view.

    Input structure:
    - A black canvas contains several disjoint hollow rectangular frames.
    - Every frame encloses a fully filled multicolor inner window, while the outside stays black.
    - All truthful windows are crops of one infinite periodic fabric at their own absolute grid positions, and one window shows a crop from a different phase.

    Rule:
    - Detect every hollow frame and read its inner rectangle as one window observation.
    - Search for the smallest row/column period whose phase table makes all but one windows exactly consistent with their absolute positions.
    - The inconsistent window must still be explainable as a whole crop from some other phase of the same fabric; output the crop that the reconstructed fabric should show at that window's true position.

    Key insight:
    - The liar patch is locally valid, so only a global phase model tied to absolute window positions can expose it; the true fabric is the smallest periodic explanation supported by the mutually consistent windows.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    windows = _detect_windows(g)
    best = _pick_best_candidate(_valid_candidates(windows, g.shape))
    return best["expected"]


def _choose_period() -> tuple[int, int]:
    branch = random.choice(["both", "row_only", "col_only"])
    if branch == "row_only":
        return random.randint(2, GEN_PERIOD[1]), 1
    if branch == "col_only":
        return 1, random.randint(2, GEN_PERIOD[1])
    return random.randint(2, GEN_PERIOD[1]), random.randint(2, GEN_PERIOD[1])


def _random_inner_size(period_len: int, lo: int, hi: int) -> int:
    low = max(lo, period_len + random.randint(0, 1) if period_len > 1 else lo)
    high = min(hi, max(low, period_len + 3))
    return random.randint(low, high)


def _place_window_boxes(height: int, width: int, sizes: list[tuple[int, int]]) -> list[tuple[int, int, int, int]] | None:
    occupied = np.zeros((height, width), dtype=bool)
    boxes: list[tuple[int, int, int, int]] = []
    order = list(enumerate(sizes))
    order.sort(key=lambda item: item[1][0] * item[1][1], reverse=True)
    chosen: dict[int, tuple[int, int, int, int]] = {}

    for original_idx, (full_h, full_w) in order:
        candidates: list[tuple[int, int]] = []
        for r0 in range(0, height - full_h + 1):
            for c0 in range(0, width - full_w + 1):
                rr0 = max(0, r0 - GEN_PADDING)
                cc0 = max(0, c0 - GEN_PADDING)
                rr1 = min(height, r0 + full_h + GEN_PADDING)
                cc1 = min(width, c0 + full_w + GEN_PADDING)
                if not occupied[rr0:rr1, cc0:cc1].any():
                    candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        rr0 = max(0, r0 - GEN_PADDING)
        cc0 = max(0, c0 - GEN_PADDING)
        rr1 = min(height, r0 + full_h + GEN_PADDING)
        cc1 = min(width, c0 + full_w + GEN_PADDING)
        occupied[rr0:rr1, cc0:cc1] = True
        chosen[original_idx] = (r0, c0, r0 + full_h - 1, c0 + full_w - 1)

    for idx in range(len(sizes)):
        boxes.append(chosen[idx])
    return boxes


def _shift_options(pr: int, pc: int) -> list[tuple[int, int]]:
    options: list[tuple[int, int]] = []
    if pr > 1:
        options.extend((shift, 0) for shift in range(1, pr))
    if pc > 1:
        options.extend((0, shift) for shift in range(1, pc))
    if pr > 1 and pc > 1:
        options.extend((rr, cc) for rr in range(1, pr) for cc in range(1, pc))
    return options


def _build_puzzle() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict], int, tuple[int, int]]:
    height = random.randint(*GEN_GRID_H)
    width = random.randint(*GEN_GRID_W)
    frame_color = random.choice(FOREGROUND_COLORS)
    palette = [color for color in FOREGROUND_COLORS if color != frame_color]

    pr, pc = _choose_period()
    tile = _random_tile(pr, pc, palette)
    if tile is None:
        raise ValueError("failed to sample a primitive fabric tile")

    window_count = random.randint(*GEN_WINDOW_COUNT)
    inner_sizes = [
        (
            _random_inner_size(pr, GEN_INNER_H[0], GEN_INNER_H[1]),
            _random_inner_size(pc, GEN_INNER_W[0], GEN_INNER_W[1]),
        )
        for _ in range(window_count)
    ]
    boxes = _place_window_boxes(height, width, [(h + 2, w + 2) for h, w in inner_sizes])
    if boxes is None:
        raise ValueError("failed to place windows")

    specs: list[dict] = []
    for (r0, c0, r1, c1), (inner_h, inner_w) in zip(boxes, inner_sizes):
        top = (r0 + 1, c0 + 1)
        specs.append(
            {
                "bbox": (r0, c0, r1, c1),
                "top": top,
                "shape": (inner_h, inner_w),
                "correct_patch": _sample_tile(tile, top[0], top[1], inner_h, inner_w),
            }
        )

    specs.sort(key=lambda item: item["bbox"])
    liar_idx = random.randrange(len(specs))
    liar = specs[liar_idx]

    options = _shift_options(pr, pc)
    random.shuffle(options)
    wrong_shift = None
    wrong_patch = None
    for shift in options:
        candidate = _sample_tile(
            tile,
            liar["top"][0] + shift[0],
            liar["top"][1] + shift[1],
            liar["shape"][0],
            liar["shape"][1],
        )
        if not np.array_equal(candidate, liar["correct_patch"]):
            wrong_shift = shift
            wrong_patch = candidate
            break
    if wrong_shift is None or wrong_patch is None:
        raise ValueError("failed to find a distinct lying phase")

    grid = np.full((height, width), BACKGROUND, dtype=int)
    for idx, spec in enumerate(specs):
        r0, c0, r1, c1 = spec["bbox"]
        grid[r0 : r1 + 1, c0] = frame_color
        grid[r0 : r1 + 1, c1] = frame_color
        grid[r0, c0 : c1 + 1] = frame_color
        grid[r1, c0 : c1 + 1] = frame_color
        patch = wrong_patch if idx == liar_idx else spec["correct_patch"]
        top, left = spec["top"]
        grid[top : top + patch.shape[0], left : left + patch.shape[1]] = patch

    expected = liar["correct_patch"]
    return grid, expected, tile, specs, liar_idx, wrong_shift


def _validate_generated(
    grid: np.ndarray,
    expected: np.ndarray,
    tile: np.ndarray,
    specs: list[dict],
    liar_idx: int,
    wrong_shift: tuple[int, int],
) -> bool:
    detected = _detect_windows(grid)
    if len(detected) != len(specs):
        return False

    for idx, (window, spec) in enumerate(zip(detected, specs)):
        if window["bbox"] != spec["bbox"]:
            return False
        top, left = spec["top"]
        shape = spec["shape"]
        truthful_patch = _sample_tile(tile, top, left, shape[0], shape[1])
        if not np.array_equal(truthful_patch, spec["correct_patch"]):
            return False
        if idx == liar_idx:
            observed = _sample_tile(tile, top + wrong_shift[0], left + wrong_shift[1], shape[0], shape[1])
            if np.array_equal(observed, truthful_patch):
                return False
            if not np.array_equal(window["patch"], observed):
                return False
        elif not np.array_equal(window["patch"], truthful_patch):
            return False

    for idx, window in enumerate(detected):
        if idx != liar_idx and window["patch"].shape == expected.shape and np.array_equal(window["patch"], expected):
            return False

    candidates = _valid_candidates(detected, grid.shape)
    if not candidates:
        return False
    best = _pick_best_candidate(candidates)
    if best["period"] != tile.shape:
        return False
    if best["liar_idx"] != liar_idx:
        return False
    if not np.array_equal(best["expected"], expected):
        return False
    if wrong_shift not in best["shifts"]:
        return False

    best_rank = _candidate_rank(best)
    conflicting = [
        cand
        for cand in candidates
        if _candidate_rank(cand) == best_rank
        and (
            cand["liar_idx"] != liar_idx
            or cand["expected"].shape != expected.shape
            or cand["expected"].tobytes() != expected.tobytes()
        )
    ]
    if conflicting:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        try:
            grid, expected, tile, specs, liar_idx, wrong_shift = _build_puzzle()
        except ValueError:
            continue
        if _validate_generated(grid, expected, tile, specs, liar_idx, wrong_shift):
            return grid
    raise RuntimeError("failed to generate a valid lying-window fabric puzzle")
