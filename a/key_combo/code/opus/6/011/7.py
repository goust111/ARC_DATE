from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
PALETTE = (1, 2, 3)
COLOR_CYCLE = {2: 1, 1: 3, 3: 2}
GENERATION_RETRIES = 400
_GEN_COUNTER = 0


def _components8(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr = rr + dr
                        nc = cc + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            q.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _touches_existing(occupied: set[tuple[int, int]], coords: list[tuple[int, int]]) -> bool:
    probe = set(coords)
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if (nr, nc) in occupied and (nr, nc) not in probe:
                    return True
    return False


def _segment_coords(diag_sum: int, c0: int, length: int) -> list[tuple[int, int]]:
    return [(diag_sum - c, c) for c in range(c0, c0 + length)]


def _extract_segments(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    segments: list[dict] = []
    for color in sorted(set(int(v) for v in np.unique(g)) - {BACKGROUND}):
        for comp in _components8(g == color):
            rows = sorted(int(r) for r, _c in comp)
            cols = sorted(int(c) for _r, c in comp)
            diag_sums = {int(r + c) for r, c in comp}
            assert len(diag_sums) == 1, "every object must lie on one anti-diagonal"
            assert rows == list(range(rows[0], rows[-1] + 1)), "segment rows must be contiguous"
            assert cols == list(range(cols[0], cols[-1] + 1)), "segment columns must be contiguous"
            assert len(rows) == len(cols) == len(comp), "segment must be one-pixel wide"
            diag_sum = int(next(iter(diag_sums)))
            c0 = cols[0]
            length = len(cols)
            coords = _segment_coords(diag_sum, c0, length)
            assert sorted(coords) == sorted((int(r), int(c)) for r, c in comp)
            c1 = c0 + length - 1
            top = (diag_sum - c1, c1)
            segments.append(
                {
                    "color": color,
                    "diag_sum": diag_sum,
                    "c0": c0,
                    "length": length,
                    "coords": coords,
                    "top": top,
                }
            )
    assert 2 <= len(segments) <= 3, "expected 2 or 3 anti-diagonal segments"
    segments.sort(key=lambda seg: (seg["top"][0], seg["top"][1]))
    return segments


def _simulate(
    grid: np.ndarray,
    *,
    truncate_keeps_color: bool,
    reverse_order: bool = False,
    ignore_collisions: bool = False,
) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    segments = _extract_segments(g)
    order = list(reversed(segments)) if reverse_order else segments
    h, w = out.shape
    infos: list[dict] = []

    for seg in order:
        for r, c in seg["coords"]:
            out[r, c] = BACKGROUND

        c1 = seg["c0"] + seg["length"] - 1
        candidates = [(seg["diag_sum"] - c, c) for c in range(c1 + 1, c1 + seg["length"] + 2)]

        actual: list[tuple[int, int]] = []
        collision_at: tuple[int, int] | None = None
        blocked_by: int | None = None
        out_of_bounds = False
        for r, c in candidates:
            if not (0 <= r < h and 0 <= c < w):
                out_of_bounds = True
                break
            if ignore_collisions:
                actual.append((r, c))
                continue
            if out[r, c] != BACKGROUND:
                collision_at = (r, c)
                blocked_by = int(out[r, c])
                break
            actual.append((r, c))

        collided = (collision_at is not None) or (out_of_bounds and len(actual) < len(candidates))
        if actual:
            if collided and truncate_keeps_color:
                draw_color = seg["color"]
            else:
                draw_color = COLOR_CYCLE[seg["color"]]
            for r, c in actual:
                out[r, c] = draw_color
        else:
            draw_color = None

        infos.append(
            {
                "color": seg["color"],
                "top": seg["top"],
                "full_length": len(candidates),
                "actual_length": len(actual),
                "collided": collided,
                "collision_at": collision_at,
                "blocked_by": blocked_by,
                "drew_color": draw_color,
            }
        )

    return out, infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each anti-diagonal segment up-right to the next longer relay position, cycling color unless a collision truncates it.

    Input structure:
    - The background is black (0).
    - There are 2-3 separated one-pixel-wide segments on anti-diagonals running down-left / up-right.
    - Each segment is monochrome, uses a distinct color from red (2), blue (1), and green (3), and has length 1-3.

    Rule:
    - Sort the segments by their highest pixel: smaller row first, and for ties smaller column first.
    - Process them in that order. Erase the current segment, then try to draw a new segment of length old_length + 1 starting immediately beyond its upper-right endpoint on the same anti-diagonal.
    - Check the candidate cells from nearest to farthest. Stop before the first occupied or out-of-bounds cell.
    - If the segment reaches its full new length, recolor it by the visible cycle red -> blue -> green -> red. If it is truncated by a collision but still draws at least one cell, keep its original color. If it draws zero cells, it disappears.

    Key insight:
    - The segments do not transform independently: processing order decides which later segments are truncated, and truncation changes both the final length and whether the color cycle happens.
    """
    out, _infos = _simulate(np.asarray(grid, dtype=int), truncate_keeps_color=True)
    return out


def _build_grid(shape: tuple[int, int], specs: list[dict]) -> np.ndarray | None:
    h, w = shape
    grid = np.zeros((h, w), dtype=int)
    occupied: set[tuple[int, int]] = set()
    for spec in specs:
        coords = _segment_coords(spec["diag_sum"], spec["c0"], spec["length"])
        if any(not (0 <= r < h and 0 <= c < w) for r, c in coords):
            return None
        if any((r, c) in occupied for r, c in coords):
            return None
        if _touches_existing(occupied, coords):
            return None
        for r, c in coords:
            grid[r, c] = spec["color"]
        occupied.update(coords)
    return grid


def _sample_diag_bounds(h: int, w: int, diag_sum: int) -> tuple[int, int]:
    lo = max(0, diag_sum - (h - 1))
    hi = min(w - 1, diag_sum)
    return lo, hi


def _clean_specs(rng: random.Random, count: int, colors: list[int], h: int, w: int) -> list[dict] | None:
    for _ in range(120):
        specs: list[dict] = []
        used_diags: set[int] = set()
        for color in colors:
            length = rng.randint(1, 3)
            diag_sum = rng.randint(3, h + w - 5)
            while diag_sum in used_diags:
                diag_sum = rng.randint(3, h + w - 5)
            used_diags.add(diag_sum)
            lo, hi = _sample_diag_bounds(h, w, diag_sum)
            max_c0 = hi - (2 * length)
            min_c0 = lo
            if max_c0 < min_c0:
                break
            c0 = rng.randint(min_c0, max_c0)
            specs.append({"diag_sum": diag_sum, "c0": c0, "length": length, "color": color})
        if len(specs) != count:
            continue
        return specs
    return None


def _collision_pair_specs(rng: random.Random, colors: list[int], h: int, w: int) -> list[dict] | None:
    for _ in range(180):
        upper_color, lower_color = colors[:2]
        len_upper, len_lower = rng.choice([(1, 2), (1, 3), (2, 3)])
        gap = 1
        diag_sum = rng.randint(4, h + w - 6)
        lo, hi = _sample_diag_bounds(h, w, diag_sum)

        total_span = len_lower + gap + len_upper
        min_a0 = lo + len_lower + gap
        max_a0 = hi - (2 * len_upper)
        if max_a0 < min_a0:
            continue
        a0 = rng.randint(min_a0, max_a0)
        b0 = a0 - gap - len_lower
        if b0 < lo:
            continue

        specs = [
            {"diag_sum": diag_sum, "c0": a0, "length": len_upper, "color": upper_color},
            {"diag_sum": diag_sum, "c0": b0, "length": len_lower, "color": lower_color},
        ]

        if len(colors) == 3:
            extra_color = colors[2]
            placed = False
            for _extra_try in range(80):
                extra_len = rng.randint(1, 3)
                extra_diag = rng.randint(3, h + w - 5)
                if extra_diag == diag_sum:
                    continue
                lo2, hi2 = _sample_diag_bounds(h, w, extra_diag)
                max_c0 = hi2 - (2 * extra_len)
                if max_c0 < lo2:
                    continue
                c0 = rng.randint(lo2, max_c0)
                candidate = specs + [
                    {"diag_sum": extra_diag, "c0": c0, "length": extra_len, "color": extra_color}
                ]
                if _build_grid((h, w), candidate) is not None:
                    specs = candidate
                    placed = True
                    break
            if not placed:
                continue

        grid = _build_grid((h, w), specs)
        if grid is None:
            continue
        return specs
    return None


def _semantic_checks(grid: np.ndarray, scenario: str) -> bool:
    try:
        out, infos = _simulate(grid, truncate_keeps_color=True)
        alt_cycle, _ = _simulate(grid, truncate_keeps_color=False)
        alt_reverse, _ = _simulate(grid, truncate_keeps_color=True, reverse_order=True)
        alt_ignore, _ = _simulate(grid, truncate_keeps_color=True, ignore_collisions=True)
    except AssertionError:
        return False

    if np.array_equal(grid, out):
        return False

    segments = _extract_segments(grid)
    if not (2 <= len(segments) <= 3):
        return False
    if len({seg["color"] for seg in segments}) != len(segments):
        return False

    collision_infos = [info for info in infos if info["collided"]]
    clean_infos = [info for info in infos if not info["collided"]]

    if scenario == "clean":
        if collision_infos:
            return False
        if not clean_infos:
            return False
        if not np.array_equal(out, alt_ignore):
            return False
    else:
        if not collision_infos:
            return False
        if not any(info["actual_length"] > 0 for info in collision_infos):
            return False
        if np.array_equal(out, alt_ignore):
            return False
        if np.array_equal(out, alt_cycle):
            return False
        if np.array_equal(out, alt_reverse):
            return False
        if not clean_infos:
            return False

    return True


def _scenario_spec(index: int) -> tuple[str, int]:
    cycle = index % 4
    if cycle == 0:
        return "clean", 3
    if cycle == 1:
        return "collision", 3
    if cycle == 2:
        return "clean", 2
    return "collision", 3


def _gen_attempt(index: int) -> np.ndarray | None:
    rng = random.Random(random.randrange(1 << 30) ^ (index * 10007))
    scenario, count = _scenario_spec(index)
    h = rng.randint(13, 18)
    w = rng.randint(13, 18)

    if count == 3:
        colors = list(PALETTE)
        rng.shuffle(colors)
    else:
        missing = index % 3
        colors = [color for i, color in enumerate(PALETTE) if i != missing]
        rng.shuffle(colors)

    if scenario == "clean":
        specs = _clean_specs(rng, count, colors, h, w)
    else:
        specs = _collision_pair_specs(rng, colors, h, w)
    if specs is None:
        return None

    grid = _build_grid((h, w), specs)
    if grid is None:
        return None
    if not _semantic_checks(grid, scenario):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_COUNTER
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(_GEN_COUNTER)
        _GEN_COUNTER += 1
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid anti-diagonal relay input")
