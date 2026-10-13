from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np


BG = 0
MARK = 8
INPUT_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]


def _get_inner_window(grid: np.ndarray) -> tuple[np.ndarray, int, int]:
    g = np.asarray(grid, dtype=int)
    if (
        g.shape[0] >= 3
        and g.shape[1] >= 3
        and np.all(g[0, :] == BG)
        and np.all(g[-1, :] == BG)
        and np.all(g[:, 0] == BG)
        and np.all(g[:, -1] == BG)
    ):
        return g[1:-1, 1:-1], 1, 1
    return g.copy(), 0, 0


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, int(n) + 1) if int(n) % d == 0]


def _majority_value(values: np.ndarray) -> tuple[int, int]:
    counts = np.bincount(values.astype(int).ravel(), minlength=10)
    color = int(np.argmax(counts))
    return color, int(counts[color])


def _reconstruct_template(inner: np.ndarray, period_h: int, period_w: int) -> tuple[np.ndarray, float]:
    h, w = map(int, inner.shape)
    template = np.zeros((period_h, period_w), dtype=int)
    agree = 0
    for r in range(period_h):
        for c in range(period_w):
            vals = inner[r:h:period_h, c:w:period_w]
            color, count = _majority_value(vals)
            template[r, c] = color
            agree += count
    return template, float(agree) / float(inner.size)


def _infer_period_and_template(inner: np.ndarray) -> tuple[int, int, np.ndarray]:
    full_h, full_w = map(int, inner.shape)
    best: tuple[float, int, int, np.ndarray] | None = None
    candidate_hs = [d for d in _divisors(full_h) if full_h // d >= 2]
    candidate_ws = [d for d in _divisors(full_w) if full_w // d >= 2]
    if not candidate_hs:
        candidate_hs = [full_h]
    if not candidate_ws:
        candidate_ws = [full_w]
    for period_h in candidate_hs:
        for period_w in candidate_ws:
            template, score = _reconstruct_template(inner, period_h, period_w)
            candidate = (score, -period_h * period_w, -period_h, -period_w)
            if best is None:
                best = (score, period_h, period_w, template)
                best_key = candidate
                continue
            if candidate > best_key:
                best = (score, period_h, period_w, template)
                best_key = candidate
    assert best is not None
    _, period_h, period_w, template = best
    return int(period_h), int(period_w), template


def _axis_ratio(template: np.ndarray, axis: str) -> float:
    if axis == "h":
        mirrored = template[::-1, :]
    elif axis == "v":
        mirrored = template[:, ::-1]
    else:
        raise ValueError(axis)
    mask = (template != BG) | (mirrored != BG)
    if not np.any(mask):
        return 0.0
    return float(np.mean(template[mask] == mirrored[mask]))


def _confirmed_axes(template: np.ndarray) -> tuple[str, ...]:
    axes = []
    for axis in ("h", "v"):
        if _axis_ratio(template, axis) >= 0.8:
            axes.append(axis)
    return tuple(axes)


def _mirror_local(r: int, c: int, height: int, width: int, axis: str) -> tuple[int, int]:
    if axis == "h":
        return int(height - 1 - r), int(c)
    if axis == "v":
        return int(r), int(width - 1 - c)
    raise ValueError(axis)


def _symmetry_violation(inner: np.ndarray, r: int, c: int, period_h: int, period_w: int, axes: Iterable[str]) -> bool:
    local_r = int(r % period_h)
    local_c = int(c % period_w)
    base_r = int(r - local_r)
    base_c = int(c - local_c)
    value = int(inner[r, c])
    for axis in axes:
        mr, mc = _mirror_local(local_r, local_c, period_h, period_w, axis)
        if value != int(inner[base_r + mr, base_c + mc]):
            return True
    return False


def _symmetry_violations(inner: np.ndarray, period_h: int, period_w: int, axes: Iterable[str]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r in range(inner.shape[0]):
        for c in range(inner.shape[1]):
            if int(inner[r, c]) == BG:
                continue
            if _symmetry_violation(inner, r, c, period_h, period_w, axes):
                out.add((int(r), int(c)))
    return out


def _period_violations(inner: np.ndarray, template: np.ndarray) -> set[tuple[int, int]]:
    period_h, period_w = map(int, template.shape)
    out: set[tuple[int, int]] = set()
    for r in range(inner.shape[0]):
        for c in range(inner.shape[1]):
            if int(inner[r, c]) == BG:
                continue
            if int(inner[r, c]) != int(template[r % period_h, c % period_w]):
                out.add((int(r), int(c)))
    return out


def _symmetry_orbits(height: int, width: int, axes: Iterable[str]) -> list[list[tuple[int, int]]]:
    seen: set[tuple[int, int]] = set()
    orbits: list[list[tuple[int, int]]] = []
    axes = tuple(axes)
    for r in range(height):
        for c in range(width):
            if (r, c) in seen:
                continue
            orbit = {(int(r), int(c))}
            frontier = [(int(r), int(c))]
            while frontier:
                rr, cc = frontier.pop()
                for axis in axes:
                    nr, nc = _mirror_local(rr, cc, height, width, axis)
                    if (nr, nc) not in orbit:
                        orbit.add((nr, nc))
                        frontier.append((nr, nc))
            items = sorted(orbit)
            for item in items:
                seen.add(item)
            orbits.append(items)
    return orbits


def _exact_min_period(tile: np.ndarray) -> tuple[int, int]:
    height, width = map(int, tile.shape)
    for period_h in _divisors(height):
        for period_w in _divisors(width):
            sub = tile[:period_h, :period_w]
            tiled = np.tile(sub, (height // period_h, width // period_w))
            if np.array_equal(tile, tiled):
                return int(period_h), int(period_w)
    return int(height), int(width)


def _build_base_tile(rng: np.random.Generator, height: int, width: int, axes: tuple[str, ...], palette: list[int]) -> np.ndarray:
    tile = np.zeros((height, width), dtype=int)
    orbits = _symmetry_orbits(height, width, axes)
    order = rng.permutation(len(orbits))
    for index, orbit_index in enumerate(order):
        orbit = orbits[int(orbit_index)]
        if index < len(palette):
            color = int(palette[index])
        else:
            color = int(rng.choice(palette))
        for r, c in orbit:
            tile[r, c] = color
    return tile


def _break_unwanted_axis(rng: np.random.Generator, tile: np.ndarray, axes: tuple[str, ...], palette: list[int]) -> None:
    height, width = map(int, tile.shape)
    if axes == ("v",):
        rows = [r for r in range(height) if r != height - 1 - r]
        if not rows:
            return
        r = int(rng.choice(rows))
        mr = int(height - 1 - r)
        color_a, color_b = rng.choice(palette, size=2, replace=False)
        tile[r, :] = int(color_a)
        tile[mr, :] = int(color_b)
    elif axes == ("h",):
        cols = [c for c in range(width) if c != width - 1 - c]
        if not cols:
            return
        c = int(rng.choice(cols))
        mc = int(width - 1 - c)
        color_a, color_b = rng.choice(palette, size=2, replace=False)
        tile[:, c] = int(color_a)
        tile[:, mc] = int(color_b)


def _inject_design_asymmetry(
    rng: np.random.Generator,
    tile: np.ndarray,
    axes: tuple[str, ...],
    palette: list[int],
) -> set[tuple[int, int]]:
    height, width = map(int, tile.shape)
    used: set[tuple[int, int]] = set()
    max_groups = max(1, min(2, int((0.2 * height * width) // 2)))
    target_groups = 1 if max_groups == 1 else int(rng.integers(1, max_groups + 1))

    for _ in range(target_groups):
        candidates = []
        for r in range(height):
            for c in range(width):
                orbit = {(int(r), int(c))}
                for axis in axes:
                    orbit.add(_mirror_local(r, c, height, width, axis))
                if all(_mirror_local(r, c, height, width, axis) == (r, c) for axis in axes):
                    continue
                if orbit & used:
                    continue
                candidates.append((int(r), int(c), orbit))
        if not candidates:
            break
        r, c, orbit = candidates[int(rng.integers(len(candidates)))]
        blocked = {int(tile[rr, cc]) for rr, cc in orbit}
        choices = [color for color in palette if color not in blocked]
        if not choices:
            choices = [color for color in palette if color != int(tile[r, c])]
        old_value = int(tile[r, c])
        tile[r, c] = int(rng.choice(choices))
        if not all(_axis_ratio(tile, axis) >= 0.8 for axis in axes):
            tile[r, c] = old_value
            continue
        used |= orbit

    return {
        (r, c)
        for r in range(height)
        for c in range(width)
        if any(tile[r, c] != tile[_mirror_local(r, c, height, width, axis)] for axis in axes)
    }


def _make_tile(rng: np.random.Generator) -> tuple[np.ndarray, tuple[str, ...], set[tuple[int, int]]]:
    axis_mode = str(rng.choice(["h", "v", "both"]))
    if axis_mode == "both":
        axes = ("h", "v")
    elif axis_mode == "h":
        axes = ("h",)
    else:
        axes = ("v",)

    for _ in range(200):
        if axes == ("h", "v"):
            height = int(rng.integers(4, 7))
            width = int(rng.integers(4, 7))
        else:
            height = int(rng.integers(4, 7))
            width = int(rng.integers(4, 7))
        color_count = int(rng.integers(3, 5))
        palette = list(rng.choice(INPUT_COLORS, size=color_count, replace=False))
        tile = _build_base_tile(rng, height, width, axes, palette)
        if axes != ("h", "v"):
            _break_unwanted_axis(rng, tile, axes, palette)
        design_cells = _inject_design_asymmetry(rng, tile, axes, palette)
        if _confirmed_axes(tile) != axes:
            continue
        if _exact_min_period(tile) != (height, width):
            continue
        if len(set(map(int, tile.ravel()))) < 3:
            continue
        if not design_cells:
            continue
        return tile, axes, design_cells
    raise RuntimeError("failed to build a non-degenerate tile")


def _candidate_defect_positions(
    tile: np.ndarray,
    axes: tuple[str, ...],
    design_cells: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    height, width = map(int, tile.shape)
    out = []
    for r in range(height):
        for c in range(width):
            if (r, c) in design_cells:
                continue
            if all(_mirror_local(r, c, height, width, axis) == (r, c) for axis in axes):
                continue
            out.append((int(r), int(c)))
    return out


def _apply_defects(
    rng: np.random.Generator,
    inner: np.ndarray,
    tile: np.ndarray,
    axes: tuple[str, ...],
    design_cells: set[tuple[int, int]],
    repeat_h: int,
    repeat_w: int,
) -> list[tuple[int, int]]:
    period_h, period_w = map(int, tile.shape)
    local_choices = _candidate_defect_positions(tile, axes, design_cells)
    rng.shuffle(local_choices)
    defect_count = min(len(local_choices), int(rng.integers(2, 5)))
    assert defect_count >= 2
    chosen_locals = local_choices[:defect_count]

    globals_out: list[tuple[int, int]] = []
    used_tiles: set[tuple[int, int]] = set()
    for local_r, local_c in chosen_locals:
        tile_choices = [(tr, tc) for tr in range(repeat_h) for tc in range(repeat_w) if (tr, tc) not in used_tiles]
        if not tile_choices:
            tile_choices = [(tr, tc) for tr in range(repeat_h) for tc in range(repeat_w)]
        tr, tc = tile_choices[int(rng.integers(len(tile_choices)))]
        used_tiles.add((int(tr), int(tc)))
        global_r = int(tr * period_h + local_r)
        global_c = int(tc * period_w + local_c)
        blocked = {int(tile[local_r, local_c])}
        for axis in axes:
            mr, mc = _mirror_local(local_r, local_c, period_h, period_w, axis)
            blocked.add(int(inner[int(tr * period_h + mr), int(tc * period_w + mc)]))
        choices = [color for color in INPUT_COLORS if color not in blocked and color != MARK]
        if not choices:
            choices = [color for color in INPUT_COLORS if color != int(tile[local_r, local_c]) and color != MARK]
        inner[global_r, global_c] = int(rng.choice(choices))
        globals_out.append((global_r, global_c))
    return globals_out


def _semantic_validate(
    grid: np.ndarray,
    tile: np.ndarray,
    axes: tuple[str, ...],
    design_cells: set[tuple[int, int]],
    defects_inner: list[tuple[int, int]],
) -> None:
    period_h, period_w = map(int, tile.shape)
    inner = grid[1:-1, 1:-1]
    infer_h, infer_w, recovered = _infer_period_and_template(inner)
    assert (infer_h, infer_w) == (period_h, period_w)
    assert np.array_equal(recovered, tile)
    assert _confirmed_axes(recovered) == axes

    out = main(grid)
    expected = grid.copy()
    defect_global = {(int(r + 1), int(c + 1)) for r, c in defects_inner}
    for r, c in defect_global:
        expected[r, c] = MARK
    assert np.array_equal(out, expected)

    sym_bad_inner = _symmetry_violations(inner, period_h, period_w, axes)
    periodic_bad_inner = _period_violations(inner, recovered)
    defects_set = set((int(r), int(c)) for r, c in defects_inner)
    assert defects_set <= periodic_bad_inner
    assert defects_set <= sym_bad_inner

    repeated_design = {
        (tr * period_h + lr, tc * period_w + lc)
        for tr in range(inner.shape[0] // period_h)
        for tc in range(inner.shape[1] // period_w)
        for lr, lc in design_cells
    }
    assert repeated_design
    assert repeated_design <= sym_bad_inner
    assert repeated_design.isdisjoint(periodic_bad_inner)
    assert any((r + 1, c + 1) not in defect_global and out[r + 1, c + 1] != MARK for r, c in repeated_design)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    tile, axes, design_cells = _make_tile(rng)
    period_h, period_w = map(int, tile.shape)

    repeat_choices = [(rh, rw) for rh in range(2, 5) for rw in range(2, 5) if period_h * rh + 2 <= 30 and period_w * rw + 2 <= 30]
    if not repeat_choices:
        return None
    repeat_h, repeat_w = repeat_choices[int(rng.integers(len(repeat_choices)))]

    inner = np.tile(tile, (repeat_h, repeat_w))
    defects_inner = _apply_defects(rng, inner, tile, axes, design_cells, repeat_h, repeat_w)

    grid = np.zeros((inner.shape[0] + 2, inner.shape[1] + 2), dtype=int)
    grid[1:-1, 1:-1] = inner

    try:
        _semantic_validate(grid, tile, axes, design_cells, defects_inner)
    except AssertionError:
        return None
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark only the pixels that disagree with both the recovered tiling period and the tile's internal symmetry.

    Input structure:
    - A 2D color grid has a 1-cell black border around a fully tiled interior rectangle.
    - The interior is filled by repetitions of one non-black tile whose true period height and width divide the interior exactly.
    - The recovered tile has one or two reflection axes, while a few repeated positions may be intentionally asymmetric in every copy.

    Rule:
    - Infer the smallest period height and width with maximal per-residue agreement, and recover the tile by majority color at each residue.
    - In that recovered tile, confirm the horizontal and/or vertical reflection axes whose mirror-agreement ratio is at least 80%.
    - For each non-black interior pixel, flag a period violation if it differs from the recovered tile value at the same residue, and flag a symmetry violation if it disagrees with any confirmed mirror mate inside its own tile copy.
    - Recolor a pixel to cyan (8) only when both flags are true; otherwise leave it unchanged.

    Key insight:
    - Majority voting over periodic copies restores the intended tile, and intersecting period errors with symmetry errors removes repeated design asymmetries while isolating true one-off defects.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    inner, r0, c0 = _get_inner_window(g)
    period_h, period_w, template = _infer_period_and_template(inner)
    axes = _confirmed_axes(template)
    assert axes, "expected at least one confirmed symmetry axis"

    for r in range(inner.shape[0]):
        for c in range(inner.shape[1]):
            value = int(inner[r, c])
            if value == BG:
                continue
            period_bad = value != int(template[r % period_h, c % period_w])
            symmetry_bad = _symmetry_violation(inner, r, c, period_h, period_w, axes)
            if period_bad and symmetry_bad:
                out[r0 + r, c0 + c] = MARK
    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
