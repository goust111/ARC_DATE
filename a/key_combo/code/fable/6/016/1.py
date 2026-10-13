from __future__ import annotations

import importlib.util
import itertools
import random
from collections import Counter
from pathlib import Path

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_UTILS_SPEC = importlib.util.spec_from_file_location("_arc_utils_shared", _ARC_UTILS_PATH)
if _ARC_UTILS_SPEC is None or _ARC_UTILS_SPEC.loader is None:
    raise ImportError(f"could not load arc_utils from {_ARC_UTILS_PATH}")
_arc_utils = importlib.util.module_from_spec(_ARC_UTILS_SPEC)
_ARC_UTILS_SPEC.loader.exec_module(_arc_utils)

Color = _arc_utils.Color
colors_present = _arc_utils.colors_present
get_bg = _arc_utils.get_bg


MAX_GENERATION_ATTEMPTS = 500
PALETTE = tuple(range(10))
LINE_KINDS = ("row", "col", "diag_down", "diag_up")


def _tile_grid(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = map(int, tile.shape)
    H, W = map(int, shape)
    reps_r = (H + h - 1) // h
    reps_c = (W + w - 1) // w
    return np.tile(tile, (reps_r, reps_c))[:H, :W]


def _line_cells(shape: tuple[int, int], kind: str, key: int) -> list[tuple[int, int]]:
    H, W = map(int, shape)
    key = int(key)
    if kind == "row":
        return [(key, c) for c in range(W)]
    if kind == "col":
        return [(r, key) for r in range(H)]
    if kind == "diag_down":
        cells = []
        for r in range(H):
            c = r - key
            if 0 <= c < W:
                cells.append((r, c))
        return cells
    if kind == "diag_up":
        cells = []
        for r in range(H):
            c = key - r
            if 0 <= c < W:
                cells.append((r, c))
        return cells
    raise ValueError(f"unknown line kind: {kind}")


def _build_line(shape: tuple[int, int], kind: str, key: int, color: int, bg_phase: int, clean: np.ndarray) -> dict:
    cells = _line_cells(shape, kind, key)
    bg_cells = [(r, c) for r, c in cells if int(clean[r, c]) == int(bg_phase)]
    return {
        "kind": kind,
        "key": int(key),
        "color": int(color),
        "cells": cells,
        "cell_set": frozenset(cells),
        "bg_cells": bg_cells,
        "bg_cell_set": frozenset(bg_cells),
    }


def _infer_texture(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int], int]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    present = colors_present(g)
    assert len(present) >= 2, "expected at least two colors in the grid"

    best: tuple[tuple[int, int, int, int, int], np.ndarray, tuple[int, int]] | None = None
    for a, b in itertools.combinations(present, 2):
        mask = (g == int(a)) | (g == int(b))
        cover = int(mask.sum())
        if cover == 0:
            continue

        for ph in range(1, min(4, H) + 1):
            if H // ph < 2:
                continue
            for pw in range(1, min(4, W) + 1):
                if W // pw < 2:
                    continue

                tile = np.empty((ph, pw), dtype=int)
                valid = True
                for r in range(ph):
                    for c in range(pw):
                        vals = g[r::ph, c::pw]
                        vals = vals[(vals == int(a)) | (vals == int(b))]
                        if vals.size == 0:
                            valid = False
                            break
                        count_a = int(np.count_nonzero(vals == int(a)))
                        count_b = int(vals.size - count_a)
                        tile[r, c] = int(a) if count_a >= count_b else int(b)
                    if not valid:
                        break
                if not valid:
                    continue
                if len(set(map(int, tile.flatten()))) != 2:
                    continue

                clean = _tile_grid(tile, g.shape)
                mism = int(np.count_nonzero(mask & (g != clean)))
                off_mask = ~mask
                if off_mask.any():
                    underlying = clean[off_mask]
                    bg_counts = Counter(map(int, underlying.flatten()))
                    off_inconsistent = int(underlying.size - bg_counts.most_common(1)[0][1])
                else:
                    off_inconsistent = 0

                score = (-cover, off_inconsistent, mism, ph * pw, ph, pw)
                if best is None or score < best[0]:
                    best = (score, clean, (int(a), int(b)))

    assert best is not None, "failed to infer a binary periodic texture"
    _score, clean, texture_colors = best

    off_mask = ~np.isin(g, np.array(texture_colors, dtype=int))
    if off_mask.any():
        underlying = clean[off_mask]
        bg_phase = int(Counter(map(int, underlying.flatten())).most_common(1)[0][0])
    else:
        bg_phase = int(get_bg(clean))

    return clean, texture_colors, bg_phase


def _infer_line_for_color(coords: list[tuple[int, int]], clean: np.ndarray, bg_phase: int, color: int) -> dict | None:
    if len(coords) < 2:
        return None

    coord_set = {(int(r), int(c)) for r, c in coords}
    if any(int(clean[r, c]) != int(bg_phase) for r, c in coord_set):
        return None

    rows = {r for r, _c in coord_set}
    cols = {c for _r, c in coord_set}
    downs = {r - c for r, c in coord_set}
    ups = {r + c for r, c in coord_set}

    candidates: list[dict] = []
    if len(rows) == 1:
        candidates.append(_build_line(clean.shape, "row", next(iter(rows)), color, bg_phase, clean))
    if len(cols) == 1:
        candidates.append(_build_line(clean.shape, "col", next(iter(cols)), color, bg_phase, clean))
    if len(downs) == 1:
        candidates.append(_build_line(clean.shape, "diag_down", next(iter(downs)), color, bg_phase, clean))
    if len(ups) == 1:
        candidates.append(_build_line(clean.shape, "diag_up", next(iter(ups)), color, bg_phase, clean))

    valid = []
    for line in candidates:
        if len(line["cells"]) < 2:
            continue
        if coord_set.issubset(line["bg_cell_set"]):
            valid.append(line)
    if len(valid) != 1:
        return None
    return valid[0]


def _simulate_hidden_input(
    clean: np.ndarray,
    bg_phase: int,
    lines: dict[int, dict],
    order: tuple[int, ...],
) -> np.ndarray:
    out = clean.copy()
    order_index = {int(color): idx for idx, color in enumerate(order)}
    visible: dict[tuple[int, int], int] = {}
    for color, line in lines.items():
        for cell in line["bg_cells"]:
            prev = visible.get(cell)
            if prev is None or order_index[int(color)] > order_index[int(prev)]:
                visible[cell] = int(color)
    for (r, c), color in visible.items():
        out[r, c] = int(color)
    return out


def _render_output(clean: np.ndarray, lines: dict[int, dict], order: tuple[int, ...]) -> np.ndarray:
    out = clean.copy()
    for color in order:
        for r, c in lines[int(color)]["cells"]:
            out[r, c] = int(color)
    return out


def _solve_case(grid: np.ndarray) -> tuple[np.ndarray, dict[int, dict], tuple[int, ...]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D ARC grid"

    clean, texture_colors, bg_phase = _infer_texture(g)
    non_texture_colors = colors_present(g, exclude=texture_colors)
    assert non_texture_colors, "expected at least one non-texture color"

    coords_by_color = {
        int(color): [tuple(map(int, rc)) for rc in np.argwhere(g == int(color))]
        for color in non_texture_colors
    }
    line_guess = {
        int(color): _infer_line_for_color(coords, clean, bg_phase, int(color))
        for color, coords in coords_by_color.items()
    }

    best_solution: tuple[tuple[int, int, int], np.ndarray, dict[int, dict], tuple[int, ...]] | None = None

    noise_options: list[int | None] = [None, *non_texture_colors]
    for noise_color in noise_options:
        excluded_color = None if noise_color is None else int(noise_color)
        active_colors = [int(c) for c in non_texture_colors if int(c) != excluded_color]
        if not active_colors:
            continue
        if any(line_guess[c] is None for c in active_colors):
            continue

        lines = {c: line_guess[c] for c in active_colors}
        assert all(line is not None for line in lines.values())
        lines = {c: line for c, line in lines.items() if line is not None}

        edge_set: set[tuple[int, int]] = set()
        valid = True
        for r, c in np.argwhere(clean == int(bg_phase)):
            rr = int(r)
            cc = int(c)
            covering = [color for color, line in lines.items() if (rr, cc) in line["bg_cell_set"]]
            if len(covering) < 2:
                continue
            observed = int(g[rr, cc])
            if observed not in covering:
                valid = False
                break
            for lower in covering:
                if int(lower) != int(observed):
                    edge_set.add((int(lower), int(observed)))
        if not valid:
            continue

        accepted_order: tuple[int, ...] | None = None
        for order in itertools.permutations(active_colors):
            order_index = {int(color): idx for idx, color in enumerate(order)}
            if any(order_index[a] >= order_index[b] for a, b in edge_set):
                continue

            simulated = _simulate_hidden_input(clean, bg_phase, lines, order)
            if noise_color is None:
                if np.array_equal(simulated, g):
                    accepted_order = tuple(map(int, order))
                    break
                continue

            noise_mask = g == int(noise_color)
            if not noise_mask.any():
                continue
            if not np.all(clean[noise_mask] == int(bg_phase)):
                continue
            if not np.array_equal(simulated[~noise_mask], g[~noise_mask]):
                continue
            if not np.all(simulated[noise_mask] == clean[noise_mask]):
                continue
            accepted_order = tuple(map(int, order))
            break

        if accepted_order is None:
            continue

        output = _render_output(clean, lines, accepted_order)
        score = (-len(active_colors), 0 if noise_color is None else 1, -len(edge_set))
        if best_solution is None or score < best_solution[0]:
            best_solution = (score, output, lines, accepted_order)

    assert best_solution is not None, "failed to reconstruct a valid line configuration"
    _score, output, lines, order = best_solution
    return output, lines, order


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Recover complete hidden straight lines from periodic-texture samples.

    Input structure:
    - A full grid is tiled by a small two-color periodic texture.
    - Non-texture colors appear only on one texture phase, mostly as visible samples of hidden full lines.
    - Each true line color belongs to exactly one horizontal, vertical, or 45-degree full-grid line.

    Rule:
    - Infer the clean minimal-period texture.
    - Treat a non-texture color as a hidden line iff all of its visible pixels fit one valid full line.
    - Restore any remaining non-line color as texture noise.
    - Draw the recovered full lines on top of the clean texture, using visible intersections to determine stacking order.

    Key insight:
    - The texture is a visibility mask: the observed colored dots are not arbitrary noise, but samples of longer lines seen only through one periodic phase.
    """
    output, _lines, _order = _solve_case(np.asarray(input_grid, dtype=int))
    return output


def _is_exact_period(tile: np.ndarray, ph: int, pw: int) -> bool:
    h, w = map(int, tile.shape)
    if h % int(ph) != 0 or w % int(pw) != 0:
        return False
    base = tile[:ph, :pw]
    return np.array_equal(tile, _tile_grid(base, (h, w)))


def _minimal_period(tile: np.ndarray) -> tuple[int, int]:
    h, w = map(int, tile.shape)
    for ph in range(1, h + 1):
        if h % ph != 0:
            continue
        for pw in range(1, w + 1):
            if w % pw != 0:
                continue
            if _is_exact_period(tile, ph, pw):
                return int(ph), int(pw)
    raise AssertionError("unreachable")


def _sample_texture_tile(ph: int, pw: int, hole_color: int, solid_color: int) -> np.ndarray:
    area = int(ph * pw)
    min_holes = max(1, area // 3)
    max_holes = min(area - 1, area - area // 3)
    for _ in range(200):
        tile = np.full((ph, pw), int(solid_color), dtype=int)
        hole_count = random.randint(min_holes, max_holes)
        indices = random.sample(range(area), k=hole_count)
        for idx in indices:
            tile.flat[int(idx)] = int(hole_color)
        if len(set(map(int, tile.flatten()))) != 2:
            continue
        if _minimal_period(tile) != (int(ph), int(pw)):
            continue
        return tile
    raise RuntimeError("failed to sample a minimal-period texture tile")


def _candidate_generator_lines(clean: np.ndarray, bg_phase: int) -> list[dict]:
    H, W = map(int, clean.shape)
    candidates: list[dict] = []
    for r in range(H):
        line = _build_line(clean.shape, "row", r, -1, bg_phase, clean)
        if len(line["bg_cells"]) >= 4:
            candidates.append(line)
    for c in range(W):
        line = _build_line(clean.shape, "col", c, -1, bg_phase, clean)
        if len(line["bg_cells"]) >= 4:
            candidates.append(line)
    for d in range(-(W - 1), H):
        line = _build_line(clean.shape, "diag_down", d, -1, bg_phase, clean)
        if len(line["bg_cells"]) >= 4:
            candidates.append(line)
    for s in range(H + W - 1):
        line = _build_line(clean.shape, "diag_up", s, -1, bg_phase, clean)
        if len(line["bg_cells"]) >= 4:
            candidates.append(line)
    return candidates


def _pairwise_intersection(a: dict, b: dict) -> tuple[int, int] | None:
    inter = a["cell_set"] & b["cell_set"]
    if len(inter) != 1:
        return None
    return next(iter(inter))


def _valid_line_combo(clean: np.ndarray, bg_phase: int, lines: list[dict]) -> bool:
    intersections: list[tuple[int, int]] = []
    for a, b in itertools.combinations(lines, 2):
        inter = _pairwise_intersection(a, b)
        if inter is None:
            return False
        if int(clean[inter]) != int(bg_phase):
            return False
        intersections.append(inter)
    return len(intersections) == len(set(intersections))


def _visible_counts(clean: np.ndarray, bg_phase: int, lines: dict[int, dict], order: tuple[int, ...]) -> Counter:
    sim = _simulate_hidden_input(clean, bg_phase, lines, order)
    counts: Counter = Counter()
    for color in order:
        counts[int(color)] = int(np.count_nonzero(sim == int(color)))
    for color in order:
        counts[int(color)] -= int(np.count_nonzero(clean == int(color)))
    return counts


def _choose_line_combo(clean: np.ndarray, bg_phase: int, line_colors: list[int]) -> dict[int, dict] | None:
    candidates = _candidate_generator_lines(clean, bg_phase)
    random.shuffle(candidates)
    target_size = len(line_colors)
    for combo in itertools.combinations(candidates, target_size):
        if not _valid_line_combo(clean, bg_phase, list(combo)):
            continue
        lines = {
            int(color): _build_line(clean.shape, combo[idx]["kind"], combo[idx]["key"], color, bg_phase, clean)
            for idx, color in enumerate(line_colors)
        }
        order = tuple(line_colors)
        visible = _simulate_hidden_input(clean, bg_phase, lines, order)
        ok = True
        for color in line_colors:
            count = int(np.count_nonzero(visible == int(color)))
            if count < 2:
                ok = False
                break
        if ok:
            return lines
    return None


def _sample_noise_coords(
    clean: np.ndarray,
    bg_phase: int,
    occupied: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    candidates = [
        (int(r), int(c))
        for r, c in np.argwhere(clean == int(bg_phase))
        if (int(r), int(c)) not in occupied
    ]
    random.shuffle(candidates)
    for size in (3, 4, 2):
        if len(candidates) < size:
            continue
        for _ in range(100):
            coords = random.sample(candidates, k=size)
            rows = {r for r, _c in coords}
            cols = {c for _r, c in coords}
            downs = {r - c for r, c in coords}
            ups = {r + c for r, c in coords}
            if len(rows) == 1 or len(cols) == 1 or len(downs) == 1 or len(ups) == 1:
                continue
            return coords
    return None


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        ph = random.randint(1, 4)
        pw = random.randint(1, 4)
        if ph == 1 and pw == 1:
            continue

        H = random.randint(max(8, ph * 3), 22)
        W = random.randint(max(8, pw * 3), 22)

        palette = list(PALETTE)
        hole_color = random.choice(palette)
        palette.remove(hole_color)
        solid_color = random.choice(palette)
        palette.remove(solid_color)

        tile = _sample_texture_tile(ph, pw, hole_color, solid_color)
        clean = _tile_grid(tile, (H, W))

        line_count = random.choice((2, 3))
        line_colors = random.sample(palette, k=line_count)
        order = tuple(line_colors)
        lines = _choose_line_combo(clean, hole_color, line_colors)
        if lines is None:
            continue

        hidden_input = _simulate_hidden_input(clean, hole_color, lines, order)
        output = _render_output(clean, lines, order)

        occupied = set()
        for line in lines.values():
            occupied.update(line["cell_set"])

        input_grid = hidden_input.copy()
        if random.random() < 0.45 and palette:
            remaining = [c for c in palette if c not in line_colors]
            if remaining:
                noise_coords = _sample_noise_coords(clean, hole_color, occupied)
                if noise_coords is not None:
                    noise_color = random.choice(remaining)
                    for r, c in noise_coords:
                        input_grid[r, c] = int(noise_color)

        line_visible_ok = all(np.count_nonzero(input_grid == int(color)) >= 2 for color in line_colors)
        if not line_visible_ok:
            continue
        if not np.array_equal(main(input_grid), output):
            continue
        return input_grid, output
    raise RuntimeError(f"failed to generate a valid ARC case after {MAX_GENERATION_ATTEMPTS} attempts")


def generate_input() -> np.ndarray:
    return _generate_case()[0]
