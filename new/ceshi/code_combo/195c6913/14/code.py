# Combo variant of 195c6913 (index 14)
# combo_spec: D8/0+D13/2
# Name: 输出裁剪为外接矩形+终止块移到终点

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import label_mask


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior:
#    Start from the original alternating-turn particle task, but change two
#    output stages together. First, the 2x2 terminal block is no longer simply
#    erased: after each marker finishes, a 2x2 copy of that block is placed in
#    the bg pocket on the blocked-turn side of the final path cell. Second, the
#    full-grid result is not shown directly; instead, we crop to the smallest
#    rectangle that contains every visible transformed object that explains the
#    motion: the start markers, the recolored trajectory cells, the single-cell
#    collision stamps, and the moved 2x2 terminal block(s). Example: if a marker
#    starts at (16,0), travels right/up through the band, leaves two corner
#    stamps, and stops near (8,13) with its blocked turn pointing upward, then
#    the output contains the repainted path, those corner stamps, a 2x2 end
#    block just above the stop, and finally only the bounding box covering that
#    entire visible cluster.
# b) Complexity calibration:
#    The crop is driven by a small, repeated set of visible objects rather than
#    by hidden coordinates or color ids. The moved end blocks reuse the existing
#    end color, so the new behavior is visually strong without adding extra
#    palettes or too many categories. With 1-2 markers per sample, the same rule
#    repeats enough times to be inferable from a few examples.
# c) Hidden-assumption audit against the original main():
#    - The original output could safely erase the end block forever; that no
#      longer holds because the end block becomes a visible moved object.
#    - The original solver only needed visited path cells. The combo must also
#      retain stop geometry so it can derive both the moved-block placement and
#      the crop bounds.
#    - Returning the full grid depended on the implicit assumption that output
#      coordinates were fixed. The crop variant breaks that assumption, so the
#      solver must explicitly collect coverage cells and slice the final grid.
#    - A stop caused by out-of-bounds would make the moved-block placement
#      ambiguous for the combo. The generator therefore rejects such cases and
#      only keeps samples where every marker stops by blocked scheduled turn.
# d) Placement legality:
#    The generator reuses the exact simulation helper from the solver, computes
#    every moved-block anchor, and rejects samples where a moved 2x2 block would
#    leave the board, overlap path/template/original end block/markers, or clash
#    with another moved block. It also rejects samples whose crop would fail to
#    shrink the canvas horizontally or vertically.
# e) Visual observability:
#    A human sees two coupled differences from the original task: the end-color
#    2x2 block reappears beside the trajectory endpoint, and the whole scene is
#    framed tightly around the active region instead of around the whole input.
#    Both differences are directly visible from geometry alone.
# f) Visual cleanliness:
#    The generator keeps moved blocks in bg pockets beside endpoints and removes
#    the template from the cropped view. This avoids stacking too many object
#    types on top of the path itself and keeps the cropped output legible.
# g) Change inferability:
#    No new arbitrary color mapping is introduced. The same end color marks both
#    corner collisions and the moved 2x2 block, which gives the solver a clear
#    cue that the source end block object has been relocated rather than deleted.
# h) Variant-specific success metric:
#    A valid generated sample must satisfy two combo signals at once:
#    `moved_blocks == number_of_markers` and `cropped_shape != input_shape`
#    with both width and height reduced relative to the input.
# i) New solver assumptions checked by the generator:
#    - The input contains exactly one detectable source 2x2 end block.
#    - Every marker has a valid path simulation and stops via blocked turn.
#    - The coverage set used for cropping is non-empty and yields a strict crop.


def _assert_path_single_component(g: np.ndarray, path_color: int) -> None:
    """Eval invariant: path_color forms a single 4-connected component."""
    _, n_path = label_mask(g == path_color, connectivity=4)
    assert n_path == 1, f"path_color must be a single 4-connected component, got {n_path}"


def _assert_marker_embedded_in_path(
    g: np.ndarray, markers: list[tuple[int, int, int]], path_color: int
) -> None:
    """Eval invariant: each marker at col 0 is embedded in a vertical path run."""
    H, _ = g.shape
    for r, c, _ in markers:
        if c != 0:
            continue
        if r > 0:
            assert int(g[r - 1, c]) == path_color, (
                f"marker at ({r},{c}) up-neighbor ({r-1},{c})={int(g[r-1,c])} "
                f"must be path_color {path_color}"
            )
        if r < H - 1:
            assert int(g[r + 1, c]) == path_color, (
                f"marker at ({r},{c}) down-neighbor ({r+1},{c})={int(g[r+1,c])} "
                f"must be path_color {path_color}"
            )


def _detect_pattern_and_end(
    g: np.ndarray, template_bg: int, path_color: int
) -> tuple[list[int], list[tuple[int, int]], int | None, tuple[int, int] | None]:
    pattern: list[int] = []
    pat_cells: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    H, W = g.shape
    for r in range(min(6, H - 1)):
        for c in range(min(25, W - 1)):
            if (int(r), int(c)) in seen:
                continue
            v = int(g[int(r), int(c)])
            if v == int(template_bg) or v == int(path_color):
                continue
            if (
                int(g[int(r + 1), int(c)]) == v
                and int(g[int(r), int(c + 1)]) == v
                and int(g[int(r + 1), int(c + 1)]) == v
            ):
                pattern.append(int(v))
                for rr, cc in ((r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)):
                    seen.add((int(rr), int(cc)))
                    pat_cells.append((int(rr), int(cc)))

    end_color = None
    end_pos = None
    pat_set = set(map(int, pattern))
    for r in range(H - 1):
        for c in range(W - 1):
            v = int(g[int(r), int(c)])
            if v == int(template_bg) or v == int(path_color) or v in pat_set:
                continue
            if (
                int(g[int(r + 1), int(c)]) == v
                and int(g[int(r), int(c + 1)]) == v
                and int(g[int(r + 1), int(c + 1)]) == v
            ):
                end_color = int(v)
                end_pos = (int(r), int(c))
                break
        if end_pos is not None:
            break
    return pattern, pat_cells, end_color, end_pos


def _detect_markers(
    g: np.ndarray, path_color: int, pattern: list[int]
) -> list[tuple[int, int, int]]:
    H, W = g.shape
    pat_set = set(map(int, pattern))
    markers: list[tuple[int, int, int]] = []
    for r in range(H):
        for c in range(W):
            v = int(g[int(r), int(c)])
            if v not in pat_set:
                continue
            same = 0
            has_path = False
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < H and 0 <= cc < W:
                    if int(g[rr, cc]) == v:
                        same += 1
                    if int(g[rr, cc]) == int(path_color):
                        has_path = True
            if same == 0 and has_path:
                markers.append((int(r), int(c), int(v)))
    return markers


def _moved_block_anchor(
    r: int, c: int, dr: int, dc: int, tdr: int, tdc: int
) -> tuple[int, int]:
    """Anchor the moved 2x2 end block on the blocked-turn side of the stop cell."""
    if (tdr, tdc) == (-1, 0):
        return int(r - 2), int(c if dc >= 0 else c - 1)
    if (tdr, tdc) == (1, 0):
        return int(r + 1), int(c if dc >= 0 else c - 1)
    if (tdr, tdc) == (0, -1):
        return int(r if dr >= 0 else r - 1), int(c - 2)
    if (tdr, tdc) == (0, 1):
        return int(r if dr >= 0 else r - 1), int(c + 1)
    raise ValueError(f"invalid turn direction {(tdr, tdc)}")


def _simulate_marker(
    g: np.ndarray,
    pattern: list[int],
    path_color: int,
    end_color: int | None,
    marker: tuple[int, int, int],
) -> dict:
    H, W = g.shape
    sr, sc, sv = marker
    dir_order = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # R, D, L, U

    init = None
    for dr, dc in dir_order:
        rr = int(sr + dr)
        cc = int(sc + dc)
        if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(path_color):
            init = (int(dr), int(dc))
            break
    if init is None or not pattern:
        return {
            "visited_cells": set(),
            "paint_steps": [],
            "collision_marks": [],
            "stop": "no_init",
            "block_anchor": None,
        }

    r, c = int(sr), int(sc)
    dr, dc = init
    idx = int(pattern.index(int(sv))) if int(sv) in pattern else 0
    hits = 0
    visited: set[tuple[int, int, int, int, int, int]] = set()
    my_cells: set[tuple[int, int]] = set()
    paint_steps: list[tuple[int, int, int]] = []
    collision_marks: list[tuple[int, int]] = []

    for _ in range(H * W * 4):
        state = (int(r), int(c), int(dr), int(dc), int(idx), int(hits & 1))
        if state in visited:
            return {
                "visited_cells": my_cells,
                "paint_steps": paint_steps,
                "collision_marks": collision_marks,
                "stop": "loop",
                "block_anchor": None,
            }
        visited.add(state)

        nr = int(r + dr)
        nc = int(c + dc)
        if not (0 <= nr < H and 0 <= nc < W):
            return {
                "visited_cells": my_cells,
                "paint_steps": paint_steps,
                "collision_marks": collision_marks,
                "stop": "out_of_bounds",
                "block_anchor": None,
            }

        if int(g[nr, nc]) == int(path_color):
            idx = int((idx + 1) % len(pattern))
            my_cells.add((int(nr), int(nc)))
            paint_steps.append((int(nr), int(nc), int(pattern[idx])))
            r, c = nr, nc
            continue

        if int(g[nr, nc]) == int(g[0, 0]) and end_color is not None:
            collision_marks.append((int(nr), int(nc)))

        left = (-int(dc), int(dr))
        right = (int(dc), -int(dr))
        tdr, tdc = left if (int(hits) % 2 == 0) else right
        rr = int(r + tdr)
        cc = int(c + tdc)
        if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(path_color):
            dr, dc = int(tdr), int(tdc)
            hits += 1
            continue

        return {
            "visited_cells": my_cells,
            "paint_steps": paint_steps,
            "collision_marks": collision_marks,
            "stop": "blocked_turn",
            "block_anchor": _moved_block_anchor(int(r), int(c), int(dr), int(dc), int(tdr), int(tdc)),
        }

    return {
        "visited_cells": my_cells,
        "paint_steps": paint_steps,
        "collision_marks": collision_marks,
        "stop": "loop_cap",
        "block_anchor": None,
    }


def _crop_bounds(cells: set[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    if not cells:
        return None
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _cover_cells_for_marker(
    marker: tuple[int, int, int], sim: dict
) -> set[tuple[int, int]]:
    sr, sc, _ = marker
    cells = {(int(sr), int(sc))}
    cells.update((int(r), int(c)) for r, c in sim["visited_cells"])
    cells.update((int(r), int(c)) for r, c in sim["collision_marks"])
    anchor = sim["block_anchor"]
    if sim["stop"] == "blocked_turn" and anchor is not None:
        ar, ac = anchor
        for rr in range(int(ar), int(ar + 2)):
            for cc in range(int(ac), int(ac + 2)):
                cells.add((int(rr), int(cc)))
    return cells


def _solve_with_metadata(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.array(grid, dtype=int)
    H, W = g.shape

    template_bg = int(g[0, 0])
    counts = Counter(map(int, g.flatten()))
    path_color = int(
        max([(cnt, int(col)) for col, cnt in counts.items() if int(col) != int(template_bg)])[1]
    )
    _assert_path_single_component(g, path_color)

    pattern, pat_cells, end_color, end_pos = _detect_pattern_and_end(g, int(template_bg), int(path_color))
    markers = _detect_markers(g, int(path_color), pattern)
    _assert_marker_embedded_in_path(g, markers, int(path_color))

    out = g.copy()
    for r, c in pat_cells:
        out[int(r), int(c)] = int(template_bg)
    if end_pos is not None:
        er, ec = end_pos
        out[int(er) : int(er + 2), int(ec) : int(ec + 2)] = int(template_bg)

    sims: list[dict] = []
    per_marker_cells: list[set[tuple[int, int]]] = []
    moved_anchors: list[tuple[int, int]] = []
    cover_cells: set[tuple[int, int]] = set()

    for marker in markers:
        sim = _simulate_marker(g, pattern, int(path_color), end_color, marker)
        sims.append(sim)
        per_marker_cells.append(set(sim["visited_cells"]))

        for rr, cc, color in sim["paint_steps"]:
            out[int(rr), int(cc)] = int(color)
        if end_color is not None:
            for rr, cc in sim["collision_marks"]:
                out[int(rr), int(cc)] = int(end_color)

        anchor = sim["block_anchor"]
        if end_color is not None and sim["stop"] == "blocked_turn" and anchor is not None:
            ar, ac = anchor
            assert 0 <= ar and 0 <= ac and ar + 1 < H and ac + 1 < W, "moved end block must stay on-grid"
            out[int(ar) : int(ar + 2), int(ac) : int(ac + 2)] = int(end_color)
            moved_anchors.append((int(ar), int(ac)))

        cover_cells.update(_cover_cells_for_marker(marker, sim))

    for i in range(len(per_marker_cells)):
        for j in range(i + 1, len(per_marker_cells)):
            overlap = per_marker_cells[i] & per_marker_cells[j]
            assert not overlap, f"marker trajectories {i},{j} overlap on {len(overlap)} cell(s)"

    if end_color is not None:
        assert len(moved_anchors) == len(markers), "every marker must receive one moved end block"
        assert len(set(moved_anchors)) == len(moved_anchors), "moved end blocks must not overlap"

    bounds = _crop_bounds(cover_cells)
    assert bounds is not None, "cropped output needs at least one visible transformed cell"

    return out, {
        "grid_shape": (int(H), int(W)),
        "markers": markers,
        "sims": sims,
        "moved_anchors": moved_anchors,
        "end_color": end_color,
        "end_pos": end_pos,
        "cover_cells": cover_cells,
        "crop_bounds": bounds,
    }


def _combo_metrics(grid: np.ndarray) -> dict:
    out, meta = _solve_with_metadata(grid)
    r0, c0, r1, c1 = meta["crop_bounds"]
    crop = out[int(r0) : int(r1 + 1), int(c0) : int(c1 + 1)]
    H, W = meta["grid_shape"]
    return {
        "input_shape": (int(H), int(W)),
        "crop_shape": tuple(map(int, crop.shape)),
        "crop_bounds": tuple(map(int, meta["crop_bounds"])),
        "n_markers": len(meta["markers"]),
        "moved_blocks": len(meta["moved_anchors"]),
        "coverage_area": len(meta["cover_cells"]),
        "crop_area": int(crop.shape[0] * crop.shape[1]),
        "full_area": int(H * W),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate alternating-turn particles, move the terminal 2x2 block to each endpoint, then crop the result.

    Input structure:
    - A top-left row of 2x2 monochrome blocks encodes the repeating paint cycle.
    - One or two isolated cycle-colored pixels on the left edge act as launch markers embedded in the path band.
    - A separate 2x2 monochrome block of a distinct color is the terminal block that will be relocated.

    Rule:
    - Each marker advances through the path, repainting visited path cells by cycling through the template colors.
    - Whenever forward motion hits bg, that collision cell is stamped with the terminal color.
    - Planned turns alternate CCW, then CW, then CCW, then CW; if the scheduled turn is not path, the particle stops.
    - The template is erased, the source terminal block is removed, a 2x2 copy of it is placed beside each final stop, and the whole output is cropped to the smallest rectangle covering the visible trajectory objects and moved terminal block(s).

    Key insight: The path dynamics are still governed by strict alternating turns, but the final answer now depends on two post-simulation bookkeeping steps: reconstructing the stop geometry for the moved 2x2 block and using the transformed objects themselves to define the cropped output window.
    """
    out, meta = _solve_with_metadata(grid)
    r0, c0, r1, c1 = meta["crop_bounds"]
    return np.array(out[int(r0) : int(r1 + 1), int(c0) : int(c1 + 1)], dtype=int)


def _moved_blocks_are_legal(
    g: np.ndarray,
    path_color: int,
    markers: list[tuple[int, int, int]],
    sims: list[dict],
    template_row: int,
    block_cols: list[int],
    end_pos: tuple[int, int],
) -> bool:
    H, W = g.shape
    occupied = np.zeros((H, W), dtype=bool)
    occupied[g == path_color] = True
    for r in range(template_row, template_row + 2):
        for bc in block_cols:
            occupied[int(r), int(bc) : int(bc + 2)] = True
    er, ec = end_pos
    occupied[int(er) : int(er + 2), int(ec) : int(ec + 2)] = True
    for r, c, _ in markers:
        occupied[int(r), int(c)] = True

    used: set[tuple[int, int]] = set()
    for sim in sims:
        if sim["stop"] != "blocked_turn" or sim["block_anchor"] is None:
            return False
        ar, ac = sim["block_anchor"]
        if not (0 <= ar and 0 <= ac and ar + 1 < H and ac + 1 < W):
            return False
        if (ar, ac) in used:
            return False
        used.add((int(ar), int(ac)))
        if occupied[int(ar) : int(ar + 2), int(ac) : int(ac + 2)].any():
            return False
        occupied[int(ar) : int(ar + 2), int(ac) : int(ac + 2)] = True
    return True


def _crop_is_meaningful(
    g: np.ndarray, markers: list[tuple[int, int, int]], sims: list[dict]
) -> bool:
    cover_cells: set[tuple[int, int]] = set()
    moved_blocks = 0
    for marker, sim in zip(markers, sims):
        cover_cells.update(_cover_cells_for_marker(marker, sim))
        if sim["stop"] == "blocked_turn" and sim["block_anchor"] is not None:
            moved_blocks += 1

    bounds = _crop_bounds(cover_cells)
    if bounds is None:
        return False

    H, W = g.shape
    r0, c0, r1, c1 = bounds
    crop_h = int(r1 - r0 + 1)
    crop_w = int(c1 - c0 + 1)
    if moved_blocks != len(markers):
        return False
    if not (crop_h < H or crop_w < W):
        return False
    if crop_h * crop_w >= H * W:
        return False
    if not (r0 > 0 or c0 > 0 or r1 < H - 1 or c1 < W - 1):
        return False
    return True


def _add_branch(mask: np.ndarray, marker_row: int, x_turn_1: int, rise: int, x_turn_2: int) -> dict:
    """Add a single deterministic R->U->R path branch rooted at col 0."""
    top_row = int(marker_row - rise)

    # Bridge around the overwritten marker cell so the remaining path stays
    # 4-connected after the marker replaces (marker_row, 0).
    for rr in range(int(marker_row - 1), int(marker_row + 2)):
        mask[int(rr), 0] = True
        mask[int(rr), 1] = True

    mask[int(marker_row), 1 : int(x_turn_1 + 1)] = True
    mask[int(top_row) : int(marker_row + 1), int(x_turn_1)] = True
    mask[int(top_row), int(x_turn_1) : int(x_turn_2 + 1)] = True

    return {
        "marker_row": int(marker_row),
        "top_row": int(top_row),
        "x_turn_1": int(x_turn_1),
        "x_turn_2": int(x_turn_2),
    }


def _gen_attempt(n_markers: int | None = None) -> np.ndarray | None:
    H = W = random.randint(20, 30)
    if n_markers is None:
        n_markers = 2 if random.random() < 0.5 else 1

    palette = list(range(10))
    random.shuffle(palette)
    bg = palette[0]
    path_color = palette[1]
    end_color = palette[2]
    n_blocks = random.randint(2, 4)
    cycle_colors = random.sample(palette[3:], n_blocks)

    template_row = random.choice([1, 2])
    block_cols = [1, 4, 7, 10][:n_blocks]
    if block_cols[-1] + 1 >= W:
        return None

    safe_top = template_row + 5
    safe_right = W - 6
    if safe_top + 5 >= H or safe_right <= 10:
        return None

    if n_markers == 1:
        marker_rows = [random.randint(max(safe_top + 4, H // 2), H - 5)]
    else:
        upper_lo = safe_top + 3
        upper_hi = H // 2 - 1
        if upper_lo > upper_hi:
            return None
        upper = random.randint(upper_lo, upper_hi)
        lower_lo = max(upper + 6, H // 2 + 2)
        lower_hi = H - 5
        if lower_lo > lower_hi:
            return None
        lower = random.randint(lower_lo, lower_hi)
        marker_rows = [upper, lower]

    mask = np.zeros((H, W), dtype=bool)
    if n_markers == 1:
        r = marker_rows[0]
        rise = random.randint(3, min(6, r - safe_top))
        x_turn_1 = random.randint(4, max(4, W // 3))
        x_turn_2 = random.randint(x_turn_1 + 4, safe_right)
        _add_branch(mask, r, x_turn_1, rise, x_turn_2)
    else:
        upper, lower = marker_rows
        upper_rise = random.randint(3, min(5, upper - safe_top))
        lower_top_min = upper + 3
        lower_rise_max = min(6, lower - lower_top_min)
        if lower_rise_max < 3:
            return None
        lower_rise = random.randint(3, lower_rise_max)

        upper_x1 = random.randint(4, max(4, W // 3 - 1))
        upper_x2 = random.randint(upper_x1 + 4, safe_right - 2)
        lower_x1 = random.randint(max(upper_x1 + 2, 7), max(max(upper_x1 + 2, 7), W // 2))
        lower_x2 = random.randint(lower_x1 + 4, safe_right)
        if lower_x1 >= lower_x2:
            return None

        top_trunk = max(upper - 1, safe_top)
        for rr in range(top_trunk, lower + 2):
            mask[int(rr), 0] = True

        _add_branch(mask, upper, upper_x1, upper_rise, upper_x2)
        _add_branch(mask, lower, lower_x1, lower_rise, lower_x2)

    for bc in block_cols:
        for dr in range(2):
            for dc in range(2):
                rr, cc = template_row + dr, bc + dc
                if mask[rr, cc]:
                    return None

    g = np.full((H, W), bg, dtype=int)
    g[mask] = path_color

    for i, bc in enumerate(block_cols):
        g[template_row : template_row + 2, bc : bc + 2] = cycle_colors[i]

    end_candidates = []
    for r in range(H // 2, H - 3):
        for c in range(W // 2, W - 3):
            if not (g[r : r + 2, c : c + 2] == bg).all():
                continue
            r0, r1_ = max(0, r - 1), min(H, r + 3)
            c0, c1_ = max(0, c - 1), min(W, c + 3)
            if (g[r0:r1_, c0:c1_] == path_color).any():
                continue
            end_candidates.append((r, c))
    if not end_candidates:
        return None
    er, ec = random.choice(end_candidates)
    g[er : er + 2, ec : ec + 2] = end_color

    for r in marker_rows:
        g[r, 0] = cycle_colors[0]

    _, n_path_after = label_mask(g == path_color, connectivity=4)
    if n_path_after != 1:
        return None
    if g[0, 0] != bg:
        return None

    cnt = Counter(map(int, g.flatten()))
    non_bg = sorted([(count, col) for col, count in cnt.items() if col != bg], reverse=True)
    if not non_bg or non_bg[0][1] != path_color:
        return None

    pattern, _, found_end_color, found_end_pos = _detect_pattern_and_end(g, bg, path_color)
    if pattern != cycle_colors or found_end_color != end_color or found_end_pos != (er, ec):
        return None
    markers = _detect_markers(g, path_color, pattern)
    if len(markers) != n_markers:
        return None
    _assert_marker_embedded_in_path(g, markers, path_color)

    sims = [_simulate_marker(g, pattern, path_color, end_color, marker) for marker in markers]
    for sim in sims:
        if sim["stop"] != "blocked_turn":
            return None

    per_marker_cells = [set(sim["visited_cells"]) for sim in sims]
    for i in range(len(per_marker_cells)):
        for j in range(i + 1, len(per_marker_cells)):
            if per_marker_cells[i] & per_marker_cells[j]:
                return None

    if not _moved_blocks_are_legal(g, path_color, markers, sims, template_row, block_cols, (er, ec)):
        return None
    if not _crop_is_meaningful(g, markers, sims):
        return None

    try:
        _ = main(g)
    except Exception:
        return None

    return g


def generate_input() -> np.ndarray:
    target_n = 2 if random.random() < 0.5 else 1
    for _ in range(250):
        try:
            g = _gen_attempt(target_n)
        except Exception:
            g = None
        if g is not None:
            return g
    for _ in range(700):
        try:
            g = _gen_attempt()
        except Exception:
            g = None
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 950 attempts")
