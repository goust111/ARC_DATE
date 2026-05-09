# Combo variant of 195c6913 (index 10)
# combo_spec: D4/4+D7/1
# Name: 箭头标记编码方向+L/T形模板

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import label_mask

# Design analysis for the fused D4/4 + D7/1 variant
# --------------------------------------------------
# a) Concrete behavior trace
# - Original task: a left-edge single pixel launched rightward by neighbor priority,
#   and the cycle came from a horizontal row of 2x2 blocks.
# - Fused task: the source is now a 3-cell arrowhead whose geometry fixes the first
#   move, while the cycle comes from a corner L/T template whose colors are read by
#   a clockwise sweep around the visible symbol skeleton.
# - Small example: an L template with colors
#     elbow=red, lower leg=blue, upper arm=yellow
#   is read as [red, blue, yellow]. An up-pointing yellow arrow starts at yellow,
#   so its first painted path cell becomes red; the old solver would instead try to
#   launch right and would also misread the same three blocks in scan order.
# b) Complexity calibration
# - Markers use only four unambiguous arrowhead shapes `> < ^ v`; this keeps the
#   D4 change visually obvious without introducing a large shape dictionary.
# - Templates use only two symbols, L and T, built from separated 2x2 blocks. That
#   is enough to make parsing nontrivial but still learnable from 2-4 ARC examples.
# - Template colors are sampled without repetition, and marker colors are chosen
#   from the decoded cycle rather than fixed to cycle[0], so the template order
#   materially affects the repaint sequence.
# c) Hidden-assumption audit
# - The original isolated-pixel marker search no longer applies: markers are now
#   8-connected size-3 components and must be decoded before tracing.
# - The original template scan order no longer applies: row-major collection would
#   collapse the D7 change, so the solver must recognize L/T geometry first and
#   then read the symbol using the shape-specific clockwise sweep order.
# - The old col-0 embedding invariant is obsolete. The new invariant is instead:
#   every decoded arrow tip must point to an ambient path cell.
# d) Placement legality
# - Generator markers are placed only on path cells, fully in-bounds, with their
#   forward neighbor also in the path, and with Chebyshev separation so equal-color
#   arrows remain separate 8-connected components.
# - The top-left path pocket is widened through rows 1..6 so either L or T template
#   fits without touching the path.
# - The end block stays in the bottom-right background region and remains disjoint
#   from both template and markers.
# e) Visual observability
# - Humans can see two independent cues in the input: arrow orientation decides the
#   launch heading, and the corner glyph shape decides the cycle order.
# - The generator requires at least one arrow whose decoded heading differs from the
#   old R,D,L,U fallback at its tip, preventing accidental regression to the old
#   launch rule.
# f) Visual cleanliness
# - The template stays in a dedicated top-left background pocket.
# - Arrow markers stay in the left-side path lobe, away from the template and end
#   block, so no region needs to encode multiple unrelated object types.
# g) Change inferability
# - No hidden color-id ordering is used. Only visible geometry matters.
# - L and T use read orders that are not mere cyclic shifts of row-major block
#   scanning, so the template rule has a real effect once the marker color fixes the
#   starting phase.
# New solver assumptions introduced here
# - There are 1-2 arrow markers, each an 8-connected size-3 component matching
#   exactly one arrowhead orientation.
# - The template is exactly one left-corner L or T arrangement of 2x2 monochrome
#   blocks, with distinct colors and the explicit clockwise-sweep read order below.
# - The generator enforces all of these assumptions before calling `main()` as an
#   oracle.

DIR4 = {
    "R": (0, 1),
    "D": (1, 0),
    "L": (0, -1),
    "U": (-1, 0),
}
DIR_ORDER = ["R", "D", "L", "U"]
ARROW_SHAPES = {
    "R": ((0, 0), (-1, -1), (1, -1)),
    "L": ((0, 0), (-1, 1), (1, 1)),
    "U": ((0, 0), (1, -1), (1, 1)),
    "D": ((0, 0), (-1, -1), (-1, 1)),
}
TEMPLATE_SHAPES: dict[str, dict[str, list[tuple[int, int]]]] = {
    "L": {
        "slots": [(0, 0), (0, 3), (3, 0)],
        "order": [(0, 0), (3, 0), (0, 3)],
    },
    "T": {
        "slots": [(0, 0), (0, 3), (0, 6), (3, 3)],
        "order": [(0, 3), (3, 3), (0, 0), (0, 6)],
    },
}


def _assert_path_single_component(g: np.ndarray, path_color: int) -> None:
    """Eval invariant: path_color forms a single 4-connected component."""
    _, n_path = label_mask(g == path_color, connectivity=4)
    assert n_path == 1, f"path_color must be a single 4-connected component, got {n_path}"


def _arrow_cells(tip_r: int, tip_c: int, direction: str) -> list[tuple[int, int]]:
    return [(int(tip_r + dr), int(tip_c + dc)) for dr, dc in ARROW_SHAPES[str(direction)]]


def _decode_arrow_component(coords: np.ndarray) -> tuple[int, int, str] | None:
    """Return (tip_r, tip_c, direction) for a 3-cell arrowhead component."""
    if int(coords.shape[0]) != 3:
        return None
    comp = {(int(r), int(c)) for r, c in coords}
    matches: list[tuple[int, int, str]] = []
    for direction in DIR_ORDER:
        for tip_r, tip_c in comp:
            if set(_arrow_cells(int(tip_r), int(tip_c), direction)) == comp:
                matches.append((int(tip_r), int(tip_c), direction))
    if len(matches) != 1:
        return None
    return matches[0]


def _is_solid_2x2(g: np.ndarray, r: int, c: int, color: int) -> bool:
    return (
        0 <= r < g.shape[0] - 1
        and 0 <= c < g.shape[1] - 1
        and int(g[r, c]) == int(color)
        and int(g[r + 1, c]) == int(color)
        and int(g[r, c + 1]) == int(color)
        and int(g[r + 1, c + 1]) == int(color)
    )


def _parse_template(
    g: np.ndarray, template_bg: int, path_color: int
) -> tuple[list[int], list[tuple[int, int]], str, tuple[int, int]]:
    """Read the corner L/T template and return the decoded cycle order."""
    H, W = g.shape
    blocks: dict[tuple[int, int], int] = {}
    seen: set[tuple[int, int]] = set()
    for r in range(min(10, H - 1)):
        for c in range(min(14, W - 1)):
            if (int(r), int(c)) in seen:
                continue
            v = int(g[r, c])
            if v == int(template_bg) or v == int(path_color):
                continue
            if _is_solid_2x2(g, int(r), int(c), int(v)):
                blocks[(int(r), int(c))] = int(v)
                for rr in (r, r + 1):
                    for cc in (c, c + 1):
                        seen.add((int(rr), int(cc)))

    assert blocks, "template blocks not found in top-left region"
    min_r = min(r for r, _ in blocks)
    min_c = min(c for _, c in blocks)
    norm = {(int(r - min_r), int(c - min_c)) for r, c in blocks}

    shape_name = None
    for name, spec in TEMPLATE_SHAPES.items():
        if norm == set(spec["slots"]):
            shape_name = name
            break
    assert shape_name is not None, f"unrecognized template shape: {sorted(norm)}"

    ordered_blocks = [
        (int(min_r + dr), int(min_c + dc))
        for dr, dc in TEMPLATE_SHAPES[shape_name]["order"]
    ]
    assert all(pos in blocks for pos in ordered_blocks), "template missing a required block"

    pattern = [int(blocks[pos]) for pos in ordered_blocks]
    assert len(pattern) == len(set(pattern)), "template colors must be distinct"

    pat_cells: list[tuple[int, int]] = []
    for br, bc in blocks:
        pat_cells.extend([(br, bc), (br + 1, bc), (br, bc + 1), (br + 1, bc + 1)])
    return pattern, pat_cells, str(shape_name), (int(min_r), int(min_c))


def _find_arrow_markers(
    g: np.ndarray, pattern: list[int], path_color: int
) -> list[tuple[int, int, int, str, tuple[tuple[int, int], ...]]]:
    """Find 3-cell arrow markers whose colors belong to the decoded cycle."""
    H, W = g.shape
    markers: list[tuple[int, int, int, str, tuple[tuple[int, int], ...]]] = []
    for color in sorted(set(map(int, pattern))):
        lab, n_comp = label_mask(g == int(color), connectivity=8)
        for k in range(1, int(n_comp) + 1):
            coords = np.argwhere(lab == k)
            decoded = _decode_arrow_component(coords)
            if decoded is None:
                continue
            tip_r, tip_c, direction = decoded
            dr, dc = DIR4[direction]
            nr, nc = int(tip_r + dr), int(tip_c + dc)
            assert 0 <= nr < H and 0 <= nc < W, (
                f"marker tip at ({tip_r},{tip_c}) points {direction} out of bounds"
            )
            assert int(g[nr, nc]) == int(path_color), (
                f"marker tip at ({tip_r},{tip_c}) points to ({nr},{nc})="
                f"{int(g[nr, nc])}, expected path_color {path_color}"
            )
            cells = tuple(sorted((int(r), int(c)) for r, c in coords.tolist()))
            markers.append((int(tip_r), int(tip_c), int(color), direction, cells))
    markers.sort(key=lambda x: (x[0], x[1], x[3], x[2]))
    return markers


def _assert_arrow_markers_valid(
    g: np.ndarray,
    markers: list[tuple[int, int, int, str, tuple[tuple[int, int], ...]]],
    path_color: int,
) -> None:
    H, W = g.shape
    assert 1 <= len(markers) <= 2, f"expected 1-2 arrow markers, got {len(markers)}"
    used: set[tuple[int, int]] = set()
    for tip_r, tip_c, color, direction, cells in markers:
        assert len(cells) == 3
        assert tuple(sorted(_arrow_cells(tip_r, tip_c, direction))) == cells, (
            f"marker at tip ({tip_r},{tip_c}) does not match decoded direction {direction}"
        )
        dr, dc = DIR4[direction]
        nr, nc = int(tip_r + dr), int(tip_c + dc)
        assert 0 <= nr < H and 0 <= nc < W and int(g[nr, nc]) == int(path_color)
        for r, c in cells:
            assert int(g[r, c]) == int(color)
            assert (int(r), int(c)) not in used, "marker cells must be disjoint"
            used.add((int(r), int(c)))


def _trace_marker(
    g: np.ndarray,
    out: np.ndarray,
    sr: int,
    sc: int,
    sv: int,
    init_dir: str,
    pattern: list[int],
    path_color: int,
    template_bg: int,
    end_color: int | None,
) -> set[tuple[int, int]]:
    """Run one particle trace and paint `out` in place."""
    H, W = g.shape
    r, c = int(sr), int(sc)
    dr, dc = DIR4[str(init_dir)]
    idx = int(pattern.index(int(sv)))
    hits = 0
    visited: set[tuple[int, int, int, int, int, int]] = set()
    my_cells: set[tuple[int, int]] = set()

    for _ in range(H * W * 4):
        state = (int(r), int(c), int(dr), int(dc), int(idx), int(hits & 1))
        if state in visited:
            break
        visited.add(state)

        nr = int(r + dr)
        nc = int(c + dc)
        if not (0 <= nr < H and 0 <= nc < W):
            break

        if int(g[nr, nc]) == int(path_color):
            idx = int((idx + 1) % len(pattern))
            out[nr, nc] = int(pattern[idx])
            my_cells.add((int(nr), int(nc)))
            r, c = nr, nc
            continue

        if int(g[nr, nc]) == int(template_bg) and end_color is not None:
            out[nr, nc] = int(end_color)

        left = (-int(dc), int(dr))
        right = (int(dc), -int(dr))
        tdr, tdc = left if (int(hits) % 2 == 0) else right
        rr = int(r + tdr)
        cc = int(c + tdc)
        if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(path_color):
            dr, dc = int(tdr), int(tdc)
            hits += 1
            continue
        break

    return my_cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate alternating-turn particles, but decode both the launch direction and the paint cycle from shape cues.

    Input structure:
    - Top-left: several 2x2 monochrome blocks arranged as a corner L or T symbol on background.
    - Start markers are 3-cell arrowheads in template colors, embedded inside the path region.
    - A separate 2x2 monochrome block of another color supplies the collision/end color.

    Rule:
    - First recognize the L/T corner symbol and read its colors in the symbol's clockwise sweep order to recover the repeating cycle.
    - Each arrow launches one particle from its tip in the direction encoded by the arrowhead shape.
    - As long as the next cell is path, the particle advances and paints the next cycle color; on collisions it stamps the end color and keeps the original strict CCW/CW alternating turn rule.
    - If the scheduled turn is not a path cell, that particle stops immediately.

    Key insight: The path dynamics are still the original alternating-turn walk, but the input now hides two independent controls in local geometry: arrow shape chooses the initial heading, and corner-template shape chooses the usable cycle order.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    template_bg = int(g[0, 0])
    counts = Counter(map(int, g.flatten()))
    path_color = int(
        max([(cnt, int(col)) for col, cnt in counts.items() if int(col) != int(template_bg)])[1]
    )
    _assert_path_single_component(g, path_color)

    pattern, pat_cells, _shape_name, _anchor = _parse_template(g, template_bg, path_color)
    pat_set = set(map(int, pattern))

    end_color = None
    end_pos = None
    for r in range(H - 1):
        for c in range(W - 1):
            v = int(g[r, c])
            if v == int(template_bg) or v == int(path_color) or v in pat_set:
                continue
            if _is_solid_2x2(g, int(r), int(c), int(v)):
                end_color = int(v)
                end_pos = (int(r), int(c))
                break
        if end_pos is not None:
            break

    markers = _find_arrow_markers(g, pattern, path_color)
    _assert_arrow_markers_valid(g, markers, path_color)

    out = g.copy()
    for r, c in pat_cells:
        out[int(r), int(c)] = int(template_bg)
    if end_pos is not None:
        er, ec = end_pos
        out[int(er) : int(er + 2), int(ec) : int(ec + 2)] = int(template_bg)

    per_marker_cells: list[set[tuple[int, int]]] = []
    for sr, sc, sv, init_dir, _cells in markers:
        per_marker_cells.append(
            _trace_marker(
                g,
                out,
                int(sr),
                int(sc),
                int(sv),
                str(init_dir),
                pattern,
                int(path_color),
                int(template_bg),
                end_color,
            )
        )

    for i in range(len(per_marker_cells)):
        for j in range(i + 1, len(per_marker_cells)):
            overlap = per_marker_cells[i] & per_marker_cells[j]
            assert not overlap, f"marker trajectories {i},{j} overlap on {len(overlap)} cell(s)"

    return out


def _build_band(H: int, W: int, lobe_rows: list[int] | None = None) -> np.ndarray:
    """Return a thick anti-diagonal path band with a top-left template pocket."""
    bot_center = random.randint(0, W // 4)
    centers = np.linspace(W - 1, bot_center, H).astype(float)
    half_w = np.array(
        [
            0.5 * (3 + (1.0 - abs(2.0 * r / max(1, H - 1) - 1.0)) * (W - 6))
            for r in range(H)
        ]
    )

    c_left = np.zeros(H, dtype=int)
    c_right = np.zeros(H, dtype=int)

    cc_jit = np.convolve(
        np.array([random.randint(-3, 3) for _ in range(H)], dtype=float),
        np.ones(3) / 3,
        mode="same",
    )
    w_jit = np.convolve(
        np.array([random.randint(-3, 3) for _ in range(H)], dtype=float),
        np.ones(3) / 3,
        mode="same",
    )

    for r in range(H):
        c_left_param = centers[r] - half_w[r]
        if c_left_param <= 0:
            c_left[r] = 0
            c_right[r] = int(round(centers[r] + cc_jit[r] + half_w[r] + w_jit[r] / 2))
        else:
            width = max(2, min(W, int(round(2 * half_w[r] + w_jit[r]))))
            cc = centers[r] + cc_jit[r]
            c_left[r] = int(round(cc - width / 2))
            c_right[r] = c_left[r] + width - 1

    if lobe_rows:
        lmin = min(lobe_rows) - 1
        lmax = H - 1
        for r in range(H):
            d_outside = max(0, lmin - r, r - lmax)
            c_left[r] = min(c_left[r], 2 * d_outside)

    for r in range(1, min(7, H)):
        c_left[r] = max(c_left[r], 12)
        if c_right[r] < c_left[r]:
            c_right[r] = c_left[r]

    c_right[0] = W - 1
    c_left[0] = max(c_left[0], W - 6)
    c_left[H - 1] = 0
    c_right[H - 1] = max(5, min(W - 1, c_right[H - 1]))

    for _ in range(2):
        for r in range(1, H - 1):
            if c_left[r] > c_left[r - 1] and c_left[r] > c_left[r + 1]:
                c_left[r] = max(c_left[r - 1], c_left[r + 1])

    for r in range(1, H):
        c_left[r] = min(c_left[r], c_left[r - 1] + 2)
        c_right[r] = min(c_right[r], c_right[r - 1] + 2)
        if c_left[r] > c_right[r - 1]:
            c_left[r] = c_right[r - 1]
        if c_right[r] < c_left[r - 1]:
            c_right[r] = c_left[r - 1]
        if c_right[r] < c_left[r]:
            c_right[r] = c_left[r]

    c_left = np.clip(c_left, 0, W - 1)
    c_right = np.clip(c_right, 0, W - 1)
    c_right[0] = W - 1
    c_left[H - 1] = 0

    mask = np.zeros((H, W), dtype=bool)
    for r in range(H):
        mask[r, c_left[r] : c_right[r] + 1] = True
    return mask


def _carve_fjord(mask: np.ndarray, r1: int, r2: int) -> bool:
    """Carve a bg separator between two marker channels without disconnecting the path."""
    H, W = mask.shape
    sep_h = random.randint(2, 3)
    sr_min = r1 + 2
    sr_max = r2 - 1 - sep_h
    if sr_max < sr_min:
        return False
    cur_top = random.randint(sr_min, sr_max)
    made_carve = False
    for k in range(W - 1):
        trial = mask.copy()
        for dr in range(sep_h):
            rr = int(cur_top + dr)
            if 0 <= rr < H:
                trial[rr, k] = False
        _, n_trial = label_mask(trial, connectivity=4)
        if n_trial != 1:
            break
        np.copyto(mask, trial)
        made_carve = True
        if random.random() < 0.3 and cur_top > 0:
            cur_top -= 1
    return made_carve


def _priority_dir(mask: np.ndarray, r: int, c: int) -> str | None:
    H, W = mask.shape
    for direction in DIR_ORDER:
        dr, dc = DIR4[direction]
        rr = int(r + dr)
        cc = int(c + dc)
        if 0 <= rr < H and 0 <= cc < W and bool(mask[rr, cc]):
            return direction
    return None


def _chebyshev_separated(
    cells_a: tuple[tuple[int, int], ...], cells_b: tuple[tuple[int, int], ...]
) -> bool:
    for r1, c1 in cells_a:
        for r2, c2 in cells_b:
            if max(abs(int(r1) - int(r2)), abs(int(c1) - int(c2))) <= 1:
                return False
    return True


def _collect_arrow_candidates(mask: np.ndarray, target_row: int) -> list[dict[str, object]]:
    H, W = mask.shape
    col_hi = max(5, W // 3)
    candidates: list[dict[str, object]] = []
    row_ranges = [
        range(max(1, target_row - 3), min(H - 1, target_row + 4)),
        range(1, H - 1),
    ]
    for row_range in row_ranges:
        candidates.clear()
        for direction in DIR_ORDER:
            dr, dc = DIR4[direction]
            for r in row_range:
                for c in range(1, min(W - 1, col_hi + 1)):
                    cells = _arrow_cells(int(r), int(c), direction)
                    if any(not (0 <= rr < H and 0 <= cc < W) for rr, cc in cells):
                        continue
                    if not all(bool(mask[rr, cc]) for rr, cc in cells):
                        continue
                    nr, nc = int(r + dr), int(c + dc)
                    if not (0 <= nr < H and 0 <= nc < W and bool(mask[nr, nc])):
                        continue
                    local = mask[max(0, r - 1) : min(H, r + 2), max(0, c - 1) : min(W, c + 2)]
                    if int(local.sum()) < 5:
                        continue
                    priority = _priority_dir(mask, int(r), int(c))
                    candidates.append(
                        {
                            "tip": (int(r), int(c)),
                            "direction": str(direction),
                            "cells": tuple(sorted(cells)),
                            "interesting": priority != direction,
                            "row_gap": abs(int(r) - int(target_row)),
                            "col": int(c),
                        }
                    )
        if candidates:
            random.shuffle(candidates)
            candidates.sort(
                key=lambda cand: (
                    0 if bool(cand["interesting"]) else 1,
                    int(cand["row_gap"]),
                    int(cand["col"]),
                )
            )
            return candidates[:60]
    return []


def _template_blocks(shape_name: str, top_row: int, left_col: int = 1) -> list[tuple[int, int]]:
    return [(top_row + dr, left_col + dc) for dr, dc in TEMPLATE_SHAPES[shape_name]["slots"]]


def _place_arrow_markers(
    g: np.ndarray,
    mask: np.ndarray,
    target_rows: list[int],
    marker_colors: list[int],
) -> list[dict[str, object]] | None:
    cand_lists = [_collect_arrow_candidates(mask, int(row)) for row in target_rows]
    if any(not cands for cands in cand_lists):
        return None

    if len(cand_lists) == 1:
        pool = [cand for cand in cand_lists[0] if bool(cand["interesting"])]
        if not pool:
            return None
        choice = random.choice(pool[: min(20, len(pool))])
        for r, c in choice["cells"]:
            g[int(r), int(c)] = int(marker_colors[0])
        placed = dict(choice)
        placed["color"] = int(marker_colors[0])
        return [placed]

    pairs: list[dict[str, object]] = []
    for cand_a in cand_lists[0]:
        for cand_b in cand_lists[1]:
            if not _chebyshev_separated(cand_a["cells"], cand_b["cells"]):
                continue
            interesting_count = int(bool(cand_a["interesting"])) + int(bool(cand_b["interesting"]))
            if interesting_count == 0:
                continue
            pairs.append(
                {
                    "a": cand_a,
                    "b": cand_b,
                    "interesting_count": int(interesting_count),
                    "row_gap": int(cand_a["row_gap"]) + int(cand_b["row_gap"]),
                    "col_sum": int(cand_a["col"]) + int(cand_b["col"]),
                }
            )
    if not pairs:
        return None

    def _pair_key(item: dict[str, object]) -> tuple[int, int, int]:
        return (
            -int(item["interesting_count"]),
            int(item["row_gap"]),
            int(item["col_sum"]),
        )

    random.shuffle(pairs)
    pairs.sort(key=_pair_key)

    pool = pairs[: min(12, len(pairs))]
    # Right-pointing arrows are structurally rarer because they usually agree
    # with the legacy R,D,L,U fallback. Reserve some two-marker samples for a
    # legal pair containing one R arrow while still requiring another
    # interesting arrow, so D4 remains observable and direction branches stay covered.
    r_pairs = [
        item
        for item in pairs
        if str(item["a"]["direction"]) == "R" or str(item["b"]["direction"]) == "R"
    ]
    if r_pairs:
        r_pairs.sort(key=_pair_key)
        pool = r_pairs[: min(12, len(r_pairs))]

    pick = random.choice(pool)
    results: list[dict[str, object]] = []
    for cand, color in zip([pick["a"], pick["b"]], marker_colors):
        for r, c in cand["cells"]:
            g[int(r), int(c)] = int(color)
        placed = dict(cand)
        placed["color"] = int(color)
        results.append(placed)
    return results


def _gen_attempt() -> np.ndarray | None:
    H = W = random.randint(20, 30)
    n_markers = 2 if random.random() < 0.7 else 1

    palette = list(range(10))
    random.shuffle(palette)
    bg = palette[0]
    path_color = palette[1]
    end_color = palette[2]

    template_shape = random.choice(["L", "T"])
    n_blocks = len(TEMPLATE_SHAPES[template_shape]["order"])
    cycle_colors = random.sample(palette[3:], n_blocks)

    if n_markers == 1:
        r1 = random.randint(H // 2, H - 4)
        target_rows = [int(r1)]
    else:
        r1 = random.randint(H // 3, H // 2)
        r2 = random.randint(max(r1 + 5, 2 * H // 3), H - 4)
        target_rows = [int(r1), int(r2)]

    mask = _build_band(H, W, lobe_rows=target_rows)
    if mask[0, 0]:
        return None
    _, n_path = label_mask(mask, connectivity=4)
    if n_path != 1:
        return None
    if not mask[0, :].any() or not mask[:, W - 1].any():
        return None
    if not mask[H - 1, :].any() or not mask[:, 0].any():
        return None

    if n_markers == 2:
        trial = mask.copy()
        if not _carve_fjord(trial, int(target_rows[0]), int(target_rows[1])):
            return None
        if random.random() < 0.25:
            k_restore = random.randint(2, 4)
            pre_carve = mask & ~trial
            for c in range(min(k_restore, W)):
                trial[:, c] |= pre_carve[:, c]
        _, nc = label_mask(trial, connectivity=4)
        if nc != 1:
            return None
        if not trial[0, :].any() or not trial[:, W - 1].any():
            return None
        if not trial[H - 1, :].any() or not trial[:, 0].any():
            return None
        mask = trial

    _, n_bg = label_mask(~mask, connectivity=4)
    if n_bg < 2:
        return None

    template_row = random.choice([1, 2])
    block_tops = _template_blocks(template_shape, template_row, 1)
    for br, bc in block_tops:
        if br + 1 >= H or bc + 1 >= W:
            return None
        for dr in range(2):
            for dc in range(2):
                if bool(mask[br + dr, bc + dc]):
                    return None

    g = np.full((H, W), bg, dtype=int)
    g[mask] = path_color

    for (br, bc), color in zip(block_tops, cycle_colors):
        g[br : br + 2, bc : bc + 2] = int(color)

    end_candidates = []
    for r in range(H // 2, H - 1):
        for c in range(W // 2, W - 1):
            if not (g[r : r + 2, c : c + 2] == bg).all():
                continue
            r0, r1_ = max(0, r - 1), min(H, r + 3)
            c0, c1_ = max(0, c - 1), min(W, c + 3)
            if (g[r0:r1_, c0:c1_] == path_color).any():
                continue
            end_candidates.append((int(r), int(c)))
    if not end_candidates:
        return None
    er, ec = random.choice(end_candidates)
    g[er : er + 2, ec : ec + 2] = int(end_color)

    marker_indices = (
        [random.randrange(n_blocks)]
        if n_markers == 1
        else random.sample(range(n_blocks), 2)
    )
    marker_colors = [int(cycle_colors[idx]) for idx in marker_indices]
    placed = _place_arrow_markers(g, mask, target_rows, marker_colors)
    if placed is None:
        return None

    _, n_path_after = label_mask(g == path_color, connectivity=4)
    if n_path_after != 1:
        return None
    if g[0, 0] != bg:
        return None

    cnt = Counter(map(int, g.flatten()))
    non_bg = sorted([(c, col) for col, c in cnt.items() if col != bg], reverse=True)
    if not non_bg or non_bg[0][1] != path_color:
        return None

    if not any(bool(cand["interesting"]) for cand in placed):
        return None

    try:
        _ = main(g)
    except Exception:
        return None

    return g


def generate_input() -> np.ndarray:
    for _ in range(300):
        try:
            g = _gen_attempt()
        except Exception:
            g = None
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 300 attempts")
