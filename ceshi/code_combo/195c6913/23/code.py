# Combo variant of 195c6913 (index 23)
# combo_spec: D9/1+D17/1
# Name: 轨迹允许交叉+无分隔条

from __future__ import annotations

# Design analysis for the combo variant
# =====================================
# a) Concretization and behavior trace
# - The input keeps the original visible vocabulary: one thick anti-diagonal
#   path band, a top-left row of monochrome 2x2 template blocks, one 2x2 end
#   block in the opposite background region, and two isolated marker pixels on
#   col 0. The fused change is purely about multi-marker interaction plus the
#   path topology that makes that interaction unavoidable.
# - Original task: when there are two markers, a bg fjord or interior bg island
#   usually separates their channels, so each marker can be simulated
#   independently and their painted path cells never overlap.
# - Combo task: remove that separator entirely so both markers share one lobe and
#   one broad path region. Each marker still follows the original straight-ahead
#   walk plus strict alternating CCW/CW turning rule, but now their visited path
#   cells may coincide. Shared cells resolve by normal overwrite order: the later
#   simulated marker simply paints over the earlier one.
# - Small trace:
#   * base task: upper marker paints cell X as color 6, lower marker can never
#     reach X because a bg separator keeps the channels apart.
#   * combo task: both markers can reach X on the same shared band. If the upper
#     marker writes 6 there first and the lower marker later arrives with color
#     4, the final output shows 4 at X. Collision dots on background still use
#     the end color exactly as before.
# - The fusion is therefore not "crossings get a new special color". The visible
#   signal is that the lower/later trajectory can reuse and overwrite parts of
#   the earlier trajectory because the separator is gone.
#
# b) Complexity calibration
# - The generator always emits exactly two markers. Otherwise D17/1 would be
#   invisible and D9/1 would reduce to the base task.
# - The overwrite rule is restricted to plain chronological overdraw. No hidden
#   color algebra or pairwise mix table is introduced, so a human can infer the
#   change from a few examples.
# - To keep the effect visible, accepted samples must contain several shared path
#   cells and at least some shared cells whose earlier and later cycle colors
#   differ, so the overwrite changes what is actually seen.
#
# c) Implicit-assumption audit
# - In the original solver, `out[nr, nc] = pattern[idx]` inside each marker loop
#   was safe because generator-side geometry guaranteed disjoint trajectories.
#   That assumption is false here, so the solver now traces markers separately
#   and then composes them in marker order.
# - The old final `assert not overlap` is removed. Overlap is legal and required.
# - Walkability must still be read from the immutable input grid `g`, not from a
#   partially painted output, otherwise earlier paint would incorrectly become an
#   obstacle.
# - Order now matters. The solver uses the natural row-scan order of the two
#   col-0 markers, which is top marker first and lower marker second; the
#   generator preserves this simple geometry so "later marker" is spatially
#   stable across samples.
#
# d) Placement legality
# - Template blocks stay in the reserved top-left background pocket and never
#   overlap the path.
# - The end block stays a 2x2 background block in the bottom-right background
#   pocket and remains isolated from path by at least one cell.
# - Both markers still overwrite path cells on col 0 and must keep path-colored
#   vertical neighbors above and below.
# - The generator does not carve any fjord or island. It also rejects masks whose
#   background has more than two 4-connected components, so the "no separator"
#   topology is enforced rather than merely suggested.
#
# e) Visual observability
# - A human sees one wide shared band instead of two separated channels.
# - In the output, some shared corridor cells continue with the lower marker's
#   cycle phase rather than the upper marker's earlier phase. Because the input
#   shows both markers on one column, the spatial order of the markers provides a
#   clue for the overwrite order.
# - No rule depends on raw color ids or other non-visual metadata.
#
# f) Visual cleanliness
# - Output still uses only ordinary cycle colors on path plus end_color on
#   background collision dots. There is no extra intersection symbol, so the
#   image remains readable.
# - The generator requires each marker to keep exclusive path length in addition
#   to shared cells, preventing collapse into one indistinguishable scribble.
#
# g) Color-change inferability
# - Template blocks still explain the cycle palette and order.
# - end_color keeps its original meaning: background collision marker only.
# - The new interaction rule is purely geometric and chronological: when two
#   trajectories overlap, the later marker's ordinary cycle color remains visible.
#
# Solver/generator contract
# - main() assumes one path component, one detectable template cycle, one
#   detectable end block, and exactly two col-0 embedded markers. generate_input()
#   constructs and checks those invariants.
# - main() additionally assumes overlap should be resolved by marker order rather
#   than rejected. generate_input() validates that accepted samples truly contain
#   overlap, no separator gap on col 0, and visible overwrite cells.

from collections import Counter
import random
import sys

import numpy as np

ROOT = "/home/lixinyu/snap/new"
if ROOT not in sys.path:
    sys.path.append(ROOT)

from arc_utils import label_mask


DIR_ORDER = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # R, D, L, U


def _assert_path_single_component(g: np.ndarray, path_color: int) -> None:
    """Eval invariant: path_color forms a single 4-connected component."""
    _, n_path = label_mask(g == path_color, connectivity=4)
    assert n_path == 1, f"path_color must be a single 4-connected component, got {n_path}"


def _assert_marker_embedded_in_path(
    g: np.ndarray, markers: list[tuple[int, int, int]], path_color: int
) -> None:
    """Eval invariant: each col-0 marker sits inside a vertical path segment."""
    H, _ = g.shape
    for r, c, _ in markers:
        if c != 0:
            continue
        if r > 0:
            assert int(g[r - 1, c]) == int(path_color), (
                f"marker at ({r},{c}) up-neighbor ({r-1},{c})={int(g[r-1,c])} "
                f"must be path_color {path_color}"
            )
        if r < H - 1:
            assert int(g[r + 1, c]) == int(path_color), (
                f"marker at ({r},{c}) down-neighbor ({r+1},{c})={int(g[r+1,c])} "
                f"must be path_color {path_color}"
            )


def _extract_scene(grid: np.ndarray) -> dict[str, object]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    template_bg = int(g[0, 0])

    counts = Counter(map(int, g.flatten()))
    path_color = int(
        max((cnt, int(col)) for col, cnt in counts.items() if int(col) != int(template_bg))[1]
    )
    _assert_path_single_component(g, path_color)

    pattern: list[int] = []
    pat_cells: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for r in range(min(6, H - 1)):
        for c in range(min(25, W - 1)):
            cell = (int(r), int(c))
            if cell in seen:
                continue
            v = int(g[r, c])
            if v == template_bg or v == path_color:
                continue
            if (
                int(g[r + 1, c]) == v
                and int(g[r, c + 1]) == v
                and int(g[r + 1, c + 1]) == v
            ):
                pattern.append(v)
                for rr, cc in ((r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)):
                    seen.add((int(rr), int(cc)))
                    pat_cells.append((int(rr), int(cc)))

    pat_set = set(pattern)

    end_color = None
    end_pos = None
    for r in range(H - 1):
        for c in range(W - 1):
            v = int(g[r, c])
            if v == template_bg or v == path_color or v in pat_set:
                continue
            if (
                int(g[r + 1, c]) == v
                and int(g[r, c + 1]) == v
                and int(g[r + 1, c + 1]) == v
            ):
                end_color = int(v)
                end_pos = (int(r), int(c))
                break
        if end_pos is not None:
            break

    markers: list[tuple[int, int, int]] = []
    for r in range(H):
        for c in range(W):
            v = int(g[r, c])
            if v not in pat_set:
                continue
            same = 0
            has_path = False
            for dr, dc in DIR_ORDER:
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < H and 0 <= cc < W:
                    if int(g[rr, cc]) == v:
                        same += 1
                    if int(g[rr, cc]) == int(path_color):
                        has_path = True
            if same == 0 and has_path:
                markers.append((int(r), int(c), int(v)))

    _assert_marker_embedded_in_path(g, markers, int(path_color))

    return {
        "grid": g,
        "H": H,
        "W": W,
        "template_bg": template_bg,
        "path_color": path_color,
        "pattern": pattern,
        "pat_cells": pat_cells,
        "end_color": end_color,
        "end_pos": end_pos,
        "markers": markers,
    }


def _trace_marker(
    g: np.ndarray,
    path_color: int,
    pattern: list[int],
    marker: tuple[int, int, int],
) -> dict[str, object]:
    H, W = g.shape
    sr, sc, sv = marker

    init = None
    for dr, dc in DIR_ORDER:
        rr = int(sr + dr)
        cc = int(sc + dc)
        if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(path_color):
            init = (int(dr), int(dc))
            break

    if init is None or not pattern:
        return {"final": {}, "bg_hits": set(), "turns": 0, "path_steps": 0}

    r, c = int(sr), int(sc)
    dr, dc = init
    idx = int(pattern.index(int(sv))) if int(sv) in pattern else 0
    hits = 0
    step_no = 0
    visited: set[tuple[int, int, int, int, int, int]] = set()
    final: dict[tuple[int, int], tuple[int, int]] = {}
    bg_hits: set[tuple[int, int]] = set()

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
            step_no += 1
            final[(int(nr), int(nc))] = (int(step_no), int(pattern[idx]))
            r, c = nr, nc
            continue

        if int(g[nr, nc]) == int(g[0, 0]):
            bg_hits.add((int(nr), int(nc)))

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

    return {"final": final, "bg_hits": bg_hits, "turns": hits, "path_steps": step_no}


def _trace_all(scene: dict[str, object]) -> list[dict[str, object]]:
    g = np.array(scene["grid"], dtype=int)
    path_color = int(scene["path_color"])
    pattern = list(scene["pattern"])
    markers = list(scene["markers"])
    return [_trace_marker(g, path_color, pattern, marker) for marker in markers]


def _make_base_output(scene: dict[str, object]) -> np.ndarray:
    out = np.array(scene["grid"], dtype=int).copy()
    template_bg = int(scene["template_bg"])
    for r, c in scene["pat_cells"]:
        out[int(r), int(c)] = int(template_bg)
    if scene["end_pos"] is not None:
        er, ec = scene["end_pos"]
        out[int(er) : int(er + 2), int(ec) : int(ec + 2)] = int(template_bg)
    return out


def _compose_output(scene: dict[str, object], traces: list[dict[str, object]]) -> np.ndarray:
    out = _make_base_output(scene)
    end_color = scene["end_color"]

    for trace in traces:
        for (r, c), (_step_no, color) in trace["final"].items():
            out[int(r), int(c)] = int(color)

    if end_color is not None:
        for trace in traces:
            for r, c in trace["bg_hits"]:
                out[int(r), int(c)] = int(end_color)

    return out


def _behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    scene = _extract_scene(np.array(grid, dtype=int))
    traces = _trace_all(scene)
    markers = list(scene["markers"])
    gap_between_markers = 0
    if len(markers) >= 2:
        r1 = int(markers[0][0])
        r2 = int(markers[1][0])
        col0 = np.array(scene["grid"], dtype=int)[:, 0]
        template_bg = int(scene["template_bg"])
        gap_between_markers = int(np.count_nonzero(col0[r1 : r2 + 1] == template_bg))

    counts = []
    for trace in traces:
        counts.append(set(trace["final"].keys()))
    overlap: set[tuple[int, int]] = set()
    if len(counts) >= 2:
        overlap = counts[0] & counts[1]

    overwrite_color_changes = 0
    if len(traces) >= 2:
        first = traces[0]["final"]
        second = traces[1]["final"]
        for cell in overlap:
            if int(first[cell][1]) != int(second[cell][1]):
                overwrite_color_changes += 1

    _, n_bg = label_mask(
        np.array(scene["grid"], dtype=int) == int(scene["template_bg"]),
        connectivity=4,
    )

    exclusive_sizes = []
    if len(counts) >= 2:
        exclusive_sizes = [len(counts[0] - overlap), len(counts[1] - overlap)]
    elif len(counts) == 1:
        exclusive_sizes = [len(counts[0])]

    return {
        "n_markers": len(traces),
        "overlap_cells": len(overlap),
        "overwrite_color_changes": int(overwrite_color_changes),
        "min_exclusive": min(exclusive_sizes) if exclusive_sizes else 0,
        "total_turns": sum(int(trace["turns"]) for trace in traces),
        "min_path_steps": min((int(trace["path_steps"]) for trace in traces), default=0),
        "col0_gap_between_markers": gap_between_markers,
        "n_bg_components": int(n_bg),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate two alternating-turn particles on one shared path band where later paint may overwrite earlier paint.

    Input structure:
    - The upper-left still contains a row of monochrome 2x2 template blocks whose left-to-right order defines the repeating cycle.
    - Exactly two isolated marker pixels sit on col 0, replacing path cells inside the same left-edge path lobe.
    - A separate monochrome 2x2 block in another color marks the terminal collision color, while the path itself is one thick anti-diagonal band with no bg separator between the two markers.

    Rule:
    - Each marker follows the original motion rule independently: move straight through path cells, advance one step in the template cycle per path cell, and on each non-path collision attempt turn by the strict alternating CCW, CW, CCW, CW schedule.
    - If a collision attempt steps into background, that background cell is painted with the terminal color; if the scheduled turn is not a path cell, that marker stops.
    - The two markers are then composed in their natural top-to-bottom order. Because they share the same path region, a later marker may revisit path cells already painted by the earlier marker, and its ordinary cycle color simply overwrites the earlier one.

    Key insight:
    - The local particle dynamics are unchanged; the new inference burden is to notice that the two left-edge markers now occupy one shared corridor with no separator, so intersections are resolved by visible later-overwrites-earlier paint rather than by forbidding overlap.
    """
    scene = _extract_scene(np.array(grid, dtype=int))
    traces = _trace_all(scene)
    return _compose_output(scene, traces)


def _build_band(H: int, W: int, lobe_rows: list[int] | None = None) -> np.ndarray:
    """Return a thick anti-diagonal band that embeds the marker rows on col 0."""
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
            center = centers[r] + cc_jit[r]
            c_left[r] = int(round(center - width / 2))
            c_right[r] = c_left[r] + width - 1

    if lobe_rows:
        lmin = min(lobe_rows) - 1
        lmax = H - 1
        for r in range(H):
            d_outside = max(0, lmin - r, r - lmax)
            c_left[r] = min(c_left[r], 2 * d_outside)

    for r in range(1, min(4, H)):
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


def _gen_attempt() -> np.ndarray | None:
    H = W = random.randint(20, 30)
    marker_rows = [
        random.randint(H // 3, H // 2),
        random.randint(max(H // 2 + 2, 2 * H // 3), H - 4),
    ]
    if marker_rows[1] - marker_rows[0] < 5:
        return None

    palette = list(range(10))
    random.shuffle(palette)
    bg = palette[0]
    path_color = palette[1]
    end_color = palette[2]
    n_blocks = random.randint(2, 4)
    cycle_colors = random.sample(palette[3:], n_blocks)

    mask = _build_band(H, W, lobe_rows=marker_rows)
    if mask[0, 0]:
        return None

    _, n_path = label_mask(mask, connectivity=4)
    if n_path != 1:
        return None
    if not mask[0, :].any() or not mask[:, W - 1].any():
        return None
    if not mask[H - 1, :].any() or not mask[:, 0].any():
        return None
    if not mask[marker_rows[0] : marker_rows[1] + 1, 0].all():
        return None
    for r in marker_rows:
        if not (mask[r, 0] and mask[r - 1, 0] and mask[r + 1, 0]):
            return None

    _, n_bg = label_mask(~mask, connectivity=4)
    if n_bg != 2:
        return None

    template_row = random.choice([1, 2])
    block_cols = [1, 4, 7, 10][:n_blocks]
    if block_cols[-1] + 1 >= W:
        return None
    for bc in block_cols:
        for dr in range(2):
            for dc in range(2):
                rr, cc = template_row + dr, bc + dc
                if rr >= H or cc >= W or mask[rr, cc]:
                    return None

    g = np.full((H, W), bg, dtype=int)
    g[mask] = path_color

    for i, bc in enumerate(block_cols):
        g[template_row : template_row + 2, bc : bc + 2] = cycle_colors[i]

    end_candidates = []
    for r in range(H // 2, H - 1):
        for c in range(W // 2, W - 1):
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
    if int(g[0, 0]) != int(bg):
        return None

    cnt = Counter(map(int, g.flatten()))
    non_bg = sorted(((c, col) for col, c in cnt.items() if int(col) != int(bg)), reverse=True)
    if not non_bg or int(non_bg[0][1]) != int(path_color):
        return None

    try:
        scene = _extract_scene(g)
        if len(scene["markers"]) != 2:
            return None
        metrics = _behavior_metrics(g)
        if metrics["n_markers"] != 2:
            return None
        if metrics["col0_gap_between_markers"] != 0:
            return None
        if metrics["n_bg_components"] != 2:
            return None
        if metrics["min_path_steps"] < 15:
            return None
        if metrics["total_turns"] < 3:
            return None
        if metrics["overlap_cells"] < 2:
            return None
        if metrics["overwrite_color_changes"] < 1:
            return None
        if metrics["min_exclusive"] < 4:
            return None
        _ = main(g)
    except Exception:
        return None

    return g


def generate_input() -> np.ndarray:
    for _ in range(600):
        try:
            g = _gen_attempt()
        except Exception:
            g = None
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 600 attempts")
