# Variant of 195c6913
# Dimension: D9 — 标记数量与基数关系 (meta)
# Change: 1或2个标记（n_markers==2出现率>=50%），各标记独立模拟，轨迹必须不相交。... → 标记轨迹允许交叉

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np

from arc_utils import label_mask

# Design analysis for the D9 crossing variant
# -------------------------------------------
# a) Concretization and behavior trace
#    - The visible objects stay the same: a top-left row of 2x2 template blocks,
#      one 2x2 end block in the opposite bg region, and two isolated marker
#      pixels on col 0.
#    - The D9 change only affects how multiple markers interact after they are
#      launched. Each marker still follows the original alternating-turn motion
#      rule on the same path band.
#    - New observable behavior: if two marker traces enter the same path cell,
#      that shared path cell turns into the distinguished end color. This uses a
#      color that is already advertised by the input's end block, so the crossing
#      rule is visible without inventing a new hidden palette role.
#    - Small trace example:
#      original: marker A paints ... -> cell X becomes cycle color 6; marker B
#      is not allowed to touch X because the generator enforces disjoint tracks.
#      variant: both markers may visit X. After tracing each marker separately,
#      X is recognized as shared and is painted with end_color instead of either
#      cycle color.
#    - The changed stage is therefore the multi-marker composition step, not the
#      local single-particle dynamics.
#
# b) Complexity calibration
#    - The rule is stronger than a trivial "remove the assert": the generator
#      always emits two-marker samples and requires visible overlap, so the
#      variant is actually exercised instead of silently degenerating to the
#      base task.
#    - The crossing color has only one role/value per sample: end_color. That is
#      enough variation to be learnable from 2-4 ARC examples, unlike a large
#      unconstrained mix table.
#
# c) Implicit-assumption audit of the original main()
#    - `out[nr, nc] = pattern[idx]` was harmless because another marker could not
#      write the same path cell later. That no longer holds, so the new solver
#      records per-marker traces first and only then composes the final output.
#    - `per_marker_cells` ended with `assert not overlap`; that is the old D9
#      invariant and must be removed.
#    - Collision detection still has to read from the original input grid `g`
#      rather than the partially painted output, otherwise a later marker could
#      mistakenly treat an earlier marker's paint as walkable/non-walkable.
#    - Marker detection still relies on isolated template-colored pixels adjacent
#      to the path. Since the crossing rule only changes the output, this input
#      assumption remains valid and the generator preserves it.
#
# d) Placement legality
#    - Template blocks remain in the reserved top-left bg region and never overlap
#      the path band.
#    - The end block remains a 2x2 bg block in the bottom-right bg region and is
#      still isolated from the path by at least one cell.
#    - Markers remain embedded on col 0 with vertical path neighbors above and
#      below, so the original initialization logic still works.
#    - For n=2, the generator no longer carves a separator that forbids overlap;
#      instead it validates the traced paths and rejects layouts with too little
#      shared area or with one marker becoming visually redundant.
#
# e) Visual observability
#    - The variant must be inferable by eye, so most n=2 samples are forced to
#      contain at least two shared path cells. Single-pixel accidents are not the
#      dominant case.
#    - Shared cells use end_color, which is already visible in the input as the
#      end block. This makes the extra color in the output attributable to an
#      existing object rather than to hidden color-id arithmetic.
#
# f) Visual cleanliness
#    - The output uses only three visual roles on the path: ordinary cycle paint,
#      shared-cell end_color, and the unchanged path cells that were never
#      visited. There is no multi-layer overdraw in the final rendering.
#    - The generator also demands exclusive cells for both markers, so the task
#      does not collapse into two markers producing one indistinguishable trace.
#
# g) Color-change inferability
#    - No sorting or recoloring depends on raw color ids. The only special color
#      transition introduced by the variant is "shared path cell -> end_color",
#      and the end_color is explicitly represented by the input's 2x2 end block.
#
# Main()/generator contract review after the change
#    - main() assumes: there is a valid template cycle, at least one marker, and
#      a path_color that is the most frequent non-template-bg color. generate_input()
#      explicitly constructs and validates those conditions.
#    - main() assumes overlap should be resolved after all independent traces are
#      known. generate_input() uses the same trace helper to check overlap counts,
#      exclusive counts, and turn counts before accepting a sample.


def _assert_path_single_component(g: np.ndarray, path_color: int) -> None:
    """Eval invariant: path_color forms a single 4-connected component."""
    _, n_path = label_mask(g == path_color, connectivity=4)
    assert n_path == 1, f"path_color must be a single 4-connected component, got {n_path}"


def _assert_marker_embedded_in_path(
    g: np.ndarray, markers: list[tuple[int, int, int]], path_color: int
) -> None:
    """Eval invariant: each col-0 marker is embedded in a vertical path segment."""
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


def _extract_scene(grid: np.ndarray) -> dict[str, object]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    template_bg = int(g[0, 0])

    counts = Counter(map(int, g.flatten()))
    path_color = int(
        max([(cnt, int(col)) for col, cnt in counts.items() if int(col) != int(template_bg)])[1]
    )
    _assert_path_single_component(g, path_color)

    pattern: list[int] = []
    pat_cells: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
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
                for rr, cc in [(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)]:
                    seen.add((int(rr), int(cc)))
                    pat_cells.append((int(rr), int(cc)))

    pat_set = set(map(int, pattern))

    end_color = None
    end_pos = None
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

    markers: list[tuple[int, int, int]] = []
    for r in range(H):
        for c in range(W):
            v = int(g[int(r), int(c)])
            if v not in pat_set:
                continue
            same = 0
            has_path = False
            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
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
    dir_order = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # R, D, L, U

    init = None
    for dr, dc in dir_order:
        rr = int(sr + dr)
        cc = int(sc + dc)
        if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(path_color):
            init = (int(dr), int(dc))
            break

    if init is None or not pattern:
        return {"final": {}, "bg_hits": set(), "turns": 0}

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

    return {"final": final, "bg_hits": bg_hits, "turns": hits}


def _trace_all(scene: dict[str, object]) -> list[dict[str, object]]:
    g = scene["grid"]
    path_color = int(scene["path_color"])
    pattern = list(scene["pattern"])
    markers = list(scene["markers"])
    return [_trace_marker(g, path_color, pattern, marker) for marker in markers]


def _crossing_metrics(scene: dict[str, object], traces: list[dict[str, object]]) -> dict[str, int]:
    if len(traces) < 2:
        only = traces[0]["final"] if traces else {}
        return {
            "n_markers": len(traces),
            "overlap_cells": 0,
            "min_exclusive": len(only),
            "total_turns": sum(int(trace["turns"]) for trace in traces),
        }

    cell_sets = [set(trace["final"].keys()) for trace in traces]
    overlap = set.intersection(*cell_sets)
    exclusive_sizes = [len(cell_set - overlap) for cell_set in cell_sets]
    return {
        "n_markers": len(traces),
        "overlap_cells": len(overlap),
        "min_exclusive": min(exclusive_sizes),
        "total_turns": sum(int(trace["turns"]) for trace in traces),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate alternating-turn particles on one shared path, now allowing crossings.

    Input structure:
    - The upper-left region contains a row of 2x2 solid blocks that encodes a
      repeating paint cycle.
    - One or two isolated pixels on the left edge act as launch markers and use
      a cycle color.
    - A separate 2x2 solid block in another color marks the special terminal ink.

    Rule:
    - Each marker is traced independently with the same straight-motion and
      alternating CCW/CW turn schedule as the base task.
    - Ordinary visited path cells take the particle's cycle color; bg collision
      cells take the terminal color.
    - If multiple markers visit the same path cell, that shared cell resolves to
      the terminal color instead of keeping either particle's cycle paint.

    Key insight:
    - The path-following mechanics are unchanged; the new inference burden is to
      notice that multi-marker interaction is no longer forbidden and that shared
      path cells become the distinguished terminal color.
    """
    scene = _extract_scene(grid)
    g = scene["grid"]
    template_bg = int(scene["template_bg"])
    end_color = scene["end_color"]

    out = g.copy()
    for r, c in scene["pat_cells"]:
        out[int(r), int(c)] = int(template_bg)
    if scene["end_pos"] is not None:
        er, ec = scene["end_pos"]
        out[int(er): int(er + 2), int(ec): int(ec + 2)] = int(template_bg)

    traces = _trace_all(scene)
    cell_to_entries: defaultdict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
    for marker_idx, trace in enumerate(traces):
        for cell, (step_no, color) in trace["final"].items():
            cell_to_entries[cell].append((int(marker_idx), int(step_no), int(color)))

    for (r, c), entries in cell_to_entries.items():
        if len(entries) >= 2 and end_color is not None:
            out[int(r), int(c)] = int(end_color)
        else:
            best = max(entries, key=lambda item: (item[1], item[0]))
            out[int(r), int(c)] = int(best[2])

    if end_color is not None:
        for trace in traces:
            for r, c in trace["bg_hits"]:
                out[int(r), int(c)] = int(end_color)

    return out


def _build_band(H: int, W: int, lobe_rows: list[int] | None = None) -> np.ndarray:
    """Return a thick anti-diagonal band that embeds the markers on col 0."""
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
            w = max(2, min(W, int(round(2 * half_w[r] + w_jit[r]))))
            cc = centers[r] + cc_jit[r]
            c_left[r] = int(round(cc - w / 2))
            c_right[r] = c_left[r] + w - 1

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
    n_markers = 2

    palette = list(range(10))
    random.shuffle(palette)
    bg = palette[0]
    path_color = palette[1]
    end_color = palette[2]
    n_blocks = random.randint(2, 4)
    cycle_pool = palette[3:]
    cycle_colors = [random.choice(cycle_pool) for _ in range(n_blocks)]

    if n_markers == 1:
        marker_rows = [random.randint(H // 2, H - 4)]
    else:
        r1 = random.randint(H // 3, H // 2)
        r2 = random.randint(max(r1 + 5, 2 * H // 3), H - 4)
        marker_rows = [r1, r2]

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
    for r in marker_rows:
        if not (mask[r, 0] and mask[r - 1, 0] and mask[r + 1, 0]):
            return None

    _, n_bg = label_mask(~mask, connectivity=4)
    if n_bg < 2:
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
    if g[0, 0] != bg:
        return None

    cnt = Counter(map(int, g.flatten()))
    non_bg = sorted([(c, col) for col, c in cnt.items() if col != bg], reverse=True)
    if not non_bg or non_bg[0][1] != path_color:
        return None

    try:
        scene = _extract_scene(g)
        if len(scene["markers"]) != n_markers:
            return None
        traces = _trace_all(scene)
        metrics = _crossing_metrics(scene, traces)

        min_path_len = min(len(trace["final"]) for trace in traces) if traces else 0
        if min_path_len < 15:
            return None
        if metrics["total_turns"] < (3 if n_markers == 2 else 1):
            return None
        if n_markers == 2:
            if metrics["overlap_cells"] < 2:
                return None
            if metrics["min_exclusive"] < 6:
                return None

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
