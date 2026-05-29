# Combo variant of 195c6913 (index 18)
# combo_spec: D3/2+D5/2+D18/1
# Name: 周期图案涂色+完成循环后停止+5-6色长循环

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import label_mask

# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    The old task had a 1D color cycle stored as one row of monochrome 2x2
#    blocks. Here the template becomes a visible 2x3 tile of six monochrome
#    2x2 blocks. A marker still launches from col 0 and follows the same
#    alternating-turn path physics, but each visited path cell is no longer
#    painted by "next color in a row". Instead the marker reveals a 2D tiling
#    anchored at its own start cell: color(nr,nc) = tile[(nr-sr)%2,(nc-sc)%3].
#    Example: if the template rows are [A B C] / [D E F], then a right-going
#    trail from the marker shows ... A B C A B C on one row, while a row one
#    cell lower shows ... D E F D E F; turns naturally expose both rows of the
#    tile. The walk stops after exactly 36 forward moves, i.e. 6 full repeats
#    of the six-cell template cycle, even if the geometry would allow more.
# b) Complexity calibration:
#    I lock the long-cycle choice to the six-color endpoint of the requested
#    5-6 range. This keeps the 2D tile rectangular and legible; a five-color
#    partial tile would add parsing ambiguity without improving the ARC signal.
#    The quota is therefore fixed at 36 painted path cells, which is long
#    enough to make the early stop visually obvious and still easy to generate.
# c) Old-code assumptions that break under the combo:
#    The original solver assumed a flat pattern list read from one template row,
#    and that each step's color depends only on the cycle index. Both are false
#    now: parsing must recover two template rows and three columns, and paint
#    color depends on spatial phase relative to the marker. The old stopping
#    logic also broke at the first blocked scheduled turn; now the solver tracks
#    progress and the generator only accepts scenes where the quota completes.
# d) Placement legality:
#    The generator still keeps the path as one 4-connected band, embeds every
#    marker inside a col-0 path segment, reserves a clear top-left rectangle for
#    the 2x3 template, and places the 2x2 end block in isolated bottom-right
#    background. Oracle simulation additionally checks quota completion, extra
#    spare path beyond the quota, and pairwise-disjoint trajectories.
# e) Visual observability:
#    A human can read the six-color 2x3 template directly from the input and
#    see the same spatial tiling appear on the output trail. Because accepted
#    samples can continue at least one extra tile cycle beyond 36 steps, the
#    output endpoint is visibly caused by the quota, not by the first dead-end.
# f) Visual cleanliness:
#    The path band remains thick and meandering, while the template and end
#    block stay in separate background lobes. Output edits only erase the two
#    reserved objects, recolor the trail, and place sparse collision dots, so
#    the grid does not collapse into overlapping clutter.
# g) Change inferability:
#    Nothing depends on numeric color ids or invisible ordering. The only
#    template readout is geometric: a 2x3 arrangement of six visible colors.
#    The same six colors reappear on the trail with 2-row by 3-column periodic
#    structure, and the quota corresponds to six full template repeats.
# h) New solver assumptions and generator guarantees:
#    The new `main()` assumes the top-left template can be uniquely parsed as
#    six 2x2 monochrome blocks arranged on two rows and three columns, and that
#    each marker can stay on the path long enough to paint all 36 steps. The
#    generator enforces both conditions before returning a sample.

TILE_ROWS = 2
TILE_COLS = 3
TILE_CELLS = TILE_ROWS * TILE_COLS
TEMPLATE_COLS = [1, 4, 7]
_next_marker_count = 2


def _assert_path_single_component(g: np.ndarray, path_color: int) -> None:
    """Eval invariant: path_color forms a single 4-connected component."""
    _, n_path = label_mask(g == path_color, connectivity=4)
    assert n_path == 1, f"path_color must be a single 4-connected component, got {n_path}"


def _assert_marker_embedded_in_path(
    g: np.ndarray, markers: list[tuple[int, int, int]], path_color: int
) -> None:
    """Each col-0 marker must sit inside a vertical path segment."""
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


def _extract_template(
    g: np.ndarray, template_bg: int, path_color: int
) -> tuple[np.ndarray, list[int], list[tuple[int, int]]]:
    H, W = g.shape
    blocks: list[tuple[int, int, int]] = []
    pat_cells: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for r in range(min(10, H - 1)):
        for c in range(min(12, W - 1)):
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
                blocks.append((int(r), int(c), int(v)))
                for rr, cc in ((r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)):
                    seen.add((int(rr), int(cc)))
                    pat_cells.append((int(rr), int(cc)))

    blocks.sort(key=lambda x: (x[0], x[1]))
    row_keys = sorted({r for r, _, _ in blocks})
    col_keys = sorted({c for _, c, _ in blocks})
    assert len(blocks) == TILE_CELLS, f"expected {TILE_CELLS} template blocks, got {len(blocks)}"
    assert len(row_keys) == TILE_ROWS and len(col_keys) == TILE_COLS, (
        f"template must form a {TILE_ROWS}x{TILE_COLS} block tile"
    )

    tile = np.zeros((TILE_ROWS, TILE_COLS), dtype=int)
    for r, c, v in blocks:
        tile[row_keys.index(int(r)), col_keys.index(int(c))] = int(v)
    pattern = [int(tile[rr, cc]) for rr in range(TILE_ROWS) for cc in range(TILE_COLS)]
    return tile, pattern, pat_cells


def _find_end_block(
    g: np.ndarray, template_bg: int, path_color: int, pat_set: set[int]
) -> tuple[int | None, tuple[int, int] | None]:
    H, W = g.shape
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
                return int(v), (int(r), int(c))
    return None, None


def _find_markers(g: np.ndarray, pat_set: set[int], path_color: int) -> list[tuple[int, int, int]]:
    H, W = g.shape
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


def _parse_scene(grid: np.ndarray) -> dict[str, object]:
    g = np.array(grid, dtype=int)
    template_bg = int(g[0, 0])
    counts = Counter(map(int, g.flatten()))
    path_color = int(
        max((cnt, int(col)) for col, cnt in counts.items() if int(col) != int(template_bg))[1]
    )
    _assert_path_single_component(g, path_color)

    tile, pattern, pat_cells = _extract_template(g, template_bg, path_color)
    pat_set = set(map(int, pattern))
    end_color, end_pos = _find_end_block(g, template_bg, path_color, pat_set)
    markers = _find_markers(g, pat_set, path_color)
    _assert_marker_embedded_in_path(g, markers, path_color)
    assert all(int(v) == int(pattern[0]) for _, _, v in markers), "markers must use the tile's first color"

    return {
        "grid": g,
        "template_bg": template_bg,
        "path_color": path_color,
        "tile": tile,
        "pattern": pattern,
        "pat_cells": pat_cells,
        "pat_set": pat_set,
        "end_color": end_color,
        "end_pos": end_pos,
        "markers": markers,
    }


def _initial_direction(g: np.ndarray, sr: int, sc: int, path_color: int) -> tuple[int, int] | None:
    H, W = g.shape
    for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):  # R, D, L, U
        rr = int(sr + dr)
        cc = int(sc + dc)
        if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(path_color):
            return int(dr), int(dc)
    return None


def _quota_steps(pattern: list[int]) -> int:
    return int(len(pattern) * len(pattern))


def _tile_color(tile: np.ndarray, sr: int, sc: int, r: int, c: int) -> int:
    return int(tile[(int(r) - int(sr)) % tile.shape[0], (int(c) - int(sc)) % tile.shape[1]])


def _trace_marker(
    g: np.ndarray,
    tile: np.ndarray,
    path_color: int,
    template_bg: int,
    end_color: int | None,
    marker: tuple[int, int, int],
    paint_limit: int | None,
    out: np.ndarray | None = None,
) -> dict[str, object]:
    sr, sc, _sv = marker
    init = _initial_direction(g, sr, sc, path_color)
    info: dict[str, object] = {
        "init_found": init is not None,
        "cells": set(),
        "hits": 0,
        "steps": 0,
        "limit_complete": False,
        "stop_reason": "no_init" if init is None else "active",
        "final_pos": (int(sr), int(sc)),
    }
    if init is None:
        return info

    r, c = int(sr), int(sc)
    dr, dc = init
    hits = 0
    steps = 0
    visited: set[tuple[int, ...]] = set()
    H, W = g.shape

    while True:
        if paint_limit is not None and steps >= paint_limit:
            info["limit_complete"] = True
            info["stop_reason"] = "quota"
            break

        state: tuple[int, ...]
        if paint_limit is None:
            state = (int(r), int(c), int(dr), int(dc), int(hits & 1))
        else:
            state = (int(r), int(c), int(dr), int(dc), int(hits & 1), int(steps))
        if state in visited:
            info["stop_reason"] = "loop"
            break
        visited.add(state)

        nr = int(r + dr)
        nc = int(c + dc)
        if not (0 <= nr < H and 0 <= nc < W):
            info["stop_reason"] = "edge"
            break

        if int(g[nr, nc]) == int(path_color):
            paint = _tile_color(tile, sr, sc, nr, nc)
            if out is not None:
                out[nr, nc] = int(paint)
            cast_cells = info["cells"]
            assert isinstance(cast_cells, set)
            cast_cells.add((int(nr), int(nc)))
            steps += 1
            r, c = nr, nc
            continue

        if int(g[nr, nc]) == int(template_bg) and end_color is not None and out is not None:
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

        info["stop_reason"] = "blocked_turn"
        break

    info["hits"] = int(hits)
    info["steps"] = int(steps)
    info["final_pos"] = (int(r), int(c))
    return info


def _collect_traces(grid: np.ndarray, paint_limit: int) -> tuple[dict[str, object], list[dict[str, object]]]:
    scene = _parse_scene(grid)
    g = scene["grid"]
    tile = scene["tile"]
    assert isinstance(g, np.ndarray)
    assert isinstance(tile, np.ndarray)
    traces = [
        _trace_marker(
            g=g,
            tile=tile,
            path_color=int(scene["path_color"]),
            template_bg=int(scene["template_bg"]),
            end_color=None if scene["end_color"] is None else int(scene["end_color"]),
            marker=marker,
            paint_limit=paint_limit,
            out=None,
        )
        for marker in scene["markers"]
    ]
    return scene, traces


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reveal a six-color 2D tile along the path for a template-coded fixed quota.

    Input structure:
    - Top-left: six monochrome 2x2 blocks arranged as a 2x3 tile template.
    - Isolated template-colored pixels on col 0 mark particle starts inside the path band.
    - A separate 2x2 monochrome block gives the collision-dot color.

    Rule:
    - Each marker follows the same path motion as the base task: move straight on path cells, and on each forward collision attempt alternate CCW and CW turns with no fallback.
    - Every visited path cell is recolored by repeating the 2x3 template as a spatial tiling anchored at that marker, rather than by advancing a 1D color list.
    - The walk stops after exactly 6 full template cycles, i.e. 36 forward paints, while collision probes still drop the end color on background.

    Key insight:
    - The template must be read as a 2D tile, not a flat sequence: the trail is the path-shaped window of that repeating tile, and its length is fixed by template progress rather than by the first blocked turn.
    """
    scene = _parse_scene(grid)
    g = scene["grid"]
    tile = scene["tile"]
    pattern = scene["pattern"]
    pat_cells = scene["pat_cells"]
    end_pos = scene["end_pos"]
    template_bg = int(scene["template_bg"])
    quota = _quota_steps(pattern)

    out = g.copy()
    for r, c in pat_cells:
        out[int(r), int(c)] = int(template_bg)
    if end_pos is not None:
        er, ec = end_pos
        out[int(er) : int(er + 2), int(ec) : int(ec + 2)] = int(template_bg)

    per_marker_cells: list[set[tuple[int, int]]] = []
    for marker in scene["markers"]:
        trace = _trace_marker(
            g=g,
            tile=tile,
            path_color=int(scene["path_color"]),
            template_bg=template_bg,
            end_color=None if scene["end_color"] is None else int(scene["end_color"]),
            marker=marker,
            paint_limit=quota,
            out=out,
        )
        assert bool(trace["init_found"]), f"marker {marker} must touch the path"
        assert bool(trace["limit_complete"]), (
            f"marker {marker} stopped before completing the tile quota: {trace['stop_reason']}"
        )
        cells = trace["cells"]
        assert isinstance(cells, set)
        per_marker_cells.append(cells)

    for i in range(len(per_marker_cells)):
        for j in range(i + 1, len(per_marker_cells)):
            overlap = per_marker_cells[i] & per_marker_cells[j]
            assert not overlap, f"marker trajectories {i},{j} overlap on {len(overlap)} cell(s)"

    return out


def _build_band(H: int, W: int, lobe_rows: list[int] | None = None, template_row: int = 1) -> np.ndarray:
    """Return a thick meandering anti-diagonal path band."""
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

    template_stop = min(H, template_row + 6)
    for r in range(1, template_stop):
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
    """Carve a narrow background fjord from the left edge without disconnecting the path."""
    H, W = mask.shape
    sep_h = random.randint(2, 3)
    sr_min = r1 + 2
    sr_max = r2 - 1 - sep_h
    if sr_max < sr_min:
        return False
    sr = random.randint(sr_min, sr_max)

    cur_top = sr
    made_carve = False
    for k in range(W - 1):
        trial = mask.copy()
        for dr in range(sep_h):
            rr = cur_top + dr
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


def _gen_attempt(target_markers: int | None = None) -> np.ndarray | None:
    H = W = random.randint(24, 30)
    n_markers = int(target_markers) if target_markers in (1, 2) else (2 if random.random() < 0.7 else 1)

    palette = list(range(10))
    random.shuffle(palette)
    bg = palette[0]
    path_color = palette[1]
    end_color = palette[2]
    cycle_colors = random.sample(palette[3:], TILE_CELLS)
    template_row = random.choice([1, 2])

    if n_markers == 1:
        r1 = random.randint(H // 2, H - 4)
        markers = [r1]
    else:
        r1 = random.randint(H // 3, H // 2)
        r2 = random.randint(max(r1 + 6, 2 * H // 3), H - 4)
        markers = [r1, r2]

    mask = _build_band(H, W, lobe_rows=markers, template_row=template_row)
    if mask[0, 0]:
        return None
    _, n_path = label_mask(mask, connectivity=4)
    if n_path != 1:
        return None
    if not mask[0, :].any() or not mask[:, W - 1].any():
        return None
    if not mask[H - 1, :].any() or not mask[:, 0].any():
        return None
    for r in markers:
        if not (mask[r, 0] and mask[r - 1, 0] and mask[r + 1, 0]):
            return None

    if n_markers == 2:
        trial = mask.copy()
        if not _carve_fjord(trial, markers[0], markers[1]):
            return None
        if random.random() < 0.25:
            keep_cols = random.randint(2, 4)
            pre_carve_cells = mask & ~trial
            for c in range(min(keep_cols, W)):
                trial[:, c] |= pre_carve_cells[:, c]
        _, nc = label_mask(trial, connectivity=4)
        if nc != 1:
            return None
        if not trial[0, :].any() or not trial[:, W - 1].any():
            return None
        if not trial[H - 1, :].any() or not trial[:, 0].any():
            return None
        for r in markers:
            if not (trial[r, 0] and trial[r - 1, 0] and trial[r + 1, 0]):
                return None
        mask = trial

    _, n_bg = label_mask(~mask, connectivity=4)
    if n_bg < 2:
        return None

    g = np.full((H, W), bg, dtype=int)
    g[mask] = path_color

    template_rows = [template_row, template_row + 3]
    color_idx = 0
    for tr in template_rows:
        for tc in TEMPLATE_COLS:
            if tr + 1 >= H or tc + 1 >= W:
                return None
            if (g[tr : tr + 2, tc : tc + 2] != bg).any():
                return None
            g[tr : tr + 2, tc : tc + 2] = cycle_colors[color_idx]
            color_idx += 1

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

    for r in markers:
        g[r, 0] = cycle_colors[0]

    _, n_path_after = label_mask(g == path_color, connectivity=4)
    if n_path_after != 1:
        return None
    if g[0, 0] != bg:
        return None

    cnt = Counter(map(int, g.flatten()))
    non_bg = sorted(((c, col) for col, c in cnt.items() if col != bg), reverse=True)
    if not non_bg or non_bg[0][1] != path_color:
        return None

    try:
        quota = _quota_steps(cycle_colors)
        scene, quota_traces = _collect_traces(g, paint_limit=quota)
        _, beyond_traces = _collect_traces(g, paint_limit=quota + TILE_CELLS)
        _ = main(g)
    except Exception:
        return None

    markers_found = scene["markers"]
    assert isinstance(markers_found, list)
    if len(markers_found) != n_markers:
        return None
    if not quota_traces or not beyond_traces:
        return None
    if not all(bool(t["limit_complete"]) for t in quota_traces):
        return None
    if not all(bool(t["limit_complete"]) for t in beyond_traces):
        return None
    if sum(int(t["hits"]) for t in quota_traces) < 1:
        return None

    tile = scene["tile"]
    assert isinstance(tile, np.ndarray)
    for marker, trace in zip(markers_found, quota_traces):
        cells = trace["cells"]
        assert isinstance(cells, set)
        seen_colors = {_tile_color(tile, marker[0], marker[1], r, c) for r, c in cells}
        if len(seen_colors) != TILE_CELLS:
            return None
    if min(int(t["steps"]) for t in quota_traces) != quota:
        return None

    return g


def generate_input() -> np.ndarray:
    global _next_marker_count

    preferred = int(_next_marker_count)
    _next_marker_count = 1 if preferred == 2 else 2

    attempt_budget = {1: 120, 2: 900}
    schedules = (
        preferred,
        1 if preferred == 2 else 2,
    )
    for target_markers in schedules:
        budget = attempt_budget[int(target_markers)]
        for _ in range(budget):
            try:
                g = _gen_attempt(target_markers=target_markers)
            except Exception:
                g = None
            if g is not None:
                return g
    raise RuntimeError("generate_input: failed after exhausting branch budgets")
