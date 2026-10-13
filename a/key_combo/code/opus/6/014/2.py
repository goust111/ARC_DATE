import random
from collections import defaultdict, deque

import numpy as np


BG = 0
LEGEND_COLS = 2
PALETTE = list(range(1, 10))
GEN_ATTEMPTS = 800


def _legend_pairs(grid: np.ndarray):
    pairs = []
    for r in range(grid.shape[0]):
        left = int(grid[r, 0])
        right = int(grid[r, 1])
        if left == BG or right == BG:
            break
        pairs.append((left, right))
    if not pairs:
        raise AssertionError("expected a non-empty two-column legend")
    sources = [src for src, _ in pairs]
    if len(set(sources)) != len(sources):
        raise AssertionError("legend source colors must be unique")
    return pairs


def _legend_rows(grid: np.ndarray) -> int:
    return len(_legend_pairs(grid))


def _cells_in_legend(r: int, c: int, n_rows: int) -> bool:
    return 0 <= r < n_rows and 0 <= c < LEGEND_COLS


def _find_dominoes(grid: np.ndarray, legend_rows: int):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    dominoes = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or _cells_in_legend(r, c, legend_rows):
                continue
            color = int(grid[r, c])
            if color == BG:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or _cells_in_legend(nr, nc, legend_rows):
                        continue
                    if int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            coords = sorted(coords)
            if len(coords) != 2:
                raise AssertionError("every non-legend object must be a 2-pixel domino")
            (r0, c0), (r1, c1) = coords
            if r0 == r1 and abs(c0 - c1) == 1:
                orient = "h"
                row = r0
                left = min(c0, c1)
                distance = min(left + 1, w - left - 1)
                dominoes.append(
                    {
                        "src": color,
                        "orient": orient,
                        "row": row,
                        "left": left,
                        "distance": distance,
                        "cells": {(row, left), (row, left + 1)},
                    }
                )
            elif c0 == c1 and abs(r0 - r1) == 1:
                orient = "v"
                col = c0
                top = min(r0, r1)
                distance = min(top + 1, h - top - 1)
                dominoes.append(
                    {
                        "src": color,
                        "orient": orient,
                        "col": col,
                        "top": top,
                        "distance": distance,
                        "cells": {(top, col), (top + 1, col)},
                    }
                )
            else:
                raise AssertionError("each object must be a horizontal or vertical domino")
    if not dominoes:
        raise AssertionError("expected at least one domino marker")
    return dominoes


def _periodic_cols(left: int, period: int, width: int):
    return {c for c in range(width) if (c - left) % period == 0 or (c - (left + 1)) % period == 0}


def _periodic_rows(top: int, period: int, height: int):
    return {r for r in range(height) if (r - top) % period == 0 or (r - (top + 1)) % period == 0}


def _claim_cells(marker, shape):
    h, w = shape
    if marker["orient"] == "h":
        cols = _periodic_cols(marker["left"], marker["distance"], w)
        return {(marker["row"], c) for c in cols}
    rows = _periodic_rows(marker["top"], marker["distance"], h)
    return {(r, marker["col"]) for r in rows}


def _render(shape, markers, mapping):
    out = np.full(shape, BG, dtype=int)
    best = {}
    for marker in markers:
        key = (int(marker["distance"]), 0 if marker["orient"] == "h" else 1)
        color = int(mapping[marker["src"]])
        for cell in _claim_cells(marker, shape):
            if cell not in best or key < best[cell][0]:
                best[cell] = (key, color)
    for (r, c), (_, color) in best.items():
        out[r, c] = color
    return out


def _render_fixed_period(shape, markers, mapping, period: int):
    adjusted = []
    for marker in markers:
        clone = dict(marker)
        clone["distance"] = period
        adjusted.append(clone)
    return _render(shape, adjusted, mapping)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read a two-column legend, then recolor each domino marker into a periodic row or column extension.

    Input structure:
    - The grid is a black background with a two-column legend in the upper-left corner.
    - Each consecutive non-black legend row maps one source color in column 0 to one output color in column 1.
    - Outside the legend, every foreground object is a same-color two-pixel domino whose color matches a legend source.

    Rule:
    - Read the legend rows, clear those legend cells to black, and use the pairs as a source->output color map.
    - A horizontal domino extends across its row; a vertical domino extends across its column.
    - The extension period is the domino's distance to the nearer grid edge along that same axis.
    - Starting from the domino's two pixels, color every period-th cell along the whole row or column with the mapped output color.
    - If a horizontal and vertical extension overlap, the domino with the smaller edge-distance wins; equal distances are broken in favor of the horizontal extension.

    Key insight:
    - The domino orientation selects the extension axis, the legend selects the extension color, and the domino's position relative to the nearest edge encodes the period and overlap priority.
    """
    g = np.array(grid, dtype=int, copy=False)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")
    pairs = _legend_pairs(g)
    mapping = dict(pairs)
    legend_rows = len(pairs)
    markers = _find_dominoes(g, legend_rows)
    if not all(marker["src"] in mapping for marker in markers):
        raise AssertionError("every marker color must appear in the legend")
    return _render(g.shape, markers, mapping)


def _candidate_lefts(width: int, distance: int):
    opts = []
    for left in range(width - 1):
        if min(left + 1, width - left - 1) == distance:
            opts.append(left)
    return opts


def _candidate_tops(height: int, distance: int):
    opts = []
    for top in range(height - 1):
        if min(top + 1, height - top - 1) == distance:
            opts.append(top)
    return opts


def _touches_any(cells, others):
    for marker in others:
        for r0, c0 in cells:
            for r1, c1 in marker["cells"]:
                if max(abs(r0 - r1), abs(c0 - c1)) <= 1:
                    return True
    return False


def _cell_claimers(markers, shape):
    claims = defaultdict(list)
    for idx, marker in enumerate(markers):
        for cell in _claim_cells(marker, shape):
            claims[cell].append(idx)
    return claims


def _dominant_overlap_types(markers, shape):
    claims = _cell_claimers(markers, shape)
    overlaps = []
    for cell, idxs in claims.items():
        if len(idxs) < 2:
            continue
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                a = markers[idxs[i]]
                b = markers[idxs[j]]
                if a["orient"] == b["orient"]:
                    continue
                tie = a["distance"] == b["distance"]
                winner = a if (a["distance"], 0 if a["orient"] == "h" else 1) < (b["distance"], 0 if b["orient"] == "h" else 1) else b
                overlaps.append((cell, tie, winner["orient"]))
    return overlaps


def _make_marker(orient: str, row_or_top: int, col_or_left: int, src: int, distance: int):
    if orient == "h":
        row = row_or_top
        left = col_or_left
        return {
            "src": src,
            "orient": "h",
            "row": row,
            "left": left,
            "distance": distance,
            "cells": {(row, left), (row, left + 1)},
        }
    top = row_or_top
    col = col_or_left
    return {
        "src": src,
        "orient": "v",
        "col": col,
        "top": top,
        "distance": distance,
        "cells": {(top, col), (top + 1, col)},
    }


def _place_base_pair(height, width, legend_rows, src_h, src_v, rng):
    max_h = min(width // 2, 6)
    max_v = min(height // 2, 6)
    d_choices_h = [d for d in range(3, max_h + 1) if _candidate_lefts(width, d)]
    d_choices_v = [d for d in range(3, max_v + 1) if _candidate_tops(height, d)]
    if not d_choices_h or not d_choices_v:
        return None
    for _ in range(200):
        dh = rng.choice(d_choices_h)
        dv_candidates = [d for d in d_choices_v if d != dh]
        if not dv_candidates:
            continue
        dv = rng.choice(dv_candidates)
        left = rng.choice([c for c in _candidate_lefts(width, dh) if c >= 2 and c <= width - 4])
        top = rng.choice([r for r in _candidate_tops(height, dv) if r >= legend_rows + 1 and r <= height - 4])
        candidate_cols = sorted(_periodic_cols(left, dh, width) - {left, left + 1})
        candidate_cols = [c for c in candidate_cols if c >= 2]
        candidate_rows = sorted(_periodic_rows(top, dv, height) - {top, top + 1})
        candidate_rows = [r for r in candidate_rows if r >= legend_rows + 1]
        rng.shuffle(candidate_cols)
        rng.shuffle(candidate_rows)
        for col in candidate_cols:
            for row in candidate_rows:
                h_marker = _make_marker("h", row, left, src_h, dh)
                v_marker = _make_marker("v", top, col, src_v, dv)
                if _touches_any(h_marker["cells"], [v_marker]) or _touches_any(v_marker["cells"], [h_marker]):
                    continue
                return [h_marker, v_marker]
    return None


def _place_tie_marker(height, width, legend_rows, base_markers, src, orient, rng):
    h0 = next(marker for marker in base_markers if marker["orient"] == "h")
    v0 = next(marker for marker in base_markers if marker["orient"] == "v")
    used_rows = {marker["row"] for marker in base_markers if marker["orient"] == "h"}
    used_cols = {marker["col"] for marker in base_markers if marker["orient"] == "v"}
    if orient == "h":
        distance = v0["distance"]
        rows = [r for r in sorted(_periodic_rows(v0["top"], v0["distance"], height) - {v0["top"], v0["top"] + 1}) if r >= legend_rows + 1 and r not in used_rows]
        lefts = [c for c in _candidate_lefts(width, distance) if c >= 2 and c <= width - 4]
        rng.shuffle(rows)
        rng.shuffle(lefts)
        for row in rows:
            for left in lefts:
                if v0["col"] not in _periodic_cols(left, distance, width) or v0["col"] in {left, left + 1}:
                    continue
                marker = _make_marker("h", row, left, src, distance)
                if _touches_any(marker["cells"], base_markers):
                    continue
                return marker
    else:
        distance = h0["distance"]
        cols = [c for c in sorted(_periodic_cols(h0["left"], h0["distance"], width) - {h0["left"], h0["left"] + 1}) if c >= 2 and c not in used_cols]
        tops = [r for r in _candidate_tops(height, distance) if r >= legend_rows + 1 and r <= height - 4]
        rng.shuffle(cols)
        rng.shuffle(tops)
        for col in cols:
            for top in tops:
                if h0["row"] not in _periodic_rows(top, distance, height) or h0["row"] in {top, top + 1}:
                    continue
                marker = _make_marker("v", top, col, src, distance)
                if _touches_any(marker["cells"], base_markers):
                    continue
                return marker
    return None


def _place_random_marker(height, width, legend_rows, markers, src, orient, rng):
    used_rows = {marker["row"] for marker in markers if marker["orient"] == "h"}
    used_cols = {marker["col"] for marker in markers if marker["orient"] == "v"}
    row_candidates = [r for r in range(legend_rows + 1, height - 1)]
    col_candidates = [c for c in range(2, width - 1)]
    distances_h = [d for d in range(3, min(width // 2, 6) + 1) if _candidate_lefts(width, d)]
    distances_v = [d for d in range(3, min(height // 2, 6) + 1) if _candidate_tops(height, d)]
    for _ in range(300):
        if orient == "h":
            if not distances_h:
                return None
            row = rng.choice(row_candidates)
            if row in used_rows:
                continue
            distance = rng.choice(distances_h)
            left = rng.choice([c for c in _candidate_lefts(width, distance) if c >= 2 and c <= width - 4])
            marker = _make_marker("h", row, left, src, distance)
        else:
            if not distances_v:
                return None
            col = rng.choice(col_candidates)
            if col in used_cols:
                continue
            distance = rng.choice(distances_v)
            top = rng.choice([r for r in _candidate_tops(height, distance) if r >= legend_rows + 1 and r <= height - 4])
            marker = _make_marker("v", top, col, src, distance)
        if _touches_any(marker["cells"], markers):
            continue
        return marker
    return None


def _build_input(shape, legend_pairs, markers):
    grid = np.zeros(shape, dtype=int)
    for r, (src, dst) in enumerate(legend_pairs):
        grid[r, 0] = src
        grid[r, 1] = dst
    for marker in markers:
        for r, c in marker["cells"]:
            grid[r, c] = marker["src"]
    return grid


def _validate_case(grid, markers, mapping, want_tie):
    legend_rows = _legend_rows(grid)
    parsed = _find_dominoes(grid, legend_rows)
    parsed_map = {marker["src"]: marker for marker in parsed}
    if set(parsed_map) != {marker["src"] for marker in markers}:
        return False
    if len({marker["distance"] for marker in markers}) < 2:
        return False
    if max(marker["distance"] for marker in markers) < 3:
        return False
    claims = _cell_claimers(markers, grid.shape)
    for marker in markers:
        for cell in marker["cells"]:
            if claims[cell] != [markers.index(marker)]:
                return False
    overlaps = _dominant_overlap_types(markers, grid.shape)
    if not overlaps:
        return False
    if not any(not tie for _, tie, _ in overlaps):
        return False
    if not any(not tie and winner == "v" for _, tie, winner in overlaps):
        return False
    if want_tie and not any(tie for _, tie, _ in overlaps):
        return False
    out = _render(grid.shape, markers, mapping)
    if np.array_equal(out, _render_fixed_period(grid.shape, markers, mapping, 2)):
        return False
    if np.count_nonzero(out) <= 10:
        return False
    if len(np.unique(out[out != BG])) < 2:
        return False
    if np.array_equal(out, grid):
        return False
    return True


def _try_generate(rng):
    height = rng.randint(13, 16)
    width = rng.randint(13, 16)
    n_markers = rng.choice([2, 3, 3, 4])
    legend_rows = n_markers
    colors = rng.sample(PALETTE, 2 * legend_rows)
    sources = colors[:legend_rows]
    outputs = colors[legend_rows:]
    legend_pairs = list(zip(sources, outputs))
    mapping = dict(legend_pairs)

    if n_markers == 2:
        orient_plan = ["h", "v"]
    elif n_markers == 3:
        orient_plan = ["h", "v", rng.choice(["h", "v"])]
    else:
        orient_plan = ["h", "h", "v", "v"]

    base = _place_base_pair(height, width, legend_rows, sources[0], sources[1], rng)
    if base is None:
        return None
    markers = base

    want_tie = n_markers >= 3 and rng.random() < 0.65
    used_sources = 2
    remaining_orients = orient_plan[2:]
    if want_tie and remaining_orients:
        tie_orient = "h" if "h" in remaining_orients and sum(o == "h" for o in orient_plan) > 1 else "v"
        tie_marker = _place_tie_marker(height, width, legend_rows, markers, sources[used_sources], tie_orient, rng)
        if tie_marker is None:
            return None
        markers.append(tie_marker)
        remaining_orients.remove(tie_orient)
        used_sources += 1

    for orient in remaining_orients:
        marker = _place_random_marker(height, width, legend_rows, markers, sources[used_sources], orient, rng)
        if marker is None:
            return None
        markers.append(marker)
        used_sources += 1

    grid = _build_input((height, width), legend_pairs, markers)
    if not _validate_case(grid, markers, mapping, want_tie):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _try_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
