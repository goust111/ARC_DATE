# Variant of 53fb4810
# Dimension: D18 — 符号编码机制 (symbol_instructions)
# Change: 无符号编码——所有信息直接由空间关系传达 → 边缘像素编码方向
#
# Design analysis
# a) Concrete behavior and trace:
#    - The original task lets each seed tile announce its ray direction by the side of the '+' that it touches.
#    - This variant keeps the same '+' anchors, the same attached payload tiles, and the same "only one ray is
#      pre-expanded in the input" clue. The change is that direction is no longer read from attachment side.
#    - Every payload now has a small border control strip in a unique control color. A strip on the top/bottom edge
#      encodes an upward/downward ray; a strip on the left/right edge encodes a leftward/rightward ray. The strip span
#      matches the payload component's current cross-axis span, so the solver can match strip <-> payload by visible
#      alignment. Example: a 2x1 payload attached below a plus may still project left if a 2-cell vertical strip sits
#      on the left border at the same rows. Original logic would extend downward; the variant must decode "left".
# b) Complexity calibration:
#    - Direction has only four visible categories (top, bottom, left, right), exactly like the original ray axis set.
#    - The matching cue is low-entropy and repeated: each grid contains 2-3 pluses, each with its own border strip.
#    - The payload still stays adjacent to its plus, so object ownership is easy to parse; the extra symbol only
#      replaces the direction source, rather than adding a second independent transformation rule.
# c) Old assumptions that no longer hold:
#    - In the original `main()`, the payload-touching side uniquely defined direction. That is false here.
#    - The original arm-tip / seed-span assertion depended on direction being tied to the touching arm. That is also
#      false here, so the solver must stop using plus nozzle geometry to infer the ray axis.
#    - The pre-expanded clue is still one connected component, so the solver must continue to tolerate components whose
#      bbox is already long; only the cross-axis span is relevant for decoding the border strip.
# d) Placement legality:
#    - Generator keeps pluses away from the border so the control color is the only color whose every occurrence lies
#      on the border.
#    - Each payload row span and column span is unique inside one grid. This guarantees that exactly one border strip
#      matches a given payload component in the solver.
#    - Rays, payloads, control strips, and pluses are all non-overlapping. Rays are also kept away from previously
#      placed plus neighborhoods so no payload accidentally appears attached to the wrong plus.
# e) Visual observability:
#    - The border strip visibly contradicts spatial attachment in at least one plus per generated grid, so the new
#      symbol mechanism is not a cosmetic duplicate of the old rule.
#    - Multiple directions are used within one grid whenever possible, giving the human solver repeated evidence that
#      border side, not attachment side, drives the output.
# f) Visual cleanliness:
#    - Only one extra symbolic element is added: a thin border strip in one dedicated color.
#    - Control strips stay on the edge and are protected from overwrite, so they do not clutter the central scene.
# g) Color inference:
#    - The control color is structural rather than semantic-by-id: it is the unique non-plus color whose occurrences
#      are border-only. No sorting by hidden color id is used to decide direction.
# 9) New solver assumptions and generator guarantees:
#    - Assumption: there is exactly one border-only control color. Generator guarantees that by reserving one color for
#      control strips and keeping all payload colors in interior cells as well.
#    - Assumption: each plus has exactly one adjacent non-control payload component. Generator builds one payload per
#      plus and prevents collisions/adjacency that would merge components.
#    - Assumption: each payload bbox matches exactly one border strip by row-span or column-span. Generator enforces
#      unique row spans and unique column spans across payloads.

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask


def detect_dominant_big_component_color(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    min_h: int = 3,
    min_w: int = 3,
    connectivity: int = 4,
) -> int | None:
    """
    Local helper (only used by this solver): choose a color that has the most "big" components.

    Big component = connected component whose bbox is at least (min_h x min_w).
    Tie-break: total area of big components (larger wins).
    """
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    best_key: tuple[int, int] | None = None
    best_col: int | None = None

    for col in sorted(set(map(int, grid.flatten())) - {int(bg)}):
        lab, n = label_mask(grid == int(col), connectivity=int(connectivity))
        big_areas: list[int] = []
        for k in range(1, int(n) + 1):
            coords = np.argwhere(lab == int(k))
            if coords.size == 0:
                continue
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            if h >= int(min_h) and w >= int(min_w):
                big_areas.append(int(coords.shape[0]))
        if not big_areas:
            continue
        key = (int(len(big_areas)), int(sum(big_areas)))
        if best_key is None or key > best_key:
            best_key = key
            best_col = int(col)

    return int(best_col) if best_col is not None else None


def detect_border_only_control_color(
    grid: np.ndarray,
    *,
    bg: int,
    exclude: set[int] | None = None,
) -> int | None:
    """Pick the unique non-bg color whose every occurrence lies on the border."""
    exclude = set() if exclude is None else set(map(int, exclude))
    H, W = grid.shape
    border = np.zeros((H, W), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    candidates: list[int] = []
    for col in sorted(set(map(int, grid.flatten())) - {int(bg)} - exclude):
        pts = np.argwhere(grid == int(col))
        if pts.size == 0:
            continue
        if all(bool(border[int(r), int(c)]) for r, c in pts):
            candidates.append(int(col))

    return int(candidates[0]) if len(candidates) == 1 else None


def extend_repeating_tile(
    out: np.ndarray,
    tile: np.ndarray,
    r0: int,
    c0: int,
    direction: tuple[int, int],
    *,
    bg: int,
    protect: set[int] | None = None,
) -> None:
    """Local helper (only used by this solver): extend a tile pattern from (r0,c0) until boundary."""
    protect = set() if protect is None else set(map(int, protect))
    H, W = out.shape
    h, w = tile.shape
    dr, dc = map(int, direction)
    dr = 0 if dr == 0 else int(dr / abs(dr))
    dc = 0 if dc == 0 else int(dc / abs(dc))
    bg = int(bg)
    r0 = int(r0)
    c0 = int(c0)

    def put(rr: int, cc: int, v: int) -> None:
        if v == bg:
            return
        if int(out[rr, cc]) in protect:
            return
        out[rr, cc] = int(v)

    if (dr, dc) == (0, 1):
        start_c = c0 + w
        for k, tc in enumerate(range(start_c, W)):
            col_idx = k % w
            for i in range(h):
                put(r0 + i, tc, int(tile[i, col_idx]))
    elif (dr, dc) == (0, -1):
        for k, tc in enumerate(range(c0 - 1, -1, -1), start=1):
            col_idx = (-k) % w
            for i in range(h):
                put(r0 + i, tc, int(tile[i, col_idx]))
    elif (dr, dc) == (1, 0):
        start_r = r0 + h
        for k, tr in enumerate(range(start_r, H)):
            row_idx = k % h
            for j in range(w):
                put(tr, c0 + j, int(tile[row_idx, j]))
    elif (dr, dc) == (-1, 0):
        for k, tr in enumerate(range(r0 - 1, -1, -1), start=1):
            row_idx = (-k) % h
            for j in range(w):
                put(tr, c0 + j, int(tile[row_idx, j]))
    else:
        raise ValueError("extend_repeating_tile: direction must be one of (±1,0),(0,±1)")


def _collect_control_specs(ctrl_mask: np.ndarray) -> list[dict]:
    """Describe each border control strip as a direction + aligned span."""
    H, W = ctrl_mask.shape
    ctrl_lab, cn = label_mask(ctrl_mask, connectivity=4)
    specs: list[dict] = []
    for k in range(1, int(cn) + 1):
        pts = np.argwhere(ctrl_lab == int(k))
        if pts.size == 0:
            continue
        r0, c0, r1, c1 = bbox_coords(pts)
        if int(r0) == 0 and int(r1) == 0:
            specs.append({"direction": (-1, 0), "span": tuple(range(int(c0), int(c1) + 1))})
        elif int(r0) == int(H - 1) and int(r1) == int(H - 1):
            specs.append({"direction": (1, 0), "span": tuple(range(int(c0), int(c1) + 1))})
        elif int(c0) == 0 and int(c1) == 0:
            specs.append({"direction": (0, -1), "span": tuple(range(int(r0), int(r1) + 1))})
        elif int(c0) == int(W - 1) and int(c1) == int(W - 1):
            specs.append({"direction": (0, 1), "span": tuple(range(int(r0), int(r1) + 1))})
    return specs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each payload tile from its current position using a direction encoded by edge controls.

    Input structure:
    - bg is the dominant background color.
    - One dominant non-bg color forms several '+'-like anchors.
    - Each '+' has one adjacent colored payload component, and one small strip of a dedicated control color sits on a border.

    Rule:
    - Match each payload to the unique border strip whose span aligns with the payload bbox: top/bottom strips match by columns, left/right strips by rows.
    - The border side of that strip gives the ray direction; repeat the payload periodically from its current bbox in that direction until the edge.
    - '+' pixels and control-strip pixels are protected and remain visible.

    Key insight:
    - The touching side of the payload is no longer the instruction. Direction is a symbolic cue carried by a border strip, while the payload component only supplies the repeated pattern.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()

    bg = int(get_bg(g))
    plus_col = detect_dominant_big_component_color(g, bg=bg, min_h=3, min_w=3, connectivity=4)
    if plus_col is None:
        return out

    ctrl_color = detect_border_only_control_color(g, bg=bg, exclude={int(plus_col)})
    if ctrl_color is None:
        return out

    plus_lab, pn = label_mask(g == int(plus_col), connectivity=4)
    ctrl_specs = _collect_control_specs(g == int(ctrl_color))

    non_mask = (g != int(bg)) & (g != int(plus_col)) & (g != int(ctrl_color))
    mark_lab, _mn = label_mask(non_mask, connectivity=4)
    H, W = g.shape

    for pk in range(1, int(pn) + 1):
        plus_mask = plus_lab == int(pk)
        plus_coords = np.argwhere(plus_mask)
        if plus_coords.size == 0:
            continue

        adj_ids: set[int] = set()
        for r, c in plus_coords:
            r = int(r)
            c = int(c)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < H and 0 <= cc < W and int(mark_lab[rr, cc]) > 0:
                    adj_ids.add(int(mark_lab[rr, cc]))

        if len(adj_ids) != 1:
            continue

        cid = next(iter(adj_ids))
        pts = np.argwhere(mark_lab == int(cid))
        if pts.size == 0:
            continue

        r0, c0, r1, c1 = bbox_coords(pts)
        tile = crop_bbox(g, (int(r0), int(c0), int(r1), int(c1)))
        span_rows = tuple(range(int(r0), int(r1) + 1))
        span_cols = tuple(range(int(c0), int(c1) + 1))

        dirs: list[tuple[int, int]] = []
        for spec in ctrl_specs:
            dr, dc = spec["direction"]
            if dc == 0 and tuple(spec["span"]) == span_cols:
                dirs.append((int(dr), int(dc)))
            elif dr == 0 and tuple(spec["span"]) == span_rows:
                dirs.append((int(dr), int(dc)))

        dirs = sorted(set(dirs))
        if len(dirs) != 1:
            continue

        dr, dc = dirs[0]
        extend_repeating_tile(
            out,
            tile,
            int(r0),
            int(c0),
            (int(dr), int(dc)),
            bg=int(bg),
            protect={int(plus_col), int(ctrl_color)},
        )

    return out


# --------------- generator ---------------

_H_RANGE = (13, 29)
_W_RANGE = (13, 23)
_N_PLUS_RANGE = (2, 4)
_SEED_ALONG_CHOICES = [1, 2]
_SEED_MAX_COLORS = 4
_MIN_RAY_UNITS = 2
_PRE_EXT_IDEAL_UNITS = 8
_PLACEMENT_TRIES = 120
_GEN_RETRIES = 400

_PLUS_SHAPES = [
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[0, 1, 1, 0], [1, 1, 1, 1], [0, 1, 1, 0]], dtype=int),
    np.array(
        [
            [0, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 1, 1, 0],
        ],
        dtype=int,
    ),
]

_DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _nozzle_span(plus_mask: np.ndarray, direction: tuple[int, int]) -> tuple[int, int]:
    """Return the inclusive cross-axis span for one side of the plus shape."""
    dr, dc = direction
    if dc == 0:
        row = plus_mask.shape[0] - 1 if dr == 1 else 0
        cols = np.where(plus_mask[row])[0]
        return int(cols[0]), int(cols[-1])
    col = plus_mask.shape[1] - 1 if dc == 1 else 0
    rows = np.where(plus_mask[:, col])[0]
    return int(rows[0]), int(rows[-1])


def _build_seed_tile(
    plus_mask: np.ndarray,
    attach_direction: tuple[int, int],
    palette: list[int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Build a small full-rectangle payload whose cross-axis matches the chosen attachment side."""
    s, e = _nozzle_span(plus_mask, attach_direction)
    cross = e - s + 1
    along = int(rng.choice(_SEED_ALONG_CHOICES))
    if along == 1 and cross == 1:
        along = 2

    dr, dc = attach_direction
    th, tw = (along, cross) if dc == 0 else (cross, along)

    n_cells = th * tw
    max_k = min(_SEED_MAX_COLORS, n_cells, len(palette))
    n_colors = int(rng.integers(1, max_k + 1))
    chosen = list(rng.choice(palette, size=n_colors, replace=False))
    vals = list(chosen) + list(rng.choice(chosen, size=n_cells - n_colors, replace=True))
    rng.shuffle(vals)
    return np.array(vals, dtype=int).reshape(th, tw)


def _anchor_seed_tile(
    plus_bbox: tuple[int, int, int, int],
    plus_mask: np.ndarray,
    attach_direction: tuple[int, int],
    tile: np.ndarray,
) -> tuple[int, int]:
    """Place the seed flush against the chosen side of the plus."""
    pr0, pc0, pr1, pc1 = plus_bbox
    dr, dc = attach_direction
    th, tw = tile.shape
    s, _ = _nozzle_span(plus_mask, attach_direction)

    if dc == 0:
        c0 = pc0 + s
        r0 = (pr1 + 1) if dr == 1 else (pr0 - th)
    else:
        r0 = pr0 + s
        c0 = (pc1 + 1) if dc == 1 else (pc0 - tw)
    return int(r0), int(c0)


def _compute_ray_units_from_seed(
    direction: tuple[int, int],
    tile: np.ndarray,
    seed_r0: int,
    seed_c0: int,
    H: int,
    W: int,
) -> int:
    """Count the number of full tile placements from the seed position to the border."""
    dr, dc = direction
    th, tw = tile.shape
    if dr == 1:
        return int((H - seed_r0) // th)
    if dr == -1:
        return int((seed_r0 + th) // th)
    if dc == 1:
        return int((W - seed_c0) // tw)
    return int((seed_c0 + tw) // tw)


def _project_tile_coords(
    H: int,
    W: int,
    tile: np.ndarray,
    seed_r0: int,
    seed_c0: int,
    direction: tuple[int, int],
) -> list[tuple[int, int, int]]:
    """Return the full seed+ray cells produced by periodic repetition from the seed bbox."""
    th, tw = tile.shape
    dr, dc = direction
    cells: list[tuple[int, int, int]] = []

    if dc == 0:
        r = int(seed_r0)
        step = th if dr == 1 else -th
        while (r < H) if dr == 1 else (r + th > 0):
            for i in range(th):
                rr = r + i
                if 0 <= rr < H:
                    for j in range(tw):
                        cells.append((int(rr), int(seed_c0 + j), int(tile[i, j])))
            r += step
    else:
        c = int(seed_c0)
        step = tw if dc == 1 else -tw
        while (c < W) if dc == 1 else (c + tw > 0):
            for i in range(th):
                for j in range(tw):
                    cc = c + j
                    if 0 <= cc < W:
                        cells.append((int(seed_r0 + i), int(cc), int(tile[i, j])))
            c += step

    return cells


def _build_control_cells(
    H: int,
    W: int,
    tile: np.ndarray,
    seed_r0: int,
    seed_c0: int,
    direction: tuple[int, int],
) -> list[tuple[int, int]]:
    """Place a border control strip whose span matches the payload cross-axis span."""
    th, tw = tile.shape
    dr, dc = direction
    if dc == 0:
        row = 0 if dr == -1 else int(H - 1)
        return [(int(row), int(seed_c0 + j)) for j in range(tw)]
    col = 0 if dc == -1 else int(W - 1)
    return [(int(seed_r0 + i), int(col)) for i in range(th)]


def _allowed_ray_directions(attach_direction: tuple[int, int]) -> list[tuple[int, int]]:
    """Keep the original direction or turn sideways, but never fire back through the plus."""
    opposite = (-int(attach_direction[0]), -int(attach_direction[1]))
    return [d for d in _DIRECTIONS if d != opposite]


def _place_plus_and_ray(
    rng: np.random.Generator,
    plus_shape: np.ndarray,
    palette: list[int],
    H: int,
    W: int,
    occ: np.ndarray,
    plus_occ: np.ndarray,
    out: np.ndarray,
    plus_color: int,
    ctrl_color: int,
    used_row_spans: set[tuple[int, ...]],
    used_col_spans: set[tuple[int, ...]],
) -> dict | None:
    """Try to place one plus, one payload, its control strip, and its full output ray."""
    ph, pw = plus_shape.shape
    min_r = 2
    max_r = H - ph - 2
    min_c = 2
    max_c = W - pw - 2
    if max_r < min_r or max_c < min_c:
        return None

    for _ in range(_PLACEMENT_TRIES):
        r0 = int(rng.integers(min_r, max_r + 1))
        c0 = int(rng.integers(min_c, max_c + 1))
        plus_cells = [
            (r0 + dr, c0 + dc)
            for dr in range(ph)
            for dc in range(pw)
            if plus_shape[dr, dc]
        ]

        if any(int(occ[r, c]) != 0 for r, c in plus_cells):
            continue
        if any(int(plus_occ[r, c]) == 1 for r, c in plus_cells):
            continue

        attach_dirs = list(_DIRECTIONS)
        rng.shuffle(attach_dirs)
        for attach_direction in attach_dirs:
            tile = _build_seed_tile(plus_shape, attach_direction, palette, rng)
            seed_r0, seed_c0 = _anchor_seed_tile(
                (r0, c0, r0 + ph - 1, c0 + pw - 1),
                plus_shape,
                attach_direction,
                tile,
            )
            th, tw = tile.shape
            if seed_r0 < 1 or seed_c0 < 1 or seed_r0 + th > H - 1 or seed_c0 + tw > W - 1:
                continue

            seed_rows = tuple(range(int(seed_r0), int(seed_r0 + th)))
            seed_cols = tuple(range(int(seed_c0), int(seed_c0 + tw)))
            if seed_rows in used_row_spans or seed_cols in used_col_spans:
                continue

            ray_dirs = [d for d in _allowed_ray_directions(attach_direction) if d != attach_direction]
            rng.shuffle(ray_dirs)
            ray_dirs.append(attach_direction)

            for ray_direction in ray_dirs:
                units = _compute_ray_units_from_seed(
                    ray_direction, tile, int(seed_r0), int(seed_c0), H, W
                )
                if units < _MIN_RAY_UNITS:
                    continue

                cells = _project_tile_coords(
                    H, W, tile, int(seed_r0), int(seed_c0), ray_direction
                )
                if any(int(occ[r, c]) != 0 for r, c, _ in cells):
                    continue
                if any(int(plus_occ[r, c]) == 1 for r, c, _ in cells):
                    continue

                control_cells = _build_control_cells(
                    H, W, tile, int(seed_r0), int(seed_c0), ray_direction
                )
                if any(int(occ[r, c]) != 0 for r, c in control_cells):
                    continue

                for r, c in plus_cells:
                    occ[r, c] = 1
                    plus_occ[r, c] = 1
                    out[r, c] = int(plus_color)
                    for dr2, dc2 in _DIRECTIONS:
                        rr = int(r + dr2)
                        cc = int(c + dc2)
                        if 0 <= rr < H and 0 <= cc < W:
                            plus_occ[rr, cc] = 1

                for r, c, v in cells:
                    occ[r, c] = 2
                    out[r, c] = int(v)

                control_set = {(int(r), int(c)) for r, c in control_cells}
                for r, c in control_set:
                    occ[r, c] = 3
                    out[r, c] = int(ctrl_color)

                used_row_spans.add(seed_rows)
                used_col_spans.add(seed_cols)
                return {
                    "tile": tile,
                    "cells": cells,
                    "control_set": control_set,
                    "seed_r0": int(seed_r0),
                    "seed_c0": int(seed_c0),
                    "units": int(units),
                    "attach_direction": tuple(map(int, attach_direction)),
                    "ray_direction": tuple(map(int, ray_direction)),
                }

    return None


def _build_input_from_output(
    out: np.ndarray,
    pluses: list[dict],
    bg: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Keep exactly one full ray as the clue; collapse all other rays back to their seed tiles."""
    pre_idx = max(range(len(pluses)), key=lambda i: pluses[i]["units"])
    if pluses[pre_idx]["units"] < _PRE_EXT_IDEAL_UNITS:
        candidates = [i for i, p in enumerate(pluses) if p["units"] >= _PRE_EXT_IDEAL_UNITS]
        if candidates:
            pre_idx = int(rng.choice(candidates))

    inp = out.copy()
    for i, p in enumerate(pluses):
        if i == pre_idx:
            continue
        th, tw = p["tile"].shape
        sr0, sc0 = int(p["seed_r0"]), int(p["seed_c0"])
        seed_set = {(sr0 + ii, sc0 + jj) for ii in range(th) for jj in range(tw)}
        for r, c, _ in p["cells"]:
            rc = (int(r), int(c))
            if rc in p["control_set"]:
                continue
            if rc not in seed_set:
                inp[rc] = int(bg)
    return inp


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    H = int(rng.integers(*_H_RANGE))
    W = int(rng.integers(*_W_RANGE))

    bg = int(rng.integers(0, 10))
    others = [c for c in range(10) if c != bg]
    plus_color = int(rng.choice(others))
    ctrl_color = int(rng.choice([c for c in others if c != plus_color]))
    palette = [c for c in others if c not in {plus_color, ctrl_color}]

    plus_shape = _PLUS_SHAPES[int(rng.integers(0, len(_PLUS_SHAPES)))]
    N = int(rng.integers(*_N_PLUS_RANGE))

    out = np.full((H, W), bg, dtype=int)
    occ = np.zeros((H, W), dtype=int)
    plus_occ = np.zeros((H, W), dtype=int)
    used_row_spans: set[tuple[int, ...]] = set()
    used_col_spans: set[tuple[int, ...]] = set()

    pluses: list[dict] = []
    for _ in range(N):
        info = _place_plus_and_ray(
            rng,
            plus_shape,
            palette,
            H,
            W,
            occ,
            plus_occ,
            out,
            plus_color,
            ctrl_color,
            used_row_spans,
            used_col_spans,
        )
        if info is None:
            return None
        pluses.append(info)

    mismatches = sum(int(p["attach_direction"] != p["ray_direction"]) for p in pluses)
    distinct_dirs = {tuple(map(int, p["ray_direction"])) for p in pluses}
    if mismatches < 1 or len(distinct_dirs) < 2:
        return None

    inp = _build_input_from_output(out, pluses, bg, rng)

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_GEN_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {_GEN_RETRIES} attempts")
