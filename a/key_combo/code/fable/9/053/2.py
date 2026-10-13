from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox_coords, components_mask, get_bg


GREY = int(Color.GREY)
BACKGROUND = int(Color.BLACK)
MOTIF_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GRID_H_RANGE = (16, 27)
GRID_W_RANGE = (16, 27)
MOTIF_H_RANGE = (2, 5)
MOTIF_W_RANGE = (2, 5)
MOTIF_COUNT_RANGE = (2, 5)
PAIR_COUNT_RANGE = (2, 5)
MAX_ATTEMPTS = 500


def _major_color(block: np.ndarray) -> int:
    vals, counts = np.unique(block, return_counts=True)
    idx = np.argsort(-counts)
    assert len(idx) >= 1
    if len(idx) > 1:
        assert int(counts[idx[0]]) > int(counts[idx[1]])
    return int(vals[idx[0]])


def _component_boxes(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    bg = int(get_bg(grid))
    comps = components_mask(grid != bg, connectivity=4)
    motifs: list[dict] = []
    singletons: list[dict] = []
    for coords in comps:
        r0, c0, r1, c1 = bbox_coords(coords)
        area = int(coords.shape[0])
        if area == 1:
            r, c = map(int, coords[0])
            singletons.append({"coord": (r, c), "color": int(grid[r, c])})
            continue
        bbox_area = (r1 - r0 + 1) * (c1 - c0 + 1)
        assert area == bbox_area, "every non-singleton component must be a filled rectangle"
        block = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(block != bg)
        motifs.append(
            {
                "bbox": (r0, c0, r1, c1),
                "block": block.copy(),
                "main_color": _major_color(block),
            }
        )
    return motifs, singletons


def _pair_markers(singletons: list[dict]) -> list[dict]:
    greys = [s for s in singletons if int(s["color"]) == GREY]
    colored = [s for s in singletons if int(s["color"]) != GREY]
    assert greys and colored
    pairs: list[dict] = []
    used_greys: set[tuple[int, int]] = set()
    for marker in colored:
        r, c = marker["coord"]
        aligned = []
        for grey in greys:
            gr, gc = grey["coord"]
            if gr == r or gc == c:
                aligned.append(grey)
        assert len(aligned) == 1, "each colored marker must match exactly one grey endpoint"
        grey = aligned[0]
        assert grey["coord"] not in used_greys, "grey endpoints must not be shared"
        used_greys.add(grey["coord"])
        gr, gc = grey["coord"]
        if gr == r:
            direction = "right" if gc > c else "left"
            assert gc != c
        else:
            direction = "down" if gr > r else "up"
            assert gr != r
        pairs.append(
            {
                "start": (r, c),
                "end": (gr, gc),
                "color": int(marker["color"]),
                "direction": direction,
            }
        )
    assert len(used_greys) == len(greys)
    return pairs


def _paint_pair(out: np.ndarray, motif: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> None:
    sr, sc = map(int, start)
    er, ec = map(int, end)
    mh, mw = motif.shape
    if sr == er:
        c_lo, c_hi = sorted((sc, ec))
        for r in range(sr, sr + mh):
            row_rel = r - sr
            for c in range(c_lo, c_hi + 1):
                axis = c - sc if ec > sc else sc - c
                col_rel = axis % mw if ec > sc else (mw - 1 - (axis % mw))
                out[r, c] = int(motif[row_rel, col_rel])
        return

    assert sc == ec
    r_lo, r_hi = sorted((sr, er))
    for r in range(r_lo, r_hi + 1):
        axis = r - sr if er > sr else sr - r
        row_rel = axis % mh if er > sr else (mh - 1 - (axis % mh))
        for c in range(sc, sc + mw):
            col_rel = c - sc
            out[r, c] = int(motif[row_rel, col_rel])


def main(grid: np.ndarray) -> np.ndarray:
    """Tile each color-matched motif from its colored endpoint toward the aligned grey endpoint, clipping at the grey end.

    Input structure:
    - A black background contains several separated rectangular motifs; each motif uses one dominant main color plus a few accent cells.
    - Elsewhere there are isolated singleton pixels: colored start markers and grey end markers, with each pair sharing one row or one column.
    - Every colored singleton's color matches the dominant color of exactly one motif.

    Rule:
    - Match every colored singleton to the unique grey singleton in the same row or column.
    - Select the motif whose dominant color equals the colored singleton. Align the motif corner nearest the start marker onto that marker.
    - Repeat the motif edge-to-edge along the segment from the colored marker toward the grey marker. Horizontal rulers use the segment as the strip's top edge; vertical rulers use it as the strip's left edge.
    - Paint over both endpoints themselves and clip away every part that would extend past the grey endpoint. Keep the original motif samples unchanged.

    Key insight:
    - The visible ruler is missing; only its two endpoints remain. The colored endpoint chooses which motif and which anchored corner to use, while the grey endpoint fixes the travel direction and exact stopping position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    motifs, singletons = _component_boxes(g)
    motif_by_color: dict[int, np.ndarray] = {}
    for motif in motifs:
        main_color = int(motif["main_color"])
        assert main_color not in motif_by_color, "main colors must identify motifs uniquely"
        motif_by_color[main_color] = motif["block"]
    pairs = _pair_markers(singletons)

    out = g.copy()
    for pair in pairs:
        color = int(pair["color"])
        assert color in motif_by_color, "marker color must match a motif main color"
        _paint_pair(out, motif_by_color[color], pair["start"], pair["end"])
    return out


def _randint_pair(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(lo, hi))


def _build_motif(rng: np.random.Generator, h: int, w: int, main_color: int) -> np.ndarray:
    motif = np.full((h, w), int(main_color), dtype=int)
    accent_pool = [c for c in MOTIF_COLORS if c != main_color]
    accent_count = int(rng.integers(1, min(3, h * w - 1) + 1))
    positions = [(r, c) for r in range(h) for c in range(w)]
    corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    inner_positions = [p for p in positions if p not in corners]
    rng.shuffle(inner_positions)
    chosen = inner_positions[:accent_count]
    accents = rng.choice(accent_pool, size=accent_count, replace=False if accent_count <= len(accent_pool) else True)
    for (r, c), val in zip(chosen, accents):
        motif[r, c] = int(val)
    assert _major_color(motif) == main_color
    return motif


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _rect_touch_or_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _strip_bbox(start: tuple[int, int], end: tuple[int, int], motif_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    sr, sc = start
    er, ec = end
    mh, mw = motif_shape
    if sr == er:
        c0, c1 = sorted((sc, ec))
        return sr, c0, sr + mh - 1, c1
    r0, r1 = sorted((sr, er))
    return r0, sc, r1, sc + mw - 1


def _paint_pair_variant(
    shape: tuple[int, int],
    motif: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    force_forward_anchor: bool = False,
) -> np.ndarray:
    canvas = np.full(shape, BACKGROUND, dtype=int)
    sr, sc = start
    er, ec = end
    mh, mw = motif.shape
    if sr == er:
        c_lo, c_hi = sorted((sc, ec))
        for r in range(sr, sr + mh):
            row_rel = r - sr
            for c in range(c_lo, c_hi + 1):
                axis = c - sc if ec > sc else sc - c
                if ec > sc or force_forward_anchor:
                    col_rel = axis % mw
                else:
                    col_rel = mw - 1 - (axis % mw)
                canvas[r, c] = int(motif[row_rel, col_rel])
        return canvas
    r_lo, r_hi = sorted((sr, er))
    for r in range(r_lo, r_hi + 1):
        axis = r - sr if er > sr else sr - r
        row_rel = axis % mh if (er > sr or force_forward_anchor) else (mh - 1 - (axis % mh))
        for c in range(sc, sc + mw):
            canvas[r, c] = int(motif[row_rel, c - sc])
    return canvas


def _sample_pair_specs(
    rng: np.random.Generator,
    motifs: list[dict],
    grid_shape: tuple[int, int],
) -> list[dict] | None:
    h, w = grid_shape
    pair_count = int(rng.integers(PAIR_COUNT_RANGE[0], min(PAIR_COUNT_RANGE[1], len(motifs)) + 1))
    motif_indices = list(rng.choice(len(motifs), size=pair_count, replace=False))
    rng.shuffle(motif_indices)
    need_h = pair_count >= 2 and rng.random() < 0.75
    need_v = pair_count >= 2 and rng.random() < 0.75
    orientations = []
    if need_h:
        orientations.append("h")
    if need_v:
        orientations.append("v")
    while len(orientations) < pair_count:
        orientations.append("h" if rng.random() < 0.5 else "v")
    rng.shuffle(orientations)

    occupied: list[tuple[int, int, int, int]] = [m["bbox"] for m in motifs]
    rows_used: set[int] = set()
    cols_used: set[int] = set()
    pair_specs: list[dict] = []
    signs_used = set()
    clipped_used = False

    for motif_idx, orient in zip(motif_indices, orientations):
        motif = motifs[motif_idx]
        mh, mw = motif["block"].shape
        main_color = int(motif["main_color"])
        axis_len = mw if orient == "h" else mh
        for _ in range(200):
            if orient == "h":
                row = _randint_pair(rng, 0, h - mh + 1)
                if row in rows_used:
                    continue
                sign = 1 if rng.random() < 0.5 else -1
                if sign > 0:
                    max_start = w - axis_len - 1
                    if max_start < 0:
                        continue
                    start_c = _randint_pair(rng, 0, max_start + 1)
                    max_span = w - start_c
                    lengths = [L for L in range(axis_len + 1, max_span + 1)]
                    if not lengths:
                        continue
                    prefer = [L for L in lengths if L % axis_len != 0]
                    exact = [L for L in lengths if L % axis_len == 0]
                    if prefer and (not clipped_used or rng.random() < 0.75):
                        span_len = int(rng.choice(prefer))
                        clipped = True
                    elif exact:
                        span_len = int(rng.choice(exact))
                        clipped = False
                    else:
                        span_len = int(rng.choice(lengths))
                        clipped = span_len % axis_len != 0
                    end_c = start_c + span_len - 1
                else:
                    min_start = axis_len - 1
                    if min_start >= w:
                        continue
                    start_c = _randint_pair(rng, min_start, w)
                    max_span = start_c + 1
                    lengths = [L for L in range(axis_len + 1, max_span + 1)]
                    if not lengths:
                        continue
                    prefer = [L for L in lengths if L % axis_len != 0]
                    exact = [L for L in lengths if L % axis_len == 0]
                    if prefer and (not clipped_used or rng.random() < 0.75):
                        span_len = int(rng.choice(prefer))
                        clipped = True
                    elif exact:
                        span_len = int(rng.choice(exact))
                        clipped = False
                    else:
                        span_len = int(rng.choice(lengths))
                        clipped = span_len % axis_len != 0
                    end_c = start_c - span_len + 1
                start = (row, start_c)
                end = (row, end_c)
            else:
                col = _randint_pair(rng, 0, w - mw + 1)
                if col in cols_used:
                    continue
                sign = 1 if rng.random() < 0.5 else -1
                if sign > 0:
                    max_start = h - axis_len - 1
                    if max_start < 0:
                        continue
                    start_r = _randint_pair(rng, 0, max_start + 1)
                    max_span = h - start_r
                    lengths = [L for L in range(axis_len + 1, max_span + 1)]
                    if not lengths:
                        continue
                    prefer = [L for L in lengths if L % axis_len != 0]
                    exact = [L for L in lengths if L % axis_len == 0]
                    if prefer and (not clipped_used or rng.random() < 0.75):
                        span_len = int(rng.choice(prefer))
                        clipped = True
                    elif exact:
                        span_len = int(rng.choice(exact))
                        clipped = False
                    else:
                        span_len = int(rng.choice(lengths))
                        clipped = span_len % axis_len != 0
                    end_r = start_r + span_len - 1
                else:
                    min_start = axis_len - 1
                    if min_start >= h:
                        continue
                    start_r = _randint_pair(rng, min_start, h)
                    max_span = start_r + 1
                    lengths = [L for L in range(axis_len + 1, max_span + 1)]
                    if not lengths:
                        continue
                    prefer = [L for L in lengths if L % axis_len != 0]
                    exact = [L for L in lengths if L % axis_len == 0]
                    if prefer and (not clipped_used or rng.random() < 0.75):
                        span_len = int(rng.choice(prefer))
                        clipped = True
                    elif exact:
                        span_len = int(rng.choice(exact))
                        clipped = False
                    else:
                        span_len = int(rng.choice(lengths))
                        clipped = span_len % axis_len != 0
                    end_r = start_r - span_len + 1
                start = (start_r, col)
                end = (end_r, col)

            strip_box = _strip_bbox(start, end, motif["block"].shape)
            if any(_rect_touch_or_overlap(strip_box, box, pad=1) for box in occupied):
                continue

            start_box = (start[0], start[1], start[0], start[1])
            end_box = (end[0], end[1], end[0], end[1])
            if any(_rect_touch_or_overlap(start_box, box, pad=1) for box in occupied):
                continue
            if any(_rect_touch_or_overlap(end_box, box, pad=1) for box in occupied):
                continue

            row_or_col_conflict = False
            for prev in pair_specs:
                psr, psc = prev["start"]
                per, pec = prev["end"]
                if per == start[0] or pec == start[1]:
                    row_or_col_conflict = True
                    break
                if psr == end[0] or psc == end[1]:
                    row_or_col_conflict = True
                    break
                if start[0] == end[0]:
                    if psr == per == start[0]:
                        row_or_col_conflict = True
                        break
                    if psc == pec and psc in {start[1], end[1]}:
                        row_or_col_conflict = True
                        break
                else:
                    if psc == pec == start[1]:
                        row_or_col_conflict = True
                        break
                    if psr == per and psr in {start[0], end[0]}:
                        row_or_col_conflict = True
                        break
            if row_or_col_conflict:
                continue

            if orient == "h" and np.array_equal(motif["block"][:, 0], motif["block"][:, -1]):
                continue
            if orient == "v" and np.array_equal(motif["block"][0, :], motif["block"][-1, :]):
                continue

            occupied.extend([strip_box, start_box, end_box])
            pair_specs.append(
                {
                    "start": start,
                    "end": end,
                    "color": main_color,
                    "motif": motif["block"],
                    "clipped": clipped,
                }
            )
            if orient == "h":
                rows_used.add(start[0])
            else:
                cols_used.add(start[1])
            signs_used.add(np.sign((end[1] - start[1]) or (end[0] - start[0])))
            clipped_used = clipped_used or clipped
            break
        else:
            return None

    if len(signs_used) < 2 and len(pair_specs) >= 2:
        return None
    if not clipped_used:
        return None
    if not any(spec["start"][0] == spec["end"][0] for spec in pair_specs):
        return None
    if not any(spec["start"][1] == spec["end"][1] for spec in pair_specs):
        return None
    return pair_specs


def _semantic_checks(grid: np.ndarray, motifs: list[dict], pair_specs: list[dict]) -> bool:
    out = main(grid)
    if np.array_equal(out, grid):
        return False

    parsed_motifs, parsed_singletons = _component_boxes(grid)
    if len(parsed_motifs) != len(motifs):
        return False
    if len(parsed_singletons) != 2 * len(pair_specs):
        return False

    for spec in pair_specs:
        motif = spec["motif"]
        start = spec["start"]
        end = spec["end"]
        region = _paint_pair_variant(grid.shape, motif, start, end)
        if not np.any(region):
            return False
        mask = region != BACKGROUND
        if not np.array_equal(out[mask], region[mask]):
            return False

        sr, sc = start
        er, ec = end
        axis_len = motif.shape[1] if sr == er else motif.shape[0]
        span_len = abs(ec - sc) + 1 if sr == er else abs(er - sr) + 1
        if span_len % axis_len == 0:
            return False

        if (sr == er and ec < sc) or (sc == ec and er < sr):
            wrong = _paint_pair_variant(grid.shape, motif, start, end, force_forward_anchor=True)
            wrong_mask = wrong != BACKGROUND
            if np.array_equal(out[wrong_mask], wrong[wrong_mask]):
                return False

    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = _randint_pair(rng, *GRID_H_RANGE)
    w = _randint_pair(rng, *GRID_W_RANGE)
    grid = np.full((h, w), BACKGROUND, dtype=int)

    motif_count = int(rng.integers(MOTIF_COUNT_RANGE[0], MOTIF_COUNT_RANGE[1]))
    if motif_count < 2:
        return None

    main_colors = list(rng.choice(MOTIF_COLORS, size=motif_count, replace=False))
    motifs: list[dict] = []
    occupied: list[tuple[int, int, int, int]] = []

    for main_color in main_colors:
        placed = False
        for _ in range(200):
            mh = _randint_pair(rng, *MOTIF_H_RANGE)
            mw = _randint_pair(rng, *MOTIF_W_RANGE)
            if mh >= h - 2 or mw >= w - 2:
                continue
            r0 = _randint_pair(rng, 1, h - mh)
            c0 = _randint_pair(rng, 1, w - mw)
            box = (r0, c0, r0 + mh - 1, c0 + mw - 1)
            if any(_rect_touch_or_overlap(box, prev, pad=1) for prev in occupied):
                continue
            motif = _build_motif(rng, mh, mw, int(main_color))
            if np.array_equal(motif[:, 0], motif[:, -1]) and np.array_equal(motif[0, :], motif[-1, :]):
                continue
            grid[r0 : r0 + mh, c0 : c0 + mw] = motif
            motifs.append({"bbox": box, "block": motif, "main_color": int(main_color)})
            occupied.append(box)
            placed = True
            break
        if not placed:
            return None

    pair_specs = _sample_pair_specs(rng, motifs, grid.shape)
    if pair_specs is None:
        return None

    for spec in pair_specs:
        sr, sc = spec["start"]
        er, ec = spec["end"]
        grid[sr, sc] = int(spec["color"])
        grid[er, ec] = GREY

    if not _semantic_checks(grid, motifs, pair_specs):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
