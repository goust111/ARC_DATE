from __future__ import annotations

from collections import Counter
import random

import numpy as np


BBox = tuple[int, int, int, int]
Coord = tuple[int, int]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def bbox(mask: np.ndarray) -> BBox | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _all_singletons(mask: np.ndarray) -> bool:
    rows, cols = mask.shape
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < rows and 0 <= nc < cols and bool(mask[nr, nc]):
                return False
    return bool(mask.any())


def _marker_sides(markers: list[Coord], box: BBox) -> dict[str, int]:
    r0, c0, r1, c1 = box
    counts = {"top": 0, "bottom": 0, "left": 0, "right": 0}
    for r, c in markers:
        if r == r0 - 1 and c0 <= c <= c1:
            counts["top"] += 1
        elif r == r1 + 1 and c0 <= c <= c1:
            counts["bottom"] += 1
        elif c == c0 - 1 and r0 <= r <= r1:
            counts["left"] += 1
        elif c == c1 + 1 and r0 <= r <= r1:
            counts["right"] += 1
        else:
            return {}
    return counts


def _infer_target_bbox(markers: list[Coord], shape: tuple[int, int]) -> BBox | None:
    rows, cols = shape
    r_values = [r for r, _ in markers]
    c_values = [c for _, c in markers]
    r0 = min(r_values) + 1
    c0 = min(c_values) + 1
    r1 = max(r_values) - 1
    c1 = max(c_values) - 1
    if not (0 <= r0 <= r1 < rows and 0 <= c0 <= c1 < cols):
        return None
    side_counts = _marker_sides(markers, (r0, c0, r1, c1))
    if not side_counts:
        return None
    if min(side_counts.values()) < 1:
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _find_min_period_axis(src: np.ndarray, pollution: int, axis: int) -> int | None:
    limit = src.shape[axis]
    for period in range(1, limit):
        saw_comparison = False
        if axis == 0:
            for r in range(limit - period):
                upper = src[r]
                lower = src[r + period]
                for a, b in zip(upper, lower):
                    aa = int(a)
                    bb = int(b)
                    if aa == pollution or bb == pollution:
                        continue
                    saw_comparison = True
                    if aa != bb:
                        break
                else:
                    continue
                break
            else:
                if saw_comparison:
                    return period
        else:
            for c in range(limit - period):
                left = src[:, c]
                right = src[:, c + period]
                for a, b in zip(left, right):
                    aa = int(a)
                    bb = int(b)
                    if aa == pollution or bb == pollution:
                        continue
                    saw_comparison = True
                    if aa != bb:
                        break
                else:
                    continue
                break
            else:
                if saw_comparison:
                    return period
    return None


def _repair_source(src: np.ndarray, pollution: int, pr: int, pc: int) -> np.ndarray | None:
    h, w = src.shape
    colors: dict[tuple[int, int], int] = {}
    for rr in range(pr):
        for cc in range(pc):
            vals = {
                int(src[r, c])
                for r in range(rr, h, pr)
                for c in range(cc, w, pc)
                if int(src[r, c]) != pollution
            }
            if not vals or len(vals) != 1:
                return None
            colors[(rr, cc)] = next(iter(vals))

    repaired = src.copy()
    polluted = repaired == pollution
    if not bool(polluted.any()):
        return None
    for r, c in np.argwhere(polluted):
        rr = int(r)
        cc = int(c)
        repaired[rr, cc] = colors[(rr % pr, cc % pc)]

    if np.array_equal(repaired, src):
        return None
    if np.any(repaired == pollution):
        return None

    for r in range(h - pr):
        if not np.array_equal(repaired[r], repaired[r + pr]):
            return None
    for c in range(w - pc):
        if not np.array_equal(repaired[:, c], repaired[:, c + pc]):
            return None
    return repaired


def _solve_instance(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = get_bg(g)
    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    candidates: list[dict[str, object]] = []

    for control in colors:
        marker_mask = g == control
        if not _all_singletons(marker_mask):
            continue
        markers = [tuple(map(int, coord)) for coord in np.argwhere(marker_mask)]
        target_box = _infer_target_bbox(markers, g.shape)
        if target_box is None:
            continue

        src_mask = (g != bg) & (g != control)
        src_box = bbox(src_mask)
        if src_box is None:
            continue
        sr0, sc0, sr1, sc1 = src_box
        tr0, tc0, tr1, tc1 = target_box
        if not (tr0 < sr0 <= sr1 < tr1 and tc0 < sc0 <= sc1 < tc1):
            continue
        src_patch = g[sr0 : sr1 + 1, sc0 : sc1 + 1]
        if np.any(src_patch == bg) or np.any(src_patch == control):
            continue
        if not np.all(src_mask[sr0 : sr1 + 1, sc0 : sc1 + 1]):
            continue

        repairs: list[dict[str, object]] = []
        for pollution in [int(c) for c in np.unique(src_patch)]:
            pr = _find_min_period_axis(src_patch, pollution, axis=0)
            pc = _find_min_period_axis(src_patch, pollution, axis=1)
            if pr is None or pc is None:
                continue
            repaired = _repair_source(src_patch, pollution, pr, pc)
            if repaired is None:
                continue
            repairs.append(
                {
                    "pollution": int(pollution),
                    "period": (int(pr), int(pc)),
                    "repaired": repaired,
                }
            )
        if len(repairs) != 1:
            continue
        candidates.append(
            {
                "bg": int(bg),
                "control": int(control),
                "markers": markers,
                "target_box": target_box,
                "source_box": src_box,
                **repairs[0],
            }
        )

    assert len(candidates) == 1, "expected a unique control/pollution interpretation"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the polluted periodic patch, then tile that repaired phase across the marker-bounded rectangle.

    Input structure:
    - The background is the dominant color.
    - A small filled rectangle contains a colorful pattern with both a vertical and a horizontal translation period, except that a few cells have been overwritten by one shared pollution color.
    - Isolated control pixels of another color sit exactly one cell outside the four sides of the larger target rectangle.

    Rule:
    - Use the translation periods visible inside the small rectangle to identify the pollution color and restore every polluted cell from the consistent non-polluted cells in its period orbit.
    - Keep the repaired small rectangle fixed in place, then extend the repaired periodic pattern in both directions to fill the target rectangle defined by the control pixels.
    - Truncate the tiling at the target border and leave the control pixels unchanged.

    Key insight:
    - The same periodic structure solves both subproblems: it first proves which cells are corrupted, then it provides the only phase-consistent way to extend the pattern beyond its original bounds.
    """
    info = _solve_instance(np.asarray(grid, dtype=int))
    out = np.asarray(grid, dtype=int).copy()

    repaired = np.asarray(info["repaired"], dtype=int)
    pr, pc = map(int, info["period"])
    sr0, sc0, _sr1, _sc1 = map(int, info["source_box"])
    tr0, tc0, tr1, tc1 = map(int, info["target_box"])

    for r in range(tr0, tr1 + 1):
        for c in range(tc0, tc1 + 1):
            out[r, c] = int(repaired[(r - sr0) % pr, (c - sc0) % pc])
    return out


PALETTE = list(range(10))
BG = 0
GEN_ATTEMPTS = 400


def _cyclic_period_rows(tile: np.ndarray) -> int:
    h = tile.shape[0]
    for period in range(1, h + 1):
        if all(np.array_equal(tile[r], tile[(r + period) % h]) for r in range(h)):
            return period
    return h


def _cyclic_period_cols(tile: np.ndarray) -> int:
    w = tile.shape[1]
    for period in range(1, w + 1):
        if all(np.array_equal(tile[:, c], tile[:, (c + period) % w]) for c in range(w)):
            return period
    return w


def _random_tile(pr: int, pc: int, palette: list[int]) -> np.ndarray:
    while True:
        n_colors = random.randint(3, min(5, len(palette)))
        colors = random.sample(palette, n_colors)
        tile = np.array([[random.choice(colors) for _ in range(pc)] for _ in range(pr)], dtype=int)
        if len(set(map(int, tile.ravel()))) < 3:
            continue
        if _cyclic_period_rows(tile) != pr or _cyclic_period_cols(tile) != pc:
            continue
        return tile


def _pick_side_offsets(length: int, count: int) -> list[int] | None:
    if length <= 0:
        return None
    preferred = list(range(1, length - 1)) if length >= 4 else list(range(length))
    for pool in (preferred, list(range(length))):
        if len(pool) < count:
            continue
        shuffled = pool[:]
        random.shuffle(shuffled)
        chosen: list[int] = []
        for pos in shuffled:
            if all(abs(pos - prev) >= 2 for prev in chosen):
                chosen.append(pos)
                if len(chosen) == count:
                    return sorted(chosen)
    return None


def _tile_color(tile: np.ndarray, r: int, c: int) -> int:
    return int(tile[r % tile.shape[0], c % tile.shape[1]])


def _fill_pattern(grid: np.ndarray, box: BBox, src_origin: Coord, tile: np.ndarray) -> None:
    r0, c0, r1, c1 = box
    sr0, sc0 = src_origin
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            grid[r, c] = _tile_color(tile, r - sr0, c - sc0)


def _wrong_unrepaired_extension(inp: np.ndarray, source_box: BBox, target_box: BBox, control: int, pollution: int, pr: int, pc: int, bg: int) -> np.ndarray | None:
    sr0, sc0, sr1, sc1 = source_box
    src = inp[sr0 : sr1 + 1, sc0 : sc1 + 1]
    residue_colors: dict[tuple[int, int], int] = {}
    for rr in range(pr):
        for cc in range(pc):
            vals = [int(src[r, c]) for r in range(rr, src.shape[0], pr) for c in range(cc, src.shape[1], pc)]
            if pollution in vals:
                residue_colors[(rr, cc)] = pollution
            else:
                uniq = set(vals)
                if len(uniq) != 1:
                    return None
                residue_colors[(rr, cc)] = next(iter(uniq))

    out = np.full_like(inp, bg)
    out[inp == control] = control
    tr0, tc0, tr1, tc1 = target_box
    for r in range(tr0, tr1 + 1):
        for c in range(tc0, tc1 + 1):
            out[r, c] = residue_colors[((r - sr0) % pr, (c - sc0) % pc)]
    return out


def _wrong_reanchored_extension(shape: tuple[int, int], target_box: BBox, tile: np.ndarray, control_markers: list[Coord], control: int, bg: int) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for r, c in control_markers:
        out[r, c] = control
    tr0, tc0, tr1, tc1 = target_box
    for r in range(tr0, tr1 + 1):
        for c in range(tc0, tc1 + 1):
            out[r, c] = _tile_color(tile, r - tr0, c - tc0)
    return out


def _semantic_checks(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        info = _solve_instance(inp)
    except Exception:
        return False
    if not np.array_equal(main(inp), expected):
        return False
    if np.array_equal(inp, expected):
        return False

    bg = int(info["bg"])
    control = int(info["control"])
    pollution = int(info["pollution"])
    pr, pc = map(int, info["period"])
    source_box = tuple(map(int, info["source_box"]))
    target_box = tuple(map(int, info["target_box"]))
    markers = [tuple(map(int, p)) for p in info["markers"]]
    repaired = np.asarray(info["repaired"], dtype=int)

    sr0, sc0, sr1, sc1 = source_box
    tr0, tc0, tr1, tc1 = target_box
    if not (tr0 < sr0 and sc0 > tc0 and sr1 < tr1 and sc1 < tc1):
        return False
    if (tr1 - tr0 + 1) % pr == 0 and (tr1 - tr0 + 1) % pr == 0 and (tc1 - tc0 + 1) % pc == 0:
        return False

    wrong_unrepaired = _wrong_unrepaired_extension(inp, source_box, target_box, control, pollution, pr, pc, bg)
    if wrong_unrepaired is None or np.array_equal(wrong_unrepaired, expected):
        return False

    wrong_phase = _wrong_reanchored_extension(inp.shape, target_box, repaired[:pr, :pc], markers, control, bg)
    if np.array_equal(wrong_phase, expected):
        return False

    return True


def _build_instance() -> tuple[np.ndarray, np.ndarray] | None:
    pr = random.randint(2, 4)
    pc = random.randint(2, 4)
    rep_r = random.randint(2, 4)
    rep_c = random.randint(2, 4)
    src_h = pr * rep_r
    src_w = pc * rep_c

    exp_top = random.randint(1, 4)
    exp_bottom = random.randint(1, 4)
    exp_left = random.randint(1, 4)
    exp_right = random.randint(1, 4)
    if exp_top % pr == 0 and exp_left % pc == 0:
        return None

    target_h = src_h + exp_top + exp_bottom
    target_w = src_w + exp_left + exp_right
    if target_h > 24 or target_w > 24:
        return None
    if target_h % pr == 0 and target_w % pc == 0:
        return None

    outer_top = random.randint(1, 2)
    outer_bottom = random.randint(1, 2)
    outer_left = random.randint(1, 2)
    outer_right = random.randint(1, 2)
    height = target_h + outer_top + outer_bottom + 2
    width = target_w + outer_left + outer_right + 2
    if height > 30 or width > 30:
        return None

    target_box = (outer_top + 1, outer_left + 1, outer_top + target_h, outer_left + target_w)
    tr0, tc0, tr1, tc1 = target_box
    source_box = (tr0 + exp_top, tc0 + exp_left, tr1 - exp_bottom, tc1 - exp_right)
    sr0, sc0, sr1, sc1 = source_box

    control = random.choice([c for c in PALETTE if c != BG])
    pollution = random.choice([c for c in PALETTE if c not in (BG, control)])
    tile_palette = [c for c in PALETTE if c not in (BG, control, pollution)]
    tile = _random_tile(pr, pc, tile_palette)

    expected = np.full((height, width), BG, dtype=int)
    _fill_pattern(expected, target_box, (sr0, sc0), tile)

    side_counts = {side: random.randint(1, 2) for side in ("top", "bottom", "left", "right")}
    top_offsets = _pick_side_offsets(target_w, side_counts["top"])
    bottom_offsets = _pick_side_offsets(target_w, side_counts["bottom"])
    left_offsets = _pick_side_offsets(target_h, side_counts["left"])
    right_offsets = _pick_side_offsets(target_h, side_counts["right"])
    if None in (top_offsets, bottom_offsets, left_offsets, right_offsets):
        return None

    markers: list[Coord] = []
    markers.extend((tr0 - 1, tc0 + off) for off in top_offsets or [])
    markers.extend((tr1 + 1, tc0 + off) for off in bottom_offsets or [])
    markers.extend((tr0 + off, tc0 - 1) for off in left_offsets or [])
    markers.extend((tr0 + off, tc1 + 1) for off in right_offsets or [])
    if len(set(markers)) != len(markers):
        return None
    for r, c in markers:
        if expected[r, c] != BG:
            return None
        expected[r, c] = control

    inp = np.full((height, width), BG, dtype=int)
    inp[expected == control] = control
    _fill_pattern(inp, source_box, (sr0, sc0), tile)

    residue_classes = [(rr, cc) for rr in range(pr) for cc in range(pc)]
    random.shuffle(residue_classes)
    n_classes = random.randint(1, min(4, len(residue_classes)))
    polluted_any = False
    for rr, cc in residue_classes[:n_classes]:
        orbit = [(r, c) for r in range(sr0 + rr, sr1 + 1, pr) for c in range(sc0 + cc, sc1 + 1, pc)]
        if len(orbit) < 2:
            return None
        n_polluted = random.randint(1, len(orbit) - 1)
        for r, c in random.sample(orbit, n_polluted):
            inp[r, c] = pollution
            polluted_any = True
    if not polluted_any:
        return None

    if np.count_nonzero(inp == BG) <= inp.size // 2:
        return None
    if np.array_equal(inp, expected):
        return None
    if not _semantic_checks(inp, expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_instance()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid repair-then-extend ARC input")
