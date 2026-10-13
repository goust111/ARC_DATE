from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from arc_utils import crop_bbox, find_components, get_bg, paste


MAX_PERIOD = 6
GEN_RETRIES = 500
BG = 0


def _border_coords(h: int, w: int) -> list[tuple[int, int]]:
    coords = [(0, c) for c in range(w)]
    coords.extend((r, w - 1) for r in range(1, h - 1))
    coords.extend((h - 1, c) for c in range(w - 1, -1, -1))
    coords.extend((r, 0) for r in range(h - 2, 0, -1))
    return coords


def _is_primitive(bits: tuple[int, ...]) -> bool:
    n = len(bits)
    for d in range(1, n):
        if n % d != 0:
            continue
        if all(bits[i] == bits[i % d] for i in range(n)):
            return False
    return True


def _majority_pattern(
    values: list[int], colors: tuple[int, int], period: int
) -> tuple[tuple[int, ...], int, int] | None:
    pattern: list[int] = []
    mismatches = 0
    tie_count = 0
    for offset in range(period):
        bucket = values[offset::period]
        count0 = sum(v == colors[0] for v in bucket)
        count1 = len(bucket) - count0
        if count0 == count1:
            tie_count += 1
        expected = colors[0] if count0 >= count1 else colors[1]
        pattern.append(expected)
        mismatches += sum(v != expected for v in bucket)
    if len(set(pattern)) != 2:
        return None
    return tuple(pattern), mismatches, tie_count


def _infer_border_model(grid: np.ndarray) -> dict:
    h, w = grid.shape
    coords = _border_coords(h, w)
    values = [int(grid[r, c]) for r, c in coords]
    colors = tuple(sorted(set(values)))
    assert len(colors) == 2, "expected exactly two border colors"

    candidates = []
    max_period = min(MAX_PERIOD, len(values) // 2)
    for period in range(2, max_period + 1):
        info = _majority_pattern(values, colors, period)
        if info is None:
            continue
        pattern, mismatches, tie_count = info
        color_bits = tuple(0 if v == colors[0] else 1 for v in pattern)
        if not _is_primitive(color_bits):
            continue
        expected = [pattern[i % period] for i in range(len(values))]
        anomaly_coords = [coords[i] for i, v in enumerate(values) if v != expected[i]]
        candidates.append(
            {
                "period": period,
                "pattern": pattern,
                "mismatches": mismatches,
                "tie_count": tie_count,
                "anomalies": anomaly_coords,
            }
        )

    assert candidates, "failed to infer a periodic border model"
    candidates.sort(key=lambda item: (item["mismatches"], item["tie_count"], item["period"]))
    best = candidates[0]
    if len(candidates) > 1:
        nxt = candidates[1]
        assert (best["mismatches"], best["tie_count"]) < (nxt["mismatches"], nxt["tie_count"]), \
            "border period is not uniquely determined"
    assert best["mismatches"] > 0, "expected at least one border anomaly"
    return best


def _extract_template(grid: np.ndarray, bg: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    comps = find_components(grid, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    interior = []
    h, w = grid.shape
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if r0 > 0 and c0 > 0 and r1 < h - 1 and c1 < w - 1:
            interior.append(comp)
    assert len(interior) == 1, "expected one interior multicolor template"
    bbox = tuple(map(int, interior[0]["bbox"]))
    patch = crop_bbox(grid, bbox)
    assert patch.shape[0] % 2 == 1 and patch.shape[1] % 2 == 1, "template bbox must have an unambiguous center"
    return patch, bbox


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the border cycle and stamp the interior sprite at the anomaly row/column intersections.

    Input structure:
    - The grid has a one-cell-thick two-color lace border running around the full perimeter.
    - A few border cells break that short cycle by swapping to the other lace color, while the interior stays black except for one small multicolor template sprite.
    - The template sprite is fully inside the border and does not touch the perimeter.

    Rule:
    - Infer the unique short periodic border pattern from the overall perimeter and mark every border cell that disagrees with that recovered pattern as an anomaly.
    - Every anomaly on the top or bottom edge contributes its column; every anomaly on the left or right edge contributes its row.
    - Keep the original template and border unchanged, then stamp transparent copies of the template centered at every Cartesian-product intersection of those anomaly rows and columns.

    Key insight:
    - The useful information is not the border colors themselves but the cells that violate the border rhythm; those mismatches encode rows and columns for centered template replication.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    assert bg == BG, "expected black interior background"

    border = _infer_border_model(g)
    rows = sorted({r for r, c in border["anomalies"] if c in {0, g.shape[1] - 1}})
    cols = sorted({c for r, c in border["anomalies"] if r in {0, g.shape[0] - 1}})
    assert rows and cols, "need both anomaly rows and anomaly columns"

    template, _bbox = _extract_template(g, bg)
    th, tw = map(int, template.shape)
    rh = th // 2
    cw = tw // 2
    mask = template != bg

    out = g.copy()
    for r in rows:
        for c in cols:
            r0 = int(r - rh)
            c0 = int(c - cw)
            assert 1 <= r0 and r0 + th <= g.shape[0] - 1
            assert 1 <= c0 and c0 + tw <= g.shape[1] - 1
            target = out[r0 : r0 + th, c0 : c0 + tw]
            assert np.all((~mask) | (target == bg) | (target == template)), "copies must not conflict"
            paste(out, template, r0, c0, transparent=bg)
    return out


def _pick_spaced(candidates: list[int], count: int, min_gap: int) -> list[int] | None:
    random.shuffle(candidates)
    picked: list[int] = []
    for value in candidates:
        if all(abs(value - prev) >= min_gap for prev in picked):
            picked.append(value)
            if len(picked) == count:
                return sorted(picked)
    return None


def _random_template(available_colors: list[int]) -> np.ndarray | None:
    for _ in range(200):
        h = random.choice([3, 5])
        w = random.choice([3, 5])
        target_area = random.randint(max(5, (h + w) // 2), min(h * w - 1, 10))
        coords = {(h // 2, w // 2)}
        while len(coords) < target_area:
            frontier = []
            for r, c in coords:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in coords:
                        frontier.append((nr, nc))
            if not frontier:
                break
            coords.add(random.choice(frontier))
        mask = np.zeros((h, w), dtype=bool)
        for r, c in coords:
            mask[r, c] = True
        if not mask[h // 2, w // 2]:
            continue
        if not mask[0].any() or not mask[h - 1].any() or not mask[:, 0].any() or not mask[:, w - 1].any():
            continue
        if any(np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3)):
            continue
        n_colors = random.randint(2, min(4, len(available_colors)))
        palette = random.sample(available_colors, n_colors)
        patch = np.zeros((h, w), dtype=int)
        used: set[int] = set()
        for r, c in coords:
            color = random.choice(palette)
            patch[r, c] = color
            used.add(color)
        if len(used) < 2:
            rr, cc = random.choice(list(coords))
            alt = next(color for color in palette if color != patch[rr, cc])
            patch[rr, cc] = alt
        return patch
    return None


def _place_template(
    grid: np.ndarray, template: np.ndarray, blocked_boxes: list[tuple[int, int, int, int]]
) -> tuple[int, int] | None:
    h, w = grid.shape
    th, tw = template.shape
    candidates = []
    for r0 in range(1, h - th):
        for c0 in range(1, w - tw):
            box = (r0, c0, r0 + th - 1, c0 + tw - 1)
            ok = True
            for br0, bc0, br1, bc1 in blocked_boxes:
                if not (box[2] < br0 - 1 or br1 + 1 < box[0] or box[3] < bc0 - 1 or bc1 + 1 < box[1]):
                    ok = False
                    break
            if ok:
                candidates.append((r0, c0))
    if not candidates:
        return None
    return random.choice(candidates)


def _assign_sides(values: list[int], side_a: str, side_b: str) -> dict[int, str]:
    if len(values) == 1:
        return {values[0]: random.choice([side_a, side_b])}
    if random.random() < 0.7:
        shuffled = values[:]
        random.shuffle(shuffled)
        assign = {shuffled[0]: side_a, shuffled[1]: side_b}
        for value in shuffled[2:]:
            assign[value] = random.choice([side_a, side_b])
        return assign
    return {value: random.choice([side_a, side_b]) for value in values}


def _render_border(
    shape: tuple[int, int], colors: tuple[int, int], pattern_bits: tuple[int, ...]
) -> tuple[np.ndarray, list[tuple[int, int]], list[int]]:
    h, w = shape
    grid = np.zeros((h, w), dtype=int)
    coords = _border_coords(h, w)
    seq = []
    for i, (r, c) in enumerate(coords):
        value = colors[pattern_bits[i % len(pattern_bits)]]
        grid[r, c] = value
        seq.append(value)
    return grid, coords, seq


def _stamp_output(
    inp: np.ndarray, template: np.ndarray, rows: list[int], cols: list[int]
) -> np.ndarray:
    out = inp.copy()
    th, tw = template.shape
    for r in rows:
        for c in cols:
            paste(out, template, r - th // 2, c - tw // 2, transparent=BG)
    return out


def _all_pattern_bits(period: int) -> list[tuple[int, ...]]:
    patterns = []
    for mask in range(1, (1 << period) - 1):
        bits = tuple((mask >> i) & 1 for i in range(period))
        if _is_primitive(bits):
            patterns.append(bits)
    return patterns


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(15, 28)
    w = random.randint(15, 28)
    if max(h, w) > 30:
        return None

    period = random.choice([2, 3, 4, 5])
    pattern_bits = random.choice(_all_pattern_bits(period))

    border_colors = tuple(random.sample(list(range(1, 10)), 2))
    template = _random_template([c for c in range(1, 10) if c not in border_colors])
    if template is None:
        return None
    th, tw = map(int, template.shape)

    row_candidates = list(range(1 + th // 2, h - 1 - th // 2))
    col_candidates = list(range(1 + tw // 2, w - 1 - tw // 2))
    if len(row_candidates) < 2 or len(col_candidates) < 2:
        return None

    row_count = random.choice([1, 2, 2, 3])
    col_count = random.choice([1, 2, 2, 3])
    if row_count * col_count < 2:
        if len(row_candidates) >= 2:
            row_count = 2
        else:
            col_count = 2

    rows = _pick_spaced(row_candidates, row_count, th + 1)
    cols = _pick_spaced(col_candidates, col_count, tw + 1)
    if rows is None or cols is None:
        return None

    copy_boxes = [(r - th // 2, c - tw // 2, r + th // 2, c + tw // 2) for r in rows for c in cols]
    base = _render_border((h, w), border_colors, pattern_bits)[0]

    origin = _place_template(base, template, copy_boxes)
    if origin is None:
        return None
    r0, c0 = origin
    paste(base, template, r0, c0, transparent=BG)

    row_sides = _assign_sides(rows, "left", "right")
    col_sides = _assign_sides(cols, "top", "bottom")
    anomaly_coords = []
    for r in rows:
        anomaly_coords.append((r, 0 if row_sides[r] == "left" else w - 1))
    for c in cols:
        anomaly_coords.append((0 if col_sides[c] == "top" else h - 1, c))

    inp = base.copy()
    for r, c in anomaly_coords:
        inp[r, c] = border_colors[0] if inp[r, c] == border_colors[1] else border_colors[1]

    out = _stamp_output(inp, template, rows, cols)

    try:
        model = _infer_border_model(inp)
        got_rows = sorted({r for r, c in model["anomalies"] if c in {0, w - 1}})
        got_cols = sorted({c for r, c in model["anomalies"] if r in {0, h - 1}})
        if got_rows != rows or got_cols != cols:
            return None
        extracted, extracted_bbox = _extract_template(inp, BG)
        if not np.array_equal(extracted, template):
            return None
        if extracted_bbox == (r0, c0, r0 + th - 1, c0 + tw - 1):
            pass
        if np.array_equal(inp, out):
            return None
        solved = main(inp)
        if not np.array_equal(solved, out):
            return None
    except AssertionError:
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        pair = _build_case()
        if pair is not None:
            return pair[0]
    raise RuntimeError("failed to generate valid input for puzzle 4")
