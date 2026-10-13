from __future__ import annotations

import random

import numpy as np


GREEN = 3
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS8 = NEIGHBORS4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
SEED_COLORS = (1, 2, 4, 5, 6, 7, 8, 9)
GEN_RETRIES = 300
PLACE_RETRIES = 120


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray, connectivity: int = 8) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    steps = NEIGHBORS8 if connectivity == 8 else NEIGHBORS4
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        seen[rr, cc] = True
        stack = [(rr, cc)]
        comp: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            comp.append((cr, cc))
            for dr, dc in steps:
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_from_component(coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask, (r0, c0, r1, c1)


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = np.asarray(mask, dtype=bool).copy()
    src = np.asarray(mask, dtype=bool)
    out[1:, :] |= src[:-1, :]
    out[:-1, :] |= src[1:, :]
    out[:, 1:] |= src[:, :-1]
    out[:, :-1] |= src[:, 1:]
    out[1:, 1:] |= src[:-1, :-1]
    out[1:, :-1] |= src[:-1, 1:]
    out[:-1, 1:] |= src[1:, :-1]
    out[:-1, :-1] |= src[1:, 1:]
    return out


def _mask_to_cells(mask: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    cells = []
    for r, c in np.argwhere(mask):
        cells.append((top + int(r), left + int(c)))
    return cells


def _pattern_cells_at_center(mask: np.ndarray, center_r: int, center_c: int) -> list[tuple[int, int]]:
    h, w = mask.shape
    top = center_r - (h - 1) // 2
    left = center_c - (w - 1) // 2
    return _mask_to_cells(mask, top, left)


def _extract_dictionary_from_band(
    band: np.ndarray, bg: int
) -> tuple[int, list[np.ndarray]] | None:
    colors = np.unique(band[band != bg])
    if colors.size != 1:
        return None
    dict_color = int(colors[0])
    comps = _components(band == dict_color, connectivity=8)
    if not (3 <= len(comps) <= 5):
        return None

    items = []
    for comp in comps:
        mask, bbox = _mask_from_component(comp)
        items.append(
            {
                "c0": int(bbox[1]),
                "c1": int(bbox[3]),
                "area": int(mask.sum()),
                "mask": mask,
            }
        )
    items.sort(key=lambda item: item["c0"])

    for prev, cur in zip(items, items[1:]):
        if int(cur["c0"]) <= int(prev["c1"]) + 1:
            return None
    areas = [int(item["area"]) for item in items]
    if any(a0 >= a1 for a0, a1 in zip(areas, areas[1:])):
        return None

    return dict_color, [np.asarray(item["mask"], dtype=bool) for item in items]


def _seed_is_isolated(grid: np.ndarray, bg: int, r: int, c: int) -> bool:
    h, w = grid.shape
    for dr, dc in NEIGHBORS8:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) != bg:
            return False
    return True


def _extract_seed_pairs(
    grid: np.ndarray, bg: int, band_rows: slice, num_patterns: int
) -> list[dict] | None:
    h, _w = grid.shape
    inside_band = np.zeros(h, dtype=bool)
    inside_band[band_rows] = True

    by_color: dict[int, list[tuple[int, int]]] = {}
    for r, c in np.argwhere(grid != bg):
        rr = int(r)
        cc = int(c)
        if inside_band[rr]:
            continue
        by_color.setdefault(int(grid[rr, cc]), []).append((rr, cc))

    if not (3 <= len(by_color) <= 5):
        return None

    pairs = []
    for color, pts in by_color.items():
        if len(pts) != 2:
            return None
        if not all(_seed_is_isolated(grid, bg, r, c) for r, c in pts):
            return None

        (r0, c0), (r1, c1) = sorted(pts)
        same_row = r0 == r1
        same_col = c0 == c1
        if same_row == same_col:
            return None

        if same_row:
            lo, hi = sorted((c0, c1))
            if any(int(grid[r0, cc]) != bg for cc in range(lo + 1, hi)):
                return None
            gap = hi - lo - 1
        else:
            lo, hi = sorted((r0, r1))
            if any(int(grid[rr, c0]) != bg for rr in range(lo + 1, hi)):
                return None
            gap = hi - lo - 1

        if not (1 <= gap <= num_patterns):
            return None

        pairs.append(
            {
                "color": int(color),
                "coords": [(r0, c0), (r1, c1)],
                "gap": int(gap),
                "center": ((r0 + r1) // 2, (c0 + c1) // 2),
            }
        )
    return pairs


def _find_dictionary_and_pairs(grid: np.ndarray) -> tuple[int, list[np.ndarray], list[dict], int]:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    h = int(g.shape[0])

    candidates = []
    for side in ("top", "bottom"):
        for band_h in range(1, h - 1):
            if side == "top":
                band = g[:band_h, :]
                band_rows = slice(0, band_h)
            else:
                band = g[h - band_h :, :]
                band_rows = slice(h - band_h, h)

            parsed = _extract_dictionary_from_band(band, bg)
            if parsed is None:
                continue
            dict_color, patterns = parsed
            pairs = _extract_seed_pairs(g, bg, band_rows, len(patterns))
            if pairs is None:
                continue
            candidates.append(
                {
                    "side": side,
                    "band_h": int(band_h),
                    "dict_color": int(dict_color),
                    "patterns": patterns,
                    "pairs": pairs,
                }
            )
            break

    if any(int(item["dict_color"]) == GREEN for item in candidates):
        candidates = [item for item in candidates if int(item["dict_color"]) == GREEN]

    assert len(candidates) == 1, "expected one valid edge dictionary"
    chosen = candidates[0]
    return bg, chosen["patterns"], chosen["pairs"], int(chosen["dict_color"])


def _paint_cells(out: np.ndarray, cells: list[tuple[int, int]], color: int, bg: int) -> None:
    h, w = out.shape
    for r, c in cells:
        assert 0 <= r < h and 0 <= c < w, "painted pattern must stay in bounds"
        assert int(out[r, c]) == bg, "painted pattern must not collide with existing structure"
        out[r, c] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-color isolated seeds, use their gap as a 1-based dictionary index,
    and repaint the chosen dictionary pattern at the pair midpoint in the seed color.
    """
    g = np.asarray(grid, dtype=int)
    bg, patterns, pairs, _dict_color = _find_dictionary_and_pairs(g)

    out = g.copy()
    for pair in pairs:
        for r, c in pair["coords"]:
            out[r, c] = bg

    for pair in pairs:
        center_r, center_c = pair["center"]
        pattern = patterns[int(pair["gap"]) - 1]
        cells = _pattern_cells_at_center(pattern, int(center_r), int(center_c))
        _paint_cells(out, cells, int(pair["color"]), bg)

    return out


def _mask_from_lines(lines: tuple[str, ...]) -> np.ndarray:
    return np.array([[ch == "#" for ch in row] for row in lines], dtype=bool)


PATTERN_LIBRARY = {
    1: [_mask_from_lines(("#",))],
    2: [_mask_from_lines(("##",)), _mask_from_lines(("#", "#"))],
    3: [_mask_from_lines(("###",)), _mask_from_lines(("#.", "##")), _mask_from_lines((".#", "##"))],
    4: [_mask_from_lines(("##", "##")), _mask_from_lines(("###", ".#."))],
    5: [_mask_from_lines((".#.", "###", ".#.")), _mask_from_lines(("##.", "###"))],
}


def _choose_patterns() -> list[np.ndarray]:
    count = random.randint(3, 5)
    return [random.choice(PATTERN_LIBRARY[idx]).copy() for idx in range(1, count + 1)]


def _reserve_mask(input_occ: np.ndarray, output_occ: np.ndarray) -> np.ndarray:
    return _dilate8(input_occ | output_occ)


def _in_bounds(cells: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _place_dictionary(
    grid: np.ndarray, out: np.ndarray, patterns: list[np.ndarray], side: str, color: int
) -> tuple[int, int]:
    band_h = max(int(pattern.shape[0]) for pattern in patterns)
    widths = [int(pattern.shape[1]) for pattern in patterns]
    gaps = [random.randint(1, 2) for _ in range(len(patterns) - 1)]
    dict_w = sum(widths) + sum(gaps)
    left = random.randint(1, grid.shape[1] - dict_w - 1)

    row0 = 0 if side == "top" else grid.shape[0] - band_h
    col = left
    for idx, pattern in enumerate(patterns):
        ph, pw = pattern.shape
        local_r = 0 if side == "top" else band_h - ph
        cells = _mask_to_cells(pattern, row0 + local_r, col)
        for r, c in cells:
            grid[r, c] = color
            out[r, c] = color
        col += pw
        if idx < len(gaps):
            col += gaps[idx]
    return band_h, left + dict_w


def _pair_seed_cells(center_r: int, center_c: int, gap: int, orientation: str) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    span = gap + 1
    if orientation == "h":
        c0 = center_c - span // 2
        c1 = c0 + span
        seeds = [(center_r, c0), (center_r, c1)]
        line = [(center_r, c) for c in range(c0 + 1, c1)]
    else:
        r0 = center_r - span // 2
        r1 = r0 + span
        seeds = [(r0, center_c), (r1, center_c)]
        line = [(r, center_c) for r in range(r0 + 1, r1)]
    return seeds, line


def _can_place_pair(
    shape: tuple[int, int],
    field_rows: tuple[int, int],
    reserve: np.ndarray,
    pattern: np.ndarray,
    center_r: int,
    center_c: int,
    gap: int,
    orientation: str,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]] | None:
    pattern_cells = _pattern_cells_at_center(pattern, center_r, center_c)
    seeds, line = _pair_seed_cells(center_r, center_c, gap, orientation)
    if not _in_bounds(pattern_cells + seeds + line, shape):
        return None

    lo_row, hi_row = field_rows
    if any(not (lo_row <= r <= hi_row) for r, _ in pattern_cells + seeds + line):
        return None
    if any(reserve[r, c] for r, c in pattern_cells):
        return None
    if any(reserve[r, c] for r, c in seeds):
        return None
    if any(reserve[r, c] for r, c in line):
        return None
    if any(cell in pattern_cells for cell in seeds):
        return None
    return pattern_cells, seeds, line


def _make_orientation_list(count: int) -> list[str]:
    choices = [random.choice(("h", "v")) for _ in range(count)]
    if count >= 2 and len(set(choices)) == 1:
        choices[random.randrange(count)] = "v" if choices[0] == "h" else "h"
    return choices


def _expected_output_matches(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        actual = main(inp)
    except Exception:
        return False
    return bool(np.array_equal(actual, expected))


def _generate_attempt() -> np.ndarray | None:
    patterns = _choose_patterns()
    band_h = max(int(pattern.shape[0]) for pattern in patterns)
    sep = random.randint(1, 2)

    min_h = max(12, band_h + sep + 7)
    height = random.randint(min_h, 20)

    widths = [int(pattern.shape[1]) for pattern in patterns]
    dict_w_max = sum(widths) + 2 * (len(patterns) - 1)
    min_w = max(16, dict_w_max + 4)
    width = random.randint(min_w, 28)

    inp = np.zeros((height, width), dtype=int)
    expected = np.zeros((height, width), dtype=int)

    side = random.choice(("top", "bottom"))
    _band_h, _dict_right = _place_dictionary(inp, expected, patterns, side, GREEN)

    if side == "top":
        field_rows = (band_h + sep, height - 1)
    else:
        field_rows = (0, height - band_h - sep - 1)
    if field_rows[1] - field_rows[0] + 1 < 6:
        return None

    pair_count = random.randint(3, 5)
    colors = random.sample(SEED_COLORS, pair_count)
    orientations = _make_orientation_list(pair_count)
    gaps = [random.randint(1, len(patterns)) for _ in range(pair_count)]

    specs = [
        {
            "color": color,
            "gap": gap,
            "orientation": orientation,
            "pattern": patterns[gap - 1],
        }
        for color, gap, orientation in zip(colors, gaps, orientations)
    ]
    specs.sort(key=lambda item: (int(item["pattern"].sum()), int(item["gap"])), reverse=True)

    input_occ = inp != 0
    output_occ = expected != 0
    for spec in specs:
        placed = False
        reserve = _reserve_mask(input_occ, output_occ)
        pat = np.asarray(spec["pattern"], dtype=bool)
        for _ in range(PLACE_RETRIES):
            center_r = random.randint(field_rows[0], field_rows[1])
            center_c = random.randint(1, width - 2)
            candidate = _can_place_pair(
                inp.shape,
                field_rows,
                reserve,
                pat,
                center_r,
                center_c,
                int(spec["gap"]),
                str(spec["orientation"]),
            )
            if candidate is None:
                continue
            pattern_cells, seeds, _line = candidate

            for r, c in seeds:
                inp[r, c] = int(spec["color"])
                input_occ[r, c] = True
            for r, c in pattern_cells:
                expected[r, c] = int(spec["color"])
                output_occ[r, c] = True
            placed = True
            break
        if not placed:
            return None

    if not _expected_output_matches(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
