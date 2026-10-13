from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GENERATION_ATTEMPTS = 400
MAX_SIDE = 30
PAIR_COUNT_OPTIONS = (3, 4)
PAIR_CASES = ("solid", "dashed", "dotted")
PAIR_ORIENTATIONS = ("h", "v")
GAP_RANGE = (4, 5)
PAD_RANGE = (1, 2)
PACK_GAP_RANGE = (1, 2)
COLOR_POOL = tuple(range(10))


def _mask_from_rows(rows: tuple[str, ...]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


SHAPE_LIBRARY = (
    _mask_from_rows(("1",)),
    _mask_from_rows(("11",)),
    _mask_from_rows(("111",)),
    _mask_from_rows(("10", "11")),
    _mask_from_rows(("100", "111")),
    _mask_from_rows(("110", "011")),
    _mask_from_rows(("111", "010")),
    _mask_from_rows(("11", "11")),
    _mask_from_rows(("101", "111")),
)


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    order = np.argsort(-counts)
    values = values[order]
    counts = counts[order]
    assert len(values) >= 1, "grid must be nonempty"
    if len(counts) > 1:
        assert int(counts[0]) > int(counts[1]), "background must be uniquely dominant"
    return int(values[0])


def _mask_key(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(map(int, mask.shape)) + tuple(mask.astype(np.uint8).ravel())


def _all_transforms(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    for k in range(4):
        rot = np.rot90(mask, k)
        variants.append(rot)
        variants.append(np.fliplr(rot))
    return variants


def _unique_transforms(mask: np.ndarray) -> list[np.ndarray]:
    unique: dict[tuple[int, ...], np.ndarray] = {}
    for variant in _all_transforms(mask):
        unique.setdefault(_mask_key(variant), variant.copy())
    return list(unique.values())


def _canonical_key(mask: np.ndarray) -> tuple[int, ...]:
    return min(_mask_key(variant) for variant in _all_transforms(mask))


def _extract_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    seen = np.zeros((height, width), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(height):
        for c in range(width):
            if seen[r, c] or int(g[r, c]) == bg:
                continue

            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < height and 0 <= nc < width):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
            mask[arr[:, 0] - int(r0), arr[:, 1] - int(c0)] = True

            rows: dict[int, list[int]] = defaultdict(list)
            cols: dict[int, list[int]] = defaultdict(list)
            for rr, cc in coords:
                rows[int(rr)].append(int(cc))
                cols[int(cc)].append(int(rr))
            for seq in rows.values():
                seq.sort()
            for seq in cols.values():
                seq.sort()

            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "mask": mask,
                    "rows": dict(rows),
                    "cols": dict(cols),
                }
            )

    return components


def _line_style(mask_a: np.ndarray, mask_b: np.ndarray) -> str:
    if mask_a.shape == mask_b.shape and np.array_equal(mask_a, mask_b):
        return "solid"
    for variant in _all_transforms(mask_a):
        if variant.shape == mask_b.shape and np.array_equal(variant, mask_b):
            return "dashed"
    return "dotted"


def _find_connection(
    grid: np.ndarray, a: dict[str, object], b: dict[str, object], bg: int
) -> dict[str, int | str]:
    candidates: list[dict[str, int | str]] = []
    rows_a = a["rows"]
    rows_b = b["rows"]
    cols_a = a["cols"]
    cols_b = b["cols"]

    for row in sorted(set(rows_a) & set(rows_b)):
        cols1 = rows_a[row]
        cols2 = rows_b[row]
        a_left, a_right = cols1[0], cols1[-1]
        b_left, b_right = cols2[0], cols2[-1]
        if a_right < b_left:
            lo, hi = int(a_right), int(b_left)
        elif b_right < a_left:
            lo, hi = int(b_right), int(a_left)
        else:
            continue
        if hi - lo <= 1:
            continue
        if np.all(grid[int(row), lo + 1 : hi] == bg):
            candidates.append(
                {"orient": "h", "fixed": int(row), "lo": lo, "hi": hi, "gap": hi - lo - 1}
            )

    for col in sorted(set(cols_a) & set(cols_b)):
        rows1 = cols_a[col]
        rows2 = cols_b[col]
        a_top, a_bottom = rows1[0], rows1[-1]
        b_top, b_bottom = rows2[0], rows2[-1]
        if a_bottom < b_top:
            lo, hi = int(a_bottom), int(b_top)
        elif b_bottom < a_top:
            lo, hi = int(b_bottom), int(a_top)
        else:
            continue
        if hi - lo <= 1:
            continue
        if np.all(grid[lo + 1 : hi, int(col)] == bg):
            candidates.append(
                {"orient": "v", "fixed": int(col), "lo": lo, "hi": hi, "gap": hi - lo - 1}
            )

    assert candidates, "each color pair must have an unobstructed row or column connection"
    candidates.sort(
        key=lambda item: (
            int(item["gap"]),
            0 if item["orient"] == "h" else 1,
            int(item["fixed"]),
            int(item["lo"]),
            int(item["hi"]),
        )
    )
    return candidates[0]


def _draw_connection(out: np.ndarray, connection: dict[str, int | str], color: int, style: str) -> None:
    stride = {"solid": 1, "dashed": 2, "dotted": 3}[style]
    lo = int(connection["lo"])
    hi = int(connection["hi"])
    fixed = int(connection["fixed"])

    if connection["orient"] == "h":
        for offset, col in enumerate(range(lo + 1, hi)):
            if offset % stride == 0:
                out[fixed, col] = int(color)
    else:
        for offset, row in enumerate(range(lo + 1, hi)):
            if offset % stride == 0:
                out[row, fixed] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-color object pair with a line whose pattern encodes shape similarity.

    Rule
    - Exact normalized-mask match: draw a solid line.
    - Match only after rotation and/or reflection: draw a dashed line.
    - Otherwise: draw a dotted line.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    bg = _get_bg(g)
    components = _extract_components(g, bg)
    assert components, "expected foreground objects"

    by_color: dict[int, list[dict[str, object]]] = defaultdict(list)
    for comp in components:
        by_color[int(comp["color"])].append(comp)

    out = g.copy()
    for color, pair in by_color.items():
        assert len(pair) == 2, "each foreground color must appear in exactly two objects"
        first, second = pair
        style = _line_style(first["mask"], second["mask"])
        connection = _find_connection(g, first, second, bg)
        _draw_connection(out, connection, int(color), style)

    return out


def _random_variant(mask: np.ndarray) -> np.ndarray:
    return random.choice(_unique_transforms(mask)).copy()


def _sample_masks(case: str) -> tuple[np.ndarray, np.ndarray]:
    if case == "solid":
        base = random.choice(SHAPE_LIBRARY)
        mask = _random_variant(base)
        return mask.copy(), mask.copy()

    if case == "dashed":
        for _ in range(100):
            base = random.choice(SHAPE_LIBRARY)
            variants = _unique_transforms(base)
            if len(variants) < 2:
                continue
            first, second = random.sample(variants, 2)
            if not np.array_equal(first, second):
                return first.copy(), second.copy()
        raise RuntimeError("failed to sample a nontrivial transformed pair")

    assert case == "dotted"
    for _ in range(200):
        first = _random_variant(random.choice(SHAPE_LIBRARY))
        second = _random_variant(random.choice(SHAPE_LIBRARY))
        if _canonical_key(first) != _canonical_key(second):
            return first.copy(), second.copy()
    raise RuntimeError("failed to sample a genuinely different pair")


def _edge_cells(mask: np.ndarray, side: str) -> list[tuple[int, int]]:
    height, width = map(int, mask.shape)
    cells: list[tuple[int, int]] = []
    for r in range(height):
        for c in range(width):
            if not bool(mask[r, c]):
                continue
            if side == "left" and (c == 0 or not bool(mask[r, c - 1])):
                cells.append((r, c))
            elif side == "right" and (c == width - 1 or not bool(mask[r, c + 1])):
                cells.append((r, c))
            elif side == "top" and (r == 0 or not bool(mask[r - 1, c])):
                cells.append((r, c))
            elif side == "bottom" and (r == height - 1 or not bool(mask[r + 1, c])):
                cells.append((r, c))
    assert cells, "each shape must expose at least one edge cell on every used side"
    return cells


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = map(int, mask.shape)
    region = grid[top : top + h, left : left + w]
    assert region.shape == mask.shape
    assert np.all(region[mask] == grid[0, 0]), "paint target must be background"
    region[mask] = int(color)


def _make_pair_zone(bg: int, color: int, case: str) -> np.ndarray:
    mask_a, mask_b = _sample_masks(case)
    orient = random.choice(PAIR_ORIENTATIONS)
    pad_top = random.randint(*PAD_RANGE)
    pad_bottom = random.randint(*PAD_RANGE)
    pad_left = random.randint(*PAD_RANGE)
    pad_right = random.randint(*PAD_RANGE)
    gap = random.randint(*GAP_RANGE)

    ha, wa = map(int, mask_a.shape)
    hb, wb = map(int, mask_b.shape)

    if orient == "h":
        exit_a_r, exit_a_c = random.choice(_edge_cells(mask_a, "right"))
        exit_b_r, exit_b_c = random.choice(_edge_cells(mask_b, "left"))
        anchor_row = pad_top + max(exit_a_r, exit_b_r)
        top_a = anchor_row - exit_a_r
        top_b = anchor_row - exit_b_r
        left_a = pad_left
        exit_a_abs = left_a + exit_a_c
        exit_b_abs = exit_a_abs + gap + 1
        left_b = exit_b_abs - exit_b_c
        zone_h = max(top_a + ha, top_b + hb) + pad_bottom
        zone_w = max(left_a + wa, left_b + wb) + pad_right
    else:
        exit_a_r, exit_a_c = random.choice(_edge_cells(mask_a, "bottom"))
        exit_b_r, exit_b_c = random.choice(_edge_cells(mask_b, "top"))
        anchor_col = pad_left + max(exit_a_c, exit_b_c)
        left_a = anchor_col - exit_a_c
        left_b = anchor_col - exit_b_c
        top_a = pad_top
        exit_a_abs = top_a + exit_a_r
        exit_b_abs = exit_a_abs + gap + 1
        top_b = exit_b_abs - exit_b_r
        zone_h = max(top_a + ha, top_b + hb) + pad_bottom
        zone_w = max(left_a + wa, left_b + wb) + pad_right

    zone = np.full((zone_h, zone_w), bg, dtype=int)
    _paint_mask(zone, mask_a, top_a, left_a, color)
    _paint_mask(zone, mask_b, top_b, left_b, color)
    return zone


def _pair_cases(pair_count: int) -> list[str]:
    cases = ["solid", "dashed", "dotted"]
    if pair_count == 4:
        cases.append(random.choice(PAIR_CASES))
    random.shuffle(cases)
    return cases


def _pack_layout(zones: list[np.ndarray], bg: int, columns: int) -> np.ndarray | None:
    rows = (len(zones) + columns - 1) // columns
    slots: list[list[np.ndarray | None]] = []
    iterator = iter(zones)
    for _ in range(rows):
        row_slots: list[np.ndarray | None] = []
        for _ in range(columns):
            row_slots.append(next(iterator, None))
        slots.append(row_slots)

    row_heights = [max((int(zone.shape[0]) for zone in row if zone is not None), default=0) for row in slots]
    col_widths = []
    for col in range(columns):
        col_widths.append(
            max((int(row[col].shape[1]) for row in slots if row[col] is not None), default=0)
        )

    top_margin = random.randint(*PAD_RANGE)
    bottom_margin = random.randint(*PAD_RANGE)
    left_margin = random.randint(*PAD_RANGE)
    right_margin = random.randint(*PAD_RANGE)
    row_gaps = [random.randint(*PACK_GAP_RANGE) for _ in range(max(0, rows - 1))]
    col_gaps = [random.randint(*PACK_GAP_RANGE) for _ in range(max(0, columns - 1))]

    total_h = top_margin + bottom_margin + sum(row_heights) + sum(row_gaps)
    total_w = left_margin + right_margin + sum(col_widths) + sum(col_gaps)
    if total_h > MAX_SIDE or total_w > MAX_SIDE:
        return None

    grid = np.full((total_h, total_w), bg, dtype=int)
    y = top_margin
    for row_idx, row in enumerate(slots):
        x = left_margin
        for col_idx, zone in enumerate(row):
            slot_h = row_heights[row_idx]
            slot_w = col_widths[col_idx]
            if zone is not None:
                max_dy = slot_h - zone.shape[0]
                max_dx = slot_w - zone.shape[1]
                dy = random.randint(0, int(max_dy)) if max_dy > 0 else 0
                dx = random.randint(0, int(max_dx)) if max_dx > 0 else 0
                region = grid[y + dy : y + dy + zone.shape[0], x + dx : x + dx + zone.shape[1]]
                region[:, :] = zone
            x += slot_w
            if col_idx < len(col_gaps):
                x += col_gaps[col_idx]
        y += row_heights[row_idx]
        if row_idx < len(row_gaps):
            y += row_gaps[row_idx]

    return grid


def _pack_zones(zones: list[np.ndarray], bg: int) -> np.ndarray | None:
    options = [1, 2] if len(zones) >= 3 else [1]
    random.shuffle(options)
    for columns in options:
        packed = _pack_layout(zones, bg, columns)
        if packed is not None:
            return packed
    return None


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choice(PAIR_COUNT_OPTIONS)
    palette = list(COLOR_POOL)
    bg = random.choice(palette)
    palette.remove(bg)
    colors = random.sample(palette, pair_count)

    zones: list[np.ndarray] = []
    for color, case in zip(colors, _pair_cases(pair_count)):
        zones.append(_make_pair_zone(bg, int(color), case))
    random.shuffle(zones)

    grid = _pack_zones(zones, bg)
    if grid is None:
        return None

    values, counts = np.unique(grid, return_counts=True)
    if len(counts) > 1:
        top_two = np.sort(counts)[-2:]
        if int(top_two[-1]) == int(top_two[-2]):
            return None
    if _get_bg(grid) != bg:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(grid, out):
        return None
    if len(_extract_components(grid, bg)) != 2 * pair_count:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid connection-pattern input")
