from __future__ import annotations

import random

import numpy as np


BLACK = 0
MAROON = 9
PALETTE = (1, 2, 3, 4, 5, 6, 7, 8)

KEEP_TO_BAR = {
    "left": "right",
    "right": "left",
    "top": "bottom",
    "bottom": "top",
}
BAR_TO_KEEP = {bar: keep for keep, bar in KEEP_TO_BAR.items()}
H_KEEP_SIDES = ("left", "right")
V_KEEP_SIDES = ("top", "bottom")

GRID_CHOICES = ((3, 3), (3, 4), (4, 3), (4, 4))
CELL_SIZE_RANGE = (5, 8)
GEN_RETRIES = 400


def _separator_color(grid: np.ndarray) -> tuple[int, list[int], list[int]]:
    h, w = map(int, grid.shape)
    candidates = []
    for color_raw in np.unique(grid):
        color = int(color_raw)
        rows = [r for r in range(h) if np.all(grid[r, :] == color)]
        cols = [c for c in range(w) if np.all(grid[:, c] == color)]
        if rows and cols:
            candidates.append((color, rows, cols))
    assert len(candidates) == 1, "expected exactly one separator color"
    return candidates[0]


def _bands(length: int, separators: list[int]) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start = 0
    for sep in sorted(set(int(v) for v in separators)):
        if start < sep:
            bands.append((start, sep - 1))
        start = sep + 1
    if start < length:
        bands.append((start, length - 1))
    return bands


def _control_info(sub: np.ndarray, color: int) -> dict[str, int] | None:
    coords = np.argwhere(sub == int(color))
    assert coords.size > 0, "region must contain a colored object"

    rows = coords[:, 0]
    cols = coords[:, 1]
    h, w = map(int, sub.shape)

    if np.all(rows == rows[0]):
        row = int(rows[0])
        cols_sorted = np.sort(cols)
        if row in (0, h - 1) and np.array_equal(
            cols_sorted, np.arange(int(cols_sorted[0]), int(cols_sorted[-1]) + 1)
        ):
            if len(cols_sorted) == 1 and int(cols_sorted[0]) in (0, w - 1):
                return None
            return {
                "edge": "top" if row == 0 else "bottom",
                "length": int(len(cols_sorted)),
            }

    if np.all(cols == cols[0]):
        col = int(cols[0])
        rows_sorted = np.sort(rows)
        if col in (0, w - 1) and np.array_equal(
            rows_sorted, np.arange(int(rows_sorted[0]), int(rows_sorted[-1]) + 1)
        ):
            if len(rows_sorted) == 1 and int(rows_sorted[0]) in (0, h - 1):
                return None
            return {
                "edge": "left" if col == 0 else "right",
                "length": int(len(rows_sorted)),
            }

    return None


def _shift_mask(mask: np.ndarray, dr: int, dc: int) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return out
    for r_raw, c_raw in coords:
        r = int(r_raw) + int(dr)
        c = int(c_raw) + int(dc)
        out[r, c] = True
    return out


def _clip_and_shift(mask: np.ndarray, keep_side: str, length: int) -> np.ndarray:
    h, w = map(int, mask.shape)
    out = mask.copy()

    if keep_side == "left":
        assert 1 <= int(length) <= w
        out[:, int(length) :] = False
        if not out.any():
            return out
        min_col = int(np.argwhere(out)[:, 1].min())
        return _shift_mask(out, 0, -min_col)

    if keep_side == "right":
        assert 1 <= int(length) <= w
        out[:, : w - int(length)] = False
        if not out.any():
            return out
        max_col = int(np.argwhere(out)[:, 1].max())
        return _shift_mask(out, 0, w - 1 - max_col)

    if keep_side == "top":
        assert 1 <= int(length) <= h
        out[int(length) :, :] = False
        if not out.any():
            return out
        min_row = int(np.argwhere(out)[:, 0].min())
        return _shift_mask(out, -min_row, 0)

    assert keep_side == "bottom"
    assert 1 <= int(length) <= h
    out[: h - int(length), :] = False
    if not out.any():
        return out
    max_row = int(np.argwhere(out)[:, 0].max())
    return _shift_mask(out, h - 1 - max_row, 0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Clip each template region by same-colored neighboring control bars, then slide the
    surviving fragment to the indicated region edge.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    separator, sep_rows, sep_cols = _separator_color(g)
    assert int(separator) == MAROON, "expected the maroon separator color"
    row_bands = _bands(g.shape[0], sep_rows)
    col_bands = _bands(g.shape[1], sep_cols)
    assert row_bands and col_bands, "expected at least one partition cell"

    regions: dict[tuple[int, int], dict[str, object]] = {}
    for ri, (r0, r1) in enumerate(row_bands):
        for ci, (c0, c1) in enumerate(col_bands):
            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            colors = [int(v) for v in np.unique(sub) if int(v) not in (BLACK, separator)]
            assert len(colors) == 1, "each region must contain exactly one non-background color"
            color = colors[0]
            info = _control_info(sub, color)
            if info is not None:
                regions[(ri, ci)] = {
                    "kind": "control",
                    "color": color,
                    "edge": str(info["edge"]),
                    "length": int(info["length"]),
                    "bbox": (r0, c0, r1, c1),
                }
            else:
                regions[(ri, ci)] = {
                    "kind": "template",
                    "color": color,
                    "sub": sub.copy(),
                    "bbox": (r0, c0, r1, c1),
                }

    out = g.copy()
    for (ri, ci), region in regions.items():
        if region["kind"] != "template":
            continue

        controls: list[tuple[str, int]] = []
        for nr, nc in ((ri - 1, ci), (ri + 1, ci), (ri, ci - 1), (ri, ci + 1)):
            neighbor = regions.get((nr, nc))
            if neighbor is None or neighbor["kind"] != "control":
                continue
            if int(neighbor["color"]) != int(region["color"]):
                continue
            keep_side = BAR_TO_KEEP[str(neighbor["edge"])]
            controls.append((keep_side, int(neighbor["length"])))

        if not controls:
            continue

        sub = np.asarray(region["sub"], dtype=int)
        mask = sub == int(region["color"])
        horizontal = [item for item in controls if item[0] in H_KEEP_SIDES]
        vertical = [item for item in controls if item[0] in V_KEEP_SIDES]

        for keep_side, length in sorted(horizontal, key=lambda item: H_KEEP_SIDES.index(item[0])):
            mask = _clip_and_shift(mask, keep_side, length)
        for keep_side, length in sorted(vertical, key=lambda item: V_KEEP_SIDES.index(item[0])):
            mask = _clip_and_shift(mask, keep_side, length)

        new_sub = np.full(sub.shape, BLACK, dtype=int)
        new_sub[mask] = int(region["color"])
        r0, c0, r1, c1 = region["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = new_sub

    return out


def _separator_positions(count: int, cell_size: int) -> list[int]:
    return [i * (int(cell_size) + 1) for i in range(int(count) + 1)]


def _cell_spans(count: int, cell_size: int) -> list[tuple[int, int]]:
    seps = _separator_positions(count, cell_size)
    return [(seps[i] + 1, seps[i + 1] - 1) for i in range(count)]


def _template_mask(cell_size: int, rng: random.Random) -> np.ndarray | None:
    min_dim = 3
    max_dim = int(cell_size) - 2
    if max_dim < min_dim:
        return None

    for _ in range(50):
        shape_h = rng.randint(min_dim, max_dim)
        shape_w = rng.randint(min_dim, max_dim)
        top = rng.randint(1, int(cell_size) - shape_h - 1)
        left = rng.randint(1, int(cell_size) - shape_w - 1)

        shape = np.ones((shape_h, shape_w), dtype=bool)
        if rng.random() < 0.65:
            notch_h = rng.randint(1, shape_h - 2)
            notch_w = rng.randint(1, shape_w - 2)
            corner = rng.choice(("tl", "tr", "bl", "br"))
            if corner == "tl":
                shape[:notch_h, :notch_w] = False
            elif corner == "tr":
                shape[:notch_h, shape_w - notch_w :] = False
            elif corner == "bl":
                shape[shape_h - notch_h :, :notch_w] = False
            else:
                shape[shape_h - notch_h :, shape_w - notch_w :] = False

        mask = np.zeros((cell_size, cell_size), dtype=bool)
        mask[top : top + shape_h, left : left + shape_w] = shape
        coords = np.argwhere(mask)
        if coords.size == 0:
            continue
        rows = coords[:, 0]
        cols = coords[:, 1]
        if len(np.unique(rows)) < 2 or len(np.unique(cols)) < 2:
            continue
        if rows.min() == 0 or rows.max() == cell_size - 1 or cols.min() == 0 or cols.max() == cell_size - 1:
            continue
        if _control_info(mask.astype(int), 1) is not None:
            continue
        return mask
    return None


def _ranges_for_keep(mask: np.ndarray) -> dict[str, tuple[int, int]]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a nonempty template mask"
    h, w = map(int, mask.shape)
    min_row = int(coords[:, 0].min())
    max_row = int(coords[:, 0].max())
    min_col = int(coords[:, 1].min())
    max_col = int(coords[:, 1].max())

    out: dict[str, tuple[int, int]] = {}
    left_range = (min_col + 1, max_col)
    right_range = (w - max_col, w - min_col - 1)
    top_range = (min_row + 1, max_row)
    bottom_range = (h - max_row, h - min_row - 1)

    for side, bounds in (
        ("left", left_range),
        ("right", right_range),
        ("top", top_range),
        ("bottom", bottom_range),
    ):
        lo, hi = bounds
        if lo <= hi:
            out[side] = (int(lo), int(hi))
    return out


def _plan_controls(mask: np.ndarray, count: int, rng: random.Random) -> list[tuple[str, int]] | None:
    if count == 0:
        return []

    ranges = _ranges_for_keep(mask)
    if count == 1:
        axes: list[str] = []
        if any(side in ranges for side in H_KEEP_SIDES):
            axes.append("h")
        if any(side in ranges for side in V_KEEP_SIDES):
            axes.append("v")
        if not axes:
            return None
        axis = rng.choice(axes)
        side_pool = H_KEEP_SIDES if axis == "h" else V_KEEP_SIDES
        side = rng.choice([name for name in side_pool if name in ranges])
        lo, hi = ranges[side]
        return [(side, rng.randint(lo, hi))]

    assert count == 2
    h_sides = [side for side in H_KEEP_SIDES if side in ranges]
    if not h_sides:
        return None

    for _ in range(80):
        h_side = rng.choice(h_sides)
        h_lo, h_hi = ranges[h_side]
        h_len = rng.randint(h_lo, h_hi)
        clipped = _clip_and_shift(mask, h_side, h_len)
        if not clipped.any():
            continue
        next_ranges = _ranges_for_keep(clipped)
        v_sides = [side for side in V_KEEP_SIDES if side in next_ranges]
        if not v_sides:
            continue
        v_side = rng.choice(v_sides)
        v_lo, v_hi = next_ranges[v_side]
        v_len = rng.randint(v_lo, v_hi)
        return [(h_side, h_len), (v_side, v_len)]
    return None


def _sample_counts(
    template_keys: list[tuple[int, int]],
    degrees: dict[tuple[int, int], int],
    control_count: int,
    rng: random.Random,
) -> dict[tuple[int, int], int] | None:
    template_total = len(template_keys)
    max_capacity = sum(min(2, int(degrees[key])) for key in template_keys)
    if int(control_count) > max_capacity:
        return None

    for _ in range(80):
        counts = {key: 0 for key in template_keys}
        if int(control_count) <= template_total:
            for key in rng.sample(template_keys, control_count):
                counts[key] = 1
        else:
            for key in template_keys:
                counts[key] = 1
            extra = int(control_count) - template_total
            twice_choices = [key for key in template_keys if degrees[key] >= 2]
            if len(twice_choices) < extra:
                continue
            for key in rng.sample(twice_choices, extra):
                counts[key] = 2

        one_keys = [key for key in template_keys if counts[key] == 1 and degrees[key] >= 2]
        max_merges = len(one_keys) // 2
        merge_total = rng.randint(0, max_merges)
        for _merge in range(merge_total):
            hi_choices = [key for key in template_keys if counts[key] == 1 and degrees[key] >= 2]
            if len(hi_choices) < 2:
                break
            hi = rng.choice(hi_choices)
            lo_choices = [key for key in hi_choices if key != hi]
            if not lo_choices:
                break
            lo = rng.choice(lo_choices)
            counts[hi] = 2
            counts[lo] = 0

        if sum(counts.values()) != int(control_count):
            continue
        if any(counts[key] > min(2, int(degrees[key])) for key in template_keys):
            continue
        return counts
    return None


def _assign_controls(
    control_keys: list[tuple[int, int]],
    neighbors: dict[tuple[int, int], list[tuple[int, int]]],
    counts: dict[tuple[int, int], int],
    rng: random.Random,
) -> dict[tuple[int, int], tuple[int, int]] | None:
    order = sorted(control_keys, key=lambda key: len(neighbors[key]))
    remaining = {key: int(value) for key, value in counts.items()}
    assignment: dict[tuple[int, int], tuple[int, int]] = {}

    def feasible(start_idx: int) -> bool:
        future = order[start_idx:]
        for template_key, need in remaining.items():
            if need < 0:
                return False
            available = sum(1 for control_key in future if template_key in neighbors[control_key])
            if need > available:
                return False
        return True

    def dfs(idx: int) -> bool:
        if idx == len(order):
            return all(value == 0 for value in remaining.values())

        control_key = order[idx]
        options = [key for key in neighbors[control_key] if remaining[key] > 0]
        rng.shuffle(options)
        for template_key in options:
            remaining[template_key] -= 1
            assignment[control_key] = template_key
            if feasible(idx + 1) and dfs(idx + 1):
                return True
            remaining[template_key] += 1
            del assignment[control_key]
        return False

    if not feasible(0):
        return None
    if dfs(0):
        return assignment
    return None


def _draw_bar(sub: np.ndarray, color: int, edge: str, length: int, rng: random.Random) -> None:
    size = int(sub.shape[0])
    assert sub.shape == (size, size), "generator uses square regions"
    assert 1 <= int(length) <= size

    if edge in ("top", "bottom"):
        start = rng.randint(0, size - int(length))
        row = 0 if edge == "top" else size - 1
        sub[row, start : start + int(length)] = int(color)
        return

    start = rng.randint(0, size - int(length))
    col = 0 if edge == "left" else size - 1
    sub[start : start + int(length), col] = int(color)


def _make_base_grid(rows: int, cols: int, cell_size: int) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    height = int(rows) * (int(cell_size) + 1) + 1
    width = int(cols) * (int(cell_size) + 1) + 1
    grid = np.full((height, width), MAROON, dtype=int)

    row_spans = _cell_spans(rows, cell_size)
    col_spans = _cell_spans(cols, cell_size)
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            grid[r0 : r1 + 1, c0 : c1 + 1] = BLACK
    return grid, row_spans, col_spans


def _cell_neighbors(key: tuple[int, int], rows: int, cols: int) -> list[tuple[int, int]]:
    r, c = key
    out = []
    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if 0 <= nr < rows and 0 <= nc < cols:
            out.append((nr, nc))
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    rows, cols = rng.choice(GRID_CHOICES)
    cell_size = rng.randint(*CELL_SIZE_RANGE)
    grid, row_spans, col_spans = _make_base_grid(rows, cols, cell_size)

    template_parity = rng.randrange(2)
    template_keys = [(r, c) for r in range(rows) for c in range(cols) if (r + c) % 2 == template_parity]
    control_keys = [(r, c) for r in range(rows) for c in range(cols) if (r + c) % 2 != template_parity]

    if len(template_keys) > len(PALETTE):
        return None

    template_colors = rng.sample(PALETTE, len(template_keys))
    templates: dict[tuple[int, int], dict[str, object]] = {}
    for key, color in zip(template_keys, template_colors):
        mask = _template_mask(cell_size, rng)
        if mask is None:
            return None
        templates[key] = {"color": int(color), "mask": mask}

    neighbors = {key: [n for n in _cell_neighbors(key, rows, cols) if n in templates] for key in control_keys}
    if any(not choices for choices in neighbors.values()):
        return None

    degrees = {
        key: sum(1 for control_key in control_keys if key in neighbors[control_key])
        for key in template_keys
    }
    counts = _sample_counts(template_keys, degrees, len(control_keys), rng)
    if counts is None:
        return None

    assignment = _assign_controls(control_keys, neighbors, counts, rng)
    if assignment is None:
        return None

    controls_by_template: dict[tuple[int, int], list[tuple[int, int]]] = {key: [] for key in template_keys}
    for control_key, template_key in assignment.items():
        controls_by_template[template_key].append(control_key)
    if any(len(controls_by_template[key]) != counts[key] for key in template_keys):
        return None

    control_specs: dict[tuple[int, int], list[tuple[str, int]]] = {}
    for template_key in template_keys:
        spec = _plan_controls(np.asarray(templates[template_key]["mask"], dtype=bool), counts[template_key], rng)
        if spec is None:
            return None
        control_specs[template_key] = spec

    for (r, c), info in templates.items():
        r0, r1 = row_spans[r]
        c0, c1 = col_spans[c]
        patch = np.full((cell_size, cell_size), BLACK, dtype=int)
        patch[np.asarray(info["mask"], dtype=bool)] = int(info["color"])
        grid[r0 : r1 + 1, c0 : c1 + 1] = patch

    for template_key, control_list in controls_by_template.items():
        specs = control_specs[template_key][:]
        rng.shuffle(specs)
        if len(specs) != len(control_list):
            return None
        color = int(templates[template_key]["color"])
        for control_key, (keep_side, length) in zip(control_list, specs):
            r, c = control_key
            r0, r1 = row_spans[r]
            c0, c1 = col_spans[c]
            patch = np.full((cell_size, cell_size), BLACK, dtype=int)
            _draw_bar(patch, color, KEEP_TO_BAR[keep_side], length, rng)
            grid[r0 : r1 + 1, c0 : c1 + 1] = patch

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
