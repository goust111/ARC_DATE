import random
from collections import deque

import numpy as np


BG = 0
FILTER = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 400


TEMPLATES = {
    1: [
        np.array([[1]], dtype=bool),
        np.array([[1], [1]], dtype=bool),
        np.array([[1], [1], [1]], dtype=bool),
    ],
    2: [
        np.array([[1, 1]], dtype=bool),
        np.array([[1, 0], [1, 1]], dtype=bool),
        np.array([[0, 1], [1, 1]], dtype=bool),
        np.array([[1, 0], [1, 1], [0, 1]], dtype=bool),
    ],
    3: [
        np.array([[1, 1, 1]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
        np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
        np.array([[1, 1, 1], [0, 0, 1]], dtype=bool),
    ],
}


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _zero_runs(row: np.ndarray) -> list[tuple[int, int]]:
    runs = []
    start = None
    for c, value in enumerate(row):
        if int(value) == BG:
            if start is None:
                start = c
        elif start is not None:
            runs.append((start, c - start))
            start = None
    if start is not None:
        runs.append((start, len(row) - start))
    return runs


def _find_filters(grid: np.ndarray) -> list[dict]:
    filters = []
    for r in range(grid.shape[0]):
        row = grid[r]
        values = set(map(int, row))
        if values - {BG, FILTER}:
            continue
        runs = _zero_runs(row)
        if len(runs) != 1:
            continue
        gap_start, gap_width = runs[0]
        if not (1 <= gap_width <= 3):
            continue
        if int(np.count_nonzero(row == FILTER)) + gap_width != grid.shape[1]:
            continue
        filters.append(
            {
                "row": r,
                "gap_start": gap_start,
                "gap_width": gap_width,
            }
        )
    return filters


def _extract_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in (BG, FILTER) or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            rel = sorted((rr - r0, cc - c0) for rr, cc in coords)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "rel": rel,
                    "r0": r0,
                    "r1": r1,
                    "c0": c0,
                    "c1": c1,
                    "height": r1 - r0 + 1,
                    "width": c1 - c0 + 1,
                    "area": len(coords),
                }
            )
    comps.sort(key=lambda comp: (-comp["r1"], -comp["r0"], comp["c0"], comp["color"]))
    return comps


def _place_component(canvas: np.ndarray, comp: dict, r0: int) -> None:
    for dr, dc in comp["rel"]:
        canvas[r0 + dr, comp["c0"] + dc] = comp["color"]


def _settle_component(comp: dict, canvas: np.ndarray, filter_widths: dict[int, int]) -> int:
    h, _w = canvas.shape
    r0 = int(comp["r0"])
    while True:
        hit_floor = False
        line_rows = set()
        blocked_by_color = False
        for dr, dc in comp["rel"]:
            nr = r0 + dr + 1
            nc = comp["c0"] + dc
            if nr >= h:
                hit_floor = True
                break
            value = int(canvas[nr, nc])
            if value == BG:
                continue
            if value == FILTER and nr in filter_widths:
                line_rows.add(nr)
            else:
                blocked_by_color = True
                break
        if hit_floor or blocked_by_color:
            return r0
        if not line_rows:
            r0 += 1
            continue
        if len(line_rows) != 1:
            return r0
        line_row = next(iter(line_rows))
        if comp["width"] > int(filter_widths[line_row]):
            return r0

        jump_r0 = line_row + 1
        if jump_r0 + comp["height"] - 1 >= h:
            return r0
        blocked_jump = False
        for dr, dc in comp["rel"]:
            nr = jump_r0 + dr
            nc = comp["c0"] + dc
            if int(canvas[nr, nc]) != BG:
                blocked_jump = True
                break
        if blocked_jump:
            return r0
        r0 = jump_r0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Let each colored 4-connected block fall through gray filter lines only when its width is small enough.

    Input structure:
    - The background is black, and 2-3 horizontal gray rows act as filter lines.
    - Each gray filter row has exactly one contiguous black gap whose width encodes that line's pass threshold.
    - Several small non-black, non-gray monochrome blocks start above the upper filters.

    Rule:
    - Detect every horizontal gray filter row and measure its gap width.
    - Split the colored pixels into 4-connected components and compute each component's bounding-box width.
    - Process components from lower original position to higher original position, moving each one downward as a rigid translation.
    - When a component is about to enter a filter row, it passes to below that row iff its width is at most the row's gap width; otherwise it stops with its lowest pixels just above that row.
    - Components that pass one filter continue falling toward later filters or the bottom, and previously settled components become solid obstacles.

    Key insight:
    - The gray rows are not ordinary floors: each row is a width filter, so the final landing place depends on combining gravity, per-line thresholds, and bottom-up collision order.
    """
    g = np.asarray(grid, dtype=int)
    filters = _find_filters(g)
    assert len(filters) >= 2, "expected at least two filter rows"
    filter_widths = {item["row"]: item["gap_width"] for item in filters}
    comps = _extract_components(g)
    assert comps, "expected at least one colored component"

    out = np.where(g == FILTER, FILTER, BG).astype(int)
    for comp in comps:
        final_r0 = _settle_component(comp, out, filter_widths)
        _place_component(out, comp, final_r0)
    return out


def _dilate(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _component_candidates(top_h: int, width: int) -> list[np.ndarray]:
    return [tpl for tpl in TEMPLATES[width] if tpl.shape[0] <= top_h - 1]


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    rows, cols = np.where(mask)
    for dr, dc in zip(rows, cols):
        grid[r0 + int(dr), c0 + int(dc)] = color


def _place_with_padding(occupied: np.ndarray, mask: np.ndarray, row_choices: list[int], col_choices: list[int]) -> tuple[int, int] | None:
    dilated = _dilate(occupied)
    mh, mw = mask.shape
    for r0 in row_choices:
        for c0 in col_choices:
            window = occupied[r0 : r0 + mh, c0 : c0 + mw]
            if np.any(window & mask):
                continue
            padded = dilated[r0 : r0 + mh, c0 : c0 + mw]
            if np.any(padded & mask):
                continue
            return r0, c0
    return None


def _first_blocking_line(width: int, gap_widths: list[int]) -> int | None:
    for idx, gap_width in enumerate(gap_widths):
        if width > gap_width:
            return idx
    return None


def _normalized_coords(coords: np.ndarray) -> list[tuple[int, int]]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    return sorted((int(r - r0), int(c - c0)) for r, c in coords)


def _semantic_validate(inp: np.ndarray, out: np.ndarray, meta: dict) -> bool:
    if not np.array_equal(np.where(inp == FILTER, FILTER, BG), np.where(out == FILTER, FILTER, BG)):
        return False

    last_filter = meta["filter_rows"][-1]
    stack_final_bottoms = {}
    for comp in meta["components"]:
        coords = np.argwhere(out == comp["color"])
        if coords.shape[0] != len(comp["rel"]):
            return False
        if _normalized_coords(coords) != comp["rel"]:
            return False
        c0 = int(coords[:, 1].min())
        if c0 != comp["c0"]:
            return False
        bottom = int(coords[:, 0].max())
        block_idx = comp["block_idx"]
        if block_idx is None:
            if bottom <= last_filter:
                return False
        else:
            if bottom != meta["filter_rows"][block_idx] - 1:
                return False
        if comp["name"].startswith("stack_"):
            stack_final_bottoms[comp["name"]] = bottom

    if stack_final_bottoms.get("stack_lower", -1) <= stack_final_bottoms.get("stack_upper", 10**9):
        return False
    return True


def _sample_scene() -> tuple[np.ndarray, dict] | None:
    width = random.randint(10, 14)
    n_filters = random.choice([2, 3])
    top_h = random.randint(5, 6)
    middle_heights = [random.randint(3, 5) for _ in range(n_filters - 1)]
    bottom_h = random.randint(4, 6)
    height = top_h + sum(middle_heights) + bottom_h + n_filters
    if height > 30:
        return None

    gap_widths = [2] + [random.choice([1, 2]) for _ in range(n_filters - 1)]
    if all(gap_width == 2 for gap_width in gap_widths[1:]):
        gap_widths[random.randrange(1, n_filters)] = 1

    filter_rows = []
    row = top_h
    filter_rows.append(row)
    for band_h in middle_heights:
        row += 1 + band_h
        filter_rows.append(row)

    grid = np.zeros((height, width), dtype=int)
    gap_specs = []
    for row, gap_width in zip(filter_rows, gap_widths):
        gap_start = random.randint(1, width - gap_width - 1)
        grid[row, :] = FILTER
        grid[row, gap_start : gap_start + gap_width] = BG
        gap_specs.append((row, gap_start, gap_width))

    colors = random.sample(PALETTE, 4 + random.randint(0, 1))
    stack_col = random.randint(1, width - 2)
    occupied = np.zeros_like(grid, dtype=bool)
    components = []

    lower_mask = random.choice([tpl for tpl in _component_candidates(top_h, 1) if tpl.shape[0] <= 2])
    upper_mask = random.choice([tpl for tpl in _component_candidates(top_h, 1) if tpl.shape[0] <= 2])
    low_h = lower_mask.shape[0]
    up_h = upper_mask.shape[0]
    lower_r0 = random.randint(2, top_h - low_h)
    upper_max = lower_r0 - up_h - 1
    if upper_max < 0:
        return None
    upper_r0 = random.randint(0, upper_max)

    stack_lower = {
        "name": "stack_lower",
        "color": colors[0],
        "mask": lower_mask,
        "r0": lower_r0,
        "c0": stack_col,
        "width": 1,
        "block_idx": None,
    }
    stack_upper = {
        "name": "stack_upper",
        "color": colors[1],
        "mask": upper_mask,
        "r0": upper_r0,
        "c0": stack_col,
        "width": 1,
        "block_idx": None,
    }
    for spec in (stack_lower, stack_upper):
        _paint_mask(grid, spec["mask"], spec["r0"], spec["c0"], spec["color"])
        sub = occupied[spec["r0"] : spec["r0"] + spec["mask"].shape[0], spec["c0"] : spec["c0"] + spec["mask"].shape[1]]
        sub |= spec["mask"]
        components.append(spec)

    width2_candidates = [tpl for tpl in _component_candidates(top_h, 2) if tpl.shape[0] <= 2]
    width3_candidates = [tpl for tpl in _component_candidates(top_h, 3) if tpl.shape[0] <= 2]
    if not width2_candidates or not width3_candidates:
        return None

    must_have = [
        ("block_later", random.choice(width2_candidates), colors[2], 2),
        ("block_first", random.choice(width3_candidates), colors[3], 3),
    ]

    if len(colors) > 4:
        extra_width = random.choice([1, 2, 3])
        extra_candidates = _component_candidates(top_h, extra_width)
        if not extra_candidates:
            return None
        must_have.append(("extra", random.choice(extra_candidates), colors[4], extra_width))

    row_choices = list(range(top_h))
    random.shuffle(row_choices)
    col_choices = list(range(width))
    random.shuffle(col_choices)

    for name, mask, color, obj_width in must_have:
        mh, mw = mask.shape
        valid_rows = [r for r in row_choices if r + mh <= top_h]
        valid_cols = [c for c in col_choices if c + mw <= width]
        location = _place_with_padding(occupied, mask, valid_rows, valid_cols)
        if location is None:
            return None
        r0, c0 = location
        _paint_mask(grid, mask, r0, c0, color)
        occupied[r0 : r0 + mh, c0 : c0 + mw] |= mask
        components.append(
            {
                "name": name,
                "color": color,
                "mask": mask,
                "r0": r0,
                "c0": c0,
                "width": obj_width,
                "block_idx": _first_blocking_line(obj_width, gap_widths),
            }
        )

    if not any(comp["block_idx"] == 0 for comp in components):
        return None
    if not any(comp["block_idx"] is None for comp in components):
        return None
    if not any(comp["block_idx"] not in (None, 0) for comp in components):
        return None

    for comp in components:
        rows, cols = np.where(comp["mask"])
        comp["rel"] = sorted((int(r), int(c)) for r, c in zip(rows, cols))
        comp["area"] = int(comp["mask"].sum())

    meta = {
        "filter_rows": filter_rows,
        "gap_widths": gap_widths,
        "gap_specs": gap_specs,
        "components": components,
    }
    return grid, meta


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        sampled = _sample_scene()
        if sampled is None:
            continue
        grid, meta = sampled
        filters = _find_filters(grid)
        if [item["gap_width"] for item in filters] != meta["gap_widths"]:
            continue
        out = main(grid)
        if not _semantic_validate(grid, out, meta):
            continue
        return grid
    raise RuntimeError("failed to generate a valid selective-filter gravity sample")
