import itertools
import math
import random

import numpy as np


DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
BRANCHES = ("same", "hmirror", "vmirror")
ORIENTATIONS = ("horizontal", "vertical")
MAX_GENERATE_ATTEMPTS = 300


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _tight_mask(mask: np.ndarray) -> bool:
    return bool(np.any(mask[0]) and np.any(mask[-1]) and np.any(mask[:, 0]) and np.any(mask[:, -1]))


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in DIRS4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]) and (rr, cc) not in seen:
                seen.add((rr, cc))
                stack.append((rr, cc))
    return len(seen) == int(mask.sum())


def _build_component(coords: list[tuple[int, int]], color: int) -> dict:
    arr = np.array(coords, dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    mask[arr[:, 0] - r0, arr[:, 1] - c0] = True
    return {
        "color": int(color),
        "coords": arr,
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "mask": mask,
    }


def _find_color_components(grid: np.ndarray, color: int) -> list[dict]:
    mask = grid == int(color)
    visited = np.zeros_like(mask, dtype=bool)
    comps: list[dict] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(visited[r, c]):
                continue
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(visited[nr, nc]):
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(_build_component(coords, color))
    return comps


def _classify_relation(mask_a: np.ndarray, mask_b: np.ndarray) -> str:
    if mask_a.shape != mask_b.shape:
        raise AssertionError("paired objects must share bbox dimensions")
    if np.array_equal(mask_a, mask_b):
        return "same"
    if np.array_equal(np.fliplr(mask_a), mask_b):
        return "hmirror"
    if np.array_equal(np.flipud(mask_a), mask_b):
        return "vmirror"
    raise AssertionError("unsupported shape relation for paired objects")


def _pair_orientation(comp_a: dict, comp_b: dict) -> tuple[str, dict, dict]:
    ar0, ac0, ar1, ac1 = comp_a["bbox"]
    br0, bc0, br1, bc1 = comp_b["bbox"]
    row_overlap = max(ar0, br0) <= min(ar1, br1)
    col_overlap = max(ac0, bc0) <= min(ac1, bc1)

    if ac1 < bc0 and row_overlap:
        return "horizontal", comp_a, comp_b
    if bc1 < ac0 and row_overlap:
        return "horizontal", comp_b, comp_a
    if ar1 < br0 and col_overlap:
        return "vertical", comp_a, comp_b
    if br1 < ar0 and col_overlap:
        return "vertical", comp_b, comp_a
    raise AssertionError("paired objects must face each other along one axis")


def _row_spans(coords: np.ndarray) -> dict[int, tuple[int, int]]:
    spans: dict[int, list[int]] = {}
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if rr not in spans:
            spans[rr] = [cc, cc]
        else:
            spans[rr][0] = min(spans[rr][0], cc)
            spans[rr][1] = max(spans[rr][1], cc)
    return {r: (bounds[0], bounds[1]) for r, bounds in spans.items()}


def _col_spans(coords: np.ndarray) -> dict[int, tuple[int, int]]:
    spans: dict[int, list[int]] = {}
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if cc not in spans:
            spans[cc] = [rr, rr]
        else:
            spans[cc][0] = min(spans[cc][0], rr)
            spans[cc][1] = max(spans[cc][1], rr)
    return {c: (bounds[0], bounds[1]) for c, bounds in spans.items()}


def _horizontal_candidates(left: dict, right: dict, *, bbox_only: bool) -> list[tuple[int, int, int]]:
    if bbox_only:
        r0 = max(int(left["bbox"][0]), int(right["bbox"][0]))
        r1 = min(int(left["bbox"][2]), int(right["bbox"][2]))
        start = int(left["bbox"][3]) + 1
        stop = int(right["bbox"][1]) - 1
        return [(r, start, stop) for r in range(r0, r1 + 1) if start <= stop]

    left_rows = _row_spans(left["coords"])
    right_rows = _row_spans(right["coords"])
    rows = sorted(set(left_rows) & set(right_rows))
    candidates = []
    for row in rows:
        start = int(left_rows[row][1]) + 1
        stop = int(right_rows[row][0]) - 1
        if start <= stop:
            candidates.append((int(row), start, stop))
    return candidates


def _vertical_candidates(top: dict, bottom: dict, *, bbox_only: bool) -> list[tuple[int, int, int]]:
    if bbox_only:
        c0 = max(int(top["bbox"][1]), int(bottom["bbox"][1]))
        c1 = min(int(top["bbox"][3]), int(bottom["bbox"][3]))
        start = int(top["bbox"][2]) + 1
        stop = int(bottom["bbox"][0]) - 1
        return [(c, start, stop) for c in range(c0, c1 + 1) if start <= stop]

    top_cols = _col_spans(top["coords"])
    bottom_cols = _col_spans(bottom["coords"])
    cols = sorted(set(top_cols) & set(bottom_cols))
    candidates = []
    for col in cols:
        start = int(top_cols[col][1]) + 1
        stop = int(bottom_cols[col][0]) - 1
        if start <= stop:
            candidates.append((int(col), start, stop))
    return candidates


def _pick_single(candidates: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    assert candidates
    indices = [item[0] for item in candidates]
    center = 0.5 * (indices[0] + indices[-1])
    chosen = min(candidates, key=lambda item: (abs(item[0] - center), item[0]))
    return [chosen]


def _pick_double(candidates: list[tuple[int, int, int]]) -> list[tuple[int, int, int]] | None:
    by_index = {item[0]: item for item in candidates}
    pairs = [(idx, idx + 1) for idx in sorted(by_index) if idx + 1 in by_index]
    if not pairs:
        return None
    center = 0.5 * (pairs[0][0] + pairs[-1][1])
    lo, hi = min(pairs, key=lambda pair: (abs(0.5 * (pair[0] + pair[1]) - center), pair[0]))
    return [by_index[lo], by_index[hi]]


def _draw_horizontal_bridge(out: np.ndarray, left: dict, right: dict, color: int, relation: str) -> None:
    candidates = _horizontal_candidates(left, right, bbox_only=False)
    if relation == "hmirror":
        lines = _pick_double(candidates)
        if lines is None:
            lines = _pick_double(_horizontal_candidates(left, right, bbox_only=True))
    else:
        if not candidates:
            candidates = _horizontal_candidates(left, right, bbox_only=True)
        lines = _pick_single(candidates)

    assert lines, "horizontal bridge must have drawable rows"
    for row, start, stop in lines:
        if relation == "vmirror":
            out[row, start : stop + 1 : 2] = int(color)
        else:
            out[row, start : stop + 1] = int(color)


def _draw_vertical_bridge(out: np.ndarray, top: dict, bottom: dict, color: int, relation: str) -> None:
    candidates = _vertical_candidates(top, bottom, bbox_only=False)
    if relation == "hmirror":
        lines = _pick_double(candidates)
        if lines is None:
            lines = _pick_double(_vertical_candidates(top, bottom, bbox_only=True))
    else:
        if not candidates:
            candidates = _vertical_candidates(top, bottom, bbox_only=True)
        lines = _pick_single(candidates)

    assert lines, "vertical bridge must have drawable columns"
    for col, start, stop in lines:
        if relation == "vmirror":
            out[start : stop + 1 : 2, col] = int(color)
        else:
            out[start : stop + 1, col] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair each same-colored object duo and draw a bridge whose style encodes the duo's shape relation.

    Input structure:
    - The background is the most frequent color.
    - Each non-background color appears in exactly two small 4-connected monochrome objects.
    - The two objects of a color face each other either left-right with overlapping rows or top-bottom with overlapping columns.
    - Every paired object has 2-5 pixels, and its partner is either identical, a left-right mirror, or an up-down mirror.

    Rule:
    - Keep every original object unchanged.
    - For each color pair, compare the two tight binary masks in this order: identical, left-right mirror, then up-down mirror.
    - If the pair is identical, draw a one-cell-thick solid bridge between the facing boundaries along one shared row or column.
    - If the pair is a left-right mirror, draw a two-cell-thick solid bridge on adjacent shared rows or columns.
    - If the pair is an up-down mirror, draw a one-cell-thick dashed bridge that fills every other cell from one boundary toward the other.

    Key insight:
    - Color identifies which objects belong together, while the exact bridge pattern is controlled by the geometric relation between the paired shapes rather than by position alone.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    out = g.copy()
    colors = [int(color) for color in np.unique(g) if int(color) != int(bg)]

    for color in colors:
        comps = _find_color_components(g, color)
        assert len(comps) == 2, "each non-background color must form exactly two components"
        relation = _classify_relation(comps[0]["mask"], comps[1]["mask"])
        orientation, first, second = _pair_orientation(comps[0], comps[1])
        if orientation == "horizontal":
            _draw_horizontal_bridge(out, first, second, color, relation)
        else:
            _draw_vertical_bridge(out, first, second, color, relation)

    return out


def _rows_with_cells(mask: np.ndarray) -> list[int]:
    return [int(r) for r in np.flatnonzero(np.any(mask, axis=1))]


def _cols_with_cells(mask: np.ndarray) -> list[int]:
    return [int(c) for c in np.flatnonzero(np.any(mask, axis=0))]


def _supports_bridge(mask_a: np.ndarray, mask_b: np.ndarray, orientation: str, width: int) -> bool:
    if orientation == "horizontal":
        lines = sorted(set(_rows_with_cells(mask_a)) & set(_rows_with_cells(mask_b)))
    else:
        lines = sorted(set(_cols_with_cells(mask_a)) & set(_cols_with_cells(mask_b)))
    if width == 1:
        return bool(lines)
    line_set = set(lines)
    return any(line + 1 in line_set for line in lines)


def _enumerate_masks() -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for height in range(1, 4):
        for width in range(1, 4):
            limit = min(5, height * width)
            for area in range(2, limit + 1):
                for cells in itertools.combinations(range(height * width), area):
                    mask = np.zeros((height, width), dtype=bool)
                    for cell in cells:
                        mask[cell // width, cell % width] = True
                    key = (mask.shape, mask.tobytes())
                    if key in seen:
                        continue
                    if not _tight_mask(mask) or not _is_connected(mask):
                        continue
                    seen.add(key)
                    masks.append(mask)
    return masks


def _related_mask(mask: np.ndarray, branch: str) -> np.ndarray:
    if branch == "same":
        return mask.copy()
    if branch == "hmirror":
        return np.fliplr(mask)
    if branch == "vmirror":
        return np.flipud(mask)
    raise AssertionError("unknown branch")


def _build_template_catalog() -> dict[tuple[str, str], list[np.ndarray]]:
    catalog: dict[tuple[str, str], list[np.ndarray]] = {(branch, orientation): [] for branch in BRANCHES for orientation in ORIENTATIONS}
    for mask in _enumerate_masks():
        if _classify_relation(mask, mask) == "same" and not np.array_equal(mask, np.fliplr(mask)) and not np.array_equal(mask, np.flipud(mask)):
            for orientation in ORIENTATIONS:
                catalog[("same", orientation)].append(mask.copy())

        hmask = np.fliplr(mask)
        if _classify_relation(mask, hmask) == "hmirror":
            for orientation in ORIENTATIONS:
                if _supports_bridge(mask, hmask, orientation, width=2):
                    catalog[("hmirror", orientation)].append(mask.copy())

        vmask = np.flipud(mask)
        if _classify_relation(mask, vmask) == "vmirror":
            for orientation in ORIENTATIONS:
                if _supports_bridge(mask, vmask, orientation, width=1):
                    catalog[("vmirror", orientation)].append(mask.copy())

    for key, items in catalog.items():
        assert items, f"missing templates for {key}"
    return catalog


TEMPLATE_CATALOG = _build_template_catalog()


def _sample_branches() -> list[str]:
    branches = list(BRANCHES)
    if random.random() < 0.5:
        random.shuffle(branches)
        return branches
    extra = random.choice(branches)
    branches.append(extra)
    random.shuffle(branches)
    return branches


def _sample_pair_spec(branch: str, orientation: str, color: int) -> dict:
    base = random.choice(TEMPLATE_CATALOG[(branch, orientation)]).copy()
    mate = _related_mask(base, branch)
    if random.random() < 0.5:
        base, mate = mate, base
    height, width = map(int, base.shape)
    gap = random.randint(3, 7) if branch == "vmirror" else random.randint(2, 6)
    foot_h = height if orientation == "horizontal" else 2 * height + gap
    foot_w = 2 * width + gap if orientation == "horizontal" else width
    return {
        "branch": branch,
        "orientation": orientation,
        "color": int(color),
        "mask_a": base,
        "mask_b": mate,
        "height": height,
        "width": width,
        "gap": gap,
        "foot_h": foot_h,
        "foot_w": foot_w,
    }


def _place_specs(specs: list[dict], height: int, width: int) -> list[tuple[int, int]] | None:
    blocked = np.zeros((height, width), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(specs)
    order = sorted(range(len(specs)), key=lambda idx: specs[idx]["foot_h"] * specs[idx]["foot_w"], reverse=True)

    for idx in order:
        spec = specs[idx]
        positions: list[tuple[int, int]] = []
        for r0 in range(height - spec["foot_h"] + 1):
            for c0 in range(width - spec["foot_w"] + 1):
                r1 = r0 + spec["foot_h"] - 1
                c1 = c0 + spec["foot_w"] - 1
                rr0 = max(0, r0 - 1)
                cc0 = max(0, c0 - 1)
                rr1 = min(height - 1, r1 + 1)
                cc1 = min(width - 1, c1 + 1)
                if not np.any(blocked[rr0 : rr1 + 1, cc0 : cc1 + 1]):
                    positions.append((r0, c0))
        if not positions:
            return None
        r0, c0 = random.choice(positions)
        placements[idx] = (r0, c0)
        r1 = r0 + spec["foot_h"] - 1
        c1 = c0 + spec["foot_w"] - 1
        blocked[max(0, r0 - 1) : min(height, r1 + 2), max(0, c0 - 1) : min(width, c1 + 2)] = True

    return [pos for pos in placements if pos is not None]


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    patch = grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]]
    patch[mask] = int(color)


def _build_input(specs: list[dict], placements: list[tuple[int, int]], bg: int, height: int, width: int) -> np.ndarray:
    grid = np.full((height, width), int(bg), dtype=int)
    for spec, (r0, c0) in zip(specs, placements):
        if spec["orientation"] == "horizontal":
            a_r, a_c = r0, c0
            b_r, b_c = r0, c0 + spec["width"] + spec["gap"]
        else:
            a_r, a_c = r0, c0
            b_r, b_c = r0 + spec["height"] + spec["gap"], c0
        _paint_mask(grid, spec["mask_a"], a_r, a_c, spec["color"])
        _paint_mask(grid, spec["mask_b"], b_r, b_c, spec["color"])
    return grid


def _grid_bounds(specs: list[dict]) -> tuple[int, int]:
    expanded_area = sum((spec["foot_h"] + 2) * (spec["foot_w"] + 2) for spec in specs)
    max_h = max(spec["foot_h"] for spec in specs) + 2
    max_w = max(spec["foot_w"] for spec in specs) + 2
    side = int(math.ceil(math.sqrt(expanded_area * 1.35)))
    lo_h = max(max_h, side - 2)
    lo_w = max(max_w, side - 2)
    hi_h = min(30, max(lo_h, side + 5))
    hi_w = min(30, max(lo_w, side + 5))
    return random.randint(lo_h, hi_h), random.randint(lo_w, hi_w)


def _valid_input(grid: np.ndarray, bg: int, pair_count: int) -> bool:
    colors = [int(color) for color in np.unique(grid) if int(color) != int(bg)]
    if len(colors) != pair_count:
        return False
    for color in colors:
        comps = _find_color_components(grid, color)
        if len(comps) != 2:
            return False
        if any(int(comp["mask"].sum()) < 2 or int(comp["mask"].sum()) > 5 for comp in comps):
            return False
        try:
            relation = _classify_relation(comps[0]["mask"], comps[1]["mask"])
            orientation, first, second = _pair_orientation(comps[0], comps[1])
        except AssertionError:
            return False
        if orientation == "horizontal":
            candidates = _horizontal_candidates(first, second, bbox_only=False)
            if relation == "hmirror" and _pick_double(candidates) is None:
                return False
            if relation != "hmirror" and not candidates:
                return False
        else:
            candidates = _vertical_candidates(first, second, bbox_only=False)
            if relation == "hmirror" and _pick_double(candidates) is None:
                return False
            if relation != "hmirror" and not candidates:
                return False
    return True


def _gen_attempt() -> np.ndarray | None:
    bg = random.randrange(10)
    branches = _sample_branches()
    pair_count = len(branches)
    palette = [color for color in range(10) if color != bg]
    colors = random.sample(palette, pair_count)
    specs = [
        _sample_pair_spec(branch, random.choice(ORIENTATIONS), color)
        for branch, color in zip(branches, colors)
    ]
    height, width = _grid_bounds(specs)
    placements = _place_specs(specs, height, width)
    if placements is None:
        return None

    grid = _build_input(specs, placements, bg, height, width)
    if not _valid_input(grid, bg, pair_count):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid bridging-pairs ARC input")
