from __future__ import annotations

import random

import numpy as np


_CORNERS = ("tl", "tr", "bl", "br")
_MISSING_TO_CORNER = {
    (0, 0): "tl",
    (0, 1): "tr",
    (1, 0): "bl",
    (1, 1): "br",
}
_BG = 0
_MAX_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()


def _tight_crop_non_bg(grid: np.ndarray, bg: int) -> np.ndarray:
    if grid.size == 0:
        return grid.copy()
    pts = np.argwhere(grid != bg)
    if pts.size == 0:
        return grid.copy()
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()


def _quadrant_region(region: tuple[int, int, int, int], corner: str) -> tuple[int, int, int, int]:
    r0, r1, c0, c1 = region
    h = r1 - r0
    w = c1 - c0
    top_end = r0 + (h + 1) // 2
    bottom_start = r0 + h // 2
    left_end = c0 + (w + 1) // 2
    right_start = c0 + w // 2
    if corner == "tl":
        return r0, top_end, c0, left_end
    if corner == "tr":
        return r0, top_end, right_start, c1
    if corner == "bl":
        return bottom_start, r1, c0, left_end
    if corner == "br":
        return bottom_start, r1, right_start, c1
    raise ValueError(corner)


def _crop_quadrant(grid: np.ndarray, corner: str) -> np.ndarray:
    r0, r1, c0, c1 = _quadrant_region((0, grid.shape[0], 0, grid.shape[1]), corner)
    return np.asarray(grid[r0:r1, c0:c1], dtype=int).copy()


def _region_area(region: tuple[int, int, int, int]) -> int:
    r0, r1, c0, c1 = region
    return max(0, r1 - r0) * max(0, c1 - c0)


def _point_in_region(point: tuple[int, int], region: tuple[int, int, int, int]) -> bool:
    r, c = point
    r0, r1, c0, c1 = region
    return r0 <= r < r1 and c0 <= c < c1


def _marker_info(
    grid: np.ndarray,
    coords: list[tuple[int, int]],
    bg: int,
) -> dict[str, object] | None:
    box = _bbox(coords)
    r0, c0, r1, c1 = box
    if (r1 - r0 + 1, c1 - c0 + 1) != (2, 2) or len(coords) != 3:
        return None
    colors = {int(grid[r, c]) for r, c in coords}
    if len(colors) != 1:
        return None
    patch = _crop_bbox(grid, box)
    if np.count_nonzero(patch != bg) != 3:
        return None
    missing = np.argwhere(patch == bg)
    if missing.shape != (1, 2):
        return None
    corner = _MISSING_TO_CORNER[(int(missing[0, 0]), int(missing[0, 1]))]
    return {
        "bbox": box,
        "color": int(next(iter(colors))),
        "corner": corner,
    }


def _apply_markers(
    mother: np.ndarray,
    ordered_markers: list[tuple[int, str]],
    bg: int,
) -> np.ndarray:
    current = np.asarray(mother, dtype=int).copy()
    for color, corner in ordered_markers:
        candidate = _crop_quadrant(current, corner)
        if candidate.size > 0 and np.any(candidate == color):
            current = candidate
    return _tight_crop_non_bg(current, bg)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _background_color(g)

    comps = _components(g != bg)
    assert len(comps) >= 3, "expected one mother pattern and at least two markers"

    mother_coords = max(comps, key=len)
    mother_box = _bbox(mother_coords)
    mother = _crop_bbox(g, mother_box)
    mother_colors = {int(v) for v in mother.ravel() if int(v) != bg}
    assert 2 <= len(mother_colors) <= 4, "mother pattern must use 2-4 foreground colors"

    markers = []
    for comp in comps:
        if comp is mother_coords:
            continue
        info = _marker_info(g, comp, bg)
        if info is not None:
            markers.append(info)
    assert 2 <= len(markers) <= 4, "expected 2-4 L markers"

    markers.sort(key=lambda d: (int(d["bbox"][0]), int(d["bbox"][1])))
    ordered = [(int(d["color"]), str(d["corner"])) for d in markers]
    return _apply_markers(mother, ordered, bg)


def _marker_patch(color: int, corner: str) -> np.ndarray:
    patch = np.full((2, 2), _BG, dtype=int)
    patch[:, :] = color
    if corner == "tl":
        patch[0, 0] = _BG
    elif corner == "tr":
        patch[0, 1] = _BG
    elif corner == "bl":
        patch[1, 0] = _BG
    elif corner == "br":
        patch[1, 1] = _BG
    else:
        raise ValueError(corner)
    return patch


def _paste_non_bg(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    sub = dst[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
    mask = patch != bg
    sub[mask] = patch[mask]


def _step_trace(
    mother: np.ndarray,
    marker_specs: list[tuple[int, str, bool]],
) -> list[bool]:
    current = np.asarray(mother, dtype=int).copy()
    applied: list[bool] = []
    for color, corner, _ in marker_specs:
        candidate = _crop_quadrant(current, corner)
        ok = bool(candidate.size > 0 and np.any(candidate == color))
        applied.append(ok)
        if ok:
            current = candidate
    return applied


def _build_constraints(
    shape: tuple[int, int],
    marker_specs: list[tuple[int, str, bool]],
) -> tuple[
    dict[int, tuple[int, int, int, int]],
    dict[int, tuple[int, int, int, int]],
    tuple[int, int, int, int],
] | None:
    current = (0, shape[0], 0, shape[1])
    required: dict[int, tuple[int, int, int, int]] = {}
    forbidden: dict[int, tuple[int, int, int, int]] = {}
    for color, corner, should_apply in marker_specs:
        region = _quadrant_region(current, corner)
        if should_apply:
            if _region_area(region) == 0:
                return None
            required[color] = region
            current = region
        else:
            forbidden[color] = region
    return required, forbidden, current


def _allowed_colors_for_cell(
    point: tuple[int, int],
    palette: list[int],
    forbidden: dict[int, tuple[int, int, int, int]],
) -> list[int]:
    return [color for color in palette if color not in forbidden or not _point_in_region(point, forbidden[color])]


def _pick_distinct_cell(
    candidates: list[tuple[int, int]],
    used: set[tuple[int, int]],
) -> tuple[int, int] | None:
    free = [cell for cell in candidates if cell not in used]
    if free:
        return random.choice(free)
    if candidates:
        return random.choice(candidates)
    return None


def _render_mother(
    h: int,
    w: int,
    palette: list[int],
    required: dict[int, tuple[int, int, int, int]],
    forbidden: dict[int, tuple[int, int, int, int]],
) -> np.ndarray | None:
    all_cells = [(r, c) for r in range(h) for c in range(w)]
    allowed_cells: dict[int, list[tuple[int, int]]] = {}
    for color in palette:
        allowed = [cell for cell in all_cells if color not in forbidden or not _point_in_region(cell, forbidden[color])]
        if not allowed:
            return None
        allowed_cells[color] = allowed

    for cell in all_cells:
        if not _allowed_colors_for_cell(cell, palette, forbidden):
            return None

    used: set[tuple[int, int]] = set()
    seed_map: dict[int, list[tuple[int, int]]] = {}
    for color in palette:
        if color in required:
            region = required[color]
            candidates = [cell for cell in allowed_cells[color] if _point_in_region(cell, region)]
        else:
            candidates = allowed_cells[color]
        primary = _pick_distinct_cell(candidates, used)
        if primary is None:
            return None
        used.add(primary)
        seed_map[color] = [primary]

        extra_count = random.randint(0, 2)
        extras = [cell for cell in allowed_cells[color] if cell not in used]
        random.shuffle(extras)
        for cell in extras[:extra_count]:
            seed_map[color].append(cell)
            used.add(cell)

    row_scales = {color: random.uniform(0.85, 1.25) for color in palette}
    col_scales = {color: random.uniform(0.85, 1.25) for color in palette}
    offsets = {
        (color, seed_idx): random.uniform(0.0, 0.35)
        for color in palette
        for seed_idx in range(len(seed_map[color]))
    }

    mother = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            point = (r, c)
            allowed = _allowed_colors_for_cell(point, palette, forbidden)
            best_color = None
            best_score = None
            for color in allowed:
                color_best = None
                for seed_idx, (sr, sc) in enumerate(seed_map[color]):
                    score = (
                        abs(r - sr) * row_scales[color]
                        + abs(c - sc) * col_scales[color]
                        + offsets[(color, seed_idx)]
                    )
                    if color_best is None or score < color_best:
                        color_best = score
                if best_score is None or color_best < best_score:
                    best_score = color_best
                    best_color = color
            assert best_color is not None
            mother[r, c] = best_color

    for color in palette:
        if not np.any(mother == color):
            return None
    for color, region in required.items():
        r0, r1, c0, c1 = region
        if not np.any(mother[r0:r1, c0:c1] == color):
            return None
    for color, region in forbidden.items():
        r0, r1, c0, c1 = region
        if np.any(mother[r0:r1, c0:c1] == color):
            return None
    return mother


def _render_input_grid(mother: np.ndarray, marker_specs: list[tuple[int, str, bool]]) -> np.ndarray:
    mh, mw = mother.shape
    marker_gap = random.randint(1, 3)
    markers_h = len(marker_specs) * 2 + (len(marker_specs) - 1) * marker_gap
    left_margin = random.randint(0, 2)
    right_margin = random.randint(0, 2)
    mother_top = random.randint(0, 2)
    marker_top = random.randint(0, 2)
    bottom_margin = random.randint(0, 2)
    gap = random.randint(3, 5)

    H = max(mother_top + mh, marker_top + markers_h) + bottom_margin
    W = left_margin + mw + gap + 2 + right_margin
    grid = np.full((H, W), _BG, dtype=int)

    grid[mother_top : mother_top + mh, left_margin : left_margin + mw] = mother
    marker_c0 = left_margin + mw + gap
    cur_r = marker_top
    for color, corner, _ in marker_specs:
        _paste_non_bg(grid, _marker_patch(color, corner), cur_r, marker_c0, _BG)
        cur_r += 2 + marker_gap
    return grid


def _expected_output(mother: np.ndarray, marker_specs: list[tuple[int, str, bool]]) -> np.ndarray:
    ordered = [(color, corner) for color, corner, _ in marker_specs]
    return _apply_markers(mother, ordered, _BG)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(6, 10)
    w = random.randint(6, 10)
    n_markers = random.randint(2, 4)
    n_colors = random.randint(max(2, n_markers), 4)
    palette = random.sample(range(1, 10), n_colors)
    marker_colors = random.sample(palette, n_markers)
    marker_specs = [
        (
            marker_colors[i],
            random.choice(_CORNERS),
            bool(random.getrandbits(1)),
        )
        for i in range(n_markers)
    ]

    constraints = _build_constraints((h, w), marker_specs)
    if constraints is None:
        return None
    required, forbidden, final_region = constraints
    if _region_area(final_region) == 0:
        return None

    mother = _render_mother(h, w, palette, required, forbidden)
    if mother is None:
        return None

    actual_trace = _step_trace(mother, marker_specs)
    target_trace = [spec[2] for spec in marker_specs]
    if actual_trace != target_trace:
        return None

    grid = _render_input_grid(mother, marker_specs)
    expected = _expected_output(mother, marker_specs)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
