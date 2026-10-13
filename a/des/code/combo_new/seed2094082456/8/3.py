from __future__ import annotations

import random

import numpy as np


BG = 0
LEGEND_FRAME_COLOR = 2
REGION_FRAME_COLOR = 5
SEED_COLORS = (1, 3, 4, 6, 7, 8, 9)
PATTERN_KINDS = ("horizontal", "checker", "vertical")
MAX_GEN_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[counts.argmax()])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _border_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _color_components(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for color0 in np.unique(grid):
        color = int(color0)
        if color == bg:
            continue
        for comp in _color_components(grid, color):
            r0, c0, r1, c1 = _bbox(comp)
            height = r1 - r0 + 1
            width = c1 - c0 + 1
            if height < 3 or width < 3:
                continue
            border = _border_mask(height, width)
            if len(comp) != int(border.sum()):
                continue
            sub = grid[r0 : r1 + 1, c0 : c1 + 1]
            if np.array_equal(sub == color, border):
                frames.append({"color": color, "bbox": (r0, c0, r1, c1)})
    return frames


def _strictly_inside(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 < ir0 and oc0 < ic0 and ir1 < or1 and ic1 < oc1


def _touches_corner(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    h, w = shape
    return (
        (r0 == 0 and c0 == 0)
        or (r0 == 0 and c1 == w - 1)
        or (r1 == h - 1 and c0 == 0)
        or (r1 == h - 1 and c1 == w - 1)
    )


def _third_sizes(length: int) -> tuple[int, int, int]:
    base, extra = divmod(length, 3)
    sizes = [base + (1 if i < extra else 0) for i in range(3)]
    assert min(sizes) >= 1, "regions must have enough rows for three vertical bands"
    return int(sizes[0]), int(sizes[1]), int(sizes[2])


def _band_limits(length: int) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    a, b, c = _third_sizes(length)
    return ((0, a), (a, a + b), (a + b, a + b + c))


def _band_of_row(row: int, height: int) -> int:
    for band, (start, stop) in enumerate(_band_limits(height)):
        if start <= row < stop:
            return band
    raise AssertionError("row must fall in one of the thirds")


def _pattern_mask(kind: str, height: int, width: int) -> np.ndarray:
    rr = np.arange(height)[:, None]
    cc = np.arange(width)[None, :]
    if kind == "horizontal":
        return np.broadcast_to((rr % 2) == 0, (height, width)).copy()
    if kind == "vertical":
        return np.broadcast_to((cc % 2) == 0, (height, width)).copy()
    if kind == "checker":
        return ((rr + cc) % 2) == 0
    raise AssertionError(f"unknown pattern kind: {kind}")


def _identify_pattern(mask: np.ndarray) -> str:
    matches = [kind for kind in PATTERN_KINDS if np.array_equal(mask, _pattern_mask(kind, *mask.shape))]
    assert len(matches) == 1, "legend sample must match exactly one canonical pattern"
    return matches[0]


def _find_legend_frame(grid: np.ndarray, frames: list[dict]) -> tuple[dict, list[dict]]:
    candidates: list[tuple[int, dict, list[dict]]] = []
    for frame in frames:
        if not _touches_corner(frame["bbox"], grid.shape):
            continue
        inside = [other for other in frames if other is not frame and _strictly_inside(frame["bbox"], other["bbox"])]
        if len(inside) != 3:
            continue
        inside_colors = {inner["color"] for inner in inside}
        if len(inside_colors) != 1 or next(iter(inside_colors)) == frame["color"]:
            continue
        area = (frame["bbox"][2] - frame["bbox"][0] + 1) * (frame["bbox"][3] - frame["bbox"][1] + 1)
        candidates.append((area, frame, inside))
    assert len(candidates) == 1, "expected one corner legend frame containing exactly three sample frames"
    _area, legend, samples = candidates[0]
    return legend, samples


def _legend_mapping(grid: np.ndarray, bg: int, sample_frames: list[dict]) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for frame in sample_frames:
        r0, c0, r1, c1 = frame["bbox"]
        sub = grid[r0 + 1 : r1, c0 + 1 : c1]
        non_bg = sub[sub != bg]
        vals, counts = np.unique(non_bg, return_counts=True)
        assert len(vals) >= 2, "legend sample needs a distinct seed marker"
        seed_color = int(vals[counts.argmin()])
        seed_cells = np.argwhere(sub == seed_color)
        assert len(seed_cells) == 1, "legend sample must have exactly one seed marker"
        band = _band_of_row(int(seed_cells[0, 0]), sub.shape[0])
        mapping[band] = _identify_pattern(sub != bg)
    assert set(mapping) == {0, 1, 2}, "legend must define all three vertical bands"
    return mapping


def _fill_region(region: np.ndarray, color: int, kind: str, bg: int) -> None:
    mask = _pattern_mask(kind, *region.shape)
    region[(region == bg) & mask] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each framed rectangle with a pattern chosen from the seed's vertical position.

    Input structure:
    - The background is black. Several one-pixel-thick gray rectangular frames enclose empty interiors.
    - Each non-legend frame contains exactly one colored seed pixel inside the border.
    - A red-bordered legend sits in one corner and contains three framed pattern samples whose seed markers are placed in the top, middle, and bottom thirds.

    Rule:
    - Read the legend to map top-third seeds to horizontal stripes, middle-third seeds to checkerboard fill, and bottom-third seeds to vertical stripes.
    - For every gray framed region outside the legend, inspect the seed's vertical band inside the interior and select the corresponding pattern.
    - Fill the interior background pixels with the seed color under that pattern, preserve the seed pixel itself, and leave the legend unchanged.

    Key insight:
    - The seed carries two pieces of information at once: its color is the paint color, while its relative vertical placement chooses which canonical tiling pattern to apply.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    frames = _find_frames(g, bg)
    legend_frame, sample_frames = _find_legend_frame(g, frames)
    mapping = _legend_mapping(g, bg, sample_frames)
    sample_bboxes = {tuple(frame["bbox"]) for frame in sample_frames}
    legend_bbox = tuple(legend_frame["bbox"])

    out = g.copy()
    for frame in frames:
        bbox = tuple(frame["bbox"])
        if bbox == legend_bbox or bbox in sample_bboxes:
            continue
        if _strictly_inside(legend_bbox, bbox):
            continue
        r0, c0, r1, c1 = bbox
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        seed_cells = np.argwhere(interior != bg)
        assert len(seed_cells) == 1, "each task region must contain exactly one seed"
        sr, sc = map(int, seed_cells[0])
        seed_color = int(interior[sr, sc])
        band = _band_of_row(sr, interior.shape[0])
        patch = out[r0 + 1 : r1, c0 + 1 : c1]
        _fill_region(patch, seed_color, mapping[band], bg)
    return out


def _pattern_for_band(band: int) -> str:
    return ("horizontal", "checker", "vertical")[band]


def _candidate_seed_positions(height: int, width: int, band: int, kind: str) -> list[tuple[int, int]]:
    mask = _pattern_mask(kind, height, width)
    start, stop = _band_limits(height)[band]
    coords: list[tuple[int, int]] = []
    for r in range(start, stop):
        for c in range(width):
            if bool(mask[r, c]):
                coords.append((r, c))
    assert coords, "every band must have at least one visible seed position"
    return coords


def _make_task_region(
    interior_h: int,
    interior_w: int,
    band: int,
    color: int,
) -> tuple[np.ndarray, tuple[int, int], str]:
    kind = _pattern_for_band(band)
    patch = np.full((interior_h + 2, interior_w + 2), BG, dtype=int)
    patch[0, :] = REGION_FRAME_COLOR
    patch[-1, :] = REGION_FRAME_COLOR
    patch[:, 0] = REGION_FRAME_COLOR
    patch[:, -1] = REGION_FRAME_COLOR
    seed_r, seed_c = random.choice(_candidate_seed_positions(interior_h, interior_w, band, kind))
    patch[1 + seed_r, 1 + seed_c] = color
    return patch, (seed_r, seed_c), kind


def _make_legend_sample(
    interior_h: int,
    interior_w: int,
    band: int,
    kind: str,
    fill_color: int,
    seed_marker: int,
) -> np.ndarray:
    patch = np.full((interior_h + 2, interior_w + 2), BG, dtype=int)
    patch[0, :] = REGION_FRAME_COLOR
    patch[-1, :] = REGION_FRAME_COLOR
    patch[:, 0] = REGION_FRAME_COLOR
    patch[:, -1] = REGION_FRAME_COLOR
    inner = patch[1:-1, 1:-1]
    inner[_pattern_mask(kind, interior_h, interior_w)] = fill_color
    seed_r, seed_c = random.choice(_candidate_seed_positions(interior_h, interior_w, band, kind))
    inner[seed_r, seed_c] = seed_marker
    return patch


def _paste(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    grid[top : top + h, left : left + w] = patch


def _build_legend() -> tuple[np.ndarray, dict[int, str]]:
    sample_h = random.choice((5, 6))
    sample_w = random.choice((3, 4))
    seed_marker = random.choice(SEED_COLORS)
    fill_pool = [c for c in SEED_COLORS if c != seed_marker]
    fill_colors = random.sample(fill_pool, 3)
    samples: list[np.ndarray] = []
    mapping: dict[int, str] = {}
    for band in range(3):
        kind = _pattern_for_band(band)
        mapping[band] = kind
        samples.append(_make_legend_sample(sample_h, sample_w, band, kind, fill_colors[band], seed_marker))

    frame_h = sample_h + 2
    frame_w = sample_w + 2
    gap = 1
    legend_h = frame_h + 4
    legend_w = 3 * frame_w + 2 * gap + 4
    legend = np.full((legend_h, legend_w), BG, dtype=int)
    legend[0, :] = LEGEND_FRAME_COLOR
    legend[-1, :] = LEGEND_FRAME_COLOR
    legend[:, 0] = LEGEND_FRAME_COLOR
    legend[:, -1] = LEGEND_FRAME_COLOR
    left = 2
    top = 2
    order = list(range(3))
    random.shuffle(order)
    cursor = left
    for idx in order:
        _paste(legend, samples[idx], top, cursor)
        cursor += frame_w + gap
    return legend, mapping


def _bbox_overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _place_patch(
    grid_shape: tuple[int, int],
    patch_shape: tuple[int, int],
    occupied: list[tuple[int, int, int, int]],
    border_pad: int,
) -> tuple[int, int] | None:
    gh, gw = grid_shape
    ph, pw = patch_shape
    candidates = []
    for top in range(1, gh - ph):
        for left in range(1, gw - pw):
            bbox = (top, left, top + ph - 1, left + pw - 1)
            if any(_bbox_overlaps(bbox, other, pad=1) for other in occupied):
                continue
            if (
                bbox[0] < border_pad
                or bbox[1] < border_pad
                or bbox[2] >= gh - border_pad
                or bbox[3] >= gw - border_pad
            ):
                continue
            candidates.append((top, left))
    if not candidates:
        return None
    return random.choice(candidates)


def _legend_anchor(shape: tuple[int, int], patch_shape: tuple[int, int], corner: str) -> tuple[int, int]:
    h, w = shape
    ph, pw = patch_shape
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, w - pw
    if corner == "bl":
        return h - ph, 0
    if corner == "br":
        return h - ph, w - pw
    raise AssertionError(f"unknown corner: {corner}")


def _expected_output(inp: np.ndarray, regions: list[dict], mapping: dict[int, str]) -> np.ndarray:
    out = inp.copy()
    for region in regions:
        r0, c0, r1, c1 = region["bbox"]
        patch = out[r0 + 1 : r1, c0 + 1 : c1]
        _fill_region(patch, int(region["color"]), mapping[int(region["band"])], BG)
    return out


def _band_sequence(count: int) -> list[int]:
    base = [0, 1, 2]
    if count == 3:
        bands = base
    elif count == 4:
        bands = base + [random.choice(base)]
    elif count == 5:
        extra = base.copy()
        random.shuffle(extra)
        bands = base + extra[:2]
    else:
        bands = base * 2
    random.shuffle(bands)
    return bands


def _generate_case() -> np.ndarray | None:
    legend, mapping = _build_legend()
    lh, lw = legend.shape
    min_rows = max(22, lh + 6)
    min_cols = max(22, lw + 2)
    if min_rows > 30 or min_cols > 30:
        return None
    rows = random.randint(min_rows, 30)
    cols = random.randint(min_cols, 30)

    grid = np.full((rows, cols), BG, dtype=int)
    corner = random.choice(("tl", "tr", "bl", "br"))
    legend_top, legend_left = _legend_anchor(grid.shape, legend.shape, corner)
    _paste(grid, legend, legend_top, legend_left)
    occupied = [(legend_top, legend_left, legend_top + lh - 1, legend_left + lw - 1)]

    count = random.randint(3, 5)
    colors = random.sample(SEED_COLORS, count)
    bands = _band_sequence(count)
    regions: list[dict] = []

    for idx in range(count):
        interior_h = random.randint(4, 8)
        interior_w = random.randint(4, 8)
        patch, seed_pos, kind = _make_task_region(interior_h, interior_w, bands[idx], colors[idx])
        location = _place_patch(grid.shape, patch.shape, occupied, border_pad=1)
        if location is None:
            return None
        top, left = location
        _paste(grid, patch, top, left)
        bbox = (top, left, top + patch.shape[0] - 1, left + patch.shape[1] - 1)
        occupied.append(bbox)
        regions.append(
            {
                "bbox": bbox,
                "band": bands[idx],
                "kind": kind,
                "color": colors[idx],
                "seed_pos": seed_pos,
            }
        )

    expected = _expected_output(grid, regions, mapping)
    try:
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None
    if int(np.count_nonzero(grid == BG)) <= grid.size // 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input for puzzle 3")
