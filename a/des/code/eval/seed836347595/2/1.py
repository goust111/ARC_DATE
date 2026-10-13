import random
import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
FRAGMENT_COLORS = (1, 2, 3, 4, 6, 7, 8)
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]] | tuple[tuple[int, int], ...] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in points]
    cs = [c for _, c in points]
    return min(rs), min(cs), max(rs), max(cs)


def _normalize(points: list[tuple[int, int]] | tuple[tuple[int, int], ...] | set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = _bbox(points)
    return tuple(sorted((r - r0, c - c0) for r, c in points))


def _shape_dims(coords: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in coords) + 1, max(c for _, c in coords) + 1


def _paint_coords(grid: np.ndarray, coords: tuple[tuple[int, int], ...], origin: tuple[int, int], color: int) -> None:
    r0, c0 = origin
    for dr, dc in coords:
        grid[r0 + dr, c0 + dc] = color


def _draw_frame(grid: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    grid[r0, c0 : c0 + w] = MAROON
    grid[r0 + h - 1, c0 : c0 + w] = MAROON
    grid[r0 : r0 + h, c0] = MAROON
    grid[r0 : r0 + h, c0 + w - 1] = MAROON


def _extract_templates(grid: np.ndarray) -> list[tuple[tuple[int, int], ...]]:
    items = []
    for pts in _components(grid == GREY):
        r0, c0, _, _ = _bbox(pts)
        items.append((r0, c0, _normalize(pts)))
    items.sort(key=lambda item: (item[0], item[1]))
    templates = [coords for _, _, coords in items]
    assert templates, "no grey templates found"
    return templates


def _extract_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for pts in _components(grid == MAROON):
        r0, c0, r1, c1 = _bbox(pts)
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "maroon frame too small"
        border = set()
        for r in range(r0, r1 + 1):
            border.add((r, c0))
            border.add((r, c1))
        for c in range(c0, c1 + 1):
            border.add((r0, c))
            border.add((r1, c))
        assert set(pts) == border, "maroon component must be a 1-pixel rectangular frame"
        frames.append((r0, c0, r1, c1))
    frames.sort()
    assert frames, "no maroon frames found"
    return frames


def _all_matches(
    fragment_pts: tuple[tuple[int, int], ...],
    templates: list[tuple[tuple[int, int], ...]],
    bounds: tuple[int, int, int, int],
) -> list[tuple[int, int, int, tuple[tuple[int, int], ...]]]:
    assert fragment_pts, "fragment cannot be empty"
    frag_set = set(fragment_pts)
    f0 = fragment_pts[0]
    rmin, cmin, rmax, cmax = bounds
    matches: dict[tuple[int, int, int], tuple[tuple[int, int], ...]] = {}
    for idx, template in enumerate(templates):
        for tr, tc in template:
            dr = f0[0] - tr
            dc = f0[1] - tc
            placed = tuple(sorted((dr + r, dc + c) for r, c in template))
            if any(rr < rmin or rr > rmax or cc < cmin or cc > cmax for rr, cc in placed):
                continue
            if frag_set.issubset(set(placed)):
                matches[(idx, dr, dc)] = placed
    return [(idx, dr, dc, placed) for (idx, dr, dc), placed in sorted(matches.items())]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each colored fragment inside a maroon frame using the unique matching grey template.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    templates = _extract_templates(g)

    for fr0, fc0, fr1, fc1 in _extract_frames(g):
        inner = g[fr0 + 1 : fr1, fc0 + 1 : fc1]
        colors = sorted(set(map(int, np.unique(inner))) - {BLACK, GREY, MAROON})
        assert len(colors) == 1, "each frame must contain exactly one fragment color"
        color = colors[0]

        fragment_local = tuple(sorted((int(r), int(c)) for r, c in np.argwhere(inner == color)))
        assert fragment_local, "frame interior must contain a fragment"
        fragment_abs = tuple(sorted((fr0 + 1 + r, fc0 + 1 + c) for r, c in fragment_local))
        matches = _all_matches(fragment_abs, templates, (fr0 + 1, fc0 + 1, fr1 - 1, fc1 - 1))
        assert len(matches) == 1, "fragment must uniquely match exactly one template placement"

        _, _, _, full_abs = matches[0]
        for r, c in full_abs:
            out[r, c] = color
    return out


def _random_shape(rng: random.Random) -> tuple[tuple[int, int], ...] | None:
    for _ in range(100):
        h = rng.randint(2, 5)
        w = rng.randint(2, 5)
        max_area = min(8, h * w)
        if max_area < 4:
            continue
        area = rng.randint(4, max_area)
        start = (rng.randrange(h), rng.randrange(w))
        cells = {start}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))
        if len(cells) != area:
            continue
        shape = _normalize(cells)
        sh, sw = _shape_dims(shape)
        if sh < 2 or sw < 2:
            continue
        return shape
    return None


def _choose_templates(rng: random.Random) -> list[tuple[tuple[int, int], ...]] | None:
    need = rng.randint(3, 5)
    templates: list[tuple[tuple[int, int], ...]] = []
    seen = set()
    for _ in range(300):
        shape = _random_shape(rng)
        if shape is None or shape in seen:
            continue
        seen.add(shape)
        templates.append(shape)
        if len(templates) == need:
            return templates
    return None


def _valid_fragment_subsets(
    full_coords: tuple[tuple[int, int], ...],
    template_idx: int,
    origin: tuple[int, int],
    templates: list[tuple[tuple[int, int], ...]],
    inner_shape: tuple[int, int],
) -> list[tuple[tuple[int, int], ...]]:
    inner_h, inner_w = inner_shape
    target = (template_idx, origin[0], origin[1])
    n = len(full_coords)
    full_mask = (1 << n) - 1
    min_keep = 1 if n <= 4 else 2
    valid: list[tuple[tuple[int, int], ...]] = []
    for mask in range(1, full_mask):
        if mask == full_mask or mask.bit_count() < min_keep:
            continue
        frag = tuple(sorted(full_coords[i] for i in range(n) if mask & (1 << i)))
        matches = _all_matches(frag, templates, (0, 0, inner_h - 1, inner_w - 1))
        if len(matches) == 1 and matches[0][:3] == target:
            valid.append(frag)
    return valid


def _make_region_spec(
    template_idx: int,
    templates: list[tuple[tuple[int, int], ...]],
    rng: random.Random,
) -> dict | None:
    template = templates[template_idx]
    th, tw = _shape_dims(template)
    max_pad = 3 if max(th, tw) <= 4 else 2
    for _ in range(120):
        pad_t = rng.randint(0, max_pad)
        pad_b = rng.randint(0, max_pad)
        pad_l = rng.randint(0, max_pad)
        pad_r = rng.randint(0, max_pad)
        if pad_t + pad_b + pad_l + pad_r == 0:
            side = rng.randrange(4)
            if side == 0:
                pad_t = 1
            elif side == 1:
                pad_b = 1
            elif side == 2:
                pad_l = 1
            else:
                pad_r = 1

        inner_h = th + pad_t + pad_b
        inner_w = tw + pad_l + pad_r
        full = tuple(sorted((pad_t + r, pad_l + c) for r, c in template))
        candidates = _valid_fragment_subsets(full, template_idx, (pad_t, pad_l), templates, (inner_h, inner_w))
        if candidates:
            fragment = rng.choice(candidates)
            return {
                "template_idx": template_idx,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "outer_h": inner_h + 2,
                "outer_w": inner_w + 2,
                "full": full,
                "fragment": fragment,
            }
    return None


def _template_band_layout(
    templates: list[tuple[tuple[int, int], ...]],
    rng: random.Random,
) -> tuple[list[tuple[int, int]], int, int]:
    max_h = max(_shape_dims(tpl)[0] for tpl in templates)
    top_pad = rng.randint(1, 2)
    bottom_pad = rng.randint(1, 2)
    positions: list[tuple[int, int]] = []
    cursor = 0
    for idx, tpl in enumerate(templates):
        th, tw = _shape_dims(tpl)
        y = top_pad + rng.randint(0, max_h - th)
        positions.append((y, cursor))
        cursor += tw
        if idx != len(templates) - 1:
            cursor += rng.randint(2, 4)
    band_h = top_pad + max_h + bottom_pad
    return positions, band_h, cursor


def _frame_row_layout(region_specs: list[dict], rng: random.Random) -> list[dict]:
    order = list(range(len(region_specs)))
    rng.shuffle(order)
    if len(order) >= 3 and rng.random() < 0.5:
        split = rng.randint(1, len(order) - 1)
        rows = [order[:split], order[split:]]
    else:
        rows = [order]

    layouts = []
    for row in rows:
        row_h = max(region_specs[idx]["outer_h"] for idx in row)
        placements = []
        cursor = 0
        for pos, idx in enumerate(row):
            spec = region_specs[idx]
            local_y = rng.randint(0, row_h - spec["outer_h"])
            placements.append((idx, local_y, cursor))
            cursor += spec["outer_w"]
            if pos != len(row) - 1:
                cursor += rng.randint(2, 4)
        layouts.append({"height": row_h, "width": cursor, "placements": placements})
    return layouts


def _region_template_indices(num_templates: int, num_regions: int, rng: random.Random) -> list[int]:
    picks = rng.sample(range(num_templates), min(num_templates, num_regions))
    while len(picks) < num_regions:
        picks.append(rng.randrange(num_templates))
    rng.shuffle(picks)
    return picks


def _build_pair(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    templates = _choose_templates(rng)
    if templates is None:
        return None

    num_regions = rng.randint(2, 4)
    region_specs = []
    for template_idx in _region_template_indices(len(templates), num_regions, rng):
        spec = _make_region_spec(template_idx, templates, rng)
        if spec is None:
            return None
        region_specs.append(spec)

    template_positions, band_h, band_w = _template_band_layout(templates, rng)
    row_layouts = _frame_row_layout(region_specs, rng)
    usable_w = max(band_w, max(row["width"] for row in row_layouts))
    left_margin = rng.randint(1, 3)
    right_margin = rng.randint(1, 3)
    width = left_margin + usable_w + right_margin

    gap_to_frames = rng.randint(2, 4)
    row_gaps = [rng.randint(2, 4) for _ in range(max(0, len(row_layouts) - 1))]
    bottom_margin = rng.randint(1, 3)
    height = band_h + gap_to_frames + sum(row["height"] for row in row_layouts) + sum(row_gaps) + bottom_margin

    inp = np.full((height, width), BLACK, dtype=int)
    out = inp.copy()

    template_x0 = left_margin + rng.randint(0, usable_w - band_w)
    for tpl, (r0, c0) in zip(templates, template_positions):
        origin = (r0, template_x0 + c0)
        _paint_coords(inp, tpl, origin, GREY)
        _paint_coords(out, tpl, origin, GREY)

    region_colors = rng.sample(FRAGMENT_COLORS, len(region_specs))
    y_cursor = band_h + gap_to_frames
    for row_idx, row in enumerate(row_layouts):
        row_x0 = left_margin + rng.randint(0, usable_w - row["width"])
        for spec_idx, local_y, local_x in row["placements"]:
            spec = region_specs[spec_idx]
            color = region_colors[spec_idx]
            fr0 = y_cursor + local_y
            fc0 = row_x0 + local_x
            _draw_frame(inp, fr0, fc0, spec["outer_h"], spec["outer_w"])
            _draw_frame(out, fr0, fc0, spec["outer_h"], spec["outer_w"])
            inner_origin = (fr0 + 1, fc0 + 1)
            _paint_coords(inp, spec["fragment"], inner_origin, color)
            _paint_coords(out, spec["full"], inner_origin, color)
        y_cursor += row["height"]
        if row_idx < len(row_layouts) - 1:
            y_cursor += row_gaps[row_idx]

    if np.array_equal(inp, out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        pair = _build_pair(rng)
        if pair is None:
            continue
        inp, out = pair
        try:
            if np.array_equal(main(inp), out):
                return inp
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid input")
