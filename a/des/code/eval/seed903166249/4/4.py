import random
from collections import Counter

import numpy as np


SIDE_CCW = ("top", "left", "bottom", "right")
BG_COLOR = 8
TEMPLATE_COLOR = 5
MARKER_POOL = (1, 2, 3, 4, 6, 7, 9)
FILL_POOL = (0, 1, 2, 3, 4, 6, 7, 9)
SIZE_OPTIONS = (3, 5, 7)
SIDES = ("top", "bottom", "left", "right")
GEN_RETRIES = 300
PLACEMENT_RETRIES = 200


def _mode_color(values: np.ndarray) -> int:
    counts = Counter(int(v) for v in np.asarray(values).ravel())
    assert counts, "expected at least one value"
    ranked = counts.most_common(2)
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "expected a unique dominant color"
    return ranked[0][0]


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _size_key(shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    return (min(h, w), max(h, w))


def _rotate_side(side: str, k: int) -> str:
    idx = SIDE_CCW.index(side)
    return SIDE_CCW[(idx + k) % 4]


def _midpoint(shape: tuple[int, int], side: str) -> tuple[int, int]:
    h, w = shape
    if side == "top":
        return 0, w // 2
    if side == "bottom":
        return h - 1, w // 2
    if side == "left":
        return h // 2, 0
    if side == "right":
        return h // 2, w - 1
    raise AssertionError(f"unknown side {side}")


def _hole_side(marker: tuple[int, int], bbox: tuple[int, int, int, int]) -> str:
    mr, mc = marker
    r0, c0, r1, c1 = bbox
    if mr == r0 - 1 and c0 <= mc <= c1:
        return "top"
    if mr == r1 + 1 and c0 <= mc <= c1:
        return "bottom"
    if mc == c0 - 1 and r0 <= mr <= r1:
        return "left"
    if mc == c1 + 1 and r0 <= mr <= r1:
        return "right"
    raise AssertionError("marker must sit on one hole side")


def _analyze_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"
    h, w = g.shape
    bg = _mode_color(g)

    fg_components = _components(g != bg)
    assert fg_components, "expected foreground objects"
    template_coords = max(fg_components, key=len)
    template_mask = np.zeros_like(g, dtype=bool)
    template_mask[template_coords[:, 0], template_coords[:, 1]] = True
    template_color = _mode_color(g[template_mask])

    holes = []
    for coords in _components(g == bg):
        if np.any(coords[:, 0] == 0) or np.any(coords[:, 0] == h - 1) or np.any(coords[:, 1] == 0) or np.any(coords[:, 1] == w - 1):
            continue

        neighbors = set()
        valid_hole = True
        for r, c in coords:
            for nr, nc in _neighbors4(int(r), int(c), h, w):
                if g[nr, nc] == bg:
                    continue
                neighbors.add((nr, nc))
                if not template_mask[nr, nc]:
                    valid_hole = False
        if not valid_hole or not neighbors:
            continue

        r0, c0, r1, c1 = _bbox(coords)
        hole_h = r1 - r0 + 1
        hole_w = c1 - c0 + 1
        if len(coords) != hole_h * hole_w:
            continue

        marker_cells = [(nr, nc) for nr, nc in neighbors if int(g[nr, nc]) != template_color]
        assert len(marker_cells) == 1, "each hole must have exactly one colored template marker"
        marker = marker_cells[0]
        holes.append(
            {
                "bbox": (r0, c0, r1, c1),
                "shape": (hole_h, hole_w),
                "marker_color": int(g[marker]),
                "marker_side": _hole_side(marker, (r0, c0, r1, c1)),
            }
        )

    assert holes, "expected at least one rectangular hole"
    hole_colors = [hole["marker_color"] for hole in holes]
    assert len(set(hole_colors)) == len(hole_colors), "hole marker colors must be unique"

    pieces = []
    piece_mask = (g != bg) & ~template_mask
    for coords in _components(piece_mask):
        r0, c0, r1, c1 = _bbox(coords)
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        assert patch.shape == (r1 - r0 + 1, c1 - c0 + 1)
        assert np.all(patch != bg), "each piece must be a solid rectangle"
        assert len(coords) == patch.size, "piece bbox must be fully occupied"

        marker_hits = []
        for side in SIDES:
            rr, cc = _midpoint(patch.shape, side)
            color = int(patch[rr, cc])
            if color in hole_colors:
                marker_hits.append((side, color))
        assert len(marker_hits) == 1, "each piece must have one edge-midpoint marker"
        marker_side, marker_color = marker_hits[0]
        pieces.append(
            {
                "bbox": (r0, c0, r1, c1),
                "shape": patch.shape,
                "patch": patch.copy(),
                "marker_side": marker_side,
                "marker_color": marker_color,
            }
        )

    assert len(pieces) == len(holes), "piece count must equal hole count"
    return {
        "grid": g,
        "bg": bg,
        "template_mask": template_mask,
        "holes": holes,
        "pieces": pieces,
    }


def main(grid: np.ndarray) -> np.ndarray:
    scene = _analyze_scene(grid)
    g = scene["grid"]
    bg = scene["bg"]
    out = np.full_like(g, bg)
    out[scene["template_mask"]] = g[scene["template_mask"]]

    holes_by_key = {}
    for hole in scene["holes"]:
        key = _size_key(hole["shape"])
        assert key not in holes_by_key, "hole sizes must be unique up to rotation"
        holes_by_key[key] = hole

    pieces_by_key = {}
    for piece in scene["pieces"]:
        key = _size_key(piece["shape"])
        assert key not in pieces_by_key, "piece sizes must be unique up to rotation"
        pieces_by_key[key] = piece

    assert set(holes_by_key) == set(pieces_by_key), "holes and pieces must match by size"

    for key, hole in holes_by_key.items():
        piece = pieces_by_key[key]
        assert piece["marker_color"] == hole["marker_color"], "size-matched piece must share the hole marker color"

        candidates = []
        for k in range(4):
            rotated = np.rot90(piece["patch"], k)
            if rotated.shape != hole["shape"]:
                continue
            if _rotate_side(piece["marker_side"], k) != hole["marker_side"]:
                continue
            rr, cc = _midpoint(rotated.shape, hole["marker_side"])
            if int(rotated[rr, cc]) != hole["marker_color"]:
                continue
            candidates.append(rotated)

        assert len(candidates) == 1, "marker side must determine a unique rotation"
        fill = candidates[0].copy()
        rr, cc = _midpoint(fill.shape, hole["marker_side"])
        fill[rr, cc] = bg

        r0, c0, r1, c1 = hole["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = fill

    return out


def _sample_branches(rng: random.Random, count: int) -> list[str]:
    branches = ["same", "swapped", "square"]
    if count == 3:
        rng.shuffle(branches)
        return branches
    return rng.sample(branches, count)


def _random_base_patch(
    rng: random.Random,
    shape: tuple[int, int],
    palette: tuple[int, ...],
    branch: str,
) -> np.ndarray | None:
    h, w = shape
    for _ in range(80):
        color_count = min(len(palette), rng.randint(2, min(4, len(palette))))
        colors = rng.sample(palette, color_count)
        patch = np.array([[colors[rng.randrange(color_count)] for _ in range(w)] for _ in range(h)], dtype=int)
        if len(set(int(v) for v in patch.ravel())) < 2:
            continue
        if branch == "square":
            if any(np.array_equal(patch, np.rot90(patch, k)) for k in (1, 2, 3)):
                continue
        else:
            if np.array_equal(patch, np.rot90(patch, 2)):
                continue
        return patch
    return None


def _build_template(
    rng: random.Random,
    hole_specs: list[dict],
) -> tuple[np.ndarray, list[dict]]:
    count = len(hole_specs)
    rows = 1 if count == 2 and rng.random() < 0.5 else 2
    cols = (count + rows - 1) // rows
    slots = [(r, c) for r in range(rows) for c in range(cols)]
    rng.shuffle(slots)
    assigned = slots[:count]

    for spec, slot in zip(hole_specs, assigned):
        spec["slot"] = slot

    border = rng.randint(1, 2)
    gap = rng.randint(2, 3)
    row_heights = [0] * rows
    col_widths = [0] * cols
    for spec in hole_specs:
        slot_r, slot_c = spec["slot"]
        row_heights[slot_r] = max(row_heights[slot_r], spec["hole_shape"][0])
        col_widths[slot_c] = max(col_widths[slot_c], spec["hole_shape"][1])

    template_h = 2 * border + sum(row_heights) + gap * (rows - 1)
    template_w = 2 * border + sum(col_widths) + gap * (cols - 1)
    template = np.full((template_h, template_w), TEMPLATE_COLOR, dtype=int)

    row_starts = []
    cur = border
    for height in row_heights:
        row_starts.append(cur)
        cur += height + gap
    col_starts = []
    cur = border
    for width in col_widths:
        col_starts.append(cur)
        cur += width + gap

    for spec in hole_specs:
        hole_h, hole_w = spec["hole_shape"]
        slot_r, slot_c = spec["slot"]
        top = row_starts[slot_r]
        left = col_starts[slot_c]
        max_dr = row_heights[slot_r] - hole_h
        max_dc = col_widths[slot_c] - hole_w
        r0 = top + rng.randint(0, max_dr)
        c0 = left + rng.randint(0, max_dc)
        r1 = r0 + hole_h - 1
        c1 = c0 + hole_w - 1
        template[r0 : r1 + 1, c0 : c1 + 1] = BG_COLOR

        mr, mc = _midpoint((hole_h, hole_w), spec["hole_side"])
        if spec["hole_side"] == "top":
            marker_rc = (r0 - 1, c0 + mc)
        elif spec["hole_side"] == "bottom":
            marker_rc = (r1 + 1, c0 + mc)
        elif spec["hole_side"] == "left":
            marker_rc = (r0 + mr, c0 - 1)
        else:
            marker_rc = (r0 + mr, c1 + 1)
        template[marker_rc] = spec["marker_color"]
        spec["hole_bbox"] = (r0, c0, r1, c1)

    return template, hole_specs


def _blocked_after_place(blocked: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(blocked.shape[0], r0 + h + 1)
    cc1 = min(blocked.shape[1], c0 + w + 1)
    blocked[rr0:rr1, cc0:cc1] = True


def _expected_grid(template_grid: np.ndarray, template_pos: tuple[int, int], hole_specs: list[dict]) -> np.ndarray:
    expected = np.full_like(template_grid, BG_COLOR)
    tr0, tc0 = template_pos
    template_h = max(spec["hole_bbox"][2] for spec in hole_specs) + 1
    template_w = max(spec["hole_bbox"][3] for spec in hole_specs) + 1
    expected[tr0 : tr0 + template_h, tc0 : tc0 + template_w] = template_grid[tr0 : tr0 + template_h, tc0 : tc0 + template_w]

    for spec in hole_specs:
        fill = np.rot90(spec["piece_patch"], spec["rotation"]).copy()
        rr, cc = _midpoint(fill.shape, spec["hole_side"])
        fill[rr, cc] = BG_COLOR
        r0, c0, r1, c1 = spec["hole_bbox"]
        expected[tr0 + r0 : tr0 + r1 + 1, tc0 + c0 : tc0 + c1 + 1] = fill
    return expected


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    piece_count = rng.choice((2, 3))
    branches = _sample_branches(rng, piece_count)
    marker_colors = rng.sample(MARKER_POOL, piece_count)
    used_keys = set()
    hole_specs = []

    non_square_keys = [(a, b) for i, a in enumerate(SIZE_OPTIONS) for b in SIZE_OPTIONS[i + 1 :]]
    square_keys = [(s, s) for s in SIZE_OPTIONS]

    for branch, marker_color in zip(branches, marker_colors):
        if branch == "square":
            candidates = [key for key in square_keys if key not in used_keys]
        else:
            candidates = [key for key in non_square_keys if key not in used_keys]
        if not candidates:
            return None
        key = rng.choice(candidates)
        used_keys.add(key)

        if branch == "square":
            hole_shape = key
            rotation = rng.randrange(4)
        else:
            hole_shape = rng.choice((key, key[::-1]))
            rotation = rng.choice((0, 2)) if branch == "same" else rng.choice((1, 3))

        hole_side = rng.choice(SIDES)
        piece_side = _rotate_side(hole_side, -rotation)
        piece_shape = hole_shape if rotation % 2 == 0 else (hole_shape[1], hole_shape[0])

        hole_specs.append(
            {
                "branch": branch,
                "marker_color": marker_color,
                "hole_shape": hole_shape,
                "hole_side": hole_side,
                "piece_shape": piece_shape,
                "piece_side": piece_side,
                "rotation": rotation,
            }
        )

    fill_palette = tuple(color for color in FILL_POOL if color not in marker_colors and color not in {BG_COLOR, TEMPLATE_COLOR})
    for spec in hole_specs:
        base_patch = _random_base_patch(rng, spec["piece_shape"], fill_palette, spec["branch"])
        if base_patch is None:
            return None
        piece_patch = base_patch.copy()
        rr, cc = _midpoint(piece_patch.shape, spec["piece_side"])
        piece_patch[rr, cc] = spec["marker_color"]
        spec["piece_patch"] = piece_patch

    template, hole_specs = _build_template(rng, hole_specs)
    template_h, template_w = template.shape
    max_piece_h = max(spec["piece_shape"][0] for spec in hole_specs)
    max_piece_w = max(spec["piece_shape"][1] for spec in hole_specs)

    top_margin = max_piece_h + rng.randint(2, 4)
    bottom_margin = max_piece_h + rng.randint(2, 4)
    left_margin = max_piece_w + rng.randint(2, 4)
    right_margin = max_piece_w + rng.randint(2, 4)
    grid_h = template_h + top_margin + bottom_margin
    grid_w = template_w + left_margin + right_margin
    if max(grid_h, grid_w) > 40:
        return None

    tr0 = top_margin
    tc0 = left_margin
    grid = np.full((grid_h, grid_w), BG_COLOR, dtype=int)
    grid[tr0 : tr0 + template_h, tc0 : tc0 + template_w] = template

    blocked = np.zeros_like(grid, dtype=bool)
    _blocked_after_place(blocked, tr0, tc0, template_h, template_w)

    for spec in rng.sample(hole_specs, len(hole_specs)):
        patch = spec["piece_patch"]
        ph, pw = patch.shape
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            r0 = rng.randint(1, grid_h - ph - 1)
            c0 = rng.randint(1, grid_w - pw - 1)
            if blocked[r0 : r0 + ph, c0 : c0 + pw].any():
                continue
            grid[r0 : r0 + ph, c0 : c0 + pw] = patch
            _blocked_after_place(blocked, r0, c0, ph, pw)
            placed = True
            break
        if not placed:
            return None

    expected = np.full_like(grid, BG_COLOR)
    expected[tr0 : tr0 + template_h, tc0 : tc0 + template_w] = template
    for spec in hole_specs:
        fill = np.rot90(spec["piece_patch"], spec["rotation"]).copy()
        rr, cc = _midpoint(fill.shape, spec["hole_side"])
        fill[rr, cc] = BG_COLOR
        r0, c0, r1, c1 = spec["hole_bbox"]
        expected[tr0 + r0 : tr0 + r1 + 1, tc0 + c0 : tc0 + c1 + 1] = fill

    try:
        scene = _analyze_scene(grid)
    except AssertionError:
        return None
    if len(scene["holes"]) != piece_count or len(scene["pieces"]) != piece_count:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
