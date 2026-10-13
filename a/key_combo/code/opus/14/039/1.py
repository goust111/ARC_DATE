from __future__ import annotations

import itertools
import random
from collections import Counter, deque

import numpy as np


BG = 0
GREY = 5
FRAME_COLOR = GREY
DIVIDER_COLOR = GREY
NON_SPECIAL_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

GRID_H_RANGE = (15, 22)
GRID_W_RANGE = (18, 28)
SPRITE_COUNT_RANGE = (4, 6)
RETRY_LIMIT = 400

CATALOG_BOX_H = 4
CATALOG_BOX_W = 4
CATALOG_AREA_RANGE = (3, 7)
MAX_SHAPE_H = 4
MAX_SHAPE_W = 3


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _label_component(mask: np.ndarray, start: tuple[int, int], seen: np.ndarray) -> np.ndarray:
    q = deque([start])
    seen[start] = True
    coords = []
    h, w = mask.shape
    while q:
        r, c = q.popleft()
        coords.append((r, c))
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))
    return np.array(coords, dtype=int)


def find_components(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    colors: list[int] | None = None,
) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    if bg is None:
        bg = get_bg(g)
    if colors is None:
        colors = sorted(set(map(int, g.ravel())) - {int(bg)})
    comps: list[dict] = []
    for color in colors:
        mask = g == int(color)
        seen = np.zeros_like(mask, dtype=bool)
        for r, c in np.argwhere(mask):
            rr = int(r)
            cc = int(c)
            if seen[rr, cc]:
                continue
            coords = _label_component(mask, (rr, cc), seen)
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _normalize_cells(cells: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in cells))


def _mask_to_key(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    return _normalize_cells([tuple(map(int, rc)) for rc in np.argwhere(mask)])


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    h = 1 + max(r for r, _ in shape)
    w = 1 + max(c for _, c in shape)
    return h, w


def _shape_array(shape: tuple[tuple[int, int], ...]) -> np.ndarray:
    h, w = _shape_dims(shape)
    arr = np.zeros((h, w), dtype=bool)
    for r, c in shape:
        arr[r, c] = True
    return arr


def _is_connected(cells: tuple[tuple[int, int], ...]) -> bool:
    cell_set = set(cells)
    q = deque([next(iter(cell_set))])
    seen = {q[0]}
    while q:
        r, c = q.popleft()
        for nb in neighbors4(r, c):
            if nb in cell_set and nb not in seen:
                seen.add(nb)
                q.append(nb)
    return len(seen) == len(cell_set)


def _build_shape_catalog() -> list[tuple[tuple[int, int], ...]]:
    cells = [(r, c) for r in range(CATALOG_BOX_H) for c in range(CATALOG_BOX_W)]
    shapes = set()
    for area in range(CATALOG_AREA_RANGE[0], CATALOG_AREA_RANGE[1] + 1):
        for combo in itertools.combinations(cells, area):
            key = _normalize_cells(combo)
            if not _is_connected(key):
                continue
            h, w = _shape_dims(key)
            if h <= MAX_SHAPE_H and w <= MAX_SHAPE_W:
                shapes.add(key)
    return sorted(shapes, key=lambda s: (len(s), _shape_dims(s), s))


SHAPE_CATALOG = _build_shape_catalog()


def _dihedral_variants(shape: tuple[tuple[int, int], ...]) -> set[tuple[tuple[int, int], ...]]:
    arr = _shape_array(shape)
    variants = set()
    for k in range(4):
        rot = np.rot90(arr, k)
        variants.add(_mask_to_key(rot))
        variants.add(_mask_to_key(np.fliplr(rot)))
    return variants


def _place_shape(grid: np.ndarray, shape: tuple[tuple[int, int], ...], r0: int, c0: int, color: int) -> None:
    for dr, dc in shape:
        grid[r0 + dr, c0 + dc] = int(color)


def _shape_from_component(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    cells = [tuple(map(int, rc)) for rc in coords]
    return _normalize_cells(cells)


def _extract_template(grid: np.ndarray, frame: dict) -> tuple[tuple[tuple[int, int], ...], int]:
    fr0, fc0, fr1, fc1 = frame["bbox"]
    region = grid[fr0 : fr1 + 1, fc0 : fc1 + 1]
    expected = np.zeros(region.shape, dtype=bool)
    expected[-1, :] = True
    expected[:, -1] = True
    assert np.array_equal(region == FRAME_COLOR, expected), "grey frame must be a bottom-right L"

    interior = region[:-1, :-1]
    colors = sorted(set(map(int, interior.ravel())) - {BG})
    assert len(colors) == 1, "frame interior must contain exactly one template color"
    template_color = colors[0]
    template_shape = _mask_to_key(interior == template_color)
    assert 3 <= len(template_shape) <= 6
    return template_shape, template_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop each upper sprite to the divider, but only the sprite matching the L-framed template passes through and changes color.

    Input structure:
    - The background is black.
    - One full-width grey horizontal row acts as a divider.
    - Below the divider sits a grey L-frame whose right column and bottom row define a template area containing one colored 4-connected shape.
    - Above the divider there are several small monochrome 4-connected sprites with disjoint column spans.

    Rule:
    - Extract the template's normalized binary mask from inside the grey L-frame, ignoring its color and any empty margin.
    - Normalize each sprite above the divider by cropping to its own bounding box.
    - Clear the original sprite positions, then drop every sprite straight down without changing columns.
    - A sprite whose normalized mask matches the template passes through the divider to the bottom row and is recolored to the template color.
    - Every non-matching sprite stops immediately above the divider and keeps its original color; the divider, frame, and template remain in place.

    Key insight:
    - The L-frame is a physical filter key rather than a simple marker: exact shape equality decides which falling sprite may penetrate the divider.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert g.ndim == 2

    bg = get_bg(g)
    assert bg == BG, "background is expected to be black"
    divider_rows = [r for r in range(h) if np.all(g[r, :] == DIVIDER_COLOR)]
    assert len(divider_rows) == 1, "expected exactly one full-width grey divider"
    divider_r = divider_rows[0]

    grey_comps = [comp for comp in find_components(g, bg=bg, colors=[FRAME_COLOR]) if comp["bbox"][0] != comp["bbox"][2]]
    frames = [comp for comp in grey_comps if comp["bbox"][2] > divider_r]
    assert len(frames) == 1, "expected exactly one lower grey frame"
    template_shape, template_color = _extract_template(g, frames[0])

    comps = find_components(g, bg=bg, colors=[c for c in sorted(set(map(int, g.ravel())) - {bg, FRAME_COLOR})])
    sprites = [comp for comp in comps if comp["bbox"][2] < divider_r]
    assert 4 <= len(sprites) <= 6, "expected 4-6 sprites above the divider"

    out = g.copy()
    for sprite in sprites:
        coords = sprite["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    matches = 0
    for sprite in sprites:
        shape = _shape_from_component(sprite["coords"])
        r0, c0, r1, _c1 = sprite["bbox"]
        if shape == template_shape:
            shift = h - 1 - r1
            color = template_color
            matches += 1
        else:
            shift = divider_r - 1 - r1
            color = int(sprite["color"])
        assert shift >= 1
        moved = sprite["coords"].copy()
        moved[:, 0] += shift
        out[moved[:, 0], moved[:, 1]] = int(color)

    assert matches == 1, "expected exactly one penetrating sprite"
    return out


def _shape_pool(
    *,
    area_min: int,
    area_max: int,
    width_kind: str | None = None,
    exclude: set[tuple[tuple[int, int], ...]] | None = None,
) -> list[tuple[tuple[int, int], ...]]:
    if exclude is None:
        exclude = set()
    out = []
    for shape in SHAPE_CATALOG:
        if shape in exclude:
            continue
        area = len(shape)
        if not (area_min <= area <= area_max):
            continue
        _h, w = _shape_dims(shape)
        if width_kind == "single" and w != 1:
            continue
        if width_kind == "wide" and w == 1:
            continue
        out.append(shape)
    return out


def _pick_template_shape(rng: random.Random, match_single: bool) -> tuple[tuple[int, int], ...]:
    width_kind = "single" if match_single else "wide"
    pool = _shape_pool(area_min=3, area_max=6, width_kind=width_kind)
    return rng.choice(pool)


def _pick_distractors(
    rng: random.Random,
    template_shape: tuple[tuple[int, int], ...],
    count: int,
) -> list[tuple[tuple[int, int], ...]]:
    used = {template_shape}
    distractors: list[tuple[tuple[int, int], ...]] = []
    match_single = _shape_dims(template_shape)[1] == 1

    required_kinds = ["single", "wide"] if match_single else ["wide", "single"]
    for kind in required_kinds:
        pool = _shape_pool(area_min=3, area_max=7, width_kind=kind, exclude=used)
        if not pool:
            raise RuntimeError("missing distractor pool")
        choice = rng.choice(pool)
        distractors.append(choice)
        used.add(choice)

    variants = [shape for shape in _dihedral_variants(template_shape) if shape not in used and 3 <= len(shape) <= 7 and _shape_dims(shape)[1] <= MAX_SHAPE_W]
    if variants and len(distractors) < count:
        choice = rng.choice(sorted(variants))
        distractors.append(choice)
        used.add(choice)

    while len(distractors) < count:
        same_area = [shape for shape in _shape_pool(area_min=3, area_max=7, exclude=used) if len(shape) == len(template_shape)]
        pool = same_area if same_area and rng.random() < 0.6 else _shape_pool(area_min=3, area_max=7, exclude=used)
        choice = rng.choice(pool)
        distractors.append(choice)
        used.add(choice)

    return distractors


def _sprite_spec(
    rng: random.Random,
    template_shape: tuple[tuple[int, int], ...],
) -> list[dict]:
    sprite_count = rng.randint(*SPRITE_COUNT_RANGE)
    distractor_shapes = _pick_distractors(rng, template_shape, sprite_count - 1)
    shapes = [{"shape": template_shape, "match": True}] + [{"shape": s, "match": False} for s in distractor_shapes]
    rng.shuffle(shapes)
    return shapes


def _build_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray, dict] | None:
    match_single = rng.random() < 0.5
    template_shape = _pick_template_shape(rng, match_single)
    sprite_specs = _sprite_spec(rng, template_shape)

    template_h, template_w = _shape_dims(template_shape)
    inner_h = template_h + rng.randint(0, 1)
    inner_w = template_w + rng.randint(0, 1)
    frame_h = inner_h + 1
    frame_w = inner_w + 1

    sprite_heights = [_shape_dims(spec["shape"])[0] for spec in sprite_specs]
    h_min = max(GRID_H_RANGE[0], max(sprite_heights) + frame_h + 6)
    h_max = GRID_H_RANGE[1]
    if h_min > h_max:
        return None
    h = rng.randint(h_min, h_max)

    divider_min = max(sprite_heights) + 2
    divider_max = h - frame_h - 3
    if divider_min > divider_max:
        return None
    divider_r = rng.randint(divider_min, divider_max)

    frame_r0 = rng.randint(divider_r + 2, h - frame_h)
    frame_c0 = rng.randint(1, 2)
    frame_bbox = (frame_r0, frame_c0, frame_r0 + frame_h - 1, frame_c0 + frame_w - 1)

    used_colors = {FRAME_COLOR}
    template_color = rng.choice(NON_SPECIAL_COLORS)
    used_colors.add(template_color)
    sprite_colors = rng.sample([c for c in NON_SPECIAL_COLORS if c not in used_colors], len(sprite_specs))

    for spec, color in zip(sprite_specs, sprite_colors):
        spec["color"] = int(color)
        spec["h"], spec["w"] = _shape_dims(spec["shape"])

    zone_start = frame_bbox[3] + 3
    widths = [spec["w"] for spec in sprite_specs]
    gaps = [rng.randint(1, 2) for _ in range(len(sprite_specs) - 1)]
    required_w = zone_start + sum(widths) + sum(gaps) + 1
    w_min = max(GRID_W_RANGE[0], required_w)
    w_max = GRID_W_RANGE[1]
    if w_min > w_max:
        return None
    w = rng.randint(w_min, w_max)

    order = sprite_specs[:]
    rng.shuffle(order)
    lead_slack = w - 1 - zone_start - (sum(spec["w"] for spec in order) + sum(gaps))
    if lead_slack < 0:
        return None
    c = zone_start + rng.randint(0, lead_slack)
    for i, spec in enumerate(order):
        spec["col"] = c
        c += spec["w"]
        if i < len(gaps):
            c += gaps[i]

    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)
    inp[divider_r, :] = DIVIDER_COLOR
    out[divider_r, :] = DIVIDER_COLOR

    fr0, fc0, fr1, fc1 = frame_bbox
    inp[fr1, fc0 : fc1 + 1] = FRAME_COLOR
    inp[fr0 : fr1 + 1, fc1] = FRAME_COLOR
    out[fr1, fc0 : fc1 + 1] = FRAME_COLOR
    out[fr0 : fr1 + 1, fc1] = FRAME_COLOR

    template_offset_r = rng.randint(0, inner_h - template_h)
    template_offset_c = rng.randint(0, inner_w - template_w)
    template_top = fr0 + template_offset_r
    template_left = fc0 + template_offset_c
    _place_shape(inp, template_shape, template_top, template_left, template_color)
    _place_shape(out, template_shape, template_top, template_left, template_color)

    for spec in order:
        max_input_top = divider_r - spec["h"] - 1
        if max_input_top < 0:
            return None
        input_top = rng.randint(0, max_input_top)
        while input_top + spec["h"] - 1 >= divider_r - 1:
            input_top = rng.randint(0, max_input_top)
        spec["input_top"] = input_top
        _place_shape(inp, spec["shape"], input_top, spec["col"], spec["color"])

        final_top = h - spec["h"] if spec["match"] else divider_r - spec["h"]
        spec["final_top"] = final_top
        final_color = template_color if spec["match"] else spec["color"]
        _place_shape(out, spec["shape"], final_top, spec["col"], final_color)

    meta = {
        "divider_r": divider_r,
        "frame_bbox": frame_bbox,
        "template_shape": template_shape,
        "template_color": template_color,
        "sprites": order,
    }
    return inp, out, meta


def _semantic_validate(inp: np.ndarray, out: np.ndarray, meta: dict) -> None:
    divider_r = meta["divider_r"]
    sprites = meta["sprites"]
    template_shape = meta["template_shape"]
    template_color = meta["template_color"]

    assert np.array_equal(inp[divider_r, :], np.full(inp.shape[1], DIVIDER_COLOR))
    assert np.array_equal(out[divider_r, :], np.where(out[divider_r, :] == DIVIDER_COLOR, DIVIDER_COLOR, out[divider_r, :]))

    intervals = []
    match_count = 0
    same_width_nonmatch = 0
    template_width = _shape_dims(template_shape)[1]
    for spec in sprites:
        col0 = spec["col"]
        col1 = col0 + spec["w"] - 1
        intervals.append((col0, col1))
        if spec["match"]:
            match_count += 1
            assert spec["color"] != template_color
            assert spec["final_top"] == inp.shape[0] - spec["h"]
        else:
            assert spec["final_top"] == divider_r - spec["h"]
            if spec["w"] == template_width:
                same_width_nonmatch += 1
        assert spec["input_top"] + spec["h"] - 1 < divider_r - 1
        assert spec["final_top"] > spec["input_top"]
        assert _normalize_cells(spec["shape"]) == spec["shape"]

    assert match_count == 1
    assert same_width_nonmatch >= 1, "need a width-based counterexample"
    intervals = sorted(intervals)
    for (_, prev_c1), (next_c0, _) in zip(intervals, intervals[1:]):
        assert prev_c1 < next_c0

    solved = main(inp)
    assert np.array_equal(solved, out), "solver output must match the constructed target"
    assert np.count_nonzero(inp != out) >= sum(spec["area"] if "area" in spec else len(spec["shape"]) for spec in sprites)


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        rng = random.Random()
        case = _build_case(rng)
        if case is None:
            continue
        inp, out, meta = case
        try:
            _semantic_validate(inp, out, meta)
        except AssertionError:
            continue
        return inp
    raise RuntimeError("failed to generate a valid input")
