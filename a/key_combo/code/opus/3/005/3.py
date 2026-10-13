from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7

SPRITE_COLORS = (BLUE, RED, YELLOW, MAGENTA)


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def draw_line(grid: np.ndarray, r: int, c: int, *, end_r: int, end_c: int, color: int) -> None:
    r0 = int(r)
    c0 = int(c)
    r1 = int(end_r)
    c1 = int(end_c)
    dr = 0 if r1 == r0 else (1 if r1 > r0 else -1)
    dc = 0 if c1 == c0 else (1 if c1 > c0 else -1)
    steps = max(abs(r1 - r0), abs(c1 - c0)) + 1
    for i in range(steps):
        grid[r0 + i * dr, c0 + i * dc] = int(color)


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 4:
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        dirs = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _component_dicts(grid: np.ndarray, colors: list[int]) -> list[dict]:
    comps: list[dict] = []
    for color in sorted(colors):
        for coords in components_mask(grid == int(color), connectivity=4):
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    comps.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["color"]))
    return comps


def _mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([[r0, c0]])
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _mask_signature(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(mask.shape) + tuple(int(v) for v in mask.astype(int).ravel())


def _is_l_frame(mask: np.ndarray) -> bool:
    if mask.shape[0] < 3 or mask.shape[1] < 3:
        return False
    expected = np.zeros_like(mask, dtype=bool)
    expected[:, -1] = True
    expected[-1, :] = True
    return np.array_equal(mask, expected)


def _is_rect_frame(mask: np.ndarray) -> bool:
    if mask.shape[0] < 3 or mask.shape[1] < 3:
        return False
    expected = np.zeros_like(mask, dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return np.array_equal(mask, expected)


def _find_l_frame(grid: np.ndarray) -> dict:
    candidates = []
    for comp in _component_dicts(grid, [GREY]):
        if _is_l_frame(_mask_from_coords(comp["coords"])):
            candidates.append(comp)
    assert len(candidates) == 1, "expected a unique grey L frame"
    return candidates[0]


def _find_rect_frame(grid: np.ndarray) -> dict:
    candidates = []
    for comp in _component_dicts(grid, [ORANGE]):
        if _is_rect_frame(_mask_from_coords(comp["coords"])):
            candidates.append(comp)
    assert len(candidates) == 1, "expected a unique orange rectangular frame"
    return candidates[0]


def _find_template(grid: np.ndarray, l_bbox: tuple[int, int, int, int]) -> dict:
    r0, c0, r1, c1 = l_bbox
    candidates = []
    for comp in _component_dicts(grid, [GREEN]):
        cr0, cc0, cr1, cc1 = comp["bbox"]
        if r0 <= cr0 and c0 <= cc0 and cr1 < r1 and cc1 < c1:
            candidates.append(comp)
    assert len(candidates) == 1, "expected a unique template inside the L frame"
    return candidates[0]


def _scaled_index(i: int, src_len: int, dst_len: int) -> int:
    assert src_len >= 2 and dst_len >= 2
    return int(round(int(i) * (dst_len - 1) / (src_len - 1)))


def _stretch_stroke(source_mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    src_h, src_w = source_mask.shape
    dst_h, dst_w = target_shape
    assert src_h >= 2 and src_w >= 2
    assert dst_h >= 2 and dst_w >= 2

    stroke = np.zeros((dst_h, dst_w), dtype=int)
    coords = np.argwhere(source_mask)
    mapped: dict[tuple[int, int], tuple[int, int]] = {}
    src_set = {tuple(map(int, rc)) for rc in coords.tolist()}

    for sr, sc in src_set:
        tr = _scaled_index(sr, src_h, dst_h)
        tc = _scaled_index(sc, src_w, dst_w)
        mapped[(sr, sc)] = (tr, tc)
        stroke[tr, tc] = 1

    for sr, sc in src_set:
        for nr, nc in ((sr + 1, sc), (sr, sc + 1)):
            if (nr, nc) not in src_set:
                continue
            r0, c0 = mapped[(sr, sc)]
            r1, c1 = mapped[(nr, nc)]
            draw_line(stroke, r0, c0, end_r=r1, end_c=c1, color=1)

    return stroke.astype(bool)


def _fill_enclosed(stroke: np.ndarray) -> np.ndarray:
    h, w = stroke.shape
    open_space = ~stroke.astype(bool)
    padded = np.pad(open_space, 1, constant_values=True)
    seen = np.zeros_like(padded, dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h + 2):
        for c in (0, w + 1):
            if padded[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w + 2):
        for r in (0, h + 1):
            if padded[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h + 2 and 0 <= nc < w + 2 and padded[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))

    holes = open_space & ~seen[1 : h + 1, 1 : w + 1]
    return stroke.astype(bool) | holes


def _stretch_mask(source_mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    return _fill_enclosed(_stretch_stroke(source_mask, target_shape))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the unique sprite matching the L-framed template, then stretch it into the hollow rectangle.

    Input structure:
    - Black background contains a grey L frame near a corner, with one green 4-connected template shape inside its enclosed region.
    - Elsewhere there is one orange one-cell-thick hollow rectangle whose interior is empty.
    - Outside both frames sit 3 or 4 small monochrome 4-connected sprites of different colors.

    Rule:
    - Read the green template inside the grey L frame and normalize it by its own bounding box.
    - Normalize every outside sprite the same way and find the unique sprite whose mask exactly matches the template, ignoring color.
    - Erase that matched sprite, erase the grey L frame, and erase the green template.
    - Map the matched sprite bbox corners to the interior corners of the orange rectangle, draw straight segments between mapped endpoints of every 4-neighbor source adjacency, then fill enclosed holes with the matched sprite color.
    - Leave the unmatched sprites and the orange frame unchanged.

    Key insight:
    - The grey L frame is only a shape query: it chooses which outside sprite to use, and that chosen sprite is then stretched into the orange frame by bbox-based line-and-fill remapping.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    l_frame = _find_l_frame(g)
    template = _find_template(g, l_frame["bbox"])
    rect_frame = _find_rect_frame(g)

    sprite_colors = sorted(set(int(v) for v in np.unique(g)) - {bg, GREY, GREEN, ORANGE})
    sprites = _component_dicts(g, sprite_colors)
    assert 3 <= len(sprites) <= 4, "expected 3 or 4 outside sprites"

    template_mask = _mask_from_coords(template["coords"])
    template_sig = _mask_signature(template_mask)
    matches = [sprite for sprite in sprites if _mask_signature(_mask_from_coords(sprite["coords"])) == template_sig]
    assert len(matches) == 1, "expected a unique shape-matching sprite"
    matched = matches[0]

    fr0, fc0, fr1, fc1 = rect_frame["bbox"]
    interior_bbox = (fr0 + 1, fc0 + 1, fr1 - 1, fc1 - 1)
    assert interior_bbox[0] <= interior_bbox[2] and interior_bbox[1] <= interior_bbox[3]
    dst_shape = (interior_bbox[2] - interior_bbox[0] + 1, interior_bbox[3] - interior_bbox[1] + 1)

    stretched = _stretch_mask(_mask_from_coords(matched["coords"]), dst_shape)
    out = g.copy()
    out[l_frame["coords"][:, 0], l_frame["coords"][:, 1]] = bg
    out[template["coords"][:, 0], template["coords"][:, 1]] = bg
    out[matched["coords"][:, 0], matched["coords"][:, 1]] = bg
    interior = out[interior_bbox[0] : interior_bbox[2] + 1, interior_bbox[1] : interior_bbox[3] + 1]
    interior[stretched] = int(matched["color"])
    return out


def _bool_mask(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


def _make_shape(mask: np.ndarray, group: str) -> dict:
    return {
        "mask": mask.copy(),
        "group": group,
        "area": int(mask.sum()),
        "shape": tuple(int(v) for v in mask.shape),
        "sig": _mask_signature(mask),
    }


_RAW_TEMPLATE_GROUPS = {
    "tall5": [
        _bool_mask(["11", "11", "10"]),
        _bool_mask(["11", "11", "01"]),
    ],
    "wide5": [
        _bool_mask(["111", "110"]),
        _bool_mask(["110", "111"]),
    ],
    "square6": [
        _bool_mask(["110", "110", "011"]),
        _bool_mask(["110", "111", "001"]),
        _bool_mask(["110", "111", "010"]),
        _bool_mask(["110", "111", "100"]),
        _bool_mask(["111", "110", "010"]),
        _bool_mask(["111", "110", "100"]),
    ],
}

_RAW_EXTRA_SHAPES = [
    _bool_mask(["10", "11"]),
    _bool_mask(["01", "11"]),
    _bool_mask(["111", "010"]),
    _bool_mask(["10", "11", "10"]),
]

TEMPLATE_GROUPS: dict[str, list[dict]] = {
    name: [_make_shape(mask, name) for mask in masks]
    for name, masks in _RAW_TEMPLATE_GROUPS.items()
}
EXTRA_SHAPES = [_make_shape(mask, "extra") for mask in _RAW_EXTRA_SHAPES]
ALL_SHAPES = [shape for shapes in TEMPLATE_GROUPS.values() for shape in shapes] + EXTRA_SHAPES


def _reserve_bbox(blocked: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> None:
    h, w = blocked.shape
    r0, c0, r1, c1 = bbox
    blocked[max(0, r0 - pad) : min(h, r1 + pad + 1), max(0, c0 - pad) : min(w, c1 + pad + 1)] = True


def _bbox_positions(
    blocked: np.ndarray,
    height: int,
    width: int,
    *,
    row_min: int = 0,
    col_min: int = 0,
) -> list[tuple[int, int]]:
    h, w = blocked.shape
    positions: list[tuple[int, int]] = []
    for r in range(row_min, h - height + 1):
        for c in range(col_min, w - width + 1):
            if not blocked[r : r + height, c : c + width].any():
                positions.append((r, c))
    return positions


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top_left: tuple[int, int], color: int) -> np.ndarray:
    r0, c0 = top_left
    h, w = mask.shape
    region = grid[r0 : r0 + h, c0 : c0 + w]
    region[mask] = int(color)
    coords = np.argwhere(mask)
    return coords + np.array([[r0, c0]])


def _draw_l_frame(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    return np.argwhere(grid == GREY)


def _draw_rect_frame(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0] = ORANGE
    grid[r0 : r1 + 1, c1] = ORANGE
    grid[r0, c0 : c1 + 1] = ORANGE
    grid[r1, c0 : c1 + 1] = ORANGE
    mask = np.zeros_like(grid, dtype=bool)
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    return np.argwhere(mask)


def _build_expected(case: dict) -> np.ndarray:
    out = case["input"].copy()
    out[case["l_coords"][:, 0], case["l_coords"][:, 1]] = BG
    out[case["template_coords"][:, 0], case["template_coords"][:, 1]] = BG
    out[case["matched"]["coords"][:, 0], case["matched"]["coords"][:, 1]] = BG
    ir0, ic0, ir1, ic1 = case["rect_interior_bbox"]
    stretched = _stretch_mask(case["matched"]["mask"], (ir1 - ir0 + 1, ic1 - ic0 + 1))
    out[ir0 : ir1 + 1, ic0 : ic1 + 1][stretched] = int(case["matched"]["color"])
    return out


def _choose_shapes(rng: random.Random, sprite_count: int) -> tuple[dict, list[dict]]:
    group_name = rng.choice(list(TEMPLATE_GROUPS))
    template = rng.choice(TEMPLATE_GROUPS[group_name])
    used = {template["sig"]}

    same_bbox_choices = [shape for shape in TEMPLATE_GROUPS[group_name] if shape["sig"] != template["sig"]]
    same_bbox = rng.choice(same_bbox_choices)
    distractors = [same_bbox]
    used.add(same_bbox["sig"])

    diff_area_choices = [shape for shape in ALL_SHAPES if shape["sig"] not in used and shape["area"] != template["area"]]
    diff_area = rng.choice(diff_area_choices)
    distractors.append(diff_area)
    used.add(diff_area["sig"])

    if sprite_count == 4:
        if template["area"] == 5:
            extra_pool = [
                shape
                for shape in ALL_SHAPES
                if shape["sig"] not in used and shape["area"] == template["area"] and shape["shape"] != template["shape"]
            ]
        else:
            extra_pool = [shape for shape in ALL_SHAPES if shape["sig"] not in used and shape["area"] == template["area"]]
        if not extra_pool:
            extra_pool = [shape for shape in ALL_SHAPES if shape["sig"] not in used]
        distractors.append(rng.choice(extra_pool))

    return template, distractors


def _generate_case(rng: random.Random) -> dict:
    sprite_count = rng.choice([3, 4])
    grid_h = rng.randint(16, 19)
    grid_w = rng.randint(16, 19)
    grid = np.zeros((grid_h, grid_w), dtype=int)

    template, distractors = _choose_shapes(rng, sprite_count)
    template_mask = template["mask"]
    src_h, src_w = template_mask.shape

    l_h = rng.randint(max(6, src_h + 3), 7)
    l_w = rng.randint(max(6, src_w + 3), 7)
    l_bbox = (0, 0, l_h - 1, l_w - 1)
    l_coords = _draw_l_frame(grid, l_bbox)

    t_r = rng.randint(1, l_h - src_h - 2)
    t_c = rng.randint(1, l_w - src_w - 2)
    template_coords = _paint_mask(grid, template_mask, (t_r, t_c), GREEN)

    blocked = np.zeros_like(grid, dtype=bool)
    _reserve_bbox(blocked, l_bbox, pad=1)

    rect_h = rng.randint(6, 7)
    rect_w = rng.randint(6, 7)
    rect_positions = _bbox_positions(blocked, rect_h, rect_w, row_min=l_h + 1, col_min=l_w + 1)
    if not rect_positions:
        raise ValueError("no room for rectangle frame")
    rect_top_left = rng.choice(rect_positions)
    rect_bbox = (
        rect_top_left[0],
        rect_top_left[1],
        rect_top_left[0] + rect_h - 1,
        rect_top_left[1] + rect_w - 1,
    )
    rect_coords = _draw_rect_frame(grid, rect_bbox)
    _reserve_bbox(blocked, rect_bbox, pad=1)

    rect_interior_bbox = (rect_bbox[0] + 1, rect_bbox[1] + 1, rect_bbox[2] - 1, rect_bbox[3] - 1)
    rect_interior_shape = (
        rect_interior_bbox[2] - rect_interior_bbox[0] + 1,
        rect_interior_bbox[3] - rect_interior_bbox[1] + 1,
    )
    stroke = _stretch_stroke(template_mask, rect_interior_shape)
    filled = _fill_enclosed(stroke)
    if rect_interior_shape == template["shape"] or not np.any(filled & ~stroke):
        raise ValueError("degenerate stretch target")

    palette = rng.sample(list(SPRITE_COLORS), k=sprite_count)
    matched_color = rng.choice(palette)
    remaining_colors = [color for color in palette if color != matched_color]
    rng.shuffle(remaining_colors)

    sprite_specs = [{"mask": template_mask, "matched": True, "color": matched_color}]
    for color, distractor in zip(remaining_colors, distractors):
        sprite_specs.append({"mask": distractor["mask"], "matched": False, "color": color})
    rng.shuffle(sprite_specs)

    sprites: list[dict] = []
    for spec in sprite_specs:
        mask = spec["mask"]
        h, w = mask.shape
        positions = _bbox_positions(blocked, h, w)
        if not positions:
            raise ValueError("no room for sprite")
        top_left = rng.choice(positions)
        coords = _paint_mask(grid, mask, top_left, spec["color"])
        bbox = (top_left[0], top_left[1], top_left[0] + h - 1, top_left[1] + w - 1)
        _reserve_bbox(blocked, bbox, pad=1)
        sprites.append(
            {
                "mask": mask.copy(),
                "sig": _mask_signature(mask),
                "area": int(mask.sum()),
                "shape": tuple(int(v) for v in mask.shape),
                "color": int(spec["color"]),
                "matched": bool(spec["matched"]),
                "coords": coords,
                "bbox": bbox,
            }
        )

    matched = next(sprite for sprite in sprites if sprite["matched"])
    return {
        "input": grid,
        "l_bbox": l_bbox,
        "l_coords": l_coords,
        "template_mask": template_mask.copy(),
        "template_coords": template_coords,
        "rect_bbox": rect_bbox,
        "rect_coords": rect_coords,
        "rect_interior_bbox": rect_interior_bbox,
        "rect_interior_shape": rect_interior_shape,
        "sprites": sprites,
        "matched": matched,
    }


def _validate_case(case: dict) -> None:
    inp = case["input"]
    out = _build_expected(case)
    assert not np.array_equal(inp, out), "input and output must differ"

    detected_l = _find_l_frame(inp)
    detected_t = _find_template(inp, detected_l["bbox"])
    detected_r = _find_rect_frame(inp)
    assert detected_l["bbox"] == case["l_bbox"]
    assert detected_r["bbox"] == case["rect_bbox"]
    assert _mask_signature(_mask_from_coords(detected_t["coords"])) == _mask_signature(case["template_mask"])

    sprite_colors = sorted(set(int(v) for v in np.unique(inp)) - {BG, GREY, GREEN, ORANGE})
    sprites = _component_dicts(inp, sprite_colors)
    assert len(sprites) == len(case["sprites"])

    template_sig = _mask_signature(case["template_mask"])
    matched_by_shape = [sprite for sprite in sprites if _mask_signature(_mask_from_coords(sprite["coords"])) == template_sig]
    assert len(matched_by_shape) == 1, "shape match must be unique"

    same_area_count = sum(sprite["area"] == int(case["template_mask"].sum()) for sprite in sprites)
    same_bbox_count = sum(_mask_from_coords(sprite["coords"]).shape == case["template_mask"].shape for sprite in sprites)
    assert same_area_count >= 2, "generator must defeat area-only shortcut"
    assert same_bbox_count >= 2, "generator must defeat bbox-only shortcut"

    stroke = _stretch_stroke(case["matched"]["mask"], case["rect_interior_shape"])
    filled = _fill_enclosed(stroke)
    assert case["rect_interior_shape"] != case["matched"]["shape"], "stretch must change bbox scale"
    assert np.any(filled & ~stroke), "fill step must matter"

    produced = main(inp)
    assert np.array_equal(produced, out), "main() must follow semantic generator output"

    matched = case["matched"]
    assert np.all(produced[matched["coords"][:, 0], matched["coords"][:, 1]] == BG), "matched sprite must be erased"
    for sprite in case["sprites"]:
        if sprite["matched"]:
            continue
        assert np.all(produced[sprite["coords"][:, 0], sprite["coords"][:, 1]] == sprite["color"])
    assert np.all(produced[case["l_coords"][:, 0], case["l_coords"][:, 1]] == BG)
    assert np.all(produced[case["template_coords"][:, 0], case["template_coords"][:, 1]] == BG)
    assert np.all(produced[case["rect_coords"][:, 0], case["rect_coords"][:, 1]] == ORANGE)


def generate_input() -> np.ndarray:
    rng = random.Random()
    last_error: Exception | None = None
    for _ in range(500):
        try:
            case = _generate_case(rng)
            _validate_case(case)
            return case["input"]
        except Exception as exc:  # noqa: PERF203
            last_error = exc
    raise RuntimeError(f"failed to generate a valid puzzle after retries: {last_error}")
