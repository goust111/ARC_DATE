from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GEN_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    if int(connectivity) != 4:
        raise ValueError("only 4-connectivity is used in this task")
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    h, w = mask.shape

    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        stack = [(rr, cc)]
        seen[rr, cc] = True
        coords: list[tuple[int, int]] = []
        while stack:
            cr, cc2 = stack.pop()
            coords.append((cr, cc2))
            for dr, dc in NEIGHBORS4:
                nr, nc = cr + dr, cc2 + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(coords, dtype=int))
    return comps


def find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    comps: list[dict] = []
    for color in sorted(set(map(int, g.reshape(-1))) - {int(bg)}):
        for coords in components_mask(g == color, connectivity=4):
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "area": int(coords.shape[0]),
                    "bbox": bbox_coords(coords),
                }
            )
    return comps


def normalize_shape(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    arr = np.asarray(coords, dtype=int)
    rows = arr[:, 0] - int(arr[:, 0].min())
    cols = arr[:, 1] - int(arr[:, 1].min())
    return tuple(sorted((int(r), int(c)) for r, c in zip(rows, cols, strict=False)))


def sig_area(sig: tuple[tuple[int, int], ...]) -> int:
    return len(sig)


def sig_bbox(sig: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _ in sig]
    cols = [c for _, c in sig]
    return max(rows) + 1, max(cols) + 1


def hole_components(crop: np.ndarray, bg: int) -> list[np.ndarray]:
    h, w = crop.shape
    holes = []
    for coords in components_mask(crop == bg, connectivity=4):
        if any(int(r) in {0, h - 1} or int(c) in {0, w - 1} for r, c in coords):
            continue
        holes.append(coords)
    return holes


def shape_patch(sig: tuple[tuple[int, int], ...], color: int, bg: int) -> np.ndarray:
    h, w = sig_bbox(sig)
    patch = np.full((h, w), int(bg), dtype=int)
    for r, c in sig:
        patch[int(r), int(c)] = int(color)
    return patch


def rect_is_free(reserved: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    hh, ww = reserved.shape
    rr0 = max(0, int(r0) - int(pad))
    cc0 = max(0, int(c0) - int(pad))
    rr1 = min(hh, int(r0) + int(h) + int(pad))
    cc1 = min(ww, int(c0) + int(w) + int(pad))
    return not bool(reserved[rr0:rr1, cc0:cc1].any())


def mark_rect(reserved: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> None:
    hh, ww = reserved.shape
    rr0 = max(0, int(r0) - int(pad))
    cc0 = max(0, int(c0) - int(pad))
    rr1 = min(hh, int(r0) + int(h) + int(pad))
    cc1 = min(ww, int(c0) + int(w) + int(pad))
    reserved[rr0:rr1, cc0:cc1] = True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the unique largest object and use external congruent fragments as color keys for its holes.

    Input structure:
    - The background is the most common color.
    - Exactly one monochrome 4-connected object has the largest area; inside it are
      one or more background holes.
    - Outside that object are several small monochrome 4-connected fragments; some
      are translationally congruent to hole shapes, while others are unrelated noise.

    Rule:
    - Find the unique largest monochrome 4-connected object and crop the output to its
      tight bounding box, preserving only that object and background.
    - Detect each internal background hole of that cropped object.
    - For each hole, look among the external fragments for a piece with exactly the same
      shape up to translation only; if found, fill the whole hole with that fragment's color.
      Holes with no congruent external fragment stay background.

    Key insight:
    - The scattered fragments are not disposable noise: shape equality between a fragment
      and a hole is the key that decides which holes get filled and with which color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    comps = find_components(g, bg=bg)
    assert comps, "expected at least one foreground component"

    max_area = max(int(comp["area"]) for comp in comps)
    largest = [comp for comp in comps if int(comp["area"]) == max_area]
    assert len(largest) == 1, "expected a unique largest object"
    target = largest[0]

    r0, c0, r1, c1 = target["bbox"]
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    color = int(target["color"])
    for rr, cc in target["coords"]:
        out[int(rr) - r0, int(cc) - c0] = color

    ext_shape_to_colors: defaultdict[tuple[tuple[int, int], ...], list[int]] = defaultdict(list)
    for comp in comps:
        if comp is target:
            continue
        sig = normalize_shape(comp["coords"])
        ext_shape_to_colors[sig].append(int(comp["color"]))

    for hole in hole_components(out, bg):
        sig = normalize_shape(hole)
        colors = set(ext_shape_to_colors.get(sig, []))
        assert len(colors) <= 1, "hole match must be color-unique"
        if not colors:
            continue
        fill = colors.pop()
        for rr, cc in hole:
            out[int(rr), int(cc)] = fill

    return out


_SHAPE_ROWS = [
    ("11",),
    ("1", "1"),
    ("111",),
    ("1", "1", "1"),
    ("11", "10"),
    ("11", "01"),
    ("10", "11"),
    ("01", "11"),
    ("11", "11"),
    ("111", "100"),
    ("111", "001"),
    ("100", "111"),
    ("001", "111"),
    ("1111",),
    ("1", "1", "1", "1"),
    ("111", "010"),
    ("010", "111"),
    ("110", "011"),
    ("011", "110"),
    ("10", "11", "01"),
    ("01", "11", "10"),
    ("10", "11", "10"),
    ("01", "11", "01"),
    ("11", "10", "10"),
    ("11", "01", "01"),
    ("111", "110"),
    ("111", "011"),
    ("110", "111"),
    ("011", "111"),
    ("111", "101"),
    ("101", "111"),
    ("10", "11", "11"),
    ("01", "11", "11"),
    ("11", "11", "10"),
    ("11", "11", "01"),
]


def _sig_from_rows(rows: tuple[str, ...]) -> tuple[tuple[int, int], ...]:
    coords = []
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == "1":
                coords.append((r, c))
    return tuple(coords)


SHAPES = []
_seen_sigs = set()
for rows in _SHAPE_ROWS:
    sig = _sig_from_rows(rows)
    if sig in _seen_sigs:
        continue
    _seen_sigs.add(sig)
    SHAPES.append({"sig": sig, "area": sig_area(sig), "bbox": sig_bbox(sig)})


def _shape_group_ok(shapes: list[dict]) -> bool:
    return any(
        s1["area"] == s2["area"] or s1["bbox"] == s2["bbox"]
        for i, s1 in enumerate(shapes)
        for s2 in shapes[i + 1 :]
    )


def _choose_holes(num_holes: int) -> tuple[list[dict], list[dict]]:
    candidates = [shape for shape in SHAPES if 2 <= int(shape["area"]) <= 5]
    for _ in range(200):
        holes = random.sample(candidates, num_holes)
        if not _shape_group_ok(holes):
            continue
        hole_sigs = {shape["sig"] for shape in holes}
        decoys = [
            shape
            for shape in candidates
            if shape["sig"] not in hole_sigs
            and any(shape["area"] == hole["area"] or shape["bbox"] == hole["bbox"] for hole in holes)
        ]
        if decoys:
            return holes, decoys
    raise RuntimeError("failed to choose non-degenerate hole shapes")


def _place_holes(object_h: int, object_w: int, holes: list[dict]) -> list[dict] | None:
    reserved = np.zeros((object_h, object_w), dtype=bool)
    placed = []
    order = holes[:]
    random.shuffle(order)
    for hole in sorted(order, key=lambda item: item["bbox"][0] * item["bbox"][1], reverse=True):
        h, w = hole["bbox"]
        spots = []
        for r0 in range(1, object_h - h):
            for c0 in range(1, object_w - w):
                if rect_is_free(reserved, r0, c0, h, w, pad=1):
                    spots.append((r0, c0))
        if not spots:
            return None
        r0, c0 = random.choice(spots)
        mark_rect(reserved, r0, c0, h, w, pad=1)
        placed.append({"shape": hole, "r0": int(r0), "c0": int(c0)})
    return placed


def _draw_object(object_h: int, object_w: int, object_color: int, bg: int, holes: list[dict]) -> np.ndarray:
    patch = np.full((object_h, object_w), int(object_color), dtype=int)
    for hole in holes:
        for dr, dc in hole["shape"]["sig"]:
            patch[hole["r0"] + int(dr), hole["c0"] + int(dc)] = int(bg)
    return patch


def _choose_colors(bg: int, needed: int) -> list[int]:
    palette = [c for c in range(10) if c != int(bg)]
    if needed <= len(palette):
        return random.sample(palette, needed)
    colors = palette[:]
    while len(colors) < needed:
        colors.append(random.choice(palette))
    random.shuffle(colors)
    return colors


def _place_external_patches(
    grid: np.ndarray,
    object_bbox: tuple[int, int, int, int],
    patches: list[np.ndarray],
) -> list[tuple[int, int]] | None:
    h, w = grid.shape
    bg = get_bg(grid)
    reserved = np.zeros((h, w), dtype=bool)
    r0, c0, r1, c1 = object_bbox
    mark_rect(reserved, r0, c0, r1 - r0 + 1, c1 - c0 + 1, pad=1)
    locs = []
    for patch in sorted(patches, key=lambda arr: int(np.count_nonzero(arr)), reverse=True):
        ph, pw = patch.shape
        spots = []
        for rr in range(0, h - ph + 1):
            for cc in range(0, w - pw + 1):
                if rect_is_free(reserved, rr, cc, ph, pw, pad=1):
                    spots.append((rr, cc))
        if not spots:
            return None
        rr, cc = random.choice(spots)
        locs.append((int(rr), int(cc)))
        mask = patch != bg
        grid[rr : rr + ph, cc : cc + pw][mask] = patch[mask]
        mark_rect(reserved, rr, cc, ph, pw, pad=1)
    return locs


def _build_expected(object_patch: np.ndarray, bg: int, holes: list[dict]) -> np.ndarray:
    out = object_patch.copy()
    for hole in holes:
        if "fill_color" not in hole:
            continue
        fill = int(hole["fill_color"])
        for dr, dc in hole["shape"]["sig"]:
            out[hole["r0"] + int(dr), hole["c0"] + int(dc)] = fill
    return out


def _validate_case(grid: np.ndarray, expected: np.ndarray, meta: dict) -> bool:
    try:
        solved = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(solved, expected):
        return False

    bg = get_bg(grid)
    counts = Counter(map(int, grid.reshape(-1)))
    if counts[bg] != max(counts.values()):
        return False

    comps = find_components(grid, bg=bg)
    if not comps:
        return False
    max_area = max(int(comp["area"]) for comp in comps)
    largest = [comp for comp in comps if int(comp["area"]) == max_area]
    if len(largest) != 1:
        return False
    target = largest[0]
    if tuple(target["bbox"]) != tuple(meta["object_bbox"]):
        return False

    crop = np.full(expected.shape, bg, dtype=int)
    for rr, cc in target["coords"]:
        crop[int(rr) - meta["object_bbox"][0], int(cc) - meta["object_bbox"][1]] = int(target["color"])
    hole_sigs = [normalize_shape(coords) for coords in hole_components(crop, bg)]
    if set(hole_sigs) != {hole["shape"]["sig"] for hole in meta["holes"]}:
        return False

    ext_shape_to_colors: defaultdict[tuple[tuple[int, int], ...], list[int]] = defaultdict(list)
    for comp in comps:
        if comp is target:
            continue
        br0, bc0, br1, bc1 = comp["bbox"]
        or0, oc0, or1, oc1 = meta["object_bbox"]
        if not (br1 < or0 or or1 < br0 or bc1 < oc0 or oc1 < bc0):
            return False
        ext_shape_to_colors[normalize_shape(comp["coords"])].append(int(comp["color"]))

    for hole in meta["holes"]:
        sig = hole["shape"]["sig"]
        colors = set(ext_shape_to_colors.get(sig, []))
        if hole.get("paired", False):
            if colors != {int(hole["fill_color"])}:
                return False
        else:
            if colors:
                return False

    if not any(
        noise["area_or_bbox_match"]
        for noise in meta["noise_shapes"]
    ):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    canvas_h = random.randint(22, 30)
    canvas_w = random.randint(22, 30)
    bg, object_color = random.sample(range(10), 2)

    object_h = random.randint(10, min(17, canvas_h - 4))
    object_w = random.randint(11, min(18, canvas_w - 4))

    num_holes = random.randint(2, 4)
    holes_catalog, decoy_pool = _choose_holes(num_holes)
    placed_holes = _place_holes(object_h, object_w, holes_catalog)
    if placed_holes is None:
        return None

    num_paired = random.randint(1, num_holes - 1)
    paired_indices = set(random.sample(range(num_holes), num_paired))

    fragment_colors = _choose_colors(bg, num_paired + 3)
    object_patch = _draw_object(object_h, object_w, object_color, bg, placed_holes)
    expected_holes = []
    color_cursor = 0
    for idx, hole in enumerate(placed_holes):
        spec = dict(hole)
        if idx in paired_indices:
            spec["paired"] = True
            spec["fill_color"] = int(fragment_colors[color_cursor])
            color_cursor += 1
        else:
            spec["paired"] = False
        expected_holes.append(spec)

    expected = _build_expected(object_patch, bg, expected_holes)

    grid = np.full((canvas_h, canvas_w), int(bg), dtype=int)
    obj_r0 = random.randint(1, canvas_h - object_h - 1)
    obj_c0 = random.randint(1, canvas_w - object_w - 1)
    grid[obj_r0 : obj_r0 + object_h, obj_c0 : obj_c0 + object_w] = object_patch
    object_bbox = (obj_r0, obj_c0, obj_r0 + object_h - 1, obj_c0 + object_w - 1)

    hole_sigs = {hole["shape"]["sig"] for hole in expected_holes}
    mandatory_decoy = random.choice(decoy_pool)
    extra_noise_count = random.randint(0, 2)
    extra_noise_pool = [shape for shape in SHAPES if shape["sig"] not in hole_sigs and shape["sig"] != mandatory_decoy["sig"]]
    if len(extra_noise_pool) < extra_noise_count:
        return None
    noise_shapes = [mandatory_decoy] + random.sample(extra_noise_pool, extra_noise_count)

    patches = []
    noise_meta = []
    for hole in expected_holes:
        if not hole["paired"]:
            continue
        patches.append(shape_patch(hole["shape"]["sig"], hole["fill_color"], bg))

    noise_color_choices = fragment_colors[color_cursor:] + [object_color]
    if not noise_color_choices:
        noise_color_choices = [object_color]
    for shape in noise_shapes:
        noise_color = int(random.choice(noise_color_choices))
        patches.append(shape_patch(shape["sig"], noise_color, bg))
        noise_meta.append(
            {
                "sig": shape["sig"],
                "area_or_bbox_match": any(
                    shape["area"] == hole["shape"]["area"] or shape["bbox"] == hole["shape"]["bbox"]
                    for hole in expected_holes
                ),
            }
        )

    if _place_external_patches(grid, object_bbox, patches) is None:
        return None

    meta = {
        "object_bbox": object_bbox,
        "holes": expected_holes,
        "noise_shapes": noise_meta,
    }
    if not _validate_case(grid, expected, meta):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for hole-key puzzle")
