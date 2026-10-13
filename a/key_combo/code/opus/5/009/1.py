from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
BLUE = 1
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
TOP_COLORS = (GREEN, YELLOW, ORANGE)

MAX_GEN_ATTEMPTS = 600


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _coords_to_bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _nonzero_components(grid: np.ndarray) -> list[dict]:
    mask = grid != BG
    comps = []
    for coords in _components_from_mask(mask):
        bbox = _coords_to_bbox(coords)
        colors = {int(grid[r, c]) for r, c in coords}
        comps.append({"coords": coords, "bbox": bbox, "colors": colors})
    return comps


def _normalize_coords(coords: np.ndarray | list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    arr = np.array(list(coords), dtype=int)
    r0, c0 = arr.min(axis=0)
    arr = arr - np.array([r0, c0], dtype=int)
    pts = sorted((int(r), int(c)) for r, c in arr)
    return tuple(pts)


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    return max(rows) + 1, max(cols) + 1


def _shape_to_mask(shape: tuple[tuple[int, int], ...]) -> np.ndarray:
    h, w = _shape_dims(shape)
    out = np.zeros((h, w), dtype=bool)
    for r, c in shape:
        out[r, c] = True
    return out


def _paint_shape(patch: np.ndarray, shape: tuple[tuple[int, int], ...], r0: int, c0: int, color: int) -> None:
    for r, c in shape:
        patch[r0 + r, c0 + c] = int(color)


def _random_connected_shape(preferred_area: int | None = None) -> tuple[tuple[int, int], ...]:
    for _ in range(300):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        max_area = min(6, h * w - 1)
        if max_area < 3:
            continue
        if preferred_area is None:
            target_area = random.randint(3, max_area)
        else:
            if not (3 <= preferred_area <= max_area):
                continue
            target_area = int(preferred_area)

        cells = {(random.randrange(h), random.randrange(w))}
        while len(cells) < target_area:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != target_area:
            continue

        shape = _normalize_coords(cells)
        sh, sw = _shape_dims(shape)
        if sh < 2 or sw < 2:
            continue
        if len(shape) == sh * sw:
            continue
        return shape
    raise RuntimeError("shape generation failed")


def _random_unique_shape(forbidden: set[tuple[tuple[int, int], ...]], preferred_area: int | None = None) -> tuple[tuple[int, int], ...]:
    for area in ([preferred_area] if preferred_area is not None else []) + [None]:
        for _ in range(300):
            shape = _random_connected_shape(preferred_area=area)
            if shape not in forbidden:
                return shape
    raise RuntimeError("unique shape generation failed")


def _build_object_patch(
    match_shape: tuple[tuple[int, int], ...],
    other_shape: tuple[tuple[int, int], ...],
    color: int,
    orientation: str,
    matched_side: str,
) -> np.ndarray | None:
    if orientation == "vertical":
        left_shape = match_shape if matched_side == "left" else other_shape
        right_shape = other_shape if matched_side == "left" else match_shape
        lh, lw = _shape_dims(left_shape)
        rh, rw = _shape_dims(right_shape)
        width = lw + 1 + rw
        min_height = max(lh, rh)
        if min_height > width - 1:
            return None
        height = random.randint(min_height, width - 1)
        patch = np.zeros((height, width), dtype=int)
        patch[:, lw] = GREY
        _paint_shape(patch, left_shape, random.randint(0, height - lh), 0, color)
        _paint_shape(patch, right_shape, random.randint(0, height - rh), lw + 1, color)
        return patch

    top_shape = match_shape if matched_side == "top" else other_shape
    bot_shape = other_shape if matched_side == "top" else match_shape
    th, tw = _shape_dims(top_shape)
    bh, bw = _shape_dims(bot_shape)
    height = th + 1 + bh
    min_width = max(tw, bw)
    if min_width > height - 1:
        return None
    width = random.randint(min_width, height - 1)
    patch = np.zeros((height, width), dtype=int)
    patch[th, :] = GREY
    _paint_shape(patch, top_shape, 0, random.randint(0, width - tw), color)
    _paint_shape(patch, bot_shape, th + 1, random.randint(0, width - bw), color)
    return patch


def _build_template_patch(shape: tuple[tuple[int, int], ...]) -> np.ndarray:
    sh, sw = _shape_dims(shape)
    rect_h = random.randint(max(5, sh + 2), 7)
    rect_w = random.randint(max(5, sw + 2), 7)
    r0 = random.randint(1, rect_h - sh - 1)
    c0 = random.randint(1, rect_w - sw - 1)
    patch = np.full((rect_h, rect_w), BLUE, dtype=int)
    for r, c in shape:
        patch[r0 + r, c0 + c] = BG
    return patch


def _line_positions(lengths: list[int], total: int, edge_min: int = 1, gap_min: int = 2) -> list[int]:
    bins = [edge_min] + [gap_min] * (len(lengths) - 1) + [edge_min]
    extra = total - (sum(lengths) + sum(bins))
    for _ in range(extra):
        bins[random.randrange(len(bins))] += 1
    pos = []
    x = bins[0]
    for i, length in enumerate(lengths):
        pos.append(x)
        x += length
        if i < len(lengths) - 1:
            x += bins[i + 1]
    return pos


def _analyze_object(grid: np.ndarray, comp: dict) -> dict:
    r0, c0, r1, c1 = comp["bbox"]
    patch = _crop(grid, comp["bbox"])
    colors = {int(v) for v in patch.flatten()} - {BG, GREY}
    assert len(colors) == 1
    color = int(next(iter(colors)))

    full_rows = [r for r in range(patch.shape[0]) if np.all(patch[r, :] == GREY)]
    full_cols = [c for c in range(patch.shape[1]) if np.all(patch[:, c] == GREY)]
    assert (len(full_rows), len(full_cols)) in {(1, 0), (0, 1)}

    if full_cols:
        assert patch.shape[0] < patch.shape[1]
        crack_rel = [(r, full_cols[0]) for r in range(patch.shape[0])]
    else:
        assert patch.shape[1] < patch.shape[0]
        crack_rel = [(full_rows[0], c) for c in range(patch.shape[1])]

    piece_comps = _components_from_mask(patch == color)
    assert len(piece_comps) == 2
    pieces = []
    for coords in piece_comps:
        norm = _normalize_coords(coords)
        pieces.append(
            {
                "norm": norm,
                "abs_coords": [(r0 + int(r), c0 + int(c)) for r, c in coords],
            }
        )
    assert pieces[0]["norm"] != pieces[1]["norm"]
    return {
        "bbox": comp["bbox"],
        "color": color,
        "crack_abs": [(r0 + r, c0 + c) for r, c in crack_rel],
        "pieces": pieces,
    }


def _analyze_template(grid: np.ndarray, comp: dict) -> dict:
    r0, c0, r1, c1 = comp["bbox"]
    patch = _crop(grid, comp["bbox"])
    assert np.all(patch[0, :] == BLUE)
    assert np.all(patch[-1, :] == BLUE)
    assert np.all(patch[:, 0] == BLUE)
    assert np.all(patch[:, -1] == BLUE)
    gap_coords = np.argwhere(patch == BG)
    assert gap_coords.size > 0
    gap_bbox = _coords_to_bbox(gap_coords)
    assert gap_bbox[0] > 0 and gap_bbox[1] > 0
    assert gap_bbox[2] < patch.shape[0] - 1 and gap_bbox[3] < patch.shape[1] - 1
    return {
        "bbox": comp["bbox"],
        "gap_norm": _normalize_coords(gap_coords),
        "gap_abs": [(r0 + int(r), c0 + int(c)) for r, c in gap_coords],
    }


def _analyze_grid(grid: np.ndarray) -> dict:
    comps = _nonzero_components(grid)
    objects = []
    templates = []
    for comp in comps:
        if GREY in comp["colors"]:
            objects.append(_analyze_object(grid, comp))
        elif comp["colors"] == {BLUE}:
            templates.append(_analyze_template(grid, comp))
        else:
            raise AssertionError("unexpected component palette")

    assert len(objects) == 3
    assert len(templates) == 3
    gap_shapes = [tpl["gap_norm"] for tpl in templates]
    assert len(set(gap_shapes)) == 3

    matches = []
    matched_piece_keys = set()
    matched_by_object: dict[int, int] = {}
    for ti, tpl in enumerate(templates):
        candidates = []
        for oi, obj in enumerate(objects):
            for pi, piece in enumerate(obj["pieces"]):
                if piece["norm"] == tpl["gap_norm"]:
                    candidates.append((oi, pi))
        assert len(candidates) == 1
        oi, pi = candidates[0]
        assert (oi, pi) not in matched_piece_keys
        matched_piece_keys.add((oi, pi))
        assert oi not in matched_by_object
        matched_by_object[oi] = pi
        matches.append({"template_index": ti, "object_index": oi, "piece_index": pi})

    assert len(matches) == 3
    assert len(matched_by_object) == 3
    gap_shape_set = set(gap_shapes)
    for oi, obj in enumerate(objects):
        assert oi in matched_by_object
        matched_pi = matched_by_object[oi]
        other_pi = 1 - matched_pi
        assert obj["pieces"][other_pi]["norm"] not in gap_shape_set

    upper_bottom = max(obj["bbox"][2] for obj in objects)
    lower_top = min(tpl["bbox"][0] for tpl in templates)
    assert 1 <= lower_top - upper_bottom - 1 <= 2
    return {"objects": objects, "templates": templates, "matches": matches}


def _apply_solution(grid: np.ndarray, analysis: dict) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    matched_piece_by_object = {
        match["object_index"]: match["piece_index"] for match in analysis["matches"]
    }

    for oi, obj in enumerate(analysis["objects"]):
        for r, c in obj["crack_abs"]:
            out[r, c] = BG
        piece = obj["pieces"][matched_piece_by_object[oi]]
        for r, c in piece["abs_coords"]:
            out[r, c] = BG

    for match in analysis["matches"]:
        obj = analysis["objects"][match["object_index"]]
        tpl = analysis["templates"][match["template_index"]]
        for r, c in tpl["gap_abs"]:
            out[r, c] = obj["color"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split each cracked top object into two hidden subshapes, then fill the matching lower template gap with the unique complementary half.

    Input structure:
    - Black background with three upper multicolor objects and three lower blue template rectangles, separated by 1-2 blank rows.
    - Each upper object has one body color (green, yellow, or orange) plus a full-span grey crack line across its bounding box short axis.
    - Each lower blue rectangle contains one irregular internal background gap whose normalized shape matches exactly one hidden subshape from the upper region.

    Rule:
    - Find each upper connected object that contains grey, detect its full grey crack row or column, and remove that crack from consideration.
    - Split the remaining body-colored pixels into exactly two 4-connected subshapes and normalize each subshape to its own tight bounding box.
    - For each lower blue rectangle, extract all background pixels inside the rectangle bbox, normalize that gap shape, and match it against the upper subshapes without rotation or reflection.
    - Fill every matched gap with the matched subshape's original body color; in the upper region erase the grey crack and the matched half, while leaving the unmatched half in place.

    Key insight:
    - The fillable shapes are not visible until the grey fracture line is removed; only after the split does one half from each object become a unique exact match for one lower gap.
    """
    g = np.asarray(grid, dtype=int)
    analysis = _analyze_grid(g)
    return _apply_solution(g, analysis)


def _semantic_check(grid: np.ndarray) -> bool:
    try:
        analysis = _analyze_grid(grid)
        out = _apply_solution(grid, analysis)
        if not np.array_equal(out, main(grid)):
            return False
        if grid.ndim != 2 or out.ndim != 2:
            return False
        if max(grid.shape) > 30 or max(out.shape) > 30:
            return False
        orientations = []
        for obj in analysis["objects"]:
            r0, c0, r1, c1 = obj["bbox"]
            h, w = r1 - r0 + 1, c1 - c0 + 1
            orientations.append("vertical" if h < w else "horizontal")
        if "vertical" not in orientations or "horizontal" not in orientations:
            return False
        return True
    except AssertionError:
        return False


def _generate_case() -> np.ndarray | None:
    colors = list(TOP_COLORS)
    random.shuffle(colors)

    forbidden: set[tuple[tuple[int, int], ...]] = set()
    match_shapes = []
    for _ in range(3):
        shape = _random_unique_shape(forbidden)
        forbidden.add(shape)
        match_shapes.append(shape)

    other_shapes = []
    for shape in match_shapes:
        preferred_area = len(shape) if random.random() < 0.65 else None
        other = _random_unique_shape(forbidden, preferred_area=preferred_area)
        forbidden.add(other)
        other_shapes.append(other)

    orientations = ["vertical", "horizontal", random.choice(["vertical", "horizontal"])]
    random.shuffle(orientations)

    object_patches = []
    for color, match_shape, other_shape, orientation in zip(colors, match_shapes, other_shapes, orientations):
        side = random.choice(["left", "right"]) if orientation == "vertical" else random.choice(["top", "bottom"])
        patch = _build_object_patch(match_shape, other_shape, color, orientation, side)
        if patch is None:
            return None
        object_patches.append(patch)

    template_patches = [_build_template_patch(shape) for shape in match_shapes]

    random.shuffle(object_patches)
    random.shuffle(template_patches)

    top_widths = [patch.shape[1] for patch in object_patches]
    bot_widths = [patch.shape[1] for patch in template_patches]
    min_w_top = sum(top_widths) + 2 * (len(top_widths) - 1) + 2
    min_w_bot = sum(bot_widths) + 2 * (len(bot_widths) - 1) + 2
    min_w = max(min_w_top, min_w_bot)
    if min_w > 30:
        return None
    width = random.randint(min_w, min(30, min_w + 4))

    top_band_h = max(patch.shape[0] for patch in object_patches)
    bot_band_h = max(patch.shape[0] for patch in template_patches)
    sep = random.randint(1, 2)
    top_margin = random.randint(1, 3)
    bottom_margin = random.randint(1, 3)
    height = top_margin + top_band_h + sep + bot_band_h + bottom_margin
    if height > 30:
        return None

    grid = np.zeros((height, width), dtype=int)
    top_cols = _line_positions(top_widths, width)
    bot_cols = _line_positions(bot_widths, width)

    top_row0 = top_margin
    bot_row0 = top_margin + top_band_h + sep
    for patch, c0 in zip(object_patches, top_cols):
        r0 = top_row0 + random.randint(0, top_band_h - patch.shape[0])
        h, w = patch.shape
        grid[r0 : r0 + h, c0 : c0 + w] = patch
    for patch, c0 in zip(template_patches, bot_cols):
        r0 = bot_row0 + random.randint(0, bot_band_h - patch.shape[0])
        h, w = patch.shape
        grid[r0 : r0 + h, c0 : c0 + w] = patch

    if not _semantic_check(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
