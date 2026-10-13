from __future__ import annotations

import random
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np


def _load_arc_utils():
    path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    spec = spec_from_file_location("_arc_utils_shared", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load arc_utils from {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ARC = _load_arc_utils()
Color = _ARC.Color
bbox = _ARC.bbox
bbox_coords = _ARC.bbox_coords
bbox_h = _ARC.bbox_h
bbox_w = _ARC.bbox_w
components_mask = _ARC.components_mask
crop_bbox = _ARC.crop_bbox
find_components = _ARC.find_components


OBJECT_COLORS = (Color.BLUE, Color.RED, Color.GREEN)
OBJECT_COLOR_SET = set(OBJECT_COLORS)
MIN_GRID_ROWS = 16
MAX_GRID_ROWS = 28
MIN_GRID_COLS = 20
MAX_GRID_COLS = 30
MAX_ATTEMPTS = 400


def _component_mask(coords: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[int(r) - r0, int(c) - c0] = True
    return mask


def _is_frame_component(grid: np.ndarray, comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    mask = _component_mask(comp["coords"], comp["bbox"])
    frame = np.zeros_like(mask)
    frame[0, :] = True
    frame[-1, :] = True
    frame[:, 0] = True
    frame[:, -1] = True
    if not np.array_equal(mask, frame):
        return False
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    interior = patch[1:-1, 1:-1]
    return Color.YELLOW in set(map(int, np.unique(interior))) and Color.BLACK not in set(map(int, np.unique(interior)))


def _bbox_inside(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> bool:
    return outer[0] <= inner[0] <= inner[2] <= outer[2] and outer[1] <= inner[1] <= inner[3] <= outer[3]


def _count_holes_for_component(grid: np.ndarray, comp: dict) -> int:
    color = int(comp["color"])
    local = crop_bbox(grid, comp["bbox"])
    black_comps = components_mask(local == Color.BLACK, connectivity=4)
    holes = 0
    for coords in black_comps:
        touches_edge = any(
            int(r) == 0
            or int(c) == 0
            or int(r) == local.shape[0] - 1
            or int(c) == local.shape[1] - 1
            for r, c in coords
        )
        if touches_edge:
            continue
        enclosed = True
        for r, c in coords:
            rr = int(r)
            cc = int(c)
            for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                if local[nr, nc] != color:
                    enclosed = False
                    break
            if not enclosed:
                break
        if enclosed:
            holes += 1
    return holes


def _analyze_scene(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    comps = find_components(g, bg=Color.BLACK, colors=OBJECT_COLORS, connectivity=4)
    by_color: dict[int, list[dict]] = {}
    for comp in comps:
        by_color.setdefault(int(comp["color"]), []).append(comp)

    records = []
    for color, group in sorted(by_color.items()):
        frames = [comp for comp in group if _is_frame_component(g, comp)]
        assert len(frames) == 1, f"expected one frame for color {color}"
        frame = frames[0]
        fr0, fc0, fr1, fc1 = frame["bbox"]
        interior = (fr0 + 1, fc0 + 1, fr1 - 1, fc1 - 1)

        seed_candidates = [
            comp
            for comp in group
            if comp is not frame and _bbox_inside(comp["bbox"], interior)
        ]
        assert len(seed_candidates) == 1, f"expected one seed for color {color}"
        seed = seed_candidates[0]

        object_candidates = [comp for comp in group if comp is not frame and comp is not seed]
        assert len(object_candidates) == 1, f"expected one left object for color {color}"
        obj = object_candidates[0]

        seed_box = seed["bbox"]
        tile = crop_bbox(g, seed_box)
        assert set(map(int, np.unique(tile))) <= {Color.YELLOW, color}

        frame_patch = g[fr0 : fr1 + 1, fc0 : fc1 + 1]
        assert np.all(frame_patch[0, :] == color)
        assert np.all(frame_patch[-1, :] == color)
        assert np.all(frame_patch[:, 0] == color)
        assert np.all(frame_patch[:, -1] == color)
        inner_patch = frame_patch[1:-1, 1:-1]
        assert set(map(int, np.unique(inner_patch))) <= {Color.YELLOW, color}

        hole_count = _count_holes_for_component(g, obj)
        assert 1 <= hole_count <= 3

        records.append(
            {
                "color": color,
                "object_bbox": obj["bbox"],
                "object_area": int(obj["area"]),
                "holes": hole_count,
                "frame_bbox": frame["bbox"],
                "interior_bbox": interior,
                "seed_bbox": seed_box,
                "tile": tile,
            }
        )

    records.sort(key=lambda rec: rec["object_bbox"][0])
    return records


def _apply_tiling(out: np.ndarray, rec: dict) -> None:
    color = int(rec["color"])
    period = int(rec["holes"])
    ir0, ic0, ir1, ic1 = rec["interior_bbox"]
    sr0, sc0, _sr1, _sc1 = rec["seed_bbox"]
    tile = rec["tile"]
    tile_h, tile_w = tile.shape

    for r in range(ir0, ir1 + 1):
        tr = (r - sr0) % tile_h
        for c in range(ic0, ic1 + 1):
            if out[r, c] != Color.YELLOW:
                continue
            tc = (c - sc0) % period
            if tc < tile_w and tile[tr, tc] == color:
                out[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count each left object's enclosed black holes and use that count as the horizontal tiling period inside the matching right cavity.

    Input structure:
    - Black background surrounds 2 or 3 colored left-side objects and the same number of right-side rectangular cavities.
    - Each left object is a single 4-connected blue/red/green component with 1 to 3 enclosed black holes.
    - Each right cavity is a one-pixel-thick frame in the matching color, filled with yellow inside except for one small same-color seed pattern.

    Rule:
    - For each left object, count the number of black 4-connected hole regions inside that object's bbox that do not touch the bbox boundary.
    - Find the cavity with the same color, take the seed pattern's bbox as the tile, and keep the seed's current top-left as the phase anchor.
    - Repeat that tile through the cavity interior with horizontal period equal to the hole count and vertical period equal to the seed bbox height, painting only yellow cells and leaving the frame and original seed untouched.

    Key insight:
    - The holes are not copied visually; only their count is extracted and reused as a spacing parameter that controls a distant periodic fill.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    records = _analyze_scene(g)
    for rec in records:
        _apply_tiling(out, rec)
    return out


def _count_mask_holes(mask: np.ndarray) -> int:
    black_comps = components_mask(~mask, connectivity=4)
    holes = 0
    for coords in black_comps:
        if any(
            int(r) == 0 or int(c) == 0 or int(r) == mask.shape[0] - 1 or int(c) == mask.shape[1] - 1
            for r, c in coords
        ):
            continue
        holes += 1
    return holes


def _make_object_patch(hole_count: int) -> np.ndarray:
    dims = [(5, 5), (5, 6), (6, 5)]
    for _ in range(200):
        h, w = random.choice(dims)
        canvas = np.zeros((h + 2, w + 2), dtype=int)
        mask = np.zeros_like(canvas, dtype=bool)
        mask[1 : h + 1, 1 : w + 1] = True

        candidates = [(r, c) for r in range(2, h) for c in range(2, w)]
        random.shuffle(candidates)
        holes = []
        for cell in candidates:
            if all(abs(cell[0] - r) + abs(cell[1] - c) > 1 for r, c in holes):
                holes.append(cell)
                if len(holes) == hole_count:
                    break
        if len(holes) != hole_count:
            continue
        for r, c in holes:
            mask[r, c] = False

        irregular = False
        for _ in range(random.randint(1, 2)):
            side = random.choice(("top", "bottom", "left", "right"))
            if side in ("top", "bottom"):
                row = 1 if side == "top" else h
                start = random.randint(2, w - 1)
                length = random.randint(1, min(2, w - start))
                for dc in range(length):
                    mask[row, start + dc] = False
            else:
                col = 1 if side == "left" else w
                start = random.randint(2, h - 1)
                length = random.randint(1, min(2, h - start))
                for dr in range(length):
                    mask[start + dr, col] = False
            irregular = True

        if random.random() < 0.8:
            side = random.choice(("top", "bottom", "left", "right"))
            if side == "top":
                c = random.randint(2, w - 1)
                mask[0, c] = True
            elif side == "bottom":
                c = random.randint(2, w - 1)
                mask[h + 1, c] = True
            elif side == "left":
                r = random.randint(2, h - 1)
                mask[r, 0] = True
            else:
                r = random.randint(2, h - 1)
                mask[r, w + 1] = True
            irregular = True

        if not irregular:
            continue
        if len(components_mask(mask, connectivity=4)) != 1:
            continue
        box = bbox(mask)
        if box is None:
            continue
        cropped = crop_bbox(mask.astype(int), box).astype(bool)
        area = int(cropped.sum())
        if not (15 <= area <= 30):
            continue
        if _count_mask_holes(cropped) != hole_count:
            continue
        return cropped.astype(int)
    raise RuntimeError(f"failed to build object patch with {hole_count} holes")


def _seed_shape_options(period: int) -> list[np.ndarray]:
    shapes = []
    shapes.append(np.array([[1], [1]], dtype=int))
    shapes.append(np.array([[1], [1], [1]], dtype=int))
    if period >= 2:
        shapes.append(np.array([[1, 0], [1, 1]], dtype=int))
        shapes.append(np.array([[1, 1], [0, 1]], dtype=int))
        shapes.append(np.array([[1, 1], [1, 0]], dtype=int))
    return shapes


def _place_patch(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, color: int) -> None:
    rr, cc = patch.shape
    for r in range(rr):
        for c in range(cc):
            if patch[r, c]:
                dst[r0 + r, c0 + c] = color


def _build_expected_from_metadata(grid: np.ndarray, metadata: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for rec in metadata:
        color = int(rec["color"])
        period = int(rec["holes"])
        ir0, ic0, ir1, ic1 = rec["interior_bbox"]
        sr0, sc0, _sr1, _sc1 = rec["seed_bbox"]
        tile = rec["tile"]
        tile_h, tile_w = tile.shape
        for r in range(ir0, ir1 + 1):
            tr = (r - sr0) % tile_h
            for c in range(ic0, ic1 + 1):
                if out[r, c] != Color.YELLOW:
                    continue
                tc = (c - sc0) % period
                if tc < tile_w and tile[tr, tc] == color:
                    out[r, c] = color
    return out


def _validate_generated(grid: np.ndarray, metadata: list[dict]) -> bool:
    parsed = _analyze_scene(grid)
    if len(parsed) != len(metadata):
        return False

    parsed_by_color = {int(rec["color"]): rec for rec in parsed}
    meta_by_color = {int(rec["color"]): rec for rec in metadata}
    if parsed_by_color.keys() != meta_by_color.keys():
        return False

    parsed_holes = []
    changed_total = 0
    for color, rec in meta_by_color.items():
        seen = parsed_by_color[color]
        if int(seen["holes"]) != int(rec["holes"]):
            return False
        if seen["frame_bbox"] != rec["frame_bbox"]:
            return False
        if seen["seed_bbox"] != rec["seed_bbox"]:
            return False
        if seen["tile"].shape[1] > seen["holes"]:
            return False
        ir0, ic0, ir1, ic1 = seen["interior_bbox"]
        if (ic1 - ic0 + 1) < 2 * seen["holes"]:
            return False
        patch_before = grid[ir0 : ir1 + 1, ic0 : ic1 + 1]
        patch_after = main(grid)[ir0 : ir1 + 1, ic0 : ic1 + 1]
        changed_total += int(np.count_nonzero((patch_before == Color.YELLOW) & (patch_after == color)))
        parsed_holes.append(int(seen["holes"]))

    if len(set(parsed_holes)) != len(parsed_holes):
        return False
    obj_order = [rec["color"] for rec in parsed]
    frame_order = [rec["color"] for rec in sorted(parsed, key=lambda rec: rec["frame_bbox"][0])]
    if obj_order != frame_order:
        return False

    expected = _build_expected_from_metadata(grid, metadata)
    if not np.array_equal(expected, main(grid)):
        return False
    if changed_total < len(metadata) * 2:
        return False
    return True


def _sample_seed(period: int) -> np.ndarray:
    options = _seed_shape_options(period)
    return random.choice(options).copy()


def _generate_case() -> tuple[np.ndarray, list[dict]] | None:
    num_objects = random.choice((2, 3))
    hole_counts = random.sample((1, 2, 3), num_objects)
    colors = random.sample(OBJECT_COLORS, num_objects)
    order = list(range(num_objects))
    random.shuffle(order)
    hole_counts = [hole_counts[i] for i in order]
    colors = [colors[i] for i in order]

    pair_specs = []
    for holes, color in zip(hole_counts, colors):
        obj_patch = _make_object_patch(holes)
        seed_patch = _sample_seed(holes)
        seed_h, seed_w = seed_patch.shape

        inner_h_lo = max(4, seed_h + 2)
        inner_h_hi = min(6, seed_h + 3)
        if inner_h_lo > inner_h_hi:
            return None
        inner_h = random.randint(inner_h_lo, inner_h_hi)
        inner_w_lo = max(6, 2 * holes, seed_w + 2)
        inner_w_hi = min(8, inner_w_lo + 2)
        if inner_w_lo > inner_w_hi:
            return None
        inner_w = random.randint(inner_w_lo, inner_w_hi)
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        pair_specs.append(
            {
                "color": color,
                "holes": holes,
                "object_patch": obj_patch,
                "seed_patch": seed_patch,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "outer_h": outer_h,
                "outer_w": outer_w,
            }
        )

    top_margin = random.randint(1, 2)
    bottom_margin = random.randint(1, 2)
    between_gap = random.randint(1, 2)
    band_heights = [max(spec["object_patch"].shape[0], spec["outer_h"]) for spec in pair_specs]
    total_rows = top_margin + bottom_margin + sum(band_heights) + between_gap * (num_objects - 1)
    if not (MIN_GRID_ROWS <= total_rows <= MAX_GRID_ROWS):
        return None

    left_zone = max(spec["object_patch"].shape[1] for spec in pair_specs) + 2
    right_zone = max(spec["outer_w"] for spec in pair_specs) + 2
    gap_cols = random.randint(3, 5)
    total_cols = 1 + left_zone + gap_cols + right_zone + 1
    if not (MIN_GRID_COLS <= total_cols <= MAX_GRID_COLS):
        return None

    grid = np.full((total_rows, total_cols), Color.BLACK, dtype=int)
    metadata = []
    cursor_r = top_margin
    frame_base_c = 1 + left_zone + gap_cols

    for spec, band_h in zip(pair_specs, band_heights):
        color = int(spec["color"])
        obj_patch = spec["object_patch"]
        seed_patch = spec["seed_patch"]
        obj_h, obj_w = obj_patch.shape
        outer_h, outer_w = spec["outer_h"], spec["outer_w"]
        inner_h, inner_w = spec["inner_h"], spec["inner_w"]

        obj_r = cursor_r + random.randint(0, band_h - obj_h)
        obj_c = 1 + random.randint(0, left_zone - obj_w - 1)
        frame_r = cursor_r + random.randint(0, band_h - outer_h)
        frame_c = frame_base_c + random.randint(0, right_zone - outer_w)

        object_cells = obj_patch.astype(bool)
        grid[obj_r : obj_r + obj_h, obj_c : obj_c + obj_w][object_cells] = color

        grid[frame_r : frame_r + outer_h, frame_c : frame_c + outer_w] = color
        grid[frame_r + 1 : frame_r + outer_h - 1, frame_c + 1 : frame_c + outer_w - 1] = Color.YELLOW

        seed_h, seed_w = seed_patch.shape
        seed_r0 = random.randint(1, inner_h - seed_h - 1)
        seed_c0 = random.randint(1, inner_w - seed_w - 1)
        seed_r = frame_r + 1 + seed_r0
        seed_c = frame_c + 1 + seed_c0
        _place_patch(grid, seed_patch, seed_r, seed_c, color)

        metadata.append(
            {
                "color": color,
                "holes": spec["holes"],
                "object_bbox": (obj_r, obj_c, obj_r + obj_h - 1, obj_c + obj_w - 1),
                "frame_bbox": (frame_r, frame_c, frame_r + outer_h - 1, frame_c + outer_w - 1),
                "interior_bbox": (frame_r + 1, frame_c + 1, frame_r + outer_h - 2, frame_c + outer_w - 2),
                "seed_bbox": (seed_r, seed_c, seed_r + seed_h - 1, seed_c + seed_w - 1),
                "tile": np.where(seed_patch == 1, color, Color.YELLOW).astype(int),
            }
        )
        cursor_r += band_h + between_gap

    expected = _build_expected_from_metadata(grid, metadata)
    if np.array_equal(expected, grid):
        return None
    if not _validate_generated(grid, metadata):
        return None
    return grid, metadata


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        result = _generate_case()
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate a valid ARC instance")
