import random
from collections import Counter

import numpy as np


FRAME = 7
BG = 0
NON_FRAME_COLORS = [1, 2, 3, 4, 5, 6, 8, 9]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _touches_edge(coords: np.ndarray, height: int, width: int) -> bool:
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == height - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == width - 1)
    )


def _components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _enclosed_regions(grid: np.ndarray) -> list[dict]:
    height, width = grid.shape
    regions: list[dict] = []
    for coords in _components(grid != FRAME):
        if _touches_edge(coords, height, width):
            continue
        box = _bbox(coords)
        r0, c0, r1, c1 = box
        if coords.shape[0] != (r1 - r0 + 1) * (c1 - c0 + 1):
            continue
        regions.append(
            {
                "bbox": box,
                "window": grid[r0 : r1 + 1, c0 : c1 + 1].copy(),
                "area": int(coords.shape[0]),
            }
        )
    return regions


def _row_major_key(region: dict) -> tuple[int, int]:
    r0, c0, _r1, _c1 = region["bbox"]
    return int(r0), int(c0)


def _split_regions(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    regions = _enclosed_regions(grid)
    assert regions, "expected enclosed rectangular regions"
    min_area = min(region["area"] for region in regions)
    control = [region for region in regions if region["area"] == min_area]
    main = [region for region in regions if region["area"] != min_area]
    assert 4 <= len(control) <= 6, "control table must contain 4-6 cells"
    assert len(main) == len(control), "main regions and control cells must pair 1-to-1"
    return sorted(main, key=_row_major_key), sorted(control, key=_row_major_key)


def _majority_color(values: np.ndarray) -> int:
    return int(Counter(map(int, values.flat)).most_common(1)[0][0])


def _infer_clean_pattern(region: np.ndarray) -> np.ndarray:
    height, width = map(int, region.shape)
    best = None
    best_clean = None
    best_tile_shape = None
    for tile_h in range(2, height + 1):
        if height % tile_h != 0 or height // tile_h < 2:
            continue
        for tile_w in range(2, width + 1):
            if width % tile_w != 0 or width // tile_w < 2:
                continue
            tile = np.zeros((tile_h, tile_w), dtype=int)
            for rr in range(tile_h):
                for cc in range(tile_w):
                    tile[rr, cc] = _majority_color(region[rr::tile_h, cc::tile_w])
            clean = np.tile(tile, (height // tile_h, width // tile_w))
            mismatches = int(np.count_nonzero(clean != region))
            key = (mismatches, tile_h * tile_w, tile_h, tile_w)
            if best is None or key < best:
                best = key
                best_clean = clean
                best_tile_shape = (tile_h, tile_w)
    assert best_clean is not None, "failed to infer a repeating tile"
    assert best_tile_shape is not None
    return best_clean


def _control_color(cell: np.ndarray) -> int:
    colors = {int(v) for v in np.unique(cell) if int(v) != BG}
    assert len(colors) <= 1, "control cells must be empty or contain one marker color"
    return next(iter(colors)) if colors else BG


def _repair_region(region: np.ndarray, marker_color: int) -> np.ndarray:
    clean = _infer_clean_pattern(region)
    out = region.copy()
    noise = region != clean
    if marker_color == BG:
        out[noise] = clean[noise]
    else:
        fix_mask = noise & (region != marker_color)
        out[fix_mask] = clean[fix_mask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Repair framed periodic regions while the control table selects which noisy color deviations survive.

    Input structure:
    - Orange (7) frame lines form closed rectangular compartments on a black background.
    - Larger compartments are main regions containing a small periodic tile repeated across the whole region, with 1-2 corrupted pixels.
    - Smaller enclosed compartments form the control table; each cell is either empty black or contains one colored marker pixel.

    Rule:
    - Find all enclosed rectangular non-orange regions and split them into large main regions and the smallest-area control cells.
    - Sort main regions and control cells by row-major position; pair them in that order.
    - For each main region, infer the smallest repeating tile that appears at least twice on each axis by majority vote per residue class.
    - Repair every pixel that disagrees with the inferred clean pattern, except pixels whose current color matches the paired control marker color; empty control cells repair all noise.
    - Finally clear every control cell interior to black and keep all orange frame lines unchanged.

    Key insight:
    - The periodic tile reveals the intended clean content, but the control table reclassifies one visible wrong color per region as an intentional mark that must remain after repair.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    main_regions, control_cells = _split_regions(g)
    out = g.copy()
    for main_region, control_cell in zip(main_regions, control_cells):
        marker_color = _control_color(control_cell["window"])
        repaired = _repair_region(main_region["window"], marker_color)
        r0, c0, r1, c1 = main_region["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = repaired
        rr0, cc0, rr1, cc1 = control_cell["bbox"]
        out[rr0 : rr1 + 1, cc0 : cc1 + 1] = BG
    return out


def _rectangles_overlap(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    pad: int = 0,
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _choose_control_layout(rng: random.Random, count: int) -> tuple[int, int]:
    if count == 4:
        return rng.choice([(2, 2), (1, 4), (4, 1)])
    if count == 5:
        return rng.choice([(1, 5), (5, 1)])
    if count == 6:
        return rng.choice([(2, 3), (3, 2)])
    raise AssertionError("unsupported control count")


def _sample_tile(rng: random.Random, region_h: int, region_w: int) -> np.ndarray:
    candidates = [2]
    if region_w % 3 == 0 and region_w // 3 >= 2:
        candidates.append(3)
    tile_w = rng.choice(candidates)
    while True:
        tile = np.zeros((2, tile_w), dtype=int)
        fill_colors = rng.sample(NON_FRAME_COLORS, k=rng.randint(2, min(4, len(NON_FRAME_COLORS))))
        for r in range(2):
            for c in range(tile_w):
                if rng.random() < 0.45:
                    tile[r, c] = BG
                else:
                    tile[r, c] = rng.choice(fill_colors)
        if not np.any(tile == BG) or not np.any(tile != BG):
            continue
        if np.array_equal(tile[0], tile[1]):
            continue
        clean = np.tile(tile, (region_h // 2, region_w // tile_w))
        inferred = _infer_clean_pattern(clean)
        if np.array_equal(inferred, clean):
            return clean


def _pick_wrong_color(rng: random.Random, correct_color: int, *, allow_black: bool) -> int:
    pool = [color for color in NON_FRAME_COLORS if color != correct_color]
    if allow_black and correct_color != BG:
        pool.append(BG)
    return int(rng.choice(pool))


def _build_main_region(
    rng: random.Random,
    mode: str,
    count_hint: int,
) -> tuple[np.ndarray, np.ndarray, int, dict]:
    size_pool = [4, 6]
    if count_hint >= 5:
        size_pool = [4, 4, 6]
    if count_hint >= 6:
        size_pool = [4, 4, 4, 6]
    region_h = rng.choice(size_pool)
    region_w = rng.choice(size_pool)
    clean = _sample_tile(rng, region_h, region_w)
    damaged = clean.copy()

    positions = [(r, c) for r in range(region_h) for c in range(region_w)]
    rng.shuffle(positions)

    if mode == "empty":
        noise_count = rng.choice([1, 2])
        control_color = BG
        wrong_colors = []
        for idx in range(noise_count):
            r, c = positions[idx]
            wrong_colors.append(_pick_wrong_color(rng, int(clean[r, c]), allow_black=True))
            damaged[r, c] = wrong_colors[-1]
    elif mode == "preserve_only":
        noise_count = rng.choice([1, 2])
        first_r, first_c = positions[0]
        control_color = _pick_wrong_color(rng, int(clean[first_r, first_c]), allow_black=False)
        for idx in range(noise_count):
            r, c = positions[idx]
            if int(clean[r, c]) == control_color:
                control_color = _pick_wrong_color(rng, int(clean[r, c]), allow_black=False)
            damaged[r, c] = control_color
    elif mode == "selective":
        noise_count = 2
        r0, c0 = positions[0]
        control_color = _pick_wrong_color(rng, int(clean[r0, c0]), allow_black=False)
        damaged[r0, c0] = control_color
        r1, c1 = positions[1]
        other = _pick_wrong_color(rng, int(clean[r1, c1]), allow_black=True)
        if other == control_color:
            other = _pick_wrong_color(rng, int(clean[r1, c1]), allow_black=False)
        damaged[r1, c1] = other
    else:
        raise AssertionError("unknown mode")

    output = _repair_region(damaged, control_color)
    noise_mask = damaged != clean
    repaired_mask = noise_mask & (output == clean)
    preserved_mask = noise_mask & (output == damaged)
    meta = {
        "mode": mode,
        "noise_count": int(np.count_nonzero(noise_mask)),
        "repaired_count": int(np.count_nonzero(repaired_mask)),
        "preserved_count": int(np.count_nonzero(preserved_mask)),
        "control_color": int(control_color),
    }
    return damaged, output, control_color, meta


def _make_region_patch(region: np.ndarray) -> np.ndarray:
    patch = np.full((region.shape[0] + 2, region.shape[1] + 2), FRAME, dtype=int)
    patch[1:-1, 1:-1] = region
    return patch


def _make_control_patch(
    count: int,
    layout: tuple[int, int],
    cell_h: int,
    cell_w: int,
    markers: list[int],
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    rows, cols = layout
    assert rows * cols == count
    patch_h = rows * cell_h + rows + 1
    patch_w = cols * cell_w + cols + 1
    patch = np.full((patch_h, patch_w), FRAME, dtype=int)
    cells: list[tuple[int, int, int, int]] = []
    for idx in range(count):
        rr = idx // cols
        cc = idx % cols
        r0 = 1 + rr * (cell_h + 1)
        c0 = 1 + cc * (cell_w + 1)
        patch[r0 : r0 + cell_h, c0 : c0 + cell_w] = BG
        marker = markers[idx]
        if marker != BG:
            mr = r0 + cell_h // 2
            mc = c0 + cell_w // 2
            patch[mr, mc] = marker
        cells.append((r0, c0, r0 + cell_h - 1, c0 + cell_w - 1))
    return patch, cells


def _place_patches(
    rng: random.Random,
    canvas_h: int,
    canvas_w: int,
    patch_shapes: list[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    placements: list[tuple[int, int] | None] = [None] * len(patch_shapes)
    occupied: list[tuple[int, int, int, int]] = []
    order = sorted(range(len(patch_shapes)), key=lambda idx: patch_shapes[idx][0] * patch_shapes[idx][1], reverse=True)
    for idx in order:
        patch_h, patch_w = patch_shapes[idx]
        if patch_h + 2 > canvas_h or patch_w + 2 > canvas_w:
            return None
        found = None
        for _ in range(300):
            r0 = rng.randint(1, canvas_h - patch_h - 1)
            c0 = rng.randint(1, canvas_w - patch_w - 1)
            box = (r0, c0, r0 + patch_h - 1, c0 + patch_w - 1)
            if any(_rectangles_overlap(box, other, pad=1) for other in occupied):
                continue
            found = (r0, c0)
            occupied.append(box)
            break
        if found is None:
            return None
        placements[idx] = found
    return [placement for placement in placements if placement is not None]


def _assemble_example(rng: random.Random) -> tuple[np.ndarray, np.ndarray, dict] | None:
    main_count = rng.choice([4, 4, 5, 5, 6, 6])
    mode_order = ["empty", "selective", "preserve_only"]
    while len(mode_order) < main_count:
        mode_order.append(rng.choice(["empty", "selective", "preserve_only"]))
    rng.shuffle(mode_order)

    main_inputs: list[np.ndarray] = []
    main_outputs: list[np.ndarray] = []
    markers: list[int] = []
    metas: list[dict] = []
    for mode in mode_order:
        region_in, region_out, marker, meta = _build_main_region(rng, mode, main_count)
        main_inputs.append(region_in)
        main_outputs.append(region_out)
        markers.append(marker)
        metas.append(meta)

    control_layout = _choose_control_layout(rng, main_count)
    cell_h = 2
    if main_count >= 6:
        cell_w = 2
    else:
        cell_w = 2 if (main_count >= 5 and rng.random() < 0.75) else rng.choice([2, 3])
    control_patch, _cells = _make_control_patch(main_count, control_layout, cell_h, cell_w, markers)

    main_patches = [_make_region_patch(region) for region in main_inputs]
    patch_shapes = [patch.shape for patch in main_patches] + [control_patch.shape]
    total_area = sum(shape[0] * shape[1] for shape in patch_shapes)
    area_side = int(np.ceil(np.sqrt(total_area * 2.2)))
    base_min = 27 if main_count >= 6 else 24 if main_count >= 5 else 22
    min_h = max(base_min, max(shape[0] for shape in patch_shapes) + 3, area_side - 2)
    min_w = max(base_min, max(shape[1] for shape in patch_shapes) + 3, area_side - 2)
    if min_h > 29 or min_w > 29:
        return None
    canvas_h = rng.randint(min_h, 29)
    canvas_w = rng.randint(min_w, 29)
    placements = _place_patches(rng, canvas_h, canvas_w, patch_shapes)
    if placements is None:
        return None

    grid = np.zeros((canvas_h, canvas_w), dtype=int)
    expected = grid.copy()

    for patch, (r0, c0) in zip(main_patches, placements[:-1]):
        grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] = patch
        expected[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] = patch

    ctrl_r, ctrl_c = placements[-1]
    grid[ctrl_r : ctrl_r + control_patch.shape[0], ctrl_c : ctrl_c + control_patch.shape[1]] = control_patch
    expected[ctrl_r : ctrl_r + control_patch.shape[0], ctrl_c : ctrl_c + control_patch.shape[1]] = control_patch

    sorted_main = sorted(
        [
            {
                "bbox": (r0 + 1, c0 + 1, r0 + patch.shape[0] - 2, c0 + patch.shape[1] - 2),
                "output": region_out,
            }
            for patch, region_out, (r0, c0) in zip(main_patches, main_outputs, placements[:-1])
        ],
        key=_row_major_key,
    )

    ctrl_regions = sorted(
        _enclosed_regions(grid[ctrl_r : ctrl_r + control_patch.shape[0], ctrl_c : ctrl_c + control_patch.shape[1]]),
        key=_row_major_key,
    )
    if len(ctrl_regions) != main_count:
        return None
    absolute_ctrl_boxes = []
    for region in ctrl_regions:
        r0, c0, r1, c1 = region["bbox"]
        absolute_ctrl_boxes.append((ctrl_r + r0, ctrl_c + c0, ctrl_r + r1, ctrl_c + c1))
    absolute_ctrl_boxes = sorted(absolute_ctrl_boxes)

    for main_region, ctrl_box in zip(sorted_main, absolute_ctrl_boxes):
        r0, c0, r1, c1 = main_region["bbox"]
        expected[r0 : r1 + 1, c0 : c1 + 1] = main_region["output"]
        rr0, cc0, rr1, cc1 = ctrl_box
        expected[rr0 : rr1 + 1, cc0 : cc1 + 1] = BG

    info = {"metas": metas, "count": main_count}
    return grid, expected, info


def _semantic_ok(grid: np.ndarray, expected: np.ndarray, info: dict) -> bool:
    produced = main(grid)
    if not np.array_equal(produced, expected):
        return False

    main_regions, control_cells = _split_regions(grid)
    if len(main_regions) != info["count"] or len(control_cells) != info["count"]:
        return False

    metas = info["metas"]
    if not any(meta["control_color"] == BG for meta in metas):
        return False
    if not any(meta["control_color"] != BG for meta in metas):
        return False
    if not any(meta["repaired_count"] > 0 for meta in metas):
        return False
    if not any(meta["preserved_count"] > 0 for meta in metas):
        return False
    if not any(meta["repaired_count"] > 0 and meta["preserved_count"] > 0 for meta in metas):
        return False

    clean_all = grid.copy()
    for region, control in zip(main_regions, control_cells):
        clean = _infer_clean_pattern(region["window"])
        r0, c0, r1, c1 = region["bbox"]
        clean_all[r0 : r1 + 1, c0 : c1 + 1] = clean
        rr0, cc0, rr1, cc1 = control["bbox"]
        clean_all[rr0 : rr1 + 1, cc0 : cc1 + 1] = BG
    if np.array_equal(clean_all, expected):
        return False
    if np.array_equal(grid, expected):
        return False
    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(1200):
        built = _assemble_example(rng)
        if built is None:
            continue
        grid, expected, info = built
        if _semantic_ok(grid, expected, info):
            return grid
    raise RuntimeError("failed to generate a valid selective periodic-repair example")
