from __future__ import annotations

from collections import Counter

import numpy as np


BG = 0
TERRAIN = 3
PLACED = 6
SOURCE_COLORS = (1, 2, 4, 5, 7, 8, 9)
MAX_GEN_ATTEMPTS = 500


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _crop_component(grid: np.ndarray, coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_coords(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    rr = coords[:, 0] - r0
    cc = coords[:, 1] - c0
    patch[rr, cc] = grid[coords[:, 0], coords[:, 1]]
    return patch


def _surface_rows(grid: np.ndarray) -> np.ndarray:
    terrain_mask = grid == TERRAIN
    assert terrain_mask.any(axis=0).all(), "expected terrain in every column"
    surface = np.argmax(terrain_mask, axis=0).astype(int)
    for c, start in enumerate(surface):
        assert not terrain_mask[:start, c].any(), "surface must be the topmost terrain cell"
        assert terrain_mask[start:, c].all(), "terrain must be a contiguous suffix in each column"
    return surface


def _find_valleys(surface: np.ndarray) -> list[dict]:
    w = int(surface.shape[0])
    valleys: dict[tuple[int, int], dict] = {}
    for left in range(1, w - 1):
        for right in range(left, w - 1):
            seg = surface[left : right + 1]
            if not (np.all(seg > surface[left - 1]) and np.all(seg > surface[right + 1])):
                continue

            extend_left = False
            if left > 1:
                seg2 = surface[left - 1 : right + 1]
                extend_left = bool(
                    np.all(seg2 > surface[left - 2]) and np.all(seg2 > surface[right + 1])
                )

            extend_right = False
            if right < w - 2:
                seg2 = surface[left : right + 2]
                extend_right = bool(
                    np.all(seg2 > surface[left - 1]) and np.all(seg2 > surface[right + 2])
                )

            if extend_left or extend_right:
                continue

            bottom = int(seg.max())
            depth = int(bottom - min(surface[left - 1], surface[right + 1]))
            valleys[(left, right)] = {
                "left": int(left),
                "right": int(right),
                "width": int(right - left + 1),
                "bottom": int(bottom),
                "depth": int(depth),
            }
    return [valleys[key] for key in sorted(valleys)]


def _find_sprites(grid: np.ndarray) -> list[dict]:
    sprite_mask = (grid != BG) & (grid != TERRAIN)
    sprites: list[dict] = []
    for coords in _components_from_mask(sprite_mask):
        colors = np.unique(grid[coords[:, 0], coords[:, 1]])
        assert len(colors) == 1, "each sprite should be monochrome"
        patch = _crop_component(grid, coords)
        r0, c0, r1, c1 = _bbox_coords(coords)
        sprites.append(
            {
                "coords": coords,
                "color": int(colors[0]),
                "bbox": (r0, c0, r1, c1),
                "patch": patch,
                "area": int(coords.shape[0]),
            }
        )
    return sprites


def _stamp_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, color: int) -> None:
    mask = patch != BG
    h, w = patch.shape
    window = grid[r0 : r0 + h, c0 : c0 + w]
    window[mask] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Embed scattered sprites into matching terrain valleys and recolor the embedded result magenta.

    Input structure:
    - Black background with a green terrain suffix in every column; the terrain surface forms 2-3 valleys.
    - Several detached monochrome sprites sit above the terrain, each separated from the others and from the terrain.
    - Every matchable sprite's cropped bounding-box width uniquely matches one valley width.

    Rule:
    - Read the terrain surface as the topmost green cell in each column, then find every maximal valley segment whose columns are all lower than both neighboring columns.
    - Crop each sprite to its bounding box and match only by unique width equality.
    - Move each matched sprite so its crop's left edge aligns to the valley's left edge and its crop's bottom row aligns to the valley bottom; sprite pixels overwrite terrain, the source copy is erased, and the placed sprite is recolored to magenta.

    Key insight:
    - The terrain contour provides both the target valley width and the vertical docking anchor, so width matching alone is insufficient without valley-bottom analysis.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    surface = _surface_rows(g)
    valleys = _find_valleys(surface)
    valley_width_counts = Counter(int(v["width"]) for v in valleys)
    unique_valleys = {
        int(v["width"]): v for v in valleys if valley_width_counts[int(v["width"])] == 1
    }

    sprites = _find_sprites(g)
    sprites_by_width: dict[int, list[dict]] = {}
    for sprite in sprites:
        width = int(sprite["patch"].shape[1])
        sprites_by_width.setdefault(width, []).append(sprite)

    out = g.copy()
    for width, same_width_sprites in sprites_by_width.items():
        if len(same_width_sprites) != 1 or width not in unique_valleys:
            continue
        sprite = same_width_sprites[0]
        valley = unique_valleys[width]
        patch = sprite["patch"]
        h, w = patch.shape
        top = int(valley["bottom"] - h + 1)
        left = int(valley["left"])
        assert top >= 0 and left >= 0 and left + w <= g.shape[1]

        coords = sprite["coords"]
        out[coords[:, 0], coords[:, 1]] = BG
        _stamp_patch(out, patch, top, left, PLACED)

    return out


def _random_sprite_patch(width: int, max_height: int, force_underfill: bool) -> np.ndarray | None:
    max_height = int(min(max_height, 4))
    valid_heights = [h for h in range(1, max_height + 1) if width * h >= 3]
    if width == 2:
        valid_heights = [h for h in valid_heights if h >= 2]
    if force_underfill:
        valid_heights = [h for h in valid_heights if h < max_height + (0 if max_height == 0 else 0)]
    if force_underfill:
        valid_heights = [h for h in valid_heights if h < max_height]
    if not valid_heights:
        return None

    multi_row = [h for h in valid_heights if h > 1]
    for _ in range(250):
        if multi_row and np.random.rand() < 0.8:
            height = int(np.random.choice(np.array(multi_row, dtype=int)))
        else:
            height = int(np.random.choice(np.array(valid_heights, dtype=int)))

        area_lo = max(3, width if height == 1 else 3)
        area_hi = min(7, width * height)
        if area_lo > area_hi:
            continue
        area = int(np.random.randint(area_lo, area_hi + 1))

        mask = np.zeros((height, width), dtype=bool)
        start = (int(np.random.randint(0, height)), int(np.random.randint(0, width)))
        mask[start] = True

        while int(mask.sum()) < area:
            frontier: set[tuple[int, int]] = set()
            for r, c in np.argwhere(mask):
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = int(r + dr), int(c + dc)
                    if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                        frontier.add((nr, nc))
            if not frontier:
                break
            nr, nc = list(frontier)[int(np.random.randint(0, len(frontier)))]
            mask[nr, nc] = True

        if int(mask.sum()) != area:
            continue

        coords = np.argwhere(mask)
        r0, c0, r1, c1 = _bbox_coords(coords)
        if r1 - r0 + 1 != height or c1 - c0 + 1 != width:
            continue
        if height > 1 and width > 1 and bool(mask.all()):
            continue

        patch = np.full((height, width), BG, dtype=int)
        patch[mask] = 1
        return patch
    return None


def _render_terrain(surface: np.ndarray, height: int) -> np.ndarray:
    grid = np.full((height, int(surface.shape[0])), BG, dtype=int)
    for c, start in enumerate(surface):
        grid[int(start) :, int(c)] = TERRAIN
    return grid


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] + 1 < b[0] or b[2] + 1 < a[0] or a[3] + 1 < b[1] or b[3] + 1 < a[1])


def _strictly_wrong_nearest_valley(
    positions: list[tuple[int, int]],
    patches: list[np.ndarray],
    valleys: list[dict],
) -> bool:
    centers = [0.5 * (v["left"] + v["right"]) for v in valleys]
    for idx, (r0, c0) in enumerate(positions):
        _ = r0
        width = int(patches[idx].shape[1])
        source_center = c0 + 0.5 * (width - 1)
        correct = abs(source_center - centers[idx])
        wrong = [abs(source_center - centers[j]) for j in range(len(valleys)) if j != idx]
        if wrong and min(wrong) < correct:
            return True
    return False


def _has_area_inversion(widths: list[int], areas: list[int]) -> bool:
    for i in range(len(widths)):
        for j in range(i + 1, len(widths)):
            if widths[i] < widths[j] and areas[i] >= areas[j]:
                return True
            if widths[j] < widths[i] and areas[j] >= areas[i]:
                return True
    return False


def _semantic_check_generated(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    try:
        surface = _surface_rows(grid)
        valleys = _find_valleys(surface)
        sprites = _find_sprites(grid)
    except Exception:
        return False

    if len(valleys) not in (2, 3) or len(sprites) != len(valleys):
        return False
    widths = [int(v["width"]) for v in valleys]
    if len(set(widths)) != len(widths):
        return False

    sprite_widths = [int(s["patch"].shape[1]) for s in sprites]
    if Counter(widths) != Counter(sprite_widths):
        return False
    if len({int(s["color"]) for s in sprites}) != len(sprites):
        return False

    valleys_by_width = {int(v["width"]): v for v in valleys}
    sprites_by_width = {int(s["patch"].shape[1]): s for s in sprites}

    if not any(len(set(surface[v["left"] : v["right"] + 1])) > 1 for v in valleys):
        return False
    if not any(
        int(sprites_by_width[w]["patch"].shape[0]) < int(valleys_by_width[w]["depth"]) for w in valleys_by_width
    ):
        return False

    sprite_order = sorted(
        ((int(sprites_by_width[w]["bbox"][1]), int(w)) for w in valleys_by_width), key=lambda item: item[0]
    )
    valley_order = sorted(((int(valleys_by_width[w]["left"]), int(w)) for w in valleys_by_width), key=lambda item: item[0])
    if [w for _c, w in sprite_order] == [w for _c, w in valley_order]:
        return False

    ordered_valleys = sorted(valleys, key=lambda v: v["left"])
    ordered_widths = [int(v["width"]) for v in ordered_valleys]
    ordered_patches = [sprites_by_width[w]["patch"] for w in ordered_widths]
    ordered_positions = [
        (int(sprites_by_width[w]["bbox"][0]), int(sprites_by_width[w]["bbox"][1])) for w in ordered_widths
    ]
    if not _strictly_wrong_nearest_valley(ordered_positions, ordered_patches, ordered_valleys):
        return False

    areas = [int(sprites_by_width[w]["area"]) for w in ordered_widths]
    if not _has_area_inversion(ordered_widths, areas):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    valley_count = int(np.random.choice(np.array([2, 3], dtype=int)))
    widths = [int(v) for v in np.random.choice(np.array([2, 3, 4, 5], dtype=int), size=valley_count, replace=False)]
    ridge_widths = [int(np.random.randint(1, 4)) for _ in range(valley_count + 1)]
    width_total = sum(widths) + sum(ridge_widths)
    if width_total > 24:
        return None

    height = int(np.random.randint(16, 21))
    ridge_min = height - 10
    ridge_max = height - 7
    ridge_rows = [int(np.random.randint(ridge_min, ridge_max + 1))]
    for _ in range(valley_count):
        next_row = ridge_rows[-1] + int(np.random.randint(-1, 2))
        next_row = int(np.clip(next_row, ridge_min, ridge_max))
        ridge_rows.append(next_row)

    valley_specs: list[dict] = []
    nonflat_candidates: list[int] = []
    for idx, width in enumerate(widths):
        left_row = ridge_rows[idx]
        right_row = ridge_rows[idx + 1]
        lower = max(max(left_row, right_row) + 1, min(left_row, right_row) + 2)
        upper = min(height - 3, min(left_row, right_row) + 4)
        if lower > upper:
            return None
        valley_specs.append(
            {
                "width": int(width),
                "left_row": int(left_row),
                "right_row": int(right_row),
                "lower": int(lower),
                "upper": int(upper),
            }
        )
        if width > 1 and lower < upper:
            nonflat_candidates.append(idx)

    if not nonflat_candidates:
        return None
    forced_nonflat = int(np.random.choice(np.array(nonflat_candidates, dtype=int)))

    surface_parts: list[int] = []
    for ridge_idx in range(valley_count + 1):
        surface_parts.extend([int(ridge_rows[ridge_idx])] * int(ridge_widths[ridge_idx]))
        if ridge_idx == valley_count:
            continue
        spec = valley_specs[ridge_idx]
        width = int(spec["width"])
        if ridge_idx == forced_nonflat:
            bottom = int(np.random.randint(spec["lower"] + 1, spec["upper"] + 1))
        else:
            bottom = int(np.random.randint(spec["lower"], spec["upper"] + 1))
        shallow = max(max(spec["left_row"], spec["right_row"]) + 1, bottom - 1)
        cols = [bottom] * width
        if ridge_idx == forced_nonflat or (shallow < bottom and np.random.rand() < 0.55):
            count = int(np.random.randint(1, width))
            chosen = np.random.choice(np.arange(width), size=count, replace=False)
            for pos in chosen:
                cols[int(pos)] = int(shallow)
            cols[int(np.random.randint(0, width))] = int(bottom)
        surface_parts.extend(cols)

    surface = np.array(surface_parts, dtype=int)
    terrain = _render_terrain(surface, height)
    valleys = _find_valleys(surface)
    if len(valleys) != valley_count:
        return None
    if sorted(int(v["width"]) for v in valleys) != sorted(widths):
        return None

    sprites_by_width: dict[int, np.ndarray] = {}
    underfill_candidates = [
        int(v["width"])
        for v in valleys
        if (
            (int(v["depth"]) >= 3 and int(v["width"]) == 2)
            or (int(v["depth"]) >= 2 and int(v["width"]) >= 3)
        )
    ]
    if not underfill_candidates:
        return None
    forced_underfill_width = int(np.random.choice(np.array(underfill_candidates, dtype=int)))

    for _ in range(120):
        sprites_by_width.clear()
        for valley in valleys:
            width = int(valley["width"])
            patch = _random_sprite_patch(
                width=width,
                max_height=int(valley["depth"]),
                force_underfill=width == forced_underfill_width,
            )
            if patch is None:
                break
            sprites_by_width[width] = patch
        if len(sprites_by_width) != len(valleys):
            continue

        area_pairs = [(int(v["width"]), int(np.count_nonzero(sprites_by_width[int(v["width"])]))) for v in valleys]
        if not _has_area_inversion([w for w, _a in area_pairs], [a for _w, a in area_pairs]):
            continue
        if not any(
            sprites_by_width[int(v["width"])].shape[0] < int(v["depth"])
            for v in valleys
        ):
            continue
        break
    if len(sprites_by_width) != len(valleys):
        return None

    valleys_sorted = sorted(valleys, key=lambda v: v["left"])
    patches_sorted = [sprites_by_width[int(v["width"])] for v in valleys_sorted]
    source_positions: list[tuple[int, int]] | None = None

    for _ in range(400):
        placed_boxes: list[tuple[int, int, int, int]] = []
        positions: list[tuple[int, int]] = [(-1, -1)] * len(valleys_sorted)
        order = list(np.random.permutation(len(valleys_sorted)))
        ok = True
        for idx in order:
            patch = patches_sorted[idx]
            h, w = patch.shape
            candidates: list[tuple[int, int, int, int, int]] = []
            for c0 in range(0, terrain.shape[1] - w + 1):
                max_r = int(surface[c0 : c0 + w].min() - h - 2)
                if max_r < 0:
                    continue
                for r0 in range(0, max_r + 1):
                    box = (r0, c0, r0 + h - 1, c0 + w - 1)
                    if any(_boxes_too_close(box, other) for other in placed_boxes):
                        continue
                    candidates.append((r0, c0, box[0], box[1], idx))
            if not candidates:
                ok = False
                break
            chosen = candidates[int(np.random.randint(0, len(candidates)))]
            r0, c0 = int(chosen[0]), int(chosen[1])
            positions[idx] = (r0, c0)
            placed_boxes.append((r0, c0, r0 + h - 1, c0 + w - 1))

        if not ok:
            continue

        left_order = sorted(range(len(positions)), key=lambda i: positions[i][1])
        if left_order == list(range(len(positions))):
            continue
        if not _strictly_wrong_nearest_valley(positions, patches_sorted, valleys_sorted):
            continue

        source_positions = positions
        break

    if source_positions is None:
        return None

    colors = [int(c) for c in np.random.choice(np.array(SOURCE_COLORS, dtype=int), size=valley_count, replace=False)]
    inp = terrain.copy()
    expected = terrain.copy()

    for idx, valley in enumerate(valleys_sorted):
        patch = patches_sorted[idx]
        source_r, source_c = source_positions[idx]
        _stamp_patch(inp, patch, source_r, source_c, colors[idx])

        target_r = int(valley["bottom"] - patch.shape[0] + 1)
        target_c = int(valley["left"])
        _stamp_patch(expected, patch, target_r, target_c, PLACED)

    if not _semantic_check_generated(inp):
        return None
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid terrain-valley embedding input")
