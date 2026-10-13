from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
SEPARATOR = 5
SOURCE_COLORS = (1, 2)
HALO_COLORS = (3, 4, 6, 7, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

BASE_TEMPLATES = (
    np.array(
        [
            [1, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 1, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 0, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 0, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 1],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 0, 1],
            [1, 1, 1],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 1, 0],
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=int,
    ),
)

MAX_ATTEMPTS = 500


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
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
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _holes_mask(patch: np.ndarray) -> np.ndarray:
    h, w = patch.shape
    holes = (patch == BG)
    outside = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if holes[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if holes[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and holes[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))
    return holes & ~outside


def _make_patch(coords: np.ndarray, color: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    box = _bbox(coords)
    r0, c0, r1, c1 = box
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = int(color)
    return patch, box


def _extract_markers(grid: np.ndarray) -> tuple[dict[int, int], dict[int, set[tuple[int, int]]]]:
    h, _w = grid.shape
    markers: dict[int, int] = {}
    source_cells: dict[int, set[tuple[int, int]]] = {}
    r = 0
    while r < h - 1:
        block = grid[r : r + 2, :2]
        if (
            block.shape == (2, 2)
            and np.all(block != BG)
            and np.all(block != SEPARATOR)
            and np.all(block[:, 0] == block[0, 0])
            and np.all(block[:, 1] == block[0, 1])
        ):
            src = int(block[0, 0])
            halo = int(block[0, 1])
            assert src in SOURCE_COLORS and halo not in SOURCE_COLORS and halo != SEPARATOR
            assert src != halo and src not in markers
            if r > 0:
                assert np.all(grid[r - 1, :2] == BG)
            if r + 2 < h:
                assert np.all(grid[r + 2, :2] == BG)
            markers[src] = halo
            source_cells[src] = {(r, 0), (r + 1, 0)}
            r += 2
            continue
        r += 1
    assert markers
    return markers, source_cells


def _parse_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    sep_cols = [c for c in range(w) if np.all(g[:, c] == SEPARATOR)]
    assert len(sep_cols) == 1
    sep = int(sep_cols[0])
    assert 8 <= sep <= w - 5
    assert np.all(g[:, sep + 1 :] == BG)

    markers, marker_source_cells = _extract_markers(g[:, :sep])
    objects: list[dict] = []

    for color, halo in markers.items():
        for coords in _components_from_mask(g[:, :sep] == color):
            coord_set = {tuple(map(int, pt)) for pt in coords}
            if coord_set <= marker_source_cells[color]:
                continue
            patch, box = _make_patch(coords, color)
            holes = _holes_mask(patch)
            hole_components = _components_from_mask(holes)
            assert hole_components
            assert all(1 <= len(comp) <= 2 for comp in hole_components)
            assert not np.array_equal(patch != BG, np.fliplr(patch != BG))
            r0, c0, r1, c1 = box
            assert c0 >= 3
            assert c1 <= sep - 2
            objects.append(
                {
                    "color": int(color),
                    "halo": int(halo),
                    "bbox": box,
                    "patch": patch,
                    "holes": holes,
                    "hole_lr_diff": not np.array_equal(holes, np.fliplr(holes)),
                    "height": int(patch.shape[0]),
                    "width": int(patch.shape[1]),
                    "top": int(r0),
                }
            )

    objects.sort(key=lambda obj: (obj["bbox"][0], obj["bbox"][1], obj["color"]))
    assert 1 <= len(objects) <= 2
    return {
        "grid": g,
        "height": h,
        "width": w,
        "sep": sep,
        "markers": markers,
        "objects": objects,
        "halo_colors": set(markers.values()),
    }


def _simulate(parsed: dict, *, collect: bool = False) -> tuple[np.ndarray, list[dict]]:
    g = parsed["grid"]
    h = parsed["height"]
    w = parsed["width"]
    sep = parsed["sep"]
    halo_colors = tuple(sorted(parsed["halo_colors"]))
    out = g.copy()
    records: list[dict] = []

    for obj_index, obj in enumerate(parsed["objects"]):
        top = obj["top"]
        patch = obj["patch"]
        obj_h = obj["height"]
        obj_w = obj["width"]
        color = obj["color"]
        halo = obj["halo"]
        x = sep + 1
        copy_index = 1
        while x + obj_w <= w:
            sprite = np.fliplr(patch) if copy_index % 2 == 1 else patch.copy()
            sprite_mask = sprite != BG
            out[top : top + obj_h, x : x + obj_w][sprite_mask] = color

            pr0 = max(0, top - 1)
            pc0 = max(0, x - 1)
            pr1 = min(h - 1, top + obj_h)
            pc1 = min(w - 1, x + obj_w)
            local = np.zeros((pr1 - pr0 + 1, pc1 - pc0 + 1), dtype=int)
            rr0 = top - pr0
            cc0 = x - pc0
            local[rr0 : rr0 + obj_h, cc0 : cc0 + obj_w][sprite_mask] = color
            holes = _holes_mask(local)
            fill = (local == BG) & ~holes

            region = out[pr0 : pr1 + 1, pc0 : pc1 + 1]
            overwrite_mask = fill & ((region == BG) | np.isin(region, halo_colors))
            region[overwrite_mask] = halo

            if collect:
                records.append(
                    {
                        "obj_index": obj_index,
                        "copy_index": copy_index,
                        "sprite": sprite.copy(),
                        "body_box": (top, x, top + obj_h - 1, x + obj_w - 1),
                        "halo_box": (pr0, pc0, pr1, pc1),
                        "local_holes": holes.copy(),
                        "halo_fill": fill.copy(),
                        "halo_color": halo,
                        "color": color,
                    }
                )
            x += obj_w
            copy_index += 1

    return out, records


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each hole-bearing source object across the separator with alternating mirror copies and one-pixel halos that preserve holes.

    Input structure:
    - The background is black (0), and a full-height grey (5) separator column divides the grid into a left source side and a blank right target side.
    - One or two left-edge 2x2 legend blocks map each source object color (red or blue) to its halo color.
    - The left side contains one or two 4-connected monochrome objects, each with at least one enclosed black hole inside its tight bounding box.

    Rule:
    - Keep the left side and the separator unchanged.
    - For each source object, use its bounding-box width as the period, then place full-width copies from the first column right of the separator onward.
    - Odd-numbered copies are left-right mirrored, even-numbered copies keep the original orientation, and each copy keeps the source object's top row.
    - Around every drawn copy, expand its bbox by one cell and recolor only background or earlier-halo cells with the legend halo color, while leaving any enclosed hole cells black.

    Key insight:
    - The object's width defines the tiling period, but every period must be reinterpreted through an alternating mirror and a fresh hole-aware halo computation.
    """
    parsed = _parse_input(grid)
    out, _ = _simulate(parsed, collect=False)
    return out


def _template_variants() -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    for template in BASE_TEMPLATES:
        for flip_ud in (False, True):
            for flip_lr in (False, True):
                cur = template.copy()
                if flip_ud:
                    cur = np.flipud(cur)
                if flip_lr:
                    cur = np.fliplr(cur)
                if any(np.array_equal(cur, seen) for seen in variants):
                    continue
                holes = _holes_mask(cur)
                if not _components_from_mask(cur.astype(bool)):
                    continue
                if len(_components_from_mask(cur.astype(bool))) != 1:
                    continue
                if np.array_equal(cur, np.fliplr(cur)):
                    continue
                hole_sizes = sorted(len(comp) for comp in _components_from_mask(holes))
                if not hole_sizes or any(size > 2 for size in hole_sizes):
                    continue
                variants.append(cur)
    return variants


TEMPLATE_VARIANTS = _template_variants()


def _random_object_patch() -> np.ndarray:
    return random.choice(TEMPLATE_VARIANTS).copy()


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 < b0 or b1 < a0)


def _padded_boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return _intervals_overlap(ar0 - pad, ar1 + pad, br0 - pad, br1 + pad) and _intervals_overlap(
        ac0 - pad, ac1 + pad, bc0 - pad, bc1 + pad
    )


def _choose_object_colors(n_objects: int) -> list[int]:
    if n_objects == 1:
        return [random.choice(SOURCE_COLORS)]
    if random.random() < 0.5:
        return [1, 2]
    color = random.choice(SOURCE_COLORS)
    return [color, color]


def _place_legends(grid: np.ndarray, unique_colors: list[int], halo_map: dict[int, int]) -> tuple[int, list[tuple[int, int, int, int]]]:
    h, _w = grid.shape
    legend_boxes: list[tuple[int, int, int, int]] = []
    gap = random.randint(1, 2)
    total_height = 2 * len(unique_colors) + gap * (len(unique_colors) - 1)
    top = h - total_height
    for idx, color in enumerate(unique_colors):
        r0 = top + idx * (2 + gap)
        grid[r0 : r0 + 2, 0] = color
        grid[r0 : r0 + 2, 1] = halo_map[color]
        legend_boxes.append((r0, 0, r0 + 1, 1))
    return top, legend_boxes


def _build_input() -> np.ndarray | None:
    n_objects = random.randint(1, 2)
    colors = _choose_object_colors(n_objects)
    patches = [_random_object_patch() for _ in range(n_objects)]
    if not any(not np.array_equal(_holes_mask(patch), np.fliplr(_holes_mask(patch))) for patch in patches):
        return None

    max_w = max(int(patch.shape[1]) for patch in patches)
    max_h = max(int(patch.shape[0]) for patch in patches)

    h = random.randint(max(11, max_h + 6), 15)
    sep = random.randint(9, 12)
    min_ext = max(3 * max_w, 10)
    max_ext = min(15, 30 - sep - 1)
    if min_ext > max_ext:
        return None
    ext_width = random.randint(min_ext, max_ext)
    w = sep + 1 + ext_width

    grid = np.zeros((h, w), dtype=int)
    grid[:, sep] = SEPARATOR

    unique_colors = sorted(set(colors))
    halo_map = {color: halo for color, halo in zip(unique_colors, random.sample(HALO_COLORS, len(unique_colors)))}
    legend_top, legend_boxes = _place_legends(grid, unique_colors, halo_map)

    available_bottom = legend_top - 2
    if available_bottom < 5:
        return None

    placed_boxes: list[tuple[int, int, int, int]] = []
    order = list(range(n_objects))
    random.shuffle(order)
    placements: list[tuple[int, int, np.ndarray, int]] = [(-1, -1, np.zeros((1, 1), dtype=int), 0) for _ in range(n_objects)]

    for idx in order:
        patch = patches[idx]
        obj_h, obj_w = patch.shape
        if available_bottom - obj_h < 1:
            return None
        c_min = 3
        c_max = sep - obj_w - 2
        if c_min > c_max:
            return None
        candidates: list[tuple[int, int]] = []
        for top in range(1, available_bottom - obj_h + 1):
            for left in range(c_min, c_max + 1):
                box = (top, left, top + obj_h - 1, left + obj_w - 1)
                if any(_padded_boxes_overlap(box, other, pad=1) for other in placed_boxes):
                    continue
                if any(_padded_boxes_overlap(box, other, pad=0) for other in legend_boxes):
                    continue
                candidates.append((top, left))
        if not candidates:
            return None
        top, left = random.choice(candidates)
        placements[idx] = (top, left, patch, colors[idx])
        placed_boxes.append((top, left, top + obj_h - 1, left + obj_w - 1))

    for top, left, patch, color in placements:
        mask = patch == 1
        grid[top : top + patch.shape[0], left : left + patch.shape[1]][mask] = color

    return grid


def _validate_generated_input(grid: np.ndarray) -> bool:
    try:
        parsed = _parse_input(grid)
        objects = parsed["objects"]
        assert any(obj["hole_lr_diff"] for obj in objects)
        assert all((parsed["width"] - parsed["sep"] - 1) // obj["width"] >= 3 for obj in objects)
        out, records = _simulate(parsed, collect=True)
        assert np.array_equal(out, main(grid))
        assert np.any(out[:, parsed["sep"] + 1 :] != BG)
        assert any(record["copy_index"] >= 2 for record in records)

        grouped: dict[int, list[dict]] = {}
        for record in records:
            grouped.setdefault(record["obj_index"], []).append(record)
        assert any(
            len(group) >= 2 and not np.array_equal(group[0]["sprite"], group[1]["sprite"])
            for group in grouped.values()
        )

        halo_pixels = 0
        for record in records:
            pr0, pc0, pr1, pc1 = record["halo_box"]
            region = out[pr0 : pr1 + 1, pc0 : pc1 + 1]
            hole_mask = record["local_holes"]
            assert np.all(region[hole_mask] == BG)
            halo_pixels += int(np.sum(region == record["halo_color"]))
        assert halo_pixels > 0
        return True
    except AssertionError:
        return False


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _build_input()
        if grid is not None and _validate_generated_input(grid):
            return grid
    raise RuntimeError(f"failed to generate a valid puzzle after {MAX_ATTEMPTS} attempts")
