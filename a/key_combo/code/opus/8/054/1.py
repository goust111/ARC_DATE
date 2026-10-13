from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
COPY_COLORS = (3, 5, 6, 7, 8, 9)


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _neighbors(r: int, c: int, shape: tuple[int, int], connectivity: int) -> Iterable[tuple[int, int]]:
    h, w = shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            if connectivity == 4 and abs(dr) + abs(dc) != 1:
                continue
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w:
                yield rr, cc


def _components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
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
                for nr, nc in _neighbors(rr, cc, mask.shape, connectivity):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _dihedral_arrays(arr: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    for flipped in (False, True):
        base = np.fliplr(arr) if flipped else arr
        for k in range(4):
            variants.append(np.rot90(base, k).copy())
    return variants


def _signature(patch: np.ndarray) -> frozenset[tuple[int, int]]:
    return frozenset((int(r), int(c)) for r, c in np.argwhere(patch == RED))


def _template_variants(template_patch: np.ndarray) -> list[dict[str, object]]:
    variants: list[dict[str, object]] = []
    for index, variant in enumerate(_dihedral_arrays(template_patch)):
        variants.append(
            {
                "index": index,
                "patch": variant,
                "shape": tuple(int(x) for x in variant.shape),
                "signature": _signature(variant),
                "support": variant != BLACK,
            }
        )
    return variants


def _extract_template(grid: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray, list[dict[str, object]]]:
    bbox = _bbox_from_mask((grid == BLUE) | (grid == YELLOW))
    assert bbox is not None, "expected a blue/yellow template"
    patch = _crop(grid, bbox)
    assert patch.ndim == 2
    colors = set(map(int, np.unique(patch)))
    assert colors.issubset({BLUE, YELLOW, RED}), "template may only contain blue, yellow, and red"
    assert RED in colors and BLUE in colors and YELLOW in colors, "template must contain both body colors and connectors"
    assert np.count_nonzero(patch == RED) == 2, "template must have exactly two red connectors"
    variants = _template_variants(patch)
    keys = {(entry["shape"], entry["signature"]) for entry in variants}
    assert len(keys) == 8, "template connector signatures must distinguish all eight orientations"
    return bbox, patch, variants


def _copy_components(grid: np.ndarray, template_bbox: tuple[int, int, int, int]) -> list[dict[str, object]]:
    mask = grid != BLACK
    r0, c0, r1, c1 = template_bbox
    mask[r0 : r1 + 1, c0 : c1 + 1] = False
    pieces = []
    for coords in _components(mask, connectivity=8):
        rr0, cc0 = coords.min(axis=0)
        rr1, cc1 = coords.max(axis=0)
        bbox = (int(rr0), int(cc0), int(rr1), int(cc1))
        patch = _crop(grid, bbox)
        pieces.append({"bbox": bbox, "patch": patch, "coords": coords})
    pieces.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return pieces


def _solve_with_metadata(grid: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    template_bbox, _template_patch, variants = _extract_template(g)
    variant_by_key = {(entry["shape"], entry["signature"]): entry for entry in variants}

    out = g.copy()
    copy_infos: list[dict[str, object]] = []
    for item in _copy_components(g, template_bbox):
        bbox = item["bbox"]
        patch = item["patch"]
        reds = np.argwhere(patch == RED)
        assert reds.shape[0] == 2, "each copy must keep both connectors"
        body_colors = set(map(int, np.unique(patch))) - {BLACK, RED}
        assert len(body_colors) == 1, "each copy must use a single foreground color"
        fg = int(next(iter(body_colors)))
        key = (tuple(int(x) for x in patch.shape), _signature(patch))
        assert key in variant_by_key, "copy connector signature must match a unique template orientation"
        variant = variant_by_key[key]
        support = variant["support"]
        rr0, cc0, rr1, cc1 = bbox
        block = out[rr0 : rr1 + 1, cc0 : cc1 + 1]
        block[np.asarray(support, dtype=bool)] = fg
        out[rr0 : rr1 + 1, cc0 : cc1 + 1] = block
        copy_infos.append(
            {
                "bbox": bbox,
                "fg": fg,
                "variant_index": int(variant["index"]),
                "shape": tuple(int(x) for x in patch.shape),
                "signature": key[1],
            }
        )

    r0, c0, r1, c1 = template_bbox
    out[r0 : r1 + 1, c0 : c1 + 1] = BLACK
    return out, copy_infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete every monochrome partial copy by using the template's two red border connectors to infer its orientation.

    Input structure:
    - A black background contains one complete rectangular template made of blue and yellow cells, with exactly two red connector cells on its border.
    - Elsewhere, there are 3-5 incomplete copies of that template under rotations/reflections.
    - Each copy keeps both red connectors, shows only part of the template, and uses one non-red foreground color distinct from the template colors.

    Rule:
    - Extract the template from the bounding box of all blue/yellow cells and record the two red connector coordinates inside that box.
    - For each incomplete copy, take the bounding box of all its visible cells, read the two red connector coordinates inside that box, and match them against the eight rotated/reflected template variants.
    - Use the uniquely matched variant to fill every missing template cell with the copy's own foreground color, then recolor the red connectors to that same color.
    - Finally erase the original template rectangle to black.

    Key insight:
    - The red connector pair is not for docking objects together; it is an orientation signature that uniquely identifies which rotated/reflected template footprint each partial copy should complete.
    """
    return _solve_with_metadata(grid)[0]


def _random_template_pattern(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    for _ in range(400):
        pattern = rng.choice([BLUE, YELLOW], size=(h, w))
        blue_count = int(np.count_nonzero(pattern == BLUE))
        yellow_count = h * w - blue_count
        if blue_count < h or yellow_count < h:
            continue
        keys = {np.asarray(variant, dtype=int).tobytes() + str(variant.shape).encode() for variant in _dihedral_arrays(pattern)}
        if len(keys) == 8:
            return pattern
    raise RuntimeError("failed to generate an asymmetric template pattern")


def _random_connectors(rng: np.random.Generator, h: int, w: int) -> list[tuple[int, int]]:
    top = [(0, c) for c in range(1, w - 1)]
    bottom = [(h - 1, c) for c in range(1, w - 1)]
    left = [(r, 0) for r in range(1, h - 1)]
    right = [(r, w - 1) for r in range(1, h - 1)]
    border = top + bottom + left + right
    for _ in range(400):
        picks = rng.choice(len(border), size=2, replace=False)
        coords = [border[int(picks[0])], border[int(picks[1])]]
        tmp = np.full((h, w), BLACK, dtype=int)
        for r, c in coords:
            tmp[r, c] = RED
        keys = {(variant.shape, _signature(variant)) for variant in _dihedral_arrays(tmp)}
        if len(keys) == 8:
            return coords
    raise RuntimeError("failed to choose unique connector locations")


def _random_path(
    rng: np.random.Generator, start: tuple[int, int], end: tuple[int, int]
) -> list[tuple[int, int]]:
    r, c = start
    tr, tc = end
    path = [(r, c)]
    while (r, c) != (tr, tc):
        options: list[tuple[int, int]] = []
        if r < tr:
            options.append((r + 1, c))
        elif r > tr:
            options.append((r - 1, c))
        if c < tc:
            options.append((r, c + 1))
        elif c > tc:
            options.append((r, c - 1))
        nxt = options[int(rng.integers(len(options)))]
        r, c = nxt
        path.append((r, c))
    return path


def _visible_mask(
    rng: np.random.Generator, h: int, w: int, connectors: list[tuple[int, int]]
) -> np.ndarray:
    area = h * w
    min_keep = max(8, int(np.ceil(area * 0.45)))
    max_keep = max(min_keep, int(np.floor(area * 0.70)))
    for _ in range(400):
        anchors = set(connectors)
        anchors.add((0, int(rng.integers(w))))
        anchors.add((h - 1, int(rng.integers(w))))
        anchors.add((int(rng.integers(h)), 0))
        anchors.add((int(rng.integers(h)), w - 1))
        anchor_list = list(anchors)
        rng.shuffle(anchor_list)
        mask = np.zeros((h, w), dtype=bool)
        for a, b in zip(anchor_list, anchor_list[1:]):
            for r, c in _random_path(rng, a, b):
                mask[r, c] = True
        if not all(mask[r, c] for r, c in connectors):
            continue
        skeleton = int(mask.sum())
        if skeleton > max_keep:
            continue
        target = int(rng.integers(max(skeleton, min_keep), max_keep + 1))
        while int(mask.sum()) < target:
            frontier: list[tuple[int, int]] = []
            for rr, cc in np.argwhere(mask):
                for nr, nc in _neighbors(int(rr), int(cc), (h, w), 4):
                    if not mask[nr, nc]:
                        frontier.append((nr, nc))
            if not frontier:
                break
            nr, nc = frontier[int(rng.integers(len(frontier)))]
            mask[nr, nc] = True
        if not np.all(mask.any(axis=0)) or not np.all(mask.any(axis=1)):
            continue
        if int(mask.sum()) >= area:
            continue
        return mask
    raise RuntimeError("failed to build a valid visible mask")


def _stamp(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    dst[top : top + h, left : left + w] = patch


def _can_place(occupied: np.ndarray, top: int, left: int, h: int, w: int) -> bool:
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(occupied.shape[0], top + h + 1)
    c1 = min(occupied.shape[1], left + w + 1)
    return not occupied[r0:r1, c0:c1].any()


def _mark_occupied(occupied: np.ndarray, top: int, left: int, h: int, w: int) -> None:
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(occupied.shape[0], top + h + 1)
    c1 = min(occupied.shape[1], left + w + 1)
    occupied[r0:r1, c0:c1] = True


def _choose_template_corner(rng: np.random.Generator, grid_h: int, grid_w: int, h: int, w: int) -> tuple[int, int]:
    corners = [
        (0, 0),
        (0, grid_w - w),
        (grid_h - h, 0),
        (grid_h - h, grid_w - w),
    ]
    return corners[int(rng.integers(len(corners)))]


def _expected_output(
    grid: np.ndarray,
    template_bbox: tuple[int, int, int, int],
    placements: list[tuple[int, int, np.ndarray, int]],
) -> np.ndarray:
    out = grid.copy()
    for top, left, variant_patch, fg in placements:
        support = variant_patch != BLACK
        block = out[top : top + variant_patch.shape[0], left : left + variant_patch.shape[1]]
        block[support] = fg
        out[top : top + variant_patch.shape[0], left : left + variant_patch.shape[1]] = block
    r0, c0, r1, c1 = template_bbox
    out[r0 : r1 + 1, c0 : c1 + 1] = BLACK
    return out


def _generate_one(rng: np.random.Generator) -> np.ndarray | None:
    grid_h = int(rng.integers(22, 31))
    grid_w = int(rng.integers(22, 31))

    if rng.random() < 0.35:
        side = int(rng.integers(5, 7))
        template_h = side
        template_w = side
    else:
        template_h = int(rng.integers(5, 7))
        template_w = int(rng.integers(5, 8))
        if template_h == template_w:
            template_w = 7 if template_w == 5 else 5

    template_body = _random_template_pattern(rng, template_h, template_w)
    connectors = _random_connectors(rng, template_h, template_w)
    template_patch = template_body.copy()
    for r, c in connectors:
        template_patch[r, c] = RED
    variants = _template_variants(template_patch)

    copy_count = int(rng.integers(3, 6))
    for _ in range(300):
        chosen = rng.choice(len(variants), size=copy_count, replace=False)
        indices = [int(x) for x in chosen]
        shapes = [variants[i]["shape"] for i in indices]
        if len(set(shapes)) == copy_count:
            continue
        if not any(i < 4 for i in indices):
            continue
        if not any(i >= 4 for i in indices):
            continue
        if template_h != template_w and rng.random() < 0.7 and len(set(shapes)) < 2:
            continue
        break
    else:
        return None

    fg_colors = rng.choice(np.array(COPY_COLORS, dtype=int), size=copy_count, replace=False)
    grid = np.zeros((grid_h, grid_w), dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    template_top, template_left = _choose_template_corner(rng, grid_h, grid_w, template_h, template_w)
    _stamp(grid, template_patch, template_top, template_left)
    _mark_occupied(occupied, template_top, template_left, template_h, template_w)
    template_bbox = (
        template_top,
        template_left,
        template_top + template_h - 1,
        template_left + template_w - 1,
    )

    placements: list[tuple[int, int, np.ndarray, int]] = []
    order = list(range(copy_count))
    rng.shuffle(order)
    for idx in order:
        variant = variants[indices[idx]]
        variant_patch = np.asarray(variant["patch"], dtype=int)
        h, w = variant_patch.shape
        top_candidates = [(r, c) for r in range(grid_h - h + 1) for c in range(grid_w - w + 1)]
        rng.shuffle(top_candidates)
        placed = False
        for top, left in top_candidates:
            if not _can_place(occupied, top, left, h, w):
                continue
            visible = _visible_mask(rng, h, w, sorted(variant["signature"]))
            copy_patch = np.zeros((h, w), dtype=int)
            copy_patch[visible] = int(fg_colors[idx])
            for r, c in variant["signature"]:
                copy_patch[int(r), int(c)] = RED
            if np.count_nonzero(copy_patch == RED) != 2:
                continue
            if np.count_nonzero(copy_patch == int(fg_colors[idx])) == 0:
                continue
            _stamp(grid, copy_patch, top, left)
            _mark_occupied(occupied, top, left, h, w)
            placements.append((top, left, variant_patch, int(fg_colors[idx])))
            placed = True
            break
        if not placed:
            return None

    expected = _expected_output(grid, template_bbox, placements)
    actual, infos = _solve_with_metadata(grid)
    if not np.array_equal(actual, expected):
        return None

    if len(infos) != copy_count:
        return None
    if len({info["variant_index"] for info in infos}) != copy_count:
        return None
    if len({info["fg"] for info in infos}) != copy_count:
        return None
    if len({(info["shape"], info["signature"]) for info in infos}) != copy_count:
        return None
    if len({info["shape"] for info in infos}) == copy_count:
        return None
    if not any(info["variant_index"] < 4 for info in infos):
        return None
    if not any(info["variant_index"] >= 4 for info in infos):
        return None

    seen_output_blocks = set()
    for info in infos:
        r0, c0, r1, c1 = info["bbox"]
        out_block = actual[r0 : r1 + 1, c0 : c1 + 1]
        seen_output_blocks.add((out_block.shape, out_block.tobytes()))
    if len(seen_output_blocks) != copy_count:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        grid = _generate_one(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
