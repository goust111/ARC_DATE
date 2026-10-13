from __future__ import annotations

import itertools
import math
import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GEN_ATTEMPTS = 400

BBox = tuple[int, int, int, int]
Coord = tuple[int, int]


def bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def bbox(mask: np.ndarray) -> BBox | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return bbox_coords(coords)


def bbox_h(box: BBox) -> int:
    return int(box[2] - box[0] + 1)


def bbox_w(box: BBox) -> int:
    return int(box[3] - box[1] + 1)


def crop_bbox(grid: np.ndarray, box: BBox) -> np.ndarray:
    r0, c0, r1, c1 = map(int, box)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if int(connectivity) == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),            (0, 1),
            (1, -1),  (1, 0),   (1, 1),
        ]

    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[Coord] = deque([(r, c)])
            coords: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in steps:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> np.ndarray:
    r0 = int(r0)
    c0 = int(c0)
    h, w = patch.shape
    assert 0 <= r0 and 0 <= c0
    assert r0 + h <= dst.shape[0] and c0 + w <= dst.shape[1]
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        view[:, :] = patch
        return dst
    mask = patch != int(transparent)
    view[mask] = patch[mask]
    return dst


def intersects(a: BBox, b: BBox) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def normalize_coords(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0, c0, _r1, _c1 = bbox_coords(coords)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def dihedral_variants(patch: np.ndarray) -> list[np.ndarray]:
    variants: dict[tuple[tuple[int, ...], ...], np.ndarray] = {}
    cur = patch
    for _ in range(4):
        for arr in (cur, np.fliplr(cur)):
            key = tuple(tuple(int(v) for v in row) for row in arr.tolist())
            variants.setdefault(key, arr.copy())
        cur = np.rot90(cur)
    return list(variants.values())


def gcd_list(values: list[int]) -> int:
    g = 0
    for value in values:
        g = math.gcd(g, int(value))
    return int(g)


def _base_sprites() -> list[np.ndarray]:
    bases = [
        np.array([[RED, RED], [RED, 0]], dtype=int),
        np.array([[RED, RED], [RED, RED]], dtype=int),
        np.array([[RED, RED, RED], [0, RED, 0]], dtype=int),
        np.array([[RED, RED, 0], [0, RED, RED]], dtype=int),
        np.array([[RED, 0], [RED, RED], [0, RED]], dtype=int),
        np.array([[0, RED, 0], [RED, RED, RED], [0, RED, 0]], dtype=int),
        np.array([[RED, RED, 0], [RED, 0, 0], [RED, RED, RED]], dtype=int),
        np.array([[RED, RED, RED], [0, 0, RED], [0, RED, RED]], dtype=int),
    ]
    out: dict[tuple[tuple[int, ...], ...], np.ndarray] = {}
    for base in bases:
        for variant in dihedral_variants(base):
            key = tuple(tuple(int(v) for v in row) for row in variant.tolist())
            out.setdefault(key, variant.copy())
    return list(out.values())


SPRITE_LIBRARY = _base_sprites()


def _solid_blue_rectangle(grid: np.ndarray) -> BBox:
    rects: list[BBox] = []
    for coords in components_mask(grid == BLUE, connectivity=4):
        box = bbox_coords(coords)
        if int(coords.shape[0]) == bbox_h(box) * bbox_w(box) and int(coords.shape[0]) >= 9:
            rects.append(box)
    assert len(rects) == 1, "expected one solid blue rectangle"
    return rects[0]


def _analyze_input(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    red_components = components_mask(g == RED, connectivity=4)
    assert len(red_components) >= 3, "expected at least three visible red copies"

    info: list[dict[str, object]] = []
    for coords in red_components:
        box = bbox_coords(coords)
        info.append(
            {
                "coords": coords,
                "bbox": box,
                "shape": normalize_coords(coords),
            }
        )
    info.sort(key=lambda item: (int(item["bbox"][1]), int(item["bbox"][0])))

    template_bbox = info[0]["bbox"]
    template = crop_bbox(g, template_bbox)
    template_shape = info[0]["shape"]
    row0 = int(template_bbox[0])
    xs = [int(item["bbox"][1]) for item in info]

    for item in info:
        assert item["shape"] == template_shape, "all visible red components must be identical"
        assert int(item["bbox"][0]) == row0, "red copies must share one horizontal band"

    diffs = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
    assert len(diffs) >= 2 and all(diff > 0 for diff in diffs)
    step = gcd_list(diffs)
    assert step > bbox_w(template_bbox), "inferred step must leave a horizontal gap"
    assert all((x - xs[0]) % step == 0 for x in xs), "copy positions must align to one arithmetic progression"

    return {
        "template": template,
        "row0": row0,
        "x0": int(xs[0]),
        "visible_xs": xs,
        "step": step,
        "rect_bbox": _solid_blue_rectangle(g),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete and continue a horizontal arithmetic stamp trail until the next copy overlaps the blue stop rectangle.

    Input structure:
    - A black background contains several separated red copies of one small 4-connected sprite, all on the same rows.
    - The leftmost red component is the original sprite template; later copies use the exact same bbox-normalized mask.
    - Some intermediate arithmetic-sequence positions are erased to black, and a separate solid blue rectangle sits to the right.

    Rule:
    - Extract the leftmost red component's bbox as the sprite template and record every visible copy's left-edge column.
    - Infer the true horizontal step as the greatest positive integer dividing every gap between consecutive visible columns.
    - Paste red copies at every missing step position between the visible ones, then keep stepping right by the same amount.
    - Stop after drawing the first additional copy whose bbox overlaps the blue rectangle's bbox; only the sprite's red cells overwrite blue.

    Key insight:
    - The visible red gaps are irregular only because some arithmetic-progression terms were erased; recovering the common step unlocks both interpolation and the final collision-stopped extrapolation.
    """
    case = _analyze_input(np.asarray(grid, dtype=int))
    template = np.asarray(case["template"], dtype=int)
    row0 = int(case["row0"])
    step = int(case["step"])
    rect_bbox = case["rect_bbox"]

    out = np.asarray(grid, dtype=int).copy()
    h, w = template.shape
    col = int(case["x0"])
    while col + w <= out.shape[1]:
        current_bbox = (row0, col, row0 + h - 1, col + w - 1)
        paste(out, template, row0, col, transparent=BG)
        if intersects(current_bbox, rect_bbox):
            return out
        col += step
    raise AssertionError("arithmetic trail never reaches the blue rectangle")


def _valid_missing_patterns(total_terms: int, missing_count: int) -> list[tuple[int, ...]]:
    patterns: list[tuple[int, ...]] = []
    for missing in itertools.combinations(range(1, total_terms - 1), missing_count):
        visible = [idx for idx in range(total_terms) if idx not in missing]
        gaps = [visible[i + 1] - visible[i] for i in range(len(visible) - 1)]
        if gcd_list(gaps) == 1 and any(gap > 1 for gap in gaps):
            patterns.append(tuple(int(v) for v in missing))
    return patterns


def _render_input(
    shape: tuple[int, int],
    patch: np.ndarray,
    row0: int,
    x0: int,
    step: int,
    total_terms: int,
    missing: tuple[int, ...],
    rect_bbox: BBox,
) -> np.ndarray:
    h, w = shape
    rect_r0, rect_c0, rect_r1, rect_c1 = rect_bbox
    grid = np.zeros((max(rect_r1 + 1, row0 + h), max(rect_c1 + 1, x0 + total_terms * step + w)), dtype=int)
    grid[rect_r0 : rect_r1 + 1, rect_c0 : rect_c1 + 1] = BLUE
    missing_set = set(map(int, missing))
    for idx in range(total_terms):
        if idx in missing_set:
            continue
        paste(grid, patch, row0, x0 + idx * step, transparent=BG)
    return grid


def _render_expected(
    inp: np.ndarray,
    patch: np.ndarray,
    row0: int,
    x0: int,
    step: int,
    collision_index: int,
) -> np.ndarray:
    out = inp.copy()
    for idx in range(collision_index + 1):
        paste(out, patch, row0, x0 + idx * step, transparent=BG)
    return out


def _gen_attempt() -> np.ndarray | None:
    patch = random.choice(SPRITE_LIBRARY).copy()
    h, w = map(int, patch.shape)

    height = random.randint(max(8, h + 2), 14)
    row0 = random.randint(0, height - h)
    x0 = random.randint(0, 1)

    step_choices = list(range(w + 1, 6))
    random.shuffle(step_choices)
    for step in step_choices:
        max_collision_index = (30 - w - x0) // step
        if max_collision_index < 5:
            continue

        missing_count = random.choice([1, 2])
        clear_extrapolated = random.choice([1, 2])
        total_candidates = [
            total_terms
            for total_terms in range(4, 8)
            if 3 <= total_terms - missing_count <= 5 and total_terms + clear_extrapolated + 1 <= max_collision_index
        ]
        random.shuffle(total_candidates)

        for total_terms in total_candidates:
            missing_patterns = _valid_missing_patterns(total_terms, missing_count)
            if not missing_patterns:
                continue
            missing = random.choice(missing_patterns)
            collision_index = total_terms + clear_extrapolated + 1

            prev_x = x0 + (collision_index - 1) * step
            coll_x = x0 + collision_index * step
            rect_c0_min = prev_x + w
            rect_c0_max = min(coll_x + w - 1, 27)
            if rect_c0_min > rect_c0_max:
                continue
            rect_c0 = random.randint(rect_c0_min, rect_c0_max)

            rect_w_max = min(5, 30 - rect_c0)
            if rect_w_max < 3:
                continue
            rect_w = random.randint(3, rect_w_max)

            rect_h = random.randint(3, min(6, height))
            rect_r0_lo = max(0, row0 - rect_h + 1)
            rect_r0_hi = min(row0 + h - 1, height - rect_h)
            if rect_r0_lo > rect_r0_hi:
                continue
            rect_r0 = random.randint(rect_r0_lo, rect_r0_hi)
            rect_bbox = (rect_r0, rect_c0, rect_r0 + rect_h - 1, rect_c0 + rect_w - 1)

            used_w = max(coll_x + w, rect_c0 + rect_w)
            if used_w > 30:
                continue
            width = used_w + random.randint(0, min(2, 30 - used_w))

            inp = np.zeros((height, width), dtype=int)
            inp[rect_bbox[0] : rect_bbox[2] + 1, rect_bbox[1] : rect_bbox[3] + 1] = BLUE
            missing_set = set(map(int, missing))
            visible_indices = [idx for idx in range(total_terms) if idx not in missing_set]
            for idx in visible_indices:
                paste(inp, patch, row0, x0 + idx * step, transparent=BG)

            expected = _render_expected(inp, patch, row0, x0, step, collision_index)

            try:
                analyzed = _analyze_input(inp)
                assert analyzed["visible_xs"] == [x0 + idx * step for idx in visible_indices]
                assert int(analyzed["step"]) == step
                assert tuple(map(int, analyzed["rect_bbox"])) == rect_bbox
                assert not intersects((row0, prev_x, row0 + h - 1, prev_x + w - 1), rect_bbox)
                assert intersects((row0, coll_x, row0 + h - 1, coll_x + w - 1), rect_bbox)
                actual = main(inp)
            except Exception:
                continue

            if not np.array_equal(actual, expected):
                continue
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arithmetic-sequence stamp puzzle")
