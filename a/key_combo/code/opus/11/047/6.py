from __future__ import annotations

import random

import numpy as np


BG = 0
BAR = 9
SPRITE_COLORS = (1, 2, 3, 4, 6)


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _normalize_cells(cells: set[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


def _shape_variants(base: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    out: set[frozenset[tuple[int, int]]] = set()
    cur = set(base)
    for _ in range(4):
        max_r = max(r for r, _ in cur)
        max_c = max(c for _, c in cur)
        out.add(_normalize_cells(cur))
        out.add(_normalize_cells({(r, max_c - c) for r, c in cur}))
        cur = {(c, max_r - r) for r, c in cur}
    return [set(v) for v in out]


_BASE_SHAPES = {
    3: [
        {(0, 0), (0, 1), (0, 2)},
        {(0, 0), (1, 0), (1, 1)},
    ],
    4: [
        {(0, 0), (0, 1), (1, 0), (1, 1)},
        {(0, 0), (1, 0), (2, 0), (2, 1)},
        {(0, 0), (0, 1), (0, 2), (1, 1)},
    ],
    5: [
        {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)},
        {(0, 0), (0, 2), (1, 0), (1, 1), (1, 2)},
        {(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)},
    ],
    6: [
        {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)},
        {(0, 0), (0, 1), (0, 2), (1, 0), (2, 0), (2, 1)},
        {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (3, 1)},
    ],
    7: [
        {(0, 0), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)},
        {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 1), (2, 2)},
        {(0, 0), (0, 1), (0, 2), (1, 1), (2, 0), (2, 1), (2, 2)},
    ],
    8: [
        {(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)},
        {(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)},
        {(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 3), (2, 0), (2, 3)},
    ],
}


def _build_shape_library() -> dict[int, list[set[tuple[int, int]]]]:
    lib: dict[int, list[set[tuple[int, int]]]] = {}
    for area, bases in _BASE_SHAPES.items():
        seen = set()
        variants: list[set[tuple[int, int]]] = []
        for base in bases:
            for shape in _shape_variants(base):
                key = tuple(sorted(shape))
                if key not in seen:
                    seen.add(key)
                    variants.append(shape)
        lib[area] = variants
    return lib


SHAPE_LIBRARY = _build_shape_library()


def _shape_size(shape: set[tuple[int, int]]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _bbox_from_shape(shape: set[tuple[int, int]], row: int, col: int) -> tuple[int, int, int, int]:
    h, w = _shape_size(shape)
    return int(row), int(col), int(row + h - 1), int(col + w - 1)


def _rect_overlap(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 <= r1 and c0 <= c1:
        return int(r0), int(c0), int(r1), int(c1)
    return None


def _find_sprites(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objs: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color in (BG, BAR):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, c0 = min(rows), min(cols)
            shape = {(rr - r0, cc - c0) for rr, cc in coords}
            objs.append(
                {
                    "color": color,
                    "area": len(coords),
                    "src_row": int(r0),
                    "src_col": int(c0),
                    "bbox0": (int(r0), int(c0), int(max(rows)), int(max(cols))),
                    "shape": shape,
                }
            )
    return objs


def _settle_sprites(
    objects: list[dict], *, descending: bool = True, blocking: bool = True, occlusion: bool = True
) -> list[dict]:
    key = (
        (lambda obj: (-obj["area"], obj["bbox0"][0], obj["bbox0"][1]))
        if descending
        else (lambda obj: (obj["area"], obj["bbox0"][0], obj["bbox0"][1]))
    )
    ordered = sorted(objects, key=key)

    occupied: set[tuple[int, int]] = set()
    placed: list[dict] = []
    for obj in ordered:
        row = int(obj["src_row"])
        col = int(obj["src_col"])
        if blocking:
            while row > 1 and not any((row + r - 1, col + c) in occupied for r, c in obj["shape"]):
                row -= 1
        else:
            row = 1
        pixels = {(row + r, col + c) for r, c in obj["shape"]}
        bbox = _bbox_from_shape(obj["shape"], row, col)
        placed.append({**obj, "row": row, "col": col, "pixels": pixels, "bbox": bbox})
        occupied |= pixels

    for i, obj in enumerate(placed):
        visible = set(obj["pixels"])
        if occlusion:
            for larger in placed[:i]:
                rect = _rect_overlap(obj["bbox"], larger["bbox"])
                if rect is None:
                    continue
                r0, c0, r1, c1 = rect
                visible = {p for p in visible if not (r0 <= p[0] <= r1 and c0 <= p[1] <= c1)}
        obj["visible"] = visible
    return placed


def _render_output(shape: tuple[int, int], placed: list[dict]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    out[0, :] = BAR
    for obj in placed:
        for r, c in obj["visible"]:
            out[r, c] = int(obj["color"])
    return out


def _solve_details(
    grid: np.ndarray, *, descending: bool = True, blocking: bool = True, occlusion: bool = True
) -> tuple[np.ndarray, list[dict]]:
    objects = _find_sprites(grid)
    placed = _settle_sprites(objects, descending=descending, blocking=blocking, occlusion=occlusion)
    return _render_output(grid.shape, placed), placed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each monochrome sprite upward in area order, then let larger settled bboxes hide smaller ones in their overlap zones.

    Input structure:
    - A black background contains a full maroon bar across row 0.
    - Below the bar sit 4-connected monochrome sprites of distinct colors, with distinct areas.
    - Each sprite is treated by its own pixel shape and bounding box, using ordinary (row, col) coordinates.

    Rule:
    - Count each sprite's pixels and sort sprites from largest area to smallest; ties break by the input bbox's top row, then left column.
    - Process sprites in that order, sliding each one straight upward until its bbox top reaches row 1 or one more upward step would place one of its pixels onto a cell already occupied by an earlier settled sprite.
    - After all sprites settle, keep the larger sprite visible inside every final bounding-box overlap rectangle and erase the smaller sprite's pixels from that overlapped screen region.
    - Keep the maroon top bar unchanged and clear all source positions.

    Key insight:
    - The same visible property, sprite area, governs the whole pipeline: it decides move order, which settled shapes become later obstacles, and which sprite wins any final overlap zone.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    assert np.all(g[0, :] == BAR), "expected a solid maroon top bar"
    assert _most_common_color(g) == BG, "expected black background to remain dominant"
    out, _placed = _solve_details(g, descending=True, blocking=True, occlusion=True)
    return out


def _choose_final_scene(rng: random.Random) -> tuple[int, int, list[dict]] | None:
    for _ in range(5000):
        h = rng.randint(15, 19)
        w = rng.randint(15, 19)
        n = rng.choice((4, 5))
        areas = sorted(rng.sample(range(3, 9), n), reverse=True)

        final_objs: list[dict] = []
        occupied: set[tuple[int, int]] = set()
        have_bbox_overlap = False
        have_blocking = False
        have_occlusion = False

        success = True
        for idx, area in enumerate(areas):
            candidates: list[tuple] = []
            variants = SHAPE_LIBRARY[area][:]
            rng.shuffle(variants)
            for shape in variants:
                sh, sw = _shape_size(shape)
                cols = list(range(1, w - sw))
                rng.shuffle(cols)
                for col in cols:
                    src_row_probe = min(14, h - sh)
                    if src_row_probe < 4:
                        continue
                    row = src_row_probe
                    while row > 1 and not any((row + r - 1, col + c) in occupied for r, c in shape):
                        row -= 1
                    if idx == 0 and row != 1:
                        continue
                    if idx > 0 and row == 1:
                        continue

                    bbox = _bbox_from_shape(shape, row, col)
                    col_overlap = any(
                        1 <= min(bbox[3], prev["bbox"][3]) - max(bbox[1], prev["bbox"][1]) + 1 <= 2
                        for prev in final_objs
                    )
                    occluded_pixels = 0
                    for prev in final_objs:
                        rect = _rect_overlap(bbox, prev["bbox"])
                        if rect is None:
                            continue
                        r0, c0, r1, c1 = rect
                        occluded_pixels += sum(
                            r0 <= row + r <= r1 and c0 <= col + c <= c1 for r, c in shape
                        )

                    candidates.append((shape, sh, sw, col, row, bbox, col_overlap, occluded_pixels))

            if not candidates:
                success = False
                break

            need_overlap = idx > 0 and not have_bbox_overlap
            need_occlusion = idx > 0 and not have_occlusion
            filtered = [
                cand
                for cand in candidates
                if (not need_overlap or cand[6]) and (not need_occlusion or cand[7] > 0)
            ]
            if not filtered:
                filtered = [cand for cand in candidates if not need_overlap or cand[6]]
            if not filtered:
                filtered = candidates

            shape, sh, sw, col, row, bbox, col_overlap, occluded_pixels = rng.choice(filtered)
            pixels = {(row + r, col + c) for r, c in shape}
            occupied |= pixels
            final_objs.append(
                {
                    "area": area,
                    "shape": shape,
                    "h": sh,
                    "w": sw,
                    "col": col,
                    "final_row": row,
                    "bbox": bbox,
                }
            )
            have_bbox_overlap |= col_overlap
            have_blocking |= row > 1
            have_occlusion |= occluded_pixels > 0

        if success and have_bbox_overlap and have_blocking and have_occlusion:
            return h, w, final_objs
    return None


def _assign_input_rows(h: int, final_objs: list[dict], rng: random.Random) -> bool:
    row_choices: list[list[int]] = []
    for obj in final_objs:
        high = min(14, h - obj["h"])
        rows = list(range(max(4, obj["final_row"] + 2), high + 1))
        rng.shuffle(rows)
        if not rows:
            return False
        row_choices.append(rows)

    occupied: set[tuple[int, int]] = set()
    assignment: list[int] = []

    def backtrack(i: int) -> bool:
        if i == len(final_objs):
            return True
        obj = final_objs[i]
        for row in row_choices[i]:
            pixels = {(row + r, obj["col"] + c) for r, c in obj["shape"]}
            if pixels & occupied:
                continue
            assignment.append(row)
            occupied.update(pixels)
            if backtrack(i + 1):
                return True
            assignment.pop()
            occupied.difference_update(pixels)
        return False

    if not backtrack(0):
        return False
    for obj, row in zip(final_objs, assignment):
        obj["src_row"] = int(row)
    return True


def _validate_scene(inp: np.ndarray, out: np.ndarray) -> bool:
    comps = _find_sprites(inp)
    if len(comps) not in (4, 5):
        return False
    if len({obj["color"] for obj in comps}) != len(comps):
        return False
    areas = [obj["area"] for obj in comps]
    if any(area < 3 or area > 8 for area in areas):
        return False
    if len(set(areas)) != len(areas):
        return False
    if any(obj["bbox0"][0] < 4 or obj["bbox0"][0] > 14 for obj in comps):
        return False
    if not any(
        1 <= min(a["bbox0"][3], b["bbox0"][3]) - max(a["bbox0"][1], b["bbox0"][1]) + 1 <= 2
        for i, a in enumerate(comps)
        for b in comps[i + 1 :]
    ):
        return False
    if np.array_equal(inp, out):
        return False
    return True


def _attempt_input(rng: random.Random) -> np.ndarray | None:
    scene = _choose_final_scene(rng)
    if scene is None:
        return None
    h, w, final_objs = scene
    if not _assign_input_rows(h, final_objs, rng):
        return None

    colors = rng.sample(SPRITE_COLORS, len(final_objs))
    inp = np.zeros((h, w), dtype=int)
    inp[0, :] = BAR
    for obj, color in zip(final_objs, colors):
        obj["color"] = int(color)
        for dr, dc in obj["shape"]:
            inp[obj["src_row"] + dr, obj["col"] + dc] = int(color)

    out, placed = _solve_details(inp, descending=True, blocking=True, occlusion=True)
    if not _validate_scene(inp, out):
        return None

    if np.array_equal(out, _solve_details(inp, descending=False, blocking=True, occlusion=True)[0]):
        return None
    if np.array_equal(out, _solve_details(inp, descending=True, blocking=False, occlusion=True)[0]):
        return None
    if np.array_equal(out, _solve_details(inp, descending=True, blocking=True, occlusion=False)[0]):
        return None

    removed = sum(len(obj["pixels"] - obj["visible"]) for obj in placed)
    if removed == 0:
        return None
    if not any(obj["row"] > 1 for obj in placed[1:]):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(300):
        inp = _attempt_input(rng)
        if inp is not None:
            return inp
    raise RuntimeError("generate_input: failed to build a non-degenerate sample")
