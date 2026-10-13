from __future__ import annotations

import numpy as np


BG = 0
FRAME = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.where(mask)
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _is_connected(mask: np.ndarray) -> bool:
    points = list(zip(*np.where(mask), strict=False))
    if not points:
        return False
    seen = {points[0]}
    stack = [points[0]]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for rr, cc in _neighbors4(int(r), int(c)):
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                stack.append((rr, cc))
    return len(seen) == len(points)


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r0 in range(h):
        for c0 in range(w):
            if seen[r0, c0] or int(grid[r0, c0]) != int(color):
                continue
            seen[r0, c0] = True
            stack = [(int(r0), int(c0))]
            comp: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                comp.append((int(r), int(c)))
                for rr, cc in _neighbors4(int(r), int(c)):
                    if 0 <= rr < h and 0 <= cc < w and not seen[rr, cc] and int(grid[rr, cc]) == int(color):
                        seen[rr, cc] = True
                        stack.append((int(rr), int(cc)))
            comps.append(comp)
    return comps


def _crop_component_mask(coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _resize_mask(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    src_h, src_w = mask.shape
    if src_h == target_h and src_w == target_w:
        return mask.copy()
    row_idx = np.rint(np.linspace(0, src_h - 1, target_h)).astype(int)
    col_idx = np.rint(np.linspace(0, src_w - 1, target_w)).astype(int)
    return mask[row_idx[:, None], col_idx[None, :]]


def _find_canvas(grid: np.ndarray) -> tuple[int, tuple[int, int, int, int]]:
    best: tuple[int, int, tuple[int, int, int, int]] | None = None
    for color in np.unique(grid):
        color = int(color)
        if color == BG:
            continue
        for coords in _components_of_color(grid, color):
            r0, c0, r1, c1 = _bbox_from_coords(coords)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h != w or h < 3:
                continue
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            if not bool((patch[0, :] == color).all() and (patch[-1, :] == color).all()):
                continue
            if not bool((patch[:, 0] == color).all() and (patch[:, -1] == color).all()):
                continue
            if h > 2 and not bool((patch[1:-1, 1:-1] == BG).all()):
                continue
            cand = (int(h), int(color), (int(r0), int(c0), int(r1), int(c1)))
            if best is None or cand[0] > best[0]:
                best = cand
    assert best is not None, "expected a square canvas frame"
    return best[1], best[2]


def _component_info(grid: np.ndarray, color: int, coords: list[tuple[int, int]]) -> dict[str, object]:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    mask = _crop_component_mask(coords)
    return {
        "color": int(color),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "mask": mask,
        "h": int(r1 - r0 + 1),
        "w": int(c1 - c0 + 1),
        "bbox_area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the grey output canvas by layering same-color fragments from outer size to inner size.

    Input structure:
    - Background is black.
    - Several non-black, non-grey monochrome fragments are scattered in the grid as separate 4-connected components.
    - Each non-grey color appears in exactly one 2- or 3-component size chain whose bounding boxes strictly nest by size.
    - A single grey 1-pixel square frame encloses an empty black canvas on the right side or bottom of the grid.

    Rule:
    - Detect the grey square frame and use its interior as the output canvas.
    - Group all other components by color; within each color, sort components by bounding-box area from largest to smallest.
    - For one color group, treat the largest bounding box as the reference size, scale every component mask into the canvas by the same bbox ratio, and center each scaled mask before overlaying the group from large to small.
    - Draw the color groups onto the canvas in descending order of their largest bounding-box area, so later smaller groups overwrite earlier larger ones where they overlap.

    Key insight: Color groups define which fragments belong together, bounding-box nesting defines the scale hierarchy inside each group, and cross-color overlap order comes from comparing the largest fragment of each color.
    """
    g = np.asarray(grid, dtype=int)
    frame_color, (r0, c0, r1, c1) = _find_canvas(g)
    canvas_size = int(r1 - r0 + 1)
    inner = int(canvas_size - 2)

    groups: dict[int, list[dict[str, object]]] = {}
    for color in np.unique(g):
        color = int(color)
        if color in (BG, frame_color):
            continue
        comps = _components_of_color(g, color)
        if not comps:
            continue
        groups[color] = [_component_info(g, color, coords) for coords in comps]

    ordered_groups: list[tuple[tuple[int, int, int, int, int], int, list[dict[str, object]]]] = []
    for color, infos in groups.items():
        infos.sort(
            key=lambda info: (
                -int(info["bbox_area"]),
                -int(info["h"]),
                -int(info["w"]),
                int(info["bbox"][0]),
                int(info["bbox"][1]),
            )
        )
        outer = infos[0]
        ordered_groups.append(
            (
                (
                    -int(outer["bbox_area"]),
                    -int(outer["h"]),
                    -int(outer["w"]),
                    int(outer["bbox"][0]),
                    int(outer["bbox"][1]),
                ),
                int(color),
                infos,
            )
        )

    ordered_groups.sort(key=lambda item: item[0])

    out = np.full((canvas_size, canvas_size), BG, dtype=int)
    out[0, :] = frame_color
    out[-1, :] = frame_color
    out[:, 0] = frame_color
    out[:, -1] = frame_color
    interior = out[1:-1, 1:-1]

    for _key, color, infos in ordered_groups:
        outer_h = int(infos[0]["h"])
        outer_w = int(infos[0]["w"])
        layer = np.zeros((inner, inner), dtype=bool)
        for info in infos:
            src_mask = np.asarray(info["mask"], dtype=bool)
            target_h = min(inner, max(1, int(round(inner * int(info["h"]) / outer_h))))
            target_w = min(inner, max(1, int(round(inner * int(info["w"]) / outer_w))))
            scaled = _resize_mask(src_mask, target_h, target_w)
            off_r = (inner - target_h) // 2
            off_c = (inner - target_w) // 2
            layer[off_r : off_r + target_h, off_c : off_c + target_w] |= scaled
        interior[layer] = int(color)

    return out


def _random_path(
    start: tuple[int, int], end: tuple[int, int], rng: np.random.Generator
) -> list[tuple[int, int]]:
    r, c = start
    out = [(int(r), int(c))]
    while (r, c) != end:
        choices = []
        if r < end[0]:
            choices.append((1, 0))
        elif r > end[0]:
            choices.append((-1, 0))
        if c < end[1]:
            choices.append((0, 1))
        elif c > end[1]:
            choices.append((0, -1))
        dr, dc = choices[int(rng.integers(0, len(choices)))]
        r += dr
        c += dc
        out.append((int(r), int(c)))
    return out


def _random_blob(side: int, rng: np.random.Generator) -> np.ndarray:
    for _ in range(120):
        if side <= 3:
            center = (side // 2, side // 2)
        else:
            mid_lo = max(1, side // 3)
            mid_hi = min(side - 2, side - side // 3 - 1)
            center = (
                int(rng.integers(mid_lo, mid_hi + 1)),
                int(rng.integers(mid_lo, mid_hi + 1)),
            )
        anchors = [
            (0, int(rng.integers(0, side))),
            (side - 1, int(rng.integers(0, side))),
            (int(rng.integers(0, side)), 0),
            (int(rng.integers(0, side)), side - 1),
        ]
        cells = {center}
        for anchor in anchors:
            cells.update(_random_path(center, anchor, rng))

        min_area = max(len(cells), int(np.ceil(0.35 * side * side)))
        max_area = max(min_area, min(side * side - 1, int(np.floor(0.72 * side * side))))
        target = int(rng.integers(min_area, max_area + 1))

        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for rr, cc in _neighbors4(int(r), int(c)):
                    if 0 <= rr < side and 0 <= cc < side and (rr, cc) not in cells:
                        frontier.append((int(rr), int(cc)))
            if not frontier:
                break
            cells.add(frontier[int(rng.integers(0, len(frontier)))])

        mask = np.zeros((side, side), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        if rng.random() < 0.5:
            mask = np.fliplr(mask)
        if rng.random() < 0.5:
            mask = np.flipud(mask)
        mask = np.rot90(mask, k=int(rng.integers(0, 4)))

        if not _is_connected(mask):
            continue
        if int(mask.sum()) >= side * side:
            continue
        if int(mask.sum()) <= side:
            continue
        r0, c0, r1, c1 = _mask_bbox(mask)
        if (r0, c0, r1, c1) != (0, 0, side - 1, side - 1):
            continue
        return mask
    raise RuntimeError("blob generation failed")


def _choose_sizes(outer_side: int, depth: int, rng: np.random.Generator) -> list[int] | None:
    if depth == 2:
        choices = [s for s in range(3, outer_side) if s < outer_side]
        if not choices:
            return None
        inner = int(rng.choice(choices))
        return [int(outer_side), inner]

    choices = []
    for middle in range(4, outer_side):
        for inner in range(3, middle):
            choices.append((middle, inner))
    if not choices:
        return None
    middle, inner = choices[int(rng.integers(0, len(choices)))]
    return [int(outer_side), int(middle), int(inner)]


def _paint_mask(grid: np.ndarray, mask: np.ndarray, color: int, r0: int, c0: int) -> None:
    sub = grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]]
    sub[mask] = int(color)


def _reserve(occ: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occ.shape[0], r0 + h + pad)
    cc1 = min(occ.shape[1], c0 + w + pad)
    occ[rr0:rr1, cc0:cc1] = True


def _find_positions(
    occ: np.ndarray,
    mask: np.ndarray,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
) -> list[tuple[int, int]]:
    h, w = mask.shape
    positions: list[tuple[int, int]] = []
    if row_end - row_start < h or col_end - col_start < w:
        return positions
    for r0 in range(row_start, row_end - h + 1):
        for c0 in range(col_start, col_end - w + 1):
            if occ[r0 - 1 : r0 + h + 1, c0 - 1 : c0 + w + 1].any():
                continue
            positions.append((int(r0), int(c0)))
    return positions


def _group_component_counts(grid: np.ndarray, frame_color: int) -> dict[int, int]:
    counts: dict[int, int] = {}
    for color in np.unique(grid):
        color = int(color)
        if color in (BG, frame_color):
            continue
        counts[color] = len(_components_of_color(grid, color))
    return counts


def _build_candidate(rng: np.random.Generator) -> np.ndarray | None:
    num_groups = int(rng.integers(2, 4))
    canvas_inner = int(rng.integers(8, 13))
    canvas_total = int(canvas_inner + 2)
    canvas_side = str(rng.choice(np.array(["right", "bottom"], dtype=object)))

    depths = [2 + int(rng.random() < 0.5) for _ in range(num_groups)]
    if 3 not in depths and rng.random() < 0.5:
        depths[int(rng.integers(0, num_groups))] = 3
    if 2 not in depths and rng.random() < 0.5:
        depths[int(rng.integers(0, num_groups))] = 2

    outer_sides: list[int] = []
    used_sides: set[int] = set()
    for depth in depths:
        options = [s for s in range(5, 10) if s not in used_sides and (depth == 2 or s >= 7)]
        if not options:
            return None
        side = int(options[int(rng.integers(0, len(options)))])
        used_sides.add(side)
        outer_sides.append(side)

    colors = [int(c) for c in rng.choice(np.array(PALETTE), size=num_groups, replace=False).tolist()]

    components: list[dict[str, object]] = []
    for color, depth, outer_side in zip(colors, depths, outer_sides, strict=False):
        sizes = _choose_sizes(int(outer_side), int(depth), rng)
        if sizes is None:
            return None
        for size in sizes:
            try:
                mask = _random_blob(int(size), rng)
            except RuntimeError:
                return None
            components.append(
                {
                    "color": int(color),
                    "mask": mask,
                    "bbox_area": int(size * size),
                }
            )

    max_patch = max(int(comp["mask"].shape[0]) for comp in components)
    if canvas_side == "right":
        max_content_w = 30 - canvas_total - 1
        if max_content_w < max_patch + 2:
            return None
        content_w = int(rng.integers(max(max_patch + 2, 12), max_content_w + 1))
        width = int(content_w + 1 + canvas_total)
        height = int(rng.integers(max(canvas_total, 18), 29))
        canvas_r = int(rng.integers(0, height - canvas_total + 1))
        canvas_c = int(width - canvas_total)
        row_start, row_end = 1, height - 1
        col_start, col_end = 1, canvas_c - 1
    else:
        max_content_h = 30 - canvas_total - 1
        if max_content_h < max_patch + 2:
            return None
        content_h = int(rng.integers(max(max_patch + 2, 12), max_content_h + 1))
        height = int(content_h + 1 + canvas_total)
        width = int(rng.integers(max(canvas_total, 18), 29))
        canvas_r = int(height - canvas_total)
        canvas_c = int(rng.integers(0, width - canvas_total + 1))
        row_start, row_end = 1, canvas_r - 1
        col_start, col_end = 1, width - 1

    grid = np.full((height, width), BG, dtype=int)
    occ = np.zeros((height, width), dtype=bool)

    grid[canvas_r : canvas_r + canvas_total, canvas_c : canvas_c + canvas_total] = BG
    grid[canvas_r, canvas_c : canvas_c + canvas_total] = FRAME
    grid[canvas_r + canvas_total - 1, canvas_c : canvas_c + canvas_total] = FRAME
    grid[canvas_r : canvas_r + canvas_total, canvas_c] = FRAME
    grid[canvas_r : canvas_r + canvas_total, canvas_c + canvas_total - 1] = FRAME
    _reserve(occ, canvas_r, canvas_c, canvas_total, canvas_total, pad=1)

    order = sorted(
        range(len(components)),
        key=lambda idx: (-int(components[idx]["bbox_area"]), int(components[idx]["color"])),
    )
    for idx in order:
        comp = components[idx]
        mask = np.asarray(comp["mask"], dtype=bool)
        positions = _find_positions(occ, mask, row_start, row_end, col_start, col_end)
        if not positions:
            return None
        r0, c0 = positions[int(rng.integers(0, len(positions)))]
        _paint_mask(grid, mask, int(comp["color"]), int(r0), int(c0))
        _reserve(occ, int(r0), int(c0), mask.shape[0], mask.shape[1], pad=1)

    counts = _group_component_counts(grid, FRAME)
    if sorted(counts.values()) != sorted(depths):
        return None
    if len(counts) != num_groups:
        return None
    outer_areas = []
    for color in counts:
        infos = [_component_info(grid, color, coords) for coords in _components_of_color(grid, color)]
        infos.sort(key=lambda info: -int(info["bbox_area"]))
        outer_areas.append(int(infos[0]["bbox_area"]))
        for i in range(len(infos) - 1):
            if not (int(infos[i]["h"]) > int(infos[i + 1]["h"]) and int(infos[i]["w"]) > int(infos[i + 1]["w"])):
                return None
    if len(set(outer_areas)) != len(outer_areas):
        return None

    out = main(grid)
    if out.shape != (canvas_total, canvas_total):
        return None
    if not bool((out[0, :] == FRAME).all() and (out[-1, :] == FRAME).all()):
        return None
    if not bool((out[:, 0] == FRAME).all() and (out[:, -1] == FRAME).all()):
        return None
    present_out = set(map(int, np.unique(out))) - {BG, FRAME}
    if present_out != set(counts):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = np.random.default_rng()
        result = _build_candidate(rng)
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after repeated attempts")
