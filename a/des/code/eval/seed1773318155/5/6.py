import random
from typing import Iterable

import numpy as np


DIVIDER_COLOR = 1
ARROW_COLOR = 2
PLACEHOLDER_COLOR = 5
FREE_COLORS = [0, 3, 4, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
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
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1


def _mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, r1, c0, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0, c1 - c0), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    return tuple(mask.shape), tuple(int(x) for x in mask.astype(int).ravel())


def _apply_transform(mask: np.ndarray, code: int) -> np.ndarray:
    if code == 0:
        return mask.copy()
    if code == 1:
        return np.fliplr(mask)
    if code == 2:
        return np.flipud(mask)
    if code == 3:
        return np.flipud(np.fliplr(mask))
    raise ValueError(code)


def _reflection_signature(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    return min(_mask_key(_apply_transform(mask, code)) for code in range(4))


def _colored_components(grid: np.ndarray, exclude: Iterable[int]) -> list[dict]:
    excluded = set(int(x) for x in exclude)
    comps: list[dict] = []
    for color in map(int, np.unique(grid)):
        if color in excluded:
            continue
        for coords in _components(grid == color):
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(coords.shape[0]),
                    "coord_set": {tuple(map(int, pt)) for pt in coords.tolist()},
                }
            )
    return comps


def _find_divider(grid: np.ndarray, bg: int) -> tuple[int, int]:
    rows = [
        r
        for r in range(grid.shape[0])
        if int(grid[r, 0]) != int(bg) and np.all(grid[r, :] == grid[r, 0])
    ]
    assert len(rows) == 1, "expected exactly one full-width non-background divider row"
    row = int(rows[0])
    return row, int(grid[row, 0])


def _parse_library(top: np.ndarray, bg: int) -> tuple[list[dict], int]:
    comps = _colored_components(top, exclude={bg})
    labels: list[dict] = []
    used_shape_ids: set[int] = set()

    for idx, comp in enumerate(comps):
        if comp["area"] != 1:
            continue
        r, c = map(int, comp["coords"][0])
        matches = [
            j
            for j, other in enumerate(comps)
            if j != idx and other["area"] > 1 and (r - 1, c) in other["coord_set"]
        ]
        if len(matches) == 1:
            labels.append({"label_idx": idx, "shape_idx": matches[0], "color": comp["color"]})

    assert labels, "expected singleton labels directly under library shapes"
    label_color = int(labels[0]["color"])
    assert all(int(item["color"]) == int(label_color) for item in labels)

    library: list[dict] = []
    seen_signatures: set[tuple[tuple[int, int], tuple[int, ...]]] = set()
    for item in labels:
        shape_idx = int(item["shape_idx"])
        assert shape_idx not in used_shape_ids, "library shapes must have unique labels"
        used_shape_ids.add(shape_idx)
        comp = comps[shape_idx]
        mask = _mask_from_coords(comp["coords"])
        sig = _reflection_signature(mask)
        assert sig not in seen_signatures, "library shapes must be unique up to reflection"
        seen_signatures.add(sig)
        library.append({"color": int(comp["color"]), "mask": mask, "sig": sig})

    assert len(library) >= 2
    return library, label_color


def _arrow_direction(comp: dict, width: int) -> str | None:
    coords = comp["coords"]
    if int(coords.shape[0]) != 3:
        return None
    r0, r1, c0, c1 = comp["bbox"]
    if r1 - r0 != 2 or c1 - c0 != 2:
        return None
    if not (c0 == 0 or c1 == width):
        return None
    rel = [(int(r - r0), int(c - c0)) for r, c in coords]
    col_counts = {}
    for _r, c in rel:
        col_counts[c] = col_counts.get(c, 0) + 1
    if sorted(col_counts.values()) != [1, 2]:
        return None
    nub_col = min(col_counts, key=lambda c: col_counts[c])
    return "left" if nub_col == min(col_counts) else "right"


def _overlap_len(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace grey blueprint placeholders with matched library shapes.

    The top half is a library of colored shapes, each marked by a grey singleton
    label directly underneath. The bottom half contains grey placeholders whose
    masks match a library shape up to reflection. A right-pointing boundary
    arrow requests a left-right mirror of the matched source shape before
    placement; a left arrow (or no arrow) uses the source's original orientation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    divider_row, divider_color = _find_divider(g, bg)
    assert divider_color != bg

    top = g[:divider_row, :]
    bottom = g[divider_row + 1 :, :]
    assert top.size > 0 and bottom.size > 0

    library, placeholder_color = _parse_library(top, bg)
    lib_by_sig = {item["sig"]: item for item in library}

    bottom_comps = _colored_components(bottom, exclude={bg})
    placeholders = [comp for comp in bottom_comps if int(comp["color"]) == int(placeholder_color) and int(comp["area"]) > 1]

    arrows: list[dict] = []
    for comp in bottom_comps:
        if int(comp["color"]) == int(placeholder_color):
            continue
        direction = _arrow_direction(comp, bottom.shape[1])
        if direction is not None:
            arrows.append({**comp, "direction": direction})

    out = g.copy()
    for arrow in arrows:
        rr = divider_row + 1 + arrow["coords"][:, 0]
        cc = arrow["coords"][:, 1]
        out[rr, cc] = int(bg)

    for comp in placeholders:
        mask = _mask_from_coords(comp["coords"])
        sig = _reflection_signature(mask)
        assert sig in lib_by_sig, "placeholder must match a library shape up to reflection"
        source = lib_by_sig[sig]

        pr0, pr1, pc0, pc1 = comp["bbox"]
        matched_arrows = []
        for arrow in arrows:
            ar0, ar1, _ac0, _ac1 = arrow["bbox"]
            overlap = _overlap_len(pr0, pr1, ar0, ar1)
            if overlap > 0:
                center_gap = abs((ar0 + ar1 - 1) - (pr0 + pr1 - 1))
                matched_arrows.append((-overlap, center_gap, arrow["direction"]))
        matched_arrows.sort()
        direction = matched_arrows[0][2] if matched_arrows else None

        paint_mask = np.fliplr(source["mask"]) if direction == "right" else source["mask"]
        assert paint_mask.shape == (pr1 - pr0, pc1 - pc0)

        rr0 = divider_row + 1 + pr0
        rr1 = divider_row + 1 + pr1
        out[rr0:rr1, pc0:pc1] = int(bg)
        region = out[rr0:rr1, pc0:pc1]
        region[paint_mask] = int(source["color"])

    return out


def _split_amount(total: int, parts: int) -> list[int]:
    buckets = [0] * parts
    for _ in range(total):
        buckets[random.randrange(parts)] += 1
    return buckets


def _random_connected_shape() -> np.ndarray:
    for _ in range(120):
        h = random.randint(3, 5)
        w = random.randint(3, 6)
        target = random.randint(max(4, (h * w + 2) // 3), max(4, h * w - 1))
        cells = {(random.randrange(h), random.randrange(w))}
        frontier = set(cells)
        while len(cells) < target and frontier:
            r, c = random.choice(tuple(frontier))
            nbrs = []
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    nbrs.append((nr, nc))
            if not nbrs:
                frontier.remove((r, c))
                continue
            pt = random.choice(nbrs)
            cells.add(pt)
            frontier.add(pt)

        arr = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            arr[r, c] = True
        coords = np.argwhere(arr)
        arr = _mask_from_coords(coords)
        if arr.shape[0] < 3 or arr.shape[1] < 3:
            continue
        if arr.sum() < 4 or arr.sum() == arr.size:
            continue
        if np.array_equal(arr, np.fliplr(arr)) or np.array_equal(arr, np.flipud(arr)):
            continue
        return arr
    raise RuntimeError("failed to build an asymmetric connected shape")


def _arrow_cells(top: int, width: int, side: str, direction: str) -> list[tuple[int, int]]:
    nub_top = random.choice([True, False])
    if side == "left":
        stem_col = 1 if direction == "left" else 0
        nub_col = 0 if direction == "left" else 1
    else:
        stem_col = width - 1 if direction == "left" else width - 2
        nub_col = width - 2 if direction == "left" else width - 1
    nub_row = top if nub_top else top + 1
    return [(top, stem_col), (top + 1, stem_col), (nub_row, nub_col)]


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(FREE_COLORS)
    source_colors = random.sample([c for c in FREE_COLORS if c != bg], random.randint(2, 4))

    shapes: list[dict] = []
    seen_signatures: set[tuple[tuple[int, int], tuple[int, ...]]] = set()
    for color in source_colors:
        for _ in range(80):
            mask = _random_connected_shape()
            sig = _reflection_signature(mask)
            if sig in seen_signatures:
                continue
            seen_signatures.add(sig)
            shapes.append({"color": int(color), "mask": mask, "sig": sig})
            break
        else:
            return None

    n_lib = len(shapes)
    n_blueprints = random.randint(max(3, n_lib), n_lib + 2)

    blueprint_shape_ids = list(range(n_lib))
    while len(blueprint_shape_ids) < n_blueprints:
        blueprint_shape_ids.append(random.randrange(n_lib))
    random.shuffle(blueprint_shape_ids)

    arrow_modes = (["left", "right", "none"] * ((n_blueprints + 2) // 3))[:n_blueprints]
    random.shuffle(arrow_modes)

    library_widths = [shape["mask"].shape[1] for shape in shapes]
    library_heights = [shape["mask"].shape[0] for shape in shapes]
    base_gaps = [random.randint(2, 4) for _ in range(n_lib + 1)]
    top_margin = random.randint(1, 2)
    top_tail = random.randint(0, 1)
    max_lib_h = max(library_heights)
    library_required_width = sum(library_widths) + sum(base_gaps)

    blueprint_masks = []
    has_shape_change = False
    for idx, shape_id in enumerate(blueprint_shape_ids):
        source_mask = shapes[shape_id]["mask"]
        final_code = 1 if arrow_modes[idx] == "right" else 0
        if random.random() < 0.65:
            candidates = [code for code in range(4) if code != final_code]
            input_code = random.choice(candidates)
        else:
            input_code = random.randrange(4)
        input_mask = _apply_transform(source_mask, input_code)
        output_mask = _apply_transform(source_mask, final_code)
        if not np.array_equal(input_mask, output_mask):
            has_shape_change = True
        blueprint_masks.append({"shape_id": shape_id, "input_code": input_code, "input_mask": input_mask})

    if not has_shape_change:
        return None

    blueprint_heights = [item["input_mask"].shape[0] for item in blueprint_masks]
    blueprint_widths = [item["input_mask"].shape[1] for item in blueprint_masks]
    bp_top_margin = random.randint(1, 2)
    bp_bottom_margin = random.randint(1, 2)
    bp_gaps = [random.randint(1, 3) for _ in range(n_blueprints - 1)]
    blueprint_height = bp_top_margin + bp_bottom_margin + sum(blueprint_heights) + sum(bp_gaps)

    min_width = max(library_required_width, max(blueprint_widths) + 6, 14)
    width = random.randint(min_width, min_width + 8)
    extra = width - library_required_width
    gap_extra = _split_amount(extra, n_lib + 1)
    lib_gaps = [base + add for base, add in zip(base_gaps, gap_extra)]

    divider_row = top_margin + max_lib_h + 1 + top_tail
    total_height = divider_row + 1 + blueprint_height
    grid = np.full((total_height, width), int(bg), dtype=int)
    grid[divider_row, :] = int(DIVIDER_COLOR)

    x = lib_gaps[0]
    lib_baseline = top_margin + max_lib_h - 1
    for shape, gap_after in zip(shapes, lib_gaps[1:]):
        mask = shape["mask"]
        h, w = mask.shape
        top = lib_baseline - h + 1
        grid[top : top + h, x : x + w][mask] = int(shape["color"])
        bottom_cols = [c for c in range(w) if mask[h - 1, c]]
        label_col = x + random.choice(bottom_cols)
        grid[top + h, label_col] = int(PLACEHOLDER_COLOR)
        x += w + gap_after

    row = bp_top_margin
    used_arrow_cells: set[tuple[int, int]] = set()
    for i, item in enumerate(blueprint_masks):
        mask = item["input_mask"]
        h, w = mask.shape
        left = random.randint(3, width - w - 3)
        abs_top = divider_row + 1 + row
        grid[abs_top : abs_top + h, left : left + w][mask] = int(PLACEHOLDER_COLOR)

        mode = arrow_modes[i]
        if mode != "none":
            side = random.choice(["left", "right"])
            arrow_top = row + random.randint(0, h - 2)
            cells = _arrow_cells(arrow_top, width, side, mode)
            if any(cell in used_arrow_cells for cell in cells):
                return None
            used_arrow_cells.update(cells)
            for rr, cc in cells:
                grid[divider_row + 1 + rr, cc] = int(ARROW_COLOR)

        if i < len(bp_gaps):
            row += h + bp_gaps[i]

    if _get_bg(grid) != int(bg):
        return None

    values, counts = np.unique(grid, return_counts=True)
    bg_count = int(counts[np.where(values == bg)[0][0]])
    if bg_count <= int(max(count for value, count in zip(values, counts) if int(value) != int(bg))):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid blueprint-to-library ARC input")
