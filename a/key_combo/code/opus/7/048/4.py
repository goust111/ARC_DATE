import random

import numpy as np


BACKGROUND = 0
HOLE = 6
MARKER_COLORS = (1, 3, 4, 8, 9)
PATTERN_COLORS = (2, 5, 7)

GRID_SIDE_RANGE = (19, 28)
INNER_SIDE_CHOICES = (5, 7, 9)
BOX_COUNT_CHOICES = (2, 3)
GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 120


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _marker_groups(grid: np.ndarray) -> list[dict]:
    values, counts = np.unique(grid, return_counts=True)
    groups: list[dict] = []
    for color, count in zip(values.tolist(), counts.tolist()):
        if color in (BACKGROUND, HOLE) or count != 3:
            continue
        coords = np.argwhere(grid == color)
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        on_border = (
            (coords[:, 0] == r0)
            | (coords[:, 0] == r1)
            | (coords[:, 1] == c0)
            | (coords[:, 1] == c1)
        )
        if not bool(np.all(on_border)):
            continue
        groups.append({"color": int(color), "bbox": (r0, c0, r1, c1)})
    return groups


def _repair_box(out: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    center_r = (r0 + r1) // 2
    center_c = (c0 + c1) // 2
    holes = [
        (r, c)
        for r in range(r0 + 1, r1)
        for c in range(c0 + 1, c1)
        if int(out[r, c]) == HOLE
    ]
    for r, c in holes:
        candidates = (
            (2 * center_r - r, 2 * center_c - c),
            (2 * center_r - r, c),
            (r, 2 * center_c - c),
        )
        fill = None
        for rr, cc in candidates:
            color = int(out[rr, cc])
            if color != HOLE:
                fill = color
                break
        assert fill is not None, "every hole must be resolvable"
        out[r, c] = fill


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each same-color three-marker bbox to repair the damaged point-symmetric interior, then draw that bbox as a border in the marker color.

    Input structure:
    - The background is black, and there are 2-3 disjoint marker groups.
    - Each group is exactly three same-color pixels, with color neither black nor magenta.
    - The three markers lie on the boundary of their shared bounding box, whose interior contains a multicolor pattern damaged by magenta holes.

    Rule:
    - Find every same-color three-pixel marker group and compute its bounding box.
    - Inside each box, replace every magenta hole by the color at its 180-degree counterpart about the box interior center; if that counterpart is also magenta, fall back to the horizontal reflection, then to the vertical reflection.
    - After all holes in a box are repaired, paint the full box perimeter in the marker color.

    Key insight:
    - The three sparse markers encode both the repair region geometry and the final border color, so the box must be inferred first and the border drawn only after the interior symmetry repair.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    groups = _marker_groups(g)
    assert 2 <= len(groups) <= 3, "expected 2-3 marker groups"

    out = g.copy()
    for group in sorted(groups, key=lambda item: (item["bbox"][1], item["bbox"][0])):
        r0, c0, r1, c1 = group["bbox"]
        _repair_box(out, (r0, c0, r1, c1))
        color = int(group["color"])
        out[r0, c0 : c1 + 1] = color
        out[r1, c0 : c1 + 1] = color
        out[r0 : r1 + 1, c0] = color
        out[r0 : r1 + 1, c1] = color
    return out


def _block_cells(center_r: int, center_c: int, dr: int, dc: int) -> dict[str, tuple[int, int]]:
    return {
        "tl": (center_r - dr, center_c - dc),
        "tr": (center_r - dr, center_c + dc),
        "bl": (center_r + dr, center_c - dc),
        "br": (center_r + dr, center_c + dc),
    }


def _paint_block(patch: np.ndarray, cells: dict[str, tuple[int, int]], role: str) -> None:
    if role == "general":
        diag_a, diag_b = random.sample(PATTERN_COLORS, 2)
        if random.random() < 0.5:
            diag_a, diag_b = diag_b, diag_a
        patch[cells["tl"]] = diag_a
        patch[cells["br"]] = diag_a
        patch[cells["tr"]] = diag_b
        patch[cells["bl"]] = diag_b
        return

    color = random.choice(PATTERN_COLORS)
    for coord in cells.values():
        patch[coord] = color


def _build_output_patch(
    inner_h: int, inner_w: int, mode: str
) -> tuple[np.ndarray, dict[str, tuple[int, int]] | None]:
    patch = np.zeros((inner_h, inner_w), dtype=int)
    center_r = inner_h // 2
    center_c = inner_w // 2

    block_keys = [(dr, dc) for dr in range(1, center_r + 1) for dc in range(1, center_c + 1)]
    general_key = random.choice(block_keys)
    uniform_key = None if mode == "rot_only" else random.choice([key for key in block_keys if key != general_key])

    for key in block_keys:
        cells = _block_cells(center_r, center_c, key[0], key[1])
        if key == general_key:
            role = "general"
        elif key == uniform_key:
            role = "uniform"
        else:
            role = random.choice(("general", "general", "uniform"))
        _paint_block(patch, cells, role)

    for dc in range(1, center_c + 1):
        color = random.choice(PATTERN_COLORS)
        patch[center_r, center_c - dc] = color
        patch[center_r, center_c + dc] = color
    for dr in range(1, center_r + 1):
        color = random.choice(PATTERN_COLORS)
        patch[center_r - dr, center_c] = color
        patch[center_r + dr, center_c] = color
    patch[center_r, center_c] = random.choice(PATTERN_COLORS)

    special_cells = None
    if uniform_key is not None:
        special_cells = _block_cells(center_r, center_c, uniform_key[0], uniform_key[1])
        special_color = random.choice(PATTERN_COLORS)
        for coord in special_cells.values():
            patch[coord] = special_color
    return patch, special_cells


def _choose_holes(
    inner_h: int, inner_w: int, mode: str, special_cells: dict[str, tuple[int, int]] | None
) -> list[tuple[int, int]]:
    center = (inner_h // 2, inner_w // 2)
    holes: list[tuple[int, int]] = []
    excluded = set()

    if mode == "h_fallback":
        assert special_cells is not None
        holes.extend([special_cells["tl"], special_cells["br"]])
        excluded.update(special_cells.values())
        target = random.randint(2, 4)
    elif mode == "v_fallback":
        assert special_cells is not None
        holes.extend([special_cells["tl"], special_cells["bl"], special_cells["br"]])
        excluded.update(special_cells.values())
        target = random.randint(3, 4)
    else:
        target = random.randint(1, 4)

    reserved = set(holes)
    while len(holes) < target:
        r = random.randrange(inner_h)
        c = random.randrange(inner_w)
        cell = (r, c)
        partner = (inner_h - 1 - r, inner_w - 1 - c)
        if cell == center:
            continue
        if cell in reserved or cell in excluded:
            continue
        if partner in reserved:
            continue
        holes.append(cell)
        reserved.add(cell)
        reserved.add(partner)
    return holes


def _sample_markers(box_h: int, box_w: int) -> list[tuple[int, int]]:
    corner = random.choice(("tl", "tr", "bl", "br"))
    edge_row = random.randint(1, box_h - 2)
    edge_col = random.randint(1, box_w - 2)
    if corner == "tl":
        return [(0, 0), (box_h - 1, edge_col), (edge_row, box_w - 1)]
    if corner == "tr":
        return [(0, box_w - 1), (box_h - 1, edge_col), (edge_row, 0)]
    if corner == "bl":
        return [(box_h - 1, 0), (0, edge_col), (edge_row, box_w - 1)]
    return [(box_h - 1, box_w - 1), (0, edge_col), (edge_row, 0)]


def _build_box(mode: str, marker_color: int) -> tuple[np.ndarray, np.ndarray]:
    inner_h = random.choice(INNER_SIDE_CHOICES)
    inner_w = random.choice(INNER_SIDE_CHOICES)
    output_patch, special_cells = _build_output_patch(inner_h, inner_w, mode)
    holes = _choose_holes(inner_h, inner_w, mode, special_cells)

    box_h = inner_h + 2
    box_w = inner_w + 2
    inp = np.zeros((box_h, box_w), dtype=int)
    out = np.zeros((box_h, box_w), dtype=int)
    inp[1:-1, 1:-1] = output_patch
    out[1:-1, 1:-1] = output_patch

    for r, c in holes:
        inp[r + 1, c + 1] = HOLE

    markers = _sample_markers(box_h, box_w)
    for r, c in markers:
        inp[r, c] = marker_color

    out[0, :] = marker_color
    out[-1, :] = marker_color
    out[:, 0] = marker_color
    out[:, -1] = marker_color
    return inp, out


def _separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return (
        a[2] + 1 < b[0]
        or b[2] + 1 < a[0]
        or a[3] + 1 < b[1]
        or b[3] + 1 < a[1]
    )


def _place_boxes(boxes: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(PLACEMENT_ATTEMPTS):
        height = random.randint(*GRID_SIDE_RANGE)
        width = random.randint(*GRID_SIDE_RANGE)
        placements: list[tuple[int, int, int, int, np.ndarray, np.ndarray]] = []
        success = True

        for inp_box, out_box in random.sample(boxes, len(boxes)):
            box_h, box_w = inp_box.shape
            candidates = []
            for r0 in range(1, height - box_h):
                for c0 in range(1, width - box_w):
                    bbox = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
                    if all(_separated(bbox, placed[:4]) for placed in placements):
                        candidates.append((r0, c0))
            if not candidates:
                success = False
                break
            r0, c0 = random.choice(candidates)
            placements.append((r0, c0, r0 + box_h - 1, c0 + box_w - 1, inp_box, out_box))

        if not success:
            continue

        inp = np.zeros((height, width), dtype=int)
        out = np.zeros((height, width), dtype=int)
        for r0, c0, _r1, _c1, inp_box, out_box in placements:
            box_h, box_w = inp_box.shape
            inp[r0 : r0 + box_h, c0 : c0 + box_w] = inp_box
            out[r0 : r0 + box_h, c0 : c0 + box_w] = out_box
        return inp, out
    return None


def _box_modes(count: int) -> list[str]:
    if count == 2:
        modes = ["h_fallback", "v_fallback"]
    else:
        modes = ["h_fallback", "v_fallback", "rot_only"]
    random.shuffle(modes)
    return modes


def _gen_attempt() -> np.ndarray | None:
    count = random.choice(BOX_COUNT_CHOICES)
    marker_colors = random.sample(MARKER_COLORS, count)
    boxes = [_build_box(mode, color) for mode, color in zip(_box_modes(count), marker_colors)]
    placed = _place_boxes(boxes)
    if placed is None:
        return None

    inp, expected = placed
    for color in PATTERN_COLORS:
        if int(np.count_nonzero(inp == color)) == 3:
            return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle-4 input")
