import random

import numpy as np


BG = 0
FRAME = 5
TAB_LEN = 2
SEP = 2
OUTER = 1
CANVAS = 30
GEN_RETRIES = 400
LAYOUT_RETRIES = 400
PLACEMENT_RETRIES = 250
PIECE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
SIDES = ("U", "R", "D", "L")


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _touches_border(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _rects_overlap(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _empty_spec() -> dict[str, int | None]:
    return {side: None for side in SIDES}


def _piece_mask(spec: dict[str, int | None], core_h: int, core_w: int) -> np.ndarray:
    top = 1 if spec["U"] is not None else 0
    bottom = 1 if spec["D"] is not None else 0
    left = 1 if spec["L"] is not None else 0
    right = 1 if spec["R"] is not None else 0
    mask = np.zeros((core_h + top + bottom, core_w + left + right), dtype=bool)
    mask[top : top + core_h, left : left + core_w] = True
    if spec["U"] is not None:
        s = int(spec["U"])
        mask[0, left + s : left + s + TAB_LEN] = True
    if spec["D"] is not None:
        s = int(spec["D"])
        mask[core_h + top, left + s : left + s + TAB_LEN] = True
    if spec["L"] is not None:
        s = int(spec["L"])
        mask[top + s : top + s + TAB_LEN, 0] = True
    if spec["R"] is not None:
        s = int(spec["R"])
        mask[top + s : top + s + TAB_LEN, core_w + left] = True
    return mask


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    sigs = []
    for k in range(4):
        rot = np.rot90(mask, k)
        sigs.append((tuple(int(x) for x in rot.shape), rot.tobytes()))
    return min(sigs)


def _frame_shape(rows: int, cols: int, core_h: int, core_w: int) -> tuple[int, int]:
    height = 2 * OUTER + rows * core_h + (rows - 1) * SEP
    width = 2 * OUTER + cols * core_w + (cols - 1) * SEP
    return int(height), int(width)


def _cell_origin(
    frame_top: int, frame_left: int, row: int, col: int, core_h: int, core_w: int
) -> tuple[int, int]:
    r0 = frame_top + OUTER + row * (core_h + SEP)
    c0 = frame_left + OUTER + col * (core_w + SEP)
    return int(r0), int(c0)


def _sample_specs(
    rows: int, cols: int, core_h: int, core_w: int, rng: random.Random
) -> list[list[dict[str, int | None]]] | None:
    vertical_starts = list(range(1, core_h - 2))
    horizontal_starts = list(range(1, core_w - 2))
    if not vertical_starts or not horizontal_starts:
        return None

    for _ in range(LAYOUT_RETRIES):
        specs = [[_empty_spec() for _ in range(cols)] for _ in range(rows)]
        for r in range(rows):
            for c in range(cols - 1):
                start = int(rng.choice(vertical_starts))
                if rng.random() < 0.5:
                    specs[r][c]["R"] = start
                else:
                    specs[r][c + 1]["L"] = start
        for r in range(rows - 1):
            for c in range(cols):
                start = int(rng.choice(horizontal_starts))
                if rng.random() < 0.5:
                    specs[r][c]["D"] = start
                else:
                    specs[r + 1][c]["U"] = start

        masks = [
            _piece_mask(specs[r][c], core_h, core_w)
            for r in range(rows)
            for c in range(cols)
        ]
        sigs = [_mask_signature(mask) for mask in masks]
        if len(set(sigs)) != rows * cols:
            continue
        return specs
    return None


def _render_frame_patch(
    rows: int, cols: int, core_h: int, core_w: int, specs: list[list[dict[str, int | None]]]
) -> tuple[np.ndarray, list[dict[str, object]]]:
    frame_h, frame_w = _frame_shape(rows, cols, core_h, core_w)
    patch = np.full((frame_h, frame_w), FRAME, dtype=int)
    holes: list[dict[str, object]] = []

    for r in range(rows):
        for c in range(cols):
            spec = specs[r][c]
            cell_r0 = OUTER + r * (core_h + SEP)
            cell_c0 = OUTER + c * (core_w + SEP)
            patch[cell_r0 : cell_r0 + core_h, cell_c0 : cell_c0 + core_w] = BG
            if spec["U"] is not None:
                s = int(spec["U"])
                patch[cell_r0 - 1, cell_c0 + s : cell_c0 + s + TAB_LEN] = BG
            if spec["D"] is not None:
                s = int(spec["D"])
                patch[cell_r0 + core_h, cell_c0 + s : cell_c0 + s + TAB_LEN] = BG
            if spec["L"] is not None:
                s = int(spec["L"])
                patch[cell_r0 + s : cell_r0 + s + TAB_LEN, cell_c0 - 1] = BG
            if spec["R"] is not None:
                s = int(spec["R"])
                patch[cell_r0 + s : cell_r0 + s + TAB_LEN, cell_c0 + core_w] = BG

            mask = _piece_mask(spec, core_h, core_w)
            top = cell_r0 - (1 if spec["U"] is not None else 0)
            left = cell_c0 - (1 if spec["L"] is not None else 0)
            holes.append(
                {
                    "row": int(r),
                    "col": int(c),
                    "top": int(top),
                    "left": int(left),
                    "mask": mask,
                }
            )
    return patch, holes


def _rotation_schedule(count: int, rng: random.Random) -> list[int]:
    rotations = [0, 1, 2, 3]
    while len(rotations) < count:
        rotations.append(rng.randrange(4))
    rng.shuffle(rotations)
    return [int(x) for x in rotations[:count]]


def _patch_from_mask(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = int(color)
    return patch


def _valid_positions(
    shape: tuple[int, int],
    placed_rects: list[tuple[int, int, int, int]],
) -> list[tuple[int, int]]:
    h, w = shape
    positions = []
    for r in range(1, CANVAS - h):
        for c in range(1, CANVAS - w):
            rect = (r, c, r + h - 1, c + w - 1)
            if any(_rects_overlap(rect, other, pad=1) for other in placed_rects):
                continue
            positions.append((int(r), int(c)))
    return positions


def _pack_layout(
    frame_shape: tuple[int, int],
    piece_shapes: list[tuple[int, int]],
    rng: random.Random,
) -> tuple[tuple[int, int], list[tuple[int, int]]] | None:
    frame_candidates = _valid_positions(frame_shape, [])
    if not frame_candidates:
        return None

    rng.shuffle(frame_candidates)
    center_r = CANVAS / 2.0
    center_c = CANVAS / 2.0
    frame_candidates.sort(
        key=lambda pos: (
            abs((pos[0] + frame_shape[0] / 2.0) - center_r)
            + abs((pos[1] + frame_shape[1] / 2.0) - center_c)
        )
    )

    for frame_top, frame_left in frame_candidates[: min(len(frame_candidates), 80)]:
        frame_rect = (
            frame_top,
            frame_left,
            frame_top + frame_shape[0] - 1,
            frame_left + frame_shape[1] - 1,
        )
        rects = [frame_rect]
        positions: list[tuple[int, int] | None] = [None] * len(piece_shapes)
        order = sorted(
            range(len(piece_shapes)),
            key=lambda idx: piece_shapes[idx][0] * piece_shapes[idx][1],
            reverse=True,
        )

        ok = True
        for idx in order:
            candidates = _valid_positions(piece_shapes[idx], rects)
            if not candidates:
                ok = False
                break
            pos = rng.choice(candidates)
            positions[idx] = pos
            rects.append(
                (pos[0], pos[1], pos[0] + piece_shapes[idx][0] - 1, pos[1] + piece_shapes[idx][1] - 1)
            )
        if ok and all(pos is not None for pos in positions):
            return (int(frame_top), int(frame_left)), [(int(r), int(c)) for r, c in positions]

    return None


def _place_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    region = grid[top : top + h, left : left + w]
    mask = patch != BG
    assert not np.any(region[mask] != BG), "patch placement overlap"
    region[mask] = patch[mask]


def _crop_with_margin(grid: np.ndarray, rng: random.Random) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    coords = np.argwhere(grid != BG)
    r0, c0, r1, c1 = _bbox(coords)
    top_margin = rng.randint(0, min(2, r0))
    left_margin = rng.randint(0, min(2, c0))
    bottom_margin = rng.randint(0, min(2, grid.shape[0] - 1 - r1))
    right_margin = rng.randint(0, min(2, grid.shape[1] - 1 - c1))
    cr0 = r0 - top_margin
    cc0 = c0 - left_margin
    cr1 = r1 + bottom_margin
    cc1 = c1 + right_margin
    return grid[cr0 : cr1 + 1, cc0 : cc1 + 1].copy(), (int(cr0), int(cc0), int(cr1), int(cc1))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each outside monochrome jigsaw piece into the uniquely matching grey slot inside the frame.

    Input structure:
    - bg = black. The largest non-background object is a grey rectangular frame subdividing the center into equal cells.
    - Each cell is an empty bg cavity whose sides may include a small 2x1 notch on an internal border; the neighboring cell keeps the complementary grey tab.
    - Outside the frame are disconnected single-color puzzle pieces made from a same-sized rectangular core plus zero or more outward 2x1 tabs.

    Rule:
    - Detect the grey frame and extract every enclosed bg cavity inside it as a target slot.
    - Detect each outside colored piece and compare its shape with every slot under the four right-angle rotations.
    - Each piece fits exactly one slot up to rotation; erase the piece from its old location and paint its color into the matched slot shape.
    - Leave the rest of the frame and background unchanged.

    Key insight:
    - Color does not choose the destination; only the tab-and-notch geometry matters, and the correct fit may require rotating the outside piece.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _mode_color(g)

    comps = _components(g != bg)
    assert comps, "expected a frame and pieces"
    frame_coords = max(comps, key=lambda pts: int(pts.shape[0]))
    fr0, fc0, fr1, fc1 = _bbox(frame_coords)

    sub = g[fr0 : fr1 + 1, fc0 : fc1 + 1]
    hole_components = [
        comp for comp in _components(sub == bg) if not _touches_border(comp, sub.shape)
    ]
    assert hole_components, "expected enclosed frame cavities"

    holes = []
    for comp in hole_components:
        hr0, hc0, hr1, hc1 = _bbox(comp)
        mask = np.zeros((hr1 - hr0 + 1, hc1 - hc0 + 1), dtype=bool)
        mask[comp[:, 0] - hr0, comp[:, 1] - hc0] = True
        holes.append(
            {
                "bbox": (fr0 + hr0, fc0 + hc0, fr0 + hr1, fc0 + hc1),
                "mask": mask,
            }
        )
    holes.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))

    pieces = []
    for comp in comps:
        if comp is frame_coords:
            continue
        pr0, pc0, pr1, pc1 = _bbox(comp)
        patch = g[pr0 : pr1 + 1, pc0 : pc1 + 1]
        colors = [int(v) for v in np.unique(patch) if int(v) != bg]
        assert len(colors) == 1, "each piece should be monochrome"
        color = colors[0]
        mask = patch == color
        pieces.append({"bbox": (pr0, pc0, pr1, pc1), "mask": mask, "color": int(color)})
    assert len(pieces) == len(holes), "piece count must equal slot count"

    candidate_pieces: list[list[int]] = []
    for hole in holes:
        hole_mask = hole["mask"]
        matches = []
        for idx, piece in enumerate(pieces):
            if any(np.array_equal(np.rot90(piece["mask"], k), hole_mask) for k in range(4)):
                matches.append(int(idx))
        assert matches, "each slot should have at least one matching piece"
        candidate_pieces.append(matches)

    assignment = [-1] * len(holes)
    used: set[int] = set()

    def search() -> bool:
        if len(used) == len(pieces):
            return True
        target = min(
            (
                idx
                for idx in range(len(holes))
                if assignment[idx] == -1
            ),
            key=lambda idx: sum(1 for p in candidate_pieces[idx] if p not in used),
        )
        options = [p for p in candidate_pieces[target] if p not in used]
        for piece_idx in options:
            assignment[target] = int(piece_idx)
            used.add(int(piece_idx))
            if search():
                return True
            used.remove(int(piece_idx))
            assignment[target] = -1
        return False

    assert search(), "slot-piece matching should be uniquely solvable"

    out = g.copy()
    for piece in pieces:
        pr0, pc0, pr1, pc1 = piece["bbox"]
        region = out[pr0 : pr1 + 1, pc0 : pc1 + 1]
        region[piece["mask"]] = bg

    for hole_idx, piece_idx in enumerate(assignment):
        hr0, hc0, hr1, hc1 = holes[hole_idx]["bbox"]
        region = out[hr0 : hr1 + 1, hc0 : hc1 + 1]
        region[holes[hole_idx]["mask"]] = pieces[piece_idx]["color"]

    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    rows, cols = rng.choice([(2, 2), (2, 3), (3, 2)])
    if (rows, cols) == (2, 2):
        core_h = rng.randint(4, 6)
        core_w = rng.randint(4, 6)
    elif (rows, cols) == (2, 3):
        core_h = rng.randint(6, 7)
        core_w = rng.randint(4, 5)
    else:
        core_h = rng.randint(4, 5)
        core_w = rng.randint(6, 7)

    specs = _sample_specs(rows, cols, core_h, core_w, rng)
    if specs is None:
        return None

    frame_patch, holes = _render_frame_patch(rows, cols, core_h, core_w, specs)
    piece_count = rows * cols
    colors = rng.sample(PIECE_COLORS, piece_count)
    rotations = _rotation_schedule(piece_count, rng)

    piece_patches = []
    for idx, hole in enumerate(holes):
        mask = np.rot90(hole["mask"], rotations[idx])
        piece_patches.append(_patch_from_mask(mask, colors[idx]))

    packed = _pack_layout(frame_patch.shape, [patch.shape for patch in piece_patches], rng)
    if packed is None:
        return None
    (frame_top, frame_left), piece_positions = packed

    full_in = np.full((CANVAS, CANVAS), BG, dtype=int)
    full_out = np.full((CANVAS, CANVAS), BG, dtype=int)
    _place_patch(full_in, frame_patch, frame_top, frame_left)
    _place_patch(full_out, frame_patch, frame_top, frame_left)

    for idx, hole in enumerate(holes):
        top = frame_top + int(hole["top"])
        left = frame_left + int(hole["left"])
        region = full_out[top : top + hole["mask"].shape[0], left : left + hole["mask"].shape[1]]
        region[hole["mask"]] = int(colors[idx])

    for patch, (top, left) in zip(piece_patches, piece_positions):
        _place_patch(full_in, patch, top, left)

    cropped_in, crop_box = _crop_with_margin(full_in, rng)
    cr0, cc0, cr1, cc1 = crop_box
    cropped_out = full_out[cr0 : cr1 + 1, cc0 : cc1 + 1].copy()
    solved = main(cropped_in)
    if not np.array_equal(solved, cropped_out):
        return None
    return cropped_in


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = random.Random()
        grid = _gen_attempt(rng)
        if grid is not None and max(grid.shape) <= 30:
            return grid
    raise RuntimeError("failed to generate a valid input")
