# Combo variant of 6ffbe589 (index 25)
# combo_spec: D15/2+D20/0
# Name: 多板散布+带状层排列

from __future__ import annotations

import random

import numpy as np

from arc_utils import find_components, get_bg, paste

# -----------------------------------------------------------------------------
# Design analysis for the fused variant
#
# - D15/2: the input contains 2 separate boards scattered on the canvas.
# - D20/0: each board is a single connected 3-band strip, either top/middle/
#   bottom or left/middle/right. The three color layers are not concentric.
# - The original same-color cue rule is preserved, but each color rotates inside
#   its own local square band. This keeps the rotation effect visible while
#   avoiding cross-band collisions.
# -----------------------------------------------------------------------------


CANVAS_SIZE = 22
BOARD_COUNT = 2
CELLS_PER_BOARD = 3
BAND_SIZES = (4, 5)
MIN_BOARD_GAP = 1
MAX_UNION_SPAN = 20
CUE_GRID = 3
CUE_BOARD_DIST = 1
CUE_INTER_DIST = 1
GEN_RETRIES = 800


HOLE_TEMPLATES: dict[int, list[np.ndarray]] = {
    4: [
        np.array(
            [
                [0, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=bool,
        ),
        np.array(
            [
                [0, 0, 0, 0],
                [0, 1, 1, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 0],
            ],
            dtype=bool,
        ),
    ],
    5: [
        np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 1, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=bool,
        ),
        np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 1, 0, 1, 0],
                [0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=bool,
        ),
        np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=bool,
        ),
    ],
}


def _bbox_h(bbox: tuple[int, int, int, int]) -> int:
    return int(bbox[2] - bbox[0] + 1)


def _bbox_w(bbox: tuple[int, int, int, int]) -> int:
    return int(bbox[3] - bbox[1] + 1)


def _bbox_union(bboxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    return (
        min(bb[0] for bb in bboxes),
        min(bb[1] for bb in bboxes),
        max(bb[2] for bb in bboxes),
        max(bb[3] for bb in bboxes),
    )


def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    dr = max(0, br0 - ar1 - 1, ar0 - br1 - 1)
    dc = max(0, bc0 - ac1 - 1, ac0 - bc1 - 1)
    return max(int(dr), int(dc))


def _is_rot_asym(mask: np.ndarray) -> bool:
    return all(not np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3))


def _transform_mask(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    if random.randrange(2):
        out = np.fliplr(out)
    if random.randrange(2):
        out = np.flipud(out)
    return np.rot90(out, random.randrange(4))


def _make_band_patch(size: int, color: int) -> np.ndarray:
    for _ in range(50):
        holes = _transform_mask(random.choice(HOLE_TEMPLATES[int(size)]))
        patch = np.full((size, size), int(color), dtype=int)
        patch[holes] = 0
        if _is_rot_asym(patch == int(color)):
            return patch
    raise RuntimeError(f"failed to build asymmetric band patch for size={size}")


def _rotate_monochrome_patch(patch: np.ndarray, color: int, rot_k: int, *, bg: int = 0) -> np.ndarray:
    k = int(rot_k) % 4
    if k == 0:
        return patch.copy()
    mask = patch == int(color)
    rotated = np.rot90(mask, k)
    out = np.full_like(patch, int(bg))
    out[rotated] = int(color)
    return out


def _board_dims(orientation: str, size: int) -> tuple[int, int]:
    if orientation == "h":
        return (size * CELLS_PER_BOARD, size)
    return (size, size * CELLS_PER_BOARD)


def _board_offsets(orientation: str, size: int) -> list[tuple[int, int]]:
    if orientation == "h":
        return [(idx * size, 0) for idx in range(CELLS_PER_BOARD)]
    return [(0, idx * size) for idx in range(CELLS_PER_BOARD)]


def _board_band_bboxes(board_bbox: tuple[int, int, int, int], orientation: str, size: int) -> list[tuple[int, int, int, int]]:
    r0, c0, _, _ = board_bbox
    return [
        (r0 + dr, c0 + dc, r0 + dr + size - 1, c0 + dc + size - 1)
        for dr, dc in _board_offsets(orientation, size)
    ]


def _cue_patch(color: int, area: int) -> np.ndarray:
    center = CUE_GRID // 2
    coords = {(center, center)}
    while len(coords) < int(area):
        frontier = set()
        for r, c in coords:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < CUE_GRID and 0 <= nc < CUE_GRID and (nr, nc) not in coords:
                        frontier.add((nr, nc))
        coords.add(random.choice(sorted(frontier)))

    patch = np.zeros((CUE_GRID, CUE_GRID), dtype=int)
    for r, c in coords:
        patch[r, c] = int(color)
    nz = np.argwhere(patch != 0)
    r0, c0 = nz.min(axis=0)
    r1, c1 = nz.max(axis=0)
    return patch[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _cue_area_for_rot_k(rot_k: int) -> int:
    residue = (-int(rot_k)) % 4
    if residue == 0:
        return 0
    if residue == 1:
        return random.choice([1, 5])
    return residue


def _detect_boards(grid: np.ndarray, *, bg: int) -> list[dict]:
    comps = find_components(
        grid,
        bg=int(bg),
        connectivity=4,
        monochromatic=False,
        return_format="dict",
    )
    boards: list[dict] = []
    for comp in comps:
        bbox = tuple(map(int, comp["bbox"]))
        h, w = _bbox_h(bbox), _bbox_w(bbox)
        orientation = None
        size = None
        if h in BAND_SIZES and w == CELLS_PER_BOARD * h:
            orientation = "v"
            size = h
        elif w in BAND_SIZES and h == CELLS_PER_BOARD * w:
            orientation = "h"
            size = w
        if orientation is None or size is None:
            continue

        min_area = CELLS_PER_BOARD * (int(size) * int(size) - 5)
        if int(comp["area"]) < min_area:
            continue

        r0, c0, r1, c1 = bbox
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        band_colors: list[int] = []
        ok = True
        for band_bbox in _board_band_bboxes(bbox, orientation, int(size)):
            br0, bc0, br1, bc1 = band_bbox
            band = grid[br0 : br1 + 1, bc0 : bc1 + 1]
            colors = sorted(int(x) for x in np.unique(band) if int(x) != int(bg))
            if len(colors) != 1:
                ok = False
                break
            band_colors.append(colors[0])
        if not ok or len(set(band_colors)) != CELLS_PER_BOARD:
            continue
        boards.append(
            {
                "bbox": bbox,
                "orientation": orientation,
                "size": int(size),
                "colors": band_colors,
                "patch": patch,
            }
        )

    assert len(boards) == BOARD_COUNT, "expected exactly two scattered strip boards"
    return sorted(boards, key=lambda board: (board["bbox"][0], board["bbox"][1]))


def _cue_size_for_color(grid: np.ndarray, *, bg: int, color: int, band_bbox: tuple[int, int, int, int]) -> int:
    comps = find_components(
        grid,
        bg=int(bg),
        connectivity=8,
        monochromatic=True,
        colors=[int(color)],
        return_format="dict",
    )
    matches = [comp for comp in comps if tuple(map(int, comp["bbox"])) != tuple(map(int, band_bbox))]
    assert len(matches) <= 1, f"expected at most one cue for color {color}"
    return int(matches[0]["area"]) if matches else 0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect two scattered strip-boards and rotate each local color band from its cue.

    Input structure:
    - Background is the most common color.
    - There are exactly 2 disconnected boards on the canvas.
    - Each board is a connected 3-band strip: either top/middle/bottom bands or
      left/middle/right bands. Each band is a square monochrome patch with a few
      internal holes, so its rotation is visible.
    - Small same-color cue blobs live away from the boards.

    Rule:
    - Find both boards and crop the union of their bboxes.
    - Split each board into its 3 equal square bands.
    - For each band color, find the same-color cue. Cue area modulo 4 encodes the
      90-degree multiple used to rotate that band inside its own square.
    - If a band color has no cue, that band stays unchanged.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    boards = _detect_boards(g, bg=int(bg))
    crop_bbox = _bbox_union([board["bbox"] for board in boards])
    out = np.full((_bbox_h(crop_bbox), _bbox_w(crop_bbox)), int(bg), dtype=int)

    for board in boards:
        for color, band_bbox in zip(
            board["colors"],
            _board_band_bboxes(board["bbox"], board["orientation"], board["size"]),
        ):
            r0, c0, r1, c1 = band_bbox
            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            cue_size = _cue_size_for_color(g, bg=int(bg), color=int(color), band_bbox=band_bbox)
            rot_k = (-cue_size) % 4
            rotated = _rotate_monochrome_patch(patch, int(color), rot_k, bg=int(bg))
            paste(out, rotated, r0 - crop_bbox[0], c0 - crop_bbox[1], transparent=int(bg))

    return out


def _make_board(colors: list[int], orientation: str, size: int) -> dict:
    rot_ks = [random.randrange(4) for _ in range(CELLS_PER_BOARD)]
    if not any(rot_ks):
        rot_ks[random.randrange(CELLS_PER_BOARD)] = random.randint(1, 3)

    out_patches = [_make_band_patch(size, color) for color in colors]
    in_patches = [
        _rotate_monochrome_patch(patch, color, (-rot_k) % 4, bg=0)
        for patch, color, rot_k in zip(out_patches, colors, rot_ks)
    ]
    return {
        "size": size,
        "orientation": orientation,
        "colors": list(colors),
        "rot_ks": rot_ks,
        "out_patches": out_patches,
        "in_patches": in_patches,
    }


def _place_board(canvas: np.ndarray, board: dict, anchor_r: int, anchor_c: int, *, use_input: bool) -> tuple[tuple[int, int, int, int], list[tuple[int, int, int, int]]]:
    patches = board["in_patches"] if use_input else board["out_patches"]
    band_bboxes = []
    for (dr, dc), patch in zip(_board_offsets(board["orientation"], board["size"]), patches):
        rr = int(anchor_r + dr)
        cc = int(anchor_c + dc)
        paste(canvas, patch, rr, cc, transparent=0)
        size = int(board["size"])
        band_bboxes.append((rr, cc, rr + size - 1, cc + size - 1))
    return _bbox_union(band_bboxes), band_bboxes


def _render_expected(placed_boards: list[dict]) -> np.ndarray:
    crop_bbox = _bbox_union([board["bbox"] for board in placed_boards])
    out = np.zeros((_bbox_h(crop_bbox), _bbox_w(crop_bbox)), dtype=int)
    for board in placed_boards:
        for patch, bbox in zip(board["out_patches"], board["band_bboxes"]):
            paste(out, patch, bbox[0] - crop_bbox[0], bbox[1] - crop_bbox[1], transparent=0)
    return out


def _place_boards_on_canvas(boards: list[dict]) -> tuple[np.ndarray, list[dict]] | None:
    for _ in range(200):
        canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
        placed: list[dict] = []
        board_bboxes: list[tuple[int, int, int, int]] = []

        order = boards[:]
        random.shuffle(order)
        ok = True
        for board in order:
            h, w = _board_dims(board["orientation"], board["size"])
            candidates = []
            for r in range(CANVAS_SIZE - h + 1):
                for c in range(CANVAS_SIZE - w + 1):
                    bbox = (r, c, r + h - 1, c + w - 1)
                    if any(_bbox_gap(bbox, prev) < MIN_BOARD_GAP for prev in board_bboxes):
                        continue
                    test_union = _bbox_union(board_bboxes + [bbox]) if board_bboxes else bbox
                    if _bbox_h(test_union) > MAX_UNION_SPAN or _bbox_w(test_union) > MAX_UNION_SPAN:
                        continue
                    candidates.append((r, c, bbox))
            if not candidates:
                ok = False
                break

            anchor_r, anchor_c, bbox = random.choice(candidates)
            placed_bbox, band_bboxes = _place_board(canvas, board, anchor_r, anchor_c, use_input=True)
            placed.append(
                {
                    **board,
                    "anchor": (anchor_r, anchor_c),
                    "bbox": placed_bbox,
                    "band_bboxes": band_bboxes,
                }
            )
            board_bboxes.append(placed_bbox)

        if not ok:
            continue

        try:
            detected = _detect_boards(canvas, bg=0)
        except AssertionError:
            continue
        if {board["orientation"] for board in detected} != {"h", "v"}:
            continue
        return canvas, sorted(placed, key=lambda board: (board["bbox"][0], board["bbox"][1]))
    return None


def _place_cues(canvas: np.ndarray, placed_boards: list[dict]) -> bool:
    board_bboxes = [board["bbox"] for board in placed_boards]
    cue_bboxes: list[tuple[int, int, int, int]] = []

    cue_items = []
    for board in placed_boards:
        for color, rot_k in zip(board["colors"], board["rot_ks"]):
            area = _cue_area_for_rot_k(rot_k)
            if area:
                cue_items.append((int(color), int(area)))
    random.shuffle(cue_items)

    for color, area in cue_items:
        patch = _cue_patch(color, area)
        h, w = patch.shape
        candidates = []
        mask = patch != 0
        for r in range(CANVAS_SIZE - h + 1):
            for c in range(CANVAS_SIZE - w + 1):
                if np.any((canvas[r : r + h, c : c + w] != 0) & mask):
                    continue
                bbox = (r, c, r + h - 1, c + w - 1)
                if any(_bbox_gap(bbox, board_bbox) < CUE_BOARD_DIST for board_bbox in board_bboxes):
                    continue
                if any(_bbox_gap(bbox, cue_bbox) < CUE_INTER_DIST for cue_bbox in cue_bboxes):
                    continue
                candidates.append((r, c, bbox))
        if not candidates:
            return False
        r, c, bbox = random.choice(candidates)
        paste(canvas, patch, r, c, transparent=0)
        cue_bboxes.append(bbox)
    return True


def _gen_attempt() -> np.ndarray | None:
    colors = random.sample(range(1, 10), BOARD_COUNT * CELLS_PER_BOARD)
    orientations = random.sample(["h", "v"], k=BOARD_COUNT)
    sizes = list(BAND_SIZES)
    random.shuffle(sizes)
    boards = [
        _make_board(
            colors[i * CELLS_PER_BOARD : (i + 1) * CELLS_PER_BOARD],
            orientations[i],
            sizes[i],
        )
        for i in range(BOARD_COUNT)
    ]
    if not all(any(rot_k != 0 for rot_k in board["rot_ks"]) for board in boards):
        return None

    placed_result = _place_boards_on_canvas(boards)
    if placed_result is None:
        return None
    canvas, placed_boards = placed_result

    if not _place_cues(canvas, placed_boards):
        return None

    expected = _render_expected(placed_boards)
    try:
        actual = main(canvas)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    try:
        detected = _detect_boards(canvas, bg=0)
    except AssertionError:
        return None
    if {board["orientation"] for board in detected} != {"h", "v"}:
        return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fused 6ffbe589 input")
