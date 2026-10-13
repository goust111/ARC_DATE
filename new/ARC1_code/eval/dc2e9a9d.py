from __future__ import annotations

import numpy as np

from arc_utils import Color, crop_bbox, find_components, get_bg, paste
import random


def _tab_side(patch: np.ndarray) -> str:
    mask = patch == Color.GREEN
    h, w = map(int, mask.shape)
    boundary_counts = {
        "top": int(mask[0, :].sum()),
        "bottom": int(mask[h - 1, :].sum()),
        "left": int(mask[:, 0].sum()),
        "right": int(mask[:, w - 1].sum()),
    }
    sides = [side for side, count in boundary_counts.items() if count == 1]
    assert len(sides) == 1, "expected exactly one one-pixel tab side"
    return sides[0]


def _assert_background_targets(
    canvas: np.ndarray,
    mask: np.ndarray,
    top: int,
    left: int,
    bg: int,
) -> None:
    height, width = map(int, canvas.shape)
    patch_h, patch_w = map(int, mask.shape)

    r0 = max(0, int(top))
    c0 = max(0, int(left))
    r1 = min(height, int(top) + patch_h)
    c1 = min(width, int(left) + patch_w)
    assert r0 < r1 and c0 < c1, "expected reflected copy to remain partly visible"

    visible_mask = mask[r0 - int(top) : r1 - int(top), c0 - int(left) : c1 - int(left)]
    assert int(visible_mask.sum()) > 0, "expected reflected green pixels in bounds"
    assert np.all(canvas[r0:r1, c0:c1][visible_mask] == int(bg)), "reflected copy must land on background"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Add a mirrored colored copy of each green tabbed outline on the side opposite its one-pixel tab.

    Input structure
    - The black background contains one or more 4-connected green outline objects.
    - Each green object has a bounding box with exactly one side containing a single green tab pixel.
    - The intended reflected copy is separated from the original bounding box by one background row or column and may be clipped by the grid edge.

    Rule
    - If the tab is on the left or right, mirror the object horizontally and place it on the opposite side in blue.
    - If the tab is on the top or bottom, mirror the object vertically and place it on the opposite side in teal.
    - Keep every original green pixel unchanged and paint only background cells reached by the reflected copy.

    Key insight
    - The lone boundary pixel identifies both the mirror axis and the side to avoid; the copy appears across the object's opposite bbox side with a one-cell gap.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    assert bg != Color.GREEN, "green objects must be foreground"
    assert set(map(int, np.unique(g))) <= {int(bg), Color.GREEN}, "expected only background and green input pixels"

    out = g.copy()
    components = find_components(g, bg=bg, colors=[Color.GREEN], connectivity=4, monochromatic=True)
    assert components, "expected at least one green object"

    for component in components:
        r0, c0, r1, c1 = map(int, component["bbox"])
        patch = crop_bbox(g, component["bbox"])
        mask = patch == Color.GREEN
        assert int(mask.sum()) == int(component["area"]), "component bbox contains unexpected green pixels"

        side = _tab_side(patch)
        h, w = map(int, patch.shape)
        if side in {"left", "right"}:
            reflected = np.fliplr(mask)
            color = Color.BLUE
            top = r0
            left = c1 + 2 if side == "left" else c0 - w - 1
        else:
            reflected = np.flipud(mask)
            color = Color.TEAL
            top = r1 + 2 if side == "top" else r0 - h - 1
            left = c0

        _assert_background_targets(out, reflected, top, left, bg)
        stamp = np.full_like(patch, int(bg))
        stamp[reflected] = int(color)
        paste(out, stamp, top, left, transparent=bg, clip=True)

    return out

# --------------- generator (o5x) ---------------

from arc_utils import bbox_coords

# Canvas bounds.
GEN_ROW_RANGE = (20, 30)
GEN_COL_RANGE = (17, 30)

# Object count and tab-side selection.
GEN_OBJECT_COUNT_RANGE = (2, 4)
GEN_LR_SIDES = ("left", "right")
GEN_TB_SIDES = ("top", "bottom")
GEN_ALL_SIDES = GEN_LR_SIDES + GEN_TB_SIDES

# Frame size constraints; tab-parallel dimensions are odd.
GEN_LR_FRAME_HEIGHTS = (3, 5)
GEN_LR_FRAME_WIDTH_RANGE = (3, 6)
GEN_TB_FRAME_HEIGHT_RANGE = (3, 5)
GEN_TB_FRAME_WIDTHS = (3, 5)

# Placement and validation.
GEN_BACKGROUND = int(Color.BLACK)
GEN_OBJECT_COLOR = int(Color.GREEN)
GEN_REFLECTION_GAP = 1
GEN_MAX_ATTEMPTS = 300

GenBBox = tuple[int, int, int, int]
GenPlacement = tuple[int, int, np.ndarray, int, int, int, GenBBox, GenBBox]


def _gen_draw_outline(patch: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    bottom = int(top) + int(height) - 1
    right = int(left) + int(width) - 1
    patch[top, left : right + 1] = GEN_OBJECT_COLOR
    patch[bottom, left : right + 1] = GEN_OBJECT_COLOR
    patch[top : bottom + 1, left] = GEN_OBJECT_COLOR
    patch[top : bottom + 1, right] = GEN_OBJECT_COLOR


def _gen_frame_dimensions(side: str) -> tuple[int, int]:
    if side in GEN_LR_SIDES:
        return random.choice(GEN_LR_FRAME_HEIGHTS), random.randint(*GEN_LR_FRAME_WIDTH_RANGE)
    return random.randint(*GEN_TB_FRAME_HEIGHT_RANGE), random.choice(GEN_TB_FRAME_WIDTHS)


def _gen_make_frame_patch(side: str) -> np.ndarray:
    rect_h, rect_w = _gen_frame_dimensions(side)
    if side in GEN_LR_SIDES:
        patch = np.full((rect_h, rect_w + 1), GEN_BACKGROUND, dtype=int)
        frame_left = 1 if side == "left" else 0
        tab_col = 0 if side == "left" else rect_w
        _gen_draw_outline(patch, 0, frame_left, rect_h, rect_w)
        patch[rect_h // 2, tab_col] = GEN_OBJECT_COLOR
        return patch

    patch = np.full((rect_h + 1, rect_w), GEN_BACKGROUND, dtype=int)
    frame_top = 1 if side == "top" else 0
    tab_row = 0 if side == "top" else rect_h
    _gen_draw_outline(patch, frame_top, 0, rect_h, rect_w)
    patch[tab_row, rect_w // 2] = GEN_OBJECT_COLOR
    return patch


def _gen_reflection(mask: np.ndarray, r0: int, c0: int, side: str) -> tuple[np.ndarray, int, int, int]:
    h, w = map(int, mask.shape)
    r1 = int(r0) + h - 1
    c1 = int(c0) + w - 1
    if side in GEN_LR_SIDES:
        reflected = np.fliplr(mask)
        top = int(r0)
        left = c1 + GEN_REFLECTION_GAP + 1 if side == "left" else int(c0) - w - GEN_REFLECTION_GAP
        return reflected, top, left, int(Color.BLUE)

    reflected = np.flipud(mask)
    top = r1 + GEN_REFLECTION_GAP + 1 if side == "top" else int(r0) - h - GEN_REFLECTION_GAP
    return reflected, top, int(c0), int(Color.TEAL)


def _gen_visible_coords_and_bbox(mask: np.ndarray, top: int, left: int, shape: tuple[int, int]):
    height, width = map(int, shape)
    coords = np.argwhere(mask)
    coords = coords + np.array([int(top), int(left)])
    keep = (
        (coords[:, 0] >= 0)
        & (coords[:, 0] < height)
        & (coords[:, 1] >= 0)
        & (coords[:, 1] < width)
    )
    visible = coords[keep]
    if visible.size == 0:
        return None, None
    return visible, bbox_coords(visible)


def _gen_bboxes_separated(a: GenBBox, b: GenBBox) -> bool:
    gap = GEN_REFLECTION_GAP
    return a[2] + gap < b[0] or b[2] + gap < a[0] or a[3] + gap < b[1] or b[3] + gap < a[1]


def _gen_all_separated(new_bboxes: list[GenBBox], existing_bboxes: list[GenBBox]) -> bool:
    for i, bbox_a in enumerate(new_bboxes):
        for bbox_b in new_bboxes[i + 1 :]:
            if not _gen_bboxes_separated(bbox_a, bbox_b):
                return False
        for bbox_b in existing_bboxes:
            if not _gen_bboxes_separated(bbox_a, bbox_b):
                return False
    return True


def _gen_cells_free(coords: np.ndarray, occupied: np.ndarray) -> bool:
    return bool(np.all(~occupied[coords[:, 0], coords[:, 1]]))


def _gen_valid_positions(
    grid_shape: tuple[int, int],
    patch: np.ndarray,
    side: str,
    occupied: np.ndarray,
    existing_bboxes: list[GenBBox],
) -> list[GenPlacement]:
    """Return all placements whose green object and visible reflection stay separated."""
    height, width = map(int, grid_shape)
    patch_h, patch_w = map(int, patch.shape)
    mask = patch == GEN_OBJECT_COLOR
    local_green = np.argwhere(mask)
    positions: list[GenPlacement] = []
    for r0 in range(height - patch_h + 1):
        for c0 in range(width - patch_w + 1):
            green_coords = local_green + np.array([r0, c0])
            if not _gen_cells_free(green_coords, occupied):
                continue

            green_bbox = (r0, c0, r0 + patch_h - 1, c0 + patch_w - 1)
            reflected, ref_top, ref_left, ref_color = _gen_reflection(mask, r0, c0, side)
            ref_coords, ref_bbox = _gen_visible_coords_and_bbox(reflected, ref_top, ref_left, grid_shape)
            if ref_coords is None or ref_bbox is None:
                continue
            if not _gen_cells_free(ref_coords, occupied):
                continue
            if not _gen_all_separated([green_bbox, ref_bbox], existing_bboxes):
                continue
            positions.append((r0, c0, reflected, ref_top, ref_left, ref_color, green_bbox, ref_bbox))
    return positions


def _gen_paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    stamp = np.full(mask.shape, GEN_BACKGROUND, dtype=int)
    stamp[mask] = int(color)
    paste(grid, stamp, int(top), int(left), transparent=GEN_BACKGROUND, clip=True)


def _gen_choose_sides(count: int) -> list[str]:
    if count == 2:
        sides = [random.choice(GEN_LR_SIDES), random.choice(GEN_TB_SIDES)]
    else:
        sides = random.sample(GEN_ALL_SIDES, count)
    random.shuffle(sides)
    return sides


def _gen_empty_scene() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[GenBBox], int]:
    rows = random.randint(*GEN_ROW_RANGE)
    cols = random.randint(*GEN_COL_RANGE)
    count = random.randint(*GEN_OBJECT_COUNT_RANGE)
    inp = np.full((rows, cols), GEN_BACKGROUND, dtype=int)
    out = np.full((rows, cols), GEN_BACKGROUND, dtype=int)
    occupied = np.zeros((rows, cols), dtype=bool)
    return inp, out, occupied, [], count


def _gen_place_pair(
    inp: np.ndarray,
    out: np.ndarray,
    occupied: np.ndarray,
    bboxes: list[GenBBox],
    side: str,
) -> np.ndarray | None:
    """Place one green object plus its expected reflected output, or fail this attempt."""
    patch = _gen_make_frame_patch(side)
    positions = _gen_valid_positions(inp.shape, patch, side, occupied, bboxes)
    if not positions:
        return None
    r0, c0, reflected, ref_top, ref_left, ref_color, green_bbox, ref_bbox = random.choice(positions)
    paste(inp, patch, r0, c0, transparent=GEN_BACKGROUND)
    paste(out, patch, r0, c0, transparent=GEN_BACKGROUND)
    _gen_paint_mask(out, reflected, ref_top, ref_left, ref_color)
    bboxes.extend([green_bbox, ref_bbox])
    return out != GEN_BACKGROUND


def _gen_matches_solver(inp: np.ndarray, out: np.ndarray) -> bool:
    """Oracle check: generated input must solve exactly to the constructed output."""
    try:
        solved = main(inp)
    except AssertionError:
        return False
    return bool(np.array_equal(solved, out))


def _gen_attempt() -> np.ndarray | None:
    inp, out, occupied, bboxes, count = _gen_empty_scene()

    for side in _gen_choose_sides(count):
        occupied = _gen_place_pair(inp, out, occupied, bboxes, side)
        if occupied is None:
            return None

    return inp if _gen_matches_solver(inp, out) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate dc2e9a9d input")
