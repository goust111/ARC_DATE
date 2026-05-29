# Combo variant of 58f5dbd5 (index 19)
# combo_spec: D7/0+D10/1+D17/1
# Name: 加法填充+motif轨迹涂色+motif位于方块内部

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import Color, bbox_h, bbox_w, find_components, get_bg, is_contiguous, paste


# Design analysis for the combo variant
# a) Pixel concretization and behavior trace:
#    - Original task: a solid 5x5 block and a same-colored external motif are matched by color; the motif is
#      centered on the block and carved out as background, then the scene is discarded and only packed carved
#      blocks remain.
#    - This combo changes three stages at once:
#      1. D7/0 reverses the stencil action, so the destination object becomes a hollow 5x5 frame and the motif
#         adds color into its empty middle instead of removing color from a solid square.
#      2. D17/1 removes the separate motif region: each motif now starts already inside its own frame, placed
#         off-center inside the 3x3 cavity, so the solver must read the embedded shape locally.
#      3. D10/1 keeps the full canvas and records motion: after the motif is moved to the canonical centered
#         position, the old-only cells and the short connector from old pose to new pose become a fixed trail
#         color rather than disappearing.
#    - Concrete example: a red 5x5 frame may contain a red 2x2 motif in the cavity's bottom-right corner.
#      The output keeps that frame in place, paints the same 2x2 motif in the centered destination pose
#      (top-left within the 3x3 cavity for a 2x2 bbox), and recolors the vacated bottom/right cells plus the
#      connector path to black. The original solver would instead have needed a separate external red sprite,
#      would have carved holes, and would have packed tiles into a new grid.
# b) Complexity calibration:
#    - The task stays visually nontrivial because each example still has 3-6 distinct colored blocks, and the
#      embedded motifs vary in shape and offset. The solver cannot merely copy pixels; it must detect the frame,
#      isolate the interior motif, infer its minimal bbox, recenter it, and mark the move trace.
#    - The design avoids overcomplication by using a small family of motif bbox sizes (1x3, 3x1, 2x2, 2x3, 3x2)
#      that all fit cleanly inside the 3x3 cavity and always admit at least one visibly displaced source pose.
#      That gives repeated evidence across examples without introducing many unrelated cases.
# c) Implicit-assumption audit against the original main():
#    - Original block detection assumed a solid 5x5 monochrome square. That no longer holds; we now detect a
#      5x5 patch whose outer ring is monochrome and whose cavity contains a connected same-colored embedded motif.
#    - Original motif matching assumed the smallest same-colored non-block component elsewhere in the canvas was
#      the source. That assumption is invalid because the motif is now inside the matched block; the new solver
#      must inspect each frame's own 3x3 cavity.
#    - Original transformation assumed subtractive carving on a fresh solid tile. That is reversed to additive
#      filling on a fresh frame tile.
#    - Original output packing is invalid after D10/1. The new output remains on the original canvas and replaces
#      each 5x5 block in place with its filled-and-trailed version.
# d) Placement legality:
#    - The generator keeps the original 5x5 block lattice with 1-cell gaps, so blocks never overlap and every
#      frame still has an isolated 5x5 bbox.
#    - Each source motif is fully contained in the frame's 3x3 cavity and chosen from a connected mask whose
#      minimal bbox is one of the supported sizes. The chosen source offset is always different from the centered
#      destination offset, so the move is nondegenerate.
#    - Because all motion happens inside each block's own cavity, trail pixels never cross into neighboring blocks
#      or out of bounds.
# e) Visual observability:
#    - Across a few examples, a human can see the same local pattern repeated: every hollow frame contains a
#      same-colored inset motif; the output keeps the frame, shifts that motif into the canonical centered pose,
#      and leaves black residue/path pixels behind. This is a strong visible contrast from both the original task
#      and the pure D7 frame-filling variant.
#    - No hidden ordering by color id is used. Colors only indicate object identity; the only special color is the
#      fixed trail color, which is visible in the output and never used as background.
# f) Visual cleanliness:
#    - There is only one object family on the canvas: spaced 5x5 hollow frames. Each frame contains exactly one
#      small interior motif, so local parsing is clean.
#    - Trails are short and confined to the 3x3 cavity, preventing the full-grid clutter that long external paths
#      could introduce.
# g) Color-rule inferability and new solver assumptions:
#    - Colors are not remapped except for the trail behavior: the moved motif's destination stays the block color,
#      while vacated/path cells become black.
#    - The new main() assumes: background is not black; every target frame is an exact 5x5 ring; each frame's
#      cavity contains exactly one connected same-colored motif component; that motif's minimal bbox belongs to the
#      supported family; and its source pose differs from the canonical centered pose. generate_input() enforces all
#      of these assumptions explicitly.


TRAIL_COLOR = int(Color.BLACK)

DST_BLOCK_SIZE = 5
DST_BLOCK_STEP = 6
INNER_SIZE = 3
INNER_OFFSET = 1

FRAME_MASK = np.ones((DST_BLOCK_SIZE, DST_BLOCK_SIZE), dtype=bool)
FRAME_MASK[INNER_OFFSET : INNER_OFFSET + INNER_SIZE, INNER_OFFSET : INNER_OFFSET + INNER_SIZE] = False


def mask_from_coords(coords, shape: tuple[int, int]) -> np.ndarray:
    """Build a local boolean mask with True at the given in-bounds coordinates."""
    out = np.zeros(shape, dtype=bool)
    H, W = shape
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if 0 <= rr < H and 0 <= cc < W:
            out[rr, cc] = True
    return out


def _make_frame_tile(bg: int, color: int) -> np.ndarray:
    tile = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), int(bg), dtype=int)
    tile[FRAME_MASK] = int(color)
    return tile


def _is_embedded_block_patch(patch: np.ndarray, color: int, bg: int) -> bool:
    """Return whether patch is a valid 5x5 hollow block with an embedded same-colored motif."""
    if patch.shape != (DST_BLOCK_SIZE, DST_BLOCK_SIZE):
        return False
    if not bool((patch[FRAME_MASK] == int(color)).all()):
        return False

    inner = patch[INNER_OFFSET : INNER_OFFSET + INNER_SIZE, INNER_OFFSET : INNER_OFFSET + INNER_SIZE]
    if not np.isin(inner, [int(bg), int(color)]).all():
        return False

    inner_mask = inner == int(color)
    if not inner_mask.any():
        return False
    return bool(is_contiguous(inner_mask.astype(int), background=0, connectivity=8))


def _extract_embedded_motif(
    grid: np.ndarray, r0: int, c0: int, color: int
) -> tuple[tuple[int, int, int, int], np.ndarray]:
    """Read the connected same-colored motif component from a frame's 3x3 cavity."""
    inner = grid[
        r0 + INNER_OFFSET : r0 + INNER_OFFSET + INNER_SIZE,
        c0 + INNER_OFFSET : c0 + INNER_OFFSET + INNER_SIZE,
    ]
    mask = inner == int(color)
    coords = np.argwhere(mask)
    assert coords.size > 0
    assert is_contiguous(mask.astype(int), background=0, connectivity=8)
    rr0, cc0 = coords.min(axis=0)
    rr1, cc1 = coords.max(axis=0)
    cropped = mask[int(rr0) : int(rr1) + 1, int(cc0) : int(cc1) + 1].copy()
    src_bbox = (
        int(INNER_OFFSET + rr0),
        int(INNER_OFFSET + cc0),
        int(INNER_OFFSET + rr1),
        int(INNER_OFFSET + cc1),
    )
    return src_bbox, cropped


def _center_of_bbox(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, bbox)
    return int((r0 + r1) // 2), int((c0 + c1) // 2)


def _cells_on_axis_path(src: tuple[int, int], dst: tuple[int, int]) -> list[tuple[int, int]]:
    """Return a deterministic L-path from src to dst, including endpoints."""
    sr, sc = map(int, src)
    tr, tc = map(int, dst)
    if sr == tr:
        step = 1 if tc >= sc else -1
        return [(int(sr), int(c)) for c in range(sc, tc + step, step)]
    if sc == tc:
        step = 1 if tr >= sr else -1
        return [(int(r), int(sc)) for r in range(sr, tr + step, step)]

    cells = [(int(sr), int(sc))]
    step_r = 1 if tr > sr else -1
    for r in range(sr + step_r, tr + step_r, step_r):
        cells.append((int(r), int(sc)))
    step_c = 1 if tc > sc else -1
    for c in range(sc + step_c, tc + step_c, step_c):
        cells.append((int(tr), int(c)))
    return cells


def _destination_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return the canonical centered bbox for a motif mask inside the 3x3 cavity."""
    h, w = map(int, mask.shape)
    off_r = int(INNER_OFFSET + (INNER_SIZE - h) // 2)
    off_c = int(INNER_OFFSET + (INNER_SIZE - w) // 2)
    return (off_r, off_c, off_r + h - 1, off_c + w - 1)


def _paint_mask(tile: np.ndarray, bbox: tuple[int, int, int, int], mask: np.ndarray, color: int) -> np.ndarray:
    """Paint mask cells into tile and return the resulting tile mask."""
    out_mask = np.zeros(tile.shape, dtype=bool)
    r0, c0, _r1, _c1 = map(int, bbox)
    for rr, cc in np.argwhere(mask):
        tr = int(r0 + int(rr))
        tc = int(c0 + int(cc))
        tile[tr, tc] = int(color)
        out_mask[tr, tc] = True
    return out_mask


def _render_block_output(bg: int, color: int, src_bbox: tuple[int, int, int, int], mask: np.ndarray) -> np.ndarray:
    """
    Build the output 5x5 tile for one frame.

    Destination color is painted first; then only vacated source cells and the short connector remain trail-colored.
    Shared source/destination cells stay in the block color so the filled motif is not obscured by the trail.
    """
    tile = _make_frame_tile(bg, color)
    dst_bbox = _destination_bbox(mask)

    dst_cells = _paint_mask(tile, dst_bbox, mask, color)
    src_cells = np.zeros(tile.shape, dtype=bool)
    src_r0, src_c0, _src_r1, _src_c1 = map(int, src_bbox)
    for rr, cc in np.argwhere(mask):
        src_cells[src_r0 + int(rr), src_c0 + int(cc)] = True

    vacated = src_cells & ~dst_cells
    tile[vacated] = int(TRAIL_COLOR)

    for r, c in _cells_on_axis_path(_center_of_bbox(src_bbox), _center_of_bbox(dst_bbox)):
        if not dst_cells[int(r), int(c)]:
            tile[int(r), int(c)] = int(TRAIL_COLOR)

    return tile


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recenter embedded motifs inside hollow blocks, fill them additively, and mark the motion trail.

    Input structure:
    - bg = most common color, and black is reserved as a non-background trail color.
    - Several 5x5 monochrome square frames appear on a 6-step lattice, each a distinct non-bg color.
    - Inside each frame's 3x3 cavity, one same-colored connected motif sits in an off-center pose entirely within the cavity.

    Rule:
    - Detect each hollow 5x5 frame and read the motif already embedded inside its empty middle.
    - Move that motif to the canonical centered pose within the cavity and paint it in the frame color, keeping the full canvas in place.
    - Recolor the vacated source-only cells and the short source-to-destination connector to black as the motion trail.

    Key insight: The motif is no longer a separate stencil to match elsewhere; it is hidden inside the destination block, and the output shows both the additive fill result and the local path of that internal move.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    assert int(bg) != int(TRAIL_COLOR)

    blocks: list[dict[str, int]] = []
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    for comp in comps:
        bbox = comp["bbox"]
        if bbox_h(bbox) != DST_BLOCK_SIZE or bbox_w(bbox) != DST_BLOCK_SIZE:
            continue
        r0, c0, _r1, _c1 = map(int, bbox)
        color = int(comp["color"])
        patch = g[r0 : r0 + DST_BLOCK_SIZE, c0 : c0 + DST_BLOCK_SIZE]
        if _is_embedded_block_patch(patch, color, bg):
            blocks.append({"color": int(color), "r0": int(r0), "c0": int(c0)})

    assert blocks

    out = np.array(g, copy=True)
    for block in blocks:
        color = int(block["color"])
        r0 = int(block["r0"])
        c0 = int(block["c0"])
        src_bbox, mask = _extract_embedded_motif(g, r0, c0, color)
        tile = _render_block_output(bg, color, src_bbox, mask)
        out[r0 : r0 + DST_BLOCK_SIZE, c0 : c0 + DST_BLOCK_SIZE] = tile

    return out


# --------------- generator (combo variant) ---------------

CANVAS_HEIGHT_RANGE = (18, 22)
CANVAS_WIDTH_RANGE = (18, 24)
COLOR_COUNT = 10

DST_LAYOUTS = {
    3: [(1, 3), (3, 1)],
    4: [(2, 2)],
    6: [(2, 3), (3, 2)],
}

OUTPUT_BORDER = 1
OUTER_MARGIN = 1
GENERATION_RETRIES = 300

MOTIF_BBOXES = [(1, 3), (2, 2), (2, 3), (3, 1), (3, 2)]
MOTIF_AREA_RANGE = (3, 6)


def _spanned_size(count: int, item_size: int, step: int) -> int:
    return (count - 1) * step + item_size


def _make_masks_by_shape() -> dict[tuple[int, int], list[np.ndarray]]:
    """Precompute connected masks whose minimal bbox is exactly the requested shape."""
    masks: dict[tuple[int, int], list[np.ndarray]] = {shape: [] for shape in MOTIF_BBOXES}
    min_area, max_area = MOTIF_AREA_RANGE
    for h, w in MOTIF_BBOXES:
        for bits in itertools.product([False, True], repeat=h * w):
            m = np.array(bits, dtype=bool).reshape(h, w)
            area = int(m.sum())
            if not (min_area <= area <= max_area):
                continue
            if not m[0].any() or not m[-1].any() or not m[:, 0].any() or not m[:, -1].any():
                continue
            if is_contiguous(m.astype(int), background=0, connectivity=8):
                masks[(h, w)].append(m)
    return masks


MASKS_BY_SHAPE = _make_masks_by_shape()


def _center_offset(shape: tuple[int, int]) -> tuple[int, int]:
    h, w = map(int, shape)
    return int((INNER_SIZE - h) // 2), int((INNER_SIZE - w) // 2)


def _source_offsets(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    center = _center_offset(shape)
    offsets = [
        (int(dr), int(dc))
        for dr in range(INNER_SIZE - h + 1)
        for dc in range(INNER_SIZE - w + 1)
        if (int(dr), int(dc)) != center
    ]
    assert offsets
    return offsets


SOURCE_OFFSETS = {shape: _source_offsets(shape) for shape in MOTIF_BBOXES}


def _choose_palette(bg: int, count: int) -> list[int]:
    palette = [c for c in range(COLOR_COUNT) if c not in {int(bg), int(TRAIL_COLOR)}]
    random.shuffle(palette)
    return palette[:count]


def _choose_config(H: int, W: int) -> tuple[int, int] | None:
    configs: list[tuple[int, int]] = []
    for _count, layouts in DST_LAYOUTS.items():
        for R, C in layouts:
            need_h = _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP) + 2 * OUTER_MARGIN
            need_w = _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP) + 2 * OUTER_MARGIN
            if need_h <= H and need_w <= W:
                configs.append((R, C))
    if not configs:
        return None
    counts = sorted({int(R * C) for R, C in configs})
    count = random.choice(counts)
    return random.choice([cfg for cfg in configs if cfg[0] * cfg[1] == count])


def _place_block_region(H: int, W: int, R: int, C: int) -> tuple[int, int]:
    span_h = _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    span_w = _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    r0 = random.randint(OUTER_MARGIN, H - OUTER_MARGIN - span_h)
    c0 = random.randint(OUTER_MARGIN, W - OUTER_MARGIN - span_w)
    return int(r0), int(c0)


def _random_mask(shape: tuple[int, int]) -> np.ndarray:
    return random.choice(MASKS_BY_SHAPE[shape]).copy()


def _random_embedded_motif() -> tuple[np.ndarray, tuple[int, int]]:
    shape = random.choice(MOTIF_BBOXES)
    mask = _random_mask(shape)
    offset = random.choice(SOURCE_OFFSETS[shape])
    return mask, offset


def _paint_embedded_motif(
    grid: np.ndarray, block_r: int, block_c: int, color: int, mask: np.ndarray, offset: tuple[int, int]
) -> None:
    patch = np.full(mask.shape, int(color), dtype=int)
    paste(
        grid,
        patch,
        block_r + INNER_OFFSET + int(offset[0]),
        block_c + INNER_OFFSET + int(offset[1]),
        mask=mask,
    )


def _render_expected(
    inp: np.ndarray,
    bg: int,
    assignments: list[dict[str, object]],
) -> np.ndarray:
    out = np.array(inp, dtype=int)
    for item in assignments:
        tile = _render_block_output(
            bg,
            int(item["color"]),
            tuple(item["src_bbox"]),
            np.array(item["mask"], dtype=bool),
        )
        br0 = int(item["block_r"])
        bc0 = int(item["block_c"])
        out[br0 : br0 + DST_BLOCK_SIZE, bc0 : bc0 + DST_BLOCK_SIZE] = tile
    return out


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*CANVAS_HEIGHT_RANGE)
    W = random.randint(*CANVAS_WIDTH_RANGE)
    bg = random.choice([c for c in range(COLOR_COUNT) if c != int(TRAIL_COLOR)])

    config = _choose_config(H, W)
    if config is None:
        return None
    R, C = config
    base_r, base_c = _place_block_region(H, W, R, C)

    count = R * C
    block_colors = _choose_palette(bg, count)
    inp = np.full((H, W), int(bg), dtype=int)
    assignments: list[dict[str, object]] = []

    for i in range(R):
        for j in range(C):
            color = int(block_colors[i * C + j])
            br0 = int(base_r + i * DST_BLOCK_STEP)
            bc0 = int(base_c + j * DST_BLOCK_STEP)
            paste(inp, _make_frame_tile(bg, color), br0, bc0)

            mask, src_offset = _random_embedded_motif()
            _paint_embedded_motif(inp, br0, bc0, color, mask, src_offset)

            h, w = map(int, mask.shape)
            src_bbox = (
                int(INNER_OFFSET + src_offset[0]),
                int(INNER_OFFSET + src_offset[1]),
                int(INNER_OFFSET + src_offset[0] + h - 1),
                int(INNER_OFFSET + src_offset[1] + w - 1),
            )
            assert src_bbox != _destination_bbox(mask)

            assignments.append(
                {
                    "color": color,
                    "block_r": br0,
                    "block_c": bc0,
                    "src_bbox": src_bbox,
                    "mask": mask,
                }
            )

    expected = _render_expected(inp, bg, assignments)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 58f5dbd5 combo input")
