from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

from arc_utils import bbox, crop_bbox, find_components, get_bg, paste


BG = 0
ANCHOR = 2
MOTIF_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)
GRID_MIN = 24
GRID_MAX = 30
SIZE_STEP = 1
MAX_GENERATION_ATTEMPTS = 400
LAYOUT_ATTEMPTS = 80

MOTIF_LIBRARY = (
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 0, 0], [0, 1, 0, 0], [0, 1, 1, 1], [0, 0, 0, 1]], dtype=bool),
)

ATTACHMENTS = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
)

STEP_DIRECTIONS = ATTACHMENTS


def _resize_nn(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    src_h, src_w = map(int, img.shape)
    new_h = int(new_h)
    new_w = int(new_w)
    assert new_h >= 1 and new_w >= 1
    if (src_h, src_w) == (new_h, new_w):
        return img.copy()
    rr = np.floor(np.arange(new_h) * src_h / new_h).astype(int)
    cc = np.floor(np.arange(new_w) * src_w / new_w).astype(int)
    rr = np.clip(rr, 0, src_h - 1)
    cc = np.clip(cc, 0, src_w - 1)
    return img[rr[:, None], cc[None, :]]


def _square_anchor_components(grid: np.ndarray, bg: int) -> list[dict]:
    anchors = []
    for comp in find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[ANCHOR],
    ):
        r0, c0, r1, c1 = comp["bbox"]
        size = int(r1 - r0 + 1)
        if size != int(c1 - c0 + 1):
            continue
        patch = crop_bbox(grid, comp["bbox"])
        if not np.all(patch == ANCHOR):
            continue
        anchors.append(
            {
                "coords": comp["coords"],
                "bbox": comp["bbox"],
                "size": size,
                "center2": (int(r0 + r1), int(c0 + c1)),
            }
        )
    anchors.sort(key=lambda item: (item["size"], item["bbox"][0], item["bbox"][1]))
    return anchors


def _extract_source_patch(grid: np.ndarray, anchor: dict, bg: int) -> np.ndarray:
    anchor_pixels = {tuple(map(int, rc)) for rc in anchor["coords"].tolist()}
    for comp in find_components(grid, bg=bg, connectivity=8, monochromatic=False):
        if any(tuple(map(int, rc)) in anchor_pixels for rc in comp["coords"]):
            return crop_bbox(grid, comp["bbox"])
    raise AssertionError("expected the smallest anchor to belong to one foreground stamp")


def _round_half_up(value: float) -> int:
    return int(np.floor(float(value) + 0.5))


def _scale_signed_offset(value: int, source_size: int, target_size: int) -> int:
    sign = 1 if int(value) >= 0 else -1
    scaled = _round_half_up(abs(int(value)) * float(target_size) / float(source_size))
    return sign * scaled


def _source_spec_from_patch(source_patch: np.ndarray, source_size: int) -> dict:
    red_box = bbox(source_patch == ANCHOR)
    motif_mask = (source_patch != BG) & (source_patch != ANCHOR)
    motif_box = bbox(motif_mask)
    assert red_box is not None and motif_box is not None
    return {
        "source_size": int(source_size),
        "motif_patch": crop_bbox(source_patch, motif_box),
        "motif_offset": (
            int(motif_box[0] - red_box[0]),
            int(motif_box[1] - red_box[1]),
        ),
    }


def _render_stamp(source_spec: dict, target_size: int) -> dict:
    source_size = int(source_spec["source_size"])
    motif_src = source_spec["motif_patch"]
    ratio = float(target_size) / float(source_size)
    new_h = max(1, _round_half_up(motif_src.shape[0] * ratio))
    new_w = max(1, _round_half_up(motif_src.shape[1] * ratio))
    motif_patch = _resize_nn(motif_src, new_h, new_w)
    motif_offset = (
        _scale_signed_offset(source_spec["motif_offset"][0], source_size, target_size),
        _scale_signed_offset(source_spec["motif_offset"][1], source_size, target_size),
    )

    min_r = min(0, motif_offset[0])
    min_c = min(0, motif_offset[1])
    max_r = max(target_size - 1, motif_offset[0] + new_h - 1)
    max_c = max(target_size - 1, motif_offset[1] + new_w - 1)
    patch = np.full((max_r - min_r + 1, max_c - min_c + 1), BG, dtype=int)

    anchor_r0 = -min_r
    anchor_c0 = -min_c
    patch[anchor_r0 : anchor_r0 + target_size, anchor_c0 : anchor_c0 + target_size] = ANCHOR

    motif_r0 = motif_offset[0] - min_r
    motif_c0 = motif_offset[1] - min_c
    paste(patch, motif_patch, motif_r0, motif_c0, transparent=BG)

    fg_coords = np.argwhere(patch != BG)
    assert fg_coords.size > 0
    return {
        "patch": patch,
        "shape": patch.shape,
        "red_offset": (int(anchor_r0), int(anchor_c0)),
        "coords": fg_coords,
    }


def _anchor_top_left_from_center2(center2: tuple[int, int], size: int) -> tuple[int, int]:
    r_num = int(center2[0] - size + 1)
    c_num = int(center2[1] - size + 1)
    assert r_num % 2 == 0 and c_num % 2 == 0
    return r_num // 2, c_num // 2


def _patch_top_left(spec: dict, anchor_top_left: tuple[int, int]) -> tuple[int, int]:
    return (
        int(anchor_top_left[0] - spec["red_offset"][0]),
        int(anchor_top_left[1] - spec["red_offset"][1]),
    )


def _visible_coords(
    grid_shape: tuple[int, int],
    spec: dict,
    anchor_top_left: tuple[int, int],
) -> list[tuple[int, int]]:
    patch_r0, patch_c0 = _patch_top_left(spec, anchor_top_left)
    h, w = map(int, grid_shape)
    coords = []
    for rr, cc in spec["coords"].tolist():
        gr = patch_r0 + int(rr)
        gc = patch_c0 + int(cc)
        if 0 <= gr < h and 0 <= gc < w:
            coords.append((gr, gc))
    return coords


def _stamp_visible(
    out: np.ndarray,
    spec: dict,
    anchor_top_left: tuple[int, int],
) -> bool:
    if not _visible_coords(out.shape, spec, anchor_top_left):
        return False
    patch_r0, patch_c0 = _patch_top_left(spec, anchor_top_left)
    paste(out, spec["patch"], patch_r0, patch_c0, transparent=BG, clip=True)
    return True


def _main_impl(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30

    bg = int(get_bg(g))
    anchors = _square_anchor_components(g, bg)
    assert len(anchors) == 3, "expected exactly three red square anchors"

    sizes = [anchor["size"] for anchor in anchors]
    ds = int(sizes[1] - sizes[0])
    assert ds > 0 and sizes[2] - sizes[1] == ds

    center2 = [anchor["center2"] for anchor in anchors]
    delta = (
        int(center2[1][0] - center2[0][0]),
        int(center2[1][1] - center2[0][1]),
    )
    assert center2[2][0] - center2[1][0] == delta[0]
    assert center2[2][1] - center2[1][1] == delta[1]

    source_patch = _extract_source_patch(g, anchors[0], bg)
    source_size = int(sizes[0])
    source_spec = _source_spec_from_patch(source_patch, source_size)

    spec_cache: dict[int, dict] = {}

    def spec_for(size: int) -> dict:
        if size not in spec_cache:
            spec_cache[size] = _render_stamp(source_spec, size)
        return spec_cache[size]

    out = g.copy()

    next_center = (center2[-1][0] + delta[0], center2[-1][1] + delta[1])
    next_size = sizes[-1] + ds
    while next_size >= 1:
        spec = spec_for(next_size)
        anchor_top_left = _anchor_top_left_from_center2(next_center, next_size)
        if not _stamp_visible(out, spec, anchor_top_left):
            break
        next_center = (next_center[0] + delta[0], next_center[1] + delta[1])
        next_size += ds

    prev_center = (center2[0][0] - delta[0], center2[0][1] - delta[1])
    prev_size = sizes[0] - ds
    while prev_size >= 1:
        spec = spec_for(prev_size)
        anchor_top_left = _anchor_top_left_from_center2(prev_center, prev_size)
        if not _stamp_visible(out, spec, anchor_top_left):
            break
        prev_center = (prev_center[0] - delta[0], prev_center[1] - delta[1])
        prev_size -= ds

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Continue an arithmetic sequence of red-anchor stamps by extrapolating both anchor positions and motif scale.

    Input structure:
    - A black background contains exactly three copies of the same foreground stamp.
    - Each copy includes a solid red square anchor whose side lengths form a consecutive arithmetic sequence.
    - The non-red motif is attached to the anchor with the same local arrangement in every copy, and each copy is a nearest-neighbor scaled version of the smallest one.

    Rule:
    - Detect the three red square anchors, sort them by size, and read both the anchor-center step and the anchor-size step from that ordered sequence.
    - Extract the full smallest stamp as the source patch, then scale that patch by the ratio between a predicted anchor size and the smallest anchor size.
    - Keep extrapolating one step forward and one step backward along the same arithmetic center step, placing each scaled patch so its red square matches the predicted anchor position.
    - Paste only in-bounds pixels, stop in a direction when the next entire stamp is outside the canvas, and keep the original three copies unchanged.

    Key insight:
    - The same arithmetic progression simultaneously controls where the next anchor lands and how large the next stamped motif must be, so position and scale must be inferred together.
    """
    return _main_impl(grid)


def _transform_mask(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = mask.copy()
    if rng.integers(2):
        out = np.fliplr(out)
    out = np.rot90(out, int(rng.integers(4)))
    return out


def _overlap_start(anchor_size: int, motif_span: int, rng: np.random.Generator) -> int:
    starts = []
    for start in range(-motif_span + 1, anchor_size):
        end = start + motif_span - 1
        if max(0, start) <= min(anchor_size - 1, end):
            starts.append(start)
    return int(rng.choice(starts))


def _build_source_patch(
    mask: np.ndarray,
    anchor_size: int,
    motif_color: int,
    attachment: tuple[int, int],
    rng: np.random.Generator,
) -> np.ndarray:
    mh, mw = map(int, mask.shape)
    dr, dc = map(int, attachment)

    if dr < 0:
        motif_r0 = -mh
    elif dr > 0:
        motif_r0 = anchor_size
    else:
        motif_r0 = _overlap_start(anchor_size, mh, rng)

    if dc < 0:
        motif_c0 = -mw
    elif dc > 0:
        motif_c0 = anchor_size
    else:
        motif_c0 = _overlap_start(anchor_size, mw, rng)

    min_r = min(0, motif_r0)
    min_c = min(0, motif_c0)
    max_r = max(anchor_size - 1, motif_r0 + mh - 1)
    max_c = max(anchor_size - 1, motif_c0 + mw - 1)
    patch = np.full((max_r - min_r + 1, max_c - min_c + 1), BG, dtype=int)

    anchor_r0 = -min_r
    anchor_c0 = -min_c
    patch[anchor_r0 : anchor_r0 + anchor_size, anchor_c0 : anchor_c0 + anchor_size] = ANCHOR

    motif_rr = motif_r0 - min_r
    motif_cc = motif_c0 - min_c
    motif_view = patch[motif_rr : motif_rr + mh, motif_cc : motif_cc + mw]
    assert not np.any(mask & (motif_view != BG))
    motif_view[mask] = int(motif_color)
    return patch


def _specs_for_sizes(source_patch: np.ndarray, source_size: int, max_size: int) -> dict[int, dict]:
    source_spec = _source_spec_from_patch(source_patch, source_size)
    return {size: _render_stamp(source_spec, size) for size in range(1, max_size + 1)}


def _axis_candidates(
    grid_len: int,
    step: int,
    offsets: list[int],
    lengths: list[int],
    mode: str,
) -> list[int]:
    lo = -max(lengths) - abs(step) * 6 - 4
    hi = grid_len + max(lengths) + abs(step) * 6 + 4
    candidates = []
    for anchor3 in range(lo, hi + 1):
        ok = True
        for idx in range(3):
            anchor_i = anchor3 - (3 - idx) * step
            top_i = anchor_i - offsets[idx]
            if top_i < 0 or top_i + lengths[idx] > grid_len:
                ok = False
                break
        if not ok:
            continue

        anchor4 = anchor3 + step
        top3 = anchor3 - offsets[3]
        top4 = anchor4 - offsets[4]

        if mode == "inside":
            if not (0 <= top3 and top3 + lengths[3] <= grid_len):
                continue
        elif mode == "partial_pos":
            if not (top3 < grid_len and top3 + lengths[3] > grid_len):
                continue
            if not (top4 >= grid_len):
                continue
        elif mode == "partial_neg":
            if not (top3 < 0 and top3 + lengths[3] > 0):
                continue
            if not (top4 + lengths[4] <= 0):
                continue
        elif mode == "inside_to_out_pos":
            if not (0 <= top3 and top3 + lengths[3] <= grid_len):
                continue
            if not (top4 >= grid_len):
                continue
        elif mode == "inside_to_out_neg":
            if not (0 <= top3 and top3 + lengths[3] <= grid_len):
                continue
            if not (top4 + lengths[4] <= 0):
                continue
        else:
            raise ValueError(f"unknown axis mode: {mode}")

        candidates.append(anchor3)
    return candidates


def _clip_bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    if not coords:
        return None
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    width = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    source_size = int(rng.choice([1, 2]))
    motif_color = int(rng.choice(MOTIF_COLORS))
    mask = _transform_mask(MOTIF_LIBRARY[int(rng.integers(len(MOTIF_LIBRARY)))], rng)
    attachment = ATTACHMENTS[int(rng.integers(len(ATTACHMENTS)))]
    source_patch = _build_source_patch(mask, source_size, motif_color, attachment, rng)

    max_size = source_size + 4
    specs = _specs_for_sizes(source_patch, source_size, max_size)
    want_forward_partial = True
    want_reverse_visible = source_size > 1 and bool(rng.integers(2))

    for _ in range(LAYOUT_ATTEMPTS):
        dir_r, dir_c = STEP_DIRECTIONS[int(rng.integers(len(STEP_DIRECTIONS)))]
        boundary_axes = [axis for axis, sign in enumerate((dir_r, dir_c)) if sign != 0]
        boundary_axis = int(rng.choice(boundary_axes))
        row_mag = 0 if dir_r == 0 else int(rng.integers(5, 10) if boundary_axis == 0 else rng.integers(2, 6))
        col_mag = 0 if dir_c == 0 else int(rng.integers(5, 10) if boundary_axis == 1 else rng.integers(2, 6))
        step_r = dir_r * row_mag
        step_c = dir_c * col_mag

        row_mode = "inside"
        col_mode = "inside"
        if boundary_axis == 0:
            row_mode = (
                "partial_pos"
                if want_forward_partial and step_r > 0
                else "partial_neg"
                if want_forward_partial and step_r < 0
                else "inside_to_out_pos"
                if step_r > 0
                else "inside_to_out_neg"
            )
        else:
            col_mode = (
                "partial_pos"
                if want_forward_partial and step_c > 0
                else "partial_neg"
                if want_forward_partial and step_c < 0
                else "inside_to_out_pos"
                if step_c > 0
                else "inside_to_out_neg"
            )

        sizes = [source_size + i for i in range(5)]
        row_offsets = [int(specs[size]["red_offset"][0]) for size in sizes]
        col_offsets = [int(specs[size]["red_offset"][1]) for size in sizes]
        heights = [int(specs[size]["shape"][0]) for size in sizes]
        widths = [int(specs[size]["shape"][1]) for size in sizes]

        row_candidates = _axis_candidates(height, step_r, row_offsets, heights, row_mode)
        col_candidates = _axis_candidates(width, step_c, col_offsets, widths, col_mode)
        if not row_candidates or not col_candidates:
            continue

        pair_attempts = min(24, max(1, len(row_candidates) * len(col_candidates)))
        for _ in range(pair_attempts):
            anchor3 = (int(rng.choice(row_candidates)), int(rng.choice(col_candidates)))
            anchors = [
                (anchor3[0] - (3 - idx) * step_r, anchor3[1] - (3 - idx) * step_c)
                for idx in range(3)
            ]
            forward_anchor = (anchor3[0] + step_r, anchor3[1] + step_c)
            reverse_anchor = (anchors[0][0] - step_r, anchors[0][1] - step_c)

            input_boxes = []
            valid = True
            for idx, anchor_top_left in enumerate(anchors):
                coords = _visible_coords((height, width), specs[source_size + idx], anchor_top_left)
                if not coords:
                    valid = False
                    break
                box = _clip_bbox(coords)
                assert box is not None
                if any(_boxes_overlap(box, other) for other in input_boxes):
                    valid = False
                    break
                input_boxes.append(box)
            if not valid:
                continue

            forward_coords = _visible_coords((height, width), specs[source_size + 3], anchor3)
            if not forward_coords:
                continue
            forward_box = _clip_bbox(forward_coords)
            assert forward_box is not None
            if any(_boxes_overlap(forward_box, other) for other in input_boxes):
                continue
            min_forward = max(3, len(specs[source_size + 3]["coords"]) // 4)
            if want_forward_partial and len(forward_coords) < min_forward:
                continue

            forward_next_coords = _visible_coords((height, width), specs[source_size + 4], forward_anchor)
            if forward_next_coords:
                continue

            reverse_visible = False
            if source_size > 1:
                reverse_coords = _visible_coords((height, width), specs[source_size - 1], reverse_anchor)
                reverse_visible = bool(reverse_coords)
                if want_reverse_visible and not reverse_visible:
                    continue
                if reverse_visible:
                    reverse_box = _clip_bbox(reverse_coords)
                    assert reverse_box is not None
                    if any(_boxes_overlap(reverse_box, other) for other in input_boxes):
                        continue
                    if _boxes_overlap(reverse_box, forward_box):
                        continue

            grid = np.full((height, width), BG, dtype=int)
            for idx, anchor_top_left in enumerate(anchors):
                spec = specs[source_size + idx]
                patch_r0, patch_c0 = _patch_top_left(spec, anchor_top_left)
                paste(grid, spec["patch"], patch_r0, patch_c0, transparent=BG)

            try:
                output = _main_impl(grid)
            except (AssertionError, ValueError):
                continue
            if output.shape != grid.shape or np.array_equal(output, grid):
                continue
            if max(output.shape) > 30:
                continue

            parsed_anchors = _square_anchor_components(grid, BG)
            if len(parsed_anchors) != 3:
                continue
            return grid

    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle-6 input")
