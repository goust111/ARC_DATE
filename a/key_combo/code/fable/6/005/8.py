from __future__ import annotations

import importlib.util
from pathlib import Path
import random

import numpy as np

try:
    from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste, random_free_location
except ImportError:
    _ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    _ARC_UTILS_SPEC = importlib.util.spec_from_file_location("_arc_utils_local", _ARC_UTILS_PATH)
    assert _ARC_UTILS_SPEC is not None and _ARC_UTILS_SPEC.loader is not None
    _ARC_UTILS = importlib.util.module_from_spec(_ARC_UTILS_SPEC)
    _ARC_UTILS_SPEC.loader.exec_module(_ARC_UTILS)
    bbox_coords = _ARC_UTILS.bbox_coords
    components_mask = _ARC_UTILS.components_mask
    crop_bbox = _ARC_UTILS.crop_bbox
    get_bg = _ARC_UTILS.get_bg
    paste = _ARC_UTILS.paste
    random_free_location = _ARC_UTILS.random_free_location


MAX_ATTEMPTS = 400
ALL_COLORS = list(range(10))


def _frame_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _component_objects(grid: np.ndarray, bg: int) -> list[dict]:
    objects: list[dict] = []
    for coords in components_mask(grid != bg, connectivity=4):
        box = bbox_coords(coords)
        objects.append({"bbox": box, "crop": crop_bbox(grid, box)})
    return objects


def _is_target_frame(patch: np.ndarray, bg: int) -> bool:
    colors = sorted(set(map(int, patch.flat)) - {int(bg)})
    return (
        patch.shape[0] >= 3
        and patch.shape[1] >= 3
        and len(colors) == 1
        and np.array_equal(patch != bg, _frame_mask(*patch.shape))
    )


def _row_band_candidates(template: np.ndarray, bg: int) -> list[tuple[int, int, int, int]]:
    h, w = template.shape
    out: list[tuple[int, int, int, int]] = []
    r = 0
    while r < h:
        r1 = r + 1
        while r1 < h and np.array_equal(template[r1], template[r]):
            r1 += 1
        row = template[r]
        bg_cols = np.flatnonzero(row == bg)
        if bg_cols.size:
            c0 = int(bg_cols[0])
            c1 = int(bg_cols[-1]) + 1
            if (
                int(bg_cols.size) == c1 - c0
                and 0 < c0 < c1 < w
                and np.all(row[:c0] == row[0])
                and np.all(row[c1:] == row[c1])
                and int(row[0]) != bg
                and int(row[c1]) != bg
            ):
                out.append((r, r1, c0, c1))
        r = r1
    return out


def _col_band_candidates(template: np.ndarray, bg: int) -> list[tuple[int, int, int, int]]:
    candidates = _row_band_candidates(template.T, bg)
    return [(r0, r1, c0, c1) for r0, r1, c0, c1 in candidates]


def _analyze_template(template: np.ndarray, bg: int) -> dict:
    row_candidates = _row_band_candidates(template, bg)
    col_candidates = _col_band_candidates(template, bg)
    assert len(row_candidates) == 1, f"expected one horizontal growth band, got {row_candidates}"
    assert len(col_candidates) == 1, f"expected one vertical growth band, got {col_candidates}"

    row0, row1, col0, col1 = row_candidates[0]
    col_band0, col_band1, row_band0, row_band1 = col_candidates[0]
    assert row0 == row_band0 and row1 == row_band1, "row/column band heights disagree"
    assert col0 == col_band0 and col1 == col_band1, "row/column band widths disagree"
    assert np.all(template[row0:row1, col0:col1] == bg), "band intersection must stay background"

    return {
        "row0": row0,
        "row1": row1,
        "col0": col0,
        "col1": col1,
        "band_h": row1 - row0,
        "band_w": col1 - col0,
    }


def _stretch_template(template: np.ndarray, target_h: int, target_w: int, bg: int) -> np.ndarray:
    info = _analyze_template(template, bg)
    h, w = template.shape
    assert target_h >= h and target_w >= w

    extra_h = int(target_h - h)
    extra_w = int(target_w - w)
    assert extra_h % info["band_h"] == 0, "target height not reachable by row-band copying"
    assert extra_w % info["band_w"] == 0, "target width not reachable by col-band copying"

    row_copies = 1 + extra_h // info["band_h"]
    col_copies = 1 + extra_w // info["band_w"]

    row_band = template[info["row0"] : info["row1"], :]
    stretched = np.concatenate(
        [template[: info["row0"], :], *[row_band.copy() for _ in range(row_copies)], template[info["row1"] :, :]],
        axis=0,
    )
    col_band = stretched[:, info["col0"] : info["col1"]]
    stretched = np.concatenate(
        [stretched[:, : info["col0"]], *[col_band.copy() for _ in range(col_copies)], stretched[:, info["col1"] :]],
        axis=1,
    )
    assert stretched.shape == (target_h, target_w)
    return stretched


def _find_roles(grid: np.ndarray, bg: int) -> tuple[dict, dict]:
    objects = _component_objects(grid, bg)
    target_candidates = [obj for obj in objects if _is_target_frame(obj["crop"], bg)]
    assert len(target_candidates) == 1, f"expected one target frame, got {len(target_candidates)}"
    target = target_candidates[0]

    template_candidates: list[dict] = []
    for obj in objects:
        if obj is target:
            continue
        try:
            _analyze_template(obj["crop"], bg)
        except AssertionError:
            continue
        template_candidates.append(obj)
    assert len(template_candidates) == 1, f"expected one template, got {len(template_candidates)}"
    template = template_candidates[0]

    th, tw = template["crop"].shape
    fh, fw = target["crop"].shape
    assert fh > th and fw > tw, "target must be larger in both dimensions"
    return template, target


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stretch the small framed template to the larger target frame while keeping corner motifs fixed.

    Input structure:
    - A dominant background contains one multicolor hollow rectangular template and one larger monochrome hollow target frame.
    - In the template, the four corners are non-uniform corner blocks, while one horizontal middle band and one vertical middle band are uniform and repeatable.
    - The target frame only supplies the destination outer rectangle and is larger than the template in both dimensions.

    Rule:
    - Identify the unique repeatable row band and column band inside the template.
    - Copy those whole bands as many times as needed so the template's outer size exactly matches the target frame's bbox.
    - Draw the stretched template over the target frame location, replacing the frame color, while leaving the original template unchanged.

    Key insight:
    - This is a 9-slice style expansion: the corners are anchored, and every size difference is absorbed only by duplicating the template's uniform middle row/column bands.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    template, target = _find_roles(g, bg)
    stretched = _stretch_template(template["crop"], target["crop"].shape[0], target["crop"].shape[1], bg)

    out = g.copy()
    r0, c0, _r1, _c1 = target["bbox"]
    paste(out, stretched, r0, c0)
    return out


def _make_corner_block(
    kind: str,
    h: int,
    w: int,
    seam_a: int,
    seam_b: int,
    fill_a: int,
    fill_b: int,
) -> np.ndarray:
    block = np.full((h, w), fill_a, dtype=int)
    rows, cols = np.indices((h, w))
    block[(rows + cols) % 2 == 1] = fill_b

    if kind == "tl":
        block[:, -1] = seam_a
        block[-1, :] = seam_b
        block[0, 0] = fill_b
    elif kind == "tr":
        block[:, 0] = seam_a
        block[-1, :] = seam_b
        block[0, -1] = fill_b
    elif kind == "bl":
        block[0, :] = seam_a
        block[:, -1] = seam_b
        block[-1, 0] = fill_b
    else:
        block[0, :] = seam_a
        block[:, 0] = seam_b
        block[-1, -1] = fill_b
    return block


def _build_template(bg: int) -> tuple[np.ndarray, dict]:
    while True:
        top_h = random.randint(1, 3)
        bottom_h = random.randint(1, 3)
        left_w = random.randint(1, 3)
        right_w = random.randint(1, 3)
        if min(top_h + left_w, top_h + right_w, bottom_h + left_w, bottom_h + right_w) < 3:
            continue
        band_h = random.randint(1, 2)
        band_w = random.randint(1, 2)
        h = top_h + band_h + bottom_h
        w = left_w + band_w + right_w
        if max(h, w) <= 8:
            break

    usable = [color for color in ALL_COLORS if color != bg]
    random.shuffle(usable)
    target_color = usable.pop()
    top_color, bottom_color, left_color, right_color = usable[:4]
    accent_a, accent_b = usable[4:6]

    template = np.full((h, w), bg, dtype=int)
    template[:top_h, left_w : left_w + band_w] = top_color
    template[top_h + band_h :, left_w : left_w + band_w] = bottom_color
    template[top_h : top_h + band_h, :left_w] = left_color
    template[top_h : top_h + band_h, left_w + band_w :] = right_color

    paste(
        template,
        _make_corner_block("tl", top_h, left_w, top_color, left_color, accent_a, accent_b),
        0,
        0,
    )
    paste(
        template,
        _make_corner_block("tr", top_h, right_w, top_color, right_color, accent_b, accent_a),
        0,
        left_w + band_w,
    )
    paste(
        template,
        _make_corner_block("bl", bottom_h, left_w, bottom_color, left_color, accent_b, accent_a),
        top_h + band_h,
        0,
    )
    paste(
        template,
        _make_corner_block("br", bottom_h, right_w, bottom_color, right_color, accent_a, accent_b),
        top_h + band_h,
        left_w + band_w,
    )

    return template, {
        "bg": bg,
        "target_color": target_color,
        "top_h": top_h,
        "bottom_h": bottom_h,
        "left_w": left_w,
        "right_w": right_w,
        "band_h": band_h,
        "band_w": band_w,
    }


def _make_target_frame(h: int, w: int, color: int, bg: int) -> np.ndarray:
    patch = np.full((h, w), bg, dtype=int)
    patch[_frame_mask(h, w)] = color
    return patch


def _place_objects(template: np.ndarray, frame: np.ndarray, bg: int) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    min_h = max(template.shape[0], frame.shape[0]) + 4
    min_w = template.shape[1] + frame.shape[1] + 5
    if min_h > 30 or min_w > 30:
        raise AssertionError("canvas too small for required separation")

    grid_h = random.randint(min_h, 30)
    grid_w = random.randint(min_w, 30)
    occ = np.zeros((grid_h, grid_w), dtype=int)
    solid_template = np.ones(template.shape, dtype=int)
    solid_frame = np.ones(frame.shape, dtype=int)

    t_pos = random_free_location(occ, solid_template, transparent=0, border=1, padding=2)
    paste(occ, solid_template, t_pos[0], t_pos[1])
    f_pos = random_free_location(occ, solid_frame, transparent=0, border=1, padding=2)

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    paste(grid, template, t_pos[0], t_pos[1])
    paste(grid, frame, f_pos[0], f_pos[1])
    return grid, t_pos, f_pos


def _expected_output(grid: np.ndarray, frame_pos: tuple[int, int], stretched: np.ndarray) -> np.ndarray:
    out = grid.copy()
    paste(out, stretched, frame_pos[0], frame_pos[1])
    return out


def _validate_generated_case(
    inp: np.ndarray,
    expected: np.ndarray,
    template: np.ndarray,
    stretched: np.ndarray,
    template_pos: tuple[int, int],
    frame_pos: tuple[int, int],
    spec: dict,
) -> bool:
    bg = int(spec["bg"])
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if get_bg(inp) != bg:
        return False
    if np.array_equal(inp, expected):
        return False

    try:
        template_obj, target_obj = _find_roles(inp, bg)
    except AssertionError:
        return False

    info = _analyze_template(template_obj["crop"], bg)
    if info["row0"] != spec["top_h"] or info["band_h"] != spec["band_h"]:
        return False
    if info["col0"] != spec["left_w"] or info["band_w"] != spec["band_w"]:
        return False
    if not np.array_equal(template_obj["crop"], template):
        return False

    target_h, target_w = target_obj["crop"].shape
    if target_h <= template.shape[0] or target_w <= template.shape[1]:
        return False
    if target_h * template.shape[1] == target_w * template.shape[0]:
        return False
    if target_h - template.shape[0] < spec["band_h"] or target_w - template.shape[1] < spec["band_w"]:
        return False

    target_r0, target_c0 = frame_pos
    if target_obj["bbox"][:2] != (target_r0, target_c0):
        return False
    if template_obj["bbox"][:2] != template_pos:
        return False

    if not np.array_equal(_stretch_template(template, target_h, target_w, bg), stretched):
        return False

    corner_blocks = [
        template[: spec["top_h"], : spec["left_w"]],
        template[: spec["top_h"], -spec["right_w"] :],
        template[-spec["bottom_h"] :, : spec["left_w"]],
        template[-spec["bottom_h"] :, -spec["right_w"] :],
    ]
    if any(len(set(map(int, block.flat))) < 2 for block in corner_blocks):
        return False

    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False

    tr0, tc0 = template_pos
    if not np.array_equal(actual[tr0 : tr0 + template.shape[0], tc0 : tc0 + template.shape[1]], template):
        return False
    if not np.array_equal(actual[target_r0 : target_r0 + target_h, target_c0 : target_c0 + target_w], stretched):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    bg = random.randint(0, 9)
    template, spec = _build_template(bg)

    extra_h = spec["band_h"] * random.randint(1, 3)
    extra_w = spec["band_w"] * random.randint(1, 3)
    target_h = template.shape[0] + extra_h
    target_w = template.shape[1] + extra_w
    if target_h * template.shape[1] == target_w * template.shape[0]:
        return None

    frame = _make_target_frame(target_h, target_w, spec["target_color"], bg)
    stretched = _stretch_template(template, target_h, target_w, bg)

    try:
        inp, template_pos, frame_pos = _place_objects(template, frame, bg)
    except (AssertionError, ValueError):
        return None

    expected = _expected_output(inp, frame_pos, stretched)
    if not _validate_generated_case(inp, expected, template, stretched, template_pos, frame_pos, spec):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid recognitive-stretch ARC input")
