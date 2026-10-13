from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_SPEC = importlib.util.spec_from_file_location("_shared_arc_utils", _ARC_UTILS_PATH)
if _ARC_SPEC is None or _ARC_SPEC.loader is None:
    raise ImportError(f"cannot load ARC helpers from {_ARC_UTILS_PATH}")
_ARC_UTILS = importlib.util.module_from_spec(_ARC_SPEC)
_ARC_SPEC.loader.exec_module(_ARC_UTILS)

Color = _ARC_UTILS.Color
bbox = _ARC_UTILS.bbox
bbox_h = _ARC_UTILS.bbox_h
bbox_w = _ARC_UTILS.bbox_w
colors_present = _ARC_UTILS.colors_present
components_mask = _ARC_UTILS.components_mask
get_bg = _ARC_UTILS.get_bg
is_contiguous = _ARC_UTILS.is_contiguous
paste = _ARC_UTILS.paste
crop_bbox = _ARC_UTILS.crop_bbox

BG = Color.BLACK
OCCLUDER = Color.GREY
PALETTE = [
    Color.BLUE,
    Color.RED,
    Color.GREEN,
    Color.YELLOW,
    Color.PINK,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
]
MAX_GENERATION_ATTEMPTS = 400


def _ri(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(int(lo), int(hi) + 1))


def _other_orientation(orientation: str) -> str:
    return "horizontal" if orientation == "vertical" else "vertical"


def _erase_occluder(grid: np.ndarray, bg: int) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    out[out == OCCLUDER] = int(bg)
    return out


def _find_occluder_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    grey_mask = np.asarray(grid, dtype=int) == OCCLUDER
    comps = components_mask(grey_mask, connectivity=4)
    assert len(comps) == 1, "expected exactly one grey component"
    box = bbox(grey_mask)
    assert box is not None, "expected a grey occluder"
    r0, c0, r1, c1 = box
    assert grey_mask[r0 : r1 + 1, c0 : c1 + 1].all(), "grey component must be a filled rectangle"
    assert int(grey_mask.sum()) == bbox_h(box) * bbox_w(box), "grey component must be rectangular"
    return int(r0), int(c0), int(r1), int(c1)


def _axis_center_key(grid: np.ndarray, orientation: str, bg: int) -> int:
    fg_box = bbox(np.asarray(grid, dtype=int) != int(bg))
    assert fg_box is not None, "expected foreground pixels after removing the occluder"
    if orientation == "vertical":
        return int(fg_box[1] + fg_box[3])
    return int(fg_box[0] + fg_box[2])


def _axis_score(grid: np.ndarray, orientation: str, axis: int, bg: int) -> int:
    g = np.asarray(grid, dtype=int)
    coords = np.argwhere(g != int(bg))
    score = 0
    if orientation == "vertical":
        width = g.shape[1]
        for r, c in coords:
            mc = int(axis) - int(c)
            if 0 <= mc < width and int(g[r, mc]) == int(g[r, c]):
                score += 1
    else:
        height = g.shape[0]
        for r, c in coords:
            mr = int(axis) - int(r)
            if 0 <= mr < height and int(g[mr, c]) == int(g[r, c]):
                score += 1
    return int(score)


def _axis_scores(grid: np.ndarray, orientation: str, bg: int) -> list[tuple[int, int]]:
    limit = 2 * (grid.shape[1] if orientation == "vertical" else grid.shape[0]) - 1
    return [(axis, _axis_score(grid, orientation, axis, bg)) for axis in range(limit)]


def _best_axis(grid: np.ndarray, orientation: str, bg: int) -> tuple[int, list[tuple[int, int]]]:
    center_key = _axis_center_key(grid, orientation, bg)
    scores = _axis_scores(grid, orientation, bg)
    best_axis, _best_score = max(scores, key=lambda item: (item[1], -abs(item[0] - center_key), -item[0]))
    return int(best_axis), scores


def _complete_symmetry(grid: np.ndarray, orientation: str, axis: int, bg: int) -> np.ndarray:
    src = np.asarray(grid, dtype=int)
    out = src.copy()
    coords = np.argwhere(src != int(bg))
    if orientation == "vertical":
        width = out.shape[1]
        for r, c in coords:
            mc = int(axis) - int(c)
            if 0 <= mc < width and int(out[r, mc]) == int(bg):
                out[r, mc] = int(src[r, c])
    else:
        height = out.shape[0]
        for r, c in coords:
            mr = int(axis) - int(r)
            if 0 <= mr < height and int(out[mr, c]) == int(bg):
                out[mr, c] = int(src[r, c])
    return out


def _foreground_crop(grid: np.ndarray, bg: int) -> np.ndarray:
    fg_box = bbox(np.asarray(grid, dtype=int) != int(bg))
    assert fg_box is not None, "expected a foreground object"
    return crop_bbox(np.asarray(grid, dtype=int), fg_box)


def _is_exactly_symmetric(grid: np.ndarray, orientation: str, axis: int) -> bool:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    if orientation == "vertical":
        for r in range(height):
            for c in range(width):
                mc = int(axis) - c
                if not (0 <= mc < width):
                    return False
                if int(g[r, c]) != int(g[r, mc]):
                    return False
    else:
        for r in range(height):
            for c in range(width):
                mr = int(axis) - r
                if not (0 <= mr < height):
                    return False
                if int(g[r, c]) != int(g[mr, c]):
                    return False
    return True


def _has_perfect_symmetry(grid: np.ndarray, orientation: str) -> bool:
    limit = 2 * (grid.shape[1] if orientation == "vertical" else grid.shape[0]) - 1
    return any(_is_exactly_symmetric(grid, orientation, axis) for axis in range(limit))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase the grey occluder and mirror-complete the hidden pattern along the axis encoded by the occluder's aspect ratio.

    Input structure:
    - A black background contains one central multicolor foreground object and one grey filled rectangular occluder.
    - The foreground object is symmetric either left-to-right or top-to-bottom, but the grey rectangle covers part of it and breaks that symmetry.
    - The grey component is the unique grey 4-connected object, and its width and height are different.

    Rule:
    - Detect the grey rectangle, compare its width and height, and erase it to black.
    - If the rectangle is wider than tall, search all vertical axes; if it is taller than wide, search all horizontal axes.
    - Score each candidate axis by counting same-color foreground pixels whose reflected partners already exist, choose the highest-scoring axis, and mirror every remaining foreground pixel across it.
    - Only fill currently black cells; preserve any foreground pixels that are already present.

    Key insight:
    - The occluder does not reveal the missing colors directly; its aspect ratio is a control signal that selects which symmetry direction should be used for reconstruction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    r0, c0, r1, c1 = _find_occluder_bbox(g)
    occ_h = r1 - r0 + 1
    occ_w = c1 - c0 + 1
    assert occ_h != occ_w, "the grey occluder must not be square"

    orientation = "vertical" if occ_w > occ_h else "horizontal"
    erased = _erase_occluder(g, bg)
    axis, _scores = _best_axis(erased, orientation, bg)
    return _complete_symmetry(erased, orientation, axis, bg)


def _make_half_pattern(height: int, width: int, colors: list[int], rng: np.random.Generator) -> np.ndarray:
    half = np.full((height, width), int(colors[0]), dtype=int)
    for color in colors[1:]:
        for _ in range(_ri(rng, 1, 2)):
            rect_h = _ri(rng, 1, min(height - 1, max(2, height // 2 + 1)))
            rect_w = _ri(rng, 1, min(width, max(1, width // 2 + 1)))
            r0 = _ri(rng, 0, height - rect_h)
            c0 = _ri(rng, 0, width - rect_w)
            half[r0 : r0 + rect_h, c0 : c0 + rect_w] = int(color)

    for _ in range(_ri(rng, 1, 2)):
        color = int(colors[_ri(rng, 0, len(colors) - 1)])
        if rng.random() < 0.5:
            row = _ri(rng, 0, height - 1)
            start = _ri(rng, 0, width - 1)
            half[row, start:] = color
        else:
            col = _ri(rng, 0, width - 1)
            start = _ri(rng, 0, height - 1)
            half[start:, col] = color
    return half


def _mirror_pattern(half: np.ndarray, orientation: str, full_height: int, full_width: int) -> np.ndarray:
    if orientation == "vertical":
        out = np.empty((full_height, full_width), dtype=int)
        out[:, : half.shape[1]] = half
        if full_width % 2 == 0:
            out[:, half.shape[1] :] = np.fliplr(half)
        else:
            out[:, half.shape[1] :] = np.fliplr(half[:, :-1])
        return out

    out = np.empty((full_height, full_width), dtype=int)
    out[: half.shape[0], :] = half
    if full_height % 2 == 0:
        out[half.shape[0] :, :] = np.flipud(half)
    else:
        out[half.shape[0] :, :] = np.flipud(half[:-1, :])
    return out


def _build_symmetric_patch(
    height: int, width: int, orientation: str, colors: list[int], rng: np.random.Generator
) -> np.ndarray | None:
    for _ in range(80):
        if orientation == "vertical":
            half = _make_half_pattern(height, (width + 1) // 2, colors, rng)
        else:
            half = _make_half_pattern((height + 1) // 2, width, colors, rng)
        patch = _mirror_pattern(half, orientation, height, width)
        if len(colors_present(patch, exclude={BG})) != len(colors):
            continue
        if _has_perfect_symmetry(patch, _other_orientation(orientation)):
            continue
        return patch
    return None


def _candidate_occluders(height: int, width: int, orientation: str) -> list[tuple[int, int, int, int]]:
    placements: list[tuple[int, int, int, int]] = []
    if orientation == "vertical":
        side_width = width // 2
        right_start = side_width + (width % 2)
        for occ_h in range(2, min(4, height) + 1):
            for occ_w in range(2, min(4, side_width) + 1):
                if occ_w <= occ_h:
                    continue
                for r0 in range(0, height - occ_h + 1):
                    for c0 in range(0, side_width - occ_w + 1):
                        placements.append((r0, c0, occ_h, occ_w))
                    for c0 in range(right_start, width - occ_w + 1):
                        placements.append((r0, c0, occ_h, occ_w))
        return placements

    side_height = height // 2
    bottom_start = side_height + (height % 2)
    for occ_h in range(2, min(4, side_height) + 1):
        for occ_w in range(2, min(4, width) + 1):
            if occ_h <= occ_w:
                continue
            for r0 in range(0, side_height - occ_h + 1):
                for c0 in range(0, width - occ_w + 1):
                    placements.append((r0, c0, occ_h, occ_w))
            for r0 in range(bottom_start, height - occ_h + 1):
                for c0 in range(0, width - occ_w + 1):
                    placements.append((r0, c0, occ_h, occ_w))
    return placements


def _centered_start(full_size: int, obj_size: int, rng: np.random.Generator) -> int:
    base = (full_size - obj_size) // 2
    choices = sorted({x for x in (base - 1, base, base + 1) if 0 <= x <= full_size - obj_size})
    return int(choices[_ri(rng, 0, len(choices) - 1)])


def _true_axis(top: int, left: int, height: int, width: int, orientation: str) -> int:
    if orientation == "vertical":
        return int(2 * left + width - 1)
    return int(2 * top + height - 1)


def _case_is_valid(inp: np.ndarray, expected: np.ndarray, orientation: str, axis: int) -> bool:
    if inp.ndim != 2 or expected.ndim != 2 or max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if not np.array_equal(main(inp), expected):
        return False
    if not is_contiguous(expected, background=BG, connectivity=4):
        return False

    fg_colors = colors_present(expected, exclude={BG})
    if not (2 <= len(fg_colors) <= 3):
        return False
    if not np.any(inp == OCCLUDER):
        return False

    erased = _erase_occluder(inp, BG)
    if np.array_equal(erased, expected):
        return False

    box = _find_occluder_bbox(inp)
    occ_h = bbox_h(box)
    occ_w = bbox_w(box)
    if occ_h == occ_w:
        return False
    if orientation == "vertical" and occ_w <= occ_h:
        return False
    if orientation == "horizontal" and occ_h <= occ_w:
        return False

    best_axis, scores = _best_axis(erased, orientation, BG)
    if best_axis != axis:
        return False
    best_score = max(score for _candidate, score in scores)
    if sum(score == best_score for _candidate, score in scores) != 1:
        return False

    crop = _foreground_crop(expected, BG)
    local_axis = crop.shape[1] - 1 if orientation == "vertical" else crop.shape[0] - 1
    if not _is_exactly_symmetric(crop, orientation, local_axis):
        return False
    if _has_perfect_symmetry(crop, _other_orientation(orientation)):
        return False

    wrong_orientation = _other_orientation(orientation)
    wrong_axis, _wrong_scores = _best_axis(erased, wrong_orientation, BG)
    wrong_completed = _complete_symmetry(erased, wrong_orientation, wrong_axis, BG)
    if np.array_equal(wrong_completed, expected):
        return False
    return True


def _generate_case(rng: np.random.Generator) -> np.ndarray | None:
    height = _ri(rng, 13, 17)
    width = _ri(rng, 13, 17)
    orientation = "vertical" if rng.random() < 0.5 else "horizontal"

    obj_h = _ri(rng, 7, min(9, height - 4))
    obj_w = _ri(rng, 7, min(9, width - 4))
    color_count = 2 + int(rng.random() < 0.45)
    colors = [int(x) for x in rng.choice(PALETTE, size=color_count, replace=False)]
    patch = _build_symmetric_patch(obj_h, obj_w, orientation, colors, rng)
    if patch is None:
        return None

    placements = _candidate_occluders(obj_h, obj_w, orientation)
    if not placements:
        return None
    rich = [p for p in placements if len(np.unique(patch[p[0] : p[0] + p[2], p[1] : p[1] + p[3]])) >= 2]
    if rich:
        placements = rich
    occ_r, occ_c, occ_h, occ_w = placements[_ri(rng, 0, len(placements) - 1)]

    top = _centered_start(height, obj_h, rng)
    left = _centered_start(width, obj_w, rng)
    expected = np.full((height, width), BG, dtype=int)
    paste(expected, patch, top, left)

    inp = expected.copy()
    inp[top + occ_r : top + occ_r + occ_h, left + occ_c : left + occ_c + occ_w] = OCCLUDER

    true_axis = _true_axis(top, left, obj_h, obj_w, orientation)
    return inp if _case_is_valid(inp, expected, orientation, true_axis) else None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        case = _generate_case(rng)
        if case is not None:
            return case
    raise RuntimeError("failed to generate a valid input for symmetry-completion occluder puzzle")
