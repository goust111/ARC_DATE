from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import crop_bbox, find_components, paste, scale_sprite


BG = 0
MARKER = 1
TEMPLATE_COLOR = 2
FRAME_COLOR = 5

MAX_ATTEMPTS = 400
GRID_RANGE = (20, 30)
GAP = 2

SCALE_OPTIONS = {
    2: {
        2: [(2, 3), (2, 4), (3, 4), (2, 5)],
        3: [(2, 3, 4), (2, 3, 5), (2, 4, 5)],
    },
    3: {
        2: [(2, 3), (2, 4), (3, 4)],
        3: [(2, 2, 3), (2, 3, 3)],
    },
}

TEMPLATE_MASKS = {
    2: [
        np.array([[1, 1], [1, 0]], dtype=int),
        np.array([[1, 1], [0, 1]], dtype=int),
        np.array([[1, 0], [1, 1]], dtype=int),
        np.array([[0, 1], [1, 1]], dtype=int),
    ],
    3: [
        np.array([[1, 1, 0], [0, 1, 0], [1, 1, 1]], dtype=int),
        np.array([[1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=int),
        np.array([[1, 0, 1], [1, 1, 1], [0, 1, 0]], dtype=int),
        np.array([[1, 1, 1], [0, 0, 1], [0, 1, 1]], dtype=int),
    ],
}


def _template_center_value(template_patch: np.ndarray) -> int:
    return int(template_patch[template_patch.shape[0] // 2, template_patch.shape[1] // 2])


def _extract_template(grid: np.ndarray) -> np.ndarray:
    colors = [int(c) for c in np.unique(grid) if int(c) not in {BG, MARKER, FRAME_COLOR}]
    comps = find_components(
        grid,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=colors,
        return_format="dict",
    )
    assert comps, "expected a non-frame template component"
    comp = min(
        comps,
        key=lambda d: (int(d["area"]), int(d["bbox"][0]), int(d["bbox"][1]), int(d["color"])),
    )
    r0, c0, r1, c1 = map(int, comp["bbox"])
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    for r, c in comp["coords"]:
        patch[int(r) - r0, int(c) - c0] = int(comp["color"])
    assert patch.shape[0] == patch.shape[1], "template bbox must be square"
    return patch


def _detect_frames(grid: np.ndarray) -> list[dict]:
    frames = []
    comps = find_components(
        grid,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=[FRAME_COLOR],
        return_format="dict",
    )
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        box = crop_bbox(grid, (r0, c0, r1, c1))
        h, w = map(int, box.shape)
        if h != w or h < 4:
            continue
        border = np.zeros_like(box, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if not np.all(box[border] == FRAME_COLOR):
            continue
        if np.any(box[1:-1, 1:-1] == FRAME_COLOR):
            continue
        if int(comp["area"]) != int(border.sum()):
            continue
        blues = np.argwhere(box[1:-1, 1:-1] == MARKER)
        assert len(blues) == 1, "each frame must contain exactly one marker"
        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "inner_size": h - 2,
                "marker_local": (int(blues[0, 0]), int(blues[0, 1])),
            }
        )
    frames.sort(key=lambda d: (d["bbox"][0], d["bbox"][1]))
    assert frames, "expected at least one valid frame"
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Scale the small template to each frame's inner size and stamp it around the frame marker.

    Input structure:
    - The background is black, with one small non-black/non-blue/non-grey template component near the upper-left.
    - Elsewhere there are 2-3 grey one-cell-thick hollow square frames, each with exactly one blue marker pixel inside.
    - Every frame interior is a square whose side length is an integer multiple of the template bbox side.

    Rule:
    - Identify the template component and crop its square bbox, preserving any background holes inside that bbox.
    - For each grey frame, compute the integer scale factor from the frame's inner side length divided by the template side length.
    - Upscale the whole template bbox by that factor, then paste its non-background pixels with the blue marker as the template center anchor.
    - Keep the original template and frame borders unchanged; the blue marker survives only when the scaled template center is background.

    Key insight:
    - The template is not stamped at a fixed size: each frame determines its own integer enlargement factor, so the same source shape expands differently in different destinations.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    template = _extract_template(g)
    k = int(template.shape[0])
    out = g.copy()

    for frame in _detect_frames(g):
        inner = int(frame["inner_size"])
        assert inner % k == 0, "frame inner size must be a multiple of template size"
        scale = inner // k
        scaled = scale_sprite(template, scale)

        r0, c0, _r1, _c1 = frame["bbox"]
        marker_r, marker_c = frame["marker_local"]
        top = int(r0 + 1 + marker_r - scaled.shape[0] // 2)
        left = int(c0 + 1 + marker_c - scaled.shape[1] // 2)

        inner_top = r0 + 1
        inner_left = c0 + 1
        assert inner_top <= top <= inner_top + inner - scaled.shape[0]
        assert inner_left <= left <= inner_left + inner - scaled.shape[1]

        paste(out, scaled, top, left, transparent=BG, overwrite=True, clip=False)

    return out


def _dihedral(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, random.randint(0, 3))
    if random.random() < 0.5:
        out = np.fliplr(out)
    return out.copy()


def _sample_template(k: int, center_is_filled: bool) -> np.ndarray:
    masks = TEMPLATE_MASKS[k]
    if k == 2:
        candidates = [m for m in masks if bool(m[k // 2, k // 2]) == center_is_filled]
        mask = random.choice(candidates).copy()
    else:
        candidates = []
        for mask in masks:
            for _ in range(8):
                transformed = _dihedral(mask)
                if bool(transformed[k // 2, k // 2]) == center_is_filled:
                    candidates.append(transformed)
        mask = random.choice(candidates).copy()
    return np.where(mask != 0, TEMPLATE_COLOR, BG).astype(int)


def _frame_patch(inner_size: int, marker_local: tuple[int, int]) -> np.ndarray:
    patch = np.full((inner_size + 2, inner_size + 2), FRAME_COLOR, dtype=int)
    patch[1:-1, 1:-1] = BG
    rr, cc = marker_local
    patch[1 + rr, 1 + cc] = MARKER
    return patch


def _valid_marker_positions(scaled_shape: tuple[int, int], inner_size: int) -> list[tuple[int, int]]:
    h, w = map(int, scaled_shape)
    out = []
    for r in range(inner_size):
        top = r - h // 2
        if not (0 <= top and top + h <= inner_size):
            continue
        for c in range(inner_size):
            left = c - w // 2
            if 0 <= left and left + w <= inner_size:
                out.append((r, c))
    return out


def _boxes_far_enough(box: tuple[int, int, int, int], others: list[tuple[int, int, int, int]], gap: int) -> bool:
    r0, c0, r1, c1 = box
    for or0, oc0, or1, oc1 in others:
        if not (r1 + gap < or0 or or1 + gap < r0 or c1 + gap < oc0 or oc1 + gap < c0):
            return False
    return True


def _place_box(h: int, w: int, H: int, W: int, others: list[tuple[int, int, int, int]], *, top_left_bias: bool = False) -> tuple[int, int] | None:
    if h > H or w > W:
        return None
    candidates = []
    row_range = range(1, min(5, H - h + 1)) if top_left_bias else range(0, H - h + 1)
    col_range = range(1, min(5, W - w + 1)) if top_left_bias else range(0, W - w + 1)
    for r0 in row_range:
        for c0 in col_range:
            box = (r0, c0, r0 + h - 1, c0 + w - 1)
            if _boxes_far_enough(box, others, GAP):
                candidates.append((r0, c0))
    if not candidates and top_left_bias:
        return _place_box(h, w, H, W, others, top_left_bias=False)
    return random.choice(candidates) if candidates else None


def _build_expected(grid: np.ndarray, template: np.ndarray, placed_frames: list[dict]) -> np.ndarray:
    out = grid.copy()
    for frame in placed_frames:
        scaled = scale_sprite(template, frame["scale"])
        r0, c0 = frame["top_left"]
        mr, mc = frame["marker_local"]
        top = int(r0 + 1 + mr - scaled.shape[0] // 2)
        left = int(c0 + 1 + mc - scaled.shape[1] // 2)
        paste(out, scaled, top, left, transparent=BG, overwrite=True, clip=False)
    return out


def _semantic_checks(inp: np.ndarray, expected: np.ndarray, template: np.ndarray, placed_frames: list[dict]) -> bool:
    if inp.ndim != 2 or expected.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if len(placed_frames) not in (2, 3):
        return False
    if len({frame["scale"] for frame in placed_frames}) < 2:
        return False
    if np.array_equal(inp, expected):
        return False
    if not np.array_equal(main(inp), expected):
        return False

    parsed_template = _extract_template(inp)
    if not np.array_equal(parsed_template, template):
        return False

    parsed_frames = _detect_frames(inp)
    if len(parsed_frames) != len(placed_frames):
        return False
    inner_sizes = sorted(frame["inner_size"] for frame in parsed_frames)
    if inner_sizes != sorted(frame["scale"] * template.shape[0] for frame in placed_frames):
        return False

    expected_blue = len(placed_frames) if _template_center_value(template) == BG else 0
    if int(np.count_nonzero(expected == MARKER)) != expected_blue:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    k = random.choice([2, 3])
    frame_count = random.choice([2, 3])
    center_is_filled = random.random() < 0.5
    template = _sample_template(k, center_is_filled=center_is_filled)
    scales = list(random.choice(SCALE_OPTIONS[k][frame_count]))
    random.shuffle(scales)

    min_side = max(22, 6 + sum(k * s + 2 for s in scales) // 2)
    H = random.randint(min_side, GRID_RANGE[1])
    W = random.randint(min_side, GRID_RANGE[1])
    grid = np.full((H, W), BG, dtype=int)

    placed_boxes: list[tuple[int, int, int, int]] = []
    template_pos = _place_box(k, k, H, W, placed_boxes, top_left_bias=True)
    if template_pos is None:
        return None
    tr, tc = template_pos
    paste(grid, template, tr, tc, transparent=BG, overwrite=True, clip=False)
    placed_boxes.append((tr, tc, tr + k - 1, tc + k - 1))

    placed_frames: list[dict] = []
    for scale in sorted(scales, reverse=True):
        inner = k * scale
        scaled = scale_sprite(template, scale)
        marker_positions = _valid_marker_positions(scaled.shape, inner)
        if not marker_positions:
            return None
        marker_local = random.choice(marker_positions)
        patch = _frame_patch(inner, marker_local)
        h, w = patch.shape
        pos = _place_box(h, w, H, W, placed_boxes)
        if pos is None:
            return None
        r0, c0 = pos
        paste(grid, patch, r0, c0, transparent=BG, overwrite=True, clip=False)
        placed_boxes.append((r0, c0, r0 + h - 1, c0 + w - 1))
        placed_frames.append(
            {
                "top_left": (r0, c0),
                "scale": scale,
                "marker_local": marker_local,
            }
        )

    expected = _build_expected(grid, template, placed_frames)
    if not _semantic_checks(grid, expected, template, placed_frames):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid sample")
