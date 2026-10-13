import math
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, components_mask, paste, scale_sprite


TEMPLATE_SIZE = 3
FRAME_COLOR = Color.GREY
MARKER_COLOR = Color.RED
BACKGROUND = Color.BLACK
GEN_ATTEMPTS = 400
FRAME_COUNT_OPTIONS = (2, 3)
SCALE_OPTIONS = (1, 2, 3)
TEMPLATE_FILL_COLORS = (
    Color.BLUE,
    Color.GREEN,
    Color.YELLOW,
    Color.PINK,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
)


def _extract_template(grid: np.ndarray) -> np.ndarray:
    template = np.asarray(grid[:TEMPLATE_SIZE, :TEMPLATE_SIZE], dtype=int).copy()
    assert template.shape == (TEMPLATE_SIZE, TEMPLATE_SIZE)
    assert np.count_nonzero(template == MARKER_COLOR) == 1
    return template


def _frame_infos(grid: np.ndarray) -> list[dict]:
    infos = []
    for coords in components_mask(grid == FRAME_COLOR, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(coords)
        window = grid[r0 : r1 + 1, c0 : c1 + 1]
        if window.shape[0] < 3 or window.shape[1] < 3:
            continue
        if not (
            np.all(window[0, :] == FRAME_COLOR)
            and np.all(window[-1, :] == FRAME_COLOR)
            and np.all(window[:, 0] == FRAME_COLOR)
            and np.all(window[:, -1] == FRAME_COLOR)
        ):
            continue
        interior = window[1:-1, 1:-1]
        if np.any(interior == FRAME_COLOR):
            continue
        marker_rel = np.argwhere(interior == MARKER_COLOR)
        if marker_rel.shape != (1, 2):
            continue
        marker_r, marker_c = map(int, marker_rel[0])
        infos.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "inner_bbox": (int(r0 + 1), int(c0 + 1), int(r1 - 1), int(c1 - 1)),
                "marker": (int(r0 + 1 + marker_r), int(c0 + 1 + marker_c)),
                "inner_shape": (int(interior.shape[0]), int(interior.shape[1])),
            }
        )
    infos.sort(key=lambda info: info["bbox"])
    return infos


def _draw_scaled_template(
    out: np.ndarray,
    template: np.ndarray,
    frame: dict,
    *,
    use_marker: bool,
    scale_override: int | None = None,
) -> None:
    marker_pos = np.argwhere(template == MARKER_COLOR)
    assert marker_pos.shape == (1, 2)
    tr, tc = map(int, marker_pos[0])

    inner_r0, inner_c0, inner_r1, inner_c1 = frame["inner_bbox"]
    inner_h, inner_w = frame["inner_shape"]
    area = int(inner_h * inner_w)
    inner_side = math.isqrt(area)
    assert inner_side * inner_side == area
    assert inner_h == inner_w == inner_side
    assert inner_side % TEMPLATE_SIZE == 0
    scale = int(scale_override) if scale_override is not None else inner_side // TEMPLATE_SIZE
    assert scale >= 1

    scaled = scale_sprite(template, scale)
    if use_marker:
        mr, mc = frame["marker"]
        top = int(mr - tr * scale)
        left = int(mc - tc * scale)
    else:
        top = int(inner_r0)
        left = int(inner_c0)

    for rr in range(scaled.shape[0]):
        gr = top + rr
        if gr < inner_r0 or gr > inner_r1:
            continue
        for cc in range(scaled.shape[1]):
            gc = left + cc
            if gc < inner_c0 or gc > inner_c1:
                continue
            out[gr, gc] = int(scaled[rr, cc])


def _solve(
    grid: np.ndarray,
    *,
    use_marker: bool = True,
    scale_override: int | None = None,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    template = _extract_template(g)
    frames = _frame_infos(g)
    assert 2 <= len(frames) <= 3
    for frame in frames:
        _draw_scaled_template(out, template, frame, use_marker=use_marker, scale_override=scale_override)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Scale the top-left template by a factor encoded in each frame's interior area and place it by marker alignment.

    Input structure:
    - Black background with a 3x3 multicolor template sprite in the top-left corner.
    - The template uses 2-3 visible colors total and contains exactly one red marker pixel.
    - Elsewhere there are 2-3 grey 1-pixel hollow square frames; each frame interior contains exactly one red marker pixel.

    Rule:
    - Read the 3x3 template and locate its red marker cell.
    - For each grey frame, compute the interior area S, take sqrt(S), and divide by 3 to get the integer scale factor.
    - Enlarge the template by that factor with nearest-neighbor pixel replication.
    - Align the enlarged template so the top-left cell of its red marker block lands on the red marker pixel inside the frame.
    - Draw only the enlarged template cells that fall inside the frame interior; keep the grey border unchanged.

    Key insight:
    - The frame interior area does not choose a color directly; its square root encodes the template scale, while the red marker fixes the placement.
    """
    return _solve(grid, use_marker=True)


def _make_template(rng: np.random.Generator) -> np.ndarray:
    template = np.empty((TEMPLATE_SIZE, TEMPLATE_SIZE), dtype=int)
    extra_count = int(rng.choice([1, 2]))
    fill_colors = list(rng.choice(TEMPLATE_FILL_COLORS, size=extra_count, replace=False))
    marker_idx = int(rng.integers(TEMPLATE_SIZE * TEMPLATE_SIZE))
    marker_r, marker_c = divmod(marker_idx, TEMPLATE_SIZE)
    coords = [(r, c) for r in range(TEMPLATE_SIZE) for c in range(TEMPLATE_SIZE) if (r, c) != (marker_r, marker_c)]
    rng.shuffle(coords)

    if extra_count == 1:
        template.fill(fill_colors[0])
    else:
        template.fill(fill_colors[0])
        template[coords[0]] = fill_colors[1]
        template[coords[1]] = fill_colors[1]
        for r, c in coords[2:]:
            template[r, c] = int(rng.choice(fill_colors))

    template[marker_r, marker_c] = MARKER_COLOR

    if extra_count == 1 and rng.random() < 0.6:
        alt_r, alt_c = coords[0]
        template[alt_r, alt_c] = fill_colors[0]

    assert np.count_nonzero(template == MARKER_COLOR) == 1
    assert len(set(map(int, template.ravel()))) in {2, 3}
    return template


def _shift_candidates(template_marker: tuple[int, int], scale: int) -> list[tuple[int, int]]:
    tr, tc = template_marker
    side = TEMPLATE_SIZE * scale
    base_r = tr * scale
    base_c = tc * scale
    candidates = []
    for dr in range(-base_r, side - base_r):
        for dc in range(-base_c, side - base_c):
            visible_h = side - abs(dr)
            visible_w = side - abs(dc)
            if visible_h < max(2, scale) or visible_w < max(2, scale):
                continue
            candidates.append((int(dr), int(dc)))
    return candidates


def _build_frame_patch(
    template_marker: tuple[int, int],
    scale: int,
    rng: np.random.Generator,
    force_shift: bool,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    side = TEMPLATE_SIZE * scale
    patch = np.zeros((side + 2, side + 2), dtype=int)
    patch[0, :] = FRAME_COLOR
    patch[-1, :] = FRAME_COLOR
    patch[:, 0] = FRAME_COLOR
    patch[:, -1] = FRAME_COLOR

    candidates = _shift_candidates(template_marker, scale)
    if force_shift:
        candidates = [delta for delta in candidates if delta != (0, 0)]
    else:
        candidates = [delta for delta in candidates if delta == (0, 0)]
    if not candidates:
        raise ValueError("no valid marker shift candidates")

    dr, dc = candidates[int(rng.integers(len(candidates)))]
    tr, tc = template_marker
    marker_r = tr * scale + dr
    marker_c = tc * scale + dc
    patch[1 + marker_r, 1 + marker_c] = MARKER_COLOR
    return patch, (int(marker_r), int(marker_c)), (int(dr), int(dc))


def _too_close(rect: tuple[int, int, int, int], other: tuple[int, int, int, int], gap: int = 1) -> bool:
    return not (
        rect[2] + gap < other[0]
        or other[2] + gap < rect[0]
        or rect[3] + gap < other[1]
        or other[3] + gap < rect[1]
    )


def _place_rectangles(
    rng: np.random.Generator,
    canvas_shape: tuple[int, int],
    patches: list[np.ndarray],
) -> list[tuple[int, int]]:
    H, W = canvas_shape
    used = [(0, 0, TEMPLATE_SIZE - 1, TEMPLATE_SIZE - 1)]
    placements: list[tuple[int, int] | None] = [None] * len(patches)
    order = sorted(range(len(patches)), key=lambda idx: patches[idx].shape[0] * patches[idx].shape[1], reverse=True)

    for idx in order:
        h, w = patches[idx].shape
        candidates = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
        rng.shuffle(candidates)
        placed = None
        for r, c in candidates:
            rect = (r, c, r + h - 1, c + w - 1)
            if all(not _too_close(rect, other, gap=1) for other in used):
                placed = (int(r), int(c))
                used.append(rect)
                break
        if placed is None:
            raise ValueError("failed to place all rectangles")
        placements[idx] = placed

    return [placement for placement in placements if placement is not None]


def _constant_scale_for_negative_test(scales: list[int]) -> int:
    for candidate in SCALE_OPTIONS:
        if candidate not in scales:
            return candidate
    return SCALE_OPTIONS[0]


def _validate_instance(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False

    template = _extract_template(grid)
    template_colors = set(map(int, template.ravel()))
    if BACKGROUND in template_colors or FRAME_COLOR in template_colors:
        return False

    frames = _frame_infos(grid)
    if len(frames) not in FRAME_COUNT_OPTIONS:
        return False

    marker_pos = tuple(map(int, np.argwhere(template == MARKER_COLOR)[0]))
    scales = []
    shifted = 0
    canonical = 0
    for frame in frames:
        inner_h, inner_w = frame["inner_shape"]
        if inner_h != inner_w:
            return False
        area = int(inner_h * inner_w)
        side = math.isqrt(area)
        if side * side != area or side != inner_h or side % TEMPLATE_SIZE != 0:
            return False
        scale = side // TEMPLATE_SIZE
        scales.append(scale)
        inner_r0, inner_c0, _, _ = frame["inner_bbox"]
        marker_rel = (frame["marker"][0] - inner_r0, frame["marker"][1] - inner_c0)
        canonical_rel = (marker_pos[0] * scale, marker_pos[1] * scale)
        if marker_rel == canonical_rel:
            canonical += 1
        else:
            shifted += 1

    if len(set(scales)) < 2 or shifted == 0 or canonical == 0:
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    if np.array_equal(out, _solve(grid, use_marker=False)):
        return False

    wrong_scale = _constant_scale_for_negative_test(scales)
    if np.array_equal(out, _solve(grid, use_marker=True, scale_override=wrong_scale)):
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    template = _make_template(rng)
    template_marker = tuple(map(int, np.argwhere(template == MARKER_COLOR)[0]))

    frame_count = int(rng.choice(FRAME_COUNT_OPTIONS))
    scales = list(rng.choice(SCALE_OPTIONS, size=frame_count, replace=False))
    rng.shuffle(scales)

    shift_flags = [False, True]
    while len(shift_flags) < frame_count:
        shift_flags.append(bool(rng.integers(0, 2)))
    rng.shuffle(shift_flags)
    if not any(shift_flags) or all(shift_flags):
        shift_flags[0] = False
        shift_flags[1] = True

    patches = []
    for scale, force_shift in zip(scales, shift_flags):
        try:
            patch, _marker_rel, _delta = _build_frame_patch(template_marker, scale, rng, force_shift=force_shift)
        except ValueError:
            return None
        patches.append(patch)

    H = int(rng.integers(18, 31))
    W = int(rng.integers(18, 31))
    if max(max(patch.shape) for patch in patches) + 1 > min(H, W):
        return None

    try:
        placements = _place_rectangles(rng, (H, W), patches)
    except ValueError:
        return None

    grid = np.zeros((H, W), dtype=int)
    paste(grid, template, 0, 0, transparent=BACKGROUND, clip=False)
    for patch, (r0, c0) in zip(patches, placements):
        paste(grid, patch, r0, c0, transparent=BACKGROUND, clip=False)

    if not _validate_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
