from __future__ import annotations

import random

import numpy as np


QUADRANTS = ("tl", "tr", "bl", "br")
GRID_HEIGHT_RANGE = (20, 30)
GRID_WIDTH_RANGE = (20, 30)
CONTAINER_COUNT_CHOICES = (1, 2, 3)
MAX_GENERATION_ATTEMPTS = 300
PACKING_PADDING = 1

BASE_FRAGMENT_MASKS = (
    np.array([[1, 1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1], [1, 0]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),
)


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _round_half_up(value: float) -> int:
    return int(np.floor(float(value) + 0.5))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for row in range(height):
        for col in range(width):
            if not mask[row, col] or seen[row, col]:
                continue
            stack = [(row, col)]
            seen[row, col] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _component_center(coords: np.ndarray) -> tuple[int, int]:
    row = _round_half_up(float(coords[:, 0].mean()))
    col = _round_half_up(float(coords[:, 1].mean()))
    return row, col


def _make_component_patch(
    coords: np.ndarray,
    color: int,
    bg: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    rel = coords - np.array([r0, c0], dtype=int)
    patch[rel[:, 0], rel[:, 1]] = int(color)
    return patch, (r0, c0, r1, c1)


def _is_hollow_frame(grid: np.ndarray, coords: np.ndarray, color: int, bg: int) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False

    local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    local[rel[:, 0], rel[:, 1]] = True

    border = np.zeros_like(local, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if not np.array_equal(local, border):
        return False

    return bool(np.all(grid[r0 + 1 : r1, c0 + 1 : c1] == int(bg)))


def _paste_transparent(grid: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    height, width = patch.shape
    target = grid[top : top + height, left : left + width]
    mask = patch != int(bg)
    assert target.shape == patch.shape
    assert np.all(target[mask] == int(bg))
    target[mask] = patch[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each colored fragment into the same-color hollow frame.

    The fragment's outside quadrant relative to the frame center determines
    which inside corner it occupies after translation.
    """
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert colors, "expected at least one non-background color"

    parsed: list[tuple[int, np.ndarray, list[np.ndarray]]] = []
    for color in colors:
        comps = _components(g == int(color))
        assert comps, f"expected components for color {color}"

        frame_indices = [
            idx for idx, comp in enumerate(comps)
            if _is_hollow_frame(g, comp, color, bg)
        ]
        assert len(frame_indices) == 1, f"expected one frame for color {color}"
        frame_idx = frame_indices[0]
        fragments = [comp for idx, comp in enumerate(comps) if idx != frame_idx]
        assert 1 <= len(fragments) <= 4, f"expected 1-4 fragments for color {color}"
        parsed.append((color, comps[frame_idx], fragments))

    out = g.copy()
    for _, _frame, fragments in parsed:
        for frag in fragments:
            out[frag[:, 0], frag[:, 1]] = int(bg)

    for color, frame, fragments in parsed:
        fr0, fc0, fr1, fc1 = _bbox(frame)
        inner_r0, inner_c0 = fr0 + 1, fc0 + 1
        inner_r1, inner_c1 = fr1 - 1, fc1 - 1
        assert inner_r0 <= inner_r1 and inner_c0 <= inner_c1
        frame_center = _component_center(frame)
        used_corners: set[str] = set()

        for frag in fragments:
            patch, _ = _make_component_patch(frag, color, bg)
            frag_center = _component_center(frag)
            assert frag_center[0] != frame_center[0] and frag_center[1] != frame_center[1]

            vertical = "t" if frag_center[0] < frame_center[0] else "b"
            horizontal = "l" if frag_center[1] < frame_center[1] else "r"
            quadrant = vertical + horizontal
            assert quadrant not in used_corners, "generator should avoid same-corner collisions"
            used_corners.add(quadrant)

            patch_h, patch_w = patch.shape
            dest_r = inner_r0 if vertical == "t" else inner_r1 - patch_h + 1
            dest_c = inner_c0 if horizontal == "l" else inner_c1 - patch_w + 1
            assert dest_r >= inner_r0 and dest_c >= inner_c0
            assert dest_r + patch_h - 1 <= inner_r1 and dest_c + patch_w - 1 <= inner_c1
            _paste_transparent(out, patch, dest_r, dest_c, bg)

    return out


def _draw_frame(patch: np.ndarray, top: int, left: int, height: int, width: int, color: int) -> None:
    bottom = top + height - 1
    right = left + width - 1
    patch[top, left : right + 1] = int(color)
    patch[bottom, left : right + 1] = int(color)
    patch[top : bottom + 1, left] = int(color)
    patch[top : bottom + 1, right] = int(color)


def _transform_mask(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    out = np.rot90(out, random.randrange(4))
    return np.asarray(out, dtype=bool)


def _sample_fragment_mask() -> np.ndarray:
    return _transform_mask(random.choice(BASE_FRAGMENT_MASKS))


def _sample_quadrants() -> list[str]:
    count = random.randint(1, len(QUADRANTS))
    return random.sample(list(QUADRANTS), count)


def _cluster_limits(container_count: int) -> tuple[int, int, int]:
    frame_extra = {1: 2, 2: 1, 3: 1}[container_count]
    margin_extra = {1: 1, 2: 1, 3: 0}[container_count]
    size_cap = {1: 24, 2: 18, 3: 15}[container_count]
    return frame_extra, margin_extra, size_cap


def _build_cluster(color: int, bg: int, container_count: int) -> tuple[np.ndarray, np.ndarray]:
    frame_extra, margin_extra, size_cap = _cluster_limits(container_count)

    for _ in range(80):
        quadrants = _sample_quadrants()
        shapes = {quadrant: _sample_fragment_mask() for quadrant in quadrants}

        top_h = max((shapes[q].shape[0] for q in ("tl", "tr") if q in shapes), default=0)
        bottom_h = max((shapes[q].shape[0] for q in ("bl", "br") if q in shapes), default=0)
        left_w = max((shapes[q].shape[1] for q in ("tl", "bl") if q in shapes), default=0)
        right_w = max((shapes[q].shape[1] for q in ("tr", "br") if q in shapes), default=0)

        interior_h_min = max(3, top_h + bottom_h)
        interior_w_min = max(3, left_w + right_w)
        interior_h = random.randint(interior_h_min, interior_h_min + frame_extra)
        interior_w = random.randint(interior_w_min, interior_w_min + frame_extra)
        frame_h = interior_h + 2
        frame_w = interior_w + 2

        top_margin = (top_h + 1 + random.randint(0, margin_extra)) if top_h else random.randint(0, margin_extra)
        bottom_margin = (bottom_h + 1 + random.randint(0, margin_extra)) if bottom_h else random.randint(0, margin_extra)
        left_margin = (left_w + 1 + random.randint(0, margin_extra)) if left_w else random.randint(0, margin_extra)
        right_margin = (right_w + 1 + random.randint(0, margin_extra)) if right_w else random.randint(0, margin_extra)

        total_h = top_margin + frame_h + bottom_margin
        total_w = left_margin + frame_w + right_margin
        if total_h > size_cap or total_w > size_cap:
            continue

        frame_top = top_margin
        frame_left = left_margin
        inner_r0, inner_c0 = frame_top + 1, frame_left + 1
        inner_r1, inner_c1 = frame_top + frame_h - 2, frame_left + frame_w - 2

        inp = np.full((total_h, total_w), bg, dtype=int)
        out = np.full((total_h, total_w), bg, dtype=int)
        _draw_frame(inp, frame_top, frame_left, frame_h, frame_w, color)
        _draw_frame(out, frame_top, frame_left, frame_h, frame_w, color)

        for quadrant, mask in shapes.items():
            frag_h, frag_w = mask.shape

            if quadrant[0] == "t":
                row_lo, row_hi = 0, frame_top - frag_h - 1
                dest_r = inner_r0
            else:
                row_lo, row_hi = frame_top + frame_h + 1, total_h - frag_h
                dest_r = inner_r1 - frag_h + 1

            if quadrant[1] == "l":
                col_lo, col_hi = 0, frame_left - frag_w - 1
                dest_c = inner_c0
            else:
                col_lo, col_hi = frame_left + frame_w + 1, total_w - frag_w
                dest_c = inner_c1 - frag_w + 1

            assert row_lo <= row_hi and col_lo <= col_hi
            src_r = random.randint(row_lo, row_hi)
            src_c = random.randint(col_lo, col_hi)
            inp[src_r : src_r + frag_h, src_c : src_c + frag_w][mask] = int(color)

            target = out[dest_r : dest_r + frag_h, dest_c : dest_c + frag_w]
            assert np.all(target[mask] == int(bg))
            target[mask] = int(color)

        return inp, out

    raise RuntimeError("failed to build cluster")


def _rectangles_conflict(
    rect_a: tuple[int, int, int, int],
    rect_b: tuple[int, int, int, int],
    padding: int,
) -> bool:
    ar0, ac0, ar1, ac1 = rect_a
    br0, bc0, br1, bc1 = rect_b
    return not (
        ar1 + padding < br0
        or br1 + padding < ar0
        or ac1 + padding < bc0
        or bc1 + padding < ac0
    )


def _pack_clusters(
    patches: list[tuple[np.ndarray, np.ndarray]],
    height: int,
    width: int,
) -> list[tuple[int, int]] | None:
    order = sorted(
        range(len(patches)),
        key=lambda idx: patches[idx][0].shape[0] * patches[idx][0].shape[1],
        reverse=True,
    )
    placements: list[tuple[int, int] | None] = [None] * len(patches)
    placed_rects: list[tuple[int, int, int, int]] = []

    for idx in order:
        patch_h, patch_w = patches[idx][0].shape
        candidates: list[tuple[int, int]] = []
        for top in range(height - patch_h + 1):
            for left in range(width - patch_w + 1):
                rect = (top, left, top + patch_h - 1, left + patch_w - 1)
                if all(not _rectangles_conflict(rect, other, PACKING_PADDING) for other in placed_rects):
                    candidates.append((top, left))
        if not candidates:
            return None
        choice = random.choice(candidates)
        placements[idx] = choice
        placed_rects.append((choice[0], choice[1], choice[0] + patch_h - 1, choice[1] + patch_w - 1))

    return [placement for placement in placements if placement is not None]


def _compose_scene(
    patches: list[tuple[np.ndarray, np.ndarray]],
    placements: list[tuple[int, int]],
    bg: int,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full(shape, bg, dtype=int)
    out = np.full(shape, bg, dtype=int)
    for (inp_patch, out_patch), (top, left) in zip(patches, placements):
        ih, iw = inp_patch.shape
        inp_slice = inp[top : top + ih, left : left + iw]
        out_slice = out[top : top + ih, left : left + iw]
        inp_mask = inp_patch != int(bg)
        out_mask = out_patch != int(bg)
        assert np.all(inp_slice[inp_mask] == int(bg))
        assert np.all(out_slice[out_mask] == int(bg))
        inp_slice[inp_mask] = inp_patch[inp_mask]
        out_slice[out_mask] = out_patch[out_mask]
    return inp, out


def _generate_candidate() -> np.ndarray | None:
    bg = random.randrange(10)
    container_count = random.choice(CONTAINER_COUNT_CHOICES)
    colors = random.sample([color for color in range(10) if color != bg], container_count)
    patches = [_build_cluster(color, bg, container_count) for color in colors]

    max_patch_h = max(patch[0].shape[0] for patch in patches)
    max_patch_w = max(patch[0].shape[1] for patch in patches)
    total_area = sum(patch[0].shape[0] * patch[0].shape[1] for patch in patches)

    for _ in range(80):
        height = random.randint(GRID_HEIGHT_RANGE[0], GRID_HEIGHT_RANGE[1])
        width = random.randint(GRID_WIDTH_RANGE[0], GRID_WIDTH_RANGE[1])
        if height < max_patch_h or width < max_patch_w:
            continue
        if height * width < total_area + 60:
            continue
        placements = _pack_clusters(patches, height, width)
        if placements is None:
            continue

        inp, expected = _compose_scene(patches, placements, bg, (height, width))
        if _dominant_color(inp) != bg:
            continue
        actual = main(inp)
        if np.array_equal(actual, expected):
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate puzzle 6 input")
