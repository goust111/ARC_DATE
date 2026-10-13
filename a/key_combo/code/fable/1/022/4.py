from __future__ import annotations

import numpy as np


_DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_UNIQUE_BITS = (1, 2, 4)
_PAIR_BITS = (3, 5, 6)
_ALL_BITS = (1, 2, 3, 4, 5, 6, 7)
_GEN_RETRIES = 400


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _split_quadrants(grid: np.ndarray) -> tuple[list[np.ndarray], int]:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    h, w = map(int, g.shape)
    if h % 2 != 1 or w % 2 != 1:
        raise ValueError("separator cross requires odd grid dimensions")

    mid_r = h // 2
    mid_c = w // 2
    cross_color = int(g[mid_r, mid_c])
    if not np.all(g[mid_r, :] == cross_color):
        raise ValueError("expected a solid separator row")
    if not np.all(g[:, mid_c] == cross_color):
        raise ValueError("expected a solid separator column")

    quads = [
        g[:mid_r, :mid_c].copy(),
        g[:mid_r, mid_c + 1 :].copy(),
        g[mid_r + 1 :, :mid_c].copy(),
        g[mid_r + 1 :, mid_c + 1 :].copy(),
    ]
    if len({quad.shape for quad in quads}) != 1:
        raise ValueError("quadrants must have equal size")
    return quads, cross_color


def _infer_background(quads: list[np.ndarray]) -> int:
    common: set[int] | None = None
    for quad in quads:
        colors = set(map(int, np.unique(quad)))
        common = colors if common is None else (common & colors)
    if common is None or len(common) != 1:
        raise ValueError("failed to infer a unique shared background color")
    return int(next(iter(common)))


def _extract_marker_and_masks(
    quads: list[np.ndarray], bg: int
) -> tuple[int, int, list[np.ndarray], list[int]]:
    marker_color: int | None = None
    marker_idx: int | None = None
    masks: list[np.ndarray] = []
    mask_colors: list[int] = []

    for idx, quad in enumerate(quads):
        fg_coords = np.argwhere(quad != bg)
        if fg_coords.shape[0] == 1:
            if marker_idx is not None:
                raise ValueError("expected exactly one marker quadrant")
            marker_idx = idx
            rr, cc = map(int, fg_coords[0])
            marker_color = int(quad[rr, cc])
            if len(_colors_present(quad, exclude={bg})) != 1:
                raise ValueError("marker quadrant must contain only one foreground color")
        else:
            fg_colors = _colors_present(quad, exclude={bg})
            if len(fg_colors) != 1:
                raise ValueError("each mask quadrant must contain one foreground color")
            masks.append(quad != bg)
            mask_colors.append(int(fg_colors[0]))

    if marker_idx is None or marker_color is None:
        raise ValueError("failed to locate the marker quadrant")
    if len(masks) != 3:
        raise ValueError("expected exactly three mask quadrants")
    if len(set(mask_colors)) != 3:
        raise ValueError("the three mask colors must be distinct")
    return int(marker_color), int(marker_idx), masks, mask_colors


def _vote_counts(masks: list[np.ndarray]) -> np.ndarray:
    counts = np.zeros(masks[0].shape, dtype=int)
    for mask in masks:
        counts += mask.astype(int)
    return counts


def _expected_output(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
    quads, _cross_color = _split_quadrants(grid)
    bg = _infer_background(quads)
    marker_color, marker_idx, masks, _mask_colors = _extract_marker_and_masks(quads, bg)
    votes = _vote_counts(masks)
    out = np.full(masks[0].shape, bg, dtype=int)
    out[votes == 1] = marker_color
    return out, votes, bg, marker_idx


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align the three quadrant masks and keep only cells occupied by exactly one mask, colored by the lone marker pixel.

    Input structure:
    - A solid full row and full column of one color form a separator cross, splitting the grid into four equal quadrants.
    - Three quadrants each contain one scattered foreground mask on a shared background, and those three mask colors are distinct.
    - The remaining quadrant contains only a single marker pixel on that same background.

    Rule:
    - Remove the separator cross and align the three mask quadrants cell by cell in quadrant-local coordinates.
    - Produce one quadrant-sized output grid on the shared background.
    - A cell is painted with the marker pixel's color iff that aligned position is foreground in exactly one of the three masks; positions hit zero, two, or three times stay background.

    Key insight:
    - The three colored shapes are only occupancy masks; their colors do not combine.
    - The singleton marker contributes only the output color and identifies the empty quadrant, so the target logic is exact-one voting rather than union, intersection, or parity.
    """
    out, _votes, _bg, _marker_idx = _expected_output(np.asarray(grid, dtype=int))
    return out


def _randint(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(int(lo), int(hi) + 1))


def _sample_connected_patch(
    rng: np.random.Generator, box_h: int, box_w: int, min_cells: int, max_cells: int
) -> np.ndarray:
    target = _randint(rng, min_cells, min(max_cells, box_h * box_w))
    patch = np.zeros((box_h, box_w), dtype=bool)
    start_r = _randint(rng, 0, box_h - 1)
    start_c = _randint(rng, 0, box_w - 1)
    patch[start_r, start_c] = True

    while int(patch.sum()) < target:
        candidates: list[tuple[int, int]] = []
        for rr, cc in np.argwhere(patch):
            for dr, dc in _DIRS4:
                nr = int(rr + dr)
                nc = int(cc + dc)
                if 0 <= nr < box_h and 0 <= nc < box_w and not patch[nr, nc]:
                    candidates.append((nr, nc))
        if not candidates:
            break
        nr, nc = candidates[_randint(rng, 0, len(candidates) - 1)]
        patch[nr, nc] = True

    return patch


def _can_place(layout: np.ndarray, top: int, left: int, patch: np.ndarray, pad: int) -> bool:
    h, w = patch.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(layout.shape[0], top + h + pad)
    c1 = min(layout.shape[1], left + w + pad)
    return not np.any(layout[r0:r1, c0:c1] != 0)


def _place_patch(
    layout: np.ndarray,
    rng: np.random.Generator,
    bit_value: int,
    *,
    min_cells: int,
    max_cells: int,
    pad: int,
    tries: int = 300,
) -> bool:
    h, w = map(int, layout.shape)
    max_box_h = min(5, h)
    max_box_w = min(5, w)

    for _ in range(tries):
        box_h = _randint(rng, 1, max_box_h)
        box_w = _randint(rng, 1, max_box_w)
        if box_h * box_w < min_cells:
            continue
        patch = _sample_connected_patch(rng, box_h, box_w, min_cells, max_cells)
        top = _randint(rng, 0, h - box_h)
        left = _randint(rng, 0, w - box_w)
        if not _can_place(layout, top, left, patch, pad):
            continue
        view = layout[top : top + box_h, left : left + box_w]
        view[patch] = int(bit_value)
        return True
    return False


def _layout_to_masks(layout: np.ndarray) -> list[np.ndarray]:
    return [
        np.isin(layout, (1, 3, 5, 7)),
        np.isin(layout, (2, 3, 6, 7)),
        np.isin(layout, (4, 5, 6, 7)),
    ]


def _build_vote_layout(rng: np.random.Generator, qh: int, qw: int) -> np.ndarray | None:
    layout = np.zeros((qh, qw), dtype=int)
    mandatory = [7, int(rng.choice(_PAIR_BITS)), 1, 2, 4]
    for bit_value in mandatory:
        max_cells = 5 if bit_value in _UNIQUE_BITS else 4
        if not _place_patch(
            layout,
            rng,
            bit_value,
            min_cells=2,
            max_cells=max_cells,
            pad=1,
        ):
            return None

    extra_count = _randint(rng, 1, 5)
    for _ in range(extra_count):
        bit_value = int(rng.choice(_ALL_BITS))
        _place_patch(
            layout,
            rng,
            bit_value,
            min_cells=1 if bit_value not in _UNIQUE_BITS else 2,
            max_cells=4,
            pad=1 if rng.random() < 0.85 else 0,
        )

    layout = np.rot90(layout, k=int(rng.integers(4)))
    if rng.random() < 0.5:
        layout = np.flipud(layout)
    if rng.random() < 0.5:
        layout = np.fliplr(layout)
    return np.asarray(layout, dtype=int)


def _compose_grid(
    rng: np.random.Generator,
    masks: list[np.ndarray],
    *,
    bg: int,
    separator_color: int,
    mask_colors: list[int],
    marker_color: int,
    marker_idx: int,
) -> np.ndarray:
    qh, qw = masks[0].shape
    grid = np.full((2 * qh + 1, 2 * qw + 1), bg, dtype=int)
    grid[qh, :] = separator_color
    grid[:, qw] = separator_color

    quad_slices = [
        (slice(0, qh), slice(0, qw)),
        (slice(0, qh), slice(qw + 1, 2 * qw + 1)),
        (slice(qh + 1, 2 * qh + 1), slice(0, qw)),
        (slice(qh + 1, 2 * qh + 1), slice(qw + 1, 2 * qw + 1)),
    ]

    mask_order = list(map(int, rng.permutation(3)))
    next_mask = 0
    for quad_idx, (rows, cols) in enumerate(quad_slices):
        patch = np.full((qh, qw), bg, dtype=int)
        if quad_idx == marker_idx:
            mr = _randint(rng, 0, qh - 1)
            mc = _randint(rng, 0, qw - 1)
            patch[mr, mc] = marker_color
        else:
            mask = masks[mask_order[next_mask]]
            color = int(mask_colors[mask_order[next_mask]])
            patch[mask] = color
            next_mask += 1
        grid[rows, cols] = patch
    return grid


def _semantic_ok(grid: np.ndarray) -> bool:
    try:
        out, votes, bg, marker_idx = _expected_output(grid)
    except ValueError:
        return False

    quads, cross_color = _split_quadrants(grid)
    marker_color, parsed_marker_idx, masks, mask_colors = _extract_marker_and_masks(quads, bg)
    if marker_idx != parsed_marker_idx:
        return False
    if marker_color == bg or marker_color == cross_color:
        return False
    if len(set(mask_colors + [marker_color, cross_color, bg])) != 6:
        return False
    if np.count_nonzero(votes == 1) == 0:
        return False
    if not {0, 1, 2, 3}.issubset(set(map(int, np.unique(votes)))):
        return False
    if np.count_nonzero(votes > 0) >= int(0.7 * votes.size):
        return False
    if any(np.count_nonzero(mask) >= int(0.55 * mask.size) for mask in masks):
        return False

    exact_one = votes == 1
    if np.array_equal(exact_one, votes > 0):
        return False
    if np.array_equal(exact_one, (votes % 2) == 1):
        return False
    if np.array_equal(exact_one, votes == 3):
        return False
    if not np.array_equal(main(grid), out):
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    qh = _randint(rng, 7, 12)
    qw = _randint(rng, 7, 12)
    layout = _build_vote_layout(rng, qh, qw)
    if layout is None:
        return None

    masks = _layout_to_masks(layout)
    bg = 0
    separator_color, marker_color, *mask_colors = map(
        int, rng.choice(np.arange(1, 10), size=5, replace=False)
    )
    marker_idx = _randint(rng, 0, 3)
    grid = _compose_grid(
        rng,
        masks,
        bg=bg,
        separator_color=separator_color,
        mask_colors=mask_colors,
        marker_color=marker_color,
        marker_idx=marker_idx,
    )
    if not _semantic_ok(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid exact-one voting input")
