from __future__ import annotations

import random

import numpy as np


_POSITIONS = ("top_left", "top_right", "bottom_left", "bottom_right")
_OPPOSITE = {
    "top_left": "bottom_right",
    "top_right": "bottom_left",
    "bottom_left": "top_right",
    "bottom_right": "top_left",
}
_MISSING_TO_ORIENTATION = {
    (1, 1): "top_left",
    (1, 0): "top_right",
    (0, 1): "bottom_left",
    (0, 0): "bottom_right",
}
_ORIENTATION_TO_MISSING = {v: k for k, v in _MISSING_TO_ORIENTATION.items()}

_GEN_RETRIES = 300


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _find_separator(grid: np.ndarray) -> tuple[int, int, int]:
    h, w = grid.shape
    row_candidates = [
        (r, int(grid[r, 0]))
        for r in range(1, h - 1)
        if np.all(grid[r, :] == grid[r, 0])
    ]
    col_candidates = [
        (c, int(grid[0, c]))
        for c in range(1, w - 1)
        if np.all(grid[:, c] == grid[0, c])
    ]

    matches = [
        (r, c, row_color)
        for r, row_color in row_candidates
        for c, col_color in col_candidates
        if row_color == col_color
    ]
    assert len(matches) == 1, "expected exactly one full-row/full-column separator cross"
    return matches[0]


def _quadrant_slices(split_r: int, split_c: int) -> dict[str, tuple[slice, slice]]:
    return {
        "top_left": (slice(0, split_r), slice(0, split_c)),
        "top_right": (slice(0, split_r), slice(split_c + 1, None)),
        "bottom_left": (slice(split_r + 1, None), slice(0, split_c)),
        "bottom_right": (slice(split_r + 1, None), slice(split_c + 1, None)),
    }


def _corner_slices(position: str, shape: tuple[int, int]) -> tuple[slice, slice]:
    h, w = shape
    if position == "top_left":
        return slice(0, 2), slice(0, 2)
    if position == "top_right":
        return slice(0, 2), slice(w - 2, w)
    if position == "bottom_left":
        return slice(h - 2, h), slice(0, 2)
    assert position == "bottom_right"
    return slice(h - 2, h), slice(w - 2, w)


def _extract_marker_box(patch: np.ndarray, position: str) -> np.ndarray:
    rs, cs = _corner_slices(position, patch.shape)
    return patch[rs, cs]


def _marker_orientation(box: np.ndarray, bg: int) -> str:
    values, counts = np.unique(box, return_counts=True)
    fg = [(int(v), int(c)) for v, c in zip(values, counts) if int(v) != int(bg)]
    assert len(fg) == 1 and fg[0][1] == 3, "marker must be a monochrome 3-pixel L"
    marker_color = fg[0][0]
    missing = np.argwhere(box != marker_color)
    assert missing.shape == (1, 2), "marker box must have one missing corner"
    return _MISSING_TO_ORIENTATION[(int(missing[0, 0]), int(missing[0, 1]))]


def _clean_patch(patch: np.ndarray, position: str, bg: int) -> np.ndarray:
    out = patch.copy()
    rs, cs = _corner_slices(position, patch.shape)
    out[rs, cs] = int(bg)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the self-pointing quadrant marker and restore that quadrant from its 180-degree opposite.

    Input structure:
    - A single-color cross separator splits the grid into four equal quadrants.
    - Each quadrant has a 3-pixel L marker in its outer 2x2 corner box, while the quadrant
      contents themselves occur in opposite pairs related by 180-degree rotation.
    - The self-indicated quadrant is damaged by a monochrome occluder, and some instances may
      include additional recoverable damage in a different opposite pair.

    Rule:
    - Find the unique L marker whose corner orientation matches the corner where it is placed.
    - Ignore all L markers when comparing quadrant content.
    - Take the opposite quadrant, erase its own marker corner, rotate it by 180 degrees, and
      output that restored content as the selected quadrant crop.

    Key insight:
    - The output choice is controlled by self-reference of the L orientation, not by which
      quadrant looks damaged; the missing pixels are recovered from the antipodal quadrant.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D ARC grid"

    bg = _get_bg(g)
    split_r, split_c, _ = _find_separator(g)
    quadrants = {
        position: g[rs, cs].copy()
        for position, (rs, cs) in _quadrant_slices(split_r, split_c).items()
    }

    marker_matches: list[str] = []
    cleaned: dict[str, np.ndarray] = {}
    for position, patch in quadrants.items():
        assert min(patch.shape) >= 2, "quadrants must be at least 2x2"
        orientation = _marker_orientation(_extract_marker_box(patch, position), bg)
        if orientation == position:
            marker_matches.append(position)
        cleaned[position] = _clean_patch(patch, position, bg)

    assert len(marker_matches) == 1, "expected exactly one self-referential corner marker"
    selected = marker_matches[0]
    source = _OPPOSITE[selected]
    return np.rot90(cleaned[source], 2).copy()


def _make_marker(orientation: str, color: int, bg: int) -> np.ndarray:
    marker = np.full((2, 2), int(bg), dtype=int)
    miss_r, miss_c = _ORIENTATION_TO_MISSING[orientation]
    for r in range(2):
        for c in range(2):
            if (r, c) != (miss_r, miss_c):
                marker[r, c] = int(color)
    return marker


def _reserved_corner_mask(shape: tuple[int, int], position: str) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    rs, cs = _corner_slices(position, shape)
    mask[rs, cs] = True
    return mask


def _random_rectangle_specs(
    height: int, width: int, forbidden: np.ndarray
) -> list[tuple[int, int, int, int]]:
    specs: list[tuple[int, int, int, int]] = []
    occupied = forbidden.copy()
    target_count = random.randint(3, 5)

    for _ in range(target_count):
        placed = False
        options = []
        for rect_h in range(1, min(4, height - 1) + 1):
            for rect_w in range(1, min(4, width - 1) + 1):
                for r0 in range(height - rect_h + 1):
                    for c0 in range(width - rect_w + 1):
                        region = occupied[r0 : r0 + rect_h, c0 : c0 + rect_w]
                        if bool(region.any()):
                            continue
                        options.append((r0, c0, rect_h, rect_w))
        random.shuffle(options)
        for spec in options:
            r0, c0, rect_h, rect_w = spec
            occupied[r0 : r0 + rect_h, c0 : c0 + rect_w] = True
            specs.append(spec)
            placed = True
            break
        if not placed:
            break
    return specs


def _generate_clean_patch(
    height: int, width: int, bg: int, colors: list[int], position: str
) -> np.ndarray | None:
    forbidden = _reserved_corner_mask((height, width), position)
    for _ in range(80):
        patch = np.full((height, width), int(bg), dtype=int)
        specs = _random_rectangle_specs(height, width, forbidden)
        if len(specs) < 3:
            continue

        used_colors: set[int] = set()
        for r0, c0, rect_h, rect_w in specs:
            color = int(random.choice(colors))
            patch[r0 : r0 + rect_h, c0 : c0 + rect_w] = color
            used_colors.add(color)

        fg_count = int(np.count_nonzero(patch != int(bg)))
        if fg_count < max(8, (height * width) // 6):
            continue
        if len(used_colors) < 2:
            continue
        return patch
    return None


def _occluder_candidates(patch: np.ndarray, bg: int, position: str) -> list[tuple[int, int, int, int]]:
    h, w = patch.shape
    forbidden = _reserved_corner_mask((h, w), position)
    candidates: list[tuple[int, int, int, int]] = []
    for occ_h in range(2, min(4, h) + 1):
        for occ_w in range(2, min(4, w) + 1):
            for r0 in range(h - occ_h + 1):
                for c0 in range(w - occ_w + 1):
                    if bool(forbidden[r0 : r0 + occ_h, c0 : c0 + occ_w].any()):
                        continue
                    region = patch[r0 : r0 + occ_h, c0 : c0 + occ_w]
                    if int(np.count_nonzero(region != int(bg))) < max(2, (occ_h * occ_w) // 2):
                        continue
                    candidates.append((r0, c0, occ_h, occ_w))
    return candidates


def _apply_occluder(
    patch: np.ndarray, occluder: tuple[int, int, int, int], color: int
) -> np.ndarray:
    r0, c0, occ_h, occ_w = occluder
    out = patch.copy()
    out[r0 : r0 + occ_h, c0 : c0 + occ_w] = int(color)
    return out


def _paint_marker(patch: np.ndarray, position: str, orientation: str, color: int, bg: int) -> np.ndarray:
    out = patch.copy()
    rs, cs = _corner_slices(position, out.shape)
    out[rs, cs] = _make_marker(orientation, color, bg)
    return out


def _assemble_grid(
    quadrants: dict[str, np.ndarray], split_r: int, split_c: int, divider: int, bg: int
) -> np.ndarray:
    h = 2 * split_r + 1
    w = 2 * split_c + 1
    grid = np.full((h, w), int(bg), dtype=int)
    grid[split_r, :] = int(divider)
    grid[:, split_c] = int(divider)
    for position, (rs, cs) in _quadrant_slices(split_r, split_c).items():
        grid[rs, cs] = quadrants[position]
    return grid


def _generate_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    quad_h = random.randint(8, 13)
    quad_w = random.randint(8, 13)

    palette = list(range(10))
    bg = int(random.choice(palette))
    remaining = [c for c in palette if c != bg]
    divider, occluder = random.sample(remaining, 2)
    remaining = [c for c in remaining if c not in {divider, occluder}]
    marker_colors = random.sample(remaining, 4)
    remaining = [c for c in remaining if c not in marker_colors]
    content_color_count = random.randint(2, min(3, len(remaining)))
    content_colors = random.sample(remaining, content_color_count)

    tl_clean = _generate_clean_patch(quad_h, quad_w, bg, content_colors, "top_left")
    tr_clean = _generate_clean_patch(quad_h, quad_w, bg, content_colors, "top_right")
    if tl_clean is None or tr_clean is None:
        return None

    clean_quadrants = {
        "top_left": tl_clean,
        "top_right": tr_clean,
        "bottom_left": np.rot90(tr_clean, 2).copy(),
        "bottom_right": np.rot90(tl_clean, 2).copy(),
    }

    selected = random.choice(_POSITIONS)
    opposite = _OPPOSITE[selected]
    expected = clean_quadrants[selected].copy()

    selected_candidates = _occluder_candidates(clean_quadrants[selected], bg, selected)
    if not selected_candidates:
        return None
    input_quadrants = {pos: patch.copy() for pos, patch in clean_quadrants.items()}
    input_quadrants[selected] = _apply_occluder(
        input_quadrants[selected], random.choice(selected_candidates), occluder
    )

    decoy_positions = [pos for pos in _POSITIONS if pos not in {selected, opposite}]
    if random.random() < 0.7:
        random.shuffle(decoy_positions)
        placed_decoy = False
        for decoy_position in decoy_positions:
            decoy_candidates = _occluder_candidates(clean_quadrants[decoy_position], bg, decoy_position)
            if not decoy_candidates:
                continue
            input_quadrants[decoy_position] = _apply_occluder(
                input_quadrants[decoy_position], random.choice(decoy_candidates), occluder
            )
            placed_decoy = True
            break
        if not placed_decoy and decoy_positions:
            return None

    orientations: dict[str, str] = {}
    for position in _POSITIONS:
        if position == selected:
            orientations[position] = position
        else:
            wrong_choices = [ori for ori in _POSITIONS if ori != position]
            orientations[position] = random.choice(wrong_choices)

    for idx, position in enumerate(_POSITIONS):
        input_quadrants[position] = _paint_marker(
            input_quadrants[position],
            position,
            orientations[position],
            marker_colors[idx],
            bg,
        )

    grid = _assemble_grid(input_quadrants, quad_h, quad_w, divider, bg)
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        attempt = _generate_attempt()
        if attempt is None:
            continue
        grid, expected = attempt
        try:
            solved = main(grid)
        except Exception:
            continue
        if np.array_equal(solved, expected):
            return grid
    raise RuntimeError("failed to generate a valid self-referential quadrant grid")
