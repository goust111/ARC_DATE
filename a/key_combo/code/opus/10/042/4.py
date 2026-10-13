from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7
MAROON = 9

SPRITE_COLORS = (BLUE, GREEN, YELLOW)
MARKER_COLORS = (ORANGE, MAROON)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _scale(sprite: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(sprite, np.ones((factor, factor), dtype=int))


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _find_region_and_markers(grid: np.ndarray):
    marker_mask = np.isin(grid, MARKER_COLORS)
    marker_coords = [tuple(map(int, rc)) for rc in np.argwhere(marker_mask)]
    assert len(marker_coords) == 4, "expected four corner markers"
    box = _bbox(marker_mask)
    assert box is not None
    r0, c0, r1, c1 = box
    expected = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    assert set(marker_coords) == expected, "markers must occupy the rectangle corners"
    maroons = [coord for coord in marker_coords if int(grid[coord]) == MAROON]
    assert len(maroons) == 1, "expected exactly one maroon corner"
    return box, marker_coords, maroons[0]


def _find_red_axis(region: np.ndarray) -> tuple[str, int]:
    full_red_rows = [r for r in range(region.shape[0]) if np.all(region[r, :] == RED)]
    full_red_cols = [c for c in range(region.shape[1]) if np.all(region[:, c] == RED)]
    assert (len(full_red_rows) == 1) ^ (len(full_red_cols) == 1), "expected one red symmetry axis"
    if full_red_rows:
        return "h", int(full_red_rows[0])
    return "v", int(full_red_cols[0])


def _extract_external_sprite(grid: np.ndarray, region_box: tuple[int, int, int, int]) -> np.ndarray:
    sprite_mask = np.isin(grid, SPRITE_COLORS)
    r0, c0, r1, c1 = region_box
    sprite_mask[r0 : r1 + 1, c0 : c1 + 1] = False
    box = _bbox(sprite_mask)
    assert box is not None, "expected one external sprite"
    sprite = _crop(grid, box)
    assert np.any(np.isin(sprite, SPRITE_COLORS))
    return sprite


def _compose_region(
    scaled: np.ndarray,
    region_shape: tuple[int, int],
    axis_kind: str,
    axis_idx: int,
    offset: int,
    markers: dict[tuple[int, int], int],
) -> np.ndarray:
    region = np.zeros(region_shape, dtype=int)
    if axis_kind == "v":
        region[:, offset : offset + scaled.shape[1]] = scaled
        region[:, axis_idx] = RED
    else:
        region[offset : offset + scaled.shape[0], :] = scaled
        region[axis_idx, :] = RED
    for (r, c), color in markers.items():
        region[r, c] = color
    return region


def _crop_selected_half(region: np.ndarray, axis_kind: str, axis_idx: int, maroon_local: tuple[int, int]) -> np.ndarray:
    mr, mc = maroon_local
    if axis_kind == "v":
        if mc < axis_idx:
            return region[:, : axis_idx + 1].copy()
        return region[:, axis_idx:].copy()
    if mr < axis_idx:
        return region[: axis_idx + 1, :].copy()
    return region[axis_idx:, :].copy()


def _solve_details(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    region_box, marker_coords, maroon_abs = _find_region_and_markers(g)
    r0, c0, r1, c1 = region_box
    region = g[r0 : r1 + 1, c0 : c1 + 1]
    axis_kind, axis_idx = _find_red_axis(region)
    sprite = _extract_external_sprite(g, region_box)
    scaled = None

    sh, sw = sprite.shape
    rh, rw = region.shape
    if axis_kind == "v":
        assert rh % sh == 0 and (rw - 1) % sw == 0
        k_row = rh // sh
        k_col = (rw - 1) // sw
        assert k_row == k_col and k_row >= 1
        scaled = _scale(sprite, k_row)
        assert scaled.shape == (rh, rw - 1)
        candidate_offsets = [0, 1]
    else:
        assert (rh - 1) % sh == 0 and rw % sw == 0
        k_row = (rh - 1) // sh
        k_col = rw // sw
        assert k_row == k_col and k_row >= 1
        scaled = _scale(sprite, k_row)
        assert scaled.shape == (rh - 1, rw)
        candidate_offsets = [0, 1]

    local_markers = {(mr - r0, mc - c0): int(g[mr, mc]) for mr, mc in marker_coords}
    maroon_local = (maroon_abs[0] - r0, maroon_abs[1] - c0)
    evidence = [
        (rr, cc, int(region[rr, cc]))
        for rr in range(rh)
        for cc in range(rw)
        if int(region[rr, cc]) in SPRITE_COLORS
    ]

    valid_offsets = []
    candidate_outputs = {}
    for offset in candidate_offsets:
        ok = True
        for rr, cc, color in evidence:
            if axis_kind == "v":
                sr, sc = rr, cc - offset
            else:
                sr, sc = rr - offset, cc
            if not (0 <= sr < scaled.shape[0] and 0 <= sc < scaled.shape[1]):
                ok = False
                break
            if int(scaled[sr, sc]) != color:
                ok = False
                break
        if ok:
            valid_offsets.append(offset)
            composed = _compose_region(scaled, (rh, rw), axis_kind, axis_idx, offset, local_markers)
            candidate_outputs[offset] = _crop_selected_half(composed, axis_kind, axis_idx, maroon_local)

    assert len(valid_offsets) == 1, "evidence must determine a unique scaled placement"
    chosen_offset = valid_offsets[0]
    return candidate_outputs[chosen_offset], {
        "region_box": region_box,
        "axis_kind": axis_kind,
        "axis_idx": axis_idx,
        "valid_offsets": tuple(valid_offsets),
        "candidate_outputs": candidate_outputs,
        "chosen_offset": chosen_offset,
        "maroon_local": maroon_local,
        "scaled": scaled,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Scale the external sprite into the corner-marked region, then crop the half chosen by the lone maroon corner.

    Input structure:
    - A small multicolor sprite of blue, green, yellow, and black sits outside a larger corner-marked rectangle.
    - Four singleton corner markers delimit the target rectangle; three are orange and one is maroon.
    - Inside that rectangle, a full red row or column forms the symmetry axis, and a few colored evidence pixels agree with one scaled placement of the sprite.

    Rule:
    - Use the four corner markers to locate the rectangle and the red line to determine whether the axis is horizontal or vertical.
    - Compute the integer scale from the sprite size and the rectangle's usable size after removing the axis line thickness, then test the two axis-adjacent placements and keep the one matching every evidence pixel.
    - Reconstruct the full scaled sprite in the rectangle, restore the red axis and corner markers, and crop the half on the maroon corner's side while keeping the red axis in the output.

    Key insight:
    - The corner markers carry two linked meanings at once: they define the scaling/cropping rectangle, and the unique maroon corner chooses which side of the red symmetry axis survives after reconstruction.
    """
    out, _details = _solve_details(np.asarray(grid, dtype=int))
    return out


def _random_sprite() -> np.ndarray:
    for _ in range(400):
        size = random.choice((3, 4))
        sprite = np.zeros((size, size), dtype=int)
        cell_count = random.randint(size + 2, size * size - 1)
        chosen = random.sample([(r, c) for r in range(size) for c in range(size)], cell_count)
        palette = random.sample(SPRITE_COLORS, random.choice((2, 3)))
        for r, c in chosen:
            sprite[r, c] = random.choice(palette)
        if any(np.all(sprite[r, :] == BLACK) for r in range(size)):
            continue
        if any(np.all(sprite[:, c] == BLACK) for c in range(size)):
            continue
        if np.count_nonzero(sprite == BLACK) == 0:
            continue
        if len({int(v) for v in sprite.flat if int(v) != BLACK}) < 2:
            continue
        if not _connected(sprite != BLACK):
            continue
        if np.array_equal(sprite, np.fliplr(sprite)) or np.array_equal(sprite, np.flipud(sprite)) or np.array_equal(sprite, np.rot90(sprite, 2)):
            continue
        return sprite
    raise RuntimeError("failed to generate a non-degenerate sprite")


def _pick_scale(sprite_size: int) -> int:
    choices = [2, 4] if sprite_size == 3 else [2, 3, 4]
    return random.choice(choices)


def _corner_positions(grid_shape: tuple[int, int], obj_shape: tuple[int, int]) -> list[tuple[int, int]]:
    gh, gw = grid_shape
    oh, ow = obj_shape
    return [
        (random.randint(0, 3), random.randint(0, 3)),
        (random.randint(0, 3), gw - ow - random.randint(0, 3)),
        (gh - oh - random.randint(0, 3), random.randint(0, 3)),
        (gh - oh - random.randint(0, 3), gw - ow - random.randint(0, 3)),
    ]


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], padding: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + padding < br0
        or br1 + padding < ar0
        or ac1 + padding < bc0
        or bc1 + padding < ac0
    )


def _choose_markers(region_shape: tuple[int, int], keep_side: str) -> tuple[dict[tuple[int, int], int], tuple[int, int]]:
    h, w = region_shape
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    if keep_side == "left":
        maroon_coord = random.choice([(0, 0), (h - 1, 0)])
    elif keep_side == "right":
        maroon_coord = random.choice([(0, w - 1), (h - 1, w - 1)])
    elif keep_side == "top":
        maroon_coord = random.choice([(0, 0), (0, w - 1)])
    else:
        maroon_coord = random.choice([(h - 1, 0), (h - 1, w - 1)])
    markers = {coord: ORANGE for coord in corners}
    markers[maroon_coord] = MAROON
    return markers, maroon_coord


def _choose_evidence(
    correct_region: np.ndarray,
    wrong_region: np.ndarray,
    kept_mask: np.ndarray,
) -> list[tuple[int, int, int]]:
    informative = []
    for r in range(correct_region.shape[0]):
        for c in range(correct_region.shape[1]):
            color = int(correct_region[r, c])
            if color not in SPRITE_COLORS:
                continue
            if int(wrong_region[r, c]) == color:
                continue
            informative.append((r, c, color))
    if len(informative) < 2:
        raise ValueError("not enough informative evidence cells")

    keep_cells = [cell for cell in informative if kept_mask[cell[0], cell[1]]]
    other_cells = [cell for cell in informative if not kept_mask[cell[0], cell[1]]]

    target_count = 3 if len(informative) >= 3 and random.random() < 0.6 else 2
    evidence: list[tuple[int, int, int]] = []

    if keep_cells:
        evidence.append(random.choice(keep_cells))
    if other_cells and len(evidence) < target_count:
        pick = random.choice(other_cells)
        if pick not in evidence:
            evidence.append(pick)

    pool = informative[:]
    random.shuffle(pool)
    for cell in pool:
        if cell not in evidence:
            evidence.append(cell)
        if len(evidence) == target_count:
            break

    if len(evidence) < 2:
        raise ValueError("failed to select enough evidence")
    return evidence


def _kept_mask(shape: tuple[int, int], axis_kind: str, axis_idx: int, keep_side: str) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    if axis_kind == "v":
        if keep_side == "left":
            mask[:, : axis_idx + 1] = True
        else:
            mask[:, axis_idx:] = True
    else:
        if keep_side == "top":
            mask[: axis_idx + 1, :] = True
        else:
            mask[axis_idx:, :] = True
    return mask


def _build_case():
    sprite = _random_sprite()
    scale = _pick_scale(sprite.shape[0])
    scaled = _scale(sprite, scale)
    axis_kind = random.choice(("v", "h"))
    if axis_kind == "v":
        region_shape = (scaled.shape[0], scaled.shape[1] + 1)
        axis_idx = region_shape[1] // 2
        keep_side = random.choice(("left", "right"))
    else:
        region_shape = (scaled.shape[0] + 1, scaled.shape[1])
        axis_idx = region_shape[0] // 2
        keep_side = random.choice(("top", "bottom"))

    markers, maroon_local = _choose_markers(region_shape, keep_side)
    correct_offset = random.choice((0, 1))
    wrong_offset = 1 - correct_offset
    correct_region = _compose_region(scaled, region_shape, axis_kind, axis_idx, correct_offset, markers)
    wrong_region = _compose_region(scaled, region_shape, axis_kind, axis_idx, wrong_offset, markers)
    expected = _crop_selected_half(correct_region, axis_kind, axis_idx, maroon_local)
    wrong_expected = _crop_selected_half(wrong_region, axis_kind, axis_idx, maroon_local)
    if np.array_equal(expected, wrong_expected):
        raise ValueError("wrong placement gives the same cropped result")

    kept_mask = _kept_mask(region_shape, axis_kind, axis_idx, keep_side)
    evidence = _choose_evidence(correct_region, wrong_region, kept_mask)

    input_region = np.zeros(region_shape, dtype=int)
    if axis_kind == "v":
        input_region[:, axis_idx] = RED
    else:
        input_region[axis_idx, :] = RED
    for (r, c), color in markers.items():
        input_region[r, c] = color
    for r, c, color in evidence:
        input_region[r, c] = color

    min_h = max(16, sprite.shape[0] + region_shape[0] + 4)
    min_w = max(16, sprite.shape[1] + region_shape[1] + 4)
    if min_h > 30 or min_w > 30:
        raise ValueError("constructed grid would exceed ARC bounds")
    grid_h = random.randint(min_h, 30)
    grid_w = random.randint(min_w, 30)

    sprite_positions = _corner_positions((grid_h, grid_w), sprite.shape)
    region_positions = _corner_positions((grid_h, grid_w), region_shape)
    random.shuffle(sprite_positions)
    random.shuffle(region_positions)

    placement = None
    for sr, sc in sprite_positions:
        sprite_box = (sr, sc, sr + sprite.shape[0] - 1, sc + sprite.shape[1] - 1)
        for rr, rc in region_positions:
            region_box = (rr, rc, rr + region_shape[0] - 1, rc + region_shape[1] - 1)
            if not _boxes_overlap(sprite_box, region_box, padding=2):
                placement = (sr, sc, rr, rc)
                break
        if placement is not None:
            break
    if placement is None:
        raise ValueError("failed to place sprite and target region apart")

    sr, sc, rr, rc = placement
    grid = np.zeros((grid_h, grid_w), dtype=int)
    grid[sr : sr + sprite.shape[0], sc : sc + sprite.shape[1]] = sprite
    grid[rr : rr + region_shape[0], rc : rc + region_shape[1]] = input_region

    solved, details = _solve_details(grid)
    if not np.array_equal(solved, expected):
        raise ValueError("solver output does not match constructed target")
    if details["chosen_offset"] != correct_offset:
        raise ValueError("solver selected the wrong scaled placement")
    if len(details["valid_offsets"]) != 1:
        raise ValueError("placement is not uniquely determined")
    if solved.shape != expected.shape:
        raise ValueError("unexpected crop shape")
    if np.count_nonzero(np.isin(solved, SPRITE_COLORS)) == 0:
        raise ValueError("cropped output lost all sprite content")
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        try:
            return _build_case()
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid input after many attempts")
