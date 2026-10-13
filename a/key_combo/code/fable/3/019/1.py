from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, components_mask, find_components, paste


@dataclass(frozen=True)
class ShieldSpec:
    center_r: int
    center_c: int
    radius: int
    blocker_mode: str = "axis"


BLUE_ANCHOR = np.full((2, 2), Color.BLUE, dtype=int)
RADIUS_CHOICES = (1, 2, 3)
MAX_ATTEMPTS = 500


def _chebyshev(r0: int, c0: int, r1: int, c1: int) -> int:
    return max(abs(int(r0) - int(r1)), abs(int(c0) - int(c1)))


def _square_mask(shape: tuple[int, int], center_r: int, center_c: int, radius: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0 = max(0, int(center_r) - int(radius))
    r1 = min(int(shape[0]) - 1, int(center_r) + int(radius))
    c0 = max(0, int(center_c) - int(radius))
    c1 = min(int(shape[1]) - 1, int(center_c) + int(radius))
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _union_mask(shape: tuple[int, int], specs: list[ShieldSpec]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for spec in specs:
        mask |= _square_mask(shape, spec.center_r, spec.center_c, spec.radius)
    return mask


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = _dilate4(mask)
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _component_roles(union_mask: np.ndarray, anchor_masks: list[np.ndarray]) -> list[tuple[np.ndarray, bool, bool]]:
    roles = []
    for coords in components_mask(union_mask, connectivity=4):
        comp = np.zeros_like(union_mask, dtype=bool)
        comp[coords[:, 0], coords[:, 1]] = True
        roles.append((comp, bool(np.any(comp & anchor_masks[0])), bool(np.any(comp & anchor_masks[1]))))
    return roles


def _anchor_details(grid: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    comps = find_components(
        grid,
        bg=Color.BLACK,
        connectivity=4,
        monochromatic=True,
        colors=[Color.BLUE],
        return_format="dict",
    )
    comps = sorted(comps, key=lambda item: item["bbox"])
    assert len(comps) == 2, "expected exactly two blue anchor blocks"

    masks: list[np.ndarray] = []
    for comp in comps:
        coords = comp["coords"]
        r0, c0, r1, c1 = comp["bbox"]
        assert comp["area"] == 4, "each anchor must be 2x2"
        assert (r1 - r0 + 1, c1 - c0 + 1) == (2, 2), "each anchor must be 2x2"
        assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == Color.BLUE), "anchors must be solid blue 2x2 blocks"
        mask = np.zeros_like(grid, dtype=bool)
        mask[coords[:, 0], coords[:, 1]] = True
        masks.append(mask)
    return masks, grid == Color.BLUE


def _solve_details(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    anchor_masks, blue_mask = _anchor_details(g)
    marker_coords = [tuple(map(int, rc)) for rc in np.argwhere(g == Color.YELLOW)]
    blocker_coords = np.argwhere(g == Color.GREY)
    assert marker_coords, "expected at least one yellow marker"
    assert blocker_coords.size > 0, "expected at least one grey blocker"

    shield_specs: list[ShieldSpec] = []
    shield_mask = np.zeros(g.shape, dtype=bool)
    marker_radii: dict[tuple[int, int], int] = {}
    for r, c in sorted(marker_coords):
        distances = np.max(np.abs(blocker_coords - np.array([r, c], dtype=int)), axis=1)
        nearest = int(distances.min())
        radius = nearest - 1
        assert radius >= 0, "nearest grey blocker must be at least one cell away"
        shield_specs.append(ShieldSpec(r, c, radius))
        shield_mask |= _square_mask(g.shape, r, c, radius)
        marker_radii[(r, c)] = radius

    union_mask = shield_mask | blue_mask
    selected_mask = None
    for comp, has_left, has_right in _component_roles(union_mask, anchor_masks):
        if has_left and has_right:
            assert selected_mask is None, "at most one component can contain both anchors"
            selected_mask = comp

    return {
        "grid": g,
        "anchor_masks": anchor_masks,
        "blue_mask": blue_mask,
        "marker_radii": marker_radii,
        "shield_specs": shield_specs,
        "shield_mask": shield_mask,
        "selected_mask": selected_mask,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer each yellow marker's hidden square shield from its nearest grey blocker, then paint only the shield cells that belong to the chain component joining both blue anchors.

    Input structure:
    - The grid has a black background with exactly two solid 2 by 2 blue anchor blocks.
    - Several yellow marker pixels indicate the centers of hidden axis-aligned square shields.
    - Several grey blocker pixels are the only clues for shield size; no shield area is drawn in the input.

    Rule:
    - For each yellow marker, compute the Chebyshev distance to the nearest grey pixel and set the shield radius to that distance minus one.
    - Form the union of every inferred filled square shield together with both blue anchors, then split that union into 4-connected components.
    - If one component contains both blue anchors, paint every black shield cell in that component red while keeping the original yellow, grey, and blue pixels unchanged; otherwise paint no red cells.

    Key insight:
    - Whether a shield is painted depends on a global 4-connected chain through inferred hidden squares, not on the marker by itself.
    """
    details = _solve_details(grid)
    out = details["grid"].copy()
    selected_mask = details["selected_mask"]
    if selected_mask is None:
        return out

    paint_mask = selected_mask & details["shield_mask"] & (details["grid"] == Color.BLACK)
    out[paint_mask] = Color.RED
    return out


def _mask_from_bbox(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = bbox
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _sample_steps(total: int, maxima: list[int]) -> list[int] | None:
    if total < len(maxima) or total > sum(maxima):
        return None
    steps: list[int] = []
    remaining = total
    for i, upper in enumerate(maxima):
        rem_max = sum(maxima[i + 1 :])
        rem_min = len(maxima) - i - 1
        low = max(1, remaining - rem_max)
        high = min(int(upper), remaining - rem_min)
        if low > high:
            return None
        step = random.randint(low, high)
        steps.append(step)
        remaining -= step
    if remaining != 0:
        return None
    return steps


def _sample_linear_chain(
    shape: tuple[int, int],
    left_bbox: tuple[int, int, int, int],
    right_bbox: tuple[int, int, int, int],
    *,
    min_len: int,
    max_len: int,
) -> list[ShieldSpec] | None:
    height, width = shape
    for _ in range(200):
        count = random.randint(min_len, max_len)
        radii = [random.choice(RADIUS_CHOICES) for _ in range(count)]
        first_c = left_bbox[3] + radii[0] + 1
        last_c = right_bbox[1] - radii[-1] - 1
        span = last_c - first_c
        maxima = [radii[i] + radii[i + 1] + 1 for i in range(count - 1)]
        steps = _sample_steps(span, maxima)
        if steps is None:
            continue

        max_radius = max(radii)
        row_low = max(left_bbox[0] - radii[0], right_bbox[0] - radii[-1], max_radius + 1)
        row_high = min(left_bbox[2] + radii[0], right_bbox[2] + radii[-1], height - max_radius - 2)
        if row_low > row_high:
            continue
        row = random.randint(row_low, row_high)

        cols = [first_c]
        for step in steps:
            cols.append(cols[-1] + step)

        specs = [ShieldSpec(row, int(col), int(radius)) for col, radius in zip(cols, radii)]
        if all(
            0 <= spec.center_r - spec.radius
            and spec.center_r + spec.radius < height
            and 0 <= spec.center_c - spec.radius
            and spec.center_c + spec.radius < width
            for spec in specs
        ):
            return specs
    return None


def _sample_core_specs(
    shape: tuple[int, int],
    left_bbox: tuple[int, int, int, int],
    right_bbox: tuple[int, int, int, int],
    expect_connected: bool,
    anchor_masks: list[np.ndarray],
    blue_mask: np.ndarray,
) -> list[ShieldSpec] | None:
    for _ in range(250):
        if expect_connected:
            chain = _sample_linear_chain(shape, left_bbox, right_bbox, min_len=3, max_len=5)
            if chain is None:
                continue
            union_mask = _union_mask(shape, chain) | blue_mask
            roles = _component_roles(union_mask, anchor_masks)
            if sum(has_left and has_right for _, has_left, has_right in roles) == 1:
                return chain
            continue

        chain = _sample_linear_chain(shape, left_bbox, right_bbox, min_len=4, max_len=6)
        if chain is None:
            continue

        drop_choices = []
        for idx in range(1, len(chain) - 1):
            left = chain[idx - 1]
            right = chain[idx + 1]
            direct_gap = right.center_c - left.center_c
            direct_limit = left.radius + right.radius + 1
            if direct_gap > direct_limit:
                drop_choices.append(idx)
        if not drop_choices:
            continue

        drop_idx = random.choice(drop_choices)
        reduced = [spec for i, spec in enumerate(chain) if i != drop_idx]
        union_mask = _union_mask(shape, reduced) | blue_mask
        roles = _component_roles(union_mask, anchor_masks)
        if any(has_left and has_right for _, has_left, has_right in roles):
            continue
        if not any(has_left and not has_right and np.any(comp & _union_mask(shape, reduced)) for comp, has_left, has_right in roles):
            continue
        if not any(has_right and not has_left and np.any(comp & _union_mask(shape, reduced)) for comp, has_left, has_right in roles):
            continue
        return reduced
    return None


def _diagonal_contact(candidate_mask: np.ndarray, existing_mask: np.ndarray) -> bool:
    return bool(np.any(candidate_mask & (_dilate8(existing_mask) & ~_dilate4(existing_mask))))


def _sample_distractors(
    shape: tuple[int, int],
    existing_mask: np.ndarray,
    corridor_row: int,
    count: int,
) -> list[ShieldSpec] | None:
    height, width = shape
    specs: list[ShieldSpec] = []
    current_mask = existing_mask.copy()
    want_diag_touch = random.random() < 0.5

    for idx in range(count):
        placed = None
        for attempt in range(120):
            radius = random.choice(RADIUS_CHOICES)
            row_choices = [
                row
                for row in range(radius + 1, height - radius - 1)
                if abs(row - corridor_row) >= radius + 3
            ]
            if not row_choices:
                return None

            if idx == 0 and want_diag_touch and attempt < 40:
                row_pool = row_choices
            else:
                row_pool = row_choices

            row = random.choice(row_pool)
            col = random.randint(radius + 1, width - radius - 2)
            candidate = ShieldSpec(row, col, radius)
            candidate_mask = _square_mask(shape, row, col, radius)

            if np.any(candidate_mask & _dilate4(current_mask)):
                continue
            if idx == 0 and want_diag_touch and attempt < 40 and not _diagonal_contact(candidate_mask, current_mask):
                continue

            placed = candidate
            specs.append(candidate)
            current_mask |= candidate_mask
            break
        if placed is None:
            return None
    return specs


def _apply_blocker_modes(specs: list[ShieldSpec]) -> list[ShieldSpec]:
    modes = [random.choice(("axis", "diag")) for _ in specs]
    indices = list(range(len(specs)))
    random.shuffle(indices)
    modes[indices[0]] = "axis"
    modes[indices[1 % len(indices)]] = "diag"
    if len(indices) > 1 and modes[indices[0]] == modes[indices[1]]:
        modes[indices[1]] = "diag" if modes[indices[0]] == "axis" else "axis"
    return [
        ShieldSpec(spec.center_r, spec.center_c, spec.radius, mode)
        for spec, mode in zip(specs, modes)
    ]


def _blocker_candidates(spec: ShieldSpec, shape: tuple[int, int]) -> list[tuple[int, int]]:
    height, width = shape
    distance = spec.radius + 1
    axis: list[tuple[int, int]] = []
    diag: list[tuple[int, int]] = []
    other: list[tuple[int, int]] = []

    for dr in range(-distance, distance + 1):
        for dc in range(-distance, distance + 1):
            if max(abs(dr), abs(dc)) != distance:
                continue
            rr = spec.center_r + dr
            cc = spec.center_c + dc
            if not (0 <= rr < height and 0 <= cc < width):
                continue
            if (dr == 0) ^ (dc == 0):
                axis.append((rr, cc))
            elif abs(dr) == distance and abs(dc) == distance:
                diag.append((rr, cc))
            else:
                other.append((rr, cc))

    random.shuffle(axis)
    random.shuffle(diag)
    random.shuffle(other)
    if spec.blocker_mode == "diag":
        groups = [diag, other, axis]
    else:
        groups = [axis, other, diag]

    ordered: list[tuple[int, int]] = []
    for group in groups:
        ordered.extend(group)
    return ordered


def _assign_blockers(
    shape: tuple[int, int],
    specs: list[ShieldSpec],
    forbidden_mask: np.ndarray,
) -> list[tuple[int, int]] | None:
    order = list(range(len(specs)))
    random.shuffle(order)
    order.sort(key=lambda idx: specs[idx].radius, reverse=True)

    chosen: list[tuple[int, int] | None] = [None] * len(specs)
    used: set[tuple[int, int]] = set()
    for idx in order:
        spec = specs[idx]
        placed = None
        for rr, cc in _blocker_candidates(spec, shape):
            if forbidden_mask[rr, cc] or (rr, cc) in used:
                continue
            if any(_chebyshev(other.center_r, other.center_c, rr, cc) < other.radius + 1 for other in specs):
                continue
            placed = (rr, cc)
            break
        if placed is None:
            return None
        chosen[idx] = placed
        used.add(placed)

    blockers = [cell for cell in chosen if cell is not None]
    if len(blockers) != len(specs):
        return None
    for spec, blocker in zip(specs, blockers):
        if _chebyshev(spec.center_r, spec.center_c, blocker[0], blocker[1]) != spec.radius + 1:
            return None
    return blockers


def _render_input(
    shape: tuple[int, int],
    left_bbox: tuple[int, int, int, int],
    right_bbox: tuple[int, int, int, int],
    specs: list[ShieldSpec],
    blockers: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    paste(grid, BLUE_ANCHOR, left_bbox[0], left_bbox[1])
    paste(grid, BLUE_ANCHOR, right_bbox[0], right_bbox[1])
    for spec in specs:
        grid[spec.center_r, spec.center_c] = Color.YELLOW
    for rr, cc in blockers:
        grid[rr, cc] = Color.GREY
    return grid


def _validate_generated(
    grid: np.ndarray,
    specs: list[ShieldSpec],
    blockers: list[tuple[int, int]],
    expect_connected: bool,
) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if set(map(int, np.unique(grid))) - {Color.BLACK, Color.BLUE, Color.YELLOW, Color.GREY}:
        return False

    try:
        details = _solve_details(grid)
    except AssertionError:
        return False

    intended_radii = {(spec.center_r, spec.center_c): spec.radius for spec in specs}
    if details["marker_radii"] != intended_radii:
        return False

    expected_shield_mask = _union_mask(grid.shape, specs)
    if not np.array_equal(details["shield_mask"], expected_shield_mask):
        return False

    roles = _component_roles(details["shield_mask"] | details["blue_mask"], details["anchor_masks"])
    distractor_roles = [
        (comp, has_left, has_right)
        for comp, has_left, has_right in roles
        if np.any(comp & details["shield_mask"]) and not (has_left and has_right)
    ]
    if not distractor_roles:
        return False

    out = main(grid)
    red_count = int(np.count_nonzero(out == Color.RED))
    if expect_connected:
        if details["selected_mask"] is None:
            return False
        if sum(has_left and has_right for _, has_left, has_right in roles) != 1:
            return False
        selected_markers = sum(bool(details["selected_mask"][spec.center_r, spec.center_c]) for spec in specs)
        if selected_markers < 2 or selected_markers >= len(specs):
            return False
        if red_count == 0:
            return False
    else:
        if details["selected_mask"] is not None:
            return False
        if any(has_left and has_right for _, has_left, has_right in roles):
            return False
        if not any(has_left and not has_right and np.any(comp & details["shield_mask"]) for comp, has_left, has_right in roles):
            return False
        if not any(has_right and not has_left and np.any(comp & details["shield_mask"]) for comp, has_left, has_right in roles):
            return False
        if red_count != 0:
            return False

    own_offsets = [(br - spec.center_r, bc - spec.center_c) for spec, (br, bc) in zip(specs, blockers)]
    has_axis = any((dr == 0) ^ (dc == 0) for dr, dc in own_offsets)
    has_diag = any(dr != 0 and abs(dr) == abs(dc) for dr, dc in own_offsets)
    if not (has_axis and has_diag):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(18, 24)
    width = random.randint(20, 28)
    shape = (height, width)

    left_r = random.randint(2, height - 4)
    right_r = min(max(left_r + random.choice((-2, -1, 0, 1, 2)), 2), height - 4)
    left_c = random.randint(1, 3)
    right_c = random.randint(width - 5, width - 3)
    left_bbox = (left_r, left_c, left_r + 1, left_c + 1)
    right_bbox = (right_r, right_c, right_r + 1, right_c + 1)

    anchor_masks = [
        _mask_from_bbox(shape, left_bbox),
        _mask_from_bbox(shape, right_bbox),
    ]
    blue_mask = anchor_masks[0] | anchor_masks[1]

    expect_connected = random.random() < 0.6
    core_specs = _sample_core_specs(shape, left_bbox, right_bbox, expect_connected, anchor_masks, blue_mask)
    if core_specs is None:
        return None

    distractor_count = random.randint(1, 3)
    distractors = _sample_distractors(
        shape,
        blue_mask | _union_mask(shape, core_specs),
        core_specs[0].center_r,
        distractor_count,
    )
    if distractors is None:
        return None

    specs = _apply_blocker_modes(core_specs + distractors)
    forbidden_mask = blue_mask | _union_mask(shape, specs)
    blockers = _assign_blockers(shape, specs, forbidden_mask)
    if blockers is None:
        return None

    grid = _render_input(shape, left_bbox, right_bbox, specs, blockers)
    if not _validate_generated(grid, specs, blockers, expect_connected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid puzzle after {MAX_ATTEMPTS} attempts")
