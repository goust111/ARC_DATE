from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import components_mask, find_components, get_bg


GEN_MAX_ATTEMPTS = 600
GEN_HALF_H_RANGE = (10, 15)
GEN_HALF_W_RANGE = (8, 12)
GEN_CONTAINER_BODY_COUNT_RANGE = (1, 2)
GEN_SOURCE_BODY_COUNT = 2

GEN_BODY_SHAPES = [
    np.array([(0, 0), (0, 1), (1, 0)], dtype=int),
    np.array([(0, 0), (1, 0), (1, 1)], dtype=int),
    np.array([(0, 0), (0, 1), (0, 2)], dtype=int),
    np.array([(0, 1), (1, 0), (1, 1)], dtype=int),
    np.array([(0, 0), (0, 1), (1, 1), (1, 2)], dtype=int),
    np.array([(0, 0), (0, 1), (0, 2), (1, 1)], dtype=int),
    np.array([(0, 0), (1, 0), (2, 0), (2, 1)], dtype=int),
    np.array([(0, 1), (1, 1), (2, 1), (2, 0)], dtype=int),
    np.array([(0, 0), (0, 1), (1, 0), (1, 1)], dtype=int),
]

GEN_OVERHANG_SHAPES = [
    np.array([(0, 0), (0, 1), (0, 2), (1, 1)], dtype=int),
    np.array([(0, 0), (1, 0), (2, 0), (2, 1)], dtype=int),
    np.array([(0, 1), (1, 1), (2, 1), (2, 0)], dtype=int),
]


def _copy_bodies(bodies: list[dict]) -> list[dict]:
    return [{"color": int(body["color"]), "coords": body["coords"].copy()} for body in bodies]


def _find_separator(grid: np.ndarray) -> tuple[int, int]:
    height, width = map(int, grid.shape)
    assert width % 2 == 1, "one separator column requires an odd grid width"
    sep_col = width // 2
    sep_values = grid[:, sep_col]
    assert np.all(sep_values == sep_values[0]), "separator column must be solid"
    return int(sep_col), int(sep_values[0])


def _find_marker_color(grid: np.ndarray, bg: int, sep_col: int, sep_color: int) -> tuple[int, bool]:
    candidates: list[int] = []
    for color in map(int, np.unique(grid)):
        if color in (bg, sep_color):
            continue
        coords = np.argwhere(grid == color)
        if coords.shape[0] != 2:
            continue
        if not (np.all(coords[:, 1] < sep_col) or np.all(coords[:, 1] > sep_col)):
            continue
        comps = components_mask(grid == color, connectivity=4)
        if len(comps) == 2 and all(comp.shape[0] == 1 for comp in comps):
            candidates.append(color)

    assert len(candidates) == 1, f"expected one marker color, found {candidates}"
    marker_color = int(candidates[0])
    marker_coords = np.argwhere(grid == marker_color)
    container_is_left = bool(np.all(marker_coords[:, 1] < sep_col))
    return marker_color, container_is_left


def _parse_half(half: np.ndarray, bg: int, marker_color: int) -> tuple[int, np.ndarray, list[dict]]:
    colors = [int(color) for color in np.unique(half) if int(color) != bg]
    bottom_colors = [color for color in colors if color != marker_color and np.any(half[-1, :] == color)]
    assert len(bottom_colors) == 1, f"expected one bottom-touching obstacle color, found {bottom_colors}"
    obstacle_color = int(bottom_colors[0])
    obstacle_mask = half == obstacle_color

    body_colors = [color for color in colors if color not in (obstacle_color, marker_color)]
    bodies = [
        {"color": int(comp["color"]), "coords": comp["coords"].copy()}
        for comp in find_components(
            half,
            bg=bg,
            connectivity=4,
            monochromatic=True,
            colors=body_colors,
        )
    ]
    return obstacle_color, obstacle_mask, bodies


def _body_can_move(body_coords: np.ndarray, other_mask: np.ndarray, obstacle_mask: np.ndarray, height: int) -> bool:
    shifted = body_coords + np.array([1, 0], dtype=int)
    if int(shifted[:, 0].max()) >= int(height):
        return False
    if np.any(obstacle_mask[shifted[:, 0], shifted[:, 1]]):
        return False
    if np.any(other_mask[shifted[:, 0], shifted[:, 1]]):
        return False
    return True


def _simulate_rigid_fall(
    height: int,
    width: int,
    obstacle_mask: np.ndarray,
    bodies: list[dict],
    *,
    reverse_order: bool = False,
) -> list[dict]:
    cur = _copy_bodies(bodies)
    while True:
        moved_any = False
        indices = range(len(cur) - 1, -1, -1) if reverse_order else range(len(cur))
        for index in indices:
            other_mask = np.zeros((height, width), dtype=bool)
            for other_index, other in enumerate(cur):
                if other_index == index:
                    continue
                other_mask[other["coords"][:, 0], other["coords"][:, 1]] = True
            if _body_can_move(cur[index]["coords"], other_mask, obstacle_mask, height):
                cur[index]["coords"] = cur[index]["coords"] + np.array([1, 0], dtype=int)
                moved_any = True
        if not moved_any:
            return cur


def _simulate_sync_fall(height: int, width: int, obstacle_mask: np.ndarray, bodies: list[dict]) -> list[dict]:
    cur = _copy_bodies(bodies)
    while True:
        occupied = np.zeros((height, width), dtype=bool)
        for body in cur:
            occupied[body["coords"][:, 0], body["coords"][:, 1]] = True

        movable: list[int] = []
        for index, body in enumerate(cur):
            other_mask = occupied.copy()
            other_mask[body["coords"][:, 0], body["coords"][:, 1]] = False
            if _body_can_move(body["coords"], other_mask, obstacle_mask, height):
                movable.append(index)

        if not movable:
            return cur

        for index in movable:
            cur[index]["coords"] = cur[index]["coords"] + np.array([1, 0], dtype=int)


def _render_half(height: int, width: int, bg: int, obstacle_color: int, obstacle_mask: np.ndarray, bodies: list[dict]) -> np.ndarray:
    out = np.full((height, width), int(bg), dtype=int)
    out[obstacle_mask] = int(obstacle_color)
    for body in bodies:
        out[body["coords"][:, 0], body["coords"][:, 1]] = int(body["color"])
    return out


def _drop_pixels_independently(
    height: int,
    width: int,
    bg: int,
    obstacle_color: int,
    obstacle_mask: np.ndarray,
    bodies: list[dict],
) -> np.ndarray:
    start = _render_half(height, width, bg, obstacle_color, obstacle_mask, bodies)
    out = np.full((height, width), int(bg), dtype=int)
    out[obstacle_mask] = int(obstacle_color)

    for c in range(width):
        obstacle_rows = np.flatnonzero(obstacle_mask[:, c]).tolist()
        bounds = [-1] + obstacle_rows + [height]
        for top, bottom in zip(bounds[:-1], bounds[1:]):
            free_rows = list(range(top + 1, bottom))
            if not free_rows:
                continue
            falling_colors = [
                int(start[r, c])
                for r in free_rows
                if int(start[r, c]) not in (bg, obstacle_color)
            ]
            for r, color in zip(reversed(free_rows), reversed(falling_colors)):
                out[r, c] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Transfer the unmarked half's cargo into the marked half, then settle all rigid bodies under gravity.

    Input structure:
    - A single solid separator column splits the grid into equal left and right halves.
    - Each half contains one fixed obstacle color touching the bottom edge and several 4-connected movable monochrome rigid bodies.
    - Exactly two singleton marker pixels of one unique color appear in the same half and do not belong to any body.

    Rule:
    - The half containing the markers is the container: keep its obstacle and its movable bodies, then erase the marker pixels back to background.
    - Copy every movable body from the other half into the container at the same local row/column coordinates; discard the other half's obstacle entirely.
    - Inside the container, repeatedly move any rigid body down by one row whenever its whole translated shape avoids the bottom edge, the fixed obstacle, and the other settled bodies, until no body can move.
    - Return only the final container half.

    Key insight:
    - The separator does not define a symmetric merge: the markers choose the only physical stage, the opposite half contributes cargo coordinates but not terrain, and gravity acts on whole rigid bodies rather than on independent columns.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    sep_col, sep_color = _find_separator(g)
    marker_color, container_is_left = _find_marker_color(g, bg, sep_col, sep_color)

    left = g[:, :sep_col].copy()
    right = g[:, sep_col + 1 :].copy()
    left_data = _parse_half(left, bg, marker_color)
    right_data = _parse_half(right, bg, marker_color)

    if container_is_left:
        obstacle_color, obstacle_mask, container_bodies = left_data
        source_bodies = right_data[2]
    else:
        obstacle_color, obstacle_mask, container_bodies = right_data
        source_bodies = left_data[2]

    height, width = map(int, obstacle_mask.shape)
    all_bodies = _copy_bodies(container_bodies) + _copy_bodies(source_bodies)
    settled = _simulate_rigid_fall(height, width, obstacle_mask, all_bodies)
    return _render_half(height, width, bg, obstacle_color, obstacle_mask, settled)


def _build_obstacle_mask(profile: list[int], height: int) -> np.ndarray:
    mask = np.zeros((height, len(profile)), dtype=bool)
    for c, col_height in enumerate(profile):
        if col_height > 0:
            mask[height - col_height :, c] = True
    return mask


def _generate_container_profile(height: int, width: int, rng: random.Random) -> list[int]:
    for _ in range(200):
        span = rng.randint(max(4, width // 2), width)
        c0 = rng.randint(0, width - span)
        profile = [0] * width
        for c in range(c0, c0 + span):
            profile[c] = 1

        for _ in range(rng.randint(1, 3)):
            a = rng.randint(c0, c0 + span - 1)
            b = rng.randint(a + 1, c0 + span)
            extra = rng.randint(1, min(3, height - 4))
            for c in range(a, b):
                profile[c] = min(height - 3, profile[c] + extra)

        occupied_heights = [value for value in profile if value > 0]
        if len(set(occupied_heights)) >= 2:
            return profile
    raise RuntimeError("failed to generate a stepped obstacle profile")


def _try_place_shape(
    shape: np.ndarray,
    height: int,
    width: int,
    reserve: np.ndarray,
    rng: random.Random,
    *,
    prefer_profile_variation: bool = False,
    profile: list[int] | None = None,
) -> np.ndarray | None:
    local = shape.copy()
    shape_h = int(local[:, 0].max()) + 1
    shape_w = int(local[:, 1].max()) + 1
    placements: list[tuple[int, int]] = []

    for r0 in range(0, height - shape_h):
        for c0 in range(0, width - shape_w + 1):
            placed = local + np.array([r0, c0], dtype=int)
            if np.any(reserve[placed[:, 0], placed[:, 1]]):
                continue
            if prefer_profile_variation and profile is not None:
                support_heights = {profile[int(c)] for c in np.unique(placed[:, 1])}
                if len(support_heights) < 2:
                    continue
            placements.append((r0, c0))

    if not placements:
        return None
    rng.shuffle(placements)
    r0, c0 = placements[0]
    return local + np.array([r0, c0], dtype=int)


def _choose_marker_cells(forbidden: np.ndarray, rng: random.Random) -> tuple[tuple[int, int], tuple[int, int]] | None:
    height, width = map(int, forbidden.shape)
    candidates: list[tuple[int, int]] = []
    for r in range(height - 1):
        for c in range(width):
            if forbidden[r, c]:
                continue
            if any(
                0 <= nr < height and 0 <= nc < width and forbidden[nr, nc]
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
            ):
                continue
            candidates.append((r, c))

    rng.shuffle(candidates)
    for index, first in enumerate(candidates):
        for second in candidates[index + 1 :]:
            if abs(first[0] - second[0]) + abs(first[1] - second[1]) <= 1:
                continue
            return first, second
    return None


def _make_source_obstacle_mask(
    height: int,
    width: int,
    reserve: np.ndarray,
    special_start: np.ndarray,
    special_end: np.ndarray,
    container_obstacle_mask: np.ndarray,
    combined_bodies: list[dict],
    bg: int,
    obstacle_color: int,
    expected: np.ndarray,
    rng: random.Random,
) -> np.ndarray | None:
    cols = sorted(set(int(c) for c in special_start[:, 1]))
    candidates: list[np.ndarray] = []
    for span_w in (1, 2):
        for j in range(len(cols) - span_w + 1):
            span = cols[j : j + span_w]
            min_body_bottom = min(int(np.max(special_start[special_start[:, 1] == c, 0])) for c in span)
            top_lo = min_body_bottom + 1
            top_hi = min(int(np.min(special_end[special_end[:, 1] == c, 0])) for c in span)
            for top in range(top_lo, top_hi + 1):
                mask = np.zeros((height, width), dtype=bool)
                for c in span:
                    mask[top:, c] = True
                if np.any(mask & reserve):
                    continue
                candidates.append(mask)

    rng.shuffle(candidates)
    for mask in candidates:
        alt_bodies = _simulate_rigid_fall(height, width, container_obstacle_mask | mask, combined_bodies)
        alt = _render_half(height, width, bg, obstacle_color, container_obstacle_mask | mask, alt_bodies)
        if not np.array_equal(alt, expected):
            return mask
    return None


def _validate_background_dominance(grid: np.ndarray, bg: int) -> bool:
    counts = Counter(map(int, grid.reshape(-1)))
    top = max(counts.values())
    return counts[bg] == top and sum(count == top for count in counts.values()) == 1


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(*GEN_HALF_H_RANGE)
    width = rng.randint(*GEN_HALF_W_RANGE)

    palette = list(range(10))
    rng.shuffle(palette)
    bg, separator_color, marker_color, container_obstacle_color, source_obstacle_color = palette[:5]

    container_body_count = rng.randint(*GEN_CONTAINER_BODY_COUNT_RANGE)
    total_bodies = container_body_count + GEN_SOURCE_BODY_COUNT
    body_colors = palette[5 : 5 + total_bodies]
    if len(body_colors) != total_bodies:
        return None

    container_profile = _generate_container_profile(height, width, rng)
    container_obstacle_mask = _build_obstacle_mask(container_profile, height)

    container_bodies: list[dict] = []
    reserve_container = container_obstacle_mask.copy()
    for index in range(container_body_count):
        placed = _try_place_shape(
            rng.choice(GEN_BODY_SHAPES),
            height,
            width,
            reserve_container,
            rng,
        )
        if placed is None:
            return None
        container_bodies.append({"color": int(body_colors[index]), "coords": placed})
        reserve_container[placed[:, 0], placed[:, 1]] = True

    source_bodies: list[dict] = []
    reserve_source = reserve_container.copy()
    special_shape = rng.choice(GEN_OVERHANG_SHAPES)
    special_coords = _try_place_shape(
        special_shape,
        height,
        width,
        reserve_source,
        rng,
        prefer_profile_variation=True,
        profile=container_profile,
    )
    if special_coords is None:
        return None
    source_bodies.append({"color": int(body_colors[container_body_count]), "coords": special_coords})
    reserve_source[special_coords[:, 0], special_coords[:, 1]] = True

    for index in range(1, GEN_SOURCE_BODY_COUNT):
        placed = _try_place_shape(
            rng.choice(GEN_BODY_SHAPES),
            height,
            width,
            reserve_source,
            rng,
        )
        if placed is None:
            return None
        color = int(body_colors[container_body_count + index])
        source_bodies.append({"color": color, "coords": placed})
        reserve_source[placed[:, 0], placed[:, 1]] = True

    combined_bodies = _copy_bodies(container_bodies) + _copy_bodies(source_bodies)
    settled = _simulate_rigid_fall(height, width, container_obstacle_mask, combined_bodies)
    expected = _render_half(
        height,
        width,
        bg,
        container_obstacle_color,
        container_obstacle_mask,
        settled,
    )

    reversed_out = _render_half(
        height,
        width,
        bg,
        container_obstacle_color,
        container_obstacle_mask,
        _simulate_rigid_fall(height, width, container_obstacle_mask, combined_bodies, reverse_order=True),
    )
    if not np.array_equal(expected, reversed_out):
        return None

    sync_out = _render_half(
        height,
        width,
        bg,
        container_obstacle_color,
        container_obstacle_mask,
        _simulate_sync_fall(height, width, container_obstacle_mask, combined_bodies),
    )
    if not np.array_equal(expected, sync_out):
        return None

    initial_container = _render_half(
        height,
        width,
        bg,
        container_obstacle_color,
        container_obstacle_mask,
        combined_bodies,
    )
    if np.array_equal(initial_container, expected):
        return None

    pixelwise = _drop_pixels_independently(
        height,
        width,
        bg,
        container_obstacle_color,
        container_obstacle_mask,
        combined_bodies,
    )
    if np.array_equal(pixelwise, expected):
        return None

    special_index = container_body_count
    special_start = source_bodies[0]["coords"]
    special_end = settled[special_index]["coords"]
    if int(special_end[:, 0].min() - special_start[:, 0].min()) < 2:
        return None

    source_obstacle_mask = _make_source_obstacle_mask(
        height,
        width,
        reserve_source,
        special_start,
        special_end,
        container_obstacle_mask,
        combined_bodies,
        bg,
        container_obstacle_color,
        expected,
        rng,
    )
    if source_obstacle_mask is None:
        return None

    source_half = np.full((height, width), int(bg), dtype=int)
    source_half[source_obstacle_mask] = int(source_obstacle_color)
    for body in source_bodies:
        source_half[body["coords"][:, 0], body["coords"][:, 1]] = int(body["color"])

    container_half = np.full((height, width), int(bg), dtype=int)
    container_half[container_obstacle_mask] = int(container_obstacle_color)
    for body in container_bodies:
        container_half[body["coords"][:, 0], body["coords"][:, 1]] = int(body["color"])

    marker_forbidden = container_obstacle_mask.copy()
    for body in container_bodies + source_bodies:
        marker_forbidden[body["coords"][:, 0], body["coords"][:, 1]] = True
    marker_cells = _choose_marker_cells(marker_forbidden, rng)
    if marker_cells is None:
        return None
    for r, c in marker_cells:
        container_half[r, c] = int(marker_color)

    wrong_with_source_obstacle = _render_half(
        height,
        width,
        bg,
        container_obstacle_color,
        container_obstacle_mask | source_obstacle_mask,
        _simulate_rigid_fall(height, width, container_obstacle_mask | source_obstacle_mask, combined_bodies),
    )
    if np.array_equal(wrong_with_source_obstacle, expected):
        return None

    swapped_stage = _render_half(
        height,
        width,
        bg,
        source_obstacle_color,
        source_obstacle_mask,
        _simulate_rigid_fall(
            height,
            width,
            source_obstacle_mask,
            _copy_bodies(source_bodies) + _copy_bodies(container_bodies),
        ),
    )
    if np.array_equal(swapped_stage, expected):
        return None

    container_left = bool(rng.getrandbits(1))
    full = np.full((height, 2 * width + 1), int(bg), dtype=int)
    full[:, width] = int(separator_color)
    if container_left:
        full[:, :width] = container_half
        full[:, width + 1 :] = source_half
    else:
        full[:, :width] = source_half
        full[:, width + 1 :] = container_half

    if not _validate_background_dominance(full, bg):
        return None
    if not np.array_equal(main(full), expected):
        return None

    return full


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cargo-bay input")
