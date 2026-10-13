from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
PALETTE = tuple(range(1, 10))
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_GENERATION_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as (N, 2) arrays of (row, column)."""
    seen = np.zeros(mask.shape, dtype=bool)
    result: list[np.ndarray] = []
    rows, cols = mask.shape
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        queue = deque([(sr, sc)])
        cells: list[tuple[int, int]] = []
        while queue:
            r, c = queue.popleft()
            cells.append((r, c))
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        result.append(np.asarray(cells, dtype=int))
    return result


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _analyze(grid: np.ndarray) -> dict:
    """Identify the unique sprite and the two same-colored singleton controls."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and 1 <= g.shape[0] <= 30 and 1 <= g.shape[1] <= 30
    colors = [int(x) for x in np.unique(g) if int(x) != BACKGROUND]
    assert len(colors) == 2, "expected one sprite color and one control color"

    control_candidates = []
    for color in colors:
        comps = _components(g == color)
        if int(np.count_nonzero(g == color)) == 2 and len(comps) == 2 and all(len(comp) == 1 for comp in comps):
            control_candidates.append(color)
    assert len(control_candidates) == 1, "two isolated control pixels must be uniquely identifiable"

    control_color = control_candidates[0]
    sprite_color = next(color for color in colors if color != control_color)
    sprite_components = _components(g == sprite_color)
    assert len(sprite_components) == 1 and len(sprite_components[0]) >= 4, "expected one nontrivial 4-connected sprite"
    sprite = sprite_components[0]

    markers = np.argwhere(g == control_color).astype(int)
    (r_a, c_a), (r_b, c_b) = markers.tolist()
    same_row = r_a == r_b
    same_col = c_a == c_b
    assert same_row != same_col, "control pixels must share exactly one grid coordinate"

    r0, c0, r1, c1 = _bbox(sprite)
    if same_row:
        orientation = "vertical"
        axes = tuple(sorted((c_a, c_b)))
        assert axes[0] < c0 <= c1 < axes[1], "sprite must lie strictly between the vertical mirrors"
        distances = (c0 - axes[0], axes[1] - c1)
    else:
        orientation = "horizontal"
        axes = tuple(sorted((r_a, r_b)))
        assert axes[0] < r0 <= r1 < axes[1], "sprite must lie strictly between the horizontal mirrors"
        distances = (r0 - axes[0], axes[1] - r1)
    assert distances[0] != distances[1], "sprite must have a unique nearer mirror"

    near_index = 0 if distances[0] < distances[1] else 1
    return {
        "control_color": control_color,
        "sprite_color": sprite_color,
        "sprite": sprite,
        "markers": markers,
        "orientation": orientation,
        "axes": axes,
        "near_index": near_index,
        "distances": distances,
    }


def _reflected(coords: np.ndarray, orientation: str, axis: int) -> np.ndarray:
    copied = coords.copy()
    dim = 1 if orientation == "vertical" else 0
    copied[:, dim] = 2 * int(axis) - copied[:, dim]
    return copied


def _draw_visible(out: np.ndarray, coords: np.ndarray, color: int) -> int:
    rows, cols = out.shape
    inside = (
        (coords[:, 0] >= 0)
        & (coords[:, 0] < rows)
        & (coords[:, 1] >= 0)
        & (coords[:, 1] < cols)
    )
    visible = coords[inside]
    if len(visible):
        out[visible[:, 0], visible[:, 1]] = int(color)
    return int(len(visible))


def _solve_with_first_mirror(grid: np.ndarray, info: dict, first_index: int, max_copies: int | None = None) -> np.ndarray:
    """Render the alternating reflection trace, clipping each visible copy."""
    out = np.asarray(grid, dtype=int).copy()
    out[info["markers"][:, 0], info["markers"][:, 1]] = BACKGROUND
    current = info["sprite"].copy()
    mirror_index = int(first_index)
    made = 0
    for _ in range(100):
        if max_copies is not None and made >= max_copies:
            break
        current = _reflected(current, info["orientation"], info["axes"][mirror_index])
        visible = _draw_visible(out, current, info["sprite_color"])
        if visible == 0:
            break
        made += 1
        mirror_index = 1 - mirror_index
    else:
        raise AssertionError("reflection trace did not leave the grid")
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Extend an asymmetric monochrome sprite through the corridor defined by two point-supported mirrors.

    Input structure:
    - Black is the background. One nontrivial monochrome 4-connected asymmetric sprite is present.
    - Two isolated pixels of a second shared color lie on one row or one column; their columns or rows are the two mirror axes.
    - The complete sprite lies strictly between the axes, with unequal edge-to-axis distances.

    Rule:
    - Erase both control pixels and retain the original sprite.
    - Reflect the sprite first across its uniquely nearer axis, then reflect that copy across the other axis, alternating axes thereafter.
    - Draw the in-grid part of every reflected copy and stop when the next copy is wholly outside the grid.

    Key insight:
    - The singleton pair defines two fixed mirrors, not destinations: alternating reflections determine both the changing handedness and the non-periodic placement trace.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze(g)
    return _solve_with_first_mirror(g, info, info["near_index"])


def _is_asymmetric(mask: np.ndarray) -> bool:
    """Reject every horizontal, vertical, or diagonal reflection axis."""
    if np.array_equal(mask, np.flipud(mask)) or np.array_equal(mask, np.fliplr(mask)):
        return False
    if mask.shape[0] == mask.shape[1]:
        if np.array_equal(mask, mask.T) or np.array_equal(mask, np.flipud(np.fliplr(mask)).T):
            return False
    return True


def _random_sprite_mask(orientation: str) -> np.ndarray | None:
    """Grow a compact asymmetric polyomino whose mirror-facing extent is small."""
    move_extent = random.choices((2, 3), weights=(5, 1))[0]
    cross_extent = random.randint(3, 5)
    h, w = (cross_extent, move_extent) if orientation == "vertical" else (move_extent, cross_extent)
    target = random.randint(max(5, h + w - 1), min(h * w - 1, 9))

    cells = {(random.randrange(h), random.randrange(w))}
    while len(cells) < target:
        frontier = set()
        for r, c in cells:
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(tuple(frontier)))

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
        return None
    if len(_components(mask)) != 1 or not _is_asymmetric(mask):
        return None
    return mask


def _trace_from_metadata(
    shape: tuple[int, int],
    sprite_coords: np.ndarray,
    orientation: str,
    axes: tuple[int, int],
    first_index: int,
) -> list[tuple[np.ndarray, int]]:
    """Independent generator oracle: return all visible full-coordinate copies and visible counts."""
    rows, cols = shape
    dim = 1 if orientation == "vertical" else 0
    current = sprite_coords.copy()
    index = int(first_index)
    trace: list[tuple[np.ndarray, int]] = []
    for _ in range(100):
        nxt = current.copy()
        nxt[:, dim] = 2 * axes[index] - current[:, dim]
        inside = (
            (nxt[:, 0] >= 0)
            & (nxt[:, 0] < rows)
            & (nxt[:, 1] >= 0)
            & (nxt[:, 1] < cols)
        )
        count = int(inside.sum())
        if count == 0:
            return trace
        trace.append((nxt, count))
        current = nxt
        index = 1 - index
    raise AssertionError("metadata trace did not terminate")


def _metadata_oracle(
    grid: np.ndarray,
    sprite_coords: np.ndarray,
    sprite_color: int,
    marker_coords: np.ndarray,
    orientation: str,
    axes: tuple[int, int],
    first_index: int,
    limit: int | None = None,
) -> np.ndarray:
    """Render directly from generator metadata rather than solver object detection."""
    out = grid.copy()
    out[marker_coords[:, 0], marker_coords[:, 1]] = BACKGROUND
    trace = _trace_from_metadata(grid.shape, sprite_coords, orientation, axes, first_index)
    if limit is not None:
        trace = trace[:limit]
    for coords, _count in trace:
        _draw_visible(out, coords, sprite_color)
    return out


def _geometry_candidates(axis_length: int, sprite_extent: int) -> list[dict]:
    """Enumerate layouts where near-first and at least three iterations are visibly testable."""
    candidates: list[dict] = []
    for gap in range(sprite_extent + 3, min(sprite_extent + 7, axis_length - 3) + 1):
        for low_axis in range(1, axis_length - gap - 1):
            high_axis = low_axis + gap
            for start in range(low_axis + 1, high_axis - sprite_extent + 1):
                end = start + sprite_extent - 1
                distances = (start - low_axis, high_axis - end)
                if distances[0] == distances[1]:
                    continue
                near_index = 0 if distances[0] < distances[1] else 1

                base = np.arange(start, end + 1, dtype=int)[:, None]
                # Use a dummy second coordinate; only the movement dimension matters here.
                dummy = np.column_stack((np.zeros(len(base), dtype=int), base[:, 0]))
                trace = _trace_from_metadata((1, axis_length), dummy, "vertical", (low_axis, high_axis), near_index)
                far_trace = _trace_from_metadata((1, axis_length), dummy, "vertical", (low_axis, high_axis), 1 - near_index)
                if len(trace) < 3 or not far_trace:
                    continue
                visible_counts = [count for _coords, count in trace]
                candidates.append(
                    {
                        "axes": (low_axis, high_axis),
                        "start": start,
                        "near_index": near_index,
                        "distances": distances,
                        "partial": any(count < sprite_extent for count in visible_counts),
                        "copies": len(trace),
                    }
                )
    return candidates


def _place_case(orientation: str, mask: np.ndarray) -> tuple[np.ndarray, dict] | None:
    axis_length = random.randint(27, 30)
    cross_length = random.randint(max(mask.shape) + 4, 16)
    move_extent = mask.shape[1] if orientation == "vertical" else mask.shape[0]
    candidates = _geometry_candidates(axis_length, move_extent)
    if not candidates:
        return None

    want_partial = random.random() < 0.35
    preferred = [item for item in candidates if item["partial"] == want_partial]
    geometry = random.choice(preferred or candidates)

    if orientation == "vertical":
        shape = (cross_length, axis_length)
        cross_extent = mask.shape[0]
    else:
        shape = (axis_length, cross_length)
        cross_extent = mask.shape[1]

    cross_starts = list(range(1, cross_length - cross_extent))
    random.shuffle(cross_starts)
    cross_start = next(
        (
            value
            for value in cross_starts
            if any(
                point < value - 1 or point > value + cross_extent
                for point in range(cross_length)
            )
        ),
        None,
    )
    if cross_start is None:
        return None
    marker_positions = [
        point
        for point in range(cross_length)
        if point < cross_start - 1 or point > cross_start + cross_extent
    ]
    if not marker_positions:
        return None
    marker_cross = random.choice(marker_positions)

    sprite_color, control_color = random.sample(PALETTE, 2)
    grid = np.zeros(shape, dtype=int)
    rr, cc = np.argwhere(mask).T
    if orientation == "vertical":
        sprite_coords = np.column_stack((rr + cross_start, cc + geometry["start"])).astype(int)
        marker_coords = np.asarray(
            [(marker_cross, geometry["axes"][0]), (marker_cross, geometry["axes"][1])], dtype=int
        )
    else:
        sprite_coords = np.column_stack((rr + geometry["start"], cc + cross_start)).astype(int)
        marker_coords = np.asarray(
            [(geometry["axes"][0], marker_cross), (geometry["axes"][1], marker_cross)], dtype=int
        )

    grid[sprite_coords[:, 0], sprite_coords[:, 1]] = sprite_color
    grid[marker_coords[:, 0], marker_coords[:, 1]] = control_color
    meta = {
        "sprite_coords": sprite_coords,
        "sprite_color": sprite_color,
        "control_color": control_color,
        "marker_coords": marker_coords,
        "orientation": orientation,
        "axes": geometry["axes"],
        "near_index": geometry["near_index"],
        "distances": geometry["distances"],
        "partial": geometry["partial"],
        "copies": geometry["copies"],
        "mask": mask,
    }
    return grid, meta


def _semantic_accepts(grid: np.ndarray, meta: dict) -> bool:
    """Check uniqueness, rule dependencies, clipping, asymmetry, and common wrong shortcuts."""
    try:
        info = _analyze(grid)
    except AssertionError:
        return False

    if info["orientation"] != meta["orientation"] or info["axes"] != meta["axes"]:
        return False
    if info["near_index"] != meta["near_index"] or info["distances"] != meta["distances"]:
        return False
    if info["sprite_color"] != meta["sprite_color"] or info["control_color"] != meta["control_color"]:
        return False
    if not np.array_equal(np.asarray(sorted(map(tuple, info["sprite"]))), np.asarray(sorted(map(tuple, meta["sprite_coords"])))):
        return False
    if len(_components(meta["mask"])) != 1 or not _is_asymmetric(meta["mask"]):
        return False

    trace = _trace_from_metadata(
        grid.shape,
        meta["sprite_coords"],
        meta["orientation"],
        meta["axes"],
        meta["near_index"],
    )
    if len(trace) < 3 or len(trace) != meta["copies"]:
        return False
    if meta["partial"] != any(count < len(meta["sprite_coords"]) for _coords, count in trace):
        return False

    expected = _metadata_oracle(
        grid,
        meta["sprite_coords"],
        meta["sprite_color"],
        meta["marker_coords"],
        meta["orientation"],
        meta["axes"],
        meta["near_index"],
    )
    only_one = _metadata_oracle(
        grid,
        meta["sprite_coords"],
        meta["sprite_color"],
        meta["marker_coords"],
        meta["orientation"],
        meta["axes"],
        meta["near_index"],
        limit=1,
    )
    only_two = _metadata_oracle(
        grid,
        meta["sprite_coords"],
        meta["sprite_color"],
        meta["marker_coords"],
        meta["orientation"],
        meta["axes"],
        meta["near_index"],
        limit=2,
    )
    far_first = _metadata_oracle(
        grid,
        meta["sprite_coords"],
        meta["sprite_color"],
        meta["marker_coords"],
        meta["orientation"],
        meta["axes"],
        1 - meta["near_index"],
    )
    controls_kept = expected.copy()
    controls_kept[meta["marker_coords"][:, 0], meta["marker_coords"][:, 1]] = meta["control_color"]

    # Each of these inequalities excludes a concrete shortcut and proves a key step changes the result.
    if any(np.array_equal(expected, wrong) for wrong in (only_one, only_two, far_first, controls_kept)):
        return False
    try:
        actual = main(grid)
    except Exception:
        return False
    return bool(np.array_equal(actual, expected))


def generate_input() -> np.ndarray:
    """Generate varied horizontal/vertical mirror corridors with semantic counterexample checks."""
    for _ in range(MAX_GENERATION_ATTEMPTS):
        orientation = random.choice(("vertical", "horizontal"))
        mask = _random_sprite_mask(orientation)
        if mask is None:
            continue
        placed = _place_case(orientation, mask)
        if placed is None:
            continue
        grid, meta = placed
        if _semantic_accepts(grid, meta):
            return grid
    raise RuntimeError(f"failed to generate a valid mirror-corridor input after {MAX_GENERATION_ATTEMPTS} attempts")
