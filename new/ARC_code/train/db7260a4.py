from __future__ import annotations

import numpy as np

from arc_utils import colors_present, find_components, get_bg, paste


def _closed_u_containers(g: np.ndarray, wall_color: int, bg: int) -> list[tuple[int, int, int, int]]:
    h, w = g.shape
    containers = []

    for bottom in range(1, h):
        for left in range(w - 2):
            if int(g[bottom, left]) != wall_color:
                continue
            for right in range(left + 2, w):
                if int(g[bottom, right]) != wall_color:
                    continue
                if not np.all(g[bottom, left : right + 1] == wall_color):
                    continue

                top = bottom
                while top > 0 and int(g[top - 1, left]) == wall_color and int(g[top - 1, right]) == wall_color:
                    top -= 1
                if top == bottom:
                    continue

                interior = g[top:bottom, left + 1 : right]
                if interior.size == 0 or not np.all(interior == bg):
                    continue
                containers.append((top, bottom, left, right))

    return containers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop the singleton marker into the matching red U-shaped container, or make a bottom-row puddle if no closed container is hit.

    Input structure
    - The grid has a dominant background, one singleton marker pixel on the top row,
      and one other foreground color forming vertical bars and bottom segments.
    - A closed container is a U shape: two vertical side bars with a continuous
      bottom segment between them and an empty rectangular interior above the bottom.

    Rule
    - Remove the top marker from its original cell.
    - If exactly one closed U-shaped container has the marker column within its
      horizontal footprint, fill that container's empty interior with the marker color.
    - If no closed container is under the marker column, fill the whole bottom row
      with the marker color.

    Key insight
    - The marker does not identify a color by value; it identifies a vertical drop
      column, and only a structurally closed U shape can catch and contain the fill.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    foreground_colors = colors_present(g, exclude={bg})
    assert len(foreground_colors) == 2, "expected marker and wall colors"

    singleton_colors = [color for color in foreground_colors if int(np.count_nonzero(g == color)) == 1]
    assert len(singleton_colors) == 1, "expected exactly one singleton marker color"
    marker_color = singleton_colors[0]

    wall_colors = [color for color in foreground_colors if color != marker_color]
    assert len(wall_colors) == 1
    wall_color = wall_colors[0]

    marker_coords = np.argwhere(g == marker_color)
    assert marker_coords.shape == (1, 2)
    marker_r, marker_c = map(int, marker_coords[0])
    assert marker_r == 0, "marker must sit on the top row"

    containers = [
        container
        for container in _closed_u_containers(g, wall_color, bg)
        if container[0] > marker_r and container[2] <= marker_c <= container[3]
    ]
    assert len(containers) <= 1, "marker column must select at most one closed container"

    out = g.copy()
    out[marker_r, marker_c] = bg
    if containers:
        top, bottom, left, right = containers[0]
        out[top:bottom, left + 1 : right] = marker_color
    else:
        assert np.all(out[-1, :] == bg), "missed marker should fall to an empty bottom row"
        out[-1, :] = marker_color
    return out

# --------------- generator (o5x) ---------------

# Fixed ARC colors and canvas.
_GEN_BG = 0
_GEN_MARKER = 1
_GEN_WALL = 2
_GEN_H = 10
_GEN_W = 10

# Vertical layout bounds; row 9 stays empty in generated inputs.
_GEN_MARKER_ROW = 0
_GEN_MIN_WALL_ROW = 2
_GEN_LAST_WALL_ROW = 8

# Closed container interior size ranges, with exclusive upper bounds.
_GEN_INTERIOR_H_RANGE = (2, 4)
_GEN_INTERIOR_W_RANGE = (1, 6)

# Broken distractor patch size ranges, with exclusive upper bounds.
_GEN_DISTRACTOR_H_RANGE = (2, 4)
_GEN_DISTRACTOR_W_RANGE = (4, 7)

# Scenario mix and distractor count ranges, with exclusive upper bounds.
_GEN_FILL_CONTAINER_PROB = 0.67
_GEN_MISSED_HAS_U_PROB = 0.55
_GEN_DISTRACTORS_WITH_U_RANGE = (0, 2)
_GEN_DISTRACTORS_WITHOUT_U_RANGE = (1, 4)

# Rejection-sampling retry budget.
_GEN_MAX_ATTEMPTS = 300


def _gen_u_patch(interior_h: int, interior_w: int) -> np.ndarray:
    patch = np.full((interior_h + 1, interior_w + 2), _GEN_BG, dtype=int)
    patch[:, 0] = _GEN_WALL
    patch[:, -1] = _GEN_WALL
    patch[-1, :] = _GEN_WALL
    return patch


def _gen_sample_u() -> tuple[int, int, int, int, int, int]:
    interior_h = int(np.random.randint(*_GEN_INTERIOR_H_RANGE))
    interior_w = int(np.random.randint(*_GEN_INTERIOR_W_RANGE))
    top = int(np.random.randint(_GEN_MIN_WALL_ROW, _GEN_LAST_WALL_ROW - interior_h + 1))
    bottom = top + interior_h
    left = int(np.random.randint(0, _GEN_W - interior_w - 1))
    right = left + interior_w + 1
    return top, bottom, left, right, interior_h, interior_w


def _gen_broken_u_patch(height: int, width: int, *, gap: int | None = None, missing_side: str | None = None) -> np.ndarray:
    patch = np.full((height, width), _GEN_BG, dtype=int)
    if missing_side != "left":
        patch[:, 0] = _GEN_WALL
    if missing_side != "right":
        patch[:, -1] = _GEN_WALL
    patch[-1, :] = _GEN_WALL
    if gap is not None:
        patch[-1, gap] = _GEN_BG
    return patch


def _gen_distractor_patches() -> list[np.ndarray]:
    patches: list[np.ndarray] = []

    for h in range(*_GEN_DISTRACTOR_H_RANGE):
        for w in range(*_GEN_DISTRACTOR_W_RANGE):
            gap = int(np.random.randint(1, w - 1))
            patches.append(_gen_broken_u_patch(h, w, gap=gap))
            patches.append(_gen_broken_u_patch(h, w, missing_side="left"))
            patches.append(_gen_broken_u_patch(h, w, missing_side="right"))

    return patches


def _gen_valid_positions(grid: np.ndarray, patch: np.ndarray) -> list[tuple[int, int]]:
    ph, pw = patch.shape
    local = np.argwhere(patch == _GEN_WALL)
    local_set = {(int(r), int(c)) for r, c in local}
    touch_local = {
        (r + dr, c + dc)
        for r, c in local_set
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        if (r + dr, c + dc) not in local_set
    }
    positions: list[tuple[int, int]] = []

    for r0 in range(_GEN_MIN_WALL_ROW, _GEN_LAST_WALL_ROW - ph + 2):
        for c0 in range(0, _GEN_W - pw + 1):
            if np.any(grid[r0 + local[:, 0], c0 + local[:, 1]] != _GEN_BG):
                continue
            touches = any(
                0 <= r0 + r < _GEN_H
                and 0 <= c0 + c < _GEN_W
                and grid[r0 + r, c0 + c] == _GEN_WALL
                for r, c in touch_local
            )
            if not touches:
                positions.append((r0, c0))

    return positions


def _gen_place_distractors(grid: np.ndarray, count: int) -> bool:
    patches = _gen_distractor_patches()
    for _ in range(count):
        placed = False
        for idx in np.random.permutation(len(patches)):
            patch = patches[int(idx)]
            positions = _gen_valid_positions(grid, patch)
            if not positions:
                continue
            r0, c0 = positions[int(np.random.randint(0, len(positions)))]
            paste(grid, patch, r0, c0, transparent=_GEN_BG)
            placed = True
            break
        if not placed:
            return False
    return True


_GenUBox = tuple[int, int, int, int]


def _gen_design_expected() -> tuple[np.ndarray, bool, _GenUBox | None, int, int]:
    """Build the expected output and return metadata needed to derive the input."""
    expected = np.full((_GEN_H, _GEN_W), _GEN_BG, dtype=int)
    fill_container = bool(np.random.random() < _GEN_FILL_CONTAINER_PROB)
    has_u = fill_container or bool(np.random.random() < _GEN_MISSED_HAS_U_PROB)
    u_box = None

    if has_u:
        top, bottom, left, right, interior_h, interior_w = _gen_sample_u()
        paste(expected, _gen_u_patch(interior_h, interior_w), top, left, transparent=_GEN_BG)
        u_box = (top, bottom, left, right)
        if fill_container:
            marker_c = int(np.random.randint(left + 1, right))
            expected[top:bottom, left + 1 : right] = _GEN_MARKER
        else:
            outside = [c for c in range(_GEN_W) if not (left <= c <= right)]
            marker_c = int(outside[int(np.random.randint(0, len(outside)))])
            expected[-1, :] = _GEN_MARKER
        distractor_count = int(np.random.randint(*_GEN_DISTRACTORS_WITH_U_RANGE))
    else:
        marker_c = int(np.random.randint(0, _GEN_W))
        expected[-1, :] = _GEN_MARKER
        distractor_count = int(np.random.randint(*_GEN_DISTRACTORS_WITHOUT_U_RANGE))

    return expected, fill_container, u_box, marker_c, distractor_count


def _gen_input_from_expected(expected: np.ndarray, fill_container: bool, u_box: _GenUBox | None, marker_c: int) -> np.ndarray:
    inp = expected.copy()
    if fill_container:
        assert u_box is not None
        top, bottom, left, right = u_box
        inp[top:bottom, left + 1 : right] = _GEN_BG
    else:
        inp[-1, :] = _GEN_BG
    inp[_GEN_MARKER_ROW, marker_c] = _GEN_MARKER
    return inp


def _gen_valid_pair(inp: np.ndarray, expected: np.ndarray, fill_container: bool, u_box: _GenUBox | None, marker_c: int) -> bool:
    """Check structural invariants and the solver oracle."""
    if np.count_nonzero(inp == _GEN_MARKER) != 1 or inp[_GEN_MARKER_ROW, marker_c] != _GEN_MARKER:
        return False
    if not np.all(inp[-1, :] == _GEN_BG):
        return False

    wall_coords = np.argwhere(inp == _GEN_WALL)
    if wall_coords.size == 0 or int(wall_coords[:, 0].min()) < _GEN_MIN_WALL_ROW or int(wall_coords[:, 0].max()) > _GEN_LAST_WALL_ROW:
        return False

    component_count = len(find_components(inp, bg=_GEN_BG, colors=[_GEN_WALL], connectivity=4))
    if not (1 <= component_count <= 3):
        return False

    containers = _closed_u_containers(inp, _GEN_WALL, _GEN_BG)
    if len(containers) > 1:
        return False
    if any(marker_c in (left, right) for _top, _bottom, left, right in containers):
        return False
    if fill_container:
        if len(containers) != 1 or containers[0] != u_box:
            return False
    elif containers and containers[0][2] <= marker_c <= containers[0][3]:
        return False

    try:
        actual = main(inp)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    expected, fill_container, u_box, marker_c, distractor_count = _gen_design_expected()
    if not _gen_place_distractors(expected, distractor_count):
        return None

    inp = _gen_input_from_expected(expected, fill_container, u_box, marker_c)
    if not _gen_valid_pair(inp, expected, fill_container, u_box, marker_c):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid db7260a4 input")
