from __future__ import annotations

import numpy as np


QUAD_SIZE = 8
GRID_SIZE = QUAD_SIZE * 2 + 1
SEPARATOR_COLOR = 5
SKELETON_COLOR = 1
ANCHOR_COLOR = 2
LABEL_COLORS = (3, 4)
BACKGROUND = 0
GEN_ATTEMPTS = 400


def _separator_index(grid: np.ndarray, axis: int) -> int:
    indices = []
    length = grid.shape[1 - axis]
    for idx in range(grid.shape[axis]):
        line = grid[idx, :] if axis == 0 else grid[:, idx]
        if np.all(line == SEPARATOR_COLOR):
            indices.append(idx)
    assert len(indices) == 1, "expected exactly one grey separator line per axis"
    idx = indices[0]
    assert idx == QUAD_SIZE and length == GRID_SIZE, "expected four 8x8 quadrants split by a central cross"
    return idx


def _split_quadrants(grid: np.ndarray) -> list[np.ndarray]:
    row = _separator_index(grid, axis=0)
    col = _separator_index(grid, axis=1)
    quadrants = [
        grid[:row, :col],
        grid[:row, col + 1 :],
        grid[row + 1 :, :col],
        grid[row + 1 :, col + 1 :],
    ]
    assert all(q.shape == (QUAD_SIZE, QUAD_SIZE) for q in quadrants)
    return quadrants


def _role_layer(quad: np.ndarray) -> np.ndarray:
    role = np.zeros_like(quad)
    mask = (quad == SKELETON_COLOR) | (quad == ANCHOR_COLOR)
    role[mask] = quad[mask]
    return role


def _label_layer(quad: np.ndarray) -> np.ndarray:
    labels = np.zeros_like(quad)
    mask = np.isin(quad, LABEL_COLORS)
    labels[mask] = quad[mask]
    return labels


def _find_complete_quadrant(quadrants: list[np.ndarray]) -> int:
    indices = [idx for idx, quad in enumerate(quadrants) if np.any(np.isin(quad, LABEL_COLORS))]
    assert len(indices) == 1, "expected exactly one fully annotated quadrant"
    return indices[0]


def _rotation_k(src_role: np.ndarray, dst_role: np.ndarray) -> int:
    matches = [k for k in range(4) if np.array_equal(np.rot90(src_role, k), dst_role)]
    assert len(matches) == 1, "skeleton+anchor alignment must determine a unique rotation"
    return matches[0]


def _overlay_priority(quadrants: list[np.ndarray]) -> np.ndarray:
    out = np.zeros((QUAD_SIZE, QUAD_SIZE), dtype=int)
    for quad in quadrants:
        fill_mask = (out == BACKGROUND) & (quad != BACKGROUND)
        out[fill_mask] = quad[fill_mask]
    return out


def main(input_grid: np.ndarray) -> np.ndarray:
    """Complete the missing quadrant annotations by anchor-aligned rotation, then overlay the four finished quadrants with fixed priority.

    Input structure:
    - A 17x17 grid is split by one full grey row and one full grey column into four 8x8 quadrants.
    - Every quadrant contains the same blue 4-connected skeleton and one red anchor pixel, but each copy may be rotated by 0/90/180/270 degrees.
    - Exactly one quadrant also contains green and/or yellow annotation pixels; the other three contain only the skeleton and anchor.

    Rule:
    - Identify the fully annotated quadrant and keep only its blue skeleton plus red anchor as the alignment template.
    - For each incomplete quadrant, find the unique quarter-turn rotation that maps the template skeleton+anchor onto that quadrant's skeleton+anchor, then rotate only the source annotation pixels by the same amount into the target quadrant.
    - Remove the grey separator cross and overlay the four completed quadrants into one 8x8 output, taking the first non-background color at each cell in priority order: top-left, top-right, bottom-left, bottom-right.

    Key insight:
    - The red anchor breaks rotational ambiguity inside each quadrant, so labels must be transferred before the final transparent overlay can correctly preserve all visible information.
    """
    grid = np.asarray(input_grid, dtype=int)
    assert grid.ndim == 2 and grid.shape == (GRID_SIZE, GRID_SIZE)

    quadrants = _split_quadrants(grid)
    complete_idx = _find_complete_quadrant(quadrants)
    source_role = _role_layer(quadrants[complete_idx])
    source_labels = _label_layer(quadrants[complete_idx])

    completed: list[np.ndarray] = []
    for idx, quad in enumerate(quadrants):
        role = _role_layer(quad)
        if idx == complete_idx:
            completed.append(quad.copy())
            continue

        assert not np.any(np.isin(quad, LABEL_COLORS)), "incomplete quadrants may not already contain labels"
        k = _rotation_k(source_role, role)
        rotated_labels = np.rot90(source_labels, k)
        filled = quad.copy()
        label_mask = rotated_labels != BACKGROUND
        assert np.all(filled[label_mask] == BACKGROUND), "transferred labels must land on empty cells"
        filled[label_mask] = rotated_labels[label_mask]
        completed.append(filled)

    return _overlay_priority(completed)


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _normalize(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _rotation_count(pattern: np.ndarray) -> int:
    seen: set[bytes] = set()
    for k in range(4):
        seen.add(np.rot90(pattern, k).tobytes())
    return len(seen)


def _random_skeleton(rng: np.random.Generator) -> tuple[tuple[int, int], ...]:
    for _ in range(200):
        size = int(rng.integers(3, 9))
        cells: set[tuple[int, int]] = {(0, 0)}
        while len(cells) < size:
            frontier = sorted({nbr for cell in cells for nbr in _neighbors4(cell)} - cells)
            candidate = frontier[int(rng.integers(0, len(frontier)))]
            cells.add(candidate)
        normalized = _normalize(cells)
        rows = [r for r, _ in normalized]
        cols = [c for _, c in normalized]
        height = max(rows) + 1
        width = max(cols) + 1
        if 2 <= height <= 5 and 2 <= width <= 5:
            return normalized
    raise RuntimeError("failed to generate a compact connected skeleton")


def _build_base_quadrant(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(200):
        skeleton_shape = _random_skeleton(rng)
        rows = [r for r, _ in skeleton_shape]
        cols = [c for _, c in skeleton_shape]
        height = max(rows) + 1
        width = max(cols) + 1
        r0 = int(rng.integers(0, QUAD_SIZE - height + 1))
        c0 = int(rng.integers(0, QUAD_SIZE - width + 1))
        skeleton = {(r0 + r, c0 + c) for r, c in skeleton_shape}

        anchor_candidates = sorted(
            {
                nbr
                for cell in skeleton
                for nbr in _neighbors4(cell)
                if 0 <= nbr[0] < QUAD_SIZE and 0 <= nbr[1] < QUAD_SIZE and nbr not in skeleton
            }
        )
        if not anchor_candidates:
            continue
        anchor = anchor_candidates[int(rng.integers(0, len(anchor_candidates)))]

        role = np.zeros((QUAD_SIZE, QUAD_SIZE), dtype=int)
        for r, c in skeleton:
            role[r, c] = SKELETON_COLOR
        role[anchor] = ANCHOR_COLOR
        if _rotation_count(role) != 4:
            continue

        occupied = skeleton | {anchor}
        near_cells = []
        for r in range(QUAD_SIZE):
            for c in range(QUAD_SIZE):
                if (r, c) in occupied:
                    continue
                min_cheb = min(max(abs(r - rr), abs(c - cc)) for rr, cc in occupied)
                if min_cheb <= 2:
                    near_cells.append((r, c))
        if len(near_cells) < 2:
            continue

        labels = np.zeros((QUAD_SIZE, QUAD_SIZE), dtype=int)
        mode = int(rng.integers(0, 3))
        label_count = int(rng.integers(1, 5))
        if mode == 2:
            label_count = max(label_count, 2)
        picks = rng.choice(len(near_cells), size=label_count, replace=False)
        chosen = [near_cells[int(i)] for i in np.atleast_1d(picks)]

        if mode == 0:
            colors = [3] * label_count
        elif mode == 1:
            colors = [4] * label_count
        else:
            colors = [3, 4]
            while len(colors) < label_count:
                colors.append(int(rng.choice(LABEL_COLORS)))
            rng.shuffle(colors)

        for (r, c), color in zip(chosen, colors):
            labels[r, c] = int(color)

        return role, labels
    raise RuntimeError("failed to generate a non-symmetric quadrant pattern")


def _render_input(
    role: np.ndarray,
    labels: np.ndarray,
    orientations: list[int],
    complete_idx: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    quadrants = []
    completed = []
    for idx, k in enumerate(orientations):
        quad_role = np.rot90(role, k)
        quad_labels = np.rot90(labels, k)

        quad = quad_role.copy()
        if idx == complete_idx:
            quad[quad_labels != BACKGROUND] = quad_labels[quad_labels != BACKGROUND]
        quadrants.append(quad)

        full_quad = quad_role.copy()
        full_quad[quad_labels != BACKGROUND] = quad_labels[quad_labels != BACKGROUND]
        completed.append(full_quad)

    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    grid[QUAD_SIZE, :] = SEPARATOR_COLOR
    grid[:, QUAD_SIZE] = SEPARATOR_COLOR
    grid[:QUAD_SIZE, :QUAD_SIZE] = quadrants[0]
    grid[:QUAD_SIZE, QUAD_SIZE + 1 :] = quadrants[1]
    grid[QUAD_SIZE + 1 :, :QUAD_SIZE] = quadrants[2]
    grid[QUAD_SIZE + 1 :, QUAD_SIZE + 1 :] = quadrants[3]
    return grid, completed


def _visible_contributors(completed: list[np.ndarray]) -> int:
    seen: set[int] = set()
    for r in range(QUAD_SIZE):
        for c in range(QUAD_SIZE):
            for idx, quad in enumerate(completed):
                if quad[r, c] != BACKGROUND:
                    seen.add(idx)
                    break
    return len(seen)


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        role, labels = _build_base_quadrant(rng)
        orientations = rng.integers(0, 4, size=4).tolist()
        if len(set(orientations)) < 2:
            continue
        complete_idx = int(rng.integers(0, 4))

        grid, completed = _render_input(role, labels, orientations, complete_idx)
        if _visible_contributors(completed) < 2:
            continue

        output = main(grid)
        if output.shape != (QUAD_SIZE, QUAD_SIZE):
            continue
        if not np.any(np.isin(output, LABEL_COLORS)):
            continue
        if not np.array_equal(output, _overlay_priority(completed)):
            continue
        return grid
    raise RuntimeError("failed to generate a valid quadrant-overlay puzzle")
