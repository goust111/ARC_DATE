from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

CONTROL_TO_EXPAND = {GREEN: 0, RED: 1, BLUE: 2}
PLUS_MASK = np.array(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ],
    dtype=bool,
)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _find_cross_lattice(grid: np.ndarray) -> tuple[list[int], list[int], dict[tuple[int, int], tuple[int, int]]]:
    grey_components = _find_components(grid == GREY)
    assert len(grey_components) == 9, "expected nine grey crosses"

    centers: list[tuple[int, int]] = []
    for coords in grey_components:
        assert coords.shape[0] == 5, "each cross must have five grey pixels"
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        assert (r1 - r0 + 1, c1 - c0 + 1) == (3, 3), "each cross must fit in a 3x3 box"
        local = np.zeros((3, 3), dtype=bool)
        for rr, cc in coords:
            local[int(rr) - r0, int(cc) - c0] = True
        assert np.array_equal(local, PLUS_MASK), "grey component must be a plus"
        centers.append((r0 + 1, c0 + 1))

    rows = sorted({r for r, _c in centers})
    cols = sorted({c for _r, c in centers})
    assert len(rows) == 3 and len(cols) == 3, "expected a 3x3 cross lattice"
    assert all(rows[i + 1] - rows[i] == 5 for i in range(2))
    assert all(cols[i + 1] - cols[i] == 5 for i in range(2))

    center_set = set(centers)
    lattice: dict[tuple[int, int], tuple[int, int]] = {}
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            assert (r, c) in center_set, "missing lattice cross"
            lattice[(i, j)] = (r, c)
    return rows, cols, lattice


def _find_control_matrix(grid: np.ndarray) -> tuple[int, int]:
    mask = np.isin(grid, [BLUE, RED, GREEN])
    coords = np.argwhere(mask)
    assert coords.shape[0] == 9, "expected a dense 3x3 control matrix"
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    assert (r1 - r0 + 1, c1 - c0 + 1) == (3, 3), "control matrix must be 3x3"
    assert np.all(mask[r0 : r1 + 1, c0 : c1 + 1]), "control matrix must be dense"
    return r0, c0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each cross according to a control matrix, with larger expansions winning overlaps.

    Input structure:
    - A black background contains a regular 3x3 lattice of identical grey 3x3 plus signs with center spacing five cells.
    - A separate dense 3x3 control matrix of blue, red, and green pixels appears in the lower-right area, indexed to the cross lattice by row and column.
    - Optional yellow marker pixels may lie on future arm-extension paths; they are not part of the grey crosses or the control matrix.

    Rule:
    - Read the control matrix and map each control color to an expansion amount: green keeps the original cross, red extends each arm outward by one more cell, and blue extends each arm outward by two more cells.
    - Recolor every cross center to its control color, then paint grey arm pixels along the four cardinal directions up to the resulting arm length, skipping yellow markers and leaving the control matrix unchanged.
    - If multiple expanded crosses claim the same pixel, keep the claim from the larger expansion; ties are broken by the earlier lattice position, preferring upper rows and then left columns.

    Key insight:
    - The control matrix encodes both geometry and priority: the same value that makes a cross larger also determines that it wins contested overlap cells.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    _rows, _cols, lattice = _find_cross_lattice(g)
    ctl_r0, ctl_c0 = _find_control_matrix(g)

    out = g.copy()
    claims: dict[tuple[int, int], tuple[tuple[int, int, int], int]] = {}
    h, w = g.shape

    for i in range(3):
        for j in range(3):
            center_r, center_c = lattice[(i, j)]
            control_color = int(g[ctl_r0 + i, ctl_c0 + j])
            assert control_color in CONTROL_TO_EXPAND
            expand = CONTROL_TO_EXPAND[control_color]
            arm_len = 1 + expand
            priority = (expand, -i, -j)

            for dr, dc in DIRS4:
                for step in range(1, arm_len + 1):
                    rr = center_r + dr * step
                    cc = center_c + dc * step
                    if not (0 <= rr < h and 0 <= cc < w):
                        continue
                    if g[rr, cc] in (YELLOW, BLUE, RED, GREEN):
                        continue
                    key = (rr, cc)
                    prev = claims.get(key)
                    if prev is None or priority > prev[0]:
                        claims[key] = (priority, GREY)

            out[center_r, center_c] = control_color

    for (r, c), (_priority, color) in claims.items():
        out[r, c] = color
    return out


def _stamp_plus(grid: np.ndarray, center_r: int, center_c: int, color: int) -> None:
    grid[center_r, center_c] = color
    grid[center_r - 1, center_c] = color
    grid[center_r + 1, center_c] = color
    grid[center_r, center_c - 1] = color
    grid[center_r, center_c + 1] = color


def _overlap_pairs() -> list[tuple[tuple[int, int], tuple[int, int]]]:
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for i in range(3):
        for j in range(3):
            if i + 1 < 3:
                pairs.append(((i, j), (i + 1, j)))
            if j + 1 < 3:
                pairs.append(((i, j), (i, j + 1)))
    return pairs


def _has_required_overlap(control: np.ndarray) -> bool:
    lengths = np.vectorize(lambda color: 1 + CONTROL_TO_EXPAND[int(color)])(control)
    for (r1, c1), (r2, c2) in _overlap_pairs():
        if abs(r1 - r2) + abs(c1 - c2) != 1:
            continue
        if int(lengths[r1, c1] + lengths[r2, c2]) >= 5:
            return True
    return False


def _sample_control_matrix(rng: np.random.Generator, tie_branch: bool) -> np.ndarray:
    control = rng.choice(np.array([GREEN, RED, BLUE], dtype=int), size=(3, 3), p=[0.45, 0.35, 0.20])
    anchor_pairs = _overlap_pairs()
    forced_a, forced_b = anchor_pairs[int(rng.integers(len(anchor_pairs)))]
    if tie_branch:
        control[forced_a] = BLUE
        control[forced_b] = BLUE
    else:
        if rng.random() < 0.5:
            control[forced_a] = BLUE
            control[forced_b] = RED
        else:
            control[forced_a] = RED
            control[forced_b] = BLUE

    # Force every control color to appear at least once for full mapping coverage.
    flat = control.reshape(-1)
    for color in (GREEN, RED, BLUE):
        if not np.any(flat == color):
            replace_idx = int(rng.integers(flat.size))
            flat[replace_idx] = color
    control = flat.reshape(3, 3)

    # Rarely the forced color repair above can remove the only overlap pair; retry locally.
    if _has_required_overlap(control):
        return control
    for _ in range(50):
        control = rng.choice(np.array([GREEN, RED, BLUE], dtype=int), size=(3, 3), p=[0.45, 0.35, 0.20])
        control[forced_a] = BLUE
        control[forced_b] = BLUE if tie_branch else RED
        flat = control.reshape(-1)
        for color in (GREEN, RED, BLUE):
            if not np.any(flat == color):
                flat[int(rng.integers(flat.size))] = color
        control = flat.reshape(3, 3)
        if _has_required_overlap(control):
            return control
    raise RuntimeError("failed to sample a valid control matrix")


def _expansion_targets(center: tuple[int, int], arm_len: int) -> list[tuple[int, int]]:
    center_r, center_c = center
    cells: list[tuple[int, int]] = []
    for dr, dc in DIRS4:
        for step in range(2, arm_len + 1):
            cells.append((center_r + dr * step, center_c + dc * step))
    return cells


def _sample_markers(
    rng: np.random.Generator,
    control: np.ndarray,
    centers: dict[tuple[int, int], tuple[int, int]],
) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for i in range(3):
        for j in range(3):
            arm_len = 1 + CONTROL_TO_EXPAND[int(control[i, j])]
            candidates.extend(_expansion_targets(centers[(i, j)], arm_len))

    unique_candidates = sorted(set(candidates))
    assert unique_candidates, "need at least one expandable arm cell for yellow markers"
    rng.shuffle(unique_candidates)
    marker_count = int(rng.integers(1, min(4, len(unique_candidates)) + 1))
    chosen = unique_candidates[:marker_count]

    # Ensure at least one marker blocks an actual extension beyond the original 3x3 crosses.
    assert chosen
    return chosen


def _build_input_from_layout(
    control: np.ndarray,
    margins: tuple[int, int, int, int, int, int],
    markers: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    top, left, gap_r, gap_c, bottom, right = margins
    h = top + 13 + gap_r + 3 + bottom
    w = left + 13 + gap_c + 3 + right
    grid = np.zeros((h, w), dtype=int)

    centers: dict[tuple[int, int], tuple[int, int]] = {}
    for i in range(3):
        for j in range(3):
            center_r = top + 1 + 5 * i
            center_c = left + 1 + 5 * j
            centers[(i, j)] = (center_r, center_c)
            _stamp_plus(grid, center_r, center_c, GREY)

    ctl_r0 = top + 13 + gap_r
    ctl_c0 = left + 13 + gap_c
    grid[ctl_r0 : ctl_r0 + 3, ctl_c0 : ctl_c0 + 3] = control

    if markers is not None:
        for r, c in markers:
            grid[r, c] = YELLOW
    return grid


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    tie_branch = bool(rng.integers(2))
    control = _sample_control_matrix(rng, tie_branch=tie_branch)

    margins = (
        int(rng.integers(1, 4)),
        int(rng.integers(1, 4)),
        int(rng.integers(1, 4)),
        int(rng.integers(1, 4)),
        int(rng.integers(1, 4)),
        int(rng.integers(1, 4)),
    )

    top, left, _gap_r, _gap_c, _bottom, _right = margins
    centers = {(i, j): (top + 1 + 5 * i, left + 1 + 5 * j) for i in range(3) for j in range(3)}
    markers = _sample_markers(rng, control, centers)
    inp = _build_input_from_layout(control, margins, markers=markers)

    try:
        out = main(inp)
    except AssertionError:
        return None

    if out.shape != inp.shape or max(inp.shape) > 30:
        return None
    if not np.any(inp == YELLOW):
        return None
    if not np.any(out == BLUE) or not np.any(out == RED) or not np.any(out == GREEN):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(300):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
