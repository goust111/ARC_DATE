import random
from collections import Counter

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []

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
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)
    return components


def _vertical_line(col: int, start: int, end: int) -> set[tuple[int, int]]:
    return {(r, col) for r in range(start, end + 1)}


def _horizontal_line(row: int, start: int, end: int) -> set[tuple[int, int]]:
    return {(row, c) for c in range(start, end + 1)}


def _extend_vertical(jr: int, jc: int, up: int, down: int, extra: int) -> set[tuple[int, int]]:
    assert (up == 0) ^ (down == 0)
    if up > 0:
        start, end = jr - up, jr + extra
    else:
        start, end = jr - extra, jr + down
    return _vertical_line(jc, start, end)


def _extend_horizontal(jr: int, jc: int, left: int, right: int, extra: int) -> set[tuple[int, int]]:
    assert (left == 0) ^ (right == 0)
    if left > 0:
        start, end = jc - left, jc + extra
    else:
        start, end = jc - extra, jc + right
    return _horizontal_line(jr, start, end)


def _analyze_component(coords: list[tuple[int, int]]) -> set[tuple[int, int]]:
    coord_set = set(coords)
    row_counts = Counter(r for r, _ in coords)
    col_counts = Counter(c for _, c in coords)

    wide_rows = [r for r, count in row_counts.items() if count > 1]
    wide_cols = [c for c, count in col_counts.items() if count > 1]
    assert len(wide_rows) == 1
    assert len(wide_cols) == 1

    jr = wide_rows[0]
    jc = wide_cols[0]

    row_cols = sorted(c for r, c in coords if r == jr)
    col_rows = sorted(r for r, c in coords if c == jc)
    assert row_cols == list(range(row_cols[0], row_cols[-1] + 1))
    assert col_rows == list(range(col_rows[0], col_rows[-1] + 1))

    expected = _horizontal_line(jr, row_cols[0], row_cols[-1]) | _vertical_line(jc, col_rows[0], col_rows[-1])
    assert coord_set == expected

    left = jc - row_cols[0]
    right = row_cols[-1] - jc
    up = jr - col_rows[0]
    down = col_rows[-1] - jr

    h_sides = int(left > 0) + int(right > 0)
    v_sides = int(up > 0) + int(down > 0)
    positive_sides = h_sides + v_sides

    if positive_sides == 2:
        assert h_sides == 1 and v_sides == 1
        horizontal_len = left + right + 1
        vertical_len = up + down + 1
        assert horizontal_len != vertical_len
        if horizontal_len > vertical_len:
            return _extend_horizontal(jr, jc, left, right, vertical_len - 1)
        return _extend_vertical(jr, jc, up, down, horizontal_len - 1)

    if positive_sides == 3:
        assert {h_sides, v_sides} == {1, 2}
        if h_sides == 1:
            assert up == down and up > 0
            trunk_len = left + right + 1
            branch_len = up + down + 1
            assert trunk_len > branch_len
            return _extend_horizontal(jr, jc, left, right, branch_len - 1)

        assert left == right and left > 0
        trunk_len = up + down + 1
        branch_len = left + right + 1
        assert trunk_len > branch_len
        return _extend_vertical(jr, jc, up, down, branch_len - 1)

    assert positive_sides == 4
    assert left == right and left > 0
    assert up == down and up > 0
    return _horizontal_line(jr, jc - left - up, jc + right + down)


def main(grid: np.ndarray) -> np.ndarray:
    """Unfold L, T, and cross shapes into straight line segments."""
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    components = _find_components(g != bg)
    assert components

    out = np.full_like(g, bg)
    for comp in components:
        color = int(g[comp[0]])
        new_coords = _analyze_component(comp)
        for r, c in new_coords:
            assert 0 <= r < g.shape[0] and 0 <= c < g.shape[1]
            assert out[r, c] == bg
            out[r, c] = color
    return out


# --------------- generator ---------------

SHAPE_TYPES = ("L", "T", "X")
MIN_OBJECTS = 1
MAX_OBJECTS = 4
BOX_GAP = 1
GEN_RETRIES = 400


def _build_l_patch(bg: int, fg: int) -> np.ndarray:
    long_len = random.randint(3, 7)
    short_len = random.randint(2, long_len - 1)
    height = long_len + short_len + 1
    width = short_len + 2
    patch = np.full((height, width), bg, dtype=int)
    joint_r = short_len
    joint_c = 1
    patch[joint_r : joint_r + long_len, joint_c] = fg
    patch[joint_r, joint_c : joint_c + short_len] = fg
    return patch


def _build_t_patch(bg: int, fg: int) -> np.ndarray:
    arm = random.randint(1, 3)
    branch_len = 2 * arm + 1
    trunk_len = random.randint(branch_len + 1, branch_len + 4)
    height = trunk_len + branch_len + 1
    width = branch_len + 2
    patch = np.full((height, width), bg, dtype=int)
    joint_r = branch_len
    joint_c = arm + 1
    patch[joint_r : joint_r + trunk_len, joint_c] = fg
    patch[joint_r, 1 : 1 + branch_len] = fg
    return patch


def _build_cross_patch(bg: int, fg: int) -> np.ndarray:
    h_arm = random.randint(1, 3)
    v_arm = random.randint(1, 3)
    height = 2 * v_arm + 3
    width = 2 * (h_arm + v_arm) + 3
    patch = np.full((height, width), bg, dtype=int)
    center_r = v_arm + 1
    center_c = h_arm + v_arm + 1
    patch[center_r, center_c - h_arm : center_c + h_arm + 1] = fg
    patch[center_r - v_arm : center_r + v_arm + 1, center_c] = fg
    return patch


def _build_patch(kind: str, bg: int, fg: int) -> np.ndarray:
    if kind == "L":
        patch = _build_l_patch(bg, fg)
    elif kind == "T":
        patch = _build_t_patch(bg, fg)
    else:
        assert kind == "X"
        patch = _build_cross_patch(bg, fg)
    turns = random.randrange(4)
    return np.rot90(patch, turns).copy()


def _sample_kinds(count: int) -> list[str]:
    if count == 3 and random.random() < 0.7:
        kinds = list(SHAPE_TYPES)
        random.shuffle(kinds)
        return kinds
    if count == 4 and random.random() < 0.7:
        kinds = list(SHAPE_TYPES) + [random.choice(SHAPE_TYPES)]
        random.shuffle(kinds)
        return kinds
    return [random.choice(SHAPE_TYPES) for _ in range(count)]


def _place_patches(patches: list[np.ndarray], bg: int) -> np.ndarray | None:
    dims = [patch.shape for patch in patches]
    max_h = max(h for h, _ in dims)
    max_w = max(w for _, w in dims)
    area = sum(h * w for h, w in dims)
    base = max(max_h, max_w, int(np.ceil(np.sqrt(area))) + 2)

    order = sorted(range(len(patches)), key=lambda i: patches[i].shape[0] * patches[i].shape[1], reverse=True)

    for growth in range(10):
        height = base + random.randint(3 + growth, 8 + growth)
        width = base + random.randint(3 + growth, 8 + growth)
        grid = np.full((height, width), bg, dtype=int)
        occupied = np.zeros((height, width), dtype=bool)
        positions: dict[int, tuple[int, int]] = {}

        success = True
        for idx in order:
            patch = patches[idx]
            ph, pw = patch.shape
            placed = False
            for _ in range(200):
                r0 = random.randint(0, height - ph)
                c0 = random.randint(0, width - pw)
                r1 = r0 + ph
                c1 = c0 + pw
                rs = max(0, r0 - BOX_GAP)
                re = min(height, r1 + BOX_GAP)
                cs = max(0, c0 - BOX_GAP)
                ce = min(width, c1 + BOX_GAP)
                if occupied[rs:re, cs:ce].any():
                    continue
                occupied[r0:r1, c0:c1] = True
                positions[idx] = (r0, c0)
                placed = True
                break
            if not placed:
                success = False
                break

        if not success:
            continue

        for idx, patch in enumerate(patches):
            r0, c0 = positions[idx]
            ph, pw = patch.shape
            grid[r0 : r0 + ph, c0 : c0 + pw] = patch

        return grid

    return None


def _oracle_accepts(grid: np.ndarray, bg: int, fg: int, expected_count: int) -> bool:
    if np.count_nonzero(grid == bg) <= np.count_nonzero(grid != bg):
        return False

    input_components = _find_components(grid != bg)
    if len(input_components) != expected_count:
        return False

    try:
        out = main(grid)
    except AssertionError:
        return False

    if out.shape != grid.shape or np.array_equal(out, grid):
        return False
    if set(map(int, np.unique(grid))) != {bg, fg}:
        return False
    if set(map(int, np.unique(out))) - {bg, fg}:
        return False

    output_components = _find_components(out != bg)
    if len(output_components) != expected_count:
        return False

    for comp in output_components:
        rows = {r for r, _ in comp}
        cols = {c for _, c in comp}
        if len(rows) != 1 and len(cols) != 1:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    bg, fg = random.sample(range(10), 2)
    count = random.randint(MIN_OBJECTS, MAX_OBJECTS)
    kinds = _sample_kinds(count)
    patches = [_build_patch(kind, bg, fg) for kind in kinds]
    grid = _place_patches(patches, bg)
    if grid is None:
        return None
    return grid if _oracle_accepts(grid, bg, fg, count) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
