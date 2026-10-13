import numpy as np


RED = 2
GRAY = 5
BG = 0
CONTROL_COLORS = (1, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 400
_RNG = np.random.default_rng()


def _rotate_cw(patch: np.ndarray, turns_cw: int) -> np.ndarray:
    return np.rot90(patch, k=(-int(turns_cw)) % 4).copy()


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _shell_templates() -> list[np.ndarray]:
    raws = [
        [
            "11110",
            "10010",
            "10011",
            "11110",
            "01000",
        ],
        [
            "11111",
            "10010",
            "10010",
            "11110",
            "00100",
        ],
        [
            "11110",
            "10010",
            "10011",
            "11111",
            "00010",
        ],
    ]
    out = []
    for raw in raws:
        patch = np.array([[RED if ch == "1" else BG for ch in row] for row in raw], dtype=int)
        out.append(patch)
    return out


SHELL_TEMPLATES = _shell_templates()


TURN_TO_MARKER = {
    0: (0, 0),
    1: (0, 1),
    2: (1, 1),
    3: (1, 0),
}
MARKER_TO_TURN = {v: k for k, v in TURN_TO_MARKER.items()}


def _find_template_patch(grid: np.ndarray) -> np.ndarray:
    reds = _find_components(grid == RED)
    red_comps = [comp for comp in reds if comp.shape[0] >= 8]
    assert len(red_comps) == 1, "expected exactly one red sprite shell"
    r0, c0, r1, c1 = _bbox(red_comps[0])
    patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    assert patch.shape == (5, 5), "sprite bbox must be 5x5"
    return patch


def _find_cavity_origin(patch: np.ndarray, allowed_gray_counts: set[int]) -> tuple[int, int]:
    h, w = patch.shape
    assert (h, w) == (5, 5)
    candidates = []
    for r0 in range(1, h - 2):
        for c0 in range(1, w - 2):
            inner = patch[r0 : r0 + 2, c0 : c0 + 2]
            if not np.all((inner == BG) | (inner == GRAY)):
                continue
            if int(np.count_nonzero(inner == GRAY)) not in allowed_gray_counts:
                continue
            ring = patch[r0 - 1 : r0 + 3, c0 - 1 : c0 + 3].copy()
            ring[1:3, 1:3] = RED
            if np.all(ring == RED):
                candidates.append((r0, c0))
    assert len(candidates) == 1, "expected a unique enclosed 2x2 cavity"
    return candidates[0]


def _find_cavity(patch: np.ndarray) -> tuple[int, int]:
    return _find_cavity_origin(patch, {1})


def _marker_turn_from_patch(patch: np.ndarray) -> int:
    r0, c0 = _find_cavity(patch)
    inner = patch[r0 : r0 + 2, c0 : c0 + 2]
    mr, mc = np.argwhere(inner == GRAY)[0]
    return MARKER_TO_TURN[(int(mr), int(mc))]


def _control_spec(grid: np.ndarray) -> tuple[list[int], list[int]]:
    coords = np.argwhere((grid != BG) & (grid != RED) & (grid != GRAY))
    assert 3 <= coords.shape[0] <= 4, "expected 3-4 control pixels"
    rows = {int(r) for r, _ in coords}
    assert len(rows) == 1, "control pixels must lie on one row"
    coords = coords[np.argsort(coords[:, 1])]
    colors = [int(grid[r, c]) for r, c in coords]
    assert len(set(colors)) == len(colors), "control colors must be distinct"
    gaps = []
    for i in range(len(colors) - 1):
        gap = int(coords[i + 1, 1] - coords[i, 1] - 1)
        assert 0 <= gap <= 2, "control gaps must be in [0, 2]"
        gaps.append(gap)
    return colors, gaps


def _render(rotated_patch: np.ndarray, control_colors: list[int], gaps: list[int]) -> np.ndarray:
    tile_h, tile_w = rotated_patch.shape
    width = len(control_colors) * tile_w + sum(gaps)
    out = np.zeros((tile_h, width), dtype=int)
    c0 = 0
    for i, color in enumerate(control_colors):
        tile = rotated_patch.copy()
        tile[tile == RED] = int(color)
        out[:, c0 : c0 + tile_w] = tile
        if i < len(gaps):
            c0 += tile_w + int(gaps[i])
    return out


def _parse_instance(grid: np.ndarray) -> tuple[np.ndarray, int, list[int], list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    patch = _find_template_patch(g)
    turns_cw = _marker_turn_from_patch(patch)
    colors, gaps = _control_spec(g)
    return patch, turns_cw, colors, gaps


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate the sprite by its internal marker code, then tile recolored copies using the control row spacing.

    Input structure:
    - Background is black.
    - The upper area contains one 5x5 red sprite shell with a single enclosed 2x2 cavity; exactly one cavity cell is gray and the other three are black.
    - The lower area contains one horizontal control row of 3-4 distinct single-pixel colors separated by 0-2 black cells.

    Rule:
    - Locate the unique enclosed 2x2 cavity inside the red shell.
    - Decode the gray marker position inside that cavity as a clockwise rotation: upper-left=0, upper-right=90, lower-right=180, lower-left=270.
    - Rotate the entire 5x5 sprite, including the gray marker, by that amount.
    - Read the control row left-to-right; for each control color, output one copy of the rotated sprite with its red shell recolored to that control color.
    - Preserve the gray marker and keep exactly the same horizontal gaps between neighboring output copies as between neighboring control pixels.

    Key insight:
    - The sprite itself carries the transformation parameter via the gray cavity marker, while the detached control row contributes only repetition order, recoloring, and spacing.
    """
    patch, turns_cw, colors, gaps = _parse_instance(grid)
    rotated = _rotate_cw(patch, turns_cw)
    return _render(rotated, colors, gaps)


def _distinct_rotations(patch: np.ndarray) -> bool:
    variants = {_rotate_cw(patch, k).tobytes() for k in range(4)}
    return len(variants) == 4


def _place_marker(shell: np.ndarray, turns_cw: int) -> np.ndarray:
    patch = shell.copy()
    r0, c0 = _find_cavity_origin(patch, {0})
    dr, dc = TURN_TO_MARKER[int(turns_cw)]
    patch[r0 + dr, c0 + dc] = GRAY
    return patch


def _build_input(
    sprite_patch: np.ndarray,
    height: int,
    width: int,
    sprite_r0: int,
    sprite_c0: int,
    control_row: int,
    control_c0: int,
    control_colors: list[int],
    gaps: list[int],
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    grid[sprite_r0 : sprite_r0 + 5, sprite_c0 : sprite_c0 + 5] = sprite_patch
    col = int(control_c0)
    for i, color in enumerate(control_colors):
        grid[control_row, col] = int(color)
        if i < len(gaps):
            col += 1 + int(gaps[i])
    return grid


def _validate_instance(
    grid: np.ndarray,
    expected_patch: np.ndarray,
    turns_cw: int,
    control_colors: list[int],
    gaps: list[int],
    control_row: int,
    control_c0: int,
) -> bool:
    parsed_patch, parsed_turns, parsed_colors, parsed_gaps = _parse_instance(grid)
    if not np.array_equal(parsed_patch, expected_patch):
        return False
    if parsed_turns != int(turns_cw):
        return False
    if parsed_colors != list(control_colors) or parsed_gaps != list(gaps):
        return False

    out = main(grid)
    expected = _render(_rotate_cw(expected_patch, turns_cw), control_colors, gaps)
    if not np.array_equal(out, expected):
        return False
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return False

    marker_variants = []
    for alt_turn in range(4):
        alt_patch = expected_patch.copy()
        alt_patch[alt_patch == GRAY] = BG
        cavity_r0, cavity_c0 = _find_cavity_origin(alt_patch, {0})
        dr, dc = TURN_TO_MARKER[alt_turn]
        alt_patch[cavity_r0 + dr, cavity_c0 + dc] = GRAY
        marker_variants.append(main(_replace_sprite(grid, alt_patch)))
    if len({v.tobytes() for v in marker_variants}) != 4:
        return False

    alt_color = next(c for c in CONTROL_COLORS if c not in control_colors)
    color_variant = grid.copy()
    color_variant[control_row, control_c0] = alt_color
    if np.array_equal(main(color_variant), out):
        return False

    alt_gaps = list(gaps)
    alt_gaps[0] = (alt_gaps[0] + 1) % 3
    if alt_gaps != list(gaps):
        gap_variant = _replace_controls(grid, control_row, control_c0, control_colors, alt_gaps)
        if np.array_equal(main(gap_variant), out):
            return False

    shell_only = expected_patch.copy()
    shell_only[shell_only == GRAY] = BG
    if not _distinct_rotations(shell_only):
        return False
    return True


def _replace_sprite(grid: np.ndarray, sprite_patch: np.ndarray) -> np.ndarray:
    out = grid.copy()
    reds = _find_components(out == RED)
    red_comps = [comp for comp in reds if comp.shape[0] >= 8]
    assert len(red_comps) == 1
    r0, c0, _, _ = _bbox(red_comps[0])
    out[r0 : r0 + 5, c0 : c0 + 5] = sprite_patch
    return out


def _replace_controls(
    grid: np.ndarray,
    control_row: int,
    control_c0: int,
    control_colors: list[int],
    gaps: list[int],
) -> np.ndarray:
    out = grid.copy()
    out[control_row, :] = np.where((out[control_row, :] != RED) & (out[control_row, :] != GRAY), BG, out[control_row, :])
    col = int(control_c0)
    for i, color in enumerate(control_colors):
        out[control_row, col] = int(color)
        if i < len(gaps):
            col += 1 + int(gaps[i])
    return out


def _gen_attempt() -> np.ndarray | None:
    shell = SHELL_TEMPLATES[int(_RNG.integers(0, len(SHELL_TEMPLATES)))].copy()
    if not _distinct_rotations(shell):
        return None
    input_rot = int(_RNG.integers(0, 4))
    shell = _rotate_cw(shell, input_rot)
    turns_cw = int(_RNG.integers(0, 4))
    sprite_patch = _place_marker(shell, turns_cw)

    n_controls = int(_RNG.choice([3, 4]))
    control_colors = list(map(int, _RNG.choice(CONTROL_COLORS, size=n_controls, replace=False)))
    gaps = [int(x) for x in _RNG.integers(0, 3, size=n_controls - 1)]
    span = n_controls + sum(gaps)

    min_width = max(12, 5, span + 4)
    width = int(_RNG.integers(min_width, 23))
    height = int(_RNG.integers(10, 17))

    sprite_r0 = int(_RNG.integers(0, min(4, height - 6)))
    sprite_c0 = int(_RNG.integers(0, width - 5 + 1))
    min_control_row = sprite_r0 + 6
    if min_control_row >= height:
        return None
    control_row = int(_RNG.integers(min_control_row, height))
    control_c0 = int(_RNG.integers(0, width - span - 1))

    grid = _build_input(
        sprite_patch,
        height,
        width,
        sprite_r0,
        sprite_c0,
        control_row,
        control_c0,
        control_colors,
        gaps,
    )
    if not _validate_instance(grid, sprite_patch, turns_cw, control_colors, gaps, control_row, control_c0):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")


def _self_check(iterations: int = 200) -> None:
    for _ in range(iterations):
        grid = generate_input()
        out = main(grid)
        assert isinstance(grid, np.ndarray)
        assert isinstance(out, np.ndarray)
        assert grid.ndim == 2 and out.ndim == 2
        assert max(grid.shape) <= 30 and max(out.shape) <= 30


if __name__ == "__main__":
    _self_check()
