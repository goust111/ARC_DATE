from __future__ import annotations

import numpy as np


ORTHO_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
PALETTE = tuple(range(10))
GENERATION_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    colors, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _components4_non_bg(grid: np.ndarray, bg: int) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    seen = np.zeros((H, W), dtype=bool)
    components: list[np.ndarray] = []

    for r in range(H):
        for c in range(W):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ORTHO_DIRS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and int(g[nr, nc]) != bg:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _cross_center(
    grid: np.ndarray,
    coords: np.ndarray,
    *,
    require_distinct_center: bool,
) -> tuple[int, int, int] | None:
    if int(coords.shape[0]) != 5:
        return None

    g = np.asarray(grid, dtype=int)
    cells = {tuple(map(int, rc)) for rc in coords}
    centers: list[tuple[int, int, int]] = []

    for r, c in cells:
        arms = {(r + dr, c + dc) for dr, dc in ORTHO_DIRS}
        if arms != cells - {(r, c)}:
            continue
        arm_colors = {int(g[rr, cc]) for rr, cc in arms}
        if len(arm_colors) != 1:
            continue
        arm_color = next(iter(arm_colors))
        center_color = int(g[r, c])
        if require_distinct_center and center_color == arm_color:
            continue
        centers.append((r, c, center_color))

    if len(centers) != 1:
        return None
    return centers[0]


def _valid_marker_map(grid: np.ndarray, bg: int) -> np.ndarray | None:
    H, W = map(int, grid.shape)
    if H % 3 != 0 or W % 3 != 0:
        return None

    block_h = H // 3
    block_w = W // 3
    out = np.full((3, 3), bg, dtype=int)

    for comp in _components4_non_bg(grid, bg):
        marker = _cross_center(grid, comp, require_distinct_center=True)
        if marker is None:
            continue
        r, c, color = marker
        br = int(r // block_h)
        bc = int(c // block_w)
        if int(out[br, bc]) != bg:
            return None
        out[br, bc] = int(color)
    return out


def _sample_color(rng: np.random.Generator, exclude: set[int]) -> int:
    choices = [color for color in PALETTE if color not in exclude]
    return int(rng.choice(choices))


def _cross_patch(bg: int, arm_color: int, center_color: int) -> np.ndarray:
    patch = np.full((3, 3), bg, dtype=int)
    for rr, cc in ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)):
        patch[rr, cc] = arm_color
    patch[1, 1] = center_color
    return patch


def _embedded_plus_patch(rng: np.random.Generator, bg: int, color: int) -> np.ndarray:
    variant = int(rng.integers(0, 2))
    if variant == 0:
        return np.full((3, 3), color, dtype=int)

    patch = np.full((4, 3), bg, dtype=int)
    for rr, cc in ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (3, 1)):
        patch[rr, cc] = color
    return patch


def _noise_patch(rng: np.random.Generator, bg: int, color: int) -> np.ndarray:
    variant = int(rng.integers(0, 5))
    if variant == 0:
        return np.array([[color]], dtype=int)
    if variant == 1:
        length = int(rng.integers(2, 5))
        if bool(rng.integers(0, 2)):
            return np.full((1, length), color, dtype=int)
        return np.full((length, 1), color, dtype=int)
    if variant == 2:
        patch = np.full((2, 2), bg, dtype=int)
        for rr, cc in ((0, 0), (0, 1), (1, 0)):
            patch[rr, cc] = color
        if bool(rng.integers(0, 2)):
            patch = np.fliplr(patch)
        if bool(rng.integers(0, 2)):
            patch = np.flipud(patch)
        return patch
    if variant == 3:
        return np.full((2, 2), color, dtype=int)

    patch = np.full((2, 3), bg, dtype=int)
    for rr, cc in ((0, 0), (0, 1), (1, 1), (1, 2)):
        patch[rr, cc] = color
    if bool(rng.integers(0, 2)):
        patch = np.fliplr(patch)
    if bool(rng.integers(0, 2)):
        patch = np.flipud(patch)
    return patch


def _try_place_patch(
    block: np.ndarray,
    blocked: np.ndarray,
    patch: np.ndarray,
    bg: int,
    rng: np.random.Generator,
) -> bool:
    bh, bw = map(int, block.shape)
    ph, pw = map(int, patch.shape)
    if ph > bh or pw > bw:
        return False

    mask = patch != bg
    anchors = [(r0, c0) for r0 in range(bh - ph + 1) for c0 in range(bw - pw + 1)]
    for idx in rng.permutation(len(anchors)):
        r0, c0 = anchors[int(idx)]
        if np.any(blocked[r0 : r0 + ph, c0 : c0 + pw] & mask):
            continue
        region = block[r0 : r0 + ph, c0 : c0 + pw]
        region[mask] = patch[mask]
        for rr, cc in np.argwhere(mask):
            ar = r0 + int(rr)
            ac = c0 + int(cc)
            blocked[max(0, ar - 1) : min(bh, ar + 2), max(0, ac - 1) : min(bw, ac + 2)] = True
        return True
    return False


def _build_block(
    kind: str,
    block_shape: tuple[int, int],
    bg: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int] | None:
    block = np.full(block_shape, bg, dtype=int)
    blocked = np.zeros(block_shape, dtype=bool)
    expected_color = bg

    required: list[np.ndarray] = []
    optional: list[np.ndarray] = []

    if kind == "valid":
        arm_color = _sample_color(rng, {bg})
        center_color = _sample_color(rng, {bg, arm_color})
        required.append(_cross_patch(bg, arm_color, center_color))
        expected_color = center_color
        if rng.random() < 0.75:
            extra_kind = str(rng.choice(["mono", "embedded", "noise"]))
            extra_color = _sample_color(rng, {bg})
            if extra_kind == "mono":
                optional.append(_cross_patch(bg, extra_color, extra_color))
            elif extra_kind == "embedded":
                optional.append(_embedded_plus_patch(rng, bg, extra_color))
            else:
                optional.append(_noise_patch(rng, bg, extra_color))
    elif kind == "mono":
        color = _sample_color(rng, {bg})
        required.append(_cross_patch(bg, color, color))
        if rng.random() < 0.5:
            optional.append(_noise_patch(rng, bg, _sample_color(rng, {bg})))
    elif kind == "embedded":
        color = _sample_color(rng, {bg})
        required.append(_embedded_plus_patch(rng, bg, color))
        if rng.random() < 0.5:
            optional.append(_noise_patch(rng, bg, _sample_color(rng, {bg})))
    elif kind == "noise":
        noise_count = int(rng.integers(1, 3))
        for _ in range(noise_count):
            required.append(_noise_patch(rng, bg, _sample_color(rng, {bg})))
    elif kind != "empty":
        return None

    required.sort(key=lambda patch: -int(np.count_nonzero(patch != bg)))
    optional.sort(key=lambda patch: -int(np.count_nonzero(patch != bg)))

    for patch in required:
        if not _try_place_patch(block, blocked, patch, bg, rng):
            return None
    for patch in optional:
        _try_place_patch(block, blocked, patch, bg, rng)

    return block, int(expected_color)


def _plus_blocks(grid: np.ndarray, bg: int) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    block_h = H // 3
    block_w = W // 3
    out = np.zeros((3, 3), dtype=bool)

    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if int(g[r, c]) == bg:
                continue
            if all(int(g[r + dr, c + dc]) != bg for dr, dc in ORTHO_DIRS):
                out[r // block_h, c // block_w] = True
    return out


def _isolated_cross_blocks(grid: np.ndarray, bg: int, *, require_distinct_center: bool) -> np.ndarray:
    H, W = map(int, grid.shape)
    block_h = H // 3
    block_w = W // 3
    out = np.zeros((3, 3), dtype=bool)

    for comp in _components4_non_bg(grid, bg):
        marker = _cross_center(grid, comp, require_distinct_center=require_distinct_center)
        if marker is None:
            continue
        r, c, _ = marker
        out[r // block_h, c // block_w] = True
    return out


def _non_bg_blocks(grid: np.ndarray, bg: int) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    block_h = H // 3
    block_w = W // 3
    out = np.zeros((3, 3), dtype=bool)

    for br in range(3):
        for bc in range(3):
            block = g[br * block_h : (br + 1) * block_h, bc * block_w : (bc + 1) * block_w]
            out[br, bc] = bool(np.any(block != bg))
    return out


def _semantic_checks(grid: np.ndarray, expected: np.ndarray) -> bool:
    bg = _get_bg(grid)
    actual = _valid_marker_map(grid, bg)
    if actual is None or not np.array_equal(actual, expected):
        return False

    invalid_blocks = expected == bg
    if int(np.count_nonzero(expected != bg)) < 2:
        return False
    if not np.any(_non_bg_blocks(grid, bg) & invalid_blocks):
        return False
    if not np.any(_plus_blocks(grid, bg) & invalid_blocks):
        return False
    if not np.any(_isolated_cross_blocks(grid, bg, require_distinct_center=False) & invalid_blocks):
        return False
    if np.array_equal(_plus_blocks(grid, bg), expected != bg):
        return False
    if np.array_equal(_isolated_cross_blocks(grid, bg, require_distinct_center=False), expected != bg):
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    block_h = int(rng.integers(6, 10))
    block_w = int(rng.integers(6, 10))
    bg = int(rng.choice(PALETTE))

    grid = np.full((3 * block_h, 3 * block_w), bg, dtype=int)
    expected = np.full((3, 3), bg, dtype=int)

    block_kinds = ["valid", "valid", "mono", "embedded"]
    while len(block_kinds) < 9:
        block_kinds.append(str(rng.choice(["valid", "mono", "embedded", "noise", "empty"])))
    rng.shuffle(block_kinds)

    for idx, kind in enumerate(block_kinds):
        built = _build_block(kind, (block_h, block_w), bg, rng)
        if built is None:
            return None
        block, color = built
        br, bc = divmod(idx, 3)
        grid[br * block_h : (br + 1) * block_h, bc * block_w : (bc + 1) * block_w] = block
        expected[br, bc] = int(color)

    if not _semantic_checks(grid, expected):
        return None
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress a 3x3 coarse partition by reading the center color of each isolated bicolored cross.

    Input structure:
    - The background is the most common color, and the full grid height and width are both divisible by 3.
    - Each coarse block may contain scattered foreground components, including isolated 5-cell crosses, larger blobs with '+'-like centers, or unrelated small distractors.
    - A usable code mark is a 4-connected foreground component of exactly five cells shaped like a thin cross.

    Rule:
    - Search the whole grid for foreground components whose five cells are exactly one center plus its four orthogonal neighbors.
    - Keep only those crosses whose four arms all share one color and whose center has a different color.
    - Output a 3x3 grid: each coarse block containing one such valid mark contributes that mark's center color; every other coarse block outputs the background color.

    Key insight:
    - A local '+' neighborhood is not enough: embedded plus-centers fail the exact-5-cell component test, and isolated monochrome crosses fail the distinct-center-color test.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    H, W = map(int, g.shape)
    assert H % 3 == 0 and W % 3 == 0, "grid must split evenly into 3x3 coarse blocks"

    bg = _get_bg(g)
    out = _valid_marker_map(g, bg)
    assert out is not None, "valid markers must not collide within a coarse block"
    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GENERATION_RETRIES} attempts")
