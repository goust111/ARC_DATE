import numpy as np


BLACK = 0
RED = 2
YELLOW = 4
VISIBLE_FOREGROUND_COLORS = (1, 3, 5, 6, 7, 8, 9)
MAX_INPUT_SIDE = 15
GENERATION_RETRIES = 400
PLACEMENT_TRIES = 200


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _neighbor_count(mask: np.ndarray, r: int, c: int) -> int:
    h, w = mask.shape
    total = 0
    for nr, nc in _neighbors4(r, c):
        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
            total += 1
    return total


def _block_for_value(color: int, degree: int) -> np.ndarray:
    if degree <= 1:
        return np.array([[color, color], [color, color]], dtype=int)
    if degree == 2:
        return np.array([[color, RED], [RED, color]], dtype=int)
    if degree == 3:
        return np.array([[color, YELLOW], [color, color]], dtype=int)
    return np.array([[YELLOW, RED], [RED, YELLOW]], dtype=int)


def main(grid: np.ndarray) -> np.ndarray:
    """Upscale each foreground pixel to a 2x2 block chosen by its original 4-neighbor foreground degree.

    Input structure:
    - A black background contains one or more shapes drawn with foreground pixels.
    - Foreground connectivity is interpreted in the original grid using 4-neighbor adjacency.
    - The generator uses one visible foreground color, but the rule depends only on foreground-vs-background.

    Rule:
    - Double the grid height and width so each input cell maps to one 2x2 output block.
    - A black cell becomes an all-black 2x2 block.
    - A foreground cell counts how many of its four orthogonal neighbors are foreground in the original grid.
    - Degree 0 or 1 becomes a solid foreground 2x2; degree 2 becomes [[fg, red], [red, fg]];
      degree 3 becomes [[fg, yellow], [fg, fg]]; degree 4 becomes [[yellow, red], [red, yellow]].

    Key insight:
    - The output is not a uniform pixel expansion: local connectivity in the input decides which replacement
      texture a foreground pixel receives after scaling.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    h, w = g.shape
    fg_mask = g != BLACK
    out = np.zeros((2 * h, 2 * w), dtype=int)

    for r in range(h):
        for c in range(w):
            if not fg_mask[r, c]:
                continue
            degree = _neighbor_count(fg_mask, r, c)
            out[2 * r : 2 * r + 2, 2 * c : 2 * c + 2] = _block_for_value(int(g[r, c]), degree)
    return out


def _make_core_component(rng: np.random.Generator) -> np.ndarray:
    rect_h = int(rng.integers(3, 6))
    rect_w = int(rng.integers(3, 6))
    margin = 4
    mask = np.zeros((rect_h + 2 * margin, rect_w + 2 * margin), dtype=bool)
    r0 = c0 = margin
    mask[r0 : r0 + rect_h, c0 : c0 + rect_w] = True

    anchors = []
    if rect_w >= 3:
        for cc in range(1, rect_w - 1):
            anchors.append(("top", cc))
            anchors.append(("bottom", cc))
    if rect_h >= 3:
        for rr in range(1, rect_h - 1):
            anchors.append(("left", rr))
            anchors.append(("right", rr))

    tail_count = int(rng.integers(1, min(4, len(anchors)) + 1))
    picks = rng.choice(len(anchors), size=tail_count, replace=False)
    for idx in np.atleast_1d(picks):
        side, pos = anchors[int(idx)]
        length = int(rng.integers(2, 5))
        if side == "top":
            col = c0 + pos
            for dr in range(1, length + 1):
                mask[r0 - dr, col] = True
        elif side == "bottom":
            col = c0 + pos
            for dr in range(1, length + 1):
                mask[r0 + rect_h - 1 + dr, col] = True
        elif side == "left":
            row = r0 + pos
            for dc in range(1, length + 1):
                mask[row, c0 - dc] = True
        else:
            row = r0 + pos
            for dc in range(1, length + 1):
                mask[row, c0 + rect_w - 1 + dc] = True

    return _crop_mask(mask)


def _make_line_component(rng: np.random.Generator) -> np.ndarray:
    length = int(rng.integers(2, 6))
    if rng.random() < 0.5:
        return np.ones((1, length), dtype=bool)
    return np.ones((length, 1), dtype=bool)


def _make_l_component(rng: np.random.Generator) -> np.ndarray:
    h = int(rng.integers(2, 5))
    w = int(rng.integers(2, 5))
    mask = np.zeros((h, w), dtype=bool)
    mask[:, 0] = True
    mask[-1, :] = True
    return mask


def _make_t_component(rng: np.random.Generator) -> np.ndarray:
    stem = int(rng.integers(2, 5))
    arm = int(rng.integers(3, 6))
    mask = np.zeros((stem + 1, arm), dtype=bool)
    mask[0, :] = True
    mask[:, arm // 2] = True
    return mask


def _make_rectangle_component(rng: np.random.Generator) -> np.ndarray:
    h = int(rng.integers(2, 4))
    w = int(rng.integers(2, 5))
    return np.ones((h, w), dtype=bool)


def _random_component(rng: np.random.Generator) -> np.ndarray:
    factories = (
        _make_line_component,
        _make_l_component,
        _make_t_component,
        _make_rectangle_component,
    )
    mask = factories[int(rng.integers(0, len(factories)))](rng)
    rotations = int(rng.integers(0, 4))
    mask = np.rot90(mask, rotations)
    if rng.random() < 0.5:
        mask = np.fliplr(mask)
    if rng.random() < 0.5:
        mask = np.flipud(mask)
    return mask.copy()


def _can_place(occupied: np.ndarray, comp: np.ndarray, top: int, left: int, pad: int = 1) -> bool:
    ch, cw = comp.shape
    if top < 0 or left < 0 or top + ch > occupied.shape[0] or left + cw > occupied.shape[1]:
        return False

    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(occupied.shape[0], top + ch + pad)
    c1 = min(occupied.shape[1], left + cw + pad)
    expanded = occupied[r0:r1, c0:c1]

    local = np.zeros_like(expanded, dtype=bool)
    rr = top - r0
    cc = left - c0
    local[rr : rr + ch, cc : cc + cw] = comp
    return not np.any(expanded & local)


def _place_component(occupied: np.ndarray, comp: np.ndarray, rng: np.random.Generator) -> bool:
    ch, cw = comp.shape
    max_top = occupied.shape[0] - ch
    max_left = occupied.shape[1] - cw
    if max_top < 0 or max_left < 0:
        return False

    for _ in range(PLACEMENT_TRIES):
        top = int(rng.integers(0, max_top + 1))
        left = int(rng.integers(0, max_left + 1))
        if _can_place(occupied, comp, top, left):
            occupied[top : top + ch, left : left + cw] |= comp
            return True
    return False


def _degree_histogram(mask: np.ndarray) -> dict[int, int]:
    hist = {k: 0 for k in range(5)}
    for r, c in np.argwhere(mask):
        hist[_neighbor_count(mask, int(r), int(c))] += 1
    return hist


def _semantic_checks(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > MAX_INPUT_SIDE:
        return False

    nonzero = sorted(int(v) for v in np.unique(grid) if int(v) != BLACK)
    if len(nonzero) != 1 or nonzero[0] in (RED, YELLOW):
        return False

    mask = grid != BLACK
    hist = _degree_histogram(mask)
    if not all(hist[k] > 0 for k in range(5)):
        return False
    if hist[2] < 2 or hist[3] < 2:
        return False

    out = main(grid)
    if out.shape != (2 * grid.shape[0], 2 * grid.shape[1]):
        return False

    blocks = set()
    for r, c in np.argwhere(mask):
        block = tuple(out[2 * r : 2 * r + 2, 2 * c : 2 * c + 2].ravel().tolist())
        blocks.add(block)
    if len(blocks) != 4:
        return False

    isolated_count = 0
    endpoint_count = 0
    for r, c in np.argwhere(mask):
        degree = _neighbor_count(mask, int(r), int(c))
        isolated_count += degree == 0
        endpoint_count += degree == 1
    return isolated_count >= 1 and endpoint_count >= 1


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    core = _make_core_component(rng)
    extras = [np.array([[True]], dtype=bool)]
    extra_count = int(rng.integers(0, 3))
    for _ in range(extra_count):
        extras.append(_random_component(rng))

    min_h = max(8, core.shape[0] + 2)
    min_w = max(8, core.shape[1] + 2)
    if min_h > MAX_INPUT_SIDE or min_w > MAX_INPUT_SIDE:
        return None

    grid_h = int(rng.integers(min_h, MAX_INPUT_SIDE + 1))
    grid_w = int(rng.integers(min_w, MAX_INPUT_SIDE + 1))
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    for comp in [core, *extras]:
        if not _place_component(occupied, comp, rng):
            return None

    fg_color = int(rng.choice(np.array(VISIBLE_FOREGROUND_COLORS, dtype=int)))
    grid = np.where(occupied, fg_color, BLACK).astype(int)
    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 10")
