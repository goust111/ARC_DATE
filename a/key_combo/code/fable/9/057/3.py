import numpy as np


BLACK = 0
BLUE = 1
RED = 2
INPUT_PALETTE = [3, 4, 5, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 400


def _is_lr_symmetric(mask: np.ndarray) -> bool:
    return bool(np.array_equal(mask, np.fliplr(mask)))


def _complete_symmetry(mask: np.ndarray) -> np.ndarray:
    return mask | np.fliplr(mask)


def _paint_output(mask: np.ndarray) -> np.ndarray:
    full = _complete_symmetry(mask)
    out = np.zeros(mask.shape, dtype=int)
    out[mask] = BLUE
    out[full & ~mask] = RED
    return out


def _mirror_col(width: int, col: int) -> int:
    return int(width - 1 - col)


def _find_scene(g: np.ndarray) -> tuple[int, int, np.ndarray, np.ndarray]:
    height, width = g.shape
    candidates: list[tuple[int, int, np.ndarray, np.ndarray]] = []

    for sep_col in range(width):
        if not np.all(g[:, sep_col] == g[0, sep_col]):
            continue

        left = g[:, :sep_col]
        right = g[:, sep_col + 1 :]
        if left.shape != right.shape or left.shape[1] == 0:
            continue

        sep_color = int(g[0, sep_col])
        left_colors = set(map(int, np.unique(left)))
        right_colors = set(map(int, np.unique(right)))
        shared = (left_colors & right_colors) - {sep_color}
        if len(shared) != 1:
            continue

        bg = int(next(iter(shared)))
        left_fg = left_colors - {bg, sep_color}
        right_fg = right_colors - {bg, sep_color}
        if len(left_fg) != 1 or len(right_fg) != 1:
            continue
        left_color = int(next(iter(left_fg)))
        right_color = int(next(iter(right_fg)))
        if left_color == right_color:
            continue

        left_mask = left == left_color
        right_mask = right == right_color
        if not left_mask.any() or not right_mask.any():
            continue
        if _is_lr_symmetric(left_mask) or _is_lr_symmetric(right_mask):
            continue

        union = left_mask | np.fliplr(right_mask)
        completed = _complete_symmetry(union)
        if np.array_equal(union, completed):
            continue
        if _is_lr_symmetric(union):
            continue

        candidates.append((sep_col, bg, left_mask, right_mask))

    assert len(candidates) == 1, "expected one valid separator scene"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold the right mask across the separator, overlay it with the left mask, and color the symmetric completion.

    Input structure:
    - A full-height separator column splits the grid into two equal-width halves.
    - Each half contains black background plus exactly one foreground mask color.
    - The left and right foreground colors differ, and neither half is left-right symmetric by itself.

    Rule:
    - Locate the separator column and split the grid into the left half and the right half.
    - Reflect the right-half mask left-right across the separator so it folds onto the left half.
    - Take the pixelwise union of the left mask and the folded-right mask.
    - Complete that union to the smallest left-right symmetric mask in the same frame.
    - Output one half-sized grid: existing union cells become blue, newly implied symmetric cells become red, and all other cells stay black.

    Key insight:
    - The symmetry does not belong to either visible half alone; it appears only after folding the right mask onto the left half and then filling the missing mirrored counterparts.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) > 0, "expected a non-empty 2D grid"

    _sep_col, _bg, left_mask, right_mask = _find_scene(g)
    union = left_mask | np.fliplr(right_mask)
    return _paint_output(union)


def _make_symmetric_mask(height: int, width: int, rng: np.random.Generator) -> np.ndarray | None:
    half = width // 2
    mask = np.zeros((height, width), dtype=bool)

    rect_count = int(rng.integers(2, 6))
    for _ in range(rect_count):
        rect_h = int(rng.integers(1, min(4, height) + 1))
        rect_w = int(rng.integers(1, min(3, max(1, half)) + 1))
        if rect_w > half:
            continue
        r0 = int(rng.integers(0, height - rect_h + 1))
        c0 = int(rng.integers(0, half - rect_w + 1))
        mask[r0 : r0 + rect_h, c0 : c0 + rect_w] = True
        mc0 = width - c0 - rect_w
        mask[r0 : r0 + rect_h, mc0 : mc0 + rect_w] = True

    if width % 2 == 1 and rng.random() < 0.7:
        center = width // 2
        for _ in range(int(rng.integers(1, 3))):
            seg_h = int(rng.integers(1, min(4, height) + 1))
            r0 = int(rng.integers(0, height - seg_h + 1))
            mask[r0 : r0 + seg_h, center] = True

    density = float(mask.mean())
    if not (0.16 <= density <= 0.48):
        return None
    if int(mask.sum()) < 10:
        return None
    if len(np.unique(np.argwhere(mask)[:, 0])) < 2:
        return None
    if len(np.unique(np.argwhere(mask)[:, 1])) < 4:
        return None
    return mask


def _choose_missing_cells(full: np.ndarray, rng: np.random.Generator) -> np.ndarray | None:
    height, width = full.shape
    side = "left" if rng.random() < 0.5 else "right"
    if side == "left":
        candidates = [(r, c) for r in range(height) for c in range(width // 2) if full[r, c]]
    else:
        start = (width + 1) // 2
        candidates = [(r, c) for r in range(height) for c in range(start, width) if full[r, c]]

    if len(candidates) < 2:
        return None

    target = int(rng.integers(2, min(6, len(candidates)) + 1))
    candidate_set = set(candidates)
    chosen: list[tuple[int, int]] = []
    used: set[tuple[int, int]] = set()

    seeds = list(candidates)
    rng.shuffle(seeds)
    while seeds and len(chosen) < target:
        seed = seeds.pop()
        if seed in used:
            continue
        cluster = [seed]
        neighbors = [
            (seed[0] - 1, seed[1]),
            (seed[0] + 1, seed[1]),
            (seed[0], seed[1] - 1),
            (seed[0], seed[1] + 1),
        ]
        rng.shuffle(neighbors)
        for cell in neighbors:
            if cell in candidate_set and cell not in used and rng.random() < 0.5 and len(chosen) + len(cluster) < target:
                cluster.append(cell)
        for cell in cluster:
            if cell not in used and len(chosen) < target:
                chosen.append(cell)
                used.add(cell)

    if len(chosen) < 2:
        return None

    union = full.copy()
    for r, c in chosen:
        mirror_c = _mirror_col(width, c)
        if not full[r, mirror_c]:
            return None
        union[r, c] = False

    if _is_lr_symmetric(union):
        return None
    if not np.array_equal(_complete_symmetry(union), full):
        return None
    return union


def _split_sources(union: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    coords = np.argwhere(union)
    if coords.shape[0] < 8:
        return None

    for _ in range(200):
        labels = rng.choice(3, size=coords.shape[0], p=[0.35, 0.35, 0.30])
        if len(set(map(int, labels))) < 3:
            continue

        left_mask = np.zeros_like(union, dtype=bool)
        folded_right = np.zeros_like(union, dtype=bool)
        for (r, c), label in zip(coords, labels):
            rr = int(r)
            cc = int(c)
            if label != 1:
                left_mask[rr, cc] = True
            if label != 0:
                folded_right[rr, cc] = True

        if not left_mask.any() or not folded_right.any():
            continue
        if not (left_mask & ~folded_right).any():
            continue
        if not (folded_right & ~left_mask).any():
            continue
        if _is_lr_symmetric(left_mask) or _is_lr_symmetric(folded_right):
            continue

        true_output = _paint_output(union)
        if np.array_equal(_paint_output(left_mask), true_output):
            continue
        if np.array_equal(_paint_output(folded_right), true_output):
            continue

        direct_or = left_mask | np.fliplr(folded_right)
        if np.array_equal(_paint_output(direct_or), true_output):
            continue
        if np.array_equal(direct_or, union):
            continue

        return left_mask, folded_right

    return None


def _build_scene(rng: np.random.Generator) -> np.ndarray | None:
    if rng.random() < 0.5:
        width = int(rng.integers(3, 7)) * 2
    else:
        width = int(rng.integers(2, 6)) * 2 + 1
    height = int(rng.integers(6, 13))

    full = _make_symmetric_mask(height, width, rng)
    if full is None:
        return None

    union = _choose_missing_cells(full, rng)
    if union is None:
        return None

    sources = _split_sources(union, rng)
    if sources is None:
        return None
    left_mask, folded_right = sources

    colors = rng.choice(INPUT_PALETTE, size=3, replace=False)
    left_color, right_color, sep_color = map(int, colors)
    grid = np.zeros((height, 2 * width + 1), dtype=int)
    grid[:, width] = sep_color
    grid[:, :width][left_mask] = left_color
    grid[:, width + 1 :][np.fliplr(folded_right)] = right_color

    if not _validate_scene(grid, union):
        return None
    return grid


def _validate_scene(grid: np.ndarray, expected_union: np.ndarray) -> bool:
    try:
        sep_col, bg, left_mask, right_mask = _find_scene(grid)
    except AssertionError:
        return False

    if bg != BLACK:
        return False
    if sep_col != expected_union.shape[1]:
        return False

    folded_union = left_mask | np.fliplr(right_mask)
    if not np.array_equal(folded_union, expected_union):
        return False
    if _is_lr_symmetric(folded_union):
        return False

    output = main(grid)
    expected_output = _paint_output(expected_union)
    if not np.array_equal(output, expected_output):
        return False

    direct_or = left_mask | right_mask
    if np.array_equal(_paint_output(direct_or), expected_output):
        return False
    if np.array_equal(_paint_output(left_mask), expected_output):
        return False
    if np.array_equal(_paint_output(np.fliplr(right_mask)), expected_output):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_scene(np.random.default_rng())
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetry-fold scene")
