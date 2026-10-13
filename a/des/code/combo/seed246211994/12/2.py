import random

import numpy as np


def _boundary_background(grid: np.ndarray) -> int:
    border = np.concatenate(
        [grid[0], grid[-1], grid[1:-1, 0], grid[1:-1, -1]]
    )
    colors, counts = np.unique(border, return_counts=True)
    return int(colors[np.argmax(counts)])


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for idx, value in enumerate(mask.tolist()):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _find_legend(grid: np.ndarray, bg: int) -> tuple[list[tuple[int, int]], np.ndarray]:
    h, w = grid.shape
    assert w >= 4, "grid is too narrow for a side legend and target field"

    candidates: list[tuple[str, int, int]] = []
    for side in ("left", "right"):
        edge = grid[:, :2] if side == "left" else grid[:, -2:]
        row_mask = np.all(edge != bg, axis=1)
        sep_col = 2 if side == "left" else w - 3

        for r0, r1 in _true_runs(row_mask):
            if r1 - r0 < 2:
                continue
            if np.any(edge[:r0] != bg) or np.any(edge[r1:] != bg):
                continue
            if np.any(grid[r0:r1, sep_col] != bg):
                continue

            legend_mask = np.zeros_like(grid, dtype=bool)
            if side == "left":
                legend_mask[r0:r1, :2] = True
            else:
                legend_mask[r0:r1, -2:] = True

            if not np.any((grid != bg) & ~legend_mask):
                continue
            candidates.append((side, r0, r1))

    assert len(candidates) == 1, "expected exactly one side legend"
    side, r0, r1 = candidates[0]

    legend_mask = np.zeros_like(grid, dtype=bool)
    rules: list[tuple[int, int]] = []
    if side == "left":
        legend_mask[r0:r1, :2] = True
        for r in range(r0, r1):
            rules.append((int(grid[r, 0]), int(grid[r, 1])))
    else:
        legend_mask[r0:r1, -2:] = True
        for r in range(r0, r1):
            rules.append((int(grid[r, -2]), int(grid[r, -1])))

    return rules, legend_mask


def _apply_rules(grid: np.ndarray, rules: list[tuple[int, int]], legend_mask: np.ndarray, bg: int) -> np.ndarray:
    out = grid.copy()
    target_mask = (~legend_mask) & (grid != bg)
    for src, dst in rules:
        out[target_mask & (out == src)] = dst
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor non-background pixels by applying the side legend's replacement rules top-to-bottom.

    The legend is a vertical two-column strip on the left or right edge. Each row
    is a source->destination color rule. Pixels outside the legend are scanned
    against the full ordered rule list, so one pixel may be replaced multiple times.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _boundary_background(g)
    rules, legend_mask = _find_legend(g, bg)
    return _apply_rules(g, rules, legend_mask, bg)


MAX_GENERATION_ATTEMPTS = 400
GRID_H_RANGE = (9, 15)
GRID_W_RANGE = (13, 20)
RULE_COUNT_RANGE = (3, 5)
EXTRA_OBJECTS_RANGE = (1, 3)
PALETTE = tuple(range(10))


def _transform_color(color: int, rules: list[tuple[int, int]]) -> tuple[int, int]:
    current = int(color)
    steps = 0
    for src, dst in rules:
        if current == src:
            current = dst
            steps += 1
    return current, steps


def _rotate_and_flip(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    out = np.rot90(mask, rng.randrange(4))
    if rng.random() < 0.5:
        out = np.fliplr(out)
    return out.copy()


def _random_shape(rng: random.Random) -> np.ndarray:
    kind = rng.choice(["rect", "box", "L", "plus", "tee", "stair"])

    if kind == "rect":
        mask = np.ones((rng.randint(1, 3), rng.randint(2, 4)), dtype=bool)
    elif kind == "box":
        h, w = rng.randint(3, 4), rng.randint(3, 4)
        mask = np.ones((h, w), dtype=bool)
        mask[1:-1, 1:-1] = False
    elif kind == "L":
        h, w = rng.randint(2, 4), rng.randint(2, 4)
        mask = np.zeros((h, w), dtype=bool)
        mask[:, 0] = True
        mask[-1, :] = True
    elif kind == "plus":
        mask = np.array(
            [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0],
            ],
            dtype=bool,
        )
    elif kind == "tee":
        mask = np.array(
            [
                [1, 1, 1],
                [0, 1, 0],
                [0, 1, 0],
            ],
            dtype=bool,
        )
    else:
        steps = rng.randint(2, 4)
        mask = np.zeros((steps, steps + 1), dtype=bool)
        for i in range(steps):
            mask[i, i : i + 2] = True

    return _rotate_and_flip(mask, rng)


def _build_linear_rules(rng: random.Random, colors: list[int], n_rules: int) -> list[tuple[int, int]]:
    chain_len = rng.randint(2, n_rules)
    nodes = rng.sample(colors, chain_len + 1)
    rules = [(nodes[i], nodes[i + 1]) for i in range(chain_len)]

    remaining = [c for c in colors if c not in nodes]
    for src in rng.sample(remaining, n_rules - chain_len):
        dst = rng.choice([c for c in colors if c != src])
        rules.append((src, dst))
    return rules


def _build_merge_rules(rng: random.Random, colors: list[int], n_rules: int) -> list[tuple[int, int]]:
    if n_rules == 3:
        a, b, c, d = rng.sample(colors, 4)
        rules = [(a, c), (b, c), (c, d)]
        used = {a, b, c, d}
    else:
        a, b, c, d, e = rng.sample(colors, 5)
        rules = [(a, c), (b, c), (c, d), (d, e)]
        used = {a, b, c, d, e}

    remaining = [c for c in colors if c not in used]
    for src in rng.sample(remaining, n_rules - len(rules)):
        dst = rng.choice([c for c in colors if c != src])
        rules.append((src, dst))
    return rules


def _build_rules(rng: random.Random, bg: int, n_rules: int) -> list[tuple[int, int]]:
    colors = [c for c in PALETTE if c != bg]
    if rng.random() < 0.5:
        return _build_linear_rules(rng, colors, n_rules)
    return _build_merge_rules(rng, colors, n_rules)


def _legend_position(rng: random.Random, h: int, n_rules: int) -> tuple[int, int]:
    r0 = rng.randint(0, h - n_rules)
    return r0, r0 + n_rules


def _field_bounds(side: str, width: int) -> tuple[int, int]:
    if side == "left":
        return 3, width - 2
    return 2, width - 3


def _valid_positions(
    occ: np.ndarray, shape: np.ndarray, col_lo: int, col_hi: int
) -> list[tuple[int, int]]:
    h, w = occ.shape
    sh, sw = shape.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(0, h - sh + 1):
        for c0 in range(col_lo, col_hi - sw + 1):
            pr0 = max(0, r0 - 1)
            pr1 = min(h, r0 + sh + 1)
            pc0 = max(0, c0 - 1)
            pc1 = min(w, c0 + sw + 1)
            if not occ[pr0:pr1, pc0:pc1].any():
                positions.append((r0, c0))
    return positions


def _paint_shape(grid: np.ndarray, shape: np.ndarray, r0: int, c0: int, color: int) -> None:
    patch = grid[r0 : r0 + shape.shape[0], c0 : c0 + shape.shape[1]]
    patch[shape] = color


def _reserve(occ: np.ndarray, r0: int, c0: int, shape: np.ndarray) -> None:
    h, w = occ.shape
    sh, sw = shape.shape
    occ[max(0, r0 - 1) : min(h, r0 + sh + 1), max(0, c0 - 1) : min(w, c0 + sw + 1)] = True


def _make_legend(
    grid: np.ndarray, bg: int, rules: list[tuple[int, int]], side: str, r0: int
) -> np.ndarray:
    legend_mask = np.zeros_like(grid, dtype=bool)
    rows = range(r0, r0 + len(rules))
    if side == "left":
        for row, (src, dst) in zip(rows, rules):
            grid[row, 0] = src
            grid[row, 1] = dst
        legend_mask[r0 : r0 + len(rules), :2] = True
        grid[r0 : r0 + len(rules), 2] = bg
    else:
        for row, (src, dst) in zip(rows, rules):
            grid[row, -2] = src
            grid[row, -1] = dst
        legend_mask[r0 : r0 + len(rules), -2:] = True
        grid[r0 : r0 + len(rules), -3] = bg
    return legend_mask


def _required_object_colors(
    rng: random.Random, bg: int, rules: list[tuple[int, int]]
) -> list[int]:
    source_colors: list[int] = []
    for src, _dst in rules:
        if src not in source_colors:
            source_colors.append(src)

    distractors = [c for c in PALETTE if c != bg and c not in source_colors]
    assert distractors, "need at least one unaffected color"

    required = list(source_colors)
    required.append(rng.choice(distractors))

    stable_outputs = [
        dst
        for dst in {dst for _src, dst in rules}
        if _transform_color(dst, rules)[1] == 0 and dst not in required
    ]
    if stable_outputs and rng.random() < 0.7:
        required.append(rng.choice(stable_outputs))

    extras = rng.randint(*EXTRA_OBJECTS_RANGE)
    extra_palette = source_colors + distractors
    for _ in range(extras):
        required.append(rng.choice(extra_palette))

    rng.shuffle(required)
    return required


def _place_objects(
    rng: random.Random,
    grid: np.ndarray,
    occ: np.ndarray,
    colors: list[int],
    col_lo: int,
    col_hi: int,
) -> bool:
    for color in colors:
        shape = _random_shape(rng)
        positions = _valid_positions(occ, shape, col_lo, col_hi)
        if not positions:
            return False
        r0, c0 = rng.choice(positions)
        _paint_shape(grid, shape, r0, c0, color)
        _reserve(occ, r0, c0, shape)
    return True


def _background_is_dominant(grid: np.ndarray, bg: int) -> bool:
    colors, counts = np.unique(grid, return_counts=True)
    max_other = 0
    for color, count in zip(colors.tolist(), counts.tolist()):
        if color != bg:
            max_other = max(max_other, count)
    return int(grid.size > 0) and int(counts[colors.tolist().index(bg)]) > max_other


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    bg = rng.choice(PALETTE)
    n_rules = rng.randint(*RULE_COUNT_RANGE)
    side = rng.choice(["left", "right"])
    rules = _build_rules(rng, bg, n_rules)

    multi_step_sources = [src for src, _dst in rules if _transform_color(src, rules)[1] >= 2]
    if not multi_step_sources:
        return None

    grid = np.full((h, w), bg, dtype=int)
    legend_r0, legend_r1 = _legend_position(rng, h, n_rules)
    legend_mask = _make_legend(grid, bg, rules, side, legend_r0)

    occ = np.zeros((h, w), dtype=bool)
    occ[:, :2] = side == "right"
    occ[:, -2:] = side == "left"
    if side == "left":
        occ[legend_r0:legend_r1, :3] = True
    else:
        occ[legend_r0:legend_r1, -3:] = True

    col_lo, col_hi = _field_bounds(side, w)
    object_colors = _required_object_colors(rng, bg, rules)
    if not _place_objects(rng, grid, occ, object_colors, col_lo, col_hi):
        return None

    if not _background_is_dominant(grid, bg):
        return None

    out = _apply_rules(grid, rules, legend_mask, bg)
    changed = ((grid != out) & ~legend_mask).sum()
    unchanged_fg = ((grid == out) & (grid != bg) & ~legend_mask).sum()
    if changed == 0 or unchanged_fg == 0:
        return None

    if not any(
        ((grid == color) & ~legend_mask).any() and _transform_color(color, rules)[1] >= 2
        for color in multi_step_sources
    ):
        return None

    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
