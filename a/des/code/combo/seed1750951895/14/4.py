import random

import numpy as np


BG = 0
LEGEND_W = 2
SEP_W = 1
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
MAX_ATTEMPTS = 300


def _separator_col(grid: np.ndarray) -> int:
    h, w = grid.shape
    for c in range(w):
        if np.all(grid[:, c] == BG) and np.any(grid[:, c + 1 :] != BG):
            return c
    raise AssertionError("expected an all-background separator column before the canvas")


def _read_rules(grid: np.ndarray) -> list[tuple[int, int]]:
    sep = _separator_col(grid)
    assert sep >= LEGEND_W, "legend must be at least two columns wide"

    rules: list[tuple[int, int]] = []
    for r in range(grid.shape[0]):
        src = int(grid[r, 0])
        dst = int(grid[r, 1])
        if src == BG and dst == BG:
            continue
        assert src != BG and dst != BG, "legend rows must contain two foreground colors"
        rules.append((src, dst))

    assert rules, "expected at least one replacement rule"
    return rules


def _apply_rules(canvas: np.ndarray, rules: list[tuple[int, int]]) -> np.ndarray:
    out = canvas.copy()
    for src, dst in rules:
        out[out == src] = dst
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Apply the ordered two-column legend as a chain of color substitutions to the right canvas.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep = _separator_col(g)
    rules = _read_rules(g)
    canvas = g[:, sep + SEP_W :]
    assert canvas.size > 0, "canvas must exist to the right of the separator"

    return _apply_rules(canvas, rules)


def _rect_free(occ: np.ndarray, r0: int, c0: int, h: int, w: int, margin: int = 1) -> bool:
    rr0 = max(0, r0 - margin)
    cc0 = max(0, c0 - margin)
    rr1 = min(occ.shape[0], r0 + h + margin)
    cc1 = min(occ.shape[1], c0 + w + margin)
    return not bool(occ[rr0:rr1, cc0:cc1].any())


def _place_rect(
    canvas: np.ndarray,
    occ: np.ndarray,
    rng: random.Random,
    color: int,
    min_h: int = 1,
    max_h: int = 4,
    min_w: int = 1,
    max_w: int = 4,
) -> bool:
    rows = list(range(canvas.shape[0]))
    cols = list(range(canvas.shape[1]))
    rng.shuffle(rows)
    rng.shuffle(cols)

    for r0 in rows:
        for c0 in cols:
            h = rng.randint(min_h, min(max_h, canvas.shape[0] - r0))
            w = rng.randint(min_w, min(max_w, canvas.shape[1] - c0))
            if not _rect_free(occ, r0, c0, h, w):
                continue
            canvas[r0 : r0 + h, c0 : c0 + w] = color
            occ[r0 : r0 + h, c0 : c0 + w] = True
            return True
    return False


def _sample_rules(rng: random.Random) -> tuple[list[tuple[int, int]], list[int], int]:
    chain_len = rng.choice([2, 3])
    color_count = rng.randint(chain_len + 3, min(8, len(PALETTE)))
    colors = rng.sample(PALETTE, color_count)

    chain = colors[: chain_len + 1]
    rules = [(chain[i], chain[i + 1]) for i in range(chain_len)]

    extras: list[tuple[int, int]] = []
    extra_rule_count = rng.randint(0, 3)
    while len(extras) < extra_rule_count:
        src, dst = rng.sample(colors, 2)
        pair = (src, dst)
        if pair in rules or pair in extras:
            continue
        extras.append(pair)

    insert_at = rng.randint(0, len(extras))
    ordered = extras[:insert_at] + rules + extras[insert_at:]

    final_colors = list({dst for _, dst in ordered} | {src for src, _ in ordered})
    unaffected_choices = [c for c in PALETTE if c not in {src for src, _ in ordered}]
    if not unaffected_choices:
        unaffected_choices = [chain[-1]]
    unaffected = rng.choice(unaffected_choices)
    return ordered, chain, unaffected


def _build_canvas(
    rng: random.Random,
    h: int,
    w: int,
    rules: list[tuple[int, int]],
    chain: list[int],
    unaffected: int,
) -> np.ndarray | None:
    canvas = np.zeros((h, w), dtype=int)
    occ = np.zeros((h, w), dtype=bool)

    required_colors = [chain[0], chain[1], unaffected]
    if len(chain) >= 4:
        required_colors.append(chain[2])

    for color in required_colors:
        if not _place_rect(canvas, occ, rng, color):
            return None

    src_colors = sorted({src for src, _ in rules})
    optional_pool = src_colors + [dst for _, dst in rules] + [unaffected]
    extra_count = rng.randint(1, 4)
    for _ in range(extra_count):
        color = rng.choice(optional_pool)
        _place_rect(canvas, occ, rng, color)

    if not np.any(canvas == chain[0]):
        return None
    if not np.any(canvas == chain[1]):
        return None
    if not np.any(canvas == unaffected):
        return None

    out = _apply_rules(canvas, rules)
    if np.array_equal(out, canvas):
        return None
    if np.any(out == chain[0]):
        return None
    if out.shape != canvas.shape:
        return None
    return canvas


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    rule_count_hint = rng.randint(2, 6)
    grid_h = max(rule_count_hint + rng.randint(0, 4), 5)
    canvas_h = grid_h
    canvas_w = rng.randint(5, 12)

    rules, chain, unaffected = _sample_rules(rng)
    if len(rules) > grid_h:
        return None

    canvas = _build_canvas(rng, canvas_h, canvas_w, rules, chain, unaffected)
    if canvas is None:
        return None

    grid = np.zeros((grid_h, LEGEND_W + SEP_W + canvas_w), dtype=int)
    legend_rows = rng.sample(range(grid_h), len(rules))
    legend_rows.sort()
    for r, (src, dst) in zip(legend_rows, rules):
        grid[r, 0] = src
        grid[r, 1] = dst
    grid[:, LEGEND_W] = BG
    grid[:, LEGEND_W + SEP_W :] = canvas

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != (grid_h, canvas_w):
        return None
    if np.any(out == chain[0]):
        return None
    if not np.any(out == chain[-1]):
        return None
    if not np.any(out == unaffected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
