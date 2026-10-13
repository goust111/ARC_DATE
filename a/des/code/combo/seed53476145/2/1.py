import random

import numpy as np


BG = 0
SEP = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
SLOT_PITCH = 6
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_separator_row(grid: np.ndarray) -> int:
    for r in range(1, grid.shape[0] - 1):
        if np.all(grid[r] == SEP):
            return r
    raise AssertionError("expected a full-width grey separator row")


def _parse_rules(legend: np.ndarray) -> list[tuple[int, int]]:
    active_rows = np.flatnonzero(np.any(legend != BG, axis=1))
    assert active_rows.size == 2, "legend must occupy exactly two non-empty rows"
    band = legend[active_rows, :]

    rules: list[tuple[int, int]] = []
    c = 0
    while c < band.shape[1]:
        if np.all(band[:, c] == BG):
            c += 1
            continue

        start = c
        while c < band.shape[1] and not np.all(band[:, c] == BG):
            assert band[0, c] == band[1, c]
            c += 1

        run = band[:, start:c]
        assert run.shape[1] % 2 == 0, "legend columns must form 2-column source/target pairs"
        for j in range(0, run.shape[1], 2):
            src = int(run[0, j])
            dst = int(run[0, j + 1])
            assert src != BG and dst != BG
            rules.append((src, dst))

    assert rules, "legend must contain at least one color-replacement pair"
    return rules


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _apply_chain(color: int, rules: list[tuple[int, int]]) -> int:
    value = int(color)
    for src, dst in rules:
        if value == src:
            value = dst
    return value


def _transform_scene(scene: np.ndarray, rules: list[tuple[int, int]]) -> np.ndarray:
    out = scene.copy()
    mask = scene != BG
    h, w = scene.shape

    for comp in _components(mask):
        for r, c in comp:
            is_border = False
            for nr, nc in _neighbors4(r, c):
                if not (0 <= nr < h and 0 <= nc < w) or scene[nr, nc] == BG:
                    is_border = True
                    break
            if is_border:
                out[r, c] = _apply_chain(int(scene[r, c]), rules)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Apply the ordered legend replacements only to sprite border pixels, then drop the legend panel.
    """
    g = np.asarray(grid, dtype=int)
    sep_row = _find_separator_row(g)
    rules = _parse_rules(g[:sep_row, :])
    scene = g[sep_row + 1 :, :]
    assert scene.size > 0
    return _transform_scene(scene, rules)


def _mask_interiors(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    interiors: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            if all(0 <= nr < h and 0 <= nc < w and mask[nr, nc] for nr, nc in _neighbors4(r, c)):
                interiors.append((r, c))
    return interiors


def _plus_mask() -> np.ndarray:
    mask = np.zeros((5, 5), dtype=bool)
    mask[1:4, :] = True
    mask[:, 1:4] = True
    return mask


def _diamond_mask() -> np.ndarray:
    mask = np.zeros((5, 5), dtype=bool)
    for r in range(5):
        for c in range(5):
            if abs(r - 2) + abs(c - 2) <= 2:
                mask[r, c] = True
    return mask


def _uniform_rect(rng: random.Random, color: int, min_side: int, max_side: int) -> np.ndarray:
    h = rng.randint(min_side, max_side)
    w = rng.randint(min_side, max_side)
    return np.full((h, w), int(color), dtype=int)


def _all_border_rect(rng: random.Random, color: int) -> np.ndarray:
    h, w = rng.choice(((2, 2), (2, 3), (3, 2), (2, 4), (4, 2)))
    return np.full((h, w), int(color), dtype=int)


def _border_inner_sprite(
    rng: random.Random,
    border_color: int,
    inner_color: int,
    shape_name: str | None = None,
) -> np.ndarray:
    if shape_name is None:
        shape_name = rng.choice(("rect", "plus", "diamond"))

    if shape_name == "rect":
        mask = np.ones((rng.randint(3, 5), rng.randint(3, 5)), dtype=bool)
    elif shape_name == "plus":
        mask = _plus_mask()
    else:
        mask = _diamond_mask()

    interiors = _mask_interiors(mask)
    assert interiors

    sprite = np.zeros(mask.shape, dtype=int)
    sprite[mask] = int(border_color)
    for r, c in interiors:
        sprite[r, c] = int(inner_color)
    return sprite


def _split_rect_sprite(rng: random.Random, color_a: int, color_b: int) -> np.ndarray:
    h = rng.randint(3, 5)
    w = rng.randint(3, 5)
    sprite = np.zeros((h, w), dtype=int)
    if rng.random() < 0.5:
        cut = rng.randint(1, w - 1)
        sprite[:, :cut] = int(color_a)
        sprite[:, cut:] = int(color_b)
    else:
        cut = rng.randint(1, h - 1)
        sprite[:cut, :] = int(color_a)
        sprite[cut:, :] = int(color_b)
    return sprite


def _mask_to_sprite(mask: np.ndarray, color: int) -> np.ndarray:
    sprite = np.zeros(mask.shape, dtype=int)
    sprite[mask] = int(color)
    return sprite


def _random_sprite(
    rng: random.Random,
    source_colors: list[int],
    non_source_colors: list[int],
) -> np.ndarray:
    all_colors = source_colors + [c for c in non_source_colors if c not in source_colors]
    kind = rng.choice(("uniform_rect", "all_border", "border_inner", "split_rect", "plus", "diamond"))

    if kind == "uniform_rect":
        return _uniform_rect(rng, rng.choice(all_colors), 3, 5)
    if kind == "all_border":
        return _all_border_rect(rng, rng.choice(all_colors))
    if kind == "border_inner":
        border = rng.choice(all_colors)
        inner = rng.choice([c for c in all_colors if c != border])
        return _border_inner_sprite(rng, border, inner)
    if kind == "split_rect":
        a = rng.choice(all_colors)
        b = rng.choice([c for c in all_colors if c != a])
        return _split_rect_sprite(rng, a, b)
    if kind == "plus":
        return _mask_to_sprite(_plus_mask(), rng.choice(all_colors))
    return _border_inner_sprite(rng, rng.choice(all_colors), rng.choice(non_source_colors), "diamond")


def _make_rules(
    rng: random.Random,
) -> tuple[list[tuple[int, int]], list[int], int, list[int], list[int]]:
    n_rules = rng.randint(2, 4)
    chain_len = 2 if n_rules == 2 else rng.choice((2, 3))
    chain_len = min(chain_len, n_rules)

    chain_nodes = rng.sample(list(COLOR_POOL), chain_len + 1)
    chain_rules = [(chain_nodes[i], chain_nodes[i + 1]) for i in range(chain_len)]

    extra_needed = n_rules - chain_len
    extra_source_pool = [c for c in COLOR_POOL if c not in set(chain_nodes)]
    extra_sources = rng.sample(extra_source_pool, extra_needed)
    all_sources = set(chain_nodes[:-1]) | set(extra_sources)

    extra_rules: list[tuple[int, int]] = []
    for src in extra_sources:
        candidates = [c for c in COLOR_POOL if c != src and c not in all_sources]
        extra_rules.append((src, rng.choice(candidates)))

    rng.shuffle(extra_rules)
    split = rng.randint(0, len(extra_rules))
    rules = extra_rules[:split] + chain_rules + extra_rules[split:]

    source_colors = [src for src, _ in rules]
    non_source_colors = [c for c in COLOR_POOL if c not in set(source_colors)]
    return rules, chain_nodes, chain_len, source_colors, non_source_colors


def _legend_span(rule_count: int) -> int:
    return 2 * rule_count + max(0, rule_count - 1)


def _build_legend(rules: list[tuple[int, int]], width: int, rng: random.Random) -> np.ndarray:
    legend = np.zeros((2, width), dtype=int)
    span = _legend_span(len(rules))
    start = rng.randint(0, width - span)
    c = start
    for i, (src, dst) in enumerate(rules):
        legend[:, c] = int(src)
        legend[:, c + 1] = int(dst)
        c += 2
        if i != len(rules) - 1:
            c += 1
    return legend


def _candidate_layouts(n_sprites: int, min_width: int) -> list[tuple[int, int]]:
    layouts: list[tuple[int, int]] = []
    for slot_rows in (2, 3):
        for slot_cols in (2, 3, 4):
            if slot_rows * slot_cols < n_sprites:
                continue
            width = 1 + SLOT_PITCH * slot_cols
            if width < min_width:
                continue
            layouts.append((slot_rows, slot_cols))
    return layouts


def _place_sprites(
    sprites: list[np.ndarray],
    scene_h: int,
    scene_w: int,
    slot_rows: int,
    slot_cols: int,
    rng: random.Random,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    scene = np.zeros((scene_h, scene_w), dtype=int)
    slots = [(sr, sc) for sr in range(slot_rows) for sc in range(slot_cols)]
    rng.shuffle(slots)
    chosen_slots = slots[: len(sprites)]
    placements: list[tuple[int, int]] = []

    for sprite, (slot_r, slot_c) in zip(sprites, chosen_slots):
        sh, sw = sprite.shape
        top = 1 + slot_r * SLOT_PITCH + rng.randint(0, 5 - sh)
        left = 1 + slot_c * SLOT_PITCH + rng.randint(0, 5 - sw)
        mask = sprite != BG
        patch = scene[top : top + sh, left : left + sw]
        assert not np.any(patch[mask] != BG)
        patch[mask] = sprite[mask]
        placements.append((top, left))

    return scene, placements


def _required_sprites(
    rng: random.Random,
    chain_nodes: list[int],
    chain_len: int,
    source_colors: list[int],
    non_source_colors: list[int],
) -> tuple[list[np.ndarray], int, int]:
    chain_start = chain_nodes[0]
    one_step_source = chain_nodes[chain_len - 1]
    preserved_source = rng.choice(chain_nodes[:-1])
    border_color = rng.choice(non_source_colors)

    sprites = [
        _uniform_rect(rng, chain_start, 3, 5),
        _border_inner_sprite(rng, border_color, preserved_source),
        _all_border_rect(rng, one_step_source),
    ]

    split_a = rng.choice(source_colors)
    split_b = rng.choice([c for c in COLOR_POOL if c != split_a])
    sprites.append(_split_rect_sprite(rng, split_a, split_b))
    return sprites, chain_start, one_step_source


def generate_input() -> np.ndarray:
    rng = random.Random()

    for _ in range(MAX_GENERATION_ATTEMPTS):
        rules, chain_nodes, chain_len, source_colors, non_source_colors = _make_rules(rng)
        n_sprites = rng.randint(4, 6)
        required, chain_start, one_step_source = _required_sprites(
            rng, chain_nodes, chain_len, source_colors, non_source_colors
        )

        sprites = required[:]
        while len(sprites) < n_sprites:
            sprites.append(_random_sprite(rng, source_colors, non_source_colors))

        legend_width = _legend_span(len(rules))
        layouts = _candidate_layouts(n_sprites, legend_width)
        if not layouts:
            continue
        slot_rows, slot_cols = rng.choice(layouts)
        scene_h = 1 + SLOT_PITCH * slot_rows + rng.randint(0, 3)
        scene_w = 1 + SLOT_PITCH * slot_cols + rng.randint(0, 3)

        scene, placements = _place_sprites(sprites, scene_h, scene_w, slot_rows, slot_cols, rng)
        legend = _build_legend(rules, scene_w, rng)
        grid = np.vstack([legend, np.full((1, scene_w), SEP, dtype=int), scene])

        out = main(grid)
        if out.shape != scene.shape or not np.any(out != scene):
            continue

        top0, left0 = placements[0]
        sprite0 = sprites[0]
        interior0 = _mask_interiors(np.ones(sprite0.shape, dtype=bool))
        if not interior0:
            continue
        ir, ic = interior0[0]
        preserved_before = int(scene[top0 + ir, left0 + ic])
        preserved_after = int(out[top0 + ir, left0 + ic])
        if preserved_before != chain_start or preserved_after != preserved_before:
            continue

        changed_before = int(scene[top0, left0])
        changed_after = int(out[top0, left0])
        if changed_before != chain_start or changed_after != _apply_chain(chain_start, rules):
            continue
        if changed_after == changed_before:
            continue

        top2, left2 = placements[2]
        if int(scene[top2, left2]) != one_step_source:
            continue
        if int(out[top2, left2]) != _apply_chain(one_step_source, rules):
            continue

        return grid

    raise RuntimeError("failed to generate a valid input")
