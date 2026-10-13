from __future__ import annotations

import numpy as np


BG = 0
GREY = 5
MAROON = 9
ROOT_COLORS = (1, 2, 3, 4, 6, 7, 8)
BRANCH_MAX = 3
GEN_ATTEMPTS = 300


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _max_branches(levels: list[tuple[int, int]]) -> tuple[int, int]:
    if not levels:
        return 0, 0
    return max(left for left, _right in levels), max(right for _left, right in levels)


def _reserve_rels(
    old_levels: list[tuple[int, int]],
    template_levels: list[tuple[int, int]],
    attach: int,
    pad: int = 1,
) -> tuple[int, int, int, int]:
    max_h = max(len(old_levels), len(template_levels))
    old_left, old_right = _max_branches(old_levels)
    tpl_left, tpl_right = _max_branches(template_levels)
    max_left = max(old_left, tpl_left)
    max_right = max(old_right, tpl_right)
    min_r_rel = -max_h - pad
    max_r_rel = 1 + pad
    min_c_rel = min(0, attach - max_left) - pad
    max_c_rel = max(1, attach + max_right) + pad
    return min_r_rel, max_r_rel, min_c_rel, max_c_rel


def _fill_span(grid: np.ndarray, row: int, c0: int, c1: int, value: int) -> None:
    if not (0 <= row < grid.shape[0]):
        return
    cc0 = max(0, c0)
    cc1 = min(grid.shape[1], c1)
    if cc0 < cc1:
        grid[row, cc0:cc1] = int(value)


def _detect_trunk_column(grid: np.ndarray, root_r: int, root_c: int) -> int:
    best: tuple[int, int, int] | None = None
    best_col = -1
    for attach in (0, 1):
        trunk_c = root_c + attach
        immediate = int(root_r > 0 and grid[root_r - 1, trunk_c] == GREY)
        height = 0
        rr = root_r - 1
        while rr >= 0 and grid[rr, trunk_c] == GREY:
            height += 1
            rr -= 1
        cand = (immediate, height, -attach)
        if best is None or cand > best:
            best = cand
            best_col = trunk_c
    assert best is not None and best[0] == 1
    return int(best_col)


def _read_levels(grid: np.ndarray, root_r: int, trunk_c: int) -> list[tuple[int, int]]:
    levels: list[tuple[int, int]] = []
    row = root_r - 1
    while row >= 0 and grid[row, trunk_c] == GREY:
        left = 0
        cc = trunk_c - 1
        while cc >= 0 and grid[row, cc] == GREY:
            left += 1
            cc -= 1
        right = 0
        cc = trunk_c + 1
        while cc < grid.shape[1] and grid[row, cc] == GREY:
            right += 1
            cc += 1
        levels.append((left, right))
        row -= 1
    return levels


def _rewrite_tree(
    out: np.ndarray,
    root_r: int,
    root_c: int,
    attach: int,
    color: int,
    old_levels: list[tuple[int, int]],
    template_levels: list[tuple[int, int]],
) -> None:
    trunk_c = root_c + attach
    old_h = len(old_levels)
    tpl_h = len(template_levels)

    for i in range(min(old_h, tpl_h)):
        row = root_r - 1 - i
        old_left, old_right = old_levels[i]
        tpl_left, tpl_right = template_levels[i]

        if tpl_left < old_left:
            _fill_span(out, row, trunk_c - old_left, trunk_c - tpl_left, BG)
        elif tpl_left > old_left:
            _fill_span(out, row, trunk_c - tpl_left, trunk_c - old_left, color)

        if tpl_right < old_right:
            _fill_span(out, row, trunk_c + tpl_right + 1, trunk_c + old_right + 1, BG)
        elif tpl_right > old_right:
            _fill_span(out, row, trunk_c + old_right + 1, trunk_c + tpl_right + 1, color)

    for i in range(tpl_h, old_h):
        row = root_r - 1 - i
        old_left, old_right = old_levels[i]
        _fill_span(out, row, trunk_c - old_left, trunk_c + old_right + 1, BG)

    for i in range(old_h, tpl_h):
        row = root_r - 1 - i
        tpl_left, tpl_right = template_levels[i]
        if 0 <= row < out.shape[0] and 0 <= trunk_c < out.shape[1]:
            out[row, trunk_c] = GREY
        _fill_span(out, row, trunk_c - tpl_left, trunk_c, color)
        _fill_span(out, row, trunk_c + 1, trunk_c + tpl_right + 1, color)


def _find_template_frame(grid: np.ndarray) -> tuple[int, int, int, int]:
    candidates: list[tuple[int, int, int, int]] = []
    for coords in _components(grid == MAROON):
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 4 or w < 4:
            continue
        perimeter = 2 * (h + w) - 4
        if coords.shape[0] != perimeter:
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not np.all(sub[0, :] == MAROON):
            continue
        if not np.all(sub[-1, :] == MAROON):
            continue
        if not np.all(sub[:, 0] == MAROON):
            continue
        if not np.all(sub[:, -1] == MAROON):
            continue
        candidates.append((r0, c0, r1, c1))
    assert candidates
    return max(candidates, key=lambda box: (box[2] + box[3], box[0], box[1]))


def _extract_template_levels(grid: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    fr0, fc0, fr1, fc1 = frame_bbox
    inner = grid[fr0 + 1 : fr1, fc0 + 1 : fc1]
    root_candidates: list[tuple[int, int]] = []
    for coords in _components(inner == MAROON):
        r0, c0, r1, c1 = _bbox(coords)
        if coords.shape[0] == 4 and r1 - r0 == 1 and c1 - c0 == 1:
            root_candidates.append((int(fr0 + 1 + r0), int(fc0 + 1 + c0)))
    assert len(root_candidates) == 1
    root_r, root_c = root_candidates[0]
    trunk_c = _detect_trunk_column(grid, root_r, root_c)
    return _read_levels(grid, root_r, trunk_c)


def _find_roots(grid: np.ndarray) -> list[tuple[int, int, int]]:
    roots: list[tuple[int, int, int]] = []
    for color in np.unique(grid):
        color = int(color)
        if color in (BG, GREY, MAROON):
            continue
        for coords in _components(grid == color):
            r0, c0, r1, c1 = _bbox(coords)
            if coords.shape[0] == 4 and r1 - r0 == 1 and c1 - c0 == 1:
                roots.append((r0, c0, color))
    roots.sort()
    return roots


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rewrite each tree's branch pattern to match the maroon-framed template tree.

    Ordinary trees have a 2x2 colored root, a 1-pixel grey trunk, and horizontal
    grey branches. Extra top levels are removed, missing levels are added, and
    changed branch pixels are painted with the root color. The template area is
    cleared to the black background.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    frame_bbox = _find_template_frame(g)
    template_levels = _extract_template_levels(g, frame_bbox)
    fr0, fc0, fr1, fc1 = frame_bbox
    out[fr0 : fr1 + 1, fc0 : fc1 + 1] = BG

    for root_r, root_c, color in _find_roots(g):
        trunk_c = _detect_trunk_column(g, root_r, root_c)
        attach = trunk_c - root_c
        old_levels = _read_levels(g, root_r, trunk_c)
        _rewrite_tree(out, root_r, root_c, attach, color, old_levels, template_levels)

    return out


def _draw_tree(
    grid: np.ndarray,
    root_r: int,
    root_c: int,
    color: int,
    attach: int,
    levels: list[tuple[int, int]],
) -> None:
    trunk_c = root_c + attach
    grid[root_r : root_r + 2, root_c : root_c + 2] = int(color)
    for i, (left, right) in enumerate(levels):
        row = root_r - 1 - i
        grid[row, trunk_c] = GREY
        if left > 0:
            grid[row, trunk_c - left : trunk_c] = GREY
        if right > 0:
            grid[row, trunk_c + 1 : trunk_c + right + 1] = GREY


def _build_template_patch(
    levels: list[tuple[int, int]],
    attach: int,
) -> np.ndarray:
    max_left, max_right = _max_branches(levels)
    min_rel = min(0, attach - max_left)
    max_rel = max(1, attach + max_right)
    tree_h = len(levels) + 2
    tree_w = max_rel - min_rel + 1
    inner = np.full((tree_h + 2, tree_w + 2), BG, dtype=int)
    root_r = 1 + len(levels)
    root_c = 1 - min_rel
    _draw_tree(inner, root_r, root_c, MAROON, attach, levels)

    out = np.full((inner.shape[0] + 2, inner.shape[1] + 2), BG, dtype=int)
    out[0, :] = MAROON
    out[-1, :] = MAROON
    out[:, 0] = MAROON
    out[:, -1] = MAROON
    out[1:-1, 1:-1] = inner
    return out


def _sample_levels(rng: np.random.Generator, height: int) -> list[tuple[int, int]]:
    levels: list[tuple[int, int]] = []
    for _ in range(height):
        left = int(rng.integers(0, BRANCH_MAX + 1))
        right = int(rng.integers(0, BRANCH_MAX + 1))
        if left == 0 and right == 0:
            if rng.random() < 0.5:
                left = int(rng.integers(1, BRANCH_MAX + 1))
            else:
                right = int(rng.integers(1, BRANCH_MAX + 1))
        levels.append((left, right))
    if height > 1 and len(set(levels)) == 1:
        idx = int(rng.integers(0, height))
        base_left, base_right = levels[idx]
        while True:
            left = int(rng.integers(0, BRANCH_MAX + 1))
            right = int(rng.integers(0, BRANCH_MAX + 1))
            if (left, right) != (base_left, base_right) and (left > 0 or right > 0):
                levels[idx] = (left, right)
                break
    return levels


def _force_difference(
    rng: np.random.Generator,
    levels: list[tuple[int, int]],
    template_levels: list[tuple[int, int]],
) -> None:
    overlap = min(len(levels), len(template_levels))
    if overlap == 0:
        return
    if any(levels[i] != template_levels[i] for i in range(overlap)):
        return
    idx = int(rng.integers(0, overlap))
    tpl_left, tpl_right = template_levels[idx]
    while True:
        left = int(rng.integers(0, BRANCH_MAX + 1))
        right = int(rng.integers(0, BRANCH_MAX + 1))
        if (left, right) != (tpl_left, tpl_right) and (left > 0 or right > 0):
            levels[idx] = (left, right)
            return


def _sample_tree_levels(
    rng: np.random.Generator,
    template_levels: list[tuple[int, int]],
    category: str,
) -> list[tuple[int, int]]:
    tpl_h = len(template_levels)
    if category == "shorter":
        height = int(rng.integers(1, tpl_h))
    elif category == "equal":
        height = tpl_h
    else:
        height = tpl_h + int(rng.integers(1, 3))
    levels = _sample_levels(rng, height)
    if category == "equal" or rng.random() < 0.7:
        _force_difference(rng, levels, template_levels)
    return levels


def _place_tree_root(
    rng: np.random.Generator,
    occupied: np.ndarray,
    template_levels: list[tuple[int, int]],
    spec: dict[str, object],
) -> tuple[int, int] | None:
    old_levels = spec["levels"]
    attach = int(spec["attach"])
    min_r_rel, max_r_rel, min_c_rel, max_c_rel = _reserve_rels(old_levels, template_levels, attach)
    h, w = occupied.shape
    root_r_min = -min_r_rel
    root_r_max = h - 1 - max_r_rel
    root_c_min = -min_c_rel
    root_c_max = w - 1 - max_c_rel
    if root_r_min > root_r_max or root_c_min > root_c_max:
        return None
    for _ in range(400):
        root_r = int(rng.integers(root_r_min, root_r_max + 1))
        root_c = int(rng.integers(root_c_min, root_c_max + 1))
        r0 = root_r + min_r_rel
        r1 = root_r + max_r_rel
        c0 = root_c + min_c_rel
        c1 = root_c + max_c_rel
        if occupied[r0 : r1 + 1, c0 : c1 + 1].any():
            continue
        occupied[r0 : r1 + 1, c0 : c1 + 1] = True
        return root_r, root_c
    return None


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    template_h = int(rng.integers(2, 5))
    template_levels = _sample_levels(rng, template_h)
    template_attach = int(rng.integers(0, 2))
    template_patch = _build_template_patch(template_levels, template_attach)
    frame_h, frame_w = map(int, template_patch.shape)

    n_trees = int(rng.integers(3, 5))
    colors = [int(x) for x in rng.choice(np.array(ROOT_COLORS, dtype=int), size=n_trees, replace=False)]
    categories = ["shorter", "equal", "taller"]
    if n_trees == 4:
        categories.append(str(rng.choice(np.array(categories, dtype=object))))
    rng.shuffle(categories)

    specs: list[dict[str, object]] = []
    for color, category in zip(colors, categories, strict=False):
        levels = _sample_tree_levels(rng, template_levels, category)
        specs.append(
            {
                "color": int(color),
                "attach": int(rng.integers(0, 2)),
                "levels": levels,
                "category": category,
            }
        )

    reserve_areas = []
    for spec in specs:
        rels = _reserve_rels(spec["levels"], template_levels, int(spec["attach"]))
        area = (rels[1] - rels[0] + 1) * (rels[3] - rels[2] + 1)
        reserve_areas.append(area)

    min_side = 20 + 2 * (n_trees - 3)
    min_h = max(frame_h + 1, min_side)
    min_w = max(frame_w + 1, min_side)
    if min_h > 30 or min_w > 30:
        return None
    h = int(rng.integers(min_h, 31))
    w = int(rng.integers(min_w, 31))

    grid = np.full((h, w), BG, dtype=int)
    frame_r = h - frame_h - int(rng.integers(0, 2))
    frame_c = w - frame_w - int(rng.integers(0, 2))
    grid[frame_r : frame_r + frame_h, frame_c : frame_c + frame_w] = template_patch

    occupied = np.zeros((h, w), dtype=bool)
    rr0 = max(0, frame_r - 1)
    rr1 = min(h, frame_r + frame_h + 1)
    cc0 = max(0, frame_c - 1)
    cc1 = min(w, frame_c + frame_w + 1)
    occupied[rr0:rr1, cc0:cc1] = True

    order = sorted(range(n_trees), key=lambda i: reserve_areas[i], reverse=True)
    for idx in order:
        pos = _place_tree_root(rng, occupied, template_levels, specs[idx])
        if pos is None:
            return None
        specs[idx]["root_r"], specs[idx]["root_c"] = pos

    for spec in specs:
        _draw_tree(
            grid,
            int(spec["root_r"]),
            int(spec["root_c"]),
            int(spec["color"]),
            int(spec["attach"]),
            spec["levels"],
        )

    expected = grid.copy()
    expected[frame_r : frame_r + frame_h, frame_c : frame_c + frame_w] = BG
    for spec in specs:
        _rewrite_tree(
            expected,
            int(spec["root_r"]),
            int(spec["root_c"]),
            int(spec["attach"]),
            int(spec["color"]),
            spec["levels"],
            template_levels,
        )

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after repeated attempts")
