from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
TEMPLATE_WORKSPACE = 7
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(int(rr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _component_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    local = coords - np.array([r0, c0], dtype=int)
    mask[local[:, 0], local[:, 1]] = True
    return mask, (r0, c0)


def _root_of_mask(mask: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(mask)
    leaves: list[tuple[int, int]] = []
    for r, c in coords:
        deg = 0
        for nr, nc in _neighbors4(int(r), int(c)):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                deg += 1
        if deg == 1:
            leaves.append((int(r), int(c)))
    assert len(leaves) == 1, "template must have exactly one root leaf"
    return leaves[0]


def _separator_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == GREY)]
    assert len(rows) == 1, "expected one full-width grey separator row"
    return int(rows[0])


def _template_library(top: np.ndarray) -> dict[int, tuple[np.ndarray, tuple[int, int]]]:
    templates: dict[int, tuple[np.ndarray, tuple[int, int]]] = {}
    for color in map(int, np.unique(top)):
        if color in (BLACK, GREY):
            continue
        comps = _find_components(top == color)
        assert len(comps) == 1, "each template color must form one component"
        mask, _offset = _component_mask(comps[0])
        root = _root_of_mask(mask)
        templates[color] = (mask, root)
    assert templates, "expected at least one template"
    return templates


def _branch_for_seed(col: int, width: int) -> str:
    mid = (width - 1) / 2.0
    if col < mid:
        return "left"
    if col > mid:
        return "right"
    return "center"


def _orient_mask(mask: np.ndarray, root: tuple[int, int], branch: str) -> tuple[np.ndarray, tuple[int, int]]:
    if branch == "center":
        return mask, root
    mirrored_root_c = mask.shape[1] - 1 - int(root[1])
    if branch == "left":
        should_flip = mirrored_root_c < int(root[1])
    else:
        should_flip = mirrored_root_c > int(root[1])
    if not should_flip:
        return mask, root
    return np.fliplr(mask), (int(root[0]), int(mirrored_root_c))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each seed pixel to the same-colored template above the separator, optionally flip that
    template horizontally based on the seed's side of the canvas, then align the template root to
    the seed. The output keeps only the lower canvas.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    assert int(g.min()) >= 0 and int(g.max()) <= 9

    sep = _separator_row(g)
    top = g[:sep]
    lower = g[sep + 1 :]
    assert top.size > 0 and lower.size > 0
    templates = _template_library(top)

    out = np.full(lower.shape, BLACK, dtype=int)
    seed_colors = {int(v) for v in np.unique(lower) if int(v) != BLACK}
    assert seed_colors == set(templates), "seed colors must match template colors exactly"

    for color in sorted(seed_colors):
        coords = np.argwhere(lower == color)
        assert coords.shape == (1, 2), "each seed color must appear exactly once"
        seed_r, seed_c = map(int, coords[0])
        mask, root = templates[color]
        branch = _branch_for_seed(seed_c, lower.shape[1])
        oriented, oriented_root = _orient_mask(mask, root, branch)
        pixels = np.argwhere(oriented)
        dr = pixels[:, 0] - int(oriented_root[0])
        dc = pixels[:, 1] - int(oriented_root[1])
        rr = seed_r + dr
        cc = seed_c + dc
        assert rr.min() >= 0 and rr.max() < out.shape[0]
        assert cc.min() >= 0 and cc.max() < out.shape[1]
        assert np.all(out[rr, cc] == BLACK), "grown templates must not overlap"
        out[rr, cc] = color

    return out


def _cells_connected(cells: set[tuple[int, int]]) -> bool:
    start = next(iter(cells))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(cells)


def _valid_template_cells(cells: set[tuple[int, int]], root: tuple[int, int]) -> bool:
    if root not in cells or not _cells_connected(cells):
        return False
    root_c = int(root[1])
    if sum(c == root_c for _, c in cells) != 1:
        return False

    leaf_count = 0
    for cell in cells:
        deg = sum((nr, nc) in cells for nr, nc in _neighbors4(*cell))
        if cell == root:
            if deg != 1:
                return False
        elif deg == 1:
            return False
        if deg == 1:
            leaf_count += 1
    return leaf_count == 1


def _cells_to_mask(cells: set[tuple[int, int]]) -> np.ndarray:
    coords = np.array(sorted(cells), dtype=int)
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _random_template_mask() -> np.ndarray | None:
    root = (random.randint(1, TEMPLATE_WORKSPACE - 2), TEMPLATE_WORKSPACE - 1)
    rect_h = random.choice([2, 3])
    rect_w = random.choice([2, 3, 4])
    min_r0 = max(0, root[0] - rect_h + 1)
    max_r0 = min(root[0], TEMPLATE_WORKSPACE - rect_h)
    rect_r0 = random.randint(min_r0, max_r0)
    rect_c1 = root[1] - 1
    rect_c0 = rect_c1 - rect_w + 1

    cells = {(r, c) for r in range(rect_r0, rect_r0 + rect_h) for c in range(rect_c0, rect_c1 + 1)}
    cells.add(root)
    target_area = random.randint(len(cells), min(len(cells) + 4, 14))

    while len(cells) < target_area:
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < TEMPLATE_WORKSPACE and 0 <= nc < TEMPLATE_WORKSPACE and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        candidates = []
        for cand in frontier:
            if cand[1] >= root[1]:
                continue
            new_cells = set(cells)
            new_cells.add(cand)
            if _valid_template_cells(new_cells, root):
                mask = _cells_to_mask(new_cells)
                if mask.shape[0] <= 6 and mask.shape[1] <= 6:
                    candidates.append(cand)
        if not candidates:
            break
        cells.add(random.choice(candidates))

    if not _valid_template_cells(cells, root):
        return None
    if random.random() < 0.5:
        cells = {(TEMPLATE_WORKSPACE - 1 - r, c) for r, c in cells}
    mask = _cells_to_mask(cells)
    _ = _root_of_mask(mask)
    return mask


def _balanced_branches(count: int, width: int) -> list[str]:
    use_center = width % 2 == 1 and count >= 3 and random.random() < 0.7
    allowed = ["left", "right"] + (["center"] if use_center else [])
    branches = allowed[:]
    counts = {branch: 1 for branch in branches}
    while len(branches) < count:
        min_count = min(counts.values())
        choices = [branch for branch in allowed if counts.get(branch, 0) == min_count]
        branch = random.choice(choices)
        branches.append(branch)
        counts[branch] = counts.get(branch, 0) + 1
    random.shuffle(branches)
    return branches


def _place_templates_in_top(top_h: int, width: int, masks: list[np.ndarray], colors: list[int]) -> np.ndarray | None:
    top = np.full((top_h, width), BLACK, dtype=int)
    occupied = np.zeros((top_h, width), dtype=bool)
    order = sorted(range(len(masks)), key=lambda i: masks[i].shape[0] * masks[i].shape[1], reverse=True)
    for idx in order:
        mask = masks[idx]
        h, w = mask.shape
        candidates = []
        for r0 in range(top_h - h + 1):
            for c0 in range(width - w + 1):
                rr0 = max(0, r0 - 1)
                rr1 = min(top_h, r0 + h + 1)
                cc0 = max(0, c0 - 1)
                cc1 = min(width, c0 + w + 1)
                if occupied[rr0:rr1, cc0:cc1].any():
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        sub = top[r0 : r0 + h, c0 : c0 + w]
        sub[mask] = colors[idx]
        occupied[max(0, r0 - 1) : min(top_h, r0 + h + 1), max(0, c0 - 1) : min(width, c0 + w + 1)] = True
    return top


def _seed_candidates(
    canvas_shape: tuple[int, int],
    occupied: np.ndarray,
    mask: np.ndarray,
    root: tuple[int, int],
    branch: str,
) -> list[tuple[int, int]]:
    h, w = canvas_shape
    coords = np.argwhere(mask)
    rel = coords - np.array(root, dtype=int)
    min_dr, min_dc = map(int, rel.min(axis=0))
    max_dr, max_dc = map(int, rel.max(axis=0))
    candidates = []
    for seed_r in range(-min_dr, h - max_dr):
        for seed_c in range(-min_dc, w - max_dc):
            if _branch_for_seed(seed_c, w) != branch:
                continue
            r0 = seed_r + min_dr
            r1 = seed_r + max_dr
            c0 = seed_c + min_dc
            c1 = seed_c + max_dc
            rr0 = max(0, r0 - 1)
            rr1 = min(h, r1 + 2)
            cc0 = max(0, c0 - 1)
            cc1 = min(w, c1 + 2)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            candidates.append((seed_r, seed_c))
    return candidates


def _render_canvas(
    canvas_h: int,
    canvas_w: int,
    masks: list[np.ndarray],
    colors: list[int],
    branches: list[str],
) -> tuple[np.ndarray, np.ndarray] | None:
    lower = np.full((canvas_h, canvas_w), BLACK, dtype=int)
    out = np.full((canvas_h, canvas_w), BLACK, dtype=int)
    occupied = np.zeros((canvas_h, canvas_w), dtype=bool)

    tasks = list(range(len(masks)))
    tasks.sort(key=lambda i: (branches[i] != "center", -(masks[i].shape[0] * masks[i].shape[1])))

    for idx in tasks:
        base_mask = masks[idx]
        base_root = _root_of_mask(base_mask)
        mask, root = _orient_mask(base_mask, base_root, branches[idx])
        candidates = _seed_candidates((canvas_h, canvas_w), occupied, mask, root, branches[idx])
        if not candidates:
            return None
        seed_r, seed_c = random.choice(candidates)
        lower[seed_r, seed_c] = colors[idx]
        pixels = np.argwhere(mask)
        rr = seed_r + pixels[:, 0] - int(root[0])
        cc = seed_c + pixels[:, 1] - int(root[1])
        assert np.all(out[rr, cc] == BLACK)
        out[rr, cc] = colors[idx]

        r0 = int(rr.min())
        r1 = int(rr.max())
        c0 = int(cc.min())
        c1 = int(cc.max())
        occupied[max(0, r0 - 1) : min(canvas_h, r1 + 2), max(0, c0 - 1) : min(canvas_w, c1 + 2)] = True

    return lower, out


def _generate_case() -> np.ndarray | None:
    n_templates = random.randint(3, 5)
    canvas_h = random.randint(8, 14)
    if random.random() < 0.65:
        canvas_w = random.choice([13, 15, 17, 19, 21])
    else:
        canvas_w = random.choice([12, 14, 16, 18, 20])

    masks = []
    for _ in range(n_templates):
        mask = _random_template_mask()
        if mask is None:
            return None
        masks.append(mask)

    colors = random.sample(PALETTE, n_templates)
    branches = _balanced_branches(n_templates, canvas_w)

    top_h = random.randint(max(max(mask.shape[0] for mask in masks) + 2, 7), 11)
    top = _place_templates_in_top(top_h, canvas_w, masks, colors)
    if top is None:
        return None

    rendered = _render_canvas(canvas_h, canvas_w, masks, colors, branches)
    if rendered is None:
        return None
    lower, expected = rendered

    separator = np.full((1, canvas_w), GREY, dtype=int)
    inp = np.vstack([top, separator, lower])
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
