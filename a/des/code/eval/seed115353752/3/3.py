from __future__ import annotations

import random

import numpy as np


BG = 0
WALL = 5
DIRS = {
    "U": (-1, 0),
    "R": (0, 1),
    "D": (1, 0),
    "L": (0, -1),
}
DIR_ORDER = ("U", "R", "D", "L")
ALLOWED_TEMPLATE_DIRS = (
    frozenset(("U", "R")),
    frozenset(("R", "D")),
    frozenset(("D", "L")),
    frozenset(("L", "U")),
    frozenset(("U", "R", "D")),
    frozenset(("R", "D", "L")),
    frozenset(("D", "L", "U")),
    frozenset(("L", "U", "R")),
    frozenset(("U", "R", "D", "L")),
)
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 400
_RNG = np.random.default_rng()


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            seen[r0, c0] = True
            stack = [(r0, c0)]
            comp: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                comp.append((int(r), int(c)))
                for dr, dc in DIRS.values():
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _dirset_from_component(comp: list[tuple[int, int]]) -> frozenset[str] | None:
    cells = set(comp)
    candidates: list[tuple[int, int, frozenset[str]]] = []
    for cr, cc in cells:
        runs: dict[str, list[int]] = {name: [] for name in DIR_ORDER}
        ok = True
        for r, c in cells:
            if (r, c) == (cr, cc):
                continue
            dr = r - cr
            dc = c - cc
            if dr != 0 and dc != 0:
                ok = False
                break
            if dr < 0:
                runs["U"].append(-dr)
            elif dr > 0:
                runs["D"].append(dr)
            elif dc < 0:
                runs["L"].append(-dc)
            elif dc > 0:
                runs["R"].append(dc)
        if not ok:
            continue
        dirs = frozenset(name for name in DIR_ORDER if runs[name])
        if dirs not in ALLOWED_TEMPLATE_DIRS:
            continue
        lengths = {max(runs[name]) for name in dirs}
        if len(lengths) != 1:
            continue
        length = next(iter(lengths))
        if not (1 <= length <= 3):
            continue
        if any(sorted(set(runs[name])) != list(range(1, length + 1)) for name in dirs):
            continue
        if 1 + len(dirs) * length != len(comp):
            continue
        candidates.append((len(dirs), length, dirs))
    if not candidates:
        return None
    best_score = max((degree, length) for degree, length, _ in candidates)
    best_dirs = {dirs for degree, length, dirs in candidates if (degree, length) == best_score}
    if len(best_dirs) != 1:
        return None
    return next(iter(best_dirs))


def _extract_templates(region: np.ndarray, wall_color: int) -> dict[int, frozenset[str]] | None:
    colors = [int(v) for v in np.unique(region) if int(v) not in (BG, wall_color)]
    if not colors:
        return None
    templates: dict[int, frozenset[str]] = {}
    for color in colors:
        comps = _components(region == color)
        if len(comps) != 1:
            return None
        dirs = _dirset_from_component(comps[0])
        if dirs is None:
            return None
        templates[color] = dirs
    if len(set(templates.values())) != len(templates):
        return None
    return templates


def _list_seeds(region: np.ndarray, wall_color: int, template_colors: set[int] | None) -> list[tuple[int, int, int]] | None:
    seeds: list[tuple[int, int, int]] = []
    for color in [int(v) for v in np.unique(region) if int(v) not in (BG, wall_color)]:
        if template_colors is not None and color not in template_colors:
            return None
        for comp in _components(region == color):
            if len(comp) != 1:
                return None
            r, c = comp[0]
            seeds.append((r, c, color))
    return seeds if seeds else None


def _trace_seed(work: np.ndarray, r: int, c: int, dirs: frozenset[str], wall_color: int) -> tuple[set[tuple[int, int]], list[str]]:
    h, w = work.shape
    filled: set[tuple[int, int]] = set()
    reasons: list[str] = []
    for name in DIR_ORDER:
        if name not in dirs:
            continue
        dr, dc = DIRS[name]
        rr, cc = r, c
        while True:
            nr, nc = rr + dr, cc + dc
            if not (0 <= nr < h and 0 <= nc < w):
                reasons.append("edge")
                break
            if work[nr, nc] != BG:
                reasons.append("wall" if int(work[nr, nc]) == wall_color else "seed")
                break
            filled.add((int(nr), int(nc)))
            rr, cc = nr, nc
    return filled, reasons


def _grow_workspace(work: np.ndarray, templates: dict[int, frozenset[str]], wall_color: int) -> np.ndarray:
    out = np.array(work, copy=True)
    seeds = _list_seeds(work, wall_color, set(templates))
    assert seeds is not None, "workspace must contain only seed pixels from legend colors"
    for r, c, color in seeds:
        filled, _ = _trace_seed(work, r, c, templates[color], wall_color)
        for rr, cc in filled:
            out[rr, cc] = int(color)
    return out


def _parse_split(grid: np.ndarray, orientation: str, index: int, wall_color: int) -> tuple[dict[int, frozenset[str]], np.ndarray] | None:
    if orientation == "H":
        first = grid[:index, :]
        second = grid[index + 1 :, :]
    else:
        first = grid[:, :index]
        second = grid[:, index + 1 :]
    if first.size == 0 or second.size == 0:
        return None
    for legend, work in ((first, second), (second, first)):
        templates = _extract_templates(legend, wall_color)
        if templates is None:
            continue
        seeds = _list_seeds(work, wall_color, set(templates))
        if seeds is None:
            continue
        return templates, np.array(work, copy=True)
    return None


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[tuple[int, int, dict[int, frozenset[str]], np.ndarray]] = []
    for r in range(h):
        if int(g[r, 0]) != BG and np.all(g[r, :] == g[r, 0]):
            parsed = _parse_split(g, "H", r, int(g[r, 0]))
            if parsed is not None:
                templates, work = parsed
                candidates.append((len(templates), work.size, templates, work))
    for c in range(w):
        if int(g[0, c]) != BG and np.all(g[:, c] == g[0, c]):
            parsed = _parse_split(g, "V", c, int(g[0, c]))
            if parsed is not None:
                templates, work = parsed
                candidates.append((len(templates), work.size, templates, work))
    assert candidates, "expected a complete separator and a decodable legend"
    _, _, templates, work = max(candidates, key=lambda item: (item[0], item[1]))
    return _grow_workspace(work, templates, WALL)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode direction templates from the legend region, then grow each workspace seed
    in those directions until a wall, the workspace boundary, or another non-background
    workspace pixel blocks the ray. The output is only the workspace region.
    """
    return _solve(grid)


def _template_patch(color: int, dirs: frozenset[str], length: int) -> np.ndarray:
    cells = [(0, 0)]
    for name in dirs:
        dr, dc = DIRS[name]
        for step in range(1, length + 1):
            cells.append((dr * step, dc * step))
    min_r = min(r for r, _ in cells)
    max_r = max(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    max_c = max(c for _, c in cells)
    patch = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=int)
    for r, c in cells:
        patch[r - min_r, c - min_c] = int(color)
    return patch


def _starts_for_strip(total: int, sizes: list[int], margin: int = 1, gap: int = 1) -> list[int] | None:
    required = margin * 2 + sum(sizes) + gap * max(0, len(sizes) - 1)
    if required > total:
        return None
    slack = total - required
    extras = _RNG.multinomial(slack, np.ones(len(sizes) + 1) / (len(sizes) + 1))
    pos = margin + int(extras[0])
    starts: list[int] = []
    for i, size in enumerate(sizes):
        starts.append(pos)
        pos += size
        if i < len(sizes) - 1:
            pos += gap + int(extras[i + 1])
    return starts


def _fits(mask: np.ndarray, patch_mask: np.ndarray, r0: int, c0: int) -> bool:
    h, w = patch_mask.shape
    return not bool((mask[r0 : r0 + h, c0 : c0 + w] & patch_mask).any())


def _random_blob_patch() -> np.ndarray:
    h = int(_RNG.integers(2, 5))
    w = int(_RNG.integers(2, 6))
    max_area = min(h * w, 8)
    area = int(_RNG.integers(3, max_area + 1))
    mask = np.zeros((h, w), dtype=bool)
    start = (int(_RNG.integers(0, h)), int(_RNG.integers(0, w)))
    mask[start] = True
    while int(mask.sum()) < area:
        frontier: set[tuple[int, int]] = set()
        for r, c in np.argwhere(mask):
            rr = int(r)
            cc = int(c)
            for dr, dc in DIRS.values():
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((nr, nc))
        nr, nc = random.choice(list(frontier))
        mask[nr, nc] = True
    return mask


def _place_walls(shape: tuple[int, int]) -> np.ndarray | None:
    h, w = shape
    work = np.zeros(shape, dtype=int)
    occupied = np.zeros(shape, dtype=bool)
    wall_count = int(_RNG.integers(1, 4))
    for _ in range(wall_count):
        placed = False
        for _attempt in range(120):
            patch_mask = _random_blob_patch()
            ph, pw = patch_mask.shape
            if ph >= h - 1 or pw >= w - 1:
                continue
            r0 = int(_RNG.integers(1, h - ph))
            c0 = int(_RNG.integers(1, w - pw))
            blocked = _dilate8(occupied)
            if not _fits(blocked, patch_mask, r0, c0):
                continue
            work[r0 : r0 + ph, c0 : c0 + pw][patch_mask] = WALL
            occupied[r0 : r0 + ph, c0 : c0 + pw] |= patch_mask
            placed = True
            break
        if not placed:
            return None
    return work


def _candidate_seed_positions(
    work: np.ndarray,
    dirs: frozenset[str],
    reserved: np.ndarray,
    target_reason: str | None,
) -> list[tuple[int, int, set[tuple[int, int]], list[str]]]:
    h, w = work.shape
    blocked_cells = _dilate8(work != BG) | reserved
    candidates: list[tuple[int, int, set[tuple[int, int]], list[str]]] = []
    for r in range(h):
        for c in range(w):
            if work[r, c] != BG or blocked_cells[r, c]:
                continue
            filled, reasons = _trace_seed(work, r, c, dirs, WALL)
            if not filled:
                continue
            if any(reserved[rr, cc] for rr, cc in filled):
                continue
            if target_reason is not None and target_reason not in reasons:
                continue
            candidates.append((r, c, filled, reasons))
    return candidates


def _place_seeds(
    walls: np.ndarray,
    templates: dict[int, frozenset[str]],
) -> np.ndarray | None:
    work = np.array(walls, copy=True)
    reserved = np.zeros(work.shape, dtype=bool)
    template_colors = list(templates)
    min_seeds = max(3, len(template_colors))
    max_seeds = min(5, len(template_colors) + 2)
    seed_total = int(_RNG.integers(min_seeds, max_seeds + 1))
    plan_colors = template_colors[:]
    while len(plan_colors) < seed_total:
        plan_colors.append(int(random.choice(template_colors)))
    random.shuffle(plan_colors)
    plan_targets = ["edge", "wall", "seed"] + [None] * max(0, seed_total - 3)
    for idx in range(3, len(plan_targets)):
        plan_targets[idx] = random.choice(["edge", "wall", "seed", None])
    for color, target in zip(plan_colors, plan_targets):
        dirs = templates[color]
        candidates = _candidate_seed_positions(work, dirs, reserved, target)
        if not candidates and target is not None:
            return None
        if not candidates:
            candidates = _candidate_seed_positions(work, dirs, reserved, None)
        if not candidates:
            return None
        r, c, filled, _ = random.choice(candidates)
        work[r, c] = int(color)
        for rr, cc in filled:
            reserved[rr, cc] = True
    reasons_seen: set[str] = set()
    growth = 0
    seeds = _list_seeds(work, WALL, set(templates))
    assert seeds is not None
    for r, c, color in seeds:
        filled, reasons = _trace_seed(work, r, c, templates[color], WALL)
        growth += len(filled)
        reasons_seen.update(reasons)
    if growth < 6:
        return None
    if not {"edge", "wall", "seed"}.issubset(reasons_seen):
        return None
    return work


def _build_vertical_case() -> np.ndarray | None:
    n_templates = int(_RNG.integers(2, 4))
    colors = random.sample(PALETTE, n_templates)
    dirsets = random.sample(list(ALLOWED_TEMPLATE_DIRS), n_templates)
    patches = []
    templates: dict[int, frozenset[str]] = {}
    for color, dirs in zip(colors, dirsets):
        length = int(_RNG.integers(1, 4))
        patch = _template_patch(color, dirs, length)
        patches.append(patch)
        templates[int(color)] = dirs
    legend_w = max(8, max(patch.shape[1] for patch in patches) + 2)
    required_h = 2 + sum(patch.shape[0] for patch in patches) + (n_templates - 1)
    total_h = int(_RNG.integers(max(15, required_h), max(24, required_h) + 1))
    work_w = int(_RNG.integers(11, 18))
    order = list(range(n_templates))
    random.shuffle(order)
    ordered_patches = [patches[idx] for idx in order]
    starts = _starts_for_strip(total_h, [patch.shape[0] for patch in ordered_patches], margin=1, gap=1)
    if starts is None:
        return None
    legend = np.zeros((total_h, legend_w), dtype=int)
    for start, patch in zip(starts, ordered_patches):
        ph, pw = patch.shape
        c0 = int(_RNG.integers(1, legend_w - pw))
        legend[start : start + ph, c0 : c0 + pw] = patch
    walls = _place_walls((total_h, work_w))
    if walls is None:
        return None
    work = _place_seeds(walls, templates)
    if work is None:
        return None
    left_is_legend = bool(_RNG.integers(0, 2))
    grid = np.zeros((total_h, legend_w + 1 + work_w), dtype=int)
    if left_is_legend:
        grid[:, :legend_w] = legend
        grid[:, legend_w] = WALL
        grid[:, legend_w + 1 :] = work
    else:
        grid[:, :work_w] = work
        grid[:, work_w] = WALL
        grid[:, work_w + 1 :] = legend
    return grid


def _build_horizontal_case() -> np.ndarray | None:
    n_templates = int(_RNG.integers(2, 4))
    colors = random.sample(PALETTE, n_templates)
    dirsets = random.sample(list(ALLOWED_TEMPLATE_DIRS), n_templates)
    patches = []
    templates: dict[int, frozenset[str]] = {}
    for color, dirs in zip(colors, dirsets):
        length = int(_RNG.integers(1, 4))
        patch = _template_patch(color, dirs, length)
        patches.append(patch)
        templates[int(color)] = dirs
    legend_h = max(8, max(patch.shape[0] for patch in patches) + 2)
    required_w = 2 + sum(patch.shape[1] for patch in patches) + (n_templates - 1)
    total_w = int(_RNG.integers(max(15, required_w), max(24, required_w) + 1))
    work_h = int(_RNG.integers(11, 18))
    order = list(range(n_templates))
    random.shuffle(order)
    ordered_patches = [patches[idx] for idx in order]
    starts = _starts_for_strip(total_w, [patch.shape[1] for patch in ordered_patches], margin=1, gap=1)
    if starts is None:
        return None
    legend = np.zeros((legend_h, total_w), dtype=int)
    for start, patch in zip(starts, ordered_patches):
        ph, pw = patch.shape
        r0 = int(_RNG.integers(1, legend_h - ph))
        legend[r0 : r0 + ph, start : start + pw] = patch
    walls = _place_walls((work_h, total_w))
    if walls is None:
        return None
    work = _place_seeds(walls, templates)
    if work is None:
        return None
    top_is_legend = bool(_RNG.integers(0, 2))
    grid = np.zeros((legend_h + 1 + work_h, total_w), dtype=int)
    if top_is_legend:
        grid[:legend_h, :] = legend
        grid[legend_h, :] = WALL
        grid[legend_h + 1 :, :] = work
    else:
        grid[:work_h, :] = work
        grid[work_h, :] = WALL
        grid[work_h + 1 :, :] = legend
    return grid


def _generate_case() -> np.ndarray | None:
    grid = _build_vertical_case() if bool(_RNG.integers(0, 2)) else _build_horizontal_case()
    if grid is None:
        return None
    try:
        out = main(grid)
    except Exception:
        return None
    if out.size == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid puzzle after {GEN_RETRIES} attempts")
