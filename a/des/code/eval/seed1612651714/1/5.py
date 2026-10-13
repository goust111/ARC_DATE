from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
ORANGE = 7

PROGRAM_COLORS = (RED, BLUE, GREEN, ORANGE)
SPRITE_COLORS = (4, 5, 6, 8, 9)
COLOR_TO_OP = {
    RED: "rot90",
    BLUE: "flip_h",
    GREEN: "flip_v",
    ORANGE: "rot180",
}

TEMP_CANVAS = 30
GEN_RETRIES = 400
GROUP_COUNT_CHOICES = (2, 3, 4)
LAYER_COUNT_CHOICES = (2, 3, 4)
MAX_OUTER_BY_GROUP_COUNT = {
    2: 16,
    3: 13,
    4: 11,
}


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _find_color_components(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == BG:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            arr = np.asarray(coords, dtype=int)
            comps.append({"color": color, "coords": arr, "bbox": _bbox(arr)})
    return comps


def _is_rect_border(comp: dict[str, object]) -> bool:
    coords = np.asarray(comp["coords"], dtype=int)
    r0, c0, r1, c1 = comp["bbox"]  # type: ignore[assignment]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return False
    if int(coords.shape[0]) != 2 * (h + w) - 4:
        return False
    for r, c in coords:
        if int(r) not in (int(r0), int(r1)) and int(c) not in (int(c0), int(c1)):
            return False
    return True


def _strictly_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return (
        outer[0] < inner[0]
        and outer[1] < inner[1]
        and outer[2] > inner[2]
        and outer[3] > inner[3]
    )


def _crop_non_bg(arr: np.ndarray) -> np.ndarray:
    coords = np.argwhere(arr != BG)
    if coords.size == 0:
        return np.zeros((0, 0), dtype=int)
    r0, c0, r1, c1 = _bbox(coords)
    return arr[r0 : r1 + 1, c0 : c1 + 1].copy()


def _apply_one(patch: np.ndarray, color: int) -> np.ndarray:
    op = COLOR_TO_OP[int(color)]
    if op == "rot90":
        return np.rot90(patch, -1)
    if op == "flip_h":
        return np.fliplr(patch)
    if op == "flip_v":
        return np.flipud(patch)
    if op == "rot180":
        return np.rot90(patch, 2)
    raise ValueError(color)


def _apply_program(patch: np.ndarray, colors: list[int]) -> np.ndarray:
    out = patch.copy()
    for color in colors:
        out = _apply_one(out, int(color))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read each nested frame color sequence as a transformation program and apply it to the enclosed sprite.
    """
    g = np.asarray(grid, dtype=int)
    out = np.full_like(g, BG)

    comps = _find_color_components(g)
    border_indices = [i for i, comp in enumerate(comps) if _is_rect_border(comp)]
    if not border_indices:
        return out

    parents: dict[int, int | None] = {i: None for i in border_indices}
    children: dict[int, list[int]] = {i: [] for i in border_indices}
    for i in border_indices:
        ibox = comps[i]["bbox"]  # type: ignore[assignment]
        containers = [
            j
            for j in border_indices
            if j != i and _strictly_contains(comps[j]["bbox"], ibox)  # type: ignore[arg-type]
        ]
        if containers:
            parent = min(
                containers,
                key=lambda j: (
                    (comps[j]["bbox"][2] - comps[j]["bbox"][0] + 1)  # type: ignore[index]
                    * (comps[j]["bbox"][3] - comps[j]["bbox"][1] + 1),  # type: ignore[index]
                    comps[j]["bbox"],  # type: ignore[index]
                ),
            )
            parents[i] = parent
            children[parent].append(i)

    roots = sorted((i for i in border_indices if parents[i] is None), key=lambda i: comps[i]["bbox"])  # type: ignore[index]
    for root in roots:
        chain = [root]
        cur = root
        while children[cur]:
            assert len(children[cur]) == 1
            cur = children[cur][0]
            chain.append(cur)

        innermost = comps[chain[-1]]
        r0, c0, r1, c1 = innermost["bbox"]  # type: ignore[assignment]
        inner = g[int(r0) + 1 : int(r1), int(c0) + 1 : int(c1)]
        sprite = _crop_non_bg(inner)
        assert sprite.size > 0

        program = [int(comps[idx]["color"]) for idx in chain]
        transformed = _apply_program(sprite, program)

        inner_h = int(r1 - r0 - 1)
        inner_w = int(c1 - c0 - 1)
        th, tw = transformed.shape
        assert th <= inner_h and tw <= inner_w
        sr = int(r0) + 1 + (inner_h - th) // 2
        sc = int(c0) + 1 + (inner_w - tw) // 2
        mask = transformed != BG
        out[sr : sr + th, sc : sc + tw][mask] = transformed[mask]

    return out


def _random_connected_mask(rng: random.Random, h: int, w: int, count: int) -> np.ndarray:
    start = (rng.randrange(h), rng.randrange(w))
    cells = {start}
    while len(cells) < count:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c, h, w):
                if (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(rng.choice(frontier))
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _sample_sprite_dims(rng: random.Random, limit: int) -> tuple[int, int] | None:
    dims = [
        (h, w)
        for h in range(2, limit + 1)
        for w in range(2, limit + 1)
        if h * w >= 6 and max(h, w) >= 3
    ]
    if not dims:
        return None
    nonsquare = [dim for dim in dims if dim[0] != dim[1]]
    pool = nonsquare if nonsquare and rng.random() < 0.75 else dims
    return rng.choice(pool)


def _colorize_mask(rng: random.Random, mask: np.ndarray) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    cells = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    rng.shuffle(cells)
    palette_size = rng.randint(1, min(3, len(SPRITE_COLORS)))
    palette = rng.sample(SPRITE_COLORS, palette_size)
    for idx, (r, c) in enumerate(cells):
        if idx < palette_size:
            color = palette[idx]
        else:
            color = rng.choice(palette)
        patch[r, c] = int(color)
    return patch


def _has_border_like_component(arr: np.ndarray) -> bool:
    for comp in _find_color_components(arr):
        if _is_rect_border(comp):
            return True
    return False


def _make_sprite(rng: random.Random, size_limit: int, program: list[int]) -> np.ndarray | None:
    dims = _sample_sprite_dims(rng, size_limit)
    if dims is None:
        return None
    ph, pw = dims
    max_cells = min(ph * pw - 1, 8)
    min_cells = min(4, max_cells)
    if min_cells > max_cells:
        return None
    for _ in range(80):
        count = rng.randint(min_cells, max_cells)
        mask = _random_connected_mask(rng, ph, pw, count)
        patch = _colorize_mask(rng, mask)
        patch = _crop_non_bg(patch)
        if patch.size == 0 or _has_border_like_component(patch):
            continue
        transformed = _apply_program(patch, program)
        if patch.shape == transformed.shape and np.array_equal(patch, transformed):
            continue
        return patch
    return None


def _build_group(rng: random.Random, program: list[int], outer_cap: int) -> np.ndarray | None:
    layers = len(program)
    feasible_gaps = []
    for gap in (0, 1):
        step = gap + 1
        inner_max = outer_cap - 2 - 2 * step * (layers - 1)
        if inner_max >= 3:
            feasible_gaps.append(gap)
    if not feasible_gaps:
        return None

    for _ in range(100):
        gap = rng.choice(feasible_gaps)
        step = gap + 1
        inner_max = outer_cap - 2 - 2 * step * (layers - 1)
        inner_h = rng.randint(3, inner_max)
        inner_w = rng.randint(3, inner_max)
        size_limit = min(inner_h, inner_w)
        sprite = _make_sprite(rng, size_limit, program)
        if sprite is None:
            continue

        sprite_h, sprite_w = sprite.shape
        if sprite_h > inner_h or sprite_w > inner_w:
            continue
        transformed = _apply_program(sprite, program)
        if transformed.shape[0] > inner_h or transformed.shape[1] > inner_w:
            continue

        outer_h = inner_h + 2 + 2 * step * (layers - 1)
        outer_w = inner_w + 2 + 2 * step * (layers - 1)
        patch = np.full((outer_h, outer_w), BG, dtype=int)

        for depth, color in enumerate(program):
            inset = depth * step
            r0, c0 = inset, inset
            r1, c1 = outer_h - 1 - inset, outer_w - 1 - inset
            patch[r0, c0 : c1 + 1] = int(color)
            patch[r1, c0 : c1 + 1] = int(color)
            patch[r0 : r1 + 1, c0] = int(color)
            patch[r0 : r1 + 1, c1] = int(color)

        inner_r0 = step * (layers - 1) + 1
        inner_c0 = step * (layers - 1) + 1
        sr = inner_r0 + rng.randint(0, inner_h - sprite_h)
        sc = inner_c0 + rng.randint(0, inner_w - sprite_w)
        mask = sprite != BG
        patch[sr : sr + sprite_h, sc : sc + sprite_w][mask] = sprite[mask]
        return patch

    return None


def _sample_programs(rng: random.Random, group_count: int) -> list[list[int]]:
    lengths = [rng.choice(LAYER_COUNT_CHOICES) for _ in range(group_count)]
    programs = [rng.sample(PROGRAM_COLORS, length) for length in lengths]
    used = set(color for program in programs for color in program)
    missing = [color for color in PROGRAM_COLORS if color not in used]
    for color in missing:
        choices = [(gi, pi) for gi, program in enumerate(programs) for pi in range(len(program)) if color not in program]
        gi, pi = rng.choice(choices)
        programs[gi][pi] = color
    return programs


def _place_groups(rng: random.Random, groups: list[np.ndarray]) -> np.ndarray | None:
    canvas = np.full((TEMP_CANVAS, TEMP_CANVAS), BG, dtype=int)
    reserved = np.zeros((TEMP_CANVAS, TEMP_CANVAS), dtype=bool)
    order = sorted(range(len(groups)), key=lambda i: groups[i].shape[0] * groups[i].shape[1], reverse=True)

    for idx in order:
        group = groups[idx]
        gh, gw = group.shape
        candidates = [(r, c) for r in range(TEMP_CANVAS - gh + 1) for c in range(TEMP_CANVAS - gw + 1)]
        rng.shuffle(candidates)
        placed = False
        for r, c in candidates[:400]:
            rr0 = max(0, r - 1)
            cc0 = max(0, c - 1)
            rr1 = min(TEMP_CANVAS, r + gh + 1)
            cc1 = min(TEMP_CANVAS, c + gw + 1)
            if reserved[rr0:rr1, cc0:cc1].any():
                continue
            mask = group != BG
            canvas[r : r + gh, c : c + gw][mask] = group[mask]
            reserved[r : r + gh, c : c + gw] = True
            placed = True
            break
        if not placed:
            return None

    coords = np.argwhere(canvas != BG)
    if coords.size == 0:
        return None
    r0, c0, r1, c1 = _bbox(coords)
    top = rng.randint(0, min(3, r0))
    left = rng.randint(0, min(3, c0))
    bottom = rng.randint(0, min(3, TEMP_CANVAS - 1 - r1))
    right = rng.randint(0, min(3, TEMP_CANVAS - 1 - c1))
    out = canvas[r0 - top : r1 + 1 + bottom, c0 - left : c1 + 1 + right].copy()
    if rng.random() < 0.5:
        out = out.T.copy()
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    group_count = rng.choice(GROUP_COUNT_CHOICES)
    programs = _sample_programs(rng, group_count)
    outer_cap = MAX_OUTER_BY_GROUP_COUNT[group_count]

    groups: list[np.ndarray] = []
    for program in programs:
        group = _build_group(rng, program, outer_cap)
        if group is None:
            return None
        groups.append(group)

    grid = _place_groups(rng, groups)
    if grid is None:
        return None
    if set(int(x) for x in np.unique(grid)) & set(PROGRAM_COLORS) != set(PROGRAM_COLORS):
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape or not np.any(out != BG):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
