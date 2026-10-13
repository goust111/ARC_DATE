from __future__ import annotations

import numpy as np

from arc_utils import components_mask, get_bg, paste
import itertools
import random


def _occupancy(shape: tuple[int, int], objects: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    occ = np.full(shape, -1, dtype=int)
    for idx, (coords, _values) in enumerate(objects):
        assert np.all(occ[coords[:, 0], coords[:, 1]] == -1), "objects must not overlap"
        occ[coords[:, 0], coords[:, 1]] = idx
    return occ


def _objects_that_can_shift_right(
    objects: list[tuple[np.ndarray, np.ndarray]], occ: np.ndarray
) -> set[int]:
    _, w = occ.shape
    movable = set(range(len(objects)))

    changed = True
    while changed:
        changed = False
        for idx in list(movable):
            coords, _values = objects[idx]
            shifted_cols = coords[:, 1] + 1
            if np.any(shifted_cols >= w):
                movable.remove(idx)
                changed = True
                continue

            labels = occ[coords[:, 0], shifted_cols]
            blockers = {int(label) for label in labels if int(label) not in {-1, idx}}
            if any(blocker not in movable for blocker in blockers):
                movable.remove(idx)
                changed = True

    return movable


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide every foreground object horizontally to the right until it settles against the border or another settled object.

    Input structure
    - The grid has a dominant background color and several separate 4-connected foreground objects.
    - Each foreground object is a small multicolor rectangle-like sprite on the same canvas.
    - Objects may overlap in their row ranges, so rightward movement can make a left object stop behind a right object.

    Rule
    - Treat each 4-connected foreground object as a rigid sprite.
    - Simultaneously move every sprite one column right whenever its next cells are inside the grid and not blocked by a sprite that cannot move.
    - Repeat the simultaneous move until no sprite can move; preserve each sprite's internal colors and shape.

    Key insight
    - The output is the terminal state of eastward gravity on the foreground sprites, not a recoloring or per-row sort.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = get_bg(g)
    components = components_mask(g != bg, connectivity=4)
    assert components, "expected at least one foreground object"

    objects = []
    for coords in components:
        coords = coords.astype(int)
        values = g[coords[:, 0], coords[:, 1]].astype(int)
        objects.append((coords.copy(), values))

    h, w = g.shape
    for _step in range(w):
        occ = _occupancy((h, w), objects)
        movable = _objects_that_can_shift_right(objects, occ)
        if not movable:
            break
        for idx in movable:
            objects[idx][0][:, 1] += 1
    else:
        raise AssertionError("rightward gravity did not settle within the grid width")

    out = np.full_like(g, bg)
    for coords, values in objects:
        assert np.all(out[coords[:, 0], coords[:, 1]] == bg), "settled objects must not overlap"
        out[coords[:, 0], coords[:, 1]] = values

    return out

# --------------- generator (o5x) ---------------

# Canvas, colors, and component rules.
GRID_HEIGHT, GRID_WIDTH = 15, 15
BACKGROUND = 0
COLOR_VALUES = tuple(range(1, 10))
CONNECTIVITY = 4

# Sprite and chain structure.
SPRITE_SIDES = (2, 3, 5)
CHAIN_COUNT_OPTIONS, CHAIN_SIZE_OPTIONS = (2, 3), (1, 2, 3)
TOTAL_SPRITES_MIN, TOTAL_SPRITES_MAX = 4, 6
CHAIN_GAP_ROWS = 1
MAX_SETTLED_CHAIN_WIDTH = GRID_WIDTH - 1

# Fork layout constraints.
MIN_FORK_ROOT_SIDE, MIN_FORK_BRANCH_SEPARATION = 3, 2

# Sampling retries and input shifts.
COLOR_PAIR_SHUFFLE_ATTEMPTS, MAX_LEFT_SHIFT, GENERATION_ATTEMPTS = 80, 11, 300


def _gen_checker(side: int, colors: tuple[int, int]) -> np.ndarray:
    rr, cc = np.indices((side, side))
    return np.where((rr + cc) % 2 == 0, colors[0], colors[1]).astype(int)


def _gen_color_pairs(n: int) -> list[tuple[int, int]]:
    colors = list(COLOR_VALUES)
    slots: list[int] = []
    while len(slots) < 2 * n:
        cycle = colors[:]
        random.shuffle(cycle)
        slots.extend(cycle)
    slots = slots[: 2 * n]

    for _ in range(COLOR_PAIR_SHUFFLE_ATTEMPTS):
        random.shuffle(slots)
        pairs: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        ok = True
        for i in range(n):
            a, b = slots[2 * i], slots[2 * i + 1]
            key = tuple(sorted((a, b)))
            if a == b or key in seen:
                ok = False
                break
            seen.add(key)
            if random.random() < 0.5:
                a, b = b, a
            pairs.append((a, b))
        if ok:
            return pairs

    all_pairs = list(itertools.combinations(colors, 2))
    random.shuffle(all_pairs)
    return [(a, b) if random.random() < 0.5 else (b, a) for a, b in all_pairs[:n]]


def _gen_chain_sizes() -> tuple[int, ...]:
    choices: list[tuple[int, ...]] = []
    for n_chains in CHAIN_COUNT_OPTIONS:
        for sizes in itertools.product(CHAIN_SIZE_OPTIONS, repeat=n_chains):
            if TOTAL_SPRITES_MIN <= sum(sizes) <= TOTAL_SPRITES_MAX and 1 in sizes:
                choices.append(tuple(sizes))
    return random.choice(choices)


def _gen_overlapping_top(parent_top: int, parent_side: int, child_side: int) -> int:
    lo = parent_top - child_side + 1
    hi = parent_top + parent_side - 1
    return random.randint(lo, hi)


def _gen_normalize_rows(sprites: list[dict]) -> int:
    min_r = min(sp["r"] for sp in sprites)
    for sp in sprites:
        sp["r"] -= min_r
    return max(sp["r"] + sp["side"] for sp in sprites)


def _gen_candidate_chain_layouts(n_sprites: int, sides: list[int]) -> list[tuple[str, tuple[int, ...]]]:
    builders: list[tuple[str, tuple[int, ...]]] = []
    for order in set(itertools.permutations(sides)):
        if n_sprites == 2:
            if sum(order) <= MAX_SETTLED_CHAIN_WIDTH:
                builders.append(("pair", order))
        elif n_sprites == 3:
            if sum(order) <= MAX_SETTLED_CHAIN_WIDTH:
                builders.append(("line", order))
            if (
                order[0] >= MIN_FORK_ROOT_SIDE
                and order[0] + order[1] <= MAX_SETTLED_CHAIN_WIDTH
                and order[0] + order[2] <= MAX_SETTLED_CHAIN_WIDTH
            ):
                builders.append(("fork", order))
    return builders


def _gen_pair_sprites(order: tuple[int, ...]) -> list[dict]:
    root_side, child_side = order
    c0 = GRID_WIDTH - root_side
    return [
        {"side": root_side, "r": 0, "c": c0, "parent": None},
        {
            "side": child_side,
            "r": _gen_overlapping_top(0, root_side, child_side),
            "c": c0 - child_side,
            "parent": 0,
        },
    ]


def _gen_line_sprites(order: tuple[int, ...]) -> list[dict]:
    root_side, mid_side, left_side = order
    c0 = GRID_WIDTH - root_side
    r1 = _gen_overlapping_top(0, root_side, mid_side)
    return [
        {"side": root_side, "r": 0, "c": c0, "parent": None},
        {"side": mid_side, "r": r1, "c": c0 - mid_side, "parent": 0},
        {
            "side": left_side,
            "r": _gen_overlapping_top(r1, mid_side, left_side),
            "c": c0 - mid_side - left_side,
            "parent": 1,
        },
    ]


def _gen_fork_sprites(order: tuple[int, ...]) -> list[dict]:
    root_side, upper_side, lower_side = order
    c0 = GRID_WIDTH - root_side
    upper_bottom = random.randint(0, root_side - MIN_FORK_ROOT_SIDE)
    lower_top = random.randint(upper_bottom + MIN_FORK_BRANCH_SEPARATION, root_side - 1)
    return [
        {"side": root_side, "r": 0, "c": c0, "parent": None},
        {
            "side": upper_side,
            "r": upper_bottom - upper_side + 1,
            "c": c0 - upper_side,
            "parent": 0,
        },
        {"side": lower_side, "r": lower_top, "c": c0 - lower_side, "parent": 0},
    ]


def _gen_build_chain(n_sprites: int, sides: list[int]) -> dict | None:
    if n_sprites == 1:
        side = sides[0]
        return {
            "sprites": [{"side": side, "r": 0, "c": GRID_WIDTH - side, "parent": None}],
            "height": side,
        }

    builders = _gen_candidate_chain_layouts(n_sprites, sides)
    if not builders:
        return None

    kind, order = random.choice(builders)
    if kind == "pair":
        sprites = _gen_pair_sprites(order)
    elif kind == "line":
        sprites = _gen_line_sprites(order)
    else:
        sprites = _gen_fork_sprites(order)

    height = _gen_normalize_rows(sprites)
    return {"sprites": sprites, "height": height}


def _gen_pack_chains(chains: list[dict]) -> bool:
    random.shuffle(chains)
    min_height = sum(ch["height"] for ch in chains) + CHAIN_GAP_ROWS * (len(chains) - 1)
    if min_height > GRID_HEIGHT:
        return False

    extra = GRID_HEIGHT - min_height
    gaps = [0] * (len(chains) + 1)
    for _ in range(extra):
        gaps[random.randrange(len(gaps))] += 1

    row = gaps[0]
    for idx, chain in enumerate(chains):
        for sp in chain["sprites"]:
            sp["r"] += row
        row += chain["height"] + CHAIN_GAP_ROWS + gaps[idx + 1]
    return True


def _gen_parent_tree(sprites: list[dict]) -> tuple[int | None, dict[int, list[int]]]:
    children: dict[int, list[int]] = {idx: [] for idx in range(len(sprites))}
    root = None
    for idx, sp in enumerate(sprites):
        parent = sp["parent"]
        if parent is None:
            root = idx
        else:
            children[parent].append(idx)
    return root, children


def _gen_assign_shifts(chain: dict) -> dict[int, int] | None:
    sprites = chain["sprites"]
    root, children = _gen_parent_tree(sprites)
    if root is None:
        return None

    caps: dict[int, int] = {}

    def visit(idx: int) -> int:
        cap = min(MAX_LEFT_SHIFT, int(sprites[idx]["c"]))
        for child in children[idx]:
            cap = min(cap, visit(child) - 1)
        caps[idx] = cap
        return cap

    if visit(root) < 1:
        return None

    shifts: dict[int, int] = {}

    def assign(idx: int, min_shift: int) -> bool:
        if min_shift > caps[idx]:
            return False
        shifts[idx] = random.randint(min_shift, caps[idx])
        for child in children[idx]:
            if not assign(child, shifts[idx] + 1):
                return False
        return True

    if not assign(root, 1):
        return None
    return shifts


def _gen_place(grid: np.ndarray, patch: np.ndarray, r: int, c: int) -> bool:
    h, w = patch.shape
    if r < 0 or c < 0 or r + h > GRID_HEIGHT or c + w > GRID_WIDTH:
        return False
    if np.any(grid[r : r + h, c : c + w] != BACKGROUND):
        return False
    paste(grid, patch, r, c)
    return True


def _gen_assign_patches(sprites: list[dict]) -> None:
    pairs = _gen_color_pairs(len(sprites))
    for sp, colors in zip(sprites, pairs):
        sp["patch"] = _gen_checker(sp["side"], colors)


def _gen_render_output(sprites: list[dict]) -> np.ndarray | None:
    out = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=int)
    for sp in sprites:
        if not _gen_place(out, sp["patch"], sp["r"], sp["c"]):
            return None
    return out


def _gen_render_input(chains: list[dict]) -> np.ndarray | None:
    inp = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=int)
    for chain in chains:
        shifts = _gen_assign_shifts(chain)
        if shifts is None:
            return None
        for idx, sp in enumerate(chain["sprites"]):
            if not _gen_place(inp, sp["patch"], sp["r"], sp["c"] - shifts[idx]):
                return None
    return inp


def _gen_validates(inp: np.ndarray, out: np.ndarray, sprite_count: int) -> bool:
    if len(components_mask(inp != BACKGROUND, connectivity=CONNECTIVITY)) != sprite_count:
        return False
    return np.array_equal(main(inp), out)


def _gen_attempt() -> np.ndarray | None:
    chain_sizes = list(_gen_chain_sizes())
    sides = [random.choice(SPRITE_SIDES) for _ in range(sum(chain_sizes))]

    chains: list[dict] = []
    pos = 0
    for n_sprites in chain_sizes:
        chain = _gen_build_chain(n_sprites, sides[pos : pos + n_sprites])
        if chain is None:
            return None
        chains.append(chain)
        pos += n_sprites

    if not _gen_pack_chains(chains):
        return None

    all_sprites = [sp for chain in chains for sp in chain["sprites"]]
    _gen_assign_patches(all_sprites)

    out = _gen_render_output(all_sprites)
    if out is None:
        return None

    inp = _gen_render_input(chains)
    if inp is None:
        return None
    if not _gen_validates(inp, out, len(all_sprites)):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid d282b262 input")
