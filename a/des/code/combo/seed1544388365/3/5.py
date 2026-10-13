import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np


BG = 0
MAX_ATTEMPTS = 300
PATTERN_MODES = ("noise", "row_bands", "col_bands", "diag_bands", "rectangles")


@dataclass
class Block:
    core: np.ndarray
    arms: dict[str, tuple[int, int]]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _parse_block(grid: np.ndarray, pts: list[tuple[int, int]]) -> Block:
    row_counts: dict[int, int] = defaultdict(int)
    col_counts: dict[int, int] = defaultdict(int)
    for r, c in pts:
        row_counts[int(r)] += 1
        col_counts[int(c)] += 1

    core_rows = sorted(r for r, cnt in row_counts.items() if cnt > 1)
    core_cols = sorted(c for c, cnt in col_counts.items() if cnt > 1)
    assert core_rows and core_cols, "component must contain a square core"
    assert core_rows == list(range(core_rows[0], core_rows[-1] + 1))
    assert core_cols == list(range(core_cols[0], core_cols[-1] + 1))

    r0, r1 = core_rows[0], core_rows[-1]
    c0, c1 = core_cols[0], core_cols[-1]
    side = r1 - r0 + 1
    assert side == c1 - c0 + 1 and side >= 2, "core must be a nontrivial square"

    core = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()
    assert np.all(core != BG), "generator and solver assume solid non-background blocks"

    ptset = {(int(r), int(c)) for r, c in pts}
    used = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    arms: dict[str, tuple[int, int]] = {}

    up_hits = [c for c in range(c0, c1 + 1) if (r0 - 1, c) in ptset]
    assert len(up_hits) <= 1
    if up_hits:
        c = up_hits[0]
        arms["up"] = (c - c0, int(grid[r0 - 1, c]))
        used.add((r0 - 1, c))

    down_hits = [c for c in range(c0, c1 + 1) if (r1 + 1, c) in ptset]
    assert len(down_hits) <= 1
    if down_hits:
        c = down_hits[0]
        arms["down"] = (c - c0, int(grid[r1 + 1, c]))
        used.add((r1 + 1, c))

    left_hits = [r for r in range(r0, r1 + 1) if (r, c0 - 1) in ptset]
    assert len(left_hits) <= 1
    if left_hits:
        r = left_hits[0]
        arms["left"] = (r - r0, int(grid[r, c0 - 1]))
        used.add((r, c0 - 1))

    right_hits = [r for r in range(r0, r1 + 1) if (r, c1 + 1) in ptset]
    assert len(right_hits) <= 1
    if right_hits:
        r = right_hits[0]
        arms["right"] = (r - r0, int(grid[r, c1 + 1]))
        used.add((r, c1 + 1))

    assert ptset == used, "component contains unsupported extra pixels"
    return Block(core=core, arms=arms)


def _match_horizontal(blocks: list[Block]) -> dict[int, int]:
    left_lookup: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, block in enumerate(blocks):
        if "left" not in block.arms:
            continue
        edge = block.core[:, 0]
        for off, color in enumerate(edge.tolist()):
            left_lookup[(int(off), int(color))].append(idx)

    right_of: dict[int, int] = {}
    for idx, block in enumerate(blocks):
        arm = block.arms.get("right")
        if arm is None:
            continue
        off, color = arm
        candidates = [j for j in left_lookup.get((int(off), int(color)), []) if j != idx]
        if len(candidates) > 1:
            filtered = []
            for j in candidates:
                target = blocks[j]
                left_off, left_color = target.arms["left"]
                if int(block.core[left_off, -1]) == int(left_color):
                    filtered.append(j)
            if filtered:
                candidates = filtered
        assert len(candidates) == 1, "right arm must identify a unique right neighbor"
        right_of[idx] = candidates[0]

    values = list(right_of.values())
    assert len(values) == len(set(values)), "two blocks point at the same horizontal target"
    return right_of


def _match_vertical(blocks: list[Block]) -> dict[int, int]:
    up_lookup: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, block in enumerate(blocks):
        if "up" not in block.arms:
            continue
        edge = block.core[0, :]
        for off, color in enumerate(edge.tolist()):
            up_lookup[(int(off), int(color))].append(idx)

    down_of: dict[int, int] = {}
    for idx, block in enumerate(blocks):
        arm = block.arms.get("down")
        if arm is None:
            continue
        off, color = arm
        candidates = [j for j in up_lookup.get((int(off), int(color)), []) if j != idx]
        if len(candidates) > 1:
            filtered = []
            for j in candidates:
                target = blocks[j]
                up_off, up_color = target.arms["up"]
                if int(block.core[-1, up_off]) == int(up_color):
                    filtered.append(j)
            if filtered:
                candidates = filtered
        assert len(candidates) == 1, "down arm must identify a unique lower neighbor"
        down_of[idx] = candidates[0]

    values = list(down_of.values())
    assert len(values) == len(set(values)), "two blocks point at the same vertical target"
    return down_of


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble same-sized square blocks using their one-pixel directional connector arms.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    comps = _components(g != BG)
    assert comps, "input must contain at least one block"
    blocks = [_parse_block(g, comp) for comp in comps]

    side_lengths = {int(block.core.shape[0]) for block in blocks}
    assert len(side_lengths) == 1, "all blocks must have the same square size"
    side = next(iter(side_lengths))

    roots = [idx for idx, block in enumerate(blocks) if "left" not in block.arms and "up" not in block.arms]
    assert len(roots) == 1, "there must be exactly one top-left block"
    root = roots[0]

    right_of = _match_horizontal(blocks)
    down_of = _match_vertical(blocks)

    coords: dict[int, tuple[int, int]] = {root: (0, 0)}
    q = deque([root])
    while q:
        idx = q.popleft()
        row, col = coords[idx]
        if idx in right_of:
            nbr = right_of[idx]
            target = (row, col + 1)
            if nbr in coords:
                assert coords[nbr] == target
            else:
                coords[nbr] = target
                q.append(nbr)
        if idx in down_of:
            nbr = down_of[idx]
            target = (row + 1, col)
            if nbr in coords:
                assert coords[nbr] == target
            else:
                coords[nbr] = target
                q.append(nbr)

    assert len(coords) == len(blocks), "all blocks must be reachable from the top-left block"
    max_row = max(r for r, _ in coords.values())
    max_col = max(c for _, c in coords.values())
    out = np.zeros(((max_row + 1) * side, (max_col + 1) * side), dtype=int)

    used_slots = set()
    for idx, (row, col) in coords.items():
        assert (row, col) not in used_slots
        used_slots.add((row, col))
        r0 = row * side
        c0 = col * side
        out[r0 : r0 + side, c0 : c0 + side] = blocks[idx].core

    assert len(used_slots) == out.shape[0] * out.shape[1] // (side * side)
    return out


def _fill_pattern(rng: random.Random, height: int, width: int, palette: list[int]) -> np.ndarray:
    arr = np.zeros((height, width), dtype=int)
    mode = rng.choice(PATTERN_MODES)

    if mode == "noise":
        for r in range(height):
            for c in range(width):
                arr[r, c] = rng.choice(palette)
    elif mode == "row_bands":
        for r in range(height):
            arr[r, :] = rng.choice(palette)
    elif mode == "col_bands":
        for c in range(width):
            arr[:, c] = rng.choice(palette)
    elif mode == "diag_bands":
        step = rng.randint(1, max(1, min(height, width) // 3))
        bias = rng.randint(0, len(palette) - 1)
        flip = rng.choice((-1, 1))
        for r in range(height):
            for c in range(width):
                idx = ((r + flip * c) // step + bias) % len(palette)
                arr[r, c] = palette[idx]
    else:
        base = rng.choice(palette)
        arr.fill(base)

    rect_count = rng.randint(2, 5)
    for _ in range(rect_count):
        color = rng.choice(palette)
        r0 = rng.randint(0, height - 1)
        c0 = rng.randint(0, width - 1)
        r1 = rng.randint(r0, min(height - 1, r0 + max(1, height // 2)))
        c1 = rng.randint(c0, min(width - 1, c0 + max(1, width // 2)))
        arr[r0 : r1 + 1, c0 : c1 + 1] = color

    color_count = min(len(palette), rng.randint(2, min(len(palette), 4)))
    accent_colors = rng.sample(palette, color_count)
    accent_cells = rng.randint(max(1, height * width // 10), max(2, height * width // 4))
    for _ in range(accent_cells):
        arr[rng.randrange(height), rng.randrange(width)] = rng.choice(accent_colors)

    if len(set(map(int, arr.flatten()))) < 2:
        arr[0, 0] = palette[0]
        arr[-1, -1] = palette[-1]
    return arr


def _choose_keys(
    rng: random.Random,
    rows: int,
    cols: int,
    side: int,
    palette: list[int],
) -> tuple[dict[tuple[int, int], tuple[int, int]], dict[tuple[int, int], tuple[int, int]]]:
    pairs = [(off, color) for off in range(side) for color in palette]
    left_positions = [(r, c) for r in range(rows) for c in range(1, cols)]
    up_positions = [(r, c) for r in range(1, rows) for c in range(cols)]

    for _ in range(100):
        left_pool = pairs.copy()
        up_pool = pairs.copy()
        rng.shuffle(left_pool)
        rng.shuffle(up_pool)

        left_keys: dict[tuple[int, int], tuple[int, int]] = {}
        for pos in left_positions:
            left_keys[pos] = left_pool.pop()

        up_keys: dict[tuple[int, int], tuple[int, int]] = {}
        ok = True
        for pos in up_positions:
            left_key = left_keys.get(pos)
            pick = None
            for idx, key in enumerate(up_pool):
                if left_key is not None and left_key[0] == 0 and key[0] == 0 and key[1] != left_key[1]:
                    continue
                pick = idx
                break
            if pick is None:
                ok = False
                break
            up_keys[pos] = up_pool.pop(pick)
        if ok:
            return left_keys, up_keys

    raise RuntimeError("failed to choose non-conflicting connector keys")


def _rects_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    at, al, ab, ar = a
    bt, bl, bb, br = b
    return not (ab + 1 < bt or bb + 1 < at or ar + 1 < bl or br + 1 < al)


def _place_blocks(
    rng: random.Random,
    blocks: list[Block],
    height: int,
    width: int,
) -> dict[int, tuple[int, int]] | None:
    order = list(range(len(blocks)))
    rng.shuffle(order)
    placements: dict[int, tuple[int, int]] = {}
    rects: list[tuple[int, int, int, int]] = []

    for idx in order:
        block = blocks[idx]
        side = int(block.core.shape[0])
        up = 1 if "up" in block.arms else 0
        down = 1 if "down" in block.arms else 0
        left = 1 if "left" in block.arms else 0
        right = 1 if "right" in block.arms else 0

        min_r = up + 1
        max_r = height - side - down - 1
        min_c = left + 1
        max_c = width - side - right - 1
        if min_r > max_r or min_c > max_c:
            return None

        placed = False
        for _ in range(400):
            r0 = rng.randint(min_r, max_r)
            c0 = rng.randint(min_c, max_c)
            rect = (r0 - up, c0 - left, r0 + side + down - 1, c0 + side + right - 1)
            if any(_rects_too_close(rect, old) for old in rects):
                continue
            placements[idx] = (r0, c0)
            rects.append(rect)
            placed = True
            break
        if not placed:
            return None

    return placements


def _build_instance(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    rows = rng.randint(2, 4)
    cols = rng.randint(2, 4)
    side = rng.randint(3, 5)

    max_in_degree = max(rows * (cols - 1), (rows - 1) * cols)
    min_palette = max(4, math.ceil(max_in_degree / side))
    palette_size = rng.randint(min_palette, min(7, 9))
    palette = rng.sample(list(range(1, 10)), palette_size)

    full = _fill_pattern(rng, rows * side, cols * side, palette)
    left_keys, up_keys = _choose_keys(rng, rows, cols, side, palette)

    for (r, c), (off, color) in left_keys.items():
        full[r * side + off, c * side] = int(color)
    for (r, c), (off, color) in up_keys.items():
        full[r * side, c * side + off] = int(color)

    left_back = {(r, c): rng.randrange(side) for r in range(rows) for c in range(1, cols)}
    up_back = {(r, c): rng.randrange(side) for r in range(1, rows) for c in range(cols)}

    blocks: list[Block] = []
    for r in range(rows):
        for c in range(cols):
            core = full[r * side : (r + 1) * side, c * side : (c + 1) * side].copy()
            arms: dict[str, tuple[int, int]] = {}
            if c > 0:
                off = left_back[(r, c)]
                arms["left"] = (off, int(full[r * side + off, c * side - 1]))
            if c < cols - 1:
                off, color = left_keys[(r, c + 1)]
                arms["right"] = (int(off), int(color))
            if r > 0:
                off = up_back[(r, c)]
                arms["up"] = (off, int(full[r * side - 1, c * side + off]))
            if r < rows - 1:
                off, color = up_keys[(r + 1, c)]
                arms["down"] = (int(off), int(color))
            blocks.append(Block(core=core, arms=arms))

    count = len(blocks)
    pitch = side + 4
    base = max(side + 4, int(math.ceil(math.sqrt(count))) * pitch)
    height = rng.randint(base, base + pitch + 6)
    width = rng.randint(base, base + pitch + 6)

    placements = _place_blocks(rng, blocks, height, width)
    if placements is None:
        raise RuntimeError("failed to place scattered blocks")

    grid = np.zeros((height, width), dtype=int)
    for idx, (r0, c0) in placements.items():
        block = blocks[idx]
        grid[r0 : r0 + side, c0 : c0 + side] = block.core
        for side_name, (off, color) in block.arms.items():
            if side_name == "up":
                grid[r0 - 1, c0 + off] = int(color)
            elif side_name == "down":
                grid[r0 + side, c0 + off] = int(color)
            elif side_name == "left":
                grid[r0 + off, c0 - 1] = int(color)
            else:
                grid[r0 + off, c0 + side] = int(color)

    return grid, full


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        try:
            grid, expected = _build_instance(rng)
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid input")
