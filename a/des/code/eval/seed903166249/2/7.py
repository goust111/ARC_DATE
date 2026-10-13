import numpy as np
from dataclasses import dataclass


BG = 0
ARROW_TO_DIR = {
    8: "U",  # azure
    9: "D",  # maroon
    6: "L",  # magenta
    5: "R",  # grey
}
DIR_TO_ARROW = {direction: color for color, direction in ARROW_TO_DIR.items()}
DIR_VECTORS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}

# Fixed visible-priority palette used consistently across all generated examples.
BORDER_PRIORITY = (2, 1, 3, 4, 7)  # red > blue > green > yellow > orange
PRIORITY_VALUE = {color: len(BORDER_PRIORITY) - index for index, color in enumerate(BORDER_PRIORITY)}

GRID_MIN = 12
GRID_MAX = 20
GEN_ATTEMPTS = 400
EXTRA_PLACEMENT_ATTEMPTS = 80


@dataclass
class Block:
    ident: int
    row: int
    col: int
    border: int
    direction: str
    alive: bool = True
    moving: bool = True

    def clone(self) -> "Block":
        return Block(self.ident, self.row, self.col, self.border, self.direction, self.alive, self.moving)


def _priority(color: int) -> int:
    return int(PRIORITY_VALUE.get(int(color), -1))


def _center_in_bounds(row: int, col: int, height: int, width: int) -> bool:
    return 1 <= int(row) < int(height - 1) and 1 <= int(col) < int(width - 1)


def _touch_or_overlap(pos_a: tuple[int, int], pos_b: tuple[int, int]) -> bool:
    dr = abs(int(pos_a[0]) - int(pos_b[0]))
    dc = abs(int(pos_a[1]) - int(pos_b[1]))
    return (dr <= 2 and dc <= 3) or (dr <= 3 and dc <= 2)


def _render(blocks: list[Block], height: int, width: int) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for block in blocks:
        if not block.alive:
            continue
        r = int(block.row)
        c = int(block.col)
        grid[r - 1 : r + 2, c - 1 : c + 2] = int(block.border)
        grid[r, c] = int(DIR_TO_ARROW[block.direction])
    return grid


def _find_blocks(grid: np.ndarray) -> list[Block]:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    seen_centers: set[tuple[int, int]] = set()
    blocks: list[Block] = []

    for row in range(1, height - 1):
        for col in range(1, width - 1):
            center = int(g[row, col])
            if center not in ARROW_TO_DIR:
                continue

            border_values = []
            ok = True
            for rr in range(row - 1, row + 2):
                for cc in range(col - 1, col + 2):
                    if rr == row and cc == col:
                        continue
                    value = int(g[rr, cc])
                    border_values.append(value)
                    if value == BG or value in ARROW_TO_DIR:
                        ok = False
            if not ok:
                continue

            border = int(border_values[0])
            if border not in PRIORITY_VALUE:
                continue
            if any(int(value) != border for value in border_values):
                continue
            if (row, col) in seen_centers:
                continue

            seen_centers.add((row, col))
            blocks.append(Block(len(blocks), int(row), int(col), border, ARROW_TO_DIR[center]))

    return blocks


def _collision_components(blocks: list[Block], proposed: dict[int, tuple[int, int]]) -> list[list[int]]:
    alive_ids = [int(block.ident) for block in blocks if block.alive]
    neighbors = {ident: set() for ident in alive_ids}
    for index, ident_a in enumerate(alive_ids):
        for ident_b in alive_ids[index + 1 :]:
            if _touch_or_overlap(proposed[ident_a], proposed[ident_b]):
                neighbors[ident_a].add(ident_b)
                neighbors[ident_b].add(ident_a)

    components: list[list[int]] = []
    visited: set[int] = set()
    for ident in alive_ids:
        if ident in visited or not neighbors[ident]:
            continue
        stack = [ident]
        comp: list[int] = []
        visited.add(ident)
        while stack:
            current = stack.pop()
            comp.append(current)
            for nxt in neighbors[current]:
                if nxt not in visited:
                    visited.add(nxt)
                    stack.append(nxt)
        components.append(comp)
    return components


def _simulate(
    blocks: list[Block],
    height: int,
    width: int,
) -> tuple[list[Block], dict]:
    states = [block.clone() for block in blocks]
    by_id = {int(block.ident): block for block in states}
    collision_wins = {int(block.ident): 0 for block in states}
    collision_count = 0
    max_component_size = 1
    ambiguous = False

    max_steps = max(1, height * width * max(1, len(states)) * 4)
    for _step in range(max_steps):
        live_states = [block for block in states if block.alive]
        if not any(block.moving for block in live_states):
            break

        proposed: dict[int, tuple[int, int]] = {}
        moved: dict[int, bool] = {}
        for block in live_states:
            if not block.moving:
                proposed[block.ident] = (int(block.row), int(block.col))
                moved[block.ident] = False
                continue
            dr, dc = DIR_VECTORS[block.direction]
            nr = int(block.row + dr)
            nc = int(block.col + dc)
            if _center_in_bounds(nr, nc, height, width):
                proposed[block.ident] = (nr, nc)
                moved[block.ident] = True
            else:
                proposed[block.ident] = (int(block.row), int(block.col))
                moved[block.ident] = False

        components = _collision_components(live_states, proposed)
        if components:
            max_component_size = max(max_component_size, max(len(component) for component in components))
            if any(len(component) > 2 for component in components):
                ambiguous = True

        in_collision = {ident for component in components for ident in component}

        for block in live_states:
            if block.ident in in_collision:
                continue
            block.row, block.col = proposed[block.ident]
            block.moving = bool(block.moving and moved[block.ident])

        for component in components:
            participants = [by_id[ident] for ident in component]
            participants.sort(key=lambda block: (-_priority(block.border), int(block.ident)))
            winner = participants[0]
            losers = participants[1:]
            if not losers:
                continue

            winner.row, winner.col = proposed[winner.ident]
            inherited = max(losers, key=lambda block: (_priority(block.border), -int(block.ident)))
            winner.direction = inherited.direction
            winner.moving = True
            collision_count += len(losers)
            collision_wins[winner.ident] += len(losers)
            for loser in losers:
                loser.alive = False
                loser.moving = False
        if ambiguous:
            break
    else:
        raise RuntimeError("simulation did not converge")

    survivors = [block.clone() for block in states if block.alive]
    stats = {
        "collision_count": int(collision_count),
        "max_component_size": int(max_component_size),
        "ambiguous": bool(ambiguous),
        "max_absorptions_by_one_block": max(collision_wins.values(), default=0),
        "survivor_count": len(survivors),
    }
    return survivors, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide 3x3 framed blocks by their center-arrow directions.

    Every block moves one cell per tick until its 3x3 body would leave the grid.
    When moved positions become edge-adjacent or overlapping, the higher-priority
    border color survives, inherits the loser's arrow direction, and keeps moving.
    """
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    blocks = _find_blocks(g)
    survivors, _stats = _simulate(blocks, height, width)
    return _render(survivors, height, width)


def _random_center(
    rng: np.random.Generator,
    height: int,
    width: int,
    existing: list[Block],
) -> tuple[int, int] | None:
    for _ in range(EXTRA_PLACEMENT_ATTEMPTS):
        row = int(rng.integers(1, height - 1))
        col = int(rng.integers(1, width - 1))
        if any(_touch_or_overlap((row, col), (block.row, block.col)) for block in existing):
            continue
        return row, col
    return None


def _mode_ok(mode: str, stats: dict) -> bool:
    if stats["ambiguous"] or stats["max_component_size"] > 2:
        return False
    if mode == "slide":
        return stats["collision_count"] == 0
    if mode == "single":
        return stats["collision_count"] == 1
    if mode == "chain":
        return stats["collision_count"] == 2
    raise ValueError(mode)


def _add_extras(
    rng: np.random.Generator,
    blocks: list[Block],
    height: int,
    width: int,
    mode: str,
) -> list[Block]:
    result = [block.clone() for block in blocks]
    used_colors = {int(block.border) for block in result}
    available = [color for color in BORDER_PRIORITY if color not in used_colors]
    rng.shuffle(available)
    target_extra = int(rng.integers(0, min(3, len(available)) + 1))

    for color in available:
        if target_extra <= 0:
            break
        placed = False
        for _ in range(EXTRA_PLACEMENT_ATTEMPTS):
            center = _random_center(rng, height, width, result)
            if center is None:
                break
            direction = str(rng.choice(tuple(DIR_VECTORS)))
            candidate = result + [Block(len(result), center[0], center[1], int(color), direction)]
            survivors, stats = _simulate(candidate, height, width)
            if not _mode_ok(mode, stats):
                continue
            if np.array_equal(_render(candidate, height, width), _render(survivors, height, width)):
                continue
            result = candidate
            target_extra -= 1
            placed = True
            break
        if not placed:
            continue
    return result


def _single_layout(
    rng: np.random.Generator,
    height: int,
    width: int,
) -> list[Block] | None:
    colors = list(rng.choice(BORDER_PRIORITY, size=2, replace=False))
    colors.sort(key=_priority, reverse=True)
    stationary_wins = bool(rng.integers(0, 2))
    if stationary_wins:
        b_color, a_color = colors[0], colors[1]
    else:
        a_color, b_color = colors[0], colors[1]

    side = str(rng.choice(("top", "bottom", "left", "right")))
    if side in ("top", "bottom"):
        row = 1 if side == "top" else height - 2
        blocker_dir = "U" if side == "top" else "D"
        attack_dir = str(rng.choice(("R", "L")))
        if attack_dir == "R":
            left_margin = 5
            right_margin = 2
            if width < left_margin + right_margin + 1:
                return None
            blocker_col = int(rng.integers(left_margin, width - right_margin))
            distance = int(rng.integers(4, min(8, blocker_col)))
            attacker_col = blocker_col - distance
        else:
            left_margin = 2
            right_margin = 5
            if width < left_margin + right_margin + 1:
                return None
            blocker_col = int(rng.integers(left_margin, width - right_margin))
            max_distance = width - 1 - blocker_col
            distance = int(rng.integers(4, min(8, max_distance + 1)))
            attacker_col = blocker_col + distance
        attacker = Block(0, row, attacker_col, int(a_color), attack_dir)
        blocker = Block(1, row, blocker_col, int(b_color), blocker_dir)
        return [attacker, blocker]

    col = 1 if side == "left" else width - 2
    blocker_dir = "L" if side == "left" else "R"
    attack_dir = str(rng.choice(("D", "U")))
    if attack_dir == "D":
        top_margin = 5
        bottom_margin = 2
        if height < top_margin + bottom_margin + 1:
            return None
        blocker_row = int(rng.integers(top_margin, height - bottom_margin))
        distance = int(rng.integers(4, min(8, blocker_row)))
        attacker_row = blocker_row - distance
    else:
        top_margin = 2
        bottom_margin = 5
        if height < top_margin + bottom_margin + 1:
            return None
        blocker_row = int(rng.integers(top_margin, height - bottom_margin))
        max_distance = height - 1 - blocker_row
        distance = int(rng.integers(4, min(8, max_distance + 1)))
        attacker_row = blocker_row + distance
    attacker = Block(0, attacker_row, col, int(a_color), attack_dir)
    blocker = Block(1, blocker_row, col, int(b_color), blocker_dir)
    return [attacker, blocker]


def _chain_layout(
    rng: np.random.Generator,
    height: int,
    width: int,
) -> list[Block] | None:
    colors = list(rng.choice(BORDER_PRIORITY, size=3, replace=False))
    colors.sort(key=_priority, reverse=True)
    b_color = colors[0]
    a_color = colors[-1]
    c_color = colors[1]

    side = str(rng.choice(("top", "bottom", "left", "right")))
    if side in ("top", "bottom"):
        row = 1 if side == "top" else height - 2
        stationary_dir = "U" if side == "top" else "D"
        attack_dir = str(rng.choice(("R", "L")))
        if width < 13:
            return None
        if attack_dir == "R":
            blocker_col = int(rng.integers(5, width - 6))
            left_dist = int(rng.integers(4, min(8, blocker_col)))
            right_space = width - 1 - blocker_col
            right_dist = int(rng.integers(4, min(8, right_space + 1)))
            attacker_col = blocker_col - left_dist
            target_col = blocker_col + right_dist
        else:
            blocker_col = int(rng.integers(6, width - 5))
            right_dist = int(rng.integers(4, min(8, width - 1 - blocker_col + 1)))
            left_dist = int(rng.integers(4, min(8, blocker_col)))
            attacker_col = blocker_col + right_dist
            target_col = blocker_col - left_dist
        attacker = Block(0, row, attacker_col, int(a_color), attack_dir)
        blocker = Block(1, row, blocker_col, int(b_color), stationary_dir)
        target = Block(2, row, target_col, int(c_color), stationary_dir)
        return [attacker, blocker, target]

    col = 1 if side == "left" else width - 2
    stationary_dir = "L" if side == "left" else "R"
    attack_dir = str(rng.choice(("D", "U")))
    if height < 13:
        return None
    if attack_dir == "D":
        blocker_row = int(rng.integers(5, height - 6))
        top_dist = int(rng.integers(4, min(8, blocker_row)))
        bottom_space = height - 1 - blocker_row
        bottom_dist = int(rng.integers(4, min(8, bottom_space + 1)))
        attacker_row = blocker_row - top_dist
        target_row = blocker_row + bottom_dist
    else:
        blocker_row = int(rng.integers(6, height - 5))
        bottom_dist = int(rng.integers(4, min(8, height - 1 - blocker_row + 1)))
        top_dist = int(rng.integers(4, min(8, blocker_row)))
        attacker_row = blocker_row + bottom_dist
        target_row = blocker_row - top_dist
    attacker = Block(0, attacker_row, col, int(a_color), attack_dir)
    blocker = Block(1, blocker_row, col, int(b_color), stationary_dir)
    target = Block(2, target_row, col, int(c_color), stationary_dir)
    return [attacker, blocker, target]


def _slide_layout(
    rng: np.random.Generator,
    height: int,
    width: int,
) -> list[Block] | None:
    count = int(rng.integers(2, min(6, len(BORDER_PRIORITY) + 1)))
    colors = list(rng.choice(BORDER_PRIORITY, size=count, replace=False))
    blocks: list[Block] = []
    for ident, color in enumerate(colors):
        center = _random_center(rng, height, width, blocks)
        if center is None:
            return None
        direction = str(rng.choice(tuple(DIR_VECTORS)))
        blocks.append(Block(ident, center[0], center[1], int(color), direction))
    return blocks


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    mode = str(rng.choice(("slide", "single", "chain")))
    height = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    width = int(rng.integers(GRID_MIN, GRID_MAX + 1))

    if mode == "slide":
        blocks = _slide_layout(rng, height, width)
    elif mode == "single":
        blocks = _single_layout(rng, height, width)
    else:
        blocks = _chain_layout(rng, height, width)
    if blocks is None:
        return None

    blocks = _add_extras(rng, blocks, height, width, mode)
    survivors, stats = _simulate(blocks, height, width)
    if not _mode_ok(mode, stats):
        return None

    inp = _render(blocks, height, width)
    out = _render(survivors, height, width)
    if np.array_equal(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
