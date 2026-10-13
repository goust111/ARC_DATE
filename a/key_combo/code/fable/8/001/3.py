import random
from collections import Counter
from dataclasses import dataclass

import numpy as np


DIR_BY_SIDE = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
ALL_COLORS = tuple(range(10))
GEN_RETRIES = 500
SHAPE_RETRIES = 80
PLACEMENT_RETRIES = 80


@dataclass(frozen=True)
class Sprite:
    cells: tuple[tuple[int, int, int], ...]
    occupied: frozenset[tuple[int, int]]
    anchor: tuple[int, int]
    direction: tuple[int, int]
    marker_color: int
    body_color: int


def _get_bg(grid: np.ndarray) -> int:
    flat = list(map(int, np.asarray(grid, dtype=int).ravel()))
    return Counter(flat).most_common(1)[0][0]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _component_coords(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _parse_sprites(grid: np.ndarray) -> tuple[int, list[Sprite], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    comps = _component_coords(g != bg)
    assert len(comps) >= 2, "expected multiple sprites"

    sprites: list[Sprite] = []
    marker_colors = []
    for comp in comps:
        colors = Counter(int(g[r, c]) for r, c in comp)
        assert len(colors) == 2, f"sprite must use exactly two colors, got {colors}"
        marker_candidates = [color for color, cnt in colors.items() if cnt == 1]
        body_candidates = [color for color, cnt in colors.items() if cnt > 1]
        assert len(marker_candidates) == 1 and len(body_candidates) == 1
        marker_color = marker_candidates[0]
        body_color = body_candidates[0]
        marker_colors.append(marker_color)

        marker_rc = next((r, c) for r, c in comp if int(g[r, c]) == marker_color)
        r0, c0, r1, c1 = _bbox(comp)
        sides = []
        if marker_rc[0] == r0:
            sides.append("up")
        if marker_rc[0] == r1:
            sides.append("down")
        if marker_rc[1] == c0:
            sides.append("left")
        if marker_rc[1] == c1:
            sides.append("right")
        assert len(sides) == 1, f"marker must lie on exactly one bbox side, got {sides}"
        direction = DIR_BY_SIDE[sides[0]]

        cells = []
        occupied = set()
        for r, c in comp:
            cells.append((r - r0, c - c0, int(g[r, c])))
            occupied.add((r - r0, c - c0))
        sprites.append(
            Sprite(
                cells=tuple(sorted(cells)),
                occupied=frozenset(occupied),
                anchor=(r0, c0),
                direction=direction,
                marker_color=marker_color,
                body_color=body_color,
            )
        )

    color_counts = Counter(marker_colors)
    assert len(color_counts) == 2, "expected one unique start marker color and one shared marker color"
    start_colors = [color for color, cnt in color_counts.items() if cnt == 1]
    shared_colors = [color for color, cnt in color_counts.items() if cnt == len(sprites) - 1]
    assert len(start_colors) == 1 and len(shared_colors) == 1
    start_idx = next(i for i, sprite in enumerate(sprites) if sprite.marker_color == start_colors[0])
    return bg, sprites, start_idx


def _occupied_at(sprite: Sprite, anchor: tuple[int, int]) -> set[tuple[int, int]]:
    ar, ac = anchor
    return {(ar + dr, ac + dc) for dr, dc in sprite.occupied}


def _slide_once(
    idx: int,
    sprites: list[Sprite],
    positions: list[tuple[int, int]],
    shape: tuple[int, int],
) -> tuple[tuple[int, int], int | None, int]:
    h, w = shape
    sprite = sprites[idx]
    dr, dc = sprite.direction
    cur_r, cur_c = positions[idx]
    occupied_by_other = [
        _occupied_at(other, positions[j]) if j != idx else set()
        for j, other in enumerate(sprites)
    ]

    steps = 0
    while True:
        next_r = cur_r + dr
        next_c = cur_c + dc
        blockers: set[int] = set()
        out_of_bounds = False
        for rr, cc in sprite.occupied:
            nr = next_r + rr
            nc = next_c + cc
            if not (0 <= nr < h and 0 <= nc < w):
                out_of_bounds = True
                break
            for j, cells in enumerate(occupied_by_other):
                if j != idx and (nr, nc) in cells:
                    blockers.add(j)
        if out_of_bounds or blockers:
            assert len(blockers) <= 1, "next step may not collide with multiple sprites"
            blocker = next(iter(blockers)) if blockers else None
            return (cur_r, cur_c), blocker, steps
        cur_r, cur_c = next_r, next_c
        steps += 1


def _simulate(
    sprites: list[Sprite],
    start_idx: int,
    shape: tuple[int, int],
) -> tuple[list[tuple[int, int]], list[int], list[dict[str, int | None]]]:
    positions = [sprite.anchor for sprite in sprites]
    moved: set[int] = set()
    order: list[int] = []
    events: list[dict[str, int | None]] = []
    active = start_idx

    while active is not None and active not in moved:
        final_anchor, blocker, steps = _slide_once(active, sprites, positions, shape)
        positions[active] = final_anchor
        moved.add(active)
        order.append(active)
        events.append(
            {
                "sprite": active,
                "blocker": blocker,
                "steps": steps,
            }
        )
        if blocker is None or blocker in moved:
            active = None
        else:
            active = blocker
    return positions, order, events


def _render(shape: tuple[int, int], bg: int, sprites: list[Sprite], positions: list[tuple[int, int]]) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for sprite, (ar, ac) in zip(sprites, positions):
        for dr, dc, color in sprite.cells:
            rr = ar + dr
            cc = ac + dc
            assert out[rr, cc] == bg, "sprites may not overlap"
            out[rr, cc] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide the unique-start marked sprite first, then propagate motion by collision through compass-marked sprites.

    Input structure:
    - A background canvas contains several 4-connected sprites.
    - Each sprite is monochrome except for one embedded marker pixel of another color.
    - The marker sits on exactly one side of that sprite's bounding box, encoding up, down, left, or right.
    - Exactly one sprite has a globally unique marker color; every other sprite uses one shared non-background marker color.

    Rule:
    - Starting from the unique-marker sprite, slide the whole sprite one cell at a time in the direction named by its own marker side.
    - Keep sliding while every next occupied cell stays in bounds and lands on background; stop immediately before the next step would leave the grid or overlap another sprite.
    - If a sprite stops because the next step would overlap an unmoved sprite, that struck sprite activates next and slides by its own marker direction under the updated obstacle layout.
    - Each sprite moves at most once; unmoved sprites stay fixed forever, moved sprites still block later motion, and all colors are preserved.

    Key insight:
    - The marker is both identity and compass, and the output depends on sequentially simulating one causal collision chain rather than moving all sprites independently.
    """
    bg, sprites, start_idx = _parse_sprites(np.asarray(grid, dtype=int))
    final_positions, _order, _events = _simulate(sprites, start_idx, grid.shape)
    return _render(grid.shape, bg, sprites, final_positions)


def _random_connected_shape() -> tuple[set[tuple[int, int]], list[str]]:
    for _ in range(SHAPE_RETRIES):
        h = random.randint(3, 6)
        w = random.randint(3, 6)
        target = random.randint(max(5, (h * w + 1) // 2), h * w)
        cells = {(random.randrange(h), random.randrange(w))}
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        r0, c0, r1, c1 = _bbox(list(cells))
        cells = {(r - r0, c - c0) for r, c in cells}
        h2 = r1 - r0 + 1
        w2 = c1 - c0 + 1
        sides = []
        if any((0, c) in cells for c in range(1, w2 - 1)):
            sides.append("up")
        if any((h2 - 1, c) in cells for c in range(1, w2 - 1)):
            sides.append("down")
        if any((r, 0) in cells for r in range(1, h2 - 1)):
            sides.append("left")
        if any((r, w2 - 1) in cells for r in range(1, h2 - 1)):
            sides.append("right")
        if len(cells) >= 5 and sides:
            return cells, sides
    raise RuntimeError("failed to sample shape")


def _make_sprite(body_color: int, marker_color: int, forced_side: str | None = None) -> Sprite:
    while True:
        cells, sides = _random_connected_shape()
        if forced_side is None:
            side = random.choice(sides)
        elif forced_side in sides:
            side = forced_side
        else:
            continue
        h = 1 + max(r for r, _c in cells)
        w = 1 + max(c for _r, c in cells)
        if side == "up":
            marker = (0, random.choice([c for c in range(1, w - 1) if (0, c) in cells]))
        elif side == "down":
            marker = (h - 1, random.choice([c for c in range(1, w - 1) if (h - 1, c) in cells]))
        elif side == "left":
            marker = (random.choice([r for r in range(1, h - 1) if (r, 0) in cells]), 0)
        else:
            marker = (random.choice([r for r in range(1, h - 1) if (r, w - 1) in cells]), w - 1)
        sprite_cells = []
        for r, c in sorted(cells):
            color = marker_color if (r, c) == marker else body_color
            sprite_cells.append((r, c, color))
        return Sprite(
            cells=tuple(sprite_cells),
            occupied=frozenset(cells),
            anchor=(0, 0),
            direction=DIR_BY_SIDE[side],
            marker_color=marker_color,
            body_color=body_color,
        )


def _sprite_with_anchor(sprite: Sprite, anchor: tuple[int, int]) -> Sprite:
    return Sprite(
        cells=sprite.cells,
        occupied=sprite.occupied,
        anchor=anchor,
        direction=sprite.direction,
        marker_color=sprite.marker_color,
        body_color=sprite.body_color,
    )


def _touches_existing(
    sprite: Sprite,
    anchor: tuple[int, int],
    placed: list[Sprite],
    shape: tuple[int, int],
) -> bool:
    h, w = shape
    cells = _occupied_at(sprite, anchor)
    if any(not (0 <= r < h and 0 <= c < w) for r, c in cells):
        return True
    forbidden = set()
    for other in placed:
        for r, c in _occupied_at(other, other.anchor):
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    forbidden.add((r + dr, c + dc))
    return any(cell in forbidden for cell in cells)


def _candidate_anchors(
    sprite: Sprite,
    placed: list[Sprite],
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    max_r = shape[0] - (1 + max(r for r, _c in sprite.occupied))
    max_c = shape[1] - (1 + max(c for _r, c in sprite.occupied))
    anchors = [
        (r, c)
        for r in range(max_r + 1)
        for c in range(max_c + 1)
        if not _touches_existing(sprite, (r, c), placed, shape)
    ]
    random.shuffle(anchors)
    return anchors


def _prefix_ok(order: list[int], events: list[dict[str, int | None]], upto: int) -> bool:
    if len(order) < upto + 1 or order[: upto + 1] != list(range(upto + 1)):
        return False
    for i in range(upto):
        if events[i]["blocker"] != i + 1 or int(events[i]["steps"]) <= 0:
            return False
    return int(events[upto]["steps"]) > 0


def _pick_mode() -> str:
    global _MODE_BAG
    if not _MODE_BAG:
        _MODE_BAG = ["boundary", "boundary", "moved", "boundary", "moved"]
        random.shuffle(_MODE_BAG)
    return _MODE_BAG.pop()


_MODE_BAG: list[str] = []


def _make_rect_sprite(body_color: int, marker_color: int, side: str, h: int, w: int) -> Sprite:
    assert h >= 2 and w >= 2
    if side in {"up", "down"}:
        assert w >= 3
    else:
        assert h >= 3
    if side == "up":
        marker = (0, random.randint(1, w - 2))
    elif side == "down":
        marker = (h - 1, random.randint(1, w - 2))
    elif side == "left":
        marker = (random.randint(1, h - 2), 0)
    else:
        marker = (random.randint(1, h - 2), w - 1)
    cells = []
    occupied = set()
    for r in range(h):
        for c in range(w):
            occupied.add((r, c))
            cells.append((r, c, marker_color if (r, c) == marker else body_color))
    return Sprite(
        cells=tuple(cells),
        occupied=frozenset(occupied),
        anchor=(0, 0),
        direction=DIR_BY_SIDE[side],
        marker_color=marker_color,
        body_color=body_color,
    )


def _shift_all(sprites: list[Sprite], dr: int, dc: int) -> list[Sprite]:
    return [_sprite_with_anchor(sprite, (sprite.anchor[0] + dr, sprite.anchor[1] + dc)) for sprite in sprites]


def _chain_bbox(sprites: list[Sprite]) -> tuple[int, int]:
    max_r = 0
    max_c = 0
    for sprite in sprites:
        h = 1 + max(r for r, _c in sprite.occupied)
        w = 1 + max(c for _r, c in sprite.occupied)
        max_r = max(max_r, sprite.anchor[0] + h)
        max_c = max(max_c, sprite.anchor[1] + w)
    return max_r, max_c


def _build_chain_sprites(
    body_colors: list[int],
    start_marker: int,
    shared_marker: int,
    terminal_mode: str,
    boundary_chain_len: int | None = None,
) -> tuple[list[Sprite], int]:
    if terminal_mode == "moved":
        chain_len = 3
        g1 = random.randint(1, 4)
        g2 = random.randint(1, 2)
        delta = random.randint(1, 2)
        h_b = random.randint(2, 3)
        h_a = max(random.randint(5, 6), h_b + g2 + 2)
        w_a = random.randint(3, 5)
        w_b = random.randint(delta + 2, 5)
        w_c = random.randint(3, 5)
        h_c = random.randint(3, 4)
        base = [
            _make_rect_sprite(body_colors[0], start_marker, "right", h_a, w_a),
            _make_rect_sprite(body_colors[1], shared_marker, "down", h_b, w_b),
            _make_rect_sprite(body_colors[2], shared_marker, "left", h_c, w_c),
        ]
        placed = [
            _sprite_with_anchor(base[0], (0, 0)),
            _sprite_with_anchor(base[1], (1, w_a + g1)),
            _sprite_with_anchor(base[2], (1 + h_b + g2, w_a + g1 + delta)),
        ]
        return placed, chain_len

    chain_len = 3 if boundary_chain_len is None else boundary_chain_len
    if chain_len == 3:
        g1 = random.randint(1, 4)
        g2 = random.randint(1, 4)
        base = [
            _make_rect_sprite(body_colors[0], start_marker, "right", random.randint(3, 5), random.randint(3, 5)),
            _make_rect_sprite(body_colors[1], shared_marker, "down", random.randint(3, 5), random.randint(3, 5)),
            _make_rect_sprite(body_colors[2], shared_marker, "left", random.randint(3, 5), random.randint(3, 5)),
        ]
        h_ab = max(
            1 + max(r for r, _c in base[0].occupied),
            1 + max(r for r, _c in base[1].occupied),
        )
        c_overlap = random.randint(
            0,
            min(
                max(c for _r, c in base[1].occupied),
                max(c for _r, c in base[2].occupied),
            ),
        )
        placed = [
            _sprite_with_anchor(base[0], (0, 0)),
            _sprite_with_anchor(base[1], (0, (1 + max(c for _r, c in base[0].occupied)) + g1)),
            _sprite_with_anchor(base[2], (h_ab + g2, (1 + max(c for _r, c in base[0].occupied)) + g1 + c_overlap)),
        ]
        return placed, chain_len

    g1 = random.randint(1, 4)
    g2 = random.randint(1, 4)
    g3 = random.randint(1, 4)
    base = [
        _make_rect_sprite(body_colors[0], start_marker, "right", random.randint(3, 5), random.randint(3, 5)),
        _make_rect_sprite(body_colors[1], shared_marker, "down", random.randint(3, 5), random.randint(3, 5)),
        _make_rect_sprite(body_colors[2], shared_marker, "left", random.randint(3, 5), random.randint(3, 5)),
        _make_rect_sprite(body_colors[3], shared_marker, "up", random.randint(3, 5), random.randint(3, 5)),
    ]
    w_d = 1 + max(c for _r, c in base[3].occupied)
    w_a = 1 + max(c for _r, c in base[0].occupied)
    w_b = 1 + max(c for _r, c in base[1].occupied)
    h_ab = max(
        1 + max(r for r, _c in base[0].occupied),
        1 + max(r for r, _c in base[1].occupied),
    )
    row_cd = h_ab + g2
    placed = [
        _sprite_with_anchor(base[0], (0, w_d + g3)),
        _sprite_with_anchor(base[1], (0, w_d + g3 + w_a + g1)),
        _sprite_with_anchor(base[2], (row_cd, w_d + g3 + w_a + g1)),
        _sprite_with_anchor(base[3], (row_cd, 0)),
    ]
    return placed, chain_len


def _random_transform(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int)
    out = np.rot90(out, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    return out.copy()


def _build_scene() -> np.ndarray | None:
    terminal_mode = _pick_mode()
    extra_count = random.randint(1, 2)
    boundary_chain_len = None if terminal_mode == "moved" else random.choice([3, 4])
    total_sprites = (3 if terminal_mode == "moved" else boundary_chain_len) + extra_count

    bg = random.choice(ALL_COLORS)
    marker_colors = random.sample([c for c in ALL_COLORS if c != bg], 2)
    start_marker, shared_marker = marker_colors
    body_pool = [c for c in ALL_COLORS if c not in {bg, start_marker, shared_marker}]
    if total_sprites > len(body_pool):
        return None
    body_colors = random.sample(body_pool, total_sprites)

    chain_sprites, chain_len = _build_chain_sprites(
        body_colors,
        start_marker,
        shared_marker,
        terminal_mode,
        boundary_chain_len=boundary_chain_len,
    )
    local_h, local_w = _chain_bbox(chain_sprites)
    shape = (
        min(30, local_h + random.randint(5, 9)),
        min(30, local_w + random.randint(5, 9)),
    )
    slack_r = shape[0] - local_h
    slack_c = shape[1] - local_w
    placed = _shift_all(chain_sprites, random.randint(0, slack_r), random.randint(0, slack_c))

    try:
        _final_positions, final_order, final_events = _simulate(placed, 0, shape)
    except AssertionError:
        return None
    if final_order != list(range(chain_len)):
        return None
    if any(int(event["steps"]) <= 0 for event in final_events):
        return None
    if terminal_mode == "boundary" and final_events[-1]["blocker"] is not None:
        return None
    if terminal_mode == "moved":
        blocker = final_events[-1]["blocker"]
        if blocker is None or int(blocker) >= chain_len - 1:
            return None

    for idx in range(chain_len, total_sprites):
        side = random.choice(list(DIR_BY_SIDE))
        dormant = _make_rect_sprite(body_colors[idx], shared_marker, side, random.randint(3, 5), random.randint(3, 5))
        found = None
        for anchor in _candidate_anchors(dormant, placed, shape)[:120]:
            trial = placed + [_sprite_with_anchor(dormant, anchor)]
            try:
                _trial_positions, order, events = _simulate(trial, 0, shape)
            except AssertionError:
                continue
            if order == final_order and events == final_events:
                found = _sprite_with_anchor(dormant, anchor)
                break
        if found is None:
            return None
        placed.append(found)

    grid = _render(shape, bg, placed, [sprite.anchor for sprite in placed])
    grid = _random_transform(grid)
    if not _semantic_check(grid, expected_chain=chain_len, terminal_mode=terminal_mode):
        return None
    return grid


def _semantic_check(grid: np.ndarray, expected_chain: int | None = None, terminal_mode: str | None = None) -> bool:
    try:
        bg, sprites, start_idx = _parse_sprites(grid)
        final_positions, order, events = _simulate(sprites, start_idx, grid.shape)
        out = _render(grid.shape, bg, sprites, final_positions)
        solved = main(grid)
    except AssertionError:
        return False

    if not np.array_equal(out, solved):
        return False
    if np.array_equal(grid, out):
        return False
    if len(order) < 2:
        return False
    if any(int(event["steps"]) <= 0 for event in events):
        return False
    marker_counts = Counter(sprite.marker_color for sprite in sprites)
    if sorted(marker_counts.values()) != [1, len(sprites) - 1]:
        return False
    if expected_chain is not None and len(order) != expected_chain:
        return False
    if terminal_mode == "boundary" and events[-1]["blocker"] is not None:
        return False
    if terminal_mode == "moved":
        blocker = events[-1]["blocker"]
        if blocker is None or int(blocker) >= len(order) - 1:
            return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_scene()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid compass-relay sliding input")
