from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


SPRITE_SIZE = 3
BG_COLOR = 0

GREEN = 3
YELLOW = 4
PINK = 6
TEAL = 8

DIR_BY_COLOR = {
    GREEN: (0, -1),
    YELLOW: (1, 0),
    PINK: (-1, 0),
    TEAL: (0, 1),
}

COLOR_POOL = tuple(DIR_BY_COLOR)

SOLID_MASK = np.ones((SPRITE_SIZE, SPRITE_SIZE), dtype=bool)
HOLLOW_MASK = np.ones((SPRITE_SIZE, SPRITE_SIZE), dtype=bool)
HOLLOW_MASK[1, 1] = False
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
NEIGHBORHOOD8 = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1))


def _get_bg(grid: np.ndarray) -> int:
    counts = Counter(map(int, np.asarray(grid).ravel()))
    return int(counts.most_common(1)[0][0])


def _component_bfs(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    seen = np.zeros((H, W), dtype=bool)
    comps: list[dict] = []

    for r in range(H):
        for c in range(W):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append({"color": color, "coords": coords})
    return comps


def _mask_for_kind(kind: str) -> np.ndarray:
    return SOLID_MASK if kind == "solid" else HOLLOW_MASK


def _sprite_patch(color: int, kind: str, bg: int) -> np.ndarray:
    patch = np.full((SPRITE_SIZE, SPRITE_SIZE), bg, dtype=int)
    patch[_mask_for_kind(kind)] = int(color)
    return patch


def _parse_sprites(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    sprites: list[dict] = []

    for comp in _component_bfs(g, bg):
        color = int(comp["color"])
        assert color in DIR_BY_COLOR, "unexpected sprite color"
        rows = [r for r, _c in comp["coords"]]
        cols = [c for _r, c in comp["coords"]]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        assert (r1 - r0 + 1, c1 - c0 + 1) == (SPRITE_SIZE, SPRITE_SIZE), "expected 3x3 sprite bbox"
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        if np.all(patch == color):
            kind = "solid"
        else:
            expected = np.full((SPRITE_SIZE, SPRITE_SIZE), bg, dtype=int)
            expected[HOLLOW_MASK] = color
            assert np.array_equal(patch, expected), "expected 3x3 solid or hollow-square sprite"
            kind = "hollow"
        sprites.append({"color": color, "kind": kind, "pos": (r0, c0)})

    solids = [i for i, sprite in enumerate(sprites) if sprite["kind"] == "solid"]
    assert len(solids) == 1, "expected exactly one solid trigger"
    sprites.sort(key=lambda sprite: (sprite["pos"][0], sprite["pos"][1], sprite["kind"], sprite["color"]))
    return bg, sprites


def _occupied_cells(kind: str, pos: tuple[int, int]) -> set[tuple[int, int]]:
    mask = _mask_for_kind(kind)
    r0, c0 = pos
    return {
        (r0 + dr, c0 + dc)
        for dr in range(SPRITE_SIZE)
        for dc in range(SPRITE_SIZE)
        if bool(mask[dr, dc])
    }


def _sprites_overlap(
    kind_a: str,
    pos_a: tuple[int, int],
    kind_b: str,
    pos_b: tuple[int, int],
) -> bool:
    return not _occupied_cells(kind_a, pos_a).isdisjoint(_occupied_cells(kind_b, pos_b))


def _sprites_touch(
    kind_a: str,
    pos_a: tuple[int, int],
    kind_b: str,
    pos_b: tuple[int, int],
) -> bool:
    occ_a = _occupied_cells(kind_a, pos_a)
    occ_b = _occupied_cells(kind_b, pos_b)
    for r, c in occ_a:
        for dr, dc in NEIGHBORHOOD8:
            if (r + dr, c + dc) in occ_b:
                return True
    return False


def _in_bounds(shape: tuple[int, int], pos: tuple[int, int]) -> bool:
    H, W = shape
    r0, c0 = pos
    return 0 <= r0 <= H - SPRITE_SIZE and 0 <= c0 <= W - SPRITE_SIZE


def _edge_flush(shape: tuple[int, int], pos: tuple[int, int], color: int) -> bool:
    H, W = shape
    r0, c0 = pos
    dr, dc = DIR_BY_COLOR[int(color)]
    if dr == -1:
        return r0 == 0
    if dr == 1:
        return r0 == H - SPRITE_SIZE
    if dc == -1:
        return c0 == 0
    return c0 == W - SPRITE_SIZE


def _slide_one(
    idx: int,
    sprites: list[dict],
    positions: list[tuple[int, int]],
    shape: tuple[int, int],
) -> tuple[tuple[int, int], int, str, int | None]:
    color = int(sprites[idx]["color"])
    kind = str(sprites[idx]["kind"])
    dr, dc = DIR_BY_COLOR[color]
    cur = positions[idx]
    steps = 0

    while True:
        nxt = (cur[0] + dr, cur[1] + dc)
        if not _in_bounds(shape, nxt):
            return cur, steps, "boundary", None
        blocker = None
        for j, other in enumerate(sprites):
            if j == idx:
                continue
            if _sprites_overlap(kind, nxt, str(other["kind"]), positions[j]):
                blocker = j
                break
        if blocker is not None:
            return cur, steps, "collision" if steps > 0 else "blocked", blocker
        cur = nxt
        steps += 1


def _simulate_chain(
    sprites: list[dict],
    shape: tuple[int, int],
) -> tuple[list[tuple[int, int]], list[dict]]:
    positions = [tuple(sprite["pos"]) for sprite in sprites]
    trigger = next(i for i, sprite in enumerate(sprites) if sprite["kind"] == "solid")
    active = trigger
    seen: set[int] = set()
    log: list[dict] = []

    while True:
        final_pos, steps, reason, blocker = _slide_one(active, sprites, positions, shape)
        positions[active] = final_pos
        log.append(
            {
                "sprite": active,
                "steps": int(steps),
                "reason": reason,
                "blocker": blocker,
            }
        )
        seen.add(active)
        if reason == "collision" and blocker is not None and blocker not in seen:
            active = blocker
            continue
        break

    return positions, log


def _render(
    shape: tuple[int, int],
    bg: int,
    sprites: list[dict],
    positions: list[tuple[int, int]],
    *,
    allow_overlap: bool = False,
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for sprite, pos in zip(sprites, positions):
        patch = _sprite_patch(int(sprite["color"]), str(sprite["kind"]), bg)
        r0, c0 = pos
        view = out[r0 : r0 + SPRITE_SIZE, c0 : c0 + SPRITE_SIZE]
        mask = patch != bg
        if not allow_overlap:
            assert not bool((view != bg)[mask].any()), "sprites must not overlap in final rendering"
        view[mask] = patch[mask]
    return out


def _slide_to_edge_only(sprites: list[dict], shape: tuple[int, int], bg: int) -> np.ndarray:
    positions = []
    H, W = shape
    for sprite in sprites:
        r0, c0 = sprite["pos"]
        dr, dc = DIR_BY_COLOR[int(sprite["color"])]
        if dr == -1:
            pos = (0, c0)
        elif dr == 1:
            pos = (H - SPRITE_SIZE, c0)
        elif dc == -1:
            pos = (r0, 0)
        else:
            pos = (r0, W - SPRITE_SIZE)
        positions.append(pos)
    return _render(shape, bg, sprites, positions, allow_overlap=True)


def _slide_trigger_only(sprites: list[dict], shape: tuple[int, int], bg: int) -> np.ndarray:
    positions = [tuple(sprite["pos"]) for sprite in sprites]
    trigger = next(i for i, sprite in enumerate(sprites) if sprite["kind"] == "solid")
    final_pos, _steps, _reason, _blocker = _slide_one(trigger, sprites, positions, shape)
    positions[trigger] = final_pos
    return _render(shape, bg, sprites, positions)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide only the struck sprites: one solid trigger block starts a color-coded domino relay among 3x3 sprites.

    Input structure:
    - The grid has one dominant background color plus several isolated 3x3 sprites.
    - Exactly one sprite is a solid 3x3 block; every other sprite is a 3x3 hollow square.
    - Sprite colors encode directions: green=left, yellow=down, pink=up, teal=right.

    Rule:
    - The solid trigger slides along its own color direction until one more step would leave the grid or overlap another sprite, then it stops.
    - If it stopped because it hit another sprite after moving, that struck sprite becomes active and now slides along its own color direction by the same stop rule.
    - The relay continues only through struck sprites; untouched sprites stay where they were, and a newly activated sprite that cannot move immediately ends the chain.

    Key insight:
    - This is a single collision-propagated relay, not a global \"move every direction-coded object\" task.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg, sprites = _parse_sprites(g)
    positions, _log = _simulate_chain(sprites, g.shape)
    return _render(g.shape, bg, sprites, positions)


GEN_SHAPE_RANGE = (16, 27)
GEN_CHAIN_MOVE_RANGE = (1, 6)
GEN_CHAIN_LEN_AFTER_MOVE = (2, 5)
GEN_CHAIN_LEN_ZERO_STOP = (3, 5)
GEN_DECOY_RANGE = (1, 4)
GEN_ATTEMPTS = 600


def _rand_shape(rng: random.Random) -> tuple[int, int]:
    return rng.randint(*GEN_SHAPE_RANGE), rng.randint(*GEN_SHAPE_RANGE)


def _max_chain_distance(shape: tuple[int, int], blocker_pos: tuple[int, int], color: int) -> int:
    H, W = shape
    r, c = blocker_pos
    dr, dc = DIR_BY_COLOR[color]
    if dr == -1:
        return H - SPRITE_SIZE - r - 3
    if dr == 1:
        return r - 3
    if dc == -1:
        return W - SPRITE_SIZE - c - 3
    return c - 3


def _backtrack_pos(blocker_pos: tuple[int, int], color: int, steps: int) -> tuple[int, int]:
    dr, dc = DIR_BY_COLOR[color]
    return blocker_pos[0] - (steps + 3) * dr, blocker_pos[1] - (steps + 3) * dc


def _random_terminal_sprite(
    rng: random.Random,
    shape: tuple[int, int],
    zero_step: bool,
) -> dict | None:
    H, W = shape
    color = int(rng.choice(COLOR_POOL))
    dr, dc = DIR_BY_COLOR[color]
    if dr == -1:
        col = rng.randrange(0, W - SPRITE_SIZE + 1)
        if zero_step:
            pos = (0, col)
        else:
            row = rng.randrange(1, H - SPRITE_SIZE + 1)
            pos = (row, col)
        if not zero_step and pos[0] < 1:
            return None
    elif dr == 1:
        col = rng.randrange(0, W - SPRITE_SIZE + 1)
        if zero_step:
            pos = (H - SPRITE_SIZE, col)
        else:
            row = rng.randrange(0, H - SPRITE_SIZE)
            pos = (row, col)
        if not zero_step and H - SPRITE_SIZE - pos[0] < 1:
            return None
    elif dc == -1:
        row = rng.randrange(0, H - SPRITE_SIZE + 1)
        if zero_step:
            pos = (row, 0)
        else:
            col = rng.randrange(1, W - SPRITE_SIZE + 1)
            pos = (row, col)
        if not zero_step and pos[1] < 1:
            return None
    else:
        row = rng.randrange(0, H - SPRITE_SIZE + 1)
        if zero_step:
            pos = (row, W - SPRITE_SIZE)
        else:
            col = rng.randrange(0, W - SPRITE_SIZE)
            pos = (row, col)
        if not zero_step and W - SPRITE_SIZE - pos[1] < 1:
            return None
    return {"color": color, "kind": "hollow", "pos": pos}


def _fits_without_touch(candidate: dict, placed: list[dict], shape: tuple[int, int]) -> bool:
    if not _in_bounds(shape, tuple(candidate["pos"])):
        return False
    for other in placed:
        if _sprites_touch(
            str(candidate["kind"]),
            tuple(candidate["pos"]),
            str(other["kind"]),
            tuple(other["pos"]),
        ):
            return False
    return True


def _make_chain(rng: random.Random, shape: tuple[int, int], zero_step_terminal: bool) -> list[dict] | None:
    chain_len_range = GEN_CHAIN_LEN_ZERO_STOP if zero_step_terminal else GEN_CHAIN_LEN_AFTER_MOVE
    total = rng.randrange(chain_len_range[0], chain_len_range[1])
    last = _random_terminal_sprite(rng, shape, zero_step_terminal)
    if last is None:
        return None

    chain = [last]
    for offset in range(total - 1):
        blocker = chain[0]
        kind = "solid" if offset == total - 2 else "hollow"
        choices = list(COLOR_POOL)
        rng.shuffle(choices)
        placed = None
        for color in choices:
            max_steps = _max_chain_distance(shape, tuple(blocker["pos"]), int(color))
            max_steps = min(max_steps, GEN_CHAIN_MOVE_RANGE[1])
            if max_steps < GEN_CHAIN_MOVE_RANGE[0]:
                continue
            for steps in rng.sample(list(range(GEN_CHAIN_MOVE_RANGE[0], max_steps + 1)), k=max_steps):
                pos = _backtrack_pos(tuple(blocker["pos"]), int(color), int(steps))
                candidate = {"color": int(color), "kind": kind, "pos": pos}
                if _fits_without_touch(candidate, chain, shape):
                    placed = candidate
                    break
            if placed is not None:
                break
        if placed is None:
            return None
        chain.insert(0, placed)
    return chain


def _add_decoys(rng: random.Random, sprites: list[dict], shape: tuple[int, int]) -> list[dict] | None:
    decoy_count = rng.randrange(*GEN_DECOY_RANGE)
    out = list(sprites)
    H, W = shape

    for _ in range(decoy_count):
        placed = None
        for _attempt in range(120):
            color = int(rng.choice(COLOR_POOL))
            dr, dc = DIR_BY_COLOR[color]
            r0 = rng.randrange(0, H - SPRITE_SIZE + 1)
            c0 = rng.randrange(0, W - SPRITE_SIZE + 1)
            if dr == -1 and r0 == 0:
                continue
            if dr == 1 and r0 == H - SPRITE_SIZE:
                continue
            if dc == -1 and c0 == 0:
                continue
            if dc == 1 and c0 == W - SPRITE_SIZE:
                continue
            candidate = {"color": color, "kind": "hollow", "pos": (r0, c0)}
            if _fits_without_touch(candidate, out, shape):
                placed = candidate
                break
        if placed is None:
            return None
        out.append(placed)
    return out


def _same_positions(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> bool:
    return all(tuple(x) == tuple(y) for x, y in zip(a, b))


def _validate_generated_case(sprites: list[dict], shape: tuple[int, int]) -> bool:
    positions_before = [tuple(sprite["pos"]) for sprite in sprites]
    positions_after, log = _simulate_chain(sprites, shape)
    if len(log) < 2:
        return False

    trigger = next(i for i, sprite in enumerate(sprites) if sprite["kind"] == "solid")
    touched = [entry["sprite"] for entry in log]
    if touched[0] != trigger:
        return False
    if any(entry["reason"] == "collision" and entry["steps"] <= 0 for entry in log):
        return False
    if sum(entry["steps"] > 0 for entry in log[1:]) < 1:
        return False

    untouched = [i for i in range(len(sprites)) if i not in touched]
    if not untouched:
        return False
    if any(positions_after[i] != positions_before[i] for i in untouched):
        return False
    if not any(not _edge_flush(shape, positions_before[i], int(sprites[i]["color"])) for i in untouched):
        return False

    final_reason = log[-1]["reason"]
    if final_reason not in {"boundary", "blocked"}:
        return False

    bg = BG_COLOR
    true_out = _render(shape, bg, sprites, positions_after)
    if np.array_equal(true_out, _slide_trigger_only(sprites, shape, bg)):
        return False
    if np.array_equal(true_out, _slide_to_edge_only(sprites, shape, bg)):
        return False
    if np.array_equal(true_out, _render(shape, bg, sprites, positions_before)):
        return False

    if not any(
        i in touched[1:] and positions_after[i] != positions_before[i]
        for i in range(len(sprites))
    ):
        return False

    for i in range(len(sprites)):
        for j in range(i + 1, len(sprites)):
            if _sprites_touch(str(sprites[i]["kind"]), positions_before[i], str(sprites[j]["kind"]), positions_before[j]):
                return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    shape = _rand_shape(rng)
    zero_step_terminal = bool(rng.randrange(2))
    chain = _make_chain(rng, shape, zero_step_terminal)
    if chain is None:
        return None
    sprites = _add_decoys(rng, chain, shape)
    if sprites is None:
        return None
    if not _validate_generated_case(sprites, shape):
        return None
    grid = _render(shape, BG_COLOR, sprites, [tuple(sprite["pos"]) for sprite in sprites])
    if not np.array_equal(main(grid), _render(shape, BG_COLOR, sprites, _simulate_chain(sprites, shape)[0])):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid relay-domino ARC input")
