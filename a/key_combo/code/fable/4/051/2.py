from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
ORANGE = 7

BLOCK_SIZE = 3

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}
SIDE_TO_DIR = {"top": "down", "bottom": "up", "left": "right", "right": "left"}
MARKER_POS = {"up": (0, 1), "down": (2, 1), "left": (1, 0), "right": (1, 2)}
MARKER_DIR = {v: k for k, v in MARKER_POS.items()}


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _touch_side(pos: tuple[int, int], shape: tuple[int, int]) -> str | None:
    r, c = pos
    h, w = shape
    sides = []
    if r == 0:
        sides.append("top")
    if r + BLOCK_SIZE - 1 == h - 1:
        sides.append("bottom")
    if c == 0:
        sides.append("left")
    if c + BLOCK_SIZE - 1 == w - 1:
        sides.append("right")
    if len(sides) == 1:
        return sides[0]
    return None


def _block_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    ar, ac = a
    br, bc = b
    return not (
        ar + BLOCK_SIZE - 1 < br
        or br + BLOCK_SIZE - 1 < ar
        or ac + BLOCK_SIZE - 1 < bc
        or bc + BLOCK_SIZE - 1 < ac
    )


def _block_touch_or_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    ar, ac = a
    br, bc = b
    return not (
        ar + BLOCK_SIZE < br
        or br + BLOCK_SIZE < ar
        or ac + BLOCK_SIZE < bc
        or bc + BLOCK_SIZE < ac
    )


def _in_bounds(pos: tuple[int, int], shape: tuple[int, int]) -> bool:
    r, c = pos
    h, w = shape
    return 0 <= r <= h - BLOCK_SIZE and 0 <= c <= w - BLOCK_SIZE


def _paint_stripe(out: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> None:
    r0, c0 = start
    r1, c1 = end
    if r0 == r1:
        out[r0 : r0 + BLOCK_SIZE, min(c0, c1) : max(c0, c1) + BLOCK_SIZE] = ORANGE
    else:
        out[min(r0, r1) : max(r0, r1) + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE] = ORANGE


def _extract_blocks(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    comps = _components(g != BLACK)
    assert comps, "expected at least one foreground block"

    red_block = None
    blue_blocks: list[dict] = []
    for coords in comps:
        assert coords.shape[0] == BLOCK_SIZE * BLOCK_SIZE, "each block must occupy a 3x3 box"
        r0, c0, r1, c1 = _bbox(coords)
        assert (r1 - r0 + 1, c1 - c0 + 1) == (BLOCK_SIZE, BLOCK_SIZE), "each block must be 3x3"
        sprite = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        colors = set(map(int, np.unique(sprite)))
        pos = (r0, c0)

        if colors == {RED}:
            side = _touch_side(pos, (h, w))
            assert side is not None, "red block must touch exactly one edge"
            assert red_block is None, "expected exactly one red block"
            red_block = {
                "id": "R",
                "kind": "red",
                "pos": pos,
                "sprite": sprite,
                "dir": SIDE_TO_DIR[side],
            }
            continue

        assert colors == {BLUE, YELLOW}, "blue relay blocks must contain only blue and yellow"
        assert int((sprite == YELLOW).sum()) == 1, "each blue block must have one yellow marker"
        assert int((sprite == BLUE).sum()) == 8, "blue relay block must have eight blue cells"
        yellow = tuple(map(int, np.argwhere(sprite == YELLOW)[0]))
        assert yellow in MARKER_DIR, "yellow marker must be adjacent to the center"
        blue_blocks.append(
            {
                "id": f"B{len(blue_blocks)}",
                "kind": "blue",
                "pos": pos,
                "sprite": sprite,
                "dir": MARKER_DIR[yellow],
            }
        )

    assert red_block is not None, "missing red starter block"
    return [red_block] + blue_blocks


def _slide_once(
    block_id: str,
    positions: dict[str, tuple[int, int]],
    direction: str,
    shape: tuple[int, int],
) -> tuple[tuple[int, int], str | None]:
    dr, dc = DIRS[direction]
    r, c = positions[block_id]
    while True:
        nxt = (r + dr, c + dc)
        if not _in_bounds(nxt, shape):
            return (r, c), None
        hits = [other_id for other_id, other_pos in positions.items() if other_id != block_id and _block_overlap(nxt, other_pos)]
        if hits:
            assert len(hits) == 1, "a relay move must collide with at most one block"
            return (r, c), hits[0]
        r, c = nxt


def _simulate(blocks: list[dict], shape: tuple[int, int]) -> tuple[dict[str, tuple[int, int]], list[dict]]:
    positions = {block["id"]: tuple(block["pos"]) for block in blocks}
    kind = {block["id"]: block["kind"] for block in blocks}
    direction = {block["id"]: block["dir"] for block in blocks}
    active = "R"
    triggered: set[str] = set()
    segments: list[dict] = []

    while True:
        start = positions[active]
        end, hit = _slide_once(active, positions, direction[active], shape)
        positions[active] = end
        segments.append({"id": active, "start": start, "end": end, "hit": hit})
        if hit is None or kind.get(hit) != "blue" or hit in triggered:
            break
        triggered.add(hit)
        active = hit

    return positions, segments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide the edge red block inward, then relay each hit blue marker block along its own indicated direction while painting orange trails.

    Input structure:
    - The background is black.
    - Exactly one solid red 3x3 block touches exactly one outer edge.
    - Inside the canvas are separated blue 3x3 blocks, each with exactly one yellow marker on one of the four cells adjacent to its center.

    Rule:
    - The red block slides straight inward, perpendicular to its touched edge, and stops immediately before it would overlap the first block it hits.
    - Each hit blue block then slides from its own current position in the direction indicated by its yellow marker, stopping immediately before overlapping the next block or when it reaches the canvas edge.
    - Every moved block paints its swept 3-cell-wide stripe orange, and each block is finally redrawn at its stopping position with its original internal colors.

    Key insight:
    - Motion is a sequential collision relay: every segment direction is re-read from the newly hit block's internal yellow offset rather than inherited from the previous segment.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"

    blocks = _extract_blocks(g)
    positions, segments = _simulate(blocks, g.shape)

    out = g.copy()
    for seg in segments:
        _paint_stripe(out, seg["start"], seg["end"])

    for block in blocks:
        r, c = positions[block["id"]]
        out[r : r + BLOCK_SIZE, c : c + BLOCK_SIZE] = block["sprite"]

    return out


def _blue_sprite(direction: str) -> np.ndarray:
    sprite = np.full((BLOCK_SIZE, BLOCK_SIZE), BLUE, dtype=int)
    mr, mc = MARKER_POS[direction]
    sprite[mr, mc] = YELLOW
    return sprite


def _red_sprite() -> np.ndarray:
    return np.full((BLOCK_SIZE, BLOCK_SIZE), RED, dtype=int)


def _stripe_mask(start: tuple[int, int], end: tuple[int, int], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0 = start
    r1, c1 = end
    if r0 == r1:
        mask[r0 : r0 + BLOCK_SIZE, min(c0, c1) : max(c0, c1) + BLOCK_SIZE] = True
    else:
        mask[min(r0, r1) : max(r0, r1) + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE] = True
    return mask


def _render_input(specs: list[dict], shape: tuple[int, int]) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for spec in specs:
        r, c = spec["pos"]
        grid[r : r + BLOCK_SIZE, c : c + BLOCK_SIZE] = spec["sprite"]
    return grid


def _valid_spacing(specs: list[dict]) -> bool:
    positions = [tuple(spec["pos"]) for spec in specs]
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if _block_touch_or_overlap(positions[i], positions[j]):
                return False
    return True


def _choose_blue_dirs(count: int, red_dir: str) -> list[str]:
    for _ in range(100):
        dirs: list[str] = []
        prev = red_dir
        for _idx in range(count):
            choices = [name for name in DIRS if name != OPPOSITE[prev]]
            if random.random() < 0.7:
                turn_choices = [name for name in choices if name != prev]
                chosen = random.choice(turn_choices or choices)
            else:
                chosen = random.choice(choices)
            dirs.append(chosen)
            prev = chosen
        if dirs[0] != red_dir or any(dirs[i] != dirs[i - 1] for i in range(1, len(dirs))):
            return dirs
    raise RuntimeError("failed to sample a turning relay chain")


def _last_position_for_edge(direction: str, steps: int, shape: tuple[int, int]) -> tuple[int, int] | None:
    h, w = shape
    if direction == "up":
        if steps > h - BLOCK_SIZE - 1:
            return None
        return steps, random.randint(1, w - BLOCK_SIZE - 1)
    if direction == "down":
        if steps > h - BLOCK_SIZE - 1:
            return None
        return h - BLOCK_SIZE - steps, random.randint(1, w - BLOCK_SIZE - 1)
    if direction == "left":
        if steps > w - BLOCK_SIZE - 1:
            return None
        return random.randint(1, h - BLOCK_SIZE - 1), steps
    if steps > w - BLOCK_SIZE - 1:
        return None
    return random.randint(1, h - BLOCK_SIZE - 1), w - BLOCK_SIZE - steps


def _previous_position(next_pos: tuple[int, int], direction: str, steps: int) -> tuple[int, int]:
    nr, nc = next_pos
    if direction == "up":
        return nr + BLOCK_SIZE + steps, nc
    if direction == "down":
        return nr - BLOCK_SIZE - steps, nc
    if direction == "left":
        return nr, nc + BLOCK_SIZE + steps
    return nr, nc - BLOCK_SIZE - steps


def _red_position(first_blue: tuple[int, int], side: str, shape: tuple[int, int]) -> tuple[int, int] | None:
    r, c = first_blue
    h, w = shape
    if side == "top":
        return (0, c) if r >= 5 else None
    if side == "bottom":
        return (h - BLOCK_SIZE, c) if r <= h - 8 else None
    if side == "left":
        return (r, 0) if c >= 5 else None
    return (r, w - BLOCK_SIZE) if c <= w - 8 else None


def _build_attempt() -> np.ndarray | None:
    shape = (random.randint(18, 30), random.randint(18, 30))
    blue_count = random.randint(2, 5)
    side = random.choice(list(SIDE_TO_DIR))
    red_dir = SIDE_TO_DIR[side]
    blue_dirs = _choose_blue_dirs(blue_count, red_dir)
    step_counts = [random.randint(2, 6) for _ in range(blue_count)]

    last_pos = _last_position_for_edge(blue_dirs[-1], step_counts[-1], shape)
    if last_pos is None:
        return None

    blue_positions = [None] * blue_count
    blue_positions[-1] = last_pos
    for idx in range(blue_count - 2, -1, -1):
        pos = _previous_position(blue_positions[idx + 1], blue_dirs[idx], step_counts[idx])
        if not _in_bounds(pos, shape):
            return None
        side_touch = _touch_side(pos, shape)
        if side_touch is not None:
            return None
        blue_positions[idx] = pos

    red_pos = _red_position(blue_positions[0], side, shape)
    if red_pos is None:
        return None

    specs = [
        {"id": "R", "kind": "red", "pos": red_pos, "sprite": _red_sprite(), "dir": red_dir},
    ]
    for idx, (pos, direction) in enumerate(zip(blue_positions, blue_dirs)):
        specs.append(
            {
                "id": f"B{idx}",
                "kind": "blue",
                "pos": pos,
                "sprite": _blue_sprite(direction),
                "dir": direction,
            }
        )

    if not _valid_spacing(specs):
        return None

    final_positions, segments = _simulate(specs, shape)
    moved_ids = [seg["id"] for seg in segments]
    expected_ids = ["R"] + [f"B{i}" for i in range(blue_count)]
    if moved_ids != expected_ids:
        return None
    if segments[-1]["hit"] is not None:
        return None

    stripe_masks = [_stripe_mask(seg["start"], seg["end"], shape) for seg in segments]
    for i in range(len(stripe_masks)):
        for j in range(i + 1, len(stripe_masks)):
            if np.any(stripe_masks[i] & stripe_masks[j]):
                return None

    finals = [final_positions[spec["id"]] for spec in specs]
    for i in range(len(finals)):
        for j in range(i + 1, len(finals)):
            if _block_overlap(finals[i], finals[j]):
                return None

    grid = _render_input(specs, shape)
    out = main(grid)
    if np.array_equal(grid, out):
        return None
    if int((out == ORANGE).sum()) < 12:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid relay-routing input")
