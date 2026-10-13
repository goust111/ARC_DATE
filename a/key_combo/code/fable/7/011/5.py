"""ARC task 5: fuse-triggered orthogonal chain reaction."""

from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
RED = 2
GREY = 5
DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return 4-connected true components, using canonical (row, column) coords."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    result: list[list[tuple[int, int]]] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        stack = [(sr, sc)]
        component: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            component.append((r, c))
            for dr, dc in DIRECTIONS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        result.append(component)
    return result


def _parse(grid: np.ndarray):
    """Extract rectangular blocks, their fuses, and the unique external spark."""
    g = np.asarray(grid)
    assert g.ndim == 2
    h, w = g.shape
    assert h <= 30 and w <= 30

    blocks = []
    owner = np.full((h, w), -1, dtype=int)
    for component in _components4(g == GREY):
        rows = [p[0] for p in component]
        cols = [p[1] for p in component]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        assert patch.shape[0] >= 2 and patch.shape[1] >= 2
        assert np.all(patch != BLACK), "a block bbox must be solid"
        fuse_cells = np.argwhere(patch != GREY)
        assert len(fuse_cells) == 1, "each block must have exactly one fuse"
        fr = r0 + int(fuse_cells[0, 0])
        fc = c0 + int(fuse_cells[0, 1])
        assert fr in (r0, r1) or fc in (c0, c1), "the fuse must lie on an edge"
        idx = len(blocks)
        assert np.all(owner[r0 : r1 + 1, c0 : c1 + 1] < 0)
        owner[r0 : r1 + 1, c0 : c1 + 1] = idx
        blocks.append({"box": (r0, c0, r1, c1), "fuse": (fr, fc),
                       "color": int(g[fr, fc])})

    assert blocks, "expected at least one block"
    outside_red = np.argwhere((g == RED) & (owner < 0))
    assert len(outside_red) == 1, "expected one isolated red spark"
    spark = tuple(map(int, outside_red[0]))
    assert all(owner[nr, nc] < 0 for dr, dc in DIRECTIONS
               if 0 <= (nr := spark[0] + dr) < h and 0 <= (nc := spark[1] + dc) < w)
    return blocks, owner, spark


def _run(grid: np.ndarray, *, propagate: bool = True, diagnostics: bool = False):
    g = np.asarray(grid, dtype=int)
    blocks, owner, spark = _parse(g)
    out = g.copy()
    lit: set[int] = set()
    queue = deque([(spark[0], spark[1], RED, 0)])
    fuse_hits: list[tuple[int, int]] = []
    solid_hits: list[tuple[int, int]] = []
    max_depth = 0

    while queue:
        sr, sc, color, depth = queue.popleft()
        max_depth = max(max_depth, depth)
        for dr, dc in DIRECTIONS:
            r, c = sr + dr, sc + dc
            while 0 <= r < g.shape[0] and 0 <= c < g.shape[1]:
                block_id = int(owner[r, c])
                if block_id >= 0:
                    block = blocks[block_id]
                    if block_id not in lit and (r, c) == block["fuse"]:
                        lit.add(block_id)
                        fuse_hits.append((depth, block_id))
                        r0, c0, r1, c1 = block["box"]
                        out[r0 : r1 + 1, c0 : c1 + 1] = block["color"]
                        if propagate:
                            queue.append((r, c, block["color"], depth + 1))
                    else:
                        solid_hits.append((depth, block_id))
                    break
                # Earlier rays and the spark stay visible when a later ray retraces them.
                if out[r, c] == BLACK:
                    out[r, c] = color
                r += dr
                c += dc

    if diagnostics:
        return out, {"blocks": blocks, "owner": owner, "lit": lit,
                     "fuse_hits": fuse_hits, "solid_hits": solid_hits,
                     "max_depth": max_depth}
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Grow colored orthogonal rays through blocks whose edge fuses they hit.

    Input structure:
    - The background is black, with one isolated red spark pixel.
    - Several separated solid grey rectangular blocks each contain exactly one
      differently colored fuse pixel on their boundary.

    Rule:
    - Every active source casts its color up, right, down, and left until the
      grid edge or the first pixel of an original block.  A ray stops before a
      block; earlier visible ray pixels are preserved if a later ray retraces them.
    - If the first block pixel is an unlit block's fuse, recolor the whole block
      with that fuse color and cast four new rays from that fuse.  Hitting any
      non-fuse block pixel does not ignite it, and a block ignites at most once.
    - Continue until the event queue contains no newly ignited blocks.

    Key insight:
    - Illumination alone is insufficient: only an exact first collision with a
      fuse moves the ray source, so the result is a fuse-hit propagation tree.
    """
    return _run(grid)


def _place_block(grid: np.ndarray, box: tuple[int, int, int, int],
                 fuse: tuple[int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    assert 0 <= r0 < r1 < grid.shape[0] and 0 <= c0 < c1 < grid.shape[1]
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == BLACK)
    grid[r0 : r1 + 1, c0 : c1 + 1] = GREY
    grid[fuse] = color


def _make_candidate() -> np.ndarray:
    """Construct a randomized, then geometrically transformed, semantic case."""
    # Three corner fuses force two changes of direction: east -> north -> west.
    a = random.randint(3, 5)
    b = random.randint(12, 15)
    east = random.randint(7, 9)
    north = random.randint(6, 8)
    west = random.randint(4, east - 2)
    size = random.choice((2, 3))
    s = (b, a)
    fa = (b, a + east)
    fb = (b - north, a + east)
    fc = (b - north, a + east - west)

    h = b + random.randint(6, 8)
    w = a + east + size + random.randint(2, 5)
    if h > 27 or w > 27 or fc[1] - size + 1 < 1 or fc[0] - 5 < 1:
        raise ValueError
    grid = np.zeros((h, w), dtype=int)
    palette = random.sample([1, 3, 4, 6, 7, 8, 9], 5)
    grid[s] = RED

    # At each fuse the block occupies the quadrant between incoming direction
    # and the side opposite the intended outgoing direction.
    _place_block(grid, (fa[0], fa[1], fa[0] + size - 1, fa[1] + size - 1), fa, palette[0])
    _place_block(grid, (fb[0] - size + 1, fb[1], fb[0], fb[1] + size - 1), fb, palette[1])
    _place_block(grid, (fc[0], fc[1] - size + 1, fc[0] + size - 1, fc[1]), fc, palette[2])

    # Two deliberate counterexamples: these blocks are struck on grey faces,
    # despite being directly illuminated, and therefore must remain unlit.
    down_top = b + random.randint(3, 4)
    _place_block(grid, (down_top, a - 1, down_top + size - 1, a + size - 2),
                 (down_top + size - 1, a - 1), palette[3])
    up_bottom = fc[0] - random.randint(3, 4)
    _place_block(grid, (up_bottom - size + 1, fc[1] - 1, up_bottom, fc[1] + size - 2),
                 (up_bottom - size + 1, fc[1] - 1), palette[4])

    # Dihedral transforms diversify absolute directions and rectangular shapes.
    if random.random() < 0.5:
        grid = np.fliplr(grid)
    if random.random() < 0.5:
        grid = np.flipud(grid)
    grid = np.rot90(grid, random.randrange(4)).copy()
    return grid


def generate_input() -> np.ndarray:
    """Generate varied non-degenerate chains and validate their stated semantics."""
    for _ in range(300):
        try:
            grid = _make_candidate()
            out, info = _run(grid, diagnostics=True)
            shallow = _run(grid, propagate=False)
        except (AssertionError, ValueError):
            continue

        # Semantic checks, rather than mere main(generate_input()) closure:
        # all three intended links ignite, both decoys remain grey, propagation
        # reaches depth three, and suppressing the chain changes the answer.
        if len(info["blocks"]) != 5 or len(info["lit"]) != 3:
            continue
        hit_depths = {depth for depth, _ in info["fuse_hits"]}
        if info["max_depth"] < 3 or hit_depths != {0, 1, 2}:
            continue
        if np.array_equal(out, shallow):
            continue
        unlit = set(range(5)) - info["lit"]
        if len(unlit) != 2:
            continue
        # Each decoy must actually be the first block met by some ray on a
        # non-fuse face; merely placing unreachable grey clutter is too weak.
        solid_block_ids = {block_id for _, block_id in info["solid_hits"]}
        if not unlit.issubset(solid_block_ids):
            continue
        if any(not np.any(out == info["blocks"][i]["color"]) for i in info["lit"]):
            continue
        if any(np.all(out[info["blocks"][i]["box"][0] : info["blocks"][i]["box"][2] + 1,
                              info["blocks"][i]["box"][1] : info["blocks"][i]["box"][3] + 1]
                      != GREY) for i in unlit):
            continue
        return grid
    raise RuntimeError("failed to generate a valid fuse-chain input")
