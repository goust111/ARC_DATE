# Variant of 78332cb0
# Dimension: D10 — 端点块约束 (meta)
# Change: 链首块的前导边缘无前景像素，链尾块的尾随边缘无前景像素... → 取消端点约束

from __future__ import annotations

from collections import Counter, deque
from pathlib import Path
import itertools
import random
import sys

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import get_bg, is_contiguous, paste


# ---------------------------------------------------------------------------
# Design analysis for the D10 variant
#
# a) Concrete behavior and trace
# - The scene still consists of 3-4 scrambled 5x5 monochrome sprites separated
#   by full magenta separator lines on an orange background.
# - Output is still one ordered strip. The changed part is the data manifold:
#   endpoint blocks no longer advertise themselves by an empty outer chain edge.
# - Example in horizontal mode with 3 blocks:
#   old:  B0.left = {},  B0.right = {1},  B1.left = {1}, B1.right = {3},
#         B2.left = {3}, B2.right = {}
#   new:  B0.left = {0}, B0.right = {1}, B1.left = {1}, B1.right = {3},
#         B2.left = {3}, B2.right = {4}
#   The valid chain is still B0 -> B1 -> B2, but endpoints are now identified
#   only because {0} matches no other right edge and {4} matches no other left
#   edge. They are not identified by emptiness.
#
# b) Complexity calibration
# - To make the change visible, endpoint outer edges are always non-empty.
# - To keep the task solvable, chain-direction contacts are controlled as one
#   unique path over offsets. For n blocks we sample n+1 distinct offsets from
#   {0,1,2,3,4}; block i uses left/top offset path[i] and right/bottom offset
#   path[i+1]. This guarantees exactly one intended predecessor/successor link.
# - Perpendicular edges may also receive foreground pixels via optional noise
#   contacts. They are exact edge pixels, not color metadata, so the change is
#   visually observable.
#
# c) Hidden-assumption review against the original main()
# - _split_blocks / separator detection only assume full separator lines and
#   fixed 5x5 block size. D10 does not change that, so the logic remains valid.
# - _contacts assumes each block background is the most common color. We keep
#   4-10 fg cells in a 25-cell block, so that remains valid.
# - Original _find_chain returned the first valid permutation and relied on the
#   generator to ensure uniqueness. With relaxed endpoint constraints, extra
#   accidental chains become easier to generate, so the solver here enumerates
#   all valid permutations and requires exactly one chain in exactly one
#   orientation.
# - Nothing in the original main() explicitly required empty endpoint edges, so
#   the behavioral change is mostly a generator-manifold change plus stronger
#   uniqueness checking in the solver.
#
# d) Placement legality
# - Input layouts remain exactly the original legal forms: 2x2 grid for some
#   4-block cases, otherwise a strip, all with one-cell separator lines.
# - Blocks are still pasted into disjoint 5x5 slots, so there is no overlap or
#   out-of-bounds placement.
#
# e) Visual observability
# - Every generated sample has both chain endpoints carrying foreground on their
#   outer chain edges, which never happened in the original generator.
# - Optional perpendicular-edge pixels create additional visible variety while
#   the ordering rule remains inferable from repeated edge-offset matches.
#
# f) Visual cleanliness
# - Each block is still a single connected monochrome sprite with small area.
# - Exact non-separator full lines are rejected to avoid blocks visually
#   blending with the magenta separator bars.
#
# g) Color inference
# - Colors are unchanged: one color per block chosen from the original palette.
#   Color never encodes order; edge geometry does.
#
# h) Variant verification metric
# - The core metric is endpoint_outer_fg_count: number of chain endpoints whose
#   outer chain edge has at least one foreground pixel. Original generator fixed
#   this at 0. This variant fixes it at 2.
# - A secondary metric is perpendicular_fg_blocks: how many blocks also have
#   foreground on an edge perpendicular to the chain direction.
#
# i) New solver assumptions and generator guarantees
# - Solver assumption: exactly one orientation has exactly one full chain.
#   Generator enforces this by counting all valid permutations in both
#   orientations and rejecting everything else.
# - Solver assumption: blocks remain 5x5 and separators remain full lines.
#   Generator renders only those layouts.
# ---------------------------------------------------------------------------


BG = 7
SEP = 6
PALETTE = [1, 2, 3, 4, 5, 8, 9]

BLOCK_SIZE = 5
FG_AREA_MIN = 4
FG_AREA_MAX = 10
CHAIN_OFFSETS = [0, 1, 2, 3, 4]
PERP_OFFSETS = [1, 2, 3]

GRID_LAYOUT_PROB = 0.5
MAX_RETRIES = 800

OTHER = {"h": "v", "v": "h"}
EDGE_PAIRS = {"h": ("right", "left"), "v": ("bot", "top")}


def _most_common(a: np.ndarray) -> int:
    return int(Counter(map(int, a.flatten())).most_common(1)[0][0])


def _find_sep(a: np.ndarray, bg: int) -> int:
    H, W = a.shape
    cand: list[int] = []
    for r in range(H):
        if bool((a[r, :] == int(a[r, 0])).all()) and int(a[r, 0]) != int(bg):
            cand.append(int(a[r, 0]))
    for c in range(W):
        if bool((a[:, c] == int(a[0, c])).all()) and int(a[0, c]) != int(bg):
            cand.append(int(a[0, c]))
    return int(Counter(cand).most_common(1)[0][0])


def _full_sep_rows(a: np.ndarray, sep: int) -> list[int]:
    return [int(r) for r in range(int(a.shape[0])) if bool((a[r, :] == int(sep)).all())]


def _full_sep_cols(a: np.ndarray, sep: int) -> list[int]:
    return [int(c) for c in range(int(a.shape[1])) if bool((a[:, c] == int(sep)).all())]


def _split_by_rows(a: np.ndarray, sep: int) -> list[np.ndarray]:
    rows = _full_sep_rows(a, sep)
    segs: list[np.ndarray] = []
    start = 0
    for r in rows + [int(a.shape[0])]:
        if int(r) > int(start):
            segs.append(a[int(start) : int(r), :].copy())
        start = int(r + 1)
    return segs


def _split_by_cols(a: np.ndarray, sep: int) -> list[np.ndarray]:
    cols = _full_sep_cols(a, sep)
    segs: list[np.ndarray] = []
    start = 0
    for c in cols + [int(a.shape[1])]:
        if int(c) > int(start):
            segs.append(a[:, int(start) : int(c)].copy())
        start = int(c + 1)
    return segs


def _split_blocks(g: np.ndarray, sep: int) -> list[np.ndarray]:
    rows = _full_sep_rows(g, sep)
    cols = _full_sep_cols(g, sep)
    if rows and cols:
        r = int(rows[0])
        c = int(cols[0])
        return [
            g[:r, :c].copy(),
            g[:r, int(c + 1) :].copy(),
            g[int(r + 1) :, :c].copy(),
            g[int(r + 1) :, int(c + 1) :].copy(),
        ]
    if rows:
        return _split_by_rows(g, sep)
    if cols:
        return _split_by_cols(g, sep)
    return []


def _edge_sets(block: np.ndarray) -> dict[str, set[int]]:
    bbg = int(_most_common(block))
    mask = block != int(bbg)
    return {
        "top": set(np.where(mask[0, :])[0].tolist()),
        "bot": set(np.where(mask[-1, :])[0].tolist()),
        "left": set(np.where(mask[:, 0])[0].tolist()),
        "right": set(np.where(mask[:, -1])[0].tolist()),
    }


def _all_valid_chains(blocks: list[np.ndarray], orient: str) -> list[tuple[int, ...]]:
    contacts = [_edge_sets(block) for block in blocks]
    lead, trail = EDGE_PAIRS[orient]

    def ok(a: int, b: int) -> bool:
        return bool(contacts[a][lead] & contacts[b][trail])

    valid: list[tuple[int, ...]] = []
    for perm in itertools.permutations(range(len(blocks))):
        if all(ok(perm[i], perm[i + 1]) for i in range(len(perm) - 1)):
            valid.append(tuple(int(x) for x in perm))
    return valid


def _render_strip_axis(blocks: list[np.ndarray], axis: int, bg: int, sep: int) -> np.ndarray:
    n = len(blocks)
    if axis == 0:
        out = np.full((n * BLOCK_SIZE + (n - 1), BLOCK_SIZE), int(bg), dtype=int)
        rr = 0
        for i, block in enumerate(blocks):
            out[rr : rr + BLOCK_SIZE, :] = block
            rr += BLOCK_SIZE
            if i < n - 1:
                out[rr, :] = int(sep)
                rr += 1
        return out
    out = np.full((BLOCK_SIZE, n * BLOCK_SIZE + (n - 1)), int(bg), dtype=int)
    cc = 0
    for i, block in enumerate(blocks):
        out[:, cc : cc + BLOCK_SIZE] = block
        cc += BLOCK_SIZE
        if i < n - 1:
            out[:, cc] = int(sep)
            cc += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    sep = _find_sep(g, bg)
    blocks = _split_blocks(g, sep)
    assert blocks, "must contain separated 5x5 blocks"

    chains_h = _all_valid_chains(blocks, "h")
    chains_v = _all_valid_chains(blocks, "v")

    if len(chains_h) == 1 and not chains_v:
        ordered = [blocks[i] for i in chains_h[0]]
        return _render_strip_axis(ordered, axis=1, bg=bg, sep=sep)
    if len(chains_v) == 1 and not chains_h:
        ordered = [blocks[i] for i in chains_v[0]]
        return _render_strip_axis(ordered, axis=0, bg=bg, sep=sep)
    assert not chains_h and not chains_v or (len(chains_h) + len(chains_v) == 1), (
        "ambiguous: expected exactly one valid chain in exactly one orientation"
    )
    raise AssertionError("must find exactly one valid chain")


def _edge_cell(edge: str, offset: int) -> tuple[int, int]:
    if edge == "top":
        return 0, int(offset)
    if edge == "bot":
        return BLOCK_SIZE - 1, int(offset)
    if edge == "left":
        return int(offset), 0
    if edge == "right":
        return int(offset), BLOCK_SIZE - 1
    raise ValueError(edge)


def _all_edge_cells() -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    S = BLOCK_SIZE
    for c in range(S):
        cells.add((0, c))
        cells.add((S - 1, c))
    for r in range(S):
        cells.add((r, 0))
        cells.add((r, S - 1))
    return cells


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    out: list[tuple[int, int]] = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < BLOCK_SIZE and 0 <= cc < BLOCK_SIZE:
            out.append((rr, cc))
    return out


def _shortest_path(
    start: tuple[int, int],
    goal: tuple[int, int],
    allowed: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    if start == goal:
        return [start]
    q: deque[tuple[int, int]] = deque([start])
    prev: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    while q:
        cur = q.popleft()
        for nxt in _neighbors4(cur):
            if nxt not in allowed or nxt in prev:
                continue
            prev[nxt] = cur
            if nxt == goal:
                path = [goal]
                node = goal
                while prev[node] is not None:
                    node = prev[node]
                    path.append(node)
                path.reverse()
                return path
            q.append(nxt)
    return None


def _build_mask_from_edges(edge_specs: dict[str, set[int]]) -> np.ndarray | None:
    required = {
        _edge_cell(edge, offset)
        for edge, offsets in edge_specs.items()
        for offset in offsets
    }
    if not required:
        return None

    edge_cells = _all_edge_cells()
    allowed = {(r, c) for r in range(BLOCK_SIZE) for c in range(BLOCK_SIZE)}
    allowed -= edge_cells - required

    hubs = list(allowed)
    random.shuffle(hubs)
    best_mask: np.ndarray | None = None
    best_area = FG_AREA_MAX + 1

    for hub in hubs:
        cells = set(required)
        ok = True
        for src in required:
            path = _shortest_path(src, hub, allowed)
            if path is None:
                ok = False
                break
            cells.update(path)
        if not ok or len(cells) > FG_AREA_MAX:
            continue
        mask = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if not is_contiguous(mask.astype(int), background=0, connectivity=4):
            continue
        if any(mask[r, :].all() for r in range(BLOCK_SIZE)):
            continue
        if any(mask[:, c].all() for c in range(BLOCK_SIZE)):
            continue
        if int(mask.sum()) < best_area:
            best_mask = mask
            best_area = int(mask.sum())

    if best_mask is None:
        return None

    target_area = random.randint(max(FG_AREA_MIN, int(best_mask.sum())), FG_AREA_MAX)
    mask = best_mask.copy()
    allowed_growth = [
        (r, c)
        for r in range(BLOCK_SIZE)
        for c in range(BLOCK_SIZE)
        if (r, c) in allowed and not mask[r, c]
    ]
    while int(mask.sum()) < target_area:
        frontier = [
            (r, c)
            for (r, c) in allowed_growth
            if any(mask[nr, nc] for nr, nc in _neighbors4((r, c)))
        ]
        if not frontier:
            break
        r, c = random.choice(frontier)
        mask[r, c] = True
        allowed_growth.remove((r, c))
        if any(mask[row, :].all() for row in range(BLOCK_SIZE)):
            mask[r, c] = False
            continue
        if any(mask[:, col].all() for col in range(BLOCK_SIZE)):
            mask[r, c] = False
            continue

    if not is_contiguous(mask.astype(int), background=0, connectivity=4):
        return None
    if not (FG_AREA_MIN <= int(mask.sum()) <= FG_AREA_MAX):
        return None
    return mask


def _make_block(color: int, edge_specs: dict[str, set[int]]) -> np.ndarray | None:
    mask = _build_mask_from_edges(edge_specs)
    if mask is None:
        return None
    block = np.full((BLOCK_SIZE, BLOCK_SIZE), BG, dtype=int)
    block[mask] = int(color)
    return block


def _render_strip(blocks: list[np.ndarray], orient: str) -> np.ndarray:
    n = len(blocks)
    stride = BLOCK_SIZE + 1
    if orient == "h":
        out = np.full((BLOCK_SIZE, n * BLOCK_SIZE + (n - 1)), BG, dtype=int)
        for i, block in enumerate(blocks):
            paste(out, block, 0, i * stride)
            if i < n - 1:
                out[:, i * stride + BLOCK_SIZE] = SEP
        return out
    out = np.full((n * BLOCK_SIZE + (n - 1), BLOCK_SIZE), BG, dtype=int)
    for i, block in enumerate(blocks):
        paste(out, block, i * stride, 0)
        if i < n - 1:
            out[i * stride + BLOCK_SIZE, :] = SEP
    return out


def _render_input(blocks: list[np.ndarray], output_orient: str) -> np.ndarray:
    n = len(blocks)
    perm = list(range(n))
    while perm == list(range(n)):
        random.shuffle(perm)
    scrambled = [blocks[i] for i in perm]

    if n == 4 and random.random() < GRID_LAYOUT_PROB:
        S = BLOCK_SIZE
        grid = np.full((2 * S + 1, 2 * S + 1), BG, dtype=int)
        grid[S, :] = SEP
        grid[:, S] = SEP
        for block, (r0, c0) in zip(scrambled, [(0, 0), (0, S + 1), (S + 1, 0), (S + 1, S + 1)]):
            paste(grid, block, r0, c0)
        return grid

    return _render_strip(scrambled, OTHER[output_orient])


def _count_orders(blocks: list[np.ndarray], orient: str) -> int:
    return len(_all_valid_chains(blocks, orient))


def _random_perpendicular_specs(orient: str) -> list[tuple[str, int]]:
    specs: list[tuple[str, int]] = []
    if orient == "h":
        if random.random() < 0.55:
            specs.append(("top", random.choice(PERP_OFFSETS)))
        if random.random() < 0.55:
            specs.append(("bot", random.choice(PERP_OFFSETS)))
    else:
        if random.random() < 0.55:
            specs.append(("left", random.choice(PERP_OFFSETS)))
        if random.random() < 0.55:
            specs.append(("right", random.choice(PERP_OFFSETS)))
    return specs


def _chain_edge_specs(n: int, orient: str) -> list[dict[str, set[int]]]:
    path = random.sample(CHAIN_OFFSETS, n + 1)
    specs = []
    for i in range(n):
        edge_spec = {"top": set(), "bot": set(), "left": set(), "right": set()}
        if orient == "h":
            edge_spec["left"].add(int(path[i]))
            edge_spec["right"].add(int(path[i + 1]))
        else:
            edge_spec["top"].add(int(path[i]))
            edge_spec["bot"].add(int(path[i + 1]))
        for edge, offset in _random_perpendicular_specs(orient):
            edge_spec[edge].add(int(offset))
        specs.append(edge_spec)
    return specs


def _endpoint_outer_fg_count(blocks: list[np.ndarray], orient: str) -> int:
    contacts = [_edge_sets(block) for block in blocks]
    if orient == "h":
        return int(bool(contacts[0]["left"])) + int(bool(contacts[-1]["right"]))
    return int(bool(contacts[0]["top"])) + int(bool(contacts[-1]["bot"]))


def _perpendicular_fg_blocks(blocks: list[np.ndarray], orient: str) -> int:
    contacts = [_edge_sets(block) for block in blocks]
    if orient == "h":
        return sum(int(bool(c["top"] or c["bot"])) for c in contacts)
    return sum(int(bool(c["left"] or c["right"])) for c in contacts)


def _gen_attempt() -> np.ndarray | None:
    n = random.choice([3, 4])
    orient = random.choice(["h", "v"])
    edge_specs = _chain_edge_specs(n, orient)

    blocks: list[np.ndarray] = []
    for spec in edge_specs:
        block = _make_block(random.choice(PALETTE), spec)
        if block is None:
            return None
        blocks.append(block)

    if _endpoint_outer_fg_count(blocks, orient) != 2:
        return None
    if _count_orders(blocks, orient) != 1 or _count_orders(blocks, OTHER[orient]) != 0:
        return None

    out = _render_strip(blocks, orient)
    inp = _render_input(blocks, orient)
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
