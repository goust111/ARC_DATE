# Combo variant of 78332cb0 (index 12)
# combo_spec: D4/1+D13/0
# Name: 迭代无缝拼接

from __future__ import annotations

from collections import Counter
import itertools
import random

import numpy as np

from arc_utils import get_bg, is_contiguous, paste

# Design analysis
# a) Concrete behavior:
#    - Source task: extract 3-4 scrambled 5x5 blocks, infer the unique horizontal/vertical chain,
#      then render a single ordered strip with magenta separators.
#    - D4/1 removes the in-strip separators and instead overlaps neighboring blocks by exactly one
#      row/column, so the matched seam cell is shared and the strip becomes compact.
#    - D13/0 should not be left as a hidden implementation detail. If we only changed the internal
#      solve order but still emitted one final strip, the task would visually collapse back to a D4
#      variant. To keep the multi-step rule observable, the output shows every cumulative seamless
#      merge result, not just the last one.
#    - Concrete horizontal example with ordered blocks A-B-C-D and seam offsets [2,1,3]:
#      step 1: smallest seam offset is 1, so merge B+C into a 5x9 compact strip.
#      step 2: remaining seam offsets are 2 and 3; merge A with (BC) into a 5x13 strip.
#      step 3: merge (ABC)+D into the final 5x17 strip.
#      These three stage strips are stacked vertically with full magenta separator rows between
#      stages. A vertical chain does the analogous thing but arranges stage panels left-to-right.
# b) Complexity calibration:
#    - Blocks stay 5x5, connected, and sparse enough to remain human-readable.
#    - We restrict shared edge offsets to the interior set {1,2,3} and sample them without
#      replacement. Distinct seam offsets make the iterative merge order visible and deterministic.
#    - Each instance uses a single foreground color across all blocks so seam fusion reads as one
#      continuous object rather than as color overwrite noise.
# c) Hidden-assumption audit:
#    - `_find_sep`, `_split_blocks`, and edge-contact extraction still rely only on the input, so
#      the source assumptions remain valid because the input layout is unchanged.
#    - The source `_render_strip` assumption is invalid twice over: output no longer inserts
#      separator lines inside the chain, and output is no longer a single panel.
#    - A new solver assumption is introduced: every true chain seam has exactly one shared offset.
#      The generator enforces this, and the solver asserts it when computing seam priorities.
#    - Another new assumption is that seam priorities are distinct, otherwise the "which pair merges
#      next?" question becomes ambiguous. The generator guarantees distinct offsets by sampling
#      without replacement from {1,2,3}.
# d) Placement legality:
#    - Input placement is unchanged from the source task, so blocks never overlap there.
#    - Output overlap is controlled: a merge only shares one row/column seam, so compact fusion is
#      bounded and cannot spill outside the target panel.
#    - Stage-panel layout uses the axis orthogonal to the chain direction, with widths/heights sized
#      from the largest intermediate composite, so no panel clips or overlaps another.
# e) Visual observability:
#    - Across 2-4 examples a human can see both changes at once:
#      1. seams are fused instead of magenta-divided;
#      2. the object grows over multiple displayed stages according to a repeatable priority rule.
#    - Distinct seam offsets make it possible for the first merge to happen in the middle of the
#      final chain, which is visually different from simply appending blocks from one end.
# f) Visual cleanliness:
#    - Each stage panel contains only one composite object plus background, and stage panels are
#      separated by full magenta lines. This avoids clutter from showing leftover pieces together
#      with the growing composite.
# g) Color inferability:
#    - No color remapping is used. Colors remain structural only, and single-color instances keep
#      the fused seams visually clean.
# h) Fusion metric used after testing:
#    - `stage_count = n - 1` must hold for every sample.
#    - For n=4, the first merged seam should sometimes be the middle seam rather than always an
#      endpoint seam. Tracking that distribution confirms the D13 priority rule is active.
# i) Generator guarantees for new solver assumptions:
#    - exactly one valid chain orientation exists;
#    - exactly one permutation realizes that chain;
#    - each seam intersection has size 1;
#    - seam offsets are pairwise distinct, which makes iterative merge priority unique.

BG = 7
SEP = 6
PALETTE = [1, 2, 3, 4, 5, 8, 9]

BLOCK_SIZE = 5
FG_AREA_MIN = 4
FG_AREA_MAX = 10
EDGE_CONTACT_OFFSETS = [1, 2, 3]
FILLER_ROWS = [1, 3]

GRID_LAYOUT_PROB = 0.5
MAX_RETRIES = 400

OTHER = {"h": "v", "v": "h"}
EDGE_PAIRS = {"h": ("right", "left"), "v": ("bot", "top")}
ORIENT_KW = {"h": ("left", "right"), "v": ("top", "bottom")}


def _most_common(a: np.ndarray) -> int:
    return int(Counter(map(int, a.flatten())).most_common(1)[0][0])


def _find_sep(a: np.ndarray, bg: int) -> int:
    cand: list[int] = []
    for r in range(int(a.shape[0])):
        if bool((a[r, :] == int(a[r, 0])).all()) and int(a[r, 0]) != int(bg):
            cand.append(int(a[r, 0]))
    for c in range(int(a.shape[1])):
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
        if r > start:
            segs.append(a[start:r, :].copy())
        start = r + 1
    return segs


def _split_by_cols(a: np.ndarray, sep: int) -> list[np.ndarray]:
    cols = _full_sep_cols(a, sep)
    segs: list[np.ndarray] = []
    start = 0
    for c in cols + [int(a.shape[1])]:
        if c > start:
            segs.append(a[:, start:c].copy())
        start = c + 1
    return segs


def _split_blocks(g: np.ndarray, sep: int) -> list[np.ndarray]:
    """Return the input 5x5 blocks in arbitrary order."""
    rows = _full_sep_rows(g, sep)
    cols = _full_sep_cols(g, sep)
    if rows and cols:
        r = int(rows[0])
        c = int(cols[0])
        return [
            g[:r, :c].copy(),
            g[:r, c + 1 :].copy(),
            g[r + 1 :, :c].copy(),
            g[r + 1 :, c + 1 :].copy(),
        ]
    if rows:
        return _split_by_rows(g, sep)
    if cols:
        return _split_by_cols(g, sep)
    return []


def _contacts(block: np.ndarray) -> dict[str, set[int]]:
    """Foreground offsets on each edge, with per-block background inferred locally."""
    block_bg = _most_common(block)
    mask = block != int(block_bg)
    return {
        "top": set(np.where(mask[0, :])[0].tolist()),
        "bot": set(np.where(mask[-1, :])[0].tolist()),
        "left": set(np.where(mask[:, 0])[0].tolist()),
        "right": set(np.where(mask[:, -1])[0].tolist()),
    }


def _chain_permutations(blocks: list[np.ndarray], orient: str) -> list[tuple[int, ...]]:
    contacts = [_contacts(block) for block in blocks]
    lead, trail = EDGE_PAIRS[orient]

    def ok(a: int, b: int) -> bool:
        return bool(contacts[a][lead] & contacts[b][trail])

    perms: list[tuple[int, ...]] = []
    for perm in itertools.permutations(range(len(blocks))):
        if all(ok(perm[i], perm[i + 1]) for i in range(len(perm) - 1)):
            perms.append(tuple(int(x) for x in perm))
    return perms


def _shared_offset(a: np.ndarray, b: np.ndarray, orient: str) -> int:
    ca = _contacts(a)
    cb = _contacts(b)
    if orient == "h":
        inter = sorted(ca["right"] & cb["left"])
    else:
        inter = sorted(ca["bot"] & cb["top"])
    assert len(inter) == 1, "each seam must have exactly one shared offset"
    return int(inter[0])


def _merge_patches(left: np.ndarray, right: np.ndarray, orient: str, bg: int) -> np.ndarray:
    """Compactly fuse two already oriented patches with a 1-cell overlap seam."""
    h1, w1 = map(int, left.shape)
    h2, w2 = map(int, right.shape)
    if orient == "h":
        assert h1 == h2 == BLOCK_SIZE
        out = np.full((h1, w1 + w2 - 1), int(bg), dtype=int)
        paste(out, left, 0, 0, transparent=int(bg))
        paste(out, right, 0, w1 - 1, transparent=int(bg))
        return out
    assert w1 == w2 == BLOCK_SIZE
    out = np.full((h1 + h2 - 1, w1), int(bg), dtype=int)
    paste(out, left, 0, 0, transparent=int(bg))
    paste(out, right, h1 - 1, 0, transparent=int(bg))
    return out


def _iterative_stage_patches(blocks: list[np.ndarray], orient: str, bg: int) -> list[np.ndarray]:
    """Merge adjacent chain items by ascending seam offset and record each cumulative composite."""
    if len(blocks) < 2:
        raise ValueError("need at least two blocks to form a chain")

    seam_offsets = [_shared_offset(blocks[i], blocks[i + 1], orient) for i in range(len(blocks) - 1)]
    assert len(set(seam_offsets)) == len(seam_offsets), "iterative merge priorities must be distinct"

    items = [{"start": i, "end": i, "patch": blocks[i]} for i in range(len(blocks))]
    stages: list[np.ndarray] = []
    while len(items) > 1:
        best_idx = min(range(len(items) - 1), key=lambda i: seam_offsets[items[i]["end"]])
        merged_patch = _merge_patches(items[best_idx]["patch"], items[best_idx + 1]["patch"], orient, bg)
        merged_item = {
            "start": int(items[best_idx]["start"]),
            "end": int(items[best_idx + 1]["end"]),
            "patch": merged_patch,
        }
        stages.append(merged_patch)
        items = items[:best_idx] + [merged_item] + items[best_idx + 2 :]
    return stages


def _render_stage_panels(stages: list[np.ndarray], orient: str, bg: int, sep: int) -> np.ndarray:
    """Render intermediate composites as panels separated orthogonally to chain orientation."""
    if not stages:
        raise ValueError("need at least one stage")
    if orient == "h":
        max_w = max(int(stage.shape[1]) for stage in stages)
        out = np.full((len(stages) * BLOCK_SIZE + (len(stages) - 1), max_w), int(bg), dtype=int)
        rr = 0
        for i, stage in enumerate(stages):
            paste(out, stage, rr, 0, transparent=int(bg))
            rr += BLOCK_SIZE
            if i < len(stages) - 1:
                out[rr, :] = int(sep)
                rr += 1
        return out

    max_h = max(int(stage.shape[0]) for stage in stages)
    out = np.full((max_h, len(stages) * BLOCK_SIZE + (len(stages) - 1)), int(bg), dtype=int)
    cc = 0
    for i, stage in enumerate(stages):
        paste(out, stage, 0, cc, transparent=int(bg))
        cc += BLOCK_SIZE
        if i < len(stages) - 1:
            out[:, cc] = int(sep)
            cc += 1
    return out


def _solve_compact_progress(blocks: list[np.ndarray], bg: int, sep: int) -> np.ndarray:
    chain_h = _chain_permutations(blocks, "h")
    chain_v = _chain_permutations(blocks, "v")

    if len(chain_h) == 1 and len(chain_v) == 0:
        ordered = [blocks[i] for i in chain_h[0]]
        return _render_stage_panels(_iterative_stage_patches(ordered, "h", bg), "h", bg, sep)
    if len(chain_v) == 1 and len(chain_h) == 0:
        ordered = [blocks[i] for i in chain_v[0]]
        return _render_stage_panels(_iterative_stage_patches(ordered, "v", bg), "v", bg, sep)

    assert not (chain_h and chain_v), "ambiguous: both H and V chains found"
    assert chain_h or chain_v, "must find at least one valid chain"
    assert False, "ambiguous: chain order must be unique"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Iteratively fuse scrambled 5x5 sprites into a seamless composite chain and show the growth stages.

    Input structure:
    - Background is the most common color; full magenta rows/columns still separate the input blocks.
    - The input contains 3 or 4 scrambled 5x5 sprites arranged as a strip or a 2x2 layout.

    Rule:
    - Detect the unique horizontal or vertical edge-matching chain exactly as in the source task.
    - Remove in-chain separator lines: whenever two adjacent chain items are merged, their matched
      edges overlap by one row/column into a seamless composite.
    - Do not place all blocks at once. Instead, repeatedly merge the currently adjacent pair whose
      shared edge offset is smallest, treat the result as one new item, and continue until one
      composite remains.
    - Output all cumulative merge results as stage panels, separated from each other by full
      magenta lines orthogonal to the chain direction.

    Key insight:
    - Edge-contact offsets do double duty: they still reveal the chain, and once made distinct they
      also define a visible priority order for the iterative seam-fusion process.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    sep = _find_sep(g, bg)
    blocks = _split_blocks(g, sep)
    assert blocks, "must detect at least one block"
    return _solve_compact_progress(blocks, bg, sep)


def _edge_sets(block: np.ndarray) -> dict[str, set[int]]:
    mask = block != BG
    return {
        "top": set(np.where(mask[0])[0].tolist()),
        "bot": set(np.where(mask[-1])[0].tolist()),
        "left": set(np.where(mask[:, 0])[0].tolist()),
        "right": set(np.where(mask[:, -1])[0].tolist()),
    }


def _count_orders(blocks: list[np.ndarray], orient: str) -> int:
    contacts = [_edge_sets(block) for block in blocks]
    lead, trail = EDGE_PAIRS[orient]

    def ok(a: int, b: int) -> bool:
        return bool(contacts[a][lead] & contacts[b][trail])

    total = 0
    for perm in itertools.permutations(range(len(blocks))):
        if all(ok(perm[i], perm[i + 1]) for i in range(len(perm) - 1)):
            total += 1
    return total


def _interior_frontier(mask: np.ndarray) -> list[tuple[int, int]]:
    interior = mask[1:4, 1:4]
    has_neighbor = np.zeros_like(interior)
    has_neighbor[1:, :] |= interior[:-1, :]
    has_neighbor[:-1, :] |= interior[1:, :]
    has_neighbor[:, 1:] |= interior[:, :-1]
    has_neighbor[:, :-1] |= interior[:, 1:]
    has_neighbor[0, :] |= mask[0, 1:4]
    has_neighbor[2, :] |= mask[4, 1:4]
    has_neighbor[:, 0] |= mask[1:4, 0]
    has_neighbor[:, 2] |= mask[1:4, 4]
    candidates = has_neighbor & ~interior
    rs, cs = np.where(candidates)
    return [(int(r) + 1, int(c) + 1) for r, c in zip(rs, cs)]


def _base_mask(left_offset: int | None, right_offset: int | None, target_area: int) -> np.ndarray | None:
    """Build a connected 5x5 mask with optional left/right edge contacts."""
    mask = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=bool)

    if left_offset is not None:
        mask[left_offset, 0:3] = True
        mask[min(left_offset, 2) : max(left_offset, 2) + 1, 2] = True
    if right_offset is not None:
        mask[right_offset, 2:BLOCK_SIZE] = True
        mask[min(right_offset, 2) : max(right_offset, 2) + 1, 2] = True

    if mask.sum() < FG_AREA_MIN:
        mask[random.choice(FILLER_ROWS), 2] = True

    while int(mask.sum()) < int(target_area):
        frontier = _interior_frontier(mask)
        if not frontier:
            return None
        r, c = random.choice(frontier)
        mask[r, c] = True

    if left_offset is None:
        if mask[:, 0].any():
            return None
    elif np.where(mask[:, 0])[0].tolist() != [left_offset]:
        return None

    if right_offset is None:
        if mask[:, BLOCK_SIZE - 1].any():
            return None
    elif np.where(mask[:, BLOCK_SIZE - 1])[0].tolist() != [right_offset]:
        return None

    if mask[0].any() or mask[BLOCK_SIZE - 1].any():
        return None
    if any(mask[r].all() for r in range(BLOCK_SIZE)):
        return None
    if any(mask[:, c].all() for c in range(BLOCK_SIZE)):
        return None
    if not is_contiguous(mask.astype(int), background=0, connectivity=4):
        return None
    return mask


def _make_block(
    color: int,
    *,
    left: int | None = None,
    right: int | None = None,
    top: int | None = None,
    bottom: int | None = None,
) -> np.ndarray | None:
    """Generate a 5x5 single-color block with the required chain-edge contacts."""
    if top is not None or bottom is not None:
        mask = _base_mask(bottom, top, random.randint(FG_AREA_MIN, FG_AREA_MAX))
        if mask is None:
            return None
        mask = np.rot90(mask, 1)
    else:
        mask = _base_mask(left, right, random.randint(FG_AREA_MIN, FG_AREA_MAX))
        if mask is None:
            return None

    block = np.full((BLOCK_SIZE, BLOCK_SIZE), BG, dtype=int)
    block[mask] = int(color)
    return block


def _render_input_strip(blocks: list[np.ndarray], orient: str) -> np.ndarray:
    """Render the scrambled input strip with magenta separators, unchanged from the source task."""
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
        grid = np.full((2 * BLOCK_SIZE + 1, 2 * BLOCK_SIZE + 1), BG, dtype=int)
        grid[BLOCK_SIZE, :] = SEP
        grid[:, BLOCK_SIZE] = SEP
        for block, (r0, c0) in zip(
            scrambled,
            [(0, 0), (0, BLOCK_SIZE + 1), (BLOCK_SIZE + 1, 0), (BLOCK_SIZE + 1, BLOCK_SIZE + 1)],
        ):
            paste(grid, block, r0, c0)
        return grid

    return _render_input_strip(scrambled, OTHER[output_orient])


def _gen_attempt() -> np.ndarray | None:
    n = random.choice([3, 4])
    orient = random.choice(["h", "v"])
    shared_offsets = random.sample(EDGE_CONTACT_OFFSETS, n - 1)
    instance_color = random.choice(PALETTE)

    lead_kw, trail_kw = ORIENT_KW[orient]
    blocks: list[np.ndarray] = []
    for i in range(n):
        kw: dict[str, int] = {}
        if i > 0:
            kw[lead_kw] = shared_offsets[i - 1]
        if i < n - 1:
            kw[trail_kw] = shared_offsets[i]
        block = _make_block(instance_color, **kw)
        if block is None:
            return None
        blocks.append(block)

    if _count_orders(blocks, orient) != 1 or _count_orders(blocks, OTHER[orient]) != 0:
        return None

    out = _render_stage_panels(_iterative_stage_patches(blocks, orient, BG), orient, BG, SEP)
    inp = _render_input(blocks, orient)
    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
