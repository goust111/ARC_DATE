from __future__ import annotations
from collections import Counter
import numpy as np
from arc_utils import get_bg
import itertools
import random
from arc_utils import is_contiguous, paste

def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder 5×5 blocks into a connected strip by matching edge foreground pixels.

    Input structure:
    - bg = background; sep = divider color (full row/column lines).
    - Several 5×5 sprite blocks in a grid, separated by sep lines.

    Rule:
    - Two blocks "connect" when their facing edges share foreground pixels at the same offset (e.g., A's right edge ∩ B's left edge).
    - Chain all blocks in the orientation (horizontal or vertical) that admits a complete sequence.
    - Output is a 1×N strip of blocks in chain order, separated by sep lines.

    Key insight: Edge-contact overlap determines both block ordering and output orientation.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    def _most_common(a: np.ndarray) -> int:
        return int(Counter(map(int, a.flatten())).most_common(1)[0][0])

    def _find_sep(a: np.ndarray) -> int:
        H, W = a.shape
        cand: list[int] = []
        for r in range(H):
            if bool((a[int(r), :] == int(a[int(r), 0])).all()) and int(a[int(r), 0]) != int(bg):
                cand.append(int(a[int(r), 0]))
        for c in range(W):
            if bool((a[:, int(c)] == int(a[0, int(c)])).all()) and int(a[0, int(c)]) != int(bg):
                cand.append(int(a[0, int(c)]))
        return int(Counter(cand).most_common(1)[0][0])

    sep = _find_sep(g)

    def _full_sep_rows(a: np.ndarray) -> list[int]:
        return [int(r) for r in range(int(a.shape[0])) if bool((a[int(r), :] == int(sep)).all())]

    def _full_sep_cols(a: np.ndarray) -> list[int]:
        return [int(c) for c in range(int(a.shape[1])) if bool((a[:, int(c)] == int(sep)).all())]

    def _split_by_rows(a: np.ndarray) -> list[np.ndarray]:
        rows = _full_sep_rows(a)
        segs: list[np.ndarray] = []
        start = 0
        for r in rows + [int(a.shape[0])]:
            if int(r) > int(start):
                segs.append(a[int(start) : int(r), :].copy())
            start = int(r + 1)
        return segs

    def _split_by_cols(a: np.ndarray) -> list[np.ndarray]:
        cols = _full_sep_cols(a)
        segs: list[np.ndarray] = []
        start = 0
        for c in cols + [int(a.shape[1])]:
            if int(c) > int(start):
                segs.append(a[:, int(start) : int(c)].copy())
            start = int(c + 1)
        return segs

    def _render_strip(blocks: list[np.ndarray], axis: int) -> np.ndarray:
        # axis=0: vertical stack (sep rows), axis=1: horizontal stack (sep cols)
        n = int(len(blocks))
        if int(axis) == 0:
            out = np.full((int(n * 5 + (n - 1)), 5), int(bg), dtype=int)
            rr = 0
            for i, b in enumerate(blocks):
                out[int(rr) : int(rr + 5), :] = b
                rr += 5
                if int(i) < int(n - 1):
                    out[int(rr), :] = int(sep)
                    rr += 1
            return out
        out = np.full((5, int(n * 5 + (n - 1))), int(bg), dtype=int)
        cc = 0
        for i, b in enumerate(blocks):
            out[:, int(cc) : int(cc + 5)] = b
            cc += 5
            if int(i) < int(n - 1):
                out[:, int(cc)] = int(sep)
                cc += 1
        return out

    def _split_blocks() -> list[np.ndarray]:
        """Return list of 5x5 blocks (order doesn't matter; we will match by connect)."""
        rows = _full_sep_rows(g)
        cols = _full_sep_cols(g)
        if rows and cols:
            r = int(rows[0])
            c = int(cols[0])
            return [
                g[:r, :c],
                g[:r, int(c + 1) :],
                g[int(r + 1) :, :c],
                g[int(r + 1) :, int(c + 1) :],
            ]
        if rows:
            return _split_by_rows(g)
        if cols:
            return _split_by_cols(g)
        return []

    def _contacts(block: np.ndarray) -> dict[str, set[int]]:
        """Edge contact sets (offsets) for the block's foreground."""
        bbg = int(_most_common(block))
        m = block != int(bbg)
        top = set(np.where(m[0, :])[0].tolist())
        bot = set(np.where(m[-1, :])[0].tolist())
        left = set(np.where(m[:, 0])[0].tolist())
        right = set(np.where(m[:, -1])[0].tolist())
        return {"top": top, "bot": bot, "left": left, "right": right}

    def _find_chain(blocks: list[np.ndarray], orient: str):
        """Find an ordering p of indices such that consecutive blocks connect in orient."""
        import itertools

        cont = [_contacts(b) for b in blocks]

        def ok(a: int, b: int) -> bool:
            if orient == "h":
                return len(cont[int(a)]["right"] & cont[int(b)]["left"]) > 0
            return len(cont[int(a)]["bot"] & cont[int(b)]["top"]) > 0

        idx = list(range(int(len(blocks))))
        for perm in itertools.permutations(idx):
            if all(ok(perm[i], perm[i + 1]) for i in range(len(idx) - 1)):
                return perm
        return None

    blocks = _split_blocks()
    assert not (not blocks)

    chain_h = _find_chain(blocks, "h")
    chain_v = _find_chain(blocks, "v")

    if chain_h is not None and chain_v is None:
        ordered = [blocks[i] for i in chain_h]
        return _render_strip(ordered, axis=1)
    if chain_v is not None and chain_h is None:
        ordered = [blocks[i] for i in chain_v]
        return _render_strip(ordered, axis=0)
    assert not (chain_h is not None and chain_v is not None), "ambiguous: both H and V chains found"
    assert chain_h is not None or chain_v is not None, "must find at least one valid chain"

# --------------- generator (Codex attempt 2) ---------------

# -- Configuration constants --------------------------------------------------
BG = 7                          # background color
SEP = 6                         # separator color
PALETTE = [1, 2, 3, 4, 5, 8, 9]  # allowed foreground colors

BLOCK_SIZE = 5                  # sprite dimensions (square)
FG_AREA_MIN = 4                 # min foreground pixels per block
FG_AREA_MAX = 10                # max foreground pixels per block
EDGE_CONTACT_OFFSETS = [1, 2, 3]  # valid shared-edge contact rows/cols (interior only)
FILLER_ROWS = [1, 3]            # fallback interior rows for minimum-area seed

GRID_LAYOUT_PROB = 0.5          # probability of 2x2 grid layout when n=4
MAX_RETRIES = 300               # outer generate_input retry budget
# -----------------------------------------------------------------------------

_OTHER = {"h": "v", "v": "h"}
# Edge-pair keys: (leading edge of predecessor, trailing edge of successor)
_EDGE_PAIRS = {"h": ("right", "left"), "v": ("bot", "top")}
# _make_block keyword mapping per orientation:
#   orient -> (leading_kw, trailing_kw)
_ORIENT_KW = {"h": ("left", "right"), "v": ("top", "bottom")}


def _edge_sets(block: np.ndarray) -> dict[str, set[int]]:
    """Return sets of fg-pixel offsets on each edge of a 5x5 block."""
    mask = block != BG
    return {
        "top": set(np.where(mask[0])[0].tolist()),
        "bot": set(np.where(mask[-1])[0].tolist()),
        "left": set(np.where(mask[:, 0])[0].tolist()),
        "right": set(np.where(mask[:, -1])[0].tolist()),
    }


def _count_orders(blocks: list[np.ndarray], orient: str) -> int:
    """Count how many permutations form a valid chain in the given orientation."""
    contacts = [_edge_sets(block) for block in blocks]
    lead, trail = _EDGE_PAIRS[orient]

    def ok(a: int, b: int) -> bool:
        return bool(contacts[a][lead] & contacts[b][trail])

    total = 0
    for perm in itertools.permutations(range(len(blocks))):
        if all(ok(perm[i], perm[i + 1]) for i in range(len(perm) - 1)):
            total += 1
    return total


def _interior_frontier(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return interior cells (rows/cols 1-3) adjacent to the mask but not yet set."""
    interior = mask[1:4, 1:4]
    # 4-neighbor dilation within interior
    has_neighbor = np.zeros_like(interior)
    has_neighbor[1:, :] |= interior[:-1, :]
    has_neighbor[:-1, :] |= interior[1:, :]
    has_neighbor[:, 1:] |= interior[:, :-1]
    has_neighbor[:, :-1] |= interior[:, 1:]
    # also check neighbors from border rows/cols of the full mask
    has_neighbor[0, :] |= mask[0, 1:4]   # row above interior
    has_neighbor[2, :] |= mask[4, 1:4]   # row below interior
    has_neighbor[:, 0] |= mask[1:4, 0]   # col left of interior
    has_neighbor[:, 2] |= mask[1:4, 4]   # col right of interior
    candidates = has_neighbor & ~interior
    rs, cs = np.where(candidates)
    return [(int(r) + 1, int(c) + 1) for r, c in zip(rs, cs)]


def _base_mask(left_offset: int | None, right_offset: int | None,
               target_area: int) -> np.ndarray | None:
    """Build a connected fg mask on a 5x5 grid with prescribed left/right edge contacts.

    The mask is seeded from the edge-contact offsets, then grown randomly through
    interior cells until target_area is reached.  Returns None on constraint violation.
    """
    S = BLOCK_SIZE
    mask = np.zeros((S, S), dtype=bool)

    # Seed from edge contacts -- horizontal arm + vertical spine to center col
    if left_offset is not None:
        mask[left_offset, 0:3] = True
        mask[min(left_offset, 2):max(left_offset, 2) + 1, 2] = True
    if right_offset is not None:
        mask[right_offset, 2:S] = True
        mask[min(right_offset, 2):max(right_offset, 2) + 1, 2] = True

    if mask.sum() < FG_AREA_MIN:
        mask[random.choice(FILLER_ROWS), 2] = True

    # Grow to target area via random frontier expansion (interior cells only)
    while mask.sum() < target_area:
        frontier = _interior_frontier(mask)
        if not frontier:
            return None
        r, c = random.choice(frontier)
        mask[r, c] = True

    # --- Validate edge constraints ---
    if left_offset is None:
        if mask[:, 0].any():
            return None
    elif np.where(mask[:, 0])[0].tolist() != [left_offset]:
        return None

    if right_offset is None:
        if mask[:, S - 1].any():
            return None
    elif np.where(mask[:, S - 1])[0].tolist() != [right_offset]:
        return None

    if mask[0].any() or mask[S - 1].any():
        return None
    if any(mask[r].all() for r in range(S)):
        return None
    if any(mask[:, c].all() for c in range(S)):
        return None
    if not is_contiguous(mask.astype(int), background=0, connectivity=4):
        return None
    return mask


def _make_block(color: int, *, left: int | None = None, right: int | None = None,
                top: int | None = None, bottom: int | None = None) -> np.ndarray | None:
    """Generate a single 5x5 block with the given color and edge-contact constraints."""
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
    block[mask] = color
    return block


def _render_strip(blocks: list[np.ndarray], orient: str) -> np.ndarray:
    """Render blocks into a 1xN strip with SEP dividers."""
    n = len(blocks)
    S = BLOCK_SIZE
    stride = S + 1  # block size + 1 separator column/row
    if orient == "h":
        out = np.full((S, n * S + (n - 1)), BG, dtype=int)
        for i, block in enumerate(blocks):
            paste(out, block, 0, i * stride)
            if i < n - 1:
                out[:, i * stride + S] = SEP
    else:
        out = np.full((n * S + (n - 1), S), BG, dtype=int)
        for i, block in enumerate(blocks):
            paste(out, block, i * stride, 0)
            if i < n - 1:
                out[i * stride + S, :] = SEP
    return out


def _render_input(blocks: list[np.ndarray], output_orient: str) -> np.ndarray:
    """Scramble blocks and arrange into an input layout (2x2 grid or strip)."""
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

    return _render_strip(scrambled, _OTHER[output_orient])


def _gen_attempt() -> np.ndarray | None:
    """Try once to generate a valid input grid. Returns None on failure."""
    n = random.choice([3, 4])
    orient = random.choice(["h", "v"])
    shared_offsets = random.sample(EDGE_CONTACT_OFFSETS, n - 1)

    lead_kw, trail_kw = _ORIENT_KW[orient]
    blocks: list[np.ndarray] = []
    for i in range(n):
        kw = {}
        if i > 0:
            kw[lead_kw] = shared_offsets[i - 1]
        if i < n - 1:
            kw[trail_kw] = shared_offsets[i]
        block = _make_block(random.choice(PALETTE), **kw)
        if block is None:
            return None
        blocks.append(block)

    if _count_orders(blocks, orient) != 1 or _count_orders(blocks, _OTHER[orient]) != 0:
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
