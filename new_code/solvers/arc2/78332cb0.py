from __future__ import annotations
from collections import Counter
import numpy as np
from arc_utils import get_bg

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
    if chain_h is not None and chain_v is not None:
        ordered = [blocks[i] for i in chain_h]
        return _render_strip(ordered, axis=1)

    axis = 1 if int(g.shape[0]) > int(g.shape[1]) else 0
    return _render_strip(blocks, axis=axis)



