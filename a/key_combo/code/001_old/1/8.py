from __future__ import annotations

import random

import numpy as np


def _edge_segment(g: np.ndarray) -> tuple[str, int, int]:
    h, w = g.shape
    pts = [(int(r), int(c)) for r, c in np.argwhere(g == 5)]
    if not pts:
        raise ValueError("missing grey edge segment")
    rows = {r for r, _c in pts}
    cols = {c for _r, c in pts}
    if len(cols) == 1 and next(iter(cols)) in (0, w - 1):
        col = next(iter(cols))
        sorted_rows = sorted(rows)
        if sorted_rows != list(range(sorted_rows[0], sorted_rows[-1] + 1)):
            raise ValueError("grey edge segment must be contiguous")
        return ("vertical", sorted_rows[0], len(sorted_rows))
    if len(rows) == 1 and next(iter(rows)) in (0, h - 1):
        row = next(iter(rows))
        sorted_cols = sorted(cols)
        if sorted_cols != list(range(sorted_cols[0], sorted_cols[-1] + 1)):
            raise ValueError("grey edge segment must be contiguous")
        return ("horizontal", sorted_cols[0], len(sorted_cols))
    raise ValueError("grey segment is not attached to one edge")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode an edge grey segment as axis, phase, and period, then draw full lines through surviving pixels.

    Input structure:
    - A black grid has one contiguous grey segment attached to a single edge.
    - Interior cells contain sparse colored singleton pixels.
    - No two colored pixels share a row or column.

    Rule:
    - A left or right grey segment filters pixels by row and draws full rows; a top or bottom segment filters by column and draws full columns.
    - The grey segment length is the period, and its topmost or leftmost coordinate is the phase.
    - Keep only pixels whose selected coordinate is congruent to the phase modulo the period, drawing a full line in that pixel's color.

    Key insight:
    One edge object encodes three parameters at once: drawing axis, modular phase, and modular period.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    kind, phase, period = _edge_segment(g)
    out = np.zeros_like(g)
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color in (0, 5):
                continue
            if kind == "vertical":
                if r % period == phase % period:
                    out[r, :] = color
            else:
                if c % period == phase % period:
                    out[:, c] = color
    return out


def generate_input() -> np.ndarray:
    for _attempt in range(200):
        h = random.randint(9, 13)
        w = random.randint(9, 13)
        g = np.zeros((h, w), dtype=int)
        side = random.choice(["left", "right", "top", "bottom"])
        length = random.randint(2, 4)
        if side in ("left", "right"):
            t = random.randint(1, h - length - 2)
            c = 0 if side == "left" else w - 1
            g[t : t + length, c] = 5
            good_rows = [r for r in range(1, h - 1) if r % length == t % length]
            bad_rows = [r for r in range(1, h - 1) if r % length != t % length]
            if len(good_rows) < 2 or len(bad_rows) < 2:
                continue
            rows = random.sample(good_rows, 2) + random.sample(bad_rows, 2)
            extra_pool = [r for r in range(1, h - 1) if r not in rows]
            max_extra = min(4, len(extra_pool), (w - 2) - len(rows))
            rows += random.sample(extra_pool, random.randint(0, max(0, max_extra)))
            cols = random.sample(range(1, w - 1), len(rows))
            random.shuffle(rows)
            for r, col in zip(rows, cols):
                g[r, col] = random.choice([x for x in range(1, 10) if x != 5])
        else:
            t = random.randint(1, w - length - 2)
            r = 0 if side == "top" else h - 1
            g[r, t : t + length] = 5
            good_cols = [c for c in range(1, w - 1) if c % length == t % length]
            bad_cols = [c for c in range(1, w - 1) if c % length != t % length]
            if len(good_cols) < 2 or len(bad_cols) < 2:
                continue
            cols = random.sample(good_cols, 2) + random.sample(bad_cols, 2)
            extra_pool = [c for c in range(1, w - 1) if c not in cols]
            max_extra = min(4, len(extra_pool), (h - 2) - len(cols))
            cols += random.sample(extra_pool, random.randint(0, max(0, max_extra)))
            rows = random.sample(range(1, h - 1), len(cols))
            random.shuffle(cols)
            for row, c in zip(rows, cols):
                g[row, c] = random.choice([x for x in range(1, 10) if x != 5])
        return g
    raise RuntimeError("failed to generate puzzle input")
