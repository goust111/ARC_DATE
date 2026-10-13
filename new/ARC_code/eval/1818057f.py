from __future__ import annotations
import numpy as np
from arc_utils import get_bg

def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect thin '+' shapes in a single foreground color and recolor them to teal.

    Input structure:
    - bg = most common color.
    - One foreground color (e.g. yellow) forms various shapes on the bg, some of which are small '+' crosses.

    Rule:
    - A '+' center is any foreground pixel whose 4 orthogonal neighbors are also foreground.
    - Recolor each such center and its 4 neighbors to teal (8).
    - All other pixels remain unchanged.

    Key insight: Only compact 5-cell crosses qualify — a pixel embedded in a larger filled region also satisfies the neighbor check, so the rule highlights every local '+' motif, not just isolated ones.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    colors = sorted(set(map(int, g.flatten())) - {bg})
    if not colors:
        return out
    fg = int(colors[0])

    H, W = g.shape
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if int(g[r, c]) != fg:
                continue
            if all(int(g[r + dr, c + dc]) == fg for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]):
                for dr, dc in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
                    out[r + dr, c + dc] = 8
    return out



# === v1_subagent ===

def _would_create_plus(grid, r, c, fg):
    """Check if setting (r,c) to fg would create any new plus center at (r,c) or its neighbors."""
    H, W = grid.shape
    candidates = [(r, c), (r-1, c), (r+1, c), (r, c-1), (r, c+1)]
    for cr, cc in candidates:
        if cr < 1 or cr >= H-1 or cc < 1 or cc >= W-1:
            continue
        all_fg = True
        for dr, dc in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
            nr, nc = cr + dr, cc + dc
            if nr < 0 or nr >= H or nc < 0 or nc >= W:
                all_fg = False
                break
            val = fg if (nr, nc) == (r, c) else int(grid[nr, nc])
            if val != fg:
                all_fg = False
                break
        if all_fg:
            return True
    return False


def _gen_attempt():
    rng = np.random.default_rng()
    H = rng.integers(10, 23)  # 10..22
    W = rng.integers(10, 23)
    bg, fg = 2, 4

    grid = np.full((H, W), bg, dtype=int)

    # Place K plus-motif centers with L1 >= 4 between any pair, >= 2 from border
    K = rng.integers(1, 10)  # 1..9
    centers = []
    for _ in range(K * 30):
        if len(centers) >= K:
            break
        cr = int(rng.integers(2, H - 2))
        cc = int(rng.integers(2, W - 2))
        if all(abs(cr - er) + abs(cc - ec) >= 4 for er, ec in centers):
            centers.append((cr, cc))

    if not centers:
        return None

    # Mark plus cells on grid as fg
    for cr, cc in centers:
        for dr, dc in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
            grid[cr + dr, cc + dc] = fg

    # Extend each plus with extra fg cells along random arm directions
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for cr, cc in centers:
        n_ext = rng.integers(1, 3)  # 1 or 2 directions
        chosen = rng.choice(4, size=n_ext, replace=False)
        for di in chosen:
            dr, dc = dirs[di]
            tip_r, tip_c = cr + dr, cc + dc
            ext_len = rng.integers(1, 4)  # 1..3 cells
            for step in range(1, ext_len + 1):
                nr, nc = tip_r + dr * step, tip_c + dc * step
                if nr < 0 or nr >= H or nc < 0 or nc >= W:
                    break
                grid[nr, nc] = fg

    # Sprinkle distractor fg cells targeting ~26-41% density
    total = H * W
    target_density = rng.uniform(0.26, 0.41)
    target_fg = int(total * target_density)
    current_fg = int(np.sum(grid != bg))

    bg_positions = list(zip(*np.where(grid == bg)))
    rng.shuffle(bg_positions)

    for r, c in bg_positions:
        if current_fg >= target_fg:
            break
        if not _would_create_plus(grid, r, c, fg):
            grid[r, c] = fg
            current_fg += 1

    # Ensure bg is most common color
    if int(np.sum(grid != bg)) >= total // 2:
        return None

    # Validate with solver
    inp = grid.copy()
    out = main(inp)
    if not np.any(out == 8):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")