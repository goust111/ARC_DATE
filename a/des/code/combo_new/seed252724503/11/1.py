import numpy as np


BLACK = 0
SPRITE = 8


def _components(mask):
    rows, cols = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(rows):
        for c in range(cols):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _find_palette(grid):
    blocks = []
    for color in np.unique(grid):
        if color == BLACK:
            continue
        mask = grid == color
        for comp in _components(mask):
            rs = [p[0] for p in comp]
            cs = [p[1] for p in comp]
            if len(comp) == 4 and max(rs) - min(rs) == 1 and max(cs) - min(cs) == 1:
                if min(rs) == 0 or max(rs) == grid.shape[0] - 1 or min(cs) == 0 or max(cs) == grid.shape[1] - 1:
                    blocks.append((min(cs), color))
    blocks.sort()
    return [color for _, color in blocks]


def _find_frames(grid, palette_colors):
    frames = []
    for color in np.unique(grid):
        if color in (BLACK, SPRITE) or color in palette_colors:
            continue
        for comp in _components(grid == color):
            rs = [p[0] for p in comp]
            cs = [p[1] for p in comp]
            r0, r1 = min(rs), max(rs)
            c0, c1 = min(cs), max(cs)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h == w and h >= 5 and len(comp) == 2 * h + 2 * w - 4:
                frames.append((r0, r1, c0, c1))
    frames.sort()
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the shared sprite inside each colored frame and fill the missing half with a palette-coded color.

    Input structure:
    - Black background with several same-sized hollow square frames in different colors.
    - Each frame contains the same sprite color, but one directional half of that sprite is missing.
    - Four monochrome 2x2 palette blocks sit on the grid edge from left to right.

    Rule:
    - Read the palette blocks left-to-right as the colors for missing left, right, top, and bottom halves.
    - Reconstruct the complete sprite by taking the union of all visible sprite pixels across frames.
    - For each frame, compare its visible sprite to the reconstructed sprite, detect which side is missing, and fill only the missing sprite cells with the corresponding palette color.

    Key insight:
    - The framed sprites are not different templates: they are partial views of one shared template, so solving the task requires union-based reconstruction before directional recoloring.
    """
    out = grid.copy()
    palette = _find_palette(grid)
    frames = _find_frames(grid, palette)
    interiors = []
    for r0, r1, c0, c1 in frames:
        inner = grid[r0 + 1 : r1, c0 + 1 : c1] == SPRITE
        interiors.append(inner)
    full_mask = np.zeros_like(interiors[0], dtype=bool)
    for inner in interiors:
        full_mask |= inner
    h, w = full_mask.shape
    center_r = (h - 1) / 2.0
    center_c = (w - 1) / 2.0
    for (r0, r1, c0, c1), inner in zip(frames, interiors):
        missing = full_mask & ~inner
        if not missing.any():
            continue
        coords = np.argwhere(missing)
        mean_r, mean_c = coords.mean(axis=0)
        if abs(mean_c - center_c) >= abs(mean_r - center_r):
            palette_idx = 0 if mean_c < center_c else 1
        else:
            palette_idx = 2 if mean_r < center_r else 3
        fill_color = palette[palette_idx]
        sub = out[r0 + 1 : r1, c0 + 1 : c1]
        sub[missing] = fill_color
    return out


def _random_mask(rng, size):
    while True:
        mask = rng.random((size, size)) < 0.45
        mask[size // 2, size // 2] = True
        if mask.sum() < size + 3:
            continue
        if not mask[:, : size // 2].any() or not mask[:, size // 2 :].any():
            continue
        if not mask[: size // 2, :].any() or not mask[size // 2 :, :].any():
            continue
        return mask


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    size = int(rng.integers(6, 8))
    inner = size - 2
    sprite = _random_mask(rng, inner)
    palette_colors = list(rng.choice([1, 2, 3, 4, 6, 7, 9], size=4, replace=False))
    frame_pool = [color for color in [1, 2, 3, 4, 5, 6, 7, 9] if color not in palette_colors]
    frame_colors = list(rng.choice(frame_pool, size=4, replace=False))
    height = 2 + 2 * size + 4
    width = max(2 * size + 4, 14)
    grid = np.zeros((height, width), dtype=int)

    # Palette on the top edge.
    start_c = int(rng.integers(0, max(1, width - 11)))
    for i, color in enumerate(palette_colors):
        c = start_c + i * 3
        grid[0:2, c : c + 2] = color

    # Four frames, one for each missing side.
    positions = [(3, 1), (3, size + 2), (size + 2, 1), (size + 2, size + 2)]
    halves = [
        lambda m: np.column_stack(np.where(m & (np.indices(m.shape)[1] < inner / 2))),
        lambda m: np.column_stack(np.where(m & (np.indices(m.shape)[1] >= inner / 2))),
        lambda m: np.column_stack(np.where(m & (np.indices(m.shape)[0] < inner / 2))),
        lambda m: np.column_stack(np.where(m & (np.indices(m.shape)[0] >= inner / 2))),
    ]
    for idx, ((r0, c0), frame_color) in enumerate(zip(positions, frame_colors)):
        grid[r0 : r0 + size, c0 : c0 + size] = frame_color
        grid[r0 + 1 : r0 + size - 1, c0 + 1 : c0 + size - 1] = 0
        sub = grid[r0 + 1 : r0 + size - 1, c0 + 1 : c0 + size - 1]
        visible = sprite.copy()
        if idx == 0:
            visible[:, : inner // 2] = False
        elif idx == 1:
            visible[:, inner // 2 :] = False
        elif idx == 2:
            visible[: inner // 2, :] = False
        else:
            visible[inner // 2 :, :] = False
        sub[visible] = SPRITE
    return grid
