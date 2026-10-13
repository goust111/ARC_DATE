import numpy as np


BLACK = 0
YELLOW = 4


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each colored frame's yellow interior pattern as a stamp and print it at every same-colored external marker.

    Input structure:
    - Black background with several colored hollow rectangular frames.
    - Inside each frame sits a yellow stamp pattern.
    - Outside the frames, singleton markers share colors with the frames.

    Rule:
    - For every frame, extract the yellow pattern relative to the frame's inner top-left cell.
    - Find all same-colored external markers for that frame.
    - Clear the original frame and its interior, then stamp the yellow pattern at every marker using the marker as the frame's inner top-left anchor.

    Key insight:
    - The colored frame is only a template container; the real output comes from linking its border color to distant marker positions.
    """
    out = grid.copy()
    frames = []
    frame_boxes = []
    for color in np.unique(grid):
        if color in (BLACK, YELLOW):
            continue
        for comp in _components(grid == color):
            rs = [p[0] for p in comp]
            cs = [p[1] for p in comp]
            r0, r1 = min(rs), max(rs)
            c0, c1 = min(cs), max(cs)
            if r1 - r0 >= 3 and c1 - c0 >= 3 and len(comp) == 2 * (r1 - r0 + 1) + 2 * (c1 - c0 + 1) - 4:
                frames.append((int(color), (r0, r1, c0, c1)))
                frame_boxes.append((r0, r1, c0, c1))
    for color, (r0, r1, c0, c1) in frames:
        pattern = np.argwhere(grid[r0 + 1 : r1, c0 + 1 : c1] == YELLOW)
        out[r0 : r1 + 1, c0 : c1 + 1] = BLACK
        markers = []
        for rr, cc in np.argwhere(grid == color):
            if not (r0 <= rr <= r1 and c0 <= cc <= c1):
                if all(not (a <= rr <= b and d <= cc <= e) for a, b, d, e in frame_boxes):
                    markers.append((rr, cc))
        for rr, cc in markers:
            for pr, pc in pattern:
                tr, tc = rr + int(pr), cc + int(pc)
                if 0 <= tr < out.shape[0] and 0 <= tc < out.shape[1]:
                    out[tr, tc] = YELLOW
    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    grid = np.zeros((24, 24), dtype=int)
    colors = list(rng.choice([1, 2, 3, 6, 7, 8, 9], size=int(rng.integers(2, 5)), replace=False))
    used = np.zeros_like(grid, dtype=bool)
    for color in colors:
        for _attempt in range(200):
            h = int(rng.integers(4, 7))
            w = int(rng.integers(4, 7))
            r0 = int(rng.integers(1, 12))
            c0 = int(rng.integers(1, 12))
            if used[r0 - 1 : r0 + h + 1, c0 - 1 : c0 + w + 1].any():
                continue
            grid[r0 : r0 + h, c0 : c0 + w] = int(color)
            grid[r0 + 1 : r0 + h - 1, c0 + 1 : c0 + w - 1] = BLACK
            pat = rng.random((h - 2, w - 2)) < 0.45
            pat[max(0, (h - 2) // 2), max(0, (w - 2) // 2)] = True
            grid[r0 + 1 : r0 + h - 1, c0 + 1 : c0 + w - 1][pat] = YELLOW
            used[r0 - 1 : r0 + h + 1, c0 - 1 : c0 + w + 1] = True
            marker_count = int(rng.integers(1, 4))
            for _ in range(marker_count):
                for _m in range(200):
                    rr = int(rng.integers(0, 24))
                    cc = int(rng.integers(0, 24))
                    if grid[rr, cc] == BLACK:
                        grid[rr, cc] = int(color)
                        break
            break
    return grid
