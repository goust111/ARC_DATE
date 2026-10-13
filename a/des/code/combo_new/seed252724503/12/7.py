import numpy as np


BLACK = 0
GREY = 5


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


def _bbox_center(comp):
    rs = [p[0] for p in comp]
    cs = [p[1] for p in comp]
    return ((min(rs) + max(rs)) // 2, (min(cs) + max(cs)) // 2)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stack each color's fragments onto its matching anchor, using the fragment's original direction from the anchor to choose overlay priority.

    Input structure:
    - Each anchor is a 3x3 grey frame with one colored center pixel.
    - Separate same-colored fragments lie elsewhere on the black background.
    - Every color has exactly one anchor and several fragments.

    Rule:
    - Match fragments to anchors by color.
    - Classify each fragment relative to its anchor as left, up, right, or down using the bbox-center displacement.
    - Translate every fragment so its bbox center lands on the anchor center, then overlay the translated fragments in the fixed order left, up, right, down.
    - Remove the original fragments and all grey anchor frames.

    Key insight:
    - Color only groups the pieces; the actual layering is controlled by directional relation to the anchor, so the original layout determines the z-order.
    """
    out = np.zeros_like(grid)
    anchors = {}
    anchor_centers = set()
    for r in range(1, grid.shape[0] - 1):
        for c in range(1, grid.shape[1] - 1):
            if grid[r, c] == BLACK or grid[r, c] == GREY:
                continue
            ring = grid[r - 1 : r + 2, c - 1 : c + 2].copy()
            if ring[1, 1] != GREY and np.all(ring[[0, 0, 0, 1, 1, 2, 2, 2], [0, 1, 2, 0, 2, 0, 1, 2]] == GREY):
                anchors[int(grid[r, c])] = (r, c)
                anchor_centers.add((r, c))
    order_rank = {"left": 0, "up": 1, "right": 2, "down": 3}
    for color, center in anchors.items():
        comps = [comp for comp in _components(grid == color) if comp != [center]]
        placed = []
        for comp in comps:
            br, bc = _bbox_center(comp)
            dr, dc = br - center[0], bc - center[1]
            direction = "up" if abs(dr) > abs(dc) and dr < 0 else "down" if abs(dr) > abs(dc) else "left" if dc < 0 else "right"
            shift_r = center[0] - br
            shift_c = center[1] - bc
            placed.append((order_rank[direction], [(r + shift_r, c + shift_c) for r, c in comp]))
        for _, coords in sorted(placed, key=lambda item: item[0]):
            for r, c in coords:
                out[r, c] = color
    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    grid = np.zeros((24, 24), dtype=int)
    colors = list(rng.choice([1, 2, 3, 4, 6, 7, 8, 9], size=int(rng.integers(2, 4)), replace=False))
    shape_lib = [
        np.array([[1, 1, 1]], dtype=bool),
        np.array([[1], [1], [1]], dtype=bool),
        np.array([[1, 1], [1, 0]], dtype=bool),
        np.array([[1, 1], [0, 1]], dtype=bool),
    ]
    anchor_positions = [(7, 7), (7, 16), (16, 7)]
    for color, center in zip(colors, anchor_positions):
        r, c = center
        grid[r - 1 : r + 2, c - 1 : c + 2] = GREY
        grid[r, c] = int(color)
        dirs = ["left", "up", "right", "down"]
        rng.shuffle(dirs)
        count = int(rng.integers(2, 5))
        picks = rng.integers(0, len(shape_lib), size=count)
        for direction, idx in zip(dirs[:count], picks):
            sh = np.array(shape_lib[int(idx)], dtype=bool)
            h, w = sh.shape
            if direction == "left":
                r0, c0 = r - h // 2, c - 7
            elif direction == "right":
                r0, c0 = r - h // 2, c + 4
            elif direction == "up":
                r0, c0 = r - 7, c - w // 2
            else:
                r0, c0 = r + 4, c - w // 2
            grid[r0 : r0 + h, c0 : c0 + w][sh] = int(color)
    return grid
