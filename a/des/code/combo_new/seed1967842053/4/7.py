import random
from collections import deque

import numpy as np


def _background(grid):
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _solid_rectangle(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    area = (r1 - r0 + 1) * (c1 - c0 + 1)
    if area != len(cells):
        return None
    return r0, r1, c0, c1


def _envelope_from_component(cells, grid, color, bg):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    expected = set()
    top_gap = (r0, (c0 + c1) // 2)
    bottom_gap = (r1, (c0 + c1) // 2)
    left_gap = ((r0 + r1) // 2, c0)
    right_gap = ((r0 + r1) // 2, c1)
    gaps = {"up": top_gap, "down": bottom_gap, "left": left_gap, "right": right_gap}

    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))

    cell_set = set(cells)
    for direction, gap in gaps.items():
        if cell_set != expected - {gap}:
            continue
        if grid[gap] != bg:
            continue
        interior = grid[r0 + 1:r1, c0 + 1:c1]
        if interior.size and np.all(interior == bg):
            return {
                "color": color,
                "direction": direction,
                "outer": (r0, r1, c0, c1),
                "inner": (r0 + 1, r1 - 1, c0 + 1, c1 - 1),
            }
    return None


def _find_objects(grid):
    bg = _background(grid)
    envelopes = []
    papers = {}
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        comps = _components(grid == color)
        for cells in comps:
            env = _envelope_from_component(cells, grid, color, bg)
            if env is not None:
                envelopes.append(env)
                continue
            rect = _solid_rectangle(cells)
            if rect is not None:
                papers.setdefault(color, []).append(rect)
    return bg, envelopes, papers


def main(grid: np.ndarray) -> np.ndarray:
    """Place each colored letter into its matching open envelope after folding.

    Input structure:
    The grid has a most-common background color. Each envelope is a one-pixel
    rectangular border in a distinct color, with exactly one centered missing
    pixel on one side that marks the opening direction. The inside of each
    envelope is background. Separate solid rectangles in the same colors are
    the letters; each letter has one dimension equal to the matching envelope
    interior and the other dimension exactly twice that interior dimension.

    Rule:
    Match every letter to the envelope with the same color. If the envelope
    opens upward or downward, fold the letter across its horizontal middle, so
    its final height is half of the source height. If the envelope opens left or
    right, fold the letter across its vertical middle, so its final width is
    half of the source width. Clear the original letter rectangle to background,
    then fill the matching envelope interior with the letter color. Envelope
    borders and their missing opening pixels remain unchanged.

    Key insight:
    The color gives the pairing, the missing border pixel gives the fold axis,
    and the two-to-one letter/envelope size relationship determines exactly
    which interior rectangle is filled.
    """
    arr = np.array(grid, copy=True)
    bg, envelopes, papers = _find_objects(arr)

    for env in envelopes:
        color = env["color"]
        ir0, ir1, ic0, ic1 = env["inner"]
        ih = ir1 - ir0 + 1
        iw = ic1 - ic0 + 1
        want = (ih * 2, iw) if env["direction"] in ("up", "down") else (ih, iw * 2)

        candidates = []
        for rect in papers.get(color, []):
            pr0, pr1, pc0, pc1 = rect
            ph = pr1 - pr0 + 1
            pw = pc1 - pc0 + 1
            if (ph, pw) == want:
                candidates.append(rect)
        if not candidates:
            continue

        pr0, pr1, pc0, pc1 = candidates[0]
        arr[pr0:pr1 + 1, pc0:pc1 + 1] = bg
        arr[ir0:ir1 + 1, ic0:ic1 + 1] = color

    return arr


def _can_place(occupied, r0, r1, c0, c1, pad=1):
    h, w = occupied.shape
    rr0 = max(0, r0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc0 = max(0, c0 - pad)
    cc1 = min(w - 1, c1 + pad)
    return not occupied[rr0:rr1 + 1, cc0:cc1 + 1].any()


def _mark(occupied, r0, r1, c0, c1):
    occupied[r0:r1 + 1, c0:c1 + 1] = True


def _place_rectangle(occupied, height, width):
    h, w = occupied.shape
    spots = []
    for r0 in range(h - height + 1):
        for c0 in range(w - width + 1):
            r1 = r0 + height - 1
            c1 = c0 + width - 1
            if _can_place(occupied, r0, r1, c0, c1):
                spots.append((r0, r1, c0, c1))
    if not spots:
        return None
    rect = random.choice(spots)
    _mark(occupied, *rect)
    return rect


def _draw_envelope(grid, r0, r1, c0, c1, color, direction):
    grid[r0, c0:c1 + 1] = color
    grid[r1, c0:c1 + 1] = color
    grid[r0:r1 + 1, c0] = color
    grid[r0:r1 + 1, c1] = color
    if direction == "up":
        grid[r0, (c0 + c1) // 2] = 0
    elif direction == "down":
        grid[r1, (c0 + c1) // 2] = 0
    elif direction == "left":
        grid[(r0 + r1) // 2, c0] = 0
    else:
        grid[(r0 + r1) // 2, c1] = 0


def generate_input() -> np.ndarray:
    height = random.randint(18, 30)
    width = random.randint(18, 30)
    bg = 0
    grid = np.full((height, width), bg, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    colors = random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], random.randint(2, 4))
    directions = ["up", "down", "left", "right"]
    random.shuffle(directions)

    made = 0
    for i, color in enumerate(colors):
        direction = directions[i % 4]
        for _ in range(80):
            if direction in ("up", "down"):
                inner_h = random.randint(2, 4)
                inner_w = random.randint(3, 7)
                env_h, env_w = inner_h + 2, inner_w + 2
                paper_h, paper_w = inner_h * 2, inner_w
            else:
                inner_h = random.randint(3, 7)
                inner_w = random.randint(2, 4)
                env_h, env_w = inner_h + 2, inner_w + 2
                paper_h, paper_w = inner_h, inner_w * 2

            env_rect = _place_rectangle(occupied, env_h, env_w)
            if env_rect is None:
                continue
            paper_rect = _place_rectangle(occupied, paper_h, paper_w)
            if paper_rect is None:
                _mark(occupied, *env_rect)
                continue

            er0, er1, ec0, ec1 = env_rect
            pr0, pr1, pc0, pc1 = paper_rect
            _draw_envelope(grid, er0, er1, ec0, ec1, color, direction)
            grid[pr0:pr1 + 1, pc0:pc1 + 1] = color
            made += 1
            break

    if made >= 2:
        return grid
    return generate_input()
