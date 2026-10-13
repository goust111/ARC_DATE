import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5


def _rect_sum(prefix, r1, c1, r2, c2):
    return (
        prefix[r2 + 1, c2 + 1]
        - prefix[r1, c2 + 1]
        - prefix[r2 + 1, c1]
        + prefix[r1, c1]
    )


def _same_color_component(grid, start, color):
    rows, cols = grid.shape
    q = deque([start])
    seen = {start}
    cells = []
    while q:
        r, c = q.popleft()
        if grid[r, c] != color:
            continue
        cells.append((r, c))
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in seen:
                if grid[nr, nc] == color:
                    seen.add((nr, nc))
                    q.append((nr, nc))
    return cells


def _outside_neighbor(r1, c1, r2, c2, gap):
    r, c = gap
    if (r in (r1, r2)) and (c in (c1, c2)):
        return None
    if r == r1:
        return (r - 1, c)
    if r == r2:
        return (r + 1, c)
    if c == c1:
        return (r, c - 1)
    if c == c2:
        return (r, c + 1)
    return None


def _find_fences(grid):
    rows, cols = grid.shape
    grey = (grid == GREY).astype(np.int16)
    prefix = np.pad(grey, ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)
    fences = []
    seen = set()

    for r1 in range(rows - 4):
        for r2 in range(r1 + 4, rows):
            height = r2 - r1 + 1
            for c1 in range(cols - 4):
                for c2 in range(c1 + 4, cols):
                    width = c2 - c1 + 1
                    perimeter = 2 * (height + width) - 4
                    grey_count = (
                        _rect_sum(prefix, r1, c1, r1, c2)
                        + _rect_sum(prefix, r2, c1, r2, c2)
                        + _rect_sum(prefix, r1 + 1, c1, r2 - 1, c1)
                        + _rect_sum(prefix, r1 + 1, c2, r2 - 1, c2)
                    )
                    if grey_count != perimeter - 2:
                        continue

                    gaps = []
                    bad_border = False
                    for c in range(c1, c2 + 1):
                        for cell in ((r1, c), (r2, c)):
                            if grid[cell] != GREY:
                                if grid[cell] != BLACK:
                                    bad_border = True
                                gaps.append(cell)
                    for r in range(r1 + 1, r2):
                        for cell in ((r, c1), (r, c2)):
                            if grid[cell] != GREY:
                                if grid[cell] != BLACK:
                                    bad_border = True
                                gaps.append(cell)
                    if bad_border or len(gaps) != 2:
                        continue

                    outer_cells = []
                    for gap in gaps:
                        outer = _outside_neighbor(r1, c1, r2, c2, gap)
                        if outer is None:
                            bad_border = True
                            break
                        orow, ocol = outer
                        if not (0 <= orow < rows and 0 <= ocol < cols):
                            bad_border = True
                            break
                        color = int(grid[orow, ocol])
                        if color in (BLACK, GREY):
                            bad_border = True
                            break
                        outer_cells.append((outer, color))
                    if bad_border:
                        continue

                    inner = grid[r1 + 1 : r2, c1 + 1 : c2]
                    inner_colors = inner[(inner != BLACK) & (inner != GREY)]
                    if inner_colors.size == 0:
                        continue
                    colors, counts = np.unique(inner_colors, return_counts=True)
                    if len(colors) != 2 or counts[0] == counts[1]:
                        continue
                    if {int(colors[0]), int(colors[1])} != {outer_cells[0][1], outer_cells[1][1]}:
                        continue

                    rect = (r1, c1, r2, c2)
                    if rect not in seen:
                        seen.add(rect)
                        fences.append((rect, gaps, outer_cells, colors.astype(int), counts.astype(int)))
    return fences


def main(grid: np.ndarray) -> np.ndarray:
    """Use interior vote pixels to recolor the losing exterior candidate region.

    Input structure:
    A black grid contains one or more separated grey one-pixel rectangular
    fences. Each fence border has exactly two black one-cell gaps. The interior
    of each fence contains scattered pixels of exactly two non-grey colors, and
    each gap has a same-colored exterior rectangular candidate region adjacent
    to the outside of the wall.

    Rule:
    For every fence, count the two interior colors. The color with the larger
    count is the winner and the other color is the loser. Clear all colored
    interior vote pixels to black. Among the two exterior candidate components
    touching the gaps, repaint the loser-colored component to the winning color;
    keep grey wall pixels, black gaps, and the winner-colored candidate as they
    are.

    Key insight:
    The grey outline is both a ballot box and a spatial router: the majority
    vote inside the fence determines which exterior color loses, and the two
    wall gaps identify the candidate regions that receive that decision.
    """
    out = grid.copy()
    for (r1, c1, r2, c2), _gaps, outer_cells, colors, counts in _find_fences(grid):
        winner = int(colors[np.argmax(counts)])
        loser = int(colors[1 - int(np.argmax(counts))])

        inner = out[r1 + 1 : r2, c1 + 1 : c2]
        inner[(inner != BLACK) & (inner != GREY)] = BLACK

        for outer, color in outer_cells:
            if color == loser:
                for rr, cc in _same_color_component(grid, outer, loser):
                    out[rr, cc] = winner
    return out


def _dilate(mask):
    rows, cols = mask.shape
    result = np.zeros_like(mask, dtype=bool)
    for dr in (-1, 0, 1):
        r_src_1 = max(0, -dr)
        r_src_2 = min(rows, rows - dr)
        r_dst_1 = max(0, dr)
        r_dst_2 = min(rows, rows + dr)
        for dc in (-1, 0, 1):
            c_src_1 = max(0, -dc)
            c_src_2 = min(cols, cols - dc)
            c_dst_1 = max(0, dc)
            c_dst_2 = min(cols, cols + dc)
            result[r_dst_1:r_dst_2, c_dst_1:c_dst_2] |= mask[
                r_src_1:r_src_2, c_src_1:c_src_2
            ]
    return result


def _perimeter_positions(r1, c1, r2, c2):
    cells = []
    for c in range(c1 + 1, c2):
        cells.append((r1, c))
        cells.append((r2, c))
    for r in range(r1 + 1, r2):
        cells.append((r, c1))
        cells.append((r, c2))
    return cells


def _candidate_bounds(r1, c1, r2, c2, gap):
    r, c = gap
    thickness = random.randint(1, 3)
    if r in (r1, r2):
        max_left = min(2, c - c1 - 1)
        max_right = min(2, c2 - c - 1)
        left = random.randint(0, max_left)
        right = random.randint(0, max_right)
        if random.random() < 0.45 and left + right < max_left + max_right:
            if left < max_left and (right == max_right or random.random() < 0.5):
                left += 1
            elif right < max_right:
                right += 1
        cc1, cc2 = c - left, c + right
        if r == r1:
            return (r1 - thickness, cc1, r1 - 1, cc2)
        return (r2 + 1, cc1, r2 + thickness, cc2)

    max_up = min(2, r - r1 - 1)
    max_down = min(2, r2 - r - 1)
    up = random.randint(0, max_up)
    down = random.randint(0, max_down)
    if random.random() < 0.45 and up + down < max_up + max_down:
        if up < max_up and (down == max_down or random.random() < 0.5):
            up += 1
        elif down < max_down:
            down += 1
    rr1, rr2 = r - up, r + down
    if c == c1:
        return (rr1, c1 - thickness, rr2, c1 - 1)
    return (rr1, c2 + 1, rr2, c2 + thickness)


def _paint_rect(grid, bounds, color):
    r1, c1, r2, c2 = bounds
    grid[r1 : r2 + 1, c1 : c2 + 1] = color


def _try_add_group(grid, occupied):
    rows, cols = grid.shape
    fence_h = random.randint(5, 9)
    fence_w = random.randint(5, 10)
    if rows < fence_h + 8 or cols < fence_w + 8:
        return False

    r1 = random.randint(3, rows - fence_h - 4)
    c1 = random.randint(3, cols - fence_w - 4)
    r2 = r1 + fence_h - 1
    c2 = c1 + fence_w - 1

    perimeter = _perimeter_positions(r1, c1, r2, c2)
    random.shuffle(perimeter)
    gaps = perimeter[:2]

    candidate_bounds = []
    for gap in gaps:
        bounds = _candidate_bounds(r1, c1, r2, c2, gap)
        br1, bc1, br2, bc2 = bounds
        if br1 < 0 or bc1 < 0 or br2 >= rows or bc2 >= cols:
            return False
        candidate_bounds.append(bounds)

    proposal = np.zeros_like(grid)
    proposal[r1, c1 : c2 + 1] = GREY
    proposal[r2, c1 : c2 + 1] = GREY
    proposal[r1 : r2 + 1, c1] = GREY
    proposal[r1 : r2 + 1, c2] = GREY
    for gap in gaps:
        proposal[gap] = BLACK

    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], 2)
    if random.random() < 0.5:
        candidate_colors = colors
    else:
        candidate_colors = [colors[1], colors[0]]

    for bounds, color in zip(candidate_bounds, candidate_colors):
        br1, bc1, br2, bc2 = bounds
        if np.any(proposal[br1 : br2 + 1, bc1 : bc2 + 1] != BLACK):
            return False
        _paint_rect(proposal, bounds, color)

    inner_cells = [(r, c) for r in range(r1 + 1, r2) for c in range(c1 + 1, c2)]
    random.shuffle(inner_cells)
    capacity = len(inner_cells)
    loser_count = random.randint(1, min(5, max(1, capacity // 3)))
    winner_count = random.randint(loser_count + 1, min(capacity - loser_count, loser_count + 6))
    winner_color = random.choice(colors)
    loser_color = colors[1] if winner_color == colors[0] else colors[0]
    for cell in inner_cells[:winner_count]:
        proposal[cell] = winner_color
    for cell in inner_cells[winner_count : winner_count + loser_count]:
        proposal[cell] = loser_color

    proposed_mask = proposal != BLACK
    if np.any(_dilate(occupied) & proposed_mask):
        return False

    grid[proposed_mask] = proposal[proposed_mask]
    occupied[proposed_mask] = True
    return True


def generate_input() -> np.ndarray:
    if random.random() < 0.34:
        target_groups = 1
        min_size = 18
    elif random.random() < 0.67:
        target_groups = 2
        min_size = 23
    else:
        target_groups = 3
        min_size = 27

    rows = random.randint(min_size, 30)
    cols = random.randint(min_size, 30)
    grid = np.zeros((rows, cols), dtype=np.int8)
    occupied = np.zeros((rows, cols), dtype=bool)

    added = 0
    for _ in range(400):
        if added >= target_groups:
            break
        if _try_add_group(grid, occupied):
            added += 1

    if added == 0:
        while not _try_add_group(grid, occupied):
            pass
    return grid
