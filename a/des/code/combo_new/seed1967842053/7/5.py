import random
import numpy as np


GREY = 5


def _background_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    best_color = int(colors[0])
    best_count = -1
    for color, count in zip(colors, counts):
        color = int(color)
        if color == GREY:
            continue
        if int(count) > best_count:
            best_color = color
            best_count = int(count)
    return best_color


def _grey_components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for sr in range(h):
        for sc in range(w):
            if not mask[sr, sc] or seen[sr, sc]:
                continue
            stack = [(sr, sc)]
            seen[sr, sc] = True
            comp = []
            while stack:
                r, c = stack.pop()
                comp.append((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _is_rectangle_border(comp):
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    top, bottom = min(rows), max(rows)
    left, right = min(cols), max(cols)
    if bottom - top < 2 or right - left < 2:
        return None

    expected = set()
    for c in range(left, right + 1):
        expected.add((top, c))
        expected.add((bottom, c))
    for r in range(top, bottom + 1):
        expected.add((r, left))
        expected.add((r, right))
    if set(comp) == expected:
        return top, bottom, left, right
    return None


def _winner_for_region(grid: np.ndarray, bg: int, box):
    top, bottom, left, right = box
    stats = {}
    for r in range(top + 1, bottom):
        for c in range(left + 1, right):
            color = int(grid[r, c])
            if color == bg or color == GREY:
                continue
            count, first = stats.get(color, (0, (grid.shape[0] + 1, grid.shape[1] + 1)))
            pos = (r, c)
            if pos < first:
                first = pos
            stats[color] = (count + 1, first)
    if not stats:
        return None

    return min(stats, key=lambda color: (-stats[color][0], stats[color][1], color))


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each grey container border by the majority vote color inside it, then erase the votes.

    Input structure:
    A mostly uniform background contains one or more single-pixel-wide grey rectangular closed borders.
    Each rectangle encloses scattered colored voting pixels that are neither grey nor background; unrelated
    colored distractors may also appear outside all rectangles.
    Rule:
    For every grey rectangular border, count the non-background, non-grey pixels strictly inside it. Recolor
    the whole border to the color with the highest count. If tied, use the tied color whose earliest internal
    vote is highest in the grid, and then leftmost if needed. Replace all internal voting pixels with the
    background color while leaving exterior distractors unchanged.
    Key insight:
    The grey outlines are containers and the interior colored dots are temporary ballots; only votes inside a
    container affect that container, and those votes disappear after deciding the border color.
    """
    arr = np.array(grid, copy=True)
    out = arr.copy()
    bg = _background_color(arr)

    for comp in _grey_components(arr == GREY):
        box = _is_rectangle_border(comp)
        if box is None:
            continue
        winner = _winner_for_region(arr, bg, box)
        if winner is None:
            continue
        top, bottom, left, right = box
        out[top, left : right + 1] = winner
        out[bottom, left : right + 1] = winner
        out[top : bottom + 1, left] = winner
        out[top : bottom + 1, right] = winner
        interior = arr[top + 1 : bottom, left + 1 : right]
        vote_mask = (interior != bg) & (interior != GREY)
        out[top + 1 : bottom, left + 1 : right][vote_mask] = bg
    return out


def _candidate_boxes(h, w):
    boxes = []
    for top in range(1, h - 4):
        for left in range(1, w - 4):
            for height in range(5, min(10, h - top) + 1):
                for width in range(5, min(10, w - left) + 1):
                    bottom = top + height - 1
                    right = left + width - 1
                    if bottom < h - 1 and right < w - 1:
                        boxes.append((top, bottom, left, right))
    random.shuffle(boxes)
    return boxes


def _separated(box, chosen):
    top, bottom, left, right = box
    for ot, ob, ol, or_ in chosen:
        if not (bottom + 1 < ot or ob + 1 < top or right + 1 < ol or or_ + 1 < left):
            return False
    return True


def _place_votes(grid, box, colors, mode):
    top, bottom, left, right = box
    cells = [(r, c) for r in range(top + 1, bottom) for c in range(left + 1, right)]
    used = set()

    def take(pool, count):
        random.shuffle(pool)
        chosen = []
        for cell in pool:
            if cell not in used:
                chosen.append(cell)
                used.add(cell)
                if len(chosen) == count:
                    break
        return chosen

    def paint(color, chosen):
        for r, col in chosen:
            grid[r, col] = color

    if mode == 0:
        random.shuffle(cells)
        paint(colors[0], cells[:4])
        paint(colors[1], cells[4:6])
        if len(colors) == 3:
            paint(colors[2], cells[6:8])
        return

    if mode == 1:
        first_a = (top + 1, random.randint(left + 1, right - 1))
        first_b = (top + 2, random.randint(left + 1, right - 1))
        used.update([first_a, first_b])
        a_cells = [first_a] + take([(top + 1, c) for c in range(left + 1, right)], 2)
        b_cells = [first_b] + take([cell for cell in cells if cell[0] >= top + 2], 2)
        paint(colors[0], a_cells)
        paint(colors[1], b_cells)
        if len(colors) == 3:
            paint(colors[2], take(cells, 1))
        return

    first_a = (top + 1, left + 1)
    first_b = (top + 1, left + 2)
    used.update([first_a, first_b])
    a_cells = [first_a] + take([cell for cell in cells if cell > first_a], 2)
    b_cells = [first_b] + take([cell for cell in cells if cell > first_b], 2)
    paint(colors[0], a_cells)
    paint(colors[1], b_cells)
    if len(colors) == 3:
        paint(colors[2], take(cells, 1))


def generate_input() -> np.ndarray:
    h = random.randint(14, 30)
    w = random.randint(14, 30)
    bg = random.choice([0, 1, 2, 3, 4, 6, 7, 8, 9])
    grid = np.full((h, w), bg, dtype=int)

    possible = _candidate_boxes(h, w)
    target_count = random.randint(1, min(4, max(1, (h * w) // 120)))
    boxes = []
    for box in possible:
        if _separated(box, boxes):
            boxes.append(box)
            if len(boxes) == target_count:
                break
    if not boxes:
        boxes = [(1, min(6, h - 2), 1, min(6, w - 2))]

    palette = [c for c in range(10) if c not in (bg, GREY)]
    for i, box in enumerate(boxes):
        top, bottom, left, right = box
        grid[top, left : right + 1] = GREY
        grid[bottom, left : right + 1] = GREY
        grid[top : bottom + 1, left] = GREY
        grid[top : bottom + 1, right] = GREY
        random.shuffle(palette)
        colors = palette[: random.randint(2, 3)]
        _place_votes(grid, box, colors, i % 3)

    outside = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] != bg:
                continue
            inside_any = False
            for top, bottom, left, right in boxes:
                if top < r < bottom and left < c < right:
                    inside_any = True
                    break
            if not inside_any:
                outside.append((r, c))
    random.shuffle(outside)
    for r, c in outside[: random.randint(1, min(6, len(outside)))]:
        grid[r, c] = random.choice(palette)

    return grid
