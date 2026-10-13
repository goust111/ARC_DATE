from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
FRAME = 9
GEN_RETRIES = 400
PLACE_RETRIES = 300


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(pts)
    return parts


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _rect_area(box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = box
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _extract_slots(sub: np.ndarray, bg: int, frame_color: int) -> list[tuple[int, int, int, int]] | None:
    h, w = sub.shape
    if h < 5 or w < 5:
        return None
    if not (
        np.all(sub[0, :] == frame_color)
        and np.all(sub[-1, :] == frame_color)
        and np.all(sub[:, 0] == frame_color)
        and np.all(sub[:, -1] == frame_color)
    ):
        return None
    if set(map(int, np.unique(sub))) - {bg, frame_color}:
        return None

    visited = np.zeros((h, w), dtype=bool)
    slots: list[tuple[int, int, int, int]] = []
    for r in range(h):
        for c in range(w):
            if sub[r, c] != bg or visited[r, c]:
                continue
            visited[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            touches_border = False
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                if rr == 0 or cc == 0 or rr == h - 1 or cc == w - 1:
                    touches_border = True
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and sub[nr, nc] == bg and not visited[nr, nc]:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            if touches_border:
                return None
            box = _bbox(pts)
            if len(pts) != _rect_area(box):
                return None
            slots.append(box)

    if not slots:
        return None

    row_tops = sorted({r0 for r0, _, _, _ in slots})
    col_lefts = sorted({c0 for _, c0, _, _ in slots})
    if len(row_tops) * len(col_lefts) != len(slots):
        return None

    by_pos = {(r0, c0): (r0, c0, r1, c1) for r0, c0, r1, c1 in slots}
    if len(by_pos) != len(slots):
        return None

    for r0 in row_tops:
        row_slots = [box for box in slots if box[0] == r0]
        if len(row_slots) != len(col_lefts):
            return None
        heights = {r1 - rr0 for rr0, _, r1, _ in row_slots}
        if len(heights) != 1:
            return None
    for c0 in col_lefts:
        col_slots = [box for box in slots if box[1] == c0]
        if len(col_slots) != len(row_tops):
            return None
        widths = {c1 - cc0 for _, cc0, _, c1 in col_slots}
        if len(widths) != 1:
            return None
    for r0 in row_tops:
        for c0 in col_lefts:
            if (r0, c0) not in by_pos:
                return None

    return sorted(slots)


def _find_frame(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int], list[tuple[int, int, int, int]]]:
    best = None
    best_score = None
    for color in sorted(set(map(int, np.unique(grid))) - {bg}):
        for pts in _components(grid == color):
            box = _bbox(pts)
            r0, c0, r1, c1 = box
            sub = grid[r0 : r1 + 1, c0 : c1 + 1]
            slots = _extract_slots(sub, bg, color)
            if slots is None:
                continue
            score = (len(slots), _rect_area(box), len(pts))
            if best is None or score > best_score:
                best = (int(color), box, slots)
                best_score = score
    assert best is not None, "main: failed to locate the shelf frame"
    return best


def _read_strip(grid: np.ndarray, bg: int, frame_color: int, frame_box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, _r1, c1 = frame_box
    assert r0 > 0, "main: frame must have a strip row directly above it"
    strip = [int(v) for v in grid[r0 - 1, c0 : c1 + 1]]
    assert len(strip) >= 2, "main: strip must contain at least two cells"
    assert all(v not in (bg, frame_color) for v in strip), "main: strip row must be fully colored"
    a = strip[0]
    b = next((v for v in strip[1:] if v != a), None)
    assert b is not None, "main: strip must alternate between two colors"
    for i, v in enumerate(strip):
        assert v == (a if i % 2 == 0 else b), "main: strip must follow A B A B ... pattern"
    return a, b


def _collect_pieces(
    grid: np.ndarray,
    bg: int,
    frame_color: int,
    frame_box: tuple[int, int, int, int],
) -> dict[tuple[int, int], tuple[np.ndarray, int]]:
    h, w = grid.shape
    r0, c0, r1, c1 = frame_box
    ignore = np.zeros((h, w), dtype=bool)
    ignore[r0 : r1 + 1, c0 : c1 + 1] = True
    ignore[r0 - 1, c0 : c1 + 1] = True

    pieces: dict[tuple[int, int], tuple[np.ndarray, int]] = {}
    for color in sorted(set(map(int, np.unique(grid))) - {bg, frame_color}):
        mask = (grid == color) & ~ignore
        for pts in _components(mask):
            box = _bbox(pts)
            pr0, pc0, pr1, pc1 = box
            piece_mask = np.zeros((pr1 - pr0 + 1, pc1 - pc0 + 1), dtype=bool)
            for rr, cc in pts:
                piece_mask[rr - pr0, cc - pc0] = True
            size = piece_mask.shape
            assert size not in pieces, f"main: duplicate piece size {size}"
            pieces[size] = (piece_mask, int(color))
    assert pieces, "main: expected at least one outside piece"
    return pieces


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fit scattered pieces into a partitioned frame and recolor every even slot-row using the strip code.

    Input structure:
    - The background is black. One large maroon rectangular frame is subdivided by maroon lines into rectangular slots whose interiors are empty.
    - Outside the frame are scattered monochrome connected fragments, one fragment per slot; each fragment's bounding box size matches exactly one slot.
    - Directly above the frame is a one-row alternating strip A B A B ... using two non-background, non-frame colors.

    Rule:
    - Identify the frame and its enclosed rectangular slots.
    - Match each outside fragment to the unique slot with the same bounding-box height and width, then place the fragment into that slot with top-left alignment.
    - Read the strip as two alternating operations: A means keep the fragment color, B means recolor to B. Odd slot rows keep their original color, even slot rows are recolored to B.

    Key insight:
    - The task combines size-based jigsaw placement with a second clue: the alternating strip encodes an odd/even row recoloring schedule, so row position determines whether a placed fragment keeps its color or turns into B.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    frame_color, frame_box, slots = _find_frame(g, bg)
    _a_color, recolor = _read_strip(g, bg, frame_color, frame_box)
    pieces = _collect_pieces(g, bg, frame_color, frame_box)

    r0, c0, r1, c1 = frame_box
    out = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    row_tops = sorted({sr0 for sr0, _, _, _ in slots})
    row_index = {top: idx for idx, top in enumerate(row_tops)}

    slot_sizes = {}
    for sr0, sc0, sr1, sc1 in slots:
        size = (sr1 - sr0 + 1, sc1 - sc0 + 1)
        assert size not in slot_sizes, f"main: duplicate slot size {size}"
        slot_sizes[size] = (sr0, sc0, sr1, sc1)

    assert set(slot_sizes) == set(pieces), "main: slots and pieces must match one-to-one by size"

    for size, (sr0, sc0, _sr1, _sc1) in slot_sizes.items():
        piece_mask, piece_color = pieces[size]
        fill_color = piece_color if row_index[sr0] % 2 == 0 else recolor
        patch = out[sr0 : sr0 + size[0], sc0 : sc0 + size[1]]
        patch[piece_mask] = fill_color

    return out


def _random_piece_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    col = random.randrange(w)
    path_cols = [col]
    mask[0, col] = True
    for r in range(1, h):
        col = min(w - 1, max(0, col + random.choice([-1, 0, 1])))
        path_cols.append(col)
        mask[r, col] = True

    cross_row = random.randrange(h)
    mask[cross_row, :] = True

    total = h * w
    low = max(int(mask.sum()), max(3, int(np.ceil(total * 0.45))))
    high = max(low, total - 1)
    target = random.randint(low, high)

    while int(mask.sum()) < target:
        frontier: set[tuple[int, int]] = set()
        for r in range(h):
            for c in range(w):
                if not mask[r, c]:
                    continue
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        frontier.add((nr, nc))
        if not frontier:
            break
        rr, cc = random.choice(tuple(frontier))
        mask[rr, cc] = True

    assert np.any(mask[0, :]) and np.any(mask[-1, :]) and np.any(mask[:, 0]) and np.any(mask[:, -1])
    assert int(mask.sum()) < total
    return mask


def _build_frame(
    row_heights: list[int],
    col_widths: list[int],
    frame_color: int,
    bg: int,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    fh = sum(row_heights) + len(row_heights) + 1
    fw = sum(col_widths) + len(col_widths) + 1
    frame = np.full((fh, fw), frame_color, dtype=int)
    slots: list[dict[str, object]] = []
    rr = 1
    for rh in row_heights:
        cc = 1
        for cw in col_widths:
            frame[rr : rr + rh, cc : cc + cw] = bg
            slots.append({"bbox": (rr, cc, rr + rh - 1, cc + cw - 1), "size": (rh, cw)})
            cc += cw + 1
        rr += rh + 1
    return frame, slots


def _expand_box(box: tuple[int, int, int, int], h: int, w: int, pad: int = 1) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _paint_piece(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    h, w = mask.shape
    patch = grid[top : top + h, left : left + w]
    patch[mask] = color


def _expected_output(
    frame: np.ndarray,
    slots: list[dict[str, object]],
    piece_by_size: dict[tuple[int, int], tuple[np.ndarray, int]],
    recolor: int,
) -> np.ndarray:
    out = frame.copy()
    row_tops = sorted({int(slot["bbox"][0]) for slot in slots})
    row_index = {top: idx for idx, top in enumerate(row_tops)}
    for slot in slots:
        sr0, sc0, _sr1, _sc1 = slot["bbox"]
        size = slot["size"]
        mask, color = piece_by_size[size]
        fill = color if row_index[sr0] % 2 == 0 else recolor
        patch = out[sr0 : sr0 + size[0], sc0 : sc0 + size[1]]
        patch[mask] = fill
    return out


def _slot_layout() -> tuple[list[int], list[int]]:
    choices = [(2, 2), (2, 3), (3, 2)]
    rows, cols = random.choice(choices)
    row_heights = random.sample([2, 3, 4, 5], rows)
    col_widths = random.sample([2, 3, 4, 5, 6], cols)
    random.shuffle(row_heights)
    random.shuffle(col_widths)
    return row_heights, col_widths


def _gen_attempt() -> np.ndarray | None:
    row_heights, col_widths = _slot_layout()
    frame, slots = _build_frame(row_heights, col_widths, FRAME, BG)
    fh, fw = frame.shape
    slot_count = len(slots)

    recolor = random.choice([1, 2, 3, 4, 5, 6, 7, 8])
    piece_palette = [c for c in range(1, 10) if c not in (FRAME, recolor)]
    if len(piece_palette) < slot_count:
        return None
    piece_colors = random.sample(piece_palette, slot_count)
    strip_a = random.choice(piece_palette)

    piece_by_size: dict[tuple[int, int], tuple[np.ndarray, int]] = {}
    for slot, color in zip(slots, piece_colors):
        size = slot["size"]
        piece_by_size[size] = (_random_piece_mask(size[0], size[1]), color)

    min_h = max(fh + 4, 18)
    min_w = max(fw + 6, 20)
    if min_h > 30 or min_w > 30:
        return None
    gh = random.randint(min_h, 30)
    gw = random.randint(min_w, 30)

    if gh - fh < 2 or gw - fw < 2:
        return None
    top = random.randint(1, gh - fh - 1)
    left = random.randint(1, gw - fw - 1)

    grid = np.full((gh, gw), BG, dtype=int)
    grid[top : top + fh, left : left + fw] = frame
    for i in range(fw):
        grid[top - 1, left + i] = strip_a if i % 2 == 0 else recolor

    blocked = np.zeros((gh, gw), dtype=bool)
    fr_box = (top, left, top + fh - 1, left + fw - 1)
    br0, bc0, br1, bc1 = _expand_box(fr_box, gh, gw, pad=1)
    blocked[br0 : br1 + 1, bc0 : bc1 + 1] = True

    pieces = list(piece_by_size.items())
    random.shuffle(pieces)
    pieces.sort(key=lambda item: item[0][0] * item[0][1], reverse=True)

    for size, (mask, color) in pieces:
        ph, pw = size
        placed = False
        for _ in range(PLACE_RETRIES):
            pr = random.randint(0, gh - ph)
            pc = random.randint(0, gw - pw)
            box = (pr, pc, pr + ph - 1, pc + pw - 1)
            if np.any(blocked[pr : pr + ph, pc : pc + pw]):
                continue
            _paint_piece(grid, pr, pc, mask, color)
            er0, ec0, er1, ec1 = _expand_box(box, gh, gw, pad=1)
            blocked[er0 : er1 + 1, ec0 : ec1 + 1] = True
            placed = True
            break
        if not placed:
            return None

    expected = _expected_output(frame, slots, piece_by_size, recolor)

    try:
        produced = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(produced, expected):
        return None

    colors = Counter(map(int, grid.ravel()))
    if colors[BG] <= max(v for k, v in colors.items() if k != BG):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
