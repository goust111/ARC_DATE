import random
from collections import Counter

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 300


def get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[np.argmax(counts)])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(sorted(pts), dtype=int))
    return comps


def piece_sig(coords: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = bbox_coords(coords)
    norm = tuple(sorted((int(r) - r0, int(c) - c0) for r, c in coords))
    return int(r1 - r0 + 1), int(c1 - c0 + 1), norm


def factor_pairs(n: int) -> list[tuple[int, int]]:
    out = []
    for h in range(1, n + 1):
        if n % h == 0:
            out.append((h, n // h))
    return out


def build_requirements(
    H: int,
    W: int,
    comps_by_color: dict[int, list[np.ndarray]],
) -> np.ndarray:
    req = np.full((H, W), -1, dtype=int)

    corner_colors = [
        color
        for color, comps in comps_by_color.items()
        if len(comps) == 4 and all(len(comp) == 1 for comp in comps)
    ]
    if len(corner_colors) > 1:
        return req
    corner_color = corner_colors[0] if corner_colors else None
    if corner_color is not None and H >= 2 and W >= 2:
        req[0, 0] = corner_color
        req[0, W - 1] = corner_color
        req[H - 1, 0] = corner_color
        req[H - 1, W - 1] = corner_color

    if W >= 3:
        bars = []
        for color, comps in comps_by_color.items():
            for comp in comps:
                ch, cw, norm = piece_sig(comp)
                if ch == 1 and cw == W - 2 and len(norm) == cw:
                    bars.append((int(comp[:, 0].min()), color))
        if 1 <= len(bars) <= 2:
            bars.sort()
            top_color = bars[0][1]
            bot_color = bars[-1][1]
            req[0, 1 : W - 1] = top_color
            req[H - 1, 1 : W - 1] = bot_color

    vertical_candidates = []
    for color, comps in comps_by_color.items():
        if len(comps) != 2:
            continue
        sigs = [piece_sig(comp) for comp in comps]
        (h0, w0, norm0), (h1, w1, norm1) = sigs
        if w0 == 1 and w1 == 1 and h0 == h1 and norm0 == norm1 and h0 < H:
            vertical_candidates.append((color, h0))
    if len(vertical_candidates) == 1 and W >= 2:
        color, length = vertical_candidates[0]
        top = (H - length) // 2
        if 0 <= top and top + length <= H:
            req[top : top + length, 0] = color
            req[top : top + length, W - 1] = color

    return req


def assemble(
    types: list[dict],
    H: int,
    W: int,
    req: np.ndarray,
) -> np.ndarray | None:
    board = np.full((H, W), -1, dtype=int)
    remaining = [int(t["cnt"]) for t in types]
    memo: set[tuple[bytes, tuple[int, ...]]] = set()

    def can_place(ti: int, r0: int, c0: int) -> bool:
        t = types[ti]
        color = int(t["color"])
        placed = {(r0 + dr, c0 + dc) for dr, dc in t["coords"]}
        for r, c in placed:
            if not (0 <= r < H and 0 <= c < W):
                return False
            if board[r, c] != -1:
                return False
            mc = W - 1 - c
            if req[r, c] not in (-1, color):
                return False
            if req[r, mc] not in (-1, color):
                return False
            if board[r, mc] not in (-1, color):
                return False
        for r, c in placed:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and board[nr, nc] == color and (nr, nc) not in placed:
                    return False
        return True

    def placements_for(ti: int) -> list[tuple[int, int]]:
        t = types[ti]
        out = []
        for r0 in range(H - int(t["h"]) + 1):
            for c0 in range(W - int(t["w"]) + 1):
                if can_place(ti, r0, c0):
                    out.append((r0, c0))
        return out

    def place(ti: int, r0: int, c0: int) -> list[tuple[int, int]]:
        color = int(types[ti]["color"])
        changed = []
        for dr, dc in types[ti]["coords"]:
            r, c = r0 + dr, c0 + dc
            board[r, c] = color
            changed.append((r, c))
        return changed

    def unplace(changed: list[tuple[int, int]]) -> None:
        for r, c in changed:
            board[r, c] = -1

    def search() -> np.ndarray | None:
        if all(v == 0 for v in remaining):
            if np.any(board == -1):
                return None
            if not np.array_equal(board, board[:, ::-1]):
                return None
            return board.copy()

        key = (board.tobytes(), tuple(remaining))
        if key in memo:
            return None

        best_ti = None
        best_places = None
        for ti, cnt in enumerate(remaining):
            if cnt == 0:
                continue
            places = placements_for(ti)
            if not places:
                memo.add(key)
                return None
            if best_places is None or len(places) < len(best_places):
                best_ti = ti
                best_places = places
            if len(best_places) == 1:
                break

        assert best_ti is not None and best_places is not None
        for r0, c0 in best_places:
            changed = place(best_ti, r0, c0)
            remaining[best_ti] -= 1
            sol = search()
            remaining[best_ti] += 1
            unplace(changed)
            if sol is not None:
                return sol
        memo.add(key)
        return None

    return search()


def solve_all(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    comps_by_color: dict[int, list[np.ndarray]] = {}
    pieces = []
    for color in sorted(set(map(int, np.unique(g))) - {bg}):
        comps = components_mask(g == color)
        comps_by_color[color] = comps
        for comp in comps:
            h, w, norm = piece_sig(comp)
            pieces.append((color, h, w, norm))

    counts = Counter(pieces)
    types = []
    for (color, h, w, norm), cnt in sorted(
        counts.items(),
        key=lambda item: (-len(item[0][3]), item[0][0], item[0][3]),
    ):
        types.append(
            {
                "color": int(color),
                "h": int(h),
                "w": int(w),
                "coords": norm,
                "cnt": int(cnt),
            }
        )

    total_area = sum(len(t["coords"]) * int(t["cnt"]) for t in types)
    solutions = []
    seen = set()
    for H, W in factor_pairs(total_area):
        if H < 2 or W < 2:
            continue
        req = build_requirements(H, W, comps_by_color)
        sol = assemble(types, H, W, req)
        if sol is None:
            continue
        key = sol.tobytes() + str(sol.shape).encode()
        if key not in seen:
            seen.add(key)
            solutions.append(sol)
    return solutions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered colored fragments into a left-right symmetric rectangle.
    """
    solutions = solve_all(np.asarray(grid, dtype=int))
    assert len(solutions) == 1, "expected a unique assembly"
    return solutions[0]


def output_components(board: np.ndarray) -> list[tuple[int, np.ndarray]]:
    comps = []
    for color in sorted(set(map(int, np.unique(board)))):
        for comp in components_mask(board == color):
            comps.append((color, comp))
    return comps


def grow_shape(open_cells: set[tuple[int, int]], seed: tuple[int, int], size: int) -> set[tuple[int, int]]:
    shape = {seed}
    frontier = set()
    while len(shape) < size:
        frontier.clear()
        for r, c in shape:
            for dr, dc in DIR4:
                nb = (r + dr, c + dc)
                if nb in open_cells and nb not in shape:
                    frontier.add(nb)
        if not frontier:
            break
        shape.add(random.choice(sorted(frontier)))
    return shape


def choose_piece_color(
    board: np.ndarray,
    cells: set[tuple[int, int]],
    palette: list[int],
) -> int | None:
    blocked = set()
    H, W = board.shape
    for r, c in cells:
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and board[nr, nc] != -1:
                blocked.add(int(board[nr, nc]))
    choices = [color for color in palette if color not in blocked]
    return random.choice(choices) if choices else None


def build_output() -> tuple[np.ndarray, int]:
    H = random.randint(4, 7)
    W = random.choice([4, 6, 8])
    bg = random.randint(0, 9)
    colors = [c for c in range(10) if c != bg]
    random.shuffle(colors)
    mode = random.randrange(4)

    board = np.full((H, W), -1, dtype=int)
    marker_colors = []

    if mode >= 1:
        corner_color = colors.pop()
        marker_colors.append(corner_color)
        board[0, 0] = corner_color
        board[0, W - 1] = corner_color
        board[H - 1, 0] = corner_color
        board[H - 1, W - 1] = corner_color

    if mode >= 2 and W >= 4:
        bar_color = colors.pop()
        marker_colors.append(bar_color)
        board[0, 1 : W - 1] = bar_color
        board[H - 1, 1 : W - 1] = bar_color

    if mode >= 3 and H >= 5:
        line_color = colors.pop()
        marker_colors.append(line_color)
        line_len = H - 2 if H >= 5 else H - 4
        top = (H - line_len) // 2
        board[top : top + line_len, 0] = np.where(board[top : top + line_len, 0] == -1, line_color, board[top : top + line_len, 0])
        board[top : top + line_len, W - 1] = np.where(board[top : top + line_len, W - 1] == -1, line_color, board[top : top + line_len, W - 1])

    left_open = {
        (r, c)
        for r in range(H)
        for c in range(W // 2)
        if board[r, c] == -1 and board[r, W - 1 - c] == -1
    }

    piece_palette = colors[:]
    while left_open:
        if not piece_palette:
            raise RuntimeError("ran out of colors")
        seed = random.choice(sorted(left_open))
        target_size = random.randint(2, 4)
        shape_left = grow_shape(left_open, seed, target_size)
        if len(shape_left) == 1:
            target_size = 1
        full_cells = set(shape_left)
        for r, c in shape_left:
            full_cells.add((r, W - 1 - c))
        color = choose_piece_color(board, full_cells, piece_palette)
        if color is None:
            raise RuntimeError("no non-touching color available")
        for r, c in full_cells:
            board[r, c] = color
        piece_palette.remove(color)
        left_open = {
            (r, c)
            for r in range(H)
            for c in range(W // 2)
            if board[r, c] == -1 and board[r, W - 1 - c] == -1
        }

    assert np.all(board != -1)
    assert np.array_equal(board, board[:, ::-1])
    return board, bg


def scatter(board: np.ndarray, bg: int) -> np.ndarray:
    comps = output_components(board)
    random.shuffle(comps)
    H, W = board.shape
    in_h = random.randint(max(H + 5, 12), max(H + 9, 16))
    in_w = random.randint(max(W + 5, 12), max(W + 10, 18))
    grid = np.full((in_h, in_w), bg, dtype=int)
    blocked = np.zeros((in_h, in_w), dtype=bool)

    bars = []
    others = []
    bar_len = W - 2
    for color, comp in comps:
        h, w, norm = piece_sig(comp)
        if h == 1 and w == bar_len and len(norm) == bar_len and W >= 3:
            bars.append((color, comp))
        else:
            others.append((color, comp))
    bars.sort(key=lambda item: item[1][:, 0].min())
    order = bars + others

    for color, comp in order:
        _, _, norm = piece_sig(comp)
        pts = np.array(norm, dtype=int)
        ph = int(pts[:, 0].max() + 1)
        pw = int(pts[:, 1].max() + 1)
        placed = False
        for _ in range(200):
            r0 = random.randint(1, in_h - ph - 2)
            c0 = random.randint(1, in_w - pw - 2)
            if blocked[r0 - 1 : r0 + ph + 1, c0 - 1 : c0 + pw + 1].any():
                continue
            for dr, dc in pts:
                grid[r0 + dr, c0 + dc] = color
            blocked[r0 - 1 : r0 + ph + 1, c0 - 1 : c0 + pw + 1] = True
            placed = True
            break
        if not placed:
            raise RuntimeError("scatter failed")
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            out, bg = build_output()
            inp = scatter(out, bg)
        except RuntimeError:
            continue
        sols = solve_all(inp)
        if len(sols) == 1 and np.array_equal(sols[0], out):
            return inp
    raise RuntimeError("failed to generate valid input")
