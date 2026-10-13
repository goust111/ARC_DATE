from __future__ import annotations

from collections import deque
import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
RNG = np.random.default_rng()

MAX_GENERATION_ATTEMPTS = 300
MAX_SCATTER_ATTEMPTS = 300


def _find_components(grid: np.ndarray, bg: int = BG) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue

            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0 = min(rows)
            r1 = max(rows)
            c0 = min(cols)
            c1 = max(cols)
            patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
            mask = patch == color
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "patch": patch,
                    "mask": mask,
                    "h": patch.shape[0],
                    "w": patch.shape[1],
                    "area": int(mask.sum()),
                }
            )

    return components


def _row_profiles(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    h, _w = mask.shape
    left = np.empty(h, dtype=int)
    right = np.empty(h, dtype=int)

    for r in range(h):
        cols = np.flatnonzero(mask[r])
        if len(cols) == 0:
            return None
        if int(cols[-1] - cols[0] + 1) != int(len(cols)):
            return None
        left[r] = int(cols[0])
        right[r] = int(cols[-1])

    return left, right


def _col_profiles(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    _h, w = mask.shape
    top = np.empty(w, dtype=int)
    bottom = np.empty(w, dtype=int)

    for c in range(w):
        rows = np.flatnonzero(mask[:, c])
        if len(rows) == 0:
            return None
        if int(rows[-1] - rows[0] + 1) != int(len(rows)):
            return None
        top[c] = int(rows[0])
        bottom[c] = int(rows[-1])

    return top, bottom


def _render_horizontal(pieces: list[dict], order: list[int], xs: list[int]) -> np.ndarray | None:
    height = int(pieces[order[0]]["h"])
    width = max(int(x) + int(pieces[idx]["w"]) for idx, x in zip(order, xs))
    out = np.full((height, width), BG, dtype=int)
    occ = np.zeros((height, width), dtype=bool)

    for idx, x0 in zip(order, xs):
        piece = pieces[idx]
        rr, cc = np.nonzero(piece["mask"])
        gc = cc + int(x0)
        if np.any(occ[rr, gc]):
            return None
        occ[rr, gc] = True
        out[rr, gc] = piece["patch"][rr, cc]

    if not bool(occ.all()):
        return None
    return out


def _render_vertical(pieces: list[dict], order: list[int], ys: list[int]) -> np.ndarray | None:
    width = int(pieces[order[0]]["w"])
    height = max(int(y) + int(pieces[idx]["h"]) for idx, y in zip(order, ys))
    out = np.full((height, width), BG, dtype=int)
    occ = np.zeros((height, width), dtype=bool)

    for idx, y0 in zip(order, ys):
        piece = pieces[idx]
        rr, cc = np.nonzero(piece["mask"])
        gr = rr + int(y0)
        if np.any(occ[gr, cc]):
            return None
        occ[gr, cc] = True
        out[gr, cc] = piece["patch"][rr, cc]

    if not bool(occ.all()):
        return None
    return out


def _solve_horizontal(pieces: list[dict]) -> np.ndarray | None:
    if not pieces:
        return None
    heights = {int(piece["h"]) for piece in pieces}
    if len(heights) != 1:
        return None

    profiles: list[dict] = []
    for piece in pieces:
        row_data = _row_profiles(piece["mask"])
        if row_data is None:
            return None
        left, right = row_data
        profiles.append(
            {
                "left": left,
                "right": right,
                "flat_left": bool(np.all(left == 0)),
                "flat_right": bool(np.all(right == int(piece["w"]) - 1)),
            }
        )

    starts = [i for i, prof in enumerate(profiles) if prof["flat_left"]]
    if not starts:
        return None

    n = len(pieces)
    matches: dict[int, list[tuple[int, int]]] = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            offset = profiles[i]["right"] - profiles[j]["left"] + 1
            if bool(np.all(offset == offset[0])) and int(offset[0]) > 0:
                matches[i].append((j, int(offset[0])))

    solutions: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}

    def dfs(path: list[int], used: set[int], xs: list[int]) -> None:
        cur = path[-1]
        if len(path) == n:
            if not profiles[cur]["flat_right"]:
                return
            out = _render_horizontal(pieces, path, xs)
            if out is None:
                return
            solutions[(tuple(map(int, out.shape)), out.tobytes())] = out
            return

        for nxt, delta in matches[cur]:
            if nxt in used:
                continue
            path.append(nxt)
            used.add(nxt)
            xs.append(xs[-1] + int(delta))
            dfs(path, used, xs)
            xs.pop()
            used.remove(nxt)
            path.pop()

    for start in starts:
        dfs([start], {start}, [0])

    if len(solutions) != 1:
        return None
    return next(iter(solutions.values()))


def _solve_vertical(pieces: list[dict]) -> np.ndarray | None:
    if not pieces:
        return None
    widths = {int(piece["w"]) for piece in pieces}
    if len(widths) != 1:
        return None

    profiles: list[dict] = []
    for piece in pieces:
        col_data = _col_profiles(piece["mask"])
        if col_data is None:
            return None
        top, bottom = col_data
        profiles.append(
            {
                "top": top,
                "bottom": bottom,
                "flat_top": bool(np.all(top == 0)),
                "flat_bottom": bool(np.all(bottom == int(piece["h"]) - 1)),
            }
        )

    starts = [i for i, prof in enumerate(profiles) if prof["flat_top"]]
    if not starts:
        return None

    n = len(pieces)
    matches: dict[int, list[tuple[int, int]]] = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            offset = profiles[i]["bottom"] - profiles[j]["top"] + 1
            if bool(np.all(offset == offset[0])) and int(offset[0]) > 0:
                matches[i].append((j, int(offset[0])))

    solutions: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}

    def dfs(path: list[int], used: set[int], ys: list[int]) -> None:
        cur = path[-1]
        if len(path) == n:
            if not profiles[cur]["flat_bottom"]:
                return
            out = _render_vertical(pieces, path, ys)
            if out is None:
                return
            solutions[(tuple(map(int, out.shape)), out.tobytes())] = out
            return

        for nxt, delta in matches[cur]:
            if nxt in used:
                continue
            path.append(nxt)
            used.add(nxt)
            ys.append(ys[-1] + int(delta))
            dfs(path, used, ys)
            ys.pop()
            used.remove(nxt)
            path.pop()

    for start in starts:
        dfs([start], {start}, [0])

    if len(solutions) != 1:
        return None
    return next(iter(solutions.values()))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered interlocking strips back into the original solid rectangle by matching jagged side profiles.

    Input structure:
    - Black background.
    - Several differently colored 4-connected puzzle fragments scattered apart.
    - Each fragment spans the full height or the full width of the hidden rectangle, so every row or every column of a fragment is one contiguous interval.
    - Internal sides may contain single-pixel teeth and matching single-pixel notches; the rectangle's outer border stays straight.

    Rule:
    - Extract each fragment and measure its side profile.
    - In a horizontal assembly, the right profile of one piece and the left profile of its neighbor must differ by one constant offset on every row; analogously, a vertical assembly uses bottom and top profiles on every column.
    - Find the unique order that makes all neighboring profiles interlock and yields flat outer borders at the two exposed ends.
    - Translate the pieces, without rotation or reflection, into that order and output the resulting filled rectangle with the original colors.

    Key insight:
    - Absolute input positions do not matter; the solution is determined by complementary row-wise or column-wise edge signatures that force a single rectangular reconstruction.
    """
    g = np.asarray(grid, dtype=int)
    pieces = _find_components(g, bg=BG)
    assert pieces, "expected at least one fragment"

    candidates: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}
    for solver in (_solve_horizontal, _solve_vertical):
        out = solver(pieces)
        if out is None:
            continue
        candidates[(tuple(map(int, out.shape)), out.tobytes())] = out

    assert len(candidates) == 1, "expected a unique rectangular assembly"
    return next(iter(candidates.values()))


def _random_profile(length: int) -> np.ndarray:
    for _ in range(200):
        profile = np.zeros(length, dtype=int)
        pos = 0
        state = 0
        touched = False
        while pos < length:
            run = int(RNG.integers(1, min(4, length - pos) + 1))
            if state == 0:
                choices = np.array([0, 1, -1], dtype=int)
            else:
                choices = np.array([state, 0], dtype=int)
            state = int(RNG.choice(choices))
            profile[pos : pos + run] = state
            touched = touched or state != 0
            pos += run
        if touched and not bool(np.all(profile == profile[0])):
            return profile
    raise RuntimeError("failed to sample a non-flat seam profile")


def _build_chain_solution() -> np.ndarray:
    piece_count = int(RNG.integers(3, 6))
    height = int(RNG.integers(4, 9))

    for _ in range(200):
        widths = RNG.integers(4, 8, size=piece_count)
        if int(widths.sum()) > 24:
            continue
        if len(set(map(int, widths.tolist()))) == 1:
            continue

        seams = [_random_profile(height) for _ in range(piece_count - 1)]
        ok = True
        for i in range(piece_count):
            left_delta = 0 if i == 0 else seams[i - 1]
            right_delta = 0 if i == piece_count - 1 else seams[i]
            row_widths = widths[i] + right_delta - left_delta
            if int(row_widths.min()) < 2:
                ok = False
                break
        if not ok:
            continue

        total_width = int(widths.sum())
        seam_cols = []
        base = 0
        for i in range(piece_count - 1):
            base += int(widths[i])
            seam_cols.append(base - 1 + seams[i])

        colors = RNG.choice(np.array(PALETTE, dtype=int), size=piece_count, replace=False)
        out = np.full((height, total_width), BG, dtype=int)
        for r in range(height):
            left = 0
            for i in range(piece_count):
                right = total_width - 1 if i == piece_count - 1 else int(seam_cols[i][r])
                out[r, left : right + 1] = int(colors[i])
                left = right + 1

        if bool((out == BG).any()):
            continue
        return out

    raise RuntimeError("failed to build a valid chain solution")


def _apply_global_transform(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, copy=True)
    if bool(RNG.integers(0, 2)):
        out = np.rot90(out, 1)
    if bool(RNG.integers(0, 2)):
        out = np.flipud(out)
    if bool(RNG.integers(0, 2)):
        out = np.fliplr(out)
    return np.array(out, copy=True)


def _dilate(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            r_src0 = max(0, -dr)
            r_src1 = min(h, h - dr)
            c_src0 = max(0, -dc)
            c_src1 = min(w, w - dc)
            r_dst0 = max(0, dr)
            r_dst1 = r_dst0 + (r_src1 - r_src0)
            c_dst0 = max(0, dc)
            c_dst1 = c_dst0 + (c_src1 - c_src0)
            out[r_dst0:r_dst1, c_dst0:c_dst1] |= mask[r_src0:r_src1, c_src0:c_src1]
    return out


def _scatter_pieces(solution: np.ndarray) -> np.ndarray | None:
    pieces = _find_components(solution, bg=BG)
    pieces = sorted(pieces, key=lambda piece: int(piece["h"]) * int(piece["w"]), reverse=True)

    max_h = max(int(piece["h"]) for piece in pieces)
    max_w = max(int(piece["w"]) for piece in pieces)
    padded_area = sum((int(piece["h"]) + 2) * (int(piece["w"]) + 2) for piece in pieces)
    min_h = max(max_h + 2, int(solution.shape[0]) + 2, 12)
    min_w = max(max_w + 2, int(solution.shape[1]) + 2, 12)

    for _ in range(MAX_SCATTER_ATTEMPTS):
        canvas_h = int(RNG.integers(min_h, 31))
        canvas_w = int(RNG.integers(min_w, 31))
        if canvas_h * canvas_w < padded_area:
            continue

        grid = np.full((canvas_h, canvas_w), BG, dtype=int)
        occupied = np.zeros((canvas_h, canvas_w), dtype=bool)
        placed = True

        for piece in pieces:
            patch = piece["patch"]
            mask = piece["mask"]
            h = int(piece["h"])
            w = int(piece["w"])
            dilated = _dilate(occupied)

            tries = 0
            success = False
            while tries < 250:
                tries += 1
                r0 = int(RNG.integers(0, canvas_h - h + 1))
                c0 = int(RNG.integers(0, canvas_w - w + 1))
                if bool((dilated[r0 : r0 + h, c0 : c0 + w] & mask).any()):
                    continue
                rr, cc = np.nonzero(mask)
                grid[r0 + rr, c0 + cc] = patch[rr, cc]
                occupied[r0 + rr, c0 + cc] = True
                success = True
                break

            if not success:
                placed = False
                break

        if placed:
            return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        solution = _apply_global_transform(_build_chain_solution())
        inp = _scatter_pieces(solution)
        if inp is None:
            continue
        try:
            if np.array_equal(main(inp), solution):
                return inp
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid puzzle instance")
