from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
FREE_COLORS = [1, 2, 3, 4, 6, 7, 8]
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_CANVAS = 30


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            stack = [(r, c)]
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _square_frames(grid: np.ndarray, color: int) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for pts in _components(grid == color):
        rows = [r for r, _ in pts]
        cols = [c for _, c in pts]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        side_h = r1 - r0 + 1
        side_w = c1 - c0 + 1
        if side_h != side_w or side_h < 3:
            continue
        side = side_h
        if len(pts) != 4 * side - 4:
            continue

        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros((side, side), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if not bool(np.all(sub[border] == color)):
            continue
        if side > 2 and bool(np.any(sub[1:-1, 1:-1] == color)):
            continue

        frames.append(
            {
                "r": int(r0),
                "c": int(c0),
                "side": int(side),
                "inner": sub[1:-1, 1:-1].copy(),
            }
        )
    frames.sort(key=lambda frame: (int(frame["r"]), int(frame["c"])))
    return frames


def _rotation_steps(reference_mask: np.ndarray, candidate_mask: np.ndarray) -> int:
    for clockwise_turns in range(4):
        if np.array_equal(np.rot90(reference_mask, -clockwise_turns), candidate_mask):
            return clockwise_turns
    raise ValueError("candidate pattern does not match the prototype under rotation")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each target patch by the angle encoded by its neighboring template frame.

    Input structure:
    - The background is black.
    - Several small maroon square frames each contain a monochrome asymmetric template on black.
    - One maroon frame in the top-left corner area is the prototype reference with no adjacent target.
    - Every other maroon frame has an equally sized grey square frame directly to its right or directly below it; the grey frame encloses scattered colored pixels on black.

    Rule:
    - Detect the prototype frame and treat its interior template as the 0-degree reference.
    - For each non-prototype maroon frame, compare its interior template to the prototype under 0/90/180/270-degree clockwise rotation to recover the encoded angle.
    - Rotate the entire interior of that frame's adjacent grey target region by the same clockwise angle, writing the rotated pixels back inside the grey border and clearing their old locations when they move.
    - Keep all maroon frames, grey frames, and the prototype contents unchanged.

    Key insight:
    - The small asymmetric template is not decorative; it is a rotation code that must be decoded relative to a single prototype before applying the same transform to a separate neighboring target.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    maroon_frames = _square_frames(g, MAROON)
    grey_frames = _square_frames(g, GREY)
    grey_lookup = {(int(frame["r"]), int(frame["c"]), int(frame["side"])): frame for frame in grey_frames}

    prototype_candidates: list[dict[str, object]] = []
    tasks: list[tuple[dict[str, object], dict[str, object]]] = []
    for frame in maroon_frames:
        r = int(frame["r"])
        c = int(frame["c"])
        side = int(frame["side"])
        matches = []
        right_key = (r, c + side, side)
        down_key = (r + side, c, side)
        if right_key in grey_lookup:
            matches.append(grey_lookup[right_key])
        if down_key in grey_lookup:
            matches.append(grey_lookup[down_key])

        if not matches:
            prototype_candidates.append(frame)
        else:
            tasks.append((frame, matches[0]))

    if not prototype_candidates:
        prototype_candidates = maroon_frames[:1]
    prototype = min(prototype_candidates, key=lambda frame: (int(frame["r"]), int(frame["c"])))
    reference_mask = np.asarray(prototype["inner"], dtype=int) != BLACK

    for template_frame, target_frame in tasks:
        template_mask = np.asarray(template_frame["inner"], dtype=int) != BLACK
        clockwise_turns = _rotation_steps(reference_mask, template_mask)
        rotated_inner = np.rot90(np.asarray(target_frame["inner"], dtype=int), -clockwise_turns)
        r = int(target_frame["r"])
        c = int(target_frame["c"])
        side = int(target_frame["side"])
        out[r + 1 : r + side - 1, c + 1 : c + side - 1] = rotated_inner

    return out


def _all_rotation_variants(mask: np.ndarray) -> int:
    return len({np.rot90(mask, k).tobytes() for k in range(4)})


def _random_connected_mask(size: int) -> np.ndarray:
    min_cells = 3
    max_cells = min(size * size - 2, size + 3)
    for _ in range(300):
        target_cells = random.randint(min_cells, max_cells)
        mask = np.zeros((size, size), dtype=bool)
        seed = (random.randrange(size), random.randrange(size))
        mask[seed] = True
        frontier = set()
        for dr, dc in NEIGHBORS4:
            nr, nc = seed[0] + dr, seed[1] + dc
            if 0 <= nr < size and 0 <= nc < size:
                frontier.add((nr, nc))

        while int(mask.sum()) < target_cells and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            if bool(mask[cell]):
                continue
            mask[cell] = True
            for dr, dc in NEIGHBORS4:
                nr, nc = cell[0] + dr, cell[1] + dc
                if 0 <= nr < size and 0 <= nc < size and not bool(mask[nr, nc]):
                    frontier.add((nr, nc))

        if int(mask.sum()) != target_cells:
            continue
        rows, cols = np.where(mask)
        if len(set(map(int, rows))) < 2 or len(set(map(int, cols))) < 2:
            continue
        if _all_rotation_variants(mask) != 4:
            continue
        return mask
    raise RuntimeError("failed to generate a non-symmetric prototype pattern")


def _random_target_inner(size: int, palette: list[int], clockwise_turns: int) -> np.ndarray:
    cells = [(r, c) for r in range(size) for c in range(size)]
    min_points = max(3, size - 1)
    max_points = min(size * size - 1, size + 4)
    for _ in range(300):
        n_colors = 2 if len(palette) == 2 else random.choice([2, 3])
        colors = random.sample(palette, n_colors)
        n_points = random.randint(max(n_colors, min_points), max_points)
        coords = random.sample(cells, n_points)
        random.shuffle(coords)
        inner = np.zeros((size, size), dtype=int)

        for color, (r, c) in zip(colors, coords, strict=False):
            inner[r, c] = int(color)
        for r, c in coords[n_colors:]:
            inner[r, c] = int(random.choice(colors))

        if len(set(map(int, inner[inner != BLACK].tolist()))) < 2:
            continue
        if clockwise_turns != 0 and bool(np.array_equal(inner, np.rot90(inner, -clockwise_turns))):
            continue
        return inner
    raise RuntimeError("failed to generate a non-degenerate target patch")


def _can_place(occupied: np.ndarray, r: int, c: int, h: int, w: int, pad: int = 1) -> bool:
    h0, w0 = occupied.shape
    if r < 0 or c < 0 or r + h > h0 or c + w > w0:
        return False
    r0 = max(0, r - pad)
    c0 = max(0, c - pad)
    r1 = min(h0, r + h + pad)
    c1 = min(w0, c + w + pad)
    return not bool(occupied[r0:r1, c0:c1].any())


def _occupy(occupied: np.ndarray, r: int, c: int, h: int, w: int, pad: int = 1) -> None:
    h0, w0 = occupied.shape
    r0 = max(0, r - pad)
    c0 = max(0, c - pad)
    r1 = min(h0, r + h + pad)
    c1 = min(w0, c + w + pad)
    occupied[r0:r1, c0:c1] = True


def _draw_square_frame(grid: np.ndarray, r: int, c: int, outer_side: int, color: int, inner: np.ndarray) -> None:
    grid[r : r + outer_side, c : c + outer_side] = BLACK
    grid[r, c : c + outer_side] = color
    grid[r + outer_side - 1, c : c + outer_side] = color
    grid[r : r + outer_side, c] = color
    grid[r : r + outer_side, c + outer_side - 1] = color
    grid[r + 1 : r + outer_side - 1, c + 1 : c + outer_side - 1] = inner


def generate_input() -> np.ndarray:
    for _ in range(400):
        inner_side = random.choice([3, 4, 5])
        outer_side = inner_side + 2
        task_count = random.randint(2, 4)
        pattern_color = random.choice(FREE_COLORS)
        target_palette = [color for color in FREE_COLORS if color != pattern_color]

        prototype_mask = _random_connected_mask(inner_side)
        prototype_inner = np.where(prototype_mask, pattern_color, BLACK).astype(int)
        task_angles = random.sample([0, 1, 2, 3], task_count)

        side_pool = ["right", "down"] * 2
        random.shuffle(side_pool)
        task_sides = side_pool[:task_count]

        grid = np.zeros((MAX_CANVAS, MAX_CANVAS), dtype=int)
        occupied = np.zeros_like(grid, dtype=bool)

        proto_r = random.choice([0, 1])
        proto_c = random.choice([0, 1])
        _draw_square_frame(grid, proto_r, proto_c, outer_side, MAROON, prototype_inner)
        _occupy(occupied, proto_r, proto_c, outer_side, outer_side, pad=1)

        ok = True
        for clockwise_turns, side in zip(task_angles, task_sides, strict=False):
            target_inner = _random_target_inner(inner_side, target_palette, clockwise_turns)
            if side == "right":
                pack_h, pack_w = outer_side, 2 * outer_side
            else:
                pack_h, pack_w = 2 * outer_side, outer_side

            placed = False
            for _trial in range(400):
                r = random.randint(0, MAX_CANVAS - pack_h)
                c = random.randint(0, MAX_CANVAS - pack_w)
                if proto_r <= r < proto_r + outer_side and proto_c <= c < proto_c + outer_side:
                    continue
                if not _can_place(occupied, r, c, pack_h, pack_w, pad=1):
                    continue

                rotated_mask = np.rot90(prototype_mask, -clockwise_turns)
                rotated_inner = np.where(rotated_mask, pattern_color, BLACK).astype(int)
                if side == "right":
                    _draw_square_frame(grid, r, c, outer_side, MAROON, rotated_inner)
                    _draw_square_frame(grid, r, c + outer_side, outer_side, GREY, target_inner)
                else:
                    _draw_square_frame(grid, r, c, outer_side, MAROON, rotated_inner)
                    _draw_square_frame(grid, r + outer_side, c, outer_side, GREY, target_inner)
                _occupy(occupied, r, c, pack_h, pack_w, pad=1)
                placed = True
                break

            if not placed:
                ok = False
                break

        if not ok:
            continue

        filled = np.argwhere(grid != BLACK)
        r1 = int(filled[:, 0].max())
        c1 = int(filled[:, 1].max())
        extra_bottom = random.randint(0, min(2, MAX_CANVAS - r1 - 1))
        extra_right = random.randint(0, min(2, MAX_CANVAS - c1 - 1))
        cropped = grid[: r1 + 1 + extra_bottom, : c1 + 1 + extra_right].copy()

        transformed = main(cropped)
        if transformed.shape != cropped.shape:
            continue
        if bool(np.array_equal(transformed, cropped)):
            continue
        return cropped

    raise RuntimeError("failed to generate a valid input grid")
