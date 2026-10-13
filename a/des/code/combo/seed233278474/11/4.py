import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
STRIPE_COLORS = (3, 4, 5, 6, 7, 8, 9)
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
PERP = {
    "up": (0, 1),
    "down": (0, 1),
    "left": (1, 0),
    "right": (1, 0),
}
GENERATION_ATTEMPTS = 400
OBJECT_PLACEMENT_ATTEMPTS = 120
POLYOMINO_ATTEMPTS = 50


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in DIRS.values():
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _mask_from_coords(coords: np.ndarray) -> tuple[np.ndarray, int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, r0, c0


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = mask.shape
    patch = grid[r0 : r0 + h, c0 : c0 + w]
    patch[mask] = int(color)


def _find_stripes(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = grid.shape
    horizontal: list[tuple[int, int]] = []
    vertical: list[tuple[int, int]] = []

    for r in range(h):
        row = grid[r, :]
        if np.all(row == row[0]):
            color = int(row[0])
            if color not in {BG, BLUE, RED}:
                horizontal.append((r, color))

    for c in range(w):
        col = grid[:, c]
        if np.all(col == col[0]):
            color = int(col[0])
            if color not in {BG, BLUE, RED}:
                vertical.append((c, color))

    return horizontal, vertical


def _nearest_stripe_color(stripes: list[tuple[int, int]], center: float) -> int:
    assert stripes, "expected at least one stripe for the arrow orientation"
    best_pos, best_color = min(stripes, key=lambda item: (abs(item[0] - center), item[0]))
    _ = best_pos
    return int(best_color)


def _decode_arrow(arrow: np.ndarray, tail: tuple[int, int], direction: str) -> int:
    pts = {tuple(map(int, pt)) for pt in arrow}
    count = len(pts) - 4
    assert count >= 1, "arrow shaft must have positive length"

    dr, dc = DIRS[direction]
    pr, pc = PERP[direction]
    tr, tc = tail
    expected = set()
    for i in range(count):
        expected.add((tr + i * dr, tc + i * dc))
    br = tr + count * dr
    bc = tc + count * dc
    expected.add((br, bc))
    expected.add((br + pr, bc + pc))
    expected.add((br - pr, bc - pc))
    expected.add((br + dr, bc + dc))
    assert pts == expected, "blue component must be a canonical shaft-plus-head arrow"
    return count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each red shape along its adjacent blue arrow direction. The arrow shaft
    length gives the number of copies, and each copy is recolored to the nearest
    stripe color measured perpendicular to the motion direction.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    horizontal_stripes, vertical_stripes = _find_stripes(g)

    red_comps = _components(g == RED)
    blue_comps = _components(g == BLUE)
    assert red_comps and blue_comps, "expected red shapes and blue arrows"

    blue_id = -np.ones((h, w), dtype=int)
    for idx, comp in enumerate(blue_comps):
        blue_id[comp[:, 0], comp[:, 1]] = idx

    out = g.copy()
    out[g == RED] = BG
    out[g == BLUE] = BG

    used_arrows: set[int] = set()
    for red in red_comps:
        mask, r0, c0 = _mask_from_coords(red)
        bh, bw = mask.shape

        touching: dict[int, list[tuple[str, tuple[int, int]]]] = {}
        for rr, cc in red:
            for name, (dr, dc) in DIRS.items():
                nr, nc = int(rr + dr), int(cc + dc)
                if 0 <= nr < h and 0 <= nc < w and blue_id[nr, nc] >= 0:
                    touching.setdefault(int(blue_id[nr, nc]), []).append((name, (nr, nc)))

        assert len(touching) == 1, "each red object must touch exactly one blue arrow"
        arrow_idx, contacts = next(iter(touching.items()))
        assert arrow_idx not in used_arrows, "blue arrows must pair one-to-one with red objects"
        used_arrows.add(arrow_idx)

        directions = {name for name, _tail in contacts}
        tails = {tail for _name, tail in contacts}
        assert len(directions) == 1 and len(tails) == 1, "arrow must touch the object on a single side"
        direction = next(iter(directions))
        tail = next(iter(tails))
        count = _decode_arrow(blue_comps[arrow_idx], tail, direction)

        if direction in ("left", "right"):
            stripes = horizontal_stripes
            step_r, step_c = 0, bw if direction == "right" else -bw
            center = r0 + (bh - 1) / 2.0
        else:
            stripes = vertical_stripes
            step_r, step_c = (bh if direction == "down" else -bh), 0
            center = c0 + (bw - 1) / 2.0

        color = _nearest_stripe_color(stripes, center)
        for i in range(1, count + 1):
            rr = r0 + i * step_r
            cc = c0 + i * step_c
            assert 0 <= rr and rr + bh <= h and 0 <= cc and cc + bw <= w, "copy must fit in the grid"
            _paint_mask(out, mask, rr, cc, color)

    return out


def _random_polyomino() -> np.ndarray:
    for _ in range(POLYOMINO_ATTEMPTS):
        h = random.randint(1, 4)
        w = random.randint(1, 4)
        area_min = 1 if h * w <= 2 else max(2, (h * w + 1) // 2)
        area = random.randint(area_min, h * w)

        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in DIRS.values():
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            cells.add(random.choice(frontier))

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        coords = np.argwhere(mask)
        r0, c0, r1, c1 = _bbox(coords)
        mask = mask[r0 : r1 + 1, c0 : c1 + 1]

        if random.choice((False, True)):
            mask = np.flipud(mask)
        if random.choice((False, True)):
            mask = np.fliplr(mask)
        if random.choice((False, True)):
            mask = mask.T.copy()
        return mask.copy()
    raise RuntimeError("failed to sample a polyomino")


def _attachment_indices(mask: np.ndarray, direction: str) -> list[int]:
    h, w = mask.shape
    if direction == "right":
        return [r for r in range(h) if mask[r, w - 1]]
    if direction == "left":
        return [r for r in range(h) if mask[r, 0]]
    if direction == "down":
        return [c for c in range(w) if mask[h - 1, c]]
    return [c for c in range(w) if mask[0, c]]


def _paint_arrow(grid: np.ndarray, tail: tuple[int, int], direction: str, count: int) -> None:
    dr, dc = DIRS[direction]
    pr, pc = PERP[direction]
    tr, tc = tail
    cells = []
    for i in range(count):
        cells.append((tr + i * dr, tc + i * dc))
    br = tr + count * dr
    bc = tc + count * dc
    cells.extend([(br, bc), (br + pr, bc + pc), (br - pr, bc - pc), (br + dr, bc + dc)])
    for r, c in cells:
        assert grid[r, c] == BG, "generated arrow must occupy free space"
        grid[r, c] = BLUE


def _choose_stripe_colors(count: int) -> list[int]:
    colors = [random.choice(STRIPE_COLORS) for _ in range(count)]
    if len(set(colors)) == 1:
        idx = random.randrange(count)
        colors[idx] = random.choice([c for c in STRIPE_COLORS if c != colors[idx]])
    return colors


def _generate_horizontal() -> np.ndarray | None:
    obj_count = random.randint(2, 4)
    gap_heights = [random.randint(4, 5) for _ in range(obj_count)]
    top_margin = random.randint(1, 2)
    bottom_margin = random.randint(1, 2)
    h = top_margin + bottom_margin + sum(gap_heights) + (obj_count + 1)
    w = random.randint(20, 30)

    grid = np.full((h, w), BG, dtype=int)
    stripe_rows = [top_margin]
    for gap in gap_heights:
        stripe_rows.append(stripe_rows[-1] + gap + 1)
    stripe_colors = _choose_stripe_colors(len(stripe_rows))
    for r, color in zip(stripe_rows, stripe_colors):
        grid[r, :] = int(color)

    bands = [(stripe_rows[i] + 1, stripe_rows[i + 1] - 1) for i in range(obj_count)]
    dir_choices = ["left", "right"]
    if obj_count >= 3 and random.random() < 0.7:
        directions = [dir_choices[i % 2] for i in range(obj_count)]
        random.shuffle(directions)
    else:
        directions = [random.choice(dir_choices) for _ in range(obj_count)]

    for (band_r0, band_r1), direction in zip(bands, directions):
        placed = False
        for _ in range(OBJECT_PLACEMENT_ATTEMPTS):
            mask = _random_polyomino()
            mh, mw = mask.shape
            attachments = _attachment_indices(mask, direction)
            if not attachments:
                continue

            counts = list(range(1, 5))
            random.shuffle(counts)
            for count in counts:
                reach = max(count * mw, count + 2)
                if direction == "right":
                    c_min, c_max = 0, w - mw - reach
                else:
                    c_min, c_max = reach, w - mw
                if c_min > c_max:
                    continue

                stripe_pref = ["top", "bottom"]
                random.shuffle(stripe_pref)
                for target in stripe_pref:
                    candidates: list[tuple[int, int]] = []
                    for attach in attachments:
                        r_lo = max(band_r0, band_r0 + 1 - attach)
                        r_hi = min(band_r1 - mh + 1, band_r1 - 1 - attach)
                        for r0 in range(r_lo, r_hi + 1):
                            center = r0 + (mh - 1) / 2.0
                            top_dist = abs((band_r0 - 1) - center)
                            bottom_dist = abs((band_r1 + 1) - center)
                            if target == "top" and top_dist < bottom_dist:
                                candidates.append((r0, attach))
                            if target == "bottom" and bottom_dist < top_dist:
                                candidates.append((r0, attach))
                    if not candidates:
                        continue

                    r0, attach = random.choice(candidates)
                    c0 = random.randint(c_min, c_max)
                    _paint_mask(grid, mask, r0, c0, RED)
                    tail_c = c0 + mw if direction == "right" else c0 - 1
                    _paint_arrow(grid, (r0 + attach, tail_c), direction, count)
                    placed = True
                    break
                if placed:
                    break
            if placed:
                break
        if not placed:
            return None

    return grid


def _generate_vertical() -> np.ndarray | None:
    obj_count = random.randint(2, 4)
    gap_widths = [random.randint(4, 5) for _ in range(obj_count)]
    left_margin = random.randint(1, 2)
    right_margin = random.randint(1, 2)
    w = left_margin + right_margin + sum(gap_widths) + (obj_count + 1)
    h = random.randint(20, 30)

    grid = np.full((h, w), BG, dtype=int)
    stripe_cols = [left_margin]
    for gap in gap_widths:
        stripe_cols.append(stripe_cols[-1] + gap + 1)
    stripe_colors = _choose_stripe_colors(len(stripe_cols))
    for c, color in zip(stripe_cols, stripe_colors):
        grid[:, c] = int(color)

    bands = [(stripe_cols[i] + 1, stripe_cols[i + 1] - 1) for i in range(obj_count)]
    dir_choices = ["up", "down"]
    if obj_count >= 3 and random.random() < 0.7:
        directions = [dir_choices[i % 2] for i in range(obj_count)]
        random.shuffle(directions)
    else:
        directions = [random.choice(dir_choices) for _ in range(obj_count)]

    for (band_c0, band_c1), direction in zip(bands, directions):
        placed = False
        for _ in range(OBJECT_PLACEMENT_ATTEMPTS):
            mask = _random_polyomino()
            mh, mw = mask.shape
            attachments = _attachment_indices(mask, direction)
            if not attachments:
                continue

            counts = list(range(1, 5))
            random.shuffle(counts)
            for count in counts:
                reach = max(count * mh, count + 2)
                if direction == "down":
                    r_min, r_max = 0, h - mh - reach
                else:
                    r_min, r_max = reach, h - mh
                if r_min > r_max:
                    continue

                stripe_pref = ["left", "right"]
                random.shuffle(stripe_pref)
                for target in stripe_pref:
                    candidates: list[tuple[int, int]] = []
                    for attach in attachments:
                        c_lo = max(band_c0, band_c0 + 1 - attach)
                        c_hi = min(band_c1 - mw + 1, band_c1 - 1 - attach)
                        for c0 in range(c_lo, c_hi + 1):
                            center = c0 + (mw - 1) / 2.0
                            left_dist = abs((band_c0 - 1) - center)
                            right_dist = abs((band_c1 + 1) - center)
                            if target == "left" and left_dist < right_dist:
                                candidates.append((c0, attach))
                            if target == "right" and right_dist < left_dist:
                                candidates.append((c0, attach))
                    if not candidates:
                        continue

                    c0, attach = random.choice(candidates)
                    r0 = random.randint(r_min, r_max)
                    _paint_mask(grid, mask, r0, c0, RED)
                    tail_r = r0 + mh if direction == "down" else r0 - 1
                    _paint_arrow(grid, (tail_r, c0 + attach), direction, count)
                    placed = True
                    break
                if placed:
                    break
            if placed:
                break
        if not placed:
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_horizontal() if random.choice((False, True)) else _generate_vertical()
        if grid is None:
            continue
        try:
            out = main(grid)
        except Exception:
            continue

        if np.any(out == RED) or np.any(out == BLUE):
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
