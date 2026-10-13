from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
SIDES = ["top", "right", "bottom", "left"]
ROT_K = {"top": 0, "right": 3, "bottom": 2, "left": 1}


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    assert pts.size, "cannot crop an empty mask"
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _mask_key(mask: np.ndarray) -> tuple[str, ...]:
    tight = _crop_mask(np.asarray(mask, dtype=bool))
    return tuple("".join("1" if x else "0" for x in row) for row in tight)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for sr in range(h):
        for sc in range(w):
            if not mask[sr, sc] or seen[sr, sc]:
                continue
            q: deque[tuple[int, int]] = deque([(sr, sc)])
            seen[sr, sc] = True
            comp: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                comp.append((int(r), int(c)))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _separator_col(g: np.ndarray) -> int:
    h, w = g.shape
    candidates = [c for c in range(w) if np.all(g[:, c] == GREY)]
    assert candidates, "expected a full-height grey separator column"
    best = None
    for c in candidates:
        left = set(map(int, np.unique(g[:, :c]))) - {BLACK, GREY}
        right = set(map(int, np.unique(g[:, c + 1 :]))) - {BLACK, GREY}
        score = (len(left & right), min(c, w - c - 1))
        if best is None or score > best[0]:
            best = (score, c)
    assert best is not None
    return int(best[1])


def _marker_side(rel_r: int, rel_c: int, n: int) -> str | None:
    mid = n // 2
    if rel_r == 0 and rel_c == mid:
        return "top"
    if rel_r == mid and rel_c == n - 1:
        return "right"
    if rel_r == n - 1 and rel_c == mid:
        return "bottom"
    if rel_r == mid and rel_c == 0:
        return "left"
    return None


def _find_frames(g: np.ndarray, sep_c: int) -> list[dict[str, int | str]]:
    h = g.shape[0]
    left_w = sep_c
    frames: list[dict[str, int | str]] = []
    max_n = min(h, left_w)
    for n in range(5, max_n + 1, 2):
        for r0 in range(0, h - n + 1):
            for c0 in range(0, left_w - n + 1):
                inner = g[r0 + 1 : r0 + n - 1, c0 + 1 : c0 + n - 1]
                if not np.all(inner == BLACK):
                    continue

                non_grey: list[tuple[int, int, int]] = []
                ok = True
                for rr in range(n):
                    for cc in range(n):
                        if rr not in (0, n - 1) and cc not in (0, n - 1):
                            continue
                        val = int(g[r0 + rr, c0 + cc])
                        if val != GREY:
                            non_grey.append((rr, cc, val))
                            if val in (BLACK, GREY):
                                ok = False
                                break
                    if not ok:
                        break
                if not ok or len(non_grey) != 1:
                    continue

                rr, cc, color = non_grey[0]
                side = _marker_side(rr, cc, n)
                if side is None:
                    continue
                frames.append({"r": r0, "c": c0, "n": n, "color": color, "side": side})

    colors = [int(f["color"]) for f in frames]
    assert frames and len(colors) == len(set(colors)), "expected uniquely colored marked frames"
    return sorted(frames, key=lambda f: (int(f["r"]), int(f["c"])))


def _templates_by_color(g: np.ndarray, sep_c: int, colors: list[int]) -> dict[int, np.ndarray]:
    right = g[:, sep_c + 1 :]
    templates: dict[int, np.ndarray] = {}
    for color in colors:
        mask = right == int(color)
        comps = _components(mask)
        assert len(comps) == 1, "each template color must form one 4-connected component"
        comp_mask = np.zeros(mask.shape, dtype=bool)
        for r, c in comps[0]:
            comp_mask[r, c] = True
        templates[int(color)] = _crop_mask(comp_mask)
    return templates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each right-panel colored template into the matching marked grey frame.

    Input structure:
    - Black background with one full-height grey separator column splitting the grid into left and right panels.
    - The left panel contains same-sized hollow grey square frames with black interiors.
    - Each frame has exactly one non-grey colored marker replacing the midpoint pixel of its top, right, bottom, or left border.
    - The right panel contains one monochrome 4-connected template per marker color, using that same color.

    Rule:
    - Match each frame to the right-panel template with the same color as its marker.
    - A top marker means no rotation; a right marker means clockwise 90 degrees; a bottom marker means 180 degrees; a left marker means counterclockwise 90 degrees.
    - Rotate the template, tightly crop it, and center it inside the matching frame interior.
    - Clear the right panel to black while keeping the separator, frame borders, and marker pixels unchanged.

    Key insight:
    - Color gives the frame-template pairing, and the marker's side of the square encodes the template rotation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    sep_c = _separator_col(g)
    frames = _find_frames(g, sep_c)
    colors = [int(f["color"]) for f in frames]
    templates = _templates_by_color(g, sep_c, colors)

    out = g.copy()
    out[:, sep_c + 1 :] = BLACK

    for frame in frames:
        r0 = int(frame["r"])
        c0 = int(frame["c"])
        n = int(frame["n"])
        color = int(frame["color"])
        side = str(frame["side"])

        rotated = np.rot90(templates[color], ROT_K[side])
        rotated = _crop_mask(rotated)
        rh, rw = rotated.shape
        interior = n - 2
        assert rh < interior + 1 and rw < interior + 1, "template must fit inside frame"

        pr = r0 + 1 + (interior - rh) // 2
        pc = c0 + 1 + (interior - rw) // 2
        patch = out[pr : pr + rh, pc : pc + rw]
        patch[rotated] = color
        out[pr : pr + rh, pc : pc + rw] = patch

    return out


def _draw_frame(grid: np.ndarray, r0: int, c0: int, n: int, side: str, color: int) -> None:
    grid[r0, c0 : c0 + n] = GREY
    grid[r0 + n - 1, c0 : c0 + n] = GREY
    grid[r0 : r0 + n, c0] = GREY
    grid[r0 : r0 + n, c0 + n - 1] = GREY
    mid = n // 2
    if side == "top":
        grid[r0, c0 + mid] = color
    elif side == "right":
        grid[r0 + mid, c0 + n - 1] = color
    elif side == "bottom":
        grid[r0 + n - 1, c0 + mid] = color
    else:
        grid[r0 + mid, c0] = color


def _place_frame_positions(h: int, left_w: int, n: int, count: int) -> list[tuple[int, int]] | None:
    positions: list[tuple[int, int]] = []
    for _ in range(count):
        candidates = [
            (r, c)
            for r in range(1, h - n)
            for c in range(1, left_w - n)
        ]
        random.shuffle(candidates)
        for r, c in candidates:
            clear = True
            for er, ec in positions:
                separate = (
                    r + n <= er - 1
                    or er + n <= r - 1
                    or c + n <= ec - 1
                    or ec + n <= c - 1
                )
                if not separate:
                    clear = False
                    break
            if clear:
                positions.append((r, c))
                break
        else:
            return None
    return positions


def _rotations_all_distinct(mask: np.ndarray) -> bool:
    return len({_mask_key(np.rot90(mask, k)) for k in range(4)}) == 4


def _random_connected_mask(max_dim: int) -> np.ndarray:
    if max_dim <= 2:
        base = np.array([[1, 1], [1, 0]], dtype=bool)
        return np.rot90(base, random.randrange(4))

    for _ in range(100):
        h = random.randint(2, max_dim)
        w = random.randint(2, max_dim)
        max_area = min(h * w - 1, max(4, max_dim + 3))
        if max_area < 3:
            continue
        target = random.randint(3, max_area)
        cells = {(random.randrange(h), random.randrange(w))}
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _crop_mask(mask)
        if mask.shape[0] <= max_dim and mask.shape[1] <= max_dim and _rotations_all_distinct(mask):
            return mask

    base = np.array([[1, 1, 0], [0, 1, 1]], dtype=bool)
    return np.rot90(base, random.randrange(4))


def _can_place_template(occupied: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> bool:
    h, w = occupied.shape
    mh, mw = mask.shape
    if r0 < 1 or c0 < 1 or r0 + mh >= h or c0 + mw >= w:
        return False
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h, r0 + mh + 1)
    cc1 = min(w, c0 + mw + 1)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _place_templates(
    grid: np.ndarray,
    sep_c: int,
    right_w: int,
    items: list[tuple[int, np.ndarray]],
) -> bool:
    h = grid.shape[0]
    occupied = np.zeros((h, right_w), dtype=bool)
    order = items[:]
    random.shuffle(order)
    for color, mask in order:
        mh, mw = mask.shape
        candidates = [
            (r, c)
            for r in range(1, h - mh)
            for c in range(1, right_w - mw)
        ]
        random.shuffle(candidates)
        for r, c in candidates:
            if not _can_place_template(occupied, mask, r, c):
                continue
            grid[r : r + mh, sep_c + 1 + c : sep_c + 1 + c + mw][mask] = int(color)
            occupied[r : r + mh, c : c + mw] = True
            break
        else:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    frame_n = random.choice([5, 7])
    count = random.choice([4, 5])
    rows_needed = 3 if count == 5 else 2
    min_h = rows_needed * frame_n + rows_needed + 1
    if min_h > 30:
        return None
    h = random.randint(min_h, 30)

    right_w = random.randint(8, 12)
    min_left_w = 2 * frame_n + 3
    max_left_w = 30 - 1 - right_w
    if min_left_w > max_left_w:
        return None
    sep_c = random.randint(min_left_w, max_left_w)
    w = sep_c + 1 + right_w

    positions = _place_frame_positions(h, sep_c, frame_n, count)
    if positions is None:
        return None

    colors = random.sample(COLORS, count)
    sides = random.sample(SIDES, len(SIDES))
    if count > 4:
        sides.append(random.choice(SIDES))
    random.shuffle(sides)

    interior = frame_n - 2
    max_template_dim = interior - 1
    masks = [_random_connected_mask(max_template_dim) for _ in range(count)]

    grid = np.zeros((h, w), dtype=int)
    grid[:, sep_c] = GREY

    for (r, c), color, side in zip(positions, colors, sides):
        _draw_frame(grid, r, c, frame_n, side, color)

    if not _place_templates(grid, sep_c, right_w, list(zip(colors, masks))):
        return None

    expected = grid.copy()
    expected[:, sep_c + 1 :] = BLACK
    for (r, c), color, side, mask in zip(positions, colors, sides, masks):
        rot = _crop_mask(np.rot90(mask, ROT_K[side]))
        rh, rw = rot.shape
        pr = r + 1 + (interior - rh) // 2
        pc = c + 1 + (interior - rw) // 2
        expected[pr : pr + rh, pc : pc + rw][rot] = color

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid seed1030756833/5 input")
