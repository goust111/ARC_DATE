import random
from collections import defaultdict

import numpy as np


BG = 0
GREY = 5

RED = 2
BLUE = 1
GREEN = 3

YELLOW = 4
MAGENTA = 6
ORANGE = 7

STAMP_COLORS = (1, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

FLIP_TO_COLOR = {"h": RED, "v": BLUE, "n": GREEN}
COLOR_TO_FLIP = {RED: "h", BLUE: "v", GREEN: "n"}

MOVE_TO_COLOR = {"r": YELLOW, "d": ORANGE, "rd": MAGENTA}
COLOR_TO_MOVE = {YELLOW: "r", ORANGE: "d", MAGENTA: "rd"}

GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_area(box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = box
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _components(grid: np.ndarray, bg: int) -> list[tuple[int, list[tuple[int, int]]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[tuple[int, list[tuple[int, int]]]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append((color, coords))
    return comps


def _is_rectangle_border(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    pts = set(coords)
    if len(pts) != 2 * (h + w) - 4:
        return False
    for r in range(r0, r1 + 1):
        if (r, c0) not in pts or (r, c1) not in pts:
            return False
    for c in range(c0, c1 + 1):
        if (r0, c) not in pts or (r1, c) not in pts:
            return False
    return True


def _find_frames(grid: np.ndarray, bg: int) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    by_color: dict[int, list[tuple[int, int, int, int]]] = defaultdict(list)
    for color, coords in _components(grid, bg):
        if _is_rectangle_border(coords):
            by_color[int(color)].append(_bbox(coords))

    candidates = [boxes for boxes in by_color.values() if len(boxes) >= 2]
    if not candidates:
        raise ValueError("expected at least two same-color rectangular frames")

    group = max(candidates, key=lambda boxes: sum(_bbox_area(box) for box in boxes))
    group = sorted(group, key=_bbox_area)
    return group[0], group[-1]


def _instruction_rows(
    grid: np.ndarray,
    bg: int,
    stamp_box: tuple[int, int, int, int],
    canvas_box: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    h, w = grid.shape
    outside = np.ones((h, w), dtype=bool)
    for r0, c0, r1, c1 in (stamp_box, canvas_box):
        outside[r0 : r1 + 1, c0 : c1 + 1] = False

    rows: list[tuple[int, int]] = []
    for r in range(h):
        cols = [c for c in range(w) if outside[r, c] and int(grid[r, c]) != bg]
        if not cols:
            continue
        if len(cols) != 2:
            raise ValueError("every instruction row must contain exactly two pixels")
        rows.append((int(grid[r, cols[0]]), int(grid[r, cols[1]])))
    return rows


def _paint_stamp(canvas: np.ndarray, stamp: np.ndarray, top: int, left: int, bg: int) -> None:
    sh, sw = stamp.shape
    if not (0 <= top and 0 <= left and top + sh <= canvas.shape[0] and left + sw <= canvas.shape[1]):
        raise ValueError("stamp placement exceeds canvas bounds")
    patch = canvas[top : top + sh, left : left + sw]
    mask = stamp != bg
    patch[mask] = stamp[mask]


def _run_program(
    stamp: np.ndarray,
    instruction_ops: list[tuple[str, str]],
    canvas_shape: tuple[int, int],
    bg: int,
) -> np.ndarray:
    out = np.full(canvas_shape, bg, dtype=int)
    cur = stamp.copy()
    sh, sw = stamp.shape
    top = 0
    left = 0
    _paint_stamp(out, cur, top, left, bg)

    for flip_op, move_op in instruction_ops:
        if flip_op == "h":
            cur = np.fliplr(cur)
        elif flip_op == "v":
            cur = np.flipud(cur)
        elif flip_op != "n":
            raise ValueError(f"unknown flip op: {flip_op}")

        if move_op == "r":
            left += sw
        elif move_op == "d":
            top += sh
        elif move_op == "rd":
            top += sh
            left += sw
        else:
            raise ValueError(f"unknown move op: {move_op}")

        _paint_stamp(out, cur, top, left, bg)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Execute the instruction strip by cumulatively flipping and shifting a framed stamp onto the output canvas.

    Input structure:
    - The background is the most common color.
    - One small rectangular border frame near the upper left contains the initial stamp pattern.
    - A larger rectangular border frame in the lower part marks an empty output canvas.
    - Outside both frames, each non-background row belongs to the vertical instruction strip and contains exactly two colored pixels.

    Rule:
    - Extract the stamp from the interior of the smaller frame and the blank canvas from the interior of the larger frame.
    - Place the initial stamp unchanged at the canvas top-left corner.
    - For each instruction row from top to bottom, use the first color to choose the next cumulative flip
      (red = left-right flip, blue = top-bottom flip, green = no flip), then use the second color to shift
      the current stamp by one stamp width and/or height (yellow = right, orange = down, magenta = right and down).
    - After each step, stamp only the non-background pixels onto the canvas, with later stamps overwriting earlier ones.

    Key insight:
    - The two-pixel instruction rows separately encode orientation and translation, and the flip always applies to the previous stamp state rather than restarting from the original pattern.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("grid must be 2D")

    bg = _get_bg(g)
    stamp_box, canvas_box = _find_frames(g, bg)

    sr0, sc0, sr1, sc1 = stamp_box
    cr0, cc0, cr1, cc1 = canvas_box
    stamp = g[sr0 + 1 : sr1, sc0 + 1 : sc1].copy()
    canvas_shape = (cr1 - cr0 - 1, cc1 - cc0 - 1)

    instruction_rows = _instruction_rows(g, bg, stamp_box, canvas_box)
    instruction_ops: list[tuple[str, str]] = []
    for flip_color, move_color in instruction_rows:
        instruction_ops.append((COLOR_TO_FLIP[flip_color], COLOR_TO_MOVE[move_color]))

    return _run_program(stamp, instruction_ops, canvas_shape, bg)


# --------------- generator (codex) ---------------


def _draw_frame(grid: np.ndarray, top: int, left: int, height: int, width: int, color: int) -> None:
    grid[top, left : left + width] = color
    grid[top + height - 1, left : left + width] = color
    grid[top : top + height, left] = color
    grid[top : top + height, left + width - 1] = color


def _random_stamp(rng: random.Random) -> np.ndarray:
    for _ in range(200):
        h = rng.randint(3, 5)
        w = rng.randint(3, 5)
        colors = rng.sample(STAMP_COLORS, rng.randint(2, len(STAMP_COLORS)))
        stamp = np.full((h, w), BG, dtype=int)

        for r in range(h):
            for c in range(w):
                if rng.random() < 0.68:
                    stamp[r, c] = rng.choice(colors)

        stamp[0, rng.randrange(w)] = rng.choice(colors)
        stamp[h - 1, rng.randrange(w)] = rng.choice(colors)
        stamp[rng.randrange(h), 0] = rng.choice(colors)
        stamp[rng.randrange(h), w - 1] = rng.choice(colors)

        if np.count_nonzero(stamp != BG) < max(5, (h * w) // 2):
            continue
        if len(set(map(int, stamp[stamp != BG].ravel()))) < 2:
            continue

        variants = {
            stamp.tobytes(),
            np.fliplr(stamp).tobytes(),
            np.flipud(stamp).tobytes(),
            np.flipud(np.fliplr(stamp)).tobytes(),
        }
        if len(variants) < 4:
            continue
        return stamp

    raise RuntimeError("failed to generate a non-degenerate stamp")


def _sample_ops(rng: random.Random) -> tuple[list[str], list[str]]:
    steps = rng.randint(3, 5)
    flip_ops = ["h", "v", "n"]
    move_ops = ["r", "d", "rd"]
    while len(flip_ops) < steps:
        flip_ops.append(rng.choice(("h", "v", "n")))
    while len(move_ops) < steps:
        move_ops.append(rng.choice(("r", "d", "rd")))
    rng.shuffle(flip_ops)
    rng.shuffle(move_ops)
    return flip_ops, move_ops


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    stamp = _random_stamp(rng)
    sh, sw = stamp.shape
    flip_ops, move_ops = _sample_ops(rng)

    down_steps = sum(1 for op in move_ops if op in ("d", "rd"))
    right_steps = sum(1 for op in move_ops if op in ("r", "rd"))
    canvas_h = sh * (1 + down_steps) + rng.randint(0, 2)
    canvas_w = sw * (1 + right_steps) + rng.randint(0, 2)

    stamp_frame_h = sh + 2
    stamp_frame_w = sw + 2
    canvas_frame_h = canvas_h + 2
    canvas_frame_w = canvas_w + 2

    instr_sep = rng.randint(1, 3)
    row_gap = rng.randint(0, 1)
    instr_h = len(flip_ops) + (len(flip_ops) - 1) * row_gap
    instr_w = instr_sep + 2
    top_gap = rng.randint(2, 5)

    top_band_h = max(stamp_frame_h, instr_h)
    top_band_w = stamp_frame_w + top_gap + instr_w

    min_h = top_band_h + 1 + canvas_frame_h
    min_w = max(top_band_w, canvas_frame_w)
    if min_h > 30 or min_w > 30:
        return None

    grid_h = rng.randint(min_h, min(30, min_h + 3))
    grid_w = rng.randint(min_w, min(30, min_w + 5))

    extra_h = grid_h - min_h
    top_margin = rng.randint(0, extra_h)
    extra_gap = rng.randint(0, extra_h - top_margin)
    canvas_top = top_margin + top_band_h + 1 + extra_gap

    top_band_left = rng.randint(0, grid_w - top_band_w)
    canvas_left = rng.randint(0, grid_w - canvas_frame_w)

    stamp_top = top_margin + rng.randint(0, top_band_h - stamp_frame_h)
    stamp_left = top_band_left
    instr_top = top_margin + rng.randint(0, top_band_h - instr_h)
    instr_left = top_band_left + stamp_frame_w + top_gap

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    _draw_frame(grid, stamp_top, stamp_left, stamp_frame_h, stamp_frame_w, GREY)
    grid[stamp_top + 1 : stamp_top + 1 + sh, stamp_left + 1 : stamp_left + 1 + sw] = stamp

    for idx, (flip_op, move_op) in enumerate(zip(flip_ops, move_ops)):
        r = instr_top + idx * (row_gap + 1)
        grid[r, instr_left] = FLIP_TO_COLOR[flip_op]
        grid[r, instr_left + instr_sep + 1] = MOVE_TO_COLOR[move_op]

    _draw_frame(grid, canvas_top, canvas_left, canvas_frame_h, canvas_frame_w, GREY)

    expected = _run_program(stamp, list(zip(flip_ops, move_ops)), (canvas_h, canvas_w), BG)
    try:
        actual = main(grid)
    except Exception:
        return None
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
