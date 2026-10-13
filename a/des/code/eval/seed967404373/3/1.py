from __future__ import annotations

import random

import numpy as np


BG = 0
FRAME_COLOR = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")
GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 80
GRID_HEIGHT_RANGE = (18, 30)
GRID_WIDTH_RANGE = (18, 30)
INNER_MIN = 3
INNER_MAX = 8
OBJECT_PAD = 2


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _classify_half(
    r: int,
    c: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    gap_side: str,
) -> str:
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    ir = r - (r0 + 1)
    ic = c - (c0 + 1)

    if gap_side == "top":
        if inner_h % 2 == 0:
            return "top" if ir < inner_h // 2 else "bottom"
        return "top" if ir <= inner_h // 2 else "bottom"
    if gap_side == "bottom":
        if inner_h % 2 == 0:
            return "top" if ir < inner_h // 2 else "bottom"
        return "top" if ir < inner_h // 2 else "bottom"
    if gap_side == "left":
        if inner_w % 2 == 0:
            return "left" if ic < inner_w // 2 else "right"
        return "left" if ic <= inner_w // 2 else "right"
    if inner_w % 2 == 0:
        return "left" if ic < inner_w // 2 else "right"
    return "left" if ic < inner_w // 2 else "right"


def _extract_frame(
    g: np.ndarray,
    pts: list[tuple[int, int]],
    color: int,
) -> dict[str, object] | None:
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 < 3 or c1 - c0 < 3:
        return None

    for r, c in pts:
        if r not in (r0, r1) and c not in (c0, c1):
            return None

    border = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }
    if len(pts) != len(border) - 1:
        return None

    missing = [(r, c) for r, c in border if int(g[r, c]) != color]
    if len(missing) != 1:
        return None
    gap_r, gap_c = missing[0]
    if int(g[gap_r, gap_c]) != BG:
        return None

    gap_sides = []
    if gap_r == r0:
        gap_sides.append("top")
    if gap_r == r1:
        gap_sides.append("bottom")
    if gap_c == c0:
        gap_sides.append("left")
    if gap_c == c1:
        gap_sides.append("right")
    if len(gap_sides) != 1:
        return None
    gap_side = gap_sides[0]

    inside = g[r0 + 1 : r1, c0 + 1 : c1]
    if np.any(inside == color):
        return None

    seed_rel = np.argwhere((inside != BG) & (inside != color))
    if seed_rel.shape[0] != 2:
        return None

    seed_info: dict[str, tuple[tuple[int, int], int]] = {}
    for rel_r, rel_c in seed_rel:
        sr = int(r0 + 1 + rel_r)
        sc = int(c0 + 1 + rel_c)
        half = _classify_half(sr, sc, r0, c0, r1, c1, gap_side)
        if half in seed_info:
            return None
        seed_info[half] = ((sr, sc), int(g[sr, sc]))

    if gap_side in ("top", "bottom"):
        needed = {"top", "bottom"}
    else:
        needed = {"left", "right"}
    if set(seed_info) != needed:
        return None

    colors = [seed_info[key][1] for key in seed_info]
    if len(set(colors)) != 2:
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "frame_color": color,
        "gap": (gap_r, gap_c),
        "gap_side": gap_side,
        "seed_colors": {key: value[1] for key, value in seed_info.items()},
    }


def _find_frames(g: np.ndarray) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for color in np.unique(g):
        color = int(color)
        if color == BG:
            continue
        for comp in _components(g == color):
            frame = _extract_frame(g, comp, color)
            if frame is not None:
                frames.append(frame)
    frames.sort(key=lambda item: tuple(item["bbox"]))  # type: ignore[arg-type]
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each rectangular fenced region by splitting its interior according to the single border gap.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for frame in _find_frames(g):
        r0, c0, r1, c1 = frame["bbox"]  # type: ignore[misc]
        gap_side = frame["gap_side"]  # type: ignore[assignment]
        seed_colors = frame["seed_colors"]  # type: ignore[assignment]
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if int(out[r, c]) != BG:
                    continue
                half = _classify_half(r, c, r0, c0, r1, c1, gap_side)
                out[r, c] = seed_colors[half]
    return out


def _pick_with_parity(lo: int, hi: int, parity: int) -> int:
    choices = [value for value in range(lo, hi + 1) if value % 2 == parity]
    if not choices:
        choices = list(range(lo, hi + 1))
    return random.choice(choices)


def _interior_size(side: str) -> tuple[int, int]:
    if side in ("top", "bottom"):
        inner_h = _pick_with_parity(INNER_MIN, INNER_MAX, random.randint(0, 1))
        inner_w = random.randint(INNER_MIN, INNER_MAX)
    else:
        inner_h = random.randint(INNER_MIN, INNER_MAX)
        inner_w = _pick_with_parity(INNER_MIN, INNER_MAX, random.randint(0, 1))
    return inner_h, inner_w


def _overlaps(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    pad: int,
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _half_cells(r0: int, c0: int, r1: int, c1: int, gap_side: str, half: str) -> list[tuple[int, int]]:
    cells = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if _classify_half(r, c, r0, c0, r1, c1, gap_side) == half:
                cells.append((r, c))
    return cells


def _gap_cell(r0: int, c0: int, r1: int, c1: int, side: str) -> tuple[int, int]:
    if side == "top":
        return r0, random.randint(c0 + 1, c1 - 1)
    if side == "bottom":
        return r1, random.randint(c0 + 1, c1 - 1)
    if side == "left":
        return random.randint(r0 + 1, r1 - 1), c0
    return random.randint(r0 + 1, r1 - 1), c1


def _paint_frame(
    inp: np.ndarray,
    out: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    gap: tuple[int, int],
    side: str,
    colors: dict[str, int],
) -> None:
    gap_r, gap_c = gap
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if r in (r0, r1) or c in (c0, c1):
                if (r, c) != (gap_r, gap_c):
                    inp[r, c] = FRAME_COLOR
                    out[r, c] = FRAME_COLOR

    if side in ("top", "bottom"):
        halves = ("top", "bottom")
    else:
        halves = ("left", "right")

    used: set[tuple[int, int]] = set()
    for half in halves:
        cell = random.choice(_half_cells(r0, c0, r1, c1, side, half))
        used.add(cell)
        inp[cell] = colors[half]
        out[cell] = colors[half]

    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if (r, c) in used:
                continue
            half = _classify_half(r, c, r0, c0, r1, c1, side)
            out[r, c] = colors[half]


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    area = h * w
    max_frames = 4 if area >= 650 else 3 if area >= 450 else 2
    n_frames = random.randint(1, max_frames)

    inp = np.full((h, w), BG, dtype=int)
    expected = np.full((h, w), BG, dtype=int)
    placed: list[tuple[int, int, int, int]] = []

    sides = list(SIDES)
    random.shuffle(sides)
    while len(sides) < n_frames:
        extra = list(SIDES)
        random.shuffle(extra)
        sides.extend(extra)

    for side in sides[:n_frames]:
        inner_h, inner_w = _interior_size(side)
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        placed_here = False

        for _ in range(PLACE_ATTEMPTS):
            r0 = random.randint(1, h - outer_h - 1)
            c0 = random.randint(1, w - outer_w - 1)
            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1
            bbox = (r0, c0, r1, c1)
            if any(_overlaps(bbox, other, OBJECT_PAD) for other in placed):
                continue

            gap = _gap_cell(r0, c0, r1, c1, side)
            if side in ("top", "bottom"):
                labels = ("top", "bottom")
            else:
                labels = ("left", "right")
            seed_a, seed_b = random.sample(SEED_COLORS, 2)
            colors = {labels[0]: seed_a, labels[1]: seed_b}

            _paint_frame(inp, expected, r0, c0, r1, c1, gap, side, colors)
            placed.append(bbox)
            placed_here = True
            break

        if not placed_here:
            return None

    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _gen_attempt()
        if built is None:
            continue
        inp, expected = built
        if np.array_equal(main(inp), expected):
            return inp
    raise RuntimeError("failed to generate a valid input")
