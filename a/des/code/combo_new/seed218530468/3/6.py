import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
RNG = np.random.default_rng()
DIRS = {
    "up": ((-1, 0), (0, 1)),
    "right": ((0, 1), (1, 0)),
    "down": ((1, 0), (0, 1)),
    "left": ((0, -1), (1, 0)),
}
MAX_ATTEMPTS = 300


def _inside(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _cell(grid: np.ndarray, r: int, c: int) -> int | None:
    if _inside(grid.shape, r, c):
        return int(grid[r, c])
    return None


def _arrow_parts(
    junction: tuple[int, int], direction: tuple[int, int], length: int
) -> tuple[list[tuple[int, int]], tuple[int, int], list[tuple[int, int]], list[tuple[int, int]]]:
    jr, jc = junction
    dr, dc = direction
    pr, pc = dc, -dr
    shaft = [(jr - k * dr, jc - k * dc) for k in range(length)]
    tip = (jr + dr, jc + dc)
    sides = [(jr + pr, jc + pc), (jr - pr, jc - pc)]
    footprint = shaft + [tip] + sides
    return shaft, tip, sides, footprint


def _is_arrow_candidate(
    grid: np.ndarray, junction: tuple[int, int], direction: tuple[int, int], length: int
) -> dict[str, object] | None:
    jr, jc = junction
    color = _cell(grid, jr, jc)
    if color in (None, BG):
        return None

    shaft, tip, sides, footprint = _arrow_parts(junction, direction, length)
    for r, c in footprint:
        if _cell(grid, r, c) != color:
            return None

    dr, dc = direction
    pr, pc = dc, -dr
    tail = (jr - length * dr, jc - length * dc)
    if _cell(grid, tail[0], tail[1]) == color:
        return None

    # Require a one-pixel-thick shaft and a clean V head.
    for r, c in shaft[1:]:
        if _cell(grid, r + pr, c + pc) == color or _cell(grid, r - pr, c - pc) == color:
            return None
    if _cell(grid, tip[0] + pr, tip[1] + pc) == color or _cell(grid, tip[0] - pr, tip[1] - pc) == color:
        return None
    if _cell(grid, sides[0][0] + pr, sides[0][1] + pc) == color:
        return None
    if _cell(grid, sides[1][0] - pr, sides[1][1] - pc) == color:
        return None

    return {
        "color": color,
        "junction": junction,
        "direction": direction,
        "length": length,
        "shaft": shaft,
        "tip": tip,
        "head": [tip] + sides,
        "footprint": footprint,
    }


def _detect_arrows(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    candidates: dict[frozenset[tuple[int, int]], dict[str, object]] = {}

    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) == BG:
                continue
            for direction, _perp in DIRS.values():
                for length in range(2, 6):
                    cand = _is_arrow_candidate(grid, (r, c), direction, length)
                    if cand is None:
                        continue
                    key = frozenset(cand["footprint"])
                    prev = candidates.get(key)
                    if prev is None or int(cand["length"]) > int(prev["length"]):
                        candidates[key] = cand

    arrows = sorted(
        candidates.values(),
        key=lambda a: (
            min(pos[0] for pos in a["footprint"]),
            min(pos[1] for pos in a["footprint"]),
            int(a["length"]),
        ),
    )

    used: set[tuple[int, int]] = set()
    out: list[dict[str, object]] = []
    for arrow in arrows:
        footprint = set(arrow["footprint"])
        if footprint & used:
            continue
        used |= footprint
        out.append(arrow)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reverse-absorb each arrow's forward color strip into its shaft.

    Input structure:
    - Black background.
    - Several monochrome arrow objects, each built from a one-pixel-thick straight shaft of length 2-5 and a 3-pixel V head.
    - In front of each arrow tip, along the pointing direction, lies a colored row or column segment that may mix colors.

    Rule:
    - Detect every arrow and measure its shaft length L, excluding the 3 head pixels.
    - Starting from the tip and moving forward, read up to L consecutive non-black pixels from the front segment.
    - Reverse that absorbed sequence, then write it onto the shaft from the cell nearest the head toward the tail; any remaining shaft cells keep the arrow color.
    - The arrow head itself stays unchanged, and the absorbed source pixels turn black.

    Key insight:
    - The arrow shape encodes both direction and the absorption distance: the shaft length is reused as the number of forward pixels to pull back, and the pulled sequence is reversed before filling the shaft.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for arrow in _detect_arrows(g):
        dr, dc = arrow["direction"]
        tip_r, tip_c = arrow["tip"]
        length = int(arrow["length"])
        shaft = list(arrow["shaft"])

        absorbed_positions: list[tuple[int, int]] = []
        absorbed_colors: list[int] = []
        for step in range(1, length + 1):
            rr = tip_r + dr * step
            cc = tip_c + dc * step
            if not _inside(g.shape, rr, cc) or int(g[rr, cc]) == BG:
                break
            absorbed_positions.append((rr, cc))
            absorbed_colors.append(int(g[rr, cc]))

        for rr, cc in absorbed_positions:
            out[rr, cc] = BG

        reversed_colors = absorbed_colors[::-1]
        for idx, color in enumerate(reversed_colors):
            rr, cc = shaft[idx]
            out[rr, cc] = color

    return out


def _scene_offsets(direction_name: str, length: int, seq_len: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    direction, _perp = DIRS[direction_name]
    shaft, tip, sides, arrow_cells = _arrow_parts((0, 0), direction, length)
    dr, dc = direction
    seq_cells = [(tip[0] + dr * step, tip[1] + dc * step) for step in range(1, seq_len + 1)]
    return arrow_cells, seq_cells


def _make_sequence_colors(arrow_colors: list[int], seq_len: int) -> list[int]:
    available = [color for color in PALETTE if color not in arrow_colors]
    seq = RNG.choice(available, size=seq_len, replace=True).astype(int).tolist()
    if seq_len >= 2 and len(set(seq)) == 1 and len(available) >= 2:
        alt_choices = [color for color in available if color != seq[0]]
        seq[int(RNG.integers(seq_len))] = int(RNG.choice(alt_choices))
    return seq


def _candidate_positions(
    blocked: np.ndarray, offsets: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    h, w = blocked.shape
    min_r = min(r for r, _ in offsets)
    max_r = max(r for r, _ in offsets)
    min_c = min(c for _, c in offsets)
    max_c = max(c for _, c in offsets)

    candidates: list[tuple[int, int]] = []
    for jr in range(-min_r, h - max_r):
        for jc in range(-min_c, w - max_c):
            cells = [(jr + dr, jc + dc) for dr, dc in offsets]
            if any(blocked[r, c] for r, c in cells):
                continue
            box_r0 = max(0, jr + min_r - 1)
            box_r1 = min(h, jr + max_r + 2)
            box_c0 = max(0, jc + min_c - 1)
            box_c1 = min(w, jc + max_c + 2)
            if blocked[box_r0:box_r1, box_c0:box_c1].any():
                continue
            candidates.append((jr, jc))
    return candidates


def _paint_scene(
    grid: np.ndarray,
    blocked: np.ndarray,
    direction_name: str,
    length: int,
    seq_len: int,
    arrow_color: int,
    seq_colors: list[int],
) -> bool:
    arrow_offsets, seq_offsets = _scene_offsets(direction_name, length, seq_len)
    all_offsets = arrow_offsets + seq_offsets
    positions = _candidate_positions(blocked, all_offsets)
    if not positions:
        return False

    jr, jc = positions[int(RNG.integers(len(positions)))]
    for dr, dc in arrow_offsets:
        grid[jr + dr, jc + dc] = arrow_color
    for (dr, dc), color in zip(seq_offsets, seq_colors):
        grid[jr + dr, jc + dc] = color

    min_r = min(dr for dr, _ in all_offsets)
    max_r = max(dr for dr, _ in all_offsets)
    min_c = min(dc for _, dc in all_offsets)
    max_c = max(dc for _, dc in all_offsets)
    box_r0 = max(0, jr + min_r - 1)
    box_r1 = min(grid.shape[0], jr + max_r + 2)
    box_c0 = max(0, jc + min_c - 1)
    box_c1 = min(grid.shape[1], jc + max_c + 2)
    blocked[box_r0:box_r1, box_c0:box_c1] = True
    return True


def _gen_attempt() -> np.ndarray | None:
    h = int(RNG.integers(12, 29))
    w = int(RNG.integers(12, 29))
    grid = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    max_arrows = min(4, max(1, (h * w) // 120 + 1))
    num_arrows = int(RNG.integers(1, max_arrows + 1))
    arrow_colors = RNG.choice(PALETTE, size=num_arrows, replace=False).astype(int).tolist()

    short_flags = [True] * (num_arrows // 2) + [False] * (num_arrows - num_arrows // 2)
    RNG.shuffle(short_flags)
    if num_arrows == 1 and RNG.random() < 0.5:
        short_flags[0] = not short_flags[0]

    specs: list[tuple[str, int, int, int, list[int]]] = []
    for arrow_color, short_flag in zip(arrow_colors, short_flags):
        direction_name = str(RNG.choice(list(DIRS.keys())))
        length = int(RNG.integers(2, 6))
        if short_flag:
            seq_len = int(RNG.integers(1, length))
        else:
            seq_len = int(RNG.integers(length, length + 4))
        seq_colors = _make_sequence_colors(arrow_colors, seq_len)
        specs.append((direction_name, length, seq_len, arrow_color, seq_colors))

    RNG.shuffle(specs)
    for direction_name, length, seq_len, arrow_color, seq_colors in specs:
        if not _paint_scene(grid, blocked, direction_name, length, seq_len, arrow_color, seq_colors):
            return None

    arrows = _detect_arrows(grid)
    if len(arrows) != num_arrows:
        return None
    if not any(int(grid[r, c]) != BG for r in range(h) for c in range(w)):
        return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after 300 attempts")
