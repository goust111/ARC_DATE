import random
from collections import Counter

import numpy as np


BG = 0
GREY = 5
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS = ("up", "down", "left", "right")
ARROW_MASKS = {
    "up": np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    "down": np.array([[0, 1, 0], [0, 1, 0], [1, 1, 1]], dtype=bool),
    "left": np.array([[1, 0, 0], [1, 1, 1], [1, 0, 0]], dtype=bool),
    "right": np.array([[0, 0, 1], [1, 1, 1], [0, 0, 1]], dtype=bool),
}
GAP_OFFSETS = {
    "up": lambda size: (0, size // 2),
    "down": lambda size: (size - 1, size // 2),
    "left": lambda size: (size // 2, 0),
    "right": lambda size: (size // 2, size - 1),
}

LEGEND_BAND_H = 3
LEGEND_ROW_GAP = 1
LEGEND_H = 4 * LEGEND_BAND_H + 3 * LEGEND_ROW_GAP
GEN_RETRIES = 300


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _color_components(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _border_coords(r0: int, c0: int, size: int) -> list[tuple[int, int]]:
    coords = []
    r1 = r0 + size - 1
    c1 = c0 + size - 1
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
    for r in range(r0 + 1, r1):
        coords.append((r, c1))
    if r1 > r0:
        for c in range(c1, c0 - 1, -1):
            coords.append((r1, c))
    if c1 > c0:
        for r in range(r1 - 1, r0, -1):
            coords.append((r, c0))
    return coords


def _border_is_color(grid: np.ndarray, r0: int, c0: int, size: int, color: int) -> bool:
    return all(int(grid[r, c]) == color for r, c in _border_coords(r0, c0, size))


def _full_frame_color(grid: np.ndarray, r0: int, c0: int, size: int, bg: int) -> int | None:
    colors = {int(grid[r, c]) for r, c in _border_coords(r0, c0, size)}
    if len(colors) != 1:
        return None
    color = colors.pop()
    if color in (bg, GREY):
        return None
    return color


def _outer_frame_info(grid: np.ndarray, r0: int, c0: int, size: int, bg: int) -> tuple[str, int] | None:
    if size < 3 or size % 2 == 0:
        return None
    border = _border_coords(r0, c0, size)
    gap_cells = {
        direction: (r0 + dr, c0 + dc)
        for direction, fn in GAP_OFFSETS.items()
        for dr, dc in [fn(size)]
    }
    bg_cells = [(r, c) for r, c in border if int(grid[r, c]) == bg]
    if len(bg_cells) != 1 or tuple(bg_cells[0]) not in gap_cells.values():
        return None
    gap_dir = next(direction for direction, cell in gap_cells.items() if cell == tuple(bg_cells[0]))
    colors = {int(grid[r, c]) for r, c in border if (r, c) != gap_cells[gap_dir]}
    if len(colors) != 1:
        return None
    color = colors.pop()
    if color in (bg, GREY):
        return None
    return gap_dir, color


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _read_ring(grid: np.ndarray, r0: int, c0: int, size: int, bg: int) -> dict | None:
    outer = _outer_frame_info(grid, r0, c0, size, bg)
    if outer is None:
        return None
    gap_dir, outer_color = outer
    frames = [{"r0": r0, "c0": c0, "size": size, "color": outer_color}]

    cur_r, cur_c, cur_size = r0, c0, size
    while True:
        inner_r = cur_r + 1
        inner_c = cur_c + 1
        inner_size = cur_size - 2
        if inner_size <= 0 or not _border_is_color(grid, inner_r, inner_c, inner_size, bg):
            return None

        next_r = cur_r + 2
        next_c = cur_c + 2
        next_size = cur_size - 4
        if next_size < 3:
            inner = grid[inner_r : inner_r + inner_size, inner_c : inner_c + inner_size]
            if inner.size and not np.all(inner == bg):
                return None
            break

        next_color = _full_frame_color(grid, next_r, next_c, next_size, bg)
        if next_color is None:
            inner = grid[next_r : next_r + next_size, next_c : next_c + next_size]
            if inner.size and not np.all(inner == bg):
                return None
            break

        frames.append({"r0": next_r, "c0": next_c, "size": next_size, "color": next_color})
        cur_r, cur_c, cur_size = next_r, next_c, next_size

    return {
        "bbox": (r0, c0, r0 + size - 1, c0 + size - 1),
        "gap_dir": gap_dir,
        "layers": len(frames),
        "frames": frames,
    }


def _find_rings(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    rings: list[dict] = []
    max_size = min(h, w)
    if max_size % 2 == 0:
        max_size -= 1
    for size in range(max_size, 2, -2):
        for r0 in range(h - size + 1):
            for c0 in range(w - size + 1):
                bbox = (r0, c0, r0 + size - 1, c0 + size - 1)
                if any(_boxes_overlap(bbox, ring["bbox"]) for ring in rings):
                    continue
                ring = _read_ring(grid, r0, c0, size, bg)
                if ring is not None:
                    rings.append(ring)
    rings.sort(key=lambda ring: (ring["bbox"][0], ring["bbox"][1]))
    return rings


def _match_arrow(mask: np.ndarray) -> str | None:
    for direction, template in ARROW_MASKS.items():
        if mask.shape == template.shape and np.array_equal(mask, template):
            return direction
    return None


def _read_sequence(grid: np.ndarray, row: int, left: int, right: int, bg: int) -> tuple[int, ...]:
    seq_left = []
    c = left - 1
    while c >= 0 and int(grid[row, c]) not in (bg, GREY):
        seq_left.append(int(grid[row, c]))
        c -= 1

    seq_right = []
    c = right + 1
    while c < grid.shape[1] and int(grid[row, c]) not in (bg, GREY):
        seq_right.append(int(grid[row, c]))
        c += 1

    assert bool(seq_left) ^ bool(seq_right), "legend row must have colors on exactly one side"
    return tuple(seq_left if seq_left else seq_right)


def _legend_mapping(grid: np.ndarray, bg: int) -> dict[str, tuple[int, ...]]:
    mapping: dict[str, tuple[int, ...]] = {}
    for comp in _color_components(grid, GREY):
        if len(comp) != 5:
            continue
        r0, c0, r1, c1 = _bbox(comp)
        if (r1 - r0 + 1, c1 - c0 + 1) != (3, 3):
            continue
        mask = np.zeros((3, 3), dtype=bool)
        for r, c in comp:
            mask[int(r) - r0, int(c) - c0] = True
        direction = _match_arrow(mask)
        if direction is None:
            continue
        seq = _read_sequence(grid, r0 + 1, c0, c1, bg)
        assert seq, "every legend arrow must have at least one color block"
        mapping[direction] = seq

    assert set(mapping) == set(DIRS), "legend must contain one arrow for each direction"
    return mapping


def _paint_frame(grid: np.ndarray, r0: int, c0: int, size: int, color: int, gap_dir: str | None = None) -> None:
    for r, c in _border_coords(r0, c0, size):
        grid[r, c] = color
    if gap_dir is not None:
        dr, dc = GAP_OFFSETS[gap_dir](size)
        grid[r0 + dr, c0 + dc] = BG


def _paint_ring(grid: np.ndarray, r0: int, c0: int, layers: int, colors: tuple[int, ...], gap_dir: str) -> None:
    outer_size = 4 * layers - 1
    for idx in range(layers):
        offset = 2 * idx
        size = outer_size - 4 * idx
        _paint_frame(grid, r0 + offset, c0 + offset, size, int(colors[idx]), gap_dir if idx == 0 else None)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor nested square rings by using the outer gap direction as a lookup into a corner legend.

    Input structure:
    - The background is black.
    - The grid contains several nested square-ring objects made of concentric 1-pixel square frames, with one background moat between neighboring layers.
    - Each object's outermost frame has a single 1-pixel gap at the midpoint of exactly one side: up, down, left, or right.
    - A corner legend contains four grey arrow markers, one per direction, and each arrow is followed by an ordered row of color blocks.

    Rule:
    - Detect every nested ring object and identify the direction of the outermost gap.
    - Find the legend row whose grey arrow points in that same direction, then read its ordered color sequence starting from the block adjacent to the arrow.
    - Repaint the object's layers from outer to inner with the corresponding colors from that sequence, while preserving the existing gap and leaving the legend unchanged.

    Key insight:
    - The outer gap is not damage to repair; it is the index key that chooses one legend row, and the legend row's visible order maps directly onto the ring layers from outside to inside.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    mapping = _legend_mapping(g, bg)
    rings = _find_rings(g, bg)

    out = g.copy()
    for ring in rings:
        seq = mapping[ring["gap_dir"]]
        assert len(seq) >= ring["layers"], "legend row must be long enough for every ring layer"
        for idx, frame in enumerate(ring["frames"]):
            _paint_frame(
                out,
                frame["r0"],
                frame["c0"],
                frame["size"],
                int(seq[idx]),
                ring["gap_dir"] if idx == 0 else None,
            )
    return out


def _random_sequences(rng: random.Random, seq_len: int) -> dict[str, tuple[int, ...]]:
    while True:
        rows = [tuple(rng.sample(COLORS, seq_len)) for _ in range(4)]
        if len(set(rows)) == 4:
            return {direction: rows[idx] for idx, direction in enumerate(DIRS)}


def _legend_bbox(height: int, width: int, seq_len: int, corner: str) -> tuple[int, int, int, int]:
    legend_w = 3 + seq_len
    top = 0 if "top" in corner else height - LEGEND_H
    left = 0 if "left" in corner else width - legend_w
    return top, left, top + LEGEND_H - 1, left + legend_w - 1


def _place_legend(
    grid: np.ndarray,
    mapping: dict[str, tuple[int, ...]],
    corner: str,
    order: list[str],
) -> tuple[int, int, int, int]:
    h, w = grid.shape
    seq_len = len(next(iter(mapping.values())))
    top, left, bottom, right = _legend_bbox(h, w, seq_len, corner)
    left_corner = "left" in corner

    for row_idx, direction in enumerate(order):
        base_r = top + row_idx * (LEGEND_BAND_H + LEGEND_ROW_GAP)
        arrow_left = left if left_corner else left + seq_len
        mask = ARROW_MASKS[direction]
        grid[base_r : base_r + 3, arrow_left : arrow_left + 3][mask] = GREY

        seq = mapping[direction]
        if left_corner:
            start = arrow_left + 3
            for idx, color in enumerate(seq):
                grid[base_r + 1, start + idx] = int(color)
        else:
            for idx, color in enumerate(reversed(seq)):
                grid[base_r + 1, left + idx] = int(color)

    return top, left, bottom, right


def _choose_layers(rng: random.Random, seq_len: int, n_objects: int) -> list[int]:
    layers = [seq_len]
    if n_objects >= 2:
        layers.append(2 if seq_len > 2 else seq_len)
    while len(layers) < n_objects:
        layers.append(rng.randint(2, seq_len))
    rng.shuffle(layers)
    return layers


def _input_colors_for_object(rng: random.Random, target_seq: tuple[int, ...], layers: int) -> tuple[int, ...]:
    chosen = []
    for idx in range(layers):
        candidates = [color for color in COLORS if color != target_seq[idx] and color not in chosen]
        chosen.append(rng.choice(candidates))
    return tuple(chosen)


def _candidate_positions(occupied: np.ndarray, size: int) -> list[tuple[int, int]]:
    h, w = occupied.shape
    positions = []
    for r0 in range(1, h - size):
        for c0 in range(1, w - size):
            if occupied[r0 - 1 : r0 + size + 1, c0 - 1 : c0 + size + 1].any():
                continue
            positions.append((r0, c0))
    return positions


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    seq_len = rng.choice([2, 3, 4])
    if seq_len == 4:
        n_objects = rng.choice([2, 3])
        size_min = 27
    elif seq_len == 3:
        n_objects = rng.choice([2, 3, 4])
        size_min = 24
    else:
        n_objects = rng.choice([2, 3, 4])
        size_min = 20

    h = rng.randint(size_min, 30)
    w = rng.randint(size_min, 30)
    grid = np.full((h, w), BG, dtype=int)

    mapping = _random_sequences(rng, seq_len)
    legend_corner = rng.choice(["top-left", "top-right", "bottom-left", "bottom-right"])
    legend_order = rng.sample(list(DIRS), len(DIRS))
    ltop, lleft, lbottom, lright = _place_legend(grid, mapping, legend_corner, legend_order)

    occupied = np.zeros((h, w), dtype=bool)
    occupied[max(0, ltop - 1) : min(h, lbottom + 2), max(0, lleft - 1) : min(w, lright + 2)] = True

    directions = rng.sample(list(DIRS), n_objects)
    layers_list = _choose_layers(rng, seq_len, n_objects)
    objects = []
    for direction, layers in zip(directions, layers_list):
        size = 4 * layers - 1
        objects.append(
            {
                "direction": direction,
                "layers": layers,
                "size": size,
                "target_colors": mapping[direction][:layers],
                "input_colors": _input_colors_for_object(rng, mapping[direction], layers),
            }
        )

    objects.sort(key=lambda obj: obj["size"], reverse=True)
    for obj in objects:
        positions = _candidate_positions(occupied, obj["size"])
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        obj["r0"] = r0
        obj["c0"] = c0
        occupied[r0 - 1 : r0 + obj["size"] + 1, c0 - 1 : c0 + obj["size"] + 1] = True

    expected = grid.copy()
    for obj in objects:
        _paint_ring(grid, obj["r0"], obj["c0"], obj["layers"], obj["input_colors"], obj["direction"])
        _paint_ring(expected, obj["r0"], obj["c0"], obj["layers"], obj["target_colors"], obj["direction"])

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    counts = Counter(obj["direction"] for obj in objects)
    assert all(count == 1 for count in counts.values())
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate input after {GEN_RETRIES} attempts")
