import random
from collections import Counter

import numpy as np


BG = 0
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 400
GRID_MAX = 30

ROW_PATTERNS = {
    2: [(2,), (1, 1)],
    3: [(3,), (2, 1), (1, 2), (1, 1, 1)],
    4: [(4,), (2, 2), (1, 3), (3, 1), (1, 1, 2), (2, 1, 1)],
}


def _get_bg(grid: np.ndarray) -> int:
    corners = [
        int(grid[0, 0]),
        int(grid[0, -1]),
        int(grid[-1, 0]),
        int(grid[-1, -1]),
    ]
    if len(set(corners)) == 1:
        return int(corners[0])
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(points: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = points.min(axis=0)
    r1, c1 = points.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            pts = []
            while stack:
                rr, cc = stack.pop()
                pts.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(np.array(pts, dtype=int))
    return comps


def _frame_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((int(r0), int(c)))
        cells.add((int(r1), int(c)))
    for r in range(r0 + 1, r1):
        cells.add((int(r), int(c0)))
        cells.add((int(r), int(c1)))
    return cells


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames = []
    for color in sorted(int(c) for c in np.unique(grid) if int(c) != bg):
        for pts in _components(grid == int(color)):
            if int(pts.shape[0]) < 8:
                continue
            r0, c0, r1, c1 = _bbox(pts)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            if set(map(tuple, pts.tolist())) != _frame_cells(r0, c0, r1, c1):
                continue
            frames.append(
                {
                    "color": int(color),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )
    frame_colors = [int(frame["color"]) for frame in frames]
    assert len(frame_colors) == len(set(frame_colors)), "frame colors must be unique"
    frames.sort(key=lambda frame: frame["bbox"])
    assert frames, "no containers found"
    return frames


def _row_segment_is_clear(grid: np.ndarray, bg: int, r: int, c0: int, c1: int) -> bool:
    h, w = grid.shape
    if c0 > 0 and int(grid[r, c0 - 1]) != bg:
        return False
    if c1 < w and int(grid[r, c1]) != bg:
        return False
    for cc in range(c0, c1):
        if r > 0 and int(grid[r - 1, cc]) != bg:
            return False
        if r + 1 < h and int(grid[r + 1, cc]) != bg:
            return False
    return True


def _find_rules(grid: np.ndarray, bg: int) -> dict[int, list[int]]:
    h, w = grid.shape
    rules: dict[int, list[int]] = {}
    for r in range(h):
        c = 0
        while c < w:
            if int(grid[r, c]) == bg:
                c += 1
                continue
            if c + 2 >= w or int(grid[r, c + 1]) != bg or int(grid[r, c + 2]) == bg:
                c += 1
                continue
            end = c + 2
            seq = []
            while end < w and int(grid[r, end]) != bg:
                seq.append(int(grid[r, end]))
                end += 1
            if 3 <= len(seq) <= 5 and len(seq) == len(set(seq)) and _row_segment_is_clear(grid, bg, r, c, end):
                marker = int(grid[r, c])
                if marker in rules:
                    assert rules[marker] == seq, "ambiguous rule rows for one frame color"
                else:
                    rules[marker] = seq
                c = end
                continue
            c += 1
    assert rules, "no rule board found"
    return rules


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder each container's beads using the color order from its matching rule row.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frames = _find_frames(g, bg)
    rules = _find_rules(g, bg)

    out = g.copy()
    for frame in frames:
        color = int(frame["color"])
        assert color in rules, "missing rule row for frame color"
        order = rules[color]

        r0, c0, r1, c1 = map(int, frame["bbox"])
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        beads = [int(v) for v in inner.ravel() if int(v) != bg]
        counts = Counter(beads)
        assert set(counts).issubset(set(order)), "container contains colors outside its rule row"

        sorted_beads = []
        for bead_color in order:
            sorted_beads.extend([int(bead_color)] * counts[int(bead_color)])

        out[r0 + 1 : r1, c0 + 1 : c1] = int(bg)
        idx = 0
        for rr in range(r1 - r0 - 1):
            for cc in range(c1 - c0 - 1):
                if idx >= len(sorted_beads):
                    break
                out[r0 + 1 + rr, c0 + 1 + cc] = int(sorted_beads[idx])
                idx += 1

    return out


def _checker_slots(inner_h: int, inner_w: int) -> list[list[tuple[int, int]]]:
    slots = []
    for row_phase in range(2):
        for col_phase in range(2):
            cells = [
                (int(r), int(c))
                for r in range(row_phase, inner_h, 2)
                for c in range(col_phase, inner_w, 2)
            ]
            slots.append(cells)
    return slots


def _sample_container_specs(rng: random.Random, frame_colors: list[int], bead_palette: list[int]) -> list[dict]:
    specs = []
    for idx, frame_color in enumerate(frame_colors):
        rule_len = rng.randint(3, 5)
        rule = rng.sample(bead_palette, rule_len)

        while True:
            inner_h = rng.randint(4, 7)
            inner_w = rng.randint(5, 9)
            candidates = [cells for cells in _checker_slots(inner_h, inner_w) if len(cells) >= rule_len]
            if candidates:
                slot_cells = rng.choice(candidates)
                break

        total_beads = rng.randint(rule_len, min(len(slot_cells), rule_len + 6))
        counts = [1] * rule_len
        for _ in range(total_beads - rule_len):
            counts[rng.randrange(rule_len)] += 1

        bead_colors = []
        for color, count in zip(rule, counts):
            bead_colors.extend([int(color)] * int(count))
        rng.shuffle(bead_colors)
        bead_positions = rng.sample(slot_cells, total_beads)

        specs.append(
            {
                "id": int(idx),
                "frame_color": int(frame_color),
                "rule": [int(c) for c in rule],
                "inner_h": int(inner_h),
                "inner_w": int(inner_w),
                "outer_h": int(inner_h + 2),
                "outer_w": int(inner_w + 2),
                "beads": bead_colors,
                "input_beads": [
                    (int(pos[0]), int(pos[1]), int(color))
                    for pos, color in zip(bead_positions, bead_colors)
                ],
            }
        )
    return specs


def _choose_row_pattern(rng: random.Random, n: int) -> tuple[int, ...]:
    return rng.choice(ROW_PATTERNS[n])


def _layout_container_block(rng: random.Random, specs: list[dict]) -> tuple[int, int, dict[int, tuple[int, int]]]:
    pool = specs[:]
    rng.shuffle(pool)
    pattern = _choose_row_pattern(rng, len(pool))

    rows = []
    idx = 0
    for row_len in pattern:
        rows.append(pool[idx : idx + row_len])
        idx += row_len

    row_heights = [max(int(spec["outer_h"]) for spec in row) for row in rows]
    row_gaps = [[rng.randint(2, 4) for _ in range(len(row) - 1)] for row in rows]
    row_widths = []
    for row, gaps in zip(rows, row_gaps):
        row_width = sum(int(spec["outer_w"]) for spec in row) + sum(gaps)
        row_widths.append(int(row_width))

    block_w = max(row_widths)
    vertical_gaps = [rng.randint(2, 4) for _ in range(len(rows) - 1)]
    block_h = sum(row_heights) + sum(vertical_gaps)

    positions: dict[int, tuple[int, int]] = {}
    y = 0
    for row_idx, row in enumerate(rows):
        row_h = row_heights[row_idx]
        row_w = row_widths[row_idx]
        x = rng.randint(0, block_w - row_w)
        for item_idx, spec in enumerate(row):
            y_off = rng.randint(0, row_h - int(spec["outer_h"]))
            positions[int(spec["id"])] = (int(y + y_off), int(x))
            x += int(spec["outer_w"])
            if item_idx < len(row) - 1:
                x += int(row_gaps[row_idx][item_idx])
        if row_idx < len(rows) - 1:
            y += int(row_h + vertical_gaps[row_idx])

    return int(block_h), int(block_w), positions


def _render_frame(grid: np.ndarray, r0: int, c0: int, outer_h: int, outer_w: int, color: int) -> None:
    r1 = r0 + outer_h - 1
    c1 = c0 + outer_w - 1
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _expected_output(inp: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = inp.copy()
    for spec in specs:
        r0, c0 = map(int, spec["top_left"])
        inner_h = int(spec["inner_h"])
        inner_w = int(spec["inner_w"])
        counts = Counter(int(c) for c in spec["beads"])

        sorted_beads = []
        for color in spec["rule"]:
            sorted_beads.extend([int(color)] * counts[int(color)])

        out[r0 + 1 : r0 + 1 + inner_h, c0 + 1 : c0 + 1 + inner_w] = int(BG)
        idx = 0
        for rr in range(inner_h):
            for cc in range(inner_w):
                if idx >= len(sorted_beads):
                    break
                out[r0 + 1 + rr, c0 + 1 + cc] = int(sorted_beads[idx])
                idx += 1
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_containers = rng.randint(2, 4)
    colors = list(range(1, 10))
    rng.shuffle(colors)

    frame_colors = colors[:n_containers]
    remaining = colors[n_containers:]
    palette_size = rng.randint(5, len(remaining))
    bead_palette = remaining[:palette_size]

    specs = _sample_container_specs(rng, frame_colors, bead_palette)
    board_specs = specs[:]
    rng.shuffle(board_specs)

    board_left = rng.randint(1, 3)
    board_top = rng.randint(1, 4)
    rule_rows = []
    cur_r = board_top
    for idx, spec in enumerate(board_specs):
        rule_rows.append((int(cur_r), spec))
        cur_r += 1
        if idx < len(board_specs) - 1:
            cur_r += rng.randint(1, 3)
    board_bottom = cur_r - 1
    board_width = 2 + max(len(spec["rule"]) for spec in specs)
    board_right = board_left + board_width - 1

    block_h, block_w, rel_positions = _layout_container_block(rng, specs)
    block_top = rng.randint(1, 4)
    block_left = board_right + 1 + rng.randint(3, 6)

    bottom_margin = rng.randint(1, 4)
    right_margin = rng.randint(1, 4)
    height = max(board_bottom + 1, block_top + block_h) + bottom_margin
    width = block_left + block_w + right_margin

    if height > GRID_MAX or width > GRID_MAX:
        return None

    grid = np.full((height, width), int(BG), dtype=int)

    for row, spec in rule_rows:
        grid[int(row), int(board_left)] = int(spec["frame_color"])
        for offset, color in enumerate(spec["rule"]):
            grid[int(row), int(board_left + 2 + offset)] = int(color)

    for spec in specs:
        rel_r, rel_c = rel_positions[int(spec["id"])]
        top = block_top + rel_r
        left = block_left + rel_c
        spec["top_left"] = (int(top), int(left))

        _render_frame(
            grid,
            int(top),
            int(left),
            int(spec["outer_h"]),
            int(spec["outer_w"]),
            int(spec["frame_color"]),
        )
        for rr, cc, color in spec["input_beads"]:
            grid[int(top + 1 + rr), int(left + 1 + cc)] = int(color)

    expected = _expected_output(grid, specs)

    counts = Counter(map(int, grid.ravel()))
    if counts[int(BG)] <= max((n for c, n in counts.items() if int(c) != int(BG)), default=0):
        return None
    if np.array_equal(grid, expected):
        return None
    try:
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
