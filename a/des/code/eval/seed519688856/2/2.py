import numpy as np


BG = 0
FRAME = 9
PALETTE = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=int)
N4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
RNG = np.random.default_rng()
GEN_RETRIES = 300


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _find_components(grid: np.ndarray, ignore: set[int] | None = None) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    blocked = set() if ignore is None else {int(v) for v in ignore}
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            color = int(g[r, c])
            if color in blocked:
                seen[r, c] = True
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in N4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "area": int(arr.shape[0]),
                    "bbox": (
                        int(arr[:, 0].min()),
                        int(arr[:, 1].min()),
                        int(arr[:, 0].max()),
                        int(arr[:, 1].max()),
                    ),
                }
            )
    return comps


def _extract_patch(grid: np.ndarray, coords: np.ndarray, bg: int) -> np.ndarray:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = grid[coords[:, 0], coords[:, 1]]
    return patch


def _frame_info(grid: np.ndarray, bg: int) -> tuple[tuple[int, int, int, int], list[tuple[int, int]]]:
    frame_comps = [comp for comp in _find_components(grid, ignore={bg}) if int(comp["color"]) == FRAME]
    if not frame_comps:
        raise ValueError("frame not found")
    frame = max(frame_comps, key=lambda comp: (int(comp["area"]), comp["bbox"]))
    r0, c0, r1, c1 = map(int, frame["bbox"])
    line_rows = [r for r in range(r0, r1 + 1) if np.all(grid[r, c0 : c1 + 1] == FRAME)]
    slots: list[tuple[int, int]] = []
    for upper, lower in zip(line_rows, line_rows[1:]):
        if lower - upper > 1:
            slots.append((int(upper + 1), int(lower - 1)))
    if not slots:
        raise ValueError("slot rows not found")
    return (r0, c0, r1, c1), slots


def _marker_map(
    grid: np.ndarray,
    slots: list[tuple[int, int]],
    frame_bbox: tuple[int, int, int, int],
    bg: int,
) -> tuple[dict[int, int], set[tuple[int, int]]]:
    _r0, c0, _r1, c1 = frame_bbox
    mapping: dict[int, int] = {}
    marker_coords: set[tuple[int, int]] = set()
    h, w = grid.shape
    for slot_idx, (sr0, sr1) in enumerate(slots):
        for r in range(sr0, sr1 + 1):
            for c in (c0 - 1, c1 + 1):
                if not (0 <= c < w):
                    continue
                color = int(grid[r, c])
                if color in (bg, FRAME):
                    continue
                mapping[color] = slot_idx
                marker_coords.add((int(r), int(c)))
    if not mapping:
        raise ValueError("markers not found")
    return mapping, marker_coords


def _bar_start(grid: np.ndarray, frame_bbox: tuple[int, int, int, int], bg: int) -> int:
    r0, _c0, r1, c1 = frame_bbox
    h, w = grid.shape
    for c in range(c1 + 1, w):
        if np.all(grid[r0 : r1 + 1, c] == bg):
            return int(c)
    return int(min(w - 1, c1 + 1))


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    frame_bbox, slots = _frame_info(g, bg)
    color_to_slot, marker_coords = _marker_map(g, slots, frame_bbox, bg)

    r0, c0, r1, c1 = frame_bbox
    slot_h = int(slots[0][1] - slots[0][0] + 1)
    bar_c0 = _bar_start(g, frame_bbox, bg)

    out = g.copy()
    for sr0, sr1 in slots:
        out[sr0 : sr1 + 1, c0 + 1 : c1] = bg

    fragments_by_slot: dict[int, list[dict]] = {idx: [] for idx in range(len(slots))}
    for comp in _find_components(g, ignore={bg, FRAME}):
        coords = comp["coords"]
        if int(comp["area"]) == 1 and tuple(map(int, coords[0])) in marker_coords:
            continue
        color = int(comp["color"])
        if color not in color_to_slot:
            continue
        patch = _extract_patch(g, coords, bg)
        if patch.shape[0] != slot_h:
            continue
        out[coords[:, 0], coords[:, 1]] = bg
        fragments_by_slot[color_to_slot[color]].append(
            {
                "color": color,
                "patch": patch,
                "left": int(comp["bbox"][1]),
                "top": int(comp["bbox"][0]),
            }
        )

    for slot_idx, (sr0, sr1) in enumerate(slots):
        row_frags = sorted(fragments_by_slot[slot_idx], key=lambda frag: (frag["left"], frag["top"]))
        cur_c = int(c0 + 1)
        for frag in row_frags:
            patch = frag["patch"]
            ph, pw = map(int, patch.shape)
            for rr in range(ph):
                for cc in range(pw):
                    if int(patch[rr, cc]) != bg:
                        out[sr0 + rr, cur_c + cc] = int(patch[rr, cc])
            cur_c += pw
        out[sr0 : sr1 + 1, bar_c0:] = bg
        if row_frags:
            width = len(row_frags)
            color = int(row_frags[-1]["color"])
            out[sr0 : sr1 + 1, bar_c0 : min(out.shape[1], bar_c0 + width)] = color

    return out


def _draw_frame(grid: np.ndarray, r0: int, c0: int, rows: int, slot_h: int, inner_w: int) -> tuple[tuple[int, int, int, int], list[tuple[int, int]]]:
    frame_h = rows * slot_h + rows + 1
    frame_w = inner_w + 2
    r1 = r0 + frame_h - 1
    c1 = c0 + frame_w - 1
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME
    divider = r0
    for _ in range(rows + 1):
        grid[divider, c0 : c1 + 1] = FRAME
        divider += slot_h + 1
    slots = []
    cur = r0 + 1
    for _ in range(rows):
        slots.append((int(cur), int(cur + slot_h - 1)))
        cur += slot_h + 1
    return (r0, c0, r1, c1), slots


def _random_positive_partition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [int(total)]
    cuts = np.sort(RNG.choice(np.arange(1, total), size=parts - 1, replace=False))
    vals = np.diff(np.concatenate(([0], cuts, [total])))
    return [int(v) for v in vals]


def _interval_choices(height: int, prev: tuple[int, int]) -> list[tuple[int, int]]:
    low, high = prev
    choices: list[tuple[int, int]] = []
    for a in range(height):
        for b in range(a, height):
            if not (b < low or a > high):
                choices.append((int(a), int(b)))
    return choices


def _random_piece_mask(height: int, width: int) -> np.ndarray:
    if width == 1:
        return np.ones((height, 1), dtype=bool)
    intervals: list[tuple[int, int] | None] = [None] * width
    spine = int(RNG.integers(0, width))
    intervals[spine] = (0, height - 1)
    prev = intervals[spine]
    for c in range(spine - 1, -1, -1):
        prev = _interval_choices(height, prev)[int(RNG.integers(0, len(_interval_choices(height, prev))))]
        intervals[c] = prev
    prev = intervals[spine]
    for c in range(spine + 1, width):
        prev = _interval_choices(height, prev)[int(RNG.integers(0, len(_interval_choices(height, prev))))]
        intervals[c] = prev
    mask = np.zeros((height, width), dtype=bool)
    for c, interval in enumerate(intervals):
        a, b = interval
        mask[a : b + 1, c] = True
    return mask


def _can_place_bbox(occ: np.ndarray, r: int, c: int, h: int, w: int) -> bool:
    H, W = occ.shape
    if r < 1 or c < 1 or r + h >= H or c + w >= W:
        return False
    rr0 = max(0, r - 1)
    rr1 = min(H, r + h + 1)
    cc0 = max(0, c - 1)
    cc1 = min(W, c + w + 1)
    return not bool(occ[rr0:rr1, cc0:cc1].any())


def _sample_counts(rows: int, max_count: int) -> list[int] | None:
    for _ in range(100):
        counts = RNG.integers(1, max_count + 1, size=rows)
        if int(counts.sum()) > len(PALETTE):
            continue
        if np.all(counts == 1) and bool(RNG.integers(0, 2)):
            continue
        return [int(v) for v in counts]
    return None


def _gen_attempt() -> np.ndarray | None:
    rows = int(RNG.integers(2, 5))
    slot_h = int(RNG.integers(2, 4))
    inner_w = int(RNG.integers(6, 11))
    max_count = min(3, slot_h * 2, inner_w)
    counts = _sample_counts(rows, max_count)
    if counts is None:
        return None

    frame_h = rows * slot_h + rows + 1
    max_bar = max(counts)
    top_margin = int(RNG.integers(2, 6))
    bottom_margin = int(RNG.integers(2, 6))
    left_margin = int(RNG.integers(5, 10))
    right_margin = int(RNG.integers(6, 11))

    H = frame_h + top_margin + bottom_margin
    W = left_margin + inner_w + 2 + 2 + max_bar + right_margin
    grid = np.full((H, W), BG, dtype=int)

    frame_r0 = top_margin
    frame_c0 = left_margin
    frame_bbox, slots = _draw_frame(grid, frame_r0, frame_c0, rows, slot_h, inner_w)
    r0, c0, r1, c1 = frame_bbox
    bar_c0 = c1 + 1

    colors = RNG.permutation(PALETTE)
    color_idx = 0
    row_pieces: list[list[dict]] = []
    marker_cells: list[tuple[int, int]] = []

    for row_idx, piece_count in enumerate(counts):
        total_width = int(RNG.integers(piece_count, inner_w + 1))
        widths = _random_positive_partition(total_width, piece_count)
        pieces: list[dict] = []
        sr0, sr1 = slots[row_idx]
        marker_options = [(r, c0 - 1) for r in range(sr0, sr1 + 1)] + [(r, c1 + 1) for r in range(sr0, sr1 + 1)]
        choice_idx = RNG.choice(len(marker_options), size=piece_count, replace=False)
        for piece_no, width in enumerate(widths):
            color = int(colors[color_idx])
            color_idx += 1
            mask = _random_piece_mask(slot_h, int(width))
            marker_r, marker_c = marker_options[int(choice_idx[piece_no])]
            grid[marker_r, marker_c] = color
            marker_cells.append((int(marker_r), int(marker_c)))
            pieces.append({"color": color, "mask": mask, "width": int(width)})
        row_pieces.append(pieces)

    occ = np.zeros((H, W), dtype=bool)
    occ[r0 - 1 : r1 + 2, max(0, c0 - 2) : min(W, c1 + 2)] = True
    occ[max(0, r0 - 1) : min(H, r1 + 2), bar_c0 : W] = True
    for mr, mc in marker_cells:
        occ[mr, mc] = True

    used_lefts: set[int] = set()
    for pieces in row_pieces:
        for piece in pieces:
            width = int(piece["width"])
            placed = False
            for _ in range(400):
                pr = int(RNG.integers(1, H - slot_h))
                pc = int(RNG.integers(1, W - width))
                if pc in used_lefts:
                    continue
                if not _can_place_bbox(occ, pr, pc, slot_h, width):
                    continue
                mask = piece["mask"]
                color = int(piece["color"])
                patch = np.full((slot_h, width), BG, dtype=int)
                patch[mask] = color
                grid[pr : pr + slot_h, pc : pc + width][mask] = color
                occ[pr : pr + slot_h, pc : pc + width] = True
                used_lefts.add(pc)
                placed = True
                break
            if not placed:
                return None

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
