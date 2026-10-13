import numpy as np
from collections import deque


BG = 0
SEP = 5
FRAGMENT_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 400
_RNG = np.random.default_rng()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].astype(bool)


def _is_connected(mask: np.ndarray) -> bool:
    comps = _components(mask)
    return len(comps) == 1


def _is_irregular(mask: np.ndarray) -> bool:
    return mask.shape[0] >= 2 and mask.shape[1] >= 2 and int(mask.sum()) < mask.shape[0] * mask.shape[1]


def _mask_cells(mask: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, cell)) for cell in np.argwhere(mask)}


def _completion_shapes(mask_a: np.ndarray, mask_b: np.ndarray) -> list[tuple[int, int]]:
    cells_a = _mask_cells(mask_a)
    area = int(mask_a.sum() + mask_b.sum())
    if not cells_a or area == 0:
        return []

    ha, wa = mask_a.shape
    shapes: set[tuple[int, int]] = set()
    for k in range(4):
        rot_b = np.rot90(mask_b, k)
        cells_b = _mask_cells(rot_b)
        hb, wb = rot_b.shape
        for dr in range(-hb, ha + 1):
            for dc in range(-wb, wa + 1):
                shifted_b = {(r + dr, c + dc) for r, c in cells_b}
                if cells_a & shifted_b:
                    continue
                union = cells_a | shifted_b
                min_r = min(r for r, _ in union)
                max_r = max(r for r, _ in union)
                min_c = min(c for _, c in union)
                max_c = max(c for _, c in union)
                rect_h = max_r - min_r + 1
                rect_w = max_c - min_c + 1
                if rect_h * rect_w == area:
                    shapes.add((int(rect_h), int(rect_w)))
    return sorted(shapes)


def _find_rectangle_completion(mask_a: np.ndarray, mask_b: np.ndarray) -> tuple[int, int] | None:
    shapes = _completion_shapes(mask_a, mask_b)
    return shapes[0] if shapes else None


def _find_legend_row(grid: np.ndarray) -> int:
    for r in range(grid.shape[0] - 1, -1, -1):
        row = grid[r]
        if np.count_nonzero(row == SEP) >= 2 and np.count_nonzero((row != BG) & (row != SEP)) >= 3:
            return int(r)
    return int(grid.shape[0] - 1)


def _parse_legend(row: np.ndarray) -> dict[frozenset[int], int]:
    mapping: dict[frozenset[int], int] = {}
    w = len(row)
    i = 0
    while i < w:
        if int(row[i]) == BG:
            i += 1
            continue

        if (
            i + 4 < w
            and int(row[i]) not in (BG, SEP)
            and int(row[i + 1]) == SEP
            and int(row[i + 2]) not in (BG, SEP)
            and int(row[i + 3]) == SEP
            and int(row[i + 4]) not in (BG, SEP)
        ):
            a = int(row[i])
            b = int(row[i + 2])
            result = int(row[i + 4])
            mapping[frozenset((a, b))] = result
            i += 5
            continue

        if (
            i + 3 < w
            and int(row[i]) not in (BG, SEP)
            and int(row[i + 1]) == SEP
            and int(row[i + 2]) not in (BG, SEP)
            and int(row[i + 3]) not in (BG, SEP)
        ):
            a = int(row[i])
            b = int(row[i + 2])
            result = int(row[i + 3])
            mapping[frozenset((a, b))] = result
            i += 4
            continue

        if (
            i + 2 < w
            and int(row[i]) not in (BG, SEP)
            and int(row[i + 1]) not in (BG, SEP)
            and int(row[i + 2]) not in (BG, SEP)
        ):
            a = int(row[i])
            b = int(row[i + 1])
            result = int(row[i + 2])
            mapping[frozenset((a, b))] = result
            i += 3
            continue

        i += 1
    return mapping


def _extract_fragments(grid: np.ndarray, legend_row: int) -> dict[int, dict[str, object]]:
    work = grid[:legend_row, :]
    mask = (work != BG) & (work != SEP)
    fragments: dict[int, dict[str, object]] = {}
    for coords in _components(mask):
        colors = {int(work[r, c]) for r, c in coords}
        if len(colors) != 1:
            continue
        color = colors.pop()
        r0, c0, r1, c1 = _bbox(coords)
        local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in coords:
            local[int(r - r0), int(c - c0)] = True
        fragments[color] = {
            "color": color,
            "coords": [(int(r), int(c)) for r, c in coords],
            "top_left": (int(r0), int(c0)),
            "mask": local,
        }
    return fragments


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("grid must be 2D")

    legend_row = _find_legend_row(g)
    legend = _parse_legend(g[legend_row])
    fragments = _extract_fragments(g, legend_row)
    out = g.copy()
    used_colors: set[int] = set()

    for pair_colors, result_color in legend.items():
        if len(pair_colors) != 2:
            continue
        color_a, color_b = sorted(pair_colors)
        if color_a in used_colors or color_b in used_colors:
            continue
        if color_a not in fragments or color_b not in fragments:
            continue

        frag_a = fragments[color_a]
        frag_b = fragments[color_b]
        rect_shape = _find_rectangle_completion(
            np.asarray(frag_a["mask"], dtype=bool),
            np.asarray(frag_b["mask"], dtype=bool),
        )
        if rect_shape is None:
            continue

        top_left = min(frag_a["top_left"], frag_b["top_left"])
        rr, cc = top_left
        rect_h, rect_w = rect_shape
        if rr + rect_h > legend_row or cc + rect_w > g.shape[1]:
            continue

        for r, c in frag_a["coords"]:
            out[r, c] = BG
        for r, c in frag_b["coords"]:
            out[r, c] = BG

        out[rr : rr + rect_h, cc : cc + rect_w] = int(result_color)
        used_colors.add(color_a)
        used_colors.add(color_b)

    return out


def _random_connected_mask(height: int, width: int, n_cells: int, rng: np.random.Generator) -> np.ndarray | None:
    if not (1 <= n_cells <= height * width):
        return None
    start = (int(rng.integers(height)), int(rng.integers(width)))
    cells = {start}
    while len(cells) < n_cells:
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        cell = list(frontier)[int(rng.integers(len(frontier)))]
        cells.add(cell)

    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _make_partition(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(250):
        height = int(rng.integers(3, 6))
        width = int(rng.integers(3, 7))
        area = height * width
        low = max(3, area // 3)
        high = min(area - 3, (2 * area) // 3)
        if low > high:
            continue
        n_a = int(rng.integers(low, high + 1))
        full_a = _random_connected_mask(height, width, n_a, rng)
        if full_a is None:
            continue
        full_b = ~full_a
        if not _is_connected(full_b):
            continue

        mask_a = _tight_mask(full_a)
        mask_b = _tight_mask(full_b)
        if int(mask_a.sum()) < 3 or int(mask_b.sum()) < 3:
            continue
        if not (_is_irregular(mask_a) and _is_irregular(mask_b)):
            continue
        if len(_completion_shapes(mask_a, mask_b)) != 1:
            continue
        return mask_a, mask_b
    return None


def _make_extra_fragment(rng: np.random.Generator) -> np.ndarray | None:
    for _ in range(120):
        height = int(rng.integers(2, 5))
        width = int(rng.integers(2, 5))
        max_cells = min(height * width - 1, 7)
        if max_cells < 3:
            continue
        n_cells = int(rng.integers(3, max_cells + 1))
        mask = _random_connected_mask(height, width, n_cells, rng)
        if mask is None:
            continue
        mask = _tight_mask(mask)
        if int(mask.sum()) < 3 or not _is_irregular(mask):
            continue
        return mask
    return None


def _can_place(blocked: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> bool:
    h, w = blocked.shape
    if top < 0 or left < 0 or top + height > h or left + width > w:
        return False
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    return not blocked[r0:r1, c0:c1].any()


def _mark(blocked: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> None:
    h, w = blocked.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    blocked[r0:r1, c0:c1] = True


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for r, c in np.argwhere(mask):
        rr = top + int(r)
        cc = left + int(c)
        grid[rr, cc] = int(color)
        coords.append((rr, cc))
    return coords


def _choose_result_color(
    used_fragment_colors: set[int],
    color_a: int,
    color_b: int,
    rng: np.random.Generator,
) -> int:
    preferred = [c for c in FRAGMENT_COLORS if c not in used_fragment_colors and c not in (color_a, color_b)]
    if preferred:
        return int(preferred[int(rng.integers(len(preferred)))])
    fallback = [c for c in FRAGMENT_COLORS if c not in (color_a, color_b)]
    return int(fallback[int(rng.integers(len(fallback)))])


def _paint_legend(grid: np.ndarray, entries: list[tuple[int, int, int]], rng: np.random.Generator) -> None:
    row = np.zeros(grid.shape[1], dtype=int)
    shuffled = entries[:]
    rng.shuffle(shuffled)

    parts: list[list[int]] = []
    for color_a, color_b, result in shuffled:
        if rng.random() < 0.5:
            color_a, color_b = color_b, color_a
        parts.append([int(color_a), SEP, int(color_b), SEP, int(result)])

    gaps = [int(rng.integers(1, 3)) for _ in range(max(0, len(parts) - 1))]
    total_width = sum(len(part) for part in parts) + sum(gaps)
    start = int(rng.integers(0, grid.shape[1] - total_width + 1))
    col = start
    for i, part in enumerate(parts):
        row[col : col + len(part)] = part
        col += len(part)
        if i < len(gaps):
            col += gaps[i]
    grid[-1, :] = row


def _gen_attempt() -> np.ndarray | None:
    rng = _RNG
    pair_count = int(rng.choice([2, 3]))
    add_extra = bool(rng.random() < 0.35)
    need_colors = 2 * pair_count + (1 if add_extra else 0)
    if need_colors > len(FRAGMENT_COLORS):
        return None

    fragment_colors = list(rng.choice(FRAGMENT_COLORS, size=need_colors, replace=False))
    extra_color = fragment_colors.pop() if add_extra else None

    pair_specs: list[dict[str, object]] = []
    color_idx = 0
    used_fragment_colors = set(fragment_colors)
    for _ in range(pair_count):
        partition = _make_partition(rng)
        if partition is None:
            return None
        mask_a, mask_b = partition
        color_a = int(fragment_colors[color_idx])
        color_b = int(fragment_colors[color_idx + 1])
        color_idx += 2

        piece_a = np.rot90(mask_a, int(rng.integers(0, 4)))
        piece_b = np.rot90(mask_b, int(rng.integers(0, 4)))
        if rng.random() < 0.5:
            anchor_mask, anchor_color = piece_a, color_a
            remote_mask, remote_color = piece_b, color_b
        else:
            anchor_mask, anchor_color = piece_b, color_b
            remote_mask, remote_color = piece_a, color_a

        shapes = _completion_shapes(anchor_mask, remote_mask)
        if len(shapes) != 1:
            return None

        pair_specs.append(
            {
                "anchor_mask": anchor_mask,
                "anchor_color": anchor_color,
                "remote_mask": remote_mask,
                "remote_color": remote_color,
                "out_shape": shapes[0],
                "result_color": _choose_result_color(used_fragment_colors, color_a, color_b, rng),
            }
        )

    extra_mask = _make_extra_fragment(rng) if add_extra else None
    if add_extra and extra_mask is None:
        return None

    legend_width = 5 * pair_count + (pair_count - 1) * 2
    max_piece_h = 1
    max_piece_w = legend_width
    total_area = 0
    for spec in pair_specs:
        out_h, out_w = spec["out_shape"]
        max_piece_h = max(max_piece_h, int(out_h), int(spec["remote_mask"].shape[0]))
        max_piece_w = max(max_piece_w, int(out_w), int(spec["remote_mask"].shape[1]))
        total_area += int(out_h * out_w) + int(np.asarray(spec["remote_mask"]).sum())
    if extra_mask is not None:
        max_piece_h = max(max_piece_h, int(extra_mask.shape[0]))
        max_piece_w = max(max_piece_w, int(extra_mask.shape[1]))
        total_area += int(extra_mask.sum())

    min_w = max(legend_width + 1, max_piece_w + 6, 14)
    grid_w = int(rng.integers(min_w, min_w + 9))

    min_h = max(max_piece_h + 4, total_area // max(1, grid_w - 2) + 6, 10)
    grid_h = int(rng.integers(min_h, min_h + 7))
    play_h = grid_h - 1
    blocked = np.zeros((play_h, grid_w), dtype=bool)
    grid = np.zeros((grid_h, grid_w), dtype=int)

    pair_order = sorted(pair_specs, key=lambda spec: int(spec["out_shape"][0] * spec["out_shape"][1]), reverse=True)
    for spec in pair_order:
        out_h, out_w = spec["out_shape"]
        remote_h, remote_w = spec["remote_mask"].shape
        placed = False
        for _ in range(250):
            anchor_top = int(rng.integers(0, play_h - out_h + 1))
            anchor_left = int(rng.integers(0, grid_w - out_w + 1))
            if not _can_place(blocked, anchor_top, anchor_left, out_h, out_w, pad=1):
                continue

            trial = blocked.copy()
            _mark(trial, anchor_top, anchor_left, out_h, out_w, pad=1)
            for _ in range(250):
                remote_top = int(rng.integers(0, play_h - remote_h + 1))
                remote_left = int(rng.integers(0, grid_w - remote_w + 1))
                if remote_top < anchor_top or (remote_top == anchor_top and remote_left <= anchor_left):
                    continue
                if not _can_place(trial, remote_top, remote_left, remote_h, remote_w, pad=1):
                    continue

                spec["anchor_pos"] = (anchor_top, anchor_left)
                spec["remote_pos"] = (remote_top, remote_left)
                blocked = trial
                _mark(blocked, remote_top, remote_left, remote_h, remote_w, pad=1)
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    if extra_mask is not None and extra_color is not None:
        extra_h, extra_w = extra_mask.shape
        extra_placed = False
        for _ in range(200):
            top = int(rng.integers(0, play_h - extra_h + 1))
            left = int(rng.integers(0, grid_w - extra_w + 1))
            if _can_place(blocked, top, left, extra_h, extra_w, pad=1):
                _mark(blocked, top, left, extra_h, extra_w, pad=1)
                extra_coords = _paint_mask(grid, extra_mask, top, left, int(extra_color))
                extra_placed = True
                break
        if not extra_placed:
            return None
    else:
        extra_coords = []

    legend_entries: list[tuple[int, int, int]] = []
    expected = np.zeros_like(grid)
    for spec in pair_order:
        anchor_top, anchor_left = spec["anchor_pos"]
        remote_top, remote_left = spec["remote_pos"]
        anchor_coords = _paint_mask(grid, spec["anchor_mask"], anchor_top, anchor_left, int(spec["anchor_color"]))
        remote_coords = _paint_mask(grid, spec["remote_mask"], remote_top, remote_left, int(spec["remote_color"]))
        spec["anchor_coords"] = anchor_coords
        spec["remote_coords"] = remote_coords
        legend_entries.append((int(spec["anchor_color"]), int(spec["remote_color"]), int(spec["result_color"])))

        for r, c in anchor_coords:
            expected[r, c] = BG
        for r, c in remote_coords:
            expected[r, c] = BG
        out_h, out_w = spec["out_shape"]
        expected[anchor_top : anchor_top + out_h, anchor_left : anchor_left + out_w] = int(spec["result_color"])

    for r, c in extra_coords:
        expected[r, c] = grid[r, c]

    _paint_legend(grid, legend_entries, rng)
    expected[-1, :] = grid[-1, :]

    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
