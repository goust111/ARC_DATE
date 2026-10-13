from __future__ import annotations

import random

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _find_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    seen = np.zeros((H, W), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(H):
        for c in range(W):
            color = int(g[r, c])
            if color == int(bg) or bool(seen[r, c]):
                continue

            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < H and 0 <= nc < W and not bool(seen[nr, nc]) and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))

            coords_arr = np.array(sorted(coords), dtype=int)
            r0 = int(coords_arr[:, 0].min())
            c0 = int(coords_arr[:, 1].min())
            r1 = int(coords_arr[:, 0].max())
            c1 = int(coords_arr[:, 1].max())
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            mask[coords_arr[:, 0] - r0, coords_arr[:, 1] - c0] = True
            comps.append(
                {
                    "color": int(color),
                    "coords": coords_arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "mask": mask,
                    "area": int(coords_arr.shape[0]),
                }
            )

    comps.sort(key=lambda comp: tuple(int(v) for v in comp["bbox"]))
    return comps


def _candidate_pairs(comps: list[dict[str, object]]) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for i, comp_a in enumerate(comps):
        a_r0, a_c0, a_r1, a_c1 = map(int, comp_a["bbox"])
        a_h = int(a_r1 - a_r0 + 1)
        a_w = int(a_c1 - a_c0 + 1)
        for j in range(i + 1, len(comps)):
            comp_b = comps[j]
            b_r0, b_c0, b_r1, b_c1 = map(int, comp_b["bbox"])
            if (a_h, a_w) != (int(b_r1 - b_r0 + 1), int(b_c1 - b_c0 + 1)):
                continue

            if a_r0 == b_r0 and a_r1 == b_r1:
                left_idx, right_idx = (i, j) if a_c0 < b_c0 else (j, i)
                left = comps[left_idx]
                right = comps[right_idx]
                l_r0, l_c0, l_r1, l_c1 = map(int, left["bbox"])
                r_r0, r_c0, r_r1, r_c1 = map(int, right["bbox"])
                if l_c1 < r_c0:
                    candidates.append(
                        {
                            "kind": "vertical",
                            "keep_idx": int(left_idx),
                            "move_idx": int(right_idx),
                            "axis2": int(l_c0 + r_c0 + a_w - 1),
                            "gap": int(r_c0 - l_c1 - 1),
                            "sort_key": (int(r_c0 - l_c1 - 1), 0, l_r0, l_c0, r_c0),
                        }
                    )

            if a_c0 == b_c0 and a_c1 == b_c1:
                top_idx, bottom_idx = (i, j) if a_r0 < b_r0 else (j, i)
                top = comps[top_idx]
                bottom = comps[bottom_idx]
                t_r0, t_c0, t_r1, t_c1 = map(int, top["bbox"])
                b_r0b, b_c0b, b_r1b, b_c1b = map(int, bottom["bbox"])
                if t_r1 < b_r0b:
                    candidates.append(
                        {
                            "kind": "horizontal",
                            "keep_idx": int(top_idx),
                            "move_idx": int(bottom_idx),
                            "axis2": int(t_r0 + b_r0b + a_h - 1),
                            "gap": int(b_r0b - t_r1 - 1),
                            "sort_key": (int(b_r0b - t_r1 - 1), 1, t_r0, t_c0, b_r0b),
                        }
                    )

    candidates.sort(key=lambda cand: cand["sort_key"])

    used: set[int] = set()
    pairs: list[dict[str, object]] = []
    for cand in candidates:
        keep_idx = int(cand["keep_idx"])
        move_idx = int(cand["move_idx"])
        if keep_idx in used or move_idx in used:
            continue
        used.add(keep_idx)
        used.add(move_idx)
        pairs.append(cand)
    return pairs


def _overlap_color(keep: dict[str, object], move: dict[str, object]) -> int:
    keep_area = int(keep["area"])
    move_area = int(move["area"])
    if keep_area < move_area:
        return int(keep["color"])
    if move_area < keep_area:
        return int(move["color"])
    return int(move["color"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find aligned object pairs and fold the right/bottom object onto the left/top one.

    Pair detection is geometric: paired objects have equal-size bounding boxes and
    share either the same row span or the same column span, which defines the
    hidden vertical or horizontal fold axis between them.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    comps = _find_components(g, bg)
    if not comps:
        return g.copy()

    out = g.copy()
    for pair in _candidate_pairs(comps):
        keep = comps[int(pair["keep_idx"])]
        move = comps[int(pair["move_idx"])]
        keep_cells = {tuple(map(int, coord)) for coord in keep["coords"]}
        merge_color = _overlap_color(keep, move)
        axis2 = int(pair["axis2"])

        for rr, cc in move["coords"]:
            out[int(rr), int(cc)] = int(bg)

        if pair["kind"] == "vertical":
            for rr, cc in move["coords"]:
                dst = (int(rr), int(axis2 - int(cc)))
                out[dst] = int(merge_color) if dst in keep_cells else int(move["color"])
        else:
            for rr, cc in move["coords"]:
                dst = (int(axis2 - int(rr)), int(cc))
                out[dst] = int(merge_color) if dst in keep_cells else int(move["color"])

    return out


def _manhattan_path(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = start
    tr, tc = end
    out = [(int(r), int(c))]
    if random.randrange(2) == 0:
        while c != tc:
            c += 1 if tc > c else -1
            out.append((int(r), int(c)))
        while r != tr:
            r += 1 if tr > r else -1
            out.append((int(r), int(c)))
    else:
        while r != tr:
            r += 1 if tr > r else -1
            out.append((int(r), int(c)))
        while c != tc:
            c += 1 if tc > c else -1
            out.append((int(r), int(c)))
    return out


def _build_scaffold(h: int, w: int) -> set[tuple[int, int]] | None:
    for _ in range(80):
        center = (random.randrange(h), random.randrange(w))
        anchors = [
            (0, random.randrange(w)),
            (h - 1, random.randrange(w)),
            (random.randrange(h), 0),
            (random.randrange(h), w - 1),
        ]
        cells: set[tuple[int, int]] = set()
        for anchor in anchors:
            cells.update(_manhattan_path(anchor, center))
        if len(cells) < h * w:
            return cells
    return None


def _grow_cells(
    seed: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    count: int,
    h: int,
    w: int,
) -> set[tuple[int, int]] | None:
    chosen: set[tuple[int, int]] = set()
    occupied = set(seed) | set(blocked)
    for _ in range(count):
        frontier: set[tuple[int, int]] = set()
        for r, c in seed | chosen:
            for dr, dc in DIR4:
                nr = int(r + dr)
                nc = int(c + dc)
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in occupied:
                    frontier.add((int(nr), int(nc)))
        if not frontier:
            return None
        pick = random.choice(sorted(frontier))
        chosen.add(pick)
        occupied.add(pick)
    return chosen


def _mask_from_cells(cells: set[tuple[int, int]], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[int(r), int(c)] = True
    return mask


def _pair_masks(h: int, w: int) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(120):
        scaffold = _build_scaffold(h, w)
        if scaffold is None:
            continue
        free = h * w - len(scaffold)
        options = [
            (a_count, b_count)
            for a_count in range(0, min(4, free) + 1)
            for b_count in range(0, min(4, free - a_count) + 1)
            if a_count != b_count and a_count + b_count >= 1
        ]
        if not options:
            continue

        a_count, b_count = random.choice(options)
        extra_a = _grow_cells(scaffold, set(), a_count, h, w)
        if extra_a is None:
            continue
        extra_b = _grow_cells(scaffold, extra_a, b_count, h, w)
        if extra_b is None:
            continue

        keep_mask = _mask_from_cells(scaffold | extra_a, h, w)
        move_mask = _mask_from_cells(scaffold | extra_b, h, w)
        if int(keep_mask.sum()) == int(move_mask.sum()):
            continue
        return keep_mask, move_mask
    return None


def _single_mask(h: int, w: int) -> np.ndarray | None:
    for _ in range(80):
        scaffold = _build_scaffold(h, w)
        if scaffold is None:
            continue
        free = h * w - len(scaffold)
        extra_count = random.randint(0, min(4, free))
        extra = _grow_cells(scaffold, set(), extra_count, h, w)
        if extra is None:
            continue
        return _mask_from_cells(scaffold | extra, h, w)
    return None


def _can_place_box(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, margin: int = 1) -> bool:
    H, W = occupied.shape
    r1 = int(r0 + h - 1)
    c1 = int(c0 + w - 1)
    rr0 = max(0, int(r0 - margin))
    cc0 = max(0, int(c0 - margin))
    rr1 = min(H, int(r1 + margin + 1))
    cc1 = min(W, int(c1 + margin + 1))
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _mark_box(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    occupied[int(r0) : int(r0 + h), int(c0) : int(c0 + w)] = True


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = mask.shape
    patch = grid[int(r0) : int(r0 + h), int(c0) : int(c0 + w)]
    patch[mask] = int(color)


def _merge_pair_patch(
    keep_mask: np.ndarray,
    move_mask: np.ndarray,
    keep_color: int,
    move_color: int,
) -> np.ndarray:
    h, w = keep_mask.shape
    patch = np.full((h, w), -1, dtype=int)
    overlap_color = keep_color if int(keep_mask.sum()) < int(move_mask.sum()) else move_color
    patch[keep_mask] = int(keep_color)
    patch[move_mask & ~keep_mask] = int(move_color)
    patch[keep_mask & move_mask] = int(overlap_color)
    return patch


def _try_place_pair(
    occupied: np.ndarray,
    H: int,
    W: int,
    dims: tuple[int, int],
    orientation: str,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    h, w = dims
    for _ in range(80):
        gap = random.randint(1, 4)
        if orientation == "vertical":
            total_w = 2 * w + gap
            if total_w > W or h > H:
                return None
            r0 = random.randint(0, H - h)
            left_c0 = random.randint(0, W - total_w)
            right_c0 = left_c0 + w + gap
            if not _can_place_box(occupied, r0, left_c0, h, w):
                continue
            if not _can_place_box(occupied, r0, right_c0, h, w):
                continue
            return (int(r0), int(left_c0)), (int(r0), int(right_c0))

        total_h = 2 * h + gap
        if total_h > H or w > W:
            return None
        c0 = random.randint(0, W - w)
        top_r0 = random.randint(0, H - total_h)
        bottom_r0 = top_r0 + h + gap
        if not _can_place_box(occupied, top_r0, c0, h, w):
            continue
        if not _can_place_box(occupied, bottom_r0, c0, h, w):
            continue
        return (int(top_r0), int(c0)), (int(bottom_r0), int(c0))
    return None


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(14, 22)
    W = random.randint(14, 22)
    bg = random.randrange(10)

    num_pairs = random.randint(1, 3)
    num_extras = random.randint(0, 2)
    total_objects = 2 * num_pairs + num_extras
    if total_objects > 9:
        return None

    palette = [color for color in range(10) if color != bg]
    random.shuffle(palette)
    colors = palette[:total_objects]

    occupied = np.zeros((H, W), dtype=bool)
    used_dims: set[tuple[int, int]] = set()
    input_grid = np.full((H, W), bg, dtype=int)
    expected = np.full((H, W), bg, dtype=int)
    color_idx = 0

    pair_specs: list[dict[str, object]] = []
    for _ in range(num_pairs):
        placed = False
        for _ in range(120):
            dims = (random.randint(3, 6), random.randint(3, 6))
            if dims in used_dims:
                continue
            orientation = random.choice(["vertical", "horizontal"])
            masks = _pair_masks(*dims)
            if masks is None:
                continue
            keep_mask, move_target = masks
            positions = _try_place_pair(occupied, H, W, dims, orientation)
            if positions is None:
                continue

            keep_pos, move_pos = positions
            keep_color = int(colors[color_idx])
            move_color = int(colors[color_idx + 1])
            color_idx += 2

            if orientation == "vertical":
                move_input = np.fliplr(move_target)
            else:
                move_input = np.flipud(move_target)

            _paint_mask(input_grid, keep_mask, keep_pos[0], keep_pos[1], keep_color)
            _paint_mask(input_grid, move_input, move_pos[0], move_pos[1], move_color)
            pair_specs.append(
                {
                    "keep_mask": keep_mask,
                    "move_target": move_target,
                    "keep_pos": keep_pos,
                    "keep_color": keep_color,
                    "move_color": move_color,
                }
            )
            _mark_box(occupied, keep_pos[0], keep_pos[1], dims[0], dims[1])
            _mark_box(occupied, move_pos[0], move_pos[1], dims[0], dims[1])
            used_dims.add(dims)
            placed = True
            break
        if not placed:
            return None

    for _ in range(num_extras):
        placed = False
        for _ in range(120):
            dims = (random.randint(2, 5), random.randint(2, 5))
            if dims in used_dims:
                continue
            mask = _single_mask(*dims)
            if mask is None:
                continue
            r0 = random.randint(0, H - dims[0])
            c0 = random.randint(0, W - dims[1])
            if not _can_place_box(occupied, r0, c0, dims[0], dims[1]):
                continue
            color = int(colors[color_idx])
            color_idx += 1
            _paint_mask(input_grid, mask, r0, c0, color)
            _paint_mask(expected, mask, r0, c0, color)
            _mark_box(occupied, r0, c0, dims[0], dims[1])
            used_dims.add(dims)
            placed = True
            break
        if not placed:
            return None

    for spec in pair_specs:
        patch = _merge_pair_patch(
            np.asarray(spec["keep_mask"], dtype=bool),
            np.asarray(spec["move_target"], dtype=bool),
            int(spec["keep_color"]),
            int(spec["move_color"]),
        )
        r0, c0 = spec["keep_pos"]
        h, w = patch.shape
        target = expected[int(r0) : int(r0 + h), int(c0) : int(c0 + w)]
        target[patch != -1] = patch[patch != -1]

    if np.array_equal(input_grid, expected):
        return None
    counts = np.bincount(input_grid.ravel(), minlength=10)
    if int(counts[bg]) <= int(max(counts[i] for i in range(10) if i != bg)):
        return None
    if not np.array_equal(main(input_grid), expected):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after repeated retries")
