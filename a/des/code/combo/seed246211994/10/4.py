from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 8
INPUT_MIN = 16
INPUT_MAX = 30
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _mask_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _count_holes(patch: np.ndarray, color: int) -> int:
    non_color = patch != int(color)
    holes = 0
    for comp in _mask_components(non_color):
        if any(
            r == 0 or c == 0 or r == patch.shape[0] - 1 or c == patch.shape[1] - 1
            for r, c in comp
        ):
            continue
        holes += 1
    return int(holes)


def _color_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == int(bg) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if bool(seen[nr, nc]) or int(g[nr, nc]) != int(color):
                        continue
                    seen[nr, nc] = True
                    stack.append((int(nr), int(nc)))
            r0, c0, r1, c1 = _bbox(coords)
            patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
            comps.append(
                {
                    "color": int(color),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "patch": patch,
                    "coords": coords,
                }
            )
    return comps


def _is_square_object(patch: np.ndarray, color: int) -> bool:
    h, w = map(int, patch.shape)
    if h != w or h < 3:
        return False
    return bool(
        np.all(patch[0, :] == int(color))
        and np.all(patch[-1, :] == int(color))
        and np.all(patch[:, 0] == int(color))
        and np.all(patch[:, -1] == int(color))
    )


def _extract_objects(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    candidates: list[dict] = []
    for comp in _color_components(grid, BG):
        color = int(comp["color"])
        patch = np.asarray(comp["patch"], dtype=int)
        if not _is_square_object(patch, color):
            continue
        size = int(patch.shape[0])
        holes = _count_holes(patch, color)
        info = {
            "color": int(color),
            "bbox": comp["bbox"],
            "patch": patch,
            "size": int(size),
            "holes": int(holes),
        }
        candidates.append(info)

    sizes = sorted({int(obj["size"]) for obj in candidates}, reverse=True)
    assert len(sizes) >= 2
    frame_size = int(sizes[0])
    sprite_size = int(sizes[1])
    frames = [obj for obj in candidates if int(obj["size"]) == int(frame_size)]
    sprites = [obj for obj in candidates if int(obj["size"]) == int(sprite_size)]
    assert frames and sprites and len(frames) == len(sprites)
    return frames, sprites


def _render_output(
    frames: list[dict],
    sprites_by_holes: dict[int, list[dict]],
    n_rows: int,
    n_cols: int,
    frame_size: int,
) -> np.ndarray:
    out = np.full((int(n_rows * frame_size), int(n_cols * frame_size)), BG, dtype=int)
    for frame in frames:
        rr = int(frame["row_index"]) * int(frame_size)
        cc = int(frame["col_index"]) * int(frame_size)
        color = int(frame["color"])
        out[rr, cc : cc + frame_size] = color
        out[rr + frame_size - 1, cc : cc + frame_size] = color
        out[rr : rr + frame_size, cc] = color
        out[rr : rr + frame_size, cc + frame_size - 1] = color

        bucket = sprites_by_holes.get(int(frame["holes"]), [])
        assert bucket
        sprite = bucket.pop(0)
        content = np.asarray(sprite["patch"], dtype=int).copy()
        sprite_color = int(sprite["color"])
        content[content != int(sprite_color)] = int(color)

        inner = int(frame_size - 2)
        sh, sw = map(int, content.shape)
        assert sh <= inner and sw <= inner
        sr = rr + 1 + max(0, (inner - sh) // 2)
        sc = cc + 1 + max(0, (inner - sw) // 2)
        out[sr : sr + sh, sc : sc + sw] = content
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair square sprites with square frames by 4-connected hole count, then
    place each sprite inside the matching frame.
    """
    g = np.asarray(grid, dtype=int)
    frames, sprites = _extract_objects(g)

    row_vals = sorted({int(obj["bbox"][0]) for obj in frames})
    col_vals = sorted({int(obj["bbox"][1]) for obj in frames})
    assert len(row_vals) * len(col_vals) == len(frames)

    row_index = {int(v): i for i, v in enumerate(row_vals)}
    col_index = {int(v): i for i, v in enumerate(col_vals)}
    for frame in frames:
        r0, c0, _r1, _c1 = frame["bbox"]
        frame["row_index"] = int(row_index[int(r0)])
        frame["col_index"] = int(col_index[int(c0)])

    frames.sort(key=lambda obj: (int(obj["row_index"]), int(obj["col_index"])))

    sprites_by_holes: dict[int, list[dict]] = defaultdict(list)
    for sprite in sprites:
        sprites_by_holes[int(sprite["holes"])].append(sprite)
    for hole_count, bucket in sprites_by_holes.items():
        if len(bucket) > 1:
            bucket.sort(key=lambda obj: (int(obj["bbox"][0]), int(obj["bbox"][1])))

    return _render_output(
        frames=frames,
        sprites_by_holes=sprites_by_holes,
        n_rows=len(row_vals),
        n_cols=len(col_vals),
        frame_size=int(frames[0]["size"]),
    )


def _factor_pairs(n: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for r in range(1, n + 1):
        if n % r == 0:
            pairs.append((int(r), int(n // r)))
    return pairs


def _random_composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [int(total)]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    vals: list[int] = []
    prev = 0
    for cut in cuts + [int(total)]:
        vals.append(int(cut - prev))
        prev = int(cut)
    return vals


def _partition_bar_positions(inner: int, parts: int) -> list[int] | None:
    if parts == 1:
        return []
    free = int(inner - (parts - 1))
    if free < parts:
        return None
    spans = _random_composition(free, parts)
    bars: list[int] = []
    pos = int(spans[0])
    for span in spans[1:]:
        bars.append(int(pos))
        pos += 1 + int(span)
    return bars


def _frame_patterns(inner: int) -> dict[int, list[tuple[int, int]]]:
    max_parts = int((inner + 1) // 2)
    out: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r_parts in range(1, max_parts + 1):
        for c_parts in range(1, max_parts + 1):
            out[int(r_parts * c_parts)].append((int(r_parts), int(c_parts)))
    return dict(out)


def _make_frame(frame_size: int, hole_count: int, color: int) -> np.ndarray | None:
    inner = int(frame_size - 2)
    patterns = _frame_patterns(inner).get(int(hole_count), [])
    if not patterns:
        return None
    r_parts, c_parts = random.choice(patterns)
    hbars = _partition_bar_positions(inner, r_parts)
    vbars = _partition_bar_positions(inner, c_parts)
    if hbars is None or vbars is None:
        return None

    frame = np.full((int(frame_size), int(frame_size)), BG, dtype=int)
    frame[0, :] = color
    frame[-1, :] = color
    frame[:, 0] = color
    frame[:, -1] = color
    for rr in hbars:
        frame[1 + int(rr), 1:-1] = color
    for cc in vbars:
        frame[1:-1, 1 + int(cc)] = color

    if _count_holes(frame, color) != int(hole_count):
        return None
    return frame


def _seed_positions(size: int, count: int) -> list[tuple[int, int]] | None:
    cells = [(r, c) for r in range(1, size - 1) for c in range(1, size - 1)]
    for _ in range(60):
        random.shuffle(cells)
        chosen: list[tuple[int, int]] = []
        for r, c in cells:
            if all(abs(r - rr) + abs(c - cc) > 1 for rr, cc in chosen):
                chosen.append((int(r), int(c)))
                if len(chosen) == count:
                    return chosen
    return None


def _make_sprite(size: int, hole_count: int, color: int) -> np.ndarray | None:
    sprite = np.full((int(size), int(size)), int(color), dtype=int)
    if hole_count == 0:
        return sprite

    seeds = _seed_positions(size, hole_count)
    if seeds is None:
        return None

    comps: list[set[tuple[int, int]]] = [{seed} for seed in seeds]
    occupied = set(seeds)
    for r, c in occupied:
        sprite[r, c] = BG

    inner_cells = (int(size) - 2) * (int(size) - 2)
    growth_budget = random.randint(hole_count, max(hole_count, inner_cells // 2))
    for _ in range(growth_budget):
        order = list(range(hole_count))
        random.shuffle(order)
        changed = False
        for idx in order:
            frontier: list[tuple[int, int]] = []
            for r, c in comps[idx]:
                for nr, nc in _neighbors4(r, c, size, size):
                    if not (1 <= nr < size - 1 and 1 <= nc < size - 1):
                        continue
                    if (nr, nc) in occupied:
                        continue
                    ok = True
                    for other_idx, other in enumerate(comps):
                        if other_idx == idx:
                            continue
                        if any((ar, ac) in other for ar, ac in _neighbors4(nr, nc, size, size)):
                            ok = False
                            break
                    if ok:
                        frontier.append((int(nr), int(nc)))
            if not frontier or random.random() < 0.35:
                continue
            nr, nc = random.choice(frontier)
            comps[idx].add((int(nr), int(nc)))
            occupied.add((int(nr), int(nc)))
            sprite[nr, nc] = BG
            changed = True
        if not changed and random.random() < 0.5:
            break

    if _count_holes(sprite, color) != int(hole_count):
        return None
    return sprite


def _max_sprite_holes(size: int) -> int:
    inner = int(size - 2)
    return int((inner * inner + 1) // 2)


def _paint(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = map(int, patch.shape)
    canvas[r0 : r0 + h, c0 : c0 + w] = patch


def _rect_free(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> bool:
    oh, ow = map(int, occupied.shape)
    if r0 < 0 or c0 < 0 or r0 + h > oh or c0 + w > ow:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(oh, r0 + h + pad)
    cc1 = min(ow, c0 + w + pad)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _reserve_rect(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> None:
    oh, ow = map(int, occupied.shape)
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(oh, r0 + h + pad)
    cc1 = min(ow, c0 + w + pad)
    occupied[rr0:rr1, cc0:cc1] = True


def _expected_output(
    frames: list[dict],
    sprites: list[dict],
    rows: int,
    cols: int,
    frame_size: int,
) -> np.ndarray:
    sprites_by_holes: dict[int, list[dict]] = defaultdict(list)
    for sprite in sprites:
        sprites_by_holes[int(sprite["holes"])].append(
            {"color": int(sprite["color"]), "patch": np.asarray(sprite["patch"], dtype=int).copy()}
        )
    return _render_output(
        frames=[dict(frame) for frame in frames],
        sprites_by_holes=sprites_by_holes,
        n_rows=int(rows),
        n_cols=int(cols),
        frame_size=int(frame_size),
    )


def _gen_attempt() -> np.ndarray | None:
    n_pairs = random.randint(2, 4)
    layout_choices = _factor_pairs(n_pairs)
    random.shuffle(layout_choices)

    frame_size = random.randint(6, 9)
    if n_pairs >= 4:
        frame_size = random.randint(7, 9)
    sprite_size = int(frame_size - 2)

    possible_counts = [
        count
        for count in sorted(_frame_patterns(sprite_size).keys())
        if int(count) <= _max_sprite_holes(sprite_size)
    ]
    if len(possible_counts) < n_pairs:
        return None
    hole_counts = random.sample(possible_counts, n_pairs)
    random.shuffle(hole_counts)

    available_colors = [c for c in range(10) if c != BG]
    if 2 * n_pairs > len(available_colors):
        return None
    colors = random.sample(available_colors, 2 * n_pairs)
    frame_colors = colors[:n_pairs]
    sprite_colors = colors[n_pairs:]

    sprites: list[dict] = []
    for hole_count, color in zip(hole_counts, sprite_colors, strict=False):
        sprite = _make_sprite(sprite_size, hole_count, color)
        if sprite is None:
            return None
        sprites.append({"holes": int(hole_count), "color": int(color), "patch": sprite})

    random.shuffle(layout_choices)
    chosen_layout = None
    for rows, cols in layout_choices:
        row_gap = random.randint(1, 3)
        col_gap = random.randint(1, 3)
        grid_h = rows * frame_size + (rows - 1) * row_gap
        grid_w = cols * frame_size + (cols - 1) * col_gap
        valid_hw: list[tuple[int, int]] = []
        for h in range(max(INPUT_MIN, grid_h + 2), INPUT_MAX + 1):
            for w in range(max(INPUT_MIN, grid_w + 2), INPUT_MAX + 1):
                free_area = h * w - grid_h * grid_w
                if free_area < n_pairs * sprite_size * sprite_size:
                    continue
                if (h - grid_h) < sprite_size + 2 and (w - grid_w) < sprite_size + 2:
                    continue
                valid_hw.append((int(h), int(w)))
        if valid_hw:
            chosen_layout = (int(rows), int(cols), int(row_gap), int(col_gap), random.choice(valid_hw))
            break
    if chosen_layout is None:
        return None

    rows, cols, row_gap, col_gap, (height, width) = chosen_layout
    grid_h = rows * frame_size + (rows - 1) * row_gap
    grid_w = cols * frame_size + (cols - 1) * col_gap
    top = random.randint(0, height - grid_h)
    left = random.randint(0, width - grid_w)

    canvas = np.full((int(height), int(width)), BG, dtype=int)
    occupied = np.zeros((int(height), int(width)), dtype=bool)
    _reserve_rect(occupied, top, left, grid_h, grid_w, pad=1)

    counts_for_frames = hole_counts[:]
    random.shuffle(counts_for_frames)
    frames: list[dict] = []
    idx = 0
    for ri in range(rows):
        for ci in range(cols):
            hole_count = int(counts_for_frames[idx])
            color = int(frame_colors[idx])
            patch = _make_frame(frame_size, hole_count, color)
            if patch is None:
                return None
            rr = top + ri * (frame_size + row_gap)
            cc = left + ci * (frame_size + col_gap)
            _paint(canvas, patch, rr, cc)
            frames.append(
                {
                    "holes": int(hole_count),
                    "color": int(color),
                    "patch": patch,
                    "row_index": int(ri),
                    "col_index": int(ci),
                }
            )
            idx += 1

    sprite_order = list(range(n_pairs))
    random.shuffle(sprite_order)
    for idx in sprite_order:
        patch = np.asarray(sprites[idx]["patch"], dtype=int)
        placed = False
        for _ in range(300):
            rr = random.randint(0, height - sprite_size)
            cc = random.randint(0, width - sprite_size)
            if not _rect_free(occupied, rr, cc, sprite_size, sprite_size, pad=1):
                continue
            _paint(canvas, patch, rr, cc)
            _reserve_rect(occupied, rr, cc, sprite_size, sprite_size, pad=1)
            placed = True
            break
        if not placed:
            return None

    expected = _expected_output(
        frames=frames,
        sprites=sprites,
        rows=rows,
        cols=cols,
        frame_size=frame_size,
    )

    try:
        actual = main(canvas)
    except Exception:
        return None
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after repeated attempts")
