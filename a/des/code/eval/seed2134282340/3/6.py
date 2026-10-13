import random

import numpy as np


BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _component_from_sprite(sprite: np.ndarray) -> dict:
    color_values = [int(v) for v in np.unique(sprite) if int(v) != BG]
    assert len(color_values) == 1
    color = color_values[0]
    mask = sprite == color
    rows, cols = np.where(mask)
    assert rows.size > 0
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    cropped = sprite[r0 : r1 + 1, c0 : c1 + 1].copy()
    return {
        "color": color,
        "sprite": cropped,
        "mask": cropped != BG,
        "height": int(cropped.shape[0]),
        "width": int(cropped.shape[1]),
    }


def _extract_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BG or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            coord_arr = np.array(coords, dtype=int)
            rows = coord_arr[:, 0]
            cols = coord_arr[:, 1]
            r0, r1 = int(rows.min()), int(rows.max())
            c0, c1 = int(cols.min()), int(cols.max())
            sprite = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
            sprite[rows - r0, cols - c0] = color
            components.append(
                {
                    "color": color,
                    "sprite": sprite,
                    "mask": sprite != BG,
                    "height": int(sprite.shape[0]),
                    "width": int(sprite.shape[1]),
                }
            )
    return components


def _assemble_pair(left: dict, right: dict) -> dict | None:
    left_mask = np.asarray(left["mask"], dtype=bool)
    right_mask = np.asarray(right["mask"], dtype=bool)
    h = int(left["height"])
    if h != int(right["height"]):
        return None

    wl = int(left["width"])
    wr = int(right["width"])
    candidates: list[dict] = []
    for shift in range(wl + 1):
        out_w = max(wl, shift + wr)
        if out_w <= 0:
            continue
        cover = np.zeros((h, out_w), dtype=np.int8)
        cover[:, :wl] += left_mask.astype(np.int8)
        cover[:, shift : shift + wr] += right_mask.astype(np.int8)
        if np.any(cover > 1):
            continue
        if not np.all(cover == 1):
            continue

        sprite = np.zeros((h, out_w), dtype=int)
        sprite[:, :wl][left_mask] = int(left["color"])
        right_view = sprite[:, shift : shift + wr]
        right_view[right_mask] = int(right["color"])
        candidates.append(
            {
                "left_color": int(left["color"]),
                "right_color": int(right["color"]),
                "width": out_w,
                "height": h,
                "area": int(h * out_w),
                "shift": shift,
                "sprite": sprite,
            }
        )

    if len(candidates) != 1:
        return None
    return candidates[0]


def _solve_pairing(components: list[dict]) -> list[dict]:
    n = len(components)
    assert n > 0 and n % 2 == 0
    heights = {int(comp["height"]) for comp in components}
    assert len(heights) == 1

    candidates: dict[tuple[int, int], dict] = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            assembled = _assemble_pair(components[i], components[j])
            if assembled is not None:
                candidates[(i, j)] = assembled

    solutions: list[list[dict]] = []

    def backtrack(remaining: tuple[int, ...], acc: list[dict]) -> None:
        if len(solutions) > 1:
            return
        if not remaining:
            solutions.append([item.copy() for item in acc])
            return
        first = remaining[0]
        tail = remaining[1:]
        for idx, other in enumerate(tail):
            if (first, other) in candidates:
                info = candidates[(first, other)].copy()
                info["left_index"] = first
                info["right_index"] = other
                next_remaining = tail[:idx] + tail[idx + 1 :]
                acc.append(info)
                backtrack(next_remaining, acc)
                acc.pop()
            if (other, first) in candidates:
                info = candidates[(other, first)].copy()
                info["left_index"] = other
                info["right_index"] = first
                next_remaining = tail[:idx] + tail[idx + 1 :]
                acc.append(info)
                backtrack(next_remaining, acc)
                acc.pop()

    backtrack(tuple(range(n)), [])
    assert len(solutions) == 1, "expected a unique complementary pairing"
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair monochrome shapes by horizontal complementarity and output the merged rectangles sorted by area.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    components = _extract_components(g)
    colors = [int(comp["color"]) for comp in components]
    assert len(colors) == len(set(colors)), "each shape should have a unique color"

    pairs = _solve_pairing(components)
    pairs.sort(key=lambda item: int(item["area"]))

    out_h = int(pairs[0]["height"])
    out_w = sum(int(item["width"]) for item in pairs) + len(pairs) - 1
    out = np.zeros((out_h, out_w), dtype=int)
    c = 0
    for item in pairs:
        sprite = np.asarray(item["sprite"], dtype=int)
        h, w = map(int, sprite.shape)
        assert h == out_h
        out[:, c : c + w] = sprite
        c += w + 1
    return out


def _random_profile(height: int, width: int) -> list[int]:
    assert width >= 4
    for _ in range(120):
        lo = random.randint(0, width - 3)
        hi = random.randint(lo + 1, width - 2)
        cur = random.randint(lo, hi)
        profile = [cur]
        for _row in range(1, height):
            step = random.choice((-1, 0, 1))
            nxt = min(hi, max(lo, cur + step))
            if random.random() < 0.25:
                nxt = random.randint(lo, hi)
            profile.append(nxt)
            cur = nxt
        if len(set(profile)) < 2:
            continue
        if min(profile) == max(profile):
            continue
        return profile
    raise RuntimeError("failed to sample a non-degenerate profile")


def _make_pair_sprites(height: int, width: int, left_color: int, right_color: int) -> tuple[np.ndarray, np.ndarray]:
    profile = _random_profile(height, width)
    min_cut = min(profile)
    max_cut = max(profile)
    left_sprite = np.zeros((height, max_cut + 1), dtype=int)
    right_sprite = np.zeros((height, width - (min_cut + 1)), dtype=int)
    for r, cut in enumerate(profile):
        left_sprite[r, : cut + 1] = left_color
        right_sprite[r, cut - min_cut :] = right_color
    return left_sprite, right_sprite


def _build_pair_specs(height: int, widths: list[int], colors: list[int]) -> list[dict]:
    specs: list[dict] = []
    color_idx = 0
    for width in widths:
        left_color = colors[color_idx]
        right_color = colors[color_idx + 1]
        color_idx += 2
        left_sprite, right_sprite = _make_pair_sprites(height, width, left_color, right_color)
        specs.append(
            {
                "output_width": width,
                "left_color": left_color,
                "right_color": right_color,
                "left_sprite": left_sprite,
                "right_sprite": right_sprite,
            }
        )
    return specs


def _validate_pair_specs(specs: list[dict], height: int) -> bool:
    components: list[dict] = []
    expected_pairs = set()
    expected_areas = []
    for pair_id, spec in enumerate(specs):
        left = _component_from_sprite(spec["left_sprite"])
        right = _component_from_sprite(spec["right_sprite"])
        components.extend([left, right])
        expected_pairs.add(frozenset({int(spec["left_color"]), int(spec["right_color"])}))
        expected_areas.append(height * int(spec["output_width"]))

    if len(expected_areas) != len(set(expected_areas)):
        return False

    try:
        solved = _solve_pairing(components)
    except AssertionError:
        return False

    solved_pairs = {
        frozenset({int(item["left_color"]), int(item["right_color"])})
        for item in solved
    }
    solved_areas = sorted(int(item["area"]) for item in solved)
    return solved_pairs == expected_pairs and solved_areas == sorted(expected_areas)


def _lane_assign(widths: list[int], lane_count: int) -> list[list[int]]:
    lanes: list[list[int]] = [[] for _ in range(lane_count)]
    loads = [0] * lane_count
    order = list(range(len(widths)))
    order.sort(key=lambda idx: widths[idx], reverse=True)
    for idx in order:
        best_lane = min(range(lane_count), key=lambda lane: (loads[lane], random.random()))
        lanes[best_lane].append(idx)
        loads[best_lane] += widths[idx]
    for lane in lanes:
        random.shuffle(lane)
    return lanes


def _place_sprites(sprites: list[np.ndarray], height: int) -> np.ndarray | None:
    widths = [int(sprite.shape[1]) for sprite in sprites]
    n = len(sprites)
    feasible_lane_counts = [lane_count for lane_count in (2, 3) if lane_count <= n]
    random.shuffle(feasible_lane_counts)
    for lane_count in feasible_lane_counts:
        lanes = _lane_assign(widths, lane_count)
        if any(len(lane) == 0 for lane in lanes):
            continue

        h_gap = random.randint(1, 3)
        v_gap = random.randint(1, 3)
        lane_widths = []
        for lane in lanes:
            total = sum(widths[idx] for idx in lane) + h_gap * (len(lane) - 1)
            lane_widths.append(total)

        grid_w = max(lane_widths) + random.randint(2, 6)
        grid_h = lane_count * height + (lane_count - 1) * v_gap + random.randint(2, 6)
        if grid_w > 30 or grid_h > 30:
            continue

        row_bases = []
        cur_r = random.randint(0, 1)
        for lane_idx in range(lane_count):
            row_bases.append(cur_r)
            cur_r += height + v_gap
        if row_bases[-1] + height > grid_h:
            continue

        grid = np.zeros((grid_h, grid_w), dtype=int)
        for lane_idx, lane in enumerate(lanes):
            used_width = lane_widths[lane_idx]
            extra = grid_w - used_width
            if extra < 0:
                break
            c = random.randint(0, extra)
            r0 = row_bases[lane_idx]
            for idx in lane:
                sprite = sprites[idx]
                sh, sw = map(int, sprite.shape)
                assert sh == height
                patch = grid[r0 : r0 + sh, c : c + sw]
                if np.any((patch != BG) & (sprite != BG)):
                    break
                patch[sprite != BG] = sprite[sprite != BG]
                c += sw + h_gap
            else:
                continue
            break
        else:
            return grid
    return None


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(2, 4)
    height = random.randint(3, 6)
    width_pool = list(range(4, 10))
    if pair_count > len(width_pool):
        return None
    widths = random.sample(width_pool, pair_count)
    random.shuffle(widths)
    colors = random.sample(PALETTE, 2 * pair_count)

    try:
        specs = _build_pair_specs(height, widths, colors)
    except RuntimeError:
        return None
    if not _validate_pair_specs(specs, height):
        return None

    sprites: list[np.ndarray] = []
    for spec in specs:
        sprites.append(spec["left_sprite"])
        sprites.append(spec["right_sprite"])
    random.shuffle(sprites)

    grid = _place_sprites(sprites, height)
    if grid is None:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape[0] != height:
        return None
    expected_widths = sorted(widths)
    actual_widths = [int(item["width"]) for item in _solve_pairing(_extract_components(grid))]
    if sorted(actual_widths) != expected_widths:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
