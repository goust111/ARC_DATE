import numpy as np


BG = 0
DIR_ORDER = ("up", "right", "down", "left")
DIR_INDEX = {name: i for i, name in enumerate(DIR_ORDER)}

# --------------- solver ---------------


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


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
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _gap_position(direction: str, h: int, w: int) -> tuple[int, int]:
    if direction == "up":
        return 0, w // 2
    if direction == "right":
        return h // 2, w - 1
    if direction == "down":
        return h - 1, w // 2
    if direction == "left":
        return h // 2, 0
    raise ValueError(f"unknown direction: {direction}")


def _stamp_mask(h: int, w: int, direction: str) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[h - 1, :] = True
    mask[:, 0] = True
    mask[:, w - 1] = True
    gr, gc = _gap_position(direction, h, w)
    mask[gr, gc] = False
    return mask


def _render_stamp(h: int, w: int, color: int, direction: str) -> np.ndarray:
    stamp = np.zeros((h, w), dtype=int)
    stamp[_stamp_mask(h, w, direction)] = int(color)
    return stamp


def _parse_stamp(grid: np.ndarray, comp: list[tuple[int, int]]) -> dict:
    coords = np.array(comp, dtype=int)
    values = np.unique(grid[coords[:, 0], coords[:, 1]])
    assert len(values) == 1 and int(values[0]) != BG

    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    assert h >= 3 and w >= 3

    color = int(values[0])
    local = grid[r0 : r1 + 1, c0 : c1 + 1] == color

    candidates = []
    for direction in DIR_ORDER:
        if direction in ("up", "down") and w % 2 == 0:
            continue
        if direction in ("left", "right") and h % 2 == 0:
            continue
        if np.array_equal(local, _stamp_mask(h, w, direction)):
            candidates.append(direction)

    assert len(candidates) == 1
    direction = candidates[0]
    return {
        "color": color,
        "h": h,
        "w": w,
        "area": h * w,
        "direction": direction,
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
    }


def _extract_stamps(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    comps = _components(g != BG)
    stamps = [_parse_stamp(g, comp) for comp in comps]
    assert stamps
    return stamps


def _sort_group(group: list[dict]) -> list[dict]:
    return sorted(
        group,
        key=lambda spec: (
            -int(spec["area"]),
            -int(spec["h"]),
            -int(spec["w"]),
            int(spec["color"]),
        ),
    )


def _nest_group(group: list[dict]) -> np.ndarray:
    ordered = _sort_group(group)
    canvas_h = max(int(spec["h"]) for spec in ordered)
    canvas_w = max(int(spec["w"]) for spec in ordered)
    block = np.zeros((canvas_h, canvas_w), dtype=int)
    for spec in ordered:
        h = int(spec["h"])
        w = int(spec["w"])
        r0 = (canvas_h - h) // 2
        c0 = (canvas_w - w) // 2
        stamp = _render_stamp(h, w, int(spec["color"]), str(spec["direction"]))
        mask = stamp != BG
        window = block[r0 : r0 + h, c0 : c0 + w]
        window[mask] = stamp[mask]
    return block


def _compose_output(stamps: list[dict]) -> np.ndarray:
    grouped = {direction: [] for direction in DIR_ORDER}
    for spec in stamps:
        grouped[str(spec["direction"])].append(spec)

    blocks = [_nest_group(grouped[direction]) for direction in DIR_ORDER if grouped[direction]]
    assert blocks

    out_h = max(block.shape[0] for block in blocks)
    out_w = sum(block.shape[1] for block in blocks) + 2 * (len(blocks) - 1)
    out = np.zeros((out_h, out_w), dtype=int)

    col = 0
    for i, block in enumerate(blocks):
        h, w = block.shape
        out[:h, col : col + w] = block
        col += w
        if i + 1 != len(blocks):
            col += 2
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group scattered rectangular stamps by the side of their central border gap,
    then nest each direction-group from largest area to smallest and lay the
    groups out left-to-right in the fixed order up, right, down, left.
    """
    g = np.asarray(grid, dtype=int)
    stamps = _extract_stamps(g)
    return _compose_output(stamps)


# --------------- generator ---------------

_RNG = np.random.default_rng()
MAX_GRID_SIZE = 30
GEN_RETRIES = 400
PLACE_RETRIES = 250

GROUP_COUNT_CHOICES = np.array([1, 2, 2, 3, 3, 4], dtype=int)
STAMP_COUNT_CHOICES = {
    1: (1, 2, 2, 3, 3, 4),
    2: (1, 1, 2, 2, 3),
    3: (1, 1, 2, 2, 3),
    4: (1, 1, 2, 2),
}
OUTER_HEIGHT_CHOICES = {
    1: (7, 9, 11, 13),
    2: (5, 7, 9, 11),
    3: (5, 7, 9, 11),
    4: (3, 5, 7, 9),
}
OUTER_WIDTH_CHOICES = {
    1: (7, 9, 11, 13),
    2: (5, 7, 9, 11),
    3: (5, 7, 9),
    4: (3, 5),
}
COLOR_POOL = np.arange(1, 10, dtype=int)


def _choose_group_count(rng: np.random.Generator) -> int:
    return int(rng.choice(GROUP_COUNT_CHOICES))


def _choose_stamp_counts(rng: np.random.Generator, num_groups: int) -> list[int]:
    choices = STAMP_COUNT_CHOICES[num_groups]
    counts = [int(rng.choice(choices)) for _ in range(num_groups)]
    if max(counts) == 1 and rng.random() < 0.75:
        idx = int(rng.integers(num_groups))
        counts[idx] = 2
    return counts


def _sample_group_specs(
    rng: np.random.Generator, direction: str, count: int, num_groups: int
) -> list[dict] | None:
    min_outer = 2 * count + 1
    h_choices = [v for v in OUTER_HEIGHT_CHOICES[num_groups] if v >= min_outer]
    w_choices = [v for v in OUTER_WIDTH_CHOICES[num_groups] if v >= min_outer]
    if not h_choices or not w_choices:
        return None

    max_h = int(rng.choice(h_choices))
    max_w = int(rng.choice(w_choices))
    heights = sorted(
        map(int, rng.choice(np.arange(3, max_h + 1, 2), size=count, replace=False)),
        reverse=True,
    )
    widths = sorted(
        map(int, rng.choice(np.arange(3, max_w + 1, 2), size=count, replace=False)),
        reverse=True,
    )

    group = []
    for h, w in zip(heights, widths):
        group.append(
            {
                "color": 1,
                "h": int(h),
                "w": int(w),
                "area": int(h * w),
                "direction": direction,
            }
        )
    return group


def _assign_colors(rng: np.random.Generator, specs: list[dict]) -> None:
    n = len(specs)
    if n == 1:
        colors = [int(rng.choice(COLOR_POOL))]
    elif rng.random() < 0.45 and n <= len(COLOR_POOL):
        colors = [int(x) for x in rng.choice(COLOR_POOL, size=n, replace=False)]
    else:
        palette_size = int(rng.integers(2, min(5, n) + 1))
        palette = [int(x) for x in rng.choice(COLOR_POOL, size=palette_size, replace=False)]
        colors = [palette[int(i)] for i in rng.integers(0, palette_size, size=n)]
        if len(set(colors)) == 1:
            colors[-1] = palette[1]

    rng.shuffle(colors)
    for spec, color in zip(specs, colors):
        spec["color"] = int(color)


def _place_specs(rng: np.random.Generator, specs: list[dict]) -> np.ndarray | None:
    max_h = max(int(spec["h"]) for spec in specs)
    max_w = max(int(spec["w"]) for spec in specs)
    padded_area = sum((int(spec["h"]) + 2) * (int(spec["w"]) + 2) for spec in specs)
    side_hint = int(np.ceil(np.sqrt(padded_area)))

    low_h = max(12, max_h + 2, side_hint - 2)
    low_w = max(12, max_w + 2, side_hint - 2)
    if low_h > MAX_GRID_SIZE or low_w > MAX_GRID_SIZE:
        return None

    H = int(rng.integers(low_h, MAX_GRID_SIZE + 1))
    W = int(rng.integers(low_w, MAX_GRID_SIZE + 1))
    grid = np.zeros((H, W), dtype=int)
    blocked = np.zeros((H, W), dtype=bool)

    ordered = sorted(
        specs,
        key=lambda spec: (-int(spec["area"]), -int(spec["h"]), -int(spec["w"])),
    )
    for spec in ordered:
        h = int(spec["h"])
        w = int(spec["w"])
        if H - h <= 1 or W - w <= 1:
            return None

        placed = False
        for _ in range(PLACE_RETRIES):
            r0 = int(rng.integers(1, H - h))
            c0 = int(rng.integers(1, W - w))
            if blocked[r0 - 1 : r0 + h + 1, c0 - 1 : c0 + w + 1].any():
                continue

            stamp = _render_stamp(h, w, int(spec["color"]), str(spec["direction"]))
            mask = stamp != BG
            grid[r0 : r0 + h, c0 : c0 + w][mask] = stamp[mask]
            blocked[r0 - 1 : r0 + h + 1, c0 - 1 : c0 + w + 1] = True
            placed = True
            break

        if not placed:
            return None

    return grid


def _spec_signature(specs: list[dict]) -> list[tuple[int, int, int, int]]:
    return sorted(
        (
            DIR_INDEX[str(spec["direction"])],
            int(spec["h"]),
            int(spec["w"]),
            int(spec["color"]),
        )
        for spec in specs
    )


def _gen_attempt() -> np.ndarray | None:
    rng = _RNG
    num_groups = _choose_group_count(rng)
    active_dirs = [str(x) for x in rng.choice(np.array(DIR_ORDER), size=num_groups, replace=False)]
    counts = _choose_stamp_counts(rng, num_groups)

    specs: list[dict] = []
    for direction, count in zip(active_dirs, counts):
        group = _sample_group_specs(rng, direction, count, num_groups)
        if group is None:
            return None
        specs.extend(group)

    _assign_colors(rng, specs)

    expected = _compose_output(specs)
    if expected.shape[0] > MAX_GRID_SIZE or expected.shape[1] > MAX_GRID_SIZE:
        return None

    grid = _place_specs(rng, specs)
    if grid is None:
        return None

    extracted = _extract_stamps(grid)
    if _spec_signature(extracted) != _spec_signature(specs):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
