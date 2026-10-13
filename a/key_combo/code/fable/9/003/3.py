import random
from collections import Counter, defaultdict, deque

import numpy as np


RED = 2
BG_COLOR = 0
BLOCK_COLORS = [1, 3, 4, 5, 6, 7, 8, 9]
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _mask_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, pts[0]))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for rr, cc in _neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                q.append((rr, cc))
    return len(seen) == int(mask.sum())


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return comps


def _drop_distance(coords: list[tuple[int, int]], occupied: np.ndarray, height: int) -> int:
    drop = 0
    while True:
        for r, c in coords:
            nr = r + drop + 1
            if nr >= height or occupied[nr, c]:
                return drop
        drop += 1


def _settle_blocks(grid: np.ndarray) -> tuple[np.ndarray, list[dict], int]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    h, w = g.shape
    comps = _find_components(g, bg)
    comps.sort(key=lambda comp: (-comp["bbox"][2], -comp["bbox"][0], comp["bbox"][1], comp["color"]))

    occupied = np.zeros((h, w), dtype=bool)
    out = np.full((h, w), bg, dtype=int)
    placed: list[dict] = []
    for comp in comps:
        drop = _drop_distance(comp["coords"], occupied, h)
        shifted = [(r + drop, c) for r, c in comp["coords"]]
        for r, c in shifted:
            occupied[r, c] = True
            out[r, c] = comp["color"]
        rows = [r for r, _ in shifted]
        cols = [c for _, c in shifted]
        placed.append(
            {
                "color": comp["color"],
                "coords": shifted,
                "bbox": (min(rows), min(cols), max(rows), max(cols)),
            }
        )
    return out, placed, bg


def _paint_enclosed_pockets(grid: np.ndarray, bg: int) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    outside = np.zeros((h, w), dtype=bool)
    q = deque()

    for c in range(w):
        if int(g[0, c]) == bg and not outside[0, c]:
            outside[0, c] = True
            q.append((0, c))
    for r in range(h):
        if int(g[r, 0]) == bg and not outside[r, 0]:
            outside[r, 0] = True
            q.append((r, 0))
        if int(g[r, w - 1]) == bg and not outside[r, w - 1]:
            outside[r, w - 1] = True
            q.append((r, w - 1))

    while q:
        r, c = q.popleft()
        for rr, cc in _neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == bg and not outside[rr, cc]:
                outside[rr, cc] = True
                q.append((rr, cc))

    out = g.copy()
    fill_mask = (g == bg) & (~outside)
    out[fill_mask] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Let all floating blocks fall vertically, then fill only the closed pockets created after stacking.

    Input structure:
    - The background color is dominant, and the grid bottom acts as ground rather than an open border.
    - Several monochrome 4-connected blocks hang above the ground; their top and bottom edges may contain one-cell-deep notches.
    - Distinct blocks are separated in the input and only move straight downward.

    Rule:
    - Settle every block by gravity, processing lower-starting blocks first, until each one hits the ground or a previously settled block.
    - After all blocks stop, inspect the remaining background cells.
    - Any background region that cannot reach the top edge, left edge, or right edge is a closed pocket and is recolored red; background that stays externally connected remains unchanged.

    Key insight:
    - The red regions are not pre-drawn holes inside a single object; they are new cavities that appear only after the whole falling-and-stacking process is mentally completed.
    """
    settled, _placed, bg = _settle_blocks(np.asarray(grid, dtype=int))
    return _paint_enclosed_pockets(settled, bg)


def _random_notch(width: int) -> tuple[int, int]:
    max_width = max(1, width - 2)
    notch_width = random.randint(1, min(3, max_width))
    start = random.randint(1, width - notch_width - 1)
    return start, notch_width


def _make_block(
    height: int,
    width: int,
    *,
    top_notch: tuple[int, int] | None = None,
    bottom_notch: tuple[int, int] | None = None,
) -> np.ndarray:
    candidates = [
        (top_notch, bottom_notch),
        (top_notch, None),
        (None, bottom_notch),
        (None, None),
    ]
    for top_cut, bottom_cut in candidates:
        mask = np.ones((height, width), dtype=bool)
        if top_cut is not None:
            start, notch_width = top_cut
            mask[0, start : start + notch_width] = False
        if bottom_cut is not None:
            start, notch_width = bottom_cut
            mask[-1, start : start + notch_width] = False
        if _mask_connected(mask):
            return mask
    raise AssertionError("failed to construct a connected block")


def _stamp_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    rows, cols = mask.shape
    for r in range(rows):
        for c in range(cols):
            if mask[r, c]:
                grid[top + r, left + c] = color


def _drop_top(mask: np.ndarray, left: int, occupied: np.ndarray) -> int:
    h, w = occupied.shape
    mh, mw = mask.shape
    assert 0 <= left and left + mw <= w
    top = 0
    while True:
        blocked = False
        for r in range(mh):
            for c in range(mw):
                if not mask[r, c]:
                    continue
                nr = top + r + 1
                if nr >= h or occupied[nr, left + c]:
                    blocked = True
                    break
            if blocked:
                break
        if blocked:
            return top
        top += 1


def _build_primary_stack(colors: list[int]) -> list[dict]:
    levels = random.choice([2, 3])
    width = random.randint(5, 8)
    closed_mode = random.choices(["ground", "stack", "both"], weights=[3, 3, 2], k=1)[0]

    top_notch = _random_notch(width) if closed_mode in {"stack", "both"} else None
    bottom_notch = _random_notch(width) if closed_mode in {"ground", "both"} else None
    stack = [
        {
            "mask": _make_block(
                random.randint(3, 4),
                width,
                top_notch=top_notch,
                bottom_notch=bottom_notch,
            ),
            "color": colors[0],
        }
    ]

    second_bottom_notch = top_notch if top_notch is not None and random.random() < 0.55 else None
    second_top_notch = _random_notch(width) if levels == 2 and random.random() < 0.8 else None
    stack.append(
        {
            "mask": _make_block(
                random.randint(2, 4),
                width,
                top_notch=second_top_notch,
                bottom_notch=second_bottom_notch,
            ),
            "color": colors[1],
        }
    )

    if levels == 3:
        stack.append(
            {
                "mask": _make_block(
                    random.randint(2, 3),
                    width,
                    top_notch=_random_notch(width) if random.random() < 0.85 else None,
                    bottom_notch=None,
                ),
                "color": colors[2],
            }
        )
    return stack


def _build_extra_stack(colors: list[int]) -> list[dict]:
    levels = random.choice([1, 2])
    width = random.randint(4, 7)
    bottom = {
        "mask": _make_block(
            random.randint(2, 4),
            width,
            top_notch=None,
            bottom_notch=_random_notch(width) if levels == 1 and random.random() < 0.25 else None,
        ),
        "color": colors[0],
    }
    stack = [bottom]
    if levels == 2:
        stack.append(
            {
                "mask": _make_block(
                    random.randint(2, 3),
                    width,
                    top_notch=_random_notch(width) if random.random() < 0.7 else None,
                    bottom_notch=None,
                ),
                "color": colors[1],
            }
        )
    return stack


def _place_stack_columns(width: int, stack_widths: list[int]) -> list[int] | None:
    placements: list[int] = []
    for sw in stack_widths:
        low = 1
        high = width - sw - 1
        if low > high:
            return None
        found = None
        for _ in range(120):
            left = random.randint(low, high)
            if all(left + sw + 1 < other or other + other_w + 1 < left for other, other_w in placements):
                found = left
                break
        if found is None:
            return None
        placements.append((found, sw))
    return [left for left, _ in placements]


def _build_output_from_stacks(height: int, width: int, stacks: list[list[dict]]) -> tuple[np.ndarray, list[dict]] | tuple[None, None]:
    stack_widths = [max(block["mask"].shape[1] for block in stack) for stack in stacks]
    xs = _place_stack_columns(width, stack_widths)
    if xs is None:
        return None, None

    occupied = np.zeros((height, width), dtype=bool)
    raw = np.full((height, width), BG_COLOR, dtype=int)
    placed_blocks: list[dict] = []
    for stack_id, (stack, left) in enumerate(zip(stacks, xs)):
        for level, block in enumerate(stack):
            mask = block["mask"]
            top = _drop_top(mask, left, occupied)
            mh, mw = mask.shape
            for r in range(mh):
                for c in range(mw):
                    if not mask[r, c]:
                        continue
                    occupied[top + r, left + c] = True
                    raw[top + r, left + c] = int(block["color"])
            placed_blocks.append(
                {
                    "stack_id": stack_id,
                    "level": level,
                    "mask": mask,
                    "color": int(block["color"]),
                    "final_top": top,
                    "left": left,
                }
            )
    return raw, placed_blocks


def _build_input_from_output(height: int, width: int, placed_blocks: list[dict]) -> np.ndarray | None:
    by_stack: dict[int, list[dict]] = defaultdict(list)
    for block in placed_blocks:
        by_stack[int(block["stack_id"])].append(block)

    inp = np.full((height, width), BG_COLOR, dtype=int)
    for stack_id, blocks in by_stack.items():
        del stack_id
        blocks.sort(key=lambda block: block["final_top"], reverse=True)
        extras = [0]
        for _ in range(1, len(blocks)):
            extras.append(extras[-1] + random.randint(2, 4))
        max_base_lift = min(int(block["final_top"]) - extra for block, extra in zip(blocks, extras))
        if max_base_lift < 1:
            return None
        base_lift = random.randint(1, max_base_lift)

        for block, extra in zip(blocks, extras):
            lift = base_lift + extra
            top = int(block["final_top"]) - lift
            mask = block["mask"]
            left = int(block["left"])
            mh, mw = mask.shape
            if top < 0 or left < 0 or top + mh > height or left + mw > width:
                return None
            for r in range(mh):
                for c in range(mw):
                    if not mask[r, c]:
                        continue
                    rr = top + r
                    cc = left + c
                    if int(inp[rr, cc]) != BG_COLOR:
                        return None
                    inp[rr, cc] = int(block["color"])
    return inp


def _count_red_cells(grid: np.ndarray) -> int:
    return int(np.sum(np.asarray(grid, dtype=int) == RED))


def _touching_components(grid: np.ndarray, bg: int) -> bool:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    for r in range(h):
        for c in range(w):
            if int(g[r, c]) == bg:
                continue
            for rr in range(max(0, r - 1), min(h, r + 2)):
                for cc in range(max(0, c - 1), min(w, c + 2)):
                    if (rr, cc) == (r, c) or int(g[rr, cc]) == bg:
                        continue
                    if abs(rr - r) + abs(cc - c) == 1 and int(g[rr, cc]) != int(g[r, c]):
                        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    primary_colors = random.sample(BLOCK_COLORS, 3)
    stacks = [_build_primary_stack(primary_colors)]

    remaining = [color for color in BLOCK_COLORS if color not in primary_colors]
    extra_count = random.choice([1, 2])
    for _ in range(extra_count):
        need = 2
        if len(remaining) < need:
            remaining = BLOCK_COLORS.copy()
        sample = random.sample(remaining, need)
        stacks.append(_build_extra_stack(sample))
        remaining = [color for color in remaining if color not in sample]

    min_width = sum(max(block["mask"].shape[1] for block in stack) for stack in stacks) + 3 * (len(stacks) - 1) + 2
    width_low = max(18, min_width)
    width_high = min(28, width_low + 6)
    if width_low > width_high:
        return None
    width = random.randint(width_low, width_high)

    min_height = 0
    for stack in stacks:
        total_height = sum(block["mask"].shape[0] for block in stack)
        min_height = max(min_height, total_height + 2 * (len(stack) - 1) + 2)
    height_low = max(16, min_height)
    height_high = min(24, height_low + 6)
    if height_low > height_high:
        return None
    height = random.randint(height_low, height_high)

    raw_output, placed_blocks = _build_output_from_stacks(height, width, stacks)
    if raw_output is None or placed_blocks is None:
        return None

    final_output = _paint_enclosed_pockets(raw_output, BG_COLOR)
    if _count_red_cells(final_output) == 0:
        return None

    inp = _build_input_from_output(height, width, placed_blocks)
    if inp is None:
        return None
    if _touching_components(inp, BG_COLOR):
        return None
    if not np.array_equal(main(inp), final_output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid falling-pocket input")
