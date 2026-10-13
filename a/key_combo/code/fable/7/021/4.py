import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import bbox_h, bbox_w, colors_present, find_components, get_bg


MARKER_SIZE = 2
ALL_COLORS = tuple(range(10))
MAX_GENERATION_ATTEMPTS = 500


def _marker_pairs(g: np.ndarray) -> tuple[int, dict[int, dict[str, object]]]:
    bg = get_bg(g)
    pairs: dict[int, dict[str, object]] = {}

    for color in colors_present(g, exclude={bg}):
        comps = find_components(
            g,
            bg=bg,
            colors=[color],
            connectivity=4,
            monochromatic=True,
            return_format="dict",
        )

        boxes = []
        ok = True
        for comp in comps:
            box = tuple(map(int, comp["bbox"]))
            area = int(comp["area"])
            if area == 4 and bbox_h(box) == 2 and bbox_w(box) == 2:
                boxes.append(box)
                continue
            if area != 1:
                ok = False
                break
        if not ok:
            continue
        if len(boxes) != 2 or len(comps) - len(boxes) > 1:
            continue

        left, right = sorted(boxes, key=lambda b: (b[1], b[0]))
        if (left[0], left[2]) != (right[0], right[2]):
            continue

        gap_width = int(right[1] - left[3] - 1)
        if gap_width <= 0:
            continue

        pairs[int(color)] = {
            "left": left,
            "right": right,
            "row": int(left[0]),
            "gap_width": gap_width,
        }

    assert len(pairs) >= 2, "expected at least two marker-color pairs"
    gap_widths = {int(info["gap_width"]) for info in pairs.values()}
    assert len(gap_widths) == 1, "all extracted regions must have the same width"
    return int(bg), pairs


def _parse_scene(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    bg, pairs = _marker_pairs(g)
    marker_colors = sorted(pairs)
    marker_palette = np.array(marker_colors, dtype=int)

    cleaned_by_color: dict[int, np.ndarray] = {}
    successor: dict[int, int] = {}
    row_order = sorted(marker_colors, key=lambda color: int(pairs[color]["row"]))

    for color in marker_colors:
        left = pairs[color]["left"]
        right = pairs[color]["right"]
        stripe = g[left[0] : left[2] + 1, left[3] + 1 : right[1]].copy()
        assert stripe.shape[0] == 2 and stripe.shape[1] == int(pairs[color]["gap_width"])

        pointer_mask = np.isin(stripe, marker_palette)
        assert int(pointer_mask.sum()) <= 1, "each region may contain at most one pointer pixel"
        if pointer_mask.any():
            pr, pc = map(int, np.argwhere(pointer_mask)[0])
            nxt = int(stripe[pr, pc])
            assert nxt != color and nxt in pairs, "pointer must name another marker color"
            successor[color] = nxt
            stripe[pr, pc] = bg

        cleaned_by_color[color] = stripe

    assert len(successor) == len(marker_colors) - 1, "exactly one extracted region must be the chain tail"
    indegree = {color: 0 for color in marker_colors}
    for src, dst in successor.items():
        assert src in indegree and dst in indegree
        indegree[dst] += 1
        assert indegree[dst] <= 1, "pointer chain must be unbranched"

    heads = [color for color in marker_colors if indegree[color] == 0]
    tails = [color for color in marker_colors if color not in successor]
    assert len(heads) == 1 and len(tails) == 1, "pointer chain must have one head and one tail"

    chain_order = []
    seen = set()
    cur = heads[0]
    while True:
        assert cur not in seen, "pointer chain must not cycle"
        seen.add(cur)
        chain_order.append(cur)
        if cur not in successor:
            break
        cur = successor[cur]

    assert len(chain_order) == len(marker_colors), "pointer chain must visit every extracted region"
    return {
        "bg": bg,
        "pairs": pairs,
        "row_order": row_order,
        "chain_order": chain_order,
        "successor": successor,
        "cleaned_by_color": cleaned_by_color,
    }


def main(input_grid: np.ndarray) -> np.ndarray:
    """Stack the marker-delimited content bands in pointer-chain order after erasing the pointer pixels.

    Input structure:
    - The grid has one dominant background, sparse distractors, and several marker colors.
    - Each marker color appears exactly twice as solid 2x2 blocks on the same two rows, with a same-width content rectangle between them.
    - Inside each content rectangle, at most one isolated pixel uses a marker color; that pixel names the next marker pair in the chain.

    Rule:
    - Identify every left/right 2x2 marker pair and crop the 2-row rectangle strictly between the two blocks.
    - Read the successor relation from the marker-colored pointer pixel inside each cropped region; the unique region with no incoming pointer is the chain head.
    - Follow the chain from head to tail, erase each region's pointer pixel back to background, and vertically stack the cleaned regions.

    Key insight:
    - The output order is encoded by the marker-colored pointer chain inside the extracted regions, not by the regions' positions in the original scene.
    """
    parsed = _parse_scene(np.asarray(input_grid, dtype=int))
    ordered = [parsed["cleaned_by_color"][color] for color in parsed["chain_order"]]
    return np.vstack(ordered)


def _pointer_slots(stripe: np.ndarray, bg: int) -> list[tuple[int, int]]:
    h, w = stripe.shape
    assert h == 2
    slots = []
    for r in range(h):
        for c in range(w):
            if int(stripe[r, c]) != bg:
                continue
            clear = True
            for rr in range(max(0, r - 1), min(h, r + 2)):
                for cc in range(max(0, c - 1), min(w, c + 2)):
                    if (rr, cc) == (r, c):
                        continue
                    if int(stripe[rr, cc]) != bg:
                        clear = False
                        break
                if not clear:
                    break
            if clear:
                slots.append((r, c))
    return slots


def _sample_clean_stripe(
    rng: np.random.Generator, width: int, bg: int, content_colors: tuple[int, ...]
) -> np.ndarray:
    assert width >= 4
    for _ in range(400):
        stripe = np.full((2, width), bg, dtype=int)
        style = int(rng.integers(0, 4))

        if style == 0:
            cells = rng.choice(2 * width, size=int(rng.integers(2, min(6, 2 * width - 1) + 1)), replace=False)
            for cell in cells:
                rr, cc = divmod(int(cell), width)
                stripe[rr, cc] = int(rng.choice(content_colors))
        elif style == 1:
            row = int(rng.integers(0, 2))
            col = int(rng.integers(0, width - 1))
            stripe[row, col : col + 2] = int(rng.choice(content_colors))
            extra = rng.choice(2 * width, size=int(rng.integers(1, 4)), replace=False)
            for cell in extra:
                rr, cc = divmod(int(cell), width)
                stripe[rr, cc] = int(rng.choice(content_colors))
        elif style == 2:
            start = int(rng.integers(0, width - 2))
            coords = [(0, start), (1, start + 1), (0, start + 2)]
            if rng.random() < 0.5:
                coords = [(1 - r, c) for r, c in coords]
            for rr, cc in coords:
                stripe[rr, cc] = int(rng.choice(content_colors))
            if rng.random() < 0.7:
                rr = int(rng.integers(0, 2))
                cc = int(rng.integers(0, width))
                stripe[rr, cc] = int(rng.choice(content_colors))
        else:
            row0 = int(rng.integers(0, 2))
            row1 = 1 - row0
            c0, c1 = sorted(map(int, rng.choice(width, size=2, replace=False)))
            stripe[row0, c0] = int(rng.choice(content_colors))
            stripe[row0, c1] = int(rng.choice(content_colors))
            if c0 + 1 < width and rng.random() < 0.7:
                stripe[row1, c0 + 1] = int(rng.choice(content_colors))
            if c1 - 1 >= 0 and rng.random() < 0.7:
                stripe[row1, c1 - 1] = int(rng.choice(content_colors))

        if int(np.count_nonzero(stripe != bg)) < 2:
            continue
        if _pointer_slots(stripe, bg):
            return stripe

    raise RuntimeError("failed to sample a non-degenerate stripe")


def _sample_chain_order(rng: np.random.Generator, row_order: list[int]) -> list[int]:
    chain = list(row_order)
    while True:
        rng.shuffle(chain)
        if chain != row_order:
            return chain


def _sample_layout(rng: np.random.Generator, num_segments: int, gap_width: int) -> tuple[int, int, list[int], list[int]]:
    pair_width = gap_width + 2 * MARKER_SIZE
    extra_height = int(rng.integers(0, 7))
    gaps = rng.integers(1, 5, size=max(0, num_segments - 1))
    height = int(2 * num_segments + gaps.sum() + extra_height)
    height = min(height, 30)

    used = 2 * num_segments + int(gaps.sum())
    slack = height - used
    start = int(rng.integers(0, slack + 1))
    row_starts = [start]
    for gap in gaps:
        start += 2 + int(gap)
        row_starts.append(start)

    width = int(rng.integers(pair_width + 1, min(31, pair_width + 13)))
    left_max = width - pair_width
    left_cols = [int(rng.integers(0, left_max + 1)) for _ in range(num_segments)]
    return height, width, row_starts, left_cols


def _place_distractor_blocks(
    rng: np.random.Generator,
    grid: np.ndarray,
    blocked: np.ndarray,
    bg: int,
    distractor_color: int,
) -> None:
    if rng.random() >= 0.65:
        return

    h, w = grid.shape
    positions = []
    for r in range(h - 1):
        for c in range(w - 1):
            if blocked[r : r + 2, c : c + 2].any():
                continue
            positions.append((r, c))
    if not positions:
        return

    first = positions[int(rng.integers(0, len(positions)))]
    r0, c0 = first
    grid[r0 : r0 + 2, c0 : c0 + 2] = distractor_color
    blocked[r0 : r0 + 2, c0 : c0 + 2] = True

    if rng.random() < 0.7:
        options = [(r, c) for r, c in positions if r != r0 and not blocked[r : r + 2, c : c + 2].any()]
        if options:
            r1, c1 = options[int(rng.integers(0, len(options)))]
            grid[r1 : r1 + 2, c1 : c1 + 2] = distractor_color
            blocked[r1 : r1 + 2, c1 : c1 + 2] = True


def _place_noise(
    rng: np.random.Generator,
    grid: np.ndarray,
    blocked: np.ndarray,
    bg: int,
    noise_colors: tuple[int, ...],
) -> None:
    h, w = grid.shape
    free = [(r, c) for r in range(h) for c in range(w) if not blocked[r, c]]
    if not free:
        return

    noise_count = int(rng.integers(1, min(len(free), max(2, grid.size // 8)) + 1))
    picks = rng.choice(len(free), size=noise_count, replace=False)
    for idx in np.atleast_1d(picks):
        r, c = free[int(idx)]
        grid[r, c] = int(rng.choice(noise_colors))
        blocked[r, c] = True


def _validate_generated(grid: np.ndarray, spec: dict[str, object]) -> bool:
    expected = spec["expected"]
    try:
        parsed = _parse_scene(grid)
        actual = main(grid)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False
    if parsed["chain_order"] != spec["chain_order"]:
        return False
    if parsed["row_order"] == parsed["chain_order"]:
        return False
    if parsed["successor"] != spec["successor"]:
        return False
    for color, stripe in spec["cleaned_by_color"].items():
        if not np.array_equal(parsed["cleaned_by_color"][color], stripe):
            return False
    if int(np.count_nonzero(grid != spec["bg"])) <= int(np.count_nonzero(expected != spec["bg"])):
        return False
    return True


def _generate_one(rng: np.random.Generator) -> np.ndarray | None:
    bg = int(rng.choice(ALL_COLORS))
    usable = [color for color in ALL_COLORS if color != bg]
    num_segments = int(rng.integers(2, 6))
    marker_colors = list(map(int, rng.choice(usable, size=num_segments, replace=False)))

    rest = [color for color in usable if color not in marker_colors]
    rng.shuffle(rest)
    content_count = int(rng.integers(2, min(4, len(rest)) + 1))
    content_colors = tuple(rest[:content_count])
    noise_palette = tuple(rest[content_count:]) or content_colors

    gap_width = int(rng.integers(4, 10))
    height, width, row_starts, left_cols = _sample_layout(rng, num_segments, gap_width)
    if max(height, width) > 30:
        return None

    row_order = [color for _, color in sorted(zip(row_starts, marker_colors))]
    chain_order = _sample_chain_order(rng, row_order)
    successor = {chain_order[i]: chain_order[i + 1] for i in range(len(chain_order) - 1)}

    cleaned_by_color: dict[int, np.ndarray] = {}
    full_by_color: dict[int, np.ndarray] = {}
    seen_clean = set()
    for color in marker_colors:
        for _ in range(200):
            stripe = _sample_clean_stripe(rng, gap_width, bg, content_colors)
            key = stripe.tobytes()
            if key not in seen_clean:
                seen_clean.add(key)
                cleaned_by_color[color] = stripe
                break
        else:
            return None

        full = cleaned_by_color[color].copy()
        if color in successor:
            slots = _pointer_slots(full, bg)
            if not slots:
                return None
            pr, pc = slots[int(rng.integers(0, len(slots)))]
            full[pr, pc] = successor[color]
        full_by_color[color] = full

    expected = np.vstack([cleaned_by_color[color] for color in chain_order])
    if max(expected.shape) > 30:
        return None

    grid = np.full((height, width), bg, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)
    for row, left, color in zip(row_starts, left_cols, marker_colors):
        right = left + MARKER_SIZE + gap_width
        grid[row : row + 2, left : left + 2] = color
        grid[row : row + 2, left + 2 : right] = full_by_color[color]
        grid[row : row + 2, right : right + 2] = color
        blocked[row : row + 2, left : right + 2] = True

    if noise_palette:
        _place_distractor_blocks(rng, grid, blocked, bg, int(noise_palette[0]))
        _place_noise(rng, grid, blocked, bg, noise_palette)

    counts = np.bincount(grid.ravel(), minlength=10)
    if counts[bg] <= max(counts[color] for color in range(10) if color != bg):
        return None

    spec = {
        "bg": bg,
        "chain_order": chain_order,
        "successor": successor,
        "cleaned_by_color": cleaned_by_color,
        "expected": expected,
    }
    return grid if _validate_generated(grid, spec) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_one(np.random.default_rng())
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pointer-chain stacking instance")
