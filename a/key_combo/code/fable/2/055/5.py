from __future__ import annotations

import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import crop_bbox, find_components, get_bg, paste


BACKGROUND = 0
PALETTE = tuple(range(1, 10))
MAX_GEN_ATTEMPTS = 400


def _component_mask(grid: np.ndarray, comp: dict) -> np.ndarray:
    return crop_bbox((grid == int(comp["color"])).astype(np.uint8), comp["bbox"]).astype(bool)


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return mask.shape, mask.astype(np.uint8).tobytes()


def _analyze_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    components = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    assert components, "expected at least one foreground component"

    singleton_counts: Counter[int] = Counter()
    scatter_coords: list[tuple[int, int]] = []
    sprite_candidates: list[dict] = []
    for comp in components:
        color = int(comp["color"])
        if int(comp["area"]) == 1:
            singleton_counts[color] += 1
            r, c = map(int, comp["coords"][0])
            scatter_coords.append((r, c))
        else:
            sprite_candidates.append(comp)

    assert singleton_counts, "expected scattered single-pixel legend points"
    assert len(set(singleton_counts.values())) == len(singleton_counts), "scatter counts must be unique"

    non_bg_colors = {int(v) for v in np.unique(g) if int(v) != bg}
    scatter_colors = set(singleton_counts)
    assert scatter_colors == non_bg_colors, "scatter colors must name the complete stack palette"

    order = [color for color, _count in sorted(singleton_counts.items(), key=lambda item: -item[1])]
    rank_by_color = {color: rank for rank, color in enumerate(order)}

    groups: dict[tuple[tuple[int, int], bytes], list[dict]] = defaultdict(list)
    for comp in sprite_candidates:
        assert int(comp["color"]) in scatter_colors, "sprite colors must also appear in the scatter legend"
        groups[_mask_key(_component_mask(g, comp))].append(comp)

    matching_groups = [group for group in groups.values() if len(group) >= 2]
    assert len(matching_groups) == 1, "expected one repeated multi-pixel sprite mask"
    sprite_group = matching_groups[0]
    assert len(sprite_group) == len(sprite_candidates), "all multi-pixel components must belong to the stack"

    sprite_mask = _component_mask(g, sprite_group[0])
    sprite_h, sprite_w = sprite_mask.shape
    placed = []
    for comp in sprite_group:
        mask = _component_mask(g, comp)
        assert mask.shape == sprite_mask.shape and np.array_equal(mask, sprite_mask)
        color = int(comp["color"])
        rank = int(rank_by_color[color])
        r0, c0, r1, c1 = map(int, comp["bbox"])
        placed.append({"color": color, "rank": rank, "top": r0, "left": c0, "bbox": (r0, c0, r1, c1)})

    assert len(placed) >= 2, "need at least two placed instances to infer the stack geometry"
    assert len({entry["color"] for entry in placed}) == len(placed), "a stack color may occupy at most one slot"
    assert len({entry["left"] for entry in placed}) == 1, "placed instances must share one left edge"

    placed.sort(key=lambda entry: (entry["rank"], entry["top"]))
    stride_values = set()
    for i, first in enumerate(placed):
        for second in placed[i + 1 :]:
            delta_rank = second["rank"] - first["rank"]
            delta_row = second["top"] - first["top"]
            assert delta_rank > 0 and delta_row > 0
            assert delta_row % delta_rank == 0, "slot spacing must be divisible by the rank gap"
            stride_values.add(delta_row // delta_rank)
    assert len(stride_values) == 1, "all placed instances must agree on one vertical stride"
    stride = stride_values.pop()
    assert stride >= sprite_h, "slots may not overlap"

    stack_top_values = {entry["top"] - entry["rank"] * stride for entry in placed}
    assert len(stack_top_values) == 1, "placed instances must agree on one stack anchor"
    stack_top = stack_top_values.pop()
    stack_left = placed[0]["left"]

    h, w = g.shape
    n_slots = len(order)
    assert 0 <= stack_top
    assert stack_top + (n_slots - 1) * stride + sprite_h <= h
    assert 0 <= stack_left and stack_left + sprite_w <= w

    placed_ranks = sorted(entry["rank"] for entry in placed)
    missing_ranks = sorted(set(range(n_slots)) - set(placed_ranks))

    return {
        "bg": bg,
        "order": order,
        "rank_by_color": rank_by_color,
        "scatter_counts": dict(singleton_counts),
        "scatter_coords": scatter_coords,
        "sprite_mask": sprite_mask,
        "sprite_h": sprite_h,
        "sprite_w": sprite_w,
        "placed": placed,
        "placed_ranks": placed_ranks,
        "missing_ranks": missing_ranks,
        "stride": stride,
        "stack_top": stack_top,
        "stack_left": stack_left,
    }


def _render_solution(grid: np.ndarray, info: dict) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    bg = int(info["bg"])
    sprite_mask = info["sprite_mask"]
    sprite_h = int(info["sprite_h"])
    sprite_w = int(info["sprite_w"])
    stack_left = int(info["stack_left"])
    stride = int(info["stride"])

    for r, c in info["scatter_coords"]:
        out[int(r), int(c)] = bg

    patch = np.full((sprite_h, sprite_w), bg, dtype=int)
    for rank, color in enumerate(info["order"]):
        patch.fill(bg)
        patch[sprite_mask] = int(color)
        r0 = int(info["stack_top"] + rank * stride)
        out[r0 : r0 + sprite_h, stack_left : stack_left + sprite_w] = bg
        paste(out, patch, r0, stack_left, transparent=bg)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read scatter-point frequencies as a legend, fill the missing slots of one vertical sprite stack, and erase the noise.

    Input structure:
    - Black background with one repeated multi-pixel sprite mask stacked vertically on one side of the grid.
    - The placed sprite instances are left-aligned, equally spaced, and leave several empty stack slots.
    - Elsewhere, every stack color also appears as isolated single-pixel scatter points, with a unique count per color.

    Rule:
    - Count the isolated scatter pixels for each color and sort colors from most frequent to least frequent.
    - Use that frequency ranking as the stack order from top slot to bottom slot.
    - Infer the shared sprite mask, stack left edge, and vertical stride from the already placed instances, stamp the missing colors into their ranked slots, and erase all scatter points.

    Key insight:
    - The apparent noise is the legend: only the scatter frequencies reveal which color belongs to each missing slot, while the existing instances fix the stack geometry.
    """
    info = _analyze_input(np.asarray(grid, dtype=int))
    return _render_solution(grid, info)


def _gen_sprite_mask(height: int, width: int) -> np.ndarray | None:
    area_min = max(4, (height * width + 2) // 3)
    area_max = max(area_min, height * width - 1)
    for _ in range(200):
        target = random.randint(area_min, area_max)
        start = (random.randrange(height), random.randrange(width))
        cells = {start}
        while len(cells) < target:
            frontier = set()
            for r, c in cells:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < height and 0 <= cc < width and (rr, cc) not in cells:
                        frontier.add((rr, cc))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))

        mask = np.zeros((height, width), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if not mask[0].any() or not mask[-1].any() or not mask[:, 0].any() or not mask[:, -1].any():
            continue
        if mask.all():
            continue
        if int(mask.sum()) < 4:
            continue
        return mask
    return None


def _choose_stack_layout(n_colors: int) -> tuple[int, int, int] | None:
    candidates = []
    for sprite_h in range(3, 6):
        for sprite_w in range(3, 7):
            for gap in range(0, 3):
                stack_h = sprite_h + (n_colors - 1) * (sprite_h + gap)
                if stack_h <= 24:
                    candidates.append((sprite_h, sprite_w, gap))
    if not candidates:
        return None
    return random.choice(candidates)


def _choose_kept_ranks(n_colors: int) -> set[int] | None:
    all_ranks = list(range(n_colors))
    for _ in range(200):
        keep_count = random.randint(2, n_colors - 2)
        kept = set(random.sample(all_ranks, keep_count))
        missing = sorted(set(all_ranks) - kept)
        kept_sorted = sorted(kept)
        if len(missing) < 2:
            continue
        if not any(kept_sorted[0] < rank < kept_sorted[-1] for rank in missing):
            continue
        if not any(b - a > 1 for a, b in zip(kept_sorted, kept_sorted[1:])):
            continue
        if not any(missing_rank < kept_rank for missing_rank in missing for kept_rank in kept_sorted):
            continue
        return kept
    return None


def _scatter_counts(n_colors: int) -> list[int]:
    start = random.randint(n_colors + 1, min(9, n_colors + 3))
    return [start - offset for offset in range(n_colors)]


def _stack_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _render_full_stack(
    colors: list[int],
    sprite_mask: np.ndarray,
    stack_top: int,
    stack_left: int,
    stride: int,
    grid_shape: tuple[int, int],
) -> np.ndarray:
    out = np.full(grid_shape, BACKGROUND, dtype=int)
    for rank, color in enumerate(colors):
        paste(out, _stack_patch(sprite_mask, color), stack_top + rank * stride, stack_left, transparent=BACKGROUND)
    return out


def _place_scatter_points(
    grid: np.ndarray,
    colors: list[int],
    counts: list[int],
    zone_cols: list[int],
) -> bool:
    h, w = grid.shape
    parity = random.randint(0, 1)
    candidates = [(r, c) for r in range(1, h - 1) for c in zone_cols if (r + c) % 2 == parity]
    total_points = sum(counts)
    if len(candidates) < total_points:
        return False

    chosen = random.sample(candidates, total_points)
    scatter_colors = []
    for color, count in zip(colors, counts):
        scatter_colors.extend([color] * count)
    random.shuffle(scatter_colors)

    for (r, c), color in zip(chosen, scatter_colors):
        grid[r, c] = int(color)
    return True


def _validate_case(inp: np.ndarray, out: np.ndarray, expected: dict) -> None:
    info = _analyze_input(inp)
    assert np.array_equal(main(inp), out)
    assert info["order"] == expected["colors"]
    assert info["stack_top"] == expected["stack_top"]
    assert info["stack_left"] == expected["stack_left"]
    assert info["stride"] == expected["stride"]
    assert info["placed_ranks"] == sorted(expected["kept_ranks"])
    assert info["missing_ranks"] == sorted(expected["missing_ranks"])
    assert len(info["missing_ranks"]) >= 2
    assert any(info["placed_ranks"][0] < rank < info["placed_ranks"][-1] for rank in info["missing_ranks"])
    assert any(b - a > 1 for a, b in zip(info["placed_ranks"], info["placed_ranks"][1:]))

    total_counts = Counter(int(v) for v in inp.ravel())
    total_order = [color for color, _count in sorted(((color, total_counts[color]) for color in info["order"]), key=lambda item: -item[1])]
    assert total_order != info["order"], "counting all pixels must not replace counting only scatter points"

    for rank, color in enumerate(expected["colors"]):
        r0 = expected["stack_top"] + rank * expected["stride"]
        patch = out[r0 : r0 + expected["sprite_mask"].shape[0], expected["stack_left"] : expected["stack_left"] + expected["sprite_mask"].shape[1]]
        assert np.all(patch[expected["sprite_mask"]] == color)
        assert np.all(patch[~expected["sprite_mask"]] == BACKGROUND)
    for r, c in info["scatter_coords"]:
        assert out[r, c] == BACKGROUND


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    n_colors = random.randint(4, 6)
    layout = _choose_stack_layout(n_colors)
    if layout is None:
        return None
    sprite_h, sprite_w, gap = layout
    sprite_mask = _gen_sprite_mask(sprite_h, sprite_w)
    if sprite_mask is None:
        return None

    kept_ranks = _choose_kept_ranks(n_colors)
    if kept_ranks is None:
        return None
    missing_ranks = sorted(set(range(n_colors)) - kept_ranks)
    colors = random.sample(PALETTE, n_colors)
    counts = _scatter_counts(n_colors)
    stride = sprite_h + gap

    stack_h = sprite_h + (n_colors - 1) * stride
    grid_h = random.randint(stack_h + 2, min(30, stack_h + 6))
    grid_w = random.randint(max(18, sprite_w + 12), 30)
    stack_top = random.randint(1, grid_h - stack_h - 1)

    stack_side = random.choice(("left", "right"))
    if stack_side == "left":
        stack_left = random.randint(1, 2)
        zone_cols = list(range(stack_left + sprite_w + 3, grid_w - 1))
    else:
        stack_left = random.randint(grid_w - sprite_w - 2, grid_w - sprite_w - 1)
        zone_cols = list(range(1, stack_left - 2))
    if len(zone_cols) < 8:
        return None

    out = _render_full_stack(colors, sprite_mask, stack_top, stack_left, stride, (grid_h, grid_w))
    inp = out.copy()
    for rank in missing_ranks:
        r0 = stack_top + rank * stride
        inp[r0 : r0 + sprite_h, stack_left : stack_left + sprite_w] = BACKGROUND
    if not _place_scatter_points(inp, colors, counts, zone_cols):
        return None
    if np.array_equal(inp, out):
        return None

    expected = {
        "colors": colors,
        "counts": counts,
        "kept_ranks": kept_ranks,
        "missing_ranks": missing_ranks,
        "sprite_mask": sprite_mask,
        "stack_top": stack_top,
        "stack_left": stack_left,
        "stride": stride,
    }
    try:
        _validate_case(inp, out, expected)
    except AssertionError:
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        attempt = _gen_attempt()
        if attempt is not None:
            return attempt[0]
    raise RuntimeError("failed to generate a valid input for puzzle 5")
