from __future__ import annotations

import random
from collections import Counter

import numpy as np


MARKER_RED = 2
MARKER_GREEN = 3
PERIOD = 4
BOTTOM_GAP = 1
LANE_WIDTH = 3
RED_COUNT_BOUNDS = (1, 4)
GREEN_INPUT_COUNT_BOUNDS = (2, 3)
GEN_RETRIES = 400

SHAPES: dict[str, np.ndarray] = {
    "tri": np.array(
        [
            [1, 0, 0],
            [1, 1, 0],
            [0, 0, 0],
        ],
        dtype=bool,
    ),
    "square": np.array(
        [
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 0],
        ],
        dtype=bool,
    ),
    "rect": np.array(
        [
            [1, 1, 1],
            [1, 1, 1],
            [0, 0, 0],
        ],
        dtype=bool,
    ),
    "ring": np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
}

AREAS = {name: int(mask.sum()) for name, mask in SHAPES.items()}


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _find_bars(grid: np.ndarray, color: int) -> list[dict]:
    bars = []
    for pts in _components(grid == color):
        r0, c0, r1, c1 = _bbox(pts)
        if min(r1 - r0 + 1, c1 - c0 + 1) != 1:
            continue
        bars.append(
            {
                "color": color,
                "pixels": set(pts),
                "bbox": (r0, c0, r1, c1),
                "center": (c0 + c1) / 2.0,
            }
        )
    bars.sort(key=lambda item: (item["center"], item["bbox"][0], item["bbox"][1]))
    return bars


def _lane_state(grid: np.ndarray, bar: dict, bg: int) -> dict:
    h, w = grid.shape
    r0, c0, r1, c1 = bar["bbox"]
    colmask = np.zeros((h, w), dtype=bool)
    colmask[:, c0 : c1 + 1] = True
    sprite_mask = (grid != bg) & colmask
    for rr, cc in bar["pixels"]:
        sprite_mask[rr, cc] = False

    comps = _components(sprite_mask)
    assert comps, "each marked lane must contain sprites"

    sprites = []
    for pts in comps:
        colors = {int(grid[rr, cc]) for rr, cc in pts}
        assert len(colors) == 1, "each sprite must be monochrome"
        sr0, sc0, sr1, sc1 = _bbox(pts)
        mask = np.zeros((sr1 - sr0 + 1, sc1 - sc0 + 1), dtype=bool)
        for rr, cc in pts:
            mask[rr - sr0, cc - sc0] = True
        sprites.append(
            {
                "coords": pts,
                "color": colors.pop(),
                "bbox": (sr0, sc0, sr1, sc1),
                "mask": mask,
                "area": len(pts),
            }
        )

    ref = sprites[0]
    template_mask = ref["mask"]
    template_area = ref["area"]
    template_color = ref["color"]
    left_anchor = ref["bbox"][1]
    top_rows = []
    for sprite in sprites:
        assert sprite["area"] == template_area
        assert sprite["color"] == template_color
        assert np.array_equal(sprite["mask"], template_mask)
        assert sprite["bbox"][1] == left_anchor
        top_rows.append(sprite["bbox"][0])
    top_rows.sort()

    if len(top_rows) >= 2:
        diffs = [b - a for a, b in zip(top_rows, top_rows[1:]) if b > a]
        period = diffs[0]
        assert all(diff == period for diff in diffs), "lane sprites must be evenly spaced"
    else:
        period = template_mask.shape[0] + 1

    offsets = [(rr - ref["bbox"][0], cc - ref["bbox"][1]) for rr, cc in ref["coords"]]
    return {
        "bar": bar,
        "center": bar["center"],
        "color": template_color,
        "area": template_area,
        "mask": template_mask,
        "offsets": offsets,
        "left": left_anchor,
        "count": len(sprites),
        "top_rows": top_rows,
        "occupied": set(top_rows),
        "period": period,
        "bottom_top": max(top_rows),
    }


def _parse_lanes(grid: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    bg = _get_bg(grid)
    reds = [_lane_state(grid, bar, bg) for bar in _find_bars(grid, MARKER_RED)]
    greens = [_lane_state(grid, bar, bg) for bar in _find_bars(grid, MARKER_GREEN)]
    assert reds and greens
    return bg, reds, greens


def _pair_reds_to_greens(reds: list[dict], greens: list[dict]) -> list[int]:
    pairing = []
    for red in reds:
        distances = [abs(red["center"] - green["center"]) for green in greens]
        best = min(distances)
        assert distances.count(best) == 1, "nearest green lane must be unique"
        pair = distances.index(best)
        pairing.append(pair)
    return pairing


def _place_sprite(grid: np.ndarray, top: int, left: int, offsets: list[tuple[int, int]], color: int) -> None:
    for dr, dc in offsets:
        grid[top + dr, left + dc] = color


def _solve(grid: np.ndarray, *, use_area: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg, reds, greens = _parse_lanes(g)
    out = g.copy()

    for lane in reds:
        for top in lane["top_rows"]:
            for dr, dc in lane["offsets"]:
                out[top + dr, lane["left"] + dc] = bg

    for bar in _find_bars(g, MARKER_RED) + _find_bars(g, MARKER_GREEN):
        for rr, cc in bar["pixels"]:
            out[rr, cc] = bg

    pairing = _pair_reds_to_greens(reds, greens)
    incoming = [0 for _ in greens]
    for red, green_idx in zip(reds, pairing):
        if use_area:
            incoming[green_idx] += red["count"] * red["area"]
        else:
            incoming[green_idx] += red["count"]

    for green, payload in zip(greens, incoming):
        if payload == 0:
            continue
        if use_area:
            assert payload % green["area"] == 0, "transferred area must match destination sprite area exactly"
            to_add = payload // green["area"]
        else:
            to_add = payload
        occupied = set(green["occupied"])
        placed = 0
        step = 0
        while placed < to_add:
            top = green["bottom_top"] - step * green["period"]
            step += 1
            if top in occupied:
                continue
            rows = [top + dr for dr, _ in green["offsets"]]
            cols = [green["left"] + dc for _, dc in green["offsets"]]
            if min(rows) < 0 or max(rows) >= out.shape[0] or min(cols) < 0 or max(cols) >= out.shape[1]:
                raise AssertionError("stacked destination sprite ran out of canvas")
            if any(int(out[rr, cc]) != bg for rr, cc in zip(rows, cols)):
                raise AssertionError("destination stack slot must be empty")
            _place_sprite(out, top, green["left"], green["offsets"], green["color"])
            occupied.add(top)
            placed += 1

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Exchange each red-lane stack's total covered area into the nearest green lane's own sprite units.

    Input structure:
    - The background is the most common color.
    - Thin red and green marker bars each define vertical lanes.
    - Every marked lane contains evenly stacked copies of one monochrome sprite; different lanes may use different shapes, colors, and single-sprite areas.

    Rule:
    - Remove every red-lane sprite stack entirely, and treat that lane's total covered area as transferable mass.
    - Send each red lane's area to the uniquely nearest green lane by horizontal lane-center distance.
    - In that green lane, keep the existing sprite shape and color, then add new copies upward on the same lattice until the added covered area exactly equals the transferred area.
    - Remove all marker bars.

    Key insight:
    - What is conserved is covered area, not sprite count; each green lane's own sprite area is the exchange rate that decides how many new copies it receives.
    """
    return _solve(grid, use_area=True)


def _stack_tops(count: int, height: int) -> list[int]:
    bottom_top = height - 5
    return [bottom_top - PERIOD * step for step in range(count - 1, -1, -1)]


def _capacity(height: int) -> int:
    tops = _stack_tops(1, height)
    bottom_top = tops[-1]
    return bottom_top // PERIOD + 1


def _paint_lane_stack(grid: np.ndarray, center: int, shape_name: str, color: int, count: int) -> None:
    left = center - 1
    mask = SHAPES[shape_name]
    for top in _stack_tops(count, grid.shape[0]):
        for rr, cc in np.argwhere(mask):
            grid[top + int(rr), left + int(cc)] = color


def _layout_scenario() -> tuple[str, list[str], dict[str, int], int]:
    scenario = random.choice(["split2", "split3", "same_middle"])
    if scenario == "split2":
        ordered = ["G0", "R0", "G1", "R1"]
        g0 = random.randint(4, 5)
        g1 = random.randint(g0 + 1, 6)
        g2 = random.randint(4, 6)
        first = random.randint(2, 5)
        gaps = [g0, g1, g2]
    elif scenario == "split3":
        ordered = ["G0", "R0", "G1", "R1", "G2"]
        g0 = random.randint(4, 5)
        g1 = random.randint(g0 + 1, 6)
        g2 = random.randint(4, 5)
        g3 = random.randint(g2 + 1, 6)
        first = random.randint(2, 3)
        gaps = [g0, g1, g2, g3]
    else:
        ordered = ["G0", "R0", "G1", "R1", "G2"]
        g1 = random.randint(4, 5)
        g0 = random.randint(g1 + 1, 6)
        g2 = random.randint(4, 5)
        g3 = random.randint(g2 + 1, 6)
        first = random.randint(2, 3)
        gaps = [g0, g1, g2, g3]

    if random.random() < 0.5:
        ordered = list(reversed(ordered))
        gaps = list(reversed(gaps))

    centers = [first]
    for gap in gaps:
        centers.append(centers[-1] + gap)
    width = centers[-1] + 3 + random.randint(0, 2)
    if width > 30:
        return _layout_scenario()
    return scenario, ordered, {label: center for label, center in zip(ordered, centers)}, width


def _candidate_red_specs(dest_area: int, max_red_count: int) -> list[dict]:
    candidates = []
    for shape_name, area in AREAS.items():
        for count in range(RED_COUNT_BOUNDS[0], min(RED_COUNT_BOUNDS[1], max_red_count) + 1):
            total = area * count
            if total % dest_area != 0:
                continue
            added = total // dest_area
            if added <= 0:
                continue
            candidates.append(
                {
                    "shape": shape_name,
                    "count": count,
                    "area": area,
                    "added": added,
                    "anti": added != count,
                }
            )
    return candidates


def _pick_specs(
    green_specs: dict[str, dict],
    mapping: dict[str, str],
    cap: int,
) -> dict[str, dict] | None:
    spare = {label: cap - spec["input_count"] for label, spec in green_specs.items()}
    cands0 = _candidate_red_specs(green_specs[mapping["R0"]]["area"], cap)
    cands1 = _candidate_red_specs(green_specs[mapping["R1"]]["area"], cap)
    pairs = []
    for spec0 in cands0:
        for spec1 in cands1:
            use = {label: 0 for label in green_specs}
            use[mapping["R0"]] += spec0["added"]
            use[mapping["R1"]] += spec1["added"]
            if any(use[label] > spare[label] for label in green_specs):
                continue
            if not (spec0["anti"] or spec1["anti"]):
                continue
            pair = {"R0": spec0, "R1": spec1}
            pairs.append(pair)
    if not pairs:
        return None
    pairs.sort(key=lambda pair: (pair["R0"]["anti"] + pair["R1"]["anti"], pair["R0"]["added"] + pair["R1"]["added"]))
    best_pairs = [pair for pair in pairs if pair["R0"]["anti"] + pair["R1"]["anti"] == pairs[-1]["R0"]["anti"] + pairs[-1]["R1"]["anti"]]
    return random.choice(best_pairs)


def _count_shortcut(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, use_area=False)


def _semantic_ok(inp: np.ndarray, out: np.ndarray, scenario: str) -> bool:
    try:
        bg, reds, greens = _parse_lanes(inp)
        pairing = _pair_reds_to_greens(reds, greens)
    except Exception:
        return False

    if len(greens) < 2 or len(reds) != 2:
        return False
    if len({lane["area"] for lane in reds + greens}) < 2:
        return False

    if scenario.startswith("split"):
        if pairing[0] == pairing[1]:
            return False
    else:
        green_order = sorted(range(len(greens)), key=lambda idx: greens[idx]["center"])
        middle = green_order[1]
        if pairing[0] != middle or pairing[1] != middle:
            return False

    try:
        if np.array_equal(_count_shortcut(inp), out):
            return False
    except Exception:
        pass

    transferred_counts = []
    for red, green_idx in zip(reds, pairing):
        transferred_counts.append((red["count"] * red["area"]) // greens[green_idx]["area"])
    if all(added == red["count"] for red, added in zip(reds, transferred_counts)):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    scenario, ordered, centers, width = _layout_scenario()
    height = random.randint(24, 30)
    cap = _capacity(height)

    bg = random.choice([0, 1, 4, 5, 6, 7, 8, 9])
    palette = [color for color in range(10) if color not in {bg, MARKER_RED, MARKER_GREEN}]

    green_labels = [label for label in ordered if label.startswith("G")]
    red_labels = [label for label in ordered if label.startswith("R")]
    random.shuffle(green_labels)
    green_shape_names = random.sample(list(SHAPES), len(green_labels))
    lane_colors = random.sample(palette, len(green_labels) + len(red_labels))

    green_specs: dict[str, dict] = {}
    for idx, label in enumerate(green_labels):
        green_specs[label] = {
            "shape": green_shape_names[idx],
            "area": AREAS[green_shape_names[idx]],
            "input_count": random.randint(GREEN_INPUT_COUNT_BOUNDS[0], min(GREEN_INPUT_COUNT_BOUNDS[1], cap - 1)),
            "color": lane_colors[idx],
        }

    green_items = sorted(((label, centers[label]) for label in green_labels), key=lambda item: item[1])
    mapping: dict[str, str] = {}
    for red in red_labels:
        rc = centers[red]
        dists = [abs(rc - center) for _, center in green_items]
        mapping[red] = green_items[dists.index(min(dists))][0]

    red_specs = _pick_specs(green_specs, mapping, cap)
    if red_specs is None:
        return None

    for idx, label in enumerate(red_labels, start=len(green_labels)):
        red_specs[label]["color"] = lane_colors[idx]

    incoming_counts = {label: 0 for label in green_labels}
    for red in red_labels:
        incoming_counts[mapping[red]] += red_specs[red]["added"]

    out = np.full((height, width), bg, dtype=int)
    for label in green_labels:
        total = green_specs[label]["input_count"] + incoming_counts[label]
        _paint_lane_stack(out, centers[label], green_specs[label]["shape"], green_specs[label]["color"], total)

    inp = out.copy()
    for label in green_labels:
        added = incoming_counts[label]
        if added == 0:
            continue
        left = centers[label] - 1
        mask = SHAPES[green_specs[label]["shape"]]
        total = green_specs[label]["input_count"] + added
        for top in _stack_tops(total, height)[:added]:
            for rr, cc in np.argwhere(mask):
                inp[top + int(rr), left + int(cc)] = bg

    for label in red_labels:
        _paint_lane_stack(inp, centers[label], red_specs[label]["shape"], red_specs[label]["color"], red_specs[label]["count"])

    for label in green_labels:
        center = centers[label]
        inp[-1, center - 1 : center + 2] = MARKER_GREEN
    for label in red_labels:
        center = centers[label]
        inp[-1, center - 1 : center + 2] = MARKER_RED

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    if not _semantic_ok(inp, out, scenario):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
