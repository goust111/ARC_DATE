import random
from collections import Counter, defaultdict

import numpy as np


PALETTE = tuple(range(10))
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, exclude=()) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    colors -= set(map(int, exclude))
    return sorted(colors)


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_from_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _mono_components(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    return _components_from_mask(grid == int(color))


def _all_fg_components(grid: np.ndarray, bg: int) -> list[list[tuple[int, int]]]:
    return _components_from_mask(grid != int(bg))


def _row_pattern(left: int, right: int, left_len: int, total_len: int) -> tuple[int, ...]:
    return tuple([int(left)] * int(left_len) + [int(right)] * int(total_len - left_len))


def _bracket_info(grid: np.ndarray, bg: int, frame_color: int, comp: list[tuple[int, int]]):
    h, w = grid.shape
    r0, c0, r1, c1 = _bbox_from_coords(comp)
    if r0 != 0 or r1 + 1 >= h or c1 - c0 + 1 < 4:
        return None
    cells = set(comp)
    expected = {(0, c) for c in range(c0, c1 + 1)}
    for r in range(1, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))
    if cells != expected:
        return None
    left_tip = int(grid[r1 + 1, c0])
    right_tip = int(grid[r1 + 1, c1])
    if left_tip in (bg, frame_color) or right_tip in (bg, frame_color) or left_tip == right_tip:
        return None
    return {
        "bbox": (r0, c0, r1, c1),
        "frame_color": int(frame_color),
        "tip_row": int(r1 + 1),
        "inner_width": int(c1 - c0 - 1),
        "left_tip": left_tip,
        "right_tip": right_tip,
        "colorset": frozenset((left_tip, right_tip)),
        "key": (int(c1 - c0 - 1), frozenset((left_tip, right_tip))),
    }


def _extract_bar(grid: np.ndarray, bg: int, frame_color: int, comp: list[tuple[int, int]]):
    colors = {int(grid[r, c]) for r, c in comp}
    if frame_color in colors:
        return None
    r0, c0, r1, c1 = _bbox_from_coords(comp)
    if c0 != c1:
        return None
    length = r1 - r0 + 1
    if len(comp) != length or length < 2:
        return None
    ordered = []
    for r in range(r0, r1 + 1):
        if (r, c0) not in comp:
            return None
        ordered.append(int(grid[r, c0]))
    runs = []
    for color in ordered:
        if not runs or runs[-1][0] != color:
            runs.append([color, 1])
        else:
            runs[-1][1] += 1
    if len(runs) != 2 or runs[0][0] == runs[1][0]:
        return None
    return {
        "coords": comp,
        "bbox": (r0, c0, r1, c1),
        "col": int(c0),
        "bottom": int(r1),
        "length": int(length),
        "top_color": int(runs[0][0]),
        "top_len": int(runs[0][1]),
        "bottom_color": int(runs[1][0]),
        "bottom_len": int(runs[1][1]),
        "colorset": frozenset((int(runs[0][0]), int(runs[1][0]))),
        "key": (int(length), frozenset((int(runs[0][0]), int(runs[1][0])))),
    }


def _parse_scene(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    bracket_candidates = []
    for color in _colors_present(g, exclude={bg}):
        comps = _mono_components(g, color)
        infos = []
        ok = len(comps) >= 4
        for comp in comps:
            info = _bracket_info(g, bg, color, comp)
            if info is None:
                ok = False
                break
            infos.append(info)
        if ok:
            bracket_candidates.append((int(color), sorted(infos, key=lambda x: x["bbox"][1])))
    assert len(bracket_candidates) == 1, "expected one frame color forming all brackets"
    frame_color, brackets = bracket_candidates[0]
    assert len(brackets) == 4, "expected exactly four brackets"

    keys = [br["key"] for br in brackets]
    assert len(set(keys)) == 4, "bracket joint keys must be unique"

    bars = []
    for comp in _all_fg_components(g, bg):
        bar = _extract_bar(g, bg, frame_color, comp)
        if bar is not None:
            bars.append(bar)
    assert len(bars) >= 4, "expected several multicolor bars"

    bracket_by_key = {br["key"]: br for br in brackets}
    for bar in bars:
        assert bar["key"] in bracket_by_key, "each bar must match one bracket by length and colorset"

    return {
        "grid": g,
        "bg": int(bg),
        "frame_color": int(frame_color),
        "brackets": brackets,
        "bars": bars,
        "bracket_by_key": bracket_by_key,
    }


def _oriented_row(bar: dict, bracket: dict) -> np.ndarray:
    left = int(bracket["left_tip"])
    right = int(bracket["right_tip"])
    if (bar["top_color"], bar["bottom_color"]) == (left, right):
        left_len = int(bar["top_len"])
    elif (bar["top_color"], bar["bottom_color"]) == (right, left):
        left_len = int(bar["bottom_len"])
    else:
        raise AssertionError("bar colors must match bracket tips as a set")
    row = np.array(_row_pattern(left, right, left_len, int(bar["length"])), dtype=int)
    return row


def _simulate(scene: dict, *, descending: bool = True) -> np.ndarray:
    g = scene["grid"]
    out = g.copy()
    bg = int(scene["bg"])
    h, _w = g.shape

    for bar in scene["bars"]:
        for r, c in bar["coords"]:
            out[r, c] = bg

    stack_heights = defaultdict(int)
    bars = sorted(
        scene["bars"],
        key=lambda b: (-int(b["bottom"]), int(b["col"])) if descending else (int(b["bottom"]), int(b["col"])),
    )
    for bar in bars:
        bracket = scene["bracket_by_key"][bar["key"]]
        row = _oriented_row(bar, bracket)
        c0 = int(bracket["bbox"][1]) + 1
        landing_r = h - 1 - stack_heights[bar["key"]]
        out[landing_r, c0 : c0 + int(bar["length"])] = row
        stack_heights[bar["key"]] += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Match each vertical bicolor bar to its bracket by joint width-and-colorset, orient it by bracket tip order, then drop the bars in low-to-high source order.

    Input structure:
    - A dominant background contains four same-frame-color brackets attached to the top edge and open downward.
    - Each bracket has two colored tip labels, one at the end of each arm, and the pair's unordered color set may repeat across brackets.
    - Several separate vertical 1-cell-wide bicolor bars appear lower in the grid; each bar has two contiguous color runs and its total length equals one bracket interior width.

    Rule:
    - For every bar, identify the unique target bracket by the joint key `(bar length, unordered pair of bar colors)`.
    - Rotate the bar into a horizontal row using the only 90-degree direction whose left-right color order matches that bracket's left and right tip colors; preserve the exact split lengths of the two color runs.
    - Translate the row into the bracket's interior columns and let the bars settle one by one in order of source height from lower to higher, stacking on the bottom edge or on previously settled rows; keep the brackets unchanged.

    Key insight:
    - Length alone and color set alone are both ambiguous, so the bar first needs a joint-key match, then the bracket's ordered tip colors choose the rotation direction, and finally stack order matters because multiple bars can share a bracket but produce different horizontal color splits.
    """
    scene = _parse_scene(grid)
    return _simulate(scene, descending=True)


def _choose_layout():
    small, large = random.choice(((2, 4), (3, 5)))
    bracket_specs = [
        {"inner_width": small, "set_id": 0},
        {"inner_width": small, "set_id": 1},
        {"inner_width": large, "set_id": 0},
        {"inner_width": large, "set_id": 1},
    ]
    random.shuffle(bracket_specs)
    return small, large, bracket_specs


def _build_brackets(h: int, w: int, bg: int, frame_color: int, brackets: list[dict], arm_depth: int) -> np.ndarray:
    grid = np.full((h, w), bg, dtype=int)
    for br in brackets:
        total_w = br["inner_width"] + 2
        c0 = br["c0"]
        c1 = c0 + total_w - 1
        grid[0, c0 : c1 + 1] = frame_color
        for r in range(1, arm_depth):
            grid[r, c0] = frame_color
            grid[r, c1] = frame_color
        grid[arm_depth, c0] = br["left_tip"]
        grid[arm_depth, c1] = br["right_tip"]
    return grid


def _place_bars(grid: np.ndarray, bar_specs: list[dict], blocked_cols: set[int], top_start: int) -> bool:
    h, w = grid.shape
    used_bottoms = set()
    occupied = np.zeros_like(grid, dtype=bool)
    occupied[:top_start, :] = True
    available_cols = [c for c in range(w) if c not in blocked_cols]
    random.shuffle(bar_specs)
    for spec in sorted(bar_specs, key=lambda s: -s["length"]):
        placed = False
        if not available_cols:
            return False
        for _ in range(300):
            c = random.choice(available_cols)
            r0_min = top_start
            r0_max = h - spec["length"]
            if r0_min > r0_max:
                return False
            r0 = random.randint(r0_min, r0_max)
            r1 = r0 + spec["length"] - 1
            if r1 in used_bottoms:
                continue
            rr0 = max(0, r0 - 1)
            rr1 = min(h, r1 + 2)
            cc0 = max(0, c - 1)
            cc1 = min(w, c + 2)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            spec["top_row"] = int(r0)
            spec["col"] = int(c)
            used_bottoms.add(int(r1))
            occupied[r0 : r1 + 1, c] = True
            placed = True
            break
        if not placed:
            return False

    for spec in bar_specs:
        r0 = spec["top_row"]
        c = spec["col"]
        split = spec["top_len"]
        grid[r0 : r0 + split, c] = spec["top_color"]
        grid[r0 + split : r0 + spec["length"], c] = spec["bottom_color"]
    return True


def _make_bar_spec(bracket: dict, left_len: int, use_ccw: bool) -> dict:
    left = int(bracket["left_tip"])
    right = int(bracket["right_tip"])
    length = int(bracket["inner_width"])
    if use_ccw:
        top_color, top_len = left, left_len
        bottom_color = right
    else:
        top_color, top_len = right, length - left_len
        bottom_color = left
    return {
        "bracket_key": bracket["key"],
        "length": length,
        "left_len": int(left_len),
        "top_color": int(top_color),
        "top_len": int(top_len),
        "bottom_color": int(bottom_color),
        "branch": "ccw" if use_ccw else "cw",
    }


def _validate_semantics(grid: np.ndarray) -> None:
    scene = _parse_scene(grid)
    brackets = scene["brackets"]
    bars = scene["bars"]
    width_counts = Counter(int(br["inner_width"]) for br in brackets)
    set_counts = Counter(br["colorset"] for br in brackets)
    assert sorted(width_counts.values()) == [2, 2]
    assert sorted(set_counts.values()) == [2, 2]

    by_key = defaultdict(list)
    branches = set()
    for bar in bars:
        bracket = scene["bracket_by_key"][bar["key"]]
        same_width = sum(int(br["inner_width"]) == int(bracket["inner_width"]) for br in brackets)
        same_set = sum(br["colorset"] == bracket["colorset"] for br in brackets)
        assert same_width >= 2
        assert same_set >= 2
        by_key[bar["key"]].append(bar)
        if (bar["top_color"], bar["bottom_color"]) == (bracket["left_tip"], bracket["right_tip"]):
            branches.add("ccw")
        elif (bar["top_color"], bar["bottom_color"]) == (bracket["right_tip"], bracket["left_tip"]):
            branches.add("cw")
        else:
            raise AssertionError("bar colors must match target tip colors")
    assert len(branches) == 2, "need both rotation directions represented"
    assert all(len(v) >= 1 for v in by_key.values()), "each bracket should receive at least one bar"
    assert any(len(v) >= 2 for v in by_key.values()), "need at least one stacked bracket"
    assert len({int(bar["bottom"]) for bar in bars}) == len(bars), "source heights should be unambiguous"

    order_sensitive = False
    for key, group in by_key.items():
        bracket = scene["bracket_by_key"][key]
        patterns = {_row_pattern(bracket["left_tip"], bracket["right_tip"], len(np.where(_oriented_row(bar, bracket) == bracket["left_tip"])[0]), bar["length"]) for bar in group}
        if len(patterns) >= 2:
            order_sensitive = True
            break
    assert order_sensitive, "stack order must visibly matter for at least one bracket"
    assert not np.array_equal(_simulate(scene, descending=True), _simulate(scene, descending=False))


def _gen_attempt() -> np.ndarray | None:
    bg, frame_color, a, b, c, d = random.sample(PALETTE, 6)
    small, large, layout = _choose_layout()
    arm_depth = random.choice((2, 3))
    gaps = [random.randint(1, 2) for _ in range(3)]
    total_width = sum(spec["inner_width"] + 2 for spec in layout) + sum(gaps)
    if total_width > 30:
        return None
    w = random.randint(total_width, 30)
    h = random.randint(max(16, arm_depth + 9), min(30, arm_depth + 20))

    left_margin = random.randint(0, w - total_width)
    x = left_margin
    sets = {0: (a, b), 1: (c, d)}
    brackets = []
    for idx, spec in enumerate(layout):
        colors = sets[spec["set_id"]]
        if random.random() < 0.5:
            left_tip, right_tip = colors
        else:
            left_tip, right_tip = colors[::-1]
        br = {
            "inner_width": int(spec["inner_width"]),
            "set_id": int(spec["set_id"]),
            "left_tip": int(left_tip),
            "right_tip": int(right_tip),
            "c0": int(x),
        }
        br["key"] = (int(br["inner_width"]), frozenset((int(left_tip), int(right_tip))))
        brackets.append(br)
        x += br["inner_width"] + 2
        if idx < 3:
            x += gaps[idx]

    counts = {idx: 1 for idx in range(4)}
    large_indices = [i for i, br in enumerate(brackets) if br["inner_width"] == large]
    critical_idx = random.choice(large_indices)
    counts[critical_idx] += random.randint(1, 2)
    if random.random() < 0.45:
        extra_idx = random.choice([i for i in range(4) if i != critical_idx])
        counts[extra_idx] += 1

    bar_specs = []
    all_branches = []
    for idx, bracket in enumerate(brackets):
        needed = counts[idx]
        possible_left_lens = list(range(1, bracket["inner_width"]))
        if idx == critical_idx and needed >= 2:
            random.shuffle(possible_left_lens)
            chosen_left_lens = possible_left_lens[:needed]
        else:
            chosen_left_lens = [random.choice(possible_left_lens) for _ in range(needed)]
        for left_len in chosen_left_lens:
            use_ccw = random.random() < 0.5
            bar_specs.append(_make_bar_spec(bracket, left_len, use_ccw))
            all_branches.append("ccw" if use_ccw else "cw")
    if len(set(all_branches)) < 2:
        return None

    max_stack = max(counts.values())
    if h <= arm_depth + 2 + max_stack:
        return None

    grid = _build_brackets(h, w, bg, frame_color, brackets, arm_depth)
    blocked_cols = set()
    for br in brackets:
        for c0 in range(br["c0"] + 1, br["c0"] + 1 + br["inner_width"]):
            blocked_cols.add(int(c0))
    top_start = arm_depth + 2
    if not _place_bars(grid, bar_specs, blocked_cols, top_start):
        return None

    try:
        _validate_semantics(grid)
        out = main(grid)
        assert out.shape == grid.shape
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
