from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np


_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _UTILS_DIR.exists():
    _utils_path = str(_UTILS_DIR)
    if _utils_path not in sys.path:
        sys.path.append(_utils_path)

try:
    from arc_utils import Color
except Exception:
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        YELLOW = 4
        GREY = 5
        PINK = 6
        ORANGE = 7
        TEAL = 8
        MAROON = 9


BG = Color.BLACK
HUB = Color.GREY
PALETTE = [
    Color.BLUE,
    Color.RED,
    Color.GREEN,
    Color.YELLOW,
    Color.PINK,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
]

DIR_ORDER = ("up", "down", "left", "right")
DIR_DELTA = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

GEN_RETRIES = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _branch_coords(center: tuple[int, int], direction: str, length: int) -> list[tuple[int, int]]:
    cr, cc = center
    dr, dc = DIR_DELTA[direction]
    r = cr + 2 * dr
    c = cc + 2 * dc
    coords = []
    for _ in range(length):
        coords.append((r, c))
        r += dr
        c += dc
    return coords


def _find_hub_center(grid: np.ndarray) -> tuple[int, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[tuple[int, int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if not np.all(g[r - 1 : r + 2, c - 1 : c + 2] == HUB):
                continue
            branch_count = 0
            for dr, dc in DIR_DELTA.values():
                rr = r + 2 * dr
                cc = c + 2 * dc
                if _in_bounds(g.shape, rr, cc) and int(g[rr, cc]) != BG:
                    branch_count += 1
            if branch_count >= 2:
                candidates.append((r, c, branch_count))
    assert candidates, "expected a 3x3 gray hub"
    max_branches = max(count for _r, _c, count in candidates)
    best = [(r, c) for r, c, count in candidates if count == max_branches]
    assert len(best) == 1, "hub must be uniquely identifiable"
    return best[0]


def _parse_branches(grid: np.ndarray) -> tuple[tuple[int, int], list[dict]]:
    g = np.asarray(grid, dtype=int)
    center = _find_hub_center(g)
    cr, cc = center
    branches = []
    for direction in DIR_ORDER:
        dr, dc = DIR_DELTA[direction]
        r = cr + 2 * dr
        c = cc + 2 * dc
        coords = []
        while _in_bounds(g.shape, r, c) and int(g[r, c]) != BG:
            coords.append((r, c))
            r += dr
            c += dc
        if coords:
            colors = [int(g[r0, c0]) for r0, c0 in coords]
            branches.append(
                {
                    "dir": direction,
                    "coords": coords,
                    "colors": colors,
                }
            )
    assert 2 <= len(branches) <= 4, "expected several straight branches"
    return center, branches


def _complete_branches(branches: list[dict]) -> tuple[list[dict], list[int], str]:
    template_branches = [branch for branch in branches if HUB not in branch["colors"]]
    assert len(template_branches) == 1, "expected exactly one fully colored template branch"
    template = template_branches[0]
    pattern = template["colors"]
    assert len(pattern) >= 2, "template branch should have length >= 2"
    assert pattern[0] != pattern[1], "template prefix must expose both template colors"
    assert len(set(pattern)) == 2, "template branch should use exactly two colors"
    template_dir = str(template["dir"])

    completed = []
    for branch in branches:
        if branch["dir"] == template_dir:
            filled = list(pattern)
        else:
            prefix = branch["colors"][:2]
            assert len(prefix) == 2 and prefix[0] != prefix[1], "non-template branch prefix must show two colors"
            mapping = {
                int(pattern[0]): int(prefix[0]),
                int(pattern[1]): int(prefix[1]),
            }
            filled = [mapping[int(pattern[i % len(pattern)])] for i in range(len(branch["coords"]))]
        completed.append(
            {
                "dir": branch["dir"],
                "coords": branch["coords"],
                "colors": filled,
            }
        )
    return completed, pattern, template_dir


def _hub_winner(completed: list[dict]) -> tuple[int, bool, Counter, Counter]:
    adjacency = Counter(int(branch["colors"][0]) for branch in completed)
    max_adj = max(adjacency.values())
    candidates = [color for color, count in adjacency.items() if count == max_adj]
    totals = Counter(color for branch in completed for color in branch["colors"])
    used_tiebreak = len(candidates) > 1
    if used_tiebreak:
        max_total = max(totals[color] for color in candidates)
        candidates = [color for color in candidates if totals[color] == max_total]
    assert len(candidates) == 1, "hub color must be uniquely determined"
    return int(candidates[0]), used_tiebreak, adjacency, totals


def _render_output_from_branches(grid: np.ndarray, completed: list[dict], center: tuple[int, int]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for branch in completed:
        for (r, c), color in zip(branch["coords"], branch["colors"]):
            out[r, c] = int(color)
    out[center] = _hub_winner(completed)[0]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each branch of a hub-and-path network by transferring a two-color template prefix map, then recolor the hub center from the completed branch statistics.

    Input structure:
    - The background is black.
    - A solid 3x3 gray hub sits near the middle, with 3-4 straight one-cell-wide cardinal branches extending outward.
    - Exactly one branch is fully colored with a two-color template sequence; every other branch has only its two hub-nearest pixels colored and the rest still gray.

    Rule:
    - Identify the unique template branch: it is the only branch with no gray cells on it.
    - Read the template branch from the hub outward. For each incomplete branch, align its first two colored pixels with the first two template positions to define a color map from the template's two colors onto that branch.
    - Repeat the full template sequence as needed, apply the branch-specific map, and use it to recolor the whole branch from the hub outward.
    - After all branches are completed, recolor only the center pixel of the gray hub to the most common hub-adjacent branch color; if several colors tie there, choose the one with more total pixels across all completed branches.

    Key insight:
    - Each spatial branch acts like a directional row: the template branch supplies the reusable sequence, while every other branch reveals only a local two-pixel prefix map that must be propagated along that branch before the hub color can be decided.
    """
    g = np.asarray(grid, dtype=int)
    center, branches = _parse_branches(g)
    completed, _pattern, _template_dir = _complete_branches(branches)
    return _render_output_from_branches(g, completed, center)


def _make_template_pattern(a: int, b: int, template_len: int, max_other_len: int) -> list[int]:
    compare_len = max(4, max_other_len)
    alt = [a if i % 2 == 0 else b for i in range(compare_len)]
    for _ in range(200):
        pattern = [a, b]
        for _idx in range(2, template_len):
            pattern.append(random.choice((a, b)))
        repeated = [pattern[i % template_len] for i in range(compare_len)]
        if repeated == alt:
            continue
        return pattern
    raise RuntimeError("failed to generate a non-degenerate template pattern")


def _pick_lengths(branch_count: int) -> tuple[list[str], str, dict[str, int], bool]:
    directions = random.sample(list(DIR_ORDER), branch_count)
    template_dir = random.choice(directions)
    require_cycle = random.random() < 0.7
    if require_cycle:
        template_len = random.randint(4, 7)
    else:
        template_len = random.randint(4, 8)

    lengths = {template_dir: template_len}
    others = [direction for direction in directions if direction != template_dir]
    long_dir = random.choice(others) if require_cycle else None
    for direction in others:
        low = 4
        high = 8
        if direction == long_dir:
            low = template_len + 1
        if low > high:
            return _pick_lengths(branch_count)
        lengths[direction] = random.randint(low, high)
    return directions, template_dir, lengths, require_cycle


def _pick_pairs(remaining: list[int], other_dirs: list[str], mode: str) -> dict[str, tuple[int, int]]:
    colors = random.sample(remaining, len(remaining))
    count = len(other_dirs)
    pairs: list[tuple[int, int]]
    if mode == "majority":
        common = colors[0]
        if count == 2:
            pairs = [(common, colors[1]), (common, colors[2])]
        elif count == 3 and random.random() < 0.5:
            pairs = [(common, colors[1]), (common, colors[2]), (common, colors[3])]
        else:
            pairs = [(common, colors[1]), (common, colors[2]), (colors[3], colors[4])]
    else:
        firsts = colors[:count]
        seconds = colors[count : 2 * count]
        pairs = [(firsts[i], seconds[i]) for i in range(count)]
    random.shuffle(pairs)
    return {direction: pairs[idx] for idx, direction in enumerate(other_dirs)}


def _case_dimensions(lengths: dict[str, int]) -> tuple[int, int, tuple[int, int]]:
    top_len = lengths.get("up", 0)
    bottom_len = lengths.get("down", 0)
    left_len = lengths.get("left", 0)
    right_len = lengths.get("right", 0)

    occ_h = top_len + 3 + bottom_len
    occ_w = left_len + 3 + right_len
    min_h = max(13, occ_h)
    min_w = max(13, occ_w)
    max_h = 22
    max_w = 22
    h = random.randint(min_h, max_h)
    w = random.randint(min_w, max_w)
    center_r = random.randint(top_len + 1, h - bottom_len - 2)
    center_c = random.randint(left_len + 1, w - right_len - 2)
    return h, w, (center_r, center_c)


def _build_input(case: dict) -> np.ndarray:
    h, w = case["shape"]
    center = case["center"]
    grid = np.full((h, w), BG, dtype=int)
    cr, cc = center
    grid[cr - 1 : cr + 2, cc - 1 : cc + 2] = HUB

    pattern = case["pattern"]
    template_dir = case["template_dir"]
    for direction in case["directions"]:
        coords = _branch_coords(center, direction, case["lengths"][direction])
        if direction == template_dir:
            colors = list(pattern)
        else:
            prefix_a, prefix_b = case["pairs"][direction]
            colors = [prefix_a, prefix_b] + [HUB] * (len(coords) - 2)
        for (r, c), color in zip(coords, colors):
            grid[r, c] = int(color)
    return grid


def _case_completed_branches(case: dict, naive_alternation: bool = False) -> list[dict]:
    pattern = case["pattern"]
    center = case["center"]
    branches = []
    for direction in case["directions"]:
        coords = _branch_coords(center, direction, case["lengths"][direction])
        if direction == case["template_dir"]:
            colors = list(pattern)
        else:
            first_color, second_color = case["pairs"][direction]
            if naive_alternation:
                colors = [first_color if i % 2 == 0 else second_color for i in range(len(coords))]
            else:
                mapping = {
                    int(pattern[0]): int(first_color),
                    int(pattern[1]): int(second_color),
                }
                colors = [mapping[int(pattern[i % len(pattern)])] for i in range(len(coords))]
        branches.append({"dir": direction, "coords": coords, "colors": colors})
    return branches


def _build_output(case: dict, naive_alternation: bool = False) -> np.ndarray:
    inp = _build_input(case)
    completed = _case_completed_branches(case, naive_alternation=naive_alternation)
    return _render_output_from_branches(inp, completed, case["center"])


def _validate_case(case: dict) -> bool:
    inp = _build_input(case)
    try:
        expected = _build_output(case, naive_alternation=False)
        naive = _build_output(case, naive_alternation=True)
    except AssertionError:
        return False
    if np.array_equal(expected, naive):
        return False

    try:
        center, branches = _parse_branches(inp)
        completed, pattern, template_dir = _complete_branches(branches)
    except AssertionError:
        return False

    if center != case["center"]:
        return False
    if template_dir != case["template_dir"]:
        return False
    if list(pattern) != list(case["pattern"]):
        return False

    if not np.array_equal(main(inp), expected):
        return False

    winner, used_tiebreak, adjacency, totals = _hub_winner(completed)
    if winner != int(expected[case["center"]]):
        return False
    if case["mode"] == "majority" and used_tiebreak:
        return False
    if case["mode"] == "tiebreak" and not used_tiebreak:
        return False

    first_colors = [case["pattern"][0]]
    for direction in case["directions"]:
        if direction != case["template_dir"]:
            first_colors.append(case["pairs"][direction][0])
    if case["mode"] == "majority":
        if max(Counter(first_colors).values()) <= 1:
            return False
    else:
        if len(set(first_colors)) != len(first_colors):
            return False

    tied = [color for color, count in adjacency.items() if count == max(adjacency.values())]
    if used_tiebreak:
        best_total = max(totals[color] for color in tied)
        if sum(totals[color] == best_total for color in tied) != 1:
            return False

    return True


def _make_case(mode: str) -> dict:
    branch_count = random.choice((3, 4))
    directions, template_dir, lengths, _require_cycle = _pick_lengths(branch_count)
    template_a, template_b = random.sample(PALETTE, 2)
    remaining = [color for color in PALETTE if color not in (template_a, template_b)]
    other_dirs = [direction for direction in directions if direction != template_dir]
    pairs = _pick_pairs(remaining, other_dirs, mode)
    max_other_len = max(lengths[direction] for direction in other_dirs)
    pattern = _make_template_pattern(template_a, template_b, lengths[template_dir], max_other_len)
    h, w, center = _case_dimensions(lengths)
    return {
        "mode": mode,
        "shape": (h, w),
        "center": center,
        "directions": directions,
        "template_dir": template_dir,
        "lengths": lengths,
        "pattern": pattern,
        "pairs": pairs,
    }


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        mode = random.choice(("majority", "tiebreak"))
        case = _make_case(mode)
        if _validate_case(case):
            return _build_input(case)
    raise RuntimeError("failed to generate a valid hub-prefix ARC instance")
