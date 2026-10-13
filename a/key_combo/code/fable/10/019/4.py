from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
NON_BLACK = tuple(range(1, 10))
GEN_RETRIES = 300
MIN_CENTER_GAP = 4


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _palette(patch: np.ndarray) -> set[int]:
    return {int(color) for color in np.unique(patch) if int(color) != BLACK}


def _find_squares(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    squares: list[dict] = []
    for coords in _find_components(g != BLACK):
        if coords.shape[0] != 9:
            continue
        r0, c0, r1, c1 = _bbox(coords)
        if r1 - r0 != 2 or c1 - c0 != 2:
            continue
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(patch == BLACK):
            continue
        squares.append(
            {
                "bbox": (r0, c0, r1, c1),
                "center": (r0 + 1, c0 + 1),
                "patch": patch.copy(),
            }
        )
    squares.sort(key=lambda sq: (sq["center"][0], sq["center"][1]))
    return squares


def _corridor_clear(grid: np.ndarray, a: dict, b: dict) -> tuple[bool, str | None]:
    (ra, ca), (rb, cb) = a["center"], b["center"]
    if ra == rb:
        c0, c1 = sorted((ca, cb))
        if c1 - c0 < 4:
            return False, None
        return bool(np.all(grid[ra, c0 + 2 : c1 - 1] == BLACK)), "horizontal"
    if ca == cb:
        r0, r1 = sorted((ra, rb))
        if r1 - r0 < 4:
            return False, None
        return bool(np.all(grid[r0 + 2 : r1 - 1, ca] == BLACK)), "vertical"
    return False, None


def _eligible_pairs(grid: np.ndarray, squares: list[dict]) -> list[dict]:
    pairs: list[dict] = []
    for i in range(len(squares)):
        for j in range(i + 1, len(squares)):
            clear, orient = _corridor_clear(grid, squares[i], squares[j])
            if not clear:
                continue
            (ra, ca), (rb, cb) = squares[i]["center"], squares[j]["center"]
            if orient == "horizontal":
                c0, c1 = sorted((ca, cb))
                seg = (ra, c0 + 2, ra, c1 - 2)
            else:
                r0, r1 = sorted((ra, rb))
                seg = (r0 + 2, ca, r1 - 2, ca)
            pairs.append({"i": i, "j": j, "orient": orient, "segment": seg})
    return pairs


def _simulate_from_grid(grid: np.ndarray) -> tuple[list[np.ndarray], list[dict], list[dict], list[list[tuple[int, int, int]]]]:
    squares = _find_squares(grid)
    assert squares, "expected at least one 3x3 multicolor square"
    current = [sq["patch"].copy() for sq in squares]
    pairs = _eligible_pairs(np.asarray(grid, dtype=int), squares)
    rounds: list[list[tuple[int, int, int]]] = []
    horizontal_segments: list[dict] = []
    vertical_segments: list[dict] = []

    while True:
        palettes = [_palette(patch) for patch in current]
        active: list[tuple[int, int, int]] = []
        for pair in pairs:
            shared = palettes[pair["i"]] & palettes[pair["j"]]
            if len(shared) == 1:
                active.append((pair["i"], pair["j"], next(iter(shared))))
        if not active:
            break

        rounds.append(active)
        consumed = [set() for _ in current]
        for i, j, color in active:
            pair = next(p for p in pairs if p["i"] == i and p["j"] == j)
            record = {"color": int(color), "segment": pair["segment"]}
            if pair["orient"] == "horizontal":
                horizontal_segments.append(record)
            else:
                vertical_segments.append(record)
            consumed[i].add(int(color))
            consumed[j].add(int(color))

        for idx, colors in enumerate(consumed):
            for color in colors:
                current[idx][current[idx] == int(color)] = BLACK

    return current, horizontal_segments, vertical_segments, rounds


def _render_output(
    shape: tuple[int, int],
    squares: list[dict],
    current_patches: list[np.ndarray],
    horizontal_segments: list[dict],
    vertical_segments: list[dict],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for square, patch in zip(squares, current_patches):
        r0, c0, r1, c1 = square["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = patch
    for item in horizontal_segments:
        r0, c0, _r1, c1 = item["segment"]
        out[r0, c0 : c1 + 1] = int(item["color"])
    for item in vertical_segments:
        r0, c0, r1, _c1 = item["segment"]
        out[r0 : r1 + 1, c0] = int(item["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repeatedly bridge uniquely shared colors between aligned multicolor squares and consume that paint from both ends.

    Input structure:
    - The background is black.
    - The foreground consists of separated filled 3x3 multicolor squares, each using two to four non-black colors.
    - Some square centers share a row or a column, and the corridors between aligned squares are initially empty.

    Rule:
    - Repeatedly find every aligned square pair whose current color palettes intersect in exactly one color.
    - For each such pair, draw a one-cell-thick bridge of that shared color between the two facing square edges, then remove every cell of that color from both endpoint squares.
    - Continue until no aligned pair has a singleton palette intersection; if a horizontal bridge crosses a vertical bridge, keep the vertical color at the crossing.

    Key insight:
    - The shared color is both the bridge color and a consumable resource, so one bridge can shrink a square's palette and unlock a later bridge that was ambiguous before.
    """
    g = np.asarray(grid, dtype=int)
    squares = _find_squares(g)
    current, horizontal_segments, vertical_segments, _rounds = _simulate_from_grid(g)
    return _render_output(g.shape, squares, current, horizontal_segments, vertical_segments)


def _sample_spaced_positions(count: int, lo: int, hi: int, min_gap: int = MIN_CENTER_GAP) -> list[int]:
    assert hi - lo + 1 >= (count - 1) * min_gap + 1
    base = [lo + i * min_gap for i in range(count)]
    slack = hi - base[-1]
    extras = sorted(random.choices(range(slack + 1), k=count))
    return [base[i] + extras[i] for i in range(count)]


def _make_square_patch(colors: set[int], emphasized: set[int]) -> np.ndarray:
    counts = {int(color): (2 if int(color) in emphasized else 1) for color in colors}
    remaining = 9 - sum(counts.values())
    palette_list = sorted(colors)
    while remaining > 0:
        counts[random.choice(palette_list)] += 1
        remaining -= 1
    cells: list[int] = []
    for color, count in counts.items():
        cells.extend([int(color)] * int(count))
    random.shuffle(cells)
    return np.array(cells, dtype=int).reshape(3, 3)


def _template0(centers: dict[str, tuple[int, int]], colors: dict[str, int]) -> tuple[dict[str, set[int]], list[tuple[str, str]], list[list[tuple[str, str, int]]]]:
    palettes = {
        "L": {colors["a"], colors["b"], colors["e"]},
        "U": {colors["a"], colors["f"]},
        "R": {colors["a"], colors["b"], colors["g"]},
        "T": {colors["d"], colors["e"]},
        "B": {colors["d"], colors["f"]},
        "Z": {colors["g"], colors["e"]},
    }
    aligned_pairs = [("U", "L"), ("L", "R"), ("T", "B"), ("B", "Z")]
    rounds = [
        [("U", "L", colors["a"]), ("T", "B", colors["d"])],
        [("L", "R", colors["b"])],
    ]
    return palettes, aligned_pairs, rounds


def _template1(centers: dict[str, tuple[int, int]], colors: dict[str, int]) -> tuple[dict[str, set[int]], list[tuple[str, str]], list[list[tuple[str, str, int]]]]:
    palettes = {
        "A": {colors["a"], colors["b"], colors["e"]},
        "B": {colors["a"], colors["b"], colors["f"]},
        "C": {colors["a"], colors["g"]},
        "D": {colors["d"], colors["e"]},
        "E": {colors["d"], colors["f"]},
        "Z": {colors["g"], colors["b"]},
    }
    aligned_pairs = [("C", "A"), ("A", "B"), ("D", "E"), ("D", "Z")]
    rounds = [
        [("C", "A", colors["a"]), ("D", "E", colors["d"])],
        [("A", "B", colors["b"])],
    ]
    return palettes, aligned_pairs, rounds


def _build_instance() -> dict:
    template_id = random.randint(0, 1)
    height = random.randint(17, 30)
    width = random.randint(17, 30)
    rows = _sample_spaced_positions(4, 2, height - 3)
    cols = _sample_spaced_positions(4, 2, width - 3)
    role_values = random.sample(list(NON_BLACK), 9)
    color_map = {name: role_values[i] for i, name in enumerate(("a", "b", "d", "e", "f", "g"))}
    private_colors = role_values[6:]

    if template_id == 0:
        centers = {
            "U": (rows[0], cols[0]),
            "L": (rows[2], cols[0]),
            "T": (rows[1], cols[1]),
            "B": (rows[3], cols[1]),
            "R": (rows[2], cols[2]),
            "Z": (rows[3], cols[3]),
        }
        palettes, aligned_pairs, rounds = _template0(centers, color_map)
    else:
        centers = {
            "C": (rows[0], cols[0]),
            "A": (rows[0], cols[1]),
            "D": (rows[1], cols[0]),
            "E": (rows[1], cols[2]),
            "Z": (rows[3], cols[0]),
            "B": (rows[3], cols[1]),
        }
        palettes, aligned_pairs, rounds = _template1(centers, color_map)

    candidates = [label for label, palette in palettes.items() if len(palette) < 4]
    random.shuffle(candidates)
    bonus_count = random.randint(0, min(3, len(candidates), len(private_colors)))
    for label, bonus in zip(candidates[:bonus_count], private_colors[:bonus_count]):
        palettes[label] = set(palettes[label])
        palettes[label].add(int(bonus))

    bridge_colors: dict[str, set[int]] = {label: set() for label in palettes}
    for round_edges in rounds:
        for left, right, color in round_edges:
            bridge_colors[left].add(int(color))
            bridge_colors[right].add(int(color))

    patches = {
        label: _make_square_patch(palettes[label], bridge_colors[label]) for label in palettes
    }

    grid = np.zeros((height, width), dtype=int)
    for label, (r, c) in centers.items():
        grid[r - 1 : r + 2, c - 1 : c + 2] = patches[label]

    return {
        "grid": grid,
        "centers": centers,
        "palettes": palettes,
        "patches": patches,
        "aligned_pairs": aligned_pairs,
        "rounds": rounds,
    }


def _pair_kind(shared_size: int) -> str:
    if shared_size == 0:
        return "zero"
    if shared_size == 1:
        return "one"
    if shared_size == 2:
        return "two"
    return "many"


def _validate_schedule(instance: dict) -> None:
    current = {label: patch.copy() for label, patch in instance["patches"].items()}
    expected_rounds = [
        {tuple(sorted((left, right))): int(color) for left, right, color in round_edges}
        for round_edges in instance["rounds"]
    ]

    for round_index, expected in enumerate(expected_rounds):
        actual: dict[tuple[str, str], int] = {}
        for left, right in instance["aligned_pairs"]:
            shared = _palette(current[left]) & _palette(current[right])
            if len(shared) == 1:
                actual[tuple(sorted((left, right)))] = int(next(iter(shared)))
        assert actual == expected, (round_index, actual, expected)
        for pair, color in actual.items():
            left, right = pair
            current[left][current[left] == color] = BLACK
            current[right][current[right] == color] = BLACK

    trailing: dict[tuple[str, str], int] = {}
    for left, right in instance["aligned_pairs"]:
        shared = _palette(current[left]) & _palette(current[right])
        if len(shared) == 1:
            trailing[tuple(sorted((left, right)))] = int(next(iter(shared)))
    assert not trailing, trailing


def _oracle_output(instance: dict) -> np.ndarray:
    current = {label: patch.copy() for label, patch in instance["patches"].items()}
    horizontal_segments: list[tuple[int, int, int, int, int]] = []
    vertical_segments: list[tuple[int, int, int, int, int]] = []

    for round_edges in instance["rounds"]:
        consumed: dict[str, set[int]] = {label: set() for label in current}
        for left, right, color in round_edges:
            ra, ca = instance["centers"][left]
            rb, cb = instance["centers"][right]
            if ra == rb:
                c0, c1 = sorted((ca, cb))
                horizontal_segments.append((ra, c0 + 2, ra, c1 - 2, int(color)))
            else:
                r0, r1 = sorted((ra, rb))
                vertical_segments.append((r0 + 2, ca, r1 - 2, ca, int(color)))
            consumed[left].add(int(color))
            consumed[right].add(int(color))
        for label, colors in consumed.items():
            for color in colors:
                current[label][current[label] == color] = BLACK

    out = np.zeros_like(instance["grid"])
    for label, patch in current.items():
        r, c = instance["centers"][label]
        out[r - 1 : r + 2, c - 1 : c + 2] = patch
    for r0, c0, _r1, c1, color in horizontal_segments:
        out[r0, c0 : c1 + 1] = int(color)
    for r0, c0, r1, _c1, color in vertical_segments:
        out[r0 : r1 + 1, c0] = int(color)
    return out


def _crossing_count(instance: dict) -> int:
    horizontals: list[tuple[int, int, int]] = []
    verticals: list[tuple[int, int, int]] = []
    for round_edges in instance["rounds"]:
        for left, right, _color in round_edges:
            ra, ca = instance["centers"][left]
            rb, cb = instance["centers"][right]
            if ra == rb:
                c0, c1 = sorted((ca, cb))
                horizontals.append((ra, c0 + 2, c1 - 2))
            else:
                r0, r1 = sorted((ra, rb))
                verticals.append((ca, r0 + 2, r1 - 2))
    total = 0
    for row, c0, c1 in horizontals:
        for col, r0, r1 in verticals:
            if c0 <= col <= c1 and r0 <= row <= r1:
                total += 1
    return total


def _validate_instance(instance: dict) -> bool:
    grid = instance["grid"]
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    try:
        _validate_schedule(instance)
    except AssertionError:
        return False

    initial_pair_types = {}
    for left, right in instance["aligned_pairs"]:
        shared = instance["palettes"][left] & instance["palettes"][right]
        initial_pair_types[tuple(sorted((left, right)))] = _pair_kind(len(shared))
    if "zero" not in initial_pair_types.values() or "two" not in initial_pair_types.values():
        return False

    if _crossing_count(instance) < 1:
        return False

    expected = _oracle_output(instance)
    if np.array_equal(expected, grid):
        return False

    squares = _find_squares(grid)
    if len(squares) != len(instance["centers"]):
        return False
    by_center = {sq["center"]: sq for sq in squares}
    if set(by_center) != set(instance["centers"].values()):
        return False

    actual_pairs = _eligible_pairs(grid, squares)
    if len(actual_pairs) != len(instance["aligned_pairs"]):
        return False

    expected_pairs = {tuple(sorted(pair)) for pair in instance["aligned_pairs"]}
    actual_pair_types = {}
    label_by_center = {center: label for label, center in instance["centers"].items()}
    for pair in actual_pairs:
        left = label_by_center[squares[pair["i"]]["center"]]
        right = label_by_center[squares[pair["j"]]["center"]]
        names = tuple(sorted((left, right)))
        shared = _palette(by_center[instance["centers"][left]]["patch"]) & _palette(
            by_center[instance["centers"][right]]["patch"]
        )
        actual_pair_types[names] = _pair_kind(len(shared))
    if set(actual_pair_types) != expected_pairs:
        return False
    if any(actual_pair_types[pair] != initial_pair_types[pair] for pair in expected_pairs):
        return False

    solved = main(grid)
    if not np.array_equal(solved, expected):
        return False

    current, horizontal_segments, vertical_segments, rounds = _simulate_from_grid(grid)
    if len(rounds) != len(instance["rounds"]):
        return False
    if len(horizontal_segments) == 0 or len(vertical_segments) == 0:
        return False
    reproduced = _render_output(grid.shape, squares, current, horizontal_segments, vertical_segments)
    return np.array_equal(reproduced, expected)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        instance = _build_instance()
        if _validate_instance(instance):
            return instance["grid"]
    raise RuntimeError("failed to generate a valid input for消耗之桥")
