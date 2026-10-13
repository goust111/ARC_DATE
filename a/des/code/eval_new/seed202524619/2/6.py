import random
from itertools import combinations

import numpy as np


_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_BG_COLOR = 0
_BASE_COLOR = 5
_PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]


def _pair_key(a: int, b: int) -> tuple[int, int]:
    a = int(a)
    b = int(b)
    return (a, b) if a < b else (b, a)


def _most_common_color(g: np.ndarray) -> int:
    values, counts = np.unique(g, return_counts=True)
    return int(values[np.argmax(counts)])


def _parse_legend(bottom_row: np.ndarray, bg: int) -> dict[tuple[int, int], int]:
    row = np.asarray(bottom_row, dtype=int)
    w = int(row.size)
    mapping: dict[tuple[int, int], int] = {}

    c = 0
    while c < w:
        if int(row[c]) == bg:
            c += 1
            continue

        start = c
        while c < w and int(row[c]) != bg:
            c += 1
        run_len = c - start
        if run_len % 3 != 0:
            raise AssertionError("legend must contain only 1x3 color bars")

        for off in range(start, c, 3):
            a = int(row[off])
            mixed = int(row[off + 1])
            b = int(row[off + 2])
            if a == b:
                raise AssertionError("legend bar endpoints must be two different colors")
            if mixed == bg:
                raise AssertionError("legend middle cell cannot be background")

            key = _pair_key(a, b)
            if key in mapping:
                assert mapping[key] == mixed
            mapping[key] = mixed

    assert mapping, "legend must encode at least one color pair"
    return mapping


def _find_towers(play: np.ndarray, bg: int) -> list[dict[str, int]]:
    h, w = play.shape
    towers: list[dict[str, int]] = []

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            center = int(play[r, c])
            if center == bg:
                continue

            patch = play[r - 1 : r + 2, c - 1 : c + 2]
            if int(patch[1, 1]) == int(patch[0, 0]):
                continue
            base = int(patch[0, 0])
            mask = np.ones((3, 3), dtype=bool)
            mask[1, 1] = False
            if np.any(patch[mask] != base):
                continue

            towers.append({"r": int(r), "c": int(c), "color": center, "base": base})

    for i in range(len(towers)):
        for j in range(i + 1, len(towers)):
            a = towers[i]
            b = towers[j]
            if abs(a["r"] - b["r"]) <= 2 and abs(a["c"] - b["c"]) <= 2:
                raise AssertionError("tower bases must not overlap")

    assert towers, "expected at least one signal tower"
    return towers


def _build_base_mask(h: int, w: int, towers: list[dict[str, int]]) -> tuple[np.ndarray, np.ndarray]:
    base = np.zeros((h, w), dtype=bool)
    owner = -np.ones((h, w), dtype=int)
    for idx, t in enumerate(towers):
        r = t["r"]
        c = t["c"]
        base[r - 1 : r + 2, c - 1 : c + 2] = True
        owner[r - 1 : r + 2, c - 1 : c + 2] = idx
    return base, owner


def _trace_lines(
    h: int, w: int, towers: list[dict[str, int]], base_mask: np.ndarray, base_owner: np.ndarray
) -> dict[tuple[int, int], set[int]]:
    hits: dict[tuple[int, int], set[int]] = {}
    for i, t in enumerate(towers):
        color = int(t["color"])
        sr = int(t["r"])
        sc = int(t["c"])
        for dr, dc in _DIRS:
            r = sr + dr
            c = sc + dc
            while 0 <= r < h and 0 <= c < w:
                if bool(base_mask[r, c]) and int(base_owner[r, c]) != i:
                    break
                hits.setdefault((int(r), int(c)), set()).add(color)
                r += dr
                c += dc
    return hits


def main(grid: np.ndarray) -> np.ndarray:
    """
    Generate signal intersections by projecting colored beams from each tower and applying legend-based color mixing at crossings.

    Input structure:
    - Input background is black.
    - The upper grid contains several signal towers: each is a 1x1 color pixel centered in a 3x3 grey base square.
    - The last row is a legend made of multiple 1x3 bars, where the two side cells are mixed input colors and the middle cell is the target mixed color.

    Rule:
    - Locate each tower center and its 3x3 base.
    - From every tower, emit four cardinal lines with the tower's color, moving through non-basement cells until hitting another base or the end of the playfield (excluding the legend row).
    - For each non-basement pixel touched by exactly one color, keep that color.
    - For pixels touched by exactly two different colors, replace them by the legend-mapped mixture color if a pair exists; otherwise keep background color.
    - Pixels with three or more touching colors stay background color.

    Key insight:
    - The legend is an explicit two-color -> mixed color table, so line cross points are not computed by arithmetic but by lookup.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D ARC grid"
    h, w = g.shape
    assert h >= 2 and w >= 3, "grid too small"

    bg = _most_common_color(g)
    legend = _parse_legend(g[h - 1, :], bg)
    play = g[: h - 1, :]
    h_play = h - 1

    towers = _find_towers(play, bg)
    base_mask, base_owner = _build_base_mask(h_play, w, towers)
    line_hits = _trace_lines(h_play, w, towers, base_mask, base_owner)

    out = g.copy()
    for (r, c), colors in line_hits.items():
        if len(colors) == 1:
            out[r, c] = next(iter(colors))
            continue
        if len(colors) == 2:
            left, right = sorted(colors)
            mixed = legend.get((left, right))
            out[r, c] = mixed if mixed is not None else bg
            continue
        out[r, c] = bg

    return out


def _pair_count(hits: dict[tuple[int, int], set[int]]) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for colors in hits.values():
        if len(colors) < 2:
            continue
        vals = sorted(colors)
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                pairs.add((int(vals[i]), int(vals[j])))
    return pairs


def _has_overlap(existing: list[tuple[int, int]], r: int, c: int) -> bool:
    for er, ec in existing:
        if abs(r - er) <= 2 and abs(c - ec) <= 2:
            return True
        if r == er and c == ec:
            return True
    return False


def _place_cross_pair(
    h: int,
    w: int,
    existing: list[dict[str, int]],
    color_a: int,
    color_b: int,
) -> list[dict[str, int]] | None:
    existing_centers = [(int(t["r"]), int(t["c"])) for t in existing]
    target_key = _pair_key(color_a, color_b)

    for _ in range(300):
        cross_r = random.randint(1, h - 2)
        cross_c = random.randint(1, w - 2)
        vert_r = random.randint(1, h - 2)
        hori_c = random.randint(1, w - 2)
        if vert_r == cross_r or hori_c == cross_c:
            continue

        cand = [
            {"r": int(vert_r), "c": int(cross_c), "color": int(color_a), "base": _BASE_COLOR},
            {"r": int(cross_r), "c": int(hori_c), "color": int(color_b), "base": _BASE_COLOR},
        ]

        if (
            any(_has_overlap(existing_centers, t["r"], t["c"]) for t in cand)
            or _has_overlap([(cand[0]["r"], cand[0]["c"])], cand[1]["r"], cand[1]["c"])
        ):
            continue

        all_t = existing + cand
        base, owner = _build_base_mask(h, w, all_t)
        hits = _trace_lines(h, w, all_t, base, owner)
        pairs = _pair_count(hits)
        if target_key in pairs:
            return cand

    return None


def _build_legend_row(
    w: int, legend_mapping: dict[tuple[int, int], int], bg: int
) -> np.ndarray:
    rows = [((a, b), m) for (a, b), m in legend_mapping.items()]
    n = len(rows)
    needed = 3 * n + max(0, n - 1)
    assert needed <= w, "legend does not fit into grid width"

    row = np.full(w, bg, dtype=int)
    start_min = 0
    start_max = w - needed
    offset = random.randint(0, start_max - start_min)

    for idx, ((a, b), m) in enumerate(rows):
        s = offset + idx * 4
        row[s] = int(a)
        row[s + 1] = int(m)
        row[s + 2] = int(b)

    return row


def generate_input() -> np.ndarray:
    for _ in range(800):
        play_h = random.randint(11, 27)
        play_w = random.randint(13, 29)
        if play_h + 1 > 30:
            continue

        # Choose tower colors and derive a legend that is intentionally incomplete.
        base_color_count = random.randint(4, min(5, len(_PALETTE)))
        tower_colors = random.sample(_PALETTE, base_color_count)
        all_pairs = [_pair_key(*p) for p in combinations(tower_colors, 2)]
        if len(all_pairs) < 2:
            continue

        legend_pair_count = random.randint(1, min(3, len(all_pairs) - 1))
        legend_pairs = random.sample(all_pairs, legend_pair_count)
        legend_map: dict[tuple[int, int], int] = {}
        for p in legend_pairs:
            a, b = int(p[0]), int(p[1])
            candidates = [c for c in _PALETTE if c not in (a, b, _BG_COLOR, _BASE_COLOR)]
            if not candidates:
                break
            legend_map[p] = random.choice(candidates)
        if len(legend_map) != len(legend_pairs):
            continue

        if not all_pairs:
            continue
        unknown_pairs = [p for p in all_pairs if p not in legend_map]
        if not unknown_pairs:
            continue

        known_pair = random.choice(list(legend_map.keys()))
        unknown_pair = random.choice(unknown_pairs)

        # Keep width large enough for the legend and forced crossings.
        needed_for_legend = 3 * len(legend_map) + max(0, len(legend_map) - 1)
        if play_w < max(9, needed_for_legend + 2):
            play_w = max(9, needed_for_legend + 2)

        towers: list[dict[str, int]] = []

        first_cross = _place_cross_pair(play_h, play_w, towers, known_pair[0], known_pair[1])
        if first_cross is None:
            continue
        towers.extend(first_cross)

        second_cross = _place_cross_pair(play_h, play_w, towers, unknown_pair[0], unknown_pair[1])
        if second_cross is None:
            continue
        towers.extend(second_cross)

        # Add more towers for visual richness.
        extra = random.randint(0, 3)
        for _ in range(extra):
            placed = False
            for _ in range(120):
                r = random.randint(1, play_h - 2)
                c = random.randint(1, play_w - 2)
                if _has_overlap([(t["r"], t["c"]) for t in towers], r, c):
                    continue
                color = random.choice(tower_colors)
                towers.append({"r": r, "c": c, "color": int(color), "base": _BASE_COLOR})
                placed = True
                break
            if not placed:
                break

        if len(towers) < 4:
            continue

        base, owner = _build_base_mask(play_h, play_w, towers)
        hits = _trace_lines(play_h, play_w, towers, base, owner)
        pair_hits = _pair_count(hits)
        if known_pair not in pair_hits or unknown_pair not in pair_hits:
            continue

        inp = np.full((play_h + 1, play_w), _BG_COLOR, dtype=int)
        for t in towers:
            r = t["r"]
            c = t["c"]
            inp[r - 1 : r + 2, c - 1 : c + 2] = _BASE_COLOR
            inp[r, c] = t["color"]

        inp[play_h, :] = _build_legend_row(play_w, legend_map, _BG_COLOR)

        out = main(inp)
        if out.shape != inp.shape:
            continue
        if np.max(out.shape) > 30:
            continue

        return inp

    raise RuntimeError("failed to generate a valid input sample for this puzzle")
