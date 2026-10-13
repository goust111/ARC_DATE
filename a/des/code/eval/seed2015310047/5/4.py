from __future__ import annotations

import itertools
import random

import numpy as np


GREY = 5
COLORS = tuple(range(10))
DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


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
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in DIRS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _aligned(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] == b[0] or a[1] == b[1]


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _classify_pair(
    positions: list[tuple[int, int]], grey_positions: set[tuple[int, int]]
) -> dict[str, object] | None:
    if len(positions) != 2 or not _aligned(positions[0], positions[1]):
        return None

    options: list[dict[str, object]] = []
    for emitter, receiver in ((positions[0], positions[1]), (positions[1], positions[0])):
        grey_neighbors = [cell for cell in _neighbors4(emitter) if cell in grey_positions]
        if len(grey_neighbors) != 1:
            continue
        grey = grey_neighbors[0]
        dr = emitter[0] - grey[0]
        dc = emitter[1] - grey[1]
        if abs(dr) + abs(dc) != 1:
            continue
        if dr != 0:
            if receiver[1] != emitter[1]:
                continue
            ray_sign = 1 if receiver[0] > emitter[0] else -1 if receiver[0] < emitter[0] else 0
            if ray_sign != dr:
                continue
        else:
            if receiver[0] != emitter[0]:
                continue
            ray_sign = 1 if receiver[1] > emitter[1] else -1 if receiver[1] < emitter[1] else 0
            if ray_sign != dc:
                continue
        options.append(
            {
                "emitter": emitter,
                "receiver": receiver,
                "dir": (dr, dc),
                "grey": grey,
            }
        )
    if len(options) != 1:
        return None
    return options[0]


def _detect_scene(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    colors = [int(c) for c in np.unique(g) if int(c) != bg]

    singleton_positions: dict[int, list[tuple[int, int]]] = {}
    prisms: list[dict[str, object]] = []
    prism_lookup: dict[tuple[int, int], int] = {}

    for color in colors:
        comps = _components(g == color)
        if all(len(comp) == 1 for comp in comps):
            singleton_positions[color] = [comp[0] for comp in comps]
            continue

        if not all(len(comp) == 4 for comp in comps):
            raise AssertionError(f"color {color} is neither singleton markers nor 2x2 prisms")

        for comp in comps:
            r0, c0, r1, c1 = _bbox(comp)
            if (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
                raise AssertionError(f"color {color} is not a solid 2x2 prism")
            prism_id = len(prisms)
            prisms.append({"color": color, "bbox": (r0, c0, r1, c1)})
            for cell in comp:
                prism_lookup[cell] = prism_id

    singleton_colors = set(singleton_positions)
    pair_like_colors = {c for c, pts in singleton_positions.items() if len(pts) == 2 and _aligned(pts[0], pts[1])}
    candidates: list[tuple[int, dict[int, dict[str, object]]]] = []

    for grey_color in singleton_colors:
        grey_positions = set(singleton_positions[grey_color])
        pairs: dict[int, dict[str, object]] = {}
        used_greys: set[tuple[int, int]] = set()
        for color in sorted(pair_like_colors - {grey_color}):
            info = _classify_pair(singleton_positions[color], grey_positions)
            if info is None:
                continue
            pairs[color] = info
            used_greys.add(info["grey"])  # type: ignore[arg-type]

        if not pairs:
            continue
        if set(pairs) | {grey_color} != singleton_colors:
            continue
        if len(singleton_positions[grey_color]) != len(pairs):
            continue
        if len(used_greys) != len(pairs):
            continue
        candidates.append((grey_color, pairs))

    if len(candidates) != 1:
        raise AssertionError(f"unable to identify a unique grey indicator color: {len(candidates)} candidates")

    grey_color, pair_map = candidates[0]
    pairs_out = []
    for color in sorted(pair_map):
        info = dict(pair_map[color])
        info["color"] = color
        pairs_out.append(info)

    remaining_colors = set(colors) - {grey_color} - {int(pair["color"]) for pair in pairs_out}
    prism_colors = {int(prism["color"]) for prism in prisms}
    if remaining_colors != prism_colors:
        raise AssertionError("every non-bg non-grey non-pair color must be a prism color")

    return {
        "bg": bg,
        "grey": grey_color,
        "pairs": pairs_out,
        "prisms": prisms,
        "prism_lookup": prism_lookup,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace each beam from the grey-marked emitter to its same-colored receiver.
    Passing through a 2x2 prism changes the beam color to the prism color.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    scene = _detect_scene(g)
    bg = int(scene["bg"])
    grey = int(scene["grey"])
    prisms = scene["prisms"]
    prism_lookup = scene["prism_lookup"]

    out = g.copy()
    for pair in scene["pairs"]:
        emitter = pair["emitter"]  # type: ignore[assignment]
        receiver = pair["receiver"]  # type: ignore[assignment]
        dr, dc = pair["dir"]  # type: ignore[assignment]
        current_color = int(pair["color"])
        pos = emitter

        while True:
            nr = pos[0] + dr
            nc = pos[1] + dc
            if not (0 <= nr < h and 0 <= nc < w):
                raise AssertionError("beam left the grid before reaching its receiver")
            if (nr, nc) == receiver:
                final_color = current_color
                break
            prism_id = prism_lookup.get((nr, nc))
            if prism_id is not None:
                prism = prisms[prism_id]
                r0, c0, r1, c1 = prism["bbox"]  # type: ignore[assignment]
                if dr == 1:
                    pos = (r1, nc)
                elif dr == -1:
                    pos = (r0, nc)
                elif dc == 1:
                    pos = (nr, c1)
                else:
                    pos = (nr, c0)
                current_color = int(prism["color"])
                continue
            if out[nr, nc] == bg:
                out[nr, nc] = current_color
            pos = (nr, nc)

        rr, cc = receiver
        for ar, ac in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
            if 0 <= ar < h and 0 <= ac < w and out[ar, ac] == bg:
                out[ar, ac] = final_color

    out[g == grey] = bg
    return out


def _split_extent(total: int, gap: int, minimum: int) -> tuple[tuple[int, int], tuple[int, int]]:
    first = random.randint(minimum, total - gap - minimum)
    second = total - gap - first
    return (0, first - 1), (first + gap, total - 1)


def _choose_pair_count() -> int:
    return random.choice([2, 2, 3, 3, 4])


def _choose_prism_counts(n_pairs: int) -> list[int]:
    if n_pairs == 2:
        patterns = [(0, 1), (1, 1), (0, 2), (1, 2)]
    elif n_pairs == 3:
        patterns = [
            (0, 0, 1),
            (0, 1, 1),
            (0, 1, 2),
            (1, 1, 1),
            (0, 0, 2),
            (1, 1, 2),
        ]
    else:
        patterns = [
            (1, 0, 0, 0),
            (1, 1, 0, 0),
            (2, 0, 0, 0),
            (2, 1, 0, 0),
            (1, 1, 1, 0),
        ]
    counts = list(random.choice(patterns))
    random.shuffle(counts)
    return counts


def _sample_directions(n_pairs: int) -> list[str]:
    names = ["up", "down", "left", "right"]
    dirs = [random.choice(names) for _ in range(n_pairs)]
    if n_pairs >= 2:
        axes = {"h" if d in ("left", "right") else "v" for d in dirs}
        if len(axes) == 1:
            idx = random.randrange(n_pairs)
            dirs[idx] = random.choice(["up", "down"] if "h" in axes else ["left", "right"])
    return dirs


def _choose_segments(start_values: list[int], count: int) -> list[int] | None:
    if count == 0:
        return []
    combos = [
        list(combo)
        for combo in itertools.combinations(start_values, count)
        if all(combo[i + 1] - combo[i] >= 3 for i in range(len(combo) - 1))
    ]
    if not combos:
        return None
    return random.choice(combos)


def _place_beam(
    grid: np.ndarray,
    zone: tuple[int, int, int, int],
    direction: str,
    pair_color: int,
    prism_colors: list[int],
) -> None:
    r0, r1, c0, c1 = zone
    zone_h = r1 - r0 + 1
    zone_w = c1 - c0 + 1

    if direction in ("left", "right"):
        beam_r = r0 + random.randint(1, zone_h - 2)
        candidate_starts = list(range(c0 + 2, c1 - 2))
        prism_starts = _choose_segments(candidate_starts, len(prism_colors))
        if prism_starts is None:
            raise ValueError("horizontal zone is too narrow for the chosen prisms")

        if direction == "right":
            grey = (beam_r, c0)
            emitter = (beam_r, c0 + 1)
            receiver = (beam_r, c1 - 1)
        else:
            grey = (beam_r, c1)
            emitter = (beam_r, c1 - 1)
            receiver = (beam_r, c0 + 1)

        grid[grey] = GREY
        grid[emitter] = pair_color
        grid[receiver] = pair_color
        for prism_color, start_c in zip(prism_colors, prism_starts, strict=False):
            top_r = beam_r if random.choice([False, True]) else beam_r - 1
            grid[top_r : top_r + 2, start_c : start_c + 2] = prism_color
        return

    beam_c = c0 + random.randint(1, zone_w - 2)
    candidate_starts = list(range(r0 + 2, r1 - 2))
    prism_starts = _choose_segments(candidate_starts, len(prism_colors))
    if prism_starts is None:
        raise ValueError("vertical zone is too short for the chosen prisms")

    if direction == "down":
        grey = (r0, beam_c)
        emitter = (r0 + 1, beam_c)
        receiver = (r1 - 1, beam_c)
    else:
        grey = (r1, beam_c)
        emitter = (r1 - 1, beam_c)
        receiver = (r0 + 1, beam_c)

    grid[grey] = GREY
    grid[emitter] = pair_color
    grid[receiver] = pair_color
    for prism_color, start_r in zip(prism_colors, prism_starts, strict=False):
        left_c = beam_c if random.choice([False, True]) else beam_c - 1
        grid[start_r : start_r + 2, left_c : left_c + 2] = prism_color


def _make_zones(h: int, w: int) -> list[tuple[int, int, int, int]]:
    row_gap = 2
    col_gap = 2
    (r0a, r1a), (r0b, r1b) = _split_extent(h, row_gap, 10)
    (c0a, c1a), (c0b, c1b) = _split_extent(w, col_gap, 10)
    return [
        (r0a, r1a, c0a, c1a),
        (r0a, r1a, c0b, c1b),
        (r0b, r1b, c0a, c1a),
        (r0b, r1b, c0b, c1b),
    ]


def _validate_generated(grid: np.ndarray, expected_pairs: int, expected_prisms: int) -> None:
    scene = _detect_scene(grid)
    assert int(scene["grey"]) == GREY
    assert len(scene["pairs"]) == expected_pairs
    assert len(scene["prisms"]) == expected_prisms
    out = main(grid)
    assert isinstance(out, np.ndarray)
    assert out.shape == grid.shape
    assert np.any(out != grid)


def generate_input() -> np.ndarray:
    """
    Build a puzzle with 2-4 independent beam zones, mixed directions,
    and a varying number of prisms so the color-change branch appears often.
    """
    for _ in range(300):
        h = random.randint(24, 30)
        w = random.randint(24, 30)
        bg = random.choice([color for color in COLORS if color != GREY])
        n_pairs = _choose_pair_count()
        prism_counts = _choose_prism_counts(n_pairs)
        total_prisms = sum(prism_counts)

        usable_colors = [color for color in COLORS if color not in (bg, GREY)]
        if n_pairs + total_prisms > len(usable_colors):
            continue

        chosen = random.sample(usable_colors, n_pairs + total_prisms)
        pair_colors = chosen[:n_pairs]
        prism_palette = chosen[n_pairs:]
        zones = _make_zones(h, w)
        zone_indices = random.sample(range(4), n_pairs)
        directions = _sample_directions(n_pairs)

        grid = np.full((h, w), bg, dtype=int)
        prism_offset = 0
        try:
            for zone_idx, direction, pair_color, prism_count in zip(
                zone_indices, directions, pair_colors, prism_counts, strict=False
            ):
                prism_colors = prism_palette[prism_offset : prism_offset + prism_count]
                prism_offset += prism_count
                _place_beam(grid, zones[zone_idx], direction, pair_color, prism_colors)
        except ValueError:
            continue

        try:
            _validate_generated(grid, n_pairs, total_prisms)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
