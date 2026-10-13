from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG_COLOR = 0
DIR_VECS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
DIR_ORDER = ("up", "down", "left", "right")
GEN_RETRIES = 400
GRID_SIZE_RANGE = (15, 29)
TOWER_COUNT_CHOICES = (2, 3)
TOWER_COUNT_WEIGHTS = (1, 1)
MIN_CENTER_GAP = 5
CENTER_MARGIN = 2
PALETTE = tuple(range(1, 10))


def _background_color(grid: np.ndarray) -> int:
    corners = [
        int(grid[0, 0]),
        int(grid[0, -1]),
        int(grid[-1, 0]),
        int(grid[-1, -1]),
    ]
    return int(Counter(corners).most_common(1)[0][0])


def _tower_cells(tower: dict) -> list[tuple[int, int]]:
    r0 = int(tower["r0"])
    c0 = int(tower["c0"])
    return [(r, c) for r in range(r0, r0 + 3) for c in range(c0, c0 + 3)]


def _inside_tower(r: int, c: int, tower: dict) -> bool:
    return int(tower["r0"]) <= r <= int(tower["r0"]) + 2 and int(tower["c0"]) <= c <= int(tower["c0"]) + 2


def _find_towers(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    towers: list[dict] = []
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            col = int(g[r0, c0])
            if col == int(bg):
                continue
            patch = g[r0 : r0 + 3, c0 : c0 + 3]
            if not bool((patch == int(col)).all()):
                continue

            allowed = {
                (r0 - 1, c0 + 1),
                (r0 + 3, c0 + 1),
                (r0 + 1, c0 - 1),
                (r0 + 1, c0 + 3),
            }
            ok = True
            for rr in range(r0 - 1, r0 + 4):
                for cc in range(c0 - 1, c0 + 4):
                    if not (0 <= rr < h and 0 <= cc < w):
                        continue
                    if r0 <= rr <= r0 + 2 and c0 <= cc <= c0 + 2:
                        continue
                    if int(g[rr, cc]) == int(col) and (rr, cc) not in allowed:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue

            towers.append(
                {
                    "color": int(col),
                    "r0": int(r0),
                    "c0": int(c0),
                    "row": int(r0 + 1),
                    "col": int(c0 + 1),
                }
            )

    towers.sort(key=lambda t: (int(t["r0"]), int(t["c0"])))
    unique: list[dict] = []
    for tower in towers:
        if any(_inside_tower(int(tower["row"]), int(tower["col"]), other) for other in unique):
            continue
        unique.append(tower)

    colors = [int(t["color"]) for t in unique]
    assert len(unique) >= 2, "expected at least two towers"
    assert len(colors) == len(set(colors)), "tower colors must be distinct"
    return unique


def _build_tower_mask(shape: tuple[int, int], towers: list[dict]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for tower in towers:
        for r, c in _tower_cells(tower):
            mask[r, c] = True
    return mask


def _trace_ray(
    shape: tuple[int, int],
    tower_mask: np.ndarray,
    tower: dict,
    direction: str,
) -> list[tuple[int, int]]:
    h, w = shape
    dr, dc = DIR_VECS[direction]
    r = int(tower["row"]) + 2 * int(dr)
    c = int(tower["col"]) + 2 * int(dc)
    path: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w and not bool(tower_mask[r, c]):
        path.append((int(r), int(c)))
        r += int(dr)
        c += int(dc)
    return path


def _ray_coverage(
    shape: tuple[int, int],
    towers: list[dict],
    active_dirs: dict[tuple[int, str], bool] | None = None,
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], int]]:
    tower_mask = _build_tower_mask(shape, towers)
    horizontal: dict[tuple[int, int], int] = {}
    vertical: dict[tuple[int, int], int] = {}

    for idx, tower in enumerate(towers):
        color = int(tower["color"])
        for direction in DIR_ORDER:
            if active_dirs is not None and not active_dirs.get((int(idx), direction), False):
                continue
            path = _trace_ray(shape, tower_mask, tower, direction)
            target = horizontal if direction in ("left", "right") else vertical
            for cell in path:
                prev = target.get(cell)
                assert prev in (None, color), "same-orientation rays should not overlap"
                target[cell] = color
    return horizontal, vertical


def _infer_mix_map(
    grid: np.ndarray,
    towers: list[dict],
    horizontal: dict[tuple[int, int], int],
    vertical: dict[tuple[int, int], int],
    bg: int,
) -> dict[frozenset[int], int]:
    g = np.asarray(grid, dtype=int)
    tower_mask = _build_tower_mask(g.shape, towers)
    tower_colors = {int(t["color"]) for t in towers}
    mix_map: dict[frozenset[int], int] = {}

    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            color = int(g[r, c])
            if color == int(bg) or color in tower_colors or bool(tower_mask[r, c]):
                continue
            cell = (int(r), int(c))
            hcol = horizontal.get(cell)
            vcol = vertical.get(cell)
            assert hcol is not None and vcol is not None and hcol != vcol, "mixed cells must be orthogonal crossings"
            key = frozenset((int(hcol), int(vcol)))
            prev = mix_map.get(key)
            if prev is None:
                mix_map[key] = int(color)
            else:
                assert prev == int(color), "mix rule must be consistent"
    return mix_map


def _render(
    shape: tuple[int, int],
    towers: list[dict],
    mix_map: dict[frozenset[int], int],
    active_dirs: dict[tuple[int, str], bool] | None = None,
) -> np.ndarray:
    out = np.full(shape, BG_COLOR, dtype=int)
    for tower in towers:
        r0 = int(tower["r0"])
        c0 = int(tower["c0"])
        out[r0 : r0 + 3, c0 : c0 + 3] = int(tower["color"])

    horizontal, vertical = _ray_coverage(shape, towers, active_dirs)
    for cell in set(horizontal) | set(vertical):
        r, c = cell
        if cell in horizontal and cell in vertical:
            key = frozenset((int(horizontal[cell]), int(vertical[cell])))
            assert key in mix_map, "missing learned mix color for an output crossing"
            out[r, c] = int(mix_map[key])
        elif cell in horizontal:
            out[r, c] = int(horizontal[cell])
        else:
            out[r, c] = int(vertical[cell])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete every signal-tower ray and fill all orthogonal crossings with the learned mix colors.

    Input structure:
    - The background is black.
    - Several distinct colors form solid 3x3 tower blocks.
    - From each tower's side midpoint, horizontal and vertical 1-cell-wide rays may already be drawn or may be missing.
    - Some existing orthogonal ray crossings are recolored with a learned mixed color that is neither tower body nor background.

    Rule:
    - Identify every 3x3 tower and reconstruct its four rays so they extend straight outward until the grid edge or another tower block.
    - Use the already-drawn mixed crossing pixels to infer the fixed color mapping for each unordered pair of tower colors.
    - Repaint the full completed ray system, keeping tower bodies unchanged and applying the learned mixed color wherever a horizontal and a vertical ray of different towers cross.

    Key insight:
    - Tower positions alone determine the complete ray geometry; the input crossings are only needed to learn the pairwise color-mixing table that must be reused at newly created intersections.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    towers = _find_towers(g, bg=bg)
    horizontal, vertical = _ray_coverage(g.shape, towers)
    mix_map = _infer_mix_map(g, towers, horizontal, vertical, bg=bg)
    return _render(g.shape, towers, mix_map)


def _sample_centers(rng: random.Random, length: int, count: int) -> list[int] | None:
    candidates = list(range(CENTER_MARGIN, length - CENTER_MARGIN))
    if len(candidates) < count:
        return None
    for _ in range(200):
        chosen = sorted(rng.sample(candidates, count))
        if all(b - a >= MIN_CENTER_GAP for a, b in zip(chosen, chosen[1:])):
            return chosen
    return None


def _pair_descriptors(towers: list[dict]) -> list[dict]:
    desc: list[dict] = []
    for i in range(len(towers)):
        for j in range(i + 1, len(towers)):
            ti = towers[i]
            tj = towers[j]
            idir = "right" if int(tj["col"]) > int(ti["col"]) else "left"
            jdir = "right" if int(ti["col"]) > int(tj["col"]) else "left"
            iupdown = "down" if int(tj["row"]) > int(ti["row"]) else "up"
            jupdown = "down" if int(ti["row"]) > int(tj["row"]) else "up"
            desc.append(
                {
                    "pair": frozenset((int(ti["color"]), int(tj["color"]))),
                    "crossings": [
                        {
                            "cell": (int(ti["row"]), int(tj["col"])),
                            "req": ((int(i), idir), (int(j), jupdown)),
                        },
                        {
                            "cell": (int(tj["row"]), int(ti["col"])),
                            "req": ((int(j), jdir), (int(i), iupdown)),
                        },
                    ],
                }
            )
    return desc


def _visible_crossings(pair_desc: list[dict], active_dirs: dict[tuple[int, str], bool]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for entry in pair_desc:
        for crossing in entry["crossings"]:
            if all(active_dirs.get(req, False) for req in crossing["req"]):
                cells.add(tuple(map(int, crossing["cell"])))
    return cells


def _pair_coverage_ok(pair_desc: list[dict], active_dirs: dict[tuple[int, str], bool]) -> bool:
    for entry in pair_desc:
        if not any(all(active_dirs.get(req, False) for req in crossing["req"]) for crossing in entry["crossings"]):
            return False
    return True


def _choose_active_dirs(
    rng: random.Random,
    towers: list[dict],
) -> dict[tuple[int, str], bool] | None:
    pair_desc = _pair_descriptors(towers)
    all_cells = {
        tuple(map(int, crossing["cell"]))
        for entry in pair_desc
        for crossing in entry["crossings"]
    }
    keys = [(idx, direction) for idx in range(len(towers)) for direction in DIR_ORDER]
    for _ in range(500):
        keep_prob = rng.uniform(0.45, 0.75)
        active = {key: rng.random() < keep_prob for key in keys}
        if not any(active.values()) or all(active.values()):
            continue
        if any(not any(active[(idx, direction)] for direction in DIR_ORDER) for idx in range(len(towers))):
            continue
        if not _pair_coverage_ok(pair_desc, active):
            continue
        visible = _visible_crossings(pair_desc, active)
        if visible == all_cells:
            continue
        return active
    return None


def _generate_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    tower_count = rng.choices(TOWER_COUNT_CHOICES, weights=TOWER_COUNT_WEIGHTS, k=1)[0]
    for _ in range(200):
        h = rng.randint(*GRID_SIZE_RANGE)
        w = rng.randint(*GRID_SIZE_RANGE)
        rows = _sample_centers(rng, h, tower_count)
        cols = _sample_centers(rng, w, tower_count)
        if rows is None or cols is None:
            continue

        rng.shuffle(cols)
        tower_colors = rng.sample(PALETTE, tower_count)
        remaining = [color for color in PALETTE if color not in tower_colors]
        pair_count = tower_count * (tower_count - 1) // 2
        mix_colors = rng.sample(remaining, pair_count)

        towers = []
        for idx in range(tower_count):
            center_r = int(rows[idx])
            center_c = int(cols[idx])
            towers.append(
                {
                    "color": int(tower_colors[idx]),
                    "r0": int(center_r - 1),
                    "c0": int(center_c - 1),
                    "row": int(center_r),
                    "col": int(center_c),
                }
            )

        pair_keys = [frozenset((int(towers[i]["color"]), int(towers[j]["color"]))) for i in range(tower_count) for j in range(i + 1, tower_count)]
        mix_map = {pair: int(color) for pair, color in zip(pair_keys, mix_colors)}

        active_dirs = _choose_active_dirs(rng, towers)
        if active_dirs is None:
            continue

        full_active = {(idx, direction): True for idx in range(tower_count) for direction in DIR_ORDER}
        inp = _render((h, w), towers, mix_map, active_dirs=active_dirs)
        out = _render((h, w), towers, mix_map, active_dirs=full_active)
        if np.array_equal(inp, out):
            continue

        tower_colors_set = set(tower_colors)
        input_mix = {int(v) for v in np.unique(inp) if int(v) not in tower_colors_set and int(v) != BG_COLOR}
        output_mix = {int(v) for v in np.unique(out) if int(v) not in tower_colors_set and int(v) != BG_COLOR}
        if len(input_mix) != pair_count or len(output_mix) != pair_count:
            continue
        if not any(
            int(out[r, c]) != int(inp[r, c]) and int(out[r, c]) in output_mix
            for r in range(h)
            for c in range(w)
        ):
            continue
        return inp, out
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        result = _generate_case(rng)
        if result is None:
            continue
        inp, _out = result
        return inp
    raise RuntimeError("failed to generate a valid signal-tower puzzle")
