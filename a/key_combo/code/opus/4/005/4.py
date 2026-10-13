from __future__ import annotations

import random
from collections import Counter
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np


def _load_arc_utils():
    path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    if not path.exists():
        return None
    try:
        return SourceFileLoader("_arc_utils_opus_4_005_4", str(path)).load_module()
    except Exception:
        return None


_ARC_UTILS = _load_arc_utils()


if _ARC_UTILS is not None:
    Color = _ARC_UTILS.Color
    get_bg = _ARC_UTILS.get_bg
else:
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

    def get_bg(grid: np.ndarray) -> int:
        return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


GEN_COLORS = [
    Color.BLUE,
    Color.RED,
    Color.GREEN,
    Color.YELLOW,
    Color.GREY,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
]
MARKER_COLOR = Color.PINK
MAX_ATTEMPTS = 600


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_of_color(grid: np.ndarray, color: int, *, row_limit: int | None = None) -> list[dict]:
    mask = grid == int(color)
    if row_limit is not None:
        mask[int(row_limit) :, :] = False
    seen = np.zeros_like(mask, dtype=bool)
    components: list[dict] = []
    h, w = grid.shape

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            arr = np.array(coords, dtype=int)
            components.append({"coords": arr, "bbox": _bbox_coords(arr)})

    components.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["bbox"][2], comp["bbox"][3]))
    return components


def _recover_seed_and_dominoes(grid: np.ndarray) -> tuple[int, int, np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    row_has_seed = np.any((g != bg) & (g != MARKER_COLOR), axis=1)
    assert np.any(row_has_seed), "expected a non-marker seed motif at the bottom"

    seed_top = int(g.shape[0] - 1)
    assert row_has_seed[seed_top], "expected the last row to belong to the seed motif"
    while seed_top > 0 and row_has_seed[seed_top - 1]:
        seed_top -= 1

    motif = g[seed_top:].copy()
    seed_h = int(motif.shape[0])
    assert seed_h > 0 and g.shape[0] % seed_h == 0, "grid height must be an integer multiple of the seed height"
    assert np.all(np.any((motif != bg) & (motif != MARKER_COLOR), axis=1)), "every seed row must contain visible motif color"

    dominoes: list[dict] = []
    for comp in _components_of_color(g, MARKER_COLOR, row_limit=seed_top):
        coords = comp["coords"]
        assert len(coords) == 2, "each marker must be a two-pixel domino"
        (r0, c0), (r1, c1) = sorted((tuple(map(int, pt)) for pt in coords))
        if r0 == r1 and abs(c1 - c0) == 1:
            dominoes.append(
                {
                    "orientation": "H",
                    "index": int(r0),
                    "bbox": comp["bbox"],
                    "cells": ((int(r0), int(min(c0, c1))), (int(r0), int(max(c0, c1)))),
                }
            )
        elif c0 == c1 and abs(r1 - r0) == 1:
            dominoes.append(
                {
                    "orientation": "V",
                    "index": int(c0),
                    "bbox": comp["bbox"],
                    "cells": ((int(min(r0, r1)), int(c0)), (int(max(r0, r1)), int(c0))),
                }
            )
        else:
            raise AssertionError("marker must be a horizontal or vertical domino")

    dominoes.sort(key=lambda dom: (dom["bbox"][0], dom["bbox"][1], dom["bbox"][2], dom["bbox"][3]))
    return bg, seed_top, motif, dominoes


def _tile_from_seed(motif: np.ndarray, seed_top: int, total_h: int) -> np.ndarray:
    out = np.empty((int(total_h), int(motif.shape[1])), dtype=int)
    seed_h = int(motif.shape[0])
    for r in range(int(total_h)):
        out[r] = motif[(int(r) - int(seed_top)) % seed_h]
    return out


def _apply_domino_shifts(base: np.ndarray, dominoes: list[dict]) -> np.ndarray:
    out = np.asarray(base, dtype=int).copy()
    for dom in dominoes:
        if dom["orientation"] == "H":
            out[int(dom["index"])] = np.roll(out[int(dom["index"])], 1)
        else:
            out[:, int(dom["index"])] = np.roll(out[:, int(dom["index"])], 1)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Periodically tile the bottom seed motif upward, then apply local row/column phase shifts encoded by magenta domino markers.

    Input structure:
    - Black is the dominant background in the blank upper region.
    - The bottom contiguous seed block spans the full width and contains the complete vertical motif, including its internal black cells.
    - Above the seed block, each magenta foreground component is a two-pixel domino marker placed on otherwise black rows.

    Rule:
    - Detect the bottom seed block and treat its height H as the vertical repeat period.
    - Repeat those H rows upward with the same phase until the whole grid is filled.
    - Process magenta dominoes in reading order: a horizontal domino shifts its entire row one cell right cyclically, while a vertical domino shifts its entire column one cell down cyclically.
    - Remove the markers by returning the shifted tiled pattern itself.

    Key insight:
    - The seed motif defines the global periodic field, and the domino orientation tells whether the local phase change happens along a row or along a column.
    """
    g = np.asarray(grid, dtype=int)
    bg, seed_top, motif, dominoes = _recover_seed_and_dominoes(g)
    _ = bg
    tiled = _tile_from_seed(motif, seed_top, g.shape[0])
    return _apply_domino_shifts(tiled, dominoes)


def _has_smaller_vertical_period(motif: np.ndarray) -> bool:
    seed_h = int(motif.shape[0])
    for period in range(1, seed_h):
        ok = True
        for r in range(seed_h - period):
            if not np.array_equal(motif[r], motif[r + period]):
                ok = False
                break
        if ok:
            return True
    return False


def _make_tile(rng: random.Random) -> tuple[np.ndarray, int, int]:
    for _ in range(200):
        seed_h = rng.randint(3, 5)
        period_w = rng.randint(4, 6)
        max_y = min(6, 30 // seed_h)
        max_x = min(5, 30 // period_w)
        if max_y < 4 or max_x < 3:
            continue
        reps_y = rng.randint(4, max_y)
        reps_x = rng.randint(3, max_x)

        palette = rng.sample(GEN_COLORS, rng.randint(3, 4))
        choices = palette + ([Color.BLACK] if rng.random() < 0.7 else [])
        tile = np.array([[rng.choice(choices) for _ in range(period_w)] for _ in range(seed_h)], dtype=int)

        if len(set(map(int, tile.flatten())) - {Color.BLACK}) < 2:
            continue
        density = float(np.mean(tile != Color.BLACK))
        if not (0.55 <= density <= 0.9):
            continue
        if _has_smaller_vertical_period(tile):
            continue
        if len({tuple(map(int, row)) for row in tile}) < 2:
            continue
        if len({tuple(map(int, tile[:, c])) for c in range(period_w)}) < 2:
            continue
        if any(not np.any(row != Color.BLACK) for row in tile):
            continue
        if any(np.array_equal(row, np.roll(row, 1)) for row in tile):
            continue
        varying_cols = [c for c in range(period_w) if not np.array_equal(tile[:, c], np.roll(tile[:, c], 1))]
        if len(varying_cols) < 2:
            continue
        return tile, reps_y, reps_x
    raise RuntimeError("failed to build a non-degenerate periodic seed tile")


def _sample_orientations(rng: random.Random, count: int) -> list[str]:
    if count == 1:
        return [rng.choice(["H", "V"])]
    if rng.random() < 0.75:
        oris = ["H", "V"] + [rng.choice(["H", "V"]) for _ in range(count - 2)]
    else:
        oris = [rng.choice(["H", "V"]) for _ in range(count)]
    rng.shuffle(oris)
    return oris


def _place_dominoes(rng: random.Random, base: np.ndarray, seed_h: int) -> list[dict] | None:
    total_h, width = map(int, base.shape)
    blank_h = int(total_h - seed_h)
    usable_h = int(blank_h - 1)
    if usable_h < 2 or width < 3:
        return None

    row_candidates = [r for r in range(usable_h) if not np.array_equal(base[r], np.roll(base[r], 1))]
    col_candidates = [c for c in range(width) if not np.array_equal(base[:, c], np.roll(base[:, c], 1))]
    if not row_candidates or not col_candidates:
        return None

    for _ in range(200):
        count = rng.choices([1, 2, 3], weights=[2, 4, 4])[0]
        orientations = _sample_orientations(rng, count)
        used_rows: set[int] = set()
        used_cols: set[int] = set()
        dominoes: list[dict] = []

        ok = True
        for ori in orientations:
            options: list[tuple[int, int]] = []
            if ori == "H":
                for r in row_candidates:
                    if r in used_rows:
                        continue
                    for c in range(width - 1):
                        if c in used_cols or c + 1 in used_cols:
                            continue
                        options.append((int(r), int(c)))
            else:
                for r in range(usable_h - 1):
                    if r in used_rows or r + 1 in used_rows:
                        continue
                    for c in col_candidates:
                        if c in used_cols:
                            continue
                        options.append((int(r), int(c)))

            if not options:
                ok = False
                break

            r, c = rng.choice(options)
            if ori == "H":
                dominoes.append(
                    {
                        "orientation": "H",
                        "index": int(r),
                        "bbox": (int(r), int(c), int(r), int(c + 1)),
                        "cells": ((int(r), int(c)), (int(r), int(c + 1))),
                    }
                )
                used_rows.add(int(r))
                used_cols.update((int(c), int(c + 1)))
            else:
                dominoes.append(
                    {
                        "orientation": "V",
                        "index": int(c),
                        "bbox": (int(r), int(c), int(r + 1), int(c)),
                        "cells": ((int(r), int(c)), (int(r + 1), int(c))),
                    }
                )
                used_rows.update((int(r), int(r + 1)))
                used_cols.add(int(c))

        if ok:
            dominoes.sort(key=lambda dom: (dom["bbox"][0], dom["bbox"][1], dom["bbox"][2], dom["bbox"][3]))
            return dominoes
    return None


def _render_input(motif: np.ndarray, dominoes: list[dict], reps_y: int) -> np.ndarray:
    seed_h, width = map(int, motif.shape)
    total_h = int(seed_h * reps_y)
    grid = np.zeros((total_h, width), dtype=int)
    grid[-seed_h:] = motif
    for dom in dominoes:
        for r, c in dom["cells"]:
            grid[int(r), int(c)] = MARKER_COLOR
    return grid


def _without(dominoes: list[dict], skip_idx: int) -> list[dict]:
    return [dom for i, dom in enumerate(dominoes) if i != skip_idx]


def _semantic_check(inp: np.ndarray, motif: np.ndarray, dominoes: list[dict]) -> bool:
    try:
        bg, seed_top, parsed_motif, parsed_dominoes = _recover_seed_and_dominoes(inp)
        if bg != Color.BLACK:
            return False
        if not np.array_equal(parsed_motif, motif):
            return False
        if len(parsed_dominoes) != len(dominoes):
            return False
        for a, b in zip(parsed_dominoes, dominoes):
            if a["orientation"] != b["orientation"] or a["index"] != b["index"] or a["cells"] != b["cells"]:
                return False

        base = _tile_from_seed(parsed_motif, seed_top, inp.shape[0])
        out = _apply_domino_shifts(base, parsed_dominoes)
        if not np.array_equal(main(inp), out):
            return False
        if np.array_equal(base, out):
            return False

        for idx, dom in enumerate(parsed_dominoes):
            reduced = _apply_domino_shifts(base, _without(parsed_dominoes, idx))
            if np.array_equal(reduced, out):
                return False
            if dom["orientation"] == "H":
                row = base[int(dom["index"])]
                if np.array_equal(row, np.roll(row, 1)):
                    return False
            else:
                col = base[:, int(dom["index"])]
                if np.array_equal(col, np.roll(col, 1)):
                    return False

        orientations = {dom["orientation"] for dom in parsed_dominoes}
        if len(parsed_dominoes) >= 2 and orientations == {"H", "V"}:
            reversed_out = _apply_domino_shifts(base, list(reversed(parsed_dominoes)))
            if np.array_equal(reversed_out, out):
                return False

        if _has_smaller_vertical_period(parsed_motif):
            return False

    except Exception:
        return False

    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    tile, reps_y, reps_x = _make_tile(rng)
    motif = np.tile(tile, (1, reps_x))
    base = np.tile(tile, (reps_y, reps_x))
    dominoes = _place_dominoes(rng, base, int(tile.shape[0]))
    if dominoes is None:
        return None
    inp = _render_input(motif, dominoes, reps_y)
    if not _semantic_check(inp, motif, dominoes):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic-shift ARC input")
