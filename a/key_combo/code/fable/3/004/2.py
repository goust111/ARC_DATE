from __future__ import annotations

from collections import Counter
from pathlib import Path
import importlib.util
import random

import numpy as np


def _load_arc_utils():
    """Use the shared ARC helpers when this file is loaded directly."""
    path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("_arc_utils", path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        return None
    return mod


_ARC_UTILS = _load_arc_utils()


def _get_bg(grid: np.ndarray) -> int:
    if _ARC_UTILS is not None:
        return int(_ARC_UTILS.get_bg(grid))
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    if _ARC_UTILS is not None:
        _ARC_UTILS.paste(dst, patch, int(r0), int(c0))
    else:
        h, w = patch.shape
        dst[int(r0) : int(r0) + h, int(c0) : int(c0) + w] = patch


def _d4_tiles(tile: np.ndarray) -> list[np.ndarray]:
    """Return the eight square-grid rotations/reflections of a tile."""
    t = np.asarray(tile, dtype=int)
    flipped = np.fliplr(t)
    return [np.rot90(t, k).copy() for k in range(4)] + [
        np.rot90(flipped, k).copy() for k in range(4)
    ]


def _axis_candidates(length: int, k: int, used: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    candidates: list[tuple[int, int, np.ndarray]] = []
    max_gap = (int(length) - k * k) // (k - 1)
    for gap in range(1, max_gap + 1):
        span = k * k + (k - 1) * gap
        for origin in range(0, int(length) - span + 1):
            ids = np.full(int(length), -1, dtype=int)
            step = k + gap
            for i in range(k):
                start = origin + i * step
                ids[start : start + k] = i
            if used.size and np.any(ids[used] < 0):
                continue
            candidates.append((origin, gap, ids))
    return candidates


def _classify_patch(patch: np.ndarray, bg: int) -> tuple[str, int | None]:
    vals = np.unique(patch)
    if len(vals) == 1:
        color = int(vals[0])
        if color == int(bg):
            return "empty", None
        return "solid", color
    return "blueprint", None


def _find_geometries(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    fg = np.argwhere(g != int(bg))
    if fg.size == 0:
        return []

    out: list[dict[str, object]] = []
    max_k = min(5, h, w)
    for k in range(2, max_k + 1):
        if k * k + (k - 1) > h or k * k + (k - 1) > w:
            continue

        row_choices = _axis_candidates(h, k, fg[:, 0])
        col_choices = _axis_candidates(w, k, fg[:, 1])
        for top, gap_r, _row_ids in row_choices:
            for left, gap_c, _col_ids in col_choices:
                solids: dict[tuple[int, int], int] = {}
                empties = 0
                blueprints: list[tuple[tuple[int, int], np.ndarray]] = []

                for i in range(k):
                    r0 = top + i * (k + gap_r)
                    for j in range(k):
                        c0 = left + j * (k + gap_c)
                        patch = g[r0 : r0 + k, c0 : c0 + k]
                        kind, color = _classify_patch(patch, bg)
                        if kind == "empty":
                            empties += 1
                        elif kind == "solid":
                            assert color is not None
                            solids[(i, j)] = int(color)
                        else:
                            blueprints.append(((i, j), patch.copy()))

                if len(blueprints) != 1:
                    continue
                if len(solids) <= empties:
                    continue
                if len(solids) < 3:
                    continue

                bp_pos, bp_tile = blueprints[0]
                out.append(
                    {
                        "k": k,
                        "top": top,
                        "left": left,
                        "gap_r": gap_r,
                        "gap_c": gap_c,
                        "solids": solids,
                        "blueprint_pos": bp_pos,
                        "blueprint": bp_tile,
                    }
                )
    return out


def _paint_decoded_layout(grid: np.ndarray, geom: dict[str, object], decoded: np.ndarray, bg: int) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    k = int(geom["k"])
    top = int(geom["top"])
    left = int(geom["left"])
    gap_r = int(geom["gap_r"])
    gap_c = int(geom["gap_c"])

    for i in range(k):
        r0 = top + i * (k + gap_r)
        for j in range(k):
            c0 = left + j * (k + gap_c)
            color = int(decoded[i, j])
            out[r0 : r0 + k, c0 : c0 + k] = color if color != int(bg) else int(bg)
    return out


def _solve_outputs(grid: np.ndarray, bg: int) -> list[np.ndarray]:
    solutions: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}
    for geom in _find_geometries(grid, bg):
        solids = geom["solids"]
        assert isinstance(solids, dict)
        bp_i, bp_j = geom["blueprint_pos"]  # type: ignore[misc]
        seen_decoded: set[bytes] = set()

        for decoded in _d4_tiles(geom["blueprint"]):  # type: ignore[arg-type]
            key = decoded.tobytes()
            if key in seen_decoded:
                continue
            seen_decoded.add(key)

            if int(decoded[int(bp_i), int(bp_j)]) == int(bg):
                continue
            ok = True
            for (i, j), color in solids.items():
                if int(decoded[int(i), int(j)]) != int(color):
                    ok = False
                    break
            if not ok:
                continue

            rendered = _paint_decoded_layout(grid, geom, decoded, bg)
            solutions[(rendered.shape, rendered.tobytes())] = rendered
    return list(solutions.values())


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the block lattice by decoding the rotated or mirrored self-referential blueprint block.

    Input structure:
    - The dominant background surrounds a K by K lattice of block slots.
    - Each slot is a K by K square, and adjacent slots are separated by background seams.
    - Most visible slots are solid one-color blocks, some slots are empty background, and exactly one slot is a non-solid K by K blueprint image.
    - The blueprint image is a rotated or reflected minimap of the whole K by K slot layout and may contain background pixels for slots that stay empty.

    Rule:
    - Enumerate the eight dihedral orientations of the blueprint.
    - Keep the unique orientation whose pixels match every already visible solid block at the corresponding lattice coordinate.
    - Interpret that oriented blueprint as the desired slot-color map.
    - Repaint every colored blueprint coordinate as a solid K by K block, and repaint every background coordinate as an empty slot.
    - This includes replacing the blueprint slot itself with the solid block color shown at its own coordinate in the decoded blueprint.

    Key insight:
    - The odd multicolor block is not decoration: it is the same lattice drawn at block scale, and the existing solid blocks are the key that reveals its orientation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    solutions = _solve_outputs(g, bg)
    assert len(solutions) == 1, "expected a unique blueprint orientation and lattice geometry"
    return solutions[0]


# ------------------------- generator -------------------------

_COLORS = list(range(10))
_MAX_ATTEMPTS = 500


def _all_d4_distinct(tile: np.ndarray) -> bool:
    keys = {x.tobytes() for x in _d4_tiles(tile)}
    return len(keys) == 8


def _make_blueprint(k: int, bg: int, palette: list[int]) -> np.ndarray | None:
    cells = [(r, c) for r in range(k) for c in range(k)]
    total = k * k
    max_bg = max(1, total // 4)

    for _ in range(300):
        bg_count = random.randint(1, max_bg)
        bg_cells = set(random.sample(cells, bg_count))
        if any(all((r, c) in bg_cells for c in range(k)) for r in range(k)):
            continue
        if any(all((r, c) in bg_cells for r in range(k)) for c in range(k)):
            continue

        arr = np.full((k, k), int(bg), dtype=int)
        colored = [cell for cell in cells if cell not in bg_cells]
        for r, c in colored:
            arr[r, c] = random.choice(palette)

        forced_colors = random.sample(palette, min(3, len(palette), len(colored)))
        forced_cells = random.sample(colored, len(forced_colors))
        for (r, c), color in zip(forced_cells, forced_colors):
            arr[r, c] = int(color)

        if len(set(map(int, arr[arr != int(bg)]))) < 2:
            continue
        if not _all_d4_distinct(arr):
            continue
        return arr
    return None


def _choose_visible_sets(
    blueprint: np.ndarray, bg: int
) -> tuple[tuple[int, int], set[tuple[int, int]], set[tuple[int, int]]] | None:
    k = int(blueprint.shape[0])
    cells = [(r, c) for r in range(k) for c in range(k)]
    colored = [cell for cell in cells if int(blueprint[cell]) != int(bg)]
    bg_cells = {cell for cell in cells if int(blueprint[cell]) == int(bg)}

    for _ in range(300):
        bp_pos = random.choice(colored)
        candidates = [cell for cell in colored if cell != bp_pos]
        if not candidates:
            continue
        max_missing = min(k, len(candidates))
        missing_count = random.randint(1, max_missing)
        missing = set(random.sample(candidates, missing_count))
        solids = set(colored) - missing - {bp_pos}
        empty_count = len(bg_cells) + len(missing)

        if len(solids) <= empty_count:
            continue

        visible = set(solids)
        visible.add(bp_pos)
        if {r for r, _ in visible} != set(range(k)):
            continue
        if {c for _, c in visible} != set(range(k)):
            continue

        return bp_pos, missing, solids
    return None


def _matching_decodes(
    visible_blueprint: np.ndarray,
    solids: set[tuple[int, int]],
    blueprint: np.ndarray,
    bp_pos: tuple[int, int],
    bg: int,
) -> list[np.ndarray]:
    matches = []
    for decoded in _d4_tiles(visible_blueprint):
        if int(decoded[bp_pos]) == int(bg):
            continue
        if all(int(decoded[pos]) == int(blueprint[pos]) for pos in solids):
            matches.append(decoded)
    return matches


def _layout_origins(k: int, gap: int) -> tuple[int, int, int, int, int, int]:
    span = k * k + (k - 1) * gap
    spare_h = 30 - span
    spare_w = 30 - span
    top = random.randint(0, min(2, spare_h))
    bottom = random.randint(0, min(2, spare_h - top))
    left = random.randint(0, min(2, spare_w))
    right = random.randint(0, min(2, spare_w - left))
    return top, bottom, left, right, span + top + bottom, span + left + right


def _render_case(
    k: int,
    gap: int,
    bg: int,
    blueprint: np.ndarray,
    visible_blueprint: np.ndarray,
    bp_pos: tuple[int, int],
    missing: set[tuple[int, int]],
    solids: set[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray] | None:
    top, _bottom, left, _right, h, w = _layout_origins(k, gap)
    inp = np.full((h, w), int(bg), dtype=int)
    out = np.full((h, w), int(bg), dtype=int)

    for i in range(k):
        r0 = top + i * (k + gap)
        for j in range(k):
            c0 = left + j * (k + gap)
            desired = int(blueprint[i, j])
            if desired != int(bg):
                block = np.full((k, k), desired, dtype=int)
                _paste(out, block, r0, c0)

            pos = (i, j)
            if pos == bp_pos:
                _paste(inp, visible_blueprint, r0, c0)
            elif pos in solids:
                block = np.full((k, k), desired, dtype=int)
                _paste(inp, block, r0, c0)
            elif pos in missing:
                pass
            else:
                pass

    if _get_bg(inp) != int(bg):
        return None
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    k = random.choice([3, 4, 5])
    if k == 5:
        gap = 1
    elif k == 4:
        gap = random.choice([1, 2])
    else:
        gap = random.choice([1, 2, 3])

    bg = random.choice(_COLORS)
    palette_size = random.randint(3, 5)
    palette = random.sample([c for c in _COLORS if c != bg], palette_size)
    blueprint = _make_blueprint(k, bg, palette)
    if blueprint is None:
        return None

    visible_sets = _choose_visible_sets(blueprint, bg)
    if visible_sets is None:
        return None
    bp_pos, missing, solids = visible_sets

    orientations = _d4_tiles(blueprint)
    visible_blueprint = random.choice(orientations)
    matches = _matching_decodes(visible_blueprint, solids, blueprint, bp_pos, bg)
    if len(matches) != 1 or not np.array_equal(matches[0], blueprint):
        return None

    rendered = _render_case(k, gap, bg, blueprint, visible_blueprint, bp_pos, missing, solids)
    if rendered is None:
        return None
    inp, expected = rendered

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None

    top_lefts = []
    for i in range(k):
        for j in range(k):
            if int(blueprint[i, j]) == int(bg):
                top_lefts.append((i, j))
    if not top_lefts:
        return None
    if not missing:
        return None

    bp_r = None
    bp_c = None
    for geom in _find_geometries(inp, bg):
        if int(geom["k"]) == k:
            pos = geom["blueprint_pos"]
            if pos == bp_pos:
                bp_r = int(geom["top"]) + bp_pos[0] * (k + int(geom["gap_r"]))
                bp_c = int(geom["left"]) + bp_pos[1] * (k + int(geom["gap_c"]))
                break
    if bp_r is None or bp_c is None:
        return None
    if np.array_equal(inp[bp_r : bp_r + k, bp_c : bp_c + k], expected[bp_r : bp_r + k, bp_c : bp_c + k]):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid self-referential blueprint input")
