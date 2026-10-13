from __future__ import annotations

import sys
from functools import lru_cache
from itertools import product
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components, get_bg, paste


BACKGROUND = Color.BLACK
INPUT_COLOR = Color.GREY
PRIMITIVES = {
    "square": ((0, 0), (0, 1), (1, 0), (1, 1)),
    "hbar": ((0, 0), (0, 1), (0, 2)),
    "vbar": ((0, 0), (1, 0), (2, 0)),
}
COLOR_BY_COUNT = {
    1: Color.BLUE,
    2: Color.RED,
    3: Color.GREEN,
    4: Color.YELLOW,
}

HEIGHT_RANGE = (7, 10)
WIDTH_RANGE = (8, 11)
GENERATION_RETRIES = 400


def _normalize_key(cells) -> tuple[tuple[int, int], ...]:
    arr = np.asarray(list(cells), dtype=int)
    assert arr.ndim == 2 and arr.shape[1] == 2 and len(arr) > 0
    r0 = int(arr[:, 0].min())
    c0 = int(arr[:, 1].min())
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in arr))


@lru_cache(None)
def _exact_cover_solutions(
    key: tuple[tuple[int, int], ...]
) -> tuple[tuple[tuple[str, tuple[tuple[int, int], ...]], ...], ...]:
    cells = set(key)
    max_r = max(r for r, _ in cells)
    max_c = max(c for _, c in cells)

    candidates: list[tuple[str, frozenset[tuple[int, int]]]] = []
    for name, offsets in PRIMITIVES.items():
        for r0 in range(max_r + 1):
            for c0 in range(max_c + 1):
                placed = frozenset((r0 + dr, c0 + dc) for dr, dc in offsets)
                if placed <= cells:
                    candidates.append((name, placed))

    cell_to_candidates = {
        cell: [i for i, (_, placed) in enumerate(candidates) if cell in placed]
        for cell in cells
    }
    solutions: list[tuple[int, ...]] = []

    def backtrack(remaining: set[tuple[int, int]], chosen: list[int]) -> None:
        if len(solutions) >= 2:
            return
        if not remaining:
            solutions.append(tuple(chosen))
            return

        best_options = None
        for cell in remaining:
            options = [i for i in cell_to_candidates[cell] if candidates[i][1] <= remaining]
            if best_options is None or len(options) < len(best_options):
                best_options = options
        if not best_options:
            return

        for index in best_options:
            backtrack(remaining - candidates[index][1], chosen + [index])

    backtrack(set(cells), [])
    return tuple(
        tuple(
            (candidates[index][0], tuple(sorted(candidates[index][1])))
            for index in solution
        )
        for solution in solutions
    )


def _unique_decomposition(
    cells,
) -> tuple[tuple[str, tuple[tuple[int, int], ...]], ...]:
    solutions = _exact_cover_solutions(_normalize_key(cells))
    assert len(solutions) == 1, "component must have a unique primitive decomposition"
    return solutions[0]


def _count_to_color(k: int) -> int:
    return int(COLOR_BY_COUNT.get(int(k), Color.ORANGE))


def _foreground_color(grid: np.ndarray, bg: int) -> int:
    colors = sorted(set(map(int, np.unique(grid))) - {int(bg)})
    assert len(colors) == 1, "expected exactly one foreground color"
    return int(colors[0])


def _component_infos(
    grid: np.ndarray,
    fg_color: int,
    *,
    seam_row: int,
) -> list[dict]:
    infos = []
    components = find_components(
        grid,
        bg=BACKGROUND,
        connectivity=4,
        monochromatic=True,
        colors=[fg_color],
    )
    for component in components:
        solution = _unique_decomposition(map(tuple, component["coords"]))
        bbox = tuple(map(int, component["bbox"]))
        infos.append(
            {
                "coords": component["coords"],
                "k": len(solution),
                "primitive_names": tuple(name for name, _ in solution),
                "crosses_seam": bbox[0] < seam_row <= bbox[2],
            }
        )
    return infos


def _paint_infos(shape: tuple[int, int], infos: list[dict]) -> np.ndarray:
    out = np.full(shape, BACKGROUND, dtype=int)
    for info in infos:
        coords = info["coords"]
        out[coords[:, 0], coords[:, 1]] = _count_to_color(info["k"])
    return out


def _connected(cells: set[tuple[int, int]]) -> bool:
    start = next(iter(cells))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(cells)


def _key_to_patch(key: tuple[tuple[int, int], ...], color: int = INPUT_COLOR) -> np.ndarray:
    arr = np.asarray(key, dtype=int)
    h = int(arr[:, 0].max()) + 1
    w = int(arr[:, 1].max()) + 1
    patch = np.full((h, w), BACKGROUND, dtype=int)
    patch[arr[:, 0], arr[:, 1]] = int(color)
    return patch


@lru_cache(None)
def _template_catalog() -> dict[int, tuple[dict, ...]]:
    catalog: dict[int, list[dict]] = {1: [], 2: []}
    seen: set[tuple[tuple[int, int], ...]] = set()

    for primitive_count in (1, 2):
        for combo in product(PRIMITIVES.items(), repeat=primitive_count):
            for positions in product(product(range(-2, 4), range(-2, 4)), repeat=primitive_count):
                cells: set[tuple[int, int]] = set()
                valid = True
                for (_name, offsets), (r0, c0) in zip(combo, positions):
                    placed = {(r0 + dr, c0 + dc) for dr, dc in offsets}
                    if cells & placed:
                        valid = False
                        break
                    cells |= placed
                if not valid or not _connected(cells):
                    continue

                key = _normalize_key(cells)
                if key in seen:
                    continue
                solution = _exact_cover_solutions(key)
                if len(solution) != 1:
                    continue

                k = len(solution[0])
                arr = np.asarray(key, dtype=int)
                h = int(arr[:, 0].max()) + 1
                w = int(arr[:, 1].max()) + 1
                area = len(key)
                if k not in catalog or h > 4 or w > 4 or area > 7:
                    continue

                names = tuple(name for name, _ in solution[0])
                catalog[k].append(
                    {
                        "key": key,
                        "patch": _key_to_patch(key),
                        "height": h,
                        "width": w,
                        "area": area,
                        "primitive_names": names,
                        "has_square": "square" in names,
                        "has_bar": any(name in {"hbar", "vbar"} for name in names),
                    }
                )
                seen.add(key)

    assert len(catalog[1]) >= 3 and len(catalog[2]) >= 10
    return {k: tuple(v) for k, v in catalog.items()}


def _choose_template(rng: np.random.Generator, templates: tuple[dict, ...]) -> dict:
    return templates[int(rng.integers(len(templates)))]


def _can_place(
    key: tuple[tuple[int, int], ...],
    forbidden: set[tuple[int, int]],
    *,
    r0: int,
    c0: int,
    height: int,
    width: int,
) -> bool:
    for r, c in key:
        rr = int(r0 + r)
        cc = int(c0 + c)
        if not (0 <= rr < height and 0 <= cc < width):
            return False
        if (rr, cc) in forbidden:
            return False
    return True


def _mark_forbidden(
    forbidden: set[tuple[int, int]],
    key: tuple[tuple[int, int], ...],
    *,
    r0: int,
    c0: int,
) -> None:
    for r, c in key:
        rr = int(r0 + r)
        cc = int(c0 + c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                forbidden.add((rr + dr, cc + dc))


def _try_place_template(
    rng: np.random.Generator,
    grid: np.ndarray,
    forbidden: set[tuple[int, int]],
    template: dict,
    *,
    merged: bool,
) -> tuple[int, int] | None:
    h = int(template["height"])
    w = int(template["width"])
    height, width = map(int, grid.shape)

    if merged:
        row_candidates = [height - h]
    else:
        max_r0 = height - h - 2
        if max_r0 < 0:
            return None
        row_candidates = list(range(max_r0 + 1))
    col_candidates = list(range(width - w + 1))
    row_candidates = list(rng.permutation(row_candidates))
    col_candidates = list(rng.permutation(col_candidates))

    for r0 in row_candidates:
        for c0 in col_candidates:
            if not _can_place(
                template["key"],
                forbidden,
                r0=int(r0),
                c0=int(c0),
                height=height,
                width=width,
            ):
                continue
            paste(grid, template["patch"], int(r0), int(c0), transparent=BACKGROUND)
            _mark_forbidden(forbidden, template["key"], r0=int(r0), c0=int(c0))
            return int(r0), int(c0)
    return None


def _build_spec_pool() -> dict[str, tuple[dict, ...]]:
    catalog = _template_catalog()
    k2_with_square = tuple(
        template
        for template in catalog[2]
        if template["has_square"] and template["has_bar"]
    )
    k2_bar_only = tuple(
        template
        for template in catalog[2]
        if not template["has_square"] and template["has_bar"]
    )
    assert k2_with_square and k2_bar_only
    return {
        "k1": catalog[1],
        "k2_any": catalog[2],
        "k2_with_square": k2_with_square,
        "k2_bar_only": k2_bar_only,
    }


SPEC_POOL = _build_spec_pool()


def _analyze_instance(grid: np.ndarray) -> list[dict]:
    bg = int(get_bg(grid))
    fg = _foreground_color(grid, bg)
    extended = np.concatenate([grid, np.flipud(grid)], axis=0)
    return _component_infos(extended, fg, seam_row=grid.shape[0])


def _valid_instance(grid: np.ndarray, placements: list[dict]) -> bool:
    try:
        bg = int(get_bg(grid))
        if bg != BACKGROUND:
            return False
        if _foreground_color(grid, bg) != INPUT_COLOR:
            return False
        input_components = find_components(
            grid,
            bg=BACKGROUND,
            connectivity=4,
            monochromatic=True,
            colors=[INPUT_COLOR],
        )
        if len(input_components) != len(placements):
            return False

        infos = _analyze_instance(grid)
        expected_component_count = sum(1 if placement["merged"] else 2 for placement in placements)
        if len(infos) != expected_component_count:
            return False

        merged_infos = [info for info in infos if info["crosses_seam"]]
        if len(merged_infos) != sum(placement["merged"] for placement in placements):
            return False

        k_values = {info["k"] for info in infos}
        if not {1, 2, 4}.issubset(k_values):
            return False
        if not any(info["crosses_seam"] and info["k"] == 2 for info in infos):
            return False
        if not any(info["crosses_seam"] and info["k"] == 4 for info in infos):
            return False
        if not any((not info["crosses_seam"]) and info["k"] == 1 for info in infos):
            return False

        primitive_names = {name for info in infos for name in info["primitive_names"]}
        if "square" not in primitive_names:
            return False
        if not ({"hbar", "vbar"} & primitive_names):
            return False

        expected = _paint_infos((grid.shape[0] * 2, grid.shape[1]), infos)
        return np.array_equal(main(grid), expected)
    except Exception:
        return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flip the input vertically, append it below, then recolor each connected region by its primitive count.

    Input structure:
    - The input is a small black-background grid with one grey foreground color.
    - Each grey 4-connected shape is a union of one or two basic primitives:
      a 2x2 square, a 1x3 horizontal bar, or a 3x1 vertical bar.
    - Some shapes touch the bottom edge so they will join their flipped copies at the append seam,
      while other shapes stay at least two rows above the bottom and remain separate from their copies.

    Rule:
    - Vertically flip the whole input and concatenate that flipped copy underneath the original.
    - In the doubled grid, find every grey 4-connected region, including seam-merged ones.
    - Uniquely decompose each region into non-overlapping 2x2 squares and 1x3/3x1 bars, count those primitives,
      and recolor the whole region by the count: 1->blue, 2->red, 3->green, 4->yellow, 5 or more->orange.
    - Keep the background black.

    Key insight:
    - The final color of a shape cannot be read from the input alone; you must first append the vertical mirror,
      because only then do bottom-touching shapes fuse across the seam and change the primitive count that drives the color encoding.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))
    fg = _foreground_color(g, bg)

    extended = np.concatenate([g, np.flipud(g)], axis=0)
    infos = _component_infos(extended, fg, seam_row=g.shape[0])
    return _paint_infos(extended.shape, infos)


def _gen_attempt(rng: np.random.Generator) -> tuple[np.ndarray, list[dict]] | None:
    height = int(rng.integers(*HEIGHT_RANGE))
    width = int(rng.integers(*WIDTH_RANGE))
    grid = np.full((height, width), BACKGROUND, dtype=int)
    forbidden: set[tuple[int, int]] = set()
    placements: list[dict] = []

    mandatory_specs = [
        {"merged": True, "templates": SPEC_POOL["k1"]},
        {"merged": True, "templates": SPEC_POOL["k2_with_square"]},
        {"merged": False, "templates": SPEC_POOL["k1"]},
        {"merged": False, "templates": SPEC_POOL["k2_bar_only"]},
    ]

    for spec in mandatory_specs:
        for _ in range(30):
            template = _choose_template(rng, spec["templates"])
            where = _try_place_template(
                rng,
                grid,
                forbidden,
                template,
                merged=bool(spec["merged"]),
            )
            if where is None:
                continue
            placements.append(
                {
                    "merged": bool(spec["merged"]),
                    "template": template,
                    "r0": where[0],
                    "c0": where[1],
                }
            )
            break
        else:
            return None

    if rng.random() < 0.6:
        extra_templates = SPEC_POOL["k2_any"] if rng.random() < 0.55 else SPEC_POOL["k1"]
        extra_merged = bool(rng.integers(2))
        for _ in range(40):
            template = _choose_template(rng, extra_templates)
            where = _try_place_template(
                rng,
                grid,
                forbidden,
                template,
                merged=extra_merged,
            )
            if where is None:
                continue
            placements.append(
                {
                    "merged": extra_merged,
                    "template": template,
                    "r0": where[0],
                    "c0": where[1],
                }
            )
            break

    return grid, placements


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_RETRIES):
        attempt = _gen_attempt(rng)
        if attempt is None:
            continue
        grid, placements = attempt
        if _valid_instance(grid, placements):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
