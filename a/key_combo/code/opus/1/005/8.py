import itertools
import random
from typing import Iterable

import numpy as np


GRID_SIZE = 8
BG = 0
GRAY = 5
SPRITE_COLORS = (1, 2, 3)
AREA_RANGE = (3, 6)
MAX_GENERATION_ATTEMPTS = 600


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_from_coords(coords: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = list(coords)
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_height(bbox: tuple[int, int, int, int]) -> int:
    return int(bbox[2] - bbox[0] + 1)


def _bbox_width(bbox: tuple[int, int, int, int]) -> int:
    return int(bbox[3] - bbox[1] + 1)


def _boxes_have_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _is_connected(coords: Iterable[tuple[int, int]]) -> bool:
    pts = set(coords)
    if not pts:
        return False
    stack = [next(iter(pts))]
    seen = {stack[0]}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in pts and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(pts)


def _find_components(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == BG:
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            bbox = _bbox_from_coords(coords)
            components.append(
                {
                    "color": color,
                    "coords": tuple(sorted(coords)),
                    "bbox": bbox,
                    "area": len(coords),
                }
            )
    return components


def _transpose_coords(coords: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((c, r) for r, c in coords))


def _sprite_record_from_component(component: dict[str, object]) -> dict[str, object]:
    tcoords = _transpose_coords(component["coords"])
    tbbox = _bbox_from_coords(tcoords)
    height = _bbox_height(tbbox)
    width = _bbox_width(tbbox)
    move_dir = "right" if width >= height else "down"
    step = width if move_dir == "right" else height
    return {
        "color": int(component["color"]),
        "coords": tuple(component["coords"]),
        "tcoords": tcoords,
        "tbbox": tbbox,
        "move_dir": move_dir,
        "step": step,
    }


def _chain_details_from_sprites(
    sprites: list[dict[str, object]],
    static_grid: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    out = static_grid.copy()
    static_blocker = static_grid != BG
    dynamic_blocker = static_blocker.copy()
    details: list[dict[str, object]] = []

    ordered = sorted(sprites, key=lambda sprite: (sprite["tbbox"][0], sprite["tbbox"][1], sprite["color"]))
    for sprite in ordered:
        shift_r, shift_c = (0, int(sprite["step"])) if sprite["move_dir"] == "right" else (int(sprite["step"]), 0)
        copies = 0
        stop_reason = "boundary"
        chain_pixels: list[tuple[int, int]] = []
        while True:
            delta_r = shift_r * (copies + 1)
            delta_c = shift_c * (copies + 1)
            shifted = [(r + delta_r, c + delta_c) for r, c in sprite["tcoords"]]
            if any(not (0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE) for r, c in shifted):
                stop_reason = "boundary"
                break
            if any(static_blocker[r, c] for r, c in shifted):
                stop_reason = "blocker"
                break
            if any(dynamic_blocker[r, c] for r, c in shifted):
                stop_reason = "chain"
                break
            for r, c in shifted:
                out[r, c] = int(sprite["color"])
                dynamic_blocker[r, c] = True
                chain_pixels.append((r, c))
            copies += 1
        details.append(
            {
                "color": int(sprite["color"]),
                "tbbox": sprite["tbbox"],
                "move_dir": sprite["move_dir"],
                "step": int(sprite["step"]),
                "copies": copies,
                "stop_reason": stop_reason,
                "chain_pixels": tuple(chain_pixels),
            }
        )
    return out, details


def _solve_with_details(grid: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_SIZE, GRID_SIZE), "expected an 8x8 grid"
    assert g.ndim == 2

    components = _find_components(g)
    sprites = [_sprite_record_from_component(component) for component in components]

    transposed = g.T.copy()
    vanished = (g != BG) & (transposed == BG)
    static_grid = transposed.copy()
    static_grid[vanished] = GRAY

    out, chain_details = _chain_details_from_sprites(sprites, static_grid)
    info = {
        "components": components,
        "sprites": sprites,
        "transposed": transposed,
        "vanished": vanished,
        "chain_details": chain_details,
    }
    return out, info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transpose the sprites across the main diagonal, mark vanished source pixels gray, then extend each transposed sprite as a fixed-step chain.

    Input structure:
    - An 8x8 black grid contains 2-3 separated monochrome sprites in colors 1, 2, and 3.
    - Every sprite is a 4-connected non-square shape of 3-6 pixels with at least one black-cell margin around its bounding box.
    - Different sprites are isolated from each other, so each color patch is a distinct object.

    Rule:
    - Transpose the whole grid across the main diagonal.
    - Any cell that was colored in the input but is black after the transpose becomes gray.
    - For each transposed sprite, inspect its transposed bounding box: if it is wider than tall, copy the whole transposed sprite repeatedly to the right; otherwise copy it repeatedly downward, using one full transposed bbox width or height as the step size.
    - Stop a chain before the next copy would leave the grid or hit an already occupied transposed or gray cell.

    Key insight:
    - The transpose changes each sprite's aspect ratio, and that new aspect ratio determines both the chain direction and the chain step size while the disappeared pre-transpose cells stay behind as gray blockers.
    """
    out, _info = _solve_with_details(grid)
    return out


def _enumerate_shapes() -> list[dict[str, object]]:
    shapes: list[dict[str, object]] = []
    dim_pairs = [
        (1, 3),
        (3, 1),
        (1, 4),
        (4, 1),
        (1, 5),
        (5, 1),
        (2, 3),
        (3, 2),
        (2, 4),
        (4, 2),
        (3, 4),
        (4, 3),
    ]
    for h, w in dim_pairs:
        cells = [(r, c) for r in range(h) for c in range(w)]
        for area in range(AREA_RANGE[0], min(AREA_RANGE[1], h * w) + 1):
            for indices in itertools.combinations(range(len(cells)), area):
                rel = tuple(sorted(cells[i] for i in indices))
                if not _is_connected(rel):
                    continue
                bbox = _bbox_from_coords(rel)
                if bbox != (0, 0, h - 1, w - 1):
                    continue
                shapes.append(
                    {
                        "rel": rel,
                        "h": h,
                        "w": w,
                        "area": area,
                        "kind": "wide" if w > h else "tall",
                    }
                )
    return shapes


SHAPES = _enumerate_shapes()


def _build_placements() -> list[dict[str, object]]:
    placements: list[dict[str, object]] = []
    for shape in SHAPES:
        h = int(shape["h"])
        w = int(shape["w"])
        for r0 in range(1, GRID_SIZE - h):
            for c0 in range(1, GRID_SIZE - w):
                coords = tuple(sorted((r0 + rr, c0 + cc) for rr, cc in shape["rel"]))
                bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
                tcoords = _transpose_coords(coords)
                tbbox = _bbox_from_coords(tcoords)
                th = _bbox_height(tbbox)
                tw = _bbox_width(tbbox)
                move_dir = "right" if tw >= th else "down"
                step = tw if move_dir == "right" else th
                boundary_capacity = (
                    (GRID_SIZE - 1 - max(c for _, c in tcoords)) // step
                    if move_dir == "right"
                    else (GRID_SIZE - 1 - max(r for r, _ in tcoords)) // step
                )
                placements.append(
                    {
                        "rel": shape["rel"],
                        "kind": shape["kind"],
                        "coords": coords,
                        "bbox": bbox,
                        "tcoords": tcoords,
                        "tbbox": tbbox,
                        "move_dir": move_dir,
                        "step": step,
                        "boundary_capacity": int(boundary_capacity),
                        "area": int(shape["area"]),
                    }
                )
    return placements


PLACEMENTS = _build_placements()
PLACEMENTS_BY_DIR = {
    "right": [placement for placement in PLACEMENTS if placement["move_dir"] == "right" and placement["boundary_capacity"] >= 1],
    "down": [placement for placement in PLACEMENTS if placement["move_dir"] == "down" and placement["boundary_capacity"] >= 1],
}


def _render_input(scene: list[dict[str, object]]) -> np.ndarray:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    for sprite in scene:
        for r, c in sprite["coords"]:
            grid[r, c] = int(sprite["color"])
    return grid


def _render_expected_from_scene(scene: list[dict[str, object]]) -> tuple[np.ndarray, dict[str, object]]:
    inp = _render_input(scene)
    transposed = np.zeros_like(inp)
    for sprite in scene:
        for r, c in sprite["tcoords"]:
            transposed[r, c] = int(sprite["color"])
    vanished = (inp != BG) & (transposed == BG)
    static_grid = transposed.copy()
    static_grid[vanished] = GRAY
    out, chain_details = _chain_details_from_sprites(scene, static_grid)
    return out, {"input": inp, "transposed": transposed, "vanished": vanished, "chain_details": chain_details}


def _scene_is_semantically_valid(scene: list[dict[str, object]], need_blocker: bool) -> bool:
    if len(scene) not in (2, 3):
        return False

    inp = _render_input(scene)
    components = _find_components(inp)
    if len(components) != len(scene):
        return False
    if any(not (AREA_RANGE[0] <= int(component["area"]) <= AREA_RANGE[1]) for component in components):
        return False
    if any(_bbox_height(component["bbox"]) == _bbox_width(component["bbox"]) for component in components):
        return False
    if any(component["bbox"][0] == 0 or component["bbox"][1] == 0 or component["bbox"][2] == GRID_SIZE - 1 or component["bbox"][3] == GRID_SIZE - 1 for component in components):
        return False

    if any(int(inp[c, r]) not in (BG, int(inp[r, c])) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if int(inp[r, c]) != BG):
        return False

    expected, info = _render_expected_from_scene(scene)
    _ = expected
    chain_details = info["chain_details"]

    if sum(int(detail["copies"]) for detail in chain_details) < 2:
        return False
    if not any(detail["move_dir"] == "right" for detail in chain_details):
        return False
    if not any(detail["move_dir"] == "down" for detail in chain_details):
        return False
    if not np.any(info["vanished"]):
        return False
    if any(detail["stop_reason"] == "chain" for detail in chain_details):
        return False
    if need_blocker and not any(detail["stop_reason"] == "blocker" for detail in chain_details):
        return False

    transposed_components = _find_components(info["transposed"])
    if len(transposed_components) != len(scene):
        return False

    return True


def _compatible_with_scene(candidate: dict[str, object], chosen: list[dict[str, object]]) -> bool:
    for other in chosen:
        if not _boxes_have_gap(candidate["bbox"], other["bbox"], gap=1):
            return False
        if not _boxes_have_gap(candidate["tbbox"], other["tbbox"], gap=1):
            return False
    return True


def _sample_scene(need_blocker: bool) -> list[dict[str, object]] | None:
    count = 2 if random.random() < 0.72 else 3
    required_dirs = ["right", "down"]
    if count == 3:
        required_dirs.append(random.choice(["right", "down"]))
    random.shuffle(required_dirs)

    colors = random.sample(list(SPRITE_COLORS), count)
    chosen: list[dict[str, object]] = []

    for direction, color in zip(required_dirs, colors):
        pool = PLACEMENTS_BY_DIR[direction][:]
        random.shuffle(pool)
        placed = False
        for candidate in pool[:180]:
            if not _compatible_with_scene(candidate, chosen):
                continue
            chosen.append(
                {
                    "color": color,
                    "coords": candidate["coords"],
                    "bbox": candidate["bbox"],
                    "tcoords": candidate["tcoords"],
                    "tbbox": candidate["tbbox"],
                    "move_dir": candidate["move_dir"],
                    "step": candidate["step"],
                }
            )
            placed = True
            break
        if not placed:
            return None

    if not _scene_is_semantically_valid(chosen, need_blocker=need_blocker):
        return None
    return chosen


def _gen_attempt() -> np.ndarray | None:
    need_blocker = random.random() < 0.75
    for _ in range(80):
        scene = _sample_scene(need_blocker=need_blocker)
        if scene is None:
            continue
        expected, info = _render_expected_from_scene(scene)
        inp = info["input"]
        try:
            actual = main(inp)
        except Exception:
            continue
        if not np.array_equal(actual, expected):
            continue
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid transposed-chain-fill input")
