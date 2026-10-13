from __future__ import annotations

from collections import Counter
import random
from typing import Iterable

import numpy as np


BACKGROUND = 0
GEN_RETRIES = 500


def neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def bbox_of(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
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
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def is_connected(mask: np.ndarray) -> bool:
    count = int(mask.sum())
    if count == 0:
        return False
    return len(connected_components(mask)) == 1


def crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def patch_cells(patch: np.ndarray, top_left: tuple[int, int]) -> list[tuple[int, int, int]]:
    r0, c0 = top_left
    cells: list[tuple[int, int, int]] = []
    for r in range(patch.shape[0]):
        for c in range(patch.shape[1]):
            color = int(patch[r, c])
            if color != BACKGROUND:
                cells.append((r0 + r, c0 + c, color))
    return cells


def coord_set(cells: Iterable[tuple[int, int, int]]) -> set[tuple[int, int]]:
    return {(r, c) for r, c, _ in cells}


def min_l1_distance(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> int:
    return min(abs(r1 - r2) + abs(c1 - c2) for r1, c1 in a for r2, c2 in b)


def parse_sprite(
    grid: np.ndarray, coords: list[tuple[int, int]], bg: int
) -> dict[str, object]:
    colors = Counter(int(grid[r, c]) for r, c in coords)
    marker_colors = [color for color, count in colors.items() if count == 1]
    body_colors = [color for color, count in colors.items() if count > 1]
    assert len(colors) == 2, f"expected two-color sprite, got {colors}"
    assert len(marker_colors) == 1, f"expected one marker color, got {colors}"
    assert len(body_colors) == 1, f"expected one body color, got {colors}"

    marker_color = int(marker_colors[0])
    body_color = int(body_colors[0])
    marker_abs = next((r, c) for r, c in coords if int(grid[r, c]) == marker_color)
    r0, c0, r1, c1 = bbox_of(coords)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()

    return {
        "coords": [(int(r), int(c)) for r, c in coords],
        "patch": patch,
        "top_left": (int(r0), int(c0)),
        "marker_abs": (int(marker_abs[0]), int(marker_abs[1])),
        "marker_local": (int(marker_abs[0] - r0), int(marker_abs[1] - c0)),
        "marker_color": marker_color,
        "body_color": body_color,
        "bg": int(bg),
    }


def solve_detailed(grid: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    comps = connected_components(g != bg)
    assert comps, "expected foreground objects"

    control_points: list[tuple[int, int]] = []
    sprites: list[dict[str, object]] = []
    for comp in comps:
        if len(comp) == 1:
            control_points.append(comp[0])
        else:
            sprites.append(parse_sprite(g, comp, bg))

    assert len(control_points) == 1, f"expected exactly one singleton control, got {len(control_points)}"
    assert sprites, "expected at least one sprite"

    marker_to_idx: dict[int, int] = {}
    for idx, sprite in enumerate(sprites):
        marker_color = int(sprite["marker_color"])
        assert marker_color not in marker_to_idx, f"duplicate marker color {marker_color}"
        marker_to_idx[marker_color] = idx

    out = g.copy()
    current_point = tuple(map(int, control_points[0]))
    current_color = int(g[current_point])
    moved_ids: set[int] = set()
    moves: list[dict[str, object]] = []

    while current_color in marker_to_idx:
        idx = marker_to_idx[current_color]
        assert idx not in moved_ids, "cyclic chain would re-summon a moved sprite"
        sprite = sprites[idx]

        old_coords = [(int(r), int(c)) for r, c in sprite["coords"]]
        old_marker = tuple(map(int, sprite["marker_abs"]))
        target = tuple(map(int, current_point))
        dr = target[0] - old_marker[0]
        dc = target[1] - old_marker[1]
        assert dr != 0 or dc != 0, "sprite move must be non-zero"

        for r, c in old_coords:
            out[r, c] = bg
        out[target] = bg
        out[old_marker] = int(sprite["body_color"])

        top_r = int(sprite["top_left"][0]) + dr
        top_c = int(sprite["top_left"][1]) + dc
        translated = patch_cells(np.asarray(sprite["patch"], dtype=int), (top_r, top_c))
        for nr, nc, _color in translated:
            assert 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1], "translated sprite must stay in bounds"
            assert out[nr, nc] == bg, "translated sprite overlaps another object or residue"
        for nr, nc, color in translated:
            out[nr, nc] = int(color)

        moves.append(
            {
                "sprite_idx": idx,
                "marker_color": int(sprite["marker_color"]),
                "body_color": int(sprite["body_color"]),
                "source_marker": old_marker,
                "target_control": target,
                "translated_top_left": (top_r, top_c),
            }
        )
        moved_ids.add(idx)
        current_point = old_marker
        current_color = int(sprite["body_color"])

    return out, {
        "bg": bg,
        "control_start": tuple(map(int, control_points[0])),
        "terminal_point": tuple(map(int, current_point)),
        "terminal_color": int(current_color),
        "moved_ids": moved_ids,
        "moves": moves,
        "sprites": sprites,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Chain-transport each summoned marked sprite until the final residue has no responder.

    Input structure:
    - Black background contains several 4-connected two-color sprites and exactly one isolated control pixel.
    - Each sprite has one repeated body color forming a small connected block plus exactly one embedded marker pixel of a different color.
    - Any color is used by at most one sprite body and at most one sprite marker.

    Rule:
    - Find the sprite whose marker color matches the current control color, then translate the whole sprite so its marker lands exactly on that control pixel.
    - Erase the sprite from its old position, consume the covered control pixel, and leave one body-colored residue pixel at the sprite's old marker location.
    - Treat that residue as the next control point and continue until no sprite marker color matches the current control color; sprites never summoned stay in place.

    Key insight:
    - Only the first control pixel is visible in the input; every later destination is created by the previous move, so the correct output requires following the causal chain in order.
    """
    out, _details = solve_detailed(grid)
    return out


def random_connected_mask() -> np.ndarray:
    for _ in range(200):
        h = random.randint(3, 5)
        w = random.randint(3, 5)
        area = random.randint(5, min(8, h * w - 1))

        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for nr, nc in neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))

        if len(cells) != area:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = crop_mask(mask)
        if min(mask.shape) < 3:
            continue
        if not is_connected(mask):
            continue
        return mask
    raise RuntimeError("failed to sample connected mask")


def make_sprite(body_color: int, marker_color: int) -> tuple[np.ndarray, tuple[int, int]]:
    for _ in range(300):
        mask = random_connected_mask()
        h, w = mask.shape
        candidates: list[tuple[int, int]] = []
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                if not mask[r, c]:
                    continue
                degree = sum(
                    1
                    for nr, nc in neighbors4(r, c)
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]
                )
                if degree < 2:
                    continue
                body_mask = mask.copy()
                body_mask[r, c] = False
                if is_connected(body_mask):
                    candidates.append((r, c))
        if not candidates:
            continue

        marker = random.choice(candidates)
        patch = np.zeros(mask.shape, dtype=int)
        patch[mask] = int(body_color)
        patch[marker] = int(marker_color)
        return patch, marker
    raise RuntimeError("failed to build sprite with embedded marker")


def sample_anchor_points(
    count: int, shape: tuple[int, int], *, margin: int = 3, min_gap: int = 6
) -> list[tuple[int, int]] | None:
    h, w = shape
    anchors: list[tuple[int, int]] = []
    for _ in range(1000):
        if len(anchors) == count:
            return anchors
        point = (
            random.randrange(margin, h - margin),
            random.randrange(margin, w - margin),
        )
        if all(abs(point[0] - r) + abs(point[1] - c) >= min_gap for r, c in anchors):
            anchors.append(point)
    return None


def in_bounds_for_patch(
    patch: np.ndarray, top_left: tuple[int, int], shape: tuple[int, int]
) -> bool:
    r0, c0 = top_left
    h, w = patch.shape
    return 0 <= r0 and 0 <= c0 and r0 + h <= shape[0] and c0 + w <= shape[1]


def state_valid(
    objects: list[list[tuple[int, int, int]]],
    singletons: list[tuple[int, int]],
    shape: tuple[int, int],
) -> bool:
    h, w = shape
    sets = [coord_set(obj) for obj in objects]

    for pts in sets:
        for r, c in pts:
            if not (0 <= r < h and 0 <= c < w):
                return False

    all_pts: set[tuple[int, int]] = set()
    for pts in sets:
        if all_pts & pts:
            return False
        all_pts |= pts

    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            if min_l1_distance(sets[i], sets[j]) < 2:
                return False

    for idx, (r, c) in enumerate(singletons):
        if not (0 <= r < h and 0 <= c < w):
            return False
        if (r, c) in all_pts:
            return False
        for pts in sets:
            if min(abs(r - rr) + abs(c - cc) for rr, cc in pts) < 2:
                return False
        for j in range(idx + 1, len(singletons)):
            rr, cc = singletons[j]
            if abs(r - rr) + abs(c - cc) < 2:
                return False

    return True


def build_grid(
    shape: tuple[int, int],
    objects: list[list[tuple[int, int, int]]],
    singletons: list[tuple[int, int, int]],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for obj in objects:
        for r, c, color in obj:
            assert grid[r, c] == BACKGROUND
            grid[r, c] = int(color)
    for r, c, color in singletons:
        assert grid[r, c] == BACKGROUND
        grid[r, c] = int(color)
    return grid


def choose_counts() -> tuple[int, int]:
    chain_len = random.choices([1, 2, 3, 4], weights=[1, 4, 4, 2])[0]
    max_unmoved = min(2, (8 - chain_len) // 2)
    options = list(range(max_unmoved + 1))
    if max_unmoved == 0:
        return chain_len, 0
    weights = [2] * len(options)
    if 0 in options:
        weights[0] = 1
    if max_unmoved >= 1:
        weights[1] = 3
    unmoved = random.choices(options, weights=weights)[0]
    return chain_len, unmoved


def verify_generated_case(
    inp: np.ndarray,
    expected: np.ndarray,
    chain_len: int,
    start_control: tuple[int, int],
    anchors: list[tuple[int, int]],
    chain_markers: list[int],
    chain_bodies: list[int],
) -> bool:
    if not np.array_equal(main(inp), expected):
        return False

    out, details = solve_detailed(inp)
    if not np.array_equal(out, expected):
        return False
    if tuple(details["control_start"]) != tuple(start_control):
        return False
    if tuple(details["terminal_point"]) != tuple(anchors[-1]):
        return False
    if int(details["terminal_color"]) != int(chain_bodies[-1]):
        return False

    moves = list(details["moves"])
    if len(moves) != chain_len:
        return False

    for i, move in enumerate(moves):
        if tuple(move["target_control"]) != tuple(anchors[i]):
            return False
        if tuple(move["source_marker"]) != tuple(anchors[i + 1]):
            return False
        if int(move["marker_color"]) != int(chain_markers[i]):
            return False
        if int(move["body_color"]) != int(chain_bodies[i]):
            return False

    comps = connected_components(inp != BACKGROUND)
    singleton_count = sum(1 for comp in comps if len(comp) == 1)
    if singleton_count != 1:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        h = random.randint(20, 26)
        w = random.randint(20, 26)
        shape = (h, w)

        chain_len, unmoved_count = choose_counts()
        total_unique_colors = chain_len + 2 * unmoved_count + 1
        palette = random.sample(list(range(1, 10)), total_unique_colors)

        start_color = palette[0]
        chain_bodies = [int(x) for x in palette[1 : 1 + chain_len]]
        unmoved_bodies = [int(x) for x in palette[1 + chain_len : 1 + chain_len + unmoved_count]]
        unmoved_markers = [int(x) for x in palette[1 + chain_len + unmoved_count :]]
        chain_markers = [start_color] + chain_bodies[:-1]

        moved_specs = []
        try:
            for body_color, marker_color in zip(chain_bodies, chain_markers):
                patch, marker_local = make_sprite(body_color, marker_color)
                moved_specs.append(
                    {
                        "patch": patch,
                        "marker_local": tuple(map(int, marker_local)),
                    }
                )
            unmoved_specs = []
            for body_color, marker_color in zip(unmoved_bodies, unmoved_markers):
                patch, marker_local = make_sprite(body_color, marker_color)
                unmoved_specs.append(
                    {
                        "patch": patch,
                        "marker_local": tuple(map(int, marker_local)),
                    }
                )
        except RuntimeError:
            continue

        anchors = sample_anchor_points(chain_len + 1, shape)
        if anchors is None:
            continue

        moved_input: list[list[tuple[int, int, int]]] = []
        moved_output: list[list[tuple[int, int, int]]] = []
        valid = True
        for i, spec in enumerate(moved_specs):
            patch = np.asarray(spec["patch"], dtype=int)
            mr, mc = map(int, spec["marker_local"])
            in_top = (anchors[i + 1][0] - mr, anchors[i + 1][1] - mc)
            out_top = (anchors[i][0] - mr, anchors[i][1] - mc)
            if not in_bounds_for_patch(patch, in_top, shape) or not in_bounds_for_patch(patch, out_top, shape):
                valid = False
                break
            moved_input.append(patch_cells(patch, in_top))
            moved_output.append(patch_cells(patch, out_top))
        if not valid:
            continue

        input_objects = list(moved_input)
        output_objects = list(moved_output)
        residue = anchors[-1]
        control = anchors[0]

        unmoved_objects: list[list[tuple[int, int, int]]] = []
        for spec in unmoved_specs:
            patch = np.asarray(spec["patch"], dtype=int)
            placed = False
            for _ in range(500):
                top_left = (
                    random.randint(0, h - patch.shape[0]),
                    random.randint(0, w - patch.shape[1]),
                )
                cells = patch_cells(patch, top_left)
                if not state_valid(input_objects + unmoved_objects + [cells], [control], shape):
                    continue
                if not state_valid(output_objects + unmoved_objects + [cells], [residue], shape):
                    continue
                unmoved_objects.append(cells)
                placed = True
                break
            if not placed:
                valid = False
                break
        if not valid:
            continue

        if not state_valid(input_objects + unmoved_objects, [control], shape):
            continue
        if not state_valid(output_objects + unmoved_objects, [residue], shape):
            continue

        inp = build_grid(shape, input_objects + unmoved_objects, [(control[0], control[1], start_color)])
        expected = build_grid(
            shape,
            output_objects + unmoved_objects,
            [(residue[0], residue[1], chain_bodies[-1])],
        )

        if np.array_equal(inp, expected):
            continue
        if not verify_generated_case(
            inp,
            expected,
            chain_len,
            control,
            anchors,
            chain_markers,
            chain_bodies,
        ):
            continue
        return inp

    raise RuntimeError("failed to generate valid residual-chain transport input")
