from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components, get_bg, paste, random_free_location


BG = Color.BLACK
BODY_COLORS = (Color.BLUE, Color.RED, Color.GREEN)
MARKER_COLORS = (Color.ORANGE, Color.TEAL, Color.MAROON)
TARGET_COLORS = (Color.YELLOW, Color.PINK, Color.GREY)
PALETTE_MAPPING = dict(zip(BODY_COLORS, TARGET_COLORS, strict=True))
ORIENTATIONS = ("dr", "dl", "ur", "ul")
LEG_CHOICES = ((2, 2), (2, 3), (2, 4), (3, 2), (3, 3), (4, 2))
GRID_ROW_RANGE = (16, 25)
GRID_COL_RANGE = (18, 29)
MAX_GENERATION_ATTEMPTS = 400
MIN_MOVE_MANHATTAN = 4


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _make_l_patch(body_color: int, marker_color: int, v_len: int, h_len: int, orient: str) -> tuple[np.ndarray, tuple[int, int]]:
    patch = np.zeros((v_len + 1, h_len + 1), dtype=int)
    if orient == "dr":
        anchor = (0, 0)
        patch[1:, 0] = body_color
        patch[0, 1:] = body_color
    elif orient == "dl":
        anchor = (0, h_len)
        patch[1:, h_len] = body_color
        patch[0, :h_len] = body_color
    elif orient == "ur":
        anchor = (v_len, 0)
        patch[:v_len, 0] = body_color
        patch[v_len, 1:] = body_color
    elif orient == "ul":
        anchor = (v_len, h_len)
        patch[:v_len, h_len] = body_color
        patch[v_len, :h_len] = body_color
    else:
        raise ValueError(f"unknown orientation: {orient}")

    patch[anchor] = marker_color
    return patch, anchor


def _find_palette(grid: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray]:
    h, w = grid.shape
    if h < 2 or w < 3:
        raise ValueError("grid too small for palette")

    candidates: list[tuple[int, dict[int, int], np.ndarray]] = []
    for r0 in range(h - 1):
        mask = np.zeros_like(grid, dtype=bool)
        mask[r0 : r0 + 2, w - 3 : w] = True
        window = grid[r0 : r0 + 2, w - 3 : w]
        top = [int(x) for x in window[0].tolist()]
        bottom = [int(x) for x in window[1].tolist()]
        if any(v == bg for v in top + bottom):
            continue
        if len(set(top)) != 3 or len(set(bottom)) != 3:
            continue

        isolated = True
        for rr, cc in zip(*np.where(mask), strict=False):
            color = int(grid[rr, cc])
            for nr, nc in _neighbors4(int(rr), int(cc)):
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and int(grid[nr, nc]) == color:
                    isolated = False
                    break
            if not isolated:
                break
        if not isolated:
            continue

        mapping = {int(src): int(dst) for src, dst in zip(top, bottom, strict=True)}
        candidates.append((len(set(top + bottom)), mapping, mask))

    if len(candidates) != 1:
        raise ValueError(f"expected exactly one right-edge palette, got {len(candidates)}")
    _score, mapping, mask = candidates[0]
    return mapping, mask


def _parse_l_sprite(grid: np.ndarray, coords: np.ndarray) -> dict:
    color_counts = Counter(int(grid[r, c]) for r, c in coords)
    if len(color_counts) != 2:
        raise ValueError("sprite must have exactly two colors")

    marker_colors = [color for color, count in color_counts.items() if count == 1]
    body_colors = [color for color, count in color_counts.items() if count >= 4]
    if len(marker_colors) != 1 or len(body_colors) != 1:
        raise ValueError("sprite must have one singleton marker color and one repeated body color")

    marker_color = int(marker_colors[0])
    body_color = int(body_colors[0])
    marker = next((int(r), int(c)) for r, c in coords if int(grid[r, c]) == marker_color)
    body_coords = [(int(r), int(c)) for r, c in coords if int(grid[r, c]) == body_color]

    mr, mc = marker
    if not all(r == mr or c == mc for r, c in body_coords):
        raise ValueError("body pixels must lie on the marker row or marker column")

    row_cols = sorted(c for r, c in body_coords if r == mr)
    col_rows = sorted(r for r, c in body_coords if c == mc)
    if not row_cols or not col_rows:
        raise ValueError("sprite must have both horizontal and vertical body legs")
    if max(row_cols) < mc:
        horiz = "left"
    elif min(row_cols) > mc:
        horiz = "right"
    else:
        raise ValueError("horizontal leg must extend on exactly one side of the marker")
    if max(col_rows) < mr:
        vert = "up"
    elif min(col_rows) > mr:
        vert = "down"
    else:
        raise ValueError("vertical leg must extend on exactly one side of the marker")

    if horiz == "left" and row_cols != list(range(min(row_cols), mc)):
        raise ValueError("horizontal leg must be contiguous")
    if horiz == "right" and row_cols != list(range(mc + 1, max(row_cols) + 1)):
        raise ValueError("horizontal leg must be contiguous")
    if vert == "up" and col_rows != list(range(min(col_rows), mr)):
        raise ValueError("vertical leg must be contiguous")
    if vert == "down" and col_rows != list(range(mr + 1, max(col_rows) + 1)):
        raise ValueError("vertical leg must be contiguous")

    return {
        "coords": [(int(r), int(c)) for r, c in coords],
        "body_color": body_color,
        "marker_color": marker_color,
        "anchor": marker,
    }


def _analyze_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    mapping, palette_mask = _find_palette(g, bg)
    scene = g.copy()
    scene[palette_mask] = bg

    components = find_components(scene, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    controls_by_color: dict[int, list[tuple[int, int]]] = defaultdict(list)
    sprites = []
    for comp in components:
        coords = comp["coords"]
        if int(comp["area"]) == 1:
            r, c = map(int, coords[0])
            controls_by_color[int(scene[r, c])].append((r, c))
        else:
            sprites.append(_parse_l_sprite(scene, coords))

    return {
        "bg": bg,
        "grid": g,
        "mapping": mapping,
        "palette_mask": palette_mask,
        "sprites": sprites,
        "controls_by_color": controls_by_color,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Translate each marked L sprite onto its matching control and recolor it via the edge palette.

    Input structure:
    - Black is the background, and a 2x3 palette legend sits on the right edge.
    - Three 4-connected multicolor L sprites each contain one repeated body color and one embedded marker pixel at the L bend.
    - Separate isolated control pixels use the same colors as the embedded markers.

    Rule:
    - Read the palette as body-color -> target-color by matching the top row to the bottom row.
    - For each sprite, match its singleton marker color to the isolated control pixel of the same color.
    - Translate the whole sprite so the marker-at-bend lands on that control pixel, erase the original sprite and the consumed control, then recolor every translated sprite pixel to the palette target color of that sprite's body color.

    Key insight:
    - Position comes from marker/control color pairing, while the final color comes from an independent palette lookup keyed by the sprite body color.
    """
    info = _analyze_scene(grid)
    g = info["grid"]
    bg = int(info["bg"])
    mapping = info["mapping"]
    controls_by_color = info["controls_by_color"]

    out = g.copy()
    moves = []
    for sprite in info["sprites"]:
        marker_color = int(sprite["marker_color"])
        controls = controls_by_color.get(marker_color, [])
        if len(controls) != 1:
            continue
        body_color = int(sprite["body_color"])
        if body_color not in mapping:
            continue

        anchor = sprite["anchor"]
        target = controls[0]
        dr = int(target[0] - anchor[0])
        dc = int(target[1] - anchor[1])
        translated = []
        for r, c in sprite["coords"]:
            nr = int(r + dr)
            nc = int(c + dc)
            if not (0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]):
                translated = []
                break
            translated.append((nr, nc))
        if not translated:
            continue
        moves.append(
            {
                "orig_coords": sprite["coords"],
                "translated": translated,
                "target": target,
                "color": int(mapping[body_color]),
            }
        )

    for move in moves:
        for r, c in move["orig_coords"]:
            out[r, c] = bg
        tr, tc = move["target"]
        out[tr, tc] = bg

    occupied: set[tuple[int, int]] = set()
    for move in moves:
        for cell in move["translated"]:
            if cell in occupied:
                raise ValueError("translated sprites overlap")
            occupied.add(cell)
        for r, c in move["translated"]:
            out[r, c] = int(move["color"])

    return out


def _signature_pool() -> list[tuple[int, int, str]]:
    return [(v_len, h_len, orient) for v_len, h_len in LEG_CHOICES for orient in ORIENTATIONS]


def _reservation_patch(patch: np.ndarray) -> np.ndarray:
    return (patch != BG).astype(int)


def _translated_coords(top_left: tuple[int, int], patch: np.ndarray) -> set[tuple[int, int]]:
    r0, c0 = top_left
    coords = set()
    for r, c in np.argwhere(patch != BG):
        coords.add((int(r0 + r), int(c0 + c)))
    return coords


def _place_palette(grid: np.ndarray, row0: int) -> np.ndarray:
    palette = np.array([BODY_COLORS, TARGET_COLORS], dtype=int)
    paste(grid, palette, row0, grid.shape[1] - 3)
    return palette


def _semantic_checks(grid: np.ndarray) -> None:
    info = _analyze_scene(grid)
    assert info["mapping"] == PALETTE_MAPPING
    assert len(info["sprites"]) == 3
    assert {int(s["body_color"]) for s in info["sprites"]} == set(BODY_COLORS)
    assert {int(s["marker_color"]) for s in info["sprites"]} == set(MARKER_COLORS)
    assert set(info["controls_by_color"]) == set(MARKER_COLORS)
    assert all(len(v) == 1 for v in info["controls_by_color"].values())

    translated_sets: list[set[tuple[int, int]]] = []
    for sprite in info["sprites"]:
        target = info["controls_by_color"][int(sprite["marker_color"])][0]
        assert target != sprite["anchor"]
        dr = int(target[0] - sprite["anchor"][0])
        dc = int(target[1] - sprite["anchor"][1])
        assert abs(dr) + abs(dc) >= MIN_MOVE_MANHATTAN
        translated = {(int(r + dr), int(c + dc)) for r, c in sprite["coords"]}
        assert all(0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] for r, c in translated)
        translated_sets.append(translated)

    for i, first in enumerate(translated_sets):
        for second in translated_sets[i + 1 :]:
            assert first.isdisjoint(second)

    out = main(grid)
    assert out.shape == grid.shape
    assert np.array_equal(out[info["palette_mask"]], grid[info["palette_mask"]])

    remaining = out.copy()
    remaining[info["palette_mask"]] = BG
    assert not np.isin(remaining, BODY_COLORS + MARKER_COLORS).any()

    for sprite in info["sprites"]:
        body_color = int(sprite["body_color"])
        target = info["controls_by_color"][int(sprite["marker_color"])][0]
        dr = int(target[0] - sprite["anchor"][0])
        dc = int(target[1] - sprite["anchor"][1])
        translated = {(int(r + dr), int(c + dc)) for r, c in sprite["coords"]}
        for r, c in translated:
            assert int(out[r, c]) == int(PALETTE_MAPPING[body_color])


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        height = random.randint(*GRID_ROW_RANGE)
        width = random.randint(*GRID_COL_RANGE)
        grid = np.full((height, width), BG, dtype=int)

        palette_row = random.randint(1, height - 3)
        _place_palette(grid, palette_row)

        placement_canvas = np.zeros_like(grid)
        placement_canvas[palette_row : palette_row + 2, width - 3 : width] = 1

        signatures = random.sample(_signature_pool(), 3)
        specs = list(zip(BODY_COLORS, MARKER_COLORS, signatures, strict=True))
        random.shuffle(specs)

        sprite_specs = []
        failed = False
        for body_color, marker_color, (v_len, h_len, orient) in sorted(
            specs,
            key=lambda item: (item[2][0] + 1) * (item[2][1] + 1),
            reverse=True,
        ):
            patch, anchor = _make_l_patch(body_color, marker_color, v_len, h_len, orient)
            reserve = _reservation_patch(patch)
            try:
                dest_r0, dest_c0 = random_free_location(
                    placement_canvas,
                    reserve,
                    transparent=BG,
                    border=1,
                    padding=1,
                    padding_connectivity=8,
                )
            except ValueError:
                failed = True
                break

            paste(placement_canvas, reserve, dest_r0, dest_c0, transparent=BG)
            control = (int(dest_r0 + anchor[0]), int(dest_c0 + anchor[1]))
            sprite_specs.append(
                {
                    "body_color": body_color,
                    "marker_color": marker_color,
                    "patch": patch,
                    "anchor": anchor,
                    "dest_top_left": (int(dest_r0), int(dest_c0)),
                    "control": control,
                }
            )
        if failed:
            continue

        for spec in sprite_specs:
            r, c = spec["control"]
            grid[r, c] = int(spec["marker_color"])

        input_canvas = placement_canvas.copy()
        for spec in sprite_specs:
            reserve = _reservation_patch(spec["patch"])
            placed = False
            for _try in range(80):
                try:
                    src_r0, src_c0 = random_free_location(
                        input_canvas,
                        reserve,
                        transparent=BG,
                        border=1,
                        padding=1,
                        padding_connectivity=8,
                    )
                except ValueError:
                    break
                src_anchor = (int(src_r0 + spec["anchor"][0]), int(src_c0 + spec["anchor"][1]))
                move_distance = abs(src_anchor[0] - spec["control"][0]) + abs(src_anchor[1] - spec["control"][1])
                if move_distance < MIN_MOVE_MANHATTAN:
                    continue
                paste(input_canvas, reserve, src_r0, src_c0, transparent=BG)
                paste(grid, spec["patch"], src_r0, src_c0, transparent=BG)
                spec["src_top_left"] = (int(src_r0), int(src_c0))
                placed = True
                break
            if not placed:
                failed = True
                break
        if failed:
            continue

        try:
            _semantic_checks(grid)
        except Exception:
            continue
        return grid

    raise RuntimeError("failed to generate a valid ARC instance")
