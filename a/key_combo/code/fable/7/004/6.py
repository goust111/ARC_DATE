from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, get_bg


_SIDES = ("top", "bottom", "left", "right")
_SPRITE_COLORS = [Color.BLUE, Color.RED, Color.GREEN, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]


def _depth_and_width(side: str, shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    return (h, w) if side in {"top", "bottom"} else (w, h)


def _coord(side: str, depth_idx: int, line_idx: int, shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    if side == "top":
        return depth_idx, line_idx
    if side == "bottom":
        return h - 1 - depth_idx, line_idx
    if side == "left":
        return line_idx, depth_idx
    if side == "right":
        return line_idx, w - 1 - depth_idx
    raise ValueError(f"unknown side: {side}")


def _find_lamp_edge(grid: np.ndarray, bg: int) -> tuple[str, int]:
    candidates: list[tuple[str, int]] = []
    if np.all(grid[0, :] == grid[0, 0]) and int(grid[0, 0]) != bg:
        candidates.append(("top", int(grid[0, 0])))
    if np.all(grid[-1, :] == grid[-1, 0]) and int(grid[-1, 0]) != bg:
        candidates.append(("bottom", int(grid[-1, 0])))
    if np.all(grid[:, 0] == grid[0, 0]) and int(grid[0, 0]) != bg:
        candidates.append(("left", int(grid[0, 0])))
    if np.all(grid[:, -1] == grid[0, -1]) and int(grid[0, -1]) != bg:
        candidates.append(("right", int(grid[0, -1])))
    if len(candidates) != 1:
        raise ValueError("expected exactly one monochrome lamp edge")
    return candidates[0]


def _screen_color(grid: np.ndarray, bg: int, lamp_color: int) -> int:
    colors = [int(c) for c in np.unique(grid) if int(c) not in {bg, lamp_color}]
    if not colors:
        raise ValueError("missing guide screen color")
    return max(colors, key=lambda color: int(np.count_nonzero(grid == color)))


def _project(grid: np.ndarray, side: str, lamp_color: int, screen_color: int, bg: int) -> np.ndarray:
    out = np.where((grid == lamp_color) | (grid == screen_color), grid, bg).astype(int)
    depth, width = _depth_and_width(side, grid.shape)

    for line_idx in range(width):
        blocker_color: int | None = None
        for depth_idx in range(1, depth):
            r, c = _coord(side, depth_idx, line_idx, grid.shape)
            value = int(grid[r, c])
            if value == bg:
                continue
            if value == screen_color:
                if blocker_color is not None and depth_idx + 1 < depth:
                    rr, cc = _coord(side, depth_idx + 1, line_idx, grid.shape)
                    out[rr, cc] = blocker_color
                break
            if value != lamp_color and blocker_color is None:
                blocker_color = value
    return out


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Project each sightline's first blocking sprite onto the first later intact screen, then erase all sprites.

    Input structure:
    - One border row or column is a solid lamp-marker strip showing the direction of parallel light.
    - Two or more one-cell-thick grey guide screens run parallel to that lamp edge, each with several single-cell gaps.
    - Other colored singleton pixels are scattered between the lamp, between screens, or beyond the last screen.

    Rule:
    - For every sightline perpendicular to the screens, scan from the lamp side inward.
    - If the first non-background cell is a sprite, keep its color and continue until the first later grey screen cell; paint that color one cell behind that screen on the shadow side.
    - If a grey screen cell is encountered before any sprite, that screen blocks the line and nothing farther on that line matters.
    - Gaps are transparent: they do not block the scan and simply hand the sightline to farther screens.
    - The output keeps only the lamp strip, the grey screens, and any projected shadow pixels; all original sprites are removed.

    Key insight:
    - A gap is not an empty event: it transfers the whole sightline to the next screen, so a sprite can project only onto the first later intact screen on that same line.
    """
    grid = np.asarray(input_grid, dtype=int)
    bg = get_bg(grid)
    side, lamp_color = _find_lamp_edge(grid, bg)
    screen_color = _screen_color(grid, bg, lamp_color)
    return _project(grid, side, lamp_color, screen_color, bg)


def _screen_depths(depth: int, n_screens: int) -> list[int]:
    for _ in range(200):
        positions = sorted(random.sample(range(2, depth - 1), n_screens))
        if positions[-1] > depth - 2:
            continue
        if all(b - a >= 3 for a, b in zip(positions, positions[1:])):
            return positions
    raise RuntimeError("failed to place screens")


def _line_shadow_depth(
    screen_depths: list[int],
    gaps: tuple[bool, ...],
    sprite_pos: int | None,
    depth: int,
) -> int | None:
    blocker = False
    for depth_idx in range(1, depth):
        if sprite_pos == depth_idx:
            blocker = True
        if depth_idx in screen_depths:
            screen_idx = screen_depths.index(depth_idx)
            if gaps[screen_idx]:
                continue
            if blocker and depth_idx + 1 < depth:
                return depth_idx + 1
            return None
    return None


def _line_spec(
    case: str,
    target_screen: int | None,
    screen_depths: list[int],
    depth: int,
    color: int | None,
) -> dict[str, object]:
    n_screens = len(screen_depths)
    if case == "pass":
        sprite_min = screen_depths[-1] + 1
        return {
            "gaps": tuple(True for _ in range(n_screens)),
            "sprite_pos": random.randint(sprite_min, depth - 1),
            "sprite_color": color,
            "case": case,
        }

    assert target_screen is not None
    gaps = [True] * target_screen + [False]
    gaps.extend(random.choice([True, False]) for _ in range(n_screens - len(gaps)))

    if case == "blocked":
        sprite_pos = None
        sprite_color = None
    elif case == "hidden":
        start = screen_depths[target_screen] + 1
        end = screen_depths[target_screen + 1] - 1 if target_screen + 1 < n_screens else depth - 1
        sprite_pos = random.randint(start, end)
        sprite_color = color
    elif case == "cast":
        start = screen_depths[target_screen - 1] + 1 if target_screen > 0 else 1
        end = screen_depths[target_screen] - 1
        sprite_pos = random.randint(start, end)
        sprite_color = color
    else:
        raise ValueError(f"unknown case: {case}")

    return {
        "gaps": tuple(gaps),
        "sprite_pos": sprite_pos,
        "sprite_color": sprite_color,
        "case": case,
    }


def _build_specs(screen_depths: list[int], depth: int, width: int) -> list[dict[str, object]]:
    n_screens = len(screen_depths)
    active_colors = random.sample(_SPRITE_COLORS, random.randint(3, min(5, len(_SPRITE_COLORS))))

    specs: list[dict[str, object]] = []
    for screen_idx in range(n_screens):
        specs.append(_line_spec("cast", screen_idx, screen_depths, depth, active_colors[screen_idx % len(active_colors)]))
    specs.append(_line_spec("hidden", random.randrange(n_screens), screen_depths, depth, random.choice(active_colors)))
    specs.append(_line_spec("blocked", random.randrange(n_screens), screen_depths, depth, None))
    specs.append(_line_spec("pass", None, screen_depths, depth, random.choice(active_colors)))

    while len(specs) < width:
        case = random.choices(
            ["cast", "hidden", "blocked", "pass"],
            weights=[5, 3, 2, 2],
            k=1,
        )[0]
        target = None if case == "pass" else random.randrange(n_screens)
        color = None if case == "blocked" else random.choice(active_colors)
        specs.append(_line_spec(case, target, screen_depths, depth, color))

    random.shuffle(specs)
    return specs[:width]


def _render_scene(
    side: str,
    shape: tuple[int, int],
    lamp_color: int,
    screen_color: int,
    bg: int,
    screen_depths: list[int],
    specs: list[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.full(shape, bg, dtype=int)
    depth, width = _depth_and_width(side, shape)

    for line_idx in range(width):
        r, c = _coord(side, 0, line_idx, shape)
        grid[r, c] = lamp_color

    for screen_idx, screen_depth in enumerate(screen_depths):
        for line_idx, spec in enumerate(specs):
            if not bool(spec["gaps"][screen_idx]):
                r, c = _coord(side, screen_depth, line_idx, shape)
                grid[r, c] = screen_color

    for line_idx, spec in enumerate(specs):
        sprite_pos = spec["sprite_pos"]
        if sprite_pos is None:
            continue
        r, c = _coord(side, int(sprite_pos), line_idx, shape)
        grid[r, c] = int(spec["sprite_color"])

    out = np.where((grid == lamp_color) | (grid == screen_color), grid, bg).astype(int)
    for line_idx, spec in enumerate(specs):
        shadow_depth = _line_shadow_depth(screen_depths, tuple(spec["gaps"]), spec["sprite_pos"], depth)
        if shadow_depth is None:
            continue
        r, c = _coord(side, shadow_depth, line_idx, shape)
        out[r, c] = int(spec["sprite_color"])

    return grid, out


def _valid_specs(specs: list[dict[str, object]], screen_depths: list[int], depth: int) -> bool:
    cast_targets = []
    has_hidden = False
    has_pass = False
    shadow_count = 0
    shadow_colors: set[int] = set()

    for spec in specs:
        case = str(spec["case"])
        if case == "cast":
            gaps = tuple(spec["gaps"])
            target = next(idx for idx, is_gap in enumerate(gaps) if not is_gap)
            cast_targets.append(target)
        if case == "hidden":
            has_hidden = True
        if case == "pass":
            has_pass = True

        shadow_depth = _line_shadow_depth(screen_depths, tuple(spec["gaps"]), spec["sprite_pos"], depth)
        if shadow_depth is not None:
            shadow_count += 1
            shadow_colors.add(int(spec["sprite_color"]))

    return (
        set(cast_targets) == set(range(len(screen_depths)))
        and has_hidden
        and has_pass
        and shadow_count >= 2
        and len(shadow_colors) >= 2
    )


def _attempt_generate() -> np.ndarray | None:
    side = random.choice(_SIDES)
    n_screens = random.randint(2, 4)
    min_width = n_screens + 3
    width = random.randint(max(8, min_width), 14)
    depth = random.randint(max(14, 3 * n_screens + 6), 26)
    screen_depths = _screen_depths(depth, n_screens)
    specs = _build_specs(screen_depths, depth, width)

    if not _valid_specs(specs, screen_depths, depth):
        return None

    shape = (depth, width) if side in {"top", "bottom"} else (width, depth)
    grid, expected = _render_scene(
        side=side,
        shape=shape,
        lamp_color=Color.YELLOW,
        screen_color=Color.GREY,
        bg=Color.BLACK,
        screen_depths=screen_depths,
        specs=specs,
    )

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    if np.array_equal(grid, expected):
        return None

    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid multi-screen projection input")
