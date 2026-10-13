from __future__ import annotations

import math
import random
from collections import Counter, defaultdict

import numpy as np


BACKGROUND = 0
SOURCE_COLORS = (2, 3, 4)
WALL_COLOR = 5
GEN_MAX_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


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
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _detect_source_columns(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    sources: list[dict] = []
    for c in range(w):
        rows = [r for r in range(h) if int(grid[r, c]) not in (int(bg), WALL_COLOR)]
        if not rows:
            continue
        colors = {int(grid[r, c]) for r in rows}
        assert len(colors) == 1, "every source column must be monochrome"
        count = len(rows)
        height = math.isqrt(count)
        assert height * height == count, "source column pixel count must be a square"
        sources.append(
            {
                "col": int(c),
                "rows": rows,
                "color": int(next(iter(colors))),
                "count": int(count),
                "height": int(height),
                "base_width": int(2 * height - 1),
            }
        )
    assert sources, "expected at least one source column"
    return sorted(sources, key=lambda item: item["col"])


def _detect_u_containers(grid: np.ndarray, bg: int) -> list[dict]:
    containers: list[dict] = []
    for comp in _components(grid == WALL_COLOR):
        rows = [r for r, _c in comp]
        cols = [c for _r, c in comp]
        top, bottom = min(rows), max(rows)
        left, right = min(cols), max(cols)
        if right - left < 2 or bottom - top < 1:
            continue

        expected = {
            *((r, left) for r in range(top, bottom + 1)),
            *((r, right) for r in range(top, bottom + 1)),
            *((bottom, c) for c in range(left, right + 1)),
        }
        coords = set(comp)
        if coords != expected:
            continue
        if not np.all(grid[top:bottom, left + 1 : right] == bg):
            continue

        containers.append(
            {
                "top": int(top),
                "bottom": int(bottom),
                "left": int(left),
                "right": int(right),
                "inner_width": int(right - left - 1),
                "inner_height": int(bottom - top),
                "center": int((left + right) // 2),
            }
        )

    assert containers, "expected at least one U-shaped container"
    return sorted(containers, key=lambda item: (item["left"], item["top"]))


def _match_sources_to_containers(sources: list[dict], containers: list[dict]) -> list[tuple[dict, dict]]:
    by_width_sources: dict[int, list[dict]] = defaultdict(list)
    by_width_containers: dict[int, list[dict]] = defaultdict(list)
    for source in sources:
        by_width_sources[int(source["base_width"])].append(source)
    for container in containers:
        by_width_containers[int(container["inner_width"])].append(container)

    matches: list[tuple[dict, dict]] = []
    for width, width_sources in by_width_sources.items():
        width_containers = sorted(by_width_containers.get(int(width), []), key=lambda item: item["left"])
        width_sources = sorted(width_sources, key=lambda item: item["col"])
        assert len(width_sources) == len(width_containers), "every pyramid width must have matching containers"
        for source, container in zip(width_sources, width_containers):
            assert int(source["height"]) <= int(container["inner_height"]), "container too short for its pyramid"
            matches.append((source, container))

    assert len(matches) == len(sources) == len(containers)
    return matches


def _paint_pyramid(out: np.ndarray, color: int, center: int, base_row: int, height: int) -> None:
    for layer in range(int(height)):
        row = int(base_row - height + 1 + layer)
        half = int(layer)
        out[row, center - half : center + half + 1] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rearrange each sparse vertical source column into a centered pyramid and drop it into the size-matching U-shaped container.

    Input structure:
    - A mostly black grid contains 2 or more sparse vertical source columns in the upper area and grey U-shaped containers in the lower area.
    - Each source column is monochrome, uses a non-grey foreground color, and contains n^2 pixels for some pyramid height n.
    - Each grey component is an upward-opening U with a black interior; a container is identified by its interior width.

    Rule:
    - Read every non-grey source column, count its pixels, and convert those n^2 pixels into a centered pyramid with row widths 1, 3, 5, ... , 2n-1.
    - Find every grey U-shaped container and measure its interior width.
    - Match each pyramid to a container whose interior width equals the pyramid base width, erase the original source pixels, and place the pyramid inside the matched container with its base resting on the container floor.

    Key insight:
    - The decisive cue is geometric compatibility: the sparse column determines a pyramid size, and that size chooses which container can receive it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    sources = _detect_source_columns(g, bg)
    containers = _detect_u_containers(g, bg)
    matches = _match_sources_to_containers(sources, containers)

    out = g.copy()
    out[(out != int(bg)) & (out != WALL_COLOR)] = int(bg)
    for source, container in matches:
        _paint_pyramid(
            out,
            color=int(source["color"]),
            center=int(container["center"]),
            base_row=int(container["bottom"] - 1),
            height=int(source["height"]),
        )
    return out


def _draw_container(grid: np.ndarray, top: int, left: int, inner_width: int, inner_height: int) -> dict:
    right = int(left + inner_width + 1)
    bottom = int(top + inner_height)
    grid[top : bottom + 1, left] = WALL_COLOR
    grid[top : bottom + 1, right] = WALL_COLOR
    grid[bottom, left : right + 1] = WALL_COLOR
    return {
        "top": int(top),
        "bottom": int(bottom),
        "left": int(left),
        "right": int(right),
        "inner_width": int(inner_width),
        "inner_height": int(inner_height),
        "center": int((left + right) // 2),
    }


def _scatter_source_column(grid: np.ndarray, col: int, color: int, height: int, row_limit: int) -> dict | None:
    count = int(height * height)
    if row_limit < count:
        return None
    rows = sorted(random.sample(range(row_limit), count))
    for row in rows:
        grid[row, col] = int(color)
    return {
        "col": int(col),
        "rows": rows,
        "color": int(color),
        "count": int(count),
        "height": int(height),
        "base_width": int(2 * height - 1),
    }


def _container_positions(width: int, outer_widths: list[int], min_gap: int = 2) -> list[int] | None:
    total = sum(outer_widths) + min_gap * (len(outer_widths) - 1)
    if total > width:
        return None
    slack = width - total
    extras = np.random.multinomial(slack, np.ones(len(outer_widths) + 1) / (len(outer_widths) + 1))
    positions: list[int] = []
    cur = int(extras[0])
    for i, outer_width in enumerate(outer_widths):
        positions.append(int(cur))
        cur += int(outer_width)
        if i < len(outer_widths) - 1:
            cur += int(min_gap + extras[i + 1])
    return positions


def _source_columns(width: int, heights_left_to_right: list[int], forbidden_order: list[int]) -> list[int] | None:
    cols = list(range(1, width - 1))
    random.shuffle(cols)
    chosen: list[int] = []
    for col in cols:
        if any(abs(col - prev) < 3 for prev in chosen):
            continue
        chosen.append(int(col))
        if len(chosen) == len(heights_left_to_right):
            break
    if len(chosen) != len(heights_left_to_right):
        return None
    chosen.sort()

    widths_left_to_right = [2 * h - 1 for h in heights_left_to_right]
    if widths_left_to_right == forbidden_order:
        return None
    return chosen


def _build_expected(shape: tuple[int, int], containers: list[dict], sources: list[dict]) -> np.ndarray:
    out = np.full(shape, BACKGROUND, dtype=int)
    for container in containers:
        out[container["top"] : container["bottom"] + 1, container["left"]] = WALL_COLOR
        out[container["top"] : container["bottom"] + 1, container["right"]] = WALL_COLOR
        out[container["bottom"], container["left"] : container["right"] + 1] = WALL_COLOR
    for source, container in _match_sources_to_containers(sources, containers):
        _paint_pyramid(out, source["color"], container["center"], container["bottom"] - 1, source["height"])
    return out


def _semantic_checks(inp: np.ndarray, expected: np.ndarray, cross_match: bool) -> bool:
    if inp.ndim != 2 or expected.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if _get_bg(inp) != BACKGROUND:
        return False

    try:
        parsed_sources = _detect_source_columns(inp, BACKGROUND)
        parsed_containers = _detect_u_containers(inp, BACKGROUND)
        pred = main(inp)
    except Exception:
        return False

    if len(parsed_sources) != 2 or len(parsed_containers) != 2:
        return False
    if sorted(source["base_width"] for source in parsed_sources) != [3, 5]:
        return False
    if sorted(container["inner_width"] for container in parsed_containers) != [3, 5]:
        return False
    if np.any(inp[(inp != BACKGROUND) & (inp != WALL_COLOR)] == WALL_COLOR):
        return False

    source_order = [source["base_width"] for source in sorted(parsed_sources, key=lambda item: item["col"])]
    container_order = [container["inner_width"] for container in sorted(parsed_containers, key=lambda item: item["left"])]
    if cross_match != (source_order != container_order):
        return False
    return np.array_equal(pred, expected)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(18, 24)
    width = random.randint(17, 24)
    grid = np.full((height, width), BACKGROUND, dtype=int)

    cross_match = bool(random.getrandbits(1))
    colors = random.sample(list(SOURCE_COLORS), 2)

    container_widths = [3, 5]
    random.shuffle(container_widths)
    outer_widths = [inner_width + 2 for inner_width in container_widths]
    left_positions = _container_positions(width, outer_widths, min_gap=2)
    if left_positions is None:
        return None

    containers: list[dict] = []
    for inner_width, left in zip(container_widths, left_positions):
        pyramid_height = (inner_width + 1) // 2
        inner_height = random.randint(pyramid_height, pyramid_height + 2)
        top_min = max(9, inner_height + 1)
        top_max = height - inner_height - 1
        if top_min > top_max:
            return None
        top = random.randint(top_min, top_max)
        containers.append(_draw_container(grid, top, left, inner_width, inner_height))
    containers = sorted(containers, key=lambda item: item["left"])

    container_order = [container["inner_width"] for container in containers]
    source_heights_left_to_right = [2 if width_ == 3 else 3 for width_ in container_order]
    if cross_match:
        source_heights_left_to_right = list(reversed(source_heights_left_to_right))

    source_cols = _source_columns(width, source_heights_left_to_right, container_order if cross_match else list(reversed(container_order)))
    if source_cols is None:
        return None

    row_limit = min(container["top"] for container in containers) - 2
    if row_limit < 9:
        return None

    sources: list[dict] = []
    for col, color, height_ in zip(source_cols, colors, source_heights_left_to_right):
        source = _scatter_source_column(grid, col, color, height_, row_limit)
        if source is None:
            return None
        sources.append(source)
    sources = sorted(sources, key=lambda item: item["col"])

    expected = _build_expected(grid.shape, containers, sources)
    if not _semantic_checks(grid, expected, cross_match):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gravity-pyramid container sample")
