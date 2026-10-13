from __future__ import annotations

import numpy as np


BLACK = 0
GREY = 5
TOWER_COLOR_POOL = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)

MIN_TOWERS = 3
MAX_TOWERS = 5
MIN_HEIGHT = 2
MAX_HEIGHT = 8
MIN_GAP = 2
MAX_GAP = 5
MIN_SIDE_MARGIN = 1
MAX_SIDE_MARGIN = 6
MIN_TOP_MARGIN = 1
MAX_TOP_MARGIN = 4
GENERATION_RETRIES = 400

_RNG = np.random.default_rng()


def _dominant_color(values: np.ndarray) -> int:
    colors, counts = np.unique(values, return_counts=True)
    return int(colors[np.argmax(counts)])


def _extract_towers(grid: np.ndarray) -> tuple[int, int, list[dict[str, int]], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height >= 4 and width >= 3, "grid is too small for towers and ground"

    ground_row = height - 1
    ground_color = int(g[ground_row, 0])
    assert np.all(g[ground_row] == ground_color), "ground must be a full-width constant row"

    bg = _dominant_color(g[:ground_row].ravel())
    assert ground_color != bg, "ground color must differ from the background"

    towers: list[dict[str, int]] = []
    tower_mask = np.zeros_like(g, dtype=bool)

    for c in range(width):
        col = g[:ground_row, c]
        non_bg_rows = np.where(col != bg)[0]
        if non_bg_rows.size == 0:
            continue

        color = int(col[non_bg_rows[-1]])
        expected_rows = np.arange(int(non_bg_rows[0]), ground_row)
        assert np.array_equal(non_bg_rows, expected_rows), "tower must be a contiguous vertical bar"
        assert np.all(col[non_bg_rows] == color), "tower pixels must be monochromatic"
        assert color != ground_color, "tower color must differ from the ground"

        top_row = int(non_bg_rows[0])
        tower_mask[top_row:ground_row, c] = True
        towers.append(
            {
                "col": int(c),
                "color": int(color),
                "height": int(ground_row - top_row),
                "top_row": int(top_row),
            }
        )

    assert MIN_TOWERS <= len(towers) <= MAX_TOWERS, "expected 3-5 towers"
    assert len({tower["color"] for tower in towers}) == len(towers), "tower colors must be unique"
    assert len({tower["height"] for tower in towers}) == len(towers), "tower heights must be unique"
    assert all(
        towers[i + 1]["col"] - towers[i]["col"] >= MIN_GAP + 1 for i in range(len(towers) - 1)
    ), "adjacent towers must be separated by at least two empty columns"

    expected_mask = g[:ground_row] != bg
    assert np.array_equal(tower_mask[:ground_row], expected_mask), "unexpected foreground outside towers"
    return bg, ground_color, towers, tower_mask


def _draw_ray(
    out: np.ndarray,
    tower_mask: np.ndarray,
    *,
    row: int,
    start_col: int,
    direction: int,
    length: int,
    color: int,
) -> None:
    width = int(out.shape[1])
    for step in range(1, length + 1):
        c = start_col + direction * step
        if not (0 <= c < width):
            break
        if tower_mask[row, c]:
            break
        out[row, c] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """Draw borrowed-color horizontal rays from each tower top.

    Input structure
    - Black background.
    - A full-width grey ground row at the bottom.
    - Several one-cell-wide colored towers standing on the ground. Tower colors
      and heights are unique, and there are at least two empty columns between
      neighboring towers.

    Rule
    - Each tower emits a left ray colored like its nearest left neighbor and a
      right ray colored like its nearest right neighbor.
    - Rays start next to the tower top, run on the tower-top row, and have
      length equal to the emitting tower's height.
    - If a ray reaches another tower pixel, it stops before that pixel.
    - Towers and the ground remain unchanged.
    """
    g = np.asarray(grid, dtype=int)
    _bg, _ground_color, towers, tower_mask = _extract_towers(g)

    out = g.copy()
    for idx, tower in enumerate(towers):
        row = int(tower["top_row"])
        col = int(tower["col"])
        ray_len = int(tower["height"])

        if idx > 0:
            _draw_ray(
                out,
                tower_mask,
                row=row,
                start_col=col,
                direction=-1,
                length=ray_len,
                color=int(towers[idx - 1]["color"]),
            )
        if idx + 1 < len(towers):
            _draw_ray(
                out,
                tower_mask,
                row=row,
                start_col=col,
                direction=1,
                length=ray_len,
                color=int(towers[idx + 1]["color"]),
            )

    return out


def _ray_stats(towers: list[dict[str, int]], width: int) -> list[dict[str, int | bool]] | None:
    stats: list[dict[str, int | bool]] = []
    for idx, tower in enumerate(towers):
        for direction in (-1, 1):
            neighbor = idx - 1 if direction < 0 else idx + 1
            if neighbor < 0 or neighbor >= len(towers):
                continue

            blocked_by = None
            for step in range(1, int(tower["height"]) + 1):
                c = int(tower["col"]) + direction * step
                if not (0 <= c < width):
                    return None
                for other_idx, other in enumerate(towers):
                    if other_idx == idx:
                        continue
                    if int(other["col"]) == c and int(other["height"]) >= int(tower["height"]):
                        blocked_by = int(other_idx)
                        break
                if blocked_by is not None:
                    stats.append(
                        {
                            "tower": int(idx),
                            "direction": int(direction),
                            "neighbor": int(neighbor),
                            "blocked": True,
                            "blocked_by": int(blocked_by),
                            "drawn": int(step - 1),
                        }
                    )
                    break
            else:
                stats.append(
                    {
                        "tower": int(idx),
                        "direction": int(direction),
                        "neighbor": int(neighbor),
                        "blocked": False,
                        "blocked_by": -1,
                        "drawn": int(tower["height"]),
                    }
                )
    return stats


def _sample_layout() -> tuple[int, int, list[dict[str, int]]] | None:
    n_towers = int(_RNG.integers(MIN_TOWERS, MAX_TOWERS + 1))

    heights = _RNG.choice(np.arange(MIN_HEIGHT, MAX_HEIGHT + 1), size=n_towers, replace=False).astype(int)
    _RNG.shuffle(heights)
    colors = _RNG.choice(TOWER_COLOR_POOL, size=n_towers, replace=False).astype(int)

    top_margin = int(_RNG.integers(MIN_TOP_MARGIN, MAX_TOP_MARGIN + 1))
    left_margin = int(_RNG.integers(MIN_SIDE_MARGIN, MAX_SIDE_MARGIN + 1))
    right_margin = int(_RNG.integers(MIN_SIDE_MARGIN, MAX_SIDE_MARGIN + 1))
    gaps = _RNG.integers(MIN_GAP, MAX_GAP + 1, size=n_towers - 1).astype(int)

    width = int(left_margin + right_margin + n_towers + gaps.sum())
    height = int(np.max(heights) + top_margin + 1)

    towers: list[dict[str, int]] = []
    c = left_margin
    for idx in range(n_towers):
        tower_height = int(heights[idx])
        towers.append(
            {
                "col": int(c),
                "color": int(colors[idx]),
                "height": int(tower_height),
                "top_row": int(height - 1 - tower_height),
            }
        )
        c += 1
        if idx + 1 < n_towers:
            c += int(gaps[idx])

    stats = _ray_stats(towers, width)
    if stats is None:
        return None

    has_blocked = any(bool(stat["blocked"]) for stat in stats)
    has_full = any(not bool(stat["blocked"]) for stat in stats)
    if not (has_blocked and has_full):
        return None

    return height, width, towers


def _render_input(height: int, width: int, towers: list[dict[str, int]]) -> np.ndarray:
    grid = np.full((height, width), BLACK, dtype=int)
    grid[height - 1, :] = GREY
    for tower in towers:
        r0 = int(tower["top_row"])
        c = int(tower["col"])
        grid[r0 : height - 1, c] = int(tower["color"])
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        layout = _sample_layout()
        if layout is None:
            continue
        height, width, towers = layout
        inp = _render_input(height, width, towers)
        try:
            out = main(inp)
        except Exception:
            continue
        if np.array_equal(out, inp):
            continue
        return inp
    raise RuntimeError(f"failed to generate a valid tower layout after {GENERATION_RETRIES} attempts")
