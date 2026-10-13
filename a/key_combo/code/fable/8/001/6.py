from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import find_components, get_bg, paste, random_free_location


PALETTE = tuple(range(10))
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
AREA_POOL = tuple(range(2, 10))
GENERATION_ATTEMPTS = 500


def _panel_bars(panel: np.ndarray) -> dict[int, int] | None:
    if panel.ndim != 2 or min(panel.shape) == 0:
        return None

    bg = get_bg(panel)
    bars: dict[int, int] = {}
    nonempty_rows = 0
    width = panel.shape[1]
    for row in panel:
        cells = np.flatnonzero(row != bg)
        if cells.size == 0:
            continue
        nonempty_rows += 1
        color = int(row[cells[0]])
        if color == bg or color in bars:
            return None
        if not np.all(row[cells] == color):
            return None
        if int(cells[-1]) != width - 1:
            return None
        if int(cells[-1] - cells[0] + 1) != int(cells.size):
            return None
        bars[color] = int(cells.size)

    if nonempty_rows < 2:
        return None
    return bars


def _scene_components(scene: np.ndarray, bars: dict[int, int]) -> tuple[int, dict[int, list[dict]]] | None:
    if scene.ndim != 2 or min(scene.shape) == 0:
        return None

    bg = get_bg(scene)
    comps = find_components(scene, bg=bg, connectivity=4, monochromatic=True)
    if not comps:
        return None

    by_color: dict[int, list[dict]] = {}
    for comp in comps:
        color = int(comp["color"])
        by_color.setdefault(color, []).append(comp)

    if any(color not in by_color for color in bars):
        return None

    for color, parts in by_color.items():
        areas = [int(part["area"]) for part in parts]
        if len(set(areas)) != len(areas):
            return None
        if color in bars:
            rank = bars[color]
            if not (1 <= rank <= len(parts)):
                return None
            if rank in areas:
                return None
    return bg, by_color


def _split_candidates(grid: np.ndarray) -> list[tuple[np.ndarray, dict[int, int], int, dict[int, list[dict]]]]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    candidates: list[tuple[np.ndarray, dict[int, int], int, dict[int, list[dict]]]] = []

    for c in range(W):
        color = int(g[0, c])
        if not np.all(g[:, c] == color) or int(np.count_nonzero(g == color)) != H:
            continue
        for panel, scene in ((g[:, :c], g[:, c + 1 :]), (g[:, c + 1 :], g[:, :c])):
            bars = _panel_bars(panel)
            if bars is None:
                continue
            scene_info = _scene_components(scene, bars)
            if scene_info is None:
                continue
            bg, by_color = scene_info
            candidates.append((scene, bars, bg, by_color))

    for r in range(H):
        color = int(g[r, 0])
        if not np.all(g[r, :] == color) or int(np.count_nonzero(g == color)) != W:
            continue
        for panel, scene in ((g[:r, :], g[r + 1 :, :]), (g[r + 1 :, :], g[:r, :])):
            bars = _panel_bars(panel)
            if bars is None:
                continue
            scene_info = _scene_components(scene, bars)
            if scene_info is None:
                continue
            bg, by_color = scene_info
            candidates.append((scene, bars, bg, by_color))

    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only the scene objects whose area rank is named by the right-aligned color bars.

    Input structure:
    - One solid separator line splits the grid into a histogram panel and a scene panel.
    - In the histogram panel, each nonempty row contains one right-aligned solid bar of a single color.
    - In the scene panel, same-color objects are 4-connected, mutually disconnected, and have distinct areas within each color.

    Rule:
    - A bar of length k and color x selects the k-th largest scene object of color x.
    - Erase every other object of color x, and erase every object whose color does not appear in the histogram.
    - Crop away the histogram panel and the separator, returning only the filtered scene panel.

    Key insight:
    - The bars are not counts and not area values; each bar length is a rank index into the size-sorted list of same-color objects.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    candidates = _split_candidates(g)
    assert len(candidates) == 1, f"expected one valid separator/panel interpretation, got {len(candidates)}"
    scene, bars, bg, by_color = candidates[0]

    out = np.full(scene.shape, bg, dtype=int)
    for color, parts in by_color.items():
        if color not in bars:
            continue
        ranked = sorted(parts, key=lambda part: int(part["area"]), reverse=True)
        chosen = ranked[bars[color] - 1]
        coords = chosen["coords"]
        out[coords[:, 0], coords[:, 1]] = int(color)
    return out


def _normalize_coords(coords: set[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _c in coords)
    min_c = min(c for _r, c in coords)
    return sorted((r - min_r, c - min_c) for r, c in coords)


def _coords_to_patch(coords: list[tuple[int, int]], color: int, bg: int) -> np.ndarray:
    height = max(r for r, _c in coords) + 1
    width = max(c for _r, c in coords) + 1
    patch = np.full((height, width), bg, dtype=int)
    for r, c in coords:
        patch[r, c] = color
    return patch


def _sample_patch(area: int, color: int, bg: int) -> np.ndarray:
    rect_dims = [(h, w) for h in range(1, 6) for w in range(1, 6) if h * w == area]
    for _ in range(200):
        if rect_dims and random.random() < 0.35:
            h, w = random.choice(rect_dims)
            return np.full((h, w), color, dtype=int)

        coords = {(0, 0)}
        while len(coords) < area:
            frontier = set()
            for r, c in coords:
                for dr, dc in DIRS4:
                    frontier.add((r + dr, c + dc))
            frontier -= coords

            choices = list(frontier)
            random.shuffle(choices)
            picked = None
            for nr, nc in choices:
                trial = _normalize_coords(coords | {(nr, nc)})
                height = max(r for r, _c in trial) + 1
                width = max(c for _r, c in trial) + 1
                if height <= 5 and width <= 5:
                    picked = (nr, nc)
                    break
            if picked is None:
                break
            coords.add(picked)

        if len(coords) != area:
            continue
        norm = _normalize_coords(coords)
        height = max(r for r, _c in norm) + 1
        width = max(c for _r, c in norm) + 1
        if min(height, width) == 1 and area >= 6 and random.random() < 0.7:
            continue
        return _coords_to_patch(norm, color, bg)

    raise RuntimeError("failed to sample a connected patch")


def _sample_rows(height: int, count: int) -> list[int]:
    for _ in range(100):
        rows = sorted(random.sample(range(height), count))
        if all(rows[i + 1] - rows[i] >= 2 for i in range(len(rows) - 1)):
            return rows
    return sorted(random.sample(range(height), count))


def _sample_color_plan() -> tuple[int, int, list[dict], list[dict]]:
    bg = random.choice(PALETTE)
    selected_count = random.randint(2, 3)
    distractor_count = random.randint(1, 2)
    pool = [color for color in PALETTE if color != bg]
    random.shuffle(pool)
    separator = pool[0]
    selected_colors = pool[1 : 1 + selected_count]
    distractor_colors = pool[1 + selected_count : 1 + selected_count + distractor_count]

    selected_specs: list[dict] = []
    interior_idx = random.randrange(selected_count)
    for idx, color in enumerate(selected_colors):
        if idx == interior_idx:
            count = random.randint(3, 4)
            rank = random.randint(2, count - 1)
        else:
            count = random.randint(2, 4)
            rank = random.choice([1, count]) if random.random() < 0.6 else random.randint(1, count)

        areas = random.sample([area for area in AREA_POOL if area != rank], count)
        selected_specs.append(
            {
                "color": int(color),
                "count": int(count),
                "rank": int(rank),
                "areas": sorted(areas, reverse=True),
            }
        )

    distractor_specs: list[dict] = []
    for color in distractor_colors:
        count = random.randint(1, 3)
        areas = random.sample(AREA_POOL, count)
        distractor_specs.append(
            {
                "color": int(color),
                "count": int(count),
                "areas": sorted(areas, reverse=True),
            }
        )

    return bg, separator, selected_specs, distractor_specs


def _place_scene_objects(
    scene_h: int,
    scene_w: int,
    bg: int,
    selected_specs: list[dict],
    distractor_specs: list[dict],
) -> tuple[np.ndarray, np.ndarray] | None:
    scene = np.full((scene_h, scene_w), bg, dtype=int)
    placed: list[dict] = []
    all_specs = selected_specs + distractor_specs

    objects = []
    for spec in all_specs:
        for area in spec["areas"]:
            patch = _sample_patch(int(area), int(spec["color"]), bg)
            objects.append({"color": int(spec["color"]), "area": int(area), "patch": patch})
    objects.sort(key=lambda obj: (obj["patch"].shape[0] * obj["patch"].shape[1], obj["area"]), reverse=True)

    for obj in objects:
        try:
            r0, c0 = random_free_location(
                scene,
                obj["patch"],
                transparent=bg,
                border=1,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return None

        paste(scene, obj["patch"], r0, c0, transparent=bg)
        coords = np.argwhere(obj["patch"] != bg).copy()
        coords[:, 0] += r0
        coords[:, 1] += c0
        placed.append({"color": obj["color"], "area": obj["area"], "coords": coords})

    if int(np.count_nonzero(scene != bg)) >= (scene_h * scene_w) // 2:
        return None

    expected = np.full_like(scene, bg)
    for spec in selected_specs:
        same_color = [item for item in placed if item["color"] == spec["color"]]
        same_color.sort(key=lambda item: int(item["area"]), reverse=True)
        chosen = same_color[int(spec["rank"]) - 1]
        expected[chosen["coords"][:, 0], chosen["coords"][:, 1]] = int(spec["color"])

    return scene, expected


def _build_input(scene: np.ndarray, panel: np.ndarray, separator: int, side: str, bg: int) -> np.ndarray:
    scene_h, scene_w = scene.shape
    panel_h, panel_w = panel.shape
    assert scene_h == panel_h
    inp = np.full((scene_h, panel_w + 1 + scene_w), bg, dtype=int)
    if side == "left":
        inp[:, :panel_w] = panel
        inp[:, panel_w] = separator
        inp[:, panel_w + 1 :] = scene
    else:
        inp[:, :scene_w] = scene
        inp[:, scene_w] = separator
        inp[:, scene_w + 1 :] = panel
    return inp


def _semantic_ok(
    inp: np.ndarray,
    expected: np.ndarray,
    selected_specs: list[dict],
    distractor_specs: list[dict],
) -> bool:
    try:
        solved = main(inp)
    except Exception:
        return False

    if not np.array_equal(solved, expected):
        return False

    if len(distractor_specs) == 0:
        return False
    if not any(spec["rank"] > 1 for spec in selected_specs):
        return False
    if not any(1 < spec["rank"] < spec["count"] for spec in selected_specs):
        return False
    if np.array_equal(expected, np.full_like(expected, get_bg(expected))):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    bg, separator, selected_specs, distractor_specs = _sample_color_plan()
    total_objects = sum(spec["count"] for spec in selected_specs + distractor_specs)
    max_rank = max(spec["rank"] for spec in selected_specs)

    scene_h = random.randint(max(10, total_objects + 3), 24)
    panel_w = max_rank + random.randint(1, 3)
    scene_w = random.randint(10, 18)
    if panel_w + 1 + scene_w > 30:
        return None

    panel = np.full((scene_h, panel_w), bg, dtype=int)
    rows = _sample_rows(scene_h, len(selected_specs))
    order = selected_specs.copy()
    random.shuffle(order)
    for row, spec in zip(rows, order):
        panel[row, panel_w - int(spec["rank"]) :] = int(spec["color"])

    placed = _place_scene_objects(scene_h, scene_w, bg, selected_specs, distractor_specs)
    if placed is None:
        return None
    scene, expected = placed

    inp = _build_input(scene, panel, separator, random.choice(("left", "right")), bg)
    if not _semantic_ok(inp, expected, selected_specs, distractor_specs):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
