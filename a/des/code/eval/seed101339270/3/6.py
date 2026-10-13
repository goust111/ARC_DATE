import random
from collections import Counter

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
MAX_RETRIES = 400


def _order_bar(grid: np.ndarray) -> list[int]:
    order = [int(v) for v in np.asarray(grid[-1], dtype=int) if int(v) != BG]
    assert len(order) >= 2, "expected at least two layer markers in the bottom order bar"
    assert len(set(order)) == len(order), "order-bar marker colors must be unique"
    return order


def _body_color(scene: np.ndarray, r0: int, c0: int, tag_colors: set[int]) -> int:
    h, w = scene.shape
    local: list[int] = []
    for rr in range(r0, min(h, r0 + 3)):
        for cc in range(c0, min(w, c0 + 3)):
            if rr == r0 and cc == c0:
                continue
            value = int(scene[rr, cc])
            if value != BG and value not in tag_colors:
                local.append(value)
    if local:
        counts = Counter(local)
        top = max(counts.values())
        choices = [color for color, count in counts.items() if count == top]
        assert len(choices) == 1, "tag corner must reveal a unique body color"
        return int(choices[0])

    for cc in range(c0 + 1, w):
        value = int(scene[r0, cc])
        if value == BG:
            continue
        assert value not in tag_colors, "top edge search hit another tag before body color"
        return value
    for rr in range(r0 + 1, h):
        value = int(scene[rr, c0])
        if value == BG:
            continue
        assert value not in tag_colors, "left edge search hit another tag before body color"
        return value
    raise AssertionError("could not recover body color for layer")


def _extract_layers(grid: np.ndarray) -> tuple[list[int], dict[int, dict[str, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 3 and g.shape[1] >= 3

    order = _order_bar(g)
    tag_colors = set(order)
    scene = g[:-1]

    tag_positions: dict[int, tuple[int, int]] = {}
    for tag in order:
        coords = np.argwhere(scene == tag)
        assert len(coords) == 1, "each order-bar color must appear exactly once as a visible tag"
        tag_positions[tag] = (int(coords[0, 0]), int(coords[0, 1]))

    layers: dict[int, dict[str, int]] = {}
    used_bodies: set[int] = set()
    for tag, (r0, c0) in tag_positions.items():
        body = _body_color(scene, r0, c0, tag_colors)
        assert body not in tag_colors, "body colors must differ from tag colors"
        assert body not in used_bodies, "body colors must be unique across visible layers"
        used_bodies.add(body)

        coords = np.argwhere(scene == body)
        assert len(coords) > 0, "each layer must expose some body pixels"
        assert np.all(coords[:, 0] >= r0) and np.all(coords[:, 1] >= c0), (
            "body pixels must lie below-right of the tag corner"
        )
        r1 = int(coords[:, 0].max())
        c1 = int(coords[:, 1].max())
        assert r1 > r0 and c1 > c0, "layers must be at least 2x2"

        layers[tag] = {
            "tag": int(tag),
            "body": int(body),
            "r0": int(r0),
            "c0": int(c0),
            "r1": int(r1),
            "c1": int(c1),
        }

    return order, layers


def _render_scene(shape: tuple[int, int], layers: dict[int, dict[str, int]], order: list[int]) -> np.ndarray:
    scene = np.full(shape, BG, dtype=int)
    for tag in order:
        layer = layers[tag]
        scene[layer["r0"] : layer["r1"] + 1, layer["c0"] : layer["c1"] + 1] = layer["body"]
        scene[layer["r0"], layer["c0"]] = layer["tag"]
    return scene


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover each tagged rectangle, ignore the input's visible occlusion order,
    and repaint the full layers using the bottom bar's marker sequence from
    bottom to top. The order bar itself is erased in the output.
    """
    g = np.asarray(grid, dtype=int)
    order, layers = _extract_layers(g)

    out = np.full_like(g, BG)
    out[:-1] = _render_scene((g.shape[0] - 1, g.shape[1]), layers, order)
    return out


def _overlap(a: dict[str, int], b: dict[str, int]) -> bool:
    return not (
        a["r1"] < b["r0"]
        or b["r1"] < a["r0"]
        or a["c1"] < b["c0"]
        or b["c1"] < a["c0"]
    )


def _choose_layers(rng: random.Random) -> tuple[int, int, dict[int, dict[str, int]], list[int], list[int]] | None:
    n_layers = rng.randint(2, 4)

    step_mode = rng.choice([(2, 3), (2, 4)])
    row_offsets = [0]
    col_offsets = [0]
    for _ in range(n_layers - 1):
        row_offsets.append(row_offsets[-1] + rng.randint(*step_mode))
        col_offsets.append(col_offsets[-1] + rng.randint(*step_mode))

    min_scene_h = row_offsets[-1] + 5
    min_scene_w = col_offsets[-1] + 5
    scene_h_max = min(18, min_scene_h + 6)
    scene_w_max = min(20, min_scene_w + 7)
    if min_scene_h > scene_h_max or min_scene_w > scene_w_max:
        return None

    scene_h = rng.randint(min_scene_h, scene_h_max)
    scene_w = rng.randint(min_scene_w, scene_w_max)
    base_r = rng.randint(0, min(3, scene_h - min_scene_h))
    base_c = rng.randint(0, min(4, scene_w - min_scene_w))

    starts = [(base_r + dr, base_c + dc) for dr, dc in zip(row_offsets, col_offsets)]
    colors = rng.sample(list(COLORS), 2 * n_layers)
    rng.shuffle(colors)
    body_colors = colors[:n_layers]
    tag_colors = colors[n_layers:]

    layers: dict[int, dict[str, int]] = {}
    for idx in range(n_layers):
        r0, c0 = starts[idx]
        min_r1 = r0 + 2
        min_c1 = c0 + 2
        if idx < n_layers - 1:
            next_r0, next_c0 = starts[idx + 1]
            min_r1 = max(min_r1, next_r0 + 1)
            min_c1 = max(min_c1, next_c0 + 1)
        if min_r1 >= scene_h or min_c1 >= scene_w:
            return None

        max_r1 = scene_h - 1 - rng.randint(0, 1)
        max_c1 = scene_w - 1 - rng.randint(0, 1)
        if min_r1 > max_r1 or min_c1 > max_c1:
            return None

        layers[tag_colors[idx]] = {
            "tag": int(tag_colors[idx]),
            "body": int(body_colors[idx]),
            "r0": int(r0),
            "c0": int(c0),
            "r1": int(rng.randint(min_r1, max_r1)),
            "c1": int(rng.randint(min_c1, max_c1)),
        }

    wrong_order = list(tag_colors)
    correct_order = wrong_order[:]
    while correct_order == wrong_order:
        rng.shuffle(correct_order)

    if not any(_overlap(layers[wrong_order[i]], layers[wrong_order[j]]) for i in range(n_layers) for j in range(i + 1, n_layers)):
        return None

    return scene_h, scene_w, layers, wrong_order, correct_order


def _place_order_bar(rng: random.Random, width: int, order: list[int]) -> np.ndarray | None:
    bar = np.zeros(width, dtype=int)
    if rng.random() < 0.5:
        gaps = [0] * (len(order) - 1)
    else:
        gaps = [rng.randint(0, 1) for _ in range(len(order) - 1)]
    needed = len(order) + sum(gaps)
    if needed > width:
        return None
    start = rng.randint(0, width - needed)
    c = start
    for idx, tag in enumerate(order):
        bar[c] = int(tag)
        c += 1
        if idx < len(order) - 1:
            c += gaps[idx]
    return bar


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    packed = _choose_layers(rng)
    if packed is None:
        return None
    scene_h, scene_w, layers, wrong_order, correct_order = packed

    bar = _place_order_bar(rng, scene_w, correct_order)
    if bar is None:
        return None

    grid = np.zeros((scene_h + 1, scene_w), dtype=int)
    grid[:-1] = _render_scene((scene_h, scene_w), layers, wrong_order)
    grid[-1] = bar

    expected = np.zeros_like(grid)
    expected[:-1] = _render_scene((scene_h, scene_w), layers, correct_order)

    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid layered-rectangle puzzle")
