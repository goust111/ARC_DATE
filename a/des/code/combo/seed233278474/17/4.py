from __future__ import annotations

import random

import numpy as np


MAX_SEEDS = 4
GEN_RETRIES = 500
SEED_COLOR = 9


def _get_bg(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _rect_border_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        if r1 != r0:
            cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        if c1 != c0:
            cells.append((r, c1))
    return cells


def _frame_bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    if len(coords) < 8:
        return None
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    expected = 2 * (h + w) - 4
    if len(coords) != expected:
        return None

    coord_set = set(coords)
    for r, c in coord_set:
        if r not in (r0, r1) and c not in (c0, c1):
            return None
    for cell in _rect_border_cells(r0, c0, r1, c1):
        if cell not in coord_set:
            return None
    return r0, c0, r1, c1


def _ring_cells(bbox: tuple[int, int, int, int], layer: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return _rect_border_cells(r0 - layer, c0 - layer, r1 + layer, c1 + layer)


def _extract_frames(grid: np.ndarray) -> tuple[int, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frames: list[dict[str, object]] = []

    for color in map(int, np.unique(g)):
        if color == bg:
            continue
        for comp in _components_mask(g == color):
            bbox = _frame_bbox(comp)
            if bbox is None:
                continue
            r0, c0, r1, c1 = bbox
            inner = g[r0 + 1 : r1, c0 + 1 : c1]
            seed_pos = np.argwhere(inner != bg)
            frames.append(
                {
                    "bbox": bbox,
                    "color": color,
                    "seed_count": int(seed_pos.shape[0]),
                }
            )

    frames.sort(key=lambda frame: (frame["bbox"][0], frame["bbox"][1]))  # type: ignore[index]
    return bg, frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each rectangular frame outward by the number of seed pixels inside it.

    The input contains several non-touching rectangular borders. Each rectangle
    encloses a few seed pixels. A frame with N interior seeds grows outward by N
    concentric rectangular rings in its own border color. Lower seed counts take
    priority when growth overlaps. Seed pixels are erased to background.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    bg, frames = _extract_frames(g)
    out = g.copy()

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]  # type: ignore[misc]
        inner = out[r0 + 1 : r1, c0 + 1 : c1]
        inner[inner != bg] = bg

    grow_order = sorted(
        frames,
        key=lambda frame: (
            int(frame["seed_count"]),
            frame["bbox"][0],  # type: ignore[index]
            frame["bbox"][1],  # type: ignore[index]
        ),
    )

    for frame in grow_order:
        color = int(frame["color"])
        layers = int(frame["seed_count"])
        bbox = frame["bbox"]  # type: ignore[assignment]
        for layer in range(1, layers + 1):
            for r, c in _ring_cells(bbox, layer):
                if 0 <= r < h and 0 <= c < w and out[r, c] == bg:
                    out[r, c] = color

    return out


def _expand_box(bbox: tuple[int, int, int, int], pad: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return r0 - pad, c0 - pad, r1 + pad, c1 + pad


def _boxes_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _seed_positions(
    bbox: tuple[int, int, int, int], count: int, rng: random.Random
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    parity_choices = []
    for parity in (0, 1):
        cells = [(r, c) for r in range(inner_h) for c in range(inner_w) if (r + c) % 2 == parity]
        if len(cells) >= count:
            parity_choices.append(cells)
    cells = rng.choice(parity_choices or [[(r, c) for r in range(inner_h) for c in range(inner_w)]])
    chosen = rng.sample(cells, count)
    return [(r0 + 1 + r, c0 + 1 + c) for r, c in chosen]


def _conflict_count(frames: list[dict[str, object]]) -> int:
    occupied = set()
    for frame in frames:
        occupied.update(_ring_cells(frame["bbox"], 0))  # type: ignore[arg-type]

    conflicts = 0
    order = sorted(
        frames,
        key=lambda frame: (
            int(frame["seed_count"]),
            frame["bbox"][0],  # type: ignore[index]
            frame["bbox"][1],  # type: ignore[index]
        ),
    )
    for frame in order:
        bbox = frame["bbox"]  # type: ignore[assignment]
        for layer in range(1, int(frame["seed_count"]) + 1):
            for cell in _ring_cells(bbox, layer):
                if cell in occupied:
                    conflicts += 1
                else:
                    occupied.add(cell)
    return conflicts


def _render_grid(
    height: int,
    width: int,
    bg: int,
    frames: list[dict[str, object]],
    rng: random.Random,
) -> np.ndarray:
    grid = np.full((height, width), bg, dtype=int)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]  # type: ignore[misc]
        color = int(frame["color"])
        for r, c in _ring_cells((r0, c0, r1, c1), 0):
            grid[r, c] = color
        for r, c in _seed_positions((r0, c0, r1, c1), int(frame["seed_count"]), rng):
            grid[r, c] = SEED_COLOR
    return grid


def _try_place_free_frame(
    placed: list[dict[str, object]],
    spec: dict[str, int],
    height: int,
    width: int,
    rng: random.Random,
) -> dict[str, object] | None:
    h = int(spec["h"])
    w = int(spec["w"])
    count = int(spec["seed_count"])
    for _ in range(200):
        r0 = rng.randint(MAX_SEEDS, height - MAX_SEEDS - h)
        c0 = rng.randint(MAX_SEEDS, width - MAX_SEEDS - w)
        bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
        expanded = _expand_box(bbox, count)
        if any(_boxes_intersect(expanded, _expand_box(other["bbox"], int(other["seed_count"]))) for other in placed):
            continue
        return {"bbox": bbox, "color": int(spec["color"]), "seed_count": count}
    return None


def _place_conflict_pair(
    spec_a: dict[str, int],
    spec_b: dict[str, int],
    height: int,
    width: int,
    rng: random.Random,
) -> list[dict[str, object]] | None:
    orient = rng.choice(["h", "v"])
    gap = rng.randint(1, min(3, int(spec_a["seed_count"]) + int(spec_b["seed_count"]) - 1))
    left_first = rng.choice([True, False])

    if orient == "h":
        total_h = max(int(spec_a["h"]), int(spec_b["h"]))
        total_w = int(spec_a["w"]) + gap + int(spec_b["w"])
        if height - 2 * MAX_SEEDS < total_h or width - 2 * MAX_SEEDS < total_w:
            return None
        r0 = rng.randint(MAX_SEEDS, height - MAX_SEEDS - total_h)
        c0 = rng.randint(MAX_SEEDS, width - MAX_SEEDS - total_w)
        if left_first:
            left, right = spec_a, spec_b
        else:
            left, right = spec_b, spec_a
        left_box = (r0, c0, r0 + int(left["h"]) - 1, c0 + int(left["w"]) - 1)
        right_c0 = left_box[3] + gap + 1
        right_box = (r0, right_c0, r0 + int(right["h"]) - 1, right_c0 + int(right["w"]) - 1)
        return [
            {"bbox": left_box, "color": int(left["color"]), "seed_count": int(left["seed_count"])},
            {"bbox": right_box, "color": int(right["color"]), "seed_count": int(right["seed_count"])},
        ]

    total_h = int(spec_a["h"]) + gap + int(spec_b["h"])
    total_w = max(int(spec_a["w"]), int(spec_b["w"]))
    if height - 2 * MAX_SEEDS < total_h or width - 2 * MAX_SEEDS < total_w:
        return None
    r0 = rng.randint(MAX_SEEDS, height - MAX_SEEDS - total_h)
    c0 = rng.randint(MAX_SEEDS, width - MAX_SEEDS - total_w)
    if left_first:
        top, bottom = spec_a, spec_b
    else:
        top, bottom = spec_b, spec_a
    top_box = (r0, c0, r0 + int(top["h"]) - 1, c0 + int(top["w"]) - 1)
    bottom_r0 = top_box[2] + gap + 1
    bottom_box = (bottom_r0, c0, bottom_r0 + int(bottom["h"]) - 1, c0 + int(bottom["w"]) - 1)
    return [
        {"bbox": top_box, "color": int(top["color"]), "seed_count": int(top["seed_count"])},
        {"bbox": bottom_box, "color": int(bottom["color"]), "seed_count": int(bottom["seed_count"])},
    ]


def _frame_specs(frame_count: int, rng: random.Random) -> list[dict[str, int]]:
    max_size = 8 if frame_count <= 3 else 6
    counts = rng.sample([1, 2, 3, 4], frame_count)
    palette = [color for color in range(10) if color != SEED_COLOR]
    bg = rng.choice(palette)
    colors = rng.sample([color for color in palette if color != bg], frame_count)
    specs = []
    for color, count in zip(colors, counts):
        specs.append(
            {
                "color": int(color),
                "seed_count": int(count),
                "h": rng.randint(5, max_size),
                "w": rng.randint(5, max_size),
                "bg": int(bg),
            }
        )
    return specs


def _choose_shape(frame_count: int, rng: random.Random) -> tuple[int, int]:
    if frame_count == 2:
        return rng.randint(20, 28), rng.randint(20, 28)
    if frame_count == 3:
        return rng.randint(24, 30), rng.randint(24, 30)
    return rng.randint(28, 32), rng.randint(28, 32)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    frame_count = rng.choices([2, 3, 4], weights=[4, 4, 1])[0]
    height, width = _choose_shape(frame_count, rng)
    specs = _frame_specs(frame_count, rng)
    bg = int(specs[0]["bg"])
    conflict_mode = rng.random() < 0.55

    placed: list[dict[str, object]] = []
    remaining_specs = specs[:]

    if conflict_mode:
        remaining_specs.sort(key=lambda spec: spec["seed_count"])
        pair = _place_conflict_pair(remaining_specs[0], remaining_specs[1], height, width, rng)
        if pair is None:
            return None
        if not _boxes_intersect(
            _expand_box(pair[0]["bbox"], int(pair[0]["seed_count"])),
            _expand_box(pair[1]["bbox"], int(pair[1]["seed_count"])),
        ):
            return None
        placed.extend(pair)
        remaining_specs = remaining_specs[2:]
    else:
        rng.shuffle(remaining_specs)

    for spec in remaining_specs:
        frame = _try_place_free_frame(placed, spec, height, width, rng)
        if frame is None:
            return None
        placed.append(frame)

    conflicts = _conflict_count(placed)
    if conflict_mode and conflicts == 0:
        return None
    if not conflict_mode and conflicts != 0:
        return None

    grid = _render_grid(height, width, bg, placed, rng)

    extracted_bg, extracted_frames = _extract_frames(grid)
    if extracted_bg != bg or len(extracted_frames) != frame_count:
        return None
    extracted_counts = sorted(int(frame["seed_count"]) for frame in extracted_frames)
    target_counts = sorted(int(frame["seed_count"]) for frame in placed)
    if extracted_counts != target_counts:
        return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    if np.any(out == SEED_COLOR):
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
