import numpy as np
from collections import deque


NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
BG = 0


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            queue = deque([(r, c)])
            visited[r, c] = True
            coords = []

            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        queue.append((nr, nc))

            coord_arr = np.array(coords, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": coord_arr,
                    "bbox": _bbox_from_coords(coord_arr),
                    "area": int(coord_arr.shape[0]),
                }
            )

    return components


def _count_holes(mask: np.ndarray) -> int:
    h, w = mask.shape
    exterior = np.zeros((h, w), dtype=bool)
    queue = deque()

    for r in range(h):
        for c in (0, w - 1):
            if not mask[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                queue.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not mask[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                queue.append((nr, nc))

    holes = 0
    seen = exterior.copy()
    for r in range(h):
        for c in range(w):
            if mask[r, c] or seen[r, c]:
                continue
            holes += 1
            queue = deque([(r, c)])
            seen[r, c] = True
            while queue:
                cr, cc = queue.popleft()
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
    return holes


def _scale_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(mask.astype(np.uint8), np.ones((factor, factor), dtype=np.uint8)).astype(bool)


def _rects_with_gap_disjoint(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _paint_binary_sprite(grid: np.ndarray, sprite: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = sprite.shape
    region = grid[top : top + h, left : left + w]
    region[sprite.astype(bool)] = int(color)


def _plain_shared_sprite(hole_count: int, tall: bool, rng: np.random.Generator) -> np.ndarray:
    if tall:
        sprite = np.ones((5, 7), dtype=np.uint8)
        choices = {
            1: [
                [(2, 2, 2, 3)],
                [(1, 4, 3, 4)],
            ],
            2: [
                [(1, 1, 1, 1), (3, 5, 3, 5)],
                [(1, 5, 1, 5), (3, 1, 3, 1)],
            ],
            3: [
                [(1, 1, 1, 1), (2, 3, 2, 3), (3, 5, 3, 5)],
                [(1, 5, 1, 5), (2, 3, 2, 3), (3, 1, 3, 1)],
            ],
        }
    else:
        sprite = np.ones((4, 7), dtype=np.uint8)
        choices = {
            1: [
                [(1, 2, 1, 3)],
                [(2, 3, 2, 3)],
            ],
            2: [
                [(1, 1, 1, 1), (2, 4, 2, 4)],
                [(1, 5, 1, 5), (2, 2, 2, 2)],
            ],
            3: [
                [(1, 1, 1, 1), (1, 3, 1, 3), (1, 5, 1, 5)],
                [(2, 1, 2, 1), (2, 3, 2, 3), (2, 5, 2, 5)],
            ],
        }
    for r0, c0, r1, c1 in choices[hole_count][int(rng.integers(len(choices[hole_count])))]:
        sprite[r0 : r1 + 1, c0 : c1 + 1] = 0
    assert _count_holes(sprite.astype(bool)) == hole_count
    return sprite


def _tabbed_sprite(hole_count: int, rng: np.random.Generator) -> np.ndarray:
    sprite = np.ones((5, 8), dtype=np.uint8)
    if int(rng.integers(2)) == 0:
        sprite[0, :2] = 0
        sprite[-1, -2:] = 0
    else:
        sprite[0, -2:] = 0
        sprite[-1, :2] = 0

    choices = {
        1: [
            [(2, 3, 2, 4)],
            [(1, 4, 3, 4)],
        ],
        2: [
            [(1, 2, 1, 2), (3, 5, 3, 5)],
            [(1, 5, 1, 5), (3, 2, 3, 2)],
        ],
    }
    for r0, c0, r1, c1 in choices[hole_count][int(rng.integers(len(choices[hole_count])))]:
        sprite[r0 : r1 + 1, c0 : c1 + 1] = 0
    assert _count_holes(sprite.astype(bool)) == hole_count
    return sprite


def _solid_sprite(rng: np.random.Generator) -> np.ndarray:
    variant = int(rng.integers(3))
    if variant == 0:
        sprite = np.ones((5, 5), dtype=np.uint8)
        sprite[0, 0] = 0
        sprite[4, 4] = 0
    elif variant == 1:
        sprite = np.ones((4, 6), dtype=np.uint8)
        sprite[0, -1] = 0
        sprite[-1, 0] = 0
    else:
        sprite = np.ones((5, 6), dtype=np.uint8)
        sprite[0, 0] = 0
        sprite[0, 1] = 0
        sprite[-1, -2] = 0
        sprite[-1, -1] = 0
    assert _count_holes(sprite.astype(bool)) == 0
    return sprite


def _small_sprite(hole_count: int, rng: np.random.Generator) -> np.ndarray:
    sprite = np.ones((5, 5), dtype=np.uint8)
    choices = {
        1: [
            [(2, 2, 2, 2)],
            [(1, 2, 3, 2)],
        ],
        2: [
            [(1, 1, 1, 1), (3, 3, 3, 3)],
            [(1, 3, 1, 3), (3, 1, 3, 1)],
        ],
        3: [
            [(1, 1, 1, 1), (1, 3, 1, 3), (3, 2, 3, 2)],
        ],
    }
    for r0, c0, r1, c1 in choices[hole_count][int(rng.integers(len(choices[hole_count])))]:
        sprite[r0 : r1 + 1, c0 : c1 + 1] = 0
    assert _count_holes(sprite.astype(bool)) == hole_count
    return sprite


def _build_specs(rng: np.random.Generator) -> list[dict]:
    duplicate_k = 1 if float(rng.random()) < 0.5 else 2
    missing_k = 2 if duplicate_k == 1 else 1
    tall_plain = bool(int(rng.integers(2)))
    colors = [int(c) for c in rng.permutation(np.arange(1, 10))]

    specs = [
        {
            "name": "plain_a",
            "family": "plain_shared",
            "role": "same_shape_low",
            "k": duplicate_k,
            "sprite": _plain_shared_sprite(duplicate_k, tall_plain, rng),
            "color": colors[0],
        },
        {
            "name": "plain_b",
            "family": "plain_shared",
            "role": "same_shape_high",
            "k": 3,
            "sprite": _plain_shared_sprite(3, tall_plain, rng),
            "color": colors[1],
        },
        {
            "name": "tabbed",
            "family": "tabbed",
            "role": "same_holes_diff_shape",
            "k": duplicate_k,
            "sprite": _tabbed_sprite(duplicate_k, rng),
            "color": colors[2],
        },
        {
            "name": "solid",
            "family": "solid",
            "role": "zero_holes",
            "k": 0,
            "sprite": _solid_sprite(rng),
            "color": colors[3],
        },
        {
            "name": "small_missing",
            "family": "small",
            "role": "missing_positive_branch",
            "k": missing_k,
            "sprite": _small_sprite(missing_k, rng),
            "color": colors[4],
        },
    ]

    if float(rng.random()) < 0.55:
        extra_kind = int(rng.integers(3))
        if extra_kind == 0:
            extra_k = 0
            extra_sprite = _solid_sprite(rng)
            family = "solid_extra"
        elif extra_kind == 1:
            extra_k = 1
            extra_sprite = _small_sprite(1, rng)
            family = "small_extra"
        else:
            extra_k = 2
            extra_sprite = _small_sprite(2, rng)
            family = "small_extra"
        specs.append(
            {
                "name": "extra",
                "family": family,
                "role": "extra",
                "k": extra_k,
                "sprite": extra_sprite,
                "color": colors[5],
            }
        )

    for spec in specs:
        spec["sprite"] = spec["sprite"].astype(np.uint8)
        spec["foot_h"] = int(spec["sprite"].shape[0] * max(1, spec["k"]))
        spec["foot_w"] = int(spec["sprite"].shape[1] * max(1, spec["k"]))
        assert _count_holes(spec["sprite"].astype(bool)) == spec["k"]
    return specs


def _place_specs(specs: list[dict], rng: np.random.Generator) -> bool:
    placed_rects: list[tuple[int, int, int, int]] = []
    for spec in sorted(specs, key=lambda s: s["foot_h"] * s["foot_w"], reverse=True):
        ok = False
        for _ in range(250):
            top = int(rng.integers(0, 30 - spec["foot_h"] + 1))
            left = int(rng.integers(0, 30 - spec["foot_w"] + 1))
            rect = (top, left, top + spec["foot_h"] - 1, left + spec["foot_w"] - 1)
            if all(_rects_with_gap_disjoint(rect, other, gap=2) for other in placed_rects):
                spec["top"] = top
                spec["left"] = left
                placed_rects.append(rect)
                ok = True
                break
        if not ok:
            return False
    return True


def _crop_canvas(grid: np.ndarray, specs: list[dict], rng: np.random.Generator) -> tuple[np.ndarray, list[dict]]:
    row0 = min(int(spec["top"]) for spec in specs)
    col0 = min(int(spec["left"]) for spec in specs)
    row1 = max(int(spec["top"]) + int(spec["foot_h"]) - 1 for spec in specs)
    col1 = max(int(spec["left"]) + int(spec["foot_w"]) - 1 for spec in specs)

    top_margin = int(rng.integers(0, min(row0, 3) + 1))
    left_margin = int(rng.integers(0, min(col0, 3) + 1))
    bottom_margin = int(rng.integers(0, min(29 - row1, 3) + 1))
    right_margin = int(rng.integers(0, min(29 - col1, 3) + 1))

    r0 = row0 - top_margin
    c0 = col0 - left_margin
    r1 = row1 + bottom_margin
    c1 = col1 + right_margin

    cropped = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    shifted_specs = []
    for spec in specs:
        shifted = dict(spec)
        shifted["top"] = int(spec["top"]) - r0
        shifted["left"] = int(spec["left"]) - c0
        shifted_specs.append(shifted)
    return cropped, shifted_specs


def _expected_output_from_specs(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for spec in specs:
        if spec["k"] == 0:
            continue
        scaled = _scale_mask(spec["sprite"].astype(bool), spec["k"])
        top = int(spec["top"])
        left = int(spec["left"])
        region = out[top : top + scaled.shape[0], left : left + scaled.shape[1]]
        assert not np.any(region[scaled] != BG)
        region[scaled] = int(spec["color"])
    return out


def _validate_instance(grid: np.ndarray, specs: list[dict]) -> None:
    assert grid.ndim == 2
    assert max(grid.shape) <= 30
    assert np.any(grid != BG)

    components = _find_components(grid, BG)
    by_color = {int(comp["color"]): comp for comp in components}
    assert len(by_color) == len(specs)

    hole_hist = {0: 0, 1: 0, 2: 0, 3: 0}
    for spec in specs:
        comp = by_color[int(spec["color"])]
        r0, c0, r1, c1 = comp["bbox"]
        local_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        coords = comp["coords"]
        local_mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
        actual_holes = _count_holes(local_mask)
        assert actual_holes == spec["k"]
        if actual_holes in hole_hist:
            hole_hist[actual_holes] += 1

    assert hole_hist[0] >= 1 and hole_hist[1] >= 1 and hole_hist[2] >= 1 and hole_hist[3] >= 1

    plain_specs = [spec for spec in specs if spec["family"] == "plain_shared"]
    assert len(plain_specs) == 2
    assert plain_specs[0]["sprite"].shape == plain_specs[1]["sprite"].shape
    assert plain_specs[0]["k"] != plain_specs[1]["k"]

    same_k = [spec for spec in specs if spec["role"] == "same_holes_diff_shape"][0]["k"]
    same_k_shapes = [spec["sprite"].shape for spec in specs if spec["k"] == same_k]
    assert len(same_k_shapes) >= 2
    assert len({tuple(shape) for shape in same_k_shapes}) >= 2

    expected = _expected_output_from_specs(grid.shape, specs)
    produced = main(grid)
    assert np.array_equal(produced, expected)
    assert not np.array_equal(produced, grid)


def generate_input() -> np.ndarray:
    for seed in range(800):
        rng = np.random.default_rng(seed=None)
        specs = _build_specs(rng)
        if not _place_specs(specs, rng):
            continue

        canvas = np.full((30, 30), BG, dtype=int)
        for spec in specs:
            _paint_binary_sprite(canvas, spec["sprite"], int(spec["top"]), int(spec["left"]), int(spec["color"]))

        cropped, shifted_specs = _crop_canvas(canvas, specs, rng)
        try:
            _validate_instance(cropped, shifted_specs)
        except AssertionError:
            continue
        return cropped
    raise RuntimeError("failed to generate a valid puzzle instance")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Scale each closed object by its own enclosed-hole count and erase solid objects.

    Input structure:
    - The background is black.
    - The foreground contains several separated monochrome 4-connected objects.
    - Different objects may share an outer silhouette while enclosing different numbers of background holes, and solid objects with no holes are also present.

    Rule:
    - For each foreground object, count the number of 4-connected background regions inside that object's bounding box that cannot reach the bounding-box border without crossing the object.
    - If an object has k = 0 holes, erase it completely to background.
    - If an object has k >= 1 holes, use the object's bounding box as its local coordinate system, scale the whole object shape by factor k with the bounding-box top-left corner fixed, and redraw the scaled object in its original color.

    Key insight:
    - The scaling factor is not a separate marker: each object carries its own multiplier through the number of holes it encloses.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    components = _find_components(g, bg)
    out = np.full(g.shape, bg, dtype=int)

    for comp in components:
        color = int(comp["color"])
        r0, c0, r1, c1 = comp["bbox"]
        local_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        coords = comp["coords"]
        local_mask[coords[:, 0] - r0, coords[:, 1] - c0] = True

        k = _count_holes(local_mask)
        if k == 0:
            continue

        scaled = _scale_mask(local_mask, k)
        h, w = scaled.shape
        assert r0 + h <= g.shape[0] and c0 + w <= g.shape[1], "scaled object must stay in bounds"
        region = out[r0 : r0 + h, c0 : c0 + w]
        assert not np.any(region[scaled] != bg), "scaled objects must not overlap"
        region[scaled] = color

    return out
