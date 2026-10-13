import random
from collections import defaultdict

import numpy as np


BG = 0
GEN_ATTEMPTS = 400
GEN_GRID_MIN = 20
GEN_GRID_MAX = 30
GEN_FRAME_COUNT_CHOICES = (2, 3, 4)
GEN_TEMPLATE_SIZE_RANGE = (3, 5)
GEN_INNER_BOUNDS = {
    2: (4, 9),
    3: (4, 8),
    4: (4, 7),
}
GEN_PADDING = 1


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == color
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _ring_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _find_frames(grid: np.ndarray) -> list[dict]:
    frames = []
    for color in sorted(int(c) for c in np.unique(grid) if int(c) != BG):
        for comp in _components_of_color(grid, color):
            r0, c0, r1, c1 = _bbox(comp)
            height = r1 - r0 + 1
            width = c1 - c0 + 1
            if height < 3 or width < 3:
                continue
            ring = _ring_mask(height, width)
            if int(comp.shape[0]) != int(ring.sum()):
                continue
            sub = grid[r0 : r1 + 1, c0 : c1 + 1]
            if not np.all(sub[ring] == color):
                continue
            if np.any(sub[~ring] == color):
                continue
            inner = sub[1:-1, 1:-1]
            seed_cells = np.argwhere(inner != BG)
            if seed_cells.shape[0] != 1:
                continue
            sr, sc = map(int, seed_cells[0])
            seed_color = int(inner[sr, sc])
            if seed_color == color:
                continue
            frames.append(
                {
                    "border_color": color,
                    "bbox": (r0, c0, r1, c1),
                    "seed_pos": (r0 + 1 + sr, c0 + 1 + sc),
                    "seed_color": seed_color,
                    "inner_shape": (height - 2, width - 2),
                }
            )
    frames.sort(key=lambda frame: frame["bbox"])
    return frames


def _extract_templates(grid: np.ndarray, frames: list[dict]) -> dict[int, np.ndarray]:
    seed_positions = defaultdict(list)
    for frame in frames:
        seed_positions[int(frame["seed_color"])].append(tuple(map(int, frame["seed_pos"])))

    templates: dict[int, np.ndarray] = {}
    for color, positions in seed_positions.items():
        mask = grid == color
        for r, c in positions:
            mask[r, c] = False
        coords = np.argwhere(mask)
        assert coords.size > 0, f"missing template for seed color {color}"
        r0, c0, r1, c1 = _bbox(coords)
        pattern = grid[r0 : r1 + 1, c0 : c1 + 1] == color
        templates[color] = pattern
    return templates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each framed region by tiling the binary template keyed by its interior seed color.

    Input structure:
    - The background is black.
    - Several disjoint one-pixel-wide rectangular borders of one color enclose hollow interiors.
    - Each border interior contains exactly one non-black seed pixel.
    - Elsewhere, each seed color also appears as a small monochrome template patch on the black background.

    Rule:
    - Detect every hollow rectangular border and read the unique seed color inside it.
    - For each seed color, extract the matching template's binary mask from the same-colored patch outside the frames.
    - Tile that mask from the interior's top-left corner across the whole framed interior, cropping partial tiles at the bottom or right edge.
    - Paint mask cells with the seed color, keep mask gaps black, and leave both the borders and template patches unchanged.

    Key insight: The seed color is a lookup key for a separate binary template; the interior is not filled uniformly but by a repeated copy of that template mask.
    """
    g = np.asarray(grid, dtype=int)
    frames = _find_frames(g)
    assert frames, "expected at least one framed region"
    templates = _extract_templates(g, frames)

    out = g.copy()
    for frame in frames:
        r0, c0, r1, c1 = map(int, frame["bbox"])
        seed_color = int(frame["seed_color"])
        pattern = templates[seed_color]
        ph, pw = pattern.shape
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        tiled = np.tile(pattern, ((inner_h + ph - 1) // ph, (inner_w + pw - 1) // pw))[:inner_h, :inner_w]
        out[r0 + 1 : r1, c0 + 1 : c1] = np.where(tiled, seed_color, BG)
    return out


def _random_template_mask(rng: random.Random, height: int, width: int) -> np.ndarray:
    ring = _ring_mask(height, width)
    for _ in range(100):
        density = rng.uniform(0.3, 0.7)
        mask = np.array([[rng.random() < density for _ in range(width)] for _ in range(height)], dtype=bool)
        mask[0, rng.randrange(width)] = True
        mask[-1, rng.randrange(width)] = True
        mask[rng.randrange(height), 0] = True
        mask[rng.randrange(height), -1] = True
        if not mask.any() or mask.all():
            continue
        if np.array_equal(mask, ring):
            continue
        return mask
    raise RuntimeError("failed to build non-degenerate template mask")


def _can_place(occupied: np.ndarray, top: int, left: int, height: int, width: int, pad: int) -> bool:
    h, w = occupied.shape
    if top < 0 or left < 0 or top + height > h or left + width > w:
        return False
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    return not occupied[r0:r1, c0:c1].any()


def _occupy(occupied: np.ndarray, top: int, left: int, height: int, width: int):
    occupied[top : top + height, left : left + width] = True


def _place_rectangles(
    rng: random.Random, shape: tuple[int, int], rects: list[dict], pad: int
) -> list[tuple[int, int]] | None:
    occupied = np.zeros(shape, dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(rects)
    order = sorted(range(len(rects)), key=lambda idx: rects[idx]["height"] * rects[idx]["width"], reverse=True)
    for idx in order:
        rect = rects[idx]
        height = int(rect["height"])
        width = int(rect["width"])
        for _ in range(300):
            top = rng.randint(0, shape[0] - height)
            left = rng.randint(0, shape[1] - width)
            if _can_place(occupied, top, left, height, width, pad):
                _occupy(occupied, top, left, height, width)
                placements[idx] = (top, left)
                break
        else:
            return None
    return [placement for placement in placements if placement is not None]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    frame_count = rng.choice(GEN_FRAME_COUNT_CHOICES)
    grid_h = rng.randint(max(GEN_GRID_MIN, 16 + frame_count * 2), GEN_GRID_MAX)
    grid_w = rng.randint(max(GEN_GRID_MIN, 16 + frame_count * 2), GEN_GRID_MAX)
    inner_lo, inner_hi = GEN_INNER_BOUNDS[frame_count]

    border_color = rng.choice(list(range(1, 10)))
    seed_colors = rng.sample([c for c in range(1, 10) if c != border_color], frame_count)

    frame_specs = []
    template_specs = []
    for seed_color in seed_colors:
        inner_h = rng.randint(inner_lo, inner_hi)
        inner_w = rng.randint(inner_lo, inner_hi)
        frame_h = inner_h + 2
        frame_w = inner_w + 2
        template_h = rng.randint(*GEN_TEMPLATE_SIZE_RANGE)
        template_w = rng.randint(*GEN_TEMPLATE_SIZE_RANGE)
        template_mask = _random_template_mask(rng, template_h, template_w)
        seed_r = rng.randrange(inner_h)
        seed_c = rng.randrange(inner_w)
        frame_specs.append(
            {
                "kind": "frame",
                "height": frame_h,
                "width": frame_w,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "seed_color": seed_color,
                "seed_offset": (seed_r, seed_c),
            }
        )
        template_specs.append(
            {
                "kind": "template",
                "height": template_h,
                "width": template_w,
                "seed_color": seed_color,
                "mask": template_mask,
            }
        )

    specs = frame_specs + template_specs
    placements = _place_rectangles(rng, (grid_h, grid_w), specs, GEN_PADDING)
    if placements is None:
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for spec, (top, left) in zip(specs, placements):
        if spec["kind"] == "frame":
            height = int(spec["height"])
            width = int(spec["width"])
            grid[top, left : left + width] = border_color
            grid[top + height - 1, left : left + width] = border_color
            grid[top : top + height, left] = border_color
            grid[top : top + height, left + width - 1] = border_color
            seed_r, seed_c = map(int, spec["seed_offset"])
            grid[top + 1 + seed_r, left + 1 + seed_c] = int(spec["seed_color"])
        else:
            mask = spec["mask"]
            color = int(spec["seed_color"])
            h, w = mask.shape
            patch = grid[top : top + h, left : left + w]
            patch[mask] = color

    frames = _find_frames(grid)
    if len(frames) != frame_count:
        return None
    if {int(frame["seed_color"]) for frame in frames} != set(seed_colors):
        return None
    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
