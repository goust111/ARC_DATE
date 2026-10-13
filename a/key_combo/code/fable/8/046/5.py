import random
from collections import Counter

import numpy as np


MAX_GENERATE_ATTEMPTS = 400
FRAME_COUNT_CHOICES = (3, 4, 5)
GRID_BOUNDS_BY_COUNT = {
    3: ((18, 30), (18, 30)),
    4: ((21, 30), (21, 30)),
    5: ((24, 30), (24, 30)),
}
FRAME_SIZE_BY_COUNT = {
    3: ((6, 11), (6, 11)),
    4: ((6, 9), (6, 9)),
    5: ((5, 8), (5, 8)),
}
FRAME_GAP = 2
COLOR_POOL = tuple(range(1, 10))


def _neighbors4(h: int, w: int, r: int, c: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(h, w, cr, cc):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _perimeter_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _perimeter_coords(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    cells = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _cycle_is_single(owner_to_sig: dict[int, int]) -> bool:
    owners = set(owner_to_sig)
    if owners != set(owner_to_sig.values()):
        return False
    start = next(iter(owners))
    seen = set()
    cur = start
    for _ in range(len(owners)):
        if cur in seen:
            return False
        seen.add(cur)
        cur = owner_to_sig[cur]
    return cur == start and seen == owners


def _extract_frames(grid: np.ndarray) -> tuple[list[dict], list[np.ndarray]]:
    comps = _components(grid != 0)
    frames: list[dict] = []
    noise: list[np.ndarray] = []
    for coords in comps:
        if len(coords) == 1:
            noise.append(coords)
            continue

        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            raise AssertionError("non-noise component is too small to be a frame")

        local = np.zeros((h, w), dtype=bool)
        local[coords[:, 0] - r0, coords[:, 1] - c0] = True
        if not np.array_equal(local, _perimeter_mask(h, w)):
            raise AssertionError("non-noise component is not a full rectangular frame")

        border_colors = [int(grid[r, c]) for r, c in coords]
        counts = Counter(border_colors).most_common()
        if len(counts) < 2 or counts[0][1] <= counts[1][1]:
            raise AssertionError("frame must have a strict border-color majority and one foreign color")
        owner = int(counts[0][0])
        foreign = [color for color, _count in counts if int(color) != owner]
        if len(foreign) != 1:
            raise AssertionError("each frame must contain exactly one foreign border color")

        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "owner": owner,
                "signature": int(foreign[0]),
                "coords": coords,
            }
        )
    return frames, noise


def _build_output_from_frames(grid: np.ndarray, frames: list[dict]) -> np.ndarray:
    owners = [frame["owner"] for frame in frames]
    owner_to_frame = {frame["owner"]: frame for frame in frames}
    if len(owner_to_frame) != len(frames):
        raise AssertionError("frame border colors must be distinct")

    owner_to_sig = {frame["owner"]: frame["signature"] for frame in frames}
    if not _cycle_is_single(owner_to_sig):
        raise AssertionError("corruption relation must be one directed cycle across all frames")

    out = np.zeros_like(grid)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        out[r0, c0 : c1 + 1] = frame["owner"]
        out[r1, c0 : c1 + 1] = frame["owner"]
        out[r0 : r1 + 1, c0] = frame["owner"]
        out[r0 : r1 + 1, c1] = frame["owner"]

    for donor in frames:
        recipient = owner_to_frame[donor["signature"]]
        r0, c0, r1, c1 = recipient["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = donor["owner"]

    if len(set(int(out[r0 + 1, c0 + 1]) for r0, c0, r1, c1 in [f["bbox"] for f in frames])) < 2:
        raise AssertionError("filled interiors should not collapse to one color")
    return out


def _wrong_forward_fill(grid: np.ndarray, frames: list[dict]) -> np.ndarray:
    out = np.zeros_like(grid)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        color = frame["owner"]
        out[r0, c0 : c1 + 1] = color
        out[r1, c0 : c1 + 1] = color
        out[r0 : r1 + 1, c0] = color
        out[r0 : r1 + 1, c1] = color
        out[r0 + 1 : r1, c0 + 1 : c1] = frame["signature"]
    return out


def _wrong_self_fill(grid: np.ndarray, frames: list[dict]) -> np.ndarray:
    out = np.zeros_like(grid)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        color = frame["owner"]
        out[r0, c0 : c1 + 1] = color
        out[r1, c0 : c1 + 1] = color
        out[r0 : r1 + 1, c0] = color
        out[r0 : r1 + 1, c1] = color
        out[r0 + 1 : r1, c0 + 1 : c1] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair each corrupted rectangular wall, then send every frame color into the interior of the frame whose signature it appears on.

    Input structure:
    - Black background contains at least three one-pixel-thick monochrome rectangular frames.
    - Each frame border is mostly one color, but a few border cells are replaced by exactly one other frame's border color.
    - Additional same-color singleton noise pixels may appear away from all frame borders, including inside frame interiors.

    Rule:
    - Treat every multi-pixel non-black 4-connected component whose occupied cells equal a full rectangular perimeter as a frame.
    - Restore each frame border to its majority border color, and read the unique minority border color on that wall as the foreign signature.
    - If frame A's wall contains frame B's border color, fill frame B's interior with frame A's restored border color.
    - Remove all singleton off-wall noise pixels.

    Key insight:
    - Border position determines meaning: minority-colored pixels on a rectangular wall encode a directed cycle of ownership transfer, while identical colors off the wall are just noise and must be erased.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frames, _noise = _extract_frames(g)
    assert len(frames) >= 3, "expected at least three frames"
    return _build_output_from_frames(g, frames)


def _separated(box: tuple[int, int, int, int], other: tuple[int, int, int, int], gap: int) -> bool:
    r0, c0, r1, c1 = box
    or0, oc0, or1, oc1 = other
    return (
        r1 + gap < or0
        or or1 + gap < r0
        or c1 + gap < oc0
        or oc1 + gap < c0
    )


def _place_boxes(h: int, w: int, sizes: list[tuple[int, int]]) -> list[tuple[int, int, int, int]] | None:
    placed: list[tuple[int, int, int, int]] = []
    for fh, fw in sorted(sizes, reverse=True):
        for _ in range(120):
            r0 = random.randint(0, h - fh)
            c0 = random.randint(0, w - fw)
            box = (r0, c0, r0 + fh - 1, c0 + fw - 1)
            if all(_separated(box, other, FRAME_GAP) for other in placed):
                placed.append(box)
                break
        else:
            return None
    random.shuffle(placed)
    return placed


def _choose_corruption_cells(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    side_cells = {
        "top": [(r0, c) for c in range(c0, c1 + 1)],
        "bottom": [(r1, c) for c in range(c0, c1 + 1)],
        "left": [(r, c0) for r in range(r0 + 1, r1)],
        "right": [(r, c1) for r in range(r0 + 1, r1)],
    }
    side_count = random.randint(1, 3)
    chosen_sides = random.sample(list(side_cells), side_count)
    cells: set[tuple[int, int]] = set()
    for side in chosen_sides:
        candidates = side_cells[side]
        take = random.randint(1, min(2, len(candidates)))
        cells.update(random.sample(candidates, take))
    return list(cells)


def _point_in_frame_interior(r: int, c: int, box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    return r0 < r < r1 and c0 < c < c1


def _can_place_noise(grid: np.ndarray, r: int, c: int) -> bool:
    if grid[r, c] != 0:
        return False
    h, w = grid.shape
    for nr, nc in _neighbors4(h, w, r, c):
        if grid[nr, nc] != 0:
            return False
    return True


def _add_noise(grid: np.ndarray, frames: list[dict], colors: list[int]) -> int:
    h, w = grid.shape
    frame_boxes = [frame["bbox"] for frame in frames]
    inside_candidates = []
    outside_candidates = []
    for r in range(h):
        for c in range(w):
            if not _can_place_noise(grid, r, c):
                continue
            if any(_point_in_frame_interior(r, c, box) for box in frame_boxes):
                inside_candidates.append((r, c))
            else:
                outside_candidates.append((r, c))

    placed = 0
    random.shuffle(inside_candidates)
    random.shuffle(outside_candidates)

    inside_target = 0
    if inside_candidates and random.random() < 0.85:
        inside_target = random.randint(1, min(4, len(inside_candidates)))
    outside_target = random.randint(1, min(6, max(1, len(outside_candidates))))

    for target_pool, target in ((inside_candidates, inside_target), (outside_candidates, outside_target)):
        idx = 0
        while placed < inside_target + outside_target and target > 0 and idx < len(target_pool):
            r, c = target_pool[idx]
            idx += 1
            if not _can_place_noise(grid, r, c):
                continue
            grid[r, c] = random.choice(colors)
            placed += 1
            target -= 1
    return placed


def _frame_specs_to_input(
    h: int,
    w: int,
    frame_specs: list[dict],
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for spec in frame_specs:
        owner = spec["owner"]
        foreign = spec["signature"]
        box = spec["bbox"]
        for r, c in _perimeter_coords(box):
            grid[r, c] = owner
        for r, c in spec["corruption"]:
            grid[r, c] = foreign
    return grid


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.choice(FRAME_COUNT_CHOICES)
    (hr, wr) = GRID_BOUNDS_BY_COUNT[frame_count]
    h = random.randint(*hr)
    w = random.randint(*wr)
    (fh_range, fw_range) = FRAME_SIZE_BY_COUNT[frame_count]

    sizes = [
        (
            random.randint(*fh_range),
            random.randint(*fw_range),
        )
        for _ in range(frame_count)
    ]
    boxes = _place_boxes(h, w, sizes)
    if boxes is None:
        return None

    owners = random.sample(COLOR_POOL, frame_count)
    cycle = owners[:]
    random.shuffle(cycle)
    signature_of = {cycle[i]: cycle[(i + 1) % frame_count] for i in range(frame_count)}

    frame_specs = []
    for owner, box in zip(owners, boxes):
        r0, c0, r1, c1 = box
        if r1 - r0 < 4 or c1 - c0 < 4:
            return None
        corruption = _choose_corruption_cells(box)
        if not corruption:
            return None
        frame_specs.append(
            {
                "bbox": box,
                "owner": owner,
                "signature": signature_of[owner],
                "corruption": corruption,
            }
        )

    grid = _frame_specs_to_input(h, w, frame_specs)
    if _add_noise(grid, frame_specs, owners) == 0:
        return None

    try:
        frames, noise = _extract_frames(grid)
        if len(frames) != frame_count or len(noise) == 0:
            return None
        actual = main(grid)
    except AssertionError:
        return None

    expected = _build_output_from_frames(grid, frames)
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(expected, _wrong_forward_fill(grid, frames)):
        return None
    if np.array_equal(expected, _wrong_self_fill(grid, frames)):
        return None

    owner_to_sig = {frame["owner"]: frame["signature"] for frame in frames}
    if not _cycle_is_single(owner_to_sig):
        return None
    if any(len(noise_comp) != 1 for noise_comp in noise):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid wall-signature puzzle")
